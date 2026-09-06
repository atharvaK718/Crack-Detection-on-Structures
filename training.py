"""U-Net training entry point using the shared preprocessing contract."""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model

from preprocessing import IMG_HEIGHT, IMG_WIDTH, preprocess_image, preprocess_mask


def load_data(image_dir: str, mask_dir: str):
    """Load matching image/mask pairs using the same preprocessing as inference."""
    images = []
    masks = []
    for image_name in sorted(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Error reading image: {image_path}. Skipping this image.")
            continue

        mask_path = os.path.join(mask_dir, image_name)
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {image_name} at {mask_path}. Skipping this image.")
            continue
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error reading mask: {mask_path}. Skipping this mask.")
            continue

        images.append(preprocess_image(image))
        masks.append(preprocess_mask(mask))

    return (
        np.asarray(images, dtype=np.float32),
        np.asarray(masks, dtype=np.float32).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1),
    )


def build_unet_model() -> Model:
    """Build the original U-Net architecture without changing its layers."""
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, 3))
    c1 = Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(32, (3, 3), activation="relu", padding="same")(p1)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = Conv2D(64, (3, 3), activation="relu", padding="same")(p2)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = Conv2D(128, (3, 3), activation="relu", padding="same")(p3)
    u1 = UpSampling2D((2, 2))(c4)
    c5 = Conv2D(64, (3, 3), activation="relu", padding="same")(concatenate([u1, c3]))
    u2 = UpSampling2D((2, 2))(c5)
    c6 = Conv2D(32, (3, 3), activation="relu", padding="same")(concatenate([u2, c2]))
    u3 = UpSampling2D((2, 2))(c6)
    c7 = Conv2D(16, (3, 3), activation="relu", padding="same")(concatenate([u3, c1]))
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c7)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def visualize_predictions(images: np.ndarray, masks: np.ndarray, model: Model) -> None:
    for index in range(min(3, len(images))):
        image = images[index]
        true_mask = masks[index].squeeze()
        predicted_mask = model.predict(image[np.newaxis, ...], verbose=0)[0].squeeze() > 0.5
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.subplot(1, 3, 2)
        plt.title("True Mask")
        plt.imshow(true_mask, cmap="gray")
        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(predicted_mask, cmap="gray")
        plt.show()


def main() -> None:
    image_path = "C:/Users/VEDANT/Desktop/Project 1/Crack Detection/Dataset/archive/crack_segmentation_dataset/train/images"
    mask_path = "C:/Users/VEDANT/Desktop/Project 1/Crack Detection/Dataset/archive/crack_segmentation_dataset/train/masks"
    images, masks = load_data(image_path, mask_path)
    print(f"Loaded {len(images)} images and {len(masks)} masks")
    if len(images) == 0 or len(masks) == 0:
        print("No valid image-mask pairs found. Please check your dataset.")
        return

    x_train, x_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)
    model = build_unet_model()
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=32)
    model.save("C:/Users/VEDANT/Desktop/Project 1/test_trained_model.h5")
    print("Model saved successfully.")
    visualize_predictions(x_val, y_val, model)


if __name__ == "__main__":
    main()
