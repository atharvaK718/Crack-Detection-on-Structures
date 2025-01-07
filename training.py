import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from sklearn.model_selection import train_test_split

# Function to load images and masks with matching names from different folders
def load_data(image_dir, mask_dir):
    images = []
    masks = []

    for img_name in sorted(os.listdir(image_dir)):
        # Read the image
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, (256, 256))  # Resize to 256x256

            # Construct corresponding mask path using the same name
            mask_path = os.path.join(mask_dir, img_name)

            # Check if the mask file exists and read it
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = cv2.resize(mask, (256, 256))
                    mask = mask / 255.0  # Normalize mask

                    # Only append when both image and mask are valid
                    images.append(img)
                    masks.append(mask)
                else:
                    print(f"Error reading mask: {mask_path}. Skipping this mask.")
            else:
                print(f"Warning: Mask file not found for {img_name} at {mask_path}. Skipping this image.")
        else:
            print(f"Error reading image: {img_path}. Skipping this image.")

    # Ensure that images and masks are paired correctly
    return np.array(images), np.array(masks).reshape(-1, 256, 256, 1)

# Paths to your images and masks
image_path = "C:/Users/VEDANT/Desktop/Project 1/Crack Detection/Dataset/archive/crack_segmentation_dataset/train/images"  # Update with your image directory
mask_path = "C:/Users/VEDANT/Desktop/Project 1/Crack Detection/Dataset/archive/crack_segmentation_dataset/train/masks"    # Update with your mask directory

# Load images and masks
images, masks = load_data(image_path, mask_path)

# Check if images and masks are loaded correctly
print(f"Loaded {len(images)} images and {len(masks)} masks")

# Ensure there are enough samples to split
if len(images) == 0 or len(masks) == 0:
    print("No valid image-mask pairs found. Please check your dataset.")
else:
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

    # Build the U-Net model
    def build_unet_model():
        inputs = Input((256, 256, 3))

        # Encoder
        c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
        p3 = MaxPooling2D((2, 2))(c3)

        # Bottleneck
        c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(p3)

        # Decoder
        u1 = UpSampling2D((2, 2))(c4)
        m1 = concatenate([u1, c3])
        c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(m1)

        u2 = UpSampling2D((2, 2))(c5)
        m2 = concatenate([u2, c2])
        c6 = Conv2D(32, (3, 3), activation='relu', padding='same')(m2)

        u3 = UpSampling2D((2, 2))(c6)
        m3 = concatenate([u3, c1])
        c7 = Conv2D(16, (3, 3), activation='relu', padding='same')(m3)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)

        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model

    # Create and compile the model
    model = build_unet_model()

    # Train the model
    
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32)

    #speed training using only 1 epoch, accuracy is very bad

    #history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=8)


    # Save the trained model
   # model.save('path/to/your/trained_model.h5')  # Update this path to where you want to save the model
    model.save('C:/Users/VEDANT/Desktop/Project 1/test_trained_model.h5')

    print("Model saved successfully.")

    # Function to visualize predictions
    def visualize_predictions(images, masks, model):
        for i in range(3):  # Display three samples
            img = images[i]
            true_mask = masks[i].squeeze()
            pred_mask = model.predict(img[np.newaxis, ...])[0].squeeze() > 0.5

            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.title("Input Image")
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            plt.subplot(1, 3, 2)
            plt.title("True Mask")
            plt.imshow(true_mask, cmap='gray')

            plt.subplot(1, 3, 3)
            plt.title("Predicted Mask")
            plt.imshow(pred_mask, cmap='gray')

            plt.show()

    # Visualize some predictions
    visualize_predictions(X_val, y_val, model)
