import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load your trained U-Net model
model = load_model('trained_model.h5')  # Update with the path to your trained model

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))  # Resize to model input size
    img = img / 255.0  # Normalize the image
    return img

# Function to refine the mask by removing noise near edges and filtering out small contours
def refine_mask(mask, border_size=20, min_contour_area=100):
    # Define kernel for morphological operations
    kernel = np.ones((5, 5), np.uint8)

    # Remove noise with morphological opening
    refined_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Clear edges to remove false detections at the borders
    refined_mask[:border_size, :] = 0  # Top edge
    refined_mask[-border_size:, :] = 0  # Bottom edge
    refined_mask[:, :border_size] = 0  # Left edge
    refined_mask[:, -border_size:] = 0  # Right edge

    # Remove small contours that are likely noise
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:  # Filter out small contours
            cv2.drawContours(refined_mask, [contour], -1, 0, -1)  # Remove these contours

    return refined_mask

# Function to detect and draw contours of cracks on the original image
def detect_and_draw_contours(image_path):
    # Preprocess the input image
    img = preprocess_image(image_path)
    input_img = np.expand_dims(img, axis=0)  # Expand dimensions for prediction

    # Make prediction using the trained model
    pred_mask = model.predict(input_img)[0, :, :, 0]  # Get the first channel of the predicted mask
    pred_mask_binary = (pred_mask > 0.3).astype(np.uint8)  # Threshold to get binary mask
    
    # Refine the predicted mask to remove false detections
    pred_mask_refined = refine_mask(pred_mask_binary)

    # Find contours from the refined mask
    contours, _ = cv2.findContours(pred_mask_refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Read the original image to draw contours
    original_img = cv2.imread(image_path)

    # Scale contours back to the original image size
    h, w, _ = original_img.shape
    scale_x = w / 256
    scale_y = h / 256

    # Draw filtered contours on the original image
    for contour in contours:
        scaled_contour = contour * [scale_x, scale_y]  # Scale contours to the original size
        scaled_contour = scaled_contour.astype(np.int32)  # Convert to integer for drawing

        # Draw the contour with a green line (color: (0, 255, 0))
        cv2.drawContours(original_img, [scaled_contour], -1, (0, 255, 0), 2)  # Thickness of 2

    # Display the results using Matplotlib
    plt.figure(figsize=(15, 5))

    # Display the original input image with detected contours
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image with Detected Crack Contours')
    plt.axis('off')

    # Display the refined predicted mask
    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask_refined, cmap='gray')
    plt.title('Refined Predicted Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Provide the path of the image you want to detect cracks in

#image_path= 'bridge.jpeg'
#image_path='road_2.png'
image_path="concrete.jpg"

detect_and_draw_contours(image_path)
