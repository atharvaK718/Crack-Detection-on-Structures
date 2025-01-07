import cv2
import numpy as np
from keras.models import load_model
#from tensorflow.keras.models import load_model

# Load your trained U-Net model
model = load_model("V:/VIT/Project I/Codes/model.h5")  # Update with the path to your trained model

# Function to preprocess the image
def preprocess_image(frame):
    img = cv2.resize(frame, (256, 256))  # Resize to model input size
    img = img / 255.0  # Normalize the image
    return img

# Function to refine the mask by removing noise near edges and filtering out small contours
def refine_mask(mask, border_size=20, min_contour_area=100):
    kernel = np.ones((5, 5), np.uint8)
    refined_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Clear edges to remove false detections at the borders
    refined_mask[:border_size, :] = 0
    refined_mask[-border_size:, :] = 0
    refined_mask[:, :border_size] = 0
    refined_mask[:, -border_size:] = 0

    # Remove small contours that are likely noise
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:
            cv2.drawContours(refined_mask, [contour], -1, 0, -1)

    return refined_mask

# Function to detect and draw contours of cracks on each frame
def detect_and_draw_contours(frame):
    img = preprocess_image(frame)
    input_img = np.expand_dims(img, axis=0)

    # Make prediction using the trained model
    pred_mask = model.predict(input_img)[0, :, :, 0]
    # pred_mask_binary = (pred_mask > 0.70).astype(np.uint8)
    pred_mask_binary = (pred_mask > 0.35).astype(np.uint8)

    # Refine the predicted mask
    pred_mask_refined = refine_mask(pred_mask_binary)

    # Find contours from the refined mask
    contours, _ = cv2.findContours(pred_mask_refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Scale contours back to the original frame size
    h, w, _ = frame.shape
    scale_x = w / 256
    scale_y = h / 256

    # Draw filtered contours on the frame
    for contour in contours:
        scaled_contour = contour * [scale_x, scale_y]
        scaled_contour = scaled_contour.astype(np.int32)
        cv2.drawContours(frame, [scaled_contour], -1, (0, 255, 0), 2)

    return frame

# Initialize webcam feed
# cap = cv2.VideoCapture(0) #hp camera
#cap = cv2.VideoCapture(2) #ready for camera

cap = cv2.VideoCapture(1) #moto camera usb

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect cracks and draw contours
    frame_with_contours = detect_and_draw_contours(frame)

    # Display the result
    cv2.imshow('Crack Detection', frame_with_contours)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
