"""Live-camera crack detection with safe camera cleanup."""

import cv2
import numpy as np

from contour_predict import refine_mask
from inference import load_model, predict_mask
from preprocessing import IMG_HEIGHT, IMG_WIDTH

CAMERA_INDEX = 1
LIVE_THRESHOLD = 0.35


def detect_and_draw_contours(frame: np.ndarray, model) -> np.ndarray:
    """Run inference on a frame and draw the refined crack contours in place."""
    refined_mask = refine_mask(predict_mask(frame, model, LIVE_THRESHOLD))
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = frame.shape[:2]
    scale = np.array([width / IMG_WIDTH, height / IMG_HEIGHT])
    for contour in contours:
        scaled_contour = (contour * scale).astype(np.int32)
        cv2.drawContours(frame, [scaled_contour], -1, (0, 255, 0), 2)
    return frame


def main(camera_index: int = CAMERA_INDEX) -> None:
    cap = cv2.VideoCapture(camera_index)
    try:
        if not cap.isOpened():
            print(
                f"Could not open camera device {camera_index}. "
                "Check the device connection or choose a valid camera index."
            )
            return

        try:
            model = load_model()
        except RuntimeError as error:
            print(f"Could not start live detection: {error}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera frame could not be read; stopping live detection.")
                break
            try:
                frame_with_contours = detect_and_draw_contours(frame, model)
            except RuntimeError as error:
                print(f"Inference failed; stopping live detection: {error}")
                break
            cv2.imshow("Crack Detection", frame_with_contours)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
