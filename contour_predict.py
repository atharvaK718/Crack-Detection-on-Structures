"""Batch crack-contour prediction for images in ``input_images/``."""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from tensorflow.keras.models import Model

from inference import DEFAULT_THRESHOLD, load_model, predict_mask
from preprocessing import IMG_HEIGHT, IMG_WIDTH, load_image

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "input_images"
OUTPUT_DIR = PROJECT_ROOT / "output_images"
MODELS_DIR = PROJECT_ROOT / "models"
BATCH_THRESHOLD = DEFAULT_THRESHOLD


def ensure_directories() -> None:
    """Create the documented batch-processing directories when absent."""
    for directory in (INPUT_DIR, OUTPUT_DIR, MODELS_DIR):
        os.makedirs(directory, exist_ok=True)


def refine_mask(mask: np.ndarray, border_size: int = 20, min_contour_area: int = 100) -> np.ndarray:
    """Remove border noise and small isolated contours from a binary mask."""
    refined_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    refined_mask[:border_size, :] = 0
    refined_mask[-border_size:, :] = 0
    refined_mask[:, :border_size] = 0
    refined_mask[:, -border_size:] = 0

    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:
            cv2.drawContours(refined_mask, [contour], -1, 0, -1)
    return refined_mask


def detect_and_draw_contours(
    image: Union[str, Path], model: Optional[Model] = None, threshold: float = BATCH_THRESHOLD
) -> Tuple[np.ndarray, np.ndarray]:
    """Return an annotated BGR image and its refined 256x256 binary mask."""
    original_image = load_image(image)
    prediction_model = model if model is not None else load_model()
    refined_mask = refine_mask(predict_mask(original_image, prediction_model, threshold))
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = original_image.shape[:2]
    scale = np.array([width / IMG_WIDTH, height / IMG_HEIGHT])
    annotated_image = original_image.copy()
    for contour in contours:
        scaled_contour = (contour * scale).astype(np.int32)
        cv2.drawContours(annotated_image, [scaled_contour], -1, (0, 255, 0), 2)
    return annotated_image, refined_mask


def main() -> None:
    ensure_directories()
    image_paths = [path for path in INPUT_DIR.iterdir() if path.is_file()]
    if not image_paths:
        print(f"No input images found in {INPUT_DIR}. Add an image and run this script again.")
        return

    try:
        model = load_model()
    except RuntimeError as error:
        print(f"Unable to start batch prediction: {error}")
        return

    for image_path in image_paths:
        try:
            annotated_image, _ = detect_and_draw_contours(image_path, model)
            output_path = OUTPUT_DIR / image_path.name
            if not cv2.imwrite(str(output_path), annotated_image):
                print(f"Could not write result: {output_path}")
            else:
                print(f"Saved result: {output_path}")
        except (RuntimeError, ValueError) as error:
            print(f"Skipping {image_path.name}: {error}")


if __name__ == "__main__":
    main()
