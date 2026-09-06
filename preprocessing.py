"""Shared image preparation for U-Net training and inference.

OpenCV decodes images in BGR order.  This module intentionally retains that
order because it is the format used when the saved model was called by the
original scripts.
"""

from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image as PILImage, ImageOps

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
NORMALIZATION_FACTOR = 255.0
MODEL_INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

ImageInput = Union[str, Path, np.ndarray]


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """Decode an image in BGR order with EXIF orientation normalization."""
    path = Path(image_path)
    pil_image = PILImage.open(path)
    pil_image = ImageOps.exif_transpose(pil_image)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    rgb_array = np.array(pil_image, dtype=np.uint8)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    if bgr_array is None or bgr_array.size == 0:
        raise ValueError(f"Could not decode input image: {path}")
    return bgr_array


def _coerce_bgr_image(image: ImageInput) -> np.ndarray:
    if isinstance(image, (str, Path)):
        return load_image(image)
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Input image must be a non-empty BGR NumPy array or a decodable image path.")
    if image.ndim != 3 or image.shape[2] != IMG_CHANNELS:
        raise ValueError(
            f"Input image must have shape (height, width, {IMG_CHANNELS}) in BGR order; got {image.shape}."
        )
    return image


def preprocess_image(image: ImageInput) -> np.ndarray:
    """Resize a BGR image to the U-Net input size and normalize it to [0, 1]."""
    bgr_image = _coerce_bgr_image(image)
    resized = cv2.resize(bgr_image, (IMG_WIDTH, IMG_HEIGHT))
    return resized.astype(np.float32) / NORMALIZATION_FACTOR


def preprocess_mask(mask: np.ndarray) -> np.ndarray:
    """Resize a grayscale training mask and normalize it to [0, 1]."""
    if not isinstance(mask, np.ndarray) or mask.size == 0:
        raise ValueError("Training mask must be a non-empty NumPy array.")
    resized = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
    return resized.astype(np.float32) / NORMALIZATION_FACTOR


def prepare_model_input(image: ImageInput) -> np.ndarray:
    """Return one preprocessed image with the batch dimension required by Keras."""
    return np.expand_dims(preprocess_image(image), axis=0)
