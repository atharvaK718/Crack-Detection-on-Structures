"""Side-effect-free model loading and crack-mask inference helpers."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
from tensorflow.keras.models import Model, load_model as keras_load_model

from preprocessing import ImageInput, prepare_model_input

DEFAULT_MODEL_PATH = Path(__file__).with_name("model.h5")
DEFAULT_THRESHOLD = 0.30


class ModelLoadError(RuntimeError):
    """Raised when the trained model cannot be opened or deserialized."""


class InferenceError(RuntimeError):
    """Raised when model inference cannot produce a valid mask."""


def load_model(model_path: Union[str, Path] = DEFAULT_MODEL_PATH) -> Model:
    """Load the saved U-Net model with an API-friendly error message."""
    path = Path(model_path)
    if not path.is_file():
        raise ModelLoadError(f"Model file was not found: {path}")
    try:
        return keras_load_model(path, compile=False)
    except Exception as error:
        raise ModelLoadError(f"Could not load model from {path}: {error}") from error


def predict_mask(
    image: ImageInput,
    model: Optional[Model] = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> np.ndarray:
    """Return a 256x256 uint8 binary crack mask for an image or image path."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be between 0 and 1; got {threshold}.")

    try:
        model_input = prepare_model_input(image)
    except ValueError as error:
        raise InferenceError(str(error)) from error

    active_model = model if model is not None else load_model()
    try:
        prediction = active_model.predict(model_input, verbose=0)
    except Exception as error:
        raise InferenceError(f"Model prediction failed: {error}") from error

    if prediction.ndim != 4 or prediction.shape[0] != 1 or prediction.shape[-1] != 1:
        raise InferenceError(f"Model returned unexpected prediction shape: {prediction.shape}.")
    probability_mask = prediction[0, :, :, 0]
    if not np.isfinite(probability_mask).all():
        raise InferenceError("Model returned a mask containing NaN or infinite values.")
    return (probability_mask > threshold).astype(np.uint8)
