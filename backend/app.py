"""Stateless HTTP API for one-at-a-time crack-image analysis."""

from __future__ import annotations

import base64
import io
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image as PILImage, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from contour_predict import refine_mask  # noqa: E402
from inference import InferenceError, ModelLoadError, load_model, predict_mask  # noqa: E402
from preprocessing import IMG_HEIGHT, IMG_WIDTH  # noqa: E402

app = FastAPI(title="Crack Detection API", version="1.0.0")
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_model = None


def get_model():
    """Load once per process; requests remain stateless."""
    global _model
    if _model is None:
        _model = load_model()
    return _model


def analyze_image(image: np.ndarray) -> dict:
    """Run U-Net inference, contour extraction, and annotate the original BGR image."""
    started_at = time.perf_counter()
    mask = refine_mask(predict_mask(image, get_model()))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = image.shape[:2]
    scale = np.array([width / IMG_WIDTH, height / IMG_HEIGHT])
    annotated = image.copy()
    for contour in contours:
        scaled_contour = (contour * scale).astype(np.int32)
        cv2.drawContours(annotated, [scaled_contour], -1, (0, 255, 0), 2)

    encoded, png = cv2.imencode(".png", annotated)
    if not encoded:
        raise RuntimeError("Could not encode the annotated PNG.")

    crack_pixels = int(np.count_nonzero(mask))
    return {
        "annotated_image": "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii"),
        "crack_detected": bool(crack_pixels),
        "crack_area_percent": round(crack_pixels / mask.size * 100, 3),
        "crack_regions": len(contours),
        "inference_time_ms": round((time.perf_counter() - started_at) * 1000, 1),
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _decode_image_with_exif(raw: bytes) -> np.ndarray:
    """Decode image bytes with EXIF orientation normalization, returning BGR numpy array."""
    pil_image = PILImage.open(io.BytesIO(raw))
    pil_image = ImageOps.exif_transpose(pil_image)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    rgb_array = np.array(pil_image, dtype=np.uint8)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return bgr_array


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
    """Analyze one uploaded image and return its annotation plus summary metrics."""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file (PNG, JPEG, or WebP).")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    try:
        image = _decode_image_with_exif(raw)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"The uploaded file could not be decoded as an image: {error}") from error
    try:
        return analyze_image(image)
    except (ModelLoadError, InferenceError, RuntimeError) as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
