# TensorFlow.js Model Assets

This directory should contain the converted TensorFlow.js model files for live browser inference.

## Conversion Instructions

```bash
# From project root with virtual environment activated
cd Crack-Detection-on-Structures

# Install tensorflowjs (requires numpy<2)
pip install "numpy<2" tensorflowjs

# Convert model with float16 quantization
python -m tensorflowjs.converter \
  --input_format=keras \
  --quantization_bytes=2 \
  model.h5 \
  web/public/tfjs
```

## Expected Files

After conversion, this directory should contain:
- `model.json` - Model topology and weight manifest
- `group1-shard1of1.bin` - Model weights (float16 quantized, ~500KB)

## Fallback Behavior

If these files are not present, the live camera page will:
1. Show a warning in the HUD: "Browser model unavailable — using server fallback"
2. The "Capture & Analyze" button will still work, sending frames to the backend for full analysis

## Model Specifications

- Input: 256×256×3 (RGB, normalized to [0,1])
- Output: 256×256×1 (sigmoid probability map)
- Threshold: 0.3 (configurable in UI)
- Architecture: U-Net (~242K parameters)