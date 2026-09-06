# Changes

## Fixed

- Centralized the U-Net input size, normalization factor, BGR color convention,
  image decoding, mask preparation, and batch shaping in `preprocessing.py`.
- Centralized safe model loading, prediction, output validation, and mask
  thresholding in `inference.py`. Both consumers now use the repository's
  `model.h5` rather than machine-specific or missing model paths.
- Added automatic creation of `input_images/`, `output_images/`, and `models/`
  before batch processing.
- Added clear camera-open errors and `try`/`finally` resource cleanup for the
  live detector.
- Added pinned runtime dependencies and documented installation and execution.

## Issues found

- **Preprocessing inconsistency:** training resized BGR inputs but did not
  divide them by 255, while both inference scripts did. Shared preprocessing
  now normalizes all image inputs consistently. This affects future training;
  the existing model was not retrained or modified.
- **Threshold inconsistency:** batch inference used `0.30` and live inference
  used `0.35`. These values are now named at their call sites and deliberately
  preserved until model validation establishes a single operating threshold.
- **Non-portable paths:** model paths were a missing `trained_model.h5` and an
  absolute `V:` drive path. Both now resolve to the tracked repository model.
- **Training data/output paths remain machine-specific:** `training.py` still
  contains the original absolute dataset and save paths. They cannot be
  selected safely without knowing the intended dataset location and model
  output policy, so they were left unchanged.
- **Camera selection remains hardware-specific:** live detection retains camera
  index `1`; selecting a new default without knowing the deployment hardware
  could connect to the wrong camera.
- `model.h5` is 2.85 MiB, below the approximately 50 MiB Git LFS threshold;
  Git LFS tracking was therefore not added.
