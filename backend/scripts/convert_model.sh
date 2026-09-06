#!/usr/bin/env sh
# Convert the repository model for browser inference. Run from repository root.
set -eu
mkdir -p web/public/tfjs
tensorflowjs_converter \
  --input_format=keras \
  --quantization_bytes=2 \
  model.h5 web/public/tfjs
