# Convert the repository model for browser inference. Run from repository root.
New-Item -ItemType Directory -Force -Path web/public/tfjs | Out-Null
tensorflowjs_converter --input_format=keras --quantization_bytes=2 model.h5 web/public/tfjs
