# Crack Detection on Structures

This project aims to develop an advanced crack detection system by integrating a U-Net deep learning model with real-time camera feeds for continuous infrastructure monitoring. Cracks in structures such as bridges, roads, and buildings pose significant risks, making timely detection crucial for maintenance and safety.

---

## Objectives

1. **Model Training and Fine-Tuning:** Develop a robust and accurate AI model capable of detecting and classifying cracks in structural elements.
2. **Contour Detection for Localization:** Apply contour detection on the predicted crack masks to effectively outline and visualize cracks.
3. **Real-Time Detection:** Implement real-time crack detection using live camera feeds to ensure smooth and accurate detection with low latency.

---

## Methodology

1. **Data Collection and Preprocessing:**
   - Curated datasets of images with crack masks are used for training and validation.
   - Preprocessing steps include normalization, resizing, and data augmentation to improve model generalization.

2. **Model Training:**
   - A U-Net deep learning architecture is employed for semantic segmentation of cracks.
   - The model is fine-tuned to improve accuracy and reduce noise in predictions.

3. **Real-Time Integration:**
   - The trained U-Net model is integrated with a real-time camera feed.
   - Frames from the camera are processed to generate segmentation masks that highlight detected cracks.

---

## Features

- **Deep Learning-Based Detection:** Utilizes a U-Net architecture for precise segmentation and detection of cracks.
- **Real-Time Monitoring:** Supports live camera feeds for continuous structural monitoring.
- **High Accuracy:** Optimized to detect cracks on various structural surfaces with precision.
- **Scalability:** Adaptable for diverse infrastructure types, including bridges, buildings, and pavements.

---

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/atharvaK718/Crack-Detection-on-Structures.git
   ```

2. **Install Dependencies:**
   ```bash
   python -m pip install -r requirements.txt
   ```

3. **Pretrained Model:**
   - `model.h5` is included at the repository root and is loaded by
     `inference.py`. It is currently 2.85 MiB, so Git LFS is not required.
   - If a future model exceeds roughly 50 MiB, track it before committing with
     `git lfs track "*.h5"` and commit the resulting `.gitattributes` file.

4. **Run the Application:**
   ```bash
   python live_detection.py
   ```

---


## Usage

1. **Real-Time Monitoring:**
   - Connect a camera to your system.
   - Start real-time detection:
     ```bash
     python live_detection.py
     ```
   - Detected cracks are highlighted in the live video feed.

2. **Batch Processing:**
   - Place images in the `input_images/` directory.
   - Process images in batch mode:
     ```bash
     python contour_predict.py
     ```
   - Results are saved in the `output_images/` directory with crack contours.

---


## Example Outputs

- Static Image Prediction
  
  ![Static Image Example](https://github.com/atharvaK718/Crack-Detection-on-Structures/raw/main/Static_Images/Static_2.png)  
  [Link to Static Image Analysis Image](https://github.com/atharvaK718/Crack-Detection-on-Structures/raw/main/Static_Images/Static_2.png)

- Real-Time Prefiction
  
  ![Real-Time Monitoring Example](https://github.com/atharvaK718/Crack-Detection-on-Structures/raw/main/Real_Time_Images/Real_Time_2.png)  
  [Link to Real-Time Monitoring Image](https://github.com/atharvaK718/Crack-Detection-on-Structures/raw/main/Real_Time_Images/Real_Time_2.png)
  
---

## Training

1. **Prepare Dataset:**
   - Organize training images and their masks in `data/train/`.
   - Validation data should be placed in `data/val/`.

2. **Train the Model:**
   ```bash
   python training.py
   ```

3. **Evaluate the Model:**
   ```bash
   python contour_predict.py
   ```

---

## System Design

1. **Image Acquisition:** Capture high-resolution images using cameras or drones.
2. **Preprocessing Module:** Normalize and augment images to improve model accuracy.
3. **Crack Detection Model:** A U-Net architecture for detecting and segmenting cracks.
4. **Result Visualization:** Overlay detected cracks on images or video feeds with contours.
5. **User Interface:** A simple UI for viewing results and exporting data.

---

## Shared inference interface

`preprocessing.py` is the single source of truth for U-Net input preparation:
images stay in OpenCV's BGR order, are resized to 256x256, converted to
`float32`, and divided by 255. `inference.py` exposes `load_model()` and
`predict_mask(image, threshold=...)`; neither opens windows or writes files,
so they can be imported directly by a future web API.

`contour_predict.py` creates `input_images/`, `output_images/`, and `models/`
on demand. Put batch images in `input_images/`; annotated results are written
to `output_images/`.

---

## Web application

The production-ready web app lives in `web/` and its stateless FastAPI service
lives in `backend/`. It provides live, browser-side TensorFlow.js scanning and
server-side single-image reports. See [DEPLOY.md](DEPLOY.md) for local setup,
HTTPS camera requirements, TensorFlow.js model conversion, and Vercel/Render
or Railway deployment instructions.

---
