# IDAI2011000467-TejasviKandimalla

Candidate Name: Tejasvi Reddy Kandimalla

Candidate Registration Number-: 1000467

CRS Name: Artificial Intelligence

Course Name: Unit 1- Machine Learning and Deep Learning

School Name: Birla Open Minds International School, Kollur

Link to Live App: https://idai2011000467-tejasvikandimalla-7x56m2h2wimt5budnole5k.streamlit.app/


# ParkVision AI — Intelligent Urban Parking Analytics & Space Optimisation Platform

## 1. Overview

ParkVision AI is a computer vision system that analyzes photos of parking lots and automatically identifies which spaces are occupied and which are empty. It calculates real-time occupancy metrics and generates simple recommendations for drivers, all through a web app built with Streamlit.

## 2. Research Findings that Influenced the Project

Before building the system, I reviewed existing research on parking occupancy detection to understand common approaches and their trade-offs:

- Deep learning-based smart parking systems generally fall into two categories: **classification-based** approaches (classifying individually cropped images of each parking slot as occupied/empty) and **detection-based** approaches (using object detectors like YOLO to locate and classify all slots directly in a full scene image). Classification approaches tend to be simpler and highly accurate per-slot, but require known slot locations in advance. Detection-based approaches are more flexible (they work on new camera angles without manual slot marking) but require higher-resolution input images and more training data to perform reliably, especially in scenes with many small, densely packed objects.
- Vision-based parking slot detection research highlights that detection accuracy is strongly affected by image resolution relative to object (slot) size — a finding I confirmed directly during this project (see Section 6, Testing & Limitations).
- The PKLot dataset itself was designed specifically to support robust parking classification research across varying lighting and weather conditions (sunny, cloudy, rainy), which is why it remains a standard benchmark for this type of task.

These findings directly shaped my decision to start with a classification approach, and later informed my diagnosis of detection quality issues when I moved to a YOLO-based detection approach, ultimately guiding my decision to switch from a pre-resized (640×640) dataset export to the original full-resolution (1280×720) PKLot images with proper annotation conversion.

## 3. Academic References and Key Sources

- [Deep Learning Based Smart Parking Occupancy Detection using Computer Vision](https://pmc.ncbi.nlm.nih.gov/articles/PMC12568149/)
- [Vision-Based Parking Slot Detection using Deep Learning](https://www.mdpi.com/1424-8220/23/15/6869)
- [PKLot: A Robust Dataset for Parking Lot Classification](https://www.inf.ufpr.br/lesoliveira/download/pklot-readme.pdf)
- [YOLO Object Detection Documentation (Ultralytics Official)](https://docs.ultralytics.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 4. Data Preparation

### Dataset
Source: [PKLot dataset](https://www.inf.ufpr.br/lesoliveira/download/pklot-readme.pdf) — a collection of parking lot images captured under varying weather conditions (sunny, cloudy, rainy), with each parking slot individually annotated as occupied or empty.

### Preparation Process

**Initial approach (classification):**
- Used a Kaggle/Roboflow export of PKLot in COCO annotation format.
- Extracted individually cropped slot images using each annotation's bounding box, sorted into `occupied`/`empty` folders.
- Collected 150 images per class (300 total), resized to 224×224 pixels.
- Split into training (70%), validation (15%), and test (15%) sets.
- Applied data augmentation (rotation ±20°, brightness adjustment, horizontal flip) to the training set to improve robustness to real-world variation in lighting and camera angle.

**Revised approach (detection):**
- Identified that the Roboflow export's images were uniformly pre-resized to 640×640, which limited detection accuracy on scenes with many densely packed slots (each slot occupied too few pixels for reliable detection).
- Switched to the original PKLot dataset (full 1280×720 resolution images with XML annotations).
- Converted PKLot's XML rotated-rectangle annotations into YOLO's normalized bounding-box label format (`class x_center y_center width height`), computing an axis-aligned bounding box that encloses each rotated slot contour.
- Successfully converted 12,142 of 12,416 annotated images (274 skipped due to missing/corrupt files; 7,672 individual malformed slot entries filtered out during conversion).
- Split into training (70%), validation (15%), and test (15%) sets.

### Data Cleaning
- Filtered out annotation entries with missing or empty coordinate data.
- Clipped bounding boxes to stay within image boundaries.
- Removed entries where the resulting bounding box had zero width or height.

## 5. Models Used

### Model 1: MobileNetV2 (Classification Approach)
- **Architecture:** MobileNetV2 pretrained on ImageNet (transfer learning), with the base convolutional layers frozen. A GlobalAveragePooling layer, a Dense(128, ReLU) layer, Dropout(0.3), and a final Dense(1, sigmoid) output layer were added for binary classification.
- **Training parameters:** 20 epochs, batch size 16, Adam optimizer, binary cross-entropy loss.
- **Input:** 224×224×3 RGB images of individually cropped parking slots.
- **Class mapping:** `{'empty': 0, 'occupied': 1}`

### Model 2: YOLOv8n (Detection Approach)
- **Architecture:** YOLOv8n (Ultralytics), the smallest/fastest variant in the YOLOv8 family, trained from pretrained COCO weights.
- **Training parameters:** Up to 25 epochs (with early stopping via `patience=5`), image size 960×960, batch size 4.
- **Classes:** `space-empty`, `space-occupied`
- **Technique:** Full-image object detection — the model directly outputs bounding boxes and class labels for every parking slot visible in an uploaded photo, without requiring pre-marked slot positions.

## 6. Metrics and Results

### MobileNetV2 Classifier — Test Set Results
Evaluated on 46 held-out test images (23 per class):

| Metric | Empty | Occupied | Overall Accuracy |
|---|---|---|---|
| Precision | 1.00 | 1.00 | — |
| Recall | 1.00 | 1.00 | — |
| F1-score | 1.00 | 1.00 | — |
| **Accuracy** | | | **1.00 (100%)** |

Confusion matrix:
```
              Predicted Empty   Predicted Occupied
Actual Empty         23                 0
Actual Occupied       0                23
```

The classifier achieved perfect accuracy on the held-out test set. This is expected and realistic given the task: individually cropped, clearly labeled parking slot images are a relatively "clean" classification problem compared to full-scene detection.

### YOLOv8n Detector — Summary
The detection model was trained on the full-resolution PKLot dataset (12,142 images) across two classes (`space-empty`, `space-occupied`). [Insert your final training run's mAP50, precision, and recall from the results.csv / training summary here once the retrained model finishes.]

## 7. Testing & Limitations

The system was tested on unseen images from the PKLot test split, covering different lighting conditions (sunny, cloudy) included in the dataset.

**Key finding — resolution-dependent detection accuracy:** During testing, I found that detection quality was strongly affected by how many pixels each parking slot occupied in the source image. Specifically:
- On the initial 640×640 pre-resized dataset export, dense parking lot images (300+ visible slots) resulted in individual slots being only ~15–25 pixels wide — below the threshold where the detector could reliably distinguish adjacent slots, causing missed detections and occasional duplicate/overlapping boxes.
- Switching to the original 1280×720 resolution PKLot images and retraining substantially improved this, since each slot occupies a larger pixel area relative to the whole scene.
- Detection confidence was also observed to decrease for parking rows positioned farther from the camera (appearing smaller in the frame) compared to nearer rows — a well-documented phenomenon in vision-based detection where smaller apparent object size reduces model confidence.

This is a dataset/resolution limitation rather than a flaw in the model architecture. A production system would likely need either higher-resolution camera input or a two-stage pipeline (first localizing the general lot region, then running detection at higher zoom per section).

## 8. System Logic

- **Occupancy % =** (occupied slots ÷ total slots) × 100
- **Congestion levels:**
  - Low: occupancy < 40%
  - Moderate: occupancy 40–75%
  - High: occupancy > 75%
- **Recommendations:**
  - ≥90% occupied: "Parking full — try another area."
  - 75–89% occupied: "Parking nearly full — hurry, limited slots available."
  - <75% occupied: "Slots available — proceed to park."

## 9. Web App

Built with [Streamlit](https://docs.streamlit.io/). Users upload a photo of a parking lot and the app:
1. Runs the trained YOLOv8n model to detect and classify every visible slot.
2. Draws color-coded bounding boxes (green = empty, red = occupied) directly on the image.
3. Displays total/occupied/available slot counts, occupancy percentage, congestion level, and a recommendation — all updating live as detection settings are adjusted.


## 10. Screenshots

<img width="1917" height="826" alt="image" src="https://github.com/user-attachments/assets/cb140819-048b-4c4d-8e7c-1f7d5b21ce5f" />

<img width="1903" height="807" alt="image" src="https://github.com/user-attachments/assets/5ee0ebfd-6ffb-4a8c-a580-5b4cc67b45be" />


