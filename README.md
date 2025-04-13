# Pallet Detection & Ground Segmentation using YOLOv8 + ROS2 Node Deployment

This repository presents an end-to-end pipeline combining computer vision and robotics:

- **Pallet Detection** using YOLOv8 object detection  
- **Ground Segmentation** using YOLOv8 instance segmentation  
- **ROS2 Nodes** to deploy both models in real-time using Python

It includes:

- Dataset processing and format conversion scripts (COCO.json → YOLO.txt)
- Training pipelines for both detection and segmentation
- Evaluation metrics (mAP, Precision, Recall, etc.)
- ROS2 launchable nodes for real-time inference
- Model visualization and result comparison

## Table of Contents

- [Installation](#️-installation-instructions)
- [Dataset Format](#dataset-format)
- [Model Training](#model-training)
- [Key Evaluation Visualizations](#key-evaluation-visualizations)
- [Model Evaluation, Metrics & Analysis](#model-evaluation--analysis)
- [ROS2 Node Setup & Inference](#ros2-node-setup--inference)
- [Ground Segmentation Model](https://github.com/SumeetShanbhag/pallet-detection_ground-segmentation/blob/main/ros2_ws_src/ground_segmentation_node/models/best.pt)
- [Pallet Detection Model](https://github.com/SumeetShanbhag/pallet-detection_ground-segmentation/blob/main/ros2_ws_src/pallet_detection_node/models/best.pt)

## Directory Structure

```
├── pallet_detection
│   ├── configs
│   │   └── dataset.yaml
│   ├── dataset
│   │   ├── test
│   │   ├── train
│   │   └── val
│   ├── runs
│   │   ├── detect
│   │   └── train
│   ├── scripts
│   │   ├── analysis.py
│   │   ├── coco2yolo.py
│   │   ├── detection_metrics.png
│   │   ├── onnx_output.jpg
│   │   ├── optimize_model.py
│   │   ├── train_detectiom.py
│   │   └── verify_onnx.py
│   └── test.py
└── segmentation
|   ├── configs
|   │   └── dataset.yaml
|   ├── dataset
|   │   ├── test
|   │   ├── train
|   │   └── val
|   └── scripts
|       ├── analysis.py
|       ├── coco2yolo.py
|       ├── inference.py
|       ├── onnx_segmentation_output.jpg
|       ├── optimize_model.py
|       ├── runs
|       ├── segmentation_metrics.png
|       ├── train_segmentation.py
|       └── verify_onnx.py
│
├── ros2_ws_src
│   ├── ground_segmentation_node
│   │   ├── ground_segmentation_node
│   │   │   ├── ground_segmenter.py
│   │   │   └── __init__.py
│   │   ├── models
│   │   │   └── best.pt
│   │   ├── package.xml
│   │   ├── resource
│   │   │   └── ground_segmentation_node
│   │   ├── setup.cfg
│   │   └── setup.py
│   └── pallet_detection_node
│       ├── launch
│       │   └── pallet_detection_launch.py
│       ├── models
│       │   └── best.pt
│       ├── package.xml
│       ├── pallet_detection_node
│       │   ├── __init__.py
│       │   ├── __main__.py
│       │   └── pallet_detector.py
│       ├── resource
│       │   └── pallet_detection_node
│       ├── setup.cfg
│       └── setup.py
├── requirements.txt
```
  
## Installation Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/SumeetShanbhag/pallet-detection_ground-segmentation.git
cd pallet-detection_ground-segmentation
```

### 2. Install Dependencies

Install the core dependencies for both segmentation and detection:

```bash
pip install -r requirements.txt
```

> Note: This includes `ultralytics`, `opencv-python`, `numpy`, and `matplotlib`.

## Dataset Preparation (Using Roboflow)

This project uses warehouse image datasets annotated and preprocessed via [Roboflow](https://roboflow.com/) for both **pallet detection** and **ground segmentation** tasks.

### Source Dataset Summary
- **519 warehouse images** were initially provided.
- An additional **654 warehouse images** were used *only for pallet detection*.

---

### Annotation Strategy

#### Manual Annotation
- **128 images** were **manually annotated** with:
  - **Bounding boxes** for pallet detection
  - **Polygon masks** for ground segmentation

#### Auto Annotation
- The **remaining images** from the 519-image dataset were **auto-annotated** using Roboflow's smart labeling tools.

---

### Dataset Composition

#### Pallet Detection Dataset
- Total dataset: **1,173 images**
  - 128 manually annotated (from 519 original)
  - Remaining 391 images auto-annotated
  - Plus 654 images from an additional warehouse dataset
- Steps:
  1. Combined into a single dataset
  2. Split into `train`, `val`, and `test` sets
  3. Data **augmentation** done in Roboflow (e.g., flipping, brightness)
  4. Exported in **COCO format**
  5. Converted to **YOLOv8 format** using [`pallet_detection/scripts/coco2yolo.py`](pallet_detection/scripts/coco2yolo.py)

####  Ground Segmentation Dataset
- Total dataset: **519 images**
  - 128 manually annotated
  - Remaining 391 auto-annotated
  - *Does not use* the 654 additional warehouse images
- Steps:
  1. Split into `train`, `val`, and `test`
  2. Augmented in Roboflow
  3. Exported in **COCO format**
  4. Converted to **YOLOv8-seg format** using [`segmentation/scripts/coco2yolo.py`](segmentation/scripts/coco2yolo.py)

---

### Final Dataset Structure


## Dataset Format

### Detection Dataset Structure (YOLO format)

```
pallet_detection/
└── dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

- Labels are in YOLOv8 format:
  ```
  <class_id> <x_center> <y_center> <width> <height>
  ```

### Segmentation Dataset Structure (YOLOv8-seg format)

```
segmentation/
└── dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

- Labels are polygon masks in YOLOv8-seg format:
  ```
  <class_id> x1 y1 x2 y2 ... xn yn
  ```

Both formats were converted from COCO using coco2yolo.py scripts in the `scripts/` directory of each task.

---

## Model Training

### YOLOv5 Pallet Detection Training

Run the following command from the `pallet_detection` directory. Please adjust based on your location: 

```bash
python3 train_detectiom.py \
  --data ../pallet_detection/configs/dataset.yaml \
  --unlabeled_dir ~/Downloads/unlabeled_test_images \
  --weights yolov8n-seg.pt \
  --epochs 50 \
  --batch_size 16 \
  --imgsz 640 \
  --project runs/train \
  --run_name ground_segmentation \
  --conf_thres 0.25
```

- Uses bounding box annotations
- Outputs are saved in `runs/train/pallet_detection4/`

### YOLOv8 Ground Segmentation Training

Run from the `segmentation` directory. Please adjust based on your location:

```bash
python3 train_segmentation.py \
  --data ../configs/dataset.yaml \
  --weights yolov8n-seg.pt \
  --imgsz 640 \
  --epochs 100 \
  --batch_size 8 \
  --project runs/train \
  --run_name ground_segmentation \
  --unlabeled_dir ../dataset/test/images \
  --conf_thres 0.4
```

- Uses polygonal segmentation masks
- Outputs are saved in `scripts/runs/train/ground_segmentation5/`

**Note:** Both commands can be customized via `args.yaml`.

---


## Key Evaluation Visualizations

Below are the primary performance plots and metrics used to evaluate model robustness and generalization across both tasks.

---

###Pallet Detection

#### Precision-Recall Curve
![PR Curve - Detection](https://github.com/SumeetShanbhag/pallet-detection_ground-segmentation/blob/main/pallet_detection/runs/train/pallet_detection4/PR_curve.png) 

#### Confusion Matrix
![Confusion Matrix - Detection](https://github.com/SumeetShanbhag/pallet-detection_ground-segmentation/blob/main/pallet_detection/runs/train/pallet_detection4/confusion_matrix.png) 

#### Training Metrics Over Epochs
![Training Results - Detection](https://github.com/SumeetShanbhag/pallet-detection_ground-segmentation/blob/main/pallet_detection/runs/train/pallet_detection4/results.png)

#### Inference Output
Example output from the trained detection model:
![Detection Output](https://github.com/SumeetShanbhag/pallet-detection_ground-segmentation/blob/main/pallet_detection/runs/detect/predict2/image_00000.jpg)

---

### Ground Segmentation

#### Precision-Recall Curve (Segmentation Masks)
![PR Curve - Segmentation](https://github.com/SumeetShanbhag/pallet-detection_ground-segmentation/blob/main/segmentation/scripts/runs/train/ground_segmentation5/MaskPR_curve.png)

#### Confusion Matrix
![Confusion Matrix - Segmentation](https://github.com/SumeetShanbhag/pallet-detection_ground-segmentation/blob/main/segmentation/scripts/runs/train/ground_segmentation5/confusion_matrix.png) 

#### Training Metrics Over Epochs
![Training Results - Segmentation](https://github.com/SumeetShanbhag/pallet-detection_ground-segmentation/blob/main/segmentation/scripts/runs/train/ground_segmentation5/results.png)

#### Inference Output
Example output from the trained segmentation model:
![Segmentation Output](https://github.com/SumeetShanbhag/pallet-detection_ground-segmentation/blob/main/segmentation/scripts/runs/segment/predict5/image_00000.jpg)


## Model Evaluation & Analysis

Each training run outputs various metrics and visualizations inside the `runs/train/.../` folders.

Run analysis to generate simplified performance plots:

### Detection:
```bash
cd pallet_detection/scripts
python3 analysis.py  # Generates detection_metrics.png
```
Ground Segmentation Performance Metrics
---
![Segmentation Metrics](https://github.com/SumeetShanbhag/pallet-detection_ground-segmentation/blob/main/segmentation/scripts/segmentation_metrics.png)

---
Pallet Detection Metrics
---
(from results.csv, epoch 50)

- **Precision (B)**: `0.92575`
- **Recall (B)**: `0.91715`
- **mAP@0.5 (B)**: `0.96188`
- **mAP@0.5:0.95 (B)**: `0.69506`

### Segmentation:
```bash
cd segmentation/scripts
python3 analysis.py  # Generates segmentation_metrics.png
```

Pallet Detection Performance Metrics:
---
![Detection Metrics](https://github.com/SumeetShanbhag/pallet-detection_ground-segmentation/blob/main/pallet_detection/scripts/detection_metrics.png)

Ground Segmentation Metrics
---
(from results.csv, epoch 100)

- **Precision (M)**: `0.92861`
- **Recall (M)**: `0.82609`
- **mAP@0.5 (M)**: `0.87897`
- **mAP@0.5:0.95 (M) (IoU-based)**: `0.81067`


These scripts parse `results.csv` and plot metrics across epochs.

---

## Optional: Model Optimization for Edge Deployment

As part of the assignment's optional optimization task, this project includes scripts that prepare the YOLOv8 models for deployment on edge devices like the NVIDIA Jetson AGX Orin.

### Optimization and Verification Scripts

#### Detection (Pallet Detection)
- [optimize_model.py](https://github.com/SumeetShanbhag/pallet-detection_ground-segmentation/blob/main/pallet_detection/scripts/optimize_model.py) — Converts the trained PyTorch model to ONNX and applies performance optimizations.
- [verify_onnx.py](https://github.com/SumeetShanbhag/pallet-detection_ground-segmentation/blob/main/pallet_detection/scripts/verify_onnx.py) — Verifies the ONNX model output matches the original PyTorch results.

#### Segmentation (Ground Segmentation)
- [optimize_model.py](https://github.com/SumeetShanbhag/pallet-detection_ground-segmentation/blob/main/segmentation/scripts/optimize_model.py) — Converts the segmentation model to ONNX and optimizes it for inference.
- [verify_onnx.py](https://github.com/SumeetShanbhag/pallet-detection_ground-segmentation/blob/main/segmentation/scripts/verify_onnx.py) — Validates the accuracy of ONNX segmentation output.


---
## ROS2 Node Setup & Inference

This project includes two ROS 2 packages:

- `pallet_detection_node`
- `ground_segmentation_node`

Both are located under the [`ros2_ws/src/`](ros2_ws/src) directory and built using `colcon`.

---

### Prerequisites

- ROS 2 Humble
- Python 3.10+
- Required Python packages (see [requirements.txt](../requirements.txt))
- `ultralytics` and `opencv-python` must be installed in the Python environment accessible to ROS nodes:

```bash
pip install ultralytics opencv-python
```

---

### Build the Workspace

```bash
cd ~/ros2_ws
colcon build
source install/setup.bash
```

---

### Inference Runtime Setup (Step-by-Step)

#### 1. Launch the ROS2 workspace

```bash
cd ~/ros2_ws
colcon build
source install/setup.bash
```

#### 2. Open a new terminal: Launch Pallet Detection Node

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run pallet_detection_node pallet_detector
```

Publishes inference results to:  
`/pallet_detection`

#### 3. Open another terminal: Launch Ground Segmentation Node

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run ground_segmentation_node ground_segmenter
```

Publishes segmentation output to:  
`/ground_segmentation`

#### 4. Open another terminal: Play the ROS bag file (replace with correct path if needed)

```bash
ros2 bag play ~/internship_assignment_sample_bag --loop
```

This will publish camera data to:  
`/robot1/zed2i/left/image_rect_color`

#### 5. Open another terminal: Launch `rqt_image_view` to visualize outputs

```bash
ros2 run rqt_image_view rqt_image_view
```

Once open, choose one of the following image topics to view:

- `/pallet_detection` → View YOLOv8 pallet detection results
- `/ground_segmentation` → View ground segmentation overlay
- `/robot1/zed2i/left/image_rect_color` → Raw input stream from ROS bag

---
### ROS Runtime results

---
#### 1. `pallet_detection_node`

 Location: `ros2_ws/src/pallet_detection_node`

**Features**:
- Subscribes to RGB image topic: `/robot1/zed2i/left/image_rect_color`
- Publishes annotated detection results to `/pallet_detection`
- Loads YOLOv8 model from: `ros2_ws/src/pallet_detection_node/models/best.pt`

/pallet_detection topic on ROS:
---
![Detection ROS](https://github.com/SumeetShanbhag/pallet-detection_ground-segmentation/blob/main/pallet_detection/ROS_detection.png)
---

#### 2. `ground_segmentation_node`

 Location: `ros2_ws/src/ground_segmentation_node`

**Features**:
- Subscribes to RGB image topic: `/robot1/zed2i/left/image_rect_color`
- Publishes segmentation overlay to `/ground_segmentation`
- Loads YOLOv8 model from: `ros2_ws/src/ground_segmentation_node/models/best.pt`

/ground_segmentation topic on ROS:
---
![Segmentation ROS](https://github.com/SumeetShanbhag/pallet-detection_ground-segmentation/blob/main/segmentation/ROS_segmentation.png)
---

**Note**:
- Ensure model weights (`best.pt`) are placed correctly under each node's `models/` folder.
- Source your workspace in every new terminal **before** running nodes or visualizers:

```bash
source ~/ros2_ws/install/setup.bash
```
