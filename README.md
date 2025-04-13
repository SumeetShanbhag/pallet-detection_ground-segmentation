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

## 📁 Directory Structure

```
├── pallet_detection
│   ├── configs
│   │   └── dataset.yaml
│   ├── dataset
│   │   ├── test
│   │   ├── train
│   │   └── val
│   ├── requirements.txt
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
```

## 📑 Table of Contents

- [Installation](#installation)
- [Dataset Format](#dataset-format)
- [Training Instructions](#training-instructions)
- [Evaluation & Metrics](#evaluation--metrics)
- [ROS2 Integration](#ros2-integration)
- [Sample Results](#sample-results)
- [License](#license)

## ⚙️ Installation Instructions

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

### 3. (Optional) Install ROS 2 Dependencies

If you plan to run the ROS2 nodes:

- Ensure you have **ROS 2 Humble** installed and sourced:
```bash
source /opt/ros/humble/setup.bash
```

- Then build the workspace:

```bash
cd ros2_ws
colcon build
source install/setup.bash
```

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

### 📁 Final Dataset Structure


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

## Model Evaluation & Analysis

Each training run outputs various metrics and visualizations inside the `runs/train/.../` folders.

### Evaluation Metrics

- **Detection (`pallet_detection`)**
  - `metrics/precision(B)`
  - `metrics/recall(B)`
  - `metrics/mAP50(B)`
  - `metrics/mAP50-95(B)`
  
- **Segmentation (`segmentation`)**
  - `metrics/precision(M)`
  - `metrics/recall(M)`
  - `metrics/mAP50(M)`
  - `metrics/mAP50-95(M)`

### Visualizations

Key evaluation plots generated by YOLO include:
- `PR_curve.png`
- `F1_curve.png`
- `P_curve.png`
- `R_curve.png`
- `confusion_matrix.png`
- `results.png`

### Custom Analysis Scripts

Run analysis to generate simplified performance plots:

#### Detection:
```bash
cd pallet_detection/scripts
python3 analysis.py  # Generates detection_metrics.png
```

#### Segmentation:
```bash
cd segmentation/scripts
python3 analysis.py  # Generates segmentation_metrics.png
```

These scripts parse `results.csv` and plot metrics across epochs.

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
- `ultralytics` package must be installed in the Python environment accessible to ROS nodes

```bash
pip install ultralytics opencv-python
```

---

### Build the Workspace

```bash
cd ros2_ws
colcon build
source install/setup.bash
```

---

### Running the Nodes

Each node loads a trained YOLOv8 model and publishes output as a sensor_msgs/Image topic.

#### Pallet Detection Node

```bash
ros2 run pallet_detection_node pallet_detector
```

Publishes inference results to:  
`/pallet_detection`

#### Ground Segmentation Node

```bash
ros2 run ground_segmentation_node ground_segmenter
```

Publishes segmentation output to:  
`/ground_segmentation`

---

### Launch File

For automated startup, use the provided launch file:

```bash
ros2 launch pallet_detection_node pallet_detection_launch.py
```

Make sure to update the launch file path if camera or topic configuration differs.

---

## Model Conversion from COCO to YOLO Format

To train models using YOLOv8, you must convert annotations into YOLO-compatible format.

---

### Detection: COCO to YOLOv8 BBox

Script: [`pallet_detection/scripts/coco2yolo.py`](pallet_detection/scripts/coco2yolo.py)

Usage:
```bash
python3 coco2yolo.py \
  --ann_path /path/to/_annotations.coco.json \
  --img_dir /path/to/images \
  --out_dir /path/to/output/labels
```

This script:
- Converts COCO-style `[x, y, width, height]` bounding boxes to YOLO format
- Normalizes values by image dimensions
- Writes `.txt` files with one line per object:  
  `<class_id> <x_center> <y_center> <width> <height>`

---

### Segmentation: COCO to YOLOv8-SEG

Script: [`segmentation/scripts/coco2yolo.py`](segmentation/scripts/coco2yolo.py)

Usage:
```bash
python3 coco2yolo.py \
  --ann_file /path/to/_annotations.coco.json \
  --img_dir /path/to/images \
  --out_dir /path/to/output/labels
```

This script:
- Converts polygon-style COCO masks into normalized YOLO segmentation labels
- Output format:  
  `<class_id> x1 y1 x2 y2 ... xn yn`

---

### Note

- Ensure class IDs are 0-indexed (`--class_offset=1` is default)
- Empty segmentations or bboxes are ignored with appropriate logging

## Results & Visualizations

After training, multiple evaluation artifacts are generated under the `runs/train/` folders.

---

### Pallet Detection

Located at:  
`pallet_detection/runs/train/pallet_detection4/`

Includes:
- `results.csv` – contains metrics per epoch
- `results.png` – metric curves
- `confusion_matrix.png` and `confusion_matrix_normalized.png`
- `F1_curve.png`, `P_curve.png`, `R_curve.png`, `PR_curve.png`
- Batch visualizations:  
  `train_batch*.jpg`, `val_batch*_labels.jpg`, `val_batch*_pred.jpg`

> 📈 See detection metrics: `pallet_detection/scripts/detection_metrics.png`

---

### Ground Segmentation

Located at:  
`segmentation/scripts/runs/train/ground_segmentation5/`

Includes:
- `results.csv` – mask-specific metrics
- `results.png`
- `MaskF1_curve.png`, `MaskP_curve.png`, `MaskR_curve.png`, `MaskPR_curve.png`
- `Box*` curves for bounding box metrics (from YOLO)
- Segmentation predictions and label overlays

> 📈 See segmentation metrics: `segmentation/scripts/segmentation_metrics.png`

##ROS2 Inference Nodes

This repository includes two ROS2 packages that perform real-time inference using trained YOLOv8 models:

### 1. `pallet_detection_node`

📁 Location: `ros2_ws/src/pallet_detection_node`

**Features**:
- Subscribes to RGB image topic (e.g., `/robot1/zed2i/left/image_rect_color`)
- Publishes annotated detection results to `/pallet_detection`
- Loads model from: `ros2_ws/src/pallet_detection_node/models/best.pt`

Run it using:

```bash
ros2 run pallet_detection_node pallet_detector
```

---

### 2. `ground_segmentation_node`

📁 Location: `ros2_ws/src/ground_segmentation_node`

**Features**:
- Subscribes to RGB image topic
- Runs ground segmentation using YOLOv8-seg
- Publishes segmentation overlay to `/ground_segmentation`
- Loads model from: `ros2_ws/src/ground_segmentation_node/models/best.pt`

Run it using:

```bash
ros2 run ground_segmentation_node ground_segmenter
```

---

 Make sure to:
- Place model weights at the respective `models/best.pt` locations.
- Source your ROS2 workspace before running:

```bash
source ~/ros2_ws/install/setup.bash
```


## 📁 Folder Structure Reference

```
.
├── pallet_detection/                 # Training pipeline for Pallet Detection (YOLOv8-Detect)
│   ├── configs/                      # Dataset YAML
│   ├── dataset/                      # (Optional) Local dataset copy
│   ├── requirements.txt              # Detection dependencies
│   ├── runs/
│   │   ├── detect/                   # Inference predictions
│   │   └── train/pallet_detection4/  # YOLOv8 training logs & model weights
│   ├── scripts/
│   │   ├── analysis.py               # Metric plotting for detection
│   │   ├── coco2yolo.py              # COCO → YOLO bbox conversion
│   │   ├── detection_metrics.png     # Output visualization
│   │   └── train_detectiom.py        # Training script
│   └── test.py

├── segmentation/                    # Training pipeline for Ground Segmentation (YOLOv8-Seg)
│   ├── configs/
│   ├── dataset/
│   ├── scripts/
│   │   ├── analysis.py               # Metric plotting for segmentation
│   │   ├── coco2yolo.py              # COCO → YOLO segmentation polygon converter
│   │   ├── inference.py              # Manual test inference
│   │   ├── segmentation_metrics.png  # Output visualization
│   │   └── train_segmentation.py     # Training script
│   └── runs/
│       ├── segment/                  # Inference results
│       └── train/ground_segmentation5/ # YOLOv8 training logs & weights

├── ros2_ws/                         # ROS2 Workspace (built using colcon)
│   └── src/
│       ├── pallet_detection_node/    # ROS2 Python node for pallet detection
│       └── ground_segmentation_node/ # ROS2 Python node for ground segmentation

└── README.md
```

## 📜 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 Sumeet Shanbhag

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — for the YOLO model training and inference framework.
- [Roboflow](https://roboflow.com/) — for dataset preparation and augmentation.
- [Label Studio](https://labelstud.io/) — for manual annotation of ground segmentation masks.
- [OpenCV](https://opencv.org/) and [cv_bridge](https://index.ros.org/p/cv_bridge/) — for image processing in ROS2.
- [ROS2 Foxy/Humble](https://docs.ros.org/) — for middleware and robotics integration.
- [PyTorch](https://pytorch.org/) — for backbone deep learning operations.
- [Matplotlib](https://matplotlib.org/) and [Pandas](https://pandas.pydata.org/) — for training metrics analysis and visualization.

Special thanks to Peer Robotics for the opportunity and for defining a clear, practical assignment.

---

_If this repository helps you, consider starring 🌟 it to support the project._






