# Pallet Detection & Ground Segmentation using YOLOv8 + ROS2

This repository presents an end-to-end pipeline combining computer vision and robotics:

- **Pallet Detection** using YOLOv8 object detection  
- **Ground Segmentation** using YOLOv8 instance segmentation  
- **ROS2 Nodes** to deploy both models in real-time using Python

It includes:

- Dataset processing and format conversion scripts (COCO → YOLO)
- Training pipelines for both detection and segmentation
- Evaluation metrics (mAP, Precision, Recall, etc.)
- ROS2 launchable nodes for real-time inference
- Model visualization and result comparison

## 📁 Directory Structure

```
├── pallet_detection
│   ├── configs/                # Dataset config YAML
│   ├── dataset/                # (Removed from repo to reduce size)
│   ├── requirements.txt        # Python dependencies
│   ├── runs/                   # Training & inference results (YOLOv8)
│   └── scripts/                # Training, evaluation, analysis scripts
│
├── segmentation
│   ├── configs/
│   ├── dataset/
│   ├── scripts/
│   │   ├── train_segmentation.py
│   │   ├── inference.py
│   │   └── analysis.py
│   └── runs/
│       ├── train/              # Training metrics and graphs
│       └── segment/           # Segmentation inference outputs
│
├── ros2_ws/
│   └── src/
│       ├── ground_segmentation_node/
│       │   └── models/         # best.pt (segmentation)
│       └── pallet_detection_node/
│           └── models/         # best.pt (detection)
```

## 📑 Table of Contents

- [Installation](#installation)
- [Dataset Format](#dataset-format)
- [Training Instructions](#training-instructions)
- [Evaluation & Metrics](#evaluation--metrics)
- [ROS2 Integration](#ros2-integration)
- [Sample Results](#sample-results)
- [License](#license)
