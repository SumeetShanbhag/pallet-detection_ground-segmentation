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
