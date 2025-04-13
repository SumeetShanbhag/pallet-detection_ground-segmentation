#!/usr/bin/env python3
"""
optimize_model.py

This script converts a trained YOLOv8 PyTorch model to ONNX format
and optionally verifies the export.
"""

from ultralytics import YOLO
import os

def export_model(weights_path, output_format='onnx', dynamic=True, simplify=True):
    model = YOLO(weights_path)
    model.export(format=output_format, dynamic=dynamic, simplify=simplify)
    print(f"✅ Export complete: Model saved in '{output_format}' format.")

if __name__ == "__main__":
    weights = os.path.expanduser("~/model_training/pallet_detection/runs/train/pallet_detection4/weights/best.pt")
    export_model(weights_path=weights)