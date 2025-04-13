#!/usr/bin/env python3
"""
train_segmentation.py

This script trains a YOLO-based segmentation model using a segmentation dataset defined
in a YAML configuration file. It then evaluates the model on the validation set and
runs inference on unlabeled images, outputting both overlaid segmentation results and
YOLO-format text predictions.
"""

import argparse
from ultralytics import YOLO

def main(args):
    # Initialize the YOLO segmentation model with the specified weights.
    print("Initializing YOLO segmentation model with weights:", args.weights)
    model = YOLO(args.weights)
    
    # Train the segmentation model using the dataset configuration.
    print("Starting segmentation training...")
    model.train(
        data=args.data,          # Path to your segmentation YAML file
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        project=args.project,
        name=args.run_name
    )
    
    # Evaluate the trained model on the validation set.
    print("Evaluating the trained segmentation model...")
    eval_results = model.val()
    print("Evaluation Metrics:")
    print(eval_results.metrics)
    
    # Run inference on unlabeled images to generate segmentation outputs.
    print("Running segmentation inference on unlabeled images...")
    preds = model.predict(
        source=args.unlabeled_dir,  # Directory containing unlabeled/test images
        imgsz=args.imgsz,
        conf=args.conf_thres,
        save=True,        # Saves images with segmentation masks overlaid
        save_txt=True     # Saves predictions in YOLO-format text files
    )
    print("Segmentation inference complete. Check the outputs in the default runs/detect folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Segmentation Training and Inference (1 class)")
    parser.add_argument('--data', type=str, required=True,
                        help='Path to segmentation YAML configuration file')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size for training/inference')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--weights', type=str, default='yolov8n-seg.pt',
                        help='Initial segmentation weights file (e.g., yolov8n-seg.pt)')
    parser.add_argument('--project', type=str, default='runs/train',
                        help='Directory to save training runs')
    parser.add_argument('--run_name', type=str, default='ground_segmentation',
                        help='Name for this segmentation training run')
    parser.add_argument('--unlabeled_dir', type=str, required=True,
                        help='Directory containing unlabeled images for segmentation inference')
    parser.add_argument('--conf_thres', type=float, default=0.25,
                        help='Confidence threshold for inference predictions')
    
    args = parser.parse_args()
    main(args)
