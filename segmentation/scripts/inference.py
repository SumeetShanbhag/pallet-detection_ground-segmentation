#!/usr/bin/env python3
from ultralytics import YOLO

# Load the trained model weights from your training directory
model = YOLO("/home/sumeet/Downloads/segmentation/scripts/runs/train/ground_segmentation5/weights/best.pt")

# Run inference on the unlabeled test images
preds = model.predict(
    source="/home/sumeet/saved_images",
    save=True,
    conf=0.25,
    show_boxes=False,   # <-- Hides the bounding-box rectangle
    show_labels=True,   # <-- If you want the class label to appear on the mask
    show_conf=True,     # <-- If you want the confidence score to appear
    retina_masks=True   # <-- Gives sharper mask edges (optional)
)

print("Inference complete. Check the output directory for visually segmented images.")
