#!/usr/bin/env python3
"""
convert_coco_seg_to_yolo.py

This script converts COCO segmentation annotations to YOLOv5-seg format.
For each image in the _annotations.json file, it creates a corresponding
.txt file (saved in an output directory) that contains lines of the form:

    <class_id> x1 y1 x2 y2 ... xn yn

where the coordinates are normalized (i.e. divided by image width/height).
"""

import os
import json
import argparse
from PIL import Image

def convert_polygon(img_size, polygon):
    """
    Normalize a polygon’s coordinates given the image size.
    :param img_size: (width, height)
    :param polygon: list of numbers [x1, y1, x2, y2, ..., xn, yn]
    :return: list of normalized coordinates
    """
    width, height = img_size
    normalized = []
    # Assume polygon is a flat list of [x1, y1, x2, y2, ...]
    for i in range(0, len(polygon), 2):
        x = polygon[i] / width
        y = polygon[i+1] / height
        normalized.extend([x, y])
    return normalized

def convert_coco_segmentation_to_yolo(ann_file, img_dir, out_dir, class_offset=1):
    """
    Converts a COCO segmentation annotations file to YOLOv5-seg formatted label files.
    :param ann_file: Path to the _annotations.json file.
    :param img_dir: Directory containing the corresponding images.
    :param out_dir: Directory to output YOLOv5-seg formatted label files.
    :param class_offset: Offset to subtract from category_id (default=1 for converting 1-indexed to 0-indexed).
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    # Build a mapping from image id to image info
    images = {img['id']: img for img in data['images']}
    
    # Group annotations by image
    ann_dict = {}
    for ann in data.get('annotations', []):
        img_id = ann['image_id']
        ann_dict.setdefault(img_id, []).append(ann)
    
    for img_id, img_info in images.items():
        img_name = img_info['file_name']
        img_path = os.path.join(img_dir, img_name)
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            continue

        label_lines = []
        annotations = ann_dict.get(img_id, [])
        for ann in annotations:
            # Convert category_id to 0-indexed class id
            cls = ann.get('category_id', 1) - class_offset
            # Get segmentation data; could be multiple polygons per annotation
            seg_data = ann.get('segmentation', [])
            if isinstance(seg_data, list):
                for polygon in seg_data:
                    if not polygon:  # Skip empty polygons
                        continue
                    normalized_poly = convert_polygon((width, height), polygon)
                    # Form a line: class followed by normalized coordinates with 6 decimal precision
                    line = f"{cls} " + " ".join(f"{coord:.6f}" for coord in normalized_poly)
                    label_lines.append(line)
        
        # Save the label file with the same basename as the image
        label_filename = os.path.splitext(img_name)[0] + ".txt"
        out_path = os.path.join(out_dir, label_filename)
        with open(out_path, 'w') as f:
            f.write("\n".join(label_lines))
        print(f"Saved {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO segmentation annotations to YOLOv8-seg format")
    parser.add_argument("--ann_file", type=str, required=True, help="Path to the _annotations.json file")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for label files")
    parser.add_argument("--class_offset", type=int, default=1, help="Offset for class IDs (default=1, to convert COCO’s 1-indexed to 0-indexed)")
    args = parser.parse_args()
    
    convert_coco_segmentation_to_yolo(args.ann_file, args.img_dir, args.out_dir, args.class_offset)
