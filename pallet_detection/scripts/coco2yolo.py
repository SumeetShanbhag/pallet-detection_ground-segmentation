#!/usr/bin/env python3

import json
import os
import argparse
from PIL import Image

def convert_bbox(size, bbox):
    """
    Convert COCO bbox [x, y, width, height] to YOLO format:
    [x_center, y_center, width, height] normalized by image dimensions.
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x, y, w, h = bbox
    x_center = x + w / 2.0
    y_center = y + h / 2.0
    x_center *= dw
    w *= dw
    y_center *= dh
    h *= dh
    return (x_center, y_center, w, h)

def convert_coco_to_yolo(ann_file, img_dir, out_dir, class_offset=1):
    """
    Converts a COCO annotation JSON file to YOLO-format label files.
    
    :param ann_file: Path to the COCO JSON annotation file.
    :param img_dir: Directory containing the corresponding images.
    :param out_dir: Directory to output YOLO-format text files.
    :param class_offset: Offset to subtract from category_id (default=1 for 0-indexed classes).
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    # Map image id to image info.
    images = {img['id']: img for img in data['images']}
    
    # Build annotations per image.
    ann_dict = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        ann_dict.setdefault(img_id, []).append(ann)
    
    for img_id, img_info in images.items():
        img_file = img_info['file_name']
        img_path = os.path.join(img_dir, img_file)
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            continue
        
        label_lines = []
        anns = ann_dict.get(img_id, [])
        for ann in anns:
            # For one class, subtract the offset so that category_id 1 becomes class 0.
            cls = ann['category_id'] - class_offset
            bbox = ann['bbox']  # Format: [x, y, width, height]
            bbox_converted = convert_bbox((width, height), bbox)
            # YOLO label format: <class> <x_center> <y_center> <width> <height>
            line = f"{cls} {' '.join([str(round(x, 6)) for x in bbox_converted])}"
            label_lines.append(line)
        
        # Save label file with the same base name as the image.
        label_filename = os.path.splitext(img_file)[0] + ".txt"
        out_path = os.path.join(out_dir, label_filename)
        with open(out_path, 'w') as f:
            f.write("\n".join(label_lines))
    
    print(f"Converted annotations saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO annotations to YOLO format")
    parser.add_argument('--ann_path', type=str, required=True, help='Path to COCO annotation JSON file')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for YOLO-format label files')
    parser.add_argument('--classes', type=int, default=1, help='Number of classes (should be 1 for pallet)')
    args = parser.parse_args()
    
    convert_coco_to_yolo(args.ann_path, args.img_dir, args.out_dir, class_offset=1)
