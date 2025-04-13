import os
import yaml
import argparse
from ultralytics import YOLO

def create_dataset_yaml(dataset_root, yaml_out_path):
    """
    Creates YOLOv8-compatible YAML file for ground segmentation task.
    """
    data = {
        'path': dataset_root,
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'train_masks': 'train/labels',
        'val_masks': 'val/labels',
        'test_masks': 'test/labels',
        'names': {0: 'floor'},
        'nc': 1,
        'task': 'segment'
    }
    with open(yaml_out_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    print(f"YAML file saved to: {yaml_out_path}")
    return yaml_out_path


def train_ground_segmentation(yaml_path, weights, imgsz=640, epochs=100, batch=16, device='0', run_name="ground_segmentation"):
    model = YOLO(weights)
    print(f"Training using weights: {weights}")
    
    results = model.train(
        data=yaml_path,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        patience=50,
        project="runs/train",
        name=run_name,
        device=device,
        task="segment"
    )
    print(f"Training completed. Best weights at: {results.save_dir}/weights/best.pt")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ground Segmentation Training")
    parser.add_argument('--dataset_root', required=True, help="Path to dataset root directory (containing train/ valid/ etc.)")
    parser.add_argument('--yaml_path', default='configs/dataset.yaml', help="Path to save dataset.yaml")
    parser.add_argument('--weights', default='yolov8n-seg.pt', help="YOLOv8 segmentation model weights")
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='0')
    parser.add_argument('--run_name', default='ground_segmentation')
    args = parser.parse_args()

    yaml_path = create_dataset_yaml(args.dataset_root, args.yaml_path)
    train_ground_segmentation(yaml_path, args.weights, args.imgsz, args.epochs, args.batch, args.device, args.run_name)
