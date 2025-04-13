from ultralytics import YOLO
import os

def export_model(weights_path, output_format='onnx', dynamic=True, simplify=True):
    model = YOLO(weights_path)
    model.export(format=output_format, dynamic=dynamic, simplify=simplify)
    print(f"✅ Export complete: Model saved in '{output_format}' format.")

if __name__ == "__main__":
    weights = os.path.expanduser("/home/sumeet/model_training/segmentation/scripts/runs/train/ground_segmentation5/weights/best.pt")
    export_model(weights_path=weights)