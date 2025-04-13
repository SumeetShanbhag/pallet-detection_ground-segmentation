#!/usr/bin/env python3
"""
verify_onnx.py

This script loads an exported YOLOv8 ONNX model and performs inference
on a sample image. It visualizes and saves the predicted bounding boxes
to confirm that the export and model function correctly.
"""

import onnxruntime as ort
import numpy as np
import cv2
import os

def preprocess(image_path, input_size=(640, 640)):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image from {image_path}")
    original = img.copy()
    img = cv2.resize(img, input_size)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    img_rgb = np.transpose(img_rgb, (2, 0, 1))  # CHW
    img_rgb = np.expand_dims(img_rgb, axis=0)  # NCHW
    return img_rgb, original

def postprocess_and_draw(output, original_img, conf_threshold=0.5):
    preds = output[0][0]  # shape: (num_boxes, 5)
    h, w = original_img.shape[:2]

    for det in preds:
        x, y, w_box, h_box, conf = det[:5]
        if conf < conf_threshold:
            continue

        # Convert xywh to xyxy
        x1 = int((x - w_box / 2) * w)
        y1 = int((y - h_box / 2) * h)
        x2 = int((x + w_box / 2) * w)
        y2 = int((y + h_box / 2) * h)

        # Draw rectangle and label
        cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{conf:.2f}"
        cv2.putText(original_img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return original_img

def run_inference(onnx_path, image_path):
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    img, original = preprocess(image_path)
    outputs = session.run(None, {input_name: img})

    print("✅ Inference successful!")
    print(f"📦 Output shape(s): {[out.shape for out in outputs]}")

    result = postprocess_and_draw(outputs, original)
    output_path = "onnx_output.jpg"
    cv2.imwrite(output_path, result)
    print(f"🖼️ Result image with boxes saved to {output_path}")

if __name__ == "__main__":
    onnx_model_path = os.path.expanduser("~/model_training/pallet_detection/runs/train/pallet_detection4/weights/best.onnx")
    sample_image_path = os.path.expanduser("~/saved_images/image_00000.png")
    run_inference(onnx_model_path, sample_image_path)
