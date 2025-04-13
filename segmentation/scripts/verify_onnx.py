#!/usr/bin/env python3
"""
verify_onnx.py

This script loads an exported YOLOv8 ONNX segmentation model and performs inference
on a sample image. It overlays the segmentation mask to visually verify the model.
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

def postprocess_mask(output, original_img):
    """
    Takes output of YOLOv8 segmentation and overlays mask onto original image.
    """
    mask_output = output[0][0]  # shape: (1, num_classes, H, W) or (1, 1, H, W)
    mask = mask_output[0]       # first class mask

    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]))

    colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img, 0.7, colored_mask, 0.3, 0)

    return overlay

def run_inference(onnx_path, image_path):
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    img, original = preprocess(image_path)
    outputs = session.run(None, {input_name: img})

    print("Inference successful!")
    print(f"Output shape(s): {[out.shape for out in outputs]}")

    result = postprocess_mask(outputs, original)
    output_path = "onnx_segmentation_output.jpg"
    cv2.imwrite(output_path, result)
    print(f"Segmentation result saved to {output_path}")

if __name__ == "__main__":
    onnx_model_path = os.path.expanduser("/home/sumeet/model_training/segmentation/scripts/runs/train/ground_segmentation5/weights/best.onnx")
    sample_image_path = os.path.expanduser("~/saved_images/image_00000.png")
    run_inference(onnx_model_path, sample_image_path)
