from ultralytics import YOLO
import numpy as np
def dummy_npwarn_decorator_factory():
  def npwarn_decorator(x):
    return x
  return npwarn_decorator
np._no_nep50_warning = getattr(np, '_no_nep50_warning', dummy_npwarn_decorator_factory)
# Load the YOLO11 model
model = YOLO("models/best4.pt")

# Export the model to TFLite format
model.export(format="tflite")  # crea9tes 'yolo11n_float32.tflite'

# Load the exported TFLite model
tflite_model = YOLO("yolo11n_float32.tflite")

# Run inference
