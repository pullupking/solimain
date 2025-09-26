from ultralytics import YOLO

import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
import time as t
import random

picam2 = Picamera2()
config = picam2.create_still_configuration(
            main={"size": (1920, 1080)},
            lores={"size": (640, 480)},
            display="lores"
)

picam2.configure(config)
picam2.set_controls({"AfMode": 2})
picam2.start()
# Load a model
model = YOLO("models/best3_float16.tflite")  # pretrained YOLO11n model

# Run batched inference on a list of images
def getColours(cls_num):
    """Generate unique colors for each class ID"""
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))
    
while True:
    print("Started..")
    tstart = t.time()
    frame =picam2.capture_array()
    print("frame alındı")
    #frame = cv2.resize(frame, (640, 640))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print("frame alındı")
    results = model(frame)  # return a list of Results objects
    
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        class_ids = result.masks  # Masks object for segmentation masks outputs
        scores = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        #result.show()  # display to screen
        #result.save(filename="result.jpg")  # save to disk
    

            # Draw boxes
    for result in results:
        class_names = result.names
        for box in result.boxes:
            if box.conf[0] > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cls = int(box.cls[0])
                class_name = class_names[cls]

                conf = float(box.conf[0])

                colour = getColours(cls)

                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                cv2.putText(frame, f"{class_name} {conf:.2f}",
                            (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, colour, 2)

            # Show live frame
    cv2.imshow("Welding Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            
    print(t.time() - tstart)
    

# Process results list
