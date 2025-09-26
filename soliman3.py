
import os
import time
import datetime
import logging
import cv2
import numpy as np
import onnxruntime
from picamera2 import Picamera2

# --- LOGGING ---
import logging

# --- LOGGING ---
logging.basicConfig(
    filename='welding_system_picamera2.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

# Konsola da log yazmak iin handler tanmlanyor
console = logging.StreamHandler()
console.setLevel(logging.INFO)

# Formatter tanmla ve console handler'a ata
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)

# Root logger'a handler' ekle
logging.getLogger('').addHandler(console)


# --- SETTINGS ---
BASE_IMAGE_DIRECTORY = "/home/solimpeks/dataset/images"
CAPTURE_INTERVAL = 2  # seconds
IMG_SIZE = 640  # YOLO input size

# --- CAMERA ---
class CameraControl:
    def __init__(self):
        self.picam2 = None
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_still_configuration(
                main={"size": (1920, 1080)},
                lores={"size": (640, 480)},
                display="lores"
            )
            self.picam2.set_controls({"AfMode": 2})
            self.picam2.configure(config)
            logging.info("Camera initialized successfully.")
        except Exception as e:
            logging.critical(f"Failed to initialize camera! Error: {e}")
            self.picam2 = None

    def start(self):
        if self.picam2:
            self.picam2.start()
            time.sleep(1)
            logging.info("Camera started.")
            return True
        return False

    def stop(self):
        if self.picam2:
            self.picam2.stop()
            logging.info("Camera stopped.")

    def set_focus_and_exposure(self, focus_position=5.0, shutter_speed=100000, gain=1.0):
        if not self.picam2:
            return False
        try:
            self.picam2.set_controls({"AfMode": 2})
            logging.info(f"Focus={focus_position}, Shutter={shutter_speed}, Gain={gain}")
            time.sleep(0.5)
            return True
        except Exception as e:
            logging.error(f"Failed to set camera controls: {e}")
            return False

    def capture_frame(self):
        if not self.picam2:
            return None
        frame= self.picam2.capture_array()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# --- YOLO ONNX DETECTOR ---
class YoloONNX:
    def __init__(self, model_path, img_size=640, conf_thres=0.25):
        self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.img_size = img_size
        self.conf_thres = conf_thres
        logging.info(f"ONNX model loaded: {model_path}")

    def letterbox(self, img, new_size=640):
        h, w = img.shape[:2]
        scale = new_size / max(h, w)
        nh, nw = int(h*scale), int(w*scale)
        img_resized = cv2.resize(img, (nw, nh))
        top = (new_size - nh) // 2
        left = (new_size - nw) // 2
        img_padded = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
        img_padded[top:top+nh, left:left+nw] = img_resized
        return img_padded, scale, top, left

    def preprocess(self, frame):
        img, scale, top, left = self.letterbox(frame, self.img_size)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))
        img = np.expand_dims(img, 0)
        return img, scale, top, left

    def postprocess(self, outputs, scale, top, left, original_shape):
        orig_h, orig_w = original_shape[:2]
        preds = np.array(outputs[0])
        if preds.ndim == 3:
            preds = np.squeeze(preds, 0)

        boxes, scores, class_ids = [], [], []
        for pred in preds:
            conf = float(pred[4])
            if conf < self.conf_thres:
                continue
            cls_id = int(pred[5])

            # Scale coordinates to original frame size
            x1 = int((pred[0] - left) / self.img_size * orig_w)
            y1 = int((pred[1] - top) / self.img_size * orig_h)
            x2 = int((pred[2] - left) / self.img_size * orig_w)
            y2 = int((pred[3] - top) / self.img_size * orig_h)

            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            class_ids.append(cls_id)

        return boxes, scores, class_ids

    def infer(self, frame):
        img, scale, top, left = self.preprocess(frame)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: img})
        return self.postprocess(outputs, scale, top, left, frame.shape)


# --- MAIN ---
camera = CameraControl()
detector = YoloONNX("/home/solimpeks/models/best.onnx", img_size=IMG_SIZE)

if camera.start():
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_dir = os.path.join(BASE_IMAGE_DIRECTORY, f"welding_test_{timestamp}")
        os.makedirs(session_dir, exist_ok=True)
        logging.info(f"Session directory: {session_dir}")

        camera.set_focus_and_exposure(focus_position=6.0, shutter_speed=60000, gain=1.0)
        photo_counter = 1
        start_time = time.time()

        while (time.time() - start_time) < 100:  # 100s test
            frame = camera.capture_frame()
            if frame is None:
                continue

            boxes, scores, class_ids = detector.infer(frame)

            for box, score, cls_id in zip(boxes, scores, class_ids):
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{cls_id} {score:.2f}", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)

            # Show live video
            cv2.imshow("Welding Detection", frame)
            filename = os.path.join(session_dir, f"photo_{photo_counter:03d}.jpg")
            cv2.imwrite(filename, frame)
            photo_counter += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(CAPTURE_INTERVAL)

    except KeyboardInterrupt:
        logging.info("Terminated by user.")
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        logging.info("Program finished.")
