from ultralytics import YOLO
import cv2
import random
from picamera2 import Picamera2
from opcua import Client
import time
import logging

# OPC UA ayarlarÄ±
OPC_SERVER_URL = "opc.tcp://10.92.8.50:4840"
STOP_NODE = 'ns=3;s="HAT2_HMI_M_Data"."WDKM"."Stop"'

def getColours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

def ensure_connection(client): 
    
    """OPC baglantisi aktif mi?."""
    
    try:
        client.get_node("i=84").get_value()
        return True
    except Exception as e:
        logging.warning(f"OPC UA baglanti kontrolunde hata: {e}")
        return False

try:
    # OPC UA baglantisi kur
    opc_client = Client(OPC_SERVER_URL)
    opc_client.connect()
    stop_node = opc_client.get_node(STOP_NODE)
    print(" OPC baglantisi kuruldu")

    # Kamera baslat
    picam2 = Picamera2()
    config = picam2.create_still_configuration(
        main={"size": (1920, 1080)},
        lores={"size": (640, 480)},
        display="lores"
    )
    picam2.configure(config)
    picam2.set_controls({"AfMode": 2})
    picam2.start()
    print(" Kamera baslatildi")

    # Modeli yukle
    model = YOLO("models/best3_float16.tflite")
    print(" YOLO modeli yuklendi")

    last_stop_state = None  # Sadece degisince sinyal gonder

    while True:
        tstart = time.time()
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model(frame)
        detected_classes = set()
        for result in results:
            boxes = result.boxes
            class_names = result.names
            for box in boxes:
                if box.conf[0] > 0.4:
                    cls = int(box.cls[0])
                    class_name = class_names[cls]
                    detected_classes.add(class_name)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    colour = getColours(cls)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                    cv2.putText(frame, f"{class_name} {conf:.2f}",
                                (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, colour, 2)

        # --- OPC UA Baglanti Kontrolu---
        if not ensure_connection(opc_client):
            logging.error("OPC UA baglantisi koptu, yeniden baslanlatiliyor...")
            try:
                opc_client.disconnect()
            except Exception:
                pass
            time.sleep(1)  # kisa bekleme
            opc_client.connect()
            stop_node = opc_client.get_node(STOP_NODE)
            logging.info("OPC UA baglantisi yeniden kuruldu.")

        # --- Gereksiz sinyal gondermemek iÃ§in kontrol ---
        stop_state = "hata" in detected_classes
        if stop_state != last_stop_state:
            stop_node.set_value(stop_state)
            if stop_state:
                print(" Hata bulundu  PLC'ye STOP=1 gonderildi")
            else:
                print(" Hata yok  PLC'ye STOP=0 gonderildi")
            last_stop_state = stop_state

        cv2.imshow("Welding Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print(f"Dongu suresi: {time.time() - tstart:.2f} sn")

except Exception as e:
    print(f"Beklenmeyen hata: {e}")
finally:
    try:
        opc_client.disconnect()
    except: pass
    try:
        picam2.stop()
    except: pass
    cv2.destroyAllWindows()
    print("Program sonlandirildi, baglantilar kapatildi")
