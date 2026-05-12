from pathlib import Path
from ultralytics import YOLO
import numpy as np
import cv2

WEIGHTS = Path("uppa_v1_yolo11n.pt")

model_salud = YOLO(str(WEIGHTS))

def detectar_salud(image_bytes: bytes) -> dict:
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    results = model_salud.predict(image, conf=0.25, verbose=False)

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            return {"sano": None, "estado": None, "confianza_salud": None, "mensaje_salud": "No se pudo determinar el estado"}

        best     = r.boxes[r.boxes.conf.argmax()]
        cid      = int(best.cls.item())
        conf     = float(best.conf.item())
        cls_name = r.names[cid]
        sano     = cls_name.lower() == "sano"

        return {
            "sano":            sano,
            "estado":          cls_name,
            "confianza_salud": round(conf * 100, 1),
            "mensaje_salud":   "Hongo sano " if sano else " Moho verde detectado :("
        }

    return {"sano": None, "estado": None, "confianza_salud": None, "mensaje_salud": "Sin resultado"}