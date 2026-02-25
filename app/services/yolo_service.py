from ultralytics import YOLO
import numpy as np
import cv2
import os

MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")

model = YOLO(MODEL_PATH)


def detect_objects(image_np: np.ndarray):
    results = model(image_np)

    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            label = f"{model.names[cls]} {conf:.2f}"

            cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(
                image_np,
                label,
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )

    return image_np