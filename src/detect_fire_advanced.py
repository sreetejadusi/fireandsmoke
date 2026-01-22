import cv2
import numpy as np
import onnxruntime as ort
import simpleaudio as sa
import time
import csv
from collections import deque
from datetime import datetime


MODEL_PATH = "model/best.onnx"
CAMERA_URL = "http://192.168.1.11:8080/video"
LOG_FILE = "advanced_fire_analysis.csv"
HISTORY_SIZE = 20
FLICKER_SENSITIVITY = 50
GROWTH_THRESHOLD = 0.1
W_YOLO = 0.4
W_FLICKER = 0.4
W_GROWTH = 0.2
FINAL_THRESH = 0.65


class FireAnalyzer:
    def __init__(self, obj_id, bbox, conf):
        self.obj_id = obj_id
        self.bbox = bbox
        self.yolo_conf = conf

        self.brightness_stream = deque(maxlen=HISTORY_SIZE)
        self.area_stream = deque(maxlen=HISTORY_SIZE)

        self.flicker_score = 0.0
        self.growth_score = 0.0
        self.fusion_score = 0.0
        self.is_growing = False

    def update_analysis(self, frame, bbox, conf):
        """
        Performs the advanced temporal analysis on the specific object.
        """
        self.bbox = bbox
        self.yolo_conf = conf

        x1, y1, x2, y2 = map(int, bbox)
        h, w, _ = frame.shape

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        current_brightness = 0.0
        if x2 > x1 and y2 > y1:
            roi = frame[y1:y2, x1:x2]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            current_brightness = np.mean(gray_roi) / 255.0

        self.brightness_stream.append(current_brightness)

        if len(self.brightness_stream) > 5:

            variance = np.var(self.brightness_stream)

            self.flicker_score = np.clip(variance * FLICKER_SENSITIVITY, 0, 1)
        else:
            self.flicker_score = 0.0

        current_area = (x2 - x1) * (y2 - y1)
        self.area_stream.append(current_area)

        if len(self.area_stream) > 5:

            past_areas = list(self.area_stream)[:-3]
            avg_past_area = np.mean(past_areas)

            if avg_past_area > 0:

                growth_rate = (current_area - avg_past_area) / avg_past_area

                self.growth_score = np.clip(growth_rate * 5, 0, 1)
                self.is_growing = growth_rate > GROWTH_THRESHOLD
            else:
                self.growth_score = 0.0

        self.fusion_score = (
            (W_YOLO * self.yolo_conf)
            + (W_FLICKER * self.flicker_score)
            + (W_GROWTH * self.growth_score)
        )


def calculate_iou(box1, box2):
    """Intersection Over Union for tracking"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (
        ((box1[2] - box1[0]) * (box1[3] - box1[1]))
        + ((box2[2] - box2[0]) * (box2[3] - box2[1]))
        - inter
    )
    return inter / union if union > 0 else 0


session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


try:
    wave_obj = sa.WaveObject.from_wave_file("alarm.wav")
    alarm_enabled = True
except:
    alarm_enabled = False

cap = cv2.VideoCapture(CAMERA_URL)
frame_skip = 3
count = 0

active_fires = {}
next_id = 0


with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "Timestamp",
            "ID",
            "YOLO_Conf",
            "Flicker_Score",
            "Growth_Score",
            "FINAL_SCORE",
            "Alarm",
        ]
    )

print("🔥 Advanced Analysis System Running...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % frame_skip != 0:
        continue

    img_size = 416
    img = cv2.resize(frame, (img_size, img_size))
    img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    outputs = session.run([output_name], {input_name: img})[0]

    detections = []
    h, w = frame.shape[:2]

    for det in outputs[0]:
        confidence = det[4]
        if confidence > 0.4:

            cx, cy, bw, bh = det[0], det[1], det[2], det[3]
            x1 = (cx - bw / 2) * (w / img_size)
            y1 = (cy - bh / 2) * (h / img_size)
            x2 = (cx + bw / 2) * (w / img_size)
            y2 = (cy + bh / 2) * (h / img_size)
            detections.append(([x1, y1, x2, y2], confidence))

    new_active_fires = {}

    for bbox, conf in detections:
        matched_id = None
        max_iou = 0

        for obj_id, analyzer in active_fires.items():
            iou = calculate_iou(bbox, analyzer.bbox)
            if iou > max_iou:
                max_iou = iou
                matched_id = obj_id

        if max_iou > 0.3:

            analyzer = active_fires[matched_id]
            analyzer.update_analysis(frame, bbox, conf)
            new_active_fires[matched_id] = analyzer
            del active_fires[matched_id]
        else:

            new_analyzer = FireAnalyzer(next_id, bbox, conf)
            new_analyzer.update_analysis(frame, bbox, conf)
            new_active_fires[next_id] = new_analyzer
            next_id += 1

    active_fires = new_active_fires

    current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    for obj_id, fire in active_fires.items():
        x1, y1, x2, y2 = map(int, fire.bbox)

        if fire.fusion_score > FINAL_THRESH:
            color = (0, 0, 255)
            label = f"FIRE DETECTED [{fire.fusion_score:.2f}]"
            if alarm_enabled:
                wave_obj.play()
            is_alarm = "YES"
        else:
            color = (0, 255, 0)
            label = f"Analyzing... [{fire.fusion_score:.2f}]"
            is_alarm = "NO"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            frame, label, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

        info_text = f"Flicker:{fire.flicker_score:.2f} | Growth:{fire.growth_score:.2f}"
        cv2.putText(
            frame,
            info_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    current_time,
                    obj_id,
                    f"{fire.yolo_conf:.2f}",
                    f"{fire.flicker_score:.2f}",
                    f"{fire.growth_score:.2f}",
                    f"{fire.fusion_score:.2f}",
                    is_alarm,
                ]
            )

    cv2.imshow("Advanced Fire Analysis", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
