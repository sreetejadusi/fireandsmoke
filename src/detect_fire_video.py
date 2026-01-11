import cv2
import numpy as np
import onnxruntime as ort
import simpleaudio as sa
import time
import csv
import os
from datetime import datetime
from collections import deque

MODEL_PATH = "model/best.onnx"
VIDEO_PATH = "video/testing.mp4"
IMG_SIZE = 416
CONF_THRESHOLD = 0.60
REQUIRED_STREAK = 10
FRAME_SKIP = 20
COOLDOWN = 10

LOG_CSV = "logs/detection_log_yolo_only.csv"
os.makedirs("logs", exist_ok=True)

detections_queue = deque(maxlen=REQUIRED_STREAK)

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

try:
    wave_obj = sa.WaveObject.from_wave_file("alarm.wav")
    alarm_enabled = True
except:
    alarm_enabled = False

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Could not open video.")
    exit()

if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "frame_idx", "max_confidence", "alarm_triggered"])

frame_idx = 0
last_alarm_time = 0

print("Running YOLO-only detection on video...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    if frame_idx % FRAME_SKIP != 0:
        continue

    timestamp = datetime.utcnow().isoformat()

    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]

    outputs = session.run(None, {input_name: img})[0][0]

    fire_conf = [det[4] for det in outputs if det[4] >= 0.01]

    max_conf = max(fire_conf) if fire_conf else 0.0
    detections_queue.append(max_conf)

    alarm = 0
    now = time.time()

    if (
        len(detections_queue) == REQUIRED_STREAK
        and all(c >= CONF_THRESHOLD for c in detections_queue)
        and now - last_alarm_time > COOLDOWN
    ):
        alarm = 1
        last_alarm_time = now
        detections_queue.clear()
        if alarm_enabled:
            wave_obj.play()

    with open(LOG_CSV, "a", newline="") as f:
        csv.writer(f).writerow([timestamp, frame_idx, f"{max_conf:.4f}", alarm])

    print(f"Frame {frame_idx} | YOLO conf: {max_conf:.2f} | Alarm: {alarm}")

cap.release()
print("YOLO-only detection finished.")
