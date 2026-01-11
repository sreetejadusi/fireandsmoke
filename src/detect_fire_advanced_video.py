import cv2
import numpy as np
import onnxruntime as ort
import simpleaudio as sa
import time
import csv
import os
from collections import deque, defaultdict
from datetime import datetime

# ===================== CONFIG =====================
MODEL_PATH = "model/best.onnx"
VIDEO_PATH = "video/testing.mp4"
LOG_CSV = "logs/detection_log_video.csv"
ALARM_WAV = "alarm.wav"

IMG_SIZE = 416
FRAME_SKIP = 60

CONF_THRESHOLD = 0.30
FINAL_SCORE_THRESH = 0.60
IOU_MATCH_THRESH = 0.4
REQUIRED_STREAK = 3
COOLDOWN = 10
MAX_HISTORY = 8

W_YOLO = 0.6
W_FLICKER = 0.2
W_GROWTH = 0.2
# ==================================================

os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)

CSV_HEADER = [
    "timestamp",
    "frame_idx",
    "obj_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "yolo_conf",
    "flicker",
    "growth",
    "final_score",
    "action",
]

write_header = not os.path.exists(LOG_CSV) or os.path.getsize(LOG_CSV) == 0
if write_header:
    with open(LOG_CSV, "w", newline="") as f:
        csv.writer(f).writerow(CSV_HEADER)


# ===================== UTILITIES =====================
def iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def compute_brightness(roi):
    if roi is None or roi.size == 0:
        return 0.0
    if roi.shape[0] < 2 or roi.shape[1] < 2:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def flicker_score(hist):
    if len(hist) < 3:
        return 0.0
    return float(np.clip(np.tanh(np.var(hist) / 100.0), 0.0, 1.0))


def growth_score(hist):
    if len(hist) < 3 or hist[0] <= 0:
        return 0.0
    return float(np.clip(np.tanh((hist[-1] - hist[0]) / (hist[0] * 2)), 0.0, 1.0))


# ===================== LOAD MODEL =====================
print("Loading ONNX model...")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print("ONNX loaded.")

# ===================== ALARM =====================
try:
    wave_obj = sa.WaveObject.from_wave_file(ALARM_WAV)
    alarm_enabled = True
except:
    alarm_enabled = False


def play_alarm():
    if alarm_enabled:
        wave_obj.play()


# ===================== STATE =====================
tracks = {}
brightness_hist = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
area_hist = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
streaks = defaultdict(int)
next_id = 0
last_alarm_time = 0

# ===================== VIDEO =====================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Could not open video.")
    exit(1)

print("🎬 Video opened. Starting advanced fire detection...\n")

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("\n✅ Video processing completed.")
        break

    frame_idx += 1
    if frame_idx % FRAME_SKIP != 0:
        continue

    h, w = frame.shape[:2]
    timestamp = datetime.now().isoformat()

    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]

    preds = session.run(None, {input_name: img})[0][0]

    detections = []
    for det in preds:
        conf = float(det[4])
        if conf < CONF_THRESHOLD:
            continue

        x1 = int((det[0] - det[2] / 2) * w)
        y1 = int((det[1] - det[3] / 2) * h)
        x2 = int((det[0] + det[2] / 2) * w)
        y2 = int((det[1] + det[3] / 2) * h)

        detections.append((x1, y1, x2, y2, conf))

    assigned = {}
    used = set()

    for i, d in enumerate(detections):
        best_iou, best_id = 0, None
        for tid, tb in tracks.items():
            if tid in used:
                continue
            iou = iou_xyxy(d[:4], tb)
            if iou > best_iou:
                best_iou, best_id = iou, tid
        if best_iou >= IOU_MATCH_THRESH:
            assigned[i] = best_id
            used.add(best_id)

    for i in range(len(detections)):
        if i not in assigned:
            next_id += 1
            assigned[i] = next_id
            tracks[next_id] = detections[i][:4]

    frame_status = "No fire detected"

    for i, tid in assigned.items():
        x1, y1, x2, y2, conf = detections[i]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        roi = frame[y1:y2, x1:x2]
        brightness = compute_brightness(roi)
        area = (x2 - x1) * (y2 - y1)

        brightness_hist[tid].append(brightness)
        area_hist[tid].append(area)

        flicker = flicker_score(brightness_hist[tid])
        growth = growth_score(area_hist[tid])
        final_score = W_YOLO * conf + W_FLICKER * flicker + W_GROWTH * growth

        action = ""
        streaks[tid] = streaks[tid] + 1 if final_score >= FINAL_SCORE_THRESH else 0

        if streaks[tid] >= REQUIRED_STREAK and time.time() - last_alarm_time > COOLDOWN:
            play_alarm()
            last_alarm_time = time.time()
            action = "ALARM"
            streaks[tid] = 0

        frame_status = f"Fire {conf*100:.1f}% | Final {final_score:.2f}"

        with open(LOG_CSV, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    timestamp,
                    frame_idx,
                    tid,
                    x1,
                    y1,
                    x2,
                    y2,
                    f"{conf:.4f}",
                    f"{flicker:.4f}",
                    f"{growth:.4f}",
                    f"{final_score:.4f}",
                    action,
                ]
            )

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    print(f"Frame {frame_idx} → {frame_status}")
    cv2.imshow("Advanced Fire Detection (Video)", cv2.resize(frame, (960, 640)))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("\n📊 Logs saved to:", LOG_CSV)
