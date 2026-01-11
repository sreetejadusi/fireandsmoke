import cv2
import numpy as np
import onnxruntime as ort
import simpleaudio as sa
import time
import csv
import os
from collections import deque, defaultdict
from datetime import datetime
import argparse


MODEL_PATH = "model/best.onnx"
CAMERA_URL = "http://192.168.1.7:8080/video"
IMG_SIZE = 416
FRAME_SKIP = 60
CONF_THRESHOLD = 0.30
IOU_MATCH_THRESH = 0.4
REQUIRED_STREAK = 3
FINAL_SCORE_THRESH = 0.60
COOLDOWN = 10
LOG_CSV = "logs/detection_log.csv"
ALARM_WAV = "alarm.wav"
MAX_HISTORY = 8


W_YOLO = 0.6
W_FLICKER = 0.2
W_GROWTH = 0.2


os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)


def iou_xyxy(a, b):

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    denom = area_a + area_b - inter + 1e-9
    return inter / denom if denom > 0 else 0.0


def xywh_to_xyxy(xywh, im_w, im_h):

    x, y, w, h = xywh
    x1 = int((x - w / 2) * im_w)
    y1 = int((y - h / 2) * im_h)
    x2 = int((x + w / 2) * im_w)
    y2 = int((y + h / 2) * im_h)
    return [x1, y1, x2, y2]


print("Loading ONNX model...")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print("ONNX loaded. Input name:", input_name)


try:
    wave_obj = sa.WaveObject.from_wave_file(ALARM_WAV)
    alarm_enabled = True
except Exception as e:
    print("Alarm load failed:", e)
    alarm_enabled = False


def play_alarm():
    if alarm_enabled:
        wave_obj.play()


next_id = 0
tracks = {}
track_hist = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
track_conf_hist = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
streaks = defaultdict(int)
last_alarm_time = 0


unassigned_life = {}


if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
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
        )


def compute_brightness(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def flicker_score_from_history(hist):

    if len(hist) < 3:
        return 0.0
    var = float(np.var(hist))

    score = np.tanh(var / 100.0)
    return float(np.clip(score, 0.0, 1.0))


def growth_score_from_history(hist):
    if len(hist) < 3:
        return 0.0

    first = hist[0]
    last = hist[-1]
    if first <= 0:
        return 0.0
    growth = (last - first) / (first + 1e-9)
    score = np.tanh(growth / 2.0)
    return float(np.clip(score, 0.0, 1.0))


cap = cv2.VideoCapture(CAMERA_URL)
if not cap.isOpened():
    print("Could not open camera stream. Exiting.")
    exit(1)

print("Camera opened. Starting advanced detection...")
frame_idx = 0

last_status_msg = "Waiting for detections..."

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not received. Retrying...")
        time.sleep(0.5)
        continue

    frame_idx += 1

    if frame_idx % FRAME_SKIP != 0:
        print(f"Frame {frame_idx} → {last_status_msg}")
        cv2.imshow("Preview (press q to quit)", cv2.resize(frame, (640, 480)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    h, w = frame.shape[:2]
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_trans = np.transpose(img_rgb, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

    outputs = session.run(None, {input_name: img_trans})
    preds = outputs[0] if outputs is not None else np.array([])

    detections = []
    for det in preds[0]:
        x_c, y_c, bw, bh = det[0], det[1], det[2], det[3]
        conf = float(det[4])
        cls = int(det[5]) if len(det) > 5 else 0
        if conf < CONF_THRESHOLD:
            continue

        x1 = int((x_c - bw / 2) * w)
        y1 = int((y_c - bh / 2) * h)
        x2 = int((x_c + bw / 2) * w)
        y2 = int((y_c + bh / 2) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        detections.append((x1, y1, x2, y2, conf, cls))

    assigned = {}
    used_tracks = set()
    det_indices = list(range(len(detections)))

    for di in det_indices:
        bx = detections[di][:4]
        best_id = None
        best_iou = 0.0
        for tid, tb in tracks.items():
            if tid in used_tracks:
                continue
            val = iou_xyxy(bx, tb)
            if val > best_iou:
                best_iou = val
                best_id = tid
        if best_iou >= IOU_MATCH_THRESH and best_id is not None:
            assigned[di] = best_id
            used_tracks.add(best_id)

    for di in det_indices:
        if di not in assigned:
            next_id += 1
            tid = next_id
            assigned[di] = tid
            tracks[tid] = detections[di][:4]
            track_hist[tid].clear()
            track_conf_hist[tid].clear()

    if "area_hist" not in globals():
        area_hist = defaultdict(lambda: deque(maxlen=MAX_HISTORY))

    timestamp = datetime.utcnow().isoformat()

    frame_status = "No fire detected"

    for di, tid in assigned.items():
        bbox = detections[di][:4]
        conf = detections[di][4]

        x1, y1, x2, y2 = bbox
        roi = frame[y1 : y2 + 1, x1 : x2 + 1]
        brightness = compute_brightness(roi) if roi.size else 0.0
        area = (x2 - x1) * (y2 - y1)

        track_hist[tid].append(brightness)
        track_conf_hist[tid].append(conf)
        area_hist[tid].append(area)

        flicker = flicker_score_from_history(list(track_hist[tid]))
        growth = growth_score_from_history(list(area_hist[tid]))

        final_score = W_YOLO * conf + W_FLICKER * flicker + W_GROWTH * growth

        action = ""
        if final_score >= FINAL_SCORE_THRESH:
            streaks[tid] += 1
        else:
            streaks[tid] = 0

        if (
            streaks[tid] >= REQUIRED_STREAK
            and (time.time() - last_alarm_time) > COOLDOWN
        ):
            play_alarm()
            last_alarm_time = time.time()
            action = "ALARM"
            streaks[tid] = 0

        color = (0, 0, 255) if final_score >= FINAL_SCORE_THRESH else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = (
            f"ID:{tid} Y:{conf:.2f} F:{flicker:.2f} G:{growth:.2f} S:{final_score:.2f}"
        )
        cv2.putText(
            frame,
            label,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )

        frame_status = f"Fire {conf*100:.1f}% | Flicker {flicker:.2f} | Growth {growth:.2f} | Final {final_score:.2f}"

        with open(LOG_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
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

    print(f"Frame {frame_idx} → {frame_status}")
    last_status_msg = frame_status

    cv2.imshow(
        "Advanced Fire Detection (press q to quit)", cv2.resize(frame, (960, 640))
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
print("Stopping. Logs saved to", LOG_CSV)
