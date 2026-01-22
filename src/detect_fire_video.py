import cv2
import numpy as np
import onnxruntime as ort
import simpleaudio as sa
import time
from datetime import datetime
from collections import deque

MODEL_PATH = "model/best.onnx"
VIDEO_PATH = "video/testing.mp4"
IMG_SIZE = 416
CONF_THRESHOLD = 0.60
REQUIRED_STREAK = 10
FRAME_SKIP = 20
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

print("Running detection on video...\n")

frame_counter = 0
last_alarm_time = 0
cooldown = 10

while True:
    ret, frame = cap.read()
    if not ret:
        print("\nFinished processing video.")
        break

    frame_counter += 1

    if frame_counter % FRAME_SKIP != 0:
        continue

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img[:, :, ::-1]
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    outputs = session.run(None, {input_name: img})[0]
    predictions = outputs[0]

    fire_confidences = []
    for det in predictions:
        conf = det[4]
        cls = int(det[5])
        if cls in [0, 1] and conf >= 0.01:
            fire_confidences.append(conf)

    if fire_confidences:
        max_conf = max(fire_confidences)
        detections_queue.append(max_conf)
        print(f"{timestamp} | Fire Probability: {max_conf * 100:.1f}%")
    else:
        detections_queue.append(0.0)
        print(f"{timestamp} | No Fire Detected")

    now = time.time()

    if (
        len(detections_queue) == REQUIRED_STREAK
        and all(c >= CONF_THRESHOLD for c in detections_queue)
        and now - last_alarm_time > cooldown
    ):
        print(f"\n{timestamp} | FIRE DETECTED — ALARM TRIGGERED\n")
        if alarm_enabled:
            wave_obj.play()
        last_alarm_time = now
        detections_queue.clear()

    cv2.imshow("🔥 Fire Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("\nDetection stopped.")