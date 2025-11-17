import cv2
import numpy as np
import onnxruntime as ort
import simpleaudio as sa
import time
from collections import deque


MODEL_PATH = "model/best.onnx"
CAMERA_URL = "http://192.168.1.11:8080/video"
IMG_SIZE = 416
CONF_THRESHOLD = 0.60
REQUIRED_STREAK = 10
FRAME_SKIP = 20
detections_queue = deque(maxlen=REQUIRED_STREAK)


print("Loading ONNX model...")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print("ONNX model loaded successfully.\n")


try:
    wave_obj = sa.WaveObject.from_wave_file("alarm.wav")
    alarm_enabled = True
    print("Alarm loaded.")
except:
    alarm_enabled = False
    print("alarm.wav not found — alarm disabled.")


def trigger_alarm():
    if alarm_enabled:
        print("\nFIRE DETECTED — ALARM TRIGGERED \n")
        wave_obj.play()


cap = cv2.VideoCapture(CAMERA_URL)

if not cap.isOpened():
    print("Could not connect to mobile camera stream.")
    exit()

print("📷 Mobile camera connected.")
print("🔥 Starting ONNX Fire & Smoke Detection...\n")

frame_counter = 0
last_alarm_time = 0
cooldown = 10


while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame received from camera.")
        break

    frame_counter += 1

    if frame_counter % FRAME_SKIP != 0:
        continue

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
        print(f"Probability {max_conf * 100:.1f}% ")
    else:
        detections_queue.append(0.0)
        print("No fire detected")

    now = time.time()

    if (
        len(detections_queue) == REQUIRED_STREAK
        and all(c >= CONF_THRESHOLD for c in detections_queue)
        and now - last_alarm_time > cooldown
    ):
        trigger_alarm()
        last_alarm_time = now
        detections_queue.clear()

    cv2.imshow("fireandsmoke", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
print("Detection stopped.")
