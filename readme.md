# Fire & Smoke Detection using YOLOv5n + ONNX Runtime

This project implements a lightweight and Raspberry-Pi-compatible fire & smoke
detection system using the YOLOv5n model, trained on an indoor fire-smoke dataset.

## 🔥 Features

- Real-time fire & smoke detection
- Uses ONNX Runtime (Pi-compatible)
- Mobile camera streaming support (IP Webcam)
- Lightweight (1 FPS processing)
- Alarm trigger when 3 consecutive frames detect fire > 60% confidence
- Fully runs on Windows, Mac, and Raspberry Pi

## 🏋️ Training

Training was performed on Windows using a YOLOv5n model.
Scripts are in `training/`.

## 🤖 ONNX Inference

Runtime detection script is located at:

src/detect_fire.py

Runs on Mac and Raspberry Pi using ONNX Runtime.

## 📱 Mobile Camera Streaming

Install IP Webcam (Android) or IP Camera Lite (iPhone) and use the stream URL:

http://<your-ip>:8080/video

## 🚨 Alarm Trigger Logic

- Model processes 1 frame per second
- Alarm triggers after 3 consecutive detections with >= 60% confidence

## 🖼 Sample Outputs

(Add screenshots here)

## 📝 License

MIT
