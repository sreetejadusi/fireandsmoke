import os
import subprocess
from datetime import datetime


PROJECT_ROOT = os.getcwd()
YOLO_DIR = os.path.join(PROJECT_ROOT, "yolov5")
DATA_YAML = os.path.join(PROJECT_ROOT, "data.yaml")

RUN_NAME = f"fire_yolov5n_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)


LOG_FILE = os.path.join(LOGS_DIR, f"{RUN_NAME}.log")

EPOCHS = 15
BATCH_SIZE = 8
IMAGE_SIZE = 416
DEVICE = "cpu"
CACHE = True

print(f"\nStarting training: {RUN_NAME}")
print(f"Logs will be saved in: {LOG_FILE}\n")


def run_with_log(cmd):
    with open(LOG_FILE, "a", buffering=1) as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

        for line in process.stdout:
            print(line, end="")
            f.write(line)

        process.stdout.close()
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)


train_cmd = [
    "python",
    os.path.join(YOLO_DIR, "train.py"),
    "--img",
    str(IMAGE_SIZE),
    "--batch",
    str(BATCH_SIZE),
    "--epochs",
    str(EPOCHS),
    "--data",
    DATA_YAML,
    "--weights",
    "yolov5n.pt",
    "--device",
    DEVICE,
    "--name",
    RUN_NAME,
    "--cache",
]

print("Running training command...\n")
run_with_log(train_cmd)


print("\nExporting model to ONNX and TFLite formats...\n")

weights_path = os.path.join(YOLO_DIR, "runs", "train", RUN_NAME, "weights", "best.pt")
export_cmd = [
    "python",
    os.path.join(YOLO_DIR, "export.py"),
    "--weights",
    weights_path,
    "--include",
    "onnx",
    "tflite",
]

run_with_log(export_cmd)


print("\nTraining & export complete!")
print(f"Best weights: {weights_path}")
print(f"Exported model: {os.path.dirname(weights_path)}")
print(f"Full log saved at: {LOG_FILE}")
