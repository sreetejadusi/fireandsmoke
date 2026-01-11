import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

HYBRID_LOG = "logs/detection_log_video.csv"
YOLO_LOG = "logs/detection_log_yolo_only.csv"

df_hybrid = pd.read_csv(HYBRID_LOG)
df_yolo = pd.read_csv(YOLO_LOG)

# Ensure numeric safety
for col in ["yolo_conf", "flicker", "growth", "final_score"]:
    if col in df_hybrid.columns:
        df_hybrid[col] = pd.to_numeric(df_hybrid[col], errors="coerce")

df_yolo["max_confidence"] = pd.to_numeric(df_yolo["max_confidence"], errors="coerce")

# ======================================================
# 1. Hybrid Final Score vs Frame Index
# ======================================================
plt.figure(figsize=(12, 4))
plt.plot(df_hybrid["frame_idx"], df_hybrid["final_score"], label="Hybrid Final Score")
plt.axhline(0.6, linestyle="--", color="red", label="Alarm Threshold")
plt.xlabel("Frame Index")
plt.ylabel("Score")
plt.title("Temporal Evolution of Hybrid Detection Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ======================================================
# 2. YOLO Confidence vs Hybrid Score
# ======================================================
plt.figure(figsize=(12, 4))
plt.plot(
    df_hybrid["frame_idx"], df_hybrid["yolo_conf"], label="YOLO Confidence", alpha=0.6
)
plt.plot(
    df_hybrid["frame_idx"], df_hybrid["final_score"], label="Hybrid Score", linewidth=2
)
plt.axhline(0.6, linestyle="--", color="red")
plt.xlabel("Frame Index")
plt.ylabel("Score")
plt.title("YOLO Confidence vs Hybrid Fused Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ======================================================
# 3. Flicker & Growth (Hybrid only)
# ======================================================
plt.figure(figsize=(12, 4))
plt.plot(df_hybrid["frame_idx"], df_hybrid["flicker"], label="Flicker Score")
plt.plot(df_hybrid["frame_idx"], df_hybrid["growth"], label="Growth Score")
plt.xlabel("Frame Index")
plt.ylabel("Score")
plt.title("Temporal Fire Characteristics (Hybrid System)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ======================================================
# 4. Alarm Timeline Comparison
# ======================================================
hybrid_alarms = df_hybrid[df_hybrid["action"] == "ALARM"]["frame_idx"]
yolo_alarms = df_yolo[df_yolo["alarm_triggered"] == 1]["frame_idx"]

plt.figure(figsize=(12, 2))
plt.scatter(hybrid_alarms, np.ones(len(hybrid_alarms)), label="Hybrid", color="green")
plt.scatter(
    yolo_alarms, np.ones(len(yolo_alarms)) * 1.1, label="YOLO-only", color="orange"
)
plt.yticks([])
plt.xlabel("Frame Index")
plt.title("Alarm Trigger Events Over Time")
plt.legend()
plt.grid(True, axis="x")
plt.tight_layout()
plt.show()

# ======================================================
# 5. False Alarm Count
# ======================================================
plt.figure(figsize=(6, 4))
plt.bar(
    ["YOLO-only", "Hybrid"],
    [len(yolo_alarms), len(hybrid_alarms)],
    color=["orange", "green"],
)
plt.ylabel("Number of Alarm Events")
plt.title("False Alarm Comparison")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# ======================================================
# 6. Detection Latency (first alarm)
# ======================================================
latency_yolo = yolo_alarms.iloc[0] if len(yolo_alarms) > 0 else np.nan
latency_hybrid = hybrid_alarms.iloc[0] if len(hybrid_alarms) > 0 else np.nan

plt.figure(figsize=(6, 4))
plt.bar(
    ["YOLO-only", "Hybrid"], [latency_yolo, latency_hybrid], color=["orange", "green"]
)
plt.ylabel("Frame Index of First Alarm")
plt.title("Detection Latency Comparison")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# ======================================================
# 7. Score Distribution
# ======================================================
plt.figure(figsize=(10, 4))
plt.hist(df_yolo["max_confidence"], bins=50, alpha=0.6, label="YOLO Confidence")
plt.hist(df_hybrid["final_score"], bins=50, alpha=0.6, label="Hybrid Final Score")
plt.axvline(0.6, linestyle="--", color="red", label="Threshold")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.title("Score Distribution Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("All research plots generated successfully.")
