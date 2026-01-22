import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# ==========================================
# 🛠️ SETUP & DATA LOADING
# ==========================================
# Load your log file
csv_path = "plots/advanced_fire_analysis.csv"
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: Could not find {csv_path}. Make sure the file exists.")
    exit()

# Filter out "None" objects (empty frames) for the main analysis
df_clean = df[df["ID"] != "None"].copy()

# Convert columns to numeric (just in case)
cols_to_numeric = ["YOLO_Conf", "Flicker_Score", "Growth_Score", "FINAL_SCORE"]
for col in cols_to_numeric:
    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

# ✅ THE FIX: Apply Rolling Mean (Smoothing)
# This removes the "jitter" and shows the actual trend
WINDOW_SIZE = 30  # Smooth over 30 frames (approx 1 second of data)
df_smooth = df_clean.copy()
df_smooth[cols_to_numeric] = (
    df_clean[cols_to_numeric].rolling(window=WINDOW_SIZE, min_periods=1).mean()
)

# Set Academic Style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.5)

# ==========================================
# 📊 CHART 1: The "Clean" Temporal Analysis
# ==========================================
plt.figure(figsize=(12, 6))

# Plot smoothed lines
plt.plot(
    df_smooth["Frame"],
    df_smooth["YOLO_Conf"],
    label="Visual Confidence (YOLO)",
    color="#1f77b4",
    linewidth=2,
    linestyle="--",
    alpha=0.7,
)
plt.plot(
    df_smooth["Frame"],
    df_smooth["Flicker_Score"],
    label="Flicker (Temporal)",
    color="#ff7f0e",
    linewidth=2,
    alpha=0.7,
)
plt.plot(
    df_smooth["Frame"],
    df_smooth["Growth_Score"],
    label="Growth (Trend)",
    color="#2ca02c",
    linewidth=2,
    alpha=0.7,
)

# Highlight the Final Fusion Score
plt.plot(
    df_smooth["Frame"],
    df_smooth["FINAL_SCORE"],
    label="Weighted Fusion Score",
    color="#d62728",
    linewidth=3,
)

# Add Threshold Line
plt.axhline(
    y=0.65, color="black", linestyle=":", linewidth=2, label="Alarm Threshold (0.65)"
)

plt.title(
    "Fig 1: Temporal Evolution of Fire Features (Smoothed)",
    fontsize=16,
    fontweight="bold",
)
plt.xlabel("Frame Number")
plt.ylabel("Normalized Score (0-1)")
plt.legend(loc="upper right", frameon=True)
plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig("plots/Fig1_Temporal_Analysis_Clean.png", dpi=300)
print("✅ Generated Fig 1 (Clean Temporal Analysis)")

# ==========================================
# 📊 CHART 2: Stacked Area (Feature Contribution)
# ==========================================
# This shows HOW the score is built. Is it mostly YOLO? Or mostly Flicker?
plt.figure(figsize=(12, 6))

# We weight the smoothed scores to match your formula: 0.4, 0.4, 0.2
y1 = df_smooth["YOLO_Conf"] * 0.4
y2 = df_smooth["Flicker_Score"] * 0.4
y3 = df_smooth["Growth_Score"] * 0.2

plt.stackplot(
    df_smooth["Frame"],
    y1,
    y2,
    y3,
    labels=[
        "YOLO Contribution (40%)",
        "Flicker Contribution (40%)",
        "Growth Contribution (20%)",
    ],
    colors=["#a6cee3", "#fdbf6f", "#b2df8a"],
    alpha=0.8,
)

plt.plot(
    df_smooth["Frame"],
    df_smooth["FINAL_SCORE"],
    color="black",
    linestyle="--",
    linewidth=2,
    label="Total Fusion Score",
)
plt.axhline(y=0.65, color="red", linestyle="-", linewidth=2, label="Alarm Threshold")

plt.title(
    "Fig 2: Contribution of Features to Final Safety Score",
    fontsize=16,
    fontweight="bold",
)
plt.xlabel("Frame Number")
plt.ylabel("Weighted Score Contribution")
plt.legend(loc="upper left", frameon=True)
plt.tight_layout()
plt.savefig("plots/Fig2_Feature_Contribution.png", dpi=300)
print("✅ Generated Fig 2 (Stacked Area Contribution)")

# ==========================================
# 📊 CHART 3: Alarm Response Timeline (Gantt Style)
# ==========================================
# This maps the Fusion Score directly to the "Alarm ON/OFF" state
plt.figure(figsize=(12, 5))

# Create a binary alarm signal (0 or 1) based on raw data
df_clean["Alarm_Binary"] = df_clean["FINAL_SCORE"].apply(lambda x: 1 if x > 0.65 else 0)

# Plot Score
ax1 = plt.gca()
ax1.plot(
    df_clean["Frame"],
    df_clean["FINAL_SCORE"],
    color="#333333",
    label="Fusion Score",
    linewidth=1.5,
)
ax1.axhline(y=0.65, color="gray", linestyle="--", alpha=0.5)
ax1.set_ylabel("Fusion Score")
ax1.set_ylim(0, 1.1)

# Overlay Alarm Regions
# We fill the area where Alarm is ACTIVE
ax1.fill_between(
    df_clean["Frame"],
    0,
    1.1,
    where=(df_clean["Alarm_Binary"] == 1),
    color="#ff9999",
    alpha=0.5,
    label="Alarm Triggered State",
)

plt.title(
    "Fig 3: System Response & Alarm Activation Regions", fontsize=16, fontweight="bold"
)
plt.xlabel("Frame Number")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("plots/Fig3_Alarm_Response.png", dpi=300)
print("✅ Generated Fig 3 (Alarm Response Timeline)")

# ==========================================
# 📊 CHART 4: Feature Distribution (Violin Plot)
# ==========================================
# Shows the statistical spread of your features
plt.figure(figsize=(10, 6))

# Melt data for Seaborn
df_melt = df_clean.melt(
    id_vars=["Frame"],
    value_vars=["YOLO_Conf", "Flicker_Score", "Growth_Score", "FINAL_SCORE"],
    var_name="Feature",
    value_name="Score",
)

sns.violinplot(x="Feature", y="Score", data=df_melt, palette="muted", inner="quartile")
plt.title(
    "Fig 4: Statistical Distribution of Feature Scores", fontsize=16, fontweight="bold"
)
plt.ylabel("Score Density")
plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig("plots/Fig4_Feature_Distribution.png", dpi=300)
print("✅ Generated Fig 4 (Violin Distribution Plot)")

plt.show()
