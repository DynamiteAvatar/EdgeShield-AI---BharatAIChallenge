
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# EDGESHIELD AI - SVM PERFORMANCE VISUALIZATION
# =====================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "ml" / "results"

with open(RESULTS / "comparison_summary.json", "r", encoding="utf-8") as f:
    data = json.load(f)

baseline = data["baseline"]
tfidf = data["tfidf"]

# Extract actual test metrics
metrics = ["accuracy", "precision", "recall", "f1"]
labels = ["Accuracy", "Precision", "Recall", "F1-score"]

baseline_scores = np.array([
    baseline["test_metrics"][m] * 100 for m in metrics
])

tfidf_scores = np.array([
    tfidf["test_metrics"][m] * 100 for m in metrics
])

# Extract confusion matrices
cm_baseline = np.array(baseline["confusion_matrix"])
cm_tfidf = np.array(tfidf["confusion_matrix"])

baseline_errors = np.array([
    cm_baseline[1, 0],  # Missed scams
    cm_baseline[0, 1],  # False alarms
    cm_baseline[1, 0] + cm_baseline[0, 1]
])

tfidf_errors = np.array([
    cm_tfidf[1, 0],
    cm_tfidf[0, 1],
    cm_tfidf[1, 0] + cm_tfidf[0, 1]
])

# Colors
BLUE = "#2563EB"
GRAY = "#94A3B8"
GREEN = "#16A34A"
RED = "#DC2626"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "axes.facecolor": "white"
})

# =====================================================
# CREATE DASHBOARD
# =====================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

fig.suptitle(
    "EdgeShield AI | CountVectorizer vs TF-IDF + LinearSVC",
    fontsize=19,
    fontweight="bold"
)

# -----------------------------------------------------
# GRAPH 1: Accuracy, Precision, Recall, F1
# -----------------------------------------------------

ax = axes[0, 0]
x = np.arange(len(labels))
width = 0.35

bars1 = ax.bar(
    x - width / 2,
    baseline_scores,
    width,
    color=GRAY,
    label="CountVectorizer + SVC"
)

bars2 = ax.bar(
    x + width / 2,
    tfidf_scores,
    width,
    color=BLUE,
    label="TF-IDF + SVC"
)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(90, 102)
ax.set_ylabel("Score (%)")
ax.set_title("1. Classification Performance")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.2)

for bars in [bars1, bars2]:
    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=8)

# -----------------------------------------------------
# GRAPH 2: Improvements in percentage points
# -----------------------------------------------------

ax = axes[0, 1]

improvements = tfidf_scores - baseline_scores

bars = ax.barh(
    labels,
    improvements,
    color=GREEN
)

ax.invert_yaxis()
ax.set_title("2. TF-IDF Performance Improvement")
ax.set_xlabel("Improvement (percentage points)")
ax.set_xlim(0, max(improvements) * 1.4)
ax.grid(axis="x", alpha=0.2)

for bar, value in zip(bars, improvements):
    ax.text(
        value + 0.025,
        bar.get_y() + bar.get_height() / 2,
        f"+{value:.2f} pp",
        va="center",
        fontsize=10,
        fontweight="bold"
    )

# -----------------------------------------------------
# GRAPH 3: Missed scams and false alarms
# -----------------------------------------------------

ax = axes[1, 0]

error_labels = [
    "Missed Scams",
    "False Alarms",
    "Total Errors"
]

x = np.arange(len(error_labels))

bars1 = ax.bar(
    x - width / 2,
    baseline_errors,
    width,
    color=GRAY,
    label="CountVectorizer + SVC"
)

bars2 = ax.bar(
    x + width / 2,
    tfidf_errors,
    width,
    color=BLUE,
    label="TF-IDF + SVC"
)

ax.set_xticks(x)
ax.set_xticklabels(error_labels)
ax.set_title("3. Classification Errors (Lower is Better)")
ax.set_ylabel("Number of SMS Messages")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.2)

for bars in [bars1, bars2]:
    ax.bar_label(bars, padding=3, fontsize=9)

# -----------------------------------------------------
# GRAPH 4: Correctly classified messages
# -----------------------------------------------------

ax = axes[1, 1]

correct_baseline = np.trace(cm_baseline)
correct_tfidf = np.trace(cm_tfidf)

bars = ax.bar(
    ["CountVectorizer + SVC", "TF-IDF + SVC"],
    [correct_baseline, correct_tfidf],
    color=[GRAY, BLUE],
    width=0.5
)

ax.set_title("4. Correctly Classified SMS Messages")
ax.set_ylabel("Number of Correct Predictions")
ax.set_ylim(0, max(correct_baseline, correct_tfidf) * 1.15)
ax.grid(axis="y", alpha=0.2)

for bar in bars:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 100,
        f"{int(bar.get_height()):,}",
        ha="center",
        fontweight="bold",
        fontsize=12
    )

# =====================================================
# SAVE AND DISPLAY
# =====================================================

plt.tight_layout(rect=[0, 0.02, 1, 0.94])

save_path = RESULTS / "EdgeShield_SVM_Comparison.png"

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

print("=" * 55)
print("EDGESHIELD AI - MODEL COMPARISON")
print("=" * 55)

print(f"Baseline F1: {baseline_scores[3]:.2f}%")
print(f"TF-IDF F1:   {tfidf_scores[3]:.2f}%")
print(f"F1 Gain:     {improvements[3]:+.2f} percentage points")

print(f"\nMissed scams reduced: {baseline_errors[0] - tfidf_errors[0]}")
print(f"False alarms reduced: {baseline_errors[1] - tfidf_errors[1]}")
print(f"Total errors reduced: {baseline_errors[2] - tfidf_errors[2]}")

print(f"\nGraph saved to:\n{save_path}")

plt.show()
