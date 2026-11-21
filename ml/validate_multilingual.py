
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "ml" / "results"

with open(RESULTS / "baseline_results.json", encoding="utf-8") as f:
    baseline = json.load(f)

with open(RESULTS / "tfidf_results.json", encoding="utf-8") as f:
    tfidf = json.load(f)

print("=" * 65)
print("EDGESHIELD AI - MULTILINGUAL MODEL VALIDATION")
print("=" * 65)

languages = ["English", "Hindi", "Hinglish"]

# Support case variations in saved language names.
def get_language_result(model, language):
    for name, values in model["language_metrics"].items():
        if name.casefold() == language.casefold():
            return values
    return None


print("\nLANGUAGE-WISE PERFORMANCE")
print("-" * 65)

for language in languages:
    a = get_language_result(baseline, language)
    b = get_language_result(tfidf, language)

    if a is None or b is None:
        print(f"\n{language}: No metrics found")
        continue

    print(f"\n{language}")
    print(f"Test messages: {b['records']}")
    print(f"Scam messages: {b['positive_records']}")

    for metric in ["accuracy", "precision", "recall", "f1"]:
        av = a[metric] * 100
        bv = b[metric] * 100

        print(
            f"{metric.capitalize():<12} "
            f"Baseline: {av:6.2f}% | "
            f"TF-IDF: {bv:6.2f}% | "
            f"Difference: {bv-av:+.2f} pp"
        )

# ======================================================
# GRAPH: Language-wise F1 and recall
# ======================================================

valid_languages = [
    lang for lang in languages
    if get_language_result(baseline, lang)
    and get_language_result(tfidf, lang)
]

if valid_languages:
    x = np.arange(len(valid_languages))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title in [
        (axes[0], "f1", "Scam F1-score by Language"),
        (axes[1], "recall", "Scam Recall by Language")
    ]:
        a = [
            get_language_result(baseline, lang)[metric] * 100
            for lang in valid_languages
        ]

        b = [
            get_language_result(tfidf, lang)[metric] * 100
            for lang in valid_languages
        ]

        bars_a = ax.bar(
            x - width / 2, a, width,
            label="CountVectorizer + SVC",
            color="#94A3B8"
        )

        bars_b = ax.bar(
            x + width / 2, b, width,
            label="TF-IDF + SVC",
            color="#2563EB"
        )

        ax.set_xticks(x, valid_languages)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Score (%)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)

        for bars in [bars_a, bars_b]:
            ax.bar_label(
                bars, fmt="%.1f%%",
                padding=3, fontsize=8
            )

    fig.suptitle(
        "EdgeShield AI - Multilingual SVM Comparison",
        fontsize=16,
        fontweight="bold"
    )

    fig.tight_layout()

    save_path = RESULTS / "multilingual_comparison.png"

    fig.savefig(save_path, dpi=300)
    print(f"\nMultilingual graph saved: {save_path}")

    plt.show()

# ======================================================
# NEAR-DUPLICATE SENSITIVITY ANALYSIS
# ======================================================

print("\n" + "=" * 65)
print("NEAR-DUPLICATE SENSITIVITY ANALYSIS")
print("=" * 65)

for name, model in [
    ("Baseline", baseline),
    ("TF-IDF", tfidf)
]:
    sensitivity = model.get(
        "sensitivity_analysis", {}
    )

    print(f"\n{name}")
    print("Status:", sensitivity.get("status", "unknown"))

    if sensitivity.get("status") == "completed":
        print(
            "Excluded flagged messages:",
            sensitivity["excluded_flagged_records"]
        )

        print(
            "Remaining test messages:",
            sensitivity["remaining_records"]
        )

        for metric, value in sensitivity["metrics"].items():
            print(f"{metric}: {value * 100:.2f}%")

print("\nValidation complete.")
