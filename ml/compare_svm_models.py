
from pathlib import Path
import gc
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ==========================================================
# EDGESHIELD AI - SVM MODEL COMPARISON
# Uses the same frozen 80:20 dataset split for both models.
# No data cleaning, resplitting or test-set tuning.
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "processed"
OUTPUT = ROOT / "ml" / "results"

OUTPUT.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA / "train_80.parquet"
TEST_PATH = DATA / "test_20.parquet"
AUDIT_PATH = DATA / "near_duplicate_audit.json"

RANDOM_SEED = 42

# Moderate feature limits for an 8 GB RAM laptop.
WORD_FEATURES = 60000
CHAR_FEATURES = 100000

# Hyperparameter search uses training data only.
C_VALUES = [0.1, 1.0, 3.0]

# Group-aware internal validation.
VALIDATION_SIZE = 0.20


def load_data():
    print("\n[1] Loading frozen datasets...")

    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    required = {
        "text", "label", "language", "leakage_group"
    }

    for name, df in [("train", train), ("test", test)]:
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} missing columns: {missing}"
            )

        if df["text"].isna().any():
            raise ValueError(f"Null text in {name}")

        if df["label"].isna().any():
            raise ValueError(f"Null labels in {name}")

        if not set(df["label"].unique()).issubset({0, 1}):
            raise ValueError(f"Invalid labels in {name}")

    if set(train["leakage_group"]) & set(
        test["leakage_group"]
    ):
        raise RuntimeError(
            "Train/test leakage_group overlap detected"
        )

    print(f"Training: {len(train):,}")
    print(f"Testing:  {len(test):,}")

    return train, test


def build_pipeline(model_name, c_value):
    classifier = LinearSVC(
        C=c_value,
        class_weight="balanced",
        max_iter=3000,
        tol=1e-3,
        random_state=RANDOM_SEED
    )

    if model_name == "baseline":
        features = CountVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=WORD_FEATURES,
            lowercase=True,
            dtype=np.float32
        )

    elif model_name == "tfidf":
        word_features = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=WORD_FEATURES,
            sublinear_tf=True,
            dtype=np.float32
        )

        char_features = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=3,
            max_features=CHAR_FEATURES,
            sublinear_tf=True,
            dtype=np.float32
        )

        features = FeatureUnion([
            ("word_tfidf", word_features),
            ("char_tfidf", char_features)
        ])

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([
        ("features", features),
        ("classifier", classifier)
    ])


def metrics(y_true, y_pred):
    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0
        )
    )

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


def tune_model(model_name, train, fit_idx, val_idx):
    print(f"\n[2] Tuning {model_name.upper()}...")

    x_fit = train.iloc[fit_idx]["text"].tolist()
    y_fit = train.iloc[fit_idx]["label"].to_numpy()

    x_val = train.iloc[val_idx]["text"].tolist()
    y_val = train.iloc[val_idx]["label"].to_numpy()

    candidates = []
    best_c = None
    best_f1 = -1.0

    for c_value in C_VALUES:
        print(f"\nTesting C={c_value}", flush=True)

        model = build_pipeline(
            model_name, c_value
        )

        started = time.perf_counter()

        model.fit(x_fit, y_fit)

        predictions = model.predict(x_val)

        result = metrics(y_val, predictions)
        elapsed = time.perf_counter() - started

        print(
            f"F1={result['f1']:.4f} | "
            f"Precision={result['precision']:.4f} | "
            f"Recall={result['recall']:.4f} | "
            f"Time={elapsed:.1f}s",
            flush=True
        )

        candidates.append({
            "C": c_value,
            "validation_metrics": result,
            "seconds": round(elapsed, 2)
        })

        # Select only using training-derived validation F1.
        if result["f1"] > best_f1:
            best_f1 = result["f1"]
            best_c = c_value

        del model
        gc.collect()

    print(f"\nSelected C={best_c}")

    return best_c, candidates


def language_metrics(test, predictions):
    output = {}

    languages = (
        test["language"]
        .fillna("unknown")
        .astype(str)
    )

    for language in sorted(languages.unique()):
        mask = (
            languages.to_numpy() == language
        )

        y_true = test.loc[mask, "label"].to_numpy()
        y_pred = predictions[mask]

        output[language] = {
            "records": int(mask.sum()),
            "positive_records": int(y_true.sum()),
            **metrics(y_true, y_pred)
        }

    return output


def sensitivity_metrics(test, predictions):
    if not AUDIT_PATH.exists():
        return {
            "status": "audit file not found"
        }

    audit = json.loads(
        AUDIT_PATH.read_text(encoding="utf-8")
    )

    if (
        audit.get("test_records") != len(test)
        or audit.get("train_records") != 68652
    ):
        return {
            "status": (
                "audit record counts do not match "
                "the expected frozen split"
            )
        }

    flagged = {
        int(match["test_row"])
        for match in audit.get("matches", [])
    }

    if any(
        i < 0 or i >= len(test)
        for i in flagged
    ):
        return {
            "status": "invalid audit row indices"
        }

    mask = np.array([
        i not in flagged
        for i in range(len(test))
    ])

    if not mask.any():
        return {
            "status": "no unflagged messages"
        }

    y_true = test["label"].to_numpy()

    return {
        "status": "completed",
        "excluded_flagged_records": len(flagged),
        "remaining_records": int(mask.sum()),
        "metrics": metrics(
            y_true[mask],
            predictions[mask]
        ),
        "note": (
            "Secondary sensitivity analysis only. "
            "Not a replacement for the full test set."
        )
    }


def save_confusion_matrix(
    y_true, predictions, model_name
):
    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Ham", "Scam"]
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d"
    )

    ax.set_title(
        f"EdgeShield AI - {model_name.upper()}"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT / f"{model_name}_confusion_matrix.png",
        dpi=180
    )

    plt.close(fig)

    return cm.tolist()


def train_and_evaluate(
    model_name,
    train,
    test,
    fit_idx,
    val_idx
):
    best_c, tuning = tune_model(
        model_name, train, fit_idx, val_idx
    )

    print(
        f"\n[3] Training final {model_name.upper()} "
        "on all training records...",
        flush=True
    )

    model = build_pipeline(
        model_name, best_c
    )

    x_train = train["text"].tolist()
    y_train = train["label"].to_numpy()

    started = time.perf_counter()
    model.fit(x_train, y_train)

    training_seconds = (
        time.perf_counter() - started
    )

    print(
        f"Training completed in "
        f"{training_seconds:.1f}s"
    )

    print("\n[4] Evaluating frozen test set...")

    x_test = test["text"].tolist()
    y_test = test["label"].to_numpy()

    started = time.perf_counter()

    predictions = model.predict(x_test)

    prediction_seconds = (
        time.perf_counter() - started
    )

    overall = metrics(y_test, predictions)

    cm = save_confusion_matrix(
        y_test,
        predictions,
        model_name
    )

    report_text = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Ham", "Scam"],
        digits=4,
        zero_division=0
    )

    (OUTPUT / f"{model_name}_classification.txt").write_text(
        report_text,
        encoding="utf-8"
    )

    model_path = OUTPUT / f"{model_name}_svm.joblib"

    joblib.dump(
        model,
        model_path,
        compress=3
    )

    model_size_mb = (
        model_path.stat().st_size / (1024 ** 2)
    )

    result = {
        "model": model_name,
        "best_C": best_c,
        "tuning": tuning,
        "test_metrics": overall,
        "confusion_matrix": cm,
        "language_metrics": language_metrics(
            test, predictions
        ),
        "sensitivity_analysis": sensitivity_metrics(
            test, predictions
        ),
        "training_seconds": round(
            training_seconds, 2
        ),
        "test_prediction_seconds": round(
            prediction_seconds, 2
        ),
        "average_prediction_ms": round(
            prediction_seconds / len(test) * 1000,
            4
        ),
        "model_size_mb": round(
            model_size_mb, 2
        )
    }

    (OUTPUT / f"{model_name}_results.json").write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 55)
    print(f"{model_name.upper()} TEST RESULTS")
    print("=" * 55)

    print(report_text)

    print("Confusion matrix:")
    print(np.asarray(cm))

    print(f"\nModel size: {model_size_mb:.2f} MB")
    print(
        "Average batch prediction time per SMS:",
        result["average_prediction_ms"],
        "ms"
    )

    del model
    gc.collect()

    return result


def main():
    print("=" * 65)
    print("EDGESHIELD AI - SVM MODEL COMPARISON")
    print("=" * 65)

    train, test = load_data()

    # One reproducible group-aware validation partition
    # entirely within the training data.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED
    )

    fit_idx, val_idx = next(
        splitter.split(
            train,
            train["label"],
            train["leakage_group"]
        )
    )

    print("\nInternal training:", len(fit_idx))
    print("Internal validation:", len(val_idx))

    results = {}

    for name in ["baseline", "tfidf"]:
        results[name] = train_and_evaluate(
            name,
            train,
            test,
            fit_idx,
            val_idx
        )

        gc.collect()

    baseline = results["baseline"]["test_metrics"]
    enhanced = results["tfidf"]["test_metrics"]

    print("\n" + "=" * 65)
    print("FINAL SVM COMPARISON")
    print("=" * 65)

    print(
        f"{'Metric':<14}"
        f"{'Baseline':>12}"
        f"{'TF-IDF':>12}"
        f"{'Difference':>14}"
    )

    for key in [
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]:
        a = baseline[key]
        b = enhanced[key]

        print(
            f"{key:<14}"
            f"{a:>12.4f}"
            f"{b:>12.4f}"
            f"{b-a:>+14.4f}"
        )

    summary = {
        "baseline": results["baseline"],
        "tfidf": results["tfidf"],
        "f1_difference": (
            enhanced["f1"] - baseline["f1"]
        )
    }

    (OUTPUT / "comparison_summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\nResults saved in:")
    print(OUTPUT)

    print(
        "\nNOTE: These are held-out results on the "
        "frozen dataset split. The near-duplicate "
        "audit identified residual leakage risk."
    )


if __name__ == "__main__":
    main()
