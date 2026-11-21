
import gzip
import json
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ============================================================
# EDGESHIELD AI - ANDROID SVM EXPORT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "ml" / "results" / "tfidf_svm.joblib"
)

TEST_PATH = (
    ROOT / "dataset" / "processed" / "test_20.parquet"
)

EXPORT_DIR = ROOT / "ml" / "android_export"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

EXPORT_PATH = EXPORT_DIR / "edgeshield_svm.json.gz"
REFERENCE_PATH = EXPORT_DIR / "parity_reference.json"
REPORT_PATH = EXPORT_DIR / "export_report.json"


def to_float_list(values):
    return np.asarray(
        values, dtype=np.float64
    ).tolist()


def export_vectorizer(vectorizer, weights):
    params = vectorizer.get_params()

    # Only support the exact configurations we trained.
    if params["analyzer"] not in ("word", "char"):
        raise ValueError("Unsupported analyzer")

    if params["norm"] != "l2":
        raise ValueError("Expected L2 normalization")

    if params["use_idf"] is not True:
        raise ValueError("Expected IDF weighting")

    if params["smooth_idf"] is not True:
        raise ValueError("Expected smooth IDF")

    if params["binary"] is not False:
        raise ValueError("Binary TF is not supported")

    if params["strip_accents"] is not None:
        raise ValueError("Unexpected accent normalization")

    if params["preprocessor"] is not None:
        raise ValueError("Custom preprocessor unsupported")

    if params["tokenizer"] is not None:
        raise ValueError("Custom tokenizer unsupported")

    if params["stop_words"] is not None:
        raise ValueError("Stop words unsupported")

    if params["lowercase"] is not True:
        raise ValueError("Expected lowercase processing")

    if params["input"] != "content":
        raise ValueError("Expected content input")

    if params["encoding"] != "utf-8":
        raise ValueError("Unexpected encoding")

    if params["analyzer"] == "word":
        if params["token_pattern"] != r"(?u)\b\w\w+\b":
            raise ValueError("Unexpected word token pattern")

    # Store vocabulary by feature index, not alphabetic order.
    vocabulary = vectorizer.vocabulary_

    ordered_vocabulary = [
        None
    ] * len(vocabulary)

    for token, index in vocabulary.items():
        ordered_vocabulary[index] = token

    if any(x is None for x in ordered_vocabulary):
        raise ValueError("Incomplete vocabulary")

    idf = np.asarray(
        vectorizer.idf_,
        dtype=np.float64
    )

    weights = np.asarray(
        weights,
        dtype=np.float64
    )

    if not (
        len(ordered_vocabulary)
        == len(idf)
        == len(weights)
    ):
        raise ValueError("Feature dimension mismatch")

    return {
        "analyzer": params["analyzer"],
        "ngram_range": list(params["ngram_range"]),
        "lowercase": params["lowercase"],
        "token_pattern": (
            params["token_pattern"]
            if params["analyzer"] == "word"
            else None
        ),
        "sublinear_tf": params["sublinear_tf"],
        "norm": params["norm"],
        "vocabulary": ordered_vocabulary,
        "idf": to_float_list(idf),
        "weights": to_float_list(weights)
    }


def main():
    print("=" * 65)
    print("EDGESHIELD AI - ANDROID MODEL EXPORT")
    print("=" * 65)

    print("\n[1/4] Loading trained model...")

    model = joblib.load(MODEL_PATH)

    features = model.named_steps["features"]
    classifier = model.named_steps["classifier"]

    transformers = dict(
        features.transformer_list
    )

    word = transformers["word_tfidf"]
    char = transformers["char_tfidf"]

    word_count = len(word.vocabulary_)
    char_count = len(char.vocabulary_)

    coef = np.asarray(
        classifier.coef_[0],
        dtype=np.float64
    )

    if len(coef) != word_count + char_count:
        raise ValueError(
            "SVM coefficients do not match feature sizes"
        )

    if not np.array_equal(
        classifier.classes_,
        [0, 1]
    ):
        raise ValueError(
            "Unexpected class order"
        )

    print("Word features:", word_count)
    print("Character features:", char_count)

    print("\n[2/4] Exporting TF-IDF and SVM parameters...")

    payload = {
        "format": "edgeshield-linear-svm-v1",
        "classes": [0, 1],
        "positive_class": 1,
        "decision_rule": (
            "score > 0 means scam; "
            "score <= 0 means ham"
        ),
        "intercept": float(
            classifier.intercept_[0]
        ),
        "feature_union": {
            "transformer_weights": None,
            "word": export_vectorizer(
                word,
                coef[:word_count]
            ),
            "char": export_vectorizer(
                char,
                coef[word_count:]
            )
        }
    }

    if features.transformer_weights is not None:
        raise ValueError(
            "Non-default FeatureUnion weights"
        )

    with gzip.open(
        EXPORT_PATH,
        "wt",
        encoding="utf-8",
        compresslevel=6
    ) as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            separators=(",", ":")
        )

    print("\n[3/4] Creating Python parity references...")

    test = pd.read_parquet(TEST_PATH)

    # Include examples from each language and class.
    # Fixed seed ensures reproducible reference cases.
    samples = []

    for language in sorted(
        test["language"].dropna().unique()
    ):
        for label in [0, 1]:
            subset = test[
                (test["language"] == language)
                & (test["label"] == label)
            ]

            if len(subset) == 0:
                continue

            chosen = subset.sample(
                n=min(15, len(subset)),
                random_state=42
            )

            samples.append(chosen)

    if not samples:
        raise RuntimeError(
            "No parity samples available"
        )

    reference_df = pd.concat(
        samples,
        ignore_index=True
    )

    texts = reference_df["text"].astype(str).tolist()

    scores = model.decision_function(texts)
    predictions = model.predict(texts)

    references = []

    for i, text in enumerate(texts):
        references.append({
            "id": i,
            "text": text,
            "language": str(
                reference_df.iloc[i]["language"]
            ),
            "actual_label": int(
                reference_df.iloc[i]["label"]
            ),
            "python_prediction": int(
                predictions[i]
            ),
            "python_decision_score": float(
                scores[i]
            )
        })

    REFERENCE_PATH.write_text(
        json.dumps(
            references,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("Reference messages:", len(references))

    print("\n[4/4] Verifying exported feature weights...")

    # Reconstruct scores from the exported feature weights.
    # This validates feature ordering and coefficient slicing.
    x_word = word.transform(texts)
    x_char = char.transform(texts)

    reconstructed = (
        np.asarray(
            x_word @ coef[:word_count]
        ).ravel()
        +
        np.asarray(
            x_char @ coef[word_count:]
        ).ravel()
        +
        float(classifier.intercept_[0])
    )

    max_difference = float(
        np.max(
            np.abs(reconstructed - scores)
        )
    )

    if not np.allclose(
        reconstructed,
        scores,
        rtol=1e-6,
        atol=1e-6
    ):
        raise RuntimeError(
            "Export parity verification failed"
        )

    digest = hashlib.sha256(
        EXPORT_PATH.read_bytes()
    ).hexdigest()

    report = {
        "format": "edgeshield-linear-svm-v1",
        "word_features": word_count,
        "character_features": char_count,
        "total_features": len(coef),
        "reference_messages": len(references),
        "max_score_difference": max_difference,
        "export_size_mb": round(
            EXPORT_PATH.stat().st_size / 1024**2,
            3
        ),
        "sha256": digest,
        "status": "python_export_verified"
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("EXPORT COMPLETED")
    print("=" * 65)

    for key, value in report.items():
        print(f"{key}: {value}")

    print("\nExported files:")
    print(EXPORT_PATH)
    print(REFERENCE_PATH)
    print(REPORT_PATH)

    print(
        "\nNext: Verify the same decision scores "
        "using an independent Java implementation."
    )


if __name__ == "__main__":
    main()
