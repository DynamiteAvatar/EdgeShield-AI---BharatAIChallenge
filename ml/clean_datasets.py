
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import re
import unicodedata

import pandas as pd

# ============================================================
# EDGESHIELD AI - DATASET CLEANING AND VALIDATION
# Approved datasets:
# 1. ScamShield (Hugging Face)
# 2. Mendeley SMS Phishing
#
# This script does NOT train or split the data.
# ============================================================

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
OUTPUT_DIR = DATASET_DIR / "processed"

SCAMSHIELD_DIR = DATASET_DIR / "scamshield"
MENDELEY_FILE = DATASET_DIR / "mendeley_phishing.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# We retain original message text as much as possible.
# Excessive cleaning can remove useful scam indicators.


def normalize_text(value):
    """Normalize whitespace and Unicode without removing URLs,
    punctuation, OTP mentions, or numbers."""

    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def duplicate_key(text):
    """Conservative key for exact-message deduplication."""

    normalized = normalize_text(text).casefold()
    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def clean_mendeley():
    print("\n[1/5] Loading Mendeley SMS Phishing...")

    if not MENDELEY_FILE.exists():
        raise FileNotFoundError(MENDELEY_FILE)

    df = pd.read_csv(
        MENDELEY_FILE,
        encoding="utf-8-sig",
        encoding_errors="replace",
        dtype=str,
        keep_default_na=False
    )

    required = {"LABEL", "TEXT"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Mendeley columns mismatch: {df.columns.tolist()}"
        )

    labels = df["LABEL"].str.strip().str.casefold()

    label_map = {
        "ham": 0,
        "spam": 1,
        "smishing": 1
    }

    unknown = sorted(
        set(labels.unique()) - set(label_map)
    )

    if unknown:
        raise ValueError(
            f"Unknown Mendeley labels: {unknown}"
        )

    result = pd.DataFrame({
        "text": df["TEXT"].map(normalize_text),
        "label": labels.map(label_map).astype(int),
        "original_label": labels,
        "source": "mendeley_phishing",
        "original_split": "unspecified",
        "language": "unknown",
        "split_group": ""
    })

    print(f"Loaded {len(result):,} records")

    return result


def clean_scamshield():
    print("\n[2/5] Loading ScamShield...")

    frames = []

    for split in ["train", "validation", "test"]:

        path = SCAMSHIELD_DIR / f"{split}.parquet"

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_parquet(path)

        required = {
            "text",
            "is_scam",
            "language",
            "source_dataset",
            "split_group"
        }

        if not required.issubset(df.columns):
            raise ValueError(
                f"ScamShield columns mismatch: {df.columns.tolist()}"
            )

        raw_labels = df["is_scam"].map(
            lambda x: str(x).strip().casefold()
        )

        label_map = {
            "0": 0,
            "1": 1,
            "false": 0,
            "true": 1
        }

        unknown = sorted(
            set(raw_labels.unique()) - set(label_map)
        )

        if unknown:
            raise ValueError(
                f"Unknown ScamShield labels: {unknown}"
            )

        result = pd.DataFrame({
            "text": df["text"].map(normalize_text),
            "label": raw_labels.map(label_map).astype(int),
            "original_label": raw_labels,
            "source": df["source_dataset"].fillna(
                "scamshield_unknown"
            ).astype(str),
            "original_split": split,
            "language": df["language"].fillna(
                "unknown"
            ).astype(str),
            "split_group": df["split_group"].fillna(
                ""
            ).astype(str)
        })

        frames.append(result)

        print(
            f"  {split}: {len(result):,} records"
        )

    return pd.concat(
        frames, ignore_index=True
    )


def audit_original_splits(df):
    print("\n[3/5] Checking ScamShield split overlap...")

    scam = df[
        df["original_split"] != "unspecified"
    ].copy()

    key_to_splits = defaultdict(set)

    for text, split in zip(
        scam["text"], scam["original_split"]
    ):
        key_to_splits[
            duplicate_key(text)
        ].add(split)

    overlap = sum(
        len(splits) > 1
        for splits in key_to_splits.values()
    )

    print(
        "Messages appearing in multiple original splits:",
        overlap
    )

    return overlap


def remove_invalid_and_duplicates(df):
    print("\n[4/5] Cleaning and deduplicating...")

    report = {}

    report["raw_records"] = int(len(df))

    invalid = (
        df["text"].eq("")
        | df["text"].str.len().lt(3)
    )

    report["invalid_or_short"] = int(invalid.sum())

    df = df.loc[~invalid].copy()

    df["duplicate_key"] = df["text"].map(
        duplicate_key
    )

    # Identify messages assigned contradictory binary labels.
    label_counts = df.groupby(
        "duplicate_key"
    )["label"].nunique()

    conflicting_keys = set(
        label_counts[label_counts > 1].index
    )

    report["conflicting_unique_messages"] = int(
        len(conflicting_keys)
    )

    conflicting_rows = df[
        df["duplicate_key"].isin(conflicting_keys)
    ].copy()

    conflicting_rows.to_csv(
        OUTPUT_DIR / "conflicting_labels.csv",
        index=False
    )

    # Quarantine all conflicting-label records.
    df = df[
        ~df["duplicate_key"].isin(conflicting_keys)
    ].copy()

    before = len(df)

    # Prefer ScamShield original test/validation examples
    # when an identical message appears in multiple sources.
    # This is provenance retention, NOT final split assignment.
    priority = {
        "test": 0,
        "validation": 1,
        "train": 2,
        "unspecified": 3
    }

    df["_priority"] = df["original_split"].map(
        priority
    ).fillna(4)

    df = df.sort_values(
        ["_priority", "duplicate_key"],
        kind="stable"
    )

    df = df.drop_duplicates(
        subset=["duplicate_key"],
        keep="first"
    )

    report["duplicate_rows_removed"] = int(
        before - len(df)
    )

    df = df.drop(
        columns=["_priority"]
    ).reset_index(drop=True)

    report["clean_records"] = int(len(df))

    return df, report


def main():

    print("=" * 65)
    print("EDGESHIELD AI - DATASET CLEANING")
    print("=" * 65)

    mendeley = clean_mendeley()
    scamshield = clean_scamshield()

    combined = pd.concat(
        [scamshield, mendeley],
        ignore_index=True
    )

    print("\nCombined raw records:", len(combined))

    split_overlap = audit_original_splits(combined)

    clean, report = remove_invalid_and_duplicates(
        combined
    )

    report["original_split_overlap"] = int(
        split_overlap
    )

    report["class_distribution"] = {
        str(k): int(v)
        for k, v in clean["label"].value_counts().items()
    }

    report["source_distribution"] = {
        str(k): int(v)
        for k, v in clean["source"].value_counts().items()
    }

    report["language_distribution"] = {
        str(k): int(v)
        for k, v in clean["language"].value_counts().items()
    }

    output_file = OUTPUT_DIR / "cleaned_sms.parquet"

    clean.to_parquet(
        output_file,
        index=False
    )

    report_file = OUTPUT_DIR / "cleaning_report.json"

    report_file.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("CLEANING COMPLETED")
    print("=" * 65)

    print("Raw records:", report["raw_records"])
    print(
        "Invalid/short:",
        report["invalid_or_short"]
    )
    print(
        "Conflicting unique messages:",
        report["conflicting_unique_messages"]
    )
    print(
        "Duplicate rows removed:",
        report["duplicate_rows_removed"]
    )
    print("Final records:", report["clean_records"])

    print("\nClass distribution:")
    print(clean["label"].value_counts())

    print("\nLanguage distribution:")
    print(clean["language"].value_counts().head(15))

    print("\nSaved:")
    print(output_file)
    print(report_file)
    print(
        OUTPUT_DIR / "conflicting_labels.csv"
    )

    print(
        "\nNOTE: Near-duplicate/template analysis "
        "and group-aware 80:20 splitting come next."
    )


if __name__ == "__main__":
    main()
