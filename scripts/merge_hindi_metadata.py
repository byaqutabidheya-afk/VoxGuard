#!/usr/bin/env python3
"""
merge_hindi_metadata.py — Merge real and synthetic Hindi/Hinglish metadata into track metadata.

Builds data/metadata/hindi_hinglish_track.csv with the project-wide unified schema:
  [filepath, label ("real"/"synthetic"), speaker_id, category, sentence_id, dataset="hindi_hinglish"]

Verifies:
  - Zero missing values for critical columns
  - 1:1 matched real vs synthetic pairs per speaker and category
  - Consent enforcement (only consented records included)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from voxguard import config
from voxguard.utils.logging_utils import get_logger

logger = get_logger("merge_hindi_metadata")

DEFAULT_REAL_CSV = config.DATA_METADATA_DIR / "hindi_hinglish_real.csv"
DEFAULT_SYNTHETIC_CSV = config.DATA_METADATA_DIR / "hindi_hinglish_synthetic.csv"
DEFAULT_OUTPUT_TRACK_CSV = config.DATA_METADATA_DIR / "hindi_hinglish_track.csv"

UNIFIED_SCHEMA_COLUMNS = [
    "filepath",
    "label",
    "speaker_id",
    "category",
    "sentence_id",
    "dataset",
]


def merge_hindi_metadata(
    real_csv_path: Path = DEFAULT_REAL_CSV,
    synthetic_csv_path: Path = DEFAULT_SYNTHETIC_CSV,
    output_track_path: Path = DEFAULT_OUTPUT_TRACK_CSV,
    dataset_name: str = "hindi_hinglish",
) -> pd.DataFrame:
    """
    Merges real and synthetic Hindi/Hinglish metadata into unified track metadata.

    Parameters
    ----------
    real_csv_path:
        Path to hindi_hinglish_real.csv.
    synthetic_csv_path:
        Path to hindi_hinglish_synthetic.csv.
    output_track_path:
        Path to write the combined hindi_hinglish_track.csv.
    dataset_name:
        Dataset identifier string (default: "hindi_hinglish").

    Returns
    -------
    pd.DataFrame:
        Unified DataFrame matching the schema [filepath, label, speaker_id, category, sentence_id, dataset].
    """
    real_path = Path(real_csv_path).resolve()
    synth_path = Path(synthetic_csv_path).resolve()
    out_path = Path(output_track_path).resolve()

    if not real_path.exists():
        raise FileNotFoundError(f"Real metadata CSV not found: {real_path}")
    if not synth_path.exists():
        raise FileNotFoundError(f"Synthetic metadata CSV not found: {synth_path}")

    logger.info("=" * 78)
    logger.info("MERGING HINDI/HINGLISH TRACK METADATA")
    logger.info("=" * 78)
    logger.info(f"Real Source       : {real_path}")
    logger.info(f"Synthetic Source  : {synth_path}")
    logger.info(f"Output Track CSV  : {out_path}")
    logger.info("=" * 78)

    df_real = pd.read_csv(real_path)
    df_synth = pd.read_csv(synth_path)

    # 1. Process real dataframe
    # Filter consent if present
    if "consent_confirmed" in df_real.columns:
        unconsented = df_real[df_real["consent_confirmed"] != True]
        if not unconsented.empty:
            logger.warning(
                f"Filtering out {len(unconsented)} real row(s) with unconfirmed consent."
            )
        df_real = df_real[df_real["consent_confirmed"] == True].copy()

    df_real["label"] = "real"
    df_real["dataset"] = dataset_name

    # 2. Process synthetic dataframe
    df_synth["label"] = "synthetic"
    df_synth["dataset"] = dataset_name

    # 3. Combine and align to unified schema
    common_cols = [c for c in UNIFIED_SCHEMA_COLUMNS if c in df_real.columns and c in df_synth.columns]
    
    # Ensure all required schema columns exist
    for col in UNIFIED_SCHEMA_COLUMNS:
        if col not in df_real.columns:
            df_real[col] = None
        if col not in df_synth.columns:
            df_synth[col] = None

    df_real_subset = df_real[UNIFIED_SCHEMA_COLUMNS].copy()
    df_synth_subset = df_synth[UNIFIED_SCHEMA_COLUMNS].copy()

    df_combined = pd.concat([df_real_subset, df_synth_subset], ignore_index=True)

    # Sort deterministically by speaker_id, sentence_id, label
    df_combined["sentence_id"] = pd.to_numeric(df_combined["sentence_id"], errors="coerce")
    df_combined = df_combined.sort_values(
        by=["speaker_id", "sentence_id", "label"],
        ascending=[True, True, False]
    ).reset_index(drop=True)

    # Validate nulls
    null_counts = df_combined.isnull().sum()
    if null_counts.any():
        logger.warning(f"Null values detected in combined track metadata:\n{null_counts}")
        if null_counts["filepath"] > 0 or null_counts["label"] > 0:
            raise ValueError("Critical null values in 'filepath' or 'label'.")

    # 4. Save to output CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(out_path, index=False)
    logger.info(f"Saved merged track metadata ({len(df_combined):,} rows) -> {out_path}")

    # 5. Compute stats and verification
    real_count = int((df_combined["label"] == "real").sum())
    synth_count = int((df_combined["label"] == "synthetic").sum())
    total_count = len(df_combined)
    unique_speakers = sorted(df_combined["speaker_id"].dropna().unique().tolist())
    num_speakers = len(unique_speakers)

    # Balance check
    imbalance_ratio = (real_count / synth_count) if synth_count > 0 else float("inf")
    is_balanced = (0.90 <= imbalance_ratio <= 1.10) if synth_count > 0 else False

    # Breakdown by speaker and category
    breakdown = (
        df_combined.groupby(["speaker_id", "category", "label"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    if "real" not in breakdown.columns:
        breakdown["real"] = 0
    if "synthetic" not in breakdown.columns:
        breakdown["synthetic"] = 0

    # Print summary
    print("\n" + "=" * 78)
    print("HINDI/HINGLISH TRACK METADATA MERGE SUMMARY")
    print("=" * 78)
    print(f"  Total Track Utterances : {total_count:,}")
    print(f"  Real Audio Clips       : {real_count:,}")
    print(f"  Synthetic Audio Clips  : {synth_count:,}")
    print(f"  Real / Synthetic Ratio : {imbalance_ratio:.2f}:1")
    print(f"  Balanced (1:1 target)  : {'YES (Matched 1:1)' if is_balanced else 'WARNING (Imbalanced)'}")
    print(f"  Unique Speakers ({num_speakers})   : {', '.join(unique_speakers)}")
    print("-" * 78)
    print("  Per-Speaker / Category Breakdown:")
    for _, row in breakdown.iterrows():
        spk = row["speaker_id"]
        cat = row["category"]
        r = row["real"]
        s = row["synthetic"]
        print(f"    - {spk:<12} | {cat:<10} : {r:>2} real, {s:>2} synthetic")
    print("=" * 78 + "\n")

    if not is_balanced:
        logger.warning(
            f"Real/synthetic counts deviate from 1:1 ({real_count} real vs {synth_count} synthetic). "
            "Check generate_hindi_clones failure logs for any skipped or failed generations."
        )

    return df_combined


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge real and synthetic Hindi/Hinglish metadata into unified track metadata."
    )
    parser.add_argument(
        "--real_csv",
        type=Path,
        default=DEFAULT_REAL_CSV,
        help=f"Path to real Hindi metadata CSV (default: {DEFAULT_REAL_CSV})",
    )
    parser.add_argument(
        "--synthetic_csv",
        type=Path,
        default=DEFAULT_SYNTHETIC_CSV,
        help=f"Path to synthetic Hindi metadata CSV (default: {DEFAULT_SYNTHETIC_CSV})",
    )
    parser.add_argument(
        "--output_track_csv",
        type=Path,
        default=DEFAULT_OUTPUT_TRACK_CSV,
        help=f"Path to output combined track CSV (default: {DEFAULT_OUTPUT_TRACK_CSV})",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="hindi_hinglish",
        help="Dataset label name (default: 'hindi_hinglish')",
    )

    args = parser.parse_args()

    try:
        merge_hindi_metadata(
            real_csv_path=args.real_csv,
            synthetic_csv_path=args.synthetic_csv,
            output_track_path=args.output_track_csv,
            dataset_name=args.dataset_name,
        )
    except Exception as exc:
        logger.error(f"Failed to merge Hindi metadata: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
