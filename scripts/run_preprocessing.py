#!/usr/bin/env python3
"""
run_preprocessing.py — Batch audio preprocessing script for VoxGuard.

Standardizes raw audio datasets (ASVspoof 2019, WaveFake, In-the-Wild) to:
- 16,000 Hz sample rate
- Single-channel (mono) float32
- Silence-trimmed (with non-empty safety fallback)
- Saved to data/processed/<dataset>/<dataset>_<unique_stem>.wav

Workflow:
1. Loads unified metadata (or rebuilds from raw per-dataset metadata CSVs).
2. If --dry_run is passed, selects only the first 20 files per dataset for a smoke test
   and DOES NOT overwrite data/metadata/unified.csv.
3. If real run (non-dry-run), processes all files and saves the resulting DataFrame
   with repo-relative 'processed_path' column to data/metadata/unified.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

from voxguard.config import BASE_DIR, DATA_METADATA_DIR, DATA_PROCESSED_DIR, SAMPLE_RATE
from voxguard.utils.logging_utils import get_logger
from voxguard.utils.metadata import load_unified_metadata, save_unified_metadata
from voxguard.utils.preprocess import preprocess_dataset

logger = get_logger("run_preprocessing")

DEFAULT_DATASETS = ["asvspoof2019", "wavefake", "in_the_wild"]


def run_preprocessing(
    dataset_names: List[str] = DEFAULT_DATASETS,
    output_dir: Path = DATA_PROCESSED_DIR,
    target_sr: int = SAMPLE_RATE,
    trim_silence: bool = True,
    dry_run: bool = False,
    dry_run_count: int = 20,
    use_full_wavefake: bool = False,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Executes the preprocessing workflow.
    """
    logger.info("=" * 78)
    logger.info("VOXGUARD DATASET PREPROCESSING PIPELINE")
    logger.info(f"Datasets        : {dataset_names}")
    logger.info(f"Target SR       : {target_sr} Hz")
    logger.info(f"Trim Silence    : {trim_silence}")
    logger.info(f"Output Dir      : {output_dir.resolve()}")
    logger.info(f"Dry Run Mode    : {dry_run}")
    logger.info(f"Full WaveFake   : {use_full_wavefake}")
    logger.info("=" * 78)

    # 1. Load metadata
    df_meta = load_unified_metadata(
        dataset_names=dataset_names,
        use_full_wavefake=use_full_wavefake,
        force_rebuild=dry_run or force_rebuild,
    )

    if df_meta.empty:
        raise ValueError("No metadata records found for specified datasets.")

    logger.info(f"Loaded metadata for {len(df_meta):,} total utterances.")

    # 2. Handle dry run sampling (first 20 per dataset)
    if dry_run:
        logger.info(f"--dry_run active: selecting first {dry_run_count} utterances per dataset...")
        sampled_dfs = []
        for dataset in df_meta["dataset"].unique():
            ds_subset = df_meta[df_meta["dataset"] == dataset].head(dry_run_count)
            sampled_dfs.append(ds_subset)
            logger.info(f"  - {dataset}: {len(ds_subset)} utterances selected for dry run.")
        working_df = pd.concat(sampled_dfs, ignore_index=True)
    else:
        working_df = df_meta

    # 3. Execute audio preprocessing
    processed_df = preprocess_dataset(
        metadata_df=working_df,
        output_dir=output_dir,
        target_sr=target_sr,
        trim_silence=trim_silence,
        project_root=BASE_DIR,
    )

    # 4. Save metadata back to unified.csv ONLY on real (non-dry-run) execution
    if dry_run:
        logger.warning("-" * 78)
        logger.warning(
            f"[DRY RUN COMPLETE] Successfully preprocessed {len(processed_df)} test clips. "
            "data/metadata/unified.csv was NOT overwritten (protected against partial truncation)."
        )
        logger.warning("-" * 78)
    else:
        unified_path = save_unified_metadata(processed_df)
        logger.info("-" * 78)
        logger.info(
            f"[SUCCESS] Full preprocessing complete ({len(processed_df):,} files). "
            f"Overwrote {unified_path} with complete 'processed_path' column."
        )
        logger.info("-" * 78)

    return processed_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess VoxGuard datasets (16kHz, mono, silence-trimmed, unique naming)."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help=f"Dataset names to preprocess (default: {' '.join(DEFAULT_DATASETS)})",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DATA_PROCESSED_DIR,
        help=f"Directory to save preprocessed audio (default: {DATA_PROCESSED_DIR})",
    )
    parser.add_argument(
        "--target_sr",
        type=int,
        default=SAMPLE_RATE,
        help=f"Target sampling rate in Hz (default: {SAMPLE_RATE})",
    )
    parser.add_argument(
        "--no_trim_silence",
        action="store_true",
        help="Disable silence trimming",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Process only the first 20 files per dataset for a quick smoke test without overwriting unified.csv",
    )
    parser.add_argument(
        "--dry_run_count",
        type=int,
        default=20,
        help="Number of files per dataset to process during dry run (default: 20)",
    )
    parser.add_argument(
        "--use_full_wavefake",
        action="store_true",
        help="Preprocess full WaveFake dataset (131k files) instead of the 8k stratified subset",
    )
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="Force re-reading raw metadata CSVs instead of using cached unified.csv",
    )

    args = parser.parse_args()

    try:
        run_preprocessing(
            dataset_names=args.datasets,
            output_dir=args.output_dir,
            target_sr=args.target_sr,
            trim_silence=not args.no_trim_silence,
            dry_run=args.dry_run,
            dry_run_count=args.dry_run_count,
            use_full_wavefake=args.use_full_wavefake,
            force_rebuild=args.force_rebuild,
        )
    except Exception as exc:
        logger.error(f"Preprocessing execution failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
