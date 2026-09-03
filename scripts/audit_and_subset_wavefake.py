#!/usr/bin/env python3
"""
audit_and_subset_wavefake.py — Audit WaveFake & In-the-Wild datasets and create a stratified WaveFake subset.

This script performs two critical data-governance tasks for the VoxGuard pipeline:
1. Audits the downloaded dataset metadata (data/metadata/wavefake.csv and data/metadata/in_the_wild.csv),
   confirming exact row counts, label distributions (bonafide vs spoof), and per-generator allocations
   before downstream processing decisions are made.
2. Builds a proportionally stratified subset of WaveFake (default: 8,000 utterances) sampled across
   BOTH `label` and `generator` dimensions. This guarantees that no single vocoder or TTS architecture
   dominates the cross-dataset evaluation distribution while maintaining the original bonafide/spoof ratio.

Outputs:
  - data/metadata/wavefake_subset.csv (new subset metadata)
  - Leaves data/metadata/wavefake.csv (the full 131k+ dataset) completely untouched.
  - Prints a before-vs-after comparison table formatted for dataset-card documentation.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

# Attempt import from voxguard package or resolve relative paths
try:
    from voxguard.config import BASE_DIR, DATA_METADATA_DIR
    from voxguard.utils.logging_utils import get_logger
except ImportError:
    # Fallback if voxguard is not installed in PYTHONPATH
    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_METADATA_DIR = BASE_DIR / "data" / "metadata"
    sys.path.insert(0, str(BASE_DIR / "src"))
    try:
        from voxguard.utils.logging_utils import get_logger
    except ImportError:
        get_logger = None

DEFAULT_WAVEFAKE_CSV = DATA_METADATA_DIR / "wavefake.csv"
DEFAULT_ITW_CSV = DATA_METADATA_DIR / "in_the_wild.csv"
DEFAULT_SUBSET_CSV = DATA_METADATA_DIR / "wavefake_subset.csv"
DEFAULT_SUBSET_SIZE = 8000
DEFAULT_RANDOM_STATE = 42
MIN_GENERATOR_SAMPLE_THRESHOLD = 20


def setup_logger(verbose: bool = True) -> logging.Logger:
    """Configures and returns a logger for audit and subset operations."""
    if get_logger is not None:
        logger = get_logger("audit_and_subset_wavefake")
        if verbose:
            logger.setLevel(logging.INFO)
        return logger

    # Fallback standalone logger
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("audit_and_subset_wavefake")


def load_dataset(csv_path: Path, dataset_name: str, logger: logging.Logger) -> pd.DataFrame:
    """Loads a metadata CSV and validates required schema."""
    if not csv_path.exists():
        msg = f"Metadata file for {dataset_name} not found at: {csv_path.resolve()}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {dataset_name} metadata ({len(df):,} rows) from {csv_path.name}")
    return df


def audit_in_the_wild(df_itw: pd.DataFrame, logger: logging.Logger) -> None:
    """Prints an audit breakdown for In-the-Wild metadata."""
    total = len(df_itw)
    logger.info("=" * 78)
    logger.info("AUDIT REPORT: In-the-Wild Dataset (data/metadata/in_the_wild.csv)")
    logger.info("=" * 78)
    logger.info(f"Total audio utterances: {total:,}")

    if "label" in df_itw.columns:
        label_counts = df_itw["label"].value_counts()
        for label, count in label_counts.items():
            pct = (count / total) * 100.0 if total > 0 else 0.0
            logger.info(f"  - Label '{label}': {count:>6,} ({pct:>5.1f}%)")
    else:
        logger.warning("  Column 'label' missing from in_the_wild.csv")

    if "speaker" in df_itw.columns:
        n_speakers = df_itw["speaker"].nunique()
        logger.info(f"  - Unique speakers: {n_speakers:,}")
    logger.info("-" * 78)


def audit_wavefake(df_wf: pd.DataFrame, logger: logging.Logger) -> None:
    """Prints an audit breakdown for WaveFake full dataset metadata."""
    total = len(df_wf)
    logger.info("=" * 78)
    logger.info("AUDIT REPORT: WaveFake Dataset Full Set (data/metadata/wavefake.csv)")
    logger.info("=" * 78)
    logger.info(f"Total audio utterances: {total:,}")

    if "label" in df_wf.columns:
        label_counts = df_wf["label"].value_counts()
        for label, count in label_counts.items():
            pct = (count / total) * 100.0 if total > 0 else 0.0
            logger.info(f"  - Label '{label}': {count:>6,} ({pct:>5.1f}%)")
    else:
        logger.warning("  Column 'label' missing from wavefake.csv")

    if "generator" in df_wf.columns:
        gen_counts = df_wf["generator"].value_counts()
        logger.info(f"  - Generators breakdown ({len(gen_counts)} distinct generators):")
        for gen, count in gen_counts.items():
            pct = (count / total) * 100.0 if total > 0 else 0.0
            logger.info(f"      * {gen.ljust(22)}: {count:>6,} ({pct:>5.1f}%)")
    else:
        logger.warning("  Column 'generator' missing from wavefake.csv")
    logger.info("-" * 78)


def stratified_subset_wavefake(
    df_wf: pd.DataFrame,
    subset_size: int,
    random_state: int = DEFAULT_RANDOM_STATE,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Builds a proportionally stratified subset of WaveFake across BOTH label and generator.

    Args:
        df_wf: Full WaveFake DataFrame with ['filepath', 'label', 'generator'].
        subset_size: Desired number of samples in the subset.
        random_state: Seed for reproducible random sampling.
        logger: Logger instance.

    Returns:
        pd.DataFrame: Stratified subset DataFrame with identical schema.
    """
    log = logger or logging.getLogger("audit_and_subset_wavefake")
    total_samples = len(df_wf)

    if subset_size <= 0:
        raise ValueError(f"subset_size must be a positive integer, got {subset_size}")

    if subset_size >= total_samples:
        log.warning(
            f"Requested subset_size ({subset_size:,}) >= total available WaveFake samples ({total_samples:,}). "
            "Returning full dataset without downsampling."
        )
        return df_wf.copy()

    # Validate required columns
    required_cols = {"filepath", "label", "generator"}
    missing = required_cols - set(df_wf.columns)
    if missing:
        raise ValueError(f"WaveFake metadata missing required column(s): {missing}")

    # Check for generators with small sample counts
    gen_counts = df_wf["generator"].value_counts()
    for gen, count in gen_counts.items():
        if count < MIN_GENERATOR_SAMPLE_THRESHOLD:
            log.warning(
                f"[GRACEFUL FALLBACK] Generator '{gen}' has only {count} total sample(s) "
                f"(< threshold {MIN_GENERATOR_SAMPLE_THRESHOLD}). It will be preserved or sampled carefully."
            )

    # Combined stratification key (label + generator)
    strat_keys = df_wf["label"].astype(str) + "___" + df_wf["generator"].astype(str)
    strat_counts = strat_keys.value_counts()

    # Determine whether sklearn train_test_split can safely be used
    min_stratum_size = strat_counts.min()
    use_sklearn = False

    # sklearn's train_test_split requires each stratum to have at least 2 samples,
    # and the proportional split shouldn't truncate any class to 0.
    if min_stratum_size >= 2:
        try:
            from sklearn.model_selection import train_test_split

            # Check if smallest class allocation is at least 1
            min_expected_alloc = (min_stratum_size * subset_size) / total_samples
            if min_expected_alloc >= 1.0:
                use_sklearn = True
        except ImportError:
            log.warning("scikit-learn not available; falling back to pandas groupby proportional sampling.")
            use_sklearn = False

    subset_df: pd.DataFrame

    if use_sklearn:
        try:
            from sklearn.model_selection import train_test_split

            log.info(
                f"Applying scikit-learn train_test_split stratification across label+generator "
                f"(subset_size={subset_size:,}, seed={random_state})..."
            )
            sampled_df, _ = train_test_split(
                df_wf,
                train_size=subset_size,
                stratify=strat_keys,
                random_state=random_state,
                shuffle=True,
            )
            subset_df = sampled_df.copy()
        except Exception as err:
            log.warning(
                f"sklearn train_test_split encountered an issue ({err}); "
                "falling back gracefully to exact pandas groupby proportional sampling."
            )
            subset_df = _proportional_groupby_sample(df_wf, strat_keys, subset_size, random_state, log)
    else:
        log.info(
            f"Applying pandas groupby proportional sampling across label+generator "
            f"(subset_size={subset_size:,}, seed={random_state})..."
        )
        subset_df = _proportional_groupby_sample(df_wf, strat_keys, subset_size, random_state, log)

    # Ensure deterministic shuffling and clean index
    subset_df = subset_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    # Ensure correct schema
    subset_df = subset_df[["filepath", "label", "generator"]]

    log.info(f"Successfully generated stratified subset with {len(subset_df):,} samples.")
    return subset_df


def _proportional_groupby_sample(
    df: pd.DataFrame,
    strat_keys: pd.Series,
    subset_size: int,
    random_state: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Performs exact largest-remainder (Hare-Niemeyer) proportional quota sampling
    across each (label, generator) stratum.
    """
    total_len = len(df)
    unique_strata = strat_keys.unique()

    # Step 1: Compute exact proportional floating quotas
    quotas: Dict[str, float] = {}
    base_counts: Dict[str, int] = {}
    remainders: Dict[str, float] = {}

    for stratum in unique_strata:
        stratum_mask = strat_keys == stratum
        stratum_total = stratum_mask.sum()
        exact_quota = (stratum_total / total_len) * subset_size
        quotas[stratum] = exact_quota
        base = int(math.floor(exact_quota))
        # Ensure at least 1 sample if stratum exists and subset_size is non-zero
        base = max(1, min(base, stratum_total))
        base_counts[stratum] = base
        remainders[stratum] = exact_quota - math.floor(exact_quota)

    # Step 2: Distribute remaining slots by highest remainder
    current_total = sum(base_counts.values())
    shortfall = subset_size - current_total

    if shortfall > 0:
        sorted_by_remainder = sorted(remainders.items(), key=lambda x: x[1], reverse=True)
        for stratum, _ in sorted_by_remainder:
            if shortfall == 0:
                break
            stratum_mask = strat_keys == stratum
            stratum_total = stratum_mask.sum()
            if base_counts[stratum] < stratum_total:
                base_counts[stratum] += 1
                shortfall -= 1
    elif shortfall < 0:
        sorted_by_remainder = sorted(remainders.items(), key=lambda x: x[1])
        for stratum, _ in sorted_by_remainder:
            if shortfall == 0:
                break
            if base_counts[stratum] > 1:
                base_counts[stratum] -= 1
                shortfall += 1

    # Step 3: Sample allocated counts per stratum
    sampled_dfs = []
    temp_df = df.copy()
    temp_df["_strat_key"] = strat_keys

    for stratum, n_samples in base_counts.items():
        stratum_subset = temp_df[temp_df["_strat_key"] == stratum]
        actual_n = min(n_samples, len(stratum_subset))
        sampled = stratum_subset.sample(n=actual_n, random_state=random_state)
        sampled_dfs.append(sampled)

    combined_df = pd.concat(sampled_dfs, ignore_index=True)
    combined_df = combined_df.drop(columns=["_strat_key"])
    return combined_df


def format_dataset_card_table(df_full: pd.DataFrame, df_subset: pd.DataFrame) -> str:
    """
    Builds a markdown before/after summary table comparing the full dataset
    and the stratified subset, ready for dataset-card documentation.
    """
    total_full = len(df_full)
    total_sub = len(df_subset)

    # Group full counts by generator and label
    full_counts = (
        df_full.groupby(["generator", "label"])
        .size()
        .reset_index(name="full_count")
    )
    sub_counts = (
        df_subset.groupby(["generator", "label"])
        .size()
        .reset_index(name="subset_count")
    )

    merged = pd.merge(full_counts, sub_counts, on=["generator", "label"], how="outer").fillna(0)
    merged["subset_count"] = merged["subset_count"].astype(int)

    # Sort: bonafide/original first, then spoof generators descending by count
    merged["is_spoof"] = merged["label"].apply(lambda x: 1 if x == "spoof" else 0)
    merged = merged.sort_values(by=["is_spoof", "full_count"], ascending=[True, False]).drop(columns=["is_spoof"])

    lines = []
    lines.append("### WaveFake Dataset Subsetting Card")
    lines.append("")
    lines.append(f"**Sampling Strategy:** Proportional stratified sampling across `label` and `generator`.")
    lines.append(f"**Full Dataset Size:** {total_full:,} utterances | **Stratified Subset Size:** {total_sub:,} utterances ({total_sub/total_full*100:.2f}% retained)")
    lines.append("")
    lines.append("| Generator / Vocoder | Label | Full Count | Full % | Subset Count | Subset % | Retained Ratio |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")

    for _, row in merged.iterrows():
        gen = row["generator"]
        lbl = row["label"]
        fc = int(row["full_count"])
        sc = int(row["subset_count"])
        fpct = (fc / total_full) * 100.0 if total_full > 0 else 0.0
        spct = (sc / total_sub) * 100.0 if total_sub > 0 else 0.0
        ret_ratio = (sc / fc) * 100.0 if fc > 0 else 0.0
        lines.append(f"| `{gen}` | **{lbl}** | {fc:,} | {fpct:.2f}% | {sc:,} | {spct:.2f}% | {ret_ratio:.2f}% |")

    # Add Summary Rows
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
    
    # Bonafide total
    full_bona = int(df_full[df_full["label"] == "bonafide"].shape[0])
    sub_bona = int(df_subset[df_subset["label"] == "bonafide"].shape[0])
    lines.append(f"| **TOTAL BONAFIDE** | `bonafide` | **{full_bona:,}** | **{full_bona/total_full*100:.2f}%** | **{sub_bona:,}** | **{sub_bona/total_sub*100:.2f}%** | **{sub_bona/full_bona*100:.2f}%** |")

    # Spoof total
    full_spoof = int(df_full[df_full["label"] == "spoof"].shape[0])
    sub_spoof = int(df_subset[df_subset["label"] == "spoof"].shape[0])
    lines.append(f"| **TOTAL SPOOF** | `spoof` | **{full_spoof:,}** | **{full_spoof/total_full*100:.2f}%** | **{sub_spoof:,}** | **{sub_spoof/total_sub*100:.2f}%** | **{sub_spoof/full_spoof*100:.2f}%** |")

    # Total all
    lines.append(f"| **TOTAL (ALL)** | `all` | **{total_full:,}** | **100.00%** | **{total_sub:,}** | **100.00%** | **{total_sub/total_full*100:.2f}%** |")
    lines.append("")

    return "\n".join(lines)


def audit_and_subset(
    wavefake_csv: Path = DEFAULT_WAVEFAKE_CSV,
    itw_csv: Path = DEFAULT_ITW_CSV,
    output_csv: Path = DEFAULT_SUBSET_CSV,
    subset_size: int = DEFAULT_SUBSET_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Main programmatic pipeline:
    1. Loads and audits wavefake.csv and in_the_wild.csv.
    2. Builds a stratified subset of WaveFake across label and generator.
    3. Saves data/metadata/wavefake_subset.csv without touching wavefake.csv.
    4. Prints formatted before/after summary table.
    """
    logger = setup_logger(verbose=verbose)

    # 1. Load datasets
    df_wf = load_dataset(wavefake_csv, "WaveFake", logger)
    
    # In-the-wild load (audit step)
    if itw_csv.exists():
        df_itw = load_dataset(itw_csv, "In-the-Wild", logger)
        audit_in_the_wild(df_itw, logger)
    else:
        logger.warning(f"In-the-Wild metadata not found at {itw_csv}. Skipping ITW audit.")

    # 2. Audit WaveFake
    audit_wavefake(df_wf, logger)

    # 3. Create stratified subset
    logger.info("=" * 78)
    logger.info(f"CREATING STRATIFIED WAVEFAKE SUBSET (Target Size: {subset_size:,})")
    logger.info("=" * 78)
    df_subset = stratified_subset_wavefake(
        df_wf=df_wf,
        subset_size=subset_size,
        random_state=random_state,
        logger=logger,
    )

    # 4. Save to wavefake_subset.csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_subset.to_csv(output_csv, index=False)
    logger.info(f"[SUCCESS] Saved stratified subset ({len(df_subset):,} rows) to: {output_csv.resolve()}")
    logger.info(f"[CONFIRMED] Full dataset remains untouched at: {wavefake_csv.resolve()} ({len(df_wf):,} rows)")

    # 5. Print Before/After Summary Table
    table_markdown = format_dataset_card_table(df_wf, df_subset)
    print("\n" + "=" * 78)
    print("DATASET CARD SUMMARY TABLE (Copy & Paste Ready):")
    print("=" * 78)
    print(table_markdown)
    print("=" * 78 + "\n")

    return df_subset


def main() -> None:
    """CLI interface for audit_and_subset_wavefake.py."""
    parser = argparse.ArgumentParser(
        description="Audit WaveFake & In-the-Wild metadata and create a proportionally stratified WaveFake subset."
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=DEFAULT_SUBSET_SIZE,
        help=f"Target number of utterances in the stratified WaveFake subset (default: {DEFAULT_SUBSET_SIZE})",
    )
    parser.add_argument(
        "--wavefake_csv",
        type=Path,
        default=DEFAULT_WAVEFAKE_CSV,
        help=f"Path to input WaveFake metadata CSV (default: {DEFAULT_WAVEFAKE_CSV})",
    )
    parser.add_argument(
        "--itw_csv",
        type=Path,
        default=DEFAULT_ITW_CSV,
        help=f"Path to In-the-Wild metadata CSV for audit (default: {DEFAULT_ITW_CSV})",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=DEFAULT_SUBSET_CSV,
        help=f"Path to output WaveFake subset CSV (default: {DEFAULT_SUBSET_CSV})",
    )
    parser.add_argument(
        "--random_state",
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        dest="random_state",
        help=f"Random seed for deterministic stratification (default: {DEFAULT_RANDOM_STATE})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed info logging and show only summary table/errors",
    )

    args = parser.parse_args()

    try:
        audit_and_subset(
            wavefake_csv=args.wavefake_csv,
            itw_csv=args.itw_csv,
            output_csv=args.output_csv,
            subset_size=args.subset_size,
            random_state=args.random_state,
            verbose=not args.quiet,
        )
    except Exception as exc:
        sys.stderr.write(f"Error during audit and subsetting: {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
