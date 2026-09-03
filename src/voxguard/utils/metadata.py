"""
metadata.py — Unified metadata loading, normalization, and persistence.

This module standardizes metadata across multiple audio deepfake datasets
(ASVspoof 2019, WaveFake, In-the-Wild, etc.) into a single canonical schema:
    [filepath, label, dataset, split] (+ optional processed_path)

Key features:
1. Canonical schema & label mapping:
   - "bonafide" / "real" -> "real"
   - "spoof" / "fake" / "synthetic" -> "synthetic"
2. WaveFake smart subset handling:
   - Reads data/metadata/wavefake_subset.csv by default when "wavefake" is requested.
   - Falls back to full wavefake.csv if subset is missing.
   - Explicit `use_full_wavefake=True` parameter to force loading the full dataset.
3. Preprocessing cache preservation:
   - If data/metadata/unified.csv exists and contains non-null `processed_path`
     values for all requested datasets, loads directly from unified.csv to avoid
     overwriting or discarding downstream preprocessing progress.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Set, Union

import pandas as pd

from voxguard.config import DATA_METADATA_DIR
from voxguard.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Canonical columns required for unified metadata
CANONICAL_COLUMNS = ["filepath", "label", "dataset", "split"]

# Mapping dictionary for labels
LABEL_MAPPING = {
    "bonafide": "real",
    "bona-fide": "real",
    "real": "real",
    "authentic": "real",
    "original": "real",
    "0": "real",
    0: "real",
    "spoof": "synthetic",
    "fake": "synthetic",
    "synthetic": "synthetic",
    "deepfake": "synthetic",
    "cloned": "synthetic",
    "1": "synthetic",
    1: "synthetic",
}

DEFAULT_UNIFIED_CSV = DATA_METADATA_DIR / "unified.csv"


def normalize_label(raw_label: Union[str, int]) -> str:
    """
    Normalizes arbitrary dataset labels to either 'real' or 'synthetic'.

    Raises:
        ValueError: If the raw label cannot be mapped unambiguously.
    """
    if pd.isna(raw_label):
        raise ValueError("Cannot normalize null or NaN label.")

    label_str = str(raw_label).strip().lower()
    if label_str in LABEL_MAPPING:
        return LABEL_MAPPING[label_str]

    raise ValueError(f"Unrecognized dataset label: {raw_label!r}")


def _normalize_dataset_name(name: str) -> str:
    """Normalizes dataset identifier string."""
    name_clean = name.strip().lower().replace("-", "_")
    if name_clean in ("asvspoof", "asvspoof2019", "asvspoof_2019", "asv_2019"):
        return "asvspoof2019"
    if name_clean in ("wavefake", "wave_fake", "wf"):
        return "wavefake"
    if name_clean in ("in_the_wild", "itw", "inthewild"):
        return "in_the_wild"
    return name_clean


def _infer_split(filepath: str, dataset_name: str, existing_split: Optional[str] = None) -> str:
    """
    Infers the partition split ('train', 'dev', 'eval', or 'unknown').
    For ASVspoof2019, inspects path conventions (LA_T_, LA_D_, LA_E_ or _train, _dev, _eval).
    """
    if existing_split and str(existing_split).strip().lower() in ("train", "dev", "eval"):
        return str(existing_split).strip().lower()

    fp_lower = filepath.lower().replace("\\", "/")
    if dataset_name == "asvspoof2019":
        if "la_train" in fp_lower or "/train/" in fp_lower or "la_t_" in fp_lower:
            return "train"
        if "la_dev" in fp_lower or "/dev/" in fp_lower or "la_d_" in fp_lower:
            return "dev"
        if "la_eval" in fp_lower or "/eval/" in fp_lower or "la_e_" in fp_lower:
            return "eval"

    # Other datasets default to unknown unless explicit
    return "unknown"


def _load_single_dataset(
    dataset_name: str,
    metadata_dir: Path,
    use_full_wavefake: bool = False,
) -> pd.DataFrame:
    """
    Loads and standardizes a single dataset's raw metadata CSV.
    """
    canon_name = _normalize_dataset_name(dataset_name)

    # Resolve CSV file path
    csv_path: Path
    if canon_name == "asvspoof2019":
        csv_path = metadata_dir / "asvspoof2019.csv"
    elif canon_name == "wavefake":
        if use_full_wavefake:
            csv_path = metadata_dir / "wavefake.csv"
            logger.info("Explicitly requested full WaveFake dataset (wavefake.csv).")
        else:
            subset_path = metadata_dir / "wavefake_subset.csv"
            if subset_path.exists():
                csv_path = subset_path
                logger.info("Using stratified WaveFake subset (wavefake_subset.csv).")
            else:
                csv_path = metadata_dir / "wavefake.csv"
                logger.warning(
                    f"WaveFake subset not found at {subset_path}. "
                    f"Falling back to full dataset at {csv_path}."
                )
    elif canon_name == "in_the_wild":
        csv_path = metadata_dir / "in_the_wild.csv"
    else:
        csv_path = metadata_dir / f"{canon_name}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Metadata CSV for dataset '{dataset_name}' not found at: {csv_path.resolve()}"
        )

    df_raw = pd.read_csv(csv_path)

    # Validate required columns
    if "filepath" not in df_raw.columns:
        raise ValueError(f"Metadata file {csv_path} missing required 'filepath' column.")
    if "label" not in df_raw.columns:
        raise ValueError(f"Metadata file {csv_path} missing required 'label' column.")

    # 1. Check for null or empty filepaths
    if df_raw["filepath"].isnull().any() or (df_raw["filepath"].astype(str).str.strip() == "").any():
        raise ValueError(f"Found null or empty filepaths in {csv_path}")

    # 2. Check for null labels
    if df_raw["label"].isnull().any():
        raise ValueError(f"Found null labels in {csv_path}")

    # 3. Standardize filepaths to absolute posix strings
    abs_filepaths = df_raw["filepath"].apply(
        lambda p: Path(p).resolve().as_posix() if not Path(p).is_absolute() else Path(p).as_posix()
    )

    # 4. Map labels to 'real' / 'synthetic'
    mapped_labels = df_raw["label"].apply(normalize_label)

    # 5. Determine split
    existing_split = df_raw["split"] if "split" in df_raw.columns else None
    if existing_split is not None:
        splits = [
            _infer_split(fp, canon_name, sp)
            for fp, sp in zip(abs_filepaths, existing_split)
        ]
    else:
        splits = [_infer_split(fp, canon_name) for fp in abs_filepaths]

    records = {
        "filepath": abs_filepaths,
        "label": mapped_labels,
        "dataset": canon_name,
        "split": splits,
    }

    # If processed_path is present in raw file, preserve it
    if "processed_path" in df_raw.columns:
        records["processed_path"] = df_raw["processed_path"]

    standardized_df = pd.DataFrame(records)
    logger.info(
        f"Standardized {len(standardized_df):,} rows from {csv_path.name} (dataset: {canon_name})"
    )
    return standardized_df


def _can_load_from_unified_cache(
    unified_csv: Path,
    requested_canonical_names: List[str],
) -> Optional[pd.DataFrame]:
    """
    Checks whether unified_csv exists, contains all requested datasets, and has a
    non-null, non-empty processed_path column for all rows belonging to those datasets.
    """
    if not unified_csv.exists():
        return None

    try:
        df_unified = pd.read_csv(unified_csv)
    except Exception as exc:
        logger.warning(f"Failed to read existing unified.csv at {unified_csv}: {exc}")
        return None

    # Check required columns
    required_cols = {"filepath", "label", "dataset", "split", "processed_path"}
    if not required_cols.issubset(df_unified.columns):
        return None

    # Check if all requested datasets exist in df_unified
    available_datasets = set(df_unified["dataset"].unique())
    requested_set = set(requested_canonical_names)

    if not requested_set.issubset(available_datasets):
        logger.debug(
            f"unified.csv missing some requested datasets: {requested_set - available_datasets}"
        )
        return None

    # Filter for requested datasets
    mask = df_unified["dataset"].isin(requested_set)
    df_filtered = df_unified[mask].copy()

    if df_filtered.empty:
        return None

    # Check if processed_path is completely non-null and non-empty for requested rows
    has_nulls = df_filtered["processed_path"].isnull().any()
    has_empties = (df_filtered["processed_path"].astype(str).str.strip() == "").any()

    if has_nulls or has_empties:
        logger.debug("unified.csv contains null or empty processed_path values for requested datasets.")
        return None

    logger.info(
        f"Found complete cached processed metadata in {unified_csv.name} "
        f"({len(df_filtered):,} rows across {list(requested_set)})."
    )
    return df_filtered.reset_index(drop=True)


def load_unified_metadata(
    dataset_names: List[str],
    use_full_wavefake: bool = False,
    force_rebuild: bool = False,
    metadata_dir: Optional[Union[Path, str]] = None,
    unified_csv_path: Optional[Union[Path, str]] = None,
) -> pd.DataFrame:
    """
    Loads and unifies metadata for one or more datasets.

    Args:
        dataset_names: List of dataset identifiers, e.g. ['asvspoof2019', 'wavefake', 'in_the_wild'].
        use_full_wavefake: If True, explicitly loads full wavefake.csv instead of wavefake_subset.csv.
        force_rebuild: If True, skips loading from unified.csv cache and rebuilds from raw CSVs.
        metadata_dir: Directory containing raw metadata CSVs (defaults to data/metadata).
        unified_csv_path: Path to unified.csv cache (defaults to data/metadata/unified.csv).

    Returns:
        pd.DataFrame: Unified DataFrame with columns [filepath, label, dataset, split] (+ processed_path if cached).
    """
    if not dataset_names:
        raise ValueError("dataset_names list must contain at least one dataset name.")

    meta_dir = Path(metadata_dir).resolve() if metadata_dir else DATA_METADATA_DIR
    unified_csv = Path(unified_csv_path).resolve() if unified_csv_path else meta_dir / "unified.csv"

    canonical_names = [_normalize_dataset_name(name) for name in dataset_names]

    # Requirement 2: Check if unified.csv already exists with valid processed_path
    if not force_rebuild:
        cached_df = _can_load_from_unified_cache(unified_csv, canonical_names)
        if cached_df is not None:
            # If wavefake is requested, check if subset vs full matches cache
            if "wavefake" in canonical_names:
                wf_cached_count = len(cached_df[cached_df["dataset"] == "wavefake"])
                if use_full_wavefake and wf_cached_count < 20000:
                    logger.info(
                        "use_full_wavefake=True requested, but cached unified.csv contains a WaveFake subset. "
                        "Rebuilding from source raw CSVs."
                    )
                elif not use_full_wavefake and wf_cached_count > 20000:
                    logger.info(
                        "WaveFake subset requested, but cached unified.csv contains the full WaveFake dataset. "
                        "Rebuilding from source raw CSVs."
                    )
                else:
                    return cached_df
            else:
                return cached_df

    # Build from individual per-dataset CSVs
    dfs: List[pd.DataFrame] = []
    for name in dataset_names:
        df_single = _load_single_dataset(
            dataset_name=name,
            metadata_dir=meta_dir,
            use_full_wavefake=use_full_wavefake,
        )
        dfs.append(df_single)

    combined_df = pd.concat(dfs, ignore_index=True)

    # Ensure no null filepaths or labels
    if combined_df["filepath"].isnull().any():
        raise ValueError("Unified metadata contains null filepaths after combination.")
    if combined_df["label"].isnull().any():
        raise ValueError("Unified metadata contains null labels after combination.")

    logger.info(
        f"Unified metadata ready: {len(combined_df):,} total rows across {canonical_names}. "
        f"Label breakdown: {combined_df['label'].value_counts().to_dict()}"
    )

    return combined_df


def save_unified_metadata(
    df: pd.DataFrame,
    output_path: Optional[Union[Path, str]] = None,
) -> Path:
    """
    Saves the unified metadata DataFrame to disk (default: data/metadata/unified.csv).

    Args:
        df: Unified metadata DataFrame.
        output_path: Target CSV file path.

    Returns:
        Path: Resolved path where the CSV was written.
    """
    out_path = Path(output_path).resolve() if output_path else DEFAULT_UNIFIED_CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Basic schema validation
    missing_cols = set(CANONICAL_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"DataFrame is missing required canonical column(s): {missing_cols}")

    if df["filepath"].isnull().any():
        raise ValueError("Cannot save unified metadata with null filepaths.")
    if df["label"].isnull().any():
        raise ValueError("Cannot save unified metadata with null labels.")

    df.to_csv(out_path, index=False)
    logger.info(f"Saved unified metadata ({len(df):,} rows) to: {out_path}")
    return out_path
