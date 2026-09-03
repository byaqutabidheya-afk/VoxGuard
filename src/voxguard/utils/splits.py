"""
splits.py — Dataset partition splits respecting official protocols.

This module provides dataset partition extraction for training and evaluation:
- ASVspoof 2019: Strictly preserves the OFFICIAL train, dev, and eval splits to
  prevent speaker or attack system leakage across subsets (preserving EER fidelity).
- WaveFake & In-the-Wild: Returned as eval-only partitions for zero-shot generalization
  evaluation (since they are strictly out-of-domain evaluation sets and never trained on).
- Raises clear errors when called on unknown or unsupported datasets.
"""

from __future__ import annotations

from typing import Tuple, Union

import pandas as pd

from voxguard.utils.logging_utils import get_logger

logger = get_logger(__name__)

SUPPORTED_DATASETS = {"asvspoof2019", "wavefake", "in_the_wild"}


def _normalize_dataset_name(name: str) -> str:
    """Normalizes dataset name string."""
    clean = str(name).strip().lower().replace("-", "_")
    if clean in ("asvspoof", "asvspoof2019", "asvspoof_2019", "asv_2019"):
        return "asvspoof2019"
    if clean in ("wavefake", "wave_fake", "wf"):
        return "wavefake"
    if clean in ("in_the_wild", "itw", "inthewild"):
        return "in_the_wild"
    return clean


def get_asvspoof_splits(
    unified_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extracts the official train, dev, and eval partitions for ASVspoof 2019.

    CRITICAL: Does NOT shuffle or re-partition the data. Preserving official
    partitions ensures zero speaker or system leakage across train/dev/eval splits.

    Args:
        unified_df: DataFrame containing ASVspoof2019 records.

    Returns:
        tuple: (train_df, dev_df, eval_df) as separate DataFrames.

    Raises:
        ValueError: If input DataFrame does not contain valid ASVspoof2019 records or
                    if any of the required official splits (train, dev, eval) are missing.
    """
    if unified_df.empty:
        raise ValueError("Cannot extract ASVspoof splits from an empty DataFrame.")

    # Filter for ASVspoof 2019 records if dataset column exists
    if "dataset" in unified_df.columns:
        asv_df = unified_df[
            unified_df["dataset"].apply(_normalize_dataset_name) == "asvspoof2019"
        ].copy()
        if asv_df.empty:
            # If dataset column exists but no 'asvspoof2019' rows match, check if all rows are ASVspoof paths
            sample_path = str(unified_df["filepath"].iloc[0]).lower()
            if "asvspoof" in sample_path or "la_t_" in sample_path or "la_d_" in sample_path or "la_e_" in sample_path:
                asv_df = unified_df.copy()
            else:
                raise ValueError("No ASVspoof2019 records found in the provided DataFrame.")
    else:
        asv_df = unified_df.copy()

    # Determine splits using 'split' column or fallback to path heuristics
    if "split" in asv_df.columns and asv_df["split"].isin(["train", "dev", "eval"]).any():
        train_df = asv_df[asv_df["split"] == "train"].copy().reset_index(drop=True)
        dev_df = asv_df[asv_df["split"] == "dev"].copy().reset_index(drop=True)
        eval_df = asv_df[asv_df["split"] == "eval"].copy().reset_index(drop=True)
    else:
        # Infer split from filename conventions: LA_T_ (train), LA_D_ (dev), LA_E_ (eval)
        fps = asv_df["filepath"].astype(str).str.lower()
        train_mask = fps.str.contains("la_t_") | fps.str.contains("la_train") | fps.str.contains("/train/")
        dev_mask = fps.str.contains("la_d_") | fps.str.contains("la_dev") | fps.str.contains("/dev/")
        eval_mask = fps.str.contains("la_e_") | fps.str.contains("la_eval") | fps.str.contains("/eval/")

        train_df = asv_df[train_mask].copy().reset_index(drop=True)
        dev_df = asv_df[dev_mask].copy().reset_index(drop=True)
        eval_df = asv_df[eval_mask].copy().reset_index(drop=True)

    if len(train_df) == 0:
        raise ValueError("ASVspoof2019 official 'train' partition is missing or empty.")
    if len(dev_df) == 0:
        raise ValueError("ASVspoof2019 official 'dev' partition is missing or empty.")
    if len(eval_df) == 0:
        raise ValueError("ASVspoof2019 official 'eval' partition is missing or empty.")

    logger.info(
        f"Extracted official ASVspoof2019 splits: "
        f"train={len(train_df):,}, dev={len(dev_df):,}, eval={len(eval_df):,}"
    )

    return train_df, dev_df, eval_df


def get_eval_only_split(
    unified_df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Extracts an eval-only partition for datasets used strictly in out-of-domain evaluation
    (such as WaveFake and In-the-Wild).

    Args:
        unified_df: DataFrame containing dataset records.
        dataset_name: Name of the dataset ('wavefake' or 'in_the_wild').

    Returns:
        pd.DataFrame: Evaluation-only dataset slice.

    Raises:
        ValueError: If dataset is unknown or has no records in unified_df.
    """
    canon_name = _normalize_dataset_name(dataset_name)
    if canon_name not in ("wavefake", "in_the_wild"):
        raise ValueError(
            f"Dataset '{dataset_name}' is not configured as an eval-only dataset. "
            "Supported eval-only datasets: ['wavefake', 'in_the_wild']."
        )

    if "dataset" in unified_df.columns:
        df_subset = unified_df[
            unified_df["dataset"].apply(_normalize_dataset_name) == canon_name
        ].copy().reset_index(drop=True)
    else:
        df_subset = unified_df.copy().reset_index(drop=True)

    if df_subset.empty:
        raise ValueError(f"No records found for dataset '{dataset_name}' in the provided DataFrame.")

    # Mark split as eval
    df_subset["split"] = "eval"

    logger.info(f"Extracted eval-only split for '{canon_name}': {len(df_subset):,} utterances.")
    return df_subset


def get_dataset_splits(
    unified_df: pd.DataFrame,
    dataset_name: str,
) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    """
    Universal entrypoint to obtain official splits for any supported dataset.

    Args:
        unified_df: DataFrame with dataset records.
        dataset_name: Dataset name ('asvspoof2019', 'wavefake', 'in_the_wild').

    Returns:
        tuple[DataFrame, DataFrame, DataFrame] for ASVspoof2019 (train, dev, eval)
        DataFrame for WaveFake or In-the-Wild (eval-only).

    Raises:
        ValueError: If dataset_name has no known official split logic implemented.
    """
    canon_name = _normalize_dataset_name(dataset_name)

    if canon_name == "asvspoof2019":
        return get_asvspoof_splits(unified_df)
    elif canon_name in ("wavefake", "in_the_wild"):
        return get_eval_only_split(unified_df, canon_name)
    else:
        raise ValueError(
            f"No official split logic implemented for dataset: '{dataset_name}'. "
            f"Supported datasets: {sorted(list(SUPPORTED_DATASETS))}."
        )
