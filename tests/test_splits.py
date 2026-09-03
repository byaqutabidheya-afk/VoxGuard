"""
Unit tests for src/voxguard/utils/splits.py.
"""

from __future__ import annotations

import pandas as pd
import pytest

from voxguard.utils.splits import (
    get_asvspoof_splits,
    get_dataset_splits,
    get_eval_only_split,
)


@pytest.fixture
def mock_unified_df() -> pd.DataFrame:
    """Creates a sample multi-dataset unified DataFrame."""
    records = []

    # ASVspoof 2019 records
    for i in range(10):
        records.append({
            "filepath": f"D:/VoxGuard/data/raw/asvspoof2019/LA/ASVspoof2019_LA_train/flac/LA_T_{i:07d}.flac",
            "label": "real" if i % 2 == 0 else "synthetic",
            "dataset": "asvspoof2019",
            "split": "train",
        })
    for i in range(5):
        records.append({
            "filepath": f"D:/VoxGuard/data/raw/asvspoof2019/LA/ASVspoof2019_LA_dev/flac/LA_D_{i:07d}.flac",
            "label": "real" if i % 2 == 0 else "synthetic",
            "dataset": "asvspoof2019",
            "split": "dev",
        })
    for i in range(8):
        records.append({
            "filepath": f"D:/VoxGuard/data/raw/asvspoof2019/LA/ASVspoof2019_LA_eval/flac/LA_E_{i:07d}.flac",
            "label": "real" if i % 2 == 0 else "synthetic",
            "dataset": "asvspoof2019",
            "split": "eval",
        })

    # WaveFake records
    for i in range(12):
        records.append({
            "filepath": f"D:/VoxGuard/data/raw/wavefake/audio_{i}.wav",
            "label": "synthetic",
            "dataset": "wavefake",
            "split": "unknown",
        })

    # In-the-Wild records
    for i in range(7):
        records.append({
            "filepath": f"D:/VoxGuard/data/raw/in_the_wild/spk_{i}.wav",
            "label": "real",
            "dataset": "in_the_wild",
            "split": "unknown",
        })

    return pd.DataFrame(records)


def test_get_asvspoof_splits_counts_and_separation(mock_unified_df: pd.DataFrame) -> None:
    """Tests that ASVspoof splits return exact train, dev, and eval sets without overlap."""
    train_df, dev_df, eval_df = get_asvspoof_splits(mock_unified_df)

    assert len(train_df) == 10
    assert len(dev_df) == 5
    assert len(eval_df) == 8

    # Ensure no filepath overlap between splits
    train_fps = set(train_df["filepath"])
    dev_fps = set(dev_df["filepath"])
    eval_fps = set(eval_df["filepath"])

    assert train_fps.isdisjoint(dev_fps)
    assert train_fps.isdisjoint(eval_fps)
    assert dev_fps.isdisjoint(eval_fps)

    # Check that splits preserve dataset name
    assert (train_df["dataset"] == "asvspoof2019").all()
    assert (dev_df["dataset"] == "asvspoof2019").all()
    assert (eval_df["dataset"] == "asvspoof2019").all()


def test_get_asvspoof_splits_path_inference() -> None:
    """Tests split inference from filepaths when split column is missing or unknown."""
    df_no_split = pd.DataFrame([
        {"filepath": "D:/data/LA_T_1001.flac", "label": "real", "dataset": "asvspoof2019"},
        {"filepath": "D:/data/LA_D_1002.flac", "label": "synthetic", "dataset": "asvspoof2019"},
        {"filepath": "D:/data/LA_E_1003.flac", "label": "synthetic", "dataset": "asvspoof2019"},
    ])

    train_df, dev_df, eval_df = get_asvspoof_splits(df_no_split)
    assert len(train_df) == 1
    assert len(dev_df) == 1
    assert len(eval_df) == 1


def test_get_eval_only_splits(mock_unified_df: pd.DataFrame) -> None:
    """Tests that WaveFake and In-the-Wild are returned as full eval-only splits."""
    wf_eval = get_eval_only_split(mock_unified_df, "wavefake")
    assert len(wf_eval) == 12
    assert (wf_eval["split"] == "eval").all()

    itw_eval = get_eval_only_split(mock_unified_df, "in_the_wild")
    assert len(itw_eval) == 7
    assert (itw_eval["split"] == "eval").all()


def test_get_dataset_splits_dispatcher(mock_unified_df: pd.DataFrame) -> None:
    """Tests the universal get_dataset_splits router."""
    # ASVspoof returns 3 DataFrames
    asv_splits = get_dataset_splits(mock_unified_df, "asvspoof2019")
    assert isinstance(asv_splits, tuple)
    assert len(asv_splits) == 3

    # WaveFake returns a single DataFrame
    wf_split = get_dataset_splits(mock_unified_df, "wavefake")
    assert isinstance(wf_split, pd.DataFrame)
    assert len(wf_split) == 12

    # In-the-Wild returns a single DataFrame
    itw_split = get_dataset_splits(mock_unified_df, "in-the-wild")
    assert isinstance(itw_split, pd.DataFrame)
    assert len(itw_split) == 7


def test_unsupported_dataset_error(mock_unified_df: pd.DataFrame) -> None:
    """Tests that an unsupported dataset raises a clear ValueError."""
    with pytest.raises(ValueError, match="No official split logic implemented"):
        get_dataset_splits(mock_unified_df, "unsupported_xyz_dataset")


def test_missing_split_error() -> None:
    """Tests error handling when required splits are missing."""
    df_missing_dev = pd.DataFrame([
        {"filepath": "D:/data/LA_T_1001.flac", "label": "real", "dataset": "asvspoof2019", "split": "train"},
        {"filepath": "D:/data/LA_E_1003.flac", "label": "synthetic", "dataset": "asvspoof2019", "split": "eval"},
    ])
    with pytest.raises(ValueError, match="official 'dev' partition is missing"):
        get_asvspoof_splits(df_missing_dev)
