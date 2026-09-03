"""
Unit tests for src/voxguard/utils/metadata.py.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from voxguard.utils.metadata import (
    CANONICAL_COLUMNS,
    load_unified_metadata,
    normalize_label,
    save_unified_metadata,
)


@pytest.fixture
def temp_metadata_dir(tmp_path: Path) -> Path:
    """Creates a temporary metadata directory with sample datasets."""
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # 1. ASVspoof 2019 sample
    asv_df = pd.DataFrame([
        {
            "filepath": "D:/VoxGuard/data/raw/asvspoof2019/LA/ASVspoof2019_LA_train/flac/LA_T_1000137.flac",
            "speaker_id": "LA_0079",
            "system_id": "-",
            "label": "bonafide",
        },
        {
            "filepath": "D:/VoxGuard/data/raw/asvspoof2019/LA/ASVspoof2019_LA_dev/flac/LA_D_1000265.flac",
            "speaker_id": "LA_0070",
            "system_id": "A01",
            "label": "spoof",
        },
        {
            "filepath": "D:/VoxGuard/data/raw/asvspoof2019/LA/ASVspoof2019_LA_eval/flac/LA_E_1000000.flac",
            "speaker_id": "LA_0060",
            "system_id": "A10",
            "label": "spoof",
        },
    ])
    asv_df.to_csv(meta_dir / "asvspoof2019.csv", index=False)

    # 2. WaveFake full sample
    wf_records = []
    for i in range(100):
        wf_records.append({
            "filepath": f"D:/VoxGuard/data/raw/wavefake/audio_{i}.wav",
            "label": "bonafide" if i < 10 else "spoof",
            "generator": "original" if i < 10 else "melgan",
        })
    pd.DataFrame(wf_records).to_csv(meta_dir / "wavefake.csv", index=False)

    # 3. WaveFake subset sample
    wf_sub_records = wf_records[:20]
    pd.DataFrame(wf_sub_records).to_csv(meta_dir / "wavefake_subset.csv", index=False)

    # 4. In-the-Wild sample
    itw_df = pd.DataFrame([
        {
            "filepath": "D:/VoxGuard/data/raw/in_the_wild/speaker1_real.wav",
            "label": "bonafide",
            "speaker": "speaker1",
        },
        {
            "filepath": "D:/VoxGuard/data/raw/in_the_wild/speaker2_fake.wav",
            "label": "spoof",
            "speaker": "speaker2",
        },
    ])
    itw_df.to_csv(meta_dir / "in_the_wild.csv", index=False)

    return meta_dir


def test_normalize_label() -> None:
    """Tests label mapping from raw strings to canonical 'real' and 'synthetic'."""
    assert normalize_label("bonafide") == "real"
    assert normalize_label("bona-fide") == "real"
    assert normalize_label("real") == "real"
    assert normalize_label("authentic") == "real"
    assert normalize_label("original") == "real"

    assert normalize_label("spoof") == "synthetic"
    assert normalize_label("fake") == "synthetic"
    assert normalize_label("synthetic") == "synthetic"
    assert normalize_label("deepfake") == "synthetic"
    assert normalize_label("cloned") == "synthetic"

    # Numeric mappings
    assert normalize_label(0) == "real"
    assert normalize_label("0") == "real"
    assert normalize_label(1) == "synthetic"
    assert normalize_label("1") == "synthetic"

    with pytest.raises(ValueError):
        normalize_label("unknown_label_xyz")

    with pytest.raises(ValueError):
        normalize_label(None)


def test_load_unified_metadata_schema_and_null_checks(temp_metadata_dir: Path) -> None:
    """Tests that loaded metadata conforms to canonical schema with no null filepaths."""
    df = load_unified_metadata(
        ["asvspoof2019", "in_the_wild"],
        metadata_dir=temp_metadata_dir,
    )

    assert set(CANONICAL_COLUMNS).issubset(df.columns)
    assert not df["filepath"].isnull().any()
    assert not df["label"].isnull().any()
    assert df["label"].isin(["real", "synthetic"]).all()
    assert len(df) == 5

    # Check ASVspoof split inference
    asv_rows = df[df["dataset"] == "asvspoof2019"]
    splits = asv_rows["split"].tolist()
    assert "train" in splits
    assert "dev" in splits
    assert "eval" in splits


def test_wavefake_subset_default_and_override(temp_metadata_dir: Path) -> None:
    """Tests that WaveFake defaults to subset (20 rows) and honors use_full_wavefake (100 rows)."""
    # 1. Default should load wavefake_subset.csv (20 rows)
    df_sub = load_unified_metadata(["wavefake"], metadata_dir=temp_metadata_dir)
    assert len(df_sub) == 20

    # 2. use_full_wavefake=True should load wavefake.csv (100 rows)
    df_full = load_unified_metadata(
        ["wavefake"], use_full_wavefake=True, metadata_dir=temp_metadata_dir
    )
    assert len(df_full) == 100


def test_wavefake_subset_fallback_when_missing(temp_metadata_dir: Path) -> None:
    """Tests fallback to wavefake.csv if wavefake_subset.csv is removed."""
    (temp_metadata_dir / "wavefake_subset.csv").unlink()
    df = load_unified_metadata(["wavefake"], metadata_dir=temp_metadata_dir)
    assert len(df) == 100


def test_unified_cache_with_processed_path(temp_metadata_dir: Path) -> None:
    """
    Tests that load_unified_metadata loads from unified.csv directly if processed_path
    is present and complete, avoiding rebuild that would discard processed_path.
    """
    unified_csv = temp_metadata_dir / "unified.csv"

    # Create a cached unified.csv with processed_path
    cached_df = pd.DataFrame([
        {
            "filepath": "D:/VoxGuard/data/raw/itw/1.wav",
            "label": "real",
            "dataset": "in_the_wild",
            "split": "unknown",
            "processed_path": "D:/VoxGuard/data/processed/in_the_wild_1.wav",
        },
        {
            "filepath": "D:/VoxGuard/data/raw/itw/2.wav",
            "label": "synthetic",
            "dataset": "in_the_wild",
            "split": "unknown",
            "processed_path": "D:/VoxGuard/data/processed/in_the_wild_2.wav",
        },
    ])
    cached_df.to_csv(unified_csv, index=False)

    # Calling load_unified_metadata for in_the_wild should return the cached dataframe with processed_path
    df_loaded = load_unified_metadata(
        ["in_the_wild"],
        metadata_dir=temp_metadata_dir,
        unified_csv_path=unified_csv,
    )

    assert "processed_path" in df_loaded.columns
    assert len(df_loaded) == 2
    assert df_loaded["processed_path"].iloc[0] == "D:/VoxGuard/data/processed/in_the_wild_1.wav"

    # Calling with force_rebuild=True should rebuild from raw in_the_wild.csv (which lacks processed_path)
    df_rebuilt = load_unified_metadata(
        ["in_the_wild"],
        force_rebuild=True,
        metadata_dir=temp_metadata_dir,
        unified_csv_path=unified_csv,
    )
    assert "processed_path" not in df_rebuilt.columns or df_rebuilt["processed_path"].isnull().all()


def test_save_unified_metadata(tmp_path: Path) -> None:
    """Tests saving unified metadata to disk and validating output."""
    out_file = tmp_path / "saved_unified.csv"
    sample_df = pd.DataFrame([
        {
            "filepath": "D:/VoxGuard/data/raw/test1.wav",
            "label": "real",
            "dataset": "asvspoof2019",
            "split": "train",
        },
        {
            "filepath": "D:/VoxGuard/data/raw/test2.wav",
            "label": "synthetic",
            "dataset": "asvspoof2019",
            "split": "dev",
        },
    ])

    saved_path = save_unified_metadata(sample_df, output_path=out_file)
    assert saved_path.exists()

    reloaded = pd.read_csv(saved_path)
    assert len(reloaded) == 2
    assert list(reloaded.columns) == CANONICAL_COLUMNS
