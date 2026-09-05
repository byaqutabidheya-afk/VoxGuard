"""
Unit tests for scripts/merge_hindi_metadata.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.merge_hindi_metadata import (
    UNIFIED_SCHEMA_COLUMNS,
    merge_hindi_metadata,
)


@pytest.fixture
def sample_hindi_metadata_env(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Creates temporary real and synthetic metadata CSVs."""
    real_csv = tmp_path / "hindi_hinglish_real.csv"
    pd.DataFrame([
        {
            "filepath": "data/raw/hindi_hinglish/real/alice_neutral_01.wav",
            "speaker_id": "alice",
            "category": "neutral",
            "sentence_id": 1,
            "consent_confirmed": True,
        },
        {
            "filepath": "data/raw/hindi_hinglish/real/alice_scam_11.wav",
            "speaker_id": "alice",
            "category": "scam",
            "sentence_id": 11,
            "consent_confirmed": True,
        },
    ]).to_csv(real_csv, index=False)

    synth_csv = tmp_path / "hindi_hinglish_synthetic.csv"
    pd.DataFrame([
        {
            "filepath": "data/raw/hindi_hinglish/synthetic/alice_neutral_01_clone.wav",
            "speaker_id": "alice",
            "category": "neutral",
            "sentence_id": 1,
            "label": "synthetic",
            "generator": "xtts_v2",
        },
        {
            "filepath": "data/raw/hindi_hinglish/synthetic/alice_scam_11_clone.wav",
            "speaker_id": "alice",
            "category": "scam",
            "sentence_id": 11,
            "label": "synthetic",
            "generator": "xtts_v2",
        },
    ]).to_csv(synth_csv, index=False)

    out_track_csv = tmp_path / "hindi_hinglish_track.csv"
    return real_csv, synth_csv, out_track_csv


def test_merge_hindi_metadata_schema_and_counts(
    sample_hindi_metadata_env: tuple[Path, Path, Path]
) -> None:
    """Tests that merged track CSV has the exact unified schema and correct 1:1 counts."""
    real_csv, synth_csv, out_track_csv = sample_hindi_metadata_env

    df_track = merge_hindi_metadata(
        real_csv_path=real_csv,
        synthetic_csv_path=synth_csv,
        output_track_path=out_track_csv,
    )

    assert out_track_csv.exists()
    assert len(df_track) == 4
    assert list(df_track.columns) == UNIFIED_SCHEMA_COLUMNS

    # Check real / synthetic counts
    assert (df_track["label"] == "real").sum() == 2
    assert (df_track["label"] == "synthetic").sum() == 2
    assert (df_track["dataset"] == "hindi_hinglish").all()
    assert df_track["filepath"].notnull().all()


def test_merge_hindi_metadata_missing_files_raises_error(tmp_path: Path) -> None:
    """Tests that missing real or synthetic files raise FileNotFoundError."""
    missing_real = tmp_path / "nonexistent_real.csv"
    dummy_synth = tmp_path / "synth.csv"
    dummy_synth.write_text("filepath,speaker_id\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        merge_hindi_metadata(
            real_csv_path=missing_real,
            synthetic_csv_path=dummy_synth,
            output_track_path=tmp_path / "track.csv",
        )
