"""
Unit tests for src/voxguard/utils/preprocess.py and scripts/run_preprocessing.py.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from scripts.run_preprocessing import run_preprocessing
from voxguard.utils.audio_io import load_audio, save_audio
from voxguard.utils.preprocess import generate_unique_filename, preprocess_dataset


@pytest.fixture
def sample_raw_audio_dir(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    """Creates temporary raw audio files with different sample rates and silence."""
    raw_dir = tmp_path / "raw_audio"
    raw_dir.mkdir(parents=True, exist_ok=True)

    records = []
    # 1. Normal sine wave at 44100 Hz with leading and trailing silence
    sr_44k = 44100
    t = np.linspace(0, 1.0, sr_44k, endpoint=False, dtype=np.float32)
    sine_44k = 0.5 * np.sin(2 * np.pi * 440 * t)
    silence = np.zeros(int(sr_44k * 0.3), dtype=np.float32)
    audio_with_silence = np.concatenate([silence, sine_44k, silence])
    p1 = raw_dir / "clip1.wav"
    sf.write(str(p1), audio_with_silence, sr_44k)
    records.append({
        "filepath": str(p1),
        "label": "real",
        "dataset": "asvspoof2019",
        "split": "train",
    })

    # 2. Stereo audio at 22050 Hz
    sr_22k = 22050
    t2 = np.linspace(0, 0.5, sr_22k // 2, endpoint=False, dtype=np.float32)
    stereo_audio = np.stack([0.3 * np.sin(2 * np.pi * 300 * t2), 0.3 * np.cos(2 * np.pi * 300 * t2)], axis=1)
    p2 = raw_dir / "clip2.wav"
    sf.write(str(p2), stereo_audio, sr_22k)
    records.append({
        "filepath": str(p2),
        "label": "synthetic",
        "dataset": "wavefake",
        "split": "unknown",
        "generator": "melgan",
    })

    # 3. Near-silent audio to test zero-length safety guard
    silent_audio = np.zeros(1600, dtype=np.float32)
    p3 = raw_dir / "clip3_silent.wav"
    sf.write(str(p3), silent_audio, 16000)
    records.append({
        "filepath": str(p3),
        "label": "synthetic",
        "dataset": "in_the_wild",
        "split": "unknown",
    })

    meta_df = pd.DataFrame(records)
    return raw_dir, meta_df


def test_unique_filename_generation() -> None:
    """Tests that filenames include dataset name and generator without collision."""
    row1 = pd.Series({"filepath": "D:/raw/audio/clip1.wav", "generator": "melgan"})
    fn1 = generate_unique_filename(row1, "wavefake")
    assert fn1 == "wavefake_melgan_clip1.wav"

    row2 = pd.Series({"filepath": "D:/raw/flac/LA_T_001.flac"})
    fn2 = generate_unique_filename(row2, "asvspoof2019")
    assert fn2 == "asvspoof2019_LA_T_001.wav"


def test_preprocess_dataset_transforms(
    tmp_path: Path, sample_raw_audio_dir: tuple[Path, pd.DataFrame]
) -> None:
    """Tests resampling, mono conversion, silence trimming, and relative path recording."""
    _, meta_df = sample_raw_audio_dir
    proc_dir = tmp_path / "processed"

    result_df = preprocess_dataset(
        metadata_df=meta_df,
        output_dir=proc_dir,
        target_sr=16000,
        trim_silence=True,
        project_root=tmp_path,
    )

    assert "processed_path" in result_df.columns
    assert len(result_df) == len(meta_df)

    # Check each processed output
    for _, row in result_df.iterrows():
        rel_path = row["processed_path"]
        assert rel_path != ""
        full_path = tmp_path / rel_path
        assert full_path.exists()

        # Load processed audio and verify 16kHz mono
        waveform, sr = load_audio(full_path, target_sr=16000)
        assert sr == 16000
        assert waveform.ndim == 1
        assert len(waveform) > 0  # Guard against zero length!


def test_preprocess_dataset_resumability(
    tmp_path: Path, sample_raw_audio_dir: tuple[Path, pd.DataFrame]
) -> None:
    """Tests that preprocessing skips already processed files on subsequent runs."""
    _, meta_df = sample_raw_audio_dir
    proc_dir = tmp_path / "processed"

    # First run
    df1 = preprocess_dataset(
        metadata_df=meta_df,
        output_dir=proc_dir,
        target_sr=16000,
        project_root=tmp_path,
    )

    # Record mtime of one file
    first_file = tmp_path / df1["processed_path"].iloc[0]
    mtime_before = first_file.stat().st_mtime

    # Second run should skip
    df2 = preprocess_dataset(
        metadata_df=meta_df,
        output_dir=proc_dir,
        target_sr=16000,
        project_root=tmp_path,
    )

    mtime_after = first_file.stat().st_mtime
    assert mtime_before == mtime_after
    assert list(df1["processed_path"]) == list(df2["processed_path"])


def test_dry_run_does_not_overwrite_unified(tmp_path: Path) -> None:
    """Tests that --dry_run processes limited samples without overwriting unified.csv."""
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir = tmp_path / "processed"

    # Create 30 mock files for dataset1
    records = []
    for i in range(30):
        audio_p = raw_dir / f"clip_{i}.wav"
        sf.write(str(audio_p), np.zeros(1600, dtype=np.float32), 16000)
        records.append({
            "filepath": str(audio_p),
            "label": "real",
        })
    pd.DataFrame(records).to_csv(meta_dir / "asvspoof2019.csv", index=False)

    # Execute dry run with dry_run_count=5
    unified_csv = meta_dir / "unified.csv"
    assert not unified_csv.exists()

    df_dry = run_preprocessing(
        dataset_names=["asvspoof2019"],
        output_dir=proc_dir,
        target_sr=16000,
        dry_run=True,
        dry_run_count=5,
        force_rebuild=True,
    )

    assert len(df_dry) == 5
    # Crucial v4 check: unified.csv must NOT have been created/overwritten by dry run!
    assert not unified_csv.exists()
