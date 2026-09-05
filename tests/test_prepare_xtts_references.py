"""
Unit tests for scripts/prepare_xtts_references.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from scripts.prepare_xtts_references import (
    find_speaker_reference_file,
    prepare_xtts_references,
    validate_and_clean_reference,
)


@pytest.fixture
def sample_reference_environment(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Creates temporary real metadata CSV and reference audio files."""
    ref_dir = tmp_path / "reference_inputs"
    ref_dir.mkdir(parents=True, exist_ok=True)

    sr = 16000
    # 1. Valid reference clip (8.0s) for speaker 'alice'
    t = np.linspace(0, 8.0, int(sr * 8.0), endpoint=False, dtype=np.float32)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)
    # Add 0.5s silence at start and end (total 9.0s)
    silence = np.zeros(int(sr * 0.5), dtype=np.float32)
    alice_audio = np.concatenate([silence, sine, silence])
    sf.write(str(ref_dir / "alice_reference.wav"), alice_audio, sr)

    # 2. Too short reference clip (3.0s) for speaker 'bob'
    t_short = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False, dtype=np.float32)
    bob_audio = 0.5 * np.sin(2 * np.pi * 440 * t_short)
    sf.write(str(ref_dir / "bob_ref.wav"), bob_audio, sr)

    # 3. Valid reference clip (7.0s) for speaker 'charlie' (but consent=False)
    t_charlie = np.linspace(0, 7.0, int(sr * 7.0), endpoint=False, dtype=np.float32)
    charlie_audio = 0.5 * np.sin(2 * np.pi * 440 * t_charlie)
    sf.write(str(ref_dir / "charlie_sample.wav"), charlie_audio, sr)

    # Metadata CSV
    meta_path = tmp_path / "hindi_hinglish_real.csv"
    pd.DataFrame([
        {"speaker_id": "alice", "consent_confirmed": True},
        {"speaker_id": "bob", "consent_confirmed": True},
        {"speaker_id": "charlie", "consent_confirmed": False},
    ]).to_csv(meta_path, index=False)

    out_dir = tmp_path / "output_references"
    manifest_path = tmp_path / "xtts_references.csv"

    return meta_path, ref_dir, out_dir, manifest_path


def test_find_speaker_reference_file(tmp_path: Path) -> None:
    """Tests speaker reference audio file discovery across different naming conventions."""
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()

    (ref_dir / "priya_reference.wav").write_bytes(b"dummy")
    (ref_dir / "rahul_ref.wav").write_bytes(b"dummy")
    (ref_dir / "amit.wav").write_bytes(b"dummy")

    assert find_speaker_reference_file("priya", ref_dir) == ref_dir / "priya_reference.wav"
    assert find_speaker_reference_file("rahul", ref_dir) == ref_dir / "rahul_ref.wav"
    assert find_speaker_reference_file("amit", ref_dir) == ref_dir / "amit.wav"
    assert find_speaker_reference_file("unknown_spk", ref_dir) is None


def test_validate_and_clean_reference_duration_filter(tmp_path: Path) -> None:
    """Tests duration threshold validation (< 5s rejected, 5-15s accepted)."""
    sr = 16000

    # Short audio (2.0s) -> should be rejected
    short_p = tmp_path / "short.wav"
    sf.write(str(short_p), np.sin(np.linspace(0, 2, sr * 2, dtype=np.float32)), sr)
    wf, dur, err = validate_and_clean_reference(short_p, target_sr=sr, min_duration=5.0)
    assert wf is None
    assert "too short" in err.lower()

    # Valid audio (6.0s) -> should be accepted
    valid_p = tmp_path / "valid.wav"
    sf.write(str(valid_p), np.sin(np.linspace(0, 6, sr * 6, dtype=np.float32)), sr)
    wf, dur, err = validate_and_clean_reference(valid_p, target_sr=sr, min_duration=5.0)
    assert wf is not None
    assert 5.5 < dur <= 6.5


def test_prepare_xtts_references_pipeline(
    sample_reference_environment: tuple[Path, Path, Path, Path],
    tmp_path: Path,
) -> None:
    """Tests end-to-end reference preparation, consent filtering, and manifest generation."""
    meta_path, ref_dir, out_dir, manifest_path = sample_reference_environment

    df_manifest = prepare_xtts_references(
        real_metadata_csv=meta_path,
        reference_dir=ref_dir,
        output_dir=out_dir,
        output_manifest=manifest_path,
        min_duration=5.0,
        max_duration=15.0,
        project_root=tmp_path,
    )

    # 1. Manifest file must exist
    assert manifest_path.exists()
    assert len(df_manifest) == 1  # Only 'alice' is consented AND >= 5s

    # 2. Check alice entry
    alice_row = df_manifest.iloc[0]
    assert alice_row["speaker_id"] == "alice"
    assert bool(alice_row["consent_confirmed"]) is True
    assert 7.5 <= float(alice_row["duration_seconds"]) <= 8.5

    # 3. Output audio file exists
    ref_audio_file = tmp_path / str(alice_row["reference_path"])
    assert ref_audio_file.exists()
