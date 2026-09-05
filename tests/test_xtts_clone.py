"""
Unit tests for src/voxguard/synth/xtts_clone.py and scripts/generate_hindi_clones.py.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from scripts.generate_hindi_clones import generate_hindi_clones
from voxguard.synth.xtts_clone import (
    clone_voice,
    clone_voice_indic_tts,
    resolve_gpu_flag,
)


@pytest.fixture
def sample_clone_environment(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Creates temporary real metadata CSV and reference manifest."""
    root_dir = tmp_path

    # Reference file
    ref_dir = root_dir / "data" / "raw" / "hindi_hinglish" / "references"
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_file = ref_dir / "priya_ref.wav"
    sf.write(str(ref_file), np.zeros(16000 * 6, dtype=np.float32), 16000)

    # Real metadata CSV
    meta_dir = root_dir / "data" / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    real_csv = meta_dir / "hindi_hinglish_real.csv"
    pd.DataFrame([
        {
            "filepath": "data/raw/hindi_hinglish/real/priya_neutral_01.wav",
            "speaker_id": "priya",
            "category": "neutral",
            "sentence_id": 1,
            "sentence_text": "Yaar, aaj office mein bahut kaam tha.",
            "consent_confirmed": True,
        },
        {
            "filepath": "data/raw/hindi_hinglish/real/priya_scam_11.wav",
            "speaker_id": "priya",
            "category": "scam",
            "sentence_id": 11,
            "sentence_text": "Sir, aapka bank account block ho jaayega.",
            "consent_confirmed": True,
        },
    ]).to_csv(real_csv, index=False)

    # References manifest
    refs_csv = meta_dir / "xtts_references.csv"
    pd.DataFrame([
        {
            "speaker_id": "priya",
            "reference_path": "data/raw/hindi_hinglish/references/priya_ref.wav",
            "duration_seconds": 6.0,
            "consent_confirmed": True,
        },
    ]).to_csv(refs_csv, index=False)

    out_dir = root_dir / "data" / "raw" / "hindi_hinglish" / "synthetic"
    out_csv = meta_dir / "hindi_hinglish_synthetic.csv"

    return real_csv, refs_csv, out_dir, out_csv


def test_indic_tts_stub_raises_not_implemented() -> None:
    """Tests that Indic TTS stub clearly raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="AI4Bharat Indic TTS"):
        clone_voice_indic_tts("dummy_ref.wav", "Sample text", language="hi")


def test_resolve_gpu_flag_behavior() -> None:
    """Tests resolve_gpu_flag with CPU default and graceful fallback."""
    # When False -> always False
    assert resolve_gpu_flag(False) is False

    # When True but CUDA unavailable -> False (graceful fallback)
    with patch("torch.cuda.is_available", return_value=False):
        assert resolve_gpu_flag(True) is False

    # When True and CUDA available -> True
    with patch("torch.cuda.is_available", return_value=True):
        assert resolve_gpu_flag(True) is True


def test_clone_voice_missing_reference_raises_file_not_found(tmp_path: Path) -> None:
    """Tests that missing reference audio raises FileNotFoundError."""
    missing_ref = tmp_path / "nonexistent_ref.wav"
    with pytest.raises(FileNotFoundError):
        clone_voice(missing_ref, "Sample text")


def test_clone_voice_mocked_execution(tmp_path: Path) -> None:
    """Tests clone_voice function invoking TTS.api.TTS tts_to_file with use_gpu."""
    ref_audio = tmp_path / "ref.wav"
    sf.write(str(ref_audio), np.zeros(16000 * 5, dtype=np.float32), 16000)
    out_audio = tmp_path / "cloned.wav"

    mock_tts = MagicMock()

    def fake_tts_to_file(text, speaker_wav, language, file_path, split_sentences=True):
        # Create output file
        sf.write(file_path, np.zeros(16000 * 2, dtype=np.float32), 16000)

    mock_tts.tts_to_file.side_effect = fake_tts_to_file

    with patch("voxguard.synth.xtts_clone.get_xtts_model", return_value=mock_tts) as mock_get_model:
        res_path = clone_voice(
            reference_audio_path=ref_audio,
            text="Testing voice cloning",
            language="hi",
            output_path=out_audio,
            use_gpu=False,
        )

        assert res_path == str(out_audio.resolve())
        assert out_audio.exists()
        mock_get_model.assert_called_once_with(use_gpu=False)
        mock_tts.tts_to_file.assert_called_once_with(
            text="Testing voice cloning",
            speaker_wav=str(ref_audio.resolve()),
            language="hi",
            file_path=str(out_audio.resolve()),
            split_sentences=True,
        )


def test_generate_hindi_clones_batch(
    sample_clone_environment: tuple[Path, Path, Path, Path],
    tmp_path: Path,
) -> None:
    """Tests batch synthesis script generating matched pairs and metadata with GPU option."""
    real_csv, refs_csv, out_dir, out_csv = sample_clone_environment

    def fake_clone_voice(reference_audio_path, text, language, output_path, use_gpu=False, **kwargs):
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_p), np.zeros(16000 * 2, dtype=np.float32), 16000)
        return str(out_p)

    with patch("scripts.generate_hindi_clones.clone_voice", side_effect=fake_clone_voice) as mock_clone:
        df_synthetic = generate_hindi_clones(
            real_metadata_csv=real_csv,
            references_manifest_csv=refs_csv,
            output_dir=out_dir,
            output_metadata_csv=out_csv,
            language="hi",
            use_gpu=True,
            project_root=tmp_path,
        )

        assert len(df_synthetic) == 2
        assert out_csv.exists()
        assert list(df_synthetic.columns) == [
            "filepath",
            "speaker_id",
            "category",
            "sentence_id",
            "label",
            "generator",
        ]
        assert (df_synthetic["label"] == "synthetic").all()
        assert (df_synthetic["generator"] == "xtts_v2").all()

        # Check matched output filenames
        filenames = [Path(fp).name for fp in df_synthetic["filepath"]]
        assert "priya_neutral_01_clone.wav" in filenames
        assert "priya_scam_11_clone.wav" in filenames

        # Verify use_gpu was passed through
        assert mock_clone.call_count == 2
        assert mock_clone.call_args[1]["use_gpu"] is True
