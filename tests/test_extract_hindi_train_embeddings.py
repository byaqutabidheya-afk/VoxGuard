"""
Unit tests for scripts/extract_hindi_train_embeddings.py.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from scripts.extract_hindi_train_embeddings import extract_hindi_embeddings
from voxguard.embeddings.cache import load_cached_embeddings


@pytest.fixture
def sample_extract_env(tmp_path: Path) -> tuple[Path, Path]:
    """Creates a sample Hindi track CSV with temporary audio files."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    records = []
    # 2 speakers (byaquta, soumya) x 2 categories x 2 sentences
    for spk in ["byaquta", "soumya"]:
        for cat in ["neutral", "scam"]:
            for i in range(1, 3):
                p_real = audio_dir / f"{spk}_{cat}_{i:02d}.wav"
                p_synth = audio_dir / f"{spk}_{cat}_{i:02d}_clone.wav"

                sf.write(str(p_real), np.zeros(16000 * 2, dtype=np.float32), 16000)
                sf.write(str(p_synth), np.zeros(16000 * 2, dtype=np.float32), 16000)

                records.append({
                    "filepath": str(p_real),
                    "speaker_id": spk,
                    "category": cat,
                    "sentence_id": i,
                    "label": "real",
                    "dataset": "hindi_hinglish",
                })
                records.append({
                    "filepath": str(p_synth),
                    "speaker_id": spk,
                    "category": cat,
                    "sentence_id": i,
                    "label": "synthetic",
                    "dataset": "hindi_hinglish",
                })

    track_csv = tmp_path / "hindi_hinglish_track.csv"
    pd.DataFrame(records).to_csv(track_csv, index=False)

    out_dir = tmp_path / "embeddings"
    return track_csv, out_dir


def test_extract_hindi_embeddings_pipeline(
    sample_extract_env: tuple[Path, Path]
) -> None:
    """Tests extract_hindi_embeddings execution and cache creation with mocked models."""
    track_csv, out_dir = sample_extract_env

    mock_extractor = MagicMock()
    mock_extractor.extract_batch.side_effect = lambda waveforms: np.ones(
        (len(waveforms), 768), dtype=np.float32
    )

    with patch(
        "scripts.extract_hindi_train_embeddings.EmbeddingExtractor",
        return_value=mock_extractor,
    ):
        extract_hindi_embeddings(
            track_csv=track_csv,
            output_dir=out_dir,
            holdout_speaker="soumya",
            model_name="facebook/wav2vec2-base",
            extract_eval=True,
            extract_prosody=True,
            extract_wavlm=False,
            force=True,
        )

    # 1. Check generated files
    train_npy = out_dir / "wav2vec2_hindi_train.npy"
    eval_npy = out_dir / "wav2vec2_hindi_eval.npy"
    prosody_train_npy = out_dir / "prosody_hindi_train.npy"
    prosody_eval_npy = out_dir / "prosody_hindi_eval.npy"

    assert train_npy.exists()
    assert eval_npy.exists()
    assert prosody_train_npy.exists()
    assert prosody_eval_npy.exists()

    # 2. Check shapes: Train = 4 real + 4 synth = 8 clips (byaquta), Eval = 8 clips (soumya)
    emb_tr, meta_tr = load_cached_embeddings(train_npy)
    emb_ev, meta_ev = load_cached_embeddings(eval_npy)

    assert emb_tr.shape == (8, 768)
    assert emb_ev.shape == (8, 768)
    assert len(meta_tr) == 8
    assert len(meta_ev) == 8
    assert (meta_tr["label"] == "real").sum() == 4
    assert (meta_tr["label"] == "synthetic").sum() == 4
