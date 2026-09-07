"""Unit tests for LiveTranscriber (Phase 7 / Prompt 9.1)."""

import numpy as np
import pytest
from pathlib import Path

from voxguard.fusion.transcribe import LiveTranscriber


def test_transcribe_init():
    """Confirms LiveTranscriber initializes cleanly on CPU with int8."""
    transcriber = LiveTranscriber(model_size="tiny", device="cpu", compute_type="int8")
    assert transcriber.device == "cpu"
    assert transcriber.compute_type == "int8"
    assert transcriber.model_size == "tiny"


def test_transcribe_chunk_silent_and_empty():
    """Confirms transcribe_chunk handles silent and empty waveforms without error."""
    transcriber = LiveTranscriber(model_size="tiny", device="cpu", compute_type="int8")

    # Empty array
    assert transcriber.transcribe_chunk(np.array([], dtype=np.float32), 16000) == ""
    # None
    assert transcriber.transcribe_chunk(None, 16000) == ""
    # Silent zeros
    zeros = np.zeros(24000, dtype=np.float32)
    assert transcriber.transcribe_chunk(zeros, 16000) == ""
    # Very short chunk
    short = np.random.randn(100).astype(np.float32) * 0.01
    assert transcriber.transcribe_chunk(short, 16000) == ""


def test_transcribe_full_missing_file():
    """Confirms transcribe_full raises FileNotFoundError when given a non-existent path."""
    transcriber = LiveTranscriber(model_size="tiny", device="cpu", compute_type="int8")
    with pytest.raises(FileNotFoundError):
        transcriber.transcribe_full("path/does/not/exist.wav")


def test_transcribe_real_audio():
    """Confirms LiveTranscriber transcribes audio chunk and full file."""
    audio_file = Path("data/raw/hindi_hinglish/real/byaquta_neutral_09.wav")
    if not audio_file.exists():
        pytest.skip("Test audio file not present")

    transcriber = LiveTranscriber(model_size="tiny", device="cpu", compute_type="int8")
    text = transcriber.transcribe_full(audio_file)
    assert isinstance(text, str)
    assert len(text) > 0
