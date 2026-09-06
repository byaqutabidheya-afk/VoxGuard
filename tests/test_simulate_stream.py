#!/usr/bin/env python3
"""
test_simulate_stream.py — Unit tests for offline streaming simulation harness.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from scripts.simulate_stream import main
from voxguard import config
from voxguard.streaming.session import StreamingSession, simulate_stream


class _DummyDetector:
    def __init__(self, score: float = 0.8) -> None:
        self.score = score
        self.calls = 0

    def predict_waveform(self, waveform: np.ndarray, sr: int) -> dict:
        self.calls += 1
        return {"label": "synthetic", "probability_synthetic": self.score}


@pytest.fixture
def sample_wav(tmp_path: Path) -> Path:
    """Creates a temporary 3-second 16kHz sine wave audio file."""
    wav_path = tmp_path / "test_stream.wav"
    t = np.linspace(0, 3.0, 16000 * 3, dtype=np.float32)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(wav_path), sine, 16000)
    return wav_path


def test_simulate_stream_with_in_memory_waveform() -> None:
    """Tests simulate_stream with in-memory waveform and dummy detector."""
    detector = _DummyDetector(score=0.9)
    session = StreamingSession(
        detector=detector,
        chunk_seconds=1.0,
        overlap_seconds=0.5,
        flag_threshold=0.7,
    )

    waveform = np.ones(16000 * 2, dtype=np.float32)  # 2 seconds
    callback_calls = []

    def on_step(res: dict) -> None:
        callback_calls.append(res)

    summary = simulate_stream(
        audio=waveform,
        session=session,
        step_seconds=0.25,
        realtime=False,
        callback=on_step,
        sr=16000,
    )

    assert summary["total_duration"] == 2.0
    assert summary["flagged"] is True
    assert summary["seconds_to_flag"] is not None
    assert summary["seconds_to_flag"] <= 2.0
    assert summary["final_running_score"] > 0.0
    assert len(summary["step_results"]) == 8  # 2.0s / 0.25s = 8 steps
    assert len(callback_calls) == 8


def test_simulate_stream_flag_threshold_fallback() -> None:
    """Tests that simulate_stream falls back to config.STREAM_FLAG_THRESHOLD when None."""
    detector = _DummyDetector(score=0.5)
    session = StreamingSession(detector=detector, flag_threshold=None)

    assert session.flag_threshold == float(config.STREAM_FLAG_THRESHOLD)


def test_simulate_stream_empty_audio() -> None:
    """Tests simulate_stream behavior on empty audio."""
    session = StreamingSession(detector=_DummyDetector())
    summary = simulate_stream(
        audio=np.empty(0, dtype=np.float32), session=session, realtime=False
    )
    assert summary["total_duration"] == 0.0
    assert summary["final_running_score"] == 0.0
    assert summary["flagged"] is False
    assert summary["seconds_to_flag"] is None
    assert summary["step_results"] == []


def test_simulate_stream_cli_execution(sample_wav: Path, capsys) -> None:
    """Tests simulate_stream CLI execution with mock detector and command line arguments."""
    with patch(
        "voxguard.streaming.session.WeightedAverageDetector",
        return_value=_DummyDetector(score=0.85),
    ):
        with patch(
            "sys.argv",
            [
                "simulate_stream.py",
                "--audio_path",
                str(sample_wav),
                "--chunk_seconds",
                "1.0",
                "--overlap_seconds",
                "0.5",
                "--flag_threshold",
                "0.7",
                "--step_seconds",
                "0.5",
                "--no_sleep",
            ],
        ):
            main()

    captured = capsys.readouterr().out
    assert "VOXGUARD REAL-TIME STREAMING SIMULATION" in captured
    assert "Total Duration Processed: 3.00s" in captured
    assert "Synthetic Flag Triggered: True" in captured
    assert "Time to First Flag:" in captured
