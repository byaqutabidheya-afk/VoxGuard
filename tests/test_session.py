"""Tests for the streaming session orchestrator."""

from __future__ import annotations

import numpy as np

from voxguard.streaming.session import StreamingSession


class _SequencedDetector:
    def __init__(self, scores: list[float]) -> None:
        self._scores = iter(scores)
        self.calls = 0

    def predict_waveform(self, waveform: np.ndarray, sr: int) -> dict:
        self.calls += 1
        return {
            "label": "synthetic",
            "probability_synthetic": next(self._scores),
        }


def test_streaming_session_tracks_running_score_flag_and_reset(monkeypatch) -> None:
    session = StreamingSession(
        detector=_SequencedDetector([0.1, 0.1, 0.9, 0.9, 0.9, 0.2]),
        chunk_seconds=1.0,
        overlap_seconds=0.0,
        alpha=0.5,
        flag_threshold=0.7,
        consecutive_flags_required=1,
    )

    frame = np.ones(16000, dtype=np.float32)

    first = session.push_audio(frame, 16000)
    second = session.push_audio(frame, 16000)
    third = session.push_audio(frame, 16000)
    fourth = session.push_audio(frame, 16000)
    fifth = session.push_audio(frame, 16000)

    assert first["running_score"] == 0.1
    assert second["running_score"] == 0.1
    assert third["running_score"] == 0.5
    assert fourth["running_score"] == 0.7
    assert fifth["running_score"] == 0.8

    assert first["flagged"] is False
    assert second["flagged"] is False
    assert third["flagged"] is False
    assert fourth["flagged"] is True
    assert fifth["flagged"] is True
    assert fourth["seconds_to_flag"] == 4.0
    assert fifth["seconds_to_flag"] == 4.0
    assert first["seconds_since_start"] == 1.0
    assert second["seconds_since_start"] == 2.0
    assert third["seconds_since_start"] == 3.0
    assert fourth["seconds_since_start"] == 4.0
    assert fifth["seconds_since_start"] == 5.0

    session.reset()

    after_reset = session.push_audio(frame, 16000)
    assert after_reset["running_score"] == 0.2
    assert after_reset["flagged"] is False
    assert after_reset["seconds_since_start"] == 1.0
    assert after_reset["seconds_to_flag"] is None


def test_streaming_session_consecutive_flags_spike_filtering() -> None:
    """Tests that transient spikes do not flag until sustained for consecutive_flags_required."""
    # Sequence of scores:
    # 1. 0.1 (EMA ~ 0.1) -> OK (count=0)
    # 2. 0.9 (EMA = 0.9) -> OK (count=1, needed 3)
    # 3. 0.9 (EMA = 0.9) -> OK (count=2, needed 3)
    # 4. 0.1 (EMA = 0.1) -> OK (count resets to 0)
    # 5. 0.9 (EMA = 0.9) -> OK (count=1)
    # 6. 0.9 (EMA = 0.9) -> OK (count=2)
    # 7. 0.9 (EMA = 0.9) -> FLAGGED (count=3, seconds_to_flag=7.0s)
    session = StreamingSession(
        detector=_SequencedDetector([0.1, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9]),
        chunk_seconds=1.0,
        overlap_seconds=0.0,
        alpha=1.0,  # alpha=1.0 makes running score equal latest score directly
        flag_threshold=0.6,
        consecutive_flags_required=3,
    )

    frame = np.ones(16000, dtype=np.float32)

    r1 = session.push_audio(frame, 16000)
    assert r1["flagged"] is False
    assert r1["seconds_to_flag"] is None

    r2 = session.push_audio(frame, 16000)
    assert r2["flagged"] is False
    assert r2["seconds_to_flag"] is None

    r3 = session.push_audio(frame, 16000)
    assert r3["flagged"] is False  # 2nd consecutive, not yet 3
    assert r3["seconds_to_flag"] is None

    r4 = session.push_audio(frame, 16000)
    assert r4["flagged"] is False  # dropped, count resets
    assert r4["seconds_to_flag"] is None

    r5 = session.push_audio(frame, 16000)
    assert r5["flagged"] is False

    r6 = session.push_audio(frame, 16000)
    assert r6["flagged"] is False

    r7 = session.push_audio(frame, 16000)
    assert r7["flagged"] is True  # 3rd consecutive!
    assert r7["seconds_to_flag"] == 7.0


def test_streaming_session_dynamic_sample_rate() -> None:
    """Verifies that StreamingSession captures incoming sample rate and scales buffer."""
    import pytest

    session = StreamingSession(
        detector=_SequencedDetector([0.8]),
        chunk_seconds=1.5,
        overlap_seconds=0.5,
    )
    assert session.sample_rate is None

    # Push 48000 Hz frame
    frame_48k = np.ones(48000, dtype=np.float32)  # 1.0s of 48kHz audio
    session.push_audio(frame_48k, 48000)

    assert session.sample_rate == 48000
    assert session.buffer.sample_rate == 48000
    assert session.buffer.chunk_samples == 72000  # 1.5s * 48000
    assert session.buffer.stride_samples == 48000  # 1.0s * 48000


def test_streaming_session_sample_rate_mismatch_error() -> None:
    """Verifies that changing sample rate mid-session raises ValueError."""
    import pytest

    session = StreamingSession(
        detector=_SequencedDetector([0.5, 0.5]),
        chunk_seconds=1.0,
        sample_rate=16000,
    )

    frame_16k = np.ones(16000, dtype=np.float32)
    session.push_audio(frame_16k, 16000)

    frame_48k = np.ones(48000, dtype=np.float32)
    with pytest.raises(ValueError, match="Sample rate mismatch"):
        session.push_audio(frame_48k, 48000)


