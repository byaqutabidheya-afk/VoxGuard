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




def _push_quarter_seconds(session: StreamingSession, seconds: float, amplitude: float = 1.0) -> list[dict]:
    """Pushes `seconds` of audio in 0.25 s steps (the simulate_stream cadence)."""
    step = np.full(4000, amplitude, dtype=np.float32)
    return [session.push_audio(step, 16000) for _ in range(int(seconds / 0.25))]


def test_consecutive_unit_rejects_unknown_value() -> None:
    import pytest

    with pytest.raises(ValueError):
        StreamingSession(detector=_SequencedDetector([0.5]), consecutive_unit="frames")


def test_consecutive_unit_updates_counts_score_updates_not_pushes() -> None:
    """With 1.5 s windows / 0.5 s overlap and 0.25 s pushes the score updates every 4th push.

    Legacy 'pushes' counting with N=2 flags on the second push after the first update (both pushes
    see the same score); 'updates' counting with N=2 must wait for a SECOND window.
    """
    kwargs = dict(chunk_seconds=1.5, overlap_seconds=0.5, alpha=1.0, flag_threshold=0.6, consecutive_flags_required=2)

    legacy = StreamingSession(detector=_SequencedDetector([0.9, 0.9, 0.9]), **kwargs)
    legacy_results = _push_quarter_seconds(legacy, 3.0)
    assert legacy_results[5]["seconds_to_flag"] is None          # push 6 (1.5 s): first update, count 1
    assert legacy_results[6]["seconds_to_flag"] == 1.75          # push 7: same score, count 2 -> flags

    updates = StreamingSession(detector=_SequencedDetector([0.9, 0.9, 0.9]), consecutive_unit="updates", **kwargs)
    results = _push_quarter_seconds(updates, 3.0)
    assert all(r["seconds_to_flag"] is None for r in results[:9])  # nothing before the second window (2.5 s)
    assert results[9]["seconds_to_flag"] == 2.5                    # push 10: second update -> flags
    assert results[9]["flagged"] is True


def test_consecutive_unit_updates_debounces_a_single_spike() -> None:
    """A lone high window must not flag with N=2 and must reset the streak when the score drops."""
    session = StreamingSession(
        detector=_SequencedDetector([0.9, 0.1, 0.9, 0.9]),
        chunk_seconds=1.5, overlap_seconds=0.5, alpha=1.0, flag_threshold=0.6,
        consecutive_flags_required=2, consecutive_unit="updates",
    )
    results = _push_quarter_seconds(session, 4.5)
    # windows complete at 1.5, 2.5, 3.5, 4.5 s -> scores 0.9, 0.1, 0.9, 0.9
    assert session._seconds_to_flag == 4.5
    assert results[5]["flagged"] is False    # 1.5 s: streak 1
    assert results[9]["flagged"] is False    # 2.5 s: dropped, streak reset
    assert results[13]["flagged"] is False   # 3.5 s: streak 1
    assert results[17]["flagged"] is True    # 4.5 s: streak 2


def test_consecutive_unit_updates_ignores_silent_windows() -> None:
    """A window skipped as silence makes no decision: it neither extends nor resets the streak."""
    session = StreamingSession(
        detector=_SequencedDetector([0.9, 0.9]),
        chunk_seconds=1.0, overlap_seconds=0.0, alpha=1.0, flag_threshold=0.6,
        consecutive_flags_required=2, consecutive_unit="updates",
    )
    loud = np.full(4000, 1.0, dtype=np.float32)
    quiet = np.zeros(4000, dtype=np.float32)

    for _ in range(4):                       # 0-1 s loud: window 1 scored 0.9 -> streak 1
        session.push_audio(loud, 16000)
    assert session._consecutive_flags == 1

    for _ in range(8):                       # 1-3 s silent: windows 2 and 3 skipped, no decision
        result = session.push_audio(quiet, 16000)
    assert session._consecutive_flags == 1   # not reset, not extended
    assert result["flagged"] is False

    for _ in range(4):                       # 3-4 s loud: window 4 scored 0.9 -> streak 2 -> flags
        result = session.push_audio(loud, 16000)
    assert result["flagged"] is True
    assert session._seconds_to_flag == 4.0


def test_default_detector_is_the_shared_streaming_production_detector(monkeypatch) -> None:
    """StreamingSession() with no detector takes the STREAMING family (not whole-clip), via the shared accessor."""
    import voxguard.streaming.session as session_module

    sentinel = _SequencedDetector([0.5])
    modes = []

    def fake_get(mode):
        modes.append(mode)
        return sentinel

    monkeypatch.setattr(session_module, "get_production_detector", fake_get)
    session = StreamingSession()
    assert session.detector is sentinel
    assert modes == ["streaming"]
    # legacy defaults are intentionally unchanged; production callers override them from config
    assert session.consecutive_unit == "pushes"
    assert session.consecutive_flags_required == 3


def test_explicit_detector_bypasses_the_default(monkeypatch) -> None:
    import voxguard.streaming.session as session_module

    def boom(mode):
        raise AssertionError("default detector must not be built when one is passed")

    monkeypatch.setattr(session_module, "get_production_detector", boom)
    detector = _SequencedDetector([0.5])
    assert StreamingSession(detector=detector).detector is detector
