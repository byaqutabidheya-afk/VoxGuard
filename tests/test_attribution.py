"""Tests for voxguard.explain.attribution.windowed_attribution.

All tests use lightweight stub detectors — no model weights are loaded.
This keeps the suite fast and isolates the attribution logic from the
detector implementations.
"""

from __future__ import annotations

import numpy as np
import pytest

from voxguard.explain.attribution import windowed_attribution

SR = 16_000  # matches config.SAMPLE_RATE


# ---------------------------------------------------------------------------
# Stub detectors
# ---------------------------------------------------------------------------


class _ConstDetector:
    """Always returns a fixed P(synthetic)."""

    def __init__(self, prob: float = 0.42) -> None:
        self.prob = float(prob)
        self.call_count = 0

    def predict_waveform(self, waveform: np.ndarray, sr: int) -> dict:
        self.call_count += 1
        return {"probability_synthetic": self.prob, "label": "synthetic"}


class _TimeDetector:
    """Returns the mean absolute value of the window as P(synthetic).

    This lets tests verify that each window corresponds to the expected
    audio segment rather than just that some score was produced.
    """

    def predict_waveform(self, waveform: np.ndarray, sr: int) -> dict:
        prob = float(np.mean(np.abs(waveform.astype(np.float64))))
        return {"probability_synthetic": min(prob, 1.0), "label": "real"}


class _RaisingDetector:
    """Always raises RuntimeError."""

    def predict_waveform(self, waveform: np.ndarray, sr: int) -> dict:
        raise RuntimeError("detector exploded")


class _NoneDetector:
    """Returns probability_synthetic=None (mimics detector's own silence gate)."""

    def predict_waveform(self, waveform: np.ndarray, sr: int) -> dict:
        return {"probability_synthetic": None, "label": "inconclusive"}


def _sine(freq: float = 440.0, duration: float = 2.0, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _noise(duration: float = 2.0, sr: int = SR, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(sr * duration)).astype(np.float32) * 0.1


# ---------------------------------------------------------------------------
# Return-type and shape contracts
# ---------------------------------------------------------------------------


def test_returns_tuple_of_two_arrays() -> None:
    result = windowed_attribution(_sine(), SR, _ConstDetector())
    assert isinstance(result, tuple) and len(result) == 2
    scores, times = result
    assert isinstance(scores, np.ndarray)
    assert isinstance(times, np.ndarray)


def test_scores_and_times_same_length() -> None:
    scores, times = windowed_attribution(_sine(), SR, _ConstDetector())
    assert len(scores) == len(times)


def test_scores_dtype_float64() -> None:
    scores, _ = windowed_attribution(_sine(), SR, _ConstDetector())
    assert scores.dtype == np.float64


def test_times_dtype_float64() -> None:
    _, times = windowed_attribution(_sine(), SR, _ConstDetector())
    assert times.dtype == np.float64


# ---------------------------------------------------------------------------
# Window count and timing
# ---------------------------------------------------------------------------


def test_window_count_non_overlapping() -> None:
    """stride == window → no overlap → floor(duration / window) windows."""
    duration = 2.0
    window = 0.5
    wav = _sine(duration=duration)
    scores, times = windowed_attribution(
        wav, SR, _ConstDetector(), window_seconds=window, stride_seconds=window
    )
    expected = int(duration / window)  # 4
    assert len(scores) == expected


def test_window_count_50pct_overlap() -> None:
    """stride = window/2 → (duration - window) / stride + 1 windows."""
    duration = 2.0
    window = 0.5
    stride = 0.25
    wav = _sine(duration=duration)
    scores, times = windowed_attribution(
        wav, SR, _ConstDetector(), window_seconds=window, stride_seconds=stride
    )
    # (2.0 - 0.5) / 0.25 + 1 = 7
    assert len(scores) == 7


def test_times_start_at_zero() -> None:
    _, times = windowed_attribution(_sine(), SR, _ConstDetector())
    assert times[0] == pytest.approx(0.0)


def test_times_are_non_decreasing() -> None:
    _, times = windowed_attribution(_sine(), SR, _ConstDetector())
    assert np.all(np.diff(times) > 0)


def test_times_step_equals_stride() -> None:
    stride = 0.25
    _, times = windowed_attribution(
        _sine(), SR, _ConstDetector(), stride_seconds=stride
    )
    diffs = np.diff(times)
    assert np.allclose(diffs, stride, atol=1 / SR)  # within one sample


def test_times_within_audio_duration() -> None:
    duration = 2.0
    wav = _sine(duration=duration)
    _, times = windowed_attribution(wav, SR, _ConstDetector())
    assert float(times[-1]) < duration


# ---------------------------------------------------------------------------
# Score values
# ---------------------------------------------------------------------------


def test_const_detector_all_scores_equal() -> None:
    prob = 0.77
    scores, _ = windowed_attribution(_sine(), SR, _ConstDetector(prob))
    non_nan = scores[~np.isnan(scores)]
    assert len(non_nan) > 0
    assert np.allclose(non_nan, prob)


def test_scores_in_unit_interval() -> None:
    scores, _ = windowed_attribution(_noise(), SR, _ConstDetector(0.55))
    non_nan = scores[~np.isnan(scores)]
    assert np.all(non_nan >= 0.0)
    assert np.all(non_nan <= 1.0)


def test_each_window_scores_its_own_segment() -> None:
    """_TimeDetector returns mean(|window|); a constant waveform of value c
    should return c for every window."""
    c = 0.3
    wav = np.full(SR * 2, c, dtype=np.float32)
    scores, _ = windowed_attribution(wav, SR, _TimeDetector())
    non_nan = scores[~np.isnan(scores)]
    assert np.allclose(non_nan, c, atol=1e-5)


# ---------------------------------------------------------------------------
# Silence gate behaviour
# ---------------------------------------------------------------------------


def test_silent_windows_produce_nan() -> None:
    """A waveform of zeros should give all-NaN scores (silence gate fires)."""
    silence = np.zeros(SR * 2, dtype=np.float32)
    scores, _ = windowed_attribution(silence, SR, _ConstDetector())
    assert np.all(np.isnan(scores))


def test_mixed_silence_and_speech_partial_nan() -> None:
    """First second silent, second second speech → first half NaN, second scored."""
    wav = np.concatenate([
        np.zeros(SR, dtype=np.float32),   # silence
        _sine(duration=1.0),               # speech
    ])
    scores, times = windowed_attribution(
        wav, SR, _ConstDetector(), window_seconds=0.5, stride_seconds=0.5
    )
    # Windows entirely in the silent first second should be NaN
    silent_mask = times < 0.5
    assert np.all(np.isnan(scores[silent_mask]))
    # At least some windows in the speech second should be scored
    speech_mask = times >= 1.0
    assert np.any(~np.isnan(scores[speech_mask]))


# ---------------------------------------------------------------------------
# Detector error handling
# ---------------------------------------------------------------------------


def test_raising_detector_produces_all_nan() -> None:
    """If the detector always raises, all scores should be NaN (not propagated)."""
    scores, _ = windowed_attribution(_noise(), SR, _RaisingDetector())
    assert np.all(np.isnan(scores))


def test_none_prob_detector_produces_all_nan() -> None:
    """detector returning probability_synthetic=None → NaN, not an error."""
    scores, _ = windowed_attribution(_noise(), SR, _NoneDetector())
    assert np.all(np.isnan(scores))


# ---------------------------------------------------------------------------
# Stereo / multi-channel downmix
# ---------------------------------------------------------------------------


def test_stereo_channels_first_accepted() -> None:
    """Shape (2, samples) should be accepted and downmixed to mono."""
    stereo = np.stack([_sine(440), _sine(880)], axis=0)
    scores, times = windowed_attribution(stereo, SR, _ConstDetector())
    assert len(scores) > 0


def test_stereo_samples_first_accepted() -> None:
    """Shape (samples, 2) should be accepted and downmixed to mono."""
    stereo = np.stack([_sine(440), _sine(880)], axis=1)
    scores, times = windowed_attribution(stereo, SR, _ConstDetector())
    assert len(scores) > 0


# ---------------------------------------------------------------------------
# Edge cases: very short waveform
# ---------------------------------------------------------------------------


def test_waveform_shorter_than_one_window_returns_empty() -> None:
    """A waveform shorter than window_seconds should return empty arrays."""
    tiny = _sine(duration=0.1)   # 0.1 s < default window 0.5 s
    scores, times = windowed_attribution(tiny, SR, _ConstDetector())
    assert len(scores) == 0
    assert len(times) == 0


def test_waveform_exactly_one_window_returns_one_score() -> None:
    exactly_one = _sine(duration=0.5)
    scores, times = windowed_attribution(
        exactly_one, SR, _ConstDetector(), window_seconds=0.5, stride_seconds=0.25
    )
    assert len(scores) == 1


# ---------------------------------------------------------------------------
# Detector call count
# ---------------------------------------------------------------------------


def test_detector_called_once_per_non_silent_window() -> None:
    """The detector must be called exactly once per non-silent window."""
    detector = _ConstDetector()
    wav = _sine(duration=2.0)  # no silence
    scores, _ = windowed_attribution(
        wav, SR, detector, window_seconds=0.5, stride_seconds=0.5
    )
    n_windows = len(scores)
    assert detector.call_count == n_windows


# ---------------------------------------------------------------------------
# Validation / error conditions
# ---------------------------------------------------------------------------


def test_no_predict_waveform_raises_type_error() -> None:
    class _BadDetector:
        pass

    with pytest.raises(TypeError, match="predict_waveform"):
        windowed_attribution(_sine(), SR, _BadDetector())


def test_empty_waveform_raises_value_error() -> None:
    with pytest.raises(ValueError, match="empty"):
        windowed_attribution(np.array([], dtype=np.float32), SR, _ConstDetector())


def test_negative_sr_raises_value_error() -> None:
    with pytest.raises(ValueError, match="sr must be positive"):
        windowed_attribution(_sine(), -1, _ConstDetector())


def test_zero_window_seconds_raises_value_error() -> None:
    with pytest.raises(ValueError, match="window_seconds"):
        windowed_attribution(_sine(), SR, _ConstDetector(), window_seconds=0)


def test_zero_stride_raises_value_error() -> None:
    with pytest.raises(ValueError, match="stride_seconds"):
        windowed_attribution(_sine(), SR, _ConstDetector(), stride_seconds=0)


def test_stride_larger_than_window_raises_value_error() -> None:
    with pytest.raises(ValueError, match="stride_seconds.*<=.*window_seconds"):
        windowed_attribution(
            _sine(), SR, _ConstDetector(), window_seconds=0.5, stride_seconds=1.0
        )


def test_3d_waveform_raises_value_error() -> None:
    with pytest.raises(ValueError, match="1-D or 2-D"):
        windowed_attribution(
            np.zeros((2, 3, 4), dtype=np.float32), SR, _ConstDetector()
        )
