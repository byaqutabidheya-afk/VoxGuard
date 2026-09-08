"""Tests for voxguard.explain.overlay.

All tests use lightweight stub detectors — no model weights are loaded.
The NaN-handling tests are the most important: they verify that a clip
with trailing/leading silence (which produces NaN windows from
windowed_attribution) never causes a crash or a blank/all-NaN render.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from voxguard.explain.overlay import (
    _interpolate_scores_to_frames,
    render_explainability_overlay,
)

SR = 16_000


# ---------------------------------------------------------------------------
# Stub detectors
# ---------------------------------------------------------------------------


class _ConstDetector:
    def predict_waveform(self, w, sr):
        return {"probability_synthetic": 0.6, "label": "synthetic"}


class _ZeroDetector:
    def predict_waveform(self, w, sr):
        return {"probability_synthetic": 0.0, "label": "real"}


# ---------------------------------------------------------------------------
# _interpolate_scores_to_frames unit tests
# ---------------------------------------------------------------------------


def _frame_times(n: int, hop: int = 512, sr: int = SR) -> np.ndarray:
    return np.arange(n, dtype=np.float64) * hop / sr


def test_interp_no_nans_passthrough():
    """Clean scores should interpolate without any NaN in output."""
    timestamps = np.array([0.0, 0.5, 1.0, 1.5])
    scores = np.array([0.1, 0.5, 0.8, 0.3])
    frame_times = _frame_times(100)
    result, mask, valid = _interpolate_scores_to_frames(scores, timestamps, frame_times)
    assert valid is True
    assert not np.any(np.isnan(result))
    assert result.shape == frame_times.shape
    assert mask.shape == frame_times.shape


def test_interp_trailing_nan_no_crash():
    """Trailing NaN (trailing silence) must not crash or produce NaN output."""
    timestamps = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    scores = np.array([0.5, 0.6, 0.7, np.nan, np.nan])
    frame_times = _frame_times(150)
    result, mask, valid = _interpolate_scores_to_frames(scores, timestamps, frame_times)
    assert valid is True
    assert not np.any(np.isnan(result))
    assert not np.any(np.isnan(mask))


def test_interp_leading_nan_no_crash():
    """Leading NaN (leading silence) must not crash or produce NaN output."""
    timestamps = np.array([0.0, 0.5, 1.0, 1.5])
    scores = np.array([np.nan, np.nan, 0.4, 0.7])
    frame_times = _frame_times(100)
    result, mask, valid = _interpolate_scores_to_frames(scores, timestamps, frame_times)
    assert valid is True
    assert not np.any(np.isnan(result))
    assert not np.any(np.isnan(mask))


def test_interp_all_nan_returns_zeros_and_false():
    """All-NaN scores → zero frame scores and any_valid=False."""
    timestamps = np.array([0.0, 0.5, 1.0])
    scores = np.full(3, np.nan)
    frame_times = _frame_times(80)
    result, mask, valid = _interpolate_scores_to_frames(scores, timestamps, frame_times)
    assert valid is False
    assert np.all(result == 0.0)
    assert np.all(mask == 0.0)
    assert not np.any(np.isnan(result))


def test_interp_single_valid_score():
    """A single valid score should fill all frames (constant extrapolation)."""
    timestamps = np.array([0.0, 0.5, 1.0])
    scores = np.array([np.nan, 0.75, np.nan])
    frame_times = _frame_times(60)
    result, mask, valid = _interpolate_scores_to_frames(scores, timestamps, frame_times)
    assert valid is True
    assert np.allclose(result, 0.75)
    assert not np.any(np.isnan(result))


def test_interp_empty_scores():
    """Empty scores array → zeros and any_valid=False."""
    result, mask, valid = _interpolate_scores_to_frames(
        np.array([]), np.array([]), _frame_times(50)
    )
    assert valid is False
    assert len(result) == len(_frame_times(50))
    assert np.all(result == 0.0)
    assert np.all(mask == 0.0)


def test_interp_output_clipped_to_unit_interval():
    """Interpolated values should stay in [0, 1] for valid scores in [0, 1]."""
    timestamps = np.linspace(0, 2.0, 10)
    scores = np.random.default_rng(0).uniform(0.0, 1.0, 10)
    frame_times = _frame_times(200)
    result, mask, valid = _interpolate_scores_to_frames(scores, timestamps, frame_times)
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


def test_interp_trailing_nan_mask_is_zero_after_last_valid():
    """Alpha mask must be 0.0 for frames after the last valid window (trailing silence)."""
    # valid windows at 0.0, 0.5, 1.0; silent windows at 1.5, 2.0
    timestamps = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    scores = np.array([0.5, 0.6, 0.7, np.nan, np.nan])
    frame_times = _frame_times(200)  # extends well past last valid window (1.0 s)
    _, mask, valid = _interpolate_scores_to_frames(scores, timestamps, frame_times)
    assert valid is True
    # Frames after the last valid time (1.0 s) must be masked out
    last_valid_t = 1.0
    after_mask = frame_times > last_valid_t
    assert np.all(mask[after_mask] == 0.0), (
        "Frames after last valid window must have alpha_mask=0 (no-data region)"
    )


def test_interp_leading_nan_mask_is_zero_before_first_valid():
    """Alpha mask must be 0.0 for frames before the first valid window (leading silence)."""
    timestamps = np.array([0.0, 0.5, 1.0, 1.5])
    scores = np.array([np.nan, np.nan, 0.4, 0.7])
    frame_times = _frame_times(150)
    _, mask, valid = _interpolate_scores_to_frames(scores, timestamps, frame_times)
    assert valid is True
    first_valid_t = 1.0  # timestamps[2]
    before_mask = frame_times < first_valid_t
    assert np.all(mask[before_mask] == 0.0), (
        "Frames before first valid window must have alpha_mask=0 (no-data region)"
    )


# ---------------------------------------------------------------------------
# render_explainability_overlay — file output
# ---------------------------------------------------------------------------


def _speech_clip(duration: float = 2.0, sr: int = SR) -> np.ndarray:
    """Sine wave — non-silent, suitable for spectrogram + attribution."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)


def _clip_with_trailing_silence(
    speech_duration: float = 1.5,
    silence_duration: float = 0.5,
    sr: int = SR,
) -> np.ndarray:
    """Speech followed by silence — produces trailing NaN attribution windows."""
    t = np.linspace(0, speech_duration, int(sr * speech_duration), endpoint=False)
    speech = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
    silence = np.zeros(int(sr * silence_duration), dtype=np.float32)
    return np.concatenate([speech, silence])


def _clip_with_leading_silence(
    silence_duration: float = 0.5,
    speech_duration: float = 1.5,
    sr: int = SR,
) -> np.ndarray:
    """Silence followed by speech — produces leading NaN attribution windows."""
    silence = np.zeros(int(sr * silence_duration), dtype=np.float32)
    t = np.linspace(0, speech_duration, int(sr * speech_duration), endpoint=False)
    speech = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
    return np.concatenate([silence, speech])


def test_render_saves_png():
    """Basic smoke: saves a non-empty PNG."""
    wav = _speech_clip()
    with tempfile.TemporaryDirectory() as d:
        out = render_explainability_overlay(wav, SR, _ConstDetector(), output_path=Path(d) / "o.png")
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0


def test_render_returns_str():
    """Return value must be a plain str (Gradio gr.Image compatibility)."""
    wav = _speech_clip()
    with tempfile.TemporaryDirectory() as d:
        out = render_explainability_overlay(wav, SR, _ConstDetector(), output_path=Path(d) / "o.png")
        assert isinstance(out, str)


def test_render_returns_absolute_path():
    wav = _speech_clip()
    with tempfile.TemporaryDirectory() as d:
        out = render_explainability_overlay(wav, SR, _ConstDetector(), output_path=Path(d) / "o.png")
        assert Path(out).is_absolute()


def test_render_creates_parent_dirs():
    wav = _speech_clip()
    with tempfile.TemporaryDirectory() as d:
        nested = Path(d) / "a" / "b" / "overlay.png"
        out = render_explainability_overlay(wav, SR, _ConstDetector(), output_path=nested)
        assert Path(out).exists()


def test_render_string_output_path():
    """output_path can be a plain str."""
    wav = _speech_clip()
    with tempfile.TemporaryDirectory() as d:
        out = render_explainability_overlay(wav, SR, _ConstDetector(), output_path=str(Path(d) / "o.png"))
        assert Path(out).exists()


# ---------------------------------------------------------------------------
# NaN-robustness tests — the most important ones per the prompt spec
# ---------------------------------------------------------------------------


def test_render_trailing_silence_does_not_crash():
    """Clip with trailing silence → NaN windows → must render without error."""
    wav = _clip_with_trailing_silence()
    with tempfile.TemporaryDirectory() as d:
        out = render_explainability_overlay(wav, SR, _ConstDetector(), output_path=Path(d) / "o.png")
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0


def test_render_leading_silence_does_not_crash():
    """Clip with leading silence → NaN windows → must render without error."""
    wav = _clip_with_leading_silence()
    with tempfile.TemporaryDirectory() as d:
        out = render_explainability_overlay(wav, SR, _ConstDetector(), output_path=Path(d) / "o.png")
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0


def test_render_all_silent_detector_does_not_crash():
    """When every attribution window returns NaN (all silence),
    the spectrogram must still render with a transparent heatmap."""
    # Build a clip that's entirely silent except one tiny non-zero sample
    # so generate_mel_spectrogram doesn't raise, but make it soft enough
    # that the silence gate fires for every attribution window.
    wav = np.zeros(SR * 2, dtype=np.float32)
    wav[0] = 1e-4   # just enough to pass the all-zero check in generate_mel_spectrogram

    class _SilenceGateDetector:
        """Always returns None (mimics detector's own silence gate)."""
        def predict_waveform(self, w, sr):
            return {"probability_synthetic": None, "label": "inconclusive"}

    with tempfile.TemporaryDirectory() as d:
        # Should NOT raise even though all windows are NaN
        out = render_explainability_overlay(
            wav, SR, _SilenceGateDetector(), output_path=Path(d) / "o.png"
        )
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0


def test_render_mixed_nan_and_valid_produces_valid_png():
    """Most realistic scenario: speech + trailing silence with NaN windows."""
    wav = _clip_with_trailing_silence(speech_duration=1.0, silence_duration=1.0)
    with tempfile.TemporaryDirectory() as d:
        out = render_explainability_overlay(wav, SR, _ZeroDetector(), output_path=Path(d) / "o.png")
        p = Path(out)
        assert p.exists()
        assert p.stat().st_size > 1024  # substantive PNG


# ---------------------------------------------------------------------------
# Figure memory: no leaks on repeated calls
# ---------------------------------------------------------------------------


def test_render_no_figure_leak():
    """Calling render multiple times must not accumulate open figures."""
    import matplotlib.pyplot as plt

    wav = _speech_clip(duration=1.0)
    with tempfile.TemporaryDirectory() as d:
        before = len(plt.get_fignums())
        for i in range(4):
            render_explainability_overlay(
                wav, SR, _ConstDetector(), output_path=Path(d) / f"o{i}.png"
            )
        after = len(plt.get_fignums())
    assert after <= before
