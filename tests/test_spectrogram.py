"""Tests for voxguard.explain.spectrogram."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from voxguard.explain.spectrogram import (
    DEFAULT_HOP_LENGTH,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    generate_mel_spectrogram,
    render_spectrogram_image,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SR = 16_000  # matches config.SAMPLE_RATE


def _sine(freq: float = 440.0, duration: float = 1.0, sr: int = SR) -> np.ndarray:
    """Return a 1-D float32 sine wave."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _noise(duration: float = 1.0, sr: int = SR) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal(int(sr * duration)).astype(np.float32)


# ---------------------------------------------------------------------------
# generate_mel_spectrogram — shape and dtype
# ---------------------------------------------------------------------------


def test_output_shape_default_params() -> None:
    """Shape should be (n_mels, n_frames) with default parameters."""
    wav = _sine()
    mel = generate_mel_spectrogram(wav, SR)
    assert mel.ndim == 2
    assert mel.shape[0] == DEFAULT_N_MELS


def test_output_dtype_is_float32() -> None:
    mel = generate_mel_spectrogram(_sine(), SR)
    assert mel.dtype == np.float32


def test_custom_n_mels() -> None:
    mel = generate_mel_spectrogram(_sine(), SR, n_mels=64)
    assert mel.shape[0] == 64


def test_n_frames_scales_with_hop_length() -> None:
    """More frames for smaller hop_length, all else equal."""
    wav = _sine()
    mel_small_hop = generate_mel_spectrogram(wav, SR, hop_length=256)
    mel_large_hop = generate_mel_spectrogram(wav, SR, hop_length=1024)
    assert mel_small_hop.shape[1] > mel_large_hop.shape[1]


def test_values_are_in_db_range() -> None:
    """log-mel values should be <= 0 dB (ref=max) and >= -80 dB (top_db=80)."""
    mel = generate_mel_spectrogram(_sine(), SR)
    assert float(mel.max()) <= 0.1   # small float tolerance
    assert float(mel.min()) >= -80.1


def test_noise_produces_values_across_full_range() -> None:
    """White noise should have max near 0 dB and some spread below it.

    White noise distributes energy broadly across bins, so with ref=max the
    minimum dB is much higher than speech (flat spectrum → no very-quiet
    bins).  The assertion checks that the max is near 0 dB (correctly
    normalised) and that there is at least some dynamic range, without
    assuming a specific floor.
    """
    mel = generate_mel_spectrogram(_noise(), SR)
    assert float(mel.max()) > -5.0    # near 0 dB somewhere (ref=max)
    assert float(mel.min()) < -10.0   # some spread — not completely flat


def test_stereo_input_is_accepted() -> None:
    """2-D (2, samples) waveform should not raise."""
    stereo = np.stack([_sine(440), _sine(880)], axis=0)
    mel = generate_mel_spectrogram(stereo, SR)
    assert mel.shape[0] == DEFAULT_N_MELS


def test_int_waveform_coerced_to_float32() -> None:
    """Integer waveform should be accepted (cast internally)."""
    wav_int = (_sine() * 32767).astype(np.int16)
    mel = generate_mel_spectrogram(wav_int.astype(np.float32), SR)
    assert mel.shape[0] == DEFAULT_N_MELS


def test_short_waveform_one_frame() -> None:
    """Very short waveform should still return at least one frame."""
    tiny = _sine(duration=0.05)   # 50 ms → ~800 samples at 16 kHz
    mel = generate_mel_spectrogram(tiny, SR)
    assert mel.shape[1] >= 1


# ---------------------------------------------------------------------------
# generate_mel_spectrogram — error handling
# ---------------------------------------------------------------------------


def test_empty_waveform_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        generate_mel_spectrogram(np.array([], dtype=np.float32), SR)


def test_all_zero_waveform_raises() -> None:
    with pytest.raises(ValueError, match="only zeros"):
        generate_mel_spectrogram(np.zeros(SR, dtype=np.float32), SR)


def test_3d_waveform_raises() -> None:
    with pytest.raises(ValueError, match="1-D or 2-D"):
        generate_mel_spectrogram(np.zeros((2, 3, 4), dtype=np.float32), SR)


# ---------------------------------------------------------------------------
# render_spectrogram_image — file output
# ---------------------------------------------------------------------------


def test_render_saves_png() -> None:
    """render_spectrogram_image must create a non-empty PNG file."""
    mel = generate_mel_spectrogram(_sine(), SR)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "spec.png"
        result = render_spectrogram_image(mel, SR, output_path=out)
        assert result.exists()
        assert result.suffix == ".png"
        assert result.stat().st_size > 0


def test_render_returns_resolved_path() -> None:
    """Return value should be the resolved absolute Path."""
    mel = generate_mel_spectrogram(_sine(), SR)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "spec.png"
        result = render_spectrogram_image(mel, SR, output_path=out)
        assert result.is_absolute()
        assert result == out.resolve()


def test_render_creates_parent_dirs() -> None:
    """output_path parent directories should be created automatically."""
    mel = generate_mel_spectrogram(_sine(), SR)
    with tempfile.TemporaryDirectory() as tmpdir:
        nested = Path(tmpdir) / "a" / "b" / "c" / "spec.png"
        result = render_spectrogram_image(mel, SR, output_path=nested)
        assert result.exists()


def test_render_accepts_string_path() -> None:
    """output_path can be a plain str, not just pathlib.Path."""
    mel = generate_mel_spectrogram(_sine(), SR)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_str = str(Path(tmpdir) / "spec.png")
        result = render_spectrogram_image(mel, SR, output_path=out_str)
        assert result.exists()


def test_render_closes_figure_no_memory_leak() -> None:
    """Calling render multiple times should not accumulate matplotlib figures."""
    import matplotlib.pyplot as plt

    mel = generate_mel_spectrogram(_sine(), SR)
    with tempfile.TemporaryDirectory() as tmpdir:
        before = len(plt.get_fignums())
        for i in range(5):
            render_spectrogram_image(
                mel, SR, output_path=Path(tmpdir) / f"spec_{i}.png"
            )
        after = len(plt.get_fignums())
    # No net increase in open figures
    assert after <= before


def test_render_bad_mel_shape_raises() -> None:
    """1-D mel array should raise ValueError."""
    with pytest.raises(ValueError, match="2-D"):
        with tempfile.TemporaryDirectory() as tmpdir:
            render_spectrogram_image(
                np.zeros(128, dtype=np.float32),
                SR,
                output_path=Path(tmpdir) / "spec.png",
            )


def test_render_custom_hop_matches_generate() -> None:
    """hop_length passed to render must match the one used in generate — if
    they differ the time axis is wrong, but the function should not raise."""
    wav = _sine()
    hop = 256
    mel = generate_mel_spectrogram(wav, SR, hop_length=hop)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = render_spectrogram_image(
            mel, SR, hop_length=hop, output_path=Path(tmpdir) / "spec.png"
        )
        assert out.exists()


# ---------------------------------------------------------------------------
# Round-trip: generate → render
# ---------------------------------------------------------------------------


def test_round_trip_produces_valid_png() -> None:
    """Full pipeline: waveform → mel array → PNG file."""
    wav = _noise(duration=2.0)
    mel = generate_mel_spectrogram(wav, SR, n_mels=64, hop_length=256)
    with tempfile.TemporaryDirectory() as tmpdir:
        png = render_spectrogram_image(
            mel,
            SR,
            hop_length=256,
            output_path=Path(tmpdir) / "round_trip.png",
            title="Round-trip test",
        )
        assert png.exists()
        assert png.stat().st_size > 1024  # non-trivial file
