"""
test_duration_match.py — tests for voxguard.utils.duration_match.
"""

from __future__ import annotations

import numpy as np
import pytest

from voxguard.utils.duration_match import duration_match_pair

SR = 16000


def _ramp(seconds: float) -> np.ndarray:
    """Distinct sample values so we can check exactly which samples survive."""
    return np.arange(int(round(seconds * SR)), dtype=np.float32)


@pytest.mark.parametrize("real_s,synth_s", [(5.0, 8.0), (8.0, 5.0)])
def test_longer_clip_centre_trimmed_to_shorter(real_s, synth_s):
    real, synth = _ramp(real_s), _ramp(synth_s)
    m_real, m_synth, info = duration_match_pair(real, synth, SR)

    assert m_real.shape[0] == m_synth.shape[0] == 5 * SR
    assert info["matched_seconds"] == pytest.approx(5.0)
    assert info["trimmed_seconds"] == pytest.approx(3.0)
    assert info["trimmed_start_seconds"] == pytest.approx(1.5)
    assert info["trimmed_end_seconds"] == pytest.approx(1.5)
    assert info["original_real_seconds"] == pytest.approx(real_s)
    assert info["original_synth_seconds"] == pytest.approx(synth_s)

    longer_name = "real" if real_s > synth_s else "synth"
    assert info["trimmed_clip"] == longer_name
    longer_in, longer_out = (real, m_real) if longer_name == "real" else (synth, m_synth)
    shorter_in, shorter_out = (synth, m_synth) if longer_name == "real" else (real, m_real)

    # 1.5 s removed from each end of the longer clip
    np.testing.assert_array_equal(longer_out, longer_in[int(1.5 * SR) : int(6.5 * SR)])
    # shorter clip untouched
    np.testing.assert_array_equal(shorter_out, shorter_in)


def test_equal_length_unchanged():
    real, synth = _ramp(3.0), _ramp(3.0) + 0.5
    m_real, m_synth, info = duration_match_pair(real, synth, SR)

    np.testing.assert_array_equal(m_real, real)
    np.testing.assert_array_equal(m_synth, synth)
    assert info["trimmed_clip"] == "none"
    assert info["trimmed_seconds"] == 0
    assert info["trimmed_start_seconds"] == 0
    assert info["trimmed_end_seconds"] == 0


def test_odd_difference_puts_extra_sample_at_end():
    real = np.arange(10, dtype=np.float32)
    synth = np.arange(7, dtype=np.float32)
    m_real, _, info = duration_match_pair(real, synth, sr=1, min_seconds=0)

    np.testing.assert_array_equal(m_real, real[1:8])  # 1 off start, 2 off end
    assert info["trimmed_start_seconds"] == 1
    assert info["trimmed_end_seconds"] == 2


def test_below_min_seconds_raises():
    with pytest.raises(ValueError, match="min_seconds"):
        duration_match_pair(_ramp(1.0), _ramp(4.0), SR, min_seconds=1.5)


@pytest.mark.parametrize("empty_side", ["real", "synth"])
def test_empty_input_raises(empty_side):
    empty, full = np.array([], dtype=np.float32), _ramp(2.0)
    args = (empty, full) if empty_side == "real" else (full, empty)
    with pytest.raises(ValueError, match="empty"):
        duration_match_pair(*args, SR)


@pytest.mark.parametrize(
    "real_n,synth_n",
    [(24000, 24000), (24000, 24001), (24001, 24000), (30000, 90000), (90001, 30000)],
)
def test_never_pads(real_n, synth_n):
    real = np.ones(real_n, dtype=np.float32)
    synth = np.ones(synth_n, dtype=np.float32)
    m_real, m_synth, _ = duration_match_pair(real, synth, SR)

    shortest = min(real_n, synth_n)
    for out in (m_real, m_synth):
        assert out.shape[0] <= real_n
        assert out.shape[0] <= synth_n
        assert out.shape[0] == shortest
    # all-ones inputs: any padding would introduce non-one values
    assert np.all(m_real == 1) and np.all(m_synth == 1)
