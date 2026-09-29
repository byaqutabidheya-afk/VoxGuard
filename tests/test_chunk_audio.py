"""
test_chunk_audio.py — tests for voxguard.utils.chunk_audio.chunk_waveform.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from voxguard import config
from voxguard.streaming.buffer import StreamingBuffer
from voxguard.streaming.scorer import SILENCE_RMS_THRESHOLD, StreamingScorer
from voxguard.utils.chunk_audio import chunk_waveform

SR = 16000


def _speech_like(seconds: float, seed: int = 0) -> np.ndarray:
    """Non-silent test signal (well above the silence gate everywhere)."""
    rng = np.random.default_rng(seed)
    n = int(round(seconds * SR))
    t = np.arange(n) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(n)).astype(np.float32)


@pytest.mark.parametrize("duration", [1.5, 2.0, 2.5, 3.0, 3.7, 5.25, 10.0])
@pytest.mark.parametrize("chunk,overlap", [(1.5, 0.5), (2.0, 0.0), (1.0, 0.75)])
def test_chunk_count_matches_closed_form(duration, chunk, overlap):
    chunks = chunk_waveform(_speech_like(duration), SR, chunk, overlap)
    stride = chunk - overlap
    expected = math.floor(round((duration - chunk) / stride, 9)) + 1
    assert len(chunks) == expected


def test_shorter_than_one_chunk_returns_empty():
    assert chunk_waveform(_speech_like(1.0), SR, 1.5, 0.5) == []


@pytest.mark.parametrize("chunk,overlap", [(1.5, 0.5), (2.0, 0.0), (1.0, 0.75)])
def test_every_chunk_has_exact_length(chunk, overlap):
    chunks = chunk_waveform(_speech_like(7.3), SR, chunk, overlap)
    assert chunks
    assert all(c.shape == (int(round(chunk * SR)),) for c in chunks)


def test_defaults_come_from_streaming_config(monkeypatch):
    monkeypatch.setattr(config, "STREAM_CHUNK_SECONDS", 2.0)
    monkeypatch.setattr(config, "STREAM_OVERLAP_SECONDS", 1.0)
    chunks = chunk_waveform(_speech_like(5.0), SR)
    assert len(chunks) == 4  # floor((5 - 2) / 1) + 1
    assert all(c.size == 2 * SR for c in chunks)


def test_all_silence_returns_empty_when_dropping():
    silence = np.zeros(5 * SR, dtype=np.float32)
    assert chunk_waveform(silence, SR) == []


def test_all_silence_kept_when_not_dropping():
    silence = np.zeros(5 * SR, dtype=np.float32)
    assert len(chunk_waveform(silence, SR, 1.5, 0.5, drop_silent=False)) == 4


def test_silence_gate_matches_streaming_scorer():
    # 3 s of signal then 3 s of near-silence: some windows straddle the boundary.
    wav = np.concatenate([_speech_like(3.0), np.full(3 * SR, 1e-4, dtype=np.float32)])
    all_windows = chunk_waveform(wav, SR, 1.5, 0.5, drop_silent=False)
    kept = chunk_waveform(wav, SR, 1.5, 0.5)

    expected = [w for w in all_windows if StreamingScorer._rms_energy(w) >= SILENCE_RMS_THRESHOLD]
    assert 0 < len(kept) < len(all_windows)
    assert len(kept) == len(expected)
    for a, b in zip(kept, expected):
        np.testing.assert_array_equal(a, b)


def test_custom_silence_threshold():
    quiet = np.full(3 * SR, 0.02, dtype=np.float32)  # RMS 0.02
    assert chunk_waveform(quiet, SR, 1.5, 0.5, silence_threshold=0.05) == []
    assert len(chunk_waveform(quiet, SR, 1.5, 0.5, silence_threshold=0.01)) == 2


@pytest.mark.parametrize("chunk,overlap", [(None, None), (1.5, 0.5), (2.0, 0.25)])
def test_identical_to_manually_driven_streaming_buffer(chunk, overlap):
    """Anti-drift: training chunks must equal what live streaming produces, sample for sample."""
    wav = _speech_like(9.37, seed=7)
    ours = chunk_waveform(wav, SR, chunk, overlap, drop_silent=False)

    # Drive a buffer the way a live stream does: many small, uneven frames.
    buffer = StreamingBuffer(sample_rate=SR, chunk_seconds=chunk, overlap_seconds=overlap)
    rng = np.random.default_rng(1)
    streamed = []
    pos = 0
    while pos < wav.size:
        step = int(rng.integers(100, 4000))
        streamed.extend(buffer.push(wav[pos : pos + step]))
        pos += step

    assert len(ours) == len(streamed) > 0
    for a, b in zip(ours, streamed):
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("chunk,overlap", [(None, None), (1.5, 0.5), (2.0, 0.25)])
def test_detailed_offsets_point_at_source_samples(chunk, overlap):
    from voxguard.utils.chunk_audio import chunk_waveform_detailed

    wav = np.concatenate([_speech_like(4.0, seed=3), np.zeros(5 * SR, dtype=np.float32)])
    detailed = chunk_waveform_detailed(wav, SR, chunk, overlap)

    assert [c.index for c in detailed] == list(range(len(detailed)))
    for c in detailed:
        start = int(round(c.start_seconds * SR))
        np.testing.assert_array_equal(c.samples, wav[start : start + c.samples.size])
    # chunk_waveform is exactly the non-silent subset of the detailed view.
    kept = chunk_waveform(wav, SR, chunk, overlap)
    non_silent = [c.samples for c in detailed if not c.is_silent]
    assert any(c.is_silent for c in detailed) and len(kept) == len(non_silent)
    for a, b in zip(kept, non_silent):
        np.testing.assert_array_equal(a, b)
