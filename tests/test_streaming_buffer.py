"""Tests for the streaming audio buffer."""

from __future__ import annotations

import numpy as np

from voxguard.streaming.buffer import StreamingBuffer


def test_streaming_buffer_emits_overlapping_windows_and_trims() -> None:
    buffer = StreamingBuffer(sample_rate=10, chunk_seconds=1.0, overlap_seconds=0.5)

    first = buffer.push(np.arange(12, dtype=np.float32))
    assert len(first) == 1
    assert first[0].tolist() == list(np.arange(10, dtype=np.float32))

    second = buffer.push(np.arange(12, 20, dtype=np.float32))
    assert len(second) == 2
    assert second[0].tolist() == list(np.arange(5, 15, dtype=np.float32))
    assert second[1].tolist() == list(np.arange(10, 20, dtype=np.float32))


def test_streaming_buffer_reset_clears_state() -> None:
    buffer = StreamingBuffer(sample_rate=10, chunk_seconds=1.0, overlap_seconds=0.5)
    buffer.push(np.arange(12, dtype=np.float32))
    buffer.reset()

    windows = buffer.push(np.arange(10, dtype=np.float32))
    assert len(windows) == 1
    assert windows[0].tolist() == list(np.arange(10, dtype=np.float32))
