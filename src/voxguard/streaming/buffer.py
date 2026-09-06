"""
buffer.py — bounded overlapping audio window buffer for streaming inference.

Worked example
--------------
With ``sample_rate=10``, ``chunk_seconds=1.0`` and ``overlap_seconds=0.5``:

- chunk length = 10 samples
- stride = 5 samples

If the buffer receives 12 samples first, only one complete window is
available: samples ``[0:10]``.

If the next push adds 8 more samples, the buffer now contains samples
``[5:20]`` after trimming and emits two windows in total across the two
pushes:

- ``[0:10]``
- ``[5:15]``

This is the expected overlap behavior: each emitted window advances by the
stride, so windows share the requested overlap instead of duplicating or
skipping samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voxguard import config


class StreamingBuffer:
    """Accumulates audio samples and yields complete overlapping windows."""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_seconds: float | None = None,
        overlap_seconds: float | None = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.chunk_seconds = float(
            config.STREAM_CHUNK_SECONDS if chunk_seconds is None else chunk_seconds
        )
        self.overlap_seconds = float(
            config.STREAM_OVERLAP_SECONDS
            if overlap_seconds is None
            else overlap_seconds
        )

        self.chunk_samples = int(round(self.chunk_seconds * self.sample_rate))
        self.overlap_samples = int(round(self.overlap_seconds * self.sample_rate))
        self.stride_samples = self.chunk_samples - self.overlap_samples

        if self.chunk_samples <= 0:
            raise ValueError(
                f"chunk_seconds must produce at least one sample; got {self.chunk_seconds!r}."
            )
        if self.overlap_samples < 0:
            raise ValueError(
                f"overlap_seconds must be non-negative; got {self.overlap_seconds!r}."
            )
        if self.stride_samples <= 0:
            raise ValueError(
                "overlap_seconds must be smaller than chunk_seconds so the stride stays positive."
            )

        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start_sample = 0
        self._next_window_start = 0

    def push(self, audio_frame: np.ndarray) -> list[np.ndarray]:
        """Appends audio samples and returns any newly available windows."""
        frame = np.asarray(audio_frame, dtype=np.float32).reshape(-1)
        if frame.size == 0:
            return []

        if self._buffer.size == 0:
            self._buffer = frame.copy()
        else:
            self._buffer = np.concatenate([self._buffer, frame])

        windows: list[np.ndarray] = []
        buffer_end_sample = self._buffer_start_sample + self._buffer.size

        while self._next_window_start + self.chunk_samples <= buffer_end_sample:
            relative_start = self._next_window_start - self._buffer_start_sample
            relative_end = relative_start + self.chunk_samples
            windows.append(self._buffer[relative_start:relative_end].copy())
            self._next_window_start += self.stride_samples

        trim_samples = self._next_window_start - self._buffer_start_sample
        if trim_samples > 0:
            self._buffer = self._buffer[trim_samples:].copy()
            self._buffer_start_sample = self._next_window_start

        return windows

    def reset(self) -> None:
        """Clears the internal buffer and read position for a new session."""
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start_sample = 0
        self._next_window_start = 0
