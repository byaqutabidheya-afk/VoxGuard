"""
chunk_audio.py — split a whole waveform into training chunks exactly as streaming inference does.

``chunk_waveform`` deliberately contains NO windowing arithmetic of its own.
It drives a ``StreamingBuffer`` (the object that windows live audio at
inference time) and applies ``StreamingScorer``'s silence gate, so chunks used
for training are, by construction, the same chunks the model is scored on.
A second implementation of stride/overlap maths would be free to drift from
the buffer's (rounding, trailing-window handling, overlap semantics) and
silently reintroduce a train/inference mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from voxguard.streaming.buffer import StreamingBuffer
from voxguard.streaming.scorer import SILENCE_RMS_THRESHOLD, StreamingScorer


@dataclass(frozen=True)
class AudioChunk:
    """One window emitted by ``StreamingBuffer`` plus where it came from.

    ``index`` counts every emitted window, silent or not, so indices of kept
    chunks can have gaps where silent windows were dropped.
    """

    index: int
    start_seconds: float
    samples: np.ndarray
    is_silent: bool


def chunk_waveform_detailed(
    waveform: np.ndarray,
    sr: int,
    chunk_seconds: Optional[float] = None,
    overlap_seconds: Optional[float] = None,
    silence_threshold: Optional[float] = None,
) -> List[AudioChunk]:
    """Every window ``StreamingBuffer`` emits for *waveform*, with offset and silence flag.

    Nothing is dropped here; ``is_silent`` marks windows the silence gate would
    discard (see ``chunk_waveform``). Window *i* starts at
    ``i * buffer.stride_samples`` — the buffer's own stride, since a single
    ``push()`` from an empty buffer emits windows at consecutive strides.
    """
    buffer = StreamingBuffer(
        sample_rate=sr,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
    )
    threshold = SILENCE_RMS_THRESHOLD if silence_threshold is None else float(silence_threshold)
    return [
        AudioChunk(
            index=i,
            start_seconds=i * buffer.stride_samples / buffer.sample_rate,
            samples=w,
            is_silent=StreamingScorer._rms_energy(w) < threshold,
        )
        for i, w in enumerate(buffer.push(waveform))
    ]


def chunk_waveform(
    waveform: np.ndarray,
    sr: int,
    chunk_seconds: Optional[float] = None,
    overlap_seconds: Optional[float] = None,
    drop_silent: bool = True,
    silence_threshold: Optional[float] = None,
) -> List[np.ndarray]:
    """Split *waveform* into the overlapping windows streaming inference would see.

    The whole waveform is pushed through a fresh ``StreamingBuffer`` in one
    ``push()`` call and the emitted windows are returned. ``chunk_seconds`` and
    ``overlap_seconds`` default (via ``StreamingBuffer``) to
    ``config.STREAM_CHUNK_SECONDS`` / ``config.STREAM_OVERLAP_SECONDS``, so
    training windows track inference windows automatically when config changes.

    A trailing partial window shorter than ``chunk_seconds`` is dropped, never
    zero-padded — that is ``StreamingBuffer``'s behaviour and none is added here.

    Silence gating (``drop_silent=True``, the default): windows whose RMS energy
    is below the threshold are discarded, using the same RMS computation and
    threshold (``SILENCE_RMS_THRESHOLD``) as ``StreamingScorer``. At inference
    time silent chunks are never scored, so training on them teaches the model
    to classify audio it will never be asked about — and, worse, assigns them a
    real/synthetic label they don't deserve (silence carries no evidence of
    either).

    Parameters
    ----------
    waveform:
        Mono audio samples (any shape is flattened, as ``StreamingBuffer`` does).
    sr:
        Sample rate of *waveform* in Hz.
    chunk_seconds, overlap_seconds:
        Window length and overlap; ``None`` uses the streaming config defaults.
    drop_silent:
        Discard windows below the silence threshold.
    silence_threshold:
        RMS threshold override; ``None`` uses ``SILENCE_RMS_THRESHOLD``.

    Returns
    -------
    list of np.ndarray
        float32 windows of exactly ``round(chunk_seconds * sr)`` samples each.
    """
    chunks = chunk_waveform_detailed(
        waveform, sr, chunk_seconds, overlap_seconds, silence_threshold
    )
    return [c.samples for c in chunks if not (drop_silent and c.is_silent)]
