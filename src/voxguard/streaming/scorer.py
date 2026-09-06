"""Streaming chunk scorer for VoxGuard detectors."""

from __future__ import annotations

from typing import Any

import numpy as np


class StreamingScorer:
    """Wraps a detector and scores in-memory audio chunks safely.

    The wrapped detector must expose ``predict_waveform(waveform, sr)`` and
    return a mapping containing ``"probability_synthetic"`` and ``"label"``.
    """

    def __init__(self, detector: Any, silence_threshold: float = 0.01) -> None:
        if not hasattr(detector, "predict_waveform"):
            raise TypeError(
                "StreamingScorer requires a detector with a predict_waveform(waveform, sr) method."
            )
        self.detector = detector
        self.silence_threshold = float(silence_threshold)

    @staticmethod
    def _rms_energy(chunk: np.ndarray) -> float:
        """Computes chunk RMS energy using a float64 accumulator."""
        chunk = np.asarray(chunk)
        if chunk.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))

    def score_chunk(self, chunk: np.ndarray, sr: int) -> float | None:
        """Scores one chunk and returns P(synthetic), or ``None`` if scoring fails.

        Chunks below ``self.silence_threshold`` RMS are skipped without
        calling the detector, since the classifier was not trained to model
        non-speech audio.
        """
        if self._rms_energy(chunk) < self.silence_threshold:
            return None

        try:
            prediction = self.detector.predict_waveform(chunk, sr)
            return float(prediction["probability_synthetic"])
        except Exception:
            return None
