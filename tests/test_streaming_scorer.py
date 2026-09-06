"""Tests for the streaming scorer wrapper."""

from __future__ import annotations

import numpy as np

from voxguard.streaming.scorer import StreamingScorer


class _DummyDetector:
    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.calls = 0

    def predict_waveform(self, waveform: np.ndarray, sr: int) -> dict:
        self.calls += 1
        if self.should_fail:
            raise RuntimeError("silence or corrupt chunk")
        return {"label": "synthetic", "probability_synthetic": 0.73}


def test_streaming_scorer_returns_probability() -> None:
    scorer = StreamingScorer(_DummyDetector())
    assert scorer.score_chunk(np.ones(10, dtype=np.float32), 16000) == 0.73


def test_streaming_scorer_returns_none_on_failure() -> None:
    scorer = StreamingScorer(_DummyDetector(should_fail=True))
    assert scorer.score_chunk(np.ones(10, dtype=np.float32), 16000) is None


def test_streaming_scorer_skips_silence_without_calling_detector() -> None:
    detector = _DummyDetector()
    scorer = StreamingScorer(detector, silence_threshold=0.01)

    assert scorer.score_chunk(np.zeros(160, dtype=np.float32), 16000) is None
    assert detector.calls == 0


def test_streaming_scorer_scores_non_silent_noise() -> None:
    detector = _DummyDetector()
    scorer = StreamingScorer(detector, silence_threshold=0.01)

    chunk = np.random.default_rng(0).normal(0.0, 0.1, size=160).astype(np.float32)
    score = scorer.score_chunk(chunk, 16000)

    assert isinstance(score, float)
    assert score == 0.73
    assert detector.calls == 1
