"""Tests for the running risk score EMA helper."""

from __future__ import annotations

from voxguard.streaming.ema import RunningRiskScore


def test_running_risk_score_tracks_expected_ema_sequence() -> None:
    tracker = RunningRiskScore(alpha=0.5)

    scores = [0.1, 0.1, None, 0.9, 0.9, 0.9]
    expected = [0.1, 0.1, 0.1, 0.5, 0.7, 0.8]

    observed = [tracker.update(score) for score in scores]

    assert observed == expected
    assert tracker.current() == expected[-1]


def test_running_risk_score_reset_clears_state() -> None:
    tracker = RunningRiskScore(alpha=0.5)

    assert tracker.update(0.4) == 0.4
    tracker.reset()

    assert tracker.current() is None
    assert tracker.update(0.8) == 0.8
