"""Unit tests for describe_attribution (Phase 9 / Explainability)."""

import numpy as np
import pytest
from voxguard.explain.describe import describe_attribution


def test_describe_attribution_uniform_scores():
    """Tests uniform/low-variance attribution description."""
    timestamps = np.array([0.0, 0.75, 1.5, 2.25])
    scores = np.array([0.90, 0.92, 0.88, 0.91])  # std < 0.15, mean ~ 0.9025
    label = "synthetic"

    text = describe_attribution(scores, timestamps, label=label)

    assert "classified as synthetic" in text
    assert "average synthetic-likelihood of 90%" in text
    assert "strongest synthetic-sounding region is around 0.8s" in text or "0.7s" in text or "0.8" in text or "0.75" in text
    assert "most natural-sounding region is around 1.5s" in text
    assert "fairly consistent throughout the clip" in text
    assert "too little energy" not in text  # 0% NaNs


def test_describe_attribution_variable_scores():
    """Tests variable/high-variance attribution description."""
    timestamps = np.array([0.0, 0.75, 1.5, 2.25, 3.0])
    scores = np.array([0.10, 0.85, 0.90, 0.20, 0.15])  # std > 0.15
    label = "synthetic"

    text = describe_attribution(scores, timestamps, label=label)

    assert "classified as synthetic" in text
    assert "strongest synthetic-sounding region is around 1.5s" in text
    assert "most natural-sounding region is around 0.0s" in text
    assert "varies notably across the clip" in text
    assert "too little energy" not in text


def test_describe_attribution_with_silence_caveat():
    """Tests that >30% NaNs adds the low-energy caveat sentence."""
    timestamps = np.array([0.0, 0.75, 1.5, 2.25, 3.0, 3.75])
    scores = np.array([np.nan, 0.80, 0.85, np.nan, np.nan, 0.75])  # 3/6 = 50% NaNs
    label = "high risk"

    text = describe_attribution(scores, timestamps, label=label)

    assert "classified as high risk" in text
    assert "strongest synthetic-sounding region is around 1.5s" in text
    assert "most natural-sounding region is around 3.8s" in text
    assert "A significant portion of this clip had too little energy to score reliably" in text


def test_describe_attribution_all_nans_or_empty():
    """Tests empty or all-NaN fallback behavior."""
    empty_text = describe_attribution(np.array([]), np.array([]), label="real")
    assert "could not be computed" in empty_text

    all_nan_text = describe_attribution(np.array([np.nan, np.nan]), np.array([0.0, 1.0]), label="real")
    assert "could not be computed" in all_nan_text


def test_no_fabricated_acoustic_claims():
    """Confirms rule-based description avoids unverified acoustic jargon."""
    timestamps = np.array([0.0, 0.75, 1.5])
    scores = np.array([0.85, 0.90, 0.80])
    text = describe_attribution(scores, timestamps, label="synthetic")

    forbidden_terms = [
        "robotic",
        "metallic",
        "unnatural pitch",
        "vocoder",
        "glitch",
        "breathiness",
        "formant",
    ]
    for term in forbidden_terms:
        assert term not in text.lower(), f"Found forbidden fabricated claim: {term}"
