"""Tests for voxguard.risk.bands.score_to_band.

Boundary convention being tested
---------------------------------
Exact boundary hits are assigned to the **higher** band — both thresholds
follow the same rule:

    score < low_max                             → "low"
    low_max <= score < medium_max               → "medium"
    score >= medium_max                         → "high"

So a score exactly equal to ``low_max`` → "medium", and a score exactly
equal to ``medium_max`` → "high".  This is the ONLY correct reading of
"boundary belongs to the stricter band" — any test asserting
``score_to_band(medium_max) == "medium"`` is wrong.

Default thresholds used throughout come from config.RISK_THRESHOLDS at
runtime, so these tests remain correct after a recalibration in Prompt 7.5.
"""

from __future__ import annotations

import pytest
from voxguard import config

from voxguard.risk.bands import (
    BAND_HIGH,
    BAND_INCONCLUSIVE,
    BAND_LOW,
    BAND_MEDIUM,
    score_to_band,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT = None  # sentinel: use config defaults
_CUSTOM = {"low_max": 0.4, "medium_max": 0.6}


# ---------------------------------------------------------------------------
# Interior values — well away from boundaries
# ---------------------------------------------------------------------------


def test_low_interior() -> None:
    assert score_to_band(0.0) == BAND_LOW


def test_low_interior_midpoint() -> None:
    assert score_to_band(0.15) == BAND_LOW


def test_medium_interior() -> None:
    assert score_to_band(0.5) == BAND_MEDIUM


def test_high_interior() -> None:
    assert score_to_band(1.0) == BAND_HIGH


def test_high_interior_midpoint() -> None:
    assert score_to_band(0.85) == BAND_HIGH


# ---------------------------------------------------------------------------
# Exact boundary values — the key convention tests
# ---------------------------------------------------------------------------


def test_boundary_low_max_maps_to_medium() -> None:
    """A score exactly equal to low_max must return 'medium', not 'low'.

    The convention is: low_max is the *first* value that triggers at least a
    'medium' alert, so it belongs to 'medium'.
    """
    low_max = float(config.RISK_THRESHOLDS["low_max"])
    assert score_to_band(low_max) == BAND_MEDIUM


def test_boundary_medium_max_maps_to_high() -> None:
    """A score exactly equal to medium_max must return 'high', not 'medium'.

    Both boundaries follow the same rule: the boundary value itself is
    promoted to the stricter band above it.  medium_max is the first value
    that triggers 'high', so it belongs to 'high'.

    This test reads medium_max from config.RISK_THRESHOLDS so it remains
    correct after a Prompt 7.5 recalibration.
    """
    medium_max = float(config.RISK_THRESHOLDS["medium_max"])
    assert score_to_band(medium_max) == BAND_HIGH


def test_boundary_medium_max_config_value_maps_to_high() -> None:
    """Explicit test: score_to_band(config.RISK_THRESHOLDS["medium_max"]) == 'high'.

    Added in Prompt 7.5 fix to guard against the specific regression where
    score_to_band used <= instead of < for the medium_max boundary, causing
    the medium_max boundary score to stay in 'medium' rather than promote to
    'high'.  The test name is intentionally verbose so failures self-document.
    """
    medium_max = float(config.RISK_THRESHOLDS["medium_max"])
    result = score_to_band(medium_max)
    assert result == BAND_HIGH, (
        f"score_to_band({medium_max}) returned {result!r}; "
        f"expected 'high' because medium_max boundaries must promote to 'high'"
    )


def test_just_below_low_max() -> None:
    """0.3 - ε should still be 'low'."""
    assert score_to_band(0.2999) == BAND_LOW


def test_just_above_low_max() -> None:
    """low_max + ε should be 'medium'."""
    low_max = float(config.RISK_THRESHOLDS["low_max"])
    assert score_to_band(low_max + 1e-4) == BAND_MEDIUM


def test_just_below_medium_max() -> None:
    """medium_max - ε should be 'medium'."""
    medium_max = float(config.RISK_THRESHOLDS["medium_max"])
    assert score_to_band(medium_max - 1e-4) == BAND_MEDIUM


def test_just_above_medium_max() -> None:
    """medium_max + ε should be 'high'."""
    medium_max = float(config.RISK_THRESHOLDS["medium_max"])
    assert score_to_band(medium_max + 1e-4) == BAND_HIGH


# ---------------------------------------------------------------------------
# None input — Phase 6 silence / non-speech gate
# ---------------------------------------------------------------------------


def test_none_returns_inconclusive() -> None:
    """None probability_synthetic must return 'inconclusive', never raise."""
    assert score_to_band(None) == BAND_INCONCLUSIVE


def test_none_inconclusive_with_explicit_thresholds() -> None:
    """None short-circuits before threshold inspection regardless of thresholds."""
    assert score_to_band(None, thresholds=_CUSTOM) == BAND_INCONCLUSIVE


# ---------------------------------------------------------------------------
# Explicit thresholds override
# ---------------------------------------------------------------------------


def test_custom_thresholds_low() -> None:
    assert score_to_band(0.2, thresholds=_CUSTOM) == BAND_LOW


def test_custom_thresholds_boundary_low_max() -> None:
    """Boundary convention holds for custom thresholds too: 0.4 → 'medium'."""
    assert score_to_band(0.4, thresholds=_CUSTOM) == BAND_MEDIUM


def test_custom_thresholds_medium_interior() -> None:
    assert score_to_band(0.5, thresholds=_CUSTOM) == BAND_MEDIUM


def test_custom_thresholds_boundary_medium_max() -> None:
    """Boundary convention holds for custom thresholds too: 0.6 → 'high', not 'medium'."""
    assert score_to_band(0.6, thresholds=_CUSTOM) == BAND_HIGH


def test_custom_thresholds_high() -> None:
    assert score_to_band(0.8, thresholds=_CUSTOM) == BAND_HIGH


# ---------------------------------------------------------------------------
# Edge-case inputs
# ---------------------------------------------------------------------------


def test_integer_input_treated_as_float() -> None:
    """int 0 and int 1 are valid numeric inputs."""
    assert score_to_band(0) == BAND_LOW
    assert score_to_band(1) == BAND_HIGH


# ---------------------------------------------------------------------------
# Error conditions
# ---------------------------------------------------------------------------


def test_bad_type_raises_type_error() -> None:
    with pytest.raises(TypeError, match="probability_synthetic must be a float"):
        score_to_band("0.5")  # type: ignore[arg-type]


def test_missing_threshold_key_raises_value_error() -> None:
    with pytest.raises(ValueError, match="missing required key"):
        score_to_band(0.5, thresholds={"low_max": 0.3})  # medium_max absent


def test_inverted_thresholds_raises_value_error() -> None:
    with pytest.raises(ValueError, match="low_max.*<=.*medium_max"):
        score_to_band(0.5, thresholds={"low_max": 0.8, "medium_max": 0.2})
