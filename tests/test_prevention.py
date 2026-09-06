"""Tests for voxguard.risk.prevention.get_prevention_message."""

from __future__ import annotations

import pytest

from voxguard.risk.prevention import (
    HIGH_RISK_MESSAGE,
    MEDIUM_RISK_MESSAGE,
    get_prevention_message,
)


# ---------------------------------------------------------------------------
# Return-value identity — must return the named constants, not copies
# ---------------------------------------------------------------------------


def test_medium_returns_medium_constant() -> None:
    assert get_prevention_message("medium") is MEDIUM_RISK_MESSAGE


def test_high_returns_high_constant() -> None:
    assert get_prevention_message("high") is HIGH_RISK_MESSAGE


# ---------------------------------------------------------------------------
# None returns for non-alerting bands
# ---------------------------------------------------------------------------


def test_low_returns_none() -> None:
    assert get_prevention_message("low") is None


def test_inconclusive_returns_none() -> None:
    assert get_prevention_message("inconclusive") is None


# ---------------------------------------------------------------------------
# Content spot-checks — verify the copy is grounded and non-empty
# ---------------------------------------------------------------------------


def test_medium_message_is_nonempty_string() -> None:
    msg = get_prevention_message("medium")
    assert isinstance(msg, str) and len(msg) > 0


def test_high_message_is_nonempty_string() -> None:
    msg = get_prevention_message("high")
    assert isinstance(msg, str) and len(msg) > 0


def test_medium_message_starts_with_warning_emoji() -> None:
    """Medium tone opens with the ⚠️ caution marker."""
    assert get_prevention_message("medium").startswith("⚠️")


def test_high_message_starts_with_alert_emoji() -> None:
    """High tone opens with the 🚨 alert marker."""
    assert get_prevention_message("high").startswith("🚨")


def test_medium_message_softer_than_high() -> None:
    """Medium must not contain the word 'end' (as in 'end this call') —
    that direct instruction belongs only in the high-band message."""
    medium = get_prevention_message("medium").lower()
    high = get_prevention_message("high").lower()
    assert "end this call" not in medium
    assert "end this call" in high


def test_medium_message_mentions_verification() -> None:
    """Medium copy must instruct the user to verify before acting."""
    assert "verify" in get_prevention_message("medium").lower()


def test_high_message_mentions_reporting() -> None:
    """High copy must instruct the user to report the incident."""
    assert "report" in get_prevention_message("high").lower()


def test_medium_and_high_messages_differ() -> None:
    """The two messages must not be identical."""
    assert get_prevention_message("medium") != get_prevention_message("high")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_unknown_band_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown risk band"):
        get_prevention_message("critical")  # type: ignore[arg-type]


def test_empty_string_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown risk band"):
        get_prevention_message("")  # type: ignore[arg-type]
