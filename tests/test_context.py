"""Unit tests for transaction context and contact familiarity multipliers (Phase 7 / Prompt 9.4)."""

import pytest
from voxguard import config
from voxguard.fusion.context import (
    get_contact_familiarity_multiplier,
    get_transaction_multiplier,
)


def test_transaction_multipliers_known_and_unknown():
    """Confirms get_transaction_multiplier retrieves valid multipliers and falls back safely."""
    assert get_transaction_multiplier("general_conversation") == 1.0
    assert get_transaction_multiplier("otp_request") == 1.3
    assert get_transaction_multiplier("fund_transfer") == 1.5
    assert get_transaction_multiplier("confidential_info_request") == 1.4

    # Unknown or empty contexts fall back to general_conversation (1.0)
    assert get_transaction_multiplier("unknown_context") == 1.0
    assert get_transaction_multiplier("") == 1.0
    assert get_transaction_multiplier(None) == 1.0


def test_contact_familiarity_three_state_logic():
    """Confirms three-state contact familiarity evaluation."""
    # 1. Verified match -> 0.9
    match_result = {"match": True, "similarity": 0.85, "enrolled_name": "byaquta"}
    assert get_contact_familiarity_multiplier(match_result) == 0.9

    # 2. Verified mismatch -> 1.3
    mismatch_result = {"match": False, "similarity": 0.35, "enrolled_name": "byaquta"}
    assert get_contact_familiarity_multiplier(mismatch_result) == 1.3

    # 3. No enrollment data / None / malformed -> 1.0
    assert get_contact_familiarity_multiplier(None) == 1.0
    assert get_contact_familiarity_multiplier({}) == 1.0
    assert get_contact_familiarity_multiplier({"similarity": 0.5}) == 1.0
    assert get_contact_familiarity_multiplier({"match": None}) == 1.0
