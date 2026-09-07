"""Unit tests for multimodal risk fusion (Phase 7 / Prompt 9.5)."""

import pytest
from voxguard import config
from voxguard.fusion.fuse import fuse_risk, fuse_risk_with_context


def test_fuse_risk_known_combinations():
    """Tests fuse_risk with known inputs and default weights (0.7 audio, 0.3 keyword)."""
    # 0.0 + 0.0 -> 0.0
    assert fuse_risk(0.0, 0.0) == 0.0

    # 1.0 + 1.0 -> 1.0
    assert fuse_risk(1.0, 1.0) == 1.0

    # 0.5 + 0.5 -> 0.5
    assert fuse_risk(0.5, 0.5) == 0.5

    # 0.8 audio + 0.2 keyword -> 0.8 * 0.7 + 0.2 * 0.3 = 0.56 + 0.06 = 0.62
    assert pytest.approx(fuse_risk(0.8, 0.2), 0.001) == 0.62


def test_fuse_risk_low_audio_high_keyword():
    """Specific case: low audio score + high keyword score produces a visibly elevated fused score."""
    audio_score = 0.10
    keyword_score = 1.00

    fused = fuse_risk(audio_score, keyword_score, audio_weight=0.7, keyword_weight=0.3)
    # Expected: 0.10 * 0.7 + 1.00 * 0.3 = 0.07 + 0.30 = 0.37
    assert pytest.approx(fused, 0.001) == 0.37
    # Crucially, language signal visibly elevated the overall risk well above the audio-only 0.10
    assert fused > audio_score
    assert fused >= 0.35


def test_fuse_risk_clipping():
    """Confirms fuse_risk clips scores to [0.0, 1.0]."""
    assert fuse_risk(-0.5, 0.0) == 0.0
    assert fuse_risk(1.5, 1.2, audio_weight=1.0, keyword_weight=1.0) == 1.0


def test_fuse_risk_with_context_contact_familiarity_three_states():
    """Tests the three contact-familiarity states independently in fuse_risk_with_context."""
    audio_score = 0.5
    keyword_score = 0.5
    # Base fused score = 0.5

    # 1. known_match: lowers the score vs. base (multiplier = 0.9 -> 0.45)
    match_vp = {"match": True, "similarity": 0.85, "enrolled_name": "byaquta"}
    res_match = fuse_risk_with_context(
        audio_score, keyword_score,
        transaction_context="general_conversation",
        voiceprint_result=match_vp,
    )
    assert res_match["base_fused_score"] == 0.5
    assert res_match["contact_multiplier"] == 0.9
    assert res_match["contextual_score"] < res_match["base_fused_score"]
    assert pytest.approx(res_match["contextual_score"], 0.001) == 0.45

    # 2. known_mismatch: raises the score vs. base (multiplier = 1.3 -> 0.65)
    mismatch_vp = {"match": False, "similarity": 0.35, "enrolled_name": "byaquta"}
    res_mismatch = fuse_risk_with_context(
        audio_score, keyword_score,
        transaction_context="general_conversation",
        voiceprint_result=mismatch_vp,
    )
    assert res_mismatch["base_fused_score"] == 0.5
    assert res_mismatch["contact_multiplier"] == 1.3
    assert res_mismatch["contextual_score"] > res_mismatch["base_fused_score"]
    assert pytest.approx(res_mismatch["contextual_score"], 0.001) == 0.65

    # 3. no_enrollment_data: leaves the score unchanged (multiplier = 1.0 -> 0.50)
    res_none = fuse_risk_with_context(
        audio_score, keyword_score,
        transaction_context="general_conversation",
        voiceprint_result=None,
    )
    assert res_none["base_fused_score"] == 0.5
    assert res_none["contact_multiplier"] == 1.0
    assert res_none["contextual_score"] == res_none["base_fused_score"]
    assert pytest.approx(res_none["contextual_score"], 0.001) == 0.50


def test_fuse_risk_with_context_transaction_types():
    """Tests that high-stakes transaction contexts raise the risk score vs. general conversation."""
    audio_score = 0.4
    keyword_score = 0.4
    # Base score = 0.4

    res_general = fuse_risk_with_context(
        audio_score, keyword_score, transaction_context="general_conversation"
    )
    res_transfer = fuse_risk_with_context(
        audio_score, keyword_score, transaction_context="fund_transfer"
    )
    res_otp = fuse_risk_with_context(
        audio_score, keyword_score, transaction_context="otp_request"
    )

    # Multipliers
    assert res_general["transaction_multiplier"] == 1.0
    assert res_transfer["transaction_multiplier"] == 1.5
    assert res_otp["transaction_multiplier"] == 1.3

    # fund_transfer (1.5x) raises the score higher than general_conversation
    assert res_transfer["contextual_score"] > res_general["contextual_score"]
    assert pytest.approx(res_transfer["contextual_score"], 0.001) == 0.60
    assert pytest.approx(res_otp["contextual_score"], 0.001) == 0.52
