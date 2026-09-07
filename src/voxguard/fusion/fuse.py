"""
fuse.py — multimodal risk fusion scoring (Phase 7 / Prompt 9.5).

Combines acoustic voice cloning detection scores with semantic red-flag keyword
scores and situational context (transaction type and contact voiceprint verification).

Weighting Rationale:
────────────────────
Audio-based acoustic cloning detection is the primary, more rigorously validated
signal (backed by evaluation on ASVspoof2019 and synthetic Hindi benchmark sets
in Phases 2-4), weighted by default at 0.70.

The semantic keyword/phrase signal is a secondary corroborating indicator (weighted
at 0.30) that can elevate overall risk when urgent scam phrases (e.g. OTP demands,
customs coercion, isolation threats) are present in the call transcript.

Weights and multipliers are configured as constants in ``config.py`` (not hardcoded)
so they can be calibrated against live operational telemetry without modifying
inference logic.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from voxguard import config
from voxguard.fusion.context import (
    get_contact_familiarity_multiplier,
    get_transaction_multiplier,
)
from voxguard.utils.logging_utils import get_logger

logger = get_logger(__name__)


def fuse_risk(
    audio_score: float,
    keyword_risk_score: float,
    audio_weight: float = config.FUSION_AUDIO_WEIGHT,
    keyword_weight: float = config.FUSION_KEYWORD_WEIGHT,
) -> float:
    """Computes a combined risk score from audio and keyword risk signals.

    Parameters
    ----------
    audio_score:
        Acoustic synthetic probability or running streaming score in ``[0.0, 1.0]``.
    keyword_risk_score:
        Transcript red-flag risk score from ``scan_for_redflags`` in ``[0.0, 1.0]``.
    audio_weight:
        Weight assigned to the acoustic signal (default: ``config.FUSION_AUDIO_WEIGHT`` = 0.7).
    keyword_weight:
        Weight assigned to the language signal (default: ``config.FUSION_KEYWORD_WEIGHT`` = 0.3).

    Returns
    -------
    float
        Fused base risk score clipped to ``[0.0, 1.0]``.
    """
    a_score = float(max(0.0, min(1.0, audio_score)))
    k_score = float(max(0.0, min(1.0, keyword_risk_score)))

    weighted_sum = (a_score * float(audio_weight)) + (k_score * float(keyword_weight))
    fused_score = min(1.0, max(0.0, round(weighted_sum, 4)))
    return fused_score


def fuse_risk_with_context(
    audio_score: float,
    keyword_risk_score: float,
    transaction_context: str = "general_conversation",
    voiceprint_result: Optional[Dict[str, Any]] = None,
    audio_weight: float = config.FUSION_AUDIO_WEIGHT,
    keyword_weight: float = config.FUSION_KEYWORD_WEIGHT,
) -> Dict[str, float]:
    """Computes base fused risk and adjusts it with situational call context.

    Computes ``base_fused_score`` via ``fuse_risk()`` first, then applies
    transaction and contact familiarity multipliers sequentially:
      ``contextual_score = base_fused_score * transaction_multiplier * contact_multiplier``
    clipped to ``[0.0, 1.0]``.

    Returning both base and contextual scores enables explaining why a score
    shifted (e.g. audio was borderline, but a fund-transfer context from an
    unverified caller pushed the risk to high).

    Parameters
    ----------
    audio_score:
        Acoustic synthetic probability or streaming score in ``[0.0, 1.0]``.
    keyword_risk_score:
        Transcript red-flag keyword score in ``[0.0, 1.0]``.
    transaction_context:
        Call or transaction type identifier (e.g. ``"general_conversation"``,
        ``"otp_request"``, ``"fund_transfer"``, ``"confidential_info_request"``).
    voiceprint_result:
        Voiceprint verification result dict (``{"match": bool, ...}``) or ``None``.
    audio_weight:
        Acoustic signal weight (default: 0.7).
    keyword_weight:
        Keyword signal weight (default: 0.3).

    Returns
    -------
    dict
        {
            "base_fused_score": float,
            "contextual_score": float,
            "transaction_multiplier": float,
            "contact_multiplier": float,
        }
    """
    base_score = fuse_risk(
        audio_score=audio_score,
        keyword_risk_score=keyword_risk_score,
        audio_weight=audio_weight,
        keyword_weight=keyword_weight,
    )

    t_mult = get_transaction_multiplier(transaction_context)
    c_mult = get_contact_familiarity_multiplier(voiceprint_result)

    raw_contextual = base_score * t_mult * c_mult
    contextual_score = min(1.0, max(0.0, round(raw_contextual, 4)))

    return {
        "base_fused_score": base_score,
        "contextual_score": contextual_score,
        "transaction_multiplier": t_mult,
        "contact_multiplier": c_mult,
    }
