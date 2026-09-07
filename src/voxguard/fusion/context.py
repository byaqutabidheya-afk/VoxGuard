"""
context.py — contextual multipliers for multimodal risk fusion (Phase 7).

Provides helper functions to adjust fraud risk scores based on situational metadata:
  1. Transaction Context (e.g. fund transfer vs. general conversation).
  2. Contact Familiarity (voiceprint verification match vs. mismatch vs. unenrolled).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from voxguard import config
from voxguard.utils.logging_utils import get_logger

logger = get_logger(__name__)


def get_transaction_multiplier(context_name: str) -> float:
    """Returns the risk multiplier associated with a transaction or call context.

    Parameters
    ----------
    context_name:
        Transaction type identifier (e.g., ``"general_conversation"``,
        ``"otp_request"``, ``"fund_transfer"``, ``"confidential_info_request"``).

    Returns
    -------
    float
        Risk multiplier from ``config.TRANSACTION_CONTEXTS``. If ``context_name``
        is unknown or empty, falls back safely to ``1.0`` (neutral).
    """
    if not context_name or not isinstance(context_name, str):
        return config.TRANSACTION_CONTEXTS.get("general_conversation", 1.0)

    clean_name = context_name.strip().lower()
    if clean_name in config.TRANSACTION_CONTEXTS:
        return float(config.TRANSACTION_CONTEXTS[clean_name])

    logger.debug(
        "Unknown transaction context '%s'; defaulting to general_conversation (1.0).",
        context_name,
    )
    return float(config.TRANSACTION_CONTEXTS.get("general_conversation", 1.0))


def get_contact_familiarity_multiplier(voiceprint_result: Optional[Dict[str, Any]]) -> float:
    """Returns the contact familiarity risk multiplier from a voiceprint result.

    Three-State Evaluation Logic:
    ─────────────────────────────
      - "known_match" (0.9): Verified genuine voiceprint match mildly reduces risk.
      - "known_mismatch" (1.3): Caller claimed to be enrolled contact but failed
        voiceprint verification; elevates risk.
      - "no_enrollment_data" (1.0): No voiceprint enrolled for this contact; neutral.

    Parameters
    ----------
    voiceprint_result:
        Dictionary returned by ``verify_speaker`` (shape:
        ``{"match": bool, "similarity": float, "enrolled_name": str}``),
        or ``None`` if no voiceprint check was run.

    Returns
    -------
    float
        Multiplier from ``config.CONTACT_FAMILIARITY_MULTIPLIERS``.
        Never raises an error on malformed or missing keys.
    """
    default_multiplier = float(
        config.CONTACT_FAMILIARITY_MULTIPLIERS.get("no_enrollment_data", 1.0)
    )

    if voiceprint_result is None or not isinstance(voiceprint_result, dict):
        return default_multiplier

    if "match" not in voiceprint_result or voiceprint_result["match"] is None:
        return default_multiplier

    is_match = bool(voiceprint_result["match"])
    if is_match:
        return float(config.CONTACT_FAMILIARITY_MULTIPLIERS.get("known_match", 0.9))
    else:
        return float(config.CONTACT_FAMILIARITY_MULTIPLIERS.get("known_mismatch", 1.3))
