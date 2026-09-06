"""prevention.py — caller-facing prevention / caution messages for each risk band.

This module is the single source of truth for the text shown to users when
VoxGuard detects a possible synthetic voice.  All copy lives here as named
module-level constants so it can be reviewed and updated as plain text
without touching any UI or rendering code.

Message design notes
--------------------
* Language is direct and action-oriented, grounded in Indian cyber-cell
  guidance (MHA Cyber Dost / CERT-In best-practice advisories).
* "medium" is deliberately softer in tone: the evidence is ambiguous, and
  an over-confident warning on a genuine call erodes user trust.  The
  instruction is to pause and verify before acting, not to hang up
  immediately.
* "high" is unambiguous: end the call, do not comply, report.  The threat
  model here is a financial-fraud or SIM-swap attempt where complying even
  once causes irreversible harm.
* "low" and "inconclusive" return None — no message is shown.  Showing a
  caution on every call creates alert fatigue that trains users to dismiss
  all warnings, including real ones.

To update copy
--------------
Edit MEDIUM_RISK_MESSAGE or HIGH_RISK_MESSAGE below.  The rest of the
codebase (app.py, future API endpoints) calls get_prevention_message() and
will pick up the new text automatically — nothing else needs to change.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Message copy — edit here, nowhere else
# ---------------------------------------------------------------------------

MEDIUM_RISK_MESSAGE: str = (
    "⚠️ Be cautious — this call shows signs that may be consistent with an "
    "AI-generated voice, though the evidence is not conclusive.\n\n"
    "- Verify before you act: hang up and call the person back on a number "
    "you already have saved, not one given to you on this call.\n"
    "- Do not share OTPs, UPI PINs, passwords, or account details under "
    "any sense of urgency — legitimate callers will not pressure you.\n"
    "- If the caller claims to be from a bank, government body, or known "
    "contact, confirm their identity through an official channel before "
    "taking any financial or account action."
)

HIGH_RISK_MESSAGE: str = (
    "🚨 High-confidence alert — strong indicators of an AI-cloned voice "
    "detected on this call.\n\n"
    "- End this call now. Do not comply with any instruction given during "
    "this call — transfers, UPI requests, OTPs, passwords, or account "
    "changes.\n"
    "- Call back the person this caller claimed to be using a number from "
    "your own contacts, not from this call.\n"
    "- Report the incident: dial the National Cyber Crime Helpline "
    "(1930) or file a complaint at cybercrime.gov.in. Preserve any "
    "call recordings or screenshots as evidence.\n"
    "- Warn others who may receive calls from the same number."
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_prevention_message(band: str) -> str | None:
    """Return the prevention / caution message for a given risk band.

    Parameters
    ----------
    band:
        One of ``"low"``, ``"medium"``, ``"high"``, or ``"inconclusive"``
        as returned by :func:`voxguard.risk.bands.score_to_band`.

    Returns
    -------
    str | None
        The message string for ``"medium"`` or ``"high"``.
        ``None`` for ``"low"`` and ``"inconclusive"`` — callers should
        render nothing (no alert fatigue on safe / ambiguous input).

    Raises
    ------
    ValueError
        If *band* is not one of the four recognised values, to surface
        unexpected inputs early rather than silently returning None.

    Examples
    --------
    >>> get_prevention_message("low") is None
    True
    >>> get_prevention_message("inconclusive") is None
    True
    >>> msg = get_prevention_message("medium")
    >>> msg.startswith("⚠️")
    True
    >>> msg = get_prevention_message("high")
    >>> msg.startswith("🚨")
    True
    """
    _VALID = {"low", "medium", "high", "inconclusive"}
    if band not in _VALID:
        raise ValueError(
            f"Unknown risk band {band!r}. Expected one of {sorted(_VALID)}."
        )

    if band == "medium":
        return MEDIUM_RISK_MESSAGE
    if band == "high":
        return HIGH_RISK_MESSAGE
    return None  # "low" and "inconclusive"
