"""
redflags.py — scam keyword & red-flag phrase scanner for call transcripts (Phase 7).

Scans transcribed text (from English, Hindi, and code-switched Hinglish speech)
for high-risk semantic patterns indicative of voice fraud, social engineering,
and phone scams.

Categories & Weights:
─────────────────────
  - "urgency": High-pressure time constraints (e.g. "right now", "abhi turant").
  - "financial_action": Demands for money or credentials (e.g. "send money", "OTP", "UPI", "bank details").
  - "authority_impersonation": Fake law enforcement or official bodies (e.g. "police", "customs", "arrest warrant", "legal action").
  - "isolation": Pressuring victim to conceal call (e.g. "don't tell anyone", "kisi ko mat batana").
  - "account_threat": Threats of service cutoff or freezing (e.g. "account blocked", "service disconnect", "kyc expired").

Note on ASR Behavior:
─────────────────────
Empirical testing shows Whisper sometimes auto-translates code-switched Hindi/Hinglish
speech to English. Therefore, RED_FLAG_PHRASES provides comprehensive coverage in both
English and transliterated Hindi/Hinglish.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from voxguard.utils.logging_utils import get_logger

logger = get_logger(__name__)

# List of (phrase_or_pattern, category, weight)
# Keep in one clearly editable place for easy tuning.
RED_FLAG_PHRASES: List[Tuple[str, str, float]] = [
    # -------------------------------------------------------------------------
    # 1. Urgency (Time pressure / panic creation)
    # -------------------------------------------------------------------------
    ("right now", "urgency", 0.30),
    ("immediately", "urgency", 0.30),
    ("urgent", "urgency", 0.25),
    ("urgently", "urgency", 0.30),
    ("asap", "urgency", 0.25),
    ("as soon as possible", "urgency", 0.25),
    ("hurry", "urgency", 0.20),
    ("last chance", "urgency", 0.30),
    ("limited time", "urgency", 0.25),
    ("within 24 hours", "urgency", 0.30),
    ("within 1 hour", "urgency", 0.35),
    ("within 10 minutes", "urgency", 0.35),
    ("expires today", "urgency", 0.30),
    ("do not delay", "urgency", 0.25),
    ("don't delay", "urgency", 0.25),
    ("act now", "urgency", 0.25),
    # Hindi / Hinglish urgency
    ("abhi turant", "urgency", 0.35),
    ("turant", "urgency", 0.30),
    ("abhi ke abhi", "urgency", 0.35),
    ("jaldi", "urgency", 0.20),
    ("fauran", "urgency", 0.30),
    ("der mat karo", "urgency", 0.25),
    ("jaldi karo", "urgency", 0.20),
    ("turant kijiye", "urgency", 0.30),
    ("turant pay", "urgency", 0.35),

    # -------------------------------------------------------------------------
    # 2. Financial Action (Credential theft / unauthorized payments)
    # -------------------------------------------------------------------------
    ("send money", "financial_action", 0.35),
    ("transfer money", "financial_action", 0.35),
    ("wire transfer", "financial_action", 0.35),
    ("bank transfer", "financial_action", 0.30),
    ("transfer funds", "financial_action", 0.35),
    ("transfer payment", "financial_action", 0.30),
    ("payment", "financial_action", 0.20),
    ("pay fine", "financial_action", 0.35),
    ("fine pay", "financial_action", 0.35),
    ("pay now", "financial_action", 0.30),
    ("fee", "financial_action", 0.20),
    ("processing fee", "financial_action", 0.30),
    ("security deposit", "financial_action", 0.30),
    ("otp", "financial_action", 0.40),
    ("one time password", "financial_action", 0.40),
    ("upi", "financial_action", 0.35),
    ("google pay", "financial_action", 0.30),
    ("gpay", "financial_action", 0.30),
    ("phonepe", "financial_action", 0.30),
    ("paytm", "financial_action", 0.30),
    ("bank details", "financial_action", 0.35),
    ("account details", "financial_action", 0.30),
    ("account number", "financial_action", 0.30),
    ("credit card", "financial_action", 0.35),
    ("debit card", "financial_action", 0.35),
    ("card number", "financial_action", 0.35),
    ("cvv", "financial_action", 0.40),
    ("pin number", "financial_action", 0.35),
    ("atm pin", "financial_action", 0.40),
    ("net banking", "financial_action", 0.30),
    ("password", "financial_action", 0.30),
    ("gift card", "financial_action", 0.35),
    ("crypto", "financial_action", 0.30),
    ("bitcoin", "financial_action", 0.35),
    ("lottery", "financial_action", 0.35),
    ("prize money", "financial_action", 0.35),
    ("cash prize", "financial_action", 0.35),
    ("refund", "financial_action", 0.25),
    # Hindi / Hinglish financial
    ("paise transfer", "financial_action", 0.35),
    ("paise bhejo", "financial_action", 0.35),
    ("paise transfer kijiye", "financial_action", 0.35),
    ("paise chahiye", "financial_action", 0.35),
    ("paise do", "financial_action", 0.30),
    ("rakam", "financial_action", 0.25),
    ("khata sankhya", "financial_action", 0.30),
    ("khate mein", "financial_action", 0.25),
    ("bank khata", "financial_action", 0.30),
    ("rupaye bhejo", "financial_action", 0.35),
    ("fine pay kijiye", "financial_action", 0.35),
    ("fine bharna", "financial_action", 0.35),

    # -------------------------------------------------------------------------
    # 3. Authority Impersonation (Intimidation & coercion)
    # -------------------------------------------------------------------------
    ("police", "authority_impersonation", 0.35),
    ("police officer", "authority_impersonation", 0.40),
    ("police station", "authority_impersonation", 0.35),
    ("arrest", "authority_impersonation", 0.40),
    ("under arrest", "authority_impersonation", 0.45),
    ("arrest warrant", "authority_impersonation", 0.45),
    ("warrant", "authority_impersonation", 0.35),
    ("court", "authority_impersonation", 0.30),
    ("court order", "authority_impersonation", 0.40),
    ("court notice", "authority_impersonation", 0.40),
    ("judge", "authority_impersonation", 0.30),
    ("customs", "authority_impersonation", 0.40),
    ("customs department", "authority_impersonation", 0.45),
    ("cbi", "authority_impersonation", 0.45),
    ("ed department", "authority_impersonation", 0.45),
    ("enforcement directorate", "authority_impersonation", 0.45),
    ("income tax", "authority_impersonation", 0.35),
    ("tax department", "authority_impersonation", 0.35),
    ("legal action", "authority_impersonation", 0.40),
    ("cyber crime", "authority_impersonation", 0.35),
    ("cyber cell", "authority_impersonation", 0.40),
    ("rbi", "authority_impersonation", 0.35),
    ("reserve bank", "authority_impersonation", 0.35),
    ("bank manager", "authority_impersonation", 0.30),
    ("fraud department", "authority_impersonation", 0.35),
    ("telecom regulatory", "authority_impersonation", 0.35),
    ("trai", "authority_impersonation", 0.35),
    ("illegal items", "authority_impersonation", 0.40),
    ("contraband", "authority_impersonation", 0.40),
    ("drugs found", "authority_impersonation", 0.45),
    ("narcotics", "authority_impersonation", 0.40),
    ("passport blocked", "authority_impersonation", 0.35),
    # Hindi / Hinglish authority
    ("kanooni karwai", "authority_impersonation", 0.40),
    ("kanooni action", "authority_impersonation", 0.40),
    ("thaana", "authority_impersonation", 0.30),
    ("giraftar", "authority_impersonation", 0.40),
    ("giraftari", "authority_impersonation", 0.40),
    ("customs department se call", "authority_impersonation", 0.45),
    ("parcel mein illegal", "authority_impersonation", 0.45),
    ("legal action liya jayega", "authority_impersonation", 0.45),

    # -------------------------------------------------------------------------
    # 4. Isolation (Preventing verification with friends/family)
    # -------------------------------------------------------------------------
    ("don't tell anyone", "isolation", 0.40),
    ("do not tell anyone", "isolation", 0.40),
    ("don't tell your family", "isolation", 0.45),
    ("do not tell your family", "isolation", 0.45),
    ("keep this confidential", "isolation", 0.35),
    ("strictly confidential", "isolation", 0.35),
    ("confidential matter", "isolation", 0.35),
    ("secret", "isolation", 0.25),
    ("private matter", "isolation", 0.25),
    ("stay on the line", "isolation", 0.30),
    ("stay on the call", "isolation", 0.30),
    ("don't disconnect", "isolation", 0.30),
    ("do not disconnect", "isolation", 0.30),
    ("do not hang up", "isolation", 0.30),
    ("don't hang up", "isolation", 0.30),
    # Hindi / Hinglish isolation
    ("kisi ko mat batana", "isolation", 0.45),
    ("kisi ko mat batao", "isolation", 0.45),
    ("family ko mat batana", "isolation", 0.45),
    ("family ko mat batao", "isolation", 0.45),
    ("kisi se baat mat karo", "isolation", 0.40),
    ("call cut mat karna", "isolation", 0.35),
    ("phone mat kaatna", "isolation", 0.35),
    ("phone disconnect mat karna", "isolation", 0.35),
    ("raaz rakhna", "isolation", 0.30),

    # -------------------------------------------------------------------------
    # 5. Account Threat & Extortion (Creating false jeopardy)
    # -------------------------------------------------------------------------
    ("account blocked", "account_threat", 0.35),
    ("account suspended", "account_threat", 0.35),
    ("account freeze", "account_threat", 0.35),
    ("service disconnect", "account_threat", 0.35),
    ("service will be disconnected", "account_threat", 0.35),
    ("electricity cut", "account_threat", 0.35),
    ("sim card blocked", "account_threat", 0.35),
    ("sim block", "account_threat", 0.35),
    ("kyc expired", "account_threat", 0.35),
    ("kyc update", "account_threat", 0.30),
    ("aadhaar block", "account_threat", 0.35),
    ("pan card block", "account_threat", 0.35),
    ("connection cut", "account_threat", 0.30),
    # Hindi / Hinglish account threat
    ("account block ho jayega", "account_threat", 0.35),
    ("service disconnect ho jayegi", "account_threat", 0.35),
    ("bijli kat jayegi", "account_threat", 0.35),
    ("sim band ho jayega", "account_threat", 0.35),
    ("kyc band", "account_threat", 0.30),
]


def normalize_apostrophes(text: str) -> str:
    """Normalizes straight, curly, and missing apostrophes for contraction matching.

    Strips standard and typographic apostrophes/quotes (', ’, ‘, `, ´) so that
    variations like "don't", "don’t", and "dont" match reliably without punctuation
    discrepancies causing false negatives.
    """
    return re.sub(r"[\u2019\u2018\u0060\u00b4\x27]", "", text)


def scan_for_redflags(text: str) -> Dict[str, Any]:
    """Scans text for scam keywords and phrases across categories.

    Uses case-insensitive regex pattern matching with word boundaries where
    appropriate, supporting English, Hindi, and transliterated Hinglish.
    Apostrophes and contraction punctuation are normalized in both the input
    text and patterns to ensure robust matching across transcription variants
    (e.g., matching both "don't tell anyone" and "dont tell anyone").

    Parameters
    ----------
    text:
        The transcription text or dialogue string to evaluate.

    Returns
    -------
    dict
        {
            "matched_phrases": list[str],
            "categories": list[str],
            "keyword_risk_score": float (in [0.0, 1.0]),
        }
    """
    if not text or not isinstance(text, str):
        return {
            "matched_phrases": [],
            "categories": [],
            "keyword_risk_score": 0.0,
        }

    clean_text = text.strip()
    if not clean_text:
        return {
            "matched_phrases": [],
            "categories": [],
            "keyword_risk_score": 0.0,
        }

    # Normalize apostrophes in input text (e.g. "don't" / "don’t" -> "dont")
    norm_text = normalize_apostrophes(clean_text)

    matched_phrases: List[str] = []
    matched_categories_set = set()
    total_score: float = 0.0

    for phrase, category, weight in RED_FLAG_PHRASES:
        # Normalize phrase pattern to match normalized text
        norm_phrase = normalize_apostrophes(phrase)
        escaped_phrase = re.escape(norm_phrase)
        pattern = rf"(?i)\b{escaped_phrase}\b"

        if re.search(pattern, norm_text):
            matched_phrases.append(phrase)
            matched_categories_set.add(category)
            total_score += weight

    # Sort categories deterministically
    categories = sorted(list(matched_categories_set))

    # Keyword risk score capped at 1.0
    keyword_risk_score = min(1.0, round(total_score, 4))

    return {
        "matched_phrases": matched_phrases,
        "categories": categories,
        "keyword_risk_score": keyword_risk_score,
    }

