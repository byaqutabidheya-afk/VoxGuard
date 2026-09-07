"""Unit tests for red-flag keyword scanning (Phase 7 / Prompt 9.3)."""

from voxguard.fusion.redflags import RED_FLAG_PHRASES, scan_for_redflags


def test_neutral_sentence_no_matches():
    """A neutral sentence with no scam indicators should produce keyword_risk_score == 0."""
    text = "Weather bahut accha hai aaj, chalo evening walk pe chalte hain."
    res = scan_for_redflags(text)
    assert res["keyword_risk_score"] == 0.0
    assert res["matched_phrases"] == []
    assert res["categories"] == []


def test_single_financial_action():
    """A sentence with one financial_action phrase should produce a moderate score."""
    text = "Please send money to my account when you get a chance."
    res = scan_for_redflags(text)
    assert "send money" in res["matched_phrases"]
    assert "financial_action" in res["categories"]
    assert 0.20 <= res["keyword_risk_score"] <= 0.50


def test_combined_multi_category_high_risk():
    """Combining urgency + financial_action + authority_impersonation phrases yields a high score."""
    text = "This is police from customs department, you must transfer money right now or face arrest."
    res = scan_for_redflags(text)

    # Must detect all 3 requested categories
    assert "urgency" in res["categories"]
    assert "financial_action" in res["categories"]
    assert "authority_impersonation" in res["categories"]

    # Score must be high and greater than a single financial action phrase
    single_res = scan_for_redflags("Please send money.")
    assert res["keyword_risk_score"] > single_res["keyword_risk_score"]
    assert res["keyword_risk_score"] >= 0.80


def test_hinglish_transliterated_matches():
    """A Hindi/Hinglish sentence with a transliterated red-flag phrase matches (non-English-only)."""
    text = "Yeh customs department se call hai, abhi turant paise transfer kijiye, kisi ko mat batana."
    res = scan_for_redflags(text)

    assert any(p in res["matched_phrases"] for p in ["abhi turant", "turant"])
    assert any(p in res["matched_phrases"] for p in ["paise transfer", "paise transfer kijiye"])
    assert any(p in res["matched_phrases"] for p in ["kisi ko mat batana"])

    assert "urgency" in res["categories"]
    assert "financial_action" in res["categories"]
    assert "isolation" in res["categories"]
    assert "authority_impersonation" in res["categories"]
    assert res["keyword_risk_score"] >= 0.80


def test_apostrophe_normalization_matches():
    """Confirms both 'don't tell anyone' and 'dont tell anyone' match the isolation category."""
    # Standard straight apostrophe
    res_with_apostrophe = scan_for_redflags("Please don't tell anyone about this transaction.")
    assert "don't tell anyone" in res_with_apostrophe["matched_phrases"]
    assert "isolation" in res_with_apostrophe["categories"]

    # Without apostrophe (common in ASR output)
    res_without_apostrophe = scan_for_redflags("Please dont tell anyone about this transaction.")
    assert "don't tell anyone" in res_without_apostrophe["matched_phrases"]
    assert "isolation" in res_without_apostrophe["categories"]

    # Typographic/curly apostrophe
    res_curly = scan_for_redflags("Please don’t tell anyone about this transaction.")
    assert "don't tell anyone" in res_curly["matched_phrases"]
    assert "isolation" in res_curly["categories"]


def test_empty_and_edge_inputs():
    """Empty or non-string inputs should return safe zero-score dicts."""
    assert scan_for_redflags("") == {"matched_phrases": [], "categories": [], "keyword_risk_score": 0.0}
    assert scan_for_redflags("   ") == {"matched_phrases": [], "categories": [], "keyword_risk_score": 0.0}
    assert scan_for_redflags(None) == {"matched_phrases": [], "categories": [], "keyword_risk_score": 0.0}
