from src.voxguard.fusion.redflags import scan_for_redflags

text_with_apostrophe = "Please send money urgently, don't tell anyone about this call."
result = scan_for_redflags(text_with_apostrophe)
print("Matched phrases:", result["matched_phrases"])
print("Categories:", result["categories"])
print("Keyword risk score:", result["keyword_risk_score"])
