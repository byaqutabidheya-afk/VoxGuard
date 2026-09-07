from src.voxguard.fusion.redflags import scan_for_redflags

test_cases = [
    "Sir, your bank account will be blocked today, if you dont share your OTP right now.",
    "Hows the weather today, are we still meeting for lunch tomorrow?",
    "Please send money urgently, dont tell anyone about this call.",
    "Yeh customs department se call hai, turant fine pay kijiye.",
]

for text in test_cases:
    result = scan_for_redflags(text)
    print("---")
    print("Text:", text)
    print("Matched phrases:", result["matched_phrases"])
    print("Categories:", result["categories"])
    print("Keyword risk score:", result["keyword_risk_score"])
