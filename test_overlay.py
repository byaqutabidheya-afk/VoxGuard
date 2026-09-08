from src.voxguard.explain.overlay import render_explainability_overlay
from src.voxguard.classifier.ensemble import WeightedAverageDetector
from src.voxguard.utils.audio_io import load_audio

det = WeightedAverageDetector(
    wav2vec2_classifier_path="models/classifiers/wav2vec2_hindi_combined_logreg.joblib",
    wavlm_classifier_path="models/classifiers/wavlm_hindi_combined_logreg.joblib",
)

wf, sr = load_audio("data/raw/hindi_hinglish/synthetic/byaquta_neutral_01_clone.wav", target_sr=16000)
path = render_explainability_overlay(wf, sr, det, "test_overlay_synthetic.png")
print("Saved to:", path)

wf2, sr2 = load_audio("data/raw/hindi_hinglish/real/byaquta_neutral_01.wav", target_sr=16000)
path2 = render_explainability_overlay(wf2, sr2, det, "test_overlay_real.png")
print("Saved to:", path2)
