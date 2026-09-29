import numpy as np
from src.voxguard.classifier.ensemble import WeightedAverageDetector
from src.voxguard.utils.audio_io import load_audio

det = WeightedAverageDetector(
    wav2vec2_classifier_path="models/classifiers/wav2vec2_hindi_combined_logreg.joblib",
    wavlm_classifier_path="models/classifiers/wavlm_hindi_combined_logreg.joblib",
)

clips = [
    ("REAL", "data/raw/hindi_hinglish/real/byaquta_neutral_01.wav"),
    ("REAL", "data/raw/hindi_hinglish/real/soumya_control_21.wav"),
    ("REAL", "data/raw/hindi_hinglish/real/soumya_scam_11.wav"),
    ("SYNTH", "data/raw/hindi_hinglish/synthetic/byaquta_neutral_01_clone.wav"),
    ("SYNTH", "data/raw/hindi_hinglish/synthetic/soumya_scam_11_clone.wav"),
]

for win in [1.5, 3.0, 4.0, 6.0]:
    print(f"=== window={win}s ===")
    for kind, path in clips:
        wf, sr = load_audio(path, target_sr=16000)
        n = int(win * sr)
        stride = int(1.0 * sr)
        scores = []
        for start in range(0, max(1, len(wf) - n + 1), stride):
            w = wf[start:start+n]
            if len(w) < sr:
                continue
            r = det.predict_waveform(w, sr)
            p = r.get("probability_synthetic")
            if p is not None:
                scores.append(p)
        if not scores:
            wfull = wf
            r = det.predict_waveform(wfull, sr)
            scores = [r.get("probability_synthetic", float("nan"))]
        print(f"  {kind:6s} {path.split('/')[-1]:38s} mean={np.mean(scores):.3f} max={np.max(scores):.3f}")
    print()
