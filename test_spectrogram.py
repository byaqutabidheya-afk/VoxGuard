from src.voxguard.explain.spectrogram import generate_mel_spectrogram, render_spectrogram_image
from src.voxguard.utils.audio_io import load_audio

wf, sr = load_audio("data/raw/hindi_hinglish/real/byaquta_neutral_01.wav", target_sr=16000)
spec = generate_mel_spectrogram(wf, sr=sr)
path = render_spectrogram_image(spec, sr=sr, output_path="test_spectrogram.png")
print("Saved to:", path)
