Activate Python Virtual Environment: .\.venv\Scripts\Activate.ps1
To check which python version the venv is using: .\.venv\Scripts\python.exe --version

Real example test: python -m src.voxguard.classifier.infer D:/VoxGuard/data/raw/asvspoof2019/LA/ASVspoof2019_LA_eval/flac/LA_E_2714189.flac

Synthetic example test: python -m src.voxguard.classifier.infer D:/VoxGuard/data/raw/asvspoof2019/LA/ASVspoof2019_LA_eval/flac/LA_E_8624167.flac

byaquta 1 data/raw/hindi_hinglish/synthetic/byaquta_neutral_01_clone.wav
byaquta 6 data/raw/hindi_hinglish/synthetic/byaquta_neutral_06_clone.wav
byaquta 10 data/raw/hindi_hinglish/synthetic/byaquta_neutral_10_clone.wav
byaquta 16 data/raw/hindi_hinglish/synthetic/byaquta_scam_16_clone.wav
mahato 1 data/raw/hindi_hinglish/synthetic/mahato_neutral_01_clone.wav
mahato 6 data/raw/hindi_hinglish/synthetic/mahato_neutral_06_clone.wav
mahato 10 data/raw/hindi_hinglish/synthetic/mahato_neutral_10_clone.wav
mahato 16 data/raw/hindi_hinglish/synthetic/mahato_scam_16_clone.wav
soumya 1 data/raw/hindi_hinglish/synthetic/soumya_neutral_01_clone.wav
soumya 6 data/raw/hindi_hinglish/synthetic/soumya_neutral_06_clone.wav
soumya 10 data/raw/hindi_hinglish/synthetic/soumya_neutral_10_clone.wav
soumya 16 data/raw/hindi_hinglish/synthetic/soumya_scam_16_clone.wav
byaquta 24 data/raw/hindi_hinglish/synthetic/byaquta_control_24_clone.wav
mahato 24 data/raw/hindi_hinglish/synthetic/mahato_control_24_clone.wav
soumya 24 data/raw/hindi_hinglish/synthetic/soumya_control_24_clone.wav

To start the gradio app:  python app\app.py