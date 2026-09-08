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
Gradio version: 4.44.1

jinja2 details:
Name: Jinja2
Version: 3.1.6
Summary: A very fast and expressive template engine.
Home-page: 
Author: 
Author-email: 
License: 
Location: D:\VoxGuard\.venv\Lib\site-packages
Requires: MarkupSafe
Required-by: gradio, torch

### Benchmark 10-Clip File List                                                                                              
                                                                                                                               
  #### Pair 1: Casual Neutral (byaquta)                                                                                        
                                                                                                                               
  • Real File: byaquta_neutral_09.wav                                                                                          
  • Synthetic Clone: byaquta_neutral_09_clone.wav                                                                              
  • Sentence: "Weather bahut accha hai aaj, chalo evening walk pe chalte hain."                                                
  ──────                                                                                                                       
  #### Pair 2: Everyday Tech (mahato)                                                                                          
                                                                                                                               
  • Real File: mahato_neutral_04.wav                                                                                           
  • Synthetic Clone: mahato_neutral_04_clone.wav                                                                               
  • Sentence: "Mera phone ka battery bahut fast drain ho raha hai these days."                                                 
  ──────                                                                                                                       
  #### Pair 3: Urgent Legal Pressure Scam (byaquta)                                                                            
                                                                                                                               
  • Real File: byaquta_scam_16.wav                                                                                             
  • Synthetic Clone: byaquta_scam_16_clone.wav                                                                                 
  • Sentence: "This is an urgent matter sir, agar aap abhi payment nahi karte to legal action liya jayega."                    
  ──────                                                                                                                       
  #### Pair 4: Authority Customs Scam (mahato)                                                                                 
                                                                                                                               
  • Real File: mahato_scam_12.wav                                                                                              
  • Synthetic Clone: mahato_scam_12_clone.wav                                                                                  
  • Sentence: "Yeh customs department se call hai, aapke parcel mein illegal items mile hain, turant fine pay kijiye."         
  ──────                                                                                                                       
  #### Pair 5: Held-Out Casual Speaker (soumya)                                                                                
                                                                                                                               
  • Real File: soumya_neutral_03.wav                                                                                           
  • Synthetic Clone: soumya_neutral_03_clone.wav                                                                               
  • Sentence: "Kal weekend hai na, let's plan a trip to the hills."     


  Pick one pair as your primary demo script — I'd lean toward DEMO-03 or DEMO-04 (the scam-phrasing ones) since they're thematically the most compelling for a voice-scam-detection demo — "watch it catch a fake IRS/customs call" is a stronger story than a neutral weather comment.
Rehearse with exactly that script, exactly that speaker, exactly this app — not a variation, not "something similar." The whole lesson from this investigation is that behavior isn't uniformly predictable across arbitrary content; stick to what's verified.
Have a backup pair ready (e.g., DEMO-01 or DEMO-05) in case something goes wrong live with the primary — mic issue, background noise, nerves changing your pacing.



LOW RISK example:
  D:/VoxGuard/data/raw/asvspoof2019/LA/ASVspoof2019_LA_eval/flac/LA_E_5849185.flac | prob: 0.011

MEDIUM RISK example:
  D:/VoxGuard/data/raw/asvspoof2019/LA/ASVspoof2019_LA_eval/flac/LA_E_2608310.flac | prob: 0.4563

HIGH RISK example:
  D:/VoxGuard/data/raw/asvspoof2019/LA/ASVspoof2019_LA_eval/flac/LA_E_2834763.flac | prob: 0.9997


API starting command: 
cd D:\VoxGuard
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --port 8000

API ping command: 
curl.exe http://127.0.0.1:8000/health