import numpy as np
import pandas as pd
from src.voxguard.classifier.head import load_classifier

model, scaler = load_classifier('models/classifiers/baseline_logreg.joblib')

emb = np.load('models/embeddings/wav2vec2_eval.npy')
manifest = pd.read_csv('models/embeddings/wav2vec2_eval.csv')

X_scaled = scaler.transform(emb)
probs = model.predict_proba(X_scaled)[:, 1]  # probability of synthetic class

manifest = manifest.copy()
manifest['probability_synthetic'] = probs

# Find one clean example in each band
low = manifest[manifest['probability_synthetic'] < 0.2].iloc[0]
medium = manifest[(manifest['probability_synthetic'] >= 0.4) & (manifest['probability_synthetic'] <= 0.6)]
high = manifest[manifest['probability_synthetic'] > 0.9].iloc[0]

print('LOW RISK example:')
print(' ', low['filepath'], '| prob:', round(low['probability_synthetic'], 4))
print()

if len(medium) > 0:
    med = medium.iloc[0]
    print('MEDIUM RISK example:')
    print(' ', med['filepath'], '| prob:', round(med['probability_synthetic'], 4))
else:
    print('No clean MEDIUM example found in 0.4-0.6 range, widening search...')
    medium_wide = manifest[(manifest['probability_synthetic'] >= 0.3) & (manifest['probability_synthetic'] <= 0.7)]
    if len(medium_wide) > 0:
        med = medium_wide.iloc[0]
        print(' ', med['filepath'], '| prob:', round(med['probability_synthetic'], 4))
print()

print('HIGH RISK example:')
print(' ', high['filepath'], '| prob:', round(high['probability_synthetic'], 4))
