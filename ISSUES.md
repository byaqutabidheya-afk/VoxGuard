# VoxGuard — Known Issues Log

Last updated: 2026-09-30 (updated after Phase F3 remediation)

This document tracks known architectural limitations, acoustic confounds, and model behaviors
identified during VoxGuard's build and remediation phases. Every issue is documented with its
empirical symptoms, root cause, and remediation status.

---

## The Causal Chain: How These Issues Interact

```
Issue 1 (duration confound)
   │  Hindi real clips avg 4.95s, synthetic avg 7.44s.
   │  The whole-clip classifier partly learned "long = synthetic".
   │
   ├──► Issue 2 (chunk-level detection fails in streaming/attribution)
   │      Fixed-size windows (1.5s) are ALL the same length.
   │      The duration signal the whole-clip classifier leaned on is
   │      structurally ABSENT from every chunk. The model falls back on
   │      noise → confident false positives on real speech.
   │
   └──► Issue 3 (WavLM head confidently wrong on real speech)
          WavLM's Hindi training rows were drowned out by 25k+ ASVspoof rows,
          and the head overfit at C=1, predicting real speech as synthetic.
```

---

## Issue 1 — Duration Confound in Hindi/Hinglish Dataset

### Description
In the original Hindi/Hinglish track (75 real + 75 synthetic clips across 3 speakers), real clips
averaged **4.95s** while XTTS-v2 synthetic clones averaged **7.44s**. A simple logistic regression
trained solely on scalar `[duration, rms]` achieved **83.33% cross-validated accuracy**, proving that
a large portion of reported whole-clip classification accuracy was propped up by clip duration.

### Phase F1 Resolution (2026-09-29)
- **Fix:** Implemented `duration_match_pair` (symmetrical trimming from both ends, never padding) in
  `src/voxguard/utils/duration_match.py`. Rebuilt corpus as `data/raw/hindi_hinglish_matched/` (75 matched pairs).
- **Verification:** Duration+RMS-alone accuracy dropped to **50.67%** (near chance).
- **Retraining:** Retrained whole-clip heads on duration-matched data (`wav2vec2_hindi_matched_logreg.joblib`,
  `wavlm_hindi_matched_logreg.joblib`). On the honest matched test set, weighted-average ensemble achieved
  **98.00% accuracy / 4.00% EER** with zero English regression on ASVspoof2019 (91.54% acc / 7.67% EER).
- **Status:** **RESOLVED.**

---

## Issue 2 — Chunk-Level Detection Reliability Gap (Streaming & Overlay)

### Description
Streaming inference (`StreamingSession`) and explainability overlay (`windowed_attribution`) evaluate
audio in short, fixed-length windows (1.5s). Whole-clip classifiers failed on short windows because
they relied on full-clip duration cues that are absent in fixed chunks, producing false alarms on real speech
(e.g., `byaquta_neutral_01.wav` and `soumya_control_21.wav` triggered false positive flags in streaming).

### Phase F2 Resolution (2026-09-30)
- **Fix:** Built chunk-native dataset and trained dedicated chunked classifiers
  (`wav2vec2_chunked_logreg.joblib`, `wavlm_chunked_logreg.joblib`) on 1.5s windows with anti-leakage clip-level partitioning.
- **Verification:** Chunk-level EER on ASVspoof eval reached **7.91%** (<25% gate target) and **28.49%** on Hindi matched eval.
- **Streaming Verification:** Replayed failing test clips through `scripts/fix_verify_streaming.py`; `byaquta_neutral_01.wav`
  false positive was resolved.
- **Status:** **RESOLVED for chunked architecture** (pending threshold recalibration in F4 and UI/API wiring in F5).

---

## Issue 3 — WavLM Head Disagreement on Unseen Real Speech

### Description
On held-out speaker `soumya`'s real clips, the baseline `wavlm_hindi_combined_logreg.joblib` head
confidently predicted real speech as synthetic ($P \approx 0.997\text{--}1.000$), directly contradicting
`wav2vec2_hindi_combined_logreg.joblib` ($P \approx 0.000$). The resulting 50/50 ensemble sat at
$P \approx 0.5000$ (an arithmetic artifact of conflict, not true model uncertainty). In baseline capture,
only **7 / 25** of `soumya`'s real clips had an ensemble margin $|P - 0.5| > 0.15$.

## Phase F3 resolution (2026-09-30)

Diagnosed via scripts/fix_diagnose_wavlm.py: hypothesis A (speaker shortcut) was tested directly
and REJECTED — WavLM is not more speaker-dominated than wav2vec2, and the same-speaker-to-soumya
generalization gap is no worse for WavLM. The actual cause is a combination of hypotheses B and C:
WavLM's Hindi training rows (100-398, depending on family) were drowned out by ~25,000+ ASVspoof
rows, and the resulting head overfit at default regularization (C=1). Fixed via
scripts/fix_retrain_wavlm_v2.py: C=0.001, Hindi rows oversampled 20x, selected via
leave-one-Hindi-speaker-out validation (never chosen by looking at soumya directly).

**Matched whole-clip family: RESOLVED.** WavLM alone on soumya: AUC 0.690 -> 0.992. Ensemble
confident-and-correct: 5/25 -> 25/25 (Gate F3 target: >=18/25, MET). Synthetic recall held at
100%. Cost: ASVspoof2019 whole-clip EER moved 7.67% -> 8.62%, synthetic recall 91.18% -> 89.16%
(within the 3pp tolerance, but using most of the allowed margin — a real, documented trade-off
for fixing Hindi held-out-speaker reliability).

**Chunked family: PARTIALLY RESOLVED.** WavLM alone on soumya: AUC 0.694 -> 0.979 (backbone fix
succeeded here too). Ensemble confident-and-correct: 10/25 -> 14/25 (target: >=18/25, NOT MET).
Root cause of the shortfall is now understood, not unexplained: wav2vec2's OWN chunked head only
reaches 13/25 alone on soumya, so no fix to WavLM could push the ensemble past that ceiling.
Reweighting was considered and rejected — the diagnosis's reweight sweep showed no weight_a in
(0.5, 1.0) exceeds 11/25 for this family, worse than the retrain result, and reweighting toward
wav2vec2 would mean quietly dropping a backbone that is now demonstrably fixed. 14/25 is recorded
as a genuine improvement over the 7/25 baseline, short of target, carried forward honestly rather
than chased via reweighting. Chunk-level EER improved substantially on Hindi (28.49% -> 13.44%).

Both v2 WavLM heads (wavlm_hindi_matched_v2_logreg.joblib, wavlm_chunked_v2_logreg.joblib) are
selected for production, pending F5 propagation. wav2vec2 heads unchanged in both families. All
v1 files preserved on disk per the guide's non-negotiable naming/rollback constraints.
