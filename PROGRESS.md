# VoxGuard Build Progress

Last updated: 2026-09-04

## Status: Phase 2 complete, Phase 3 starting (Prompt 3.1 built, about to run)

---

## Environment
- Local: Windows, project root `D:\VoxGuard`, venv at `.venv`
- Kaggle: username `byaqutabidheyabehera`, notebook slug `voxguard`
  (`https://www.kaggle.com/code/byaqutabidheyabehera/voxguard`)
- Kaggle dataset (Phase 1 packaged data): `byaqutabidheyabehera/voxguard-preprocessed-data`
  — mounts at `/kaggle/input/datasets/byaqutabidheyabehera/voxguard-preprocessed-data/`
  (note: nested under `datasets/<username>/`, not directly under `/kaggle/input/<slug>/`)
- Known Kaggle gotchas: repo clone is case-sensitive (`VoxGuard`, not `voxguard`); numpy
  needs downgrading (`pip install "numpy<2.0.0"` + `os._exit(00)` restart) almost every fresh
  session due to base-image/requirements.txt conflict; PYTHONPATH must be set explicitly for
  `!python scripts/...` subprocess calls (`os.environ["PYTHONPATH"] = "/kaggle/working/VoxGuard/src"`)
  AND separately via `sys.path.insert()` for in-kernel imports.

## Phase 1 — Data pipeline: DONE
- ASVspoof2019 LA: 121,461 rows (train 25,380 / dev 24,844 / eval 71,237), official splits preserved
- WaveFake: full set 131,083 rows; stratified subset (Prompt 1.2b) = 8,000 rows in
  `data/metadata/wavefake_subset.csv`
- In-the-Wild: 5,000-row stratified subset in `data/metadata/in_the_wild.csv`
- `data/metadata/unified.csv`: 134,461 rows, `processed_path` populated (relative paths) after
  Prompt 1.4 preprocessing
- Kaggle combined dataset uploaded: `byaqutabidheyabehera/voxguard-preprocessed-data`
  (asvspoof2019/, wavefake/, in_the_wild/, metadata/unified.csv — ~10.89GB packaged)

## Phase 2 — Embeddings, prosody, classifier: DONE
- **Two real bugs found and fixed after initial Phase 2 completion — see "Post-Phase-2 fixes" below.**
- All ten embedding caches present in `models/embeddings/` (wav2vec2_{train,dev,eval,wavefake,in_the_wild}.npy
  + wavlm_ equivalents), **re-extracted on Kaggle after Fix 1** (see below) — do not trust any
  version of these files older than the Fix 1 patch.
- Prosody features cached: `prosody_{train,dev,eval}.npy` (ASVspoof2019 only — Phase 2 selected
  the BASELINE, non-prosody-augmented classifier, so WaveFake/In-the-Wild prosody caches were
  never built and are not needed)
- Four classifiers trained (`models/classifiers/`): baseline_logreg, baseline_mlp, prosody_logreg,
  prosody_mlp — retrained post-fix
- **Decision (Phase 2, Prompt 2.6): `baseline_logreg.joblib` is the chosen model.**
  EER 0.109, AUC 0.955, on ASVspoof2019 eval (post-fix, standardized). Prosody did not
  meaningfully help logreg (+0.06pp, noise) and MLP underperformed logreg outright.
  Recorded in `models/reports/decision_notes.md`.
- `VoxGuardDetector` (`src/voxguard/classifier/infer.py`) implemented and verified:
  - `predict()` verified against real ground truth at scale: 90.5% real recall (6655/7355),
    88.3% synthetic recall (56429/63882) on ASVspoof2019 eval — matches evaluate_classifier.py's
    numbers, confirming the live inference path and offline eval agree.
  - `predict_waveform()` verified to exactly match `predict()`'s output on the same file.
  - Default classifier_path points to `baseline_logreg.joblib`.
- All 25 tests in `tests/` pass.

### Post-Phase-2 fixes (important — read before touching embeddings or classifiers)
1. **Fix 1 — Batching corruption in `extract_and_cache`** (`src/voxguard/embeddings/cache.py`):
   naive batching mixed clips up to 9.1x apart in duration; wav2vec2/WavLM's positional conv
   smeared zero-padding into real frames, corrupting ~78% of cached embeddings (mean cosine
   similarity only 0.687 vs. clean extraction). Fixed by sorting by duration and batching
   length-adjacent clips, then scattering results back to original row order. **All ten Kaggle
   embedding caches were re-extracted after this fix** — the original Phase 2 Kaggle session's
   output was discarded.
2. **Fix 2 — Missing feature standardization** (`src/voxguard/classifier/head.py`): F0 prosody
   features are ~1000x the magnitude of embedding dims, distorting both lbfgs convergence and MLP
   training, and originally masked prosody's real effect. Fixed by fitting a `StandardScaler` per
   feature-set (baseline 768-dim, prosody-augmented 778-dim) and persisting it alongside the
   model. **This changed the save/load contract**: `load_classifier()` now returns
   `(model, scaler)`, not just `model`. `VoxGuardDetector` applies `scaler.transform()` before
   scoring — any new code touching inference must do the same or predictions will be silently
   wrong (not obviously broken).
- Both fixes together took EER from ~25-30% (broken) down to 10.9% (baseline_logreg, correct).

## Phase 3 — Cross-dataset generalization: DONE
- Prompt 3.1 (`src/voxguard/classifier/cross_eval.py`, `scripts/run_cross_eval.py`): `zero_shot_eval`,
  `resolve_cache_path`, `zero_shot_eval_from_cache`, `zero_shot_eval_weighted_average_from_cache`.
- Prompt 3.2: WavLM support confirmed via unit test. `models/classifiers/wavlm_logreg.joblib`
  trained (baseline/logreg only, matching Phase 2's choice) — EER 0.1028 on ASVspoof2019 eval
  (vs wav2vec2's 0.1090), standardized with its own persisted scaler.
- Prompt 3.3 (`src/voxguard/classifier/ensemble.py`, `scripts/train_ensemble_classifier.py`):
  `EnsembleDetector` (concatenated, 1536-dim, `models/classifiers/ensemble_logreg.joblib`) and
  `WeightedAverageDetector` (both live in `ensemble.py`) implemented. Both verified working
  end-to-end (`predict()`/`predict_waveform()` consistent).
- Prompt 3.4: full 4-variant x 3-dataset report in `models/reports/generalization_before_after.md`.

  | dataset | wav2vec2-only | WavLM-only | concatenated ensemble | weighted-average ensemble |
  |---|---|---|---|---|
  | ASVspoof2019 eval | 88.6% / 10.9% | 89.7% / 10.3% | 90.5% / 9.1% | **91.7% / 7.9%** |
  | WaveFake | 88.1% / 31.0% | 87.4% / 34.9% | 73.9%\* / 30.3% | 89.0% / 30.4% |
  | In-the-Wild | 50.9% / 20.1% | 47.4% / 20.1% | 47.0% / 23.0% | 45.3% / **16.0%** |

  (\*concatenated ensemble's WaveFake accuracy looks anomalous — high EER-vs-accuracy divergence
  indicates default-threshold miscalibration on this dataset, not a genuinely worse model; its
  EER is actually competitive. Root cause not fully investigated — worth a footnote in any
  presentation of this table.)

  In-the-Wild accuracy is weak (45-51%) across ALL four variants uniformly, while EER is much
  more reasonable (16-23%) — points to a global threshold-calibration/class-balance mismatch
  between ASVspoof (what thresholds were tuned on) and In-the-Wild, rather than a fundamental
  inability to separate real/fake. Not yet root-caused — worth revisiting if time allows,
  otherwise report honestly as a known limitation.

- **DECISION: weighted-average ensemble (`WeightedAverageDetector`,
  `src/voxguard/classifier/ensemble.py`) is the chosen production detector**, superseding
  Phase 2's `VoxGuardDetector`/`baseline_logreg`. Best or near-best EER on all three datasets,
  and — critically — does not suffer the threshold-collapse failure mode the concatenated
  ensemble shows on WaveFake. Constructor takes no required args; defaults already point at
  `baseline_logreg.joblib` + `wavlm_logreg.joblib`, weight_a=0.5.
- **Calibration bug found and fixed (post-Phase-3):** In-the-Wild's near-coin-flip accuracy and
  the concatenated ensemble's anomalous WaveFake accuracy were BOTH the same root cause —
  threshold miscalibration, not a modeling bug. ASVspoof/WaveFake are ~90% spoof; In-the-Wild is
  62.8% bonafide (inverted). Classifiers trained against the 90%-spoof world push probabilities
  systematically high out-of-domain (mean P(synth) on genuine speech: 0.105 ASVspoof / 0.869
  WaveFake / 0.769 In-the-Wild), so the default 0.5 threshold mislabels most genuine In-the-Wild
  speech as synthetic. AUC/EER were fine throughout (0.836/0.20) — confirms ranking ability was
  never the problem, only the cutoff. Preprocessing ruled out as a cause (verified identical
  16kHz/mono/PCM_16 + silence-trim across all three datasets via file headers, not metadata).
  Ensemble concatenation order also ruled out (verified column order matches scaler mean at
  eval time).
  FIX: added a `threshold` parameter (default preserves old 0.5 behavior, byte-identical
  outputs verified) to `VoxGuardDetector`, `EnsembleDetector`, `WeightedAverageDetector`'s
  `predict()`/`predict_waveform()`, plus a `threshold_from_eval(metrics)` helper to pull a
  calibrated cutoff from any eval result. Only the label changes with threshold, never the
  probability. In-the-Wild accuracy at its own eer_threshold (0.999998): 45.3% -> 84.0%,
  exactly matching the 1-EER ceiling.
  **CAVEAT — carry forward to Phase 5:** that 0.999998 threshold was fit ON In-the-Wild itself,
  so it's optimistic/overfit as a real deployment value. Do NOT hardcode it. Before the live
  demo, calibrate on a genuinely held-out real-world sample (own/teammate recordings) instead,
  or at minimum use a threshold derived from a dataset not used to select it.
  Full writeup appended to `models/reports/generalization_before_after.md`.

## Not started
- Phase 4 (Hindi/Hinglish track)
- Phase 5 (real-time streaming + challenge-response)
- Speaker voiceprint verification, multimodal call-context fusion, explainability overlay,
  Gradio demo, privacy module

## Key naming conventions (for consistency across all future prompts)
- ASVspoof2019 embedding cache: `models/embeddings/{model}_{split}.npy` (e.g. `wav2vec2_train.npy`)
- WaveFake/In-the-Wild embedding cache: `models/embeddings/{model}_{dataset}.npy`
  (e.g. `wav2vec2_wavefake.npy`)
- Prosody cache: `models/embeddings/prosody_{split}.npy` (ASVspoof) or
  `models/embeddings/prosody_{dataset}.npy` (others)
- Classifiers: `models/classifiers/{featureset}_{type}.joblib` (e.g. `baseline_logreg.joblib`),
  with a JSON sidecar of the same base name holding `{"type", "input_dim", "scaler_path"}`
