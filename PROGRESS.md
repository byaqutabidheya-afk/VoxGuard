# VoxGuard Build Progress

Last updated: 2026-09-06

## Status: Phases 1-6 complete. Phase 7 (risk meter) not yet started. See ISSUES.md for
three known, documented-not-fixed issues before trusting any Hindi-track accuracy number in
isolation.

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

## Phase 4 — Hindi/Hinglish track: DONE (with a known, documented confound — read before trusting the numbers)
- 3 speakers (byaquta, mahato, soumya), 25 sentences each, 75 real clips + 75 matched XTTS-v2
  synthetic clips = 150 total. Consent for recording AND cloning obtained in writing from all
  three before any clone was generated.
- Two real bugs found and fixed during synthesis (Prompt 4.3): (1) `num2words` has no Hindi
  support, so any sentence with a bare digit crashes XTTS-v2's Hindi text normalization —
  worked around by spelling numbers out ("ten AM" not "10 AM"); (2) XTTS-v2 generation is
  stochastic and default settings produced frequently unintelligible output — fixed with
  temperature=0.3, repetition_penalty=10.0, split_sentences=True, and a retry-on-validation-
  failure loop (max_retries=3). Final quality: pure-Hindi sentences broadly intelligible,
  code-switched sentences ~60% intelligible. No clips were cherry-picked.
- Train/eval split (Prompt 4.6): `speaker_holdout` mode, `holdout_speaker='soumya'`. Train =
  byaquta + mahato (100 clips). Eval = soumya (50 clips), fully unseen. This holdout speaker
  MUST stay identical across Prompts 4.7/4.8/4.10 — already was, for this run.
- Phase 3 crowned the WEIGHTED-AVERAGE ensemble, not a single backbone — so Prompts 4.7/4.8/4.10
  were all adapted from their v3/base-spec form to train and evaluate wav2vec2 and WavLM
  independently (no shared feature space to build), matching WeightedAverageDetector's own
  architecture. No prosody involved anywhere (Phase 2 selected the baseline, non-prosody variant).
- Full comparison report: `models/reports/hindi_training_comparison.md`. Headline numbers:
  zero-shot production detector (weighted-average, no Hindi training) scored 50.0% acc / 20.0%
  EER on held-out Hindi; Hindi-combined production detector scored 96.0% acc / 2.0% EER, with
  no meaningful English regression (91.5% vs 91.7% ASVspoof2019 accuracy). Hindi-only training
  (no English data) caused catastrophic English collapse (61.3% ASVspoof accuracy, down from
  91.5%), confirming English data acts as a regularizer.

  **CRITICAL CAVEAT — read before citing any Hindi-track accuracy/EER number:** post-hoc
  confound analysis found that clip DURATION and RMS ENERGY ALONE (no embeddings, no acoustic
  content, just two scalar values) predict real-vs-synthetic label with 83.3% cross-validated
  accuracy on this dataset. Real clips average ~4.95s; synthetic clips average ~7.44s — a large,
  near-non-overlapping gap, most likely because XTTS-v2's `split_sentences=True` + retry-on-
  validation-failure pipeline tends to produce longer output than natural read-aloud speech.
  This means a substantial, unquantified portion of the reported 96-100% Hindi-eval accuracies
  very likely reflects this duration/energy confound rather than purely embedding-based
  synthetic-speech detection. The confound affects ALL speakers equally (it's a pipeline
  artifact, not a per-speaker one), so the speaker-holdout eval design does NOT rule it out —
  holding out a speaker only rules out speaker-identity shortcuts, not a pipeline-level
  real-vs-synthetic confound present in every pair regardless of who's speaking.
  Decision: documented as an explicit known limitation in the dataset card (Path A) rather than
  fixed now (Path B would be duration-matching real/synthetic clips and re-running Prompts
  4.7/4.8/4.10 — a real chunk of work, deferred given 7 phases still remain).
  **Anyone using these classifiers or numbers downstream (demo, slides, further phases) should
  cite them with this caveat attached, not as a clean result.**
- **Phase 4 fully closed out:** all 19 Hindi-related tests pass. `WeightedAverageDetector`
  verified working when pointed at the Hindi-combined classifiers directly:
  `WeightedAverageDetector(wav2vec2_classifier_path='models/classifiers/wav2vec2_hindi_combined_logreg.joblib',
  wavlm_classifier_path='models/classifiers/wavlm_hindi_combined_logreg.joblib')`.
  **Phase 5 onward MUST wrap this exact configuration** (the Hindi-combined pair), not the
  original English-only baseline_logreg/wavlm_logreg pair used in Phases 2-3 — this is the
  "Variant A" the guide's Definition of Done requires downstream phases to use, adapted for the
  weighted-average-ensemble case. No English regression confirmed (91.5% vs 91.7% baseline).

## Phase 6 — Gradio demo app: DONE
- `app/app.py` built across all six prompts: two-tab Blocks app (Live Mic, Upload File),
  privacy-preserving session logging, README documentation, basic layout polish.
- Wraps the Phase 4 production configuration (`WeightedAverageDetector` with
  `wav2vec2_hindi_combined_logreg.joblib` + `wavlm_hindi_combined_logreg.joblib`), consistent
  with Phase 5's streaming default.

- **Three real, durable dependency-version bugs found and fixed** (not workarounds — pinned in
  `requirements.txt`, documented in `README.md`'s "Running Locally" section so they don't get
  silently reintroduced by a future `pip install --upgrade`):
  1. Gradio 4.44.1 calling Starlette's `Jinja2Templates.TemplateResponse` with an old argument
     order that Starlette 1.0+ removed support for → fixed by pinning `starlette<1.0` and
     `fastapi<0.115` (fastapi pulls in starlette transitively; pinning starlette alone wasn't
     sufficient, fastapi had to be pinned too). A runtime monkey-patch was tried first and
     worked, but was replaced with the version pin as the durable fix — confirmed the app
     launches clean with the patch fully removed.
  2. `TypeError: argument of type 'bool' is not iterable` in `gradio_client/utils.py`'s
     `get_type()` — Pydantic 2.10+ changed how it emits `additionalProperties` in JSON schemas
     (object → bare bool), which Gradio 4.x's schema-to-Python-type conversion doesn't handle.
     Fixed by pinning `pydantic<2.10`. This crash was ALSO the hidden cause of a separate-looking
     `ValueError: When localhost is not accessible...` error — Gradio's own startup reachability
     self-check hit this same crashing code path, and Gradio misinterpreted the resulting request
     failure as "can't reach localhost." Both errors resolved together from the one pydantic fix.
  3. Mojibake corruption (`â€”` instead of `—`) in the app title/header from a UTF-8-as-Latin-1
     encoding mismatch — fixed, file re-saved as explicit UTF-8.

- **A fourth, separate bug found and fixed during Phase 6 (not a dependency issue — a real logic
  bug):** `StreamingSession` never propagated the actual incoming sample rate down to
  `StreamingBuffer`, which defaulted to assuming 16000 Hz. Gradio's browser microphone delivers
  48000 Hz. This meant every "1.5 second" streaming window during live mic use actually contained
  only 0.5 real seconds of audio (further truncated after resampling), feeding the classifier
  choppy, mid-word fragments — a highly plausible explanation for erratic live-mic behavior,
  including false positives on the user's own real voice. **Fixed:** `StreamingSession` now
  dynamically captures the real sample rate on the first `push_audio()` call and (re)builds its
  buffer to match, with a mid-session rate-mismatch guard (raises clearly if `sr` changes
  unexpectedly). Verified via 82/82 passing tests and matched 16kHz-vs-48kHz-simulated-stream
  behavior.
  - **Important:** this fix did NOT fully resolve real-voice false positives on its own — e.g.
    `byaquta_neutral_01` (real) still flags in streaming post-fix (score ~0.44-0.46). That
    residual behavior is Issue #2/#3 (see ISSUES.md), not this bug — this fix corrected a real,
    separate plumbing defect; it's a necessary fix but not sufficient to fully solve streaming
    reliability.

- **Live demo script identified and verified.** Following the sample-rate fix, benchmarked the
  actual planned demo script content (not generic test sentences) through the real
  `StreamingSession` path and found **5 verified real/synthetic pairs, spanning 3 speakers
  (including held-out speaker soumya) and 4 content categories** (casual, tech, two scam-call
  styles) that behave reliably: real clips score 0.00-0.35 (well under the 0.6 threshold,
  correctly unflagged) and synthetic clones flag confidently within exactly 2.0 seconds. Full
  details and the 5 script/speaker pairs in `PHASE5_STREAMING_NOTES.md`.
  **ACTION FOR PHASE 11: the live demo script/speaker MUST be drawn from this verified set —
  do not assume untested content or speakers will behave the same way.**

- Upload File tab shows BOTH whole-clip and streaming-simulation verdicts side by side for any
  uploaded file — this makes Issue #2 (whole-clip vs. streaming disagreement) directly visible
  and explainable in the UI itself, which is a genuine asset if a judge asks a sharp question
  about it, not something hidden.

- `SessionLogger` (`src/voxguard/privacy/session_log.py`) implemented and verified: logs only
  timestamp/event_type/risk_band/probability_synthetic/flagged/category-labels — structurally
  cannot log raw audio, transcript text, or speaker identity (not in the function signature at
  all). `data/logs/` gitignored. 30-day purge implemented and tested. Wired into both Live Mic
  and Upload File handlers, plus a startup purge call in `app.py`.

- **New consolidated tracking file created: `ISSUES.md`** — lists all THREE still-unresolved,
  documented-not-fixed issues (duration confound, chunk-level reliability gap, WavLM Hindi-head
  disagreement) in one place, cross-referenced to `VoxGuard_Remediation_Guide.md` (a full
  phased-prompt remediation plan for all three, written in BuildGuidev4's own format, for use
  only if real time remains after core phases + demo rehearsal).

## Phase 7+ — Not started
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
