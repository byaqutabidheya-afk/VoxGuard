# VoxGuard — FIX Guide (Post-Build Remediation)

**Builds on:** `BuildGuidev4.md` Phases 0–11, all complete.
**Fixes:** All three issues in `ISSUES.md`.
**Estimated total time:** 23–25 hours if every phase is run (F0 2h + F1 4h + F2 8–10h + F3 3h +
F4 1.5h + F5 2.5h + F6 2h). Phase F2 includes one Kaggle GPU session. See Appendix A for a
one-day path that skips F2.
**Phase numbering:** F0–F6, deliberately outside BuildGuidev4's 0–11 range so nothing collides.

---

## Read This First

This guide is not a list of independent patches. The three issues in `ISSUES.md` are causally
linked, and fixing them in the wrong order wastes hours or produces numbers you can't trust.

### The causal chain (this is the core insight of this guide)

```
Issue 1 (duration confound)
   │  Hindi real clips avg 4.95s, synthetic avg 7.44s.
   │  The whole-clip classifier partly learned "long = synthetic".
   │
   ├──► Issue 2 (chunk-level detection fails)
   │      Fixed-size windows are ALL the same length.
   │      The duration signal the classifier leaned on is
   │      structurally ABSENT from every chunk. The model is
   │      asked to classify using a cue that no longer exists,
   │      so it falls back on noise → confident false positives
   │      on real speech, missed detections on synthetic.
   │
   └──► Issue 3 (WavLM head confidently wrong on real speech)
          Plausibly the same confound plus a WavLM-specific factor
          (see F3.1) — the head found an easy shortcut and
          never learned the hard signal.
```

**Consequence:** fixing Issue 1 first is not optional sequencing preference. Retraining anything
on un-matched Hindi data reproduces the same shortcut, and any conclusion you draw about Issues
2 or 3 from that retrain is unreliable.

**Second consequence, and this one is good news:** chunk-level training (F2) *structurally*
eliminates the duration confound for the chunked model family, because every training row is
exactly the same number of samples. The confound cannot be learned. F1 still matters because the
whole-clip model (used by "Upload & Analyze", `/analyze`, and the explainability overlay) keeps
operating on full clips.

### Why WavLM specifically fails (hypothesis to test in F3.1)

WavLM's pretraining includes a speaker-discriminative objective (utterance mixing / denoising,
designed to make embeddings separate speakers). wav2vec2's does not, to the same degree. Your
Hindi training set contains **two speakers** (byaquta, mahato), with soumya held out.

A head trained on WavLM embeddings over two speakers can achieve near-perfect training accuracy
by learning *"is this byaquta or mahato speaking?"* rather than *"is this synthesized?"* — and
that shortcut collapses completely on an unseen speaker. This is consistent with every observed
symptom: perfect CV accuracy (Prompt 4.8's 1.0000 ± 0.0000), confident wrongness on held-out
soumya, and no equivalent problem in Phase 3's English-only evaluation (which had many speakers).

F3.1 tests this directly rather than assuming it.

---

## Scope and Non-Goals

### In scope
- Duration-matching the Hindi/Hinglish corpus and retraining the whole-clip Hindi heads (F1)
- A second, chunk-native classifier family trained on fixed-length windows (F2)
- Diagnosing and correcting the WavLM head / ensemble weighting (F3)
- Recalibrating every threshold invalidated by the above (F4)
- Propagating all of it into the Gradio UI, the REST API, and every script (F5)
- Regression-proving the result and updating documentation (F6)

### Explicitly NOT in scope
- Regenerating the Hindi synthetic audio with a different TTS (ElevenLabs / IndicF5). Out of
  scope, would invalidate the dataset card and consent record.
- Fine-tuning wav2vec2 or WavLM backbones. They stay frozen, as in Phase 2.
- Recording more speakers. The 3-speaker limit and its consequences stay a documented limitation.
- Any change to speaker verification (Phase 8). It works; `verify_speaker`'s 0.7 threshold and
  ECAPA embeddings are untouched by this guide.

### Non-negotiable constraints
1. **Never overwrite an existing model file.** Every artifact this guide produces gets a new name.
   Rollback must always be one config line.
2. **Never delete `models/voiceprints/`.** Enrollment has been wiped twice by test runs already.
3. **Every phase ends at a gate.** If the gate fails, you stop and read the failure section — you
   do not proceed to the next phase hoping it resolves itself.

---

## Naming Conventions (load-bearing — do not improvise)

The existing project already depends on `{model}_{split}.npy` and
`{model}_hindi_combined_logreg.joblib` conventions in many places. These new names extend that
pattern without colliding.

| Artifact | Existing (never overwrite) | New in this guide |
|---|---|---|
| Hindi track metadata | `hindi_hinglish_track.csv` | `hindi_hinglish_track_matched.csv` |
| Hindi audio | `data/raw/hindi_hinglish/{real,synthetic}/` | `data/raw/hindi_hinglish_matched/{real,synthetic}/` |
| Hindi whole-clip embeddings | `{model}_hindi_train.npy` | `{model}_hindi_train_matched.npy` |
| Whole-clip Hindi heads | `{model}_hindi_combined_logreg.joblib` | `{model}_hindi_matched_logreg.joblib` |
| Chunk embeddings | *(none)* | `{model}_{dataset}_{split}_chunked.npy` |
| Chunked heads | *(none)* | `{model}_chunked_logreg.joblib` |
| Reports | `hindi_training_comparison.md` | `fix_comparison_report.md` |

`{model}` is always `wav2vec2` or `wavlm`. Every classifier keeps its `.json` sidecar and
persisted `StandardScaler`, per Phase 2's Fix 2 contract (`load_classifier` returns
`(model, scaler)`).

---

## Single Source of Truth (the mechanism that makes UI propagation work)

**The root cause of "I changed the model but the UI still uses the old one" is that classifier
paths are currently hardcoded in at least six places.** F1–F4 produce new model files; **F5 is
the phase that makes anything actually use them**, by replacing every hardcoded path with one
config block. Do not skip F5 — without it you will have new models on disk, new numbers in your
reports, and a UI still silently running the old ones, which is worse than not fixing anything.

Between F1 and F5 the new models exist but nothing consumes them. That is expected and correct;
evaluation scripts in F1–F4 point at the new files explicitly by argument, while the app and API
keep running the old ones until F5 rewires them deliberately.

Target end state in `src/voxguard/config.py`:

```python
PRODUCTION_WHOLECLIP_CLASSIFIERS = {
    "wav2vec2": "models/classifiers/wav2vec2_hindi_matched_logreg.joblib",
    "wavlm":    "models/classifiers/wavlm_hindi_matched_logreg.joblib",
}
PRODUCTION_STREAMING_CLASSIFIERS = {
    "wav2vec2": "models/classifiers/wav2vec2_chunked_logreg.joblib",
    "wavlm":    "models/classifiers/wavlm_chunked_logreg.joblib",
}
PRODUCTION_ENSEMBLE_WEIGHT_A = 0.5   # set by F3
```

Two separate families, because they are genuinely different models solving genuinely different
input distributions. Whole-clip for uploads and the API; chunked for live streaming.

---

## Rollback Plan (read before starting, not after something breaks)

At any point you can revert to the current, known-behaviour system by pointing the config block
above back at:

```python
PRODUCTION_WHOLECLIP_CLASSIFIERS = {
    "wav2vec2": "models/classifiers/wav2vec2_hindi_combined_logreg.joblib",
    "wavlm":    "models/classifiers/wavlm_hindi_combined_logreg.joblib",
}
PRODUCTION_STREAMING_CLASSIFIERS = PRODUCTION_WHOLECLIP_CLASSIFIERS  # pre-fix behaviour
PRODUCTION_ENSEMBLE_WEIGHT_A = 0.5                                   # pre-fix value
```

**A full rollback is four values, not two.** Restore all of these from the F0.2 capture:
`RISK_THRESHOLDS`, `STREAM_FLAG_THRESHOLD`, `consecutive_flags_required` (StreamingSession's
default), and `PRODUCTION_ENSEMBLE_WEIGHT_A`. Restoring the classifier paths alone leaves you
running old models against thresholds calibrated for new ones, which is a third behaviour that
was never tested.

Commit before starting each phase so `git revert` is always available.

---

## Environment Gotchas (all previously hit in this project — do not rediscover them)

**Local (Windows / PowerShell):**
- Multi-line `python -c "..."` breaks on quotes and apostrophes. Always use the here-string form:
  `@'` … `'@ | Out-File -FilePath tmp.py -Encoding utf8` then `python tmp.py`. Single-quoted
  `@'...'@` (not `@"..."@`) avoids escaping entirely.
- A broken `-c "` leaves PowerShell waiting for a closing quote and silently swallows your next
  paste. If output looks wrong, press Ctrl+C and start clean.
- Pinned versions that must not drift: `pydantic<2.10`, `starlette<1.0`, `fastapi<0.115`.
- FFmpeg shared build must remain on PATH for torchcodec.
- First model load per process is slow and silent: wav2vec2 ~90s, Whisper ~3min, ECAPA ~30s.
  Silence is not a hang.

**Kaggle (F2 only):**
- Repo clones as `VoxGuard` (capital V, capital G). Linux is case-sensitive.
- Dataset mounts at `/kaggle/input/datasets/byaqutabidheyabehera/voxguard-preprocessed-data`
  — note the extra `datasets/<username>/` nesting.
- `pip install -r requirements.txt` downgrades numpy and breaks C extensions. Install
  `"numpy<2.0.0"` in the same command, then `os._exit(00)` to restart the kernel once.
- `os.environ["PYTHONPATH"]` fixes `!python scripts/...` subprocesses only. In-kernel imports
  additionally need `sys.path.insert(0, "/kaggle/working/VoxGuard/src")` **and**
  `importlib.invalidate_caches()` plus clearing `sys.path_importer_cache` for that directory.
- Always re-assert `os.getcwd() == "/kaggle/working/VoxGuard"` at the top of any cell that writes
  files. A stale cwd after a kernel restart silently writes outputs somewhere you'll never find.

---

## Assumptions This Guide Makes About Your Repo

Verify these before starting. Each one is depended on by at least one phase, and a mismatch will
surface as a confusing failure several hours in rather than immediately.

```powershell
cd D:\VoxGuard
.\.venv\Scripts\Activate.ps1
```

```powershell
@'
from pathlib import Path
import importlib

checks = []

# 1. Config placeholders this guide extends
from src.voxguard import config
for name in ["RISK_THRESHOLDS", "STREAM_CHUNK_SECONDS", "STREAM_OVERLAP_SECONDS"]:
    checks.append((f"config.{name}", hasattr(config, name), getattr(config, name, None)))
checks.append(("config.STREAM_FLAG_THRESHOLD", hasattr(config, "STREAM_FLAG_THRESHOLD"),
               getattr(config, "STREAM_FLAG_THRESHOLD", None)))

# 2. Modules this guide imports from
for mod in [
    "src.voxguard.streaming.buffer",
    "src.voxguard.streaming.session",
    "src.voxguard.streaming.scorer",
    "src.voxguard.embeddings.cache",
    "src.voxguard.classifier.head",
    "src.voxguard.classifier.ensemble",
    "src.voxguard.classifier.cross_eval",
    "src.voxguard.utils.hindi_splits",
    "src.voxguard.utils.splits",
    "src.voxguard.explain.attribution",
]:
    try:
        importlib.import_module(mod)
        checks.append((mod, True, "importable"))
    except Exception as e:
        checks.append((mod, False, repr(e)))

# 3. Files this guide reads
for p in [
    "data/metadata/hindi_hinglish_track.csv",
    "data/metadata/unified.csv",
    "models/classifiers/wav2vec2_hindi_combined_logreg.joblib",
    "models/classifiers/wavlm_hindi_combined_logreg.joblib",
    "models/embeddings/wav2vec2_train.npy",
    "models/embeddings/wavlm_train.npy",
    "models/embeddings/wav2vec2_eval.npy",
    "models/embeddings/wavlm_eval.npy",
]:
    checks.append((p, Path(p).exists(), ""))

for name, ok, extra in checks:
    print(("OK   " if ok else "MISS "), name, extra)
'@ | Out-File -FilePath check_fix_assumptions.py -Encoding utf8

python check_fix_assumptions.py
Remove-Item check_fix_assumptions.py
```

**If anything reports MISS:**
- A missing `config` attribute means a name differs in your repo — grep for the actual name and
  substitute it consistently throughout this guide rather than adding a duplicate.
- A failed import usually means a module path differs (e.g. `scorer.py` living elsewhere).
  Correct the path in the relevant prompt before handing it to a coding agent.
- A missing `.npy` means that cache was never built or was cleaned up. F0.2 and F1.4 both assume
  the ASVspoof train/eval caches exist; if they don't, you will need to re-extract them (Kaggle)
  before F1, which adds time not budgeted here.
- A missing `hindi_hinglish_track.csv` blocks F1 entirely — that is the corpus being matched.

Also confirm the enrolled voiceprint survives, since it has been wiped twice:

```powershell
python -c "from src.voxguard.speaker.enrollment import list_enrolled_speakers; print(list_enrolled_speakers())"
```

---

# Phase F0 — Baseline Capture & Diagnostics

**Estimated time:** ~2 hours
**Depends on:** Nothing. Start here.
**Gate:** You have a written baseline you can prove improvement against, and you know whether
ASVspoof has its own duration confound.

## Objective

Measure the current system precisely and immutably before changing anything. Without this you
cannot demonstrate that the fix helped, and you cannot tell a regression from noise. This phase
also runs one diagnostic nobody has run yet, which may change how you interpret every English
number in the project.

## Build Prompts

### Prompt F0.1 — Freeze the current state

```
Create a git tag or branch marking the pre-fix state: `git add -A`, `git commit -m "Pre-FIX
baseline: Phases 0-11 complete"`, `git tag pre-fix-baseline`. Then copy the entire
models/classifiers/ directory to models/classifiers_prefix_backup/ (a plain filesystem copy, not
a move) and add models/classifiers_prefix_backup/ to .gitignore if model files aren't already
gitignored. This is the rollback anchor for the whole guide — every later phase assumes it
exists. Print a confirmation listing both the tag and the backup directory contents.
```

### Prompt F0.2 — Capture the numeric baseline

```
Write scripts/fix_capture_baseline.py that records, in one run, every number this guide will
later claim to have improved, and writes them to models/reports/fix_baseline.json plus a
human-readable models/reports/fix_baseline.md. Capture:

1. Current config values: RISK_THRESHOLDS, STREAM_FLAG_THRESHOLD, STREAM_CHUNK_SECONDS,
   STREAM_OVERLAP_SECONDS, consecutive_flags_required default, and the classifier paths
   currently used by StreamingSession and app/app.py (read them, don't assume).
2. Whole-clip performance of the current production WeightedAverageDetector on the ASVspoof2019
   eval split, from cached embeddings (models/embeddings/wav2vec2_eval.npy and wavlm_eval.npy —
   do NOT re-extract): accuracy, ROC-AUC, EER, and per-class recall.
3. Whole-clip performance on the full Hindi/Hinglish eval split (held-out speaker soumya): same
   metrics.
4. Per-backbone whole-clip predictions on all 25 of soumya's real clips: wav2vec2 probability,
   WavLM probability, ensemble probability, and the absolute margin from 0.5. Record how many of
   the 25 have margin > 0.15 (the Issue 3 metric — currently 7/25).
5. Streaming behaviour on the 5 verified demo pairs from Phase 6 and on the 3 real clips used in
   the 2026-09-12 window sweep: for each, flagged True/False, seconds_to_flag, final running
   score, and max running score reached.
6. The full window-size sweep table from ISSUES.md Issue 2, regenerated live (windows 1.5 / 3.0 /
   4.0 / 6.0 s, stride 1.0s) so the baseline file is self-contained.

Every metric must be labelled with which model file produced it. Print a summary table at the end.
```

### Prompt F0.3 — The ASVspoof duration-confound check (new diagnostic)

```
Write scripts/fix_check_asvspoof_confound.py that applies the exact diagnostic used to discover
Issue 1 in the Hindi data, but to ASVspoof2019 — this has never been run and its result
materially affects how much of the project's English performance can be trusted.

For the ASVspoof2019 train split (using data/metadata/unified.csv's processed_path column and
the official split logic in get_asvspoof_splits):
1. Compute duration and RMS energy for every clip (use soundfile.info for duration where possible
   rather than decoding the full waveform — 25,380 clips, header reads are far faster).
2. Report mean/std duration and RMS separately for bonafide and spoof.
3. Fit a LogisticRegression on ONLY [duration, rms] with 5-fold stratified cross-validation and
   report accuracy — the same 83.3% number that exposed the Hindi confound.
4. Do the same for the eval split (subsample to 10,000 clips if runtime is a concern; note the
   subsampling in the output).
Write results to models/reports/fix_asvspoof_confound.md.

Interpretation guidance to print alongside the numbers: accuracy near 50-60% means no meaningful
duration confound in ASVspoof and the English results stand as reported; accuracy above ~70%
means ASVspoof carries its own confound and the project's English EER figures also partly
reflect a length shortcut, which must be disclosed in ISSUES.md and the final report.
```

## Tests

```powershell
cd D:\VoxGuard
.\.venv\Scripts\Activate.ps1
python scripts\fix_capture_baseline.py
python scripts\fix_check_asvspoof_confound.py
git tag --list | Select-String "pre-fix-baseline"
Get-ChildItem models\classifiers_prefix_backup
```

## Gate F0 — do not proceed until all true

- [ ] `git tag pre-fix-baseline` exists and `models/classifiers_prefix_backup/` contains every
      `.joblib`, `.json`, and scaler file from `models/classifiers/`
- [ ] `models/reports/fix_baseline.json` and `.md` exist and contain all six captured items
- [ ] The Issue 3 margin metric is recorded (expect ~7/25 confident predictions)
- [ ] `models/reports/fix_asvspoof_confound.md` exists, and you have **read** the duration+RMS
      accuracy number and know which interpretation branch you're on
- [ ] `list_enrolled_speakers()` returns `['byaquta']` — check now, re-check after every phase

## Common Pitfalls

- **Regenerating the baseline later.** Once F1 changes anything, the baseline is unreproducible.
  Capture it completely now, or you lose the ability to prove improvement.
- **Trusting a remembered number.** Every figure in the baseline must come from a script run
  today, not from `PROGRESS.md`. Model files may have been touched since those were written.
- **Assuming ASVspoof is clean.** If F0.3 comes back above 70%, stop and tell the whole story
  honestly in F6 — a confound found in your *own* English baseline is a more significant finding
  than the Hindi one, and hiding it would be worse than the confound itself.

---

# Phase F1 — Fix the Duration Confound (Issue 1)

**Estimated time:** ~4 hours
**Depends on:** F0 complete and gated.
**Gate:** Duration+RMS-alone accuracy on the matched Hindi corpus drops to near chance, and the
matched whole-clip classifiers are trained and evaluated.

## Objective

Remove the duration shortcut from the Hindi/Hinglish corpus so the Hindi-adapted classifiers are
forced to learn acoustic synthesis artifacts rather than clip length. Produce a second,
duration-matched corpus and a second family of whole-clip Hindi heads, leaving the originals
untouched for comparison.

## Design Decisions (fixed — do not improvise)

- **Trim, never pad.** Padding the shorter clip with silence introduces a trivially detectable
  new artifact (a run of near-zero samples) and simply swaps one confound for another.
- **Trim from both ends equally.** Trimming only the tail could systematically remove
  synthesis-specific trailing artifacts, which is itself a form of label leakage in reverse.
- **Match within pairs, not globally.** Each real clip is matched against *its own* synthetic
  clone (same speaker, same sentence_id), so the matched corpus preserves the deliberate
  matched-pair design from Phase 4.
- **Enforce a floor.** If matching would take either clip below 1.5 seconds (one full streaming
  window), skip the pair and log it rather than producing a degenerate sample.

## Build Prompts

### Prompt F1.1 — Duration-matching utility

```
Implement src/voxguard/utils/duration_match.py with:

- duration_match_pair(real_waveform, synth_waveform, sr, min_seconds=1.5) ->
  (matched_real, matched_synth, info_dict): trims the longer of the two waveforms down to the
  shorter one's sample count, removing an equal number of samples from the START and the END
  (if the difference is odd, put the extra sample at the end). Never pads. Returns the two
  matched arrays plus an info dict recording original durations, matched duration, seconds
  trimmed, and which clip was trimmed.
  Raise a clear ValueError if either input is empty, or if the resulting matched duration would
  be below min_seconds — callers are expected to catch this and skip the pair.

- Write tests/test_duration_match.py verifying: a 5s and an 8s array both come back at exactly 5s
  with 1.5s removed from each end of the longer one; two equal-length arrays come back unchanged
  with zero trimmed; a pair that would match below min_seconds raises ValueError; and that the
  function never returns an array longer than either input (i.e. proves it never pads).
```

### Prompt F1.2 — Rebuild the corpus

```
Write scripts/fix_rebuild_matched_hindi.py that:
1. Loads data/metadata/hindi_hinglish_track.csv and groups rows into (speaker_id, sentence_id)
   pairs, each of which should contain exactly one real and one synthetic row. Log and skip any
   group that doesn't (there should be none — the corpus is 75 matched pairs).
2. For each pair, loads both clips at 16kHz mono via audio_io.load_audio, applies
   duration_match_pair, and writes both matched clips to
   data/raw/hindi_hinglish_matched/{real,synthetic}/ preserving the original filenames.
3. Writes data/metadata/hindi_hinglish_track_matched.csv with the same schema as the original
   track CSV (filepath, label, speaker_id, category, sentence_id, dataset) pointing at the new
   matched files. filepath must be REPO-RELATIVE, matching the convention established in Phase 1
   Prompt 1.4 — absolute paths break portability.
4. Prints a before/after table: mean and std duration for real vs synthetic, BEFORE matching and
   AFTER matching, plus total seconds trimmed and a list of any skipped pairs with the reason.

Do NOT modify or delete anything under data/raw/hindi_hinglish/ — the original corpus stays
intact for comparison and because the dataset card documents it.
```

### Prompt F1.3 — Prove the confound is gone

```
Write scripts/fix_verify_matched_confound.py that runs the identical diagnostic that exposed
Issue 1, against data/metadata/hindi_hinglish_track_matched.csv:
1. Compute duration and RMS energy per clip.
2. Fit LogisticRegression on ONLY [duration, rms], 5-fold stratified CV, report mean accuracy
   and per-fold scores.
3. Also report the same for the ORIGINAL corpus in the same run, side by side, so the
   before/after is visible in one output (expected: original ~83.3%, matched near chance).
4. Assert and exit non-zero if matched accuracy is still above 0.65 — if the matching didn't
   work, retraining on this data is wasted effort and the run should fail loudly rather than
   proceed silently.
Additionally report RMS-only accuracy separately from duration-only accuracy, so that if residual
signal remains you know which of the two is responsible (matching fixes duration; it does not
by itself fix an energy/loudness difference, which would need separate peak normalization).
```

### Prompt F1.4 — Re-extract and retrain (whole-clip, local CPU)

```
Write scripts/fix_retrain_matched_wholeclip.py that, entirely on local CPU (this is ~150 clips —
Phase 4 Prompt 4.7 established Kaggle is not worth the round trip at this size):

1. Applies the SAME speaker-holdout split used in Phase 4 (speaker_holdout mode,
   holdout_speaker='soumya') to hindi_hinglish_track_matched.csv, via the existing
   get_hindi_hinglish_splits — do not reimplement the split logic, and do not change the holdout
   speaker, or the new numbers won't be comparable to the baseline.
2. Extracts whole-clip embeddings for the matched Hindi train split for BOTH backbones using the
   existing extract_and_cache (which already contains the Phase 2 length-sorted batching fix),
   saving to models/embeddings/wav2vec2_hindi_train_matched.npy and
   models/embeddings/wavlm_hindi_train_matched.npy. Also extract the matched Hindi EVAL split to
   {model}_hindi_eval_matched.npy — F1.5 (the very next prompt), F3's diagnosis, and F6's final
   report all need it, and it has never been cached before.
   If the ORIGINAL (unmatched) Hindi eval embeddings were also never cached, extract those too as
   {model}_hindi_eval.npy — F1.5 requires both splits to make a valid comparison against the F0
   baseline.
3. For each backbone independently (the production architecture is the weighted-average ensemble,
   so there is no shared feature space — this mirrors Phase 4 Prompt 4.8's structure exactly):
   builds the combined training set by row-wise concatenating the cached ASVspoof2019 train
   embeddings ({model}_train.npy, unchanged) with the new matched Hindi train embeddings, then
   trains a logistic-regression head via train_logistic_regression, fitting and persisting a
   StandardScaler per Phase 2's Fix 2 contract.
4. Saves to models/classifiers/{model}_hindi_matched_logreg.joblib with metadata sidecars.
   NEVER overwrite {model}_hindi_combined_logreg.joblib.
5. Prints 5-fold stratified CV accuracy per backbone, and explicitly flags any CV score above
   0.99 as suspicious rather than reporting it as success — a perfect score on this dataset is
   the exact signature that started this whole investigation.

Note: no prosody features anywhere. Phase 2 selected the baseline (non-prosody) variant.
```

### Prompt F1.5 — Evaluate matched vs original

```
Write scripts/fix_evaluate_matched.py producing models/reports/fix_matched_comparison.md, a
markdown table with rows = model variant and columns = [ASVspoof2019 eval acc/EER, Hindi matched
eval acc/EER], covering:
  (a) wav2vec2 hindi_combined (original, baseline)
  (b) wavlm hindi_combined (original, baseline)
  (c) weighted-average of (a)+(b) — the current production detector
  (d) wav2vec2 hindi_matched (new)
  (e) wavlm hindi_matched (new)
  (f) weighted-average of (d)+(e) — the candidate production detector

Evaluate all six on THREE test sets, and label every column with which one it used:
  - ASVspoof2019 eval (confirms no English regression)
  - the ORIGINAL Hindi eval split (soumya, unmatched) — needed for continuity with the F0
    baseline, which was captured before the matched corpus existed
  - the MATCHED Hindi eval split (soumya, duration-matched) — the honest test set going forward

CRITICAL: the F0 baseline's Hindi numbers were measured on the ORIGINAL eval split. Comparing a
post-fix number measured on the MATCHED split against that baseline is comparing two different
test sets and is meaningless. Evaluating all six variants on both Hindi splits is what makes the
comparison valid: variant (c) on the original split reproduces the baseline, variant (c) on the
matched split isolates the test-set change, and variant (f) on the matched split is the honest
post-fix number. State this explicitly in the report so nobody later quotes a cross-split delta.

Reuse zero_shot_eval_from_cache / zero_shot_eval_weighted_average_from_cache from Phase 3
Prompt 3.1 rather than writing new evaluation code.

Print an explicit interpretation block stating: a LOWER Hindi accuracy for (f) than for (c) is
the EXPECTED and CORRECT outcome, because the original number was partly produced by the duration
shortcut. The question this table answers is not "did accuracy go up" but "what is the honest
Hindi performance once the shortcut is removed, and did English performance hold steady".
```

## Tests

```powershell
cd D:\VoxGuard
.\.venv\Scripts\Activate.ps1
python -m pytest tests/test_duration_match.py -v
python scripts\fix_rebuild_matched_hindi.py
python scripts\fix_verify_matched_confound.py
python scripts\fix_retrain_matched_wholeclip.py
python scripts\fix_evaluate_matched.py
python -m pytest tests/ -q

# Gate check: originals untouched (compares hashes against the F0 backup)
Get-ChildItem models\classifiers_prefix_backup -Filter *.joblib | ForEach-Object {
    $orig = Get-FileHash $_.FullName
    $live = Get-FileHash "models\classifiers\$($_.Name)" -ErrorAction SilentlyContinue
    "{0}: {1}" -f $_.Name, $(if ($live -and $live.Hash -eq $orig.Hash) {"UNCHANGED"} else {"CHANGED OR MISSING"})
}
```

## Gate F1 — do not proceed until all true

- [ ] Matched corpus exists: 75 real + 75 synthetic under `data/raw/hindi_hinglish_matched/`
      (or fewer, with every skipped pair logged and explained)
- [ ] Real and synthetic mean durations are now within ~0.1s of each other
- [ ] Duration+RMS-alone accuracy on the matched corpus is **below 0.65** (script exits non-zero
      otherwise)
- [ ] `{model}_hindi_matched_logreg.joblib` exists for both backbones, with sidecars and scalers
- [ ] `{model}_hindi_combined_logreg.joblib` still exists, byte-identical to the F0 backup
- [ ] `fix_matched_comparison.md` generated, and ASVspoof2019 English accuracy for variant (f)
      is within ~2 percentage points of variant (c) — a larger English drop means something
      broke, not that the confound was removed
- [ ] `list_enrolled_speakers()` still returns `['byaquta']`

## Common Pitfalls

- **Padding instead of trimming.** Silence padding is trivially detectable and replaces one
  confound with a worse one.
- **Skipping F1.3 and going straight to retraining.** If the matching has an off-by-one or a
  path bug, you'll spend an hour training on data that still has the confound and won't know.
- **Expecting accuracy to improve.** It should go *down*. A lower, honest number is the deliverable.
  If Hindi accuracy stays at 96–100% after matching, be suspicious: check that the script is
  actually reading the matched CSV and not silently falling back to the original.
- **Changing the holdout speaker.** Keep soumya. Changing it silently invalidates every
  comparison against the F0 baseline.
- **Forgetting relative paths.** Phase 1's `processed_path` convention is repo-relative; absolute
  paths here will break the Kaggle symlink approach in F2.

---

# Phase F2 — Chunk-Native Classifier Family (Issue 2)

**Estimated time:** ~8–10 hours (including one Kaggle GPU session of roughly 2–4 hours)
**Depends on:** F1 complete and gated. The matched Hindi corpus must exist before the Kaggle upload.
**Gate:** Chunk-trained heads separate real from synthetic *at chunk level* on held-out data,
and the previously-failing streaming cases now behave correctly.

## Objective

Train a classifier family whose training input distribution matches what `StreamingBuffer`
actually delivers at inference time: fixed-length 1.5-second windows. This is the only real fix
for Issue 2 — the F0/2026-09-12 window sweep proved no window size makes the whole-clip model
work, because the mismatch is in *what the model learned*, not in window geometry.

## The leakage trap (read this twice)

**Split at CLIP level first, then chunk within each split.**

If you chunk the whole corpus and then split randomly, chunks from the same clip land in both
train and eval. The model memorizes clip-specific acoustics and reports a spectacular, entirely
fake EER. This is the single most likely way this phase produces a wrong-but-convincing result.

- ASVspoof2019: use the official train/dev/eval partition (`get_asvspoof_splits`) at clip level,
  then chunk each partition separately.
- Hindi: use `get_hindi_hinglish_splits(mode='speaker_holdout', holdout_speaker='soumya')` at
  clip level, then chunk. Soumya's chunks must never appear in training.

## Scale and GPU budget

Chunking multiplies row count. Rough arithmetic at 1.5s window / 1.0s stride: a clip of duration
`d` yields `floor((d - 1.5) / 1.0) + 1` windows, so a 3.5s clip yields 3.

To keep the Kaggle session inside one sitting, F2.2 subsamples. Full train is kept (the model
needs it); dev and eval are stratified-subsampled, which is statistically ample for EER estimation.

| Split | Clips used | Rationale |
|---|---|---|
| ASVspoof train | all 25,380 | needed for learning |
| ASVspoof dev | stratified 8,000 | threshold calibration only |
| ASVspoof eval | stratified 12,000 | EER estimation only |
| Hindi train (matched) | all | tiny |
| Hindi eval (matched) | all | tiny |

Run the calibration cell (F2.3 step 6) before committing to the full run, exactly as Phase 2 did.

## Build Prompts

### Prompt F2.1 — Chunking utility that cannot drift from inference

```
Implement src/voxguard/utils/chunk_audio.py with
chunk_waveform(waveform, sr, chunk_seconds=None, overlap_seconds=None,
               drop_silent=True, silence_threshold=None) -> list[np.ndarray]

CRITICAL: this function must NOT contain its own windowing arithmetic. It must instantiate a
StreamingBuffer (Phase 5, Prompt 5.1) with the given parameters, push the entire waveform through
its push() method in one call, and collect the emitted windows. Any second implementation of
stride/overlap maths creates a train/inference mismatch and silently reintroduces exactly the
class of bug this phase exists to fix. Defaults come from config.STREAM_CHUNK_SECONDS and
config.STREAM_OVERLAP_SECONDS, so training windows track inference windows automatically.

drop_silent: when True (the default), apply the SAME RMS silence gate used by StreamingScorer
(Phase 5, Prompt 5.2 as amended) and discard windows below threshold. Rationale to put in the
docstring: at inference time silent chunks are never scored, so training on them teaches the
model to classify audio it will never be asked about, and worse, assigns them a synthetic/real
label they don't deserve. Import the threshold constant from wherever StreamingScorer defines it
rather than redefining the number.

Trailing partial windows shorter than chunk_seconds are dropped, not zero-padded (StreamingBuffer
already behaves this way; do not add padding here).

Write tests/test_chunk_audio.py asserting: chunk count matches the closed-form
floor((d - chunk)/stride)+1 for several durations; every returned chunk has exactly
chunk_seconds*sr samples; an all-silence input returns an empty list when drop_silent=True;
and a synthetic waveform chunked by this function is element-wise identical to what a manually
driven StreamingBuffer produces for the same input (this is the anti-drift test — do not omit it).
```

### Prompt F2.2 — Chunk-level embedding extraction CLI

```
Write scripts/extract_chunked_embeddings.py as a CLI (--dataset {asvspoof2019,hindi},
--split {train,dev,eval}, --model {wav2vec2,wavlm}, --chunk_seconds, --overlap_seconds,
--max_clips, --output_dir) that:

1. Loads the clip-level split FIRST (get_asvspoof_splits for asvspoof2019;
   get_hindi_hinglish_splits with mode='speaker_holdout', holdout_speaker='soumya' reading
   hindi_hinglish_track_matched.csv for hindi). Apply --max_clips as a STRATIFIED subsample at
   clip level (stratify on label; for asvspoof also stratify on the attack/system id column if
   present) — never a plain head() or random slice.
2. Chunks each clip via chunk_waveform (Prompt F2.1). Every chunk inherits its parent clip's
   label. Document in the module docstring that this label inheritance is a deliberate
   simplification and a known source of label noise (a low-information chunk from inside a
   synthetic clip is still labelled synthetic).
3. Extracts an embedding per chunk with the existing EmbeddingExtractor, batched through the
   existing extract_and_cache batching path so the Phase 2 length-sorted batching fix still
   applies (all chunks are equal length here, so batching is trivially safe — but reuse the same
   code path rather than writing a new loop).
4. Saves models/embeddings/{model}_{dataset}_{split}_chunked.npy plus the parallel .csv manifest,
   extending the standard manifest columns with parent_filepath, chunk_index, and chunk_start_seconds
   so any chunk can be traced back to its source clip and time offset. The parent_filepath column
   is REQUIRED — F2.4's clip-level aggregation depends on it.
5. Logs total clips processed, total chunks produced, chunks dropped as silent, and the
   chunks-per-clip distribution (min/median/max).
6. Is resumable in the same skip-if-exists / force=True manner as extract_and_cache.

Run ASVspoof2019 extractions on Kaggle GPU (see the workflow in this phase's notes); run the
hindi extractions locally on CPU — they are ~150 clips and the Kaggle round trip is not worth it.
```

### Prompt F2.3 — The Kaggle session (operational, run by hand)

This is a checklist, not an agent prompt. Run the cells in order.

**Before opening Kaggle:** commit and push everything from F1 and F2.1/F2.2 to GitHub — the
notebook clones the repo, so anything uncommitted will not be there.

**You do NOT need to upload the matched Hindi corpus to Kaggle.** Hindi chunk extraction runs
locally on CPU (~150 clips); only the ASVspoof2019 extractions need the GPU. The existing
`voxguard-preprocessed-data` dataset already has everything the Kaggle session touches.

```python
# Cell 1 — clean clone (one cell: clone AND cd, so re-running can't nest)
%cd /kaggle/working
!rm -rf VoxGuard
!git clone https://github.com/byaqutabidheya-afk/VoxGuard.git
%cd VoxGuard
!pwd && ls scripts/extract_chunked_embeddings.py
```

```python
# Cell 2 — install with the numpy pin in the SAME command, then restart once
!pip install -r requirements.txt "numpy<2.0.0" --quiet
import os
os._exit(00)
```

```python
# Cell 3 — after restart: paths, cwd, import cache
import os, sys, shutil, importlib
os.chdir('/kaggle/working/VoxGuard')
sys.path.insert(0, '/kaggle/working/VoxGuard')
sys.path.insert(0, '/kaggle/working/VoxGuard/src')
os.environ["PYTHONPATH"] = "/kaggle/working/VoxGuard/src"
if '/kaggle/working/VoxGuard/src' in sys.path_importer_cache:
    del sys.path_importer_cache['/kaggle/working/VoxGuard/src']
importlib.invalidate_caches()
assert os.getcwd() == "/kaggle/working/VoxGuard"
print("cwd OK:", os.getcwd())
```

```python
# Cell 4 — verify the dataset mount path, then symlink
from pathlib import Path
root = Path("/kaggle/input/datasets/byaqutabidheyabehera/voxguard-preprocessed-data")
print("exists:", root.exists(), "| contents:", list(root.iterdir()) if root.exists() else "—")
# if False, walk /kaggle/input one level at a time with iterdir() (NOT rglob — 134k files)
```

```python
# Cell 5 — reconstruct local layout
Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("data/metadata").mkdir(parents=True, exist_ok=True)
shutil.copy2(root / "metadata" / "unified.csv", "data/metadata/unified.csv")
for ds in ["asvspoof2019", "wavefake", "in_the_wild"]:
    dest = Path(f"data/processed/{ds}")
    if dest.is_symlink() or dest.exists():
        dest.unlink() if dest.is_symlink() or dest.is_file() else shutil.rmtree(dest)
    dest.symlink_to(root / ds, target_is_directory=True)
print(os.listdir("data/processed"))
```

```python
# Cell 6 — CALIBRATION. Do not skip. Time a small run before committing hours of GPU.
import time, torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
t0 = time.time()
!python scripts/extract_chunked_embeddings.py --dataset asvspoof2019 --split dev --model wav2vec2 --max_clips 200
print(f"200 clips took {time.time()-t0:.0f}s")
# Extrapolate: (25380 + 8000 + 12000) clips x 2 backbones. If the projection exceeds ~4 hours,
# lower --max_clips for dev/eval before the real run rather than discovering it at hour three.
```

```python
# Cell 7 — the real runs, one cell each so a timeout tells you exactly where to resume
!python scripts/extract_chunked_embeddings.py --dataset asvspoof2019 --split train --model wav2vec2
```
```python
!python scripts/extract_chunked_embeddings.py --dataset asvspoof2019 --split dev   --model wav2vec2 --max_clips 8000
```
```python
!python scripts/extract_chunked_embeddings.py --dataset asvspoof2019 --split eval  --model wav2vec2 --max_clips 12000
```
```python
!python scripts/extract_chunked_embeddings.py --dataset asvspoof2019 --split train --model wavlm
```
```python
!python scripts/extract_chunked_embeddings.py --dataset asvspoof2019 --split dev   --model wavlm --max_clips 8000
```
```python
!python scripts/extract_chunked_embeddings.py --dataset asvspoof2019 --split eval  --model wavlm --max_clips 12000
```

```python
# Cell 8 — verify shapes before saving, while you can still re-run cheaply
import numpy as np, os
for f in sorted(os.listdir("models/embeddings")):
    if f.endswith("_chunked.npy"):
        a = np.load(f"models/embeddings/{f}")
        print(f, a.shape)
        assert a.shape[1] == 768, f"{f} wrong embedding dim"
```

Then **Save Version (Quick Save — not "Run All")**, and download locally into a throwaway
staging folder (`kaggle kernels output` returns the whole working tree, not just the folder you
point it at):

```powershell
cd D:\VoxGuard
kaggle kernels output byaqutabidheyabehera/<notebook-slug> -p kaggle_fix_staging
Move-Item "kaggle_fix_staging\VoxGuard\models\embeddings\*_chunked.*" "models\embeddings\" -Force
Remove-Item "kaggle_fix_staging" -Recurse -Force
```

Finally, run the two Hindi extractions locally:

```powershell
python scripts\extract_chunked_embeddings.py --dataset hindi --split train --model wav2vec2
python scripts\extract_chunked_embeddings.py --dataset hindi --split train --model wavlm
python scripts\extract_chunked_embeddings.py --dataset hindi --split eval  --model wav2vec2
python scripts\extract_chunked_embeddings.py --dataset hindi --split eval  --model wavlm
```

### Prompt F2.4 — Train and evaluate the chunked heads

```
Write scripts/train_chunked_classifier.py that, per backbone independently (wav2vec2, wavlm —
matching the weighted-average ensemble architecture, no shared feature space):

1. Loads {model}_asvspoof2019_train_chunked.npy and {model}_hindi_train_chunked.npy and
   row-wise concatenates them (axis=0) into one training matrix, exactly as Phase 4 Prompt 4.8
   did for whole-clip features. This is a row-wise combine, NOT the feature-axis concatenation
   used for the concatenated ensemble — do not confuse the two.
2. Trains via train_logistic_regression with class_weight="balanced", fitting and persisting a
   StandardScaler on the chunk feature space (the chunk distribution differs from the whole-clip
   distribution, so the whole-clip scaler must NOT be reused).
3. Saves models/classifiers/{model}_chunked_logreg.joblib with its metadata sidecar
   (type, input_dim=768, scaler_path).
4. Reports 5-fold GROUP-aware cross-validation using parent_filepath as the group key
   (sklearn StratifiedGroupKFold), so chunks from one clip never straddle a CV fold. Plain
   StratifiedKFold here would leak and inflate the score — using the group-aware variant is
   mandatory, not a refinement.

Then write scripts/evaluate_chunked.py producing models/reports/fix_chunked_eval.md with:
  (a) CHUNK-LEVEL metrics on the held-out chunked eval sets (ASVspoof eval, Hindi matched eval):
      accuracy, ROC-AUC, EER, per-class recall, for wav2vec2 alone, wavlm alone, and the
      weighted average.
  (b) CLIP-LEVEL metrics computed by aggregating each clip's chunk scores back to one score per
      clip (report BOTH mean-aggregation and max-aggregation), so these numbers are directly
      comparable to the whole-clip baseline captured in F0.2. State plainly in the report which
      aggregation is used where.
  (c) A per-clip table for the 5 verified demo pairs and the 3 real clips from the window sweep,
      showing chunk-score mean / max / min, so the specific previously-failing cases are visible.
```

### Prompt F2.5 — Re-run the failing cases through the real streaming path

```
Write scripts/fix_verify_streaming.py that constructs a StreamingSession explicitly wired to the
NEW chunked classifiers and replays, through the real simulate_stream path (not a bespoke loop),
every clip that previously misbehaved:
  - soumya_neutral_01_clone.wav (synthetic; previously NEVER flagged)
  - byaquta_neutral_01.wav (real; previously flagged — false positive)
  - soumya_control_21.wav (real; previously flagged — false positive)
  - soumya_scam_11.wav (real; previously flagged at 4.0s window in the sweep)
  - plus all 5 verified Phase 6 demo pairs, real and synthetic
For each, report flagged True/False, seconds_to_flag, final score, and max score.
Print a PASS/FAIL summary line per clip against the expected outcome (synthetic → flagged,
real → not flagged) and an overall count. Exit non-zero if any REAL clip flags — a false
positive on genuine human speech is the failure mode this whole phase exists to eliminate, and
it should fail the script loudly rather than appear as one row in a table.

Note that flag_threshold and consecutive_flags_required are still at their pre-fix values at this
point; F4 recalibrates them against the new score distribution. If results are close but not
clean here, do not hand-tune the threshold in this script — record it and let F4 do it properly.
```

## Tests

```powershell
python -m pytest tests/test_chunk_audio.py -v
python scripts\train_chunked_classifier.py
python scripts\evaluate_chunked.py
python scripts\fix_verify_streaming.py
python -m pytest tests/ -q
```

## Gate F2 — do not proceed until all true

- [ ] `chunk_waveform` provably matches `StreamingBuffer` output (the anti-drift test passes)
- [ ] Chunked embedding caches exist for both backbones across ASVspoof train/dev/eval and Hindi
      train/eval, all with `shape[1] == 768`
- [ ] Every chunk manifest has a populated `parent_filepath` column
- [ ] Group-aware CV was used for the chunked heads (confirm `StratifiedGroupKFold` in the code,
      not plain `StratifiedKFold`)
- [ ] **HARD GATE:** chunk-level EER on held-out ASVspoof eval is **below 0.25** — if it is near
      0.5, the model has learned nothing at chunk level and F2 has failed; see the failure branch
      below. This is the one criterion that must pass before F3.
- [ ] **HARD GATE:** chunk-level EER on the held-out Hindi matched eval split is below 0.40
      (looser than the English bar: one held-out speaker, small sample, and a genuinely harder
      language setting)
- [ ] Original whole-clip classifiers untouched
- [ ] `fix_verify_streaming.py` has been RUN and its output recorded — **but its pass/fail is not
      a gate at this point.** It runs against pre-fix `STREAM_FLAG_THRESHOLD` and
      `consecutive_flags_required`, which F4 has not yet recalibrated for the new score
      distribution, so a false positive here may be a threshold artifact rather than a model
      failure. Record the numbers, do not hand-tune, and re-run it as a real gate in F4.
- [ ] Note whether `soumya_neutral_01_clone.wav` now flags. If it still does not even at a
      permissive threshold, that is a signal the chunked model has the same blind spot and is
      worth raising before F4 rather than after.

### If Gate F2 fails

If chunk-level EER sits near 0.5, the honest conclusion is that 1.5 seconds does not carry enough
signal for frozen mean-pooled embeddings to separate these classes, and no amount of retraining
at that window size will fix it. Two branches:

1. **Retrain at a longer window** — set `STREAM_CHUNK_SECONDS` to 3.0s (stride 1.0s), re-run
   F2.2–F2.4. Detection latency rises to ~3s, which is still a legitimate real-time claim, and
   the training distribution is closer to ASVspoof's native clip lengths. Budget another Kaggle
   session.
2. **Accept and document** — keep streaming as an architecture demonstration, ship whole-clip
   detection as the product, and update `ISSUES.md` with the chunk-level EER you measured. This
   is a real result, not a failure to report: "we retrained natively at chunk level and the
   signal still isn't there at 1.5s" is a much stronger statement than the current
   "we didn't have time to try."

Choose branch 1 only if you have a full day. Otherwise take branch 2 and move to F3.

## Common Pitfalls

- **Chunking before splitting.** The single most dangerous mistake in this phase. Split at clip
  level, then chunk.
- **Plain StratifiedKFold on chunk data.** Same leakage, quieter. Use `StratifiedGroupKFold` with
  `parent_filepath` as groups.
- **Reusing the whole-clip StandardScaler.** Chunk embeddings have a different distribution. Fit
  a fresh scaler.
- **Reimplementing the window maths.** Call `StreamingBuffer`. Always.
- **Training on silent chunks.** They carry a label they don't deserve and are never scored at
  inference. Gate them out at training time with the same threshold used at inference.
- **Skipping the Kaggle calibration cell.** Discovering the run needs six hours at hour three is
  avoidable.
- **Downloading straight into `models/embeddings/`.** `kaggle kernels output` dumps the entire
  working tree. Stage, move, delete.

---

# Phase F3 — WavLM Head & Ensemble Weighting (Issue 3)

**Estimated time:** ~3 hours
**Depends on:** F1 and F2 complete and gated (F3 must be evaluated against the new models, not
the old ones).
**Gate:** The confident-prediction margin metric improves substantially over the F0 baseline of
7/25, for both the whole-clip and chunked families.

## Objective

Determine *why* WavLM's Hindi-adapted head confidently misclassifies real speech from an unseen
speaker, then fix it at the right layer — retraining the head if the cause is a training defect,
or reweighting the ensemble if the cause is an inherent backbone mismatch. Diagnose before
prescribing.

## Build Prompts

### Prompt F3.1 — Diagnose (do not fix anything in this prompt)

```
Write scripts/fix_diagnose_wavlm.py that tests the leading hypothesis and two alternatives, and
writes models/reports/fix_wavlm_diagnosis.md. Change no model files in this script.

HYPOTHESIS A — speaker shortcut. WavLM's pretraining includes speaker-discriminative objectives,
so its embeddings separate speakers strongly. With only TWO speakers in the Hindi training set
(byaquta, mahato), the head may have learned speaker identity rather than real-vs-synthetic,
which would collapse on held-out soumya. Test:
  1. Train a throwaway logistic regression on the Hindi train embeddings to predict SPEAKER
     (byaquta vs mahato), separately for each backbone. If WavLM predicts speaker with markedly
     higher accuracy than wav2vec2, its embedding space is more speaker-dominated — supporting A.
  2. Train a throwaway head on Hindi train to predict LABEL, then report its accuracy separately
     on (i) held-out chunks/clips from the SAME speakers and (ii) soumya. A large gap between
     (i) and (ii) for WavLM but not wav2vec2 is direct evidence of the speaker shortcut.

HYPOTHESIS B — the Hindi rows are drowned out. ASVspoof contributes ~25,380 training rows to the
Hindi-combined head's ~25,480. Test by reporting the norm of the head's coefficient vector and
its decision values on Hindi-only vs ASVspoof-only validation rows: if the head is effectively
an English-only model that ignores the Hindi rows, Hindi-specific decision values will cluster
tightly near the boundary.

HYPOTHESIS C — regularization. Sweep C in [0.001, 0.01, 0.1, 1, 10] for the WavLM Hindi head and
report held-out-speaker real-clip accuracy at each. A strong improvement at low C indicates the
current head is overfitting the small Hindi portion.

Also reproduce, against BOTH the F1 matched whole-clip heads and the F2 chunked heads, the
original Issue 3 metric: per-backbone probabilities on all 25 of soumya's real clips, and the
count with ensemble margin > 0.15 from 0.5. Report all four numbers (old whole-clip, new matched
whole-clip, chunked, and per-backbone breakdowns) in one table so it is immediately visible
whether F1/F2 already resolved Issue 3 as a side effect.

Conclude the report with an explicit recommendation: FIX-BY-RETRAIN (with the specific
hyperparameter change) or FIX-BY-REWEIGHT, and the evidence for it.
```

### Prompt F3.2a — Retrain branch (only if F3.1 recommends it)

```
Write scripts/fix_retrain_wavlm_v2.py that retrains the WavLM heads (both the matched whole-clip
head and the chunked head) applying the specific remedy F3.1 identified — most likely one or more
of: a lower C (stronger regularization), oversampling the Hindi training rows by a factor of 5-20
so they are not drowned out by ASVspoof, or both. Save as wavlm_hindi_matched_v2_logreg.joblib
and wavlm_chunked_v2_logreg.joblib; do NOT overwrite v1.

If F3.1's evidence indicates wav2vec2's heads share the same defect (unlikely given the observed
asymmetry, but check the diagnosis rather than assuming), apply the same remedy to them and save
the corresponding wav2vec2_*_v2 files. Otherwise leave wav2vec2's heads untouched and say so in
the output, so it is unambiguous which files changed.

Re-run the Issue 3 metric (25 soumya real clips, count with margin > 0.15) and the 5-clip
backbone-disagreement check (soumya_control_21, soumya_control_22, soumya_control_25,
soumya_neutral_06, soumya_scam_15) against the v2 heads. Report before/after side by side.

Then re-run the FULL evaluation from F1.5 and F2.4 with the v2 heads substituted, to confirm the
fix did not cost English or synthetic-detection performance — a WavLM head that never says
"synthetic" would score perfectly on this metric while being useless. Report synthetic-class
recall explicitly alongside the real-class improvement.
```

### Prompt F3.2b — Reweight branch (only if F3.1 recommends it, or if F3.2a underdelivers)

```
Write scripts/fix_tune_ensemble_weight.py sweeping weight_a (the wav2vec2 share) over
[0.5, 0.6, 0.7, 0.8, 0.9, 1.0] for BOTH model families (matched whole-clip, chunked), evaluated
on the FULL held-out sets — not the five example clips that first surfaced the issue, which would
overfit the fix to its own symptom.

For each weight report: overall EER, overall accuracy, real-class false-positive rate (fraction
of genuine real clips scoring above the decision threshold), and synthetic-class recall.

Select on the real-class false-positive rate subject to synthetic recall staying within 3
percentage points of its best value — NOT on overall accuracy, which Phase 5 already demonstrated
can look healthy while confidence is meaningless. Print the full table, print the recommendation,
and require typed confirmation before writing PRODUCTION_ENSEMBLE_WEIGHT_A into config.py
(mirroring the confirm-before-write pattern from Phase 7's calibrate_thresholds.py).

Note for the implementer: this project's calibration scripts have twice crashed on Windows cp1252
consoles when printing box-drawing or arrow characters. Use ASCII only in table output.
```

## Tests

```powershell
python scripts\fix_diagnose_wavlm.py
# then exactly one of:
python scripts\fix_retrain_wavlm_v2.py
python scripts\fix_tune_ensemble_weight.py
python -m pytest tests/ -q
```

## Gate F3 — do not proceed until all true

- [ ] `fix_wavlm_diagnosis.md` exists and states which hypothesis the evidence supports
- [ ] The Issue 3 margin metric is reported for all model families (old, matched, chunked)
- [ ] Confident-prediction count on soumya's 25 real clips is **at least 18/25** (up from 7/25)
      for whichever family is being promoted to production — see the failure branch if not
- [ ] Synthetic-class recall did not drop more than 3 percentage points versus F0 baseline
- [ ] A weight or a v2 head is selected, with the decision written down and justified

### If Gate F3's 18/25 target is not met

18/25 is a target, not a law of nature — it was chosen as "a clear majority of held-out real
clips get a confident, correct verdict", against a baseline of 7/25. If the best achievable is,
say, 13/25:

1. **Check whether it improved at all.** Anything above ~12/25 is a real, reportable improvement
   over 7/25 even if it falls short of the target. Ship it and state the actual number.
2. **Check whether the remaining near-boundary clips are the SAME ones** across every fix
   attempted. A stable subset of hard clips is a property of the data (one speaker, one
   recording setup), not a bug in the head — say so rather than continuing to tune.
3. **Do not chase the target by weighting WavLM to zero.** If `weight_a = 1.0` is the only thing
   that hits 18/25, you have not fixed the ensemble, you have deleted half of it. That may still
   be the right call, but report it as "we dropped WavLM from the Hindi path because its head
   does not generalize to unseen speakers", which is an honest architectural decision, not a
   tuning result.
4. Record the outcome in `ISSUES.md` as partially resolved with the measured numbers. Proceed to
   F4 either way — thresholds still need recalibrating for whatever configuration you ship.

## Common Pitfalls

- **Fixing before diagnosing.** Reweighting away from WavLM hides a training defect instead of
  correcting it, and you lose a genuinely useful backbone.
- **Evaluating only on the five clips that exposed the bug.** That is fitting to the symptom.
  Use the full 25-clip set and the full eval splits.
- **Optimizing overall accuracy.** The failure mode is confident wrongness on real speech;
  accuracy can stay flat while that gets worse.
- **Forgetting F1/F2 may have already fixed it.** Run the diagnosis against the new models first.
  If matched + chunked retraining resolved the disagreement, record that and skip the fix.

---

# Phase F4 — Threshold Recalibration

**Estimated time:** ~1.5 hours
**Depends on:** F1, F2, F3 complete and gated.
**Gate:** Every threshold in `config.py` has been re-derived against the models actually going to
production.

## Objective

Every threshold currently in the project was calibrated against the *old* score distributions.
New models produce new distributions, so `RISK_THRESHOLDS = {0.50, 0.80}` and
`STREAM_FLAG_THRESHOLD = 0.6` are now arbitrary numbers wearing the costume of calibrated ones.
This phase is short, unglamorous, and is the most likely single thing to be skipped — don't.

## Build Prompts

### Prompt F4.1 — Recalibrate risk bands

```
Re-run the existing scripts/calibrate_thresholds.py against the NEW production whole-clip
detector (the F1 matched heads, with the F3 weight), on the ASVspoof2019 DEV split — never eval,
per the leakage discipline established in Phase 7.

Two required modifications to the existing script:
1. Read the classifier paths from config.PRODUCTION_WHOLECLIP_CLASSIFIERS and the weight from
   config.PRODUCTION_ENSEMBLE_WEIGHT_A rather than the hardcoded paths currently in it.
2. Replace any non-ASCII characters in its table output (the arrow and box-drawing characters)
   with ASCII — this script is known to crash on this machine's cp1252 console.

Keep the existing candidate sweep, the FNR/FPR tradeoff table, the printed recommendation, and
the require-typed-confirmation-before-writing behaviour. Record the chosen pair and the reasoning
in the config comment, replacing the now-stale Phase 7 rationale block (which cites the old
0.20/0.60, 0.30/0.70, 0.50/0.80 candidate numbers against the old model).
```

### Prompt F4.2 — Recalibrate the streaming flag threshold

```
Write scripts/fix_calibrate_stream_threshold.py that sweeps STREAM_FLAG_THRESHOLD over
[0.4, 0.5, 0.6, 0.7, 0.8] crossed with consecutive_flags_required over [1, 2, 3, 4], evaluating
each combination by replaying a held-out clip set through the real StreamingSession wired to the
NEW chunked classifiers.

Clip set: all Hindi MATCHED eval clips (soumya, held out — both real and synthetic), which is the
only held-out speaker available, plus the MATCHED versions of the 5 verified Phase 6 demo pairs
for continuity with earlier reporting. Use matched clips throughout — mixing matched and
unmatched audio in a threshold sweep reintroduces exactly the duration variation F1 removed, and
the resulting threshold would be tuned partly against an artifact.

Note that 3 of the 5 demo pairs involve byaquta and mahato, who ARE in the chunked model's
training set. Their clips are useful for demo continuity but are NOT held-out evidence — report
soumya's clips and the demo-pair clips as two separate blocks in the output, and select the
threshold on soumya's numbers alone.

For each combination report: false-positive rate (real clips that flag), detection rate
(synthetic clips that flag), and median seconds_to_flag among detected synthetic clips.

Selection rule, stated explicitly in the output: choose the combination with the LOWEST
false-positive rate; break ties by highest detection rate; break remaining ties by lowest median
seconds_to_flag. A false positive on genuine human speech is the failure this whole fix guide
exists to eliminate, and it outranks latency. Require typed confirmation before writing to
config.py.
```

### Prompt F4.3 — Re-verify the explainability window

```
The explainability overlay's window_seconds=1.5 / stride=0.75 defaults were chosen empirically
against the OLD whole-clip model (Phase 10), where they happened to separate one specific clip
pair. With chunk-native classifiers now available, windowed_attribution should use the CHUNKED
classifiers (via config.PRODUCTION_STREAMING_CLASSIFIERS) rather than the whole-clip ones — the
chunked family is trained precisely for scoring short windows, which is exactly what attribution
does.

Set windowed_attribution's default window_seconds to match config.STREAM_CHUNK_SECONDS exactly
(the window size the chunked heads were trained on) rather than leaving it independently
hardcoded — a mismatch here puts the chunked model back outside its training distribution, which
is the whole problem this guide exists to fix. The STRIDE may differ freely from the training
stride (it only controls heatmap time-resolution, not what the model sees per scoring call), so
keep stride at 0.75s or whatever gives a readable overlay.

Update render_explainability_overlay and windowed_attribution to take their detector from the
streaming/chunked config by default, then re-run the real-vs-synthetic separation check across
ALL 5 verified demo pairs (not just soumya_neutral_01), reporting nanmean per clip. Use the
MATCHED versions of those clips if F1 produced them, so the overlay is evaluated on the same
corpus the models were trained against. Confirm every real clip's mean sits below every synthetic
clip's mean. If that now holds across all five pairs, update the UI caption to drop the "verified
on one pair only" caveat and replace it with the actual number of pairs verified — and if it does
not hold, keep the caveat and say so.
```

## Tests

```powershell
python scripts\calibrate_thresholds.py
python scripts\fix_calibrate_stream_threshold.py
python -c "from src.voxguard.config import RISK_THRESHOLDS, STREAM_FLAG_THRESHOLD; print(RISK_THRESHOLDS, STREAM_FLAG_THRESHOLD)"
python -m pytest tests/ -q
```

## Gate F4 — do not proceed until all true

- [ ] `RISK_THRESHOLDS` re-derived against the new whole-clip detector, with an updated rationale
      comment in `config.py`
- [ ] `STREAM_FLAG_THRESHOLD` and `consecutive_flags_required` re-derived against the new chunked
      detector, selected on false-positive rate
- [ ] Explainability window re-verified against all 5 demo pairs, with the caption updated to
      match whatever is actually true
- [ ] Any test that hardcodes old threshold values now derives them from `config` (this broke
      three tests last time thresholds changed — expect it again)

## Common Pitfalls

- **Skipping this phase because the models "look fine."** Thresholds are model-specific. New
  model, new thresholds. Non-negotiable.
- **Calibrating on eval.** Use dev. Phase 7 got this right; keep it right.
- **Leaving stale test literals.** `tests/test_risk_bands.py` should read thresholds from config,
  not hardcode them — if it still hardcodes, fix it now rather than watching it fail again.

---

# Phase F5 — Propagate Everything Into the UI and API

**Estimated time:** ~2.5 hours
**Depends on:** F1–F4 complete and gated.
**Gate:** Every entry point demonstrably uses the new models, verified by observation rather than
by reading code.

## Objective

**This is the phase that makes the fix real.** Everything before this changed files in
`models/classifiers/`. If the Gradio app and REST API keep loading the old paths, nothing the
user or a judge sees has changed. The project currently hardcodes classifier paths in at least
six places; this phase replaces all of them with one config lookup and then *proves* it took.

## Call-Site Inventory

Every location known to construct a detector or reference a classifier path. Audit each one.

| # | File | What to change |
|---|---|---|
| 1 | `src/voxguard/config.py` | **Add** the two `PRODUCTION_*_CLASSIFIERS` dicts and `PRODUCTION_ENSEMBLE_WEIGHT_A` |
| 2 | `src/voxguard/streaming/session.py` | `StreamingSession.__init__` default detector → `PRODUCTION_STREAMING_CLASSIFIERS` |
| 3 | `app/app.py` | `get_detector()` → `PRODUCTION_WHOLECLIP_CLASSIFIERS`; streaming path → streaming set |
| 4 | `api/main.py` | startup model load → same two config dicts |
| 5 | `src/voxguard/explain/attribution.py` / `overlay.py` | default detector → `PRODUCTION_STREAMING_CLASSIFIERS` (per F4.3) |
| 6 | `scripts/simulate_stream.py` | → `PRODUCTION_STREAMING_CLASSIFIERS` |
| 7 | `scripts/evaluate_hindi_variants.py` | add the new variants to the comparison |
| 8 | `scripts/calibrate_thresholds.py` | → config (done in F4.1) |
| 9 | `tests/test_integration_e2e.py` | → config |
| 10 | `tests/test_api.py` | → config |
| 11 | `tests/test_app.py` | → config |
| 12 | Cross-Dataset Results tab content | regenerate reports (F6.1) |

## Build Prompts

### Prompt F5.1 — Single source of truth

```
Add to src/voxguard/config.py:

PRODUCTION_WHOLECLIP_CLASSIFIERS = {
    "wav2vec2": "models/classifiers/wav2vec2_hindi_matched_logreg.joblib",
    "wavlm":    "models/classifiers/wavlm_hindi_matched_logreg.joblib",
}
PRODUCTION_STREAMING_CLASSIFIERS = {
    "wav2vec2": "models/classifiers/wav2vec2_chunked_logreg.joblib",
    "wavlm":    "models/classifiers/wavlm_chunked_logreg.joblib",
}
PRODUCTION_ENSEMBLE_WEIGHT_A = <value chosen in F3>

(substituting the _v2 filenames if F3.2a produced them). Include a comment block explaining WHY
there are two families — whole-clip models score complete uploaded files, chunk-native models
score fixed-length streaming windows, and they are trained on different input distributions, so
using one where the other belongs reintroduces Issue 2.

Note: if F3 took the reweight branch (F3.2b), PRODUCTION_ENSEMBLE_WEIGHT_A may already exist in
config.py — that script writes it. Do not add a second definition; verify the existing value
matches what F3 selected and leave it.

Then add a helper in the same module or in src/voxguard/classifier/ensemble.py:

def build_production_detector(mode: str = "wholeclip") -> WeightedAverageDetector

returning a WeightedAverageDetector wired to the appropriate config dict and weight, raising a
clear ValueError on an unknown mode. Every call site in the inventory should call this helper
rather than passing paths itself. Add a startup-time existence check: if any configured
classifier file is missing, raise immediately with the missing path named, rather than failing
later inside an inference call where the error will be confusing.
```

### Prompt F5.2 — Rewire every call site

```
Update every file in the call-site inventory to obtain its detector from
build_production_detector(mode=...), removing all hardcoded classifier path strings. Specifically:

- StreamingSession's default detector: mode="streaming"
- app/app.py's get_detector(): mode="wholeclip" for Upload & Analyze and the API-equivalent
  paths; the Live Call Simulation tab must use the STREAMING detector via StreamingSession's
  default, not the whole-clip one
- api/main.py: load the WHOLE-CLIP detector once at startup (lifespan event). /analyze and
  /analyze-context both operate on complete uploaded files, so whole-clip is the correct family
  for every current endpoint. There is no streaming endpoint, so do NOT load the chunked detector
  — it would double API startup time and memory for nothing. Add a one-line comment saying
  exactly that, so a future reader doesn't "fix" the apparent omission.
- explain/attribution.py and overlay.py: mode="streaming" (per F4.3)
- simulate_stream.py, the evaluation scripts, and all three test files: mode-appropriate

Then grep the whole repository for the literal strings "hindi_combined_logreg" and
"baseline_logreg" and report every remaining occurrence with its file and line. Legitimate
remaining uses are: the F0 backup directory, comparison/evaluation scripts that deliberately
evaluate old versus new, and documentation. Any occurrence in app/, api/, or src/voxguard/ that
is not clearly a deliberate comparison is a missed call site — list them explicitly rather than
silently fixing, so they can be reviewed.
```

### Prompt F5.3 — Surface the change in the UI

```
The UI currently gives a user no way to know which model version produced a verdict, which makes
a model swap invisible and unverifiable. Add:

1. A small, always-visible model-provenance line in the app footer or under the title, reading
   the values from config at render time, e.g.:
   "Detector: weighted-average ensemble (wav2vec2 + WavLM) | whole-clip: hindi_matched |
    streaming: chunked | ensemble weight 0.5 | risk bands 0.5/0.8"
   This must be generated from config values, never a hardcoded string, so it cannot drift.

2. In the "Upload & Analyze" tab, label the whole-clip verdict and the streaming-simulation
   verdict with which model family produced each, since they are now genuinely different models
   and a judge asking "why do these two numbers differ?" deserves a precise answer.

3. Update the "Cross-Dataset Results" tab to display the NEW reports produced by this fix guide
   (fix_matched_comparison.md, fix_chunked_eval.md, fix_comparison_report.md) alongside the
   original Phase 3/4 tables, clearly labelled BEFORE FIX and AFTER FIX. Do not delete the
   originals — the before/after contrast is the most credible thing in the tab.
```

## Tests — verify by observation, not by reading code

The point of this phase is that the change is *visible*. Prove it three ways.

```powershell
# 1. Config is the only source
python -c "from src.voxguard.config import PRODUCTION_WHOLECLIP_CLASSIFIERS, PRODUCTION_STREAMING_CLASSIFIERS, PRODUCTION_ENSEMBLE_WEIGHT_A; print(PRODUCTION_WHOLECLIP_CLASSIFIERS); print(PRODUCTION_STREAMING_CLASSIFIERS); print(PRODUCTION_ENSEMBLE_WEIGHT_A)"

# 2. No stragglers outside deliberate comparisons
#    (Get-ChildItem -Recurse, not a ** glob — PowerShell's Select-String -Path does not
#     expand ** reliably and will silently match fewer files than you expect)
Get-ChildItem -Path src,app,api -Recurse -Filter *.py |
    Select-String -Pattern "hindi_combined_logreg|baseline_logreg" |
    Select-Object Path, LineNumber, Line

# 3. The app actually loads the new files — watch the startup log
python app\app.py
```

In the startup log, the `Loaded logreg classifier ... from ...` lines must name the **new**
filenames. This is the single most reliable check in the whole guide: the app tells you, in its
own logs, which files it opened.

Then in the browser:
- Provenance line shows the new model names
- Upload `soumya_neutral_01_clone.wav` → whole-clip verdict synthetic, streaming verdict flagged
- Upload `byaquta_neutral_01.wav` → whole-clip real, streaming **not** flagged (this was the
  false positive; its absence is the proof)
- Live Mic: speak normally for 30 seconds → does not flag
- Cross-Dataset Results tab shows both BEFORE FIX and AFTER FIX tables

And the API:

```powershell
uvicorn api.main:app --port 8000
# in a second terminal
curl.exe http://127.0.0.1:8000/health
```
Confirm the API startup log also names the new classifier files.

## Gate F5 — do not proceed until all true

- [ ] `build_production_detector()` exists and every inventory call site uses it
- [ ] Grep returns no unexplained old-path references in `src/`, `app/`, `api/`
- [ ] **The Gradio startup log names the new classifier files** (observed, not inferred)
- [ ] **The API startup log names the new classifier files**
- [ ] Provenance line renders in the UI with values read from config
- [ ] `byaquta_neutral_01.wav` no longer flags in the Live/streaming path through the actual UI
- [ ] Cross-Dataset Results tab shows before/after
- [ ] Full test suite green

## Common Pitfalls

- **Declaring success from code review.** Read the startup logs. Twice in this project a "fixed"
  component was still loading the old artifact.
- **Updating the app but not the API.** They are separate processes with separate startup paths.
- **Leaving `simulate_stream.py` on the old detector.** It is the script you will reach for when
  demonstrating streaming, and it will quietly contradict the UI.
- **Missing the explainability path.** `windowed_attribution` takes a detector argument; a caller
  passing the whole-clip detector silently defeats F4.3.

---

# Phase F6 — Verification, Regression & Documentation

**Estimated time:** ~2 hours
**Depends on:** F1–F5 complete and gated.
**Gate:** A single before/after report exists, every document reflects reality, and the full
suite is green.

## Objective

Prove the fix worked, quantify by how much, and leave the repository's documentation honest. An
undocumented fix is indistinguishable from an unfixed problem to anyone reading the repo.

## Build Prompts

### Prompt F6.1 — The before/after report

```
Write scripts/fix_generate_final_report.py producing models/reports/fix_comparison_report.md: a
single document comparing the F0 baseline against the post-fix system, reading the baseline from
models/reports/fix_baseline.json rather than from any remembered value. Include:

1. Issue 1 — duration+RMS-alone accuracy on the Hindi corpus, before vs after (expect ~83.3% →
   near chance). Plus the honest Hindi eval accuracy before vs after, with a sentence explaining
   that a DROP here is the correct outcome.
2. Issue 2 — chunk-level EER before (whole-clip model scored on chunks) vs after (chunk-native
   model), plus the per-clip streaming PASS/FAIL table for the previously-failing cases, plus the
   regenerated window-size sweep showing whether the new model separates where the old did not.
3. Issue 3 — soumya's 25 real clips: count with ensemble margin > 0.15, before vs after, plus the
   per-backbone agreement rate.
4. Regression check — ASVspoof2019 English eval accuracy/EER before vs after, to demonstrate the
   fixes did not cost English performance.
5. A short "what remains unfixed" section listing anything still open, written in the same
   register as ISSUES.md.

Every row must state which model file produced each number.
```

### Prompt F6.2 — Update ISSUES.md honestly

```
Rewrite ISSUES.md to reflect the post-fix state. For each of the three issues, either:
  - move it to a new "## Resolved" section with: what the fix was, the measured before/after, the
    commit/tag, and any residual limitation that survived the fix; or
  - keep it in the unresolved section with the NEW measurements, updated root-cause understanding
    from F3.1's diagnosis, and an honest statement of what was attempted and why it did not close.

Keep the file's existing structure, tone, and the "How These Three Interact" section (updating it
to describe the post-fix relationships). Add a dated changelog line at the top noting that this
revision follows FIX.md phases F0-F6.

Do not quietly delete an issue that was only partially fixed. A partial fix with measurements is
more credible than a deletion.
```

### Prompt F6.3 — Update the rest of the documentation

```
Update, in one pass:
1. PROGRESS.md — add a "Post-Build Fix (FIX.md F0-F6)" section summarizing what changed, the new
   model families and their config keys, the new thresholds, and a pointer to
   fix_comparison_report.md. Update the status line at the top.
2. HINDI_HINGLISH_DATASET_CARD.md — add a section describing the duration-matched corpus: how it
   was produced (trim-both-ends, never pad), how many pairs were matched and how many skipped, the
   before/after duration statistics, and the resulting honest evaluation numbers. Keep the
   original confound disclosure — the story is "found it AND fixed it", and deleting the first
   half weakens the second.
3. PHASE5_STREAMING_NOTES.md — add the chunk-native retraining outcome and the recalibrated
   streaming thresholds; mark the superseded window-size investigation as superseded rather than
   deleting it.
4. README.md — update any quoted accuracy/EER figures to the post-fix numbers, and add
   chunk-extraction and retraining to the reproduction instructions.
5. VoxGuard_Remediation_Guide.md — mark R1/R2/R3 as executed via FIX.md, with pointers, so the
   two documents don't contradict each other.
```

### Prompt F6.4 — Regression suite

```
Write tests/test_fix_regression.py asserting the specific behaviours this guide exists to
guarantee, so they cannot silently regress later:

1. config.PRODUCTION_WHOLECLIP_CLASSIFIERS and PRODUCTION_STREAMING_CLASSIFIERS point at files
   that exist on disk.
2. They point at DIFFERENT files (a copy-paste that makes streaming reuse the whole-clip model
   silently reintroduces Issue 2).
3. build_production_detector("wholeclip") and build_production_detector("streaming") both
   construct successfully and return objects exposing predict_waveform.
4. chunk_waveform's output is element-wise identical to StreamingBuffer's for the same input
   (the anti-drift guarantee).
5. A known real clip (byaquta_neutral_01.wav) does NOT flag through StreamingSession with
   production defaults — this is the exact false positive that motivated the fix, pinned as a test.
6. A known synthetic clip (soumya_neutral_01_clone.wav) DOES flag through StreamingSession with
   production defaults.
7. Thresholds in config are within sane ranges (0 < low_max < medium_max < 1;
   0 < STREAM_FLAG_THRESHOLD < 1) — cheap insurance against a bad calibration write.

Mark tests 5 and 6 with a pytest marker (e.g. @pytest.mark.slow) since they load models, and
document how to run the fast subset in the test file's docstring.
```

## Tests

```powershell
python scripts\fix_generate_final_report.py
python -m pytest tests/test_fix_regression.py -v
python -m pytest tests/ -q
git add -A; git commit -m "FIX.md F0-F6: duration confound, chunk-native streaming, WavLM head, recalibration"; git tag post-fix
```

## Gate F6 — final

- [ ] `fix_comparison_report.md` exists with all five sections and per-row model attribution
- [ ] `ISSUES.md` rewritten; nothing partially-fixed was quietly deleted
- [ ] `PROGRESS.md`, dataset card, streaming notes, README, remediation guide all updated
- [ ] `tests/test_fix_regression.py` passes, including the pinned false-positive test
- [ ] Full suite green
- [ ] `git tag post-fix` created

---

# Master Definition of Done

## Issue 1 — Duration confound
- [ ] Duration-matched corpus built (trim-both-ends, never padded), originals preserved
- [ ] Duration+RMS-alone accuracy dropped from ~83.3% to below 0.65
- [ ] Whole-clip Hindi heads retrained on matched data
- [ ] Honest post-fix Hindi numbers reported, with the drop explained rather than hidden
- [ ] ASVspoof's own duration confound checked and disclosed either way

## Issue 2 — Chunk-level detection
- [ ] Chunk-native classifier family trained, split at clip level before chunking
- [ ] Group-aware CV used; no chunk leakage
- [ ] `chunk_waveform` provably identical to `StreamingBuffer`
- [ ] Chunk-level EER on held-out data below 0.25, or branch-2 documented honestly
- [ ] `soumya_neutral_01_clone.wav` now flags; no real clip flags
- [ ] Streaming thresholds recalibrated on the new distribution

## Issue 3 — WavLM head
- [ ] Root cause diagnosed with evidence, not assumed
- [ ] Fix applied at the correct layer (retrain or reweight), justified in writing
- [ ] Confident-prediction count on soumya's real clips ≥ 18/25 (from 7/25)
- [ ] Synthetic recall preserved within 3 points

## Propagation (the part that makes it real)
- [ ] One config block is the only source of classifier paths
- [ ] Gradio startup log names the new files — **observed**
- [ ] API startup log names the new files — **observed**
- [ ] UI provenance line renders from config
- [ ] Cross-Dataset Results tab shows before/after
- [ ] No unexplained old-path references in `src/`, `app/`, `api/`

## Safety
- [ ] `pre-fix-baseline` tag and `models/classifiers_prefix_backup/` intact
- [ ] Rollback is one config edit plus threshold restore
- [ ] `list_enrolled_speakers()` returns `['byaquta']`
- [ ] Full test suite green

---

# Appendix A — Time Budget

| Phase | Estimate | Can it be cut? |
|---|---|---|
| F0 Baseline & diagnostics | 2h | No. Without it you cannot prove anything. |
| F1 Duration confound | 4h | No. Upstream of everything. |
| F2 Chunk-native training | 8–10h | Yes — take branch 2 of the failure path and document. |
| F3 WavLM head | 3h | Partially — F3.2b (reweight) alone is ~1h. |
| F4 Threshold recalibration | 1.5h | No. New models need new thresholds. |
| F5 UI/API propagation | 2.5h | No. Skipping it means nothing visibly changed. |
| F6 Verification & docs | 2h | Trim to F6.1 + F6.2 if desperate. |

**If you have only one working day:** F0 → F1 → F3 (reweight branch) → F4 → F5 → F6.1/F6.2.
That fixes Issues 1 and 3, recalibrates, ships it visibly, and documents it — leaving Issue 2
open but now with a measured, retraining-scoped explanation rather than an untested one.

**Do not start F2 without a clear 10-hour window.** A half-finished chunk extraction leaves you
with mismatched artifacts and a config pointing at files that do not exist.

# Appendix B — Quick Reference

```powershell
# Standard local preamble
cd D:\VoxGuard
.\.venv\Scripts\Activate.ps1

# Multi-line python without PowerShell quoting pain
@'
<python here>
'@ | Out-File -FilePath tmp.py -Encoding utf8
python tmp.py
Remove-Item tmp.py

# Health checks to run after every phase
python -c "from src.voxguard.speaker.enrollment import list_enrolled_speakers; print(list_enrolled_speakers())"
python -m pytest tests/ -q
git status --short
```

**Model files are never overwritten.** If a script would write to an existing `.joblib`, that is
a bug in the script, not a convenience.
