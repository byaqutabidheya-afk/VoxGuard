# VoxGuard — Complete Vibe-Coding Build Guide (v4)
### SIH26104 — AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks

This guide turns `Techstack.md` and `voice_cloning_features.md` into a phase-by-phase build plan you can feed directly to a coding agent (Claude Code, Cursor, etc.), one phase — often one prompt — at a time.

**v4 change:** built to make Phase 2-3's full three-dataset training/eval pipeline (ASVspoof2019 + WaveFake + In-the-Wild) actually fit inside **Kaggle's free tier (30 GPU-hours/week, 12-hour session cap) within a one-week solo build** — the scenario this revision was commissioned for: Phase 0 and Phase 1 Prompts 1.1-1.2 already complete, everything else still ahead, one week on the clock. Two changes drive it: (1) a defensible, stratified subset of WaveFake for cross-dataset eval — mirroring the subsetting pattern v3 already used for In-the-Wild, and already implied by v3's own Phase 1 disk-budget line ("...WaveFake subset...") but never actually implemented in Prompt 1.2's code, and (2) routing ALL embedding extraction — including the WaveFake/In-the-Wild cross-dataset eval that v3 left as local, uncached, file-by-file CPU inference via `detector.predict()` — through a single consolidated Kaggle session with proper caching. **Nothing about the classifier architecture, the feature set, or any of the 9 standout features changes** — this is a scheduling and data-plumbing fix to a real bottleneck v3 didn't fully account for, not a scope change. Phase 0 and Phase 1 Prompts 1.1-1.2 are byte-for-byte unchanged from v3 (they were already built when this revision was commissioned); Phases 5-11 are also unchanged from v3, since none of them touch Kaggle or the three-dataset pipeline. See Phase 1's new "Kaggle Free-Tier Master Strategy" section, inserted directly after Prompt 1.2, for the full reasoning, the GPU-hour math, and a day-by-day one-week schedule.

**v2 change:** this revision is built for **local-first development on a CPU/iGPU-only laptop (no dedicated GPU)** — specifically profiled against an AMD Ryzen AI 7 350 (8-core Zen5/Zen5c, Radeon 860M iGPU, XDNA2 NPU) — using **Kaggle's free GPU tier** as the one deliberate cloud burst for the single step that genuinely needs a GPU (embedding extraction). Everything else — the app, streaming, training, cloning generation, and the live demo — runs entirely on the local machine, with no dependency on Colab quotas or a paid cloud tier. See Phase 0's new "Local Hardware Profile & Free-Service Workflow" section before starting.

**v3 change:** a direct gap analysis against the **official SIH26104 problem statement text** (not just `voice_cloning_features.md`'s derived feature list) surfaced four named PS components with no corresponding build step: prosody/behavioral analysis, contextual risk enrichment (known contacts + transaction context), a platform integration API, and a privacy/compliance module. This revision folds all four into the existing phases rather than adding new ones, so the phase numbering and dependency structure from v2 are unchanged — only Phases 2, 6, 8, 9, and 11 grew new prompts. See each phase's updated Objective for what changed and why.

---

## How to use this guide

1. Work through phases **in order**. Each phase declares prerequisites from prior phases — don't skip ahead.
2. Each phase contains **numbered prompts**. Paste each prompt into your coding agent as its own turn. Don't paste an entire phase at once — the agent does better with one deliverable per prompt, and you get a natural checkpoint to run tests before moving on.
3. After every prompt, run the **Tests** listed for that step before continuing. If a test fails, paste the failure output back to the agent and ask it to fix it before moving to the next prompt.
4. Tick the **Definition of Done checklist** at the end of each phase before starting the next one. Partial phases compound into painful debugging later — don't skip the checklist.
5. Time estimates are carried over from `voice_cloning_features.md` and assume a solo builder with an AI coding agent doing the typing. Adjust for your team size.

---

## Feature → Phase Map

| Feature # | Feature | Category | Phase | Est. Hours |
|---|---|---|---|---|
| — | Environment, repo, accounts | Infra | Phase 0 | 2 |
| — | Dataset acquisition & preprocessing | Infra | Phase 1 | 4 |
| 1 | Embedding + Classifier Core **+ Prosody/Behavioral Features** | Must-have | Phase 2 | 10 |
| 2 | Cross-Dataset Generalization Layer | Must-have | Phase 3 | 4 |
| 3 | Code-Switched Hindi/Hinglish Test & Training Track | Must-have | Phase 4 | 9-10 |
| 4 | Real-Time Chunked Streaming Engine | Must-have | Phase 5 | 5 |
| 5 | Live Interactive Demo App (Gradio) **+ Privacy-Preserving Session Logging** | Must-have | Phase 6 | 7 |
| 6 | Graduated Risk Meter + Prevention Prompt | Must-have | Phase 7 | 3 |
| 7 | Speaker Voiceprint Verification **+ Deletion/Retention** | Standout (top priority) | Phase 8 | 7.5 |
| 8 | Multimodal Call-Context Risk Fusion **+ Contextual Enrichment (known contacts, transaction context)** | Standout | Phase 9 | 9 |
| 9 | Explainability Overlay | Standout | Phase 10 | 4 |
| — | Integration, deployment, demo rehearsal **+ REST API + Privacy Documentation** | Infra | Phase 11 | 8.5 |

**Total: ~73-74 hours.** This is a meaningful jump from v2's ~62-63 hours — the v3 additions (prosody features, contextual enrichment, the API layer, privacy logging) are a direct response to the official PS naming these as distinct components, not scope creep for its own sake, but they do cost real hours. If your timeline can't absorb this, here's the priority order for what to cut, cheapest/lowest-value-lost first:

1. **Phase 11's API layer (Prompt 11.3, ~3h)** — cut first if pressed. It's a genuine PS-fidelity answer but the least central to the live demo; you can honestly tell judges "the architecture is API-ready, here's the endpoint design" without having built it, which is a materially weaker but still defensible answer.
2. **Phase 9's contextual enrichment (Prompts 9.5-9.6, ~3h)** — cut second. Without it, Phase 9 still delivers the full audio+language fusion story from v2, just without the transaction-context/known-contact multipliers on top.
3. **Phase 2's prosody features (Prompts 2.3-2.4, ~3h)** — cut only if desperate; this is the PS's most explicitly named analysis layer ("prosody and behavioral analysis... modeling speech rhythm, pitch contours, pauses") and the cheapest to demo convincingly once built (it's just two extra rows in Phase 2's comparison table).
4. **Phase 6/8's privacy additions (session logging + voiceprint deletion, ~2h combined)** — cut last; these are cheap, and "privacy and compliance" being visibly addressed (even minimally) is worth more per hour spent than almost anything else on this list.

Also still available from v2: Phase 4's Prompt 4.9 (optional data augmentation), Phase 4's stretch tier (stick to 3 speakers, not 4-5), and Phase 10's optional gradient-saliency method. Features 10 (Safe-Word Companion) and 11 (Generalization Dashboard) from the original standout list remain **intentionally excluded** — Phase 3 already produces the underlying generalization numbers in table form if you want to add a chart later, and Phase 8's enrollment flow is a natural base for a safe-word feature if you have hours left over. These hour estimates assume Phase 2/3/4's embedding-extraction steps run on Kaggle's free GPU tier as directed in Phase 0 — running that same step on CPU-only local hardware instead would multiply those specific sub-steps by roughly 8-15x, so don't skip the Kaggle setup to "save a step."

**v4 addendum note:** the hour estimates above are unchanged from v3 — v4 doesn't add or remove build work, it fixes a scheduling gap in how Phase 1-3's dataset work was routed through Kaggle so the plan actually fits a one-week solo timeline without a hidden multi-day CPU bottleneck. See Phase 1's "Kaggle Free-Tier Master Strategy" section for the full reasoning and GPU-hour math, and its "Suggested One-Week Schedule" subsection for how these hours map onto seven days starting from Phase 1 Prompt 1.3.

---

## Global Prerequisites (before Phase 0)

- A laptop with a working microphone (needed for Phases 6, 8, 9 demo/testing) — no dedicated GPU required, see Phase 0's hardware profile
- Python 3.10+ installable on your machine
- A free HuggingFace account
- A free Kaggle account, **with phone number verification completed** (this unlocks GPU accelerators and is a Day-1 bottleneck exactly like the ASVspoof registration below — do it today, not when you first need a GPU)
- A free Google account (Colab) — kept only as a backup GPU option if Kaggle is ever unavailable
- A GitHub account
- Your AI coding agent of choice, connected to a terminal/repo it can read and write

## Explicitly Out of Scope (keep the agent from wandering into these)

Paste this into your coding agent's system/project instructions if it supports one, so it doesn't "helpfully" scope-creep:

| Cut | Why |
|---|---|
| Real telecom/carrier-level call interception | No hackathon team has telecom infra access. The demo simulates a "call" at the app level (mic input / uploaded audio). |
| Training a cloning/TTS model from scratch | XTTS-v2 / Indic TTS are used only to *generate* test clones, not built from scratch. |
| Production-scale enrollment database / auth system | One or two enrolled demo voiceprints is enough to prove the mechanism. |
| Native mobile app | Web app (Gradio) only. |
| Languages beyond English + Hindi/Hinglish | One well-executed non-English language beats three shallow ones. |
| Safe-Word Companion, Generalization Dashboard (features 10, 11) | Not requested for this build — revisit only if all of Phases 0-11 are done early. |
| Full production API: authentication, rate-limiting, multi-tenant SDKs, gRPC | Phase 11's REST API (Prompt 11.3) demonstrates the integration *pattern* the PS asks for with a single unauthenticated FastAPI service — a real deployment needs the production hardening this explicitly does not build. |
| SMS/email/in-app alert delivery channels, automated org-level response workflows | The PS's "Alerting and User Interaction Layer" names multi-channel delivery and configurable workflows — this guide builds the UI-prompt layer (Phase 7) only; wiring to an actual SMS/email gateway or workflow engine is enterprise integration work outside a hackathon's reach. |
| Formal data-protection/DPA compliance certification, encryption at rest | Phase 11's Privacy & Compliance README section documents the *pattern* (on-device inference, minimal retention, feature-only logging, right-to-erasure) a production system would extend, not a certified-compliant implementation. |
| Explicit phase-spectrum/group-delay synthesis-artifact detectors | The PS's acoustic-analysis bullet mentions "phase inconsistencies" specifically; this guide's classifier learns synthesis artifacts end-to-end from frozen embeddings rather than hand-engineering a dedicated phase-domain feature — a legitimate but different approach, worth naming to judges as a scoped-out sub-component rather than silently omitting. |
| Historical fraud-indicator databases, cross-session tracking beyond one enrolled voiceprint | The PS mentions "historical fraud indicators" as a contextual-enrichment input; this guide's Phase 8 tracks one static voiceprint per contact, not an evolving fraud-history record — a reasonable hackathon-scale substitute, not the full mechanism. |

## Problem Statement Coverage

A direct mapping from the official SIH26104 problem statement's five named Key Components to where each is (or isn't) addressed, so this is answerable in one look rather than reconstructed from memory during Q&A:

| PS Key Component | Status | Where |
|---|---|---|
| Multi-Layer Voice Authenticity Analysis: acoustic/spectral | ✅ Built | Phase 2 (embeddings + classifier) |
| Multi-Layer Voice Authenticity Analysis: prosody/behavioral | ✅ Built (v3) | Phase 2, Prompts 2.3-2.4 |
| Multi-Layer Voice Authenticity Analysis: phase inconsistencies specifically | ⚠️ Scoped out | Out of Scope table — end-to-end learned detection instead of a dedicated phase-domain feature |
| Multi-Layer Voice Authenticity Analysis: cross-session consistency | ✅ Built | Phase 8 (voiceprint enrollment + verification) |
| Real-Time Risk Scoring Engine: continuous score | ✅ Built | Phase 5 (streaming EMA) |
| Real-Time Risk Scoring Engine: configurable thresholds | ✅ Built | Phase 7 |
| Real-Time Risk Scoring Engine: contextual enrichment (known contacts, transaction context) | ✅ Built (v3) | Phase 9, Prompts 9.5-9.7 |
| Real-Time Risk Scoring Engine: historical fraud indicators | ⚠️ Scoped out | Out of Scope table |
| Alerting and User Interaction Layer: UI prompts | ✅ Built | Phase 7 |
| Alerting and User Interaction Layer: multi-channel (SMS/email), org workflows | ⚠️ Scoped out | Out of Scope table |
| Privacy and Compliance Module | ✅ Built (v3) | Phases 6, 8, 11 (session logging, voiceprint deletion, documented policy) |
| Platform and Integration APIs | ✅ Built (v3) | Phase 11, Prompt 11.3 |
| Multilingual support (Indian languages/accents) | ⚠️ Deliberately scoped down | One language track (Hindi/Hinglish) by design, not all — see Out of Scope table |

The ⚠️ rows are the honest, defensible answer to give a judge who asks about them directly — each has a documented reason, not a silent gap.



Any voice cloning you generate for testing (Phase 4) or any voiceprint you enroll (Phase 8) must be your own voice or a teammate's voice with explicit consent. Don't clone or enroll a real person's voice without their knowledge — this applies even inside a hackathon sandbox, and it's exactly the failure mode this project exists to detect in others.

---

## File Index

| File | Contents |
|---|---|
| `01_phase0_environment_setup.md` | Repo scaffold, dependencies (incl. FastAPI stack), accounts, local hardware profile, and the Kaggle free-GPU workflow used throughout Phases 2-4 |
| `02_phase1_data_acquisition.md` | ASVspoof2019/2021, WaveFake, In-the-Wild download + preprocessing |
| `03_phase2_embedding_classifier_core.md` | Feature 1 — embeddings + classifier, **plus prosody/behavioral feature extraction and baseline-vs-prosody-augmented comparison (v3)** |
| `04_phase3_cross_dataset_generalization.md` | Feature 2 |
| `05_phase4_hindi_hinglish_track.md` | Feature 3 — self-collected dataset build (25-sentence script, XTTS-v2 cloning), train/eval split, and actual classifier retraining (not zero-shot-only) |
| `06_phase5_realtime_streaming.md` | Feature 4 |
| `07_phase6_gradio_demo_app.md` | Feature 5 — live app, **plus privacy-preserving session logging with retention/purge (v3)** |
| `08_phase7_risk_meter_prevention.md` | Feature 6 |
| `09_phase8_speaker_voiceprint.md` | Feature 7 (standout) — enrollment + verification, **plus deletion/right-to-erasure (v3)** |
| `10_phase9_multimodal_fusion.md` | Feature 8 (standout) — audio+language fusion, **plus contextual risk enrichment: known-contact and transaction-context signals (v3)** |
| `11_phase10_explainability.md` | Feature 9 (standout) |
| `12_phase11_integration_deployment.md` | Final integration, **REST API layer (v3)**, HF Spaces deploy, privacy/compliance documentation, demo rehearsal |
| `Buildguidev3.md` | All of the above concatenated into one file |
| `BuildGuidev4.md` | **This file.** v3 concatenated, plus Phase 1's new "Kaggle Free-Tier Master Strategy" section (inserted after Prompt 1.2) and the resulting changes to Phases 1-3's Kaggle/dataset prompts. Phase 0 and Phase 1 Prompts 1.1-1.2 are unchanged; Phases 5-11 are unchanged. |

---
# Phase 0 — Environment, Repo & Account Setup

**Maps to:** Infrastructure (no feature number — everything else depends on this)
**Estimated time:** ~2 hours
**Depends on:** Nothing. Start here.

---

## Objective

Get a clean, reproducible Python environment, a sane repo structure, and every account/registration that has a waiting period started on day 1 — so that waiting for approval happens in parallel with coding, not in serial before it.

## Prerequisites

- Python 3.10+ installed and on PATH
- Git installed
- A terminal your coding agent can execute commands in
- Admin rights to install FFmpeg on your OS

## Accounts to create/register NOW (these can have approval delays)

- [ ] HuggingFace account — https://huggingface.co/join
- [ ] Accept model terms for `pyannote/embedding` on its HuggingFace model page (one click, but only visible after logging in)
- [ ] ASVspoof 2019 registration/download form (used in Phase 1) — start this immediately, it is historically the single slowest thing in this entire build
- [ ] Kaggle account, **with phone number verification completed** — this is your primary free GPU source (see below) and phone verification is a hard gate before GPU accelerators unlock on a notebook. Treat this exactly like the ASVspoof registration: start it now, don't wait until Phase 2 when you actually need the GPU
- [ ] Google account with Colab access (kept only as a backup GPU option)
- [ ] GitHub account + a new repository created for this project

---

## Local Hardware Profile & Free-Service Workflow

This guide assumes you're developing on a laptop with **no dedicated GPU** — profiled specifically against an AMD Ryzen AI 7 350 (8 cores: 4× Zen5 + 4× Zen5c, Radeon 860M iGPU, XDNA2 NPU rated ~50 TOPS). If your machine differs, the same reasoning still applies to any modern CPU-only or iGPU-only laptop.

**What runs locally, no GPU needed:** environment setup, data preprocessing, classifier head training (LogReg/MLP — trains in seconds regardless of hardware), XTTS-v2 clone generation (Phase 4 — slower per-clip than a GPU, but it's a one-time batch job you kick off and walk away from), the live Gradio app itself (Phase 6 onward — single-chunk inference on *base*-size transformer models is cheap enough for real-time use on an 8-core CPU), SpeechBrain speaker embeddings (Phase 8), and faster-whisper transcription (Phase 9 — already designed for CPU + int8 quantization, not a compromise).

**What genuinely needs a GPU:** extracting wav2vec2/WavLM embeddings across the full ASVspoof2019 dataset (~121k clips across train/dev/eval, doubled for Phase 3's dual-backbone ensemble). On this CPU alone that's realistically an 18-34 hour job. On a free cloud GPU it's roughly 1-3 hours. This is the **one** step this guide routes off your laptop.

**Deliberately skipped, and why:** the Radeon 860M iGPU (ROCm support for this generation of AMD mobile iGPU is still immature for PyTorch/HuggingFace workloads) and the XDNA2 NPU (would require exporting models through the Ryzen AI SDK/ONNX + quantization — a real embedded-AI workflow, but a multi-day detour with uncertain payoff for a hackathon timeline). Don't spend build time chasing either; CPU-only PyTorch plus the Kaggle GPU burst below covers everything this project needs.

**Free-service stack used throughout this guide:**

| Service | Role | Why |
|---|---|---|
| **Kaggle** | GPU burst for embedding extraction (Phases 2, 3, 4) | **30 GPU-hours/week, fixed and predictable** — comfortably covers this project's entire GPU need in far less than a week |
| Colab | Backup only, if Kaggle is ever unavailable | Free but with dynamic, unpublished quota that throttles unpredictably — don't build a critical-path dependency on it |
| HuggingFace Hub | Durable backup storage for extracted embeddings/checkpoints (private dataset repo) | Free, saves you from ever re-running a Kaggle session because local storage hiccuped |
| HuggingFace Spaces | Secondary public app link — code/report browsing, not the live-demo surface (see Phase 11) | Free CPU-basic tier |
| GitHub | Repo, version control, submission | Free |

**One-time Kaggle setup (do this once, in Phase 0, not later):**
1. Kaggle account settings → API → "Create New Token" → downloads `kaggle.json`
2. Place it at `~/.kaggle/kaggle.json` (`%USERPROFILE%\.kaggle\kaggle.json` on Windows) and `chmod 600` it on Linux/macOS
3. `pip install kaggle` (already added to `requirements.txt` by Prompt 0.2)
4. Run `python scripts/kaggle_sync.py --check` (built in Prompt 0.6 below) to confirm authentication works, before you need it under time pressure in Phase 2

---

## Build Prompts

### Prompt 0.1 — Repo scaffold

```
Create a Python project scaffold for a voice-cloning detection app called "VoxGuard".
Directory structure:

voxguard/
  data/                # raw and processed audio, gitignored except .gitkeep
    raw/
    processed/
    metadata/
  models/               # saved classifier heads / checkpoints, gitignored except .gitkeep
  src/
    voxguard/
      __init__.py
      config.py         # central paths, constants, thresholds
      embeddings/
        __init__.py
      features/          # prosody/behavioral feature extraction + feature composition (Phase 2)
        __init__.py
      classifier/
        __init__.py
      streaming/
        __init__.py
      speaker/
        __init__.py
      fusion/
        __init__.py
      explain/
        __init__.py
      privacy/           # session logging, retention policy (Phases 6, 8, 11)
        __init__.py
      utils/
        __init__.py
        audio_io.py     # shared audio loading/resampling helpers
        logging_utils.py
  app/
    app.py              # Gradio entrypoint (built in later phases)
  api/
    main.py             # FastAPI entrypoint (built in Phase 11) — separate process from app/app.py,
                         # both import from src/voxguard so there's no duplicated logic
  tests/
    __init__.py
  scripts/               # one-off data download/prep scripts, including the Kaggle sync helper
  notebooks/             # Kaggle notebooks for GPU work (primary), Colab as backup
  requirements.txt
  .gitignore
  README.md

Generate this structure with placeholder files. .gitignore should exclude data/raw, data/processed,
models/, __pycache__, .venv, *.pyc, .ipynb_checkpoints. README.md should have a one-paragraph
project description based on: "VoxGuard is a real-time voice cloning detection and prevention
system built for SIH26104. It classifies audio as real or synthetic using pretrained
self-supervised speech embeddings and handcrafted prosody features, adds a prevention layer with
actionable guidance and contextual risk enrichment, and extends to speaker voiceprint
verification, multimodal call-context risk fusion, an explainability overlay, a REST API for
platform integration, and a privacy-preserving logging layer." Do not implement any logic yet —
this is scaffolding only.
```

### Prompt 0.2 — Dependencies

```
Create requirements.txt for the VoxGuard project (Python 3.10+) pinning to compatible major
versions (not exact patch pins) for: torch, torchaudio, transformers, huggingface_hub,
scikit-learn, librosa, soundfile, numpy, pandas, matplotlib, gradio, speechbrain,
pyannote.audio, faster-whisper, TTS (Coqui TTS package for XTTS-v2), kaggle, fastapi, uvicorn,
python-multipart (needed for FastAPI file-upload endpoints in Phase 11), python-dotenv,
pytest, httpx (needed for FastAPI's TestClient in Phase 11's API tests).
Add a comment block at the top noting: install the standard CPU build of torch/torchaudio
(`pip install torch torchaudio` with no special index-url) — this project has no local CUDA GPU
to target, so there's no reason to install a CUDA build locally. GPU-accelerated work happens on
Kaggle notebooks instead (see Phase 0's Kaggle workflow), which come with torch+CUDA preinstalled.
Then create scripts/setup_env.sh (bash) that:
1. Creates a venv at .venv
2. Activates it
3. Upgrades pip
4. Installs from requirements.txt
5. Prints ffmpeg version to confirm it's on PATH (does not attempt to install ffmpeg itself,
   just checks and prints a warning with OS-specific install instructions if missing)
6. Prints the detected CPU core count (os.cpu_count() via a small inline python -c call) as a
   reminder to benchmark torch.set_num_threads() in Phase 2 before running a full embedding
   extraction locally for a smoke test
Make the script idempotent (safe to re-run).
```

### Prompt 0.3 — Config module

```
Implement src/voxguard/config.py as a single source of truth for the whole project:
- BASE_DIR, DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_METADATA_DIR, MODELS_DIR resolved as
  absolute paths relative to this file's location
- SAMPLE_RATE = 16000
- Placeholder constants for later phases (leave as None or commented with a phase reference,
  don't invent values yet): EMBEDDING_MODEL_NAME, RISK_THRESHOLDS, STREAM_CHUNK_SECONDS,
  STREAM_OVERLAP_SECONDS, TRANSACTION_CONTEXTS, CONTACT_FAMILIARITY_MULTIPLIERS (the latter two
  filled in during Phase 9, Prompt 9.5)
- A get_device() function returning "cuda" if torch.cuda.is_available() else "cpu" — on a
  CPU/iGPU-only laptop this will always resolve to "cpu" locally, which is correct and expected;
  it only ever resolves to "cuda" when the same codebase runs unmodified inside a Kaggle GPU
  notebook, which is exactly the point of writing it this way rather than hardcoding "cpu"
- A get_num_threads_hint() function that returns os.cpu_count() (or a lower explicit override via
  a VOXGUARD_NUM_THREADS env var), used later to call torch.set_num_threads() before local CPU
  inference — leave a comment noting that hyperthreaded/SMT sibling counts don't always help
  transformer inference and it's worth a quick benchmark (Phase 2) rather than assuming more
  threads is always faster
Add a docstring at the top of the file explaining it is the single place to change paths/constants
project-wide — no other module should hardcode a path.
```

### Prompt 0.4 — Logging utility

```
Implement src/voxguard/utils/logging_utils.py with a get_logger(name) function that returns a
configured Python logging.Logger (INFO level by default, overridable via a VOXGUARD_LOG_LEVEL
env var, timestamped format). Every module built in later phases should use this instead of
print() for anything other than user-facing Gradio output.
```

### Prompt 0.5 — Shared audio I/O helper

```
Implement src/voxguard/utils/audio_io.py with:
- load_audio(path, target_sr=16000) -> (waveform: np.ndarray mono float32, sr: int) using
  soundfile/librosa, resampling to target_sr and downmixing to mono if needed
- save_audio(path, waveform, sr) for writing processed clips
- get_duration_seconds(path) -> float
Include type hints and docstrings. This will be reused by every phase from here on — keep it
dependency-light (soundfile + librosa only, no ffmpeg subprocess calls needed for this helper).
```

### Prompt 0.6 — Kaggle dataset sync helper

```
Write scripts/kaggle_sync.py as a thin wrapper around the official `kaggle` Python package/CLI
(add kaggle to requirements.txt) with two functions:
- upload_dataset(local_dir: str, dataset_slug: str, title: str, update: bool = False):
  creates a new private Kaggle Dataset from local_dir (using kaggle datasets create) if
  update=False, or pushes a new version (kaggle datasets version) if update=True. Print the
  resulting dataset URL. Handle the case where the local directory exceeds a rough 19GB warning
  threshold (Kaggle Datasets have a practical size ceiling) by warning, not failing.
- download_output(kernel_slug: str, output_dir: str):
  downloads the output files of a completed Kaggle Notebook (kernel) run via
  `kaggle kernels output` into output_dir, for pulling back embeddings/checkpoints produced by a
  GPU notebook run.
Also write a one-time setup note at the top of the file (as a comment, not code) explaining: the
user must place their kaggle.json API token (downloaded from Kaggle account settings) at
~/.kaggle/kaggle.json with permissions 600, before this script will authenticate. Add a
--check flag that just verifies the token is present and valid (calls a lightweight kaggle API
list command) without doing any upload/download, for a quick sanity test. See this phase's
"Local Hardware Profile & Free-Service Workflow" section above for the one-time Kaggle account
setup steps (API token creation, placement, permissions) this script depends on.
```

---

## Tests

```
# From repo root, after running scripts/setup_env.sh and activating .venv
python -c "import torch; print(torch.__version__)"
python -c "import transformers, gradio, sklearn, librosa; print('core deps OK')"
ffmpeg -version | head -n 1
huggingface-cli whoami   # should print your username, not an error
python scripts/kaggle_sync.py --check   # confirms Kaggle API token is valid
python -m pytest tests/ -q   # should collect 0 tests and exit 0 (no failures) at this stage
```

Manual checks:
- [ ] `voxguard/` structure matches the scaffold exactly
- [ ] `.gitignore` correctly excludes `data/raw`, `data/processed`, `models/` (confirm with `git status` after adding a dummy file to each)
- [ ] `config.py` imports cleanly with no side effects (`python -c "import src.voxguard.config"`)
- [ ] `load_audio`/`save_audio` round-trip a short test WAV without distortion (load, save, reload, compare shapes)

## Definition of Done Checklist

- [ ] Repo pushed to GitHub with the scaffold above
- [ ] `.venv` created and all dependencies install cleanly on your machine
- [ ] FFmpeg confirmed installed and on PATH
- [ ] HuggingFace account created, logged in via `huggingface-cli login`, pyannote terms accepted
- [ ] ASVspoof 2019 registration form submitted (even if approval is still pending — don't block on this, move to Phase 1's other datasets while it processes)
- [ ] Kaggle account created, **phone number verified**, `kaggle.json` API token installed, GPU accelerator confirmed selectable on a blank test notebook (Settings → Accelerator → GPU T4x2 or P100)
- [ ] Colab account confirmed working as a backup option (open a blank notebook, confirm a GPU runtime is selectable) — not required to actually use it, just confirmed available if Kaggle has an outage
- [ ] `scripts/kaggle_sync.py --check` passes
- [ ] `config.py`, `logging_utils.py`, `audio_io.py` implemented and unit-testable

## Common Pitfalls

- Don't skip the ASVspoof registration on day 1 — it is the one dependency in this entire project that isn't under your control and can take the longest.
- **Don't skip Kaggle's phone verification on day 1 either** — it's the second such dependency, and forgetting it until you're mid-Phase-2 and actually need the GPU will cost you a full day's delay for no good reason.
- CPU-only PyTorch is correct and sufficient for everything on this machine except the full-dataset embedding extraction in Phases 2-4 — that step alone goes to Kaggle. Don't try to install ROCm for the iGPU or chase NPU acceleration; both are immature/high-effort detours with no real payoff here (see the Local Hardware Profile section above).
- `pyannote/embedding` will silently fail with an auth error if you haven't accepted its model terms on HuggingFace even after logging in via CLI — this is a common early blocker, check it explicitly.
- A Kaggle API token (`kaggle.json`) with wrong file permissions (not `600` on Linux/macOS) will fail silently or with a confusing error on some setups — set permissions explicitly, don't assume the default is fine.

---
# Phase 1 — Data Acquisition & Preprocessing

**Maps to:** Infrastructure supporting Features 1, 2, 3
**Estimated time:** ~4 hours (plus dataset download/registration wait time, which runs in parallel with Phase 0/2 work)
**Depends on:** Phase 0 complete

---

## Objective

Acquire ASVspoof2019 (primary training data), ASVspoof2021/WaveFake/In-the-Wild (cross-dataset generalization eval), build a unified metadata schema across all of them, and build a standardized preprocessing pipeline (mono, 16kHz, silence-trimmed, cached as .wav).

## Prerequisites

- Phase 0 complete
- ASVspoof 2019 registration approved (check email) — if still pending, proceed with WaveFake/In-the-Wild first, they need no registration
- Disk space check: budget roughly 10-15GB free for ASVspoof2019 LA partition + WaveFake subset; In-the-Wild is the largest at roughly 30GB if you download the full set (you can use a subset for the cross-dataset eval — see Prompt 1.2)

## Dataset Reference

| Dataset | Source | Registration | Used For |
|---|---|---|---|
| ASVspoof 2019 (LA partition) | University of Edinburgh ASVspoof site | Yes | Primary training (Phase 2) |
| ASVspoof 2021 | ASVspoof site | Yes (same account) | Optional extra eval |
| WaveFake | Zenodo | No | Cross-dataset zero-shot eval (Phase 3) |
| In-the-Wild | Public download | No | Cross-dataset zero-shot eval (Phase 3) |

---

## Build Prompts

### Prompt 1.1 — Download & verify ASVspoof2019

```
Write scripts/download_asvspoof2019.py. It should NOT attempt to auto-download the dataset
(ASVspoof requires manual registration/download from their portal) — instead:
1. Print clear step-by-step manual download instructions with the current ASVspoof2019 LA
   partition URL structure and expected file names (LA_train, LA_dev, LA_eval, plus the
   protocol/label files).
2. Accept a --src_dir argument pointing to wherever the user manually placed the downloaded
   archives, and extract them into data/raw/asvspoof2019/ with subfolders train/dev/eval.
3. After extraction, verify integrity by counting files per split and comparing against the
   known official counts (LA train: 25380 utterances, dev: 24844, eval: 71237 — confirm these
   numbers by checking the ASVspoof2019 protocol documentation and use whatever the official
   docs state; if the script can't verify online, print the expected counts as constants at the
   top of the file with a comment to double check against the official protocol file, and compare
   actual extracted file counts against them, warning (not failing) on mismatch).
4. Parse the ASVspoof2019 protocol/label file into a pandas DataFrame with columns:
   [filepath, speaker_id, system_id, label] where label is "bonafide" or "spoof", and save it to
   data/metadata/asvspoof2019.csv.
```

### Prompt 1.2 — Download WaveFake + In-the-Wild

```
Write scripts/download_wavefake_itw.py with two functions:

download_wavefake(dest_dir): prints the Zenodo record URL and instructions (no login needed),
accepts --src_dir for a manually downloaded archive, extracts to data/raw/wavefake/, and builds
data/metadata/wavefake.csv with columns [filepath, label, generator] (label "bonafide"/"spoof",
generator = which TTS/vocoder produced each spoof file, parsed from WaveFake's folder naming
convention).

download_in_the_wild(dest_dir, subset_size=None): same pattern for the In-the-Wild dataset,
building data/metadata/in_the_wild.csv with columns [filepath, label, speaker]. Add a
subset_size parameter that, if set, randomly samples that many files (stratified across
bonafide/spoof) instead of using the full ~30GB set — default this to 5000 for a build-friendly
subset, with a clear docstring saying the user should re-run with subset_size=None before final
grading if they have the disk space and time.
```

---

## Kaggle Free-Tier Master Strategy (v4 — read this before Prompt 1.3)

**Why this section exists.** v3's Kaggle usage was split across two separate visits (Phase 2 for wav2vec2, Phase 3 for WavLM), and its cross-dataset zero-shot eval (Phase 3, Prompt 3.1) was never routed through Kaggle at all — it calls `detector.predict()` once per file, which does its own embedding extraction internally, on whatever hardware runs it. Run locally on CPU against WaveFake's full release (commonly cited at over 100,000 clips across roughly half a dozen vocoder/TTS systems — treat that count as approximate until Prompt 1.2b below confirms exactly what you downloaded), that's the same order of magnitude as the 18-34 hour full-ASVspoof CPU extraction Phase 2 already warns against. That gap, not Kaggle's GPU-hour quota, is the real risk to a one-week timeline — and it's the thing this section actually fixes.

### The constraint, stated plainly

| Limit | Value | Source |
|---|---|---|
| Weekly GPU quota | 30 GPU-hours/week | Kaggle's published free-tier limit (Phase 0) |
| Single-session runtime cap | 12 hours (GPU), then the kernel is killed regardless of progress | Kaggle notebook documentation |
| `/kaggle/working` auto-saved output | 20 GB per notebook version | Kaggle notebook documentation |
| Private Kaggle Dataset size | On the order of 100 GB (uncompressed) historically — verify the current figure in Kaggle's docs at upload time, since platform limits do change | Kaggle documentation (verify before committing to a packaging plan) |

### Decision 1 — Subset WaveFake for cross-dataset eval, the same way v3 already subsetted In-the-Wild

v3's own Phase 1 Prerequisites line already budgeted disk space for "ASVspoof2019 LA partition + **WaveFake subset**" — the intent to subset WaveFake was already there, it just was never wired into Prompt 1.2's actual code (only In-the-Wild got a `subset_size` parameter). v4 completes that: **Prompt 1.2b** below builds a stratified WaveFake eval subset (default 8,000 clips — the same order of magnitude as In-the-Wild's already-accepted 5,000 default, sized larger since WaveFake spans more generator systems to stratify across) from what you already downloaded in Prompt 1.2. The full WaveFake metadata and raw files are left untouched on disk, so a full-scale rerun stays possible later if time and Kaggle budget allow — this is the identical "well-documented subset is legitimate, an undocumented one is not" standard v3's Phase 1 Common Pitfalls already applied to In-the-Wild, just consistently extended to WaveFake.

This is a scope-preserving efficiency decision, not a feature cut: Feature 2 (Cross-Dataset Generalization Layer, Phase 3) still evaluates on all three datasets, honestly, exactly as v3 specified — it just does so against a defensible, disclosed sample of WaveFake instead of its full ~100k+-clip release, the same tradeoff v3 already made for In-the-Wild without controversy.

### Decision 2 — Consolidate all Kaggle GPU work into one session, with everything cached

v3 spread GPU work across Phase 2 (wav2vec2 only) and Phase 3 (WavLM, plus an uncached local zero-shot eval). v4 pulls all of it into **one Master Kaggle Session**, executed once you reach Phase 2:

- Both embedding backbones (wav2vec2 **and** WavLM — pulling Phase 3's pass forward)
- All three dataset scopes (ASVspoof2019 full train/dev/eval, the WaveFake subset, the In-the-Wild subset)
- Six extraction jobs total, all resumable independently, all downloaded and cached locally before you leave the session

Phase 3 then reads everything it needs from local cache — see that phase's new "v4: Zero-Shot Eval Now Runs on Cached Embeddings" section for how `zero_shot_eval_from_cache` (new in v4) replaces the slow per-file path for full-scale runs, without changing `predict()`/`predict_waveform()`'s signatures that Phase 5 onward depends on.

### Calibrate before you commit to a schedule

Don't trust a canned throughput number — clip-length distribution, batch size, and which GPU Kaggle assigns you (T4 x2 vs P100) all move the real number. Before running the full six-job extraction, time a fixed batch. This reads directly from the packaged **processed** audio (globbed by extension, not through the metadata CSV) specifically so it doesn't depend on the same-session path-wiring you're about to test with the real run — deliberately the simplest possible thing that can fail:

```
# Run this FIRST in the Kaggle notebook, before the full extraction — takes under a minute
import time, glob
import numpy as np, soundfile as sf
from src.voxguard.embeddings.extractor import EmbeddingExtractor
sample_paths = glob.glob('/kaggle/input/<dataset-slug>/asvspoof2019/**/*.wav', recursive=True)[:1000]
assert len(sample_paths) > 0, "No .wav files found under the attached input — check the dataset slug and Prompt 1.6's folder structure before going further"
waveforms = [sf.read(p)[0].astype('float32') for p in sample_paths]
ext = EmbeddingExtractor()
t0 = time.time()
ext.extract_batch(waveforms)
elapsed = time.time() - t0
print(f'{len(waveforms)/elapsed:.1f} clips/sec on this session\'s GPU — use this to sanity-check the plan below')
```

### Worked GPU-hour math (illustrative — recalibrate with your own number above)

Using a deliberately conservative placeholder of **40 clips/sec** (a pessimistic floor for a base-sized transformer on a T4 with batched short 2-4 second clips — your calibration step will very likely beat this):

| Job | Clips | × backbones | Forward passes | ÷ 40 clips/sec |
|---|---|---|---|---|
| ASVspoof2019 (train+dev+eval, official counts) | 121,461 | × 2 | 242,922 | ≈ 1.7 hours |
| WaveFake subset (Prompt 1.2b default) | 8,000 | × 2 | 16,000 | ≈ 0.11 hours |
| In-the-Wild subset (Prompt 1.2 default) | 5,000 | × 2 | 10,000 | ≈ 0.07 hours |
| **Total** | | | | **≈ 1.9 GPU-hours** |

Add session overhead (model downloads, `pip install`, attaching input, occasional restart) and budget **3-4 hours of wall-clock time across one, at most two, sessions**. That's comfortably inside both the 12-hour single-session cap and the 30-hour/week quota — even at 4x this pessimistic estimate (10 clips/sec, a genuinely bad-case throughput), total GPU time stays under 8 hours. **The 30-hour/week Kaggle quota is very unlikely to be this project's actual bottleneck** — the bottleneck v3 actually had was the uncached local CPU fallback described above, which this plan eliminates by construction.

### The Master Kaggle Session Plan

**Prep (no GPU, do this now — Prompts 1.2b through 1.6 below):**
1. Run Prompt 1.2b to audit actual WaveFake/In-the-Wild counts and build the stratified WaveFake subset.
2. Run Prompt 1.3 to build `data/metadata/unified.csv` (raw `filepath` rows at this point — no `processed_path` yet).
3. Run Prompt 1.4's preprocessing over all three: ASVspoof2019 (full), WaveFake (subset only), In-the-Wild (subset, already default from Prompt 1.2) — and confirm its v4 fix actually ran: `data/metadata/unified.csv` should now have a populated `processed_path` column, not just `filepath`. This is the single most load-bearing file in the whole v4 plan; if this step is skipped, everything from here through Phase 3's cache-based eval breaks in ways that won't necessarily fail loudly.
4. Run Prompt 1.6 to package all three into **one** combined Kaggle Dataset (three audio subfolders + one `metadata/unified.csv`, one slug, one "Add Input" step) and upload it.

**Session 1 (GPU — do this at the start of Phase 2; budget ~3-4 hours wall-clock):**
1. Open a Kaggle Notebook. Settings → Accelerator → GPU T4 x2 (or P100). Settings → Internet → On.
2. `!git clone` your repo, "Add Input" → attach the single combined dataset from the prep step, `pip install` any missing dependencies, then symlink the audio subfolders and copy `metadata/unified.csv` into place (full commands in Phase 2's Kaggle workflow section — this is the part that makes the code run unmodified whether it's on your laptop or here).
3. Run the calibration cell above; note your actual clips/sec.
4. Extract wav2vec2 embeddings: ASVspoof train/dev/eval, then the WaveFake subset, then the In-the-Wild subset.
5. Extract WavLM embeddings for the same three scopes — this is v3's Phase 3 pass, pulled forward so Kaggle is only opened once.
6. Download all six `.npy`/`.csv` pairs (`kaggle_sync.py`'s `download_output()`, or "Save Version" + manual download) into local `models/embeddings/`.
7. If a 12-hour boundary is approaching before all six finish, "Save Version" to persist completed jobs (each is independently resumable) and finish the rest in a short second session — the math above shows you should have five-plus weekly hours of slack even in the pessimistic case.

**Session 2:** only if Session 1 didn't finish everything. Most builds following this plan won't need it. Phase 3, under this plan, needs **zero further Kaggle sessions**.

### Suggested One-Week Schedule

Grounded in this guide's own per-phase hour estimates (Feature → Phase Map table above), starting from "Phase 0 and Phase 1 Prompts 1.1-1.2 done." Remaining work totals roughly 70-71 hours across 7 days — about 10 hours/day on average, which is aggressive; treat this as a starting point to adjust to your own capacity, not a guarantee.

| Day | Focus | Notes |
|---|---|---|
| 1 | Finish Phase 1 (Prompts 1.2b-1.6, ~3-4h) + write/smoke-test Phase 2's Prompts 2.1-2.6 locally (no GPU needed yet) | Kick off ASVspoof registration on day 1 if not already approved — see Phase 0 |
| 2 | Run Master Kaggle Session 1 (background, ~3-4h wall-clock) while continuing Phase 2 local work; finish Phase 2; do Phase 3 (should need ~0 new Kaggle time, ~4h) | Phase 3 is unusually fast this week specifically because Session 1 already did its GPU work |
| 3 | Phase 4 — Hindi/Hinglish track (~9-10h, the single largest remaining phase) | XTTS-v2 generation (Prompt 4.3) runs as a local CPU background batch job — kick it off, work on the rest of the phase while it runs |
| 4 | Phase 5 (~5h) + Phase 6 (~7h) | **Go/no-go checkpoint at end of today** — if meaningfully behind, use the priority cut list in this file's Feature → Phase Map section (API layer first, then contextual enrichment, then prosody, then privacy additions last) |
| 5 | Phase 7 (~3h) + Phase 8 (~7.5h) | |
| 6 | Phase 9 (~9h) | The single largest remaining phase after Phase 4 — if day 4's checkpoint flagged you as behind, this is a candidate to trim via Prompt 9.5-9.6 per the cut list |
| 7 | Phase 10 (~4h) + Phase 11 (~8.5h) | Tightest day in the plan (~12.5h) — start Phase 11's HF Spaces account and README skeleton earlier in the week on any lighter day if slack appears |

### Master Strategy — Definition of Done

- [ ] Ran the WaveFake/In-the-Wild count audit and built a stratified, documented WaveFake eval subset (Prompt 1.2b)
- [ ] Preprocessed and packaged all three datasets into one combined Kaggle Dataset (Prompt 1.6, v4 addition)
- [ ] Ran the calibration cell on an actual Kaggle GPU session before committing to the full extraction
- [ ] Master Kaggle Session 1 completed: wav2vec2 AND WavLM embeddings extracted and cached locally for all three dataset scopes
- [ ] Confirmed Phase 3 needs zero (or near-zero) additional Kaggle time
- [ ] Reviewed the one-week schedule and adjusted it against your own actual daily capacity

---

### Prompt 1.2b — Audit dataset scale and build a stratified WaveFake eval subset

```
Write scripts/audit_and_subset_wavefake.py that:
1. Loads data/metadata/wavefake.csv (built by Prompt 1.2) and data/metadata/in_the_wild.csv,
   and prints total row counts, plus counts broken down by label (bonafide/spoof) and, for
   WaveFake, by the "generator" column — this confirms exactly what you actually downloaded
   before deciding anything, rather than assuming a number from documentation.
2. Builds a stratified subset of WaveFake sized by a --subset_size argument (default 8000),
   sampling proportionally across BOTH label AND generator so no single vocoder/TTS system
   dominates the eval story (use sklearn's train_test_split with a combined label+generator
   stratification key, or a pandas groupby-sample if the smallest stratum is too small for
   sklearn's stratify to handle — fall back gracefully and warn if any generator has fewer than
   20 total samples, since that group can't be meaningfully subsampled further).
3. Saves the result to data/metadata/wavefake_subset.csv, and leaves data/metadata/wavefake.csv
   (the full set) completely untouched, so a full-scale WaveFake eval remains possible later if
   time and Kaggle budget allow.
4. Prints a before/after summary table (total clips, per-generator breakdown, before vs after
   subsetting) suitable for pasting into a dataset-card-style note about why this subset was
   used — this is the same documented-subset standard v3's Common Pitfalls already required for
   In-the-Wild, consistently extended to WaveFake here.
Do NOT re-download or delete any raw WaveFake files — this only changes which rows downstream
preprocessing (Prompt 1.4) and packaging (Prompt 1.6) operate on.
```

### Prompt 1.3 — Unified metadata schema

```
Implement src/voxguard/utils/metadata.py with a function
load_unified_metadata(dataset_names: list[str]) -> pd.DataFrame that reads the per-dataset CSVs
from data/metadata/ and returns a single DataFrame with a consistent schema:
[filepath (absolute path), label ("real"/"synthetic" — map "bonafide"->"real",
"spoof"->"synthetic"), dataset (source dataset name), split ("train"/"dev"/"eval" where known,
else "unknown")]. Also implement save_unified_metadata() to cache the combined result to
data/metadata/unified.csv. Add unit tests in tests/test_metadata.py checking the label mapping
and that no filepath is null after loading.
```

**v4 addition to Prompt 1.3:** two changes to `load_unified_metadata`:
1. It should read WaveFake's metadata from `data/metadata/wavefake_subset.csv` (Prompt 1.2b) by default when `"wavefake"` is requested, falling back to the full `data/metadata/wavefake.csv` only if the subset file doesn't exist. Add an explicit `use_full_wavefake: bool = False` parameter so using the full set is always an intentional override, never a silent default in either direction.
2. **(v4 fix — closes a gap in v3's original spec):** before rebuilding from the raw per-dataset CSVs, check whether `data/metadata/unified.csv` already exists AND its rows for the requested `dataset_names` already have a non-null `processed_path` column — if so, load and filter FROM `unified.csv` instead of rebuilding from source. Without this, `load_unified_metadata` as originally specified would rebuild from `data/metadata/wavefake_subset.csv` etc. every time it's called — which never has `processed_path` on it — silently discarding the column Prompt 1.4 added, no matter how many times `unified.csv` gets re-saved. This is what lets the SAME function correctly serve both the pre-preprocessing stage (Prompts 1.3-1.5, returns raw `filepath` rows) and everything from Phase 2 onward (returns `processed_path`-complete rows), including this file's v4 dataset-scoped extraction, without needing two different loader functions.

Everything else about this function (label mapping, schema, null-checks) is exactly as specified above — this is additive, not a rewrite.

### Prompt 1.4 — Preprocessing pipeline

```
Implement src/voxguard/utils/preprocess.py with a function
preprocess_dataset(metadata_df, output_dir, target_sr=16000, trim_silence=True) that, for every
row in metadata_df: loads the audio via the existing audio_io.load_audio helper, resamples to
target_sr, converts to mono if needed, optionally trims leading/trailing silence using
librosa.effects.trim, and writes the result to output_dir preserving a flattened but unique
filename (include the dataset name and original filename to avoid collisions). Return an updated
DataFrame with a new "processed_path" column. Make it resumable — skip files whose processed
output already exists. Log progress every 500 files using the logging_utils logger. Add a
--dry_run flag support (as a script wrapper in scripts/run_preprocessing.py) that processes only
the first 20 files of each dataset for a quick smoke test.
```

**v4 note on Prompt 1.4:** no code changes needed — `preprocess_dataset` already operates on whatever rows the DataFrame it's given contains. Because Prompt 1.3 now feeds it the WaveFake *subset* by default (not the full 100k+-clip release), this step's wall-clock time drops substantially — expect well under an hour for WaveFake's share instead of a multi-hour local CPU job. If you deliberately pass `use_full_wavefake=True` upstream, budget accordingly; that tradeoff is now explicit rather than accidental.

**v4 fix — where `processed_path` actually gets saved (closes a gap in v3's original spec):** v3 never explicitly stated where the `processed_path`-augmented DataFrame `preprocess_dataset` returns gets persisted, even though `get_asvspoof_splits` (Prompt 1.5), `package_for_kaggle.py` (Prompt 1.6), and this file's dataset-scoped extraction (Prompt 2.2, v4) all depend on it existing on disk. Have `scripts/run_preprocessing.py` call `save_unified_metadata()` (Prompt 1.3) again on the returned DataFrame, **overwriting** `data/metadata/unified.csv` — combined with Prompt 1.3's v4 cache-preference fix above, this is what makes `load_unified_metadata` return `processed_path`-complete rows from this point in the pipeline onward. Store `processed_path` as a path **relative to the repo root** (e.g. `data/processed/asvspoof2019/xxx.wav`), not an absolute local path — this is what makes the Kaggle symlink approach in Phase 2's workflow section resolve correctly with zero path-rewriting, on Kaggle or locally, without any code change between environments.

**Guard this behind `--dry_run` (v4, important):** this phase's Tests section runs `python scripts/run_preprocessing.py --dry_run`, which only processes the first 20 files per dataset. If the resave above fires on a dry run too, it would overwrite `unified.csv` with a 20-row-per-dataset DataFrame, silently destroying the full metadata the moment someone follows this guide's own recommended test command. Only call `save_unified_metadata()` on a real (non-`--dry_run`) run.

### Prompt 1.5 — Train/val/test split respecting official protocols

```
Implement src/voxguard/utils/splits.py with get_asvspoof_splits(unified_df) that returns
(train_df, dev_df, eval_df) using ASVspoof2019's OFFICIAL train/dev/eval split (do not
re-shuffle their official partition — that would leak speakers/systems across splits and
invalidate the classifier's EER numbers). For WaveFake and In-the-Wild (used only for zero-shot
eval in Phase 3), return the full processed set as a single eval-only split since they are never
trained on. Raise a clear error if called on a dataset with no known official split logic
implemented yet.
```

**v4 note on Prompt 1.5:** no code change — "the full processed set" here now means whatever Prompt 1.3/1.4 actually processed, which is the WaveFake *subset* by default (Prompt 1.2b) unless `use_full_wavefake=True` was set upstream. The split logic itself is unaffected either way.

### Prompt 1.6 — Package preprocessed data for Kaggle

```
Write scripts/package_for_kaggle.py that takes the fully preprocessed data/processed/ directory
plus data/metadata/unified.csv, organizes them into a clean folder structure suitable for a
Kaggle Dataset (a top-level folder with the processed audio and a metadata/ subfolder — Kaggle
Datasets work best as a flat, self-describing directory since Phase 2's Kaggle notebook will
mount it read-only at /kaggle/input/<dataset-slug>/), and calls scripts/kaggle_sync.py's
upload_dataset() (Phase 0, Prompt 0.6) to push it as a private Kaggle Dataset. Print the
resulting dataset slug/URL and remind the user to note it down — Phase 2's Kaggle notebook needs
it to attach the dataset. Support --update to push a new version if the processed data changes
later (e.g., after Phase 4 adds the Hindi/Hinglish track and it needs to be included too).
```

**v4 addition to Prompt 1.6:** structure the packaged output as **one** combined Kaggle Dataset: three top-level audio subfolders — `asvspoof2019/`, `wavefake/`, `in_the_wild/` (named to match the `dataset` values used everywhere else in this guide — `--dataset wavefake`, `load_unified_metadata(['wavefake'])`, `{model}_wavefake.npy` — not `wavefake_subset/`, which would be a needless naming mismatch, even though the *content* is the Prompt 1.2b subset) — plus **one** top-level `metadata/unified.csv`, a straight copy of the local `data/metadata/unified.csv` Prompt 1.4's fix already made `processed_path`-complete. Don't package a separate metadata CSV per subfolder; one merged file is simpler to reconstruct on the Kaggle side (see Phase 2's workflow, which copies this single file into place rather than concatenating several). One combined dataset means the Master Kaggle Session (see "Kaggle Free-Tier Master Strategy" above) needs only a single "Add Input" step to reach everything it extracts embeddings for. Expect combined size in the low tens of GB after WaveFake's subsetting — comfortably clear of Kaggle's private-dataset size ceiling, which has historically been on the order of 100 GB; verify the current figure in Kaggle's docs at upload time rather than assuming it, since platform limits change. If the packaged size still feels uncomfortably large, lowering `--subset_size` on Prompt 1.2b (WaveFake) or Prompt 1.2 (In-the-Wild) and re-running is a one-line change, not a re-download.

---

## Tests

```
python scripts/audit_and_subset_wavefake.py --subset_size 8000   # v4: run this first
python -m pytest tests/test_metadata.py -q
python scripts/run_preprocessing.py --dry_run
python -c "
from src.voxguard.utils.metadata import load_unified_metadata
df = load_unified_metadata(['asvspoof2019'])
assert df['label'].isin(['real','synthetic']).all()
assert df['filepath'].notnull().all()
print(df['label'].value_counts())
"
python -c "
# v4: confirm the WaveFake subset is actually stratified, not just truncated
import pandas as pd
df = pd.read_csv('data/metadata/wavefake_subset.csv')
print(df.groupby(['label','generator']).size())
assert df['generator'].nunique() > 1, 'Subset collapsed to a single generator — check stratification'
"
python scripts/run_preprocessing.py   # the FULL run, not --dry_run — v3 never actually stated this
                                        # step explicitly; without it nothing past this point has
                                        # real audio to work with
python -c "
# v4: confirm processed_path actually made it back into unified.csv for ALL rows, not just the
# dry-run's 20 — this is the single most load-bearing check in this whole phase
from src.voxguard.utils.metadata import load_unified_metadata
df = load_unified_metadata(['asvspoof2019', 'wavefake', 'in_the_wild'])
assert 'processed_path' in df.columns, 'processed_path missing — Prompt 1.4 v4 fix did not run'
assert df['processed_path'].notnull().all(), 'Some rows never got preprocessed — check for silent failures'
print(df.groupby('dataset').size())
"
python scripts/package_for_kaggle.py
```

Manual checks:
- [ ] Kaggle Dataset uploaded successfully; slug/URL noted down for use in Phase 2
- [ ] Extracted ASVspoof2019 file counts roughly match official protocol counts (small mismatches are fine if you're using a subset intentionally, large ones are not)
- [ ] Spot-play 3-5 random processed files per dataset — audio should be audible, not silent or corrupted
- [ ] `data/metadata/unified.csv` has no null filepaths and a sane real/synthetic balance per dataset
- [ ] Preprocessing script is resumable — kill it mid-run and re-run, confirm it skips already-processed files instead of redoing them
- [ ] (v4) WaveFake subset's per-generator breakdown looks proportionate to the full set's breakdown printed by Prompt 1.2b — no single generator disappeared or dominates

## Definition of Done Checklist

- [ ] ASVspoof2019 downloaded, extracted, and its official protocol parsed into metadata
- [ ] WaveFake downloaded/extracted with metadata
- [ ] In-the-Wild downloaded (or a stratified subset) with metadata
- [ ] All three datasets pass through the shared preprocessing pipeline into `data/processed/`
- [ ] Unified metadata schema implemented and tested
- [ ] Official ASVspoof2019 train/dev/eval split preserved (not re-shuffled)
- [ ] Total disk usage checked and within your machine's budget
- [ ] Preprocessed data packaged and uploaded as a private Kaggle Dataset, ready for Phase 2's GPU notebook to attach
- [ ] (v4) WaveFake audited (Prompt 1.2b) and a stratified, documented eval subset built — full WaveFake metadata preserved untouched for a possible future full-scale rerun
- [ ] (v4) All three datasets packaged into ONE combined Kaggle Dataset (three audio subfolders + one `metadata/unified.csv`, one slug) — not three separate uploads
- [ ] (v4) `scripts/run_preprocessing.py` run WITHOUT `--dry_run` at least once, and `unified.csv` confirmed to have `processed_path` populated for every row across all three datasets, not just the 20-file smoke test
- [ ] (v4) Reviewed the "Kaggle Free-Tier Master Strategy" section above and understand the Master Kaggle Session Plan before starting Phase 2

## Common Pitfalls

- Re-shuffling ASVspoof2019 instead of using its official split will let the same speaker/attack system appear in both train and eval, making your EER numbers look artificially good — a technical judge who knows this dataset will catch it immediately.
- Silence-trimming too aggressively on very short utterances can produce empty audio — guard against zero-length output in the preprocessing function.
- If In-the-Wild's full download is impractical on your connection/disk, a well-documented stratified subset is a legitimate and defensible choice — an undocumented one is not. Say so in your README.
- Package and upload only `data/processed/` (16kHz mono WAVs) to Kaggle, not `data/raw/` — the raw archives are much larger and Phase 2's GPU notebook only needs the preprocessed audio.
- (v4) Subsetting WaveFake without stratifying by generator would silently bias the cross-dataset eval toward whichever vocoder happens to be overrepresented in a naive random sample — always stratify (Prompt 1.2b handles this; don't hand-roll a simpler `df.sample(n)` instead).
- (v4) `--dry_run` must never trigger the `unified.csv` resave (Prompt 1.4's v4 fix) — if it does, running this phase's own recommended `--dry_run` test command silently truncates your entire metadata down to 20 rows per dataset. If `package_for_kaggle.py` or Phase 2 ever complain about a suspiciously small dataset, check `unified.csv`'s row count against what you expect before assuming the bug is elsewhere.
- (v4) Packaging the three datasets as three separate Kaggle Dataset uploads instead of one combined dataset means three separate "Add Input" steps in the Master Kaggle Session and more chances to forget one mid-session — combine them into one dataset with subfolders.

---
# Phase 2 — Embedding + Classifier Core

**Maps to:** Must-Have Feature 1
**Estimated time:** ~10 hours (revised up from ~7 hours to add prosody/behavioral feature extraction — see Prompts 2.3-2.6 and the note below)
**Depends on:** Phase 1 complete (preprocessed ASVspoof2019 with official splits)

---

## Objective

Build the engine everything else sits on: extract embeddings from a frozen pretrained self-supervised speech model (wav2vec2 or WavLM), extract a small set of handcrafted prosody/behavioral features (pitch, pauses, speaking rate — the acoustic cues a pure embedding classifier doesn't explicitly model), train a lightweight classifier head on top of each (and their combination), and wrap the winner in a clean inference function.

## Why Prosody Features Were Added

The problem statement names "prosody and behavioral analysis — modeling speech rhythm, pitch contours, pauses, and microvariations" as its own distinct analysis layer, separate from acoustic/spectral synthesis-artifact detection. A frozen transformer embedding *may* implicitly capture some of this, but nothing in the pipeline explicitly targets it, and there's no way to verify or claim it does. Prompts 2.3-2.4 add a small, explicit, interpretable prosody feature vector alongside the embeddings, and Prompts 2.5-2.6 honestly compare embedding-only vs. embedding+prosody classifiers rather than assuming the addition helps — the same honest-comparison pattern already used for Phase 3's ensembling and Phase 4's Hindi training.

## Prerequisites

- Phase 1 complete: `data/processed/` populated, ASVspoof2019 official splits available via `get_asvspoof_splits`, all three datasets packaged and uploaded as ONE combined Kaggle Dataset (Phase 1 Prompt 1.6, v4)
- Kaggle account with phone verification done and `kaggle.json` API token installed (Phase 0)
- HuggingFace account logged in (needed for auto-download of `facebook/wav2vec2-base` and `microsoft/wavlm-base-plus`)
- (v4) Reviewed Phase 1's "Kaggle Free-Tier Master Strategy" section — this phase is where you actually execute Master Kaggle Session 1

## Local Hardware Note

This is the one phase where local CPU-only hardware genuinely matters. Everything you *write and test* in this phase (correctness on a handful of files) runs fine locally on the Ryzen laptop. The *full-dataset production run* (all of train/dev/eval, ~121k clips, plus the cross-dataset scopes) should run on a Kaggle GPU notebook instead — see the workflow below. Don't run the full extraction locally; budget it for Kaggle from the start.

## Kaggle GPU Notebook Workflow (v4 — executes Phase 1's Master Kaggle Session Plan)

**v4 change from v3:** this session now also does WavLM extraction (v3 deferred that to Phase 3) and extracts embeddings for the WaveFake/In-the-Wild cross-dataset eval scopes (v3 left those to an uncached local CPU path in Phase 3 that this revision found to be a real one-week-timeline risk — see Phase 1's Master Strategy section for the full reasoning). Read that section first if you haven't; these are the mechanics, that section has the reasoning and GPU-hour math.

1. **Write and smoke-test locally first.** Implement Prompts 2.1-2.2 and run the small local smoke test in this phase's Tests section (a handful of files) to confirm correctness before spending any Kaggle GPU time on it — debugging is much faster locally than iterating inside a notebook session.
2. **Open a new Kaggle Notebook.** Settings → Accelerator → **GPU T4 x2** (or P100 if offered). Settings → Internet → **On** (needed for `pip install`, `git clone`, and downloading `facebook/wav2vec2-base` / `microsoft/wavlm-base-plus` from HuggingFace).
3. **Clone your code onto the notebook** (v4, new — neither v3 nor the steps below work without this, and it was never actually stated). Kaggle notebooks start empty except for the attached input; your `src/voxguard` package has to get there somehow. Simplest: `!git clone https://github.com/<you>/voxguard.git` as the first cell (your repo from Phase 0), then `%cd voxguard`. If the repo is private, don't paste a token into the notebook cell in plaintext — add it via Kaggle's **Add-ons → Secrets** and read it as an environment variable instead (`!git clone https://$GITHUB_TOKEN@github.com/<you>/voxguard.git`).
4. **Attach your Phase 1 combined dataset.** "Add Input" → search for the single dataset slug you noted from Prompt 1.6 (v4: now one dataset covering all three sources) → it mounts read-only at `/kaggle/input/<dataset-slug>/`.
5. **Install any extra dependencies** not already in the Kaggle base image (Kaggle notebooks ship with `torch`+CUDA and `transformers` preinstalled already, which covers most of this phase — you'll likely only need `pip install soundfile` or similar small additions; check against your `requirements.txt`).
6. **Reconstruct the local layout the code expects** (v4, made explicit): symlink each attached audio subfolder to where the code looks for it locally, and copy the single packaged metadata file into place:
   ```
   !mkdir -p data/processed data/metadata
   !ln -s /kaggle/input/<dataset-slug>/asvspoof2019 data/processed/asvspoof2019
   !ln -s /kaggle/input/<dataset-slug>/wavefake     data/processed/wavefake
   !ln -s /kaggle/input/<dataset-slug>/in_the_wild  data/processed/in_the_wild
   !cp /kaggle/input/<dataset-slug>/metadata/unified.csv data/metadata/unified.csv
   ```
   This is what makes `load_unified_metadata` (Prompt 1.3, v4) resolve `processed_path` correctly on Kaggle with zero code changes — the exact same calls that work locally now work here, because the relative paths inside `unified.csv` now resolve through these symlinks. Output embeddings to `/kaggle/working/` (Kaggle's writable output directory) via `--output_dir` if `scripts/extract_embeddings.py` supports it, or write there directly.
7. **Run the calibration cell** (v4, new — see Phase 1's Master Strategy section for the exact code) before committing to the full run, so your session plan is based on this session's actual throughput, not a guess.
8. **Run the full extraction — six jobs total, all in this same session (v4):** `--dataset asvspoof2019 --split train/dev/eval --model wav2vec2`, then `--dataset wavefake --model wav2vec2`, then `--dataset in_the_wild --model wav2vec2`, then repeat all three with `--model wavlm`. Each job is independently resumable (Prompt 2.2's `output_path already exists` check) — if the session times out partway through, "Save Version," open a fresh session, and it picks up where it left off.
9. **Persist and pull the results back.** Either "Save Version" on the notebook (Kaggle keeps output files attached to that version, downloadable from the notebook's Output tab) or use `scripts/kaggle_sync.py`'s `download_output()` (Phase 0, Prompt 0.6) from your local machine once the run completes. Drop all six downloaded `.npy`/`.csv` pairs into your local `models/embeddings/`, matching Prompt 2.2's (now dataset-scoped) naming convention exactly.
10. **Track your 30 GPU-hours/week budget.** Kaggle shows remaining weekly quota in the notebook's Settings panel. Per Phase 1's worked math, all six jobs together should be a small fraction of that — you have plenty of headroom, and (v4) this should be the ONLY Kaggle session the whole project needs; Phase 3 reads from the cache this session produced rather than opening Kaggle again.

---

## Build Prompts

### Prompt 2.1 — Embedding extractor module

```
First, in src/voxguard/config.py, set the EMBEDDING_MODEL_NAME placeholder (added in Phase 0) to
"facebook/wav2vec2-base" as the project default, with a comment noting "microsoft/wavlm-base-plus"
is the alternative used later in Phase 3.
Implement src/voxguard/embeddings/extractor.py with a class EmbeddingExtractor:
- __init__(self, model_name=None, device=None): if model_name is None, default to
  config.EMBEDDING_MODEL_NAME rather than hardcoding a string here, so the project-wide default
  lives in one place. Loads the model and
  processor/feature-extractor from HuggingFace transformers (Wav2Vec2Model + Wav2Vec2FeatureExtractor,
  or the WavLM equivalents if model_name contains "wavlm"), moves model to device (use
  config.get_device() if device is None), sets model.eval(), and freezes all parameters
  (requires_grad_(False)) since we are NOT fine-tuning the backbone.
- extract(self, waveform: np.ndarray, sr: int) -> np.ndarray: resamples to 16000 if needed,
  runs the model under torch.no_grad(), mean-pools the last hidden state over the time dimension,
  and returns a 1D numpy array (768-dim for the base models). Handle both single-file and,
  as a separate method extract_batch(self, waveforms: list) -> np.ndarray, batched input for
  efficiency (pad to the longest sequence in the batch using the feature extractor's built-in
  padding, mean-pool per-sample using the attention mask so padding doesn't pollute the mean).
Add type hints, docstrings, and a __repr__ showing the model name and device.
```

### Prompt 2.2 — Feature caching

```
Implement src/voxguard/embeddings/cache.py with:
- extract_and_cache(df: pd.DataFrame, extractor: EmbeddingExtractor, output_path: str,
  path_col="processed_path", batch_size=16) -> None: iterates the DataFrame's audio files in
  batches, extracts embeddings via extractor.extract_batch, and saves the full embedding matrix
  as a .npy file at output_path, plus a parallel .csv (same base filename) mapping row index ->
  original filepath and label so embeddings can be matched back to metadata later. Log progress
  every 10 batches. Make it resumable: if output_path already exists, skip and warn rather than
  recomputing (offer a force=True override).
- load_cached_embeddings(path) -> (np.ndarray, pd.DataFrame): loads the .npy + matching .csv pair
  back.
Then write scripts/extract_embeddings.py as a CLI script (argparse: --split train/dev/eval,
--model wav2vec2/wavlm, --batch_size, --processed_dir with a default of config.DATA_PROCESSED_DIR
so it can be overridden to /kaggle/input/<dataset-slug>/processed when run on a Kaggle notebook
without touching any other code) that loads the relevant split via get_asvspoof_splits, runs
extract_and_cache, and saves to models/embeddings/{model}_{split}.npy (or /kaggle/working/ when
--output_dir is overridden similarly). This script is designed to run unmodified in three
places: a local smoke test on a handful of files, a full run on a Kaggle GPU notebook (primary —
see this phase's Kaggle workflow section), and Colab as a backup if Kaggle is ever unavailable.
```

**v4 addition to Prompt 2.2:** generalize `scripts/extract_embeddings.py`'s CLI with a `--dataset` argument (`asvspoof2019` / `wavefake` / `in_the_wild`, defaulting to `asvspoof2019` so v3's original behavior is preserved exactly when unspecified). For `asvspoof2019`, behavior is unchanged (uses `get_asvspoof_splits` and `--split`). For `wavefake` and `in_the_wild`, call `load_unified_metadata(['wavefake'])` / `(['in_the_wild'])` (Phase 1, Prompt 1.3) — **not** `get_asvspoof_splits`, which is ASVspoof-specific and will raise on these — and use the returned DataFrame directly, filtered to that dataset name if `load_unified_metadata` was called with more than one name. This is deliberate: it reuses the exact same `unified.csv` (already carrying `processed_path` once Prompt 1.4's run has re-saved it, per that prompt's v4 fix) rather than re-reading `data/metadata/wavefake_subset.csv` directly, which only has `[filepath, label, generator]` and no `processed_path` — reading it directly would point `extract_and_cache` at nonexistent (unprocessed, and on Kaggle, non-uploaded) files. Treat the whole returned DataFrame as one eval-only batch (`--split` is unused for these two, consistent with Prompt 1.5's eval-only split logic). Output filenames become `models/embeddings/{model}_{dataset}.npy` for these two (e.g. `wav2vec2_wavefake.npy`), extending — not replacing — the existing `{model}_{split}.npy` convention used for ASVspoof2019. This is what lets the Master Kaggle Session run all six extraction jobs through the exact same script and Kaggle setup.

**v4 correction — ignore the `--processed_dir` override mentioned above:** the original prompt text above describes overriding `--processed_dir` to `/kaggle/input/<dataset-slug>/processed` as the way to point the script at Kaggle's paths. Don't build that — it doesn't match what Prompt 1.6 actually packages (three separate `asvspoof2019/`/`wavefake/`/`in_the_wild/` folders, not one `processed/` folder), and it's unnecessary duplicate machinery. Phase 2's Kaggle workflow (step 6) uses the symlink approach exclusively — `data/processed/{dataset}` symlinked per dataset, `data/metadata/unified.csv` copied into place — so `processed_path` values resolve identically on Kaggle and locally with zero CLI flags or code branches for "which environment am I in." Skip implementing the `--processed_dir` argument; `--output_dir` (for where the resulting `.npy`/`.csv` go) is the only Kaggle-relevant flag actually needed.

### Prompt 2.3 — Prosody & behavioral feature extractor

```
Implement src/voxguard/features/prosody.py with a class ProsodyFeatureExtractor:
- extract(self, waveform: np.ndarray, sr: int) -> np.ndarray: computes a fixed 10-dimensional
  handcrafted feature vector per clip, entirely with librosa (CPU-only, no GPU/model download
  needed):
  1. F0 mean (Hz) — via librosa.yin (NOT librosa.pyin: yin is the faster, non-probabilistic
     variant and is the right tradeoff here since this needs to run per-chunk in Phase 5's
     streaming path later; note this speed-vs-accuracy choice in a code comment), computed only
     over frames where a pitch is detected
  2. F0 std (Hz)
  3. F0 range (max - min, Hz)
  4. Voiced fraction — proportion of frames with a detected pitch (voicing rate)
  5. F0 jitter proxy — mean absolute frame-to-frame F0 difference, a simple microvariation measure
  6. Pause ratio — proportion of frames below a short-time RMS energy threshold (use
     librosa.feature.rms with a fixed frame/hop length; pick a threshold as a fraction of the
     clip's own max RMS so it's loudness-invariant, not an absolute dB cutoff)
  7. Speaking rate proxy — onset count per second via librosa.onset.onset_detect
  8. RMS energy mean
  9. RMS energy std
  10. Zero-crossing rate mean (a cheap proxy for spectral noisiness/frication, complements F0)
  Handle short/edge-case clips gracefully: if no pitch is detected at all (e.g., near-silent
  clip), return 0.0 for the F0-derived features rather than raising, and log a warning.
  Return the 10 values as a fixed-order np.ndarray so downstream concatenation is deterministic.
- Add a FEATURE_NAMES class attribute (list of the 10 names above, in order) so any later
  explainability or debugging code can label the vector without hardcoding indices elsewhere.
Add a __main__ smoke test printing the vector (with labels) for a sample file passed via argv.
```

### Prompt 2.4 — Shared feature composition utility

```
Implement src/voxguard/features/compose.py with:
- extract_and_cache_prosody(df: pd.DataFrame, output_path: str, path_col="processed_path") ->
  None: mirrors Phase 2 Prompt 2.2's extract_and_cache resumability pattern (skip if
  output_path exists, force=True override, progress logging every 500 files — this one is much
  faster than the embedding version since there's no model forward pass, expect it to run in
  minutes not hours) but calls ProsodyFeatureExtractor instead of an embedding model. Explicitly
  run this LOCALLY, never on Kaggle — it's pure CPU librosa work with no benefit from a GPU
  notebook, and keeping it local avoids complicating the Kaggle workflow with an extra
  dependency for no speed gain.
- load_combined_features(cached_paths: list[str]) -> (np.ndarray, pd.DataFrame): loads two or
  more cached (.npy, .csv) pairs (e.g., a wav2vec2 embedding cache and a prosody cache),
  validates that their manifests align on the SAME filepaths in the SAME row order (raise a
  clear, specific error naming which paths mismatched if not — do not silently reindex or
  attempt a fuzzy join, a silent misalignment here would corrupt every downstream training run
  without any visible error), and returns the horizontally-concatenated feature matrix
  (np.concatenate along axis=1) plus the shared manifest DataFrame.
This is the ONE place feature concatenation happens in the whole project — Phase 3's ensembling
and Phase 4's Hindi training should both call this function rather than writing their own
concatenation logic, so a bug fix here fixes every phase at once.
```

**v4 addition to Prompt 2.4 — naming convention for the real prosody caches (closes a gap in v3's original spec):** v3 never actually named the output files Prompts 2.5/2.6 read prosody features from — only the smoke-test example above (`prosody_dev_smoke.npy`) has an explicit name anywhere in this guide. Standardize on `models/embeddings/prosody_{split}.npy` for ASVspoof2019 (`prosody_train.npy`, `prosody_dev.npy`, `prosody_eval.npy`), mirroring `extract_embeddings.py`'s `{model}_{split}.npy` convention exactly. This is what Prompt 2.5/2.6 below should actually write to and read from, and it's what `resolve_cache_path` (Phase 3, Prompt 3.1, v4) assumes when resolving the prosody path for `dataset == "asvspoof2019"`.

### Prompt 2.5 — Classifier head training (baseline vs. prosody-augmented)

```
Implement src/voxguard/classifier/head.py with:
- train_logistic_regression(X_train, y_train) -> sklearn.linear_model.LogisticRegression: fits
  with class_weight="balanced" (ASVspoof is heavily imbalanced toward spoof samples) and returns
  the fitted model.
- A PyTorch alternative class MLPClassifierHead(nn.Module): a 2-layer MLP
  (input_dim -> 128 -> 1 with ReLU + dropout(0.3), sigmoid output) plus a train_mlp(X_train,
  y_train, X_val, y_val, epochs=20, lr=1e-3) function using BCELoss, Adam, and early stopping on
  validation loss (patience=3). input_dim must NOT be hardcoded anywhere — always read from
  X_train.shape[1], since this now varies between the baseline (768) and prosody-augmented (778)
  feature sets built below.
- save_classifier(model, path) / load_classifier(path) using joblib for sklearn models and
  torch.save/torch.load (state_dict) for the MLP, auto-detecting type on load via a small
  metadata sidecar file (json with {"type": "logreg"|"mlp", "input_dim": ...}) — input_dim in
  this sidecar is what later lets VoxGuardDetector (Prompt 2.7) know whether a given saved
  classifier expects embedding-only or embedding+prosody input, so don't skip writing it.
Write scripts/train_classifier.py as a CLI script that:
1. Loads cached train+dev embeddings via load_cached_embeddings (Prompt 2.2) — this is the
   BASELINE feature set (768-dim).
2. Also loads cached train+dev prosody features (Prompt 2.4's extract_and_cache_prosody output)
   and builds the PROSODY-AUGMENTED feature set via load_combined_features([embedding_path,
   prosody_path]) (778-dim).
3. Trains both logreg and MLP classifier types on EACH feature set (4 classifiers total:
   baseline-logreg, baseline-mlp, prosody-logreg, prosody-mlp), saving all four to
   models/classifiers/ with clearly distinguishing filenames (e.g., baseline_logreg.joblib,
   prosody_logreg.joblib), for comparison in the next prompt.
```

### Prompt 2.6 — Evaluation script (EER-focused, baseline vs. prosody-augmented)

```
Implement src/voxguard/classifier/evaluate.py with:
- compute_eer(y_true, y_scores) -> float: computes Equal Error Rate, the standard ASVspoof
  metric (find the threshold where false acceptance rate equals false rejection rate). Implement
  this via scipy/sklearn's roc_curve to get fpr/tpr/thresholds, then find the point where
  fnr (=1-tpr) is closest to fpr, and return the EER as their average at that point.
- evaluate_classifier(model, X_eval, y_eval) -> dict with keys: accuracy, roc_auc, eer,
  eer_threshold, confusion_matrix (as nested list). Support both sklearn and MLP model types
  (dispatch on the same metadata sidecar from Prompt 2.5).
Write scripts/evaluate_classifier.py as a CLI script that loads the ASVspoof2019 eval-split
cached embeddings AND cached prosody features, builds both the baseline and prosody-augmented
eval feature sets (same construction as training), evaluates all FOUR trained classifier
variants, and prints/saves a comparison table (models/reports/asvspoof2019_eval_report.csv) with
columns [model, feature_set, accuracy, roc_auc, eer]. This table is what decides whether prosody
augmentation is worth keeping — don't skip straight to using it without checking this comparison.
```

### Prompt 2.7 — Inference wrapper

```
Implement src/voxguard/classifier/infer.py with a class VoxGuardDetector:
- __init__(self, embedding_model_name=None, classifier_path=..., use_prosody=None): loads an
  EmbeddingExtractor, and — if use_prosody is True, or if it's None and the loaded classifier's
  metadata sidecar (Prompt 2.5) indicates input_dim > the embedding-only dimension — also loads
  a ProsodyFeatureExtractor (Prompt 2.3). This auto-detection means the SAME VoxGuardDetector
  class transparently works with either a baseline or prosody-augmented classifier just by
  pointing classifier_path at the right saved model; callers never need to know which variant
  they're using.
- predict(self, audio_path: str) -> dict with keys {"label": "real"|"synthetic",
  "probability_synthetic": float in [0,1]}: loads audio via audio_io.load_audio, extracts the
  embedding (and prosody vector, concatenated in the same fixed order Prompt 2.4 uses, if
  use_prosody is active), runs the classifier's predict_proba (sklearn) or forward pass (MLP),
  and returns the dict.
- predict_waveform(self, waveform: np.ndarray, sr: int) -> dict: the same logic as predict()
  but taking an in-memory waveform directly instead of a file path — this is the method Phase 5's
  streaming engine calls per chunk, so it must not do any disk I/O.
This is the function every later phase (streaming, Gradio, fusion) will call — keep predict()'s
and predict_waveform()'s signatures stable and well-documented since they're now load-bearing
interfaces.
Add a __main__ block so `python -m src.voxguard.classifier.infer path/to/file.wav` prints the
prediction dict, for quick manual testing.
```

---

## Tests

```
python -m pytest tests/ -q -k embedding
python -c "
from src.voxguard.embeddings.extractor import EmbeddingExtractor
import numpy as np
ext = EmbeddingExtractor()
wf = np.random.randn(16000).astype('float32')  # 1 second of noise
emb = ext.extract(wf, sr=16000)
assert emb.shape == (768,), emb.shape
print('embedding shape OK:', emb.shape)
"
python -c "
from src.voxguard.features.prosody import ProsodyFeatureExtractor
import numpy as np
pext = ProsodyFeatureExtractor()
wf = np.random.randn(16000).astype('float32')
feat = pext.extract(wf, sr=16000)
assert feat.shape == (10,), feat.shape
print('prosody feature shape OK:', feat.shape, dict(zip(pext.FEATURE_NAMES, feat)))
"
python -c "
import torch, time
torch.set_num_threads(4)   # try 4 vs 8 and compare — more threads isn't always faster
from src.voxguard.embeddings.extractor import EmbeddingExtractor
import numpy as np
ext = EmbeddingExtractor()
batch = [np.random.randn(64000).astype('float32') for _ in range(8)]  # 8 x 4-second clips
t0 = time.time()
ext.extract_batch(batch)
print(f'8-clip batch: {time.time()-t0:.2f}s with {torch.get_num_threads()} threads')
"
# Local smoke test only — a few dozen files, to verify correctness before spending Kaggle GPU time
python scripts/extract_embeddings.py --split dev --model wav2vec2 --batch_size 8   # small local smoke run first
python -c "
from src.voxguard.features.compose import extract_and_cache_prosody
from src.voxguard.utils.splits import get_asvspoof_splits
from src.voxguard.utils.metadata import load_unified_metadata
train_df, dev_df, _ = get_asvspoof_splits(load_unified_metadata(['asvspoof2019']))
extract_and_cache_prosody(dev_df.head(20), 'models/embeddings/prosody_dev_smoke.npy')  # smoke test only
"

# Full-dataset run happens on Kaggle for embeddings (see this phase's Kaggle workflow section
# above) and locally for prosody (Prompt 2.4 — fast enough not to need Kaggle). v4: this single
# Kaggle session also covers WavLM and the cross-dataset scopes, run in the SAME session:
#   python scripts/extract_embeddings.py --dataset asvspoof2019 --split train --model wav2vec2
#   python scripts/extract_embeddings.py --dataset asvspoof2019 --split dev   --model wav2vec2
#   python scripts/extract_embeddings.py --dataset asvspoof2019 --split eval  --model wav2vec2
#   python scripts/extract_embeddings.py --dataset wavefake                   --model wav2vec2
#   python scripts/extract_embeddings.py --dataset in_the_wild                --model wav2vec2
#   python scripts/extract_embeddings.py --dataset asvspoof2019 --split train --model wavlm
#   python scripts/extract_embeddings.py --dataset asvspoof2019 --split dev   --model wavlm
#   python scripts/extract_embeddings.py --dataset asvspoof2019 --split eval  --model wavlm
#   python scripts/extract_embeddings.py --dataset wavefake                   --model wavlm
#   python scripts/extract_embeddings.py --dataset in_the_wild                --model wavlm
# then, back on your local machine once all outputs are downloaded to models/embeddings/:
python scripts/train_classifier.py
python scripts/evaluate_classifier.py
python -m src.voxguard.classifier.infer data/processed/asvspoof2019/<some_eval_file>.wav
```

Sanity thresholds to check in the eval report:
- Accuracy on the ASVspoof2019 eval split should clear a real classifier bar — well above 50/50 chance, and not suspiciously near 100% either (near-perfect on eval usually means a split leak, go back and check Prompt 1.5's split logic)
- EER in the single-digit-to-teens percent range is a reasonable frozen-embedding-plus-linear-head result; a very high EER (>30%) signals something is broken in embedding extraction or label mapping, not just "needs more tuning"
- Compare the baseline (768-dim) and prosody-augmented (778-dim) rows directly — prosody augmentation should either measurably lower EER, or should be honestly reported as not helping if it doesn't. Either outcome is a legitimate finding; silently shipping prosody without checking this comparison is not

## Definition of Done Checklist

- [ ] `EmbeddingExtractor` produces correctly-shaped embeddings for both single and batched input
- [ ] `ProsodyFeatureExtractor` produces a correctly-shaped 10-dim vector, with graceful zero-fallback on unvoiced/near-silent input
- [ ] Local smoke test (dev split, small batch) passes before any Kaggle GPU time is spent
- [ ] Thread-count benchmark run locally, `torch.set_num_threads()` set to whichever value was actually faster (don't assume — verify)
- [ ] Full train/dev/eval embeddings extracted on a Kaggle GPU notebook and downloaded back to local `models/embeddings/`, resumable
- [ ] Full train/dev/eval prosody features extracted LOCALLY (no Kaggle needed), resumable
- [ ] (v4) Calibration cell run on an actual Kaggle GPU session and compared against Phase 1's worked estimate before committing to the full six-job run
- [ ] (v4) WavLM embeddings for ASVspoof train/dev/eval extracted in this SAME Kaggle session (pulled forward from Phase 3) — Phase 3 should need zero additional Kaggle time
- [ ] (v4) wav2vec2 AND WavLM embeddings extracted and cached for the WaveFake subset and In-the-Wild subset, in this same session
- [ ] `load_combined_features` correctly concatenates and validates manifest alignment — confirm it actually raises on a deliberately misaligned pair, not just that it succeeds on an aligned one
- [ ] All four classifier variants (baseline-logreg, baseline-mlp, prosody-logreg, prosody-mlp) trained and saved with distinguishing filenames
- [ ] `compute_eer` implemented and validated against a hand-checkable toy example (e.g., perfectly separated scores should give EER ≈ 0)
- [ ] Evaluation report generated comparing all four variants on ASVspoof2019 eval
- [ ] Baseline-vs-prosody-augmented decision made and documented, based on the comparison table, not assumed
- [ ] `VoxGuardDetector.predict()` and `predict_waveform()` both work end-to-end, and correctly auto-detect whether the loaded classifier expects prosody features
- [ ] Chosen best classifier type AND feature set documented (e.g., "prosody-augmented MLP") with the metric that decided it

## Common Pitfalls

- Forgetting `model.eval()` and `torch.no_grad()` during embedding extraction will make it painfully slow and can introduce dropout-noise inconsistency between runs.
- Mean-pooling over padded timesteps without using the attention mask will corrupt embeddings for batched short clips — always mask before pooling.
- ASVspoof is class-imbalanced (many more spoof than bonafide samples) — `class_weight="balanced"` or an equivalent weighting in the MLP loss is not optional, skipping it will bias the classifier toward always predicting "synthetic."
- **Don't run the full-dataset extraction locally "just to see."** At CPU-only speeds this is realistically an 18-34 hour job for the full train+dev+eval split — the Kaggle workflow above exists specifically so you never need to find that out the hard way.
- Kaggle notebook sessions have a continuous runtime limit (several hours) and can disconnect on idle — this is exactly why Prompt 2.2's resumability matters here; save/persist outputs periodically rather than assuming one session will finish everything.
- If a `pip install` inside the Kaggle notebook seems to hang or fail, confirm the notebook's Internet toggle (Settings → Internet) is actually On — it's off by default on new notebooks and is a common silent blocker.
- **Row-order misalignment between an embedding cache and a prosody cache is the single most dangerous silent bug this phase can introduce** — if the two were extracted from DataFrames sorted or filtered differently, concatenating them without `load_combined_features`'s manifest validation would pair each clip's embedding with a DIFFERENT clip's prosody vector. Always go through `load_combined_features`, never concatenate the raw `.npy` arrays directly.
- Using `librosa.pyin` instead of `librosa.yin` will work but is noticeably slower — fine for the one-time batch extraction in this phase, but if you're tempted to reuse `ProsodyFeatureExtractor` anywhere in the Phase 5 streaming path, confirm you kept `yin`, since `pyin`'s extra cost compounds badly per-chunk.
- (v4) Don't split the wav2vec2 and WavLM extraction across two separate Kaggle visits "to keep the phases clean." The entire point of the v4 consolidation is running both backbones, across all three dataset scopes, in one sitting — splitting it back apart reintroduces the session-setup overhead (and the risk of forgetting to come back for the second half) this revision exists to remove.
- (v4) If you skip the calibration cell and the full six-job run looks like it will blow past 12 hours, don't panic-cut a dataset scope — "Save Version" to persist whatever finished (each job is independently resumable) and continue in a second session; Phase 1's math shows you should have several weekly hours of slack even in a pessimistic case.

---
# Phase 3 — Cross-Dataset Generalization Layer

**Maps to:** Must-Have Feature 2
**Estimated time:** ~4 hours
**Depends on:** Phase 2 complete (trained classifier, working inference wrapper)

---

## Objective

Prove the classifier isn't overfit to ASVspoof2019 by evaluating it zero-shot on WaveFake and In-the-Wild, then close some of the generalization gap by ensembling wav2vec2 + WavLM embeddings. This is the phase a technical judge will probe hardest — treat the honesty of the reported numbers as more valuable than making them look good.

## Prerequisites

- Phase 2 complete: trained classifier + `EmbeddingExtractor` + `VoxGuardDetector` working, and the baseline-vs-prosody-augmented decision (Phase 2, Prompt 2.6) already made
- WaveFake (subset) and In-the-Wild (subset) preprocessed with metadata (Phase 1)
- (v4) WavLM embeddings for ASVspoof2019 AND cross-dataset embeddings (WaveFake subset, In-the-Wild subset, both backbones) were already extracted in Phase 2's consolidated Master Kaggle Session (see Phase 1's "Kaggle Free-Tier Master Strategy") — **this phase should need zero new Kaggle GPU time**. If you skipped that consolidation, open Kaggle now for just the missing pieces before continuing.

## Carrying Forward Phase 2's Prosody Decision

This phase adds a second embedding backbone (WavLM), not a second prosody feature set — there's still only ONE `ProsodyFeatureExtractor` and ONE cached prosody vector per clip (Phase 2, Prompt 2.4), since prosody is a signal-processing feature independent of which transformer backbone is used. If Phase 2's comparison selected the prosody-augmented variant, this phase's ensemble should be **wav2vec2 + WavLM + prosody** (768+768+10 = 1546-dim); if Phase 2 selected the baseline, the ensemble stays **wav2vec2 + WavLM only** (1536-dim, as originally planned). Prompt 3.3 below is written to carry this forward automatically via `load_combined_features` rather than needing a separate decision here.

## v4: Zero-Shot Eval Now Runs on Cached Embeddings, Not Per-File CPU Extraction

v3's `zero_shot_eval` (Prompt 3.1 below) calls `detector.predict()` once per file in the WaveFake/In-the-Wild metadata — and `predict()` does its own embedding extraction internally (Phase 2, Prompt 2.7), on whatever hardware runs it. Run locally on CPU against a full, un-subsetted WaveFake set, this would cost the same order of magnitude as the "18-34 hour" full-ASVspoof CPU extraction Phase 2 already warns against — a real risk to a one-week timeline that v3 didn't flag, since nothing routed this particular call through Kaggle.

v4 fixes this two ways, without touching `predict()`/`predict_waveform()`'s signatures — Phase 5 onward still calls these exactly as documented; the live streaming and demo paths are completely unaffected:

1. **WaveFake is subsetted** (Phase 1, Prompt 1.2b) to a size Kaggle extracts in minutes, not hours.
2. **Zero-shot eval gets a cache-based path** (`zero_shot_eval_from_cache`, new in Prompt 3.1 below) that loads the embeddings Phase 2's Master Kaggle Session already extracted and cached, and runs ONLY the lightweight classifier head — no re-extraction, no GPU needed, fast enough to run locally even for Prompt 3.4's four-model × three-dataset sweep. The original file-by-file `zero_shot_eval` function still exists, is still correct, and is still the right tool for a quick ad hoc check on a handful of files — it's just no longer the recommended path for full-scale runs.

**A dependency this fix has that's easy to miss:** if Phase 2's comparison (Prompt 2.6) selected the **prosody-augmented** variant, the cache-based functions above need a prosody cache for WaveFake and In-the-Wild too, not just ASVspoof2019 — and nothing in v3's original Phase 2 ever extracted prosody for those two (Prompt 2.4's `extract_and_cache_prosody` was only ever pointed at ASVspoof2019's splits). Before running any cache-based cross-dataset eval under the prosody-augmented variant, run this once, locally (no Kaggle needed — same reasoning as Phase 2's original prosody step: pure CPU librosa work, and at subset scale, ~13,000 clips combined, it finishes in well under an hour):
```
from src.voxguard.features.compose import extract_and_cache_prosody
from src.voxguard.utils.metadata import load_unified_metadata
extract_and_cache_prosody(load_unified_metadata(['wavefake']), 'models/embeddings/prosody_wavefake.npy')
extract_and_cache_prosody(load_unified_metadata(['in_the_wild']), 'models/embeddings/prosody_in_the_wild.npy')
```
Skip this entirely if Phase 2 selected the baseline (embedding-only) variant — `use_prosody=False` throughout and neither cache file is ever read.

---

## Build Prompts

### Prompt 3.1 — Zero-shot cross-dataset evaluation

```
Implement src/voxguard/classifier/cross_eval.py with a function
zero_shot_eval(detector: VoxGuardDetector, metadata_df: pd.DataFrame) -> dict that runs
detector.predict on every file in metadata_df (WaveFake or In-the-Wild), compares against the
"label" column, and returns the same metrics dict shape as Phase 2's evaluate_classifier
(accuracy, roc_auc, eer where computable — note EER needs continuous scores which
predict()'s probability_synthetic already provides). Handle failures on individual files
gracefully (log and skip, don't crash the whole eval on one corrupt file) and report the skip
count in the returned dict.
Write scripts/run_cross_eval.py as a CLI script that runs this on WaveFake and In-the-Wild
separately, and appends both rows to models/reports/cross_dataset_report.csv alongside the
original ASVspoof2019 eval row from Phase 2, so the three are comparable in one table.
```

**v4 addition to Prompt 3.1:** the four-way comparison Prompt 3.4 needs (wav2vec2-only, WavLM-only, concatenated ensemble, weighted-average ensemble) means a single-backbone cache function isn't enough — implement two functions in this file, covering all four variants between them:

1. `resolve_cache_path(model_name: str, dataset: str, split: str = "eval") -> str` — a small shared helper both functions below use: if `dataset == "asvspoof2019"`, return `models/embeddings/{model_name}_{split}.npy` (Phase 2's original split-based cache, `split` defaulting to `"eval"` since that's what cross-dataset comparison needs); otherwise return `models/embeddings/{model_name}_{dataset}.npy` (Prompt 2.2's v4 dataset-based cache). This one function is what lets everything below work identically for all three datasets in Prompt 3.4's table, including the ASVspoof2019 eval column, without duplicating path logic three times.

2. `zero_shot_eval_from_cache(classifier_path: str, model_names: list[str], dataset: str, use_prosody: bool = False, split: str = "eval") -> dict` — resolves a cache path per `model_names` entry via `resolve_cache_path`, and, if `use_prosody`, ALSO resolves a prosody cache path the same way — `resolve_cache_path`'s rule applies equally to prosody caches: `models/embeddings/prosody_{split}.npy` for `dataset == "asvspoof2019"` (Prompt 2.4's v4 naming convention), or `models/embeddings/prosody_{dataset}.npy` otherwise (this phase's prosody note above). Then:
   - if `len(model_names) == 1` and `use_prosody` is False: loads directly via `load_cached_embeddings` (Phase 2, Prompt 2.2)
   - otherwise: builds the list of resolved paths (one per backbone in `model_names`, plus the resolved prosody path if `use_prosody`) and calls `load_combined_features` (Phase 2, Prompt 2.4) to get the aligned, concatenated matrix
   - loads the classifier via `load_classifier` (Phase 2, Prompt 2.5), runs `predict_proba`/forward pass on the whole matrix in one vectorized call, and returns the same metrics dict shape `zero_shot_eval` returns.
   This ONE function covers three of Prompt 3.4's four variants just by varying `model_names`: `["wav2vec2"]` for (a), `["wavlm"]` for (b), `["wav2vec2", "wavlm"]` for (c) — reusing `load_combined_features` for the concatenation exactly the way Prompt 3.3's `extract_dual_embeddings` does it live, just from cache.

3. `zero_shot_eval_weighted_average_from_cache(classifier_a_path: str, model_a: str, classifier_b_path: str, model_b: str, dataset: str, weight_a: float = 0.5, use_prosody: bool = False, split: str = "eval") -> dict` — covers variant (d). Loads model_a's and model_b's cached embeddings SEPARATELY (each via `resolve_cache_path` + `load_cached_embeddings`, plus each one's own prosody-augmented version via `load_combined_features` if `use_prosody` is set — do NOT concatenate the two backbones together here, each classifier scores its own feature space). Validate the two manifests share the same filepaths in the same row order (reuse `load_combined_features`'s alignment check for this even though you discard its concatenated matrix — call it once, purely to validate and to get the shared label column). Run each classifier's `predict_proba` separately, combine per-row via `weighted_average_ensemble(prob_a, prob_b, weight_a)` (Prompt 3.3), and return the same metrics dict shape.

Update `scripts/run_cross_eval.py` to use `zero_shot_eval_from_cache(classifier_path, [model_name], dataset)` **by default** for its single-model comparison, falling back to the original file-by-file `zero_shot_eval` only if the relevant cache files are missing — print a clear, loud warning when that fallback triggers, so a slow multi-hour run is never a silent surprise.

### Prompt 3.2 — Second embedding extractor (WavLM)

```
Confirm src/voxguard/embeddings/extractor.py's EmbeddingExtractor already supports
model_name="microsoft/wavlm-base-plus" via the branch added in Phase 2 Prompt 2.1. If it
doesn't cleanly support both models yet, refactor it now so EmbeddingExtractor(model_name=...)
is a single class handling both wav2vec2 and WavLM (both expose the same
last_hidden_state-based pooling interface in transformers, so this should not require separate
classes). Add a unit test confirming both model names load without error and produce
same-shaped output for the same input audio.
```

**v4 note on Prompt 3.2:** the unit test here is still the right thing to write, but the actual full WavLM extraction it confirms the class is ready for already happened in Phase 2's Master Kaggle Session — there's no separate WavLM Kaggle run to do in this phase under v4.

**v4 fix — train the actual WavLM-only classifier (closes a gap in v3's original spec):** Prompt 3.4 below expects "(b) the WavLM-only classifier trained the same way" as (a), but nothing in v3 — not this prompt, not Prompt 3.3 (which trains the *concatenated* ensemble, not a standalone WavLM classifier) — ever actually specifies training one. Add this step here: reuse `train_logistic_regression`/`train_mlp` (Phase 2, Prompt 2.5) on the cached `wavlm_train.npy`/`wavlm_dev.npy` embeddings (concatenated with the prosody cache too, if Phase 2 selected the prosody-augmented variant — same `load_combined_features` pattern Phase 2 used, just swapping which embedding cache goes in), and save as `models/classifiers/wavlm_logreg.joblib` / `wavlm_mlp.joblib` (mirroring Phase 2's `baseline_logreg.joblib` naming, one backbone swapped for the other). Train only the ONE configuration matching whatever Phase 2's comparison actually selected — both the feature-set (baseline vs. prosody-augmented) AND the classifier type (logreg vs. MLP) — not a fresh four-way sweep; this keeps the wav2vec2-vs-WavLM comparison in Prompt 3.4 an apples-to-apples swap of backbone only, everything else held constant. This is what `zero_shot_eval_from_cache(wavlm_classifier_path, ["wavlm"], dataset)` and `zero_shot_eval_weighted_average_from_cache`'s `classifier_b_path` (Prompt 3.1, v4) actually point at.

### Prompt 3.3 — Ensembling

```
Implement src/voxguard/classifier/ensemble.py with:
- extract_dual_embeddings(waveform, sr, wav2vec2_extractor, wavlm_extractor) -> np.ndarray:
  concatenates both models' pooled embeddings into a single vector (768+768=1536-dim for the
  base models).
- A class EnsembleDetector mirroring VoxGuardDetector's predict() AND predict_waveform()
  interface (Phase 2, Prompt 2.7 — the streaming engine in Phase 5 needs this method on whatever
  detector it wraps, ensemble or not) but internally extracting and concatenating both embeddings
  before calling a classifier trained on the concatenated feature space. Like VoxGuardDetector,
  accept a use_prosody flag (auto-detected from the classifier's metadata sidecar the same way)
  and, if active, also concatenate the ProsodyFeatureExtractor vector (Phase 2, Prompt 2.3) via
  the SAME src/voxguard/features/compose.py utility Phase 2 introduced — do not write a second,
  separate concatenation path here.
Write scripts/train_ensemble_classifier.py that:
1. Extracts and caches WavLM embeddings for ASVspoof2019 train/dev (reuse Phase 2's caching
   utility with model="wavlm" — run the actual full-dataset extraction on the same Kaggle GPU
   notebook setup from Phase 2, same workflow, just `--model wavlm` this time; don't rebuild
   the Kaggle setup from scratch)
2. Builds concatenated train features via load_combined_features([wav2vec2_path, wavlm_path]
   if Phase 2 selected the baseline variant, or [wav2vec2_path, wavlm_path, prosody_path] if
   Phase 2 selected prosody-augmented — reusing Phase 2's cached prosody features directly, no
   need to re-extract them
3. Trains a new classifier head (reuse train_logistic_regression / train_mlp from Phase 2) on
   the concatenated features
4. Saves it as models/classifiers/ensemble_logreg.joblib (or .pt) with its metadata sidecar
Also add a simpler ALTERNATIVE ensembling function
weighted_average_ensemble(prob_a: float, prob_b: float, weight_a=0.5) -> float that just
averages the two single-model classifiers' output probabilities, as a lighter-weight fallback if
the concatenated-feature classifier doesn't clearly outperform it — implement both, the eval in
Prompt 3.4 will decide which one you keep for the final app.
```

**v4 note on Prompt 3.3:** step 1 above ("Extracts and caches WavLM embeddings for ASVspoof2019 train/dev... run the actual full-dataset extraction on the same Kaggle GPU notebook setup from Phase 2") was v3's instruction to do that extraction now, in this phase. Under v4, don't — that extraction already happened in Phase 2's Master Kaggle Session (see Phase 1's Master Strategy and Phase 2's Prerequisites). Replace step 1 with: load the already-cached `wav2vec2_train.npy`/`wavlm_train.npy` (and `_dev` equivalents) via `load_cached_embeddings`, and skip straight to step 2. If those files aren't present locally, that's a sign Phase 2's session didn't fully complete — go back and finish it via a short follow-up Kaggle session rather than re-triggering a fresh extraction pass here. `extract_dual_embeddings` as specified above is still correct and still what `EnsembleDetector.predict()`/`predict_waveform()` use for live single-file inference (Phase 5 onward — unchanged). For OFFLINE batch evaluation across WaveFake/In-the-Wild (Prompt 3.4), load both backbones' cached arrays directly via `load_combined_features` instead of calling `extract_dual_embeddings` in a per-file loop — numerically identical (both are just concatenation), but the cached path is what makes Prompt 3.4's four-model sweep fast rather than re-running extraction four times over.

### Prompt 3.4 — Before/after report generator

```
Implement src/voxguard/reports/generalization_report.py with a function
build_generalization_report() that runs zero_shot_eval (Prompt 3.1) using: (a) the wav2vec2-only
classifier from Phase 2, (b) the WavLM-only classifier trained the same way, (c) the concatenated
ensemble classifier, and (d) the weighted-average ensemble — across all three datasets
(ASVspoof2019 eval, WaveFake, In-the-Wild) — and produces a single markdown table (rows =
dataset, columns = model variant, cell = accuracy/EER) saved to
models/reports/generalization_before_after.md. This table is your generalization evidence for
the judges — make column headers and the EER/accuracy formatting clean enough to paste directly
into a slide.
```

**v4 note on Prompt 3.4:** build all twelve cells (4 variants × 3 datasets) from cache — nothing in this report needs live audio or a GPU. Concretely:
- (a) wav2vec2-only: `zero_shot_eval_from_cache(wav2vec2_classifier_path, ["wav2vec2"], dataset)`
- (b) WavLM-only: `zero_shot_eval_from_cache(wavlm_classifier_path, ["wavlm"], dataset)`
- (c) concatenated ensemble: `zero_shot_eval_from_cache(ensemble_classifier_path, ["wav2vec2", "wavlm"], dataset)`
- (d) weighted-average ensemble: `zero_shot_eval_weighted_average_from_cache(wav2vec2_classifier_path, "wav2vec2", wavlm_classifier_path, "wavlm", dataset)`

— called once per `dataset` in `["asvspoof2019", "wavefake", "in_the_wild"]`, with `use_prosody` set consistently with whichever variant Phase 2/3 selected. `resolve_cache_path` (Prompt 3.1, v4) handles the ASVspoof2019-vs-cross-dataset naming difference transparently, so the ASVspoof2019 eval column runs through the exact same four calls as the other two columns — no separate code path needed for it. All twelve cells should complete in well under a minute total, since every embedding involved was already extracted in Phase 2's Master Kaggle Session. If generation is taking more than a few minutes, a cache file is missing and it's silently falling back to a slow per-file path — check for the warning Prompt 3.1's fallback prints rather than assuming the model itself is just slow.

**v4 fix — what happens if variant (d) actually wins (closes a gap in v3's original spec):** `VoxGuardDetector` (Phase 2) and `EnsembleDetector` (Prompt 3.3, for the concatenated variant) both exist for live single-file inference, but nothing anywhere defines an equivalent for the weighted-average variant — if this table's winner turns out to be (d), Phase 5 has nothing to wrap. Decide this NOW, not when Phase 5 needs it: if (d) wins, either (i) implement a small `WeightedAverageDetector` mirroring `EnsembleDetector`'s `predict()`/`predict_waveform()` interface exactly, but running both extractors independently and combining via `weighted_average_ensemble` instead of concatenating features, or (ii) treat this table as evidence only and ship whichever of (a)/(b)/(c) is close behind for production, since a real detector class already exists for those. Document whichever choice you make in this comparison's write-up — don't leave it implicit. In practice this is a low-probability path: concatenated ensembles (c) usually match or beat weighted-averaging (d) in accuracy without the added inference-time complexity of two backbones scored separately, so if the table is close, (c) is the simpler choice already backed by a working detector class.

---

## Tests

```
python -c "
# v4: train the standalone WavLM-only classifier (Prompt 3.2's v4 fix) — Prompt 3.4's variant
# (b) and the weighted-average function both need this to exist before they can run
from src.voxguard.embeddings.cache import load_cached_embeddings
from src.voxguard.classifier.head import train_logistic_regression
from src.voxguard.classifier.head import save_classifier
X_train, meta_train = load_cached_embeddings('models/embeddings/wavlm_train.npy')
clf = train_logistic_regression(X_train, meta_train['label'])
save_classifier(clf, 'models/classifiers/wavlm_logreg.joblib')
print('WavLM-only classifier trained and saved')
"
python scripts/run_cross_eval.py
python scripts/train_ensemble_classifier.py
python -c "
from src.voxguard.reports.generalization_report import build_generalization_report
build_generalization_report()
print(open('models/reports/generalization_before_after.md').read())
"
python -m pytest tests/ -q -k ensemble
# v4: confirm the cache-based path is actually being used — should complete in a couple of
# seconds, not minutes. If this hangs, a Phase 2 Kaggle extraction step was likely skipped or
# a cache file is misnamed.
python -c "
import time
from src.voxguard.classifier.cross_eval import zero_shot_eval_from_cache
t0 = time.time()
result = zero_shot_eval_from_cache('models/classifiers/baseline_logreg.joblib', ['wav2vec2'], 'wavefake', use_prosody=False)
print(result, f'{time.time()-t0:.2f}s')
"
```

What to check in the resulting table:
- [ ] The wav2vec2-only classifier's accuracy/EER on WaveFake and In-the-Wild is measurably worse than on ASVspoof2019 eval (this drop is expected and is the whole point of this phase — don't be alarmed by it, document it)
- [ ] At least one ensembling approach (concatenated or weighted-average) improves accuracy or EER on at least one cross-dataset benchmark versus the single-model baseline
- [ ] The ensemble's performance on the original ASVspoof2019 eval set hasn't badly regressed compared to Phase 2's single-model result

## Definition of Done Checklist

- [ ] Zero-shot eval implemented and run on both WaveFake and In-the-Wild
- [ ] WavLM embedding extraction confirmed working through the same `EmbeddingExtractor` class
- [ ] (v4) Standalone WavLM-only classifier trained and saved (`wavlm_logreg.joblib`/`wavlm_mlp.joblib`, Prompt 3.2's v4 fix) — not just the concatenated ensemble; Prompt 3.4's variant (b) and the weighted-average function both depend on this existing
- [ ] Both ensembling strategies (concatenated-feature and weighted-average) implemented
- [ ] Before/after generalization report generated as a clean markdown table
- [ ] Best-performing detector variant selected and documented for use in later phases (Phase 5 onward should import whichever detector — `VoxGuardDetector` or `EnsembleDetector` — wins this comparison; if the weighted-average variant wins instead, see the v4 fix on Prompt 3.4 before proceeding — no detector class exists for it out of the box)
- [ ] (v4) `zero_shot_eval_from_cache` and `zero_shot_eval_weighted_average_from_cache` implemented and confirmed fast (seconds, not minutes/hours) against the embeddings cached in Phase 2 — confirm all four Prompt 3.4 variants (wav2vec2-only, WavLM-only, concatenated, weighted-average) actually produce a result for all three datasets, not just ASVspoof2019
- [ ] (v4) If Phase 2 selected the prosody-augmented variant: prosody features extracted locally for the WaveFake and In-the-Wild subsets (`prosody_wavefake.npy`, `prosody_in_the_wild.npy`) before running the cache-based eval — confirmed this step was skipped cleanly (not silently broken) if the baseline variant was selected instead
- [ ] (v4) Confirmed this phase needed zero new Kaggle GPU time (or, if some was needed, documented specifically why the Phase 2 consolidation didn't fully cover it)

## Common Pitfalls

- Don't cherry-pick which dataset to report if the ensemble doesn't win everywhere — report all three honestly. A judge who asks "did it help on all of them?" and gets a straight "no, here's where it didn't" answer is more convincing than a slide that quietly omits a bad result.
- WaveFake's and In-the-Wild's audio formats/sample rates can differ from ASVspoof2019's — confirm Phase 1's preprocessing was actually applied to both before blaming the model for a generalization gap that's really a preprocessing bug.
- You already have Phase 2's wav2vec2 AND (v4) WavLM embeddings downloaded locally for all three dataset scopes — don't burn Kaggle GPU-hours re-extracting any of them here. Under v4 this phase should need no Kaggle time at all.
- (v4) If `run_cross_eval.py` or `build_generalization_report()` seems to hang or take unexpectedly long, it's very likely silently using the slow per-file fallback because a cache file from Phase 2's session is missing or misnamed — check the printed warning rather than assuming the model itself is just slow.
- (v4) `load_combined_features`'s manifest-alignment check (Phase 2, Prompt 2.4) only catches a misalignment if the two caches actually have DIFFERENT filepath orderings — it can't catch a case where both caches happen to be wrong in the same way. Extract wav2vec2 and WavLM (and, if prosody-augmented, the prosody cache) for the SAME dataset scope from the SAME unmodified metadata DataFrame, in the same session, without an intervening resample/shuffle/subset step — this is exactly the "single most dangerous silent bug" already called out in Phase 2's Common Pitfalls, now applying equally to concatenating across dataset scopes here, not just across feature types there.

---
# Phase 4 — Code-Switched Hindi/Hinglish Test & Training Track

**Maps to:** Must-Have Feature 3
**Estimated time:** ~9-10 hours (revised up from an earlier 6-hour estimate — see the framing note below for why)
**Depends on:** Phase 2 complete, ideally Phase 3 too (so the embedding config used here matches whichever variant Phase 3 crowned as best)

---

## Objective

No public benchmark covers code-switched Hindi/Hinglish cloning detection, so this phase builds one from nothing: record real Hindi/Hinglish speech, generate matched synthetic clones with XTTS-v2, and — unlike a pure generalization-only approach — actually retrain the classifier head on part of this data so the shipped detector genuinely works on Hindi/Hinglish, not just as a zero-shot curiosity.

## A Framing Note: Train vs. Eval, and Why It Matters Here

It would be simpler to treat this whole track as held-out eval only (feed it through the Phase 2/3 classifier zero-shot and report the accuracy drop). That's a clean, defensible generalization claim, and it's what an earlier version of this phase did. But the more useful outcome — a detector that actually catches Hindi/Hinglish clones well — requires training on some of this data. Both are legitimate; they just answer different questions, and conflating them is the one mistake to actively avoid here:

- **Zero-shot only:** "does an English-trained classifier generalize to Hindi/Hinglish at all?"
- **Trained on part of this data (this phase's approach):** "can we build a classifier that actually detects Hindi/Hinglish clones well?"

The fix is standard: **split your self-built dataset into train and eval before touching anything (Prompt 4.6), train only on the train portion, and keep the eval portion completely untouched until final evaluation.** You then get to report both numbers side by side — your trained model's real capability on held-out Hindi/Hinglish, and (for comparison) the original English-only baseline's zero-shot performance on that same held-out set. That comparison is a stronger result than either number alone, and it's worth stating explicitly in your submission and to judges.

## Prerequisites

- Phase 2 (and ideally Phase 3) complete
- Coqui XTTS-v2 installable (`TTS` package, already in `requirements.txt` from Phase 0) — CPML license, free for personal/research/non-commercial use; flag if this project ever goes commercial post-SIH
- AI4Bharat Indic TTS identified as a fallback if XTTS-v2 setup stalls
- One or more consenting speakers (yourself, and ideally 2+ teammates/friends) — **do not record or clone anyone who hasn't explicitly agreed to it**, see the consent template below
- (v4 note: this phase's entirely-local workflow is unaffected by the v4 Kaggle consolidation described in Phase 1 — nothing below changes.)

## Local Hardware Note

XTTS-v2 clone generation (Prompt 4.3) runs entirely on CPU here — there's no local GPU to target, so this is the slowest local step in the project. It's still practical: for ~150-300 clips (the recommended tier), treat it as a background batch job (kick it off, work on something else, check back), not something to babysit. It does not need Kaggle — XTTS-v2's model weights and inference don't benefit enough from a short-lived notebook session to justify the upload/download overhead, and running it locally keeps voice recordings and clones off a third-party service by default. Prompt 4.7's embedding extraction, by contrast, is small enough (only your Hindi train split, a few dozen to ~150 clips) to run locally on CPU directly too — this is the one embedding-extraction step in the whole guide that does NOT need Kaggle, precisely because it's operating on your self-collected data, not the full ASVspoof2019 scale.

---

## Planning: How Much Data You Actually Need

You will not match ASVspoof2019's scale, and you don't need to. A judge respects "75 real clips, 75 matched clones, 3 speakers, fully documented" far more than an inflated or vague number.

| Tier | Speakers | Sentences per speaker | Real clips | Matched synthetic clips | Total clips | Notes |
|---|---|---|---|---|---|---|
| Minimum viable | 1 (you) | 25 (script below) | 25 | 25 | 50 | Works, but zero speaker diversity |
| **Recommended** | 3 (you + 2 consenting others) | 25 each | 75 | 75 | 150 | Enough to hold one full speaker out for a genuine speaker-level eval split |
| Stretch (only if time allows) | 4-5 | 25 each, plus a 2nd take of 10 sentences each | 120-150 | 120-150 | 240-300 | Chase this only after everything else in the project works |

**Recommendation: aim for the 3-speaker tier.** It's the smallest size where a genuine speaker-held-out eval split (Prompt 4.6) is possible, which matters more for a credible result than raw clip count.

## Recording Setup

No studio needed — consistency matters more than gear.

- **Equipment:** phone voice recorder (Android Recorder, iOS Voice Memos) or laptop + Audacity, either is fine
- **Environment:** a quiet room with soft furnishings (curtains, couch, bed) has far less echo than bare tile/concrete
- **Mic distance:** ~15-20cm, kept consistent across all takes by the same speaker
- **Format:** record at your device's highest available quality (typically 44.1kHz/16-bit+); you'll downsample to 16kHz mono during preprocessing, but can't recover quality you didn't capture
- **One sentence = one file.** Don't record one long continuous take — individual takes are faster to redo when you stumble and match ASVspoof2019's utterance-level format
- Record one extra **6-10 second natural reference clip** per speaker, separate from the script — this is what XTTS-v2 clones from later, and doesn't need to be scripted, just clean continuous speech

**File naming convention:** `{speaker_id}_{category}_{sentence_id}.wav` (e.g. `priya_neutral_03.wav`, `priya_scam_11.wav`) — set this now, it saves a cleanup pass later.

**Per-speaker checklist:**
- [ ] Quiet room confirmed (do a 5-second silent test recording, listen for hum/echo before real takes)
- [ ] Consistent mic distance for every sentence
- [ ] Natural conversational pace — read each sentence like you're actually saying it, code-switch included, not in a slow over-enunciated "reading voice" (an unnaturally careful reading style is itself a cue a classifier can latch onto that has nothing to do with real-vs-synthetic)
- [ ] Re-record any take with a stumble, cough, or background interruption
- [ ] One 6-10 second natural reference clip recorded separately

## What To Say: The Reading Script

25 sentences across three categories, written as natural code-switched Hinglish (the way people actually speak it, not formal transliterated Hindi). Every speaker reads all 25. The categories are chosen to double as usable content for **Phase 9's red-flag keyword scanner**, so recording this script once serves both this phase and that one.

**Category A — Neutral/Casual Conversation (01-10):** ordinary code-switched talk, no scam content — your clean everyday-speech coverage and a clean negative control for the keyword scanner.

1. Yaar, aaj office mein bahut kaam tha, I'm so tired now.
2. Mummy ne bola ki dinner ready hai, chalo khaane baith jaate hain.
3. Kal weekend hai na, let's plan a trip to the hills.
4. Mera phone ka battery bahut fast drain ho raha hai these days.
5. Traffic itna zyada tha ki main meeting ke liye late ho gaya.
6. Tumne wo new web series dekhi kya, it's actually really good.
7. Is mahine ka budget thoda tight hai, we need to cut down on eating out.
8. Doctor ne kaha hai ki mujhe zyada paani peena chahiye, and exercise daily.
9. Weather bahut accha hai aaj, chalo evening walk pe chalte hain.
10. Project deadline agle Friday hai, I think we can manage it easily.

**Category B — Scam-Pattern Speech (11-20):** mirrors real voice-scam patterns (urgency, financial-action requests, authority impersonation, isolation tactics) — the exact categories Phase 9's red-flag scanner looks for. A scam script read by a genuine (non-cloned) voice should trigger the keyword layer but NOT the cloning classifier — that distinction is worth demonstrating deliberately in your demo.

11. Sir, aapka bank account today block ho jaayega agar aap abhi apna OTP share nahi karte.
12. Yeh customs department se call hai, aapke parcel mein illegal items mile hain, turant fine pay kijiye.
13. Beta, main tumhari maa bol rahi hoon, mujhe abhi ke abhi paise chahiye, kisi ko mat batana.
14. Police station se bol raha hoon, aapke naam par ek warrant issue hua hai, abhi arrest ho sakta hai.
15. Aapka UPI account verify karna zaroori hai warna aapka paisa freeze ho jayega, please share your PIN now.
16. This is an urgent matter sir, agar aap abhi payment nahi karte to legal action liya jayega.
17. Aapko lottery mein paanch lakh rupaye jeete hain, bas processing fee ke liye account details bhejiye.
18. Please don't tell your family about this call, yeh confidential matter hai.
19. Aapka credit card suspicious activity ke wajah se block kar diya gaya hai, verify karne ke liye apna CVV batayein.
20. Abhi turant is number par paise transfer kijiye warna aapki service disconnect ho jayegi.

**Category C — Neutral Phone-Call Style (21-25):** legitimate-sounding calls (bank notifications, delivery updates, appointment reminders, family) using similar topics to Category B without scam intent — the hardest negative controls for the keyword scanner, so it doesn't just learn to flag any mention of banking or money.

21. Hello, main XYZ bank se bol raha hoon, aapka statement email par bhej diya gaya hai.
22. Aapka order successfully deliver ho gaya hai, kripya feedback share kijiye.
23. Beta, ghar kab aa rahe ho, dinner ready hai.
24. Hi, this is a reminder call for your appointment tomorrow at 10 AM.
25. Aapka recharge successful ho gaya hai, thank you for using our services.

**Optional bonus (only with spare time):** 60-90 seconds of unscripted free speech per speaker, sliced into 3-4 second segments during preprocessing — diversifies the real-speech distribution beyond "person reading a script," since read speech has different prosody than spontaneous speech.

## Consent

Get explicit written consent before recording or cloning anyone besides yourself — a message thread is enough for a hackathon, but get it in writing.

> Hi [name], for my SIH project (VoxGuard, a voice-cloning detector) I'd like to record a few short sentences in your voice, and — separately — use one of those recordings as a reference to generate an AI-cloned version of your voice, purely for testing whether my detector can tell the difference. This is for a hackathon prototype only, not for any commercial use, and the recordings/clones will only be used within this project. Are you okay with that? Let me know if you'd rather not have your voice cloned but are fine with just the real recordings — either is completely fine.

If someone consents to recordings but not cloning, use their real clips as extra "real" data only — don't clone them. Document who consented to what in the dataset card (Prompt 4.5).

---

## Build Prompts

### Prompt 4.1 — Ingest and organize real recordings

```
Write scripts/organize_hindi_recordings.py that:
1. Accepts a --input_dir of manually-placed raw recordings following the
   {speaker_id}_{category}_{sentence_id}.wav naming convention above.
2. Converts each file to 16kHz mono WAV using the existing audio_io helpers (reuse
   src/voxguard/utils/audio_io.py from Phase 0, don't rewrite audio loading logic), saving
   results to data/raw/hindi_hinglish/real/.
3. Builds data/metadata/hindi_hinglish_real.csv with columns:
   [filepath, speaker_id, category (neutral/scam/control), sentence_id, sentence_text
   (looked up from a SENTENCES dict you define in this script matching the 25-sentence list
   above), consent_confirmed (bool, defaulting to False — force the user to explicitly flip this
   to True per speaker after confirming consent, rather than assuming it)].
4. Prints a summary: total clips per speaker, per category, and total duration, plus a warning
   listing any speaker with consent_confirmed=False so nothing gets used accidentally without
   consent on record.
```

### Prompt 4.2 — Reference clip preparation

```
Write scripts/prepare_xtts_references.py that, given data/metadata/hindi_hinglish_real.csv and
a --reference_dir of the separate 6-10 second natural reference clips recorded per speaker,
copies/validates each reference clip: confirm duration is 5-15 seconds (warn and skip if
shorter — XTTS-v2 clones poorly from very short references), confirm it's clean (no long
silences at start/end — auto-trim using audio_io if needed), and save a manifest
data/metadata/xtts_references.csv mapping speaker_id -> validated reference clip path.
```

### Prompt 4.3 — XTTS-v2 setup and batch clone generation

```
Implement src/voxguard/synth/xtts_clone.py with a function
clone_voice(reference_audio_path: str, text: str, language="hi", output_path: str = None) ->
str (returns output path) using the Coqui TTS library's XTTS-v2 model
(TTS.api.TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False) — explicitly gpu=False
since this project has no local CUDA GPU to target; the TTS library falls back to CPU cleanly
with this flag, no special handling needed beyond setting it explicitly rather than letting it
autodetect). Add a fallback stub clone_voice_indic_tts(...) with a NotImplementedError and a
comment pointing to AI4Bharat's Indic TTS repo if XTTS-v2 setup proves unworkable within a
couple hours.

Then implement scripts/generate_hindi_clones.py that:
1. Loads xtts_references.csv (Prompt 4.2) and hindi_hinglish_real.csv (Prompt 4.1).
2. For every speaker with a validated reference clip and consent_confirmed=True, and for every
   sentence in the 25-sentence script, calls clone_voice() with language="hi", the speaker's
   reference clip, and that sentence's text, saving output to
   data/raw/hindi_hinglish/synthetic/{speaker_id}_{category}_{sentence_id}_clone.wav. This
   produces a MATCHED pair for every real clip — same speaker, same sentence, real vs
   synthetic — which is the controlled comparison you want for both training and eval.
3. Builds data/metadata/hindi_hinglish_synthetic.csv with columns [filepath, speaker_id
   (source speaker cloned), category, sentence_id, label="synthetic", generator="xtts_v2"].
4. Logs any generation failures per speaker/sentence without killing the whole batch, and
   prints a final count of successfully generated pairs.
```

**A known rough edge:** XTTS-v2's Hindi mode can mispronounce or badly render the English words embedded in a Hinglish sentence. This is expected, not a script bug.
- [ ] Listen to at least 5 synthetic clips per speaker before trusting the batch
- [ ] If English-word pronunciation is consistently bad, try transliterating just those words phonetically into Hindi-sounding spelling (e.g. "ऑफिस"/"aufis" instead of "office") as a debugging step, and note in the dataset card if you did this
- [ ] If quality is still imperfect after that, keep the audio anyway — a slightly awkward clone is still a legitimate positive example. Don't cherry-pick only the best-sounding clones; that biases eval toward easy cases
- [ ] Run the batch generation with the laptop plugged in and somewhere with airflow — sustained CPU load across ~150-300 clips can trigger thermal throttling on a thin laptop chassis partway through, meaning clip #40 may generate slower than clip #5. Not a bug, just don't be surprised by it

### Prompt 4.4 — Merge into unified track metadata

```
Merge hindi_hinglish_real.csv and hindi_hinglish_synthetic.csv into
data/metadata/hindi_hinglish_track.csv with the project-wide unified schema:
[filepath, label ("real"/"synthetic"), speaker_id, category, sentence_id, dataset="hindi_hinglish"].
Print final totals: real clips, synthetic clips, unique speakers, and confirm real/synthetic
counts are roughly 1:1 — a large imbalance usually means some clone generations failed
silently, check Prompt 4.3's failure log.
```

### Prompt 4.5 — Dataset card

```
Write data/metadata/HINDI_HINGLISH_DATASET_CARD.md documenting: number of speakers and who
consented to what (recordings only vs recordings+cloning — "Speaker A, recordings only" is fine
if you'd rather not name individuals), total real/synthetic clip counts, total duration, the
train/eval split decision and rationale (Prompt 4.6), the XTTS-v2 CPML license note, and the
Prompt 4.3 pronunciation-quality caveat. This is the document you hand a judge who asks "how was
this built" — and the place where the train-vs-eval honesty note from the top of this phase
should be spelled out clearly.
```

### Prompt 4.6 — Train/eval split

```
Implement src/voxguard/utils/hindi_splits.py with a function
get_hindi_hinglish_splits(track_df, mode="speaker_holdout", holdout_speaker=None) -> (train_df,
eval_df):
- mode="speaker_holdout": requires holdout_speaker to be specified (the speaker_id to fully
  exclude from train, used for eval only); raises a clear error if track_df has fewer than 2
  unique speakers. Use this mode if you recorded 3+ speakers (the recommended tier) — it gives
  a genuine "does it generalize to an unseen voice" test.
- mode="utterance_stratified": performs an 80/20 stratified split by category (sklearn's
  train_test_split with stratify=track_df['category']), for the 1-2 speaker fallback case. Note
  in the docstring that this mode's eval number reflects "detects clones of a partially-seen
  voice," not generalization to a new speaker — state this limitation directly in the dataset
  card if this is the mode you end up using.
Print which mode was used and the resulting split sizes whenever called, so the honesty
tradeoff is always visible in the logs.
```

### Prompt 4.7 — Extract embeddings for the Hindi/Hinglish train split

```
Reuse src/voxguard/embeddings/cache.py's extract_and_cache function (Phase 2) to extract
embeddings for the Hindi/Hinglish train split produced by Prompt 4.6, using the SAME
EmbeddingExtractor configuration (same model_name — whichever Phase 3 crowned as best) already
used for ASVspoof2019, so the feature spaces are directly comparable. Save to
models/embeddings/{model}_hindi_train.npy following Phase 2's existing naming convention. Run
this one LOCALLY on CPU — unlike Phase 2/3's full ASVspoof2019-scale extraction, this is only a
few dozen to ~150 clips (your Hindi train split), which finishes in a couple of minutes on CPU.
Kaggle's upload/download round-trip isn't worth it at this size.

If Phase 2's comparison (Prompt 2.6) selected the prosody-augmented variant, ALSO run Phase 2's
extract_and_cache_prosody (Prompt 2.4) over this same Hindi train split — same reasoning, small
enough to stay local — and use load_combined_features to build the Hindi train feature matrix
consistently with whatever feature set Phase 2/3 decided on. Getting this wrong (e.g., training
Prompt 4.8's variants on embedding-only Hindi features while the production classifier expects
embedding+prosody input) would silently break VoxGuardDetector's auto-detection from Phase 2,
Prompt 2.7 — the dimensions simply wouldn't match, and it's worth confirming this explicitly
rather than assuming it lines up.
```

**v4 fix — closes a gap in v3's original spec for one specific case:** the prompt above says "model_name — whichever Phase 3 crowned as best," which implicitly assumes the winner is a single backbone (wav2vec2 or WavLM alone). If Phase 3's comparison (Prompt 3.4) instead crowned the **concatenated ensemble**, a single `EmbeddingExtractor` call can't produce that feature space — extract BOTH `wav2vec2_hindi_train.npy` and `wavlm_hindi_train.npy` (two calls, same pattern as above, both still local/CPU/fast at this size) and build the Hindi train feature matrix via `load_combined_features([wav2vec2_hindi_train_path, wavlm_hindi_train_path], ...)`, adding the prosody path too if prosody-augmented — exactly mirroring how Phase 3's `train_ensemble_classifier.py` built the ASVspoof2019 side of this same feature space, just applied to the Hindi split. If Phase 3 instead crowned the **weighted-average ensemble**, there is no single combined feature space to build here — Prompt 4.8's row-wise combine needs to happen separately for each backbone (train a Hindi-augmented head for wav2vec2 alone AND one for WavLM alone, then continue applying the same weighted average at inference time), so extract and train against both backbones independently rather than picking one arbitrarily. In the common case (Phase 3 crowned a single-backbone variant), nothing changes from the original prompt above.

### Prompt 4.8 — Train both classifier variants

```
Write scripts/train_hindi_augmented_classifier.py that:
1. Loads cached ASVspoof2019 train embeddings (Phase 2) and the new Hindi train embeddings
   (Prompt 4.7) — both already built using the SAME feature set decided in Phase 2/3 (baseline
   embedding-only, or embedding+prosody — whichever it was, both sides of this row-wise combine
   must match, or the concatenation below will silently produce a feature matrix that doesn't
   correspond to anything meaningful).
2. Builds Variant A's training set (COMBINED — the recommended, primary variant) by
   concatenating both embedding matrices and label vectors row-wise (np.concatenate along
   axis 0 — same feature dimensionality, so this is a simple row-wise combine, NOT the
   feature-axis concatenation from Phase 3's ensembling — don't confuse the two).
3. Builds Variant B's training set (HINDI-ONLY, an ablation/comparison) from the Hindi train
   embeddings alone.
4. Trains both using the existing train_logistic_regression and train_mlp functions from
   Phase 2's src/voxguard/classifier/head.py (reuse, don't reimplement).
5. Because the Hindi training set is small, wrap training in a StratifiedKFold(n_splits=5)
   cross-validation loop for Variant B specifically (and optionally also A), reporting mean and
   standard deviation of validation accuracy across folds rather than trusting a single
   train/val split.
6. Saves both variants to models/classifiers/ as hindi_combined_{logreg,mlp}.joblib/.pt and
   hindi_only_{logreg,mlp}.joblib/.pt, clearly distinguished from Phase 2's original
   English-only baseline models so all three stay comparable.

Architecture reminder: the embedding backbone (wav2vec2/WavLM) stays frozen throughout — only
the lightweight classifier head is being retrained here, exactly as in Phase 2, just with an
expanded training set. This retrain is fast (seconds to minutes), not another multi-hour
training run.
```

### Prompt 4.9 — (Optional, time permitting) Light data augmentation

```
Implement src/voxguard/utils/augment.py with augment_waveform(waveform, sr) -> np.ndarray using
the audiomentations library (add to requirements.txt), applying ONE randomly-chosen light
transform per call from: mild Gaussian noise addition, small pitch shift (±1-2 semitones), or
small time-stretch (0.95-1.05x) — keep these subtle, simulating realistic variation like a
different phone mic, not distorting the audio into unrecognizability. Add an
--augment_multiplier flag to scripts/train_hindi_augmented_classifier.py that, if set to e.g. 3,
generates that many augmented copies of each Hindi training clip's embedding (re-extracting
embeddings on the fly per augmented waveform) before training, to grow the small training set.
Document in the dataset card that augmented copies were used for training only, never for eval.
```

### Prompt 4.10 — Full honest comparison report

```
Write scripts/evaluate_hindi_variants.py that runs evaluate_classifier (Phase 2) across the SAME
Hindi/Hinglish held-out eval split (Prompt 4.6) for three models: (1) Phase 2's original
English-only baseline classifier — this is the zero-shot number, (2) Variant A, the
combined-data head (Prompt 4.8), (3) Variant B, the Hindi-only head (Prompt 4.8). ALSO re-run
each of these three on the original ASVspoof2019 eval split (English) to confirm Variant A
hasn't meaningfully regressed on English performance versus the Phase 2 baseline — this is the
check that proves the combined approach didn't just trade English accuracy for Hindi accuracy.
Save a single markdown table to models/reports/hindi_training_comparison.md with rows = model
variant, columns = [ASVspoof2019 eval accuracy/EER, Hindi/Hinglish eval accuracy/EER], so the
zero-shot-vs-trained story is visible in one place. If Variant A does NOT clearly beat the
baseline, don't drop this section — report it as-is with a likely-cause note (usually too little
Hindi training data, or too small an eval split) in the dataset card; a well-described negative
result is still a credible engineering finding.
```

---

## Tests

```
python scripts/organize_hindi_recordings.py --input_dir <recordings_dir>
python -c "
import pandas as pd
df = pd.read_csv('data/metadata/hindi_hinglish_real.csv')
assert df['consent_confirmed'].all(), 'Found clips without confirmed consent — fix before continuing'
print(df.groupby(['speaker_id','category']).size())
"
python scripts/prepare_xtts_references.py --reference_dir <references_dir>
python -m src.voxguard.synth.xtts_clone <path_to_reference_clip.wav>   # smoke test
python scripts/generate_hindi_clones.py
python -c "
import pandas as pd, os
df = pd.read_csv('data/metadata/hindi_hinglish_track.csv')
assert df['label'].isin(['real','synthetic']).all()
assert df['filepath'].apply(os.path.exists).all()
print(df['label'].value_counts())
"
python -m pytest tests/ -q -k hindi
python scripts/train_hindi_augmented_classifier.py
python scripts/evaluate_hindi_variants.py
cat models/reports/hindi_training_comparison.md
```

What a good outcome looks like:
- Variant A (combined) clearly outperforms the English-only baseline on the Hindi/Hinglish eval split — your headline "trained on real code-switched data and it helped" result
- Variant A's ASVspoof2019 (English) performance stays close to the Phase 2 baseline — small movement either direction is normal, a big drop is not
- Variant B (Hindi-only) likely shows high variance across cross-validation folds — expected with this little data, and exactly why Variant A is the one to ship

Manual checks:
- [ ] Every real clip has a matched synthetic clone (1:1 ratio, minus documented generation failures)
- [ ] Consent confirmed for every speaker before any cloning happened
- [ ] Split mode (speaker-holdout vs utterance-stratified) matches your actual speaker count and is stated in the dataset card
- [ ] Variant A vs. English-only baseline comparison run and reported, whichever direction it comes out
- [ ] Variant A's English performance checked for regression, not assumed fine

## Definition of Done Checklist

- [ ] At least 1 speaker (ideally 3+) recorded reading all 25 sentences, with consent documented
- [ ] Matched XTTS-v2 clones generated for every consenting speaker's real clips
- [ ] Unified Hindi/Hinglish metadata built and merged
- [ ] Train/eval split implemented with an explicit, documented rationale for which mode was used
- [ ] Embeddings extracted for the Hindi train split using the same extractor config as the rest of the project — **if Phase 3 crowned an ensemble variant (concatenated or weighted-average) rather than a single backbone, confirmed the v4 fix on Prompt 4.7 was followed (both backbones extracted for Hindi, not just one)**
- [ ] Both Variant A (combined) and Variant B (Hindi-only) classifier heads trained
- [ ] Cross-validation used for the small Hindi-only training set given its size
- [ ] Full 3-way comparison (English-only baseline vs. combined vs. Hindi-only) evaluated on both the Hindi eval split and the original English eval split
- [ ] Dataset card written, including the pronunciation-quality caveat and the train/eval split honesty note
- [ ] Variant A designated as the production classifier — **downstream phases (5 onward) should wrap Variant A, not the Phase 2/3 English-only model**, since it's a strict upgrade as long as Prompt 4.10 confirms no English regression

## Common Pitfalls

- **Training and evaluating on the same Hindi clips.** The single most damaging mistake possible here — it will make Hindi numbers look great and mean nothing. Prompt 4.6's split exists specifically to prevent this; don't skip it under time pressure.
- **Cherry-picking clean-sounding clones.** Keep awkward-pronunciation clones in the dataset rather than filtering to only the best-sounding ones — a detector that only sees easy synthetic examples during training won't generalize to messier real-world clones.
- **Forgetting the English regression check.** A combined-head classifier that gains Hindi performance by quietly losing English performance isn't a win — always report both numbers together.
- **Treating a 1-2 speaker eval number as a generalization claim.** It isn't one. Have the honest answer ready if a judge asks how many speakers were held out.
- XTTS-v2's first run downloads a multi-GB model — do this well before demo day, not the night before.
- Don't route Prompt 4.3 (XTTS-v2 generation) or Prompt 4.7 (Hindi embedding extraction) through Kaggle — both are small, one-time, local-friendly jobs, and the upload/download round-trip would cost more time than it saves. Kaggle is reserved for Phase 2/3's full ASVspoof2019-scale extraction only.

---
# Phase 5 — Real-Time Chunked Streaming Engine

**Maps to:** Must-Have Feature 4
**Estimated time:** ~5 hours
**Depends on:** Phase 2 (and ideally Phase 3 and Phase 4) complete — needs a working `predict()`-style inference function

**Which detector to wrap:** if Phase 4 was completed and Prompt 4.10 confirmed no English regression, wrap **Phase 4's Variant A (combined English+Hindi classifier head)** in `StreamingScorer` below instead of the Phase 2/3 English-only model — it's a strict upgrade (handles both languages) at no cost to English performance. If Phase 4 was skipped, fall back to whichever model Phase 3 crowned as best, or Phase 2's baseline if Phase 3 was also skipped.

---

## Objective

Turn the static "one number for a whole clip" classifier into a streaming engine: chunk incoming audio into overlapping 1-2 second windows, score each chunk, and maintain a running confidence score — so the headline demo claim becomes "flagged within N seconds," not just "X% accurate."

## Prerequisites

- Phase 2/3 complete with a chosen best detector (`VoxGuardDetector` or `EnsembleDetector`) whose `predict()` is fast enough for near-real-time use on a short chunk (benchmark this before building the streaming layer around it — see Prompt 5.1's test)

## Local Hardware Note

This entire phase runs locally on CPU, no GPU involved anywhere — that's expected and fine. A single ~1.5 second chunk through a *base*-size transformer (wav2vec2/WavLM-base) is a small, single-sample forward pass; on a modern 8-core laptop CPU this typically completes well under the chunk's own duration, which is exactly the property "real-time" requires. Confirm this empirically for your specific hardware and chosen model with the benchmark below rather than assuming it — but don't expect to need any acceleration beyond the standard CPU PyTorch build already installed in Phase 0.

**If Phase 2's prosody-augmented classifier won the comparison** (Prompt 2.6) and is what you're wrapping here, re-run the benchmark below with `use_prosody=True` — `ProsodyFeatureExtractor.extract()` (Phase 2, Prompt 2.3) adds pitch tracking and onset detection on top of the transformer forward pass, and while `librosa.yin` was deliberately chosen over the slower `librosa.pyin` for exactly this reason, it's still worth confirming the combined per-chunk latency on your hardware rather than assuming the earlier embedding-only benchmark still holds.

---

## Build Prompts

### Prompt 5.1 — Streaming buffer class

```
First, in src/voxguard/config.py, set the STREAM_CHUNK_SECONDS and STREAM_OVERLAP_SECONDS
placeholders (added in Phase 0) to 1.5 and 0.5 respectively, so these live as project-wide
defaults rather than being re-declared per class.
Implement src/voxguard/streaming/buffer.py with a class StreamingBuffer:
- __init__(self, sample_rate=16000, chunk_seconds=None, overlap_seconds=None): if chunk_seconds
  or overlap_seconds is None, fall back to config.STREAM_CHUNK_SECONDS /
  config.STREAM_OVERLAP_SECONDS. Stores chunk length
  and stride (chunk_seconds - overlap_seconds) in samples, and an internal growing numpy buffer.
- push(self, audio_frame: np.ndarray) -> list[np.ndarray]: appends new audio samples to the
  internal buffer, and returns a list of any complete overlapping windows now available (there
  may be zero, one, or more depending on how much new audio arrived), advancing an internal
  read-position pointer by stride for each window emitted so windows correctly overlap rather
  than duplicate. Trim old buffer data that's no longer needed (i.e., before the earliest point
  any future window could still need) to keep memory bounded during a long call.
- reset(self): clears internal state for a new call/session.
Add a docstring with a worked example of the expected window boundaries for a simple case (e.g.,
chunk_seconds=1.0, overlap_seconds=0.5, sample_rate=10 for an easy-to-hand-verify test case).
```

### Prompt 5.2 — Per-chunk classifier wiring

```
Implement src/voxguard/streaming/scorer.py with a class StreamingScorer:
- __init__(self, detector): wraps an existing detector (VoxGuardDetector or EnsembleDetector).
  Note: predict_waveform(waveform: np.ndarray, sr: int) -> dict was already added to
  VoxGuardDetector back in Phase 2's Prompt 2.7 specifically so the streaming engine could score
  in-memory numpy arrays without writing temp files for every chunk — if you're wrapping
  EnsembleDetector (Phase 3) instead, confirm it also exposes predict_waveform with the same
  dict return shape ({"label": ..., "probability_synthetic": ...}); add it there now, matching
  Phase 2's implementation, if Phase 3 didn't already carry it forward.
- score_chunk(self, chunk: np.ndarray, sr: int) -> float: calls detector.predict_waveform(chunk,
  sr) and returns just the "probability_synthetic" value from the result dict, wrapping the
  detector call in a try/except that returns None on failure (e.g., a chunk that's pure silence)
  rather than crashing the stream.
```

### Prompt 5.3 — EMA accumulation

```
Implement src/voxguard/streaming/ema.py with a class RunningRiskScore:
- __init__(self, alpha=0.3): alpha is the EMA smoothing factor (higher = more weight on newest
  chunk).
- update(self, new_score: float) -> float: if this is the first update, set the running value to
  new_score; otherwise running = alpha * new_score + (1 - alpha) * running. Ignore None scores
  (from Prompt 5.2's silence-chunk case) without disturbing the running average. Return the
  updated running value.
- current(self) -> float | None: returns the current running value without updating.
- reset(self): clears state for a new call/session.
Write a unit test in tests/test_ema.py that feeds a known fixed sequence of scores (e.g.,
[0.1, 0.1, 0.9, 0.9, 0.9]) through a RunningRiskScore(alpha=0.5) and asserts the running value
sequence matches hand-computed expected values at each step.
```

### Prompt 5.4 — Flag event + timing

```
Implement src/voxguard/streaming/session.py with a class StreamingSession that composes
StreamingBuffer + StreamingScorer + RunningRiskScore into one object:
- __init__(self, detector, chunk_seconds=None, overlap_seconds=None, alpha=0.3,
  flag_threshold=0.7): chunk_seconds/overlap_seconds default to config.STREAM_CHUNK_SECONDS /
  config.STREAM_OVERLAP_SECONDS when None, same pattern as StreamingBuffer above, so the whole
  streaming stack shares one source of truth for window sizing. Wires the three components
  together and records a start_time on first push().
- push_audio(self, audio_frame: np.ndarray, sr: int) -> dict: pushes into the buffer, scores any
  new complete windows, updates the running score, and returns
  {"running_score": float, "flagged": bool, "seconds_since_start": float,
  "seconds_to_flag": float | None} — seconds_to_flag is set exactly once, the first time
  flagged becomes True, and stays fixed after that (don't keep updating it on every subsequent
  chunk).
- reset(self): resets all three sub-components and clears timing state for a new call.
```

### Prompt 5.5 — Offline simulation test harness

```
Write scripts/simulate_stream.py as a CLI script (--audio_path, --chunk_seconds,
--overlap_seconds, --flag_threshold) that reads a full pre-recorded audio file, feeds it into a
StreamingSession in small real-time-paced increments (simulate arrival using time.sleep to pace
playback speed — e.g., feed 250ms of audio every 250ms of wall-clock time) to validate the
system's timing behavior end-to-end BEFORE wiring it to a live microphone in Phase 6. Print each
push_audio() result as it happens, and print a final summary: total duration processed, whether
it flagged, and seconds_to_flag if it did.
```

---

## Tests

```
python -m pytest tests/test_ema.py tests/test_buffer.py -q

# Latency benchmark — confirm single-chunk inference is faster than the chunk's own duration
# before trusting the streaming design; run this on the actual detector Phase 5 will wrap
python -c "
import time, numpy as np
from src.voxguard.classifier.infer import VoxGuardDetector  # or EnsembleDetector / Phase 4's Variant A
det = VoxGuardDetector()
chunk = np.random.randn(24000).astype('float32')  # 1.5s at 16kHz, matching STREAM_CHUNK_SECONDS
times = []
for _ in range(10):
    t0 = time.time()
    det.predict_waveform(chunk, sr=16000)  # in-memory variant from Phase 2 Prompt 2.7 — returns
                                            # a dict, same shape as predict()
    times.append(time.time() - t0)
print(f'mean: {np.mean(times):.3f}s | max: {np.max(times):.3f}s | chunk duration: 1.5s')
assert np.mean(times) < 1.5, 'Single-chunk inference is slower than real-time — see Common Pitfalls'
"
python scripts/simulate_stream.py --audio_path <known_synthetic_clip.wav> --chunk_seconds 1.5 --overlap_seconds 0.5 --flag_threshold 0.7
python scripts/simulate_stream.py --audio_path <known_real_clip.wav> --chunk_seconds 1.5 --overlap_seconds 0.5 --flag_threshold 0.7
```

What to check:
- [ ] Mean single-chunk latency is comfortably under the chunk's own duration (1.5s) on your actual hardware — a mean close to the limit means occasional slow chunks will visibly lag the live demo, not just fail an assertion
- [ ] `StreamingBuffer.push()` produces correctly overlapping windows for the hand-verifiable toy example in its docstring
- [ ] `RunningRiskScore` EMA sequence matches hand-computed values in the unit test
- [ ] A known synthetic clip flags within a plausible number of seconds (a few seconds, not immediately on the first noisy chunk and not only at the very end of the clip)
- [ ] A known real/bonafide clip does NOT flag over the full simulated duration (or flags much later / with much lower confidence — false positives here undercut the "prevention" story)
- [ ] Feeding a 5-minute simulated call doesn't cause unbounded memory growth (`StreamingBuffer` correctly trims old data)

## Definition of Done Checklist

- [ ] `StreamingBuffer` implemented and unit-tested for correct windowing
- [ ] `StreamingScorer` wraps the Phase 2/3 detector for in-memory chunk scoring
- [ ] `RunningRiskScore` EMA implemented and unit-tested
- [ ] `StreamingSession` composes all three with a working flag/timing event
- [ ] Offline simulation harness validates timing behavior on both a real and a synthetic sample before any microphone integration
- [ ] `flag_threshold` default chosen based on actual simulation results, not an arbitrary guess

## Common Pitfalls

- Don't skip the offline simulation step and jump straight to live mic integration in Phase 6 — debugging timing/windowing bugs is far easier against a fixed, replayable file than against live audio.
- A per-chunk classifier call that takes longer than the chunk's real-time duration (e.g., 2 seconds of compute for a 1.5 second chunk) will make "real-time" false — benchmark single-chunk inference latency early and consider a lighter/faster classifier path if needed.
- Silence or near-silence chunks (e.g., dead air on a call) can produce erratic embedding-based scores — the `None`-on-failure handling in Prompt 5.2 and the EMA's `None`-ignoring behavior in Prompt 5.3 exist specifically to keep these from corrupting the running score.

---
# Phase 6 — Live Interactive Demo App (Gradio)

**Maps to:** Must-Have Feature 5
**Estimated time:** ~7 hours (revised up from ~6 hours to add privacy-preserving session logging — see Prompt 6.6)
**Depends on:** Phase 5 complete (`StreamingSession` working and validated via offline simulation)

---

## Objective

Build the browser app that turns everything so far into a live demo: microphone input with a real-time-updating risk score, plus a file-upload mode. This is the single highest-leverage "vibe-codeable" piece of the whole project — Gradio does most of the frontend work for you.

## Prerequisites

- Phase 5 complete and validated offline
- Gradio installed (from Phase 0's requirements.txt)
- A working microphone on the machine you'll demo from
- No GPU needed — this app is designed to run entirely on the local CPU-only machine, including at demo time (Phase 11 covers this in full)

---

## Build Prompts

### Prompt 6.1 — App scaffold with two tabs

```
Create app/app.py as a Gradio Blocks app with a title "VoxGuard — Voice Cloning Detection &
Prevention" and two tabs:
1. "Live Mic" — will hold streaming mic input (built in Prompt 6.2)
2. "Upload File" — will hold file upload + chunked-replay scoring (built in Prompt 6.3)
For now, scaffold both tabs with placeholder components (a gr.Audio input and a gr.Textbox
output each) and a launch() call at the bottom guarded by if __name__ == "__main__". Confirm the
app launches locally on `python app/app.py` before adding any real logic.
```

### Prompt 6.2 — Live mic streaming tab

```
Implement the "Live Mic" tab in app/app.py using gr.Audio(sources=["microphone"],
streaming=True) wired to a stream() event handler. On each streamed audio chunk callback:
1. Maintain one StreamingSession (from Phase 5) per Gradio session using gr.State, not a global
   variable — a global would leak state across concurrent users if this is ever hosted for more
   than one person at a time (relevant for Phase 11's HF Spaces deployment).
2. Call session.push_audio() with the incoming chunk and sample rate provided by Gradio's audio
   streaming callback signature.
3. Update a live-updating output: for now a gr.Textbox or gr.Number showing running_score,
   flagged, and seconds_to_flag if set (the visual risk meter with colors comes in Phase 7 — keep
   this phase's UI functional but plain).
4. Add a "Reset Session" button that calls session.reset() and clears the state, for starting a
   fresh call simulation without reloading the page.
```

### Prompt 6.3 — File upload tab

```
Implement the "Upload File" tab in app/app.py using gr.Audio(sources=["upload"],
streaming=False) plus a "Analyze" button. On click:
1. Run the uploaded file through BOTH: (a) the plain Phase 2/3 detector for a single whole-clip
   verdict, and (b) a full simulated run through a fresh StreamingSession (feeding the file in
   chunks programmatically, not in real time here since there's no live pacing need for an
   uploaded file — process it as fast as possible) to also report seconds_to_flag as if it had
   been a live call.
2. Display both results: whole-clip label + probability, and the streaming simulation's
   flagged/seconds_to_flag outcome.
Reuse scripts/simulate_stream.py's core logic (refactor the chunk-feeding loop out of that script
into a reusable function in src/voxguard/streaming/session.py if it isn't already factored out,
so both the CLI script and the Gradio app call the same code path — don't duplicate the loop).
```

### Prompt 6.4 — Basic UI polish (functional, not final styling)

```
Add basic layout polish to app/app.py: group each tab's inputs/outputs in gr.Row/gr.Column for a
clean layout, add a one-paragraph description under the title explaining what the app does and
that this is a hackathon prototype simulating call audio via mic/file input (not real telecom
interception — matches the Out-of-Scope note from the master guide). Don't build the final
color-coded risk meter yet — that's Phase 7's job and depends on the threshold logic built there.
```

### Prompt 6.5 — Local run & smoke test instructions

```
Add a "Running Locally" section to README.md with exact commands to activate the venv and run
`python app/app.py`, expected local URL, and a note that microphone access requires the browser
tab to be served over localhost or HTTPS (Gradio's default local server satisfies this).
```

### Prompt 6.6 — Privacy-preserving session logger

```
Implement src/voxguard/privacy/session_log.py with a class SessionLogger:
- __init__(self, log_path=None): defaults log_path to a project-local
  data/logs/session_events.jsonl (add data/logs/ to .gitignore — logs are runtime data, not
  source).
- log_event(self, event_type: str, risk_band: str, probability_synthetic: float,
  flagged: bool, matched_redflag_categories: list[str] = None, transaction_context: str = None,
  contact_match: bool | None = None) -> None: appends ONE line of JSON to log_path with a
  timestamp plus exactly these fields — deliberately NEVER accept or write raw audio bytes,
  full transcript text, or the enrolled speaker's name/identity in this function's signature at
  all, so it is structurally impossible to accidentally log anything more sensitive than a risk
  score and category labels. matched_redflag_categories should be the CATEGORY NAMES from
  Phase 9's red-flag scanner (e.g., "urgency", "financial_action"), never the matched phrase
  text itself.
- purge_older_than(self, days: int = 30) -> int: deletes log lines older than the given number
  of days (parse each line's timestamp, rewrite the file keeping only recent lines), returns the
  count of purged entries. This is the retention-policy enforcement mechanism — document in a
  docstring that 30 days is a sensible hackathon-prototype default, not a regulatory-compliant
  value, and that a production deployment would set this per applicable data protection
  requirements.
- read_recent(self, limit: int = 50) -> list[dict]: returns the most recent N logged events, for
  an optional debug/audit view.
Wire log_event calls into Prompt 6.2's Live Mic handler (once per completed session or once per
flag event, not once per chunk — that would be excessive) and Prompt 6.3's Upload tab (once per
analysis). Call purge_older_than(30) once on app startup in Prompt 6.1's app.py, not on every
request.
```

---

## Tests

```
python app/app.py
# then in browser:
# - Live Mic tab: speak normally, confirm running_score updates and doesn't error
# - Live Mic tab: click "Reset Session", confirm state clears
# - Upload File tab: upload a known real clip, confirm sane low-risk output
# - Upload File tab: upload a known synthetic clip, confirm sane high-risk output and a
#   seconds_to_flag value is reported
# - Upload File tab: upload a silent/near-silent clip, confirm no crash (graceful "inconclusive"
#   style handling per Phase 5's None-score handling)
python -c "
from src.voxguard.privacy.session_log import SessionLogger
import json
sl = SessionLogger(log_path='data/logs/test_session_events.jsonl')
sl.log_event('upload_analysis', risk_band='high', probability_synthetic=0.91, flagged=True)
events = sl.read_recent(limit=1)
assert len(events) == 1
assert 'probability_synthetic' in events[0]
assert 'audio' not in json.dumps(events[0]).lower()  # sanity check: nothing audio-like leaked in
print('session logger OK:', events[0])
"
```

Manual checks:
- [ ] App launches without error and both tabs render
- [ ] Live mic updates feel responsive (not multi-second UI lag beyond what the underlying chunk size implies)
- [ ] Concurrent users would each get their own `StreamingSession` (confirm via `gr.State` usage, not a module-level global)
- [ ] No unhandled exceptions in the terminal log during normal use
- [ ] `data/logs/session_events.jsonl` contains only risk scores/bands/categories after a normal session — manually inspect a few lines and confirm no raw audio, transcript text, or speaker names appear anywhere in the file

## Definition of Done Checklist

- [ ] Gradio app scaffolded with Live Mic and Upload File tabs
- [ ] Live mic tab streams audio into a per-session `StreamingSession` and shows live-updating results
- [ ] Upload tab reports both whole-clip and simulated-streaming results, reusing shared code (no duplicated chunking logic)
- [ ] Reset Session button works
- [ ] App runs locally end-to-end with no crashes across the manual test cases above
- [ ] `SessionLogger` implemented, wired into both tabs, and confirmed to log only feature-only data (never raw audio or verbatim transcript)
- [ ] `purge_older_than()` runs on app startup and is unit-tested with a deliberately old, manually-inserted log line

## Common Pitfalls

- Using a plain Python global variable for session state will cause one user's audio to bleed into another's score if more than one browser tab/user hits the app at once — always use `gr.State`.
- Gradio's streaming audio callback signature and sample-rate delivery format have changed across versions — check your installed Gradio version's docs for the exact `(sample_rate, numpy_array)` tuple shape it passes to your callback before wiring it to `StreamingSession.push_audio`.
- Don't let the coding agent quietly rebuild chunking/scoring logic inside `app.py` instead of importing Phase 5's modules — that duplication will drift out of sync the moment you fix a bug in one place and not the other.
- Don't let `log_event` calls quietly grow extra parameters over time (a future edit adding "just log the transcript too, it's useful for debugging") — the whole point of Prompt 6.6's narrow function signature is that logging anything more sensitive requires a deliberate signature change, not an easy accidental addition.

---
# Phase 7 — Graduated Risk Meter + Prevention Prompt

**Maps to:** Must-Have Feature 6
**Estimated time:** ~3 hours
**Depends on:** Phase 6 complete (Gradio app functional with raw score output)

---

## Objective

Replace the binary/raw-number output with a traffic-light risk meter (low/medium/high) and a concrete prevention action prompt on medium-high risk. This is the "Prevention" half of the problem statement title — most competing teams skip it, and it's mostly UI + threshold logic on top of what already exists.

## Prerequisites

- Phase 6 complete
- A validation set available for threshold calibration (ASVspoof2019 dev split works well here — don't calibrate on the eval split, that would be another form of leakage)

---

## Build Prompts

### Prompt 7.1 — Threshold configuration

```
Add to src/voxguard/config.py a RISK_THRESHOLDS dict: {"low_max": 0.3, "medium_max": 0.7}
(probability_synthetic below low_max = "low", between low_max and medium_max = "medium", above
medium_max = "high"). Add a docstring noting these are starting defaults to be recalibrated in
Prompt 7.5 against real validation data, not final values.
```

### Prompt 7.2 — Risk-band mapping function

```
Implement src/voxguard/risk/bands.py with a function
score_to_band(probability_synthetic: float, thresholds=None) -> str returning "low", "medium", or
"high" using config.RISK_THRESHOLDS if thresholds isn't explicitly passed. Write
tests/test_risk_bands.py checking exact boundary behavior (e.g., a score exactly at low_max
should consistently fall into one band or the other — pick a convention, document it in the
docstring, and test it).
```

### Prompt 7.3 — UI risk meter + prevention prompt

```
Update app/app.py's Live Mic and Upload File tabs to replace the plain numeric output with:
1. A color-coded risk display: use gr.HTML or a styled gr.Label to show "LOW RISK" in green,
   "MEDIUM RISK" in amber/yellow, "HIGH RISK" in red, driven by score_to_band().
2. On medium or high risk, show a prevention prompt block (built in Prompt 7.4) below the meter;
   hide it entirely on low risk so it doesn't create alert fatigue.
Keep the underlying raw probability visible in smaller text alongside the band label for
transparency (judges and technically curious users will want to see the actual number, not just
the color).
```

### Prompt 7.4 — Prevention prompt copy

```
Implement src/voxguard/risk/prevention.py with a function get_prevention_message(band: str) ->
str | None returning None for "low", and for "medium"/"high" a short, direct action prompt
grounded in real cyber-cell guidance, e.g.: "Verify before you act — hang up and call the person
back on a number you already have saved, not one given to you on this call. Do not share OTPs,
UPI PINs, or send money based on urgency alone." Keep "medium" slightly softer in tone
("Be cautious — consider verifying independently before acting") and "high" more direct than
"medium". Store the exact copy as named constants at the top of the file so it's easy to review
and edit as plain text, not buried in UI code.
```

### Prompt 7.5 — Threshold calibration script

```
Write scripts/calibrate_thresholds.py that loads the ASVspoof2019 dev-split predictions
(re-run the chosen best detector from Phase 3 over the dev split, or reuse cached dev
predictions if Phase 2/3 already saved them), and for a range of candidate low_max/medium_max
threshold pairs, computes the resulting false-positive rate (real audio flagged medium/high) and
false-negative rate (synthetic audio left at low). Print a small table of a few reasonable
candidate threshold pairs with their tradeoffs, and update config.py's RISK_THRESHOLDS to
whichever pair the user picks after reviewing the table (don't auto-overwrite silently — print
the recommendation and require the user to confirm or edit the values by hand).
```

---

## Tests

```
python -m pytest tests/test_risk_bands.py -q
python scripts/calibrate_thresholds.py
python app/app.py
# - upload a known real clip: confirm LOW RISK, green, no prevention prompt shown
# - upload a known borderline/ambiguous clip if you have one: confirm MEDIUM RISK, amber,
#   softer prevention prompt shown
# - upload a known synthetic clip: confirm HIGH RISK, red, direct prevention prompt shown
```

Manual checks:
- [ ] Band boundaries behave as documented (test the exact threshold value cases, not just clearly-low/clearly-high examples)
- [ ] Prevention prompt text is legible, not alarmist to the point of seeming untrustworthy, and actually actionable (a judge reading it should immediately understand what to do)
- [ ] Low risk never shows a prevention prompt (avoids alert fatigue / crying wolf)

## Definition of Done Checklist

- [ ] `RISK_THRESHOLDS` centrally configured
- [ ] `score_to_band()` implemented and boundary-tested
- [ ] Gradio UI shows a color-coded band label alongside the raw probability
- [ ] Prevention prompt copy implemented, tone-differentiated between medium and high
- [ ] Thresholds recalibrated against real validation data (not left at arbitrary defaults) and the calibration rationale documented in a comment or README section

## Common Pitfalls

- Calibrating thresholds on the eval split instead of dev split quietly reintroduces the same leakage problem flagged in Phase 1/2 — keep dev and eval strictly separated for this too.
- Showing a prevention prompt on every single medium-risk case with no distinction from high-risk can train users to ignore it — the tone differentiation in Prompt 7.4 exists specifically to avoid this.
- Don't hardcode threshold values in multiple places (UI code, calibration script, tests) — everything should read from `config.RISK_THRESHOLDS` so a single recalibration updates the whole app.

---
# Phase 8 — Speaker Voiceprint Verification (Standout Feature 7)

**Maps to:** Standout Feature 7 — top-priority stretch feature per the source doc
**Estimated time:** ~7.5 hours (small addition for voiceprint deletion/retention — see Prompt 8.2's update)
**Depends on:** Phase 6/7 app running

---

## Objective

Change the question from "is this voice synthetic" to "is this actually who they claim to be." Enroll a reference voiceprint for a trusted contact, then compare live/uploaded audio against it via speaker embeddings and cosine similarity — this is what catches the case where an attacker uses a different real human voice, not a clone at all, which a pure cloning classifier would completely miss. The enrolled-speaker registry built here also becomes the "known contact information" signal the problem statement asks for under contextual risk enrichment — Phase 9 consumes this phase's verification result directly (see Prompt 8.4's shared state).

## Prerequisites

- Phase 6/7 complete
- SpeechBrain installed (Apache 2.0, pip-installable, checkpoints auto-download on first use)
- pyannote/embedding available as backup (MIT license, requires HuggingFace terms acceptance from Phase 0)
- A consenting speaker to enroll (yourself or a teammate — same consent requirement as Phase 4)
- No GPU needed — ECAPA-TDNN is a small model, inference runs fine on CPU

---

## Build Prompts

### Prompt 8.1 — Speaker embedding extractor

```
Implement src/voxguard/speaker/embedding.py with a class SpeakerEmbedder:
- __init__(self, backend="speechbrain"): loads SpeechBrain's ECAPA-TDNN model
  (speechbrain.inference.speaker.EncoderClassifier, "speechbrain/spkrec-ecapa-voxceleb" or
  equivalent current model id) by default. If backend="pyannote", load pyannote/embedding
  instead via the pyannote.audio Inference API. Both branches should expose the same interface.
- extract(self, waveform: np.ndarray, sr: int) -> np.ndarray: resamples if needed, runs the
  model, returns a 1D embedding vector. Handle short/edge-case audio (very short clips) by
  raising a clear ValueError with a minimum-duration recommendation rather than letting the
  backend fail cryptically.
Add a fallback in __init__: if SpeechBrain model loading fails (e.g., a download issue), print a
clear message and offer to fall back to backend="pyannote" automatically, logging which backend
ended up active.
```

### Prompt 8.2 — Enrollment flow

```
Implement src/voxguard/speaker/enrollment.py with:
- enroll_speaker(name: str, reference_clips: list[str], embedder: SpeakerEmbedder) -> None:
  extracts embeddings for each reference clip (recommend 2-3 clips of 5-10 seconds each for a
  more robust average voiceprint than a single clip), averages them into one voiceprint vector,
  and saves it to models/voiceprints/{name}.npy. This is intentionally simple local file storage
  — per the project's explicit scope, a production enrollment database is out of scope; one or
  two enrolled demo voiceprints is enough to prove the mechanism. Note in a docstring, for the
  privacy/compliance story: this already stores ONLY the averaged embedding vector, never the
  raw reference audio itself — the reference clips can be deleted from disk immediately after
  enrollment and the voiceprint keeps working, which is worth stating explicitly in
  documentation rather than leaving implicit.
- list_enrolled_speakers() -> list[str]: lists available voiceprint files.
- load_voiceprint(name: str) -> np.ndarray: loads a saved voiceprint.
- delete_speaker(name: str) -> bool: removes models/voiceprints/{name}.npy if it exists,
  returns True if something was deleted, False if the name wasn't found (don't raise on a
  missing name — deleting something already gone should be a no-op, not an error). This is the
  right-to-erasure mechanism for enrolled voiceprints, called out explicitly in Phase 11's
  privacy documentation.
Add a CLI wrapper scripts/enroll_speaker.py (--name, --clips, space-separated paths) for
enrolling from the command line during testing, before the Gradio enrollment panel exists. Add
a --delete NAME flag to the same script for removing an enrollment from the command line too.
```

### Prompt 8.3 — Verification function

```
Implement src/voxguard/speaker/verify.py with:
- cosine_similarity(a: np.ndarray, b: np.ndarray) -> float
- verify_speaker(live_waveform: np.ndarray, sr: int, enrolled_name: str, embedder:
  SpeakerEmbedder, threshold=0.75) -> dict returning {"match": bool, "similarity": float,
  "enrolled_name": str}. Load the enrolled voiceprint via load_voiceprint, extract the live
  embedding, compute cosine similarity, and compare to threshold. Document that this threshold
  is a placeholder to be calibrated in Prompt 8.5, same pattern as Phase 7's risk thresholds.
```

### Prompt 8.4 — Gradio integration

```
Add a new "Voiceprint Verification" tab to app/app.py:
1. An "Enroll" section: name textbox + gr.Audio(sources=["microphone","upload"]) accepting
   multiple reference clips, an "Enroll" button calling enroll_speaker, and a confirmation
   message plus an updated dropdown (refreshed via list_enrolled_speakers) for selecting whose
   voiceprint to verify against. Add a "Remove Enrollment" button next to the dropdown, calling
   delete_speaker on the selected name, refreshing the dropdown afterward — this is the
   right-to-erasure control surface for Prompt 8.2's delete_speaker function.
2. A "Verify" section: dropdown to pick an enrolled name, gr.Audio input for the clip/live audio
   to check, a "Verify" button calling verify_speaker, and a clear result display (MATCH / 
   MISMATCH with the similarity score shown, styled consistently with Phase 7's risk meter
   colors — e.g., green for match, red for mismatch).
3. Store the most recent verify_speaker() result in a shared gr.State named
   last_voiceprint_result (dict, or None if no verification has run yet this session) at the
   app-level scope in app/app.py — Phase 9's fusion UI reads this same state to fold "is this a
   known contact" into its risk score, so the state variable name and shape here are a contract
   Phase 9 depends on: keep it exactly {"match": bool, "similarity": float, "enrolled_name": str}
   as returned by verify_speaker, or None.
Keep this tab's own display independent of the cloning-detection tabs for now (a full fused view
comes together via Phase 9's contextual fusion, which reads last_voiceprint_result rather than
this tab needing to push data anywhere itself).
```

### Prompt 8.5 — Threshold calibration for genuine acceptance vs impostor rejection

```
Write scripts/calibrate_speaker_threshold.py that, given an enrolled speaker's held-out
additional samples (genuine trials) and a handful of other speakers' samples (impostor trials —
can reuse a few speakers from the ASVspoof2019 metadata's bonafide clips as generic "other
speaker" impostor samples, since any real human who isn't the enrolled contact serves this
purpose), computes similarity scores for both trial types and reports the same-style false-accept
/ false-reject tradeoff table as Phase 7's calibration script, letting the user pick and set the
final threshold in Prompt 8.3's default value.
```

---

## Tests

```
python scripts/enroll_speaker.py --name teammate_a --clips clip1.wav clip2.wav clip3.wav
python -m pytest tests/test_speaker_verify.py -q
python scripts/calibrate_speaker_threshold.py
python scripts/enroll_speaker.py --delete teammate_a
python -c "
from src.voxguard.speaker.enrollment import list_enrolled_speakers
assert 'teammate_a' not in list_enrolled_speakers(), 'delete_speaker did not remove the enrollment'
print('deletion OK')
"
python scripts/enroll_speaker.py --name teammate_a --clips clip1.wav clip2.wav clip3.wav  # re-enroll for the rest of testing
python app/app.py
# - Voiceprint tab: verify a genuine held-out clip of the enrolled speaker -> MATCH
# - Voiceprint tab: verify a clip of a different real speaker -> MISMATCH
# - Voiceprint tab: verify a cloned/synthetic clip of the enrolled speaker (from Phase 4's
#   XTTS-v2 output, if you cloned that speaker) -> document what happens here explicitly, since
#   a well-matched clone MAY pass speaker verification even while failing cloning detection —
#   this is exactly why the two features are complementary, not redundant, and is worth
#   highlighting in your demo narrative rather than treating as a bug
# - Voiceprint tab: click "Remove Enrollment", confirm the dropdown updates and the removed name
#   can no longer be selected for verification
```

Manual checks:
- [ ] Same-speaker cosine similarity is meaningfully and consistently higher than different-speaker similarity across several trial pairs
- [ ] Enrollment persists across app restarts (files on disk, not just in-memory)
- [ ] Verification UI clearly distinguishes "voice doesn't match this contact" from "voice sounds synthetic" — these are different findings and the UI shouldn't blur them together
- [ ] `delete_speaker` removes the voiceprint file from disk, not just from an in-memory list
- [ ] `last_voiceprint_result` gr.State updates after each verification and is readable — confirm with a temporary debug print in Phase 9's fusion wiring once that phase is built

## Definition of Done Checklist

- [ ] `SpeakerEmbedder` implemented with SpeechBrain primary + pyannote fallback
- [ ] Enrollment flow saves and reloads voiceprints from local storage, storing only the embedding vector (never raw reference audio)
- [ ] `verify_speaker` returns a well-calibrated match/mismatch decision
- [ ] `delete_speaker` implemented and wired to a "Remove Enrollment" UI control
- [ ] Gradio "Voiceprint Verification" tab functional end-to-end (enroll + verify + remove)
- [ ] `last_voiceprint_result` shared state established at app-level scope with the exact shape Phase 9 expects
- [ ] Threshold calibrated against genuine vs impostor trials, not left at an arbitrary default
- [ ] The "real voice, wrong person" demo case (Step 6 of the original demo script) explicitly tested and working

## Common Pitfalls

- Don't conflate this feature's output with the cloning classifier's output in the UI — a mismatch here means "not the enrolled person," which is a different and complementary signal from "sounds synthetic." Fusing them (Phase 9) is fine; blurring their meaning in the UI is not.
- Very short reference/verification clips (under ~3 seconds) produce unreliable speaker embeddings — enforce and communicate a minimum duration rather than silently returning a low-confidence result.
- Averaging enrollment embeddings from clips recorded in very different acoustic conditions (e.g., one clean, one noisy) can produce a worse voiceprint than using the single cleanest clip — if similarity scores look off during calibration, check clip quality before assuming the model is at fault.
- Deleting an enrollment mid-demo (e.g., during a "Remove Enrollment" test) will make `last_voiceprint_result` stale until the next verification — Phase 9's fusion logic should treat a stale/missing state as "no enrollment data," never as an automatic match or mismatch (see Phase 9's three-state contact-familiarity logic).

---
# Phase 9 — Multimodal Call-Context Risk Fusion (Standout Feature 8)

**Maps to:** Standout Feature 8
**Estimated time:** ~9 hours (revised up from ~6 hours to add contextual risk enrichment — known-contact and transaction-context signals, see Prompts 9.4-9.5)
**Depends on:** Phases 5-7 complete (streaming engine + risk bands); Phase 8 complements it and is now actively read from (not just optionally complementary — Prompt 9.5's contact-familiarity signal consumes Phase 8's `last_voiceprint_result`, degrading gracefully to neutral if Phase 8 wasn't built)

---

## Objective

Transcribe the call live, scan for scam red-flag phrases, and fuse that language-based risk signal — plus two contextual enrichment signals (whether the caller matches a known/enrolled contact, and what kind of transaction is being requested) — with the audio-based cloning score into one number. Most competing teams will submit a pure audio classifier — fusing audio, language, and call-context signals makes the risk score feel like an actual product decision, not just a raw model output, and directly answers the problem statement's named requirement for "contextual enrichment using metadata such as ... known contact information, transaction context."

## Prerequisites

- Phases 5-7 complete
- `faster-whisper` installed (from Phase 0 requirements) — explicitly use the open-source local model, not a paid API, per the tech stack doc
- A curated red-flag keyword list (English + Hindi/Hinglish transliteration variants) built in Prompt 9.2
- Phase 8 complete if you want the known-contact signal active (Prompt 9.4/9.5 handle Phase 8 being absent gracefully, but the contextual-enrichment story is stronger with it)

---

## Build Prompts

### Prompt 9.1 — Transcription module

```
Implement src/voxguard/fusion/transcribe.py with a class LiveTranscriber:
- __init__(self, model_size="base", device=None): loads a faster_whisper.WhisperModel of the
  given size on config.get_device() (on this project's CPU-only local hardware, always use
  device="cpu" with compute_type="int8" — faster-whisper is specifically designed to run well
  this way, it's not a fallback or compromise here, so don't gate this behind a GPU-availability
  check, just default to it),
  document this tradeoff in a comment — smaller/faster models are fine for keyword-spotting
  purposes, this doesn't need to be a highly accurate general transcription system).
- transcribe_chunk(self, waveform: np.ndarray, sr: int) -> str: transcribes a short audio chunk
  (designed to be called on the same ~1.5-2 second windows StreamingBuffer already produces in
  Phase 5, reusing that windowing rather than building a separate one) and returns the text.
  Handle empty/silent chunks by returning an empty string rather than erroring.
- transcribe_full(self, audio_path: str) -> str: transcribes an entire file at once, for the
  Upload File tab's non-streaming use case.
Support language="hi" or language=None (auto-detect) as a constructor option, since this needs
to handle code-switched Hindi/Hinglish audio, not just English.
```

### Prompt 9.2 — Red-flag keyword list and scanner

```
Implement src/voxguard/fusion/redflags.py with:
- A RED_FLAG_PHRASES list of (phrase_or_pattern, category, weight) tuples covering common scam
  indicators in English and Hindi/Hinglish transliteration, e.g. categories like "urgency"
  ("right now", "abhi turant"), "financial_action" ("send money", "UPI", "OTP", "bank details"),
  "authority_impersonation" ("police", "arrest", "court", "customs"), "isolation" ("don't tell
  anyone", "kisi ko mat batana"). Use simple case-insensitive regex/substring matching, not exact
  string equality, so minor phrasing variants still match. Keep this list in one clearly editable
  place — it will need tuning after early tests.
- scan_for_redflags(text: str) -> dict returning {"matched_phrases": list[str], "categories":
  list[str], "keyword_risk_score": float in [0,1]} where keyword_risk_score is a simple function
  of match count and category weights (e.g., sum of matched weights, capped at 1.0) — keep the
  scoring function simple and documented rather than over-engineered, this is a supporting signal
  not the primary classifier.
```

### Prompt 9.3 — Keyword scoring function tests

```
Write tests/test_redflags.py with cases: a neutral sentence with no matches (expect
keyword_risk_score == 0), a sentence with one financial_action phrase (expect a moderate score),
a sentence combining urgency + financial_action + authority_impersonation phrases (expect a high
score, higher than the single-phrase case), and a Hindi/Hinglish sentence with a transliterated
red-flag phrase (expect it to match, confirming the scanner isn't English-only).
```

### Prompt 9.4 — Transaction context definitions

```
In src/voxguard/config.py, add:
- A TRANSACTION_CONTEXTS dict mapping context name -> risk multiplier: {"general_conversation":
  1.0, "otp_request": 1.3, "fund_transfer": 1.5, "confidential_info_request": 1.4}. Add a
  docstring explaining these are starting defaults (same "needs real calibration, not gospel"
  framing as RISK_THRESHOLDS) representing the relative stakes named in the problem statement's
  examples ("high-value transaction calls, privileged access approvals").
- A CONTACT_FAMILIARITY_MULTIPLIERS dict: {"known_match": 0.9, "known_mismatch": 1.3,
  "no_enrollment_data": 1.0}. Document the reasoning for each value directly in the comment: a
  verified match against an enrolled voiceprint mildly LOWERS risk (0.9) — mildly, not
  dramatically, because a good clone of a known contact's voice would also pass this check, so
  it's corroborating evidence, not proof; a verified MISMATCH meaningfully raises risk (1.3) —
  someone claiming to be a known contact whose voice doesn't match is a strong signal on its
  own; the absence of any enrollment data stays perfectly neutral (1.0) — an unenrolled caller
  is not inherently suspicious, and this system must not punish every unenrolled legitimate
  caller just because no voiceprint exists for them.
Implement src/voxguard/fusion/context.py with a get_transaction_multiplier(context_name: str)
-> float and get_contact_familiarity_multiplier(voiceprint_result: dict | None) -> float, where
the latter reads voiceprint_result (the same shape Phase 8's verify_speaker/last_voiceprint_result
produces: {"match": bool, "similarity": float, "enrolled_name": str}, or None) and returns the
correct multiplier from CONTACT_FAMILIARITY_MULTIPLIERS based on the three-state logic above —
None or a dict missing the expected keys both map to "no_enrollment_data", never to an error.
```

### Prompt 9.5 — Contextual fusion function

```
Implement src/voxguard/fusion/fuse.py with a function
fuse_risk(audio_score: float, keyword_risk_score: float, audio_weight=0.7,
keyword_weight=0.3) -> float returning a weighted sum clipped to [0, 1]. Add a docstring
explaining the weighting rationale: audio-based cloning detection is the primary, more validated
signal (backed by Phases 2-3's evaluation), while the language signal is a secondary corroborating
signal — weights are configurable constants in config.py, not hardcoded here, so they can be
retuned without touching this function.

Then add fuse_risk_with_context(audio_score: float, keyword_risk_score: float,
transaction_context: str = "general_conversation", voiceprint_result: dict | None = None,
audio_weight=0.7, keyword_weight=0.3) -> dict returning {"base_fused_score": float,
"contextual_score": float, "transaction_multiplier": float, "contact_multiplier": float}:
computes base_fused_score via fuse_risk() first, then applies
get_transaction_multiplier(transaction_context) (Prompt 9.4) and
get_contact_familiarity_multiplier(voiceprint_result) (Prompt 9.4) as sequential multipliers on
top of it, clipping the final contextual_score to [0, 1]. Returning both the base and contextual
scores (not just the final number) matters for the demo — being able to show "audio+language
alone said X, but knowing this is a fund-transfer request from an unverified caller pushed it to
Y" is a more convincing product story than a single opaque number.

Write tests/test_fuse.py checking: fuse_risk()'s known input/output combinations, including the
specific case the source doc calls out (low audio score + high keyword score should still
produce a visibly elevated fused score, not get washed out by the audio weight); AND
fuse_risk_with_context()'s three contact-familiarity states independently (known_match lowers
the score vs. the base, known_mismatch raises it, no_enrollment_data leaves it unchanged) and at
least one transaction-context case (fund_transfer raises the score vs. general_conversation for
the same underlying audio/keyword inputs).
```

### Prompt 9.6 — Gradio integration

```
Update app/app.py's Live Mic tab (and Upload File tab) to additionally:
1. Run LiveTranscriber.transcribe_chunk on each streaming window (or transcribe_full for
   uploads) alongside the existing StreamingSession scoring.
2. Run scan_for_redflags on the transcribed text.
3. Add a "Transaction Context" gr.Dropdown (options matching TRANSACTION_CONTEXTS' keys from
   Prompt 9.4, human-readable labels like "General conversation", "OTP request", "Fund
   transfer", "Confidential info request", default "general_conversation") to both tabs, so the
   user (playing the role of the call recipient) can indicate what the call is actually about.
4. Read the shared last_voiceprint_result gr.State established in Phase 8's Prompt 8.4 (default
   to None if Phase 8 wasn't built or nothing has been verified yet this session — the fusion
   function handles this gracefully per Prompt 9.4/9.5).
5. Compute fuse_risk_with_context() combining the audio score, keyword_risk_score, the selected
   transaction context, and last_voiceprint_result, and feed the resulting contextual_score into
   Phase 7's score_to_band()/prevention-prompt logic instead of the raw audio-only score or the
   Prompt 9.5-only base_fused_score — the risk meter and prevention prompt should now reflect
   the fully contextually-enriched signal.
6. Display the live transcript with matched red-flag phrases visually highlighted (e.g., wrapped
   in <mark> tags if rendering via gr.HTML, or listed separately below the transcript), AND show
   base_fused_score alongside the final contextual_score (small text, not the main risk meter)
   so it's visible how much the context adjustment changed the number — this transparency is
   worth keeping even in the final UI, not just for debugging.
Be explicit in a code comment that this changes the meaning of the risk meter built in Phase 7 —
from "audio cloning risk" to "overall contextual call risk" — and update any UI labels
accordingly so it's clear to a user what's being measured.
```

---

## Tests

```
python -m pytest tests/test_redflags.py tests/test_fuse.py -q
python -c "
from src.voxguard.fusion.transcribe import LiveTranscriber
t = LiveTranscriber()
print(t.transcribe_full('<sample_clip.wav>'))
"
python -c "
from src.voxguard.fusion.fuse import fuse_risk_with_context
# no enrollment data, general conversation — should equal the plain fused score
r1 = fuse_risk_with_context(audio_score=0.3, keyword_risk_score=0.2, transaction_context='general_conversation', voiceprint_result=None)
# same inputs, but a fund-transfer request from a MISMATCHED known contact — should be visibly higher
r2 = fuse_risk_with_context(audio_score=0.3, keyword_risk_score=0.2, transaction_context='fund_transfer', voiceprint_result={'match': False, 'similarity': 0.2, 'enrolled_name': 'boss'})
assert r2['contextual_score'] > r1['contextual_score'], 'contextual enrichment did not raise risk as expected'
print('r1:', r1)
print('r2:', r2)
"
python app/app.py
# - Live Mic: read a scripted "neutral chat" sentence, confirm low keyword risk, fused score
#   close to the raw audio score
# - Live Mic: read a scripted "scam call" sentence (e.g., "this is urgent, send the OTP right
#   now or you will be arrested") over a REAL (non-cloned) voice, confirm keyword risk is high
#   and the fused score is visibly elevated above what the audio-only score alone would suggest
# - Upload File: test with a synthetic clip whose transcript ALSO contains red-flag phrases,
#   confirm fused score is at or near maximum
# - Live Mic: set Transaction Context to "Fund transfer" and verify against a MISMATCHED voice
#   (Phase 8), confirm the final contextual_score is visibly higher than base_fused_score, and
#   that the small "base vs. contextual" display makes this shift legible
# - Live Mic: set Transaction Context to "General conversation" with no voiceprint verification
#   performed this session, confirm contextual_score equals base_fused_score exactly (neutral
#   multiplier applied, not a silent penalty for having no enrollment data)
```

Manual checks:
- [ ] Transcription runs fast enough not to visibly lag behind the audio-score updates in the live UI
- [ ] Red-flag scanner correctly matches at least one Hindi/Hinglish transliterated phrase, not just English
- [ ] Fused score behaves as documented in the "low audio + high keyword" test case — this is the feature's whole point, verify it explicitly rather than assuming the math works
- [ ] All three contact-familiarity states produce the documented directional effect (match lowers, mismatch raises, no-data is neutral) — test each explicitly, don't just check one and assume the others follow
- [ ] Selecting each of the four transaction contexts visibly changes the contextual score for otherwise-identical inputs

## Definition of Done Checklist

- [ ] `LiveTranscriber` implemented using faster-whisper, supports both chunked and full-file transcription
- [ ] Red-flag phrase list covers English and Hindi/Hinglish, organized by category with weights
- [ ] `scan_for_redflags` implemented and unit-tested including a code-switched case
- [ ] `fuse_risk` implemented, configurable weights, unit-tested including the low-audio/high-keyword case
- [ ] `TRANSACTION_CONTEXTS` and `CONTACT_FAMILIARITY_MULTIPLIERS` defined in config.py with documented rationale for each value
- [ ] `fuse_risk_with_context` implemented and unit-tested for all three contact-familiarity states and at least one transaction-context case
- [ ] Gradio app's risk meter now reflects `contextual_score` (not just `base_fused_score`), with UI labeling updated to reflect that this measures overall contextual call risk
- [ ] Transaction Context dropdown wired into both Live Mic and Upload File tabs
- [ ] `last_voiceprint_result` correctly read from Phase 8's shared state, with graceful "no data" handling if Phase 8 wasn't built or nothing has been verified yet
- [ ] Live transcript with highlighted red-flag terms visible in the UI
- [ ] base_fused_score vs. contextual_score both visible in the UI, so the context adjustment's effect is transparent, not hidden

## Common Pitfalls

- Running full Whisper transcription on every tiny audio chunk can be slow enough to break the "live" feel — use the smallest model size that still transcribes intelligibly, and consider transcribing on a slightly longer window than the audio-scoring window if latency is an issue (they don't have to share the exact same chunk size, just be reasonably in sync).
- An overly broad keyword list (e.g., matching on "money" alone with no context) will false-positive constantly on ordinary conversation — keep phrases specific enough to be genuinely scam-associated, and validate against a few neutral conversation samples before the demo.
- Don't let the fusion weights drift to effectively ignoring one signal (e.g., audio_weight=0.99) without a documented reason — the whole point of this feature is that both signals meaningfully contribute.
- **Don't let a missing Phase 8 integration silently break this phase.** If Phase 8 wasn't built, `last_voiceprint_result` simply doesn't exist as a gr.State — guard the read with a safe default (None) rather than assuming the state variable is always present, or the Gradio app will crash for anyone who skipped Phase 8.
- **Don't let the contact-familiarity multiplier punish unenrolled callers.** This is the one design mistake that would make the feature actively harmful rather than helpful — a legitimate caller who simply isn't in the enrolled-contacts list must never be treated as more suspicious than a caller with no context data at all. Re-check the `no_enrollment_data: 1.0` mapping specifically if these numbers ever get retuned.

---
# Phase 10 — Explainability Overlay (Standout Feature 9)

**Maps to:** Standout Feature 9
**Estimated time:** ~4 hours
**Depends on:** Phase 2 complete (classifier + embedding extractor); `librosa` installed

---

## Objective

Give judges something visual to look at beyond a probability number: a spectrogram with the regions that most influenced the "synthetic" verdict highlighted. This answers the "how do we trust this" question any credible detection system needs to address.

## Prerequisites

- Phase 2 complete
- `librosa` and `matplotlib` installed (from Phase 0 requirements)

---

## Build Prompts

### Prompt 10.1 — Spectrogram generation

```
Implement src/voxguard/explain/spectrogram.py with a function
generate_mel_spectrogram(waveform: np.ndarray, sr: int, n_mels=128) -> np.ndarray returning a
log-scaled mel-spectrogram (librosa.feature.melspectrogram followed by
librosa.power_to_db). Add a render_spectrogram_image(mel_spec: np.ndarray, sr: int,
hop_length=512, output_path: str) function that renders it via
librosa.display.specshow/matplotlib and saves a PNG, for use in the Gradio overlay.
```

### Prompt 10.2 — Coarse windowed attribution (primary approach)

```
Implement src/voxguard/explain/attribution.py with a function
windowed_attribution(waveform: np.ndarray, sr: int, detector, window_seconds=0.5,
stride_seconds=0.25) -> np.ndarray that segments the input waveform into small overlapping
windows, runs the SAME detector used elsewhere in the project on each window independently
(reusing whatever predict_waveform-style method Phase 5 already added to the detector classes),
and returns a 1D array of per-window synthetic-probability scores aligned to time. Document this
clearly as a coarse, model-agnostic attribution method — it directly reuses the trusted, already-
validated classifier rather than requiring a new gradient-based method to also be validated for
correctness under time pressure. This is a deliberate, defensible choice for a hackathon
timeline: prefer explaining via the actual scoring function over building a separate saliency
mechanism that could itself be wrong in ways nobody has time to double check.

As a secondary/bonus path only if time allows, add a function
gradient_saliency(waveform, sr, embedding_extractor, classifier) that computes a simple
gradient-based saliency map (input waveform gradient w.r.t. the classifier's synthetic-class
output, using the embedding extractor's model in a mode that allows gradients through the input
even though its own parameters stay frozen) — implement this as a clearly separate, optional
function so a bug here doesn't block Prompt 10.3's overlay if the windowed approach alone is
used for the demo.
```

### Prompt 10.3 — Overlay rendering

```
Implement src/voxguard/explain/overlay.py with a function
render_explainability_overlay(waveform: np.ndarray, sr: int, detector, output_path: str) ->
str that: generates the mel-spectrogram (Prompt 10.1), computes windowed_attribution
(Prompt 10.2), resamples/interpolates the per-window attribution scores onto the spectrogram's
time axis, and renders a combined matplotlib figure showing the spectrogram with a semi-
transparent heatmap overlay (e.g., a red colormap with alpha scaling by attribution score) on
top, plus a colorbar legend labeled "synthetic-likelihood by region". Save to output_path and
return the path.
```

### Prompt 10.4 — Gradio integration

```
Add an "Explainability" tab (or a collapsible section within the Upload File tab, whichever
fits the layout better) to app/app.py: after analyzing an uploaded clip, call
render_explainability_overlay and display the resulting image via gr.Image. Tie it to whichever
clip was most recently analyzed (pass the audio through gr.State if needed to avoid re-uploading)
so the overlay always corresponds to the currently-displayed verdict, not a stale previous clip.
```

---

## Tests

```
python -c "
from src.voxguard.explain.spectrogram import generate_mel_spectrogram
import numpy as np
wf = np.random.randn(16000).astype('float32')
spec = generate_mel_spectrogram(wf, sr=16000)
assert spec.ndim == 2
print('spectrogram shape:', spec.shape)
"
python -c "
from src.voxguard.explain.overlay import render_explainability_overlay
from src.voxguard.classifier.infer import VoxGuardDetector
from src.voxguard.utils.audio_io import load_audio
det = VoxGuardDetector()
wf, sr = load_audio('<some_synthetic_clip.wav>')
path = render_explainability_overlay(wf, sr, det, 'test_overlay.png')
print('saved to', path)
"
python app/app.py
# - Upload a known real clip, view its explainability overlay
# - Upload a known synthetic clip, view its explainability overlay
# - Visually confirm the two overlays look meaningfully different from each other
```

Manual checks:
- [ ] Overlay image renders without matplotlib backend errors (common headless-server issue — confirm `matplotlib.use("Agg")` is set if running without a display, e.g. on a server/Colab)
- [ ] Overlay corresponds to the currently-analyzed clip, not a leftover from a previous upload
- [ ] Real vs. synthetic overlays are visually distinguishable in at least a few spot-checked examples — document this qualitatively even if you don't have a formal quantitative attribution-quality metric

## Definition of Done Checklist

- [ ] Mel-spectrogram generation implemented
- [ ] Windowed attribution implemented as the primary, defensible explainability method
- [ ] (Optional, time-permitting) Gradient-based saliency implemented as a secondary method
- [ ] Combined overlay rendering implemented and saved as a displayable image
- [ ] Gradio "Explainability" view integrated and tied to the currently-analyzed clip
- [ ] At least a few real-vs-synthetic overlay comparisons visually spot-checked and documented

## Common Pitfalls

- Matplotlib defaults to an interactive backend that can crash or hang in headless environments (Colab, servers, some CI setups) — set the `Agg` backend explicitly before any plotting when running outside a normal desktop session.
- Don't oversell the windowed-attribution approach as true model interpretability in your pitch — it's a legitimate, defensible, honestly-described method ("which time regions the same trusted classifier finds most suspicious"), not a formal saliency/attribution technique with theoretical guarantees. Describe it accurately to judges; the honesty itself is a point in your favor.
- Interpolating coarse window scores onto a much finer spectrogram time axis can look artificially precise — a simple, visibly blocky/smoothed overlay that's honest about its resolution is better than one that implies pixel-level precision it doesn't have.

---
# Phase 11 — Integration, Deployment & Demo Rehearsal

**Maps to:** Final infrastructure phase — ties together Features 1-9, plus the problem statement's named "Platform and Integration APIs" and "Privacy and Compliance Module" components (Prompts 11.3 and 11.5)
**Estimated time:** ~8.5 hours (revised up from ~5 hours to add a REST API layer and privacy/compliance documentation — see Prompts 11.3 and 11.5)
**Depends on:** Phases 0-10 all complete and individually tested

---

## Objective

Merge every module into one cohesive Gradio app, expose the same underlying system as a small REST API for platform/banking-app integration demonstrability, run the live demo primarily from your own local machine (established as the right call in Phase 0 — this app is designed to run entirely on CPU), deploy a secondary copy to HuggingFace Spaces for judges to browse code and reports, document the privacy/retention posture, and rehearse the exact demo script until it's reliable under time pressure.

## Prerequisites

- Phases 0-10 all individually complete and passing their own tests
- A HuggingFace Spaces account (same account as Phase 0) — for the secondary deployment only
- GitHub repo up to date
- `fastapi`, `uvicorn`, `python-multipart`, `httpx` installed (added to `requirements.txt` in Phase 0)

## Demo Strategy: Local-Primary, Not Cloud-Primary

Decide this now, not on demo day: **your live demo runs from your own laptop, in the room.** This isn't a fallback-if-the-cloud-fails plan — it's the primary plan, decided upfront because the hardware profile is already known. Free-tier HuggingFace Spaces (CPU-only, shared/throttled resources) is not a good fit for live mic streaming + Whisper transcription + a cloning classifier running simultaneously, and there's no reason to find that out under pressure during judging when you already know it going in.

- **Primary: your laptop, live, in the room.** Full control, no network dependency, no cloud cold-start risk. This is where the "Live Call Simulation" tab (Prompt 11.1) runs during judging.
- **Secondary: HuggingFace Spaces (free CPU tier).** Deploy it so judges have a shareable link to browse your code, read the generalization/Hindi-training reports (the "Cross-Dataset Results" tab), and try the non-realtime "Upload & Analyze" tab at their own pace before or after your slot. Don't route the live-mic centerpiece through it.
- **Recorded backup video: essential, not optional.** Since your live demo depends on one physical laptop in the room, a pre-recorded run-through is your only safety net if that laptop has any issue on the day — a bad Wi-Fi handoff, a sleep/update prompt, anything. Record it once everything works, well before judging, not the night before.

---

## Build Prompts

### Prompt 11.1 — Full integration pass

```
Review app/app.py end to end and consolidate it into a single, cohesive multi-tab Gradio app
with this final tab structure:
1. "Live Call Simulation" — merges Phase 6's Live Mic tab with Phase 7's risk meter, Phase 9's
   transcript + red-flag highlighting, and the fused risk score/prevention prompt, all updating
   together as one live view (this is the centerpiece demo tab).
2. "Upload & Analyze" — merges Phase 6's Upload tab with Phase 7-9's fused scoring and Phase 10's
   explainability overlay for any uploaded file.
3. "Voiceprint Verification" — Phase 8's enroll/verify tab.
4. "Cross-Dataset Results" — a simple read-only tab (gr.Markdown or gr.Dataframe) displaying
   Phase 3's generalization_before_after.md table and Phase 4's hindi_training_comparison.md
   table (the English-only baseline vs. combined vs. Hindi-only comparison), so judges can see
   both the generalization evidence and the Hindi/Hinglish training results without leaving the app.
Remove any now-redundant duplicate UI elements left over from earlier phases' incremental builds,
and do one pass making sure every tab uses gr.State correctly for per-session isolation (per
Phase 6's Prompt 6.2 requirement) rather than any leftover global variables.
```

### Prompt 11.2 — End-to-end integration test script

```
Write tests/test_integration_e2e.py (can use pytest, doesn't need to spin up the actual Gradio
server — call the underlying functions directly) that runs, in sequence, against a small set of
known sample files: whole-clip detection including prosody features if Phase 2 selected the
prosody-augmented variant (Phase 2/3), streaming simulation with flag timing (Phase 5), risk
banding + prevention prompt (Phase 7), transcription + red-flag scan + CONTEXTUAL fusion with at
least one non-default transaction context and one voiceprint result (Phase 9), speaker
verification for at least one enrolled name (Phase 8), and explainability overlay generation
(Phase 10) — asserting each step returns a well-formed result (right types, right value ranges)
without needing to assert exact "correct" model outputs (that's what Phases 2-9's own eval
scripts already established). This is a smoke test that the pieces are wired together, not a
re-run of every phase's own metric validation. (The REST API gets its own dedicated test file,
tests/test_api.py, built in Prompt 11.3 — don't duplicate that coverage here.)
```

### Prompt 11.3 — REST API layer (Platform and Integration APIs)

```
The problem statement names "Platform and Integration APIs — REST/gRPC APIs and SDKs for
integration with core banking systems, contact center platforms, enterprise communication
tools, and telecom networks" as its own distinct Key Component. Implement a small, honestly-
scoped demonstration of this — a working single-service REST API, not a production multi-tenant
SDK (that remains explicitly out of scope, see the master guide's Out of Scope table) — as
api/main.py using FastAPI:

- GET /health -> {"status": "ok"}: liveness check.
- POST /analyze (multipart file upload, audio/wav) -> {"label": str, "probability_synthetic":
  float, "risk_band": str, "prevention_message": str | None}: wraps whichever detector Phase
  2/3/4 designated as production (Phase 4's Variant A if it exists, per that phase's Definition
  of Done — see Phase 4's guidance on which classifier downstream phases should wrap) plus
  Phase 7's score_to_band() and get_prevention_message(). This is the core "call this from your
  banking app's backend" endpoint.
- POST /verify-speaker (multipart file upload + form field enrolled_name) -> {"match": bool,
  "similarity": float, "enrolled_name": str}: wraps Phase 8's verify_speaker(). Return a 404
  with a clear error body if enrolled_name isn't found via list_enrolled_speakers().
- POST /analyze-context (multipart file upload + form fields transaction_context,
  enrolled_name (optional)) -> {"base_fused_score": float, "contextual_score": float,
  "risk_band": str, "transcript": str, "matched_redflag_categories": list[str]}: wraps Phase
  9's full pipeline (transcription, red-flag scan, fuse_risk_with_context) — if enrolled_name is
  provided, call verify_speaker first and feed its result in as voiceprint_result, otherwise
  pass None. This is the richest endpoint and the one worth demonstrating live if you have time.
Load all underlying models ONCE at API startup (FastAPI's lifespan/startup event), not per
request — reuse the exact same detector/embedder/transcriber classes built in Phases 2-9,
imported from src/voxguard, never reimplemented here. Add basic request validation (reject
non-audio content types with a 415, reject files above a reasonable size limit like 25MB with a
413) but do NOT build authentication, rate-limiting, or multi-tenancy — call this out explicitly
in a code comment as intentionally out of scope for a hackathon prototype, consistent with the
master guide's existing "Production-scale enrollment database / auth system" scope cut.
This runs as a SEPARATE process from app/app.py (e.g., `uvicorn api.main:app --port 8000`
alongside `python app/app.py` on its own port) — both import from the same src/voxguard modules,
so there's no duplicated detection logic between the demo UI and the API.
Write tests/test_api.py using FastAPI's TestClient (httpx-based) hitting /health and /analyze
with a small sample file, asserting response shape and status codes, including the 415/413
validation cases.
```

### Prompt 11.4 — HuggingFace Spaces deployment

```
Prepare the repo for HuggingFace Spaces deployment as a SECONDARY surface (per this phase's Demo
Strategy section) — not the live-demo surface: create a Space-compatible app entrypoint
(HF Spaces with the Gradio SDK expects app.py at the repo root by default, or a configured path
in a README.md front-matter block — set this up correctly), a requirements.txt at the location
HF Spaces expects, and a README.md front-matter block declaring sdk: gradio and the correct
Python version. In the Space's README/landing text, briefly note that live mic streaming is
best experienced in the in-person demo and that this hosted copy is provided for code/report
browsing and the non-realtime "Upload & Analyze" tab. Note explicitly in a comment/README
section which features run slower on the free CPU tier (faster-whisper transcription is the
most CPU-hungry inference step there — XTTS-v2 cloning generation should NOT run on the Space
at all regardless, it's a pre-build step done locally in Phase 4, only inference runs on the
Space). Test the deployed Space's "Upload & Analyze" tab specifically, since that's the one
judges are actually expected to use there.
```

### Prompt 11.5 — README, submission documentation, and privacy/compliance policy

```
Write a final root README.md covering: project summary, problem statement reference (SIH26104),
architecture overview (a text-based description of the pipeline: audio in -> embeddings +
prosody features -> classifier -> streaming/contextual fusion -> risk output, plus how
voiceprint verification, explainability, and the REST API plug in), a note on local hardware
requirements (CPU-only is sufficient, no GPU needed to run the app — Kaggle GPU was only used
during development for embedding extraction, per Phase 0/2's workflow, and is not required to
run the shipped app), setup instructions, how to run locally (the primary way to experience the
app), how to run the REST API (Prompt 11.3) alongside it, the HuggingFace Spaces link (secondary,
code/report browsing), a summary of the generalization results table from Phase 3/4, and links
to each dataset card. Include the Explicitly Out of Scope table from the master guide so judges
see scope boundaries were deliberate, not accidental gaps.

Add a dedicated "Privacy & Compliance" section covering, directly answering the problem
statement's named Privacy and Compliance Module:
- On-device/edge inference: the entire detection pipeline runs locally on the user's own
  machine by design (per Phase 0's local-hardware architecture) — no audio is sent to a third
  party for inference. Kaggle was used only during development for training-data embedding
  extraction, never for live inference on call audio.
- Minimal retention: Phase 6's SessionLogger stores only risk scores, bands, and red-flag
  CATEGORY labels — never raw audio, never verbatim transcript text — with a documented
  default 30-day auto-purge (Phase 6, Prompt 6.6).
- Voiceprints store only an averaged embedding vector, never raw reference audio (Phase 8,
  Prompt 8.2), and can be deleted on request via delete_speaker() (Phase 8, Prompt 8.2) or the
  "Remove Enrollment" UI control (Phase 8, Prompt 8.4) — the right-to-erasure mechanism.
- State plainly what ISN'T covered: this is a prototype-level privacy posture (no encryption at
  rest, no formal DPA/consent-management workflow) suitable for demonstrating the *pattern* a
  production deployment would follow, not a certified-compliant system — say this directly
  rather than implying more than what's built.
```

### Prompt 11.6 — Demo rehearsal checklist

```
Do not write code for this prompt — instead, produce a printable demo-day checklist document
scripts/DEMO_CHECKLIST.md that walks through this exact sequence (matching the original demo
script, extended with the contextual-enrichment and API additions):
1. Baseline: play a genuine clip, confirm low risk, green.
2. Cloned clip (English): play an ASVspoof/WaveFake sample, confirm it flags within a few
   seconds, risk meter goes red, prevention prompt appears.
3. Cross-dataset proof: show the "Cross-Dataset Results" tab.
4. Code-switched Hindi: play the self-built Hindi/Hinglish real and cloned pair, confirm both are
   handled correctly.
5. Live challenge: invite a judge to speak into the mic (running locally on your laptop per
   this phase's Demo Strategy — not the hosted Space), then play a pre-loaded cloned clip and
   show the live streaming score react in real time.
6. Voiceprint mismatch: play a real, unsynthesized voice that isn't the enrolled contact, show it
   still gets flagged as a mismatch — the "not a clone, but not them either" case.
7. Contextual enrichment: replay the same mismatched-voice clip from step 6, but this time set
   Transaction Context to "Fund transfer" first, and show the risk score visibly jump above what
   step 6 alone produced — pointing at the visible base-score-vs-contextual-score display from
   Phase 9, Prompt 9.6. This is the moment that directly answers the problem statement's
   "contextual enrichment" requirement, so it's worth calling out explicitly in your narration,
   not just letting the number change quietly.
8. Platform integration: with api/main.py running in a second terminal, `curl -X POST
   http://localhost:8000/analyze -F file=@sample.wav` live (or a pre-typed command ready to
   paste) to show the same detection logic is callable as a plain REST endpoint — this answers
   "how would a bank actually integrate this" in about 15 seconds without derailing the main UI
   demo.
9. Close on the fused, contextually-enriched risk score and the concrete prevention prompt,
   tying back to the real scam pattern (cloned voice + urgency + request for money over UPI, now
   further elevated by fund-transfer context and an unmatched voiceprint) and what the system
   does about it in the moment.
For each step, list: what to click/say, what result to expect on screen, and a one-line fallback
if that exact result doesn't reproduce live (e.g., a pre-recorded backup clip, a cached
screenshot, or — for step 8 specifically — a pre-recorded terminal screenshot/GIF of the curl
call and its JSON response as a fallback if live API demonstration risks eating too much time).
```

---

## Tests

```
# Fresh-clone test — do this on a second machine or a clean directory if possible
git clone <your_repo_url> voxguard_fresh_test
cd voxguard_fresh_test
bash scripts/setup_env.sh
python -m pytest tests/ -q
python app/app.py
# confirm this launches and all tabs work with no GPU present — this IS your demo machine setup

# API test — run alongside the Gradio app on a separate port
uvicorn api.main:app --port 8000 &
curl http://localhost:8000/health
curl -X POST http://localhost:8000/analyze -F "file=@<sample.wav>"
python -m pytest tests/test_api.py -q

# HF Spaces test — after deployment (secondary surface only)
# open the Spaces URL in an incognito browser window and manually test the "Upload & Analyze"
# and "Cross-Dataset Results" tabs specifically — these are what judges will actually use there
```

Manual checks:
- [ ] Fresh clone + fresh venv installs and runs with zero missing-dependency errors, on CPU only, no GPU present
- [ ] All tabs load and function in the consolidated app, not just the phase they were originally built in
- [ ] `api/main.py` runs alongside `app/app.py` without port conflicts, and all three endpoints (`/health`, `/analyze`, `/verify-speaker`, `/analyze-context`) return well-formed responses
- [ ] Local live-mic demo tested on the actual laptop/room setup you'll use for judging, not just at your desk — different room, different Wi-Fi, different power situation can all matter
- [ ] HuggingFace Spaces link loads and its "Upload & Analyze" + "Cross-Dataset Results" tabs work — this is the secondary surface, not where the live-mic centerpiece is expected to run
- [ ] Full demo script rehearsed successfully at least twice back-to-back, locally, without manual code fixes in between, including the new contextual-enrichment (step 7) and API (step 8) beats
- [ ] Every fallback listed in `DEMO_CHECKLIST.md` has actually been prepared in advance (backup clips exist on disk, not just described in the checklist — this now includes a pre-recorded API-call screenshot/GIF fallback)

## Final Submission Checklist

- [ ] Repo public (or shared with judges per SIH submission instructions) with clean commit history
- [ ] README.md complete: summary, architecture, local-hardware note, setup, results, scope boundaries, Privacy & Compliance section
- [ ] All dataset cards present (`HINDI_HINGLISH_DATASET_CARD.md` at minimum)
- [ ] `requirements.txt` accurate and installs cleanly from scratch on a CPU-only machine
- [ ] Generalization report (Phase 3) and Hindi/Hinglish eval (Phase 4) results included in README or linked
- [ ] HuggingFace Spaces link live and tested from an incognito/unauthenticated browser session, as the secondary code/report-browsing surface
- [ ] REST API (`api/main.py`) implemented, tested, and documented in the README as the answer to the problem statement's Platform and Integration APIs component
- [ ] Privacy & Compliance section written, honestly scoping what is and isn't covered
- [ ] Local demo laptop confirmed as the primary live-demo machine, tested in the actual room/setup if possible
- [ ] Recorded demo video backup exists — this is essential given the local-primary plan, not a nice-to-have
- [ ] `scripts/DEMO_CHECKLIST.md` printed or otherwise available during the actual judging slot

## Common Pitfalls

- Don't do your first-ever full integration test the night before judging — Prompt 11.2's e2e test and the fresh-clone test exist specifically to surface wiring bugs while there's still time to fix them.
- Don't let the free-tier HF Spaces deployment become a distraction from the local demo that actually matters — it's secondary by design here, not a discovery you make late. Spend rehearsal time on the local laptop setup, not on debugging Spaces' CPU tier for live streaming it was never meant to carry.
- Always have a recorded video backup. Live mic demos are the most impressive when they work and the single riskiest point of failure in the room — especially with a local-primary plan where everything rides on one physical machine — a backup costs you 10 minutes to record and can save the entire presentation.
- Don't let the API demo (step 8) eat time from the main narrative — it's a 15-second "and yes, this is also callable as a REST endpoint" beat, not a second full demo. Have the curl command pre-typed or in shell history, don't type it live under time pressure.
- If asked whether the API has authentication, rate-limiting, or multi-tenant support, answer honestly that it's a scoped prototype demonstrating the integration pattern, not a production-ready service — this is a documented, deliberate scope cut, not a gap to talk around.

---
