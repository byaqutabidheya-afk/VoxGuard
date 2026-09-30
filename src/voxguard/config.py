"""
config.py — single source of truth for paths, constants, and runtime settings.

Every other module in VoxGuard should import what it needs from here.
No module should hardcode a file-system path or a tunable constant —
change it once here and the whole project picks it up.

Layout of this file
───────────────────
  1. File-system paths          (always absolute, derived from __file__)
  2. Audio constants            (sample rate, etc.)
  3. Phase-specific placeholders (filled in as each phase is implemented)
  4. Runtime helpers            (device selection, thread-count hint)
"""

from __future__ import annotations

import os
from pathlib import Path

# =============================================================================
# 1. File-system paths
# =============================================================================
# config.py lives at:  <repo>/src/voxguard/config.py
# parents[0] → src/voxguard/
# parents[1] → src/
# parents[2] → <repo root>
BASE_DIR: Path = Path(__file__).resolve().parents[2]

DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
DATA_METADATA_DIR: Path = BASE_DIR / "data" / "metadata"
MODELS_DIR: Path = BASE_DIR / "models"

# =============================================================================
# 2. Audio constants
# =============================================================================

# All audio is resampled to this rate before any feature extraction or
# embedding.  16 kHz is the native rate of wav2vec2 / HuBERT / WavLM and
# most speech-domain models.
SAMPLE_RATE: int = 16_000

# =============================================================================
# 3. Phase-specific placeholders
#    — set to None until the relevant phase is implemented.
#    — each entry notes which phase/prompt will fill it in.
# =============================================================================

# Phase 1 — embedding backbone
# Hugging Face model-hub identifier for the SSL speech embedding backbone.
# "facebook/wav2vec2-base" is the project default. "microsoft/wavlm-base-plus"
# is the alternative used later in Phase 3.
EMBEDDING_MODEL_NAME: str = "facebook/wav2vec2-base"

# Phase 3 / F4 — classification thresholds
# Re-calibrated in F4 via scripts/calibrate_thresholds.py on the ASVspoof2019
# DEV split (24,844 clips: 2,548 real, 22,296 synthetic; never eval) using the
# production WHOLE-CLIP detector: PRODUCTION_WHOLECLIP_CLASSIFIERS (F1 matched
# wav2vec2 + F3 v2 WavLM) at PRODUCTION_ENSEMBLE_WEIGHT_A = 0.5. The earlier
# Phase 7 rationale was measured on the old hindi_combined heads and no longer
# applies. Sweep on the new detector (FNR_low = synthetic left "low";
# FPR_flag = real reaching medium/high; TPR_high = synthetic reaching "high"):
# - 0.20/0.60 -> FNR_low=0.13%, FPR_flag=23.35%, FPR_high=3.89%, TPR_high=98.10%
# - 0.30/0.70 -> FNR_low=0.25%, FPR_flag=15.62%, FPR_high=2.28%, TPR_high=95.69%
# - 0.30/0.75 -> FNR_low=0.25%, FPR_flag=15.62%, FPR_high=1.45%, TPR_high=93.91%
# - 0.50/0.80 -> FNR_low=0.70%, FPR_flag= 7.73%, FPR_high=0.78%, TPR_high=91.49%
#
# Chosen: 0.50/0.80 (unchanged from Phase 7). The script's printed rule
# (lowest FNR_low with FPR_high < 5%) recommends 0.20/0.60, but that flags
# 23.35% of genuine clips as medium/high, roughly 3x the rate of 0.50/0.80
# (7.73%), for 0.57 pp fewer missed synthetics; false alarms on real calls are
# what destroy user trust. Every pair keeps FNR_low under 1%, so the choice is
# between false-alarm rate and how many synthetic clips reach "high" rather
# than "medium". Against the old model at the same pair, FPR_high improved
# (2.16% -> 0.78%) but TPR_high fell (97.45% -> 91.49%) and FNR_low rose
# (0.46% -> 0.70%): more synthetic clips now land in the "medium" band.
# Caveat: ASVspoof2019 is English and this is a dev-split operating point;
# no Hindi set is large enough (25 real clips) to calibrate on. The streaming
# (chunk-native) family is not calibrated here; see STREAM_FLAG_THRESHOLD.
RISK_THRESHOLDS: dict = {
    # probability_synthetic < low_max              → risk level "low"
    # low_max ≤ probability_synthetic ≤ medium_max → risk level "medium"
    # probability_synthetic > medium_max           → risk level "high"
    "low_max": 0.5,
    "medium_max": 0.8,
}

# Phase 4 — streaming / real-time inference
# Duration of each audio chunk fed to the model (seconds).
STREAM_CHUNK_SECONDS: float | None = 1.5

# Overlap between consecutive chunks to avoid boundary artefacts (seconds).
STREAM_OVERLAP_SECONDS: float | None = 0.5

# --- F4 stream calibration (written by scripts/fix_calibrate_stream_threshold.py) ---
# Calibrated in F4 by replaying the soumya MATCHED eval clips (held-out; 25 real + 25 synthetic) through the real
# StreamingSession wired to PRODUCTION_STREAMING_CLASSIFIERS (chunked wav2vec2 + v2 chunked WavLM, weight 0.5).
# Sweep: threshold in [0.4..0.8] x consecutive score updates in [1..4]. Rule: lowest false-positive rate on real clips; ties ->
# highest detection rate; remaining ties -> lowest median seconds_to_flag. A false positive on genuine speech outranks
# latency. Selected on soumya alone; the Phase 6 demo pairs (mostly training speakers) were reported but not used.
# Result at 0.8/2: false positives 0/25 real clips, detections 22/25 synthetic clips,
# median seconds_to_flag 2.50s. Highest real-clip max running score: 1.000.
# STREAM_CONSECUTIVE_FLAGS_REQUIRED counts consecutive per-stride SCORE UPDATES (StreamingSession(consecutive_unit=
# "updates")): the running score only changes when a window completes (first at 1.5 s, then every 1.0 s), so N means N
# independent decisions in a row and the earliest flag is 1.5 s + (N-1) x 1.0 s. The session default unit is still "pushes";
# call sites must pass STREAM_CONSECUTIVE_UNIT. 25 real clips = 4 pp per clip; treat small differences as noise.
# The whole-clip RISK_THRESHOLDS are calibrated separately and are not affected. Call sites read this in F5.
STREAM_FLAG_THRESHOLD: float | None = 0.8
STREAM_CONSECUTIVE_FLAGS_REQUIRED: int = 2
STREAM_CONSECUTIVE_UNIT: str = "updates"
# --- end F4 stream calibration ---

# Phase 7 / Prompt 9.4 — Multimodal Risk Fusion Context Tables
#
# Starting defaults representing the relative stakes named in the problem
# statement's examples ("high-value transaction calls, privileged access approvals").
# Like RISK_THRESHOLDS, these values represent starting heuristic defaults
# requiring empirical calibration against production fraud telemetry rather
# than immutable constants.
TRANSACTION_CONTEXTS: dict[str, float] = {
    "general_conversation": 1.0,
    "otp_request": 1.3,
    "fund_transfer": 1.5,
    "confidential_info_request": 1.4,
}

# Contact Familiarity Multipliers:
# - "known_match" (0.9): A verified match against an enrolled voiceprint
#   mildly LOWERS risk (0.9) — mildly, not dramatically, because a good clone
#   of a known contact's voice would also pass this check, so it's corroborating
#   evidence, not proof.
# - "known_mismatch" (1.3): A verified MISMATCH meaningfully raises risk (1.3) —
#   someone claiming to be a known contact whose voice doesn't match is a strong
#   signal on its own.
# - "no_enrollment_data" (1.0): The absence of any enrollment data stays
#   perfectly neutral (1.0) — an unenrolled caller is not inherently suspicious,
#   and this system must not punish every unenrolled legitimate caller just
#   because no voiceprint exists for them.
CONTACT_FAMILIARITY_MULTIPLIERS: dict[str, float] = {
    "known_match": 0.9,
    "known_mismatch": 1.3,
    "no_enrollment_data": 1.0,
}

# Multimodal Risk Fusion Weights (Phase 7 / Prompt 9.5)
# Audio-based acoustic cloning detection is the primary validated signal (0.7),
# while semantic keyword scanning provides corroborating context (0.3).
FUSION_AUDIO_WEIGHT: float = 0.7
FUSION_KEYWORD_WEIGHT: float = 0.3

# Production classifier heads (Phase F3)
# There are TWO families of heads because the two inference paths feed the
# model different input distributions:
# - PRODUCTION_WHOLECLIP_CLASSIFIERS score a COMPLETE uploaded file: one
#   embedding pooled over the whole clip. They were trained on whole-clip
#   embeddings (ASVspoof2019 + the duration-matched Hindi corpus).
# - PRODUCTION_STREAMING_CLASSIFIERS score fixed-length STREAMING windows
#   (STREAM_CHUNK_SECONDS with STREAM_OVERLAP_SECONDS). They are chunk-native:
#   trained on chunk-level embeddings, whose distribution differs from
#   whole-clip embeddings, so a whole-clip head must not score a chunk or
#   vice versa.
# Each family holds one head per backbone, combined by a weighted average.
# The WavLM heads are the _v2 retrains from Phase F3.2a (C=0.001, Hindi rows
# oversampled 20x); the wav2vec2 heads are unchanged from F1/F2. F3 took the
# retrain branch, not the reweight branch, so the ensemble weight stays at the
# default 0.5: only the WavLM head itself changed.
# Paths are relative to BASE_DIR. Call sites are not wired to these yet (F5).
PRODUCTION_WHOLECLIP_CLASSIFIERS: dict[str, str] = {
    "wav2vec2": "models/classifiers/wav2vec2_hindi_matched_logreg.joblib",
    "wavlm": "models/classifiers/wavlm_hindi_matched_v2_logreg.joblib",
}
PRODUCTION_STREAMING_CLASSIFIERS: dict[str, str] = {
    "wav2vec2": "models/classifiers/wav2vec2_chunked_logreg.joblib",
    "wavlm": "models/classifiers/wavlm_chunked_v2_logreg.joblib",
}
# Weight on wav2vec2 in the weighted average; WavLM gets 1 - this.
PRODUCTION_ENSEMBLE_WEIGHT_A: float = 0.5



# =============================================================================
# 4. Runtime helpers
# =============================================================================


def get_device() -> str:
    """Return the best available compute device as a torch-compatible string.

    Returns ``"cuda"`` when a CUDA-capable GPU is visible to PyTorch,
    otherwise ``"cpu"``.

    On a CPU-only or iGPU-only laptop this will always return ``"cpu"``,
    which is correct — there is nothing to gain from a CUDA build locally.
    The same code running unmodified inside a Kaggle GPU notebook will
    return ``"cuda"`` automatically, which is the entire point: one
    codebase, no environment-specific branches.
    """
    try:
        import torch  # local import so config.py stays importable without torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def get_num_threads_hint() -> int:
    """Return a suggested value for ``torch.set_num_threads()``.

    Priority order
    ──────────────
    1. ``VOXGUARD_NUM_THREADS`` environment variable (explicit override).
    2. ``os.cpu_count()`` — the number of *logical* processors the OS reports.

    Usage in inference code::

        import torch
        from voxguard.config import get_num_threads_hint
        torch.set_num_threads(get_num_threads_hint())

    Benchmarking note (Phase 2)
    ───────────────────────────
    ``os.cpu_count()`` counts *logical* cores, which includes SMT/hyper-
    threading siblings.  For transformer-based inference workloads the
    sibling threads often share the same execution units and memory
    bandwidth, so doubling the thread count past the physical-core count
    can actually reduce throughput due to contention.

    Before committing to a thread count for the Phase 2 embedding
    extraction smoke test, run a short benchmark:

        for n in [1, 2, 4, physical_cores, logical_cores]:
            torch.set_num_threads(n)
            # time a representative batch of embedding extractions

    A good conservative starting point is ``os.cpu_count() // 2`` on a
    machine with SMT enabled (i.e. using only physical cores).  This
    function returns the full logical count by default so that the caller
    can decide — override via ``VOXGUARD_NUM_THREADS`` to lock in the
    benchmark winner without changing code.
    """
    env_override = os.environ.get("VOXGUARD_NUM_THREADS")
    if env_override is not None:
        try:
            value = int(env_override)
            if value > 0:
                return value
        except ValueError:
            pass  # malformed env var — fall through to os.cpu_count()

    return os.cpu_count() or 1
