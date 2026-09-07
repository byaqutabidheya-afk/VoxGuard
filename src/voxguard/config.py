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

# Phase 3 — classification thresholds
# Calibrated via scripts/calibrate_thresholds.py on ASVspoof2019 dev using
# the production WeightedAverageDetector
# (wav2vec2_hindi_combined_logreg + wavlm_hindi_combined_logreg, weight_a=0.5).
#
# Candidate tradeoffs from that calibration sweep:
# - 0.20/0.60 -> FNR_low=0.10%, FPR_flag=23.19%, FPR_high=3.92%, TPR_high=98.65%
# - 0.30/0.70 -> FNR_low=0.14%, FPR_flag=19.27%, FPR_high=3.02%, TPR_high=98.08%
# - 0.50/0.80 -> FNR_low=0.46%, FPR_flag= 7.69%, FPR_high=2.16%, TPR_high=97.45%
#
# We deliberately keep 0.50/0.80 for demo behavior: it greatly reduces false
# alarms on genuine speech (FPR_flag 7.69% vs 23.19% at 0.20/0.60) while still
# keeping synthetic misses very low (FNR_low 0.46%). A production deployment
# might reasonably choose a tighter setting to prioritize catch rate further.
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

# Decision threshold used by the streaming session wrapper.
STREAM_FLAG_THRESHOLD: float | None = 0.6

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
