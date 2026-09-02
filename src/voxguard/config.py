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

DATA_RAW_DIR: Path       = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
DATA_METADATA_DIR: Path  = BASE_DIR / "data" / "metadata"
MODELS_DIR: Path         = BASE_DIR / "models"

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
# Hugging Face model-hub identifier, e.g. "facebook/wav2vec2-base" or
# "microsoft/wavlm-base-plus".  Left unset until Phase 1 selects the backbone.
EMBEDDING_MODEL_NAME: str | None = None  # TODO Phase 1

# Phase 3 — classification thresholds
# Dict mapping label → probability threshold, e.g. {"synthetic": 0.5}.
# Tuned on the validation set in Phase 3; do not invent a value here.
RISK_THRESHOLDS: dict | None = None  # TODO Phase 3

# Phase 4 — streaming / real-time inference
# Duration of each audio chunk fed to the model (seconds).
STREAM_CHUNK_SECONDS: float | None = None  # TODO Phase 4

# Overlap between consecutive chunks to avoid boundary artefacts (seconds).
STREAM_OVERLAP_SECONDS: float | None = None  # TODO Phase 4

# Phase 9 (Prompt 9.5) — multimodal risk fusion context tables
# Maps a call/transaction context string to a risk-weight multiplier,
# e.g. {"banking": 1.5, "general": 1.0}.  Filled in during Phase 9 Prompt 9.5
# once the fusion feature set is finalised.
TRANSACTION_CONTEXTS: dict | None = None           # TODO Phase 9 Prompt 9.5

# Maps a contact-familiarity label to a risk-weight multiplier,
# e.g. {"unknown": 1.3, "known": 0.8}.  Same phase as above.
CONTACT_FAMILIARITY_MULTIPLIERS: dict | None = None  # TODO Phase 9 Prompt 9.5

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
