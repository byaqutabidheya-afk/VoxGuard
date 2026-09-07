"""
verify.py — speaker voiceprint verification (Phase 8).

Answers a different question than the cloning classifier: not "is this
voice synthetic," but "is this actually the enrolled contact." Compares a
live/uploaded clip's speaker embedding against a previously enrolled
voiceprint (``voxguard.speaker.enrollment``) via cosine similarity — this
is what catches an attacker using a different real human voice, which a
pure cloning classifier would completely miss.
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np

from voxguard.speaker.embedding import SpeakerEmbedder
from voxguard.speaker.enrollment import load_voiceprint
from voxguard.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Calibrated via scripts/calibrate_speaker_threshold.py — same pattern as
# Phase 7's config.RISK_THRESHOLDS. Measured against 'byaquta's 23 held-out
# genuine clips (the 3 enrollment clips excluded) vs. 75 scored impostor
# trials (mahato/soumya's real clips + ASVspoof2019 bonafide dev clips,
# SpeechBrain ECAPA-TDNN backend): thresholds 0.45-0.70 all achieved
# FRR=0.00% / FAR=0.00% on this trial set (genuine similarity ranged
# 0.716-0.911, impostor -0.168-0.423, a wide gap), so 0.70 — the strictest
# threshold in that tied band — was chosen for maximum safety margin
# against impostors not seen during this calibration run. Re-run the
# calibration script if the embedder backend, the enrolled speaker, or the
# impostor pool changes meaningfully.
DEFAULT_VERIFY_THRESHOLD = 0.7


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Computes the cosine similarity between two vectors.

    Parameters
    ----------
    a, b:
        1-D vectors of the same length (e.g. two speaker embeddings).

    Returns
    -------
    float
        Cosine similarity in ``[-1, 1]`` (in practice closer to ``[0, 1]``
        for speaker embeddings, which are rarely anti-correlated).

    Raises
    ------
    ValueError
        If either vector has zero norm (undefined cosine similarity).
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Cannot compute cosine similarity with a zero-norm vector.")

    return float(np.dot(a, b) / (norm_a * norm_b))


def verify_speaker(
    live_waveform: np.ndarray,
    sr: int,
    enrolled_name: str,
    embedder: SpeakerEmbedder,
    threshold: float = DEFAULT_VERIFY_THRESHOLD,
) -> Dict[str, Union[bool, float, str]]:
    """Verifies whether *live_waveform* matches an enrolled speaker's voiceprint.

    Loads the enrolled voiceprint (``voxguard.speaker.enrollment.load_voiceprint``),
    extracts *live_waveform*'s embedding with *embedder*, and compares the two
    via cosine similarity against *threshold*.

    Parameters
    ----------
    live_waveform:
        1-D array of the audio to verify.
    sr:
        Sample rate of *live_waveform* in Hz.
    enrolled_name:
        Speaker identifier previously enrolled via
        ``voxguard.speaker.enrollment.enroll_speaker``.
    embedder:
        A loaded ``SpeakerEmbedder``. Must use the same backend the
        voiceprint was enrolled with — comparing embeddings from different
        backends (different vector spaces) produces a meaningless
        similarity score.
    threshold:
        Cosine-similarity cutoff at or above which the clip is reported as
        a match. Defaults to ``DEFAULT_VERIFY_THRESHOLD`` (0.70), calibrated
        via ``scripts/calibrate_speaker_threshold.py`` against real genuine
        vs. impostor trials — the same pattern used for Phase 7's
        ``config.RISK_THRESHOLDS``. Re-run that script if the embedder
        backend or the enrolled/impostor population changes meaningfully.

    Returns
    -------
    dict
        ``{"match": bool, "similarity": float, "enrolled_name": str}``.
        This exact shape is a contract other phases depend on directly —
        Phase 8's Gradio tab stores it verbatim in ``last_voiceprint_result``,
        and Phase 9's fusion UI reads that state to fold "is this a known
        contact" into its risk score. Do not add, rename, or drop keys.

    Raises
    ------
    FileNotFoundError
        If no voiceprint is enrolled under *enrolled_name*.
    ValueError
        If *live_waveform* is too short for a reliable embedding (raised
        by ``embedder.extract``).
    """
    voiceprint = load_voiceprint(enrolled_name)
    live_embedding = embedder.extract(live_waveform, sr)

    similarity = cosine_similarity(live_embedding, voiceprint)
    match = similarity >= threshold

    logger.info(
        "verify_speaker(enrolled_name=%r): similarity=%.4f threshold=%.4f -> match=%s",
        enrolled_name,
        similarity,
        threshold,
        match,
    )

    return {
        "match": bool(match),
        "similarity": similarity,
        "enrolled_name": enrolled_name,
    }
