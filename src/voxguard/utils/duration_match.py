"""
duration_match.py — trim a real/synthetic waveform pair to equal length.

When comparing a real utterance against its synthetic clone, a duration
mismatch is itself a trivially learnable cue.  ``duration_match_pair``
removes it by centre-cropping the longer clip down to the shorter clip's
sample count.  It never pads: padding would inject silence (another
spurious cue), so a pair that is too short after matching is rejected
with ``ValueError`` and callers are expected to skip it.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from voxguard.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _validate(name: str, waveform: np.ndarray) -> np.ndarray:
    arr = np.asarray(waveform)
    if arr.ndim != 1:
        raise ValueError(
            f"{name} waveform must be 1-D mono audio, got shape {arr.shape}"
        )
    if arr.size == 0:
        raise ValueError(f"{name} waveform is empty")
    return arr


def duration_match_pair(
    real_waveform: np.ndarray,
    synth_waveform: np.ndarray,
    sr: int,
    min_seconds: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Trim the longer of two waveforms to the shorter one's length.

    An equal number of samples is removed from the start and the end of
    the longer clip; if the difference is odd, the extra sample comes off
    the end.  The shorter clip is returned untouched.  Never pads.

    Args:
        real_waveform: 1-D real-speech waveform.
        synth_waveform: 1-D synthetic-speech waveform.
        sr: Sample rate shared by both waveforms (Hz).
        min_seconds: Minimum acceptable matched duration.

    Returns:
        ``(matched_real, matched_synth, info)`` where ``info`` contains
        ``original_real_seconds``, ``original_synth_seconds``,
        ``matched_seconds``, ``matched_samples``, ``trimmed_seconds``
        (total removed), ``trimmed_start_seconds``,
        ``trimmed_end_seconds`` and ``trimmed_clip``
        (``"real"``, ``"synth"`` or ``"none"``).

    Raises:
        ValueError: If ``sr`` is not positive, either input is empty or
            not 1-D, or the matched duration is below ``min_seconds``.
    """
    if sr <= 0:
        raise ValueError(f"sr must be positive, got {sr}")

    real = _validate("real", real_waveform)
    synth = _validate("synth", synth_waveform)

    n_real, n_synth = real.shape[0], synth.shape[0]
    target = min(n_real, n_synth)
    matched_seconds = target / sr

    if matched_seconds < min_seconds:
        raise ValueError(
            f"matched duration {matched_seconds:.3f}s is below "
            f"min_seconds={min_seconds}s (real={n_real / sr:.3f}s, "
            f"synth={n_synth / sr:.3f}s)"
        )

    diff = abs(n_real - n_synth)
    start = diff // 2
    end_trim = diff - start  # odd extra sample goes to the end

    if n_real > n_synth:
        trimmed_clip = "real"
        real = real[start : start + target]
    elif n_synth > n_real:
        trimmed_clip = "synth"
        synth = synth[start : start + target]
    else:
        trimmed_clip = "none"

    info: Dict[str, Any] = {
        "original_real_seconds": n_real / sr,
        "original_synth_seconds": n_synth / sr,
        "matched_seconds": matched_seconds,
        "matched_samples": target,
        "trimmed_seconds": diff / sr,
        "trimmed_start_seconds": start / sr,
        "trimmed_end_seconds": end_trim / sr,
        "trimmed_clip": trimmed_clip,
    }
    logger.debug("duration_match_pair: %s", info)
    return real, synth, info
