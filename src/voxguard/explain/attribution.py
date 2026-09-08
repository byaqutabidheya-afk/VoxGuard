"""attribution.py — coarse, window-based attribution of synthetic-speech probability.

Provides one public function:

* :func:`windowed_attribution` — slides overlapping windows over a waveform,
  scores each window with the detector already used elsewhere in the project,
  and returns a 1-D array of per-window P(synthetic) values aligned to time.

Attribution method
------------------
This module deliberately uses **direct window scoring** rather than a
gradient-based saliency method.  The choice is intentional and defensible:

* The same :meth:`predict_waveform` path used by the live inference and
  streaming engines is also used here.  That path is already validated
  end-to-end by Phase 2–4 tests; there is no new model logic to introduce
  or verify.

* Gradient-based saliency (SHAP, Integrated Gradients, GradCAM for audio)
  requires a differentiable path through the backbone *and* the classifier
  head, which in VoxGuard's case means backpropagating through a frozen
  HuggingFace transformer and a fitted scikit-learn classifier that has no
  PyTorch autograd graph.  Bridging that gap correctly under competition
  time pressure would itself introduce a new failure surface.

* A sliding-window attribution is model-agnostic: it treats the detector as
  a black box and measures how its output varies as a function of which
  audio segment is being scored.  The interpretation is direct — high
  P(synthetic) on a window means the detector found cloning artefacts *in
  that segment*.  No proxy model, no approximation assumptions.

* The resolution (window_seconds × stride_seconds) is coarse by design.
  It is calibrated to the classifier's minimum useful input length (~0.5 s)
  rather than sample-level precision.  Claiming sub-frame precision would be
  misleading given the backbone's own temporal pooling.

Returned scores of ``None`` for a window indicate that the silence-detection
gate fired (near-zero RMS) or that the detector raised on that window.
Callers can treat ``None`` as "no information" rather than "no synthetic
signal" — they are distinct states.

Usage example::

    from voxguard.classifier.ensemble import WeightedAverageDetector
    from voxguard.explain.attribution import windowed_attribution
    from voxguard.utils.audio_io import load_audio

    waveform, sr = load_audio("call.wav")
    detector = WeightedAverageDetector(...)
    scores, times = windowed_attribution(waveform, sr, detector)
    # scores[i] is P(synthetic) for the window starting at times[i] seconds
"""

from __future__ import annotations

from typing import Any

import numpy as np

from voxguard import config
from voxguard.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Minimum RMS energy for a window to be scored rather than skipped.
# Matches the threshold used in StreamingScorer and VoxGuardDetector._rms_energy.
_SILENCE_THRESHOLD: float = 0.01


def windowed_attribution(
    waveform: np.ndarray,
    sr: int,
    detector: Any,
    window_seconds: float = 0.5,
    stride_seconds: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Score overlapping windows of *waveform* and return ``(scores, timestamps)``.

    Segments the input into overlapping windows of length *window_seconds*,
    advances by *stride_seconds* each step, calls
    ``detector.predict_waveform(window, sr)`` on each window, and returns a
    tuple of two aligned 1-D arrays: per-window P(synthetic) scores and the
    corresponding **window start time** in seconds.

    This is a coarse, model-agnostic attribution method — it explains
    *where* in the clip the detector's signal is strongest by directly
    reusing the validated scoring function, not a separate saliency
    mechanism.  See the module docstring for the full rationale.

    Parameters
    ----------
    waveform:
        1-D float32 array of mono audio samples at sample rate *sr*.
        Multi-channel input is accepted and downmixed to mono before
        windowing.
    sr:
        Sample rate of *waveform* in Hz.
    detector:
        Any object that exposes ``predict_waveform(waveform: np.ndarray, sr: int)``
        returning a dict with a ``"probability_synthetic"`` key.  This is
        the same interface required by ``StreamingScorer`` — both
        ``VoxGuardDetector`` and ``WeightedAverageDetector`` satisfy it.
    window_seconds:
        Duration of each analysis window in seconds.  Defaults to 0.5 s,
        which is the practical minimum for the SSL backbones (shorter
        windows produce meaningless embeddings due to temporal pooling).
    stride_seconds:
        Step size between consecutive window starts in seconds.  Defaults
        to 0.25 s (50 % overlap).  Must be > 0 and ≤ *window_seconds*.

    Returns
    -------
    scores : np.ndarray, shape (n_windows,), dtype float64
        ``scores[i]`` is P(synthetic) ∈ [0, 1] for the *i*-th window, or
        ``np.nan`` if the window was skipped (silence gate fired) or the
        detector raised an exception.  ``np.nan`` means "no information" —
        it is distinct from 0.0, which would mean "definitely real speech".

    timestamps : np.ndarray, shape (n_windows,), dtype float64
        ``timestamps[i]`` is the **start time in seconds** of the *i*-th
        window — i.e. ``starts[i] / sr``, where ``starts[i]`` is the
        sample index of the first sample in that window.

        **Convention: start time, not center time.**
        Start times are used because they align directly with the x-axis of
        ``librosa.display.specshow`` and with the frame indices produced by
        ``librosa.feature.melspectrogram`` (both use left-edge time
        references).  To convert to center times, add ``window_seconds / 2``
        to every element.

        ``timestamps[0]`` is always 0.0 (first window begins at sample 0).
        Consecutive timestamps differ by exactly ``stride_seconds`` (within
        one sample's rounding tolerance).

    Both arrays have the same length ``n_windows``.  When *waveform* is
    shorter than one full window, both arrays are empty (shape ``(0,)``).

    Raises
    ------
    TypeError
        If *detector* does not expose ``predict_waveform``.
    ValueError
        If *waveform* is empty, *sr* ≤ 0, *window_seconds* ≤ 0,
        *stride_seconds* ≤ 0, or *stride_seconds* > *window_seconds*.

    Notes
    -----
    * Windows near the end of the clip that are shorter than
      *window_seconds* are **dropped** rather than zero-padded.  Zero-
      padding would add silence artefacts that distort the score; it is
      more honest to leave those frames unscored.
    * A ``np.nan`` score should be interpreted as "no information for this
      window", not "the window is real speech".  Silence and scoring
      failures are distinct from a low-probability-synthetic result.
    * The function is intentionally stateless — it does not modify or reset
      the detector.  Detectors that maintain internal state (e.g. EMA
      smoothing) should be reset by the caller before and after this call
      if isolation is required.
    * **Time convention summary**: ``timestamps[i] = i * stride_seconds``
      (for integer-aligned strides).  To overlay on a spectrogram produced
      with ``hop_length`` *h* at sample rate *sr*, the spectrogram frame at
      index *k* covers time ``k * h / sr`` — both share the same left-edge
      origin, so no offset correction is needed.

    Examples
    --------
    >>> import numpy as np
    >>> from voxguard.explain.attribution import windowed_attribution
    >>> class _ConstDetector:
    ...     def predict_waveform(self, w, sr):
    ...         return {"probability_synthetic": 0.42, "label": "synthetic"}
    >>> rng = np.random.default_rng(0)
    >>> wav = (rng.standard_normal(16000) * 0.1).astype(np.float32)
    >>> scores, timestamps = windowed_attribution(wav, 16000, _ConstDetector())
    >>> scores.shape == timestamps.shape     # both (n_windows,)
    True
    >>> scores.dtype == timestamps.dtype == np.float64
    True
    >>> float(timestamps[0])                 # first window starts at t=0
    0.0
    >>> float(scores[0])                     # P(synthetic) for first window
    0.42
    >>> # Convert to center times if needed:
    >>> center_times = timestamps + 0.5 / 2  # window_seconds=0.5 default
    """
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if not hasattr(detector, "predict_waveform"):
        raise TypeError(
            "detector must expose predict_waveform(waveform, sr); "
            f"got {type(detector).__name__!r} which has no such attribute."
        )

    waveform = np.asarray(waveform, dtype=np.float32)

    # Downmix stereo/multi-channel to mono
    if waveform.ndim == 2:
        # (channels, samples) or (samples, channels) — assume longer axis is samples
        if waveform.shape[0] <= waveform.shape[1]:
            waveform = waveform.mean(axis=0)
        else:
            waveform = waveform.mean(axis=1)
    elif waveform.ndim != 1:
        raise ValueError(
            f"waveform must be 1-D or 2-D; got shape {waveform.shape}"
        )

    if waveform.size == 0:
        raise ValueError("waveform is empty.")

    if sr <= 0:
        raise ValueError(f"sr must be positive; got {sr!r}.")

    if window_seconds <= 0:
        raise ValueError(f"window_seconds must be > 0; got {window_seconds!r}.")

    if stride_seconds <= 0:
        raise ValueError(f"stride_seconds must be > 0; got {stride_seconds!r}.")

    if stride_seconds > window_seconds:
        raise ValueError(
            f"stride_seconds ({stride_seconds}) must be <= window_seconds "
            f"({window_seconds}); otherwise windows would have gaps."
        )

    # ------------------------------------------------------------------
    # Derive sample counts
    # ------------------------------------------------------------------
    window_samples: int = int(round(window_seconds * sr))
    stride_samples: int = int(round(stride_seconds * sr))

    if window_samples < 1:
        raise ValueError(
            f"window_seconds={window_seconds} at sr={sr} gives "
            f"{window_samples} samples — too short to score."
        )

    n_total = len(waveform)

    # Collect window start positions (drop trailing incomplete windows)
    starts = list(range(0, n_total - window_samples + 1, stride_samples))

    if not starts:
        logger.warning(
            "windowed_attribution: waveform (%d samples, %.2f s) is shorter "
            "than one window (%d samples, %.2f s) — returning empty arrays.",
            n_total,
            n_total / sr,
            window_samples,
            window_seconds,
        )
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    n_windows = len(starts)
    scores = np.full(n_windows, np.nan, dtype=np.float64)
    times = np.array(starts, dtype=np.float64) / sr

    logger.debug(
        "windowed_attribution: %d samples (%.2f s), window=%.2f s, "
        "stride=%.2f s → %d windows",
        n_total,
        n_total / sr,
        window_seconds,
        stride_seconds,
        n_windows,
    )

    # ------------------------------------------------------------------
    # Score each window
    # ------------------------------------------------------------------
    n_skipped_silence = 0
    n_skipped_error = 0

    for i, start in enumerate(starts):
        window = waveform[start : start + window_samples]

        # Silence gate — matches the threshold in StreamingScorer and
        # VoxGuardDetector to ensure consistent behaviour across the project.
        rms = float(np.sqrt(np.mean(window.astype(np.float64) ** 2)))
        if rms < _SILENCE_THRESHOLD:
            n_skipped_silence += 1
            continue  # scores[i] stays np.nan

        try:
            result = detector.predict_waveform(window, sr)
            prob = result.get("probability_synthetic")
            if prob is None:
                # Detector itself returned None (e.g. its own silence gate
                # fired on a window that passed our RMS check due to a
                # slightly different threshold).
                n_skipped_error += 1
            else:
                scores[i] = float(prob)
        except Exception as exc:
            n_skipped_error += 1
            logger.debug(
                "windowed_attribution: window %d (t=%.2f s) raised %s: %s",
                i,
                times[i],
                type(exc).__name__,
                exc,
            )

    n_scored = int(np.sum(~np.isnan(scores)))
    logger.debug(
        "windowed_attribution: scored=%d  silence_skipped=%d  error_skipped=%d",
        n_scored,
        n_skipped_silence,
        n_skipped_error,
    )

    return scores, times
