"""overlay.py — combined mel-spectrogram + attribution heatmap figure.

Provides one public function:

* :func:`render_explainability_overlay` — generates a mel-spectrogram,
  runs windowed attribution, resamples the per-window scores onto the
  spectrogram's dense time axis, and renders a combined matplotlib figure
  with the spectrogram as the base layer and a semi-transparent red
  heatmap showing synthetic-likelihood by region.

NaN handling
------------
``windowed_attribution`` can return ``np.nan`` for any window that was
skipped by the silence gate or where the detector raised.  This is normal
— trailing silence is common, leading silence occurs frequently in
uploaded files.

Strategy: interpolate scores over **valid** (non-NaN) windows, but track
*separately* which frames actually fall within the time span those valid
windows cover, and force everything outside that span fully transparent —
regardless of what value the interpolation math produced for it.

* ``np.interp`` clamps to the nearest valid neighbour at extrapolation
  edges, which is fine for computing *a* number but wrong for *display*:
  a trailing-silence region isn't "whatever the last scored window
  returned" — a previous version of this code rendered it as exactly that,
  and when the last valid window happened to score 1.0 (confidently
  synthetic), the entire silent tail rendered as solid, high-alpha red —
  reading as "very confidently synthetic" for a region with no signal at
  all. ``_interpolate_scores_to_frames`` now also returns a per-frame
  ``frame_alpha_mask`` that is 0.0 for every frame before the first valid
  window or after the last one, so those regions always render at
  alpha=0 (plain spectrogram, no colour) no matter what score value was
  extrapolated there.

* If **all** windows are NaN (entire clip is silence or detector failure),
  the mask is zero everywhere — same effect, degenerately applied to the
  whole clip — and a note is added to the figure title so the viewer
  knows attribution was unavailable.

This means NaN propagation into a crashed/blank render is impossible, and
"no data" can never be mistaken for "confidently synthetic."
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np

from voxguard.explain.attribution import windowed_attribution
from voxguard.explain.describe import describe_attribution
from voxguard.explain.spectrogram import (

    DEFAULT_HOP_LENGTH,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    generate_mel_spectrogram,
)
from voxguard.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _interpolate_scores_to_frames(
    scores: np.ndarray,
    timestamps: np.ndarray,
    frame_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Resample per-window attribution scores onto the spectrogram frame grid.

    Parameters
    ----------
    scores:
        1-D float64 array of per-window P(synthetic), may contain NaN.
    timestamps:
        1-D float64 array of window start times in seconds, same length
        as *scores*.
    frame_times:
        1-D float64 array of spectrogram frame centre/start times in
        seconds, length = n_frames.

    Returns
    -------
    frame_scores : np.ndarray, shape (n_frames,), dtype float64
        Attribution scores resampled onto *frame_times* via linear
        interpolation between valid neighbours. Guaranteed NaN-free, but
        frames outside the valid-window time span are extrapolated
        (clamped to the nearest edge score) and must not be displayed
        without applying *frame_alpha_mask* below — the clamped value can
        be arbitrarily high or low and does not mean anything for a region
        with no actual data.
    frame_alpha_mask : np.ndarray, shape (n_frames,), dtype float64
        ``1.0`` for frames within the time span covered by valid windows
        (interpolation, not extrapolation — legitimate to colour);
        ``0.0`` for frames before the first or after the last valid
        window (leading/trailing silence or detector failure — genuinely
        "no data" and must render fully transparent regardless of
        *frame_scores*' value there).
    any_valid : bool
        ``True`` if at least one valid (non-NaN) score was available.
        ``False`` means the entire clip was silent / unscored — callers
        should suppress the heatmap overlay entirely.
    """
    if len(scores) == 0:
        zeros = np.zeros(len(frame_times), dtype=np.float64)
        return zeros, zeros.copy(), False

    valid_mask = ~np.isnan(scores)
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        logger.debug(
            "_interpolate_scores_to_frames: all %d windows are NaN — "
            "returning zero-alpha frame scores.",
            len(scores),
        )
        zeros = np.zeros(len(frame_times), dtype=np.float64)
        return zeros, zeros.copy(), False

    valid_times = timestamps[valid_mask]
    valid_scores = scores[valid_mask]

    # np.interp extrapolates by clamping (left/right fill). That clamped
    # *value* is kept (it's a harmless number), but frame_alpha_mask below
    # is what actually gates whether it's ever drawn — extrapolated frames
    # always get alpha=0, so a high clamped value can no longer render as
    # a confident-looking block over silence.
    frame_scores = np.interp(frame_times, valid_times, valid_scores)

    first_valid_t = float(valid_times.min())
    last_valid_t = float(valid_times.max())
    frame_alpha_mask = np.where(
        (frame_times >= first_valid_t) & (frame_times <= last_valid_t), 1.0, 0.0
    ).astype(np.float64)

    logger.debug(
        "_interpolate_scores_to_frames: %d valid/%d windows -> "
        "%d frame scores  min=%.3f max=%.3f  masked-out (no-data) frames=%d/%d",
        n_valid,
        len(scores),
        len(frame_scores),
        float(frame_scores.min()),
        float(frame_scores.max()),
        int((frame_alpha_mask == 0.0).sum()),
        len(frame_alpha_mask),
    )
    return frame_scores, frame_alpha_mask, True


def render_explainability_overlay(
    waveform: np.ndarray,
    sr: int,
    detector: Any,
    output_path: Union[str, Path] = "overlay.png",
    n_mels: int = DEFAULT_N_MELS,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_fft: int = DEFAULT_N_FFT,
    window_seconds: float = 1.5,
    stride_seconds: float = 0.75,
    heatmap_alpha: float = 0.45,
    figsize: tuple[float, float] = (12.0, 4.5),
    dpi: int = 150,
    title: str = "VoxGuard — Mel Spectrogram + Synthetic-Likelihood Overlay",
) -> str:
    """Render a mel-spectrogram with a synthetic-likelihood heatmap overlay.

    Pipeline
    --------
    1. Compute the log-mel spectrogram of *waveform* via
       :func:`~voxguard.explain.spectrogram.generate_mel_spectrogram`.
    2. Run :func:`~voxguard.explain.attribution.windowed_attribution` with
       the same *detector* used throughout the project.
    3. Interpolate the per-window scores (which may contain ``np.nan`` for
       silent windows) onto the dense spectrogram time axis, and separately
       compute a per-frame alpha mask marking which frames actually fall
       within scored regions — see *NaN handling* in the module docstring.
    4. Render with matplotlib:
       * Base layer: spectrogram via ``librosa.display.specshow`` (magma
         colormap). Its axes limits are captured immediately afterward —
         the overlay reuses them exactly (see the inline comment at the
         overlay ``imshow`` call) so the two layers share one coordinate
         system and neither rescales the other out of view.
       * Overlay layer: ``imshow`` of the 2-D score heatmap (red channel =
         score, alpha = *heatmap_alpha* × score × alpha-mask), so
         low-confidence regions are nearly transparent, high-confidence
         regions are visibly red but never opaque, and no-data regions
         (leading/trailing silence) are fully transparent regardless of
         score.
       * Two colorbars: one for power (dB), one for synthetic likelihood.
       * Always calls ``plt.close(fig)`` in a ``finally`` block to prevent
         memory leaks on repeated calls.

    Parameters
    ----------
    waveform:
        1-D or 2-D float32 array of audio samples.  Stereo is downmixed
        to mono internally.
    sr:
        Sample rate in Hz.
    detector:
        Any object exposing ``predict_waveform(waveform, sr)`` that returns
        a dict with ``"probability_synthetic"``.  Both ``VoxGuardDetector``
        and ``WeightedAverageDetector`` qualify.
    output_path:
        Destination path for the PNG.  Parent directories are created
        automatically.  Accepts ``str`` or ``pathlib.Path``.  Returns the
        resolved path as a ``str`` for Gradio compatibility.
    n_mels:
        Number of mel filter bank channels (default 128).
    hop_length:
        STFT hop size in samples (default 512).  Must match between
        spectrogram and the ``x_axis="time"`` argument to ``specshow``.
    n_fft:
        FFT window size in samples (default 2048).
    window_seconds:
        Attribution window length in seconds.  Defaults to 1.5 s —
        empirically verified on the ``soumya_neutral_01`` real/synthetic
        pair to produce a correctly-directioned, visually clear overlay.
        The originally speculative 0.5 s default did not reliably
        discriminate real from synthetic on that pair or other tested clips
        (short windows give the classifier too little context per step,
        adding attribution noise without adding real signal).
        Still fully overridable per call.
    stride_seconds:
        Attribution window stride in seconds.  Defaults to 0.75 s (50 %
        overlap with the 1.5 s window) — empirically validated alongside
        *window_seconds* on the same test pairs.  Still fully overridable
        per call.
    heatmap_alpha:
        Maximum alpha of the synthetic-likelihood heatmap overlay.
        Each pixel's actual alpha = ``heatmap_alpha × score``, so a score
        of 1.0 renders at full *heatmap_alpha* and a score of 0.0 is
        fully transparent.  Defaults to 0.45.
    figsize:
        Figure size in inches ``(width, height)``.
    dpi:
        Output PNG resolution.
    title:
        Figure title string.

    Returns
    -------
    str
        The resolved absolute path of the saved PNG, as a ``str`` (Gradio
        ``gr.Image`` accepts a string path directly).

    Raises
    ------
    ValueError
        If *waveform* is empty, all-zero, or shorter than one attribution
        window (which would also be too short for a meaningful spectrogram).
    """
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Mel spectrogram                                                  #
    # ------------------------------------------------------------------ #
    mel_db = generate_mel_spectrogram(
        waveform,
        sr,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft,
    )
    # mel_db shape: (n_mels, n_frames)
    n_frames = mel_db.shape[1]

    # Build the spectrogram's own time axis (left-edge convention, matching
    # timestamps from windowed_attribution — no offset correction needed).
    frame_times = np.arange(n_frames, dtype=np.float64) * hop_length / sr

    # ------------------------------------------------------------------ #
    # 2. Windowed attribution                                             #
    # ------------------------------------------------------------------ #
    scores, timestamps = windowed_attribution(
        waveform,
        sr,
        detector,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
    )

    # ------------------------------------------------------------------ #
    # 3. Interpolate scores → per-frame scores                           #
    # ------------------------------------------------------------------ #
    frame_scores, frame_alpha_mask, any_valid = _interpolate_scores_to_frames(
        scores, timestamps, frame_times
    )
    # frame_scores shape: (n_frames,) — guaranteed NaN-free
    # frame_alpha_mask shape: (n_frames,) — 0.0 over no-data (leading/
    # trailing silence) frames, 1.0 elsewhere; gates display, not colour.

    # Build 2-D heatmap: broadcast scores across all mel bins so the
    # colour varies only along the time axis (the relevant dimension for
    # detecting *when* the voice sounds synthetic).
    # Shape: (n_mels, n_frames)
    heatmap_2d = np.tile(frame_scores, (n_mels, 1))
    mask_2d = np.tile(frame_alpha_mask, (n_mels, 1))

    # ------------------------------------------------------------------ #
    # 4. Render                                                           #
    # ------------------------------------------------------------------ #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import librosa.display

    fig, ax = plt.subplots(figsize=figsize)
    try:
        # --- Base layer: mel spectrogram --------------------------------
        spec_img = librosa.display.specshow(
            mel_db,
            sr=sr,
            hop_length=hop_length,
            x_axis="time",
            y_axis="mel",
            cmap="magma",
            ax=ax,
        )

        # --- Overlay layer: synthetic-likelihood heatmap ----------------
        # CRITICAL: specshow's y_axis="mel" does NOT plot on a [0, n_mels]
        # scale — it plots on the actual mel-warped Hz range (e.g. roughly
        # -12 to 8097 for 128 mels at 16 kHz), which librosa computes
        # internally and sets as the axes' data limits. A previous version
        # of this code used extent=[0, duration, 0, n_mels] for the overlay
        # imshow; matplotlib autoscales an axes' limits to a newly-drawn
        # image's extent, so that call silently RESCALED the y-axis from
        # (-12, 8097) down to (0, 128) — crushing the entire spectrogram
        # (drawn against the old, correct range) into an invisible ~1.6%
        # sliver at the bottom, while the heatmap (now exactly matching the
        # new, wrong range) filled the whole visible plot. That was the
        # actual cause of "the heatmap completely obscures the spectrogram"
        # — not the alpha formula. Capturing and reusing specshow's own
        # limits keeps both layers on the identical coordinate system, so
        # neither rescales the other.
        spec_xlim = ax.get_xlim()
        spec_ylim = ax.get_ylim()
        extent = [spec_xlim[0], spec_xlim[1], spec_ylim[0], spec_ylim[1]]

        if any_valid:
            # Per-pixel alpha: score x heatmap_alpha (capped well below 1.0
            # so the spectrogram always shows through even at score=1.0),
            # then masked to zero over no-data frames regardless of score —
            # see module docstring's NaN handling section.
            alpha_2d = np.clip(heatmap_2d * heatmap_alpha, 0.0, 1.0) * mask_2d

            # Build an RGBA array: red channel = score, alpha = alpha_2d.
            # Shape: (n_mels, n_frames, 4)
            rgba = np.zeros((*heatmap_2d.shape, 4), dtype=np.float32)
            rgba[:, :, 0] = heatmap_2d.astype(np.float32)   # R channel
            rgba[:, :, 3] = alpha_2d.astype(np.float32)     # A channel

            ax.imshow(
                rgba,
                aspect="auto",
                origin="lower",
                extent=extent,
                interpolation="bilinear",
                zorder=2,   # above the spectrogram
            )
            # Belt-and-suspenders: force the limits back explicitly in case
            # of any floating-point autoscale drift, so a future refactor
            # can't silently reintroduce the axis-collision bug above.
            ax.set_xlim(spec_xlim)
            ax.set_ylim(spec_ylim)
            overlay_label = "Synthetic likelihood"
        else:
            # No valid scores — skip the heatmap entirely and note it in title
            title = f"{title}  [attribution unavailable — clip may be silent]"
            overlay_label = None
            logger.warning(
                "render_explainability_overlay: all attribution scores are NaN; "
                "heatmap suppressed.  Output: %s",
                out,
            )

        # --- Colorbars --------------------------------------------------
        cbar_spec = fig.colorbar(
            spec_img, ax=ax, pad=0.01, fraction=0.035, format="%+2.0f dB"
        )
        cbar_spec.set_label("Power (dB)", fontsize=9)

        if any_valid:
            # Synthetic-likelihood colorbar using a plain Reds scale
            sm = plt.cm.ScalarMappable(
                cmap="Reds", norm=mcolors.Normalize(vmin=0.0, vmax=1.0)
            )
            sm.set_array([])
            cbar_heat = fig.colorbar(sm, ax=ax, pad=0.12, fraction=0.035)
            cbar_heat.set_label("Synthetic likelihood by region", fontsize=9)

        # --- Labels and title -------------------------------------------
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Frequency (mel)", fontsize=9)

        fig.tight_layout()
        fig.savefig(out, dpi=dpi, bbox_inches="tight")

        logger.debug(
            "render_explainability_overlay: saved %s "
            "(mel=%s any_valid=%s scores_min=%.3f scores_max=%.3f)",
            out,
            mel_db.shape,
            any_valid,
            float(frame_scores.min()),
            float(frame_scores.max()),
        )

    finally:
        plt.close(fig)

    return str(out)
