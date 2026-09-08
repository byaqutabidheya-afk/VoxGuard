"""spectrogram.py — mel-spectrogram generation and image rendering.

Provides two public functions:

* :func:`generate_mel_spectrogram` — converts a raw waveform to a
  log-scaled (dB) mel-spectrogram array ready for downstream processing
  or visualisation.

* :func:`render_spectrogram_image` — renders that array to a PNG file via
  ``librosa.display.specshow`` and matplotlib, for use in the Gradio
  explainability overlay.

Design notes
------------
* matplotlib is imported *inside* :func:`render_spectrogram_image` so that
  modules that only call :func:`generate_mel_spectrogram` (e.g. test
  helpers, feature pipelines) do not pay the import cost or require a
  display.
* The ``Agg`` backend is selected before importing ``pyplot`` to ensure the
  function is safe in headless server environments (Gradio workers, CI
  runners, Kaggle notebooks).  If a caller has already imported pyplot with
  a different backend this is a no-op, which is the correct behaviour.
* :func:`render_spectrogram_image` always closes the figure it creates,
  preventing memory leaks when called repeatedly during a streaming session.
* ``output_path`` accepts both ``str`` and ``pathlib.Path``.  The parent
  directory is created automatically so callers do not need to mkdir first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import librosa
import numpy as np

from voxguard import config
from voxguard.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Default mel-bank parameters — kept as module constants so they can be
# referenced in tests without re-deriving them from function defaults.
DEFAULT_N_MELS: int = 128
DEFAULT_HOP_LENGTH: int = 512
DEFAULT_N_FFT: int = 2048

# power_to_db reference value: use the max power in each spectrogram so the
# colour scale is always relative to the clip's own loudest bin (avoids
# washing out quiet clips and slamming loud ones into a narrow range).
_DB_REF = np.max


def generate_mel_spectrogram(
    waveform: np.ndarray,
    sr: int,
    n_mels: int = DEFAULT_N_MELS,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_fft: int = DEFAULT_N_FFT,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Convert a waveform to a log-scaled mel-spectrogram.

    Parameters
    ----------
    waveform:
        1-D float32 array of mono audio samples.  Stereo input is accepted
        but will be downmixed to mono before processing (same convention as
        ``voxguard.utils.audio_io.load_audio``).
    sr:
        Sample rate of *waveform* in Hz.  Does **not** need to be
        ``config.SAMPLE_RATE`` — the mel filter bank is computed for the
        actual rate supplied.
    n_mels:
        Number of mel filter bank channels.  Defaults to 128.
    hop_length:
        STFT hop length in samples.  Defaults to 512, giving a frame rate
        of ``sr / hop_length`` frames per second (31.25 fps at 16 kHz).
    n_fft:
        FFT window size in samples.  Defaults to 2048.
    fmin:
        Lowest frequency for the mel filter bank in Hz.  Defaults to 0.
    fmax:
        Highest frequency for the mel filter bank in Hz.  ``None`` (the
        default) sets it to ``sr / 2`` (Nyquist).

    Returns
    -------
    np.ndarray
        2-D float32 array of shape ``(n_mels, n_frames)`` containing the
        mel-spectrogram in dB, where 0 dB is the maximum power bin of the
        clip and the dynamic range is clipped at −80 dB (librosa's default
        ``top_db=80``).

    Raises
    ------
    ValueError
        If *waveform* is empty or all-zero (no energy to transform).
    """
    waveform = np.asarray(waveform, dtype=np.float32)

    # Downmix stereo to mono
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1 if waveform.shape[0] > waveform.shape[1] else 0)
    elif waveform.ndim > 2:
        raise ValueError(
            f"waveform must be 1-D or 2-D; got shape {waveform.shape}"
        )

    waveform = waveform.ravel()

    if waveform.size == 0:
        raise ValueError("waveform is empty — cannot compute spectrogram.")

    if not np.any(waveform):
        raise ValueError(
            "waveform contains only zeros — spectrogram would be uninformative. "
            "Check the silence-detection gate upstream."
        )

    mel_power = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )

    mel_db = librosa.power_to_db(mel_power, ref=_DB_REF)

    logger.debug(
        "generate_mel_spectrogram: sr=%d n_mels=%d hop_length=%d "
        "n_fft=%d → shape=%s min=%.1f dB max=%.1f dB",
        sr,
        n_mels,
        hop_length,
        n_fft,
        mel_db.shape,
        float(mel_db.min()),
        float(mel_db.max()),
    )

    return mel_db.astype(np.float32)


def render_spectrogram_image(
    mel_spec: np.ndarray,
    sr: int,
    hop_length: int = DEFAULT_HOP_LENGTH,
    output_path: Union[str, Path] = "spectrogram.png",
    fmin: float = 0.0,
    fmax: float | None = None,
    figsize: tuple[float, float] = (10.0, 4.0),
    dpi: int = 150,
    colormap: str = "magma",
    title: str = "Mel Spectrogram",
) -> Path:
    """Render a log-mel spectrogram array to a PNG file.

    Uses ``librosa.display.specshow`` for axis labelling (time on x,
    mel-frequency on y) and ``matplotlib`` for figure management.

    The function always closes the figure it creates, so repeated calls in
    a streaming session do not accumulate figures in memory.

    Parameters
    ----------
    mel_spec:
        2-D array ``(n_mels, n_frames)`` as returned by
        :func:`generate_mel_spectrogram`.
    sr:
        Sample rate of the original audio in Hz.  Used by ``specshow`` to
        compute the time axis.
    hop_length:
        STFT hop length used when computing *mel_spec*.  Must match the
        value passed to :func:`generate_mel_spectrogram` or the time axis
        will be incorrect.
    output_path:
        Destination path for the PNG file.  Parent directories are created
        automatically.  Accepts ``str`` or ``pathlib.Path``.
    fmin:
        Lowest frequency of the mel filter bank (Hz), passed to
        ``specshow`` for correct y-axis labelling.
    fmax:
        Highest frequency of the mel filter bank (Hz).  ``None`` → Nyquist.
    figsize:
        ``(width, height)`` of the figure in inches.  Defaults to
        ``(10, 4)`` — wide enough to show a few seconds of speech clearly.
    dpi:
        Dots per inch for the saved PNG.  Defaults to 150, giving a
        1500 × 600 px output at the default figsize.
    colormap:
        Matplotlib colormap name.  Defaults to ``"magma"``, which is
        perceptually uniform and prints readably in greyscale.
    title:
        Figure title string.

    Returns
    -------
    pathlib.Path
        The resolved absolute path of the saved PNG.

    Raises
    ------
    ValueError
        If *mel_spec* is not 2-D.
    """
    mel_spec = np.asarray(mel_spec, dtype=np.float32)
    if mel_spec.ndim != 2:
        raise ValueError(
            f"mel_spec must be a 2-D array (n_mels × n_frames); "
            f"got shape {mel_spec.shape}"
        )

    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # Use the non-interactive Agg backend.  matplotlib.use() is a no-op if
    # a backend has already been set — it does not raise or reset pyplot state.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa.display

    fig, ax = plt.subplots(figsize=figsize)
    try:
        img = librosa.display.specshow(
            mel_spec,
            sr=sr,
            hop_length=hop_length,
            x_axis="time",
            y_axis="mel",
            fmin=fmin,
            fmax=fmax,
            cmap=colormap,
            ax=ax,
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB", label="Power (dB)")
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (mel)")
        fig.tight_layout()
        fig.savefig(out, dpi=dpi, bbox_inches="tight")

        logger.debug(
            "render_spectrogram_image: saved %s (shape=%s sr=%d hop=%d dpi=%d)",
            out,
            mel_spec.shape,
            sr,
            hop_length,
            dpi,
        )
    finally:
        # Always release memory — critical when called per-chunk in streaming.
        plt.close(fig)

    return out
