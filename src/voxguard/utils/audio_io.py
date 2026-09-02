"""
audio_io.py — shared audio loading, saving, and inspection helpers.

Every VoxGuard module that needs to read or write audio should import
from here.  Keeping I/O logic in one place means resampling, mono
downmixing, and dtype normalisation happen consistently across the whole
pipeline — no silent mismatches between feature extraction, embedding,
and streaming inference.

Dependencies: soundfile (fast read/write, no ffmpeg needed) + librosa
(resampling, mono downmix).  No subprocess calls to ffmpeg are made here;
soundfile handles the common formats (WAV, FLAC, OGG, AIFF) natively via
libsndfile, which is sufficient for the dataset formats used in this project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import librosa
import numpy as np
import soundfile as sf

from voxguard.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Type alias accepted by all path parameters
PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_audio(
    path: PathLike,
    target_sr: int = 16_000,
) -> tuple[np.ndarray, int]:
    """Load an audio file, resample to *target_sr*, and downmix to mono.

    Uses ``soundfile`` for the initial read (fast, no subprocess) and
    ``librosa.resample`` when the file's native sample rate differs from
    *target_sr*.

    Parameters
    ----------
    path:
        Path to the audio file.  Formats supported by libsndfile are
        accepted (WAV, FLAC, OGG, AIFF, …).
    target_sr:
        Desired output sample rate in Hz.  Defaults to
        ``config.SAMPLE_RATE`` (16 000 Hz), the native rate of the SSL
        embedding backbones used in Phase 1.

    Returns
    -------
    waveform : np.ndarray, shape ``(n_samples,)``, dtype ``float32``
        Mono audio normalised to the range ``[-1.0, 1.0]``.
    sr : int
        The output sample rate (always equal to *target_sr*).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    RuntimeError
        If soundfile cannot decode the file (e.g. unsupported codec).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    logger.debug("Loading audio: %s", path)

    # soundfile returns (samples, channels) for multi-channel files and
    # (samples,) for mono files.  always_2d=False keeps mono files as 1-D.
    try:
        waveform, native_sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:
        raise RuntimeError(f"soundfile could not read '{path}': {exc}") from exc

    # --- Downmix to mono -------------------------------------------------------
    if waveform.ndim == 2:
        # Shape is (n_samples, n_channels) — average across channels.
        logger.debug(
            "Downmixing %d channels to mono: %s", waveform.shape[1], path.name
        )
        waveform = waveform.mean(axis=1)

    # waveform is now 1-D float32

    # --- Resample if necessary -------------------------------------------------
    if native_sr != target_sr:
        logger.debug(
            "Resampling %s from %d Hz → %d Hz", path.name, native_sr, target_sr
        )
        waveform = librosa.resample(
            waveform,
            orig_sr=native_sr,
            target_sr=target_sr,
            res_type="kaiser_best",  # high-quality sinc resampler
        )

    # Ensure float32 (librosa.resample can return float64 in some versions)
    waveform = waveform.astype(np.float32)

    logger.debug(
        "Loaded '%s': %d samples @ %d Hz (%.2f s)",
        path.name,
        len(waveform),
        target_sr,
        len(waveform) / target_sr,
    )
    return waveform, target_sr


def save_audio(path: PathLike, waveform: np.ndarray, sr: int) -> None:
    """Write a mono float32 waveform to disk as a WAV file.

    The parent directory is created automatically if it does not exist.

    Parameters
    ----------
    path:
        Destination file path.  The extension determines the format;
        ``.wav`` is recommended and always safe.  soundfile supports
        ``.flac`` and ``.ogg`` as well.
    waveform:
        1-D ``float32`` array of audio samples in the range ``[-1.0, 1.0]``.
        Values outside this range will be clipped by libsndfile on write.
    sr:
        Sample rate of *waveform* in Hz.

    Raises
    ------
    ValueError
        If *waveform* is not 1-D or not float32.
    RuntimeError
        If soundfile cannot write the file (e.g. bad extension or path).
    """
    path = Path(path)

    if waveform.ndim != 1:
        raise ValueError(
            f"save_audio expects a 1-D waveform; got shape {waveform.shape}. "
            "Downmix to mono before saving."
        )
    if waveform.dtype != np.float32:
        logger.debug(
            "Casting waveform dtype %s → float32 before saving.", waveform.dtype
        )
        waveform = waveform.astype(np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        sf.write(str(path), waveform, samplerate=sr)
    except Exception as exc:
        raise RuntimeError(f"soundfile could not write '{path}': {exc}") from exc

    logger.debug(
        "Saved audio: %s (%d samples @ %d Hz)", path.name, len(waveform), sr
    )


def get_duration_seconds(path: PathLike) -> float:
    """Return the duration of an audio file in seconds without loading samples.

    Uses ``soundfile.info()`` to read only the file header — no audio data
    is decoded into memory.  This makes it cheap to call on large files or
    to scan a whole dataset directory.

    Parameters
    ----------
    path:
        Path to the audio file.

    Returns
    -------
    float
        Duration in seconds.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    RuntimeError
        If soundfile cannot read the file's header.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        info = sf.info(str(path))
    except Exception as exc:
        raise RuntimeError(
            f"soundfile could not read header of '{path}': {exc}"
        ) from exc

    duration: float = info.frames / info.samplerate
    logger.debug("Duration of '%s': %.3f s", path.name, duration)
    return duration
