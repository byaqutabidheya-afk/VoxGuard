"""
enrollment.py — local speaker voiceprint enrollment store (Phase 5).

Averages a handful of reference clips into one voiceprint vector per
speaker and persists it as a single ``.npy`` file under
``models/voiceprints/``. Intentionally simple local file storage — a
production enrollment database is explicitly out of scope for this
project; one or two enrolled demo voiceprints is enough to prove the
mechanism end-to-end.

Privacy note: ``enroll_speaker`` stores ONLY the averaged embedding
vector, never the raw reference audio. Once enrollment completes, the
reference clips can be deleted from disk immediately — the voiceprint
keeps working without them, since nothing here depends on the original
audio persisting anywhere. This is also what makes ``delete_speaker`` a
genuine right-to-erasure mechanism (Phase 11): removing a speaker's
single voiceprint file removes every trace this system retains of their
voice.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from voxguard import config
from voxguard.speaker.embedding import SpeakerEmbedder
from voxguard.utils.audio_io import load_audio
from voxguard.utils.logging_utils import get_logger

logger = get_logger(__name__)

VOICEPRINTS_DIR = config.MODELS_DIR / "voiceprints"


def _voiceprint_path(name: str) -> Path:
    """Resolves a speaker name to its voiceprint file path.

    Rejects empty names and path separators so a typo'd *name* can't
    accidentally read/write outside ``VOICEPRINTS_DIR``.
    """
    if not name or not name.strip():
        raise ValueError("Speaker name must be a non-empty string.")
    if "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError(f"Speaker name must not contain path separators: {name!r}")
    return VOICEPRINTS_DIR / f"{name}.npy"


def enroll_speaker(
    name: str, reference_clips: List[str], embedder: SpeakerEmbedder
) -> None:
    """Enrolls a speaker by averaging embeddings from multiple reference clips.

    Recommend 2-3 clips of 5-10 seconds each per speaker: averaging over
    several clips produces a more robust voiceprint than relying on a
    single clip, smoothing out clip-specific noise and recording
    variation while still characterizing the speaker's voice.

    Privacy: this stores ONLY the resulting averaged embedding vector, at
    ``models/voiceprints/{name}.npy`` — the raw reference audio is never
    written or retained by this function. *reference_clips* can be
    deleted from disk immediately after this call returns; the saved
    voiceprint keeps working without them.

    Parameters
    ----------
    name:
        Speaker identifier. Used verbatim as the voiceprint filename
        (``models/voiceprints/{name}.npy``), so keep it filesystem-safe
        (no path separators).
    reference_clips:
        Paths to one or more reference audio clips for this speaker.
    embedder:
        A loaded ``SpeakerEmbedder`` used to extract each clip's
        embedding.

    Raises
    ------
    ValueError
        If *name* is empty/unsafe, or *reference_clips* is empty.
    """
    if not reference_clips:
        raise ValueError(
            f"enroll_speaker requires at least one reference clip for '{name}'."
        )

    path = _voiceprint_path(name)

    embeddings = []
    for clip_path in reference_clips:
        waveform, sr = load_audio(clip_path, target_sr=config.SAMPLE_RATE)
        embeddings.append(embedder.extract(waveform, sr))

    voiceprint = np.mean(np.stack(embeddings), axis=0).astype(np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, voiceprint)

    logger.info(
        "Enrolled speaker '%s' from %d reference clip(s) -> %s (embedding_dim=%d)",
        name,
        len(reference_clips),
        path,
        voiceprint.shape[0],
    )


def list_enrolled_speakers() -> List[str]:
    """Lists the names of all currently enrolled speakers.

    Returns
    -------
    list[str]
        Speaker names (voiceprint filenames without the ``.npy``
        extension), sorted alphabetically. Empty if no speakers are
        enrolled or the voiceprints directory doesn't exist yet.
    """
    if not VOICEPRINTS_DIR.is_dir():
        return []
    return sorted(p.stem for p in VOICEPRINTS_DIR.glob("*.npy"))


def load_voiceprint(name: str) -> np.ndarray:
    """Loads a previously enrolled speaker's voiceprint.

    Parameters
    ----------
    name:
        Speaker identifier, as passed to ``enroll_speaker``.

    Returns
    -------
    np.ndarray
        The averaged embedding vector.

    Raises
    ------
    FileNotFoundError
        If no voiceprint is enrolled under *name*.
    """
    path = _voiceprint_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"No enrolled voiceprint found for '{name}' at {path}. "
            f"Enrolled speakers: {list_enrolled_speakers()}"
        )
    return np.load(path)


def delete_speaker(name: str) -> bool:
    """Removes an enrolled speaker's voiceprint (right-to-erasure mechanism).

    This is the deletion path referenced in Phase 11's privacy
    documentation: since ``enroll_speaker`` never retains raw reference
    audio, deleting the single ``.npy`` file this function removes erases
    every trace this system keeps of that speaker's voice.

    Parameters
    ----------
    name:
        Speaker identifier to remove.

    Returns
    -------
    bool
        ``True`` if a voiceprint was found and deleted, ``False`` if no
        voiceprint was enrolled under *name* — deleting an
        already-absent enrollment is a no-op, not an error.
    """
    path = _voiceprint_path(name)
    if not path.exists():
        logger.info(
            "delete_speaker('%s'): no voiceprint found at %s; nothing to do.", name, path
        )
        return False

    path.unlink()
    logger.info("Deleted voiceprint for speaker '%s' (%s).", name, path)
    return True
