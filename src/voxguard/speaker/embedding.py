"""
embedding.py — speaker voiceprint embedding extraction (Phase 5).

Wraps two interchangeable speaker-embedding backends behind a single
interface:

  - "speechbrain" (default): SpeechBrain's ECAPA-TDNN
    (``speechbrain/spkrec-ecapa-voxceleb``), 192-dim embeddings.
  - "pyannote": ``pyannote.audio``'s ``Inference`` API over
    ``pyannote/embedding``, 512-dim embeddings.

Both backends expose the same ``SpeakerEmbedder.extract(waveform, sr) ->
np.ndarray`` interface, so callers never write backend-specific code.
"""

from __future__ import annotations

from typing import Optional

import librosa
import numpy as np

from voxguard import config
from voxguard.utils.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_SPEECHBRAIN_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_PYANNOTE_SOURCE = "pyannote/embedding"

# Speaker-embedding models need enough signal to characterize a voice;
# below this they degrade sharply (an empirical floor, not a hard
# architectural limit shared by both backends).
MIN_DURATION_SECONDS = 1.0


class SpeakerEmbedder:
    """Extracts a fixed-length speaker-identity embedding ("voiceprint").

    Two interchangeable backends, both exposing the same ``extract()``
    interface so calling code never needs to know which one is active:

      - ``"speechbrain"`` (default): ECAPA-TDNN via
        ``speechbrain.inference.speaker.EncoderClassifier``
        (``speechbrain/spkrec-ecapa-voxceleb``), 192-dim.
      - ``"pyannote"``: ``pyannote.audio``'s ``Inference`` API over
        ``pyannote/embedding``, 512-dim.

    If the SpeechBrain backend is requested but fails to load (e.g. a
    HuggingFace download issue), the constructor prints a clear message,
    automatically falls back to the pyannote backend, and logs which
    backend ended up active — construction only raises if *both* the
    requested backend and the fallback fail.

    Parameters
    ----------
    backend:
        ``"speechbrain"`` (default) or ``"pyannote"``.
    device:
        Torch device string. Defaults to ``config.get_device()``.

    Raises
    ------
    ValueError
        If *backend* is neither ``"speechbrain"`` nor ``"pyannote"``.
    """

    def __init__(self, backend: str = "speechbrain", device: Optional[str] = None) -> None:
        if backend not in ("speechbrain", "pyannote"):
            raise ValueError(
                f"Unknown backend {backend!r}; expected 'speechbrain' or 'pyannote'."
            )

        self.requested_backend = backend
        self.device = device or config.get_device()
        self._model = None
        self.backend: str = backend
        self.embedding_dim: int = 0

        if backend == "speechbrain":
            try:
                self._load_speechbrain()
            except Exception as exc:
                message = (
                    f"[SpeakerEmbedder] Could not load the SpeechBrain ECAPA-TDNN model "
                    f"'{DEFAULT_SPEECHBRAIN_SOURCE}' ({exc}). Falling back to "
                    f"backend='pyannote' automatically."
                )
                print(message)
                logger.warning(
                    "SpeechBrain speaker-embedding backend failed to load (%s); "
                    "falling back to pyannote.",
                    exc,
                )
                self._load_pyannote()
                self.backend = "pyannote"
        else:
            self._load_pyannote()

        logger.info(
            "SpeakerEmbedder ready: backend=%s (requested=%s), embedding_dim=%d, device=%s",
            self.backend,
            self.requested_backend,
            self.embedding_dim,
            self.device,
        )

    def _load_speechbrain(self) -> None:
        """Loads SpeechBrain's ECAPA-TDNN speaker-embedding model."""
        from speechbrain.inference.speaker import EncoderClassifier

        savedir = str(config.MODELS_DIR / "pretrained" / "spkrec-ecapa-voxceleb")
        self._model = EncoderClassifier.from_hparams(
            source=DEFAULT_SPEECHBRAIN_SOURCE,
            savedir=savedir,
            run_opts={"device": self.device},
        )
        self.embedding_dim = 192

    def _load_pyannote(self) -> None:
        """Loads the pyannote.audio embedding model via its Inference API."""
        try:
            from pyannote.audio import Inference, Model
        except Exception as exc:
            raise ImportError(
                "pyannote.audio is not installed or failed to import. Install it with "
                "`pip install pyannote.audio`, accept the pyannote/embedding model terms "
                "on HuggingFace, and set an auth token (`huggingface-cli login`) to use "
                "backend='pyannote'."
            ) from exc

        model = Model.from_pretrained(DEFAULT_PYANNOTE_SOURCE)
        inference = Inference(model, window="whole")
        try:
            import torch

            inference.to(torch.device(self.device))
        except Exception:
            # Device placement is a best-effort optimization; pyannote's
            # Inference API has changed across versions, and falling back
            # to its own default device is preferable to failing here.
            pass

        self._model = inference
        self.embedding_dim = 512

    def extract(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Extracts a 1-D speaker embedding ("voiceprint") from *waveform*.

        Parameters
        ----------
        waveform:
            1-D array of audio samples.
        sr:
            Sample rate of *waveform* in Hz. Resampled to
            ``config.SAMPLE_RATE`` (16000 Hz) first if it differs.

        Returns
        -------
        np.ndarray, shape ``(embedding_dim,)``
            192-dim for the speechbrain backend, 512-dim for pyannote.

        Raises
        ------
        ValueError
            If *waveform* is shorter than ``MIN_DURATION_SECONDS`` — rather
            than letting the backend fail on it cryptically (e.g. a NaN
            embedding or an opaque shape error from a near-empty batch).
        """
        waveform = np.asarray(waveform, dtype=np.float32)
        duration = len(waveform) / sr if sr > 0 else 0.0

        if duration < MIN_DURATION_SECONDS:
            raise ValueError(
                f"Audio too short for a reliable speaker embedding: {duration:.2f}s "
                f"(minimum recommended: {MIN_DURATION_SECONDS:.1f}s). Speaker-embedding "
                "models need enough signal to characterize a voice — pass a longer clip "
                "rather than a short chunk."
            )

        if sr != config.SAMPLE_RATE:
            waveform = librosa.resample(
                waveform, orig_sr=sr, target_sr=config.SAMPLE_RATE
            ).astype(np.float32)
            sr = config.SAMPLE_RATE

        if self.backend == "speechbrain":
            return self._extract_speechbrain(waveform)
        return self._extract_pyannote(waveform, sr)

    def _extract_speechbrain(self, waveform: np.ndarray) -> np.ndarray:
        import torch

        wav_tensor = torch.from_numpy(waveform).unsqueeze(0)  # (1, T)
        with torch.no_grad():
            embedding = self._model.encode_batch(wav_tensor)  # (1, 1, D)
        return embedding.squeeze().cpu().numpy().astype(np.float32)

    def _extract_pyannote(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        import torch

        wav_tensor = torch.from_numpy(waveform).unsqueeze(0)  # (1, T)
        embedding = self._model({"waveform": wav_tensor, "sample_rate": sr})
        return np.asarray(embedding, dtype=np.float32).reshape(-1)

    def __repr__(self) -> str:
        return (
            f"SpeakerEmbedder(backend={self.backend!r}, "
            f"embedding_dim={self.embedding_dim}, device={self.device!r})"
        )
