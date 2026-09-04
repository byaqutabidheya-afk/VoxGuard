"""
extractor.py — frozen self-supervised speech embedding extraction (Phase 1).

Wraps a pretrained Wav2Vec2 / WavLM backbone (chosen via
``config.EMBEDDING_MODEL_NAME``) to turn raw audio waveforms into fixed-
length embedding vectors. The backbone is never fine-tuned: all parameters
are frozen and the model is kept in eval mode, since Phase 1 only needs it
as a feature extractor for downstream classification.
"""

from __future__ import annotations

import librosa
import numpy as np
import torch

from voxguard import config
from voxguard.utils.logging_utils import get_logger

logger = get_logger(__name__)


class EmbeddingExtractor:
    """Extracts fixed-length embeddings from raw audio using a frozen SSL backbone.

    Loads a pretrained Wav2Vec2 or WavLM model (selected by *model_name*)
    together with its matching HuggingFace feature extractor, moves the
    model to *device*, sets it to eval mode, and freezes every parameter
    (``requires_grad_(False)``) — the backbone is used purely for inference.

    Parameters
    ----------
    model_name:
        HuggingFace model-hub identifier, e.g. ``"facebook/wav2vec2-base"``
        or ``"microsoft/wavlm-base-plus"``. Defaults to
        ``config.EMBEDDING_MODEL_NAME`` when ``None``, so the project-wide
        default backbone lives in one place.
    device:
        Torch device string (``"cuda"`` or ``"cpu"``). Defaults to
        ``config.get_device()`` when ``None``.
    """

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        self.model_name: str = model_name if model_name is not None else config.EMBEDDING_MODEL_NAME
        self.device: str = device if device is not None else config.get_device()

        # WavLM and Wav2Vec2 share the same feature-extractor class; only the
        # model class differs.
        if "wavlm" in self.model_name.lower():
            from transformers import Wav2Vec2FeatureExtractor, WavLMModel

            model_cls = WavLMModel
        else:
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

            model_cls = Wav2Vec2Model

        logger.info("Loading embedding backbone '%s' on %s", self.model_name, self.device)

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
        self.model = model_cls.from_pretrained(self.model_name)

        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    def _maybe_resample(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Resample *waveform* to ``config.SAMPLE_RATE`` if it isn't already."""
        if sr == config.SAMPLE_RATE:
            return waveform
        logger.debug("Resampling waveform %d Hz -> %d Hz", sr, config.SAMPLE_RATE)
        return librosa.resample(
            waveform.astype(np.float32), orig_sr=sr, target_sr=config.SAMPLE_RATE
        )

    def extract(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Extract a single mean-pooled embedding from one waveform.

        Parameters
        ----------
        waveform:
            1-D array of audio samples.
        sr:
            Sample rate of *waveform* in Hz. Resampled to
            ``config.SAMPLE_RATE`` (16000 Hz) first if it differs.

        Returns
        -------
        np.ndarray, shape ``(hidden_size,)``
            The mean-pooled embedding over the time dimension (768-dim for
            the base models).
        """
        waveform = self._maybe_resample(waveform, sr)

        inputs = self.feature_extractor(
            waveform, sampling_rate=config.SAMPLE_RATE, return_tensors="pt"
        )
        input_values = inputs["input_values"].to(self.device)

        with torch.no_grad():
            last_hidden_state = self.model(input_values).last_hidden_state  # (1, T, H)
            embedding = last_hidden_state.mean(dim=1).squeeze(0)  # (H,)

        return embedding.cpu().numpy()

    def extract_batch(self, waveforms: list[np.ndarray]) -> np.ndarray:
        """Extract mean-pooled embeddings for a batch of waveforms at once.

        Waveforms are expected to already be at ``config.SAMPLE_RATE`` (use
        ``voxguard.utils.audio_io.load_audio`` to ensure this) — unlike
        :meth:`extract`, no per-sample resampling is done here since a batch
        has a single implicit sample rate.

        Sequences shorter than the longest one in the batch are zero-padded
        by the feature extractor; the padded positions are excluded from the
        mean via the feature extractor's attention mask, mapped down to the
        model's output time resolution, so padding never pollutes the pooled
        embedding.

        Parameters
        ----------
        waveforms:
            List of 1-D audio arrays, all at ``config.SAMPLE_RATE``.

        Returns
        -------
        np.ndarray, shape ``(batch_size, hidden_size)``
            One mean-pooled embedding per input waveform.
        """
        inputs = self.feature_extractor(
            waveforms,
            sampling_rate=config.SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        input_values = inputs["input_values"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            last_hidden_state = self.model(
                input_values, attention_mask=attention_mask
            ).last_hidden_state  # (B, T, H)

            # Map the sample-level attention mask down to the reduced time
            # resolution of the conv feature encoder so it lines up with
            # last_hidden_state before pooling.
            feature_mask = self.model._get_feature_vector_attention_mask(
                last_hidden_state.shape[1], attention_mask
            )  # (B, T)
            mask = feature_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # (B, T, 1)

            summed = (last_hidden_state * mask).sum(dim=1)  # (B, H)
            counts = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
            embeddings = summed / counts

        return embeddings.cpu().numpy()

    def __repr__(self) -> str:
        return f"EmbeddingExtractor(model_name={self.model_name!r}, device={self.device!r})"
