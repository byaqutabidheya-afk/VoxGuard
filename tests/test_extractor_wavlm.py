"""Tests for EmbeddingExtractor WavLM / Wav2Vec2 support."""

from __future__ import annotations

import types
import sys

import numpy as np
import pytest
import torch

import voxguard.embeddings.extractor as extractor_module
from voxguard.embeddings.extractor import EmbeddingExtractor


class _DummyFeatureExtractor:
    @classmethod
    def from_pretrained(cls, model_name: str):
        return cls(model_name)

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def __call__(
        self,
        waveform_or_waveforms,
        sampling_rate: int,
        return_tensors: str,
        padding: bool = False,
        return_attention_mask: bool = False,
    ):
        if isinstance(waveform_or_waveforms, list):
            lengths = [len(waveform) for waveform in waveform_or_waveforms]
            max_len = max(lengths)
            padded = []
            masks = []
            for waveform in waveform_or_waveforms:
                pad_len = max_len - len(waveform)
                padded.append(
                    np.pad(np.asarray(waveform, dtype=np.float32), (0, pad_len))
                )
                masks.append(
                    np.concatenate(
                        [
                            np.ones(len(waveform), dtype=np.int64),
                            np.zeros(pad_len, dtype=np.int64),
                        ]
                    )
                )
            result = {
                "input_values": torch.tensor(np.asarray(padded), dtype=torch.float32)
            }
            if return_attention_mask:
                result["attention_mask"] = torch.tensor(
                    np.asarray(masks), dtype=torch.int64
                )
            return result

        waveform = np.asarray(waveform_or_waveforms, dtype=np.float32)
        result = {"input_values": torch.tensor(waveform[None, :], dtype=torch.float32)}
        if return_attention_mask:
            result["attention_mask"] = torch.ones(
                (1, waveform.shape[0]), dtype=torch.int64
            )
        return result


class _DummyBackbone:
    def __init__(self, model_name: str, hidden_size: int = 8) -> None:
        self.model_name = model_name
        self.config = types.SimpleNamespace(hidden_size=hidden_size)
        self.hidden_size = hidden_size
        self._feature_vector_attention_mask = self._feature_vector_attention_mask_impl

    @classmethod
    def from_pretrained(cls, model_name: str):
        return cls(model_name)

    def to(self, device: str):
        return self

    def eval(self):
        return self

    def requires_grad_(self, flag: bool):
        return self

    def __call__(self, input_values, attention_mask=None):
        batch_size, time_steps = input_values.shape
        hidden = torch.arange(self.hidden_size, dtype=torch.float32).view(1, 1, -1)
        last_hidden_state = hidden.repeat(batch_size, time_steps, 1)
        return types.SimpleNamespace(last_hidden_state=last_hidden_state)

    def _feature_vector_attention_mask_impl(
        self, feature_vector_length: int, attention_mask: torch.Tensor
    ):
        return torch.ones(
            (attention_mask.shape[0], feature_vector_length), dtype=torch.int64
        )


@pytest.fixture
def dummy_transformers(monkeypatch: pytest.MonkeyPatch):
    dummy_module = types.SimpleNamespace(
        Wav2Vec2FeatureExtractor=_DummyFeatureExtractor,
        Wav2Vec2Model=_DummyBackbone,
        WavLMModel=_DummyBackbone,
    )
    monkeypatch.setitem(sys.modules, "transformers", dummy_module)
    yield


def test_embedding_extractor_supports_wav2vec2_and_wavlm(dummy_transformers) -> None:
    waveform = np.sin(
        np.linspace(0, 2 * np.pi, 320, endpoint=False, dtype=np.float32)
    ).astype(np.float32)

    wav2vec2_extractor = EmbeddingExtractor(
        model_name="facebook/wav2vec2-base", device="cpu"
    )
    wavlm_extractor = EmbeddingExtractor(
        model_name="microsoft/wavlm-base-plus", device="cpu"
    )

    wav2vec2_embedding = wav2vec2_extractor.extract(waveform, sr=16_000)
    wavlm_embedding = wavlm_extractor.extract(waveform, sr=16_000)

    assert wav2vec2_embedding.shape == wavlm_embedding.shape == (8,)
    assert wav2vec2_embedding.dtype == wavlm_embedding.dtype == np.float32
