"""Tests for the dual-backbone ensemble detector and training helper."""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LogisticRegression

import voxguard.classifier.ensemble as ensemble_module
import voxguard.classifier.infer as infer_module
from scripts.train_ensemble_classifier import train_ensemble_classifier
from voxguard.classifier.ensemble import (
    EnsembleDetector,
    extract_dual_embeddings,
    WeightedAverageDetector,
    weighted_average_ensemble,
)


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
        waveform = np.asarray(waveform_or_waveforms, dtype=np.float32)
        if waveform.ndim == 1:
            waveform = waveform[None, :]
        result = {"input_values": torch.tensor(waveform, dtype=torch.float32)}
        if return_attention_mask:
            result["attention_mask"] = torch.ones(
                (waveform.shape[0], waveform.shape[1]), dtype=torch.int64
            )
        return result


class _DummyBackbone:
    def __init__(self, model_name: str, hidden_size: int = 4) -> None:
        self.model_name = model_name
        self.config = types.SimpleNamespace(hidden_size=hidden_size)

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
        base = torch.arange(self.config.hidden_size, dtype=torch.float32).view(1, 1, -1)
        return types.SimpleNamespace(
            last_hidden_state=base.repeat(batch_size, time_steps, 1)
        )

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.Tensor
    ):
        return torch.ones(
            (attention_mask.shape[0], feature_vector_length), dtype=torch.int64
        )


class _DummyClassifier:
    def predict_proba(self, x):
        probs = np.linspace(0.1, 0.9, x.shape[0], dtype=np.float32)
        return np.column_stack([1.0 - probs, probs])


class _IdentityScaler:
    def transform(self, x):
        return x


@pytest.fixture
def dummy_backbones(monkeypatch: pytest.MonkeyPatch):
    dummy_module = types.SimpleNamespace(
        Wav2Vec2FeatureExtractor=_DummyFeatureExtractor,
        Wav2Vec2Model=_DummyBackbone,
        WavLMModel=_DummyBackbone,
    )
    monkeypatch.setitem(ensemble_module.sys.modules, "transformers", dummy_module)
    fitted_classifier = LogisticRegression().fit(
        np.array([[0.0] * 8, [1.0] * 8], dtype=np.float32),
        np.array([0, 1], dtype=np.int64),
    )
    monkeypatch.setattr(
        ensemble_module,
        "load_classifier",
        lambda path: (fitted_classifier, _IdentityScaler()),
    )
    fitted_single_classifier = LogisticRegression().fit(
        np.array([[0.0] * 4, [1.0] * 4], dtype=np.float32),
        np.array([0, 1], dtype=np.int64),
    )
    monkeypatch.setattr(
        infer_module,
        "load_classifier",
        lambda path: (fitted_single_classifier, _IdentityScaler()),
    )
    monkeypatch.setattr(
        infer_module.VoxGuardDetector,
        "_read_input_dim",
        staticmethod(lambda path: 4),
    )
    monkeypatch.setattr(
        ensemble_module.EnsembleDetector,
        "_read_input_dim",
        staticmethod(lambda path: 8),
    )
    yield


def test_extract_dual_embeddings_concatenates_two_pooled_vectors(
    dummy_backbones,
) -> None:
    waveform = np.ones(32, dtype=np.float32)
    w2v2 = ensemble_module.EmbeddingExtractor(
        model_name="facebook/wav2vec2-base", device="cpu"
    )
    wavlm = ensemble_module.EmbeddingExtractor(
        model_name="microsoft/wavlm-base-plus", device="cpu"
    )

    embedding = extract_dual_embeddings(waveform, 16000, w2v2, wavlm)

    assert embedding.shape == (8,)


def test_ensemble_detector_predict_waveform_uses_dual_embeddings(
    dummy_backbones,
) -> None:
    detector = EnsembleDetector(
        classifier_path="models/classifiers/ensemble_logreg.joblib", use_prosody=False
    )
    result = detector.predict_waveform(np.ones(32, dtype=np.float32), 16000)

    assert result["label"] in {"real", "synthetic"}
    assert 0.0 <= result["probability_synthetic"] <= 1.0


def test_weighted_average_detector_predict_waveform_uses_score_averaging(
    dummy_backbones,
) -> None:
    detector = WeightedAverageDetector(
        wav2vec2_classifier_path="models/classifiers/baseline_logreg.joblib",
        wavlm_classifier_path="models/classifiers/wavlm_logreg.joblib",
        weight_a=0.25,
    )
    result = detector.predict_waveform(np.ones(32, dtype=np.float32), 16000)

    assert result["label"] in {"real", "synthetic"}
    assert 0.0 <= result["probability_synthetic"] <= 1.0


def test_weighted_average_ensemble_returns_weighted_probability() -> None:
    assert weighted_average_ensemble(0.2, 0.8, weight_a=0.25) == pytest.approx(0.65)


def test_train_ensemble_classifier_uses_dual_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    embeddings_dir = tmp_path / "embeddings"
    train_paths = [
        str(embeddings_dir / "wav2vec2_train.npy"),
        str(embeddings_dir / "wavlm_train.npy"),
    ]
    dev_paths = [
        str(embeddings_dir / "wav2vec2_dev.npy"),
        str(embeddings_dir / "wavlm_dev.npy"),
    ]
    X_train = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32)
    X_dev = np.array([[3.0, 2.0, 1.0, 0.0]], dtype=np.float32)
    manifest_df = pd.DataFrame({"label": ["real"]})
    loaded_paths: list[str] = []

    monkeypatch.setattr(
        "scripts.train_ensemble_classifier.load_cached_embeddings",
        lambda path: loaded_paths.append(str(path)) or (X_train, manifest_df),
    )
    monkeypatch.setattr(
        "scripts.train_ensemble_classifier.load_combined_features",
        lambda paths: (X_train if paths == train_paths else X_dev, manifest_df),
    )
    monkeypatch.setattr(
        "scripts.train_ensemble_classifier.fit_scaler", lambda x: _IdentityScaler()
    )
    monkeypatch.setattr(
        "scripts.train_ensemble_classifier.train_logistic_regression",
        lambda x, y: _DummyClassifier(),
    )

    saved = {}

    def fake_save_classifier(model, path, scaler):
        saved["path"] = Path(path)
        saved["scaler"] = scaler

    monkeypatch.setattr(
        "scripts.train_ensemble_classifier.save_classifier", fake_save_classifier
    )

    output_path = train_ensemble_classifier(
        embeddings_dir=embeddings_dir,
        output_dir=tmp_path / "out",
        head="logreg",
        use_prosody=False,
    )

    assert output_path.name == "ensemble_logreg.joblib"
    assert saved["path"].name == "ensemble_logreg"
    assert loaded_paths == [
        str(embeddings_dir / "wav2vec2_train.npy"),
        str(embeddings_dir / "wav2vec2_dev.npy"),
        str(embeddings_dir / "wavlm_train.npy"),
        str(embeddings_dir / "wavlm_dev.npy"),
    ]
