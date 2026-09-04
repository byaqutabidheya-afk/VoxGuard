"""Tests for cross-dataset evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from voxguard.classifier import cross_eval


def test_resolve_cache_path_uses_split_for_asvspoof_and_dataset_for_others() -> None:
    asv = cross_eval.resolve_cache_path("wav2vec2", "asvspoof2019")
    wf = cross_eval.resolve_cache_path("wav2vec2", "wavefake")

    assert asv.endswith(r"models\embeddings\wav2vec2_eval.npy") or asv.endswith(
        "models/embeddings/wav2vec2_eval.npy"
    )
    assert wf.endswith(r"models\embeddings\wav2vec2_wavefake.npy") or wf.endswith(
        "models/embeddings/wav2vec2_wavefake.npy"
    )


@dataclass
class _DummyDetector:
    fail_on: str | None = None

    def predict(self, audio_path: str) -> dict:
        if self.fail_on is not None and audio_path.endswith(self.fail_on):
            raise RuntimeError("corrupt file")
        return {
            "label": "synthetic",
            "probability_synthetic": 0.9 if "fake" in audio_path else 0.1,
        }


def test_zero_shot_eval_skips_failed_files_and_reports_metrics() -> None:
    df = pd.DataFrame(
        [
            {"filepath": "C:/audio/real_1.wav", "label": "real"},
            {"filepath": "C:/audio/fake_1.wav", "label": "synthetic"},
            {"filepath": "C:/audio/bad.wav", "label": "synthetic"},
        ]
    )

    metrics = cross_eval.zero_shot_eval(_DummyDetector(fail_on="bad.wav"), df)

    assert metrics["skip_count"] == 1
    assert metrics["confusion_matrix"] == [[1, 0], [0, 1]]
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["eer"] == pytest.approx(0.0)


def test_zero_shot_eval_from_cache_uses_combined_features_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    manifest = pd.DataFrame(
        {"filepath": ["a.wav", "b.wav"], "label": ["real", "synthetic"]}
    )

    called = {"combined": False}

    def fake_load_cached_embeddings(path):
        return X, manifest

    def fake_load_combined_features(paths):
        called["combined"] = True
        return X, manifest

    model = LogisticRegression().fit(X, [0, 1])

    class IdentityScaler:
        def transform(self, X_eval):
            return X_eval

    monkeypatch.setattr(
        cross_eval, "load_cached_embeddings", fake_load_cached_embeddings
    )
    monkeypatch.setattr(
        cross_eval, "load_combined_features", fake_load_combined_features
    )
    monkeypatch.setattr(
        cross_eval, "load_classifier", lambda path: (model, IdentityScaler())
    )

    metrics = cross_eval.zero_shot_eval_from_cache(
        classifier_path="models/classifiers/baseline_logreg.joblib",
        model_names=["wav2vec2", "wavlm"],
        dataset="wavefake",
    )

    assert called["combined"] is True
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["skip_count"] == 0
