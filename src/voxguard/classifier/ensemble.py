"""
ensemble.py — dual-backbone ensemble detector and simple score averaging.

The concatenated-feature path mirrors ``VoxGuardDetector`` but uses both
wav2vec2 and WavLM embeddings as the base representation. A lightweight
weighted-average helper is also provided for fallback score-level
ensembling.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional, Union

import librosa
import numpy as np
from sklearn.linear_model import LogisticRegression

from voxguard import config
from voxguard.classifier.head import MLPClassifierHead, load_classifier
from voxguard.classifier.infer import DECISION_THRESHOLD, VoxGuardDetector
from voxguard.embeddings.extractor import EmbeddingExtractor
from voxguard.features.prosody import ProsodyFeatureExtractor
from voxguard.utils.audio_io import load_audio
from voxguard.utils.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_CLASSIFIER_PATH = "models/classifiers/ensemble_logreg.joblib"
DEFAULT_WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-base"
DEFAULT_WAVLM_MODEL_NAME = "microsoft/wavlm-base-plus"
DEFAULT_WAV2VEC2_CLASSIFIER_PATH = "models/classifiers/baseline_logreg.joblib"
DEFAULT_WAVLM_CLASSIFIER_PATH = "models/classifiers/wavlm_logreg.joblib"


def extract_dual_embeddings(
    waveform: np.ndarray,
    sr: int,
    wav2vec2_extractor: EmbeddingExtractor,
    wavlm_extractor: EmbeddingExtractor,
) -> np.ndarray:
    """Extracts and concatenates wav2vec2 and WavLM pooled embeddings."""
    wav2vec2_embedding = wav2vec2_extractor.extract(waveform, sr)
    wavlm_embedding = wavlm_extractor.extract(waveform, sr)
    return np.concatenate([wav2vec2_embedding, wavlm_embedding]).astype(
        np.float32, copy=False
    )


def weighted_average_ensemble(
    prob_a: float, prob_b: float, weight_a: float = 0.5
) -> float:
    """Returns a weighted average of two synthetic-class probabilities."""
    if not 0.0 <= weight_a <= 1.0:
        raise ValueError(f"weight_a must be in [0, 1]; got {weight_a!r}.")
    return float(weight_a * float(prob_a) + (1.0 - weight_a) * float(prob_b))


class EnsembleDetector:
    """Detects synthetic speech using concatenated wav2vec2 + WavLM embeddings."""

    def __init__(
        self,
        wav2vec2_model_name: Optional[str] = None,
        wavlm_model_name: Optional[str] = None,
        classifier_path: Union[str, Path] = DEFAULT_CLASSIFIER_PATH,
        use_prosody: Optional[bool] = None,
        threshold: float = DECISION_THRESHOLD,
    ) -> None:
        self.threshold = float(threshold)
        path = Path(classifier_path)
        if not path.is_absolute():
            path = config.BASE_DIR / path
        self.classifier_path = path

        self.wav2vec2_extractor = EmbeddingExtractor(
            model_name=wav2vec2_model_name or DEFAULT_WAV2VEC2_MODEL_NAME
        )
        self.wavlm_extractor = EmbeddingExtractor(
            model_name=wavlm_model_name or DEFAULT_WAVLM_MODEL_NAME
        )
        self.embedding_dim: int = int(
            self.wav2vec2_extractor.model.config.hidden_size
        ) + int(self.wavlm_extractor.model.config.hidden_size)

        self.model, self.scaler = load_classifier(path)
        self.input_dim: int = self._read_input_dim(path)

        if use_prosody is None:
            self.use_prosody = self.input_dim > self.embedding_dim
        else:
            self.use_prosody = bool(use_prosody)

        self.prosody_extractor: Optional[ProsodyFeatureExtractor] = (
            ProsodyFeatureExtractor() if self.use_prosody else None
        )

        expected = self.embedding_dim + (
            len(ProsodyFeatureExtractor.FEATURE_NAMES) if self.use_prosody else 0
        )
        if expected != self.input_dim:
            raise ValueError(
                f"Feature width mismatch: classifier at {path} expects input_dim={self.input_dim}, "
                f"but this configuration builds {expected}-dim vectors "
                f"({self.embedding_dim}-dim wav2vec2+WavLM embedding"
                + (
                    f" + {len(ProsodyFeatureExtractor.FEATURE_NAMES)}-dim prosody"
                    if self.use_prosody
                    else ""
                )
                + "). Check that use_prosody and the backbone pair match the classifier's training configuration."
            )

        logger.info(
            "EnsembleDetector ready: wav2vec2=%s wavlm=%s classifier=%s input_dim=%d use_prosody=%s",
            self.wav2vec2_extractor.model_name,
            self.wavlm_extractor.model_name,
            path.name,
            self.input_dim,
            self.use_prosody,
        )

    @staticmethod
    def _read_input_dim(path: Path) -> int:
        meta_path = path.with_suffix(".json")
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Classifier metadata sidecar not found: {meta_path}"
            )
        with open(meta_path) as f:
            return int(json.load(f)["input_dim"])

    def _build_features(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        waveform = np.asarray(waveform, dtype=np.float32)
        if sr != config.SAMPLE_RATE:
            waveform = librosa.resample(
                waveform, orig_sr=sr, target_sr=config.SAMPLE_RATE
            ).astype(np.float32)
            sr = config.SAMPLE_RATE

        features = extract_dual_embeddings(
            waveform,
            sr,
            self.wav2vec2_extractor,
            self.wavlm_extractor,
        )

        if self.use_prosody:
            prosody = self.prosody_extractor.extract(waveform, sr)
            features = np.concatenate([features, prosody])

        return self.scaler.transform(features.reshape(1, -1))

    def _score(self, features: np.ndarray) -> float:
        if isinstance(self.model, LogisticRegression):
            return float(self.model.predict_proba(features)[0, 1])
        if isinstance(self.model, MLPClassifierHead):
            return float(self.model.predict_proba(features)[0])
        raise TypeError(f"Unsupported classifier type: {type(self.model)!r}")

    def predict(
        self, audio_path: str, threshold: Optional[float] = None
    ) -> Dict[str, object]:
        """Predicts whether the audio file at *audio_path* is synthetic.

        *threshold* optionally overrides ``self.threshold`` for this call;
        ``None`` (the default) preserves the detector's configured cutoff.
        """
        waveform, sr = load_audio(audio_path, target_sr=config.SAMPLE_RATE)
        return self.predict_waveform(waveform, sr, threshold=threshold)

    def predict_waveform(
        self, waveform: np.ndarray, sr: int, threshold: Optional[float] = None
    ) -> Dict[str, object]:
        """Predicts whether an in-memory waveform is synthetic (no disk I/O).

        *threshold* optionally overrides ``self.threshold`` for this call;
        ``None`` (the default) preserves the detector's configured cutoff.
        """
        features = self._build_features(waveform, sr)
        probability_synthetic = self._score(features)
        cutoff = self.threshold if threshold is None else float(threshold)
        return {
            "label": "synthetic" if probability_synthetic >= cutoff else "real",
            "probability_synthetic": probability_synthetic,
        }

    def __repr__(self) -> str:
        return (
            f"EnsembleDetector(wav2vec2={self.wav2vec2_extractor.model_name!r}, "
            f"wavlm={self.wavlm_extractor.model_name!r}, "
            f"classifier={self.classifier_path.name!r}, input_dim={self.input_dim}, "
            f"use_prosody={self.use_prosody}, threshold={self.threshold:.4f})"
        )


class WeightedAverageDetector:
    """Detects synthetic speech by averaging wav2vec2-only and WavLM-only scores."""

    def __init__(
        self,
        wav2vec2_classifier_path: Union[str, Path] = DEFAULT_WAV2VEC2_CLASSIFIER_PATH,
        wavlm_classifier_path: Union[str, Path] = DEFAULT_WAVLM_CLASSIFIER_PATH,
        wav2vec2_model_name: Optional[str] = None,
        wavlm_model_name: Optional[str] = None,
        weight_a: float = 0.5,
        threshold: float = DECISION_THRESHOLD,
    ) -> None:
        self.weight_a = float(weight_a)
        self.threshold = float(threshold)
        self.detector_a = VoxGuardDetector(
            embedding_model_name=wav2vec2_model_name or DEFAULT_WAV2VEC2_MODEL_NAME,
            classifier_path=wav2vec2_classifier_path,
            use_prosody=False,
        )
        self.detector_b = VoxGuardDetector(
            embedding_model_name=wavlm_model_name or DEFAULT_WAVLM_MODEL_NAME,
            classifier_path=wavlm_classifier_path,
            use_prosody=False,
        )

    def predict(
        self, audio_path: str, threshold: Optional[float] = None
    ) -> Dict[str, object]:
        """Predicts whether the audio file at *audio_path* is synthetic.

        The sub-detectors' own labels are discarded — only their
        probabilities are averaged — so *threshold* is applied once, here,
        to the combined score. ``None`` (the default) uses
        ``self.threshold``.
        """
        prediction_a = self.detector_a.predict(audio_path)
        prediction_b = self.detector_b.predict(audio_path)
        return self._combine(prediction_a, prediction_b, threshold)

    def predict_waveform(
        self, waveform: np.ndarray, sr: int, threshold: Optional[float] = None
    ) -> Dict[str, object]:
        """Predicts whether an in-memory waveform is synthetic (no disk I/O).

        *threshold* optionally overrides ``self.threshold`` for this call;
        ``None`` (the default) preserves the detector's configured cutoff.
        """
        prediction_a = self.detector_a.predict_waveform(waveform, sr)
        prediction_b = self.detector_b.predict_waveform(waveform, sr)
        return self._combine(prediction_a, prediction_b, threshold)

    def _combine(
        self,
        prediction_a: Dict[str, object],
        prediction_b: Dict[str, object],
        threshold: Optional[float],
    ) -> Dict[str, object]:
        """Averages the two sub-scores and applies the decision threshold."""
        probability_synthetic = weighted_average_ensemble(
            float(prediction_a["probability_synthetic"]),
            float(prediction_b["probability_synthetic"]),
            weight_a=self.weight_a,
        )
        cutoff = self.threshold if threshold is None else float(threshold)
        return {
            "label": "synthetic" if probability_synthetic >= cutoff else "real",
            "probability_synthetic": probability_synthetic,
        }

    def __repr__(self) -> str:
        return (
            f"WeightedAverageDetector(wav2vec2={self.detector_a.extractor.model_name!r}, "
            f"wavlm={self.detector_b.extractor.model_name!r}, "
            f"weight_a={self.weight_a:.2f}, threshold={self.threshold:.4f})"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m voxguard.classifier.ensemble <audio_file>")
        print("   or: python -m src.voxguard.classifier.ensemble <audio_file>")
        sys.exit(1)

    detector = EnsembleDetector()
