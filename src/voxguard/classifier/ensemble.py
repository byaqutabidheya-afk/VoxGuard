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
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

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
        if VoxGuardDetector._rms_energy(waveform) < 0.01:
            return {
                "label": "inconclusive",
                "probability_synthetic": None,
            }

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
        prob_a = prediction_a.get("probability_synthetic")
        prob_b = prediction_b.get("probability_synthetic")
        if prob_a is None or prob_b is None:
            return {
                "label": "inconclusive",
                "probability_synthetic": None,
            }
        probability_synthetic = weighted_average_ensemble(
            float(prob_a),
            float(prob_b),
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


# =============================================================================
# Production detector factory (Phase F4 prep; call sites are rewired in F5)
# =============================================================================

# Detector mode -> the config dict that names its classifier heads. Two families exist because the
# two inference paths feed the model different input distributions: "wholeclip" heads score a complete
# uploaded file, "streaming" heads are chunk-native and score fixed-length windows (see config.py).
_PRODUCTION_MODES: Dict[str, str] = {
    "wholeclip": "PRODUCTION_WHOLECLIP_CLASSIFIERS",
    "streaming": "PRODUCTION_STREAMING_CLASSIFIERS",
}
_BACKBONE_KEYS = ("wav2vec2", "wavlm")


def _production_head_paths(mode: str) -> Dict[str, Path]:
    """Resolved (absolute) head paths for *mode*; ValueError on an unknown mode or a malformed config entry."""
    if mode not in _PRODUCTION_MODES:
        raise ValueError(
            f"Unknown production detector mode {mode!r}; expected one of {sorted(_PRODUCTION_MODES)}."
        )
    name = _PRODUCTION_MODES[mode]
    mapping = getattr(config, name)          # read at call time so a config change is always honoured
    missing_keys = [k for k in _BACKBONE_KEYS if k not in mapping]
    if missing_keys:
        raise ValueError(f"config.{name} is missing the backbone key(s) {missing_keys}; it needs {list(_BACKBONE_KEYS)}.")
    return {k: config.BASE_DIR / mapping[k] for k in _BACKBONE_KEYS}


def _required_files(model_path: Path) -> list[Path]:
    """Every file ``load_classifier`` needs for one head: the model, its JSON sidecar and its scaler.

    The scaler name is read from the sidecar when it is readable (that is what ``load_classifier`` uses),
    else the ``save_classifier`` convention ``<stem>_scaler.joblib`` is assumed. Checking only the model
    file would let a missing sidecar or scaler surface later as a confusing error inside inference.
    """
    sidecar = model_path.with_suffix(".json")
    scaler_name = f"{model_path.stem}_scaler.joblib"
    if sidecar.exists():
        try:
            scaler_name = json.loads(sidecar.read_text(encoding="utf-8")).get("scaler_path") or scaler_name
        except (OSError, ValueError):
            pass
    return [model_path, sidecar, model_path.with_name(scaler_name)]


def verify_production_classifiers(
    modes: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, Path]]:
    """Checks that every production classifier file exists on disk; raises naming ALL that are missing.

    Parameters
    ----------
    modes:
        Which families to check: any of ``"wholeclip"`` / ``"streaming"``. ``None`` (the default) checks
        BOTH, which is what an application should call once at startup so a partial deployment fails
        immediately and loudly, before any request is served.

    Returns
    -------
    dict
        ``{mode: {"wav2vec2": Path, "wavlm": Path}}`` of resolved model paths, for the modes checked.

    Raises
    ------
    ValueError
        Unknown mode, or a malformed ``PRODUCTION_*_CLASSIFIERS`` entry.
    FileNotFoundError
        One or more files are missing. The message names each missing path, the config entry it came from,
        and the ``scripts/`` command that produces it.
    """
    selected = list(_PRODUCTION_MODES) if modes is None else list(modes)
    resolved = {mode: _production_head_paths(mode) for mode in selected}      # validates modes first

    missing: list[str] = []
    for mode, heads in resolved.items():
        for backbone, model_path in heads.items():
            for f in _required_files(model_path):
                if not f.exists():
                    missing.append(f"  - {f}   (config.{_PRODUCTION_MODES[mode]}[{backbone!r}])")
    if missing:
        raise FileNotFoundError(
            "Production classifier file(s) missing on disk:\n" + "\n".join(missing) + "\n"
            "Train them (e.g. scripts/fix_retrain_wavlm_v2.py for the *_v2 WavLM heads, "
            "scripts/fix_retrain_matched_wholeclip.py, scripts/train_chunked_classifier.py) "
            "or correct the PRODUCTION_*_CLASSIFIERS paths in config.py."
        )
    return resolved


def build_production_detector(mode: str = "wholeclip") -> "WeightedAverageDetector":
    """Returns a ``WeightedAverageDetector`` wired to the production heads for *mode*.

    ``"wholeclip"`` uses ``config.PRODUCTION_WHOLECLIP_CLASSIFIERS`` (scores a complete uploaded file);
    ``"streaming"`` uses ``config.PRODUCTION_STREAMING_CLASSIFIERS`` (chunk-native, scores fixed-length
    windows). Both use ``config.PRODUCTION_ENSEMBLE_WEIGHT_A``. Every place that needs a production
    detector should call this rather than pass classifier paths itself.

    The existence check runs HERE, on the first (every) call, for the files of the requested mode only,
    before either SSL backbone is loaded: a missing head then fails in milliseconds with its path named,
    instead of after loading two backbones and then deep inside an inference call. Applications should
    additionally call :func:`verify_production_classifiers` once at startup to check both families.

    Raises
    ------
    ValueError
        Unknown *mode*.
    FileNotFoundError
        A classifier file for *mode* is missing (see :func:`verify_production_classifiers`).
    """
    paths = verify_production_classifiers([mode])[mode]
    return WeightedAverageDetector(
        wav2vec2_classifier_path=paths["wav2vec2"],
        wavlm_classifier_path=paths["wavlm"],
        weight_a=config.PRODUCTION_ENSEMBLE_WEIGHT_A,
    )


@lru_cache(maxsize=None)
def get_production_detector(mode: str = "wholeclip") -> "WeightedAverageDetector":
    """Process-wide SHARED production detector for *mode*, built once by :func:`build_production_detector`.

    Loading a detector loads two SSL backbones, and inference does not mutate it, so code that would otherwise
    build one per call or per session (``StreamingSession``'s default detector, the attribution default) shares
    one instance per mode instead. Use :func:`build_production_detector` when a fresh instance is wanted.
    A failed build (unknown mode, missing file) is not cached, so it raises again on the next call.
    """
    return build_production_detector(mode)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m voxguard.classifier.ensemble <audio_file>")
        print("   or: python -m src.voxguard.classifier.ensemble <audio_file>")
        sys.exit(1)

    detector = EnsembleDetector()
