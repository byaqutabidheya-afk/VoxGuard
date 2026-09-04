"""
infer.py — end-to-end inference: audio in, spoof verdict out (Phase 2, Prompt 2.7).

``VoxGuardDetector`` is the single entry point every later phase calls —
Phase 4's streaming engine, the Gradio app, and Phase 9's multimodal fusion
all go through :meth:`VoxGuardDetector.predict` or
:meth:`VoxGuardDetector.predict_waveform`. Both signatures are load-bearing
interfaces: keep them stable.

The detector hides which classifier variant is in use. Point
*classifier_path* at a baseline (768-dim, embedding-only) or a
prosody-augmented (778-dim) saved model and the feature pipeline
reconfigures itself from the model's metadata sidecar — callers never need
to know which one they have.
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
from voxguard.embeddings.extractor import EmbeddingExtractor
from voxguard.features.prosody import ProsodyFeatureExtractor
from voxguard.utils.audio_io import load_audio
from voxguard.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Phase 2's evaluation picked this model; see models/reports/decision_notes.md.
# baseline_logreg won on EER (0.1090 vs 0.1096 prosody-logreg, 0.1270
# baseline-mlp, 0.1137 prosody-mlp) on the corrected, standardized
# ASVspoof2019 eval set. Prosody augmentation is a wash for the logistic
# head (+0.0006 EER) and only helps the weaker MLP, so the baseline is both
# the most accurate option and the one that keeps per-clip librosa prosody
# extraction out of the inference path.
DEFAULT_CLASSIFIER_PATH = "models/classifiers/baseline_logreg.joblib"

# Probability at or above which a clip is reported as synthetic. Phase 3
# replaces this with a tuned, per-context value via config.RISK_THRESHOLDS.
DECISION_THRESHOLD = 0.5


class VoxGuardDetector:
    """Detects synthetic (cloned/spoofed) speech in an audio clip.

    Loads a frozen embedding backbone, a trained classifier head with its
    fitted feature scaler, and — when the classifier expects them — a
    prosody feature extractor. All model loading happens once here in the
    constructor, so the per-clip predict methods do no setup work.

    Parameters
    ----------
    embedding_model_name:
        HuggingFace identifier for the embedding backbone. Defaults to
        ``config.EMBEDDING_MODEL_NAME`` (``facebook/wav2vec2-base``). This
        must match the backbone the classifier was trained on — a WavLM
        classifier scored on wav2vec2 embeddings produces silent nonsense.
    classifier_path:
        Path to a classifier saved by ``save_classifier``, with or without
        extension. Relative paths resolve against ``config.BASE_DIR`` so
        the detector works regardless of the caller's working directory.
        Defaults to Phase 2's chosen best model (see
        ``DEFAULT_CLASSIFIER_PATH``).
    use_prosody:
        Whether to concatenate the 10-dim prosody vector onto the
        embedding. Leave as ``None`` to auto-detect from the classifier's
        metadata sidecar: prosody is enabled when the model's
        ``input_dim`` exceeds the backbone's embedding width. Pass
        ``True``/``False`` only to override that detection.

    Raises
    ------
    ValueError
        If *use_prosody* is set explicitly in a way that cannot produce the
        feature width the classifier expects.
    """

    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        classifier_path: Union[str, Path] = DEFAULT_CLASSIFIER_PATH,
        use_prosody: Optional[bool] = None,
    ) -> None:
        path = Path(classifier_path)
        if not path.is_absolute():
            path = config.BASE_DIR / path
        self.classifier_path = path

        self.extractor = EmbeddingExtractor(model_name=embedding_model_name)
        self.embedding_dim: int = int(self.extractor.model.config.hidden_size)

        # load_classifier returns the model together with the StandardScaler
        # fitted on its training features. The scaler is not optional: the
        # model was fitted on standardized inputs, so scoring raw features
        # returns confident-looking, meaningless probabilities rather than
        # raising. See head.save_classifier's contract.
        self.model, self.scaler = load_classifier(path)
        self.input_dim: int = self._read_input_dim(path)

        if use_prosody is None:
            # Auto-detect: a classifier wider than the embedding itself was
            # trained on embedding+prosody, so the same detector class works
            # with either variant purely from what it was pointed at.
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
                f"Feature width mismatch: classifier at {path} expects input_dim="
                f"{self.input_dim}, but this configuration builds {expected}-dim vectors "
                f"({self.embedding_dim}-dim {self.extractor.model_name} embedding"
                + (
                    f" + {len(ProsodyFeatureExtractor.FEATURE_NAMES)}-dim prosody"
                    if self.use_prosody
                    else ""
                )
                + "). Check that use_prosody and the embedding backbone match the "
                "classifier's training configuration."
            )

        logger.info(
            "VoxGuardDetector ready: backbone=%s, classifier=%s, input_dim=%d, use_prosody=%s",
            self.extractor.model_name,
            path.name,
            self.input_dim,
            self.use_prosody,
        )

    @staticmethod
    def _read_input_dim(path: Path) -> int:
        """Reads ``input_dim`` from the classifier's JSON metadata sidecar."""
        meta_path = path.with_suffix(".json")
        if not meta_path.exists():
            raise FileNotFoundError(f"Classifier metadata sidecar not found: {meta_path}")
        with open(meta_path) as f:
            return int(json.load(f)["input_dim"])

    def _build_features(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Builds the model's input vector from an in-memory waveform.

        Returns a ``(1, input_dim)`` matrix, already standardized with the
        classifier's own scaler and ready to score. Prosody features are
        appended after the embedding, matching the column order
        ``load_combined_features`` produces at training time
        (``[embedding_path, prosody_path]``).
        """
        embedding = self.extractor.extract(waveform, sr)

        if self.use_prosody:
            prosody = self.prosody_extractor.extract(waveform, sr)
            features = np.concatenate([embedding, prosody])
        else:
            features = embedding

        # Standardize with the scaler fitted on this model's training data.
        return self.scaler.transform(features.reshape(1, -1))

    def _score(self, features: np.ndarray) -> float:
        """Returns P(synthetic) for a single standardized feature row."""
        if isinstance(self.model, LogisticRegression):
            return float(self.model.predict_proba(features)[0, 1])
        if isinstance(self.model, MLPClassifierHead):
            # MLPClassifierHead.forward returns raw logits; predict_proba
            # applies the sigmoid and returns a 1-D probability array.
            return float(self.model.predict_proba(features)[0])
        raise TypeError(f"Unsupported classifier type: {type(self.model)!r}")

    def predict(self, audio_path: str) -> Dict[str, object]:
        """Predicts whether the audio file at *audio_path* is synthetic.

        Stable interface — later phases depend on this signature.

        Parameters
        ----------
        audio_path:
            Path to an audio file readable by ``audio_io.load_audio``
            (WAV, FLAC, OGG, ...). Resampled to ``config.SAMPLE_RATE`` and
            downmixed to mono automatically.

        Returns
        -------
        dict
            ``{"label": "real"|"synthetic", "probability_synthetic": float}``
            where the probability is in ``[0, 1]``.
        """
        waveform, sr = load_audio(audio_path, target_sr=config.SAMPLE_RATE)
        return self.predict_waveform(waveform, sr)

    def predict_waveform(self, waveform: np.ndarray, sr: int) -> Dict[str, object]:
        """Predicts whether an in-memory waveform is synthetic.

        Identical logic to :meth:`predict` without the file read. **Performs
        no disk I/O** — every model is loaded once in the constructor — so
        Phase 4's streaming engine can call this per chunk on the hot path.

        Stable interface — later phases depend on this signature.

        Parameters
        ----------
        waveform:
            1-D array of mono audio samples.
        sr:
            Sample rate of *waveform* in Hz. Resampled to
            ``config.SAMPLE_RATE`` if it differs, so that prosody features
            are computed at the same rate they were trained on.

        Returns
        -------
        dict
            ``{"label": "real"|"synthetic", "probability_synthetic": float}``
            where the probability is in ``[0, 1]``.
        """
        waveform = np.asarray(waveform, dtype=np.float32)

        if sr != config.SAMPLE_RATE:
            waveform = librosa.resample(
                waveform, orig_sr=sr, target_sr=config.SAMPLE_RATE
            ).astype(np.float32)
            sr = config.SAMPLE_RATE

        probability_synthetic = self._score(self._build_features(waveform, sr))

        return {
            "label": "synthetic" if probability_synthetic >= DECISION_THRESHOLD else "real",
            "probability_synthetic": probability_synthetic,
        }

    def __repr__(self) -> str:
        return (
            f"VoxGuardDetector(backbone={self.extractor.model_name!r}, "
            f"classifier={self.classifier_path.name!r}, "
            f"input_dim={self.input_dim}, use_prosody={self.use_prosody})"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        # Both forms work from the repo root; the installed-package form is
        # canonical and also works from anywhere.
        print("Usage: python -m voxguard.classifier.infer <audio_file>")
        print("   or: python -m src.voxguard.classifier.infer <audio_file>")
        sys.exit(1)

    detector = VoxGuardDetector()
    print(detector)
    print(detector.predict(sys.argv[1]))
