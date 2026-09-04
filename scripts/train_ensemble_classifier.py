#!/usr/bin/env python3
"""
train_ensemble_classifier.py — Train the dual-backbone ensemble classifier.

This script consumes the already-cached ASVspoof2019 wav2vec2 and WavLM
train/dev embeddings, validates that they are present, then trains a
classifier on the concatenated features produced by
``load_combined_features`` and saves it with its scaler sidecar.

If the WavLM cache is missing, that means the Phase 2 Kaggle session did
not complete; finish that session rather than regenerating embeddings here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from voxguard import config
from voxguard.classifier.head import (
    fit_scaler,
    save_classifier,
    train_logistic_regression,
    train_mlp,
)
from voxguard.embeddings.cache import load_cached_embeddings
from voxguard.features.compose import load_combined_features
from voxguard.utils.logging_utils import get_logger

logger = get_logger("train_ensemble_classifier")

DEFAULT_EMBEDDINGS_DIR = config.MODELS_DIR / "embeddings"
DEFAULT_CLASSIFIERS_DIR = config.MODELS_DIR / "classifiers"
DEFAULT_HEAD = "logreg"


def _load_required_caches(embeddings_dir: Path) -> None:
    """Validates that the cached ASVspoof2019 embedding pairs are present."""
    required_paths = [
        embeddings_dir / "wav2vec2_train.npy",
        embeddings_dir / "wav2vec2_dev.npy",
        embeddings_dir / "wavlm_train.npy",
        embeddings_dir / "wavlm_dev.npy",
    ]

    for cache_path in required_paths:
        load_cached_embeddings(cache_path)


def _feature_paths(
    embeddings_dir: Path,
    use_prosody: bool,
) -> tuple[list[str], list[str]]:
    wav2vec2_train = str(embeddings_dir / "wav2vec2_train.npy")
    wav2vec2_dev = str(embeddings_dir / "wav2vec2_dev.npy")
    wavlm_train = str(embeddings_dir / "wavlm_train.npy")
    wavlm_dev = str(embeddings_dir / "wavlm_dev.npy")

    train_paths = [wav2vec2_train, wavlm_train]
    dev_paths = [wav2vec2_dev, wavlm_dev]
    if use_prosody:
        prosody_train = str(embeddings_dir / "prosody_train.npy")
        prosody_dev = str(embeddings_dir / "prosody_dev.npy")
        train_paths.append(prosody_train)
        dev_paths.append(prosody_dev)

    return train_paths, dev_paths


def train_ensemble_classifier(
    embeddings_dir: Path,
    output_dir: Path,
    head: str = DEFAULT_HEAD,
    use_prosody: bool = False,
) -> Path:
    """Trains the ensemble classifier on concatenated cached features."""
    _load_required_caches(embeddings_dir)

    train_paths, dev_paths = _feature_paths(embeddings_dir, use_prosody=use_prosody)
    logger.info("Loading concatenated train features from %s", train_paths)
    X_train, manifest_train = load_combined_features(train_paths)
    X_dev, manifest_dev = load_combined_features(dev_paths)

    y_train = manifest_train["label"].values
    y_dev = manifest_dev["label"].values

    scaler = fit_scaler(X_train)
    X_train_s = scaler.transform(X_train)
    X_dev_s = scaler.transform(X_dev)

    if head == "logreg":
        model = train_logistic_regression(X_train_s, y_train)
        save_path = output_dir / "ensemble_logreg"
    elif head == "mlp":
        model = train_mlp(X_train_s, y_train, X_dev_s, y_dev)
        save_path = output_dir / "ensemble_mlp"
    else:
        raise ValueError("head must be either 'logreg' or 'mlp'.")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_classifier(model, save_path, scaler)
    return save_path.with_suffix(".joblib" if head == "logreg" else ".pt")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the dual-backbone ensemble classifier."
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default=str(DEFAULT_EMBEDDINGS_DIR),
        help=f"Directory containing cached embeddings (default: {DEFAULT_EMBEDDINGS_DIR}).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_CLASSIFIERS_DIR),
        help=f"Directory to save the ensemble classifier (default: {DEFAULT_CLASSIFIERS_DIR}).",
    )
    parser.add_argument(
        "--head",
        choices=["logreg", "mlp"],
        default=DEFAULT_HEAD,
        help="Classifier head type to train (default: logreg).",
    )
    parser.add_argument(
        "--use_prosody",
        action="store_true",
        help="Append cached prosody features to the dual-embedding feature vector.",
    )

    args = parser.parse_args()

    try:
        train_ensemble_classifier(
            embeddings_dir=Path(args.embeddings_dir),
            output_dir=Path(args.output_dir),
            head=args.head,
            use_prosody=args.use_prosody,
        )
    except Exception as exc:
        logger.error("Ensemble training failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
