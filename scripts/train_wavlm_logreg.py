#!/usr/bin/env python3
"""
train_wavlm_logreg.py — Train the WavLM-only Phase 3 baseline classifier.

Loads the cached WavLM train embeddings, fits the Phase 3 standard scaler,
trains the baseline logistic-regression head on the standardized features,
and saves the classifier plus scaler sidecar as
``models/classifiers/wavlm_logreg.joblib``.
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
)
from voxguard.embeddings.cache import load_cached_embeddings
from voxguard.utils.logging_utils import get_logger

logger = get_logger("train_wavlm_logreg")

DEFAULT_EMBEDDINGS_DIR = config.MODELS_DIR / "embeddings"
DEFAULT_OUTPUT_DIR = config.MODELS_DIR / "classifiers"


def train_wavlm_logreg(
    embeddings_dir: Path,
    output_dir: Path,
) -> Path:
    """Trains and saves the WavLM-only logistic-regression classifier."""
    wavlm_train = embeddings_dir / "wavlm_train.npy"
    logger.info("Loading cached WavLM train embeddings from %s", wavlm_train)

    X_train, manifest_train = load_cached_embeddings(wavlm_train)
    y_train = manifest_train["label"].values

    logger.info(
        "Fitting scaler and training logreg on %d samples (input_dim=%d)",
        X_train.shape[0],
        X_train.shape[1],
    )
    scaler = fit_scaler(X_train)
    X_train_scaled = scaler.transform(X_train)

    model = train_logistic_regression(X_train_scaled, y_train)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_classifier(model, output_dir / "wavlm_logreg", scaler)

    saved_path = output_dir / "wavlm_logreg.joblib"
    logger.info("Saved WavLM-only classifier to %s", saved_path)
    return saved_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the WavLM-only baseline logistic-regression classifier."
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default=str(DEFAULT_EMBEDDINGS_DIR),
        help=f"Directory containing wavlm_train.npy/.csv (default: {DEFAULT_EMBEDDINGS_DIR}).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to save wavlm_logreg.joblib and sidecars (default: {DEFAULT_OUTPUT_DIR}).",
    )

    args = parser.parse_args()

    try:
        train_wavlm_logreg(Path(args.embeddings_dir), Path(args.output_dir))
    except Exception as exc:
        logger.error("WavLM logreg training failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
