#!/usr/bin/env python3
"""
train_classifier.py — Trains and saves VoxGuard's Phase 3 classifier heads.

Trains both classifier head types (logistic regression + MLP) on both
feature sets:

  - baseline:          wav2vec2 embeddings only (768-dim)
  - prosody-augmented: wav2vec2 embeddings + prosody features (778-dim),
                        combined via voxguard.features.compose.load_combined_features

producing 4 classifiers total, saved to models/classifiers/ for comparison
in the next phase:
  - baseline_logreg.joblib / .json
  - baseline_mlp.pt / .json
  - prosody_logreg.joblib / .json
  - prosody_mlp.pt / .json

Requires that scripts/extract_embeddings.py (wav2vec2, --split train/dev)
and the prosody cache (voxguard.features.compose.extract_and_cache_prosody,
output models/embeddings/prosody_{train,dev}.npy) have already been run.
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

logger = get_logger("train_classifier")

EMBEDDINGS_DIR = config.MODELS_DIR / "embeddings"
DEFAULT_CLASSIFIERS_DIR = config.MODELS_DIR / "classifiers"


def _load_feature_sets(embeddings_dir):
    """Loads the baseline and prosody-augmented train/dev feature sets.

    Returns
    -------
    dict with keys "baseline" and "prosody", each mapping to
    (X_train, y_train, X_dev, y_dev).
    """
    wav2vec2_train = embeddings_dir / "wav2vec2_train.npy"
    wav2vec2_dev = embeddings_dir / "wav2vec2_dev.npy"
    prosody_train = embeddings_dir / "prosody_train.npy"
    prosody_dev = embeddings_dir / "prosody_dev.npy"

    logger.info("Loading baseline (wav2vec2-only) train/dev embeddings...")
    X_train_base, manifest_train_base = load_cached_embeddings(wav2vec2_train)
    X_dev_base, manifest_dev_base = load_cached_embeddings(wav2vec2_dev)

    logger.info("Loading prosody-augmented train/dev feature sets...")
    X_train_pros, manifest_train_pros = load_combined_features(
        [str(wav2vec2_train), str(prosody_train)]
    )
    X_dev_pros, manifest_dev_pros = load_combined_features(
        [str(wav2vec2_dev), str(prosody_dev)]
    )

    logger.info(
        "Feature sets ready: baseline train=%s dev=%s | prosody-augmented train=%s dev=%s",
        X_train_base.shape,
        X_dev_base.shape,
        X_train_pros.shape,
        X_dev_pros.shape,
    )

    return {
        "baseline": (X_train_base, manifest_train_base["label"].values, X_dev_base, manifest_dev_base["label"].values),
        "prosody": (X_train_pros, manifest_train_pros["label"].values, X_dev_pros, manifest_dev_pros["label"].values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train baseline and prosody-augmented logreg + MLP classifier heads."
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default=str(EMBEDDINGS_DIR),
        help=f"Directory containing cached wav2vec2/prosody .npy+.csv pairs (default: {EMBEDDINGS_DIR}).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_CLASSIFIERS_DIR),
        help=f"Directory to save trained classifiers (default: {DEFAULT_CLASSIFIERS_DIR}).",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Max MLP training epochs (default: 20).")
    parser.add_argument("--lr", type=float, default=1e-3, help="MLP Adam learning rate (default: 1e-3).")
    parser.add_argument("--patience", type=int, default=3, help="MLP early-stopping patience (default: 3).")
    parser.add_argument("--batch_size", type=int, default=64, help="MLP training batch size (default: 64).")

    args = parser.parse_args()

    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 78)
    logger.info("VOXGUARD CLASSIFIER TRAINING (Phase 3)")
    logger.info("Embeddings dir : %s", embeddings_dir)
    logger.info("Output dir     : %s", output_dir)
    logger.info("=" * 78)

    try:
        feature_sets = _load_feature_sets(embeddings_dir)
    except Exception as exc:
        logger.error("Failed to load feature sets: %s", exc)
        sys.exit(1)

    for feature_set_name, (X_train, y_train, X_dev, y_dev) in feature_sets.items():
        logger.info("-" * 78)
        logger.info("Training on '%s' feature set (input_dim=%d)", feature_set_name, X_train.shape[1])

        # One scaler per feature set (baseline 768-dim and prosody 778-dim are
        # different feature spaces), fit on train only and applied to train and
        # dev alike. Each model persists its own copy so every saved classifier
        # is self-contained.
        scaler = fit_scaler(X_train)
        X_train_s = scaler.transform(X_train)
        X_dev_s = scaler.transform(X_dev)

        logreg = train_logistic_regression(X_train_s, y_train)
        save_classifier(logreg, output_dir / f"{feature_set_name}_logreg", scaler)

        mlp = train_mlp(
            X_train_s,
            y_train,
            X_dev_s,
            y_dev,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            batch_size=args.batch_size,
        )
        save_classifier(mlp, output_dir / f"{feature_set_name}_mlp", scaler)

    logger.info("-" * 78)
    logger.info("[SUCCESS] Trained and saved 4 classifiers to %s", output_dir)
    logger.info("-" * 78)


if __name__ == "__main__":
    main()
