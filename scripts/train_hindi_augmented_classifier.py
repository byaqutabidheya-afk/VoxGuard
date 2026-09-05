#!/usr/bin/env python3
"""
train_hindi_augmented_classifier.py — Trains Hindi-augmented and Hindi-only classifier heads.

Adapts Phase 4 training for the weighted-average ensemble architecture (Phase 3's winner):
since there is no single combined feature space, training occurs independently for each
backbone (wav2vec2, WavLM) without feature mixing.

For EACH backbone in [wav2vec2, wavlm]:
  1. Loads ASVspoof2019 train embeddings (models/embeddings/{backbone}_train.npy, 25,380 samples)
     and Hindi train embeddings (models/embeddings/{backbone}_hindi_train.npy, 100 samples).
     Uses baseline (768-dim) embeddings only (Phase 2 selected non-prosody baseline).
  2. Variant A (COMBINED, primary): row-wise concatenation along axis 0 (25,480 samples total).
  3. Variant B (HINDI-ONLY, ablation): Hindi train embeddings alone (100 samples).
  4. Runs 5-fold StratifiedKFold cross-validation on both variants, computing mean ± std accuracy.
  5. Fits standard scalers and trains class-balanced LogisticRegression heads on full variant data.
  6. Persists all 4 models (+ scalers and metadata sidecars) to models/classifiers/:
       - wav2vec2_hindi_combined_logreg.joblib
       - wav2vec2_hindi_only_logreg.joblib
       - wavlm_hindi_combined_logreg.joblib
       - wavlm_hindi_only_logreg.joblib
  7. Prints a formatted summary table: Backbone × Variant × Mean CV Accuracy ± Std.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from voxguard import config
from voxguard.classifier.head import (
    _encode_labels,
    fit_scaler,
    save_classifier,
    train_logistic_regression,
)
from voxguard.embeddings.cache import load_cached_embeddings
from voxguard.utils.logging_utils import get_logger

logger = get_logger("train_hindi_augmented_classifier")

DEFAULT_EMBEDDINGS_DIR = config.MODELS_DIR / "embeddings"
DEFAULT_CLASSIFIERS_DIR = config.MODELS_DIR / "classifiers"
DEFAULT_BACKBONES = ["wav2vec2", "wavlm"]
DEFAULT_N_SPLITS = 5
DEFAULT_RANDOM_STATE = 42


def load_backbone_train_data(
    embeddings_dir: Path,
    backbone: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads ASVspoof2019 train and Hindi train embedding caches for a given backbone.

    Parameters
    ----------
    embeddings_dir:
        Directory containing .npy and .csv embedding caches.
    backbone:
        Backbone name ('wav2vec2' or 'wavlm').

    Returns
    -------
    X_asv:
        ASVspoof2019 train embeddings matrix (n_samples, 768).
    y_asv:
        ASVspoof2019 train labels array (n_samples,).
    X_hindi:
        Hindi train embeddings matrix (n_samples, 768).
    y_hindi:
        Hindi train labels array (n_samples,).
    """
    asv_path = embeddings_dir / f"{backbone}_train.npy"
    hindi_path = embeddings_dir / f"{backbone}_hindi_train.npy"

    logger.info("Loading ASVspoof2019 train embeddings from %s", asv_path)
    X_asv, manifest_asv = load_cached_embeddings(asv_path)
    y_asv = manifest_asv["label"].values

    logger.info("Loading Hindi train embeddings from %s", hindi_path)
    X_hindi, manifest_hindi = load_cached_embeddings(hindi_path)
    y_hindi = manifest_hindi["label"].values

    if X_asv.shape[1] != X_hindi.shape[1]:
        raise ValueError(
            f"Feature dimensionality mismatch for backbone '{backbone}': "
            f"ASVspoof2019 dim={X_asv.shape[1]}, Hindi dim={X_hindi.shape[1]}"
        )

    logger.info(
        "Loaded %s data: ASVspoof2019 train shape=%s, Hindi train shape=%s",
        backbone,
        X_asv.shape,
        X_hindi.shape,
    )
    return X_asv, y_asv, X_hindi, y_hindi


def build_variant_datasets(
    X_asv: np.ndarray,
    y_asv: np.ndarray,
    X_hindi: np.ndarray,
    y_hindi: np.ndarray,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Builds Variant A (combined) and Variant B (hindi_only) datasets.

    Parameters
    ----------
    X_asv, y_asv:
        ASVspoof2019 train embeddings and labels.
    X_hindi, y_hindi:
        Hindi train embeddings and labels.

    Returns
    -------
    dict:
        Mapping 'combined' -> (X_combined, y_combined) and
        'hindi_only' -> (X_hindi, y_hindi).
    """
    X_combined = np.concatenate([X_asv, X_hindi], axis=0)
    y_combined = np.concatenate([y_asv, y_hindi], axis=0)

    return {
        "combined": (X_combined, y_combined),
        "only": (X_hindi, y_hindi),
    }


def cross_validate_variant(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[float, float, List[float]]:
    """Evaluates a variant using StratifiedKFold cross-validation.

    In each fold:
      1. Fits a StandardScaler on the training split.
      2. Scales train and validation splits.
      3. Trains LogisticRegression with class_weight='balanced'.
      4. Measures validation classification accuracy.

    Parameters
    ----------
    X:
        Feature matrix (n_samples, input_dim).
    y:
        Labels array (strings or ints).
    n_splits:
        Number of cross-validation folds (default 5).
    random_state:
        Random state for StratifiedKFold shuffle.

    Returns
    -------
    mean_accuracy:
        Mean validation accuracy across folds.
    std_accuracy:
        Standard deviation of validation accuracy across folds.
    fold_accuracies:
        List of validation accuracies per fold.
    """
    y_enc = _encode_labels(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_accuracies: List[float] = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_enc), start=1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y_enc[train_idx], y_enc[val_idx]

        scaler = fit_scaler(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_val_s = scaler.transform(X_val)

        model = train_logistic_regression(X_tr_s, y_tr)
        y_pred = model.predict(X_val_s)
        acc = float(accuracy_score(y_val, y_pred))
        fold_accuracies.append(acc)

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))
    return mean_acc, std_acc, fold_accuracies


def train_and_save_variant(
    X: np.ndarray,
    y: np.ndarray,
    output_path: Path,
) -> Path:
    """Fits StandardScaler and LogisticRegression on full data and saves model + scaler.

    Parameters
    ----------
    X:
        Feature matrix (n_samples, input_dim).
    y:
        Labels array.
    output_path:
        Target path (without extension) where classifier and scaler will be saved.

    Returns
    -------
    Path:
        Path to the saved .joblib model file.
    """
    scaler = fit_scaler(X)
    X_scaled = scaler.transform(X)
    model = train_logistic_regression(X_scaled, y)

    save_classifier(model, output_path, scaler)
    return output_path.with_suffix(".joblib")


def train_hindi_augmented_classifiers(
    embeddings_dir: Path = DEFAULT_EMBEDDINGS_DIR,
    output_dir: Path = DEFAULT_CLASSIFIERS_DIR,
    backbones: List[str] = DEFAULT_BACKBONES,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, Dict[str, Any]]:
    """Trains and evaluates Hindi-augmented & Hindi-only models for all backbones.

    Parameters
    ----------
    embeddings_dir:
        Directory containing cached embedding files.
    output_dir:
        Directory where trained models and sidecars will be stored.
    backbones:
        List of backbone names (e.g. ['wav2vec2', 'wavlm']).
    n_splits:
        Number of cross-validation folds.
    random_state:
        Seed for reproducible StratifiedKFold splits.

    Returns
    -------
    dict:
        Nested dictionary with CV metrics and saved paths for each backbone and variant.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, Any]] = {}

    for backbone in backbones:
        logger.info("=== Processing backbone: %s ===", backbone)
        X_asv, y_asv, X_hindi, y_hindi = load_backbone_train_data(embeddings_dir, backbone)
        variants = build_variant_datasets(X_asv, y_asv, X_hindi, y_hindi)

        results[backbone] = {}

        for variant_name, (X_var, y_var) in variants.items():
            logger.info(
                "Running %d-fold Stratified CV for %s [%s] (%d samples)...",
                n_splits,
                backbone,
                variant_name,
                X_var.shape[0],
            )
            mean_acc, std_acc, fold_accs = cross_validate_variant(
                X_var, y_var, n_splits=n_splits, random_state=random_state
            )
            logger.info(
                "%s [%s] CV Accuracy: %.4f ± %.4f (folds: %s)",
                backbone,
                variant_name,
                mean_acc,
                std_acc,
                [round(a, 4) for a in fold_accs],
            )

            # Persist model trained on full variant dataset
            save_name = f"{backbone}_hindi_{variant_name}_logreg"
            out_target = output_dir / save_name
            saved_path = train_and_save_variant(X_var, y_var, out_target)

            results[backbone][variant_name] = {
                "n_samples": X_var.shape[0],
                "mean_cv_accuracy": mean_acc,
                "std_cv_accuracy": std_acc,
                "fold_accuracies": fold_accs,
                "saved_path": str(saved_path),
            }

    # Print summary table
    print("\n" + "=" * 78)
    print(" HINDI CLASSIFIER HEAD TRAINING SUMMARY (Weighted-Average Ensemble Backbones)")
    print("=" * 78)
    print(f"{'Backbone':<12} | {'Variant':<12} | {'Train Samples':<14} | {'Mean CV Acc ± Std':<20} | {'Saved Model'}")
    print("-" * 78)
    for backbone, v_dict in results.items():
        for variant_name, metrics in v_dict.items():
            acc_str = f"{metrics['mean_cv_accuracy']:.4f} ± {metrics['std_cv_accuracy']:.4f}"
            saved_name = Path(metrics['saved_path']).name
            print(
                f"{backbone:<12} | {variant_name:<12} | {metrics['n_samples']:<14} | "
                f"{acc_str:<20} | {saved_name}"
            )
    print("=" * 78 + "\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Hindi-augmented (combined) and Hindi-only logistic-regression classifier heads."
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
        help=f"Directory to save trained classifiers (default: {DEFAULT_CLASSIFIERS_DIR}).",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help=f"Number of StratifiedKFold CV splits (default: {DEFAULT_N_SPLITS}).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random state seed for CV splits (default: {DEFAULT_RANDOM_STATE}).",
    )

    args = parser.parse_args()

    try:
        train_hindi_augmented_classifiers(
            embeddings_dir=Path(args.embeddings_dir),
            output_dir=Path(args.output_dir),
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
    except Exception as exc:
        logger.error("Hindi augmented classifier training failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
