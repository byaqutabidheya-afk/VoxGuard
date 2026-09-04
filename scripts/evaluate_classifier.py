#!/usr/bin/env python3
"""
evaluate_classifier.py — Evaluates all 4 trained Phase 3 classifiers on the
ASVspoof2019 eval split and produces the baseline-vs-prosody comparison
table that decides whether prosody augmentation is worth keeping.

Loads the eval-split cached wav2vec2 embeddings and cached prosody
features, builds the baseline (768-dim) and prosody-augmented (778-dim)
eval feature sets using the exact same construction as
scripts/train_classifier.py, evaluates all four trained classifiers
(baseline_logreg, baseline_mlp, prosody_logreg, prosody_mlp), and writes a
comparison table to models/reports/asvspoof2019_eval_report.csv with
columns [model, feature_set, accuracy, roc_auc, eer].
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from voxguard import config
from voxguard.classifier.evaluate import evaluate_classifier
from voxguard.classifier.head import load_classifier
from voxguard.embeddings.cache import load_cached_embeddings
from voxguard.features.compose import load_combined_features
from voxguard.utils.logging_utils import get_logger

logger = get_logger("evaluate_classifier")

EMBEDDINGS_DIR = config.MODELS_DIR / "embeddings"
CLASSIFIERS_DIR = config.MODELS_DIR / "classifiers"
DEFAULT_REPORT_PATH = config.MODELS_DIR / "reports" / "asvspoof2019_eval_report.csv"

FEATURE_SET_NAMES = ["baseline", "prosody"]
MODEL_NAMES = ["logreg", "mlp"]


def _load_eval_feature_sets(embeddings_dir: Path) -> dict:
    """Loads the baseline and prosody-augmented eval feature sets.

    Mirrors the exact construction used in scripts/train_classifier.py so
    each classifier is evaluated on the same kind of feature vector it was
    trained on.
    """
    wav2vec2_eval = embeddings_dir / "wav2vec2_eval.npy"
    prosody_eval = embeddings_dir / "prosody_eval.npy"

    logger.info("Loading baseline (wav2vec2-only) eval embeddings...")
    X_eval_base, manifest_base = load_cached_embeddings(wav2vec2_eval)

    logger.info("Loading prosody-augmented eval feature set...")
    X_eval_pros, manifest_pros = load_combined_features([str(wav2vec2_eval), str(prosody_eval)])

    logger.info(
        "Eval feature sets ready: baseline=%s prosody-augmented=%s",
        X_eval_base.shape,
        X_eval_pros.shape,
    )

    return {
        "baseline": (X_eval_base, manifest_base["label"].values),
        "prosody": (X_eval_pros, manifest_pros["label"].values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline and prosody-augmented classifiers on the ASVspoof2019 eval split."
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default=str(EMBEDDINGS_DIR),
        help=f"Directory containing cached wav2vec2/prosody eval .npy+.csv pairs (default: {EMBEDDINGS_DIR}).",
    )
    parser.add_argument(
        "--classifiers_dir",
        type=str,
        default=str(CLASSIFIERS_DIR),
        help=f"Directory containing the 4 trained classifiers (default: {CLASSIFIERS_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_REPORT_PATH),
        help=f"Destination CSV for the comparison table (default: {DEFAULT_REPORT_PATH}).",
    )

    args = parser.parse_args()

    embeddings_dir = Path(args.embeddings_dir)
    classifiers_dir = Path(args.classifiers_dir)
    output_path = Path(args.output)

    logger.info("=" * 78)
    logger.info("VOXGUARD CLASSIFIER EVALUATION (Phase 3) — ASVspoof2019 eval split")
    logger.info("Embeddings dir  : %s", embeddings_dir)
    logger.info("Classifiers dir : %s", classifiers_dir)
    logger.info("Report output   : %s", output_path)
    logger.info("=" * 78)

    try:
        feature_sets = _load_eval_feature_sets(embeddings_dir)
    except Exception as exc:
        logger.error("Failed to load eval feature sets: %s", exc)
        sys.exit(1)

    rows = []
    for feature_set_name in FEATURE_SET_NAMES:
        X_eval, y_eval = feature_sets[feature_set_name]
        for model_name in MODEL_NAMES:
            classifier_path = classifiers_dir / f"{feature_set_name}_{model_name}"
            logger.info("-" * 78)
            logger.info("Evaluating %s / %s (%s)", feature_set_name, model_name, classifier_path)
            try:
                # load_classifier returns the model with the scaler fitted on its
                # training features; eval features must go through that same
                # transform before scoring.
                model, scaler = load_classifier(classifier_path)
                metrics = evaluate_classifier(model, scaler.transform(X_eval), y_eval)
            except Exception as exc:
                logger.error("Failed to evaluate %s / %s: %s", feature_set_name, model_name, exc)
                sys.exit(1)

            # Surface the confusion matrix: with a ~90% spoof-majority eval set,
            # accuracy alone hides a classifier that simply over-predicts spoof,
            # so the per-class breakdown is what makes that visible.
            cm = metrics["confusion_matrix"]
            (tn, fp), (fn, tp) = cm[0], cm[1]
            total = tn + fp + fn + tp
            logger.info(
                "  confusion matrix (rows=true, cols=pred):\n"
                "      true real      : pred_real=%6d  pred_synthetic=%6d  (bonafide recall=%.4f)\n"
                "      true synthetic : pred_real=%6d  pred_synthetic=%6d  (spoof    recall=%.4f)\n"
                "      predicted-synthetic rate=%.4f | eer_threshold=%.4f",
                tn,
                fp,
                tn / (tn + fp) if (tn + fp) else float("nan"),
                fn,
                tp,
                tp / (fn + tp) if (fn + tp) else float("nan"),
                (fp + tp) / total if total else float("nan"),
                metrics["eer_threshold"],
            )

            rows.append(
                {
                    "model": model_name,
                    "feature_set": feature_set_name,
                    "accuracy": metrics["accuracy"],
                    "roc_auc": metrics["roc_auc"],
                    "eer": metrics["eer"],
                }
            )

    report_df = pd.DataFrame(rows, columns=["model", "feature_set", "accuracy", "roc_auc", "eer"])

    logger.info("-" * 78)
    logger.info("ASVspoof2019 eval comparison (baseline vs. prosody-augmented):")
    print(report_df.to_string(index=False))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, index=False)
    logger.info("[SUCCESS] Saved comparison report to %s", output_path)


if __name__ == "__main__":
    main()
