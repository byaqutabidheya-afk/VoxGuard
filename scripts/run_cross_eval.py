#!/usr/bin/env python3
"""
run_cross_eval.py — Cross-dataset evaluation CLI for VoxGuard.

Produces a single table that compares the chosen Phase 2 ASVspoof2019
baseline row against the same classifier on WaveFake and In-the-Wild.

By default the WaveFake / In-the-Wild runs use the cached feature path.
If the relevant cache files are missing, the script falls back to the
file-by-file ``VoxGuardDetector.predict`` path and prints a loud warning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from voxguard import config
from voxguard.classifier.cross_eval import (
    resolve_cache_path,
    zero_shot_eval,
    zero_shot_eval_from_cache,
)
from voxguard.classifier.infer import VoxGuardDetector
from voxguard.utils.logging_utils import get_logger
from voxguard.utils.metadata import load_unified_metadata

logger = get_logger("run_cross_eval")

DEFAULT_CLASSIFIER_PATH = config.MODELS_DIR / "classifiers" / "baseline_logreg.joblib"
DEFAULT_PHASE2_REPORT = config.MODELS_DIR / "reports" / "asvspoof2019_eval_report.csv"
DEFAULT_OUTPUT = config.MODELS_DIR / "reports" / "cross_dataset_report.csv"
DEFAULT_MODEL_NAME = "wav2vec2"
DATASETS = ["wavefake", "in_the_wild"]


def _pick_phase2_reference_row(
    phase2_report_path: Path, classifier_path: str, model_name: str
) -> dict:
    """Returns the canonical ASVspoof2019 comparison row from Phase 2."""
    if phase2_report_path.exists():
        report_df = pd.read_csv(phase2_report_path)
        if {"model", "feature_set", "accuracy", "roc_auc", "eer"}.issubset(
            report_df.columns
        ):
            mask = report_df["model"].eq("logreg") & report_df["feature_set"].eq(
                "baseline"
            )
            if mask.any():
                row = report_df.loc[mask].iloc[0].to_dict()
                return {
                    "dataset": "asvspoof2019",
                    "model": row.get("model", "logreg"),
                    "feature_set": row.get("feature_set", "baseline"),
                    "accuracy": row["accuracy"],
                    "roc_auc": row["roc_auc"],
                    "eer": row["eer"],
                    "skip_count": 0,
                }

    logger.warning(
        "Phase 2 report row not found at %s; recomputing the ASVspoof2019 reference row from cache.",
        phase2_report_path,
    )
    metrics = zero_shot_eval_from_cache(
        classifier_path=classifier_path,
        model_names=[model_name],
        dataset="asvspoof2019",
    )
    return {
        "dataset": "asvspoof2019",
        "model": "logreg",
        "feature_set": "baseline",
        "accuracy": metrics["accuracy"],
        "roc_auc": metrics["roc_auc"],
        "eer": metrics["eer"],
        "skip_count": metrics.get("skip_count", 0),
    }


def _dataset_row(dataset: str, classifier_path: str, model_name: str) -> dict:
    """Scores one out-of-domain dataset, preferring cache-backed eval."""
    cache_path = Path(resolve_cache_path(model_name, dataset))
    cache_csv = cache_path.with_suffix(".csv")

    if cache_path.exists() and cache_csv.exists():
        metrics = zero_shot_eval_from_cache(
            classifier_path=classifier_path, model_names=[model_name], dataset=dataset
        )
        return {
            "dataset": dataset,
            "model": model_name,
            "feature_set": "baseline",
            "accuracy": metrics["accuracy"],
            "roc_auc": metrics["roc_auc"],
            "eer": metrics["eer"],
            "skip_count": metrics.get("skip_count", 0),
        }

    warning = f"!!! CACHE MISSING FOR {dataset.upper()} — FALLING BACK TO FILE-BY-FILE detector.predict() EVAL !!!"
    print(warning)
    logger.warning(warning)

    metadata_df = load_unified_metadata([dataset])
    detector = VoxGuardDetector(classifier_path=classifier_path, use_prosody=False)
    metrics = zero_shot_eval(detector, metadata_df)
    return {
        "dataset": dataset,
        "model": model_name,
        "feature_set": "baseline",
        "accuracy": metrics["accuracy"],
        "roc_auc": metrics["roc_auc"],
        "eer": metrics["eer"],
        "skip_count": metrics.get("skip_count", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run VoxGuard cross-dataset evaluation."
    )
    parser.add_argument(
        "--classifier_path",
        type=str,
        default=str(DEFAULT_CLASSIFIER_PATH),
        help=f"Saved classifier to score (default: {DEFAULT_CLASSIFIER_PATH}).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Backbone cache prefix to use (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--phase2_report",
        type=str,
        default=str(DEFAULT_PHASE2_REPORT),
        help=f"Phase 2 ASVspoof2019 report to copy the reference row from (default: {DEFAULT_PHASE2_REPORT}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Destination CSV for the cross-dataset comparison (default: {DEFAULT_OUTPUT}).",
    )

    args = parser.parse_args()

    classifier_path = str(args.classifier_path)
    model_name = str(args.model_name)
    phase2_report_path = Path(args.phase2_report)
    output_path = Path(args.output)

    logger.info("=" * 78)
    logger.info("VOXGUARD CROSS-DATASET EVALUATION")
    logger.info("Classifier : %s", classifier_path)
    logger.info("Backbone   : %s", model_name)
    logger.info("Phase 2 ref: %s", phase2_report_path)
    logger.info("Output     : %s", output_path)
    logger.info("=" * 78)

    rows = [_pick_phase2_reference_row(phase2_report_path, classifier_path, model_name)]
    for dataset in DATASETS:
        logger.info("Evaluating %s ...", dataset)
        rows.append(_dataset_row(dataset, classifier_path, model_name))

    report_df = pd.DataFrame(
        rows,
        columns=[
            "dataset",
            "model",
            "feature_set",
            "accuracy",
            "roc_auc",
            "eer",
            "skip_count",
        ],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, index=False)

    logger.info("Cross-dataset report:")
    print(report_df.to_string(index=False))
    logger.info("Saved cross-dataset report to %s", output_path)


if __name__ == "__main__":
    main()
