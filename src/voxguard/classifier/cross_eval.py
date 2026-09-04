"""
cross_eval.py — zero-shot cross-dataset evaluation helpers (Phase 3 / v4).

Provides a file-by-file fallback for out-of-domain evaluation when cached
features are unavailable, plus cache-backed helpers for the single-backbone
and two-backbone comparison paths used in Prompt 3.4.

All helpers return the same core metric shape as
``voxguard.classifier.evaluate.evaluate_classifier``:
``accuracy``, ``roc_auc``, ``eer``, ``eer_threshold``, and
``confusion_matrix``. ``zero_shot_eval`` additionally reports ``skip_count``
so callers can see how many files were skipped because an individual
prediction failed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

from voxguard import config
from voxguard.classifier.head import MLPClassifierHead, _encode_labels, load_classifier
from voxguard.embeddings.cache import load_cached_embeddings
from voxguard.features.compose import load_combined_features
from voxguard.utils.logging_utils import get_logger

logger = get_logger(__name__)

EMBEDDINGS_DIR = config.MODELS_DIR / "embeddings"


def resolve_cache_path(model_name: str, dataset: str, split: str = "eval") -> str:
    """Resolves the canonical cache path for a backbone/model and dataset."""
    dataset = str(dataset)
    model_name = str(model_name)
    split = str(split)

    if dataset == "asvspoof2019":
        return str(EMBEDDINGS_DIR / f"{model_name}_{split}.npy")
    return str(EMBEDDINGS_DIR / f"{model_name}_{dataset}.npy")


def weighted_average_ensemble(
    prob_a: np.ndarray, prob_b: np.ndarray, weight_a: float = 0.5
) -> np.ndarray:
    """Returns the per-row weighted average of two probability vectors."""
    if not 0.0 <= weight_a <= 1.0:
        raise ValueError(f"weight_a must be in [0, 1]; got {weight_a!r}.")
    prob_a = np.asarray(prob_a, dtype=np.float64)
    prob_b = np.asarray(prob_b, dtype=np.float64)
    return weight_a * prob_a + (1.0 - weight_a) * prob_b


def _eer_and_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, float]:
    """Computes EER and the threshold where it occurs."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(thresholds[idx])
    return eer, eer_threshold


def _metric_dict(
    y_true: Sequence[object],
    y_scores: Sequence[float],
    skip_count: int = 0,
    threshold: float = 0.5,
) -> dict:
    """Builds the canonical metric dict shape used throughout Phase 3.

    *threshold* is the probability cutoff used for the label-dependent
    metrics (``accuracy``, ``confusion_matrix``); ``roc_auc``, ``eer`` and
    ``eer_threshold`` are threshold-independent and unaffected. It defaults
    to 0.5 so existing results reproduce exactly. Pass the dataset's own
    ``eer_threshold`` to score a domain whose calibration differs from
    ASVspoof2019 — on In-the-Wild, 0.5 false-positives 77% of bonafide
    clips and drags accuracy to ~51% despite an EER near 0.16.
    """
    y_true_arr = _encode_labels(np.asarray(y_true))
    y_scores_arr = np.asarray(y_scores, dtype=np.float64)

    if y_true_arr.size == 0:
        return {
            "accuracy": float("nan"),
            "roc_auc": float("nan"),
            "eer": float("nan"),
            "eer_threshold": float("nan"),
            "confusion_matrix": [[0, 0], [0, 0]],
            "skip_count": int(skip_count),
        }

    y_pred = (y_scores_arr >= float(threshold)).astype(np.int64)
    accuracy = float(accuracy_score(y_true_arr, y_pred))

    try:
        roc_auc = float(roc_auc_score(y_true_arr, y_scores_arr))
    except ValueError:
        roc_auc = float("nan")

    if np.unique(y_true_arr).size < 2:
        eer = float("nan")
        eer_threshold = float("nan")
    else:
        eer, eer_threshold = _eer_and_threshold(y_true_arr, y_scores_arr)

    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).tolist()

    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "confusion_matrix": cm,
        "skip_count": int(skip_count),
    }


def _prediction_path_from_row(row: pd.Series) -> str:
    """Selects the best on-disk path for one metadata row."""
    if "processed_path" in row.index:
        processed_path = row["processed_path"]
        if pd.notna(processed_path) and str(processed_path).strip():
            return str(processed_path)
    return str(row["filepath"])


def _predict_scores(model, X_eval: np.ndarray) -> np.ndarray:
    """Returns P(synthetic) for each row of X_eval, regardless of head type."""
    if isinstance(model, LogisticRegression):
        return np.asarray(model.predict_proba(X_eval)[:, 1], dtype=np.float64)
    if isinstance(model, MLPClassifierHead):
        return np.asarray(model.predict_proba(X_eval), dtype=np.float64)
    raise TypeError(
        f"Unsupported classifier type: {type(model)!r}. "
        "Expected sklearn.linear_model.LogisticRegression or MLPClassifierHead."
    )


def _validate_manifest_alignment(
    reference_path: str,
    reference_manifest: pd.DataFrame,
    other_path: str,
    other_manifest: pd.DataFrame,
) -> None:
    """Checks that two manifests are row-aligned exactly like load_combined_features."""
    if len(other_manifest) != len(reference_manifest):
        raise ValueError(
            f"Cannot combine cached features: '{reference_path}' has "
            f"{len(reference_manifest):,} rows but '{other_path}' has "
            f"{len(other_manifest):,} rows. Caches must come from the same "
            "DataFrame/split to be combined."
        )

    ref_filepaths = reference_manifest["filepath"].values
    other_filepaths = other_manifest["filepath"].values
    mismatches = np.flatnonzero(ref_filepaths != other_filepaths)
    if mismatches.size > 0:
        first = int(mismatches[0])
        raise ValueError(
            f"Cannot combine cached features: '{reference_path}' and '{other_path}' "
            f"do not share the same filepaths in the same row order "
            f"({mismatches.size} mismatched row(s)). First mismatch at row {first}: "
            f"'{ref_filepaths[first]}' (from {reference_path}) vs "
            f"'{other_filepaths[first]}' (from {other_path})."
        )

    ref_labels = reference_manifest["label"].values
    other_labels = other_manifest["label"].values
    label_mismatches = np.flatnonzero(ref_labels != other_labels)
    if label_mismatches.size > 0:
        first = int(label_mismatches[0])
        raise ValueError(
            f"Cannot combine cached features: '{reference_path}' and '{other_path}' "
            f"disagree on label at row {first} (filepath="
            f"'{ref_filepaths[first]}'): '{ref_labels[first]}' vs '{other_labels[first]}'."
        )


def zero_shot_eval(detector, metadata_df: pd.DataFrame) -> dict:
    """Scores every file in *metadata_df* with detector.predict and returns metrics."""
    if "label" not in metadata_df.columns:
        raise ValueError("metadata_df is missing required 'label' column.")
    if "filepath" not in metadata_df.columns:
        raise ValueError("metadata_df is missing required 'filepath' column.")

    y_true: list[object] = []
    y_scores: list[float] = []
    skip_count = 0

    for _, row in metadata_df.iterrows():
        audio_path = _prediction_path_from_row(row)
        try:
            prediction = detector.predict(audio_path)
        except Exception as exc:
            skip_count += 1
            logger.warning("Skipping %s after prediction failure: %s", audio_path, exc)
            continue

        if "probability_synthetic" not in prediction:
            skip_count += 1
            logger.warning(
                "Skipping %s because detector.predict() did not return probability_synthetic.",
                audio_path,
            )
            continue

        y_true.append(row["label"])
        y_scores.append(float(prediction["probability_synthetic"]))

    metrics = _metric_dict(y_true, y_scores, skip_count=skip_count)
    logger.info(
        "Zero-shot eval complete: n_rows=%d n_scored=%d skipped=%d accuracy=%.4f roc_auc=%.4f eer=%.4f",
        len(metadata_df),
        len(y_true),
        skip_count,
        metrics["accuracy"],
        metrics["roc_auc"],
        metrics["eer"],
    )
    return metrics


def zero_shot_eval_from_cache(
    classifier_path: str,
    model_names: list[str],
    dataset: str,
    use_prosody: bool = False,
    split: str = "eval",
    threshold: float = 0.5,
) -> dict:
    """Evaluates a saved classifier against one or more cached feature sources.

    *threshold* sets the cutoff for accuracy/confusion-matrix only (default
    0.5, preserving prior results); pass a dataset's own ``eer_threshold``
    to score it at a calibrated operating point.
    """
    if not model_names:
        raise ValueError("model_names must contain at least one backbone name.")

    cache_paths = [
        resolve_cache_path(model_name, dataset, split=split)
        for model_name in model_names
    ]
    if use_prosody:
        cache_paths.append(resolve_cache_path("prosody", dataset, split=split))

    if len(model_names) == 1 and not use_prosody:
        X_eval, manifest = load_cached_embeddings(cache_paths[0])
    else:
        X_eval, manifest = load_combined_features(cache_paths)

    model, scaler = load_classifier(classifier_path)
    scores = _predict_scores(model, scaler.transform(X_eval))
    return _metric_dict(manifest["label"].values, scores, threshold=threshold)


def zero_shot_eval_weighted_average_from_cache(
    classifier_a_path: str,
    model_a: str,
    classifier_b_path: str,
    model_b: str,
    dataset: str,
    weight_a: float = 0.5,
    use_prosody: bool = False,
    split: str = "eval",
    threshold: float = 0.5,
) -> dict:
    """Evaluates two classifiers, averages their probabilities row-wise, and scores.

    *threshold* sets the cutoff for accuracy/confusion-matrix only (default
    0.5, preserving prior results).
    """
    cache_a_path = resolve_cache_path(model_a, dataset, split=split)
    cache_b_path = resolve_cache_path(model_b, dataset, split=split)

    X_a, manifest_a = load_cached_embeddings(cache_a_path)
    X_b, manifest_b = load_cached_embeddings(cache_b_path)

    if use_prosody:
        prosody_path = resolve_cache_path("prosody", dataset, split=split)
        X_prosody, prosody_manifest = load_cached_embeddings(prosody_path)
        _validate_manifest_alignment(cache_a_path, manifest_a, cache_b_path, manifest_b)
        _validate_manifest_alignment(
            cache_a_path, manifest_a, prosody_path, prosody_manifest
        )
        _validate_manifest_alignment(
            cache_b_path, manifest_b, prosody_path, prosody_manifest
        )
        X_a = np.concatenate([X_a, X_prosody], axis=1)
        X_b = np.concatenate([X_b, X_prosody], axis=1)
        reference_manifest = manifest_a
    else:
        _validate_manifest_alignment(cache_a_path, manifest_a, cache_b_path, manifest_b)
        reference_manifest = manifest_a

    model_a_obj, scaler_a = load_classifier(classifier_a_path)
    model_b_obj, scaler_b = load_classifier(classifier_b_path)

    scores_a = _predict_scores(model_a_obj, scaler_a.transform(X_a))
    scores_b = _predict_scores(model_b_obj, scaler_b.transform(X_b))
    ensemble_scores = weighted_average_ensemble(scores_a, scores_b, weight_a=weight_a)
    return _metric_dict(
        reference_manifest["label"].values, ensemble_scores, threshold=threshold
    )
