"""
evaluate.py — classifier evaluation metrics (Phase 3).

Standard ASVspoof-style evaluation for the classifier heads trained in
``voxguard.classifier.head``: accuracy, ROC-AUC, and Equal Error Rate (EER)
— the threshold-independent metric ASVspoof challenge results are reported
in, where the false-acceptance rate equals the false-rejection rate.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

from voxguard.classifier.head import MLPClassifierHead, _encode_labels
from voxguard.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _eer_and_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """Computes EER and the decision threshold it occurs at.

    Finds the point on the ROC curve where the false-negative rate
    (``1 - tpr``) is closest to the false-positive rate, and returns the EER
    as the average of the two at that point — the standard approximation
    used when the curve is estimated from a finite set of thresholds rather
    than continuous.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(thresholds[idx])
    return eer, eer_threshold


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Computes Equal Error Rate (EER), the standard ASVspoof metric.

    Uses ``sklearn.metrics.roc_curve`` to get the fpr/tpr/thresholds, then
    finds the point where the false-negative rate (``1 - tpr``) is closest
    to the false-positive rate, and returns the EER as their average at
    that point.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels (0/1).
    y_scores:
        Continuous decision scores/probabilities for the positive class.

    Returns
    -------
    float
        The equal error rate, in ``[0, 1]``.
    """
    eer, _ = _eer_and_threshold(y_true, y_scores)
    return eer


def evaluate_classifier(
    model: Union[LogisticRegression, MLPClassifierHead],
    X_eval: np.ndarray,
    y_eval,
) -> Dict[str, object]:
    """Evaluates a trained classifier head on held-out data.

    Dispatches on model type — the same sklearn-vs-MLP typing established by
    ``voxguard.classifier.head``'s metadata sidecar (whichever type
    ``load_classifier`` reconstructed) — to get decision scores: sklearn's
    ``predict_proba`` for ``LogisticRegression``, and ``predict_proba``
    (sigmoid over the raw logits ``forward`` returns) for
    ``MLPClassifierHead``.

    Parameters
    ----------
    model:
        A fitted ``LogisticRegression`` or ``MLPClassifierHead``.
    X_eval:
        Evaluation feature matrix, shape ``(n_samples, input_dim)``.
    y_eval:
        Evaluation labels — canonical ``'real'``/``'synthetic'`` strings or
        0/1 ints.

    Returns
    -------
    dict
        ``{"accuracy": float, "roc_auc": float, "eer": float,
        "eer_threshold": float, "confusion_matrix": list[list[int]]}``
    """
    y_true = _encode_labels(y_eval)

    if isinstance(model, LogisticRegression):
        y_scores = model.predict_proba(X_eval)[:, 1]
        y_pred = model.predict(X_eval)
    elif isinstance(model, MLPClassifierHead):
        # predict_proba applies the sigmoid; forward() returns raw logits.
        y_scores = model.predict_proba(X_eval)
        y_pred = (y_scores > 0.5).astype(int)
    else:
        raise TypeError(
            f"Unsupported classifier type: {type(model)!r}. "
            "Expected sklearn.linear_model.LogisticRegression or MLPClassifierHead."
        )

    accuracy = float(accuracy_score(y_true, y_pred))
    roc_auc = float(roc_auc_score(y_true, y_scores))
    eer, eer_threshold = _eer_and_threshold(y_true, y_scores)
    cm = confusion_matrix(y_true, y_pred).tolist()

    logger.info(
        "Evaluated %s: accuracy=%.4f roc_auc=%.4f eer=%.4f (threshold=%.4f)",
        type(model).__name__,
        accuracy,
        roc_auc,
        eer,
        eer_threshold,
    )

    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "confusion_matrix": cm,
    }
