"""
head.py — classifier heads for VoxGuard's embedding (+ prosody) feature sets (Phase 3).

Two interchangeable classifier heads sit on top of the cached feature
matrices built in Phases 1-2 (wav2vec2/WavLM embeddings, optionally
concatenated with prosody features via
``voxguard.features.compose.load_combined_features``):

  - A scikit-learn ``LogisticRegression``, trained with
    ``class_weight="balanced"`` since ASVspoof2019 is heavily imbalanced
    toward spoof (synthetic) samples.
  - A small PyTorch MLP (``MLPClassifierHead``), trained with early
    stopping on validation loss.

Both heads are trained on **standardized** features: fit a scaler with
``fit_scaler(X_train)`` and apply its ``transform`` to train/val/eval alike
before calling the training or evaluation functions.

``save_classifier`` / ``load_classifier`` persist either type to disk with a
JSON metadata sidecar recording
``{"type": "logreg"|"mlp", "input_dim": ..., "scaler_path": ...}`` so callers
(e.g. ``VoxGuardDetector``) can auto-detect which kind of model — and which
input feature dimensionality (768 baseline vs. 778 prosody-augmented) — a
saved classifier expects, without inspecting the model file itself.
``load_classifier`` returns the model and its scaler as an inseparable pair,
so no caller can score raw features against a model trained on standardized
ones.

.. note::
   **Required change for VoxGuardDetector (Prompt 2.7), not yet implemented.**
   ``load_classifier`` now returns ``(model, scaler)`` rather than just a
   model. ``VoxGuardDetector`` must therefore:

   1. unpack both in its constructor and keep the scaler as an attribute;
   2. apply ``self.scaler.transform(features)`` to the feature vector in
      every predict path — after embedding extraction (and prosody
      concatenation, when using the 778-dim head) and before scoring;
   3. use ``model.predict_proba`` for the MLP head, since ``forward`` now
      returns raw logits rather than probabilities.

   Skipping step 2 will not raise — it silently produces meaningless
   scores, because the model was fitted on standardized features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple, Union

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from voxguard.utils.logging_utils import get_logger

logger = get_logger(__name__)

# VoxGuard's canonical string labels, mapped to the binary target used by
# both classifier heads. 1 = synthetic (the positive / "risk" class).
LABEL_TO_INT = {"real": 0, "synthetic": 1}


def _encode_labels(y) -> np.ndarray:
    """Maps canonical string labels ('real'/'synthetic') to 0/1 ints.

    Arrays that are already numeric are passed through (cast to int64)
    unchanged.
    """
    y = np.asarray(y)
    if y.dtype.kind in ("U", "S", "O"):
        try:
            return np.array([LABEL_TO_INT[str(v)] for v in y], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(
                f"Unrecognized label value: {exc}. Expected 'real' or 'synthetic'."
            ) from exc
    return y.astype(np.int64)


# =============================================================================
# Feature standardization
# =============================================================================

def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fits a ``StandardScaler`` on the training features.

    Fit on the training set only — never on validation or eval data — and
    apply the returned scaler's ``transform`` to train, val, and eval
    features alike before handing them to
    :func:`train_logistic_regression`, :func:`train_mlp`, or
    ``evaluate_classifier``.

    Each feature space needs its own scaler: the 768-dim embedding-only set
    and the 778-dim prosody-augmented set are different spaces, and the
    prosody block in particular carries features whose raw magnitudes
    (``f0_range_hz`` averages ~220) are three orders of magnitude larger
    than an embedding dimension (~0.19). Left unstandardized, those columns
    dominate the L2 penalty and the gradient scale, which measurably hurt
    both heads and made the prosody feature set look worse than it is.

    Parameters
    ----------
    X_train:
        Training feature matrix, shape ``(n_samples, input_dim)``.

    Returns
    -------
    StandardScaler
        The fitted scaler. Persist it with the model via
        :func:`save_classifier`.
    """
    scaler = StandardScaler().fit(X_train)
    logger.info(
        "Fitted StandardScaler on %d training samples (input_dim=%d).",
        X_train.shape[0],
        X_train.shape[1],
    )
    return scaler


# =============================================================================
# Logistic regression head
# =============================================================================

def train_logistic_regression(X_train: np.ndarray, y_train) -> LogisticRegression:
    """Fits a class-balanced logistic-regression classifier head.

    ``class_weight="balanced"`` compensates for ASVspoof2019's heavy skew
    toward spoof (synthetic) samples.

    Parameters
    ----------
    X_train:
        Feature matrix, shape ``(n_samples, input_dim)``.
    y_train:
        Labels — either the canonical ``'real'``/``'synthetic'`` strings or
        already-encoded 0/1 ints.

    Returns
    -------
    LogisticRegression
        The fitted model.
    """
    y_enc = _encode_labels(y_train)
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_enc)
    logger.info(
        "Trained LogisticRegression on %d samples (input_dim=%d).",
        X_train.shape[0],
        X_train.shape[1],
    )
    return model


# =============================================================================
# MLP head
# =============================================================================

class MLPClassifierHead(nn.Module):
    """A 2-layer MLP classifier head: ``input_dim -> 128 -> 1``, **logit** output.

    ``input_dim`` is always read from the training data's feature
    dimensionality (never hardcoded), since it varies between the baseline
    (768-dim, embedding-only) and prosody-augmented (778-dim) feature sets.

    ``forward`` deliberately returns raw logits rather than probabilities:
    training uses ``BCEWithLogitsLoss`` (which applies the sigmoid
    internally, in a numerically stable way, and is what accepts the
    ``pos_weight`` needed to counteract ASVspoof2019's ~90% spoof
    majority). Callers that want probabilities — including inference code
    such as ``VoxGuardDetector`` — should use :meth:`predict_proba`.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits, shape ``(n_samples,)`` — NOT probabilities."""
        return self.net(x).squeeze(-1)

    def predict_proba(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Returns P(synthetic) for each row of *x* as a 1-D numpy array.

        This is the probability-valued entry point for inference — the
        sigmoid that used to live inside ``forward`` is applied here
        instead, so training can use ``BCEWithLogitsLoss`` on raw logits
        while downstream consumers still get probabilities.
        """
        self.eval()
        device = next(self.parameters()).device
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(np.asarray(x), dtype=torch.float32)
        x = x.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            return torch.sigmoid(self(x)).cpu().numpy()


def train_mlp(
    X_train: np.ndarray,
    y_train,
    X_val: np.ndarray,
    y_val,
    epochs: int = 20,
    lr: float = 1e-3,
    patience: int = 3,
    batch_size: int = 64,
    device: Optional[str] = None,
) -> MLPClassifierHead:
    """Trains ``MLPClassifierHead`` with BCEWithLogitsLoss, Adam, and early stopping.

    The loss is class-weighted via ``pos_weight = n_negative / n_positive``
    computed from *y_train*, counteracting ASVspoof2019's ~90% spoof
    majority — the same imbalance correction ``class_weight="balanced"``
    provides for the logistic-regression head. The model emits raw logits
    (see ``MLPClassifierHead``); use ``model.predict_proba`` for
    probabilities at inference time.

    ``input_dim`` is read from ``X_train.shape[1]`` — never hardcoded — so
    the same function trains both the 768-dim baseline and 778-dim
    prosody-augmented heads without modification.

    Parameters
    ----------
    X_train, y_train:
        Training feature matrix and labels (canonical strings or 0/1 ints).
    X_val, y_val:
        Validation feature matrix and labels, used for early stopping.
    epochs:
        Maximum number of training epochs.
    lr:
        Adam learning rate.
    patience:
        Number of consecutive epochs without validation-loss improvement
        before stopping early.
    batch_size:
        Mini-batch size for training.
    device:
        Torch device string. Defaults to ``config.get_device()``.

    Returns
    -------
    MLPClassifierHead
        The trained model (weights restored to the best validation-loss
        epoch), in eval mode.
    """
    from voxguard import config

    device = device or config.get_device()

    y_train_enc = _encode_labels(y_train)
    y_val_enc = _encode_labels(y_val)

    input_dim = X_train.shape[1]
    model = MLPClassifierHead(input_dim=input_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ASVspoof2019 is ~90% spoof (the positive class), which left an
    # unweighted BCE objective over-predicting spoof: it scored below the
    # trivial always-predict-majority accuracy while catching only ~26% of
    # bonafide clips. pos_weight = n_negative / n_positive rescales the
    # positive term so both classes contribute comparably, mirroring what
    # class_weight="balanced" already does for the logistic-regression head.
    n_pos = int((y_train_enc == 1).sum())
    n_neg = int((y_train_enc == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"train_mlp needs both classes present; got {n_neg} negative / {n_pos} positive samples."
        )
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_train_t = torch.tensor(np.asarray(X_train), dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train_enc, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(np.asarray(X_val), dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val_enc, dtype=torch.float32, device=device)

    n_samples = X_train_t.shape[0]

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    logger.info(
        "Training MLPClassifierHead (input_dim=%d) on %d train / %d val samples, device=%s, "
        "pos_weight=%.4f (%d neg / %d pos)",
        input_dim,
        n_samples,
        X_val_t.shape[0],
        device,
        float(pos_weight.item()),
        n_neg,
        n_pos,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_samples, device=device)
        running_loss = 0.0
        for start in range(0, n_samples, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / n_samples

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        logger.info(
            "Epoch %d/%d - train_loss=%.4f val_loss=%.4f", epoch, epochs, train_loss, val_loss
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d, best_val_loss=%.4f).",
                    epoch,
                    patience,
                    best_val_loss,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model


# =============================================================================
# Persistence
# =============================================================================

def save_classifier(
    model: Union[LogisticRegression, "MLPClassifierHead"],
    path: Union[str, Path],
    scaler: StandardScaler,
) -> None:
    """Saves a classifier head together with its fitted feature scaler.

    *path* may be given with or without an extension — the correct
    extension for the model's type (``.joblib`` for sklearn, ``.pt`` for the
    MLP's state_dict) is applied automatically via ``Path.with_suffix``. A
    JSON metadata sidecar (same base filename, ``.json``) is always written
    alongside it, recording
    ``{"type": "logreg"|"mlp", "input_dim": ..., "scaler_path": ...}``.

    *scaler* is required, not optional: a model trained on standardized
    features produces meaningless scores on raw features, so the scaler is
    part of the model's identity rather than an accessory. It is dumped to
    ``<basename>_scaler.joblib`` next to the model and referenced by
    ``scaler_path`` as a **bare filename** (not an absolute path), so a
    model directory stays portable between machines — which matters here,
    since embeddings are produced on Kaggle and consumed locally.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if scaler is None:
        raise ValueError(
            "save_classifier requires the fitted StandardScaler used on the training "
            "features; saving a model without its scaler would let downstream code "
            "score raw, unstandardized features against it."
        )

    if isinstance(model, LogisticRegression):
        model_type = "logreg"
        input_dim = int(model.coef_.shape[1])
        saved_path = path.with_suffix(".joblib")
        joblib.dump(model, saved_path)
    elif isinstance(model, MLPClassifierHead):
        model_type = "mlp"
        input_dim = model.input_dim
        saved_path = path.with_suffix(".pt")
        torch.save(model.state_dict(), saved_path)
    else:
        raise TypeError(
            f"Unsupported classifier type: {type(model)!r}. "
            "Expected sklearn.linear_model.LogisticRegression or MLPClassifierHead."
        )

    scaler_path = saved_path.with_name(f"{saved_path.stem}_scaler.joblib")
    joblib.dump(scaler, scaler_path)

    meta_path = saved_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "type": model_type,
                "input_dim": input_dim,
                "scaler_path": scaler_path.name,
            },
            f,
            indent=2,
        )

    logger.info(
        "Saved %s classifier (input_dim=%d) to %s (scaler: %s, metadata: %s)",
        model_type,
        input_dim,
        saved_path,
        scaler_path.name,
        meta_path,
    )


def load_classifier(
    path: Union[str, Path]
) -> Tuple[Union[LogisticRegression, MLPClassifierHead], StandardScaler]:
    """Loads a classifier head **and its fitted scaler** as an inseparable pair.

    Auto-detects sklearn vs. MLP via the JSON metadata sidecar (same base
    filename as *path*, ``.json`` extension) — the sidecar's ``"type"``
    field selects ``joblib.load`` vs. ``torch.load`` + ``MLPClassifierHead``
    reconstruction, its ``"input_dim"`` reconstructs the MLP with the
    correct input layer size, and its ``"scaler_path"`` locates the
    ``StandardScaler`` fitted on that model's training features.

    Returning a pair is deliberate: callers cannot obtain the model without
    also receiving the scaler it requires, so no downstream code can
    silently score raw features against a model trained on standardized
    ones. **Always** apply ``scaler.transform(X)`` before predicting.

    Parameters
    ----------
    path:
        Path to the saved classifier (with or without extension — only the
        base filename is used to locate the model, scaler, and sidecar).

    Returns
    -------
    (model, scaler):
        The fitted ``LogisticRegression`` or ``MLPClassifierHead``, and the
        ``StandardScaler`` that must be applied to features before
        inference.
    """
    path = Path(path)
    meta_path = path.with_suffix(".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Classifier metadata sidecar not found: {meta_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    model_type = meta["type"]
    input_dim = meta["input_dim"]

    if "scaler_path" not in meta:
        raise ValueError(
            f"Classifier at {meta_path} predates the feature-standardization contract "
            "(no 'scaler_path' in its metadata sidecar) and was trained on unstandardized "
            "features. Retrain it with scripts/train_classifier.py rather than loading it."
        )

    if model_type == "logreg":
        model = joblib.load(meta_path.with_suffix(".joblib"))
    elif model_type == "mlp":
        model = MLPClassifierHead(input_dim=input_dim)
        state_dict = torch.load(meta_path.with_suffix(".pt"), map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
    else:
        raise ValueError(f"Unknown classifier type in metadata sidecar {meta_path}: {model_type!r}")

    # scaler_path is stored as a bare filename, resolved next to the sidecar,
    # so the model directory can move between machines (Kaggle -> local).
    scaler_path = meta_path.parent / meta["scaler_path"]
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler referenced by {meta_path} not found at {scaler_path}. "
            "The model cannot be used without the scaler it was trained with."
        )
    scaler = joblib.load(scaler_path)

    logger.info(
        "Loaded %s classifier (input_dim=%d) + scaler from %s", model_type, input_dim, meta_path
    )
    return model, scaler
