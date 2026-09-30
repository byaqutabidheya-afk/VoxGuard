#!/usr/bin/env python3
"""
scripts/train_chunked_classifier.py — Train the chunk-native classifier heads (Phase F2.4).

Per backbone independently (wav2vec2, WavLM — the weighted-average ensemble has
no shared feature space):

1. Loads ``{model}_asvspoof2019_train_chunked.npy`` and
   ``{model}_hindi_train_chunked.npy`` and combines them ROW-WISE (axis=0) into
   one training matrix, as Phase 4 Prompt 4.8 did for whole-clip features. This
   is NOT the feature-axis concatenation used by the concatenated ensemble.
2. Fits a fresh ``StandardScaler`` on the chunk feature space (the chunk
   distribution differs from the whole-clip one, so the whole-clip scaler is
   never reused) and trains ``train_logistic_regression`` (class-balanced).
3. Saves ``models/classifiers/{model}_chunked_logreg.joblib`` with its metadata
   sidecar (type, input_dim, scaler_path), the scaler, and a ``_training.json``
   provenance record.
4. Reports 5-fold GROUP-aware cross-validation with ``parent_filepath`` as the
   group key (``StratifiedGroupKFold``). Every chunk of a clip lands in the same
   fold, so chunks of one clip never straddle train and validation. Plain
   ``StratifiedKFold`` would put near-duplicate neighbouring windows of one clip
   on both sides of the split and inflate the score. The scaler is refit inside
   each fold on that fold's training chunks only.

All CV scores are CHUNK-level (one row = one chunk, labels inherited from the
parent clip). They are not comparable to clip-level whole-clip CV numbers; clip
aggregation is done in evaluate_chunked.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from voxguard import config
from voxguard.classifier.evaluate import compute_eer
from voxguard.classifier.head import (
    _encode_labels,
    fit_scaler,
    save_classifier,
    train_logistic_regression,
)
from voxguard.embeddings.cache import load_cached_embeddings
from voxguard.utils.logging_utils import get_logger

logger = get_logger("train_chunked_classifier")

EMBEDDINGS_DIR = config.MODELS_DIR / "embeddings"
CLASSIFIERS_DIR = config.MODELS_DIR / "classifiers"
BACKBONES = ["wav2vec2", "wavlm"]
N_SPLITS = 5
RANDOM_STATE = 42
INPUT_DIM = 768
SUSPICIOUS_THRESHOLD = 0.99


def load_chunked_train(
    embeddings_dir: Path, backbone: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Row-wise combines the ASVspoof2019 and Hindi chunked train caches.

    Returns ``(X, y, groups, is_hindi)``: chunk embeddings, string labels, the
    ``parent_filepath`` group key per chunk, and a mask of the Hindi rows.
    """
    parts = {}
    for dataset in ("asvspoof2019", "hindi"):
        path = embeddings_dir / f"{backbone}_{dataset}_train_chunked.npy"
        X, manifest = load_cached_embeddings(path)
        if "parent_filepath" not in manifest.columns or manifest["parent_filepath"].isna().any():
            raise ValueError(f"{path.with_suffix('.csv')} needs a fully populated parent_filepath column.")
        if X.shape[1] != INPUT_DIM:
            raise ValueError(f"{path.name}: expected {INPUT_DIM}-dim chunks, got {X.shape[1]}.")
        parts[dataset] = (X, manifest)

    (X_asv, m_asv), (X_hi, m_hi) = parts["asvspoof2019"], parts["hindi"]

    overlap = set(m_asv["parent_filepath"]) & set(m_hi["parent_filepath"])
    if overlap:
        raise ValueError(
            f"{len(overlap)} parent_filepath values appear in both datasets "
            f"(e.g. {sorted(overlap)[:3]}); group keys must be unique per clip."
        )

    X = np.concatenate([X_asv, X_hi], axis=0)  # row-wise, NOT feature-axis
    y = np.concatenate([m_asv["label"].values, m_hi["label"].values], axis=0)
    groups = np.concatenate([m_asv["parent_filepath"].values, m_hi["parent_filepath"].values])
    is_hindi = np.r_[np.zeros(len(X_asv), bool), np.ones(len(X_hi), bool)]

    # Chunks inherit the parent label, so a group with mixed labels means a broken manifest
    # (and would make StratifiedGroupKFold's stratification meaningless).
    labels_per_group = pd.Series(y).groupby(groups).nunique()
    if (labels_per_group > 1).any():
        bad = labels_per_group[labels_per_group > 1].index[:3].tolist()
        raise ValueError(f"Clips with chunks of conflicting labels, e.g. {bad}")

    logger.info(
        "%s: combined chunk train %s (ASVspoof %d + Hindi %d chunks; %d clips)",
        backbone, X.shape, len(X_asv), len(X_hi), len(labels_per_group),
    )
    return X, y, groups, is_hindi


def cross_validate_grouped(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    is_hindi: np.ndarray,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """5-fold StratifiedGroupKFold CV on chunks, grouped by parent clip.

    Per fold the scaler is fit on the training chunks only. Each fold's
    train/validation group sets are asserted disjoint — the property this CV
    exists to guarantee.
    """
    y_enc = _encode_labels(y)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds: List[Dict[str, float]] = []
    for fold, (tr, va) in enumerate(sgkf.split(X, y_enc, groups), start=1):
        shared = set(groups[tr]) & set(groups[va])
        if shared:
            raise RuntimeError(f"Fold {fold}: {len(shared)} clips straddle train/val — CV is leaking.")

        scaler = fit_scaler(X[tr])
        model = train_logistic_regression(scaler.transform(X[tr]), y_enc[tr])
        X_va = scaler.transform(X[va])
        pred = model.predict(X_va)
        score = model.predict_proba(X_va)[:, 1]

        hi = is_hindi[va]
        y_va = y_enc[va]
        folds.append({
            "n_val_chunks": int(len(va)),
            "n_val_clips": int(len(set(groups[va]))),
            "accuracy": float(accuracy_score(y_va, pred)),
            "real_recall": float(recall_score(y_va, pred, pos_label=0)),
            "synthetic_recall": float(recall_score(y_va, pred, pos_label=1)),
            "roc_auc": float(roc_auc_score(y_va, score)),
            "eer": float(compute_eer(y_va, score)),
            "hindi_accuracy": float(accuracy_score(y_va[hi], pred[hi])) if hi.any() else float("nan"),
        })

    keys = ["accuracy", "real_recall", "synthetic_recall", "roc_auc", "eer", "hindi_accuracy"]
    summary = {
        k: {
            "mean": float(np.nanmean([f[k] for f in folds])),
            "std": float(np.nanstd([f[k] for f in folds])),
        }
        for k in keys
    }
    return {"n_splits": n_splits, "random_state": random_state, "folds": folds, "summary": summary}


def train_backbone(
    backbone: str, embeddings_dir: Path, output_dir: Path, n_splits: int, random_state: int
) -> Dict[str, Any]:
    X, y, groups, is_hindi = load_chunked_train(embeddings_dir, backbone)

    cv = cross_validate_grouped(X, y, groups, is_hindi, n_splits, random_state)

    # Final model: fresh scaler on the chunk feature space, all training chunks.
    scaler = fit_scaler(X)
    model = train_logistic_regression(scaler.transform(X), y)
    out_stem = output_dir / f"{backbone}_chunked_logreg"
    save_classifier(model, out_stem, scaler)

    meta_path = out_stem.with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    if meta["type"] != "logreg" or meta["input_dim"] != INPUT_DIM or not meta.get("scaler_path"):
        raise RuntimeError(f"Unexpected metadata sidecar for {out_stem}: {meta}")

    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/train_chunked_classifier.py",
        "backbone": backbone,
        "features": "chunk-level SSL embedding (no prosody); labels inherited from parent clip",
        "train_sources": {
            "asvspoof2019_train_chunked": int((~is_hindi).sum()),
            "hindi_train_chunked": int(is_hindi.sum()),
        },
        "n_train_clips": int(len(set(groups))),
        "label_counts": {k: int(v) for k, v in pd.Series(y).value_counts().items()},
        "cv": {"scheme": "StratifiedGroupKFold", "group_key": "parent_filepath", **cv},
    }
    out_stem.with_name(out_stem.name + "_training.json").write_text(json.dumps(provenance, indent=2))

    return {
        "n_chunks": int(len(X)),
        "n_clips": int(len(set(groups))),
        "n_hindi_chunks": int(is_hindi.sum()),
        "saved": out_stem.with_suffix(".joblib").name,
        "cv": cv,
    }


def print_summary(results: Dict[str, Dict[str, Any]]) -> List[str]:
    bar = "=" * 92
    print("\n" + bar)
    print(" CHUNKED HEAD TRAINING - 5-fold StratifiedGroupKFold CV (group = parent_filepath)")
    print(" Chunk-level scores; chunks of one clip never straddle a fold. Not clip-level metrics.")
    print(bar)

    suspicious: List[str] = []
    for backbone, r in results.items():
        s = r["cv"]["summary"]
        print(f" {backbone}  ({r['n_chunks']} chunks / {r['n_clips']} clips, "
              f"{r['n_hindi_chunks']} Hindi chunks)  -> {r['saved']}")
        for key, label in [
            ("accuracy", "accuracy"),
            ("real_recall", "recall (real)"),
            ("synthetic_recall", "recall (synthetic)"),
            ("roc_auc", "ROC-AUC"),
            ("eer", "EER"),
            ("hindi_accuracy", "accuracy, Hindi rows"),
        ]:
            m, sd = s[key]["mean"], s[key]["std"]
            flag = ""
            # Only the Hindi rows: a perfect score there was the Issue 1 shortcut signature.
            # In-distribution ASVspoof AUC/accuracy near 0.99 is expected, not suspicious.
            if key == "hindi_accuracy" and m > SUSPICIOUS_THRESHOLD:
                flag = "  <-- SUSPICIOUS (>0.99)"
                suspicious.append(f"{backbone} {label} mean {m:.4f}")
            print(f"   {label:<22} {m:.4f} +/- {sd:.4f}{flag}")
        print(f"   {'fold accuracies':<22} " + "  ".join(f"{f['accuracy']:.3f}" for f in r["cv"]["folds"]))
        print()
    print(bar)
    if suspicious:
        print(" SUSPICIOUS SCORES - investigate before trusting these heads:")
        for item in suspicious:
            print(f"   - {item}")
        print(bar)
    print(" Note: ASVspoof chunks are ~92% synthetic, so plain accuracy is dominated by that class;")
    print(" read recall (real), ROC-AUC and EER alongside it.")
    print(bar + "\n")
    return suspicious


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--embeddings_dir", type=str, default=str(EMBEDDINGS_DIR))
    parser.add_argument("--output_dir", type=str, default=str(CLASSIFIERS_DIR))
    parser.add_argument("--n_splits", type=int, default=N_SPLITS)
    parser.add_argument("--random_state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    t0 = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        results = {
            b: train_backbone(b, Path(args.embeddings_dir), output_dir, args.n_splits, args.random_state)
            for b in BACKBONES
        }
    except Exception as exc:
        logger.error("Chunked classifier training failed: %s", exc)
        sys.exit(1)

    print_summary(results)
    logger.info("Finished in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
