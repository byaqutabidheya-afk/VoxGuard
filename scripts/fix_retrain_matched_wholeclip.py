#!/usr/bin/env python3
"""
scripts/fix_retrain_matched_wholeclip.py — Retrain whole-clip Hindi heads on the duration-matched corpus.

Runs entirely on local CPU (~150 clips; Phase 4 Prompt 4.7 established Kaggle is
not worth the round trip at this size). Mirrors Phase 4 Prompt 4.8's structure:

1. Applies the SAME speaker-holdout split as Phase 4 (holdout_speaker='soumya')
   to data/metadata/hindi_hinglish_track_matched.csv via get_hindi_hinglish_splits.
2. For BOTH backbones (wav2vec2, WavLM), caches whole-clip embeddings with
   extract_and_cache (length-sorted batching):
     - {model}_hindi_train_matched.npy   matched train split
     - {model}_hindi_eval_matched.npy    matched eval split (needed by F1.5/F3/F6)
     - {model}_hindi_eval.npy            ORIGINAL eval split, only if not already cached
3. Per backbone independently (weighted-average ensemble: no shared feature
   space), concatenates the unchanged ASVspoof2019 train cache ({model}_train.npy)
   with the matched Hindi train embeddings, fits a StandardScaler, and trains a
   class-balanced logistic-regression head.
4. Saves models/classifiers/{model}_hindi_matched_logreg.joblib (+ scaler and
   metadata sidecars, plus a _training.json provenance record). Never touches
   {model}_hindi_combined_logreg.joblib.
5. Prints 5-fold stratified CV accuracy per backbone — overall and on the Hindi
   rows of each validation fold — and flags any score above 0.99 as SUSPICIOUS:
   a perfect score on this dataset is the signature that exposed Issue 1.

No prosody features: Phase 2 selected the baseline (non-prosody) variant.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from voxguard import config
from voxguard.classifier.head import (
    _encode_labels,
    fit_scaler,
    save_classifier,
    train_logistic_regression,
)
from voxguard.embeddings.cache import extract_and_cache, load_cached_embeddings
from voxguard.embeddings.extractor import EmbeddingExtractor
from voxguard.utils.hindi_splits import get_hindi_hinglish_splits
from voxguard.utils.logging_utils import get_logger

logger = get_logger("fix_retrain_matched_wholeclip")

MATCHED_TRACK_CSV = config.DATA_METADATA_DIR / "hindi_hinglish_track_matched.csv"
ORIGINAL_TRACK_CSV = config.DATA_METADATA_DIR / "hindi_hinglish_track.csv"
EMBEDDINGS_DIR = config.MODELS_DIR / "embeddings"
CLASSIFIERS_DIR = config.MODELS_DIR / "classifiers"

# Must match Phase 4 exactly, or the numbers are not comparable to the baseline.
HOLDOUT_SPEAKER = "soumya"
BACKBONES: Dict[str, str] = {
    "wav2vec2": "facebook/wav2vec2-base",
    "wavlm": "microsoft/wavlm-base-plus",
}
N_SPLITS = 5
RANDOM_STATE = 42
SUSPICIOUS_THRESHOLD = 0.99


def _split(track_csv: Path) -> Dict[str, pd.DataFrame]:
    df = pd.read_csv(track_csv)
    train_df, eval_df = get_hindi_hinglish_splits(
        df, mode="speaker_holdout", holdout_speaker=HOLDOUT_SPEAKER
    )
    logger.info(
        "%s: train=%d (speakers %s), eval=%d (held out '%s')",
        track_csv.name, len(train_df), sorted(train_df["speaker_id"].unique()),
        len(eval_df), HOLDOUT_SPEAKER,
    )
    return {"train": train_df.reset_index(drop=True), "eval": eval_df.reset_index(drop=True)}


def _check_cache_matches(npy_path: Path, split_df: pd.DataFrame) -> None:
    """Guard against a stale cache: its manifest must list exactly this split's clips."""
    X, manifest = load_cached_embeddings(npy_path)
    expected = split_df["filepath"].astype(str).tolist()
    actual = manifest["filepath"].astype(str).tolist()
    if X.shape[0] != len(split_df) or actual != expected:
        raise ValueError(
            f"Cached embeddings at {npy_path} do not match the current split "
            f"({X.shape[0]} rows vs {len(split_df)} expected, or filepaths differ). "
            "Delete the cache and re-run."
        )


def extract_all(batch_size: int, force: bool) -> Dict[str, List[str]]:
    """Extract matched train/eval (+ original eval if missing) for both backbones."""
    matched = _split(MATCHED_TRACK_CSV)
    original = _split(ORIGINAL_TRACK_CSV)

    jobs = [
        ("hindi_train_matched", matched["train"], force),
        ("hindi_eval_matched", matched["eval"], force),
        # Original eval is only extracted if absent; never forced, so the
        # cache the F0 baseline was scored on is not silently replaced.
        ("hindi_eval", original["eval"], False),
    ]

    status: Dict[str, List[str]] = {"extracted": [], "reused": []}
    for backbone, model_name in BACKBONES.items():
        targets = [(EMBEDDINGS_DIR / f"{backbone}_{suffix}.npy", df, f) for suffix, df, f in jobs]
        extractor = None
        if any(job_force or not npy_path.exists() for npy_path, _, job_force in targets):
            logger.info("Loading %s on CPU", model_name)
            extractor = EmbeddingExtractor(model_name=model_name, device="cpu")
        for npy_path, split_df, job_force in targets:
            if npy_path.exists() and not job_force:
                status["reused"].append(npy_path.name)
            else:
                extract_and_cache(
                    df=split_df,
                    extractor=extractor,
                    output_path=str(npy_path),
                    path_col="filepath",
                    batch_size=batch_size,
                    force=job_force,
                )
                status["extracted"].append(npy_path.name)
            _check_cache_matches(npy_path, split_df)
        del extractor
    return status


def cross_validate(
    X: np.ndarray, y: np.ndarray, is_hindi: np.ndarray
) -> Dict[str, Any]:
    """5-fold stratified CV (same fold scheme as Prompt 4.8), scored overall and on Hindi rows."""
    y_enc = _encode_labels(y)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    overall: List[float] = []
    hindi: List[float] = []
    for tr, va in skf.split(X, y_enc):
        scaler = fit_scaler(X[tr])
        model = train_logistic_regression(scaler.transform(X[tr]), y_enc[tr])
        pred = model.predict(scaler.transform(X[va]))
        correct = pred == y_enc[va]
        overall.append(float(correct.mean()))
        h = is_hindi[va]
        hindi.append(float(correct[h].mean()) if h.any() else float("nan"))
    return {
        "overall_mean": float(np.mean(overall)),
        "overall_std": float(np.std(overall)),
        "overall_folds": overall,
        "hindi_mean": float(np.nanmean(hindi)),
        "hindi_std": float(np.nanstd(hindi)),
        "hindi_folds": hindi,
    }


def train_backbone(backbone: str) -> Dict[str, Any]:
    asv_path = EMBEDDINGS_DIR / f"{backbone}_train.npy"
    hindi_path = EMBEDDINGS_DIR / f"{backbone}_hindi_train_matched.npy"
    X_asv, m_asv = load_cached_embeddings(asv_path)
    X_hi, m_hi = load_cached_embeddings(hindi_path)
    if X_asv.shape[1] != X_hi.shape[1]:
        raise ValueError(
            f"{backbone}: dim mismatch ASVspoof={X_asv.shape[1]} vs Hindi={X_hi.shape[1]}"
        )

    X = np.concatenate([X_asv, X_hi], axis=0)
    y = np.concatenate([m_asv["label"].values, m_hi["label"].values], axis=0)
    is_hindi = np.r_[np.zeros(len(X_asv), bool), np.ones(len(X_hi), bool)]
    logger.info("%s: combined train %s (ASVspoof %d + matched Hindi %d)",
                backbone, X.shape, len(X_asv), len(X_hi))

    cv = cross_validate(X, y, is_hindi)

    out_stem = CLASSIFIERS_DIR / f"{backbone}_hindi_matched_logreg"
    protected = CLASSIFIERS_DIR / f"{backbone}_hindi_combined_logreg"
    if out_stem.resolve() == protected.resolve():
        raise RuntimeError(f"Refusing to overwrite production model {protected}.joblib")

    scaler = fit_scaler(X)
    model = train_logistic_regression(scaler.transform(X), y)
    save_classifier(model, out_stem, scaler)

    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/fix_retrain_matched_wholeclip.py",
        "backbone": backbone,
        "model_name": BACKBONES[backbone],
        "track_csv": MATCHED_TRACK_CSV.relative_to(config.BASE_DIR).as_posix(),
        "split": {"mode": "speaker_holdout", "holdout_speaker": HOLDOUT_SPEAKER},
        "train_sources": {
            "asvspoof2019_train": {"cache": asv_path.name, "n": int(len(X_asv))},
            "hindi_train_matched": {"cache": hindi_path.name, "n": int(len(X_hi))},
        },
        "features": "whole-clip SSL embedding (no prosody)",
        "cv": {"n_splits": N_SPLITS, "random_state": RANDOM_STATE, **cv},
    }
    with open(out_stem.with_name(out_stem.name + "_training.json"), "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    return {"n_samples": int(len(X)), "n_hindi": int(len(X_hi)),
            "saved": out_stem.with_suffix(".joblib").name, **cv}


def _flag(score: float) -> str:
    return "  <-- SUSPICIOUS (>0.99)" if score > SUSPICIOUS_THRESHOLD else ""


def print_summary(results: Dict[str, Dict[str, Any]], status: Dict[str, List[str]]) -> List[str]:
    bar = "=" * 86
    print("\n" + bar)
    print(" MATCHED-CORPUS WHOLE-CLIP RETRAIN - 5-fold stratified CV (ASVspoof2019 + matched Hindi)")
    print(bar)
    print(f" Embeddings extracted: {', '.join(status['extracted']) or 'none'}")
    print(f" Embeddings reused:    {', '.join(status['reused']) or 'none'}")
    print("-" * 86)

    suspicious: List[str] = []
    for backbone, r in results.items():
        print(f" {backbone}  ({r['n_samples']} samples, {r['n_hindi']} Hindi)  -> {r['saved']}")
        for scope in ("overall", "hindi"):
            label = "all rows" if scope == "overall" else "Hindi rows only"
            mean, std, folds = r[f"{scope}_mean"], r[f"{scope}_std"], r[f"{scope}_folds"]
            print(f"   {label:<16} mean {mean:.4f} +/- {std:.4f}{_flag(mean)}")
            print(f"   {'':<16} folds " + "  ".join(f"{s:.3f}" for s in folds))
            if mean > SUSPICIOUS_THRESHOLD:
                suspicious.append(f"{backbone} {label} mean {mean:.4f}")
            for i, s in enumerate(folds, 1):
                if s > SUSPICIOUS_THRESHOLD:
                    suspicious.append(f"{backbone} {label} fold {i} {s:.4f}")
        print()

    print(bar)
    if suspicious:
        print(" SUSPICIOUS SCORES (> 0.99) - NOT evidence of success. A perfect score on this")
        print(" dataset is the signature of a shortcut; investigate before trusting these heads:")
        for s in suspicious:
            print(f"   - {s}")
        print(" Note: Hindi-row fold scores cover only ~20 clips each, so a single fold at 1.000")
        print(" is weak evidence alone; the held-out-speaker eval (F1.5) is the real test.")
    else:
        print(" No CV score above 0.99.")
    print(bar + "\n")
    return suspicious


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--force", action="store_true",
                        help="Re-extract the matched train/eval caches even if present.")
    args = parser.parse_args()

    t0 = time.time()
    try:
        status = extract_all(batch_size=args.batch_size, force=args.force)
        results = {b: train_backbone(b) for b in BACKBONES}
    except Exception as exc:
        logger.error("Matched whole-clip retrain failed: %s", exc)
        sys.exit(1)

    print_summary(results, status)
    logger.info("Finished in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
