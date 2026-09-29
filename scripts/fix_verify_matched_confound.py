#!/usr/bin/env python3
"""
scripts/fix_verify_matched_confound.py — Verify the duration-matched Hindi corpus removes Issue 1.

Runs the same duration + RMS energy diagnostic that exposed Issue 1 on BOTH the
original Hindi/Hinglish track and the duration-matched track, side by side:
1. Computes duration and RMS energy for every clip (read at native sample rate).
2. Fits LogisticRegression on ONLY [duration, rms] with 5-fold stratified CV and
   reports mean accuracy and per-fold scores.
3. Repeats the fit on duration-only and RMS-only, so any residual signal can be
   attributed: duration matching fixes duration, not loudness — an RMS-only
   signal would need separate peak normalization.
4. Exits non-zero if the matched corpus's [duration, rms] accuracy is still above
   the gate (default 0.65): retraining on data that still carries the confound
   is wasted effort.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from voxguard import config
from voxguard.utils.logging_utils import get_logger

logger = get_logger("fix_verify_matched_confound")

DEFAULT_ORIGINAL_CSV = config.DATA_METADATA_DIR / "hindi_hinglish_track.csv"
DEFAULT_MATCHED_CSV = config.DATA_METADATA_DIR / "hindi_hinglish_track_matched.csv"
DEFAULT_GATE = 0.65
N_SPLITS = 5

FEATURE_SETS: Dict[str, List[int]] = {
    "duration+rms": [0, 1],
    "duration only": [0],
    "rms only": [1],
}


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else config.BASE_DIR / p


def extract_features(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (X[:, [duration, rms]], y (1 = synthetic), per-clip dataframe)."""
    df = pd.read_csv(csv_path)
    feats: List[Tuple[float, float]] = []
    for fp in df["filepath"]:
        y, sr = sf.read(str(_resolve(fp)), dtype="float32")
        feats.append((len(y) / float(sr), float(np.sqrt(np.mean(y ** 2)))))
    X = np.asarray(feats, dtype=np.float64)
    labels = df["label"].astype(str).str.lower()
    unknown = set(labels) - {"real", "synthetic"}
    if unknown:
        raise ValueError(f"{csv_path}: unexpected labels {sorted(unknown)}")
    y = (labels == "synthetic").to_numpy().astype(np.int64)
    per_clip = pd.DataFrame({"label": labels, "duration": X[:, 0], "rms": X[:, 1]})
    logger.info("Extracted [duration, rms] for %d clips from %s", len(df), csv_path.name)
    return X, y, per_clip


def evaluate(X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """5-fold stratified CV accuracy of LogisticRegression for each feature set.

    Deliberately the exact recipe that produced the Issue 1 figure (83.3%):
    default LogisticRegression on unscaled features, unshuffled
    StratifiedKFold (what ``cv=5`` resolves to for a classifier).  Changing
    it (scaling, shuffling) moves the original-corpus number to 0.85-0.88,
    which would make the before/after comparison inconsistent with the
    documented baseline.
    """
    cv = StratifiedKFold(n_splits=N_SPLITS)
    results: Dict[str, Dict[str, Any]] = {}
    for name, cols in FEATURE_SETS.items():
        model = LogisticRegression()
        scores = cross_val_score(model, X[:, cols], y, cv=cv, scoring="accuracy")
        results[name] = {"mean": float(scores.mean()), "folds": [float(s) for s in scores]}
    return results


def _class_stats(per_clip: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for label, grp in per_clip.groupby("label"):
        out[str(label)] = {
            "n": int(len(grp)),
            "dur_mean": float(grp["duration"].mean()),
            "dur_std": float(grp["duration"].std(ddof=0)),
            "rms_mean": float(grp["rms"].mean()),
            "rms_std": float(grp["rms"].std(ddof=0)),
        }
    return out


def print_report(
    stats: Dict[str, Dict[str, Dict[str, float]]],
    results: Dict[str, Dict[str, Dict[str, Any]]],
    gate: float,
) -> None:
    corpora = list(results)
    bar = "=" * 86

    print("\n" + bar)
    print(" HINDI/HINGLISH DURATION & RMS CONFOUND - ORIGINAL vs DURATION-MATCHED")
    print(bar)
    print(f" {'Corpus':<10} {'Label':<10} {'N':>4} {'Duration mean+/-std (s)':>24} {'RMS mean+/-std':>22}")
    print("-" * 86)
    for corpus in corpora:
        for label in ("real", "synthetic"):
            s = stats[corpus][label]
            print(
                f" {corpus:<10} {label:<10} {s['n']:>4} "
                f"{s['dur_mean']:>12.3f} +/- {s['dur_std']:<9.3f} "
                f"{s['rms_mean']:>10.5f} +/- {s['rms_std']:<9.5f}"
            )

    print("\n LogisticRegression, 5-fold stratified CV accuracy (chance = 0.50)")
    print("-" * 86)
    print(f" {'Features':<15} {'Corpus':<10} {'Mean':>7}   Per-fold")
    print("-" * 86)
    for feat in FEATURE_SETS:
        for corpus in corpora:
            r = results[corpus][feat]
            folds = "  ".join(f"{s:.3f}" for s in r["folds"])
            print(f" {feat:<15} {corpus:<10} {r['mean']:>7.3f}   {folds}")
        print()
    print(bar)

    matched = results["matched"]
    verdict = "PASS" if matched["duration+rms"]["mean"] <= gate else "FAIL"
    print(
        f" Gate: matched [duration, rms] accuracy {matched['duration+rms']['mean']:.3f} "
        f"{'<=' if verdict == 'PASS' else '>'} {gate:.2f}  ->  {verdict}"
    )
    dur_acc, rms_acc = matched["duration only"]["mean"], matched["rms only"]["mean"]
    if rms_acc > gate:
        print(
            f" Residual signal is from RMS energy ({rms_acc:.3f}); duration matching cannot fix a "
            "loudness difference; apply peak/loudness normalization."
        )
    if dur_acc > gate:
        print(f" Residual signal is from duration ({dur_acc:.3f}); duration matching did not take effect.")
    print(bar + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--original-csv", type=Path, default=DEFAULT_ORIGINAL_CSV)
    parser.add_argument("--matched-csv", type=Path, default=DEFAULT_MATCHED_CSV)
    parser.add_argument("--gate", type=float, default=DEFAULT_GATE,
                        help="Max allowed matched [duration, rms] accuracy (default: 0.65).")
    args = parser.parse_args()

    stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for corpus, csv_path in (("original", args.original_csv), ("matched", args.matched_csv)):
        X, y, per_clip = extract_features(csv_path)
        stats[corpus] = _class_stats(per_clip)
        results[corpus] = evaluate(X, y)

    print_report(stats, results, args.gate)

    matched_acc = results["matched"]["duration+rms"]["mean"]
    if matched_acc > args.gate:
        logger.error(
            "Matched corpus still separable on [duration, rms]: %.3f > %.2f. "
            "Do not retrain on this data.", matched_acc, args.gate,
        )
        sys.exit(1)
    logger.info("Matched corpus passes the confound gate (%.3f <= %.2f).", matched_acc, args.gate)


if __name__ == "__main__":
    main()
