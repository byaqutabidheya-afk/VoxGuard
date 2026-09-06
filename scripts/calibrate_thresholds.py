#!/usr/bin/env python
"""calibrate_thresholds.py — calibrate RISK_THRESHOLDS against the dev split.

Usage
-----
    python scripts/calibrate_thresholds.py [--split dev|eval] [--weight-a 0.5]

What it does
------------
1. Loads cached dev (or eval) embeddings for the wav2vec2 and WavLM backbones
   from models/embeddings/.
2. Scores every clip with the WeightedAverageDetector (the same pair of
   classifiers the app uses: wav2vec2_hindi_combined_logreg +
   wavlm_hindi_combined_logreg), exactly as the live inference path does.
3. Sweeps a grid of (low_max, medium_max) candidate threshold pairs.
4. For each pair, reports:
     FPR_low   — fraction of real clips left in the "low" band   (missed alert)
     FNR_high  — fraction of synthetic clips flagged "high"       (true positives, used
                 to check we're not being too conservative)
     FPR_flag  — fraction of real clips reaching "medium" or "high" (false-alarm rate
                 a user sees in practice)
     FNR_low   — fraction of synthetic clips left at "low"        (missed detection)
5. Prints a ranked table and marks the current config.RISK_THRESHOLDS values.
6. Prompts the user to accept or change the thresholds, then updates
   config.py — only after explicit user confirmation.

Why the dev split?
------------------
ASVspoof2019's dev split is the conventional calibration set (held out from
training, not used as a final-reporting eval set).  The eval split is kept
for the Phase 3 headline numbers; calibrating on it would be data leakage.

Which detector?
---------------
WeightedAverageDetector (wav2vec2_hindi_combined_logreg +
wavlm_hindi_combined_logreg, weight_a=0.5) — the same one the app runs.
Using any other classifier here would produce thresholds that don't match
the app's actual score distribution.
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make the library importable regardless of working directory
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from voxguard import config  # noqa: E402
from voxguard.classifier.cross_eval import (  # noqa: E402
    _predict_scores,
    _validate_manifest_alignment,
    weighted_average_ensemble,
)
from voxguard.classifier.head import load_classifier  # noqa: E402
from voxguard.embeddings.cache import load_cached_embeddings  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EMBEDDINGS_DIR = config.MODELS_DIR / "embeddings"
CLASSIFIERS_DIR = config.MODELS_DIR / "classifiers"

WAV2VEC2_CACHE_TEMPLATE = str(EMBEDDINGS_DIR / "wav2vec2_{split}.npy")
WAVLM_CACHE_TEMPLATE = str(EMBEDDINGS_DIR / "wavlm_{split}.npy")

CLASSIFIER_A_PATH = str(CLASSIFIERS_DIR / "wav2vec2_hindi_combined_logreg")
CLASSIFIER_B_PATH = str(CLASSIFIERS_DIR / "wavlm_hindi_combined_logreg")

CONFIG_PATH = _REPO_ROOT / "src" / "voxguard" / "config.py"

# ---------------------------------------------------------------------------
# Threshold grid to sweep
# ---------------------------------------------------------------------------
# Candidates chosen to be meaningful operating points:
# - low_max must be < medium_max
# - both must be in (0, 1)
# Each row is (low_max, medium_max).
CANDIDATE_PAIRS: list[tuple[float, float]] = [
    (0.20, 0.60),
    (0.25, 0.65),
    (0.30, 0.70),   # current default
    (0.35, 0.70),
    (0.30, 0.75),
    (0.40, 0.75),
    (0.40, 0.80),
    (0.50, 0.80),
]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def load_ensemble_scores(split: str, weight_a: float) -> tuple[np.ndarray, np.ndarray]:
    """Load dev/eval embeddings, score with the WeightedAverageDetector, return
    (scores, y_true) where y_true is 1=synthetic, 0=real."""
    cache_a = WAV2VEC2_CACHE_TEMPLATE.format(split=split)
    cache_b = WAVLM_CACHE_TEMPLATE.format(split=split)

    print(f"  Loading wav2vec2 cache: {cache_a}")
    X_a, manifest_a = load_cached_embeddings(cache_a)
    print(f"  Loading WavLM cache:    {cache_b}")
    X_b, manifest_b = load_cached_embeddings(cache_b)

    _validate_manifest_alignment(cache_a, manifest_a, cache_b, manifest_b)

    print(f"  Loading classifiers …")
    model_a, scaler_a = load_classifier(CLASSIFIER_A_PATH)
    model_b, scaler_b = load_classifier(CLASSIFIER_B_PATH)

    scores_a = _predict_scores(model_a, scaler_a.transform(X_a))
    scores_b = _predict_scores(model_b, scaler_b.transform(X_b))
    ensemble = weighted_average_ensemble(scores_a, scores_b, weight_a=weight_a)

    labels = manifest_a["label"].values
    # encode: 1 = synthetic (positive), 0 = real (negative)
    y_true = np.where(labels == "synthetic", 1, 0).astype(np.int8)

    n_real = int((y_true == 0).sum())
    n_synth = int((y_true == 1).sum())
    print(f"  Loaded {len(y_true):,} clips  ({n_real:,} real, {n_synth:,} synthetic)\n")
    return ensemble, y_true


def compute_rates(
    scores: np.ndarray,
    y_true: np.ndarray,
    low_max: float,
    medium_max: float,
) -> dict[str, float]:
    """Compute false-positive and false-negative rates for one threshold pair.

    Band assignment (from voxguard.risk.bands convention — boundary → higher band):
        score <  low_max                  → "low"
        low_max <= score <= medium_max    → "medium"
        score >  medium_max              → "high"

    Metrics
    -------
    fnr_low   : fraction of synthetic clips assigned "low"   (missed detections)
    fpr_medium_high : fraction of real clips assigned "medium" or "high" (false alarms)
    fpr_medium: fraction of real clips assigned "medium"
    fpr_high  : fraction of real clips assigned "high"
    tpr_high  : fraction of synthetic clips assigned "high" (confident true positives)
    tpr_medium_high : fraction of synthetic clips assigned "medium" or "high"
    """
    real_mask  = y_true == 0
    synth_mask = y_true == 1

    # band assignment
    low_mask    = scores <  low_max
    medium_mask = (scores >= low_max) & (scores <= medium_max)
    high_mask   = scores >  medium_max

    n_real  = real_mask.sum()
    n_synth = synth_mask.sum()

    def rate(numerator_mask: np.ndarray, denominator_mask: np.ndarray) -> float:
        denom = denominator_mask.sum()
        return float((numerator_mask & denominator_mask).sum()) / float(denom) if denom else float("nan")

    return {
        "fnr_low":           rate(low_mask,               synth_mask),
        "fpr_medium_high":   rate(medium_mask | high_mask, real_mask),
        "fpr_medium":        rate(medium_mask,             real_mask),
        "fpr_high":          rate(high_mask,               real_mask),
        "tpr_high":          rate(high_mask,               synth_mask),
        "tpr_medium_high":   rate(medium_mask | high_mask, synth_mask),
    }


def build_table(
    scores: np.ndarray,
    y_true: np.ndarray,
    current_low_max: float,
    current_medium_max: float,
) -> pd.DataFrame:
    """Return a DataFrame of metrics for all candidate pairs."""
    rows = []
    for low_max, medium_max in CANDIDATE_PAIRS:
        r = compute_rates(scores, y_true, low_max, medium_max)
        is_current = (
            abs(low_max - current_low_max) < 1e-9
            and abs(medium_max - current_medium_max) < 1e-9
        )
        rows.append(
            {
                "low_max":         low_max,
                "medium_max":      medium_max,
                "FNR_low(miss%)":  round(r["fnr_low"] * 100, 2),
                "FPR_flag(FA%)":   round(r["fpr_medium_high"] * 100, 2),
                "FPR_high(FA%)":   round(r["fpr_high"] * 100, 2),
                "TPR_high(det%)":  round(r["tpr_high"] * 100, 2),
                "current":         "← current" if is_current else "",
            }
        )
    return pd.DataFrame(rows)


def print_table(df: pd.DataFrame) -> None:
    """Pretty-print the metrics table."""
    header = (
        f"  {'low_max':>8}  {'med_max':>8}  "
        f"{'FNR_low':>10}  {'FPR_flag':>10}  "
        f"{'FPR_high':>10}  {'TPR_high':>10}  {'':>12}"
    )
    sub = (
        f"  {'':>8}  {'':>8}  "
        f"{'(miss%)':>10}  {'(false-alm%)':>10}  "
        f"{'(FA%)':>10}  {'(det%)':>10}"
    )
    sep = "  " + "-" * (len(header) - 2)
    print(header)
    print(sub)
    print(sep)
    for _, row in df.iterrows():
        marker = str(row["current"])
        print(
            f"  {row['low_max']:>8.2f}  {row['medium_max']:>8.2f}  "
            f"{row['FNR_low(miss%)']:>10.2f}  {row['FPR_flag(FA%)']:>10.2f}  "
            f"{row['FPR_high(FA%)']:>10.2f}  {row['TPR_high(det%)']:>10.2f}  "
            f"  {marker:<12}"
        )
    print()


def explain_columns() -> None:
    print(textwrap.dedent("""\
        Column guide
        ────────────
        FNR_low (miss%)     — % of synthetic clips that slip through as "low risk"
                              (missed detections). Lower is better for security.
        FPR_flag (false-alm%) — % of real clips that trigger "medium" or "high".
                              (false alarms seen by users). Lower is better for UX.
        FPR_high (FA%)      — % of real clips that trigger "high" specifically.
                              Aim for near-zero; high false alarms on real calls destroy trust.
        TPR_high (det%)     — % of synthetic clips confidently caught as "high risk".
                              Higher is better; pairs with FNR_low to show the full picture.

        Recommended selection heuristic
        ────────────────────────────────
        Pick the row where FPR_high is < 5 % AND FNR_low is minimised.
        If FNR_low < 5 % is achievable, prefer the pair with lower FPR_flag.
    """))


def patch_config(new_low_max: float, new_medium_max: float) -> None:
    """Rewrite RISK_THRESHOLDS in config.py in place."""
    text = CONFIG_PATH.read_text(encoding="utf-8")

    # Match the dict literal inside RISK_THRESHOLDS — replace only the values.
    pattern = re.compile(
        r'(RISK_THRESHOLDS\s*:\s*dict\s*=\s*\{[^}]*"low_max"\s*:\s*)'
        r'[\d.]+([^}]*"medium_max"\s*:\s*)[\d.]+',
        re.DOTALL,
    )
    new_text = pattern.sub(
        lambda m: f"{m.group(1)}{new_low_max}{m.group(2)}{new_medium_max}",
        text,
    )

    if new_text == text:
        print(
            "\n  WARNING: Pattern match failed — config.py was not modified.\n"
            "  Edit RISK_THRESHOLDS manually:\n"
            f"    low_max:    {new_low_max}\n"
            f"    medium_max: {new_medium_max}\n"
        )
        return

    CONFIG_PATH.write_text(new_text, encoding="utf-8")
    print(f"\n  config.py updated: low_max={new_low_max}, medium_max={new_medium_max}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--split",
        default="dev",
        choices=["dev", "eval"],
        help="Which cached split to score (default: dev — the calibration split).",
    )
    parser.add_argument(
        "--weight-a",
        type=float,
        default=0.5,
        metavar="W",
        help="Weight for the wav2vec2 classifier in the ensemble (default: 0.5).",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("VoxGuard — RISK_THRESHOLDS calibration")
    print("=" * 72)
    print(f"\nDetector  : WeightedAverageDetector")
    print(f"  Backbone A : facebook/wav2vec2-base  → wav2vec2_hindi_combined_logreg")
    print(f"  Backbone B : microsoft/wavlm-base-plus → wavlm_hindi_combined_logreg")
    print(f"  weight_a={args.weight_a}")
    print(f"\nSplit     : {args.split} (ASVspoof2019)")
    print()

    # ------------------------------------------------------------------
    # 1. Load scores
    # ------------------------------------------------------------------
    print("Loading embeddings and scoring …")
    scores, y_true = load_ensemble_scores(args.split, args.weight_a)

    # ------------------------------------------------------------------
    # 2. Read current config values
    # ------------------------------------------------------------------
    current = config.RISK_THRESHOLDS
    current_low    = float(current.get("low_max", 0.3))
    current_medium = float(current.get("medium_max", 0.7))
    print(f"Current config.RISK_THRESHOLDS: low_max={current_low}, medium_max={current_medium}\n")

    # ------------------------------------------------------------------
    # 3. Build and print table
    # ------------------------------------------------------------------
    df = build_table(scores, y_true, current_low, current_medium)
    print("Candidate threshold pairs — WeightedAverageDetector on ASVspoof2019 "
          f"{args.split} split\n")
    print_table(df)
    explain_columns()

    # ------------------------------------------------------------------
    # 4. Recommendation: lowest FNR_low with FPR_high < 5 %
    # ------------------------------------------------------------------
    candidates = df[df["FPR_high(FA%)"] < 5.0].copy()
    if candidates.empty:
        candidates = df.copy()
        print("  Note: no pair achieves FPR_high < 5 %; showing best FNR_low overall.\n")

    best_idx = candidates["FNR_low(miss%)"].idxmin()
    best = df.loc[best_idx]
    rec_low    = float(best["low_max"])
    rec_medium = float(best["medium_max"])

    print(
        f"  Recommendation: low_max={rec_low:.2f}, medium_max={rec_medium:.2f}\n"
        f"    FNR_low={best['FNR_low(miss%)']:.2f}%  "
        f"FPR_flag={best['FPR_flag(FA%)']:.2f}%  "
        f"FPR_high={best['FPR_high(FA%)']:.2f}%  "
        f"TPR_high={best['TPR_high(det%)']:.2f}%"
    )

    if abs(rec_low - current_low) < 1e-9 and abs(rec_medium - current_medium) < 1e-9:
        print("\n  The recommended pair matches the current config — no change needed.")
    print()

    # ------------------------------------------------------------------
    # 5. Interactive confirmation
    # ------------------------------------------------------------------
    print("-" * 72)
    print("Enter the threshold values to write to config.py, or press Enter to")
    print("accept the recommendation, or type 'q' to quit without changes.\n")

    while True:
        prompt_default = f"{rec_low:.2f},{rec_medium:.2f}"
        raw = input(
            f"  low_max,medium_max  [{prompt_default}]: "
        ).strip()

        if raw.lower() in ("q", "quit", "exit"):
            print("\n  Aborted — config.py unchanged.")
            sys.exit(0)

        if raw == "":
            new_low, new_medium = rec_low, rec_medium
        else:
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) != 2:
                print("  Please enter exactly two comma-separated values, e.g. 0.30,0.70")
                continue
            try:
                new_low    = float(parts[0])
                new_medium = float(parts[1])
            except ValueError:
                print("  Could not parse as floats. Try again.")
                continue
            if not (0.0 < new_low < new_medium < 1.0):
                print(
                    f"  Invalid: need 0.0 < low_max ({new_low}) < medium_max ({new_medium}) < 1.0"
                )
                continue

        # Confirm
        print(f"\n  Will write to config.py: low_max={new_low}, medium_max={new_medium}")
        confirm = input("  Confirm? [y/N]: ").strip().lower()
        if confirm in ("y", "yes"):
            patch_config(new_low, new_medium)
            break
        else:
            print("  Not confirmed — try again or type 'q' to quit.\n")


if __name__ == "__main__":
    main()
