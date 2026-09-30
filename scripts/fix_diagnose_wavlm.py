#!/usr/bin/env python3
"""
scripts/fix_diagnose_wavlm.py — Diagnose why WavLM's Hindi head misjudges an unseen speaker (Phase F3.1).

DIAGNOSIS ONLY. Every head trained here is a throwaway held in memory; no model file is
written, and the script snapshots ``models/classifiers/`` before and after and aborts if
anything changed. Output: ``models/reports/fix_wavlm_diagnosis.md`` (+ a small ``.json``
of the verdicts for the F3.2 scripts).

Hypotheses
----------
A  Speaker shortcut. (1) speaker (byaquta vs mahato) probe per backbone; (2) a Hindi-only
   label head scored on held-out clips of the SAME speakers vs on soumya; plus a
   leave-one-training-speaker-out (LOSO) check that gives two more unseen-speaker points.
B  Hindi rows drowned out by ASVspoof. Norm of the saved heads' coefficient vectors and their
   decision values on Hindi-train / Hindi-validation / ASVspoof-validation rows, against an
   ASVspoof-ONLY reference head (a norm means nothing without one).
C  Regularisation. C in [0.001, 0.01, 0.1, 1, 10] for the WavLM Hindi head, crossed with the
   Hindi oversampling factors the guide names (1, 5, 10, 20x), for the three head families.

EXPLORATORY additions (labelled as such in the report; added after a first run showed the specified B/C
tests could not see the oversampling effect): a B "oversampling rescue" test, a C x oversampling verdict, and
a per-speaker domain-outlier check (D). The pre-declared LOSO selector (balanced accuracy) stays primary;
LOSO-AUC is reported beside it as a sensitivity check because oversampling shifts calibration.

Issue 3 metric, reproduced for the OLD whole-clip heads, the F1 MATCHED whole-clip heads and the
F2 CHUNKED heads: per-backbone probabilities on soumya's 25 real clips and the count with
ensemble margin > 0.15 from 0.5 (both the original direction-agnostic definition and the
stricter "confident AND correct").

Method notes that matter for reading the report
-----------------------------------------------
* Chunked heads are scored per chunk and aggregated to a clip by MEAN, so every family is
  reported at clip level and the numbers are comparable.
* Hyperparameters are selected on LOSO validation (train on ASVspoof + ONE Hindi speaker,
  validate on the other), NOT on soumya. Choosing a cell by soumya's score and then reporting
  soumya's score would be selection on the test set — the overfit-to-the-symptom failure the
  guide warns about in F3.2b. soumya is used only to confirm the selected cell.
* "Held-out clips of the same speakers" are grouped by utterance (a real clip and its clone share
  a group, and all chunks of a clip share it) so a clip never sits in train while its own clone
  is held out.
* The throwaway recipe is the production one (StandardScaler + class-balanced LogisticRegression,
  max_iter=1000); at C=1 / no oversampling it must reproduce the saved heads, and the report
  states the measured difference.
* Sample sizes are tiny (25 real + 25 synthetic clips per speaker). Wilson 95% intervals are
  printed and the verdict thresholds are heuristics, stated in the report.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from voxguard import config
from voxguard.classifier.head import _encode_labels, load_classifier
from voxguard.embeddings.cache import load_cached_embeddings
from voxguard.utils.logging_utils import get_logger

logger = get_logger("fix_diagnose_wavlm")

EMB = config.MODELS_DIR / "embeddings"
CLF = config.MODELS_DIR / "classifiers"
DEFAULT_REPORT_PATH = config.MODELS_DIR / "reports" / "fix_wavlm_diagnosis.md"
BASELINE_JSON = config.MODELS_DIR / "reports" / "fix_baseline.json"

BACKBONES = ("wav2vec2", "wavlm")
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
OVERSAMPLE_GRID = [1, 5, 10, 20]
CURRENT_CELL = (1.0, 1)
SEEDS = [0, 1, 2, 3, 4]
N_SPLITS = 5
WEIGHT_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

MARGIN = 0.15               # Issue 3 metric: |P - 0.5| > MARGIN
GATE_CONFIDENT = 18         # Gate F3: confident real clips out of 25
SYNTH_RECALL_TOLERANCE = 0.03   # Gate F3: synthetic recall may drop at most 3 pp vs F0

# Verdict heuristics (documented in the report; they are judgement calls, not statistics).
SPK_DIFF_MARKED = 0.05      # A1: WavLM speaker accuracy exceeds wav2vec2's by this much = "markedly"
GAP_DIFF_LARGE = 0.15       # A2: WavLM's (same-speaker - soumya) gap exceeds wav2vec2's by this much
B_FIT_MIN = 0.90            # B: head fits its own Hindi TRAIN rows worse than this = Hindi drowned
B_NEAR_ABS = 0.40           # B: share of Hindi-val rows with |logit| < 1 ...
B_NEAR_RATIO = 2.0          # ... and at least this many times the ASV-val share = clustered at boundary
C_IMPROVE = 0.15            # C: low-C real-clip accuracy gain over C=1 that counts as "strong"
# Exploratory checks added after the first run showed the spec's B/C tests cannot see the
# oversampling effect; labelled as such in the report.
B_REMEDY_AUC = 0.10         # oversampling (C=1) must lift WavLM-alone soumya AUC by this much (and real acc by C_IMPROVE)
ADEQUATE_AUC = 0.95         # a WavLM-alone soumya AUC at/above this = the backbone CAN separate the classes
D_OUTLIER_GAP = 0.30        # D: soumya real-clip accuracy this far below the worst training speaker = domain outlier

FAMILIES: Dict[str, Dict[str, str]] = {
    "old": {
        "label": "OLD whole-clip (F0: hindi_combined, original audio)",
        "asv_train": "{b}_train", "hi_train": "{b}_hindi_train", "hi_eval": "{b}_hindi_eval",
        "asv_val": "{b}_eval", "head": "{b}_hindi_combined_logreg",
    },
    "matched": {
        "label": "MATCHED whole-clip (F1: hindi_matched)",
        "asv_train": "{b}_train", "hi_train": "{b}_hindi_train_matched", "hi_eval": "{b}_hindi_eval_matched",
        "asv_val": "{b}_eval", "head": "{b}_hindi_matched_logreg",
    },
    "chunked": {
        "label": "CHUNKED (F2: chunked_logreg, mean-aggregated to clips)",
        "asv_train": "{b}_asvspoof2019_train_chunked", "hi_train": "{b}_hindi_train_chunked",
        "hi_eval": "{b}_hindi_eval_chunked", "asv_val": "{b}_asvspoof2019_eval_chunked",
        "head": "{b}_chunked_logreg",
    },
}
DECISION_FAMILIES = ("matched", "chunked")   # the families a fix would be promoted from


# =============================================================================
# Data
# =============================================================================

@dataclass(frozen=True)
class Rows:
    X: np.ndarray
    y: np.ndarray       # 0 = real, 1 = synthetic
    clip: np.ndarray    # source-clip key (parent_filepath for chunk caches)


def _norm(p: Any) -> str:
    return str(p).replace("\\", "/")


def _stem(p: str) -> str:
    return Path(p).stem


def speaker_of(p: str) -> str:
    return _stem(p).split("_")[0]


def pair_key(p: str) -> str:
    """Utterance key shared by a real clip and its clone."""
    s = _stem(p)
    return s[: -len("_clone")] if s.endswith("_clone") else s


@lru_cache(maxsize=None)
def load_rows(stem: str) -> Rows:
    X, m = load_cached_embeddings(EMB / f"{stem}.npy")
    col = "parent_filepath" if "parent_filepath" in m.columns else "filepath"
    return Rows(X, _encode_labels(m["label"].values), np.array([_norm(c) for c in m[col]]))


def fam_rows(fam: str, part: str, backbone: str) -> Rows:
    return load_rows(FAMILIES[fam][part].format(b=backbone))


def hindi_meta(rows: Rows) -> Tuple[np.ndarray, np.ndarray]:
    return (np.array([speaker_of(c) for c in rows.clip]), np.array([pair_key(c) for c in rows.clip]))


def snapshot_models() -> Dict[str, Tuple[int, int]]:
    return {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(CLF.glob("*")) if p.is_file()}


# =============================================================================
# Fitting / metrics
# =============================================================================

def fit_lr(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> Tuple[LogisticRegression, StandardScaler]:
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, C=C).fit(scaler.transform(X), y)
    return model, scaler


def predict(model: LogisticRegression, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(scaler.transform(X))[:, 1]


def clip_level(y_row: np.ndarray, p_row: np.ndarray, clip_row: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-aggregates row scores to one per clip (identity for whole-clip caches). Sorted by clip key."""
    g = pd.DataFrame({"clip": clip_row, "y": y_row, "p": p_row}).groupby("clip", sort=True)
    if (g["y"].nunique() > 1).any():
        raise ValueError("A clip has rows with conflicting labels.")
    return g["y"].first().values, g["p"].mean().values, g["y"].first().index.values


def metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, Any]:
    pred = p >= 0.5
    real, synth = y == 0, y == 1
    both = real.any() and synth.any()
    return {
        "acc": float((pred == y).mean()),
        "auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "real_acc": float((~pred[real]).mean()) if real.any() else float("nan"),
        "synth_recall": float(pred[synth].mean()) if synth.any() else float("nan"),
        "n": int(len(y)), "n_real": int(real.sum()), "n_synth": int(synth.sum()),
        "k_correct": int((pred == y).sum()),
    }


def mean_metrics(ms: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {k: float(np.nanmean([m[k] for m in ms])) for k in ("acc", "auc", "real_acc", "synth_recall")}
    out["acc_sd"] = float(np.std([m["acc"] for m in ms]))
    out["n"] = ms[0]["n"]
    out["k_correct"] = float(np.mean([m["k_correct"] for m in ms]))
    return out


def wilson(k: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def ens_summary(p_a: np.ndarray, p_b: np.ndarray, y: np.ndarray, w: float = 0.5) -> Dict[str, Any]:
    """Issue 3 numbers for w*p_a + (1-w)*p_b. w=1 -> backbone a alone; w=0 -> backbone b alone."""
    p = w * p_a + (1.0 - w) * p_b
    real, synth = y == 0, y == 1
    pr = p[real]
    confident = np.abs(pr - 0.5) > MARGIN
    m = metrics(y, p)
    return {
        "n_real": int(real.sum()),
        "real_confident": int(confident.sum()),                       # original, direction-agnostic
        "real_confident_correct": int((confident & (pr < 0.5)).sum()),  # stricter
        "real_correct": int((pr < 0.5).sum()),
        "disagree": int(((p_a[real] >= 0.5) != (p_b[real] >= 0.5)).sum()),
        "median_real_p": float(np.median(pr)),
        "synth_recall": m["synth_recall"], "auc": m["auc"], "real_acc": m["real_acc"],
    }


# =============================================================================
# Saved heads -> Issue 3 metric
# =============================================================================

def score_saved_heads() -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Clip-level P(synthetic) on soumya's eval clips from each SAVED head."""
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for fam, spec in FAMILIES.items():
        for b in BACKBONES:
            model, scaler = load_classifier(CLF / spec["head"].format(b=b))
            ev = fam_rows(fam, "hi_eval", b)
            y, p, clips = clip_level(ev.y, predict(model, scaler, ev.X), ev.clip)
            out[(fam, b)] = {"y": y, "p": p, "clip": clips}
        a, w = out[(fam, "wav2vec2")], out[(fam, "wavlm")]
        if not (np.array_equal(a["clip"], w["clip"]) and np.array_equal(a["y"], w["y"])):
            raise ValueError(f"{fam}: wav2vec2 / wavlm eval clips are not aligned.")
    return out


def issue3(saved: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    res: Dict[str, Dict[str, Any]] = {}
    for fam in FAMILIES:
        a, w = saved[(fam, "wav2vec2")], saved[(fam, "wavlm")]
        res[fam] = {
            "wav2vec2": ens_summary(a["p"], w["p"], a["y"], 1.0),
            "wavlm": ens_summary(a["p"], w["p"], a["y"], 0.0),
            "ens": ens_summary(a["p"], w["p"], a["y"], 0.5),
        }
    return res


def check_f0_reproduction(saved, base: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The OLD family must reproduce fix_baseline.json's 7/25 and its per-clip probabilities."""
    if base is None:
        return None
    a, w = saved[("old", "wav2vec2")], saved[("old", "wavlm")]
    ens = 0.5 * a["p"] + 0.5 * w["p"]
    got = {Path(c).name: float(p) for c, p, y in zip(a["clip"], ens, a["y"]) if y == 0}
    rec = {c["filename"]: c["ensemble_prob_synthetic"] for c in base["item4_soumya_predictions"]["clips"]}
    if set(got) != set(rec):
        return {"ok": False, "detail": "clip sets differ", "max_diff": float("nan")}
    diff = max(abs(got[k] - rec[k]) for k in got)
    n_got = sum(abs(p - 0.5) > MARGIN for p in got.values())
    n_base = base["item4_soumya_predictions"]["confident_predictions_count"]
    return {"ok": diff < 1e-6 and n_got == n_base, "max_diff": diff, "count": n_got, "baseline_count": n_base}


# =============================================================================
# Hypothesis A
# =============================================================================

def oof_predict(X: np.ndarray, target: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    oof = np.zeros(len(target))
    for tr, va in StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=seed).split(X, target, groups):
        model, scaler = fit_lr(X[tr], target[tr])
        oof[va] = predict(model, scaler, X[va])
    return oof


def run_a(fam: str, b: str) -> Dict[str, Any]:
    hi, ev = fam_rows(fam, "hi_train", b), fam_rows(fam, "hi_eval", b)
    spk, groups = hindi_meta(hi)
    if set(spk) != {"byaquta", "mahato"}:
        raise ValueError(f"{fam}/{b}: expected Hindi train speakers byaquta+mahato, got {sorted(set(spk))}")
    if set(hindi_meta(ev)[0]) != {"soumya"}:
        raise ValueError(f"{fam}/{b}: Hindi eval is not soumya only.")
    spk_int = (spk == "mahato").astype(int)

    # A1: speaker probe (grouped CV; chance = majority share).
    accs, aucs, lls = [], [], []
    for s in SEEDS:
        oof = oof_predict(hi.X, spk_int, groups, s)
        accs.append(float(((oof >= 0.5) == spk_int).mean()))
        aucs.append(float(roc_auc_score(spk_int, oof)))
        lls.append(float(log_loss(spk_int, np.clip(oof, 1e-6, 1 - 1e-6))))
    a1 = {"acc": float(np.mean(accs)), "acc_sd": float(np.std(accs)), "auc": float(np.mean(aucs)),
          "logloss": float(np.mean(lls)), "chance": float(max(spk_int.mean(), 1 - spk_int.mean()))}

    # A2 (i): label head, held-out clips of the SAME speakers.
    same = []
    for s in SEEDS:
        oof = oof_predict(hi.X, hi.y, groups, s)
        yc, pc, _ = clip_level(hi.y, oof, hi.clip)
        same.append(metrics(yc, pc))
    # A2 (ii): trained on all Hindi train, scored on soumya.
    model, scaler = fit_lr(hi.X, hi.y)
    yc, pc, _ = clip_level(ev.y, predict(model, scaler, ev.X), ev.clip)
    soumya = metrics(yc, pc)
    # A2 (iii): LOSO inside the training speakers.
    loso = []
    for held in ("byaquta", "mahato"):
        tr, va = spk != held, spk == held
        m2, s2 = fit_lr(hi.X[tr], hi.y[tr])
        yc, pc, _ = clip_level(hi.y[va], predict(m2, s2, hi.X[va]), hi.clip[va])
        loso.append(metrics(yc, pc))
    return {"speaker": a1, "same": mean_metrics(same), "soumya": soumya, "loso": mean_metrics(loso),
            "n_train_rows": int(len(hi.y)), "n_train_clips": int(len(set(hi.clip)))}


def verdict_a(A: Dict, fam: str) -> Tuple[str, str]:
    w, v = A[(fam, "wavlm")], A[(fam, "wav2vec2")]
    spk_diff = w["speaker"]["acc"] - v["speaker"]["acc"]
    gap_w = w["same"]["acc"] - w["soumya"]["acc"]
    gap_v = v["same"]["acc"] - v["soumya"]["acc"]
    gap_diff = gap_w - gap_v
    worse = w["soumya"]["acc"] < v["soumya"]["acc"]
    detail = (f"speaker acc WavLM {w['speaker']['acc']:.3f} vs wav2vec2 {v['speaker']['acc']:.3f} (diff {spk_diff:+.3f}); "
              f"same-speaker -> soumya gap WavLM {gap_w:+.3f} vs wav2vec2 {gap_v:+.3f} (diff {gap_diff:+.3f}); "
              f"soumya acc WavLM {w['soumya']['acc']:.3f} vs wav2vec2 {v['soumya']['acc']:.3f}")
    if gap_diff >= GAP_DIFF_LARGE and worse and spk_diff >= SPK_DIFF_MARKED:
        return "SUPPORTED", detail
    if gap_diff >= GAP_DIFF_LARGE and worse:
        return "PARTIAL", detail + " (generalisation gap present, but speaker decodability is not markedly higher)"
    return "NOT SUPPORTED", detail


# =============================================================================
# Hypothesis B
# =============================================================================

def z_stats(z: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    per = {}
    for cls, name in ((0, "real"), (1, "synth")):
        zc = z[y == cls]
        per[name] = {"median_z": float(np.median(zc)), "median_abs": float(np.median(np.abs(zc))),
                     "near1": float((np.abs(zc) < 1.0).mean())}
    return {
        "n": int(len(z)),
        "median_z_real": per["real"]["median_z"], "median_z_synth": per["synth"]["median_z"],
        "median_abs": float(np.mean([per["real"]["median_abs"], per["synth"]["median_abs"]])),  # class-macro
        "near1": float(np.mean([per["real"]["near1"], per["synth"]["near1"]])),                  # class-macro
        "acc_row": float(((z >= 0) == (y == 1)).mean()),
        "auc_row": float(roc_auc_score(y, z)),
    }


def run_b(fam: str, b: str) -> Dict[str, Any]:
    model, scaler = load_classifier(CLF / FAMILIES[fam]["head"].format(b=b))
    hi_tr, hi_ev = fam_rows(fam, "hi_train", b), fam_rows(fam, "hi_eval", b)
    asv_val, asv_tr = fam_rows(fam, "asv_val", b), fam_rows(fam, "asv_train", b)

    def z(rows: Rows) -> np.ndarray:
        return model.decision_function(scaler.transform(rows.X))

    sets = {"hi_train": z_stats(z(hi_tr), hi_tr.y), "hi_val": z_stats(z(hi_ev), hi_ev.y),
            "asv_val": z_stats(z(asv_val), asv_val.y)}

    ref_model, ref_scaler = fit_lr(asv_tr.X, asv_tr.y)   # ASVspoof-ONLY reference head
    z_ref_hi = ref_model.decision_function(ref_scaler.transform(hi_ev.X))
    cos = float(np.dot(model.coef_[0], ref_model.coef_[0]) /
                (np.linalg.norm(model.coef_) * np.linalg.norm(ref_model.coef_)))
    yc, pc, _ = clip_level(hi_ev.y, predict(ref_model, ref_scaler, hi_ev.X), hi_ev.clip)

    # D: the ASV-only head never saw any Hindi, so its per-speaker accuracy is a clean read of
    # whether soumya is a domain outlier relative to the two training speakers.
    all_y = np.concatenate([hi_tr.y, hi_ev.y])
    all_clip = np.concatenate([hi_tr.clip, hi_ev.clip])
    yc_all, pc_all, cc_all = clip_level(all_y, predict(ref_model, ref_scaler, np.concatenate([hi_tr.X, hi_ev.X])), all_clip)
    spk_c = np.array([speaker_of(c) for c in cc_all])
    by_speaker = {}
    for s in ("byaquta", "mahato", "soumya"):
        real, synth = (spk_c == s) & (yc_all == 0), (spk_c == s) & (yc_all == 1)
        by_speaker[s] = {"real_acc": float((pc_all[real] < 0.5).mean()), "median_p_real": float(np.median(pc_all[real])),
                         "synth_recall": float((pc_all[synth] >= 0.5).mean())}
    return {
        "coef_norm": float(np.linalg.norm(model.coef_)), "ref_coef_norm": float(np.linalg.norm(ref_model.coef_)),
        "cosine_vs_asv_only": cos, "z_corr_vs_asv_only": float(np.corrcoef(z(hi_ev), z_ref_hi)[0, 1]),
        "asv_only_soumya": metrics(yc, pc), "asv_only_by_speaker": by_speaker, "sets": sets,
    }


def verdict_b(B: Dict, fam: str) -> Tuple[str, str]:
    w = B[(fam, "wavlm")]
    fit, nh, na = w["sets"]["hi_train"]["acc_row"], w["sets"]["hi_val"]["near1"], w["sets"]["asv_val"]["near1"]
    detail = (f"WavLM head fits its own Hindi TRAIN rows at {fit:.3f}; share of |logit|<1: Hindi-val {nh:.2f} vs ASV-val {na:.2f}; "
              f"coef cosine vs ASV-only head {w['cosine_vs_asv_only']:.3f}, Hindi-val logit correlation {w['z_corr_vs_asv_only']:.3f}")
    poorly_fit = fit < B_FIT_MIN
    clustered = nh >= B_NEAR_ABS and nh >= B_NEAR_RATIO * max(na, 1e-9)
    if poorly_fit or clustered:
        return "SUPPORTED", detail
    return "NOT SUPPORTED", detail


# =============================================================================
# Hypothesis C (+ oversampling) grid, with LOSO selection
# =============================================================================

def build_train(asv: Rows, hi: Rows, idx: np.ndarray, factor: int) -> Tuple[np.ndarray, np.ndarray]:
    X = np.concatenate([asv.X] + [hi.X[idx]] * factor)
    y = np.concatenate([asv.y] + [hi.y[idx]] * factor)
    return X, y


def run_grid(fam: str, b: str, saved) -> Dict[Tuple[float, int], Dict[str, Any]]:
    asv, hi, ev = fam_rows(fam, "asv_train", b), fam_rows(fam, "hi_train", b), fam_rows(fam, "hi_eval", b)
    spk, _ = hindi_meta(hi)
    all_idx = np.arange(len(hi.y))
    other = "wav2vec2" if b == "wavlm" else "wavlm"
    partner = saved[(fam, other)]["p"]
    out: Dict[Tuple[float, int], Dict[str, Any]] = {}
    for C in C_GRID:
        for os_ in OVERSAMPLE_GRID:
            X, y = build_train(asv, hi, all_idx, os_)
            model, scaler = fit_lr(X, y, C)
            yc, pc, _ = clip_level(ev.y, predict(model, scaler, ev.X), ev.clip)
            cell: Dict[str, Any] = {"soumya": metrics(yc, pc), "p": pc}
            if b == "wavlm":   # ensemble with the current, saved wav2vec2 head
                cell["ens"] = ens_summary(partner, pc, yc, 0.5)
            loso = []
            for held in ("byaquta", "mahato"):
                tr, va = np.where(spk != held)[0], np.where(spk == held)[0]
                Xl, yl = build_train(asv, hi, tr, os_)
                m2, s2 = fit_lr(Xl, yl, C)
                yc2, pc2, _ = clip_level(hi.y[va], predict(m2, s2, hi.X[va]), hi.clip[va])
                loso.append(metrics(yc2, pc2))
            lm = mean_metrics(loso)
            lm["bal"] = (lm["real_acc"] + lm["synth_recall"]) / 2.0
            cell["loso"] = lm
            out[(C, os_)] = cell
        logger.info("%s/%s: C=%g done", fam, b, C)
    return out


def select_cell(grid: Dict[Tuple[float, int], Dict[str, Any]], by: str = "bal") -> Tuple[float, int]:
    """LOSO-selected cell. ``by='bal'`` (the PRE-DECLARED rule): best LOSO balanced accuracy, then AUC.
    ``by='auc'`` (sensitivity check): best LOSO AUC, then balanced accuracy. Ties prefer the current
    recipe, then less change. Balanced accuracy is threshold-dependent and oversampling shifts
    calibration, so the two selectors can disagree; the report shows both.
    """
    def key(cell):
        C, os_ = cell
        g = grid[cell]["loso"]
        primary = (-round(g["bal"], 6), -round(g["auc"], 6)) if by == "bal" else (-round(g["auc"], 6), -round(g["bal"], 6))
        return (*primary, cell != CURRENT_CELL, os_, abs(math.log10(C)))
    return min(grid, key=key)


def verdict_c(grid: Dict[Tuple[float, int], Dict[str, Any]]) -> Tuple[str, str]:
    cur = grid[CURRENT_CELL]
    lows = [(C, grid[(C, 1)]) for C in C_GRID if C < 1.0
            if grid[(C, 1)]["soumya"]["synth_recall"] >= cur["soumya"]["synth_recall"] - SYNTH_RECALL_TOLERANCE]
    best = max(lows, key=lambda t: t[1]["soumya"]["real_acc"], default=None)
    if best is None:
        return "NOT SUPPORTED", "no lower C keeps synthetic recall within 3 pp of the current head"
    C, cell = best
    gain = cell["soumya"]["real_acc"] - cur["soumya"]["real_acc"]
    loso_ok = cell["loso"]["bal"] >= cur["loso"]["bal"]
    detail = (f"best lower C={C:g}: soumya real-clip acc {cell['soumya']['real_acc']:.2f} vs {cur['soumya']['real_acc']:.2f} at C=1 "
              f"({gain:+.2f}); LOSO balanced acc {cell['loso']['bal']:.3f} vs {cur['loso']['bal']:.3f}")
    if gain >= C_IMPROVE and loso_ok:
        return "SUPPORTED", detail
    if gain >= C_IMPROVE:
        return "PARTIAL", detail + " (soumya improves but LOSO does not agree)"
    return "NOT SUPPORTED", detail


def verdict_b_remedy(grid: Dict[Tuple[float, int], Dict[str, Any]]) -> Tuple[str, str]:
    """EXPLORATORY. Directly tests 'Hindi rows are drowned out': upweight them (C=1) and see if WavLM alone recovers."""
    cur = grid[CURRENT_CELL]["soumya"]
    best = max(((os_, grid[(1.0, os_)]["soumya"]) for os_ in OVERSAMPLE_GRID if os_ > 1), key=lambda t: t[1]["auc"])
    os_, cell = best
    d_auc, d_real = cell["auc"] - cur["auc"], cell["real_acc"] - cur["real_acc"]
    detail = (f"WavLM alone on soumya, C=1: {os_}x oversampling gives AUC {cell['auc']:.3f} vs {cur['auc']:.3f} ({d_auc:+.3f}), "
              f"real-clip acc {cell['real_acc']:.2f} vs {cur['real_acc']:.2f} ({d_real:+.2f})")
    return ("SUPPORTED" if d_auc >= B_REMEDY_AUC and d_real >= C_IMPROVE else "NOT SUPPORTED"), detail


def verdict_c_x_b(grid: Dict[Tuple[float, int], Dict[str, Any]]) -> Tuple[str, str]:
    """EXPLORATORY. Low C combined with oversampling, at the LOSO-selected cells (never picked by soumya)."""
    cur = grid[CURRENT_CELL]
    parts, ok = [], False
    for by in ("bal", "auc"):
        cell = select_cell(grid, by)
        g = grid[cell]
        gain = g["soumya"]["auc"] - cur["soumya"]["auc"]
        parts.append(f"LOSO-{by} cell C={cell[0]:g}, {cell[1]}x: soumya AUC {g['soumya']['auc']:.3f} ({gain:+.3f} vs current), "
                     f"LOSO AUC {g['loso']['auc']:.3f} vs {cur['loso']['auc']:.3f}")
        ok |= cell != CURRENT_CELL and gain >= 0.15 and g["loso"]["auc"] >= cur["loso"]["auc"]
    return ("SUPPORTED" if ok else "NOT SUPPORTED"), "; ".join(parts)


D_FLOOR = 0.15              # D: if every speaker is at/below this, the probe cannot discriminate


def verdict_d(B: Dict, grid: Dict[Tuple[float, int], Dict[str, Any]], fam: str) -> Tuple[str, str]:
    """EXPLORATORY. Is soumya a domain outlier? ASV-only WavLM head (never saw Hindi) per-speaker real-clip accuracy.

    INCONCLUSIVE when every speaker is at floor: an ASV-only head that calls all Hindi real speech synthetic says
    nothing about who is an outlier. The current combined head's LOSO-vs-soumya contrast is reported alongside as
    the remaining (unexplained) hint of soumya-specificity.
    """
    s = B[(fam, "wavlm")]["asv_only_by_speaker"]
    accs = {k: s[k]["real_acc"] for k in ("byaquta", "mahato", "soumya")}
    cur = grid[CURRENT_CELL]
    detail = ("ASV-only WavLM head, real-clip accuracy: " +
              ", ".join(f"{k} {s[k]['real_acc']:.2f} (median P {s[k]['median_p_real']:.2f})" for k in accs) +
              f"; combined WavLM head (current recipe) real-clip accuracy: {cur['loso']['real_acc']:.2f} on the held-out TRAINING speaker (LOSO) "
              f"vs {cur['soumya']['real_acc']:.2f} on soumya")
    if max(accs.values()) < D_FLOOR:
        return "INCONCLUSIVE", detail + " (floor effect: the ASV-only head calls every speaker's real Hindi speech synthetic)"
    outlier = min(accs["byaquta"], accs["mahato"]) - accs["soumya"] >= D_OUTLIER_GAP
    return ("SUPPORTED" if outlier else "NOT SUPPORTED"), detail


# =============================================================================
# Reweight probe and recommendation
# =============================================================================

def reweight_probe(saved) -> Dict[str, Dict[float, Dict[str, Any]]]:
    out = {}
    for fam in FAMILIES:
        a, w = saved[(fam, "wav2vec2")], saved[(fam, "wavlm")]
        out[fam] = {wt: ens_summary(a["p"], w["p"], a["y"], wt) for wt in WEIGHT_GRID}
    return out


def meets(e: Dict[str, Any], floor: float) -> bool:
    return e["real_confident_correct"] >= GATE_CONFIDENT and e["synth_recall"] >= floor


def recommend(fam: str, R: Dict[str, Any], floor: float) -> Dict[str, Any]:
    """Retrain vs reweight, decided by whether WavLM ITSELF can do the job.

    The ensemble-level criterion (confident-and-correct count) can be met by merely making a wrong WavLM less
    extreme, so a retrain only counts as a fix if WavLM ALONE separates soumya's classes at that cell
    (AUC >= ADEQUATE_AUC): that is the evidence the backbone is adequate and the head was the defect.
    Cells come from the two LOSO selectors, never from soumya.
    """
    cur = R["issue3"][fam]["ens"]
    if meets(cur, floor):
        return {"action": "NO-FIX-NEEDED", "why": (
            f"current heads already give {cur['real_confident_correct']}/{cur['n_real']} confident-and-correct real clips "
            f"(>= {GATE_CONFIDENT}) with synthetic recall {cur['synth_recall']:.2f} (>= {floor:.2f})")}

    grid = R["grid"][(fam, "wavlm")]
    cur_auc = grid[CURRENT_CELL]["soumya"]["auc"]
    cells: Dict[str, Dict[str, Any]] = {}
    for by in ("bal", "auc"):
        c = select_cell(grid, by)
        g = grid[c]
        cells[by] = {
            "C": c[0], "oversample": c[1], "wavlm_auc": g["soumya"]["auc"], "wavlm_real_acc": g["soumya"]["real_acc"],
            "loso_auc": g["loso"]["auc"], "ens": g["ens"],
            "meets": c != CURRENT_CELL and meets(g["ens"], floor) and g["soumya"]["auc"] >= ADEQUATE_AUC,
        }
    passing = [by for by in ("bal", "auc") if cells[by]["meets"]]
    # Most confident-and-correct clips; ties go to the cell whose WavLM ALONE separates soumya best.
    best_retrain = max(cells.values(), key=lambda v: (v["ens"]["real_confident_correct"], v["wavlm_auc"]))

    rw = R["reweight"][fam]
    rw_ok = [wt for wt in WEIGHT_GRID if 0.5 < wt < 1.0 and meets(rw[wt], floor)]
    rw_best = max((rw[wt]["real_confident_correct"] for wt in WEIGHT_GRID if 0.5 < wt < 1.0), default=0)
    w2v_alone = R["issue3"][fam]["wav2vec2"]["real_confident_correct"]
    detail = {"loso_selected_cells": cells, "reweight_passing_weights": rw_ok, "reweight_best_count": rw_best,
              "wav2vec2_alone_confident_correct": w2v_alone, "current_wavlm_soumya_auc": cur_auc}

    if passing:
        pick = passing[0]   # pre-declared selector first
        c = cells[pick]
        action = "FIX-BY-RETRAIN"
        why = (f"retraining WavLM at C={c['C']:g} with {c['oversample']}x Hindi oversampling (LOSO-{pick} cell) lifts WavLM ALONE on soumya "
               f"from AUC {cur_auc:.3f} to {c['wavlm_auc']:.3f} and gives {c['ens']['real_confident_correct']}/25 confident+correct with "
               f"synthetic recall {pct(c['ens']['synth_recall'])}. The backbone is adequate; the head was the defect"
               + (f". A reweight to weight_a in {rw_ok} would also clear the ensemble criteria but only by muting a WavLM that is wrong, "
                  "and it is validated on the same 50 symptom clips" if rw_ok else ""))
        detail["recommended_cell"] = {"C": c["C"], "oversample": c["oversample"], "selector": f"LOSO-{pick}"}
    elif rw_ok:
        action = "FIX-BY-REWEIGHT"
        why = (f"no LOSO-selected retrain cell both clears the ensemble criteria and makes WavLM alone adequate (AUC >= {ADEQUATE_AUC}); "
               f"weight_a in {rw_ok} clears them")
    else:
        improves = any(v["wavlm_auc"] >= max(cur_auc + 0.15, ADEQUATE_AUC) for v in cells.values())
        if improves and best_retrain["ens"]["real_confident_correct"] > cur["real_confident_correct"] \
                and best_retrain["ens"]["real_confident_correct"] >= rw_best:
            action = "FIX-BY-RETRAIN (PARTIAL: criteria not reached)"
            why = (f"retraining fixes WavLM itself (alone on soumya: AUC {cur_auc:.3f} -> up to "
                   f"{max(v['wavlm_auc'] for v in cells.values()):.3f}) and is the best available remedy "
                   f"({best_retrain['ens']['real_confident_correct']}/25 vs {cur['real_confident_correct']}/25 now; no weight_a in (0.5, 1.0) "
                   f"exceeds {rw_best}/25), but NO remedy reaches {GATE_CONFIDENT}/25 with synthetic recall >= {pct(floor)}. "
                   f"The wav2vec2 head alone gets only {w2v_alone}/25, so the count is limited by the wav2vec2 head / aggregated-score "
                   "calibration, not by WavLM: it cannot be expected to reach the target by fixing WavLM alone")
            detail["recommended_cell"] = {"C": best_retrain["C"], "oversample": best_retrain["oversample"], "selector": "best of the two LOSO cells"}
        elif rw_best > cur["real_confident_correct"]:
            action = "FIX-BY-REWEIGHT (PARTIAL: criteria not reached)"
            why = f"the best weight_a in (0.5, 1.0) reaches {rw_best}/25; no retrain cell fixes WavLM alone"
        else:
            action = "NEITHER-PROVEN"
            why = f"no retrain cell or weight_a in (0.5, 1.0) improves on the current {cur['real_confident_correct']}/25"
    return {"action": action, "why": why, **detail}


# =============================================================================
# Report
# =============================================================================

def pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.0f}%"


def ci(k: float, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k / n * 100:.0f}% [{lo * 100:.0f}-{hi * 100:.0f}]"


def md_table(header: List[str], rows: List[List[str]]) -> List[str]:
    return ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + [":---:"] * (len(header) - 1)) + "|"] + \
           ["| " + " | ".join(r) + " |" for r in rows]


def build_report(R: Dict[str, Any], floor: float, repro: Optional[Dict[str, Any]], unchanged: bool) -> str:
    fams = list(FAMILIES)
    L: List[str] = [
        "# WavLM Head Diagnosis (Phase F3.1) — diagnosis only, no model files changed",
        "",
        f"**Generated at:** {datetime.now(timezone.utc).isoformat()}",
        "**Script:** `scripts/fix_diagnose_wavlm.py`",
        f"**Model files unchanged (snapshot of `models/classifiers/` before vs after):** {'yes' if unchanged else '**NO — INVESTIGATE**'}",
        "",
        "Every head trained here is a throwaway kept in memory. All numbers are CLIP level (chunked heads are "
        "mean-aggregated). Data: 2 training speakers (byaquta, mahato; 25 real + 25 synthetic clips each) and 1 "
        "held-out speaker (soumya; 25 real + 25 synthetic clips). **With 50 held-out clips one clip is 2 pp; treat "
        "differences of a few clips as noise.** Wilson 95% intervals are shown where a single proportion carries a verdict.",
        "",
        "## 1. Issue 3 metric across all model families",
        "",
        "soumya's 25 REAL clips. *Confident* = |P(synth) - 0.5| > 0.15, the original direction-agnostic definition "
        "(it also counts confidently WRONG clips). *Confident + correct* additionally requires P < 0.5 and is the number "
        "that matters. *Disagree* = clips where wav2vec2 and WavLM fall on opposite sides of 0.5.",
        "",
    ]
    rows = []
    for fam in fams:
        i3 = R["issue3"][fam]
        for name, key in (("wav2vec2 alone", "wav2vec2"), ("WavLM alone", "wavlm"), ("**Ensemble 0.5/0.5**", "ens")):
            e = i3[key]
            rows.append([FAMILIES[fam]["label"] if key == "wav2vec2" else "", name, f"{e['real_correct']}/{e['n_real']}",
                         f"{e['median_real_p']:.3f}", f"{e['real_confident']}/{e['n_real']}",
                         f"**{e['real_confident_correct']}/{e['n_real']}**" if key == "ens" else f"{e['real_confident_correct']}/{e['n_real']}",
                         str(e["disagree"]) if key == "ens" else "", pct(e["synth_recall"])])
    L += md_table(["Family", "Model", "Real correct (P<0.5)", "Median P(synth) on reals", "Confident (>0.15)",
                   "Confident + correct", "Backbones disagree", "Synth recall (25 clips)"], rows)
    L += ["", f"F0 baseline: 7/25 confident. Gate F3 target: >= {GATE_CONFIDENT}/25 confident-and-correct, synthetic recall >= {pct(floor)} "
          f"(F0's Hindi synthetic recall minus 3 pp)."]
    if repro is not None:
        state = "REPRODUCED" if repro["ok"] else "NOT REPRODUCED — the OLD row cannot be trusted"
        L.append(f"OLD row vs `fix_baseline.json` item 4: **{state}** (max per-clip |diff| {repro['max_diff']:.2e}; "
                 f"{repro.get('count')}/25 vs baseline {repro.get('baseline_count')}/25).")
    L += ["", "## 2. Hypothesis A — speaker shortcut", "",
          "**A1. Speaker probe** (byaquta vs mahato; LR on Hindi-train embeddings, 5-fold grouped CV x 5 seeds; sample level). "
          "If WavLM is markedly more speaker-dominated its accuracy / AUC should exceed wav2vec2's; near-ceiling values for both "
          "cannot discriminate, in which case log-loss (lower = more decodable) is the finer read.", ""]
    rows = []
    for fam in fams:
        for b in BACKBONES:
            s = R["A"][(fam, b)]["speaker"]
            rows.append([FAMILIES[fam]["label"] if b == "wav2vec2" else "", b, str(R["A"][(fam, b)]["n_train_rows"]),
                         f"{s['chance']:.2f}", f"{s['acc']:.3f} +/- {s['acc_sd']:.3f}", f"{s['auc']:.3f}", f"{s['logloss']:.4f}"])
    L += md_table(["Family", "Backbone", "Rows", "Chance", "Speaker accuracy", "AUC", "Log-loss"], rows)
    L += ["", "**A2. Label head trained on Hindi-train only**, scored on (i) held-out clips of the SAME speakers (grouped by utterance), "
          "(ii) soumya, and (iii) supplementary leave-one-training-speaker-out (train on one of byaquta/mahato, test on the other). "
          "The gap (i) - (ii) is the shortcut signature if it is large for WavLM but not for wav2vec2.", ""]
    rows = []
    for fam in fams:
        for b in BACKBONES:
            a = R["A"][(fam, b)]
            n = a["soumya"]["n"]
            rows.append([FAMILIES[fam]["label"] if b == "wav2vec2" else "", b,
                         f"{a['same']['acc'] * 100:.0f}% / {a['same']['auc']:.3f}",
                         f"{ci(a['soumya']['k_correct'], n)} / {a['soumya']['auc']:.3f}",
                         f"{a['loso']['acc'] * 100:.0f}% / {a['loso']['auc']:.3f}",
                         f"{(a['same']['acc'] - a['soumya']['acc']) * 100:+.0f} pp",
                         f"{a['soumya']['real_acc'] * 100:.0f}% / {a['soumya']['synth_recall'] * 100:.0f}%"])
    L += md_table(["Family", "Backbone", "(i) same-speaker held-out: acc / AUC", "(ii) soumya: acc [95% CI] / AUC",
                   "(iii) LOSO: acc / AUC", "Gap (i)-(ii)", "soumya real acc / synth recall"], rows)
    L += ["", "**Verdict A** (SUPPORTED needs: WavLM gap exceeds wav2vec2's by >= "
          f"{GAP_DIFF_LARGE * 100:.0f} pp, WavLM worse on soumya, and speaker accuracy higher by >= {SPK_DIFF_MARKED * 100:.0f} pp):", ""]
    for fam in fams:
        v, d = R["verdicts"]["A"][fam]
        L.append(f"- **{fam}: {v}** — {d}")

    L += ["", "## 3. Hypothesis B — Hindi rows drowned out by ASVspoof", "",
          "Saved heads. `logit` = `decision_function`. *Near boundary* = share of rows with |logit| < 1 (class-macro-averaged so the "
          "ASVspoof class imbalance does not decide it). The ASVspoof-ONLY reference head is trained on ASVspoof train alone: "
          "if the combined head is effectively English-only its coefficients and Hindi-val logits should match the reference.", ""]
    rows = []
    for fam in fams:
        for b in BACKBONES:
            B = R["B"][(fam, b)]
            s = B["sets"]
            rows.append([FAMILIES[fam]["label"] if b == "wav2vec2" else "", b, f"{B['coef_norm']:.2f} / {B['ref_coef_norm']:.2f}",
                         f"{B['cosine_vs_asv_only']:.3f}", f"{B['z_corr_vs_asv_only']:.3f}",
                         f"{s['hi_train']['acc_row']:.3f}",
                         f"{s['hi_val']['median_abs']:.2f} / {s['hi_val']['near1']:.2f}",
                         f"{s['asv_val']['median_abs']:.2f} / {s['asv_val']['near1']:.2f}"])
    L += md_table(["Family", "Backbone", "Coef norm: combined / ASV-only", "Coef cosine vs ASV-only", "Hindi-val logit corr vs ASV-only",
                   "Fit acc on Hindi TRAIN rows", "Hindi-val: median |logit| / near boundary",
                   "ASV-val: median |logit| / near boundary"], rows)
    L += ["", "Adding the Hindi rows changes soumya performance of the head as follows (clip level, acc / AUC, ASV-only reference -> saved head):", ""]
    rows = []
    for fam in fams:
        for b in BACKBONES:
            ref = R["B"][(fam, b)]["asv_only_soumya"]
            sv = R["saved"][(fam, b)]
            m = metrics(sv["y"], sv["p"])
            rows.append([FAMILIES[fam]["label"] if b == "wav2vec2" else "", b,
                         f"{ref['acc'] * 100:.0f}% / {ref['auc']:.3f}", f"{m['acc'] * 100:.0f}% / {m['auc']:.3f}"])
    L += md_table(["Family", "Backbone", "ASV-only head on soumya", "Saved combined head on soumya"], rows)
    L += ["", f"**Verdict B — logit statistics (as specified)** (SUPPORTED if the WavLM head fits its own Hindi train rows below {B_FIT_MIN:.2f}, or Hindi-val rows sit near the "
          f"boundary: share >= {B_NEAR_ABS:.2f} and >= {B_NEAR_RATIO:.0f}x the ASV-val share):", ""]
    for fam in fams:
        v, d = R["verdicts"]["B"][fam]
        L.append(f"- **{fam}: {v}** — {d}")
    L += ["", f"**Verdict B — oversampling rescue (EXPLORATORY: added after the first run showed the logit statistics cannot see this effect).** "
          f"Directly tests 'the Hindi rows are drowned out': upweight them at C=1 and check WavLM ALONE on soumya "
          f"(SUPPORTED if AUC rises by >= {B_REMEDY_AUC:.2f} and real-clip accuracy by >= {C_IMPROVE * 100:.0f} pp):", ""]
    for fam in fams:
        v, d = R["verdicts"]["B_remedy"][fam]
        L.append(f"- **{fam}: {v}** — {d}")

    L += ["", "### Is soumya a domain outlier? (EXPLORATORY, alternative D)", "",
          "The ASVspoof-ONLY heads never saw any Hindi, so their per-speaker accuracy on REAL clips is a clean read of whether a speaker's recordings "
          "look like ASVspoof spoofs to that backbone. If soumya is far worse than byaquta/mahato, the failure is specific to soumya's recordings, "
          "not to 'unseen speakers' in general (which is what hypothesis A predicts). **If every speaker is at ~0%, the probe is at floor and cannot "
          "discriminate: an English-trained head simply calls Hindi real speech synthetic for everyone.**", ""]
    rows = []
    for fam in fams:
        for b in BACKBONES:
            s = R["B"][(fam, b)]["asv_only_by_speaker"]
            rows.append([FAMILIES[fam]["label"] if b == "wav2vec2" else "", b] +
                        [f"{s[k]['real_acc'] * 100:.0f}% / {s[k]['median_p_real']:.2f}" for k in ("byaquta", "mahato", "soumya")])
    L += md_table(["Family", "Backbone", "byaquta (train): real acc / median P(synth)", "mahato (train): real acc / median P(synth)",
                   "soumya (held-out): real acc / median P(synth)"], rows)
    L += ["", f"**Verdict D** (SUPPORTED if soumya's real-clip accuracy under the ASV-only WavLM head is >= {D_OUTLIER_GAP * 100:.0f} pp below the worse training speaker):", ""]
    for fam in fams:
        v, d = R["verdicts"]["D"][fam]
        L.append(f"- **{fam}: {v}** — {d}")

    L += ["", "## 4. Hypothesis C — regularisation (and Hindi oversampling)", "",
          "Throwaway heads: ASVspoof train + Hindi train (Hindi rows duplicated `os` times), scaler refit, class-balanced LR at the stated C. "
          "**soumya** columns are the held-out-speaker result. **LOSO** columns train on ASVspoof + ONE Hindi speaker and validate on the other "
          "(mean of both directions): they are the selection criterion, because picking a cell by soumya and reporting soumya is selection on the test set. "
          "`Ens` columns pair the WavLM variant with the CURRENT saved wav2vec2 head (0.5/0.5); `conf+ok` = confident-and-correct real clips of 25.", ""]
    recipe_diff = R["repro"]
    L.append("Recipe check — throwaway head at C=1, no oversampling vs the saved head's soumya probabilities (max |diff|): " +
             ", ".join(f"{fam}/{b} {recipe_diff[(fam, b)]:.1e}" for fam in fams for b in BACKBONES) + ".")
    L.append("")

    def grid_rows(fam: str, b: str, cells: List[Tuple[float, int]]) -> List[List[str]]:
        g = R["grid"][(fam, b)]
        sel = R["selected"][fam] if b == "wavlm" else {}
        out = []
        for c in cells:
            cell = g[c]
            tag = (" (current)" if c == CURRENT_CELL else "")
            tag += " **<- LOSO-balanced pick**" if sel.get("bal") == c else ""
            tag += " **<- LOSO-AUC pick**" if sel.get("auc") == c else ""
            row = [f"C={c[0]:g}, os={c[1]}x{tag}", f"{cell['soumya']['real_acc'] * 100:.0f}%", f"{cell['soumya']['synth_recall'] * 100:.0f}%",
                   f"{cell['soumya']['auc']:.3f}"]
            if b == "wavlm":
                e = cell["ens"]
                row += [f"{e['real_confident_correct']}/25", f"{e['synth_recall'] * 100:.0f}%"]
            row += [f"{cell['loso']['real_acc'] * 100:.0f}%", f"{cell['loso']['synth_recall'] * 100:.0f}%", f"{cell['loso']['auc']:.3f}", f"{cell['loso']['bal']:.3f}"]
            out.append(row)
        return out

    def grid_header(b: str) -> List[str]:
        h = ["Setting", "soumya real acc", "soumya synth recall", "soumya AUC"]
        if b == "wavlm":
            h += ["Ens conf+ok", "Ens synth recall"]
        return h + ["LOSO real acc", "LOSO synth recall", "LOSO AUC", "LOSO balanced"]

    for fam in fams:
        L += [f"### {FAMILIES[fam]['label']}", "", "**WavLM — C sweep (no oversampling):**", ""]
        L += md_table(grid_header("wavlm"), grid_rows(fam, "wavlm", [(C, 1) for C in C_GRID]))
        L += ["", "**WavLM — Hindi oversampling sweep (C=1):**", ""]
        L += md_table(grid_header("wavlm"), grid_rows(fam, "wavlm", [(1.0, o) for o in OVERSAMPLE_GRID]))
        L += ["", "<details><summary>WavLM — full C x oversampling grid</summary>", ""]
        L += md_table(grid_header("wavlm"), grid_rows(fam, "wavlm", [(C, o) for C in C_GRID for o in OVERSAMPLE_GRID]))
        L += ["", "</details>", "", "**wav2vec2 control — C sweep (no oversampling):**", ""]
        L += md_table(grid_header("wav2vec2"), grid_rows(fam, "wav2vec2", [(C, 1) for C in C_GRID]))
        L.append("")
    L += ["**Verdict C — C alone (as specified)** (SUPPORTED if some lower C, WITHOUT oversampling, raises soumya real-clip accuracy by >= "
          f"{C_IMPROVE * 100:.0f} pp over C=1 without costing more than 3 pp of synthetic recall AND LOSO agrees):", ""]
    for fam in fams:
        v, d = R["verdicts"]["C"][fam]
        L.append(f"- **{fam}: {v}** — {d}")
    L += ["", "**Verdict C x oversampling (EXPLORATORY, added after the first run):** low C combined with Hindi oversampling, judged at the "
          "two LOSO-selected cells (never chosen by soumya). SUPPORTED if a selected cell (not the current recipe) lifts WavLM-alone soumya AUC by >= 0.15 "
          "and LOSO AUC does not drop:", ""]
    for fam in fams:
        v, d = R["verdicts"]["C_x_B"][fam]
        L.append(f"- **{fam}: {v}** — {d}")
    L += ["", "The two selectors can disagree because balanced accuracy is threshold-dependent and oversampling shifts calibration; LOSO AUC is threshold-free. "
          "The balanced-accuracy rule was declared before running and is treated as primary; the AUC pick is a sensitivity check."]

    L += ["", "## 5. Reweight probe (ensemble weight_a = wav2vec2 share), saved heads", "",
          "Indicative only: these are the same 50 clips that surfaced the issue, so a weight chosen here would be overfit to its symptom. "
          "F3.2b sweeps on the FULL held-out sets. `weight_a = 1.0` drops WavLM entirely, so it is not a fix for WavLM.", ""]
    rows = []
    for fam in fams:
        for wt in WEIGHT_GRID:
            e = R["reweight"][fam][wt]
            rows.append([FAMILIES[fam]["label"] if wt == WEIGHT_GRID[0] else "", f"{wt:.1f}", f"{e['real_confident_correct']}/25",
                         f"{e['real_correct']}/25", pct(e["synth_recall"]), f"{e['auc']:.3f}", "yes" if meets(e, floor) else "no"])
    L += md_table(["Family", "weight_a", "Confident + correct", "Real correct", "Synth recall", "AUC", "Meets criteria"], rows)

    L += ["", "## 6. Conclusion and recommendation", "",
          "### Which hypothesis does the evidence support?", ""]
    names = {"A": "A. Speaker shortcut", "B": "B. Hindi rows drowned out (logit statistics, as specified)",
             "B_remedy": "B. Hindi rows drowned out (oversampling rescue, exploratory)",
             "C": "C. Regularisation: C alone (as specified)", "C_x_B": "C x oversampling (exploratory)",
             "D": "D. soumya is a domain outlier (exploratory)"}
    rows = [[names[h], *[R["verdicts"][h][fam][0] for fam in fams]] for h in ("A", "B", "B_remedy", "C", "C_x_B", "D")]
    L += md_table(["Hypothesis", *[f"{f}" for f in fams]], rows)
    L += ["", "### Recommendation", "",
          f"Criteria for a fix (mirrors Gate F3): >= {GATE_CONFIDENT}/25 confident-and-correct real clips AND ensemble synthetic recall >= {pct(floor)}. "
          "Retrain hyperparameters come from LOSO validation (two selectors), never from soumya. Because the ensemble count can be met by merely muting a wrong WavLM, "
          f"a retrain only counts as a fix if WavLM ALONE reaches soumya AUC >= {ADEQUATE_AUC} at that cell (evidence the backbone is adequate and the head was the defect). "
          "If a fix qualifies, retrain is preferred over reweight; reweight is recommended only when no retrain cell qualifies.", ""]
    for fam in DECISION_FAMILIES:
        rec = R["rec"][fam]
        L.append(f"**{fam} family: {rec['action']}** — {rec['why']}.")
        if rec["action"] != "NO-FIX-NEEDED":
            for by, c in rec["loso_selected_cells"].items():
                e = c["ens"]
                L.append(f"  - LOSO-{by} cell: C={c['C']:g}, Hindi oversample {c['oversample']}x -> WavLM alone on soumya AUC {c['wavlm_auc']:.3f} "
                         f"(current {rec['current_wavlm_soumya_auc']:.3f}); ensemble {e['real_confident_correct']}/25 confident+correct, "
                         f"synthetic recall {pct(e['synth_recall'])}; qualifies: {'yes' if c['meets'] else 'no'}.")
            L.append(f"  - Reweight: weights in (0.5, 1.0) meeting the criteria: {rec['reweight_passing_weights'] or 'none'} "
                     f"(best count in that range {rec['reweight_best_count']}/25; wav2vec2 alone {rec['wav2vec2_alone_confident_correct']}/25). "
                     "Validated only on the 50 symptom clips.")
            if "recommended_cell" in rec:
                rc = rec["recommended_cell"]
                L.append(f"  - **Hyperparameter change to carry into F3.2a:** C = {rc['C']:g} (from 1), Hindi oversampling {rc['oversample']}x (from 1x) [{rc['selector']}].")
        L.append("")
    actions = {R["rec"][f]["action"] for f in DECISION_FAMILIES}
    L.append(f"**Overall:** {actions.pop() if len(actions) == 1 else 'families differ - see the per-family lines above'}.")
    L += ["", "### Caveats", "",
          "- soumya is one speaker with 25 real clips; every accuracy above has a wide interval (see the Wilson intervals). "
          "The verdict thresholds are heuristics stated inline, not significance tests.",
          "- Two training speakers means the LOSO validation trains on ONE Hindi speaker; it is a weak but independent selector.",
          "- A chunked head's clip score here is the MEAN of its chunk probabilities; a different aggregation would change the chunked rows.",
          ""]
    return "\n".join(L)


def print_summary(R: Dict[str, Any], out: Path, floor: float) -> None:
    bar = "=" * 100
    print("\n" + bar)
    print(" WAVLM DIAGNOSIS (F3.1) - Issue 3 metric, soumya's 25 real clips")
    print(bar)
    print(f" {'Family':<10}{'w2v ok':<9}{'wavlm ok':<10}{'ens conf':<10}{'ens conf+ok':<13}{'disagree':<10}{'synth recall':<13}")
    for fam in FAMILIES:
        i = R["issue3"][fam]
        e = i["ens"]
        print(f" {fam:<10}{i['wav2vec2']['real_correct']:<9}{i['wavlm']['real_correct']:<10}{e['real_confident']:<10}"
              f"{e['real_confident_correct']:<13}{e['disagree']:<10}{e['synth_recall']:<13.2f}")
    print(bar)
    for h in R["verdicts"]:
        print(f" Hypothesis {h:<9}: " + "   ".join(f"{fam}={R['verdicts'][h][fam][0]}" for fam in FAMILIES))
    print(bar)
    for fam in DECISION_FAMILIES:
        rec = R["rec"][fam]
        print(f" {fam}: {rec['action']}")
        print(f"    {rec['why']}")
        if "recommended_cell" in rec:
            rc = rec["recommended_cell"]
            print(f"    -> C={rc['C']:g}, oversample {rc['oversample']}x ({rc['selector']})")
    print(bar)
    print(f" Report written to: {out}\n")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output_md", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    before = snapshot_models()
    base = json.loads(BASELINE_JSON.read_text(encoding="utf-8")) if BASELINE_JSON.exists() else None
    floor = (base["item3_hindi"]["synthetic_recall"] if base else 1.0) - SYNTH_RECALL_TOLERANCE

    try:
        R: Dict[str, Any] = {"A": {}, "B": {}, "grid": {}, "selected": {}, "repro": {}}
        R["saved"] = score_saved_heads()
        R["issue3"] = issue3(R["saved"])
        repro = check_f0_reproduction(R["saved"], base)
        for fam in FAMILIES:
            for b in BACKBONES:
                logger.info("Hypotheses A/B: %s / %s", fam, b)
                R["A"][(fam, b)] = run_a(fam, b)
                R["B"][(fam, b)] = run_b(fam, b)
        for fam in FAMILIES:
            for b in ("wavlm", "wav2vec2"):
                logger.info("Grid: %s / %s", fam, b)
                g = run_grid(fam, b, R["saved"])
                R["grid"][(fam, b)] = g
                R["repro"][(fam, b)] = float(np.max(np.abs(g[CURRENT_CELL]["p"] - R["saved"][(fam, b)]["p"])))
            R["selected"][fam] = {by: select_cell(R["grid"][(fam, "wavlm")], by) for by in ("bal", "auc")}
        R["reweight"] = reweight_probe(R["saved"])
        R["verdicts"] = {
            "A": {f: verdict_a(R["A"], f) for f in FAMILIES},
            "B": {f: verdict_b(R["B"], f) for f in FAMILIES},
            "B_remedy": {f: verdict_b_remedy(R["grid"][(f, "wavlm")]) for f in FAMILIES},
            "C": {f: verdict_c(R["grid"][(f, "wavlm")]) for f in FAMILIES},
            "C_x_B": {f: verdict_c_x_b(R["grid"][(f, "wavlm")]) for f in FAMILIES},
            "D": {f: verdict_d(R["B"], R["grid"][(f, "wavlm")], f) for f in FAMILIES},
        }
        R["rec"] = {f: recommend(f, R, floor) for f in DECISION_FAMILIES}
    except Exception as exc:
        logger.error("Diagnosis failed: %s", exc)
        raise

    unchanged = snapshot_models() == before
    if not unchanged:
        logger.error("models/classifiers/ changed during a diagnosis-only script.")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_report(R, floor, repro, unchanged), encoding="utf-8")
    args.output_md.with_suffix(".json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "synthetic_recall_floor": floor,
        "verdicts": {h: {f: {"verdict": v[0], "detail": v[1]} for f, v in R["verdicts"][h].items()} for h in R["verdicts"]},
        "recommendation": R["rec"],
        "issue3": R["issue3"],
    }, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    print_summary(R, args.output_md, floor)
    if not unchanged:
        sys.exit(2)


if __name__ == "__main__":
    main()
