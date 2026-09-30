#!/usr/bin/env python3
"""
scripts/fix_retrain_wavlm_v2.py — Retrain the WavLM Hindi heads with the F3.1 remedy (Phase F3.2a).

Retrains BOTH WavLM heads with C=0.001 and the Hindi training rows oversampled 20x:

  matched whole-clip  -> models/classifiers/wavlm_hindi_matched_v2_logreg.joblib
  chunked             -> models/classifiers/wavlm_chunked_v2_logreg.joblib

The hyperparameters are F3.1's LOSO-selected cell (models/reports/fix_wavlm_diagnosis.md,
"Hypothesis C x oversampling"), not chosen by looking at soumya. The script asserts they agree with
``recommended_cell`` in ``fix_wavlm_diagnosis.json``.

wav2vec2's heads are NOT retrained: F3.1 never implicated them (hypotheses A/B/C evidence is about
WavLM; wav2vec2 alone already puts soumya's real clips on the right side). No ``wav2vec2_*_v2``
file is written. Every v1 file in ``models/classifiers/`` is snapshotted before and after and the
script aborts if any changed, so it is unambiguous which files are new.

Recipe (identical to the F3.1 throwaway heads, so the numbers there are reproducible here):
ASVspoof train rows + Hindi train rows repeated 20x, a StandardScaler fit on that stacked matrix,
class-balanced ``LogisticRegression(C=0.001, max_iter=1000)``. Saved with ``save_classifier``
(model + scaler + JSON sidecar) plus a ``_training.json`` provenance record.

Then, against the v2 heads (wav2vec2 = the unchanged v1 head), it reports:

1. Issue 3 metric (soumya's 25 REAL clips, |P - 0.5| > 0.15) before/after, for both families, with
   WavLM alone, wav2vec2 alone and the 0.5/0.5 ensemble. Both the original direction-agnostic count
   ("confident") and the stricter "confident AND correct" are printed.
2. The 5-clip backbone-disagreement check (soumya_control_21/22/25, soumya_neutral_06, soumya_scam_15).
3. The FULL F1.5 evaluation (fix_evaluate_matched.py: ASVspoof eval, Hindi original, Hindi matched)
   and the FULL F2.4 evaluation (evaluate_chunked.py: chunk- and clip-level) with v2 substituted, with
   REAL-class and SYNTHETIC-class recall side by side. A WavLM head that stopped saying "synthetic"
   would look perfect on the real-clip metric; the synthetic-recall columns exist to catch that.

Writes models/reports/fix_wavlm_v2_retrain.md. Touches nothing in config.py and no production path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Sibling scripts: reused so the "full evaluation" is the F1.5 / F2.4 code itself, not a re-implementation.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_chunked as ec  # noqa: E402
import fix_evaluate_matched as fem  # noqa: E402

from voxguard import config  # noqa: E402
from voxguard.classifier.head import _encode_labels, load_classifier, save_classifier  # noqa: E402
from voxguard.classifier.cross_eval import weighted_average_ensemble  # noqa: E402
from voxguard.embeddings.cache import load_cached_embeddings  # noqa: E402
from voxguard.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger("fix_retrain_wavlm_v2")

EMB = config.MODELS_DIR / "embeddings"
CLF = config.MODELS_DIR / "classifiers"
REPORT_PATH = config.MODELS_DIR / "reports" / "fix_wavlm_v2_retrain.md"
DIAGNOSIS_JSON = config.MODELS_DIR / "reports" / "fix_wavlm_diagnosis.json"
BASELINE_JSON = config.MODELS_DIR / "reports" / "fix_baseline.json"

# F3.1's LOSO-selected cell. The task fixes these; the JSON cross-check below guards against drift.
C_V2 = 0.001
OVERSAMPLE_V2 = 20

MARGIN = 0.15                   # Issue 3: |P - 0.5| > MARGIN
GATE_CONFIDENT = 18             # Gate F3
SYNTH_TOLERANCE = 0.03          # Gate F3: synthetic recall may drop at most 3 pp vs F0
THRESHOLD = 0.5
# F3.1's numbers for this cell (WavLM ALONE on soumya, throwaway head): the retrain must reproduce them.
EXPECTED_AUC = {"matched": 0.992, "chunked": 0.979}
AUC_REPRO_TOL = 0.005

FIVE_CLIPS = ["soumya_control_21", "soumya_control_22", "soumya_control_25",
              "soumya_neutral_06", "soumya_scam_15"]

FAMILIES: Dict[str, Dict[str, str]] = {
    "matched": {
        "label": "MATCHED whole-clip (F1)",
        "asv_train": "wavlm_train", "hi_train": "wavlm_hindi_train_matched",
        "hi_eval": "{b}_hindi_eval_matched",
        "v1": "{b}_hindi_matched_logreg", "v2": "wavlm_hindi_matched_v2_logreg",
    },
    "chunked": {
        "label": "CHUNKED (F2)",
        "asv_train": "wavlm_asvspoof2019_train_chunked", "hi_train": "wavlm_hindi_train_chunked",
        "hi_eval": "{b}_hindi_eval_chunked",
        "v1": "{b}_chunked_logreg", "v2": "wavlm_chunked_v2_logreg",
    },
}
BACKBONES = ("wav2vec2", "wavlm")


# =============================================================================
# Helpers
# =============================================================================

def snapshot_models() -> Dict[str, Tuple[int, int]]:
    return {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(CLF.glob("*")) if p.is_file()}


def _norm(p: Any) -> str:
    return str(p).replace("\\", "/")


def _pct(x: Optional[float], nd: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{nd}f}%"


def load_rows(stem: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, m = load_cached_embeddings(EMB / f"{stem}.npy")
    col = "parent_filepath" if "parent_filepath" in m.columns else "filepath"
    return X, _encode_labels(m["label"].values), np.array([_norm(c) for c in m[col]])


def clip_level(y: np.ndarray, p: np.ndarray, clip: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-aggregates row scores to one per clip (identity for whole-clip caches); sorted by clip key."""
    g = pd.DataFrame({"clip": clip, "y": y, "p": p}).groupby("clip", sort=True)
    if (g["y"].nunique() > 1).any():
        raise ValueError("A clip has rows with conflicting labels.")
    return g["y"].first().values, g["p"].mean().values, g["y"].first().index.values


def score(head: Tuple[Any, Any], X: np.ndarray) -> np.ndarray:
    model, scaler = head
    return model.predict_proba(scaler.transform(X))[:, 1]


# =============================================================================
# Retrain
# =============================================================================

def retrain(fam: str) -> Dict[str, Any]:
    spec = FAMILIES[fam]
    X_asv, y_asv, _ = load_rows(spec["asv_train"])
    X_hi, y_hi, _ = load_rows(spec["hi_train"])
    if X_asv.shape[1] != X_hi.shape[1]:
        raise ValueError(f"{fam}: dim mismatch ASVspoof={X_asv.shape[1]} vs Hindi={X_hi.shape[1]}")

    X = np.concatenate([X_asv] + [X_hi] * OVERSAMPLE_V2)
    y = np.concatenate([y_asv] + [y_hi] * OVERSAMPLE_V2)
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, C=C_V2).fit(scaler.transform(X), y)

    out_stem = CLF / spec["v2"]
    v1_stem = CLF / spec["v1"].format(b="wavlm")
    if out_stem.resolve() == v1_stem.resolve():
        raise RuntimeError(f"Refusing to overwrite v1 head {v1_stem}.joblib")
    save_classifier(model, out_stem, scaler)

    prov = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/fix_retrain_wavlm_v2.py",
        "backbone": "wavlm",
        "family": fam,
        "replaces_v1": v1_stem.name + ".joblib (left unchanged)",
        "hyperparameters": {"C": C_V2, "hindi_oversample": OVERSAMPLE_V2, "class_weight": "balanced",
                            "max_iter": 1000, "scaler": "StandardScaler fit on the oversampled stack"},
        "selection": "F3.1 LOSO-selected cell (models/reports/fix_wavlm_diagnosis.md, Hypothesis C x oversampling); "
                     "not selected on soumya",
        "train_sources": {"asvspoof_train": {"cache": spec["asv_train"] + ".npy", "rows": int(len(X_asv))},
                          "hindi_train": {"cache": spec["hi_train"] + ".npy", "rows": int(len(X_hi)),
                                          "rows_after_oversample": int(len(X_hi) * OVERSAMPLE_V2)}},
        "label_counts_after_oversample": {"real": int((y == 0).sum()), "synthetic": int((y == 1).sum())},
    }
    out_stem.with_name(out_stem.name + "_training.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    meta = json.loads(out_stem.with_suffix(".json").read_text())
    if meta["type"] != "logreg" or not meta.get("scaler_path"):
        raise RuntimeError(f"Unexpected sidecar for {out_stem}: {meta}")
    return {"saved": out_stem.with_suffix(".joblib").name, "n_rows": int(len(X)),
            "n_hindi_rows": int(len(X_hi)), "n_asv_rows": int(len(X_asv)),
            "coef_norm": float(np.linalg.norm(model.coef_))}


def check_against_diagnosis() -> List[str]:
    """The hyperparameters this script applies must be the ones F3.1 recommended."""
    notes: List[str] = []
    if not DIAGNOSIS_JSON.exists():
        return ["fix_wavlm_diagnosis.json not found - hyperparameters not cross-checked."]
    rec = json.loads(DIAGNOSIS_JSON.read_text(encoding="utf-8"))["recommendation"]
    for fam in FAMILIES:
        cell = rec[fam]["recommended_cell"]
        same = abs(cell["C"] - C_V2) < 1e-12 and int(cell["oversample"]) == OVERSAMPLE_V2
        notes.append(f"{fam}: diagnosis recommended C={cell['C']:g}, {cell['oversample']}x ({cell['selector']}) -> "
                     + ("matches this script" if same else "**DIFFERS from this script**"))
        if not same:
            raise ValueError(f"{fam}: script hyperparameters (C={C_V2:g}, {OVERSAMPLE_V2}x) differ from F3.1's "
                             f"recommended cell {cell}. Fix the constants or the task before retraining.")
    return notes


# =============================================================================
# Issue 3 + five-clip check
# =============================================================================

def issue3_row(p_w2v: np.ndarray, p_wavlm: np.ndarray, y: np.ndarray, w: float) -> Dict[str, Any]:
    """Issue 3 numbers for w*p_w2v + (1-w)*p_wavlm on soumya. w=1 -> wav2vec2 alone, w=0 -> WavLM alone."""
    p = w * p_w2v + (1.0 - w) * p_wavlm
    real, synth = y == 0, y == 1
    pr = p[real]
    conf = np.abs(pr - 0.5) > MARGIN
    return {
        "real_correct": int((pr < 0.5).sum()), "n_real": int(real.sum()),
        "confident": int(conf.sum()), "confident_correct": int((conf & (pr < 0.5)).sum()),
        "median_real_p": float(np.median(pr)),
        "disagree": int(((p_w2v[real] >= 0.5) != (p_wavlm[real] >= 0.5)).sum()),
        "synth_recall": float((p[synth] >= 0.5).mean()), "n_synth": int(synth.sum()),
        "auc": float(roc_auc_score(y, p)),
    }


def soumya_scores(fam: str) -> Dict[str, Any]:
    """Clip-level P(synthetic) on soumya from the v1 heads and the v2 WavLM head."""
    spec = FAMILIES[fam]
    out: Dict[str, Any] = {}
    ref_clip = None
    for name, stem in (("wav2vec2", spec["v1"].format(b="wav2vec2")),
                       ("wavlm_v1", spec["v1"].format(b="wavlm")),
                       ("wavlm_v2", spec["v2"])):
        b = "wav2vec2" if name == "wav2vec2" else "wavlm"
        X, y, clip = load_rows(spec["hi_eval"].format(b=b))
        yc, pc, cc = clip_level(y, score(load_classifier(CLF / stem), X), clip)
        if ref_clip is None:
            ref_clip, out["y"], out["clip"] = cc, yc, cc
        elif not (np.array_equal(cc, ref_clip) and np.array_equal(yc, out["y"])):
            raise ValueError(f"{fam}: backbone eval clips are not aligned.")
        out[name] = pc
    return out


def before_after(s: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    y, a = s["y"], s["wav2vec2"]
    return {
        "wav2vec2 alone (unchanged)": issue3_row(a, s["wavlm_v1"], y, 1.0),
        "WavLM alone - v1 (before)": issue3_row(a, s["wavlm_v1"], y, 0.0),
        "WavLM alone - v2 (after)": issue3_row(a, s["wavlm_v2"], y, 0.0),
        "Ensemble 0.5/0.5 - v1 (before)": issue3_row(a, s["wavlm_v1"], y, 0.5),
        "Ensemble 0.5/0.5 - v2 (after)": issue3_row(a, s["wavlm_v2"], y, 0.5),
    }


def five_clip_rows(s: Dict[str, Any]) -> List[Dict[str, Any]]:
    stems = {Path(c).stem: i for i, c in enumerate(s["clip"])}
    rows = []
    for name in FIVE_CLIPS:
        i = stems[name]
        if s["y"][i] != 0:
            raise ValueError(f"{name} is not a REAL clip.")
        a, w1, w2 = s["wav2vec2"][i], s["wavlm_v1"][i], s["wavlm_v2"][i]
        e1, e2 = 0.5 * (a + w1), 0.5 * (a + w2)
        rows.append({"clip": name, "w2v": float(a), "wavlm_v1": float(w1), "wavlm_v2": float(w2),
                     "ens_v1": float(e1), "ens_v2": float(e2),
                     "disagree_v1": bool((a >= 0.5) != (w1 >= 0.5)), "disagree_v2": bool((a >= 0.5) != (w2 >= 0.5)),
                     "conf_v1": bool(abs(e1 - 0.5) > MARGIN and e1 < 0.5),
                     "conf_v2": bool(abs(e2 - 0.5) > MARGIN and e2 < 0.5)})
    return rows


# =============================================================================
# F1.5 full evaluation (whole-clip), v2 substituted
# =============================================================================

def _clf(name: str) -> str:
    return str(CLF / f"{name}.joblib")


F15_VARIANTS: List[Dict[str, Any]] = [
    {"key": "c", "label": "(c) F0 production: w2v hindi_combined + WavLM hindi_combined (0.5/0.5)",
     "pair": (("wav2vec2", _clf("wav2vec2_hindi_combined_logreg")), ("wavlm", _clf("wavlm_hindi_combined_logreg")))},
    {"key": "d", "label": "(d) wav2vec2 hindi_matched (unchanged)",
     "single": ("wav2vec2", _clf("wav2vec2_hindi_matched_logreg"))},
    {"key": "e", "label": "(e) WavLM hindi_matched v1 alone (before)",
     "single": ("wavlm", _clf("wavlm_hindi_matched_logreg"))},
    {"key": "e2", "label": "(e2) WavLM hindi_matched v2 alone (after)",
     "single": ("wavlm", _clf("wavlm_hindi_matched_v2_logreg"))},
    {"key": "f", "label": "(f) matched ensemble v1: (d)+(e) (before)",
     "pair": (("wav2vec2", _clf("wav2vec2_hindi_matched_logreg")), ("wavlm", _clf("wavlm_hindi_matched_logreg")))},
    {"key": "g", "label": "(g) matched ensemble v2: (d)+(e2) (after)",
     "pair": (("wav2vec2", _clf("wav2vec2_hindi_matched_logreg")), ("wavlm", _clf("wavlm_hindi_matched_v2_logreg")))},
]


def recalls(m: Dict[str, Any]) -> Dict[str, float]:
    (tn, fp), (fn, tp) = m["confusion_matrix"]
    return {"real_recall": tn / (tn + fp) if (tn + fp) else float("nan"),
            "synth_recall": tp / (tp + fn) if (tp + fn) else float("nan"),
            "n_real": tn + fp, "n_synth": fn + tp, "pred_synth_frac": (fp + tp) / max(1, tn + fp + fn + tp)}


def run_f15() -> Dict[str, Dict[str, Dict[str, Any]]]:
    res: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for v in F15_VARIANTS:
        res[v["key"]] = {}
        for ts_key, _, dataset, split in fem.TEST_SETS:
            logger.info("F1.5 eval (%s) on %s", v["key"], ts_key)
            m = fem.evaluate_variant(v, dataset, split)
            res[v["key"]][ts_key] = {**m, **recalls(m)}
    return res


# =============================================================================
# F2.4 full evaluation (chunked), v2 substituted
# =============================================================================

def run_f24() -> Dict[str, Any]:
    """Chunk- and clip-level metrics for the v1 and v2 chunked WavLM heads (wav2vec2 = unchanged v1)."""
    w2v = load_classifier(CLF / "wav2vec2_chunked_logreg")
    head_sets = {"v1": {"wav2vec2": w2v, "wavlm": load_classifier(CLF / "wavlm_chunked_logreg")},
                 "v2": {"wav2vec2": w2v, "wavlm": load_classifier(CLF / "wavlm_chunked_v2_logreg")}}
    out: Dict[str, Any] = {}
    for key, title, dataset, split in ec.EVAL_SETS:
        out[key] = {"title": title}
        for ver, heads in head_sets.items():
            cs = ec.load_chunk_set(dataset, split, heads)
            man = cs["manifest"]
            entry: Dict[str, Any] = {"chunk": {}, "clip": {}}
            for model in (*ec.BACKBONES, "ensemble"):
                entry["chunk"][model] = ec.metrics_with_recall(man["label"].values, cs["scores"][model])
                for how in ("mean", "max"):
                    _, labels, agg = ec.aggregate_clips(man, cs["scores"][model], how)
                    entry["clip"][(model, how)] = ec.metrics_with_recall(labels, agg)
            entry["n_chunks"], entry["n_clips"] = len(man), int(man["parent_filepath"].nunique())
            out[key][ver] = entry
    return out


# =============================================================================
# Report
# =============================================================================

def _md(header: List[str], rows: List[List[str]]) -> List[str]:
    return (["| " + " | ".join(header) + " |", "|---|" + ":---:|" * (len(header) - 1)]
            + ["| " + " | ".join(r) + " |" for r in rows])


def _delta(new: float, old: float) -> str:
    return f"{(new - old) * 100:+.1f} pp"


def build_report(ctx: Dict[str, Any]) -> str:
    L: List[str] = [
        "# WavLM v2 Retrain (Phase F3.2a)",
        "",
        f"**Generated at:** {datetime.now(timezone.utc).isoformat()}",
        "**Script:** `scripts/fix_retrain_wavlm_v2.py`",
        f"**Remedy (F3.1 LOSO-selected cell):** C = {C_V2:g} (from 1), Hindi training rows oversampled {OVERSAMPLE_V2}x (from 1x); "
        "class-balanced logistic regression, scaler refit on the oversampled stack.",
        "",
        "## Files",
        "",
        "| Status | File |",
        "|---|---|",
    ]
    for r in ctx["train"].values():
        L.append(f"| **NEW** | `{r['saved']}` (+ `_scaler.joblib`, `.json`, `_training.json`) |")
    L += [
        "| unchanged | every v1 file in `models/classifiers/`, including `wavlm_hindi_matched_logreg`, `wavlm_chunked_logreg` |",
        "| unchanged | **wav2vec2's heads** (`wav2vec2_hindi_matched_logreg`, `wav2vec2_chunked_logreg`): no wav2vec2 v2 file was written |",
        "",
        f"Snapshot of `models/classifiers/` before vs after: existing files unchanged = **{ctx['snapshot_ok']}**; "
        f"new files = {', '.join('`' + n + '`' for n in ctx['new_files'])}.",
        "",
        "**Why wav2vec2 was left alone.** F3.1's hypothesis A/B/C evidence never implicated it: wav2vec2 ALONE already "
        "calls soumya's real clips real (matched 25/25, chunked 20/25), and its control C-sweep never improved on the "
        "current head without costing synthetic recall (e.g. chunked C=0.001: real 84%, synthetic recall 88% vs 100% at C=1). "
        "The asymmetry is WavLM-specific.",
        "",
        "**Hyperparameter cross-check against `fix_wavlm_diagnosis.json`:**",
        "",
    ] + [f"- {n}" for n in ctx["diag_notes"]] + [
        "",
        "**Reproduction of F3.1's throwaway heads** (WavLM ALONE on soumya, AUC): "
        + "; ".join(f"{fam} {ctx['i3'][fam]['WavLM alone - v2 (after)']['auc']:.3f} vs F3.1 {EXPECTED_AUC[fam]:.3f}"
                    for fam in FAMILIES) + ".",
        "",
        "> **Read the sample size first.** soumya is ONE held-out speaker with 25 real + 25 synthetic clips: one clip is 4 pp of "
        "real recall or synthetic recall. The same 25 real clips motivated F3, so they are a confirmation set, not an "
        "independent test.",
        "",
    ]

    # ---- 1. Issue 3
    L += ["## 1. Issue 3 metric, before / after (soumya's 25 REAL clips)", "",
          "*Confident* = |P(synth) - 0.5| > 0.15, the original direction-agnostic count (also counts confidently WRONG clips). "
          "*Conf+correct* also requires P < 0.5. *Disagree* = clips where wav2vec2 and WavLM fall on opposite sides of 0.5. "
          "Synth recall is on soumya's 25 SYNTHETIC clips (at 0.5).", ""]
    for fam, spec in FAMILIES.items():
        L += [f"### {spec['label']}", ""]
        rows = []
        for name, r in ctx["i3"][fam].items():
            bold = name.startswith("Ensemble") and "after" in name
            cells = [f"{r['real_correct']}/{r['n_real']}", f"{r['median_real_p']:.3f}", f"{r['confident']}/{r['n_real']}",
                     f"{r['confident_correct']}/{r['n_real']}", str(r["disagree"]), _pct(r["synth_recall"], 0),
                     f"{r['auc']:.3f}"]
            rows.append([f"**{name}**" if bold else name] + [f"**{c}**" if bold else c for c in cells])
        L += _md(["Model", "Real correct", "Median P(synth) on reals", "Confident (>0.15)", "Conf+correct",
                  "Backbones disagree", "Synth recall", "AUC"], rows)
        b, a = ctx["i3"][fam]["Ensemble 0.5/0.5 - v1 (before)"], ctx["i3"][fam]["Ensemble 0.5/0.5 - v2 (after)"]
        L += ["",
              f"Ensemble confident: **{b['confident']}/25 -> {a['confident']}/25** (F0 baseline 7/25); conf+correct "
              f"{b['confident_correct']}/25 -> {a['confident_correct']}/25; Gate F3 target >= {GATE_CONFIDENT}/25.", ""]

    # ---- 2. five clips
    L += ["## 2. Backbone-disagreement check (5 named REAL clips)", "",
          "P(synthetic); a REAL clip should be < 0.5. `Disagree` = wav2vec2 and WavLM on opposite sides of 0.5. "
          "`Conf+ok` = ensemble margin > 0.15 AND correct.", ""]
    for fam, spec in FAMILIES.items():
        L += [f"### {spec['label']}", ""]
        rows = []
        for r in ctx["five"][fam]:
            rows.append([f"`{r['clip']}`", f"{r['w2v']:.4f}", f"{r['wavlm_v1']:.4f}", f"**{r['wavlm_v2']:.4f}**",
                         f"{r['ens_v1']:.4f}", f"**{r['ens_v2']:.4f}**",
                         f"{'YES' if r['disagree_v1'] else 'no'} -> **{'YES' if r['disagree_v2'] else 'no'}**",
                         f"{'yes' if r['conf_v1'] else 'no'} -> **{'yes' if r['conf_v2'] else 'no'}**"])
        L += _md(["Clip", "wav2vec2", "WavLM v1 (before)", "WavLM v2 (after)", "Ensemble v1", "Ensemble v2",
                  "Disagree before -> after", "Conf+ok before -> after"], rows)
        n1 = sum(r["disagree_v1"] for r in ctx["five"][fam])
        n2 = sum(r["disagree_v2"] for r in ctx["five"][fam])
        L += ["", f"Backbones disagree on {n1}/5 before, {n2}/5 after.", ""]

    # ---- 3. F1.5
    f15 = ctx["f15"]
    ts_titles = {k: t for k, t, _, _ in fem.TEST_SETS}
    L += ["## 3. Full F1.5 evaluation (whole-clip matched heads), v2 substituted", "",
          f"Same code path as `fix_evaluate_matched.py` (`zero_shot_eval_from_cache`, weight_a={fem.WEIGHT_A}, threshold {fem.THRESHOLD}). "
          "Real recall = fraction of REAL clips scored < 0.5; **synthetic recall = fraction of SYNTHETIC clips scored >= 0.5**. "
          "`% pred synth` is the share of ALL clips the model calls synthetic: a head that stopped saying \"synthetic\" shows up here.", ""]
    for ts_key, title in ts_titles.items():
        n = f15["g"][ts_key]
        L += [f"### {title}  (n_real={n['n_real']:,}, n_synth={n['n_synth']:,})", ""]
        rows = []
        for v in F15_VARIANTS:
            m = f15[v["key"]][ts_key]
            bold = v["key"] in ("f", "g", "e2")
            cells = [_pct(m["accuracy"], 2), f"{m['roc_auc']:.4f}", _pct(m["eer"], 2), _pct(m["real_recall"], 1),
                     _pct(m["synth_recall"], 1), _pct(m["pred_synth_frac"], 1)]
            rows.append([f"**{v['label']}**" if bold else v["label"]] + ([f"**{c}**" for c in cells] if bold else cells))
        L += _md(["Variant", "Acc @0.5", "ROC-AUC", "EER", "Real recall", "Synth recall", "% pred synth"], rows)
        L.append("")

    f, g, e, e2 = f15["f"], f15["g"], f15["e"], f15["e2"]
    L += ["### F1.5 deltas, ensemble (f) v1 -> (g) v2 (same test set in every row)", ""]
    rows = []
    for ts_key, title in ts_titles.items():
        rows.append([title, _delta(g[ts_key]["accuracy"], f[ts_key]["accuracy"]), _delta(g[ts_key]["eer"], f[ts_key]["eer"]),
                     _delta(g[ts_key]["real_recall"], f[ts_key]["real_recall"]),
                     _delta(g[ts_key]["synth_recall"], f[ts_key]["synth_recall"])])
    L += _md(["Test set", "Delta acc", "Delta EER", "Delta real recall", "Delta SYNTH recall"], rows)
    L += ["", "### F1.5 deltas, WavLM alone (e) v1 -> (e2) v2", ""]
    rows = []
    for ts_key, title in ts_titles.items():
        rows.append([title, _delta(e2[ts_key]["accuracy"], e[ts_key]["accuracy"]), _delta(e2[ts_key]["eer"], e[ts_key]["eer"]),
                     _delta(e2[ts_key]["real_recall"], e[ts_key]["real_recall"]),
                     _delta(e2[ts_key]["synth_recall"], e[ts_key]["synth_recall"])])
    L += _md(["Test set", "Delta acc", "Delta EER", "Delta real recall", "Delta SYNTH recall"], rows)

    repro = ctx["f0_repro"]
    L += ["", f"F0 baseline reproduction by variant (c): **{'REPRODUCED' if repro['ok'] else 'NOT REPRODUCED'}** "
              f"(accuracy and EER on ASVspoof / Hindi original vs `fix_baseline.json`).", ""]

    # ---- 4. F2.4
    f24 = ctx["f24"]
    L += ["## 4. Full F2.4 evaluation (chunked heads), v2 substituted", "",
          "Same code path as `evaluate_chunked.py` (`load_chunk_set`, `aggregate_clips`, ensemble per chunk then aggregated). "
          "wav2vec2 = the unchanged v1 chunked head in both columns of every comparison.", ""]
    for key in ("asvspoof", "hindi"):
        s = f24[key]
        n1 = s["v1"]
        L += [f"### {s['title']}  ({n1['n_chunks']:,} chunks / {n1['n_clips']:,} clips)", ""]
        rows = []
        for level, how in (("chunk", None), ("clip", "mean"), ("clip", "max")):
            for model, mlabel in (("wavlm", "WavLM alone"), ("ensemble", "Ensemble 0.5/0.5")):
                for ver, vlabel in (("v1", "v1 (before)"), ("v2", "v2 (after)")):
                    m = s[ver][level][model if how is None else (model, how)]
                    bold = ver == "v2"
                    cells = [_pct(m["accuracy"], 2), f"{m['roc_auc']:.4f}", _pct(m["eer"], 2),
                             _pct(m["real_recall"], 1), _pct(m["synthetic_recall"], 1)]
                    lvl = "chunk" if how is None else f"clip-{how}"
                    rows.append([f"{lvl}: {mlabel} {vlabel}"] + ([f"**{c}**" for c in cells] if bold else cells))
        L += _md(["Level / model", "Acc @0.5", "ROC-AUC", "EER", "Real recall", "Synth recall"], rows)
        L.append("")
    gate_rows = []
    for key, ceiling in (("asvspoof", ec.GATE_EER_ASVSPOOF), ("hindi", ec.GATE_EER_HINDI)):
        for ver in ("v1", "v2"):
            eer = f24[key][ver]["chunk"]["ensemble"]["eer"]
            gate_rows.append([f24[key]["title"], ver, _pct(eer, 2), f"< {ceiling * 100:.0f}%", "PASS" if eer < ceiling else "**FAIL**"])
    L += ["### Gate F2 chunk-EER check (ensemble), v1 vs v2", ""] + _md(["Test set", "Heads", "Chunk EER", "Gate", "Status"], gate_rows) + [""]

    # ---- 5. Summary
    L += ["## 5. Gate F3 read-out", ""] + [f"- {s}" for s in ctx["gate_lines"]] + [""]
    return "\n".join(L)


def gate_lines(i3, f15, f24, base) -> List[str]:
    lines: List[str] = []
    f0_hi_synth = float(base["item3_hindi"]["synthetic_recall"])
    f0_asv_synth = float(base["item2_asvspoof"]["synthetic_recall"])
    for fam, spec in FAMILIES.items():
        a = i3[fam]["Ensemble 0.5/0.5 - v2 (after)"]
        b = i3[fam]["Ensemble 0.5/0.5 - v1 (before)"]
        ok = a["confident_correct"] >= GATE_CONFIDENT
        lines.append(f"{spec['label']}: ensemble confident+correct {b['confident_correct']}/25 -> {a['confident_correct']}/25 "
                     f"(target >= {GATE_CONFIDENT}): **{'MET' if ok else 'NOT MET'}**; original-definition confident "
                     f"{b['confident']}/25 -> {a['confident']}/25.")
        floor = f0_hi_synth - SYNTH_TOLERANCE
        lines.append(f"{spec['label']}: soumya synthetic recall {_pct(b['synth_recall'], 0)} -> {_pct(a['synth_recall'], 0)} "
                     f"(F0 {_pct(f0_hi_synth, 0)}, floor {_pct(floor, 0)}): **{'MET' if a['synth_recall'] >= floor - 1e-9 else 'NOT MET'}**"
                     f" (25 clips: one clip = 4 pp).")
    g, f = f15["g"], f15["f"]
    for ts in ("asvspoof", "hindi_matched"):
        lines.append(f"F1.5 matched ensemble on {ts}: synthetic recall {_pct(f[ts]['synth_recall'], 2)} -> {_pct(g[ts]['synth_recall'], 2)}, "
                     f"real recall {_pct(f[ts]['real_recall'], 2)} -> {_pct(g[ts]['real_recall'], 2)}, "
                     f"EER {_pct(f[ts]['eer'], 2)} -> {_pct(g[ts]['eer'], 2)}.")
    floor = f0_asv_synth - SYNTH_TOLERANCE
    lines.append(f"ASVspoof synthetic recall vs F0 ({_pct(f0_asv_synth, 2)}, floor {_pct(floor, 2)}): matched ensemble v2 "
                 f"{_pct(g['asvspoof']['synth_recall'], 2)} -> **{'within tolerance' if g['asvspoof']['synth_recall'] >= floor else 'MORE than 3 pp below F0'}**.")
    for key in ("asvspoof", "hindi"):
        m1, m2 = f24[key]["v1"]["clip"][("ensemble", "mean")], f24[key]["v2"]["clip"][("ensemble", "mean")]
        lines.append(f"F2.4 chunked ensemble, clip-mean on {f24[key]['title']}: synthetic recall {_pct(m1['synthetic_recall'], 2)} -> "
                     f"{_pct(m2['synthetic_recall'], 2)}, real recall {_pct(m1['real_recall'], 2)} -> {_pct(m2['real_recall'], 2)}, "
                     f"EER {_pct(m1['eer'], 2)} -> {_pct(m2['eer'], 2)}.")
    return lines


def print_summary(ctx: Dict[str, Any], out: Path) -> None:
    bar = "=" * 100
    print("\n" + bar)
    print(f" WAVLM v2 RETRAIN - C={C_V2:g}, Hindi oversample {OVERSAMPLE_V2}x (F3.1 LOSO-selected cell)")
    print(bar)
    print(" NEW files:       " + ", ".join(ctx["new_files_short"]))
    print(" UNCHANGED:       all v1 files; wav2vec2 heads were NOT retrained (no wav2vec2_*_v2 written)")
    print(" v1 files intact: " + str(ctx["snapshot_ok"]))
    print("-" * 100)
    for fam, spec in FAMILIES.items():
        print(f" {spec['label']} - soumya's 25 REAL clips")
        print(f"   {'model':<34}{'real ok':>8}{'conf>0.15':>11}{'conf+ok':>9}{'disagree':>10}{'synth rec':>11}{'AUC':>7}")
        for name, r in ctx["i3"][fam].items():
            print(f"   {name:<34}{r['real_correct']:>5}/25{r['confident']:>8}/25{r['confident_correct']:>6}/25"
                  f"{r['disagree']:>10}{_pct(r['synth_recall'], 0):>11}{r['auc']:>7.3f}")
        print("   5-clip check (ensemble P: before -> after; disagree before -> after):")
        for r in ctx["five"][fam]:
            print(f"     {r['clip']:<20} w2v {r['w2v']:.3f}  wavlm {r['wavlm_v1']:.3f} -> {r['wavlm_v2']:.3f}  "
                  f"ens {r['ens_v1']:.3f} -> {r['ens_v2']:.3f}  "
                  f"disagree {'Y' if r['disagree_v1'] else 'n'} -> {'Y' if r['disagree_v2'] else 'n'}")
        print()
    print("-" * 100)
    for s in ctx["gate_lines"]:
        print(" - " + s)
    print(bar)
    print(f" Report written to: {out}\n")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output_md", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    t0 = time.time()
    try:
        before = snapshot_models()
        diag_notes = check_against_diagnosis()
        train = {fam: retrain(fam) for fam in FAMILIES}
        after = snapshot_models()

        # v2 files may already exist on a re-run (they are ours to rewrite); everything else must be untouched.
        v2_stems = tuple(spec["v2"] for spec in FAMILIES.values())
        is_v2 = lambda n: n.startswith(v2_stems)  # noqa: E731
        changed = [n for n, sig in before.items() if not is_v2(n) and after.get(n) != sig]
        if changed:
            raise RuntimeError(f"Existing model files changed (must never happen): {changed}")
        new_files = sorted(n for n in after if is_v2(n))
        stray = sorted(set(after) - set(before) - set(new_files))
        if stray:
            raise RuntimeError(f"Unexpected new files in models/classifiers/: {stray}")

        soumya = {fam: soumya_scores(fam) for fam in FAMILIES}
        i3 = {fam: before_after(soumya[fam]) for fam in FAMILIES}
        for fam in FAMILIES:
            got = i3[fam]["WavLM alone - v2 (after)"]["auc"]
            if abs(got - EXPECTED_AUC[fam]) > AUC_REPRO_TOL:
                logger.warning("%s: WavLM v2 soumya AUC %.3f differs from F3.1's %.3f", fam, got, EXPECTED_AUC[fam])
        five = {fam: five_clip_rows(soumya[fam]) for fam in FAMILIES}

        f15 = run_f15()
        f24 = run_f24()
        f0_repro = fem.check_baseline_reproduction({"c": f15["c"]})
        base = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))

        ctx = {
            "train": train, "diag_notes": diag_notes, "snapshot_ok": "yes", "new_files": new_files,
            "new_files_short": [t["saved"] for t in train.values()],
            "i3": i3, "five": five, "f15": f15, "f24": f24, "f0_repro": f0_repro,
        }
        ctx["gate_lines"] = gate_lines(i3, f15, f24, base)
        report = build_report(ctx)
    except Exception as exc:
        logger.error("WavLM v2 retrain failed: %s", exc)
        raise

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")
    logger.info("Saved report to %s (%.1fs)", args.output_md, time.time() - t0)
    print_summary(ctx, args.output_md)


if __name__ == "__main__":
    main()
