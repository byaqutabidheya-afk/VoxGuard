#!/usr/bin/env python3
"""
scripts/evaluate_chunked.py — Evaluate the chunk-native heads (Phase F2.4) -> fix_chunked_eval.md.

Scores the held-out chunked eval caches with the NEW ``{model}_chunked_logreg``
heads (from cached chunk embeddings — nothing is re-extracted) and writes
``models/reports/fix_chunked_eval.md`` with:

(a) CHUNK-LEVEL metrics — one row of the eval cache = one chunk, label inherited
    from the parent clip — on ASVspoof eval and the Hindi MATCHED eval, for
    wav2vec2 alone, WavLM alone and the weighted-average ensemble.
(b) CLIP-LEVEL metrics — each clip's chunk scores aggregated to one score, under
    BOTH mean- and max-aggregation — beside the whole-clip heads scored on the
    *same clips*, and beside the F0 baseline exactly as recorded.
(c) A per-clip table (chunk-score mean / max / min) for the 5 verified demo pairs
    and the 3 window-sweep real clips.

Conventions (stated again in the report)
----------------------------------------
* Ensemble = ``0.5 * P_wav2vec2 + 0.5 * P_wavlm`` applied PER CHUNK, then
  aggregated per clip (mean or max). Single-backbone rows aggregate that
  backbone's own chunk probabilities. For mean-aggregation the order is
  irrelevant; for max-aggregation it is not.
* Accuracy / per-class recall use threshold 0.5 (same as the F0 baseline);
  ROC-AUC and EER are threshold-free.
* Chunk-level rows are dominated by longer clips (a clip with 5 chunks counts 5x)
  and inherit label noise; clip-level rows weight every clip once.

Two comparability traps this script guards against (see the report's notes):
the chunked ASVspoof eval is a SUBSET of the eval split the F0 baseline used, and
the chunked Hindi eval is the duration-MATCHED audio while the F0 baseline used
the ORIGINAL audio. So the whole-clip reference rows are re-scored on exactly the
chunk-eval clips, and the F0 numbers are shown separately, as recorded.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from voxguard import config
from voxguard.classifier.cross_eval import (
    _metric_dict,
    _predict_scores,
    weighted_average_ensemble,
)
from voxguard.classifier.head import load_classifier
from voxguard.embeddings.cache import load_cached_embeddings
from voxguard.utils.logging_utils import get_logger

logger = get_logger("evaluate_chunked")

EMBEDDINGS_DIR = config.MODELS_DIR / "embeddings"
CLASSIFIERS_DIR = config.MODELS_DIR / "classifiers"
DEFAULT_REPORT_PATH = config.MODELS_DIR / "reports" / "fix_chunked_eval.md"
BASELINE_JSON = config.MODELS_DIR / "reports" / "fix_baseline.json"

BACKBONES = ("wav2vec2", "wavlm")
WEIGHT_A = 0.5          # weight on wav2vec2; same as the F0 baseline / production detector
THRESHOLD = 0.5
REPRO_TOLERANCE = 1e-6
ASV_ID_RE = re.compile(r"LA_E_\d+")

# Gate F2 (Instructions2.md): chunk-level EER ceilings.
GATE_EER_ASVSPOOF = 0.25
GATE_EER_HINDI = 0.40

MATCHED_DIR = "data/raw/hindi_hinglish_matched"

# The 5 verified Phase 6 demo pairs and the 3 window-sweep real clips — the same
# clips scripts/fix_capture_baseline.py used, addressed here by file name and
# resolved to their duration-MATCHED versions (what the chunk caches contain).
DEMO_PAIRS: List[Tuple[str, str, str]] = [
    ("Pair 1: Casual Neutral (byaquta)", "byaquta_neutral_09", "byaquta_neutral_09_clone"),
    ("Pair 2: Everyday Tech (mahato)", "mahato_neutral_04", "mahato_neutral_04_clone"),
    ("Pair 3: Urgent Legal Scam (byaquta)", "byaquta_scam_16", "byaquta_scam_16_clone"),
    ("Pair 4: Authority Customs Scam (mahato)", "mahato_scam_12", "mahato_scam_12_clone"),
    ("Pair 5: Held-Out Casual (soumya)", "soumya_neutral_03", "soumya_neutral_03_clone"),
]
SWEEP_REALS: List[str] = ["byaquta_neutral_01", "soumya_control_21", "soumya_scam_11"]

# (key, title, dataset, split) for the two held-out chunk-eval sets.
EVAL_SETS = [
    ("asvspoof", "ASVspoof2019 eval", "asvspoof2019", "eval"),
    ("hindi", "Hindi MATCHED eval (soumya)", "hindi", "eval"),
]

# Whole-clip reference heads scored on the same clips as the chunk eval.
WHOLE_CLIP_REFS = [
    ("c", "Whole-clip (c): F0 production heads `hindi_combined`", "hindi_combined"),
    ("f", "Whole-clip (f): `hindi_matched` heads", "hindi_matched"),
]


# =============================================================================
# Scoring
# =============================================================================

def _load_heads(pattern: str) -> Dict[str, Tuple[Any, Any]]:
    """Loads both backbones' (model, scaler) pairs; pattern is e.g. '{}_chunked_logreg'."""
    return {b: load_classifier(CLASSIFIERS_DIR / pattern.format(b)) for b in BACKBONES}


def _score(X: np.ndarray, head: Tuple[Any, Any]) -> np.ndarray:
    model, scaler = head
    return _predict_scores(model, scaler.transform(X))


def _norm(p: Any) -> str:
    return str(p).replace("\\", "/")


def load_chunk_set(dataset: str, split: str, heads: Dict[str, Tuple[Any, Any]]) -> Dict[str, Any]:
    """Scores one chunked cache with the chunked heads. Returns manifest + per-model scores."""
    manifests: Dict[str, pd.DataFrame] = {}
    scores: Dict[str, np.ndarray] = {}
    for b in BACKBONES:
        X, m = load_cached_embeddings(EMBEDDINGS_DIR / f"{b}_{dataset}_{split}_chunked.npy")
        manifests[b] = m
        scores[b] = _score(X, heads[b])

    key_cols = ["parent_filepath", "chunk_index", "label"]
    a, b = (manifests[k][key_cols].reset_index(drop=True) for k in BACKBONES)
    if not a.equals(b):
        raise ValueError(f"{dataset}/{split}: wav2vec2 and wavlm chunk manifests are not row-aligned.")

    manifest = manifests[BACKBONES[0]].reset_index(drop=True)
    manifest["parent_filepath"] = manifest["parent_filepath"].map(_norm)
    scores["ensemble"] = weighted_average_ensemble(scores["wav2vec2"], scores["wavlm"], WEIGHT_A)
    return {"manifest": manifest, "scores": scores}


def metrics_with_recall(y: Any, s: np.ndarray) -> Dict[str, Any]:
    m = _metric_dict(y, s, threshold=THRESHOLD)
    (tn, fp), (fn, tp) = m["confusion_matrix"]
    m["real_recall"] = tn / (tn + fp) if (tn + fp) else float("nan")
    m["synthetic_recall"] = tp / (tp + fn) if (tp + fn) else float("nan")
    m["n_real"], m["n_synthetic"] = tn + fp, fn + tp
    return m


def aggregate_clips(
    manifest: pd.DataFrame, scores: np.ndarray, how: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One score per clip. Returns (clip ids, labels, aggregated scores), sorted by clip id."""
    df = pd.DataFrame(
        {"clip": manifest["parent_filepath"].values, "label": manifest["label"].values, "score": scores}
    )
    g = df.groupby("clip", sort=True)
    if (g["label"].nunique() > 1).any():
        raise ValueError("A clip has chunks with conflicting labels.")
    agg = g["score"].mean() if how == "mean" else g["score"].max()
    return agg.index.values, g["label"].first().values, agg.values


# =============================================================================
# Whole-clip references (same clips) and the F0 reproduction check
# =============================================================================

def score_whole_clip(
    cache_stem: str, heads: Dict[str, Tuple[Any, Any]], key_fn
) -> pd.DataFrame:
    """Ensemble whole-clip scores for a whole-clip cache; returns DataFrame[key, label, score]."""
    manifests, scores = {}, {}
    for b in BACKBONES:
        X, m = load_cached_embeddings(EMBEDDINGS_DIR / f"{b}_{cache_stem}.npy")
        manifests[b], scores[b] = m, _score(X, heads[b])
    if list(manifests["wav2vec2"]["filepath"]) != list(manifests["wavlm"]["filepath"]):
        raise ValueError(f"{cache_stem}: wav2vec2 and wavlm whole-clip manifests are not aligned.")
    m = manifests["wav2vec2"]
    return pd.DataFrame({
        "key": [key_fn(p) for p in m["filepath"]],
        "label": m["label"].values,
        "score": weighted_average_ensemble(scores["wav2vec2"], scores["wavlm"], WEIGHT_A),
    })


def _asv_key(p: Any) -> str:
    hit = ASV_ID_RE.search(str(p))
    if hit is None:
        raise ValueError(f"No ASVspoof eval id in path {p!r}")
    return hit.group(0)


def whole_clip_on_same_clips(
    eval_key: str, clip_ids: np.ndarray, clip_labels: np.ndarray, heads_name: str
) -> Dict[str, Any]:
    """Whole-clip ensemble metrics restricted to exactly the chunk-eval clips."""
    heads = _load_heads(f"{{}}_{heads_name}_logreg")
    if eval_key == "asvspoof":
        whole = score_whole_clip("eval", heads, _asv_key)
        wanted = pd.Series(clip_labels, index=[_asv_key(c) for c in clip_ids])
    else:
        whole = score_whole_clip("hindi_eval_matched", heads, _norm)
        wanted = pd.Series(clip_labels, index=[_norm(c) for c in clip_ids])

    whole = whole.set_index("key")
    missing = wanted.index.difference(whole.index)
    if len(missing):
        raise ValueError(f"{eval_key}: {len(missing)} chunk-eval clips missing from whole-clip cache, e.g. {list(missing[:3])}")
    sub = whole.loc[wanted.index]
    if (sub["label"].values != wanted.values).any():
        raise ValueError(f"{eval_key}: label disagreement between whole-clip and chunk manifests.")
    return metrics_with_recall(sub["label"].values, sub["score"].values)


def check_f0_reproduction() -> Optional[Dict[str, Any]]:
    """Re-scores the F0 production heads on the FULL eval sets the baseline used; compares to the JSON."""
    if not BASELINE_JSON.exists():
        return None
    base = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    heads = _load_heads("{}_hindi_combined_logreg")
    rows, ok = [], True
    for label, stem, key in (
        ("ASVspoof2019 eval (full)", "eval", "item2_asvspoof"),
        ("Hindi ORIGINAL eval", "hindi_eval", "item3_hindi"),
    ):
        whole = score_whole_clip(stem, heads, _norm)
        got = metrics_with_recall(whole["label"].values, whole["score"].values)
        for metric in ("accuracy", "eer"):
            match = abs(got[metric] - float(base[key][metric])) <= REPRO_TOLERANCE
            ok &= match
            rows.append((label, metric, float(base[key][metric]), got[metric], match))
    return {"ok": ok, "rows": rows, "base": base}


# =============================================================================
# Per-clip table (c)
# =============================================================================

def _find_clip(kind: str, name: str, sets: Dict[str, Dict[str, Any]]) -> Tuple[str, str, np.ndarray, str]:
    """Locates a matched clip in the Hindi chunk caches. Returns (path, split, ensemble scores, label)."""
    path = f"{MATCHED_DIR}/{kind}/{name}.wav"
    for split, cs in sets.items():
        mask = (cs["manifest"]["parent_filepath"] == path).values
        if mask.any():
            return path, split, cs["scores"]["ensemble"][mask], str(cs["manifest"]["label"].values[mask][0])
    raise ValueError(f"{path} not found in the Hindi chunked train/eval caches.")


def _f0_stream_lookup(base: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """File name -> F0 whole-clip streaming outcome (recorded on the ORIGINAL, unmatched audio)."""
    if not base:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for p in base["item5_streaming"]["demo_pairs"]:
        out[p["real"]["file"]] = p["real"]
        out[p["synthetic"]["file"]] = p["synthetic"]
    for r in base["item5_streaming"]["sweep_real_clips"]:
        out[r["file"]] = r
    return out


def build_clip_rows(hindi_sets: Dict[str, Dict[str, Any]], base: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    f0 = _f0_stream_lookup(base)
    rows: List[Dict[str, Any]] = []

    def add(group: str, kind: str, name: str) -> None:
        folder = "real" if kind == "REAL" else "synthetic"
        path, split, s, label = _find_clip(folder, name, hindi_sets)
        rows.append({
            "group": group, "kind": kind, "file": f"{name}.wav", "split": split, "label": label,
            "n": int(len(s)), "mean": float(s.mean()), "max": float(s.max()), "min": float(s.min()),
            "mean_correct": (s.mean() >= THRESHOLD) == (label == "synthetic"),
            "f0": f0.get(f"{name}.wav"),
        })

    for group, real, synth in DEMO_PAIRS:
        add(group, "REAL", real)
        add(group, "SYNTH", synth)
    for name in SWEEP_REALS:
        add("Window-sweep real clip", "REAL", name)
    return rows


# =============================================================================
# Report
# =============================================================================

def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.2f}%"


def _md_table(header: List[str], rows: List[List[str]], align: Optional[List[str]] = None) -> List[str]:
    align = align or ["---"] + [":---:"] * (len(header) - 1)
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def _metric_cells(m: Dict[str, Any]) -> List[str]:
    return [_pct(m["accuracy"]), f"{m['roc_auc']:.4f}", _pct(m["eer"]),
            _pct(m["real_recall"]), _pct(m["synthetic_recall"])]


METRIC_HEADER = ["Acc @0.5", "ROC-AUC", "EER", "Real recall", "Synth recall"]
MODEL_LABELS = {"wav2vec2": "wav2vec2 alone", "wavlm": "WavLM alone", "ensemble": "Weighted average (0.5 / 0.5)"}


def compute_all() -> Dict[str, Any]:
    heads = _load_heads("{}_chunked_logreg")
    res: Dict[str, Any] = {"sets": {}, "hindi_train": None}

    for key, title, dataset, split in EVAL_SETS:
        cs = load_chunk_set(dataset, split, heads)
        man = cs["manifest"]
        entry: Dict[str, Any] = {
            "title": title, "n_chunks": len(man), "n_clips": int(man["parent_filepath"].nunique()),
            "chunk": {}, "clip": {},
        }
        for model in (*BACKBONES, "ensemble"):
            entry["chunk"][model] = metrics_with_recall(man["label"].values, cs["scores"][model])
            for how in ("mean", "max"):
                ids, labels, agg = aggregate_clips(man, cs["scores"][model], how)
                entry["clip"][(model, how)] = metrics_with_recall(labels, agg)
                entry["clip_ids"], entry["clip_labels"] = ids, labels
        entry["whole_clip"] = {
            ref_key: whole_clip_on_same_clips(key, entry["clip_ids"], entry["clip_labels"], heads_name)
            for ref_key, _, heads_name in WHOLE_CLIP_REFS
        }
        res["sets"][key] = entry
        res["hindi_eval_scored" if key == "hindi" else "asv_eval_scored"] = cs

    res["hindi_sets"] = {
        "eval": res["hindi_eval_scored"],
        "train": load_chunk_set("hindi", "train", heads),
    }
    return res


def build_report(res: Dict[str, Any], repro: Optional[Dict[str, Any]], clip_rows: List[Dict[str, Any]]) -> str:
    base = repro["base"] if repro else None
    sets = res["sets"]
    asv, hin = sets["asvspoof"], sets["hindi"]

    L: List[str] = [
        "# Chunked Heads — Held-Out Evaluation (Phase F2.4)",
        "",
        f"**Generated at:** {datetime.now(timezone.utc).isoformat()}",
        "**Script:** `scripts/evaluate_chunked.py`",
        "**Models:** `models/classifiers/wav2vec2_chunked_logreg.joblib`, "
        "`models/classifiers/wavlm_chunked_logreg.joblib` (each with its own scaler)",
        f"**Ensemble:** `{WEIGHT_A} * P(wav2vec2) + {1 - WEIGHT_A} * P(WavLM)`, applied per chunk. "
        f"Accuracy / recall at threshold {THRESHOLD}; ROC-AUC and EER are threshold-free.",
        "**Inputs:** cached chunk embeddings only (`{model}_{dataset}_{split}_chunked.npy`); nothing re-extracted.",
        "",
        "## Which aggregation is used where",
        "",
        "- **Section (a) — chunk level:** NO aggregation. Every chunk is one sample; its label is inherited "
        "from its parent clip. Long clips contribute more chunks, so long clips weigh more here.",
        "- **Section (b) — clip level:** each clip's chunk probabilities are aggregated to ONE score, and "
        "results are reported under **both mean-aggregation and max-aggregation**. The ensemble average is "
        "taken per chunk first, then aggregated. Every clip counts once.",
        "- **Section (c) — per clip:** raw chunk-score mean / max / min of the ensemble; no aggregation choice "
        "is applied to a verdict, except the explicit `mean >= 0.5` column.",
        "",
        "> [!WARNING]",
        "> **Read the comparability notes in section (b) before comparing anything to the F0 baseline.** "
        "The chunked ASVspoof eval is a subset of the F0 eval, and the chunked Hindi eval is duration-MATCHED "
        "audio while the F0 Hindi numbers are on the ORIGINAL audio.",
        "",
        "## Test sets",
        "",
    ]
    L += _md_table(
        ["Test set", "Chunks", "Clips", "Chunks real / synth", "Clips real / synth"],
        [[s["title"], f"{s['n_chunks']:,}", f"{s['n_clips']:,}",
          f"{s['chunk']['ensemble']['n_real']:,} / {s['chunk']['ensemble']['n_synthetic']:,}",
          f"{s['clip'][('ensemble', 'mean')]['n_real']:,} / {s['clip'][('ensemble', 'mean')]['n_synthetic']:,}"]
         for s in sets.values()],
        ["---", ":---:", ":---:", ":---:", ":---:"],
    )
    L += [
        "",
        "Clips with no non-silent chunk are absent from the caches and therefore from every row below. "
        f"The Hindi eval has only {hin['n_clips']} clips, so each clip is worth {100 / hin['n_clips']:.1f} pp "
        "of clip-level accuracy and its EER is coarse.",
        "",
        "## (a) Chunk-level metrics",
        "",
    ]
    rows = []
    for s in sets.values():
        for model in (*BACKBONES, "ensemble"):
            bold = model == "ensemble"
            name = f"**{MODEL_LABELS[model]}**" if bold else MODEL_LABELS[model]
            rows.append([s["title"], name] + _metric_cells(s["chunk"][model]))
    L += _md_table(["Test set", "Model"] + METRIC_HEADER, rows, ["---", "---"] + [":---:"] * 5)

    # Gate check
    L += ["", "### Gate F2 EER check (chunk level)", ""]
    gate_rows = []
    for s, ceiling in ((asv, GATE_EER_ASVSPOOF), (hin, GATE_EER_HINDI)):
        for model in (*BACKBONES, "ensemble"):
            eer = s["chunk"][model]["eer"]
            gate_rows.append([s["title"], MODEL_LABELS[model], _pct(eer), f"< {ceiling * 100:.0f}%",
                              "PASS" if eer < ceiling else "**FAIL**"])
    L += _md_table(["Test set", "Model", "Chunk EER", "Gate", "Status"], gate_rows, ["---", "---"] + [":---:"] * 3)
    L += [
        "",
        "The gate text does not name a model; the ensemble is the production detector, so it is the row that "
        "decides. The single-backbone rows are shown for completeness.",
        "",
        "## (b) Clip-level metrics (aggregated chunk scores)",
        "",
    ]

    for s in (asv, hin):
        L += [f"### {s['title']}", ""]
        rows = []
        for model in (*BACKBONES, "ensemble"):
            for how in ("mean", "max"):
                name = f"Chunked {MODEL_LABELS[model]} — {how}-aggregated"
                if model == "ensemble":
                    name = f"**{name}**"
                rows.append([name] + _metric_cells(s["clip"][(model, how)]))
        for ref_key, ref_label, _ in WHOLE_CLIP_REFS:
            rows.append([f"{ref_label} — same {s['n_clips']:,} clips"] + _metric_cells(s["whole_clip"][ref_key]))
        L += _md_table(["Row (all rows scored on the same clips)"] + METRIC_HEADER, rows,
                       ["---"] + [":---:"] * 5)
        L.append("")

    L += ["### F0 baseline as recorded (different test sets — do not subtract from the tables above)", ""]
    if base:
        rows = []
        for label, key in (("ASVspoof2019 eval, FULL split", "item2_asvspoof"),
                           ("Hindi ORIGINAL eval (unmatched audio)", "item3_hindi")):
            b = base[key]
            rows.append([label, f"{b['total_eval_samples']:,}", _pct(b["accuracy"]), f"{b['roc_auc']:.4f}",
                         _pct(b["eer"]), _pct(b["real_recall"]), _pct(b["synthetic_recall"])])
        L += _md_table(["Test set", "Clips"] + METRIC_HEADER, rows, ["---", ":---:"] + [":---:"] * 5)
        L.append("")
        if repro:
            state = "REPRODUCED" if repro["ok"] else "NOT REPRODUCED — the whole-clip reference rows cannot be trusted"
            L.append(
                f"Re-scoring the F0 heads from cache on those full splits vs `{BASELINE_JSON.name}` "
                f"(accuracy and EER, tolerance {REPRO_TOLERANCE:g}): **{state}**."
            )
    else:
        L.append(f"`{BASELINE_JSON.name}` not found — F0 numbers unavailable.")

    asv_real_frac = asv["clip"][("ensemble", "mean")]["n_real"] / asv["n_clips"]
    f0_frac = None
    if base:
        cm = base["item2_asvspoof"]["confusion_matrix"]
        f0_frac = (cm[0][0] + cm[0][1]) / base["item2_asvspoof"]["total_eval_samples"]
    L += [
        "",
        "### Comparability notes",
        "",
        f"1. **ASVspoof subset.** The chunked eval holds {asv['n_clips']:,} clips"
        + (f" vs {base['item2_asvspoof']['total_eval_samples']:,} in the F0 baseline" if base else "")
        + f", and its class mix differs ({asv_real_frac * 100:.1f}% real"
        + (f" vs {f0_frac * 100:.1f}%" if f0_frac is not None else "")
        + "), so accuracy on the two is not comparable even roughly. The whole-clip rows in the table "
        "above are the like-for-like comparison: the same model family scored on exactly these clips.",
        "2. **Hindi matched vs original.** The F0 Hindi numbers were measured on the ORIGINAL (unmatched) "
        "eval audio, which carried a duration shortcut. The chunked eval uses the duration-MATCHED audio. "
        "Compare chunked to the whole-clip rows in the Hindi table (same matched clips), never to the F0 "
        "Hindi line.",
        "3. **Max-aggregation at threshold 0.5.** The maximum of several noisy chunk probabilities is biased "
        "upward, so max-aggregated accuracy and real recall at 0.5 are penalised by construction. "
        "ROC-AUC and EER are threshold-free and are the fairer read of max-aggregation.",
        "4. **Head-to-head vs (c)/(f) is not an ablation of chunking alone.** The chunked heads differ from "
        "the whole-clip heads in their training rows as well as their input, so a difference reflects the "
        "whole recipe, not the chunking by itself.",
        "",
        "## (c) Per-clip chunk-score table: demo pairs and window-sweep real clips",
        "",
        "Scores are the **ensemble** P(synthetic) per chunk, on the duration-MATCHED audio. "
        "`Split` says whether the chunked heads were TRAINED on the clip (`train`: the score is not "
        "held-out evidence) or never saw its speaker (`eval`). "
        "**Only `soumya_*` clips are held-out;** the byaquta and mahato clips are training data.",
        "",
        "`F0 stream flag` is what the pre-fix whole-clip StreamingSession did on the ORIGINAL audio "
        "(`fix_baseline.json`) — context for which cases were previously failing, not a like-for-like column.",
        "",
    ]
    tab = []
    for r in clip_rows:
        f0 = r["f0"]
        f0_cell = "n/a" if f0 is None else ("flagged" if f0["flagged"] else "not flagged")
        # A pre-fix failure: a REAL clip that flagged, or a SYNTH clip that did not.
        if f0 is not None and ((r["kind"] == "REAL" and f0["flagged"]) or (r["kind"] == "SYNTH" and not f0["flagged"])):
            f0_cell = f"**{f0_cell} (wrong)**"
        verdict = "yes" if r["mean_correct"] else "**NO**"
        tab.append([r["group"], r["kind"], f"`{r['file']}`", r["split"], str(r["n"]),
                    f"{r['mean']:.3f}", f"{r['max']:.3f}", f"{r['min']:.3f}", verdict, f0_cell])
    L += _md_table(
        ["Pair / group", "Kind", "File", "Split", "Chunks", "Mean", "Max", "Min", "Mean-agg correct @0.5", "F0 stream flag"],
        tab, ["---", ":---:", "---"] + [":---:"] * 7)
    real_max_over = [r["file"] for r in clip_rows if r["kind"] == "REAL" and r["max"] >= THRESHOLD]
    L += [
        "",
        "Real clips whose single highest chunk reaches 0.5: "
        + (", ".join(f"`{f}`" for f in real_max_over) if real_max_over else "none")
        + ". Any such clip would flag under a max-style rule even where its mean is low.",
        "",
    ]
    return "\n".join(L)


def print_summary(res: Dict[str, Any], repro: Optional[Dict[str, Any]], out: Path) -> bool:
    bar = "=" * 100
    print("\n" + bar)
    print(" CHUNKED HEADS - held-out evaluation (ensemble; chunk level, then clip level)")
    print(bar)
    gate_ok = True
    for key, ceiling in (("asvspoof", GATE_EER_ASVSPOOF), ("hindi", GATE_EER_HINDI)):
        s = res["sets"][key]
        c = s["chunk"]["ensemble"]
        ok = c["eer"] < ceiling
        gate_ok &= ok
        print(f" {s['title']:<32} {s['n_chunks']:>6} chunks / {s['n_clips']:>5} clips")
        print(f"   chunk-level  acc {_pct(c['accuracy'])}  AUC {c['roc_auc']:.4f}  EER {_pct(c['eer'])}  "
              f"[gate < {ceiling * 100:.0f}%: {'PASS' if ok else 'FAIL'}]")
        for how in ("mean", "max"):
            m = s["clip"][("ensemble", how)]
            print(f"   clip {how:<4}    acc {_pct(m['accuracy'])}  AUC {m['roc_auc']:.4f}  EER {_pct(m['eer'])}  "
                  f"real recall {_pct(m['real_recall'])}")
        for ref_key, ref_label, _ in WHOLE_CLIP_REFS:
            m = s["whole_clip"][ref_key]
            print(f"   whole-clip ({ref_key}) acc {_pct(m['accuracy'])}  AUC {m['roc_auc']:.4f}  EER {_pct(m['eer'])}  "
                  f"real recall {_pct(m['real_recall'])}   (same clips)")
        print()
    if repro:
        print(f" F0 baseline reproduced from cache: {'YES' if repro['ok'] else 'NO - investigate'}")
    print(f" Gate F2 chunk-EER check (ensemble): {'PASS' if gate_ok else 'FAIL'}")
    print(bar)
    print(f" Report written to: {out}\n")
    return gate_ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output_md", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    try:
        res = compute_all()
        repro = check_f0_reproduction()
        clip_rows = build_clip_rows(res["hindi_sets"], repro["base"] if repro else None)
        report = build_report(res, repro, clip_rows)
    except Exception as exc:
        logger.error("Chunked evaluation failed: %s", exc)
        sys.exit(1)

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")
    logger.info("Saved chunked evaluation report to %s", args.output_md)
    print_summary(res, repro, args.output_md)


if __name__ == "__main__":
    main()
