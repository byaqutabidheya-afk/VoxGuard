#!/usr/bin/env python3
"""
scripts/fix_evaluate_matched.py — Compare original vs duration-matched Hindi heads on three test sets.

Evaluates six variants:
  (a) wav2vec2 hindi_combined            (original, baseline)
  (b) wavlm hindi_combined               (original, baseline)
  (c) weighted average of (a)+(b)        (current production detector)
  (d) wav2vec2 hindi_matched             (new)
  (e) wavlm hindi_matched                (new)
  (f) weighted average of (d)+(e)        (candidate production detector)

on three test sets:
  - ASVspoof2019 eval                    (English regression check)
  - ORIGINAL Hindi eval  (soumya, unmatched)        — continuity with the F0 baseline
  - MATCHED Hindi eval   (soumya, duration-matched) — the honest test set going forward

Evaluation reuses zero_shot_eval_from_cache / zero_shot_eval_weighted_average_from_cache
(Phase 3 Prompt 3.1) with the same settings the F0 baseline used (weight_a=0.5,
threshold=0.5). Variant (c) on the original split is checked against
models/reports/fix_baseline.json to confirm the baseline is reproduced.

Writes models/reports/fix_matched_comparison.md.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from voxguard import config
from voxguard.classifier.cross_eval import (
    zero_shot_eval_from_cache,
    zero_shot_eval_weighted_average_from_cache,
)
from voxguard.utils.logging_utils import get_logger

logger = get_logger("fix_evaluate_matched")

CLASSIFIERS_DIR = config.MODELS_DIR / "classifiers"
DEFAULT_REPORT_PATH = config.MODELS_DIR / "reports" / "fix_matched_comparison.md"
BASELINE_JSON = config.MODELS_DIR / "reports" / "fix_baseline.json"

# Same settings as the F0 baseline capture (scripts/fix_capture_baseline.py).
WEIGHT_A = 0.5
THRESHOLD = 0.5
REPRO_TOLERANCE = 1e-6

# (key, label, dataset, split) — dataset/split feed cross_eval.resolve_cache_path:
#   asvspoof2019/eval     -> {model}_eval.npy
#   hindi_eval            -> {model}_hindi_eval.npy          (original soumya split)
#   hindi_eval_matched    -> {model}_hindi_eval_matched.npy  (matched soumya split)
TEST_SETS = [
    ("asvspoof", "ASVspoof2019 eval", "asvspoof2019", "eval"),
    ("hindi_original", "Hindi eval ORIGINAL (soumya, unmatched)", "hindi_eval", "eval"),
    ("hindi_matched", "Hindi eval MATCHED (soumya, duration-matched)", "hindi_eval_matched", "eval"),
]


def _clf(name: str) -> str:
    return str(CLASSIFIERS_DIR / f"{name}.joblib")


VARIANTS: List[Dict[str, Any]] = [
    {"key": "a", "label": "wav2vec2 hindi_combined (original, baseline)",
     "single": ("wav2vec2", _clf("wav2vec2_hindi_combined_logreg"))},
    {"key": "b", "label": "wavlm hindi_combined (original, baseline)",
     "single": ("wavlm", _clf("wavlm_hindi_combined_logreg"))},
    {"key": "c", "label": "weighted avg (a)+(b) — CURRENT production",
     "pair": (("wav2vec2", _clf("wav2vec2_hindi_combined_logreg")),
              ("wavlm", _clf("wavlm_hindi_combined_logreg")))},
    {"key": "d", "label": "wav2vec2 hindi_matched (new)",
     "single": ("wav2vec2", _clf("wav2vec2_hindi_matched_logreg"))},
    {"key": "e", "label": "wavlm hindi_matched (new)",
     "single": ("wavlm", _clf("wavlm_hindi_matched_logreg"))},
    {"key": "f", "label": "weighted avg (d)+(e) — CANDIDATE production",
     "pair": (("wav2vec2", _clf("wav2vec2_hindi_matched_logreg")),
              ("wavlm", _clf("wavlm_hindi_matched_logreg")))},
]

INTERPRETATION = [
    "A LOWER Hindi accuracy for (f) than for (c) is the EXPECTED and CORRECT outcome. The",
    "original Hindi number was partly produced by the duration shortcut (duration + RMS alone",
    "predicted the label with 83.3% accuracy on the original corpus). Removing the shortcut",
    "removes the accuracy it was propping up.",
    "",
    "The question this table answers is NOT \"did accuracy go up\". It is: what is the honest",
    "Hindi performance once the shortcut is removed (row f, MATCHED column), and did English",
    "performance hold steady (rows c vs f, ASVspoof2019 column).",
]

SPLIT_WARNING = [
    "The F0 baseline's Hindi numbers were measured on the ORIGINAL eval split. Only compare",
    "numbers WITHIN the same column. A delta between a MATCHED-split number and an",
    "ORIGINAL-split number (e.g. \"(f) matched vs the F0 baseline\") compares two different test",
    "sets and is meaningless — do not quote it.",
    "",
    "- (c) on ORIGINAL reproduces the F0 baseline.",
    "- (c) ORIGINAL -> (c) MATCHED isolates the effect of the test-set change alone (same model).",
    "- (c) MATCHED -> (f) MATCHED is the effect of retraining, on the honest test set.",
    "- (f) on MATCHED is the honest post-fix Hindi number.",
]


def evaluate_variant(variant: Dict[str, Any], dataset: str, split: str) -> Dict[str, Any]:
    if "single" in variant:
        backbone, clf_path = variant["single"]
        return zero_shot_eval_from_cache(
            clf_path, [backbone], dataset, split=split, threshold=THRESHOLD
        )
    (model_a, clf_a), (model_b, clf_b) = variant["pair"]
    return zero_shot_eval_weighted_average_from_cache(
        clf_a, model_a, clf_b, model_b, dataset,
        weight_a=WEIGHT_A, split=split, threshold=THRESHOLD,
    )


def run_all() -> Dict[str, Dict[str, Dict[str, Any]]]:
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for v in VARIANTS:
        results[v["key"]] = {}
        for ts_key, _, dataset, split in TEST_SETS:
            logger.info("Evaluating (%s) on %s", v["key"], ts_key)
            results[v["key"]][ts_key] = evaluate_variant(v, dataset, split)
    return results


def check_baseline_reproduction(results: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Compare variant (c) on ASVspoof / original Hindi against fix_baseline.json."""
    if not BASELINE_JSON.exists():
        return {"available": False, "ok": False, "rows": []}
    with open(BASELINE_JSON, encoding="utf-8") as f:
        base = json.load(f)

    rows = []
    ok = True
    for ts_key, base_key in (("asvspoof", "item2_asvspoof"), ("hindi_original", "item3_hindi")):
        for metric in ("accuracy", "eer"):
            expected = float(base[base_key][metric])
            got = float(results["c"][ts_key][metric])
            match = abs(expected - got) <= REPRO_TOLERANCE
            ok &= match
            rows.append({"test_set": ts_key, "metric": metric,
                         "baseline": expected, "reproduced": got, "match": match})
    return {"available": True, "ok": ok, "rows": rows}


def observed_outcome(results: Dict[str, Dict[str, Dict[str, Any]]]) -> List[str]:
    """What this run actually shows for (c) vs (f), each on the same Hindi split."""
    lines = []
    for ts_key, name in (("hindi_matched", "MATCHED"), ("hindi_original", "ORIGINAL")):
        c_acc = results["c"][ts_key]["accuracy"]
        f_acc = results["f"][ts_key]["accuracy"]
        n = sum(sum(row) for row in results["c"][ts_key]["confusion_matrix"])
        clips = round(abs(f_acc - c_acc) * n)
        if f_acc < c_acc:
            verdict = "LOWER than (c), consistent with the expectation above"
        elif f_acc > c_acc:
            verdict = "HIGHER than (c), NOT the expected direction"
        else:
            verdict = "EQUAL to (c)"
        lines.append(
            f"Observed on {name} split: (f) {_pct(f_acc)} vs (c) {_pct(c_acc)}: {verdict} "
            f"({clips} of {n} clips; each clip is {100 / n:.1f} pp)."
        )
    return lines


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.2f}%"


def _delta_pp(new: float, old: float) -> str:
    return f"{(new - old) * 100:+.2f} pp"


def build_report(
    results: Dict[str, Dict[str, Dict[str, Any]]], repro: Dict[str, Any]
) -> str:
    ts_labels = {k: label for k, label, _, _ in TEST_SETS}
    lines: List[str] = [
        "# Duration-Matched Hindi Heads — Three-Test-Set Comparison",
        "",
        f"**Generated at:** {datetime.now(timezone.utc).isoformat()}",
        "**Script:** `scripts/fix_evaluate_matched.py`",
        f"**Settings:** weighted average weight_a={WEIGHT_A} (wav2vec2) / {1 - WEIGHT_A} (WavLM); "
        f"accuracy at threshold {THRESHOLD}; EER is threshold-free.",
        "",
        "> [!WARNING]",
        "> **Do not compare across Hindi columns.**",
    ]
    lines += [f"> {s}" if s else ">" for s in SPLIT_WARNING]
    lines += [
        "",
        "## Results",
        "",
        "Columns name the test set each number was measured on.",
        "",
        "| Variant | ASVspoof2019 eval — Acc | ASVspoof2019 eval — EER "
        "| Hindi ORIGINAL eval (soumya) — Acc | Hindi ORIGINAL eval (soumya) — EER "
        "| Hindi MATCHED eval (soumya) — Acc | Hindi MATCHED eval (soumya) — EER |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for v in VARIANTS:
        r = results[v["key"]]
        cells = []
        for ts_key, _, _, _ in TEST_SETS:
            cells += [_pct(r[ts_key]["accuracy"]), _pct(r[ts_key]["eer"])]
        bold = v["key"] in ("c", "f")
        label = f"**({v['key']}) {v['label']}**" if bold else f"({v['key']}) {v['label']}"
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    c, f_ = results["c"], results["f"]
    lines += [
        "",
        "## Valid same-test-set deltas",
        "",
        "| Comparison | Test set | Δ Accuracy | Δ EER | What it measures |",
        "|---|---|:---:|:---:|---|",
        f"| (c) → (f) | {ts_labels['asvspoof']} | {_delta_pp(f_['asvspoof']['accuracy'], c['asvspoof']['accuracy'])} "
        f"| {_delta_pp(f_['asvspoof']['eer'], c['asvspoof']['eer'])} | English regression check |",
        f"| (c) → (f) | {ts_labels['hindi_matched']} | {_delta_pp(f_['hindi_matched']['accuracy'], c['hindi_matched']['accuracy'])} "
        f"| {_delta_pp(f_['hindi_matched']['eer'], c['hindi_matched']['eer'])} | Effect of retraining, honest test set |",
        f"| (c) → (f) | {ts_labels['hindi_original']} | {_delta_pp(f_['hindi_original']['accuracy'], c['hindi_original']['accuracy'])} "
        f"| {_delta_pp(f_['hindi_original']['eer'], c['hindi_original']['eer'])} | Effect of retraining, shortcut-bearing test set |",
        f"| (c) ORIGINAL → (c) MATCHED | same model, two test sets | "
        f"{_delta_pp(c['hindi_matched']['accuracy'], c['hindi_original']['accuracy'])} "
        f"| {_delta_pp(c['hindi_matched']['eer'], c['hindi_original']['eer'])} "
        f"| Test-set change alone (how much the production model leaned on duration) |",
        "",
        "## F0 baseline reproduction check",
        "",
    ]
    if not repro["available"]:
        lines.append(f"`{BASELINE_JSON.name}` not found — reproduction not checked.")
    else:
        status = "REPRODUCED" if repro["ok"] else "NOT REPRODUCED — investigate before using this table"
        lines += [
            f"Variant (c) vs `{BASELINE_JSON.name}`: **{status}** (tolerance {REPRO_TOLERANCE:g}).",
            "",
            "| Test set | Metric | F0 baseline | Variant (c) this run | Match |",
            "|---|---|:---:|:---:|:---:|",
        ]
        for row in repro["rows"]:
            lines.append(
                f"| {ts_labels[row['test_set']]} | {row['metric']} | {_pct(row['baseline'])} "
                f"| {_pct(row['reproduced'])} | {'yes' if row['match'] else '**NO**'} |"
            )
    lines += ["", "## Interpretation", ""]
    lines += [f"> {s}" if s else ">" for s in INTERPRETATION]
    lines += ["", "**This run:**", ""]
    lines += [f"- {s}" for s in observed_outcome(results)]
    lines.append("")
    return "\n".join(lines)


def print_summary(results: Dict[str, Dict[str, Dict[str, Any]]], repro: Dict[str, Any], out: Path) -> None:
    bar = "=" * 96
    print("\n" + bar)
    print(" MATCHED vs ORIGINAL HINDI HEADS - Acc / EER per test set")
    print(bar)
    print(f" {'Variant':<48} {'ASVspoof2019 eval':>15} {'Hindi ORIGINAL':>15} {'Hindi MATCHED':>15}")
    print("-" * 96)
    for v in VARIANTS:
        r = results[v["key"]]
        label = f"({v['key']}) {v['label']}".replace("—", "-")[:48]
        cells = [f"{_pct(r[k]['accuracy'])}/{_pct(r[k]['eer'])}" for k, _, _, _ in TEST_SETS]
        print(f" {label:<48} " + " ".join(f"{c:>15}" for c in cells))
    print("-" * 96)
    if repro["available"]:
        print(f" F0 baseline reproduced by (c): {'YES' if repro['ok'] else 'NO - investigate'}")
    print(bar)
    print(" DO NOT COMPARE ACROSS HINDI COLUMNS:")
    for s in SPLIT_WARNING:
        print(f"   {s.replace('—', '-')}")
    print(bar)
    print(" INTERPRETATION:")
    for s in INTERPRETATION:
        print(f"   {s}")
    print()
    for s in observed_outcome(results):
        print(f"   {s}")
    print(bar)
    print(f" Report written to: {out}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output_md", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    try:
        results = run_all()
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc)
        sys.exit(1)

    repro = check_baseline_reproduction(results)
    report = build_report(results, repro)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")
    logger.info("Saved comparison report to %s", args.output_md)

    print_summary(results, repro, args.output_md)
    if repro["available"] and not repro["ok"]:
        logger.warning("Variant (c) did not reproduce the F0 baseline; see report.")


if __name__ == "__main__":
    main()
