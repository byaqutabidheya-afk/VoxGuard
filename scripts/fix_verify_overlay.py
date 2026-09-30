#!/usr/bin/env python3
"""
scripts/fix_verify_overlay.py - Re-verify the explainability overlay's real-vs-synthetic separation (Phase F4.3).

Runs ``windowed_attribution`` with ALL DEFAULTS (no detector, no window, no stride passed), so what is
verified is exactly what the app gets: the chunk-native heads in ``config.PRODUCTION_STREAMING_CLASSIFIERS``
(weight ``config.PRODUCTION_ENSEMBLE_WEIGHT_A``), a window of ``config.STREAM_CHUNK_SECONDS`` and the
default 0.75 s stride. The resolved values are printed and asserted so a silent drift cannot pass.

Clips: the duration-MATCHED versions of the 5 verified Phase 6 demo pairs (the corpus the chunked heads were
trained against). Reported per clip as the nanmean of its window scores.

Block 1 - the 5 demo pairs. Each pair is labelled HELD-OUT (soumya) or TRAINING-SEEN (byaquta, mahato: the
chunked heads were trained on those speakers, so those pairs are not held-out evidence). Checks:
  (a) GLOBAL: every real clip's nanmean is below every synthetic clip's nanmean (max real < min synthetic).
  (b) PER PAIR: real < synthetic within each pair.
  (c) The held-out soumya pair, stated on its own.
Block 2 (supplementary, NOT part of the pass rule) - all 25 real + 25 synthetic soumya matched eval clips: the
only held-out speaker, so the only place a claim about generalisation can be tested at more than one pair.

The script does not edit the UI caption; it prints what the evidence supports.
Writes models/reports/fix_overlay_separation.md (+ .json) and renders each demo clip's overlay with the
defaults into models/reports/fix_overlay_check/ as an end-to-end check that the render path works. ASCII-only output.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from voxguard import config
from voxguard.explain.attribution import DEFAULT_STRIDE_SECONDS, get_default_detector, windowed_attribution
from voxguard.explain.overlay import render_explainability_overlay
from voxguard.utils.audio_io import load_audio
from voxguard.utils.hindi_splits import get_hindi_hinglish_splits
from voxguard.utils.logging_utils import get_logger

logger = get_logger("fix_verify_overlay")

MATCHED_DIR = config.DATA_RAW_DIR / "hindi_hinglish_matched"
MATCHED_TRACK_CSV = config.DATA_METADATA_DIR / "hindi_hinglish_track_matched.csv"
REPORT_PATH = config.MODELS_DIR / "reports" / "fix_overlay_separation.md"
OVERLAY_DIR = config.MODELS_DIR / "reports" / "fix_overlay_check"
HELD_OUT_SPEAKER = "soumya"

DEMO_PAIRS: List[Tuple[str, str, str]] = [
    ("Pair 1: Casual Neutral (byaquta)", "byaquta_neutral_09", "byaquta_neutral_09_clone"),
    ("Pair 2: Everyday Tech (mahato)", "mahato_neutral_04", "mahato_neutral_04_clone"),
    ("Pair 3: Urgent Legal Scam (byaquta)", "byaquta_scam_16", "byaquta_scam_16_clone"),
    ("Pair 4: Authority Customs Scam (mahato)", "mahato_scam_12", "mahato_scam_12_clone"),
    ("Pair 5: Held-Out Casual (soumya)", "soumya_neutral_03", "soumya_neutral_03_clone"),
]


def score_clip(path: Path) -> Dict[str, Any]:
    wav, sr = load_audio(path, target_sr=config.SAMPLE_RATE)
    scores, times = windowed_attribution(wav, sr)          # ALL defaults, on purpose
    valid = scores[~np.isnan(scores)]
    if valid.size == 0:
        raise RuntimeError(f"{path.name}: no window was scored; a mean would be vacuous.")
    return {
        "file": path.name, "duration": float(len(wav) / sr), "n_windows": int(len(scores)), "n_scored": int(valid.size),
        "n_nan": int(np.isnan(scores).sum()), "nanmean": float(np.nanmean(scores)),
        "min": float(valid.min()), "max": float(valid.max()),
    }


def auc(real: List[float], synth: List[float]) -> float:
    """P(random synthetic mean > random real mean); ties count half."""
    wins = sum((s > r) + 0.5 * (s == r) for r in real for s in synth)
    return wins / (len(real) * len(synth))


def resolve_defaults() -> Dict[str, Any]:
    det = get_default_detector()
    wired = (Path(det.detector_a.classifier_path).resolve(), Path(det.detector_b.classifier_path).resolve())
    want = tuple((config.BASE_DIR / config.PRODUCTION_STREAMING_CLASSIFIERS[b]).resolve() for b in ("wav2vec2", "wavlm"))
    if wired != want:
        raise RuntimeError(f"Default detector is not wired to PRODUCTION_STREAMING_CLASSIFIERS: {wired}")
    if det.weight_a != config.PRODUCTION_ENSEMBLE_WEIGHT_A:
        raise RuntimeError(f"Default detector weight {det.weight_a} != PRODUCTION_ENSEMBLE_WEIGHT_A.")
    return {"heads": [p.name for p in wired], "weight_a": det.weight_a, "window_seconds": float(config.STREAM_CHUNK_SECONDS),
            "stride_seconds": DEFAULT_STRIDE_SECONDS}


def block1() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for group, real, synth in DEMO_PAIRS:
        for kind, stem, folder in (("REAL", real, "real"), ("SYNTH", synth, "synthetic")):
            path = MATCHED_DIR / folder / f"{stem}.wav"
            if not path.exists():
                raise FileNotFoundError(path)
            speaker = stem.split("_")[0]
            rows.append({"group": group, "kind": kind, "speaker": speaker, "held_out": speaker == HELD_OUT_SPEAKER,
                         "path": str(path), **score_clip(path)})
    real = [r for r in rows if r["kind"] == "REAL"]
    synth = [r for r in rows if r["kind"] == "SYNTH"]
    max_real = max(real, key=lambda r: r["nanmean"])
    min_synth = min(synth, key=lambda r: r["nanmean"])
    pairs = []
    for group, _, _ in DEMO_PAIRS:
        r = next(x for x in real if x["group"] == group)
        s = next(x for x in synth if x["group"] == group)
        pairs.append({"group": group, "held_out": r["held_out"], "speaker": r["speaker"], "real": r["nanmean"],
                      "synth": s["nanmean"], "gap": s["nanmean"] - r["nanmean"], "separates": r["nanmean"] < s["nanmean"]})
    return {
        "rows": rows, "pairs": pairs,
        "global_holds": max_real["nanmean"] < min_synth["nanmean"],
        "max_real": {"file": max_real["file"], "nanmean": max_real["nanmean"]},
        "min_synth": {"file": min_synth["file"], "nanmean": min_synth["nanmean"]},
        "global_gap": min_synth["nanmean"] - max_real["nanmean"],
        "pairs_separating": sum(p["separates"] for p in pairs), "n_pairs": len(pairs),
        "heldout_pair_separates": next(p for p in pairs if p["held_out"])["separates"],
    }


def block2() -> Dict[str, Any]:
    df = pd.read_csv(MATCHED_TRACK_CSV)
    _, ev = get_hindi_hinglish_splits(df, mode="speaker_holdout", holdout_speaker=HELD_OUT_SPEAKER)
    rows = []
    for _, r in ev.sort_values(["label", "filepath"]).iterrows():
        rows.append({"kind": "REAL" if r["label"] == "real" else "SYNTH", **score_clip(config.BASE_DIR / r["filepath"])})
    real = [r for r in rows if r["kind"] == "REAL"]
    synth = [r for r in rows if r["kind"] == "SYNTH"]
    rv, sv = [r["nanmean"] for r in real], [r["nanmean"] for r in synth]
    by_stem = {Path(r["file"]).stem: r for r in rows}
    utter = []
    for r in real:
        c = by_stem.get(Path(r["file"]).stem + "_clone")
        if c is not None:
            utter.append({"real": r["file"], "real_mean": r["nanmean"], "synth_mean": c["nanmean"], "separates": r["nanmean"] < c["nanmean"]})
    overlapping_real = sorted((r for r in real if r["nanmean"] >= min(sv)), key=lambda r: -r["nanmean"])
    overlapping_synth = sorted((r for r in synth if r["nanmean"] <= max(rv)), key=lambda r: r["nanmean"])
    return {
        "n_real": len(real), "n_synth": len(synth), "global_holds": max(rv) < min(sv),
        "max_real": {"file": max(real, key=lambda r: r["nanmean"])["file"], "nanmean": max(rv)},
        "min_synth": {"file": min(synth, key=lambda r: r["nanmean"])["file"], "nanmean": min(sv)},
        "auc": auc(rv, sv), "median_real": float(np.median(rv)), "median_synth": float(np.median(sv)),
        "pairs_separating": sum(u["separates"] for u in utter), "n_pairs": len(utter),
        "misordered_pairs": [u for u in utter if not u["separates"]],
        "real_over_min_synth": [(r["file"], r["nanmean"]) for r in overlapping_real],
        "synth_under_max_real": [(r["file"], r["nanmean"]) for r in overlapping_synth],
        "rows": rows,
    }


def render_demo_overlays(rows: List[Dict[str, Any]]) -> List[str]:
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for r in rows:
        wav, sr = load_audio(r["path"], target_sr=config.SAMPLE_RATE)
        out.append(render_explainability_overlay(wav, sr, output_path=OVERLAY_DIR / f"{Path(r['file']).stem}_overlay.png"))
    return out


def verdict_text(b1: Dict[str, Any], b2: Dict[str, Any]) -> List[str]:
    lines = []
    if b1["global_holds"]:
        lines.append(f"Global separation HOLDS across the {b1['n_pairs']} demo pairs: highest real mean {b1['max_real']['nanmean']:.3f} "
                     f"(`{b1['max_real']['file']}`) < lowest synthetic mean {b1['min_synth']['nanmean']:.3f} (`{b1['min_synth']['file']}`), "
                     f"gap {b1['global_gap']:+.3f}.")
    else:
        lines.append(f"Global separation does NOT hold across the {b1['n_pairs']} demo pairs: highest real mean {b1['max_real']['nanmean']:.3f} "
                     f"(`{b1['max_real']['file']}`) is not below the lowest synthetic mean {b1['min_synth']['nanmean']:.3f} "
                     f"(`{b1['min_synth']['file']}`), gap {b1['global_gap']:+.3f}.")
    n_held = sum(p["held_out"] for p in b1["pairs"])
    lines.append(f"Pairs separating within the pair: {b1['pairs_separating']}/{b1['n_pairs']}. Only {n_held} of the {b1['n_pairs']} pairs is held-out "
                 f"(soumya); the other {b1['n_pairs'] - n_held} are speakers the chunked heads were trained on.")
    hp = next(p for p in b1["pairs"] if p["held_out"])
    lines.append(f"HELD-OUT soumya pair ({hp['group']}): real {hp['real']:.3f} vs synthetic {hp['synth']:.3f} -> "
                 f"{'separates' if hp['separates'] else 'DOES NOT separate'} (gap {hp['gap']:+.3f}).")
    lines.append(f"Supplementary, all soumya matched eval clips ({b2['n_real']} real + {b2['n_synth']} synthetic): global separation "
                 f"{'holds' if b2['global_holds'] else 'DOES NOT hold'} (highest real {b2['max_real']['nanmean']:.3f} `{b2['max_real']['file']}`, "
                 f"lowest synthetic {b2['min_synth']['nanmean']:.3f} `{b2['min_synth']['file']}`); clip-mean AUC {b2['auc']:.3f}; "
                 f"real-vs-own-clone ordering correct in {b2['pairs_separating']}/{b2['n_pairs']} utterances.")
    return lines


def build_markdown(ctx: Dict[str, Any]) -> str:
    d, b1, b2 = ctx["defaults"], ctx["b1"], ctx["b2"]
    L = [
        "# Explainability Overlay - Separation Re-verification (Phase F4.3)",
        "",
        f"**Generated at:** {datetime.now(timezone.utc).isoformat()}",
        "**Script:** `scripts/fix_verify_overlay.py`",
        "**Call under test:** `windowed_attribution(waveform, sr)` with ALL defaults.",
        f"**Resolved defaults:** heads {', '.join('`' + h + '`' for h in d['heads'])} (weight_a={d['weight_a']}); "
        f"window {d['window_seconds']}s (= `config.STREAM_CHUNK_SECONDS`); stride {d['stride_seconds']}s.",
        "**Audio:** duration-MATCHED clips only. Score per clip = nanmean of its window scores (NaN = silent window, excluded).",
        "",
        "## Verdict",
        "",
    ] + [f"- {s}" for s in ctx["verdict"]] + ["",
        "## Block 1 - the 5 verified demo pairs", "",
        "| Pair | Speaker | Evidence status | Real nanmean | Synthetic nanmean | Gap | Real < synthetic |", "|---|:---:|:---:|:---:|:---:|:---:|:---:|"]
    for p in b1["pairs"]:
        L.append(f"| {p['group']} | {p['speaker']} | {'HELD-OUT' if p['held_out'] else 'training-seen'} | {p['real']:.3f} | {p['synth']:.3f} | "
                 f"{p['gap']:+.3f} | {'yes' if p['separates'] else '**NO**'} |")
    L += ["", "### Per clip", "", "| Kind | File | Speaker | Status | Duration | Windows (scored / NaN) | nanmean | min | max |", "|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"]
    for r in b1["rows"]:
        L.append(f"| {r['kind']} | `{r['file']}` | {r['speaker']} | {'HELD-OUT' if r['held_out'] else 'training-seen'} | {r['duration']:.1f}s | "
                 f"{r['n_scored']} / {r['n_nan']} | {r['nanmean']:.3f} | {r['min']:.3f} | {r['max']:.3f} |")
    L += ["", f"Highest real mean: `{b1['max_real']['file']}` {b1['max_real']['nanmean']:.3f}. Lowest synthetic mean: "
          f"`{b1['min_synth']['file']}` {b1['min_synth']['nanmean']:.3f}.", "",
          "## Block 2 (supplementary, not part of the pass rule) - all soumya matched eval clips, the only held-out speaker", "",
          f"{b2['n_real']} real + {b2['n_synth']} synthetic clips. Median nanmean: real {b2['median_real']:.3f}, synthetic {b2['median_synth']:.3f}. "
          f"Clip-mean AUC {b2['auc']:.3f}. Real-vs-own-clone ordering correct in {b2['pairs_separating']}/{b2['n_pairs']} utterances.", "",
          f"Real clips whose mean is at/above the LOWEST synthetic mean ({b2['min_synth']['nanmean']:.3f}): "
          + (", ".join(f"`{f}` {m:.3f}" for f, m in b2["real_over_min_synth"]) or "none") + ".", "",
          f"Synthetic clips whose mean is at/below the HIGHEST real mean ({b2['max_real']['nanmean']:.3f}): "
          + (", ".join(f"`{f}` {m:.3f}" for f, m in b2["synth_under_max_real"]) or "none") + ".", ""]
    if b2["misordered_pairs"]:
        L += ["Utterances where the real clip scores at/above its own clone: "
              + ", ".join(f"`{u['real']}` ({u['real_mean']:.3f} vs {u['synth_mean']:.3f})" for u in b2["misordered_pairs"]) + ".", ""]
    L += ["## Overlay render check", "", f"Overlays rendered with the defaults for all {len(ctx['overlays'])} demo clips into `{OVERLAY_DIR.relative_to(config.BASE_DIR).as_posix()}/`.", ""]
    return "\n".join(L)


def print_console(ctx: Dict[str, Any]) -> None:
    d, b1, b2 = ctx["defaults"], ctx["b1"], ctx["b2"]
    bar = "=" * 104
    print("\n" + bar)
    print(" OVERLAY SEPARATION RE-VERIFICATION - windowed_attribution with ALL defaults, MATCHED audio")
    print(bar)
    print(f" heads: {', '.join(d['heads'])}  weight_a={d['weight_a']}   window={d['window_seconds']}s (config.STREAM_CHUNK_SECONDS)   stride={d['stride_seconds']}s")
    print("-" * 104)
    print(f" {'Kind':<6}{'File':<30}{'Speaker':<10}{'Status':<15}{'Win ok/NaN':<12}{'nanmean':>9}{'min':>8}{'max':>8}")
    for r in b1["rows"]:
        print(f" {r['kind']:<6}{r['file']:<30}{r['speaker']:<10}{'HELD-OUT' if r['held_out'] else 'training-seen':<15}"
              f"{str(r['n_scored']) + '/' + str(r['n_nan']):<12}{r['nanmean']:>9.3f}{r['min']:>8.3f}{r['max']:>8.3f}")
    print("-" * 104)
    print(f" {'Pair':<42}{'Status':<15}{'real':>8}{'synth':>8}{'gap':>9}  separates")
    for p in b1["pairs"]:
        print(f" {p['group']:<42}{'HELD-OUT' if p['held_out'] else 'training-seen':<15}{p['real']:>8.3f}{p['synth']:>8.3f}{p['gap']:>+9.3f}  {'yes' if p['separates'] else 'NO'}")
    print("-" * 104)
    for s in ctx["verdict"]:
        print(" - " + s.replace("`", ""))
    print(bar)
    print(f" Report written to: {REPORT_PATH}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output_md", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    try:
        defaults = resolve_defaults()
        b1 = block1()
        b2 = block2()
        overlays = render_demo_overlays(b1["rows"])
    except Exception as exc:
        logger.error("Overlay verification aborted: %s", exc)
        sys.exit(2)
    ctx = {"defaults": defaults, "b1": b1, "b2": b2, "overlays": overlays, "verdict": verdict_text(b1, b2)}
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_markdown(ctx), encoding="utf-8")
    args.output_md.with_suffix(".json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(), "defaults": defaults,
        "block1": {k: v for k, v in b1.items()}, "block2": {k: v for k, v in b2.items() if k != "rows"}, "verdict": ctx["verdict"],
    }, indent=2), encoding="utf-8")
    print_console(ctx)


if __name__ == "__main__":
    main()
