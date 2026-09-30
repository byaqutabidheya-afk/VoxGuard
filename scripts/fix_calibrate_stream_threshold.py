#!/usr/bin/env python3
"""
scripts/fix_calibrate_stream_threshold.py - Calibrate the streaming flag rule (Phase F4).

Sweeps STREAM_FLAG_THRESHOLD in [0.4, 0.5, 0.6, 0.7, 0.8] crossed with
consecutive_flags_required in [1, 2, 3, 4]. Every combination is evaluated by replaying a held-out
clip set through the REAL ``StreamingSession`` / ``simulate_stream`` path (real buffer, real EMA,
real flag logic), wired to the production STREAMING classifiers
(``config.PRODUCTION_STREAMING_CLASSIFIERS`` = the chunked wav2vec2 head + the v2 chunked WavLM head,
weight ``config.PRODUCTION_ENSEMBLE_WEIGHT_A``).

Clip sets (duration-MATCHED audio throughout; matched and unmatched audio are never mixed)
------------------------------------------------------------------------------------------
* SELECTION block: every Hindi MATCHED eval clip - soumya, the held-out speaker (25 real + 25
  synthetic). The threshold is selected on this block ALONE.
* DEMO block: the matched versions of the 5 verified Phase 6 demo pairs, for continuity with earlier
  reporting. Four of the five pairs are byaquta / mahato clips, speakers the chunked heads were
  TRAINED on, so this block is NOT held-out evidence. It is reported separately and never used to
  select. (Pair 5 is soumya, so its two clips also appear in the selection block.)

Selection rule (also printed in the output)
-------------------------------------------
Choose the combination with the LOWEST false-positive rate (real clips that flag); break ties by
HIGHEST detection rate (synthetic clips that flag); break remaining ties by LOWEST median
seconds_to_flag among detected synthetic clips. A false positive on genuine human speech is the
failure this fix guide exists to eliminate, so it outranks latency. If combinations are still tied
after all three keys, the first in grid order (lowest threshold, then lowest count) is taken, and
the number of exactly-tied combinations is printed.

How the sweep is run without 20x the model inference
----------------------------------------------------
The threshold and the consecutive count only change the FLAGGING logic, never the chunk scores.
Each unique audio window is therefore sent through the real detector once and its score cached
(``CachedScorer``, keyed by a hash of the window's samples); every one of the 20 combinations still
runs a fresh real ``StreamingSession`` over every clip, using the cached score for windows it has
seen. A cross-check re-runs a few clips at the current config settings with an UNCACHED
``StreamingScorer`` and aborts if the outcome differs.

What ``consecutive_flags_required`` counts (revised)
----------------------------------------------------
The first version of this sweep counted consecutive ``push_audio`` CALLS (the session's original
behaviour). ``simulate_stream`` pushes 0.25 s steps but the running score only changes when a window
completes, once per 1.0 s stride (1.5 s window, 0.5 s overlap), so counts 1..4 all resolved inside one
score update and tested no persistence. The sweep now uses ``StreamingSession(consecutive_unit="updates")``:
the count is of consecutive per-stride SCORE UPDATES (scored windows), so N means N independent
decisions in a row and the earliest possible flag is 1.5 s + (N-1) * 1.0 s. The real push/update ratio is
MEASURED from a live StreamingSession at run time (not assumed) and printed. The old push-step grid is
still computed, cheaply, and shown as a labelled comparison so the change can be seen. A separate section
reports whether a real debounce (N >= 2) removes the residual false positives at threshold 0.8, and at
what latency cost.

Output: models/reports/fix_calibrate_stream.md (+ .json). config.py is written ONLY after a typed
confirmation (STREAM_FLAG_THRESHOLD, and a new STREAM_CONSECUTIVE_FLAGS_REQUIRED). Call sites are
not touched here. ASCII-only console output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from voxguard import config
from voxguard.classifier.ensemble import WeightedAverageDetector
from voxguard.streaming.scorer import StreamingScorer
from voxguard.streaming.session import StreamingSession, simulate_stream
from voxguard.utils.audio_io import load_audio
from voxguard.utils.hindi_splits import get_hindi_hinglish_splits
from voxguard.utils.logging_utils import get_logger

logger = get_logger("fix_calibrate_stream_threshold")

MATCHED_TRACK_CSV = config.DATA_METADATA_DIR / "hindi_hinglish_track_matched.csv"
MATCHED_DIR = config.DATA_RAW_DIR / "hindi_hinglish_matched"
REPORT_PATH = config.MODELS_DIR / "reports" / "fix_calibrate_stream.md"
CONFIG_PATH = config.BASE_DIR / "src" / "voxguard" / "config.py"

HELD_OUT_SPEAKER = "soumya"
THRESHOLD_GRID = [0.4, 0.5, 0.6, 0.7, 0.8]
CONSECUTIVE_GRID = [1, 2, 3, 4]
STEP_SECONDS = 0.25          # simulate_stream default push size
COUNT_UNIT = "updates"       # the primary sweep: consecutive per-stride score updates
LEGACY_UNIT = "pushes"       # the original counting, kept only as a labelled comparison
DEBOUNCE_THRESHOLD = 0.8     # threshold at which the residual false positives are analysed
DEBOUNCE_N = [1, 2, 3, 4, 5, 6]   # analysis only; the SELECTION grid stays CONSECUTIVE_GRID
CROSSCHECK = (0.6, 3)        # the pre-fix session defaults; the cross-check compares cached vs uncached here
N_CROSSCHECK_CLIPS = 4

# The 5 verified Phase 6 demo pairs (MATCHED audio): (group, real stem, synthetic stem).
DEMO_PAIRS: List[Tuple[str, str, str]] = [
    ("Pair 1: Casual Neutral (byaquta)", "byaquta_neutral_09", "byaquta_neutral_09_clone"),
    ("Pair 2: Everyday Tech (mahato)", "mahato_neutral_04", "mahato_neutral_04_clone"),
    ("Pair 3: Urgent Legal Scam (byaquta)", "byaquta_scam_16", "byaquta_scam_16_clone"),
    ("Pair 4: Authority Customs Scam (mahato)", "mahato_scam_12", "mahato_scam_12_clone"),
    ("Pair 5: Held-Out Casual (soumya)", "soumya_neutral_03", "soumya_neutral_03_clone"),
]

SELECTION_RULE = [
    "1. Lowest false-positive rate (fraction of soumya's REAL clips that flag).",
    "2. Ties: highest detection rate (fraction of soumya's SYNTHETIC clips that flag).",
    "3. Remaining ties: lowest median seconds_to_flag among the detected synthetic clips.",
    "4. Still tied (not in the spec, needed for determinism): first in grid order (lowest threshold, "
    "then lowest count); the number of exactly-tied combinations is printed.",
    "A false positive on genuine human speech is the failure this fix guide exists to eliminate, so it "
    "outranks detection and latency. Selection uses soumya's clips ONLY.",
]

BLOCK_BEGIN = "# --- F4 stream calibration (written by scripts/fix_calibrate_stream_threshold.py) ---"
BLOCK_END = "# --- end F4 stream calibration ---"


class ReplayError(RuntimeError):
    """The replay cannot be trusted."""


# =============================================================================
# Clips
# =============================================================================

def _clip(block: str, group: str, kind: str, path: Path) -> Dict[str, Any]:
    speaker = path.stem.split("_")[0]
    return {"block": block, "group": group, "kind": kind, "file": path.name, "path": path,
            "speaker": speaker, "held_out": speaker == HELD_OUT_SPEAKER}


def resolve_clips() -> List[Dict[str, Any]]:
    df = pd.read_csv(MATCHED_TRACK_CSV)
    _, eval_df = get_hindi_hinglish_splits(df, mode="speaker_holdout", holdout_speaker=HELD_OUT_SPEAKER)
    if set(eval_df["speaker_id"]) != {HELD_OUT_SPEAKER}:
        raise ReplayError(f"Hindi matched eval is not {HELD_OUT_SPEAKER} only: {sorted(set(eval_df['speaker_id']))}")

    clips: List[Dict[str, Any]] = []
    for _, row in eval_df.sort_values(["label", "filepath"]).iterrows():
        kind = "REAL" if row["label"] == "real" else "SYNTH"
        clips.append(_clip("selection", "soumya matched eval", kind, config.BASE_DIR / row["filepath"]))
    for group, real, synth in DEMO_PAIRS:
        clips.append(_clip("demo", group, "REAL", MATCHED_DIR / "real" / f"{real}.wav"))
        clips.append(_clip("demo", group, "SYNTH", MATCHED_DIR / "synthetic" / f"{synth}.wav"))

    missing = [str(c["path"]) for c in clips if not c["path"].exists()]
    if missing:
        raise ReplayError("Clips not found:\n  " + "\n  ".join(missing))
    bad = [c["file"] for c in clips if "hindi_hinglish_matched" not in str(c["path"]).replace("\\", "/")]
    if bad:
        raise ReplayError(f"Non-MATCHED audio in the clip set: {bad}")
    sel = [c for c in clips if c["block"] == "selection"]
    if any(not c["held_out"] for c in sel):
        raise ReplayError("The selection block contains a non-held-out speaker.")
    n_real = sum(c["kind"] == "REAL" for c in sel)
    n_synth = len(sel) - n_real
    if (n_real, n_synth) != (25, 25):
        raise ReplayError(f"Expected 25 real + 25 synthetic soumya eval clips, got {n_real} + {n_synth}.")
    return clips


# =============================================================================
# Detector and scorers
# =============================================================================

def build_detector() -> WeightedAverageDetector:
    """Detector wired to config.PRODUCTION_STREAMING_CLASSIFIERS (verified after construction)."""
    paths = {b: config.BASE_DIR / p for b, p in config.PRODUCTION_STREAMING_CLASSIFIERS.items()}
    for p in paths.values():
        if not p.exists():
            raise ReplayError(f"Streaming classifier not found: {p}")
    detector = WeightedAverageDetector(
        wav2vec2_classifier_path=paths["wav2vec2"],
        wavlm_classifier_path=paths["wavlm"],
        weight_a=config.PRODUCTION_ENSEMBLE_WEIGHT_A,
    )
    wired = (detector.detector_a.classifier_path.resolve(), detector.detector_b.classifier_path.resolve())
    if wired != (paths["wav2vec2"].resolve(), paths["wavlm"].resolve()):
        raise ReplayError(f"Detector is not wired to the production streaming heads: {wired}")
    return detector


class CachedScorer(StreamingScorer):
    """StreamingScorer that scores each UNIQUE window once and reuses the score.

    Returns exactly what ``StreamingScorer.score_chunk`` would (silent windows -> None, scores are the
    detector's), so the session sees no difference. Also refuses to hide a detector failure: a window
    that is loud enough to score but returned None means the detector raised and ``score_chunk``
    swallowed it, which would make every result vacuous.
    """

    def __init__(self, detector: Any) -> None:
        super().__init__(detector)
        self._cache: Dict[str, Optional[float]] = {}
        self.n_calls = 0
        self.n_detector_calls = 0

    def score_chunk(self, chunk, sr):
        self.n_calls += 1
        key = f"{sr}:{hashlib.sha1(np.ascontiguousarray(chunk, dtype=np.float32).tobytes()).hexdigest()}"
        if key not in self._cache:
            self.n_detector_calls += 1
            score = super().score_chunk(chunk, sr)
            if score is None and self._rms_energy(chunk) >= self.silence_threshold:
                raise ReplayError("An audible window failed to score (the detector raised); results would be meaningless.")
            self._cache[key] = score
        return self._cache[key]


class CountingScorer(StreamingScorer):
    """Uncached scorer used only by the cross-check (same failure guard as CachedScorer)."""

    def score_chunk(self, chunk, sr):
        score = super().score_chunk(chunk, sr)
        if score is None and self._rms_energy(chunk) >= self.silence_threshold:
            raise ReplayError("An audible window failed to score in the cross-check.")
        return score


def replay(clip: Dict[str, Any], waveform: np.ndarray, sr: int, detector: Any, scorer: StreamingScorer,
           threshold: float, consecutive: int, unit: str = COUNT_UNIT) -> Dict[str, Any]:
    """One fresh REAL StreamingSession over one clip, via the real simulate_stream."""
    session = StreamingSession(
        detector=detector,
        chunk_seconds=config.STREAM_CHUNK_SECONDS,
        overlap_seconds=config.STREAM_OVERLAP_SECONDS,
        flag_threshold=threshold,
        consecutive_flags_required=consecutive,
        consecutive_unit=unit,
    )
    session.scorer = scorer
    update_scores: List[float] = []
    _orig_update = session.risk_score.update

    def _recording_update(new_score):
        current = _orig_update(new_score)
        if new_score is not None:
            update_scores.append(float(current))
        return current

    session.risk_score.update = _recording_update
    sim = simulate_stream(waveform, session=session, real_time_paced=False, step_seconds=STEP_SECONDS, sr=sr)
    return {
        "update_scores": update_scores,
        "flagged": bool(sim["flagged"]),
        "seconds_to_flag": sim["seconds_to_flag"],
        "max_score": float(max(r["running_score"] for r in sim["step_results"])),
        "final_score": float(sim["final_running_score"]),
        "duration": float(sim["total_duration"]),
    }


def run_sweep_grid_extension(clips: List[Dict[str, Any]], detector: Any, scorer: CachedScorer,
                             audio: Dict[str, Tuple[np.ndarray, int]]) -> Dict[Tuple[float, int], List[Dict[str, Any]]]:
    """Counts beyond CONSECUTIVE_GRID, at DEBOUNCE_THRESHOLD only: analysis of the debounce, never used for selection."""
    out: Dict[Tuple[float, int], List[Dict[str, Any]]] = {}
    for n in DEBOUNCE_N:
        if n in CONSECUTIVE_GRID:
            continue
        out[(DEBOUNCE_THRESHOLD, n)] = [
            {**{k: c[k] for k in ("block", "group", "kind", "file", "speaker", "held_out")},
             **replay(c, *audio[str(c["path"])], detector, scorer, DEBOUNCE_THRESHOLD, n, COUNT_UNIT)}
            for c in clips]
    return out


def measure_cadence(stub_score: float = 0.5) -> Dict[str, Any]:
    """MEASURES the push -> score-update cadence from a live StreamingSession (nothing assumed).

    Feeds audio through a real session with a stub detector and records the audio time at which each
    window is scored.
    """
    class _Stub:
        def predict_waveform(self, waveform, sr):
            return {"probability_synthetic": stub_score, "label": "synthetic"}

    sr = config.SAMPLE_RATE
    session = StreamingSession(detector=_Stub(), chunk_seconds=config.STREAM_CHUNK_SECONDS,
                               overlap_seconds=config.STREAM_OVERLAP_SECONDS)
    times: List[float] = []
    original = session.scorer.score_chunk

    def spy(chunk, sr_):
        times.append(session._audio_seconds_elapsed)
        return original(chunk, sr_)

    session.scorer.score_chunk = spy
    wav = np.full(int(12 * sr), 0.1, dtype=np.float32)
    step = int(round(STEP_SECONDS * sr))
    for off in range(0, len(wav), step):
        session.push_audio(wav[off: off + step], sr)
    gaps = sorted({round(b - a, 6) for a, b in zip(times, times[1:])})
    if len(times) < 3 or len(gaps) != 1:
        raise ReplayError(f"Score-update cadence is not regular: update times {times}")
    return {"first_update_s": times[0], "update_gap_s": gaps[0],
            "pushes_per_update": round(gaps[0] / STEP_SECONDS, 6),
            "first_update_push": round(times[0] / STEP_SECONDS), "step_s": STEP_SECONDS,
            "window_s": config.STREAM_CHUNK_SECONDS, "overlap_s": config.STREAM_OVERLAP_SECONDS}


# =============================================================================
# Sweep and aggregation
# =============================================================================

def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load_all_audio(clips: List[Dict[str, Any]]) -> Dict[str, Tuple[np.ndarray, int]]:
    audio: Dict[str, Tuple[np.ndarray, int]] = {}
    for c in clips:
        wav, sr = load_audio(c["path"], target_sr=config.SAMPLE_RATE)
        if wav.size == 0:
            raise ReplayError(f"{c['file']}: empty audio.")
        audio[str(c["path"])] = (wav, sr)
    return audio


def run_sweep(clips: List[Dict[str, Any]], detector: Any, scorer: CachedScorer,
              audio: Dict[str, Tuple[np.ndarray, int]], unit: str) -> Dict[str, Any]:
    results: Dict[Tuple[float, int], List[Dict[str, Any]]] = {}
    combos = [(t, n) for t in THRESHOLD_GRID for n in CONSECUTIVE_GRID]
    for i, (thr, n) in enumerate(combos, 1):
        rows = []
        for c in clips:
            wav, sr = audio[str(c["path"])]
            rows.append({**{k: c[k] for k in ("block", "group", "kind", "file", "speaker", "held_out")},
                         **replay(c, wav, sr, detector, scorer, thr, n, unit)})
        results[(thr, n)] = rows
        logger.info("[%s %d/%d] thr=%.1f consecutive=%d done (%d windows offered, %d scored by the detector)",
                    unit, i, len(combos), thr, n, scorer.n_calls, scorer.n_detector_calls)

    # Scoring must actually have happened for every clip, or 'not flagged' is vacuous.
    for row in results[combos[0]]:
        if row["duration"] <= 0:
            raise ReplayError(f"{row['file']}: zero duration.")
    return {"results": results, "scorer": scorer, "audio": audio, "unit": unit}


def block_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    real = [r for r in rows if r["kind"] == "REAL"]
    synth = [r for r in rows if r["kind"] == "SYNTH"]
    fp = sum(r["flagged"] for r in real)
    det = [r for r in synth if r["flagged"]]
    secs = [r["seconds_to_flag"] for r in det if r["seconds_to_flag"] is not None]
    return {
        "n_real": len(real), "n_synth": len(synth), "fp": int(fp), "det": len(det),
        "fpr": fp / len(real) if real else float("nan"),
        "det_rate": len(det) / len(synth) if synth else float("nan"),
        "median_s2f": float(statistics.median(secs)) if secs else None,
    }


def select(stats: Dict[Tuple[float, int], Dict[str, Any]]) -> Tuple[Tuple[float, int], int]:
    """The stated selection rule. Returns (combo, number of exactly-tied combos)."""
    def key(combo):
        s = stats[combo]
        return (s["fp"] / s["n_real"], -s["det"] / s["n_synth"],
                s["median_s2f"] if s["median_s2f"] is not None else float("inf"))

    order = [(t, n) for t in THRESHOLD_GRID for n in CONSECUTIVE_GRID]
    best = min(order, key=lambda c: (key(c), order.index(c)))
    tied = sum(1 for c in order if key(c) == key(best))
    return best, tied


def crosscheck(clips: List[Dict[str, Any]], detector: Any, sweep: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Re-run a few clips with an UNCACHED scorer at the pre-fix defaults and compare to the sweep."""
    thr, n = CROSSCHECK
    sel = [c for c in clips if c["block"] == "selection"]
    picks = [c for c in sel if c["kind"] == "REAL"][: N_CROSSCHECK_CLIPS // 2] + \
            [c for c in sel if c["kind"] == "SYNTH"][: N_CROSSCHECK_CLIPS // 2]
    swept = {r["file"]: r for r in sweep["results"][(thr, n)] if r["block"] == "selection"}
    out = []
    for c in picks:
        wav, sr = sweep["audio"][str(c["path"])]
        fresh = replay(c, wav, sr, detector, CountingScorer(detector), thr, n, sweep["unit"])
        ref = swept[c["file"]]
        same = (fresh["flagged"] == ref["flagged"] and fresh["seconds_to_flag"] == ref["seconds_to_flag"]
                and abs(fresh["max_score"] - ref["max_score"]) < 1e-9)
        out.append({"file": c["file"], "kind": c["kind"], "flagged": fresh["flagged"],
                    "seconds_to_flag": fresh["seconds_to_flag"], "max_score": fresh["max_score"], "match": same})
        if not same:
            raise ReplayError(f"Cached and uncached replay disagree on {c['file']} at thr={thr}, n={n}: "
                              f"{fresh} vs {ref}. The score cache cannot be trusted.")
    return out


# =============================================================================
# Report
# =============================================================================

def _sec(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.2f}s"


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _cell(s: Dict[str, Any]) -> str:
    return f"{s['fp']}/{s['n_real']} | {s['det']}/{s['n_synth']} | {_sec(s['median_s2f'])}"


def grid_lines(stats: Dict[Tuple[float, int], Dict[str, Any]], best: Optional[Tuple[float, int]], md: bool) -> List[str]:
    head = ["threshold"] + [f"consec={n}" for n in CONSECUTIVE_GRID]
    rows = []
    for t in THRESHOLD_GRID:
        cells = []
        for n in CONSECUTIVE_GRID:
            c = _cell(stats[(t, n)])
            cells.append(f"**{c}** <- selected" if best == (t, n) and md else (c + (" *" if best == (t, n) else "")))
        rows.append([f"{t:.1f}"] + cells)
    if md:
        return (["| " + " | ".join(head) + " |", "|---|" + ":---:|" * len(CONSECUTIVE_GRID)]
                + ["| " + " | ".join(r) + " |" for r in rows])
    w = 24
    out = ["  " + f"{'thr':<5}" + "".join(f"{h:^{w}}" for h in head[1:])]
    out.append("  " + "-" * (5 + w * len(CONSECUTIVE_GRID)))
    for r in rows:
        out.append("  " + f"{r[0]:<5}" + "".join(f"{c:^{w}}" for c in r[1:]))
    return out


def headroom(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Max running score per clip (independent of threshold and count): how far each clip is from the line."""
    real = sorted(((r["max_score"], r["file"]) for r in rows if r["kind"] == "REAL"), reverse=True)
    synth = sorted(((r["max_score"], r["file"]) for r in rows if r["kind"] == "SYNTH"))
    return {"real_top": real[:6], "synth_bottom": synth[:6],
            "real_over": {t: sum(m >= t for m, _ in real) for t in THRESHOLD_GRID},
            "synth_under": {t: sum(m < t for m, _ in synth) for t in THRESHOLD_GRID}}


def cadence_note(c: Dict[str, Any]) -> str:
    return (f"**What `consecutive` counts.** `StreamingSession(consecutive_unit=\"updates\")`: consecutive per-stride SCORE UPDATES "
            f"(scored windows), not push calls. Measured from a live session (window {c['window_s']}s, overlap {c['overlap_s']}s, "
            f"{c['step_s']}s pushes): the first score update lands at {c['first_update_s']:.2f}s (push {c['first_update_push']}) and updates then "
            f"arrive every {c['update_gap_s']:.2f}s = every {c['pushes_per_update']:g} pushes. So N=1..4 mean 1..4 independent decisions in a row, "
            f"and the earliest possible flag is {c['first_update_s']:.1f}s + (N-1) x {c['update_gap_s']:.1f}s.")


def debounce_analysis(results: Dict[Tuple[float, int], List[Dict[str, Any]]], thr: float) -> Dict[str, Any]:
    """Does requiring N consecutive per-stride decisions remove the residual false positives at ``thr``?

    The residual set is soumya's REAL clips that flag at N=1. ``longest_run`` is not measured directly; a clip
    that still flags at N was at/above the threshold for at least N consecutive updates.
    """
    sel = lambda rows: [r for r in rows if r["block"] == "selection"]  # noqa: E731
    base = sel(results[(thr, 1)])
    residual = sorted(r["file"] for r in base if r["kind"] == "REAL" and r["flagged"])
    rows_by_n = []
    for n in DEBOUNCE_N:
        rows = sel(results[(thr, n)])
        by_file = {r["file"]: r for r in rows}
        s = block_stats(rows)
        rows_by_n.append({
            "n": n, "stats": s,
            "residual": {f: {"flagged": by_file[f]["flagged"], "s2f": by_file[f]["seconds_to_flag"],
                             "max_score": by_file[f]["max_score"]} for f in residual},
            "still_flagged": sorted(f for f in residual if by_file[f]["flagged"]),
            "cleared": sorted(f for f in residual if not by_file[f]["flagged"]),
            "synth_missed": sorted(r["file"] for r in rows if r["kind"] == "SYNTH" and not r["flagged"]),
        })
    # Robustness: the highest threshold at which each residual clip would STILL flag under N consecutive updates.
    # floor_n = max over windows of n consecutive updates of the minimum running score in that window. A clip
    # flags at (thr, n) iff floor_n >= thr, so thr - floor_n is how far under the line the debounce leaves it.
    base_by_file = {r["file"]: r for r in base}
    for r in rows_by_n:
        r["floors"] = {}
        for f in residual:
            u = base_by_file[f]["update_scores"]
            k = r["n"]
            r["floors"][f] = (max(min(u[i:i + k]) for i in range(len(u) - k + 1)) if len(u) >= k else None)
    n1 = rows_by_n[0]["stats"]
    for r in rows_by_n:
        s = r["stats"]
        r["d_fp"] = s["fp"] - n1["fp"]
        r["d_det"] = s["det"] - n1["det"]
        r["d_latency"] = (None if s["median_s2f"] is None or n1["median_s2f"] is None else s["median_s2f"] - n1["median_s2f"])
    return {"threshold": thr, "residual": residual, "by_n": rows_by_n,
            "update_scores": {f: [round(x, 4) for x in base_by_file[f]["update_scores"]] for f in residual}}


def _fl(x: Optional[float], nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _margin(thr: float, floor: Optional[float]) -> str:
    return "n/a" if floor is None else f"{thr - floor:+.3f}"


def debounce_lines(d: Dict[str, Any], md: bool) -> List[str]:
    res = d["residual"]
    out: List[str] = []
    if md:
        out += [f"At threshold {d['threshold']} and N=1 the residual false positives are {len(res)} real soumya clips: "
                + ", ".join(f"`{f}`" for f in res) + ".", "",
                "| N (consecutive score updates) | False positives | Residual FPs still flagging | Detections | Median seconds_to_flag | "
                "Latency vs N=1 | Synthetic clips lost vs N=1 |", "|---|:---:|---|:---:|:---:|:---:|---|"]
    else:
        out.append(f"  {'N':<3}{'FP':>6}{'det':>8}{'med s2f':>9}{'d lat':>8}   residual FPs still flagging (first flag time)")
        out.append("  " + "-" * 96)
    n1_missed = set(d["by_n"][0]["synth_missed"])
    for r in d["by_n"]:
        s = r["stats"]
        lost = sorted(set(r["synth_missed"]) - n1_missed)
        still = ", ".join(f"{f} ({_sec(r['residual'][f]['s2f'])})" for f in r["still_flagged"]) or "none"
        dl = "n/a" if r["d_latency"] is None else f"{r['d_latency']:+.2f}s"
        if md:
            out.append(f"| {r['n']} | {s['fp']}/{s['n_real']} | {still} | {s['det']}/{s['n_synth']} | {_sec(s['median_s2f'])} | {dl} | "
                       f"{', '.join('`' + f + '`' for f in lost) or 'none'} |")
        else:
            out.append(f"  {r['n']:<3}{str(s['fp']) + '/' + str(s['n_real']):>6}{str(s['det']) + '/' + str(s['n_synth']):>8}"
                       f"{_sec(s['median_s2f']):>9}{dl:>8}   {still}" + (f"   [lost: {', '.join(lost)}]" if lost else ""))
    if md:
        out += ["", "**How far under the line the debounce leaves each residual false positive.** `floor(N)` = the highest threshold at which the "
                "clip would still flag with N consecutive score updates (max over runs of N updates of the minimum running score). "
                f"A clip flags at threshold {d['threshold']} iff floor(N) >= {d['threshold']}.", "",
                "| Clip | Update scores (running EMA) | floor(1) = max | floor(2) | margin under 0.8 at N=2 | floor(3) |", "|---|---|:---:|:---:|:---:|:---:|"]
        for f in res:
            fl = {r["n"]: r["floors"][f] for r in d["by_n"]}
            out.append(f"| `{f}` | {', '.join(f'{x:.2f}' for x in d['update_scores'][f])} | {_fl(fl[1])} | {_fl(fl[2])} | "
                       f"{_margin(d['threshold'], fl[2])} | {_fl(fl[3])} |")
    else:
        out.append("")
        out.append(f"  floor(N) = highest threshold at which the clip would still flag with N consecutive updates (flags at {d['threshold']} iff floor >= {d['threshold']}):")
        out.append(f"  {'clip':<24}{'floor(1)':>9}{'floor(2)':>9}{'floor(3)':>9}{'margin@N=2':>12}   update scores")
        for f in res:
            fl = {r["n"]: r["floors"][f] for r in d["by_n"]}
            out.append(f"  {f:<24}{_fl(fl[1]):>9}{_fl(fl[2]):>9}{_fl(fl[3]):>9}{_margin(d['threshold'], fl[2]):>12}   "
                       + " ".join(f"{x:.2f}" for x in d["update_scores"][f]))
    return out


def build_markdown(ctx: Dict[str, Any]) -> str:
    sel_stats, demo_stats, best, tied = ctx["sel_stats"], ctx["demo_stats"], ctx["best"], ctx["tied"]
    bs = sel_stats[best]
    lo, hi = wilson(bs["fp"], bs["n_real"])
    L = [
        "# Streaming Threshold Calibration (Phase F4)",
        "",
        f"**Generated at:** {datetime.now(timezone.utc).isoformat()}",
        "**Script:** `scripts/fix_calibrate_stream_threshold.py`",
        "**Heads:** " + ", ".join(f"`{Path(p).name}`" for p in config.PRODUCTION_STREAMING_CLASSIFIERS.values())
        + f" (weighted average, weight_a={config.PRODUCTION_ENSEMBLE_WEIGHT_A})",
        f"**Session:** real `StreamingSession` + `simulate_stream(step_seconds={STEP_SECONDS}, real_time_paced=False)`; chunk="
        f"{config.STREAM_CHUNK_SECONDS}s, overlap={config.STREAM_OVERLAP_SECONDS}s, EMA alpha=0.3; one fresh session per clip per combination.",
        "**Audio:** duration-MATCHED clips only.",
        "",
        "> " + cadence_note(ctx["cadence"]),
        "",
        "> **Revision.** The first version of this sweep counted `push_audio` calls, so N=1..4 all resolved inside one score update "
        "(see the legacy grid below). This version counts score updates.",
        "",
        "## Selection rule",
        "",
    ] + [f"- {s}" for s in SELECTION_RULE] + ["",
        "Each cell: `false positives | detections | median seconds_to_flag (detected synthetic)`, i.e. real clips flagged / real clips, "
        "synthetic clips flagged / synthetic clips, median over the detected ones.", ""]
    L += ["## Block 1 - soumya matched eval (HELD OUT; the ONLY block used for selection)", "",
          f"{bs['n_real']} real + {bs['n_synth']} synthetic clips; one clip is {100 / bs['n_real']:.0f} pp. Speaker {HELD_OUT_SPEAKER} was never seen by the chunked heads.", ""]
    L += grid_lines(sel_stats, best, True)
    L += ["", f"**Selected: STREAM_FLAG_THRESHOLD = {best[0]}, consecutive_flags_required = {best[1]}** - "
              f"false-positive rate {_pct(bs['fpr'])} ({bs['fp']}/{bs['n_real']}; Wilson 95% [{lo * 100:.0f}%, {hi * 100:.0f}%]), "
              f"detection {_pct(bs['det_rate'])} ({bs['det']}/{bs['n_synth']}), median seconds_to_flag {_sec(bs['median_s2f'])}. "
              f"{tied} combination(s) tie exactly on all three keys.", ""]

    h = ctx["headroom"]
    L += ["### Headroom (independent of threshold and count)", "",
          "Each clip's highest running score. A real clip whose max is above a threshold flags at that threshold whatever the count "
          "(given enough sustained steps); a synthetic clip whose max is below it can never flag.", "",
          "| Threshold | soumya REAL clips with max >= thr | soumya SYNTH clips with max < thr (unreachable) |", "|---|:---:|:---:|"]
    for t in THRESHOLD_GRID:
        L.append(f"| {t:.1f} | {h['real_over'][t]} | {h['synth_under'][t]} |")
    L += ["", "Highest-scoring REAL clips: " + ", ".join(f"`{f}` {m:.3f}" for m, f in h["real_top"]) + ".",
          "Lowest-scoring SYNTHETIC clips: " + ", ".join(f"`{f}` {m:.3f}" for m, f in h["synth_bottom"]) + ".", ""]

    L += [f"## Does a real debounce remove the residual false positives? (soumya, threshold {ctx['debounce']['threshold']})", "",
          "Analysis grid N=1..6 (the selection grid stays 1..4). N counts consecutive per-stride score updates at or above the threshold; "
          "'still flagging' means the clip stayed at/above the threshold for at least N updates in a row at some point.", ""]
    L += debounce_lines(ctx["debounce"], True)
    L += [""]
    L += ["### Legacy push-step counting (superseded; comparison only)", "",
          "The original session counting, same clips and scores. If its columns are identical, that confirms counts 1..4 resolved inside "
          "one score update.", ""]
    L += grid_lines(ctx["legacy_stats"], None, True)
    L += [""]

    L += ["## Block 2 - Phase 6 demo pairs (MATCHED audio; NOT held-out evidence, NOT used for selection)", "",
          "Pairs 1-4 are byaquta / mahato clips: the chunked heads were TRAINED on these speakers. Pair 5 is soumya (its two clips are also in Block 1). "
          "Shown for continuity with earlier reporting only.", ""]
    L += grid_lines(demo_stats, None, True)
    L += ["", f"Demo block at the selected combination ({best[0]}, {best[1]}):", "",
          "| Group | Kind | File | Speaker | Flagged | Sec to flag | Max score |", "|---|:---:|---|:---:|:---:|:---:|:---:|"]
    for r in ctx["demo_rows"]:
        L.append(f"| {r['group']} | {r['kind']} | `{r['file']}` | {'held-out' if r['held_out'] else 'TRAIN speaker'} | "
                 f"{r['flagged']} | {_sec(r['seconds_to_flag'])} | {r['max_score']:.3f} |")
    L += ["", "## Cross-check (cached vs uncached scorer at the pre-fix defaults "
          f"thr={CROSSCHECK[0]}, consecutive={CROSSCHECK[1]})", "",
          "| File | Kind | Flagged | Sec to flag | Max score | Cached run identical |", "|---|:---:|:---:|:---:|:---:|:---:|"]
    for r in ctx["crosscheck"]:
        L.append(f"| `{r['file']}` | {r['kind']} | {r['flagged']} | {_sec(r['seconds_to_flag'])} | {r['max_score']:.4f} | {'yes' if r['match'] else '**NO**'} |")
    L += ["", f"Windows offered to the scorer across the whole sweep: {ctx['n_calls']:,}; unique windows scored by the real detector: {ctx['n_detector_calls']:,}.", ""]
    return "\n".join(L)


def print_console(ctx: Dict[str, Any]) -> None:
    bar = "=" * 100
    best, bs = ctx["best"], ctx["sel_stats"][ctx["best"]]
    print("\n" + bar)
    print(" STREAM FLAG CALIBRATION - real StreamingSession, production STREAMING heads, MATCHED audio")
    print(bar)
    print(" Heads: " + ", ".join(Path(p).name for p in config.PRODUCTION_STREAMING_CLASSIFIERS.values())
          + f"  (weight_a={config.PRODUCTION_ENSEMBLE_WEIGHT_A})")
    print(f" Session: chunk={config.STREAM_CHUNK_SECONDS}s overlap={config.STREAM_OVERLAP_SECONDS}s step={STEP_SECONDS}s EMA alpha=0.3")
    c = ctx["cadence"]
    print(f" MEASURED cadence: first score update at {c['first_update_s']:.2f}s (push {c['first_update_push']}), then every "
          f"{c['update_gap_s']:.2f}s = every {c['pushes_per_update']:g} pushes of {c['step_s']}s.")
    print(" 'consecutive' now counts per-stride SCORE UPDATES (consecutive_unit='updates'): N=1..4 = 1..4 independent decisions in a row;")
    print(f" earliest flag = {c['first_update_s']:.1f}s + (N-1) x {c['update_gap_s']:.1f}s.")
    print("-" * 100)
    print(" SELECTION RULE:")
    for s in SELECTION_RULE:
        print("   " + s)
    print("-" * 100)
    print(" BLOCK 1 - soumya matched eval (HELD OUT, the ONLY block used for selection)")
    print(" cell = false positives / real | detections / synthetic | median seconds_to_flag")
    for line in grid_lines(ctx["sel_stats"], best, False):
        print(line)
    print(f"\n   SELECTED: threshold={best[0]}  consecutive={best[1]}  FPR {_pct(bs['fpr'])} ({bs['fp']}/{bs['n_real']})  "
          f"detection {_pct(bs['det_rate'])} ({bs['det']}/{bs['n_synth']})  median s2f {_sec(bs['median_s2f'])}  "
          f"[{ctx['tied']} combination(s) tied on all keys]")
    h = ctx["headroom"]
    print("   headroom: REAL clips with max running score >= thr: "
          + "  ".join(f"{t:.1f}:{h['real_over'][t]}" for t in THRESHOLD_GRID)
          + " | SYNTH clips that can never reach thr: " + "  ".join(f"{t:.1f}:{h['synth_under'][t]}" for t in THRESHOLD_GRID))
    print("   highest REAL max: " + ", ".join(f"{f} {m:.3f}" for m, f in h["real_top"][:4]))
    print("-" * 100)
    print(f" DEBOUNCE AT THRESHOLD {ctx['debounce']['threshold']} - does N >= 2 remove the residual false positives? (soumya, held out)")
    print(f" residual FPs at N=1: {', '.join(ctx['debounce']['residual'])}")
    for line in debounce_lines(ctx["debounce"], False):
        print(line)
    print("-" * 100)
    print(" LEGACY push-step counting (superseded, comparison only):")
    for line in grid_lines(ctx["legacy_stats"], None, False):
        print(line)
    print("-" * 100)
    print(" BLOCK 2 - Phase 6 demo pairs, MATCHED audio (pairs 1-4 are TRAINING speakers: NOT held-out, NOT used to select)")
    for line in grid_lines(ctx["demo_stats"], None, False):
        print(line)
    print(bar)
    print(f" Cross-check cached vs uncached: {sum(r['match'] for r in ctx['crosscheck'])}/{len(ctx['crosscheck'])} identical")
    print(f" Report written to: {REPORT_PATH}\n")


# =============================================================================
# Config write (typed confirmation)
# =============================================================================

def config_block(thr: float, n: int, ctx: Dict[str, Any]) -> str:
    s = ctx["sel_stats"][(thr, n)]
    h = ctx["headroom"]
    return "\n".join([
        BLOCK_BEGIN,
        "# Calibrated in F4 by replaying the soumya MATCHED eval clips (held-out; 25 real + 25 synthetic) through the real",
        "# StreamingSession wired to PRODUCTION_STREAMING_CLASSIFIERS (chunked wav2vec2 + v2 chunked WavLM, weight 0.5).",
        "# Sweep: threshold in [0.4..0.8] x consecutive score updates in [1..4]. Rule: lowest false-positive rate on real clips; ties ->",
        "# highest detection rate; remaining ties -> lowest median seconds_to_flag. A false positive on genuine speech outranks",
        "# latency. Selected on soumya alone; the Phase 6 demo pairs (mostly training speakers) were reported but not used.",
        f"# Result at {thr}/{n}: false positives {s['fp']}/{s['n_real']} real clips, detections {s['det']}/{s['n_synth']} synthetic clips,",
        f"# median seconds_to_flag {_sec(s['median_s2f'])}. Highest real-clip max running score: {h['real_top'][0][0]:.3f}.",
        "# STREAM_CONSECUTIVE_FLAGS_REQUIRED counts consecutive per-stride SCORE UPDATES (StreamingSession(consecutive_unit=",
        "# \"updates\")): the running score only changes when a window completes (first at 1.5 s, then every 1.0 s), so N means N",
        "# independent decisions in a row and the earliest flag is 1.5 s + (N-1) x 1.0 s. The session default unit is still \"pushes\";",
        "# call sites must pass STREAM_CONSECUTIVE_UNIT. 25 real clips = 4 pp per clip; treat small differences as noise.",
        "# The whole-clip RISK_THRESHOLDS are calibrated separately and are not affected. Call sites read this in F5.",
        f"STREAM_FLAG_THRESHOLD: float | None = {thr}",
        f"STREAM_CONSECUTIVE_FLAGS_REQUIRED: int = {n}",
        'STREAM_CONSECUTIVE_UNIT: str = "updates"',
        BLOCK_END,
    ])


def patch_config(thr: float, n: int, ctx: Dict[str, Any]) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    block = config_block(thr, n, ctx)
    marked = re.compile(re.escape(BLOCK_BEGIN) + r".*?" + re.escape(BLOCK_END), re.DOTALL)
    original = re.compile(r"# Decision threshold used by the streaming session wrapper\.\nSTREAM_FLAG_THRESHOLD: float \| None = [\d.]+")
    if marked.search(text):
        new_text = marked.sub(lambda m: block, text, count=1)
    elif original.search(text):
        new_text = original.sub(lambda m: block, text, count=1)
    else:
        print("\n  WARNING: could not find STREAM_FLAG_THRESHOLD in config.py - NOT modified. Set manually:\n"
              f"    STREAM_FLAG_THRESHOLD = {thr}\n    STREAM_CONSECUTIVE_FLAGS_REQUIRED = {n}\n")
        return
    CONFIG_PATH.write_text(new_text, encoding="utf-8")
    print(f"\n  config.py updated: STREAM_FLAG_THRESHOLD={thr}, STREAM_CONSECUTIVE_FLAGS_REQUIRED={n}")


def confirm_and_write(ctx: Dict[str, Any]) -> None:
    rec = ctx["best"]
    print("-" * 72)
    print("Enter 'threshold,consecutive' to write to config.py, press Enter to accept the")
    print("recommendation (selected by the rule above on soumya alone), or 'q' to quit.\n")
    while True:
        try:
            raw = input(f"  threshold,consecutive  [{rec[0]},{rec[1]}]: ").strip()
        except EOFError:
            print("\n  No input - config.py unchanged.")
            return
        if raw.lower() in ("q", "quit", "exit"):
            print("\n  Aborted - config.py unchanged.")
            return
        if raw == "":
            thr, n = rec
        else:
            parts = [p.strip() for p in raw.split(",")]
            try:
                thr, n = float(parts[0]), int(parts[1])
                assert len(parts) == 2
            except (ValueError, IndexError, AssertionError):
                print("  Enter exactly two values, e.g. 0.6,3")
                continue
            if (round(thr, 6), n) not in {(round(t, 6), c) for t in THRESHOLD_GRID for c in CONSECUTIVE_GRID}:
                print(f"  Not in the swept grid (thresholds {THRESHOLD_GRID}, counts {CONSECUTIVE_GRID}); the sweep says nothing about it.")
                continue
        if (thr, n) != rec:
            s = ctx["sel_stats"][(thr, n)]
            print(f"  NOTE: this overrides the rule's pick {rec}; its soumya result is FP {s['fp']}/{s['n_real']}, "
                  f"detection {s['det']}/{s['n_synth']}, median s2f {_sec(s['median_s2f'])}.")
        print(f"\n  Will write to config.py: STREAM_FLAG_THRESHOLD={thr}, STREAM_CONSECUTIVE_FLAGS_REQUIRED={n}")
        if input("  Confirm? [y/N]: ").strip().lower() in ("y", "yes"):
            patch_config(thr, n, ctx)
            return
        print("  Not confirmed - try again or type 'q' to quit.\n")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output_md", type=Path, default=REPORT_PATH)
    parser.add_argument("--no_write", action="store_true", help="Print and save the report; never offer to write config.py.")
    args = parser.parse_args()

    try:
        cadence = measure_cadence()
        clips = resolve_clips()
        detector = build_detector()
        scorer = CachedScorer(detector)
        audio = load_all_audio(clips)
        sweep = run_sweep(clips, detector, scorer, audio, COUNT_UNIT)
        debounce_sweep = run_sweep_grid_extension(clips, detector, scorer, audio)
        legacy = run_sweep(clips, detector, scorer, audio, LEGACY_UNIT)
        checks = crosscheck(clips, detector, sweep)
    except Exception as exc:
        logger.error("Stream calibration aborted (results untrustworthy): %s", exc)
        sys.exit(2)

    res = sweep["results"]
    sel_stats = {c: block_stats([r for r in rows if r["block"] == "selection"]) for c, rows in res.items()}
    demo_stats = {c: block_stats([r for r in rows if r["block"] == "demo"]) for c, rows in res.items()}
    best, tied = select(sel_stats)
    legacy_stats = {c: block_stats([r for r in rows if r["block"] == "selection"]) for c, rows in legacy["results"].items()}
    ctx = {
        "cadence": cadence, "legacy_stats": legacy_stats,
        "debounce": debounce_analysis({**res, **debounce_sweep}, DEBOUNCE_THRESHOLD),
        "sel_stats": sel_stats, "demo_stats": demo_stats, "best": best, "tied": tied,
        "headroom": headroom([r for r in res[best] if r["block"] == "selection"]),
        "demo_rows": [r for r in res[best] if r["block"] == "demo"],
        "crosscheck": checks, "n_calls": sweep["scorer"].n_calls, "n_detector_calls": sweep["scorer"].n_detector_calls,
    }

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_markdown(ctx), encoding="utf-8")
    args.output_md.with_suffix(".json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selected": {"threshold": best[0], "consecutive": best[1], "tied_combinations": tied},
        "selection_rule": SELECTION_RULE,
        "cadence": cadence, "count_unit": COUNT_UNIT,
        "debounce_at_threshold": ctx["debounce"],
        "results": [{"threshold": t, "consecutive": n, "selection": sel_stats[(t, n)], "demo": demo_stats[(t, n)]}
                    for (t, n) in sel_stats],
        "per_clip": [{"threshold": t, "consecutive": n, **{k: v for k, v in r.items() if k != "update_scores"}}
                     for (t, n), rows in res.items() for r in rows],
    }, indent=2), encoding="utf-8")
    logger.info("Saved %s", args.output_md)
    print_console(ctx)

    if not args.no_write:
        confirm_and_write(ctx)


if __name__ == "__main__":
    main()
