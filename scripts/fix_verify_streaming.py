#!/usr/bin/env python3
"""
scripts/fix_verify_streaming.py — Replay the previously-misbehaving clips through the NEW chunked heads (Phase F2.5).

Builds a ``StreamingSession`` explicitly wired to
``models/classifiers/{wav2vec2,wavlm}_chunked_logreg.joblib`` (via
``WeightedAverageDetector``) and replays every clip that misbehaved before the fix
through the real ``simulate_stream`` path — no bespoke loop:

  - soumya_neutral_01_clone.wav  (synthetic; previously NEVER flagged)
  - byaquta_neutral_01.wav       (real; previously flagged — false positive)
  - soumya_control_21.wav        (real; previously flagged — false positive)
  - soumya_scam_11.wav           (real; previously flagged at the 4.0 s sweep window)
  - all 5 verified Phase 6 demo pairs, real and synthetic

Per clip: flagged, seconds_to_flag, final running score, max running score, plus a
PASS/FAIL line against the expected outcome (synthetic -> flagged, real -> NOT flagged).

flag_threshold and consecutive_flags_required are deliberately left at the session
defaults (the pre-fix values; F4 recalibrates them against the new score
distribution). A close-but-wrong result here is RECORDED, never tuned away: the
report shows each clip's max score relative to the threshold so F4 can use it.

Exit codes
----------
0  no real clip flagged (missed synthetic clips are printed as FAIL but do not change
   the exit code at this stage — the F2 gate does not depend on them)
1  at least one REAL clip flagged: a false positive on genuine human speech
2  the replay itself is not trustworthy (a chunk failed to score, or a clip produced
   no scored chunk at all) — see below

Why exit 2 exists: ``StreamingScorer.score_chunk`` swallows every exception and returns
``None``. If the chunked heads were mis-wired (wrong width, bad scaler), every window would
silently score ``None``, the running score would sit at 0.0, and every REAL clip would
"PASS" vacuously. The session's scorer is therefore wrapped in a counting subclass that
does not change what it returns, only records windows offered / scored / failed.

Audio set: by default the ORIGINAL clips under ``data/raw/hindi_hinglish/`` (the files
the F0 baseline used, so results are like-for-like). ``--audio_set matched`` /
``both`` adds the duration-MATCHED versions the chunked heads were trained on. The real
clips are byte-identical between the two folders; only the synthetic clones differ.

Held-out status: the chunked heads were trained on byaquta and mahato, so their clips are
NOT held-out evidence. Only ``soumya_*`` clips are (the report labels every row).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from voxguard import config
from voxguard.classifier.ensemble import WeightedAverageDetector
from voxguard.streaming.scorer import StreamingScorer
from voxguard.streaming.session import StreamingSession, simulate_stream
from voxguard.utils.logging_utils import get_logger

logger = get_logger("fix_verify_streaming")

CHUNKED_WAV2VEC2_CLF = config.MODELS_DIR / "classifiers" / "wav2vec2_chunked_logreg.joblib"
CHUNKED_WAVLM_CLF = config.MODELS_DIR / "classifiers" / "wavlm_chunked_logreg.joblib"
DEFAULT_REPORT_PATH = config.MODELS_DIR / "reports" / "fix_verify_streaming.md"
BASELINE_JSON = config.MODELS_DIR / "reports" / "fix_baseline.json"

HELD_OUT_SPEAKER = "soumya"
AUDIO_DIRS = {"original": "hindi_hinglish", "matched": "hindi_hinglish_matched"}

# (group, kind, file stem). "Previously misbehaved" clips first, then the 5 demo pairs.
CLIPS: List[tuple] = [
    ("Previously misbehaved", "SYNTH", "soumya_neutral_01_clone"),
    ("Previously misbehaved", "REAL", "byaquta_neutral_01"),
    ("Previously misbehaved", "REAL", "soumya_control_21"),
    ("Previously misbehaved", "REAL", "soumya_scam_11"),
    ("Pair 1: Casual Neutral (byaquta)", "REAL", "byaquta_neutral_09"),
    ("Pair 1: Casual Neutral (byaquta)", "SYNTH", "byaquta_neutral_09_clone"),
    ("Pair 2: Everyday Tech (mahato)", "REAL", "mahato_neutral_04"),
    ("Pair 2: Everyday Tech (mahato)", "SYNTH", "mahato_neutral_04_clone"),
    ("Pair 3: Urgent Legal Scam (byaquta)", "REAL", "byaquta_scam_16"),
    ("Pair 3: Urgent Legal Scam (byaquta)", "SYNTH", "byaquta_scam_16_clone"),
    ("Pair 4: Authority Customs Scam (mahato)", "REAL", "mahato_scam_12"),
    ("Pair 4: Authority Customs Scam (mahato)", "SYNTH", "mahato_scam_12_clone"),
    ("Pair 5: Held-Out Casual (soumya)", "REAL", "soumya_neutral_03"),
    ("Pair 5: Held-Out Casual (soumya)", "SYNTH", "soumya_neutral_03_clone"),
]


class ReplayError(RuntimeError):
    """The replay cannot be trusted (exit code 2)."""


class InstrumentedScorer(StreamingScorer):
    """StreamingScorer that records what happened to each window without altering it.

    ``n_failed`` counts windows that were LOUD enough to be scored (RMS at or above the
    silence gate) yet still returned ``None`` — i.e. the detector raised and
    ``score_chunk`` swallowed it.
    """

    def __init__(self, detector: Any) -> None:
        super().__init__(detector)
        self.n_windows = 0
        self.n_scored = 0
        self.n_failed = 0

    def score_chunk(self, chunk, sr):
        self.n_windows += 1
        score = super().score_chunk(chunk, sr)
        if score is not None:
            self.n_scored += 1
        elif self._rms_energy(chunk) >= self.silence_threshold:
            self.n_failed += 1
        return score


def build_detector() -> WeightedAverageDetector:
    """Detector explicitly wired to the NEW chunked heads (never the whole-clip defaults)."""
    for p in (CHUNKED_WAV2VEC2_CLF, CHUNKED_WAVLM_CLF):
        if not p.exists():
            raise ReplayError(f"Chunked classifier not found: {p} (run train_chunked_classifier.py).")
    detector = WeightedAverageDetector(
        wav2vec2_classifier_path=CHUNKED_WAV2VEC2_CLF,
        wavlm_classifier_path=CHUNKED_WAVLM_CLF,
    )
    wired = (detector.detector_a.classifier_path.resolve(), detector.detector_b.classifier_path.resolve())
    if wired != (CHUNKED_WAV2VEC2_CLF.resolve(), CHUNKED_WAVLM_CLF.resolve()):
        raise ReplayError(f"Detector is not wired to the chunked heads: {wired}")
    return detector


def resolve_clips(audio_sets: List[str]) -> List[Dict[str, Any]]:
    """Expands CLIPS over the audio sets and checks every file exists BEFORE loading any model."""
    clips: List[Dict[str, Any]] = []
    missing: List[str] = []
    for audio_set in audio_sets:
        for group, kind, stem in CLIPS:
            folder = "real" if kind == "REAL" else "synthetic"
            path = config.BASE_DIR / "data" / "raw" / AUDIO_DIRS[audio_set] / folder / f"{stem}.wav"
            if not path.exists():
                missing.append(str(path))
            speaker = stem.split("_")[0]
            clips.append({
                "audio_set": audio_set, "group": group, "kind": kind, "file": f"{stem}.wav",
                "path": path, "speaker": speaker, "held_out": speaker == HELD_OUT_SPEAKER,
            })
    if missing:
        raise ReplayError("Clips not found:\n  " + "\n  ".join(missing))
    return clips


def load_f0_outcomes() -> Dict[str, Dict[str, Any]]:
    """File name -> F0 whole-clip streaming outcome, where fix_baseline.json recorded one."""
    if not BASELINE_JSON.exists():
        return {}
    base = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))["item5_streaming"]
    out: Dict[str, Dict[str, Any]] = {}
    for p in base["demo_pairs"]:
        out[p["real"]["file"]] = p["real"]
        out[p["synthetic"]["file"]] = p["synthetic"]
    for r in base["sweep_real_clips"]:
        out[r["file"]] = r
    return out


def replay_clip(detector: WeightedAverageDetector, clip: Dict[str, Any]) -> Dict[str, Any]:
    """One fresh session per clip, replayed through the real simulate_stream."""
    session = StreamingSession(
        detector=detector,
        chunk_seconds=config.STREAM_CHUNK_SECONDS,
        overlap_seconds=config.STREAM_OVERLAP_SECONDS,
    )
    scorer = InstrumentedScorer(detector)
    session.scorer = scorer

    sim = simulate_stream(clip["path"], session=session, real_time_paced=False)

    if scorer.n_failed:
        raise ReplayError(
            f"{clip['file']}: {scorer.n_failed} audible window(s) failed to score "
            "(the detector raised and StreamingScorer swallowed it). Results would be meaningless."
        )
    if scorer.n_scored == 0:
        raise ReplayError(
            f"{clip['file']}: no window was scored ({scorer.n_windows} offered, "
            f"{sim['total_duration']:.2f}s of audio). A 'not flagged' result here would be vacuous."
        )

    max_running = max(r["running_score"] for r in sim["step_results"])
    expected_flag = clip["kind"] == "SYNTH"
    return {
        **{k: clip[k] for k in ("audio_set", "group", "kind", "file", "speaker", "held_out")},
        "flagged": bool(sim["flagged"]),
        "seconds_to_flag": sim["seconds_to_flag"],
        "final_score": float(sim["final_running_score"]),
        "max_score": float(max_running),
        "duration": float(sim["total_duration"]),
        "windows_scored": scorer.n_scored,
        "windows_offered": scorer.n_windows,
        "expected_flag": expected_flag,
        "passed": bool(sim["flagged"]) == expected_flag,
        "threshold": session.flag_threshold,
        "consecutive": session.consecutive_flags_required,
    }


# =============================================================================
# Reporting
# =============================================================================

def _sec(x: Optional[float]) -> str:
    return "N/A" if x is None else f"{x:.2f}s"


def _held(r: Dict[str, Any]) -> str:
    return "held-out" if r["held_out"] else "TRAIN speaker"


def _f0_cell(r: Dict[str, Any], f0: Dict[str, Dict[str, Any]]) -> str:
    rec = f0.get(r["file"])
    if rec is None:
        return "n/a"
    return "flagged" if rec["flagged"] else "not flagged"


def build_markdown(
    results: List[Dict[str, Any]], f0: Dict[str, Dict[str, Any]], threshold: float, consecutive: int
) -> str:
    n_pass = sum(r["passed"] for r in results)
    real_fp = [r for r in results if r["kind"] == "REAL" and r["flagged"]]
    synth_miss = [r for r in results if r["kind"] == "SYNTH" and not r["flagged"]]
    L = [
        "# Streaming Verification — NEW Chunked Heads (Phase F2.5)",
        "",
        f"**Generated at:** {datetime.now(timezone.utc).isoformat()}",
        "**Script:** `scripts/fix_verify_streaming.py`",
        f"**Heads:** `{CHUNKED_WAV2VEC2_CLF.name}` + `{CHUNKED_WAVLM_CLF.name}` (weighted average 0.5 / 0.5)",
        f"**Session settings (UNCHANGED pre-fix values, not tuned here):** chunk={config.STREAM_CHUNK_SECONDS}s, "
        f"overlap={config.STREAM_OVERLAP_SECONDS}s, flag_threshold={threshold}, consecutive_flags_required={consecutive}, EMA alpha=0.3",
        "**Path:** `StreamingSession` + `simulate_stream(real_time_paced=False)`; one fresh session per clip.",
        "",
        f"**Result: {n_pass}/{len(results)} PASS.** "
        f"Real clips flagged (false positives): **{len(real_fp)}**. Synthetic clips not flagged: **{len(synth_miss)}**.",
        "",
        "> Held-out evidence is `soumya_*` only. The chunked heads were trained on byaquta and mahato, so "
        "those rows show the model behaving on data it has seen and are not proof of generalisation.",
        "",
        "> Near misses are recorded, not tuned: `Max - thr` is the clip's highest running score minus the flag "
        "threshold. F4 recalibrates the threshold and consecutive-flag count against the new score distribution.",
        "",
        "| Audio | Group | Kind | File | Speaker | Flagged | Sec to flag | Final | Max | Max − thr | Windows scored | Expected | Result | F0 (pre-fix, original audio) |",
        "|---|---|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in results:
        exp = "flag" if r["expected_flag"] else "no flag"
        res = "PASS" if r["passed"] else "**FAIL**"
        L.append(
            f"| {r['audio_set']} | {r['group']} | {r['kind']} | `{r['file']}` | {_held(r)} | {r['flagged']} | "
            f"{_sec(r['seconds_to_flag'])} | {r['final_score']:.4f} | {r['max_score']:.4f} | "
            f"{r['max_score'] - threshold:+.4f} | {r['windows_scored']}/{r['windows_offered']} | {exp} | {res} | {_f0_cell(r, f0)} |"
        )
    L += ["", "`Windows scored` is scored / offered; the difference is windows below the silence gate.", ""]
    return "\n".join(L)


def print_results(results: List[Dict[str, Any]], threshold: float, consecutive: int) -> None:
    bar = "=" * 118
    print("\n" + bar)
    print(" STREAMING VERIFICATION - NEW chunked heads through StreamingSession / simulate_stream")
    print(f" flag_threshold={threshold}  consecutive_flags_required={consecutive}  (pre-fix values; F4 recalibrates)")
    print(bar)
    print(f" {'Audio':<9}{'Kind':<6}{'File':<32}{'Speaker':<14}{'Flagged':<9}{'ToFlag':<8}{'Final':<8}{'Max':<8}{'Max-thr':<9}{'Win':<7}Result")
    print("-" * 118)
    for r in results:
        print(
            f" {r['audio_set']:<9}{r['kind']:<6}{r['file']:<32}{_held(r):<14}{str(r['flagged']):<9}"
            f"{_sec(r['seconds_to_flag']):<8}{r['final_score']:<8.4f}{r['max_score']:<8.4f}"
            f"{r['max_score'] - threshold:<+9.4f}{r['windows_scored']}/{r['windows_offered']:<4}"
            f"{'PASS' if r['passed'] else 'FAIL'}"
        )
    print(bar)
    for r in results:
        want = "flagged" if r["expected_flag"] else "not flagged"
        got = "flagged" if r["flagged"] else "not flagged"
        print(f" [{'PASS' if r['passed'] else 'FAIL'}] {r['audio_set']}/{r['file']}: expected {want}, got {got}")
    n_pass = sum(r["passed"] for r in results)
    print(bar)
    print(f" OVERALL: {n_pass}/{len(results)} PASS")

    key = [r for r in results if r["file"] == "soumya_neutral_01_clone.wav"]
    for r in key:
        verdict = "NOW FLAGS" if r["flagged"] else "STILL NOT FLAGGED"
        print(
            f" Gate F2 note - soumya_neutral_01_clone.wav [{r['audio_set']}]: {verdict} "
            f"(max running score {r['max_score']:.4f} vs threshold {threshold})"
        )
    misses = [r for r in results if r["kind"] == "SYNTH" and not r["flagged"]]
    if misses:
        print(f" {len(misses)} synthetic clip(s) not flagged - printed as FAIL but does NOT change the exit code at this stage.")
    print(bar + "\n")


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--audio_set", choices=["original", "matched", "both"], default="original",
                        help="Which audio folder to replay (default: original, the F0-comparable files).")
    parser.add_argument("--output_md", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    audio_sets = ["original", "matched"] if args.audio_set == "both" else [args.audio_set]

    try:
        clips = resolve_clips(audio_sets)
        detector = build_detector()
        f0 = load_f0_outcomes()
        results: List[Dict[str, Any]] = []
        for i, clip in enumerate(clips, 1):
            logger.info("[%d/%d] replaying %s/%s", i, len(clips), clip["audio_set"], clip["file"])
            results.append(replay_clip(detector, clip))
    except Exception as exc:
        logger.error("Streaming verification aborted (results untrustworthy): %s", exc)
        sys.exit(2)

    threshold, consecutive = results[0]["threshold"], results[0]["consecutive"]
    print_results(results, threshold, consecutive)

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_markdown(results, f0, threshold, consecutive), encoding="utf-8")
    json_path = args.output_md.with_suffix(".json")
    json_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "heads": [str(CHUNKED_WAV2VEC2_CLF.relative_to(config.BASE_DIR)), str(CHUNKED_WAVLM_CLF.relative_to(config.BASE_DIR))],
        "flag_threshold": threshold, "consecutive_flags_required": consecutive,
        "chunk_seconds": config.STREAM_CHUNK_SECONDS, "overlap_seconds": config.STREAM_OVERLAP_SECONDS,
        "results": results,
    }, indent=2), encoding="utf-8")
    logger.info("Saved %s and %s", args.output_md, json_path)

    false_positives = [r for r in results if r["kind"] == "REAL" and r["flagged"]]
    if false_positives:
        print("!" * 118, file=sys.stderr)
        print(" FALSE POSITIVE ON GENUINE HUMAN SPEECH - the failure mode this phase exists to eliminate:", file=sys.stderr)
        for r in false_positives:
            print(f"   - {r['audio_set']}/{r['file']} ({_held(r)}): flagged at {_sec(r['seconds_to_flag'])}, "
                  f"max running score {r['max_score']:.4f}", file=sys.stderr)
        print(" Do NOT hand-tune the threshold here; record it and let F4 recalibrate.", file=sys.stderr)
        print("!" * 118, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
