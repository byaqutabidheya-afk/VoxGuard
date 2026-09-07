#!/usr/bin/env python
"""calibrate_speaker_threshold.py — calibrate verify_speaker's decision threshold.

Usage
-----
    python scripts/calibrate_speaker_threshold.py [--enrolled-name byaquta]
        [--impostor-speakers mahato soumya] [--n-asvspoof-impostors 25]
        [--backend speechbrain]

What it does
------------
1. Loads the already-enrolled voiceprint for --enrolled-name (default: byaquta)
   via voxguard.speaker.enrollment.load_voiceprint. This script does NOT enroll
   or re-enroll anyone itself — it assumes enrollment already happened via
   scripts/enroll_speaker.py, and exits with clear instructions if it hasn't,
   rather than silently mutating a real enrollment as a side effect of running
   a calibration sweep.
2. Builds GENUINE trials from every byaquta_*.wav clip under
   data/raw/hindi_hinglish/real/, EXCEPT the clips already used for enrollment
   (byaquta_ref.wav, byaquta_neutral_01.wav, byaquta_scam_11.wav) — these are
   excluded because the voiceprint is literally the average of their
   embeddings, so scoring one of them would optimistically bias the
   genuine-trial scores (measuring "how similar is a clip to an average that
   includes itself" instead of a genuine held-out verification attempt).
3. Builds IMPOSTOR trials from every real clip belonging to
   --impostor-speakers (default: mahato, soumya — any real human who isn't the
   enrolled contact serves this purpose), plus, optionally, a sample of
   ASVspoof2019 bonafide dev clips for broader impostor diversity.
4. Extracts a speaker embedding per trial clip (same SpeakerEmbedder backend
   the enrollment used) and computes cosine similarity against the enrolled
   voiceprint.
5. Sweeps a grid of candidate thresholds, reporting for each:
     FRR (miss genuine%)   — fraction of genuine trials wrongly rejected
     FAR (accept impostor%) — fraction of impostor trials wrongly accepted
6. Prints a ranked table and marks the current
   voxguard.speaker.verify.DEFAULT_VERIFY_THRESHOLD value.
7. Prompts the user to accept or change the threshold, then updates verify.py
   — only after explicit confirmation. Never silently auto-applies, mirroring
   scripts/calibrate_thresholds.py's pattern exactly.

Which embedder?
----------------
The same SpeakerEmbedder backend the enrollment used (default: "speechbrain",
matching SpeakerEmbedder's own default and what scripts/enroll_speaker.py uses
unless overridden). Scoring with a different backend produces a different
vector space and a meaningless similarity score against the enrolled
voiceprint — pass --backend to match if the enrollment used something else.
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
from voxguard.speaker.embedding import SpeakerEmbedder  # noqa: E402
from voxguard.speaker.enrollment import load_voiceprint  # noqa: E402
from voxguard.speaker.verify import DEFAULT_VERIFY_THRESHOLD, cosine_similarity  # noqa: E402
from voxguard.utils.audio_io import load_audio  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HINDI_REAL_DIR = config.BASE_DIR / "data" / "raw" / "hindi_hinglish" / "real"
HINDI_REFERENCES_DIR = config.BASE_DIR / "data" / "raw" / "hindi_hinglish" / "references"
UNIFIED_METADATA_CSV = config.DATA_METADATA_DIR / "unified.csv"
VERIFY_PATH = _REPO_ROOT / "src" / "voxguard" / "speaker" / "verify.py"

DEFAULT_ENROLLED_NAME = "byaquta"
DEFAULT_IMPOSTOR_SPEAKERS = ["mahato", "soumya"]

# The clips used to enroll each speaker below — MUST be excluded from that
# speaker's genuine trials (see module docstring). Update this mapping if the
# canonical enrollment clips for a speaker ever change.
ENROLLMENT_CLIPS: dict[str, list[Path]] = {
    "byaquta": [
        HINDI_REFERENCES_DIR / "byaquta_ref.wav",
        HINDI_REAL_DIR / "byaquta_neutral_01.wav",
        HINDI_REAL_DIR / "byaquta_scam_11.wav",
    ],
}

# ---------------------------------------------------------------------------
# Threshold grid to sweep
# ---------------------------------------------------------------------------
CANDIDATE_THRESHOLDS: list[float] = [
    0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def ensure_enrolled(name: str) -> np.ndarray:
    """Loads the already-enrolled voiceprint for *name*, failing clearly if absent.

    This script deliberately does not enroll anyone itself — see the module
    docstring for why.
    """
    try:
        return load_voiceprint(name)
    except FileNotFoundError as exc:
        clips = ENROLLMENT_CLIPS.get(name)
        hint = (
            "\n\n  python scripts/enroll_speaker.py --name "
            + name
            + " --clips "
            + " ".join(str(p) for p in clips)
            if clips
            else ""
        )
        print(
            f"\nERROR: No enrolled voiceprint found for '{name}'.\n"
            "This script calibrates against an EXISTING enrollment — it does not "
            "enroll anyone itself, since silently overwriting a real enrollment as "
            f"a side effect of calibration would be a bad surprise.\nEnroll first:{hint}\n"
        )
        sys.exit(1)


def collect_genuine_clips(name: str) -> list[Path]:
    """All real clips for *name* under HINDI_REAL_DIR, excluding enrollment clips."""
    excluded = {p.resolve() for p in ENROLLMENT_CLIPS.get(name, []) if p.exists()}
    candidates = sorted(HINDI_REAL_DIR.glob(f"{name}_*.wav"))
    return [p for p in candidates if p.resolve() not in excluded]


def collect_impostor_clips(impostor_speakers: list[str]) -> list[Path]:
    """All real clips (+ reference clip, if present) for each impostor speaker."""
    clips: list[Path] = []
    for speaker in impostor_speakers:
        clips.extend(sorted(HINDI_REAL_DIR.glob(f"{speaker}_*.wav")))
        ref = HINDI_REFERENCES_DIR / f"{speaker}_ref.wav"
        if ref.exists():
            clips.append(ref)
    return clips


def sample_asvspoof_impostors(n: int, seed: int = 42) -> list[Path]:
    """Samples up to *n* ASVspoof2019 bonafide dev clips for impostor diversity.

    Returns an empty list (with an explanatory print) if unified.csv or the
    dev-split bonafide rows aren't available — this source is explicitly
    optional ("if easily available").
    """
    if n <= 0:
        return []
    if not UNIFIED_METADATA_CSV.exists():
        print(f"  (unified.csv not found at {UNIFIED_METADATA_CSV}; skipping ASVspoof impostors)")
        return []

    df = pd.read_csv(UNIFIED_METADATA_CSV)
    mask = (df["dataset"] == "asvspoof2019") & (df["label"] == "real") & (df["split"] == "dev")
    bonafide = df.loc[mask, "processed_path"].dropna()
    if bonafide.empty:
        print("  (no ASVspoof2019 bonafide dev clips found in unified.csv; skipping)")
        return []

    n = min(n, len(bonafide))
    sampled = bonafide.sample(n=n, random_state=seed)
    return [config.BASE_DIR / p for p in sampled]


def score_clips(
    paths: list[Path], voiceprint: np.ndarray, embedder: SpeakerEmbedder
) -> tuple[np.ndarray, list[str]]:
    """Extracts an embedding per clip and returns cosine-similarity scores.

    Clips that fail to load/embed (e.g. too short) are skipped and reported
    rather than aborting the whole run.
    """
    scores: list[float] = []
    skipped: list[str] = []
    for path in paths:
        try:
            waveform, sr = load_audio(path, target_sr=config.SAMPLE_RATE)
            embedding = embedder.extract(waveform, sr)
            scores.append(cosine_similarity(embedding, voiceprint))
        except Exception as exc:
            skipped.append(f"{path.name}: {exc}")
    return np.array(scores, dtype=np.float64), skipped


def compute_rates(genuine: np.ndarray, impostor: np.ndarray, threshold: float) -> dict[str, float]:
    """Computes false-reject and false-accept rates for one candidate threshold.

    FRR — fraction of genuine trials with similarity < threshold (wrongly rejected)
    FAR — fraction of impostor trials with similarity >= threshold (wrongly accepted)
    """
    frr = float((genuine < threshold).sum()) / len(genuine) if len(genuine) else float("nan")
    far = float((impostor >= threshold).sum()) / len(impostor) if len(impostor) else float("nan")
    return {"frr": frr, "far": far}


def build_table(genuine: np.ndarray, impostor: np.ndarray, current_threshold: float) -> pd.DataFrame:
    """Returns a DataFrame of FRR/FAR for every candidate threshold."""
    rows = []
    for threshold in CANDIDATE_THRESHOLDS:
        r = compute_rates(genuine, impostor, threshold)
        is_current = abs(threshold - current_threshold) < 1e-9
        rows.append(
            {
                "threshold": threshold,
                "FRR(miss genuine%)": round(r["frr"] * 100, 2),
                "FAR(accept impostor%)": round(r["far"] * 100, 2),
                # ASCII-only: some Windows consoles default stdout to a
                # legacy codepage (e.g. cp1252) that can't encode U+2190
                # and would crash mid-table on this exact row.
                "current": "<- current" if is_current else "",
            }
        )
    return pd.DataFrame(rows)


def print_table(df: pd.DataFrame) -> None:
    """Pretty-prints the FRR/FAR table."""
    header = f"  {'threshold':>10}  {'FRR':>12}  {'FAR':>16}  {'':>12}"
    sub = f"  {'':>10}  {'(miss gen%)':>12}  {'(accept imp%)':>16}"
    sep = "  " + "-" * (len(header) - 2)
    print(header)
    print(sub)
    print(sep)
    for _, row in df.iterrows():
        marker = str(row["current"])
        print(
            f"  {row['threshold']:>10.2f}  {row['FRR(miss genuine%)']:>12.2f}  "
            f"{row['FAR(accept impostor%)']:>16.2f}  {marker:<12}"
        )
    print()


def explain_columns() -> None:
    print(
        textwrap.dedent(
            """\
        Column guide
        ------------
        FRR (miss gen%)      - % of genuine held-out clips wrongly rejected as
                                "not a match". Lower is better for usability - a
                                real contact shouldn't be told they're an impostor.
        FAR (accept imp%)    - % of impostor clips (other real speakers) wrongly
                                accepted as a match. Lower is better for security -
                                this is exactly what an impersonation attempt needs
                                to slip past.

        Recommended selection heuristic
        --------------------------------
        Pick the threshold with the lowest FRR among those with FAR < 5% —
        security matters more here than convenience, since this gates "is this
        actually who they claim to be."
    """
        )
    )


def patch_verify_threshold(new_threshold: float) -> None:
    """Rewrites DEFAULT_VERIFY_THRESHOLD in verify.py in place."""
    text = VERIFY_PATH.read_text(encoding="utf-8")

    pattern = re.compile(r"(DEFAULT_VERIFY_THRESHOLD\s*=\s*)[\d.]+")
    new_text = pattern.sub(lambda m: f"{m.group(1)}{new_threshold}", text, count=1)

    if new_text == text:
        print(
            f"\n  WARNING: Pattern match failed — {VERIFY_PATH} was not modified.\n"
            f"  Edit DEFAULT_VERIFY_THRESHOLD manually: {new_threshold}\n"
        )
        return

    VERIFY_PATH.write_text(new_text, encoding="utf-8")
    print(f"\n  {VERIFY_PATH} updated: DEFAULT_VERIFY_THRESHOLD = {new_threshold}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--enrolled-name",
        default=DEFAULT_ENROLLED_NAME,
        help=f"Enrolled speaker to calibrate against (default: {DEFAULT_ENROLLED_NAME}).",
    )
    parser.add_argument(
        "--impostor-speakers",
        nargs="+",
        default=DEFAULT_IMPOSTOR_SPEAKERS,
        help=f"Other speakers whose clips serve as impostor trials (default: {DEFAULT_IMPOSTOR_SPEAKERS}).",
    )
    parser.add_argument(
        "--n-asvspoof-impostors",
        type=int,
        default=25,
        help="ASVspoof2019 bonafide dev clips to add for broader impostor diversity "
        "(0 to disable). Default: 25.",
    )
    parser.add_argument(
        "--backend",
        default="speechbrain",
        choices=["speechbrain", "pyannote"],
        help="SpeakerEmbedder backend — must match the one the enrollment used "
        "(default: speechbrain).",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("VoxGuard — speaker verification threshold calibration")
    print("=" * 72)

    voiceprint = ensure_enrolled(args.enrolled_name)
    print(f"\nEnrolled speaker : '{args.enrolled_name}' (voiceprint dim={voiceprint.shape[0]})")

    print("\nLoading speaker embedder...")
    embedder = SpeakerEmbedder(backend=args.backend)
    print(f"  backend={embedder.backend}")

    # ------------------------------------------------------------------
    # 1. Collect trial clips
    # ------------------------------------------------------------------
    genuine_clip_paths = collect_genuine_clips(args.enrolled_name)
    excluded = [p for p in ENROLLMENT_CLIPS.get(args.enrolled_name, []) if p.exists()]
    print(f"\nGenuine trials   : {len(genuine_clip_paths)} held-out clip(s) for '{args.enrolled_name}'")
    if excluded:
        print(f"  (excluded {len(excluded)} enrollment clip(s): " + ", ".join(p.name for p in excluded) + ")")

    impostor_clip_paths = collect_impostor_clips(args.impostor_speakers)
    print(f"\nImpostor trials  : {len(impostor_clip_paths)} clip(s) from {args.impostor_speakers}")

    asvspoof_clip_paths = sample_asvspoof_impostors(args.n_asvspoof_impostors)
    if asvspoof_clip_paths:
        print(f"                   + {len(asvspoof_clip_paths)} ASVspoof2019 bonafide dev clip(s)")
        impostor_clip_paths = impostor_clip_paths + asvspoof_clip_paths
    print(f"                   = {len(impostor_clip_paths)} total impostor trial(s)")

    # ------------------------------------------------------------------
    # 2. Score trials
    # ------------------------------------------------------------------
    print("\nScoring genuine trials...")
    genuine_scores, genuine_skipped = score_clips(genuine_clip_paths, voiceprint, embedder)
    print(f"  scored {len(genuine_scores)}/{len(genuine_clip_paths)}")
    for s in genuine_skipped:
        print(f"    skipped: {s}")

    print("\nScoring impostor trials...")
    impostor_scores, impostor_skipped = score_clips(impostor_clip_paths, voiceprint, embedder)
    print(f"  scored {len(impostor_scores)}/{len(impostor_clip_paths)}")
    for s in impostor_skipped:
        print(f"    skipped: {s}")

    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        print("\nERROR: need at least one scored trial of each type to calibrate. Aborting.")
        sys.exit(1)

    print(
        f"\nGenuine similarity  : mean={genuine_scores.mean():.4f}  "
        f"min={genuine_scores.min():.4f}  max={genuine_scores.max():.4f}"
    )
    print(
        f"Impostor similarity : mean={impostor_scores.mean():.4f}  "
        f"min={impostor_scores.min():.4f}  max={impostor_scores.max():.4f}"
    )

    # ------------------------------------------------------------------
    # 3. Build and print table
    # ------------------------------------------------------------------
    current_threshold = DEFAULT_VERIFY_THRESHOLD
    print(f"\nCurrent verify.DEFAULT_VERIFY_THRESHOLD: {current_threshold}\n")

    df = build_table(genuine_scores, impostor_scores, current_threshold)
    print(f"Candidate thresholds — cosine similarity vs. '{args.enrolled_name}'s enrolled voiceprint\n")
    print_table(df)
    explain_columns()

    # ------------------------------------------------------------------
    # 4. Recommendation: lowest FRR with FAR < 5%, tie-broken toward the
    #    stricter (safer) end when multiple thresholds perform identically.
    # ------------------------------------------------------------------
    candidates = df[df["FAR(accept impostor%)"] < 5.0].copy()
    if candidates.empty:
        candidates = df.copy()
        print("  Note: no threshold achieves FAR < 5%; showing best FRR overall.\n")

    min_frr = candidates["FRR(miss genuine%)"].min()
    tied = candidates[candidates["FRR(miss genuine%)"] == min_frr]
    # Among FRR ties, prefer the lowest FAR; among further ties (as happens
    # when several thresholds all score 0%/0% on this trial set), prefer the
    # higher/stricter threshold — it gives more safety margin against unseen
    # impostors than a threshold that merely happened to also work here.
    tied = tied.sort_values(by=["FAR(accept impostor%)", "threshold"], ascending=[True, False])
    best = tied.iloc[0]
    rec_threshold = float(best["threshold"])

    print(
        f"  Recommendation: threshold={rec_threshold:.2f}\n"
        f"    FRR={best['FRR(miss genuine%)']:.2f}%  "
        f"FAR={best['FAR(accept impostor%)']:.2f}%"
    )
    if abs(rec_threshold - current_threshold) < 1e-9:
        print("\n  The recommended threshold matches the current default — no change needed.")
    print()

    # ------------------------------------------------------------------
    # 5. Interactive confirmation
    # ------------------------------------------------------------------
    print("-" * 72)
    print("Enter the threshold value to write to verify.py, or press Enter to")
    print("accept the recommendation, or type 'q' to quit without changes.\n")

    while True:
        raw = input(f"  threshold [{rec_threshold:.2f}]: ").strip()

        if raw.lower() in ("q", "quit", "exit"):
            print("\n  Aborted — verify.py unchanged.")
            sys.exit(0)

        if raw == "":
            new_threshold = rec_threshold
        else:
            try:
                new_threshold = float(raw)
            except ValueError:
                print("  Could not parse as a float. Try again.")
                continue
            if not (0.0 < new_threshold < 1.0):
                print(f"  Invalid: threshold must be in (0.0, 1.0), got {new_threshold}")
                continue

        print(f"\n  Will write to verify.py: DEFAULT_VERIFY_THRESHOLD = {new_threshold}")
        confirm = input("  Confirm? [y/N]: ").strip().lower()
        if confirm in ("y", "yes"):
            patch_verify_threshold(new_threshold)
            break
        else:
            print("  Not confirmed — try again or type 'q' to quit.\n")


if __name__ == "__main__":
    main()
