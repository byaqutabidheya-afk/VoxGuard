#!/usr/bin/env python3
"""
organize_hindi_recordings.py — Ingest and organize Phase 4's self-recorded
Hindi/Hinglish speech (BuildGuide.md, Phase 4, Prompt 4.1).

Takes raw recordings placed by hand under --input_dir, following the
``{speaker_id}_{category}_{sentence_id}.wav`` naming convention (e.g.
``priya_neutral_03.wav``, ``priya_scam_11.wav``), converts each to 16kHz
mono WAV via the existing voxguard.utils.audio_io helpers (no audio-loading
logic is duplicated here), and writes:

  - data/raw/hindi_hinglish/real/{speaker_id}_{category}_{sentence_id}.wav
  - data/metadata/hindi_hinglish_real.csv

Consent handling: every row's ``consent_confirmed`` defaults to False. This
script never sets it to True on its own — after confirming consent with a
speaker, open hindi_hinglish_real.csv and flip that speaker's rows to True
by hand. Re-running this script (e.g. after adding more speakers) preserves
any consent already recorded in the existing CSV rather than resetting it,
so recording more speakers can never silently revert an earlier speaker's
confirmed consent back to False.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from voxguard import config
from voxguard.utils.audio_io import load_audio, save_audio
from voxguard.utils.logging_utils import get_logger

logger = get_logger("organize_hindi_recordings")

OUTPUT_AUDIO_DIR = config.DATA_RAW_DIR / "hindi_hinglish" / "real"
OUTPUT_METADATA_CSV = config.DATA_METADATA_DIR / "hindi_hinglish_real.csv"

# The 25-sentence reading script (BuildGuide.md, Phase 4), verbatim.
SENTENCES: Dict[int, str] = {
    # Category A — Neutral/Casual Conversation (01-10)
    1: "Yaar, aaj office mein bahut kaam tha, I'm so tired now.",
    2: "Mummy ne bola ki dinner ready hai, chalo khaane baith jaate hain.",
    3: "Kal weekend hai na, let's plan a trip to the hills.",
    4: "Mera phone ka battery bahut fast drain ho raha hai these days.",
    5: "Traffic itna zyada tha ki main meeting ke liye late ho gaya.",
    6: "Tumne wo new web series dekhi kya, it's actually really good.",
    7: "Is mahine ka budget thoda tight hai, we need to cut down on eating out.",
    8: "Doctor ne kaha hai ki mujhe zyada paani peena chahiye, and exercise daily.",
    9: "Weather bahut accha hai aaj, chalo evening walk pe chalte hain.",
    10: "Project deadline agle Friday hai, I think we can manage it easily.",
    # Category B — Scam-Pattern Speech (11-20)
    11: "Sir, aapka bank account today block ho jaayega agar aap abhi apna OTP share nahi karte.",
    12: "Yeh customs department se call hai, aapke parcel mein illegal items mile hain, turant fine pay kijiye.",
    13: "Beta, main tumhari maa bol rahi hoon, mujhe abhi ke abhi paise chahiye, kisi ko mat batana.",
    14: "Police station se bol raha hoon, aapke naam par ek warrant issue hua hai, abhi arrest ho sakta hai.",
    15: "Aapka UPI account verify karna zaroori hai warna aapka paisa freeze ho jayega, please share your PIN now.",
    16: "This is an urgent matter sir, agar aap abhi payment nahi karte to legal action liya jayega.",
    17: "Aapko lottery mein paanch lakh rupaye jeete hain, bas processing fee ke liye account details bhejiye.",
    18: "Please don't tell your family about this call, yeh confidential matter hai.",
    19: "Aapka credit card suspicious activity ke wajah se block kar diya gaya hai, verify karne ke liye apna CVV batayein.",
    20: "Abhi turant is number par paise transfer kijiye warna aapki service disconnect ho jayegi.",
    # Category C — Neutral Phone-Call Style (21-25)
    21: "Hello, main XYZ bank se bol raha hoon, aapka statement email par bhej diya gaya hai.",
    22: "Aapka order successfully deliver ho gaya hai, kripya feedback share kijiye.",
    23: "Beta, ghar kab aa rahe ho, dinner ready hai.",
    24: "Hi, this is a reminder call for your appointment tomorrow at 10 AM.",
    25: "Aapka recharge successful ho gaya hai, thank you for using our services.",
}

VALID_CATEGORIES = ("neutral", "scam", "control")

# Which category each sentence_id is expected to belong to, per the reading
# script — used only to warn on a likely filename typo, not to override
# whatever category the filename actually says.
EXPECTED_CATEGORY_BY_SENTENCE_ID: Dict[int, str] = {
    **{i: "neutral" for i in range(1, 11)},
    **{i: "scam" for i in range(11, 21)},
    **{i: "control" for i in range(21, 26)},
}

FILENAME_PATTERN = re.compile(
    r"^(?P<speaker_id>.+)_(?P<category>neutral|scam|control)_(?P<sentence_id>\d{1,2})$",
    re.IGNORECASE,
)


def _load_existing_consent(csv_path: Path) -> Dict[str, bool]:
    """Reads speaker_id -> consent_confirmed from a prior run's CSV, if any.

    Preserves already-confirmed consent across re-runs: recording additional
    speakers later must never reset an earlier speaker's confirmed consent
    back to False.
    """
    if not csv_path.exists():
        return {}
    try:
        existing = pd.read_csv(csv_path)
    except Exception as exc:
        logger.warning("Could not read existing %s (%s); starting fresh.", csv_path, exc)
        return {}
    if "speaker_id" not in existing.columns or "consent_confirmed" not in existing.columns:
        return {}
    return (
        existing.groupby("speaker_id")["consent_confirmed"]
        .any()
        .astype(bool)
        .to_dict()
    )


def _parse_filename(path: Path) -> Optional[dict]:
    """Parses a recording's filename into speaker_id/category/sentence_id.

    Returns ``None`` (after logging a warning) if the filename doesn't match
    the ``{speaker_id}_{category}_{sentence_id}.wav`` convention.
    """
    match = FILENAME_PATTERN.match(path.stem)
    if not match:
        logger.warning(
            "Skipping '%s': doesn't match {speaker_id}_{category}_{sentence_id}.wav "
            "(category must be one of %s).",
            path.name,
            VALID_CATEGORIES,
        )
        return None

    speaker_id = match.group("speaker_id")
    category = match.group("category").lower()
    sentence_id = int(match.group("sentence_id"))

    if sentence_id not in SENTENCES:
        logger.warning(
            "Skipping '%s': sentence_id %d is outside the 25-sentence script (1-25).",
            path.name,
            sentence_id,
        )
        return None

    expected_category = EXPECTED_CATEGORY_BY_SENTENCE_ID[sentence_id]
    if category != expected_category:
        logger.warning(
            "'%s': category '%s' in filename doesn't match sentence %d's expected "
            "category '%s' in the reading script. Using the filename's category — "
            "double-check this isn't a typo.",
            path.name,
            category,
            sentence_id,
            expected_category,
        )

    return {"speaker_id": speaker_id, "category": category, "sentence_id": sentence_id}


def organize_recordings(input_dir: Path, force: bool = False) -> pd.DataFrame:
    """Converts and catalogs every recording under *input_dir*.

    Parameters
    ----------
    input_dir:
        Directory of manually-placed raw recordings.
    force:
        If True, re-convert files even if the destination WAV already
        exists. Otherwise, matches the project's usual resumable behavior
        (voxguard.utils.preprocess.preprocess_dataset) — an existing
        destination is left untouched and its duration is measured directly
        rather than re-decoding the source.

    Returns
    -------
    pd.DataFrame
        Columns: [filepath, speaker_id, category, sentence_id, sentence_text,
        consent_confirmed]. ``filepath`` is repo-relative.
    """
    if not input_dir.is_dir():
        raise FileNotFoundError(f"--input_dir does not exist or is not a directory: {input_dir}")

    OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    existing_consent = _load_existing_consent(OUTPUT_METADATA_CSV)

    # Match .wav/.WAV/.Wav etc., but de-duplicate by resolved path: on a
    # case-insensitive filesystem (Windows, default macOS), glob("*.wav")
    # and glob("*.WAV") both match every file, which would otherwise flag
    # every single recording as a spurious duplicate of itself.
    candidates = sorted(
        {p.resolve() for p in input_dir.iterdir() if p.suffix.lower() == ".wav"}
    )
    if not candidates:
        raise FileNotFoundError(f"No .wav files found under {input_dir}")

    seen_keys: Dict[tuple, Path] = {}
    rows: List[dict] = []

    for src_path in candidates:
        parsed = _parse_filename(src_path)
        if parsed is None:
            continue

        key = (parsed["speaker_id"], parsed["category"], parsed["sentence_id"])
        if key in seen_keys:
            logger.warning(
                "Duplicate recording for speaker=%s category=%s sentence_id=%02d: "
                "keeping '%s', skipping '%s'.",
                *key,
                seen_keys[key].name,
                src_path.name,
            )
            continue
        seen_keys[key] = src_path

        dest_name = f"{parsed['speaker_id']}_{parsed['category']}_{parsed['sentence_id']:02d}.wav"
        dest_path = OUTPUT_AUDIO_DIR / dest_name

        if dest_path.exists() and not force:
            duration = _duration_of(dest_path)
        else:
            waveform, sr = load_audio(src_path, target_sr=config.SAMPLE_RATE)
            save_audio(dest_path, waveform, sr=sr)
            duration = len(waveform) / sr

        rel_path = dest_path.relative_to(config.BASE_DIR).as_posix()
        rows.append(
            {
                "filepath": rel_path,
                "speaker_id": parsed["speaker_id"],
                "category": parsed["category"],
                "sentence_id": parsed["sentence_id"],
                "sentence_text": SENTENCES[parsed["sentence_id"]],
                "consent_confirmed": bool(existing_consent.get(parsed["speaker_id"], False)),
                "_duration_s": duration,
            }
        )

    if not rows:
        raise ValueError(f"No validly-named recordings found under {input_dir}")

    df = pd.DataFrame(rows).sort_values(["speaker_id", "sentence_id"]).reset_index(drop=True)
    return df


def _duration_of(path: Path) -> float:
    """Duration in seconds, read from the file header (no full decode)."""
    from voxguard.utils.audio_io import get_duration_seconds

    return get_duration_seconds(path)


def print_summary(df: pd.DataFrame) -> None:
    """Prints per-speaker, per-category, and total-duration summaries."""
    total_duration = df["_duration_s"].sum()

    print("=" * 78)
    print("HINDI/HINGLISH RECORDINGS SUMMARY")
    print("=" * 78)

    print("\nClips per speaker:")
    for speaker_id, count in df["speaker_id"].value_counts().sort_index().items():
        speaker_duration = df.loc[df["speaker_id"] == speaker_id, "_duration_s"].sum()
        print(f"  {speaker_id:<20} {count:3d} clips   {speaker_duration:7.1f}s")

    print("\nClips per category:")
    for category in VALID_CATEGORIES:
        count = int((df["category"] == category).sum())
        print(f"  {category:<20} {count:3d} clips")

    print(f"\nTotal clips    : {len(df)}")
    print(f"Total duration : {total_duration:.1f}s ({total_duration / 60:.1f} min)")

    unconfirmed = sorted(
        df.loc[~df["consent_confirmed"], "speaker_id"].unique().tolist()
    )
    print("\n" + "-" * 78)
    if unconfirmed:
        print(
            "WARNING: consent NOT confirmed for the following speaker(s) — their "
            "clips must NOT be used (training, cloning, demo, or otherwise) until "
            "this is resolved:"
        )
        for speaker_id in unconfirmed:
            print(f"    - {speaker_id}")
        print(
            f"\n  To confirm: after getting explicit consent, open {OUTPUT_METADATA_CSV}"
            " and set consent_confirmed=True for that speaker's rows."
        )
    else:
        print("All speakers have consent_confirmed=True.")
    print("-" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest and organize Phase 4 Hindi/Hinglish recordings into the "
        "project's raw-audio and metadata layout."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory of raw recordings named {speaker_id}_{category}_{sentence_id}.wav",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-convert files even if the destination WAV already exists.",
    )
    args = parser.parse_args()

    try:
        df = organize_recordings(args.input_dir, force=args.force)
    except Exception as exc:
        logger.error("Failed to organize recordings: %s", exc)
        sys.exit(1)

    OUTPUT_METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["_duration_s"]).to_csv(OUTPUT_METADATA_CSV, index=False)
    logger.info("Saved metadata (%d rows) to %s", len(df), OUTPUT_METADATA_CSV)

    print_summary(df)


if __name__ == "__main__":
    main()
