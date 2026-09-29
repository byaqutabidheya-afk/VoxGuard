#!/usr/bin/env python3
"""
scripts/fix_rebuild_matched_hindi.py — Build a duration-matched copy of the Hindi/Hinglish track.

For every (speaker_id, sentence_id) pair in data/metadata/hindi_hinglish_track.csv
(one real + one synthetic clip), both clips are loaded at 16 kHz mono and passed
through ``duration_match_pair``, which centre-trims the longer clip to the
shorter clip's length (never pads).  Matched clips are written to
data/raw/hindi_hinglish_matched/{real,synthetic}/ with their original filenames,
and a new track CSV with the same schema and repo-relative filepaths is written to
data/metadata/hindi_hinglish_track_matched.csv.

The original corpus under data/raw/hindi_hinglish/ is read-only here — nothing
in it is modified or deleted.

Prints a before/after duration table, total seconds trimmed, and any skipped
pairs with the reason.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from voxguard import config
from voxguard.utils.audio_io import load_audio, save_audio
from voxguard.utils.duration_match import duration_match_pair
from voxguard.utils.logging_utils import get_logger

logger = get_logger("fix_rebuild_matched_hindi")

SAMPLE_RATE = 16_000
DEFAULT_INPUT_CSV = config.DATA_METADATA_DIR / "hindi_hinglish_track.csv"
DEFAULT_OUTPUT_CSV = config.DATA_METADATA_DIR / "hindi_hinglish_track_matched.csv"
DEFAULT_OUTPUT_DIR = config.DATA_RAW_DIR / "hindi_hinglish_matched"
ORIGINAL_CORPUS_DIR = config.DATA_RAW_DIR / "hindi_hinglish"
SCHEMA = ["filepath", "label", "speaker_id", "category", "sentence_id", "dataset"]
LABEL_DIRS = {"real": "real", "synthetic": "synthetic"}


def _resolve(path_str: str) -> Path:
    """Resolve a CSV filepath (repo-relative or absolute) to an absolute path."""
    p = Path(path_str)
    return p if p.is_absolute() else config.BASE_DIR / p


def _repo_relative(path: Path) -> str:
    """Repo-relative POSIX path, matching the track CSV convention."""
    return path.resolve().relative_to(config.BASE_DIR.resolve()).as_posix()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def rebuild(
    input_csv: Path,
    output_csv: Path,
    output_dir: Path,
    min_seconds: float,
) -> Tuple[pd.DataFrame, Dict[str, List[float]], List[Dict[str, Any]], float]:
    """Build the matched corpus.

    Returns:
        (matched_df, durations, skipped, total_trimmed_seconds) where
        ``durations`` has keys real_before/synth_before/real_after/synth_after.
    """
    if _is_within(output_dir, ORIGINAL_CORPUS_DIR):
        raise ValueError(
            f"Output dir {output_dir} is inside the original corpus "
            f"{ORIGINAL_CORPUS_DIR}; refusing to write there."
        )

    df = pd.read_csv(input_csv)
    missing = set(SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(f"{input_csv} is missing columns: {sorted(missing)}")

    durations: Dict[str, List[float]] = {
        "real_before": [], "synth_before": [], "real_after": [], "synth_after": [],
    }
    skipped: List[Dict[str, Any]] = []
    out_rows: List[Dict[str, Any]] = []
    total_trimmed = 0.0

    for (speaker_id, sentence_id), group in df.groupby(["speaker_id", "sentence_id"], sort=True):
        key = f"{speaker_id}/{sentence_id}"
        labels = group["label"].tolist()
        if len(group) != 2 or sorted(labels) != ["real", "synthetic"]:
            reason = f"expected exactly one real + one synthetic row, got labels={labels}"
            logger.warning("Skipping %s: %s", key, reason)
            skipped.append({"pair": key, "reason": reason})
            continue

        real_row = group[group["label"] == "real"].iloc[0]
        synth_row = group[group["label"] == "synthetic"].iloc[0]

        try:
            real_wav, _ = load_audio(_resolve(real_row["filepath"]), target_sr=SAMPLE_RATE)
            synth_wav, _ = load_audio(_resolve(synth_row["filepath"]), target_sr=SAMPLE_RATE)
            m_real, m_synth, info = duration_match_pair(
                real_wav, synth_wav, SAMPLE_RATE, min_seconds=min_seconds
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            logger.warning("Skipping %s: %s", key, reason)
            skipped.append({"pair": key, "reason": reason})
            continue

        durations["real_before"].append(info["original_real_seconds"])
        durations["synth_before"].append(info["original_synth_seconds"])
        durations["real_after"].append(info["matched_seconds"])
        durations["synth_after"].append(info["matched_seconds"])
        total_trimmed += info["trimmed_seconds"]

        for row, wav in ((real_row, m_real), (synth_row, m_synth)):
            out_path = output_dir / LABEL_DIRS[row["label"]] / Path(row["filepath"]).name
            save_audio(out_path, np.ascontiguousarray(wav), SAMPLE_RATE)
            new_row = {col: row[col] for col in SCHEMA}
            new_row["filepath"] = _repo_relative(out_path)
            out_rows.append(new_row)

        logger.debug("Matched %s: %s", key, info)

    matched_df = pd.DataFrame(out_rows, columns=SCHEMA)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    matched_df.to_csv(output_csv, index=False)
    logger.info("Wrote %d rows to %s", len(matched_df), output_csv)
    return matched_df, durations, skipped, total_trimmed


def _fmt(values: List[float]) -> Tuple[str, str]:
    if not values:
        return "n/a", "n/a"
    arr = np.asarray(values)
    return f"{arr.mean():.3f}", f"{arr.std():.3f}"


def print_report(
    durations: Dict[str, List[float]],
    skipped: List[Dict[str, Any]],
    total_trimmed: float,
    n_pairs: int,
) -> None:
    print()
    print(f"Duration-matched pairs: {n_pairs}")
    print()
    print(f"{'Stage':<8} {'Label':<10} {'N':>4} {'Mean (s)':>10} {'Std (s)':>10}")
    print("-" * 46)
    for stage in ("before", "after"):
        for label, key in (("real", "real"), ("synthetic", "synth")):
            vals = durations[f"{key}_{stage}"]
            mean, std = _fmt(vals)
            print(f"{stage.upper():<8} {label:<10} {len(vals):>4} {mean:>10} {std:>10}")
    print("-" * 46)
    print(f"Total seconds trimmed: {total_trimmed:.3f}")
    print()
    if skipped:
        print(f"Skipped pairs ({len(skipped)}):")
        for s in skipped:
            print(f"  - {s['pair']}: {s['reason']}")
    else:
        print("Skipped pairs: none")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-seconds", type=float, default=1.5)
    args = parser.parse_args()

    matched_df, durations, skipped, total_trimmed = rebuild(
        args.input_csv, args.output_csv, args.output_dir, args.min_seconds
    )
    print_report(durations, skipped, total_trimmed, len(matched_df) // 2)


if __name__ == "__main__":
    main()
