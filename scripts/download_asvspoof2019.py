#!/usr/bin/env python3
"""
download_asvspoof2019.py — Prepare and verify ASVspoof2019 LA dataset.

This script verifies an existing local copy of the ASVspoof 2019 Logical Access
(LA) dataset, validates official protocol counts for train/dev/eval splits,
verifies that all corresponding FLAC audio files exist, and generates the
canonical metadata CSV at data/metadata/asvspoof2019.csv.

Official CM Protocol Counts:
  - Train: 25,380 utterances
  - Dev:   24,844 utterances
  - Eval:  71,237 utterances
  - Total: 121,461 utterances

Protocol line format (5 space-separated tokens):
  <speaker_id> <utterance_id> <environment> <system_id> <label>
  e.g.:
    LA_0079 LA_T_1138215 - - bonafide
    LA_0039 LA_E_2834763 - A11 spoof

Generated metadata CSV format:
  filepath,speaker_id,system_id,label
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants & Split Specifications
# ---------------------------------------------------------------------------

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLIT_CONFIGS = [
    {
        "split": "train",
        "protocol_filename": "ASVspoof2019.LA.cm.train.trn.txt",
        "audio_subdir": "ASVspoof2019_LA_train/flac",
        "expected_count": 25380,
    },
    {
        "split": "dev",
        "protocol_filename": "ASVspoof2019.LA.cm.dev.trl.txt",
        "audio_subdir": "ASVspoof2019_LA_dev/flac",
        "expected_count": 24844,
    },
    {
        "split": "eval",
        "protocol_filename": "ASVspoof2019.LA.cm.eval.trl.txt",
        "audio_subdir": "ASVspoof2019_LA_eval/flac",
        "expected_count": 71237,
    },
]

CSV_HEADER = ["filepath", "speaker_id", "system_id", "label"]


class ProtocolRecord(NamedTuple):
    filepath: str
    speaker_id: str
    system_id: str
    label: str
    split: str


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = True) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("download_asvspoof2019")


# ---------------------------------------------------------------------------
# Parsing and Verification Helpers
# ---------------------------------------------------------------------------

def parse_protocol_file(
    protocol_path: Path,
    flac_dir: Path,
    split_name: str,
    project_root: Path,
    check_audio_exists: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[ProtocolRecord], List[str]]:
    """
    Parses a single ASVspoof 2019 LA CM protocol file and resolves audio filepaths.

    Returns:
        records: List of parsed ProtocolRecord objects.
        missing_files: List of filepaths where the audio file was not found on disk.
    """
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")

    records: List[ProtocolRecord] = []
    missing_files: List[str] = []

    with open(protocol_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            tokens = line.split()
            if len(tokens) != 5:
                raise ValueError(
                    f"Invalid protocol format in {protocol_path.name}:{line_num} "
                    f"— expected 5 tokens, got {len(tokens)}: '{line}'"
                )

            speaker_id, utterance_id, _, system_id, label = tokens

            if label not in ("bonafide", "spoof"):
                raise ValueError(
                    f"Unexpected label '{label}' in {protocol_path.name}:{line_num}"
                )

            flac_file = (flac_dir / f"{utterance_id}.flac").resolve()

            if check_audio_exists and not flac_file.is_file():
                missing_files.append(str(flac_file))

            # Store absolute path to the FLAC file using forward slashes
            abs_path = flac_file.as_posix()

            records.append(
                ProtocolRecord(
                    filepath=abs_path,
                    speaker_id=speaker_id,
                    system_id=system_id,
                    label=label,
                    split=split_name,
                )
            )

    return records, missing_files


def prepare_asvspoof2019(
    la_dir: Path,
    output_csv: Path,
    project_root: Path,
    check_audio: bool = True,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    Main preparation and verification routine.
    """
    log = logger or logging.getLogger("download_asvspoof2019")
    log.info("=" * 70)
    log.info("ASVspoof 2019 LA Dataset Preparation & Verification")
    log.info("=" * 70)
    log.info(f"LA Root Directory : {la_dir}")
    log.info(f"Project Root      : {project_root}")
    log.info(f"Output CSV Path   : {output_csv}")

    if not la_dir.exists():
        log.error(
            f"Dataset directory does not exist: {la_dir}\n"
            "Please ensure ASVspoof2019 LA is extracted to the expected path."
        )
        return False

    protocols_dir = la_dir / "ASVspoof2019_LA_cm_protocols"
    if not protocols_dir.exists():
        log.error(f"CM Protocols directory missing: {protocols_dir}")
        return False

    all_records: List[ProtocolRecord] = []
    total_missing: List[str] = []
    verification_passed = True

    split_stats: Dict[str, Dict[str, int]] = {}

    for config in SPLIT_CONFIGS:
        split = config["split"]
        proto_file = protocols_dir / config["protocol_filename"]
        flac_dir = la_dir / config["audio_subdir"]
        expected_count = config["expected_count"]

        log.info("-" * 70)
        log.info(f"Processing split: {split.upper()}")
        log.info(f"  Protocol file : {proto_file.name}")
        log.info(f"  Audio dir     : {flac_dir}")

        if not proto_file.exists():
            log.error(f"  [FAIL] Missing protocol file: {proto_file}")
            verification_passed = False
            continue

        if not flac_dir.exists():
            log.error(f"  [FAIL] Missing audio directory: {flac_dir}")
            verification_passed = False
            continue

        records, missing = parse_protocol_file(
            protocol_path=proto_file,
            flac_dir=flac_dir,
            split_name=split,
            project_root=project_root,
            check_audio_exists=check_audio,
            logger=log,
        )

        actual_count = len(records)
        bonafide_count = sum(1 for r in records if r.label == "bonafide")
        spoof_count = sum(1 for r in records if r.label == "spoof")

        split_stats[split] = {
            "total": actual_count,
            "bonafide": bonafide_count,
            "spoof": spoof_count,
            "missing": len(missing),
        }

        log.info(
            f"  Count check   : {actual_count:,} / {expected_count:,} expected "
            f"({'MATCH' if actual_count == expected_count else 'MISMATCH'})"
        )
        log.info(f"  Class breakdown: bonafide={bonafide_count:,}, spoof={spoof_count:,}")

        if actual_count != expected_count:
            log.warning(
                f"  [WARN] Split '{split}' count ({actual_count}) does not match "
                f"expected official count ({expected_count})!"
            )
            verification_passed = False

        if missing:
            log.error(f"  [FAIL] {len(missing):,} audio files referenced in protocol were not found on disk!")
            total_missing.extend(missing[:10])  # Log first few
            verification_passed = False
        else:
            log.info(f"  Audio check   : All {actual_count:,} FLAC files verified on disk.")

        all_records.extend(records)

    log.info("=" * 70)
    log.info("Summary of Splits:")
    for split, stats in split_stats.items():
        log.info(
            f"  {split.ljust(6)}: {stats['total']:>7,} files "
            f"(bonafide: {stats['bonafide']:>6,}, spoof: {stats['spoof']:>6,}, "
            f"missing: {stats['missing']:>3})"
        )
    log.info(f"  TOTAL : {len(all_records):>7,} files across all splits")
    log.info("=" * 70)

    if total_missing:
        log.error("Sample missing files:")
        for mf in total_missing[:10]:
            log.error(f"  - {mf}")

    # Write output CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Writing metadata to {output_csv}...")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for record in all_records:
            writer.writerow([
                record.filepath,
                record.speaker_id,
                record.system_id,
                record.label,
            ])

    log.info(f"[SUCCESS] Wrote {len(all_records):,} records to {output_csv}")

    if verification_passed:
        log.info("[OK] All official CM protocol counts and audio files verified successfully.")
    else:
        log.warning("[WARNING] Dataset processed with warnings/mismatches noted above.")

    return verification_passed


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify and prepare ASVspoof2019 LA dataset metadata."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_PROJECT_ROOT / "data" / "raw" / "asvspoof2019" / "LA",
        help="Path to the extracted ASVspoof2019 LA directory (default: data/raw/asvspoof2019/LA)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_PROJECT_ROOT / "data" / "metadata" / "asvspoof2019.csv",
        help="Path to output metadata CSV (default: data/metadata/asvspoof2019.csv)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Project root directory for relative filepath resolution",
    )
    parser.add_argument(
        "--skip-audio-check",
        action="store_true",
        help="Skip checking existence of each individual audio file on disk",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose log messages",
    )

    args = parser.parse_args()
    logger = setup_logging(verbose=not args.quiet)

    success = prepare_asvspoof2019(
        la_dir=args.data_dir.resolve(),
        output_csv=args.output_csv.resolve(),
        project_root=args.project_root.resolve(),
        check_audio=not args.skip_audio_check,
        logger=logger,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
