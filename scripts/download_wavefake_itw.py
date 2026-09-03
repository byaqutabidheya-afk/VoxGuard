#!/usr/bin/env python3
"""
download_wavefake_itw.py — Prepare WaveFake and In-the-Wild datasets.

This script manages the extraction, organization, and metadata generation for:
  1. WaveFake Dataset (Zenodo):
     - Source: https://zenodo.org/records/5642694
     - Metadata: data/metadata/wavefake.csv [filepath, label, generator]
  2. In-the-Wild Audio Deepfake Dataset (Fraunhofer AISEC / Deepfake Total):
     - Source: https://deepfake-total.com/in_the_wild
     - Metadata: data/metadata/in_the_wild.csv [filepath, label, speaker]

Since these datasets require manual download due to hosting restrictions and size,
this script provides the exact download instructions and URLs, accepts the downloaded
archives via `--src_dir`, extracts them to data/raw/, and builds canonical metadata CSVs.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent

WAVEFAKE_ZENODO_URL = "https://zenodo.org/records/5642694"
WAVEFAKE_GITHUB_URL = "https://github.com/RUB-SysSec/WaveFake"
WAVEFAKE_DEFAULT_DEST = DEFAULT_PROJECT_ROOT / "data" / "raw" / "wavefake"
WAVEFAKE_DEFAULT_CSV = DEFAULT_PROJECT_ROOT / "data" / "metadata" / "wavefake.csv"

ITW_WEBSITE_URL = "https://deepfake-total.com/in_the_wild"
ITW_KAGGLE_URL = "https://www.kaggle.com/datasets/thedevastator/in-the-wild-audio-deepfake-dataset"
ITW_HUGGINGFACE_URL = "https://huggingface.co/datasets/NicolasM/In-the-wild-audio-deepfake"
ITW_DEFAULT_DEST = DEFAULT_PROJECT_ROOT / "data" / "raw" / "in_the_wild"
ITW_DEFAULT_CSV = DEFAULT_PROJECT_ROOT / "data" / "metadata" / "in_the_wild.csv"

AUDIO_EXTENSIONS: Set[str] = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

# Known generator substrings in WaveFake folder naming convention
WAVEFAKE_GENERATOR_PATTERNS = [
    ("multi_band_melgan", "multi_band_melgan"),
    ("full_band_melgan", "full_band_melgan"),
    ("parallel_wavegan", "parallel_wavegan"),
    ("pwg", "parallel_wavegan"),
    ("melgan_large", "melgan_large"),
    ("melgan", "melgan"),
    ("hifigan", "hifiGAN"),
    ("hifi_gan", "hifiGAN"),
    ("waveglow", "waveglow"),
    ("elevenlabs", "elevenlabs"),
    ("uberduck", "uberduck"),
    ("fastspeech2", "fastspeech2"),
    ("tacotron", "tacotron"),
    ("conformer", "conformer"),
]

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
    return logging.getLogger("download_wavefake_itw")


# ---------------------------------------------------------------------------
# Extraction Helpers
# ---------------------------------------------------------------------------

def extract_archive(
    archive_path: Path,
    dest_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Extracts a zip or tar archive into dest_dir.
    """
    log = logger or logging.getLogger("download_wavefake_itw")
    dest_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Extracting '{archive_path.name}' to '{dest_dir}'...")

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    log.info(f"Extraction complete for '{archive_path.name}'.")


def resolve_source_to_destination(
    src_path: Path,
    dest_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """
    Resolves the user-provided src_dir (archive file or extracted folder)
    and ensures files are placed/available under dest_dir.
    """
    log = logger or logging.getLogger("download_wavefake_itw")

    if not src_path.exists():
        raise FileNotFoundError(f"Specified source path does not exist: {src_path}")

    if src_path.is_file():
        # Single archive file
        extract_archive(src_path, dest_dir, log)
        return dest_dir

    if src_path.is_dir():
        # Check if src_dir contains archive files (.zip, .tar.gz, etc.)
        archives = [
            p for p in src_path.iterdir()
            if p.is_file() and (zipfile.is_zipfile(p) or tarfile.is_tarfile(p))
        ]
        if archives:
            for arch in archives:
                extract_archive(arch, dest_dir, log)
            return dest_dir

        # If src_path is already an extracted folder and differs from dest_dir
        if src_path.resolve() != dest_dir.resolve():
            dest_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"Using extracted dataset directory: {src_path}")
            return src_path

        return dest_dir

    raise ValueError(f"Invalid source path: {src_path}")


# ---------------------------------------------------------------------------
# WaveFake Preparation
# ---------------------------------------------------------------------------

def parse_wavefake_generator_and_label(
    folder_name: str, file_name: str
) -> Optional[Tuple[str, str]]:
    """
    Determines the label ('bonafide' or 'spoof') and generator name from the
    WaveFake directory or file naming convention.

    WaveFake folders are typically structured as:
      - ljspeech_melgan/
      - ljspeech_melgan_large/
      - ljspeech_hifiGAN/
      - ljspeech_parallel_wavegan/
      - ljspeech_waveglow/
      - ljspeech_multi_band_melgan/
      - ljspeech_full_band_melgan/
      - jsut_multi_band_melgan/
      - jsut_parallel_wavegan/
      - ljspeech_real/ or real/ or bonafide/ (bona fide reference audio)

    Returns:
        (label, generator) tuple if recognized, or None if unrecognized.
    """
    folder_lower = folder_name.lower()
    file_lower = file_name.lower()

    # Check for bona fide / original reference speech
    if any(k in folder_lower for k in ("real", "bonafide", "original", "reference", "clean")) or \
       any(k in file_lower for k in ("_real", "_bonafide", ".r.wav", "-original")):
        return "bonafide", "original"

    # Match known generator patterns
    for pattern, canonical_name in WAVEFAKE_GENERATOR_PATTERNS:
        if pattern in folder_lower or pattern in file_lower:
            return "spoof", canonical_name

    # Do not silently guess or invent a generator name for unrecognized patterns
    return None


def download_wavefake(
    dest_dir: str | Path = WAVEFAKE_DEFAULT_DEST,
    src_dir: Optional[str | Path] = None,
    output_csv: str | Path = WAVEFAKE_DEFAULT_CSV,
    project_root: Optional[str | Path] = None,
) -> bool:
    """
    Prepares the WaveFake dataset and generates data/metadata/wavefake.csv.

    If src_dir is not provided and dest_dir contains no audio files, this function
    prints download instructions with the official Zenodo record URL and exits.

    Args:
        dest_dir: Directory where WaveFake data should reside (default: data/raw/wavefake).
        src_dir: Path to a manually downloaded archive (.zip/.tar.gz) or extracted folder.
        output_csv: Destination path for the metadata CSV (default: data/metadata/wavefake.csv).
        project_root: VoxGuard project root directory.

    Returns:
        bool: True if dataset verification and metadata generation succeeded, False otherwise.
    """
    log = setup_logging()
    dest_path = Path(dest_dir).resolve()
    out_csv_path = Path(output_csv).resolve()
    root_path = Path(project_root).resolve() if project_root else DEFAULT_PROJECT_ROOT

    log.info("=" * 70)
    log.info("WaveFake Dataset Setup & Metadata Generation")
    log.info("=" * 70)

    # 1. Print Zenodo download instructions if no source provided and dest_dir is empty
    existing_audio = list(dest_path.rglob("*.wav")) if dest_path.exists() else []
    if not src_dir and not existing_audio:
        log.warning("No --src_dir provided and no existing WaveFake audio found.")
        print("\n" + "=" * 70)
        print("MANUAL DOWNLOAD INSTRUCTIONS: WaveFake Dataset")
        print("=" * 70)
        print("WaveFake is hosted on Zenodo (open access, no login required):")
        print(f"  Zenodo Record URL : {WAVEFAKE_ZENODO_URL}")
        print(f"  GitHub Repository : {WAVEFAKE_GITHUB_URL}")
        print("\nSteps to prepare:")
        print(f"  1. Download the WaveFake archive (e.g. WaveFake.zip / wavefake_data.zip)")
        print(f"     from: {WAVEFAKE_ZENODO_URL}")
        print(f"  2. Run this script pointing to your downloaded archive:")
        print(f"     python scripts/download_wavefake_itw.py wavefake --src_dir <path-to-archive-or-folder>")
        print("=" * 70 + "\n")
        return False

    # 2. Extract or resolve dataset directory
    effective_data_dir = dest_path
    if src_dir:
        src_path = Path(src_dir).resolve()
        log.info(f"Resolving source archive / directory: {src_path}")
        effective_data_dir = resolve_source_to_destination(src_path, dest_path, log)

    # 3. Scan audio files
    audio_files: List[Path] = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(effective_data_dir.rglob(f"*{ext}"))
        audio_files.extend(effective_data_dir.rglob(f"*{ext.upper()}"))

    # Remove duplicates and sort deterministically
    audio_files = sorted(list(set(audio_files)))

    if not audio_files:
        log.error(f"No audio files ({', '.join(AUDIO_EXTENSIONS)}) found under {effective_data_dir}")
        return False

    log.info(f"Found {len(audio_files):,} audio files in WaveFake dataset.")

    # 4. Parse metadata
    records: List[Dict[str, str]] = []
    skipped_files: List[str] = []
    bonafide_count = 0
    spoof_count = 0
    generator_counts: Dict[str, int] = {}

    for audio_path in audio_files:
        parent_folder = audio_path.parent.name
        parsed = parse_wavefake_generator_and_label(parent_folder, audio_path.name)
        if parsed is None:
            log.warning(
                f"Skipping unrecognized WaveFake path (cannot determine generator/label): {audio_path}"
            )
            skipped_files.append(str(audio_path))
            continue

        label, generator = parsed

        if label == "bonafide":
            bonafide_count += 1
        else:
            spoof_count += 1
            generator_counts[generator] = generator_counts.get(generator, 0) + 1

        abs_path = audio_path.resolve().as_posix()
        records.append({
            "filepath": abs_path,
            "label": label,
            "generator": generator,
        })

    if skipped_files:
        log.warning(
            f"Skipped {len(skipped_files):,} audio files whose generator could not be reliably determined."
        )

    if not records:
        log.error("No valid WaveFake records could be parsed from the dataset directory.")
        return False

    # 5. Write metadata CSV [filepath, label, generator]
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "generator"])
        writer.writeheader()
        writer.writerows(records)

    log.info("-" * 70)
    log.info(f"WaveFake Metadata Summary:")
    log.info(f"  Total audio files : {len(records):,}")
    log.info(f"  Bonafide files    : {bonafide_count:,}")
    log.info(f"  Spoof files       : {spoof_count:,}")
    log.info(f"  Generators detected:")
    for gen, count in sorted(generator_counts.items(), key=lambda x: -x[1]):
        log.info(f"    - {gen.ljust(22)}: {count:>6,} files")
    log.info(f"[SUCCESS] Wrote metadata to {out_csv_path}")
    log.info("=" * 70)

    return True


# ---------------------------------------------------------------------------
# In-the-Wild Preparation
# ---------------------------------------------------------------------------

def parse_itw_speaker_and_label(
    audio_path: Path,
    meta_lookup: Optional[Dict[str, Tuple[str, str]]] = None,
) -> Optional[Tuple[str, str]]:
    """
    Determines the label ('bonafide' or 'spoof') and speaker identifier for an
    In-the-Wild audio file, using meta.csv if present, or falling back to file/directory naming.

    Returns:
        (label, speaker) tuple if recognized, or None if label cannot be determined.
    """
    filename = audio_path.name
    stem = audio_path.stem

    # Check preloaded meta.csv lookup
    if meta_lookup and filename in meta_lookup:
        return meta_lookup[filename]
    if meta_lookup and stem in meta_lookup:
        return meta_lookup[stem]

    # Heuristic parsing from folder and filename
    parent_lower = audio_path.parent.name.lower()
    file_lower = filename.lower()

    # Determine label explicitly without guessing
    if any(k in parent_lower for k in ("bona-fide", "bonafide", "real", "authentic")) or \
       any(k in file_lower for k in ("_real", "_bonafide", "_authentic")):
        label = "bonafide"
    elif any(k in parent_lower for k in ("spoof", "fake", "deepfake", "synthetic", "cloned")) or \
         any(k in file_lower for k in ("_fake", "_spoof", "_deepfake", "_synthetic", "_cloned")):
        label = "spoof"
    else:
        # Do not silently default unclassified files to spoof
        return None

    # Determine speaker
    parts = stem.split("_")
    if len(parts) >= 2:
        speaker = parts[0]
    else:
        speaker = audio_path.parent.name

    return label, speaker


def load_itw_meta_csv(meta_file: Path) -> Dict[str, Tuple[str, str]]:
    """
    Parses In-the-Wild dataset's official meta.csv if available.
    Returns mapping: filename -> (label, speaker).
    """
    lookup: Dict[str, Tuple[str, str]] = {}
    with open(meta_file, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Match common column naming variations
            fn = row.get("file") or row.get("filename") or row.get("path") or row.get("file_name")
            spk = row.get("speaker") or row.get("speaker_id") or row.get("name") or "unknown"
            raw_lbl = (row.get("label") or row.get("type") or row.get("class") or "").lower().strip()

            if not fn:
                continue

            fn_key = Path(fn).name
            stem_key = Path(fn).stem

            if raw_lbl in ("bona-fide", "bonafide", "real", "authentic", "0"):
                lbl = "bonafide"
            elif raw_lbl in ("spoof", "fake", "deepfake", "synthetic", "cloned", "1"):
                lbl = "spoof"
            else:
                # Do not assign a label if unrecognized in meta.csv
                continue

            lookup[fn_key] = (lbl, spk)
            lookup[stem_key] = (lbl, spk)

    return lookup


def download_in_the_wild(
    dest_dir: str | Path = ITW_DEFAULT_DEST,
    subset_size: Optional[int] = 5000,
    src_dir: Optional[str | Path] = None,
    output_csv: str | Path = ITW_DEFAULT_CSV,
    random_seed: int = 42,
    project_root: Optional[str | Path] = None,
) -> bool:
    """
    Prepares the In-the-Wild Audio Deepfake Dataset and generates data/metadata/in_the_wild.csv.

    Note on `subset_size`:
        The complete In-the-Wild dataset contains ~30GB of audio (~38 hours across 58 speakers).
        To facilitate fast development, prototyping, and resource-efficient local training,
        `subset_size` defaults to 5,000 files sampled with stratified balance across bonafide
        and spoof classes.
        
        Before final project grading, full evaluation, or production benchmarking, you should
        re-run this function with `subset_size=None` (or CLI flag `--subset_size all`) to index
        the complete dataset if sufficient disk space and processing time are available.

    Args:
        dest_dir: Directory where In-the-Wild data should reside (default: data/raw/in_the_wild).
        subset_size: Number of stratified random samples to include (default: 5000; None for full).
        src_dir: Path to a manually downloaded archive (.zip/.tar.gz) or extracted folder.
        output_csv: Destination path for the metadata CSV (default: data/metadata/in_the_wild.csv).
        random_seed: Seed for reproducible stratified sampling (default: 42).
        project_root: VoxGuard project root directory.

    Returns:
        bool: True if dataset verification and metadata generation succeeded, False otherwise.
    """
    log = setup_logging()
    dest_path = Path(dest_dir).resolve()
    out_csv_path = Path(output_csv).resolve()
    root_path = Path(project_root).resolve() if project_root else DEFAULT_PROJECT_ROOT

    log.info("=" * 70)
    log.info("In-the-Wild Dataset Setup & Metadata Generation")
    log.info("=" * 70)

    # 1. Print download instructions if no source provided and dest_dir is empty
    existing_audio = list(dest_path.rglob("*.wav")) if dest_path.exists() else []
    if not src_dir and not existing_audio:
        log.warning("No --src_dir provided and no existing In-the-Wild audio found.")
        print("\n" + "=" * 70)
        print("MANUAL DOWNLOAD INSTRUCTIONS: In-the-Wild Audio Deepfake Dataset")
        print("=" * 70)
        print("In-the-Wild Audio Deepfake Dataset (Fraunhofer AISEC / Deepfake Total):")
        print(f"  Official Portal   : {ITW_WEBSITE_URL}")
        print(f"  Kaggle Mirror     : {ITW_KAGGLE_URL}")
        print(f"  HuggingFace Mirror: {ITW_HUGGINGFACE_URL}")
        print("\nSteps to prepare:")
        print("  1. Download 'release_in_the_wild.zip' from https://deepfake-total.com/in_the_wild")
        print("     or from the Kaggle/HuggingFace mirrors above.")
        print("  2. Run this script pointing to your downloaded archive:")
        print("     python scripts/download_wavefake_itw.py in-the-wild --src_dir <path-to-archive-or-folder>")
        print("=" * 70 + "\n")
        return False

    # 2. Extract or resolve dataset directory
    effective_data_dir = dest_path
    if src_dir:
        src_path = Path(src_dir).resolve()
        log.info(f"Resolving source archive / directory: {src_path}")
        effective_data_dir = resolve_source_to_destination(src_path, dest_path, log)

    # 3. Check for meta.csv
    meta_files = list(effective_data_dir.rglob("meta.csv")) + list(dest_path.rglob("meta.csv"))
    meta_lookup: Optional[Dict[str, Tuple[str, str]]] = None
    if meta_files:
        log.info(f"Found dataset metadata file: {meta_files[0]}")
        meta_lookup = load_itw_meta_csv(meta_files[0])

    # 4. Scan audio files
    audio_files: List[Path] = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(effective_data_dir.rglob(f"*{ext}"))
        audio_files.extend(effective_data_dir.rglob(f"*{ext.upper()}"))

    audio_files = sorted(list(set(audio_files)))

    if not audio_files:
        log.error(f"No audio files ({', '.join(AUDIO_EXTENSIONS)}) found under {effective_data_dir}")
        return False

    log.info(f"Found {len(audio_files):,} audio files in In-the-Wild dataset.")

    # 5. Parse records into bonafide and spoof collections
    bonafide_records: List[Dict[str, str]] = []
    spoof_records: List[Dict[str, str]] = []
    skipped_files: List[str] = []
    speakers_seen: Set[str] = set()

    for audio_path in audio_files:
        parsed = parse_itw_speaker_and_label(audio_path, meta_lookup)
        if parsed is None:
            log.warning(
                f"Skipping In-the-Wild path with undetermined label: {audio_path}"
            )
            skipped_files.append(str(audio_path))
            continue

        label, speaker = parsed
        speakers_seen.add(speaker)
        abs_path = audio_path.resolve().as_posix()
        record = {
            "filepath": abs_path,
            "label": label,
            "speaker": speaker,
        }
        if label == "bonafide":
            bonafide_records.append(record)
        else:
            spoof_records.append(record)

    if skipped_files:
        log.warning(
            f"Skipped {len(skipped_files):,} In-the-Wild files where label could not be reliably determined."
        )

    total_available = len(bonafide_records) + len(spoof_records)
    if total_available == 0:
        log.error("No valid In-the-Wild records could be parsed from the dataset directory.")
        return False

    log.info(f"Parsed records — Total: {total_available:,} (bonafide: {len(bonafide_records):,}, spoof: {len(spoof_records):,})")

    # 6. Apply stratified subset sampling if requested
    selected_records: List[Dict[str, str]] = []
    if subset_size is not None and subset_size < total_available:
        log.info(f"Applying stratified sampling: selecting {subset_size:,} samples (seed={random_seed})...")
        rng = random.Random(random_seed)

        # Proportional allocation
        n_bonafide = int(round(subset_size * len(bonafide_records) / total_available))
        n_spoof = subset_size - n_bonafide

        # Ensure we don't exceed available counts
        n_bonafide = min(n_bonafide, len(bonafide_records))
        n_spoof = min(n_spoof, len(spoof_records))

        sampled_bonafide = rng.sample(bonafide_records, n_bonafide)
        sampled_spoof = rng.sample(spoof_records, n_spoof)

        selected_records = sampled_bonafide + sampled_spoof
        # Deterministic shuffle of combined subset
        rng.shuffle(selected_records)
        log.info(f"Stratified sample breakdown: bonafide={len(sampled_bonafide):,}, spoof={len(sampled_spoof):,}")
    else:
        selected_records = bonafide_records + spoof_records
        log.info(f"Using full In-the-Wild dataset ({len(selected_records):,} files).")

    # 7. Write metadata CSV [filepath, label, speaker]
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "speaker"])
        writer.writeheader()
        writer.writerows(selected_records)

    log.info("-" * 70)
    log.info(f"In-the-Wild Metadata Summary:")
    log.info(f"  Total records written : {len(selected_records):,}")
    log.info(f"  Distinct speakers     : {len(speakers_seen):,}")
    log.info(f"  Bonafide in CSV       : {sum(1 for r in selected_records if r['label'] == 'bonafide'):,}")
    log.info(f"  Spoof in CSV          : {sum(1 for r in selected_records if r['label'] == 'spoof'):,}")
    log.info(f"[SUCCESS] Wrote metadata to {out_csv_path}")
    log.info("=" * 70)

    return True


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and generate metadata for WaveFake and In-the-Wild datasets."
    )
    subparsers = parser.add_subparsers(dest="dataset", help="Dataset to process")

    # WaveFake subcommand
    wf_parser = subparsers.add_parser("wavefake", help="Prepare WaveFake dataset")
    wf_parser.add_argument(
        "--src_dir",
        type=str,
        default=None,
        help="Path to downloaded WaveFake archive (.zip/.tar.gz) or extracted directory",
    )
    wf_parser.add_argument(
        "--dest_dir",
        type=str,
        default=str(WAVEFAKE_DEFAULT_DEST),
        help=f"Extraction destination directory (default: {WAVEFAKE_DEFAULT_DEST})",
    )
    wf_parser.add_argument(
        "--output_csv",
        type=str,
        default=str(WAVEFAKE_DEFAULT_CSV),
        help=f"Output metadata CSV path (default: {WAVEFAKE_DEFAULT_CSV})",
    )

    # In-the-Wild subcommand
    itw_parser = subparsers.add_parser("in-the-wild", help="Prepare In-the-Wild dataset")
    itw_parser.add_argument(
        "--src_dir",
        type=str,
        default=None,
        help="Path to downloaded In-the-Wild archive (.zip/.tar.gz) or extracted directory",
    )
    itw_parser.add_argument(
        "--dest_dir",
        type=str,
        default=str(ITW_DEFAULT_DEST),
        help=f"Extraction destination directory (default: {ITW_DEFAULT_DEST})",
    )
    itw_parser.add_argument(
        "--output_csv",
        type=str,
        default=str(ITW_DEFAULT_CSV),
        help=f"Output metadata CSV path (default: {ITW_DEFAULT_CSV})",
    )
    itw_parser.add_argument(
        "--subset_size",
        type=str,
        default="5000",
        help="Number of stratified samples to select (default: 5000; set to 'all' or 'none' for full dataset)",
    )

    # All datasets subcommand
    all_parser = subparsers.add_parser("all", help="Prepare both WaveFake and In-the-Wild datasets")
    all_parser.add_argument("--wavefake_src", type=str, default=None, help="Path to WaveFake archive/folder")
    all_parser.add_argument("--itw_src", type=str, default=None, help="Path to In-the-Wild archive/folder")
    all_parser.add_argument("--itw_subset_size", type=str, default="5000", help="Subset size for In-the-Wild")

    args = parser.parse_args()

    if not args.dataset:
        parser.print_help()
        print("\nQuick Start Examples:")
        print("  python scripts/download_wavefake_itw.py wavefake --src_dir path/to/WaveFake.zip")
        print("  python scripts/download_wavefake_itw.py in-the-wild --src_dir path/to/release_in_the_wild.zip --subset_size 5000")
        sys.exit(0)

    if args.dataset == "wavefake":
        success = download_wavefake(
            dest_dir=args.dest_dir,
            src_dir=args.src_dir,
            output_csv=args.output_csv,
        )
        sys.exit(0 if success else 1)

    elif args.dataset == "in-the-wild":
        subset: Optional[int] = None
        if args.subset_size and args.subset_size.lower() not in ("all", "none", "full"):
            try:
                subset = int(args.subset_size)
            except ValueError:
                print(f"Error: Invalid --subset_size '{args.subset_size}'. Expected integer or 'all'.")
                sys.exit(1)

        success = download_in_the_wild(
            dest_dir=args.dest_dir,
            subset_size=subset,
            src_dir=args.src_dir,
            output_csv=args.output_csv,
        )
        sys.exit(0 if success else 1)

    elif args.dataset == "all":
        wf_success = download_wavefake(src_dir=args.wavefake_src)
        itw_subset: Optional[int] = None
        if args.itw_subset_size and args.itw_subset_size.lower() not in ("all", "none", "full"):
            try:
                itw_subset = int(args.itw_subset_size)
            except ValueError:
                itw_subset = 5000

        itw_success = download_in_the_wild(src_dir=args.itw_src, subset_size=itw_subset)
        sys.exit(0 if (wf_success and itw_success) else 1)


if __name__ == "__main__":
    main()
