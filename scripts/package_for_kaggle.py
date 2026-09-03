#!/usr/bin/env python3
"""
package_for_kaggle.py — Package and upload preprocessed VoxGuard audio and metadata to Kaggle.

This script structures the preprocessed datasets into one combined Kaggle Dataset:
    <staging_dir>/
      asvspoof2019/       # Preprocessed 16kHz ASVspoof 2019 audio
        *.wav
      wavefake/          # Preprocessed 16kHz WaveFake subset audio (named 'wavefake', not 'wavefake_subset')
        *.wav
      in_the_wild/       # Preprocessed 16kHz In-the-Wild audio
        *.wav
      metadata/
        unified.csv      # Direct copy of local data/metadata/unified.csv (with processed_path column)

Why one combined dataset:
  - Phase 2's Master Kaggle Session requires only a SINGLE "Add Input" step to mount
    all three dataset scopes at `/kaggle/input/<dataset-slug>/`.
  - No separate metadata CSVs per subfolder; the single `metadata/unified.csv` simplifies
    loading on Kaggle without needing multiple concatenation steps.

Usage:
  # Create a new private Kaggle dataset
  python scripts/package_for_kaggle.py --slug <username>/voxguard-preprocessed-data

  # Push a new version after updates (e.g. after adding Hindi/Hinglish track)
  python scripts/package_for_kaggle.py --slug <username>/voxguard-preprocessed-data --update

  # Stage only without uploading (dry run)
  python scripts/package_for_kaggle.py --stage-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

from voxguard.config import BASE_DIR, DATA_METADATA_DIR, DATA_PROCESSED_DIR
from voxguard.utils.logging_utils import get_logger

# Import upload_dataset from kaggle_sync helper
try:
    from scripts.kaggle_sync import upload_dataset
except ImportError:
    sys.path.insert(0, str(BASE_DIR))
    from scripts.kaggle_sync import upload_dataset

logger = get_logger("package_for_kaggle")

DEFAULT_DATASET_NAME = "voxguard-preprocessed-data"
DEFAULT_TITLE = "VoxGuard Preprocessed Audio and Unified Metadata"
DEFAULT_STAGING_DIR = BASE_DIR / "data" / "kaggle_package"

# Canonical subfolder names required for downstream Phase 2+ compatibility
CANONICAL_DATASET_MAPPING = {
    "asvspoof": "asvspoof2019",
    "asvspoof2019": "asvspoof2019",
    "asv_2019": "asvspoof2019",
    "wavefake": "wavefake",
    "wavefake_subset": "wavefake",
    "wf": "wavefake",
    "in_the_wild": "in_the_wild",
    "in-the-wild": "in_the_wild",
    "itw": "in_the_wild",
}


def get_default_kaggle_username() -> Optional[str]:
    """Attempts to read the Kaggle username from ~/.kaggle/kaggle.json."""
    token_path = Path.home() / ".kaggle" / "kaggle.json"
    if token_path.exists():
        try:
            creds = json.loads(token_path.read_text(encoding="utf-8"))
            return creds.get("username")
        except Exception:
            pass
    return None


def resolve_dataset_slug(slug_arg: Optional[str]) -> str:
    """
    Resolves the Kaggle dataset slug into 'owner/dataset-name' format.
    """
    if slug_arg:
        if "/" in slug_arg:
            return slug_arg.strip()
        username = get_default_kaggle_username()
        if username:
            return f"{username}/{slug_arg.strip()}"
        return slug_arg.strip()

    username = get_default_kaggle_username()
    if username:
        return f"{username}/{DEFAULT_DATASET_NAME}"

    raise ValueError(
        "Kaggle username could not be automatically detected. "
        "Please provide --slug in 'owner/dataset-name' format (e.g. --slug alice/voxguard-preprocessed-data)."
    )


def canonicalize_dataset_folder_name(folder_name: str) -> str:
    """Maps dataset folder names or subsets to standard canonical names."""
    clean = folder_name.strip().lower().replace("-", "_")
    return CANONICAL_DATASET_MAPPING.get(clean, clean)


def compute_dir_stats(directory: Path) -> Tuple[int, float]:
    """Computes total file count and size in megabytes for a directory."""
    files = [f for f in directory.rglob("*") if f.is_file()]
    total_bytes = sum(f.stat().st_size for f in files)
    return len(files), total_bytes / (1024 * 1024)


def prepare_kaggle_package(
    processed_dir: Path = DATA_PROCESSED_DIR,
    metadata_csv: Path = DATA_METADATA_DIR / "unified.csv",
    staging_dir: Path = DEFAULT_STAGING_DIR,
) -> Path:
    """
    Assembles processed audio folders and metadata/unified.csv into one combined staging directory.

    Structure created:
      <staging_dir>/
        asvspoof2019/
          *.wav
        wavefake/
          *.wav
        in_the_wild/
          *.wav
        metadata/
          unified.csv

    Args:
        processed_dir: Directory containing preprocessed audio subfolders.
        metadata_csv: Path to unified.csv metadata file.
        staging_dir: Destination packaging directory.

    Returns:
        Path: Resolved staging directory path.
    """
    proc_path = Path(processed_dir).resolve()
    meta_path = Path(metadata_csv).resolve()
    stage_path = Path(staging_dir).resolve()

    if not proc_path.exists():
        raise FileNotFoundError(f"Processed audio directory not found at: {proc_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Unified metadata CSV not found at: {meta_path}")

    logger.info(f"Preparing Kaggle combined package staging directory: {stage_path}")
    stage_path.mkdir(parents=True, exist_ok=True)

    # 1. Stage metadata/unified.csv (single combined metadata file)
    meta_subfolder = stage_path / "metadata"
    meta_subfolder.mkdir(parents=True, exist_ok=True)
    target_meta_file = meta_subfolder / "unified.csv"
    shutil.copy2(meta_path, target_meta_file)
    logger.info(f"Staged metadata -> {target_meta_file.relative_to(stage_path).as_posix()}")

    # 2. Stage processed audio folders with canonical naming
    dataset_dirs = [p for p in proc_path.iterdir() if p.is_dir() and p.name != "metadata"]
    package_stats: Dict[str, Tuple[int, float]] = {}

    for ds_dir in dataset_dirs:
        canonical_name = canonicalize_dataset_folder_name(ds_dir.name)
        dest_ds_dir = stage_path / canonical_name
        dest_ds_dir.mkdir(parents=True, exist_ok=True)

        audio_files = list(ds_dir.glob("*.wav"))
        logger.info(
            f"Staging {len(audio_files):,} audio clips from '{ds_dir.name}' into '{canonical_name}/'..."
        )

        for audio_file in audio_files:
            target_audio = dest_ds_dir / audio_file.name
            if not target_audio.exists():
                try:
                    os.link(audio_file, target_audio)
                except Exception:
                    shutil.copy2(audio_file, target_audio)

        cnt, mb = compute_dir_stats(dest_ds_dir)
        package_stats[canonical_name] = (cnt, mb)

    meta_cnt, meta_mb = compute_dir_stats(meta_subfolder)
    package_stats["metadata"] = (meta_cnt, meta_mb)

    # Print summary table
    logger.info("-" * 78)
    logger.info("COMBINED KAGGLE PACKAGE BREAKDOWN:")
    total_files = sum(s[0] for s in package_stats.values())
    total_mb = sum(s[1] for s in package_stats.values())
    total_gb = total_mb / 1024.0

    for name, (cnt, mb) in sorted(package_stats.items()):
        if name == "metadata":
            logger.info(f"  - {name.ljust(16)}: {cnt:>7,} file(s)  ({mb:>8.2f} MB)")
        else:
            logger.info(f"  - {name.ljust(16)}: {cnt:>7,} wavs     ({mb:>8.2f} MB)")
    logger.info(f"  TOTAL PACKAGE  : {total_files:>7,} file(s)  ({total_gb:>8.2f} GB)")
    logger.info("-" * 78)

    return stage_path


def package_and_upload(
    processed_dir: Path = DATA_PROCESSED_DIR,
    metadata_csv: Path = DATA_METADATA_DIR / "unified.csv",
    staging_dir: Path = DEFAULT_STAGING_DIR,
    dataset_slug: Optional[str] = None,
    title: str = DEFAULT_TITLE,
    update: bool = False,
    stage_only: bool = False,
) -> str:
    """
    Main orchestration routine for staging and uploading dataset to Kaggle.
    """
    # 1. Prepare packaging directory
    staged_path = prepare_kaggle_package(
        processed_dir=processed_dir,
        metadata_csv=metadata_csv,
        staging_dir=staging_dir,
    )

    if stage_only:
        logger.info("-" * 78)
        logger.info(f"[STAGE ONLY] Package prepared at {staged_path}. Upload skipped.")
        logger.info("-" * 78)
        return str(staged_path)

    # 2. Resolve dataset slug
    slug = resolve_dataset_slug(dataset_slug)
    dataset_id = slug.split("/")[-1] if "/" in slug else slug
    dataset_url = f"https://www.kaggle.com/datasets/{slug}"
    mount_path = f"/kaggle/input/{dataset_id}/"

    logger.info("=" * 78)
    logger.info("UPLOADING COMBINED DATASET TO KAGGLE")
    logger.info(f"Target Slug : {slug}")
    logger.info(f"Title       : {title}")
    logger.info(f"Update Mode : {update}")
    logger.info(f"Source Dir  : {staged_path}")
    logger.info("=" * 78)

    # 3. Call kaggle_sync upload_dataset
    upload_dataset(
        local_dir=str(staged_path),
        dataset_slug=slug,
        title=title,
        update=update,
    )

    # 4. Print prominent instructions for Phase 2 Master Session
    print("\n" + "=" * 78)
    print("KAGGLE DATASET UPLOAD COMPLETE!")
    print("=" * 78)
    print(f"  Dataset Slug : {slug}")
    print(f"  Dataset URL  : {dataset_url}")
    print(f"  Mount Point  : {mount_path}")
    print("\nIMPORTANT FOR MASTER KAGGLE GPU SESSION (Phase 2):")
    print(f"  1. Open your Phase 2 Master Kaggle GPU Notebook.")
    print(f"  2. Single 'Add Input' step -> 'Your Datasets' -> Attach '{slug}'.")
    print(f"  3. All three datasets mount under: {mount_path}")
    print(f"       - {mount_path}asvspoof2019/")
    print(f"       - {mount_path}wavefake/")
    print(f"       - {mount_path}in_the_wild/")
    print(f"       - {mount_path}metadata/unified.csv")
    print("  4. Save this dataset slug for all GPU notebook extractions.")
    print("=" * 78 + "\n")

    return dataset_url


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package preprocessed VoxGuard datasets into one combined private Kaggle Dataset."
    )
    parser.add_argument(
        "--slug",
        type=str,
        default=None,
        help="Kaggle dataset slug (format: 'owner/dataset-name', e.g. 'alice/voxguard-preprocessed-data')",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=DEFAULT_TITLE,
        help=f"Dataset title or version note (default: '{DEFAULT_TITLE}')",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DATA_PROCESSED_DIR,
        help=f"Path to preprocessed audio directory (default: {DATA_PROCESSED_DIR})",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=DATA_METADATA_DIR / "unified.csv",
        help=f"Path to unified metadata CSV (default: {DATA_METADATA_DIR / 'unified.csv'})",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=DEFAULT_STAGING_DIR,
        help=f"Packaging staging directory (default: {DEFAULT_STAGING_DIR})",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Push a new version of an existing dataset instead of creating a new one",
    )
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="Assemble the staging directory without executing the Kaggle upload",
    )

    args = parser.parse_args()

    try:
        package_and_upload(
            processed_dir=args.processed_dir,
            metadata_csv=args.metadata_csv,
            staging_dir=args.staging_dir,
            dataset_slug=args.slug,
            title=args.title,
            update=args.update,
            stage_only=args.stage_only,
        )
    except Exception as exc:
        logger.error(f"Packaging and upload failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
