#!/usr/bin/env python3
"""
package_for_kaggle.py — Package and upload preprocessed VoxGuard audio and metadata to Kaggle.

This script structures the preprocessed dataset into a clean, self-describing directory:
    <staging_dir>/
      asvspoof2019/
        *.wav
      wavefake/
        *.wav
      in_the_wild/
        *.wav
      metadata/
        unified.csv

It then pushes the directory as a private Kaggle Dataset via `scripts/kaggle_sync.py`
so that Phase 2's GPU embedding extraction notebook can mount it at `/kaggle/input/<dataset-slug>/`.

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
from typing import Optional

from voxguard.config import BASE_DIR, DATA_METADATA_DIR, DATA_PROCESSED_DIR
from voxguard.utils.logging_utils import get_logger

# Import upload_dataset from kaggle_sync helper
try:
    from scripts.kaggle_sync import upload_dataset
except ImportError:
    # If scripts is not in python path directly
    sys.path.insert(0, str(BASE_DIR))
    from scripts.kaggle_sync import upload_dataset

logger = get_logger("package_for_kaggle")

DEFAULT_DATASET_NAME = "voxguard-preprocessed-data"
DEFAULT_TITLE = "VoxGuard Preprocessed Audio and Unified Metadata"
DEFAULT_STAGING_DIR = BASE_DIR / "data" / "kaggle_package"


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


def prepare_kaggle_package(
    processed_dir: Path = DATA_PROCESSED_DIR,
    metadata_csv: Path = DATA_METADATA_DIR / "unified.csv",
    staging_dir: Path = DEFAULT_STAGING_DIR,
    symlink_mode: bool = False,
) -> Path:
    """
    Assembles processed audio folders and metadata/unified.csv into the staging directory.

    Args:
        processed_dir: Directory containing preprocessed audio subfolders.
        metadata_csv: Path to unified.csv metadata file.
        staging_dir: Destination packaging directory.
        symlink_mode: If True, attempts symlinks/hardlinks instead of copying.

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

    logger.info(f"Preparing Kaggle package staging directory: {stage_path}")
    stage_path.mkdir(parents=True, exist_ok=True)

    # 1. Stage metadata/unified.csv
    meta_subfolder = stage_path / "metadata"
    meta_subfolder.mkdir(parents=True, exist_ok=True)
    target_meta_file = meta_subfolder / "unified.csv"
    shutil.copy2(meta_path, target_meta_file)
    logger.info(f"Staged metadata -> {target_meta_file.relative_to(stage_path).as_posix()}")

    # 2. Stage processed audio folders
    dataset_dirs = [p for p in proc_path.iterdir() if p.is_dir() and p.name != "metadata"]
    total_staged_audio = 0

    for ds_dir in dataset_dirs:
        dest_ds_dir = stage_path / ds_dir.name
        dest_ds_dir.mkdir(parents=True, exist_ok=True)

        audio_files = list(ds_dir.glob("*.wav"))
        logger.info(f"Staging {len(audio_files):,} audio clips for '{ds_dir.name}'...")

        for audio_file in audio_files:
            target_audio = dest_ds_dir / audio_file.name
            if not target_audio.exists():
                try:
                    os.link(audio_file, target_audio)
                except Exception:
                    shutil.copy2(audio_file, target_audio)
            total_staged_audio += 1

    logger.info(
        f"Kaggle package assembly complete: {total_staged_audio:,} audio files "
        f"across {len(dataset_dirs)} datasets + metadata/unified.csv."
    )
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
    logger.info("UPLOADING DATASET TO KAGGLE")
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

    # 4. Print prominent instructions for Phase 2
    print("\n" + "=" * 78)
    print("KAGGLE DATASET UPLOAD COMPLETE!")
    print("=" * 78)
    print(f"  Dataset Slug : {slug}")
    print(f"  Dataset URL  : {dataset_url}")
    print(f"  Mount Point  : {mount_path}")
    print("\nIMPORTANT FOR PHASE 2 GPU NOTEBOOKS:")
    print(f"  1. Open your Phase 2 Kaggle GPU Notebook.")
    print(f"  2. Click 'Add Input' -> 'Your Datasets' -> Search '{dataset_id}'.")
    print(f"  3. Attach '{slug}'. It will mount at: {mount_path}")
    print(f"  4. Keep this dataset slug noted down for all subsequent GPU notebook runs.")
    print("=" * 78 + "\n")

    return dataset_url


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package preprocessed VoxGuard datasets and upload as a private Kaggle Dataset."
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
