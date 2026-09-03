"""
preprocess.py — Shared standardized audio preprocessing pipeline.

This module processes raw audio from any supported dataset into standardized
formats for feature extraction, embedding backbones, and classifier training:
- Resampling to target sample rate (default: 16 kHz)
- Downmixing multi-channel audio to mono float32
- Silence trimming with safety guards against zero-length output
- Flattened collision-free output paths
- Resumable processing (skips already processed audio files)
- Stores relative paths (e.g. data/processed/<dataset>/<filename>.wav)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
import pandas as pd

from voxguard.config import BASE_DIR, DATA_PROCESSED_DIR, SAMPLE_RATE
from voxguard.utils.audio_io import load_audio, save_audio
from voxguard.utils.logging_utils import get_logger

logger = get_logger(__name__)


def generate_unique_filename(row: pd.Series, dataset: str) -> str:
    """
    Generates a collision-resistant unique filename for a preprocessed audio clip.

    Format:
      - If generator is present (WaveFake): <dataset>_<generator>_<orig_stem>.wav
      - If parent directory carries differentiator (e.g. LJSpeech vocoders): <dataset>_<parent>_<orig_stem>.wav
      - Default: <dataset>_<orig_stem>.wav
    """
    raw_path = Path(str(row.get("filepath", "")))
    orig_stem = raw_path.stem

    # Check for generator metadata (WaveFake)
    generator = row.get("generator")
    if pd.notna(generator) and str(generator).strip() and str(generator).lower() not in ("nan", "none", "unknown"):
        return f"{dataset}_{str(generator).strip()}_{orig_stem}.wav"

    # Check parent folder name if it's descriptive (e.g. vocoder name or partition)
    parent_name = raw_path.parent.name
    if parent_name and parent_name.lower() not in ("flac", "wav", "audio", "data", "raw"):
        # If parent already starts with dataset name or is informative
        if not parent_name.lower().startswith(dataset.lower()):
            return f"{dataset}_{parent_name}_{orig_stem}.wav"

    return f"{dataset}_{orig_stem}.wav"


def preprocess_dataset(
    metadata_df: pd.DataFrame,
    output_dir: Union[Path, str] = DATA_PROCESSED_DIR,
    target_sr: int = SAMPLE_RATE,
    trim_silence: bool = True,
    project_root: Optional[Union[Path, str]] = None,
    top_db: int = 30,
) -> pd.DataFrame:
    """
    Standardizes and preprocesses all audio files referenced in metadata_df.

    For every row:
      1. Resolves destination path under output_dir/<dataset>/<filename>.wav.
      2. Skips file if processed output already exists on disk (resumable).
      3. Loads and resamples audio via audio_io.load_audio to target_sr mono float32.
      4. Optionally trims leading/trailing silence using librosa.effects.trim.
      5. Guards against zero-length output (falls back to untrimmed audio if trim empties it).
      6. Writes processed 16kHz WAV using audio_io.save_audio.
      7. Records repo-relative path in the 'processed_path' column.

    Args:
        metadata_df: DataFrame with at least ['filepath', 'dataset'].
        output_dir: Base directory to store processed audio (default: data/processed).
        target_sr: Target sample rate in Hz (default: 16,000 Hz).
        trim_silence: Whether to trim leading/trailing silence (default: True).
        project_root: VoxGuard project root for computing relative paths (defaults to BASE_DIR).
        top_db: Silence threshold in decibels below peak for trimming (default: 30 dB).

    Returns:
        pd.DataFrame: Updated DataFrame with populated 'processed_path' column.
    """
    if "filepath" not in metadata_df.columns:
        raise ValueError("metadata_df missing required 'filepath' column.")
    if "dataset" not in metadata_df.columns:
        raise ValueError("metadata_df missing required 'dataset' column.")

    base_out = Path(output_dir).resolve()
    root_path = Path(project_root).resolve() if project_root else BASE_DIR

    total_files = len(metadata_df)
    processed_paths: list[str] = []
    skipped_count = 0
    written_count = 0
    failed_count = 0

    logger.info(
        f"Starting preprocessing for {total_files:,} files "
        f"(target_sr={target_sr} Hz, trim_silence={trim_silence}, out_dir={base_out})"
    )

    for idx, (_, row) in enumerate(metadata_df.iterrows()):
        raw_filepath = Path(str(row["filepath"]))
        dataset = str(row["dataset"]).strip()

        # Generate unique filename and destination path
        unique_name = generate_unique_filename(row, dataset)
        
        # Organize in per-dataset subfolders if output_dir is top-level processed dir
        if base_out.name == dataset:
            dest_file = base_out / unique_name
        else:
            dest_file = base_out / dataset / unique_name

        dest_file.parent.mkdir(parents=True, exist_ok=True)

        # Calculate repo-relative path (e.g. data/processed/asvspoof2019/xxx.wav)
        try:
            rel_path = dest_file.relative_to(root_path).as_posix()
        except ValueError:
            # Fallback if dest_file is outside project root
            rel_path = dest_file.as_posix()

        # Resumability check: if target already exists with non-zero size, skip processing
        if dest_file.exists() and dest_file.stat().st_size > 0:
            skipped_count += 1
            processed_paths.append(rel_path)
        else:
            try:
                # Load, downmix, and resample
                waveform, sr = load_audio(raw_filepath, target_sr=target_sr)

                # Silence trimming
                if trim_silence:
                    trimmed_waveform, _ = librosa.effects.trim(waveform, top_db=top_db)
                    if len(trimmed_waveform) > 0:
                        waveform = trimmed_waveform
                    else:
                        logger.warning(
                            f"Silence trim resulted in 0 samples for '{raw_filepath.name}'; "
                            "preserving untrimmed audio."
                        )

                # Save preprocessed audio
                save_audio(dest_file, waveform, sr=sr)
                written_count += 1
                processed_paths.append(rel_path)

            except Exception as exc:
                failed_count += 1
                logger.error(f"Failed to process '{raw_filepath}': {exc}")
                # Append empty string or relative path to indicate failure
                processed_paths.append("")

        # Progress logging every 500 files
        if (idx + 1) % 500 == 0 or (idx + 1) == total_files:
            logger.info(
                f"Progress: [{idx + 1:,}/{total_files:,}] "
                f"({written_count:,} written, {skipped_count:,} cached/skipped, {failed_count} failed)"
            )

    result_df = metadata_df.copy()
    result_df["processed_path"] = processed_paths

    logger.info(
        f"Preprocessing finished: {total_files:,} total files "
        f"({written_count:,} written, {skipped_count:,} skipped/cached, {failed_count} failed)."
    )

    return result_df
