#!/usr/bin/env python3
"""
prepare_xtts_references.py — Validate and prepare XTTS-v2 reference voice clips.

This script processes separate natural 6-10 second reference recordings per speaker
to prepare them for voice cloning with Coqui XTTS-v2:
1. Reads data/metadata/hindi_hinglish_real.csv to verify speaker consent (consent_confirmed == True).
2. Discovers reference clips in --reference_dir matching each consented speaker.
3. Loads, converts to 16kHz mono WAV, and trims leading/trailing silence using audio_io.
4. Validates duration is between 5.0 and 15.0 seconds (XTTS-v2 clones poorly from < 5s clips).
5. Saves clean reference clips to data/raw/hindi_hinglish/references/{speaker_id}_ref.wav.
6. Writes a manifest to data/metadata/xtts_references.csv mapping speaker_id -> validated reference clip.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd

from voxguard import config
from voxguard.utils.audio_io import load_audio, save_audio
from voxguard.utils.logging_utils import get_logger

logger = get_logger("prepare_xtts_references")

DEFAULT_REAL_METADATA_CSV = config.DATA_METADATA_DIR / "hindi_hinglish_real.csv"
DEFAULT_REFERENCE_DIR = config.DATA_RAW_DIR / "hindi_hinglish_staging"
DEFAULT_OUTPUT_DIR = config.DATA_RAW_DIR / "hindi_hinglish" / "references"
DEFAULT_OUTPUT_MANIFEST = config.DATA_METADATA_DIR / "xtts_references.csv"

MIN_REFERENCE_DURATION_SECONDS = 5.0
MAX_REFERENCE_DURATION_SECONDS = 15.0


def find_speaker_reference_file(
    speaker_id: str,
    reference_dir: Path,
) -> Optional[Path]:
    """
    Locates a reference audio file for a given speaker in reference_dir.

    Looks for common naming patterns:
      - {speaker_id}_reference.wav / .flac / .mp3 / .m4a
      - {speaker_id}_ref.wav
      - {speaker_id}.wav
      - {speaker_id}_sample.wav
      - Any audio file starting with {speaker_id}_ref or {speaker_id}
    """
    clean_id = speaker_id.strip().lower()

    if not reference_dir.exists():
        return None

    # Exact pattern checks in priority order
    candidate_patterns = [
        f"{clean_id}_reference.*",
        f"{clean_id}_ref.*",
        f"{clean_id}_sample.*",
        f"{clean_id}.*",
    ]

    for pat in candidate_patterns:
        matches = list(reference_dir.glob(pat))
        valid_matches = [m for m in matches if m.is_file() and m.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg", ".m4a")]
        if valid_matches:
            return valid_matches[0]

    # Fallback substring search
    all_files = [f for f in reference_dir.iterdir() if f.is_file() and f.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg", ".m4a")]
    for f in all_files:
        stem = f.stem.lower()
        if stem.startswith(clean_id) and any(k in stem for k in ("ref", "sample", "voice")):
            return f

    return None


def validate_and_clean_reference(
    audio_path: Path,
    target_sr: int = config.SAMPLE_RATE,
    top_db: int = 30,
    min_duration: float = MIN_REFERENCE_DURATION_SECONDS,
    max_duration: float = MAX_REFERENCE_DURATION_SECONDS,
) -> Tuple[Optional[np.ndarray], float, Optional[str]]:
    """
    Loads, cleans, and validates a reference audio file.

    Returns:
        (waveform, duration_seconds, error_or_warning_message)
    """
    try:
        waveform, sr = load_audio(audio_path, target_sr=target_sr)
    except Exception as exc:
        return None, 0.0, f"Failed to load audio: {exc}"

    # Trim leading and trailing silence
    trimmed_waveform, _ = librosa.effects.trim(waveform, top_db=top_db)
    if len(trimmed_waveform) > 0:
        waveform = trimmed_waveform
    else:
        logger.warning(f"Silence trimming resulted in 0 samples for '{audio_path.name}'; using untrimmed waveform.")

    duration = len(waveform) / target_sr

    # Check minimum duration constraint
    if duration < min_duration:
        msg = (
            f"Reference audio too short ({duration:.2f}s < {min_duration:.1f}s minimum). "
            "XTTS-v2 clones poorly from very short references."
        )
        return None, duration, msg

    # Warning / gentle trim for maximum duration constraint
    msg = None
    if duration > max_duration:
        max_samples = int(max_duration * target_sr)
        waveform = waveform[:max_samples]
        msg = f"Reference audio ({duration:.2f}s) exceeded {max_duration:.1f}s; capped to {max_duration:.1f}s."
        duration = max_duration

    return waveform, duration, msg


def prepare_xtts_references(
    real_metadata_csv: Path = DEFAULT_REAL_METADATA_CSV,
    reference_dir: Path = DEFAULT_REFERENCE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    output_manifest: Path = DEFAULT_OUTPUT_MANIFEST,
    min_duration: float = MIN_REFERENCE_DURATION_SECONDS,
    max_duration: float = MAX_REFERENCE_DURATION_SECONDS,
    target_sr: int = config.SAMPLE_RATE,
    project_root: Path = config.BASE_DIR,
) -> pd.DataFrame:
    """
    Validates and prepares XTTS reference clips for all consented speakers.

    Returns:
        pd.DataFrame: Manifest mapping speaker_id -> validated reference clip path.
    """
    real_meta_path = Path(real_metadata_csv).resolve()
    ref_dir = Path(reference_dir).resolve()
    out_dir = Path(output_dir).resolve()
    manifest_path = Path(output_manifest).resolve()
    root_path = Path(project_root).resolve()

    logger.info("=" * 78)
    logger.info("VOXGUARD XTTS-v2 REFERENCE CLIP PREPARATION")
    logger.info("=" * 78)
    logger.info(f"Real Metadata CSV : {real_meta_path}")
    logger.info(f"Reference Inputs  : {ref_dir}")
    logger.info(f"Output References : {out_dir}")
    logger.info(f"Output Manifest   : {manifest_path}")
    logger.info(f"Valid Duration    : [{min_duration:.1f}s - {max_duration:.1f}s]")
    logger.info("=" * 78)

    if not real_meta_path.exists():
        raise FileNotFoundError(f"Real Hindi metadata CSV not found: {real_meta_path}")

    df_real = pd.read_csv(real_meta_path)
    if "speaker_id" not in df_real.columns:
        raise ValueError(f"Missing 'speaker_id' in {real_meta_path}")

    # Determine unique speakers and their consent status
    speaker_consent: Dict[str, bool] = {}
    for _, row in df_real.iterrows():
        spk = str(row["speaker_id"]).strip()
        consent = bool(row.get("consent_confirmed", False))
        speaker_consent[spk] = consent

    logger.info(f"Found {len(speaker_consent)} unique speaker(s) in real recordings metadata.")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_records: List[Dict[str, str | float | bool]] = []
    summary_rows: List[Dict[str, str]] = []

    for speaker_id, consent_ok in speaker_consent.items():
        status_note = ""
        # 1. Consent check
        if not consent_ok:
            logger.warning(
                f"[SKIPPED] Speaker '{speaker_id}' has consent_confirmed=False. "
                "Explicit consent is required before preparing reference clips for cloning."
            )
            summary_rows.append({
                "speaker_id": speaker_id,
                "input_file": "-",
                "duration": "-",
                "consent": "False",
                "status": "SKIPPED (No Consent)",
            })
            continue

        # 2. Reference file search
        ref_file = find_speaker_reference_file(speaker_id, ref_dir)
        if ref_file is None:
            logger.warning(f"[MISSING] No reference clip found for speaker '{speaker_id}' in {ref_dir}")
            summary_rows.append({
                "speaker_id": speaker_id,
                "input_file": "Not Found",
                "duration": "-",
                "consent": "True",
                "status": "MISSING FILE",
            })
            continue

        # 3. Audio validation & clean
        waveform, duration, msg = validate_and_clean_reference(
            audio_path=ref_file,
            target_sr=target_sr,
            min_duration=min_duration,
            max_duration=max_duration,
        )

        if waveform is None:
            logger.warning(f"[SKIPPED] {speaker_id}: {msg}")
            summary_rows.append({
                "speaker_id": speaker_id,
                "input_file": ref_file.name,
                "duration": f"{duration:.2f}s",
                "consent": "True",
                "status": f"REJECTED ({msg})",
            })
            continue

        if msg:
            logger.info(f"[NOTICE] {speaker_id}: {msg}")

        # 4. Save validated reference clip
        dest_filename = f"{speaker_id}_ref.wav"
        dest_path = out_dir / dest_filename
        save_audio(dest_path, waveform, sr=target_sr)

        # Compute repo-relative path
        try:
            rel_path = dest_path.relative_to(root_path).as_posix()
        except ValueError:
            rel_path = dest_path.as_posix()

        manifest_records.append({
            "speaker_id": speaker_id,
            "reference_path": rel_path,
            "duration_seconds": round(duration, 3),
            "consent_confirmed": True,
        })

        summary_rows.append({
            "speaker_id": speaker_id,
            "input_file": ref_file.name,
            "duration": f"{duration:.2f}s",
            "consent": "True",
            "status": "VALIDATED",
        })
        logger.info(f"[VALIDATED] Speaker '{speaker_id}' reference prepared -> {rel_path} ({duration:.2f}s)")

    # 5. Save manifest CSV
    manifest_df = pd.DataFrame(manifest_records)
    if not manifest_df.empty:
        manifest_df.to_csv(manifest_path, index=False)
        logger.info(f"[SUCCESS] Wrote XTTS references manifest to: {manifest_path}")
    else:
        logger.error("[ERROR] No valid reference clips could be prepared.")

    # 6. Print formatted summary table
    print("\n" + "=" * 85)
    print("XTTS-v2 REFERENCE PREPARATION SUMMARY")
    print("=" * 85)
    print(f"{'Speaker ID':<16} | {'Input File':<24} | {'Duration':<10} | {'Consent':<8} | {'Status':<20}")
    print("-" * 85)
    for r in summary_rows:
        print(f"{r['speaker_id']:<16} | {r['input_file']:<24} | {r['duration']:<10} | {r['consent']:<8} | {r['status']:<20}")
    print("=" * 85 + "\n")

    return manifest_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and validate XTTS-v2 reference voice clips for Hindi/Hinglish cloning."
    )
    parser.add_argument(
        "--reference_dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
        help=f"Directory containing raw reference audio files (default: {DEFAULT_REFERENCE_DIR})",
    )
    parser.add_argument(
        "--real_metadata_csv",
        type=Path,
        default=DEFAULT_REAL_METADATA_CSV,
        help=f"Path to hindi_hinglish_real.csv (default: {DEFAULT_REAL_METADATA_CSV})",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save validated reference WAVs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output_manifest",
        type=Path,
        default=DEFAULT_OUTPUT_MANIFEST,
        help=f"Path to output manifest CSV (default: {DEFAULT_OUTPUT_MANIFEST})",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=MIN_REFERENCE_DURATION_SECONDS,
        help=f"Minimum allowable reference duration in seconds (default: {MIN_REFERENCE_DURATION_SECONDS})",
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=MAX_REFERENCE_DURATION_SECONDS,
        help=f"Maximum allowable reference duration in seconds (default: {MAX_REFERENCE_DURATION_SECONDS})",
    )
    parser.add_argument(
        "--target_sr",
        type=int,
        default=config.SAMPLE_RATE,
        help=f"Target sampling rate in Hz (default: {config.SAMPLE_RATE})",
    )

    args = parser.parse_args()

    try:
        prepare_xtts_references(
            real_metadata_csv=args.real_metadata_csv,
            reference_dir=args.reference_dir,
            output_dir=args.output_dir,
            output_manifest=args.output_manifest,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            target_sr=args.target_sr,
        )
    except Exception as exc:
        logger.error(f"Reference clip preparation failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
