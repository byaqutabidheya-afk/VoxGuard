#!/usr/bin/env python3
"""
generate_hindi_clones.py — Batch synthesize matched XTTS-v2 clones for Hindi/Hinglish speech.

This script pairs every real recording from data/metadata/hindi_hinglish_real.csv
with a corresponding synthetic clone using the speaker's validated natural reference clip
from data/metadata/xtts_references.csv:
- Produces matched pairs: same speaker, same 25-sentence script, real vs synthetic.
- Output path: data/raw/hindi_hinglish/synthetic/{speaker_id}_{category}_{sentence_id:02d}_clone.wav
- Output metadata: data/metadata/hindi_hinglish_synthetic.csv [filepath, speaker_id, category, sentence_id, label, generator]
- Resumable: skips already generated audio files if they exist on disk.
- Fault-tolerant: logs per-clip generation failures without terminating the entire batch.
- Hardware: defaults to CPU execution; opt-in GPU support via --gpu flag.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from voxguard import config
from voxguard.synth.xtts_clone import clone_voice
from voxguard.utils.logging_utils import get_logger

logger = get_logger("generate_hindi_clones")

DEFAULT_REAL_CSV = config.DATA_METADATA_DIR / "hindi_hinglish_real.csv"
DEFAULT_REFERENCES_CSV = config.DATA_METADATA_DIR / "xtts_references.csv"
DEFAULT_OUTPUT_DIR = config.DATA_RAW_DIR / "hindi_hinglish" / "synthetic"
DEFAULT_OUTPUT_CSV = config.DATA_METADATA_DIR / "hindi_hinglish_synthetic.csv"


def generate_hindi_clones(
    real_metadata_csv: Path = DEFAULT_REAL_CSV,
    references_manifest_csv: Path = DEFAULT_REFERENCES_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    output_metadata_csv: Path = DEFAULT_OUTPUT_CSV,
    language: str = "hi",
    use_gpu: bool = False,
    speaker_filter: Optional[str] = None,
    sentence_limit: Optional[int] = None,
    project_root: Path = config.BASE_DIR,
) -> pd.DataFrame:
    """
    Generates synthetic matched clones for consented speakers.

    Returns:
        pd.DataFrame: Synthetic dataset metadata table.
    """
    real_meta_path = Path(real_metadata_csv).resolve()
    refs_meta_path = Path(references_manifest_csv).resolve()
    out_audio_dir = Path(output_dir).resolve()
    out_csv_path = Path(output_metadata_csv).resolve()
    root_path = Path(project_root).resolve()

    mode_str = "Running on GPU" if use_gpu else "Running on CPU"

    logger.info("=" * 78)
    logger.info("VOXGUARD HINDI/HINGLISH XTTS-v2 CLONE GENERATION")
    logger.info("=" * 78)
    logger.info(f"Execution Mode          : {mode_str}")
    logger.info(f"Real Metadata CSV       : {real_meta_path}")
    logger.info(f"References Manifest     : {refs_meta_path}")
    logger.info(f"Output Audio Directory  : {out_audio_dir}")
    logger.info(f"Output Metadata CSV     : {out_csv_path}")
    logger.info(f"Language                : {language}")
    logger.info("=" * 78)

    if not real_meta_path.exists():
        raise FileNotFoundError(f"Real metadata CSV not found: {real_meta_path}")
    if not refs_meta_path.exists():
        raise FileNotFoundError(f"References manifest CSV not found: {refs_meta_path}")

    df_real = pd.read_csv(real_meta_path)
    df_refs = pd.read_csv(refs_meta_path)

    # 1. Map validated reference paths per consented speaker
    speaker_references: Dict[str, Path] = {}
    for _, ref_row in df_refs.iterrows():
        spk = str(ref_row["speaker_id"]).strip()
        consent_ok = bool(ref_row.get("consent_confirmed", False))
        ref_rel_path = str(ref_row["reference_path"]).strip()
        ref_full_path = root_path / ref_rel_path if not Path(ref_rel_path).is_absolute() else Path(ref_rel_path)

        if consent_ok and ref_full_path.exists():
            speaker_references[spk] = ref_full_path
        else:
            logger.warning(
                f"Skipping reference for '{spk}': consent={consent_ok}, exists={ref_full_path.exists()}"
            )

    logger.info(
        f"Loaded {len(speaker_references)} validated speaker reference(s): {list(speaker_references.keys())}"
    )

    if not speaker_references:
        raise ValueError("No valid and consented speaker references available for cloning.")

    # 2. Filter real records for synthesis
    valid_speakers = set(speaker_references.keys())
    if speaker_filter:
        filter_spk = speaker_filter.strip().lower()
        if filter_spk not in valid_speakers:
            raise ValueError(f"Requested speaker '{speaker_filter}' not in valid references: {valid_speakers}")
        valid_speakers = {filter_spk}

    filtered_real = df_real[df_real["speaker_id"].isin(valid_speakers)].copy()

    # Filter out unconsented rows from real CSV as an additional safety guard
    if "consent_confirmed" in filtered_real.columns:
        filtered_real = filtered_real[filtered_real["consent_confirmed"] == True]

    if filtered_real.empty:
        raise ValueError("No consented real records found to generate matched clones.")

    # Apply per-speaker sentence limit if requested
    if sentence_limit is not None and sentence_limit > 0:
        filtered_real = filtered_real.groupby("speaker_id").head(sentence_limit).reset_index(drop=True)

    total_tasks = len(filtered_real)
    logger.info(f"Total matched clone utterances to synthesize: {total_tasks:,}")

    out_audio_dir.mkdir(parents=True, exist_ok=True)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    synthetic_records: List[Dict[str, str | int]] = []
    success_count = 0
    skipped_count = 0
    failure_count = 0

    t_start = time.time()

    # 3. Batch synthesis loop
    for idx, (_, row) in enumerate(filtered_real.iterrows(), start=1):
        speaker_id = str(row["speaker_id"]).strip()
        category = str(row["category"]).strip()
        sentence_id = int(row["sentence_id"])
        sentence_text = str(row["sentence_text"]).strip()

        # Filename format: {speaker_id}_{category}_{sentence_id:02d}_clone.wav
        dest_filename = f"{speaker_id}_{category}_{sentence_id:02d}_clone.wav"
        dest_audio_path = out_audio_dir / dest_filename

        # Repo-relative path for CSV consistency
        try:
            rel_dest_path = dest_audio_path.relative_to(root_path).as_posix()
        except ValueError:
            rel_dest_path = dest_audio_path.as_posix()

        ref_path = speaker_references[speaker_id]

        # Resumability check: if clone already exists and is non-empty, skip synthesis
        if dest_audio_path.exists() and dest_audio_path.stat().st_size > 1024:
            skipped_count += 1
            synthetic_records.append({
                "filepath": rel_dest_path,
                "speaker_id": speaker_id,
                "category": category,
                "sentence_id": sentence_id,
                "label": "synthetic",
                "generator": "xtts_v2",
            })
            continue

        logger.info(
            f"[{idx}/{total_tasks}] Synthesizing ({speaker_id}, {category} #{sentence_id:02d}): "
            f"'{sentence_text[:45]}...'"
        )

        try:
            t0 = time.time()
            clone_voice(
                reference_audio_path=ref_path,
                text=sentence_text,
                language=language,
                output_path=dest_audio_path,
                use_gpu=use_gpu,
            )
            elapsed = time.time() - t0
            success_count += 1
            logger.info(f" -> Generated in {elapsed:.2f}s -> {dest_filename}")

            synthetic_records.append({
                "filepath": rel_dest_path,
                "speaker_id": speaker_id,
                "category": category,
                "sentence_id": sentence_id,
                "label": "synthetic",
                "generator": "xtts_v2",
            })
        except Exception as exc:
            failure_count += 1
            logger.error(
                f" -> [FAILURE] Failed to clone sentence #{sentence_id} for speaker '{speaker_id}': {exc}"
            )

    total_elapsed = time.time() - t_start

    # 4. Save metadata CSV
    df_synthetic = pd.DataFrame(synthetic_records)
    if not df_synthetic.empty:
        df_synthetic.to_csv(out_csv_path, index=False)
        logger.info(f"Saved synthetic Hindi metadata ({len(df_synthetic):,} rows) to: {out_csv_path}")
    else:
        logger.error("No synthetic records generated to save.")

    # 5. Print summary
    print("\n" + "=" * 78)
    print("HINDI/HINGLISH XTTS-v2 CLONE GENERATION SUMMARY")
    print("=" * 78)
    print(f"  Execution mode          : {mode_str}")
    print(f"  Total matched targets   : {total_tasks:,}")
    print(f"  Successfully synthesized: {success_count:,}")
    print(f"  Skipped (cached/exist)  : {skipped_count:,}")
    print(f"  Failed synthesis calls  : {failure_count}")
    print(f"  Total valid clones ready: {len(df_synthetic):,}")
    print(f"  Total processing time   : {total_elapsed:.2f}s")
    print(f"  Metadata CSV saved at   : {out_csv_path}")
    print("=" * 78 + "\n")

    return df_synthetic


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch generate matched XTTS-v2 synthetic clones for Hindi/Hinglish speech."
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Run XTTS-v2 voice cloning on GPU (default: False, runs on CPU)",
    )
    parser.add_argument(
        "--real_csv",
        type=Path,
        default=DEFAULT_REAL_CSV,
        help=f"Path to hindi_hinglish_real.csv (default: {DEFAULT_REAL_CSV})",
    )
    parser.add_argument(
        "--references_csv",
        type=Path,
        default=DEFAULT_REFERENCES_CSV,
        help=f"Path to xtts_references.csv (default: {DEFAULT_REFERENCES_CSV})",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to store generated clone WAVs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Path to output hindi_hinglish_synthetic.csv (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="hi",
        help="Language code for synthesis (default: 'hi')",
    )
    parser.add_argument(
        "--speaker_id",
        type=str,
        default=None,
        help="Optional single speaker to clone (e.g. --speaker_id byaquta)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sentence limit per speaker (for testing)",
    )

    args = parser.parse_args()

    try:
        generate_hindi_clones(
            real_metadata_csv=args.real_csv,
            references_manifest_csv=args.references_csv,
            output_dir=args.output_dir,
            output_metadata_csv=args.output_csv,
            language=args.language,
            use_gpu=args.gpu,
            speaker_filter=args.speaker_id,
            sentence_limit=args.limit,
        )
    except Exception as exc:
        logger.error(f"Clone generation script failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
