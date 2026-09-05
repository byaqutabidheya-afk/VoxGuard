#!/usr/bin/env python3
"""
extract_hindi_train_embeddings.py — Extract and cache feature embeddings for the Hindi/Hinglish track.

Extracts frozen self-supervised speech embeddings (wav2vec2 / wavlm) and optional
handcrafted prosody features for the Hindi/Hinglish train (and eval) splits
produced by Prompt 4.6 (speaker_holdout with 'soumya' held out).

Outputs saved to models/embeddings/:
  - models/embeddings/wav2vec2_hindi_train.npy (and .csv)
  - models/embeddings/wav2vec2_hindi_eval.npy (and .csv)
  - models/embeddings/wavlm_hindi_train.npy (and .csv)
  - models/embeddings/prosody_hindi_train.npy (and .csv)

Runs locally on CPU in seconds to minutes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from voxguard import config
from voxguard.embeddings.cache import extract_and_cache
from voxguard.embeddings.extractor import EmbeddingExtractor
from voxguard.features.compose import extract_and_cache_prosody
from voxguard.utils.hindi_splits import get_hindi_hinglish_splits
from voxguard.utils.logging_utils import get_logger

logger = get_logger("extract_hindi_train_embeddings")

DEFAULT_TRACK_CSV = config.DATA_METADATA_DIR / "hindi_hinglish_track.csv"
DEFAULT_OUTPUT_DIR = config.MODELS_DIR / "embeddings"
DEFAULT_HOLDOUT_SPEAKER = "soumya"


def extract_hindi_embeddings(
    track_csv: Path = DEFAULT_TRACK_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    holdout_speaker: str = DEFAULT_HOLDOUT_SPEAKER,
    model_name: str = "facebook/wav2vec2-base",
    extract_eval: bool = True,
    extract_prosody: bool = True,
    extract_wavlm: bool = True,
    batch_size: int = 16,
    force: bool = False,
) -> None:
    """
    Extracts embeddings for the Hindi/Hinglish train and eval splits locally on CPU.
    """
    track_path = Path(track_csv).resolve()
    out_dir = Path(output_dir).resolve()

    if not track_path.exists():
        raise FileNotFoundError(f"Hindi track CSV not found: {track_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    df_track = pd.read_csv(track_path)
    logger.info(f"Loaded Hindi track metadata ({len(df_track)} rows) from {track_path}")

    # 1. Split into train (byaquta + mahato) and eval (soumya)
    train_df, eval_df = get_hindi_hinglish_splits(
        df_track,
        mode="speaker_holdout",
        holdout_speaker=holdout_speaker,
    )

    logger.info(
        f"Partitioned Hindi/Hinglish track: train={len(train_df)} clips (speakers: {sorted(train_df['speaker_id'].unique())}), "
        f"eval={len(eval_df)} clips (held-out speaker: '{holdout_speaker}')"
    )

    t0 = time.time()

    # 2. Extract Primary Backbone (wav2vec2)
    logger.info("=" * 78)
    logger.info(f"Extracting primary embeddings: {model_name} (CPU)")
    logger.info("=" * 78)
    extractor_wav2vec2 = EmbeddingExtractor(model_name=model_name, device="cpu")

    # Short prefix name for file (e.g. wav2vec2)
    model_prefix = "wav2vec2" if "wav2vec2" in model_name.lower() else "backbone"

    train_npy_wav2vec2 = out_dir / f"{model_prefix}_hindi_train.npy"
    extract_and_cache(
        df=train_df,
        extractor=extractor_wav2vec2,
        output_path=str(train_npy_wav2vec2),
        path_col="filepath",
        batch_size=batch_size,
        force=force,
    )

    if extract_eval:
        eval_npy_wav2vec2 = out_dir / f"{model_prefix}_hindi_eval.npy"
        extract_and_cache(
            df=eval_df,
            extractor=extractor_wav2vec2,
            output_path=str(eval_npy_wav2vec2),
            path_col="filepath",
            batch_size=batch_size,
            force=force,
        )

    # 3. Optional: Extract secondary backbone (WavLM)
    if extract_wavlm:
        wavlm_model_name = "microsoft/wavlm-base-plus"
        logger.info("=" * 78)
        logger.info(f"Extracting secondary embeddings: {wavlm_model_name} (CPU)")
        logger.info("=" * 78)
        extractor_wavlm = EmbeddingExtractor(model_name=wavlm_model_name, device="cpu")

        train_npy_wavlm = out_dir / "wavlm_hindi_train.npy"
        extract_and_cache(
            df=train_df,
            extractor=extractor_wavlm,
            output_path=str(train_npy_wavlm),
            path_col="filepath",
            batch_size=batch_size,
            force=force,
        )

        if extract_eval:
            eval_npy_wavlm = out_dir / "wavlm_hindi_eval.npy"
            extract_and_cache(
                df=eval_df,
                extractor=extractor_wavlm,
                output_path=str(eval_npy_wavlm),
                path_col="filepath",
                batch_size=batch_size,
                force=force,
            )

    # 4. Optional: Extract prosody features
    if extract_prosody:
        logger.info("=" * 78)
        logger.info("Extracting handcrafted 10-dim prosody features (CPU)")
        logger.info("=" * 78)
        prosody_train_npy = out_dir / "prosody_hindi_train.npy"
        extract_and_cache_prosody(
            df=train_df,
            output_path=str(prosody_train_npy),
            path_col="filepath",
            force=force,
        )

        if extract_eval:
            prosody_eval_npy = out_dir / "prosody_hindi_eval.npy"
            extract_and_cache_prosody(
                df=eval_df,
                output_path=str(prosody_eval_npy),
                path_col="filepath",
                force=force,
            )

    elapsed = time.time() - t0
    logger.info("=" * 78)
    logger.info(f"Hindi/Hinglish feature extraction finished in {elapsed:.2f}s.")
    logger.info(f"Output files stored in: {out_dir}")
    logger.info("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and cache embeddings for Hindi/Hinglish train/eval splits on CPU."
    )
    parser.add_argument(
        "--track_csv",
        type=Path,
        default=DEFAULT_TRACK_CSV,
        help=f"Path to hindi_hinglish_track.csv (default: {DEFAULT_TRACK_CSV})",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination directory for embeddings (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--holdout_speaker",
        type=str,
        default=DEFAULT_HOLDOUT_SPEAKER,
        help=f"Speaker to hold out for evaluation (default: '{DEFAULT_HOLDOUT_SPEAKER}')",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/wav2vec2-base",
        help="Pretrained backbone name (default: 'facebook/wav2vec2-base')",
    )
    parser.add_argument(
        "--no_wavlm",
        action="store_true",
        help="Skip extracting WavLM embeddings",
    )
    parser.add_argument(
        "--no_prosody",
        action="store_true",
        help="Skip extracting prosody features",
    )
    parser.add_argument(
        "--no_eval",
        action="store_true",
        help="Skip extracting eval split features",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for extraction (default: 16)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction and overwrite existing cache files",
    )

    args = parser.parse_args()

    try:
        extract_hindi_embeddings(
            track_csv=args.track_csv,
            output_dir=args.output_dir,
            holdout_speaker=args.holdout_speaker,
            model_name=args.model_name,
            extract_eval=not args.no_eval,
            extract_prosody=not args.no_prosody,
            extract_wavlm=not args.no_wavlm,
            batch_size=args.batch_size,
            force=args.force,
        )
    except Exception as exc:
        logger.error(f"Hindi embeddings extraction failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
