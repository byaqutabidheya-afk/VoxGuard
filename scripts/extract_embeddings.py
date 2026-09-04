#!/usr/bin/env python3
"""
extract_embeddings.py — Batch embedding extraction CLI for VoxGuard.

Runs a frozen SSL backbone (wav2vec2 or WavLM) over a dataset split and
caches the resulting embedding matrix (+ index-map CSV) to disk via
voxguard.embeddings.cache.extract_and_cache.

Dataset handling:
  - asvspoof2019 (default): uses the official train/dev/eval protocol split
    via voxguard.utils.splits.get_asvspoof_splits; --split selects which
    partition to embed. Output: {output_dir}/{model}_{split}.npy
  - wavefake / in_the_wild: eval-only, out-of-domain datasets. Loaded via
    voxguard.utils.metadata.load_unified_metadata([dataset]) directly (NOT
    get_asvspoof_splits, which only understands ASVspoof2019). The entire
    returned DataFrame is treated as one eval-only batch; --split is unused.
    Output: {output_dir}/{model}_{dataset}.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from voxguard import config
from voxguard.embeddings.cache import extract_and_cache
from voxguard.embeddings.extractor import EmbeddingExtractor
from voxguard.utils.logging_utils import get_logger
from voxguard.utils.metadata import load_unified_metadata
from voxguard.utils.splits import get_asvspoof_splits

logger = get_logger("extract_embeddings")

MODEL_NAME_MAP = {
    "wav2vec2": "facebook/wav2vec2-base",
    "wavlm": "microsoft/wavlm-base-plus",
}

EVAL_ONLY_DATASETS = ("wavefake", "in_the_wild")
DEFAULT_OUTPUT_DIR = config.MODELS_DIR / "embeddings"


def _select_dataframe(dataset: str, split: str | None) -> tuple[pd.DataFrame, str]:
    """Resolves *dataset*/*split* to a DataFrame to embed and an output-name suffix.

    Returns
    -------
    df:
        The rows to embed.
    name_suffix:
        Used to build the output filename: ``{model}_{name_suffix}.npy``.
    """
    if dataset == "asvspoof2019":
        if split is None:
            raise ValueError("--split is required when --dataset asvspoof2019 (choose train/dev/eval).")

        unified_df = load_unified_metadata(["asvspoof2019"])
        train_df, dev_df, eval_df = get_asvspoof_splits(unified_df)
        split_map = {"train": train_df, "dev": dev_df, "eval": eval_df}
        return split_map[split], split

    # wavefake / in_the_wild: eval-only, out-of-domain datasets.
    if split is not None:
        logger.warning("--split is ignored for dataset=%s (treated as a single eval-only batch).", dataset)

    df = load_unified_metadata([dataset])
    if "dataset" in df.columns:
        df = df[df["dataset"] == dataset].reset_index(drop=True)

    return df, dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and cache SSL backbone embeddings for a VoxGuard dataset split."
    )
    parser.add_argument(
        "--dataset",
        choices=["asvspoof2019", "wavefake", "in_the_wild"],
        default="asvspoof2019",
        help="Dataset to embed (default: asvspoof2019).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "dev", "eval"],
        default=None,
        help="Official split to embed. Required for --dataset asvspoof2019; ignored for wavefake/in_the_wild.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_NAME_MAP.keys()),
        default="wav2vec2",
        help="SSL backbone to use (default: wav2vec2).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of files per extraction batch (default: 16).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to save the .npy/.csv outputs (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite even if the cached .npy already exists.",
    )

    args = parser.parse_args()

    try:
        df, name_suffix = _select_dataframe(args.dataset, args.split)
    except Exception as exc:
        logger.error("Failed to resolve dataset/split: %s", exc)
        sys.exit(1)

    if df.empty:
        logger.error("Resolved DataFrame for dataset=%s split=%s is empty.", args.dataset, args.split)
        sys.exit(1)

    model_name = MODEL_NAME_MAP[args.model]
    output_path = Path(args.output_dir) / f"{args.model}_{name_suffix}.npy"

    logger.info("=" * 78)
    logger.info("VOXGUARD EMBEDDING EXTRACTION")
    logger.info("Dataset      : %s", args.dataset)
    logger.info("Split        : %s", args.split if args.dataset == "asvspoof2019" else "eval (whole dataset)")
    logger.info("Model        : %s (%s)", args.model, model_name)
    logger.info("Rows         : %d", len(df))
    logger.info("Batch size   : %d", args.batch_size)
    logger.info("Output       : %s", output_path)
    logger.info("=" * 78)

    extractor = EmbeddingExtractor(model_name=model_name, device=config.get_device())

    try:
        extract_and_cache(
            df=df,
            extractor=extractor,
            output_path=output_path,
            path_col="processed_path",
            batch_size=args.batch_size,
            force=args.force,
        )
    except Exception as exc:
        logger.error("Embedding extraction failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
