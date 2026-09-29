#!/usr/bin/env python3
"""
scripts/extract_chunked_embeddings.py — Chunk-level embedding extraction (Phase F2.2).

Builds the training/eval inputs for the chunk-native classifier family: one
embedding per fixed-length window, where the windows are exactly those
``StreamingBuffer`` delivers at inference time.

Pipeline
--------
1. Load the CLIP-level split first (``get_asvspoof_splits`` for ASVspoof2019;
   ``get_hindi_hinglish_splits(mode='speaker_holdout', holdout_speaker='soumya')``
   on the duration-matched Hindi track). Splitting before chunking is what
   keeps chunks of one clip from landing in both train and eval.
2. ``--max_clips`` subsamples at clip level, stratified on label (and, for
   ASVspoof, the attack/system id column when one is present).
3. Each clip is chunked with ``chunk_waveform_detailed`` (the same windows and
   silence gate as ``chunk_waveform`` / streaming inference). Silent windows
   are dropped and counted.
4. Chunks are embedded through ``embed_length_sorted`` — the same batching path
   ``extract_and_cache`` uses — in groups of clips so memory stays bounded.
5. Saves ``{model}_{dataset}_{split}_chunked.npy`` and a parallel ``.csv``
   manifest with the standard ``[row_index, filepath, label]`` columns plus
   ``parent_filepath``, ``chunk_index`` and ``chunk_start_seconds``.
   ``filepath`` and ``parent_filepath`` both hold the source clip's
   repo-relative path; ``parent_filepath`` is the column F2.4's clip-level
   aggregation groups on.

Label inheritance (known label noise)
-------------------------------------
Every chunk inherits its parent clip's label. This is a deliberate
simplification: no chunk-level ground truth exists. It is a known source of
label noise — a low-information chunk from inside a synthetic clip (a breath,
a pause just above the silence gate, a stretch with no audible artefacts) is
still labelled synthetic, and the classifier is asked to call it so.

Resumable in the same way as ``extract_and_cache``: if the output ``.npy``
already exists, extraction is skipped unless ``--force`` is given. The
``.npy`` is written last, via a temp file, so an interrupted run never leaves
a file that looks complete.

Where to run: ASVspoof2019 on Kaggle GPU (see Phase F2.3); Hindi locally on
CPU (~150 clips).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from voxguard import config
from voxguard.embeddings.cache import _resolve_audio_path, embed_length_sorted
from voxguard.embeddings.extractor import EmbeddingExtractor
from voxguard.utils.audio_io import load_audio
from voxguard.utils.chunk_audio import chunk_waveform_detailed
from voxguard.utils.hindi_splits import get_hindi_hinglish_splits
from voxguard.utils.logging_utils import get_logger
from voxguard.utils.splits import get_asvspoof_splits

logger = get_logger("extract_chunked_embeddings")

MODEL_NAMES: Dict[str, str] = {
    "wav2vec2": "facebook/wav2vec2-base",
    "wavlm": "microsoft/wavlm-base-plus",
}
UNIFIED_CSV = config.DATA_METADATA_DIR / "unified.csv"
HINDI_MATCHED_CSV = config.DATA_METADATA_DIR / "hindi_hinglish_track_matched.csv"
HOLDOUT_SPEAKER = "soumya"
DEFAULT_OUTPUT_DIR = config.MODELS_DIR / "embeddings"
SUBSAMPLE_SEED = 42
# Candidate ASVspoof attack/system id columns, used for stratification if present.
ATTACK_COLUMNS = ("attack_id", "system_id", "attack", "system")
# Clips chunked + embedded per group; bounds chunk memory (~3 chunks/clip x 96 KB).
CLIPS_PER_GROUP = 512

MANIFEST_COLUMNS = [
    "row_index", "filepath", "label", "parent_filepath", "chunk_index", "chunk_start_seconds",
]


def load_clip_split(dataset: str, split: str) -> tuple[pd.DataFrame, str]:
    """Returns (clip-level split DataFrame, name of the column holding the audio path)."""
    if dataset == "asvspoof2019":
        train_df, dev_df, eval_df = get_asvspoof_splits(pd.read_csv(UNIFIED_CSV))
        df = {"train": train_df, "dev": dev_df, "eval": eval_df}[split]
        # processed_path is repo-relative (portable to the Kaggle symlinked layout);
        # filepath holds machine-specific absolute raw paths.
        path_col = "processed_path"
    elif dataset == "hindi":
        if split == "dev":
            raise ValueError("The Hindi track has no dev split (speaker_holdout gives train/eval).")
        train_df, eval_df = get_hindi_hinglish_splits(
            pd.read_csv(HINDI_MATCHED_CSV),
            mode="speaker_holdout",
            holdout_speaker=HOLDOUT_SPEAKER,
        )
        df = {"train": train_df, "eval": eval_df}[split]
        path_col = "filepath"
    else:
        raise ValueError(f"Unknown dataset {dataset!r}")
    return df.reset_index(drop=True), path_col


def stratified_subsample(
    df: pd.DataFrame, max_clips: Optional[int], extra_strata: Optional[str] = None
) -> pd.DataFrame:
    """Clip-level subsample preserving the label (and optional extra column) mix.

    Each stratum gets ``max_clips * share`` rows (largest-remainder rounding so
    the total is exact), sampled with a fixed seed. Never a head() or plain
    random slice.
    """
    if max_clips is None or max_clips >= len(df):
        return df
    if max_clips <= 0:
        raise ValueError(f"--max_clips must be positive, got {max_clips}")

    keys = ["label"] + ([extra_strata] if extra_strata else [])
    strata = df.groupby(keys, dropna=False, sort=True)
    sizes = strata.size()
    exact = sizes / sizes.sum() * max_clips
    alloc = np.floor(exact).astype(int)
    remainder = int(max_clips - alloc.sum())
    if remainder:
        order = (exact - alloc).sort_values(ascending=False, kind="stable").index[:remainder]
        alloc.loc[order] += 1

    parts = [
        group.sample(n=int(alloc.loc[key]), random_state=SUBSAMPLE_SEED)
        for key, group in strata
        if alloc.loc[key] > 0
    ]
    out = pd.concat(parts).sort_index().reset_index(drop=True)
    logger.info(
        "Stratified subsample on %s: %d -> %d clips; label counts %s",
        keys, len(df), len(out), out["label"].value_counts().to_dict(),
    )
    return out


def extract_chunked(
    dataset: str,
    split: str,
    model: str,
    chunk_seconds: Optional[float] = None,
    overlap_seconds: Optional[float] = None,
    max_clips: Optional[int] = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    batch_size: int = 16,
    force: bool = False,
    extractor: Optional[EmbeddingExtractor] = None,
) -> Optional[Path]:
    """Runs the full pipeline; returns the .npy path, or None if skipped as already cached."""
    output_dir = Path(output_dir)
    npy_path = output_dir / f"{model}_{dataset}_{split}_chunked.npy"
    csv_path = npy_path.with_suffix(".csv")
    if npy_path.exists() and not force:
        logger.warning(
            "Chunked embedding cache already exists at %s — skipping extraction "
            "(pass --force to recompute).", npy_path,
        )
        return None

    clips, path_col = load_clip_split(dataset, split)
    extra = None
    if dataset == "asvspoof2019":
        extra = next((c for c in ATTACK_COLUMNS if c in clips.columns), None)
        if extra is None:
            logger.warning(
                "No attack/system id column (%s) in the split; stratifying on label only.",
                ", ".join(ATTACK_COLUMNS),
            )
    clips = stratified_subsample(clips, max_clips, extra)
    if clips.empty:
        raise ValueError(f"No clips in {dataset}/{split}.")

    if extractor is None:
        extractor = EmbeddingExtractor(model_name=MODEL_NAMES[model])
    sr = config.SAMPLE_RATE
    logger.info(
        "%s/%s with %s: %d clips, chunk=%ss overlap=%ss",
        dataset, split, model, len(clips),
        config.STREAM_CHUNK_SECONDS if chunk_seconds is None else chunk_seconds,
        config.STREAM_OVERLAP_SECONDS if overlap_seconds is None else overlap_seconds,
    )

    t0 = time.time()
    emb_parts: List[np.ndarray] = []
    rows: List[dict] = []
    chunks_per_clip: List[int] = []
    zero_chunk_labels: List[str] = []
    n_silent = 0
    n_failed = 0

    for g_start in range(0, len(clips), CLIPS_PER_GROUP):
        group = clips.iloc[g_start : g_start + CLIPS_PER_GROUP]
        group_chunks: List[np.ndarray] = []
        for _, clip in group.iterrows():
            rel_path = str(clip[path_col])
            try:
                wav, _ = load_audio(_resolve_audio_path(rel_path), target_sr=sr)
            except (FileNotFoundError, RuntimeError) as exc:
                logger.warning("Skipping unreadable clip %s: %s", rel_path, exc)
                n_failed += 1
                continue
            detailed = chunk_waveform_detailed(wav, sr, chunk_seconds, overlap_seconds)
            kept = [c for c in detailed if not c.is_silent]
            n_silent += len(detailed) - len(kept)
            chunks_per_clip.append(len(kept))
            if not kept:
                zero_chunk_labels.append(str(clip["label"]))
            for c in kept:
                group_chunks.append(c.samples)
                rows.append({
                    "filepath": rel_path,
                    "label": clip["label"],  # inherited from the parent clip (see docstring)
                    "parent_filepath": rel_path,
                    "chunk_index": c.index,
                    "chunk_start_seconds": round(c.start_seconds, 6),
                })

        if group_chunks:
            durations = np.array([w.size / sr for w in group_chunks])
            emb_parts.append(
                embed_length_sorted(
                    durations, group_chunks.__getitem__, extractor,
                    batch_size=batch_size, log_progress=False,
                )
            )
        logger.info(
            "Progress: %d/%d clips, %d chunks so far (%.0fs)",
            min(g_start + CLIPS_PER_GROUP, len(clips)), len(clips), len(rows), time.time() - t0,
        )

    if not rows:
        raise ValueError(f"No non-silent chunks produced for {dataset}/{split}.")

    embeddings = np.concatenate(emb_parts, axis=0)
    manifest = pd.DataFrame(rows)
    manifest.insert(0, "row_index", np.arange(len(manifest)))
    manifest = manifest[MANIFEST_COLUMNS]
    assert len(manifest) == embeddings.shape[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(csv_path, index=False)
    tmp_path = npy_path.with_name(npy_path.stem + ".partial.npy")
    np.save(tmp_path, embeddings)
    os.replace(tmp_path, npy_path)

    cpc = np.asarray(chunks_per_clip)
    logger.info("=" * 70)
    logger.info("Saved %s (shape=%s) and %s", npy_path.name, embeddings.shape, csv_path.name)
    logger.info("Clips processed:        %d (unreadable, skipped: %d)", len(cpc), n_failed)
    logger.info("Chunks produced (kept): %d", len(manifest))
    logger.info("Chunks dropped silent:  %d", n_silent)
    logger.info(
        "Chunks per clip:        min %d / median %.1f / max %d (clips with 0 chunks: %d)",
        cpc.min(), float(np.median(cpc)), cpc.max(), int((cpc == 0).sum()),
    )
    logger.info("Chunk labels:           %s", manifest["label"].value_counts().to_dict())
    # Clips shorter than one window (or all silent) contribute nothing. If this
    # skews by label, the chunk-level class balance differs from the clip level.
    clip_labels = clips["label"].value_counts().to_dict()
    zero_by_label = pd.Series(zero_chunk_labels, dtype=object).value_counts().to_dict()
    for label, n_clips in sorted(clip_labels.items()):
        n_zero = int(zero_by_label.get(label, 0))
        logger.info(
            "Zero-chunk clips [%s]: %d / %d (%.1f%%)",
            label, n_zero, n_clips, 100.0 * n_zero / max(n_clips, 1),
        )
    logger.info("Elapsed:                %.1fs", time.time() - t0)
    logger.info("=" * 70)
    return npy_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract chunk-level embeddings (Phase F2.2).")
    parser.add_argument("--dataset", required=True, choices=["asvspoof2019", "hindi"])
    parser.add_argument("--split", required=True, choices=["train", "dev", "eval"])
    parser.add_argument("--model", required=True, choices=sorted(MODEL_NAMES))
    parser.add_argument("--chunk_seconds", type=float, default=None,
                        help="Window length (default: config.STREAM_CHUNK_SECONDS).")
    parser.add_argument("--overlap_seconds", type=float, default=None,
                        help="Window overlap (default: config.STREAM_OVERLAP_SECONDS).")
    parser.add_argument("--max_clips", type=int, default=None,
                        help="Stratified clip-level subsample size (default: all clips).")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--force", action="store_true",
                        help="Recompute and overwrite an existing cache.")
    args = parser.parse_args()

    try:
        extract_chunked(
            dataset=args.dataset,
            split=args.split,
            model=args.model,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
            max_clips=args.max_clips,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            force=args.force,
        )
    except Exception as exc:
        logger.error("Chunked extraction failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
