"""
compose.py — prosody feature caching + cross-source feature composition (Phase 2).

Two responsibilities:

1. ``extract_and_cache_prosody`` mirrors the resumable extract-and-cache
   pattern from ``voxguard.embeddings.cache.extract_and_cache`` (Phase 1),
   but runs ``ProsodyFeatureExtractor`` instead of an embedding model. Pure
   CPU librosa work with no batching or GPU forward pass, so it's expected
   to run in minutes, not hours, and should always be run LOCALLY — never
   on Kaggle. A GPU notebook buys nothing here, and routing it through the
   Kaggle workflow would add a dependency for zero speed gain.

   Naming convention: cache ASVspoof2019 prosody outputs as
   ``models/embeddings/prosody_{split}.npy`` (``prosody_train.npy``,
   ``prosody_dev.npy``, ``prosody_eval.npy``), mirroring
   ``scripts/extract_embeddings.py``'s ``{model}_{split}.npy`` convention.

2. ``load_combined_features`` is the ONE place in the project where
   different cached feature sources (e.g. a wav2vec2 embedding cache and a
   prosody cache) are horizontally concatenated into a single feature
   matrix. Phase 3's ensembling and Phase 4's Hindi training should both
   call this function rather than writing their own concatenation logic,
   so a bug fix here fixes every phase at once.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from voxguard import config
from voxguard.embeddings.cache import load_cached_embeddings
from voxguard.features.prosody import ProsodyFeatureExtractor
from voxguard.utils.audio_io import load_audio
from voxguard.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _resolve_audio_path(path_str: str) -> Path:
    """Resolves a (possibly repo-relative) audio path to an absolute path.

    ``processed_path`` values are stored repo-relative (e.g.
    ``"data/processed/asvspoof2019/xxx.wav"``); already-absolute paths are
    returned unchanged.
    """
    p = Path(str(path_str))
    return p if p.is_absolute() else (config.BASE_DIR / p)


def extract_and_cache_prosody(
    df: pd.DataFrame,
    output_path: str,
    path_col: str = "processed_path",
    force: bool = False,
) -> None:
    """Extracts prosody features for every row of *df* and caches them to disk.

    Iterates *df* file-by-file (no batching — there's no model forward pass
    to amortize), loading each row's audio via *path_col* (repo-relative
    paths resolved against ``config.BASE_DIR``) and extracting a 10-dim
    vector via ``ProsodyFeatureExtractor``. The resulting ``(n_rows, 10)``
    matrix is saved as a ``.npy`` file at *output_path*, alongside a ``.csv``
    sidecar (same base filename) with columns ``[row_index, filepath, label]``
    — the same cache format used by ``voxguard.embeddings.cache``, so both
    can be loaded and combined via ``load_combined_features``.

    Resumable: if *output_path* already exists, extraction is skipped and a
    warning is logged instead of recomputing — pass ``force=True`` to
    recompute and overwrite.

    Run this locally, not on Kaggle: it's pure CPU librosa work and gains
    nothing from a GPU notebook.

    Parameters
    ----------
    df:
        Metadata DataFrame. Must contain *path_col* and a ``label`` column.
    output_path:
        Destination ``.npy`` file path. The sidecar CSV is written next to
        it, same base filename, ``.csv`` extension. Recommended convention
        for ASVspoof2019: ``models/embeddings/prosody_{split}.npy``.
    path_col:
        Column in *df* holding the audio file path to process (repo-relative
        or absolute). Defaults to ``"processed_path"``.
    force:
        If ``True``, recompute and overwrite even if *output_path* already
        exists.

    Returns
    -------
    None
    """
    output_path = Path(output_path)
    csv_path = output_path.with_suffix(".csv")

    if output_path.exists() and not force:
        logger.warning(
            "Prosody feature cache already exists at %s — skipping extraction "
            "(pass force=True to recompute).",
            output_path,
        )
        return

    if path_col not in df.columns:
        raise ValueError(f"DataFrame is missing required path column: '{path_col}'")
    if "label" not in df.columns:
        raise ValueError("DataFrame is missing required 'label' column.")

    df = df.reset_index(drop=True)
    n_rows = len(df)
    if n_rows == 0:
        raise ValueError("Cannot extract prosody features from an empty DataFrame.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    extractor = ProsodyFeatureExtractor()
    n_features = len(extractor.FEATURE_NAMES)
    feature_matrix = np.zeros((n_rows, n_features), dtype=np.float32)

    logger.info(
        "Extracting prosody features for %d files (CPU-only librosa, local run) into %d-dim vectors",
        n_rows,
        n_features,
    )

    for i, row_path in enumerate(df[path_col]):
        waveform, sr = load_audio(_resolve_audio_path(row_path), target_sr=config.SAMPLE_RATE)
        feature_matrix[i] = extractor.extract(waveform, sr)

        if (i + 1) % 500 == 0 or (i + 1) == n_rows:
            logger.info("Progress: [%d/%d] files processed", i + 1, n_rows)

    np.save(output_path, feature_matrix)

    filepath_col = "filepath" if "filepath" in df.columns else path_col
    index_map = pd.DataFrame(
        {
            "row_index": np.arange(n_rows),
            "filepath": df[filepath_col].values,
            "label": df["label"].values,
        }
    )
    index_map.to_csv(csv_path, index=False)

    logger.info(
        "Saved %d prosody feature vectors (shape=%s) to %s and index map to %s",
        n_rows,
        feature_matrix.shape,
        output_path,
        csv_path,
    )


def load_combined_features(cached_paths: List[str]) -> Tuple[np.ndarray, pd.DataFrame]:
    """Loads and horizontally concatenates two or more cached feature sources.

    This is the ONE place in the project where feature concatenation
    happens (e.g. combining a wav2vec2 embedding cache with a prosody
    cache) — all downstream code (Phase 3 ensembling, Phase 4 Hindi
    training, ...) should call this rather than concatenating caches
    itself, so a bug fix here fixes every phase at once.

    Each cached source is loaded via
    ``voxguard.embeddings.cache.load_cached_embeddings`` (both the
    embedding cache and the prosody cache use the same ``.npy`` + ``.csv``
    format). Manifests are validated to share the exact same filepaths in
    the exact same row order before concatenating — no reindexing or fuzzy
    join is attempted, since a silent misalignment here would corrupt every
    downstream training run without any visible error.

    Parameters
    ----------
    cached_paths:
        Two or more paths to cached ``.npy`` (or ``.csv``) files, e.g.
        ``["models/embeddings/wav2vec2_train.npy",
        "models/embeddings/prosody_train.npy"]``.

    Returns
    -------
    combined : np.ndarray, shape ``(n_rows, sum_of_feature_dims)``
        The horizontally-concatenated feature matrix.
    manifest : pd.DataFrame
        The shared ``[row_index, filepath, label]`` manifest (identical
        across all inputs, by construction of the alignment check).

    Raises
    ------
    ValueError
        If fewer than two paths are given, or if any two manifests don't
        share identical row counts, filepaths (in order), or labels.
    """
    if len(cached_paths) < 2:
        raise ValueError(
            "load_combined_features requires at least two cached (.npy, .csv) pairs to combine; "
            f"got {len(cached_paths)}."
        )

    feature_matrices: List[np.ndarray] = []
    manifests: List[pd.DataFrame] = []
    for p in cached_paths:
        emb, manifest = load_cached_embeddings(p)
        feature_matrices.append(emb)
        manifests.append(manifest)

    reference_path = cached_paths[0]
    reference_manifest = manifests[0]

    for other_path, other_manifest in zip(cached_paths[1:], manifests[1:]):
        if len(other_manifest) != len(reference_manifest):
            raise ValueError(
                f"Cannot combine cached features: '{reference_path}' has "
                f"{len(reference_manifest):,} rows but '{other_path}' has "
                f"{len(other_manifest):,} rows. Caches must come from the same "
                "DataFrame/split to be combined."
            )

        ref_filepaths = reference_manifest["filepath"].values
        other_filepaths = other_manifest["filepath"].values
        mismatches = np.flatnonzero(ref_filepaths != other_filepaths)
        if mismatches.size > 0:
            first = int(mismatches[0])
            raise ValueError(
                f"Cannot combine cached features: '{reference_path}' and '{other_path}' "
                f"do not share the same filepaths in the same row order "
                f"({mismatches.size} mismatched row(s)). First mismatch at row {first}: "
                f"'{ref_filepaths[first]}' (from {reference_path}) vs "
                f"'{other_filepaths[first]}' (from {other_path})."
            )

        ref_labels = reference_manifest["label"].values
        other_labels = other_manifest["label"].values
        label_mismatches = np.flatnonzero(ref_labels != other_labels)
        if label_mismatches.size > 0:
            first = int(label_mismatches[0])
            raise ValueError(
                f"Cannot combine cached features: '{reference_path}' and '{other_path}' "
                f"disagree on label at row {first} (filepath="
                f"'{ref_filepaths[first]}'): '{ref_labels[first]}' vs '{other_labels[first]}'."
            )

    combined = np.concatenate(feature_matrices, axis=1)

    logger.info(
        "Combined %d feature sources into shape=%s for %d rows: %s",
        len(cached_paths),
        combined.shape,
        len(reference_manifest),
        cached_paths,
    )

    return combined, reference_manifest
