"""
cache.py — embedding extraction + on-disk caching (Phase 1).

Runs a frozen ``EmbeddingExtractor`` over a metadata DataFrame's audio files
in batches and persists the result as a single ``.npy`` embedding matrix plus
a parallel ``.csv`` sidecar mapping row index -> original filepath and label,
so cached embeddings can be matched back to metadata later without re-running
the (expensive) backbone forward pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd

from voxguard import config
from voxguard.embeddings.extractor import EmbeddingExtractor
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


def extract_and_cache(
    df: pd.DataFrame,
    extractor: EmbeddingExtractor,
    output_path: str,
    path_col: str = "processed_path",
    batch_size: int = 16,
    force: bool = False,
) -> None:
    """Extracts embeddings for every row of *df* and caches them to disk.

    Iterates *df* in batches of *batch_size*, loading each row's audio file
    (resolved from *path_col*, repo-relative paths resolved against
    ``config.BASE_DIR``) and embedding the batch via
    ``extractor.extract_batch``. The full ``(n_rows, hidden_size)`` embedding
    matrix is saved as a ``.npy`` file at *output_path*, alongside a ``.csv``
    sidecar (same base filename) with columns ``[row_index, filepath, label]``
    so embeddings can be matched back to metadata later.

    Resumable: if *output_path* already exists, extraction is skipped and a
    warning is logged instead of recomputing — pass ``force=True`` to
    recompute and overwrite.

    Parameters
    ----------
    df:
        Metadata DataFrame. Must contain *path_col* and a ``label`` column.
    extractor:
        A loaded ``EmbeddingExtractor`` used to embed each batch.
    output_path:
        Destination ``.npy`` file path. The sidecar CSV is written next to
        it, same base filename, ``.csv`` extension.
    path_col:
        Column in *df* holding the audio file path to embed (repo-relative
        or absolute). Defaults to ``"processed_path"``.
    batch_size:
        Number of files embedded per ``extractor.extract_batch`` call.
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
            "Embedding cache already exists at %s — skipping extraction "
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
        raise ValueError("Cannot extract embeddings from an empty DataFrame.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_batches = (n_rows + batch_size - 1) // batch_size
    logger.info(
        "Extracting embeddings for %d files in %d batches (batch_size=%d) using %r",
        n_rows,
        n_batches,
        batch_size,
        extractor,
    )

    embedding_batches: list[np.ndarray] = []
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_rows)
        batch_paths = df[path_col].iloc[start:end]

        waveforms = [
            load_audio(_resolve_audio_path(p), target_sr=config.SAMPLE_RATE)[0]
            for p in batch_paths
        ]
        embedding_batches.append(extractor.extract_batch(waveforms))

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
            logger.info(
                "Progress: batch [%d/%d] (%d/%d files embedded)",
                batch_idx + 1,
                n_batches,
                end,
                n_rows,
            )

    embeddings = np.concatenate(embedding_batches, axis=0)
    np.save(output_path, embeddings)

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
        "Saved %d embeddings (shape=%s) to %s and index map to %s",
        n_rows,
        embeddings.shape,
        output_path,
        csv_path,
    )


def load_cached_embeddings(path: Union[str, Path]) -> Tuple[np.ndarray, pd.DataFrame]:
    """Loads a cached embedding matrix and its matching index-map CSV.

    Parameters
    ----------
    path:
        Path to the cached embedding file — either the ``.npy`` file or its
        ``.csv`` sidecar; the companion is derived from the same base
        filename either way.

    Returns
    -------
    embeddings : np.ndarray, shape ``(n_rows, hidden_size)``
    index_map : pd.DataFrame
        Columns ``[row_index, filepath, label]``, row-aligned with
        *embeddings*.

    Raises
    ------
    FileNotFoundError
        If either the ``.npy`` or ``.csv`` file is missing.
    ValueError
        If the two files' row counts don't match.
    """
    path = Path(path)
    npy_path = path.with_suffix(".npy")
    csv_path = path.with_suffix(".csv")

    if not npy_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {npy_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Embedding index-map CSV not found: {csv_path}")

    embeddings = np.load(npy_path)
    index_map = pd.read_csv(csv_path)

    if len(embeddings) != len(index_map):
        raise ValueError(
            f"Embedding matrix rows ({len(embeddings)}) do not match "
            f"index map rows ({len(index_map)}) for '{path}'."
        )

    logger.info(
        "Loaded %d cached embeddings (shape=%s) from %s", len(embeddings), embeddings.shape, npy_path
    )
    return embeddings, index_map
