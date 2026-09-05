"""
hindi_splits.py — Partition splits for the Hindi/Hinglish self-collected track.

Provides speaker-aware and utterance-stratified partitioning strategies for Phase 4:
- `mode="speaker_holdout"`: Excludes an entire speaker from training, holding them out
  strictly for evaluation. This provides a genuine zero-shot "generalization to unseen voices"
  benchmark (recommended when 3+ speakers are available).
- `mode="utterance_stratified"`: 80/20 train/eval stratified split by sentence category.
  Used as a fallback for 1-2 speaker setups. Note: evaluation results on this split reflect
  clone detection on partially-seen voices (same speaker in train and eval), not generalization
  to new speakers.
"""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from voxguard.utils.logging_utils import get_logger

logger = get_logger("hindi_splits")

SUPPORTED_MODES = {"speaker_holdout", "utterance_stratified"}


def get_hindi_hinglish_splits(
    track_df: pd.DataFrame,
    mode: str = "speaker_holdout",
    holdout_speaker: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Partitions the Hindi/Hinglish track into train and evaluation subsets.

    Parameters
    ----------
    track_df:
        DataFrame containing Hindi/Hinglish records with columns `speaker_id`,
        `label`, and `category`.
    mode:
        Split strategy:
        - "speaker_holdout": Holds out `holdout_speaker` entirely for eval (unseen speaker test).
        - "utterance_stratified": 80/20 category-stratified split across all utterances.
    holdout_speaker:
        The speaker_id to hold out for evaluation when `mode="speaker_holdout"`.
    test_size:
        Fraction of data for eval when `mode="utterance_stratified"` (default: 0.2).
    random_state:
        Random seed for reproducible shuffling in stratified mode.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]:
        (train_df, eval_df)

    Raises
    ------
    ValueError:
        If parameters are invalid, holdout speaker is missing/not found, or dataframe has < 2 speakers.
    """
    if track_df is None or track_df.empty:
        raise ValueError("Cannot split an empty or null Hindi/Hinglish track DataFrame.")

    required_cols = {"speaker_id", "label", "category"}
    missing_cols = required_cols - set(track_df.columns)
    if missing_cols:
        raise ValueError(f"track_df is missing required columns: {missing_cols}")

    clean_mode = str(mode).strip().lower()

    if clean_mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Unknown split mode: '{mode}'. Supported modes are: {sorted(SUPPORTED_MODES)}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Mode 1: Speaker Holdout (Unseen Speaker Generalization Test)
    # ─────────────────────────────────────────────────────────────────────────
    if clean_mode == "speaker_holdout":
        if not holdout_speaker or not str(holdout_speaker).strip():
            raise ValueError(
                "holdout_speaker must be specified when mode='speaker_holdout' "
                "(e.g. holdout_speaker='soumya')."
            )

        target_holdout = str(holdout_speaker).strip()
        available_speakers = sorted(track_df["speaker_id"].dropna().unique().tolist())

        if len(available_speakers) < 2:
            raise ValueError(
                f"speaker_holdout mode requires at least 2 unique speakers in track_df, "
                f"found {len(available_speakers)}: {available_speakers}."
            )

        if target_holdout not in available_speakers:
            raise ValueError(
                f"holdout_speaker '{target_holdout}' not found in track_df. "
                f"Available speakers: {available_speakers}"
            )

        train_df = track_df[track_df["speaker_id"] != target_holdout].copy().reset_index(drop=True)
        eval_df = track_df[track_df["speaker_id"] == target_holdout].copy().reset_index(drop=True)

        train_speakers = sorted(train_df["speaker_id"].unique().tolist())
        eval_speakers = sorted(eval_df["speaker_id"].unique().tolist())

        # Log split honesty metrics
        logger.info("=" * 78)
        logger.info("HINDI/HINGLISH SPLIT: Mode = 'speaker_holdout' (Unseen Speaker Generalization)")
        logger.info("=" * 78)
        logger.info(f"Train Set : {len(train_df):>3} clips | Speakers: {train_speakers}")
        logger.info(f"            - Real: {(train_df['label'] == 'real').sum()}, Synthetic: {(train_df['label'] == 'synthetic').sum()}")
        logger.info(f"Eval Set  : {len(eval_df):>3} clips | Held-out Speaker: '{target_holdout}' ({eval_speakers})")
        logger.info(f"            - Real: {(eval_df['label'] == 'real').sum()}, Synthetic: {(eval_df['label'] == 'synthetic').sum()}")
        logger.info("Honesty   : Eval evaluates detector on an UNSEEN voice absent from train.")
        logger.info("=" * 78)

        return train_df, eval_df

    # ─────────────────────────────────────────────────────────────────────────
    # Mode 2: Utterance Stratified (Partially-Seen Voice / Intra-Speaker Test)
    # ─────────────────────────────────────────────────────────────────────────
    stratify_key = track_df["category"].astype(str) + "_" + track_df["label"].astype(str)

    train_df, eval_df = train_test_split(
        track_df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_key,
    )
    train_df = train_df.copy().reset_index(drop=True)
    eval_df = eval_df.copy().reset_index(drop=True)

    # Log split honesty metrics and limitation warning
    logger.warning("=" * 78)
    logger.warning("HINDI/HINGLISH SPLIT: Mode = 'utterance_stratified' (Partially-Seen Voices)")
    logger.warning("=" * 78)
    logger.warning(
        "LIMITATION NOTE: Utterance-stratified eval reflects clone detection on a "
        "PARTIALLY-SEEN voice (speakers are shared between train and eval). "
        "It does NOT evaluate generalization to an unseen voice."
    )
    logger.info(f"Train Set : {len(train_df):>3} clips (Real: {(train_df['label'] == 'real').sum()}, Synthetic: {(train_df['label'] == 'synthetic').sum()})")
    logger.info(f"Eval Set  : {len(eval_df):>3} clips (Real: {(eval_df['label'] == 'real').sum()}, Synthetic: {(eval_df['label'] == 'synthetic').sum()})")
    logger.warning("=" * 78)

    return train_df, eval_df
