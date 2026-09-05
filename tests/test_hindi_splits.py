"""
Unit tests for src/voxguard/utils/hindi_splits.py.
"""

from __future__ import annotations

import pandas as pd
import pytest

from voxguard.utils.hindi_splits import get_hindi_hinglish_splits


@pytest.fixture
def sample_track_dataframe() -> pd.DataFrame:
    """Creates a sample multi-speaker Hindi/Hinglish track DataFrame."""
    records = []
    speakers = ["alice", "bob", "charlie"]
    categories = ["neutral", "scam", "control"]

    for spk in speakers:
        for cat in categories:
            for i in range(1, 11):
                records.append({
                    "filepath": f"data/raw/hindi_hinglish/real/{spk}_{cat}_{i:02d}.wav",
                    "speaker_id": spk,
                    "category": cat,
                    "sentence_id": i,
                    "label": "real",
                    "dataset": "hindi_hinglish",
                })
                records.append({
                    "filepath": f"data/raw/hindi_hinglish/synthetic/{spk}_{cat}_{i:02d}_clone.wav",
                    "speaker_id": spk,
                    "category": cat,
                    "sentence_id": i,
                    "label": "synthetic",
                    "dataset": "hindi_hinglish",
                })
    return pd.DataFrame(records)


def test_speaker_holdout_split_integrity(sample_track_dataframe: pd.DataFrame) -> None:
    """Tests speaker_holdout mode completely isolates the held-out speaker into eval."""
    df = sample_track_dataframe
    train_df, eval_df = get_hindi_hinglish_splits(
        track_df=df,
        mode="speaker_holdout",
        holdout_speaker="charlie",
    )

    # 1. Total records preserved
    assert len(train_df) + len(eval_df) == len(df)

    # 2. Speaker disjointness (zero speaker leakage)
    assert "charlie" not in train_df["speaker_id"].values
    assert set(eval_df["speaker_id"].unique()) == {"charlie"}
    assert set(train_df["speaker_id"].unique()) == {"alice", "bob"}

    # 3. Balance preserved
    assert (train_df["label"] == "real").sum() == (train_df["label"] == "synthetic").sum()
    assert (eval_df["label"] == "real").sum() == (eval_df["label"] == "synthetic").sum()


def test_speaker_holdout_missing_speaker_error(sample_track_dataframe: pd.DataFrame) -> None:
    """Tests that missing holdout_speaker parameter raises ValueError."""
    with pytest.raises(ValueError, match="holdout_speaker must be specified"):
        get_hindi_hinglish_splits(sample_track_dataframe, mode="speaker_holdout", holdout_speaker=None)


def test_speaker_holdout_unknown_speaker_error(sample_track_dataframe: pd.DataFrame) -> None:
    """Tests that an unknown holdout_speaker raises ValueError."""
    with pytest.raises(ValueError, match="not found in track_df"):
        get_hindi_hinglish_splits(
            sample_track_dataframe,
            mode="speaker_holdout",
            holdout_speaker="nonexistent_speaker",
        )


def test_speaker_holdout_insufficient_speakers_error() -> None:
    """Tests that single-speaker dataset raises ValueError in speaker_holdout mode."""
    single_speaker_df = pd.DataFrame([
        {"filepath": "f1.wav", "speaker_id": "alice", "category": "neutral", "label": "real"},
        {"filepath": "f2.wav", "speaker_id": "alice", "category": "neutral", "label": "synthetic"},
    ])
    with pytest.raises(ValueError, match="requires at least 2 unique speakers"):
        get_hindi_hinglish_splits(
            single_speaker_df,
            mode="speaker_holdout",
            holdout_speaker="alice",
        )


def test_utterance_stratified_split(sample_track_dataframe: pd.DataFrame) -> None:
    """Tests utterance_stratified mode produces 80/20 train/eval partition."""
    df = sample_track_dataframe
    train_df, eval_df = get_hindi_hinglish_splits(
        track_df=df,
        mode="utterance_stratified",
        test_size=0.2,
        random_state=42,
    )

    total_len = len(df)
    assert len(train_df) + len(eval_df) == total_len
    assert len(eval_df) == pytest.approx(total_len * 0.2, abs=2)


def test_invalid_mode_raises_error(sample_track_dataframe: pd.DataFrame) -> None:
    """Tests that unrecognized mode raises ValueError."""
    with pytest.raises(ValueError, match="Unknown split mode"):
        get_hindi_hinglish_splits(sample_track_dataframe, mode="invalid_random_mode")
