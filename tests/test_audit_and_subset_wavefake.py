"""
Unit tests for scripts/audit_and_subset_wavefake.py.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.audit_and_subset_wavefake import (
    audit_and_subset,
    format_dataset_card_table,
    load_dataset,
    stratified_subset_wavefake,
)


@pytest.fixture
def sample_wavefake_df() -> pd.DataFrame:
    """Creates a synthetic WaveFake DataFrame matching the actual distribution schema."""
    generators = {
        "original": ("bonafide", 500),
        "parallel_wavegan": ("spoof", 1200),
        "multi_band_melgan": ("spoof", 700),
        "full_band_melgan": ("spoof", 500),
        "hifiGAN": ("spoof", 500),
        "melgan": ("spoof", 500),
        "melgan_large": ("spoof", 500),
        "waveglow": ("spoof", 500),
    }
    records = []
    for gen, (label, count) in generators.items():
        for i in range(count):
            records.append({
                "filepath": f"/dummy/path/{gen}/{label}_{i}.wav",
                "label": label,
                "generator": gen,
            })
    return pd.DataFrame(records)


def test_stratified_subset_size_and_schema(sample_wavefake_df: pd.DataFrame) -> None:
    """Tests that stratified subset produces exact target size and expected columns."""
    target_size = 500
    subset_df = stratified_subset_wavefake(
        sample_wavefake_df, subset_size=target_size, random_state=42
    )

    assert len(subset_df) == target_size
    assert list(subset_df.columns) == ["filepath", "label", "generator"]
    assert subset_df["label"].isin(["bonafide", "spoof"]).all()


def test_stratification_proportions(sample_wavefake_df: pd.DataFrame) -> None:
    """Tests that bonafide and per-generator proportions are preserved within statistical tolerance."""
    total_samples = len(sample_wavefake_df)
    target_size = 1000
    subset_df = stratified_subset_wavefake(
        sample_wavefake_df, subset_size=target_size, random_state=42
    )

    # Check bonafide proportion
    orig_bona_ratio = (sample_wavefake_df["label"] == "bonafide").mean()
    sub_bona_ratio = (subset_df["label"] == "bonafide").mean()
    assert abs(orig_bona_ratio - sub_bona_ratio) < 0.02

    # Check generator proportions
    for gen in sample_wavefake_df["generator"].unique():
        orig_gen_ratio = (sample_wavefake_df["generator"] == gen).mean()
        sub_gen_ratio = (subset_df["generator"] == gen).mean()
        assert abs(orig_gen_ratio - sub_gen_ratio) < 0.02


def test_small_generator_graceful_fallback() -> None:
    """Tests that strata with fewer than 20 samples trigger graceful handling without crashing."""
    records = []
    # Dominant class
    for i in range(1000):
        records.append({"filepath": f"/d/{i}.wav", "label": "spoof", "generator": "melgan"})
    # Normal bonafide class
    for i in range(200):
        records.append({"filepath": f"/d/b_{i}.wav", "label": "bonafide", "generator": "original"})
    # Extremely rare generator (<20 samples)
    for i in range(5):
        records.append({"filepath": f"/d/rare_{i}.wav", "label": "spoof", "generator": "rare_tts"})

    df = pd.DataFrame(records)
    target_size = 300
    subset_df = stratified_subset_wavefake(df, subset_size=target_size, random_state=42)

    assert len(subset_df) == target_size
    assert "rare_tts" in subset_df["generator"].values


def test_format_dataset_card_table(sample_wavefake_df: pd.DataFrame) -> None:
    """Tests formatting of the dataset-card markdown table."""
    subset_df = stratified_subset_wavefake(sample_wavefake_df, subset_size=500, random_state=42)
    table_str = format_dataset_card_table(sample_wavefake_df, subset_df)

    assert "### WaveFake Dataset Subsetting Card" in table_str
    assert "| Generator / Vocoder | Label |" in table_str
    assert "TOTAL BONAFIDE" in table_str
    assert "TOTAL SPOOF" in table_str
    assert "TOTAL (ALL)" in table_str


def test_audit_and_subset_e2e(tmp_path: Path, sample_wavefake_df: pd.DataFrame) -> None:
    """Tests full end-to-end audit and subset saving pipeline."""
    wf_path = tmp_path / "wavefake.csv"
    itw_path = tmp_path / "in_the_wild.csv"
    out_path = tmp_path / "wavefake_subset.csv"

    sample_wavefake_df.to_csv(wf_path, index=False)
    # Write mock ITW
    itw_df = pd.DataFrame([
        {"filepath": "/itw/1.wav", "label": "bonafide", "speaker": "spk1"},
        {"filepath": "/itw/2.wav", "label": "spoof", "speaker": "spk2"},
    ])
    itw_df.to_csv(itw_path, index=False)

    df_subset = audit_and_subset(
        wavefake_csv=wf_path,
        itw_csv=itw_path,
        output_csv=out_path,
        subset_size=400,
        random_state=42,
        verbose=False,
    )

    # Verify subset file was created
    assert out_path.exists()
    assert len(df_subset) == 400

    # Verify original wavefake file was not mutated
    df_orig_reloaded = pd.read_csv(wf_path)
    assert len(df_orig_reloaded) == len(sample_wavefake_df)
