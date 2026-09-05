#!/usr/bin/env python3
"""
test_train_hindi_augmented_classifier.py — Unit tests for Hindi classifier head training.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from scripts.train_hindi_augmented_classifier import (
    build_variant_datasets,
    cross_validate_variant,
    load_backbone_train_data,
    train_and_save_variant,
    train_hindi_augmented_classifiers,
)
from voxguard.classifier.head import load_classifier


@pytest.fixture
def sample_mock_embeddings_dir(tmp_path: Path) -> Path:
    """Creates mock ASVspoof and Hindi embedding caches for wav2vec2 and wavlm."""
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)

    for backbone in ["wav2vec2", "wavlm"]:
        # Mock ASVspoof2019 train: 50 samples (25 real, 25 synthetic)
        X_asv = rng.randn(50, 768).astype(np.float32)
        meta_asv = pd.DataFrame({
            "row_index": range(50),
            "filepath": [f"asv_{i}.wav" for i in range(50)],
            "label": ["real"] * 25 + ["synthetic"] * 25,
        })
        np.save(emb_dir / f"{backbone}_train.npy", X_asv)
        meta_asv.to_csv(emb_dir / f"{backbone}_train.csv", index=False)

        # Mock Hindi train: 20 samples (10 real, 10 synthetic)
        X_hindi = rng.randn(20, 768).astype(np.float32)
        meta_hindi = pd.DataFrame({
            "row_index": range(20),
            "filepath": [f"hindi_{i}.wav" for i in range(20)],
            "label": ["real"] * 10 + ["synthetic"] * 10,
        })
        np.save(emb_dir / f"{backbone}_hindi_train.npy", X_hindi)
        meta_hindi.to_csv(emb_dir / f"{backbone}_hindi_train.csv", index=False)

    return emb_dir


def test_load_backbone_train_data(sample_mock_embeddings_dir: Path) -> None:
    """Tests loading cached ASV and Hindi train embeddings for a backbone."""
    X_asv, y_asv, X_hindi, y_hindi = load_backbone_train_data(
        sample_mock_embeddings_dir, "wav2vec2"
    )

    assert X_asv.shape == (50, 768)
    assert len(y_asv) == 50
    assert X_hindi.shape == (20, 768)
    assert len(y_hindi) == 20

    # Dimension mismatch check
    bad_dir = sample_mock_embeddings_dir / "bad"
    bad_dir.mkdir()
    np.save(bad_dir / "test_train.npy", np.zeros((10, 768)))
    pd.DataFrame({"row_index": range(10), "filepath": "a", "label": "real"}).to_csv(
        bad_dir / "test_train.csv", index=False
    )
    np.save(bad_dir / "test_hindi_train.npy", np.zeros((10, 512)))
    pd.DataFrame({"row_index": range(10), "filepath": "a", "label": "real"}).to_csv(
        bad_dir / "test_hindi_train.csv", index=False
    )

    with pytest.raises(ValueError, match="Feature dimensionality mismatch"):
        load_backbone_train_data(bad_dir, "test")


def test_build_variant_datasets() -> None:
    """Tests constructing Variant A (combined) and Variant B (only) datasets."""
    X_asv = np.ones((50, 768), dtype=np.float32)
    y_asv = np.array(["real"] * 25 + ["synthetic"] * 25)

    X_hindi = np.zeros((20, 768), dtype=np.float32)
    y_hindi = np.array(["real"] * 10 + ["synthetic"] * 10)

    variants = build_variant_datasets(X_asv, y_asv, X_hindi, y_hindi)

    assert "combined" in variants
    assert "only" in variants

    X_comb, y_comb = variants["combined"]
    assert X_comb.shape == (70, 768)
    assert len(y_comb) == 70
    assert np.all(X_comb[:50] == 1.0)
    assert np.all(X_comb[50:] == 0.0)

    X_only, y_only = variants["only"]
    assert X_only.shape == (20, 768)
    assert len(y_only) == 20
    assert np.all(X_only == 0.0)


def test_cross_validate_variant() -> None:
    """Tests 5-fold StratifiedKFold cross-validation metric calculation."""
    rng = np.random.RandomState(42)
    X = rng.randn(40, 768).astype(np.float32)
    y = np.array(["real"] * 20 + ["synthetic"] * 20)

    mean_acc, std_acc, fold_accs = cross_validate_variant(
        X, y, n_splits=5, random_state=42
    )

    assert len(fold_accs) == 5
    assert 0.0 <= mean_acc <= 1.0
    assert 0.0 <= std_acc <= 1.0
    assert np.isclose(mean_acc, np.mean(fold_accs))
    assert np.isclose(std_acc, np.std(fold_accs))


def test_train_and_save_variant(tmp_path: Path) -> None:
    """Tests training and saving a variant classifier model and its scaler."""
    rng = np.random.RandomState(42)
    X = rng.randn(30, 768).astype(np.float32)
    y = np.array(["real"] * 15 + ["synthetic"] * 15)

    out_target = tmp_path / "test_model_logreg"
    saved_path = train_and_save_variant(X, y, out_target)

    assert saved_path == tmp_path / "test_model_logreg.joblib"
    assert saved_path.exists()
    assert (tmp_path / "test_model_logreg_scaler.joblib").exists()
    assert (tmp_path / "test_model_logreg.json").exists()

    model, scaler = load_classifier(out_target)
    assert scaler.mean_.shape == (768,)
    X_s = scaler.transform(X)
    preds = model.predict(X_s)
    assert preds.shape == (30,)


def test_train_hindi_augmented_classifiers_end_to_end(
    sample_mock_embeddings_dir: Path,
    tmp_path: Path,
) -> None:
    """Tests end-to-end multi-backbone training script workflow."""
    out_dir = tmp_path / "classifiers"

    results = train_hindi_augmented_classifiers(
        embeddings_dir=sample_mock_embeddings_dir,
        output_dir=out_dir,
        backbones=["wav2vec2", "wavlm"],
        n_splits=3,
        random_state=42,
    )

    expected_files = [
        "wav2vec2_hindi_combined_logreg.joblib",
        "wav2vec2_hindi_combined_logreg_scaler.joblib",
        "wav2vec2_hindi_combined_logreg.json",
        "wav2vec2_hindi_only_logreg.joblib",
        "wav2vec2_hindi_only_logreg_scaler.joblib",
        "wav2vec2_hindi_only_logreg.json",
        "wavlm_hindi_combined_logreg.joblib",
        "wavlm_hindi_combined_logreg_scaler.joblib",
        "wavlm_hindi_combined_logreg.json",
        "wavlm_hindi_only_logreg.joblib",
        "wavlm_hindi_only_logreg_scaler.joblib",
        "wavlm_hindi_only_logreg.json",
    ]

    for fname in expected_files:
        assert (out_dir / fname).exists(), f"Missing expected artifact: {fname}"

    assert "wav2vec2" in results
    assert "wavlm" in results
    assert "combined" in results["wav2vec2"]
    assert "only" in results["wav2vec2"]
    assert results["wav2vec2"]["combined"]["n_samples"] == 70
    assert results["wav2vec2"]["only"]["n_samples"] == 20
