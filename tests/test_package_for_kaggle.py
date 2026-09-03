"""
Unit tests for scripts/package_for_kaggle.py.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from scripts.package_for_kaggle import (
    package_and_upload,
    prepare_kaggle_package,
    resolve_dataset_slug,
)


@pytest.fixture
def sample_package_environment(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Creates temporary preprocessed audio and metadata directories."""
    proc_dir = tmp_path / "processed"
    asv_dir = proc_dir / "asvspoof2019"
    asv_dir.mkdir(parents=True, exist_ok=True)
    (asv_dir / "clip1.wav").write_bytes(b"RIFFdummywav1")
    (asv_dir / "clip2.wav").write_bytes(b"RIFFdummywav2")

    wf_dir = proc_dir / "wavefake"
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / "wf_clip1.wav").write_bytes(b"RIFFdummywav3")

    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_csv = meta_dir / "unified.csv"
    pd.DataFrame([
        {"filepath": "dummy1.wav", "label": "real", "dataset": "asvspoof2019", "split": "train"},
        {"filepath": "dummy2.wav", "label": "synthetic", "dataset": "wavefake", "split": "eval"},
    ]).to_csv(meta_csv, index=False)

    staging_dir = tmp_path / "kaggle_staging"
    return proc_dir, meta_csv, staging_dir


def test_prepare_kaggle_package_structure(
    sample_package_environment: tuple[Path, Path, Path]
) -> None:
    """Tests that staging folder contains dataset subfolders and metadata/unified.csv."""
    proc_dir, meta_csv, staging_dir = sample_package_environment

    staged_path = prepare_kaggle_package(
        processed_dir=proc_dir,
        metadata_csv=meta_csv,
        staging_dir=staging_dir,
    )

    assert staged_path.exists()

    # Check metadata/unified.csv
    staged_meta = staged_path / "metadata" / "unified.csv"
    assert staged_meta.exists()

    # Check audio folders
    staged_asv = staged_path / "asvspoof2019"
    assert staged_asv.exists()
    assert (staged_asv / "clip1.wav").exists()
    assert (staged_asv / "clip2.wav").exists()

    staged_wf = staged_path / "wavefake"
    assert staged_wf.exists()
    assert (staged_wf / "wf_clip1.wav").exists()


def test_resolve_dataset_slug() -> None:
    """Tests dataset slug parsing and username prefixing."""
    # When full slug is given
    assert resolve_dataset_slug("testuser/my-dataset") == "testuser/my-dataset"

    # When short slug is given with mocked username
    with patch("scripts.package_for_kaggle.get_default_kaggle_username", return_value="mockuser"):
        assert resolve_dataset_slug("my-dataset") == "mockuser/my-dataset"
        assert resolve_dataset_slug(None) == "mockuser/voxguard-preprocessed-data"


def test_package_and_upload_stage_only(
    sample_package_environment: tuple[Path, Path, Path]
) -> None:
    """Tests stage_only execution without attempting upload."""
    proc_dir, meta_csv, staging_dir = sample_package_environment

    result = package_and_upload(
        processed_dir=proc_dir,
        metadata_csv=meta_csv,
        staging_dir=staging_dir,
        stage_only=True,
    )

    assert result == str(staging_dir.resolve())
    assert (staging_dir / "metadata" / "unified.csv").exists()
