"""
Unit tests for scripts/package_for_kaggle.py (v4 combined dataset spec).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from scripts.package_for_kaggle import (
    canonicalize_dataset_folder_name,
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

    # Even if source directory were named wavefake_subset, it should stage to wavefake/
    wf_dir = proc_dir / "wavefake"
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / "wf_clip1.wav").write_bytes(b"RIFFdummywav3")

    itw_dir = proc_dir / "in_the_wild"
    itw_dir.mkdir(parents=True, exist_ok=True)
    (itw_dir / "itw_clip1.wav").write_bytes(b"RIFFdummywav4")

    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_csv = meta_dir / "unified.csv"
    pd.DataFrame([
        {"filepath": "dummy1.wav", "label": "real", "dataset": "asvspoof2019", "split": "train"},
        {"filepath": "dummy2.wav", "label": "synthetic", "dataset": "wavefake", "split": "eval"},
        {"filepath": "dummy3.wav", "label": "synthetic", "dataset": "in_the_wild", "split": "eval"},
    ]).to_csv(meta_csv, index=False)

    staging_dir = tmp_path / "kaggle_staging"
    return proc_dir, meta_csv, staging_dir


def test_canonicalize_dataset_folder_name() -> None:
    """Tests that folder names are mapped to exact canonical names (never wavefake_subset)."""
    assert canonicalize_dataset_folder_name("wavefake") == "wavefake"
    assert canonicalize_dataset_folder_name("wavefake_subset") == "wavefake"
    assert canonicalize_dataset_folder_name("wf") == "wavefake"
    assert canonicalize_dataset_folder_name("asvspoof") == "asvspoof2019"
    assert canonicalize_dataset_folder_name("asvspoof2019") == "asvspoof2019"
    assert canonicalize_dataset_folder_name("in_the_wild") == "in_the_wild"
    assert canonicalize_dataset_folder_name("in-the-wild") == "in_the_wild"
    assert canonicalize_dataset_folder_name("itw") == "in_the_wild"


def test_prepare_kaggle_package_structure(
    sample_package_environment: tuple[Path, Path, Path]
) -> None:
    """Tests that staging folder contains exactly the 3 canonical dataset subfolders and metadata/unified.csv."""
    proc_dir, meta_csv, staging_dir = sample_package_environment

    staged_path = prepare_kaggle_package(
        processed_dir=proc_dir,
        metadata_csv=meta_csv,
        staging_dir=staging_dir,
    )

    assert staged_path.exists()

    # Check single top-level metadata/unified.csv
    staged_meta = staged_path / "metadata" / "unified.csv"
    assert staged_meta.exists()

    # Check the 3 canonical audio subfolders
    staged_asv = staged_path / "asvspoof2019"
    assert staged_asv.exists()
    assert (staged_asv / "clip1.wav").exists()
    assert (staged_asv / "clip2.wav").exists()

    staged_wf = staged_path / "wavefake"
    assert staged_wf.exists()
    assert (staged_wf / "wf_clip1.wav").exists()

    staged_itw = staged_path / "in_the_wild"
    assert staged_itw.exists()
    assert (staged_itw / "itw_clip1.wav").exists()

    # Ensure no separate individual per-dataset CSVs are created
    assert not (staged_asv / "asvspoof2019.csv").exists()
    assert not (staged_wf / "wavefake.csv").exists()


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
