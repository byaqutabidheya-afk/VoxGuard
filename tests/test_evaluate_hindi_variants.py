#!/usr/bin/env python3
"""
test_evaluate_hindi_variants.py — Unit tests for Hindi classifier evaluation and reporting.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.evaluate_hindi_variants import (
    VARIANTS_CONFIG,
    _format_cell,
    build_comparison_report,
    evaluate_all_variants,
    evaluate_variant,
)


def test_format_cell() -> None:
    """Tests accuracy and EER formatting helper."""
    assert _format_cell({"accuracy": 0.915, "eer": 0.079}) == "91.50% / 7.90%"
    assert _format_cell({"accuracy": 1.0, "eer": 0.0}) == "100.00% / 0.00%"
    assert _format_cell({"accuracy": None, "eer": 0.05}) == "n/a"
    assert _format_cell({"accuracy": float("nan"), "eer": 0.05}) == "n/a"


def test_evaluate_variant_dispatch(tmp_path: Path) -> None:
    """Tests that evaluate_variant dispatches correctly for single and ensemble types."""
    single_cfg = VARIANTS_CONFIG[0]  # wav2vec2 baseline
    ensemble_cfg = VARIANTS_CONFIG[2]  # Weighted-Average Ensemble (1 + 2)

    mock_metrics = {
        "accuracy": 0.90,
        "roc_auc": 0.95,
        "eer": 0.08,
        "eer_threshold": 0.5,
        "confusion_matrix": [[45, 5], [5, 45]],
    }

    with patch(
        "scripts.evaluate_hindi_variants.zero_shot_eval_from_cache",
        return_value=mock_metrics,
    ) as mock_single, patch(
        "scripts.evaluate_hindi_variants.zero_shot_eval_weighted_average_from_cache",
        return_value=mock_metrics,
    ) as mock_ens:
        res_single = evaluate_variant(single_cfg, "asvspoof2019", tmp_path, split="eval")
        assert res_single["accuracy"] == 0.90
        mock_single.assert_called_once()

        res_ens = evaluate_variant(ensemble_cfg, "hindi_eval", tmp_path, split="eval")
        assert res_ens["eer"] == 0.08
        mock_ens.assert_called_once()


def test_evaluate_all_variants_returns_9_rows(tmp_path: Path) -> None:
    """Tests that evaluate_all_variants evaluates all 9 variants on both datasets."""
    mock_metrics = {
        "accuracy": 0.90,
        "roc_auc": 0.95,
        "eer": 0.08,
        "eer_threshold": 0.5,
        "confusion_matrix": [[45, 5], [5, 45]],
    }

    with patch(
        "scripts.evaluate_hindi_variants.zero_shot_eval_from_cache",
        return_value=mock_metrics,
    ), patch(
        "scripts.evaluate_hindi_variants.zero_shot_eval_weighted_average_from_cache",
        return_value=mock_metrics,
    ):
        results = evaluate_all_variants(classifiers_dir=tmp_path)

        assert len(results) == 9
        for i, row in enumerate(results, start=1):
            assert row["id"] == i
            assert "asvspoof_metrics" in row
            assert "hindi_metrics" in row


def test_build_comparison_report() -> None:
    """Tests markdown report generation and required section presence."""
    mock_results = []
    for cfg in VARIANTS_CONFIG:
        mock_results.append({
            "id": cfg["id"],
            "name": cfg["name"],
            "description": cfg["description"],
            "asvspoof_metrics": {"accuracy": 0.9151, "eer": 0.0767},
            "hindi_metrics": {"accuracy": 0.9600, "eer": 0.0200},
        })

    report = build_comparison_report(mock_results)

    assert "# Hindi/Hinglish Adaptation & Cross-Lingual Evaluation Report" in report
    for cfg in VARIANTS_CONFIG:
        assert cfg["name"] in report
    assert "91.51% / 7.67%" in report
    assert "96.00% / 2.00%" in report
    assert "Primary Comparison: Zero-Shot vs. Hindi-Combined" in report
    assert "Ablation Analysis: Combined vs. Hindi-Only" in report
    assert "Honest Methodology & Evaluation Caveats" in report
