"""Tests for the generalization report markdown output."""

from __future__ import annotations

from pathlib import Path

from voxguard.reports import generalization_report as report_module


def test_build_generalization_report_writes_slide_ready_markdown(
    monkeypatch, tmp_path: Path
) -> None:
    metrics_map = {
        ("wav2vec2",): {"accuracy": 0.91, "eer": 0.11},
        ("wavlm",): {"accuracy": 0.92, "eer": 0.12},
        ("wav2vec2", "wavlm"): {"accuracy": 0.95, "eer": 0.09},
        ("weighted",): {"accuracy": 0.93, "eer": 0.10},
    }

    calls = []

    def fake_zero_shot_eval_from_cache(
        classifier_path, model_names, dataset, use_prosody=False, split="eval"
    ):
        calls.append(
            (
                Path(classifier_path).name,
                tuple(model_names),
                dataset,
                use_prosody,
                split,
            )
        )
        return metrics_map[tuple(model_names)]

    def fake_zero_shot_eval_weighted_average_from_cache(
        classifier_a_path,
        model_a,
        classifier_b_path,
        model_b,
        dataset,
        weight_a=0.5,
        use_prosody=False,
        split="eval",
    ):
        calls.append(
            (
                Path(classifier_a_path).name,
                model_a,
                Path(classifier_b_path).name,
                model_b,
                dataset,
                weight_a,
                use_prosody,
                split,
            )
        )
        return metrics_map[("weighted",)]

    monkeypatch.setattr(
        report_module,
        "zero_shot_eval_from_cache",
        fake_zero_shot_eval_from_cache,
    )
    monkeypatch.setattr(
        report_module,
        "zero_shot_eval_weighted_average_from_cache",
        fake_zero_shot_eval_weighted_average_from_cache,
    )

    output_path = report_module.build_generalization_report(
        tmp_path / "generalization_before_after.md"
    )

    content = output_path.read_text(encoding="utf-8")
    assert "# Generalization Before/After" in content
    assert "ASVspoof2019 eval" in content
    assert "WaveFake" in content
    assert "In-the-Wild" in content
    assert "wav2vec2-only" in content
    assert "WavLM-only" in content
    assert "concatenated ensemble" in content
    assert "weighted-average ensemble" in content
    assert "91.0% / 11.0%" in content
    assert "95.0% / 9.0%" in content
    assert len(calls) == 12
