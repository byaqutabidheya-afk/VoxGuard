"""
generalization_report.py — cross-dataset generalization evidence for VoxGuard.

Builds a compact markdown table summarizing accuracy and EER for the main
classifier variants on ASVspoof2019 eval, WaveFake, and In-the-Wild.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from voxguard import config
from voxguard.classifier.cross_eval import (
    zero_shot_eval_from_cache,
    zero_shot_eval_weighted_average_from_cache,
)
from voxguard.utils.logging_utils import get_logger

logger = get_logger(__name__)

REPORT_PATH = config.MODELS_DIR / "reports" / "generalization_before_after.md"

WAV2VEC2_CLASSIFIER = config.MODELS_DIR / "classifiers" / "baseline_logreg.joblib"
WAVLM_CLASSIFIER = config.MODELS_DIR / "classifiers" / "wavlm_logreg.joblib"
ENSEMBLE_CLASSIFIER = config.MODELS_DIR / "classifiers" / "ensemble_logreg.joblib"

DISPLAY_NAMES = {
    "asvspoof2019": "ASVspoof2019 eval",
    "wavefake": "WaveFake",
    "in_the_wild": "In-the-Wild",
}

MODEL_HEADERS = [
    "wav2vec2-only",
    "WavLM-only",
    "concatenated ensemble",
    "weighted-average ensemble",
]


def _format_cell(metrics: dict) -> str:
    """Formats a metric pair for slide-friendly markdown cells."""
    accuracy = metrics.get("accuracy")
    eer = metrics.get("eer")
    if accuracy is None or eer is None:
        return "n/a"
    if pd.isna(accuracy) or pd.isna(eer):
        return "n/a"
    return f"{accuracy * 100:.1f}% / {eer * 100:.1f}%"


def _build_markdown_table(rows: list[dict]) -> str:
    """Builds a compact markdown table with rows=datasets and columns=variants."""
    df = pd.DataFrame(rows)
    ordered_columns = ["dataset", *MODEL_HEADERS]
    df = df[ordered_columns]

    header = "| " + " | ".join(ordered_columns) + " |"
    separator = "| " + " | ".join(["---"] * len(ordered_columns)) + " |"
    body_rows = [
        "| " + " | ".join(str(row[col]) for col in ordered_columns) + " |"
        for _, row in df.iterrows()
    ]

    lines = [
        "# Generalization Before/After",
        "",
        "Accuracy / EER on each evaluation set. Higher accuracy is better; lower EER is better.",
        "",
        header,
        separator,
        *body_rows,
        "",
    ]
    return "\n".join(lines)


def build_generalization_report(output_path: Path | None = None) -> Path:
    """Builds and saves the generalization report markdown table."""
    output_path = output_path or REPORT_PATH

    eval_datasets = ["asvspoof2019", "wavefake", "in_the_wild"]

    report_rows: list[dict] = []
    for dataset in eval_datasets:
        row = {"dataset": DISPLAY_NAMES[dataset]}
        cache_metrics = {
            "wav2vec2-only": zero_shot_eval_from_cache(
                str(WAV2VEC2_CLASSIFIER), ["wav2vec2"], dataset, use_prosody=False
            ),
            "WavLM-only": zero_shot_eval_from_cache(
                str(WAVLM_CLASSIFIER), ["wavlm"], dataset, use_prosody=False
            ),
            "concatenated ensemble": zero_shot_eval_from_cache(
                str(ENSEMBLE_CLASSIFIER),
                ["wav2vec2", "wavlm"],
                dataset,
                use_prosody=False,
            ),
            "weighted-average ensemble": zero_shot_eval_weighted_average_from_cache(
                str(WAV2VEC2_CLASSIFIER),
                "wav2vec2",
                str(WAVLM_CLASSIFIER),
                "wavlm",
                dataset,
                weight_a=0.5,
                use_prosody=False,
            ),
        }

        for column_name, metrics in cache_metrics.items():
            row[column_name] = _format_cell(metrics)
            logger.info(
                "generalization report: dataset=%s model=%s accuracy=%.4f eer=%.4f",
                dataset,
                column_name,
                metrics["accuracy"],
                metrics["eer"],
            )
        report_rows.append(row)

    markdown = _build_markdown_table(report_rows)
    markdown += (
        "\nNote: the weighted-average column is now backed by a dedicated "
        "WeightedAverageDetector, so Phase 5 can wrap it directly if it wins. "
        "If the table is close, the concatenated ensemble remains the simpler "
        "production default because it already uses a single detector class.\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    logger.info("Saved generalization report to %s", output_path)
    return output_path
