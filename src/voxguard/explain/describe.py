"""
describe.py — rule-based descriptive explanation of windowed attribution (Phase 9).

Generates grounded, descriptive explanations from windowed attribution scores
and timestamps without fabricating acoustic or causal explanations the model
does not provide.
"""

from __future__ import annotations

import numpy as np

from voxguard.utils.logging_utils import get_logger

logger = get_logger(__name__)


def describe_attribution(
    scores: np.ndarray,
    timestamps: np.ndarray,
    label: str,
) -> str:
    """Generates a rule-based descriptive explanation from windowed attribution.

    Covers:
      1. Overall verdict framing with average synthetic-likelihood.
      2. Highest and lowest scoring window timestamps and scores.
      3. Consistency / variance assessment across the clip.
      4. Low-energy / silent region caveat if > 30% of windows were unscored.

    Parameters
    ----------
    scores:
        1-D array of per-window synthetic-likelihood scores in ``[0.0, 1.0]``,
        may contain ``np.nan`` for silent/unscored regions.
    timestamps:
        1-D array of window start times in seconds aligned with ``scores``.
    label:
        Classification verdict label (e.g. ``"synthetic"`` or ``"real"``).

    Returns
    -------
    str
        2-4 grounded descriptive sentences.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    timestamps_arr = np.asarray(timestamps, dtype=np.float64)

    if len(scores_arr) == 0 or np.all(np.isnan(scores_arr)):
        return (
            f"This clip was classified as {label}, but attribution could not be computed "
            "because no regions contained sufficient speech energy to score reliably."
        )

    valid_mask = ~np.isnan(scores_arr)
    valid_scores = scores_arr[valid_mask]
    n_total = len(scores_arr)
    n_nan = int(np.isnan(scores_arr).sum())
    nan_ratio = n_nan / n_total if n_total > 0 else 0.0

    mean_score = float(np.nanmean(scores_arr))
    std_score = float(np.nanstd(scores_arr)) if len(valid_scores) > 1 else 0.0

    max_idx = int(np.nanargmax(scores_arr))
    min_idx = int(np.nanargmin(scores_arr))

    high_time = float(timestamps_arr[max_idx])
    high_score = float(scores_arr[max_idx])
    low_time = float(timestamps_arr[min_idx])
    low_score = float(scores_arr[min_idx])

    sentences: list[str] = []

    # 1. Overall verdict framing
    sentences.append(
        f"This clip was classified as {label} with an average synthetic-likelihood of "
        f"{mean_score:.0%} across the scored regions."
    )

    # 2. Strongest & weakest regional signals
    if len(valid_scores) == 1:
        sentences.append(
            f"The scored region is around {high_time:.1f}s (score {high_score:.0%})."
        )
    else:
        sentences.append(
            f"The strongest synthetic-sounding region is around {high_time:.1f}s (score {high_score:.0%}); "
            f"the most natural-sounding region is around {low_time:.1f}s (score {low_score:.0%})."
        )

    # 3. Consistency framing
    if len(valid_scores) > 1:
        if std_score < 0.15:
            sentences.append(
                "The synthetic-likelihood is fairly consistent throughout the clip."
            )
        else:
            sentences.append(
                "The synthetic-likelihood varies notably across the clip, which may indicate "
                "the audio quality or detectability differs by region."
            )

    # 4. Caveat for clips with >30% unscored windows
    if nan_ratio > 0.30:
        sentences.append(
            "A significant portion of this clip had too little energy to score reliably; "
            "the explanation above is based only on the scored regions."
        )

    return " ".join(sentences)
