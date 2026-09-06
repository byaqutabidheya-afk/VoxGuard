"""bands.py — map a synthetic-speech probability score to a named risk band.

The single public function, :func:`score_to_band`, converts a float in
``[0.0, 1.0]`` (the ``probability_synthetic`` output of the classifier) into
one of four string labels:

    "low"           — clearly real; confidence of synthesis is low.
    "medium"        — ambiguous; warrants user attention.
    "high"          — strong signal of synthetic speech.
    "inconclusive"  — input was ``None`` (e.g. silence / non-speech gate
                      fired in Phase 6); no probability was produced, so no
                      band can be assigned.

Boundary convention
-------------------
Threshold boundaries are **inclusive on the higher band** — a score exactly
equal to a threshold is promoted to the next, stricter band:

    score < low_max                           → "low"
    low_max <= score < medium_max             → "medium"
    score >= medium_max                       → "high"

In other words, a score *exactly equal* to ``low_max`` falls into "medium",
and a score *exactly equal* to ``medium_max`` falls into "high".
This was chosen because at a boundary the evidence is genuinely ambiguous —
it is safer to surface the stricter alert than to silently suppress it.

Recalibration note
------------------
The default thresholds (``low_max=0.3``, ``medium_max=0.7``) come from
:data:`voxguard.config.RISK_THRESHOLDS` and are **starting defaults only**.
They must be replaced in Prompt 7.5 after fitting the bucket boundaries to the
real validation-set score distribution.  Pass an explicit ``thresholds`` dict
to override them without touching ``config.py``.
"""

from __future__ import annotations

from typing import Optional

from voxguard import config

# The four valid return values — kept as module-level constants so that
# callers can compare against them without hard-coding bare string literals.
BAND_LOW: str = "low"
BAND_MEDIUM: str = "medium"
BAND_HIGH: str = "high"
BAND_INCONCLUSIVE: str = "inconclusive"


def score_to_band(
    probability_synthetic: Optional[float],
    thresholds: Optional[dict] = None,
) -> str:
    """Convert a synthetic-speech probability into a named risk band.

    Parameters
    ----------
    probability_synthetic:
        Output of the classifier — a float in ``[0.0, 1.0]`` representing the
        probability that the audio is synthetic.  Pass ``None`` when the
        upstream silence-detection gate (Phase 6) could not produce a score
        (near-silent or non-speech input).

    thresholds:
        Optional override dict with keys ``"low_max"`` and ``"medium_max"``.
        When omitted (or ``None``), :data:`voxguard.config.RISK_THRESHOLDS` is
        used.  Both values must be floats satisfying
        ``0.0 <= low_max <= medium_max <= 1.0``.

    Returns
    -------
    str
        One of ``"low"``, ``"medium"``, ``"high"``, or ``"inconclusive"``.

    Raises
    ------
    TypeError
        If ``probability_synthetic`` is not a ``float``/``int`` or ``None``.
    ValueError
        If the resolved thresholds dict is missing required keys, or if
        ``low_max > medium_max``.

    Boundary convention
    -------------------
    Exact boundary hits are assigned to the **higher** band:

    * ``score < low_max``               → ``"low"``
    * ``low_max <= score < medium_max`` → ``"medium"``
    * ``score >= medium_max``           → ``"high"``

    Both thresholds follow the same rule: a score *exactly equal* to a
    threshold is promoted to the stricter band above it.

    Examples
    --------
    >>> score_to_band(0.1)
    'low'
    >>> score_to_band(0.3)   # exactly at low_max → 'medium'
    'medium'
    >>> score_to_band(0.5)
    'medium'
    >>> score_to_band(0.7)   # exactly at medium_max → 'high'
    'high'
    >>> score_to_band(0.9)
    'high'
    >>> score_to_band(None)
    'inconclusive'
    """
    # ------------------------------------------------------------------ #
    # 1. None input — silence / non-speech gate fired; no score produced. #
    # ------------------------------------------------------------------ #
    if probability_synthetic is None:
        return BAND_INCONCLUSIVE

    # ------------------------------------------------------------------ #
    # 2. Type guard — reject anything that is not numeric.                #
    # ------------------------------------------------------------------ #
    if not isinstance(probability_synthetic, (int, float)):
        raise TypeError(
            f"probability_synthetic must be a float, int, or None; "
            f"got {type(probability_synthetic).__name__!r}"
        )

    # ------------------------------------------------------------------ #
    # 3. Resolve thresholds.                                              #
    # ------------------------------------------------------------------ #
    resolved: dict = thresholds if thresholds is not None else config.RISK_THRESHOLDS

    try:
        low_max: float = float(resolved["low_max"])
        medium_max: float = float(resolved["medium_max"])
    except KeyError as exc:
        raise ValueError(
            f"thresholds dict is missing required key {exc}; "
            f"expected keys: 'low_max', 'medium_max'"
        ) from exc

    if low_max > medium_max:
        raise ValueError(
            f"low_max ({low_max}) must be <= medium_max ({medium_max})"
        )

    # ------------------------------------------------------------------ #
    # 4. Bucketing — both boundaries promote exact hits to the higher    #
    #    band (see module docstring for the chosen convention).           #
    # ------------------------------------------------------------------ #
    score: float = float(probability_synthetic)

    if score < low_max:
        return BAND_LOW
    if score < medium_max:
        return BAND_MEDIUM
    return BAND_HIGH
