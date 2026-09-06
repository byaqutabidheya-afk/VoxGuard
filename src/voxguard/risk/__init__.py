"""voxguard.risk — risk-level utilities for synthetic-speech detection."""

from voxguard.risk.bands import score_to_band
from voxguard.risk.prevention import get_prevention_message

__all__ = ["score_to_band", "get_prevention_message"]
