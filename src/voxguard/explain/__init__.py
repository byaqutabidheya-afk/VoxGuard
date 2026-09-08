# explain — explainability overlay for detection decisions (Phase 9).

from voxguard.explain.attribution import windowed_attribution
from voxguard.explain.describe import describe_attribution
from voxguard.explain.overlay import render_explainability_overlay
from voxguard.explain.spectrogram import (
    generate_mel_spectrogram,
    render_spectrogram_image,
)

__all__ = [
    "windowed_attribution",
    "describe_attribution",
    "render_explainability_overlay",
    "generate_mel_spectrogram",
    "render_spectrogram_image",
]

