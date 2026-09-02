"""
app.py — Gradio entrypoint for VoxGuard.

Launches an interactive web UI for submitting audio files or live
microphone input and receiving real/synthetic verdicts with explanations.

This process imports detection logic from src/voxguard and is kept
separate from the FastAPI REST process (api/main.py) so the two can
be deployed independently while sharing the same core library.

TODO (Phase 4 / UI phases):
  - build Gradio interface with file-upload and microphone components
  - wire up voxguard.classifier and voxguard.explain for live feedback
  - add prevention-layer panel with actionable guidance
"""

# Placeholder — no logic implemented yet.
