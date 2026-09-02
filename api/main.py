"""
main.py — FastAPI entrypoint for VoxGuard (Phase 11).

Exposes a REST API for platform integration.  Runs as a separate
process from the Gradio UI (app/app.py) but imports shared detection
logic from src/voxguard — no duplicated business logic.

TODO (Phase 11):
  - POST /detect    — submit audio bytes, receive detection result + score
  - POST /enroll    — enroll a speaker voiceprint (Phase 5)
  - POST /verify    — verify a speaker against an enrolled voiceprint
  - GET  /health    — liveness probe for container orchestration
  - wire up voxguard.privacy session logging on every request
"""

# Placeholder — no logic implemented yet.
