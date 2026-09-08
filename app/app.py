"""
app.py — Gradio entrypoint for VoxGuard.

Launches an interactive web UI for submitting audio files or live
microphone input and receiving real/synthetic verdicts with explanations.

This process imports detection logic from src/voxguard and is kept
separate from the FastAPI REST process (api/main.py) so the two can
be deployed independently while sharing the same core library.
"""

from __future__ import annotations

import html
import logging
import re
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np

from voxguard.classifier.ensemble import WeightedAverageDetector
from voxguard.explain import (
    describe_attribution,
    render_explainability_overlay,
    windowed_attribution,
)
from voxguard.fusion.fuse import fuse_risk_with_context
from voxguard.fusion.redflags import scan_for_redflags
from voxguard.fusion.transcribe import LiveTranscriber
from voxguard.privacy.session_log import SessionLogger
from voxguard.risk.bands import score_to_band
from voxguard.risk.prevention import get_prevention_message
from voxguard.speaker.embedding import SpeakerEmbedder
from voxguard.speaker.enrollment import (
    delete_speaker,
    enroll_speaker,
    list_enrolled_speakers,
)
from voxguard.speaker.verify import verify_speaker
from voxguard.streaming.session import StreamingSession, simulate_stream
from voxguard.utils.audio_io import load_audio

logger = logging.getLogger(__name__)

# =============================================================================
# "Midnight control room" visual theme — presentation only.
# =============================================================================
# Everything in this block is CSS/markup for the existing tabs and controls.
# No feature, callback, tab, or output is added, removed, or renamed here —
# see build_app() below, which wires the exact same functions to the exact
# same inputs/outputs as before, just inside restyled containers.
_CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons+Outlined|Material+Icons|Material+Symbols+Outlined');

:root {
    --vg-bg-0: #05070c;
    --vg-bg-1: #0a1120;
    --vg-bg-2: #0d1526;
    --vg-surface: rgba(17, 25, 40, 0.62);
    --vg-surface-strong: rgba(15, 22, 36, 0.85);
    --vg-border: rgba(148, 163, 184, 0.16);
    --vg-border-strong: rgba(148, 163, 184, 0.28);
    --vg-cyan: #22d3ee;
    --vg-cyan-soft: rgba(34, 211, 238, 0.35);
    --vg-amber: #f59e0b;
    --vg-green: #22c55e;
    --vg-red: #ef4444;
    --vg-text: #e6edf5;
    --vg-text-dim: #94a3b8;
    --vg-mono: 'Consolas', 'SFMono-Regular', ui-monospace, 'Cascadia Code', monospace;
    --vg-sans: 'Segoe UI', system-ui, -apple-system, sans-serif;
}

/* ---------------------------------------------------------------------- */
/* Material Icons / Symbols base styling                                 */
/* ---------------------------------------------------------------------- */
.material-symbols-outlined,
.material-icons-outlined,
.material-icons {
    font-family: 'Material Symbols Outlined', 'Material Icons Outlined', 'Material Icons', sans-serif !important;
    font-weight: normal;
    font-style: normal;
    font-size: 1.15em;
    line-height: 1;
    letter-spacing: normal;
    text-transform: none;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    white-space: nowrap;
    word-wrap: normal;
    direction: ltr;
    vertical-align: -2px;
    font-feature-settings: 'liga';
    -webkit-font-smoothing: antialiased;
}

.vg-sec-icon {
    font-size: 1.25em !important;
    vertical-align: -3px !important;
    margin-right: 7px !important;
    color: var(--vg-cyan) !important;
    filter: drop-shadow(0 0 6px var(--vg-cyan-soft));
    display: inline-flex !important;
}

.vg-card-icon {
    font-size: 1.2em !important;
    vertical-align: -3px !important;
    margin-right: 8px !important;
    display: inline-flex !important;
}

.vg-badge-icon {
    font-size: 1.05em !important;
    vertical-align: -2px !important;
    margin-right: 4px !important;
    display: inline-flex !important;
}

.vg-hero-shield-icon {
    font-size: 0.92em !important;
    vertical-align: -4px !important;
    margin-right: 8px !important;
    color: var(--vg-cyan) !important;
    filter: drop-shadow(0 0 10px var(--vg-cyan-soft));
    display: inline-flex !important;
}

/* ---------------------------------------------------------------------- */
/* Base page: deep charcoal / blue-black gradient + faint grid + scanline */
/* ---------------------------------------------------------------------- */
.gradio-container {
    /* Gradio's internal components read colors from these theme custom
       properties (not just inherited `color`) — overriding them here is
       what actually re-themes markdown text, labels, inputs, tab nav, etc.
       throughout the app, not just the elements we touch directly. */
    --body-text-color: var(--vg-text);
    --body-text-color-subdued: var(--vg-text-dim);
    --body-background-fill: transparent;
    --background-fill-primary: var(--vg-bg-1);
    --background-fill-secondary: var(--vg-bg-2);
    --border-color-primary: var(--vg-border);
    --border-color-accent: var(--vg-cyan);
    --block-background-fill: var(--vg-surface);
    --block-border-color: var(--vg-border);
    --block-label-text-color: var(--vg-text-dim);
    --block-label-background-fill: transparent;
    --block-title-text-color: var(--vg-text);
    --block-info-text-color: var(--vg-text-dim);
    --panel-background-fill: var(--vg-surface);
    --panel-border-color: var(--vg-border);
    --input-background-fill: rgba(10, 16, 28, 0.6);
    --input-border-color: var(--vg-border-strong);
    --input-placeholder-color: var(--vg-text-dim);
    --color-accent: var(--vg-cyan);
    --color-accent-soft: rgba(34, 211, 238, 0.12);
    --link-text-color: var(--vg-cyan);
    --checkbox-label-text-color: var(--vg-text);
    --neutral-100: #131c2b;
    --neutral-200: #1b2537;

    background:
        radial-gradient(circle at 12% -8%, rgba(34, 211, 238, 0.09), transparent 42%),
        radial-gradient(circle at 88% 2%, rgba(245, 158, 11, 0.07), transparent 38%),
        linear-gradient(180deg, var(--vg-bg-0) 0%, var(--vg-bg-1) 45%, var(--vg-bg-2) 100%) !important;
    background-attachment: fixed !important;
    color: var(--vg-text) !important;
    font-family: var(--vg-sans) !important;
    position: relative;
}

/* Markdown/plain text blocks: Gradio's markdown component sometimes sets
   its own literal text color rather than the variable above — force it
   explicitly so headings and body copy stay legible on the dark surface. */
.gradio-container .prose,
.gradio-container .prose * ,
.gradio-container label,
.gradio-container span {
    color: var(--vg-text);
}
.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3 {
    color: #f8fafc;
}

.gradio-container::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
    background-image:
        linear-gradient(rgba(148, 163, 184, 0.055) 1px, transparent 1px),
        linear-gradient(90deg, rgba(148, 163, 184, 0.055) 1px, transparent 1px);
    background-size: 44px 44px;
}

.gradio-container::after {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    opacity: 0.05;
    background: repeating-linear-gradient(
        0deg,
        rgba(34, 211, 238, 0.7) 0px,
        rgba(34, 211, 238, 0.7) 1px,
        transparent 1px,
        transparent 3px
    );
    animation: vg-scan-drift 10s linear infinite;
}

@keyframes vg-scan-drift {
    0%   { transform: translateY(0); }
    100% { transform: translateY(44px); }
}

@media (prefers-reduced-motion: reduce) {
    .gradio-container::after { animation: none; }
}

/* Everything Gradio renders should sit above the decorative layers. */
.gradio-container > * { position: relative; z-index: 1; }

/* ---------------------------------------------------------------------- */
/* Hero header                                                            */
/* ---------------------------------------------------------------------- */
.vg-hero {
    padding: 28px 32px 24px 32px;
    margin-bottom: 6px;
    border-radius: 16px;
    border: 1px solid var(--vg-border-strong);
    background:
        linear-gradient(135deg, rgba(34, 211, 238, 0.07), rgba(10, 14, 24, 0) 55%),
        var(--vg-surface-strong);
    box-shadow: 0 0 0 1px rgba(34, 211, 238, 0.04) inset,
                0 20px 60px -25px rgba(0, 0, 0, 0.65);
    animation: vg-fade-slide-in 0.7s ease-out both;
}

.vg-hero-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
}

.vg-hero-title {
    font-size: 1.9em;
    font-weight: 800;
    letter-spacing: 0.01em;
    margin: 0;
    color: #f8fafc;
}

.vg-hero-title .vg-hero-mark {
    color: var(--vg-cyan);
    text-shadow: 0 0 18px var(--vg-cyan-soft);
}

.vg-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    border-radius: 999px;
    border: 1px solid rgba(34, 197, 94, 0.35);
    background: rgba(34, 197, 94, 0.08);
    font-family: var(--vg-mono);
    font-size: 0.78em;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #86efac;
    white-space: nowrap;
}

.vg-status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--vg-green);
    box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.6);
    animation: vg-pulse-dot 2.2s ease-out infinite;
}

@keyframes vg-pulse-dot {
    0%   { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.55); }
    70%  { box-shadow: 0 0 0 8px rgba(34, 197, 94, 0); }
    100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
}

.vg-hero-mission {
    margin: 12px 0 0 0;
    color: var(--vg-text-dim);
    font-size: 0.95em;
    line-height: 1.55;
    max-width: 900px;
}

.vg-hero-mission strong { color: var(--vg-text); }

/* ---------------------------------------------------------------------- */
/* Section headings inside tabs                                          */
/* ---------------------------------------------------------------------- */
.vg-section-title {
    font-family: var(--vg-mono);
    font-size: 0.82em;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--vg-cyan);
    margin: 4px 0 12px 0;
    padding-bottom: 7px;
    border-bottom: 1px solid var(--vg-border);
    opacity: 0.95;
    display: flex;
    align-items: center;
}

/* ---------------------------------------------------------------------- */
/* Staggered card reveal                                                  */
/* ---------------------------------------------------------------------- */
@keyframes vg-fade-slide-in {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

.vg-reveal {
    animation: vg-fade-slide-in 0.55s ease-out both;
}
.vg-reveal-1 { animation-delay: 0.05s; }
.vg-reveal-2 { animation-delay: 0.12s; }
.vg-reveal-3 { animation-delay: 0.19s; }
.vg-reveal-4 { animation-delay: 0.26s; }

@media (prefers-reduced-motion: reduce) {
    .vg-reveal, .vg-hero, .vg-card { animation: none !important; }
}

/* ---------------------------------------------------------------------- */
/* Glass panel wrapper for control/result columns                        */
/* ---------------------------------------------------------------------- */
.vg-panel {
    border-radius: 14px !important;
    border: 1px solid var(--vg-border) !important;
    background: var(--vg-surface) !important;
    padding: 18px !important;
    box-shadow: 0 12px 34px -22px rgba(0, 0, 0, 0.7);
    backdrop-filter: blur(6px);
}

/* ---------------------------------------------------------------------- */
/* Risk / status HTML cards produced by _risk_html, _voiceprint_result_html */
/* ---------------------------------------------------------------------- */
.vg-card {
    animation: vg-fade-slide-in 0.45s ease-out both;
    backdrop-filter: blur(8px);
}

.vg-card-live {
    animation: vg-fade-slide-in 0.45s ease-out both, vg-live-pulse 2.6s ease-in-out infinite 0.5s;
}

@keyframes vg-live-pulse {
    0%, 100% { box-shadow: 0 0 0 0 var(--vg-glow-color, transparent), 0 10px 30px -18px rgba(0,0,0,0.6); }
    50%      { box-shadow: 0 0 0 6px var(--vg-glow-color, transparent), 0 10px 30px -18px rgba(0,0,0,0.6); }
}

/* Prevention alert card entrance — gentle drop-and-settle, not a jump-scare */
.vg-alert-card {
    animation: vg-alert-in 0.5s cubic-bezier(0.22, 1, 0.36, 1) both;
}

@keyframes vg-alert-in {
    from { opacity: 0; transform: translateY(-6px) scale(0.985); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

/* ---------------------------------------------------------------------- */
/* Tabs                                                                   */
/* ---------------------------------------------------------------------- */
.tabs { border: none !important; background: transparent !important; }
.tab-nav {
    border-bottom: 1px solid var(--vg-border) !important;
    gap: 4px;
}
.tab-nav button {
    font-family: var(--vg-mono) !important;
    font-size: 0.82em !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: var(--vg-text-dim) !important;
    border-radius: 10px 10px 0 0 !important;
}
.tab-nav button.selected {
    color: var(--vg-cyan) !important;
    background: rgba(34, 211, 238, 0.07) !important;
    box-shadow: inset 0 -2px 0 var(--vg-cyan);
}

/* ---------------------------------------------------------------------- */
/* Buttons                                                                */
/* ---------------------------------------------------------------------- */
button.primary {
    background: linear-gradient(135deg, #0ea5b7, #22d3ee) !important;
    border: none !important;
    color: #04141a !important;
    font-weight: 700 !important;
    box-shadow: 0 6px 22px -8px var(--vg-cyan-soft);
}
button.secondary {
    background: rgba(148, 163, 184, 0.1) !important;
    border: 1px solid var(--vg-border-strong) !important;
    color: var(--vg-text) !important;
}
button.stop {
    background: linear-gradient(135deg, #b91c1c, #ef4444) !important;
    border: none !important;
    color: #fff5f5 !important;
}

/* ---------------------------------------------------------------------- */
/* Splash / intro screen — purely decorative, never blocks interaction    */
/* ---------------------------------------------------------------------- */
.vg-splash {
    position: fixed;
    inset: 0;
    z-index: 9999;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 14px;
    background: radial-gradient(circle at 50% 40%, #0d1a2b 0%, #05070c 70%);
    pointer-events: none; /* always click-through, even mid-animation */
    animation: vg-splash-life 2.4s ease-in-out forwards;
}
.vg-splash-mark {
    font-family: var(--vg-mono);
    font-size: 2.6em;
    font-weight: 800;
    letter-spacing: 0.06em;
    color: #f8fafc;
    text-shadow: 0 0 26px var(--vg-cyan-soft);
}
.vg-splash-mark span { color: var(--vg-cyan); }
.vg-splash-sub {
    font-family: var(--vg-mono);
    font-size: 0.82em;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--vg-text-dim);
    animation: vg-splash-blink 1.4s ease-in-out infinite;
}
.vg-splash-bar {
    width: 220px;
    height: 2px;
    background: rgba(148, 163, 184, 0.15);
    overflow: hidden;
    border-radius: 2px;
}
.vg-splash-bar::after {
    content: "";
    display: block;
    height: 100%;
    width: 40%;
    background: linear-gradient(90deg, transparent, var(--vg-cyan), transparent);
    animation: vg-splash-sweep 1.1s ease-in-out infinite;
}
@keyframes vg-splash-sweep {
    0%   { transform: translateX(-120%); }
    100% { transform: translateX(360%); }
}
@keyframes vg-splash-blink {
    0%, 100% { opacity: 0.5; }
    50%      { opacity: 1; }
}
@keyframes vg-splash-life {
    0%   { opacity: 0; }
    10%  { opacity: 1; }
    78%  { opacity: 1; }
    100% { opacity: 0; visibility: hidden; }
}
@media (prefers-reduced-motion: reduce) {
    .vg-splash { display: none; }
}
"""

_HEAD_HTML = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" />
<link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Icons|Material+Icons+Outlined" />
"""


# ---------------------------------------------------------------------------
# Module-level globals — audited (Phase 6, Prompt 6.2): these hold only
# process-wide, read-only/shared resources, never per-session mutable state.
#   - _DETECTOR / _SPEAKER_EMBEDDER / _TRANSCRIBER: lazily-built model
#     instances. Expensive to load and stateless once built (inference does
#     not mutate them), so sharing one instance across every session is
#     correct and is what Gradio's own docs recommend — the alternative
#     (reloading the ensemble/whisper model per gr.State) would make every
#     session pay multi-second load latency for no isolation benefit.
#   - _SESSION_LOGGER: an append-only audit log for the whole process, not
#     per-user data.
# Every value that actually varies per user/session — the streaming
# session, live transcript, last-analyzed clip, enrollment clip list, and
# the shared voiceprint verification result — is threaded through as
# gr.State below, scoped inside build_app(). There is no global mutable
# session state in this file.
_DETECTOR: WeightedAverageDetector | None = None
_SPEAKER_EMBEDDER: SpeakerEmbedder | None = None
_TRANSCRIBER: LiveTranscriber | None = None
_SESSION_LOGGER = SessionLogger()
_SESSION_LOGGER.purge_older_than(30)

TRANSACTION_CHOICES: list[tuple[str, str]] = [
    ("General conversation (1.0x)", "general_conversation"),
    ("OTP request (1.3x)", "otp_request"),
    ("Fund transfer (1.5x)", "fund_transfer"),
    ("Confidential info request (1.4x)", "confidential_info_request"),
]


# ---------------------------------------------------------------------------
# Risk-band visual rendering
# ---------------------------------------------------------------------------
#
# Contrast design principle
# ─────────────────────────
# Every element that carries an explicit `background` must also carry an
# explicit `color` referencing a shade that contrasts against *that
# background*, not against the page. Gradio's dark-mode theme sets a
# near-white inherited foreground on all elements; once we paint our own
# background we can no longer rely on inheritance — we own the foreground.
#
# "Midnight control room" theme: dark glass surfaces with a bright signal
# color per band, rather than the earlier light pastel cards. All text
# colors below are bright tints chosen for ≥ 7:1 contrast against their
# paired near-black translucent background (checked against the *opaque*
# worst case, i.e. background composited over the app's own near-black
# page — the actual glass card is always at least as dark as that).
#
# "muted" = the color used for secondary / small-print text inside a card.
# It is always explicitly set — never left as `color:inherit`.
# "glow" = the box-shadow color used for the live-pulse animation.

_BAND_STYLES: dict[str, dict[str, str]] = {
    "low": {
        "bg": "rgba(16, 44, 30, 0.65)",     # dark green glass
        "border": "#22c55e",                # signal green
        "text": "#a7f3d0",                  # bright mint  — ~12:1 on near-black
        "muted": "#6ee7b7",                 # softer green —  ~9:1 on near-black
        "glow": "rgba(34, 197, 94, 0.45)",
        "label": "LOW RISK",
        "icon": "verified_user",
        "prev_bg": "rgba(16, 44, 30, 0.65)",
        "prev_text": "#a7f3d0",
    },
    "medium": {
        "bg": "rgba(56, 40, 6, 0.68)",      # dark amber glass
        "border": "#f59e0b",                # amber
        "text": "#fde68a",                  # bright amber — ~13:1 on near-black
        "muted": "#fbbf24",                 # amber        — ~10:1 on near-black
        "glow": "rgba(245, 158, 11, 0.45)",
        "label": "MEDIUM RISK",
        "icon": "warning",
        "prev_bg": "rgba(56, 40, 6, 0.72)",
        "prev_text": "#fde68a",
    },
    "high": {
        "bg": "rgba(56, 12, 14, 0.7)",      # dark red glass
        "border": "#ef4444",                # warning red
        "text": "#fecaca",                  # bright red   — ~12:1 on near-black
        "muted": "#fca5a5",                 # red          —  ~9:1 on near-black
        "glow": "rgba(239, 68, 68, 0.5)",
        "label": "HIGH RISK",
        "icon": "gpp_bad",
        "prev_bg": "rgba(56, 12, 14, 0.78)",
        "prev_text": "#fecaca",
    },
    "inconclusive": {
        "bg": "rgba(22, 30, 44, 0.65)",     # dark slate glass
        "border": "#64748b",                # slate
        "text": "#cbd5e1",                  # bright slate — ~11:1 on near-black
        "muted": "#94a3b8",                 #              —  ~7:1 on near-black
        "glow": "rgba(100, 116, 139, 0.4)",
        "label": "INCONCLUSIVE",
        "icon": "help_center",
        "prev_bg": "rgba(22, 30, 44, 0.65)",
        "prev_text": "#cbd5e1",
    },
}


def _risk_html(
    probability_synthetic: float | None,
    context: str = "",
    base_fused_score: float | None = None,
    audio_score: float | None = None,
    keyword_score: float | None = None,
    transaction_multiplier: float | None = None,
    contact_multiplier: float | None = None,
) -> str:
    """Return an HTML block showing the color-coded overall contextual call risk band.

    NOTE on Phase 7 vs. Phase 9 Semantics:
    ───────────────────────────────────────
    In Phase 7, this meter measured raw "audio cloning risk". In Phase 9, it
    represents "overall contextual call risk" — the fusion of acoustic cloning
    detection (70%), semantic red-flag phrase scanning (30%), transaction stakes
    multipliers (e.g. fund transfer 1.5x), and contact voiceprint verification
    multipliers (match 0.9x / mismatch 1.3x).

    Parameters
    ----------
    probability_synthetic:
        Contextual call risk score in [0, 1], or None (silence / non-speech gate).
    context:
        Short descriptive label shown above the band badge, e.g.
        "Whole-Clip" or "Streaming Simulation".
    base_fused_score:
        Audio + Keyword weighted sum before context multipliers.
    audio_score:
        Raw acoustic synthetic score.
    keyword_score:
        Raw transcript red-flag risk score.
    transaction_multiplier:
        Multiplier from transaction context (e.g. 1.5 for fund transfer).
    contact_multiplier:
        Multiplier from voiceprint verification (0.9 match, 1.3 mismatch).
    """
    band = score_to_band(probability_synthetic)
    s = _BAND_STYLES[band]

    prob_text = (
        f"Contextual Call Risk: {probability_synthetic:.4f}"
        if probability_synthetic is not None
        else "Contextual Call Risk: N/A (no speech detected)"
    )
    prob_line = (
        f'<p style="margin:6px 0 0 0; font-size:0.92em; font-weight:600; '
        f"color:{s['text']}; background:transparent; font-family:var(--vg-mono); "
        f'display:flex; align-items:center; gap:6px;\">'
        f'<span class="material-symbols-outlined" style="font-size:15px; opacity:0.85;">speed</span>'
        f"{prob_text}</p>"
    )

    breakdown_lines: list[str] = []
    if base_fused_score is not None:
        breakdown_lines.append(
            f'<span class="material-symbols-outlined" style="font-size:13px; vertical-align:-2px; margin-right:4px;">tune</span>'
            f"Base fused score (70% audio + 30% text): <b>{base_fused_score:.4f}</b>"
        )
    if audio_score is not None and keyword_score is not None:
        breakdown_lines.append(
            f'<span class="material-symbols-outlined" style="font-size:13px; vertical-align:-2px; margin-right:4px;">graphic_eq</span>'
            f"Signals: Audio score = {audio_score:.4f} · Red-flag score = {keyword_score:.4f}"
        )
    if transaction_multiplier is not None and contact_multiplier is not None:
        breakdown_lines.append(
            f'<span class="material-symbols-outlined" style="font-size:13px; vertical-align:-2px; margin-right:4px;">calculate</span>'
            f"Multipliers: Transaction ×{transaction_multiplier:.2f} · Contact ×{contact_multiplier:.2f}"
        )

    breakdown_html = ""
    if breakdown_lines:
        items = "<br>".join(breakdown_lines)
        breakdown_html = (
            f'<div style="margin-top:8px; padding-top:6px; border-top:1px dashed {s["border"]}; '
            f'font-size:0.80em; color:{s["muted"]}; line-height:1.5;">'
            f"{items}"
            f"</div>"
        )

    context_line = (
        f'<p style="margin:0 0 6px 0; font-size:0.76em; font-weight:700; '
        f"color:{s['muted']}; background:transparent; "
        f'font-family:var(--vg-mono); '
        f'text-transform:uppercase; letter-spacing:0.08em; '
        f'display:flex; align-items:center; gap:5px;">'
        f'<span class="material-symbols-outlined" style="font-size:14px;">analytics</span>'
        f"{context}</p>"
        if context
        else ""
    )

    # A "live" readout (streaming, still updating) gets a subtle continuous
    # pulse so it visually reads as an active signal, not a static result —
    # purely a CSS animation class, the score/label content is unchanged.
    is_live = "live" in context.lower() or "streaming" in context.lower()
    card_class = "vg-card vg-card-live" if is_live else "vg-card"

    return (
        f'<div class="{card_class}" style="'
        f"--vg-glow-color:{s['glow']}; "
        f"background:{s['bg']}; "
        f"color:{s['text']}; "
        f"border:1px solid {s['border']}; "
        f"border-radius:12px; "
        f"padding:14px 18px; "
        f"box-shadow:0 10px 30px -18px rgba(0,0,0,0.6); "
        f'margin:4px 0;">'
        f"{context_line}"
        f'<p style="margin:0; font-size:1.35em; font-weight:800; '
        f"color:{s['text']}; background:transparent; letter-spacing:0.02em; "
        f'display:flex; align-items:center; gap:8px;\">'
        f'<span class="material-symbols-outlined vg-card-icon" style="color:{s["border"]};">{s.get("icon", "shield")}</span>'
        f"{s['label']}</p>"
        f"{prob_line}"
        f"{breakdown_html}"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Transcript & Red-flag visual rendering
# ---------------------------------------------------------------------------


def _render_transcript_html(
    text: str,
    matched_phrases: list[str],
    categories: list[str],
) -> str:
    """Renders the transcript with matched red-flag phrases highlighted in <mark> tags."""
    if not text or not text.strip():
        return (
            '<div class="vg-card" style="background:rgba(17,25,40,0.55); color:#94a3b8; '
            'border:1px solid var(--vg-border, rgba(148,163,184,0.16)); '
            'border-radius:10px; padding:14px; font-style:italic; display:flex; align-items:center; gap:8px;">'
            '<span class="material-symbols-outlined" style="font-size:18px; color:#64748b;">mic_off</span>'
            "No speech transcribed yet."
            "</div>"
        )

    clean_text = text.strip()

    # Highlight matched phrases using a single union regex to avoid nested <mark> corruption
    if matched_phrases:
        sorted_phrases = sorted(matched_phrases, key=len, reverse=True)
        union_escaped = "|".join(re.escape(p) for p in sorted_phrases)
        pattern = rf"(?i)\b(?:{union_escaped})\b"

        def _replace_match(m: re.Match) -> str:
            matched_str = html.escape(m.group(0))
            return (
                f'<mark style="background:#f59e0b; color:#1c1200; padding:2px 5px; '
                f'border-radius:4px; font-weight:700; box-shadow:0 0 10px rgba(245,158,11,0.45);">'
                f'{matched_str}</mark>'
            )

        highlighted_body = re.sub(pattern, _replace_match, clean_text)
    else:
        highlighted_body = html.escape(clean_text)

    badge_html = ""
    if categories:
        badges = " ".join(
            f'<span style="background:rgba(239,68,68,0.14); color:#fca5a5; '
            f'border:1px solid rgba(239,68,68,0.4); '
            f'padding:2px 8px; border-radius:999px; font-size:0.78em; font-weight:700; '
            f'font-family:var(--vg-mono); display:inline-flex; align-items:center; gap:4px; '
            f'text-transform:uppercase; letter-spacing:0.04em;">'
            f'<span class="material-symbols-outlined" style="font-size:12px;">warning</span>'
            f'{cat.replace("_", " ")}</span>'
            for cat in categories
        )
        badge_html = (
            f'<div style="margin-top:10px; display:flex; gap:6px; flex-wrap:wrap; align-items:center;">'
            f'<strong style="font-size:0.8em; color:#94a3b8; '
            f'text-transform:uppercase; letter-spacing:0.05em; font-family:var(--vg-mono); '
            f'display:flex; align-items:center; gap:4px;">'
            f'<span class="material-symbols-outlined" style="font-size:14px; color:#ef4444;">flag</span>'
            f'Red-flag categories:</strong> {badges}'
            f"</div>"
        )

    return (
        f'<div class="vg-card" style="background:rgba(17,25,40,0.6); color:#e6edf5; '
        f'border:1px solid var(--vg-border, rgba(148,163,184,0.16)); '
        f'border-radius:10px; padding:14px; font-size:0.92em; line-height:1.6;">'
        f'<div style="margin-bottom:8px; font-weight:700; color:#22d3ee; '
        f'font-size:0.78em; text-transform:uppercase; letter-spacing:0.08em; '
        f'font-family:var(--vg-mono); display:flex; align-items:center; gap:6px;">'
        f'<span class="material-symbols-outlined" style="font-size:16px;">transcribe</span>'
        f'Transcript:</div>'
        f'<div style="color:#e6edf5;">{highlighted_body}</div>'
        f"{badge_html}"
        f"</div>"
    )


def _render_attribution_explanation_html(text: str) -> str:
    """Renders the descriptive attribution text in a styled explanation card."""
    if not text or not text.strip():
        return ""
    escaped = html.escape(text.strip())
    return (
        f'<div class="vg-card" style="background:rgba(15,30,48,0.65); color:#bae6fd; '
        f'border:1px solid rgba(34,211,238,0.28); '
        f'border-left:3px solid #22d3ee; border-radius:10px; padding:14px 18px; '
        f'margin:8px 0; font-size:0.92em; line-height:1.6; '
        f'box-shadow:0 10px 30px -20px rgba(34,211,238,0.35);">'
        f'<div style="margin-bottom:8px; font-weight:700; color:#67e8f9; font-size:0.8em; '
        f'text-transform:uppercase; letter-spacing:0.08em; font-family:var(--vg-mono); '
        f'display:flex; align-items:center; gap:6px;">'
        f'<span class="material-symbols-outlined" style="font-size:16px; color:#22d3ee;">insights</span>'
        f'Attribution Analysis</div>'
        f'<div style="color:#e0f2fe;">{escaped}</div>'
        f'</div>'
    )



# ---------------------------------------------------------------------------
# Prevention prompt rendering
# ---------------------------------------------------------------------------


def _prevention_html(band: str) -> str:
    """Return an HTML prevention-prompt block for medium/high bands.

    Returns an empty string for low and inconclusive (no alert fatigue).
    """
    if band not in ("medium", "high"):
        return ""

    s = _BAND_STYLES[band]
    accent = s["border"]
    bg = s["prev_bg"]
    fg = s["prev_text"]
    md_text = get_prevention_message(band)

    html_text = re.sub(
        r"\*\*(.*?)\*\*",
        lambda m: (
            f'<strong style="color:{fg}; background:transparent;">'
            f"{m.group(1)}</strong>"
        ),
        md_text,
    )
    lines = html_text.splitlines()
    html_lines: list[str] = []
    in_list = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- "):
            if not in_list:
                html_lines.append(
                    f"<ul style='margin:6px 0 0 0; padding-left:20px; "
                    f"color:{fg}; background:transparent;'>"
                )
                in_list = True
            html_lines.append(
                f"<li style='margin:3px 0; color:{fg}; "
                f"background:transparent;'>{stripped[2:]}</li>"
            )
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            if stripped:
                html_lines.append(
                    f"<p style='margin:0 0 6px 0; color:{fg}; "
                    f"background:transparent;'>{stripped}</p>"
                )
    if in_list:
        html_lines.append("</ul>")
    body = "\n".join(html_lines)

    header_label = "PREVENTION GUIDANCE — ACT NOW" if band == "high" else "PREVENTION GUIDANCE"
    icon_name = "crisis_alert" if band == "high" else "notification_important"
    header_html = (
        f'<div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">'
        f'<span class="material-symbols-outlined" style="font-size:18px; color:{accent};">{icon_name}</span>'
        f'<span style="font-family:var(--vg-mono); font-size:0.76em; font-weight:800; '
        f'letter-spacing:0.09em; text-transform:uppercase; color:{fg};">{header_label}</span>'
        f"</div>"
    )

    return (
        f'<div class="vg-card vg-alert-card" style="'
        f"background:{bg}; "
        f"color:{fg}; "
        f"border:1px solid {accent}; "
        f"border-left:4px solid {accent}; "
        f"border-radius:8px; "
        f"box-shadow:0 14px 34px -20px {accent}66, 0 0 0 1px rgba(255,255,255,0.02) inset; "
        f"padding:14px 18px; "
        f'margin:10px 0; font-size:0.92em; line-height:1.55;">'
        f"{header_html}"
        f"{body}"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Voiceprint verification result rendering
# ---------------------------------------------------------------------------


def _voiceprint_result_html(result: dict[str, Any]) -> str:
    """Renders a verify_speaker() result as a MATCH/MISMATCH styled card."""
    match = bool(result["match"])
    similarity = float(result["similarity"])
    enrolled_name = str(result["enrolled_name"])

    s = _BAND_STYLES["low" if match else "high"]
    label = "MATCH" if match else "MISMATCH"
    icon = "verified_user" if match else "no_accounts"

    return (
        f'<div class="vg-card" style="'
        f"--vg-glow-color:{s['glow']}; "
        f"background:{s['bg']}; "
        f"color:{s['text']}; "
        f"border:1px solid {s['border']}; "
        f"border-radius:12px; "
        f"box-shadow:0 10px 30px -18px rgba(0,0,0,0.6); "
        f"padding:14px 18px; "
        f'margin:4px 0;">'
        f'<p style="margin:0 0 6px 0; font-size:0.76em; font-weight:700; '
        f"color:{s['muted']}; background:transparent; font-family:var(--vg-mono); "
        f'text-transform:uppercase; letter-spacing:0.07em; display:flex; align-items:center; gap:5px;">'
        f'<span class="material-symbols-outlined" style="font-size:14px;">fingerprint</span>'
        f"Voiceprint check vs. &#39;{enrolled_name}&#39;</p>"
        f'<p style="margin:0; font-size:1.35em; font-weight:800; '
        f"color:{s['text']}; background:transparent; letter-spacing:0.02em; "
        f'display:flex; align-items:center; gap:8px;\">'
        f'<span class="material-symbols-outlined vg-card-icon" style="color:{s["border"]};">{icon}</span>'
        f"{label}</p>"
        f'<p style="margin:6px 0 0 0; font-size:0.82em; font-family:var(--vg-mono); '
        f"color:{s['muted']}; background:transparent; display:flex; align-items:center; gap:5px;\">"
        f'<span class="material-symbols-outlined" style="font-size:14px;">compare_arrows</span>'
        f"Cosine similarity: {similarity:.4f}</p>"
        f"</div>"
    )


def _voiceprint_placeholder_html(message: str) -> str:
    """Renders a neutral placeholder for the voiceprint result card."""
    return (
        f'<p style="color:#94a3b8; font-style:italic; font-size:0.92em; display:flex; align-items:center; gap:6px;">'
        f'<span class="material-symbols-outlined" style="font-size:16px; color:#64748b;">info</span>'
        f'{message}</p>'
    )


def _format_clip_list(clips: list[str]) -> str:
    """Renders the accumulated enrollment reference-clip list for display."""
    if not clips:
        return "No clips added yet."
    lines = [f"{i + 1}. {Path(c).name}" for i, c in enumerate(clips)]
    return f"{len(clips)} clip(s) added:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared detector / session helpers
# ---------------------------------------------------------------------------


def get_detector() -> WeightedAverageDetector:
    """Lazy-initializes and returns the shared WeightedAverageDetector instance."""
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = WeightedAverageDetector(
            wav2vec2_classifier_path="models/classifiers/wav2vec2_hindi_combined_logreg.joblib",
            wavlm_classifier_path="models/classifiers/wavlm_hindi_combined_logreg.joblib",
        )
    return _DETECTOR


def get_speaker_embedder() -> SpeakerEmbedder:
    """Lazy-initializes and returns the shared SpeakerEmbedder instance."""
    global _SPEAKER_EMBEDDER
    if _SPEAKER_EMBEDDER is None:
        _SPEAKER_EMBEDDER = SpeakerEmbedder()
    return _SPEAKER_EMBEDDER


def get_transcriber() -> LiveTranscriber:
    """Lazy-initializes and returns the shared LiveTranscriber instance."""
    global _TRANSCRIBER
    if _TRANSCRIBER is None:
        _TRANSCRIBER = LiveTranscriber(model_size="base")
    return _TRANSCRIBER


def create_session(sample_rate: int | None = None) -> StreamingSession:
    """Creates a new StreamingSession instance with the shared detector."""
    return StreamingSession(detector=get_detector(), sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# Live Mic callbacks
# ---------------------------------------------------------------------------


def process_audio_chunk(
    audio: tuple[int, np.ndarray] | None,
    session: StreamingSession | None,
    tx_context: str = "general_conversation",
    last_voiceprint_result: dict[str, Any] | None = None,
    transcript_state: str = "",
) -> tuple[StreamingSession, str, str, str, str, str, str]:
    """Processes an incoming streaming audio chunk and updates contextual risk state.

    Returns
    -------
    session, risk_html, prevention_html, transcript_html, flagged_str, seconds_to_flag_str, transcript_state
    """
    if audio is None:
        if session is None:
            session = create_session()
        current_score = session.risk_score.current()
        running_score = 0.0 if current_score is None else float(current_score)
        flagged = session._consecutive_flags >= session.consecutive_flags_required
        s2f_str = (
            f"{session._seconds_to_flag:.2f}s"
            if session._seconds_to_flag is not None
            else "N/A"
        )

        redflags = scan_for_redflags(transcript_state)
        kw_score = float(redflags["keyword_risk_score"])
        fusion_res = fuse_risk_with_context(
            audio_score=running_score,
            keyword_risk_score=kw_score,
            transaction_context=tx_context,
            voiceprint_result=last_voiceprint_result,
        )
        contextual_score = fusion_res["contextual_score"]
        band = score_to_band(contextual_score)

        return (
            session,
            _risk_html(
                contextual_score,
                context="Live Streaming",
                base_fused_score=fusion_res["base_fused_score"],
                audio_score=running_score,
                keyword_score=kw_score,
                transaction_multiplier=fusion_res["transaction_multiplier"],
                contact_multiplier=fusion_res["contact_multiplier"],
            ),
            _prevention_html(band),
            _render_transcript_html(
                transcript_state, redflags["matched_phrases"], redflags["categories"]
            ),
            str(flagged),
            s2f_str,
            transcript_state,
        )

    sr, waveform = audio
    if session is None:
        session = create_session(sample_rate=sr)

    y = np.asarray(waveform)
    if np.issubdtype(y.dtype, np.integer):
        y = y.astype(np.float32) / 32768.0
    else:
        y = y.astype(np.float32)

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    if y.size == 0:
        current_score = session.risk_score.current()
        running_score = 0.0 if current_score is None else float(current_score)
        flagged = session._consecutive_flags >= session.consecutive_flags_required
        s2f_str = (
            f"{session._seconds_to_flag:.2f}s"
            if session._seconds_to_flag is not None
            else "N/A"
        )
        redflags = scan_for_redflags(transcript_state)
        kw_score = float(redflags["keyword_risk_score"])
        fusion_res = fuse_risk_with_context(
            audio_score=running_score,
            keyword_risk_score=kw_score,
            transaction_context=tx_context,
            voiceprint_result=last_voiceprint_result,
        )
        contextual_score = fusion_res["contextual_score"]
        band = score_to_band(contextual_score)

        return (
            session,
            _risk_html(
                contextual_score,
                context="Live Streaming",
                base_fused_score=fusion_res["base_fused_score"],
                audio_score=running_score,
                keyword_score=kw_score,
                transaction_multiplier=fusion_res["transaction_multiplier"],
                contact_multiplier=fusion_res["contact_multiplier"],
            ),
            _prevention_html(band),
            _render_transcript_html(
                transcript_state, redflags["matched_phrases"], redflags["categories"]
            ),
            str(flagged),
            s2f_str,
            transcript_state,
        )

    # 1. Acoustic streaming score
    result = session.push_audio(y, sr=sr)
    running_score = float(result["running_score"])
    flagged = bool(result["flagged"])
    s2f = result["seconds_to_flag"]
    s2f_str = f"{s2f:.2f}s" if s2f is not None else "N/A"

    # 2. Live streaming chunk transcription
    try:
        transcriber = get_transcriber()
        chunk_text = transcriber.transcribe_chunk(y, sr)
        if chunk_text:
            if transcript_state:
                accumulated_transcript = f"{transcript_state} {chunk_text}".strip()
            else:
                accumulated_transcript = chunk_text.strip()
        else:
            accumulated_transcript = transcript_state
    except Exception as exc:
        logger.warning("Live transcription chunk error: %s", exc)
        accumulated_transcript = transcript_state

    # 3. Red-flag keyword scan
    redflags = scan_for_redflags(accumulated_transcript)
    kw_score = float(redflags["keyword_risk_score"])

    # 4. Contextual risk fusion
    fusion_res = fuse_risk_with_context(
        audio_score=running_score,
        keyword_risk_score=kw_score,
        transaction_context=tx_context,
        voiceprint_result=last_voiceprint_result,
    )
    contextual_score = fusion_res["contextual_score"]
    band = score_to_band(contextual_score)

    if flagged and not getattr(session, "_logged_flag_event", False):
        session._logged_flag_event = True
        _SESSION_LOGGER.log_event(
            event_type="flag_event",
            risk_band=band,
            probability_synthetic=contextual_score,
            flagged=flagged,
        )

    risk_html = _risk_html(
        contextual_score,
        context="Live Streaming",
        base_fused_score=fusion_res["base_fused_score"],
        audio_score=running_score,
        keyword_score=kw_score,
        transaction_multiplier=fusion_res["transaction_multiplier"],
        contact_multiplier=fusion_res["contact_multiplier"],
    )
    prevention_html = _prevention_html(band)
    transcript_html = _render_transcript_html(
        accumulated_transcript, redflags["matched_phrases"], redflags["categories"]
    )

    return (
        session,
        risk_html,
        prevention_html,
        transcript_html,
        str(flagged),
        s2f_str,
        accumulated_transcript,
    )


def reset_streaming_session(
    session: StreamingSession | None,
) -> tuple[StreamingSession, str, str, str, str, str, str]:
    """Resets the streaming session, clears transcript, and returns reset indicators."""
    if session is not None:
        current_score = session.risk_score.current()
        current_prob = 0.0 if current_score is None else float(current_score)
        _SESSION_LOGGER.log_event(
            event_type="session_reset",
            risk_band=score_to_band(current_prob),
            probability_synthetic=current_prob,
            flagged=session._consecutive_flags >= session.consecutive_flags_required,
        )
        session.reset()
    else:
        session = create_session()
    return (
        session,
        _risk_html(0.0),
        "",
        _render_transcript_html("", [], []),
        "False",
        "N/A",
        "",
    )


# ---------------------------------------------------------------------------
# Upload File callback
# ---------------------------------------------------------------------------


def analyze_uploaded_file(
    audio_path: str | None,
    tx_context: str = "general_conversation",
    last_voiceprint_result: dict[str, Any] | None = None,
) -> tuple[str, str, str, str, str, str | None]:
    """Run whole-clip, streaming-simulation, and transcript analysis on an uploaded audio file.

    Returns
    -------
    whole_risk_html, whole_prevention_html, stream_risk_html, stream_prevention_html,
    transcript_html, audio_path_passthrough
    """
    if audio_path is None:
        placeholder = (
            '<p style="color:#94a3b8; font-style:italic;">Upload a file and click Analyze.</p>'
        )
        return placeholder, "", placeholder, "", _render_transcript_html("", [], []), None

    waveform, sr = load_audio(audio_path, target_sr=16_000)
    duration = float(len(waveform)) / float(sr)

    # 1. Full-file transcription & red-flag scanning
    transcriber = get_transcriber()
    try:
        full_text = transcriber.transcribe_full(audio_path)
    except Exception as exc:
        logger.warning("transcribe_full failed on '%s': %s", audio_path, exc)
        full_text = ""

    redflags = scan_for_redflags(full_text)
    kw_score = float(redflags["keyword_risk_score"])
    transcript_html = _render_transcript_html(
        full_text, redflags["matched_phrases"], redflags["categories"]
    )

    # 2. Whole-clip analysis & fusion
    whole_clip = get_detector().predict_waveform(waveform, sr)
    audio_probability = whole_clip.get("probability_synthetic")
    raw_audio_score = 0.0 if audio_probability is None else float(audio_probability)

    whole_fusion = fuse_risk_with_context(
        audio_score=raw_audio_score,
        keyword_risk_score=kw_score,
        transaction_context=tx_context,
        voiceprint_result=last_voiceprint_result,
    )
    whole_contextual = whole_fusion["contextual_score"]
    whole_band = score_to_band(whole_contextual)
    whole_risk = _risk_html(
        whole_contextual,
        context=f"Whole-Clip · {duration:.2f}s",
        base_fused_score=whole_fusion["base_fused_score"],
        audio_score=raw_audio_score,
        keyword_score=kw_score,
        transaction_multiplier=whole_fusion["transaction_multiplier"],
        contact_multiplier=whole_fusion["contact_multiplier"],
    )
    whole_prev = _prevention_html(whole_band)

    _SESSION_LOGGER.log_event(
        event_type="upload_analysis_whole_clip",
        risk_band=whole_band,
        probability_synthetic=whole_contextual,
        flagged=whole_band in ("medium", "high"),
    )

    # 3. Streaming simulation & fusion
    session = create_session()
    summary = simulate_stream(
        audio=waveform,
        session=session,
        real_time_paced=False,
    )
    stream_audio_score = float(summary.get("final_running_score", 0.0))
    stream_flagged = bool(summary.get("flagged", False))
    stream_s2f = summary.get("seconds_to_flag")
    stream_s2f_str = f"{stream_s2f:.2f}s" if stream_s2f is not None else "N/A"

    stream_fusion = fuse_risk_with_context(
        audio_score=stream_audio_score,
        keyword_risk_score=kw_score,
        transaction_context=tx_context,
        voiceprint_result=last_voiceprint_result,
    )
    stream_contextual = stream_fusion["contextual_score"]
    stream_band = score_to_band(stream_contextual)
    stream_context = (
        f"Streaming Simulation · flagged={stream_flagged} · "
        f"time-to-flag={stream_s2f_str}"
    )
    stream_risk = _risk_html(
        stream_contextual,
        context=stream_context,
        base_fused_score=stream_fusion["base_fused_score"],
        audio_score=stream_audio_score,
        keyword_score=kw_score,
        transaction_multiplier=stream_fusion["transaction_multiplier"],
        contact_multiplier=stream_fusion["contact_multiplier"],
    )
    stream_prev = _prevention_html(stream_band)

    _SESSION_LOGGER.log_event(
        event_type="upload_analysis_streaming",
        risk_band=stream_band,
        probability_synthetic=stream_contextual,
        flagged=stream_flagged,
    )

    return whole_risk, whole_prev, stream_risk, stream_prev, transcript_html, audio_path


# ---------------------------------------------------------------------------
# Explainability overlay callback
# ---------------------------------------------------------------------------

_OVERLAY_CAPTION = (
    "**Explainability overlay** — coarse, chunk-level attribution using "
    "1.5 s windows (stride 0.75 s, 50 % overlap). Each column of the heatmap "
    "shows the detector's synthetic-likelihood score for that time region; "
    "the spectrogram beneath shows the acoustic content. "
    "**Reliability note:** individual time-points may not align with fine "
    "acoustic detail — this is most reliable at clip granularity. "
    "Verified directionally correct on the soumya_neutral_01 real/synthetic "
    "pair; results on arbitrary clips are not guaranteed. "
    "See `data/metadata/PHASE5_STREAMING_NOTES.md` § Phase 10 for full details."
)

_OVERLAY_OUTPUT_DIR = Path("data") / "processed" / "overlays"


def generate_overlay(audio_path: str | None) -> tuple[str | None, str, str]:
    """Generate the explainability overlay PNG and descriptive attribution for the last-analyzed clip.

    Parameters
    ----------
    audio_path:
        File path of the audio to overlay.  Comes from ``last_audio_state``
        (the path stored by ``analyze_uploaded_file`` on its last run).

    Returns
    -------
    image_path_or_none : str | None
        Resolved path to the saved PNG, or ``None`` if generation failed.
    status_html : str
        Short status message rendered in the UI.
    explanation_html : str
        Descriptive rule-based explanation card rendered in the UI.
    """
    if audio_path is None:
        return (
            None,
            (
                '<p style="color:#94a3b8; font-style:italic;">'
                "Analyze a clip first, then click Generate Overlay.</p>"
            ),
            "",
        )

    try:
        waveform, sr = load_audio(audio_path, target_sr=16_000)
    except Exception as exc:
        logger.warning("generate_overlay: failed to load '%s': %s", audio_path, exc)
        return None, f'<p style="color:#fca5a5;">Could not load audio: {exc}</p>', ""

    out_name = Path(audio_path).stem + "_overlay.png"
    out_path = (_OVERLAY_OUTPUT_DIR / out_name).resolve()
    detector = get_detector()

    try:
        saved = render_explainability_overlay(
            waveform=waveform,
            sr=sr,
            detector=detector,
            output_path=out_path,
            # window_seconds and stride_seconds deliberately not overridden —
            # render_explainability_overlay defaults are 1.5 s / 0.75 s,
            # the empirically verified values from PHASE5_STREAMING_NOTES.md.
        )
        logger.debug("generate_overlay: saved overlay to %s", saved)

        scores, timestamps = windowed_attribution(
            waveform=waveform,
            sr=sr,
            detector=detector,
        )
        pred = detector.predict_waveform(waveform, sr)
        label = str(pred.get("label", "real"))
        desc = describe_attribution(scores=scores, timestamps=timestamps, label=label)
        explanation_html = _render_attribution_explanation_html(desc)

        status_html = (
            '<p style="color:#6ee7b7; font-size:0.88em;">'
            f"Overlay generated from: <code>{Path(audio_path).name}</code></p>"
        )
        return saved, status_html, explanation_html
    except Exception as exc:
        logger.exception("generate_overlay failed for '%s'", audio_path)
        return (
            None,
            (
                f'<p style="color:#fca5a5;">Overlay generation failed: '
                f"<code>{type(exc).__name__}: {exc}</code></p>"
            ),
            "",
        )


# ---------------------------------------------------------------------------
# Voiceprint Verification tab callbacks
# ---------------------------------------------------------------------------


def add_reference_clip(
    clip_path: str | None,
    clips: list[str],
) -> tuple[list[str], str, Any]:
    """Appends one recorded/uploaded clip's path to the enrollment clip list."""
    clips = list(clips or [])
    if clip_path:
        clips.append(clip_path)
    return clips, _format_clip_list(clips), gr.update(value=None)


def do_enroll(
    name: str | None,
    clips: list[str],
) -> tuple[str, Any, list[str], str]:
    """Enrolls a speaker from the accumulated reference-clip list."""
    clips = list(clips or [])

    if not name or not name.strip():
        return (
            "Please enter a speaker name before enrolling.",
            gr.update(choices=list_enrolled_speakers()),
            clips,
            _format_clip_list(clips),
        )
    if not clips:
        return (
            "Add at least one reference clip before enrolling.",
            gr.update(choices=list_enrolled_speakers()),
            clips,
            _format_clip_list(clips),
        )

    clean_name = name.strip()
    try:
        enroll_speaker(clean_name, clips, get_speaker_embedder())
    except Exception as exc:
        logger.exception("Voiceprint enrollment failed for '%s'", clean_name)
        return (
            f"Enrollment failed: {exc}",
            gr.update(choices=list_enrolled_speakers()),
            clips,
            _format_clip_list(clips),
        )

    speakers = list_enrolled_speakers()
    status = f"Enrolled '{clean_name}' from {len(clips)} reference clip(s)."
    return status, gr.update(choices=speakers, value=clean_name), [], _format_clip_list([])


def do_remove_enrollment(selected_name: str | None) -> tuple[str, Any]:
    """Deletes the selected speaker's voiceprint (right-to-erasure control)."""
    if not selected_name:
        return "No speaker selected to remove.", gr.update(choices=list_enrolled_speakers())

    deleted = delete_speaker(selected_name)
    speakers = list_enrolled_speakers()
    status = (
        f"Removed enrollment for '{selected_name}'."
        if deleted
        else f"No enrollment found for '{selected_name}' — nothing to remove."
    )
    return status, gr.update(choices=speakers, value=None)


def do_verify(
    selected_name: str | None,
    clip_path: str | None,
) -> tuple[str, dict[str, Any] | None]:
    """Verifies a clip against the selected enrolled speaker's voiceprint."""
    if not selected_name:
        return (
            _voiceprint_placeholder_html("Select an enrolled speaker first."),
            None,
        )
    if not clip_path:
        return (
            _voiceprint_placeholder_html("Provide a clip to verify and click Verify."),
            None,
        )

    try:
        waveform, sr = load_audio(clip_path, target_sr=16_000)
        result = verify_speaker(waveform, sr, selected_name, get_speaker_embedder())
    except FileNotFoundError:
        return (
            _voiceprint_placeholder_html(
                f"No enrolled voiceprint found for '{selected_name}'."
            ),
            None,
        )
    except ValueError as exc:
        return _voiceprint_placeholder_html(str(exc)), None
    except Exception as exc:
        logger.exception("Voiceprint verification failed for '%s'", selected_name)
        return _voiceprint_placeholder_html(f"Verification failed: {exc}"), None

    return _voiceprint_result_html(result), result


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

_DISCLAIMER = (
    "VoxGuard is a real-time voice cloning detection and prevention safeguard built "
    "using dual embedding backbones (wav2vec2 + WavLM), streaming risk scoring, "
    "multimodal context fusion (faster-whisper + red-flag scanner), and speaker voiceprint verification. "
    "All processing runs locally on your machine."
)

_DIVERGENCE_NOTE = (
    "> **Why two meters?** Whole-clip and streaming analysis evaluate audio from complementary "
    "perspectives — whole-clip inspects the complete waveform at once, streaming makes incremental "
    "sliding-window decisions. Both results are enriched with contextual multipliers and shown independently."
)

# ---------------------------------------------------------------------------
# Cross-Dataset Results tab content
# ---------------------------------------------------------------------------
# Read-only: loads the two evaluation reports straight from disk (rather
# than re-typing/duplicating their tables here) so this tab always reflects
# whatever scripts/evaluate_*.py last wrote, with no separate copy to fall
# out of sync.

_GENERALIZATION_REPORT_PATH = Path("models") / "reports" / "generalization_before_after.md"
_HINDI_COMPARISON_REPORT_PATH = Path("models") / "reports" / "hindi_training_comparison.md"

# IMPORTANT: VoxGuard's shipped production detector (get_detector(), used by
# every tab above) is the WEIGHTED-AVERAGE ENSEMBLE of the two Hindi-combined
# backbones — wav2vec2_hindi_combined_logreg + wavlm_hindi_combined_logreg
# (row 6 / the "Weighted-Average Ensemble (4+5)" row in the Hindi comparison
# table below). It is NOT either individual backbone alone, and the reports'
# own per-backbone "Variant A" labels refer to a *training strategy*
# (combined ASVspoof2019 + Hindi training data), not a single classifier
# that was shipped by itself — the production system always ensembles both.
_PRODUCTION_DETECTOR_NOTE = (
    "**What's actually shipped:** VoxGuard's production detector "
    "(`WeightedAverageDetector`, loaded by every tab above) is the "
    "**weighted-average ensemble of the two Hindi-combined backbones** — "
    "`wav2vec2_hindi_combined_logreg` + `wavlm_hindi_combined_logreg` "
    "(row 6 in the table below). Individual backbone rows and the "
    "\"Variant A / Variant B\" labels describe *training strategies* that "
    "were compared during development, not alternative single-model "
    "deployments — nothing here ships as a lone classifier."
)


def _load_report_markdown(path: Path) -> str:
    """Loads a report's raw Markdown content, or a clear placeholder if missing.

    Read-only display: never regenerates or edits the report. A missing
    file degrades to an explanatory message instead of crashing tab build.
    """
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("Cross-Dataset Results: report not found at %s", path)
        return (
            f"_Report not found at `{path.as_posix()}`. Run the corresponding "
            f"evaluation script (see BuildGuide.md) to generate it._"
        )
    except Exception as exc:
        logger.exception("Cross-Dataset Results: failed to read %s", path)
        return f"_Could not load report `{path.as_posix()}`: {exc}_"


def _splash_html() -> str:
    """A purely decorative intro screen that fades into the dashboard.

    Click-through from frame one (``pointer-events: none``) and fades out
    via a CSS-only animation — it never blocks or gates access to the real
    UI beneath it, and implies no feature beyond what's already rendered.
    """
    return (
        '<div class="vg-splash" aria-hidden="true">'
        '<div class="vg-splash-mark">'
        '<span class="material-symbols-outlined" style="font-size:0.9em; vertical-align:-4px; margin-right:8px; color:var(--vg-cyan);">shield</span>'
        'Vox<span>Guard</span>'
        '</div>'
        '<div class="vg-splash-sub">Initializing local detection engine</div>'
        '<div class="vg-splash-bar"></div>'
        '</div>'
    )


def _hero_html() -> str:
    """Renders the top hero banner: title, mission statement, status pill."""
    return (
        '<div class="vg-hero vg-reveal">'
        '<div class="vg-hero-top">'
        '<h1 class="vg-hero-title">'
        '<span class="material-symbols-outlined vg-hero-shield-icon">shield</span>'
        '<span class="vg-hero-mark">Vox</span>Guard'
        '</h1>'
        '<span class="vg-status-pill">'
        '<span class="vg-status-dot"></span>'
        '<span class="material-symbols-outlined" style="font-size:14px; vertical-align:-1px; margin-right:2px;">sensors</span>'
        'Local Engine Active'
        '</span>'
        '</div>'
        f'<p class="vg-hero-mission">{_DISCLAIMER}</p>'
        '</div>'
    )


def build_app() -> gr.Blocks:
    """Builds the Gradio UI shell for the VoxGuard demo app."""
    with gr.Blocks(
        title="VoxGuard — Voice Cloning Detection & Prevention",
        css=_CUSTOM_CSS,
        head=_HEAD_HTML,
    ) as demo:
        gr.HTML(_splash_html())
        gr.HTML(_hero_html())

        # App-level shared state for voiceprint verification across tabs
        last_voiceprint_result: gr.State = gr.State(value=None)

        with gr.Tabs():

            # ================================================================
            # Live Call Simulation tab — the centerpiece demo: live mic
            # streaming + the risk meter + live transcript/red-flag
            # highlighting + fused contextual score + prevention prompt,
            # all updating together as one view.
            # ================================================================
            with gr.Tab("Live Call Simulation"):
                gr.Markdown(
                    "Speak into the microphone to simulate a live call. VoxGuard "
                    "continuously fuses acoustic synthetic-voice detection with live "
                    "speech-to-text red-flag scanning and situational context "
                    "multipliers into one running contextual risk score — the risk "
                    "meter, transcript, and prevention guidance below all update "
                    "together in real time from the same stream."
                )
                session_state = gr.State(create_session)
                mic_transcript_state = gr.State("")

                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["vg-panel", "vg-reveal", "vg-reveal-1"]):
                        gr.HTML('<div class="vg-section-title"><span class="material-symbols-outlined vg-sec-icon">mic</span>Input &amp; Context</div>')
                        mic_input = gr.Audio(
                            sources=["microphone"],
                            streaming=True,
                            type="numpy",
                            label="Live Mic Input",
                        )
                        tx_context_mic = gr.Dropdown(
                            choices=TRANSACTION_CHOICES,
                            value="general_conversation",
                            label="Transaction / Call Context",
                            info="Select what the call is about to apply risk multipliers.",
                        )
                        reset_btn = gr.Button("Reset Session", variant="secondary")

                    with gr.Column(scale=1, elem_classes=["vg-panel", "vg-reveal", "vg-reveal-2"]):
                        gr.HTML('<div class="vg-section-title"><span class="material-symbols-outlined vg-sec-icon">radar</span>Live Risk Readout</div>')
                        mic_risk_html = gr.HTML(
                            value=_risk_html(0.0),
                            label="Contextual Risk Level",
                        )
                        mic_prevention_html = gr.HTML(
                            value="",
                            label="Prevention Guidance",
                        )
                        mic_transcript_html = gr.HTML(
                            value=_render_transcript_html("", [], []),
                            label="Live Call Transcript & Red-Flag Cues",
                        )
                        with gr.Row():
                            mic_flagged_out = gr.Textbox(
                                label="Acoustic Clone Flagged",
                                value="False",
                                interactive=False,
                            )
                            mic_s2f_out = gr.Textbox(
                                label="Seconds to Flag",
                                value="N/A",
                                interactive=False,
                            )

                mic_input.stream(
                    fn=process_audio_chunk,
                    inputs=[
                        mic_input,
                        session_state,
                        tx_context_mic,
                        last_voiceprint_result,
                        mic_transcript_state,
                    ],
                    outputs=[
                        session_state,
                        mic_risk_html,
                        mic_prevention_html,
                        mic_transcript_html,
                        mic_flagged_out,
                        mic_s2f_out,
                        mic_transcript_state,
                    ],
                )

                reset_btn.click(
                    fn=reset_streaming_session,
                    inputs=[session_state],
                    outputs=[
                        session_state,
                        mic_risk_html,
                        mic_prevention_html,
                        mic_transcript_html,
                        mic_flagged_out,
                        mic_s2f_out,
                        mic_transcript_state,
                    ],
                )

            # ================================================================
            # Upload & Analyze tab — whole-clip + streaming-simulation
            # detection, transcript/red-flag scanning, fused contextual
            # scoring, and the Phase 10 explainability overlay for any
            # uploaded file.
            # ================================================================
            with gr.Tab("Upload & Analyze"):
                gr.Markdown(
                    "Upload an audio file to run whole-clip detection, streaming-simulation "
                    "replay, automated speech transcription with scam keyword detection, and "
                    "an explainability overlay — all fused into the same contextual risk score "
                    "used on the Live Call Simulation tab."
                )
                gr.Markdown(_DIVERGENCE_NOTE)

                # Stores the filepath of the most-recently analyzed clip so the
                # explainability overlay always corresponds to the current verdict.
                last_audio_state = gr.State(value=None)

                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["vg-panel", "vg-reveal", "vg-reveal-1"]):
                        gr.HTML('<div class="vg-section-title"><span class="material-symbols-outlined vg-sec-icon">audio_file</span>Input &amp; Context</div>')
                        upload_audio = gr.Audio(
                            sources=["upload"],
                            streaming=False,
                            type="filepath",
                            label="Upload Audio File",
                        )
                        tx_context_upload = gr.Dropdown(
                            choices=TRANSACTION_CHOICES,
                            value="general_conversation",
                            label="Transaction / Call Context",
                            info="Select what the call is about to apply risk multipliers.",
                        )
                        analyze_btn = gr.Button("Analyze", variant="primary")

                    with gr.Column(scale=2, elem_classes=["vg-panel", "vg-reveal", "vg-reveal-2"]):
                        gr.HTML('<div class="vg-section-title"><span class="material-symbols-outlined vg-sec-icon">subtitles</span>Transcript &amp; Red-Flag Cues</div>')
                        upload_transcript_html = gr.HTML(
                            value=_render_transcript_html("", [], []),
                            label="Call Transcript & Red-Flag Cues",
                        )

                        gr.HTML('<div class="vg-section-title"><span class="material-symbols-outlined vg-sec-icon">graphic_eq</span>Whole-Clip Analysis</div>')
                        upload_whole_risk = gr.HTML(
                            value=(
                                '<p style="color:#94a3b8; font-style:italic;">'
                                "Upload a file and click Analyze.</p>"
                            ),
                        )
                        upload_whole_prev = gr.HTML(value="")

                        gr.HTML('<div class="vg-section-title"><span class="material-symbols-outlined vg-sec-icon">stream</span>Streaming Simulation</div>')
                        upload_stream_risk = gr.HTML(
                            value=(
                                '<p style="color:#94a3b8; font-style:italic;">'
                                "Upload a file and click Analyze.</p>"
                            ),
                        )
                        upload_stream_prev = gr.HTML(value="")

                # ---- Explainability section --------------------------------
                with gr.Accordion("Explainability Overlay & Attribution Analysis", open=False, elem_classes=["vg-panel"]):
                    gr.HTML('<div class="vg-section-title"><span class="material-symbols-outlined vg-sec-icon">insights</span>Spectrogram Heatmap &amp; Attribution Breakdown</div>')
                    gr.Markdown(
                        "Generates a mel-spectrogram with a synthetic-likelihood heatmap "
                        "overlay and rule-based attribution analysis for the clip analyzed above. "
                        "Uses 1.5 s windows / 0.75 s stride (empirically verified defaults)."
                    )
                    overlay_btn = gr.Button(
                        "Generate Explainability Overlay", variant="secondary"
                    )
                    overlay_status = gr.HTML(value="")
                    overlay_explanation = gr.HTML(value="")
                    overlay_image = gr.Image(
                        label="Explainability Overlay",
                        type="filepath",
                        show_download_button=True,
                        visible=True,
                        value=None,
                    )
                    gr.Markdown(_OVERLAY_CAPTION)

                analyze_btn.click(
                    fn=analyze_uploaded_file,
                    inputs=[
                        upload_audio,
                        tx_context_upload,
                        last_voiceprint_result,
                    ],
                    outputs=[
                        upload_whole_risk,
                        upload_whole_prev,
                        upload_stream_risk,
                        upload_stream_prev,
                        upload_transcript_html,
                        last_audio_state,
                    ],
                )

                overlay_btn.click(
                    fn=generate_overlay,
                    inputs=[last_audio_state],
                    outputs=[overlay_image, overlay_status, overlay_explanation],
                )

            # ================================================================
            # Voiceprint Verification tab
            # ================================================================
            with gr.Tab("Voiceprint Verification"):
                gr.Markdown(
                    "Enroll a trusted contact's voice, then verify whether a "
                    "live or uploaded clip is actually them. This answers a "
                    "different question than the tabs above — not \"is this "
                    "voice synthetic,\" but \"is this who they claim to be\" — "
                    "which catches an attacker using a *different real* voice, "
                    "not a clone at all. Successful or failed verification here "
                    "automatically updates the contact familiarity multiplier across all tabs."
                )

                clips_state = gr.State([])

                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["vg-panel", "vg-reveal", "vg-reveal-1"]):
                        gr.HTML('<div class="vg-section-title"><span class="material-symbols-outlined vg-sec-icon">person_add</span>Enroll a Speaker</div>')
                        enroll_name = gr.Textbox(
                            label="Speaker Name",
                            placeholder="e.g. priya",
                        )
                        enroll_clip_input = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="Reference Clip (2-3 clips of 5-10s recommended)",
                        )
                        add_clip_btn = gr.Button("Add Clip", variant="secondary")
                        clips_display = gr.Textbox(
                            label="Reference Clips Added",
                            value=_format_clip_list([]),
                            interactive=False,
                            lines=4,
                        )
                        enroll_btn = gr.Button("Enroll", variant="primary")
                        enroll_status = gr.Markdown(value="")

                        gr.HTML('<div class="vg-section-title"><span class="material-symbols-outlined vg-sec-icon">group</span>Enrolled Speakers</div>')
                        gr.Markdown(
                            "Select a speaker below to remove their enrollment, or to "
                            "verify a clip against them on the right."
                        )
                        enrolled_dropdown = gr.Dropdown(
                            choices=list_enrolled_speakers(),
                            label="Enrolled Speaker",
                            value=None,
                        )
                        remove_btn = gr.Button("Remove Enrollment", variant="stop")
                        remove_status = gr.Markdown(value="")

                    with gr.Column(scale=1, elem_classes=["vg-panel", "vg-reveal", "vg-reveal-2"]):
                        gr.HTML('<div class="vg-section-title"><span class="material-symbols-outlined vg-sec-icon">how_to_reg</span>Verify a Clip</div>')
                        verify_clip_input = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="Clip to Verify",
                        )
                        verify_btn = gr.Button("Verify", variant="primary")
                        verify_result_html = gr.HTML(
                            value=_voiceprint_placeholder_html(
                                "Enroll a speaker on the left, then verify a clip "
                                "against them here."
                            )
                        )

                add_clip_btn.click(
                    fn=add_reference_clip,
                    inputs=[enroll_clip_input, clips_state],
                    outputs=[clips_state, clips_display, enroll_clip_input],
                )

                enroll_btn.click(
                    fn=do_enroll,
                    inputs=[enroll_name, clips_state],
                    outputs=[enroll_status, enrolled_dropdown, clips_state, clips_display],
                )

                remove_btn.click(
                    fn=do_remove_enrollment,
                    inputs=[enrolled_dropdown],
                    outputs=[remove_status, enrolled_dropdown],
                )

                verify_btn.click(
                    fn=do_verify,
                    inputs=[enrolled_dropdown, verify_clip_input],
                    outputs=[verify_result_html, last_voiceprint_result],
                )

            # ================================================================
            # Cross-Dataset Results tab — read-only. Displays Phase 3's
            # generalization report and Phase 4's Hindi/Hinglish training
            # comparison report so judges can see both without leaving the
            # app. No inputs, no callbacks — content is loaded once at
            # app-build time straight from the report files on disk.
            # ================================================================
            with gr.Tab("Cross-Dataset Results"):
                gr.Markdown(
                    "Read-only evaluation evidence generated by this project's "
                    "training/evaluation scripts — not live-recomputed here."
                )
                gr.Markdown(_PRODUCTION_DETECTOR_NOTE, elem_classes=["vg-panel"])

                with gr.Column(elem_classes=["vg-panel", "vg-reveal", "vg-reveal-1"]):
                    gr.HTML(
                        '<div class="vg-section-title">'
                        '<span class="material-symbols-outlined vg-sec-icon">public</span>'
                        "Cross-Dataset Generalization (Phase 3)</div>"
                    )
                    gr.Markdown(_load_report_markdown(_GENERALIZATION_REPORT_PATH))

                with gr.Column(elem_classes=["vg-panel", "vg-reveal", "vg-reveal-2"]):
                    gr.HTML(
                        '<div class="vg-section-title">'
                        '<span class="material-symbols-outlined vg-sec-icon">translate</span>'
                        "Hindi/Hinglish Training Comparison (Phase 4)</div>"
                    )
                    gr.Markdown(_load_report_markdown(_HINDI_COMPARISON_REPORT_PATH))

    return demo


app = build_app()


if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", show_api=False)
