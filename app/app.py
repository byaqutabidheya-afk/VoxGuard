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
from voxguard.fusion.context import (
    get_contact_familiarity_multiplier,
    get_transaction_multiplier,
)
from voxguard.fusion.fuse import fuse_risk_with_context
from voxguard.fusion.redflags import normalize_apostrophes, scan_for_redflags
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
# All text colors below have been chosen for ≥ 4.5:1 contrast ratio
# (WCAG AA) against their paired background, verified with the WebAIM
# contrast checker.
#
# "muted" = the color used for secondary / small-print text inside a card.
# It is always explicitly set — never left as `color:inherit`.

_BAND_STYLES: dict[str, dict[str, str]] = {
    "low": {
        "bg": "#d4edda",        # light green tint
        "border": "#28a745",    # green
        "text": "#0d3b1e",      # very dark green  — 10.2:1 on #d4edda
        "muted": "#2d6a3f",     # dark green        —  5.1:1 on #d4edda
        "label": "LOW RISK",
        "prev_bg": "#d4edda",
        "prev_text": "#0d3b1e",
    },
    "medium": {
        "bg": "#fff3cd",        # light amber tint
        "border": "#d97706",    # darker amber border
        "text": "#3d2000",      # very dark brown   — 11.4:1 on #fff3cd
        "muted": "#7a4400",     # dark amber-brown  —  5.4:1 on #fff3cd
        "label": "MEDIUM RISK",
        "prev_bg": "#fef9e7",   # slightly warmer off-white amber tint
        "prev_text": "#3d2000", # same dark brown
    },
    "high": {
        "bg": "#f8d7da",        # light red/pink tint
        "border": "#c0392b",    # deep red border
        "text": "#4a0010",      # very dark crimson — 10.8:1 on #f8d7da
        "muted": "#7b1d2a",     # dark red           —  5.2:1 on #f8d7da
        "label": "HIGH RISK",
        "prev_bg": "#fdf0f1",   # very pale pink
        "prev_text": "#4a0010", # same dark crimson
    },
    "inconclusive": {
        "bg": "#e2e3e5",        # light gray tint
        "border": "#5a6270",    # mid-dark gray border
        "text": "#1a1d21",      # near-black         — 11.6:1 on #e2e3e5
        "muted": "#3b4149",     # dark gray           —  6.5:1 on #e2e3e5
        "label": "INCONCLUSIVE",
        "prev_bg": "#e2e3e5",
        "prev_text": "#1a1d21",
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
        f'<p style="margin:4px 0 0 0; font-size:0.92em; font-weight:600; '
        f"color:{s['text']}; background:transparent;\">"
        f"{prob_text}</p>"
    )

    breakdown_lines: list[str] = []
    if base_fused_score is not None:
        breakdown_lines.append(
            f"Base fused score (70% audio + 30% text): <b>{base_fused_score:.4f}</b>"
        )
    if audio_score is not None and keyword_score is not None:
        breakdown_lines.append(
            f"Signals: Audio score = {audio_score:.4f} · Red-flag score = {keyword_score:.4f}"
        )
    if transaction_multiplier is not None and contact_multiplier is not None:
        breakdown_lines.append(
            f"Multipliers: Transaction ×{transaction_multiplier:.2f} · Contact ×{contact_multiplier:.2f}"
        )

    breakdown_html = ""
    if breakdown_lines:
        items = "<br>".join(breakdown_lines)
        breakdown_html = (
            f'<div style="margin-top:8px; padding-top:6px; border-top:1px dashed {s["border"]}; '
            f'font-size:0.80em; color:{s["muted"]}; line-height:1.4;">'
            f"{items}"
            f"</div>"
        )

    context_line = (
        f'<p style="margin:0 0 4px 0; font-size:0.78em; font-weight:600; '
        f"color:{s['muted']}; background:transparent; "
        f'text-transform:uppercase; letter-spacing:0.05em;">'
        f"{context}</p>"
        if context
        else ""
    )

    return (
        f'<div style="'
        f"background:{s['bg']}; "
        f"color:{s['text']}; "
        f"border:2px solid {s['border']}; "
        f"border-radius:8px; "
        f"padding:12px 16px; "
        f'margin:4px 0;">'
        f"{context_line}"
        f'<p style="margin:0; font-size:1.3em; font-weight:700; '
        f"color:{s['text']}; background:transparent;\">"
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
            '<div style="background:#f8f9fa; color:#666; border:1px solid #ced4da; '
            'border-radius:6px; padding:12px; font-style:italic;">'
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
                f'<mark style="background:#ffeb3b; color:#212121; padding:2px 4px; '
                f'border-radius:3px; font-weight:600;">{matched_str}</mark>'
            )

        highlighted_body = re.sub(pattern, _replace_match, clean_text)
    else:
        highlighted_body = html.escape(clean_text)

    badge_html = ""
    if categories:
        badges = " ".join(
            f'<span style="background:#ffebee; color:#c62828; border:1px solid #ef9a9a; '
            f'padding:2px 6px; border-radius:4px; font-size:0.8em; font-weight:600; '
            f'text-transform:uppercase;">{cat.replace("_", " ")}</span>'
            for cat in categories
        )
        badge_html = (
            f'<div style="margin-top:8px; display:flex; gap:6px; flex-wrap:wrap; align-items:center;">'
            f'<strong style="font-size:0.85em; color:#495057;">Red-flag categories:</strong> {badges}'
            f"</div>"
        )

    return (
        f'<div style="background:#f8f9fa; color:#212529; border:1px solid #ced4da; '
        f'border-radius:6px; padding:12px; font-size:0.92em; line-height:1.5;">'
        f'<div style="margin-bottom:6px; font-weight:600; color:#495057;">Transcript:</div>'
        f'<div style="color:#212529;">{highlighted_body}</div>'
        f"{badge_html}"
        f"</div>"
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

    return (
        f'<div style="'
        f"background:{bg}; "
        f"color:{fg}; "
        f"border-left:4px solid {accent}; "
        f"border-radius:0 6px 6px 0; "
        f"padding:12px 16px; "
        f'margin:8px 0; font-size:0.92em; line-height:1.5;">'
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

    return (
        f'<div style="'
        f"background:{s['bg']}; "
        f"color:{s['text']}; "
        f"border:2px solid {s['border']}; "
        f"border-radius:8px; "
        f"padding:12px 16px; "
        f'margin:4px 0;">'
        f'<p style="margin:0 0 4px 0; font-size:0.78em; font-weight:600; '
        f"color:{s['muted']}; background:transparent; "
        f'text-transform:uppercase; letter-spacing:0.05em;">'
        f"Voiceprint check vs. &#39;{enrolled_name}&#39;</p>"
        f'<p style="margin:0; font-size:1.3em; font-weight:700; '
        f"color:{s['text']}; background:transparent;\">"
        f"{label}</p>"
        f'<p style="margin:4px 0 0 0; font-size:0.82em; '
        f"color:{s['muted']}; background:transparent;\">"
        f"Cosine similarity: {similarity:.4f}</p>"
        f"</div>"
    )


def _voiceprint_placeholder_html(message: str) -> str:
    """Renders a neutral italic placeholder for the voiceprint result card."""
    return f'<p style="color:#666; font-style:italic;">{message}</p>'


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
) -> tuple[str, str, str, str, str]:
    """Run whole-clip, streaming-simulation, and transcript analysis on an uploaded audio file.

    Returns
    -------
    whole_risk_html, whole_prevention_html, stream_risk_html, stream_prevention_html, transcript_html
    """
    if audio_path is None:
        placeholder = (
            '<p style="color:#666; font-style:italic;">Upload a file and click Analyze.</p>'
        )
        return placeholder, "", placeholder, "", _render_transcript_html("", [], [])

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

    return whole_risk, whole_prev, stream_risk, stream_prev, transcript_html


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


def build_app() -> gr.Blocks:
    """Builds the Gradio UI shell for the VoxGuard demo app."""
    with gr.Blocks(title="VoxGuard — Voice Cloning Detection & Prevention") as demo:
        gr.Markdown("# VoxGuard — Voice Cloning Detection & Prevention")
        gr.Markdown(_DISCLAIMER)

        # App-level shared state for voiceprint verification across tabs
        last_voiceprint_result: gr.State = gr.State(value=None)

        with gr.Tabs():

            # ================================================================
            # Live Mic tab
            # ================================================================
            with gr.Tab("Live Mic"):
                gr.Markdown(
                    "Speak into the microphone to stream audio in real time. "
                    "VoxGuard continuously calculates overall contextual call risk by combining "
                    "acoustic synthetic voice detection with live speech-to-text red-flag scanning "
                    "and situational context multipliers."
                )
                session_state = gr.State(create_session)
                mic_transcript_state = gr.State("")

                with gr.Row():
                    with gr.Column(scale=1):
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

                    with gr.Column(scale=1):
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
            # Upload File tab
            # ================================================================
            with gr.Tab("Upload File"):
                gr.Markdown(
                    "Upload an audio file to run whole-clip detection, streaming-simulation replay, "
                    "and automated speech transcription with scam keyword detection."
                )
                gr.Markdown(_DIVERGENCE_NOTE)

                with gr.Row():
                    with gr.Column(scale=1):
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

                    with gr.Column(scale=2):
                        upload_transcript_html = gr.HTML(
                            value=_render_transcript_html("", [], []),
                            label="Call Transcript & Red-Flag Cues",
                        )

                        gr.Markdown("### Whole-Clip Analysis")
                        upload_whole_risk = gr.HTML(
                            value=(
                                '<p style="color:#666; font-style:italic;">'
                                "Upload a file and click Analyze.</p>"
                            ),
                        )
                        upload_whole_prev = gr.HTML(value="")

                        gr.Markdown("### Streaming Simulation")
                        upload_stream_risk = gr.HTML(
                            value=(
                                '<p style="color:#666; font-style:italic;">'
                                "Upload a file and click Analyze.</p>"
                            ),
                        )
                        upload_stream_prev = gr.HTML(value="")

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
                    ],
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
                    with gr.Column(scale=1):
                        gr.Markdown("### Enroll a Speaker")
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

                        gr.Markdown("### Enrolled Speakers")
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

                    with gr.Column(scale=1):
                        gr.Markdown("### Verify a Clip")
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

    return demo


app = build_app()


if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", show_api=False)
