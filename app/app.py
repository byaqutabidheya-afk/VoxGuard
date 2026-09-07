"""
app.py — Gradio entrypoint for VoxGuard.

Launches an interactive web UI for submitting audio files or live
microphone input and receiving real/synthetic verdicts with explanations.

This process imports detection logic from src/voxguard and is kept
separate from the FastAPI REST process (api/main.py) so the two can
be deployed independently while sharing the same core library.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np

from voxguard.classifier.ensemble import WeightedAverageDetector
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
_SESSION_LOGGER = SessionLogger()
_SESSION_LOGGER.purge_older_than(30)


# ---------------------------------------------------------------------------
# Risk-band visual rendering
# ---------------------------------------------------------------------------
#
# Contrast design principle
# ─────────────────────────
# Every element that carries an explicit `background` must also carry an
# explicit `color` referencing a shade that contrasts against *that
# background*, not against the page.  Gradio's dark-mode theme sets a
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
        # prevention bg — unused for low, included for consistency
        "prev_bg": "#d4edda",
        "prev_text": "#0d3b1e",
    },
    "medium": {
        "bg": "#fff3cd",        # light amber tint
        "border": "#d97706",    # darker amber border (was #ffc107 — poor contrast)
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
) -> str:
    """Return an HTML block showing the color-coded risk band + raw probability.

    All colors are set explicitly on every element so the block is readable
    in both Gradio light mode and dark mode regardless of theme inheritance.

    Parameters
    ----------
    probability_synthetic:
        Raw classifier output in [0, 1], or None (silence / non-speech gate).
    context:
        Short descriptive label shown above the band badge, e.g.
        "Whole-Clip" or "Streaming Simulation".
    """
    band = score_to_band(probability_synthetic)
    s = _BAND_STYLES[band]

    prob_text = (
        f"Raw probability: {probability_synthetic:.4f}"
        if probability_synthetic is not None
        else "Raw probability: N/A (no speech detected)"
    )
    prob_line = (
        f'<p style="margin:4px 0 0 0; font-size:0.82em; '
        f"color:{s['muted']}; background:transparent;\">"
        f"{prob_text}</p>"
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
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Prevention prompt rendering
# ---------------------------------------------------------------------------
# Message copy lives in voxguard.risk.prevention (MEDIUM_RISK_MESSAGE,
# HIGH_RISK_MESSAGE).  This function is responsible only for wrapping that
# text in styled HTML — it never owns the copy itself.


def _prevention_html(band: str) -> str:
    """Return an HTML prevention-prompt block for medium/high bands.

    Returns an empty string for low and inconclusive (no alert fatigue).

    All colors are set explicitly — background, text, list items, strong
    tags — so the block is readable in both Gradio light mode and dark mode
    regardless of any inherited theme foreground.
    """
    if band not in ("medium", "high"):
        return ""

    s = _BAND_STYLES[band]
    accent = s["border"]
    bg = s["prev_bg"]
    fg = s["prev_text"]
    md_text = get_prevention_message(band)  # sourced from voxguard.risk.prevention

    # Convert simple markdown (bold **…** and bullets -) to HTML inline.
    import re

    html_text = re.sub(
        r"\*\*(.*?)\*\*",
        # <strong> needs an explicit color too — it inherits from the <div>,
        # but some browsers / Gradio shadow-DOM resets strip that; be explicit.
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
# Reuses _BAND_STYLES's "low" (green) and "high" (red) palettes so a MATCH/
# MISMATCH card reads consistently with the risk meter above, per the same
# contrast rules (every background carries an explicit, paired foreground).


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


def create_session(sample_rate: int | None = None) -> StreamingSession:
    """Creates a new StreamingSession instance with the shared detector."""
    return StreamingSession(detector=get_detector(), sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# Live Mic callbacks
# ---------------------------------------------------------------------------


def process_audio_chunk(
    audio: tuple[int, np.ndarray] | None,
    session: StreamingSession | None,
) -> tuple[StreamingSession, str, str, str, str]:
    """Processes an incoming streaming audio chunk and updates session risk state.

    Returns
    -------
    session, risk_html, prevention_html, flagged_str, seconds_to_flag_str
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
        band = score_to_band(running_score)
        return (
            session,
            _risk_html(running_score),
            _prevention_html(band),
            str(flagged),
            s2f_str,
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

    try:
        buf_sr = float(session.buffer.sample_rate)
        chunk_s = float(session.buffer.chunk_samples)
        stride_s = float(session.buffer.stride_samples)
        print(
            f"[DEBUG Gradio Audio Chunk] incoming_sr={sr!r}, "
            f"raw_shape={getattr(waveform, 'shape', None)}, raw_dtype={getattr(waveform, 'dtype', None)}, "
            f"processed_samples={y.size}, "
            f"buffer_sr={int(buf_sr)}, chunk_samples={int(chunk_s)} "
            f"({chunk_s / buf_sr:.2f}s), "
            f"stride_samples={int(stride_s)} "
            f"({stride_s / buf_sr:.2f}s)"
        )
    except Exception:
        pass

    if y.size == 0:
        current_score = session.risk_score.current()
        running_score = 0.0 if current_score is None else float(current_score)
        flagged = session._consecutive_flags >= session.consecutive_flags_required
        s2f_str = (
            f"{session._seconds_to_flag:.2f}s"
            if session._seconds_to_flag is not None
            else "N/A"
        )
        band = score_to_band(running_score)
        return (
            session,
            _risk_html(running_score),
            _prevention_html(band),
            str(flagged),
            s2f_str,
        )

    result = session.push_audio(y, sr=sr)
    running_score = float(result["running_score"])
    flagged = bool(result["flagged"])
    s2f = result["seconds_to_flag"]
    s2f_str = f"{s2f:.2f}s" if s2f is not None else "N/A"
    band = score_to_band(running_score)

    if flagged and not getattr(session, "_logged_flag_event", False):
        session._logged_flag_event = True
        _SESSION_LOGGER.log_event(
            event_type="flag_event",
            risk_band=band,
            probability_synthetic=running_score,
            flagged=flagged,
        )

    return (
        session,
        _risk_html(running_score),
        _prevention_html(band),
        str(flagged),
        s2f_str,
    )


def reset_streaming_session(
    session: StreamingSession | None,
) -> tuple[StreamingSession, str, str, str, str]:
    """Resets the streaming session and returns cleared indicators."""
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
    return session, _risk_html(0.0), "", "False", "N/A"


# ---------------------------------------------------------------------------
# Upload File callback
# ---------------------------------------------------------------------------


def analyze_uploaded_file(
    audio_path: str | None,
) -> tuple[str, str, str, str]:
    """Run whole-clip and streaming-simulation analysis on an uploaded audio file.

    Returns
    -------
    whole_risk_html, whole_prevention_html, stream_risk_html, stream_prevention_html
    """
    if audio_path is None:
        placeholder = (
            '<p style="color:#666; font-style:italic;">Upload a file and click Analyze.</p>'
        )
        return placeholder, "", placeholder, ""

    waveform, sr = load_audio(audio_path, target_sr=16_000)
    duration = float(len(waveform)) / float(sr)

    # ---- Whole-clip -------------------------------------------------------
    whole_clip = get_detector().predict_waveform(waveform, sr)
    probability = whole_clip.get("probability_synthetic")
    probability = None if probability is None else float(probability)

    whole_band = score_to_band(probability)
    whole_risk = _risk_html(probability, context=f"Whole-Clip · {duration:.2f}s")
    whole_prev = _prevention_html(whole_band)

    _SESSION_LOGGER.log_event(
        event_type="upload_analysis_whole_clip",
        risk_band=whole_band,
        probability_synthetic=probability if probability is not None else 0.0,
        flagged=whole_band in ("medium", "high"),
    )

    # ---- Streaming simulation ---------------------------------------------
    session = create_session()
    summary = simulate_stream(
        audio=waveform,
        session=session,
        real_time_paced=False,
    )
    stream_prob = float(summary.get("final_running_score", 0.0))
    stream_flagged = bool(summary.get("flagged", False))
    stream_s2f = summary.get("seconds_to_flag")
    stream_s2f_str = f"{stream_s2f:.2f}s" if stream_s2f is not None else "N/A"

    stream_band = score_to_band(stream_prob)
    stream_context = (
        f"Streaming Simulation · flagged={stream_flagged} · "
        f"time-to-flag={stream_s2f_str}"
    )
    stream_risk = _risk_html(stream_prob, context=stream_context)
    stream_prev = _prevention_html(stream_band)

    _SESSION_LOGGER.log_event(
        event_type="upload_analysis_streaming",
        risk_band=stream_band,
        probability_synthetic=stream_prob,
        flagged=stream_flagged,
    )

    return whole_risk, whole_prev, stream_risk, stream_prev


# ---------------------------------------------------------------------------
# Voiceprint Verification tab callbacks
# ---------------------------------------------------------------------------
# gr.Audio in the pinned Gradio version (4.44.1) has no multi-file/file_count
# option — its `value` type is a single str|Path|(sr, array), not a list — so
# multi-clip enrollment uses an "add another clip" pattern instead: one
# gr.Audio recorder/uploader, an "Add Clip" button that appends its path to a
# gr.State list, and "Enroll" consuming the accumulated list.


def add_reference_clip(
    clip_path: str | None,
    clips: list[str],
) -> tuple[list[str], str, Any]:
    """Appends one recorded/uploaded clip's path to the enrollment clip list.

    Returns the updated list, its display text, and a reset (cleared)
    audio-input value so the recorder is ready for the next take.
    """
    clips = list(clips or [])
    if clip_path:
        clips.append(clip_path)
    return clips, _format_clip_list(clips), gr.update(value=None)


def do_enroll(
    name: str | None,
    clips: list[str],
) -> tuple[str, Any, list[str], str]:
    """Enrolls a speaker from the accumulated reference-clip list.

    Returns a status message, an updated enrolled-speaker dropdown, and a
    reset clip list/display (successful enrollment clears the working list
    so the next enrollment doesn't accidentally reuse this speaker's clips).
    """
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
    """Verifies a clip against the selected enrolled speaker's voiceprint.

    Returns the rendered result card and the raw verify_speaker() result
    (or None) to store in the shared last_voiceprint_result state.
    """
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
    "VoxGuard is a hackathon prototype that detects AI-cloned voice in call audio "
    "using dual embedding backbones and streaming risk scoring. This demo simulates "
    "call audio via live microphone input or uploaded files — it does **not** intercept "
    "real telecom traffic. All processing runs locally on your machine."
)

_DIVERGENCE_NOTE = (
    "> **Why two meters?** Whole-clip and streaming analysis can disagree on the same "
    "audio — whole-clip sees the full waveform at once, streaming makes incremental "
    "decisions. Both results are shown independently so you can see the difference "
    "rather than have it hidden by averaging."
)


def build_app() -> gr.Blocks:
    """Builds the Gradio UI shell for the VoxGuard demo app."""
    with gr.Blocks(title="VoxGuard — Voice Cloning Detection & Prevention") as demo:
        gr.Markdown("# VoxGuard — Voice Cloning Detection & Prevention")
        gr.Markdown(_DISCLAIMER)

        # App-level shared state (not nested in any single tab): Phase 9's
        # fusion UI reads this same state object to fold "is this a known
        # contact" into its risk score, so both the state's name and its
        # shape — exactly {"match": bool, "similarity": float,
        # "enrolled_name": str}, or None before any verification has run —
        # are a contract that phase depends on. Only the Voiceprint
        # Verification tab's Verify button writes to it.
        last_voiceprint_result: gr.State = gr.State(value=None)

        with gr.Tabs():

            # ================================================================
            # Live Mic tab
            # ================================================================
            with gr.Tab("Live Mic"):
                gr.Markdown(
                    "Speak into the microphone to stream audio in real time. "
                    "VoxGuard continuously scores chunked risk and flags sustained anomalies."
                )
                session_state = gr.State(create_session)

                with gr.Row():
                    with gr.Column(scale=1):
                        mic_input = gr.Audio(
                            sources=["microphone"],
                            streaming=True,
                            type="numpy",
                            label="Live Mic Input",
                        )
                        reset_btn = gr.Button("Reset Session", variant="secondary")

                    with gr.Column(scale=1):
                        mic_risk_html = gr.HTML(
                            value=_risk_html(0.0),
                            label="Risk Level",
                        )
                        mic_prevention_html = gr.HTML(
                            value="",
                            label="Prevention Guidance",
                        )
                        mic_flagged_out = gr.Textbox(
                            label="Flagged (Synthetic Voice Detected)",
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
                    inputs=[mic_input, session_state],
                    outputs=[
                        session_state,
                        mic_risk_html,
                        mic_prevention_html,
                        mic_flagged_out,
                        mic_s2f_out,
                    ],
                )

                reset_btn.click(
                    fn=reset_streaming_session,
                    inputs=[session_state],
                    outputs=[
                        session_state,
                        mic_risk_html,
                        mic_prevention_html,
                        mic_flagged_out,
                        mic_s2f_out,
                    ],
                )

            # ================================================================
            # Upload File tab
            # ================================================================
            with gr.Tab("Upload File"):
                gr.Markdown(
                    "Upload an audio file to run both whole-clip detection and "
                    "a fast streaming-simulation replay."
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
                        analyze_btn = gr.Button("Analyze", variant="primary")

                    with gr.Column(scale=2):
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
                    inputs=[upload_audio],
                    outputs=[
                        upload_whole_risk,
                        upload_whole_prev,
                        upload_stream_risk,
                        upload_stream_prev,
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
                    "not a clone at all."
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
