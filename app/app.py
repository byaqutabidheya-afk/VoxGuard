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
from typing import Any

import gradio as gr
import numpy as np

from voxguard.classifier.ensemble import WeightedAverageDetector
from voxguard.privacy.session_log import SessionLogger
from voxguard.risk.bands import score_to_band
from voxguard.risk.prevention import get_prevention_message
from voxguard.streaming.session import StreamingSession, simulate_stream
from voxguard.utils.audio_io import load_audio

logger = logging.getLogger(__name__)

_DETECTOR: WeightedAverageDetector | None = None
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

    return demo


app = build_app()


if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", show_api=False)
