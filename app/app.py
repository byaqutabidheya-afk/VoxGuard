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

from __future__ import annotations

import logging
from typing import Any

import gradio as gr
import numpy as np

from voxguard.classifier.ensemble import WeightedAverageDetector
from voxguard.privacy.session_log import SessionLogger
from voxguard.streaming.session import StreamingSession, simulate_stream
from voxguard.utils.audio_io import load_audio

logger = logging.getLogger(__name__)

_DETECTOR: WeightedAverageDetector | None = None
_SESSION_LOGGER = SessionLogger()
_SESSION_LOGGER.purge_older_than(30)


def _risk_band(score: float) -> str:
    if score < 0.4:
        return "low"
    if score < 0.7:
        return "medium"
    return "high"


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


def process_audio_chunk(
    audio: tuple[int, np.ndarray] | None,
    session: StreamingSession | None,
) -> tuple[StreamingSession, float, str, str]:
    """Processes an incoming streaming audio chunk and updates session risk state."""
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
        return session, round(running_score, 4), str(flagged), s2f_str

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
        return session, round(running_score, 4), str(flagged), s2f_str

    result = session.push_audio(y, sr=sr)
    running_score = float(result["running_score"])
    flagged = bool(result["flagged"])
    s2f = result["seconds_to_flag"]
    s2f_str = f"{s2f:.2f}s" if s2f is not None else "N/A"

    if flagged and not getattr(session, "_logged_flag_event", False):
        session._logged_flag_event = True
        _SESSION_LOGGER.log_event(
            event_type="flag_event",
            risk_band=_risk_band(running_score),
            probability_synthetic=running_score,
            flagged=flagged,
        )

    return session, round(running_score, 4), str(flagged), s2f_str


def reset_streaming_session(
    session: StreamingSession | None,
) -> tuple[StreamingSession, float, str, str]:
    """Resets the streaming session and returns cleared indicators."""
    if session is not None:
        current_score = session.risk_score.current()
        current_prob = 0.0 if current_score is None else float(current_score)
        _SESSION_LOGGER.log_event(
            event_type="session_reset",
            risk_band=_risk_band(current_prob),
            probability_synthetic=current_prob,
            flagged=session._consecutive_flags >= session.consecutive_flags_required,
        )
        session.reset()
    else:
        session = create_session()
    return session, 0.0, "False", "N/A"


def analyze_uploaded_file(
    audio_path: str | None,
) -> tuple[str, str]:
    """Run whole-clip and streaming-simulation analysis on an uploaded audio file."""
    if audio_path is None:
        return "Please upload an audio file first.", ""

    waveform, sr = load_audio(audio_path, target_sr=16_000)
    duration = float(len(waveform)) / float(sr)

    whole_clip = get_detector().predict_waveform(waveform, sr)
    label = str(whole_clip.get("label", "unknown"))
    probability = whole_clip.get("probability_synthetic")

    if label == "inconclusive" or probability is None:
        whole_clip_md = (
            f"**Whole-clip verdict:** {label}  "
            f"\\\n**Reason:** Input audio is near-silent; the classifier was not "
            f"trained to score non-speech audio, so no meaningful synthetic-voice "
            f"probability can be produced."
        )
        _SESSION_LOGGER.log_event(
            event_type="upload_analysis",
            risk_band="inconclusive",
            probability_synthetic=0.0,
            flagged=False,
        )
    else:
        probability = float(probability)
        whole_clip_md = (
            f"**Whole-clip verdict:** {label}  "
            f"\\\n**Probability synthetic:** {probability:.4f}  "
            f"\\\n**Duration:** {duration:.2f}s"
        )
        _SESSION_LOGGER.log_event(
            event_type="upload_analysis",
            risk_band=_risk_band(probability),
            probability_synthetic=probability,
            flagged=probability >= get_detector().threshold,
        )

    session = create_session()
    summary = simulate_stream(
        audio=waveform,
        session=session,
        real_time_paced=False,
    )
    stream_flagged = bool(summary.get("flagged", False))
    stream_s2f = summary.get("seconds_to_flag")
    stream_s2f_str = f"{stream_s2f:.2f}s" if stream_s2f is not None else "N/A"

    stream_md = (
        f"**Streaming simulation flagged:** {stream_flagged}  "
        f"\\\n**Seconds to flag:** {stream_s2f_str}"
    )

    return whole_clip_md, stream_md


def build_app() -> gr.Blocks:
    """Builds the Gradio UI shell for the VoxGuard demo app."""
    with gr.Blocks(title="VoxGuard — Voice Cloning Detection & Prevention") as demo:
        gr.Markdown("# VoxGuard — Voice Cloning Detection & Prevention")
        gr.Markdown(
            "VoxGuard is a hackathon prototype that detects AI-cloned voice in call audio "
            "using dual embedding backbones and streaming risk scoring. This demo simulates "
            "call audio via live microphone input or uploaded files — it does **not** intercept "
            "real telecom traffic. All processing runs locally on your machine."
        )

        with gr.Tabs():
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
                        running_score_out = gr.Number(
                            label="Running Risk Score",
                            value=0.0,
                            precision=4,
                        )
                        flagged_out = gr.Textbox(
                            label="Flagged (Synthetic Voice Detected)",
                            value="False",
                        )
                        seconds_to_flag_out = gr.Textbox(
                            label="Seconds to Flag",
                            value="N/A",
                        )

                mic_input.stream(
                    fn=process_audio_chunk,
                    inputs=[mic_input, session_state],
                    outputs=[
                        session_state,
                        running_score_out,
                        flagged_out,
                        seconds_to_flag_out,
                    ],
                )

                reset_btn.click(
                    fn=reset_streaming_session,
                    inputs=[session_state],
                    outputs=[
                        session_state,
                        running_score_out,
                        flagged_out,
                        seconds_to_flag_out,
                    ],
                )

            with gr.Tab("Upload File"):
                gr.Markdown(
                    "Upload an audio file to run both whole-clip detection and "
                    "a fast streaming-simulation replay."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        upload_audio = gr.Audio(
                            sources=["upload"],
                            streaming=False,
                            type="filepath",
                            label="Upload Audio File",
                        )
                        analyze_btn = gr.Button("Analyze", variant="primary")

                    with gr.Column(scale=1):
                        upload_result_whole = gr.Markdown(label="Whole-Clip Verdict")
                        upload_result_stream = gr.Markdown(label="Streaming Simulation")

                analyze_btn.click(
                    fn=analyze_uploaded_file,
                    inputs=[upload_audio],
                    outputs=[
                        upload_result_whole,
                        upload_result_stream,
                    ],
                )

    return demo


app = build_app()


if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", show_api=False)
