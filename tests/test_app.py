from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import gradio as gr

from app.app import (
    analyze_uploaded_file,
    build_app,
    create_session,
    get_detector,
    process_audio_chunk,
    reset_streaming_session,
)
from voxguard.streaming.session import StreamingSession


@patch("app.app.get_detector")
def test_build_app_structure(mock_get_det: MagicMock) -> None:
    demo = build_app()
    assert isinstance(demo, gr.Blocks)
    assert demo.title == "VoxGuard — Voice Cloning Detection & Prevention"


def test_process_audio_chunk_none_and_empty() -> None:
    mock_session = MagicMock()
    mock_session.risk_score.current.return_value = 0.42
    mock_session._consecutive_flags = 1
    mock_session.consecutive_flags_required = 3
    mock_session._seconds_to_flag = None
    mock_session.buffer.sample_rate = 16000
    mock_session.buffer.chunk_samples = 24000
    mock_session.buffer.stride_samples = 16000

    session, risk_html, prevention_html, transcript_html, flagged, s2f, transcript_state = (
        process_audio_chunk(None, mock_session)
    )
    assert session is mock_session
    assert "LOW RISK" in risk_html
    assert prevention_html == ""
    assert "No speech transcribed yet" in transcript_html
    assert flagged == "False"
    assert s2f == "N/A"
    assert transcript_state == ""

    session, risk_html, prevention_html, transcript_html, flagged, s2f, transcript_state = (
        process_audio_chunk((16000, np.array([])), mock_session)
    )
    assert session is mock_session
    assert "LOW RISK" in risk_html
    assert prevention_html == ""
    assert flagged == "False"
    assert s2f == "N/A"
    assert transcript_state == ""


@patch("app.app.get_transcriber")
def test_process_audio_chunk_real_and_fusion(mock_get_trans: MagicMock) -> None:
    mock_transcriber = MagicMock()
    mock_transcriber.transcribe_chunk.return_value = "customs department fine pay kijiye"
    mock_get_trans.return_value = mock_transcriber

    mock_session = MagicMock()
    mock_session.buffer.sample_rate = 16000
    mock_session.buffer.chunk_samples = 24000
    mock_session.buffer.stride_samples = 16000
    mock_session.push_audio.return_value = {
        "running_score": 0.8523,
        "flagged": True,
        "seconds_since_start": 2.5,
        "seconds_to_flag": 2.0,
    }

    int16_stereo = np.ones((1600, 2), dtype=np.int16) * 16384
    session, risk_html, prevention_html, transcript_html, flagged, s2f, transcript_state = (
        process_audio_chunk(
            (16000, int16_stereo),
            mock_session,
            tx_context="fund_transfer",
            last_voiceprint_result={"match": False, "similarity": 0.2, "enrolled_name": "byaquta"},
            transcript_state="abhi turant",
        )
    )

    assert session is mock_session
    assert "HIGH RISK" in risk_html
    assert "High-confidence alert" in prevention_html
    assert "customs department" in transcript_html
    assert flagged == "True"
    assert s2f == "2.00s"
    assert mock_session.push_audio.called
    assert "abhi turant customs department" in transcript_state


def test_reset_streaming_session() -> None:
    mock_session = MagicMock()
    mock_session._consecutive_flags = 0
    mock_session.consecutive_flags_required = 3
    session, risk_html, prevention_html, transcript_html, flagged, s2f, transcript_state = (
        reset_streaming_session(mock_session)
    )

    assert session is mock_session
    mock_session.reset.assert_called_once()
    assert "LOW RISK" in risk_html
    assert prevention_html == ""
    assert "No speech transcribed yet" in transcript_html
    assert flagged == "False"
    assert s2f == "N/A"
    assert transcript_state == ""


@patch("app.app.WeightedAverageDetector")
def test_create_session_uses_detector(mock_detector_cls: MagicMock) -> None:
    import app.app as app_module

    app_module._DETECTOR = None

    session = create_session()
    assert isinstance(session, StreamingSession)
    assert app_module._DETECTOR is not None
