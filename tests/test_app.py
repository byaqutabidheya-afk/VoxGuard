from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import gradio as gr

from app.app import (
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

    session, score, flagged, s2f = process_audio_chunk(None, mock_session)
    assert session is mock_session
    assert score == 0.42
    assert flagged == "False"
    assert s2f == "N/A"

    session, score, flagged, s2f = process_audio_chunk((16000, np.array([])), mock_session)
    assert session is mock_session
    assert score == 0.42
    assert flagged == "False"
    assert s2f == "N/A"


def test_process_audio_chunk_real_and_int16_conversion() -> None:
    mock_session = MagicMock()
    mock_session.push_audio.return_value = {
        "running_score": 0.8523,
        "flagged": True,
        "seconds_since_start": 2.5,
        "seconds_to_flag": 2.0,
    }

    int16_stereo = np.ones((1600, 2), dtype=np.int16) * 16384
    session, score, flagged, s2f = process_audio_chunk((16000, int16_stereo), mock_session)

    assert session is mock_session
    assert score == 0.8523
    assert flagged == "True"
    assert s2f == "2.00s"
    assert mock_session.push_audio.called

    call_args = mock_session.push_audio.call_args
    pushed_frame = call_args.args[0]
    pushed_sr = call_args.kwargs.get("sr", call_args.args[1] if len(call_args.args) > 1 else None)

    assert pushed_frame.ndim == 1
    assert pushed_frame.dtype == np.float32
    assert pushed_sr == 16000
    assert float(pushed_frame[0]) == pytest.approx(0.5, rel=1e-3)


def test_reset_streaming_session() -> None:
    mock_session = MagicMock()
    mock_session._consecutive_flags = 0
    mock_session.consecutive_flags_required = 3
    session, score, flagged, s2f = reset_streaming_session(mock_session)

    assert session is mock_session
    mock_session.reset.assert_called_once()
    assert score == 0.0
    assert flagged == "False"
    assert s2f == "N/A"


@patch("app.app.WeightedAverageDetector")
def test_create_session_uses_detector(mock_detector_cls: MagicMock) -> None:
    import app.app as app_module
    app_module._DETECTOR = None

    session = create_session()
    assert isinstance(session, StreamingSession)
    assert app_module._DETECTOR is not None
