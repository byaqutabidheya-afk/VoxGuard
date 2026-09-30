from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import gradio as gr

from app.app import (
    _render_attribution_explanation_html,
    analyze_uploaded_file,
    build_app,
    create_session,
    generate_overlay,
    get_detector,
    process_audio_chunk,
    reset_streaming_session,
)
from voxguard import config
from voxguard.streaming.session import StreamingSession

# Mocked probabilities derived from the LIVE config thresholds (never hardcoded), so recalibrating
# RISK_THRESHOLDS cannot silently move them into the wrong band.
_LOW_SCORE = float(config.RISK_THRESHOLDS["low_max"]) / 2.0                      # well inside "low"
_HIGH_SCORE = (float(config.RISK_THRESHOLDS["medium_max"]) + 1.0) / 2.0          # well inside "high"


@patch("app.app.get_detector")
def test_build_app_structure(mock_get_det: MagicMock) -> None:
    demo = build_app()
    assert isinstance(demo, gr.Blocks)
    assert demo.title == "VoxGuard — Voice Cloning Detection & Prevention"


def test_process_audio_chunk_none_and_empty() -> None:
    mock_session = MagicMock()
    mock_session.risk_score.current.return_value = _LOW_SCORE
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
        "running_score": _HIGH_SCORE,
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


@patch("voxguard.streaming.session.get_production_detector")
def test_create_session_uses_streaming_detector_and_calibrated_rule(mock_get_prod: MagicMock) -> None:
    """Live sessions take the STREAMING detector (StreamingSession's default), never the whole-clip one,
    and run the calibrated flag rule with the unit it was calibrated in."""
    import app.app as app_module

    app_module._DETECTOR = None
    mock_get_prod.return_value = MagicMock()

    session = create_session()
    assert isinstance(session, StreamingSession)
    mock_get_prod.assert_called_once_with("streaming")
    assert session.detector is mock_get_prod.return_value
    assert app_module._DETECTOR is None            # the whole-clip detector was not built for a live session
    assert session.flag_threshold == float(config.STREAM_FLAG_THRESHOLD)
    assert session.consecutive_flags_required == config.STREAM_CONSECUTIVE_FLAGS_REQUIRED
    assert session.consecutive_unit == config.STREAM_CONSECUTIVE_UNIT


@patch("app.app.build_production_detector")
def test_get_detector_is_the_wholeclip_production_detector(mock_build: MagicMock) -> None:
    import app.app as app_module

    app_module._DETECTOR = None
    mock_build.return_value = MagicMock()
    try:
        assert get_detector() is mock_build.return_value
        assert get_detector() is mock_build.return_value      # cached: built once
        mock_build.assert_called_once_with("wholeclip")
    finally:
        app_module._DETECTOR = None


def test_build_app_verifies_production_classifiers_once_at_startup() -> None:
    with patch("app.app.verify_production_classifiers") as mock_verify, patch("app.app.get_detector"):
        build_app()
    mock_verify.assert_called_once_with()


def test_build_app_fails_loudly_when_a_production_file_is_missing() -> None:
    with patch("app.app.verify_production_classifiers", side_effect=FileNotFoundError("missing: x.joblib")):
        with pytest.raises(FileNotFoundError, match="x.joblib"):
            build_app()


def test_render_attribution_explanation_html() -> None:
    assert _render_attribution_explanation_html("") == ""
    assert _render_attribution_explanation_html("   ") == ""
    html_out = _render_attribution_explanation_html("This is a test <attribution> text.")
    assert "Attribution Analysis" in html_out
    assert "&lt;attribution&gt;" in html_out


def test_generate_overlay_none() -> None:
    img, status, expl = generate_overlay(None)
    assert img is None
    assert "Analyze a clip first" in status
    assert expl == ""


@patch("app.app.load_audio")
def test_generate_overlay_load_error(mock_load: MagicMock) -> None:
    mock_load.side_effect = RuntimeError("Failed to decode")
    img, status, expl = generate_overlay("bad_file.wav")
    assert img is None
    assert "Could not load audio" in status
    assert expl == ""


@patch("app.app.load_audio")
@patch("app.app.get_detector")
@patch("app.app.render_explainability_overlay")
@patch("app.app.windowed_attribution")
def test_generate_overlay_success(
    mock_attribution: MagicMock,
    mock_render: MagicMock,
    mock_get_detector: MagicMock,
    mock_load: MagicMock,
) -> None:
    mock_load.return_value = (np.zeros(32000, dtype=np.float32), 16000)
    mock_detector = MagicMock()
    mock_detector.predict_waveform.return_value = {"label": "synthetic", "probability_synthetic": 0.88}
    mock_get_detector.return_value = mock_detector
    mock_render.return_value = "data/processed/overlays/sample_overlay.png"
    mock_attribution.return_value = (
        np.array([0.85, 0.90, 0.88]),
        np.array([0.0, 0.75, 1.5]),
    )

    img, status, expl = generate_overlay("sample.wav")
    assert img == "data/processed/overlays/sample_overlay.png"
    assert "Overlay generated from: <code>sample.wav</code>" in status
    assert "Attribution Analysis" in expl
    assert "classified as synthetic" in expl
    assert "average synthetic-likelihood of 88%" in expl



# ---------------------------------------------------------------------------
# Model provenance in the UI (F5.3): every label is generated from config, never hardcoded
# ---------------------------------------------------------------------------

import app.app as _app_module  # noqa: E402


@pytest.mark.parametrize(
    "backbone,path,expected",
    [
        ("wav2vec2", "models/classifiers/wav2vec2_hindi_matched_logreg.joblib", "hindi_matched"),
        ("wavlm", "models/classifiers/wavlm_hindi_matched_v2_logreg.joblib", "hindi_matched_v2"),
        ("wavlm", "models/classifiers/wavlm_chunked_v2_logreg.joblib", "chunked_v2"),
        ("wav2vec2", "models/classifiers/other_name.joblib", "other_name"),
    ],
)
def test_head_id_strips_backbone_prefix_and_logreg_suffix(backbone: str, path: str, expected: str) -> None:
    assert _app_module._head_id(backbone, path) == expected


def test_provenance_line_is_generated_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "PRODUCTION_WHOLECLIP_CLASSIFIERS", {
        "wav2vec2": "m/wav2vec2_zzz_logreg.joblib", "wavlm": "m/wavlm_yyy_v9_logreg.joblib"})
    monkeypatch.setattr(config, "PRODUCTION_STREAMING_CLASSIFIERS", {
        "wav2vec2": "m/wav2vec2_sss_logreg.joblib", "wavlm": "m/wavlm_ttt_v3_logreg.joblib"})
    monkeypatch.setattr(config, "PRODUCTION_ENSEMBLE_WEIGHT_A", 0.3)
    monkeypatch.setattr(config, "RISK_THRESHOLDS", {"low_max": 0.25, "medium_max": 0.65})
    monkeypatch.setattr(config, "STREAM_FLAG_THRESHOLD", 0.7)
    monkeypatch.setattr(config, "STREAM_CONSECUTIVE_FLAGS_REQUIRED", 3)
    monkeypatch.setattr(config, "STREAM_CONSECUTIVE_UNIT", "pushes")
    monkeypatch.setattr(config, "STREAM_CHUNK_SECONDS", 2.0)

    line = _app_module._provenance_html()
    for fragment in ("whole-clip: wav2vec2 zzz + WavLM yyy_v9", "streaming: wav2vec2 sss + WavLM ttt_v3",
                     "2 s windows", "ensemble weight 0.3/0.7", "risk bands 0.25/0.65", "0.7 for 3 consecutive pushes"):
        assert fragment in line, fragment
    assert "hindi_matched" not in line


def test_provenance_line_reads_the_live_config_values() -> None:
    line = _app_module._provenance_html()
    assert f"risk bands {config.RISK_THRESHOLDS['low_max']:g}/{config.RISK_THRESHOLDS['medium_max']:g}" in line
    assert f"{config.STREAM_FLAG_THRESHOLD:g} for {config.STREAM_CONSECUTIVE_FLAGS_REQUIRED} consecutive" in line
    for mapping in (config.PRODUCTION_WHOLECLIP_CLASSIFIERS, config.PRODUCTION_STREAMING_CLASSIFIERS):
        for backbone, rel in mapping.items():
            assert _app_module._head_id(backbone, rel) in line


def test_ui_source_names_no_classifier_file_as_shipped() -> None:
    """The stale 'hindi_combined' claims are gone and cannot creep back: the UI gets model names from config."""
    source = Path(_app_module.__file__).read_text(encoding="utf-8")
    assert "hindi_combined" not in source


def test_risk_card_shows_model_line_and_only_the_leading_label_makes_it_live() -> None:
    card = _app_module._risk_html(0.3, context="Whole-Clip · 3.30s", model="whole-clip · wav2vec2 a + WavLM b")
    assert "Model: whole-clip · wav2vec2 a + WavLM b" in card
    assert "vg-card-live" not in card
    tricky = _app_module._risk_html(0.3, context="Whole-Clip · 1s", model="live streaming x")
    assert "vg-card-live" not in tricky                      # model text must not flip the live animation
    assert "vg-card-live" in _app_module._risk_html(0.3, context="Streaming Simulation · flagged=False", model="m")
    assert "vg-card-live" in _app_module._risk_html(0.3, context="Live Streaming")
    assert "Model:" not in _app_module._risk_html(0.3, context="Live Streaming")          # no model -> no line


def test_risk_card_escapes_the_model_label() -> None:
    assert "<script>" not in _app_module._risk_html(0.3, context="c", model="<script>x</script>")


@patch("app.app.simulate_stream")
@patch("app.app.create_session")
@patch("app.app.get_detector")
@patch("app.app.get_transcriber")
@patch("app.app.load_audio")
def test_upload_verdicts_are_labelled_with_their_own_model_family(
    mock_load: MagicMock, mock_trans: MagicMock, mock_get_det: MagicMock,
    mock_create: MagicMock, mock_sim: MagicMock,
) -> None:
    mock_load.return_value = (np.zeros(32000, dtype=np.float32), 16000)
    mock_trans.return_value.transcribe_full.return_value = ""
    mock_get_det.return_value.predict_waveform.return_value = {"probability_synthetic": 0.4, "label": "real"}
    mock_sim.return_value = {"final_running_score": 0.3, "flagged": False, "seconds_to_flag": None}

    whole_risk, _wp, stream_risk, _sp, _tr, _path = analyze_uploaded_file("clip.wav")

    whole_label = _app_module._wholeclip_model_label()
    stream_label = _app_module._streaming_model_label()
    assert f"Model: {whole_label}" in whole_risk and stream_label not in whole_risk
    assert f"Model: {stream_label}" in stream_risk and whole_label not in stream_risk
    assert whole_label != stream_label                        # genuinely different models


def test_divergence_note_explains_the_two_families_from_config() -> None:
    note = _app_module._divergence_note()
    assert _app_module._family_heads(config.PRODUCTION_WHOLECLIP_CLASSIFIERS) in note
    assert _app_module._family_heads(config.PRODUCTION_STREAMING_CLASSIFIERS) in note
    assert "different model families" in note


# ---------------------------------------------------------------------------
# Cross-Dataset Results tab: BEFORE FIX / AFTER FIX registry
# ---------------------------------------------------------------------------


def _names(stage: str) -> list[str]:
    return [p.name for s, _t, _i, p in _app_module._CROSS_DATASET_SOURCES if s == stage]


def test_cross_dataset_tab_keeps_the_originals_as_before_fix() -> None:
    assert _names("BEFORE FIX") == ["generalization_before_after.md", "hindi_training_comparison.md"]


def test_cross_dataset_tab_shows_the_new_reports_as_after_fix() -> None:
    assert _names("AFTER FIX") == ["fix_matched_comparison.md", "fix_chunked_eval.md"]


def test_cross_dataset_tab_does_not_reference_a_report_that_does_not_exist_yet() -> None:
    assert all("fix_comparison_report" not in str(p) for *_x, p in _app_module._CROSS_DATASET_SOURCES)
    source = Path(_app_module.__file__).read_text(encoding="utf-8")
    assert "fix_comparison_report" not in source


def test_cross_dataset_stage_notes_are_non_empty_and_flag_the_v1_heads_caveat() -> None:
    assert _app_module._cross_dataset_stage_note("BEFORE FIX").strip()
    after = _app_module._cross_dataset_stage_note("AFTER FIX")
    assert "F3 WavLM retrain" in after
    assert _app_module._head_id("wavlm", config.PRODUCTION_WHOLECLIP_CLASSIFIERS["wavlm"]) in after


def test_build_app_renders_every_registered_report_and_a_new_row_needs_no_other_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Appending one row to the registry (a future third AFTER FIX report) renders it, with no other edit."""
    extra = tmp_path / "future_after_fix_report.md"
    extra.write_text("FUTURE REPORT BODY", encoding="utf-8")
    monkeypatch.setattr(
        _app_module, "_CROSS_DATASET_SOURCES",
        [*_app_module._CROSS_DATASET_SOURCES, ("AFTER FIX", "Future report", "summarize", extra)],
    )
    loaded: list[Path] = []
    real_loader = _app_module._load_report_markdown

    def recording_loader(path: Path) -> str:
        loaded.append(path)
        return real_loader(path)

    monkeypatch.setattr(_app_module, "_load_report_markdown", recording_loader)
    with patch("app.app.verify_production_classifiers"), patch("app.app.get_detector"):
        build_app()
    assert loaded == [p for *_x, p in _app_module._CROSS_DATASET_SOURCES]
    assert extra in loaded
