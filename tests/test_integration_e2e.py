"""
test_integration_e2e.py — end-to-end cross-phase pipeline smoke test (Prompt 11.2).

Exercises the real pipeline VoxGuard's app wires together — calling each
phase's underlying function directly (no Gradio server, no HTTP) against a
small set of known sample files — and asserts every step returns a
well-formed result (right type, right value range).

This is deliberately NOT a re-validation of any phase's own accuracy/EER
numbers; Phases 2-9 each already have their own eval scripts for that (see
models/reports/*.md). This file only proves the pieces are still wired
together correctly after later phases' changes.

Scope note: the REST API gets its own dedicated test file
(tests/test_api.py, Prompt 11.3) — this file does not duplicate that
coverage; nothing here goes through HTTP.

Phase 2/3 note: this project's Phase 2 decision
(models/reports/decision_notes.md) selected the BASELINE (non-prosody
-augmented) feature set, not the prosody-augmented variant — so there is no
prosody branch to exercise here. That decision is a fact about this
project's classifier, not something this test re-derives.

Detector configuration: the fixtures below use
wav2vec2_hindi_combined_logreg + wavlm_hindi_combined_logreg — the exact
production configuration used by both app.app.get_detector() and
StreamingSession's own default detector — so "whole-clip", "streaming",
and "explainability overlay" below all exercise the one detector instance
actually shipped, not a stand-in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voxguard import config
from voxguard.classifier.ensemble import WeightedAverageDetector
from voxguard.explain import describe_attribution, render_explainability_overlay, windowed_attribution
from voxguard.fusion.fuse import fuse_risk_with_context
from voxguard.fusion.redflags import scan_for_redflags
from voxguard.fusion.transcribe import LiveTranscriber
from voxguard.risk.bands import score_to_band
from voxguard.risk.prevention import get_prevention_message
from voxguard.speaker.embedding import SpeakerEmbedder
from voxguard.speaker.enrollment import enroll_speaker, list_enrolled_speakers
from voxguard.speaker.verify import verify_speaker
from voxguard.streaming.session import StreamingSession, simulate_stream
from voxguard.utils.audio_io import load_audio

# ---------------------------------------------------------------------------
# Fixed sample data — known files, not synthesized on the fly, so results
# are reproducible run to run.
# ---------------------------------------------------------------------------
SAMPLE_CLIP = Path("data/raw/hindi_hinglish/real/byaquta_scam_11.wav")
VERIFY_CLIP = Path("data/raw/hindi_hinglish/real/byaquta_neutral_02.wav")  # deliberately not SAMPLE_CLIP
ENROLLED_NAME = "byaquta"
NON_DEFAULT_TRANSACTION_CONTEXT = "otp_request"

# byaquta's canonical enrollment clips (same 3 used throughout Phase 8's
# calibration work) — used only as a fallback if the enrollment is missing.
# This project's models/voiceprints/ store has been accidentally wiped by
# test runs twice before, so tests must never assume it's present.
_CANONICAL_ENROLLMENT_CLIPS = [
    Path("data/raw/hindi_hinglish/references/byaquta_ref.wav"),
    Path("data/raw/hindi_hinglish/real/byaquta_neutral_01.wav"),
    Path("data/raw/hindi_hinglish/real/byaquta_scam_11.wav"),
]

_PRODUCTION_WAV2VEC2_CLASSIFIER = "models/classifiers/wav2vec2_hindi_combined_logreg.joblib"
_PRODUCTION_WAVLM_CLASSIFIER = "models/classifiers/wavlm_hindi_combined_logreg.joblib"


def _ensure_byaquta_enrolled(embedder: SpeakerEmbedder) -> None:
    """Re-enrolls ENROLLED_NAME from its canonical clips if not already enrolled.

    Checks membership, not just "is the store non-empty" — a non-empty
    store missing this specific speaker is just as broken for this test.
    """
    if ENROLLED_NAME in list_enrolled_speakers():
        return
    clips = [str(p) for p in _CANONICAL_ENROLLMENT_CLIPS if p.exists()]
    assert clips, (
        f"Cannot re-enroll '{ENROLLED_NAME}': none of its canonical reference "
        f"clips exist on disk ({_CANONICAL_ENROLLMENT_CLIPS})."
    )
    enroll_speaker(ENROLLED_NAME, clips, embedder)
    assert ENROLLED_NAME in list_enrolled_speakers(), (
        f"Re-enrollment of '{ENROLLED_NAME}' did not take effect."
    )


# ---------------------------------------------------------------------------
# Shared fixtures — module-scoped so expensive model loads (embedding
# backbones, whisper, speaker embedder) happen once for the whole sequence.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def detector() -> WeightedAverageDetector:
    """The production detector configuration (mirrors app.app.get_detector())."""
    return WeightedAverageDetector(
        wav2vec2_classifier_path=_PRODUCTION_WAV2VEC2_CLASSIFIER,
        wavlm_classifier_path=_PRODUCTION_WAVLM_CLASSIFIER,
    )


@pytest.fixture(scope="module")
def speaker_embedder() -> SpeakerEmbedder:
    return SpeakerEmbedder()


@pytest.fixture(scope="module")
def transcriber() -> LiveTranscriber:
    return LiveTranscriber(model_size="base")


@pytest.fixture(scope="module")
def sample_waveform() -> tuple[np.ndarray, int]:
    assert SAMPLE_CLIP.exists(), f"Missing known sample file: {SAMPLE_CLIP}"
    return load_audio(SAMPLE_CLIP, target_sr=config.SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Phase 2/3 — whole-clip detection
# ---------------------------------------------------------------------------

def test_whole_clip_detection(detector, sample_waveform):
    waveform, sr = sample_waveform
    result = detector.predict_waveform(waveform, sr)

    assert isinstance(result, dict)
    assert result["label"] in ("real", "synthetic")
    probability = result["probability_synthetic"]
    assert isinstance(probability, float)
    assert 0.0 <= probability <= 1.0


# ---------------------------------------------------------------------------
# Phase 5 — streaming simulation with flag timing
# ---------------------------------------------------------------------------

def test_streaming_simulation(detector, sample_waveform):
    waveform, sr = sample_waveform
    session = StreamingSession(detector=detector, sample_rate=sr)
    summary = simulate_stream(audio=waveform, session=session, real_time_paced=False, sr=sr)

    assert isinstance(summary, dict)
    assert set(summary.keys()) >= {
        "total_duration",
        "final_running_score",
        "flagged",
        "seconds_to_flag",
        "step_results",
    }

    assert isinstance(summary["total_duration"], float)
    assert summary["total_duration"] > 0.0

    running_score = summary["final_running_score"]
    assert isinstance(running_score, float)
    assert 0.0 <= running_score <= 1.0

    assert isinstance(summary["flagged"], bool)

    seconds_to_flag = summary["seconds_to_flag"]
    assert seconds_to_flag is None or (
        isinstance(seconds_to_flag, (int, float)) and seconds_to_flag >= 0.0
    )

    assert isinstance(summary["step_results"], list)
    assert len(summary["step_results"]) > 0
    for step in summary["step_results"]:
        assert isinstance(step, dict)
        assert 0.0 <= float(step["running_score"]) <= 1.0


# ---------------------------------------------------------------------------
# Phase 7 — risk banding + prevention prompt
# ---------------------------------------------------------------------------

def test_risk_band_and_prevention(detector, sample_waveform):
    waveform, sr = sample_waveform
    probability = detector.predict_waveform(waveform, sr)["probability_synthetic"]

    band = score_to_band(probability)
    assert band in ("low", "medium", "high", "inconclusive")

    message = get_prevention_message(band)
    if band in ("medium", "high"):
        assert isinstance(message, str) and message.strip()
    else:
        assert message is None

    # The sample clip's own score only ever lands in one band per run; also
    # check the return-shape contract for every band directly (still just
    # checking well-formedness, not re-deriving Phase 7's thresholds).
    assert get_prevention_message("low") is None
    assert get_prevention_message("inconclusive") is None
    assert isinstance(get_prevention_message("medium"), str) and get_prevention_message("medium").strip()
    assert isinstance(get_prevention_message("high"), str) and get_prevention_message("high").strip()


# ---------------------------------------------------------------------------
# Phase 9 — transcription + red-flag scan + CONTEXTUAL fusion
# (non-default transaction context + a real voiceprint result)
# ---------------------------------------------------------------------------

def test_transcription_redflag_and_contextual_fusion(
    detector, transcriber, sample_waveform, speaker_embedder
):
    waveform, sr = sample_waveform

    # 1. Transcription (Phase 9) — well-formedness only; byaquta_scam_11.wav
    # is code-switched Hinglish and whisper may mis-transcribe it, which is
    # fine here — this is not a transcription-accuracy test.
    text = transcriber.transcribe_full(str(SAMPLE_CLIP))
    assert isinstance(text, str)

    # 2. Red-flag keyword scan (Phase 9)
    redflags = scan_for_redflags(text)
    assert isinstance(redflags, dict)
    assert {"matched_phrases", "categories", "keyword_risk_score"} <= redflags.keys()
    assert isinstance(redflags["matched_phrases"], list)
    assert isinstance(redflags["categories"], list)
    keyword_score = redflags["keyword_risk_score"]
    assert isinstance(keyword_score, float)
    assert 0.0 <= keyword_score <= 1.0

    # 3. A real voiceprint result (Phase 8) to feed into fusion.
    _ensure_byaquta_enrolled(speaker_embedder)
    voiceprint_result = verify_speaker(waveform, sr, ENROLLED_NAME, speaker_embedder)
    assert isinstance(voiceprint_result, dict) and "match" in voiceprint_result

    # 4. Contextual fusion (Phase 9) with a NON-DEFAULT transaction context
    # and the voiceprint result above.
    audio_score = detector.predict_waveform(waveform, sr)["probability_synthetic"]
    fusion = fuse_risk_with_context(
        audio_score=audio_score,
        keyword_risk_score=keyword_score,
        transaction_context=NON_DEFAULT_TRANSACTION_CONTEXT,
        voiceprint_result=voiceprint_result,
    )
    assert isinstance(fusion, dict)
    for key in ("base_fused_score", "contextual_score", "transaction_multiplier", "contact_multiplier"):
        assert key in fusion
        assert isinstance(fusion[key], float)

    assert 0.0 <= fusion["base_fused_score"] <= 1.0
    assert 0.0 <= fusion["contextual_score"] <= 1.0

    # otp_request must NOT resolve to the neutral 1.0x general_conversation
    # multiplier -- otherwise this "non-default context" assertion would
    # pass even if the context argument were silently ignored.
    assert fusion["transaction_multiplier"] != 1.0
    # Likewise, a real match/mismatch result must not resolve to the neutral
    # "no_enrollment_data" 1.0x multiplier.
    assert fusion["contact_multiplier"] != 1.0


# ---------------------------------------------------------------------------
# Phase 8 — speaker verification for an enrolled name
# ---------------------------------------------------------------------------

def test_speaker_verification(speaker_embedder):
    _ensure_byaquta_enrolled(speaker_embedder)
    assert VERIFY_CLIP.exists(), f"Missing known verification file: {VERIFY_CLIP}"
    waveform, sr = load_audio(VERIFY_CLIP, target_sr=config.SAMPLE_RATE)

    result = verify_speaker(waveform, sr, ENROLLED_NAME, speaker_embedder)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"match", "similarity", "enrolled_name"}
    assert isinstance(result["match"], bool)
    similarity = result["similarity"]
    assert isinstance(similarity, float)
    assert -1.0 <= similarity <= 1.0
    assert result["enrolled_name"] == ENROLLED_NAME


# ---------------------------------------------------------------------------
# Phase 10 — explainability overlay generation
# ---------------------------------------------------------------------------

def test_explainability_overlay(detector, sample_waveform, tmp_path):
    waveform, sr = sample_waveform
    out_path = tmp_path / "integration_overlay.png"

    saved_path = render_explainability_overlay(
        waveform=waveform, sr=sr, detector=detector, output_path=out_path,
    )
    saved = Path(saved_path)
    assert saved.exists()
    assert saved.suffix == ".png"
    assert saved.stat().st_size > 0

    scores, timestamps = windowed_attribution(waveform=waveform, sr=sr, detector=detector)
    assert isinstance(scores, np.ndarray)
    assert isinstance(timestamps, np.ndarray)
    assert scores.shape == timestamps.shape
    assert scores.size > 0

    prediction = detector.predict_waveform(waveform, sr)
    description = describe_attribution(
        scores=scores, timestamps=timestamps, label=str(prediction["label"])
    )
    assert isinstance(description, str) and description.strip()
