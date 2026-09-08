"""
test_api.py - unit and integration tests for VoxGuard FastAPI REST API.

Tests endpoints:
  - GET /health
  - POST /analyze (including 415 content-type and 413 file-size error handling)
  - POST /verify-speaker (including 404 for non-enrolled speakers)
  - POST /analyze-context (with and without optional voiceprint verification)
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from api.main import MAX_FILE_SIZE_BYTES, app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Creates a FastAPI TestClient instance."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_wav_bytes() -> bytes:
    """Generates a small valid WAV byte string for testing."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    waveform = 0.25 * np.sin(2 * np.pi * 440.0 * t)
    buf = io.BytesIO()
    sf.write(buf, waveform, sr, format="WAV")
    return buf.getvalue()


SAMPLE_REAL_CLIP = Path("data/raw/hindi_hinglish/real/byaquta_scam_11.wav")


# ---------------------------------------------------------------------------
# 1. GET /health
# ---------------------------------------------------------------------------


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# 2. POST /analyze
# ---------------------------------------------------------------------------


def test_analyze_endpoint_mocked(client: TestClient, sample_wav_bytes: bytes) -> None:
    with patch("api.main.get_detector") as mock_get_det:
        mock_detector = MagicMock()
        mock_detector.predict_waveform.return_value = {
            "label": "real",
            "probability_synthetic": 0.08,
        }
        mock_get_det.return_value = mock_detector

        response = client.post(
            "/analyze",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == "real"
        assert data["probability_synthetic"] == pytest.approx(0.08, abs=1e-4)
        assert data["risk_band"] == "low"
        assert data["prevention_message"] is None


def test_analyze_endpoint_high_risk_prevention_message(
    client: TestClient, sample_wav_bytes: bytes
) -> None:
    with patch("api.main.get_detector") as mock_get_det:
        mock_detector = MagicMock()
        mock_detector.predict_waveform.return_value = {
            "label": "synthetic",
            "probability_synthetic": 0.92,
        }
        mock_get_det.return_value = mock_detector

        response = client.post(
            "/analyze",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == "synthetic"
        assert data["probability_synthetic"] == pytest.approx(0.92, abs=1e-4)
        assert data["risk_band"] == "high"
        assert data["prevention_message"] is not None
        assert "High-confidence alert" in data["prevention_message"]


def test_analyze_endpoint_unsupported_media_type_415(client: TestClient) -> None:
    txt_content = b"This is a text file, not audio."
    response = client.post(
        "/analyze",
        files={"file": ("document.txt", txt_content, "text/plain")},
    )
    assert response.status_code == 415
    assert "Unsupported media type" in response.json()["detail"]


def test_analyze_endpoint_file_too_large_413(client: TestClient) -> None:
    # 26 MB dummy payload
    oversized_content = b"RIFF" + b"\x00" * (MAX_FILE_SIZE_BYTES + 1024)
    response = client.post(
        "/analyze",
        files={"file": ("huge.wav", oversized_content, "audio/wav")},
    )
    assert response.status_code == 413
    assert "exceeds maximum allowable size" in response.json()["detail"]


def test_analyze_endpoint_empty_file_400(client: TestClient) -> None:
    response = client.post(
        "/analyze",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 3. POST /verify-speaker
# ---------------------------------------------------------------------------


def test_verify_speaker_endpoint_success(
    client: TestClient, sample_wav_bytes: bytes
) -> None:
    with (
        patch("api.main.list_enrolled_speakers", return_value=["byaquta", "priya"]),
        patch("api.main.verify_speaker") as mock_verify,
        patch("api.main.get_speaker_embedder") as mock_embedder,
    ):
        mock_verify.return_value = {
            "match": True,
            "similarity": 0.8842,
            "enrolled_name": "byaquta",
        }

        response = client.post(
            "/verify-speaker",
            files={"file": ("clip.wav", sample_wav_bytes, "audio/wav")},
            data={"enrolled_name": "byaquta"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["match"] is True
        assert data["similarity"] == pytest.approx(0.8842, abs=1e-4)
        assert data["enrolled_name"] == "byaquta"


def test_verify_speaker_endpoint_not_found_404(
    client: TestClient, sample_wav_bytes: bytes
) -> None:
    with patch("api.main.list_enrolled_speakers", return_value=["byaquta"]):
        response = client.post(
            "/verify-speaker",
            files={"file": ("clip.wav", sample_wav_bytes, "audio/wav")},
            data={"enrolled_name": "unknown_speaker"},
        )
        assert response.status_code == 404
        assert "No enrolled voiceprint found" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 4. POST /analyze-context
# ---------------------------------------------------------------------------


def test_analyze_context_with_enrolled_speaker(
    client: TestClient, sample_wav_bytes: bytes
) -> None:
    with (
        patch("api.main.list_enrolled_speakers", return_value=["byaquta"]),
        patch("api.main.verify_speaker") as mock_verify,
        patch("api.main.get_transcriber") as mock_transcriber,
        patch("api.main.get_detector") as mock_detector_getter,
    ):
        mock_verify.return_value = {
            "match": False,
            "similarity": 0.25,
            "enrolled_name": "byaquta",
        }
        mock_trans = MagicMock()
        mock_trans.transcribe_chunk.return_value = "police customs arrest warrant OTP"
        mock_transcriber.return_value = mock_trans

        mock_det = MagicMock()
        mock_det.predict_waveform.return_value = {
            "label": "synthetic",
            "probability_synthetic": 0.85,
        }
        mock_detector_getter.return_value = mock_det

        response = client.post(
            "/analyze-context",
            files={"file": ("clip.wav", sample_wav_bytes, "audio/wav")},
            data={
                "transaction_context": "fund_transfer",
                "enrolled_name": "byaquta",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "base_fused_score" in data
        assert "contextual_score" in data
        assert data["risk_band"] == "high"
        assert "police customs arrest warrant OTP" in data["transcript"]
        assert len(data["matched_redflag_categories"]) > 0
        assert "authority_impersonation" in data["matched_redflag_categories"]
        assert "financial_action" in data["matched_redflag_categories"]


def test_analyze_context_without_enrolled_speaker(
    client: TestClient, sample_wav_bytes: bytes
) -> None:
    with (
        patch("api.main.get_transcriber") as mock_transcriber,
        patch("api.main.get_detector") as mock_detector_getter,
    ):
        mock_trans = MagicMock()
        mock_trans.transcribe_chunk.return_value = "hello how are you today"
        mock_transcriber.return_value = mock_trans

        mock_det = MagicMock()
        mock_det.predict_waveform.return_value = {
            "label": "real",
            "probability_synthetic": 0.05,
        }
        mock_detector_getter.return_value = mock_det

        response = client.post(
            "/analyze-context",
            files={"file": ("clip.wav", sample_wav_bytes, "audio/wav")},
            data={"transaction_context": "general_conversation"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["base_fused_score"] == pytest.approx(0.035, abs=0.01)
        assert data["contextual_score"] == pytest.approx(0.035, abs=0.01)
        assert data["risk_band"] == "low"
        assert data["transcript"] == "hello how are you today"
        assert data["matched_redflag_categories"] == []


def test_analyze_context_unknown_enrolled_speaker_404(
    client: TestClient, sample_wav_bytes: bytes
) -> None:
    with patch("api.main.list_enrolled_speakers", return_value=["byaquta"]):
        response = client.post(
            "/analyze-context",
            files={"file": ("clip.wav", sample_wav_bytes, "audio/wav")},
            data={"enrolled_name": "ghost_speaker"},
        )
        assert response.status_code == 404
        assert "No enrolled voiceprint found" in response.json()["detail"]


@pytest.mark.skipif(
    not SAMPLE_REAL_CLIP.exists(),
    reason=f"Sample file {SAMPLE_REAL_CLIP} not found on disk",
)
def test_real_clip_end_to_end_analyze(client: TestClient) -> None:
    """End-to-end integration test against real audio on disk (unmocked detector)."""
    with open(SAMPLE_REAL_CLIP, "rb") as f:
        response = client.post(
            "/analyze",
            files={"file": (SAMPLE_REAL_CLIP.name, f, "audio/wav")},
        )
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert 0.0 <= data["probability_synthetic"] <= 1.0
    assert data["risk_band"] in ("low", "medium", "high", "inconclusive")
