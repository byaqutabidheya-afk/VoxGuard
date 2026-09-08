"""
main.py — FastAPI REST API for VoxGuard platform and integration services.

Exposes a clean REST API for integration with core banking systems, contact center
platforms, enterprise communication tools, and telecom networks.

Runs as a separate service process from the Gradio UI (app/app.py):
    uvicorn api.main:app --host 127.0.0.1 --port 8000

Both api/main.py and app/app.py import shared detection, embedding, verification,
and fusion modules directly from src/voxguard — ensuring zero duplicated business logic.

Out of Scope Note:
──────────────────
Production-scale authentication (OAuth/JWT), multi-tenancy, rate limiting, and
distributed voiceprint databases are intentionally out of scope for this hackathon
prototype, consistent with the master guide's defined scope boundaries.
"""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from voxguard.classifier.ensemble import WeightedAverageDetector
from voxguard.fusion.fuse import fuse_risk_with_context
from voxguard.fusion.redflags import scan_for_redflags
from voxguard.fusion.transcribe import LiveTranscriber
from voxguard.privacy.session_log import SessionLogger
from voxguard.risk.bands import score_to_band
from voxguard.risk.prevention import get_prevention_message
from voxguard.speaker.embedding import SpeakerEmbedder
from voxguard.speaker.enrollment import list_enrolled_speakers
from voxguard.speaker.verify import verify_speaker

logger = logging.getLogger(__name__)

# Maximum file size: 25 MB
MAX_FILE_SIZE_BYTES: int = 25 * 1024 * 1024

ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/x-pn-wav",
    "audio/vnd.wave",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/x-ogg",
    "audio/flac",
    "audio/x-flac",
    "audio/webm",
    "audio/mp4",
    "audio/m4a",
    "audio/x-m4a",
}

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm", ".wave"}

# Shared singleton model holders
_DETECTOR: WeightedAverageDetector | None = None
_SPEAKER_EMBEDDER: SpeakerEmbedder | None = None
_TRANSCRIBER: LiveTranscriber | None = None
_SESSION_LOGGER = SessionLogger()
_SESSION_LOGGER.purge_older_than(30)


def get_detector() -> WeightedAverageDetector:
    """Lazy-initializes and returns the shared production WeightedAverageDetector instance."""
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = WeightedAverageDetector(
            wav2vec2_classifier_path="models/classifiers/wav2vec2_hindi_combined_logreg.joblib",
            wavlm_classifier_path="models/classifiers/wavlm_hindi_combined_logreg.joblib",
            threshold=0.6,
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
        _TRANSCRIBER = LiveTranscriber(model_size="base", device="cpu")
    return _TRANSCRIBER


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager: pre-loads models once at startup."""
    logger.info("Initializing VoxGuard models for FastAPI service...")
    get_detector()
    get_speaker_embedder()
    get_transcriber()
    logger.info("VoxGuard models initialized.")
    yield
    logger.info("Shutting down VoxGuard FastAPI service.")


app = FastAPI(
    title="VoxGuard REST API",
    description="Real-time Voice Cloning Detection and Threat Prevention API",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for cross-origin platform integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service liveness status")


class AnalyzeResponse(BaseModel):
    label: str = Field(..., description="Classification verdict: 'real' or 'synthetic'")
    probability_synthetic: float = Field(..., description="Estimated synthetic speech probability [0.0, 1.0]")
    risk_band: str = Field(..., description="Assigned risk band: 'low', 'medium', 'high', or 'inconclusive'")
    prevention_message: Optional[str] = Field(None, description="Actionable security prompt for high/medium risk")


class VerifySpeakerResponse(BaseModel):
    match: bool = Field(..., description="Whether audio matches the enrolled voiceprint")
    similarity: float = Field(..., description="Cosine similarity score against enrolled voiceprint")
    enrolled_name: str = Field(..., description="Enrolled speaker identifier")


class AnalyzeContextResponse(BaseModel):
    base_fused_score: float = Field(..., description="Acoustic + semantic fused score before context multipliers")
    contextual_score: float = Field(..., description="Final risk score after transaction and contact multipliers")
    risk_band: str = Field(..., description="Assigned contextual risk band: 'low', 'medium', 'high'")
    transcript: str = Field(..., description="Transcribed text from speech recognition")
    matched_redflag_categories: List[str] = Field(default_factory=list, description="Scam phrase categories detected")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def validate_audio_content_type(file: UploadFile) -> None:
    """Validates that the uploaded file is a supported audio format (HTTP 415 on mismatch)."""
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    content_type = (file.content_type or "").lower()

    is_audio = (
        content_type in ALLOWED_CONTENT_TYPES
        or content_type.startswith("audio/")
        or (suffix in ALLOWED_EXTENSIONS and content_type in ("application/octet-stream", "binary/octet-stream", ""))
    )

    if not is_audio and suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported media type '{file.content_type}'. "
                "Please upload a supported audio file (e.g., audio/wav)."
            ),
        )


async def read_audio_file(file: UploadFile, target_sr: int = 16_000) -> tuple[np.ndarray, int]:
    """Reads, validates size and format, and decodes UploadFile to 16kHz mono float32."""
    validate_audio_content_type(file)

    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowable size of 25MB ({len(content)} bytes uploaded).",
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty.",
        )

    try:
        waveform, native_sr = sf.read(io.BytesIO(content), dtype="float32", always_2d=False)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not decode audio file: {exc}",
        ) from exc

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    if native_sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=native_sr, target_sr=target_sr)

    return waveform.astype(np.float32), target_sr


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness probe for platform integration and container orchestration."""
    return HealthResponse(status="ok")


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_audio(
    file: UploadFile = File(..., description="WAV or supported audio file to analyze"),
) -> AnalyzeResponse:
    """Classifies audio using the production dual-backbone ensemble detector."""
    waveform, sr = await read_audio_file(file)
    detector = get_detector()
    result = detector.predict_waveform(waveform, sr)

    prob_synth = result.get("probability_synthetic")
    probability_synthetic = 0.0 if prob_synth is None else float(prob_synth)
    label = str(result.get("label", "real"))
    risk_band = score_to_band(probability_synthetic)
    prevention_msg = get_prevention_message(risk_band)

    _SESSION_LOGGER.log_event(
        event_type="api_analyze",
        risk_band=risk_band,
        probability_synthetic=probability_synthetic,
        flagged=risk_band in ("medium", "high"),
    )

    return AnalyzeResponse(
        label=label,
        probability_synthetic=probability_synthetic,
        risk_band=risk_band,
        prevention_message=prevention_msg,
    )


@app.post("/verify-speaker", response_model=VerifySpeakerResponse)
async def verify_speaker_endpoint(
    file: UploadFile = File(..., description="Audio clip to verify against enrolled voiceprint"),
    enrolled_name: str = Form(..., description="Name of the previously enrolled trusted contact"),
) -> VerifySpeakerResponse:
    """Verifies whether caller audio matches an enrolled contact's voiceprint."""
    clean_name = enrolled_name.strip()
    if clean_name not in list_enrolled_speakers():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No enrolled voiceprint found for speaker '{clean_name}'. Available speakers: {list_enrolled_speakers()}",
        )

    waveform, sr = await read_audio_file(file)
    embedder = get_speaker_embedder()
    try:
        result = verify_speaker(
            live_waveform=waveform,
            sr=sr,
            enrolled_name=clean_name,
            embedder=embedder,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Voiceprint verification failed: {exc}",
        ) from exc

    match = bool(result["match"])
    similarity = float(result["similarity"])

    _SESSION_LOGGER.log_event(
        event_type="api_verify_speaker",
        risk_band="low" if match else "high",
        probability_synthetic=1.0 - max(0.0, similarity),
        flagged=not match,
    )

    return VerifySpeakerResponse(
        match=match,
        similarity=similarity,
        enrolled_name=clean_name,
    )


@app.post("/analyze-context", response_model=AnalyzeContextResponse)
async def analyze_context_endpoint(
    file: UploadFile = File(..., description="Audio file to analyze with full multimodal context"),
    transaction_context: str = Form(
        "general_conversation",
        description="Transaction or call type: 'general_conversation', 'otp_request', 'fund_transfer', 'confidential_info_request'",
    ),
    enrolled_name: Optional[str] = Form(
        None,
        description="Optional enrolled contact name to verify against for contact familiarity adjustment",
    ),
) -> AnalyzeContextResponse:
    """Runs the multimodal fusion pipeline: transcription, red flags, acoustic classification, and context multipliers."""
    voiceprint_result = None
    clean_name = enrolled_name.strip() if enrolled_name else None

    if clean_name:
        if clean_name not in list_enrolled_speakers():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No enrolled voiceprint found for speaker '{clean_name}'. Available speakers: {list_enrolled_speakers()}",
            )
        waveform, sr = await read_audio_file(file)
        embedder = get_speaker_embedder()
        try:
            voiceprint_result = verify_speaker(
                live_waveform=waveform,
                sr=sr,
                enrolled_name=clean_name,
                embedder=embedder,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Voiceprint verification failed: {exc}",
            ) from exc
    else:
        waveform, sr = await read_audio_file(file)

    # 1. Speech-to-text
    transcriber = get_transcriber()
    transcript = transcriber.transcribe_chunk(waveform, sr)

    # 2. Red-flag keyword scan
    redflags = scan_for_redflags(transcript)
    kw_score = float(redflags["keyword_risk_score"])
    matched_categories = list(redflags["categories"])

    # 3. Acoustic classifier
    detector = get_detector()
    det_result = detector.predict_waveform(waveform, sr)
    prob_synth = det_result.get("probability_synthetic")
    raw_audio_score = 0.0 if prob_synth is None else float(prob_synth)

    # 4. Multimodal context fusion
    fusion = fuse_risk_with_context(
        audio_score=raw_audio_score,
        keyword_risk_score=kw_score,
        transaction_context=transaction_context,
        voiceprint_result=voiceprint_result,
    )
    base_fused = float(fusion["base_fused_score"])
    contextual = float(fusion["contextual_score"])
    band = score_to_band(contextual)

    _SESSION_LOGGER.log_event(
        event_type="api_analyze_context",
        risk_band=band,
        probability_synthetic=contextual,
        flagged=band in ("medium", "high"),
    )

    return AnalyzeContextResponse(
        base_fused_score=base_fused,
        contextual_score=contextual,
        risk_band=band,
        transcript=transcript,
        matched_redflag_categories=matched_categories,
    )
