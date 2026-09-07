"""
transcribe.py — real-time and full-file speech-to-text via faster-whisper (Phase 7).

Provides ``LiveTranscriber`` for lightweight, low-latency automatic speech
recognition (ASR) to support multimodal call-context risk fusion and scam
keyword/phrase detection.

Tradeoff Note:
──────────────
Smaller/faster models (e.g. "base" or "tiny") with int8 quantization on CPU
are deliberately chosen here. The goal of this transcription layer is
semantic keyword spotting and red-flag phrase scanning (e.g., urgency cues,
OTP/bank requests, authority impersonation), not verbatim high-fidelity
general ASR. On this project's CPU-only local hardware, running int8 on CPU
achieves the throughput and low latency needed for real-time streaming
without requiring dedicated GPU hardware — this is an intentional design
decision, not a compromise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np

from voxguard import config
from voxguard.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Minimum RMS energy threshold below which audio chunks are treated as silence
SILENCE_RMS_THRESHOLD: float = 1e-3
MIN_CHUNK_SAMPLES: int = 800  # 0.05s at 16kHz


class LiveTranscriber:
    """Lightweight streaming and batch transcriber using faster-whisper.

    Parameters
    ----------
    model_size:
        Whisper model size identifier (e.g. ``"base"``, ``"tiny"``, ``"small"``).
        Default is ``"base"``.
    device:
        Inference device (``"cpu"`` or ``"cuda"``). If None, defaults to
        ``"cpu"`` (or ``config.get_device()``). On CPU, ``compute_type="int8"``
        is always used for fast execution.
    language:
        Target language code (e.g. ``"hi"`` for Hindi) or ``None`` (default)
        for automatic language detection across code-switched Hindi/Hinglish
        and English audio.
    compute_type:
        Quantization type for CTranslate2 engine. If None, defaults to ``"int8"``
        when running on CPU, or ``"float16"`` on CUDA.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        language: Optional[str] = None,
        compute_type: Optional[str] = None,
    ) -> None:
        from faster_whisper import WhisperModel

        self.model_size = model_size
        self.device = device or "cpu"
        self.language = language

        # On CPU, int8 quantization is specifically designed to run with low latency;
        # on CUDA, float16 is standard.
        if compute_type is None:
            self.compute_type = "int8" if self.device == "cpu" else "float16"
        else:
            self.compute_type = compute_type

        logger.info(
            "Loading faster-whisper model '%s' on %s (compute_type=%s, language=%s)...",
            self.model_size,
            self.device,
            self.compute_type,
            self.language,
        )

        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

        logger.info("LiveTranscriber ready: model='%s', device=%s", self.model_size, self.device)

    def transcribe_chunk(self, waveform: np.ndarray, sr: int) -> str:
        """Transcribes a short audio chunk (e.g. ~1.5-2.0s streaming window).

        Designed to be called directly on the windows emitted by ``StreamingBuffer``.
        Handles empty or silent chunks gracefully by returning an empty string.

        Parameters
        ----------
        waveform:
            1-D numpy array containing audio samples.
        sr:
            Sample rate of the incoming waveform in Hz.

        Returns
        -------
        str
            Transcribed text segment (stripped), or ``""`` if empty/silent.
        """
        if waveform is None or len(waveform) == 0:
            return ""

        audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if len(audio) < MIN_CHUNK_SAMPLES:
            return ""

        # Energy / silence check to avoid transcribing background silence or hallucinating tokens
        rms = float(np.sqrt(np.mean(audio**2)))
        if rms < SILENCE_RMS_THRESHOLD:
            return ""

        # Resample to 16000 Hz if necessary
        if sr != config.SAMPLE_RATE and sr > 0:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=config.SAMPLE_RATE
            ).astype(np.float32)

        try:
            segments, _ = self.model.transcribe(
                audio,
                language=self.language,
                beam_size=1,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            text_pieces = [s.text.strip() for s in segments if s.text and s.text.strip()]
            return " ".join(text_pieces)
        except Exception as exc:
            logger.warning("transcribe_chunk encountered error: %s", exc)
            return ""

    def transcribe_full(self, audio_path: Union[str, Path]) -> str:
        """Transcribes an entire audio file at once.

        Suitable for non-streaming batch analysis and the Upload File workflow.

        Parameters
        ----------
        audio_path:
            Path to the audio file on disk.

        Returns
        -------
        str
            Full transcribed text.

        Raises
        ------
        FileNotFoundError
            If ``audio_path`` does not exist.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found for transcription: {audio_path}")

        try:
            segments, _ = self.model.transcribe(
                str(path),
                language=self.language,
                beam_size=1,
                vad_filter=True,
            )
            text_pieces = [s.text.strip() for s in segments if s.text and s.text.strip()]
            return " ".join(text_pieces)
        except Exception as exc:
            logger.exception("transcribe_full failed for '%s': %s", audio_path, exc)
            raise
