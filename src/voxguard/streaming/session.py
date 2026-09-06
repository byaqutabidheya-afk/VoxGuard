"""Streaming session orchestration for chunking, scoring, and smoothing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from voxguard import config
from voxguard.classifier.ensemble import WeightedAverageDetector
from voxguard.streaming.buffer import StreamingBuffer
from voxguard.streaming.ema import RunningRiskScore
from voxguard.streaming.scorer import StreamingScorer


class StreamingSession:
    """Composes buffering, chunk scoring, and running risk tracking."""

    def __init__(
        self,
        detector: Any | None = None,
        sample_rate: int | None = None,
        chunk_seconds: float | None = None,
        overlap_seconds: float | None = None,
        alpha: float = 0.3,
        flag_threshold: float | None = None,
        consecutive_flags_required: int = 3,
    ) -> None:
        self.detector = detector or WeightedAverageDetector(
            wav2vec2_classifier_path="models/classifiers/wav2vec2_hindi_combined_logreg.joblib",
            wavlm_classifier_path="models/classifiers/wavlm_hindi_combined_logreg.joblib",
        )
        self._initial_sample_rate = int(sample_rate) if sample_rate is not None else None
        self.sample_rate = self._initial_sample_rate
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds

        effective_sr = (
            self.sample_rate if self.sample_rate is not None else config.SAMPLE_RATE
        )
        self.buffer = StreamingBuffer(
            sample_rate=effective_sr,
            chunk_seconds=config.STREAM_CHUNK_SECONDS
            if chunk_seconds is None
            else chunk_seconds,
            overlap_seconds=config.STREAM_OVERLAP_SECONDS
            if overlap_seconds is None
            else overlap_seconds,
        )
        self.scorer = StreamingScorer(self.detector)
        self.risk_score = RunningRiskScore(alpha=alpha)
        self.flag_threshold = float(
            config.STREAM_FLAG_THRESHOLD if flag_threshold is None else flag_threshold
        )
        self.consecutive_flags_required = max(1, int(consecutive_flags_required))
        self._consecutive_flags = 0
        self._start_time: float | None = None
        self._seconds_to_flag: float | None = None
        self._audio_seconds_elapsed = 0.0
        self._logged_flag_event = False

    def push_audio(self, audio_frame: np.ndarray, sr: int) -> dict:
        """Push audio into the session and return the current risk state."""
        if sr <= 0:
            raise ValueError(f"sr must be positive; got {sr!r}.")

        if self._start_time is None:
            self._start_time = 0.0

        if self.sample_rate is None:
            self.sample_rate = int(sr)
            self.buffer = StreamingBuffer(
                sample_rate=self.sample_rate,
                chunk_seconds=self.chunk_seconds,
                overlap_seconds=self.overlap_seconds,
            )
        elif self.sample_rate != sr:
            raise ValueError(
                f"Sample rate mismatch: session initialized for {self.sample_rate} Hz, "
                f"but push_audio received {sr} Hz."
            )

        frame = np.asarray(audio_frame)
        self._audio_seconds_elapsed += float(frame.size) / float(sr)

        windows = self.buffer.push(frame)
        for window in windows:
            score = self.scorer.score_chunk(window, sr)
            self.risk_score.update(score)

        current_score = self.risk_score.current()
        running_score = 0.0 if current_score is None else float(current_score)

        if current_score is not None and running_score >= self.flag_threshold:
            self._consecutive_flags += 1
            if self._consecutive_flags >= self.consecutive_flags_required:
                if self._seconds_to_flag is None:
                    self._seconds_to_flag = self._audio_seconds_elapsed
        else:
            self._consecutive_flags = 0

        flagged = self._consecutive_flags >= self.consecutive_flags_required

        return {
            "running_score": running_score,
            "flagged": flagged,
            "seconds_since_start": self._audio_seconds_elapsed,
            "seconds_to_flag": self._seconds_to_flag,
        }

    def reset(self) -> None:
        """Reset buffering, smoothing, and timing state for a fresh session."""
        self.sample_rate = self._initial_sample_rate
        effective_sr = (
            self.sample_rate if self.sample_rate is not None else config.SAMPLE_RATE
        )
        self.buffer = StreamingBuffer(
            sample_rate=effective_sr,
            chunk_seconds=self.chunk_seconds,
            overlap_seconds=self.overlap_seconds,
        )
        self.risk_score.reset()
        self._consecutive_flags = 0
        self._start_time = None
        self._seconds_to_flag = None
        self._audio_seconds_elapsed = 0.0
        self._logged_flag_event = False


def simulate_stream(
    audio: str | Path | np.ndarray,
    session: StreamingSession | None = None,
    chunk_seconds: float | None = None,
    overlap_seconds: float | None = None,
    flag_threshold: float | None = None,
    consecutive_flags_required: int = 3,
    step_seconds: float = 0.25,
    realtime: bool = True,
    real_time_paced: bool | None = None,
    callback: Callable[[dict], None] | None = None,
    sr: int = config.SAMPLE_RATE,
) -> dict:
    """Feeds audio in small increments into a StreamingSession and returns a summary.

    Parameters
    ----------
    audio:
        Path to an audio file or an in-memory 1-D numpy waveform.
    session:
        Optional StreamingSession instance. If None, one is constructed with the
        specified chunk_seconds, overlap_seconds, flag_threshold, and
        consecutive_flags_required (or config defaults).
    chunk_seconds, overlap_seconds, flag_threshold, consecutive_flags_required:
        Parameters passed to StreamingSession if session is None.
    step_seconds:
        Duration of each incremental audio slice pushed to session (default 0.25s / 250ms).
    realtime:
        If True, calls time.sleep between chunks to pace playback at real-time wall-clock speed.
        Deprecated in favor of ``real_time_paced``; kept for backward compatibility.
    real_time_paced:
        If provided, overrides ``realtime``. True sleeps between chunks (live-simulation pacing),
        False processes all chunks as fast as possible (uploaded-file replay use case).
    callback:
        Optional callback invoked with each push_audio() result dict.
    sr:
        Sample rate (used when audio is an in-memory waveform or when resampling).

    Returns
    -------
    dict:
        {
            "total_duration": float,
            "final_running_score": float,
            "flagged": bool,
            "seconds_to_flag": float | None,
            "step_results": list[dict],
        }
    """
    import time
    from voxguard.utils.audio_io import load_audio

    if real_time_paced is not None:
        realtime = real_time_paced

    if session is None:
        session = StreamingSession(
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
            flag_threshold=flag_threshold,
            consecutive_flags_required=consecutive_flags_required,
        )

    if isinstance(audio, (str, Path)):
        waveform, audio_sr = load_audio(audio, target_sr=sr)
    else:
        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
        audio_sr = sr

    if waveform.size == 0:
        return {
            "total_duration": 0.0,
            "final_running_score": 0.0,
            "flagged": False,
            "seconds_to_flag": None,
            "step_results": [],
        }

    step_samples = max(1, int(round(step_seconds * audio_sr)))
    step_results: list[dict] = []

    for offset in range(0, len(waveform), step_samples):
        frame = waveform[offset : offset + step_samples]
        result = session.push_audio(frame, audio_sr)
        step_results.append(result)

        if callback is not None:
            callback(result)

        if realtime:
            frame_duration = float(len(frame)) / float(audio_sr)
            time.sleep(frame_duration)

    total_duration = float(len(waveform)) / float(audio_sr)
    last_res = step_results[-1]
    ever_flagged = bool(session._seconds_to_flag is not None or last_res["flagged"])

    return {
        "total_duration": total_duration,
        "final_running_score": float(last_res["running_score"]),
        "flagged": ever_flagged,
        "seconds_to_flag": session._seconds_to_flag,
        "step_results": step_results,
    }

