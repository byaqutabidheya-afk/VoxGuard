"""Streaming session orchestration for chunking, scoring, and smoothing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from voxguard import config
from voxguard.classifier.ensemble import get_production_detector
from voxguard.streaming.buffer import StreamingBuffer
from voxguard.streaming.ema import RunningRiskScore
from voxguard.streaming.scorer import StreamingScorer


CONSECUTIVE_UNITS = ("pushes", "updates")


class StreamingSession:
    """Composes buffering, chunk scoring, and running risk tracking.

    ``consecutive_flags_required`` counts consecutive *decisions* at or above
    ``flag_threshold``. What one decision is depends on ``consecutive_unit``:

    - ``"pushes"`` (default, the original behaviour): one ``push_audio`` call.
      The running score only changes when a window completes (every stride,
      1.0 s with the default 1.5 s window / 0.5 s overlap) but pushes arrive
      more often (0.25 s in ``simulate_stream``), so with 0.25 s pushes a count
      of 1..4 all resolve inside a single score update and test no persistence.
    - ``"updates"``: one score UPDATE, i.e. one scored window. The running
      score is compared to the threshold each time a window is scored; a window
      skipped as silence makes no decision and leaves the count unchanged.
      A count of N then means N consecutive per-stride decisions.
    """

    def __init__(
        self,
        detector: Any | None = None,
        sample_rate: int | None = None,
        chunk_seconds: float | None = None,
        overlap_seconds: float | None = None,
        alpha: float = 0.3,
        flag_threshold: float | None = None,
        consecutive_flags_required: int = 3,
        consecutive_unit: str = "pushes",
    ) -> None:
        if consecutive_unit not in CONSECUTIVE_UNITS:
            raise ValueError(
                f"consecutive_unit must be one of {CONSECUTIVE_UNITS}; got {consecutive_unit!r}."
            )
        self.consecutive_unit = consecutive_unit
        # Default: the shared chunk-native STREAMING detector (config.PRODUCTION_STREAMING_CLASSIFIERS): a
        # streaming session scores fixed-length windows, which is what those heads were trained on.
        # NOTE: consecutive_flags_required / consecutive_unit keep their original defaults (3 / "pushes") for
        # backward compatibility. Production callers must pass config.STREAM_CONSECUTIVE_FLAGS_REQUIRED and
        # config.STREAM_CONSECUTIVE_UNIT: the F4 calibration was measured with per-score-update counting.
        self.detector = detector or get_production_detector("streaming")
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
            if score is not None and self.consecutive_unit == "updates":
                self._register_decision(float(self.risk_score.current()))

        current_score = self.risk_score.current()
        running_score = 0.0 if current_score is None else float(current_score)

        if self.consecutive_unit == "pushes":
            self._register_decision(None if current_score is None else running_score)

        flagged = self._consecutive_flags >= self.consecutive_flags_required

        return {
            "running_score": running_score,
            "flagged": flagged,
            "seconds_since_start": self._audio_seconds_elapsed,
            "seconds_to_flag": self._seconds_to_flag,
        }

    def _register_decision(self, running_score: float | None) -> None:
        """Counts one decision: extends the streak if at/above threshold, else resets it."""
        if running_score is not None and running_score >= self.flag_threshold:
            self._consecutive_flags += 1
            if self._consecutive_flags >= self.consecutive_flags_required:
                if self._seconds_to_flag is None:
                    self._seconds_to_flag = self._audio_seconds_elapsed
        else:
            self._consecutive_flags = 0

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

