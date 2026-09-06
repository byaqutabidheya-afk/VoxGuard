#!/usr/bin/env python3
"""
simulate_stream.py — Offline real-time streaming simulation CLI harness.

Reads a full pre-recorded audio file and feeds it into a StreamingSession in small
real-time-paced increments (pacing playback speed using time.sleep) to validate the
system's timing, buffering, scoring, and flagging behavior end-to-end before wiring
to a live microphone.

Usage:
  python scripts/simulate_stream.py --audio_path path/to/clip.wav
  python scripts/simulate_stream.py --audio_path path/to/clip.wav --chunk_seconds 1.5 --overlap_seconds 0.5 --flag_threshold 0.6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from voxguard import config
from voxguard.streaming.session import StreamingSession, simulate_stream
from voxguard.utils.logging_utils import get_logger

logger = get_logger("simulate_stream")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate real-time streaming audio inference on a pre-recorded audio file."
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="Path to pre-recorded audio file (.wav, .flac, etc.) to simulate streaming on.",
    )
    parser.add_argument(
        "--chunk_seconds",
        type=float,
        default=None,
        help=f"Streaming chunk window size in seconds (default: {config.STREAM_CHUNK_SECONDS}s from config).",
    )
    parser.add_argument(
        "--overlap_seconds",
        type=float,
        default=None,
        help=f"Streaming chunk overlap in seconds (default: {config.STREAM_OVERLAP_SECONDS}s from config).",
    )
    parser.add_argument(
        "--flag_threshold",
        type=float,
        default=None,
        help=f"Running risk score flag threshold (default: {config.STREAM_FLAG_THRESHOLD} from config).",
    )
    parser.add_argument(
        "--consecutive_flags_required",
        type=int,
        default=3,
        help="Number of consecutive above-threshold pushes required to trigger a flag (default: 3).",
    )
    parser.add_argument(
        "--step_seconds",
        type=float,
        default=0.25,
        help="Incremental audio slice size pushed per step in seconds (default: 0.25s / 250ms).",
    )
    parser.add_argument(
        "--no_sleep",
        action="store_true",
        default=False,
        help="Disable real-time time.sleep pacing and process all chunks as fast as possible.",
    )
    parser.add_argument(
        "--real_time_paced",
        action="store_true",
        default=None,
        help="Enable real-time time.sleep pacing (default when neither --no_sleep nor --real_time_paced is given).",
    )

    args = parser.parse_args()
    audio_path = Path(args.audio_path)

    if not audio_path.exists():
        logger.error("Audio file does not exist: %s", audio_path)
        sys.exit(1)

    # Instantiate session honoring the 'None defers to config default' pattern
    session = StreamingSession(
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        flag_threshold=args.flag_threshold,
        consecutive_flags_required=args.consecutive_flags_required,
    )

    chunk_sec_display = (
        session.buffer.chunk_seconds
        if args.chunk_seconds is None
        else args.chunk_seconds
    )
    overlap_sec_display = (
        session.buffer.overlap_seconds
        if args.overlap_seconds is None
        else args.overlap_seconds
    )
    flag_thresh_display = (
        session.flag_threshold
        if args.flag_threshold is None
        else args.flag_threshold
    )
    if args.real_time_paced is not None:
        realtime_pacing = args.real_time_paced
    else:
        realtime_pacing = not args.no_sleep

    print("\n" + "=" * 70)
    print(" VOXGUARD REAL-TIME STREAMING SIMULATION")
    print("=" * 70)
    print(f" Audio File:       {audio_path}")
    print(f" Chunk Size:       {chunk_sec_display:.2f}s")
    print(f" Overlap Size:     {overlap_sec_display:.2f}s")
    print(f" Flag Threshold:   {flag_thresh_display:.2f}")
    print(f" Consecutive Req:  {session.consecutive_flags_required} frames")
    print(f" Feed Step Size:   {args.step_seconds * 1000:.0f}ms increments")
    print(f" Real-time Pacing: {'Enabled (time.sleep paced)' if realtime_pacing else 'Disabled (fast processing)'}")
    print("-" * 70)
    print(" Pushing audio frames to StreamingSession...")
    print("-" * 70)

    def print_step(result: dict) -> None:
        t = result["seconds_since_start"]
        score = result["running_score"]
        flagged = result["flagged"]
        flag_str = "[FLAGGED]" if flagged else "[OK]"
        flag_info = ""
        if result.get("seconds_to_flag") is not None and flagged:
            flag_info = f" (seconds to flag: {result['seconds_to_flag']:.2f}s)"
        print(f"  [t={t:5.2f}s] Running Risk Score: {score:.4f}  |  Status: {flag_str:<9}{flag_info}")

    try:
        summary = simulate_stream(
            audio=audio_path,
            session=session,
            step_seconds=args.step_seconds,
            real_time_paced=realtime_pacing,
            callback=print_step,
        )

        print("-" * 70)
        print("\n" + "=" * 70)
        print(" SIMULATION SUMMARY")
        print("=" * 70)
        print(f" Total Duration Processed: {summary['total_duration']:.2f}s")
        print(f" Final Running Risk Score: {summary['final_running_score']:.4f}")
        print(f" Synthetic Flag Triggered: {summary['flagged']}")
        if summary["seconds_to_flag"] is not None:
            print(f" Time to First Flag:       {summary['seconds_to_flag']:.2f}s")
        else:
            print(" Time to First Flag:       N/A (Did not flag)")
        print("=" * 70 + "\n")

    except Exception as exc:
        logger.error("Streaming simulation failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
