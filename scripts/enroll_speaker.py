#!/usr/bin/env python3
"""
enroll_speaker.py — CLI wrapper for Phase 5 speaker voiceprint enrollment.

Enroll or delete a speaker's voiceprint from the command line, for testing
before the Gradio enrollment panel exists.

Examples
--------
Enroll a speaker from 2-3 reference clips (5-10s each recommended)::

    python scripts/enroll_speaker.py --name priya \\
        --clips data/raw/hindi_hinglish/references/priya_ref.wav \\
                data/raw/hindi_hinglish/real/priya_neutral_01.wav

Delete an enrolled speaker's voiceprint::

    python scripts/enroll_speaker.py --delete priya
"""

from __future__ import annotations

import argparse
import sys

from voxguard.speaker.embedding import SpeakerEmbedder
from voxguard.speaker.enrollment import delete_speaker, enroll_speaker
from voxguard.utils.logging_utils import get_logger

logger = get_logger("enroll_speaker")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enroll or delete a speaker voiceprint (Phase 5)."
    )
    parser.add_argument("--name", type=str, help="Speaker identifier to enroll.")
    parser.add_argument(
        "--clips",
        nargs="+",
        default=None,
        metavar="PATH",
        help="Space-separated paths to reference audio clips "
        "(recommend 2-3 clips, 5-10s each, for a robust voiceprint).",
    )
    parser.add_argument(
        "--delete",
        metavar="NAME",
        type=str,
        default=None,
        help="Delete an enrolled speaker's voiceprint instead of enrolling.",
    )
    parser.add_argument(
        "--backend",
        choices=["speechbrain", "pyannote"],
        default="speechbrain",
        help="Speaker-embedding backend to use for enrollment (default: speechbrain).",
    )

    args = parser.parse_args()

    if args.delete:
        if args.name or args.clips:
            parser.error("--delete cannot be combined with --name/--clips.")
        deleted = delete_speaker(args.delete)
        if deleted:
            print(f"Deleted voiceprint for '{args.delete}'.")
        else:
            print(f"No voiceprint found for '{args.delete}' — nothing to delete.")
        return

    if not args.name or not args.clips:
        parser.error("--name and --clips are both required to enroll a speaker (or use --delete NAME).")

    try:
        embedder = SpeakerEmbedder(backend=args.backend)
        enroll_speaker(args.name, args.clips, embedder)
    except Exception as exc:
        logger.error("Enrollment failed: %s", exc)
        sys.exit(1)

    print(f"Enrolled '{args.name}' from {len(args.clips)} clip(s).")
    print(f"Voiceprint saved to models/voiceprints/{args.name}.npy")


if __name__ == "__main__":
    main()
