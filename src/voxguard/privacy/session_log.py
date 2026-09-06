"""Session-level privacy logging and retention enforcement."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_LOG_PATH = Path("data/logs/session_events.jsonl")


def _risk_band(score: float) -> str:
    """Map a synthetic-voice probability to a coarse risk band."""
    if score < 0.4:
        return "low"
    if score < 0.7:
        return "medium"
    return "high"


class SessionLogger:
    """Append-only JSONL session logger with time-based retention enforcement.

    The log schema is intentionally narrow: it records risk scores and category
    labels only. This function signature structurally prevents logging of raw
    audio bytes, full transcripts, or speaker identity information.
    """

    def __init__(self, log_path: str | Path | None = None) -> None:
        self.log_path = Path(log_path) if log_path is not None else _DEFAULT_LOG_PATH
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        event_type: str,
        risk_band: str,
        probability_synthetic: float,
        flagged: bool,
        matched_redflag_categories: list[str] | None = None,
        transaction_context: str | None = None,
        contact_match: bool | None = None,
    ) -> None:
        """Append one structured JSON log line.

        Parameters
        ----------
        event_type:
            Short label for the event (e.g. ``"flag_event"``, ``"session_reset"``,
            ``"upload_analysis"``).
        risk_band:
            Coarse risk category (``"low"``, ``"medium"``, or ``"high"``).
        probability_synthetic:
            The synthetic-voice probability score in ``[0.0, 1.0]``.
        flagged:
            Whether the detector flagged this event as synthetic.
        matched_redflag_categories:
            Category names from the Phase 9 red-flag scanner (e.g.
            ``["urgency", "financial_action"]``). Never include raw matched phrase
            text here.
        transaction_context:
            Optional short context label from the multimodal fusion layer
            (Phase 11+). ``None`` when not available.
        contact_match:
            Optional speaker-voiceprint match result (Phase 5+). ``None`` when
            not available.

        Notes
        -----
        This method deliberately does **not** accept raw audio data, full
        transcripts, or enrolled speaker identifiers. If you find yourself
        wanting to log those, stop — they belong in a separate, access-controlled
        audit pipeline with different retention and access controls.
        """
        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "risk_band": risk_band,
            "probability_synthetic": float(probability_synthetic),
            "flagged": bool(flagged),
            "matched_redflag_categories": matched_redflag_categories or [],
            "transaction_context": transaction_context,
            "contact_match": contact_match,
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Failed to write session log entry: %s", exc)

    def purge_older_than(self, days: int = 30) -> int:
        """Delete log lines older than *days* and return the count removed.

        Parameters
        ----------
        days:
            Retention window in days. **30 days is a sensible hackathon-prototype
            default, not a regulatory-compliant value.** A production deployment
            should set this per applicable data-protection requirements (e.g.
            GDPR, DPDP) and legal-hold policies.
        """
        if days <= 0:
            return 0
        cutoff = datetime.now(timezone.utc).timestamp() - (days * 86400)
        kept: list[str] = []
        purged = 0
        if not self.log_path.exists():
            return 0
        try:
            with open(self.log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        ts_str = record.get("timestamp")
                        if ts_str:
                            ts = datetime.fromisoformat(ts_str).timestamp()
                            if ts >= cutoff:
                                kept.append(line)
                            else:
                                purged += 1
                        else:
                            kept.append(line)
                    except (json.JSONDecodeError, ValueError, OSError):
                        kept.append(line)
            with open(self.log_path, "w", encoding="utf-8") as fh:
                for line in kept:
                    fh.write(line + "\n")
        except OSError as exc:
            logger.warning("Failed to purge session log: %s", exc)
        return purged

    def read_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent *limit* logged events as parsed dicts."""
        entries: list[dict[str, Any]] = []
        if not self.log_path.exists():
            return entries
        try:
            with open(self.log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError as exc:
            logger.warning("Failed to read session log: %s", exc)
        entries.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        return entries[: max(0, limit)]
