# fusion — multimodal call-context risk fusion (Phase 7).

from voxguard.fusion.context import (
    get_contact_familiarity_multiplier,
    get_transaction_multiplier,
)
from voxguard.fusion.redflags import RED_FLAG_PHRASES, scan_for_redflags
from voxguard.fusion.transcribe import LiveTranscriber

__all__ = [
    "LiveTranscriber",
    "RED_FLAG_PHRASES",
    "scan_for_redflags",
    "get_transaction_multiplier",
    "get_contact_familiarity_multiplier",
]

