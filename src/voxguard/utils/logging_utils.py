"""
logging_utils.py — centralised logger factory for VoxGuard.

Every module in the project should obtain its logger via::

    from voxguard.utils.logging_utils import get_logger
    logger = get_logger(__name__)

and then use ``logger.info()``, ``logger.warning()``, etc. rather than
``print()``.  The only legitimate use of ``print()`` in this codebase is
user-facing output in the Gradio UI (app/app.py).

Log level
─────────
The default level is INFO.  Override for a session by setting the
``VOXGUARD_LOG_LEVEL`` environment variable before starting the process::

    VOXGUARD_LOG_LEVEL=DEBUG python -m voxguard ...

Valid values (case-insensitive): DEBUG, INFO, WARNING, ERROR, CRITICAL.
An unrecognised value is silently ignored and INFO is used instead.
"""

from __future__ import annotations

import logging
import os
import sys

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# All VoxGuard loggers share a single top-level handler attached to the
# root "voxguard" logger.  This is configured exactly once on first import
# and never again, so repeated calls to get_logger() are safe.
_HANDLER_ATTACHED = False

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _resolve_level() -> int:
    """Read VOXGUARD_LOG_LEVEL from the environment and return a logging level int.

    Falls back to ``logging.INFO`` for any missing or unrecognised value.
    """
    raw = os.environ.get("VOXGUARD_LOG_LEVEL", "INFO").upper().strip()
    level = getattr(logging, raw, None)
    if isinstance(level, int):
        return level
    # Unrecognised string — warn on stderr once and fall back.
    print(
        f"[voxguard] WARNING: unrecognised VOXGUARD_LOG_LEVEL={raw!r}; "
        "defaulting to INFO.",
        file=sys.stderr,
    )
    return logging.INFO


def _configure_root_logger() -> None:
    """Attach a single StreamHandler to the 'voxguard' root logger.

    Called at most once per process (guarded by ``_HANDLER_ATTACHED``).
    Uses ``sys.stderr`` so that log output does not pollute ``sys.stdout``
    (important when the FastAPI or Gradio process captures stdout).
    """
    global _HANDLER_ATTACHED  # noqa: PLW0603
    if _HANDLER_ATTACHED:
        return

    root = logging.getLogger("voxguard")
    root.setLevel(_resolve_level())

    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)  # handler passes everything; root level gates it
        formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
        handler.setFormatter(formatter)
        root.addHandler(handler)

    # Prevent log records from propagating to the (often noisy) root logger
    # of the host process (e.g. uvicorn's default root handler).
    root.propagate = False

    _HANDLER_ATTACHED = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a configured ``logging.Logger`` for the given *name*.

    Intended usage in every VoxGuard module::

        from voxguard.utils.logging_utils import get_logger
        logger = get_logger(__name__)

    The returned logger is a child of the top-level ``"voxguard"`` logger,
    so all VoxGuard loggers share the same handler and level configuration.
    If *name* already starts with ``"voxguard"`` it is used as-is; otherwise
    it is prefixed with ``"voxguard."`` to keep the hierarchy tidy.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module, e.g.
        ``"voxguard.embeddings.extractor"``.

    Returns
    -------
    logging.Logger
        A fully configured logger ready to use.
    """
    _configure_root_logger()

    # Normalise the name so all loggers sit under the "voxguard" hierarchy.
    if not name.startswith("voxguard"):
        name = f"voxguard.{name}"

    return logging.getLogger(name)
