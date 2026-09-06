"""Running exponential moving average for streaming risk scores."""

from __future__ import annotations


class RunningRiskScore:
    """Tracks a smoothed risk score across a streaming session."""

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = float(alpha)
        self._running: float | None = None

    def update(self, new_score: float | None) -> float | None:
        """Update the running score with a new value and return the current average."""
        if new_score is None:
            return self._running

        if self._running is None:
            self._running = float(new_score)
        else:
            self._running = (
                self.alpha * float(new_score) + (1.0 - self.alpha) * self._running
            )

        return self._running

    def current(self) -> float | None:
        """Return the current running score without changing state."""
        return self._running

    def reset(self) -> None:
        """Clear the accumulated running score for a new session."""
        self._running = None
