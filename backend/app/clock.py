"""Injectable clock for domain time.

Application code never calls datetime.now() for domain timestamps. The API uses
SystemClock; the simulator uses SimulatedClock to fast-forward weeks in seconds.
"""

from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class SimulatedClock:
    def __init__(self, start: datetime):
        if start.tzinfo is None:
            raise ValueError("SimulatedClock needs a timezone-aware start time")
        self._now = start

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        self._now += delta


_system_clock = SystemClock()


def get_clock() -> Clock:
    """FastAPI dependency; tests override it with a SimulatedClock."""
    return _system_clock
