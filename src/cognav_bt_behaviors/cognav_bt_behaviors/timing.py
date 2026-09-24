"""Freshness deadline derived from the safety margin.

A command computed from a frame stays in force until the next command. The
exposure window is W = age + T_h, where `age` is the frame age at consumption
(now - header.stamp, which already includes capture and inference latency) and
T_h is the hold time of a command. Requiring v_max * W <= M gives

    age <= M / v_max - T_h

which `ExposureBudget.deadline_sec` returns. `RuntimeBuffer` uses it to judge
freshness and the mission loop uses it as the wait bound before commanding
zero. The floor keeps jitter from driving the deadline to zero; reaching it
means the configuration cannot satisfy the bound and max_v must drop or the
margin grow.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ExposureBudget:
    """Derives the freshness deadline from the margin it defends."""

    safety_margin_m: float
    max_v: float
    deadline_floor_sec: float = 0.10

    @property
    def total_sec(self) -> float:
        """M / v_max: the whole exposure window the margin can cover."""
        return self.safety_margin_m / max(self.max_v, 1e-6)

    def deadline_sec(self, hold_sec: float) -> float:
        """Maximum tolerable frame age given how long a command stays in force."""
        return max(self.deadline_floor_sec, self.total_sec - max(hold_sec, 0.0))

    def is_exhausted(self, hold_sec: float) -> bool:
        """True when the hold time alone consumes the whole budget."""
        return self.total_sec - max(hold_sec, 0.0) < self.deadline_floor_sec

    def travel_m(self, window_sec: float) -> float:
        """Distance covered at v_max over a window."""
        return self.max_v * max(window_sec, 0.0)


class PeriodEstimator:
    """Median of recent inter-frame gaps, used as the hold time T_h.

    A median, so the single long gap spanning an outage does not move it.
    """

    def __init__(self, window: int = 9) -> None:
        self._gaps: deque[float] = deque(maxlen=window)
        self._previous: float | None = None

    def observe(self, stamp_sec: float) -> None:
        """Feed one frame timestamp. Out-of-order or repeated stamps are ignored."""
        if self._previous is not None and stamp_sec > self._previous:
            self._gaps.append(stamp_sec - self._previous)
        if self._previous is None or stamp_sec > self._previous:
            self._previous = stamp_sec

    @property
    def period_sec(self) -> float | None:
        """Estimated frame period, or None until two frames have been seen."""
        return statistics.median(self._gaps) if self._gaps else None

    def reset(self) -> None:
        self._gaps.clear()
        self._previous = None
