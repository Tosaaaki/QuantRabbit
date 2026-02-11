"""Simple sliding-window rate limiter shared across workers."""

from __future__ import annotations

import time
from collections import deque
from typing import Deque


class SlidingWindowRateLimiter:
    def __init__(self, max_per_minute: int, min_interval_sec: float) -> None:
        self._max_per_minute = max(1, max_per_minute)
        self._min_interval_sec = max(0.0, min_interval_sec)
        self._events: Deque[float] = deque()
        self._last_event: float = 0.0

    def allow(self, now: float | None = None) -> bool:
        ts = now if now is not None else time.monotonic()
        while self._events and ts - self._events[0] > 60.0:
            self._events.popleft()
        if self._events and ts - self._events[-1] < self._min_interval_sec:
            return False
        if len(self._events) >= self._max_per_minute:
            return False
        return True

    def record(self, now: float | None = None) -> None:
        ts = now if now is not None else time.monotonic()
        self._events.append(ts)
        self._last_event = ts

    @property
    def since_last(self) -> float:
        if not self._last_event:
            return float("inf")
        return time.monotonic() - self._last_event
