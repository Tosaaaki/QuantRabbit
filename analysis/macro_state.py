"""
analysis.macro_state
~~~~~~~~~~~~~~~~~~~~
Centralises macro-level context such as yield differentials, risk metrics,
news-derived currency bias, and upcoming event windows so that the trading loop
can make consistent gating and sizing decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import datetime as dt
import json


@dataclass(slots=True)
class MacroSnapshot:
    asof: str
    vix: float
    dxy: float
    yield2y: Dict[str, float]
    news_ccy_score: Dict[str, float]
    events: List[Tuple[str, str, str]]


class MacroState:
    """Lightweight snapshot-backed service with helper methods."""

    def __init__(self, snap: MacroSnapshot, *, deadzone: float = 0.25):
        self._snap = snap
        self._deadzone = max(0.0, deadzone)

    @staticmethod
    def load_json(path: str | Path, *, deadzone: float = 0.25) -> "MacroState":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        snap = MacroSnapshot(**data)
        return MacroState(snap, deadzone=deadzone)

    @property
    def snapshot(self) -> MacroSnapshot:
        return self._snap

    @property
    def deadzone(self) -> float:
        return self._deadzone

    def bias(self, pair: str) -> float:
        """Return signed bias in [-1, 1]; deadzone collapses small magnitudes to 0."""
        clean = "".join(ch for ch in pair.upper() if ch.isalpha())
        if len(clean) != 6:
            return 0.0
        base, quote = clean[:3], clean[3:]
        carry = (
            self._snap.yield2y.get(base, 0.0) - self._snap.yield2y.get(quote, 0.0)
        ) / 3.0
        news = (
            self._snap.news_ccy_score.get(base, 0.0)
            - self._snap.news_ccy_score.get(quote, 0.0)
        ) * 0.5
        score = carry + news
        if self._snap.vix >= 25.0 and score > 0.0:
            score -= 0.3
        score = max(-1.0, min(1.0, score))
        if abs(score) < self._deadzone:
            return 0.0
        return score

    def in_event_window(
        self,
        pair: str,
        *,
        before_hours: float = 2.0,
        after_hours: float = 1.0,
        now: dt.datetime | None = None,
    ) -> bool:
        """True if pair trades during configured event window."""
        if not self._snap.events:
            return False
        now = now or dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
        clean = "".join(ch for ch in pair.upper() if ch.isalpha())
        if len(clean) != 6:
            return False
        base, quote = clean[:3], clean[3:]
        for iso_ts, ccy, _ in self._snap.events:
            if ccy not in (base, quote):
                continue
            try:
                event_ts = dt.datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
            except ValueError:
                continue
            window_start = event_ts - dt.timedelta(hours=before_hours)
            window_end = event_ts + dt.timedelta(hours=after_hours)
            if window_start <= now <= window_end:
                return True
        return False
