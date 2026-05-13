"""Daily-review → trader_brain feedback loop.

Reads `data/trader_overrides.json` written by `daily-review` (or the
operator manually) and exposes per-pair / per-direction score
modifiers + hard blocks to trader_brain scoring.

Format of `data/trader_overrides.json`:
```json
{
  "date_jst": "2026-05-13",
  "expires_at_utc": "2026-05-14T00:00:00+00:00",
  "bias_overrides": {
    "GBP_USD": {"LONG": -25.0, "SHORT": 0.0},
    "EUR_USD": {"LONG": -15.0}
  },
  "blocked_lanes": [
    "failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE"
  ],
  "narrative_summary": "USD strong post-CPI 3.8%; risk-off; fade USD-pair LONG"
}
```

Expiry semantics: when `expires_at_utc` is past the current UTC, the
overrides are ignored. This avoids dragging yesterday's regime call
into a new session. Missing file → empty overrides → no behavior change.

Why this matters: the 2026-05-13 incident showed the trader trading
the wrong direction for hours because last night's daily-review insight
("USD strong, fade LONG") never reached this morning's cycle. A static
JSON read at trader_brain bootstrap is the simplest reliable bridge.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


TRADER_OVERRIDES_FILENAME = "trader_overrides.json"


@dataclass(frozen=True)
class TraderOverrides:
    bias_modifiers: Dict[tuple[str, str], float]  # (pair, direction) -> delta
    blocked_lane_ids: frozenset[str]
    narrative_summary: str
    expires_at_utc: Optional[datetime]
    source_path: Optional[Path]

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        if self.expires_at_utc is None:
            return False
        now = now or datetime.now(timezone.utc)
        return now >= self.expires_at_utc


def _parse_expiry(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def load_trader_overrides(data_root: Path) -> TraderOverrides:
    """Read trader_overrides.json. Returns empty overrides on any error."""
    path = data_root / TRADER_OVERRIDES_FILENAME
    empty = TraderOverrides(
        bias_modifiers={},
        blocked_lane_ids=frozenset(),
        narrative_summary="",
        expires_at_utc=None,
        source_path=None,
    )
    if not path.exists():
        return empty
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return empty
    if not isinstance(payload, dict):
        return empty

    expires = _parse_expiry(payload.get("expires_at_utc"))
    bias_raw = payload.get("bias_overrides") or {}
    bias: Dict[tuple[str, str], float] = {}
    if isinstance(bias_raw, dict):
        for pair, dirmap in bias_raw.items():
            if not isinstance(dirmap, dict):
                continue
            for direction, delta in dirmap.items():
                try:
                    bias[(str(pair), str(direction).upper())] = float(delta)
                except (TypeError, ValueError):
                    continue
    blocked = frozenset(
        str(x) for x in (payload.get("blocked_lanes") or []) if isinstance(x, str)
    )
    narrative = str(payload.get("narrative_summary") or "")[:500]

    overrides = TraderOverrides(
        bias_modifiers=bias,
        blocked_lane_ids=blocked,
        narrative_summary=narrative,
        expires_at_utc=expires,
        source_path=path,
    )
    if overrides.is_expired():
        return empty
    return overrides


def overrides_score_delta(
    overrides: TraderOverrides,
    pair: str,
    direction: str,
) -> tuple[float, str | None]:
    """Lookup additive score delta for a (pair, direction)."""
    key = (pair, direction.upper())
    delta = overrides.bias_modifiers.get(key, 0.0)
    if delta == 0.0:
        return 0.0, None
    sign = "+" if delta >= 0 else ""
    rationale = f"trader_overrides {pair}:{direction} {sign}{delta:.1f} (from daily-review)"
    return delta, rationale


def overrides_block_check(
    overrides: TraderOverrides, lane_id: str
) -> tuple[bool, str | None]:
    """Check whether a lane_id is in the blocked list."""
    if lane_id in overrides.blocked_lane_ids:
        return True, f"trader_overrides blocked {lane_id} (from daily-review)"
    return False, None
