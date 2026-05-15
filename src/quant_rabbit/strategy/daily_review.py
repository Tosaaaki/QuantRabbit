"""Generate `data/trader_overrides.json` from yesterday's P&L.

The `daily-review` CLI command runs this module once per day to convert
the previous day's realized P&L (read from `execution_ledger.db`) into
direction-bias overrides + blocked-lane hints for tomorrow's trader.

This module closes the "daily-review → trader 知見伝達の断絶" called
out in `project_v8_postmortem_20260327.md`. Module C in trader_brain
already reads `trader_overrides.json`; this is the producer side.

Heuristics (deterministic, env-tunable):

- **Direction bias**: when (pair, direction) has ≥ N_TRADES_FOR_BIAS
  realized trade outcomes with net P&L beyond ±BIAS_NET_PL_THRESHOLD, emit a
  bias_override scaled `tanh(net_pl / SATURATION) × MAX_BIAS`.
- **Lane block**: when a specific lane_id has ≥ N_LOSSES_FOR_BLOCK
  losing realized outcomes in the lookback window, add it to blocked_lanes.
- **Expiry**: next JST 00:00 in UTC, so the override is valid for the
  current trading day only.

The output is JSON; the reader (`trader_overrides.load_trader_overrides`)
gracefully ignores expired or malformed entries.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Tuple


DAILY_REVIEW_LOOKBACK_HOURS = float(os.environ.get("QR_DAILY_REVIEW_LOOKBACK_HOURS", "24"))
DAILY_REVIEW_N_TRADES_FOR_BIAS = int(os.environ.get("QR_DAILY_REVIEW_N_TRADES_FOR_BIAS", "3"))
DAILY_REVIEW_BIAS_PL_THRESHOLD = float(os.environ.get("QR_DAILY_REVIEW_BIAS_PL_THRESHOLD", "1000"))
DAILY_REVIEW_BIAS_SATURATION = float(os.environ.get("QR_DAILY_REVIEW_BIAS_SATURATION", "3000"))
DAILY_REVIEW_MAX_BIAS = float(os.environ.get("QR_DAILY_REVIEW_MAX_BIAS", "20.0"))
DAILY_REVIEW_N_LOSSES_FOR_BLOCK = int(os.environ.get("QR_DAILY_REVIEW_N_LOSSES_FOR_BLOCK", "3"))


@dataclass
class DailyReviewReport:
    bias_overrides: Dict[str, Dict[str, float]] = field(default_factory=dict)
    blocked_lanes: list[str] = field(default_factory=list)
    narrative_summary: str = ""
    expires_at_utc: str = ""
    source_window_start_utc: str = ""
    source_window_end_utc: str = ""
    pair_pl_breakdown: Dict[str, float] = field(default_factory=dict)
    lane_loss_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {
            "expires_at_utc": self.expires_at_utc,
            "narrative_summary": self.narrative_summary,
            "bias_overrides": self.bias_overrides,
            "blocked_lanes": self.blocked_lanes,
            "_diagnostics": {
                "source_window_start_utc": self.source_window_start_utc,
                "source_window_end_utc": self.source_window_end_utc,
                "pair_direction_net_pl_jpy": {
                    f"{k[0]}:{k[1]}": v for k, v in self.pair_pl_breakdown.items()
                } if all(isinstance(k, tuple) for k in self.pair_pl_breakdown.keys()) else self.pair_pl_breakdown,
                "lane_loss_counts": self.lane_loss_counts,
            },
        }
        return out


def _next_jst_midnight_utc(now: datetime | None = None) -> datetime:
    """Compute the next JST 00:00 in UTC. JST is UTC+9."""
    now = now or datetime.now(timezone.utc)
    jst_now = now + timedelta(hours=9)
    jst_midnight = jst_now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    return jst_midnight - timedelta(hours=9)


def _read_recent_closes(
    db_path: Path,
    window_start_utc: datetime,
    window_end_utc: datetime,
) -> list[tuple[str, str, float, str | None]]:
    """Return (pair, original_side, realized_pl_jpy, lane_id) tuples for
    realized trade outcomes inside the window."""
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    rows: list[tuple[str, str, float, str | None]] = []
    try:
        cur = conn.execute(
            """
            WITH entries AS (
                SELECT
                    trade_id,
                    CASE
                        WHEN MAX(units) > 0 THEN 'LONG'
                        WHEN MIN(units) < 0 THEN 'SHORT'
                        ELSE NULL
                    END AS position_side
                FROM execution_events
                WHERE event_type = 'ORDER_FILLED'
                  AND trade_id IS NOT NULL
                  AND trade_id != ''
                  AND units IS NOT NULL
                GROUP BY trade_id
            ),
            realized AS (
                SELECT
                    COALESCE(NULLIF(e.trade_id, ''), e.event_uid) AS outcome_id,
                    e.ts_utc,
                    e.pair,
                    e.lane_id,
                    COALESCE(
                        entries.position_side,
                        CASE
                            WHEN UPPER(e.side) = 'LONG' THEN 'SHORT'
                            WHEN UPPER(e.side) = 'SHORT' THEN 'LONG'
                            ELSE NULL
                        END
                    ) AS original_side,
                    e.realized_pl_jpy
                FROM execution_events e
                LEFT JOIN entries ON entries.trade_id = e.trade_id
                WHERE e.event_type IN ('TRADE_CLOSED', 'TRADE_REDUCED')
                  AND e.realized_pl_jpy IS NOT NULL
                  AND e.pair IS NOT NULL
                  AND e.ts_utc >= ?
                  AND e.ts_utc <= ?
            )
            SELECT
                pair,
                original_side,
                SUM(realized_pl_jpy) AS realized_pl_jpy,
                lane_id,
                MAX(ts_utc) AS ts_utc
            FROM realized
            WHERE original_side IS NOT NULL
            GROUP BY outcome_id, pair, original_side, lane_id
            ORDER BY ts_utc ASC
            """,
            (
                window_start_utc.isoformat().replace("+00:00", "Z"),
                window_end_utc.isoformat().replace("+00:00", "Z"),
            ),
        )
        for pair, original_side, pl, lane_id, _ts in cur.fetchall():
            rows.append((str(pair), original_side, float(pl), lane_id))
    except sqlite3.Error:
        pass
    finally:
        conn.close()
    return rows


def compute_daily_review(
    db_path: Path,
    *,
    now: datetime | None = None,
    lookback_hours: float = DAILY_REVIEW_LOOKBACK_HOURS,
) -> DailyReviewReport:
    """Build a DailyReviewReport from recent closed trades."""
    now = now or datetime.now(timezone.utc)
    window_end = now
    window_start = now - timedelta(hours=lookback_hours)

    rows = _read_recent_closes(db_path, window_start, window_end)

    # Aggregate per (pair, direction)
    pair_pl: Dict[Tuple[str, str], float] = {}
    pair_count: Dict[Tuple[str, str], int] = {}
    lane_losses: Dict[str, int] = {}
    for pair, direction, pl, lane_id in rows:
        key = (pair, direction)
        pair_pl[key] = pair_pl.get(key, 0.0) + pl
        pair_count[key] = pair_count.get(key, 0) + 1
        if lane_id and pl < 0:
            lane_losses[lane_id] = lane_losses.get(lane_id, 0) + 1

    # Direction bias: per (pair, direction) where count ≥ N and |net_pl| beyond threshold
    bias_overrides: Dict[str, Dict[str, float]] = {}
    for (pair, direction), net_pl in pair_pl.items():
        count = pair_count[(pair, direction)]
        if count < DAILY_REVIEW_N_TRADES_FOR_BIAS:
            continue
        if abs(net_pl) < DAILY_REVIEW_BIAS_PL_THRESHOLD:
            continue
        magnitude = DAILY_REVIEW_MAX_BIAS * math.tanh(abs(net_pl) / DAILY_REVIEW_BIAS_SATURATION)
        sign = 1.0 if net_pl > 0 else -1.0
        bias_overrides.setdefault(pair, {})[direction] = round(sign * magnitude, 2)

    # Blocked lanes
    blocked = [
        lane_id for lane_id, losses in lane_losses.items()
        if losses >= DAILY_REVIEW_N_LOSSES_FOR_BLOCK
    ]

    # Narrative summary (1-2 lines)
    losers = [
        f"{k[0]}:{k[1]} {v:+.0f}JPY"
        for k, v in sorted(pair_pl.items(), key=lambda x: x[1])[:3]
        if v < -DAILY_REVIEW_BIAS_PL_THRESHOLD
    ]
    winners = [
        f"{k[0]}:{k[1]} {v:+.0f}JPY"
        for k, v in sorted(pair_pl.items(), key=lambda x: -x[1])[:3]
        if v > DAILY_REVIEW_BIAS_PL_THRESHOLD
    ]
    parts: list[str] = []
    if losers:
        parts.append("losing: " + ", ".join(losers))
    if winners:
        parts.append("winning: " + ", ".join(winners))
    if not parts:
        parts.append(f"no decisive (pair, direction) signal in last {lookback_hours:.0f}h")
    narrative = "; ".join(parts)

    expires = _next_jst_midnight_utc(now)

    return DailyReviewReport(
        bias_overrides=bias_overrides,
        blocked_lanes=blocked,
        narrative_summary=narrative,
        expires_at_utc=expires.isoformat().replace("+00:00", "Z"),
        source_window_start_utc=window_start.isoformat().replace("+00:00", "Z"),
        source_window_end_utc=window_end.isoformat().replace("+00:00", "Z"),
        pair_pl_breakdown={k: round(v, 2) for k, v in pair_pl.items()},
        lane_loss_counts=lane_losses,
    )


def write_trader_overrides(
    report: DailyReviewReport,
    output_path: Path,
) -> None:
    """Serialize the report to `data/trader_overrides.json` shape.

    The reader (`strategy/trader_overrides.load_trader_overrides`)
    ignores the `_diagnostics` key, so we keep it for human review
    without affecting trader behavior.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
