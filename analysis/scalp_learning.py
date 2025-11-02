from __future__ import annotations

"""Utilities to compute performance-based adjustments for scalping."""

import sqlite3
import pathlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from analysis.scalp_config import load_overrides, store_overrides

_DB_PATH = pathlib.Path("logs/trades.db")


@dataclass
class ScalpStats:
    trades: int
    wins: int
    losses: int
    sum_pips: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0

    @property
    def avg_pips(self) -> float:
        return self.sum_pips / self.trades if self.trades else 0.0


@dataclass
class ScalpAdjustments:
    lot_multiplier: float = 1.0
    deviation_offset: float = 0.0
    disable_mean_revert: bool = False
    disable_minutes: int = 0
    note: str = ""


def _connect() -> sqlite3.Connection | None:
    if not _DB_PATH.exists():
        return None
    try:
        return sqlite3.connect(_DB_PATH)
    except Exception:
        return None


def load_stats(lookback_minutes: int = 720) -> ScalpStats:
    """Aggregate scalp pocket performance over the recent window."""

    con = _connect()
    if con is None:
        return ScalpStats(trades=0, wins=0, losses=0, sum_pips=0.0)

    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)).isoformat(timespec="seconds")
    try:
        rows = con.execute(
            "SELECT pl_pips FROM trades WHERE pocket=? AND close_time >= ?",
            ("scalp", cutoff),
        ).fetchall()
    except Exception:
        return ScalpStats(trades=0, wins=0, losses=0, sum_pips=0.0)
    finally:
        con.close()

    trades = 0
    wins = 0
    losses = 0
    sum_pips = 0.0
    for (pl,) in rows:
        if pl is None:
            continue
        trades += 1
        try:
            value = float(pl)
        except (TypeError, ValueError):
            continue
        sum_pips += value
        if value > 0:
            wins += 1
        elif value < 0:
            losses += 1
    return ScalpStats(trades=trades, wins=wins, losses=losses, sum_pips=sum_pips)


LOSS_SUM_HARD = -80.0
LOSS_SUM_SOFT = -40.0
MIN_TRADES = 12
DISABLE_MINUTES_HARD = 360
DISABLE_MINUTES_SOFT = 180
RECOVER_SUM = 15.0


def compute_adjustments(stats: ScalpStats) -> ScalpAdjustments:
    """Derive runtime adjustments based on performance."""

    if stats.trades < MIN_TRADES:
        return ScalpAdjustments(note="insufficient_data")

    lot_multiplier = 1.0
    deviation_offset = 0.0
    disable_mean_revert = False
    disable_minutes = 0
    note = "baseline"

    if stats.win_rate >= 0.6 and stats.avg_pips >= 2.0 and stats.sum_pips >= 25.0:
        lot_multiplier = 1.2
        deviation_offset = -0.4
        note = "outperform"
    elif stats.sum_pips <= LOSS_SUM_HARD or stats.avg_pips <= -1.6:
        lot_multiplier = 0.35
        deviation_offset = 1.0
        disable_mean_revert = True
        disable_minutes = DISABLE_MINUTES_HARD
        note = "loss_hard"
    elif stats.sum_pips <= LOSS_SUM_SOFT or stats.avg_pips <= -1.1:
        lot_multiplier = 0.7
        deviation_offset = 0.5
        disable_mean_revert = True
        disable_minutes = DISABLE_MINUTES_SOFT
        note = "loss_soft"
    elif stats.sum_pips >= RECOVER_SUM and stats.win_rate >= 0.52:
        lot_multiplier = 1.05
        deviation_offset = -0.2
        note = "recover"

    lot_multiplier = max(0.5, min(lot_multiplier, 1.4))
    deviation_offset = max(-1.0, min(deviation_offset, 1.0))

    if disable_mean_revert:
        lot_multiplier = min(lot_multiplier, 0.7)

    return ScalpAdjustments(
        lot_multiplier=lot_multiplier,
        deviation_offset=deviation_offset,
        disable_mean_revert=disable_mean_revert,
        disable_minutes=disable_minutes,
        note=note,
    )


def _parse_iso(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        if dt_str.endswith("Z"):
            dt_str = dt_str[:-1] + "+00:00"
        return datetime.fromisoformat(dt_str)
    except Exception:
        return None


def _persist_learning(stats: ScalpStats, adj: ScalpAdjustments) -> None:
    try:
        overrides = load_overrides()
    except Exception:
        return

    changed = False
    now = datetime.now(timezone.utc)

    learning_payload = {
        "last_updated": now.isoformat(timespec="seconds"),
        "last_trades": stats.trades,
        "last_sum_pips": round(stats.sum_pips, 2),
        "win_rate": round(stats.win_rate, 4),
        "avg_pips": round(stats.avg_pips, 3),
        "lot_multiplier": round(adj.lot_multiplier, 3),
        "deviation_offset": round(adj.deviation_offset, 3),
        "disable_mean_revert": adj.disable_mean_revert,
        "note": adj.note,
    }

    existing_learning = overrides.get("learning") if isinstance(overrides.get("learning"), dict) else {}
    compare_new = dict(learning_payload)
    compare_new.pop("last_updated", None)
    compare_old = dict(existing_learning)
    compare_old.pop("last_updated", None)
    if compare_old != compare_new:
        overrides["learning"] = learning_payload
        changed = True
    elif "last_updated" not in existing_learning:
        overrides["learning"] = learning_payload
        changed = True

    disable_until: Optional[datetime] = None
    if adj.disable_mean_revert:
        minutes = max(30, adj.disable_minutes or DISABLE_MINUTES_SOFT)
        disable_until = now + timedelta(minutes=minutes)
        until_iso = disable_until.isoformat(timespec="seconds")
        if not overrides.get("mean_revert_disabled") or overrides.get("mean_revert_disabled_until") != until_iso:
            overrides["mean_revert_disabled"] = True
            overrides["mean_revert_disabled_until"] = until_iso
            changed = True
    else:
        existing_disable_until = _parse_iso(overrides.get("mean_revert_disabled_until"))
        if overrides.get("mean_revert_disabled"):
            if existing_disable_until and existing_disable_until > now and stats.sum_pips <= LOSS_SUM_SOFT:
                # keep disabled until timer expires
                pass
            else:
                overrides["mean_revert_disabled"] = False
                overrides.pop("mean_revert_disabled_until", None)
                changed = True

    if changed:
        store_overrides(overrides)


def load_adjustments(lookback_minutes: int = 720) -> Dict[str, float]:
    stats = load_stats(lookback_minutes=lookback_minutes)
    adj = compute_adjustments(stats)
    _persist_learning(stats, adj)
    return {
        "trades": stats.trades,
        "win_rate": stats.win_rate,
        "avg_pips": stats.avg_pips,
        "lot_multiplier": adj.lot_multiplier,
        "deviation_offset": adj.deviation_offset,
        "disable_mean_revert": adj.disable_mean_revert,
        "note": adj.note,
    }
