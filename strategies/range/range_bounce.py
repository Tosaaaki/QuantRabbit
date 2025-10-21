from __future__ import annotations

import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

PIP = 0.01  # USD/JPY

_DB_PATH = Path("logs/trades.db")
try:
    _DB_PATH.parent.mkdir(exist_ok=True)
    _DB_CONN = sqlite3.connect(_DB_PATH)
except Exception:
    _DB_CONN = None

_GUARD_LOSS_PIPS = max(0.0, float(os.getenv("RANGE_BOUNCE_RECENT_GUARD_PIPS", "6.0")))
_GUARD_COOLDOWN_MINUTES = max(0.0, float(os.getenv("RANGE_BOUNCE_RECENT_GUARD_MINUTES", "45.0")))
_GUARD_MAX_TRADES = max(1, int(os.getenv("RANGE_BOUNCE_RECENT_GUARD_MAX_TRADES", "5")))
_GUARD_SINGLE_LOSS_PIPS = max(0.0, float(os.getenv("RANGE_BOUNCE_COOLDOWN_LOSS_PIPS", "2.2")))
_GUARD_CACHE_SEC = max(0.0, float(os.getenv("RANGE_BOUNCE_GUARD_CACHE_SEC", "15.0")))
_GUARD_CACHE = {"ts": 0.0, "block": False}


@dataclass
class RangeSignal:
    action: str
    sl_pips: float
    tp_pips: float


class RangeBounce:
    name = "RangeBounce"
    pocket = "micro"

    @staticmethod
    def check(
        fac_m1: Dict[str, float],
        fac_h4: Dict[str, float],
        micro_regime: str,
        macro_regime: str,
    ) -> Dict[str, float] | None:
        if _exhausted_recent_losses():
            return None

        if micro_regime not in ("Range", "Mixed"):
            return None

        close = fac_m1.get("close")
        ma20 = fac_m1.get("ma20")
        bbw = fac_m1.get("bbw")
        atr = fac_m1.get("atr")
        if None in (close, ma20, bbw, atr):
            return None

        close = float(close)
        ma20 = float(ma20)
        bbw = float(bbw)
        atr = float(atr)
        atr_pips = max(atr / PIP, 0.1)

        # ブレイク仕掛け中はスキップ
        bbw_h4 = float(fac_h4.get("bbw", 1.0) or 1.0)
        if macro_regime == "Breakout" or bbw_h4 > 0.6:
            return None

        deviation_pips = (close - ma20) / PIP
        threshold = max(0.35, atr_pips * 0.2)
        far_threshold = threshold * 1.1

        if deviation_pips <= -threshold:
            sl = max(1.5, atr_pips * 0.5)
            tp = max(2.2, sl * 1.35)
            return {"action": "buy", "sl_pips": sl, "tp_pips": tp}
        if deviation_pips >= threshold:
            sl = max(1.5, atr_pips * 0.5)
            tp = max(2.2, sl * 1.35)
            return {"action": "sell", "sl_pips": sl, "tp_pips": tp}
        return None


def _exhausted_recent_losses() -> bool:
    if (_GUARD_LOSS_PIPS <= 0 and _GUARD_SINGLE_LOSS_PIPS <= 0) or _DB_CONN is None:
        return False

    now = time.time()
    if _GUARD_CACHE_SEC > 0 and now - _GUARD_CACHE["ts"] < _GUARD_CACHE_SEC:
        return bool(_GUARD_CACHE["block"])

    block = False
    try:
        fetch_limit = max(_GUARD_MAX_TRADES * 3, 10)
        query = (
            "SELECT pl_pips, close_time FROM trades "
            "WHERE strategy=? AND pocket='micro' AND pl_pips IS NOT NULL "
            "AND state='CLOSED' AND close_time IS NOT NULL "
            "ORDER BY datetime(close_time) DESC LIMIT ?"
        )
        rows = _DB_CONN.execute(query, (RangeBounce.name, fetch_limit)).fetchall()
        loss_sum = 0.0
        losses = []
        cutoff = None
        if _GUARD_COOLDOWN_MINUTES > 0:
            cutoff = datetime.utcnow() - timedelta(minutes=_GUARD_COOLDOWN_MINUTES)

        for row in rows:
            pl = float(row[0])
            ts_raw = row[1]
            try:
                close_ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
            except Exception:
                close_ts = None
            if close_ts and cutoff and close_ts < cutoff:
                continue
            if pl >= 0:
                continue
            losses.append((pl, close_ts))
            loss_sum += abs(pl)
            if len(losses) >= _GUARD_MAX_TRADES:
                break

        if losses:
            cumulative = loss_sum >= _GUARD_LOSS_PIPS > 0
            single = any(abs(pl) >= _GUARD_SINGLE_LOSS_PIPS > 0 for pl, _ in losses)
            block = cumulative or single
            if block:
                last_ts = losses[0][1].isoformat() if losses[0][1] else "unknown"
                logging.info(
                    "[RANGE_BOUNCE_GUARD] cooldown count=%s loss_sum=%.2f last_close=%s",
                    len(losses),
                    loss_sum,
                    last_ts,
                )
    except Exception as exc:  # noqa: BLE001
        logging.warning("[RANGE_BOUNCE_GUARD] failed to evaluate guard: %s", exc)
        block = False

    _GUARD_CACHE.update({"ts": now, "block": block})
    return block
