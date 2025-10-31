"""Pullback continuation scalp worker."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from typing import Optional

from execution.order_manager import market_order
from execution.position_manager import PositionManager
from market_data import spread_monitor, tick_window

from . import config

LOG = logging.getLogger(__name__)


def _client_id(side: str) -> str:
    ts_ms = int(time.time() * 1000)
    digest = hashlib.sha1(f"{ts_ms}-{side}".encode("utf-8")).hexdigest()[:6]
    return f"qr-pullback-{ts_ms}-{side[0]}{digest}"


def _z_score(values: list[float]) -> Optional[float]:
    if len(values) < 20:
        return None
    sample = values[-20:]
    mean_val = sum(sample) / len(sample)
    var = sum((v - mean_val) ** 2 for v in sample) / max(len(sample) - 1, 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    return (sample[-1] - mean_val) / std


def _rsi(values: list[float], period: int) -> Optional[float]:
    if len(values) <= 1:
        return None
    gains = []
    losses = []
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        if diff >= 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-diff)
    period = min(period, len(gains))
    if period <= 0:
        return 50.0
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0.0:
        return 100.0
    if avg_gain == 0.0:
        return 0.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(values: list[float], period: int) -> float:
    if len(values) <= 1:
        return 0.0
    true_ranges = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
    period = min(period, len(true_ranges))
    if period <= 0:
        return 0.0
    return (sum(true_ranges[-period:]) / period) / config.PIP_VALUE


async def pullback_scalp_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled", config.LOG_PREFIX)
        return

    LOG.info("%s worker starting", config.LOG_PREFIX)
    pos_manager = PositionManager()
    cooldown_until = 0.0
    last_spread_log = 0.0
    try:
        while True:
            await asyncio.sleep(config.LOOP_INTERVAL_SEC)
            now_monotonic = time.monotonic()
            if now_monotonic < cooldown_until:
                continue

            blocked, _, spread_state, spread_reason = spread_monitor.is_blocked()
            spread_pips = float((spread_state or {}).get("spread_pips", 0.0) or 0.0)
            if blocked or spread_pips > config.MAX_SPREAD_PIPS:
                if now_monotonic - last_spread_log > 30.0:
                    LOG.info(
                        "%s spread gate active spread=%.2fp reason=%s",
                        config.LOG_PREFIX,
                        spread_pips,
                        spread_reason or "guard",
                    )
                    last_spread_log = now_monotonic
                continue

            ticks_m1 = tick_window.recent_ticks(seconds=config.M1_WINDOW_SEC, limit=900)
            ticks_m5 = tick_window.recent_ticks(seconds=config.M5_WINDOW_SEC, limit=1800)
            if len(ticks_m1) < 20 or len(ticks_m5) < 40:
                continue

            mids_m1 = [float(t.get("mid")) for t in ticks_m1]
            mids_m5 = [float(t.get("mid")) for t in ticks_m5]
            z_m1 = _z_score(mids_m1)
            z_m5 = _z_score(mids_m5)
            if z_m1 is None or z_m5 is None:
                continue
            rsi_m1 = _rsi(mids_m1, config.RSI_PERIOD)
            atr_m1 = _atr(mids_m1, min(12, max(6, config.RSI_PERIOD)))

            side: Optional[str] = None
            if config.M1_Z_MIN <= z_m1 <= config.M1_Z_MAX and z_m5 <= config.M5_Z_SHORT_MAX:
                if rsi_m1 is None or config.RSI_SHORT_RANGE[0] <= rsi_m1 <= config.RSI_SHORT_RANGE[1]:
                    side = "short"
            elif -config.M1_Z_MAX <= z_m1 <= -config.M1_Z_MIN and z_m5 >= config.M5_Z_LONG_MIN:
                if rsi_m1 is None or config.RSI_LONG_RANGE[0] <= rsi_m1 <= config.RSI_LONG_RANGE[1]:
                    side = "long"
            if side is None:
                continue

            pockets = pos_manager.get_open_positions()
            scalp_pos = pockets.get("scalp") or {}
            tagged = []
            for tr in scalp_pos.get("open_trades") or []:
                thesis = tr.get("entry_thesis") or {}
                if thesis.get("strategy_tag") == "pullback_scalp":
                    tagged.append(tr)
            if tagged:
                latest_tr = sorted(tagged, key=lambda tr: tr.get("open_time") or "")[-1]
                prev_price = float(latest_tr.get("price") or 0.0)
                if prev_price:
                    delta = abs(prev_price - mids_m1[-1]) / config.PIP_VALUE
                    if delta < config.STAGE_MIN_DELTA_PIPS:
                        continue
            if len(tagged) >= config.MAX_ACTIVE_TRADES:
                continue

            latest_tick = ticks_m1[-1]
            entry_price = float(latest_tick.get("ask")) if side == "long" else float(latest_tick.get("bid"))

            tp_pips = config.TP_PIPS
            tp_price = round(
                entry_price + tp_pips * config.PIP_VALUE if side == "long" else entry_price - tp_pips * config.PIP_VALUE,
                3,
            )
            sl_price = None
            if config.USE_INITIAL_SL:
                sl_pips = max(config.MIN_SL_PIPS, atr_m1 * config.SL_ATR_MULT)
                sl_price = round(
                    entry_price - sl_pips * config.PIP_VALUE if side == "long" else entry_price + sl_pips * config.PIP_VALUE,
                    3,
                )

            units = config.ENTRY_UNITS if side == "long" else -config.ENTRY_UNITS
            client_id = _client_id(side)
            thesis = {
                "strategy_tag": "pullback_scalp",
                "z_m1": None if z_m1 is None else round(z_m1, 2),
                "z_m5": None if z_m5 is None else round(z_m5, 2),
                "rsi_m1": None if rsi_m1 is None else round(rsi_m1, 1),
                "atr_m1_pips": round(atr_m1, 2),
                "spread_pips": round(spread_pips, 2),
            }

            try:
                trade_id, _ = await market_order(
                    "USD_JPY",
                    units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    pocket="scalp",
                    client_order_id=client_id,
                    entry_thesis=thesis,
                )
            except Exception as exc:
                LOG.error("%s order error side=%s exc=%s", config.LOG_PREFIX, side, exc)
                cooldown_until = now_monotonic + config.COOLDOWN_SEC
                continue

            if trade_id:
                LOG.info(
                    "%s entry trade=%s side=%s units=%s tp=%.3f z1=%.2f z5=%.2f rsi=%.1f",
                    config.LOG_PREFIX,
                    trade_id,
                    side,
                    units,
                    tp_price,
                    z_m1 or 0.0,
                    z_m5 or 0.0,
                    rsi_m1 or -1.0,
                )
                cooldown_until = now_monotonic + config.COOLDOWN_SEC
            else:
                cooldown_until = now_monotonic + 10.0
    except asyncio.CancelledError:
        LOG.info("%s worker cancelled", config.LOG_PREFIX)
        raise
    finally:
        try:
            pos_manager.close()
        except Exception:
            LOG.exception("%s failed to close PositionManager", config.LOG_PREFIX)
