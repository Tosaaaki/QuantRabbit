"""London session momentum worker."""

from __future__ import annotations

import asyncio
import datetime
import logging
from typing import Optional

from execution.order_manager import market_order
from execution.position_manager import PositionManager
from execution.risk_guard import allowed_lot
from execution.stage_tracker import StageTracker
from indicators.factor_cache import all_factors
from market_data import spread_monitor
from utils.oanda_account import get_account_snapshot

from . import config

LOG = logging.getLogger(__name__)


def _time_in_window(now: datetime.datetime) -> bool:
    start_h, start_m = [int(part) for part in config.SESSION_START_UTC.split(":")]
    end_h, end_m = [int(part) for part in config.SESSION_END_UTC.split(":")]
    start = now.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
    end = now.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
    if end <= start:
        end += datetime.timedelta(days=1)
        if now < start:
            now += datetime.timedelta(days=1)
    return start <= now <= end


async def london_momentum_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled", config.LOG_PREFIX)
        return

    stage_tracker = StageTracker()
    pos_manager = PositionManager()
    try:
        while True:
            now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
            if not _time_in_window(now):
                await asyncio.sleep(max(5.0, config.LOOP_INTERVAL_SEC * 4))
                continue

            factors = all_factors()
            fac_m5 = dict(factors.get("M5") or {})
            fac_h1 = dict(factors.get("H1") or {})
            price = fac_m5.get("close")
            ema20_h1 = fac_h1.get("ema20")
            ema50_h1 = fac_h1.get("ema50") or fac_h1.get("ma50")
            ema20_m5 = fac_m5.get("ema20")
            if any(val is None for val in (price, ema20_h1, ema50_h1, ema20_m5)):
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            try:
                spread_blocked, _, spread_state, spread_reason = spread_monitor.is_blocked()
            except Exception:
                spread_blocked = False
                spread_state = None
                spread_reason = ""
            spread_pips = float((spread_state or {}).get("spread_pips", 0.0) or 0.0)
            if spread_blocked or spread_pips > config.MAX_SPREAD_PIPS:
                if spread_reason:
                    LOG.info("%s spread guard: %s", config.LOG_PREFIX, spread_reason)
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            trend_gap = float(ema20_h1) - float(ema50_h1)
            if abs(trend_gap) < config.TREND_GAP_MIN:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            direction = "long" if trend_gap > 0 else "short"
            momentum = float(price) - float(ema20_m5)
            if direction == "short":
                momentum = -momentum
            if momentum < config.MOMENTUM_MIN:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            atr_pips = fac_m5.get("atr_pips")
            if atr_pips is None:
                atr_raw = fac_m5.get("atr")
                atr_pips = (atr_raw or 0.0) * 100.0
            try:
                atr_pips = float(atr_pips or 0.0)
            except (TypeError, ValueError):
                atr_pips = 0.0
            if atr_pips < config.MIN_ATR_PIPS:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            pockets = pos_manager.get_open_positions()
            pocket_info = pockets.get(config.POCKET) or {}
            if pocket_info.get("open_trades"):
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            blocked, remain, reason = stage_tracker.is_blocked(config.POCKET, direction, now=now)
            if blocked:
                LOG.debug(
                    "%s cooldown pocket=%s dir=%s remain=%s reason=%s",
                    config.LOG_PREFIX,
                    config.POCKET,
                    direction,
                    remain,
                    reason or "cooldown",
                )
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            try:
                snapshot = get_account_snapshot()
                equity = snapshot.nav or snapshot.balance
                margin_available = snapshot.margin_available
                margin_rate = snapshot.margin_rate
            except Exception:
                equity = None
                margin_available = None
                margin_rate = None

            base_equity = equity or pocket_info.get("pocket_equity") or 10_000.0
            lot = allowed_lot(
                base_equity,
                sl_pips=config.SL_PIPS,
                margin_available=margin_available,
                price=price,
                margin_rate=margin_rate,
                risk_pct_override=config.RISK_PCT,
                pocket=config.POCKET,
            )
            units = int(round(lot * 100000))
            if units < config.MIN_UNITS:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue
            if direction == "short":
                units = -units

            sl_delta = config.SL_PIPS * 0.01
            tp_delta = config.TP_PIPS * 0.01
            price_float = float(price)
            if direction == "long":
                sl_price = round(price_float - sl_delta, 3)
                tp_price = round(price_float + tp_delta, 3)
            else:
                sl_price = round(price_float + sl_delta, 3)
                tp_price = round(price_float - tp_delta, 3)

            try:
                trade_id = await market_order(
                    "USD_JPY",
                    units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    pocket=config.POCKET,
                    client_order_id=f"qr-lm-{int(now.timestamp()*1000)}",
                    entry_thesis={
                        "strategy_tag": "LondonMomentum",
                        "trend_gap": round(trend_gap, 5),
                        "momentum": round(momentum, 5),
                        "atr_pips": round(atr_pips, 2),
                        "spread_pips": round(spread_pips, 2),
                    },
                )
            except Exception as exc:  # pragma: no cover
                LOG.error("%s order error dir=%s err=%s", config.LOG_PREFIX, direction, exc)
                trade_id = None

            if not trade_id:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            LOG.info(
                "%s entry trade=%s dir=%s units=%s sl=%.3f tp=%.3f",
                config.LOG_PREFIX,
                trade_id,
                direction,
                units,
                sl_price,
                tp_price,
            )
            pos_manager.register_open_trade(trade_id, config.POCKET)
            stage_tracker.set_stage(config.POCKET, direction, 1, now=now)
            stage_tracker.set_cooldown(
                config.POCKET,
                direction,
                reason="lm_entry",
                seconds=config.COOLDOWN_SEC,
                now=now,
            )

            await asyncio.sleep(config.LOOP_INTERVAL_SEC)
    except asyncio.CancelledError:
        LOG.info("%s worker cancelled", config.LOG_PREFIX)
        raise
    finally:
        try:
            stage_tracker.close()
        except Exception:  # pragma: no cover
            pass
        try:
            pos_manager.close()
        except Exception:  # pragma: no cover
            pass


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(london_momentum_worker())
