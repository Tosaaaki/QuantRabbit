"""Entry loop for MicroTrendRetest (micro pocket)."""

from __future__ import annotations

import asyncio
import datetime
import logging
import time
from typing import Optional, Tuple

from execution.order_ids import build_client_order_id
from execution.order_manager import market_order
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp
from indicators.factor_cache import all_factors, refresh_cache_from_disk
from market_data import tick_window
from strategies.micro.trend_retest import MicroTrendRetest
from utils.market_hours import is_market_open
from utils.metrics_logger import log_metric
from utils.oanda_account import get_account_snapshot, get_position_summary
from workers.common import perf_guard
from workers.common.size_utils import scale_base_units

from . import config

LOG = logging.getLogger(__name__)
PIP = 0.01


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _factor_age_seconds(factors: dict) -> float:
    ts_raw = factors.get("timestamp") if isinstance(factors, dict) else None
    if not ts_raw:
        return float("inf")
    try:
        if isinstance(ts_raw, (int, float)):
            ts_val = float(ts_raw)
            ts_dt = datetime.datetime.utcfromtimestamp(ts_val).replace(tzinfo=datetime.timezone.utc)
        else:
            ts_txt = str(ts_raw)
            if ts_txt.endswith("Z"):
                ts_txt = ts_txt.replace("Z", "+00:00")
            ts_dt = datetime.datetime.fromisoformat(ts_txt)
            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=datetime.timezone.utc)
    except Exception:
        return float("inf")
    now = _utc_now()
    return max(0.0, (now - ts_dt).total_seconds())


def _latest_quote() -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    try:
        ticks = tick_window.recent_ticks(seconds=2.0, limit=1)
    except Exception:
        ticks = None
    if not ticks:
        return None, None, None, None
    tick = ticks[-1]
    try:
        bid = float(tick.get("bid")) if tick.get("bid") is not None else None
        ask = float(tick.get("ask")) if tick.get("ask") is not None else None
        mid = float(tick.get("mid")) if tick.get("mid") is not None else None
    except Exception:
        return None, None, None, None
    lag_ms = None
    try:
        epoch = float(tick.get("epoch") or 0.0)
        if epoch > 0:
            lag_ms = max(0.0, (time.time() - epoch) * 1000.0)
    except Exception:
        lag_ms = None
    return bid, ask, mid, lag_ms


def _spread_pips(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return None
    return max(0.0, (ask - bid) / PIP)


def _confidence_scale(conf: int) -> float:
    lo = int(config.CONFIDENCE_FLOOR)
    hi = int(config.CONFIDENCE_CEIL)
    if conf <= lo:
        return 0.55
    if conf >= hi:
        return 1.0
    span = (conf - lo) / max(1.0, hi - lo)
    return 0.55 + span * 0.45


async def micro_trendretest_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled (idle)", config.LOG_PREFIX)
        try:
            while True:
                await asyncio.sleep(3600.0)
        except asyncio.CancelledError:
            return

    LOG.info(
        "%s worker start (interval=%.1fs tf=%s instrument=%s dry_run=%s)",
        config.LOG_PREFIX,
        config.LOOP_INTERVAL_SEC,
        config.SIGNAL_TF,
        config.INSTRUMENT,
        config.DRY_RUN,
    )

    last_entry_mono = 0.0
    last_signal_key: str | None = None
    last_block_log_mono = 0.0
    last_heartbeat_mono = 0.0

    try:
        while True:
            await asyncio.sleep(config.LOOP_INTERVAL_SEC)

            now_mono = time.monotonic()
            market_open = is_market_open()
            trade_ok = can_trade(config.POCKET)
            if now_mono - last_heartbeat_mono > 120.0:
                LOG.info("%s heartbeat market_open=%s can_trade=%s", config.LOG_PREFIX, market_open, trade_ok)
                last_heartbeat_mono = now_mono

            if not market_open:
                continue
            if not trade_ok:
                continue

            now = _utc_now()
            current_hour = now.hour
            if config.BLOCK_HOURS_UTC and current_hour in config.BLOCK_HOURS_UTC:
                continue

            try:
                refresh_cache_from_disk()
            except Exception:
                pass
            factors = all_factors()
            fac_signal = factors.get(config.SIGNAL_TF) or {}
            fac_m1 = factors.get("M1") or {}
            fac_h4 = factors.get("H4") or {}

            factor_age = _factor_age_seconds(fac_signal)
            if factor_age > config.MAX_FACTOR_AGE_SEC:
                log_metric(
                    "micro_trt_skip",
                    float(factor_age),
                    tags={"reason": "factor_stale", "tf": str(config.SIGNAL_TF)},
                    ts=now,
                )
                continue

            bid, ask, mid, tick_lag_ms = _latest_quote()
            if tick_lag_ms is not None and tick_lag_ms > config.MAX_TICK_LAG_MS:
                log_metric(
                    "micro_trt_skip",
                    float(tick_lag_ms),
                    tags={"reason": "tick_lag"},
                    ts=now,
                )
                continue

            spread_pips = _spread_pips(bid, ask)
            if spread_pips is not None and spread_pips > config.MAX_SPREAD_PIPS:
                log_metric(
                    "micro_trt_skip",
                    float(spread_pips),
                    tags={"reason": "spread"},
                    ts=now,
                )
                continue

            price = mid
            if price is None or price <= 0:
                try:
                    price = float(fac_signal.get("close") or 0.0)
                except Exception:
                    price = None
            if price is None or price <= 0:
                continue

            fac_eval = dict(fac_signal)
            fac_eval["close"] = price
            signal = MicroTrendRetest.check(fac_eval)
            if not signal:
                continue

            perf_decision = perf_guard.is_allowed(MicroTrendRetest.name, config.POCKET)
            if not perf_decision.allowed:
                now_mono = time.monotonic()
                if now_mono - last_block_log_mono > 120.0:
                    LOG.info(
                        "%s perf_block tag=%s reason=%s",
                        config.LOG_PREFIX,
                        MicroTrendRetest.name,
                        perf_decision.reason,
                    )
                    last_block_log_mono = now_mono
                continue

            signal_action = str(signal.get("action") or "")
            side = "long" if signal_action == "OPEN_LONG" else "short"
            signal_tag = str(signal.get("tag") or MicroTrendRetest.name)

            fac_ts = fac_signal.get("timestamp")
            signal_key = f"{config.SIGNAL_TF}:{fac_ts}:{signal_tag}"

            if config.MIN_ENTRY_INTERVAL_SEC > 0 and (now_mono - last_entry_mono) < config.MIN_ENTRY_INTERVAL_SEC:
                continue
            if last_signal_key and signal_key == last_signal_key:
                continue

            sl_pips = float(signal.get("sl_pips") or 0.0)
            tp_pips = float(signal.get("tp_pips") or 0.0)
            if sl_pips <= 0.0:
                continue

            snap = get_account_snapshot()
            equity = float(snap.nav or snap.balance or 0.0)
            balance = float(snap.balance or snap.nav or 0.0)

            base_units = scale_base_units(
                int(config.BASE_ENTRY_UNITS),
                equity=balance if balance > 0 else equity,
                ref_equity=balance if balance > 0 else None,
                min_units=int(config.MIN_UNITS),
            )

            conf = int(signal.get("confidence", 0) or 0)
            conf_scale = _confidence_scale(conf)

            long_units = 0.0
            short_units = 0.0
            try:
                long_units, short_units = get_position_summary(config.INSTRUMENT, timeout=2.5)
            except Exception:
                long_units, short_units = 0.0, 0.0

            lot = allowed_lot(
                equity,
                sl_pips,
                margin_available=float(snap.margin_available or 0.0),
                margin_used=float(snap.margin_used or 0.0),
                price=price,
                margin_rate=float(snap.margin_rate or 0.0),
                pocket=config.POCKET,
                side=side,
                open_long_units=long_units,
                open_short_units=short_units,
                strategy_tag=signal_tag,
                fac_m1=fac_m1,
                fac_h4=fac_h4,
            )
            units_risk = int(round(lot * 100000))
            units = int(round(base_units * conf_scale))
            units = min(units, units_risk)
            if units < config.MIN_UNITS:
                continue
            if side == "short":
                units = -abs(units)

            if side == "long":
                sl_price = round(price - sl_pips * PIP, 3)
                tp_price = round(price + tp_pips * PIP, 3) if tp_pips > 0 else None
            else:
                sl_price = round(price + sl_pips * PIP, 3)
                tp_price = round(price - tp_pips * PIP, 3) if tp_pips > 0 else None

            sl_price, tp_price = clamp_sl_tp(
                price=price,
                sl=sl_price,
                tp=tp_price,
                is_buy=side == "long",
            )

            client_id = build_client_order_id(config.POCKET, signal_tag)
            entry_thesis: dict[str, object] = {
                "strategy_tag": signal_tag,
                "signal_action": signal_action,
                "signal_side": side,
                "profile": signal.get("profile"),
                "confidence": conf,
                "tp_pips": tp_pips,
                "sl_pips": sl_pips,
                "hard_stop_pips": sl_pips,
                "signal_tf": config.SIGNAL_TF,
                "factor_age_sec": round(float(factor_age), 3),
            }
            if tick_lag_ms is not None:
                entry_thesis["tick_lag_ms"] = round(float(tick_lag_ms), 1)
            if spread_pips is not None:
                entry_thesis["spread_pips"] = round(float(spread_pips), 2)

            if config.DRY_RUN:
                tp_txt = f"{tp_price:.3f}" if tp_price is not None else "None"
                LOG.info(
                    "%s DRY_RUN signal=%s side=%s units=%s price=%.3f sl=%.3f tp=%s",
                    config.LOG_PREFIX,
                    signal_tag,
                    side,
                    units,
                    price,
                    sl_price or 0.0,
                    tp_txt,
                )
                last_entry_mono = now_mono
                last_signal_key = signal_key
                continue

            try:
                ticket = await market_order(
                    config.INSTRUMENT,
                    units,
                    sl_price,
                    tp_price,
                    config.POCKET,
                    client_order_id=client_id,
                    strategy_tag=signal_tag,
                    entry_thesis=entry_thesis,
                    confidence=conf,
                )
            except Exception:
                LOG.exception("%s order failed tag=%s", config.LOG_PREFIX, signal_tag)
                continue

            if ticket:
                LOG.info(
                    "%s OPEN ok trade=%s tag=%s side=%s units=%s price=%.3f",
                    config.LOG_PREFIX,
                    ticket,
                    signal_tag,
                    side,
                    units,
                    price,
                )
                log_metric(
                    "micro_trt_open",
                    1.0,
                    tags={"side": side, "tf": str(config.SIGNAL_TF)},
                    ts=now,
                )
                last_entry_mono = now_mono
                last_signal_key = signal_key
    except asyncio.CancelledError:
        LOG.info("%s worker cancelled", config.LOG_PREFIX)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    asyncio.run(micro_trendretest_worker())
