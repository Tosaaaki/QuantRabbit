"""
Margin-driven hedge balancer.

マージン使用率が高まったときに逆方向の reduce-only シグナルを signal_bus へ送り、
ネットエクスポージャを軽くしつつ余力を回復する。
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import time
from typing import Optional, Tuple

from market_data import tick_window
from utils import signal_bus
from utils.market_hours import is_market_open
from utils.oanda_account import AccountSnapshot, get_account_snapshot, get_position_summary

from . import config

LOG = logging.getLogger(__name__)


def _latest_mid(fallback: Optional[float] = None) -> Optional[float]:
    ticks = tick_window.recent_ticks(seconds=5.0, limit=5)
    if not ticks:
        return fallback
    mids = []
    for t in ticks:
        try:
            mids.append(float(t.get("mid") or 0.0))
        except Exception:
            continue
    if mids:
        try:
            return sum(mids) / len(mids)
        except Exception:
            return fallback
    return fallback


def _margin_usage(snapshot: AccountSnapshot) -> Tuple[Optional[float], Optional[float]]:
    try:
        total_margin = float(snapshot.margin_available + snapshot.margin_used)
        if total_margin <= 0:
            return None, None
        usage = float(snapshot.margin_used) / total_margin
        return usage, total_margin
    except Exception:
        return None, None


def _plan_units(
    snapshot: AccountSnapshot,
    net_units: int,
    price: float,
) -> Tuple[Optional[int], Optional[str], Optional[float]]:
    usage, total_margin = _margin_usage(snapshot)
    if usage is None or total_margin is None:
        return None, None, usage
    margin_rate = snapshot.margin_rate
    if margin_rate <= 0:
        return None, None, usage
    margin_per_unit = price * margin_rate
    if margin_per_unit <= 0:
        return None, None, usage
    target_usage = min(config.TARGET_MARGIN_USAGE, config.TRIGGER_MARGIN_USAGE)
    target_used = total_margin * target_usage
    target_net_units = target_used / margin_per_unit

    desired_reduction = abs(net_units) - target_net_units
    if desired_reduction <= 0:
        desired_reduction = abs(net_units) * 0.25
    max_reduce = abs(net_units) * config.MAX_REDUCTION_FRACTION
    capped = min(desired_reduction, max_reduce, config.MAX_HEDGE_UNITS)
    final_units = int(max(config.MIN_HEDGE_UNITS, capped))
    if final_units <= 0:
        return None, None, usage
    reason = "margin_usage_high"
    if snapshot.free_margin_ratio is not None and snapshot.free_margin_ratio <= config.TRIGGER_FREE_MARGIN_RATIO:
        reason = "free_margin_low"
    return final_units, reason, usage


async def hedge_balancer_worker() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    if not config.ENABLED:
        LOG.info("%s disabled via config.", config.LOG_PREFIX)
        return
    LOG.info(
        "%s start interval=%.1fs trigger_usage=%.2f target_usage=%.2f free_margin<=%.3f pocket=%s",
        config.LOG_PREFIX,
        config.LOOP_INTERVAL_SEC,
        config.TRIGGER_MARGIN_USAGE,
        config.TARGET_MARGIN_USAGE,
        config.TRIGGER_FREE_MARGIN_RATIO,
        config.POCKET,
    )
    last_action_ts = 0.0

    while True:
        await asyncio.sleep(config.LOOP_INTERVAL_SEC)
        now = datetime.datetime.utcnow()
        if not is_market_open(now):
            continue

        price_hint = _latest_mid()
        if price_hint is None or price_hint < config.MIN_PRICE:
            continue

        try:
            snapshot = get_account_snapshot(cache_ttl_sec=2.0)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("%s snapshot fetch failed: %s", config.LOG_PREFIX, exc)
            continue
        try:
            long_units, short_units = get_position_summary(timeout=4.0)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("%s position fetch failed: %s", config.LOG_PREFIX, exc)
            continue

        net_units = int(round(long_units - short_units))
        abs_net = abs(net_units)
        if abs_net < config.MIN_NET_UNITS:
            continue

        hedge_units, reason, usage = _plan_units(snapshot, net_units, price_hint)
        if hedge_units is None or hedge_units <= 0:
            continue

        trigger_usage = usage is not None and usage >= config.TRIGGER_MARGIN_USAGE
        trigger_free = (
            snapshot.free_margin_ratio is not None
            and snapshot.free_margin_ratio <= config.TRIGGER_FREE_MARGIN_RATIO
        )
        if not (trigger_usage or trigger_free):
            continue

        if time.monotonic() - last_action_ts < config.COOLDOWN_SEC:
            continue

        direction = "OPEN_SHORT" if net_units > 0 else "OPEN_LONG"
        proposed_units = min(hedge_units, abs_net)
        signal_bus.enqueue(
            {
                "strategy": "HedgeBalancer",
                "pocket": config.POCKET,
                "action": direction,
                "confidence": config.CONFIDENCE,
                "sl_pips": config.SL_PIPS,
                "tp_pips": config.TP_PIPS,
                "entry_price": price_hint,
                "reduce_only": True,
                "reduce_cap_units": abs_net,
                "proposed_units": proposed_units,
                "tag": "HedgeBalancer",
                "entry_type": "market",
                "source": "hedge_balancer",
                "meta": {
                    "net_units": net_units,
                    "hedge_units": proposed_units,
                    "margin_usage": usage,
                    "free_margin_ratio": snapshot.free_margin_ratio,
                    "target_usage": config.TARGET_MARGIN_USAGE,
                    "reason": reason,
                },
            }
        )
        last_action_ts = time.monotonic()
        LOG.info(
            "%s enqueue dir=%s units=%d net=%d usage=%.3f free=%.3f reason=%s",
            config.LOG_PREFIX,
            direction,
            proposed_units,
            net_units,
            usage if usage is not None else -1.0,
            snapshot.free_margin_ratio if snapshot.free_margin_ratio is not None else -1.0,
            reason or "unknown",
        )


if __name__ == "__main__":
    asyncio.run(hedge_balancer_worker())
