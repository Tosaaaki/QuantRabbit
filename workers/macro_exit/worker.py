"""Dedicated exit loop for the macro pocket."""

from __future__ import annotations

import asyncio
import datetime
import logging
from typing import Dict, List

from analysis import policy_bus
from execution.exit_manager import ExitManager
from execution.order_ids import build_client_order_id
from execution.order_manager import close_trade, market_order
from execution.position_manager import PositionManager
from execution.stage_tracker import StageTracker
from indicators.factor_cache import all_factors

from . import config

LOG = logging.getLogger(__name__)
POCKET = "macro"
PIP = 0.01


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _make_signals(fac_h4: dict, fac_h1: dict, pocket_policy: dict | None) -> List[dict]:
    """Macro bias signals for ExitManager reverse detection."""
    signals: List[dict] = []
    bias = (pocket_policy or {}).get("bias")
    conf_bias = int(round(config.CONFIDENCE_BIAS * 100))
    conf_neutral = int(round(config.CONFIDENCE_NEUTRAL * 100))

    def _add(action: str, confidence: int, tag: str) -> None:
        signals.append(
            {
                "pocket": POCKET,
                "action": action,
                "confidence": max(0, min(100, confidence)),
                "strategy": tag,
                "tag": tag,
            }
        )

    if bias == "long":
        _add("OPEN_LONG", conf_bias, "bias_long")
    elif bias == "short":
        _add("OPEN_SHORT", conf_bias, "bias_short")

    try:
        ma10 = float(fac_h4.get("ma10") or fac_h4.get("ema10") or 0.0)
        ma20 = float(fac_h4.get("ma20") or fac_h4.get("ema20") or 0.0)
        rsi = float(fac_h4.get("rsi") or fac_h1.get("rsi") or 50.0)
    except Exception:
        ma10 = ma20 = rsi = 0.0
    gap_pips = abs(ma10 - ma20) / PIP if ma10 and ma20 else 0.0
    if gap_pips >= config.MA_GAP_MIN_PIPS:
        if ma10 > ma20:
            _add("OPEN_LONG", max(conf_neutral, conf_bias), "ma_bias_long")
        elif ma20 > ma10:
            _add("OPEN_SHORT", max(conf_neutral, conf_bias), "ma_bias_short")
    if rsi >= config.RSI_LONG:
        _add("OPEN_LONG", conf_neutral, "rsi_long")
    if rsi <= config.RSI_SHORT:
        _add("OPEN_SHORT", conf_neutral, "rsi_short")
    return signals


async def _execute_exit(
    decision,
    pos_manager: PositionManager,
    stage_tracker: StageTracker,
    now: datetime.datetime,
) -> None:
    positions = pos_manager.get_open_positions()
    pocket_info = positions.get(POCKET) or {}
    remaining = abs(decision.units)
    target_side = "long" if decision.units < 0 else "short"
    trades = [t for t in (pocket_info.get("open_trades") or []) if t.get("side") == target_side]
    for tr in trades:
        if remaining <= 0:
            break
        trade_units = abs(int(tr.get("units", 0) or 0))
        if trade_units == 0:
            continue
        close_amount = min(remaining, trade_units)
        sign = 1 if target_side == "long" else -1
        trade_id = tr.get("trade_id")
        if not trade_id:
            continue
        ok = await close_trade(trade_id, sign * close_amount)
        if ok:
            LOG.info(
                "%s exit trade=%s pocket=%s units=%s reason=%s",
                config.LOG_PREFIX,
                trade_id,
                POCKET,
                sign * close_amount,
                decision.reason,
            )
            remaining -= close_amount
    if remaining > 0:
        client_id = build_client_order_id("macro_exit", decision.tag)
        fallback_units = -remaining if decision.units < 0 else remaining
        trade_id = await market_order(
            "USD_JPY",
            fallback_units,
            None,
            None,
            POCKET,
            client_order_id=client_id,
            reduce_only=True,
        )
        if trade_id:
            LOG.info(
                "%s exit-fallback trade=%s pocket=%s units=%s reason=%s",
                config.LOG_PREFIX,
                trade_id,
                POCKET,
                fallback_units,
                decision.reason,
            )
            remaining = 0
        else:
            LOG.error(
                "%s exit-fallback failed pocket=%s units=%s reason=%s",
                config.LOG_PREFIX,
                POCKET,
                decision.units,
                decision.reason,
            )
    if remaining <= 0:
        cooldown_seconds = 240
        stage_tracker.set_cooldown(
            POCKET,
            target_side,
            reason=decision.reason,
            seconds=cooldown_seconds,
            now=now,
        )
        if decision.reason == "reverse_signal":
            opposite = "short" if target_side == "long" else "long"
            stage_tracker.set_cooldown(
                POCKET,
                opposite,
                reason="flip_guard",
                seconds=min(360, max(120, cooldown_seconds // 2)),
                now=now,
            )


async def macro_exit_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled", config.LOG_PREFIX)
        return

    LOG.info("%s worker starting (interval %.2fs)", config.LOG_PREFIX, config.LOOP_INTERVAL_SEC)
    exit_manager = ExitManager()
    stage_tracker = StageTracker()
    pos_manager = PositionManager()
    try:
        while True:
            await asyncio.sleep(config.LOOP_INTERVAL_SEC)
            now = _utc_now()
            factors = all_factors()
            fac_m1 = factors.get("M1") or {}
            fac_h4 = factors.get("H4") or {}
            fac_h1 = factors.get("H1") or {}
            if not fac_h4.get("close") and not fac_m1.get("close"):
                continue
            if fac_m1 and not fac_m1.get("close") and fac_h4.get("close"):
                fac_m1 = {"close": fac_h4.get("close")}

            positions = pos_manager.get_open_positions()
            macro_info = positions.get(POCKET) or {}
            trades = macro_info.get("open_trades") or []
            if not trades:
                continue

            stage_tracker.clear_expired(now)
            stage_state = {
                POCKET: {
                    "long": stage_tracker.get_stage(POCKET, "long"),
                    "short": stage_tracker.get_stage(POCKET, "short"),
                }
            }
            clamp_state = stage_tracker.get_clamp_state(now=now)

            policy_snapshot = policy_bus.latest()
            pocket_policy: Dict[str, object] | None = None
            if policy_snapshot and getattr(policy_snapshot, "pockets", None):
                pocket_policy = policy_snapshot.pockets.get(POCKET)
            range_mode = bool((policy_snapshot.range_mode if policy_snapshot else False) or fac_h4.get("range_active"))

            signals = _make_signals(fac_h4, fac_h1, pocket_policy if isinstance(pocket_policy, dict) else None)
            decisions = exit_manager.plan_closures(
                {POCKET: macro_info},
                signals,
                fac_m1,
                fac_h4,
                fac_h1=fac_h1,
                event_soon=False,
                range_mode=range_mode,
                stage_state=stage_state,
                now=now,
                stage_tracker=stage_tracker,
                clamp_state=clamp_state,
            )
            for decision in decisions:
                if decision.pocket != POCKET:
                    continue
                await _execute_exit(decision, pos_manager, stage_tracker, now)
    except asyncio.CancelledError:
        LOG.info("%s worker cancelled", config.LOG_PREFIX)
        raise
    finally:
        try:
            pos_manager.close()
        except Exception:  # pragma: no cover - best-effort cleanup
            LOG.exception("%s failed to close PositionManager", config.LOG_PREFIX)
        try:
            stage_tracker.close()
        except Exception:
            LOG.exception("%s failed to close StageTracker", config.LOG_PREFIX)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    asyncio.run(macro_exit_worker())
