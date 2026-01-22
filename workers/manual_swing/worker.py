"""Replicate the July 2025 manual swing behaviour with automated guardrails."""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import math
import time
from typing import Dict, Iterable, List, Optional, Tuple

from execution.order_manager import market_order
from execution.position_manager import PositionManager
from execution.risk_guard import can_trade, clamp_sl_tp, loss_cooldown_status
from indicators.factor_cache import all_factors
from market_data import spread_monitor, tick_window
from utils.market_hours import is_market_open
from utils.oanda_account import get_account_snapshot
from workers.common.quality_gate import current_regime

from . import config

LOG = logging.getLogger(__name__)
STRATEGY_TAG = "manual_swing"


def _client_order_id(suffix: str) -> str:
    ts_ms = int(time.time() * 1000)
    return f"qr-manual-{ts_ms}-{suffix}"


def _latest_mid(fallback: float) -> float:
    """Fallback-safe mid price from recent tick."""
    ticks = tick_window.recent_ticks(seconds=15.0, limit=1)
    if ticks:
        tick = ticks[-1]
        try:
            bid = float(tick.get("bid"))
            ask = float(tick.get("ask"))
            if math.isfinite(bid) and math.isfinite(ask) and bid > 0 and ask > 0:
                return (bid + ask) * 0.5
        except (TypeError, ValueError):
            pass
        mid = tick.get("mid")
        if mid is not None:
            try:
                mid_f = float(mid)
                if math.isfinite(mid_f) and mid_f > 0:
                    return mid_f
            except (TypeError, ValueError):
                pass
    return fallback


def _stage_thresholds(stages: Iterable[int]) -> List[int]:
    thresholds: List[int] = []
    total = 0
    for units in stages:
        total += abs(int(units))
        thresholds.append(total)
    return thresholds


def _scaled_stage_units(equity: float) -> List[int]:
    if equity <= 0:
        equity = config.REFERENCE_EQUITY
    scale = equity / config.REFERENCE_EQUITY if config.REFERENCE_EQUITY else 1.0
    units: List[int] = []
    for base in config.STAGE_UNITS_BASE[: config.MAX_ACTIVE_STAGES]:
        scaled = int(round(base * scale))
        units.append(max(config.MIN_STAGE_UNITS, scaled))
    return units


def _parse_iso(ts: Optional[str]) -> Optional[dt.datetime]:
    if not ts:
        return None
    try:
        parsed = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _calc_sl_tp(
    side: str,
    price: float,
    sl_pips: float,
    tp_pips: float,
) -> Tuple[Optional[float], Optional[float]]:
    pip = config.PIP_VALUE
    if side == "long":
        sl = price - sl_pips * pip
        tp = price + tp_pips * pip
    else:
        sl = price + sl_pips * pip
        tp = price - tp_pips * pip
    return round(sl, 3), round(tp, 3)


def _trend_bias(
    fac_h1: Dict[str, float],
    fac_h4: Dict[str, float],
) -> Tuple[Optional[str], Dict[str, float]]:
    features: Dict[str, float] = {}
    try:
        ma10_h4 = float(fac_h4.get("ma10"))
        ma20_h4 = float(fac_h4.get("ma20"))
        features["ma_gap_h4"] = ma10_h4 - ma20_h4
    except (TypeError, ValueError):
        features["ma_gap_h4"] = 0.0
    try:
        ma10_h1 = float(fac_h1.get("ma10"))
        ma20_h1 = float(fac_h1.get("ma20"))
        features["ma_gap_h1"] = ma10_h1 - ma20_h1
    except (TypeError, ValueError):
        features["ma_gap_h1"] = 0.0
    try:
        adx = float(fac_h1.get("adx"))
    except (TypeError, ValueError):
        adx = 0.0
    features["adx"] = adx
    direction: Optional[str] = None
    if (
        features["ma_gap_h4"] >= config.H4_GAP_MIN
        and features["ma_gap_h1"] >= config.H1_GAP_MIN
        and adx >= config.ADX_MIN
    ):
        direction = "long"
    elif (
        features["ma_gap_h4"] <= -config.H4_GAP_MIN
        and features["ma_gap_h1"] <= -config.H1_GAP_MIN
        and adx >= config.ADX_MIN
    ):
        direction = "short"
    return direction, features


def _open_position_summary(
    pocket_state: Dict[str, object]
) -> Tuple[int, Optional[dt.datetime]]:
    net_units = int(pocket_state.get("units") or 0)
    open_trades: List[Dict[str, object]] = pocket_state.get("open_trades") or []
    if open_trades:
        oldest = min(
            (_parse_iso(trade.get("open_time")) for trade in open_trades),
            key=lambda x: x or dt.datetime.now(dt.timezone.utc),
        )
    else:
        oldest = None
    return net_units, oldest


def _hold_hours(opened_at: Optional[dt.datetime], now: dt.datetime) -> Optional[float]:
    if not opened_at:
        return None
    delta = now - opened_at
    return delta.total_seconds() / 3600.0


async def manual_swing_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled", config.LOG_PREFIX)
        return

    if config.POCKET not in {"micro", "macro", "scalp"}:
        LOG.error(
            "%s invalid pocket=%s; allowed only micro/macro/scalp",
            config.LOG_PREFIX,
            config.POCKET,
        )
        return

    LOG.info("%s worker starting (interval=%.1fs)", config.LOG_PREFIX, config.LOOP_INTERVAL_SEC)
    pos_manager = PositionManager()
    stage_cooldown_until: Dict[str, float] = {"long": 0.0, "short": 0.0}
    last_sync_perf = 0.0
    skip_state: Dict[str, float] = {"ts": 0.0, "reason": ""}
    position_tracker: Dict[str, object] = {
        "side": None,
        "best_pips": 0.0,
        "last_stage_price": None,
        "stage_count": 0,
    }

    try:
        while True:
            await asyncio.sleep(config.LOOP_INTERVAL_SEC)
            now_mono = time.monotonic()
            now_utc = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)

            if config.ALLOWED_HOURS_UTC and now_utc.hour not in config.ALLOWED_HOURS_UTC:
                continue
            if config.BLOCKED_WEEKDAYS:
                if str(now_utc.weekday()) in config.BLOCKED_WEEKDAYS:
                    continue

            if not is_market_open(now_utc):
                continue

            if not can_trade(config.POCKET):
                continue

            loss_block, loss_remain = loss_cooldown_status(
                config.POCKET,
                max_losses=config.LOSS_STREAK_MAX,
                cooldown_minutes=config.LOSS_STREAK_COOLDOWN_MIN,
            )
            if loss_block:
                if config.LOG_SKIP_REASON and now_mono - skip_state.get("ts", 0.0) > 120.0:
                    LOG.info(
                        "%s cooldown in effect pocket=%s remaining=%.0fs",
                        config.LOG_PREFIX,
                        config.POCKET,
                        loss_remain,
                    )
                    skip_state["ts"] = now_mono
                continue

            blocked, recovery, spread_state, reason = spread_monitor.is_blocked()
            spread_pips = float((spread_state or {}).get("spread_pips") or 0.0)
            if blocked or spread_pips > config.SPREAD_MAX_PIPS:
                continue
            if recovery and spread_pips > config.SPREAD_RECOVERY_PIPS:
                continue

            regime = current_regime()
            if regime and regime.lower() in {"event", "halt"}:
                continue

            factors = all_factors()
            fac_h1 = factors.get("H1")
            fac_h4 = factors.get("H4")
            if not fac_h1 or not fac_h4:
                continue
            direction, features = _trend_bias(fac_h1, fac_h4)
            if direction is None:
                continue

            candles_h1 = fac_h1.get("candles") or []
            try:
                last_close = float(candles_h1[-1]["close"])
            except Exception:
                last_close = float(fac_h1.get("ma10") or 0.0)
            price = _latest_mid(last_close or 150.0)

            atr_pips = float(fac_h1.get("atr_pips") or 0.0)
            if atr_pips < config.ATR_MIN_PIPS:
                continue

            snapshot = get_account_snapshot(timeout=6.0)
            if (
                snapshot.free_margin_ratio is not None
                and snapshot.free_margin_ratio < config.MIN_FREE_MARGIN_RATIO
            ):
                LOG.warning(
                    "%s free margin %.2f below limit %.2f – skipping entries",
                    config.LOG_PREFIX,
                    snapshot.free_margin_ratio,
                    config.MIN_FREE_MARGIN_RATIO,
                )
                direction = None

            pocket_positions = pos_manager.get_open_positions()
            pocket_state: Dict[str, object] = pocket_positions.get(config.POCKET, {})
            net_units, oldest_trade = _open_position_summary(pocket_state)
            sign = 1 if net_units >= 0 else -1
            hold_hours = _hold_hours(oldest_trade, now_utc)

            if net_units == 0:
                position_tracker["side"] = None
                position_tracker["best_pips"] = 0.0
                position_tracker["last_stage_price"] = None
                position_tracker["stage_count"] = 0

            stage_units = _scaled_stage_units(snapshot.nav or snapshot.balance or config.REFERENCE_EQUITY)
            if not stage_units:
                continue
            max_stage_count = min(config.MAX_ACTIVE_STAGES, len(stage_units))

            # Flatten if regime flips or hold exceeds max
            should_exit = False
            exit_reason = ""
            if net_units != 0:
                active_side = "long" if net_units > 0 else "short"
                if direction and active_side != direction:
                    should_exit = True
                    exit_reason = "trend_flip"
                elif hold_hours and hold_hours >= config.MAX_HOLD_HOURS:
                    should_exit = True
                    exit_reason = "max_hold"
                elif (
                    snapshot.health_buffer is not None
                    and snapshot.health_buffer < config.MARGIN_HEALTH_EXIT
                ):
                    should_exit = True
                    exit_reason = "margin_health"
                else:
                    # drawdown guard: compare price vs entry avg
                    avg_price = pocket_state.get("avg_price")
                    if avg_price:
                        try:
                            avg_price_val = float(avg_price)
                            favourable_pips = (
                                (price - avg_price_val) / config.PIP_VALUE
                                if net_units > 0
                                else (avg_price_val - price) / config.PIP_VALUE
                            )
                            if favourable_pips < -config.MAX_DRAWDOWN_PIPS:
                                should_exit = True
                                exit_reason = "max_drawdown"
                            else:
                                current_side = "long" if net_units > 0 else "short"
                                if position_tracker.get("side") != current_side:
                                    position_tracker["side"] = current_side
                                    position_tracker["best_pips"] = 0.0
                                best_pips = float(position_tracker.get("best_pips") or 0.0)
                                if favourable_pips > best_pips:
                                    position_tracker["best_pips"] = favourable_pips
                                    best_pips = favourable_pips
                                if best_pips >= config.PROFIT_TRIGGER_PIPS:
                                    should_exit = True
                                    exit_reason = "profit_lock"
                                elif (
                                    best_pips >= config.TRAIL_TRIGGER_PIPS
                                    and favourable_pips
                                    <= best_pips - config.TRAIL_BACKOFF_PIPS
                                ):
                                    should_exit = True
                                    exit_reason = "trail_backoff"
                                elif (
                                    features.get("ma_gap_h1") is not None
                                    and (
                                        (
                                            net_units > 0
                                            and features["ma_gap_h1"]
                                            < -config.REVERSAL_GAP_EXIT
                                        )
                                        or (
                                            net_units < 0
                                            and features["ma_gap_h1"]
                                            > config.REVERSAL_GAP_EXIT
                                        )
                                    )
                                ):
                                    should_exit = True
                                    exit_reason = "gap_reversal"
                        except Exception:
                            pass

            if should_exit and net_units != 0:
                units_to_close = -net_units
                LOG.info(
                    "%s closing position units=%s reason=%s",
                    config.LOG_PREFIX,
                    units_to_close,
                    exit_reason,
                )
                await market_order(
                    "USD_JPY",
                    units_to_close,
                    sl_price=None,
                    tp_price=None,
                    pocket=config.POCKET,  # type: ignore[arg-type]
                    reduce_only=True,
                    client_order_id=_client_order_id("close"),
                    strategy_tag=STRATEGY_TAG,
                    entry_thesis={"exit_reason": exit_reason, "strategy_tag": STRATEGY_TAG},
                )
                stage_cooldown_until["long"] = now_mono + 300.0
                stage_cooldown_until["short"] = now_mono + 300.0
                position_tracker["side"] = None
                position_tracker["best_pips"] = 0.0
                position_tracker["last_stage_price"] = None
                position_tracker["stage_count"] = 0
                continue

            if direction is None:
                continue

            # If in cooldown for this direction, skip new entries
            if now_mono < stage_cooldown_until[direction]:
                continue

            current_stage_count = int(position_tracker.get("stage_count") or 0)
            if net_units != 0:
                position_tracker["side"] = "long" if net_units > 0 else "short"
                inferred_stage = 0
                cumulative = 0
                abs_units = abs(net_units)
                for size in stage_units:
                    cumulative += size
                    if abs_units >= cumulative * 0.6:
                        inferred_stage += 1
                if inferred_stage > current_stage_count:
                    current_stage_count = min(inferred_stage, max_stage_count)
                    position_tracker["stage_count"] = current_stage_count
                    if position_tracker.get("last_stage_price") is None:
                        try:
                            position_tracker["last_stage_price"] = float(pocket_state.get("avg_price"))
                        except Exception:
                            position_tracker["last_stage_price"] = price

            if current_stage_count >= max_stage_count:
                continue

            stage_size = stage_units[current_stage_count]
            if stage_size <= 0:
                continue

            if current_stage_count > 0 and position_tracker.get("side") == direction:
                anchor_price = position_tracker.get("last_stage_price")
                if anchor_price is None:
                    anchor_price = pocket_state.get("avg_price") or price
                try:
                    anchor_price_val = float(anchor_price)
                    favourable_move = (
                        (price - anchor_price_val) / config.PIP_VALUE
                        if direction == "long"
                        else (anchor_price_val - price) / config.PIP_VALUE
                    )
                except Exception:
                    favourable_move = 0.0
                if favourable_move < config.STAGE_ADD_TRIGGER_PIPS:
                    continue
            elif net_units != 0:
                continue

            sl_pips = max(config.MIN_SL_PIPS, atr_pips * config.SL_ATR_MULT)
            tp_pips = max(config.MIN_TP_PIPS, atr_pips * config.TP_ATR_MULT)

            # Determine units based on free margin usage
            margin_available = snapshot.margin_available
            margin_rate = snapshot.margin_rate
            if margin_available <= 0 or margin_rate <= 0:
                continue
            leverage_budget = margin_available * config.RISK_FREE_MARGIN_FRACTION
            units_budget = int(leverage_budget / (price * margin_rate))
            units_budget = max(units_budget, 0)
            if units_budget <= 0:
                continue
            incremental_units = min(stage_size, units_budget)
            if incremental_units < config.MIN_STAGE_UNITS:
                continue

            side = direction
            units_to_send = incremental_units if side == "long" else -incremental_units
            sl_price, tp_price = _calc_sl_tp(side, price, sl_pips, tp_pips)
            sl_price, tp_price = clamp_sl_tp(price, sl_price, tp_price, side.upper())

            entry_meta = {
                "stage": current_stage_count + 1,
                "atr_pips": atr_pips,
                "sl_pips": sl_pips,
                "tp_pips": tp_pips,
                "hard_stop_pips": sl_pips,
                "stage_units": stage_size,
                "ma_gap_h1": features["ma_gap_h1"],
                "ma_gap_h4": features["ma_gap_h4"],
                "adx": features["adx"],
            }
            LOG.info(
                "%s opening stage=%s units=%s dir=%s price=%.3f",
                config.LOG_PREFIX,
                current_stage_count + 1,
                units_to_send,
                side,
                price,
            )
            await market_order(
                "USD_JPY",
                units_to_send,
                sl_price=sl_price,
                tp_price=tp_price,
                pocket=config.POCKET,  # type: ignore[arg-type]
                client_order_id=_client_order_id(f"{side}-stage{current_stage_count+1}"),
                reduce_only=False,
                strategy_tag=STRATEGY_TAG,
                entry_thesis={**entry_meta, "strategy_tag": STRATEGY_TAG},
            )
            stage_cooldown_until[direction] = now_mono + config.STAGE_COOLDOWN_MINUTES * 60.0
            position_tracker["side"] = side
            position_tracker["best_pips"] = float(position_tracker.get("best_pips") or 0.0)
            position_tracker["last_stage_price"] = price
            if current_stage_count == 0:
                position_tracker["best_pips"] = 0.0
            position_tracker["stage_count"] = current_stage_count + 1

            if now_mono - last_sync_perf >= config.PERF_SYNC_INTERVAL_SEC:
                try:
                    pos_manager.sync_trades()
                except Exception as exc:  # noqa: BLE001
                    LOG.exception("%s sync failed: %s", config.LOG_PREFIX, exc)
                last_sync_perf = now_mono
    finally:
        pos_manager.close()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(manual_swing_worker())
