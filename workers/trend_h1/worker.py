"""H1 trend-following worker leveraging the MovingAverageCross strategy."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import time
from typing import Dict, Optional

from indicators.factor_cache import all_factors
from execution.order_manager import market_order
from execution.position_manager import PositionManager
from execution.risk_guard import (
    allowed_lot,
    can_trade,
    clamp_sl_tp,
    loss_cooldown_status,
)
from market_data import spread_monitor, tick_window
from strategies.trend.ma_cross import MovingAverageCross
from utils.market_hours import is_market_open
from utils.oanda_account import get_account_snapshot
from workers.common.quality_gate import current_regime

from . import config

LOG = logging.getLogger(__name__)
PIP = 0.01


def _parse_iso8601(value: str) -> Optional[datetime.datetime]:
    """Decode ISO8601 timestamps produced by factor cache."""
    try:
        dt = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


def _latest_mid(fallback: float) -> float:
    """Pick the most recent mid price from tick cache, fallback to provided value."""
    ticks = tick_window.recent_ticks(seconds=20.0, limit=1)
    if ticks:
        tick = ticks[-1]
        mid_val = tick.get("mid")
        if mid_val is not None:
            try:
                return float(mid_val)
            except (TypeError, ValueError):
                pass
        bid = tick.get("bid")
        ask = tick.get("ask")
        if bid is not None and ask is not None:
            try:
                return (float(bid) + float(ask)) / 2.0
            except (TypeError, ValueError):
                return fallback
    return fallback


def _confidence_scale(confidence: int) -> float:
    """Map confidence score (0-100) into a lot multiplier."""
    floor = config.CONFIDENCE_FLOOR
    ceil = config.CONFIDENCE_CEIL
    if ceil <= floor:
        return config.MAX_CONFIDENCE_SCALE
    if confidence <= floor:
        return config.MIN_CONFIDENCE_SCALE
    if confidence >= ceil:
        return config.MAX_CONFIDENCE_SCALE
    ratio = (confidence - floor) / float(ceil - floor)
    span = config.MAX_CONFIDENCE_SCALE - config.MIN_CONFIDENCE_SCALE
    return config.MIN_CONFIDENCE_SCALE + max(0.0, min(1.0, ratio)) * span


def _client_order_id(tag: str) -> str:
    """Generate a quasi-unique client order id that follows QuantRabbit conventions."""
    ts_ms = int(time.time() * 1000)
    sanitized = "".join(ch.lower() for ch in tag if ch.isalnum())[:8] or "trend"
    digest = hashlib.sha1(f"{ts_ms}-{tag}".encode("utf-8")).hexdigest()[:9]
    return f"qr-{ts_ms}-macro-{sanitized}{digest}"


def _log_skip(reason: str, state: Dict[str, float]) -> None:
    """Throttle skip logging so the worker does not spam the logs."""
    if not config.LOG_SKIP_REASON:
        return
    now = time.monotonic()
    last_reason = state.get("reason", "")
    last_ts = state.get("ts", 0.0)
    if reason == last_reason and now - last_ts < 60.0:
        return
    LOG.debug("%s skip: %s", config.LOG_PREFIX, reason)
    state["reason"] = reason
    state["ts"] = now


def _direction_allowed(
    fac_h1: Dict[str, float],
    fac_h4: Optional[Dict[str, float]],
    direction: str,
    atr_pips: float,
) -> bool:
    """Require higher timeframe alignment before taking the trade."""
    if not fac_h4:
        return True
    ma10_h4 = fac_h4.get("ma10")
    ma20_h4 = fac_h4.get("ma20")
    if ma10_h4 is None or ma20_h4 is None:
        return True
    gap_h4 = float(ma10_h4) - float(ma20_h4)
    bias_buffer = 0.00018
    override_allowed = False
    if direction == "long" and gap_h4 < -bias_buffer:
        ma10_h1 = fac_h1.get("ma10")
        ma20_h1 = fac_h1.get("ma20")
        if ma10_h1 is not None and ma20_h1 is not None:
            gap_h1 = (float(ma10_h1) - float(ma20_h1)) / PIP
            if (
                gap_h1 >= config.H1_OVERRIDE_GAP_PIPS
                and atr_pips >= config.H1_OVERRIDE_ATR_PIPS
            ):
                override_allowed = True
        if not override_allowed:
            return False
    if direction == "short" and gap_h4 > bias_buffer:
        ma10_h1 = fac_h1.get("ma10")
        ma20_h1 = fac_h1.get("ma20")
        if ma10_h1 is not None and ma20_h1 is not None:
            gap_h1 = (float(ma10_h1) - float(ma20_h1)) / PIP
            if (
                gap_h1 <= -config.H1_OVERRIDE_GAP_PIPS
                and atr_pips >= config.H1_OVERRIDE_ATR_PIPS
            ):
                override_allowed = True
        if not override_allowed:
            return False
    ma10_h1 = fac_h1.get("ma10")
    ma20_h1 = fac_h1.get("ma20")
    if ma10_h1 is None or ma20_h1 is None:
        return True
    gap_h1 = float(ma10_h1) - float(ma20_h1)
    micro_buffer = 0.00006
    if direction == "long" and gap_h1 < -micro_buffer:
        return False
    if direction == "short" and gap_h1 > micro_buffer:
        return False
    return True


async def trend_h1_worker() -> None:
    """Async loop that scans H1 factors and places macro-pocket trend trades."""
    if not config.ENABLED:
        LOG.info("%s disabled via configuration", config.LOG_PREFIX)
        return

    LOG.info(
        "%s worker starting (interval=%.1fs)",
        config.LOG_PREFIX,
        config.LOOP_INTERVAL_SEC,
    )
    pos_manager = PositionManager()
    cooldown_until = 0.0
    last_direction_entry: Dict[str, float] = {"long": 0.0, "short": 0.0}
    recent_signal_gate: Dict[str, float] = {}
    skip_state: Dict[str, float] = {"reason": "", "ts": 0.0}

    try:
        while True:
            try:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                now_mono = time.monotonic()
                if now_mono < cooldown_until:
                    continue

                now_utc = datetime.datetime.utcnow()
                if not is_market_open(now_utc):
                    _log_skip("market_closed", skip_state)
                    continue

                if not can_trade(config.POCKET):
                    _log_skip("pocket_drawdown_guard", skip_state)
                    continue

                loss_block, loss_remain = loss_cooldown_status(
                    config.POCKET,
                    max_losses=config.LOSS_STREAK_MAX,
                    cooldown_minutes=config.LOSS_STREAK_COOLDOWN_MIN,
                )
                if loss_block:
                    _log_skip(f"loss_cooldown {loss_remain:.0f}s", skip_state)
                    continue

                blocked, _, spread_state, spread_reason = spread_monitor.is_blocked()
                spread_pips = float((spread_state or {}).get("spread_pips") or 0.0)
                if blocked:
                    _log_skip(f"spread_block {spread_reason or ''}".strip(), skip_state)
                    continue
                if config.SPREAD_MAX_PIPS > 0.0 and spread_pips > config.SPREAD_MAX_PIPS:
                    _log_skip(f"spread {spread_pips:.2f}p>limit", skip_state)
                    continue

                factors = all_factors()
                fac_h1 = factors.get("H1")
                fac_h4 = factors.get("H4")
                if not fac_h1:
                    _log_skip("missing_h1_factors", skip_state)
                    continue

                candles = fac_h1.get("candles")
                if not isinstance(candles, list) or len(candles) < config.MIN_CANDLES:
                    _log_skip("insufficient_candles", skip_state)
                    continue

                timestamp_raw = fac_h1.get("timestamp")
                if isinstance(timestamp_raw, str):
                    ts = _parse_iso8601(timestamp_raw)
                    if ts:
                        age = (
                            now_utc.replace(tzinfo=datetime.timezone.utc) - ts
                        ).total_seconds()
                        if config.DATA_STALE_SECONDS > 0 and age > config.DATA_STALE_SECONDS:
                            _log_skip(f"stale_data age={age:.0f}s", skip_state)
                            continue

                fac_signal = dict(fac_h1)
                atr_val = fac_signal.get("atr_pips")
                if atr_val is None:
                    atr_raw = fac_signal.get("atr")
                    if isinstance(atr_raw, (int, float)):
                        atr_val = float(atr_raw) * 100.0
                        fac_signal["atr_pips"] = atr_val
                try:
                    atr_pips = float(atr_val) if atr_val is not None else None
                except (TypeError, ValueError):
                    atr_pips = None
                if atr_pips is None:
                    _log_skip("atr_missing", skip_state)
                    continue
                if atr_pips < config.MIN_ATR_PIPS:
                    _log_skip(f"atr_low {atr_pips:.1f}p", skip_state)
                    continue
                if atr_pips > config.MAX_ATR_PIPS:
                    _log_skip(f"atr_high {atr_pips:.1f}p", skip_state)
                    continue

                close_price = fac_signal.get("close")
                vwap = fac_signal.get("vwap")
                adx_h1 = float(fac_signal.get("adx") or 0.0)
                bbw_h1 = float(fac_signal.get("bbw") or 0.0)
                try:
                    close_f = float(close_price) if close_price is not None else 0.0
                except (TypeError, ValueError):
                    close_f = 0.0
                if bbw_h1 > 0.0 and bbw_h1 < 0.0010 and adx_h1 < 16.0:
                    _log_skip(f"range_guard adx={adx_h1:.1f} bbw={bbw_h1:.4f}", skip_state)
                    continue
                if vwap is not None and close_price is not None:
                    try:
                        vwap_gap = abs(close_f - float(vwap)) / PIP
                        if vwap_gap < 1.0:
                            _log_skip(f"vwap_gap {vwap_gap:.2f}p", skip_state)
                            continue
                    except Exception:
                        pass

                regime = current_regime("H1")
                if config.REQUIRE_REGIME and regime and regime not in config.REQUIRE_REGIME:
                    _log_skip(f"regime_guard {regime}", skip_state)
                    continue
                if config.BLOCK_REGIME and regime and regime in config.BLOCK_REGIME:
                    _log_skip(f"blocked_regime {regime}", skip_state)
                    continue

                decision = MovingAverageCross.check(fac_signal)
                if not decision:
                    _log_skip("no_signal", skip_state)
                    continue

                confidence = int(decision.get("confidence", 0))
                if confidence < config.MIN_CONFIDENCE:
                    _log_skip(f"confidence_low {confidence}", skip_state)
                    continue

                action = decision.get("action")
                if action not in {"OPEN_LONG", "OPEN_SHORT"}:
                    _log_skip(f"unsupported_action {action}", skip_state)
                    continue

                direction = "long" if action == "OPEN_LONG" else "short"
                if (
                    config.ALLOWED_DIRECTIONS
                    and direction.title() not in config.ALLOWED_DIRECTIONS
                ):
                    _log_skip(f"direction_block {direction}", skip_state)
                    continue
                meta = decision.get("_meta") or {}
                fast_gap = float(meta.get("price_to_fast_pips", 0.0) or 0.0)
                if direction == "long" and fast_gap < config.MIN_FAST_GAP_PIPS:
                    _log_skip("fast_gap_insufficient", skip_state)
                    continue
                if direction == "short" and fast_gap > -config.MIN_FAST_GAP_PIPS:
                    _log_skip("fast_gap_insufficient", skip_state)
                    continue
                if not _direction_allowed(fac_signal, fac_h4, direction, atr_pips):
                    _log_skip("direction_mismatch", skip_state)
                    continue

                sig_key = f"{decision.get('tag')}:{direction}"
                prev_ts = recent_signal_gate.get(sig_key, 0.0)
                if config.REPEAT_BLOCK_SEC > 0 and now_mono - prev_ts < config.REPEAT_BLOCK_SEC:
                    _log_skip("repeat_signal_block", skip_state)
                    continue

                open_positions = pos_manager.get_open_positions()
                macro_positions = open_positions.get(config.POCKET, {})
                trades = macro_positions.get("open_trades", [])
                if len(trades) >= config.MAX_ACTIVE_TRADES:
                    _log_skip("max_active_trades", skip_state)
                    continue
                directional_trades = [
                    tr for tr in trades if (tr.get("side") or "").lower() == direction
                ]
                if len(directional_trades) >= config.MAX_DIRECTIONAL_TRADES:
                    _log_skip("directional_trade_cap", skip_state)
                    continue
                directional_units = sum(
                    abs(int(tr.get("units", 0) or 0)) for tr in directional_trades
                )
                if directional_units >= config.MAX_DIRECTIONAL_UNITS:
                    _log_skip("directional_units_cap", skip_state)
                    continue
                stage_idx = len(directional_trades)
                if stage_idx >= len(config.STAGE_RATIOS):
                    _log_skip("stage_limit", skip_state)
                    continue

                last_entry_ts = last_direction_entry.get(direction, 0.0)
                if now_mono - last_entry_ts < config.REENTRY_COOLDOWN_SEC:
                    _log_skip("directional_cooldown", skip_state)
                    continue

                try:
                    snapshot = get_account_snapshot()
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("%s account snapshot failed: %s", config.LOG_PREFIX, exc)
                    _log_skip("account_snapshot_failed", skip_state)
                    continue

                equity = float(getattr(snapshot, "nav", 0.0) or getattr(snapshot, "balance", 0.0))
                if equity <= 0.0:
                    _log_skip("equity_zero", skip_state)
                    continue

                try:
                    sl_pips = float(decision.get("sl_pips") or 0.0)
                    tp_pips = float(decision.get("tp_pips") or 0.0)
                except (TypeError, ValueError):
                    _log_skip("invalid_sl_tp", skip_state)
                    continue
                if sl_pips <= 0.0 or tp_pips <= 0.0:
                    _log_skip("invalid_sl_tp", skip_state)
                    continue

                price_hint = float(fac_signal.get("close") or 0.0)
                entry_price = _latest_mid(price_hint)

                lot = allowed_lot(
                    equity,
                    sl_pips,
                    margin_available=getattr(snapshot, "margin_available", None),
                    price=entry_price,
                    margin_rate=getattr(snapshot, "margin_rate", None),
                    risk_pct_override=config.RISK_PCT,
                    pocket=config.POCKET,
                    strategy_tag=signal.get("tag"),
                    fac_h4=fac_h4,
                )
                if lot <= 0.0:
                    _log_skip("lot_zero", skip_state)
                    continue

                # 市況に応じた lot/TP の動的調整
                if atr_pips < 6.0:
                    lot *= 0.92
                    tp_pips = max(6.0, tp_pips * 0.9)
                elif atr_pips > 22.0:
                    lot *= 1.06
                    tp_pips = min(36.0, tp_pips * 1.05)
                try:
                    vwap_gap = abs(entry_price - float(vwap)) / PIP if vwap is not None else None
                except Exception:
                    vwap_gap = None
                if vwap_gap is not None:
                    if vwap_gap >= 2.0:
                        tp_pips = min(38.0, tp_pips + 1.0)
                    elif vwap_gap <= 1.0:
                        tp_pips = max(5.0, tp_pips * 0.9)

                lot *= _confidence_scale(confidence)
                lot = max(config.MIN_LOT, min(config.MAX_LOT, lot))
                stage_ratio = config.STAGE_RATIOS[stage_idx]
                lot *= max(0.01, stage_ratio)
                units = int(round(lot * 100000))
                if units < config.MIN_UNITS:
                    _log_skip("units_below_min", skip_state)
                    continue
                if direction == "short":
                    units = -units

                sl_price = (
                    entry_price - sl_pips * PIP if direction == "long" else entry_price + sl_pips * PIP
                )
                tp_price = (
                    entry_price + tp_pips * PIP if direction == "long" else entry_price - tp_pips * PIP
                )
                sl_price, tp_price = clamp_sl_tp(entry_price, sl_price, tp_price, direction == "long")

                strategy_tag = decision.get("tag") or "trend_h1"
                client_id = _client_order_id(strategy_tag or "trend")
                thesis = {
                    "strategy_tag": strategy_tag,
                    "confidence": confidence,
                    "atr_pips": atr_pips,
                    "regime": regime,
                    "stage": stage_idx + 1,
                    "tp_pips": tp_pips,
                    "sl_pips": sl_pips,
                    "hard_stop_pips": sl_pips,
                }
                entry_meta = decision.get("_meta") or {}
                entry_meta["stage_index"] = stage_idx

                LOG.info(
                    "%s signal=%s dir=%s conf=%d lot=%.4f units=%d sl=%.2fp tp=%.2fp price=%.3f atr=%.1fp",
                    config.LOG_PREFIX,
                    decision.get("tag"),
                    direction,
                    confidence,
                    lot,
                    units,
                    sl_pips,
                    tp_pips,
                    entry_price,
                    atr_pips,
                )

                ticket_id = await market_order(
                    "USD_JPY",
                    units,
                    sl_price,
                    tp_price,
                    config.POCKET,
                    client_order_id=client_id,
                    entry_thesis=thesis,
                    meta=entry_meta,
                )
                if ticket_id:
                    pos_manager.register_open_trade(ticket_id, config.POCKET, client_id)
                    last_direction_entry[direction] = now_mono
                    recent_signal_gate[sig_key] = now_mono
                    cooldown_until = now_mono + config.ENTRY_COOLDOWN_SEC
                    skip_state["reason"] = ""
                    skip_state["ts"] = 0.0
                    LOG.info(
                        "%s order filled ticket=%s units=%d sl=%.3f tp=%.3f",
                        config.LOG_PREFIX,
                        ticket_id,
                        units,
                        sl_price,
                        tp_price,
                    )
                else:
                    cooldown_until = now_mono + min(config.ENTRY_COOLDOWN_SEC, 60.0)
                    _log_skip("order_failed", skip_state)
            except asyncio.CancelledError:
                LOG.info("%s worker cancelled", config.LOG_PREFIX)
                raise
            except Exception as exc:  # noqa: BLE001
                LOG.exception("%s loop error: %s", config.LOG_PREFIX, exc)
                cooldown_until = time.monotonic() + max(config.LOOP_INTERVAL_SEC, 15.0)
                continue
    except asyncio.CancelledError:
        LOG.info("%s worker cancelled", config.LOG_PREFIX)
        raise
    except Exception as exc:  # noqa: BLE001
        LOG.exception("%s worker crashed: %s", config.LOG_PREFIX, exc)
    finally:
        pos_manager.close()
        LOG.info("%s worker stopped", config.LOG_PREFIX)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(trend_h1_worker())
