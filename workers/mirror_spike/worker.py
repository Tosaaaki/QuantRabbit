"""
Async worker that mirrors the discretionary spike-reversal scalp.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

from analysis.range_model import compute_range_snapshot
from execution.order_manager import market_order
from execution.position_manager import PositionManager
from indicators.factor_cache import get_candles_snapshot
from market_data import spread_monitor, tick_window
from workers.fast_scalp.signal import SignalFeatures, extract_features
from utils.metrics_logger import log_metric
from workers.common import perf_guard

from . import config

_LOGGER = logging.getLogger(__name__)
MR_RANGE_LOOKBACK = 20
MR_RANGE_HI_PCT = 95.0
MR_RANGE_LO_PCT = 5.0


@dataclass
class SpikeSignal:
    side: str  # "short" or "long"
    entry_price: float
    extreme_price: float  # recent peak (short) or trough (long)
    reference_price: float  # prior trough (short) or peak (long)
    spike_height_pips: float
    retrace_pips: float
    extreme_age_sec: float
    features: SignalFeatures
    tick_rate: float


def _build_client_order_id(side: str) -> str:
    ts_ms = int(time.time() * 1000)
    digest = hashlib.sha1(f"{ts_ms}-{side}".encode("utf-8")).hexdigest()[:6]
    return f"qr-mirror-{ts_ms}-{side[0]}{digest}"


def _build_reversion_meta(reference_price: float, atr_pips: Optional[float]) -> dict:
    meta: Dict[str, object] = {
        "env_tf": "M5",
        "struct_tf": "M1",
        "range_method": "percentile",
        "range_lookback": MR_RANGE_LOOKBACK,
        "range_hi_pct": MR_RANGE_HI_PCT,
        "range_lo_pct": MR_RANGE_LO_PCT,
        "tp_mode": "soft_zone",
        "tp_target": "entry_mean",
        "tp_pad_atr": 0.05,
        # mirror_spike はMA構造で十分なので追加の構造否定は無効化
        "structure_break": {"buffer_atr": 0.10, "confirm_closes": 0},
        "reversion_failure": {
            "z_ext": 0.45,
            "contraction_min": 0.45,
            "bars_budget": {"k_per_z": 2.5, "min": 2, "max": 8},
            "trend_takeover": {"require_env_trend_bars": 2},
        },
    }
    if reference_price and reference_price > 0:
        meta["entry_mean"] = float(reference_price)
    if atr_pips and atr_pips > 0:
        meta["atr_entry"] = float(atr_pips)
    candles = get_candles_snapshot("M5", limit=MR_RANGE_LOOKBACK)
    snapshot = compute_range_snapshot(
        candles,
        lookback=MR_RANGE_LOOKBACK,
        method="percentile",
        hi_pct=MR_RANGE_HI_PCT,
        lo_pct=MR_RANGE_LO_PCT,
    )
    if snapshot:
        meta["range_snapshot"] = snapshot.to_dict()
    return meta


def _detect_short_signal(
    ticks: list[dict[str, float]], features: SignalFeatures
) -> Optional[SpikeSignal]:
    """
    Determine whether recent price action forms an exhaustion spike suitable for
    a short entry. We look for:
      * sharp upside spike within LOOKBACK window
      * peak inside PEAK_WINDOW
      * price has retraced a minimum distance
      * RSI remains elevated (overbought)
    """
    if len(ticks) < 2:
        return None

    mids = [float(t["mid"]) for t in ticks]
    epochs = [float(t["epoch"]) for t in ticks]

    latest_epoch = epochs[-1]
    span_seconds = max(float(features.span_seconds or 0.0), 0.1)
    tick_rate = float(features.tick_count) / span_seconds if span_seconds > 0 else 0.0
    atr_val = features.atr_pips
    if config.MIN_ATR_PIPS > 0.0 and atr_val is not None and atr_val < config.MIN_ATR_PIPS:
        return None
    if tick_rate < config.MIN_TICK_RATE:
        return None
    if features.rsi is None or features.rsi < config.RSI_OVERBOUGHT:
        return None
    if features.short_momentum_pips < -config.SHORT_MAX_DOWNSLOPE_PIPS:
        return None
    peak_idx = max(range(len(mids)), key=mids.__getitem__)
    peak_price = mids[peak_idx]
    peak_epoch = epochs[peak_idx]
    peak_age_sec = latest_epoch - peak_epoch
    if peak_age_sec < 0:
        peak_age_sec = 0.0
    if peak_age_sec > config.PEAK_WINDOW_SEC:
        return None

    trough_price = min(mids[: peak_idx + 1]) if peak_idx > 0 else mids[0]
    spike_height_pips = (peak_price - trough_price) / config.PIP_VALUE
    short_spike_floor = max(config.SPIKE_THRESHOLD_PIPS, config.SHORT_MIN_SPIKE_PIPS)
    if spike_height_pips < short_spike_floor:
        return None

    retrace_pips = (peak_price - features.latest_mid) / config.PIP_VALUE
    if retrace_pips < config.RETRACE_TRIGGER_PIPS:
        return None
    if retrace_pips < config.MIN_RETRACE_PIPS:
        return None

    # Price should still be above trough to avoid chasing a complete reversal.
    if features.latest_mid <= trough_price:
        return None

    return SpikeSignal(
        side="short",
        entry_price=features.latest_mid,
        extreme_price=peak_price,
        reference_price=trough_price,
        spike_height_pips=spike_height_pips,
        retrace_pips=retrace_pips,
        extreme_age_sec=peak_age_sec,
        features=features,
        tick_rate=tick_rate,
    )


def _detect_long_signal(
    ticks: list[dict[str, float]], features: SignalFeatures
) -> Optional[SpikeSignal]:
    if len(ticks) < 2:
        return None

    mids = [float(t["mid"]) for t in ticks]
    epochs = [float(t["epoch"]) for t in ticks]

    latest_epoch = epochs[-1]
    span_seconds = max(float(features.span_seconds or 0.0), 0.1)
    tick_rate = float(features.tick_count) / span_seconds if span_seconds > 0 else 0.0
    atr_val = features.atr_pips
    if config.MIN_ATR_PIPS > 0.0 and atr_val is not None and atr_val < config.MIN_ATR_PIPS:
        return None
    if tick_rate < config.MIN_TICK_RATE:
        return None
    if features.rsi is None or features.rsi > config.RSI_OVERSOLD:
        return None
    if features.short_momentum_pips > config.LONG_MAX_UPSLOPE_PIPS:
        return None

    trough_idx = min(range(len(mids)), key=mids.__getitem__)
    trough_price = mids[trough_idx]
    trough_epoch = epochs[trough_idx]
    trough_age_sec = latest_epoch - trough_epoch
    if trough_age_sec < 0:
        trough_age_sec = 0.0
    if trough_age_sec > config.PEAK_WINDOW_SEC:
        return None

    peak_price = max(mids[: trough_idx + 1]) if trough_idx > 0 else mids[0]
    spike_height_pips = (peak_price - trough_price) / config.PIP_VALUE
    long_spike_floor = max(config.SPIKE_THRESHOLD_PIPS, config.LONG_MIN_SPIKE_PIPS)
    if spike_height_pips < long_spike_floor:
        return None

    retrace_pips = (features.latest_mid - trough_price) / config.PIP_VALUE
    if retrace_pips < config.RETRACE_TRIGGER_PIPS:
        return None
    if retrace_pips < config.MIN_RETRACE_PIPS:
        return None

    if features.latest_mid >= peak_price:
        return None

    return SpikeSignal(
        side="long",
        entry_price=features.latest_mid,
        extreme_price=trough_price,
        reference_price=peak_price,
        spike_height_pips=spike_height_pips,
        retrace_pips=retrace_pips,
        extreme_age_sec=trough_age_sec,
        features=features,
        tick_rate=tick_rate,
    )


def _detect_signal(
    ticks: list[dict[str, float]], features: SignalFeatures
) -> Optional[SpikeSignal]:
    signal = _detect_short_signal(ticks, features)
    if signal:
        return signal
    return _detect_long_signal(ticks, features)


async def _log_direction_path(side: str, entry_price: float, entry_epoch: float) -> None:
    """Log short-horizon direction hit/move + MFE/MAE for precision tracking."""
    horizons = (30, 60)
    logger = logging.getLogger(__name__)
    try:
        for horizon in horizons:
            delay = horizon - (time.time() - entry_epoch)
            if delay > 0:
                await asyncio.sleep(delay)
            ticks = tick_window.recent_ticks(seconds=max(horizon + 10, 90), limit=720)
            mids = []
            for t in ticks:
                try:
                    if float(t.get("epoch", 0.0)) >= entry_epoch:
                        mids.append(float(t.get("mid")))
                except Exception:
                    continue
            if not mids:
                continue
            last_mid = mids[-1]
            move = (last_mid - entry_price) / config.PIP_VALUE
            mfe = (max(mids) - entry_price) / config.PIP_VALUE
            mae = (entry_price - min(mids)) / config.PIP_VALUE
            if side == "short":
                move = -move
                mfe = (entry_price - min(mids)) / config.PIP_VALUE
                mae = (max(mids) - entry_price) / config.PIP_VALUE
            tags = {"side": side, "horizon": f"{horizon}s"}
            log_metric("mirror_spike_dir_hit", 1.0 if move > 0 else 0.0, tags=tags)
            log_metric("mirror_spike_move_pips", move, tags=tags)
            log_metric("mirror_spike_mfe_pips", mfe, tags=tags)
            log_metric("mirror_spike_mae_pips", mae, tags=tags)
    except Exception:
        logger.debug("%s metric logging failed", config.LOG_PREFIX, exc_info=True)


async def mirror_spike_worker() -> None:
    """
    Separate loop from FastScalp that imitates the discretionary spike fade.
    """
    logger = logging.getLogger(__name__)
    logger.info("%s worker starting.", config.LOG_PREFIX)
    pos_manager = PositionManager()
    cooldown_until = 0.0
    post_exit_cooldown_until = 0.0
    last_spread_block_log = 0.0
    last_hour_log = 0.0
    last_perf_log = 0.0
    try:
        while True:
            await asyncio.sleep(0.6)
            now_monotonic = time.monotonic()

            if now_monotonic < post_exit_cooldown_until:
                continue
            if now_monotonic < cooldown_until:
                continue
            if config.ACTIVE_HOURS_UTC:
                current_hour = time.gmtime().tm_hour
                if current_hour not in config.ACTIVE_HOURS_UTC:
                    if now_monotonic - last_hour_log > 300.0:
                        logger.info(
                            "%s outside active hours hour=%02d",
                            config.LOG_PREFIX,
                            current_hour,
                        )
                        last_hour_log = now_monotonic
                    continue
                last_hour_log = now_monotonic

            # Skip if mirror spike trades are already stacked in the scalp pocket.
            pockets = pos_manager.get_open_positions()
            scalp_pos = pockets.get("scalp")
            open_trades = (scalp_pos or {}).get("open_trades") or []
            tracked_trades = [
                tr
                for tr in open_trades
                if (tr.get("entry_thesis") or {}).get("strategy_tag") == "mirror_spike"
            ]
            existing_side: Optional[str] = None
            latest_trade: Optional[dict] = None
            if tracked_trades:
                tracked_trades.sort(key=lambda tr: tr.get("open_time") or "", reverse=True)
                latest_trade = tracked_trades[0]
                candidate_side = latest_trade.get("side")
                if not candidate_side:
                    units_val = latest_trade.get("units", 0)
                    candidate_side = "long" if units_val > 0 else "short"
                existing_side = candidate_side
                if len(tracked_trades) >= config.MAX_ACTIVE_TRADES:
                    continue

            blocked, _, spread_state, spread_reason = spread_monitor.is_blocked()
            spread_pips = 0.0
            if spread_state:
                try:
                    spread_pips = float(spread_state.get("spread_pips", 0.0) or 0.0)
                except (TypeError, ValueError):
                    spread_pips = 0.0
            if blocked or spread_pips > config.SPREAD_MAX_PIPS:
                if now_monotonic - last_spread_block_log > 30.0:
                    logger.info(
                        "%s spread gating active spread=%.2fp reason=%s",
                        config.LOG_PREFIX,
                        spread_pips,
                        spread_reason or "guard_active",
                    )
                    last_spread_block_log = now_monotonic
                continue

            ticks = tick_window.recent_ticks(seconds=config.LOOKBACK_SEC, limit=360)
            if len(ticks) < config.MIN_TICK_COUNT:
                continue

            features = extract_features(spread_pips, ticks=ticks)
            if not features:
                continue

            signal = _detect_signal(ticks, features)
            if not signal:
                continue
            perf_decision = perf_guard.is_allowed("mirror_spike", "scalp")
            if not perf_decision.allowed:
                if now_monotonic - last_perf_log > 120.0:
                    logger.info(
                        "%s perf_block tag=%s reason=%s",
                        config.LOG_PREFIX,
                        "mirror_spike",
                        perf_decision.reason,
                    )
                    last_perf_log = now_monotonic
                continue
            if existing_side and signal.side != existing_side:
                continue

            latest_tick = ticks[-1]
            try:
                last_bid = float(latest_tick.get("bid") or signal.entry_price)
                last_ask = float(latest_tick.get("ask") or signal.entry_price)
            except (TypeError, ValueError):
                last_bid = signal.entry_price
                last_ask = signal.entry_price

            tp_pips = config.SHORT_TP_PIPS if signal.side == "short" else config.TP_PIPS
            sl_pips = config.SHORT_SL_PIPS if signal.side == "short" else config.SL_PIPS
            if signal.side == "short":
                entry_price = last_bid
                tp_price = round(entry_price - tp_pips * config.PIP_VALUE, 3)
            else:
                entry_price = last_ask
                tp_price = round(entry_price + tp_pips * config.PIP_VALUE, 3)
            if tp_price is not None and tp_price <= 0:
                tp_price = None
            if signal.side == "short":
                sl_price = round(entry_price + sl_pips * config.PIP_VALUE, 3)
            else:
                sl_price = round(entry_price - sl_pips * config.PIP_VALUE, 3)
            if sl_price is not None and sl_price <= 0:
                sl_price = None

            if latest_trade is not None:
                prev_price = float(latest_trade.get("price") or entry_price)
                delta_pips = abs(prev_price - entry_price) / config.PIP_VALUE
                if delta_pips < config.STAGE_MIN_DELTA_PIPS:
                    continue

            units = -config.ENTRY_UNITS if signal.side == "short" else config.ENTRY_UNITS
            client_id = _build_client_order_id(signal.side)
            entry_thesis = {
                "strategy_tag": "mirror_spike",
                "pattern": f"spike_reversal_{signal.side}",
                "spike_height_pips": round(signal.spike_height_pips, 3),
                "retrace_pips": round(signal.retrace_pips, 3),
                "extreme_age_sec": round(signal.extreme_age_sec, 2),
                "extreme_price": round(signal.extreme_price, 5),
                "reference_price": round(signal.reference_price, 5),
                "tp_pips": round(tp_pips, 2),
                "sl_pips": round(sl_pips, 2),
                "entry_price_snapshot": round(entry_price, 5),
                "latest_mid": round(signal.entry_price, 5),
                "spread_pips": round(spread_pips, 3),
                "tick_count": len(ticks),
                "tick_rate": round(signal.tick_rate, 2),
                "rsi": None if signal.features.rsi is None else round(signal.features.rsi, 2),
                "atr_pips": None
                if signal.features.atr_pips is None
                else round(signal.features.atr_pips, 3),
                "momentum_pips": round(signal.features.momentum_pips, 3),
                "short_momentum_pips": round(signal.features.short_momentum_pips, 3),
            }
            entry_thesis.update(
                _build_reversion_meta(
                    signal.reference_price,
                    signal.features.atr_pips,
                )
            )
            try:
                spike_height = float(signal.spike_height_pips)
            except Exception:
                spike_height = None
            rf = entry_thesis.get("reversion_failure")
            if isinstance(rf, dict) and spike_height is not None:
                bars_budget = rf.get("bars_budget")
                if not isinstance(bars_budget, dict):
                    bars_budget = {}
                    rf["bars_budget"] = bars_budget
                if spike_height >= 7.0:
                    bars_budget["k_per_z"] = 3.0
                    bars_budget["max"] = 10
                elif spike_height <= 4.0:
                    bars_budget["k_per_z"] = 2.0
                    bars_budget["max"] = 6

            try:
                trade_id = await market_order(
                    "USD_JPY",
                    units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    pocket="scalp",
                    client_order_id=client_id,
                    entry_thesis=entry_thesis,
                )
            except Exception as exc:  # pragma: no cover - network path
                logger.error(
                    "%s order error side=%s units=%s exc=%s",
                    config.LOG_PREFIX,
                    signal.side,
                    units,
                    exc,
                )
                cooldown_until = now_monotonic + config.COOLDOWN_SEC
                continue

            if trade_id:
                fill_price = entry_price
                logger.info(
                    "%s entry trade_id=%s side=%s units=%s exec=%.3f tp=%s spike=%.2fp retrace=%.2fp",
                    config.LOG_PREFIX,
                    trade_id,
                    signal.side,
                    units,
                    fill_price,
                    f"{tp_price:.3f}" if tp_price is not None else "n/a",
                    signal.spike_height_pips,
                    signal.retrace_pips,
                )
                asyncio.create_task(_log_direction_path(signal.side, entry_price, time.time()))
                cooldown_until = now_monotonic + config.COOLDOWN_SEC
                post_exit_cooldown_until = now_monotonic + config.POST_EXIT_COOLDOWN_SEC
            else:
                cooldown_until = now_monotonic + 10.0
    except asyncio.CancelledError:
        logger.info("%s worker cancelled.", config.LOG_PREFIX)
        raise
    except Exception:
        logger.exception("%s worker crashed.", config.LOG_PREFIX)
        raise
    finally:
        try:
            pos_manager.close()
        except Exception:
            logger.exception("%s failed to close PositionManager", config.LOG_PREFIX)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    logger = logging.getLogger(__name__)
    loop_sec = getattr(config, "LOOP_INTERVAL_SEC", 0.5)
    logger.info("%s worker boot (loop %.2fs)", config.LOG_PREFIX, loop_sec)
    asyncio.run(mirror_spike_worker())
