"""Rule-based scalping executor."""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from indicators.factor_cache import all_factors, on_candle
from utils.market_hours import is_market_open, is_scalp_session_open
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp
from execution.order_manager import market_order
from strategies.scalp.basic import BasicScalpStrategy, ScalpSignal
from analysis.scalp_learning import load_adjustments
from analysis.scalp_config import ensure_overrides, load_overrides
from market_data.candle_fetcher import fetch_historical_candles


@dataclass
class ScalpConfig:
    def __init__(self, overrides: Optional[Dict[str, Any]] = None) -> None:
        overrides = overrides or {}
        self.enabled = os.getenv("SCALP_ENABLED", "true").lower() == "true"
        self.loop_interval_sec = _float_safe(os.getenv("SCALP_LOOP_INTERVAL_SEC"), 3.0)
        self.cooldown_sec = _float_safe(os.getenv("SCALP_COOLDOWN_SEC"), 6.0)
        base_lot_env = _float_safe(os.getenv("SCALP_BASE_LOT"), _INITIAL_BASE_LOT)
        self.base_lot = _float_safe(overrides.get("base_lot"), base_lot_env)
        self.reference_equity = _float_safe(
            os.getenv("SCALP_REFERENCE_EQUITY", os.getenv("REFERENCE_EQUITY")),
            10000.0,
        )
        self.default_spread_pips = _float_safe(os.getenv("SCALP_DEFAULT_SPREAD_PIPS"), 0.6)
        self.max_spread_pips = _float_safe(os.getenv("SCALP_MAX_SPREAD_PIPS"), 1.3)
        self.min_atr_pips = _float_safe(
            overrides.get("min_atr_pips"),
            _float_safe(os.getenv("SCALP_MIN_ATR_PIPS"), 1.8),
        )
        self.deviation_pips = _float_safe(
            overrides.get("deviation_pips"),
            _float_safe(os.getenv("SCALP_DEVIATION_PIPS"), 1.4),
        )
        self.atr_threshold_mult = _float_safe(os.getenv("SCALP_ATR_THRESHOLD_MULT"), 0.15)
        self.min_sl_pips = _float_safe(os.getenv("SCALP_MIN_SL_PIPS"), 6.0)
        self.tp_multiplier = _float_safe(os.getenv("SCALP_TP_MULTIPLIER"), 1.2)
        self.stale_after_sec = int(os.getenv("SCALP_STALE_AFTER_SEC", os.getenv("MARKET_STALE_AFTER_SEC", "600")))
        self.pip_size = _float_safe(os.getenv("SCALP_PIP_SIZE"), 0.01)
        self.max_tick_velocity = _float_safe(os.getenv("SCALP_MAX_TICK_VELOCITY_30S"), 8.0)
        self.max_tick_range = _float_safe(os.getenv("SCALP_MAX_TICK_RANGE_30S"), 12.0)
        self.momentum_flip_pips = _float_safe(os.getenv("SCALP_MOMENTUM_FLIP_PIPS"), 0.7)
        self.momentum_confirm_pips = _float_safe(os.getenv("SCALP_MOMENTUM_CONFIRM_PIPS"), 0.5)
        self.momentum_velocity_cap = _float_safe(os.getenv("SCALP_MOMENTUM_VELOCITY_CAP"), 6.0)
        self.momentum_range_min = _float_safe(os.getenv("SCALP_MOMENTUM_RANGE_MIN"), 5.0)
        self.enable_trend_follow = os.getenv("SCALP_ENABLE_TREND_FOLLOW", "true").lower() == "true"
        self.trend_velocity_min = _float_safe(os.getenv("SCALP_TREND_VELOCITY_MIN"), 6.5)
        self.trend_range_min = _float_safe(os.getenv("SCALP_TREND_RANGE_MIN"), 8.0)
        self.trend_momentum_min = _float_safe(os.getenv("SCALP_TREND_MOMENTUM_MIN"), 0.8)
        self.trend_rsi_buy = _float_safe(os.getenv("SCALP_TREND_RSI_BUY"), 60.0)
        self.trend_rsi_sell = _float_safe(os.getenv("SCALP_TREND_RSI_SELL"), 40.0)
        self.trend_sl_pips = _float_safe(os.getenv("SCALP_TREND_SL_PIPS"), 7.0)
        self.trend_tp_multiplier = _float_safe(os.getenv("SCALP_TREND_TP_MULTIPLIER"), 1.8)
        window_override = overrides.get("loss_cluster_window_sec") if overrides else None
        max_override = overrides.get("loss_cluster_max") if overrides else None
        cooldown_override = overrides.get("loss_cluster_cooldown_sec") if overrides else None
        self.loss_cluster_window_sec = int(
            float(
                window_override
                or os.getenv("SCALP_LOSS_CLUSTER_WINDOW_SEC")
                or 900
            )
        )
        self.loss_cluster_max = int(
            float(max_override or os.getenv("SCALP_LOSS_CLUSTER_MAX") or 3)
        )
        self.loss_cluster_cooldown_sec = int(
            float(
                cooldown_override
                or os.getenv("SCALP_LOSS_CLUSTER_COOLDOWN_SEC")
                or 900
            )
        )
        self.max_open_positions = int(os.getenv("SCALP_MAX_OPEN_POSITIONS", "2"))
        self.loss_streak_window_sec = int(os.getenv("SCALP_LOSS_STREAK_WINDOW_SEC", "1800"))
        self.loss_streak_deviation_boost = _float_safe(os.getenv("SCALP_LOSS_STREAK_DEV_BOOST"), 0.25)
        self.pre_range_minutes = int(os.getenv("SCALP_PRE_RANGE_MINUTES", "15"))
        self.pre_range_threshold_pips = _float_safe(
            os.getenv("SCALP_PRE_RANGE_THRESHOLD_PIPS"), 10.0
        )
        self.recent_range_minutes = int(os.getenv("SCALP_RECENT_RANGE_MINUTES", "5"))
        self.recent_range_threshold_pips = _float_safe(
            os.getenv("SCALP_RECENT_RANGE_THRESHOLD_PIPS"), 4.0
        )
        self.pre_range_relax_count = int(
            _float_safe(
                overrides.get("pre_range_relax_count"),
                _float_safe(os.getenv("SCALP_PRE_RANGE_RELAX_COUNT"), 4),
            )
        )
        self.pre_range_relax_add = _float_safe(
            overrides.get("pre_range_relax_add"),
            _float_safe(os.getenv("SCALP_PRE_RANGE_RELAX_ADD"), 1.5),
        )
        self.recent_range_relax_count = int(
            _float_safe(
                overrides.get("recent_range_relax_count"),
                _float_safe(os.getenv("SCALP_RECENT_RANGE_RELAX_COUNT"), 3),
            )
        )
        self.recent_range_relax_add = _float_safe(
            overrides.get("recent_range_relax_add"),
            _float_safe(os.getenv("SCALP_RECENT_RANGE_RELAX_ADD"), 0.8),
        )
        self.trend_override_ma_pips = _float_safe(
            overrides.get("trend_override_ma_pips"),
            _float_safe(os.getenv("SCALP_TREND_OVERRIDE_MA_PIPS"), 6.0),
        )
        self.trend_override_rsi_buy = _float_safe(
            overrides.get("trend_override_rsi_buy"),
            _float_safe(os.getenv("SCALP_TREND_OVERRIDE_RSI_BUY"), 62.0),
        )
        self.trend_override_rsi_sell = _float_safe(
            overrides.get("trend_override_rsi_sell"),
            _float_safe(os.getenv("SCALP_TREND_OVERRIDE_RSI_SELL"), 38.0),
        )
        self.trend_override_velocity = _float_safe(
            overrides.get("trend_override_velocity"),
            _float_safe(os.getenv("SCALP_TREND_OVERRIDE_VELOCITY"), 5.5),
        )
        self.trend_override_duration_sec = int(
            _float_safe(
                overrides.get("trend_override_duration_sec"),
                _float_safe(os.getenv("SCALP_TREND_OVERRIDE_DURATION_SEC"), 120),
            )
        )
        self.per_minute_entry_cap = int(
            _float_safe(
                overrides.get("per_minute_entry_cap"),
                _float_safe(os.getenv("SCALP_PER_MINUTE_ENTRY_CAP"), 3),
            )
        )
        self.revert_range_block_pips = _float_safe(
            overrides.get("revert_range_block_pips"),
            _float_safe(os.getenv("SCALP_REVERT_RANGE_BLOCK_PIPS"), 3.0),
        )
        self.revert_range_widen_pips = _float_safe(
            overrides.get("revert_range_widen_pips"),
            _float_safe(os.getenv("SCALP_REVERT_RANGE_WIDEN_PIPS"), 2.5),
        )
        self.revert_range_sl_boost = _float_safe(
            overrides.get("revert_range_sl_boost"),
            _float_safe(os.getenv("SCALP_REVERT_RANGE_SL_BOOST"), 1.35),
        )
        self.revert_gap_pips = _float_safe(
            overrides.get("revert_gap_pips"),
            _float_safe(os.getenv("SCALP_REVERT_GAP_PIPS"), 6.5),
        )
        self.revert_rsi_long_max = _float_safe(
            overrides.get("revert_rsi_long_max"),
            _float_safe(os.getenv("SCALP_REVERT_RSI_LONG_MAX"), 34.0),
        )
        self.revert_rsi_short_min = _float_safe(
            overrides.get("revert_rsi_short_min"),
            _float_safe(os.getenv("SCALP_REVERT_RSI_SHORT_MIN"), 66.0),
        )
        disable_flag = overrides.get("mean_revert_disabled") if overrides else None
        if disable_flag is None:
            disable_flag = os.getenv("SCALP_MEAN_REVERT_DISABLED")
        if isinstance(disable_flag, str):
            disable_flag = disable_flag.lower() in {"1", "true", "yes", "on"}
        self.mean_revert_disabled = bool(disable_flag)
        disable_until_raw = None
        if overrides and overrides.get("mean_revert_disabled_until"):
            disable_until_raw = overrides.get("mean_revert_disabled_until")
        elif os.getenv("SCALP_MEAN_REVERT_DISABLED_UNTIL"):
            disable_until_raw = os.getenv("SCALP_MEAN_REVERT_DISABLED_UNTIL")
        self.mean_revert_disabled_until: Optional[datetime] = None
        if disable_until_raw:
            try:
                value = disable_until_raw
                if isinstance(value, (int, float)):
                    dt_val = datetime.fromtimestamp(float(value), tz=timezone.utc)
                else:
                    if isinstance(value, str) and value.endswith("Z"):
                        value = value[:-1] + "+00:00"
                    dt_val = datetime.fromisoformat(value)
                    if dt_val.tzinfo is None:
                        dt_val = dt_val.replace(tzinfo=timezone.utc)
                self.mean_revert_disabled_until = dt_val
            except Exception:
                self.mean_revert_disabled_until = None
        self.learning_meta = (
            overrides.get("learning") if isinstance(overrides.get("learning"), dict) else {}
        )
        if (
            self.mean_revert_disabled
            and self.mean_revert_disabled_until
            and self.mean_revert_disabled_until <= datetime.now(timezone.utc)
        ):
            self.mean_revert_disabled = False
            self.mean_revert_disabled_until = None


@dataclass
class ScalpRuntimeState:
    last_attempt: Optional[datetime] = None
    last_trade: Optional[datetime] = None
    last_reason: str = "init"
    last_signal: Optional[str] = None
    last_order_id: Optional[str] = None
    last_error: Optional[str] = None
    cooldown_remaining: float = 0.0
    last_result: Dict[str, Any] = field(default_factory=dict)
    adjustments: Dict[str, float] = field(default_factory=dict)
    cluster_block_until: Optional[datetime] = None
    pre_range_blocks: int = 0
    recent_range_blocks: int = 0
    trend_bias_until: Optional[datetime] = None
    trend_bias_direction: Optional[str] = None
    per_minute_counts: Dict[str, int] = field(default_factory=dict)


_OVERRIDES = ensure_overrides()


def _float_safe(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


_ENV_BASE_LOT = _float_safe(os.getenv("SCALP_BASE_LOT"), 0.008)
_INITIAL_BASE_LOT = _float_safe(_OVERRIDES.get("base_lot"), _ENV_BASE_LOT)


_CONFIG = ScalpConfig(overrides=_OVERRIDES)
_CONFIG_REFRESH_SEC = int(os.getenv("SCALP_CONFIG_REFRESH_SEC", "180"))
_CONFIG_SIGNATURE = tuple(sorted(_OVERRIDES.items())) if _OVERRIDES else None
_CONFIG_LAST_REFRESH = 0.0
_STATE = ScalpRuntimeState()

_SCALP_DB_PATH = pathlib.Path("logs/scalp_trades.db")
try:
    _SCALP_DB_PATH.parent.mkdir(exist_ok=True)
    _SCALP_DB = sqlite3.connect(_SCALP_DB_PATH, check_same_thread=False)
    _SCALP_DB.execute(
        """
        CREATE TABLE IF NOT EXISTS scalp_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            action TEXT,
            price REAL,
            sl_pips REAL,
            tp_pips REAL,
            spread_pips REAL,
            atr_pips REAL,
            deviation_pips REAL,
            lot REAL,
            units INTEGER,
            trade_id TEXT,
            reason TEXT
        )
        """
    )
    _SCALP_DB.commit()
except Exception:
    _SCALP_DB = None

_TRADES_DB_PATH = pathlib.Path("logs/trades.db")
try:
    _TRADES_DB_PATH.parent.mkdir(exist_ok=True)
    _TRADES_CONN = sqlite3.connect(_TRADES_DB_PATH, check_same_thread=False)
except Exception:
    _TRADES_CONN = None

_ADJUST_REFRESH_SEC = int(os.getenv("SCALP_LEARNING_REFRESH_SEC", "600"))
_ADJUST_CACHE: Dict[str, Any] = {"ts": None, "data": {"lot_multiplier": 1.0, "deviation_offset": 0.0}}

_REST_REFRESH_COOLDOWN = float(os.getenv("SCALP_REST_REFRESH_SEC", "90"))
_REST_REFRESH_COUNT = int(os.getenv("SCALP_REST_FETCH_COUNT", "30"))
_REST_REFRESH_MAX = max(5, min(_REST_REFRESH_COUNT, 120))
_last_rest_refresh = 0.0


def _signature_from_overrides(data: Optional[Dict[str, Any]]) -> Optional[tuple]:
    if not data:
        return None
    try:
        return tuple(sorted((k, float(v)) for k, v in data.items()))
    except Exception:
        return tuple(sorted(data.items()))


def _maybe_refresh_config(now: datetime) -> None:
    global _CONFIG, _CONFIG_SIGNATURE, _CONFIG_LAST_REFRESH
    ts = now.timestamp()
    if ts - _CONFIG_LAST_REFRESH < _CONFIG_REFRESH_SEC:
        return
    _CONFIG_LAST_REFRESH = ts
    try:
        overrides = load_overrides()
    except Exception as exc:  # noqa: BLE001
        logging.warning("[SCALP] failed to load overrides: %s", exc)
        overrides = None
    signature = _signature_from_overrides(overrides)
    if signature == _CONFIG_SIGNATURE:
        return
    try:
        _CONFIG = ScalpConfig(overrides=overrides)
        _CONFIG_SIGNATURE = signature
        logging.info("[SCALP] config refreshed: %s", overrides)
    except Exception as exc:  # noqa: BLE001
        logging.warning("[SCALP] config refresh error: %s", exc)


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_utc(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    value = ts
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _range_from_candles(candles: List[Dict[str, Any]], minutes: int, pip: float) -> Optional[float]:
    if not candles or minutes <= 0 or pip <= 0:
        return None
    count = min(len(candles), max(1, int(minutes)))
    window = candles[-count:]
    try:
        high = max(float(c.get("high")) for c in window)
        low = min(float(c.get("low")) for c in window)
    except (TypeError, ValueError):
        return None
    return (high - low) / pip if high >= low else None


def _recent_loss_cluster_active(config: ScalpConfig, now: datetime) -> tuple[bool, int]:
    if (
        _TRADES_CONN is None
        or config.loss_cluster_max <= 0
        or config.loss_cluster_window_sec <= 0
    ):
        return False, 0
    cutoff = _iso_utc(now - timedelta(seconds=config.loss_cluster_window_sec))
    cutoff_dt = now - timedelta(seconds=config.loss_cluster_window_sec)
    limit = max(config.loss_cluster_max * 6, 40)
    try:
        rows = _TRADES_CONN.execute(
            """
            SELECT close_time, close_reason
            FROM trades
            WHERE pocket='scalp' AND close_time IS NOT NULL AND close_time != '' AND close_time >= ?
            ORDER BY close_time DESC
            LIMIT ?
            """,
            (cutoff, limit),
        ).fetchall()
    except Exception:
        return False, 0
    stop_losses = 0
    for close_time, reason in rows:
        if not close_time:
            continue
        dt = _parse_utc(close_time)
        if dt is None or dt < cutoff_dt:
            continue
        if (reason or "").upper() != "STOP_LOSS_ORDER":
            continue
        stop_losses += 1
    return (stop_losses >= config.loss_cluster_max, stop_losses)


def _open_scalp_positions() -> int:
    if _TRADES_CONN is None:
        return 0
    try:
        row = _TRADES_CONN.execute(
            "SELECT COUNT(*) FROM trades WHERE pocket='scalp' AND state='OPEN'"
        ).fetchone()
    except Exception:
        return 0
    if not row:
        return 0
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return 0


def _recent_loss_streak(now: datetime, window_sec: int) -> int:
    if _TRADES_CONN is None or window_sec <= 0:
        return 0
    cutoff = _iso_utc(now - timedelta(seconds=window_sec))
    try:
        rows = _TRADES_CONN.execute(
            """
            SELECT realized_pl
            FROM trades
            WHERE pocket='scalp' AND close_time IS NOT NULL AND close_time != '' AND close_time >= ?
            ORDER BY close_time DESC
            LIMIT 12
            """,
            (cutoff,),
        ).fetchall()
    except Exception:
        return 0
    streak = 0
    for (pl,) in rows:
        try:
            val = float(pl)
        except (TypeError, ValueError):
            continue
        if val < 0.0:
            streak += 1
        else:
            break
    return streak


def _detect_trend_override(
    fac_m1: Dict[str, Any],
    config: ScalpConfig,
    *,
    direction_hint: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    try:
        price = float(fac_m1.get("close"))
        ma10 = float(fac_m1.get("ma10"))
        rsi = float(fac_m1.get("rsi"))
        velocity = float(fac_m1.get("tick_velocity_30s") or 0.0)
    except (TypeError, ValueError):
        return False, None

    pip = config.pip_size or 0.01
    if pip <= 0:
        pip = 0.01
    ma_gap_pips = (price - ma10) / pip

    long_bias = (
        ma_gap_pips >= config.trend_override_ma_pips
        and rsi >= config.trend_override_rsi_buy
        and velocity >= config.trend_override_velocity
    )
    short_bias = (
        ma_gap_pips <= -config.trend_override_ma_pips
        and rsi <= config.trend_override_rsi_sell
        and velocity <= -config.trend_override_velocity
    )

    if direction_hint == "long" and not long_bias:
        long_bias = ma_gap_pips >= config.trend_override_ma_pips * 0.8 and velocity >= config.trend_override_velocity * 0.6
    elif direction_hint == "short" and not short_bias:
        short_bias = ma_gap_pips <= -config.trend_override_ma_pips * 0.8 and velocity <= -config.trend_override_velocity * 0.6

    if long_bias and not short_bias:
        return True, "long"
    if short_bias and not long_bias:
        return True, "short"
    return False, None


def _active_trend_bias(now: datetime) -> tuple[bool, Optional[str]]:
    if _STATE.trend_bias_until and now < _STATE.trend_bias_until:
        return True, _STATE.trend_bias_direction
    return False, None


def _set_trend_bias(direction: str, now: datetime, duration_sec: int) -> None:
    if duration_sec <= 0:
        _STATE.trend_bias_until = None
        _STATE.trend_bias_direction = None
        return
    _STATE.trend_bias_direction = direction
    _STATE.trend_bias_until = now + timedelta(seconds=duration_sec)


def _minute_key(now: datetime, action: str) -> str:
    return f"{now.strftime('%Y-%m-%dT%H:%M')}:{action.lower()}"


def _prune_minute_counts(now: datetime, retention_min: int = 5) -> None:
    if not _STATE.per_minute_counts:
        return
    cutoff = now - timedelta(minutes=retention_min)
    keys_to_delete = []
    for key in _STATE.per_minute_counts.keys():
        minute_part, _, _ = key.partition(':')
        try:
            minute_dt = datetime.fromisoformat(minute_part + ':00+00:00')
        except ValueError:
            keys_to_delete.append(key)
            continue
        if minute_dt < cutoff:
            keys_to_delete.append(key)
    for key in keys_to_delete:
        _STATE.per_minute_counts.pop(key, None)


def get_scalp_state() -> Dict[str, Any]:
    """Expose runtime state for monitoring."""

    out = asdict(_STATE)
    for key in ("last_attempt", "last_trade"):
        value = out.get(key)
        if isinstance(value, datetime):
            out[key] = value.isoformat(timespec="seconds")
    return out


async def _execute_order(config: ScalpConfig, signal: ScalpSignal, price: float, lot_multiplier: float) -> Dict[str, Any]:
    lot_cap = max(config.base_lot * max(lot_multiplier, 0.0), 0.0)
    lot_allowed = allowed_lot(config.reference_equity, signal.sl_pips)
    lot = min(lot_cap, lot_allowed)
    if lot <= 0:
        return {"success": False, "error": "lot_zero", "lot": lot}

    units = int(round(lot * 100000))
    if units == 0:
        return {"success": False, "error": "lot_zero", "lot": lot, "units": units}
    if signal.action == "sell":
        units *= -1

    pip = config.pip_size
    if signal.action == "buy":
        sl_price = price - signal.sl_pips * pip
        tp_price = price + signal.tp_pips * pip
    else:
        sl_price = price + signal.sl_pips * pip
        tp_price = price - signal.tp_pips * pip
    sl_price, tp_price = clamp_sl_tp(price, sl_price, tp_price, signal.action == "buy")

    result = await market_order(
        "USD_JPY",
        units,
        sl_price,
        tp_price,
        "scalp",
        strategy=BasicScalpStrategy.name,
    )
    result["lot"] = lot
    result["units"] = units
    return result


def _remaining_cooldown(config: ScalpConfig, now: datetime) -> float:
    if _STATE.last_trade is None:
        return 0.0
    elapsed = (now - _STATE.last_trade).total_seconds()
    remaining = max(0.0, config.cooldown_sec - elapsed)
    _STATE.cooldown_remaining = remaining
    return remaining


def _evaluate_signal(
    config: ScalpConfig,
    fac_m1: Dict[str, Any],
    spread_pips: float,
    deviation_pips: float,
    *,
    momentum_flip_pips: float,
    momentum_confirm_pips: float,
    momentum_velocity_cap: float,
    momentum_range_min: float,
    enable_trend_follow: bool,
    trend_velocity_min: float,
    trend_range_min: float,
    trend_momentum_min: float,
    trend_rsi_buy: float,
    trend_rsi_sell: float,
    trend_sl_pips: float,
    trend_tp_multiplier: float,
    candle_range_pips: float,
    revert_range_block_pips: float,
    revert_range_widen_pips: float,
    revert_range_sl_boost: float,
    force_trend: bool,
    forced_direction: Optional[str],
    allow_mean_revert: bool,
) -> tuple[Optional[ScalpSignal], str]:
    return BasicScalpStrategy.evaluate(
        fac_m1,
        spread_pips=spread_pips,
        max_spread_pips=config.max_spread_pips,
        min_atr_pips=config.min_atr_pips,
        deviation_pips=deviation_pips,
        atr_threshold_mult=config.atr_threshold_mult,
        min_sl_pips=config.min_sl_pips,
        tp_multiplier=config.tp_multiplier,
        momentum_flip_pips=momentum_flip_pips,
        momentum_confirm_pips=momentum_confirm_pips,
        momentum_velocity_cap=momentum_velocity_cap,
        momentum_range_min=momentum_range_min,
        enable_trend_follow=enable_trend_follow,
        trend_velocity_min=trend_velocity_min,
        trend_range_min=trend_range_min,
        trend_momentum_min=trend_momentum_min,
        trend_rsi_buy=trend_rsi_buy,
        trend_rsi_sell=trend_rsi_sell,
        trend_sl_pips=trend_sl_pips,
        trend_tp_multiplier=trend_tp_multiplier,
        candle_range_pips=candle_range_pips,
        revert_range_block_pips=revert_range_block_pips,
        revert_range_widen_pips=revert_range_widen_pips,
        revert_range_sl_boost=revert_range_sl_boost,
        allow_mean_revert=allow_mean_revert,
        force_trend=force_trend,
        forced_direction=forced_direction,
    )


def _record_trade(
    *,
    now: datetime,
    signal: ScalpSignal,
    price: float,
    spread_pips: float,
    atr_pips: float,
    deviation_pips: float,
    order: Dict[str, Any],
) -> None:
    if _SCALP_DB is None:
        return
    try:
        _SCALP_DB.execute(
            """
            INSERT INTO scalp_trades(
                ts, action, price, sl_pips, tp_pips, spread_pips,
                atr_pips, deviation_pips, lot, units, trade_id, reason
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                now.isoformat(timespec="seconds"),
                signal.action,
                price,
                signal.sl_pips,
                signal.tp_pips,
                spread_pips,
                atr_pips,
                deviation_pips,
                order.get("lot"),
                order.get("units"),
                order.get("trade_id"),
                _STATE.last_reason,
            ),
        )
        _SCALP_DB.commit()
    except Exception:
        pass


def _get_adjustments(now: datetime) -> Dict[str, float]:
    cached_ts = _ADJUST_CACHE.get("ts")
    if cached_ts is None or (now - cached_ts).total_seconds() >= _ADJUST_REFRESH_SEC:
        try:
            data = load_adjustments()
        except Exception as exc:  # noqa: BLE001
            logging.warning("[SCALP] adjustment load failed: %s", exc)
            data = _ADJUST_CACHE.get("data", {"lot_multiplier": 1.0, "deviation_offset": 0.0})
        else:
            _ADJUST_CACHE["data"] = data
            _ADJUST_CACHE["ts"] = now
    return dict(_ADJUST_CACHE.get("data", {"lot_multiplier": 1.0, "deviation_offset": 0.0}))


async def _attempt_trade(
    now: datetime,
    *,
    fac_m1: Optional[Dict[str, Any]] = None,
    spread_override: Optional[float] = None,
) -> Dict[str, Any]:
    _STATE.last_attempt = now
    _maybe_refresh_config(now)
    config = _CONFIG

    if not config.enabled:
        _STATE.last_reason = "disabled"
        _STATE.last_result = {"executed": False, "reason": "disabled"}
        return _STATE.last_result

    if _STATE.cluster_block_until and now >= _STATE.cluster_block_until:
        _STATE.cluster_block_until = None

    if _STATE.cluster_block_until and now < _STATE.cluster_block_until:
        remaining = max(0.0, (_STATE.cluster_block_until - now).total_seconds())
        _STATE.last_reason = "loss_cluster_cooldown"
        _STATE.last_result = {
            "executed": False,
            "reason": "loss_cluster_cooldown",
            "cooldown_remaining": round(remaining, 1),
        }
        return _STATE.last_result

    if not is_scalp_session_open(now=now):
        _STATE.last_reason = "session_closed"
        _STATE.last_result = {"executed": False, "reason": "session_closed"}
        return _STATE.last_result

    refreshed = False

    async def _maybe_refresh_from_rest() -> None:
        nonlocal refreshed
        global _last_rest_refresh
        if refreshed:
            return
        if _REST_REFRESH_COOLDOWN > 0:
            now_monotonic = time.monotonic()
            if now_monotonic - _last_rest_refresh < _REST_REFRESH_COOLDOWN:
                return
            _last_rest_refresh = now_monotonic
        try:
            candles = await fetch_historical_candles("USD_JPY", "M1", _REST_REFRESH_MAX)
        except Exception as exc:  # noqa: BLE001
            logging.warning("[SCALP] refresh_from_rest_failed: %s", exc)
            return
        try:
            for candle in candles[-min(_REST_REFRESH_MAX, 60):]:
                await on_candle("M1", candle)
            refreshed = True
        except Exception as exc:  # noqa: BLE001
            logging.warning("[SCALP] rest_backfill_error: %s", exc)

    if fac_m1 is None:
        fac_m1 = all_factors().get("M1")
    if not fac_m1 or fac_m1.get("close") is None:
        await _maybe_refresh_from_rest()
        fac_m1 = all_factors().get("M1")
        if not fac_m1 or fac_m1.get("close") is None:
            _STATE.last_reason = "no_factors"
            _STATE.last_result = {"executed": False, "reason": "no_factors"}
            return _STATE.last_result

    market_open, age = is_market_open(fac_m1, now=now, stale_after_sec=config.stale_after_sec)
    if not market_open:
        await _maybe_refresh_from_rest()
        fac_m1 = all_factors().get("M1") or fac_m1
        market_open, age = is_market_open(fac_m1, now=now, stale_after_sec=config.stale_after_sec)
        if not market_open:
            _STATE.last_reason = "stale"
            _STATE.last_result = {"executed": False, "reason": "stale", "age_sec": age}
            return _STATE.last_result

    pip_size = config.pip_size or 0.01
    candles = fac_m1.get("candles") or []
    latest_candle_range_pips: Optional[float] = None
    if candles:
        last_candle = candles[-1]
        try:
            high_val = float(last_candle.get("high"))
            low_val = float(last_candle.get("low"))
        except (TypeError, ValueError):
            high_val = low_val = None
        if (
            high_val is not None
            and low_val is not None
            and pip_size > 0
            and high_val >= low_val
        ):
            latest_candle_range_pips = (high_val - low_val) / pip_size

    _prune_minute_counts(now)

    bias_active, bias_direction = _active_trend_bias(now)
    trend_override = False
    trend_direction: Optional[str] = bias_direction

    if config.enable_trend_follow:
        detected, detected_direction = _detect_trend_override(
            fac_m1,
            config,
            direction_hint=bias_direction,
        )
        if detected and detected_direction:
            trend_override = True
            trend_direction = detected_direction
            _set_trend_bias(
                detected_direction,
                now,
                config.trend_override_duration_sec,
            )
        elif bias_active and bias_direction:
            trend_override = True
            trend_direction = bias_direction
            _set_trend_bias(
                bias_direction,
                now,
                config.trend_override_duration_sec,
            )
        else:
            trend_direction = None
            _STATE.trend_bias_until = None
            _STATE.trend_bias_direction = None
    else:
        trend_direction = None
        _STATE.trend_bias_until = None
        _STATE.trend_bias_direction = None

    pre_range_pips = _range_from_candles(candles, config.pre_range_minutes, pip_size)
    adaptive_pre_threshold = config.pre_range_threshold_pips
    if config.pre_range_relax_count > 0 and _STATE.pre_range_blocks >= config.pre_range_relax_count:
        relax_steps = _STATE.pre_range_blocks - config.pre_range_relax_count + 1
        adaptive_pre_threshold += max(0.0, config.pre_range_relax_add) * relax_steps

    if (
        pre_range_pips is not None
        and adaptive_pre_threshold > 0
        and pre_range_pips >= adaptive_pre_threshold
    ):
        _STATE.pre_range_blocks += 1
        if trend_override and trend_direction:
            logging.info(
                "[SCALP] trend_override pre_range range=%.2f threshold=%.2f blocks=%s direction=%s",
                pre_range_pips,
                adaptive_pre_threshold,
                _STATE.pre_range_blocks,
                trend_direction,
            )
        else:
            _STATE.last_reason = "pre_range_high"
            _STATE.last_result = {
                "executed": False,
                "reason": "pre_range_high",
                "pre_range_pips": round(pre_range_pips, 2),
                "threshold": round(adaptive_pre_threshold, 2),
                "blocks": _STATE.pre_range_blocks,
            }
            logging.info(
                "[SCALP] block reason=pre_range range=%.2f threshold=%.2f blocks=%s",
                pre_range_pips,
                adaptive_pre_threshold,
                _STATE.pre_range_blocks,
            )
            return _STATE.last_result
    else:
        _STATE.pre_range_blocks = 0

    recent_range_pips = _range_from_candles(candles, config.recent_range_minutes, pip_size)
    adaptive_recent_threshold = config.recent_range_threshold_pips
    if (
        config.recent_range_relax_count > 0
        and _STATE.recent_range_blocks >= config.recent_range_relax_count
    ):
        relax_steps = _STATE.recent_range_blocks - config.recent_range_relax_count + 1
        adaptive_recent_threshold += max(0.0, config.recent_range_relax_add) * relax_steps

    if (
        recent_range_pips is not None
        and adaptive_recent_threshold > 0
        and recent_range_pips >= adaptive_recent_threshold
    ):
        _STATE.recent_range_blocks += 1
        if trend_override and trend_direction:
            logging.info(
                "[SCALP] trend_override recent_range range=%.2f threshold=%.2f blocks=%s direction=%s",
                recent_range_pips,
                adaptive_recent_threshold,
                _STATE.recent_range_blocks,
                trend_direction,
            )
        else:
            _STATE.last_reason = "recent_range_high"
            _STATE.last_result = {
                "executed": False,
                "reason": "recent_range_high",
                "recent_range_pips": round(recent_range_pips, 2),
                "threshold": round(adaptive_recent_threshold, 2),
                "blocks": _STATE.recent_range_blocks,
            }
            logging.info(
                "[SCALP] block reason=recent_range range=%.2f threshold=%.2f blocks=%s",
                recent_range_pips,
                adaptive_recent_threshold,
                _STATE.recent_range_blocks,
            )
            return _STATE.last_result
    else:
        _STATE.recent_range_blocks = 0

    cooldown = _remaining_cooldown(config, now)
    if cooldown > 0:
        _STATE.last_reason = "cooldown"
        _STATE.last_result = {"executed": False, "reason": "cooldown", "cooldown_remaining": round(cooldown, 2)}
        return _STATE.last_result

    if config.max_open_positions > 0:
        open_positions = _open_scalp_positions()
        if open_positions >= config.max_open_positions:
            _STATE.last_reason = "max_open_positions"
            _STATE.last_result = {
                "executed": False,
                "reason": "max_open_positions",
                "open_positions": open_positions,
            }
            logging.info(
                "[SCALP] block reason=max_open open=%s limit=%s",
                open_positions,
                config.max_open_positions,
            )
            return _STATE.last_result

    price_raw = fac_m1.get("close")
    try:
        price = float(price_raw)
    except (TypeError, ValueError):
        _STATE.last_reason = "invalid_price"
        _STATE.last_result = {"executed": False, "reason": "invalid_price"}
        return _STATE.last_result

    spread_pips = spread_override if spread_override is not None else config.default_spread_pips
    adjustments = _get_adjustments(now)
    _STATE.adjustments = adjustments
    allow_mean_revert = True
    if config.mean_revert_disabled:
        allow_mean_revert = False
    if adjustments.get("disable_mean_revert"):
        allow_mean_revert = False

    velocity_pips = abs(float(fac_m1.get("tick_velocity_30s", 0.0) or 0.0))
    range_pips = float(fac_m1.get("tick_range_30s", 0.0) or 0.0)
    if (
        (config.max_tick_velocity > 0 and velocity_pips > config.max_tick_velocity)
        or (config.max_tick_range > 0 and range_pips > config.max_tick_range)
    ):
        _STATE.last_reason = "volatility_gate"
        _STATE.last_result = {
            "executed": False,
            "reason": "volatility_gate",
            "velocity_30s": round(velocity_pips, 2),
            "range_30s": round(range_pips, 2),
        }
        logging.info(
            "[SCALP] block reason=volatility velocity=%.2f range=%.2f limits=(%.2f, %.2f)",
            velocity_pips,
            range_pips,
            config.max_tick_velocity,
            config.max_tick_range,
        )
        return _STATE.last_result

    loss_cluster_hit, cluster_count = _recent_loss_cluster_active(config, now)
    if loss_cluster_hit:
        cooldown_until = now + timedelta(seconds=config.loss_cluster_cooldown_sec)
        if _STATE.cluster_block_until is None or cooldown_until > _STATE.cluster_block_until:
            _STATE.cluster_block_until = cooldown_until
        remaining = max(0.0, (_STATE.cluster_block_until - now).total_seconds())
        _STATE.last_reason = "loss_cluster"
        _STATE.last_result = {
            "executed": False,
            "reason": "loss_cluster",
            "cluster_count": cluster_count,
            "window_sec": config.loss_cluster_window_sec,
            "cooldown_sec": round(remaining, 1),
        }
        logging.info(
            "[SCALP] block reason=loss_cluster count=%s window=%ss cooldown=%.1fs",
            cluster_count,
            config.loss_cluster_window_sec,
            remaining,
        )
        return _STATE.last_result

    loss_streak = _recent_loss_streak(now, config.loss_streak_window_sec)
    deviation_override = max(
        0.0,
        config.deviation_pips
        + adjustments.get("deviation_offset", 0.0)
        + loss_streak * config.loss_streak_deviation_boost,
    )
    force_trend = False
    forced_direction: Optional[str] = None
    if trend_override and trend_direction:
        force_trend = True
        forced_direction = trend_direction
    else:
        bias_active_final, bias_direction_final = _active_trend_bias(now)
        if bias_active_final and bias_direction_final and config.enable_trend_follow:
            force_trend = True
            forced_direction = bias_direction_final

    signal, eval_reason = _evaluate_signal(
        config,
        fac_m1,
        spread_pips,
        deviation_override,
        momentum_flip_pips=config.momentum_flip_pips
        + config.loss_streak_deviation_boost * loss_streak,
        momentum_confirm_pips=config.momentum_confirm_pips
        + config.loss_streak_deviation_boost * max(loss_streak - 1, 0),
        momentum_velocity_cap=config.momentum_velocity_cap,
        momentum_range_min=config.momentum_range_min,
        enable_trend_follow=config.enable_trend_follow,
        trend_velocity_min=config.trend_velocity_min,
        trend_range_min=config.trend_range_min,
        trend_momentum_min=config.trend_momentum_min,
        trend_rsi_buy=config.trend_rsi_buy,
        trend_rsi_sell=config.trend_rsi_sell,
        trend_sl_pips=config.trend_sl_pips,
        trend_tp_multiplier=config.trend_tp_multiplier,
        candle_range_pips=latest_candle_range_pips or 0.0,
        revert_range_block_pips=config.revert_range_block_pips,
        revert_range_widen_pips=config.revert_range_widen_pips,
        revert_range_sl_boost=config.revert_range_sl_boost,
        revert_gap_pips=config.revert_gap_pips,
        revert_rsi_long_max=config.revert_rsi_long_max,
        revert_rsi_short_min=config.revert_rsi_short_min,
        force_trend=force_trend,
        forced_direction=forced_direction,
        allow_mean_revert=allow_mean_revert,
    )
    if not signal:
        atr_dbg = float(fac_m1.get("atr", 0.0) or 0.0) / config.pip_size
        candle_dbg = (
            round(latest_candle_range_pips, 2)
            if latest_candle_range_pips is not None
            else None
        )
        log_msg = "[SCALP] skip reason=%s spread=%.2f atr=%.2f loss_streak=%s"
        log_args = [eval_reason, spread_pips, atr_dbg, loss_streak]
        if candle_dbg is not None:
            log_msg += " candle_range=%.2f"
            log_args.append(candle_dbg)
        logging.info(log_msg, *log_args)
        _STATE.last_reason = eval_reason
        _STATE.last_result = {
            "executed": False,
            "reason": eval_reason,
            "adjustments": adjustments,
        }
        if candle_dbg is not None:
            _STATE.last_result["candle_range_pips"] = candle_dbg
        return _STATE.last_result

    if not can_trade("scalp"):
        _STATE.last_reason = "dd_limit"
        _STATE.last_result = {"executed": False, "reason": "dd_limit", "adjustments": adjustments}
        return _STATE.last_result

    ma10_val = fac_m1.get("ma10", price)
    try:
        ma10_val = float(ma10_val)
    except (TypeError, ValueError):
        ma10_val = price
    atr_pips = float(fac_m1.get("atr", 0.0) or 0.0) / config.pip_size
    deviation_pips = abs(price - ma10_val) / config.pip_size

    minute_key_for_order: Optional[str] = None
    if config.per_minute_entry_cap > 0:
        minute_key = _minute_key(now, signal.action)
        count = _STATE.per_minute_counts.get(minute_key, 0)
        if count >= config.per_minute_entry_cap:
            _STATE.last_reason = "minute_cap"
            _STATE.last_result = {
                "executed": False,
                "reason": "minute_cap",
                "count": count,
                "cap": config.per_minute_entry_cap,
                "action": signal.action,
            }
            logging.info(
                "[SCALP] skip reason=minute_cap key=%s count=%s cap=%s",
                minute_key,
                count,
                config.per_minute_entry_cap,
            )
            return _STATE.last_result
        minute_key_for_order = minute_key

    order = await _execute_order(config, signal, price, adjustments.get("lot_multiplier", 1.0))
    if order.get("success"):
        _STATE.last_trade = now
        _STATE.last_reason = "executed"
        _STATE.last_signal = signal.action
        _STATE.last_order_id = order.get("trade_id")
        _STATE.last_error = None
        _STATE.pre_range_blocks = 0
        _STATE.recent_range_blocks = 0
        if minute_key_for_order and config.per_minute_entry_cap > 0:
            _STATE.per_minute_counts[minute_key_for_order] = (
                _STATE.per_minute_counts.get(minute_key_for_order, 0) + 1
            )
        candle_dbg = (
            round(latest_candle_range_pips, 2)
            if latest_candle_range_pips is not None
            else None
        )
        logging.info(
            "[SCALP] executed",
            extra={
                "action": signal.action,
                "trade_id": order.get("trade_id"),
                "sl_pips": signal.sl_pips,
                "tp_pips": signal.tp_pips,
                "lot": order.get("lot"),
                "candle_range_pips": candle_dbg,
            },
        )
        _record_trade(
            now=now,
            signal=signal,
            price=price,
            spread_pips=spread_pips,
            atr_pips=atr_pips,
            deviation_pips=deviation_pips,
            order=order,
        )
        result = {
            "executed": True,
            "reason": "executed",
            "trade_id": order.get("trade_id"),
            "signal": asdict(signal),
            "adjustments": adjustments,
        }
        if candle_dbg is not None:
            result["candle_range_pips"] = candle_dbg
    else:
        _STATE.last_reason = order.get("error") or "order_failed"
        _STATE.last_error = order.get("error")
        logging.warning(
            "[SCALP] order_failed error=%s action=%s sl=%.2f tp=%.2f",
            order.get("error"),
            signal.action,
            signal.sl_pips,
            signal.tp_pips,
        )
        result = {
            "executed": False,
            "reason": _STATE.last_reason,
            "error": order.get("error"),
            "adjustments": adjustments,
        }
        if latest_candle_range_pips is not None:
            result["candle_range_pips"] = round(latest_candle_range_pips, 2)
    _STATE.last_result = result
    return result


async def scalp_loop() -> None:
    logging.info("[SCALP] loop started")
    while True:
        try:
            now = datetime.now(timezone.utc)
            _maybe_refresh_config(now)
            cfg = _CONFIG
            interval = max(cfg.loop_interval_sec, 1.0)
            if cfg.enabled:
                await _attempt_trade(now)
            else:
                _STATE.last_reason = "disabled"
        except Exception:  # noqa: BLE001
            logging.exception("[SCALP] unexpected error in loop")
        interval = max(_CONFIG.loop_interval_sec, 1.0)
        await asyncio.sleep(interval if _CONFIG.enabled else max(interval, 30.0))


def run_scalp_once(
    fac_m1: Dict[str, Any],
    now: datetime,
    *,
    spread_pips: Optional[float] = None,
) -> Dict[str, Any]:
    """Execute a single scalping evaluation synchronously (Cloud Run)."""

    return asyncio.run(_attempt_trade(now, fac_m1=fac_m1, spread_override=spread_pips))
