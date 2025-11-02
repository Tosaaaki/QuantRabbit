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
from collections import deque
from typing import Any, Dict, List, Optional

from indicators.factor_cache import all_factors, on_candle
from analysis.summary_ingestor import check_event_soon
from utils.market_hours import is_market_open, is_scalp_session_open
from execution.risk_guard import MAX_LEVERAGE, allowed_lot, can_trade, clamp_sl_tp
from execution.order_manager import market_order
from strategies.scalp.basic import BasicScalpStrategy, ScalpSignal
from analysis.scalp_learning import load_adjustments
from analysis.scalp_config import ensure_overrides, load_overrides
from analysis.regime_classifier import classify_scalp
from market_data.candle_fetcher import fetch_historical_candles
from utils.trade_repository import get_trade_repository
from execution.position_manager import PositionManager
from execution.account_info import get_account_state


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
        self.min_sl_pips = _float_safe(os.getenv("SCALP_MIN_SL_PIPS"), 7.0)
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
                or 600
            )
        )
        self.loss_cluster_max = int(
            float(max_override or os.getenv("SCALP_LOSS_CLUSTER_MAX") or 3)
        )
        if self.loss_cluster_max > 0 and self.loss_cluster_max > 2:
            # default tighten to 2 if not explicitly overridden
            try:
                if os.getenv("SCALP_LOSS_CLUSTER_MAX") is None:
                    self.loss_cluster_max = 2
            except Exception:
                self.loss_cluster_max = max(2, self.loss_cluster_max)
        self.loss_cluster_cooldown_sec = int(
            float(
                cooldown_override
                or os.getenv("SCALP_LOSS_CLUSTER_COOLDOWN_SEC")
                or 1200
            )
        )
        self.max_open_positions = int(os.getenv("SCALP_MAX_OPEN_POSITIONS", "1"))
        self.loss_streak_window_sec = int(os.getenv("SCALP_LOSS_STREAK_WINDOW_SEC", "1800"))
        self.loss_streak_deviation_boost = _float_safe(os.getenv("SCALP_LOSS_STREAK_DEV_BOOST"), 0.25)
        self.loss_streak_lot_decay = _float_safe(os.getenv("SCALP_LOSS_STREAK_LOT_DECAY"), 0.18)
        self.loss_streak_lot_floor = _float_safe(os.getenv("SCALP_LOSS_STREAK_LOT_FLOOR"), 0.35)
        self.loss_streak_lot_decay = max(0.0, min(0.8, self.loss_streak_lot_decay))
        self.loss_streak_lot_floor = max(0.05, min(1.0, self.loss_streak_lot_floor))
        self.pre_range_minutes = int(os.getenv("SCALP_PRE_RANGE_MINUTES", "15"))
        self.pre_range_threshold_pips = _float_safe(
            os.getenv("SCALP_PRE_RANGE_THRESHOLD_PIPS"), 12.0
        )
        # ATR連動のしきい値係数（pre/recent）
        self.pre_range_atr_k = _float_safe(
            os.getenv("SCALP_PRE_RANGE_ATR_K"), 1.2
        )
        self.recent_range_minutes = int(os.getenv("SCALP_RECENT_RANGE_MINUTES", "5"))
        self.recent_range_threshold_pips = _float_safe(
            os.getenv("SCALP_RECENT_RANGE_THRESHOLD_PIPS"), 4.0
        )
        self.recent_range_atr_k = _float_safe(
            os.getenv("SCALP_RECENT_RANGE_ATR_K"), 0.6
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
        # Mean-revert gating mode: off|extreme|range|auto (default=extreme)
        self.mean_revert_mode = str(
            overrides.get("mean_revert_mode")
            if overrides and overrides.get("mean_revert_mode") is not None
            else os.getenv("SCALP_MEAN_REVERT_MODE", "extreme")
        ).strip().lower() or "extreme"
        # Extreme-mode thresholds
        self.mr_ext_vwap_min = _float_safe(os.getenv("SCALP_MR_EXT_VWAP_MIN"), 0.20)
        self.mr_ext_dev_atr_k = _float_safe(os.getenv("SCALP_MR_EXT_DEV_ATR_K"), 0.35)
        self.mr_ext_rsi_long_max = _float_safe(
            overrides.get("mr_ext_rsi_long_max"),
            _float_safe(os.getenv("SCALP_MR_EXT_RSI_LONG_MAX"), 34.0),
        )
        self.mr_ext_rsi_short_min = _float_safe(
            overrides.get("mr_ext_rsi_short_min"),
            _float_safe(os.getenv("SCALP_MR_EXT_RSI_SHORT_MIN"), 66.0),
        )
        # Range-mode thresholds
        self.mr_range_adx_max = _float_safe(os.getenv("SCALP_MR_RANGE_ADX_MAX"), 18.0)
        self.mr_range_bbw_max = _float_safe(os.getenv("SCALP_MR_RANGE_BBW_MAX"), 0.25)
        self.mr_range_velocity_max = _float_safe(os.getenv("SCALP_MR_RANGE_VEL_MAX"), 3.5)
        self.mr_range_range30_max = _float_safe(os.getenv("SCALP_MR_RANGE_RNG30_MAX"), 6.0)
        # Event window guard (±minutes)
        self.disable_on_event = os.getenv("SCALP_DISABLE_ON_EVENT", "true").lower() == "true"
        self.event_window_min = int(os.getenv("SCALP_EVENT_WINDOW_MIN", "30"))
        self.event_min_impact = int(os.getenv("SCALP_EVENT_MIN_IMPACT", "3"))
        # レンジ超過時のソフト許容幅（ATR基準）と劣化モード設定
        self.range_soft_over_atr_k = _float_safe(
            os.getenv("SCALP_RANGE_SOFT_OVER_ATR_K"), 0.5
        )
        self.degrade_lot_factor = _float_safe(os.getenv("SCALP_DEGRADE_LOT_FACTOR"), 0.5)
        self.degrade_widen_mult = _float_safe(os.getenv("SCALP_DEGRADE_WIDEN_MULT"), 1.5)
        self.degrade_sl_mult = _float_safe(os.getenv("SCALP_DEGRADE_SL_MULT"), 1.2)
        # セッション別ロット係数
        self.session_mult_asia = _float_safe(os.getenv("SCALP_SESSION_MULT_ASIA"), 1.0)
        self.session_mult_europe = _float_safe(os.getenv("SCALP_SESSION_MULT_EUROPE"), 1.0)
        self.session_mult_us = _float_safe(os.getenv("SCALP_SESSION_MULT_US"), 1.0)
        self.session_mult_late = _float_safe(os.getenv("SCALP_SESSION_MULT_LATE"), 0.9)
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
        # Revert-specific SL policy (only applied to mean-revert entries)
        self.revert_sl_atr_k = _float_safe(
            os.getenv("SCALP_REVERT_SL_ATR_K"), 2.2
        )
        self.revert_sl_min_pips = _float_safe(
            os.getenv("SCALP_REVERT_SL_MIN_PIPS"), 8.0
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
    current_regime: str = "init"


_OVERRIDES = ensure_overrides()


def _float_safe(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


_ENV_BASE_LOT = _float_safe(os.getenv("SCALP_BASE_LOT"), 0.008)
_INITIAL_BASE_LOT = _float_safe(_OVERRIDES.get("base_lot"), _ENV_BASE_LOT)
MARGIN_BUFFER_RATIO = max(
    0.0, min(1.0, _float_safe(os.getenv("MARGIN_BUFFER_RATIO"), 0.98))
)


_CONFIG = ScalpConfig(overrides=_OVERRIDES)
_CONFIG_REFRESH_SEC = int(os.getenv("SCALP_CONFIG_REFRESH_SEC", "180"))
_CONFIG_SIGNATURE = tuple(sorted(_OVERRIDES.items())) if _OVERRIDES else None
_CONFIG_LAST_REFRESH = 0.0
_STATE = ScalpRuntimeState()

# Tick buffer for 30s metrics (mid-price). Size ~120 ticks (sufficient for ~60s at 0.5s cadence)
_TICK_BUF = deque(maxlen=240)

def _tick_metrics(now: datetime, price: float, pip: float) -> Dict[str, float]:
    try:
        p = float(price)
    except (TypeError, ValueError):
        return {"tick_velocity_30s": 0.0, "tick_range_30s": 0.0}
    _TICK_BUF.append((now, p))
    # prune older than 30s
    cutoff = now - timedelta(seconds=30)
    while _TICK_BUF and _TICK_BUF[0][0] < cutoff:
        _TICK_BUF.popleft()
    if not _TICK_BUF:
        return {"tick_velocity_30s": 0.0, "tick_range_30s": 0.0, "tick_momentum_5": 0.0, "tick_momentum_10": 0.0}
    ref = _TICK_BUF[0][1]
    velocity = (p - ref) / (pip or 0.01)
    prices = [pp for _, pp in _TICK_BUF]
    rng = (max(prices) - min(prices)) / (pip or 0.01)
    # Momentum by N ticks back (if available)
    try:
        mom5 = (p - _TICK_BUF[-6][1]) / (pip or 0.01) if len(_TICK_BUF) >= 6 else 0.0
    except Exception:
        mom5 = 0.0
    try:
        mom10 = (p - _TICK_BUF[-11][1]) / (pip or 0.01) if len(_TICK_BUF) >= 11 else 0.0
    except Exception:
        mom10 = 0.0
    return {
        "tick_velocity_30s": velocity,
        "tick_range_30s": rng,
        "tick_momentum_5": mom5,
        "tick_momentum_10": mom10,
    }

def get_tick_metrics_snapshot(pip: float = 0.01) -> Dict[str, float]:
    """Return current tick metrics snapshot without appending a new tick.

    Uses the contents of the internal 30s buffer. Safe for Cloud Run to
    consult after feeding the latest mid price via run_scalp_once.
    """
    if not _TICK_BUF:
        return {"tick_velocity_30s": 0.0, "tick_range_30s": 0.0, "tick_momentum_5": 0.0, "tick_momentum_10": 0.0}
    now, last_price = _TICK_BUF[-1]
    cutoff = now - timedelta(seconds=30)
    # Find reference price at 30s cutoff
    ref_price = _TICK_BUF[0][1]
    for t0, p0 in _TICK_BUF:
        if t0 >= cutoff:
            ref_price = p0
            break
    velocity = (last_price - ref_price) / (pip or 0.01)
    # Recompute from buffer directly
    window_prices = [p for t, p in _TICK_BUF if t >= cutoff]
    rng = ((max(window_prices) - min(window_prices)) / (pip or 0.01)) if window_prices else 0.0
    mom5 = (last_price - _TICK_BUF[-6][1]) / (pip or 0.01) if len(_TICK_BUF) >= 6 else 0.0
    mom10 = (last_price - _TICK_BUF[-11][1]) / (pip or 0.01) if len(_TICK_BUF) >= 11 else 0.0
    return {
        "tick_velocity_30s": velocity,
        "tick_range_30s": rng,
        "tick_momentum_5": mom5,
        "tick_momentum_10": mom10,
    }

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

_TRADE_REPO = get_trade_repository()

_ADJUST_REFRESH_SEC = int(os.getenv("SCALP_LEARNING_REFRESH_SEC", "600"))
_ADJUST_CACHE: Dict[str, Any] = {"ts": None, "data": {"lot_multiplier": 1.0, "deviation_offset": 0.0}}
_PM: Optional[PositionManager] = None

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
    if config.loss_cluster_max <= 0 or config.loss_cluster_window_sec <= 0:
        return False, 0
    cutoff_dt = now - timedelta(seconds=config.loss_cluster_window_sec)
    limit = max(config.loss_cluster_max * 6, 40)
    records = _TRADE_REPO.recent_trades("scalp", limit=limit, closed_only=True)
    stop_losses = 0
    for rec in records:
        if rec.close_time is None or rec.close_time < cutoff_dt:
            continue
        if (rec.close_reason or "").upper() != "STOP_LOSS_ORDER":
            continue
        stop_losses += 1
    return (stop_losses >= config.loss_cluster_max, stop_losses)


def _open_scalp_positions() -> int:
    count = _TRADE_REPO.count_open_trades("scalp")
    if count is None:
        logging.warning("[SCALP] Unable to determine open scalp positions (trade data unavailable)")
        return 0
    return count


def _recent_loss_streak(now: datetime, window_sec: int) -> int:
    if window_sec <= 0:
        return 0
    cutoff_dt = now - timedelta(seconds=window_sec)
    records = _TRADE_REPO.recent_trades("scalp", limit=12, closed_only=True)
    streak = 0
    for rec in records:
        if rec.close_time is None or rec.close_time < cutoff_dt:
            continue
        val = rec.realized_pl if rec.realized_pl is not None else rec.pl_pips
        if val is None:
            continue
        try:
            val = float(val)
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
    try:
        tm = get_tick_metrics_snapshot(pip=_CONFIG.pip_size)
        out.setdefault("tick_velocity_30s", round(float(tm.get("tick_velocity_30s", 0.0)), 3))
        out.setdefault("tick_range_30s", round(float(tm.get("tick_range_30s", 0.0)), 3))
        if "tick_momentum_5" in tm:
            out.setdefault("tick_momentum_5", round(float(tm.get("tick_momentum_5", 0.0)), 3))
        if "tick_momentum_10" in tm:
            out.setdefault("tick_momentum_10", round(float(tm.get("tick_momentum_10", 0.0)), 3))
    except Exception:
        pass
    return out


async def _execute_order(config: ScalpConfig, signal: ScalpSignal, price: float, lot_multiplier: float) -> Dict[str, Any]:
    lot_cap = max(config.base_lot * max(lot_multiplier, 0.0), 0.0)
    try:
        account_snapshot = await get_account_state(ttl=30.0)
    except Exception as exc:
        logging.warning("[SCALP] account snapshot fallback: %s", exc)
        account_snapshot = {}
    account_equity = float(account_snapshot.get("equity") or config.reference_equity)
    margin_available = float(account_snapshot.get("marginAvailable") or 0.0)
    margin_rate = account_snapshot.get("marginRate")
    if not margin_rate or margin_rate <= 0:
        margin_rate = (1.0 / MAX_LEVERAGE) if MAX_LEVERAGE > 0 else 0.05
    usable_margin = max(margin_available, 0.0) * MARGIN_BUFFER_RATIO
    lot_allowed = allowed_lot(
        account_equity,
        signal.sl_pips,
        margin_available=usable_margin,
        margin_rate=margin_rate,
    )
    lot = min(lot_cap, lot_allowed)
    if lot <= 0:
        return {
            "success": False,
            "error": "lot_zero",
            "lot": lot,
            "margin_available": margin_available,
        }

    units = int(round(lot * 100000))
    if units == 0:
        return {"success": False, "error": "lot_zero", "lot": lot, "units": units}
    if signal.action == "sell":
        units *= -1

    required_margin = abs(units) * margin_rate
    if required_margin > usable_margin:
        logging.info(
            "[SCALP] margin insufficient required=%.2f available=%.2f",
            required_margin,
            usable_margin,
        )
        return {
            "success": False,
            "error": "margin_insufficient",
            "lot": lot,
            "units": units,
            "required_margin": required_margin,
            "margin_available": margin_available,
        }

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
        price_anchor=price,
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
    revert_gap_pips: float,
    revert_rsi_long_max: float,
    revert_rsi_short_min: float,
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
        revert_gap_pips=revert_gap_pips,
        revert_rsi_long_max=revert_rsi_long_max,
        revert_rsi_short_min=revert_rsi_short_min,
        revert_sl_atr_k=config.revert_sl_atr_k,
        revert_sl_min_pips=config.revert_sl_min_pips,
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


def _log_position_entry(now: datetime, order: Dict[str, Any], price: float) -> None:
    trade_id = order.get("trade_id")
    units = order.get("units")
    if not trade_id or units is None:
        return
    lot = order.get("lot")
    try:
        pm = _get_position_manager()
    except Exception as exc:  # noqa: BLE001
        logging.warning("[SCALP] position_manager_unavailable: %s", exc)
        return
    try:
        pm.log_local_entry(
            ticket_id=str(trade_id),
            pocket=BasicScalpStrategy.pocket,
            strategy=BasicScalpStrategy.name,
            units=int(units),
            entry_price=float(price),
            entry_time=now.isoformat(timespec="seconds"),
        )
    except Exception as exc:  # noqa: BLE001
        logging.warning("[SCALP] log_local_entry failed: %s", exc, extra={"trade_id": trade_id, "lot": lot})


def _get_position_manager() -> PositionManager:
    global _PM
    if _PM is None:
        _PM = PositionManager()
    return _PM


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
    tick_price: Optional[float] = None,
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

    # Optional: block new scalp entries around high-impact events
    try:
        if _CONFIG.disable_on_event and check_event_soon(
            within_minutes=_CONFIG.event_window_min, min_impact=_CONFIG.event_min_impact
        ):
            _STATE.last_reason = "event_window"
            _STATE.last_result = {"executed": False, "reason": "event_window"}
            return _STATE.last_result
    except Exception as exc:
        logging.warning("[SCALP] event check failed: %s", exc)

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
    atr_pips_dyn = float(fac_m1.get("atr", 0.0) or 0.0) / pip_size if pip_size > 0 else 0.0
    adaptive_pre_threshold = config.pre_range_threshold_pips + max(0.0, config.pre_range_atr_k) * max(0.0, atr_pips_dyn)
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
    adaptive_recent_threshold = config.recent_range_threshold_pips + max(0.0, config.recent_range_atr_k) * max(0.0, atr_pips_dyn)
    degrade_mode = False
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
        # ソフト許容（ATR基準）
        soft_over = max(0.0, config.range_soft_over_atr_k) * max(0.0, atr_pips_dyn)
        if soft_over > 0 and recent_range_pips <= adaptive_recent_threshold + soft_over:
            degrade_mode = True
            logging.info(
                "[SCALP] degrade recent_range=%.2f threshold=%.2f(+%.2f) blocks=%s",
                recent_range_pips,
                adaptive_recent_threshold,
                soft_over,
                _STATE.recent_range_blocks,
            )
        elif trend_override and trend_direction:
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

    # Prefer real-time tick price if provided; fallback to M1 close
    price: float
    if tick_price is not None:
        try:
            price = float(tick_price)
        except (TypeError, ValueError):
            price = float(fac_m1.get("close") or 0.0)
    else:
        price = float(fac_m1.get("close") or 0.0)
    if not price:
        _STATE.last_reason = "invalid_price"
        _STATE.last_result = {"executed": False, "reason": "invalid_price"}
        return _STATE.last_result

    spread_pips = spread_override if spread_override is not None else config.default_spread_pips
    adjustments = _get_adjustments(now)
    _STATE.adjustments = adjustments
    scalp_regime = classify_scalp(fac_m1)
    _STATE.current_regime = scalp_regime
    def _allow_mean_revert_by_mode(cfg: ScalpConfig, fac: Dict[str, Any]) -> tuple[bool, str]:
        mode = (cfg.mean_revert_mode or "extreme").lower()
        if cfg.mean_revert_disabled:
            return False, "mr_mode_disabled"
        if adjustments.get("disable_mean_revert"):
            return False, "mr_learning_disabled"
        if mode == "off":
            return False, "mr_mode_off"
        rsi = float(fac.get("rsi", 50.0) or 50.0)
        adx = float(fac.get("adx", 0.0) or 0.0)
        bbw = float(fac.get("bbw", 1.0) or 1.0)
        vwap_delta = float(fac.get("tick_vwap_delta", 0.0) or 0.0)
        velocity = abs(float(fac.get("tick_velocity_30s", 0.0) or 0.0))
        range30 = float(fac.get("tick_range_30s", 0.0) or 0.0)
        price = float(fac.get("close", 0.0) or 0.0)
        ma10 = float(fac.get("ma10", 0.0) or 0.0)
        atr = float(fac.get("atr", 0.0) or 0.0)
        pip = cfg.pip_size or 0.01
        atr_pips = (atr / pip) if pip > 0 else 0.0
        deviation_pips = abs(price - ma10) / (pip or 0.01)

        def _extreme() -> bool:
            if atr_pips <= 0.0:
                return False
            if abs(vwap_delta) < cfg.mr_ext_vwap_min:
                return False
            if deviation_pips < max(cfg.revert_gap_pips, atr_pips * cfg.mr_ext_dev_atr_k):
                return False
            if not (rsi <= cfg.mr_ext_rsi_long_max or rsi >= cfg.mr_ext_rsi_short_min):
                return False
            return True

        def _range() -> bool:
            if adx > cfg.mr_range_adx_max:
                return False
            if bbw > cfg.mr_range_bbw_max:
                return False
            if velocity > cfg.mr_range_velocity_max:
                return False
            if range30 > cfg.mr_range_range30_max:
                return False
            return True

        if mode == "extreme":
            return (_extreme(), "mr_extreme" if _extreme() else "mr_not_extreme")
        if mode == "range":
            return (_range(), "mr_range" if _range() else "mr_not_range")
        # auto: choose by environment
        return (_extreme() if velocity >= 4.0 or range30 >= 6.0 else _range(), "mr_auto")

    allow_mean_revert, mr_reason = _allow_mean_revert_by_mode(config, fac_m1)
    if scalp_regime == "Breakout":
        allow_mean_revert = False
        mr_reason = "scalp_breakout"
    elif scalp_regime == "Range" and not config.mean_revert_disabled:
        allow_mean_revert = True
        mr_reason = "scalp_range"
    elif scalp_regime == "Trend" and not trend_override:
        ma_ref = float(fac_m1.get("ma20", price) or price)
        direction = "long" if price >= ma_ref else "short"
        trend_override = True
        trend_direction = direction
        _set_trend_bias(direction, now, max(config.trend_override_duration_sec, 45))

    # Compute tick-based metrics locally to avoid dependency on external tick stream
    tickm = _tick_metrics(now, price, pip_size)
    velocity_pips = abs(float(tickm.get("tick_velocity_30s", 0.0)))
    range_pips = float(tickm.get("tick_range_30s", 0.0) or 0.0)
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

    # If mean‑revert is not allowed by mode, short‑circuit here before evaluation
    if not allow_mean_revert:
        _STATE.last_reason = mr_reason
        _STATE.last_result = {
            "executed": False,
            "reason": mr_reason,
        }
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

    # 劣化モード時はレンジ耐性を強化
    revert_widen = config.revert_range_widen_pips * (config.degrade_widen_mult if degrade_mode else 1.0)
    revert_sl_boost = config.revert_range_sl_boost * (config.degrade_sl_mult if degrade_mode else 1.0)

    fac_eval = dict(fac_m1)
    try:
        fac_eval.update({k: float(v) for k, v in tickm.items()})
    except Exception:
        fac_eval.update(tickm)

    signal, eval_reason = _evaluate_signal(
        config,
        fac_eval,
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
        revert_range_widen_pips=revert_widen,
        revert_range_sl_boost=revert_sl_boost,
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

    # セッション/学習ベースのロット係数 + 劣化モード時の縮小
    def _session_label(dt: datetime) -> str:
        h = dt.hour
        if 0 <= h < 7:
            return "asia"
        if 7 <= h < 12:
            return "europe"
        if 12 <= h < 20:
            return "us"
        return "late"
    sess = _session_label(now)
    sess_mult = {
        "asia": config.session_mult_asia,
        "europe": config.session_mult_europe,
        "us": config.session_mult_us,
        "late": config.session_mult_late,
    }.get(sess, 1.0)
    try:
        learn_mult = float(risk_multiplier("scalp", BasicScalpStrategy.name))
    except Exception:
        learn_mult = 1.0
    streak_decay = 1.0 - max(0, loss_streak) * config.loss_streak_lot_decay
    streak_decay = max(config.loss_streak_lot_floor, streak_decay)
    lot_mult_effective = (
        adjustments.get("lot_multiplier", 1.0)
        * sess_mult
        * learn_mult
        * (config.degrade_lot_factor if degrade_mode else 1.0)
        * streak_decay
    )
    order = await _execute_order(config, signal, price, lot_mult_effective)
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
        try:
            _log_position_entry(now, order, price)
        except Exception as exc:  # noqa: BLE001
            logging.warning("[SCALP] log_position_entry error: %s", exc)
        result = {
            "executed": True,
            "reason": "executed",
            "trade_id": order.get("trade_id"),
            "signal": asdict(signal),
            "adjustments": adjustments,
            "lot_multiplier": round(lot_mult_effective, 3),
        }
        result["scalp_regime"] = scalp_regime
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
        result["scalp_regime"] = scalp_regime
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
    tick_price: Optional[float] = None,
) -> Dict[str, Any]:
    """Execute a single scalping evaluation synchronously (Cloud Run)."""

    return asyncio.run(_attempt_trade(now, fac_m1=fac_m1, spread_override=spread_pips, tick_price=tick_price))
