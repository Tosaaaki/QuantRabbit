"""Worker that owns micro-pocket strategy execution."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from analysis import policy_bus
from analysis.range_guard import detect_range_mode
from analytics.realtime_metrics_client import (
    ConfidencePolicy,
    RealtimeMetricsClient,
)
from execution.order_manager import market_order
from execution.position_manager import PositionManager
from execution.risk_guard import (
    allowed_lot,
    build_exposure_state,
    can_trade,
    update_dd_context,
)
from execution.stage_tracker import StageTracker
from indicators.factor_cache import all_factors
from market_data import spread_monitor
from strategies.mean_reversion.bb_rsi import BBRsi
from strategies.micro.trend_momentum import TrendMomentumMicro
from strategies.micro.momentum_burst import MomentumBurstMicro
from strategies.micro.range_break import MicroRangeBreak
from strategies.micro.pullback_ema import MicroPullbackEMA
from strategies.micro.momentum_stack import MicroMomentumStack
from strategies.micro.level_reactor import MicroLevelReactor
from strategies.micro.vwap_bound_revert import MicroVWAPBound
from utils.metrics_logger import log_metric
from utils.oanda_account import get_account_snapshot
from analysis.perf_monitor import snapshot as get_perf
from utils.hold_monitor import HoldMonitor

from . import config

LOG = logging.getLogger(__name__)

POCKET = "micro"
PIP = 0.01
DEFAULT_STRATEGIES = [
    "MicroVWAPBound",
    "BB_RSI",
    "MicroRangeBreak",
    "TrendMomentumMicro",
    "MicroMomentumStack",
    "MicroPullbackEMA",
    "MicroLevelReactor",
    "MomentumBurst",
]
POCKET_STRATEGY_MAP = {
    "micro": {
        "BB_RSI",
        "MicroRangeBreak",
        "TrendMomentumMicro",
        "MicroMomentumStack",
        "MicroPullbackEMA",
        "MicroLevelReactor",
        "MomentumBurst",
        "MicroVWAPBound",
    }
}
BASE_RISK_PCT = 0.02
FALLBACK_EQUITY = 10_000.0
ENTRY_COOLDOWN_SEC = config.COOLDOWN_BASE_SEC
STRATEGY_CLASSES = {
    "BB_RSI": BBRsi,
    "MicroRangeBreak": MicroRangeBreak,
    "TrendMomentumMicro": TrendMomentumMicro,
    "MicroMomentumStack": MicroMomentumStack,
    "MicroPullbackEMA": MicroPullbackEMA,
    "MicroLevelReactor": MicroLevelReactor,
    "MomentumBurst": MomentumBurstMicro,
    "MicroVWAPBound": MicroVWAPBound,
}
def _env_set(name: str, default: str = "") -> set[str]:
    raw = os.getenv(name)
    if raw is None:
        raw = default
    return {item.strip() for item in raw.split(",") if item.strip()}

MANUAL_SENTINEL_POCKETS = set()
MANUAL_SENTINEL_MIN_UNITS = int(os.getenv("MANUAL_SENTINEL_MIN_UNITS", "0"))
MANUAL_SENTINEL_RELEASE_CYCLES = max(
    1, int(os.getenv("MANUAL_SENTINEL_RELEASE_CYCLES", "1"))
)
MANUAL_SENTINEL_BLOCK_POCKETS = _env_set("MANUAL_SENTINEL_BLOCK_POCKETS", "")
HOLD_RATIO_LOOKBACK_HOURS = float(os.getenv("HOLD_RATIO_LOOKBACK_HOURS", "6.0"))
HOLD_RATIO_MIN_SAMPLES = int(os.getenv("HOLD_RATIO_MIN_SAMPLES", "80"))
HOLD_RATIO_MAX = float(os.getenv("HOLD_RATIO_MAX", "0.30"))
HOLD_RATIO_RELEASE_FACTOR = float(os.getenv("HOLD_RATIO_RELEASE_FACTOR", "0.8"))
HOLD_RATIO_CHECK_INTERVAL_SEC = float(os.getenv("HOLD_RATIO_CHECK_INTERVAL_SEC", "900"))
HOLD_RATIO_GUARD_DISABLED = str(os.getenv("HOLD_RATIO_GUARD_DISABLED", "1")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MARGIN_GUARD_THRESHOLD = float(os.getenv("MICRO_MARGIN_GUARD_THRESHOLD", "0.025"))
MARGIN_GUARD_RELEASE = float(os.getenv("MICRO_MARGIN_GUARD_RELEASE", "0.035"))
MARGIN_GUARD_CHECK_INTERVAL = float(os.getenv("MICRO_MARGIN_GUARD_CHECK_INTERVAL", "75"))
MARGIN_GUARD_STOP = float(os.getenv("MICRO_MARGIN_GUARD_STOP", "0.018"))
MARGIN_BUFFER_EWMA_ALPHA = float(os.getenv("MICRO_MARGIN_BUFFER_EWMA_ALPHA", "0.2"))
MARGIN_BUFFER_HYST = float(os.getenv("MICRO_MARGIN_BUFFER_HYST", "0.002"))
TREND_GAP_PIPS = float(os.getenv("MICRO_TREND_GAP_PIPS", "0.50"))
TREND_ADX_MIN = float(os.getenv("MICRO_TREND_ADX_MIN", "21.0"))
MICRO_SPREAD_DAMP_PIPS = (
    float(os.getenv("MICRO_SPREAD_WARN_PIPS", "0.80")),
    float(os.getenv("MICRO_SPREAD_ALERT_PIPS", "1.00")),
    float(os.getenv("MICRO_SPREAD_BLOCK_PIPS", "1.20")),
)
MICRO_SPREAD_DAMP_FACTORS = (
    float(os.getenv("MICRO_SPREAD_WARN_FACTOR", "0.75")),
    float(os.getenv("MICRO_SPREAD_ALERT_FACTOR", "0.55")),
    float(os.getenv("MICRO_SPREAD_BLOCK_FACTOR", "0.35")),
)
MICRO_MARGIN_BUFFER_LIMIT = float(os.getenv("MICRO_MARGIN_BUFFER_LIMIT", "0.04"))
MICRO_MARGIN_BUFFER_FACTOR = float(os.getenv("MICRO_MARGIN_BUFFER_FACTOR", "0.50"))
MICRO_MAX_ACTIVE_TRADES = max(0, int(os.getenv("MICRO_MAX_ACTIVE_TRADES", "6")))
MICRO_MAX_TRADES_PER_DIRECTION = max(
    0, int(os.getenv("MICRO_MAX_TRADES_PER_DIRECTION", "4"))
)
MICRO_MAX_TRADES_PER_STRATEGY = max(
    0, int(os.getenv("MICRO_MAX_TRADES_PER_STRATEGY", "3"))
)
RANGE_STRONG_THRESHOLD = float(os.getenv("MICRO_RANGE_STRONG_THRESHOLD", "0.5"))
RANGE_WEAK_THRESHOLD = float(os.getenv("MICRO_RANGE_WEAK_THRESHOLD", "0.35"))
BB_BASE_MIN_DISTANCE_RANGE = float(os.getenv("MICRO_BB_BASE_DISTANCE_RANGE", "0.05"))
BB_BASE_MIN_DISTANCE_TREND = float(os.getenv("MICRO_BB_BASE_DISTANCE_TREND", "0.12"))
MOMENTUM_STACK_VOL_MIN = float(os.getenv("MICRO_MOMENTUM_STACK_VOL_MIN", "0.65"))
MOMENTUM_STACK_ATR_MIN = float(os.getenv("MICRO_MOMENTUM_STACK_ATR_MIN", "1.3"))
BB_SUPPRESS_GAP_DELTA = float(os.getenv("MICRO_BB_SUPPRESS_GAP_DELTA", "0.35"))
BB_SUPPRESS_ADX_DELTA = float(os.getenv("MICRO_BB_SUPPRESS_ADX_DELTA", "10.0"))


def _derive_dynamic_profile(fac_m1: dict, range_ctx) -> dict:
    def _as_float(val, default=0.0):
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    atr = _as_float(fac_m1.get("atr_pips"))
    vol5 = _as_float(fac_m1.get("vol_5m"))
    adx = _as_float(fac_m1.get("adx"))
    range_score = _as_float(getattr(range_ctx, "score", 0.0))

    gap_dyn = TREND_GAP_PIPS
    if atr > 2.0:
        gap_dyn -= min(0.18, (atr - 2.0) * 0.07)
    elif atr < 1.25:
        gap_dyn += min(0.14, (1.25 - atr) * 0.08)
    if vol5 > 1.1:
        gap_dyn -= 0.02
    elif vol5 < 0.6:
        gap_dyn += 0.02
    gap_dyn = max(0.3, round(gap_dyn, 3))

    adx_dyn = TREND_ADX_MIN
    if atr > 2.2:
        adx_dyn -= min(4.0, (atr - 2.2) * 1.5)
    elif atr < 1.0:
        adx_dyn += min(3.5, (1.0 - atr) * 2.3)
    if vol5 < 0.6:
        adx_dyn += 0.8
    adx_dyn = max(14.0, round(adx_dyn, 2))

    bb_range_min = max(
        0.035,
        BB_BASE_MIN_DISTANCE_RANGE
        - min(0.02, range_score * 0.02)
        + max(0.0, (MOMENTUM_STACK_VOL_MIN - vol5) * 0.015),
    )
    bb_trend_min = max(
        BB_BASE_MIN_DISTANCE_TREND,
        BB_BASE_MIN_DISTANCE_TREND + max(0.0, (0.65 - range_score) * 0.08),
    )

    try:
        ma10 = float(fac_m1.get("ma10"))
        ma20 = float(fac_m1.get("ma20"))
        ma_gap_pips = abs((ma10 - ma20) / PIP)
        ma_gap_signed = (ma10 - ma20) / PIP
    except (TypeError, ValueError):
        ma_gap_pips = 0.0
        ma_gap_signed = 0.0
    strong_candidate = ma_gap_pips >= gap_dyn and adx >= adx_dyn
    momentum_ready = (
        strong_candidate
        and atr >= MOMENTUM_STACK_ATR_MIN
        and vol5 >= MOMENTUM_STACK_VOL_MIN
    )
    bias = "long" if ma_gap_signed > 0 else "short" if ma_gap_signed < 0 else "neutral"
    suppress_bb = strong_candidate and (
        ma_gap_pips >= (gap_dyn + BB_SUPPRESS_GAP_DELTA)
        or adx >= (adx_dyn + BB_SUPPRESS_ADX_DELTA)
        or atr >= (MOMENTUM_STACK_ATR_MIN + 1.2)
    )

    return {
        "trend_gap_dynamic": gap_dyn,
        "trend_adx_dynamic": adx_dyn,
        "bb_min_distance_range": round(bb_range_min, 3),
        "bb_min_distance_trend": round(bb_trend_min, 3),
        "momentum_stack_ready": momentum_ready,
        "momentum_stack_bias": bias,
        "bb_suppress_strong": suppress_bb,
    }


def _classify_regime(range_ctx, fac_m1: dict, fac_h4: dict) -> tuple[str, str]:
    range_score = getattr(range_ctx, "score", 0.0) or 0.0
    try:
        range_score = float(range_score)
    except (TypeError, ValueError):
        range_score = 0.0
    range_active = bool(getattr(range_ctx, "active", False))
    try:
        ma10 = float(fac_m1.get("ma10"))
        ma20 = float(fac_m1.get("ma20"))
    except (TypeError, ValueError):
        ma10 = ma20 = None
    try:
        adx = float(fac_m1.get("adx") or 0.0)
    except (TypeError, ValueError):
        adx = 0.0
    regime_dir = "neutral"
    try:
        dyn_gap = float(fac_m1.get("trend_gap_dynamic") or TREND_GAP_PIPS)
    except (TypeError, ValueError):
        dyn_gap = TREND_GAP_PIPS
    try:
        dyn_adx = float(fac_m1.get("trend_adx_dynamic") or TREND_ADX_MIN)
    except (TypeError, ValueError):
        dyn_adx = TREND_ADX_MIN

    strong_trend = False
    if ma10 is not None and ma20 is not None:
        gap = (ma10 - ma20) / PIP
        if abs(gap) >= dyn_gap and adx >= dyn_adx:
            strong_trend = True
            regime_dir = "long" if gap > 0 else "short"
    if range_active and range_score >= RANGE_STRONG_THRESHOLD and not strong_trend:
        return "range", regime_dir
    if strong_trend:
        return f"trend_{regime_dir}", regime_dir

    try:
        ma10_h4 = float(fac_h4.get("ma10"))
        ma20_h4 = float(fac_h4.get("ma20"))
    except (TypeError, ValueError):
        ma10_h4 = ma20_h4 = None

    if ma10_h4 is not None and ma20_h4 is not None:
        gap_h4 = (ma10_h4 - ma20_h4) / PIP
        h4_gap_req = max(dyn_gap * 1.3, TREND_GAP_PIPS * 1.3)
        adx_req = max(dyn_adx - 1.0, TREND_ADX_MIN - 1.0)
        if abs(gap_h4) >= h4_gap_req and adx >= adx_req:
            regime_dir = "long" if gap_h4 > 0 else "short"
            return f"trend_{regime_dir}", regime_dir

    if range_active and range_score >= RANGE_WEAK_THRESHOLD and not strong_trend:
        return "range_soft", regime_dir

    return "neutral", regime_dir


def _build_client_order_id(strategy: str) -> str:
    ts_ms = int(time.time() * 1000)
    digest = hashlib.sha1(
        f"{ts_ms}-{strategy}-{os.getpid()}".encode("utf-8")
    ).hexdigest()[:9]
    return f"qr-micro-{ts_ms}-{strategy[:4]}-{digest}"


def _ma_bias(factors: Optional[dict], *, threshold: float = 0.002) -> str:
    if not factors:
        return "neutral"
    try:
        ma10 = float(factors.get("ma10"))
        ma20 = float(factors.get("ma20"))
    except (TypeError, ValueError):
        return "neutral"
    diff = ma10 - ma20
    if diff > threshold:
        return "long"
    if diff < -threshold:
        return "short"
    return "neutral"


def _default_micro_policy() -> Dict[str, Any]:
    return {
        "enabled": True,
        "bias": "neutral",
        "confidence": 0.0,
        "units_cap": None,
        "entry_gates": {
            "allow_new": True,
            "require_retest": False,
            "spread_ok": True,
            "event_ok": True,
        },
        "exit_profile": {
            "reverse_threshold": 70,
            "allow_negative_exit": False,
        },
        "be_profile": {
            "enabled": True,
            "trigger_pips": 2.4,
            "cooldown_sec": 45.0,
            "lock_ratio": 0.5,
            "min_lock_pips": 0.35,
        },
        "partial_profile": {
            "thresholds_pips": [1.6, 3.0],
            "fractions": [0.45, 0.30],
            "min_units": 40,
        },
        "strategies": [],
        "pending_orders": [],
    }


def _select_reduction_candidate(open_trades: List[dict]) -> Optional[dict]:
    """
    Pick one trade to trim based on loss size, margin weight, and pocket priority.
    Returns only a candidate; caller decides whether/how to execute reduce-only.
    """
    if not open_trades:
        return None

    pocket_priority = {"scalp": 0, "scalp_fast": 0, "micro": 1, "macro": 2, "manual": 3}

    def _score(tr: dict) -> tuple:
        unreal = tr.get("unrealized_pl") or 0.0
        units = abs(int(tr.get("units", 0) or 0))
        pocket = str(tr.get("pocket", "")).lower()
        pprio = pocket_priority.get(pocket, 4)
        # loss first (more negative wins), then larger units, then pocket priority
        return (unreal, -units, pprio)

    enriched: List[dict] = []
    for tr in open_trades:
        p = tr.get("pocket")
        if not p:
            thesis = tr.get("entry_thesis") or {}
            if isinstance(thesis, dict):
                p = thesis.get("pocket")
        e = dict(tr)
        if p:
            e["pocket"] = p
        enriched.append(e)
    enriched.sort(key=_score)
    return enriched[0] if enriched else None


def _micro_flow_factor(
    spread_pips: Optional[float],
    margin_buffer: Optional[float],
) -> tuple[float, List[str]]:
    factor = 1.0
    reasons: List[str] = []
    try:
        spread_val = float(spread_pips) if spread_pips is not None else None
    except (TypeError, ValueError):
        spread_val = None
    warn_p, alert_p, block_p = MICRO_SPREAD_DAMP_PIPS
    warn_f, alert_f, block_f = MICRO_SPREAD_DAMP_FACTORS
    if spread_val is not None:
        if spread_val >= block_p:
            factor *= block_f
            reasons.append(f"spread={spread_val:.2f}p(block)")
        elif spread_val >= alert_p:
            factor *= alert_f
            reasons.append(f"spread={spread_val:.2f}p(alert)")
        elif spread_val >= warn_p:
            factor *= warn_f
            reasons.append(f"spread={spread_val:.2f}p(warn)")
    if margin_buffer is not None and margin_buffer < MICRO_MARGIN_BUFFER_LIMIT:
        factor *= MICRO_MARGIN_BUFFER_FACTOR
        reasons.append(f"margin={margin_buffer:.3f}")
    return max(0.0, min(1.0, factor)), reasons


def _trade_strategy_tag(trade: dict) -> str:
    if not trade:
        return ""
    tag = trade.get("strategy_tag")
    if tag:
        return str(tag)
    thesis = trade.get("entry_thesis")
    if isinstance(thesis, dict):
        tag = thesis.get("strategy_tag") or thesis.get("strategy")
        if tag:
            return str(tag)
    client_id = trade.get("client_id")
    return str(client_id or "")


def _build_trade_counters(open_trades: List[dict]) -> Dict[str, Any]:
    counters: Dict[str, Any] = {
        "total": 0,
        "direction": {"long": 0, "short": 0},
        "strategy": {},
    }
    for tr in open_trades or []:
        direction = tr.get("side") or "long"
        if direction not in counters["direction"]:
            counters["direction"][direction] = 0
        counters["direction"][direction] += 1
        counters["total"] += 1
        strat_tag = _trade_strategy_tag(tr)
        if strat_tag:
            counters["strategy"][strat_tag] = counters["strategy"].get(strat_tag, 0) + 1
    return counters


async def micro_core_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled by configuration", config.LOG_PREFIX)
        return

    stage_tracker = StageTracker()
    pos_manager = PositionManager()
    metrics_client = RealtimeMetricsClient()
    confidence_policy = ConfidencePolicy()
    last_published_version = 0
    cooldown_rsi: Dict[str, float] = {}
    hold_monitor = HoldMonitor(
        db_path=Path(os.getenv("HOLD_RATIO_DB", "logs/trades.db")),
        lookback_hours=HOLD_RATIO_LOOKBACK_HOURS,
        min_samples=HOLD_RATIO_MIN_SAMPLES,
    )
    last_hold_ratio_check = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
    hold_ratio_guard_active = False
    manual_guard_active = False
    manual_clear_cycles = 0
    margin_guard_active = False
    margin_guard_buffer: Optional[float] = None
    margin_buffer_smoothed: Optional[float] = None
    last_margin_guard_check = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
    last_regime_mode: Optional[str] = None

    last_dyn_profile: Optional[dict] = None
    last_stack_ready = False

    try:
        while True:
            now = datetime.datetime.now(datetime.timezone.utc)
            factors = all_factors()
            fac_m1 = dict(factors.get("M1") or {})
            fac_h4 = factors.get("H4") or {}
            if not fac_m1.get("close"):
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            try:
                blocked, _, spread_state, reason = spread_monitor.is_blocked()
            except Exception:
                blocked = False
                spread_state = None
                reason = ""
            if blocked:
                LOG.info(
                    "%s spread gate active (%s)",
                    config.LOG_PREFIX,
                    reason or "guard",
                )
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue
            spread_pips = 0.0
            if spread_state and isinstance(spread_state, dict):
                try:
                    spread_pips = float(spread_state.get("spread_pips") or 0.0)
                except (TypeError, ValueError):
                    spread_pips = 0.0
            fac_m1["spread_pips"] = spread_pips
            if HOLD_RATIO_GUARD_DISABLED:
                hold_ratio_guard_active = False
            elif (now - last_hold_ratio_check).total_seconds() >= HOLD_RATIO_CHECK_INTERVAL_SEC:
                ratio, total, lt60 = hold_monitor.sample()
                last_hold_ratio_check = now
                if ratio is not None:
                    if (not hold_ratio_guard_active) and ratio > HOLD_RATIO_MAX:
                        hold_ratio_guard_active = True
                        log_metric(
                            "hold_ratio_guard",
                            1.0,
                            tags={"scope": "micro_worker", "ratio": f"{ratio:.3f}", "samples": str(total)},
                        )
                        LOG.warning(
                            "%s hold ratio guard activated (ratio=%.1f%% total=%s)",
                            config.LOG_PREFIX,
                            ratio * 100,
                            total,
                        )
                    elif hold_ratio_guard_active and ratio < HOLD_RATIO_MAX * HOLD_RATIO_RELEASE_FACTOR:
                        hold_ratio_guard_active = False
                        log_metric(
                            "hold_ratio_guard",
                            0.0,
                            tags={"scope": "micro_worker", "ratio": f"{ratio:.3f}", "samples": str(total)},
                        )
                        LOG.info(
                            "%s hold ratio guard released (ratio=%.1f%% total=%s)",
                            config.LOG_PREFIX,
                            ratio * 100,
                            total,
                        )
            range_ctx = detect_range_mode(fac_m1, fac_h4)
            dyn_profile = _derive_dynamic_profile(fac_m1, range_ctx)
            if config.DYNAMIC_LOG_ENABLED:
                log_changed = False
                if last_dyn_profile is None:
                    log_changed = True
                else:
                    for key in (
                        "trend_gap_dynamic",
                        "trend_adx_dynamic",
                        "bb_min_distance_range",
                        "bb_min_distance_trend",
                    ):
                        prev_val = float(last_dyn_profile.get(key, 0.0))
                        curr_val = float(dyn_profile.get(key, 0.0))
                        if abs(curr_val - prev_val) > 0.02:
                            log_changed = True
                            break
                    if not log_changed and dyn_profile.get("momentum_stack_bias") != last_dyn_profile.get(
                        "momentum_stack_bias"
                    ):
                        log_changed = True
                atr_curr = float(fac_m1.get("atr_pips") or 0.0)
                vol_curr = float(fac_m1.get("vol_5m") or 0.0)
                adx_curr = float(fac_m1.get("adx") or 0.0)
                if log_changed:
                    LOG.info(
                        "%s dyn gap=%.2f adx=%.1f bb_range=%.3f bb_trend=%.3f atr=%.2f vol=%.2f stack=%s bias=%s bb_suppress=%s",
                        config.LOG_PREFIX,
                        float(dyn_profile.get("trend_gap_dynamic", 0.0)),
                        float(dyn_profile.get("trend_adx_dynamic", 0.0)),
                        float(dyn_profile.get("bb_min_distance_range", 0.0)),
                        float(dyn_profile.get("bb_min_distance_trend", 0.0)),
                        atr_curr,
                        vol_curr,
                        bool(dyn_profile.get("momentum_stack_ready")),
                        dyn_profile.get("momentum_stack_bias"),
                        bool(dyn_profile.get("bb_suppress_strong")),
                    )
                stack_ready = bool(dyn_profile.get("momentum_stack_ready"))
                if last_dyn_profile is not None and stack_ready != last_stack_ready:
                    LOG.info(
                        "%s momentum_stack_ready=%s (gap=%.2f adx=%.1f)",
                        config.LOG_PREFIX,
                        stack_ready,
                        float(dyn_profile.get("trend_gap_dynamic", 0.0)),
                        adx_curr,
                    )
                last_dyn_profile = dyn_profile.copy()
                last_stack_ready = stack_ready
            fac_m1.update(dyn_profile)
            regime_mode, regime_direction = _classify_regime(range_ctx, fac_m1, fac_h4)
            try:
                range_score = float(getattr(range_ctx, "score", 0.0) or 0.0)
            except (TypeError, ValueError):
                range_score = 0.0
            if (
                last_regime_mode in {"range", "range_soft"}
                and regime_mode.startswith("trend")
            ):
                LOG.info(
                    "%s regime shift range->%s (range_score=%.2f adx=%.2f)",
                    config.LOG_PREFIX,
                    regime_mode,
                    range_score,
                    fac_m1.get("adx") or 0.0,
                )
            last_regime_mode = regime_mode
            if (
                MARGIN_GUARD_THRESHOLD > 0.0
                and (now - last_margin_guard_check).total_seconds()
                >= MARGIN_GUARD_CHECK_INTERVAL
            ):
                try:
                    snapshot = get_account_snapshot()
                    margin_guard_buffer = (
                        snapshot.health_buffer if snapshot else None
                    )
                except Exception as exc:  # noqa: BLE001
                    LOG.warning(
                        "%s margin snapshot failed: %s",
                        config.LOG_PREFIX,
                        exc,
                    )
                    margin_guard_buffer = None
                last_margin_guard_check = now
                if margin_guard_buffer is not None:
                    if margin_buffer_smoothed is None:
                        margin_buffer_smoothed = margin_guard_buffer
                    else:
                        margin_buffer_smoothed = (
                            margin_buffer_smoothed * (1.0 - MARGIN_BUFFER_EWMA_ALPHA)
                            + margin_guard_buffer * MARGIN_BUFFER_EWMA_ALPHA
                        )
                    try:
                        log_metric(
                            "margin_buffer",
                            margin_guard_buffer,
                            tags={
                                "smoothed": f"{margin_buffer_smoothed:.4f}",
                                "alpha": f"{MARGIN_BUFFER_EWMA_ALPHA:.2f}",
                            },
                        )
                    except Exception:
                        pass
                if margin_guard_buffer is not None:
                    buffer_val = (
                        margin_buffer_smoothed
                        if margin_buffer_smoothed is not None
                        else margin_guard_buffer
                    )
                    start_thresh = max(0.0, MARGIN_GUARD_STOP - MARGIN_BUFFER_HYST)
                    release_thresh = MARGIN_GUARD_RELEASE + MARGIN_BUFFER_HYST
                    if buffer_val < start_thresh:
                        if not margin_guard_active:
                            margin_guard_active = True
                            LOG.warning(
                                "%s margin guard activated buffer=%.3f",
                                config.LOG_PREFIX,
                                buffer_val,
                            )
                    elif (
                        margin_guard_active
                        and buffer_val >= release_thresh
                    ):
                        margin_guard_active = False
                        LOG.info(
                            "%s margin guard released buffer=%.3f",
                            config.LOG_PREFIX,
                            buffer_val,
                        )
            if margin_guard_active:
                selected = _select_reduction_candidate(open_trades)
                if selected:
                    try:
                        log_metric(
                            "margin_guard_delever_candidate",
                            abs(float(selected.get("units", 0) or 0)),
                            tags={
                                "pocket": str(selected.get("pocket") or ""),
                                "trade_id": str(selected.get("trade_id") or ""),
                                "unreal": f"{selected.get('unrealized_pl', 0.0)}",
                            },
                        )
                    except Exception:
                        pass
                LOG.info(
                    "%s margin guard active buffer=%.3f; skip entries (candidate=%s)",
                    config.LOG_PREFIX,
                    margin_guard_buffer if margin_guard_buffer is not None else -1.0,
                    selected.get("trade_id") if selected else "none",
                )
                await _publish_micro_policy(
                    fac_m1,
                    range_ctx,
                    strategies=ranked_strategies,
                    confidence=0.0,
                    open_trades=open_trades,
                    executed=False,
                    base_version=last_published_version,
                )
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            fac_m1["range_score"] = getattr(range_ctx, "score", 0.0)
            fac_m1["range_active"] = getattr(range_ctx, "active", False)
            fac_m1["range_reason"] = getattr(range_ctx, "reason", "")
            fac_m1["regime_mode"] = regime_mode
            fac_m1["regime_direction"] = regime_direction
            plan_snapshot = policy_bus.latest()
            strategies_hint: List[str] = []
            if plan_snapshot and getattr(plan_snapshot, "pockets", None):
                raw = plan_snapshot.pockets.get(POCKET)
                if isinstance(raw, dict):
                    hint = raw.get("strategies")
                    if isinstance(hint, list):
                        strategies_hint = [str(s) for s in hint if str(s)]
                last_published_version = plan_snapshot.version

            ranked_strategies = strategies_hint or list(DEFAULT_STRATEGIES)
            pref_range = [
                "MicroVWAPBound",
                "MicroRangeBreak",
                "BB_RSI",
                "MicroMomentumStack",
                "MicroPullbackEMA",
                "TrendMomentumMicro",
            ]
            pref_trend = [
                "TrendMomentumMicro",
                "MicroMomentumStack",
                "MicroPullbackEMA",
                "BB_RSI",
                "MicroRangeBreak",
            ]
            preference = pref_range if regime_mode.startswith("range") else pref_trend
            def _sort_key(name: str) -> int:
                try:
                    return preference.index(name)
                except ValueError:
                    return len(preference)
            ranked_strategies = list(dict.fromkeys(sorted(ranked_strategies, key=_sort_key)))
            perf_cache = get_perf()

            positions = pos_manager.get_open_positions()
            micro_info = positions.get(POCKET) or {}
            open_trades = micro_info.get("open_trades") or []
            manual_block, manual_units, manual_details = _manual_guard_state(positions)
            # fallback: if no manual/unknown exposure in openTrades, force-clear sentinel
            meta = positions.get("__meta__", {})
            if manual_block and meta and meta.get("consecutive_failures", 0) == 0:
                manual_units = 0
                manual_details = "auto_clear_no_exposure"
                manual_block = False
            if manual_block:
                manual_clear_cycles = 0
                if not manual_guard_active:
                    manual_guard_active = True
                    log_metric(
                        "manual_halt_active",
                        1.0,
                        tags={"scope": "micro_worker", "units": str(manual_units)},
                    )
                    LOG.warning(
                        "%s manual/unknown exposure detected (units=%s detail=%s); continuing with reduced headroom.",
                        config.LOG_PREFIX,
                        manual_units,
                        manual_details or "-",
                    )
            if manual_guard_active and not manual_block:
                manual_clear_cycles += 1
                if manual_clear_cycles >= MANUAL_SENTINEL_RELEASE_CYCLES:
                    manual_guard_active = False
                    log_metric(
                        "manual_halt_active",
                        0.0,
                        tags={"scope": "micro_worker", "units": "0"},
                    )
                    LOG.info("%s manual/unknown exposure cleared", config.LOG_PREFIX)
            elif not manual_guard_active:
                manual_clear_cycles = min(
                    manual_clear_cycles + 1, MANUAL_SENTINEL_RELEASE_CYCLES
                )
            if hold_ratio_guard_active:
                LOG.warning(
                    "%s hold ratio guard active; halt new entries",
                    config.LOG_PREFIX,
                )
                await _publish_micro_policy(
                    fac_m1,
                    range_ctx,
                    strategies=ranked_strategies,
                    confidence=0.0,
                    open_trades=open_trades,
                    base_version=last_published_version,
                )
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            signals = _collect_micro_signals(
                ranked_strategies,
                fac_m1,
                metrics_client,
                confidence_policy,
                regime_mode,
                regime_direction,
            )

            trade_allowed = can_trade(POCKET)
            if not signals or not trade_allowed:
                if not signals:
                    LOG.info(
                        "%s no signals (regime=%s dir=%s spread=%.2fp adx=%.1f rsi=%.1f atr=%.2f)",
                        config.LOG_PREFIX,
                        getattr(regime_mode, "name", regime_mode),
                        getattr(regime_direction, "name", regime_direction),
                        spread_pips,
                        float(fac_m1.get("adx") or 0.0),
                        float(fac_m1.get("rsi") or 0.0),
                        float(fac_m1.get("atr_pips") or 0.0),
                    )
                if not trade_allowed:
                    LOG.warning(
                        "%s pocket drawdown guard blocks entries (open_trades=%d units=%d)",
                        config.LOG_PREFIX,
                        len(open_trades),
                        int(abs(micro_info.get("units", 0) or 0)),
                    )
                await _publish_micro_policy(
                    fac_m1,
                    range_ctx,
                    strategies=ranked_strategies,
                    confidence=0.0,
                    open_trades=open_trades,
                    base_version=last_published_version,
                )
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            try:
                snapshot = get_account_snapshot()
                equity = snapshot.nav or snapshot.balance or FALLBACK_EQUITY
                margin_available = snapshot.margin_available
                margin_rate = snapshot.margin_rate
                margin_used = snapshot.margin_used
            except Exception:
                equity = FALLBACK_EQUITY
                margin_available = None
                margin_rate = None
                margin_used = None

            avg_sl = max(1.0, sum(s["sl_pips"] for s in signals) / len(signals))
            lot_total = allowed_lot(
                equity,
                sl_pips=avg_sl,
                margin_available=margin_available,
                price=fac_m1.get("close"),
                margin_rate=margin_rate,
                risk_pct_override=BASE_RISK_PCT,
            )
            exposure_state = build_exposure_state(
                positions,
                equity=equity,
                price=fac_m1.get("close"),
                margin_used=margin_used,
                margin_available=margin_available,
                margin_rate=margin_rate,
            )
            if exposure_state:
                lot_total = min(
                    lot_total,
                    exposure_state.available_units() / 100000.0,
                )
            update_dd_context(equity, weight_macro=0.0, scalp_share=0.0)

            flow_factor, flow_reasons = _micro_flow_factor(
                spread_pips, margin_guard_buffer
            )
            if flow_factor < 0.999:
                scaled = round(lot_total * flow_factor, 4)
                LOG.info(
                    "%s lot scaled %.4f -> %.4f (%s)",
                    config.LOG_PREFIX,
                    lot_total,
                    scaled,
                    ", ".join(flow_reasons) or "pressure",
                )
                lot_total = scaled
            if lot_total <= 0:
                await _publish_micro_policy(
                    fac_m1,
                    range_ctx,
                    strategies=ranked_strategies,
                    confidence=0.0,
                    open_trades=open_trades,
                    executed=False,
                    base_version=last_published_version,
                )
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            trade_counters = _build_trade_counters(open_trades)
            executed_ids: Set[str] = set()
            executed_any = False
            now = datetime.datetime.now(datetime.timezone.utc)
            atr_pips = fac_m1.get("atr_pips")
            if atr_pips is None:
                atr_raw = fac_m1.get("atr")
                atr_pips = (atr_raw or 0.0) * 100.0
            try:
                atr_pips = float(atr_pips or 0.0)
            except (TypeError, ValueError):
                atr_pips = 0.0
            atr_ratio = atr_pips / config.COOLDOWN_ATR_REF_PIPS if config.COOLDOWN_ATR_REF_PIPS else 1.0
            atr_ratio = max(config.COOLDOWN_ATR_MIN_FACTOR, min(config.COOLDOWN_ATR_MAX_FACTOR, atr_ratio or 0.0))
            dynamic_cooldown = max(
                5,
                int(round(config.COOLDOWN_BASE_SEC * atr_ratio)),
            )

            for signal in signals:
                strategy = signal["strategy"]
                confidence = max(0, min(100, signal["confidence"]))
                confidence_factor = max(0.2, confidence / 100.0)
                staged_lot = lot_total * confidence_factor
                units = int(round(staged_lot * 100000))
                if units < 1000:
                    continue

                action = signal["action"]
                direction = "long" if action == "OPEN_LONG" else "short"
                if (
                    MICRO_MAX_ACTIVE_TRADES
                    and trade_counters["total"] >= MICRO_MAX_ACTIVE_TRADES
                ):
                    LOG.debug(
                        "%s active trade cap reached (%s); halt new entries",
                        config.LOG_PREFIX,
                        MICRO_MAX_ACTIVE_TRADES,
                    )
                    break
                dir_count = trade_counters["direction"].get(direction, 0)
                if (
                    MICRO_MAX_TRADES_PER_DIRECTION
                    and dir_count >= MICRO_MAX_TRADES_PER_DIRECTION
                ):
                    LOG.debug(
                        "%s direction cap reached dir=%s active=%s limit=%s",
                        config.LOG_PREFIX,
                        direction,
                        dir_count,
                        MICRO_MAX_TRADES_PER_DIRECTION,
                    )
                    continue
                strat_tag_entry = signal.get("tag") or strategy
                if (
                    MICRO_MAX_TRADES_PER_STRATEGY
                    and trade_counters["strategy"].get(strat_tag_entry, 0)
                    >= MICRO_MAX_TRADES_PER_STRATEGY
                ):
                    LOG.debug(
                        "%s strategy cap reached strategy=%s",
                        config.LOG_PREFIX,
                        strat_tag_entry,
                    )
                    continue
                blocked_cd, remain, reason = stage_tracker.is_blocked(
                    POCKET, direction, now
                )
                if blocked_cd:
                    key = f"{POCKET}:{direction}"
                    rsi_val = fac_m1.get("rsi")
                    release = False
                    if (
                        remain
                        and remain >= config.COOLDOWN_RELEASE_MIN_REMAIN_SEC
                        and rsi_val is not None
                        and key in cooldown_rsi
                        and spread_pips <= config.COOLDOWN_RELEASE_MAX_SPREAD
                    ):
                        try:
                            current_rsi = float(rsi_val)
                        except (TypeError, ValueError):
                            current_rsi = None
                        if current_rsi is not None:
                            prev_rsi = cooldown_rsi.get(key)
                            if prev_rsi is not None and abs(current_rsi - prev_rsi) >= config.COOLDOWN_RELEASE_RSI_DELTA:
                                if stage_tracker.clear_cooldown(POCKET, direction):
                                    cooldown_rsi.pop(key, None)
                                    LOG.info(
                                        "%s cooldown cleared due to RSI reversal dir=%s delta=%.2f",
                                        config.LOG_PREFIX,
                                        direction,
                                        current_rsi - prev_rsi,
                                    )
                                    blocked_cd = False
                                else:
                                    LOG.debug("%s cooldown clear requested but no record found", config.LOG_PREFIX)
                    if blocked_cd:
                        LOG.info(
                            "%s cooldown pocket=%s dir=%s remain=%s reason=%s",
                            config.LOG_PREFIX,
                            POCKET,
                            direction,
                            remain,
                            reason or "cooldown",
                        )
                        continue

                client_id = _build_client_order_id(strategy)
                if client_id in executed_ids:
                    continue

                sl = signal["sl_pips"] * PIP
                tp = signal["tp_pips"] * PIP
                price = float(fac_m1.get("close") or 0.0)
                if action == "OPEN_LONG":
                    sl_price = round(price - sl, 3) if signal["sl_pips"] > 0 else None
                    tp_price = round(price + tp, 3) if signal["tp_pips"] > 0 else None
                    units_signed = units
                else:
                    sl_price = round(price + sl, 3) if signal["sl_pips"] > 0 else None
                    tp_price = round(price - tp, 3) if signal["tp_pips"] > 0 else None
                    units_signed = -units
                if exposure_state and exposure_state.would_exceed(units_signed):
                    LOG.info(
                        "%s exposure cap reached ratio=%.3f; skip %s",
                        config.LOG_PREFIX,
                        exposure_state.ratio(),
                        strategy,
                    )
                    continue

                try:
                    trade_id = await market_order(
                        "USD_JPY",
                        units_signed,
                        sl_price,
                        tp_price,
                        POCKET,
                        client_order_id=client_id,
                        entry_thesis={
                            "strategy_tag": signal.get("tag") or strategy,
                            "strategy": strategy,
                            "pocket": POCKET,
                            "reason": signal.get("tag"),
                            "profile": signal.get("profile"),
                            "min_hold_sec": signal.get("min_hold_sec"),
                            "loss_guard_pips": signal.get("loss_guard_pips"),
                            "target_tp_pips": signal.get("target_tp_pips") or signal.get("tp_pips"),
                            "sl_pips": signal.get("sl_pips"),
                            "tp_pips": signal.get("tp_pips"),
                        },
                        meta={"entry_price": price},
                    )
                except Exception as exc:  # noqa: BLE001
                    LOG.error(
                        "%s order error strategy=%s err=%s",
                        config.LOG_PREFIX,
                        strategy,
                        exc,
                    )
                    continue

                if not trade_id:
                    LOG.error(
                        "%s order rejected strategy=%s",
                        config.LOG_PREFIX,
                        strategy,
                    )
                    continue

                LOG.info(
                    "%s entry trade=%s strategy=%s units=%s sl=%s tp=%s",
                    config.LOG_PREFIX,
                    trade_id,
                    strategy,
                    units_signed,
                    f"{sl_price:.3f}" if sl_price is not None else "None",
                    f"{tp_price:.3f}" if tp_price is not None else "None",
                )
                pos_manager.register_open_trade(trade_id, POCKET, client_id)
                stage_tracker.set_stage(POCKET, direction, 1, now=now)
                stage_tracker.set_cooldown(
                    POCKET,
                    direction,
                    reason="entry_rate_limit",
                    seconds=dynamic_cooldown,
                    now=now,
                )
                try:
                    cooldown_rsi[f"{POCKET}:{direction}"] = float(fac_m1.get("rsi") or 50.0)
                except (TypeError, ValueError):
                    cooldown_rsi[f"{POCKET}:{direction}"] = 50.0
                executed_ids.add(client_id)
                executed_any = True
                if exposure_state:
                    exposure_state.allocate(units_signed)
                trade_counters["total"] += 1
                trade_counters["direction"][direction] = (
                    trade_counters["direction"].get(direction, 0) + 1
                )
                if strat_tag_entry:
                    trade_counters["strategy"][strat_tag_entry] = (
                        trade_counters["strategy"].get(strat_tag_entry, 0) + 1
                    )
                break

            await _publish_micro_policy(
                fac_m1,
                range_ctx,
                strategies=ranked_strategies,
                confidence=sum(s.get("confidence", 0) for s in signals)
                / max(1, len(signals)),
                open_trades=open_trades,
                executed=executed_any,
                base_version=last_published_version,
            )

            await asyncio.sleep(config.LOOP_INTERVAL_SEC)
    except asyncio.CancelledError:
        LOG.info("%s worker cancelled", config.LOG_PREFIX)
        raise
    finally:
        try:
            stage_tracker.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            pos_manager.close()
        except Exception:  # noqa: BLE001
            pass


def _collect_micro_signals(
    ranked: List[str],
    fac_m1: dict,
    metrics_client: RealtimeMetricsClient,
    confidence_policy: ConfidencePolicy,
    regime_mode: str,
    regime_direction: str,
) -> List[dict]:
    signals: List[dict] = []
    try:
        ma10 = float(fac_m1.get("ma10"))
        ma20 = float(fac_m1.get("ma20"))
        ma_gap_pips = abs((ma10 - ma20) / PIP)
    except (TypeError, ValueError):
        ma_gap_pips = 0.0
    try:
        adx_val = float(fac_m1.get("adx") or 0.0)
    except (TypeError, ValueError):
        adx_val = 0.0
    try:
        dyn_gap = float(fac_m1.get("trend_gap_dynamic") or TREND_GAP_PIPS)
    except (TypeError, ValueError):
        dyn_gap = TREND_GAP_PIPS
    try:
        dyn_adx = float(fac_m1.get("trend_adx_dynamic") or TREND_ADX_MIN)
    except (TypeError, ValueError):
        dyn_adx = TREND_ADX_MIN
    strong_trend_bias = ma_gap_pips >= dyn_gap and adx_val >= dyn_adx
    bb_suppress = bool(fac_m1.get("bb_suppress_strong"))

    allow_range = regime_mode.startswith("range")
    allow_trend = regime_mode.startswith("trend")
    allow_stack = allow_trend and bool(fac_m1.get("momentum_stack_ready"))
    for sname in ranked:
        cls = STRATEGY_CLASSES.get(sname)
        if not cls:
            continue
        if sname not in POCKET_STRATEGY_MAP["micro"]:
            continue
        if sname == "MicroVWAPBound" and (not allow_range or not fac_m1.get("range_active")):
            continue
        if sname == "BB_RSI" and (not allow_range or strong_trend_bias or bb_suppress):
            continue
        if sname == "MicroRangeBreak" and not allow_range:
            continue
        if sname == "TrendMomentumMicro" and not allow_trend:
            continue
        if sname == "MicroPullbackEMA" and not allow_trend:
            continue
        if sname == "MicroMomentumStack" and not allow_stack:
            continue
        raw_signal = cls.check(fac_m1)

        if not raw_signal:
            continue
        action = raw_signal.get("action")
        if action not in {"OPEN_LONG", "OPEN_SHORT"}:
            continue
        if sname in {"TrendMomentumMicro", "MicroPullbackEMA", "MicroMomentumStack"} and regime_direction in {"long", "short"}:
            desired = "OPEN_LONG" if regime_direction == "long" else "OPEN_SHORT"
            if action != desired:
                continue
        signal = {
            "strategy": sname,
            "action": action,
            "sl_pips": float(raw_signal.get("sl_pips") or 0.0),
            "tp_pips": float(raw_signal.get("tp_pips") or 0.0),
            "confidence": int(raw_signal.get("confidence", 50) or 50),
            "tag": raw_signal.get("tag", sname),
        }
        # Apply strategy health scaling
        confidence_policy.reset()
        health = metrics_client.evaluate(sname, POCKET)
        health = confidence_policy.apply(health)
        if not health.allowed:
            continue
        scaled_conf = int(signal["confidence"] * health.confidence_scale)
        signal["confidence"] = max(0, min(100, scaled_conf))
        signals.append(signal)
    return signals


async def _publish_micro_policy(
    fac_m1: dict,
    range_ctx,
    *,
    strategies: List[str],
    confidence: float,
    open_trades: List[dict],
    executed: bool = False,
    base_version: int = 0,
) -> None:
    snapshot = policy_bus.latest()
    if snapshot:
        data = snapshot.to_dict()
    else:
        data = {
            "version": base_version,
            "generated_ts": time.time(),
            "air_score": getattr(range_ctx, "score", 0.0),
            "uncertainty": 0.0,
            "event_lock": False,
            "range_mode": getattr(range_ctx, "active", False),
            "notes": {},
            "pockets": {},
        }

    pockets = data.setdefault("pockets", {})
    micro_entry = pockets.get(POCKET) or _default_micro_policy()
    micro_entry.update(
        {
            "bias": _ma_bias(fac_m1),
            "confidence": max(0.0, min(1.0, confidence / 100.0)),
            "strategies": list(strategies),
            "pending_orders": [],
            "open_trades": len(open_trades),
            "executed": executed,
        }
    )
    pockets[POCKET] = micro_entry
    data["version"] = max(base_version, int(data.get("version", 0))) + 1
    data["generated_ts"] = time.time()
    policy_bus.publish(data)
def _manual_guard_state(positions: Dict[str, Dict]) -> tuple[bool, int, str]:
    total_units = 0
    pockets: list[str] = []
    for name in MANUAL_SENTINEL_POCKETS:
        info = positions.get(name) or {}
        units = int(abs(info.get("units", 0) or 0))
        if units > 0:
            total_units += units
            pockets.append(f"{name}:{units}")
    # manual sentinel を当面無効化して、誤検知でブロックしないようにする
    return False, total_units, ",".join(pockets)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )


def _main() -> None:  # pragma: no cover
    _configure_logging()
    LOG.info("%s worker starting", config.LOG_PREFIX)
    asyncio.run(micro_core_worker())


if __name__ == "__main__":  # pragma: no cover
    _main()
