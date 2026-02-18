"""Per-trade EXIT loop for M1Scalper (scalp pocket) – 利確優先 + 安全弁。"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Sequence, Set

from analysis.range_guard import detect_range_mode
from workers.common.exit_forecast import (
    apply_exit_forecast_to_targets,
    build_exit_forecast_adjustment,
)
from workers.common.exit_scaling import momentum_scale, scale_value
from workers.common.exit_utils import close_trade, mark_pnl_pips
from workers.common.rollout_gate import load_rollout_start_ts, trade_passes_rollout
from execution.position_manager import PositionManager
from indicators.factor_cache import all_factors
from market_data import tick_window
from utils.metrics_logger import log_metric
from workers.common.pro_stop import maybe_close_pro_stop


from . import config
from utils.env_utils import env_bool, env_float

_BB_ENV_PREFIX = getattr(config, "ENV_PREFIX", "")
_BB_EXIT_ENABLED = env_bool("BB_EXIT_ENABLED", True, prefix=_BB_ENV_PREFIX)
_BB_EXIT_REVERT_PIPS = env_float("BB_EXIT_REVERT_PIPS", 2.0, prefix=_BB_ENV_PREFIX)
_BB_EXIT_REVERT_RATIO = env_float("BB_EXIT_REVERT_RATIO", 0.20, prefix=_BB_ENV_PREFIX)
_BB_EXIT_TREND_EXT_PIPS = env_float("BB_EXIT_TREND_EXT_PIPS", 3.0, prefix=_BB_ENV_PREFIX)
_BB_EXIT_TREND_EXT_RATIO = env_float("BB_EXIT_TREND_EXT_RATIO", 0.35, prefix=_BB_ENV_PREFIX)
_BB_EXIT_SCALP_REVERT_PIPS = env_float("BB_EXIT_SCALP_REVERT_PIPS", 1.6, prefix=_BB_ENV_PREFIX)
_BB_EXIT_SCALP_REVERT_RATIO = env_float("BB_EXIT_SCALP_REVERT_RATIO", 0.18, prefix=_BB_ENV_PREFIX)
_BB_EXIT_SCALP_EXT_PIPS = env_float("BB_EXIT_SCALP_EXT_PIPS", 2.0, prefix=_BB_ENV_PREFIX)
_BB_EXIT_SCALP_EXT_RATIO = env_float("BB_EXIT_SCALP_EXT_RATIO", 0.28, prefix=_BB_ENV_PREFIX)
_BB_EXIT_MID_BUFFER_PIPS = env_float("BB_EXIT_MID_BUFFER_PIPS", 0.4, prefix=_BB_ENV_PREFIX)
_BB_EXIT_BYPASS_TOKENS = {
    "hard_stop",
    "structure",
    "time_stop",
    "timeout",
    "max_hold",
    "max_adverse",
    "force",
    "margin",
    "risk",
    "health",
    "event",
    "session",
    "halt",
    "liquid",
}
_BB_EXIT_TF = "M1"
_BB_PIP = 0.01

def _env_bool_opt(name: str) -> Optional[bool]:
    raw = os.getenv(name)
    if raw is None:
        return None
    return raw.strip().lower() in {"1", "true", "yes"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes"}


_REENTRY_ENABLED = _env_bool_opt("M1SCALP_REENTRY_ENABLE")
if _REENTRY_ENABLED is None:
    _REENTRY_ENABLED = _env_bool_opt("REENTRY_ENABLE_ALL")
if _REENTRY_ENABLED is None:
    _REENTRY_ENABLED = False
_REENTRY_SHADOW = _env_bool_opt("M1SCALP_REENTRY_SHADOW")
if _REENTRY_SHADOW is None:
    _REENTRY_SHADOW = _env_bool_opt("REENTRY_SHADOW_ALL")
if _REENTRY_SHADOW is None:
    _REENTRY_SHADOW = True
_REENTRY_REVERT_MIN = float(os.getenv("M1SCALP_REENTRY_REVERT_MIN", "0.65"))
_REENTRY_TREND_MIN = float(os.getenv("M1SCALP_REENTRY_TREND_MIN", "0.60"))
_REENTRY_TREND_MAX = float(os.getenv("M1SCALP_REENTRY_TREND_MAX", "0.45"))
_REENTRY_EDGE_MIN = float(os.getenv("M1SCALP_REENTRY_EDGE_MIN", "0.55"))
_REENTRY_MIN_ADVERSE_PIPS = float(os.getenv("M1SCALP_REENTRY_MIN_ADVERSE_PIPS", "2.5"))
_REENTRY_MIN_ADVERSE_ATR = float(os.getenv("M1SCALP_REENTRY_MIN_ADVERSE_ATR", "1.0"))
_REENTRY_LOG_INTERVAL_SEC = float(os.getenv("M1SCALP_REENTRY_LOG_INTERVAL_SEC", "8.0"))
_LAST_REENTRY_LOG_TS: float = 0.0


def _bb_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bb_levels(fac):
    if not fac:
        return None
    upper = _bb_float(fac.get("bb_upper"))
    lower = _bb_float(fac.get("bb_lower"))
    mid = _bb_float(fac.get("bb_mid")) or _bb_float(fac.get("ma20"))
    bbw = _bb_float(fac.get("bbw")) or 0.0
    if upper is None or lower is None:
        if mid is None or bbw <= 0:
            return None
        half = abs(mid) * bbw / 2.0
        upper = mid + half
        lower = mid - half
    span = upper - lower
    if span <= 0:
        return None
    return upper, mid if mid is not None else (upper + lower) / 2.0, lower, span, span / _BB_PIP


def _bb_exit_price(fac):
    price = None
    latest_mid = globals().get("_latest_mid")
    if callable(latest_mid):
        try:
            price = latest_mid()
        except Exception:
            price = None
    if price is None:
        latest_quote = globals().get("_latest_quote")
        if callable(latest_quote):
            try:
                _, _, mid = latest_quote()
                price = mid
            except Exception:
                price = None
    if price is None or price <= 0:
        try:
            price = float(fac.get("close") or 0.0)
        except Exception:
            price = None
    return price


def _bb_exit_should_bypass(reason, pnl, allow_negative):
    if allow_negative:
        return True
    if pnl is not None:
        try:
            if float(pnl) <= 0:
                return True
        except Exception:
            pass
    if not reason:
        return False
    reason_key = str(reason).lower()
    for token in _BB_EXIT_BYPASS_TOKENS:
        if token in reason_key:
            return True
    return False


def _bb_exit_allowed(style, side, price, fac, *, range_active=None):
    if not _BB_EXIT_ENABLED:
        return True
    if price is None or price <= 0:
        return True
    levels = _bb_levels(fac)
    if not levels:
        return True
    upper, mid, lower, span, span_pips = levels
    side_key = str(side or "").lower()
    if side_key in {"buy", "long", "open_long"}:
        direction = "long"
    else:
        direction = "short"
    orig_style = style
    if style == "scalp" and range_active:
        style = "reversion"
    mid_buffer = max(_BB_EXIT_MID_BUFFER_PIPS, span_pips * 0.05)
    if style == "reversion":
        base_pips = _BB_EXIT_SCALP_REVERT_PIPS if orig_style == "scalp" else _BB_EXIT_REVERT_PIPS
        base_ratio = _BB_EXIT_SCALP_REVERT_RATIO if orig_style == "scalp" else _BB_EXIT_REVERT_RATIO
        threshold = max(base_pips, span_pips * base_ratio)
        if direction == "long":
            dist = (price - lower) / _BB_PIP
        else:
            dist = (upper - price) / _BB_PIP
        return dist >= threshold
    band_buffer = max(_BB_EXIT_TREND_EXT_PIPS, span_pips * _BB_EXIT_TREND_EXT_RATIO)
    if orig_style == "scalp":
        band_buffer = max(_BB_EXIT_SCALP_EXT_PIPS, span_pips * _BB_EXIT_SCALP_EXT_RATIO)
    if direction == "long":
        if price <= mid - mid_buffer * _BB_PIP:
            return True
        return price >= (upper - band_buffer * _BB_PIP)
    if price >= mid + mid_buffer * _BB_PIP:
        return True
    return price <= (lower + band_buffer * _BB_PIP)


def _clamp01(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _norm(value: Optional[float], low: float, high: float) -> Optional[float]:
    if value is None:
        return None
    if high <= low:
        return 0.0
    return _clamp01((float(value) - low) / (high - low))


def _weighted_score(items: Sequence[tuple[Optional[float], float]]) -> Optional[float]:
    total = 0.0
    weight = 0.0
    for value, w in items:
        if value is None:
            continue
        total += float(value) * float(w)
        weight += float(w)
    if weight <= 0:
        return None
    return _clamp01(total / weight)


def _reentry_scores(
    *,
    side: str,
    rsi: Optional[float],
    adx: Optional[float],
    atr_pips: Optional[float],
    bbw: Optional[float],
    vwap_gap: Optional[float],
    ma_pair: Optional[tuple[float, float]],
    range_active: bool,
) -> tuple[Optional[float], Optional[float]]:
    side_key = str(side).lower()
    is_short = side_key in {"short", "sell"}

    rsi_score = None
    if rsi is not None:
        if is_short:
            rsi_score = _norm(rsi, 55.0, 70.0)
        else:
            rsi_score = _norm(45.0 - rsi, 0.0, 15.0)

    adx_revert = _norm(25.0 - adx, 0.0, 10.0) if adx is not None else None
    bbw_revert = _norm(0.22 - bbw, 0.0, 0.12) if bbw is not None else None

    vwap_revert = None
    if vwap_gap is not None:
        gap = vwap_gap if is_short else -vwap_gap
        vwap_revert = _norm(gap, 0.6, 2.4)

    revert_score = _weighted_score(
        [
            (rsi_score, 0.35),
            (adx_revert, 0.25),
            (bbw_revert, 0.25),
            (vwap_revert, 0.15),
        ]
    )

    adx_trend = _norm(adx, 18.0, 35.0) if adx is not None else None
    atr_trend = _norm(atr_pips, 6.0, 14.0) if atr_pips is not None else None
    ma_trend = None
    if ma_pair is not None:
        ma10, ma20 = ma_pair
        if is_short:
            ma_trend = 1.0 if ma10 > ma20 else 0.0
        else:
            ma_trend = 1.0 if ma10 < ma20 else 0.0
    vwap_trend = None
    if vwap_gap is not None:
        gap = vwap_gap if is_short else -vwap_gap
        vwap_trend = _norm(gap, 0.6, 2.6)

    trend_score = _weighted_score(
        [
            (adx_trend, 0.35),
            (atr_trend, 0.30),
            (ma_trend, 0.25),
            (vwap_trend, 0.10),
        ]
    )

    if range_active:
        if revert_score is not None:
            revert_score = _clamp01(revert_score + 0.15)
        if trend_score is not None:
            trend_score = _clamp01(trend_score - 0.15)

    return revert_score, trend_score


def _reentry_edge(adverse_pips: float, atr_pips: Optional[float]) -> float:
    if adverse_pips <= 0:
        return 0.0
    base = adverse_pips / max(atr_pips or 6.0, 0.1)
    return float(_clamp01((base - 0.8) / 1.4) or 0.0)


def _log_reentry_decision(*, decision: str, tags: dict) -> None:
    global _LAST_REENTRY_LOG_TS
    now = time.monotonic()
    if now - _LAST_REENTRY_LOG_TS < _REENTRY_LOG_INTERVAL_SEC:
        return
    _LAST_REENTRY_LOG_TS = now
    payload = dict(tags)
    payload["decision"] = decision
    log_metric("m1scalp_reentry_decision", 1.0, tags=payload)

BB_STYLE = "scalp"
LOG = logging.getLogger(__name__)

ALLOWED_TAGS = {"M1Scalper", "m1scalper", "m1_scalper"}
_REASON_RSI_FADE = "m1_rsi_fade"
_REASON_VWAP_CUT = "m1_vwap_cut"
_REASON_STRUCTURE_BREAK = "m1_structure_break"
_REASON_ATR_SPIKE = "m1_atr_spike"


def _float_env(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _latest_mid() -> Optional[float]:
    tick = tick_window.recent_ticks(seconds=2.0, limit=1)
    if tick:
        try:
            return float(tick[-1]["mid"])
        except Exception:
            pass
    try:
        return float(all_factors().get("M1", {}).get("close"))
    except Exception:
        return None


def _filter_trades(trades: Sequence[dict], tags: Set[str]) -> list[dict]:
    if not tags:
        return list(trades)
    filtered: list[dict] = []
    for tr in trades:
        thesis = tr.get("entry_thesis") or {}
        tag = (
            thesis.get("strategy_tag")
            or thesis.get("strategy_tag_raw")
            or thesis.get("strategy")
            or thesis.get("tag")
            or tr.get("strategy_tag")
            or tr.get("strategy")
        )
        if not tag:
            continue
        tag_str = str(tag)
        base_tag = tag_str.split("-", 1)[0]
        if tag_str in tags or base_tag in tags:
            filtered.append(tr)
    return filtered


@dataclass
class _TradeState:
    peak: float
    lock_floor: Optional[float] = None
    hard_stop: Optional[float] = None
    tp_hint: Optional[float] = None

    def update(self, pnl: float, lock_buffer: float) -> None:
        if pnl > self.peak:
            self.peak = pnl
        if pnl > 0:
            floor = max(0.0, pnl - lock_buffer)
            self.lock_floor = floor if self.lock_floor is None else max(self.lock_floor, floor)


async def _run_exit_loop(
    *,
    pocket: str,
    tags: Set[str],
    profit_take: float,
    trail_start: float,
    trail_backoff: float,
    lock_buffer: float,
    min_hold_sec: float,
    max_hold_sec: float,
    max_adverse_pips: float,
    trail_from_tp_ratio: float,
    lock_from_tp_ratio: float,
    loop_interval: float,
) -> None:
    pos_manager = PositionManager()
    states: Dict[str, _TradeState] = {}
    rollout_skip_log_ts: Dict[str, float] = {}
    rsi_fade_long = _float_env("M1SCALP_EXIT_RSI_FADE_LONG", 44.0)
    rsi_fade_short = _float_env("M1SCALP_EXIT_RSI_FADE_SHORT", 56.0)
    vwap_gap_pips = _float_env("M1SCALP_EXIT_VWAP_GAP_PIPS", 0.8)
    structure_adx = _float_env("M1SCALP_EXIT_STRUCTURE_ADX", 20.0)
    structure_gap_pips = _float_env("M1SCALP_EXIT_STRUCTURE_GAP_PIPS", 1.8)
    atr_spike_pips = _float_env("M1SCALP_EXIT_ATR_SPIKE_PIPS", 5.0)
    rollout_start_ts = load_rollout_start_ts("M1SCALP_EXIT_POLICY_START_TS")
    allow_negative_exit = os.getenv("M1SCALP_EXIT_ALLOW_NEGATIVE", "1").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
    }

    def _context() -> tuple[
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[tuple[float, float]],
        Optional[float],
    ]:
        fac_m1 = all_factors().get("M1") or {}
        rsi = _safe_float(fac_m1.get("rsi"))
        adx = _safe_float(fac_m1.get("adx"))
        atr_pips = _safe_float(fac_m1.get("atr_pips"))
        if atr_pips is None:
            atr_val = _safe_float(fac_m1.get("atr"))
            if atr_val is not None:
                atr_pips = atr_val * 100.0
        vwap_gap = _safe_float(fac_m1.get("vwap_gap"))
        ma10 = _safe_float(fac_m1.get("ma10"))
        ma20 = _safe_float(fac_m1.get("ma20"))
        bbw = _safe_float(fac_m1.get("bbw"))
        return rsi, adx, atr_pips, vwap_gap, (ma10, ma20) if ma10 is not None and ma20 is not None else None, bbw

    def _maybe_log_rollout_skip(
        *,
        trade_id: str,
        side: str,
        pnl: float,
        hold_sec: float,
    ) -> None:
        now_mono = time.monotonic()
        last = float(rollout_skip_log_ts.get(trade_id) or 0.0)
        if now_mono - last < 60.0:
            return
        rollout_skip_log_ts[trade_id] = now_mono
        log_metric(
            "m1scalp_rollout_skip",
            float(pnl),
            tags={"side": side, "reason": "negative_exit"},
        )
        LOG.info(
            "[EXIT-%s] rollout skip trade=%s pnl=%.2fp hold=%.0fs",
            pocket,
            trade_id,
            pnl,
            hold_sec,
        )


    async def _close(
        trade_id: str,
        units: int,
        reason: str,
        client_order_id: Optional[str],
        allow_negative: bool = False,
    ) -> bool:
        if _BB_EXIT_ENABLED:
            allow_neg = bool(locals().get("allow_negative"))
            pnl_val = locals().get("pnl")
            if not _bb_exit_should_bypass(reason, pnl_val, allow_neg):
                factors = all_factors()
                fac_m1 = factors.get("M1") or {}
                fac_h4 = factors.get("H4") or {}
                range_active = False
                try:
                    range_active = bool(detect_range_mode(fac_m1, fac_h4).active)
                except Exception:
                    range_active = False
                fac = factors.get(_BB_EXIT_TF) or fac_m1
                price = _bb_exit_price(fac)
                side = "long" if units > 0 else "short"
                if not _bb_exit_allowed(BB_STYLE, side, price, fac, range_active=range_active):
                    LOG.info("[exit-bb] trade=%s reason=%s price=%.3f", trade_id, reason, price or 0.0)
                    return False
        ok = await close_trade(
            trade_id,
            units,
            client_order_id=client_order_id,
            allow_negative=allow_negative,
            exit_reason=reason,
            env_prefix=_BB_ENV_PREFIX,
        )
        if ok:
            LOG.info("[EXIT-%s] trade=%s units=%s reason=%s", pocket, trade_id, units, reason)
        else:
            LOG.error("[EXIT-%s] close failed trade=%s units=%s reason=%s", pocket, trade_id, units, reason)
        return ok

    async def _review_trade(trade: dict, now: datetime) -> None:
        trade_id = str(trade.get("trade_id"))
        if not trade_id:
            return
        units = int(trade.get("units", 0) or 0)
        if units == 0:
            return

        price_entry = float(trade.get("price") or 0.0)
        if price_entry <= 0.0:
            return

        side = "long" if units > 0 else "short"
        current = _latest_mid()
        if current is None:
            return
        pnl = mark_pnl_pips(price_entry, units, mid=current)
        allow_negative = allow_negative_exit or pnl <= 0

        opened_at = _parse_time(trade.get("open_time"))
        hold_sec = (now - opened_at).total_seconds() if opened_at else 0.0
        policy_active = trade_passes_rollout(
            opened_at,
            rollout_start_ts,
            unknown_is_new=False,
        )

        thesis = trade.get("entry_thesis") or {}
        if not isinstance(thesis, dict):
            thesis = {}

        state = states.get(trade_id)
        if state is None:
            hard_stop = thesis.get("hard_stop_pips")
            tp_hint = thesis.get("tp_pips")
            try:
                hard_stop_val = float(hard_stop) if hard_stop is not None else None
            except Exception:
                hard_stop_val = None
            try:
                tp_hint_val = float(tp_hint) if tp_hint is not None else None
            except Exception:
                tp_hint_val = None
            state = _TradeState(peak=pnl, hard_stop=hard_stop_val, tp_hint=tp_hint_val)
            states[trade_id] = state

        strategy_tag = (
            thesis.get("strategy_tag")
            or thesis.get("strategy_tag_raw")
            or thesis.get("strategy")
            or thesis.get("tag")
            or trade.get("strategy_tag")
            or trade.get("strategy")
            or "m1scalper"
        )
        scale, _ = momentum_scale(
            pocket=pocket,
            strategy_tag=strategy_tag,
            entry_thesis=thesis,
            env_prefix=_BB_ENV_PREFIX,
        )

        min_hold = scale_value(min_hold_sec, scale=scale, floor=min_hold_sec)
        max_hold = scale_value(max_hold_sec, scale=scale, floor=max_hold_sec)
        max_adverse = scale_value(max_adverse_pips, scale=scale, floor=max_adverse_pips)

        tp = scale_value(profit_take, scale=scale, floor=profit_take)
        ts = scale_value(trail_start, scale=scale, floor=trail_start)
        tb = scale_value(trail_backoff, scale=scale, floor=trail_backoff)
        lb = scale_value(lock_buffer, scale=scale, floor=lock_buffer)
        forecast_adj = build_exit_forecast_adjustment(
            side=side,
            entry_thesis=thesis,
            env_prefix=_BB_ENV_PREFIX,
        )
        tp, ts, tb, lb = apply_exit_forecast_to_targets(
            profit_take=tp,
            trail_start=ts,
            trail_backoff=tb,
            lock_buffer=lb,
            adjustment=forecast_adj,
            profit_take_floor=0.5,
            trail_start_floor=0.5,
            trail_backoff_floor=0.05,
            lock_buffer_floor=0.05,
        )
        if forecast_adj.enabled and max_adverse > 0.0:
            max_adverse = max(0.5, max_adverse * forecast_adj.loss_cut_mult)
        if forecast_adj.enabled and max_hold > 0.0:
            max_hold = max(min_hold, max_hold * forecast_adj.max_hold_mult)

        if state.tp_hint:
            tp = max(tp, max(1.0, state.tp_hint * 0.9))
            ts = max(ts, max(1.0, tp * trail_from_tp_ratio))
            lb = max(lb, tp * lock_from_tp_ratio)

        state.update(pnl, lb)

        client_ext = trade.get("clientExtensions")
        client_id = trade.get("client_order_id")
        if not client_id and isinstance(client_ext, dict):
            client_id = client_ext.get("id")

        # 最低保有時間まではクローズ禁止（スプレッド負け防止）
        if hold_sec < min_hold:
            return
        if await maybe_close_pro_stop(trade, now=now):
            return

        candle_reason = _exit_candle_reversal("long" if units > 0 else "short")
        if candle_reason and pnl >= 0:
            candle_client_id = trade.get("client_order_id")
            if not candle_client_id:
                client_ext = trade.get("clientExtensions")
                if isinstance(client_ext, dict):
                    candle_client_id = client_ext.get("id")
            if candle_client_id:
                await _close(trade_id, -units, candle_reason, candle_client_id, allow_negative=allow_negative)
                states.pop(trade_id, None)
                return
        if not client_id:
            LOG.warning("[EXIT-%s] missing client_id trade=%s skip close", pocket, trade_id)
            return

        lock_trigger = max(0.6, tp * 0.35)
        if (
            state.lock_floor is not None
            and state.peak >= lock_trigger
            and pnl > 0
            and pnl <= state.lock_floor
        ):
            await _close(trade_id, -units, "lock_floor", client_id, allow_negative=allow_negative)
            states.pop(trade_id, None)
            return

        if pnl < 0 and not policy_active:
            _maybe_log_rollout_skip(
                trade_id=trade_id,
                side=side,
                pnl=float(pnl),
                hold_sec=hold_sec,
            )
            return

        if pnl < 0:
            rsi, adx, atr_pips, vwap_gap, ma_pair, bbw = _context()
            skip_soft = False
            if _REENTRY_ENABLED:
                range_active = False
                try:
                    factors = all_factors()
                    fac_m1 = factors.get("M1") or {}
                    fac_h4 = factors.get("H4") or {}
                    range_active = bool(detect_range_mode(fac_m1, fac_h4).active)
                except Exception:
                    range_active = False

                adverse_pips = abs(float(pnl))
                edge = _reentry_edge(adverse_pips, atr_pips)
                revert_score, trend_score = _reentry_scores(
                    side=side,
                    rsi=rsi,
                    adx=adx,
                    atr_pips=atr_pips,
                    bbw=bbw,
                    vwap_gap=vwap_gap,
                    ma_pair=ma_pair,
                    range_active=range_active,
                )

                min_adverse = _REENTRY_MIN_ADVERSE_PIPS
                if atr_pips is not None:
                    min_adverse = max(min_adverse, atr_pips * _REENTRY_MIN_ADVERSE_ATR)

                decision = None
                if adverse_pips >= min_adverse and revert_score is not None and trend_score is not None:
                    if revert_score >= _REENTRY_REVERT_MIN and trend_score <= _REENTRY_TREND_MAX:
                        decision = "hold"
                    elif trend_score >= _REENTRY_TREND_MIN and edge >= _REENTRY_EDGE_MIN:
                        decision = "exit_reentry"

                if decision:
                    _log_reentry_decision(
                        decision=decision,
                        tags={
                            "side": side,
                            "revert": f"{revert_score:.2f}" if revert_score is not None else "na",
                            "trend": f"{trend_score:.2f}" if trend_score is not None else "na",
                            "edge": f"{edge:.2f}",
                            "pnl": f"{pnl:.2f}",
                        },
                    )
                    if decision == "hold":
                        skip_soft = True
                    elif decision == "exit_reentry":
                        if not _REENTRY_SHADOW:
                            ok = await _close(
                                trade_id,
                                -units,
                                "reentry_reset",
                                client_id,
                                allow_negative=True,
                            )
                            if ok:
                                states.pop(trade_id, None)
                                return

            if not skip_soft:
                soft_failed = False
                if rsi is not None:
                    if side == "long" and rsi <= rsi_fade_long:
                        ok = await _close(
                            trade_id,
                            -units,
                            _REASON_RSI_FADE,
                            client_id,
                            allow_negative=allow_negative,
                        )
                        if ok:
                            states.pop(trade_id, None)
                            return
                        soft_failed = True
                    if side == "short" and rsi >= rsi_fade_short:
                        ok = await _close(
                            trade_id,
                            -units,
                            _REASON_RSI_FADE,
                            client_id,
                            allow_negative=allow_negative,
                        )
                        if ok:
                            states.pop(trade_id, None)
                            return
                        soft_failed = True
                if not soft_failed and vwap_gap is not None and abs(vwap_gap) <= vwap_gap_pips:
                    ok = await _close(
                        trade_id,
                        -units,
                        _REASON_VWAP_CUT,
                        client_id,
                        allow_negative=allow_negative,
                    )
                    if ok:
                        states.pop(trade_id, None)
                        return
                    soft_failed = True
                if not soft_failed and adx is not None and ma_pair is not None:
                    ma10, ma20 = ma_pair
                    gap = abs(ma10 - ma20) / 0.01
                    cross_bad = (side == "long" and ma10 <= ma20) or (side == "short" and ma10 >= ma20)
                    if adx <= structure_adx and (cross_bad or gap <= structure_gap_pips):
                        ok = await _close(
                            trade_id,
                            -units,
                            _REASON_STRUCTURE_BREAK,
                            client_id,
                            allow_negative=allow_negative,
                        )
                        if ok:
                            states.pop(trade_id, None)
                            return
                        soft_failed = True
                if not soft_failed and atr_pips is not None and atr_pips >= atr_spike_pips:
                    ok = await _close(
                        trade_id,
                        -units,
                        _REASON_ATR_SPIKE,
                        client_id,
                        allow_negative=allow_negative,
                    )
                    if ok:
                        states.pop(trade_id, None)
                        return

        if policy_active and max_adverse > 0 and pnl <= -max_adverse:
            log_metric("m1scalp_max_adverse", pnl, tags={"side": side})
            await _close(trade_id, -units, "max_adverse", client_id, allow_negative=allow_negative)
            states.pop(trade_id, None)
            return

        if policy_active and max_hold > 0 and hold_sec >= max_hold and pnl <= 0:
            log_metric("m1scalp_max_hold", pnl, tags={"side": side})
            await _close(trade_id, -units, "max_hold", client_id, allow_negative=allow_negative)
            states.pop(trade_id, None)
            return

        trail_trigger = max(ts, 0.0)

        # プラス圏のみでクローズする
        if (
            state.peak > 0
            and state.peak >= trail_trigger
            and pnl > 0
            and pnl <= state.peak - tb
        ):
            await _close(trade_id, -units, "trail_take", client_id, allow_negative=allow_negative)
            states.pop(trade_id, None)
            return

        if pnl >= tp:
            await _close(trade_id, -units, "take_profit", client_id, allow_negative=allow_negative)
            states.pop(trade_id, None)
            return

    LOG.info(
        "[EXIT-%s] start tags=%s policy_start_ts=%.3f",
        pocket,
        ",".join(sorted(tags)),
        float(rollout_start_ts or 0.0),
    )
    try:
        while True:
            await asyncio.sleep(loop_interval)
            positions = pos_manager.get_open_positions()
            pocket_info = positions.get(pocket) or {}
            trades = pocket_info.get("open_trades") or []
            trades = _filter_trades(trades, tags)
            active_ids = {str(tr.get("trade_id")) for tr in trades if tr.get("trade_id")}
            for tid in list(states.keys()):
                if tid not in active_ids:
                    states.pop(tid, None)
            for tid in list(rollout_skip_log_ts.keys()):
                if tid not in active_ids:
                    rollout_skip_log_ts.pop(tid, None)
            if not trades:
                continue
            now = _utc_now()
            for tr in trades:
                try:
                    await _review_trade(tr, now)
                except Exception:
                    LOG.exception("[EXIT-%s] review failed trade=%s", pocket, tr.get("trade_id"))
                    continue
    except asyncio.CancelledError:  # pragma: no cover - loop cancellation
        pass


async def m1_scalper_exit_worker() -> None:
    min_hold_sec = 10.0
    max_hold_sec = max(min_hold_sec + 1.0, _float_env("M1SCALP_EXIT_MAX_HOLD_SEC", 12 * 60))
    max_adverse_pips = max(0.0, _float_env("M1SCALP_EXIT_MAX_ADVERSE_PIPS", 6.0))
    await _run_exit_loop(
        pocket="scalp",
        tags=ALLOWED_TAGS,
        profit_take=2.2,
        trail_start=2.8,
        trail_backoff=0.9,
        lock_buffer=0.5,
        min_hold_sec=min_hold_sec,
        max_hold_sec=max_hold_sec,
        max_adverse_pips=max_adverse_pips,
        trail_from_tp_ratio=0.82,
        lock_from_tp_ratio=0.45,
        loop_interval=0.8,
    )


_CANDLE_PIP = 0.01
_CANDLE_EXIT_MIN_CONF = 0.35
_CANDLE_EXIT_SCORE = -0.5
_CANDLE_WORKER_NAME = (__file__.replace("\\", "/").split("/")[-2] if "/" in __file__ else "").lower()


def _candle_tf_for_worker() -> str:
    name = _CANDLE_WORKER_NAME
    if "macro" in name or "trend_h1" in name or "manual" in name:
        return "H1"
    if "scalp" in name or "s5" in name or "fast" in name:
        return "M1"
    return "M5"


def _extract_candles(raw):
    candles = []
    for candle in raw or []:
        try:
            o = float(candle.get("open", candle.get("o")))
            h = float(candle.get("high", candle.get("h")))
            l = float(candle.get("low", candle.get("l")))
            c = float(candle.get("close", candle.get("c")))
        except Exception:
            continue
        if h <= 0 or l <= 0:
            continue
        candles.append((o, h, l, c))
    return candles


def _detect_candlestick_pattern(candles):
    if len(candles) < 2:
        return None
    o0, h0, l0, c0 = candles[-2]
    o1, h1, l1, c1 = candles[-1]
    body0 = abs(c0 - o0)
    body1 = abs(c1 - o1)
    range1 = max(h1 - l1, _CANDLE_PIP * 0.1)
    upper_wick = h1 - max(o1, c1)
    lower_wick = min(o1, c1) - l1

    if body1 <= range1 * 0.1:
        return {
            "type": "doji",
            "confidence": round(min(1.0, (range1 - body1) / range1), 3),
            "bias": None,
        }

    if (
        c1 > o1
        and c0 < o0
        and c1 >= max(o0, c0)
        and o1 <= min(o0, c0)
        and body1 > body0
    ):
        return {
            "type": "bullish_engulfing",
            "confidence": round(min(1.0, body1 / range1 + 0.3), 3),
            "bias": "up",
        }
    if (
        c1 < o1
        and c0 > o0
        and o1 >= min(o0, c0)
        and c1 <= max(o0, c0)
        and body1 > body0
    ):
        return {
            "type": "bearish_engulfing",
            "confidence": round(min(1.0, body1 / range1 + 0.3), 3),
            "bias": "down",
        }
    if lower_wick > body1 * 2.5 and upper_wick <= body1 * 0.6:
        return {
            "type": "hammer" if c1 >= o1 else "inverted_hammer",
            "confidence": round(min(1.0, lower_wick / range1 + 0.25), 3),
            "bias": "up",
        }
    if upper_wick > body1 * 2.5 and lower_wick <= body1 * 0.6:
        return {
            "type": "shooting_star" if c1 <= o1 else "hanging_man",
            "confidence": round(min(1.0, upper_wick / range1 + 0.25), 3),
            "bias": "down",
        }
    return None


def _score_candle(*, candles, side, min_conf):
    pattern = _detect_candlestick_pattern(_extract_candles(candles))
    if not pattern:
        return None, {}
    bias = pattern.get("bias")
    conf = float(pattern.get("confidence") or 0.0)
    if conf < min_conf:
        return None, {"type": pattern.get("type"), "confidence": round(conf, 3)}
    if bias is None:
        return 0.0, {"type": pattern.get("type"), "confidence": round(conf, 3), "bias": None}
    match = (side == "long" and bias == "up") or (side == "short" and bias == "down")
    score = conf if match else -conf * 0.7
    score = max(-1.0, min(1.0, score))
    return score, {"type": pattern.get("type"), "confidence": round(conf, 3), "bias": bias}


def _exit_candle_reversal(side):
    tf = _candle_tf_for_worker()
    candles = (all_factors().get(tf) or {}).get("candles") or []
    if not candles:
        return None
    score, detail = _score_candle(candles=candles, side=side, min_conf=_CANDLE_EXIT_MIN_CONF)
    if score is None:
        return None
    if score <= _CANDLE_EXIT_SCORE:
        detail_type = detail.get("type") if isinstance(detail, dict) else None
        return f"candle_{detail_type}" if detail_type else "candle_reversal"
    return None


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    asyncio.run(m1_scalper_exit_worker())
