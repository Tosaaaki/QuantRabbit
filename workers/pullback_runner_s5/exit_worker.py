"""Exit loop for pullback_runner_s5 worker (scalp pocket) with technical filters."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

from analysis.range_guard import detect_range_mode
from workers.common.exit_utils import close_trade, mark_pnl_pips
from execution.position_manager import PositionManager
from indicators.factor_cache import all_factors
from market_data import tick_window
from utils.metrics_logger import log_metric
from workers.common.exit_scaling import TPScaleConfig, apply_tp_virtual_floor
from workers.common.pullback_touch import PullbackTouchResult, count_pullback_touches


_BB_EXIT_ENABLED = os.getenv("BB_EXIT_ENABLED", "1").strip().lower() not in {"", "0", "false", "no"}
_BB_EXIT_REVERT_PIPS = float(os.getenv("BB_EXIT_REVERT_PIPS", "2.0"))
_BB_EXIT_REVERT_RATIO = float(os.getenv("BB_EXIT_REVERT_RATIO", "0.20"))
_BB_EXIT_TREND_EXT_PIPS = float(os.getenv("BB_EXIT_TREND_EXT_PIPS", "3.0"))
_BB_EXIT_TREND_EXT_RATIO = float(os.getenv("BB_EXIT_TREND_EXT_RATIO", "0.35"))
_BB_EXIT_SCALP_REVERT_PIPS = float(os.getenv("BB_EXIT_SCALP_REVERT_PIPS", "1.6"))
_BB_EXIT_SCALP_REVERT_RATIO = float(os.getenv("BB_EXIT_SCALP_REVERT_RATIO", "0.18"))
_BB_EXIT_SCALP_EXT_PIPS = float(os.getenv("BB_EXIT_SCALP_EXT_PIPS", "2.0"))
_BB_EXIT_SCALP_EXT_RATIO = float(os.getenv("BB_EXIT_SCALP_EXT_RATIO", "0.28"))
_BB_EXIT_MID_BUFFER_PIPS = float(os.getenv("BB_EXIT_MID_BUFFER_PIPS", "0.4"))
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

BB_STYLE = "trend"
LOG = logging.getLogger(__name__)

ALLOWED_TAGS = {"pullback_runner_s5"}
POCKET = "scalp"


def _float_env(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _bool_env(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


STRICT_TAG = _bool_env("PULLBACK_RUNNER_S5_EXIT_STRICT_TAG", True)


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


def _client_id(trade: dict) -> Optional[str]:
    client_ext = trade.get("clientExtensions")
    client_id = trade.get("client_order_id")
    if not client_id and isinstance(client_ext, dict):
        client_id = client_ext.get("id")
    return client_id


@dataclass
class _TradeState:
    peak: float
    lock_floor: Optional[float] = None
    hard_stop: Optional[float] = None
    tp_hint: Optional[float] = None
    last_touch_count: Optional[int] = None
    hard_stop_breach_count: int = 0


@dataclass
class _Context:
    mid: Optional[float]
    rsi: Optional[float]
    adx: Optional[float]
    bbw: Optional[float]
    atr_pips: Optional[float]
    vwap_gap_pips: Optional[float]
    range_active: bool


class PullbackRunnerExitWorker:
    """PnL + RSI/ATR/VWAP/レンジ判定を組み合わせた pullback_runner_s5 EXIT."""

    def __init__(self) -> None:
        self.tp_scale = TPScaleConfig()
        self.loop_interval = max(0.3, _float_env("PULLBACK_RUNNER_S5_EXIT_LOOP_INTERVAL_SEC", 1.0))
        self._pos_manager = PositionManager()
        self._states: Dict[str, _TradeState] = {}

        self.profit_take = max(1.0, _float_env("PULLBACK_RUNNER_S5_EXIT_PROFIT_PIPS", 2.4))
        self.trail_start = max(1.2, _float_env("PULLBACK_RUNNER_S5_EXIT_TRAIL_START_PIPS", 3.2))
        self.trail_backoff = max(0.3, _float_env("PULLBACK_RUNNER_S5_EXIT_TRAIL_BACKOFF_PIPS", 1.1))
        self.stop_loss = max(0.8, _float_env("PULLBACK_RUNNER_S5_EXIT_STOP_LOSS_PIPS", 1.7))
        self.max_hold_sec = max(60.0, _float_env("PULLBACK_RUNNER_S5_EXIT_MAX_HOLD_SEC", 25 * 60))
        self.lock_trigger = max(0.6, _float_env("PULLBACK_RUNNER_S5_EXIT_LOCK_TRIGGER_PIPS", 1.1))
        self.lock_buffer = max(0.1, _float_env("PULLBACK_RUNNER_S5_EXIT_LOCK_BUFFER_PIPS", 0.6))

        self.range_profit_take = max(0.8, _float_env("PULLBACK_RUNNER_S5_EXIT_RANGE_PROFIT_PIPS", 1.8))
        self.range_trail_start = max(1.0, _float_env("PULLBACK_RUNNER_S5_EXIT_RANGE_TRAIL_START_PIPS", 2.3))
        self.range_trail_backoff = max(0.2, _float_env("PULLBACK_RUNNER_S5_EXIT_RANGE_TRAIL_BACKOFF_PIPS", 0.7))
        self.range_stop_loss = max(0.6, _float_env("PULLBACK_RUNNER_S5_EXIT_RANGE_STOP_LOSS_PIPS", 1.1))
        self.range_max_hold_sec = max(45.0, _float_env("PULLBACK_RUNNER_S5_EXIT_RANGE_MAX_HOLD_SEC", 17 * 60))
        self.range_lock_trigger = max(0.35, _float_env("PULLBACK_RUNNER_S5_EXIT_RANGE_LOCK_TRIGGER_PIPS", 0.8))
        self.range_lock_buffer = max(0.1, _float_env("PULLBACK_RUNNER_S5_EXIT_RANGE_LOCK_BUFFER_PIPS", 0.35))

        self.hard_stop_emergency_mult = max(
            1.0, _float_env("PULLBACK_RUNNER_S5_EXIT_HARD_STOP_EMERGENCY_MULT", 1.6)
        )
        self.hard_stop_score_count = max(
            1, int(_float_env("PULLBACK_RUNNER_S5_EXIT_HARD_STOP_SCORE_COUNT", 2))
        )
        self.hard_stop_confirm_ticks = max(
            0, int(_float_env("PULLBACK_RUNNER_S5_EXIT_HARD_STOP_CONFIRM_TICKS", 2))
        )
        self.structure_score_neg = max(
            1, int(_float_env("PULLBACK_RUNNER_S5_EXIT_STRUCTURE_SCORE_NEG", 2))
        )
        self.structure_score_pos = max(
            self.structure_score_neg,
            int(_float_env("PULLBACK_RUNNER_S5_EXIT_STRUCTURE_SCORE_POS", 3)),
        )

        self.range_adx = max(5.0, _float_env("PULLBACK_RUNNER_S5_EXIT_RANGE_ADX", 22.0))
        self.range_bbw = max(0.02, _float_env("PULLBACK_RUNNER_S5_EXIT_RANGE_BBW", 0.20))
        self.range_atr = max(0.4, _float_env("PULLBACK_RUNNER_S5_EXIT_RANGE_ATR", 6.0))

        self.rsi_fade_long = _float_env("PULLBACK_RUNNER_S5_EXIT_RSI_FADE_LONG", 42.0)
        self.rsi_fade_short = _float_env("PULLBACK_RUNNER_S5_EXIT_RSI_FADE_SHORT", 58.0)
        self.rsi_take_long = _float_env("PULLBACK_RUNNER_S5_EXIT_RSI_TAKE_LONG", 70.0)
        self.rsi_take_short = _float_env("PULLBACK_RUNNER_S5_EXIT_RSI_TAKE_SHORT", 30.0)
        self.negative_hold_sec = max(20.0, _float_env("PULLBACK_RUNNER_S5_EXIT_NEG_HOLD_SEC", 150.0))
        self.allow_negative_exit = _bool_env("PULLBACK_RUNNER_S5_EXIT_ALLOW_NEGATIVE", True)

        self.vwap_grab_gap = max(0.15, _float_env("PULLBACK_RUNNER_S5_EXIT_VWAP_GAP_PIPS", 0.9))

        self.atr_hot = max(0.5, _float_env("PULLBACK_RUNNER_S5_EXIT_ATR_HOT_PIPS", 5.0))
        self.atr_cold = max(0.3, _float_env("PULLBACK_RUNNER_S5_EXIT_ATR_COLD_PIPS", 1.2))

        self.touch_enabled = _bool_env("PULLBACK_RUNNER_S5_EXIT_TOUCH_ENABLED", True)
        self.touch_window_sec = max(30.0, _float_env("PULLBACK_RUNNER_S5_EXIT_TOUCH_WINDOW_SEC", 180.0))
        self.touch_min_ticks = max(20, int(_float_env("PULLBACK_RUNNER_S5_EXIT_TOUCH_MIN_TICKS", 40)))
        self.touch_pullback_atr_mult = max(
            0.1, _float_env("PULLBACK_RUNNER_S5_EXIT_TOUCH_PULLBACK_ATR_MULT", 0.7)
        )
        self.touch_pullback_min_pips = max(
            0.1, _float_env("PULLBACK_RUNNER_S5_EXIT_TOUCH_PULLBACK_MIN_PIPS", 0.6)
        )
        self.touch_pullback_max_pips = max(
            self.touch_pullback_min_pips,
            _float_env("PULLBACK_RUNNER_S5_EXIT_TOUCH_PULLBACK_MAX_PIPS", 2.6),
        )
        self.touch_trend_atr_mult = max(
            0.1, _float_env("PULLBACK_RUNNER_S5_EXIT_TOUCH_TREND_ATR_MULT", 1.5)
        )
        self.touch_trend_min_pips = max(
            0.2, _float_env("PULLBACK_RUNNER_S5_EXIT_TOUCH_TREND_MIN_PIPS", 1.2)

        )
        self.touch_trend_max_pips = max(
            self.touch_trend_min_pips,
            _float_env("PULLBACK_RUNNER_S5_EXIT_TOUCH_TREND_MAX_PIPS", 4.8),
        )
        self.touch_reset_ratio = max(
            0.1, min(0.9, _float_env("PULLBACK_RUNNER_S5_EXIT_TOUCH_RESET_RATIO", 0.45)))
        self.touch_tighten_count = max(
            1, int(_float_env("PULLBACK_RUNNER_S5_EXIT_TOUCH_TIGHTEN_COUNT", 3))
        )
        self.touch_tighten_ratio = max(
            0.5, min(1.0, _float_env("PULLBACK_RUNNER_S5_EXIT_TOUCH_TIGHTEN_RATIO", 0.85))
        )
        self.touch_force_count = max(
            self.touch_tighten_count + 1,
            int(_float_env("PULLBACK_RUNNER_S5_EXIT_TOUCH_FORCE_COUNT", 5)),
        )
        self.touch_force_min_pips = max(
            0.1, _float_env("PULLBACK_RUNNER_S5_EXIT_TOUCH_FORCE_MIN_PIPS", 0.7)
        )

    def _filter_trades(self, trades: list[dict]) -> list[dict]:
        if not ALLOWED_TAGS:
            return trades
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
            if STRICT_TAG:
                if tag_str in ALLOWED_TAGS:
                    filtered.append(tr)
            else:
                if tag_str in ALLOWED_TAGS or base_tag in ALLOWED_TAGS:
                    filtered.append(tr)
        return filtered

    def _context(self) -> _Context:
        factors = all_factors()
        fac_m1 = factors.get("M1") or {}
        fac_h4 = factors.get("H4") or {}

        def _safe_float(val: object, default: Optional[float] = None) -> Optional[float]:
            if val is None:
                return default
            try:
                return float(val)
            except (TypeError, ValueError):
                return default

        atr_pips = _safe_float(fac_m1.get("atr_pips"))
        if atr_pips is None:
            atr_pips = _safe_float(fac_m1.get("atr"), 0.0)
            if atr_pips is not None:
                atr_pips *= 100.0

        range_ctx = detect_range_mode(
            fac_m1,
            fac_h4,
            adx_threshold=self.range_adx,
            bbw_threshold=self.range_bbw,
            atr_threshold=self.range_atr,
        )

        return _Context(
            mid=_latest_mid(),
            rsi=_safe_float(fac_m1.get("rsi"), None),
            adx=_safe_float(fac_m1.get("adx"), None),
            bbw=_safe_float(fac_m1.get("bbw"), None),
            atr_pips=atr_pips,
            vwap_gap_pips=_safe_float(fac_m1.get("vwap_gap"), None),
            range_active=bool(range_ctx.active),
        )

    def _touch_stats(self, side: str, ctx: _Context) -> Optional[PullbackTouchResult]:
        if not self.touch_enabled:
            return None
        ticks = tick_window.recent_ticks(seconds=self.touch_window_sec, limit=2400)
        if len(ticks) < self.touch_min_ticks:
            return None
        prices: list[float] = []
        times: list[float] = []
        for row in ticks:
            mid = row.get("mid")
            epoch = row.get("epoch")
            if mid is None or epoch is None:
                continue
            try:
                prices.append(float(mid))
                times.append(float(epoch))
            except (TypeError, ValueError):
                continue
        if len(prices) < self.touch_min_ticks:
            return None
        atr_ref = ctx.atr_pips or 0.0
        if atr_ref <= 0.0:
            atr_ref = self.touch_pullback_min_pips
        pullback_pips = max(
            self.touch_pullback_min_pips,
            min(self.touch_pullback_max_pips, atr_ref * self.touch_pullback_atr_mult),
        )
        trend_pips = max(
            self.touch_trend_min_pips,
            min(self.touch_trend_max_pips, atr_ref * self.touch_trend_atr_mult),
        )
        reset_pips = max(self.touch_pullback_min_pips * 0.5, pullback_pips * self.touch_reset_ratio)
        return count_pullback_touches(
            prices,
            side,
            pullback_pips=pullback_pips,
            trend_confirm_pips=trend_pips,
            reset_pips=reset_pips,
            pip_value=0.01,
            timestamps=times,
        )

    async def _close(
        self,
        trade_id: str,
        units: int,
        reason: str,
        pnl: float,
        side: str,
        range_mode: bool,
        client_id: str,
        allow_negative: bool = False,
        touch_count: Optional[int] = None,
    ) -> bool:
        if _BB_EXIT_ENABLED:
            allow_neg = bool(locals().get("allow_negative"))
            pnl_val = locals().get("pnl")
            if not _bb_exit_should_bypass(reason, pnl_val, allow_neg):
                fac = all_factors().get(_BB_EXIT_TF) or {}
                price = _bb_exit_price(fac)
                side = "long" if units > 0 else "short"
                if not _bb_exit_allowed(BB_STYLE, side, price, fac):
                    LOG.info("[exit-bb] trade=%s reason=%s price=%.3f", trade_id, reason, price or 0.0)
                    return False
        if pnl <= 0:
            allow_negative = True
        ok = await close_trade(
            trade_id,
            units,
            client_order_id=client_id,
            allow_negative=allow_negative,
            exit_reason=reason,
        )
        if ok:
            LOG.info(
                "[EXIT-pullback_runner_s5] trade=%s units=%s reason=%s pnl=%.2fp range=%s touch=%s",
                trade_id,
                units,
                reason,
                pnl,
                range_mode,
                "n/a" if touch_count is None else touch_count,
            )
            tags = {"reason": reason, "range": str(range_mode), "side": side}
            if touch_count is not None:
                tags["touch_count"] = touch_count
            log_metric(
                "pullback_runner_s5_exit",
                pnl,
                tags=tags,
                ts=_utc_now(),
            )
        else:
            LOG.error("[EXIT-pullback_runner_s5] close failed trade=%s units=%s reason=%s", trade_id, units, reason)
        return ok

    def _evaluate(self, trade: dict, ctx: _Context, now: datetime) -> Optional[str]:
        trade_id = str(trade.get("trade_id"))
        if not trade_id or ctx.mid is None:
            return None

        units = int(trade.get("units", 0) or 0)
        if units == 0:
            return None

        entry_price = float(trade.get("price") or 0.0)
        if entry_price <= 0.0:
            return None

        side = "long" if units > 0 else "short"
        pnl = mark_pnl_pips(entry_price, units, mid=ctx.mid)
        opened_at = _parse_time(trade.get("open_time"))
        hold_sec = (now - opened_at).total_seconds() if opened_at else 0.0
        candle_reason = _exit_candle_reversal("long" if units > 0 else "short")
        if candle_reason and pnl >= 0:
            return candle_reason

        state = self._states.get(trade_id)
        if state is None:
            thesis = trade.get("entry_thesis") or {}
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
            self._states[trade_id] = state
        else:
            state.peak = max(state.peak, pnl)

        range_mode = ctx.range_active or (
            ctx.adx is not None and ctx.adx <= self.range_adx and ctx.bbw is not None and ctx.bbw <= self.range_bbw
        )

        profit_take = self.range_profit_take if range_mode else self.profit_take
        trail_start = self.range_trail_start if range_mode else self.trail_start
        trail_backoff = self.range_trail_backoff if range_mode else self.trail_backoff
        stop_loss = self.range_stop_loss if range_mode else self.stop_loss
        lock_trigger = self.range_lock_trigger if range_mode else self.lock_trigger
        lock_buffer = self.range_lock_buffer if range_mode else self.lock_buffer
        max_hold = self.range_max_hold_sec if range_mode else self.max_hold_sec

        # エントリーメタに合わせてEXIT閾値をスケール
        if state.hard_stop:
            hard_stop_factor = 1.0 if range_mode else 0.7
            stop_loss = max(stop_loss, max(0.8, state.hard_stop * hard_stop_factor))
            lock_trigger = max(lock_trigger, max(0.25, state.hard_stop * 0.25))
            trail_start = max(trail_start, max(1.0, state.hard_stop * 0.6))
            max_hold = max(max_hold, self.max_hold_sec * 1.05)

        profit_take, trail_start, lock_buffer, stop_loss = apply_tp_virtual_floor(
            profit_take,
            trail_start,
            lock_buffer,
            stop_loss,
            state,
            self.tp_scale,
        )

        atr = ctx.atr_pips or 0.0
        if atr >= self.atr_hot:
            profit_take += 0.3
            trail_start += 0.3
        elif 0.0 < atr <= self.atr_cold:
            profit_take = max(1.0, profit_take * 0.9)
            stop_loss = max(0.7, stop_loss * 0.9)
        lock_buffer = max(lock_buffer, stop_loss * 0.35)

        touch_stats = self._touch_stats(side, ctx)
        state.last_touch_count = touch_stats.count if touch_stats else None
        if touch_stats and touch_stats.count >= self.touch_tighten_count:
            profit_take = max(0.7, profit_take * self.touch_tighten_ratio)
            trail_start = max(0.9, trail_start * self.touch_tighten_ratio)
            trail_backoff = max(0.25, trail_backoff * self.touch_tighten_ratio)
            lock_trigger = max(0.35, lock_trigger * self.touch_tighten_ratio)
            lock_buffer = max(0.1, lock_buffer * self.touch_tighten_ratio)

        rsi_fade = False
        if ctx.rsi is not None:
            if side == "long" and ctx.rsi <= self.rsi_fade_long:
                rsi_fade = True
            if side == "short" and ctx.rsi >= self.rsi_fade_short:
                rsi_fade = True
        vwap_tight = ctx.vwap_gap_pips is not None and abs(ctx.vwap_gap_pips) <= self.vwap_grab_gap

        structure_signal = False
        structure_score = 0
        structure_reason = None
        if ctx.adx is not None and ctx.adx < self.range_adx:
            try:
                factors = all_factors().get("M1") or {}
                ma10 = float(factors.get("ma10"))
                ma20 = float(factors.get("ma20"))
                gap = abs(ma10 - ma20) / 0.01
                cross_bad = (side == "long" and ma10 <= ma20) or (side == "short" and ma10 >= ma20)
                if cross_bad or (ctx.adx < 14.0 and gap < 2.0):
                    structure_signal = True
                    structure_score += 1
                    structure_reason = "structure_break"
            except Exception:
                pass
        if structure_signal:
            if rsi_fade:
                structure_score += 1
                if structure_reason is None:
                    structure_reason = "rsi_fade"
            if vwap_tight:
                structure_score += 1
                if structure_reason is None:
                    structure_reason = "vwap_cut"
            if touch_stats and touch_stats.count >= self.touch_tighten_count:
                structure_score += 1
                if structure_reason is None:
                    structure_reason = "touch_exhaust"

        if structure_signal:
            score_req = self.structure_score_pos if pnl > 0 else self.structure_score_neg
            if structure_score >= score_req:
                return structure_reason or "structure_break"

        if pnl <= -stop_loss:
            # Emergency stop if loss expands too far beyond threshold
            if pnl <= -(stop_loss * self.hard_stop_emergency_mult):
                return "hard_stop"
            neg_score = 0
            if structure_signal:
                neg_score += 1
            if rsi_fade:
                neg_score += 1
            if vwap_tight:
                neg_score += 1
            if touch_stats and touch_stats.count >= self.touch_tighten_count:
                neg_score += 1
            if neg_score >= self.hard_stop_score_count:
                return "hard_stop"
            state.hard_stop_breach_count += 1
            if self.hard_stop_confirm_ticks > 0 and state.hard_stop_breach_count < self.hard_stop_confirm_ticks:
                return None
            return "hard_stop"

        if pnl < 0 and hold_sec >= self.negative_hold_sec:
            if self.allow_negative_exit:
                return "time_cut"

        if pnl < 0:
            # Reset hard-stop trigger if price recovers meaningfully
            if state.hard_stop_breach_count and pnl > -(stop_loss * 0.6):
                state.hard_stop_breach_count = 0
            if side == "long" and ctx.rsi is not None and ctx.rsi <= self.rsi_fade_long:
                if self.allow_negative_exit or atr >= self.atr_hot:
                    return "rsi_fade"
            if side == "short" and ctx.rsi is not None and ctx.rsi >= self.rsi_fade_short:
                if self.allow_negative_exit or atr >= self.atr_hot:
                    return "rsi_fade"

        if hold_sec >= max_hold and pnl <= profit_take * 0.7:
            return "time_stop"

        if state.lock_floor is None and pnl >= lock_trigger:
            state.lock_floor = max(0.0, pnl - lock_buffer)
        if state.lock_floor is not None and pnl <= state.lock_floor:
            return "lock_release"

        if state.peak >= trail_start and pnl <= state.peak - trail_backoff:
            return "trail_take"

        if touch_stats and touch_stats.count >= self.touch_force_count and pnl >= self.touch_force_min_pips:
            return "touch_exhaust"

        if ctx.vwap_gap_pips is not None and abs(ctx.vwap_gap_pips) <= self.vwap_grab_gap:
            if pnl > 0.25:
                return "vwap_gravity"
            if pnl < 0 and self.allow_negative_exit:
                return "vwap_cut"

        if pnl >= profit_take * 0.65 and ctx.rsi is not None:
            if side == "long" and ctx.rsi >= self.rsi_take_long:
                return "rsi_take"
            if side == "short" and ctx.rsi <= self.rsi_take_short:
                return "rsi_take"

        if range_mode and pnl >= profit_take:
            return "range_take"

        if pnl >= profit_take:
            return "take_profit"

        return None

    async def run(self) -> None:
        LOG.info(
            "[EXIT-pullback_runner_s5] worker starting (interval=%.2fs tags=%s)",
            self.loop_interval,
            ",".join(sorted(ALLOWED_TAGS)),
        )
        try:
            while True:
                await asyncio.sleep(self.loop_interval)
                positions = self._pos_manager.get_open_positions()
                pocket_info = positions.get(POCKET) or {}
                trades = self._filter_trades(pocket_info.get("open_trades") or [])
                active_ids = {str(tr.get("trade_id")) for tr in trades if tr.get("trade_id")}
                for tid in list(self._states.keys()):
                    if tid not in active_ids:
                        self._states.pop(tid, None)
                if not trades:
                    continue

                ctx = self._context()
                if ctx.mid is None:
                    continue

                now = _utc_now()
                for tr in trades:
                    try:
                        reason = self._evaluate(tr, ctx, now)
                    except Exception:
                        LOG.exception("[EXIT-pullback_runner_s5] evaluate failed trade=%s", tr.get("trade_id"))
                        continue
                    if not reason:
                        continue
                    trade_id = str(tr.get("trade_id"))
                    units = int(tr.get("units", 0) or 0)
                    client_id = _client_id(tr)
                    if not client_id:
                        LOG.warning("[EXIT-pullback_runner_s5] missing client_id trade=%s skip close", trade_id)
                        continue
                    state = self._states.get(trade_id)
                    touch_count = state.last_touch_count if state else None
                    side = "long" if units > 0 else "short"
                    pnl = mark_pnl_pips(float(tr.get("price") or 0.0), units, mid=ctx.mid)
                    allow_negative = pnl <= 0
                    await self._close(
                        trade_id,
                        -units,
                        reason,
                        pnl,
                        side,
                        ctx.range_active,
                        client_id,
                        allow_negative=allow_negative,
                        touch_count=touch_count,
                    )
                    self._states.pop(trade_id, None)
        except asyncio.CancelledError:
            LOG.info("[EXIT-pullback_runner_s5] worker cancelled")
            raise
        finally:
            try:
                self._pos_manager.close()
            except Exception:
                LOG.exception("[EXIT-pullback_runner_s5] failed to close PositionManager")


async def pullback_runner_s5_exit_worker() -> None:
    worker = PullbackRunnerExitWorker()
    await worker.run()


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
    asyncio.run(pullback_runner_s5_exit_worker())
