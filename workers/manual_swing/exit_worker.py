"""Exit loop for manual_swing worker (macro pocket) with technical filters."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Set

from analysis.range_guard import detect_range_mode
from workers.common.exit_utils import close_trade, mark_pnl_pips
from workers.common.pro_stop import maybe_close_pro_stop
from workers.common.reentry_decider import decide_reentry
from execution.position_manager import PositionManager
from indicators.factor_cache import all_factors
from market_data import tick_window
from utils.metrics_logger import log_metric
from workers.common.exit_scaling import TPScaleConfig, apply_tp_virtual_floor


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
_BB_EXIT_TF = "H1"
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

# manual_swing はタグが固定されないため、空集合で pocket 全体を対象にする
ALLOWED_TAGS: Set[str] = set()
POCKET = "macro"


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
    tick = tick_window.recent_ticks(seconds=3.0, limit=1)
    if tick:
        try:
            return float(tick[-1]["mid"])
        except Exception:
            pass
    try:
        return float(all_factors().get("H1", {}).get("close"))
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


@dataclass
class _Context:
    mid: Optional[float]
    rsi: Optional[float]
    adx: Optional[float]
    bbw: Optional[float]
    atr_pips: Optional[float]
    vwap_gap_pips: Optional[float]
    range_active: bool


class ManualSwingExitWorker:
    """PnL + RSI/ATR/VWAP/レンジ判定を組み合わせた manual_swing EXIT."""

    def __init__(self) -> None:
        self.tp_scale = TPScaleConfig()
        self.loop_interval = max(1.0, _float_env("MANUAL_SWING_EXIT_LOOP_INTERVAL_SEC", 2.5))
        self._pos_manager = PositionManager()
        self._states: Dict[str, _TradeState] = {}

        self.profit_take = max(2.0, _float_env("MANUAL_SWING_EXIT_PROFIT_PIPS", 5.5))
        self.trail_start = max(2.5, _float_env("MANUAL_SWING_EXIT_TRAIL_START_PIPS", 7.5))
        self.trail_backoff = max(0.6, _float_env("MANUAL_SWING_EXIT_TRAIL_BACKOFF_PIPS", 2.5))
        self.stop_loss = max(1.5, _float_env("MANUAL_SWING_EXIT_STOP_LOSS_PIPS", 3.5))
        self.max_hold_sec = max(1200.0, _float_env("MANUAL_SWING_EXIT_MAX_HOLD_SEC", 6 * 3600))
        self.lock_trigger = max(1.0, _float_env("MANUAL_SWING_EXIT_LOCK_TRIGGER_PIPS", 2.5))
        self.lock_buffer = max(0.3, _float_env("MANUAL_SWING_EXIT_LOCK_BUFFER_PIPS", 1.0))

        self.range_profit_take = max(1.8, _float_env("MANUAL_SWING_EXIT_RANGE_PROFIT_PIPS", 4.4))
        self.range_trail_start = max(2.2, _float_env("MANUAL_SWING_EXIT_RANGE_TRAIL_START_PIPS", 6.0))
        self.range_trail_backoff = max(0.5, _float_env("MANUAL_SWING_EXIT_RANGE_TRAIL_BACKOFF_PIPS", 2.0))
        self.range_stop_loss = max(1.2, _float_env("MANUAL_SWING_EXIT_RANGE_STOP_LOSS_PIPS", 2.8))
        self.range_max_hold_sec = max(1200.0, _float_env("MANUAL_SWING_EXIT_RANGE_MAX_HOLD_SEC", 5 * 3600))
        self.range_lock_trigger = max(0.9, _float_env("MANUAL_SWING_EXIT_RANGE_LOCK_TRIGGER_PIPS", 2.0))
        self.range_lock_buffer = max(0.25, _float_env("MANUAL_SWING_EXIT_RANGE_LOCK_BUFFER_PIPS", 0.8))

        self.range_adx = max(5.0, _float_env("MANUAL_SWING_EXIT_RANGE_ADX", 22.0))
        self.range_bbw = max(0.02, _float_env("MANUAL_SWING_EXIT_RANGE_BBW", 0.24))
        self.range_atr = max(0.4, _float_env("MANUAL_SWING_EXIT_RANGE_ATR", 7.0))

        self.rsi_fade_long = _float_env("MANUAL_SWING_EXIT_RSI_FADE_LONG", 44.0)
        self.rsi_fade_short = _float_env("MANUAL_SWING_EXIT_RSI_FADE_SHORT", 56.0)
        self.rsi_take_long = _float_env("MANUAL_SWING_EXIT_RSI_TAKE_LONG", 72.0)
        self.rsi_take_short = _float_env("MANUAL_SWING_EXIT_RSI_TAKE_SHORT", 28.0)
        self.negative_hold_sec = max(360.0, _float_env("MANUAL_SWING_EXIT_NEG_HOLD_SEC", 2100.0))
        self.allow_negative_exit = _bool_env("MANUAL_SWING_EXIT_ALLOW_NEGATIVE", True)

        self.vwap_grab_gap = max(0.1, _float_env("MANUAL_SWING_EXIT_VWAP_GAP_PIPS", 1.2))

        self.atr_hot = max(1.0, _float_env("MANUAL_SWING_EXIT_ATR_HOT_PIPS", 10.0))
        self.atr_cold = max(0.5, _float_env("MANUAL_SWING_EXIT_ATR_COLD_PIPS", 3.0))

    def _filter_trades(self, trades: list[dict]) -> list[dict]:
        if not ALLOWED_TAGS:
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

            if tag_str in ALLOWED_TAGS or base_tag in ALLOWED_TAGS:
                filtered.append(tr)
        return filtered

    def _context(self) -> _Context:
        factors = all_factors()
        fac_h1 = factors.get("H1") or {}
        fac_h4 = factors.get("H4") or {}

        def _safe_float(val: object, default: Optional[float] = None) -> Optional[float]:
            if val is None:
                return default
            try:
                return float(val)
            except (TypeError, ValueError):
                return default

        atr_pips = _safe_float(fac_h1.get("atr_pips"))
        if atr_pips is None:
            atr_pips = _safe_float(fac_h1.get("atr"), 0.0)
            if atr_pips is not None:
                atr_pips *= 100.0

        range_ctx = detect_range_mode(
            fac_h1 if fac_h1 else {},
            fac_h4,
            adx_threshold=self.range_adx,
            bbw_threshold=self.range_bbw,
            atr_threshold=self.range_atr,
        )

        return _Context(
            mid=_latest_mid(),
            rsi=_safe_float(fac_h1.get("rsi"), None),
            adx=_safe_float(fac_h1.get("adx"), None),
            bbw=_safe_float(fac_h1.get("bbw"), None),
            atr_pips=atr_pips,
            vwap_gap_pips=_safe_float(fac_h1.get("vwap_gap"), None),
            range_active=bool(range_ctx.active),
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
            env_prefix=_BB_ENV_PREFIX,
        )
        if ok:
            LOG.info(
                "[EXIT-manual_swing] trade=%s units=%s reason=%s pnl=%.2fp range=%s",
                trade_id,
                units,
                reason,
                pnl,
                range_mode,
            )
            log_metric(
                "manual_swing_exit",
                pnl,
                tags={"reason": reason, "range": str(range_mode), "side": side},
                ts=_utc_now(),
            )
        else:
            LOG.error("[EXIT-manual_swing] close failed trade=%s units=%s reason=%s", trade_id, units, reason)
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
            stop_loss = max(stop_loss, max(1.1, state.hard_stop * 0.5))
            lock_trigger = max(lock_trigger, max(0.6, state.hard_stop * 0.25))
            trail_start = max(trail_start, max(2.0, state.hard_stop * 0.6))
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
            profit_take += 0.5
            trail_start += 0.5
        elif 0.0 < atr <= self.atr_cold:
            profit_take = max(2.0, profit_take * 0.92)
            stop_loss = max(1.3, stop_loss * 0.9)
        lock_buffer = max(lock_buffer, stop_loss * 0.35)

        client_id = _client_id(trade)
        if not client_id:
            LOG.warning("[EXIT-manual_swing] missing client_id trade=%s skip close", trade_id)
            return None

        if pnl < 0:
            factors = all_factors().get("H1") or {}
            try:
                ma10 = float(factors.get("ma10"))
            except Exception:
                ma10 = None
            try:
                ma20 = float(factors.get("ma20"))
            except Exception:
                ma20 = None
            ma_pair = (ma10, ma20) if ma10 is not None and ma20 is not None else None
            reentry = decide_reentry(
                prefix="MANUAL_SWING",
                side=side,
                pnl_pips=pnl,
                rsi=ctx.rsi,
                adx=ctx.adx,
                atr_pips=ctx.atr_pips,
                bbw=ctx.bbw,
                vwap_gap=ctx.vwap_gap_pips,
                ma_pair=ma_pair,
                range_active=range_mode,
                log_tags={"trade": trade_id},
            )
            if reentry.action == "hold":
                return None
            if reentry.action == "exit_reentry" and not reentry.shadow:
                return "reentry_reset"

        # 構造崩れ（H1 MA逆転/ADX低下＋ギャップ縮小）で撤退
        if ctx.adx is not None and ctx.adx < self.range_adx:
            try:
                factors = all_factors().get("H1") or {}
                ma10 = float(factors.get("ma10"))
                ma20 = float(factors.get("ma20"))
                gap = abs(ma10 - ma20) / 0.01
                if (side == "long" and ma10 <= ma20) or (side == "short" and ma10 >= ma20) or (ctx.adx < 16.0 and gap < 3.5):
                    return "structure_break"
            except Exception:
                pass

        if pnl <= -stop_loss:
            return "hard_stop"

        if pnl < 0 and hold_sec >= self.negative_hold_sec:
            if self.allow_negative_exit:
                return "time_cut"

        if pnl < 0:
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

        if ctx.vwap_gap_pips is not None and abs(ctx.vwap_gap_pips) <= self.vwap_grab_gap:
            if pnl > 0.4:
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
            "[EXIT-manual_swing] worker starting (interval=%.2fs)",
            self.loop_interval,
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
                    trade_id = str(tr.get("trade_id"))
                    if await maybe_close_pro_stop(tr, now=now):
                        if trade_id:
                            self._states.pop(trade_id, None)
                        continue
                    try:
                        reason = self._evaluate(tr, ctx, now)
                    except Exception:
                        LOG.exception("[EXIT-manual_swing] evaluate failed trade=%s", tr.get("trade_id"))
                        continue
                    if not reason:
                        continue
                    trade_id = str(tr.get("trade_id"))
                    units = int(tr.get("units", 0) or 0)
                    client_id = _client_id(tr)
                    if not client_id:
                        LOG.warning("[EXIT-manual_swing] missing client_id trade=%s skip close", trade_id)
                        continue
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
                    )
                    self._states.pop(trade_id, None)
        except asyncio.CancelledError:
            LOG.info("[EXIT-manual_swing] worker cancelled")
            raise
        finally:
            try:
                self._pos_manager.close()
            except Exception:
                LOG.exception("[EXIT-manual_swing] failed to close PositionManager")


async def manual_swing_exit_worker() -> None:
    worker = ManualSwingExitWorker()
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    asyncio.run(manual_swing_exit_worker())
