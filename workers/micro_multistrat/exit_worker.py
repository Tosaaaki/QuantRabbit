"""Exit loop for micro_multistrat (micro pocket) - positive exits with MR exception."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Sequence, Set

from analysis.range_guard import detect_range_mode
from workers.common.exit_scaling import momentum_scale, scale_value
from workers.common.exit_utils import close_trade, mark_pnl_pips
from workers.common.reentry_decider import decide_reentry
from execution.strategy_entry import set_trade_protections
from execution.position_manager import PositionManager
from execution.reversion_failure import evaluate_reversion_failure, evaluate_tp_zone
from indicators.factor_cache import all_factors
from market_data import tick_window
from utils.metrics_logger import log_metric
from workers.common.pro_stop import maybe_close_pro_stop
from workers.common.loss_cut import pick_loss_cut_reason, resolve_loss_cut

from . import config
from utils.env_utils import env_bool, env_float
from utils.strategy_protection import exit_profile_for_tag

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


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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

BB_STYLE_DEFAULT = "trend"

LOG = logging.getLogger(__name__)

ALLOWED_TAGS: Set[str] = {
    "MomentumBurst",
    "MicroMomentumStack",
    "MicroPullbackEMA",
    "MicroLevelReactor",
    "MicroRangeBreak",
    "MicroVWAPBound",
    "MicroVWAPRevert",
    "MicroCompressionRevert",
    "MicroTrendRetest",
    "TrendMomentumMicro",
    "VolCompressionBreak",
    "MomentumPulse",
}
REVERSAL_TAG_PREFIXES: Set[str] = {"MicroVWAPBound", "MicroVWAPRevert", "MicroCompressionRevert"}
REVERSAL_PROFILES: Set[str] = {
    "bb_range_reversion",
    "micro_vwap_bound",
    "micro_vwap_revert",
    "compression_revert",
}
OVERLAY_TAG_PREFIXES: Set[str] = {"VolCompressionBreak", "MomentumPulse"}
REVERSAL_BASE_TAGS: Set[str] = {
    "MicroVWAPBound",
    "MicroVWAPRevert",
    "MicroRangeBreak",
    "MicroLevelReactor",
    "MicroCompressionRevert",
    "BB_RSI",
}
POCKET = "micro"


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
    return raw.strip().lower() not in {"", "0", "false", "no"}


EXIT_ENABLED = _bool_env("MICRO_MULTI_EXIT_ENABLED", False)


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
            # タグ欠損はEXIT対象外（誤爆防止）
            continue
        tag_str = str(tag)
        base_tag = tag_str.split("-", 1)[0]
        if tag_str in tags or base_tag in tags:
            filtered.append(tr)
    return filtered


def _reversion_kind(trade: dict) -> Optional[str]:
    thesis = trade.get("entry_thesis") or {}
    tag = (
        thesis.get("strategy_tag_raw")
        or thesis.get("strategy_tag")
        or thesis.get("strategy")
        or trade.get("strategy")
    )
    if tag:
        tag_str = str(tag)
        base_tag = tag_str.split("-", 1)[0]
        if base_tag in REVERSAL_TAG_PREFIXES:
            return "range_mr"
        if base_tag in OVERLAY_TAG_PREFIXES:
            return "mr_overlay"
        tag_lower = tag_str.lower()
        if tag_lower.startswith("mlr-fade") or tag_lower.startswith("mlr-bounce"):
            return "range_mr"
    profile = thesis.get("profile") or thesis.get("strategy_profile")
    if profile:
        return "range_mr" if str(profile) in REVERSAL_PROFILES else None
    return None


def _base_tag(trade: dict) -> Optional[str]:
    thesis = trade.get("entry_thesis") or {}
    tag = (
        thesis.get("strategy_tag_raw")
        or thesis.get("strategy_tag")
        or thesis.get("strategy")
        or thesis.get("tag")
        or trade.get("strategy_tag")
        or trade.get("strategy")
    )
    if not tag:
        return None
    return str(tag).split("-", 1)[0]


def _bb_style_for_trade(trade: dict) -> str:
    if _reversion_kind(trade):
        return "reversion"
    base = _base_tag(trade) or ""
    if base in REVERSAL_BASE_TAGS:
        return "reversion"
    return "trend"


@dataclass
class _TradeState:
    peak: float
    lock_floor: Optional[float] = None
    tp_hint: Optional[float] = None
    trend_hits: int = 0
    partial_done: bool = False
    be_moved: bool = False

    def update(self, pnl: float, lock_buffer: float) -> None:
        if pnl > self.peak:
            self.peak = pnl
        if pnl > 0:
            floor = max(0.0, pnl - lock_buffer)
            self.lock_floor = floor if self.lock_floor is None else max(self.lock_floor, floor)


class MicroMultiExitWorker:
    """
    micro_multistrat 用の専用 EXIT。
    - min_hold 経過まではクローズ禁止
    - PnL>0 のみクローズ（TP/トレール/RSI利確/レンジ長時間微益のみ）
    - ポケット共通 ExitManager 非依存
    """

    def __init__(self) -> None:
        self.loop_interval = max(0.3, _float_env("MICRO_MULTI_EXIT_LOOP_INTERVAL_SEC", 1.0))
        self._pos_manager = PositionManager()
        self._states: Dict[str, _TradeState] = {}
        self._loss_cut_last_ts: Dict[str, float] = {}

        self.profit_take = max(1.2, _float_env("MICRO_MULTI_EXIT_PROFIT_PIPS", 2.0))
        self.trail_start = max(1.4, _float_env("MICRO_MULTI_EXIT_TRAIL_START_PIPS", 2.8))
        self.trail_backoff = max(0.2, _float_env("MICRO_MULTI_EXIT_TRAIL_BACKOFF_PIPS", 0.9))
        self.lock_buffer = max(0.15, _float_env("MICRO_MULTI_EXIT_LOCK_BUFFER_PIPS", 0.4))
        self.min_hold_sec = max(5.0, _float_env("MICRO_MULTI_EXIT_MIN_HOLD_SEC", 18.0))

        self.range_profit_take = max(1.0, _float_env("MICRO_MULTI_EXIT_RANGE_PROFIT_PIPS", 1.6))
        self.range_trail_start = max(1.1, _float_env("MICRO_MULTI_EXIT_RANGE_TRAIL_START_PIPS", 2.1))
        self.range_trail_backoff = max(0.2, _float_env("MICRO_MULTI_EXIT_RANGE_TRAIL_BACKOFF_PIPS", 0.6))
        self.range_lock_buffer = max(0.1, _float_env("MICRO_MULTI_EXIT_RANGE_LOCK_BUFFER_PIPS", 0.3))
        self.range_max_hold_sec = max(90.0, _float_env("MICRO_MULTI_EXIT_RANGE_MAX_HOLD_SEC", 30 * 60))

        self.range_adx = max(5.0, _float_env("MICRO_MULTI_EXIT_RANGE_ADX", 22.0))
        self.range_bbw = max(0.02, _float_env("MICRO_MULTI_EXIT_RANGE_BBW", 0.22))
        self.range_atr = max(0.4, _float_env("MICRO_MULTI_EXIT_RANGE_ATR", 6.0))

        self.rsi_take_long = _float_env("MICRO_MULTI_EXIT_RSI_TAKE_LONG", 70.0)
        self.rsi_take_short = _float_env("MICRO_MULTI_EXIT_RSI_TAKE_SHORT", 30.0)

        self.partial_trigger = max(0.9, _float_env("MICRO_MULTI_EXIT_PARTIAL_TRIGGER_PIPS", 1.1))
        self.partial_trigger_low_vol = max(0.8, _float_env("MICRO_MULTI_EXIT_PARTIAL_TRIGGER_PIPS_LOWVOL", 0.9))
        self.partial_fraction = min(0.8, max(0.1, _float_env("MICRO_MULTI_EXIT_PARTIAL_FRACTION", 0.35)))
        self.partial_min_units = max(20, int(_float_env("MICRO_MULTI_EXIT_PARTIAL_MIN_UNITS", 500)))
        self.partial_min_remaining = max(20, int(_float_env("MICRO_MULTI_EXIT_PARTIAL_MIN_REMAIN", 500)))
        self.be_buffer_pips = max(0.05, _float_env("MICRO_MULTI_EXIT_BE_BUFFER_PIPS", 0.2))

    def _context(self) -> tuple[Optional[float], Optional[float], bool, Optional[float], Optional[float]]:
        factors = all_factors()
        fac_m1 = factors.get("M1") or {}
        fac_h4 = factors.get("H4") or {}

        def _safe_float(val: object) -> Optional[float]:
            try:
                return float(val)
            except Exception:
                return None

        adx = _safe_float(fac_m1.get("adx"))
        bbw = _safe_float(fac_m1.get("bbw"))
        atr = _safe_float(fac_m1.get("atr_pips")) or (_safe_float(fac_m1.get("atr")) or 0.0) * 100.0
        range_ctx = detect_range_mode(
            fac_m1,
            fac_h4,
            adx_threshold=self.range_adx,
            bbw_threshold=self.range_bbw,
            atr_threshold=self.range_atr,
        )
        rsi = _safe_float(fac_m1.get("rsi"))
        return _latest_mid(), rsi, bool(range_ctx.active), _safe_float(fac_m1.get("adx")), _safe_float(fac_m1.get("bbw"))

    def _trade_has_stop_loss(self, trade: dict) -> bool:
        sl = trade.get("stop_loss")
        if isinstance(sl, dict):
            price = sl.get("price")
            try:
                return float(price) > 0.0
            except (TypeError, ValueError):
                return False
        return False

    async def _close(
        self,
        trade_id: str,
        units: int,
        reason: str,
        pnl: float,
        client_order_id: Optional[str],
        bb_style: Optional[str] = None,
        allow_negative: bool = False,
    ) -> None:
        if _BB_EXIT_ENABLED:
            style = bb_style or BB_STYLE_DEFAULT
            if not _bb_exit_should_bypass(reason, pnl, allow_negative):
                fac = all_factors().get(_BB_EXIT_TF) or {}
                price = _bb_exit_price(fac)
                side = "long" if units > 0 else "short"
                if not _bb_exit_allowed(style, side, price, fac):
                    LOG.info(
                        "[exit-bb] trade=%s reason=%s price=%.3f style=%s",
                        trade_id,
                        reason,
                        price or 0.0,
                        style,
                    )
                    return
        if pnl <= 0:
            allow_negative = True
        ok = await close_trade(
            trade_id,
            units,
            client_order_id=client_order_id,
            allow_negative=allow_negative,
            exit_reason=reason,
            env_prefix=_BB_ENV_PREFIX,
        )
        if ok:
            LOG.info("[EXIT-micro_multi] trade=%s units=%s reason=%s pnl=%.2fp", trade_id, units, reason, pnl)
        else:
            LOG.error("[EXIT-micro_multi] close failed trade=%s units=%s reason=%s", trade_id, units, reason)

    async def _review_trade(
        self,
        trade: dict,
        now: datetime,
        mid: float,
        rsi: Optional[float],
        range_active: bool,
        adx: Optional[float],
        bbw: Optional[float],
    ) -> None:
        trade_id = str(trade.get("trade_id"))
        if not trade_id:
            return
        units = int(trade.get("units", 0) or 0)
        if units == 0:
            return
        bb_style = _bb_style_for_trade(trade)

        entry = float(trade.get("price") or 0.0)
        if entry <= 0.0:
            return

        side = "long" if units > 0 else "short"
        pnl = mark_pnl_pips(entry, units, mid=mid)
        opened_at = _parse_time(trade.get("open_time"))
        hold_sec = (now - opened_at).total_seconds() if opened_at else 0.0

        thesis = trade.get("entry_thesis") or {}
        if not isinstance(thesis, dict):
            thesis = {}
        reversion_kind = _reversion_kind(trade)

        state = self._states.get(trade_id)
        if state is None:
            tp_hint = thesis.get("tp_pips")
            try:
                tp_hint_val = float(tp_hint) if tp_hint is not None else None
            except Exception:
                tp_hint_val = None
            state = _TradeState(peak=pnl, tp_hint=tp_hint_val)
            self._states[trade_id] = state

        strategy_tag = (
            thesis.get("strategy_tag")
            or thesis.get("strategy_tag_raw")
            or thesis.get("strategy")
            or thesis.get("tag")
            or trade.get("strategy_tag")
            or trade.get("strategy")
            or "micro_multi"
        )
        scale, _ = momentum_scale(
            pocket=POCKET,
            strategy_tag=strategy_tag,
            entry_thesis=thesis,
            range_active=range_active,
            env_prefix=_BB_ENV_PREFIX,
        )

        min_hold = self.min_hold_sec
        if not range_active:
            min_hold = scale_value(self.min_hold_sec, scale=scale, floor=self.min_hold_sec)

        profit_take = self.range_profit_take if range_active else scale_value(
            self.profit_take, scale=scale, floor=self.profit_take
        )
        trail_start = self.range_trail_start if range_active else scale_value(
            self.trail_start, scale=scale, floor=self.trail_start
        )
        trail_backoff = self.range_trail_backoff if range_active else scale_value(
            self.trail_backoff, scale=scale, floor=self.trail_backoff
        )
        lock_buffer = self.range_lock_buffer if range_active else scale_value(
            self.lock_buffer, scale=scale, floor=self.lock_buffer
        )

        state.update(pnl, lock_buffer)

        client_ext = trade.get("clientExtensions")
        client_id = trade.get("client_order_id")
        if not client_id and isinstance(client_ext, dict):
            client_id = client_ext.get("id")
        if not client_id:
            LOG.warning("[EXIT-micro_multi] missing client_id trade=%s skip close", trade_id)
            return
        if await maybe_close_pro_stop(trade, now=now):
            return

        if hold_sec < min_hold:
            return

        # Strategy-level "loss-cut": once a trade is beyond the point of return, exit and redeploy.
        # Configured per-strategy via config/strategy_exit_protections.yaml -> exit_profile.loss_cut_*.
        if pnl <= 0:
            base_tag = _base_tag(trade) or str(strategy_tag).split("-", 1)[0]
            exit_profile = exit_profile_for_tag(base_tag or str(strategy_tag))
            sl_hint = _safe_float(thesis.get("hard_stop_pips") or thesis.get("sl_pips"))
            params = resolve_loss_cut(exit_profile, sl_pips=sl_hint)
            reason = pick_loss_cut_reason(
                pnl_pips=float(pnl),
                hold_sec=float(hold_sec),
                params=params,
                has_stop_loss=self._trade_has_stop_loss(trade),
            )
            if reason:
                now_mono = time.monotonic()
                last = self._loss_cut_last_ts.get(trade_id, 0.0)
                if params.cooldown_sec > 0.0 and (now_mono - last) < params.cooldown_sec:
                    return
                self._loss_cut_last_ts[trade_id] = now_mono
                await self._close(
                    trade_id,
                    -units,
                    reason,
                    pnl,
                    client_id,
                    bb_style=bb_style,
                    allow_negative=True,
                )
                self._states.pop(trade_id, None)
                return

        if pnl < 0 and not reversion_kind:
            fac_m1 = all_factors().get("M1") or {}
            rsi_val = rsi
            adx_val = adx
            bbw_val = bbw
            atr_pips = _safe_float(fac_m1.get("atr_pips"))
            if atr_pips is None:
                atr_val = _safe_float(fac_m1.get("atr"))
                if atr_val is not None:
                    atr_pips = atr_val * 100.0
            vwap_gap = _safe_float(fac_m1.get("vwap_gap"))
            ma10 = _safe_float(fac_m1.get("ma10"))
            ma20 = _safe_float(fac_m1.get("ma20"))
            ma_pair = (ma10, ma20) if ma10 is not None and ma20 is not None else None
            reentry = decide_reentry(
                prefix="MICRO_MULTI",
                side=side,
                pnl_pips=pnl,
                rsi=rsi_val,
                adx=adx_val,
                atr_pips=atr_pips,
                bbw=bbw_val,
                vwap_gap=vwap_gap,
                ma_pair=ma_pair,
                range_active=range_active,
                log_tags={"trade": trade_id},
            )
            if reentry.action == "hold":
                return
            if reentry.action == "exit_reentry" and not reentry.shadow:
                await self._close(
                    trade_id,
                    -units,
                    "reentry_reset",
                    pnl,
                    client_id,
                    bb_style=bb_style,
                    allow_negative=True,
                )
                self._states.pop(trade_id, None)
                return

        candle_reason = _exit_candle_reversal("long" if units > 0 else "short")
        if candle_reason and pnl >= 0:
            candle_client_id = trade.get("client_order_id")
            if not candle_client_id:
                client_ext = trade.get("clientExtensions")
                if isinstance(client_ext, dict):
                    candle_client_id = client_ext.get("id")
            if candle_client_id:
                await self._close(
                    trade_id,
                    -units,
                    candle_reason,
                    pnl,
                    candle_client_id,
                    bb_style=bb_style,
                    allow_negative=False,
                )
                if hasattr(self, "_states"):
                    self._states.pop(trade_id, None)
                return
        low_vol = bool(thesis.get("low_vol"))
        if pnl > 0 and reversion_kind and not state.partial_done:
            trigger = self.partial_trigger_low_vol if low_vol else self.partial_trigger
            if pnl >= trigger:
                reduce_units = int(abs(units) * self.partial_fraction)
                remaining = abs(units) - reduce_units
                if reduce_units >= self.partial_min_units and remaining >= self.partial_min_remaining:
                    ok = await close_trade(
                        trade_id,
                        reduce_units,
                        client_order_id=client_id,
                        allow_negative=False,
                        exit_reason="partial_take",
                        env_prefix=_BB_ENV_PREFIX,
                    )
                    if ok:
                        state.partial_done = True
                        log_metric(
                            "micro_multi_partial_take",
                            pnl,
                            tags={
                                "side": side,
                                "kind": reversion_kind,
                                "low_vol": str(low_vol),
                            },
                            ts=now,
                        )
                        be = entry + (self.be_buffer_pips * 0.01) if side == "long" else entry - (self.be_buffer_pips * 0.01)
                        be_ok = await set_trade_protections(trade_id, sl_price=round(be, 3), tp_price=None)
                        if be_ok:
                            state.be_moved = True
                        return

        if state.partial_done and not state.be_moved:
            be = entry + (self.be_buffer_pips * 0.01) if side == "long" else entry - (self.be_buffer_pips * 0.01)
            be_ok = False
            if side == "long" and mid >= be:
                be_ok = await set_trade_protections(trade_id, sl_price=round(be, 3), tp_price=None)
            elif side == "short" and mid <= be:
                be_ok = await set_trade_protections(trade_id, sl_price=round(be, 3), tp_price=None)
            if be_ok:
                state.be_moved = True
        if pnl <= 0 and reversion_kind:
            struct_override = [] if reversion_kind == "mr_overlay" else None
            decision = evaluate_reversion_failure(
                trade,
                current_price=mid,
                now=now,
                side=side,
                env_tf="H1",
                struct_tf="M5",
                trend_hits=state.trend_hits,
                struct_candles=struct_override,
            )
            state.trend_hits = decision.trend_hits
            if decision.should_exit and decision.reason:
                LOG.info(
                    "[EXIT-micro_multi] reversion_exit trade=%s reason=%s detail=%s",
                    trade_id,
                    decision.reason,
                    decision.debug,
                )
                log_metric(
                    "micro_multi_reversion_exit",
                    pnl,
                    tags={"reason": decision.reason, "side": side, "kind": reversion_kind},
                    ts=now,
                )
                await self._close(
                    trade_id,
                    -units,
                    decision.reason,
                    pnl,
                    client_id,
                    bb_style=bb_style,
                    allow_negative=True,
                )
                self._states.pop(trade_id, None)
                return

        if pnl > 0 and reversion_kind in {"range_mr", "mr_overlay"}:
            tp_decision = evaluate_tp_zone(
                trade,
                current_price=mid,
                side=side,
                env_tf="H1",
                struct_tf="M5",
            )
            if tp_decision.should_exit:
                LOG.info(
                    "[EXIT-micro_multi] tp_zone trade=%s detail=%s",
                    trade_id,
                    tp_decision.debug,
                )
                log_metric(
                    "micro_multi_tp_zone",
                    pnl,
                    tags={"side": side, "kind": reversion_kind},
                    ts=now,
                )
                await self._close(trade_id, -units, "take_profit_zone", pnl, client_id, bb_style=bb_style)
                self._states.pop(trade_id, None)
                return

        base_tag = _base_tag(trade)
        if base_tag == "VolCompressionBreak" and pnl < 0:
            compression_active = range_active or (
                adx is not None
                and bbw is not None
                and adx <= self.range_adx
                and bbw <= self.range_bbw
            )
            if compression_active:
                await self._close(
                    trade_id,
                    -units,
                    "compression_fail",
                    pnl,
                    client_id,
                    bb_style=bb_style,
                    allow_negative=True,
                )
                self._states.pop(trade_id, None)
                return

        if state.tp_hint:
            profit_take = max(profit_take, max(1.0, state.tp_hint * 0.9))
            trail_start = max(trail_start, profit_take * 0.9)
            lock_buffer = max(lock_buffer, profit_take * 0.4)

        lock_trigger = max(0.8, profit_take * 0.35)
        if (
            state.lock_floor is not None
            and state.peak >= lock_trigger
            and pnl > 0
            and pnl <= state.lock_floor
        ):
            await self._close(trade_id, -units, "lock_floor", pnl, client_id, bb_style=bb_style)
            self._states.pop(trade_id, None)
            return

        if state.peak > 0 and state.peak >= trail_start and pnl > 0 and pnl <= state.peak - trail_backoff:
            await self._close(trade_id, -units, "trail_take", pnl, client_id, bb_style=bb_style)
            self._states.pop(trade_id, None)
            return

        if pnl >= profit_take:
            await self._close(trade_id, -units, "take_profit", pnl, client_id, bb_style=bb_style)
            self._states.pop(trade_id, None)
            return

        if pnl > 0 and rsi is not None:
            if side == "long" and rsi >= self.rsi_take_long:
                await self._close(trade_id, -units, "rsi_take", pnl, client_id, bb_style=bb_style)
                self._states.pop(trade_id, None)
                return
            if side == "short" and rsi <= self.rsi_take_short:
                await self._close(trade_id, -units, "rsi_take", pnl, client_id, bb_style=bb_style)
                self._states.pop(trade_id, None)
                return

        if range_active and hold_sec >= self.range_max_hold_sec and pnl > 0:
            await self._close(trade_id, -units, "range_timeout", pnl, client_id, bb_style=bb_style)
            self._states.pop(trade_id, None)
            return

    async def run(self) -> None:
        if not EXIT_ENABLED:
            LOG.info("[EXIT-micro_multistrat] disabled (idle)")
            try:
                while True:
                    await asyncio.sleep(3600.0)
            except asyncio.CancelledError:
                return
        LOG.info(
            "[EXIT-micro_multistrat] worker starting (interval=%.2fs tags=%s)",
            self.loop_interval,
            ",".join(sorted(ALLOWED_TAGS)),
        )
        try:
            while True:
                await asyncio.sleep(self.loop_interval)
                positions = self._pos_manager.get_open_positions()
                pocket_info = positions.get(POCKET) or {}
                trades = _filter_trades(pocket_info.get("open_trades") or [], ALLOWED_TAGS)
                active_ids = {str(tr.get("trade_id")) for tr in trades if tr.get("trade_id")}
                for tid in list(self._states.keys()):
                    if tid not in active_ids:
                        self._states.pop(tid, None)
                for tid in list(self._loss_cut_last_ts.keys()):
                    if tid not in active_ids:
                        self._loss_cut_last_ts.pop(tid, None)
                if not trades:
                    continue

                mid, rsi, range_active, adx, bbw = self._context()
                if mid is None:
                    continue

                now = _utc_now()
                for tr in trades:
                    try:
                        await self._review_trade(tr, now, mid, rsi, range_active, adx, bbw)
                    except Exception:
                        LOG.exception("[EXIT-micro_multistrat] review failed trade=%s", tr.get("trade_id"))
                        continue
        except asyncio.CancelledError:
            LOG.info("[EXIT-micro_multistrat] worker cancelled")
            raise
        finally:
            try:
                self._pos_manager.close()
            except Exception:
                LOG.exception("[EXIT-micro_multistrat] failed to close PositionManager")


async def micro_multistrat_exit_worker() -> None:
    worker = MicroMultiExitWorker()
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
    asyncio.run(micro_multistrat_exit_worker())
