"""Exit loop for mirror_spike_tight worker (scalp pocket) with technical filters."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

from analysis.ma_projection import compute_adx_projection, compute_bbw_projection, compute_ma_projection, compute_rsi_projection
from analysis.range_guard import detect_range_mode
from workers.common.exit_utils import close_trade
from execution.position_manager import PositionManager
from execution.reversion_failure import evaluate_reversion_failure, evaluate_tp_zone
from indicators.factor_cache import all_factors
from market_data import tick_window
from utils.metrics_logger import log_metric
from workers.common.exit_scaling import TPScaleConfig, apply_tp_virtual_floor


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
_NEG_EXIT_GATE_ENABLED = os.getenv("MIRROR_SPIKE_TIGHT_EXIT_NEG_GATE", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}
_NEG_EXIT_PROJ_SCORE = float(os.getenv("MIRROR_SPIKE_TIGHT_EXIT_NEG_PROJ_SCORE", "0.35"))
_NEG_EXIT_CANDLE_SCORE = float(os.getenv("MIRROR_SPIKE_TIGHT_EXIT_NEG_CANDLE_SCORE", "0.35"))
_NEG_EXIT_CANDLE_MIN_CONF = float(os.getenv("MIRROR_SPIKE_TIGHT_EXIT_NEG_CANDLE_MIN_CONF", "0.32"))
_NEG_EXIT_LOG_INTERVAL_SEC = float(os.getenv("MIRROR_SPIKE_TIGHT_EXIT_NEG_LOG_INTERVAL_SEC", "6.0"))
_PROJ_TF_MINUTES = {"M1": 1.0, "M5": 5.0, "H1": 60.0, "H4": 240.0, "D1": 1440.0}
_LAST_NEG_GATE_LOG_TS = 0.0


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

def _proj_score_ma(ma, side, opp_block_bars):
    if ma is None:
        return None
    align = ma.gap_pips >= 0 if side == "long" else ma.gap_pips <= 0
    cross_soon = ma.projected_cross_bars is not None and ma.projected_cross_bars <= opp_block_bars
    if align and not cross_soon:
        return 0.7
    if align and cross_soon:
        return -0.4
    if cross_soon:
        return -0.8
    return -0.5


def _proj_score_rsi(rsi, side, long_target, short_target, overheat_bars):
    if rsi is None:
        return None
    score = 0.0
    if side == "long":
        if rsi.rsi >= long_target and rsi.slope_per_bar > 0:
            score = 0.4
        elif rsi.rsi <= (long_target - 8) and rsi.slope_per_bar < 0:
            score = -0.4
        if rsi.eta_upper_bars is not None and rsi.eta_upper_bars <= overheat_bars:
            score -= 0.2
    else:
        if rsi.rsi <= short_target and rsi.slope_per_bar < 0:
            score = 0.4
        elif rsi.rsi >= (short_target + 8) and rsi.slope_per_bar > 0:
            score = -0.4
        if rsi.eta_lower_bars is not None and rsi.eta_lower_bars <= overheat_bars:
            score -= 0.2
    return score


def _proj_score_adx(adx, trend_mode, threshold):
    if adx is None:
        return None
    if trend_mode:
        if adx.adx >= threshold and adx.slope_per_bar >= 0:
            return 0.4
        if adx.adx <= threshold and adx.slope_per_bar < 0:
            return -0.4
        return 0.0
    if adx.adx >= threshold and adx.slope_per_bar > 0:
        return -0.5
    if adx.adx <= threshold and adx.slope_per_bar < 0:
        return 0.3
    return 0.0


def _proj_score_bbw(bbw, threshold):
    if bbw is None:
        return None
    if bbw.bbw <= threshold and bbw.slope_per_bar <= 0:
        return 0.5
    if bbw.bbw > threshold and bbw.slope_per_bar > 0:
        return -0.5
    return 0.0


def _projection_score(side: str) -> tuple[Optional[float], dict]:
    fac = all_factors().get("M1") or {}
    candles = fac.get("candles") or []
    if not candles or len(candles) < 30:
        return None, {}

    params = {
        "adx_threshold": 16.0,
        "bbw_threshold": 0.14,
        "opp_block_bars": 4.0,
        "long_target": 45.0,
        "short_target": 55.0,
        "overheat_bars": 3.0,
        "weights": {"bbw": 0.40, "rsi": 0.35, "adx": 0.25},
    }
    minutes = _PROJ_TF_MINUTES.get("M1", 1.0)
    ma = compute_ma_projection({"candles": candles}, timeframe_minutes=minutes)
    rsi = compute_rsi_projection(candles, timeframe_minutes=minutes)
    adx = compute_adx_projection(candles, timeframe_minutes=minutes, trend_threshold=params["adx_threshold"])
    bbw = compute_bbw_projection(candles, timeframe_minutes=minutes, squeeze_threshold=params["bbw_threshold"])

    scores = {}
    ma_score = _proj_score_ma(ma, side, params["opp_block_bars"])
    if ma_score is not None and "ma" in params["weights"]:
        scores["ma"] = ma_score
    rsi_score = _proj_score_rsi(rsi, side, params["long_target"], params["short_target"], params["overheat_bars"])
    if rsi_score is not None and "rsi" in params["weights"]:
        scores["rsi"] = rsi_score
    adx_score = _proj_score_adx(adx, False, params["adx_threshold"])
    if adx_score is not None and "adx" in params["weights"]:
        scores["adx"] = adx_score
    bbw_score = _proj_score_bbw(bbw, params["bbw_threshold"])
    if bbw_score is not None and "bbw" in params["weights"]:
        scores["bbw"] = bbw_score

    weight_sum = 0.0
    score_sum = 0.0
    for key, score in scores.items():
        weight = params["weights"].get(key, 0.0)
        weight_sum += weight
        score_sum += weight * score
    score = score_sum / weight_sum if weight_sum > 0 else 0.0
    detail = {
        "tf": "M1",
        "mode": "range",
        "score": round(score, 3),
        "scores": {k: round(v, 3) for k, v in scores.items()},
    }
    return score, detail


def _candle_reversal_score(side: str, *, min_conf: float) -> tuple[Optional[float], dict]:
    tf = _candle_tf_for_worker()
    candles = (all_factors().get(tf) or {}).get("candles") or []
    if not candles:
        return None, {}
    score, detail = _score_candle(candles=candles, side=side, min_conf=min_conf)
    detail = detail if isinstance(detail, dict) else {}
    detail["tf"] = tf
    return score, detail


def _neg_exit_gate(side: str) -> tuple[bool, dict]:
    if not _NEG_EXIT_GATE_ENABLED:
        return True, {}
    candle_score, candle_detail = _candle_reversal_score(side, min_conf=_NEG_EXIT_CANDLE_MIN_CONF)
    proj_score, proj_detail = _projection_score(side)
    candle_ok = candle_score is not None and candle_score <= -_NEG_EXIT_CANDLE_SCORE
    proj_ok = proj_score is not None and proj_score <= -_NEG_EXIT_PROJ_SCORE
    ok = candle_ok and proj_ok
    detail = {
        "candle_ok": candle_ok,
        "proj_ok": proj_ok,
        "candle": candle_detail,
        "projection": proj_detail,
    }
    return ok, detail


def _log_neg_gate_block(reason: str, pnl: float, detail: dict) -> None:
    global _LAST_NEG_GATE_LOG_TS
    now = time.monotonic()
    if now - _LAST_NEG_GATE_LOG_TS < _NEG_EXIT_LOG_INTERVAL_SEC:
        return
    _LAST_NEG_GATE_LOG_TS = now
    LOG.info("[exit-neg-gate] block reason=%s pnl=%.2f detail=%s", reason, pnl, detail)
    try:
        log_metric(
            "mirror_spike_tight_neg_gate_block",
            float(pnl),
            tags={"reason": reason},
            ts=_utc_now(),
        )
    except Exception:
        pass

BB_STYLE = "reversion"
LOG = logging.getLogger(__name__)

ALLOWED_TAGS = {"mirror_spike_tight"}
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


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _latest_quote() -> tuple[Optional[float], Optional[float], Optional[float]]:
    tick = tick_window.recent_ticks(seconds=2.0, limit=1)
    if tick:
        last = tick[-1]
        bid = last.get("bid")
        ask = last.get("ask")
        mid = last.get("mid")
        try:
            bid_f = float(bid) if bid is not None else None
        except Exception:
            bid_f = None
        try:
            ask_f = float(ask) if ask is not None else None
        except Exception:
            ask_f = None
        try:
            mid_f = float(mid) if mid is not None else None
        except Exception:
            mid_f = None
        if mid_f is None and bid_f is not None and ask_f is not None:
            try:
                mid_f = (bid_f + ask_f) / 2.0
            except Exception:
                mid_f = None
        return bid_f, ask_f, mid_f
    try:
        mid_f = float(all_factors().get("M1", {}).get("close"))
    except Exception:
        mid_f = None
    return None, None, mid_f


def _latest_mid() -> Optional[float]:
    _, _, mid = _latest_quote()
    return mid


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
    trend_hits: int = 0


@dataclass
class _Context:
    mid: Optional[float]
    bid: Optional[float]
    ask: Optional[float]
    rsi: Optional[float]
    adx: Optional[float]
    bbw: Optional[float]
    atr_pips: Optional[float]
    vwap_gap_pips: Optional[float]
    range_active: bool


class MirrorSpikeTightExitWorker:
    """PnL + RSI/ATR/VWAP/レンジ判定を組み合わせた mirror_spike_tight EXIT."""

    def __init__(self) -> None:
        self.tp_scale = TPScaleConfig()
        self.loop_interval = max(0.3, _float_env("MIRROR_SPIKE_TIGHT_EXIT_LOOP_INTERVAL_SEC", 0.8))
        self._pos_manager = PositionManager()
        self._states: Dict[str, _TradeState] = {}

        self.profit_take = max(0.8, _float_env("MIRROR_SPIKE_TIGHT_EXIT_PROFIT_PIPS", 1.7))
        self.trail_start = max(1.0, _float_env("MIRROR_SPIKE_TIGHT_EXIT_TRAIL_START_PIPS", 2.3))
        self.trail_backoff = max(0.2, _float_env("MIRROR_SPIKE_TIGHT_EXIT_TRAIL_BACKOFF_PIPS", 0.7))
        self.stop_loss = max(0.6, _float_env("MIRROR_SPIKE_TIGHT_EXIT_STOP_LOSS_PIPS", 1.2))
        self.max_hold_sec = max(30.0, _float_env("MIRROR_SPIKE_TIGHT_EXIT_MAX_HOLD_SEC", 18 * 60))
        self.lock_trigger = max(0.3, _float_env("MIRROR_SPIKE_TIGHT_EXIT_LOCK_TRIGGER_PIPS", 0.75))
        self.lock_buffer = max(0.1, _float_env("MIRROR_SPIKE_TIGHT_EXIT_LOCK_BUFFER_PIPS", 0.4))

        self.range_profit_take = max(0.6, _float_env("MIRROR_SPIKE_TIGHT_EXIT_RANGE_PROFIT_PIPS", 1.4))
        self.range_trail_start = max(0.9, _float_env("MIRROR_SPIKE_TIGHT_EXIT_RANGE_TRAIL_START_PIPS", 1.9))
        self.range_trail_backoff = max(0.2, _float_env("MIRROR_SPIKE_TIGHT_EXIT_RANGE_TRAIL_BACKOFF_PIPS", 0.55))
        self.range_stop_loss = max(0.5, _float_env("MIRROR_SPIKE_TIGHT_EXIT_RANGE_STOP_LOSS_PIPS", 0.95))
        self.range_max_hold_sec = max(30.0, _float_env("MIRROR_SPIKE_TIGHT_EXIT_RANGE_MAX_HOLD_SEC", 14 * 60))
        self.range_lock_trigger = max(0.25, _float_env("MIRROR_SPIKE_TIGHT_EXIT_RANGE_LOCK_TRIGGER_PIPS", 0.65))
        self.range_lock_buffer = max(0.1, _float_env("MIRROR_SPIKE_TIGHT_EXIT_RANGE_LOCK_BUFFER_PIPS", 0.28))

        self.range_adx = max(5.0, _float_env("MIRROR_SPIKE_TIGHT_EXIT_RANGE_ADX", 22.0))
        self.range_bbw = max(0.02, _float_env("MIRROR_SPIKE_TIGHT_EXIT_RANGE_BBW", 0.20))
        self.range_atr = max(0.4, _float_env("MIRROR_SPIKE_TIGHT_EXIT_RANGE_ATR", 6.0))
        self._range_thresholds = (self.range_adx, self.range_bbw, self.range_atr)

        self.rsi_fade_long = _float_env("MIRROR_SPIKE_TIGHT_EXIT_RSI_FADE_LONG", 42.0)
        self.rsi_fade_short = _float_env("MIRROR_SPIKE_TIGHT_EXIT_RSI_FADE_SHORT", 58.0)
        self.rsi_take_long = _float_env("MIRROR_SPIKE_TIGHT_EXIT_RSI_TAKE_LONG", 69.0)
        self.rsi_take_short = _float_env("MIRROR_SPIKE_TIGHT_EXIT_RSI_TAKE_SHORT", 31.0)
        self.negative_hold_sec = max(15.0, _float_env("MIRROR_SPIKE_TIGHT_EXIT_NEG_HOLD_SEC", 120.0))
        self.allow_negative_exit = _bool_env("MIRROR_SPIKE_TIGHT_EXIT_ALLOW_NEGATIVE", True)

        self.vwap_grab_gap = max(0.1, _float_env("MIRROR_SPIKE_TIGHT_EXIT_VWAP_GAP_PIPS", 0.75))

        self.atr_hot = max(0.5, _float_env("MIRROR_SPIKE_TIGHT_EXIT_ATR_HOT_PIPS", 5.0))
        self.atr_cold = max(0.3, _float_env("MIRROR_SPIKE_TIGHT_EXIT_ATR_COLD_PIPS", 1.1))

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

        adx_th, bbw_th, atr_th = self._range_thresholds
        range_ctx = detect_range_mode(
            fac_m1,
            fac_h4,
            adx_threshold=adx_th,
            bbw_threshold=bbw_th,
            atr_threshold=atr_th,
        )

        bid, ask, mid = _latest_quote()
        return _Context(
            mid=mid,
            bid=bid,
            ask=ask,
            rsi=_safe_float(fac_m1.get("rsi"), None),
            adx=_safe_float(fac_m1.get("adx"), None),
            bbw=_safe_float(fac_m1.get("bbw"), None),
            atr_pips=atr_pips,
            vwap_gap_pips=_safe_float(fac_m1.get("vwap_gap"), None),
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
        )
        if ok:
            LOG.info(
                "[EXIT-mirror_spike_tight] trade=%s units=%s reason=%s pnl=%.2fp range=%s",
                trade_id,
                units,
                reason,
                pnl,
                range_mode,
            )
            log_metric(
                "mirror_spike_tight_exit",
                pnl,
                tags={"reason": reason, "range": str(range_mode), "side": side},
                ts=_utc_now(),
            )
        else:
            LOG.error("[EXIT-mirror_spike_tight] close failed trade=%s units=%s reason=%s", trade_id, units, reason)
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
        exit_px = None
        if side == "long":
            exit_px = ctx.bid if ctx.bid is not None else ctx.mid
        else:
            exit_px = ctx.ask if ctx.ask is not None else ctx.mid
        if exit_px is None:
            return None
        pnl = (exit_px - entry_price) * 100.0 if side == "long" else (entry_price - exit_px) * 100.0
        opened_at = _parse_time(trade.get("open_time"))
        hold_sec = (now - opened_at).total_seconds() if opened_at else 0.0
        candle_reason = _exit_candle_reversal("long" if units > 0 else "short")
        if candle_reason and pnl >= 0:
            return candle_reason
        neg_gate_ok = True
        neg_gate_detail: dict = {}
        if pnl < 0 and self.allow_negative_exit:
            neg_gate_ok, neg_gate_detail = _neg_exit_gate(side)

        state = self._states.get(trade_id)
        if state is None:
            state = _TradeState(peak=pnl)
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

        if state.hard_stop:
            stop_loss = max(stop_loss, state.hard_stop)
            lock_trigger = max(lock_trigger, max(0.25, state.hard_stop * 0.25))
            trail_start = max(trail_start, max(1.0, state.hard_stop * 0.6))
            max_hold = max(max_hold, self.max_hold_sec * 1.1)

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
            profit_take += 0.25
            trail_start += 0.25
        elif 0.0 < atr <= self.atr_cold:
            profit_take = max(0.9, profit_take * 0.9)
            stop_loss = max(0.6, stop_loss * 0.9)

        client_id = _client_id(trade)
        if not client_id:
            LOG.warning("[EXIT-mirror_spike_tight] missing client_id trade=%s skip close", trade_id)
            return None

        if pnl < 0:
            factors = all_factors().get("M1") or {}
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
                prefix="MIRROR_SPIKE_TIGHT",
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

        if ctx.adx is not None and ctx.adx < self.range_adx:
            try:
                factors = all_factors().get("M1") or {}
                ma10 = float(factors.get("ma10"))
                ma20 = float(factors.get("ma20"))
                gap = abs(ma10 - ma20) / 0.01
                dir_long = units > 0
                if (dir_long and ma10 <= ma20) or ((not dir_long) and ma10 >= ma20) or (ctx.adx < 12.0 and gap < 2.0):
                    if pnl >= 0 or neg_gate_ok:
                        return "structure_break"
                    _log_neg_gate_block("structure_break", pnl, neg_gate_detail)
            except Exception:
                pass

        if pnl <= -stop_loss:
            return "hard_stop"

        if pnl <= 0 and self.allow_negative_exit:
            if neg_gate_ok:
                decision = evaluate_reversion_failure(
                    trade,
                    current_price=ctx.mid,
                    now=now,
                    side=side,
                    env_tf="M5",
                    struct_tf="M1",
                    trend_hits=state.trend_hits,
                )
                state.trend_hits = decision.trend_hits
                if decision.should_exit and decision.reason:
                    return decision.reason
            else:
                _log_neg_gate_block("reversion_failure", pnl, neg_gate_detail)

        if pnl > 0:
            tp_decision = evaluate_tp_zone(
                trade,
                current_price=ctx.mid,
                side=side,
                env_tf="M5",
                struct_tf="M1",
            )
            if tp_decision.should_exit:
                return "take_profit_zone"

        if pnl < 0 and hold_sec >= self.negative_hold_sec:
            if self.allow_negative_exit and neg_gate_ok:
                return "time_cut"
            _log_neg_gate_block("time_cut", pnl, neg_gate_detail)

        if pnl < 0:
            if side == "long" and ctx.rsi is not None and ctx.rsi <= self.rsi_fade_long:
                if (self.allow_negative_exit and neg_gate_ok) or atr >= self.atr_hot:
                    return "rsi_fade"
                _log_neg_gate_block("rsi_fade", pnl, neg_gate_detail)
            if side == "short" and ctx.rsi is not None and ctx.rsi >= self.rsi_fade_short:
                if (self.allow_negative_exit and neg_gate_ok) or atr >= self.atr_hot:
                    return "rsi_fade"
                _log_neg_gate_block("rsi_fade", pnl, neg_gate_detail)

        if hold_sec >= max_hold and pnl <= profit_take * 0.7:
            if pnl >= 0 or not self.allow_negative_exit or neg_gate_ok:
                return "time_stop"
            _log_neg_gate_block("time_stop", pnl, neg_gate_detail)

        if state.lock_floor is None and pnl >= lock_trigger:
            state.lock_floor = max(0.0, pnl - lock_buffer)
        if state.lock_floor is not None and pnl <= state.lock_floor:
            return "lock_release"

        if state.peak >= trail_start and pnl <= state.peak - trail_backoff:
            return "trail_take"

        if ctx.vwap_gap_pips is not None and abs(ctx.vwap_gap_pips) <= self.vwap_grab_gap:
            if pnl > 0.15:
                return "vwap_gravity"
            if pnl < 0 and self.allow_negative_exit:
                if neg_gate_ok:
                    return "vwap_cut"
                _log_neg_gate_block("vwap_cut", pnl, neg_gate_detail)

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
            "[EXIT-mirror_spike_tight] worker starting (interval=%.2fs tags=%s)",
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
                        LOG.exception("[EXIT-mirror_spike_tight] evaluate failed trade=%s", tr.get("trade_id"))
                        continue
                    if not reason:
                        continue
                    trade_id = str(tr.get("trade_id"))
                    units = int(tr.get("units", 0) or 0)
                    client_id = _client_id(tr)
                    if not client_id:
                        LOG.warning("[EXIT-mirror_spike_tight] missing client_id trade=%s skip close", trade_id)
                        continue
                    side = "long" if units > 0 else "short"
                    exit_px = None
                    if side == "long":
                        exit_px = ctx.bid if ctx.bid is not None else ctx.mid
                    else:
                        exit_px = ctx.ask if ctx.ask is not None else ctx.mid
                    if exit_px is None:
                        continue
                    entry_px = float(tr.get("price") or 0.0)
                    pnl = (exit_px - entry_px) * 100.0 if side == "long" else (entry_px - exit_px) * 100.0
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
            LOG.info("[EXIT-mirror_spike_tight] worker cancelled")
            raise
        finally:
            try:
                self._pos_manager.close()
            except Exception:
                LOG.exception("[EXIT-mirror_spike_tight] failed to close PositionManager")


async def mirror_spike_tight_exit_worker() -> None:
    worker = MirrorSpikeTightExitWorker()
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
    asyncio.run(mirror_spike_tight_exit_worker())
