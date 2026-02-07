from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from execution.order_manager import close_trade as _close_trade
from indicators.factor_cache import all_factors
from utils.env_utils import env_bool, env_float, env_get
from utils.metrics_logger import log_metric
from workers.common.exit_emergency import should_allow_negative_close
try:  # optional in offline/backtest
    from market_data import tick_window
except Exception:  # pragma: no cover
    tick_window = None

@dataclass(frozen=True, slots=True)
class ExitCompositeCfg:
    enabled: bool
    reasons: set[str]
    min_score: float
    failopen: bool
    rsi_fade_long: float
    rsi_fade_short: float
    vwap_gap_pips: float
    structure_adx: float
    structure_gap_pips: float
    atr_spike_pips: float


_EXIT_COMPOSITE_CFG_CACHE: dict[str, ExitCompositeCfg] = {}


def _env_bool_truey(name: str, default: bool, *, env_prefix: str | None = None) -> bool:
    raw = env_get(name, None, prefix=env_prefix)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes"}


def _get_exit_composite_cfg(env_prefix: str | None) -> ExitCompositeCfg:
    key = str(env_prefix or "").strip()
    cached = _EXIT_COMPOSITE_CFG_CACHE.get(key)
    if cached is not None:
        return cached

    reasons_raw = env_get(
        "EXIT_COMPOSITE_REASONS",
        "rsi_fade,vwap_cut,atr_spike,structure_break,tech_hard_stop",
        prefix=env_prefix,
    )
    reasons = {
        token.strip().lower()
        for token in str(reasons_raw or "").split(",")
        if token.strip()
    }
    cfg = ExitCompositeCfg(
        enabled=env_bool("EXIT_COMPOSITE_ENABLED", True, prefix=env_prefix),
        reasons=reasons,
        min_score=env_float("EXIT_COMPOSITE_MIN_SCORE", 3.0, prefix=env_prefix),
        failopen=_env_bool_truey("EXIT_COMPOSITE_FAILOPEN", False, env_prefix=env_prefix),
        rsi_fade_long=env_float("EXIT_COMPOSITE_RSI_FADE_LONG", 44.0, prefix=env_prefix),
        rsi_fade_short=env_float("EXIT_COMPOSITE_RSI_FADE_SHORT", 56.0, prefix=env_prefix),
        vwap_gap_pips=env_float("EXIT_COMPOSITE_VWAP_GAP_PIPS", 0.8, prefix=env_prefix),
        structure_adx=env_float("EXIT_COMPOSITE_STRUCTURE_ADX", 20.0, prefix=env_prefix),
        structure_gap_pips=env_float("EXIT_COMPOSITE_STRUCTURE_GAP_PIPS", 1.8, prefix=env_prefix),
        atr_spike_pips=env_float("EXIT_COMPOSITE_ATR_SPIKE_PIPS", 5.0, prefix=env_prefix),
    )
    _EXIT_COMPOSITE_CFG_CACHE[key] = cfg
    return cfg

_LAST_COMPOSITE_LOG_TS = 0.0


def _latest_bid_ask_mid() -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if tick_window is None:
        return None, None, None
    try:
        ticks = tick_window.recent_ticks(seconds=3.0, limit=1)
    except Exception:
        return None, None, None
    if not ticks:
        return None, None, None
    tick = ticks[-1]
    bid = _safe_float(tick.get("bid"))
    ask = _safe_float(tick.get("ask"))
    mid = _safe_float(tick.get("mid"))
    if mid is None and bid is not None and ask is not None:
        mid = (bid + ask) / 2.0
    return bid, ask, mid


def mark_pnl_pips(entry_price: float, units: int, *, mid: Optional[float] = None) -> Optional[float]:
    if entry_price <= 0 or units == 0:
        return None
    bid, ask, latest_mid = _latest_bid_ask_mid()
    if mid is None:
        mid = latest_mid
    if units > 0:
        price = bid if bid is not None else mid
        if price is None:
            return None
        return (price - entry_price) / 0.01
    price = ask if ask is not None else mid
    if price is None:
        return None
    return (entry_price - price) / 0.01


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _reason_matches_composite(reason_key: str, reasons: set[str]) -> bool:
    for token in reasons:
        t = str(token).strip().lower()
        if not t:
            continue
        if t.endswith("*") and reason_key.startswith(t[:-1]):
            return True
        if t.startswith("*") and reason_key.endswith(t[1:]):
            return True
        if reason_key == t:
            return True
        if reason_key.endswith(t):
            return True
    return False


def _composite_exit_allowed(
    *,
    reason: str | None,
    close_units: int | None,
    cfg: ExitCompositeCfg,
) -> bool:
    if not cfg.enabled:
        return True
    if not reason:
        return True
    reason_key = str(reason).strip().lower()
    if not _reason_matches_composite(reason_key, cfg.reasons):
        return True
    if close_units is None or close_units == 0:
        return cfg.failopen
    side = "long" if close_units < 0 else "short"
    try:
        fac = (all_factors().get("M1") or {})
    except Exception:
        fac = {}
    if not fac:
        return cfg.failopen

    rsi = _safe_float(fac.get("rsi"))
    adx = _safe_float(fac.get("adx"))
    vwap_gap = _safe_float(fac.get("vwap_gap"))
    atr_pips = _safe_float(fac.get("atr_pips"))
    if atr_pips is None:
        atr = _safe_float(fac.get("atr"))
        if atr is not None:
            atr_pips = atr * 100.0
    ma10 = _safe_float(fac.get("ma10"))
    ma20 = _safe_float(fac.get("ma20"))

    rsi_fade = False
    if rsi is not None:
        rsi_fade = (side == "long" and rsi <= cfg.rsi_fade_long) or (
            side == "short" and rsi >= cfg.rsi_fade_short
        )

    vwap_cut = vwap_gap is not None and abs(vwap_gap) <= cfg.vwap_gap_pips

    atr_spike = atr_pips is not None and atr_pips >= cfg.atr_spike_pips

    structure_break = False
    if adx is not None and ma10 is not None and ma20 is not None:
        gap = abs(ma10 - ma20) / 0.01
        cross_bad = (side == "long" and ma10 <= ma20) or (side == "short" and ma10 >= ma20)
        structure_break = adx <= cfg.structure_adx and (cross_bad or gap <= cfg.structure_gap_pips)

    score = 0
    if rsi_fade:
        score += 1
    if vwap_cut:
        score += 1
    if atr_spike:
        score += 1
    if structure_break:
        score += 2

    ok = score >= cfg.min_score
    if not ok:
        global _LAST_COMPOSITE_LOG_TS
        now = time.monotonic()
        if now - _LAST_COMPOSITE_LOG_TS >= 5.0:
            log_metric(
                "exit_composite_block",
                float(score),
                tags={
                    "reason": reason_key,
                    "side": side,
                    "min_score": cfg.min_score,
                },
            )
            _LAST_COMPOSITE_LOG_TS = now
    return ok


async def close_trade(
    trade_id: str,
    units: int | None = None,
    client_order_id: str | None = None,
    allow_negative: bool = False,
    exit_reason: str | None = None,
    env_prefix: str | None = None,
) -> bool:
    if not allow_negative:
        allow_negative = should_allow_negative_close()
    if allow_negative and not _composite_exit_allowed(
        reason=exit_reason,
        close_units=units,
        cfg=_get_exit_composite_cfg(env_prefix),
    ):
        return False
    return await _close_trade(
        trade_id,
        units,
        client_order_id=client_order_id,
        allow_negative=allow_negative,
        exit_reason=exit_reason,
    )
