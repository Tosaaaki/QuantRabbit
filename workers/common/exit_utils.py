from __future__ import annotations

import os
import time

from execution.order_manager import close_trade as _close_trade
from indicators.factor_cache import all_factors
from utils.metrics_logger import log_metric
from workers.common.exit_emergency import should_allow_negative_close

_EXIT_COMPOSITE_ENABLED = os.getenv("EXIT_COMPOSITE_ENABLED", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}
_EXIT_COMPOSITE_REASONS = {
    token.strip().lower()
    for token in os.getenv(
        "EXIT_COMPOSITE_REASONS",
        "rsi_fade,vwap_cut,atr_spike",
    ).split(",")
    if token.strip()
}
_EXIT_COMPOSITE_MIN_SCORE = float(os.getenv("EXIT_COMPOSITE_MIN_SCORE", "2"))
_EXIT_COMPOSITE_FAILOPEN = os.getenv("EXIT_COMPOSITE_FAILOPEN", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}

_EXIT_RSI_FADE_LONG = float(os.getenv("EXIT_COMPOSITE_RSI_FADE_LONG", "44.0"))
_EXIT_RSI_FADE_SHORT = float(os.getenv("EXIT_COMPOSITE_RSI_FADE_SHORT", "56.0"))
_EXIT_VWAP_GAP_PIPS = float(os.getenv("EXIT_COMPOSITE_VWAP_GAP_PIPS", "0.8"))
_EXIT_STRUCTURE_ADX = float(os.getenv("EXIT_COMPOSITE_STRUCTURE_ADX", "20.0"))
_EXIT_STRUCTURE_GAP_PIPS = float(os.getenv("EXIT_COMPOSITE_STRUCTURE_GAP_PIPS", "1.8"))
_EXIT_ATR_SPIKE_PIPS = float(os.getenv("EXIT_COMPOSITE_ATR_SPIKE_PIPS", "5.0"))

_LAST_COMPOSITE_LOG_TS = 0.0


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _composite_exit_allowed(
    *,
    reason: str | None,
    close_units: int | None,
) -> bool:
    if not _EXIT_COMPOSITE_ENABLED:
        return True
    if not reason:
        return True
    reason_key = str(reason).strip().lower()
    if reason_key not in _EXIT_COMPOSITE_REASONS:
        return True
    if close_units is None or close_units == 0:
        return _EXIT_COMPOSITE_FAILOPEN
    side = "long" if close_units < 0 else "short"
    try:
        fac = (all_factors().get("M1") or {})
    except Exception:
        fac = {}
    if not fac:
        return _EXIT_COMPOSITE_FAILOPEN

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
        rsi_fade = (side == "long" and rsi <= _EXIT_RSI_FADE_LONG) or (
            side == "short" and rsi >= _EXIT_RSI_FADE_SHORT
        )

    vwap_cut = vwap_gap is not None and abs(vwap_gap) <= _EXIT_VWAP_GAP_PIPS

    atr_spike = atr_pips is not None and atr_pips >= _EXIT_ATR_SPIKE_PIPS

    structure_break = False
    if adx is not None and ma10 is not None and ma20 is not None:
        gap = abs(ma10 - ma20) / 0.01
        cross_bad = (side == "long" and ma10 <= ma20) or (side == "short" and ma10 >= ma20)
        structure_break = adx <= _EXIT_STRUCTURE_ADX and (cross_bad or gap <= _EXIT_STRUCTURE_GAP_PIPS)

    score = 0
    if rsi_fade:
        score += 1
    if vwap_cut:
        score += 1
    if atr_spike:
        score += 1
    if structure_break:
        score += 2

    ok = score >= _EXIT_COMPOSITE_MIN_SCORE
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
                    "min_score": _EXIT_COMPOSITE_MIN_SCORE,
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
) -> bool:
    if not allow_negative:
        allow_negative = should_allow_negative_close()
    if allow_negative and not _composite_exit_allowed(reason=exit_reason, close_units=units):
        return False
    return await _close_trade(
        trade_id,
        units,
        client_order_id=client_order_id,
        allow_negative=allow_negative,
        exit_reason=exit_reason,
    )
