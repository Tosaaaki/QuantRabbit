"""Entry loop for TrendMomentumMicro (micro pocket)."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import time
from typing import Dict, Optional, Tuple

from analysis import perf_monitor
from analysis.range_guard import detect_range_mode
from analysis.range_model import compute_range_snapshot
from execution.order_manager import market_order
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp
from indicators.factor_cache import all_factors, get_candles_snapshot, refresh_cache_from_disk
from market_data import tick_window
from strategies.micro.trend_momentum import TrendMomentumMicro
from utils.market_hours import is_market_open
from utils.oanda_account import get_account_snapshot, get_position_summary
from workers.common.dyn_cap import compute_cap
from workers.common import perf_guard

from . import config

LOG = logging.getLogger(__name__)

MR_RANGE_LOOKBACK = 20
MR_RANGE_HI_PCT = 95.0
MR_RANGE_LO_PCT = 5.0

STRATEGY = TrendMomentumMicro
IS_TREND = True
IS_PULLBACK = False
IS_RANGE = False


def _latest_mid(fallback: float) -> float:
    ticks = tick_window.recent_ticks(seconds=10.0, limit=1)
    if ticks:
        tick = ticks[-1]
        mid_val = tick.get("mid")
        if mid_val is not None:
            try:
                return float(mid_val)
            except Exception:
                pass
        bid = tick.get("bid")
        ask = tick.get("ask")
        if bid is not None and ask is not None:
            try:
                return (float(bid) + float(ask)) / 2.0
            except Exception:
                return fallback
    return fallback


def _client_order_id(tag: str) -> str:
    ts_ms = int(time.time() * 1000)
    sanitized = "".join(ch.lower() for ch in tag if ch.isalnum())[:8] or "micro"
    digest = hashlib.sha1(f"{ts_ms}-{tag}".encode("utf-8")).hexdigest()[:9]
    return f"qr-{ts_ms}-micro-{sanitized}{digest}"


def _confidence_scale(conf: int, *, lo: int, hi: int) -> float:
    if conf <= lo:
        return 0.55
    if conf >= hi:
        return 1.0
    span = (conf - lo) / max(1.0, hi - lo)
    return 0.55 + span * 0.45


def _compute_cap(*, cap_min: float, cap_max: float, **kwargs) -> Tuple[float, Dict[str, float]]:
    res = compute_cap(cap_min=cap_min, cap_max=cap_max, **kwargs)
    return res.cap, res.reasons


def _is_mr_signal(tag: str) -> bool:
    tag_str = (tag or "").strip()
    if not tag_str:
        return False
    base_tag = tag_str.split("-", 1)[0]
    if base_tag in {"MicroVWAPBound", "BB_RSI"}:
        return True
    lower = tag_str.lower()
    return lower.startswith("mlr-fade") or lower.startswith("mlr-bounce")


def _build_mr_entry_thesis(
    signal: Dict,
    *,
    strategy_tag: str,
    atr_entry: float,
    entry_mean: Optional[float],
) -> Dict:
    thesis: Dict[str, object] = {
        "strategy_tag": strategy_tag,
        "profile": signal.get("profile"),
        "confidence": signal.get("confidence", 0),
        "env_tf": "H1",
        "struct_tf": "M5",
        "range_method": "percentile",
        "range_lookback": MR_RANGE_LOOKBACK,
        "range_hi_pct": MR_RANGE_HI_PCT,
        "range_lo_pct": MR_RANGE_LO_PCT,
        "atr_entry": atr_entry,
        "structure_break": {"buffer_atr": 0.10, "confirm_closes": 2},
        "tp_mode": "soft_zone",
        "tp_target": "entry_mean",
        "tp_pad_atr": 0.05,
        "reversion_failure": {
            "z_ext": 0.55,
            "contraction_min": 0.50,
            "bars_budget": {"k_per_z": 3.5, "min": 2, "max": 12},
            "trend_takeover": {"require_env_trend_bars": 2},
        },
    }
    if entry_mean is not None and entry_mean > 0:
        thesis["entry_mean"] = float(entry_mean)
    candles = get_candles_snapshot("H1", limit=MR_RANGE_LOOKBACK)
    snapshot = compute_range_snapshot(
        candles,
        lookback=MR_RANGE_LOOKBACK,
        method="percentile",
        hi_pct=MR_RANGE_HI_PCT,
        lo_pct=MR_RANGE_LO_PCT,
    )
    if snapshot:
        thesis["range_snapshot"] = snapshot.to_dict()
        thesis.setdefault("entry_mean", snapshot.mid)
    return thesis


def _factor_age_seconds(factors: Dict[str, float]) -> float:
    ts_raw = factors.get("timestamp") if isinstance(factors, dict) else None
    if not ts_raw:
        return float("inf")
    try:
        if isinstance(ts_raw, (int, float)):
            ts_val = float(ts_raw)
            ts_dt = datetime.datetime.utcfromtimestamp(ts_val).replace(tzinfo=datetime.timezone.utc)
        else:
            ts_txt = str(ts_raw)
            if ts_txt.endswith("Z"):
                ts_txt = ts_txt.replace("Z", "+00:00")
            ts_dt = datetime.datetime.fromisoformat(ts_txt)
            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=datetime.timezone.utc)
    except Exception:
        return float("inf")
    now = datetime.datetime.now(datetime.timezone.utc)
    return max(0.0, (now - ts_dt).total_seconds())


async def micro_trendmomentum_worker() -> None:
    enabled = getattr(config, "ENABLED", True)
    strategy_name = getattr(STRATEGY, "name", STRATEGY.__name__)
    log_prefix = getattr(config, "LOG_PREFIX", f"[{strategy_name}]")
    if not enabled:
        LOG.info("%s disabled", log_prefix)
        return
    loop_interval = float(getattr(config, "LOOP_INTERVAL_SEC", 8.0))
    pocket = getattr(config, "POCKET", "micro")
    min_units = int(getattr(config, "MIN_UNITS", 1500))
    base_units_cfg = int(getattr(config, "BASE_ENTRY_UNITS", 20000))
    conf_floor = int(getattr(config, "CONFIDENCE_FLOOR", 35))
    conf_ceil = int(getattr(config, "CONFIDENCE_CEIL", 90))
    cap_min = float(getattr(config, "CAP_MIN", 0.15))
    cap_max = float(getattr(config, "CAP_MAX", 0.95))
    range_only_score = float(getattr(config, "RANGE_ONLY_SCORE", 0.45))
    max_factor_age = float(getattr(config, "MAX_FACTOR_AGE_SEC", 90.0))
    trend_block_hours = getattr(config, "TREND_BLOCK_HOURS_UTC", frozenset())

    LOG.info("%s worker start (interval=%.1fs)", log_prefix, loop_interval)
    last_stale_log = 0.0
    last_perf_block_log = 0.0

    while True:
        await asyncio.sleep(loop_interval)
        now = datetime.datetime.utcnow()
        if not is_market_open(now):
            continue
        if not can_trade(pocket):
            continue
        if IS_TREND and trend_block_hours and now.hour in trend_block_hours:
            continue

        try:
            refresh_cache_from_disk()
        except Exception:
            pass
        factors = all_factors()
        fac_m1 = factors.get("M1") or {}
        fac_h4 = factors.get("H4") or {}
        fac_m5 = factors.get("M5") or {}
        age_m1 = _factor_age_seconds(fac_m1)
        if age_m1 > max_factor_age:
            if time.time() - last_stale_log > 30.0:
                LOG.warning(
                    "%s stale factors age=%.1fs limit=%.1fs (proceeding)",
                    log_prefix,
                    age_m1,
                    max_factor_age,
                )
                last_stale_log = time.time()

        range_ctx = detect_range_mode(fac_m1, fac_h4)
        range_score = 0.0
        try:
            range_score = float(range_ctx.score or 0.0)
        except Exception:
            range_score = 0.0
        range_only = range_ctx.active or range_score >= range_only_score
        if range_only and not IS_RANGE:
            continue
        fac_m1 = dict(fac_m1)
        fac_m1["range_active"] = bool(range_ctx.active)
        fac_m1["range_score"] = range_score
        fac_m1["range_reason"] = range_ctx.reason
        fac_m1["range_mode"] = range_ctx.mode

        signal = STRATEGY.check(fac_m1)
        if not signal:
            continue
        perf_decision = perf_guard.is_allowed(strategy_name, pocket)
        if not perf_decision.allowed:
            now_mono = time.monotonic()
            if now_mono - last_perf_block_log > 120.0:
                LOG.info(
                    "%s perf_block tag=%s reason=%s",
                    log_prefix,
                    strategy_name,
                    perf_decision.reason,
                )
                last_perf_block_log = now_mono
            continue

        perf = perf_monitor.snapshot()
        pf = None
        try:
            pf = float((perf.get(pocket) or {}).get("pf"))
        except Exception:
            pf = None

        snap = get_account_snapshot()
        free_ratio = float(snap.free_margin_ratio or 0.0) if snap.free_margin_ratio is not None else 0.0
        try:
            atr_pips = float(fac_m1.get("atr_pips") or 0.0)
        except Exception:
            atr_pips = 0.0
        try:
            atr_m5 = float(fac_m5.get("atr_pips") or 0.0)
        except Exception:
            atr_m5 = 0.0
        pos_bias = 0.0
        try:
            open_positions = snap.positions or {}
            micro_pos = open_positions.get("micro") or {}
            pos_bias = abs(float(micro_pos.get("units", 0.0) or 0.0)) / max(1.0, float(snap.nav or 1.0))
        except Exception:
            pos_bias = 0.0

        cap, cap_reason = _compute_cap(
            atr_pips=atr_pips,
            free_ratio=free_ratio,
            range_active=range_ctx.active,
            perf_pf=pf,
            pos_bias=pos_bias,
            cap_min=cap_min,
            cap_max=cap_max,
        )
        if cap <= 0.0:
            continue

        try:
            price = float(fac_m1.get("close") or 0.0)
        except Exception:
            price = 0.0
        price = _latest_mid(price)
        if price <= 0.0:
            continue

        long_units = 0.0
        short_units = 0.0
        try:
            long_units, short_units = get_position_summary("USD_JPY", timeout=3.0)
        except Exception:
            long_units, short_units = 0.0, 0.0

        side = "long" if signal.get("action") == "OPEN_LONG" else "short"
        sl_pips = float(signal.get("sl_pips") or 0.0)
        tp_pips = float(signal.get("tp_pips") or 0.0)
        if sl_pips <= 0.0:
            continue

        tp_scale = 10.0 / max(1.0, tp_pips)
        tp_scale = max(0.4, min(1.1, tp_scale))
        base_units = int(round(base_units_cfg * tp_scale))
        conf_scale = _confidence_scale(int(signal.get("confidence", 50)), lo=conf_floor, hi=conf_ceil)
        lot = allowed_lot(
            float(snap.nav or 0.0),
            sl_pips,
            margin_available=float(snap.margin_available or 0.0),
            price=price,
            margin_rate=float(snap.margin_rate or 0.0),
            pocket=pocket,
            side=side,
            open_long_units=long_units,
            open_short_units=short_units,
            strategy_tag=signal.get("tag"),
            fac_m1=fac_m1,
            fac_h4=fac_h4,
        )
        units_risk = int(round(lot * 100000))
        units = int(round(base_units * conf_scale))
        units = min(units, units_risk)
        units = int(round(units * cap))
        if units < min_units:
            continue
        if side == "short":
            units = -abs(units)

        if side == "long":
            sl_price = round(price - sl_pips * 0.01, 3)
            tp_price = round(price + tp_pips * 0.01, 3) if tp_pips > 0 else None
        else:
            sl_price = round(price + sl_pips * 0.01, 3)
            tp_price = round(price - tp_pips * 0.01, 3) if tp_pips > 0 else None
        sl_price, tp_price = clamp_sl_tp(price=price, sl=sl_price, tp=tp_price, is_buy=side == "long")

        signal_tag = signal.get("tag", strategy_name)
        client_id = _client_order_id(signal_tag)
        entry_thesis: Dict[str, object] = {
            "strategy_tag": signal_tag,
            "profile": signal.get("profile"),
            "confidence": signal.get("confidence", 0),
            "tp_pips": tp_pips,
            "sl_pips": sl_pips,
            "hard_stop_pips": sl_pips,
            "range_active": bool(range_ctx.active),
            "range_score": round(range_score, 3),
            "range_reason": range_ctx.reason,
            "range_mode": range_ctx.mode,
        }
        if IS_TREND:
            entry_thesis["entry_guard_trend"] = True
            entry_thesis["entry_tf"] = "M5"
        if IS_PULLBACK:
            entry_thesis["entry_guard_pullback"] = True
            entry_thesis["entry_guard_pullback_only"] = True
        if _is_mr_signal(signal_tag):
            entry_thesis["entry_guard_trend"] = False
            entry_mean = None
            base_tag = signal_tag.split("-", 1)[0] if signal_tag else ""
            if base_tag == "MicroVWAPBound":
                notes = signal.get("notes") or {}
                if isinstance(notes, dict):
                    try:
                        entry_mean = float(notes.get("vwap"))
                    except Exception:
                        entry_mean = None
            elif base_tag == "BB_RSI":
                try:
                    entry_mean = float(fac_m1.get("ma20") or fac_m1.get("ma10") or 0.0) or None
                except Exception:
                    entry_mean = None
            entry_thesis.update(
                _build_mr_entry_thesis(
                    signal,
                    strategy_tag=signal_tag,
                    atr_entry=atr_m5 or atr_pips or 1.0,
                    entry_mean=entry_mean,
                )
            )
            if base_tag == "MicroVWAPBound":
                notes = signal.get("notes") or {}
                z_val = None
                if isinstance(notes, dict):
                    try:
                        z_val = abs(float(notes.get("z")))
                    except Exception:
                        z_val = None
                rf = entry_thesis.get("reversion_failure")
                if isinstance(rf, dict) and z_val is not None:
                    bars_budget = rf.get("bars_budget")
                    if not isinstance(bars_budget, dict):
                        bars_budget = {}
                        rf["bars_budget"] = bars_budget
                    if z_val >= 2.5:
                        bars_budget["k_per_z"] = 4.0
                        bars_budget["max"] = 14
                    elif z_val <= 1.4:
                        bars_budget["k_per_z"] = 3.0
                        bars_budget["max"] = 10

        res = await market_order(
            instrument="USD_JPY",
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            pocket=pocket,
            client_order_id=client_id,
            strategy_tag=signal_tag,
            confidence=int(signal.get("confidence", 0)),
            entry_thesis=entry_thesis,
        )
        LOG.info(
            "%s sent units=%s side=%s price=%.3f sl=%.3f tp=%.3f conf=%.0f cap=%.2f reasons=%s res=%s",
            log_prefix,
            units,
            side,
            price,
            sl_price or 0.0,
            tp_price or 0.0,
            signal.get("confidence", 0),
            cap,
            cap_reason,
            res or "none",
        )


if __name__ == "__main__":
    asyncio.run(micro_trendmomentum_worker())
