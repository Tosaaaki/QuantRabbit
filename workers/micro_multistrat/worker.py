"""Micro multi-strategy worker with dynamic cap."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

from analysis.range_guard import detect_range_mode
from analysis.range_model import compute_range_snapshot
from indicators.factor_cache import all_factors, get_candles_snapshot, refresh_cache_from_disk
from execution.order_manager import market_order
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp
from market_data import tick_window
from strategies.micro.momentum_burst import MomentumBurstMicro
from strategies.micro.momentum_stack import MicroMomentumStack
from strategies.micro.pullback_ema import MicroPullbackEMA
from strategies.micro.level_reactor import MicroLevelReactor
from strategies.micro.range_break import MicroRangeBreak
from strategies.micro.vwap_bound_revert import MicroVWAPBound
from strategies.micro.trend_momentum import TrendMomentumMicro
from utils.market_hours import is_market_open
from utils.oanda_account import get_account_snapshot, get_position_summary
from utils.metrics_logger import log_metric
from workers.common.dyn_cap import compute_cap
from workers.common import perf_guard
from analysis import perf_monitor

from . import config

LOG = logging.getLogger(__name__)

MR_RANGE_LOOKBACK = 20
MR_RANGE_HI_PCT = 95.0
MR_RANGE_LO_PCT = 5.0
_STRATEGY_LAST_TS: Dict[str, float] = {}
_TREND_STRATEGIES = {
    MomentumBurstMicro.name,
    MicroMomentumStack.name,
    MicroPullbackEMA.name,
    TrendMomentumMicro.name,
}
_PULLBACK_STRATEGIES = {
    MicroPullbackEMA.name,
}
_RANGE_STRATEGIES = {
    MicroRangeBreak.name,
    MicroVWAPBound.name,
    MicroLevelReactor.name,
}


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


def _confidence_scale(conf: int) -> float:
    lo = config.CONFIDENCE_FLOOR
    hi = config.CONFIDENCE_CEIL
    if conf <= lo:
        return 0.55
    if conf >= hi:
        return 1.0
    span = (conf - lo) / max(1.0, hi - lo)
    return 0.55 + span * 0.45


def _compute_cap(*args, **kwargs) -> Tuple[float, Dict[str, float]]:
    res = compute_cap(cap_min=config.CAP_MIN, cap_max=config.CAP_MAX, *args, **kwargs)
    return res.cap, res.reasons


def _allowed_strategies() -> List:
    """
    Return strategy classes filtered by MICRO_STRATEGY_ALLOWLIST.
    Set env MICRO_STRATEGY_ALLOWLIST=\"MicroVWAPBound,TrendMomentumMicro\" to run only those.
    """
    allow_raw = (os.getenv("MICRO_STRATEGY_ALLOWLIST") or "").strip()
    all_classes = [
        MomentumBurstMicro,
        MicroMomentumStack,
        MicroPullbackEMA,
        MicroLevelReactor,
        MicroRangeBreak,
        MicroVWAPBound,
        TrendMomentumMicro,
    ]
    if not allow_raw:
        return all_classes
    allow = {s.strip() for s in allow_raw.split(",") if s.strip()}
    if not allow:
        return all_classes
    filtered = [cls for cls in all_classes if getattr(cls, "name", cls.__name__) in allow]
    if not filtered:
        LOG.warning("%s allowlist empty; using all strategies", config.LOG_PREFIX)
        return all_classes
    LOG.info(
        "%s allowlist applied: %s",
        config.LOG_PREFIX,
        ",".join(getattr(c, "name", c.__name__) for c in filtered),
    )
    return filtered


def _strategy_list() -> List:
    return _allowed_strategies()


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


def _diversity_bonus(strategy_name: str, now_ts: float) -> float:
    if not config.DIVERSITY_ENABLED:
        return 0.0
    last_ts = _STRATEGY_LAST_TS.get(strategy_name)
    if last_ts is None:
        return config.DIVERSITY_MAX_BONUS
    idle = max(0.0, now_ts - last_ts)
    if idle < config.DIVERSITY_IDLE_SEC:
        return 0.0
    scale = max(1.0, config.DIVERSITY_SCALE_SEC)
    bonus = (idle - config.DIVERSITY_IDLE_SEC) / scale * config.DIVERSITY_MAX_BONUS
    return min(config.DIVERSITY_MAX_BONUS, bonus)


async def micro_multi_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled", config.LOG_PREFIX)
        return
    LOG.info("%s worker start (interval=%.1fs)", config.LOG_PREFIX, config.LOOP_INTERVAL_SEC)
    last_trend_block_log = 0.0
    last_stale_log = 0.0
    last_perf_block_log = 0.0

    while True:
        await asyncio.sleep(config.LOOP_INTERVAL_SEC)
        now = datetime.datetime.utcnow()
        if not is_market_open(now):
            continue
        if not can_trade(config.POCKET):
            continue
        current_hour = now.hour
        now_ts = time.time()

        # 最新キャッシュに更新（他プロセスが書いた factor_cache.json を取り込む）
        try:
            refresh_cache_from_disk()
        except Exception:
            pass
        factors = all_factors()
        fac_m1 = factors.get("M1") or {}
        fac_h4 = factors.get("H4") or {}
        fac_m5 = factors.get("M5") or {}
        age_m1 = _factor_age_seconds(fac_m1)
        if age_m1 > config.MAX_FACTOR_AGE_SEC:
            # 入口を止める代わりにログだけ残して評価を継続（データ欠損で固まらないようにする）
            log_metric(
                "micro_multi_skip",
                float(age_m1),
                tags={"reason": "factor_stale_warn", "tf": "M1"},
                ts=now,
            )
            if now_ts - last_stale_log > 30.0:
                LOG.warning(
                    "%s stale factors age=%.1fs limit=%.1fs (proceeding anyway)",
                    config.LOG_PREFIX,
                    age_m1,
                    config.MAX_FACTOR_AGE_SEC,
                )
                last_stale_log = now_ts
        range_ctx = detect_range_mode(fac_m1, fac_h4)
        range_score = 0.0
        try:
            range_score = float(range_ctx.score or 0.0)
        except Exception:
            range_score = 0.0
        range_only = range_ctx.active or range_score >= config.RANGE_ONLY_SCORE
        range_bias = range_score >= config.RANGE_BIAS_SCORE
        fac_m1 = dict(fac_m1)
        fac_m1["range_active"] = bool(range_ctx.active)
        fac_m1["range_score"] = range_score
        fac_m1["range_reason"] = range_ctx.reason
        fac_m1["range_mode"] = range_ctx.mode
        perf = perf_monitor.snapshot()
        pf = None
        try:
            pf = float((perf.get(config.POCKET) or {}).get("pf"))
        except Exception:
            pf = None

        candidates: List[Tuple[float, int, Dict, str]] = []
        for strat in _strategy_list():
            strategy_name = getattr(strat, "name", strat.__name__)
            if (
                strategy_name == TrendMomentumMicro.name
                and current_hour in config.TREND_BLOCK_HOURS_UTC
            ):
                now_mono = time.monotonic()
                if now_mono - last_trend_block_log > 300.0:
                    LOG.info(
                        "%s skip %s at hour=%02d block_hours=%s",
                        config.LOG_PREFIX,
                        TrendMomentumMicro.name,
                        current_hour,
                        sorted(config.TREND_BLOCK_HOURS_UTC),
                    )
                    last_trend_block_log = now_mono
                continue
            if range_only and strategy_name not in _RANGE_STRATEGIES:
                continue
            cand = strat.check(fac_m1)
            if not cand:
                continue
            perf_decision = perf_guard.is_allowed(strategy_name, config.POCKET)
            if not perf_decision.allowed:
                now_mono = time.monotonic()
                if now_mono - last_perf_block_log > 120.0:
                    LOG.info(
                        "%s perf_block tag=%s reason=%s",
                        config.LOG_PREFIX,
                        strategy_name,
                        perf_decision.reason,
                    )
                    last_perf_block_log = now_mono
                continue
            base_conf = int(cand.get("confidence", 0) or 0)
            bonus = _diversity_bonus(strategy_name, now_ts)
            score = base_conf + bonus
            if range_bias:
                if strategy_name in _RANGE_STRATEGIES:
                    score += config.RANGE_STRATEGY_BONUS * range_score
                else:
                    score -= config.RANGE_TREND_PENALTY * range_score
            candidates.append((score, base_conf, cand, strategy_name))
        if not candidates:
            continue
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        max_signals = max(1, int(config.MAX_SIGNALS_PER_CYCLE))
        selected = candidates[:max_signals]

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
        )
        if cap <= 0.0:
            continue
        multi_scale = max(
            float(config.MULTI_SIGNAL_MIN_SCALE),
            1.0 / max(1, len(selected)),
        )

        try:
            price = float(fac_m1.get("close") or 0.0)
        except Exception:
            price = 0.0
        price = _latest_mid(price)
        long_units = 0.0
        short_units = 0.0
        try:
            long_units, short_units = get_position_summary("USD_JPY", timeout=3.0)
        except Exception:
            long_units, short_units = 0.0, 0.0
        for _, _, signal, strategy_name in selected:
            side = "long" if signal["action"] == "OPEN_LONG" else "short"
            sl_pips = float(signal.get("sl_pips") or 0.0)
            tp_pips = float(signal.get("tp_pips") or 0.0)
            if price <= 0.0 or sl_pips <= 0.0:
                continue

            tp_scale = 10.0 / max(1.0, tp_pips)
            tp_scale = max(0.4, min(1.1, tp_scale))
            base_units = int(round(config.BASE_ENTRY_UNITS * tp_scale))

            conf_scale = _confidence_scale(int(signal.get("confidence", 50)))
            lot = allowed_lot(
                float(snap.nav or 0.0),
                sl_pips,
                margin_available=float(snap.margin_available or 0.0),
                price=price,
                margin_rate=float(snap.margin_rate or 0.0),
                pocket=config.POCKET,
                side=side,
                open_long_units=long_units,
                open_short_units=short_units,
            )
            units_risk = int(round(lot * 100000))
            units = int(round(base_units * conf_scale))
            units = min(units, units_risk)
            units = int(round(units * cap))
            units = int(round(units * multi_scale))
            if units < config.MIN_UNITS:
                continue
            if side == "short":
                units = -abs(units)

            if side == "long":
                sl_price = round(price - sl_pips * 0.01, 3)
                tp_price = round(price + tp_pips * 0.01, 3) if tp_pips > 0 else None
            else:
                sl_price = round(price + sl_pips * 0.01, 3)
                tp_price = round(price - tp_pips * 0.01, 3) if tp_pips > 0 else None

            sl_price, tp_price = clamp_sl_tp(
                price=price,
                sl=sl_price,
                tp=tp_price,
                is_buy=side == "long",
            )
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
            if strategy_name in _TREND_STRATEGIES:
                entry_thesis["entry_guard_trend"] = True
                entry_thesis["entry_tf"] = "M5"
            if strategy_name in _PULLBACK_STRATEGIES:
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
                pocket=config.POCKET,
                client_order_id=client_id,
                strategy_tag=signal_tag,
                confidence=int(signal.get("confidence", 0)),
                entry_thesis=entry_thesis,
            )
            _STRATEGY_LAST_TS[strategy_name] = time.time()
            LOG.info(
                "%s strat=%s sent units=%s side=%s price=%.3f sl=%.3f tp=%.3f conf=%.0f cap=%.2f multi=%.2f reasons=%s res=%s",
                config.LOG_PREFIX,
                strategy_name,
                units,
                side,
                price,
                sl_price,
                tp_price,
                signal.get("confidence", 0),
                cap,
                multi_scale,
                {**cap_reason, "tp_scale": round(tp_scale, 3)},
                res or "none",
            )


if __name__ == "__main__":
    asyncio.run(micro_multi_worker())
