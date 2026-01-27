"""BB_RSI dedicated micro worker with dynamic cap."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import time
from typing import Dict, Optional, Tuple

from analysis.range_guard import detect_range_mode
from analysis.range_model import compute_range_snapshot
from indicators.factor_cache import all_factors, get_candles_snapshot
from execution.order_manager import market_order
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp
from market_data import tick_window
from strategies.mean_reversion.bb_rsi import BBRsi
from utils.market_hours import is_market_open
from utils.oanda_account import get_account_snapshot
from workers.common.dyn_cap import compute_cap
from analysis import perf_monitor

from . import config

LOG = logging.getLogger(__name__)

MR_RANGE_LOOKBACK = 20
MR_RANGE_HI_PCT = 95.0
MR_RANGE_LO_PCT = 5.0


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
    sanitized = "".join(ch.lower() for ch in tag if ch.isalnum())[:8] or "bbrsi"
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


def _build_entry_thesis(signal: Dict, *, atr_entry: float, entry_mean: Optional[float]) -> Dict:
    tag = signal.get("tag", BBRsi.name)
    thesis: Dict[str, object] = {
        "strategy_tag": tag,
        "profile": signal.get("profile"),
        "confidence": signal.get("confidence", 0),
        "tp_pips": signal.get("tp_pips"),
        "sl_pips": signal.get("sl_pips"),
        "hard_stop_pips": signal.get("sl_pips"),
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


async def micro_bbrsi_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled", config.LOG_PREFIX)
        return
    LOG.info("%s worker start (interval=%.1fs)", config.LOG_PREFIX, config.LOOP_INTERVAL_SEC)

    while True:
        await asyncio.sleep(config.LOOP_INTERVAL_SEC)
        now = datetime.datetime.utcnow()
        if not is_market_open(now):
            continue
        if not can_trade(config.POCKET):
            continue

        factors = all_factors()
        fac_m1 = factors.get("M1") or {}
        fac_h4 = factors.get("H4") or {}
        fac_m5 = factors.get("M5") or {}
        range_ctx = detect_range_mode(fac_m1, fac_h4)
        range_score = 0.0
        try:
            range_score = float(range_ctx.score or 0.0)
        except Exception:
            range_score = 0.0
        if not range_ctx.active and range_score < config.RANGE_ONLY_SCORE:
            continue
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

        signal = BBRsi.check(fac_m1)
        if not signal:
            continue

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

        try:
            price = float(fac_m1.get("close") or 0.0)
        except Exception:
            price = 0.0
        price = _latest_mid(price)
        side = "long" if signal["action"] == "OPEN_LONG" else "short"
        sl_pips = float(signal.get("sl_pips") or 0.0)
        tp_pips = float(signal.get("tp_pips") or 0.0)
        if price <= 0.0 or sl_pips <= 0.0:
            continue

        tp_scale = 8.0 / max(1.0, tp_pips)
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
            strategy_tag=signal.get("tag"),
            fac_m1=fac_m1,
            fac_h4=fac_h4,
        )
        units_risk = int(round(lot * 100000))
        units = int(round(base_units * conf_scale))
        units = min(units, units_risk)
        units = int(round(units * cap))
        if units < config.MIN_UNITS:
            continue
        if side == "short":
            units = -abs(units)

        sl_price, tp_price = clamp_sl_tp(
            price=price,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            side="BUY" if side == "long" else "SELL",
        )
        client_id = _client_order_id(signal.get("tag", BBRsi.name))

        entry_mean = None
        try:
            entry_mean = float(fac_m1.get("ma20") or fac_m1.get("ma10") or 0.0) or None
        except Exception:
            entry_mean = None
        entry_thesis = _build_entry_thesis(
            signal,
            atr_entry=atr_m5 or atr_pips or 1.0,
            entry_mean=entry_mean,
        )
        trend_bias = bool(signal.get("trend_bias"))
        if trend_bias or (signal.get("trend_score") or 0.0) > 0.0:
            entry_thesis["trend_bias"] = True
            entry_thesis["trend_score"] = signal.get("trend_score")
            rf = entry_thesis.get("reversion_failure")
            if isinstance(rf, dict):
                rf["z_ext"] = 0.45
                rf["contraction_min"] = 0.60
                bars_budget = rf.get("bars_budget")
                if isinstance(bars_budget, dict):
                    bars_budget["k_per_z"] = 2.5
                    bars_budget["min"] = 2
                    bars_budget["max"] = 8
        res = await market_order(
            instrument="USD_JPY",
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            pocket=config.POCKET,
            client_order_id=client_id,
            strategy_tag=signal.get("tag", BBRsi.name),
            entry_thesis=entry_thesis,
        )
        LOG.info(
            "%s sent units=%s side=%s price=%.3f sl=%.3f tp=%.3f conf=%.0f cap=%.2f reasons=%s res=%s",
            config.LOG_PREFIX,
            units,
            side,
            price,
            sl_price,
            tp_price,
            signal.get("confidence", 0),
            cap,
            {**cap_reason, "tp_scale": round(tp_scale, 3)},
            res if res else "none",
        )


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    LOG.info("%s worker starting", config.LOG_PREFIX)
    asyncio.run(micro_bbrsi_worker())
