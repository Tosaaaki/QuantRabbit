from __future__ import annotations

import os
import pathlib
import sqlite3
import time
from typing import Dict, Optional

from .common import (
    atr_pips,
    candle_body_pips,
    clamp,
    latest_candles,
    price_delta_pips,
    to_float,
    typical_price,
)
from utils.tuning_loader import get_tuning_value

_HIST_ENABLED = os.getenv("MICRO_VWAPREVERT_HIST_ENABLED", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}
_HIST_DB_PATH = os.getenv("MICRO_VWAPREVERT_HIST_DB_PATH", "logs/trades.db")
_HIST_TTL_SEC = float(os.getenv("MICRO_VWAPREVERT_HIST_TTL_SEC", "30"))
_HIST_LOOKBACK_DAYS = int(os.getenv("MICRO_VWAPREVERT_HIST_LOOKBACK_DAYS", "7"))
_HIST_MIN_TRADES = int(os.getenv("MICRO_VWAPREVERT_HIST_MIN_TRADES", "12"))
_HIST_REGIME_MIN_TRADES = int(os.getenv("MICRO_VWAPREVERT_HIST_REGIME_MIN_TRADES", "8"))
_HIST_PF_CAP = float(os.getenv("MICRO_VWAPREVERT_HIST_PF_CAP", "2.0"))
_HIST_SKIP_SCORE = float(os.getenv("MICRO_VWAPREVERT_HIST_SKIP_SCORE", "0.34"))
_HIST_LOT_MIN = float(os.getenv("MICRO_VWAPREVERT_HIST_LOT_MIN", "0.72"))
_HIST_LOT_MAX = float(os.getenv("MICRO_VWAPREVERT_HIST_LOT_MAX", "1.28"))

_HISTORY_PROFILE_CACHE: Dict[tuple[str, str, str], tuple[float, Dict[str, object]]] = {}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalize_tag_key(raw: object) -> str:
    if not raw:
        return ""
    key = str(raw).strip().lower()
    if not key:
        return ""
    if "-" in key:
        key = key.split("-", 1)[0].strip()
    return key


def _normalize_regime_label(raw: object) -> str:
    if not raw:
        return ""
    return str(raw).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def _history_profile_cache_key(strategy_key: str, pocket: str, regime_label: str) -> tuple[str, str, str]:
    return (_normalize_tag_key(strategy_key), str(pocket).strip().lower(), regime_label)


def _query_strategy_history(
    *,
    strategy_key: str,
    pocket: str,
    regime_label: Optional[str],
) -> Dict[str, object]:
    strategy_key = _normalize_tag_key(strategy_key)
    if not strategy_key:
        return {"n": 0, "pf": 1.0, "win_rate": 0.0, "avg_pips": 0.0}
    db_path = pathlib.Path(_HIST_DB_PATH)
    if not db_path.exists():
        return {"n": 0, "pf": 1.0, "win_rate": 0.0, "avg_pips": 0.0}

    lookback = max(1, int(_HIST_LOOKBACK_DAYS))
    params: list[object] = [str(pocket).strip().lower(), f"-{lookback} day"]
    conditions = [
        "LOWER(pocket) = ?",
        "close_time IS NOT NULL",
        "datetime(close_time) >= datetime('now', ?)",
    ]

    pattern = f"{strategy_key}-%"
    conditions.append(
        "(LOWER(strategy) = ? OR LOWER(NULLIF(strategy_tag, '')) = ? OR LOWER(NULLIF(strategy_tag, '')) LIKE ?)"
    )
    params.extend([strategy_key, strategy_key, pattern])

    if regime_label:
        normalized_regime = _normalize_regime_label(regime_label)
        if normalized_regime:
            conditions.append(
                "(LOWER(COALESCE(micro_regime, '')) = ? OR LOWER(COALESCE(macro_regime, '')) = ?)"
            )
            params.extend([normalized_regime, normalized_regime])

    where = " AND ".join(conditions)
    con: sqlite3.Connection | None = None
    try:
        con = sqlite3.connect(str(db_path))
        con.row_factory = sqlite3.Row
        row = con.execute(
            f"""
            SELECT
              COUNT(*) AS n,
              SUM(CASE WHEN pl_pips > 0 THEN pl_pips ELSE 0 END) AS profit,
              SUM(CASE WHEN pl_pips < 0 THEN ABS(pl_pips) ELSE 0 END) AS loss,
              SUM(CASE WHEN pl_pips > 0 THEN 1 ELSE 0 END) AS win,
              SUM(pl_pips) AS sum_pips
            FROM trades
            WHERE {where}
            """,
            params,
        ).fetchone()
    except Exception:
        return {"n": 0, "pf": 1.0, "win_rate": 0.0, "avg_pips": 0.0}
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass

    if not row:
        return {"n": 0, "pf": 1.0, "win_rate": 0.0, "avg_pips": 0.0}

    n = int(row["n"] or 0)
    if n <= 0:
        return {"n": 0, "pf": 1.0, "win_rate": 0.0, "avg_pips": 0.0}

    profit = float(row["profit"] or 0.0)
    loss = float(row["loss"] or 0.0)
    win = float(row["win"] or 0.0)
    sum_pips = float(row["sum_pips"] or 0.0)
    pf = profit / loss if loss > 0 else float("inf")
    win_rate = win / n
    avg_pips = sum_pips / n
    return {"n": n, "pf": pf, "win_rate": win_rate, "avg_pips": avg_pips}


def _derive_history_score(row: Dict[str, object]) -> float:
    n = int(row.get("n", 0) or 0)
    pf = float(row.get("pf") or 1.0)
    win_rate = float(row.get("win_rate") or 0.0)
    avg_pips = float(row.get("avg_pips") or 0.0)

    if n <= 0:
        return 0.5

    if pf == float("inf"):
        pf_norm = 1.0
    else:
        pf_cap = max(1.001, float(_HIST_PF_CAP))
        pf_norm = (pf - 1.0) / (pf_cap - 1.0)
        pf_norm = _clamp01(pf_norm)

    win_norm = _clamp01(win_rate)
    avg_norm = _clamp01((avg_pips + 4.0) / 8.0)
    score = 0.40 * pf_norm + 0.45 * win_norm + 0.15 * avg_norm

    if n < _HIST_MIN_TRADES:
        weight = n / max(1.0, float(_HIST_MIN_TRADES))
        score = 0.5 + (score - 0.5) * weight
    return _clamp01(score)


def _history_profile(strategy_key: str, pocket: str, regime_label: Optional[str]) -> Dict[str, object]:
    if not _HIST_ENABLED:
        return {
            "enabled": False,
            "strategy_key": strategy_key,
            "pocket": pocket,
            "used_regime": False,
            "n": 0,
            "pf": 1.0,
            "win_rate": 0.0,
            "avg_pips": 0.0,
            "score": 0.5,
            "lot_multiplier": 1.0,
            "skip": False,
            "source": "disabled",
        }

    normalized_regime = _normalize_regime_label(regime_label)
    cache_key = _history_profile_cache_key(strategy_key, pocket, normalized_regime)
    now = time.time()
    cached = _HISTORY_PROFILE_CACHE.get(cache_key)
    if cached and (now - cached[0]) <= max(1.0, float(_HIST_TTL_SEC)):
        return dict(cached[1])

    row = _query_strategy_history(
        strategy_key=strategy_key,
        pocket=pocket,
        regime_label=normalized_regime,
    )
    used_regime = bool(normalized_regime)
    source = "regime"
    if used_regime and int(row.get("n", 0) or 0) < max(1, int(_HIST_REGIME_MIN_TRADES)):
        fallback = _query_strategy_history(strategy_key=strategy_key, pocket=pocket, regime_label=None)
        if int(fallback.get("n", 0) or 0) > 0:
            row = fallback
            used_regime = False
            source = "global"

    n = int(row.get("n", 0) or 0)
    score = _derive_history_score(row)
    lot_mult = _HIST_LOT_MIN + (_HIST_LOT_MAX - _HIST_LOT_MIN) * score
    lot_mult = max(_HIST_LOT_MIN, min(_HIST_LOT_MAX, lot_mult))
    skip = bool(n >= _HIST_MIN_TRADES and score < _HIST_SKIP_SCORE)

    pf = float(row.get("pf", 1.0))
    profile = {
        "enabled": True,
        "strategy_key": strategy_key,
        "pocket": pocket,
        "used_regime": used_regime,
        "source": source,
        "n": n,
        "pf": pf if pf != float("inf") else float(_HIST_PF_CAP),
        "win_rate": float(row.get("win_rate", 0.0)),
        "avg_pips": float(row.get("avg_pips", 0.0)),
        "score": score,
        "lot_multiplier": lot_mult,
        "skip": skip,
    }
    _HISTORY_PROFILE_CACHE[cache_key] = (now, dict(profile))
    return profile


def _tuned_float(keys: tuple[str, ...], default: float) -> float:
    raw = get_tuning_value(keys)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


class MicroVWAPRevert:
    name = "MicroVWAPRevert"
    pocket = "micro"

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        close = to_float(fac.get("close"))
        ema20 = to_float(fac.get("ema20"))
        ma10 = to_float(fac.get("ma10"), ema20)
        adx = to_float(fac.get("adx"))
        vol_5m = to_float(fac.get("vol_5m"))
        bbw = to_float(fac.get("bbw"))

        if None in (close, ema20, ma10):
            return None
        if adx is not None and adx > 24.0:
            return None
        if vol_5m is not None and vol_5m > 1.1:
            return None

        atr = atr_pips(fac)
        if atr <= 0.5 or atr > 2.8:
            return None
        if bbw is not None and bbw > 0.30:
            return None

        candles = latest_candles(fac, 9)
        if len(candles) < 4:
            return None

        typicals = [typical_price(c) for c in candles[-7:]]
        typicals = [t for t in typicals if t is not None]
        if not typicals:
            return None
        vwap = sum(typicals) / len(typicals)
        deviation = price_delta_pips(close, vwap)

        drift = price_delta_pips(ema20, ma10)
        body_bias_raw = candle_body_pips(candles[-1]) if candles else 0.0
        body_bias = float(body_bias_raw or 0.0)

        threshold = max(1.05, atr * 0.55)
        tuned_min = get_tuning_value(("strategies", "MicroVWAPRevert", "vwap_z_min"))
        if tuned_min is not None:
            try:
                threshold = max(threshold, float(tuned_min))
            except (TypeError, ValueError):
                pass
        if abs(deviation) < threshold:
            return None

        prev_close = to_float(candles[-2].get("close")) if len(candles) >= 2 else None
        if prev_close is None:
            return None
        prev_deviation = price_delta_pips(prev_close, vwap)
        retrace_min = max(
            0.0,
            _tuned_float(("strategies", "MicroVWAPRevert", "retrace_min_pips"), 0.18),
        )
        extension_mult = max(
            1.0,
            _tuned_float(("strategies", "MicroVWAPRevert", "extension_mult"), 1.08),
        )
        max_counter_drift = max(
            0.4,
            _tuned_float(("strategies", "MicroVWAPRevert", "max_counter_drift_pips"), 1.45),
        )
        confirm_body_min = max(
            0.0,
            _tuned_float(("strategies", "MicroVWAPRevert", "confirm_body_min_pips"), 0.08),
        )

        retrace_from_prev = 0.0
        if deviation <= 0.0 and prev_deviation <= 0.0:
            retrace_from_prev = deviation - prev_deviation
        elif deviation >= 0.0 and prev_deviation >= 0.0:
            retrace_from_prev = prev_deviation - deviation

        direction: Optional[str] = None
        if deviation <= -threshold:
            if prev_deviation > -(threshold * extension_mult):
                return None
            if retrace_from_prev < retrace_min:
                return None
            if body_bias < confirm_body_min:
                return None
            if drift > max_counter_drift:
                return None
            direction = "OPEN_LONG"
        elif deviation >= threshold:
            if prev_deviation < threshold * extension_mult:
                return None
            if retrace_from_prev < retrace_min:
                return None
            if body_bias > -confirm_body_min:
                return None
            if drift < -max_counter_drift:
                return None
            direction = "OPEN_SHORT"
        else:
            return None

        regime_label = _normalize_regime_label(
            fac.get("micro_regime")
            or fac.get("macro_regime")
            or fac.get("regime")
            or fac.get("range_mode")
        )
        hist_profile = _history_profile(
            strategy_key=MicroVWAPRevert.name,
            pocket=MicroVWAPRevert.pocket,
            regime_label=regime_label,
        )
        if bool(hist_profile.get("skip")):
            return None

        conf_base = 50.0
        conf_base += clamp(abs(deviation) - threshold, 0.0, 3.0) * 2.6
        conf_base += clamp(retrace_from_prev, 0.0, 2.0) * 2.0
        conf_base += clamp(max(0.0, 1.3 - (vol_5m or 1.3)), 0.0, 1.1) * 4.5
        conf_base += clamp(max(0.0, 0.9 - abs(drift)), 0.0, 0.9) * 3.2
        conf_base -= clamp(max(0.0, abs(body_bias) - 0.8), 0.0, 2.0) * 2.4
        confidence = int(clamp(conf_base, 44.0, 84.0))
        history_score = float(hist_profile.get("score", 0.5))
        confidence = int(
            clamp(
                confidence + int((history_score - 0.5) * 18.0),
                44.0,
                84.0,
            )
        )

        sl = clamp(atr * 1.18, 1.2, 2.4)
        tp = clamp(sl * 0.9, 0.9, 2.2)

        tag_suffix = "long" if direction == "OPEN_LONG" else "short"
        return {
            "action": direction,
            "sl_pips": round(sl, 2),
            "tp_pips": round(tp, 2),
            "confidence": confidence,
            "tag": f"{MicroVWAPRevert.name}-{tag_suffix}",
            "notes": {
                "deviation": round(deviation, 2),
                "threshold": round(threshold, 2),
                "vwap": round(vwap, 3),
                "atr": round(atr, 2),
                "vol5m": round(vol_5m or 0.0, 2),
                "regime": regime_label,
                "drift": round(drift, 2),
                "prev_deviation": round(prev_deviation, 2),
                "retrace": round(retrace_from_prev, 2),
                "history_score": round(float(hist_profile.get("score", 0.5)), 3),
                "history_n": int(hist_profile.get("n", 0) or 0),
                "history_source": hist_profile.get("source", "disabled"),
                "history_lot_mult": round(float(hist_profile.get("lot_multiplier", 1.0)), 3),
                "history_pf": round(float(hist_profile.get("pf", 1.0)), 3),
                "history_used_regime": bool(hist_profile.get("used_regime", False)),
            },
        }
