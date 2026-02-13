from __future__ import annotations

import pathlib
import sqlite3
import time
from typing import Dict, Optional

from .common import PIP, atr_pips, candle_body_pips, clamp, latest_candles, price_delta_pips, to_float
from utils.tuning_loader import get_tuning_value

_HIST_ENABLED = True
_HIST_DB_PATH = "logs/trades.db"
_HIST_LOOKBACK_DAYS = 10
_HIST_MIN_TRADES = 10
_HIST_REGIME_MIN_TRADES = 7
_HIST_TTL_SEC = 45.0
_HIST_PF_CAP = 2.1
_HIST_SKIP_SCORE = 0.35
_HIST_LOT_MIN = 0.72
_HIST_LOT_MAX = 1.28
_HIST_PROFILE_CACHE: dict[tuple[str, str, str], tuple[float, Dict[str, object]]] = {}


def _tuned_float(keys: tuple[str, ...], default: float) -> float:
    value = get_tuning_value(keys)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


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
    *, strategy_key: str, pocket: str, regime_label: Optional[str]
) -> Dict[str, object]:
    strategy_key = _normalize_tag_key(strategy_key)
    if not strategy_key:
        return {"n": 0, "pf": 1.0, "win_rate": 0.0, "avg_pips": 0.0}

    db_path = pathlib.Path(_HIST_DB_PATH)
    if not db_path.exists():
        return {"n": 0, "pf": 1.0, "win_rate": 0.0, "avg_pips": 0.0}

    params: list[object] = [str(pocket).strip().lower(), f"-{int(_HIST_LOOKBACK_DAYS)} day"]
    pattern = f"{strategy_key}-%"
    conditions = [
        "LOWER(pocket) = ?",
        "close_time IS NOT NULL",
        "datetime(close_time) >= datetime('now', ?)",
        "(LOWER(strategy) = ? OR LOWER(NULLIF(strategy_tag, '')) = ? OR LOWER(NULLIF(strategy_tag, '')) LIKE ?)",
    ]
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

    score = 0.45 * pf_norm + 0.42 * win_norm + 0.13 * avg_norm

    if n < _HIST_MIN_TRADES:
        score = 0.5 + (score - 0.5) * max(0.0, min(1.0, n / float(_HIST_MIN_TRADES)))
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
    cached = _HIST_PROFILE_CACHE.get(cache_key)
    if cached is not None:
        cached_ts, cached_profile = cached
        if now - cached_ts <= max(1.0, float(_HIST_TTL_SEC)):
            return dict(cached_profile)

    row = _query_strategy_history(
        strategy_key=strategy_key,
        pocket=pocket,
        regime_label=normalized_regime,
    )
    used_regime = bool(normalized_regime)
    source = "regime"

    if used_regime and int(row.get("n", 0) or 0) < max(1, _HIST_REGIME_MIN_TRADES):
        fallback = _query_strategy_history(strategy_key=strategy_key, pocket=pocket, regime_label=None)
        if int(fallback.get("n", 0) or 0) > 0:
            row = fallback
            used_regime = False
            source = "global"

    n = int(row.get("n", 0) or 0)
    score = _derive_history_score(row)
    lot_multiplier = _HIST_LOT_MIN + (_HIST_LOT_MAX - _HIST_LOT_MIN) * score
    lot_multiplier = max(_HIST_LOT_MIN, min(_HIST_LOT_MAX, lot_multiplier))

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
        "lot_multiplier": lot_multiplier,
        "skip": bool(n >= _HIST_MIN_TRADES and score < _HIST_SKIP_SCORE),
    }
    _HIST_PROFILE_CACHE[cache_key] = (now, dict(profile))
    return profile


def _bb_levels(fac: Dict[str, object]) -> Optional[tuple[float, float, float, float]]:
    upper = to_float(fac.get("bb_upper"))
    lower = to_float(fac.get("bb_lower"))
    mid = to_float(fac.get("bb_mid"))
    bbw = to_float(fac.get("bbw"))
    if mid is None:
        mid = to_float(fac.get("ma20"))
    if upper is None or lower is None:
        if mid is None or bbw is None or bbw <= 0:
            return None
        half = abs(mid) * bbw / 2.0
        upper = mid + half
        lower = mid - half
    span = upper - lower
    if span <= 0:
        return None
    span_pips = span / PIP
    return upper, mid if mid is not None else (upper + lower) / 2.0, lower, span_pips


class MicroAdaptiveRevert:
    """Micro adaptive reversion strategy using factor trend/vol and historical profile."""

    name = "MicroAdaptiveRevert"
    pocket = "micro"

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        close = to_float(fac.get("close"))
        rsi = to_float(fac.get("rsi"))
        stoch = to_float(fac.get("stoch_rsi"))
        adx = to_float(fac.get("adx"))
        vwap_gap = to_float(fac.get("vwap_gap"), 0.0) or 0.0
        bbw = to_float(fac.get("bbw"))

        if close is None or rsi is None or stoch is None:
            return None

        atr = atr_pips(fac)
        if atr <= 0:
            return None

        levels = _bb_levels(fac)
        if not levels:
            return None
        upper, mid, lower, span_pips = levels

        bb_ratio = _tuned_float(("strategies", "MicroAdaptiveRevert", "band_ratio"), 0.22)
        bb_min = _tuned_float(("strategies", "MicroAdaptiveRevert", "band_min_pips"), 1.0)
        band_touch = max(bb_min, span_pips * bb_ratio)
        if span_pips < _tuned_float(("strategies", "MicroAdaptiveRevert", "span_min_pips"), 1.5):
            return None

        if bbw is None or bbw > _tuned_float(("strategies", "MicroAdaptiveRevert", "bbw_max"), 0.36):
            return None

        if adx is not None and adx > _tuned_float(("strategies", "MicroAdaptiveRevert", "adx_max"), 33.0):
            return None

        candles = latest_candles(fac, 9)
        if len(candles) < 5:
            return None

        typicals = [to_float(c.get("close")) for c in candles[-7:]]
        typicals = [t for t in typicals if t is not None]
        if not typicals:
            return None
        vwap = sum(typicals) / len(typicals)
        dev = price_delta_pips(close, vwap)

        prev_close = to_float(candles[-2].get("close"))
        if prev_close is None:
            return None
        prev_dev = price_delta_pips(prev_close, vwap)

        body = candle_body_pips(candles[-1]) or 0.0
        vol = to_float(fac.get("vol_5m"))
        if vol is not None and vol > _tuned_float(("strategies", "MicroAdaptiveRevert", "vol_5m_max"), 2.0):
            return None

        ema_slope = to_float(fac.get("ema_slope_10"), 0.0) or 0.0
        ema_slope_max = _tuned_float(("strategies", "MicroAdaptiveRevert", "ema_slope_max_pip"), 0.55)
        if abs(ema_slope) / PIP > ema_slope_max:
            return None

        dist_low = max(0.0, (close - lower) / PIP)
        dist_high = max(0.0, (upper - close) / PIP)

        threshold = max(1.05, atr * _tuned_float(("strategies", "MicroAdaptiveRevert", "atr_dev_mult"), 0.55))
        if abs(dev) < threshold:
            return None

        # touch-based reversion condition
        long_touch = dist_low <= band_touch
        short_touch = dist_high <= band_touch
        if not long_touch and not short_touch:
            return None

        retrace = abs(dev - prev_dev)
        if retrace < _tuned_float(("strategies", "MicroAdaptiveRevert", "retrace_min_pips"), 0.12):
            return None

        action: Optional[str] = None
        if long_touch and dev <= -threshold and (rsi <= _tuned_float(("strategies", "MicroAdaptiveRevert", "rsi_long"), 34.0)) and stoch <= _tuned_float(("strategies", "MicroAdaptiveRevert", "stoch_long"), 0.22):
            if body > 0:
                return None
            action = "OPEN_LONG"
        elif short_touch and dev >= threshold and (rsi >= _tuned_float(("strategies", "MicroAdaptiveRevert", "rsi_short"), 66.0) and stoch >= _tuned_float(("strategies", "MicroAdaptiveRevert", "stoch_short"), 0.78)):
            if body < 0:
                return None
            action = "OPEN_SHORT"
        if action is None:
            return None

        range_score = to_float(fac.get("range_score"), 0.0) or 0.0
        regime_label = _normalize_regime_label(
            fac.get("micro_regime") or fac.get("macro_regime") or fac.get("range_mode") or fac.get("regime")
        )
        history_profile = _history_profile(
            strategy_key=MicroAdaptiveRevert.name,
            pocket=MicroAdaptiveRevert.pocket,
            regime_label=regime_label,
        )
        if bool(history_profile.get("skip")):
            return None

        conf = 48.0
        conf += clamp(abs(dev) - threshold, 0.0, 2.5) * 2.4
        conf += clamp(retrace, 0.0, 2.0) * 1.9
        conf += clamp(vwap_gap and abs(vwap_gap) or 0.0, 0.0, 1.2) * 1.0
        conf -= clamp(abs(body), 0.0, 2.0) * 1.2
        conf += clamp(range_score, 0.0, 1.0) * 3.5

        if action == "OPEN_LONG":
            conf += _tuned_float(("strategies", "MicroAdaptiveRevert", "rsi_bonus"), 0.45) * clamp(
                1.0 - rsi / 100.0, 0.0, 1.0
            ) * 9.0
            conf += clamp(1.0 - dist_low / max(1.0, band_touch), 0.0, 1.0) * 4.2
        else:
            conf += (
                clamp(
                    (rsi - _tuned_float(("strategies", "MicroAdaptiveRevert", "rsi_bonus"), 58.0)) / 100.0,
                    0.0,
                    1.0,
                )
                * 7.0
            )
            conf += clamp(1.0 - dist_high / max(1.0, band_touch), 0.0, 1.0) * 4.2

        conf += (float(history_profile.get("score", 0.5)) - 0.5) * 18.0
        confidence = int(clamp(conf, 44.0, 89.0))

        sl_base = _tuned_float(("strategies", "MicroAdaptiveRevert", "sl_mul"), 1.18)
        tp_base = _tuned_float(("strategies", "MicroAdaptiveRevert", "tp_mul"), 0.98)
        sl_pips = clamp(atr * sl_base, 1.0, _tuned_float(("strategies", "MicroAdaptiveRevert", "sl_max"), 2.4))
        tp_pips = clamp(
            max(0.8, sl_pips * tp_base),
            _tuned_float(("strategies", "MicroAdaptiveRevert", "tp_min"), 0.8),
            _tuned_float(("strategies", "MicroAdaptiveRevert", "tp_max"), 2.2),
        )

        notes = {
            "close": round(close, 3),
            "atr": round(atr, 2),
            "deviation": round(dev, 2),
            "threshold": round(threshold, 2),
            "bbw": round(bbw or 0.0, 3),
            "adx": round(adx or 0.0, 2),
            "rsi": round(rsi, 2),
            "stoch": round(stoch, 3),
            "dist_low": round(dist_low, 2),
            "dist_high": round(dist_high, 2),
            "retrace": round(retrace, 2),
            "regime": regime_label,
            "history_score": round(float(history_profile.get("score", 0.5)), 3),
            "history_n": int(history_profile.get("n", 0) or 0),
            "history_lot_mult": round(float(history_profile.get("lot_multiplier", 1.0)), 3),
            "history_source": history_profile.get("source", "disabled"),
            "history_pf": round(float(history_profile.get("pf", 1.0)), 3),
            "history_used_regime": bool(history_profile.get("used_regime", False)),
        }

        return {
            "action": action,
            "sl_pips": round(float(sl_pips), 2),
            "tp_pips": round(float(tp_pips), 2),
            "confidence": confidence,
            "tag": f"{MicroAdaptiveRevert.name}-{'long' if action == 'OPEN_LONG' else 'short'}",
            "profile": "adaptive_revert",
            "notes": notes,
        }
