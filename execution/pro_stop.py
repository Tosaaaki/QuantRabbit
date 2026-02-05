"""Expert-style stop loss planning for open trades.

This module provides a deterministic, rule-based loss-cutting plan intended to
behave like a skilled discretionary stop: wait for a minimum hold, then cut
losers on structural invalidation or time failure, and hard-stop at a larger
adverse move.
"""

from __future__ import annotations

import logging
import os
import pathlib
import time
from datetime import datetime, timezone
from typing import Any, Optional

from utils.metrics_logger import log_metric

try:  # optional in offline/backtest
    from market_data import tick_window
except Exception:  # pragma: no cover
    tick_window = None
try:  # optional config
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional
    yaml = None

_FALSEY = {"", "0", "false", "no", "off"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in _FALSEY


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_csv_set(name: str, default: str) -> set[str]:
    raw = os.getenv(name, default)
    raw = str(raw or "").strip()
    if not raw:
        return set()
    return {token.strip().lower() for token in raw.split(",") if token.strip()}


def _pocket_env_float(base: str, pocket: str, default: float) -> float:
    if pocket:
        key = f"{base}_{str(pocket).upper()}"
        raw = os.getenv(key)
        if raw is not None:
            try:
                return float(raw)
            except ValueError:
                return default
    return _env_float(base, default)


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _latest_bid_ask_mid() -> tuple[Optional[float], Optional[float], Optional[float]]:
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


def _pnl_pips(
    entry: float,
    units: int,
    bid: Optional[float],
    ask: Optional[float],
    mid: Optional[float],
) -> Optional[float]:
    if entry <= 0 or units == 0:
        return None
    if units > 0:
        price = bid if bid is not None else mid
        if price is None:
            return None
        return (price - entry) / 0.01
    price = ask if ask is not None else mid
    if price is None:
        return None
    return (entry - price) / 0.01


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _loss_guard_pips(thesis: Optional[dict]) -> Optional[float]:
    if not isinstance(thesis, dict):
        return None
    candidates = [
        thesis.get("loss_guard_pips"),
        thesis.get("hard_stop_pips"),
        thesis.get("profile_sl_pips"),
        thesis.get("sl_pips"),
        thesis.get("loss_guard"),
    ]
    best = None
    for cand in candidates:
        val = _safe_float(cand)
        if val is None or val <= 0:
            continue
        if best is None or val > best:
            best = val
    return best


def _min_hold_sec(thesis: Optional[dict]) -> Optional[float]:
    if not isinstance(thesis, dict):
        return None
    for key in ("min_hold_sec", "min_hold_seconds"):
        val = _safe_float(thesis.get(key))
        if val is not None and val > 0:
            return val
    for key in ("min_hold_min", "min_hold_minutes"):
        val = _safe_float(thesis.get(key))
        if val is not None and val > 0:
            return val * 60.0
    return None


def _max_hold_sec(thesis: Optional[dict]) -> Optional[float]:
    if not isinstance(thesis, dict):
        return None
    for key in ("max_hold_sec", "max_hold_seconds"):
        val = _safe_float(thesis.get(key))
        if val is not None and val > 0:
            return val
    for key in ("max_hold_min", "max_hold_minutes"):
        val = _safe_float(thesis.get(key))
        if val is not None and val > 0:
            return val * 60.0
    return None


_PRO_STOP_ENABLED = _env_bool("PRO_STOP_ENABLED", False)
_PRO_STOP_POCKETS = _env_csv_set("PRO_STOP_POCKETS", "micro,macro,scalp,scalp_fast")
_PRO_STOP_SCORE_MIN = _env_float("PRO_STOP_SCORE_MIN", 2.0)
_PRO_STOP_COOLDOWN_SEC = max(1.0, _env_float("PRO_STOP_COOLDOWN_SEC", 20.0))
_PRO_STOP_SOFT_RATIO = _env_float("PRO_STOP_SOFT_RATIO", 1.0)
_PRO_STOP_HARD_RATIO = _env_float("PRO_STOP_HARD_RATIO", 1.35)
_PRO_STOP_REQUIRE_SIGNAL = _env_bool("PRO_STOP_REQUIRE_SIGNAL", True)
_PRO_STOP_USE_POCKET_DEFAULTS = _env_bool("PRO_STOP_USE_POCKET_DEFAULTS", False)
_GLOBAL_SOFT_DEFAULT = _env_float("PRO_STOP_SOFT_PIPS", 3.4)
_GLOBAL_HARD_DEFAULT = _env_float("PRO_STOP_HARD_PIPS", 9.5)
_GLOBAL_MIN_HOLD = _env_float("PRO_STOP_MIN_HOLD_SEC", 203.0)
_GLOBAL_MAX_HOLD = _env_float("PRO_STOP_MAX_HOLD_SEC", 1760.0)
_GLOBAL_SOFT_ATR_MULT = _env_float("PRO_STOP_SOFT_ATR_MULT", 1.0)
_GLOBAL_HARD_ATR_MULT = _env_float("PRO_STOP_HARD_ATR_MULT", 1.2)

_SOFT_DEFAULTS = {"macro": 6.5, "micro": 2.2, "scalp": 1.2}
_HARD_DEFAULTS = {"macro": 9.5, "micro": 3.6, "scalp": 2.0}
_MIN_HOLD_DEFAULTS = {"macro": 120.0, "micro": 40.0, "scalp": 20.0}
_MAX_HOLD_DEFAULTS = {"macro": 1800.0, "micro": 600.0, "scalp": 180.0}
_SOFT_ATR_MULT = {"macro": 1.1, "micro": 0.9, "scalp": 0.7}
_HARD_ATR_MULT = {"macro": 1.45, "micro": 1.15, "scalp": 0.9}

_RSI_FADE_LONG = _env_float("PRO_STOP_RSI_FADE_LONG", 44.0)
_RSI_FADE_SHORT = _env_float("PRO_STOP_RSI_FADE_SHORT", 56.0)
_VWAP_GAP_PIPS = _env_float("PRO_STOP_VWAP_GAP_PIPS", 0.8)
_STRUCTURE_ADX = _env_float("PRO_STOP_STRUCTURE_ADX", 20.0)
_STRUCTURE_GAP_PIPS = _env_float("PRO_STOP_STRUCTURE_GAP_PIPS", 1.8)
_ATR_SPIKE_PIPS = _env_float("PRO_STOP_ATR_SPIKE_PIPS", 5.0)

_LAST_PRO_STOP: dict[str, float] = {}

_PRO_STOP_CONFIG_ENABLED = _env_bool("PRO_STOP_CONFIG_ENABLED", True)
_PRO_STOP_CONFIG_PATH = pathlib.Path(
    os.getenv("PRO_STOP_CONFIG_PATH", "config/strategy_exit_protections.yaml")
)
_PRO_STOP_CONFIG_TTL_SEC = _env_float("PRO_STOP_CONFIG_TTL_SEC", 12.0)
_PRO_STOP_CONFIG_CACHE: dict[str, Any] = {"ts": 0.0, "data": None}
_STRATEGY_ALIAS_BASE = {
    "bbrsi": "BB_RSI",
    "bb_rsi": "BB_RSI",
    "trendma": "TrendMA",
    "donchian": "Donchian55",
    "donchian55": "Donchian55",
    "h1momentum": "H1Momentum",
    "m1scalper": "M1Scalper",
    "microlevelreactor": "MicroLevelReactor",
    "microrangebreak": "MicroRangeBreak",
    "microvwapbound": "MicroVWAPBound",
    "momentumburst": "MomentumBurst",
    "techfusion": "TechFusion",
    "macrotechfusion": "MacroTechFusion",
    "micropullbackfib": "MicroPullbackFib",
    "scalpreversalnwave": "ScalpReversalNWave",
    "rangecompressionbreak": "RangeCompressionBreak",
}


def _base_strategy_tag(tag: Optional[str]) -> str:
    if not tag:
        return ""
    text = str(tag).strip()
    if not text:
        return ""
    base = text.split("-", 1)[0].strip() or text
    alias = _STRATEGY_ALIAS_BASE.get(base.lower())
    return alias or base


def _load_pro_stop_config() -> dict:
    if not _PRO_STOP_CONFIG_ENABLED:
        return {"defaults": {}, "strategies": {}}
    now = time.monotonic()
    cached_ts = float(_PRO_STOP_CONFIG_CACHE.get("ts") or 0.0)
    if (now - cached_ts) < _PRO_STOP_CONFIG_TTL_SEC and isinstance(
        _PRO_STOP_CONFIG_CACHE.get("data"), dict
    ):
        return _PRO_STOP_CONFIG_CACHE["data"]  # type: ignore[return-value]
    payload: dict[str, Any] = {"defaults": {}, "strategies": {}}
    if yaml is not None and _PRO_STOP_CONFIG_PATH.exists():
        try:
            loaded = yaml.safe_load(_PRO_STOP_CONFIG_PATH.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                payload = loaded
        except Exception:
            payload = {"defaults": {}, "strategies": {}}
    _PRO_STOP_CONFIG_CACHE["ts"] = now
    _PRO_STOP_CONFIG_CACHE["data"] = payload
    return payload


def _strategy_override(config: dict, strategy_tag: Optional[str]) -> dict:
    if not isinstance(config, dict):
        return {}
    strategies = config.get("strategies")
    if not isinstance(strategies, dict) or not strategy_tag:
        return {}
    base = _base_strategy_tag(strategy_tag)
    candidates = [
        strategy_tag,
        base,
        str(strategy_tag).lower(),
        str(base).lower(),
    ]
    for key in candidates:
        if not key:
            continue
        override = strategies.get(key)
        if isinstance(override, dict):
            return override
    return {}


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return default
    return text not in _FALSEY


def _coerce_float(value: Any, default: Optional[float]) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _default_soft(pocket: str) -> float:
    if _PRO_STOP_USE_POCKET_DEFAULTS:
        return _pocket_env_float(
            "PRO_STOP_SOFT_PIPS",
            pocket,
            _SOFT_DEFAULTS.get(pocket, _GLOBAL_SOFT_DEFAULT),
        )
    return _env_float("PRO_STOP_SOFT_PIPS", _GLOBAL_SOFT_DEFAULT)


def _default_hard(pocket: str, soft: float) -> float:
    if _PRO_STOP_USE_POCKET_DEFAULTS:
        base = _pocket_env_float(
            "PRO_STOP_HARD_PIPS",
            pocket,
            _HARD_DEFAULTS.get(pocket, soft * 1.5),
        )
    else:
        base = _env_float("PRO_STOP_HARD_PIPS", _GLOBAL_HARD_DEFAULT)
    if base < soft:
        base = soft * 1.1
    return base


def _default_min_hold(pocket: str) -> float:
    if _PRO_STOP_USE_POCKET_DEFAULTS:
        return _pocket_env_float(
            "PRO_STOP_MIN_HOLD_SEC",
            pocket,
            _MIN_HOLD_DEFAULTS.get(pocket, _GLOBAL_MIN_HOLD),
        )
    return _env_float("PRO_STOP_MIN_HOLD_SEC", _GLOBAL_MIN_HOLD)


def _default_max_hold(pocket: str) -> float:
    if _PRO_STOP_USE_POCKET_DEFAULTS:
        return _pocket_env_float(
            "PRO_STOP_MAX_HOLD_SEC",
            pocket,
            _MAX_HOLD_DEFAULTS.get(pocket, _GLOBAL_MAX_HOLD),
        )
    return _env_float("PRO_STOP_MAX_HOLD_SEC", _GLOBAL_MAX_HOLD)


def _default_soft_atr_mult(pocket: str) -> float:
    if _PRO_STOP_USE_POCKET_DEFAULTS:
        return _pocket_env_float(
            "PRO_STOP_SOFT_ATR_MULT",
            pocket,
            _SOFT_ATR_MULT.get(pocket, _GLOBAL_SOFT_ATR_MULT),
        )
    return _env_float("PRO_STOP_SOFT_ATR_MULT", _GLOBAL_SOFT_ATR_MULT)


def _default_hard_atr_mult(pocket: str) -> float:
    if _PRO_STOP_USE_POCKET_DEFAULTS:
        return _pocket_env_float(
            "PRO_STOP_HARD_ATR_MULT",
            pocket,
            _HARD_ATR_MULT.get(pocket, _GLOBAL_HARD_ATR_MULT),
        )
    return _env_float("PRO_STOP_HARD_ATR_MULT", _GLOBAL_HARD_ATR_MULT)


def plan_pro_stop_closes(
    open_positions: dict,
    fac_m1: dict,
    fac_h4: dict,
    *,
    now: Optional[datetime] = None,
) -> list[dict]:
    if not _PRO_STOP_ENABLED:
        return []
    if not open_positions:
        return []

    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    cfg = _load_pro_stop_config()
    defaults_cfg = cfg.get("defaults", {}) if isinstance(cfg, dict) else {}
    default_pro = defaults_cfg.get("pro_stop", {}) if isinstance(defaults_cfg, dict) else {}
    bid, ask, mid = _latest_bid_ask_mid()
    if mid is None:
        mid = _safe_float((fac_m1 or {}).get("close"))

    atr_m1 = _safe_float((fac_m1 or {}).get("atr_pips"))
    if atr_m1 is None:
        atr_m1 = _safe_float((fac_m1 or {}).get("atr"))
        if atr_m1 is not None:
            atr_m1 *= 100.0
    atr_h4 = _safe_float((fac_h4 or {}).get("atr_pips"))
    if atr_h4 is None:
        atr_h4 = _safe_float((fac_h4 or {}).get("atr"))
        if atr_h4 is not None:
            atr_h4 *= 100.0

    actions: list[dict] = []
    for pocket, info in open_positions.items():
        if pocket == "__net__":
            continue
        pocket_key = str(pocket or "").lower()
        if _PRO_STOP_POCKETS and pocket_key not in _PRO_STOP_POCKETS:
            continue
        trades = info.get("open_trades") or []
        if not trades:
            continue

        for tr in trades:
            trade_id = tr.get("trade_id")
            if not trade_id:
                continue
            client_id = str(tr.get("client_id") or tr.get("client_order_id") or "")
            if not client_id.startswith("qr-"):
                continue

            thesis = tr.get("entry_thesis")
            if isinstance(thesis, dict):
                if thesis.get("disable_pro_stop") or thesis.get("pro_stop_enabled") is False:
                    continue
            strategy_tag = tr.get("strategy_tag")
            if isinstance(thesis, dict) and not strategy_tag:
                strategy_tag = thesis.get("strategy_tag") or thesis.get("strategy")
            strategy_tag = str(strategy_tag).strip() if strategy_tag else ""
            override = _strategy_override(cfg, strategy_tag) if strategy_tag else {}
            pro_cfg = override.get("pro_stop", {}) if isinstance(override, dict) else {}
            pro_enabled = _coerce_bool(
                pro_cfg.get("enabled"),
                _coerce_bool(default_pro.get("enabled"), True),
            )
            if not pro_enabled:
                continue
            score_min = _coerce_float(pro_cfg.get("score_min"), _coerce_float(default_pro.get("score_min"), None))
            if score_min is None:
                score_min = _PRO_STOP_SCORE_MIN
            require_signal = _coerce_bool(
                pro_cfg.get("require_signal"),
                _coerce_bool(default_pro.get("require_signal"), _PRO_STOP_REQUIRE_SIGNAL),
            )

            entry = _safe_float(tr.get("price")) or 0.0
            units = int(tr.get("units") or 0)
            if entry <= 0 or units == 0:
                continue

            pnl_pips = _pnl_pips(entry, units, bid, ask, mid)
            if pnl_pips is None:
                pnl_pips = _safe_float(tr.get("unrealized_pl_pips"))
            if pnl_pips is None or pnl_pips >= 0:
                continue

            opened_at = _parse_time(tr.get("open_time"))
            age_sec = None
            if opened_at:
                age_sec = max(0.0, (now - opened_at).total_seconds())

            min_hold = _coerce_float(pro_cfg.get("min_hold_sec"), _coerce_float(default_pro.get("min_hold_sec"), None))
            if min_hold is None:
                min_hold = _min_hold_sec(thesis)
            if min_hold is None:
                min_hold = _default_min_hold(pocket_key)
            if age_sec is not None and age_sec < min_hold:
                continue

            max_hold = _coerce_float(pro_cfg.get("max_hold_sec"), _coerce_float(default_pro.get("max_hold_sec"), None))
            if max_hold is None:
                max_hold = _max_hold_sec(thesis)
            if max_hold is None:
                max_hold = _default_max_hold(pocket_key)

            loss_guard = _loss_guard_pips(thesis)
            soft = _coerce_float(pro_cfg.get("soft_pips"), _coerce_float(default_pro.get("soft_pips"), None))
            if soft is None:
                soft = _default_soft(pocket_key)
            hard = _coerce_float(pro_cfg.get("hard_pips"), _coerce_float(default_pro.get("hard_pips"), None))
            if hard is None:
                hard = _default_hard(pocket_key, soft)
            if loss_guard:
                soft = max(soft, loss_guard * _PRO_STOP_SOFT_RATIO)
                hard = max(hard, loss_guard * _PRO_STOP_HARD_RATIO)

            atr_ref = atr_h4 if pocket_key == "macro" and atr_h4 else atr_m1
            if atr_ref and atr_ref > 0:
                soft_mult = _coerce_float(
                    pro_cfg.get("soft_atr_mult"),
                    _coerce_float(default_pro.get("soft_atr_mult"), None),
                )
                if soft_mult is None:
                    soft_mult = _default_soft_atr_mult(pocket_key)
                hard_mult = _coerce_float(
                    pro_cfg.get("hard_atr_mult"),
                    _coerce_float(default_pro.get("hard_atr_mult"), None),
                )
                if hard_mult is None:
                    hard_mult = _default_hard_atr_mult(pocket_key)
                soft = max(soft, atr_ref * soft_mult)
                hard = max(hard, atr_ref * hard_mult)

            if hard < soft:
                hard = soft * 1.1

            fac_struct = fac_h4 if pocket_key == "macro" and fac_h4 else fac_m1
            rsi = _safe_float((fac_m1 or {}).get("rsi"))
            vwap_gap = _safe_float((fac_m1 or {}).get("vwap_gap"))
            adx = _safe_float((fac_struct or {}).get("adx"))
            ma10 = _safe_float((fac_struct or {}).get("ma10"))
            ma20 = _safe_float((fac_struct or {}).get("ma20"))
            atr_pips = _safe_float((fac_m1 or {}).get("atr_pips"))
            if atr_pips is None:
                atr_pips = _safe_float((fac_m1 or {}).get("atr"))
                if atr_pips is not None:
                    atr_pips *= 100.0

            structure_break = False
            if adx is not None and ma10 is not None and ma20 is not None:
                gap = abs(ma10 - ma20) / 0.01
                side_long = units > 0
                cross_bad = (side_long and ma10 <= ma20) or ((not side_long) and ma10 >= ma20)
                if adx <= _STRUCTURE_ADX and (cross_bad or gap <= _STRUCTURE_GAP_PIPS):
                    structure_break = True

            rsi_fade = False
            if rsi is not None:
                if units > 0 and rsi <= _RSI_FADE_LONG:
                    rsi_fade = True
                elif units < 0 and rsi >= _RSI_FADE_SHORT:
                    rsi_fade = True

            vwap_cut = vwap_gap is not None and abs(vwap_gap) <= _VWAP_GAP_PIPS
            atr_spike = atr_pips is not None and atr_pips >= _ATR_SPIKE_PIPS

            score = 0
            if rsi_fade:
                score += 1
            if vwap_cut:
                score += 1
            if atr_spike:
                score += 1
            if structure_break:
                score += 2

            reason = None
            if pnl_pips <= -hard:
                reason = "hard_stop"
            elif max_hold and age_sec is not None and age_sec >= max_hold:
                reason = "time_stop"
            elif pnl_pips <= -soft:
                if not require_signal or score >= score_min:
                    if structure_break:
                        reason = "structure_break"
                    elif rsi_fade:
                        reason = "rsi_fade"
                    elif vwap_cut:
                        reason = "vwap_cut"
                    elif atr_spike:
                        reason = "atr_spike"
                    else:
                        reason = "tech_hard_stop"

            if not reason:
                continue

            last_ts = _LAST_PRO_STOP.get(str(trade_id))
            now_ts = time.time()
            if last_ts is not None and now_ts - last_ts < _PRO_STOP_COOLDOWN_SEC:
                continue
            _LAST_PRO_STOP[str(trade_id)] = now_ts

            actions.append(
                {
                    "trade_id": str(trade_id),
                    "client_id": client_id,
                    "pocket": pocket_key,
                    "reason": reason,
                    "pnl_pips": float(pnl_pips),
                    "score": score,
                    "soft_pips": round(soft, 3),
                    "hard_pips": round(hard, 3),
                }
            )

            log_metric(
                "pro_stop_trigger",
                float(pnl_pips),
                tags={
                    "reason": reason,
                    "pocket": pocket_key,
                    "score": str(score),
                },
            )
            logging.info(
                "[PRO_STOP] trade=%s pocket=%s pnl=%.2fp reason=%s soft=%.2f hard=%.2f score=%s",
                trade_id,
                pocket_key,
                float(pnl_pips),
                reason,
                soft,
                hard,
                score,
            )

    return actions
