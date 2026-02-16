from __future__ import annotations
from typing import Dict, Optional
import json
import os
import logging
from pathlib import Path
import time

try:
    from analysis.patterns import NWaveStructure, detect_latest_n_wave
except ModuleNotFoundError:  # pragma: no cover - fallback when optional module not deployed
    NWaveStructure = None  # type: ignore

    def detect_latest_n_wave(*args, **kwargs):  # type: ignore
        return None

try:
    from analysis.pattern_stats import derive_pattern_signature
except ModuleNotFoundError:  # pragma: no cover - optional in slim deployments
    derive_pattern_signature = None  # type: ignore

from market_data import orderbook_state

_PIP = 0.01
_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_CONFIG_PATH = _ROOT / "logs" / "tuning" / "scalp_active_params.json"
_LEGACY_CONFIG_PATH = _ROOT / "configs" / "scalp_active_params.json"
_CONFIG_OVERRIDE = os.getenv("SCALP_ACTIVE_PARAMS_PATH")
_PARAM_CACHE: Dict[str, Dict] = {"path": None, "mtime": None, "data": {}}
_LOGGER = logging.getLogger(__name__)
_EMPTY_TICK_LOG_DEBOUNCE_SEC = 20.0
_last_no_tick_log_ts = 0.0
_last_shock_ts = 0.0

# 強いトレンドを逆行するエントリーを抑制するための閾値
STRONG_TREND_PIPS = 5.0   # ema10-ema20差や価格-ema20差の目安
STRONG_MOMENTUM = 0.0035  # 価格変化量（約0.35pips）


def _log(reason: str, **kwargs: object) -> None:
    if not kwargs:
        _LOGGER.info("[M1SCALPER] %s", reason)
        return
    payload = " ".join(f"{key}={value}" for key, value in kwargs.items())
    _LOGGER.info("[M1SCALPER] %s %s", reason, payload)


def _attach_kill(signal: Dict) -> Dict:
    """Ensure kill/fast_cut opt-in tags are present."""
    tags = []
    raw_tags = signal.get("exit_tags") or signal.get("tags")
    if raw_tags:
        if isinstance(raw_tags, str):
            tags = [raw_tags]
        elif isinstance(raw_tags, (list, tuple)):
            tags = list(raw_tags)
    tags = [t for t in tags if isinstance(t, str)]
    if "kill" not in [t.lower() for t in tags]:
        tags.append("kill")
    if "fast_cut" not in [t.lower() for t in tags]:
        tags.append("fast_cut")
    signal["exit_tags"] = tags
    signal["kill_switch"] = True
    return signal


def _load_scalper_config() -> Dict:
    if _CONFIG_OVERRIDE:
        path = Path(_CONFIG_OVERRIDE)
    elif _RUNTIME_CONFIG_PATH.exists():
        path = _RUNTIME_CONFIG_PATH
    else:
        path = _LEGACY_CONFIG_PATH
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return {}
    cached_path = _PARAM_CACHE.get("path")
    cached_mtime = _PARAM_CACHE.get("mtime")
    if cached_path != str(path) or cached_mtime != mtime:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
        _PARAM_CACHE["path"] = str(path)
        _PARAM_CACHE["mtime"] = mtime
        _PARAM_CACHE["data"] = data.get("M1Scalper", data.get("m1scalper", {}))
    return _PARAM_CACHE.get("data", {})


def _to_float(value: object, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no"}
    if value is None:
        return default
    try:
        return bool(value)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return default if val is None else (_to_float(val, default) or default)


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    try:
        return int(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    return _to_bool(val, default)


_NWAVE_ALIGN_ENABLED = _env_bool("M1SCALP_NWAVE_ALIGN_ENABLED", True)
_NWAVE_ALIGN_PREV_BODY_LONG = abs(_env_float("M1SCALP_NWAVE_ALIGN_PREV_BODY_PIPS", 0.35))
_NWAVE_ALIGN_LAST_BODY_LONG = abs(_env_float("M1SCALP_NWAVE_ALIGN_LAST_BODY_PIPS", 0.15))
_NWAVE_ALIGN_PREV_BODY_SHORT = abs(_env_float("M1SCALP_NWAVE_ALIGN_PREV_BODY_SHORT_PIPS", 0.35))
_NWAVE_ALIGN_LAST_BODY_SHORT = abs(_env_float("M1SCALP_NWAVE_ALIGN_LAST_BODY_SHORT_PIPS", 0.15))


def _candle_body_pips(candle: Dict[str, float]) -> Optional[float]:
    open_px = _to_float(candle.get("open"))
    close_px = _to_float(candle.get("close"))
    if open_px is None or close_px is None:
        return None
    return (close_px - open_px) / _PIP


def _candle_high_low(candle: Dict[str, float]) -> tuple[Optional[float], Optional[float]]:
    high = _to_float(candle.get("high") or candle.get("h"))
    low = _to_float(candle.get("low") or candle.get("l"))
    return high, low


def _candle_range_pips(candle: Dict[str, float]) -> Optional[float]:
    high, low = _candle_high_low(candle)
    if high is None or low is None:
        return None
    return (high - low) / _PIP


SESSION_BIAS_ENABLED = _env_bool("M1SCALP_SESSION_BIAS_ENABLED", False)


def _force_mode() -> bool:
    return os.getenv("SCALP_FORCE_ALWAYS", "0").strip().lower() not in {"", "0", "false", "no"}


def _cfg_float(section: Dict, key: str, default: float) -> float:
    val = _to_float(section.get(key))
    return default if val is None else val


def _shock_guard(candles: list[dict], atr_pips: Optional[float]) -> bool:
    if not _env_bool("M1SCALP_SHOCK_GUARD_ENABLED", True):
        return True
    if _force_mode():
        return True
    if not candles:
        return True
    now_ts = time.time()
    cooldown = max(10.0, _env_float("M1SCALP_SHOCK_COOLDOWN_SEC", 90.0))
    global _last_shock_ts
    if _last_shock_ts and (now_ts - _last_shock_ts) < cooldown:
        return False
    last = candles[-1]
    range_1m = _candle_range_pips(last) or 0.0
    body_1m = abs(_candle_body_pips(last) or 0.0)
    range_2m = 0.0
    if len(candles) >= 2:
        h1, l1 = _candle_high_low(candles[-1])
        h2, l2 = _candle_high_low(candles[-2])
        if h1 is not None and l1 is not None and h2 is not None and l2 is not None:
            range_2m = (max(h1, h2) - min(l1, l2)) / _PIP
    atr_val = atr_pips or 0.0
    range_floor = _env_float("M1SCALP_SHOCK_RANGE_PIPS", 9.0)
    atr_mult = _env_float("M1SCALP_SHOCK_RANGE_ATR_MULT", 2.2)
    range_thresh = max(range_floor, atr_val * max(0.5, atr_mult))
    body_thresh = _env_float("M1SCALP_SHOCK_BODY_PIPS", 6.0)
    range2_thresh = _env_float("M1SCALP_SHOCK_RANGE2_PIPS", 12.0)
    if range_1m >= range_thresh or body_1m >= body_thresh or range_2m >= range2_thresh:
        _last_shock_ts = now_ts
        _log(
            "shock_guard_block",
            range_1m=round(range_1m, 2),
            body_1m=round(body_1m, 2),
            range_2m=round(range_2m, 2),
            atr=round(atr_val, 2),
            cooldown=int(cooldown),
        )
        return False
    return True


def _liquidity_guard() -> bool:
    # Disabled by default: orderbook "liquidity" snapshots are noisy and can unintentionally
    # freeze entries for long periods depending on the stream/provider.
    if not _env_bool("M1SCALP_LIQ_GUARD_ENABLED", False):
        return True
    if _force_mode():
        return True
    max_age_ms = _env_float("M1SCALP_LIQ_MAX_AGE_MS", 2500.0)
    min_size = _env_float("M1SCALP_LIQ_MIN_SIZE", 600000.0)
    depth = _env_int("M1SCALP_LIQ_DEPTH", 1)
    failopen = _env_bool("M1SCALP_LIQ_FAILOPEN", True)
    snapshot = orderbook_state.get_latest(max_age_ms=max_age_ms)
    if snapshot is None:
        return True if failopen else False
    if orderbook_state.has_sufficient_depth(snapshot, depth=depth, min_size=min_size):
        return True
    age_ms = orderbook_state.latest_age_ms()
    _log(
        "liq_guard_block",
        depth=depth,
        min_size=int(min_size),
        bid=round(snapshot.bid_levels[0].price, 3),
        ask=round(snapshot.ask_levels[0].price, 3),
        age_ms=int(age_ms or 0),
    )
    return False


def _tech_multiplier(fac: Dict[str, object]) -> float:
    """Composite multiplier from expanded technical set."""
    try:
        adx = float(fac.get("adx") or 0.0)
        rsi = float(fac.get("rsi") or 50.0)
        bbw = float(fac.get("bbw") or 0.0)
        vol5 = float(fac.get("vol_5m") or 0.0)
        macd_hist = float(fac.get("macd_hist") or 0.0)
        stoch = float(fac.get("stoch_rsi") or 0.0)
        plus_di = float(fac.get("plus_di") or 0.0)
        minus_di = float(fac.get("minus_di") or 0.0)
        kc_width = float(fac.get("kc_width") or 0.0)
        don_width = float(fac.get("donchian_width") or 0.0)
        chaikin_vol = float(fac.get("chaikin_vol") or 0.0)
        vwap_gap = float(fac.get("vwap_gap") or 0.0)
        roc5 = float(fac.get("roc5") or 0.0)
        roc10 = float(fac.get("roc10") or 0.0)
        cci = float(fac.get("cci") or 0.0)
        ichimoku_pos = float(fac.get("ichimoku_cloud_pos") or 0.0)
        cluster_high = float(fac.get("cluster_high_gap") or 0.0)
        cluster_low = float(fac.get("cluster_low_gap") or 0.0)
        ma10 = float(fac.get("ma10") or 0.0)
        ma20 = float(fac.get("ma20") or 0.0)
    except Exception:
        adx = rsi = bbw = vol5 = macd_hist = stoch = plus_di = minus_di = kc_width = don_width = chaikin_vol = vwap_gap = roc5 = roc10 = cci = ichimoku_pos = cluster_high = cluster_low = ma10 = ma20 = 0.0

    score = 0.0
    # トレンド強/順行
    if adx >= 25:
        score += 0.4
    if macd_hist > 0:
        score += 0.3
    elif macd_hist < 0:
        score -= 0.3
    dmi_diff = plus_di - minus_di
    if dmi_diff > 5:
        score += 0.2
    elif dmi_diff < -5:
        score -= 0.2
    # モメンタム
    if roc5 > 0:
        score += 0.1
    elif roc5 < 0:
        score -= 0.1
    if roc10 > 0:
        score += 0.08
    elif roc10 < 0:
        score -= 0.08
    if cci >= 100:
        score += 0.05
    elif cci <= -100:
        score -= 0.05
    # オシレーター極端は減点
    if stoch >= 0.8 or stoch <= 0.2:
        score -= 0.1
    # ボラ幅
    if kc_width > 0.015 or don_width > 0.015 or chaikin_vol > 0.2:
        score -= 0.1
    elif 0.006 <= kc_width <= 0.012:
        score += 0.05
    # レンジ/低ボラ
    if bbw <= 0.0013 and adx <= 14:
        score -= 0.1
    if vol5 <= 0.5:
        score -= 0.05
    elif vol5 >= 1.5:
        score += 0.05
    # VWAP乖離で逆張り余地
    if abs(vwap_gap) >= 5.0:
        score += 0.05
    # RSI極端は抑制
    if rsi <= 25 or rsi >= 75:
        score -= 0.05
    # Ichimoku/クラスタバイアス
    if ichimoku_pos > 0.8:
        score += 0.08
    elif ichimoku_pos < -0.8:
        score -= 0.1
    distances = [c for c in (cluster_high, cluster_low) if c and c > 0]
    min_cluster = float(min(distances)) if distances else 0.0
    if min_cluster > 0:
        if min_cluster < 3.0:
            score -= 0.08
        elif min_cluster > 7.0:
            score += 0.05

    # MA整列/スロープで方向強化
    slope = (ma10 - ma20) / _PIP if _PIP else 0.0
    if slope > 2.0:
        score += 0.08
    elif slope < -2.0:
        score -= 0.08

    mult = 1.0 + score * 0.08
    # セッションバイアス（任意）
    if SESSION_BIAS_ENABLED:
        hour = time.gmtime().tm_hour
        if 7 <= hour < 17 or 17 <= hour < 23:
            mult *= 1.03
        else:
            mult *= 0.97
    return max(0.7, min(1.35, mult))


def _structure_targets(signal: Dict[str, object], fac: Dict[str, object], direction: str) -> Dict[str, object]:
    """
    リアルタイムのスイング高安・VWAP・MA傾きを使ってTP/SLを補強する。
    - TP: 直近スイング方向の伸び余地とMA傾きで底上げ
    - SL: 直近逆方向スイングにクッションを置き、デフォルトより緩めるのみ（削らない）
    """
    sig = dict(signal)
    tp_default = _to_float(sig.get("tp_pips")) or 0.0
    sl_default = _to_float(sig.get("sl_pips")) or 0.0
    candles = list(fac.get("candles") or [])
    close_px = _to_float(fac.get("close"))
    if close_px is None and candles:
        close_px = _to_float(candles[-1].get("close"))
    highs: list[float] = []
    lows: list[float] = []
    for c in candles[-20:]:
        h = _to_float(c.get("high") or c.get("h"))
        l = _to_float(c.get("low") or c.get("l"))
        if h is not None:
            highs.append(h)
        if l is not None:
            lows.append(l)
    swing_high = max(highs) if highs else None
    swing_low = min(lows) if lows else None
    vwap_gap_pips = None
    try:
        vwap = _to_float(fac.get("vwap"))
        if vwap is not None and close_px is not None:
            vwap_gap_pips = abs(close_px - vwap) / _PIP
    except Exception:
        vwap_gap_pips = None
    try:
        ma_fast = _to_float(fac.get("ma10")) or _to_float(fac.get("ema12")) or 0.0
        ma_slow = _to_float(fac.get("ma20")) or _to_float(fac.get("ema20")) or 0.0
        slope_boost = abs(ma_fast - ma_slow) / _PIP * 0.08
    except Exception:
        slope_boost = 0.0

    tp_out = tp_default
    sl_out = sl_default
    if close_px is not None:
        if direction == "long":
            swing_room = (swing_high - close_px) / _PIP if swing_high and swing_high > close_px else None
            if swing_room is not None:
                tp_out = max(tp_out, min(tp_out * 1.8, swing_room * 0.9 + slope_boost))
            swing_buffer = (close_px - swing_low) / _PIP * 0.9 if swing_low and swing_low < close_px else None
            if swing_buffer:
                sl_out = max(sl_out, min(sl_out * 1.8, swing_buffer))
        else:
            swing_room = (close_px - swing_low) / _PIP if swing_low and swing_low < close_px else None
            if swing_room is not None:
                tp_out = max(tp_out, min(tp_out * 1.8, swing_room * 0.9 + slope_boost))
            swing_buffer = (swing_high - close_px) / _PIP * 0.9 if swing_high and swing_high > close_px else None
            if swing_buffer:
                sl_out = max(sl_out, min(sl_out * 1.8, swing_buffer))
    if vwap_gap_pips is not None and vwap_gap_pips > 4.0 and tp_out > 0:
        # 大きく乖離しているときは半分を戻しに充てる程度に抑える
        cap = tp_default + max(0.0, (vwap_gap_pips - 4.0) * 0.15)
        tp_out = min(tp_out, cap)
    sig["tp_pips"] = round(max(tp_out, tp_default), 2)
    sig["sl_pips"] = round(max(sl_out, sl_default), 2)
    return sig


def _scale_signal(signal: Dict[str, object], mult: float) -> Dict[str, object]:
    """Apply composite multiplier to confidence/tp."""
    if not signal or not isinstance(signal, dict):
        return signal
    sig = dict(signal)
    conf = int(sig.get("confidence", 50) or 50)
    conf = int(max(0, min(100, conf * mult)))
    sig["confidence"] = conf
    tp = _to_float(sig.get("tp_pips"))
    if tp is not None:
        tp_scaled = max(0.6, min(tp * mult, tp * 1.35))
        sig["tp_pips"] = round(tp_scaled, 2)
    return sig

class M1Scalper:
    name = "M1Scalper"
    pocket = "scalp"

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        cfg = _load_scalper_config()
        fallback_cfg = cfg.get("fallback", {}) if isinstance(cfg, dict) else {}
        nwave_cfg = cfg.get("nwave", {}) if isinstance(cfg, dict) else {}
        scalp_tactical = _to_bool(cfg.get("tactical") or cfg.get("scalp_tactical"), False)

        def _fallback_float(key: str, default: float) -> float:
            return _cfg_float(fallback_cfg, key, default)

        def _nwave_float(key: str, default: float) -> float:
            return _cfg_float(nwave_cfg, key, default)

        candles = fac.get("candles") or []
        nwave = detect_latest_n_wave(candles) if detect_latest_n_wave else None
        close = fac.get("close")
        ema20 = fac.get("ema20")
        rsi = fac.get("rsi")
        atr = fac.get("atr", 0.02)
        adx = fac.get("adx", 0.0) or 0.0
        vol5 = fac.get("vol_5m", 0.0) or 0.0
        bbw = fac.get("bbw") or 0.0
        bb_upper = fac.get("bb_upper")
        bb_lower = fac.get("bb_lower")
        cci = fac.get("cci")
        stoch = fac.get("stoch_rsi")
        vwap = fac.get("vwap")
        if close is None or ema20 is None or rsi is None:
            return None

        tech_mult = _tech_multiplier(fac)
        tech_mult_r = round(tech_mult, 3)

        def _apply_mult(signal: Dict[str, object]) -> Dict[str, object]:
            sig = _scale_signal(signal, tech_mult)
            if isinstance(sig, dict):
                notes = sig.get("notes") or {}
                if not isinstance(notes, dict):
                    notes = {}
                notes["tech_mult"] = tech_mult_r
                sig["notes"] = notes
            return sig

        momentum = close - ema20
        ema10 = fac.get("ema10")
        ema_gap_pips = 0.0
        if ema10 is not None:
            try:
                ema_gap_pips = (float(ema10) - float(ema20)) / _PIP
            except Exception:
                ema_gap_pips = 0.0
        price_gap_pips = (float(close) - float(ema20)) / _PIP if close is not None else 0.0
        # Prefer explicit atr_pips if provided; otherwise convert ATR (price units) to pips
        atr_pips = _to_float(fac.get("atr_pips"))
        if atr_pips is None:
            atr_pips = (atr or 0.0) * 100
        if not _shock_guard(candles, atr_pips):
            return None
        if not _liquidity_guard():
            return None

        def _short_trap_pattern() -> tuple[bool, Optional[str]]:
            if not _env_bool("M1SCALP_BLOCK_MARU_UP_SHORT", True):
                return False, None
            if derive_pattern_signature is None:
                return False, None
            last = candles[-1] if candles else {}
            if not isinstance(last, dict):
                return False, None
            open_px = _to_float(fac.get("open") or last.get("open") or last.get("o"))
            high_px = _to_float(fac.get("high") or last.get("high") or last.get("h"))
            low_px = _to_float(fac.get("low") or last.get("low") or last.get("l"))
            close_px = _to_float(fac.get("close") or last.get("close") or last.get("c"))
            if None in (open_px, high_px, low_px, close_px):
                return False, None
            pattern_fac = {
                "open": open_px,
                "high": high_px,
                "low": low_px,
                "close": close_px,
                "ma10": fac.get("ma10"),
                "ma20": fac.get("ma20"),
                "rsi": rsi,
                "atr_pips": atr_pips,
                "bbw": bbw,
            }
            tag, _ = derive_pattern_signature(pattern_fac, action="OPEN_SHORT")
            if not tag:
                return False, None
            parts = {}
            for part in tag.split("|"):
                if ":" not in part:
                    continue
                key, value = part.split(":", 1)
                parts[key] = value
            is_trap = (
                parts.get("c") == "maru_up"
                and parts.get("w") == "upper"
                and parts.get("tr") in {"flat", "dn_mild"}
                and parts.get("rsi") in {"mid_high", "ob"}
                and parts.get("vol") == "tight"
                and parts.get("atr") in {"low", "ultra_low"}
            )
            return is_trap, tag

        def _long_trap_pattern() -> tuple[bool, Optional[str]]:
            if not _env_bool("M1SCALP_BLOCK_MARU_DN_LONG", True):
                return False, None
            if derive_pattern_signature is None:
                return False, None
            last = candles[-1] if candles else {}
            if not isinstance(last, dict):
                return False, None
            open_px = _to_float(fac.get("open") or last.get("open") or last.get("o"))
            high_px = _to_float(fac.get("high") or last.get("high") or last.get("h"))
            low_px = _to_float(fac.get("low") or last.get("low") or last.get("l"))
            close_px = _to_float(fac.get("close") or last.get("close") or last.get("c"))
            if None in (open_px, high_px, low_px, close_px):
                return False, None
            pattern_fac = {
                "open": open_px,
                "high": high_px,
                "low": low_px,
                "close": close_px,
                "ma10": fac.get("ma10"),
                "ma20": fac.get("ma20"),
                "rsi": rsi,
                "atr_pips": atr_pips,
                "bbw": bbw,
            }
            tag, _ = derive_pattern_signature(pattern_fac, action="OPEN_LONG")
            if not tag:
                return False, None
            parts = {}
            for part in tag.split("|"):
                if ":" not in part:
                    continue
                key, value = part.split(":", 1)
                parts[key] = value
            is_trap = (
                parts.get("c") == "maru_dn"
                and parts.get("w") == "lower"
                and parts.get("tr") in {"flat", "up_mild"}
                and parts.get("rsi") in {"mid_low", "os"}
                and parts.get("vol") == "tight"
                and parts.get("atr") in {"low", "ultra_low"}
            )
            return is_trap, tag

        def _adjust_tp(tp: float, conf: int) -> float:
            """TPを信頼度とボラ/レンジ状態で可変化する。"""
            if low_vol_range:
                base = 2.0
                if conf >= 85 and atr_pips >= 2.0:
                    base = min(max(base, atr_pips * 1.2), 4.5)
                elif conf >= 70:
                    base = max(base, min(3.0, 1.0 + atr_pips))
                return round(base, 2)
            out = tp
            if conf >= 85 and atr_pips >= 2.5:
                out = min(tp * 1.2, 9.0)
            elif conf <= 60:
                out = max(3.0, min(tp, 6.0))
            return round(out, 2)

        # Precision gates (configurable via fallback section)
        # Allow entries in thinner markets by lowering the default ATR/volatility floors.
        atr_floor = _fallback_float("atr_floor", 0.4)
        if scalp_tactical:
            atr_floor = _fallback_float("atr_floor_tactical", atr_floor)
        vol5_min = _fallback_float("vol5_min", 0.05)
        adx_min = _fallback_float("adx_min", 5.0)
        momentum_thresh = _fallback_float("momentum_thresh", 0.0010)
        rsi_long_max = _fallback_float("rsi_long_max", 48.0)
        rsi_short_min = _fallback_float("rsi_short_min", 52.0)
        rsi_trend_long_max = _fallback_float("rsi_trend_long_max", 70.0)
        rsi_trend_short_min = _fallback_float("rsi_trend_short_min", 45.0)
        rsi_edge_min = _fallback_float("rsi_edge_min", 0.25)
        buy_dip_rsi_avoid_min = _fallback_float("buy_dip_rsi_avoid_min", 0.0)
        buy_dip_rsi_avoid_max = _fallback_float("buy_dip_rsi_avoid_max", 0.0)
        buy_dip_min_atr = _fallback_float("buy_dip_min_atr", 0.0)
        buy_dip_min_vol = _fallback_float("buy_dip_min_vol5", 0.0)
        range_flip_block_score = _fallback_float("range_flip_block_score", 0.58)
        range_active = _to_bool(fac.get("range_active"), False)
        range_score = _to_float(fac.get("range_score"), 0.0) or 0.0
        range_mode = str(fac.get("range_mode") or "").strip().upper()

        # レンジ・低ボラを検知し、帯付近のみエントリーを許可
        low_vol_range = (adx < 18.0 and bbw > 0.0 and bbw < 0.0016 and atr_pips < 2.4)
        if low_vol_range:
            # BB 上下どちらかのバンドに近い場合のみ許可（2.5pips以内へ緩和）
            near_band = False
            if bb_upper is not None and bb_lower is not None:
                try:
                    dist_upper = (float(bb_upper) - float(close)) / _PIP
                    dist_lower = (float(close) - float(bb_lower)) / _PIP
                    if dist_upper <= 2.5 or dist_lower <= 2.5:
                        near_band = True
                except Exception:
                    near_band = False
            # VWAP 乖離が大きい場合も許可（中心回帰狙い）
            if not near_band and vwap is not None:
                try:
                    vwap_gap = abs(float(close) - float(vwap)) / _PIP
                    if vwap_gap >= 1.4:
                        near_band = True
                except Exception:
                    pass
            if not near_band:
                low_vol_block = _to_bool(fallback_cfg.get("low_vol_block", True), True)
                if low_vol_block:
                    _log(
                        "range_block_no_edge",
                        bbw=round(bbw, 5),
                        adx=round(adx, 2),
                        atr_pips=round(atr_pips, 2),
                    )
                    return None
                # allow but log when explicitly configured
                _log("range_allow_far_band", bbw=round(bbw, 5), adx=round(adx, 2), atr_pips=round(atr_pips, 2))
        range_reversion_only = (
            low_vol_range
            or range_active
            or range_mode == "RANGE"
            or range_score >= range_flip_block_score
        )

        # Tighten gates to reduce low-quality entries
        if atr_pips < atr_floor:
            return None
        if vol5 < vol5_min:
            return None
        if adx < adx_min:
            return None
        # トレンド方向を判定（強い順行なら逆張りを避け、順張りに寄せる）
        diff_pips = momentum / _PIP
        trend_up = diff_pips >= 3.0 and momentum > 0.003 and ema_gap_pips > -0.6
        trend_down = diff_pips <= -3.0 and momentum < -0.003 and ema_gap_pips < -0.6
        strong_up = (price_gap_pips >= STRONG_TREND_PIPS or ema_gap_pips >= STRONG_TREND_PIPS) and momentum > STRONG_MOMENTUM
        strong_down = (price_gap_pips <= -STRONG_TREND_PIPS or ema_gap_pips <= -STRONG_TREND_PIPS) and momentum < -STRONG_MOMENTUM

        # Dynamic TP/SL (pips) tuned to recent volatility
        # - TP ≈ 3x ATR (pips) within [5, 9]
        # - SL ≈ min(2x ATR, 0.95*TP) with a floor of 4, keeping RR >= ~1.05
        tp_dyn = max(5.0, min(9.0, atr_pips * 3.0))
        sl_dyn = max(4.0, min(atr_pips * 2.0, tp_dyn * 0.95))
        tp_dyn = round(tp_dyn, 2)
        sl_dyn = round(sl_dyn, 2)
        fast_cut = max(5.0, atr_pips * 0.85)
        fast_cut_time = max(50.0, atr_pips * 12.0)
        if abs(momentum) > 0.0045:
            fast_cut *= 0.85
            fast_cut_time *= 0.85
        conf_scale = 1.0
        if atr_pips > 4.0:
            conf_scale = 0.8

        if momentum < -momentum_thresh and rsi < rsi_long_max:
            speed = abs(momentum) / max(0.0005, atr)
            rsi_gap = max(0.0, 55 - rsi) / 10
            confidence = int(
                max(40.0, min(95.0, 45.0 + speed * 30.0 + rsi_gap * 25.0))
            )
            action = "OPEN_LONG"
            if trend_down:
                if range_reversion_only:
                    _log(
                        "range_hold_reversion_long",
                        trend="down",
                        range_score=round(range_score, 3),
                        mode=range_mode or "-",
                    )
                else:
                    # 強い下落トレンドでは順張りショートに切替
                    action = "OPEN_SHORT"
                    confidence = int(confidence * 0.9)
                    if rsi < rsi_trend_short_min:
                        _log("trend_block_short_rsi", rsi=round(rsi, 2))
                        return None
            if action == "OPEN_LONG":
                if buy_dip_rsi_avoid_max > buy_dip_rsi_avoid_min and buy_dip_rsi_avoid_min <= rsi <= buy_dip_rsi_avoid_max:
                    _log(
                        "buy_dip_block_rsi_band",
                        rsi=round(rsi, 2),
                        band=f"{buy_dip_rsi_avoid_min:.1f}-{buy_dip_rsi_avoid_max:.1f}",
                    )
                    return None
                if buy_dip_min_atr > 0.0 and atr_pips < buy_dip_min_atr:
                    _log(
                        "buy_dip_block_atr",
                        atr_pips=round(atr_pips, 2),
                        min_atr=round(buy_dip_min_atr, 2),
                    )
                    return None
                if buy_dip_min_vol > 0.0 and vol5 < buy_dip_min_vol:
                    _log(
                        "buy_dip_block_vol",
                        vol5=round(vol5, 2),
                        min_vol=round(buy_dip_min_vol, 2),
                    )
                    return None
            if action == "OPEN_LONG" and strong_down:
                _log("trend_block_long", momentum=round(momentum, 5), ema_gap=round(ema_gap_pips, 3), price_gap=round(price_gap_pips, 3))
                return None
            if action == "OPEN_LONG":
                trap_hit, trap_tag = _long_trap_pattern()
                if trap_hit:
                    _log(
                        "pattern_block_long_trap",
                        tag=trap_tag,
                        rsi=round(rsi, 2),
                        atr=round(atr_pips, 2),
                        bbw=round(bbw, 5),
                    )
                    return None
            tp_dyn_adj = _adjust_tp(tp_dyn, confidence)
            signal = _apply_mult({
                "action": action,
                "sl_pips": sl_dyn,
                "tp_pips": tp_dyn_adj,
                "confidence": int(confidence * conf_scale),
                "fast_cut_pips": round(fast_cut, 2),
                "fast_cut_time_sec": int(fast_cut_time),
                "fast_cut_hard_mult": 1.6,
                "tag": f"{M1Scalper.name}-buy-dip" if action == "OPEN_LONG" else f"{M1Scalper.name}-trend-short",
            })
            return _attach_kill(signal)
        if momentum > momentum_thresh and rsi > rsi_short_min:
            speed = abs(momentum) / max(0.0005, atr)
            rsi_gap = max(0.0, rsi - 45) / 10
            confidence = int(
                max(40.0, min(95.0, 45.0 + speed * 30.0 + rsi_gap * 25.0))
            )
            action = "OPEN_SHORT"
            if trend_up:
                if range_reversion_only:
                    _log(
                        "range_hold_reversion_short",
                        trend="up",
                        range_score=round(range_score, 3),
                        mode=range_mode or "-",
                    )
                else:
                    # 強い上昇トレンドでは順張りロングに切替
                    action = "OPEN_LONG"
                    confidence = int(confidence * 0.9)
                    if rsi > rsi_trend_long_max:
                        _log("trend_block_long_rsi", rsi=round(rsi, 2))
                        return None
            if action == "OPEN_SHORT" and strong_up:
                _log("trend_block_short", momentum=round(momentum, 5), ema_gap=round(ema_gap_pips, 3), price_gap=round(price_gap_pips, 3))
                return None
            if action == "OPEN_SHORT":
                trap_hit, trap_tag = _short_trap_pattern()
                if trap_hit:
                    _log(
                        "pattern_block_short_trap",
                        tag=trap_tag,
                        rsi=round(rsi, 2),
                        atr=round(atr_pips, 2),
                        bbw=round(bbw, 5),
                    )
                    return None
            if action == "OPEN_LONG":
                trap_hit, trap_tag = _long_trap_pattern()
                if trap_hit:
                    _log(
                        "pattern_block_long_trap",
                        tag=trap_tag,
                        rsi=round(rsi, 2),
                        atr=round(atr_pips, 2),
                        bbw=round(bbw, 5),
                    )
                    return None
            tp_dyn_adj = _adjust_tp(tp_dyn, confidence)
            signal = _apply_mult({
                "action": action,
                "sl_pips": sl_dyn,
                "tp_pips": tp_dyn_adj,
                "confidence": int(confidence * conf_scale),
                "fast_cut_pips": round(fast_cut, 2),
                "fast_cut_time_sec": int(fast_cut_time),
                "fast_cut_hard_mult": 1.6,
                "tag": f"{M1Scalper.name}-sell-rally" if action == "OPEN_SHORT" else f"{M1Scalper.name}-trend-long",
            })
            return _attach_kill(signal)

        def _alignment_ok(side: str) -> bool:
            if not _NWAVE_ALIGN_ENABLED:
                return True
            if len(candles) < 2:
                return True
            last_body = _candle_body_pips(candles[-1])
            prev_body = _candle_body_pips(candles[-2])
            if last_body is None or prev_body is None:
                return True
            if side == "long":
                if _NWAVE_ALIGN_PREV_BODY_LONG <= 0.0 or _NWAVE_ALIGN_LAST_BODY_LONG <= 0.0:
                    return True
                return prev_body <= -_NWAVE_ALIGN_PREV_BODY_LONG and last_body >= _NWAVE_ALIGN_LAST_BODY_LONG
            if _NWAVE_ALIGN_PREV_BODY_SHORT <= 0.0 or _NWAVE_ALIGN_LAST_BODY_SHORT <= 0.0:
                return True
            return prev_body >= _NWAVE_ALIGN_PREV_BODY_SHORT and last_body <= -_NWAVE_ALIGN_LAST_BODY_SHORT

        story_levels = fac.get("story_levels") or {}
        d1_levels = story_levels.get("d1", {})
        h4_levels = story_levels.get("h4", {})

        def _level_bias(price: float) -> float:
            candidates = []
            for level in (*d1_levels.values(), *h4_levels.values()):
                val = _to_float(level)
                if val is None:
                    continue
                candidates.append(abs(price - val) / _PIP)
            if not candidates:
                return 0.0
            closest = min(candidates)
            if closest <= 6.0:
                return max(0.0, 1.0 - closest / 6.0)
            return 0.0

        if nwave:
            direction = nwave.direction
            entry_price = round(nwave.entry_price, 3)
            pullback_mult = _nwave_float("pullback_mult", 1.6)
            hard_sl_floor = _nwave_float("hard_sl_floor", 4.0)
            target_scale = _nwave_float("target_scale", 0.55)
            target_floor = _nwave_float("target_floor", 1.05)
            target_cap = _nwave_float("target_cap", 1.8)
            invalid_pips = max(hard_sl_floor, nwave.pullback_pips * pullback_mult)
            target_pips = max(target_floor, min(target_cap, nwave.amplitude_pips * target_scale))
            quality = nwave.quality
            proximity_bias = _level_bias(entry_price)
            base_conf = 55.0 + (min(quality, 2.0) * 20.0)
            base_conf -= proximity_bias * 15.0
            base_conf = max(40.0, min(96.0, base_conf))
            tolerance_default = max(_nwave_float("tolerance_default", 0.24), 0.42)
            tolerance_tactical = max(
                _nwave_float("tolerance_tactical", tolerance_default + 0.12),
                tolerance_default + 0.12,
            )
            tolerance_pips = tolerance_tactical if scalp_tactical else tolerance_default
            hard_sl_mult = _nwave_float("hard_sl_atr_mult", 1.8)
            hard_sl = max(invalid_pips, atr_pips * hard_sl_mult, hard_sl_floor)

            if direction == "long":
                if close > entry_price + tolerance_pips * _PIP:
                    _log(
                        "skip_nwave_long_late",
                        price=round(close, 3),
                        entry=entry_price,
                        tolerance=tolerance_pips,
                    )
                    return None
                if not _alignment_ok("long"):
                    if len(candles) >= 2:
                        _log(
                            "skip_nwave_long_alignment",
                            price=round(close, 3),
                            prev_body=round(_candle_body_pips(candles[-2]) or 0.0, 3),
                            last_body=round(_candle_body_pips(candles[-1]) or 0.0, 3),
                            enabled=_NWAVE_ALIGN_ENABLED,
                            prev_body_thr=_NWAVE_ALIGN_PREV_BODY_LONG,
                            last_body_thr=_NWAVE_ALIGN_LAST_BODY_LONG,
                        )
                    else:
                        _log("skip_nwave_long_alignment", price=round(close, 3))
                    return None
                signal = {
                    "action": "OPEN_LONG",
                    "entry_type": "limit",
                    "entry_price": entry_price,
                    "entry_tolerance_pips": tolerance_pips,
                    "limit_expiry_seconds": 120 if scalp_tactical else 150,
                    "sl_pips": round(hard_sl, 2),
                    "tp_pips": round(target_pips, 2),
                    "confidence": int(base_conf),
                    "fast_cut_pips": round(fast_cut, 2),
                    "fast_cut_time_sec": int(fast_cut_time),
                    "fast_cut_hard_mult": 1.6,
                    "tag": f"{M1Scalper.name}-nwave-long",
                }
                signal = _structure_targets(signal, fac, "long")
                signal = _apply_mult(signal)
                _log(
                    "signal_nwave_long",
                    entry=entry_price,
                    sl=signal["sl_pips"],
                    tp=signal["tp_pips"],
                    conf=signal["confidence"],
                    atr=round(atr_pips, 2),
                    rsi=round(rsi, 2),
                )
                return _attach_kill(signal)

            if close < entry_price - tolerance_pips * _PIP:
                _log(
                    "skip_nwave_short_late",
                    price=round(close, 3),
                    entry=entry_price,
                    tolerance=tolerance_pips,
                )
                return None
            if not _alignment_ok("short"):
                if len(candles) >= 2:
                    _log(
                        "skip_nwave_short_alignment",
                        price=round(close, 3),
                        prev_body=round(_candle_body_pips(candles[-2]) or 0.0, 3),
                        last_body=round(_candle_body_pips(candles[-1]) or 0.0, 3),
                        enabled=_NWAVE_ALIGN_ENABLED,
                        prev_body_thr=_NWAVE_ALIGN_PREV_BODY_SHORT,
                        last_body_thr=_NWAVE_ALIGN_LAST_BODY_SHORT,
                    )
                else:
                    _log("skip_nwave_short_alignment", price=round(close, 3))
                return None
            signal = {
                "action": "OPEN_SHORT",
                "entry_type": "limit",
                "entry_price": entry_price,
                "entry_tolerance_pips": tolerance_pips,
                "limit_expiry_seconds": 120 if scalp_tactical else 150,
                "sl_pips": round(hard_sl, 2),
                "tp_pips": round(target_pips, 2),
                "confidence": int(base_conf),
                "fast_cut_pips": round(fast_cut, 2),
                "fast_cut_time_sec": int(fast_cut_time),
                "fast_cut_hard_mult": 1.6,
                "tag": f"{M1Scalper.name}-nwave-short",
            }
            signal = _structure_targets(signal, fac, "short")
            signal = _apply_mult(signal)
            _log(
                "signal_nwave_short",
                entry=entry_price,
                sl=signal["sl_pips"],
                tp=signal["tp_pips"],
                conf=signal["confidence"],
                atr=round(atr_pips, 2),
                rsi=round(rsi, 2),
            )
            return _attach_kill(signal)

        # Fallback microstructure scalp (limit entry every cycle)
        fallback_enabled = _to_bool(fallback_cfg.get("enabled", True), True) or _force_mode()
        if fallback_enabled:
            ticks = fac.get("recent_ticks") or []
            summary = fac.get("recent_tick_summary") or {}
            if not ticks:
                global _last_no_tick_log_ts
                now_ts = time.time()
                if now_ts - _last_no_tick_log_ts >= _EMPTY_TICK_LOG_DEBOUNCE_SEC:
                    msg = "no_recent_ticks"
                    if _force_mode():
                        _LOGGER.warning(
                            "[FORCE_SCALP] M1Scalper %s atr=%.2f rsi=%.2f",
                            msg,
                            round(atr_pips, 2),
                            round(rsi, 2),
                        )
                    else:
                        _log(
                            msg,
                            atr=round(atr_pips, 2),
                            rsi=round(rsi, 2),
                        )
                    _last_no_tick_log_ts = now_ts
            if ticks:
                try:
                    mid_latest = float(ticks[-1]["mid"])
                except (TypeError, ValueError):
                    mid_latest = close
                if summary:
                    high_mid = float(summary.get("high_mid", mid_latest) or mid_latest)
                    low_mid = float(summary.get("low_mid", mid_latest) or mid_latest)
                else:
                    highs = [t.get("mid") for t in ticks if t.get("mid") is not None]
                    if highs:
                        high_mid = max(float(x) for x in highs)
                        low_mid = min(float(x) for x in highs)
                    else:
                        high_mid = mid_latest
                        low_mid = mid_latest
            else:
                mid_latest = close
                span = max(0.0008, min(0.003, abs(momentum) * 4 or 0.0012))
                high_mid = mid_latest + span
                low_mid = mid_latest - span

            span_mid = max(high_mid - low_mid, 0.0002)
            span_pips = max(0.2, min(4.0, span_mid / _PIP))
            dist_high_pips = max((high_mid - mid_latest) / _PIP, 0.05)
            dist_low_pips = max((mid_latest - low_mid) / _PIP, 0.05)
            min_span_pips = _fallback_float("min_span_pips", 0.8)
            if span_pips < min_span_pips:
                _log("skip_fallback_span", span=round(span_pips, 2), min_span=min_span_pips)
                return None

            base_tp_floor = _fallback_float("tp_floor", 1.0)
            tp_cap = _fallback_float("tp_cap", 1.9)
            if scalp_tactical:
                tp_cap = _fallback_float("tp_cap_tactical", tp_cap)
            tp_dyn = max(base_tp_floor, min(tp_cap, (span_pips * 0.5) + 0.5))
            if scalp_tactical:
                tp_dyn = min(tp_cap, max(0.9, tp_dyn * 0.9))

            sl_floor = _fallback_float("sl_floor", 6.0)
            if scalp_tactical:
                sl_floor = _fallback_float("sl_floor_tactical", sl_floor)
            sl_cap = _fallback_float("sl_cap", 13.0)
            if scalp_tactical:
                sl_cap = _fallback_float("sl_cap_tactical", sl_cap)
            sl_mult = _fallback_float("sl_atr_mult", 2.2)
            if scalp_tactical:
                sl_mult = _fallback_float("sl_atr_mult_tactical", sl_mult)
            sl_dyn = max(sl_floor, atr_pips * sl_mult)
            sl_dyn = min(sl_cap, sl_dyn)

            entry_base = span_pips * 0.35
            entry_offset_pips = max(0.05, min(0.28, entry_base))
            tolerance_pips = max(0.05, min(0.35, entry_offset_pips * 1.2))

            if dist_low_pips <= dist_high_pips:
                direction = "long"
            else:
                direction = "short"

            if direction == "long":
                entry_price = round(mid_latest - entry_offset_pips * _PIP, 3)
                floor_price = round(low_mid + 0.0004, 3)
                if entry_price < floor_price:
                    entry_price = floor_price
            else:
                entry_price = round(mid_latest + entry_offset_pips * _PIP, 3)
                cap_price = round(high_mid - 0.0004, 3)
                if entry_price > cap_price:
                    entry_price = cap_price

            mom_norm = abs(momentum) / max(0.0001, atr or 0.0001)
            min_mom_norm = _fallback_float("min_mom_norm", 0.7)
            if mom_norm < min_mom_norm and abs(momentum) < momentum_thresh:
                _log("skip_fallback_momentum", mom=round(momentum, 5), mom_norm=round(mom_norm, 2))
                return None
            rsi_bias = abs(rsi - 50.0) / 25.0
            if rsi_bias < rsi_edge_min:
                _log("skip_fallback_rsi_edge", rsi=round(rsi, 2), edge=round(rsi_bias, 2))
                return None
            conf_base = 58.0 + (mom_norm * 18.0) + (rsi_bias * 14.0)
            if scalp_tactical:
                conf_base += 6.0
            conf_base += max(0.0, (span_pips - 0.6) * 2.5)
            confidence = int(max(48.0, min(96.0, conf_base)))

            action = "OPEN_LONG" if direction == "long" else "OPEN_SHORT"
            # トレンドと逆なら方向をスイッチ
            if action == "OPEN_LONG" and trend_down:
                if not range_reversion_only:
                    action = "OPEN_SHORT"
                    confidence = int(confidence * 0.9)
            elif action == "OPEN_SHORT" and trend_up:
                if not range_reversion_only:
                    action = "OPEN_LONG"
                    confidence = int(confidence * 0.9)
            entry_type = "limit"
            limit_expiry = 35 if scalp_tactical else 50
            entry_price_out = entry_price
            if _force_mode():
                entry_type = "market"
                entry_price_out = round(mid_latest, 3)
                tolerance_pips = 0.0
                limit_expiry = 0
                confidence = max(confidence, 72)
            signal = {
                "action": action,
                "entry_type": entry_type,
                "entry_price": entry_price_out,
                "entry_tolerance_pips": round(tolerance_pips, 2),
                "limit_expiry_seconds": limit_expiry,
                "sl_pips": round(sl_dyn, 2),
                "tp_pips": round(tp_dyn, 2),
                "confidence": confidence,
                "tag": f"{M1Scalper.name}-micro-{direction}",
                "notes": {
                    "span_pips": round(span_pips, 2),
                    "dist_high": round(dist_high_pips, 2),
                    "dist_low": round(dist_low_pips, 2),
                },
            }
            signal = _structure_targets(signal, fac, "long" if action == "OPEN_LONG" else "short")
            signal = _apply_mult(signal)
            if _force_mode():
                _LOGGER.warning("[FORCE_SCALP] issuing market signal %s", signal)
            _log(
                "signal_micro_limit",
                direction=direction,
                entry=signal["entry_price"],
                tp=signal["tp_pips"],
                sl=signal["sl_pips"],
                span=round(span_pips, 2),
                momentum=round(momentum, 5),
                rsi=round(rsi, 2),
                ticks=len(ticks),
                tech_mult=tech_mult_r,
            )
            return _attach_kill(signal)

        _log(
            "skip_no_trigger",
            momentum=round(momentum, 5),
            rsi=round(rsi, 2),
            tactical=scalp_tactical,
        )
        return None
