"""Worker that combines H1 trend confirmation with an M5 breakout trigger."""

from __future__ import annotations
from analysis.ma_projection import compute_adx_projection, compute_bbw_projection, compute_ma_projection, compute_rsi_projection
from analysis.ma_projection import score_ma_for_side
from indicators.factor_cache import all_factors, get_candles_snapshot

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

import statistics

# Optional higher-level dependencies. The worker keeps graceful fallbacks so it can
# operate in isolation (e.g. during replay runs) even if these modules are absent.
try:  # pragma: no cover - optional dependency
    from strategies.trend.h1_momentum import signal as h1_trend_signal
except Exception:  # pragma: no cover - best-effort import
    h1_trend_signal = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from execution.exit_manager import ExitManager
except Exception:  # pragma: no cover - best-effort import
    ExitManager = None  # type: ignore[assignment]


class DataFeed(Protocol):
    """Minimal interface required from upstream data providers."""

    def get_bars(self, symbol: str, timeframe: str, count: int) -> Any: ...

    def last(self, symbol: str) -> float: ...


class Broker(Protocol):
    """Minimal broker interface so the worker can place market orders."""

    def send(self, order: Dict[str, Any]) -> Any: ...


@dataclass
class TradeIntent:
    """Signal payload emitted by the worker."""

    symbol: str
    side: str  # "long" or "short"
    strength: float  # 0..1
    entry_px: Optional[float]
    meta: Dict[str, Any]


from . import config
from utils.env_utils import env_bool, env_float

_BB_ENV_PREFIX = getattr(config, "ENV_PREFIX", "")
_BB_ENTRY_ENABLED = env_bool("BB_ENTRY_ENABLED", True, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_REVERT_PIPS = env_float("BB_ENTRY_REVERT_PIPS", 2.4, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_REVERT_RATIO = env_float("BB_ENTRY_REVERT_RATIO", 0.22, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_TREND_EXT_PIPS = env_float("BB_ENTRY_TREND_EXT_PIPS", 3.5, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_TREND_EXT_RATIO = env_float("BB_ENTRY_TREND_EXT_RATIO", 0.40, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_SCALP_REVERT_PIPS = env_float("BB_ENTRY_SCALP_REVERT_PIPS", 2.0, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_SCALP_REVERT_RATIO = env_float("BB_ENTRY_SCALP_REVERT_RATIO", 0.20, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_SCALP_EXT_PIPS = env_float("BB_ENTRY_SCALP_EXT_PIPS", 2.4, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_SCALP_EXT_RATIO = env_float("BB_ENTRY_SCALP_EXT_RATIO", 0.30, prefix=_BB_ENV_PREFIX)
_BB_PIP = 0.01


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


def _bb_entry_allowed(style, side, price, fac_m1, *, range_active=None):
    if not _BB_ENTRY_ENABLED:
        return True
    if price is None or price <= 0:
        return True
    levels = _bb_levels(fac_m1)
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
    if style == "reversion":
        base_pips = _BB_ENTRY_SCALP_REVERT_PIPS if orig_style == "scalp" else _BB_ENTRY_REVERT_PIPS
        base_ratio = _BB_ENTRY_SCALP_REVERT_RATIO if orig_style == "scalp" else _BB_ENTRY_REVERT_RATIO
        threshold = max(base_pips, span_pips * base_ratio)
        if direction == "long":
            dist = (price - lower) / _BB_PIP
        else:
            dist = (upper - price) / _BB_PIP
        return dist <= threshold
    if direction == "long":
        if price < mid:
            return False
        ext = max(0.0, price - upper) / _BB_PIP
    else:
        if price > mid:
            return False
        ext = max(0.0, lower - price) / _BB_PIP
    max_ext = max(_BB_ENTRY_TREND_EXT_PIPS, span_pips * _BB_ENTRY_TREND_EXT_RATIO)
    if orig_style == "scalp":
        max_ext = max(_BB_ENTRY_SCALP_EXT_PIPS, span_pips * _BB_ENTRY_SCALP_EXT_RATIO)
    return ext <= max_ext

BB_STYLE = "trend"



_PROJ_TF_MINUTES = {"M1": 1.0, "M5": 5.0, "H1": 60.0, "H4": 240.0, "D1": 1440.0}


def _projection_mode(pocket, mode_override=None):
    if mode_override:
        return mode_override
    if globals().get("IS_RANGE"):
        return "range"
    if globals().get("IS_PULLBACK"):
        return "pullback"
    if pocket in {"scalp", "scalp_fast"}:
        return "scalp"
    return "trend"


def _projection_tfs(pocket, mode):
    if pocket == "macro":
        return ("H4", "H1")
    if pocket == "micro":
        return ("M5", "M1")
    if pocket in {"scalp", "scalp_fast"}:
        return ("M1",)
    return ("M5", "M1")


def _projection_candles(tfs):
    for tf in tfs:
        candles = get_candles_snapshot(tf, limit=120)
        if candles and len(candles) >= 30:
            return tf, list(candles)
    return None, None


def _score_ma(ma, side, opp_block_bars):
    if ma is None:
        return None
    return score_ma_for_side(ma, side, opp_block_bars)


def _score_rsi(rsi, side, long_target, short_target, overheat_bars):
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


def _score_adx(adx, trend_mode, threshold):
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


def _score_bbw(bbw, threshold):
    if bbw is None:
        return None
    if bbw.bbw <= threshold and bbw.slope_per_bar <= 0:
        return 0.5
    if bbw.bbw > threshold and bbw.slope_per_bar > 0:
        return -0.5
    return 0.0


def _projection_decision(side, pocket, mode_override=None):
    mode = _projection_mode(pocket, mode_override=mode_override)
    tfs = _projection_tfs(pocket, mode)
    tf, candles = _projection_candles(tfs)
    if not candles:
        return True, 1.0, {}
    minutes = _PROJ_TF_MINUTES.get(tf, 1.0)

    if mode == "trend":
        params = {
            "adx_threshold": 20.0,
            "bbw_threshold": 0.16,
            "opp_block_bars": 5.0,
            "long_target": 52.0,
            "short_target": 48.0,
            "overheat_bars": 3.0,
            "weights": {"ma": 0.45, "rsi": 0.25, "adx": 0.30},
            "block_score": -0.6,
            "size_scale": 0.18,
        }
    elif mode == "pullback":
        params = {
            "adx_threshold": 18.0,
            "bbw_threshold": 0.16,
            "opp_block_bars": 4.0,
            "long_target": 50.0,
            "short_target": 50.0,
            "overheat_bars": 3.0,
            "weights": {"ma": 0.40, "rsi": 0.40, "adx": 0.20},
            "block_score": -0.55,
            "size_scale": 0.15,
        }
    elif mode == "scalp":
        params = {
            "adx_threshold": 18.0,
            "bbw_threshold": 0.16,
            "opp_block_bars": 3.0,
            "long_target": 52.0,
            "short_target": 48.0,
            "overheat_bars": 2.0,
            "weights": {"ma": 0.50, "rsi": 0.30, "adx": 0.20},
            "block_score": -0.6,
            "size_scale": 0.12,
        }
    else:
        params = {
            "adx_threshold": 16.0,
            "bbw_threshold": 0.14,
            "opp_block_bars": 4.0,
            "long_target": 45.0,
            "short_target": 55.0,
            "overheat_bars": 3.0,
            "weights": {"bbw": 0.40, "rsi": 0.35, "adx": 0.25},
            "block_score": -0.5,
            "size_scale": 0.15,
        }

    ma = compute_ma_projection({"candles": candles}, timeframe_minutes=minutes)
    rsi = compute_rsi_projection(candles, timeframe_minutes=minutes)
    adx = compute_adx_projection(candles, timeframe_minutes=minutes, trend_threshold=params["adx_threshold"])
    bbw = None
    if mode == "range":
        bbw = compute_bbw_projection(candles, timeframe_minutes=minutes, squeeze_threshold=params["bbw_threshold"])

    scores = {}
    ma_score = _score_ma(ma, side, params["opp_block_bars"])
    if ma_score is not None and "ma" in params["weights"]:
        scores["ma"] = ma_score
    rsi_score = _score_rsi(rsi, side, params["long_target"], params["short_target"], params["overheat_bars"])
    if rsi_score is not None and "rsi" in params["weights"]:
        scores["rsi"] = rsi_score
    adx_score = _score_adx(adx, mode != "range", params["adx_threshold"])
    if adx_score is not None and "adx" in params["weights"]:
        scores["adx"] = adx_score
    bbw_score = _score_bbw(bbw, params["bbw_threshold"])
    if bbw_score is not None and "bbw" in params["weights"]:
        scores["bbw"] = bbw_score

    weight_sum = 0.0
    score_sum = 0.0
    for key, score in scores.items():
        weight = params["weights"].get(key, 0.0)
        weight_sum += weight
        score_sum += weight * score
    score = score_sum / weight_sum if weight_sum > 0 else 0.0

    allow = score > params["block_score"]
    size_mult = 1.0 + max(0.0, score) * params["size_scale"]
    size_mult = max(0.8, min(1.35, size_mult))

    detail = {
        "mode": mode,
        "tf": tf,
        "score": round(score, 3),
        "size_mult": round(size_mult, 3),
        "scores": {k: round(v, 3) for k, v in scores.items()},
    }
    return allow, size_mult, detail
def _as_list_of_dicts(bars: Any) -> List[Dict[str, float]]:
    """Best-effort normalisation of bar data into dictionaries."""

    def _extract(item: Any, key: str, fallback_index: int | None = None) -> float:
        if isinstance(item, dict):
            val = item.get(key)
        else:
            val = getattr(item, key, None)
            if val is None and fallback_index is not None and isinstance(item, (list, tuple)):
                try:
                    val = item[fallback_index]
                except IndexError:
                    val = None
        if val is None:
            raise ValueError(f"missing {key}")
        return float(val)

    normalised: List[Dict[str, float]] = []
    if bars is None:
        return normalised
    if isinstance(bars, list):
        for item in bars:
            try:
                normalised.append(
                    {
                        "open": _extract(item, "open", 0),
                        "high": _extract(item, "high", 1),
                        "low": _extract(item, "low", 2),
                        "close": _extract(item, "close", 3),
                    }
                )
            except Exception:
                continue
        return normalised
    try:
        open_vals = bars["open"]
        high_vals = bars["high"]
        low_vals = bars["low"]
        close_vals = bars["close"]
        length = len(close_vals)
        for idx in range(length):
            normalised.append(
                {
                    "open": float(open_vals[idx]),
                    "high": float(high_vals[idx]),
                    "low": float(low_vals[idx]),
                    "close": float(close_vals[idx]),
                }
            )
    except Exception:
        return []
    return normalised


def _highest_high(bars: List[Dict[str, float]]) -> float:
    return max((b["high"] for b in bars), default=float("-inf"))


def _lowest_low(bars: List[Dict[str, float]]) -> float:
    return min((b["low"] for b in bars), default=float("inf"))


def _argmax_high(bars: List[Dict[str, float]]) -> int:
    idx = -1
    value = float("-inf")
    for i, bar in enumerate(bars):
        if bar["high"] > value:
            value = bar["high"]
            idx = i
    return idx


def _argmin_low(bars: List[Dict[str, float]]) -> int:
    idx = -1
    value = float("inf")
    for i, bar in enumerate(bars):
        if bar["low"] < value:
            value = bar["low"]
            idx = i
    return idx


def _atr(bars: List[Dict[str, float]], period: int = 14) -> float:
    if len(bars) < period + 1:
        return 0.0
    ranges: List[float] = []
    for i in range(1, period + 1):
        h = bars[-i]["high"]
        l = bars[-i]["low"]
        prev_close = bars[-i - 1]["close"]
        ranges.append(max(h - l, abs(h - prev_close), abs(l - prev_close)))
    return sum(ranges) / len(ranges)


LOG = logging.getLogger(__name__)


class MtfBreakoutWorker:
    """H1 trend filter combined with an M5 breakout entry trigger."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        broker: Optional[Broker],
        datafeed: DataFeed,
        logger: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.broker = broker
        self.datafeed = datafeed
        self.log = logger

        self.tf_entry = str(cfg.get("timeframe_entry", "M5"))
        self.tf_trend = str(cfg.get("timeframe_trend", "H1"))
        self.lookback = int(cfg.get("breakout_lookback", 20))
        self.pullback_min = max(0, int(cfg.get("pullback_min_bars", 2)))
        self.cooldown_bars = max(0, int(cfg.get("cooldown_bars", 3)))
        self.edge_threshold = float(cfg.get("edge_threshold", 0.25))
        self.place_orders = bool(cfg.get("place_orders", False))
        self.budget_bps = float(cfg.get("budget_bps", 25.0))
        self.min_trend_bars = max(10, int(cfg.get("trend_min_bars", 48)))
        self._last_entry_bar: Dict[str, int] = {}
        self.max_concurrent = int(cfg.get("max_concurrent", 3))
        self.pocket = str(cfg.get("pocket", "micro"))

        # ExitManager currently handles live position monitoring and does not expose
        # attach hooks, but we keep the import so future integrations can wire it in.
        self.exit_manager = ExitManager() if ExitManager else None  # pragma: no cover

    # --------------------------------------------------------------------- public
    def run_once(self) -> List[TradeIntent]:
        """Scan configured symbols and emit breakout intents."""
        intents: List[TradeIntent] = []
        universe = self.cfg.get("universe") or []
        if not isinstance(universe, list):
            return intents
        for symbol in universe:
            if self.max_concurrent > 0 and len(intents) >= self.max_concurrent:
                break
            try:
                intent = self.edge(str(symbol))
            except Exception as exc:  # pragma: no cover - protective belt
                if self.log:
                    self.log.warning("mtf_breakout edge error: %s", exc, exc_info=True)
                continue
            if not intent:
                continue
            proj_allow, proj_mult, proj_detail = _projection_decision(intent.side, self.pocket)
            if not proj_allow:
                continue
            if proj_detail:
                intent.meta["projection"] = proj_detail
            intents.append(intent)
            if self.place_orders and self.broker is not None:
                order = self._mk_order(intent, proj_mult)
                self.broker.send(order)
        return intents

    def edge(self, symbol: str) -> Optional[TradeIntent]:
        """Return a TradeIntent when all pre-conditions align."""
        h1_bars = _as_list_of_dicts(self.datafeed.get_bars(symbol, self.tf_trend, 300))
        if len(h1_bars) < self.min_trend_bars:
            return None
        side, trend_strength = self._trend_side(h1_bars)
        if side is None or trend_strength < self.edge_threshold:
            return None

        m5_bars = _as_list_of_dicts(
            self.datafeed.get_bars(symbol, self.tf_entry, max(self.lookback + 5, 60))
        )
        if len(m5_bars) < self.lookback + 2:
            return None
        current_idx = len(m5_bars) - 1
        if self._is_cooldown(symbol, current_idx):
            return None
        window = m5_bars[-(self.lookback + 1) : -1]
        last_close = m5_bars[-1]["close"]
        if side == "long":
            extreme = _highest_high(window)
            bars_since = current_idx - 1 - _argmax_high(window)
            breakout = last_close > extreme and bars_since >= self.pullback_min
        else:
            extreme = _lowest_low(window)
            bars_since = current_idx - 1 - _argmin_low(window)
            breakout = last_close < extreme and bars_since >= self.pullback_min
        if not breakout:
            return None

        fac_m1 = all_factors().get("M1") or {}
        if not _bb_entry_allowed(BB_STYLE, side, last_close, fac_m1):
            return None

        atr_val = _atr(m5_bars, 14)
        strength = min(1.0, max(0.0, 0.5 * trend_strength + 0.5))
        intent = TradeIntent(
            symbol=symbol,
            side=side,
            strength=strength,
            entry_px=last_close,
            meta={
                "atr": atr_val,
                "timeframe_entry": self.tf_entry,
                "timeframe_trend": self.tf_trend,
                "bars_since_extreme": bars_since,
                "extreme": extreme,
            },
        )
        self._last_entry_bar[symbol] = current_idx
        return intent

    # ------------------------------------------------------------------ internals
    def _trend_side(self, bars: List[Dict[str, float]]) -> tuple[Optional[str], float]:
        if h1_trend_signal is not None:  # pragma: no cover - optional path
            try:
                payload = h1_trend_signal(bars)
                side = payload.get("side")
                strength = float(payload.get("strength", 0.0))
                if side in {"long", "short"} and 0.0 <= strength <= 1.0:
                    return side, strength
            except Exception:
                pass
        closes = [bar["close"] for bar in bars]
        min_required = max(self.min_trend_bars, 20)
        if len(closes) < min_required:
            return None, 0.0
        ema_fast = self._ema(closes, min(12, len(closes)))
        ema_slow = self._ema(closes, min(40, len(closes)))
        diff = [f - s for f, s in zip(ema_fast, ema_slow)]
        if not diff:
            return None, 0.0
        lookback = min(5, len(diff) - 1) if len(diff) > 1 else 0
        slope = diff[-1] - diff[-1 - lookback] if lookback > 0 else diff[-1]
        dispersion_window = min(len(diff), max(20, self.min_trend_bars))
        if len(diff) >= 2 and dispersion_window >= 2:
            dispersion = statistics.pstdev(diff[-dispersion_window:])
        elif len(diff) >= 2:
            dispersion = statistics.pstdev(diff)
        else:
            dispersion = 0.0
        if dispersion <= 0.0:
            dispersion = 1.0
        strength = min(1.0, max(0.0, abs(slope) / (dispersion + 1e-8)))
        if diff[-1] > 0:
            return "long", strength
        if diff[-1] < 0:
            return "short", strength
        return None, 0.0

    def _ema(self, values: List[float], period: int) -> List[float]:
        if not values:
            return []
        k = 2.0 / (period + 1.0)
        ema_vals: List[float] = []
        ema_prev = values[0]
        for price in values:
            ema_prev = price * k + ema_prev * (1.0 - k)
            ema_vals.append(ema_prev)
        return ema_vals

    def _is_cooldown(self, symbol: str, current_idx: int) -> bool:
        last = self._last_entry_bar.get(symbol)
        if last is None:
            return False
        return current_idx - last < self.cooldown_bars

    def _mk_order(self, intent: TradeIntent, size_mult: float = 1.0) -> Dict[str, Any]:
        entry_px = intent.entry_px or self.datafeed.last(intent.symbol)
        side = "buy" if intent.side == "long" else "sell"
        size = max(0.0, self.budget_bps / 10000.0)
        if size_mult > 1.0:
            size *= size_mult
        return {
            "symbol": intent.symbol,
            "side": side,
            "type": "market",
            "size": size,
            "price": entry_px,
            "metadata": {
                "worker_id": self.cfg.get("id", "mtf_breakout"),
                "strength": intent.strength,
                "timeframes": {
                    "entry": self.tf_entry,
                    "trend": self.tf_trend,
                },
            },
        }


async def _idle_loop() -> None:
    """Keep the systemd service alive when no live wiring is present."""
    LOG.info("mtf_breakout worker idle loop started (no live broker/datafeed wiring)")
    try:
        while True:
            await asyncio.sleep(3600.0)
    except asyncio.CancelledError:  # pragma: no cover - service shutdown path
        LOG.info("mtf_breakout worker idle loop cancelled")
        raise


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    from .config import DEFAULT_CONFIG
    from workers.common import addon_live

    cfg = addon_live.apply_env_overrides(
        "MTF_BREAKOUT",
        DEFAULT_CONFIG,
        default_universe=["USD_JPY"],
        default_pocket="micro",
        default_loop=30.0,
    )
    if not cfg.get("live_enabled"):
        LOG.info("mtf_breakout idle mode (set MTF_BREAKOUT_LIVE=1 or ADDON_LIVE_MODE=1 to enable)")
        asyncio.run(_idle_loop())
        raise SystemExit(0)

    datafeed = addon_live.LiveDataFeed(
        default_timeframe=str(cfg.get("timeframe_entry", "M5")),
        logger=LOG,
    )
    broker = addon_live.AddonLiveBroker(
        worker_id=str(cfg.get("id", "mtf_breakout")),
        pocket=str(cfg.get("pocket", "micro")),
        datafeed=datafeed,
        exit_cfg=cfg.get("exit"),
        atr_len=int(cfg.get("atr_len", 14)),
        atr_timeframe=str(cfg.get("timeframe_entry", "M5")),
        default_budget_bps=cfg.get("budget_bps"),
        ttl_ms=float(cfg.get("ttl_ms", 800.0)),
        require_passive=bool(cfg.get("require_passive", True)),
        logger=LOG,
    )
    worker = MtfBreakoutWorker(cfg, broker=broker, datafeed=datafeed, logger=LOG)
    addon_live.run_loop(
        worker,
        loop_interval_sec=float(cfg.get("loop_interval_sec", 30.0)),
        pocket=str(cfg.get("pocket", "micro")),
        logger=LOG,
    )
