"""Worker that combines H1 trend confirmation with an M5 breakout trigger."""

from __future__ import annotations

import asyncio
import logging
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
            intents.append(intent)
            if self.place_orders and self.broker is not None:
                order = self._mk_order(intent)
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

    def _mk_order(self, intent: TradeIntent) -> Dict[str, Any]:
        entry_px = intent.entry_px or self.datafeed.last(intent.symbol)
        side = "buy" if intent.side == "long" else "sell"
        size = max(0.0, self.budget_bps / 10000.0)
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
    asyncio.run(_idle_loop())
