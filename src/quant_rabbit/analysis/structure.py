"""Price-action / Smart-Money-Concepts structural analysis.

Computes:
- Swing high/low pivots (fractal style)
- Break of Structure (BOS) and Change of Character (CHoCH)
- Bullish / bearish Order Blocks (last opposite-color candle before an impulsive move)
- Fair Value Gaps (FVG) — 3-candle imbalance
- Liquidity pools — equal highs / equal lows that act as stop-magnet zones

All outputs are deterministic and pure-Python so the trader can cite specific
prices in the decision receipt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from quant_rabbit.analysis.candles import Candle


@dataclass(frozen=True)
class SwingPivot:
    index: int
    timestamp_iso: str
    price: float
    side: str  # "HIGH" or "LOW"


@dataclass(frozen=True)
class StructureEvent:
    index: int
    timestamp_iso: str
    kind: str  # "BOS_UP" / "BOS_DOWN" / "CHOCH_UP" / "CHOCH_DOWN"
    broken_pivot_price: float
    # True when the candle that printed the new swing pivot ALSO closed
    # beyond the broken pivot price (real structural break). False when
    # only the wick of the new pivot pierced the prior pivot — the classic
    # stop-hunt / liquidity-sweep pattern that should NOT authorize CLOSE
    # Gate A on its own. Default True for backward compatibility with
    # legacy fixtures and existing test snapshots.
    close_confirmed: bool = True


@dataclass(frozen=True)
class OrderBlock:
    index: int
    timestamp_iso: str
    side: str  # "BULL" (demand) or "BEAR" (supply)
    high: float
    low: float


@dataclass(frozen=True)
class FairValueGap:
    index: int
    timestamp_iso: str
    direction: str  # "UP" / "DOWN"
    upper: float
    lower: float
    filled: bool


@dataclass(frozen=True)
class LiquidityCluster:
    side: str  # "EQ_HIGH" / "EQ_LOW"
    price: float
    indices: tuple[int, ...]


@dataclass(frozen=True)
class StructureReading:
    swings: tuple[SwingPivot, ...] = field(default_factory=tuple)
    structure_events: tuple[StructureEvent, ...] = field(default_factory=tuple)
    last_event: StructureEvent | None = None
    order_blocks: tuple[OrderBlock, ...] = field(default_factory=tuple)
    fair_value_gaps: tuple[FairValueGap, ...] = field(default_factory=tuple)
    liquidity: tuple[LiquidityCluster, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "swings": [
                {"index": s.index, "timestamp": s.timestamp_iso, "price": s.price, "side": s.side}
                for s in self.swings
            ],
            "structure_events": [
                {"index": e.index, "timestamp": e.timestamp_iso, "kind": e.kind,
                 "broken_pivot_price": e.broken_pivot_price,
                 "close_confirmed": e.close_confirmed}
                for e in self.structure_events
            ],
            "last_event": (
                {"index": self.last_event.index, "timestamp": self.last_event.timestamp_iso,
                 "kind": self.last_event.kind,
                 "broken_pivot_price": self.last_event.broken_pivot_price,
                 "close_confirmed": self.last_event.close_confirmed}
                if self.last_event else None
            ),
            "order_blocks": [
                {"index": ob.index, "timestamp": ob.timestamp_iso, "side": ob.side,
                 "high": ob.high, "low": ob.low}
                for ob in self.order_blocks
            ],
            "fair_value_gaps": [
                {"index": fvg.index, "timestamp": fvg.timestamp_iso, "direction": fvg.direction,
                 "upper": fvg.upper, "lower": fvg.lower, "filled": fvg.filled}
                for fvg in self.fair_value_gaps
            ],
            "liquidity": [
                {"side": lc.side, "price": lc.price, "indices": list(lc.indices)}
                for lc in self.liquidity
            ],
        }


def analyze_structure(
    candles: Sequence[Candle],
    *,
    pivot_strength: int = 3,
    impulse_atr_mult: float = 1.0,
    fvg_lookback: int = 60,
    eq_tolerance_pips: float = 1.5,
    pip_size: float = 0.01,
    max_swings_kept: int = 12,
    max_obs_kept: int = 6,
    max_fvgs_kept: int = 6,
) -> StructureReading:
    """Run the full structural pass and return a `StructureReading`.

    `pivot_strength` is the fractal radius (3 = pivot must be the highest/lowest
    of 3 bars on each side). `impulse_atr_mult` controls Order Block detection
    (the impulse move that anchors the OB must be at least N×ATR).
    """

    n = len(candles)
    if n < pivot_strength * 2 + 1:
        return StructureReading()

    swings = _detect_swings(candles, strength=pivot_strength)
    events = _detect_structure_events(swings, candles)
    obs = _detect_order_blocks(candles, events, impulse_atr_mult=impulse_atr_mult)
    fvgs = _detect_fvg(candles, lookback=fvg_lookback)
    liquidity = _detect_liquidity(swings, eq_tolerance_pips=eq_tolerance_pips, pip_size=pip_size)

    return StructureReading(
        swings=tuple(swings[-max_swings_kept:]),
        structure_events=tuple(events[-max_swings_kept:]),
        last_event=events[-1] if events else None,
        order_blocks=tuple(obs[-max_obs_kept:]),
        fair_value_gaps=tuple(fvgs[-max_fvgs_kept:]),
        liquidity=tuple(liquidity),
    )


# ---------------------------------------------------------------------------
# Detection primitives
# ---------------------------------------------------------------------------


def _detect_swings(candles: Sequence[Candle], *, strength: int = 3) -> list[SwingPivot]:
    n = len(candles)
    swings: list[SwingPivot] = []
    for i in range(strength, n - strength):
        h = candles[i].high
        l = candles[i].low
        is_high = all(candles[j].high <= h for j in range(i - strength, i + strength + 1) if j != i)
        is_low = all(candles[j].low >= l for j in range(i - strength, i + strength + 1) if j != i)
        ts = candles[i].timestamp_utc.isoformat()
        if is_high:
            swings.append(SwingPivot(index=i, timestamp_iso=ts, price=h, side="HIGH"))
        if is_low:
            swings.append(SwingPivot(index=i, timestamp_iso=ts, price=l, side="LOW"))
    return swings


def _detect_structure_events(swings: list[SwingPivot], candles: Sequence[Candle]) -> list[StructureEvent]:
    """Walk swings forward and tag BOS / CHoCH whenever price breaks a prior pivot.

    BOS = continuation in the same direction as previous trend.
    CHoCH = first break in the opposite direction (regime shift).
    """
    if not swings:
        return []
    events: list[StructureEvent] = []
    last_trend: str | None = None  # "UP" or "DOWN"
    last_high: SwingPivot | None = None
    last_low: SwingPivot | None = None
    n_candles = len(candles)
    for piv in swings:
        if piv.side == "HIGH":
            if last_high is not None and piv.price > last_high.price:
                kind = "BOS_UP" if last_trend == "UP" else "CHOCH_UP"
                # The new swing pivot's candle close confirms the break only
                # when it also clears the broken pivot price. Otherwise the
                # high is a wick — the prior pivot was swept but the market
                # closed back inside the range (stop-hunt pattern).
                close_confirmed = (
                    0 <= piv.index < n_candles
                    and candles[piv.index].close > last_high.price
                )
                events.append(StructureEvent(
                    index=piv.index, timestamp_iso=piv.timestamp_iso, kind=kind,
                    broken_pivot_price=last_high.price,
                    close_confirmed=close_confirmed,
                ))
                last_trend = "UP"
            last_high = piv
        else:
            if last_low is not None and piv.price < last_low.price:
                kind = "BOS_DOWN" if last_trend == "DOWN" else "CHOCH_DOWN"
                close_confirmed = (
                    0 <= piv.index < n_candles
                    and candles[piv.index].close < last_low.price
                )
                events.append(StructureEvent(
                    index=piv.index, timestamp_iso=piv.timestamp_iso, kind=kind,
                    broken_pivot_price=last_low.price,
                    close_confirmed=close_confirmed,
                ))
                last_trend = "DOWN"
            last_low = piv
    return events


def _detect_order_blocks(
    candles: Sequence[Candle], events: list[StructureEvent], *, impulse_atr_mult: float = 1.0
) -> list[OrderBlock]:
    """Order Block = last opposite-color candle immediately before a BOS impulse.

    A bullish OB is the last bearish candle before an upward BOS.
    A bearish OB is the last bullish candle before a downward BOS.
    """
    obs: list[OrderBlock] = []
    if not events:
        return obs
    avg_range = _avg_true_range_simple(candles, period=14) or 0.0
    threshold = avg_range * impulse_atr_mult
    for ev in events:
        if not ev.kind.startswith("BOS_") and not ev.kind.startswith("CHOCH_"):
            continue
        # search back from event index for the last opposite-color candle
        target_color = "DOWN" if "UP" in ev.kind else "UP"
        for j in range(ev.index, max(ev.index - 20, -1), -1):
            c = candles[j]
            color_up = c.close >= c.open
            if (target_color == "DOWN" and not color_up) or (target_color == "UP" and color_up):
                # Confirm the move from this candle to the breakout was impulsive
                impulse_size = abs(candles[ev.index].close - c.close)
                if threshold and impulse_size < threshold:
                    continue
                ts = c.timestamp_utc.isoformat()
                side = "BULL" if "UP" in ev.kind else "BEAR"
                obs.append(OrderBlock(
                    index=j, timestamp_iso=ts, side=side, high=c.high, low=c.low,
                ))
                break
    return obs


def _detect_fvg(candles: Sequence[Candle], *, lookback: int = 60) -> list[FairValueGap]:
    """3-candle Fair Value Gap.

    Bullish FVG: candle[i].low > candle[i-2].high → gap between [i-2].high and [i].low
    Bearish FVG: candle[i].high < candle[i-2].low → gap between [i].high and [i-2].low
    Marked filled if a later candle's range covers the gap.
    """
    n = len(candles)
    if n < 3:
        return []
    fvgs: list[FairValueGap] = []
    start = max(2, n - lookback)
    for i in range(start, n):
        c0 = candles[i - 2]
        c2 = candles[i]
        if c2.low > c0.high:
            upper = c2.low
            lower = c0.high
            filled = any(candles[j].low <= lower for j in range(i + 1, n))
            fvgs.append(FairValueGap(
                index=i - 1, timestamp_iso=candles[i - 1].timestamp_utc.isoformat(),
                direction="UP", upper=upper, lower=lower, filled=filled,
            ))
        elif c2.high < c0.low:
            upper = c0.low
            lower = c2.high
            filled = any(candles[j].high >= upper for j in range(i + 1, n))
            fvgs.append(FairValueGap(
                index=i - 1, timestamp_iso=candles[i - 1].timestamp_utc.isoformat(),
                direction="DOWN", upper=upper, lower=lower, filled=filled,
            ))
    return fvgs


def _detect_liquidity(
    swings: list[SwingPivot], *, eq_tolerance_pips: float = 1.5, pip_size: float = 0.01
) -> list[LiquidityCluster]:
    """Cluster swing highs / lows whose prices fall within `eq_tolerance_pips`.

    These clusters flag stop-loss magnets where price often sweeps before reversing.
    """
    if not swings:
        return []
    tol = eq_tolerance_pips * pip_size
    clusters: list[LiquidityCluster] = []
    for side in ("HIGH", "LOW"):
        side_swings = [s for s in swings if s.side == side]
        used: set[int] = set()
        for i, s in enumerate(side_swings):
            if i in used:
                continue
            group = [s]
            for j in range(i + 1, len(side_swings)):
                if abs(side_swings[j].price - s.price) <= tol:
                    group.append(side_swings[j])
                    used.add(j)
            if len(group) >= 2:
                avg_price = sum(g.price for g in group) / len(group)
                clusters.append(LiquidityCluster(
                    side="EQ_HIGH" if side == "HIGH" else "EQ_LOW",
                    price=avg_price,
                    indices=tuple(g.index for g in group),
                ))
    return clusters


def _avg_true_range_simple(candles: Sequence[Candle], period: int = 14) -> float | None:
    n = len(candles)
    if n <= period:
        return None
    trs: list[float] = []
    for i in range(1, n):
        h = candles[i].high
        l = candles[i].low
        pc = candles[i - 1].close
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < period:
        return None
    return sum(trs[-period:]) / period
