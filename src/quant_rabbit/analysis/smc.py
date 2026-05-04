"""Smart Money Concepts (SMC) / ICT primitive detectors.

This module gives the discretionary trader a structured view of the
price-action concepts described in the SMC/ICT literature: swing pivots,
break-of-structure / change-of-character events, order blocks, fair-value
gaps, equal-highs/lows liquidity, plus the higher-tier primitives —
liquidity sweeps, breaker blocks, mitigation blocks, inversion FVGs,
displacement candles, the dealing range, and premium/discount/OTE zones.

It is intentionally pure-Python (stdlib only) so it runs without any
extra installs and stays cheap to call inside the trader cycle. All
detectors operate on the `Candle` records produced by
`quant_rabbit.analysis.candles`.

Numeric defaults are documented inline per `docs/AGENT_CONTRACT.md` §3.5
with (a) what market reality the value represents, (b) why it is held
constant rather than market-derived, and (c) what should replace it if
it ever needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from quant_rabbit.analysis.candles import Candle


# ---------------------------------------------------------------------------
# Base structural primitives (swings, BOS/CHOCH, OB, FVG, EQH/EQL)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SwingPivot:
    """A confirmed local high or low within a candle series."""

    index: int
    timestamp_iso: str
    price: float
    kind: str  # "HIGH" | "LOW"


@dataclass(frozen=True)
class StructureEvent:
    """A break-of-structure or change-of-character event."""

    index: int
    timestamp_iso: str
    kind: str  # "BOS_UP" | "BOS_DOWN" | "CHOCH_UP" | "CHOCH_DOWN"
    reference_price: float
    triggering_close: float


@dataclass(frozen=True)
class OrderBlock:
    """The last opposite-color candle before a displacement leg.

    A bullish OB is the last bearish candle before an up-displacement; a
    bearish OB is the last bullish candle before a down-displacement.
    """

    index: int
    timestamp_iso: str
    side: str  # "BULL" | "BEAR"
    high: float
    low: float
    displacement_index: int


@dataclass(frozen=True)
class FairValueGap:
    """A 3-candle imbalance where candle[i-1].high < candle[i+1].low (UP)
    or candle[i-1].low > candle[i+1].high (DOWN).
    """

    index: int
    timestamp_iso: str
    direction: str  # "UP" | "DOWN"
    upper: float
    lower: float
    filled: bool


@dataclass(frozen=True)
class LiquidityCluster:
    """A pair of equal (or near-equal) highs or lows representing
    pooled stop-liquidity.
    """

    indices: tuple[int, ...]
    price: float
    side: str  # "EQ_HIGH" | "EQ_LOW"


# ---------------------------------------------------------------------------
# --- Sweeps / Breakers / Mitigation / iFVG / Displacement / Dealing Range / OTE ---
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiquiditySweep:
    """Wick beyond a prior pivot/EQH/EQL/PDH/PDL that closes back inside.

    Bearish sweep (`SWEEP_HIGH`): `high > prior_swing_high` AND
    `close < prior_swing_high`. Bullish sweep (`SWEEP_LOW`) is symmetric.
    """

    index: int
    timestamp_iso: str
    side: str  # "SWEEP_HIGH" | "SWEEP_LOW"
    swept_pivot_price: float
    wick_extreme: float
    close_back_inside: bool


@dataclass(frozen=True)
class BreakerBlock:
    """An OB where price first swept it, then structure shifted opposite.

    Higher conviction than a plain OB — the sweep confirms liquidity was
    taken before the reversal.
    """

    base_ob: OrderBlock
    sweep: LiquiditySweep
    shift_event: StructureEvent


@dataclass(frozen=True)
class MitigationBlock:
    """Like a breaker but no liquidity sweep before reversal.

    Lower conviction than a `BreakerBlock`.
    """

    base_ob: OrderBlock
    shift_event: StructureEvent
    note: str


@dataclass(frozen=True)
class InversionFVG:
    """An FVG that was filled and is now respected as opposing S/R."""

    base_fvg: FairValueGap
    filled_at_index: int
    rejected_at_index: int


@dataclass(frozen=True)
class DisplacementCandle:
    """An impulse candle whose body is large relative to recent ATR.

    A displacement is the canonical signature of smart-money intent — it
    typically creates an FVG and breaks structure in the same move.
    """

    index: int
    timestamp_iso: str
    direction: str  # "UP" | "DOWN"
    body_atr_ratio: float
    creates_fvg: bool
    breaks_structure: bool


@dataclass(frozen=True)
class DealingRange:
    """The most recent significant HTF swing-high / swing-low pair.

    `equilibrium` is the 50% midpoint. `premium_zone` is the upper half
    (sell zone), `discount_zone` is the lower half (buy zone).
    `ote_zone` is the 0.62-0.79 retracement of the latest impulse leg
    inside the range, with `ote_sweet_spot` at 0.705.
    """

    swing_high: SwingPivot
    swing_low: SwingPivot
    equilibrium: float
    premium_zone: tuple[float, float]   # (eq, swing_high)
    discount_zone: tuple[float, float]  # (swing_low, eq)
    ote_zone: tuple[float, float]
    ote_sweet_spot: float


# ---------------------------------------------------------------------------
# Aggregate readings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructureReading:
    """Aggregate of the base structural primitives."""

    pivots: tuple[SwingPivot, ...]
    events: tuple[StructureEvent, ...]
    order_blocks: tuple[OrderBlock, ...]
    fair_value_gaps: tuple[FairValueGap, ...]
    liquidity_clusters: tuple[LiquidityCluster, ...]


@dataclass(frozen=True)
class SMCReading:
    """Full SMC snapshot: base structure + sweeps/breakers/iFVGs/etc."""

    structure: StructureReading
    sweeps: tuple[LiquiditySweep, ...]
    breakers: tuple[BreakerBlock, ...]
    mitigations: tuple[MitigationBlock, ...]
    inversion_fvgs: tuple[InversionFVG, ...]
    displacements: tuple[DisplacementCandle, ...]
    dealing_range: DealingRange | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso(candle: Candle) -> str:
    return candle.timestamp_utc.isoformat()


def _atr(candles: Sequence[Candle], window: int) -> list[float | None]:
    """Per-bar Wilder ATR over a `window` lookback.

    Returns a list aligned with `candles`. Bars where ATR is undefined
    return `None`.
    """

    n = len(candles)
    if n == 0 or window <= 0:
        return []
    trs: list[float] = []
    for i in range(n):
        if i == 0:
            tr = candles[i].high - candles[i].low
        else:
            prev_close = candles[i - 1].close
            tr = max(
                candles[i].high - candles[i].low,
                abs(candles[i].high - prev_close),
                abs(candles[i].low - prev_close),
            )
        trs.append(tr)
    out: list[float | None] = [None] * n
    if n < window:
        return out
    seed = sum(trs[:window]) / window
    out[window - 1] = seed
    prev = seed
    for i in range(window, n):
        cur = (prev * (window - 1) + trs[i]) / window
        out[i] = cur
        prev = cur
    return out


# ---------------------------------------------------------------------------
# Base detectors
# ---------------------------------------------------------------------------


def detect_swing_pivots(
    candles: Sequence[Candle], *, strength: int = 2
) -> tuple[SwingPivot, ...]:
    """Return confirmed swing pivots.

    `strength` is the number of bars required on each side for confirmation.
    Default `strength=2` (a) treats a bar as a pivot when 2 bars on each
    side fail to exceed it — the minimum window classical price-action
    teaching uses for "fractal" pivots; (b) is held constant rather than
    market-derived because increasing it monotonically smooths the swing
    list and there is no robust market quantity that maps to it; (c) bump
    to 3 or 5 if the trader needs HTF-only swings, or expose as a kwarg
    to a calling adapter.
    """

    n = len(candles)
    out: list[SwingPivot] = []
    if n < 2 * strength + 1:
        return tuple(out)
    for i in range(strength, n - strength):
        h = candles[i].high
        low = candles[i].low
        is_high = all(candles[i - k].high < h for k in range(1, strength + 1)) and all(
            candles[i + k].high < h for k in range(1, strength + 1)
        )
        is_low = all(candles[i - k].low > low for k in range(1, strength + 1)) and all(
            candles[i + k].low > low for k in range(1, strength + 1)
        )
        if is_high:
            out.append(
                SwingPivot(index=i, timestamp_iso=_iso(candles[i]), price=h, kind="HIGH")
            )
        if is_low:
            out.append(
                SwingPivot(index=i, timestamp_iso=_iso(candles[i]), price=low, kind="LOW")
            )
    return tuple(out)


def detect_structure_events(
    candles: Sequence[Candle], pivots: Sequence[SwingPivot]
) -> tuple[StructureEvent, ...]:
    """Detect BOS / CHOCH events using the pivot stream.

    Walks bars in order, tracking the most recent confirmed swing high/low
    and the prevailing bias. A break of the prior swing in the bias
    direction is a BOS; a break against the bias is a CHOCH.
    """

    out: list[StructureEvent] = []
    if not pivots:
        return tuple(out)
    pivot_by_index: dict[int, SwingPivot] = {p.index: p for p in pivots}
    last_high: SwingPivot | None = None
    last_low: SwingPivot | None = None
    bias: str | None = None  # "UP" | "DOWN"
    for i in range(len(candles)):
        if i in pivot_by_index:
            piv = pivot_by_index[i]
            if piv.kind == "HIGH":
                last_high = piv
            else:
                last_low = piv
            continue
        c = candles[i]
        if last_high is not None and c.close > last_high.price:
            kind = "BOS_UP" if bias == "UP" else "CHOCH_UP"
            out.append(
                StructureEvent(
                    index=i,
                    timestamp_iso=_iso(c),
                    kind=kind,
                    reference_price=last_high.price,
                    triggering_close=c.close,
                )
            )
            bias = "UP"
            last_high = None  # consumed; wait for next pivot
        elif last_low is not None and c.close < last_low.price:
            kind = "BOS_DOWN" if bias == "DOWN" else "CHOCH_DOWN"
            out.append(
                StructureEvent(
                    index=i,
                    timestamp_iso=_iso(c),
                    kind=kind,
                    reference_price=last_low.price,
                    triggering_close=c.close,
                )
            )
            bias = "DOWN"
            last_low = None
    return tuple(out)


def detect_order_blocks(
    candles: Sequence[Candle],
    events: Sequence[StructureEvent],
) -> tuple[OrderBlock, ...]:
    """Last opposite-color candle preceding a structure-breaking impulse."""

    out: list[OrderBlock] = []
    for ev in events:
        if ev.kind not in ("BOS_UP", "CHOCH_UP", "BOS_DOWN", "CHOCH_DOWN"):
            continue
        bull_break = ev.kind.endswith("UP")
        # walk backwards from ev.index-1 looking for opposite-color candle
        for j in range(ev.index - 1, -1, -1):
            c = candles[j]
            opposite_color = (
                (c.close < c.open) if bull_break else (c.close > c.open)
            )
            if opposite_color:
                out.append(
                    OrderBlock(
                        index=j,
                        timestamp_iso=_iso(c),
                        side="BULL" if bull_break else "BEAR",
                        high=c.high,
                        low=c.low,
                        displacement_index=ev.index,
                    )
                )
                break
    return tuple(out)


def detect_fair_value_gaps(candles: Sequence[Candle]) -> tuple[FairValueGap, ...]:
    """3-candle imbalance detector."""

    out: list[FairValueGap] = []
    n = len(candles)
    for i in range(1, n - 1):
        prev_h = candles[i - 1].high
        prev_l = candles[i - 1].low
        next_h = candles[i + 1].high
        next_l = candles[i + 1].low
        if prev_h < next_l:
            upper, lower = next_l, prev_h
            filled = any(candles[k].low <= lower for k in range(i + 2, n))
            out.append(
                FairValueGap(
                    index=i,
                    timestamp_iso=_iso(candles[i]),
                    direction="UP",
                    upper=upper,
                    lower=lower,
                    filled=filled,
                )
            )
        elif prev_l > next_h:
            upper, lower = prev_l, next_h
            filled = any(candles[k].high >= upper for k in range(i + 2, n))
            out.append(
                FairValueGap(
                    index=i,
                    timestamp_iso=_iso(candles[i]),
                    direction="DOWN",
                    upper=upper,
                    lower=lower,
                    filled=filled,
                )
            )
    return tuple(out)


def detect_liquidity_clusters(
    pivots: Sequence[SwingPivot], *, tolerance: float = 0.0005
) -> tuple[LiquidityCluster, ...]:
    """Equal-highs / equal-lows pairs.

    `tolerance` is the relative distance below which two pivots are
    treated as equal — (a) it represents the typical noise band on a
    forex chart (5 bp ≈ ~half a pip on a 1.10 EURUSD); (b) is held
    constant rather than market-derived because at the M5/M15 timeframes
    EQH/EQL detection is dominated by visual symmetry, not tick noise;
    (c) callers may pass a tighter value (e.g. 0.0001) for HTF charts
    or a looser value for index/JPY pairs if needed.
    """

    out: list[LiquidityCluster] = []
    highs = [p for p in pivots if p.kind == "HIGH"]
    lows = [p for p in pivots if p.kind == "LOW"]
    for group, side in ((highs, "EQ_HIGH"), (lows, "EQ_LOW")):
        used: set[int] = set()
        for i, a in enumerate(group):
            if i in used:
                continue
            cluster_idxs = [a.index]
            cluster_prices = [a.price]
            for j in range(i + 1, len(group)):
                if j in used:
                    continue
                b = group[j]
                if a.price == 0:
                    continue
                if abs(b.price - a.price) / abs(a.price) <= tolerance:
                    cluster_idxs.append(b.index)
                    cluster_prices.append(b.price)
                    used.add(j)
            if len(cluster_idxs) >= 2:
                out.append(
                    LiquidityCluster(
                        indices=tuple(cluster_idxs),
                        price=sum(cluster_prices) / len(cluster_prices),
                        side=side,
                    )
                )
    return tuple(out)


def analyze_structure(candles: Sequence[Candle]) -> StructureReading:
    """Run the base structural detectors over `candles`."""

    pivots = detect_swing_pivots(candles)
    events = detect_structure_events(candles, pivots)
    obs = detect_order_blocks(candles, events)
    fvgs = detect_fair_value_gaps(candles)
    clusters = detect_liquidity_clusters(pivots)
    return StructureReading(
        pivots=pivots,
        events=events,
        order_blocks=obs,
        fair_value_gaps=fvgs,
        liquidity_clusters=clusters,
    )


# ---------------------------------------------------------------------------
# --- Sweeps ---
# ---------------------------------------------------------------------------


def detect_sweeps(
    candles: Sequence[Candle],
    pivots: Sequence[SwingPivot],
    *,
    close_back_inside: bool = True,
) -> tuple[LiquiditySweep, ...]:
    """Detect liquidity-sweep wicks against prior swing highs/lows.

    `close_back_inside=True` (a) enforces the classical SMC rule that a
    sweep is only valid when the candle wicks beyond a prior pivot but
    closes back inside (a "failure to break") — wicks without that close
    are simply continuation breakouts; (b) is held constant rather than
    market-derived because the close-back-inside requirement is the
    definitional signature of stop-hunt liquidity behaviour, not a tunable
    threshold; (c) pass `False` if a caller wants raw wick taps for a
    different research workflow.
    """

    out: list[LiquiditySweep] = []
    if not pivots:
        return tuple(out)
    highs = sorted(
        (p for p in pivots if p.kind == "HIGH"), key=lambda p: p.index
    )
    lows = sorted(
        (p for p in pivots if p.kind == "LOW"), key=lambda p: p.index
    )
    for i, c in enumerate(candles):
        prior_high = next(
            (p for p in reversed(highs) if p.index < i), None
        )
        if prior_high is not None and c.high > prior_high.price:
            closed_back = c.close < prior_high.price
            if (not close_back_inside) or closed_back:
                out.append(
                    LiquiditySweep(
                        index=i,
                        timestamp_iso=_iso(c),
                        side="SWEEP_HIGH",
                        swept_pivot_price=prior_high.price,
                        wick_extreme=c.high,
                        close_back_inside=closed_back,
                    )
                )
        prior_low = next(
            (p for p in reversed(lows) if p.index < i), None
        )
        if prior_low is not None and c.low < prior_low.price:
            closed_back = c.close > prior_low.price
            if (not close_back_inside) or closed_back:
                out.append(
                    LiquiditySweep(
                        index=i,
                        timestamp_iso=_iso(c),
                        side="SWEEP_LOW",
                        swept_pivot_price=prior_low.price,
                        wick_extreme=c.low,
                        close_back_inside=closed_back,
                    )
                )
    return tuple(out)


# ---------------------------------------------------------------------------
# --- Breakers ---
# ---------------------------------------------------------------------------


def detect_breakers(
    structure: StructureReading,
    sweeps: Sequence[LiquiditySweep],
) -> tuple[BreakerBlock, ...]:
    """Order Blocks that were swept then flipped via a structure shift.

    Pattern: OB at index `b`, then a sweep at `s > b` whose wick traverses
    the OB range, then a CHOCH/BOS in the opposite direction at `e > s`.
    """

    out: list[BreakerBlock] = []
    for ob in structure.order_blocks:
        opposite_kinds = (
            ("BOS_DOWN", "CHOCH_DOWN") if ob.side == "BULL" else ("BOS_UP", "CHOCH_UP")
        )
        for sweep in sweeps:
            if sweep.index <= ob.index:
                continue
            wick_in_ob = (
                sweep.side == "SWEEP_HIGH"
                and ob.side == "BULL"
                and sweep.wick_extreme >= ob.low
            ) or (
                sweep.side == "SWEEP_LOW"
                and ob.side == "BEAR"
                and sweep.wick_extreme <= ob.high
            )
            # General fallback: sweep wick inside OB band
            if not wick_in_ob:
                wick_in_ob = ob.low <= sweep.wick_extreme <= ob.high
            if not wick_in_ob:
                continue
            shift = next(
                (
                    e
                    for e in structure.events
                    if e.index > sweep.index and e.kind in opposite_kinds
                ),
                None,
            )
            if shift is None:
                continue
            out.append(BreakerBlock(base_ob=ob, sweep=sweep, shift_event=shift))
            break
    return tuple(out)


# ---------------------------------------------------------------------------
# --- Mitigation ---
# ---------------------------------------------------------------------------


def detect_mitigations(structure: StructureReading) -> tuple[MitigationBlock, ...]:
    """OBs followed by an opposing CHOCH without a preceding sweep.

    Lower probability than a breaker — there was no liquidity grab to
    confirm smart-money intent.
    """

    out: list[MitigationBlock] = []
    for ob in structure.order_blocks:
        opposite_kinds = (
            ("CHOCH_DOWN", "BOS_DOWN") if ob.side == "BULL" else ("CHOCH_UP", "BOS_UP")
        )
        shift = next(
            (
                e
                for e in structure.events
                if e.index > ob.displacement_index and e.kind in opposite_kinds
            ),
            None,
        )
        if shift is None:
            continue
        out.append(
            MitigationBlock(
                base_ob=ob,
                shift_event=shift,
                note="no preceding sweep — lower probability",
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# --- iFVG ---
# ---------------------------------------------------------------------------


def detect_inversion_fvgs(
    candles: Sequence[Candle], fvgs: Sequence[FairValueGap]
) -> tuple[InversionFVG, ...]:
    """FVGs that were filled and then respected as opposing S/R."""

    out: list[InversionFVG] = []
    n = len(candles)
    for fvg in fvgs:
        if not fvg.filled:
            continue
        # Find first bar after fvg.index that fully fills the gap
        filled_at: int | None = None
        for k in range(fvg.index + 2, n):
            c = candles[k]
            if fvg.direction == "UP" and c.low <= fvg.lower:
                filled_at = k
                break
            if fvg.direction == "DOWN" and c.high >= fvg.upper:
                filled_at = k
                break
        if filled_at is None:
            continue
        # Find a subsequent rejection: for a filled UP-FVG (now resistance),
        # price re-touches the gap from below and rejects (close < lower).
        rejected_at: int | None = None
        for k in range(filled_at + 1, n):
            c = candles[k]
            if fvg.direction == "UP":
                # gap now acts as resistance: tap into it then close back below
                if c.high >= fvg.lower and c.close < fvg.lower:
                    rejected_at = k
                    break
            else:
                # gap now acts as support: tap up to upper then close back above
                if c.low <= fvg.upper and c.close > fvg.upper:
                    rejected_at = k
                    break
        if rejected_at is None:
            continue
        out.append(
            InversionFVG(
                base_fvg=fvg, filled_at_index=filled_at, rejected_at_index=rejected_at
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# --- Displacement ---
# ---------------------------------------------------------------------------


def detect_displacement(
    candles: Sequence[Candle],
    *,
    atr_window: int = 14,
    body_atr_threshold: float = 1.5,
) -> tuple[DisplacementCandle, ...]:
    """Flag candles whose body is `body_atr_threshold` × ATR(N) or larger.

    `atr_window=14` (a) is the canonical Wilder ATR window used by the
    indicator stack and matches what the trader already reads on the
    chart; (b) is held constant rather than market-derived so the
    displacement test is calibrated against the same volatility lens used
    elsewhere in the system; (c) bump to 20 for HTF charts or pass a
    different window if a caller needs a longer-memory volatility filter.

    `body_atr_threshold=1.5` (a) is the textbook ICT threshold separating
    a routine bar from a true displacement impulse — a body 1.5× the
    average true range is a statistically uncommon move that typically
    signals smart-money intent; (b) is held constant rather than
    market-derived because the threshold is a heuristic on the
    body/volatility ratio, not a market price; (c) loosen to 1.2 for
    quieter pairs or tighten to 2.0 for sweep-trap research.
    """

    out: list[DisplacementCandle] = []
    n = len(candles)
    if n == 0:
        return tuple(out)
    atrs = _atr(candles, atr_window)
    fvgs = detect_fair_value_gaps(candles)
    fvg_centers = {fvg.index for fvg in fvgs}
    pivots = detect_swing_pivots(candles)
    events = detect_structure_events(candles, pivots)
    event_indices = {e.index for e in events}
    for i, c in enumerate(candles):
        atr = atrs[i] if i < len(atrs) else None
        if atr is None or atr <= 0:
            continue
        body = abs(c.close - c.open)
        ratio = body / atr
        if ratio < body_atr_threshold:
            continue
        direction = "UP" if c.close > c.open else "DOWN"
        creates_fvg = (i - 1) in fvg_centers or i in fvg_centers or (i + 1) in fvg_centers
        breaks_structure = i in event_indices
        out.append(
            DisplacementCandle(
                index=i,
                timestamp_iso=_iso(c),
                direction=direction,
                body_atr_ratio=ratio,
                creates_fvg=creates_fvg,
                breaks_structure=breaks_structure,
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# --- Dealing Range / Premium-Discount / OTE ---
# ---------------------------------------------------------------------------


# Fibonacci retracement levels for OTE.
# (a) 0.62 / 0.79 are the classical ICT "Optimal Trade Entry" boundary
#     retracements; 0.705 is the published sweet-spot midpoint between them.
# (b) These are constants of the SMC/ICT methodology, not market-derived
#     quantities — every public ICT teaching uses these exact values.
# (c) If the methodology evolves (e.g. an institution publishes a refined
#     band), update the trio together; never tune them individually.
OTE_LOWER_FIB: float = 0.62
OTE_UPPER_FIB: float = 0.79
OTE_SWEET_SPOT_FIB: float = 0.705


def compute_dealing_range(
    candles: Sequence[Candle],
    *,
    lookback: int = 200,
    swing_strength: int = 5,
) -> DealingRange | None:
    """Return the dealing range over the last `lookback` bars.

    `lookback=200` (a) corresponds to roughly one trading day of M5 bars
    or one trading week of H1 bars — enough history for an HTF dealing
    range without dragging in stale structure; (b) is held constant
    rather than market-derived because the dealing-range concept is a
    "recent meaningful swing" and 200 bars is the round figure ICT
    teaching uses; (c) shrink to 100 for intraday-only analysis or grow
    to 500 for swing/position trading workflows.

    `swing_strength=5` (a) demands 5 bars of confirmation on each side of
    a pivot — an HTF-grade swing rather than a noisy fractal; (b) is held
    constant rather than market-derived because dealing-range pivots
    must be visually obvious to count, and 5 is the canonical strength
    used in published SMC examples; (c) drop to 3 for choppy markets or
    raise to 8 for very smooth markets.
    """

    n = len(candles)
    if n == 0:
        return None
    start = max(0, n - lookback)
    window = candles[start:]
    pivots = detect_swing_pivots(window, strength=swing_strength)
    if len(pivots) < 2:
        return None
    highs = [p for p in pivots if p.kind == "HIGH"]
    lows = [p for p in pivots if p.kind == "LOW"]
    if not highs or not lows:
        return None
    highest = max(highs, key=lambda p: p.price)
    lowest = min(lows, key=lambda p: p.price)
    # re-anchor pivot indices to the original candle series
    swing_high = SwingPivot(
        index=highest.index + start,
        timestamp_iso=highest.timestamp_iso,
        price=highest.price,
        kind="HIGH",
    )
    swing_low = SwingPivot(
        index=lowest.index + start,
        timestamp_iso=lowest.timestamp_iso,
        price=lowest.price,
        kind="LOW",
    )
    high_p = swing_high.price
    low_p = swing_low.price
    if high_p <= low_p:
        return None
    eq = (high_p + low_p) / 2.0
    span = high_p - low_p

    # Latest impulse leg = direction from earliest of the two anchors to the latest.
    if swing_low.index < swing_high.index:
        # Up-leg: OTE measured from high back toward low (retracement of the up-leg)
        ote_lower = high_p - span * OTE_UPPER_FIB
        ote_upper = high_p - span * OTE_LOWER_FIB
        ote_sweet = high_p - span * OTE_SWEET_SPOT_FIB
    else:
        # Down-leg: OTE measured from low back toward high
        ote_lower = low_p + span * OTE_LOWER_FIB
        ote_upper = low_p + span * OTE_UPPER_FIB
        ote_sweet = low_p + span * OTE_SWEET_SPOT_FIB
    ote_zone = (min(ote_lower, ote_upper), max(ote_lower, ote_upper))
    return DealingRange(
        swing_high=swing_high,
        swing_low=swing_low,
        equilibrium=eq,
        premium_zone=(eq, high_p),
        discount_zone=(low_p, eq),
        ote_zone=ote_zone,
        ote_sweet_spot=ote_sweet,
    )


def compute_premium_discount(price: float, dealing_range: DealingRange) -> str:
    """Classify `price` against the dealing range as PREMIUM/DISCOUNT/EQUILIBRIUM.

    A small equilibrium band is intentional: a price within
    `EQUILIBRIUM_BAND_PCT` of the midpoint is treated as "at equilibrium"
    rather than borderline premium/discount.
    """

    eq = dealing_range.equilibrium
    span = dealing_range.swing_high.price - dealing_range.swing_low.price
    if span <= 0:
        return "EQUILIBRIUM"
    band = span * EQUILIBRIUM_BAND_PCT
    if abs(price - eq) <= band:
        return "EQUILIBRIUM"
    return "PREMIUM" if price > eq else "DISCOUNT"


# Width of the "at equilibrium" band as a fraction of the dealing range.
# (a) 1% of the range is a tight visual midline — narrow enough that
#     PREMIUM/DISCOUNT classifications are not flipped by a single tick of
#     drift, wide enough that the trader can label "near 50%" zones cleanly.
# (b) Held constant rather than market-derived because the band is a
#     classification convenience, not a market quantity.
# (c) Shrink toward 0 for a pure 50-50 split, or widen to 0.05 if the
#     trader wants a more permissive equilibrium zone.
EQUILIBRIUM_BAND_PCT: float = 0.01


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def analyze_smc(candles: Sequence[Candle]) -> SMCReading:
    """Run the full SMC pipeline and return the aggregate reading."""

    structure = analyze_structure(candles)
    sweeps = detect_sweeps(candles, structure.pivots)
    breakers = detect_breakers(structure, sweeps)
    mitigations = detect_mitigations(structure)
    inversions = detect_inversion_fvgs(candles, structure.fair_value_gaps)
    displacements = detect_displacement(candles)
    dealing_range = compute_dealing_range(candles)
    return SMCReading(
        structure=structure,
        sweeps=sweeps,
        breakers=breakers,
        mitigations=mitigations,
        inversion_fvgs=inversions,
        displacements=displacements,
        dealing_range=dealing_range,
    )
