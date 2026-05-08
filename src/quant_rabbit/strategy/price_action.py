"""Price action lens for the trader.

Reads SMC-style structural data already produced into pair_charts
(swings, structure_events, liquidity, order_blocks, fair_value_gaps,
dealing_range) and surfaces composite signals that the MTF confluence
scorer and the adaptive TP / position-manager logic can consume.

User directive 2026-05-08「市況を読んでほしい」「足の形や、何回高値や
底値をアタックしたかとか、フラッグかとか。いろいろみれるよね」.

The module is intentionally pure: every helper takes a per-TF view dict
(the entries inside `pair_charts['charts'][N]['views']`) and returns a
typed result. No file I/O, no broker calls — caller wires the data in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Swing pattern classifications. The series is read newest-first.
SWING_HH_HL = "HH_HL"      # higher highs + higher lows → uptrend
SWING_LH_LL = "LH_LL"      # lower highs + lower lows → downtrend
SWING_HH_LL = "EXPANDING"  # range expanding (volatility regime)
SWING_LH_HL = "CONTRACTING"  # consolidation / triangle
SWING_MIXED = "MIXED"      # fewer than 4 swings or noisy


@dataclass(frozen=True)
class SwingPattern:
    """Classify the most recent 4 swing pivots into a directional pattern."""
    pattern: str
    last_high: float | None
    last_low: float | None
    prev_high: float | None
    prev_low: float | None
    high_attempts: int        # number of equal/near-equal highs in the recent window
    low_attempts: int         # number of equal/near-equal lows


def classify_swings(view: dict[str, Any], near_ratio: float = 0.0007) -> SwingPattern:
    """Classify HH/HL / LH/LL / range from the swings array.

    near_ratio: two highs/lows within this fractional distance count as the
    same level (default 7 bps ≈ 7 pips on a 1.17xx EUR/USD price). Tunable
    per pair if FX vs index volatility differs.
    """
    structure = view.get("structure") if isinstance(view, dict) else None
    swings = (structure or {}).get("swings") or []
    highs = [s for s in swings if s.get("side") == "HIGH"]
    lows = [s for s in swings if s.get("side") == "LOW"]
    last_high = highs[-1]["price"] if highs else None
    prev_high = highs[-2]["price"] if len(highs) >= 2 else None
    last_low = lows[-1]["price"] if lows else None
    prev_low = lows[-2]["price"] if len(lows) >= 2 else None

    # Count near-equal attempts on the most recent levels (signal of
    # resistance/support being respected).
    def _attempts(target: float | None, peers: list[dict[str, Any]]) -> int:
        if target is None or not peers:
            return 0
        return sum(
            1 for s in peers if abs((s.get("price", target) - target) / target) <= near_ratio
        )

    high_attempts = _attempts(last_high, highs)
    low_attempts = _attempts(last_low, lows)

    if last_high is None or prev_high is None or last_low is None or prev_low is None:
        return SwingPattern(SWING_MIXED, last_high, last_low, prev_high, prev_low, high_attempts, low_attempts)

    higher_highs = last_high > prev_high
    higher_lows = last_low > prev_low
    lower_highs = last_high < prev_high
    lower_lows = last_low < prev_low

    if higher_highs and higher_lows:
        pattern = SWING_HH_HL
    elif lower_highs and lower_lows:
        pattern = SWING_LH_LL
    elif higher_highs and lower_lows:
        pattern = SWING_HH_LL
    elif lower_highs and higher_lows:
        pattern = SWING_LH_HL
    else:
        pattern = SWING_MIXED
    return SwingPattern(pattern, last_high, last_low, prev_high, prev_low, high_attempts, low_attempts)


@dataclass(frozen=True)
class StructureEventsRead:
    """Summarise the most recent BOS/CHOCH events for trend vs reversal vs range."""
    classification: str        # "TREND_UP", "TREND_DOWN", "REVERSAL_UP", "REVERSAL_DOWN", "RANGE", "MIXED"
    recent_kinds: tuple[str, ...]
    last_kind: str | None
    last_pivot_price: float | None
    direction_changes: int     # count of UP↔DOWN flips in the recent window


def classify_structure_events(view: dict[str, Any], lookback: int = 5) -> StructureEventsRead:
    structure = view.get("structure") if isinstance(view, dict) else None
    events = (structure or {}).get("structure_events") or []
    recent = list(events)[-lookback:]
    kinds = tuple(str(ev.get("kind", "")) for ev in recent)
    last_kind = kinds[-1] if kinds else None
    last_pivot = recent[-1].get("broken_pivot_price") if recent else None

    if not kinds:
        return StructureEventsRead("MIXED", (), None, None, 0)

    def _dir(k: str) -> str:
        if k.endswith("_UP"):
            return "UP"
        if k.endswith("_DOWN"):
            return "DOWN"
        return "?"

    dirs = [_dir(k) for k in kinds if _dir(k) != "?"]
    direction_changes = sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i - 1])

    last_dir = _dir(last_kind or "")
    if direction_changes >= 2:
        # Multiple UP/DOWN flips inside the lookback = ranging
        cls = "RANGE"
    elif direction_changes == 1 and last_dir != "?":
        # Most recent direction differs from the start = reversal in progress
        cls = f"REVERSAL_{last_dir}"
    elif last_dir != "?":
        cls = f"TREND_{last_dir}"
    else:
        cls = "MIXED"
    return StructureEventsRead(cls, kinds, last_kind, last_pivot, direction_changes)


@dataclass(frozen=True)
class LiquidityLevel:
    side: str        # "EQ_HIGH" or "EQ_LOW"
    price: float
    touches: int


def strong_liquidity_levels(
    view: dict[str, Any],
    *,
    min_touches: int = 3,
    side_filter: str | None = None,
    above_price: float | None = None,
    below_price: float | None = None,
) -> tuple[LiquidityLevel, ...]:
    """Return liquidity pools touched at least `min_touches` times.

    `above_price` and `below_price` filter to levels that are *currently* on
    the right side of price for resistance/support purposes. EQ_HIGH below
    current price has already been broken — it's not resistance any more.
    Always pass the current quote when scoring entry/hold decisions; omit
    only when surveying historical levels for context.

    Strong levels (high touch count) act as institutional resistance/support
    — price tends to either reverse from them or sweep them as a liquidity
    grab before continuing.
    """
    structure = view.get("structure") if isinstance(view, dict) else None
    pools = (structure or {}).get("liquidity") or []
    out: list[LiquidityLevel] = []
    for p in pools:
        side = str(p.get("side") or "")
        if side_filter and side != side_filter:
            continue
        indices = p.get("indices") or []
        n = len(indices) if isinstance(indices, list) else 0
        if n < min_touches:
            continue
        try:
            price = float(p.get("price"))
        except (TypeError, ValueError):
            continue
        if above_price is not None and price <= above_price:
            continue
        if below_price is not None and price >= below_price:
            continue
        out.append(LiquidityLevel(side=side, price=price, touches=n))
    out.sort(key=lambda lvl: lvl.touches, reverse=True)
    return tuple(out)


@dataclass(frozen=True)
class DealingRangePosition:
    """Where current price sits inside the SMC dealing range."""
    region: str        # "ABOVE_HIGH", "PREMIUM", "EQUILIBRIUM", "DISCOUNT", "BELOW_LOW"
    swing_high: float | None
    swing_low: float | None
    equilibrium: float | None
    ote_sweet_spot: float | None
    distance_to_high_pips: float | None
    distance_to_low_pips: float | None
    range_size_pips: float | None


def classify_dealing_range_position(
    view: dict[str, Any],
    current_price: float,
    pip_factor: float,
) -> DealingRangePosition:
    """Locate `current_price` inside the dealing range surfaced by SMC.

    Premium = above equilibrium (sell zone). Discount = below equilibrium
    (buy zone). Above swing_high or below swing_low signals breakout/breakdown.
    """
    smc = view.get("smc") if isinstance(view, dict) else None
    dr = (smc or {}).get("dealing_range") if isinstance(smc, dict) else None
    if not isinstance(dr, dict):
        return DealingRangePosition("MIXED", None, None, None, None, None, None, None)
    try:
        sh = float(dr["swing_high"])
        sl = float(dr["swing_low"])
        eq = float(dr["equilibrium"])
    except (KeyError, TypeError, ValueError):
        return DealingRangePosition("MIXED", None, None, None, None, None, None, None)
    ote = dr.get("ote_sweet_spot")
    try:
        ote_v = float(ote) if ote is not None else None
    except (TypeError, ValueError):
        ote_v = None

    if current_price > sh:
        region = "ABOVE_HIGH"
    elif current_price < sl:
        region = "BELOW_LOW"
    elif current_price >= eq:
        region = "PREMIUM"
    elif ote_v is not None and current_price <= ote_v:
        region = "DISCOUNT"
    else:
        # Between OTE sweet-spot and equilibrium → call it equilibrium
        region = "EQUILIBRIUM"

    return DealingRangePosition(
        region=region,
        swing_high=sh,
        swing_low=sl,
        equilibrium=eq,
        ote_sweet_spot=ote_v,
        distance_to_high_pips=(sh - current_price) * pip_factor,
        distance_to_low_pips=(current_price - sl) * pip_factor,
        range_size_pips=(sh - sl) * pip_factor,
    )


@dataclass(frozen=True)
class OrderBlockProximity:
    """Nearest unmitigated order block on each side of current price."""
    nearest_bull_low: float | None        # support OB (LONG entry zone)
    nearest_bull_high: float | None
    bull_distance_pips: float | None
    nearest_bear_low: float | None        # resistance OB (SHORT entry zone)
    nearest_bear_high: float | None
    bear_distance_pips: float | None


def classify_order_block_proximity(
    view: dict[str, Any],
    current_price: float,
    pip_factor: float,
) -> OrderBlockProximity:
    """Identify the nearest BULL OB below price and nearest BEAR OB above.

    Pullback into a BULL OB = LONG re-entry candidate. Rally into a BEAR OB
    = SHORT entry candidate. Distance in pips lets the caller gate by
    proximity (e.g. only act when within 1 ATR).
    """
    structure = view.get("structure") if isinstance(view, dict) else None
    obs = (structure or {}).get("order_blocks") or []
    bull_below: list[tuple[float, float, float]] = []  # (low, high, distance)
    bear_above: list[tuple[float, float, float]] = []
    for ob in obs:
        try:
            high = float(ob["high"])
            low = float(ob["low"])
        except (KeyError, TypeError, ValueError):
            continue
        side = str(ob.get("side") or "")
        if side == "BULL" and high <= current_price:
            bull_below.append((low, high, (current_price - high) * pip_factor))
        elif side == "BEAR" and low >= current_price:
            bear_above.append((low, high, (low - current_price) * pip_factor))
    bull_below.sort(key=lambda x: x[2])
    bear_above.sort(key=lambda x: x[2])
    bull = bull_below[0] if bull_below else None
    bear = bear_above[0] if bear_above else None
    return OrderBlockProximity(
        nearest_bull_low=bull[0] if bull else None,
        nearest_bull_high=bull[1] if bull else None,
        bull_distance_pips=bull[2] if bull else None,
        nearest_bear_low=bear[0] if bear else None,
        nearest_bear_high=bear[1] if bear else None,
        bear_distance_pips=bear[2] if bear else None,
    )


@dataclass(frozen=True)
class PriceActionRead:
    """Composite read for one timeframe."""
    swing: SwingPattern
    events: StructureEventsRead
    range_position: DealingRangePosition
    obs: OrderBlockProximity
    strong_resistance: tuple[LiquidityLevel, ...]
    strong_support: tuple[LiquidityLevel, ...]


def read_timeframe(view: dict[str, Any], current_price: float, pip_factor: float) -> PriceActionRead:
    return PriceActionRead(
        swing=classify_swings(view),
        events=classify_structure_events(view),
        range_position=classify_dealing_range_position(view, current_price, pip_factor),
        obs=classify_order_block_proximity(view, current_price, pip_factor),
        # Only count EQ_HIGH levels that sit ABOVE current price (true resistance)
        # and EQ_LOW levels that sit BELOW current price (true support). Levels
        # already broken are no longer the same role.
        strong_resistance=strong_liquidity_levels(
            view, min_touches=3, side_filter="EQ_HIGH", above_price=current_price
        ),
        strong_support=strong_liquidity_levels(
            view, min_touches=3, side_filter="EQ_LOW", below_price=current_price
        ),
    )


def aggregate_price_action_score(
    pair_chart: dict[str, Any] | None,
    side: str,
    current_price: float,
    pip_factor: float,
) -> tuple[float, list[str]]:
    """Multi-TF price-action read aggregated across H4/H1/M30/M15/M5.

    Higher TFs get heavier weight (macro structure trumps M5 noise). M1 is
    excluded — its swings/structure events are too noisy for direction-level
    decisions and are already caught by the trader_brain micro classifier.

    Returns (delta_score, flat_reason_list). Caller adds the delta to the
    existing MTF confluence number.

    Score envelope: roughly ±25 across the full TF stack when every lens
    aligns. The weights are intentionally ≈ MTF confluence's ±30 ceiling
    so price-action evidence shades the call rather than dominating in
    isolation; only strong agreement across PA + MTF + structural lenses
    will move a borderline lane between LONG/SHORT.
    """
    if not isinstance(pair_chart, dict):
        return 0.0, []
    weights = {"H4": 0.28, "H1": 0.24, "M30": 0.20, "M15": 0.16, "M5": 0.12}
    total = 0.0
    summary: list[str] = []
    for view in pair_chart.get("views") or []:
        gran = str(view.get("granularity") or "")
        if gran not in weights:
            continue
        read = read_timeframe(view, current_price, pip_factor)
        delta, reasons = price_action_score(read, side)
        weighted = delta * weights[gran]
        total += weighted
        if abs(weighted) >= 1.0 and reasons:
            # Surface the most informative reason per TF
            summary.append(f"{gran}({weighted:+.1f}): {reasons[0]}")
    return total, summary


def price_action_score(read: PriceActionRead, side: str) -> tuple[float, list[str]]:
    """Return (delta_score, reasons) for a candidate entry/hold in `side`.

    Positive delta = price-action evidence supports `side`. Negative delta =
    counter-evidence (entering into resistance, swing pattern flipping, etc).
    Caller composes this with the existing MTF confluence number.

    Score ranges (intentionally small relative to MTF +30 ceiling so price
    action shades a marginal call rather than dominating):
      ±8 swing pattern alignment
      ±6 structure events (TREND_X reinforcing, REVERSAL_X opposing)
      ±5 dealing range region (DISCOUNT favors LONG, PREMIUM favors SHORT)
      ±4 order block proximity (BULL OB nearby + LONG = pullback entry)
      ±5 strong liquidity above/below price (resistance/support proximity)
    """
    if side not in {"LONG", "SHORT"}:
        return 0.0, []
    target_up = side == "LONG"
    delta = 0.0
    reasons: list[str] = []

    # --- swing pattern ---
    sp = read.swing.pattern
    if sp == SWING_HH_HL:
        if target_up:
            delta += 8.0
            reasons.append("swing HH/HL supports LONG")
        else:
            delta -= 8.0
            reasons.append("swing HH/HL opposes SHORT")
    elif sp == SWING_LH_LL:
        if not target_up:
            delta += 8.0
            reasons.append("swing LH/LL supports SHORT")
        else:
            delta -= 8.0
            reasons.append("swing LH/LL opposes LONG")
    elif sp == SWING_HH_LL:
        delta -= 2.0
        reasons.append("swing HH/LL expanding range — caution")
    elif sp == SWING_LH_HL:
        delta -= 1.0
        reasons.append("swing LH/HL contracting — breakout pending")

    # Repeated attempts on a level mean it's well-defended; entering INTO it
    # without breakout confirmation is risky.
    if read.swing.last_high is not None and read.swing.high_attempts >= 2 and target_up:
        delta -= 3.0
        reasons.append(f"resistance attempted {read.swing.high_attempts}× (LONG into resistance)")
    if read.swing.last_low is not None and read.swing.low_attempts >= 2 and not target_up:
        delta -= 3.0
        reasons.append(f"support attempted {read.swing.low_attempts}× (SHORT into support)")

    # --- structure events ---
    cls = read.events.classification
    if cls == "TREND_UP":
        delta += 6.0 if target_up else -6.0
        reasons.append(f"BOS/CHOCH events trending UP ({'+' if target_up else '−'})")
    elif cls == "TREND_DOWN":
        delta += 6.0 if not target_up else -6.0
        reasons.append(f"BOS/CHOCH events trending DOWN ({'+' if not target_up else '−'})")
    elif cls == "REVERSAL_UP":
        # A confirmed reversal up — supports LONG re-entry, opposes SHORT
        delta += 4.0 if target_up else -6.0
        reasons.append("REVERSAL_UP recently printed")
    elif cls == "REVERSAL_DOWN":
        delta += 4.0 if not target_up else -6.0
        reasons.append("REVERSAL_DOWN recently printed")
    elif cls == "RANGE":
        delta -= 2.0
        reasons.append("BOS/CHOCH events flipping repeatedly — range")

    # --- dealing range region ---
    region = read.range_position.region
    if region == "DISCOUNT":
        delta += 5.0 if target_up else -5.0
        reasons.append("price in DISCOUNT (below OTE) — LONG zone")
    elif region == "PREMIUM":
        delta += 5.0 if not target_up else -5.0
        reasons.append("price in PREMIUM (above EQ) — SHORT zone")
    elif region == "ABOVE_HIGH":
        # Breakout above range high — could be continuation, could be exhaustion
        delta += 1.0 if target_up else -2.0
        reasons.append("price ABOVE swing_high — breakout zone")
    elif region == "BELOW_LOW":
        delta += 1.0 if not target_up else -2.0
        reasons.append("price BELOW swing_low — breakdown zone")

    # --- order block proximity ---
    if target_up and read.obs.bull_distance_pips is not None and read.obs.bull_distance_pips <= 10.0:
        delta += 4.0
        reasons.append(
            f"near BULL OB {read.obs.nearest_bull_low}-{read.obs.nearest_bull_high} "
            f"({read.obs.bull_distance_pips:.1f}p above)"
        )
    if not target_up and read.obs.bear_distance_pips is not None and read.obs.bear_distance_pips <= 10.0:
        delta += 4.0
        reasons.append(
            f"near BEAR OB {read.obs.nearest_bear_low}-{read.obs.nearest_bear_high} "
            f"({read.obs.bear_distance_pips:.1f}p below)"
        )

    # --- strong liquidity overhead/underneath ---
    if target_up and read.strong_resistance:
        top = read.strong_resistance[0]
        delta -= 3.0
        reasons.append(f"strong resistance @{top.price} ({top.touches} touches) overhead")
    if not target_up and read.strong_support:
        bot = read.strong_support[0]
        delta -= 3.0
        reasons.append(f"strong support @{bot.price} ({bot.touches} touches) underneath")

    return delta, reasons
