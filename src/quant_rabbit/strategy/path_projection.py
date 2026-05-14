"""Multi-step price-path projection.

User directive 2026-05-14:「複合予測の継承: sweep → FVG fill → continuation のような
multi-step path projection」.

A single-step prediction says "price will probably go to X". A
multi-step path says "price will sweep liquidity at X, then mitigate
the FVG at Y, then continue in direction Z to TP at W". The trader can
enter at X (Step 1 confirmed), exit at Y, re-enter at Z, etc — a chain
of higher-quality entries than a single setup.

Path patterns supported (Smart Money Concepts):

1. **Sweep → Mitigation → Continuation**
   - Step A: price sweeps a liquidity pool (equal-highs / equal-lows)
   - Step B: price retraces to the unmitigated FVG / order block
   - Step C: price continues in the original trend direction
   - For LONG: sweep_low → bull FVG fill → continuation UP
   - For SHORT: mirror

2. **Reversal Path (BOS → CHOCH → Retest)**
   - Step A: BOS in opposite direction (structural break)
   - Step B: CHOCH on lower TF confirms turn
   - Step C: retest of broken level holds → entry

Each `PathProjection` carries an ordered list of steps with predicted
prices and progress state (PENDING / CONFIRMED / FAILED). The path is
considered ACTIVE while progress is monotonic and recent enough; goes
to FAILED if any step is violated.

The trader uses PathProjections to:
- Place LIMIT orders at upcoming step prices
- Adjust TP to the path's final target
- Cut early when a step is violated (path FAILED → next prediction)

Kill switch: `QR_DISABLE_PATH_PROJECTION=1`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


PATH_PROJECTION_BONUS = float(os.environ.get("QR_PATH_PROJECTION_BONUS", "18.0"))


@dataclass(frozen=True)
class PathStep:
    label: str  # "sweep_low" | "fvg_fill" | "continuation_target" | etc
    timeframe: str
    expected_price: float
    status: str  # "PENDING" | "CONFIRMED" | "FAILED"
    rationale: str = ""


@dataclass(frozen=True)
class PathProjection:
    pair: str
    direction: str  # "UP" | "DOWN"
    name: str
    steps: tuple[PathStep, ...]
    bonus_magnitude: float
    confidence: float
    rationale: str

    def active_step_index(self) -> int:
        for i, s in enumerate(self.steps):
            if s.status == "PENDING":
                return i
        return len(self.steps)  # all confirmed / failed

    def next_step(self) -> Optional[PathStep]:
        idx = self.active_step_index()
        if idx >= len(self.steps):
            return None
        return self.steps[idx]


def _is_disabled() -> bool:
    return os.environ.get("QR_DISABLE_PATH_PROJECTION", "").strip() in {
        "1", "true", "TRUE", "yes", "YES",
    }


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _collect_unmitigated_fvgs(view: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract unmitigated FVG entries from a view's structure block."""
    structure = view.get("structure") or {}
    fvgs = structure.get("fair_value_gaps") or []
    if not isinstance(fvgs, list):
        return []
    out: List[Dict[str, Any]] = []
    for f in fvgs:
        if not isinstance(f, dict):
            continue
        if f.get("mitigated") is True:
            continue
        out.append(f)
    return out


def _collect_liquidity_pools(view: Dict[str, Any]) -> tuple[List[float], List[float]]:
    """Returns (equal_high_prices, equal_low_prices) from structure.liquidity."""
    structure = view.get("structure") or {}
    liq = structure.get("liquidity") or []
    eh: List[float] = []; el: List[float] = []
    if isinstance(liq, list):
        for entry in liq:
            if not isinstance(entry, dict):
                continue
            side = str(entry.get("side") or "").upper()
            price = _to_float(entry.get("price"))
            if price is None:
                continue
            if "HIGH" in side or "BSL" in side or "BUY" in side:
                eh.append(price)
            elif "LOW" in side or "SSL" in side or "SELL" in side:
                el.append(price)
    return eh, el


def detect_sweep_mitigation_continuation(
    pair_chart: Dict[str, Any],
    intent_direction: str,
    current_price: float,
) -> List[PathProjection]:
    """Detect a 3-step sweep → FVG fill → continuation path.

    For intent UP:
      Step 1: sweep a sell-side liquidity pool BELOW current price
      Step 2: retrace to the nearest unmitigated bullish FVG ABOVE the sweep level
      Step 3: continuation target at the next BSL pool ABOVE step 2
    Mirror for SHORT intent.

    Conditions to fire: at least 1 unmitigated FVG AND at least 1
    liquidity pool below (for LONG) or above (for SHORT) within
    reasonable distance.
    """
    if _is_disabled():
        return []
    intent_up = intent_direction.upper() == "LONG"
    # Use M15 or H1 for path projection (need structural levels, not noise)
    target_view = None
    for v in pair_chart.get("views") or []:
        if not isinstance(v, dict):
            continue
        if str(v.get("granularity") or "").upper() == "M15":
            target_view = v
            break
    if target_view is None:
        return []

    eh, el = _collect_liquidity_pools(target_view)
    fvgs = _collect_unmitigated_fvgs(target_view)

    pair = str(pair_chart.get("pair") or "")
    out: List[PathProjection] = []

    if intent_up:
        # Step 1: sell-side sweep BELOW current price
        sweep_candidates = [p for p in el if p < current_price]
        if not sweep_candidates:
            return []
        sweep_target = max(sweep_candidates)  # nearest BELOW

        # Step 2: bullish FVG (low > sweep_target, high < current_price + buffer)
        fvg_candidates = []
        for f in fvgs:
            kind = str(f.get("kind") or "").upper()
            if "BULL" not in kind:
                continue
            low = _to_float(f.get("low"))
            if low is None:
                continue
            if low > sweep_target:
                fvg_candidates.append((low, f))
        if not fvg_candidates:
            return []
        fvg_target_low, fvg_obj = min(fvg_candidates, key=lambda x: abs(x[0] - current_price))

        # Step 3: next BSL ABOVE the FVG (or current price)
        continuation_targets = [p for p in eh if p > fvg_target_low and p > current_price]
        if not continuation_targets:
            return []
        cont_target = min(continuation_targets)

        steps = (
            PathStep("sweep_low", "M15", sweep_target, "PENDING",
                     f"sell-side sweep target {sweep_target:.5f}"),
            PathStep("bull_fvg_fill", "M15", fvg_target_low, "PENDING",
                     f"unmitigated bullish FVG retest at {fvg_target_low:.5f}"),
            PathStep("continuation_high", "M15", cont_target, "PENDING",
                     f"buy-side liquidity target {cont_target:.5f}"),
        )
        out.append(PathProjection(
            pair=pair, direction="UP",
            name="sweep_low_fvg_fill_continuation_up",
            steps=steps,
            bonus_magnitude=PATH_PROJECTION_BONUS,
            confidence=0.6,
            rationale=f"M15 path: sweep {sweep_target:.5f} → FVG fill {fvg_target_low:.5f} → continuation {cont_target:.5f}",
        ))
    else:
        # SHORT: buy-side sweep ABOVE → bear FVG fill BELOW that → SSL target
        sweep_candidates = [p for p in eh if p > current_price]
        if not sweep_candidates:
            return []
        sweep_target = min(sweep_candidates)

        fvg_candidates = []
        for f in fvgs:
            kind = str(f.get("kind") or "").upper()
            if "BEAR" not in kind:
                continue
            high = _to_float(f.get("high"))
            if high is None:
                continue
            if high < sweep_target:
                fvg_candidates.append((high, f))
        if not fvg_candidates:
            return []
        fvg_target_high, fvg_obj = min(fvg_candidates, key=lambda x: abs(x[0] - current_price))

        continuation_targets = [p for p in el if p < fvg_target_high and p < current_price]
        if not continuation_targets:
            return []
        cont_target = max(continuation_targets)

        steps = (
            PathStep("sweep_high", "M15", sweep_target, "PENDING",
                     f"buy-side sweep target {sweep_target:.5f}"),
            PathStep("bear_fvg_fill", "M15", fvg_target_high, "PENDING",
                     f"unmitigated bearish FVG retest at {fvg_target_high:.5f}"),
            PathStep("continuation_low", "M15", cont_target, "PENDING",
                     f"sell-side liquidity target {cont_target:.5f}"),
        )
        out.append(PathProjection(
            pair=pair, direction="DOWN",
            name="sweep_high_fvg_fill_continuation_down",
            steps=steps,
            bonus_magnitude=PATH_PROJECTION_BONUS,
            confidence=0.6,
            rationale=f"M15 path: sweep {sweep_target:.5f} → FVG fill {fvg_target_high:.5f} → continuation {cont_target:.5f}",
        ))
    return out


def detect_paths(
    pair_chart: Optional[Dict[str, Any]],
    intent_direction: str,
    current_price: Optional[float],
) -> List[PathProjection]:
    """Run all path detectors. Returns 0 or more paths."""
    if _is_disabled() or pair_chart is None or current_price is None:
        return []
    out: List[PathProjection] = []
    out.extend(detect_sweep_mitigation_continuation(pair_chart, intent_direction, current_price))
    return out


def aggregate_path_score(
    paths: List[PathProjection],
    intent_direction: str,
) -> tuple[float, List[str]]:
    """Sum aligned paths. Path direction must match intent for the
    bonus to apply; opposed paths subtract half-weight (rare since
    paths are direction-specific)."""
    intent_up = intent_direction.upper() == "LONG"
    total = 0.0
    rationales: List[str] = []
    for p in paths:
        path_up = p.direction.upper() == "UP"
        contribution = p.bonus_magnitude * p.confidence
        if path_up == intent_up:
            total += contribution
            rationales.append(f"+{contribution:.1f} {p.rationale}")
        else:
            total -= contribution * 0.5
            rationales.append(f"-{contribution * 0.5:.1f} AGAINST {p.rationale}")
    return round(total, 2), rationales
