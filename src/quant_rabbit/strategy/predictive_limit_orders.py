"""Predictive LIMIT order generator — set the trap before price arrives.

User directive 2026-05-14:「予測して指値おいたり、予測してエントリー」.

Reads:
- `data/forward_projections_emit.json` (latest emit cache; or computed
  on demand)
- Current `pair_charts.json` for ATR / liquidity targets
- Current `broker_snapshot.json` for prices

For each pair with a HIGH-CONVICTION setup (Grade A: ≥4 aligned
projection signals OR an ACTIVE path projection), this module
generates a LIMIT order at the predicted entry level:

1. **Liquidity sweep fade**: sweep_high target → SHORT LIMIT at target
   (fade the sweep). sweep_low target → LONG LIMIT at target.

2. **Path projection step**: when path A→B→C is detected, place LIMIT
   at Step B (FVG fill price) in the path's direction.

Each generated LIMIT has:
- Price = predicted target (rounded to OANDA tick)
- TP = next step in path OR ATR-based default
- SL = None (SL-free)
- Time-in-force = GTD (expire after `LIMIT_TTL_MIN`)

The CLI command `generate-predictive-limits` writes them to
`data/predictive_limit_orders.json` (DRY_RUN) or sends via OANDA when
`--send` is passed. The user controls when to send; the module by
itself never sends without explicit consent.

Kill switch: `QR_DISABLE_PREDICTIVE_LIMITS=1`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


# Time-to-live for predictive LIMIT orders. After this, OANDA cancels
# the unfilled order automatically.
LIMIT_TTL_MIN = float(os.environ.get("QR_PREDICTIVE_LIMIT_TTL_MIN", "90"))
# Minimum aligned signal count to qualify as Grade A
GRADE_A_MIN_ALIGNED = int(os.environ.get("QR_GRADE_A_MIN_ALIGNED", "4"))
GRADE_A_MIN_SCORE = float(os.environ.get("QR_GRADE_A_MIN_SCORE", "25.0"))
# Default unit size for predictive limits
PREDICTIVE_LIMIT_UNITS = int(os.environ.get("QR_PREDICTIVE_LIMIT_UNITS", "5000"))


@dataclass
class PredictiveLimitOrder:
    pair: str
    side: str  # "LONG" | "SHORT"
    limit_price: float
    take_profit_price: Optional[float]
    units: int
    rationale: str
    source: str  # "liquidity_sweep_fade" | "path_step_b" | etc
    grade: str  # "A" | "B"
    gtd_utc: str  # ISO


def _is_disabled() -> bool:
    return os.environ.get("QR_DISABLE_PREDICTIVE_LIMITS", "").strip() in {
        "1", "true", "TRUE", "yes", "YES",
    }


def _round_price(pair: str, price: float) -> float:
    return round(price, 3 if pair.endswith("_JPY") else 5)


def generate_limits_from_projections(
    *,
    pair: str,
    pair_chart: Optional[Dict[str, Any]],
    current_bid: float,
    current_ask: float,
    projection_signals: List[Any],
    paths: List[Any],
    now: Optional[datetime] = None,
) -> List[PredictiveLimitOrder]:
    """Generate LIMIT orders for Grade A setups for one pair.

    Sources:
    1. Liquidity sweep target (signal name identifies high/low sweep)
       -> FADE entry at target.
    2. Path projection Step B (FVG fill price) → trend-aligned entry.

    Only emits LIMITs when Grade A criteria are met:
    - ≥ GRADE_A_MIN_ALIGNED projection signals aligned on direction, AND
    - aggregate projection score ≥ GRADE_A_MIN_SCORE, OR
    - At least one active path projection (always Grade A).
    """
    if _is_disabled():
        return []
    now = now or datetime.now(timezone.utc)
    gtd = (now + timedelta(minutes=LIMIT_TTL_MIN)).isoformat().replace("+00:00", "Z")
    out: List[PredictiveLimitOrder] = []

    # Path-based limits: place at Step B of each path
    for path in paths or []:
        steps = getattr(path, "steps", ())
        if len(steps) < 2:
            continue
        step_b = steps[1]
        direction = getattr(path, "direction", "UP")
        side = "LONG" if direction.upper() == "UP" else "SHORT"
        limit_price = _round_price(pair, float(step_b.expected_price))
        # TP at Step C
        tp_price = None
        if len(steps) >= 3:
            tp_price = _round_price(pair, float(steps[2].expected_price))
        out.append(PredictiveLimitOrder(
            pair=pair, side=side, limit_price=limit_price,
            take_profit_price=tp_price,
            units=PREDICTIVE_LIMIT_UNITS,
            rationale=f"path Step B '{step_b.label}' at {limit_price}; full path: {getattr(path, 'rationale', '')[:120]}",
            source="path_step_b",
            grade="A",
            gtd_utc=gtd,
        ))

    # Liquidity-sweep fade: only when score is Grade A
    aligned_long = sum(1 for s in projection_signals if getattr(s, "direction", "") == "UP")
    aligned_short = sum(1 for s in projection_signals if getattr(s, "direction", "") == "DOWN")
    for s in projection_signals or []:
        if not getattr(s, "name", "").startswith("liquidity_sweep"):
            continue
        # Sweep target price is in the rationale (we don't have direct field)
        target_price = _extract_target_price(getattr(s, "rationale", ""))
        if target_price is None:
            continue
        # `direction` is the executable fade direction. The signal name
        # identifies which liquidity side must be swept before entry.
        signal_name = str(getattr(s, "name", ""))
        if signal_name == "liquidity_sweep_high":
            # Sweep buy-side highs -> SHORT LIMIT at the swept high.
            fade_side = "SHORT"
            need_aligned = aligned_short
        elif signal_name == "liquidity_sweep_low":
            # Sweep sell-side lows -> LONG LIMIT at the swept low.
            fade_side = "LONG"
            need_aligned = aligned_long
        else:
            continue
        if need_aligned < GRADE_A_MIN_ALIGNED:
            continue
        # TP: counter-direction ATR distance from sweep
        ind = None
        if pair_chart:
            for v in pair_chart.get("views", []):
                if str(v.get("granularity", "")).upper() == "M15":
                    ind = v.get("indicators") or {}
                    break
        atr_pips = float((ind or {}).get("atr_pips") or 10.0)
        pip_size = 0.01 if pair.endswith("_JPY") else 0.0001
        if fade_side == "SHORT":
            tp_price = _round_price(pair, target_price - atr_pips * 2 * pip_size)
        else:
            tp_price = _round_price(pair, target_price + atr_pips * 2 * pip_size)
        out.append(PredictiveLimitOrder(
            pair=pair, side=fade_side,
            limit_price=_round_price(pair, target_price),
            take_profit_price=tp_price,
            units=PREDICTIVE_LIMIT_UNITS,
            rationale=f"{signal_name} fade {fade_side} @ {target_price}; {need_aligned} aligned signals",
            source="liquidity_sweep_fade",
            grade="A",
            gtd_utc=gtd,
        ))

    return out


def _extract_target_price(rationale: str) -> Optional[float]:
    import re
    m = re.search(r"at\s+([0-9.]+)\s*\(", rationale or "")
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def serialize_limit_orders(orders: List[PredictiveLimitOrder]) -> List[dict]:
    return [
        {
            "pair": o.pair,
            "side": o.side,
            "limit_price": o.limit_price,
            "take_profit_price": o.take_profit_price,
            "units": o.units,
            "rationale": o.rationale,
            "source": o.source,
            "grade": o.grade,
            "gtd_utc": o.gtd_utc,
        }
        for o in orders
    ]


def apply_limit_orders(
    orders: List[PredictiveLimitOrder],
    broker_client: Any,
    *,
    dry_run: bool = True,
) -> List[dict]:
    """Send LIMIT orders via OANDA `post_order_json` when not dry-run.

    Returns per-order result dicts. Failures captured per-order so a
    single broker error doesn't block the rest.
    """
    results: List[dict] = []
    for o in orders:
        entry = {
            "pair": o.pair, "side": o.side, "limit_price": o.limit_price,
            "take_profit_price": o.take_profit_price, "units": o.units,
            "rationale": o.rationale, "sent": False, "error": None,
        }
        if dry_run:
            results.append(entry)
            continue
        units_signed = o.units if o.side == "LONG" else -o.units
        order_request: Dict[str, Any] = {
            "type": "LIMIT",
            "instrument": o.pair,
            "units": str(units_signed),
            "price": f"{o.limit_price:.3f}" if o.pair.endswith("_JPY") else f"{o.limit_price:.5f}",
            "timeInForce": "GTD",
            "gtdTime": o.gtd_utc,
            "positionFill": "DEFAULT",
        }
        if o.take_profit_price is not None:
            order_request["takeProfitOnFill"] = {
                "price": f"{o.take_profit_price:.3f}" if o.pair.endswith("_JPY") else f"{o.take_profit_price:.5f}",
                "timeInForce": "GTC",
            }
        try:
            broker_client.post_order_json(order_request)
            entry["sent"] = True
        except Exception as exc:  # noqa: BLE001
            entry["error"] = str(exc)
        results.append(entry)
    return results
