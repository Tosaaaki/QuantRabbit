"""Predictive LIMIT order generator — set the trap before price arrives.

User directive 2026-05-14:「予測して指値おいたり、予測してエントリー」.

Reads:
- `data/forward_projections_emit.json` (latest emit cache; or computed
  on demand)
- Current `pair_charts.json` for ATR / liquidity targets
- Current `broker_snapshot.json` for prices

For each pair with a HIGH-CONVICTION setup (Grade A: ≥4 aligned
projection signals OR an ACTIVE path projection), this module
generates a LIMIT order at the predicted entry level. It also allows
smaller Grade B "early-turn" liquidity-sweep limits when price is
already at an extreme and the sweep target is near; this catches the
start of a move without waiting for the whole M15/H1 confirmation stack.

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
# Grade B is intentionally smaller and stricter on context. It is for
# early turns at extremes, not for generic weak signals.
GRADE_B_MIN_ALIGNED = int(os.environ.get("QR_GRADE_B_MIN_ALIGNED", "1"))
# Default unit size for predictive limits
PREDICTIVE_LIMIT_UNITS = int(os.environ.get("QR_PREDICTIVE_LIMIT_UNITS", "5000"))
PREDICTIVE_LIMIT_GRADE_B_UNITS = int(
    os.environ.get("QR_PREDICTIVE_LIMIT_GRADE_B_UNITS", str(max(1000, PREDICTIVE_LIMIT_UNITS // 2)))
)
EARLY_TURN_EXTREME_PCTILE = float(os.environ.get("QR_EARLY_TURN_EXTREME_PCTILE", "0.25"))
EARLY_TURN_EXTREME_7D_PCTILE = float(os.environ.get("QR_EARLY_TURN_EXTREME_7D_PCTILE", "0.10"))


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

    Emits LIMITs when Grade A criteria are met:
    - ≥ GRADE_A_MIN_ALIGNED projection signals aligned on direction, AND
    - aggregate projection score ≥ GRADE_A_MIN_SCORE, OR
    - At least one active path projection (always Grade A).

    It also emits smaller Grade B liquidity-sweep fades when the same sweep
    signal appears at a price extreme with short-term exhaustion. This is a
    timing repair: place the trap before the confirmed reversal is obvious.
    """
    if _is_disabled():
        return []
    now = now or datetime.now(timezone.utc)
    gtd = (now + timedelta(minutes=LIMIT_TTL_MIN)).isoformat().replace("+00:00", "Z")
    out: List[PredictiveLimitOrder] = []
    seen_order_keys: set[tuple] = set()

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
        _append_unique(out, seen_order_keys, PredictiveLimitOrder(
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
        grade = "A"
        units = PREDICTIVE_LIMIT_UNITS
        context_note = f"{need_aligned} aligned signals"
        if need_aligned < GRADE_A_MIN_ALIGNED:
            early_context = _early_turn_context(pair_chart, fade_side)
            if need_aligned < GRADE_B_MIN_ALIGNED or early_context is None:
                continue
            grade = "B"
            units = PREDICTIVE_LIMIT_GRADE_B_UNITS
            context_note = f"{need_aligned} aligned {_signal_word(need_aligned)}; early-turn {early_context}"
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
        _append_unique(out, seen_order_keys, PredictiveLimitOrder(
            pair=pair, side=fade_side,
            limit_price=_round_price(pair, target_price),
            take_profit_price=tp_price,
            units=units,
            rationale=f"{signal_name} fade {fade_side} @ {target_price}; {context_note}",
            source="liquidity_sweep_fade",
            grade=grade,
            gtd_utc=gtd,
        ))

    return out


def _append_unique(
    out: List[PredictiveLimitOrder],
    seen_order_keys: set[tuple],
    order: PredictiveLimitOrder,
) -> None:
    key = (
        order.pair,
        order.side,
        order.limit_price,
        order.take_profit_price,
        order.source,
    )
    if key in seen_order_keys:
        return
    for idx, existing in enumerate(out):
        if not _same_predictive_trap(existing, order):
            continue
        if _predictive_order_rank(order) > _predictive_order_rank(existing):
            out[idx] = order
        return
    seen_order_keys.add(key)
    out.append(order)


def _same_predictive_trap(left: PredictiveLimitOrder, right: PredictiveLimitOrder) -> bool:
    if left.pair != right.pair or left.side != right.side or left.source != right.source:
        return False
    # One instrument pip is the duplicate bucket for this execution path:
    # liquidity levels often arrive once per timeframe with a few tenths of a
    # pip of candle-rounding drift, while the broker would treat stacked LIMITs
    # there as separate exposure. Keep this constant at the pip granularity
    # until projection signals carry a stable source-level id across timeframes.
    tolerance = 0.01 if left.pair.endswith("_JPY") else 0.0001
    if abs(float(left.limit_price) - float(right.limit_price)) > tolerance:
        return False
    if left.take_profit_price is None or right.take_profit_price is None:
        return left.take_profit_price is None and right.take_profit_price is None
    return abs(float(left.take_profit_price) - float(right.take_profit_price)) <= tolerance


def _predictive_order_rank(order: PredictiveLimitOrder) -> tuple[int, int]:
    grade_rank = 2 if str(order.grade).upper() == "A" else 1
    return grade_rank, int(order.units)


def _signal_word(count: int) -> str:
    return "signal" if count == 1 else "signals"


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _early_turn_context(pair_chart: Optional[Dict[str, Any]], fade_side: str) -> Optional[str]:
    if not pair_chart:
        return None
    side = str(fade_side or "").upper()
    if side not in {"LONG", "SHORT"}:
        return None
    confluence = pair_chart.get("confluence") or {}
    pct24 = _to_float(confluence.get("price_percentile_24h"))
    pct7 = _to_float(confluence.get("price_percentile_7d"))
    if side == "LONG":
        extreme = (
            (pct24 is not None and pct24 <= EARLY_TURN_EXTREME_PCTILE)
            or (pct7 is not None and pct7 <= EARLY_TURN_EXTREME_7D_PCTILE)
        )
    else:
        extreme = (
            (pct24 is not None and pct24 >= 1.0 - EARLY_TURN_EXTREME_PCTILE)
            or (pct7 is not None and pct7 >= 1.0 - EARLY_TURN_EXTREME_7D_PCTILE)
        )
    if not extreme:
        return None

    exhaustion_hits = 0
    reversal_hits = 0
    for view in pair_chart.get("views", []) or []:
        tf = str(view.get("granularity") or "").upper()
        if tf not in {"M1", "M5", "M15"}:
            continue
        indicators = view.get("indicators") or {}
        rsi = _to_float(indicators.get("rsi_14"))
        williams = _to_float(indicators.get("williams_r_14"))
        mfi = _to_float(indicators.get("mfi_14"))
        close = _to_float(indicators.get("close"))
        bb_lower = _to_float(indicators.get("bb_lower"))
        bb_upper = _to_float(indicators.get("bb_upper"))
        bb_middle = _to_float(indicators.get("bb_middle"))
        if side == "LONG":
            if (
                (rsi is not None and rsi <= 40.0)
                or (williams is not None and williams <= -80.0)
                or (mfi is not None and mfi <= 35.0)
                or _close_near_band(close, bb_lower, bb_middle, lower_side=True)
            ):
                exhaustion_hits += 1
        else:
            if (
                (rsi is not None and rsi >= 60.0)
                or (williams is not None and williams >= -20.0)
                or (mfi is not None and mfi >= 65.0)
                or _close_near_band(close, bb_upper, bb_middle, lower_side=False)
            ):
                exhaustion_hits += 1
        last_event = (view.get("structure") or {}).get("last_event") or {}
        kind = str(last_event.get("kind") or "").upper()
        if bool(last_event.get("close_confirmed")) and (
            (side == "LONG" and kind.endswith("_UP"))
            or (side == "SHORT" and kind.endswith("_DOWN"))
        ):
            reversal_hits += 1
    if exhaustion_hits <= 0:
        return None
    if reversal_hits > 0:
        return f"extreme+exhaustion({exhaustion_hits})+micro_flip({reversal_hits})"
    return f"extreme+exhaustion({exhaustion_hits})"


def _close_near_band(
    close: Optional[float],
    band: Optional[float],
    middle: Optional[float],
    *,
    lower_side: bool,
) -> bool:
    if close is None or band is None:
        return False
    if middle is None or middle == band:
        return close <= band if lower_side else close >= band
    quarter_band = abs(middle - band) * 0.25
    if lower_side:
        return close <= band + quarter_band
    return close >= band - quarter_band


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
    confirm_live: bool = False,
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
        if not _predictive_limit_live_send_enabled(confirm_live):
            entry["error"] = (
                "PREDICTIVE_LIMIT_LIVE_GATE_BLOCKED: set QR_LIVE_ENABLED=1 and pass "
                "--send --confirm-live before predictive LIMIT orders may be sent"
            )
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


def _predictive_limit_live_send_enabled(confirm_live: bool) -> bool:
    if not confirm_live:
        return False
    return os.environ.get("QR_LIVE_ENABLED", "").strip() in {"1", "true", "TRUE", "yes", "YES"}
