"""Dynamic TP rebalancing on open positions.

AGENT_CONTRACT §10 historically said "Existing TP is not moved by the
protection gateway." That rule was the conservative default to stop
the protection layer from silently shrinking a planned TP. But static
entry-time TP ignores live market conditions — when the regime
expands, the original TP is too tight; when it contracts, the
original TP becomes unreachable.

User directive 2026-05-13:
  「エントリー時だけではなく、途中で伸び縮みできるようにしてほしい。
   市況によって。それって当たり前だよね？」

This module reads each trader-owned position's current pair_charts
context, recomputes the market-derived reward_risk (same function as
intent_generator), derives a fresh TP distance from current ATR, and
adjusts the broker's TP order if the change exceeds the hysteresis.

Invariants:
- TP must remain on the correct side of entry (LONG: above; SHORT:
  below) and at least `MIN_TP_TO_MARKET_PIPS` from current price so
  the rebalance never fires the TP accidentally on the same tick.
- Only trader-owned positions are touched. Manual / unknown-owner
  positions are skipped — operator discretion is preserved.
- SL is NEVER touched here. This module is TP-only. The SL-free
  invariant `stop_loss is None` is respected by skipping any
  attempt to read/write SL.
- Hysteresis: ignore changes smaller than `HYSTERESIS_PIPS` to avoid
  noise-driven broker churn.
- Kill switch: `QR_DISABLE_TP_REBALANCE=1` short-circuits the
  rebalancer to a no-op.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


HYSTERESIS_PIPS = float(os.environ.get("QR_TP_REBALANCE_HYSTERESIS_PIPS", "10"))
MIN_TP_TO_MARKET_PIPS = float(os.environ.get("QR_TP_REBALANCE_MIN_TP_TO_MARKET", "5"))
MAX_TP_DISTANCE_ATR_MULT = float(os.environ.get("QR_TP_REBALANCE_MAX_DISTANCE_ATR", "10"))

# Contract-mode tuning (2026-05-13 second iteration after user feedback).
# When a position is ≥ ADVERSE_ATR_MULT × ATR underwater AND no reversal
# signal is firing, the trader pulls TP closer to entry to lock in a
# small bounce-back profit instead of waiting for the original wide
# target that may never be hit. User directive:「下げ基調のとき TP を狭める
# のはいい。下げを否定し始めたときに TP 広げて利鞘を稼ぐ」.
ADVERSE_ATR_MULT = float(os.environ.get("QR_TP_REBALANCE_ADVERSE_ATR_MULT", "1.0"))
MIN_LOCK_IN_PIPS = float(os.environ.get("QR_TP_REBALANCE_MIN_LOCK_IN_PIPS", "8"))
LOCK_IN_ATR_MULT = float(os.environ.get("QR_TP_REBALANCE_LOCK_IN_ATR_MULT", "0.5"))


@dataclass(frozen=True)
class TPAdjustment:
    trade_id: str
    pair: str
    side: str
    entry_price: float
    current_tp: float
    new_tp: float
    distance_pips_old: float
    distance_pips_new: float
    rationale: str


def _pip_factor(pair: str) -> int:
    return 100 if pair.endswith("_JPY") else 10000


def _round_price(pair: str, price: float) -> float:
    return round(price, 3 if pair.endswith("_JPY") else 5)


def _is_disabled() -> bool:
    return os.environ.get("QR_DISABLE_TP_REBALANCE", "").strip() in {"1", "true", "TRUE", "yes", "YES"}


def compute_tp_adjustment(
    *,
    trade_id: str,
    pair: str,
    side: str,
    entry_price: float,
    current_tp: Optional[float],
    current_price: float,
    atr_pips: float,
    reward_risk: float,
    is_reversal_firing: bool = False,
    owner: str = "trader",
) -> Optional[TPAdjustment]:
    """Compute a new TP for one position.

    Three modes (2026-05-13 iteration after the 471029 regression):

    1. **expand_reversal** — reversal signal fires for this side:
       use the full `reward_risk × ATR` distance from entry, letting
       the bounce run further. May expand or be a no-op (never
       contract here — reversal means we want MORE room).
    2. **contract_adverse** — position is ≥ `ADVERSE_ATR_MULT × ATR`
       underwater AND no reversal signal: pull TP to
       `entry + max(MIN_LOCK_IN_PIPS, atr × LOCK_IN_ATR_MULT)` (LONG;
       mirrored for SHORT). User directive 2026-05-13:「下げ基調の
       とき TP を狭めるのはいい。下げを否定し始めたときに TP 広げて
       利鞘を稼がないといけないんじゃないの？」 — settle for a small
       bounce-back profit instead of waiting for the original wide
       target that may never be hit.
    3. **expand_only** — otherwise (in profit, or barely adverse, no
       reversal signal): TP can only move FURTHER from entry. The
       original "let winners run" rule.

    Returns None when the position should not be adjusted (wrong
    owner, no existing TP, change below hysteresis, safety violation).
    """
    if _is_disabled():
        return None
    if owner != "trader":
        return None
    if current_tp is None:
        return None
    if atr_pips <= 0 or reward_risk <= 0:
        return None

    pip_factor = _pip_factor(pair)
    pip_size = 1.0 / pip_factor
    side_up = side.upper()
    if side_up not in ("LONG", "SHORT"):
        return None

    distance_old = abs(current_tp - entry_price) * pip_factor
    desired_distance_pips = min(reward_risk * atr_pips, MAX_TP_DISTANCE_ATR_MULT * atr_pips)

    # Adverse detection (only matters for contract_adverse mode).
    if side_up == "LONG":
        is_adverse = current_price < entry_price
        adverse_pips = (entry_price - current_price) * pip_factor
    else:
        is_adverse = current_price > entry_price
        adverse_pips = (current_price - entry_price) * pip_factor
    is_significant_adverse = is_adverse and adverse_pips >= ADVERSE_ATR_MULT * atr_pips

    # Pick mode.
    if is_reversal_firing:
        mode = "expand_reversal"
        if side_up == "LONG":
            candidate_tp = entry_price + desired_distance_pips * pip_size
        else:
            candidate_tp = entry_price - desired_distance_pips * pip_size
        # In reversal mode we never want to contract; if the desired
        # distance is shorter than the old TP, leave it alone.
        if desired_distance_pips <= distance_old:
            return None
    elif is_significant_adverse:
        mode = "contract_adverse"
        lock_in_pips = max(MIN_LOCK_IN_PIPS, atr_pips * LOCK_IN_ATR_MULT)
        if side_up == "LONG":
            candidate_tp = entry_price + lock_in_pips * pip_size
        else:
            candidate_tp = entry_price - lock_in_pips * pip_size
        # Only fire if the lock-in TP is actually CLOSER than the
        # existing TP (we're contracting). If it would EXPAND, fall
        # through to expand_only mode which has stricter rules.
        if abs(candidate_tp - entry_price) >= distance_old:
            mode = "expand_only"
            if desired_distance_pips <= distance_old:
                return None
            if side_up == "LONG":
                candidate_tp = entry_price + desired_distance_pips * pip_size
            else:
                candidate_tp = entry_price - desired_distance_pips * pip_size
    else:
        mode = "expand_only"
        if desired_distance_pips <= distance_old:
            return None
        if side_up == "LONG":
            candidate_tp = entry_price + desired_distance_pips * pip_size
        else:
            candidate_tp = entry_price - desired_distance_pips * pip_size

    # Safety: TP must not fire immediately. Keep at least
    # MIN_TP_TO_MARKET_PIPS distance from current price. TP must remain
    # on the correct side of entry (avoid locking in a loss).
    safety_margin = MIN_TP_TO_MARKET_PIPS * pip_size
    if side_up == "LONG":
        if candidate_tp < current_price + safety_margin:
            return None
        if candidate_tp <= entry_price:
            return None
    else:
        if candidate_tp > current_price - safety_margin:
            return None
        if candidate_tp >= entry_price:
            return None

    new_tp = _round_price(pair, candidate_tp)
    change_pips = abs(new_tp - current_tp) * pip_factor
    if change_pips < HYSTERESIS_PIPS:
        return None

    distance_new = abs(new_tp - entry_price) * pip_factor
    direction_label = "expanded" if distance_new > distance_old else "contracted"
    rationale = (
        f"TP {direction_label} {distance_old:.1f}→{distance_new:.1f}pip "
        f"(mode={mode}, reward_risk={reward_risk:.2f}, atr={atr_pips:.1f}pip, "
        f"change={change_pips:.1f}pip)"
    )
    return TPAdjustment(
        trade_id=trade_id,
        pair=pair,
        side=side_up,
        entry_price=entry_price,
        current_tp=current_tp,
        new_tp=new_tp,
        distance_pips_old=distance_old,
        distance_pips_new=distance_new,
        rationale=rationale,
    )


def compute_all_tp_adjustments(
    *,
    positions: Iterable[Any],
    quotes: Dict[str, Dict[str, float]],
    pair_charts: Dict[str, Dict[str, Any]],
    market_reward_risk_fn,
) -> list[TPAdjustment]:
    """Loop trader-owned positions and compute TP adjustments.

    `quotes` is the per-pair bid/ask dict from broker_snapshot.
    `pair_charts` is the keyed-by-pair dict (same shape as
    `trader_brain._load_full_pair_charts_for_brain`'s return value).
    `market_reward_risk_fn(chart_context)` is injected so the caller
    can use the same dynamic reward_risk computation as
    intent_generator (avoids a circular import).
    """
    if _is_disabled():
        return []
    adjustments: list[TPAdjustment] = []
    for position in positions:
        owner = getattr(position, "owner", None)
        owner_str = owner.value if hasattr(owner, "value") else str(owner or "")
        if owner_str.lower() != "trader":
            continue
        pair = getattr(position, "pair", None)
        if not pair or pair not in quotes:
            continue
        # Current price for the side's exit:
        # LONG exits with sell at bid; SHORT exits with buy at ask.
        side = getattr(position, "side", None)
        side_value = side.value if hasattr(side, "value") else str(side or "")
        side_up = side_value.upper()
        quote = quotes.get(pair) or {}
        if side_up == "LONG":
            current_price = float(quote.get("bid") or 0.0)
        elif side_up == "SHORT":
            current_price = float(quote.get("ask") or 0.0)
        else:
            continue
        if current_price <= 0:
            continue

        chart = pair_charts.get(pair) or {}
        # Compute fresh reward_risk via injected fn (intent_generator's
        # `_market_derived_reward_risk` returns (rr, rationale_list)).
        try:
            reward_risk_value, _ = market_reward_risk_fn(_chart_context_from_chart(chart))
        except Exception:
            continue

        atr_pips = _extract_atr_pips(chart, pair)
        if atr_pips is None or atr_pips <= 0:
            continue

        # Check reversal signal for the position's direction. If firing,
        # tp_rebalancer enters expand_reversal mode and the contract
        # branch is short-circuited.
        try:
            from quant_rabbit.strategy.reversal_signal import detect_reversal as _detect_reversal
            reversal = _detect_reversal(chart, side_up)
        except Exception:
            reversal = None

        adj = compute_tp_adjustment(
            trade_id=str(getattr(position, "trade_id", "")),
            pair=pair,
            side=side_up,
            entry_price=float(getattr(position, "entry_price", 0.0)),
            current_tp=_optional_float(getattr(position, "take_profit", None)),
            current_price=current_price,
            atr_pips=atr_pips,
            reward_risk=float(reward_risk_value),
            is_reversal_firing=(reversal is not None),
            owner=owner_str.lower(),
        )
        if adj is not None:
            adjustments.append(adj)
    return adjustments


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _chart_context_from_chart(chart: Dict[str, Any]) -> Dict[str, Any]:
    """Project the full pair_chart entry to the flat context shape that
    `_market_derived_reward_risk` expects.

    pair_charts views use `granularity` (not `timeframe`) and put
    indicators under `indicators.{adx_14, atr_pips, ...}`.
    """
    context: Dict[str, Any] = {}
    context["confluence"] = chart.get("confluence") or {}
    for view in chart.get("views", []) or []:
        if not isinstance(view, dict):
            continue
        tf = str(view.get("granularity") or view.get("timeframe") or view.get("tf") or "").upper()
        indicators = view.get("indicators") or {}
        adx = indicators.get("adx_14") or indicators.get("adx") or indicators.get("ADX")
        if adx is None:
            continue
        if tf == "H1":
            context["h1_adx"] = adx
        elif tf == "H4":
            context["h4_adx"] = adx
    context["session_current_tag"] = (
        chart.get("session_current_tag")
        or chart.get("session_bucket")
        or chart.get("session")
        or (chart.get("confluence") or {}).get("session_current_tag")
    )
    return context


def _extract_atr_pips(chart: Dict[str, Any], pair: str) -> Optional[float]:
    """Pull current ATR (in pips) from chart.

    Preference order (pair_charts schema 2026-05-13):
    1. confluence.h4_atr_pips (intent_generator pipeline projection)
    2. confluence.atr_pips
    3. views[granularity=H4].indicators.atr_pips
    4. views[granularity=H1].indicators.atr_pips
    5. Any view's indicators.atr_pips
    """
    confluence = chart.get("confluence") or {}
    for key in ("h4_atr_pips", "h1_atr_pips", "atr_pips"):
        raw = confluence.get(key)
        if raw is None:
            continue
        try:
            v = float(raw)
            if v > 0:
                return v
        except (TypeError, ValueError):
            continue

    # Preferred per-granularity lookup.
    preference = ("H4", "H1", "M30", "M15", "M5", "M1", "D")
    by_gran: Dict[str, float] = {}
    for view in chart.get("views", []) or []:
        if not isinstance(view, dict):
            continue
        tf = str(view.get("granularity") or view.get("timeframe") or view.get("tf") or "").upper()
        indicators = view.get("indicators") or {}
        raw = indicators.get("atr_pips")
        if raw is None:
            continue
        try:
            v = float(raw)
            if v > 0:
                by_gran[tf] = v
        except (TypeError, ValueError):
            continue
    for tf in preference:
        if tf in by_gran:
            return by_gran[tf]
    return None


def apply_tp_adjustments(
    adjustments: Iterable[TPAdjustment],
    broker_client: Any,
    *,
    dry_run: bool = False,
) -> list[dict]:
    """Send the TP modify requests to the broker.

    Returns a list of result dicts (one per adjustment, including
    failed calls — broker exceptions are logged into the dict, never
    raised, so a single failure doesn't block other adjustments).
    """
    results: list[dict] = []
    for adj in adjustments:
        entry = {
            "trade_id": adj.trade_id,
            "pair": adj.pair,
            "side": adj.side,
            "current_tp": adj.current_tp,
            "new_tp": adj.new_tp,
            "distance_pips_old": adj.distance_pips_old,
            "distance_pips_new": adj.distance_pips_new,
            "rationale": adj.rationale,
            "sent": False,
            "error": None,
        }
        if dry_run:
            results.append(entry)
            continue
        try:
            broker_client.replace_trade_dependent_orders(
                adj.trade_id,
                {"takeProfit": {"price": f"{adj.new_tp:.5f}".rstrip("0").rstrip(".") if not adj.pair.endswith("_JPY") else f"{adj.new_tp:.3f}", "timeInForce": "GTC"}},
            )
            entry["sent"] = True
        except Exception as exc:  # noqa: BLE001 — keep loop running
            entry["error"] = str(exc)
        results.append(entry)
    return results
