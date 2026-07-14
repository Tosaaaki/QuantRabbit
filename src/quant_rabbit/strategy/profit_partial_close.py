"""Profit-side partial closes for runner management.

This is deliberately separate from ``adverse_partial_close``. The adverse
module was disabled because it realized losses from a lagging P/L threshold.
This module acts only on trader-owned positions that are already in profit:
bank part of the move at ATR-derived milestones, keep the remainder running,
and remember the last milestone per trade so the same profit band is not
closed repeatedly. Manual, tagless, operator-managed, and external positions
remain out of scope unless a future content-addressed operator authorization
contract is implemented at both calculation and fresh-broker send boundaries.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from quant_rabbit.paths import DEFAULT_PROFIT_PARTIAL_CLOSE_STATE
from quant_rabbit.predictive_scout import predictive_scout_broker_raw_claimed


# These are profit-partial workflow thresholds, not an entry-size floor.
# OANDA accepts positive integer FX units and sub-1,000u positions remain valid.
# The partial-close bot only splits positions from 2,000u upward and keeps a
# 1,000u runner so it does not fragment a small profitable position into many
# broker close transactions; smaller positions stay intact for their TP/normal
# management path. Operators can override the workflow thresholds explicitly.
MIN_POSITION_UNITS_FOR_PROFIT_PARTIAL = int(os.environ.get("QR_PROFIT_PARTIAL_MIN_UNITS", "2000"))
MIN_RUNNER_UNITS_AFTER_PROFIT_PARTIAL = int(os.environ.get("QR_PROFIT_PARTIAL_MIN_RUNNER_UNITS", "1000"))
PROFIT_PARTIAL_ROUND_UNITS = int(os.environ.get("QR_PROFIT_PARTIAL_ROUND_UNITS", "100"))
PROFIT_PARTIAL_CLOSE_PROVENANCE = "profit_partial_close"


@dataclass(frozen=True)
class ProfitPartialCloseAction:
    trade_id: str
    pair: str
    side: str
    original_units: int
    close_units: int
    remaining_units: int
    profit_pips: float
    atr_pips: float
    trigger_mult: float
    fraction: float
    milestone: int
    prior_milestone: int
    rationale: str
    owner: str = "unknown"


def _pip_factor(pair: str) -> int:
    return 100 if pair.endswith("_JPY") else 10000


def _is_disabled() -> bool:
    return os.environ.get("QR_DISABLE_PROFIT_PARTIAL_CLOSE", "").strip() in {
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
    }


def _normalized_owner(owner: Any) -> str:
    value = getattr(owner, "value", owner)
    if not isinstance(value, str):
        return "unknown"
    normalized = value.strip().lower()
    return normalized or "unknown"


def _profit_take_owner_allowed(owner: Any) -> bool:
    return _normalized_owner(owner) == "trader"


def compute_profit_partial_close(
    *,
    trade_id: str,
    pair: str,
    side: str,
    units: int,
    entry_price: float,
    current_price: float,
    atr_pips: float,
    owner: str = "unknown",
    chart_context: Optional[Dict[str, Any]] = None,
    last_milestone: int = 0,
) -> Optional[ProfitPartialCloseAction]:
    if _is_disabled():
        return None
    if not _profit_take_owner_allowed(owner):
        return None
    if atr_pips <= 0 or current_price <= 0 or entry_price <= 0:
        return None

    side_up = side.upper()
    pip_factor = _pip_factor(pair)
    if side_up == "LONG":
        profit_pips = (current_price - entry_price) * pip_factor
    elif side_up == "SHORT":
        profit_pips = (entry_price - current_price) * pip_factor
    else:
        return None
    if profit_pips <= 0:
        return None

    from quant_rabbit.strategy.dynamic_position_policy import (
        partial_close_fraction,
        trailing_trigger_mult,
    )

    trigger_mult, trigger_reasons = trailing_trigger_mult(chart_context)
    threshold_pips = trigger_mult * atr_pips
    if threshold_pips <= 0:
        return None
    # Float arithmetic can render an exact 3.0 milestone as 2.999999999;
    # the epsilon is an engineering tolerance, not a market threshold.
    milestone = math.floor((profit_pips / threshold_pips) + 1e-9)
    if milestone <= 0 or milestone <= int(last_milestone or 0):
        return None

    absolute_units = abs(int(units))
    if absolute_units < MIN_POSITION_UNITS_FOR_PROFIT_PARTIAL:
        return None
    fraction, fraction_reasons = partial_close_fraction(chart_context, side_up)
    close_units = int(absolute_units * fraction)
    close_units = (close_units // PROFIT_PARTIAL_ROUND_UNITS) * PROFIT_PARTIAL_ROUND_UNITS
    if close_units <= 0:
        return None
    if absolute_units - close_units < MIN_RUNNER_UNITS_AFTER_PROFIT_PARTIAL:
        close_units = absolute_units - MIN_RUNNER_UNITS_AFTER_PROFIT_PARTIAL
        close_units = (close_units // PROFIT_PARTIAL_ROUND_UNITS) * PROFIT_PARTIAL_ROUND_UNITS
    if close_units <= 0 or close_units >= absolute_units:
        return None

    remaining = absolute_units - close_units
    rationale_bits = [
        f"profit {profit_pips:.1f}pip reached milestone {milestone}",
        f"trigger={trigger_mult:.2f}×ATR ({threshold_pips:.1f}pip)",
        f"close {close_units}/{absolute_units} ({fraction:.0%})",
        f"runner {remaining}",
    ]
    if trigger_reasons:
        rationale_bits.append("trigger: " + "; ".join(trigger_reasons[:2]))
    if fraction_reasons:
        rationale_bits.append("fraction: " + "; ".join(fraction_reasons[:2]))
    return ProfitPartialCloseAction(
        trade_id=trade_id,
        pair=pair,
        side=side_up,
        original_units=absolute_units,
        close_units=close_units,
        remaining_units=remaining,
        profit_pips=profit_pips,
        atr_pips=atr_pips,
        trigger_mult=trigger_mult,
        fraction=fraction,
        milestone=milestone,
        prior_milestone=int(last_milestone or 0),
        rationale="; ".join(rationale_bits),
        owner=_normalized_owner(owner),
    )


def compute_all_profit_partial_closes(
    *,
    positions: Iterable[Any],
    quotes: Dict[str, Dict[str, float]],
    pair_charts: Dict[str, Dict[str, Any]],
    state: Dict[str, Any] | None = None,
) -> list[ProfitPartialCloseAction]:
    if _is_disabled():
        return []
    from quant_rabbit.strategy.tp_rebalancer import (
        _chart_context_from_chart,
        _extract_atr_pips,
    )

    state = state or {}
    milestones = state.get("trade_milestones") if isinstance(state.get("trade_milestones"), dict) else {}
    actions: list[ProfitPartialCloseAction] = []
    for position in positions:
        if predictive_scout_broker_raw_claimed(getattr(position, "raw", None)):
            continue
        owner = getattr(position, "owner", None)
        owner_str = owner.value if hasattr(owner, "value") else str(owner or "")
        if not _profit_take_owner_allowed(owner_str):
            continue
        pair = getattr(position, "pair", None)
        if not pair or pair not in quotes:
            continue
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
        chart = pair_charts.get(pair) or {}
        atr_pips = _extract_atr_pips(chart, pair)
        if atr_pips is None or atr_pips <= 0:
            continue
        trade_id = str(getattr(position, "trade_id", ""))
        action = compute_profit_partial_close(
            trade_id=trade_id,
            pair=pair,
            side=side_up,
            units=int(getattr(position, "units", 0)),
            entry_price=float(getattr(position, "entry_price", 0.0)),
            current_price=current_price,
            atr_pips=atr_pips,
            owner=owner_str.lower(),
            chart_context=_chart_context_from_chart(chart),
            last_milestone=int(milestones.get(trade_id, 0) or 0),
        )
        if action is not None:
            actions.append(action)
    return actions


def apply_profit_partial_closes(
    actions: Iterable[ProfitPartialCloseAction],
    broker_client: Any,
    *,
    send: bool = False,
    live_enabled: bool = False,
    confirm_live: bool = False,
    forbidden_trade_reasons: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    forbidden_reasons = {
        str(trade_id): str(reason)
        for trade_id, reason in (forbidden_trade_reasons or {}).items()
    }
    for action in actions:
        action_owner = _normalized_owner(action.owner)
        entry: dict[str, Any] = {
            "trade_id": action.trade_id,
            "pair": action.pair,
            "side": action.side,
            "original_units": action.original_units,
            "close_units": action.close_units,
            "remaining_units": action.remaining_units,
            "profit_pips": round(action.profit_pips, 3),
            "atr_pips": round(action.atr_pips, 3),
            "milestone": action.milestone,
            "prior_milestone": action.prior_milestone,
            "rationale": action.rationale,
            "sent": False,
            "broker_post_attempted": False,
            "error": None,
            "response": None,
            "provenance": PROFIT_PARTIAL_CLOSE_PROVENANCE,
            "owner": action_owner,
            "action_owner": action_owner,
            "fresh_owner_validated": False,
            "fresh_send_boundary_validated": False,
        }
        if not send:
            if not _profit_take_owner_allowed(action_owner):
                entry["error"] = (
                    "NON_TRADER_PROFIT_PARTIAL_CLOSE_FORBIDDEN: manual, tagless, "
                    "operator-managed, unknown-owner, and external positions are monitor-only"
                )
            results.append(entry)
            continue
        if not live_enabled:
            entry["error"] = "LIVE_DISABLED: profit partial close requires QR_LIVE_ENABLED=1"
            results.append(entry)
            continue
        if not confirm_live:
            entry["error"] = "LIVE_CONFIRMATION_REQUIRED: pass --confirm-live with --send"
            results.append(entry)
            continue

        fresh_owner, fresh_evidence, fresh_error = _fresh_profit_partial_close_send_boundary(
            broker_client=broker_client,
            action=action,
        )
        entry["owner"] = fresh_owner
        entry["fresh_broker_evidence"] = fresh_evidence
        entry["fresh_owner_validated"] = fresh_owner == "trader"
        entry["fresh_send_boundary_validated"] = fresh_error is None
        if fresh_error is not None:
            entry["error"] = fresh_error
            results.append(entry)
            continue
        if not _profit_take_owner_allowed(action_owner):
            entry["error"] = (
                "NON_TRADER_PROFIT_PARTIAL_CLOSE_FORBIDDEN: the action owner is not "
                "explicitly trader-owned"
            )
            results.append(entry)
            continue
        if str(action.trade_id) in forbidden_reasons:
            entry["error"] = forbidden_reasons[str(action.trade_id)]
            results.append(entry)
            continue
        try:
            entry["broker_post_attempted"] = True
            response = _close_trade_with_supported_provenance(
                broker_client,
                action.trade_id,
                str(action.close_units),
                provenance=PROFIT_PARTIAL_CLOSE_PROVENANCE,
            )
            entry["sent"] = True
            entry["response"] = response
        except Exception as exc:  # noqa: BLE001
            entry["error"] = str(exc)
        results.append(entry)
    return results


def _fresh_profit_partial_close_send_boundary(
    *,
    broker_client: Any,
    action: ProfitPartialCloseAction,
) -> tuple[str, dict[str, Any], str | None]:
    """Re-prove ownership, units, and executable profit before one close POST."""

    evidence: dict[str, Any] = {
        "trade_id": str(action.trade_id or "").strip(),
        "pair": str(action.pair or "").strip(),
        "owner": "unknown",
        "position_units": None,
        "entry_price": None,
        "unrealized_pl_jpy": None,
        "quote_bid": None,
        "quote_ask": None,
        "executable_price": None,
        "executable_profit_pips": None,
        "fresh_milestone": None,
        "remaining_units_after_close": None,
        "snapshot_fetched_at_utc": None,
    }

    snapshot_fn = getattr(broker_client, "snapshot", None)
    if not callable(snapshot_fn):
        return (
            "unknown",
            evidence,
            "FRESH_BROKER_POSITION_TRUTH_REQUIRED: profit partial close requires a "
            "fresh broker snapshot immediately before each send",
        )
    pair = evidence["pair"]
    trade_id = evidence["trade_id"]
    if not pair or not trade_id:
        return (
            "unknown",
            evidence,
            "FRESH_BROKER_POSITION_IDENTITY_REQUIRED: profit partial-close pair and "
            "trade id must be non-empty",
        )
    try:
        snapshot = snapshot_fn((pair,))
        positions = tuple(getattr(snapshot, "positions"))
    except Exception as exc:  # noqa: BLE001
        return (
            "unknown",
            evidence,
            "FRESH_BROKER_POSITION_TRUTH_REQUIRED: profit partial-close broker refresh "
            f"failed: {exc}",
        )
    fetched_at = getattr(snapshot, "fetched_at_utc", None)
    if fetched_at is not None:
        evidence["snapshot_fetched_at_utc"] = (
            fetched_at.isoformat() if hasattr(fetched_at, "isoformat") else str(fetched_at)
        )
    matches = [
        position
        for position in positions
        if str(getattr(position, "trade_id", "") or "").strip() == trade_id
    ]
    if not matches:
        return (
            "unknown",
            evidence,
            "FRESH_BROKER_POSITION_NOT_FOUND: profit partial-close target is no longer open",
        )
    if len(matches) != 1:
        return (
            "unknown",
            evidence,
            "FRESH_BROKER_POSITION_AMBIGUOUS: profit partial-close trade id is not unique",
        )

    position = matches[0]
    fresh_owner = _normalized_owner(getattr(position, "owner", None))
    evidence["owner"] = fresh_owner
    fresh_units_raw = getattr(position, "units", None)
    fresh_units = _strict_finite_number(fresh_units_raw)
    fresh_entry = _strict_finite_number(getattr(position, "entry_price", None))
    fresh_unrealized = _strict_finite_number(
        getattr(position, "unrealized_pl_jpy", None)
    )
    evidence["position_units"] = (
        fresh_units_raw if _is_exact_int(fresh_units_raw) else fresh_units
    )
    evidence["entry_price"] = fresh_entry
    evidence["unrealized_pl_jpy"] = fresh_unrealized
    if fresh_owner != "trader":
        return (
            fresh_owner,
            evidence,
            "NON_TRADER_PROFIT_PARTIAL_CLOSE_FORBIDDEN: fresh broker truth identifies "
            f"owner={fresh_owner}; only exact owner=trader may be reduced",
        )
    fresh_pair = str(getattr(position, "pair", "") or "").strip()
    fresh_side_value = getattr(position, "side", None)
    fresh_side = str(getattr(fresh_side_value, "value", fresh_side_value) or "").upper()
    if fresh_pair != pair or fresh_side != str(action.side or "").upper():
        return (
            fresh_owner,
            evidence,
            "FRESH_BROKER_POSITION_IDENTITY_MISMATCH: pair or side changed before "
            "profit partial-close send",
        )
    if predictive_scout_broker_raw_claimed(getattr(position, "raw", None)):
        return (
            fresh_owner,
            evidence,
            "PREDICTIVE_SCOUT_EXIT_GEOMETRY_FROZEN: fresh broker truth identifies an "
            "exact TP/SL forward vehicle",
        )

    if not _is_exact_int(fresh_units_raw) or int(fresh_units or 0) <= 0:
        return (
            fresh_owner,
            evidence,
            "FRESH_BROKER_POSITION_UNITS_INVALID: broker position units must be a "
            "positive integer",
        )
    if any(
        not _is_exact_int(value)
        for value in (action.original_units, action.close_units, action.remaining_units)
    ):
        return (
            fresh_owner,
            evidence,
            "PROFIT_PARTIAL_CLOSE_ACTION_UNITS_INVALID: action units must be exact integers",
        )
    fresh_units_int = int(fresh_units)
    if fresh_units_int != action.original_units:
        return (
            fresh_owner,
            evidence,
            "FRESH_BROKER_POSITION_UNITS_CHANGED: position units changed after the "
            "profit partial-close action was calculated",
        )
    if not 0 < action.close_units < fresh_units_int:
        return (
            fresh_owner,
            evidence,
            "PROFIT_PARTIAL_CLOSE_MUST_LEAVE_RUNNER: close units must be strictly below "
            "fresh broker position units",
        )
    fresh_remaining_units = fresh_units_int - action.close_units
    evidence["remaining_units_after_close"] = fresh_remaining_units
    if action.remaining_units != fresh_remaining_units:
        return (
            fresh_owner,
            evidence,
            "PROFIT_PARTIAL_CLOSE_ACTION_UNITS_MISMATCH: action remaining units do not "
            "match fresh broker truth",
        )

    quotes = getattr(snapshot, "quotes", None)
    quote = quotes.get(pair) if isinstance(quotes, dict) else None
    if quote is None:
        return (
            fresh_owner,
            evidence,
            "FRESH_EXECUTABLE_QUOTE_REQUIRED: profit partial close requires a fresh "
            "pair quote from the same broker snapshot",
        )
    quote_bid = _strict_finite_number(
        quote.get("bid") if isinstance(quote, dict) else getattr(quote, "bid", None)
    )
    quote_ask = _strict_finite_number(
        quote.get("ask") if isinstance(quote, dict) else getattr(quote, "ask", None)
    )
    evidence["quote_bid"] = quote_bid
    evidence["quote_ask"] = quote_ask
    if (
        quote_bid is None
        or quote_ask is None
        or quote_bid <= 0
        or quote_ask <= 0
        or quote_bid > quote_ask
    ):
        return (
            fresh_owner,
            evidence,
            "FRESH_EXECUTABLE_QUOTE_INVALID: broker bid/ask must be finite, positive, "
            "and ordered",
        )
    if fresh_entry is None or fresh_entry <= 0:
        return (
            fresh_owner,
            evidence,
            "FRESH_BROKER_POSITION_ENTRY_INVALID: entry price must be finite and positive",
        )
    executable_price = quote_bid if fresh_side == "LONG" else quote_ask
    pip_factor = _pip_factor(pair)
    fresh_profit_pips = (
        (executable_price - fresh_entry) * pip_factor
        if fresh_side == "LONG"
        else (fresh_entry - executable_price) * pip_factor
    )
    evidence["executable_price"] = executable_price
    evidence["executable_profit_pips"] = fresh_profit_pips
    if (
        fresh_unrealized is None
        or fresh_unrealized <= 0
        or not math.isfinite(fresh_profit_pips)
        or fresh_profit_pips <= 0
    ):
        return (
            fresh_owner,
            evidence,
            "FRESH_PROFIT_PARTIAL_CLOSE_NOT_PROFITABLE: both broker unrealized P/L and "
            "executable quote profit must remain strictly positive",
        )

    action_atr = _strict_finite_number(action.atr_pips)
    action_trigger = _strict_finite_number(action.trigger_mult)
    if (
        action_atr is None
        or action_trigger is None
        or action_atr <= 0
        or action_trigger <= 0
        or not _is_exact_int(action.milestone)
        or not _is_exact_int(action.prior_milestone)
        or action.prior_milestone < 0
        or action.milestone <= action.prior_milestone
    ):
        return (
            fresh_owner,
            evidence,
            "PROFIT_PARTIAL_CLOSE_ACTION_MILESTONE_INVALID: action ATR, trigger, and "
            "milestones must be internally valid",
        )
    threshold_pips = action_atr * action_trigger
    fresh_milestone = math.floor((fresh_profit_pips / threshold_pips) + 1e-9)
    evidence["fresh_milestone"] = fresh_milestone
    if fresh_milestone < action.milestone:
        return (
            fresh_owner,
            evidence,
            "FRESH_PROFIT_MILESTONE_NO_LONGER_REACHED: executable profit fell below the "
            "milestone that created the partial-close action",
        )
    return fresh_owner, evidence, None


def _strict_finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _is_exact_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _close_trade_with_supported_provenance(
    broker_client: Any,
    trade_id: str,
    units: str,
    *,
    provenance: str,
) -> dict[str, Any]:
    close_with_provenance = getattr(broker_client, "close_trade_with_provenance", None)
    class_method = getattr(type(broker_client), "close_trade_with_provenance", None)
    if callable(close_with_provenance) and callable(class_method):
        return close_with_provenance(trade_id, units, provenance=provenance)
    raise RuntimeError(
        "profit partial close requires close_trade_with_provenance; "
        "raw close_trade fallback is disabled by the position execution contract"
    )


def load_profit_partial_state(path: Path = DEFAULT_PROFIT_PARTIAL_CLOSE_STATE) -> dict[str, Any]:
    if not path.exists():
        return {"trade_milestones": {}}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"trade_milestones": {}}
    milestones = payload.get("trade_milestones") if isinstance(payload, dict) else {}
    return {"trade_milestones": dict(milestones) if isinstance(milestones, dict) else {}}


def save_profit_partial_state_from_results(
    results: Iterable[dict[str, Any]],
    *,
    path: Path = DEFAULT_PROFIT_PARTIAL_CLOSE_STATE,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = state or {"trade_milestones": {}}
    milestones = state.get("trade_milestones") if isinstance(state.get("trade_milestones"), dict) else {}
    changed = False
    for result in results:
        if not result.get("sent"):
            continue
        trade_id = str(result.get("trade_id") or "")
        if not trade_id:
            continue
        milestone = int(result.get("milestone") or 0)
        if milestone > int(milestones.get(trade_id, 0) or 0):
            milestones[trade_id] = milestone
            changed = True
    state["trade_milestones"] = milestones
    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return state
