"""Profit-side partial closes for runner management.

This is deliberately separate from ``adverse_partial_close``. The adverse
module was disabled because it realized losses from a lagging P/L threshold.
This module only acts on trader-owned or operator-managed manual/tagless
positions that are already in profit: bank part of the move at ATR-derived
milestones, keep the remainder running, and remember the last milestone per
trade so the same profit band is not closed repeatedly. External positions
remain out of scope.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from quant_rabbit.paths import DEFAULT_PROFIT_PARTIAL_CLOSE_STATE


# Minimums are broker/execution-shape constants, not market predictions.
# OANDA accepts integer FX units, but closing tiny sub-1000u remainders would
# leave spread-dominated dust. Keep one micro-lot on the runner and round close
# requests to 100u so fills stay operationally meaningful.
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


def _profit_take_owner_allowed(owner: str) -> bool:
    return owner.strip().lower() in {"trader", "manual", "unknown"}


def compute_profit_partial_close(
    *,
    trade_id: str,
    pair: str,
    side: str,
    units: int,
    entry_price: float,
    current_price: float,
    atr_pips: float,
    owner: str = "trader",
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
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for action in actions:
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
            "error": None,
            "response": None,
            "provenance": PROFIT_PARTIAL_CLOSE_PROVENANCE,
        }
        if not send:
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
        try:
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
    return broker_client.close_trade(trade_id, units=units)


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
