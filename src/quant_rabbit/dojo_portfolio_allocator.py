"""Deterministic, read-only capital allocation for same-epoch DOJO intents.

The virtual broker remains the fill authority.  This module sits before it and
answers a narrower question: when several workers observe an entry at the same
market epoch, which single intent has the best bounded use of the shared
account?  It deliberately records every candidate, including candidates that
are skipped, so a later replay can resolve counterfactual opportunity cost.

Version 4 is intentionally conservative:

* open positions and every pending order reserve gross margin and gross
  stop-loss risk;
* pending orders receive no OCO/netting discount;
* version 4 accepts only JPY-quoted pairs and fixes their conversion at 1.0;
  non-JPY quotes stay fail-closed until an independently verified market-data
  conversion receipt exists outside the caller-controlled allocation payload;
* at most one incumbent position may be reduced for one new intent per epoch;
* the result is a preflight diagnostic, not fill-time risk enforcement, and has
  no broker or live-order authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from quant_rabbit.currency_exposure_guard import (
    evaluate_currency_exposure,
    net_currency_exposure,
)

ALLOCATION_CONTRACT = "QR_DOJO_PORTFOLIO_ALLOCATION_V4"
OPPORTUNITY_SCORE_CONTRACT = "QR_DOJO_CANDIDATE_SELECTION_OPPORTUNITY_DIAGNOSTIC_V1"
REDUCTION_STEPS = (0.25, 0.5, 0.75, 1.0)


class DojoPortfolioAllocatorError(ValueError):
    """Raised when an allocation input or evidence receipt is malformed."""


def _canonical_sha(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DojoPortfolioAllocatorError(
            "allocation evidence is not canonical JSON"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _number(
    value: Any,
    *,
    field: str,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DojoPortfolioAllocatorError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise DojoPortfolioAllocatorError(f"{field} must be finite")
    if positive and number <= 0:
        raise DojoPortfolioAllocatorError(f"{field} must be positive")
    if non_negative and number < 0:
        raise DojoPortfolioAllocatorError(f"{field} must be non-negative")
    return number


def _integer(value: Any, *, field: str, non_negative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DojoPortfolioAllocatorError(f"{field} must be an integer")
    if non_negative and value < 0:
        raise DojoPortfolioAllocatorError(f"{field} must be non-negative")
    return value


def _identity(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise DojoPortfolioAllocatorError(f"{field} must contain 1..128 characters")
    if any(ord(character) < 33 or ord(character) > 126 for character in value):
        raise DojoPortfolioAllocatorError(f"{field} must be visible ASCII")
    return value


def _pair(value: Any, *, field: str) -> str:
    pair = _identity(value, field=field)
    parts = pair.split("_")
    if len(parts) != 2 or any(
        len(part) != 3 or not part.isalpha() or not part.isupper() for part in parts
    ):
        raise DojoPortfolioAllocatorError(f"{field} is invalid")
    if parts[0] == parts[1]:
        raise DojoPortfolioAllocatorError(f"{field} is degenerate")
    return pair


def _side(value: Any, *, field: str) -> str:
    if value not in {"LONG", "SHORT"}:
        raise DojoPortfolioAllocatorError(f"{field} must be LONG or SHORT")
    return str(value)


def _nullable_price(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    return _number(value, field=field, positive=True)


def _quote_timestamp_at_epoch(value: Any, *, field: str, decision_epoch: int) -> str:
    if not isinstance(value, str) or not value:
        raise DojoPortfolioAllocatorError(f"{field} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.split("#", 1)[0])
    except ValueError as exc:
        raise DojoPortfolioAllocatorError(
            f"{field} must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DojoPortfolioAllocatorError(f"{field} must include an explicit offset")
    epoch = parsed.timestamp()
    if (
        not math.isfinite(epoch)
        or not epoch.is_integer()
        or int(epoch) != decision_epoch
    ):
        raise DojoPortfolioAllocatorError(f"{field} must exactly match decision_epoch")
    return value


def _derived_notional_jpy(
    *, units: float, price: float, jpy_per_quote_unit: float
) -> float:
    notional = units * price * jpy_per_quote_unit
    if not math.isfinite(notional) or notional <= 0:
        raise DojoPortfolioAllocatorError("derived notional_jpy is invalid")
    return notional


def _require_keys(value: Any, *, field: str, expected: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise DojoPortfolioAllocatorError(f"{field} schema mismatch")
    return value


def _validated_conversion(
    *,
    pair: str,
    jpy_per_quote_unit: Any,
    conversion_snapshot_id: Any,
    conversion_snapshot_sha256: Any,
    field: str,
) -> tuple[float, str | None, str | None]:
    rate = _number(
        jpy_per_quote_unit,
        field=f"{field}.jpy_per_quote_unit",
        positive=True,
    )
    quote_currency = pair.split("_")[1]
    if quote_currency != "JPY":
        raise DojoPortfolioAllocatorError(
            f"{field} non-JPY quote is unsupported until an independently "
            "verified conversion receipt is implemented"
        )
    if rate != 1.0:
        raise DojoPortfolioAllocatorError(
            f"{field}.jpy_per_quote_unit must be exactly 1.0 for JPY quote"
        )
    if conversion_snapshot_id is not None or conversion_snapshot_sha256 is not None:
        raise DojoPortfolioAllocatorError(
            f"{field} must not reference a conversion snapshot for JPY quote"
        )
    return rate, None, None


def _normalize_position(
    value: Any,
    *,
    index: int,
    decision_epoch: int,
    decision_quote_timestamp: str,
    decision_quote_sequence: int,
) -> dict[str, Any]:
    field = f"open_positions[{index}]"
    row = _require_keys(
        value,
        field=field,
        expected={
            "position_id",
            "owner_id",
            "pair",
            "side",
            "units",
            "mark_price",
            "bid_price",
            "ask_price",
            "quote_timestamp",
            "quote_sequence",
            "sl_price",
            "stress_cost_pips",
            "jpy_per_quote_unit",
            "conversion_snapshot_id",
            "conversion_snapshot_sha256",
            "continuation_edge_jpy",
            "max_reduction_fraction",
        },
    )
    max_reduction = _number(
        row["max_reduction_fraction"],
        field=f"{field}.max_reduction_fraction",
        non_negative=True,
    )
    if max_reduction > 1:
        raise DojoPortfolioAllocatorError(
            f"{field}.max_reduction_fraction must be <= 1"
        )
    units = _number(row["units"], field=f"{field}.units", positive=True)
    mark_price = _number(row["mark_price"], field=f"{field}.mark_price", positive=True)
    bid_price = _number(row["bid_price"], field=f"{field}.bid_price", positive=True)
    ask_price = _number(row["ask_price"], field=f"{field}.ask_price", positive=True)
    if ask_price < bid_price or not math.isclose(
        mark_price,
        (bid_price + ask_price) / 2.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise DojoPortfolioAllocatorError(
            f"{field}.mark_price must be the bid/ask midpoint"
        )
    sl_price = _number(row["sl_price"], field=f"{field}.sl_price", positive=True)
    stress_cost_pips = _number(
        row["stress_cost_pips"],
        field=f"{field}.stress_cost_pips",
        non_negative=True,
    )
    pair = _pair(row["pair"], field=f"{field}.pair")
    side = _side(row["side"], field=f"{field}.side")
    executable_price = bid_price if side == "LONG" else ask_price
    if (side == "LONG" and sl_price >= executable_price) or (
        side == "SHORT" and sl_price <= executable_price
    ):
        raise DojoPortfolioAllocatorError(
            f"{field}.sl_price is already breached by executable price"
        )
    jpy_per_quote_unit, snapshot_id, snapshot_sha = _validated_conversion(
        pair=pair,
        jpy_per_quote_unit=row["jpy_per_quote_unit"],
        conversion_snapshot_id=row["conversion_snapshot_id"],
        conversion_snapshot_sha256=row["conversion_snapshot_sha256"],
        field=field,
    )
    pip_size = 0.01
    quote_timestamp = _quote_timestamp_at_epoch(
        row["quote_timestamp"],
        field=f"{field}.quote_timestamp",
        decision_epoch=decision_epoch,
    )
    quote_sequence = _integer(
        row["quote_sequence"], field=f"{field}.quote_sequence", non_negative=True
    )
    if quote_sequence <= 0:
        raise DojoPortfolioAllocatorError(f"{field}.quote_sequence must be positive")
    if (
        quote_timestamp != decision_quote_timestamp
        or quote_sequence != decision_quote_sequence
    ):
        raise DojoPortfolioAllocatorError(
            f"{field} quote phase/watermark does not match the allocation decision"
        )
    stop_distance = (
        executable_price - sl_price if side == "LONG" else sl_price - executable_price
    )
    stress_cost_jpy = stress_cost_pips * pip_size * units * jpy_per_quote_unit
    stop_loss_jpy = stop_distance * units * jpy_per_quote_unit + stress_cost_jpy
    return {
        "position_id": _identity(row["position_id"], field=f"{field}.position_id"),
        "owner_id": _identity(row["owner_id"], field=f"{field}.owner_id"),
        "pair": pair,
        "side": side,
        "units": units,
        "mark_price": mark_price,
        "bid_price": bid_price,
        "ask_price": ask_price,
        "quote_timestamp": quote_timestamp,
        "quote_sequence": quote_sequence,
        "sl_price": sl_price,
        "stress_cost_pips": stress_cost_pips,
        "stress_cost_jpy": stress_cost_jpy,
        "stop_loss_jpy": stop_loss_jpy,
        "jpy_per_quote_unit": jpy_per_quote_unit,
        "conversion_snapshot_id": snapshot_id,
        "conversion_snapshot_sha256": snapshot_sha,
        "notional_jpy": _derived_notional_jpy(
            units=units,
            price=mark_price,
            jpy_per_quote_unit=jpy_per_quote_unit,
        ),
        "continuation_edge_jpy": _number(
            row["continuation_edge_jpy"],
            field=f"{field}.continuation_edge_jpy",
        ),
        "max_reduction_fraction": max_reduction,
    }


def _normalize_pending(
    value: Any,
    *,
    index: int,
) -> dict[str, Any]:
    field = f"pending_orders[{index}]"
    row = _require_keys(
        value,
        field=field,
        expected={
            "order_id",
            "owner_id",
            "pair",
            "side",
            "units",
            "trigger_price",
            "sl_pips",
            "stress_cost_pips",
            "jpy_per_quote_unit",
            "conversion_snapshot_id",
            "conversion_snapshot_sha256",
        },
    )
    units = _number(row["units"], field=f"{field}.units", positive=True)
    trigger_price = _number(
        row["trigger_price"], field=f"{field}.trigger_price", positive=True
    )
    sl_pips = _number(row["sl_pips"], field=f"{field}.sl_pips", positive=True)
    stress_cost_pips = _number(
        row["stress_cost_pips"],
        field=f"{field}.stress_cost_pips",
        non_negative=True,
    )
    pair = _pair(row["pair"], field=f"{field}.pair")
    jpy_per_quote_unit, snapshot_id, snapshot_sha = _validated_conversion(
        pair=pair,
        jpy_per_quote_unit=row["jpy_per_quote_unit"],
        conversion_snapshot_id=row["conversion_snapshot_id"],
        conversion_snapshot_sha256=row["conversion_snapshot_sha256"],
        field=field,
    )
    stress_cost_jpy = stress_cost_pips * 0.01 * units * jpy_per_quote_unit
    stop_loss_jpy = sl_pips * 0.01 * units * jpy_per_quote_unit + stress_cost_jpy
    return {
        "order_id": _identity(row["order_id"], field=f"{field}.order_id"),
        "owner_id": _identity(row["owner_id"], field=f"{field}.owner_id"),
        "pair": pair,
        "side": _side(row["side"], field=f"{field}.side"),
        "units": units,
        "trigger_price": trigger_price,
        "sl_pips": sl_pips,
        "stress_cost_pips": stress_cost_pips,
        "stress_cost_jpy": stress_cost_jpy,
        "stop_loss_jpy": stop_loss_jpy,
        "jpy_per_quote_unit": jpy_per_quote_unit,
        "conversion_snapshot_id": snapshot_id,
        "conversion_snapshot_sha256": snapshot_sha,
        "notional_jpy": _derived_notional_jpy(
            units=units,
            price=trigger_price,
            jpy_per_quote_unit=jpy_per_quote_unit,
        ),
    }


def _normalize_candidate(
    value: Any,
    *,
    index: int,
    decision_epoch: int,
) -> dict[str, Any]:
    field = f"candidate_intents[{index}]"
    row = _require_keys(
        value,
        field=field,
        expected={
            "intent_id",
            "observed_epoch",
            "owner_id",
            "strategy_family",
            "pair",
            "side",
            "order_kind",
            "units",
            "entry_price",
            "jpy_per_quote_unit",
            "conversion_snapshot_id",
            "conversion_snapshot_sha256",
            "tp_price",
            "sl_price",
            "stress_cost_pips",
            "expected_net_edge_jpy",
            "expected_holding_seconds",
            "valid_until_epoch",
        },
    )
    observed_epoch = _integer(
        row["observed_epoch"], field=f"{field}.observed_epoch", non_negative=True
    )
    if observed_epoch != decision_epoch:
        raise DojoPortfolioAllocatorError(
            "every candidate must be observed at the allocation decision epoch"
        )
    valid_until = _integer(
        row["valid_until_epoch"],
        field=f"{field}.valid_until_epoch",
        non_negative=True,
    )
    if valid_until < decision_epoch:
        raise DojoPortfolioAllocatorError(
            f"{field}.valid_until_epoch precedes the decision epoch"
        )
    order_kind = row["order_kind"]
    if order_kind not in {"MARKET", "LIMIT", "STOP"}:
        raise DojoPortfolioAllocatorError(
            f"{field}.order_kind must be MARKET, LIMIT, or STOP"
        )
    side = _side(row["side"], field=f"{field}.side")
    entry = _number(row["entry_price"], field=f"{field}.entry_price", positive=True)
    units = _number(row["units"], field=f"{field}.units", positive=True)
    pair = _pair(row["pair"], field=f"{field}.pair")
    jpy_per_quote_unit, snapshot_id, snapshot_sha = _validated_conversion(
        pair=pair,
        jpy_per_quote_unit=row["jpy_per_quote_unit"],
        conversion_snapshot_id=row["conversion_snapshot_id"],
        conversion_snapshot_sha256=row["conversion_snapshot_sha256"],
        field=field,
    )
    tp = _nullable_price(row["tp_price"], field=f"{field}.tp_price")
    sl = _nullable_price(row["sl_price"], field=f"{field}.sl_price")
    stress_cost_pips = _number(
        row["stress_cost_pips"],
        field=f"{field}.stress_cost_pips",
        non_negative=True,
    )
    if tp is not None and (
        (side == "LONG" and tp <= entry) or (side == "SHORT" and tp >= entry)
    ):
        raise DojoPortfolioAllocatorError(f"{field}.tp_price is on the wrong side")
    if sl is not None and (
        (side == "LONG" and sl >= entry) or (side == "SHORT" and sl <= entry)
    ):
        raise DojoPortfolioAllocatorError(f"{field}.sl_price is on the wrong side")
    normalized = {
        "intent_id": _identity(row["intent_id"], field=f"{field}.intent_id"),
        "observed_epoch": observed_epoch,
        "owner_id": _identity(row["owner_id"], field=f"{field}.owner_id"),
        "strategy_family": _identity(
            row["strategy_family"], field=f"{field}.strategy_family"
        ),
        "pair": pair,
        "side": side,
        "order_kind": str(order_kind),
        "units": units,
        "entry_price": entry,
        "jpy_per_quote_unit": jpy_per_quote_unit,
        "conversion_snapshot_id": snapshot_id,
        "conversion_snapshot_sha256": snapshot_sha,
        "tp_price": tp,
        "sl_price": sl,
        "stress_cost_pips": stress_cost_pips,
        "stress_cost_jpy": (
            stress_cost_pips
            * (0.01 if str(row["pair"]).endswith("JPY") else 0.0001)
            * units
            * jpy_per_quote_unit
        ),
        "finite_exit_bound": sl is not None,
        "exit_bound_kind": "BROKER_SL" if sl is not None else "MISSING",
        "stop_loss_jpy": (
            units * abs(entry - sl) * jpy_per_quote_unit
            + stress_cost_pips
            * (0.01 if str(row["pair"]).endswith("JPY") else 0.0001)
            * units
            * jpy_per_quote_unit
            if sl is not None
            else None
        ),
        "notional_jpy": _derived_notional_jpy(
            units=units,
            price=entry,
            jpy_per_quote_unit=jpy_per_quote_unit,
        ),
        "expected_net_edge_jpy": _number(
            row["expected_net_edge_jpy"],
            field=f"{field}.expected_net_edge_jpy",
        ),
        "expected_holding_seconds": _integer(
            row["expected_holding_seconds"],
            field=f"{field}.expected_holding_seconds",
            non_negative=True,
        ),
        "valid_until_epoch": valid_until,
    }
    if normalized["expected_holding_seconds"] <= 0:
        raise DojoPortfolioAllocatorError(
            f"{field}.expected_holding_seconds must be positive"
        )
    return normalized


def _unique_sorted(
    rows: Sequence[dict[str, Any]], *, identity_key: str, field: str
) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda row: row[identity_key])
    identities = [row[identity_key] for row in sorted_rows]
    if len(identities) != len(set(identities)):
        raise DojoPortfolioAllocatorError(f"duplicate {field} identity")
    return sorted_rows


def _normalize_owner_concurrency_caps(
    value: Any,
) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DojoPortfolioAllocatorError("owner_concurrency_caps must be a sequence")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        field = f"owner_concurrency_caps[{index}]"
        row = _require_keys(
            raw,
            field=field,
            expected={
                "owner_id",
                "max_concurrent_per_pair",
                "global_max_concurrent",
            },
        )
        pair_cap = _integer(
            row["max_concurrent_per_pair"],
            field=f"{field}.max_concurrent_per_pair",
            non_negative=True,
        )
        global_cap = _integer(
            row["global_max_concurrent"],
            field=f"{field}.global_max_concurrent",
            non_negative=True,
        )
        if pair_cap <= 0 or global_cap <= 0:
            raise DojoPortfolioAllocatorError(
                "owner concurrency caps must be positive integers"
            )
        normalized.append(
            {
                "owner_id": _identity(row["owner_id"], field=f"{field}.owner_id"),
                "max_concurrent_per_pair": pair_cap,
                "global_max_concurrent": global_cap,
            }
        )
    return _unique_sorted(
        normalized,
        identity_key="owner_id",
        field="owner concurrency cap",
    )


def _exposure_rows(
    rows_to_convert: Sequence[Mapping[str, Any]], *, equity_jpy: float
) -> list[dict[str, Any]]:
    rows = []
    for item in rows_to_convert:
        rows.append(
            {
                "pair": item["pair"],
                "side": item["side"],
                "nav_exposure_fraction": float(item["notional_jpy"]) / equity_jpy,
            }
        )
    return rows


def _pending_currency_gross_shadow(
    pending_orders: Sequence[Mapping[str, Any]], *, equity_jpy: float
) -> dict[str, float]:
    """Worst-case per-currency pending exposure without fill/netting assumptions."""

    gross: dict[str, float] = {}
    for order in pending_orders:
        one_order = net_currency_exposure(
            _exposure_rows([order], equity_jpy=equity_jpy)
        )
        for currency, fraction in one_order.items():
            gross[currency] = gross.get(currency, 0.0) + abs(float(fraction))
    return {currency: round(value, 12) for currency, value in sorted(gross.items())}


def _worst_case_currency_shadow(
    base_exposure: Mapping[str, float], pending_gross: Mapping[str, float]
) -> dict[str, float]:
    currencies = sorted(set(base_exposure) | set(pending_gross))
    return {
        currency: round(
            abs(float(base_exposure.get(currency, 0.0)))
            + float(pending_gross.get(currency, 0.0)),
            12,
        )
        for currency in currencies
    }


def _margin_jpy(rows: Sequence[Mapping[str, Any]], *, leverage: float) -> float:
    return sum(float(row["notional_jpy"]) for row in rows) / leverage


def _gross_stop_loss_jpy(rows: Sequence[Mapping[str, Any]]) -> float:
    """Return gross bounded loss without hedge, netting, or OCO discounts."""

    return math.fsum(float(row["stop_loss_jpy"]) for row in rows)


def _candidate_gate(
    candidate: Mapping[str, Any],
    *,
    positions: Sequence[Mapping[str, Any]],
    pending_orders: Sequence[Mapping[str, Any]],
    equity_jpy: float,
    leverage: float,
    global_margin_cap_fraction: float,
    currency_cap_fraction: float,
    max_candidate_loss_fraction: float,
    max_portfolio_loss_fraction: float,
    owner_concurrency_caps: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    current_margin = _margin_jpy([*positions, *pending_orders], leverage=leverage)
    candidate_margin = float(candidate["notional_jpy"]) / leverage
    projected_margin = current_margin + candidate_margin
    margin_cap_jpy = equity_jpy * global_margin_cap_fraction
    open_stop_loss = _gross_stop_loss_jpy(positions)
    pending_stop_loss = _gross_stop_loss_jpy(pending_orders)
    reserved_stop_loss = math.fsum((open_stop_loss, pending_stop_loss))
    candidate_stop_loss = (
        float(candidate["stop_loss_jpy"])
        if candidate.get("finite_exit_bound") is True
        else None
    )
    projected_stop_loss = (
        math.fsum((reserved_stop_loss, candidate_stop_loss))
        if candidate_stop_loss is not None
        else None
    )
    portfolio_loss_cap_jpy = equity_jpy * max_portfolio_loss_fraction
    exposure_rows = _exposure_rows(positions, equity_jpy=equity_jpy)
    candidate_exposure = {
        "pair": candidate["pair"],
        "side": candidate["side"],
        "nav_exposure_fraction": float(candidate["notional_jpy"]) / equity_jpy,
    }
    try:
        exposure = evaluate_currency_exposure(
            exposure_rows,
            candidate_exposure,
            currency_cap_fraction=currency_cap_fraction,
        ).payload()
    except ValueError as exc:
        raise DojoPortfolioAllocatorError(
            "currency exposure evaluation failed"
        ) from exc
    pending_currency_gross = _pending_currency_gross_shadow(
        pending_orders, equity_jpy=equity_jpy
    )
    worst_case_currency = _worst_case_currency_shadow(
        exposure["projected_exposure"], pending_currency_gross
    )
    pending_currency_breaches = (
        sorted(
            currency
            for currency, fraction in worst_case_currency.items()
            if fraction > currency_cap_fraction + 1e-12
            and float(pending_currency_gross.get(currency, 0.0)) > 0
            and abs(float(exposure["projected_exposure"].get(currency, 0.0)))
            <= currency_cap_fraction + 1e-12
        )
        if pending_orders
        else []
    )
    owner_id = str(candidate["owner_id"])
    owner_cap = owner_concurrency_caps[owner_id]
    owner_rows = [
        row for row in [*positions, *pending_orders] if row["owner_id"] == owner_id
    ]
    active_global = len(owner_rows)
    active_pair = sum(row["pair"] == candidate["pair"] for row in owner_rows)
    reasons: list[str] = []
    if projected_margin > margin_cap_jpy + 1e-9:
        reasons.append("GLOBAL_MARGIN_CAP_EXCEEDED")
        if pending_orders:
            reasons.append("PENDING_SHADOW_MARGIN_CONSUMED")
    if exposure["admitted"] is not True:
        reasons.append("CURRENCY_EXPOSURE_CAP_EXCEEDED")
    if pending_currency_breaches:
        reasons.append("PENDING_SHADOW_CURRENCY_CAP_EXCEEDED")
    if candidate.get("finite_exit_bound") is not True:
        reasons.append("FINITE_EXIT_BOUND_MISSING")
    else:
        if (
            float(candidate["stop_loss_jpy"]) / equity_jpy
            > max_candidate_loss_fraction + 1e-12
        ):
            reasons.append("CANDIDATE_LOSS_CAP_EXCEEDED")
        if (
            projected_stop_loss is not None
            and projected_stop_loss > portfolio_loss_cap_jpy + 1e-9
        ):
            reasons.append("PORTFOLIO_STOP_LOSS_CAP_EXCEEDED")
            if pending_stop_loss > 0:
                reasons.append("PENDING_SHADOW_STOP_LOSS_CONSUMED")
    if active_pair >= int(owner_cap["max_concurrent_per_pair"]):
        reasons.append("OWNER_PAIR_CONCURRENCY_CAP_REACHED")
    if active_global >= int(owner_cap["global_max_concurrent"]):
        reasons.append("OWNER_GLOBAL_CONCURRENCY_CAP_REACHED")
    return {
        "admitted": not reasons,
        "reason_codes": reasons,
        "current_margin_jpy": round(current_margin, 8),
        "candidate_margin_jpy": round(candidate_margin, 8),
        "projected_margin_jpy": round(projected_margin, 8),
        "projected_margin_fraction": round(projected_margin / equity_jpy, 12),
        "global_margin_cap_jpy": round(margin_cap_jpy, 8),
        "current_open_stop_loss_jpy": round(open_stop_loss, 8),
        "pending_shadow_stop_loss_jpy": round(pending_stop_loss, 8),
        "reserved_stop_loss_jpy": round(reserved_stop_loss, 8),
        "candidate_stop_loss_jpy": (
            round(candidate_stop_loss, 8) if candidate_stop_loss is not None else None
        ),
        "projected_portfolio_stop_loss_jpy": (
            round(projected_stop_loss, 8) if projected_stop_loss is not None else None
        ),
        "projected_portfolio_stop_loss_fraction": (
            round(projected_stop_loss / equity_jpy, 12)
            if projected_stop_loss is not None
            else None
        ),
        "max_portfolio_stop_loss_jpy": round(portfolio_loss_cap_jpy, 8),
        "currency_exposure": exposure,
        "pending_currency_gross_shadow": pending_currency_gross,
        "worst_case_currency_exposure_fraction": worst_case_currency,
        "pending_shadow_breached_currencies": pending_currency_breaches,
        "owner_concurrency": {
            "owner_id": owner_id,
            "active_pair_positions_and_pending": active_pair,
            "max_concurrent_per_pair": owner_cap["max_concurrent_per_pair"],
            "active_global_positions_and_pending": active_global,
            "global_max_concurrent": owner_cap["global_max_concurrent"],
        },
    }


def _reduced_positions(
    positions: Sequence[Mapping[str, Any]],
    *,
    position_id: str,
    reduction_fraction: float,
) -> list[dict[str, Any]]:
    reduced: list[dict[str, Any]] = []
    for position in positions:
        row = dict(position)
        if row["position_id"] == position_id:
            row["units"] = float(row["units"]) * (1.0 - reduction_fraction)
            residual = float(row["notional_jpy"]) * (1.0 - reduction_fraction)
            if residual <= 1e-12:
                continue
            row["notional_jpy"] = residual
            row["stress_cost_jpy"] = float(row["stress_cost_jpy"]) * (
                1.0 - reduction_fraction
            )
            row["stop_loss_jpy"] = float(row["stop_loss_jpy"]) * (
                1.0 - reduction_fraction
            )
            row["continuation_edge_jpy"] = float(row["continuation_edge_jpy"]) * (
                1.0 - reduction_fraction
            )
        reduced.append(row)
    return reduced


def _candidate_efficiency(candidate: Mapping[str, Any], *, leverage: float) -> float:
    margin = float(candidate["notional_jpy"]) / leverage
    holding_hours = float(candidate["expected_holding_seconds"]) / 3600.0
    return float(candidate["expected_net_edge_jpy"]) / (margin * holding_hours)


def build_portfolio_allocation(
    *,
    decision_epoch: int,
    decision_quote_timestamp: str,
    decision_quote_sequence: int,
    equity_jpy: float,
    leverage: float,
    global_margin_cap_fraction: float,
    currency_cap_fraction: float,
    max_candidate_loss_fraction: float,
    max_portfolio_loss_fraction: float,
    owner_concurrency_caps: Sequence[Mapping[str, Any]],
    open_positions: Sequence[Mapping[str, Any]],
    pending_orders: Sequence[Mapping[str, Any]],
    candidate_intents: Sequence[Mapping[str, Any]],
    switching_cost_jpy: float = 0.0,
) -> dict[str, Any]:
    """Build one order-independent, content-addressed allocation receipt.

    The allocator selects at most one new intent.  It may keep all incumbents,
    reduce one incumbent by a declared bounded fraction, or fully rotate one
    incumbent.  The broker still rechecks every actual fill.
    """

    epoch = _integer(decision_epoch, field="decision_epoch", non_negative=True)
    decision_timestamp = _quote_timestamp_at_epoch(
        decision_quote_timestamp,
        field="decision_quote_timestamp",
        decision_epoch=epoch,
    )
    decision_sequence = _integer(
        decision_quote_sequence,
        field="decision_quote_sequence",
        non_negative=True,
    )
    if decision_sequence <= 0:
        raise DojoPortfolioAllocatorError("decision_quote_sequence must be positive")
    equity = _number(equity_jpy, field="equity_jpy", positive=True)
    leverage_value = _number(leverage, field="leverage", positive=True)
    margin_cap = _number(
        global_margin_cap_fraction,
        field="global_margin_cap_fraction",
        positive=True,
    )
    if margin_cap > 1:
        raise DojoPortfolioAllocatorError("global_margin_cap_fraction must be <= 1")
    currency_cap = _number(
        currency_cap_fraction, field="currency_cap_fraction", positive=True
    )
    candidate_loss_cap = _number(
        max_candidate_loss_fraction,
        field="max_candidate_loss_fraction",
        positive=True,
    )
    if candidate_loss_cap > 1:
        raise DojoPortfolioAllocatorError("max_candidate_loss_fraction must be <= 1")
    portfolio_loss_cap = _number(
        max_portfolio_loss_fraction,
        field="max_portfolio_loss_fraction",
        positive=True,
    )
    if portfolio_loss_cap > 1:
        raise DojoPortfolioAllocatorError("max_portfolio_loss_fraction must be <= 1")
    switching_cost = _number(
        switching_cost_jpy, field="switching_cost_jpy", non_negative=True
    )

    positions = _unique_sorted(
        [
            _normalize_position(
                value,
                index=index,
                decision_epoch=epoch,
                decision_quote_timestamp=decision_timestamp,
                decision_quote_sequence=decision_sequence,
            )
            for index, value in enumerate(open_positions)
        ],
        identity_key="position_id",
        field="position",
    )
    pending = _unique_sorted(
        [
            _normalize_pending(
                value,
                index=index,
            )
            for index, value in enumerate(pending_orders)
        ],
        identity_key="order_id",
        field="pending order",
    )
    candidates = _unique_sorted(
        [
            _normalize_candidate(
                value,
                index=index,
                decision_epoch=epoch,
            )
            for index, value in enumerate(candidate_intents)
        ],
        identity_key="intent_id",
        field="candidate intent",
    )
    normalized_owner_caps = _normalize_owner_concurrency_caps(owner_concurrency_caps)
    owner_caps_by_id = {row["owner_id"]: row for row in normalized_owner_caps}
    missing_owner_caps = sorted(
        {candidate["owner_id"] for candidate in candidates} - set(owner_caps_by_id)
    )
    if missing_owner_caps:
        raise DojoPortfolioAllocatorError(
            "candidate owner concurrency cap is missing: "
            + ",".join(missing_owner_caps)
        )

    current_exposure_rows = _exposure_rows(positions, equity_jpy=equity)
    try:
        current_position_currency_exposure = (
            net_currency_exposure(current_exposure_rows)
            if current_exposure_rows
            else {}
        )
    except ValueError as exc:
        raise DojoPortfolioAllocatorError("current currency exposure failed") from exc
    pending_currency_gross = _pending_currency_gross_shadow(pending, equity_jpy=equity)
    reserved_worst_case_currency = _worst_case_currency_shadow(
        current_position_currency_exposure, pending_currency_gross
    )
    current_position_margin = _margin_jpy(positions, leverage=leverage_value)
    pending_shadow_margin = _margin_jpy(pending, leverage=leverage_value)
    current_position_stop_loss = _gross_stop_loss_jpy(positions)
    pending_shadow_stop_loss = _gross_stop_loss_jpy(pending)
    reserved_stop_loss = math.fsum(
        (current_position_stop_loss, pending_shadow_stop_loss)
    )
    portfolio_loss_cap_jpy = equity * portfolio_loss_cap

    candidate_evaluations: dict[str, dict[str, Any]] = {}
    feasible_plans: list[dict[str, Any]] = []
    for candidate in candidates:
        intent_id = str(candidate["intent_id"])
        identity_sha = _canonical_sha(candidate)
        efficiency = _candidate_efficiency(candidate, leverage=leverage_value)
        initial_gate = _candidate_gate(
            candidate,
            positions=positions,
            pending_orders=pending,
            equity_jpy=equity,
            leverage=leverage_value,
            global_margin_cap_fraction=margin_cap,
            currency_cap_fraction=currency_cap,
            max_candidate_loss_fraction=candidate_loss_cap,
            max_portfolio_loss_fraction=portfolio_loss_cap,
            owner_concurrency_caps=owner_caps_by_id,
        )
        candidate_evaluations[intent_id] = {
            "intent": candidate,
            "intent_identity_sha256": identity_sha,
            "capital_efficiency_jpy_per_margin_hour": round(efficiency, 12),
            "initial_admission": initial_gate,
            "owner_release_eligible_position_ids": [
                position["position_id"]
                for position in positions
                if position["owner_id"] == candidate["owner_id"]
            ],
        }
        if float(candidate["expected_net_edge_jpy"]) <= 0:
            continue
        if initial_gate["admitted"] is True:
            feasible_plans.append(
                {
                    "intent_id": intent_id,
                    "position_id": None,
                    "reduction_fraction": 0.0,
                    "action": "HOLD_FULL",
                    "incremental_expected_edge_jpy": float(
                        candidate["expected_net_edge_jpy"]
                    ),
                    "lost_continuation_edge_jpy": 0.0,
                    "switching_cost_jpy": 0.0,
                    "post_release_admission": initial_gate,
                    "capital_efficiency_jpy_per_margin_hour": efficiency,
                }
            )
        for position in positions:
            if position["owner_id"] != candidate["owner_id"]:
                continue
            max_reduction = float(position["max_reduction_fraction"])
            for reduction_fraction in REDUCTION_STEPS:
                if reduction_fraction > max_reduction + 1e-12:
                    continue
                post_release_positions = _reduced_positions(
                    positions,
                    position_id=str(position["position_id"]),
                    reduction_fraction=reduction_fraction,
                )
                gate = _candidate_gate(
                    candidate,
                    positions=post_release_positions,
                    pending_orders=pending,
                    equity_jpy=equity,
                    leverage=leverage_value,
                    global_margin_cap_fraction=margin_cap,
                    currency_cap_fraction=currency_cap,
                    max_candidate_loss_fraction=candidate_loss_cap,
                    max_portfolio_loss_fraction=portfolio_loss_cap,
                    owner_concurrency_caps=owner_caps_by_id,
                )
                if gate["admitted"] is not True:
                    continue
                lost_continuation = (
                    float(position["continuation_edge_jpy"]) * reduction_fraction
                )
                exit_cost = switching_cost * reduction_fraction
                incremental_edge = (
                    float(candidate["expected_net_edge_jpy"])
                    - lost_continuation
                    - exit_cost
                )
                if incremental_edge <= 0:
                    continue
                incremental_efficiency = incremental_edge / (
                    float(candidate["notional_jpy"])
                    / leverage_value
                    * (float(candidate["expected_holding_seconds"]) / 3600.0)
                )
                feasible_plans.append(
                    {
                        "intent_id": intent_id,
                        "position_id": position["position_id"],
                        "reduction_fraction": reduction_fraction,
                        "action": (
                            "CUT_ROTATE"
                            if math.isclose(reduction_fraction, 1.0)
                            else "HOLD_REDUCE"
                        ),
                        "incremental_expected_edge_jpy": incremental_edge,
                        "lost_continuation_edge_jpy": lost_continuation,
                        "switching_cost_jpy": exit_cost,
                        "post_release_admission": gate,
                        "capital_efficiency_jpy_per_margin_hour": (
                            incremental_efficiency
                        ),
                    }
                )

    selected_plan = None
    if feasible_plans:
        selected_plan = sorted(
            feasible_plans,
            key=lambda plan: (
                -float(plan["capital_efficiency_jpy_per_margin_hour"]),
                -float(plan["incremental_expected_edge_jpy"]),
                float(plan["reduction_fraction"]),
                str(plan["intent_id"]),
                str(plan["position_id"] or ""),
            ),
        )[0]

    if selected_plan is not None:
        action = str(selected_plan["action"])
        selected_intent_id = str(selected_plan["intent_id"])
        reason_codes = {
            "HOLD_FULL": ["SELECTED_INTENT_FITS_WITHOUT_RELEASING_CAPITAL"],
            "HOLD_REDUCE": [
                "PARTIAL_RELEASE_RESTORES_CAPACITY_AND_IMPROVES_EXPECTED_EDGE"
            ],
            "CUT_ROTATE": ["FULL_RELEASE_RESTORES_CAPACITY_AND_IMPROVES_EXPECTED_EDGE"],
        }[action]
    elif (
        not candidates
        and (positions or pending)
        and reserved_stop_loss > portfolio_loss_cap_jpy + 1e-9
    ):
        action = "SKIP"
        selected_intent_id = None
        reason_codes = ["EXISTING_PORTFOLIO_LOSS_CAP_EXCEEDED"]
    elif not candidates and (positions or pending):
        action = "HOLD_FULL"
        selected_intent_id = None
        reason_codes = ["NO_CANDIDATE_INTENTS_KEEP_EXISTING_PORTFOLIO"]
    elif not candidates:
        action = "SKIP"
        selected_intent_id = None
        reason_codes = ["NO_POSITION_AND_NO_CANDIDATE_INTENTS"]
    else:
        action = "SKIP"
        selected_intent_id = None
        reason_codes = ["NO_POSITIVE_ADMISSIBLE_BOUNDED_ALLOCATION"]

    intent_log: list[dict[str, Any]] = []
    feasible_intent_ids = {str(plan["intent_id"]) for plan in feasible_plans}
    for candidate in candidates:
        intent_id = str(candidate["intent_id"])
        evaluation = candidate_evaluations[intent_id]
        disposition_reasons: list[str] = []
        if intent_id == selected_intent_id:
            disposition = "SELECTED"
            disposition_reasons.append(
                "BEST_CAPITAL_EFFICIENCY_THEN_PORTFOLIO_INCREMENTAL_EDGE"
            )
        else:
            disposition = "SKIPPED"
            if float(candidate["expected_net_edge_jpy"]) <= 0:
                disposition_reasons.append("NON_POSITIVE_EXPECTED_EDGE")
            elif intent_id in feasible_intent_ids:
                disposition_reasons.append("LOWER_PLAN_CAPITAL_EFFICIENCY_OR_TIEBREAK")
            else:
                disposition_reasons.extend(
                    evaluation["initial_admission"]["reason_codes"]
                )
                if positions and not evaluation["owner_release_eligible_position_ids"]:
                    disposition_reasons.append("CROSS_OWNER_RELEASE_FORBIDDEN")
                disposition_reasons.append("NO_ADMISSIBLE_BOUNDED_RELEASE")
        intent_log.append(
            {
                **evaluation,
                "disposition": disposition,
                "disposition_reason_codes": list(dict.fromkeys(disposition_reasons)),
                "counterfactual_outcome_required": True,
            }
        )

    normalized_plan = None
    if selected_plan is not None:
        normalized_plan = {
            key: (
                round(float(value), 12)
                if key
                in {
                    "reduction_fraction",
                    "incremental_expected_edge_jpy",
                    "lost_continuation_edge_jpy",
                    "switching_cost_jpy",
                    "capital_efficiency_jpy_per_margin_hour",
                }
                else value
            )
            for key, value in selected_plan.items()
        }

    body: dict[str, Any] = {
        "contract": ALLOCATION_CONTRACT,
        "schema_version": 4,
        "decision_epoch": epoch,
        "decision_quote_timestamp": decision_timestamp,
        "decision_quote_sequence": decision_sequence,
        "policy": {
            "leverage": leverage_value,
            "global_margin_cap_fraction": margin_cap,
            "currency_cap_fraction": currency_cap,
            "max_candidate_loss_fraction": candidate_loss_cap,
            "max_portfolio_loss_fraction": portfolio_loss_cap,
            "switching_cost_jpy": switching_cost,
            "reduction_steps": list(REDUCTION_STEPS),
            "margin_model": "GROSS_OPEN_PLUS_ALL_PENDING_SHADOW_DIVIDED_BY_LEVERAGE",
            "portfolio_loss_model": (
                "GROSS_OPEN_MARK_TO_SL_PLUS_ALL_PENDING_TRIGGER_TO_SL_PLUS_CANDIDATE_WITH_STRESS"
            ),
            "pending_oco_netting_allowed": False,
            "portfolio_loss_netting_allowed": False,
            "portfolio_loss_enforcement_scope": (
                "PREFLIGHT_DIAGNOSTIC_NOT_FILL_TIME_ENFORCEMENT"
            ),
            "open_units_semantics": "REMAINING_UNITS_AFTER_PARTIAL_CLOSE",
            "max_new_intents_per_epoch": 1,
            "max_incumbents_released_per_epoch": 1,
            "release_owner_policy": "CANDIDATE_OWNER_ONLY",
            "owner_concurrency_caps": normalized_owner_caps,
            "owner_concurrency_model": (
                "OPEN_POSITIONS_PLUS_ALL_PENDING_SHADOW_BY_OWNER"
            ),
        },
        "account": {
            "equity_jpy": equity,
            "open_position_margin_jpy": round(current_position_margin, 8),
            "pending_shadow_margin_jpy": round(pending_shadow_margin, 8),
            "reserved_margin_jpy": round(
                current_position_margin + pending_shadow_margin, 8
            ),
            "reserved_margin_fraction": round(
                (current_position_margin + pending_shadow_margin) / equity, 12
            ),
            "global_margin_cap_jpy": round(equity * margin_cap, 8),
            "open_position_stop_loss_jpy": round(current_position_stop_loss, 8),
            "pending_stop_loss_gross_shadow_jpy": round(pending_shadow_stop_loss, 8),
            "reserved_stop_loss_jpy": round(reserved_stop_loss, 8),
            "reserved_stop_loss_fraction": round(reserved_stop_loss / equity, 12),
            "max_portfolio_stop_loss_jpy": round(portfolio_loss_cap_jpy, 8),
            "existing_portfolio_loss_cap_breached": (
                reserved_stop_loss > portfolio_loss_cap_jpy + 1e-9
            ),
            "current_position_currency_exposure": current_position_currency_exposure,
            "pending_currency_gross_shadow": pending_currency_gross,
            "reserved_worst_case_currency_exposure_fraction": (
                reserved_worst_case_currency
            ),
        },
        "open_positions": positions,
        "pending_orders": pending,
        "candidate_intent_log": intent_log,
        "decision": {
            "action": action,
            "reason_codes": reason_codes,
            "selected_intent_id": selected_intent_id,
            "selected_plan": normalized_plan,
            "entry_admitted": selected_intent_id is not None,
        },
        "candidate_count": len(candidates),
        "all_candidate_intents_recorded": True,
        "order_authority": "NONE",
        "live_permission": False,
        "broker_mutation_allowed": False,
    }
    return {**body, "allocation_sha256": _canonical_sha(body)}


def score_allocation_opportunity_cost(
    allocation: Mapping[str, Any],
    resolved_outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score candidate-selection regret, not released-incumbent portfolio PnL."""

    allocation_row = _require_keys(
        allocation,
        field="allocation",
        expected={
            "contract",
            "schema_version",
            "decision_epoch",
            "decision_quote_timestamp",
            "decision_quote_sequence",
            "policy",
            "account",
            "open_positions",
            "pending_orders",
            "candidate_intent_log",
            "decision",
            "candidate_count",
            "all_candidate_intents_recorded",
            "order_authority",
            "live_permission",
            "broker_mutation_allowed",
            "allocation_sha256",
        },
    )
    allocation_body = {
        key: value
        for key, value in allocation_row.items()
        if key != "allocation_sha256"
    }
    if (
        allocation_row["contract"] != ALLOCATION_CONTRACT
        or isinstance(allocation_row["schema_version"], bool)
        or allocation_row["schema_version"] != 4
        or allocation_row["allocation_sha256"] != _canonical_sha(allocation_body)
    ):
        raise DojoPortfolioAllocatorError("allocation receipt verification failed")
    if (
        allocation_row["all_candidate_intents_recorded"] is not True
        or allocation_row["order_authority"] != "NONE"
        or allocation_row["live_permission"] is not False
        or allocation_row["broker_mutation_allowed"] is not False
    ):
        raise DojoPortfolioAllocatorError("allocation receipt safety boundary failed")
    policy = _require_keys(
        allocation_row["policy"],
        field="allocation.policy",
        expected={
            "leverage",
            "global_margin_cap_fraction",
            "currency_cap_fraction",
            "max_candidate_loss_fraction",
            "max_portfolio_loss_fraction",
            "switching_cost_jpy",
            "reduction_steps",
            "margin_model",
            "portfolio_loss_model",
            "pending_oco_netting_allowed",
            "portfolio_loss_netting_allowed",
            "portfolio_loss_enforcement_scope",
            "open_units_semantics",
            "max_new_intents_per_epoch",
            "max_incumbents_released_per_epoch",
            "release_owner_policy",
            "owner_concurrency_caps",
            "owner_concurrency_model",
        },
    )
    account = _require_keys(
        allocation_row["account"],
        field="allocation.account",
        expected={
            "equity_jpy",
            "open_position_margin_jpy",
            "pending_shadow_margin_jpy",
            "reserved_margin_jpy",
            "reserved_margin_fraction",
            "global_margin_cap_jpy",
            "open_position_stop_loss_jpy",
            "pending_stop_loss_gross_shadow_jpy",
            "reserved_stop_loss_jpy",
            "reserved_stop_loss_fraction",
            "max_portfolio_stop_loss_jpy",
            "existing_portfolio_loss_cap_breached",
            "current_position_currency_exposure",
            "pending_currency_gross_shadow",
            "reserved_worst_case_currency_exposure_fraction",
        },
    )
    open_positions = allocation_row["open_positions"]
    pending_orders = allocation_row["pending_orders"]
    raw_intent_log = allocation_row["candidate_intent_log"]
    if (
        not isinstance(open_positions, list)
        or not isinstance(pending_orders, list)
        or not isinstance(raw_intent_log, list)
    ):
        raise DojoPortfolioAllocatorError("allocation source collections are malformed")
    position_sources = [
        {
            key: value
            for key, value in row.items()
            if key not in {"notional_jpy", "stress_cost_jpy", "stop_loss_jpy"}
        }
        if isinstance(row, Mapping)
        else row
        for row in open_positions
    ]
    pending_sources = [
        {
            key: value
            for key, value in row.items()
            if key not in {"notional_jpy", "stress_cost_jpy", "stop_loss_jpy"}
        }
        if isinstance(row, Mapping)
        else row
        for row in pending_orders
    ]
    candidate_sources = []
    for row in raw_intent_log:
        if not isinstance(row, Mapping) or not isinstance(row.get("intent"), Mapping):
            raise DojoPortfolioAllocatorError(
                "allocation candidate source is malformed"
            )
        candidate_sources.append(
            {
                key: value
                for key, value in row["intent"].items()
                if key
                not in {
                    "notional_jpy",
                    "finite_exit_bound",
                    "exit_bound_kind",
                    "stress_cost_jpy",
                    "stop_loss_jpy",
                }
            }
        )
    rebuilt = build_portfolio_allocation(
        decision_epoch=allocation_row["decision_epoch"],
        decision_quote_timestamp=allocation_row["decision_quote_timestamp"],
        decision_quote_sequence=allocation_row["decision_quote_sequence"],
        equity_jpy=account["equity_jpy"],
        leverage=policy["leverage"],
        global_margin_cap_fraction=policy["global_margin_cap_fraction"],
        currency_cap_fraction=policy["currency_cap_fraction"],
        max_candidate_loss_fraction=policy["max_candidate_loss_fraction"],
        max_portfolio_loss_fraction=policy["max_portfolio_loss_fraction"],
        owner_concurrency_caps=policy["owner_concurrency_caps"],
        open_positions=position_sources,
        pending_orders=pending_sources,
        candidate_intents=candidate_sources,
        switching_cost_jpy=policy["switching_cost_jpy"],
    )
    if dict(allocation_row) != rebuilt:
        raise DojoPortfolioAllocatorError(
            "allocation receipt canonical reconstruction mismatch"
        )
    decision = _require_keys(
        allocation_row["decision"],
        field="allocation.decision",
        expected={
            "action",
            "reason_codes",
            "selected_intent_id",
            "selected_plan",
            "entry_admitted",
        },
    )
    if decision["action"] not in {"HOLD_FULL", "HOLD_REDUCE", "CUT_ROTATE", "SKIP"}:
        raise DojoPortfolioAllocatorError("allocation decision action is invalid")
    intent_log = raw_intent_log
    if not isinstance(intent_log, list):
        raise DojoPortfolioAllocatorError("allocation intent log is missing")
    if (
        isinstance(allocation_row["candidate_count"], bool)
        or not isinstance(allocation_row["candidate_count"], int)
        or allocation_row["candidate_count"] != len(intent_log)
    ):
        raise DojoPortfolioAllocatorError("allocation candidate count mismatch")
    expected: dict[str, str] = {}
    selected_rows: list[str] = []
    for index, raw_log_row in enumerate(intent_log):
        log_row = _require_keys(
            raw_log_row,
            field=f"allocation.candidate_intent_log[{index}]",
            expected={
                "intent",
                "intent_identity_sha256",
                "capital_efficiency_jpy_per_margin_hour",
                "initial_admission",
                "owner_release_eligible_position_ids",
                "disposition",
                "disposition_reason_codes",
                "counterfactual_outcome_required",
            },
        )
        intent = log_row["intent"]
        if not isinstance(intent, Mapping):
            raise DojoPortfolioAllocatorError(
                "allocation candidate intent is malformed"
            )
        normalized_input = {
            key: value
            for key, value in intent.items()
            if key
            not in {
                "notional_jpy",
                "finite_exit_bound",
                "exit_bound_kind",
                "stress_cost_jpy",
                "stop_loss_jpy",
            }
        }
        normalized_intent = _normalize_candidate(
            normalized_input,
            index=index,
            decision_epoch=_integer(
                allocation_row["decision_epoch"],
                field="allocation.decision_epoch",
                non_negative=True,
            ),
        )
        if dict(intent) != normalized_intent:
            raise DojoPortfolioAllocatorError(
                "allocation candidate normalization mismatch"
            )
        intent_id = normalized_intent["intent_id"]
        if intent_id in expected:
            raise DojoPortfolioAllocatorError("duplicate allocation candidate intent")
        identity_sha = log_row["intent_identity_sha256"]
        if identity_sha != _canonical_sha(normalized_intent):
            raise DojoPortfolioAllocatorError(
                "allocation candidate identity verification failed"
            )
        expected[intent_id] = str(identity_sha)
        if log_row["counterfactual_outcome_required"] is not True:
            raise DojoPortfolioAllocatorError(
                "allocation candidate counterfactual requirement missing"
            )
        if log_row["disposition"] == "SELECTED":
            selected_rows.append(intent_id)
        elif log_row["disposition"] != "SKIPPED":
            raise DojoPortfolioAllocatorError(
                "allocation candidate disposition is invalid"
            )
    selected_intent_id = decision["selected_intent_id"]
    if selected_intent_id is None:
        if selected_rows or decision["entry_admitted"] is not False:
            raise DojoPortfolioAllocatorError("allocation selection is inconsistent")
    elif (
        selected_intent_id not in expected
        or selected_rows != [selected_intent_id]
        or decision["entry_admitted"] is not True
    ):
        raise DojoPortfolioAllocatorError("allocation selection is inconsistent")
    outcomes: dict[str, float] = {}
    normalized_outcomes: list[dict[str, Any]] = []
    for index, raw in enumerate(resolved_outcomes):
        field = f"resolved_outcomes[{index}]"
        row = _require_keys(
            raw,
            field=field,
            expected={
                "intent_id",
                "intent_identity_sha256",
                "resolved_net_pnl_jpy",
            },
        )
        intent_id = _identity(row["intent_id"], field=f"{field}.intent_id")
        if intent_id not in expected:
            raise DojoPortfolioAllocatorError("outcome references an unknown intent")
        identity_sha = row["intent_identity_sha256"]
        if identity_sha != expected[intent_id]:
            raise DojoPortfolioAllocatorError("outcome intent identity mismatch")
        if intent_id in outcomes:
            raise DojoPortfolioAllocatorError("duplicate resolved outcome")
        pnl = _number(
            row["resolved_net_pnl_jpy"],
            field=f"{field}.resolved_net_pnl_jpy",
        )
        outcomes[intent_id] = pnl
        normalized_outcomes.append(
            {
                "intent_id": intent_id,
                "intent_identity_sha256": identity_sha,
                "resolved_net_pnl_jpy": pnl,
            }
        )
    normalized_outcomes.sort(key=lambda row: row["intent_id"])
    missing = sorted(set(expected) - set(outcomes))
    if missing:
        status = "PENDING_COUNTERFACTUAL_OUTCOMES"
        selected_pnl = None
        best_intent_id = None
        best_available_pnl = None
        opportunity_loss = None
    else:
        status = "COMPLETE"
        selected_pnl = outcomes.get(str(selected_intent_id), 0.0)
        ranked = sorted(
            (
                (intent_id, pnl)
                for intent_id, pnl in outcomes.items()
                if intent_id != selected_intent_id
            ),
            key=lambda item: (-item[1], item[0]),
        )
        if ranked and ranked[0][1] > 0:
            best_intent_id, best_available_pnl = ranked[0]
        else:
            best_intent_id, best_available_pnl = None, 0.0
        opportunity_loss = max(0.0, best_available_pnl - selected_pnl)
    body = {
        "contract": OPPORTUNITY_SCORE_CONTRACT,
        "schema_version": 1,
        "allocation_sha256": allocation_row["allocation_sha256"],
        "classification": "SELF_ATTESTED_COUNTERFACTUAL_DIAGNOSTIC",
        "score_scope": "CANDIDATE_SELECTION_ONLY",
        "incumbent_release_outcome_included": False,
        "portfolio_opportunity_cost_claim_allowed": False,
        "allocation_or_entry_admission_allowed": False,
        "proof_eligible": False,
        "promotion_eligible": False,
        "tuning_proof_eligible": False,
        "status": status,
        "selected_intent_id": selected_intent_id,
        "selected_resolved_net_pnl_jpy": selected_pnl,
        "best_counterfactual_intent_id": best_intent_id,
        "best_available_net_pnl_jpy": best_available_pnl,
        "candidate_selection_opportunity_loss_jpy": opportunity_loss,
        "resolved_outcomes": normalized_outcomes,
        "missing_intent_ids": missing,
        "all_candidate_intents_recorded": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    return {**body, "score_sha256": _canonical_sha(body)}


__all__ = [
    "ALLOCATION_CONTRACT",
    "DojoPortfolioAllocatorError",
    "OPPORTUNITY_SCORE_CONTRACT",
    "build_portfolio_allocation",
    "score_allocation_opportunity_cost",
]
