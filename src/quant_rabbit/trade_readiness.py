from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

from quant_rabbit.instruments import NORMAL_SPREAD_PIPS, instrument_pip_factor
from quant_rabbit.inventory_controller import broker_position_identity
from quant_rabbit.models import BrokerSnapshot, Owner, Side
from quant_rabbit.risk import RiskPolicy


READINESS_CONTRACT = "QR_TRADE_READINESS_V1"
HARD_MAX_MARGIN_CLOSEOUT_PERCENT = 0.85
THROTTLED_PROMOTE_CURRENT_MCP = 0.82
THROTTLED_PROMOTE_STRESS_MCP = 0.87
THROTTLED_RETAIN_CURRENT_MCP = 0.85
THROTTLED_RETAIN_STRESS_MCP = 0.90
FULL_PROMOTE_CURRENT_MCP = 0.70
FULL_PROMOTE_STRESS_MCP = 0.75
FACTOR_BUDGET_NAV_MULTIPLE = 3.0


class RuntimeMode(str, Enum):
    FULL_LIVE = "FULL_LIVE"
    THROTTLED_LIVE = "THROTTLED_LIVE"
    SHADOW_ONLY = "SHADOW_ONLY"
    FREEZE_NEW = "FREEZE_NEW"
    DRAINING = "DRAINING"


@dataclass(frozen=True, slots=True)
class ExplicitRiskLimits:
    max_loss_per_order_jpy: float | None = None
    stop_drawdown_jpy: float | None = None
    minimum_margin_buffer_jpy: float | None = None

    @property
    def complete(self) -> bool:
        return all(
            _positive(value)
            for value in (
                self.max_loss_per_order_jpy,
                self.stop_drawdown_jpy,
                self.minimum_margin_buffer_jpy,
            )
        )


@dataclass(frozen=True, slots=True)
class SignalSizingInput:
    requested_units: int
    broker_minimum_units: int
    margin_jpy_per_unit: float
    closeout_margin_jpy_per_unit: float
    stress_closeout_margin_jpy_per_unit: float
    factor_delta_jpy_per_unit: Mapping[str, float]


def size_signal_for_runtime_mode(
    *,
    previous_mode: RuntimeMode,
    inventory_state: str,
    has_bot_inventory: bool,
    nav_jpy: float,
    margin_available_jpy: float,
    current_mcp: float,
    factor_exposure_jpy: Mapping[str, float],
    limits: ExplicitRiskLimits,
    software_ready: bool,
    signal: SignalSizingInput,
) -> dict[str, Any]:
    """Calculate a bounded lot and one mode receipt without broker mutation."""

    if inventory_state == RuntimeMode.DRAINING.value:
        return _mode_receipt(RuntimeMode.DRAINING, 0, signal, "INVENTORY_DRAINING")
    if inventory_state == RuntimeMode.FREEZE_NEW.value:
        return _mode_receipt(RuntimeMode.FREEZE_NEW, 0, signal, "INVENTORY_FREEZE_NEW")
    if not software_ready or not limits.complete:
        return _mode_receipt(
            RuntimeMode.SHADOW_ONLY, 0, signal, "SOFTWARE_OR_EXPLICIT_LIMITS_INCOMPLETE"
        )
    if any(
        not _positive(value)
        for value in (
            nav_jpy,
            signal.margin_jpy_per_unit,
            signal.closeout_margin_jpy_per_unit,
            signal.stress_closeout_margin_jpy_per_unit,
        )
    ):
        return _mode_receipt(RuntimeMode.SHADOW_ONLY, 0, signal, "MARGIN_MODEL_INVALID")

    requested = max(0, int(signal.requested_units))
    minimum = max(1, int(signal.broker_minimum_units))
    buffer_capacity = max(
        0,
        math.floor(
            (margin_available_jpy - float(limits.minimum_margin_buffer_jpy or 0.0))
            / signal.margin_jpy_per_unit
        ),
    )
    current_mcp_capacity = max(
        0,
        math.floor(
            (THROTTLED_RETAIN_CURRENT_MCP - current_mcp)
            * nav_jpy
            / signal.closeout_margin_jpy_per_unit
        ),
    )
    stress_mcp_capacity = max(
        0,
        math.floor(
            (THROTTLED_RETAIN_STRESS_MCP - current_mcp)
            * nav_jpy
            / signal.stress_closeout_margin_jpy_per_unit
        ),
    )
    factor_budget = nav_jpy * FACTOR_BUDGET_NAV_MULTIPLE
    factor_capacities = [requested]
    for currency, delta_per_unit in signal.factor_delta_jpy_per_unit.items():
        delta = float(delta_per_unit)
        current = float(factor_exposure_jpy.get(currency, 0.0))
        if not math.isfinite(delta) or not math.isfinite(current):
            return _mode_receipt(RuntimeMode.SHADOW_ONLY, 0, signal, "FACTOR_MODEL_INVALID")
        if delta == 0.0 or current * delta < 0.0:
            continue
        remaining = factor_budget - abs(current)
        factor_capacities.append(max(0, math.floor(remaining / abs(delta))))
    units = min(
        [requested, buffer_capacity, current_mcp_capacity, stress_mcp_capacity, *factor_capacities]
    )
    post_current_mcp = current_mcp + units * signal.closeout_margin_jpy_per_unit / nav_jpy
    post_stress_mcp = current_mcp + units * signal.stress_closeout_margin_jpy_per_unit / nav_jpy

    if units < minimum:
        mode = RuntimeMode.FREEZE_NEW if has_bot_inventory else RuntimeMode.SHADOW_ONLY
        return _mode_receipt(
            mode,
            0,
            signal,
            "CALCULATED_LOT_BELOW_BROKER_MINIMUM",
            post_current_mcp=post_current_mcp,
            post_stress_mcp=post_stress_mcp,
        )
    if post_current_mcp > THROTTLED_RETAIN_CURRENT_MCP or post_stress_mcp > THROTTLED_RETAIN_STRESS_MCP:
        mode = RuntimeMode.FREEZE_NEW if has_bot_inventory else RuntimeMode.SHADOW_ONLY
        return _mode_receipt(
            mode,
            0,
            signal,
            "POST_ENTRY_MARGIN_GATE_FAILED",
            post_current_mcp=post_current_mcp,
            post_stress_mcp=post_stress_mcp,
        )

    if (
        post_current_mcp <= FULL_PROMOTE_CURRENT_MCP
        and post_stress_mcp <= FULL_PROMOTE_STRESS_MCP
        and margin_available_jpy >= 2.0 * float(limits.minimum_margin_buffer_jpy or 0.0)
    ):
        mode = RuntimeMode.FULL_LIVE
    else:
        promote_current = (
            THROTTLED_RETAIN_CURRENT_MCP
            if previous_mode in {RuntimeMode.THROTTLED_LIVE, RuntimeMode.FULL_LIVE}
            else THROTTLED_PROMOTE_CURRENT_MCP
        )
        promote_stress = (
            THROTTLED_RETAIN_STRESS_MCP
            if previous_mode in {RuntimeMode.THROTTLED_LIVE, RuntimeMode.FULL_LIVE}
            else THROTTLED_PROMOTE_STRESS_MCP
        )
        mode = (
            RuntimeMode.THROTTLED_LIVE
            if post_current_mcp <= promote_current and post_stress_mcp <= promote_stress
            else RuntimeMode.SHADOW_ONLY
        )
    if mode is RuntimeMode.SHADOW_ONLY:
        units = 0
    return _mode_receipt(
        mode,
        units,
        signal,
        "PRE_FIXED_MODE_THRESHOLDS",
        post_current_mcp=post_current_mcp,
        post_stress_mcp=post_stress_mcp,
    )


def _mode_receipt(
    mode: RuntimeMode,
    units: int,
    signal: SignalSizingInput,
    reason: str,
    *,
    post_current_mcp: float | None = None,
    post_stress_mcp: float | None = None,
) -> dict[str, Any]:
    return {
        "mode": mode.value,
        "transition_reason": reason,
        "requested_units": signal.requested_units,
        "calculated_units": units,
        "broker_minimum_units": signal.broker_minimum_units,
        "post_entry_current_mcp": post_current_mcp,
        "post_entry_stress_mcp": post_stress_mcp,
        "mutation_allowed": mode in {RuntimeMode.FULL_LIVE, RuntimeMode.THROTTLED_LIVE}
        and units >= signal.broker_minimum_units,
    }


def screen_trade_readiness(
    *,
    snapshot: BrokerSnapshot,
    raw_account: Mapping[str, Any],
    limits: ExplicitRiskLimits,
    software_ready: bool,
    now_utc: datetime,
) -> dict[str, Any]:
    now = _aware_utc(now_utc)
    account = raw_account.get("account") if isinstance(raw_account.get("account"), Mapping) else raw_account
    margin_available = _number(account.get("marginAvailable"))
    margin_used = _number(account.get("marginUsed"))
    nav = _number(account.get("NAV"))
    mcp = _number(account.get("marginCloseoutPercent"))

    system_positions = [
        position for position in snapshot.positions if broker_position_identity(position) is not None
    ]
    no_touch_positions = [
        position for position in snapshot.positions if broker_position_identity(position) is None
    ]
    pending_entries = [
        order
        for order in snapshot.orders
        if not order.trade_id
        and str(order.order_type or "").upper()
        in {"LIMIT", "STOP", "MARKET_IF_TOUCHED", "STOP-ENTRY"}
    ]
    quotes: dict[str, dict[str, Any]] = {}
    quote_blockers: list[str] = []
    for pair in ("EUR_USD", "USD_JPY"):
        quote = snapshot.quotes.get(pair)
        if quote is None:
            quotes[pair] = {"available": False}
            quote_blockers.append(f"{pair}_QUOTE_MISSING")
            continue
        age = (now - quote.timestamp_utc.astimezone(timezone.utc)).total_seconds()
        spread = (quote.ask - quote.bid) * instrument_pip_factor(pair)
        baseline = float(NORMAL_SPREAD_PIPS[pair])
        fresh = 0.0 <= age <= RiskPolicy().max_quote_age_seconds
        spread_ok = spread <= baseline * RiskPolicy().max_spread_multiple
        if not fresh:
            quote_blockers.append(f"{pair}_QUOTE_STALE")
        if not spread_ok:
            quote_blockers.append(f"{pair}_SPREAD_ANOMALY")
        quotes[pair] = {
            "available": True,
            "bid": quote.bid,
            "ask": quote.ask,
            "quote_timestamp_utc": quote.timestamp_utc.isoformat(),
            "quote_age_seconds": round(age, 6),
            "fresh": fresh,
            "spread_pips": round(spread, 6),
            "calibrated_baseline_pips": baseline,
            "max_allowed_spread_pips": round(
                baseline * RiskPolicy().max_spread_multiple, 6
            ),
            "spread_ok": spread_ok,
        }

    blockers: list[str] = []
    if not software_ready:
        blockers.append("SOFTWARE_READINESS_NOT_SEALED")
    if not limits.complete:
        blockers.append("EXPLICIT_THREE_RISK_LIMITS_NOT_FIXED")
    if system_positions:
        blockers.append("BOT_INVENTORY_NOT_FLAT")
    if pending_entries:
        blockers.append("PENDING_ENTRY_ORDER_PRESENT")
    if mcp is None or mcp >= HARD_MAX_MARGIN_CLOSEOUT_PERCENT:
        blockers.append("MARGIN_CLOSEOUT_PERCENT_ABOVE_HARD_CAP")
    if (
        limits.minimum_margin_buffer_jpy is not None
        and (margin_available is None or margin_available < limits.minimum_margin_buffer_jpy)
    ):
        blockers.append("MINIMUM_MARGIN_BUFFER_NOT_MET")
    factor_exposure_jpy = _currency_factor_jpy(snapshot)
    factor_budget_jpy = (nav or 0.0) * FACTOR_BUDGET_NAV_MULTIPLE
    if factor_budget_jpy <= 0.0 or any(
        abs(value) > factor_budget_jpy for value in factor_exposure_jpy.values()
    ):
        blockers.append("CURRENCY_FACTOR_CONCENTRATION_ABOVE_BUDGET")
    blockers.extend(quote_blockers)

    if not software_ready:
        status = "software_unready"
    elif not limits.complete:
        status = "ready_waiting_for_risk_limits"
    elif blockers:
        status = "ready_waiting_for_margin"
    else:
        status = "ready_for_final_screen"
    lifecycle = "waiting_external_state" if status.startswith("ready_waiting") else status

    return {
        "contract": READINESS_CONTRACT,
        "evaluated_at_utc": now.isoformat(),
        "status": status,
        "lifecycle": lifecycle,
        "orders_sent": 0,
        "broker_write_performed": False,
        "manual_tagless_policy": "NO_TOUCH",
        "existing_tp_sl_policy": "NO_TOUCH",
        "software_ready": software_ready,
        "risk_limits": {
            "complete": limits.complete,
            "max_loss_per_order_jpy": limits.max_loss_per_order_jpy,
            "stop_drawdown_jpy": limits.stop_drawdown_jpy,
            "minimum_margin_buffer_jpy": limits.minimum_margin_buffer_jpy,
        },
        "account": {
            "nav_jpy": nav,
            "margin_used_jpy": margin_used,
            "margin_available_jpy": margin_available,
            "margin_closeout_percent": mcp,
            "hard_max_margin_closeout_percent": HARD_MAX_MARGIN_CLOSEOUT_PERCENT,
            "open_position_count": len(snapshot.positions),
            "no_touch_position_count": len(no_touch_positions),
            "system_owned_position_count": len(system_positions),
            "pending_entry_count": len(pending_entries),
            "attached_tp_count": sum(
                1 for position in snapshot.positions if position.take_profit is not None
            ),
        },
        "currency_exposure": _currency_exposure(snapshot),
        "currency_factor_jpy": factor_exposure_jpy,
        "currency_factor_budget_jpy": factor_budget_jpy,
        "observations": (
            ["EXISTING_NO_TOUCH_POSITIONS"] if no_touch_positions else []
        ),
        "quotes": quotes,
        "blockers": sorted(set(blockers)),
    }


def _currency_exposure(snapshot: BrokerSnapshot) -> dict[str, float]:
    exposure: dict[str, float] = {}
    for position in snapshot.positions:
        if "_" not in position.pair:
            continue
        base, quote_currency = position.pair.split("_", 1)
        signed_base = position.units * (1 if position.side is Side.LONG else -1)
        quote = snapshot.quotes.get(position.pair)
        reference = quote.mid if quote is not None else position.entry_price
        exposure[base] = exposure.get(base, 0.0) + signed_base
        exposure[quote_currency] = exposure.get(quote_currency, 0.0) - signed_base * reference
    return {key: round(value, 6) for key, value in sorted(exposure.items())}


def _currency_factor_jpy(snapshot: BrokerSnapshot) -> dict[str, float]:
    raw = _currency_exposure(snapshot)
    usd_jpy = snapshot.quotes.get("USD_JPY")
    eur_usd = snapshot.quotes.get("EUR_USD")
    usd_to_jpy = usd_jpy.mid if usd_jpy is not None else None
    rates = {"JPY": 1.0}
    if usd_to_jpy is not None:
        rates["USD"] = usd_to_jpy
    if usd_to_jpy is not None and eur_usd is not None:
        rates["EUR"] = eur_usd.mid * usd_to_jpy
    return {
        currency: round(value * rates[currency], 6)
        for currency, value in sorted(raw.items())
        if currency in rates
    }


def _positive(value: object) -> bool:
    parsed = _number(value)
    return parsed is not None and parsed > 0.0


def _number(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("now_utc must be timezone-aware")
    return value.astimezone(timezone.utc)
