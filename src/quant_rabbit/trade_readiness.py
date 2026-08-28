from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

from quant_rabbit.instruments import NORMAL_SPREAD_PIPS, instrument_pip_factor
from quant_rabbit.inventory_controller import broker_position_identity
from quant_rabbit.models import BrokerSnapshot, Owner, Side
from quant_rabbit.risk import RiskPolicy


READINESS_CONTRACT = "QR_TRADE_READINESS_V1"
STRESS_PIPS = 25.0
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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
    max_post_entry_current_mcp: float | None = None
    max_post_entry_stress_mcp: float | None = None
    max_currency_factor_nav_multiple: float | None = None
    max_bot_positions: int | None = None
    mode_hysteresis_mcp: float | None = None
    forward_proof_sha256: str | None = None
    risk_contract_sha256: str | None = None

    @property
    def numeric_complete(self) -> bool:
        numeric_complete = all(
            _positive(value)
            for value in (
                self.max_loss_per_order_jpy,
                self.stop_drawdown_jpy,
                self.minimum_margin_buffer_jpy,
                self.max_currency_factor_nav_multiple,
                self.mode_hysteresis_mcp,
            )
        )
        current_cap = _number(self.max_post_entry_current_mcp)
        stress_cap = _number(self.max_post_entry_stress_mcp)
        hysteresis = _number(self.mode_hysteresis_mcp)
        position_cap = self.max_bot_positions
        return bool(
            numeric_complete
            and current_cap is not None
            and stress_cap is not None
            and hysteresis is not None
            and 0.0 < current_cap < stress_cap < 1.0
            and 0.0 < hysteresis < current_cap
            and isinstance(position_cap, int)
            and not isinstance(position_cap, bool)
            and position_cap > 0
        )

    @property
    def proof_sealed(self) -> bool:
        return bool(
            _SHA256_RE.fullmatch(str(self.forward_proof_sha256 or ""))
            and _SHA256_RE.fullmatch(str(self.risk_contract_sha256 or ""))
        )

    @property
    def complete(self) -> bool:
        return self.numeric_complete and self.proof_sealed


@dataclass(frozen=True, slots=True)
class SignalSizingInput:
    requested_units: int
    broker_minimum_units: int
    margin_jpy_per_unit: float
    closeout_margin_jpy_per_unit: float
    stress_closeout_margin_jpy_per_unit: float
    loss_jpy_per_unit: float
    factor_delta_jpy_per_unit: Mapping[str, float]


def size_signal_for_runtime_mode(
    *,
    previous_mode: RuntimeMode,
    inventory_state: str,
    has_bot_inventory: bool,
    nav_jpy: float,
    margin_available_jpy: float,
    current_mcp: float,
    stress_baseline_mcp: float,
    campaign_drawdown_jpy: float,
    current_bot_position_count: int,
    cooldown_elapsed: bool,
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
    if not software_ready or not limits.complete or not cooldown_elapsed:
        return _mode_receipt(
            RuntimeMode.SHADOW_ONLY,
            0,
            signal,
            "SOFTWARE_RISK_PROOF_OR_COOLDOWN_BLOCKED",
        )
    if any(
        not _positive(value)
        for value in (
            nav_jpy,
            signal.margin_jpy_per_unit,
            signal.closeout_margin_jpy_per_unit,
            signal.stress_closeout_margin_jpy_per_unit,
            signal.loss_jpy_per_unit,
        )
    ):
        return _mode_receipt(RuntimeMode.SHADOW_ONLY, 0, signal, "MARGIN_MODEL_INVALID")
    if (
        not math.isfinite(float(margin_available_jpy))
        or margin_available_jpy < 0.0
        or not math.isfinite(float(current_mcp))
        or current_mcp < 0.0
        or not math.isfinite(float(stress_baseline_mcp))
        or stress_baseline_mcp < current_mcp
        or not math.isfinite(float(campaign_drawdown_jpy))
        or campaign_drawdown_jpy < 0.0
        or isinstance(current_bot_position_count, bool)
        or not isinstance(current_bot_position_count, int)
        or current_bot_position_count < 0
    ):
        return _mode_receipt(RuntimeMode.SHADOW_ONLY, 0, signal, "STRESS_BASELINE_INVALID")
    if campaign_drawdown_jpy >= float(limits.stop_drawdown_jpy or 0.0):
        mode = RuntimeMode.FREEZE_NEW if has_bot_inventory else RuntimeMode.SHADOW_ONLY
        return _mode_receipt(mode, 0, signal, "STOP_DRAWDOWN_REACHED")
    if current_bot_position_count >= int(limits.max_bot_positions or 0):
        return _mode_receipt(RuntimeMode.SHADOW_ONLY, 0, signal, "MAX_BOT_POSITIONS_REACHED")

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
            (float(limits.max_post_entry_current_mcp or 0.0) - current_mcp)
            * nav_jpy
            / signal.closeout_margin_jpy_per_unit
        ),
    )
    stress_mcp_capacity = max(
        0,
        math.floor(
            (float(limits.max_post_entry_stress_mcp or 0.0) - stress_baseline_mcp)
            * nav_jpy
            / signal.stress_closeout_margin_jpy_per_unit
        ),
    )
    loss_capacity = max(
        0,
        math.floor(float(limits.max_loss_per_order_jpy or 0.0) / signal.loss_jpy_per_unit),
    )
    factor_budget = nav_jpy * float(limits.max_currency_factor_nav_multiple or 0.0)
    factor_capacities = [requested]
    for currency, delta_per_unit in signal.factor_delta_jpy_per_unit.items():
        delta = float(delta_per_unit)
        current = float(factor_exposure_jpy.get(currency, 0.0))
        if not math.isfinite(delta) or not math.isfinite(current):
            return _mode_receipt(RuntimeMode.SHADOW_ONLY, 0, signal, "FACTOR_MODEL_INVALID")
        if delta == 0.0:
            continue
        if delta > 0.0:
            capacity = math.floor((factor_budget - current) / delta)
        else:
            capacity = math.floor((factor_budget + current) / abs(delta))
        factor_capacities.append(max(0, capacity))
    safe_unit_capacity = min(
        [
            requested,
            buffer_capacity,
            current_mcp_capacity,
            stress_mcp_capacity,
            loss_capacity,
            *factor_capacities,
        ]
    )
    # Progressive live is deliberately a micro-lot lane.  The larger safe
    # capacity is retained in the receipt for audit, but is never treated as
    # permission to scale the live order automatically.
    units = minimum if safe_unit_capacity >= minimum else 0
    post_current_mcp = current_mcp + units * signal.closeout_margin_jpy_per_unit / nav_jpy
    post_stress_mcp = (
        stress_baseline_mcp
        + units * signal.stress_closeout_margin_jpy_per_unit / nav_jpy
    )
    post_factor_exposure = {
        currency: float(factor_exposure_jpy.get(currency, 0.0))
        + units * float(signal.factor_delta_jpy_per_unit.get(currency, 0.0))
        for currency in set(factor_exposure_jpy) | set(signal.factor_delta_jpy_per_unit)
    }
    planned_loss_jpy = units * signal.loss_jpy_per_unit
    post_margin_available_jpy = margin_available_jpy - units * signal.margin_jpy_per_unit
    post_max_factor_nav_multiple = max(
        (abs(value) / nav_jpy for value in post_factor_exposure.values()),
        default=0.0,
    )

    if safe_unit_capacity < minimum:
        mode = RuntimeMode.FREEZE_NEW if has_bot_inventory else RuntimeMode.SHADOW_ONLY
        return _mode_receipt(
            mode,
            0,
            signal,
            "CALCULATED_LOT_BELOW_BROKER_MINIMUM",
            post_current_mcp=post_current_mcp,
            post_stress_mcp=post_stress_mcp,
            safe_unit_capacity=safe_unit_capacity,
            planned_loss_jpy=planned_loss_jpy,
            post_margin_available_jpy=post_margin_available_jpy,
            post_max_currency_factor_nav_multiple=post_max_factor_nav_multiple,
        )
    if (
        post_current_mcp > float(limits.max_post_entry_current_mcp or 0.0)
        or post_stress_mcp > float(limits.max_post_entry_stress_mcp or 0.0)
        or any(abs(value) > factor_budget for value in post_factor_exposure.values())
    ):
        mode = RuntimeMode.FREEZE_NEW if has_bot_inventory else RuntimeMode.SHADOW_ONLY
        return _mode_receipt(
            mode,
            0,
            signal,
            "POST_ENTRY_MARGIN_OR_FACTOR_GATE_FAILED",
            post_current_mcp=post_current_mcp,
            post_stress_mcp=post_stress_mcp,
            safe_unit_capacity=safe_unit_capacity,
            planned_loss_jpy=planned_loss_jpy,
            post_margin_available_jpy=post_margin_available_jpy,
            post_max_currency_factor_nav_multiple=post_max_factor_nav_multiple,
        )

    retained_live = previous_mode in {RuntimeMode.THROTTLED_LIVE, RuntimeMode.FULL_LIVE}
    hysteresis = 0.0 if retained_live else float(limits.mode_hysteresis_mcp or 0.0)
    promote_current = float(limits.max_post_entry_current_mcp or 0.0) - hysteresis
    promote_stress = float(limits.max_post_entry_stress_mcp or 0.0) - hysteresis
    if post_current_mcp > promote_current or post_stress_mcp > promote_stress:
        mode = RuntimeMode.SHADOW_ONLY
    else:
        mode = RuntimeMode.THROTTLED_LIVE
    if mode is RuntimeMode.SHADOW_ONLY:
        units = 0
    return _mode_receipt(
        mode,
        units,
        signal,
        "PRE_FIXED_MODE_THRESHOLDS",
        post_current_mcp=post_current_mcp,
        post_stress_mcp=post_stress_mcp,
        safe_unit_capacity=safe_unit_capacity,
        planned_loss_jpy=planned_loss_jpy,
        post_margin_available_jpy=post_margin_available_jpy,
        post_max_currency_factor_nav_multiple=post_max_factor_nav_multiple,
    )


def _mode_receipt(
    mode: RuntimeMode,
    units: int,
    signal: SignalSizingInput,
    reason: str,
    *,
    post_current_mcp: float | None = None,
    post_stress_mcp: float | None = None,
    safe_unit_capacity: int | None = None,
    planned_loss_jpy: float | None = None,
    post_margin_available_jpy: float | None = None,
    post_max_currency_factor_nav_multiple: float | None = None,
) -> dict[str, Any]:
    return {
        "mode": mode.value,
        "transition_reason": reason,
        "requested_units": signal.requested_units,
        "calculated_units": units,
        "broker_minimum_units": signal.broker_minimum_units,
        "post_entry_current_mcp": post_current_mcp,
        "post_entry_stress_mcp": post_stress_mcp,
        "safe_unit_capacity": safe_unit_capacity,
        "planned_loss_jpy": planned_loss_jpy,
        "post_entry_margin_available_jpy": post_margin_available_jpy,
        "post_entry_max_currency_factor_nav_multiple": (
            post_max_currency_factor_nav_multiple
        ),
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
    stress_mcp = estimate_account_stress_mcp(
        snapshot=snapshot,
        raw_account=raw_account,
        stress_pips=STRESS_PIPS,
    )

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
    if not limits.numeric_complete:
        blockers.append("EXPLICIT_RISK_LIMITS_NOT_FIXED")
    elif not limits.proof_sealed:
        blockers.append("FORWARD_PROOF_OR_RISK_CONTRACT_UNSEALED")
    if system_positions:
        blockers.append("BOT_INVENTORY_NOT_FLAT")
    if pending_entries:
        blockers.append("PENDING_ENTRY_ORDER_PRESENT")
    if limits.numeric_complete and (
        mcp is None or mcp > float(limits.max_post_entry_current_mcp or 0.0)
    ):
        blockers.append("MARGIN_CLOSEOUT_PERCENT_ABOVE_RISK_CONTRACT")
    if limits.numeric_complete and (
        stress_mcp is None
        or stress_mcp > float(limits.max_post_entry_stress_mcp or 0.0)
    ):
        blockers.append("STRESS_MARGIN_CLOSEOUT_PERCENT_ABOVE_RISK_CONTRACT")
    if (
        limits.minimum_margin_buffer_jpy is not None
        and (margin_available is None or margin_available < limits.minimum_margin_buffer_jpy)
    ):
        blockers.append("MINIMUM_MARGIN_BUFFER_NOT_MET")
    factor_exposure_jpy = _currency_factor_jpy(snapshot)
    factor_budget_jpy = (nav or 0.0) * float(
        limits.max_currency_factor_nav_multiple or 0.0
    )
    if limits.numeric_complete and (factor_budget_jpy <= 0.0 or any(
        abs(value) > factor_budget_jpy for value in factor_exposure_jpy.values()
    )):
        blockers.append("CURRENCY_FACTOR_CONCENTRATION_ABOVE_BUDGET")
    blockers.extend(quote_blockers)

    if not software_ready:
        status = "software_unready"
    elif not limits.numeric_complete:
        status = "ready_waiting_for_risk_limits"
    elif not limits.proof_sealed:
        status = "ready_waiting_for_forward_admission"
    elif blockers:
        status = "ready_waiting_for_margin"
    else:
        status = "ready_for_final_screen"
    lifecycle = (
        "needs_user_decision"
        if status == "ready_waiting_for_risk_limits"
        else "waiting_external_state"
        if status.startswith("ready_waiting")
        else status
    )

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
            "numeric_complete": limits.numeric_complete,
            "proof_sealed": limits.proof_sealed,
            "max_loss_per_order_jpy": limits.max_loss_per_order_jpy,
            "stop_drawdown_jpy": limits.stop_drawdown_jpy,
            "minimum_margin_buffer_jpy": limits.minimum_margin_buffer_jpy,
            "max_post_entry_current_mcp": limits.max_post_entry_current_mcp,
            "max_post_entry_stress_mcp": limits.max_post_entry_stress_mcp,
            "max_currency_factor_nav_multiple": limits.max_currency_factor_nav_multiple,
            "max_bot_positions": limits.max_bot_positions,
            "mode_hysteresis_mcp": limits.mode_hysteresis_mcp,
            "forward_proof_sha256": limits.forward_proof_sha256,
            "risk_contract_sha256": limits.risk_contract_sha256,
        },
        "account": {
            "nav_jpy": nav,
            "margin_used_jpy": margin_used,
            "margin_available_jpy": margin_available,
            "margin_closeout_percent": mcp,
            "stress_margin_closeout_percent": stress_mcp,
            "stress_pips": STRESS_PIPS,
            "margin_available_nav_ratio": (
                margin_available / nav
                if margin_available is not None and nav is not None and nav > 0.0
                else None
            ),
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
    result = {
        currency: round(value * rates[currency], 6)
        for currency, value in sorted(raw.items())
        if currency in rates
    }
    return {currency: result.get(currency, 0.0) for currency in ("USD", "EUR", "JPY")}


def estimate_account_stress_mcp(
    *,
    snapshot: BrokerSnapshot,
    raw_account: Mapping[str, Any],
    stress_pips: float,
) -> float | None:
    """Conservatively shock every open lot adversely without mutating it."""

    account = raw_account.get("account") if isinstance(raw_account.get("account"), Mapping) else raw_account
    closeout_nav = _number(account.get("marginCloseoutNAV")) or _number(account.get("NAV"))
    closeout_margin = _number(account.get("marginCloseoutMarginUsed")) or _number(
        account.get("marginUsed")
    )
    if not _positive(closeout_nav) or closeout_margin is None or closeout_margin < 0.0:
        return None
    shock = _number(stress_pips)
    if shock is None or shock <= 0.0:
        return None
    total_loss_jpy = 0.0
    usd_jpy = snapshot.quotes.get("USD_JPY")
    for position in snapshot.positions:
        if "_" not in position.pair:
            return None
        quote_currency = position.pair.split("_", 1)[1]
        pip_value_quote = position.units / instrument_pip_factor(position.pair)
        if quote_currency == "JPY":
            quote_to_jpy = 1.0
        elif quote_currency == "USD" and usd_jpy is not None:
            quote_to_jpy = usd_jpy.mid
        else:
            conversion = _number(snapshot.home_conversions.get(quote_currency))
            if conversion is None or conversion <= 0.0:
                return None
            quote_to_jpy = conversion
        total_loss_jpy += shock * pip_value_quote * quote_to_jpy
    stressed_nav = closeout_nav - total_loss_jpy
    if stressed_nav <= 0.0:
        return math.inf
    return closeout_margin / stressed_nav


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
