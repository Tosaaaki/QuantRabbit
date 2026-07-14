from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


P1_MARGIN_WARNING_CONTRACT = "QR_GUARDIAN_P1_MARGIN_WARNING_V1"
P0_MARGIN_HARD_CAP_CONTRACT = "QR_GUARDIAN_P0_MARGIN_HARD_CAP_V1"
MARGIN_PRESSURE_WARNING_CAP_FRACTION = 0.90


def exact_p1_margin_warning_source_event(
    *,
    event_type: Any,
    event_severity: Any,
    action: Any,
    details: Mapping[str, Any] | None,
    expected_max_margin_utilization_pct: float | None = None,
) -> bool:
    """Validate the canonical source-event meaning of a P1 margin warning.

    This function intentionally does not inspect a candidate order or grant
    routing permission.  Risk/Gateway callers layer their independently
    computed *projected* utilization on top of this immutable Guardian source
    contract.  Keeping the threshold and marker interpretation here prevents
    watchdog, GPT, and the final receipt-scope fence from disagreeing about
    whether the same event was P1 observation evidence or P0 hard-cap state.
    """

    if (
        str(event_type or "").strip().upper() != "MARGIN_PRESSURE"
        or str(event_severity or "").strip().upper() != "P1"
        or str(action or "").strip().upper() not in {"HOLD", "NO_ACTION"}
        or not isinstance(details, Mapping)
    ):
        return False
    if (
        details.get("fresh_entry_risk_block_active") is not False
        or str(details.get("fresh_entry_risk_block_reason") or "").upper()
        != "MARGIN_PRESSURE"
        or details.get("fresh_entry_risk_observation_only") is not True
        or str(details.get("fresh_entry_margin_contract") or "")
        != P1_MARGIN_WARNING_CONTRACT
    ):
        return False
    cap = strict_number(details.get("max_margin_utilization_pct"))
    nav = strict_number(details.get("nav_jpy"))
    used = strict_number(details.get("margin_used_jpy"))
    available = strict_number(details.get("margin_available_jpy"))
    expected_cap = strict_number(expected_max_margin_utilization_pct)
    if (
        cap is None
        or cap <= 0.0
        or cap > 100.0
        or nav is None
        or nav <= 0.0
        or used is None
        or used < 0.0
        or available is None
        or available <= 0.0
        or (
            expected_max_margin_utilization_pct is not None
            and (
                expected_cap is None
                or not math.isclose(
                    cap,
                    expected_cap,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            )
        )
    ):
        return False
    utilization_pct = used / nav * 100.0
    return (
        cap * MARGIN_PRESSURE_WARNING_CAP_FRACTION
        <= utilization_pct
        < cap
    )


def strict_number(value: Any) -> float | None:
    if value.__class__ not in {int, float}:
        return None
    number = float(value)
    return number if math.isfinite(number) else None
