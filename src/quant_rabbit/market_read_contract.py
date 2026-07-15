"""Shared semantic contract for GPT-authored MARKET_READ_FIRST payloads."""

from __future__ import annotations

from typing import Any, Mapping


MARKET_READ_NAKED_FIELDS = (
    "currency_bought",
    "currency_sold",
    "cleanest_pair_expression",
    "is_cleanest_currency_theme",
    "location_24h",
    "h1_h4_alignment",
    "tape_state",
    "known_winning_trade_shape_match",
    "proposed_building_style_allowed",
    "thesis_state",
    "what_price_is_trying_to_do_now",
)
MARKET_READ_PREDICTION_FIELDS = (
    "pair",
    "direction",
    "expected_path",
    "target_zone",
    "invalidation",
)
MARKET_READ_FORCED_TRADE_FIELDS = (
    "pair",
    "direction",
    "vehicle",
    "entry",
    "tp",
    "sl",
    "why_this_pays",
)
MARKET_READ_TAPE_STATES = frozenset({"TREND", "RANGE", "SQUEEZE", "FADE", "ROTATION"})
MARKET_READ_LOCATIONS_24H = frozenset({"LOWER", "MIDDLE", "UPPER", "UNKNOWN"})
MARKET_READ_THESIS_STATES = frozenset({"ALIVE", "WOUNDED", "INVALIDATED", "EMERGENCY", "UNKNOWN"})
MARKET_READ_VEHICLES = frozenset({"MARKET", "LIMIT", "STOP"})
MARKET_READ_SECTION_FIELDS = {
    "naked_read": MARKET_READ_NAKED_FIELDS,
    "next_30m_prediction": MARKET_READ_PREDICTION_FIELDS,
    "next_2h_prediction": MARKET_READ_PREDICTION_FIELDS,
    "best_trade_if_forced": MARKET_READ_FORCED_TRADE_FIELDS,
}


def market_read_missing_fields(market_read: Mapping[str, Any]) -> list[str]:
    """Return missing or noncanonical semantic fields in deterministic order."""

    missing: list[str] = []
    for unknown_section in sorted(set(market_read) - set(MARKET_READ_SECTION_FIELDS)):
        missing.append(f"market_read_first.{unknown_section}")
    for parent, fields in MARKET_READ_SECTION_FIELDS.items():
        section = market_read.get(parent)
        if not isinstance(section, Mapping):
            missing.append(parent)
            continue
        for field_name in fields:
            value = section.get(field_name)
            if not str(value or "").strip():
                missing.append(f"{parent}.{field_name}")
        for unknown_field in sorted(set(section) - set(fields)):
            missing.append(f"{parent}.{unknown_field}")

    forced = market_read.get("best_trade_if_forced")
    vehicle = str(forced.get("vehicle") or "").strip().upper() if isinstance(forced, Mapping) else ""
    if vehicle and vehicle not in MARKET_READ_VEHICLES:
        missing.append("best_trade_if_forced.vehicle")

    naked = market_read.get("naked_read")
    tape_state = str(naked.get("tape_state") or "").strip().upper() if isinstance(naked, Mapping) else ""
    if tape_state and tape_state not in MARKET_READ_TAPE_STATES:
        missing.append("naked_read.tape_state")
    location = str(naked.get("location_24h") or "").strip().upper() if isinstance(naked, Mapping) else ""
    if location and location not in MARKET_READ_LOCATIONS_24H:
        missing.append("naked_read.location_24h")
    thesis_state = str(naked.get("thesis_state") or "").strip().upper() if isinstance(naked, Mapping) else ""
    if thesis_state and thesis_state not in MARKET_READ_THESIS_STATES:
        missing.append("naked_read.thesis_state")
    return missing


def market_read_contract_payload() -> dict[str, Any]:
    """Machine-readable schema hints embedded in the evidence packet for GPT."""

    return {
        "required_sections": [
            *MARKET_READ_SECTION_FIELDS,
        ],
        "required_fields": {
            section: list(fields)
            for section, fields in MARKET_READ_SECTION_FIELDS.items()
        },
        "enums": {
            "naked_read.tape_state": sorted(MARKET_READ_TAPE_STATES),
            "naked_read.location_24h": sorted(MARKET_READ_LOCATIONS_24H),
            "naked_read.thesis_state": sorted(MARKET_READ_THESIS_STATES),
            "best_trade_if_forced.vehicle": sorted(MARKET_READ_VEHICLES),
        },
        "unknown_aliases_are_not_accepted": True,
        "derived_plan_and_operator_summary_must_match_final_market_read": True,
        "forced_prediction_current_technical_context_is_read_only": True,
        "forced_prediction_without_lane_must_request_intent_bridge": True,
        "forced_prediction_without_lane_may_grant_live_permission": False,
    }


def market_read_prediction_summary(market_read: Mapping[str, Any]) -> str:
    """Render the canonical forecast sentence used by final receipt prose.

    The deterministic draft and the later GPT market-read overlay both use
    this one renderer. That prevents the final action plan from retaining a
    superseded baseline direction after the overlay replaces MARKET_READ_FIRST.
    """

    next_30m = market_read.get("next_30m_prediction")
    next_30m = next_30m if isinstance(next_30m, Mapping) else {}
    next_2h = market_read.get("next_2h_prediction")
    next_2h = next_2h if isinstance(next_2h, Mapping) else {}
    if not next_30m and not next_2h:
        return (
            "MARKET READ FIRST next 30m/next 2h prediction is unavailable "
            "in this draft."
        )
    return (
        f"MARKET READ FIRST next 30m {next_30m.get('pair') or 'UNKNOWN_PAIR'} "
        f"{next_30m.get('direction') or 'UNKNOWN'} toward "
        f"{next_30m.get('target_zone') or 'unknown'}; next 2h "
        f"{next_2h.get('pair') or 'UNKNOWN_PAIR'} "
        f"{next_2h.get('direction') or 'UNKNOWN'} toward "
        f"{next_2h.get('target_zone') or 'unknown'}."
    )
