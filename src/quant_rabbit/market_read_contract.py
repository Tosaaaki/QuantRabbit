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


def market_read_missing_fields(market_read: Mapping[str, Any]) -> list[str]:
    """Return missing or noncanonical semantic fields in deterministic order."""

    missing: list[str] = []
    nested_requirements = (
        ("naked_read", MARKET_READ_NAKED_FIELDS),
        ("next_30m_prediction", MARKET_READ_PREDICTION_FIELDS),
        ("next_2h_prediction", MARKET_READ_PREDICTION_FIELDS),
        ("best_trade_if_forced", MARKET_READ_FORCED_TRADE_FIELDS),
    )
    for parent, fields in nested_requirements:
        section = market_read.get(parent)
        if not isinstance(section, Mapping):
            missing.append(parent)
            continue
        for field_name in fields:
            value = section.get(field_name)
            if not str(value or "").strip():
                missing.append(f"{parent}.{field_name}")

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
            "naked_read",
            "next_30m_prediction",
            "next_2h_prediction",
            "best_trade_if_forced",
        ],
        "required_fields": {
            "naked_read": list(MARKET_READ_NAKED_FIELDS),
            "next_30m_prediction": list(MARKET_READ_PREDICTION_FIELDS),
            "next_2h_prediction": list(MARKET_READ_PREDICTION_FIELDS),
            "best_trade_if_forced": list(MARKET_READ_FORCED_TRADE_FIELDS),
        },
        "enums": {
            "naked_read.tape_state": sorted(MARKET_READ_TAPE_STATES),
            "naked_read.location_24h": sorted(MARKET_READ_LOCATIONS_24H),
            "naked_read.thesis_state": sorted(MARKET_READ_THESIS_STATES),
            "best_trade_if_forced.vehicle": sorted(MARKET_READ_VEHICLES),
        },
        "unknown_aliases_are_not_accepted": True,
    }
