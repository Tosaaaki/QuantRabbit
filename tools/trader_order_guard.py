#!/usr/bin/env python3
"""Shared exact-pretrade guard for live trader entry orders."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "collab_trade" / "memory"))

from pretrade_check import assess_risk  # type: ignore  # noqa: E402


ENTRY_PENDING_TYPES = {"LIMIT", "STOP", "MARKET_IF_TOUCHED"}
ALLOWED_REQUESTED_STYLES = {
    "PASS": set(),
    "LIMIT": {"LIMIT"},
    "STOP-ENTRY": {"LIMIT", "STOP-ENTRY"},
    "MARKET": {"LIMIT", "STOP-ENTRY", "MARKET"},
}


def requested_style_from_order_type(order_type: str | None) -> str | None:
    upper = str(order_type or "").strip().upper()
    if upper == "MARKET":
        return "MARKET"
    if upper == "LIMIT":
        return "LIMIT"
    if upper in {"STOP", "STOP-ENTRY", "MARKET_IF_TOUCHED"}:
        return "STOP-ENTRY"
    return None


def exact_execution_style(result: dict | None) -> str:
    return str(((result or {}).get("execution_plan") or {}).get("style") or "PASS").upper()


def exact_execution_note(result: dict | None) -> str:
    return str(((result or {}).get("execution_plan") or {}).get("note") or "").strip()


def exact_allocation_grade(result: dict | None) -> str | None:
    setup = (result or {}).get("setup_quality") or {}
    value = str(setup.get("allocation_grade") or setup.get("grade") or "").strip().upper()
    return value or None


def exact_allocation_band(result: dict | None) -> str | None:
    setup = (result or {}).get("setup_quality") or {}
    value = str(
        setup.get("allocation_band")
        or setup.get("allocation_grade")
        or setup.get("grade")
        or ""
    ).strip().upper()
    return value or None


def exact_pretrade_label(result: dict | None) -> str:
    setup = (result or {}).get("setup_quality") or {}
    edge = str(setup.get("edge_grade") or setup.get("grade") or "?").strip().upper() or "?"
    raw_score = setup.get("quality_score")
    if raw_score is None:
        raw_score = setup.get("score")
    try:
        score = int(float(raw_score or 0))
    except (TypeError, ValueError):
        score = 0
    try:
        score_max = int(float(setup.get("score_max") or (7 if setup.get("is_counter") else 10)))
    except (TypeError, ValueError):
        score_max = 7 if setup.get("is_counter") else 10
    allocation = exact_allocation_grade(result)
    band = exact_allocation_band(result)
    label = f"{edge}({score}/{score_max})"
    if allocation and band and band != allocation:
        label += f"->{band}"
    return label


def exact_pretrade_hard_blockers(result: dict | None) -> list[str]:
    blockers = []
    payload = (result or {}).get("hard_execution_blockers") or []
    if not payload:
        payload = ((result or {}).get("execution_plan") or {}).get("hard_blockers") or []
    for item in payload:
        text = " ".join(str(item or "").split()).strip()
        if text and text not in blockers:
            blockers.append(text)
    return blockers


def exact_style_allows_requested(exact_style: str, requested_style: str | None) -> bool:
    if not requested_style:
        return False
    allowed = ALLOWED_REQUESTED_STYLES.get(str(exact_style or "PASS").upper(), set())
    return requested_style in allowed


def exact_pretrade_issues(
    *,
    requested_style: str | None,
    result: dict | None,
) -> list[str]:
    hard_blockers = exact_pretrade_hard_blockers(result)
    if hard_blockers:
        return hard_blockers
    if not requested_style:
        return [f"unsupported requested order style for exact pretrade guard: {requested_style!r}"]
    return []


def exact_pretrade_advisories(
    *,
    requested_style: str | None,
    result: dict | None,
) -> list[str]:
    if exact_pretrade_hard_blockers(result):
        return []
    exact_style = exact_execution_style(result)
    note = exact_execution_note(result)
    if not requested_style:
        return []
    if exact_style_allows_requested(exact_style, requested_style):
        return []
    if exact_style == "PASS":
        reason = "exact pretrade would stand down here; trader override is discretionary, not geometry-forced"
    elif exact_style == "LIMIT":
        reason = "exact pretrade prefers LIMIT here; trader is overriding the price-improvement preference"
    elif exact_style == "STOP-ENTRY":
        reason = "exact pretrade prefers STOP-ENTRY/LIMIT here; trader is overriding the trigger-first timing"
    else:
        reason = f"requested {requested_style} is more aggressive than exact pretrade {exact_style}"
    if note:
        reason += f": {note}"
    return [reason]


def run_exact_pretrade(
    *,
    pair: str,
    direction: str,
    entry_price: float | None,
    tp_price: float,
    sl_price: float,
    counter: bool = False,
    regime: str | None = None,
    spread_pips: float | None = None,
    live_tape: dict | None = None,
) -> dict:
    return assess_risk(
        pair,
        direction,
        counter=counter,
        regime=regime,
        entry_price=entry_price,
        tp_price=tp_price,
        sl_price=sl_price,
        spread_pips=spread_pips,
        live_tape=live_tape,
        log_result=False,
    )
