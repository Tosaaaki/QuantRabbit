#!/usr/bin/env python3
"""Fail-closed research decision engine for the exact profitable vehicle."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any


TARGET = {
    "pair": "EUR_USD",
    "side": "SHORT",
    "strategy": "BREAKOUT_FAILURE",
    "order_type": "LIMIT",
    "exit_policy": "ATTACHED_TECHNICAL_TP_HARVEST",
}
REQUIRED_SNAPSHOT = {
    "decision_time",
    "causal_cutoff",
    "pair",
    "side",
    "strategy",
    "order_type",
    "exit_policy",
    "bid",
    "ask",
    "quote_time",
    "completed_bar",
    "prior_resistance",
    "wick_high",
    "body_close",
    "limit_price",
    "take_profit",
    "stop_loss",
    "fillability_known",
    "financing_known",
    "margin_available",
    "margin_required",
    "unwind_known",
}


def _utc(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def decide(snapshot: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    """Return exactly one decision without consulting realized outcomes."""

    blockers: list[str] = []
    missing = sorted(REQUIRED_SNAPSHOT - set(snapshot))
    if missing:
        blockers.append("MISSING_INPUTS:" + ",".join(missing))

    for key, expected in TARGET.items():
        if str(snapshot.get(key) or "").upper() != expected:
            blockers.append(f"VEHICLE_MISMATCH:{key}")

    decision_time = _utc(snapshot.get("decision_time"))
    causal_cutoff = _utc(snapshot.get("causal_cutoff"))
    quote_time = _utc(snapshot.get("quote_time"))
    if decision_time is None or causal_cutoff is None or quote_time is None:
        blockers.append("INVALID_TIME_LINEAGE")
    else:
        if causal_cutoff > decision_time:
            blockers.append("FUTURE_INPUT")
        quote_age = (decision_time - quote_time).total_seconds()
        if quote_age < 0:
            blockers.append("FUTURE_QUOTE")
        elif quote_age > 5:
            blockers.append("STALE_QUOTE")

    bid = _number(snapshot.get("bid"))
    ask = _number(snapshot.get("ask"))
    resistance = _number(snapshot.get("prior_resistance"))
    wick_high = _number(snapshot.get("wick_high"))
    body_close = _number(snapshot.get("body_close"))
    entry = _number(snapshot.get("limit_price"))
    tp = _number(snapshot.get("take_profit"))
    sl = _number(snapshot.get("stop_loss"))
    if None in (bid, ask, resistance, wick_high, body_close, entry, tp, sl):
        blockers.append("INVALID_PRICE_GEOMETRY")
    else:
        assert bid is not None and ask is not None and resistance is not None
        assert wick_high is not None and body_close is not None
        assert entry is not None and tp is not None and sl is not None
        if bid > ask or bid <= 0:
            blockers.append("INVALID_BID_ASK")
        else:
            spread_pips = (ask - bid) * 10_000.0
            if spread_pips > 2.0:
                blockers.append("SPREAD_ABOVE_FROZEN_BOUND")
        if snapshot.get("completed_bar") is not True:
            blockers.append("UNFINISHED_BAR")
        if not (wick_high > resistance and body_close < resistance):
            blockers.append("NO_COMPLETED_FAILED_BREAKOUT")
        tp_distance = (entry - tp) * 10_000.0
        if not (5.0 <= tp_distance <= 15.0 and sl > entry):
            blockers.append("INVALID_ATTACHED_TP_SL")

    for field, code in (
        ("fillability_known", "FILLABILITY_MISSING"),
        ("financing_known", "FINANCING_MISSING"),
        ("unwind_known", "UNWIND_MISSING"),
    ):
        if snapshot.get(field) is not True:
            blockers.append(code)
    margin_available = _number(snapshot.get("margin_available"))
    margin_required = _number(snapshot.get("margin_required"))
    if margin_available is None or margin_required is None or margin_required <= 0:
        blockers.append("MARGIN_MISSING")
    elif margin_available < 1.5 * margin_required:
        blockers.append("MARGIN_HEADROOM_LOW")

    sample_count = int(evidence.get("independent_samples") or 0)
    active_days = int(evidence.get("active_days") or 0)
    positive_day_rate = _number(evidence.get("positive_day_rate"))
    lcb = _number(evidence.get("lcb_jpy_per_1000u"))
    pf = _number(evidence.get("profit_factor"))
    if sample_count < 20:
        blockers.append("SAMPLE_FLOOR_NOT_MET")
    if active_days < 10:
        blockers.append("ACTIVE_DAY_FLOOR_NOT_MET")
    if positive_day_rate is None or positive_day_rate < 0.6:
        blockers.append("POSITIVE_DAY_RATE_NOT_MET")
    if lcb is None or lcb <= 0:
        blockers.append("LCB_NOT_POSITIVE")
    if pf is None or pf <= 1:
        blockers.append("PF_NOT_ABOVE_ONE")

    unique_blockers = sorted(set(blockers))
    action = "TRADE" if not unique_blockers else "WAIT"
    lineage = {
        "snapshot": snapshot,
        "evidence": evidence,
        "outcome_fields_consumed": [],
    }
    lineage_sha = hashlib.sha256(
        json.dumps(lineage, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "decision_id": hashlib.sha256(
            f"{snapshot.get('decision_time')}|{snapshot.get('pair')}|{snapshot.get('side')}".encode()
        ).hexdigest()[:24],
        "action": action,
        "pair": snapshot.get("pair"),
        "side": snapshot.get("side"),
        "entry": snapshot.get("limit_price") if action == "TRADE" else None,
        "take_profit": snapshot.get("take_profit") if action == "TRADE" else None,
        "stop_loss": snapshot.get("stop_loss") if action == "TRADE" else None,
        "decisive_constraint": unique_blockers[0] if unique_blockers else None,
        "abstain_reasons": unique_blockers,
        "outcome_fields_consumed": [],
        "input_output_lineage_sha256": lineage_sha,
    }
