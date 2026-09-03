"""Method-aware, entry-time-only edge snapshot for fast-bot shadow signals.

The snapshot deliberately makes no profitability claim.  It records one
precommitted decision from the same sealed regime row used to emit a shadow
signal, so a later exact-S5 scorer can compare accepted and vetoed entries
without reconstructing mutable chart state or looking at outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping


ENTRY_EDGE_CONTRACT = "QR_FAST_BOT_ENTRY_EDGE_SNAPSHOT_V1"
ENTRY_EDGE_POLICY = "METHOD_AWARE_CAUSAL_ENTRY_EDGE_V1"
REQUIRED_TIMEFRAMES = ("M1", "M5", "M15", "H1", "H4")
SUPPORTED_METHODS = {
    "BREAKOUT_FAILURE",
    "RANGE_ROTATION",
    "TREND_CONTINUATION",
}
SIDE_DIRECTION = {"LONG": "UP", "SHORT": "DOWN"}


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return {**body, "contract_sha256": _canonical_sha(body)}


def entry_edge_snapshot_valid(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    stored = value.get("contract_sha256")
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return bool(
        value.get("contract") == ENTRY_EDGE_CONTRACT
        and value.get("policy") == ENTRY_EDGE_POLICY
        and value.get("method") in SUPPORTED_METHODS
        and value.get("side") in SIDE_DIRECTION
        and isinstance(value.get("accepted"), bool)
        and isinstance(value.get("blockers"), list)
        and isinstance(stored, str)
        and stored == _canonical_sha(body)
    )


def _vote(votes: Mapping[str, Any], timeframe: str) -> Mapping[str, Any]:
    value = votes.get(timeframe)
    return value if isinstance(value, Mapping) else {}


def _aligned(vote: Mapping[str, Any], direction: str) -> bool:
    return (
        vote.get("evidence_complete") is True
        and str(vote.get("observed_direction") or "").upper() == direction
    )


def _opposed(vote: Mapping[str, Any], direction: str) -> bool:
    opposite = "DOWN" if direction == "UP" else "UP"
    return (
        vote.get("evidence_complete") is True
        and str(vote.get("observed_direction") or "").upper() == opposite
    )


def _triggered(vote: Mapping[str, Any]) -> bool:
    return str(vote.get("readiness") or "").upper() in {"TRIGGERED", "ACTIVE"}


def _same_side_exhausted(vote: Mapping[str, Any], side: str) -> bool:
    extension = str(vote.get("extension") or "").upper()
    value_zone = str(vote.get("value_zone") or "").upper()
    if side == "LONG":
        return extension in {"STRETCHED_UP", "OVERBOUGHT"} or value_zone == "DEEP_PREMIUM"
    return extension in {"STRETCHED_DOWN", "OVERSOLD"} or value_zone == "DEEP_DISCOUNT"


def _range_edge(vote: Mapping[str, Any], side: str) -> bool:
    location = str(vote.get("location") or "").upper()
    value_zone = str(vote.get("value_zone") or "").upper()
    if side == "LONG":
        return location == "LOWER_THIRD" or value_zone in {"DISCOUNT", "DEEP_DISCOUNT"}
    return location == "UPPER_THIRD" or value_zone in {"PREMIUM", "DEEP_PREMIUM"}


def build_entry_edge_snapshot(
    regime_row: Mapping[str, Any],
    *,
    reward_risk: float,
    spread_to_m5_atr: float | None,
) -> dict[str, Any]:
    """Build a sealed, method-aware decision using only emission-time fields."""

    side = str(regime_row.get("side") or "").upper()
    method = str(regime_row.get("method") or "").upper()
    direction = SIDE_DIRECTION.get(side)
    votes = (
        regime_row.get("timeframe_votes")
        if isinstance(regime_row.get("timeframe_votes"), Mapping)
        else {}
    )
    blockers: list[str] = []
    if direction is None:
        blockers.append("SIDE_UNSUPPORTED")
    if method not in SUPPORTED_METHODS:
        blockers.append("METHOD_UNSUPPORTED")
    incomplete = [
        timeframe
        for timeframe in REQUIRED_TIMEFRAMES
        if _vote(votes, timeframe).get("evidence_complete") is not True
    ]
    if incomplete:
        blockers.append("ENTRY_EDGE_EVIDENCE_INCOMPLETE:" + ",".join(incomplete))
    if (
        not isinstance(reward_risk, (int, float))
        or isinstance(reward_risk, bool)
        or not math.isfinite(float(reward_risk))
        or float(reward_risk) < 1.0
    ):
        blockers.append("REWARD_RISK_BELOW_ONE")
    if (
        not isinstance(spread_to_m5_atr, (int, float))
        or isinstance(spread_to_m5_atr, bool)
        or not math.isfinite(float(spread_to_m5_atr))
        or float(spread_to_m5_atr) > 0.35
    ):
        blockers.append("SPREAD_TO_M5_ATR_NOT_ECONOMIC")

    m1 = _vote(votes, "M1")
    m5 = _vote(votes, "M5")
    m15 = _vote(votes, "M15")
    h1 = _vote(votes, "H1")
    h4 = _vote(votes, "H4")
    if direction is not None:
        if not _triggered(m1):
            blockers.append("M1_REVERSAL_OR_CONTINUATION_NOT_TRIGGERED")
        if method == "TREND_CONTINUATION":
            if not all(_aligned(vote, direction) for vote in (m1, m5, m15)):
                blockers.append("FAST_TREND_DIRECTION_NOT_UNANIMOUS")
            if not _triggered(m5):
                blockers.append("M5_CONTINUATION_NOT_TRIGGERED")
            if not any(_aligned(vote, direction) for vote in (h1, h4)):
                blockers.append("NO_HIGHER_TIMEFRAME_DIRECTION_SUPPORT")
            if any(_same_side_exhausted(vote, side) for vote in (m5, m15)):
                blockers.append("FAST_TREND_ALREADY_EXHAUSTED")
            if all(_same_side_exhausted(vote, side) for vote in (h1, h4)):
                blockers.append("HIGHER_TIMEFRAME_ROOM_EXHAUSTED")
        elif method == "RANGE_ROTATION":
            if not _aligned(m1, direction):
                blockers.append("M1_RANGE_REVERSAL_DIRECTION_NOT_CONFIRMED")
            if not (_aligned(m5, direction) and _triggered(m5)):
                blockers.append("M5_RANGE_REVERSAL_NOT_TRIGGERED")
            if sum(
                str(vote.get("phase") or "").upper() in {"PRE_RANGE", "RANGE"}
                for vote in (m5, m15)
            ) < 2:
                blockers.append("M5_M15_RANGE_CONTEXT_NOT_CONFIRMED")
            if not any(_range_edge(vote, side) for vote in (m5, m15)):
                blockers.append("OPERATING_RANGE_EDGE_NOT_FAVORABLE")
            if all(_opposed(vote, direction) for vote in (h1, h4)):
                blockers.append("HIGHER_TIMEFRAME_BOTH_OPPOSE_RANGE_REVERSAL")
        elif method == "BREAKOUT_FAILURE":
            if str(regime_row.get("failed_break_direction") or "").upper() != side:
                blockers.append("FAILED_BREAK_NOT_BOUND_TO_SIDE")
            if not all(_aligned(vote, direction) for vote in (m1, m5)):
                blockers.append("M1_M5_FAILED_BREAK_DIRECTION_NOT_CONFIRMED")
            if not _triggered(m5):
                blockers.append("M5_FAILED_BREAK_NOT_TRIGGERED")
            if str(m15.get("phase") or "").upper() not in {"PRE_RANGE", "RANGE"}:
                blockers.append("M15_FAILURE_CONTEXT_NOT_CONFIRMED")
            if any(_same_side_exhausted(vote, side) for vote in (m5, m15)):
                blockers.append("FAILED_BREAK_REVERSAL_ALREADY_EXHAUSTED")

    normalized_blockers = sorted(set(blockers))
    feature_summary = {
        timeframe: {
            key: _vote(votes, timeframe).get(key)
            for key in (
                "observed_direction",
                "phase",
                "readiness",
                "trigger",
                "structure",
                "location",
                "value_zone",
                "extension",
                "evidence_complete",
            )
        }
        for timeframe in REQUIRED_TIMEFRAMES
    }
    return _seal(
        {
            "contract": ENTRY_EDGE_CONTRACT,
            "schema_version": 1,
            "policy": ENTRY_EDGE_POLICY,
            "side": side,
            "method": method,
            "accepted": not normalized_blockers,
            "blockers": normalized_blockers,
            "reward_risk": round(float(reward_risk), 6),
            "spread_to_m5_atr": (
                round(float(spread_to_m5_atr), 6)
                if isinstance(spread_to_m5_atr, (int, float))
                and not isinstance(spread_to_m5_atr, bool)
                and math.isfinite(float(spread_to_m5_atr))
                else None
            ),
            "failed_break_direction": regime_row.get("failed_break_direction"),
            "timeframe_features": feature_summary,
            "lookahead_used": False,
            "outcome_fields_used": [],
            "execution_authority": "NONE",
            "live_permission": False,
        }
    )
