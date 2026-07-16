"""Deterministic 30-second bot supervision and forward-shadow signals.

The fast bot does not ask an AI model to approve individual trades.  It binds
M1 execution, M5/M15/M30 operating state, H1/H4 structure, and the D anchor
into one finite GO/CAUTION/STOP contract.  AI output may pause or de-risk a
pair after a material regime review, but cannot manufacture an entry signal.

This module is deliberately shadow-only.  It emits immutable passive-entry
hypotheses for exact OANDA S5 bid/ask scoring.  A separate, content-addressed
promotion contract is required before any signal can reach LiveOrderGateway.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from quant_rabbit.guardian_observation import (
    CURRENT_M1_CONTRACT,
    SLOW_RETENTION_CONTRACT,
    validate_current_m1_contract,
    validate_slow_retention_contract,
)
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS
from quant_rabbit.instruments import instrument_pip_factor
from quant_rabbit.strategy.failed_break_evidence import (
    build_m5_failed_break_evidence,
    failed_break_direction,
)


REGIME_CONTRACT = "QR_HIERARCHICAL_BOT_REGIME_V1"
SHADOW_CONTRACT = "QR_FAST_BOT_FORWARD_SHADOW_V1"
SIGNAL_CONTRACT = "QR_FAST_BOT_SHADOW_SIGNAL_V1"
AI_SUPERVISION_CONTRACT = "QR_AI_REGIME_SUPERVISION_V1"
ENTRY_EXPERIMENT_CONTRACT = "QR_FAST_BOT_PASSIVE_ENTRY_EXPERIMENT_V1"
# These are canonical quartiles of the observed executable spread, not tuned
# price literals.  They precommit a maker-aggressiveness grid from the quote's
# passive near side toward (but never through) the opposite side.  Forward
# paired S5 evidence should replace the grid if a later experiment changes it.
ENTRY_ARM_SPREAD_FRACTIONS = (
    ("PASSIVE_NEAR_SIDE", 0.0),
    ("PASSIVE_QUARTER_SPREAD", 0.25),
    ("PASSIVE_MID_SPREAD", 0.50),
    ("PASSIVE_THREE_QUARTER_SPREAD", 0.75),
)
ENTRY_TTL_SECONDS = 90
MAX_HOLD_SECONDS = 15 * 60
HORIZON_LANE = "M1_EXECUTION_15M_HOLD"
TIMEFRAME_ROLES = {
    "execution": ("M1",),
    "operating": ("M5", "M15", "M30"),
    "structure": ("H1", "H4"),
    "anchor": ("D",),
}
METHODS = ("BREAKOUT_FAILURE", "RANGE_ROTATION", "TREND_CONTINUATION")
SIDES = ("LONG", "SHORT")
SIDE_DIRECTION = {"LONG": "UP", "SHORT": "DOWN"}
OPPOSITE_DIRECTION = {"UP": "DOWN", "DOWN": "UP"}
AI_WAKE_EVENT_TYPES = {
    "TECHNICAL_STATE_CHANGE",
    "SPREAD_ANOMALY",
    "VOLATILITY_SHOCK",
    "PERFORMANCE_DEGRADATION",
}
HARD_STALE_EVENT_TYPES = {"TECHNICAL_INPUT_STALE"}
ACTIVE_AI_MODES = {"GO", "CAUTION", "STOP"}
FAST_CHART_MAX_AGE_SECONDS = 180
QUOTE_MAX_AGE_SECONDS = 45
AI_TUNING_INTERVAL_SECONDS = 6 * 60 * 60
AI_SUPERVISION_HANDOFF_GRACE_SECONDS = 15 * 60
TIMEFRAME_SECONDS = {
    "M1": 60,
    "M5": 5 * 60,
    "M15": 15 * 60,
    "M30": 30 * 60,
    "H1": 60 * 60,
    "H4": 4 * 60 * 60,
    "D": 24 * 60 * 60,
}


def build_hierarchical_regime_contract(
    *,
    fast_pair_charts: Mapping[str, Any],
    slow_pair_charts: Mapping[str, Any],
    broker_snapshot: Mapping[str, Any],
    guardian_events: Mapping[str, Any] | None = None,
    ai_supervision: Mapping[str, Any] | None = None,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Build pair/side/method GO gates without a per-trade AI decision."""

    now = _aware_utc(now_utc or datetime.now(timezone.utc))
    fast_pair_charts = _validated_fast_pair_charts(fast_pair_charts, now_utc=now)
    slow_pair_charts = _validated_slow_pair_charts(slow_pair_charts)
    fast_generated = _parse_utc(fast_pair_charts.get("generated_at_utc"))
    slow_generated = _parse_utc(slow_pair_charts.get("generated_at_utc"))
    snapshot_at = _parse_utc(broker_snapshot.get("fetched_at_utc"))
    fast_sha = _canonical_sha(fast_pair_charts)
    slow_sha = _canonical_sha(slow_pair_charts)
    snapshot_sha = _canonical_sha(broker_snapshot)
    event_payload = guardian_events if isinstance(guardian_events, Mapping) else {}
    events = [item for item in event_payload.get("events", []) or [] if isinstance(item, Mapping)]
    validated_supervision = (
        ai_supervision
        if _ai_supervision_contract_valid(ai_supervision or {})
        else {}
    )
    stale_pairs = {
        _pair(item.get("pair"))
        for item in events
        if str(item.get("event_type") or "").upper() in HARD_STALE_EVENT_TYPES
    }
    ai_wake_events = [
        _compact_event(item)
        for item in events
        if str(item.get("event_type") or "").upper() in AI_WAKE_EVENT_TYPES
    ]
    fast_by_pair = _charts_by_pair(fast_pair_charts)
    slow_by_pair = _charts_by_pair(slow_pair_charts)
    quotes = broker_snapshot.get("quotes") if isinstance(broker_snapshot.get("quotes"), Mapping) else {}
    pairs = sorted(set(fast_by_pair) | set(slow_by_pair))
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        merged_views = _merged_views(fast_by_pair.get(pair), slow_by_pair.get(pair))
        merged_chart = {"pair": pair, "views": list(merged_views.values())}
        m5_failed_break = build_m5_failed_break_evidence(merged_chart)
        quote = quotes.get(pair) if isinstance(quotes.get(pair), Mapping) else {}
        quote_at = _parse_utc(quote.get("timestamp_utc"))
        spread = _spread_pips(pair, quote)
        m5_atr = _view_atr_pips(merged_views.get("M5"))
        spread_to_m5_atr = (
            spread / m5_atr
            if spread is not None and m5_atr is not None and m5_atr > 0.0
            else None
        )
        common_hard: list[str] = []
        common_caution: list[str] = []
        if not _timestamp_current(
            now,
            fast_generated,
            max_age_seconds=FAST_CHART_MAX_AGE_SECONDS,
        ):
            common_hard.append("FAST_CHART_PACKET_STALE")
        if not _timestamp_current(
            now,
            snapshot_at,
            max_age_seconds=QUOTE_MAX_AGE_SECONDS,
        ):
            common_hard.append("BROKER_SNAPSHOT_OR_QUOTES_STALE")
        if not _timestamp_current(
            now,
            quote_at,
            max_age_seconds=QUOTE_MAX_AGE_SECONDS,
        ):
            common_hard.append("PAIR_QUOTE_STALE_OR_FUTURE")
        if pair in stale_pairs:
            common_hard.append("TECHNICAL_INPUT_STALE")
        if any(_view_integrity_blocked(merged_views.get(tf)) for tf in ("M1", "M5", "M15")):
            common_hard.append("FAST_TECHNICAL_CANDLE_INTEGRITY_BLOCKED")
        missing_fast = [tf for tf in ("M1", "M5", "M15") if not _usable_view(merged_views.get(tf))]
        if missing_fast:
            common_hard.append("FAST_TIMEFRAME_EVIDENCE_MISSING:" + ",".join(missing_fast))
        stale_fast = [
            tf
            for tf in ("M1", "M5", "M15")
            if tf not in missing_fast
            and not _view_candle_fresh(merged_views.get(tf), timeframe=tf, now=now)
        ]
        if stale_fast:
            common_hard.append("FAST_CLOSED_CANDLE_STALE_OR_FUTURE:" + ",".join(stale_fast))
        missing_slow = [tf for tf in ("M30", "H1", "H4", "D") if not _usable_view(merged_views.get(tf))]
        if missing_slow:
            common_caution.append("SLOW_TIMEFRAME_EVIDENCE_MISSING:" + ",".join(missing_slow))
        stale_slow = [
            tf
            for tf in ("M30", "H1", "H4", "D")
            if tf not in missing_slow
            and not _view_candle_fresh(merged_views.get(tf), timeframe=tf, now=now)
        ]
        if stale_slow:
            common_caution.append("SLOW_CLOSED_CANDLE_STALE_OR_FUTURE:" + ",".join(stale_slow))
        if spread is None or spread_to_m5_atr is None:
            common_hard.append("SPREAD_OR_M5_ATR_UNAVAILABLE")
        elif spread > 3.0 or spread_to_m5_atr > 0.35:
            common_hard.append("SPREAD_ANOMALY")

        ai_state = _ai_pair_state(validated_supervision, pair=pair, now=now)
        if ai_state["mode"] == "STOP":
            common_hard.append("AI_REGIME_SUPERVISOR_STOP")
        elif ai_state["mode"] == "CAUTION":
            common_caution.append("AI_REGIME_SUPERVISOR_CAUTION")

        for side in SIDES:
            for method in METHODS:
                evaluated = _evaluate_method(
                    pair=pair,
                    side=side,
                    method=method,
                    views=merged_views,
                    failed_break=m5_failed_break,
                )
                hard = [*common_hard, *evaluated["hard_blockers"]]
                caution = [*common_caution, *evaluated["caution_reasons"]]
                if hard:
                    state = "STOP"
                elif evaluated["go"] and not caution:
                    state = "GO"
                else:
                    state = "CAUTION"
                rows.append(
                    {
                        "pair": pair,
                        "side": side,
                        "method": method,
                        "state": state,
                        "score": round(float(evaluated["score"]), 6),
                        "execution_enabled": state == "GO",
                        "size_cap_multiple": 1.0 if state == "GO" else 0.0,
                        "hard_blockers": sorted(set(hard)),
                        "caution_reasons": sorted(set(caution)),
                        "timeframe_votes": evaluated["timeframe_votes"],
                        "m1_closed_candle_utc": _latest_complete_candle_close_time(
                            merged_views.get("M1"),
                            timeframe="M1",
                        ),
                        "m5_atr_pips": _round(m5_atr, 6),
                        "spread_pips": _round(spread, 6),
                        "spread_to_m5_atr": _round(spread_to_m5_atr, 6),
                        "ai_supervision": ai_state,
                        "failed_break_direction": failed_break_direction(m5_failed_break),
                    }
                )

    last_tuned = _parse_utc(
        validated_supervision.get("last_tuned_at_utc")
        if isinstance(validated_supervision, Mapping)
        else None
    )
    tuning_due = last_tuned is None or _age_seconds(now, last_tuned) >= AI_TUNING_INTERVAL_SECONDS
    body = {
        "contract": REGIME_CONTRACT,
        "schema_version": 1,
        "generated_at_utc": now.isoformat(),
        "timeframe_roles": {key: list(value) for key, value in TIMEFRAME_ROLES.items()},
        "execution_cadence_seconds": 30,
        "entry_decision_authority": "DETERMINISTIC_BOT",
        "ai_role": "MATERIAL_REGIME_REVIEW_AND_PERIODIC_TUNING_ONLY",
        "ai_per_trade_approval_required": False,
        "ai_wake_required": bool(ai_wake_events or tuning_due),
        "ai_wake_reasons": [
            *(f"GUARDIAN_EVENT:{item['event_type']}:{item['pair']}" for item in ai_wake_events),
            *(["PERIODIC_TUNING_DUE"] if tuning_due else []),
        ],
        "ai_wake_events": ai_wake_events,
        "tuning_due": tuning_due,
        "last_tuned_at_utc": last_tuned.isoformat() if last_tuned else None,
        "rows": rows,
        "sources": {
            "fast_pair_charts_sha256": fast_sha,
            "slow_pair_charts_sha256": slow_sha,
            "broker_snapshot_sha256": snapshot_sha,
            "fast_pair_charts_generated_at_utc": fast_generated.isoformat() if fast_generated else None,
            "slow_pair_charts_generated_at_utc": slow_generated.isoformat() if slow_generated else None,
            "broker_snapshot_fetched_at_utc": snapshot_at.isoformat() if snapshot_at else None,
        },
    }
    return _seal(body)


def build_fast_bot_shadow(
    regime_contract: Mapping[str, Any],
    *,
    broker_snapshot: Mapping[str, Any],
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Emit one passive, attached-TP/SL hypothesis per GO pair and M1 close."""

    now = _aware_utc(now_utc or datetime.now(timezone.utc))
    if not _sealed_contract_valid(regime_contract, REGIME_CONTRACT):
        return _seal(
            {
                "contract": SHADOW_CONTRACT,
                "schema_version": 1,
                "generated_at_utc": now.isoformat(),
                "status": "INVALID_REGIME_CONTRACT",
                "shadow_only": True,
                "live_permission": False,
                "broker_mutation_allowed": False,
                "signals": [],
            }
        )
    quotes = broker_snapshot.get("quotes") if isinstance(broker_snapshot.get("quotes"), Mapping) else {}
    go_rows = [
        dict(item)
        for item in regime_contract.get("rows", []) or []
        if isinstance(item, Mapping)
        and item.get("state") == "GO"
        and item.get("execution_enabled") is True
    ]
    signals: list[dict[str, Any]] = []
    for row in sorted(
        go_rows,
        key=lambda item: (
            str(item.get("pair") or ""),
            str(item.get("side") or ""),
            str(item.get("method") or ""),
        ),
    ):
        pair = str(row.get("pair") or "")
        quote = quotes.get(pair) if isinstance(quotes.get(pair), Mapping) else {}
        bid = _positive_number(quote.get("bid"))
        ask = _positive_number(quote.get("ask"))
        quote_at = _parse_utc(quote.get("timestamp_utc"))
        if (
            bid is None
            or ask is None
            or ask <= bid
            or not _timestamp_current(
                now,
                quote_at,
                max_age_seconds=QUOTE_MAX_AGE_SECONDS,
            )
        ):
            continue
        # Persist and derive every price-dependent field from the same broker
        # precision.  This keeps the append-only signal fully reproducible even
        # when an upstream JSON number carried extra binary precision.
        bid = _price(pair, bid)
        ask = _price(pair, ask)
        side = str(row["side"])
        method = str(row["method"])
        spread = round(
            (ask - bid) * float(instrument_pip_factor(pair)),
            6,
        )
        atr = round(float(row.get("m5_atr_pips") or 0.0), 6)
        if spread <= 0.0 or atr <= 0.0:
            continue
        tp_pips, sl_pips = _shadow_geometry_pips(method, spread=spread, m5_atr=atr)
        entry_arms = _entry_experiment_arms(
            pair=pair,
            side=side,
            bid=bid,
            ask=ask,
            tp_pips=tp_pips,
            sl_pips=sl_pips,
        )
        primary = entry_arms[0]
        entry = float(primary["entry"])
        tp = float(primary["take_profit"])
        sl = float(primary["stop_loss"])
        actual_tp_pips = float(primary["take_profit_pips"])
        actual_sl_pips = float(primary["stop_loss_pips"])
        identity = {
            "identity_contract": "QR_FAST_BOT_SHADOW_IDENTITY_V3",
            "pair": pair,
            "m1_closed_candle_utc": row.get("m1_closed_candle_utc"),
            "side": side,
            "method": method,
            "horizon_lane": HORIZON_LANE,
        }
        evidence_binding = {
            **identity,
            "regime_contract_sha256": regime_contract.get("contract_sha256"),
        }
        signal_body = {
            "contract": SIGNAL_CONTRACT,
            "schema_version": 3,
            "signal_id": _canonical_sha(identity)[:24],
            **evidence_binding,
            "generated_at_utc": now.isoformat(),
            "quote_timestamp_utc": quote_at.isoformat(),
            "quote_bid": bid,
            "quote_ask": ask,
            "spread_pips": round(spread, 6),
            "m5_atr_pips": round(atr, 6),
            "geometry_policy": "METHOD_SPREAD_M5_ATR_V1",
            "order_type": "LIMIT",
            "entry_reference": "PASSIVE_NEAR_SIDE",
            "entry": _price(pair, entry),
            "take_profit": _price(pair, tp),
            "stop_loss": _price(pair, sl),
            "take_profit_pips": actual_tp_pips,
            "stop_loss_pips": actual_sl_pips,
            "reward_risk": round(actual_tp_pips / actual_sl_pips, 6),
            "entry_ttl_seconds": ENTRY_TTL_SECONDS,
            "max_hold_seconds": MAX_HOLD_SECONDS,
            "entry_experiment_contract": ENTRY_EXPERIMENT_CONTRACT,
            "entry_experiment_arms": entry_arms,
            "attached_take_profit_required": True,
            "attached_stop_loss_required": True,
            "regime_score": row.get("score"),
            "shadow_only": True,
            "live_permission": False,
            "broker_mutation_allowed": False,
        }
        signals.append({**signal_body, "signal_sha256": _canonical_sha(signal_body)})
    body = {
        "contract": SHADOW_CONTRACT,
        "schema_version": 1,
        "generated_at_utc": now.isoformat(),
        "status": "EMITTED" if signals else "NO_GO_SIGNAL",
        "shadow_only": True,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "ai_per_trade_approval_required": False,
        "candidate_projection": "ALL_GO_ROWS_NO_PAIR_OR_SIDE_NETTING",
        "candidate_count_by_horizon_lane": {HORIZON_LANE: len(signals)},
        "signals": signals,
        "promotion_contract": {
            "status": "BLOCKED_PENDING_FORWARD_PROOF",
            "minimum_exact_s5_resolved_fills": 100,
            "minimum_active_days": 10,
            "minimum_profit_factor": 1.25,
            "minimum_one_sided_95_expectancy_lower_pips": 0.0,
            "maximum_spread_anomaly_rate": 0.02,
            "initial_live_risk_pct_nav": 0.05,
            "blockers": [
                "EXACT_OANDA_S5_BID_ASK_FORWARD_OUTCOMES_REQUIRED",
                "POST_COST_EXPECTANCY_LOWER_BOUND_NOT_PROVEN",
                "HORIZON_AWARE_MULTI_GO_PORTFOLIO_FORWARD_PROOF_REQUIRED",
                "COUNTERFACTUAL_LEARNING_OUTCOME_PROOF_REQUIRED",
                "SEPARATE_CONTENT_ADDRESSED_LIVE_PROMOTION_REQUIRED",
            ],
        },
        "regime_contract_sha256": regime_contract.get("contract_sha256"),
    }
    return _seal(body)


def run_fast_bot_shadow(
    *,
    fast_pair_charts_path: Path,
    slow_pair_charts_path: Path,
    broker_snapshot_path: Path,
    guardian_events_path: Path,
    ai_supervision_path: Path | None,
    regime_output_path: Path,
    shadow_output_path: Path,
    shadow_ledger_path: Path,
    report_path: Path,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Read current artifacts and atomically persist the bot shadow cycle."""

    fast = _read_object(fast_pair_charts_path)
    slow = _read_object(slow_pair_charts_path)
    snapshot = _read_object(broker_snapshot_path)
    events = _read_object(guardian_events_path)
    supervision = _read_object(ai_supervision_path) if ai_supervision_path else {}
    contract = build_hierarchical_regime_contract(
        fast_pair_charts=fast,
        slow_pair_charts=slow,
        broker_snapshot=snapshot,
        guardian_events=events,
        ai_supervision=supervision,
        now_utc=now_utc,
    )
    shadow = build_fast_bot_shadow(contract, broker_snapshot=snapshot, now_utc=now_utc)
    _write_json_atomic(regime_output_path, contract)
    _write_json_atomic(shadow_output_path, shadow)
    appended = _append_signals_once(shadow_ledger_path, shadow)
    _write_report(report_path, contract=contract, shadow=shadow, appended=appended)
    return {
        "status": shadow.get("status"),
        "go_gate_count": sum(
            1 for item in contract.get("rows", []) if isinstance(item, Mapping) and item.get("state") == "GO"
        ),
        "signal_count": len(shadow.get("signals", []) or []),
        "ledger_appended": appended,
        "ai_wake_required": contract.get("ai_wake_required"),
        "ai_wake_reasons": contract.get("ai_wake_reasons"),
        "shadow_only": True,
        "live_permission": False,
        "broker_mutation": False,
        "regime_output": str(regime_output_path),
        "shadow_output": str(shadow_output_path),
        "shadow_ledger": str(shadow_ledger_path),
        "report": str(report_path),
    }


def _evaluate_method(
    *,
    pair: str,
    side: str,
    method: str,
    views: Mapping[str, Mapping[str, Any]],
    failed_break: Mapping[str, Any],
) -> dict[str, Any]:
    del pair
    direction = SIDE_DIRECTION[side]
    votes = {tf: _view_vote(views.get(tf), direction=direction) for tf in ("M1", "M5", "M15", "M30", "H1", "H4", "D")}
    hard: list[str] = []
    caution: list[str] = []
    score = sum(votes[tf]["direction_score"] * weight for tf, weight in {
        "M1": 1.5, "M5": 2.0, "M15": 2.0, "M30": 1.5, "H1": 1.5, "H4": 1.0, "D": 0.5,
    }.items())
    m1 = votes["M1"]
    m5 = votes["M5"]
    m15 = votes["M15"]
    m30 = votes["M30"]
    h1 = votes["H1"]
    h4 = votes["H4"]
    day = votes["D"]
    readiness = {m1["readiness"], m5["readiness"]}
    trigger_ready = bool(readiness & {"TRIGGERED", "ARMED"})

    if method == "BREAKOUT_FAILURE":
        proof_side = failed_break_direction(failed_break)
        if proof_side != side:
            hard.append("M5_FAILED_BREAK_DIRECTION_NOT_BOUND_TO_SIDE")
        if not trigger_ready:
            caution.append("FAST_REVERSAL_TRIGGER_NOT_READY")
        if m15["phase"] not in {"PRE_RANGE", "RANGE"}:
            caution.append("M15_NOT_IN_FAILURE_OR_ROTATION_PHASE")
        if h1["direction_score"] < 0 and h4["direction_score"] < 0:
            caution.append("H1_H4_BOTH_OPPOSE_FAILED_BREAK_REVERSAL")
        go = proof_side == side and trigger_ready and m15["phase"] in {"PRE_RANGE", "RANGE"}
        score += 4.0 if proof_side == side else -4.0
    elif method == "RANGE_ROTATION":
        range_count = sum(vote["phase"] in {"PRE_RANGE", "RANGE"} for vote in (m5, m15, m30))
        location_ok = _range_location_supports_side(m1, side) or _range_location_supports_side(m5, side)
        if range_count < 2:
            hard.append("OPERATING_RANGE_PHASE_NOT_CONFIRMED")
        if not location_ok:
            hard.append("RANGE_EDGE_LOCATION_DOES_NOT_SUPPORT_SIDE")
        if not trigger_ready:
            caution.append("RANGE_ROTATION_TRIGGER_NOT_READY")
        if h1["direction_score"] < 0 and h4["direction_score"] < 0:
            caution.append("H1_H4_TREND_OPPOSES_RANGE_ROTATION")
        go = range_count >= 2 and location_ok and trigger_ready
        score += float(range_count) + (2.0 if location_ok else -2.0)
    else:
        operating_align = sum(vote["direction_score"] > 0 for vote in (m5, m15, m30))
        structure_align = sum(vote["direction_score"] > 0 for vote in (h1, h4))
        trend_phase_count = sum(vote["phase"] in {"PRE_TREND", "TREND"} for vote in (m1, m5, m15))
        if m1["direction_score"] <= 0:
            hard.append("M1_EXECUTION_DIRECTION_NOT_ALIGNED")
        if operating_align < 2:
            hard.append("OPERATING_DIRECTION_NOT_ALIGNED")
        if structure_align < 1:
            caution.append("H1_H4_STRUCTURE_NOT_ALIGNED")
        if trend_phase_count < 2:
            hard.append("FAST_TREND_PHASE_NOT_CONFIRMED")
        if not trigger_ready:
            caution.append("CONTINUATION_TRIGGER_NOT_READY")
        if day["direction_score"] < 0 and day["phase"] == "TREND":
            caution.append("D_ANCHOR_STRONGLY_OPPOSES_CONTINUATION")
        go = (
            m1["direction_score"] > 0
            and operating_align >= 2
            and structure_align >= 1
            and trend_phase_count >= 2
            and trigger_ready
        )
        score += float(operating_align + structure_align + trend_phase_count)
    return {
        "go": bool(go),
        "score": score,
        "hard_blockers": hard,
        "caution_reasons": caution,
        "timeframe_votes": votes,
    }


def _view_vote(view: Mapping[str, Any] | None, *, direction: str) -> dict[str, Any]:
    market = view.get("market_state") if isinstance(view, Mapping) and isinstance(view.get("market_state"), Mapping) else {}
    observed_direction = str(market.get("direction") or "UNKNOWN").upper()
    score = 1 if observed_direction == direction else -1 if observed_direction == OPPOSITE_DIRECTION[direction] else 0
    return {
        "observed_direction": observed_direction,
        "direction_score": score,
        "phase": str(market.get("phase") or "UNKNOWN").upper(),
        "readiness": str(market.get("readiness") or "UNKNOWN").upper(),
        "trigger": str(market.get("trigger") or "UNKNOWN").upper(),
        "structure": str(market.get("structure") or "UNKNOWN").upper(),
        "location": str(market.get("location") or "UNKNOWN").upper(),
        "value_zone": str(market.get("value_zone") or "UNKNOWN").upper(),
        "extension": str(market.get("extension") or "UNKNOWN").upper(),
        "evidence_complete": market.get("evidence_complete") is True,
    }


def _range_location_supports_side(vote: Mapping[str, Any], side: str) -> bool:
    observed = str(vote.get("observed_direction") or "")
    location = str(vote.get("location") or "")
    value = str(vote.get("value_zone") or "")
    extension = str(vote.get("extension") or "")
    if side == "LONG":
        return observed == "DOWN" and (
            location == "LOWER_THIRD"
            or value in {"DISCOUNT", "DEEP_DISCOUNT"}
            or extension in {"OVERSOLD", "STRETCHED_DOWN"}
        )
    return observed == "UP" and (
        location == "UPPER_THIRD"
        or value in {"PREMIUM", "DEEP_PREMIUM"}
        or extension in {"OVERBOUGHT", "STRETCHED_UP"}
    )


def _shadow_geometry_pips(method: str, *, spread: float, m5_atr: float) -> tuple[float, float]:
    if method == "TREND_CONTINUATION":
        tp = max(3.0, spread * 4.0, m5_atr * 0.8)
        sl = max(2.5, spread * 4.0, m5_atr * 0.6)
    else:
        tp = max(2.0, spread * 3.0, m5_atr * 0.6)
        sl = max(3.0, spread * 4.0, m5_atr * 0.8)
    return min(tp, 15.0), min(sl, 30.0)


def _entry_experiment_arms(
    *,
    pair: str,
    side: str,
    bid: float,
    ask: float,
    tp_pips: float,
    sl_pips: float,
) -> list[dict[str, Any]]:
    """Freeze passive spread-quartile entries before any outcome is known."""

    if side not in SIDES or ask <= bid or tp_pips <= 0.0 or sl_pips <= 0.0:
        raise ValueError("invalid passive entry experiment geometry")
    pip_size = 1.0 / float(instrument_pip_factor(pair))
    quote_width = ask - bid
    price_digits = 3 if pair.endswith("_JPY") else 5
    price_tick = 10.0 ** (-price_digits)
    arms: list[dict[str, Any]] = []
    for arm_id, fraction in ENTRY_ARM_SPREAD_FRACTIONS:
        raw_entry = (
            bid + quote_width * fraction
            if side == "LONG"
            else ask - quote_width * fraction
        )
        rounded_entry = _price(pair, raw_entry)
        # A rounded passive target must never land on the opposite quote and
        # become marketable.  When the spread is too narrow to represent all
        # quartiles, arms intentionally collapse to the nearest passive tick.
        entry = (
            max(bid, min(rounded_entry, ask - price_tick))
            if side == "LONG"
            else min(ask, max(rounded_entry, bid + price_tick))
        )
        entry = _price(pair, entry)
        effective_fraction = (
            (entry - bid) / quote_width
            if side == "LONG"
            else (ask - entry) / quote_width
        )
        tp = (
            entry + tp_pips * pip_size
            if side == "LONG"
            else entry - tp_pips * pip_size
        )
        sl = (
            entry - sl_pips * pip_size
            if side == "LONG"
            else entry + sl_pips * pip_size
        )
        rounded_tp = _price(pair, tp)
        rounded_sl = _price(pair, sl)
        actual_tp_pips = round(abs(rounded_tp - entry) * instrument_pip_factor(pair), 6)
        actual_sl_pips = round(abs(entry - rounded_sl) * instrument_pip_factor(pair), 6)
        arms.append(
            {
                "arm_id": arm_id,
                "spread_fraction_toward_market": fraction,
                "effective_spread_fraction_toward_market": round(
                    effective_fraction,
                    6,
                ),
                "order_type": "LIMIT",
                "entry_reference": "PASSIVE_SPREAD_FRACTION",
                "execution_truth": "OANDA_S5_BID_ASK_TOUCH",
                "entry": entry,
                "take_profit": rounded_tp,
                "stop_loss": rounded_sl,
                "take_profit_pips": actual_tp_pips,
                "stop_loss_pips": actual_sl_pips,
                "entry_ttl_seconds": ENTRY_TTL_SECONDS,
                "max_hold_seconds": MAX_HOLD_SECONDS,
            }
        )
    return arms


def _ai_pair_state(value: Mapping[str, Any] | None, *, pair: str, now: datetime) -> dict[str, Any]:
    if not _ai_supervision_contract_valid(value or {}):
        return {"mode": "UNSUPERVISED", "reason": "NO_CURRENT_AI_REGIME_OVERRIDE", "expires_at_utc": None}
    pairs = value.get("pairs") if isinstance(value.get("pairs"), Mapping) else {}
    row = pairs.get(pair) if isinstance(pairs.get(pair), Mapping) else {}
    mode = str(row.get("mode") or "UNSUPERVISED").upper()
    expires = _parse_utc(row.get("expires_at_utc"))
    if mode not in ACTIVE_AI_MODES or expires is None:
        return {"mode": "UNSUPERVISED", "reason": "AI_REGIME_OVERRIDE_MISSING_OR_EXPIRED", "expires_at_utc": None}
    if expires <= now:
        expired_for = (now - expires).total_seconds()
        if mode == "STOP" and expired_for <= AI_SUPERVISION_HANDOFF_GRACE_SECONDS:
            return {
                "mode": "STOP",
                "reason": (
                    str(row.get("reason") or "MATERIAL_REGIME_REVIEW")
                    + "; SCHEDULED_SUPERVISOR_HANDOFF_GRACE"
                ),
                "expires_at_utc": expires.isoformat(),
            }
        return {"mode": "UNSUPERVISED", "reason": "AI_REGIME_OVERRIDE_MISSING_OR_EXPIRED", "expires_at_utc": None}
    return {
        "mode": mode,
        "reason": str(row.get("reason") or "MATERIAL_REGIME_REVIEW"),
        "expires_at_utc": expires.isoformat(),
    }


def _charts_by_pair(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        pair: item
        for item in payload.get("charts", []) or []
        if isinstance(item, Mapping)
        for pair in (_pair(item.get("pair")),)
        if pair
    }


def _validated_fast_pair_charts(
    payload: Mapping[str, Any],
    *,
    now_utc: datetime,
) -> Mapping[str, Any]:
    if payload.get("contract") != CURRENT_M1_CONTRACT:
        return payload
    if validate_current_m1_contract(payload, now_utc=now_utc):
        return payload
    return {
        "generated_at_utc": None,
        "charts": [{"pair": pair, "views": []} for pair in DEFAULT_TRADER_PAIRS],
    }


def _validated_slow_pair_charts(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("contract") != SLOW_RETENTION_CONTRACT:
        return payload
    if validate_slow_retention_contract(payload):
        return payload
    return {
        "generated_at_utc": None,
        "charts": [{"pair": pair, "views": []} for pair in DEFAULT_TRADER_PAIRS],
    }


def _merged_views(fast_chart: Mapping[str, Any] | None, slow_chart: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for chart in (slow_chart, fast_chart):
        if not isinstance(chart, Mapping):
            continue
        for view in chart.get("views", []) or []:
            if not isinstance(view, Mapping):
                continue
            timeframe = str(view.get("granularity") or "").upper()
            if timeframe in {tf for values in TIMEFRAME_ROLES.values() for tf in values}:
                out[timeframe] = view
    return out


def _usable_view(view: Mapping[str, Any] | None) -> bool:
    if not isinstance(view, Mapping):
        return False
    market = view.get("market_state")
    candles = view.get("recent_candles")
    return bool(
        isinstance(market, Mapping)
        and market.get("evidence_complete") is True
        and isinstance(candles, list)
        and any(isinstance(item, Mapping) and item.get("complete") is True for item in candles)
    )


def _view_integrity_blocked(view: Mapping[str, Any] | None) -> bool:
    integrity = view.get("candle_integrity") if isinstance(view, Mapping) and isinstance(view.get("candle_integrity"), Mapping) else {}
    return integrity.get("forecast_blocking") is True


def _view_atr_pips(view: Mapping[str, Any] | None) -> float | None:
    indicators = view.get("indicators") if isinstance(view, Mapping) and isinstance(view.get("indicators"), Mapping) else {}
    return _positive_number(indicators.get("atr_pips"))


def _latest_complete_candle_time(view: Mapping[str, Any] | None) -> str | None:
    candles = view.get("recent_candles") if isinstance(view, Mapping) else None
    times = [
        parsed
        for item in candles or []
        if isinstance(item, Mapping) and item.get("complete") is True
        for parsed in (_parse_utc(item.get("t")),)
        if parsed is not None
    ]
    return max(times).isoformat() if times else None


def _latest_complete_candle_close_time(
    view: Mapping[str, Any] | None,
    *,
    timeframe: str,
) -> str | None:
    started_text = _latest_complete_candle_time(view)
    started = _parse_utc(started_text)
    seconds = TIMEFRAME_SECONDS.get(timeframe)
    if started is None or seconds is None:
        return None
    return (started + timedelta(seconds=seconds)).isoformat()


def _view_candle_fresh(
    view: Mapping[str, Any] | None,
    *,
    timeframe: str,
    now: datetime,
) -> bool:
    started_text = _latest_complete_candle_time(view)
    started = _parse_utc(started_text)
    seconds = TIMEFRAME_SECONDS.get(timeframe)
    if started is None or seconds is None:
        return False
    closed = started + timedelta(seconds=seconds)
    age = (now - closed).total_seconds()
    # A producer marking an in-progress candle complete is a hard clock defect.
    # Two completed bars of allowance covers one missed refresh without using
    # an arbitrarily old M1 trigger in the 30-second executor.
    return 0.0 <= age <= float(seconds * 2)


def _spread_pips(pair: str, quote: Mapping[str, Any]) -> float | None:
    bid = _positive_number(quote.get("bid"))
    ask = _positive_number(quote.get("ask"))
    if bid is None or ask is None or ask <= bid:
        return None
    try:
        return (ask - bid) * float(instrument_pip_factor(pair))
    except (KeyError, TypeError, ValueError):
        return None


def _compact_event(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_id": str(item.get("event_id") or ""),
        "event_type": str(item.get("event_type") or "").upper(),
        "pair": _pair(item.get("pair")),
        "severity": str(item.get("severity") or ""),
        "dedupe_key": str(item.get("dedupe_key") or ""),
    }


def _append_signals_once(path: Path, shadow: Mapping[str, Any]) -> int:
    signals = [item for item in shadow.get("signals", []) or [] if isinstance(item, Mapping)]
    if not signals:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        seen: set[str] = set()
        seen_identities: set[tuple[str, str, str, str, str]] = set()
        for line in handle:
            try:
                item = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(item, Mapping) and _signal_digest_valid(item):
                seen.add(str(item["signal_id"]))
                seen_identities.add(_signal_identity(item))
        appended = 0
        handle.seek(0, os.SEEK_END)
        for signal in signals:
            signal_id = str(signal.get("signal_id") or "")
            identity = _signal_identity(signal)
            if (
                not signal_id
                or signal_id in seen
                or identity in seen_identities
                or not _signal_digest_valid(signal)
            ):
                continue
            handle.write(json.dumps(dict(signal), ensure_ascii=False, sort_keys=True) + "\n")
            seen.add(signal_id)
            seen_identities.add(identity)
            appended += 1
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return appended


def _write_report(path: Path, *, contract: Mapping[str, Any], shadow: Mapping[str, Any], appended: int) -> None:
    states = {state: 0 for state in ("GO", "CAUTION", "STOP")}
    for item in contract.get("rows", []) or []:
        if isinstance(item, Mapping) and item.get("state") in states:
            states[str(item["state"])] += 1
    lines = [
        "# Fast Bot Shadow",
        "",
        f"- Generated: `{shadow.get('generated_at_utc')}`",
        "- Entry authority: deterministic bot (AI per-trade approval is not required)",
        "- Timeframes: M1 execution / M5-M15-M30 operating / H1-H4 structure / D anchor",
        f"- Gates: GO={states['GO']} CAUTION={states['CAUTION']} STOP={states['STOP']}",
        f"- Signals this cycle: {len(shadow.get('signals', []) or [])}; newly appended: {appended}",
        f"- AI wake required: `{contract.get('ai_wake_required')}` ({', '.join(contract.get('ai_wake_reasons', []) or []) or 'none'})",
        "- Broker mutation: `false`",
        "- Live permission: `false` until exact S5 bid/ask forward promotion passes",
        "",
        "## Signals",
        "",
    ]
    signals = shadow.get("signals", []) or []
    if not signals:
        lines.append("None.")
    else:
        lines.extend(
            f"- `{item.get('pair')} {item.get('side')} {item.get('method')}` "
            f"LIMIT {item.get('entry')} TP {item.get('take_profit')} SL {item.get('stop_loss')} "
            f"signal `{item.get('signal_id')}`"
            for item in signals
            if isinstance(item, Mapping)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(path, "\n".join(lines) + "\n")


def _read_object(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return {**body, "contract_sha256": _canonical_sha(body)}


def _sealed_contract_valid(value: Mapping[str, Any], contract: str) -> bool:
    if not isinstance(value, Mapping) or value.get("contract") != contract:
        return False
    stored = str(value.get("contract_sha256") or "")
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return bool(stored and stored == _canonical_sha(body))


def _ai_supervision_contract_valid(value: Mapping[str, Any]) -> bool:
    """Accept only sealed supervision that is structurally unable to order."""

    if not _sealed_contract_valid(value, AI_SUPERVISION_CONTRACT):
        return False
    schema_version = value.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != 1:
        return False
    if value.get("ai_role") != "REGIME_REVIEW_AND_PERIODIC_TUNING_ONLY":
        return False
    if value.get("ai_order_authority") != "NONE":
        return False
    if value.get("live_permission") is not False:
        return False
    if value.get("broker_mutation_allowed") is not False:
        return False
    pairs = value.get("pairs")
    if not isinstance(pairs, Mapping):
        return False
    for pair, row in pairs.items():
        if not isinstance(pair, str) or not isinstance(row, Mapping):
            return False
        if set(row) != {"mode", "reason", "expires_at_utc"}:
            return False
        if str(row.get("mode") or "").upper() not in ACTIVE_AI_MODES:
            return False
        if not str(row.get("reason") or "").strip():
            return False
        if _parse_utc(row.get("expires_at_utc")) is None:
            return False
    return True


def _canonical_sha(value: Any) -> str:
    try:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError):
        raw = b"INVALID"
    return hashlib.sha256(raw).hexdigest()


def _sha256_text(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _signal_identity(signal: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(signal.get("pair") or ""),
        str(signal.get("m1_closed_candle_utc") or ""),
        str(signal.get("side") or ""),
        str(signal.get("method") or ""),
        str(signal.get("horizon_lane") or ""),
    )


def _signal_digest_valid(signal: Mapping[str, Any]) -> bool:
    signal_sha = str(signal.get("signal_sha256") or "")
    signal_body = {key: item for key, item in signal.items() if key != "signal_sha256"}
    return _sha256_text(signal_sha) and signal_sha == _canonical_sha(signal_body)


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _age_seconds(now: datetime, then: datetime) -> float:
    return max(0.0, (now - then).total_seconds())


def _timestamp_current(
    now: datetime,
    then: datetime | None,
    *,
    max_age_seconds: float,
    max_future_skew_seconds: float = 0.0,
) -> bool:
    if then is None:
        return False
    age_seconds = (now - then).total_seconds()
    return -max_future_skew_seconds <= age_seconds <= max_age_seconds


def _positive_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0.0 else None


def _round(value: float | None, digits: int) -> float | None:
    return round(value, digits) if value is not None and math.isfinite(value) else None


def _price(pair: str, value: float) -> float:
    return round(value, 3 if pair.endswith("_JPY") else 5)


def _pair(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text if "_" in text else ""
