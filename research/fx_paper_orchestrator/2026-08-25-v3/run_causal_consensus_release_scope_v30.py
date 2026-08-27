"""V30 one-shot paper replay for one preregistered peer-scope change.

V29's release direction, unanimity, minimum peer count, timestamps, inventory
rules, and all 500 V25 RAW signals are frozen.  The sole strategy change is
peer membership: only simultaneous signals from already-open peer inventory
with the same signed USD exposure as the target sleeve enter consensus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import run_causal_basket_consensus_release_v29 as frozen_v29
import run_causal_min_spread_representative_v26 as frozen_v26
from run_liquid_major_universe_v9 import UNIVERSE
from run_portfolio_episode_netting_v15 import PERIODS


CYCLE_ID = "V30"
EXPERIMENT = "FX_CAUSAL_CONSENSUS_RELEASE_SCOPE_V30"
SINGLE_CHANGED_VARIABLE = (
    "one_preregistered_causal_consensus_release_scope_rule_preserving_all_v25_raw_signals_and_fixed_sleeves"
)
MIN_PEER_SIGNALS = frozen_v29.MIN_PEER_SIGNALS
TARGET_HOLD_SECONDS = frozen_v29.TARGET_HOLD_SECONDS
HARD_MAX_AGE_SECONDS = frozen_v29.HARD_MAX_AGE_SECONDS
WEIGHT_PER_PAIR = frozen_v29.WEIGHT_PER_PAIR
INITIAL_EQUITY_JPY = frozen_v29.INITIAL_EQUITY_JPY
PARENT_RESULT_SHA256 = frozen_v29.PARENT_RESULT_SHA256
PARENT_LEDGER_SHA256 = frozen_v29.PARENT_LEDGER_SHA256
PARENT_SIGNAL_ID_SET_SHA256 = frozen_v29.PARENT_SIGNAL_ID_SET_SHA256
V29_RESULT_SHA256 = "87124a2588bd583127c45026e6b09852040893ba2dc3d6011f19a193f8ad4303"
V29_LEDGER_SHA256 = "63410e258c884f9e29cdfefc59e4f7c1c2583e74d6fe8220c1fe4386c55dab90"
ARMS = tuple(frozen_v29.ARMS)
AUTHORITY = dict(frozen_v29.AUTHORITY)


def canonical_bytes(value: object) -> bytes:
    return frozen_v29.canonical_bytes(value)


def embedded_hash(payload: dict, field: str) -> str:
    return frozen_v29.embedded_hash(payload, field)


def ns(value: str) -> int:
    return frozen_v29.ns(value)


def elapsed_seconds(start: str, end: str) -> float:
    return frozen_v29.elapsed_seconds(start, end)


def implied_usd_direction(pair: str, direction: int) -> int:
    return frozen_v29.implied_usd_direction(pair, direction)


def scoped_peer_signals(
    simultaneous_signals: dict[str, dict],
    target_pair: str,
    position_snapshot: dict[str, dict[str, Any]],
) -> list[dict]:
    """Return the causal active same-signed USD inventory component."""
    target = position_snapshot.get(target_pair)
    if target is None:
        return []
    target_usd = implied_usd_direction(target_pair, int(target["direction"]))
    scoped = []
    for pair, signal in sorted(simultaneous_signals.items()):
        peer = position_snapshot.get(pair)
        if pair == target_pair or peer is None:
            continue
        peer_usd = implied_usd_direction(pair, int(peer["direction"]))
        if peer_usd == target_usd:
            scoped.append(signal)
    return scoped


def build_period_plans(
    corpus: dict[str, list], parent_rows: list[dict], start: str, end: str
) -> dict[str, dict]:
    period_bars = {pair: [bar for bar in corpus[pair] if start <= bar.time[:10] < end]
                   for pair in sorted(UNIVERSE)}
    if any(not bars for bars in period_bars.values()):
        raise ValueError("missing completed period bars")
    for pair, bars in period_bars.items():
        if any(ns(left.time) >= ns(right.time) for left, right in zip(bars, bars[1:])):
            raise ValueError(f"non-increasing completed bar chronology for {pair}")
    bar_times = {pair: {bar.time for bar in bars} for pair, bars in period_bars.items()}
    last_time = {pair: bars[-1].time for pair, bars in period_bars.items()}

    signals_by_time: dict[str, list[dict]] = defaultdict(list)
    period_signals = frozen_v29.frozen_v28._signal_rows(parent_rows, start, end)
    for row in period_signals:
        signals_by_time[row["fill_time"]].append(row)
    for stamp, rows in signals_by_time.items():
        if len({row["pair"] for row in rows}) != len(rows):
            raise ValueError(f"multiple same-pair signals at one timestamp: {stamp}")

    plans = {
        pair: {
            "pair": pair,
            "signals": sorted([row for row in period_signals if row["pair"] == pair],
                              key=lambda row: (row["fill_time"], row["signal_id"])),
            "period_bars": period_bars[pair],
            "signal_events": [],
            "close_events": [],
            "episodes": [],
        }
        for pair in sorted(UNIVERSE)
    }
    positions: dict[str, dict[str, Any] | None] = {pair: None for pair in sorted(UNIVERSE)}

    def open_position(signal: dict, stamp: str) -> dict[str, Any]:
        return {
            "entry_time": stamp,
            "direction": int(signal["direction"]),
            "target_expiry_ns": ns(stamp) + TARGET_HOLD_SECONDS * 1_000_000_000,
            "source_signal_ids": [signal["signal_id"]],
        }

    def close_position(
        pair: str, stamp: str, exit_at_open: bool, reason: str, audit: dict | None = None
    ) -> None:
        position = positions[pair]
        if position is None:
            raise ValueError("attempted to close absent inventory")
        age = elapsed_seconds(position["entry_time"], stamp)
        if age < 0 or age > HARD_MAX_AGE_SECONDS:
            raise ValueError(f"hard inventory age exceeded for {pair}: {age}")
        episode = {
            "pair": pair,
            "entry_time": position["entry_time"],
            "exit_time": stamp,
            "direction": position["direction"],
            "exit_at_open": exit_at_open,
            "close_reason": reason,
            "inventory_age_seconds": age,
            "source_signal_ids": list(position["source_signal_ids"]),
        }
        event = {
            "event_type": reason,
            "pair": pair,
            "time": stamp,
            "exit_at_open": exit_at_open,
            "entry_time": position["entry_time"],
            "direction": position["direction"],
        }
        if audit is not None:
            episode["consensus_audit"] = audit
            event["consensus_audit"] = audit
        plans[pair]["episodes"].append(episode)
        plans[pair]["close_events"].append(event)
        positions[pair] = None

    timeline = sorted(set().union(*(times for times in bar_times.values())), key=ns)
    for stamp in timeline:
        present = {pair for pair in UNIVERSE if stamp in bar_times[pair]}
        expired: set[str] = set()
        for pair in sorted(present):
            position = positions[pair]
            if position is not None and position["target_expiry_ns"] <= ns(stamp):
                close_position(pair, stamp, True, "MAX_AGE_CLOSE")
                expired.add(pair)

        simultaneous = {row["pair"]: row for row in signals_by_time.get(stamp, [])}
        if any(pair not in present for pair in simultaneous):
            raise ValueError("signal has no completed executable fill bar")
        snapshot = {
            pair: json.loads(json.dumps(position, sort_keys=True))
            for pair, position in positions.items() if position is not None
        }
        releases: list[tuple[str, dict]] = []
        for pair in sorted(present):
            position = snapshot.get(pair)
            if position is None or pair in simultaneous:
                continue
            scoped = scoped_peer_signals(simultaneous, pair, snapshot)
            audit = frozen_v29.consensus_vote(scoped, pair)
            inventory_usd = implied_usd_direction(pair, int(position["direction"]))
            if audit["unanimous"] and inventory_usd * audit["consensus_usd_direction"] < 0:
                market_wide_count = len([row for q, row in simultaneous.items() if q != pair])
                releases.append((pair, {
                    **audit,
                    "inventory_usd_direction": inventory_usd,
                    "peer_scope": "ACTIVE_SAME_SIGNED_USD_INVENTORY_SUBGRAPH",
                    "market_wide_peer_count": market_wide_count,
                    "scoped_out_peer_count": market_wide_count - audit["peer_count"],
                }))
        for pair, audit in releases:
            close_position(pair, stamp, True, "BASKET_CONSENSUS_RELEASE", audit)

        for pair, signal in sorted(simultaneous.items()):
            direction = int(signal["direction"])
            position = positions[pair]
            if position is None:
                action = "MAX_AGE_CLOSE_THEN_OPEN" if pair in expired else "OPEN_FIXED_ONE_SEVENTH"
                positions[pair] = open_position(signal, stamp)
            elif int(position["direction"]) == direction:
                action = "HOLD_EXISTING_NO_ADD_NO_EXPIRY_EXTENSION"
                position["source_signal_ids"].append(signal["signal_id"])
            else:
                close_position(pair, stamp, True, "OPPOSITE_SIGNAL_CLOSE")
                positions[pair] = open_position(signal, stamp)
                action = "REVERSE_FIXED_ONE_SEVENTH"
            plans[pair]["signal_events"].append({
                "signal_id": signal["signal_id"], "pair": pair, "time": stamp,
                "direction": direction, "action": action,
            })

        for pair in sorted(present):
            if stamp == last_time[pair] and positions[pair] is not None:
                close_position(pair, stamp, False, "TERMINAL_LIQUIDATION")

    if any(position is not None for position in positions.values()):
        raise ValueError("terminal inventory was not liquidated")
    actual_ids = []
    for pair, plan in plans.items():
        if len(plan["signal_events"]) != len(plan["signals"]):
            raise ValueError(f"signal event mismatch for {pair}")
        material = {"signal_events": plan["signal_events"], "close_events": plan["close_events"]}
        plan["transition_sha256"] = hashlib.sha256(canonical_bytes(material)).hexdigest()
        actual_ids.extend(event["signal_id"] for event in plan["signal_events"])
    expected_ids = [row["signal_id"] for row in period_signals]
    if sorted(actual_ids) != sorted(expected_ids) or len(actual_ids) != len(set(actual_ids)):
        raise ValueError("period execution plan does not preserve the RAW signal-id set")
    return plans


def arm_metrics(plans: dict[str, dict], arm: str) -> dict:
    metrics = frozen_v29.arm_metrics(plans, arm)
    release_events = [event for plan in plans.values() for event in plan["close_events"]
                      if event["event_type"] == "BASKET_CONSENSUS_RELEASE"]
    metrics["scope_release_count"] = len(release_events)
    metrics["scope_members_removed_at_releases"] = sum(
        event["consensus_audit"]["scoped_out_peer_count"] for event in release_events
    )
    return metrics


def period_payload(corpus: dict[str, list], parent_rows: list[dict], start: str, end: str) -> dict:
    plans = build_period_plans(corpus, parent_rows, start, end)
    arms = {arm: arm_metrics(plans, arm) for arm in ARMS}
    if len({arms[arm]["execution_state_transition_sha256"] for arm in ARMS}) != 1:
        raise ValueError("cost arms do not share identical execution-state transitions")
    signals = frozen_v29.frozen_v28._signal_rows(parent_rows, start, end)
    counts = Counter(event["action"] for plan in plans.values() for event in plan["signal_events"])
    return {
        "raw_diagnostics": {
            "signals": len(signals), "effective_bet_days": len({row["utc_day"] for row in signals}),
            "processed_signals": sum(counts.values()), "state_action_counts": dict(sorted(counts.items())),
            "scope_release_count": arms["RAW_SIGNAL"]["scope_release_count"],
            "scope_members_removed_at_releases": arms["RAW_SIGNAL"]["scope_members_removed_at_releases"],
            "raw_definition_changed": False, "cost_used_for_state_transition": False,
        },
        **arms,
    }


def load_v29_reference(result_path: Path, ledger_path: Path) -> dict:
    if frozen_v26.sha256_file(result_path) != V29_RESULT_SHA256:
        raise ValueError("sealed V29 result hash mismatch")
    if frozen_v26.sha256_file(ledger_path) != V29_LEDGER_SHA256:
        raise ValueError("sealed V29 ledger hash mismatch")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("result_sha256") != embedded_hash(payload, "result_sha256"):
        raise ValueError("sealed V29 embedded result hash mismatch")
    return payload


def comparisons(corpus: dict[str, list], rows: list[dict], v29: dict, periods: dict) -> dict:
    keys = [
        "gross_edge_bps", "realized_cost_bps", "net_edge_bps", "turnover_nav",
        "break_even_cost_bps", "direction_accuracy", "equity_multiple", "max_drawdown",
        "terminal_inventory_mtm", "max_inventory_age_seconds", "N_eff_days", "N_eff_episodes",
        "max_gross_exposure_nav", "max_currency_abs_exposure_nav", "max_margin_requirement_jpy_at_1x",
    ]
    result = {}
    for period_name, (start, end) in PERIODS.items():
        result[period_name] = {}
        for arm in ARMS:
            v25 = frozen_v26.arm_metrics(corpus, rows, rows, arm, start, end)
            old = v29["periods"][period_name][arm]
            new = periods[period_name][arm]
            values = {}
            for key in keys:
                v25_value = v25.get(key)
                if key == "max_gross_exposure_nav" and v25_value is None:
                    v25_value = 1.0
                if key == "max_currency_abs_exposure_nav" and v25_value is None:
                    v25_value = 1.0
                if key == "max_margin_requirement_jpy_at_1x" and v25_value is None:
                    v25_value = INITIAL_EQUITY_JPY
                values[key] = {
                    "V25": v25_value, "V29": old.get(key), "V30": new.get(key),
                    "delta_V30_minus_V29": new[key] - old[key],
                }
            values["release_count"] = {
                "V29": old.get("basket_consensus_releases"),
                "V30": new["scope_release_count"],
                "delta_V30_minus_V29": new["scope_release_count"] - old.get("basket_consensus_releases", 0),
            }
            result[period_name][arm] = values
    return result


def automatic_rejection(periods: dict) -> dict:
    months = ("MONTH_2026_05", "MONTH_2026_06")
    normal_pass = all(periods[m]["EXECUTABLE_BASE"]["equity_multiple"] >= 2.0 for m in months)
    adverse_pass = all(periods[m]["ADVERSE_STRESS"]["equity_multiple"] >= 2.0 for m in months)
    walk = periods["WALK_FORWARD"]
    if walk["RAW_SIGNAL"]["equity_multiple"] <= 1.0:
        reason = "CONSENSUS_RELEASE_SCOPE_RAW_EDGE_ABSENT"
    elif walk["EXECUTABLE_BASE"]["equity_multiple"] <= 1.0:
        reason = "CONSENSUS_RELEASE_SCOPE_COST_DOMINANT"
    elif walk["ADVERSE_STRESS"]["equity_multiple"] <= 1.0:
        reason = "CONSENSUS_RELEASE_SCOPE_ADVERSE_COST_FRAGILE"
    else:
        reason = "MONTHLY_2X_AND_UNOPENED_HOLDOUT_NOT_MET"
    return {
        "rejected": not (normal_pass and adverse_pass and False), "reason_code": reason,
        "normal_full_month_2x_pass": normal_pass, "adverse_full_month_2x_pass": adverse_pass,
        "holdout_reproduced": False, "adoption_authorized": False,
    }


def build_execution_ledger(parent_rows: list[dict], corpus: dict[str, list]) -> list[dict]:
    plans = build_period_plans(corpus, parent_rows, min(row["fill_time"][:10] for row in parent_rows), "9999-12-31")
    actions = {event["signal_id"]: event["action"] for plan in plans.values() for event in plan["signal_events"]}
    rows = []
    for parent in parent_rows:
        row = json.loads(json.dumps(parent, sort_keys=True, allow_nan=False))
        row["execution_selected"] = True
        row["execution_action"] = actions[parent["signal_id"]]
        row["arm_actions"] = {arm: row["execution_action"] for arm in ARMS}
        row["consensus_release_peer_scope_rule"] = {
            "name": "ACTIVE_SAME_SIGNED_USD_INVENTORY_SUBGRAPH",
            "minimum_peer_signals": MIN_PEER_SIGNALS, "unanimity_required": True,
            "same_timestamp_required": True, "self_pair_excluded": True,
            "cost_inputs": False, "target_hold_seconds": TARGET_HOLD_SECONDS,
            "hard_max_age_seconds": HARD_MAX_AGE_SECONDS,
        }
        rows.append(row)
    rows.sort(key=lambda row: (row["fill_time"], row["signal_id"]))
    if len(rows) != frozen_v26.PARENT_RAW_SIGNALS:
        raise ValueError("unexpected parent ledger size")
    return rows


def run(input_root: Path, parent_ledger: Path, parent_result: Path, v29_result: Path,
        v29_ledger: Path, output_root: Path) -> dict:
    frozen_v29.frozen_v28.runtime_v27.install_timestamp_compatibility()
    parent, parent_rows = frozen_v26.load_parent(parent_result, parent_ledger)
    reference_v29 = load_v29_reference(v29_result, v29_ledger)
    corpus, source_audit = frozen_v26.load_corpus(input_root)
    rows = build_execution_ledger(parent_rows, corpus)
    identity = ("signal_id", "pair", "utc_day", "direction", "decision_time", "fill_time", "exit_time")
    if [[row[field] for field in identity] for row in rows] != [[row[field] for field in identity] for row in parent_rows]:
        raise ValueError("V30 changed frozen V25 RAW signal identity")
    if any(set(row["arm_actions"]) != set(ARMS) or len(set(row["arm_actions"].values())) != 1 for row in rows):
        raise ValueError("V30 cost arms differ in execution-state action")
    periods = {name: period_payload(corpus, rows, start, end) for name, (start, end) in PERIODS.items()}
    output_root.mkdir(parents=True, exist_ok=True)
    ledger = output_root / "proposal_ledger_causal_consensus_release_scope_v30.jsonl"
    ledger.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    action_material = [[row["signal_id"], row["execution_action"]] for row in rows]
    payload = {
        "cycle_id": CYCLE_ID, "experiment": EXPERIMENT, "family": "FX_SESSION_CURRENCY_COHERENCE",
        "family_hypotheses": 5, "evidence_class": "opened_development_not_future_holdout",
        "single_changed_variable": SINGLE_CHANGED_VARIABLE, "parent_cycle": "V29", "raw_signal_parent_cycle": "V25",
        "parent_result_sha256": frozen_v26.sha256_file(parent_result),
        "parent_ledger_sha256": frozen_v26.sha256_file(parent_ledger),
        "parent_signal_id_set_sha256": frozen_v26.signal_id_set_hash(parent_rows),
        "v29_result_sha256": frozen_v26.sha256_file(v29_result), "v29_ledger_sha256": frozen_v26.sha256_file(v29_ledger),
        "raw_signal_definition": parent["indicator"], "raw_signals": len(rows),
        "effective_bet_days": len({row["utc_day"] for row in rows}), "cost_suppressed_raw_signals": 0,
        "same_signal_stream_all_cost_arms": True,
        "same_parent_signal_id_set": frozen_v26.signal_id_set_hash(rows) == PARENT_SIGNAL_ID_SET_SHA256,
        "same_parent_decision_timestamps": all(a["decision_time"] == b["decision_time"] for a, b in zip(parent_rows, rows)),
        "same_parent_directions": all(a["direction"] == b["direction"] for a, b in zip(parent_rows, rows)),
        "same_execution_state_transitions_all_cost_arms": True,
        "execution_action_sha256": hashlib.sha256(canonical_bytes(action_material)).hexdigest(),
        "execution_rule": {
            "name": "CAUSAL_CONSENSUS_RELEASE_WITH_ACTIVE_SAME_SIGNED_USD_SCOPE",
            "only_changed_field_from_v29": "peer_membership_scope",
            "peer_scope": "ACTIVE_SAME_SIGNED_USD_INVENTORY_SUBGRAPH",
            "minimum_peer_signals": MIN_PEER_SIGNALS, "unanimity_required": True,
            "same_timestamp_required": True, "self_pair_excluded": True,
            "direction_formula_changed_from_v29": False,
            "release_inequality": "inventory_usd_direction * consensus_usd_direction < 0",
            "own_pair_signal_prevents_consensus_release": True,
            "target_hold_seconds": TARGET_HOLD_SECONDS, "hard_max_age_seconds": HARD_MAX_AGE_SECONDS,
            "same_direction": "HOLD_EXISTING_NO_ADD_NO_EXPIRY_EXTENSION",
            "opposite_direction": "CLOSE_AT_SIGNAL_EXECUTABLE_OPEN_THEN_OPEN_REVERSE_FIXED_SLEEVE",
            "tie_or_insufficient_consensus": "UNCHANGED_V29_DEFAULT",
            "event_precedence": "MAX_AGE_AT_OPEN_THEN_IMMUTABLE_SCOPE_SNAPSHOT_THEN_SIMULTANEOUS_RELEASE_THEN_V29_SIGNAL_THEN_TERMINAL_MTM",
            "state_inputs": ["completed_source_timestamps", "pair", "signal_id", "direction", "fill_time", "predecision_inventory"],
            "cost_or_outcome_inputs": False,
        },
        "cost_provenance": {
            "source_price": "ACTUAL_COMPLETED_M5_BID_ASK",
            "base": "FROZEN_V7_BID_ASK_SLIPPAGE_COMMISSION_FINANCING",
            "adverse": "FROZEN_V7_ADVERSE_BID_ASK_SLIPPAGE_COMMISSION_FINANCING",
            "latency": "FROZEN_V25_DECISION_TIME_TO_FILL_TIME_CHRONOLOGY",
            "applied_after_transition_plan": True,
        },
        "portfolio": {"pair_count": 7, "weight_per_pair": WEIGHT_PER_PAIR, "gross_leverage_cap": 1.0,
                      "rule_max_gross_leverage": 1.0, "currency_abs_exposure_cap": 1.0,
                      "add_to_position": False, "margin_method": "NOTIONAL_AT_1X_CONSERVATIVE_NO_BROKER_MARGIN_ASSUMPTION"},
        "periods": periods, "metric_comparison_vs_v25_and_v29": comparisons(corpus, rows, reference_v29, periods),
        "source_audit": source_audit, "proposal_ledger": str(ledger),
        "proposal_ledger_sha256": frozen_v26.sha256_file(ledger), "automatic_rejection": automatic_rejection(periods),
        "development_admitted": False, "final_admitted": False, "terminal_inventory_mtm_hidden": False,
        "holdout": {"label": "FUTURE_FX_HOLDOUT_AFTER_2026_07_15", "state": "UNOPENED"},
        **AUTHORITY,
        "admission_blockers": ["opened 2026 data are development evidence", "untouched future FX holdout remains unopened",
                               "both normal and adverse full-month 2.0x gates are mandatory", "strategy adoption remains a separate gate"],
    }
    payload["result_sha256"] = embedded_hash(payload, "result_sha256")
    result = output_root / "result_causal_consensus_release_scope_v30.json"
    result.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--parent-ledger", type=Path, required=True)
    parser.add_argument("--parent-result", type=Path, required=True)
    parser.add_argument("--v29-result", type=Path, required=True)
    parser.add_argument("--v29-ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input_root, args.parent_ledger, args.parent_result, args.v29_result, args.v29_ledger, args.output_root)
    print(json.dumps({"cycle_id": result["cycle_id"], "raw_signals": result["raw_signals"],
                      "walk_forward": result["periods"]["WALK_FORWARD"],
                      "automatic_rejection": result["automatic_rejection"],
                      "result_sha256": result["result_sha256"]}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
