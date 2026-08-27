"""V29 one-shot paper replay for one preregistered basket consensus release rule.

The sealed V25 500-signal ledger and V28 basket-hold state machine are preserved.
The sole strategy change is an outcome- and cost-independent release: while a
pair is held and has no own signal at a completed timestamp, two or more
simultaneous peer-pair signals must unanimously imply the opposite USD
direction before the sleeve is closed at that timestamp's executable open.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import run_causal_basket_hold_v28 as frozen_v28
import run_causal_min_spread_representative_v26 as frozen_v26
from run_liquid_major_universe_v9 import UNIVERSE
from run_portfolio_episode_netting_v15 import PERIODS


CYCLE_ID = "V29"
EXPERIMENT = "FX_CAUSAL_BASKET_CONSENSUS_RELEASE_V29"
SINGLE_CHANGED_VARIABLE = (
    "one_preregistered_causal_basket_consensus_release_rule_preserving_all_v25_raw_signals_and_fixed_sleeves"
)
MIN_PEER_SIGNALS = 2
TARGET_HOLD_SECONDS = frozen_v28.TARGET_HOLD_SECONDS
HARD_MAX_AGE_SECONDS = frozen_v28.HARD_MAX_AGE_SECONDS
WEIGHT_PER_PAIR = frozen_v28.WEIGHT_PER_PAIR
INITIAL_EQUITY_JPY = frozen_v28.INITIAL_EQUITY_JPY
PARENT_RESULT_SHA256 = frozen_v28.PARENT_RESULT_SHA256
PARENT_LEDGER_SHA256 = frozen_v28.PARENT_LEDGER_SHA256
PARENT_SIGNAL_ID_SET_SHA256 = frozen_v28.PARENT_SIGNAL_ID_SET_SHA256
V28_RESULT_SHA256 = "be6914d6bef4268d39022cb134bbf9ab4fd72206f5b8fe980c05c64c919c343f"
V28_LEDGER_SHA256 = "ce386c8fc9fc1a99fca82cd180f967fcfc26ea75fb170abd157edfa9f1c09ade"
ARMS = tuple(frozen_v28.ARMS)
AUTHORITY = dict(frozen_v28.AUTHORITY)


def canonical_bytes(value: object) -> bytes:
    return frozen_v28.canonical_bytes(value)


def embedded_hash(payload: dict, field: str) -> str:
    return frozen_v28.embedded_hash(payload, field)


def ns(value: str) -> int:
    return frozen_v28.ns(value)


def elapsed_seconds(start: str, end: str) -> float:
    return frozen_v28.elapsed_seconds(start, end)


def implied_usd_direction(pair: str, direction: int) -> int:
    """Return +1 for buying USD and -1 for selling USD."""
    base, quote = pair.split("_")
    if base == "USD":
        return int(direction)
    if quote == "USD":
        return -int(direction)
    raise ValueError(f"pair has no USD node: {pair}")


def consensus_vote(signals: list[dict], held_pair: str) -> dict:
    peers = [row for row in signals if row["pair"] != held_pair]
    votes = [implied_usd_direction(row["pair"], int(row["direction"])) for row in peers]
    vote_sum = sum(votes)
    unanimous = len(votes) >= MIN_PEER_SIGNALS and abs(vote_sum) == len(votes)
    return {
        "peer_count": len(votes),
        "vote_sum": vote_sum,
        "unanimous": unanimous,
        "consensus_usd_direction": (1 if vote_sum > 0 else -1 if vote_sum < 0 else 0),
        "peer_signal_ids": sorted(row["signal_id"] for row in peers),
    }


def build_pair_plan(pair: str, bars: list, parent_rows: list[dict], start: str, end: str) -> dict:
    pair_signals = sorted(frozen_v28._signal_rows(parent_rows, start, end, pair), key=lambda row: (
        row["fill_time"], row["signal_id"]
    ))
    if len({row["fill_time"] for row in pair_signals}) != len(pair_signals):
        raise ValueError(f"multiple same-pair signals at one timestamp: {pair}")
    by_fill = {row["fill_time"]: row for row in pair_signals}
    global_by_fill: dict[str, list[dict]] = {}
    for row in frozen_v28._signal_rows(parent_rows, start, end):
        global_by_fill.setdefault(row["fill_time"], []).append(row)
    period_bars = [bar for bar in bars if start <= bar.time[:10] < end]
    if not period_bars:
        raise ValueError(f"no completed bars for {pair} in {start}..{end}")
    if any(ns(left.time) >= ns(right.time) for left, right in zip(period_bars, period_bars[1:])):
        raise ValueError(f"non-increasing completed bar chronology for {pair}")

    position: dict[str, Any] | None = None
    episodes: list[dict] = []
    signal_events: list[dict] = []
    close_events: list[dict] = []

    def open_position(signal: dict, bar_time: str) -> dict:
        return {
            "entry_time": bar_time,
            "direction": int(signal["direction"]),
            "target_expiry_ns": ns(bar_time) + TARGET_HOLD_SECONDS * 1_000_000_000,
            "source_signal_ids": [signal["signal_id"]],
        }

    def close_position(bar_time: str, exit_at_open: bool, reason: str, audit: dict | None = None) -> None:
        nonlocal position
        if position is None:
            raise ValueError("attempted to close absent inventory")
        age = elapsed_seconds(position["entry_time"], bar_time)
        if age < 0 or age > HARD_MAX_AGE_SECONDS:
            raise ValueError(f"hard inventory age exceeded for {pair}: {age}")
        episode = {
            "pair": pair,
            "entry_time": position["entry_time"],
            "exit_time": bar_time,
            "direction": position["direction"],
            "exit_at_open": exit_at_open,
            "close_reason": reason,
            "inventory_age_seconds": age,
            "source_signal_ids": list(position["source_signal_ids"]),
        }
        event = {
            "event_type": reason,
            "pair": pair,
            "time": bar_time,
            "exit_at_open": exit_at_open,
            "entry_time": position["entry_time"],
            "direction": position["direction"],
        }
        if audit is not None:
            episode["consensus_audit"] = audit
            event["consensus_audit"] = audit
        episodes.append(episode)
        close_events.append(event)
        position = None

    for bar in period_bars:
        signal = by_fill.get(bar.time)
        expired_at_this_open = position is not None and position["target_expiry_ns"] <= ns(bar.time)
        if expired_at_this_open:
            close_position(bar.time, True, "MAX_AGE_CLOSE")

        if position is not None and signal is None:
            audit = consensus_vote(global_by_fill.get(bar.time, []), pair)
            inventory_usd = implied_usd_direction(pair, int(position["direction"]))
            if audit["unanimous"] and inventory_usd * audit["consensus_usd_direction"] < 0:
                audit = {**audit, "inventory_usd_direction": inventory_usd}
                close_position(bar.time, True, "BASKET_CONSENSUS_RELEASE", audit)

        if signal is None:
            continue
        direction = int(signal["direction"])
        if position is None:
            action = "MAX_AGE_CLOSE_THEN_OPEN" if expired_at_this_open else "OPEN_FIXED_ONE_SEVENTH"
            position = open_position(signal, bar.time)
        elif position["direction"] == direction:
            action = "HOLD_EXISTING_NO_ADD_NO_EXPIRY_EXTENSION"
            position["source_signal_ids"].append(signal["signal_id"])
        else:
            close_position(bar.time, True, "OPPOSITE_SIGNAL_CLOSE")
            position = open_position(signal, bar.time)
            action = "REVERSE_FIXED_ONE_SEVENTH"
        signal_events.append({
            "signal_id": signal["signal_id"], "pair": pair, "time": bar.time,
            "direction": direction, "action": action,
        })

    if len(signal_events) != len(pair_signals):
        raise ValueError("signal has no completed executable fill bar")
    if position is not None:
        close_position(period_bars[-1].time, False, "TERMINAL_LIQUIDATION")
    transition_material = {"signal_events": signal_events, "close_events": close_events}
    return {
        "pair": pair, "signals": pair_signals, "period_bars": period_bars,
        "signal_events": signal_events, "close_events": close_events, "episodes": episodes,
        "transition_sha256": hashlib.sha256(canonical_bytes(transition_material)).hexdigest(),
    }


def build_period_plans(corpus: dict[str, list], parent_rows: list[dict], start: str, end: str) -> dict[str, dict]:
    plans = {pair: build_pair_plan(pair, corpus[pair], parent_rows, start, end) for pair in sorted(UNIVERSE)}
    actual = [event["signal_id"] for plan in plans.values() for event in plan["signal_events"]]
    expected = [row["signal_id"] for row in frozen_v28._signal_rows(parent_rows, start, end)]
    if sorted(actual) != sorted(expected) or len(actual) != len(set(actual)):
        raise ValueError("period execution plan does not preserve the RAW signal-id set")
    return plans


def arm_metrics(plans: dict[str, dict], arm: str) -> dict:
    metrics = frozen_v28.arm_metrics(plans, arm)
    pair_active = {}
    pair_directions = {}
    for pair, plan in sorted(plans.items()):
        _marks, pair_active[pair], pair_directions[pair], _returns = frozen_v28._pair_marks(plan, arm)
    common = set.intersection(*(set(values) for values in pair_active.values()))
    max_currency_abs = 0.0
    for stamp in sorted(common):
        exposures: dict[str, float] = {}
        for pair in sorted(UNIVERSE):
            direction = pair_directions[pair][stamp]
            if direction == 0:
                continue
            base, quote = pair.split("_")
            signed = WEIGHT_PER_PAIR * direction
            exposures[base] = exposures.get(base, 0.0) + signed
            exposures[quote] = exposures.get(quote, 0.0) - signed
        max_currency_abs = max([max_currency_abs] + [abs(value) for value in exposures.values()])
    if max_currency_abs > 1.0 + 1e-12:
        raise ValueError("currency absolute exposure cap exceeded")
    metrics["max_currency_abs_exposure_nav"] = max_currency_abs
    metrics["basket_consensus_releases"] = sum(
        event["event_type"] == "BASKET_CONSENSUS_RELEASE"
        for plan in plans.values() for event in plan["close_events"]
    )
    return metrics


def period_payload(corpus: dict[str, list], parent_rows: list[dict], start: str, end: str) -> dict:
    plans = build_period_plans(corpus, parent_rows, start, end)
    arms = {arm: arm_metrics(plans, arm) for arm in ARMS}
    if len({arms[arm]["execution_state_transition_sha256"] for arm in ARMS}) != 1:
        raise ValueError("cost arms do not share identical execution-state transitions")
    signals = frozen_v28._signal_rows(parent_rows, start, end)
    counts = Counter(event["action"] for plan in plans.values() for event in plan["signal_events"])
    releases = arms["RAW_SIGNAL"]["basket_consensus_releases"]
    return {
        "raw_diagnostics": {
            "signals": len(signals), "effective_bet_days": len({row["utc_day"] for row in signals}),
            "processed_signals": sum(counts.values()), "state_action_counts": dict(sorted(counts.items())),
            "basket_consensus_release_count": releases, "raw_definition_changed": False,
            "cost_used_for_state_transition": False,
        },
        **arms,
    }


def load_v28_reference(result_path: Path, ledger_path: Path) -> dict:
    if frozen_v26.sha256_file(result_path) != V28_RESULT_SHA256:
        raise ValueError("sealed V28 result hash mismatch")
    if frozen_v26.sha256_file(ledger_path) != V28_LEDGER_SHA256:
        raise ValueError("sealed V28 ledger hash mismatch")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("result_sha256") != embedded_hash(payload, "result_sha256"):
        raise ValueError("sealed V28 embedded result hash mismatch")
    return payload


def comparisons(corpus: dict[str, list], rows: list[dict], v28: dict, periods: dict) -> dict:
    keys = [
        "gross_edge_bps", "realized_cost_bps", "net_edge_bps", "turnover_nav",
        "break_even_cost_bps", "direction_accuracy", "equity_multiple", "max_drawdown",
        "terminal_inventory_mtm", "max_inventory_age_seconds", "N_eff_days", "N_eff_episodes",
        "max_gross_exposure_nav", "max_margin_requirement_jpy_at_1x",
    ]
    result = {}
    for period_name, (start, end) in PERIODS.items():
        result[period_name] = {}
        for arm in ARMS:
            v25 = frozen_v26.arm_metrics(corpus, rows, rows, arm, start, end)
            old = v28["periods"][period_name][arm]
            new = periods[period_name][arm]
            result[period_name][arm] = {
                key: {
                    "V25": (1.0 if key == "max_gross_exposure_nav" and v25.get(key) is None
                            else INITIAL_EQUITY_JPY if key == "max_margin_requirement_jpy_at_1x" and v25.get(key) is None
                            else v25.get(key)),
                    "V28": old.get(key), "V29": new.get(key),
                    "delta_V29_minus_V28": (new[key] - old[key]),
                } for key in keys
            }
    return result


def automatic_rejection(periods: dict) -> dict:
    months = ("MONTH_2026_05", "MONTH_2026_06")
    normal_pass = all(periods[m]["EXECUTABLE_BASE"]["equity_multiple"] >= 2.0 for m in months)
    adverse_pass = all(periods[m]["ADVERSE_STRESS"]["equity_multiple"] >= 2.0 for m in months)
    walk = periods["WALK_FORWARD"]
    if walk["RAW_SIGNAL"]["equity_multiple"] <= 1.0:
        reason = "BASKET_CONSENSUS_RELEASE_RAW_EDGE_ABSENT"
    elif walk["EXECUTABLE_BASE"]["equity_multiple"] <= 1.0:
        reason = "BASKET_CONSENSUS_RELEASE_COST_DOMINANT"
    elif walk["ADVERSE_STRESS"]["equity_multiple"] <= 1.0:
        reason = "BASKET_CONSENSUS_RELEASE_ADVERSE_COST_FRAGILE"
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
        row["basket_consensus_release_rule"] = {
            "minimum_peer_signals": MIN_PEER_SIGNALS, "unanimity_required": True,
            "cost_inputs": False, "target_hold_seconds": TARGET_HOLD_SECONDS,
            "hard_max_age_seconds": HARD_MAX_AGE_SECONDS,
        }
        rows.append(row)
    rows.sort(key=lambda row: (row["fill_time"], row["signal_id"]))
    if len(rows) != frozen_v26.PARENT_RAW_SIGNALS:
        raise ValueError("unexpected parent ledger size")
    return rows


def run(input_root: Path, parent_ledger: Path, parent_result: Path, v28_result: Path,
        v28_ledger: Path, output_root: Path) -> dict:
    frozen_v28.runtime_v27.install_timestamp_compatibility()
    parent, parent_rows = frozen_v26.load_parent(parent_result, parent_ledger)
    reference_v28 = load_v28_reference(v28_result, v28_ledger)
    corpus, source_audit = frozen_v26.load_corpus(input_root)
    rows = build_execution_ledger(parent_rows, corpus)
    identity = ("signal_id", "pair", "utc_day", "direction", "decision_time", "fill_time", "exit_time")
    if [[row[field] for field in identity] for row in rows] != [[row[field] for field in identity] for row in parent_rows]:
        raise ValueError("V29 changed frozen V25 RAW signal identity")
    if any(set(row["arm_actions"]) != set(ARMS) or len(set(row["arm_actions"].values())) != 1 for row in rows):
        raise ValueError("V29 cost arms differ in execution-state action")
    periods = {name: period_payload(corpus, rows, start, end) for name, (start, end) in PERIODS.items()}
    output_root.mkdir(parents=True, exist_ok=True)
    ledger = output_root / "proposal_ledger_causal_basket_consensus_release_v29.jsonl"
    ledger.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    action_material = [[row["signal_id"], row["execution_action"]] for row in rows]
    payload = {
        "cycle_id": CYCLE_ID, "experiment": EXPERIMENT, "family": "FX_SESSION_CURRENCY_COHERENCE",
        "family_hypotheses": 4, "evidence_class": "opened_development_not_future_holdout",
        "single_changed_variable": SINGLE_CHANGED_VARIABLE, "parent_cycle": "V28", "raw_signal_parent_cycle": "V25",
        "parent_result_sha256": frozen_v26.sha256_file(parent_result),
        "parent_ledger_sha256": frozen_v26.sha256_file(parent_ledger),
        "parent_signal_id_set_sha256": frozen_v26.signal_id_set_hash(parent_rows),
        "v28_result_sha256": frozen_v26.sha256_file(v28_result), "v28_ledger_sha256": frozen_v26.sha256_file(v28_ledger),
        "raw_signal_definition": parent["indicator"], "raw_signals": len(rows),
        "effective_bet_days": len({row["utc_day"] for row in rows}), "cost_suppressed_raw_signals": 0,
        "same_signal_stream_all_cost_arms": True,
        "same_parent_signal_id_set": frozen_v26.signal_id_set_hash(rows) == PARENT_SIGNAL_ID_SET_SHA256,
        "same_parent_decision_timestamps": all(a["decision_time"] == b["decision_time"] for a, b in zip(parent_rows, rows)),
        "same_parent_directions": all(a["direction"] == b["direction"] for a, b in zip(parent_rows, rows)),
        "same_execution_state_transitions_all_cost_arms": True,
        "execution_action_sha256": hashlib.sha256(canonical_bytes(action_material)).hexdigest(),
        "execution_rule": {
            "name": "CAUSAL_BASKET_CONSENSUS_RELEASE", "units": "signed USD-direction votes per simultaneous peer pair",
            "minimum_peer_signals": MIN_PEER_SIGNALS, "unanimity_required": True,
            "release_inequality": "inventory_usd_direction * consensus_usd_direction < 0",
            "own_pair_signal_prevents_consensus_release": True,
            "target_hold_seconds": TARGET_HOLD_SECONDS, "hard_max_age_seconds": HARD_MAX_AGE_SECONDS,
            "same_direction": "HOLD_EXISTING_NO_ADD_NO_EXPIRY_EXTENSION",
            "opposite_direction": "CLOSE_AT_SIGNAL_EXECUTABLE_OPEN_THEN_OPEN_REVERSE_FIXED_SLEEVE",
            "tie_or_insufficient_consensus": "UNCHANGED_V28_DEFAULT",
            "event_precedence": "MAX_AGE_AT_OPEN_THEN_CONSENSUS_IF_NO_OWN_SIGNAL_THEN_V28_SIGNAL_THEN_TERMINAL_MTM",
            "state_inputs": ["completed_source_timestamps", "pair", "signal_id", "direction", "fill_time"],
            "cost_or_outcome_inputs": False,
        },
        "portfolio": {"pair_count": 7, "weight_per_pair": WEIGHT_PER_PAIR, "gross_leverage_cap": 1.0,
                      "rule_max_gross_leverage": 1.0, "currency_abs_exposure_cap": 1.0,
                      "add_to_position": False, "margin_method": "NOTIONAL_AT_1X_CONSERVATIVE_NO_BROKER_MARGIN_ASSUMPTION"},
        "periods": periods, "metric_comparison_vs_v25_and_v28": comparisons(corpus, rows, reference_v28, periods),
        "source_audit": source_audit, "proposal_ledger": str(ledger),
        "proposal_ledger_sha256": frozen_v26.sha256_file(ledger), "automatic_rejection": automatic_rejection(periods),
        "development_admitted": False, "final_admitted": False, "terminal_inventory_mtm_hidden": False,
        "holdout": {"label": "FUTURE_FX_HOLDOUT_AFTER_2026_07_15", "state": "UNOPENED"},
        **AUTHORITY,
        "admission_blockers": ["opened 2026 data are development evidence", "untouched future FX holdout remains unopened",
                               "both normal and adverse full-month 2.0x gates are mandatory", "strategy adoption remains a separate gate"],
    }
    payload["result_sha256"] = embedded_hash(payload, "result_sha256")
    result = output_root / "result_causal_basket_consensus_release_v29.json"
    result.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--parent-ledger", type=Path, required=True)
    parser.add_argument("--parent-result", type=Path, required=True)
    parser.add_argument("--v28-result", type=Path, required=True)
    parser.add_argument("--v28-ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input_root, args.parent_ledger, args.parent_result, args.v28_result, args.v28_ledger, args.output_root)
    print(json.dumps({"cycle_id": result["cycle_id"], "raw_signals": result["raw_signals"],
                      "walk_forward": result["periods"]["WALK_FORWARD"],
                      "automatic_rejection": result["automatic_rejection"],
                      "result_sha256": result["result_sha256"]}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
