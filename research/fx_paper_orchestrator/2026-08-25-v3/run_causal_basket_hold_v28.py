"""V28 one-shot paper replay for one preregistered causal basket-hold rule.

All 500 V25 RAW signals, ids, decision timestamps, fill timestamps, directions,
and fixed 1/7 pair sleeves remain unchanged.  Only the execution-state rule is
changed: same-pair/same-direction signals aggregate into the current holding
without adding units or extending expiry; opposite signals reverse at the
current executable open; nominal expiry is 48 hours and liquidation occurs at
the first completed executable bar thereafter, with a 96-hour hard observed
age cap.  Every period ends with explicit terminal liquidation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import run_causal_min_spread_representative_v26 as frozen_v26
import run_causal_min_spread_representative_v27 as runtime_v27
from run_liquid_major_universe_v9 import UNIVERSE
from run_portfolio_episode_netting_v15 import PERIODS


CYCLE_ID = "V28"
EXPERIMENT = "FX_CAUSAL_BASKET_HOLD_V28"
TARGET_HOLD_SECONDS = 48 * 60 * 60
HARD_MAX_AGE_SECONDS = 96 * 60 * 60
WEIGHT_PER_PAIR = 1 / 7
INITIAL_EQUITY_JPY = 200000
PARENT_RESULT_SHA256 = frozen_v26.PARENT_RESULT_SHA256
PARENT_LEDGER_SHA256 = frozen_v26.PARENT_LEDGER_SHA256
PARENT_SIGNAL_ID_SET_SHA256 = frozen_v26.PARENT_SIGNAL_ID_SET_SHA256
V27_RESULT_SHA256 = "b6b7679143690f79f8d9e9662db40fe7c9062ad1f7c7bb8c43828cdd9de99b87"
V27_LEDGER_SHA256 = "8cabfb97eb8de5b071edc60ba45a351cf57e385aeb5880a920a3124313e3979d"
ARMS = tuple(frozen_v26.ARMS)
AUTHORITY = dict(frozen_v26.AUTHORITY)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def embedded_hash(payload: dict, field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return hashlib.sha256(canonical_bytes(unsigned)).hexdigest()


def ns(value: str) -> int:
    return runtime_v27.parse_utc_nanoseconds(value).value


def elapsed_seconds(start: str, end: str) -> float:
    return (ns(end) - ns(start)) / 1_000_000_000


def load_v27_reference(result_path: Path, ledger_path: Path) -> dict:
    if frozen_v26.sha256_file(result_path) != V27_RESULT_SHA256:
        raise ValueError("sealed V27 result hash mismatch")
    if frozen_v26.sha256_file(ledger_path) != V27_LEDGER_SHA256:
        raise ValueError("sealed V27 ledger hash mismatch")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("result_sha256") != embedded_hash(payload, "result_sha256"):
        raise ValueError("sealed V27 embedded result hash mismatch")
    return payload


def _signal_rows(parent_rows: list[dict], start: str, end: str, pair: str | None = None) -> list[dict]:
    return [
        row for row in parent_rows
        if start <= row["fill_time"][:10] < end and (pair is None or row["pair"] == pair)
    ]


def build_pair_plan(pair: str, bars: list, parent_rows: list[dict], start: str, end: str) -> dict:
    """Freeze price-independent execution transitions from completed timestamps and directions."""
    signals = sorted(_signal_rows(parent_rows, start, end, pair), key=lambda row: (
        row["fill_time"], row["signal_id"]
    ))
    if len({row["fill_time"] for row in signals}) != len(signals):
        raise ValueError(f"multiple same-pair signals at one timestamp: {pair}")
    by_fill = {row["fill_time"]: row for row in signals}
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

    def close_position(bar_time: str, exit_at_open: bool, reason: str) -> None:
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
        episodes.append(episode)
        close_events.append({
            "event_type": reason,
            "pair": pair,
            "time": bar_time,
            "exit_at_open": exit_at_open,
            "entry_time": position["entry_time"],
            "direction": position["direction"],
        })
        position = None

    for bar in period_bars:
        signal = by_fill.get(bar.time)
        expired_at_this_open = position is not None and position["target_expiry_ns"] <= ns(bar.time)
        if expired_at_this_open:
            close_position(bar.time, True, "MAX_AGE_CLOSE")
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
            "signal_id": signal["signal_id"],
            "pair": pair,
            "time": bar.time,
            "direction": direction,
            "action": action,
        })

    if len(signal_events) != len(signals):
        missing = sorted(set(row["signal_id"] for row in signals) - set(
            event["signal_id"] for event in signal_events
        ))
        raise ValueError(f"signal has no completed executable fill bar: {missing[:3]}")
    if position is not None:
        close_position(period_bars[-1].time, False, "TERMINAL_LIQUIDATION")
    if any(episode["inventory_age_seconds"] > HARD_MAX_AGE_SECONDS for episode in episodes):
        raise ValueError("finite hard max-age failed")
    transition_material = {"signal_events": signal_events, "close_events": close_events}
    return {
        "pair": pair,
        "signals": signals,
        "period_bars": period_bars,
        "signal_events": signal_events,
        "close_events": close_events,
        "episodes": episodes,
        "transition_sha256": hashlib.sha256(canonical_bytes(transition_material)).hexdigest(),
    }


def build_period_plans(corpus: dict[str, list], parent_rows: list[dict], start: str, end: str) -> dict[str, dict]:
    plans = {
        pair: build_pair_plan(pair, corpus[pair], parent_rows, start, end)
        for pair in sorted(UNIVERSE)
    }
    all_signal_ids = [
        event["signal_id"] for plan in plans.values() for event in plan["signal_events"]
    ]
    expected = [row["signal_id"] for row in _signal_rows(parent_rows, start, end)]
    if sorted(all_signal_ids) != sorted(expected) or len(all_signal_ids) != len(set(all_signal_ids)):
        raise ValueError("period execution plan does not preserve the RAW signal-id set")
    return plans


def _pair_marks(plan: dict, arm: str) -> tuple[dict[str, float], dict[str, int], dict[str, int], list[float]]:
    bars = plan["period_bars"]
    by_time = {bar.time: bar for bar in bars}
    episodes = plan["episodes"]
    entries = {episode["entry_time"]: episode for episode in episodes}
    exit_open = {episode["exit_time"]: episode for episode in episodes if episode["exit_at_open"]}
    exit_close = {episode["exit_time"]: episode for episode in episodes if not episode["exit_at_open"]}
    wealth = 1.0
    current: dict | None = None
    marks: dict[str, float] = {}
    active: dict[str, int] = {}
    directions: dict[str, int] = {}
    returns: list[float] = []
    for bar in bars:
        episode = exit_open.get(bar.time)
        if episode is not None:
            if current is None or current["entry_time"] != episode["entry_time"]:
                raise ValueError("open-price close does not match current inventory")
            value = frozen_v26.roundtrip_return(
                by_time[episode["entry_time"]], bar, episode["direction"], arm, True
            )
            returns.append(value)
            wealth *= max(1.0 + value, 1e-12)
            current = None
        episode = entries.get(bar.time)
        if episode is not None:
            if current is not None:
                raise ValueError("execution plan overlaps same-pair inventory")
            current = episode
        episode = exit_close.get(bar.time)
        if episode is not None:
            if current is None or current["entry_time"] != episode["entry_time"]:
                raise ValueError("close-price liquidation does not match current inventory")
            value = frozen_v26.roundtrip_return(
                by_time[episode["entry_time"]], bar, episode["direction"], arm, False
            )
            returns.append(value)
            wealth *= max(1.0 + value, 1e-12)
            current = None
        if current is None:
            marks[bar.time] = wealth
            active[bar.time] = 0
            directions[bar.time] = 0
        else:
            mtm = frozen_v26.roundtrip_return(
                by_time[current["entry_time"]], bar, current["direction"], arm, False
            )
            marks[bar.time] = wealth * max(1.0 + mtm, 1e-12)
            active[bar.time] = 1
            directions[bar.time] = int(current["direction"])
    if current is not None:
        raise ValueError("terminal inventory was not liquidated")
    if len(returns) != len(episodes):
        raise ValueError("not every inventory episode was realized")
    return marks, active, directions, returns


def _currency_exposures(active_directions: dict[str, dict[str, int]], stamp: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for pair in sorted(UNIVERSE):
        direction = active_directions[pair][stamp]
        if direction == 0:
            continue
        base, quote = pair.split("_")
        signed = WEIGHT_PER_PAIR * direction
        values[base] = values.get(base, 0.0) + signed
        values[quote] = values.get(quote, 0.0) - signed
    return values


def arm_metrics(plans: dict[str, dict], arm: str) -> dict:
    pair_marks: dict[str, dict[str, float]] = {}
    pair_active: dict[str, dict[str, int]] = {}
    pair_directions: dict[str, dict[str, int]] = {}
    pair_returns: dict[str, list[float]] = {}
    for pair, plan in sorted(plans.items()):
        pair_marks[pair], pair_active[pair], pair_directions[pair], pair_returns[pair] = _pair_marks(plan, arm)
    common = set.intersection(*(set(values) for values in pair_marks.values()))
    if not common:
        raise ValueError("pair mark timelines have no common timestamps")
    stamps = sorted(common)
    equity_path = [statistics.fmean(pair_marks[pair][stamp] for pair in sorted(UNIVERSE)) for stamp in stamps]
    peak = equity_path[0]
    max_drawdown = 0.0
    for value in equity_path:
        peak = max(peak, value)
        max_drawdown = min(max_drawdown, value / peak - 1.0)
    max_gross = max(sum(pair_active[pair][stamp] for pair in UNIVERSE) * WEIGHT_PER_PAIR for stamp in stamps)
    max_currency_abs = max(
        [0.0] + [max([0.0] + [abs(value) for value in _currency_exposures(pair_directions, stamp).values()])
                  for stamp in stamps]
    )
    all_episodes = [episode for plan in plans.values() for episode in plan["episodes"]]
    all_net = [value for pair in sorted(UNIVERSE) for value in pair_returns[pair]]
    all_raw = []
    for pair in sorted(UNIVERSE):
        raw_values = _pair_marks(plans[pair], "RAW_SIGNAL")[3]
        all_raw.extend(raw_values)
    signals = [event for plan in plans.values() for event in plan["signal_events"]]
    action_counts = Counter(event["action"] for event in signals)
    transition_hash = hashlib.sha256(canonical_bytes({
        pair: plans[pair]["transition_sha256"] for pair in sorted(plans)
    })).hexdigest()
    opens = len(all_episodes)
    closes = len(all_episodes)
    gross = statistics.fmean(all_raw) * 10000.0 if all_raw else None
    net = statistics.fmean(all_net) * 10000.0 if all_net else None
    return {
        "source_signals": len(signals),
        "processed_raw_signals": len(signals),
        "executed_signals": opens,
        "cash_signals": 0,
        "position_opens": opens,
        "position_closes": closes,
        "aggregated_hold_signals": action_counts["HOLD_EXISTING_NO_ADD_NO_EXPIRY_EXTENSION"],
        "reversals": action_counts["REVERSE_FIXED_ONE_SEVENTH"],
        "max_age_reopens": action_counts["MAX_AGE_CLOSE_THEN_OPEN"],
        "terminal_closes": sum(
            event["event_type"] == "TERMINAL_LIQUIDATION"
            for plan in plans.values() for event in plan["close_events"]
        ),
        "terminal_open_inventory": 0,
        "terminal_inventory_mtm": 0.0,
        "gross_edge_bps": gross,
        "realized_cost_bps": (statistics.fmean(g - n for g, n in zip(all_raw, all_net)) * 10000.0
                              if all_net else None),
        "net_edge_bps": net,
        "break_even_cost_bps": gross,
        "direction_accuracy": (sum(value > 0 for value in all_raw) / len(all_raw) if all_raw else None),
        "equity_multiple": equity_path[-1],
        "max_drawdown": max_drawdown,
        "turnover_nav": (opens + closes) * WEIGHT_PER_PAIR,
        "max_inventory_age_seconds": max(
            [0.0] + [episode["inventory_age_seconds"] for episode in all_episodes]
        ),
        "N_eff_days": len({row["utc_day"] for plan in plans.values() for row in plan["signals"]}),
        "N_eff_episodes": len(all_episodes),
        "max_gross_exposure_nav": max_gross,
        "max_currency_abs_exposure_nav": max_currency_abs,
        "max_margin_requirement_jpy_at_1x": INITIAL_EQUITY_JPY * max_gross,
        "initial_equity_jpy": INITIAL_EQUITY_JPY,
        "ending_equity_jpy": INITIAL_EQUITY_JPY * equity_path[-1],
        "execution_state_transition_sha256": transition_hash,
        "pair_audit": {
            pair: {
                "source_signals": len(plans[pair]["signal_events"]),
                "episodes": len(plans[pair]["episodes"]),
                "transition_sha256": plans[pair]["transition_sha256"],
            }
            for pair in sorted(plans)
        },
    }


def period_payload(corpus: dict[str, list], parent_rows: list[dict], start: str, end: str) -> dict:
    plans = build_period_plans(corpus, parent_rows, start, end)
    arm_payload = {arm: arm_metrics(plans, arm) for arm in ARMS}
    hashes = {arm_payload[arm]["execution_state_transition_sha256"] for arm in ARMS}
    if len(hashes) != 1:
        raise ValueError("cost arms do not share identical execution-state transitions")
    signals = _signal_rows(parent_rows, start, end)
    counts = Counter(
        event["action"] for plan in plans.values() for event in plan["signal_events"]
    )
    return {
        "raw_diagnostics": {
            "signals": len(signals),
            "effective_bet_days": len({row["utc_day"] for row in signals}),
            "processed_signals": sum(counts.values()),
            "state_action_counts": dict(sorted(counts.items())),
            "raw_definition_changed": False,
            "cost_used_for_state_transition": False,
        },
        **arm_payload,
    }


def _metric_comparison(v25: dict, v27: dict, v28: dict) -> dict:
    keys = [
        "gross_edge_bps", "realized_cost_bps", "net_edge_bps", "turnover_nav",
        "break_even_cost_bps", "direction_accuracy", "equity_multiple", "max_drawdown",
        "terminal_inventory_mtm", "max_inventory_age_seconds", "N_eff_days",
        "max_gross_exposure_nav", "max_margin_requirement_jpy_at_1x",
    ]
    result = {}
    for key in keys:
        v25_value = v25.get(key)
        if key == "max_gross_exposure_nav" and v25_value is None:
            v25_value = 1.0
        if key == "max_margin_requirement_jpy_at_1x" and v25_value is None:
            v25_value = INITIAL_EQUITY_JPY
        v27_value = v27.get(key)
        result[key] = {
            "V25": v25_value,
            "V27": v27_value,
            "V28": v28[key],
            "delta_V28_minus_V25": v28[key] - v25_value,
            "delta_V28_minus_V27": v28[key] - v27_value,
        }
    return result


def comparisons(corpus: dict[str, list], parent_rows: list[dict], v27: dict, periods: dict) -> dict:
    result = {}
    for period_name, (start, end) in PERIODS.items():
        result[period_name] = {}
        for arm in ARMS:
            v25 = frozen_v26.arm_metrics(corpus, parent_rows, parent_rows, arm, start, end)
            if "max_gross_exposure_nav" not in v27["periods"][period_name][arm]:
                raise ValueError("sealed V27 reference lacks margin metrics")
            result[period_name][arm] = _metric_comparison(
                v25, v27["periods"][period_name][arm], periods[period_name][arm]
            )
    return result


def automatic_rejection(periods: dict) -> dict:
    months = ("MONTH_2026_05", "MONTH_2026_06")
    normal_pass = all(periods[month]["EXECUTABLE_BASE"]["equity_multiple"] >= 2.0 for month in months)
    adverse_pass = all(periods[month]["ADVERSE_STRESS"]["equity_multiple"] >= 2.0 for month in months)
    walk = periods["WALK_FORWARD"]
    if walk["RAW_SIGNAL"]["equity_multiple"] <= 1.0:
        reason = "BASKET_HOLD_RAW_EDGE_ABSENT"
    elif walk["EXECUTABLE_BASE"]["equity_multiple"] <= 1.0:
        reason = "BASKET_HOLD_RAW_EDGE_COST_DOMINANT"
    elif walk["ADVERSE_STRESS"]["equity_multiple"] <= 1.0:
        reason = "BASKET_HOLD_ADVERSE_COST_FRAGILE"
    else:
        reason = "MONTHLY_2X_AND_UNOPENED_HOLDOUT_NOT_MET"
    return {
        "rejected": not (normal_pass and adverse_pass and False),
        "reason_code": reason,
        "normal_full_month_2x_pass": normal_pass,
        "adverse_full_month_2x_pass": adverse_pass,
        "holdout_reproduced": False,
        "adoption_authorized": False,
    }


def build_execution_ledger(parent_rows: list[dict], corpus: dict[str, list]) -> list[dict]:
    """Annotate the unchanged 500-signal ledger with one global state narrative."""
    start = min(row["fill_time"][:10] for row in parent_rows)
    last_date = max(row["fill_time"][:10] for row in parent_rows)
    end = "9999-12-31"
    plans = build_period_plans(corpus, parent_rows, start, end)
    actions = {
        event["signal_id"]: event["action"]
        for plan in plans.values() for event in plan["signal_events"]
    }
    rows = []
    for parent in parent_rows:
        row = json.loads(json.dumps(parent, sort_keys=True, allow_nan=False))
        row["execution_selected"] = True
        row["execution_action"] = actions[parent["signal_id"]]
        row["arm_actions"] = {arm: row["execution_action"] for arm in ARMS}
        row["basket_hold_rule"] = {
            "target_hold_seconds": TARGET_HOLD_SECONDS,
            "hard_max_age_seconds": HARD_MAX_AGE_SECONDS,
            "same_direction_add_units": 0,
            "same_direction_expiry_extension_seconds": 0,
        }
        rows.append(row)
    rows.sort(key=lambda row: (row["fill_time"], row["signal_id"]))
    if len(rows) != frozen_v26.PARENT_RAW_SIGNALS or last_date >= "9999-12-31":
        raise ValueError("unexpected parent ledger boundary")
    return rows


def run(input_root: Path, parent_ledger: Path, parent_result: Path,
        v27_result: Path, v27_ledger: Path, output_root: Path) -> dict:
    runtime_v27.install_timestamp_compatibility()
    parent, parent_rows = frozen_v26.load_parent(parent_result, parent_ledger)
    reference_v27 = load_v27_reference(v27_result, v27_ledger)
    corpus, source_audit = frozen_v26.load_corpus(input_root)
    rows = build_execution_ledger(parent_rows, corpus)
    identity = ("signal_id", "pair", "utc_day", "direction", "decision_time", "fill_time", "exit_time")
    if [[row[field] for field in identity] for row in rows] != [
            [row[field] for field in identity] for row in parent_rows]:
        raise ValueError("V28 changed frozen V25 RAW signal identity")
    if any(set(row["arm_actions"]) != set(ARMS) or len(set(row["arm_actions"].values())) != 1
           for row in rows):
        raise ValueError("V28 cost arms differ in execution-state action")
    periods = {
        name: period_payload(corpus, rows, start, end)
        for name, (start, end) in PERIODS.items()
    }
    output_root.mkdir(parents=True, exist_ok=True)
    ledger = output_root / "proposal_ledger_causal_basket_hold_v28.jsonl"
    ledger.write_text("".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
    ), encoding="utf-8")
    action_material = [[row["signal_id"], row["execution_action"]] for row in rows]
    payload = {
        "cycle_id": CYCLE_ID,
        "experiment": EXPERIMENT,
        "family": "FX_SESSION_CURRENCY_COHERENCE",
        "family_hypotheses": 3,
        "evidence_class": "opened_development_not_future_holdout",
        "single_changed_variable": "one_preregistered_causal_basket_hold_rule_that_preserves_all_v25_raw_signals_and_fixed_sleeves",
        "parent_cycle": "V27",
        "raw_signal_parent_cycle": "V25",
        "parent_result_sha256": frozen_v26.sha256_file(parent_result),
        "parent_ledger_sha256": frozen_v26.sha256_file(parent_ledger),
        "parent_signal_id_set_sha256": frozen_v26.signal_id_set_hash(parent_rows),
        "v27_result_sha256": frozen_v26.sha256_file(v27_result),
        "v27_ledger_sha256": frozen_v26.sha256_file(v27_ledger),
        "raw_signal_definition": parent["indicator"],
        "raw_signals": len(rows),
        "effective_bet_days": len({row["utc_day"] for row in rows}),
        "cost_suppressed_raw_signals": 0,
        "same_signal_stream_all_cost_arms": True,
        "same_parent_signal_id_set": frozen_v26.signal_id_set_hash(rows) == PARENT_SIGNAL_ID_SET_SHA256,
        "same_parent_decision_timestamps": all(
            left["decision_time"] == right["decision_time"] for left, right in zip(parent_rows, rows)
        ),
        "same_parent_directions": all(
            left["direction"] == right["direction"] for left, right in zip(parent_rows, rows)
        ),
        "same_execution_state_transitions_all_cost_arms": True,
        "execution_action_sha256": hashlib.sha256(canonical_bytes(action_material)).hexdigest(),
        "execution_rule": {
            "name": "CAUSAL_BASKET_HOLD_NO_ADD",
            "units": "one fixed 1/7 NAV sleeve per pair",
            "target_hold_seconds": TARGET_HOLD_SECONDS,
            "hard_max_age_seconds": HARD_MAX_AGE_SECONDS,
            "same_direction": "HOLD_EXISTING_NO_ADD_NO_EXPIRY_EXTENSION",
            "opposite_direction": "CLOSE_AT_SIGNAL_EXECUTABLE_OPEN_THEN_OPEN_REVERSE_FIXED_SLEEVE",
            "target_expiry": "FIRST_COMPLETED_EXECUTABLE_OPEN_AT_OR_AFTER_48H_TARGET",
            "terminal": "LAST_COMPLETED_BAR_CLOSE_OF_EVALUATION_PERIOD",
            "event_precedence": "MAX_AGE_CLOSE_AT_OPEN_THEN_SIGNAL_AT_SAME_OPEN_THEN_CLOSE_MTM",
            "state_inputs": ["completed_source_timestamps", "pair", "signal_id", "direction", "fill_time"],
            "cost_or_outcome_inputs": False,
        },
        "portfolio": {
            "pair_count": 7,
            "weight_per_pair": WEIGHT_PER_PAIR,
            "gross_leverage_cap": 1.0,
            "rule_max_gross_leverage": 1.0,
            "currency_abs_exposure_cap": 1.0,
            "add_to_position": False,
            "margin_method": "NOTIONAL_AT_1X_CONSERVATIVE_NO_BROKER_MARGIN_ASSUMPTION",
        },
        "periods": periods,
        "metric_comparison_vs_v25_and_v27": comparisons(corpus, rows, reference_v27, periods),
        "source_audit": source_audit,
        "proposal_ledger": str(ledger),
        "proposal_ledger_sha256": frozen_v26.sha256_file(ledger),
        "automatic_rejection": automatic_rejection(periods),
        "development_admitted": False,
        "final_admitted": False,
        "terminal_inventory_mtm_hidden": False,
        "holdout": {"label": "FUTURE_FX_HOLDOUT_AFTER_2026_07_15", "state": "UNOPENED"},
        **AUTHORITY,
        "admission_blockers": [
            "opened 2026 data are development evidence",
            "untouched future FX holdout remains unopened",
            "both normal and adverse full-month 2.0x gates are mandatory",
            "strategy adoption remains a separate gate",
        ],
    }
    payload["result_sha256"] = embedded_hash(payload, "result_sha256")
    result = output_root / "result_causal_basket_hold_v28.json"
    result.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--parent-ledger", type=Path, required=True)
    parser.add_argument("--parent-result", type=Path, required=True)
    parser.add_argument("--v27-result", type=Path, required=True)
    parser.add_argument("--v27-ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input_root, args.parent_ledger, args.parent_result,
                 args.v27_result, args.v27_ledger, args.output_root)
    print(json.dumps({
        "cycle_id": result["cycle_id"],
        "raw_signals": result["raw_signals"],
        "walk_forward": result["periods"]["WALK_FORWARD"],
        "automatic_rejection": result["automatic_rejection"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
