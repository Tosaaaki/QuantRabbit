from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


V2_DIR = Path(__file__).resolve().parents[1] / "2026-08-25-v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from fx_original_indicators import Bar, load_bars, sha256_file  # noqa: E402
import run_causal_consensus_release_persistence_v31 as frozen_v31  # noqa: E402
from run_asian_box_sweep_reclaim_v24 import (  # noqa: E402
    expected_stamp,
    path_for_signal,
    raw_path_metrics,
    timestamp,
)
from run_liquid_major_universe_v9 import UNIVERSE  # noqa: E402
from run_portfolio_episode_netting_v15 import PERIODS  # noqa: E402


DECISION_MINUTE = 5 * 60 + 55
FILL_MINUTE = 6 * 60
EXIT_MINUTE = 11 * 60 + 55
FIXED_PAIR_SLEEVE = 1 / 7
INITIAL_EQUITY_JPY = 200_000
TARGET_HOLD_SECONDS = frozen_v31.TARGET_HOLD_SECONDS
HARD_MAX_AGE_SECONDS = frozen_v31.HARD_MAX_AGE_SECONDS
ARMS = frozen_v31.ARMS
TRAINING_START = "2026-03-11"
TRAINING_END_EXCLUSIVE = "2026-05-01"
TRAINING_ABS_DISPLACEMENT_Q75 = {
    "AUD_USD": 0.002602051767152891,
    "EUR_USD": 0.0016711786566631797,
    "GBP_USD": 0.0020130857326814042,
    "NZD_USD": 0.0033787658794393136,
    "USD_CAD": 0.0009860867272267774,
    "USD_CHF": 0.001641535012627776,
    "USD_JPY": 0.0014524649063044073,
}


def _validated_map(pair: str, bars: list[Bar]) -> tuple[object, dict]:
    if not bars:
        raise ValueError(f"empty day for {pair}")
    parsed = [(timestamp(bar.time), bar) for bar in bars]
    if any(bar.pair != pair for _, bar in parsed):
        raise ValueError("pair/day map contains a different pair")
    day = parsed[0][0]
    if any(stamp.date() != day.date() for stamp, _ in parsed):
        raise ValueError("pair/day map spans multiple UTC dates")
    by_stamp = {stamp: bar for stamp, bar in parsed}
    if len(by_stamp) != len(parsed):
        raise ValueError("duplicate timestamp in pair/day map")
    return day, by_stamp


def detect_day_signals(pair_day_bars: dict[str, list[Bar]]) -> list[dict]:
    """Use only the completed Asian displacement to emit fixed handoff fades."""
    if set(pair_day_bars) != set(UNIVERSE):
        return []
    rows = []
    common_date = None
    for pair in sorted(UNIVERSE):
        try:
            day, by_stamp = _validated_map(pair, pair_day_bars[pair])
        except ValueError:
            return []
        if common_date is None:
            common_date = day.date()
        elif day.date() != common_date:
            return []
        required_completed = [expected_stamp(day, minute) for minute in range(0, DECISION_MINUTE + 1, 5)]
        if any(stamp not in by_stamp for stamp in required_completed):
            return []
        first = by_stamp[expected_stamp(day, 0)]
        completed = by_stamp[expected_stamp(day, DECISION_MINUTE)]
        displacement = math.log(completed.mid_c / first.mid_o)
        threshold = TRAINING_ABS_DISPLACEMENT_Q75[pair]
        if abs(displacement) < threshold or displacement == 0:
            continue
        direction = -1 if displacement > 0 else 1
        rows.append({
            "signal_id": (
                f"ADHF::{day.date().isoformat()}::{pair}::"
                f"{'FADE_LONG' if direction > 0 else 'FADE_SHORT'}"
            ),
            "pair": pair,
            "utc_day": day.date().isoformat(),
            "decision_time": by_stamp[expected_stamp(day, DECISION_MINUTE)].time,
            "fill_time": expected_stamp(day, FILL_MINUTE).isoformat().replace("+00:00", "Z"),
            "exit_time": expected_stamp(day, EXIT_MINUTE).isoformat().replace("+00:00", "Z"),
            "direction": direction,
            "diagnostics": {
                "native_asian_log_displacement": displacement,
                "training_abs_displacement_q75": threshold,
                "threshold_quantile": 0.75,
                "direction_rule": "NEGATIVE_SIGN_OF_NATIVE_DISPLACEMENT",
            },
        })
    return rows


def summarize_with_independence(rows: list[dict], start: str, end: str) -> dict:
    selected = [row for row in rows if start <= row["fill_time"][:10] < end and row["exit_time"][:10] < end]
    days = {row["utc_day"] for row in selected}
    return {
        "signals": len(selected),
        "effective_bet_days": len(days),
        "N_eff_days": len(days),
        "tickets_per_effective_bet": len(selected) / len(days) if days else None,
        "raw_definition_changed": True,
        "cost_used_for_signal": False,
    }


def simulate_portfolio(
    corpus: dict[str, list[Bar]], rows: list[dict], arm: str, start: str, end: str
) -> dict:
    """Apply the frozen V31 inventory/exit planner after the V32 signal ledger is fixed."""
    plans = frozen_v31.build_period_plans(corpus, rows, start, end)
    return frozen_v31.arm_metrics(plans, arm)


def build_execution_ledger(rows: list[dict], corpus: dict[str, list[Bar]]) -> list[dict]:
    plans = frozen_v31.build_period_plans(
        corpus, rows, min(row["fill_time"][:10] for row in rows), "9999-12-31"
    )
    actions = {
        event["signal_id"]: event["action"]
        for plan in plans.values() for event in plan["signal_events"]
    }
    result = []
    for source in rows:
        row = json.loads(json.dumps(source, sort_keys=True, allow_nan=False))
        row["execution_selected"] = True
        row["execution_action"] = actions[row["signal_id"]]
        row["arm_actions"] = {arm: row["execution_action"] for arm in ARMS}
        result.append(row)
    result.sort(key=lambda row: (row["fill_time"], row["signal_id"]))
    return result


def run(input_root: Path, output_root: Path) -> dict:
    frozen_v31.frozen_v30.frozen_v29.frozen_v28.runtime_v27.install_timestamp_compatibility()
    corpus = {}
    grouped: dict[str, dict[str, list[Bar]]] = {}
    source_audit = []
    for pair in sorted(UNIVERSE):
        matches = sorted((input_root / pair).glob("*_M5_BA_*.jsonl.gz"))
        if len(matches) != 1:
            raise ValueError(f"expected one source file for {pair}, got {len(matches)}")
        bars = load_bars(matches[0])
        corpus[pair] = bars
        grouped[pair] = defaultdict(list)
        for bar in bars:
            grouped[pair][bar.time[:10]].append(bar)
        source_audit.append({"pair": pair, "source_sha256": sha256_file(matches[0]), "bars": len(bars)})

    common_days = set.intersection(*(set(grouped[pair]) for pair in sorted(UNIVERSE)))
    rows = []
    for utc_day in sorted(common_days):
        pair_day_bars = {pair: grouped[pair][utc_day] for pair in sorted(UNIVERSE)}
        for signal in detect_day_signals(pair_day_bars):
            signal["raw_path"] = raw_path_metrics(
                path_for_signal(pair_day_bars[signal["pair"]], signal), int(signal["direction"])
            )
            rows.append(signal)
    rows.sort(key=lambda row: (row["fill_time"], row["signal_id"]))
    rows = build_execution_ledger(rows, corpus)

    periods = {}
    for name, (start, end) in PERIODS.items():
        raw_summary = summarize_with_independence(rows, start, end)
        periods[name] = {
            "raw_diagnostics": raw_summary,
            **{
                arm: simulate_portfolio(corpus, rows, arm, start, end)
                for arm in ARMS
            },
        }
        if len({periods[name][arm]["execution_state_transition_sha256"] for arm in ARMS}) != 1:
            raise ValueError("cost arms do not share identical execution-state transitions")
    monthly_profit_pass = all(
        periods[name][arm]["equity_multiple"] >= 2.0
        for name in ("MONTH_2026_05", "MONTH_2026_06")
        for arm in ("EXECUTABLE_BASE", "ADVERSE_STRESS")
    )
    walk = periods["WALK_FORWARD"]
    if walk["RAW_SIGNAL"]["equity_multiple"] <= 1.0:
        reason = "FX_SESSION_HANDOFF_FADE_RAW_EDGE_ABSENT"
    elif walk["EXECUTABLE_BASE"]["equity_multiple"] <= 1.0:
        reason = "FX_SESSION_HANDOFF_FADE_COST_DOMINANT"
    elif walk["ADVERSE_STRESS"]["equity_multiple"] <= 1.0:
        reason = "FX_SESSION_HANDOFF_FADE_ADVERSE_COST_FRAGILE"
    else:
        reason = "MONTHLY_2X_AND_UNOPENED_HOLDOUT_NOT_MET"

    output_root.mkdir(parents=True, exist_ok=True)
    ledger = output_root / "proposal_ledger_asian_displacement_handoff_fade_v32.jsonl"
    ledger.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    signal_ids = sorted(row["signal_id"] for row in rows)
    payload = {
        "cycle_id": "V32",
        "experiment": "FX_ASIAN_DISPLACEMENT_HANDOFF_FADE_V32",
        "family": "FX_SESSION_HANDOFF_MEAN_REVERSION",
        "family_hypotheses": 1,
        "evidence_class": "opened_development_not_future_holdout",
        "single_changed_variable": "fx_specific_asian_displacement_handoff_fade_signal_family",
        "indicator": {
            "training_window": [TRAINING_START, TRAINING_END_EXCLUSIVE],
            "training_abs_displacement_quantile": 0.75,
            "training_abs_displacement_q75": TRAINING_ABS_DISPLACEMENT_Q75,
            "measurement_window_utc": "00:00-05:55",
            "decision_utc": "05:55_COMPLETED",
            "fill_utc": "06:00_EXECUTABLE_OPEN",
            "fixed_exit_utc_bar": "11:55_COMPLETED_CLOSE",
            "direction_formula": "-sign(log(mid_close_05:55 / mid_open_00:00))",
            "cost_used_for_signal": False,
            "post_entry_outcome_used_for_signal": False,
            "evaluation_month_used_for_threshold": False,
        },
        "execution_contract": {
            "same_actions_all_cost_arms": True,
            "frozen_v31_inventory_and_exit_planner": True,
            "actual_bid_ask_for_executable_arms": True,
            "frozen_v7_slippage_commission_financing": True,
            "decision_to_fill_latency_seconds": 300,
            "target_hold_seconds": TARGET_HOLD_SECONDS,
            "finite_max_age_seconds": HARD_MAX_AGE_SECONDS,
            "terminal_liquidation_required": True,
        },
        "portfolio": {
            "pair_count": 7,
            "weight_per_pair": FIXED_PAIR_SLEEVE,
            "gross_leverage_cap": 1.0,
            "currency_abs_exposure_cap": 1.0,
        },
        "raw_signals": len(rows),
        "effective_bet_days": len({row["utc_day"] for row in rows}),
        "signal_id_set_sha256": hashlib.sha256(
            json.dumps(signal_ids, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        "cost_suppressed_raw_signals": 0,
        "same_signal_stream_all_cost_arms": True,
        "same_execution_actions_all_cost_arms": True,
        "same_execution_state_transitions_all_cost_arms": True,
        "execution_rule": {
            "name": "FROZEN_V31_CAUSAL_CONSENSUS_RELEASE_WITH_TWO_EVENT_PERSISTENCE",
            "changed_from_v31": False,
            "required_consecutive_confirmations": 2,
            "peer_scope": "ACTIVE_SAME_SIGNED_USD_INVENTORY_SUBGRAPH",
            "minimum_peer_signals": 2,
            "unanimity_required": True,
            "same_timestamp_required": True,
            "self_pair_excluded": True,
            "same_direction": "HOLD_EXISTING_NO_ADD_NO_EXPIRY_EXTENSION",
            "opposite_direction": "CLOSE_AT_SIGNAL_EXECUTABLE_OPEN_THEN_OPEN_REVERSE_FIXED_SLEEVE",
            "target_hold_seconds": TARGET_HOLD_SECONDS,
            "hard_max_age_seconds": HARD_MAX_AGE_SECONDS,
            "terminal_liquidation": True,
            "cost_or_outcome_inputs": False,
        },
        "proposal_ledger": str(ledger),
        "proposal_ledger_sha256": sha256_file(ledger),
        "periods": periods,
        "source_audit": source_audit,
        "development_admitted": monthly_profit_pass,
        "final_admitted": False,
        "automatic_rejection": {
            "rejected": True,
            "reason_code": reason,
            "numeric_results_preserved": True,
        },
        "holdout": {"label": "FUTURE_FX_HOLDOUT_AFTER_2026_07_15", "state": "UNOPENED", "may_execute": False},
        "terminal_inventory_mtm_hidden": False,
        "authority": {
            "paper_only": True,
            "live_authority": False,
            "broker_account_access": False,
            "credential_access": False,
            "order_endpoint": False,
            "external_orders": 0,
            "deploy": False,
            "external_config_mutation": False,
        },
        "live_authority": False,
        "external_orders": 0,
        "admission_blockers": [
            "opened 2026 data are development evidence",
            "untouched future FX holdout remains unopened",
            "both full comparable months must reach 2.0x under normal and adverse costs",
        ],
    }
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    result = output_root / "result_asian_displacement_handoff_fade_v32.json"
    result.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input_root, args.output_root)
    print(json.dumps({
        "raw_signals": result["raw_signals"],
        "effective_bet_days": result["effective_bet_days"],
        "periods": result["periods"],
        "development_admitted": result["development_admitted"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
