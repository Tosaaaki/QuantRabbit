"""Exact forward outcome and fixed-cohort evaluation for technical shadows.

Only post-lock OANDA S5 bid/ask candles may enter this module.  Unfilled
passive orders contribute zero, ambiguous exits are charged at the stop, and
filled positions still open at the frozen horizon are charged full risk.  A
positive scorecard is review evidence only; it never grants live permission.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_rabbit.instruments import instrument_pip_factor
from quant_rabbit.technical_forecast_forward_shadow import (
    FORWARD_SHADOW_CONTRACT,
    validate_forward_candidate,
)


FORWARD_OUTCOME_CONTRACT = "QR_TECHNICAL_FORECAST_FORWARD_OUTCOME_V1"
FORWARD_SCORECARD_CONTRACT = "QR_TECHNICAL_FORECAST_FORWARD_SCORECARD_V1"
TRUTH_CONTRACT = "QR_TECHNICAL_FORECAST_OANDA_S5_BA_TRUTH_V1"


@dataclass(frozen=True)
class S5BidAskCandle:
    timestamp_utc: datetime
    bid_o: float
    bid_h: float
    bid_l: float
    bid_c: float
    ask_o: float
    ask_h: float
    ask_l: float
    ask_c: float


def load_forward_shadows(
    path: Path,
    *,
    candidate_sha256: str,
) -> list[dict[str, Any]]:
    """Load a strict append-only shadow ledger for one candidate hash."""

    candidate_sha = _sha256(candidate_sha256, name="candidate_sha256")
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    decision_ids: set[str] = set()
    signal_ids: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read forward shadow ledger: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"forward shadow ledger line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"forward shadow ledger line {line_number} is not an object")
        _validate_shadow_row(row, candidate_sha=candidate_sha)
        decision_id = str(row["decision_id"])
        if decision_id in decision_ids:
            raise ValueError(f"duplicate forward decision_id: {decision_id}")
        decision_ids.add(decision_id)
        for signal in row["signals"]:
            signal_id = str(signal["signal_sha256"])
            if signal_id in signal_ids:
                raise ValueError(f"duplicate forward signal_sha256: {signal_id}")
            signal_ids.add(signal_id)
        rows.append(row)
    rows.sort(key=lambda row: (str(row["decision_at_utc"]), str(row["decision_id"])))
    return rows


def load_forward_outcomes(
    path: Path,
    *,
    candidate_sha256: str,
) -> list[dict[str, Any]]:
    """Load and verify the hash-chained outcome ledger."""

    candidate_sha = _sha256(candidate_sha256, name="candidate_sha256")
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read forward outcome ledger: {path}") from exc
    rows: list[dict[str, Any]] = []
    previous: str | None = None
    signal_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"forward outcome ledger line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"forward outcome ledger line {line_number} is not an object")
        if row.get("contract") != FORWARD_OUTCOME_CONTRACT:
            raise ValueError(f"forward outcome contract mismatch at line {line_number}")
        if row.get("schema_version") != 1:
            raise ValueError(f"forward outcome schema mismatch at line {line_number}")
        if row.get("candidate_sha256") != candidate_sha:
            raise ValueError(f"forward outcome candidate mismatch at line {line_number}")
        _validate_outcome_shape(row)
        if row.get("previous_outcome_sha256") != previous:
            raise ValueError(f"forward outcome chain mismatch at line {line_number}")
        payload_digest = _sha256(row.get("outcome_payload_sha256"), name="outcome_payload_sha256")
        payload_body = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "outcome_payload_sha256",
                "previous_outcome_sha256",
                "outcome_sha256",
            }
        }
        if _stable_digest(payload_body) != payload_digest:
            raise ValueError(f"forward outcome payload digest mismatch at line {line_number}")
        outcome_digest = _sha256(row.get("outcome_sha256"), name="outcome_sha256")
        chain_body = {key: value for key, value in row.items() if key != "outcome_sha256"}
        if _stable_digest(chain_body) != outcome_digest:
            raise ValueError(f"forward outcome digest mismatch at line {line_number}")
        signal_id = _sha256(row.get("signal_sha256"), name="signal_sha256")
        if signal_id in signal_ids:
            raise ValueError(f"duplicate forward outcome signal: {signal_id}")
        signal_ids.add(signal_id)
        rows.append(row)
        previous = outcome_digest
    return rows


def pending_forward_signals(
    candidate: Mapping[str, Any],
    shadows: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    *,
    as_of_utc: datetime,
) -> dict[str, Any]:
    """Select the bounded oldest due signals without any broker access."""

    _candidate(candidate)
    as_of = _aware_utc(as_of_utc)
    resolved = {str(row.get("signal_sha256") or "") for row in outcomes}
    tasks = _signal_tasks(candidate, shadows)
    expected = {str(task["signal_sha256"]) for task in tasks}
    unexpected = sorted(resolved - expected)
    if unexpected:
        raise ValueError("outcome ledger contains a signal absent from shadow ledger")
    due = [
        task
        for task in tasks
        if task["eligible_for_resolution_at"] <= as_of
        and task["signal_sha256"] not in resolved
    ]
    pending = [
        task
        for task in tasks
        if task["eligible_for_resolution_at"] > as_of
        and task["signal_sha256"] not in resolved
    ]
    limit = int(candidate["resolver"]["max_due_signals_per_run"])
    selected = due[:limit]
    return {
        "status": "DUE" if selected else "NO_DUE_SIGNALS",
        "as_of_utc": as_of.isoformat(),
        "emitted_signal_count": len(tasks),
        "resolved_signal_count": len(resolved),
        "pending_not_mature_count": len(pending),
        "due_unresolved_count": len(due),
        "selected_due_count": len(selected),
        "deferred_due_count": max(0, len(due) - len(selected)),
        "selected": selected,
    }


def resolve_frozen_forward_signal(
    candidate: Mapping[str, Any],
    task: Mapping[str, Any],
    candles: Sequence[S5BidAskCandle],
    *,
    candidate_sha256: str,
    resolved_at_utc: datetime,
    truth_chunk_sha256: Sequence[str],
) -> dict[str, Any]:
    """Resolve one frozen signal with exact executable S5 bid/ask geometry."""

    _candidate(candidate)
    candidate_sha = _sha256(candidate_sha256, name="candidate_sha256")
    signal = task.get("signal")
    if not isinstance(signal, Mapping):
        raise ValueError("forward task signal is missing")
    decision_id = _sha256(task.get("decision_id"), name="decision_id")
    _validate_signal(signal, expected_decision_id=decision_id)
    decision_at = _utc(task.get("decision_at_utc"), name="decision_at_utc")
    recorded_at = _utc(
        task.get("shadow_generated_at_utc"),
        name="shadow_generated_at_utc",
    )
    quote_at = _utc(signal.get("quote_timestamp_utc"), name="quote_timestamp_utc")
    entry_expires_at = _utc(
        signal.get("entry_expires_at_utc"),
        name="entry_expires_at_utc",
    )
    activation_at = _ceil_interval(
        max(quote_at, recorded_at),
        seconds=int(candidate["resolver"]["candle_interval_seconds"]),
    )
    if not decision_at <= quote_at < entry_expires_at:
        raise ValueError("forward signal quote is outside the frozen entry window")
    maturity_at = _utc(task.get("maturity_at_utc"), name="maturity_at_utc")
    resolved_at = _aware_utc(resolved_at_utc)
    grace = timedelta(seconds=int(candidate["resolver"]["truth_close_grace_seconds"]))
    if resolved_at < maturity_at + grace:
        raise ValueError("forward signal is not eligible for resolution")
    normalized = _validated_candles(
        candles,
        start=decision_at,
        end=maturity_at,
        interval_seconds=int(candidate["resolver"]["candle_interval_seconds"]),
    )
    chunk_hashes = [
        _sha256(value, name="truth_chunk_sha256") for value in truth_chunk_sha256
    ]
    if not chunk_hashes:
        raise ValueError("truth chunk hashes are required")
    signal_id = _sha256(signal.get("signal_sha256"), name="signal_sha256")
    side = str(signal["side"])
    pair = str(signal["pair"])
    entry = _positive_finite(signal.get("entry_price"), name="entry_price")
    target = _positive_finite(
        signal.get("take_profit_price"),
        name="take_profit_price",
    )
    stop = _positive_finite(signal.get("stop_loss_price"), name="stop_loss_price")
    pip = 1.0 / instrument_pip_factor(pair)
    base = {
        "contract": FORWARD_OUTCOME_CONTRACT,
        "schema_version": 1,
        "candidate_sha256": candidate_sha,
        "decision_id": decision_id,
        "signal_sha256": signal_id,
        "pair": pair,
        "predicted_direction": signal["predicted_direction"],
        "side": side,
        "decision_at_utc": decision_at.isoformat(),
        "shadow_generated_at_utc": recorded_at.isoformat(),
        "quote_timestamp_utc": quote_at.isoformat(),
        "activation_at_utc": activation_at.isoformat(),
        "entry_expires_at_utc": entry_expires_at.isoformat(),
        "maturity_at_utc": maturity_at.isoformat(),
        "resolved_at_utc": resolved_at.isoformat(),
        "entry_price": entry,
        "take_profit_price": target,
        "stop_loss_price": stop,
        "take_profit_pips": float(signal["take_profit_pips"]),
        "stop_loss_pips": float(signal["stop_loss_pips"]),
        "truth_contract": TRUTH_CONTRACT,
        "truth_source": candidate["resolver"]["truth_source"],
        "truth_granularity": "S5",
        "truth_interval_start_utc": decision_at.isoformat(),
        "truth_interval_end_utc": maturity_at.isoformat(),
        "truth_candle_count": len(normalized),
        "truth_candles_sha256": _stable_digest(
            [_candle_payload(candle) for candle in normalized]
        ),
        "truth_chunk_sha256": chunk_hashes,
        "missing_no_tick_intervals_synthesized": False,
        "shadow_only": True,
        "live_ready": False,
        "promotion_allowed": False,
        "broker_mutation_allowed": False,
        "order_intents": [],
    }
    executable_candles = [
        candle for candle in normalized if candle.timestamp_utc >= activation_at
    ]
    entry_candles = [
        candle
        for candle in executable_candles
        if candle.timestamp_utc < entry_expires_at
    ]
    fill_index = next(
        (
            index
            for index, candle in enumerate(entry_candles)
            if (side == "LONG" and candle.ask_l <= entry)
            or (side == "SHORT" and candle.bid_h >= entry)
        ),
        None,
    )
    if fill_index is None:
        return _seal_payload(
            {
                **base,
                "status": "MATURE_UNFILLED",
                "filled": False,
                "fill_at_utc": None,
                "exit_reason": "UNFILLED_EXPIRED",
                "exit_at_utc": None,
                "realized_pips": 0.0,
                "conservative_pips": 0.0,
                "gap_through_stop": False,
            }
        )
    fill_candle = entry_candles[fill_index]
    fill_position = executable_candles.index(fill_candle)
    exit_result = _attached_exit(
        side=side,
        entry=entry,
        target=target,
        stop=stop,
        candles=executable_candles[fill_position:],
        pip=pip,
        interval_seconds=int(candidate["resolver"]["candle_interval_seconds"]),
    )
    return _seal_payload(
        {
            **base,
            "filled": True,
            "fill_at_utc": fill_candle.timestamp_utc.isoformat(),
            **exit_result,
        }
    )


def append_forward_outcomes(
    path: Path,
    outcomes: Sequence[Mapping[str, Any]],
    *,
    candidate_sha256: str,
) -> int:
    """Append new outcomes once under a hash-chain lock."""

    candidate_sha = _sha256(candidate_sha256, name="candidate_sha256")
    if not outcomes:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    appended = 0
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = load_forward_outcomes(path, candidate_sha256=candidate_sha)
        existing = {str(row["signal_sha256"]) for row in current}
        previous = str(current[-1]["outcome_sha256"]) if current else None
        ordered = sorted(
            outcomes,
            key=lambda row: (
                str(row.get("maturity_at_utc") or ""),
                str(row.get("signal_sha256") or ""),
            ),
        )
        with path.open("a", encoding="utf-8") as handle:
            for raw in ordered:
                row = dict(raw)
                if row.get("candidate_sha256") != candidate_sha:
                    raise ValueError("outcome candidate hash mismatch")
                _validate_outcome_shape(row)
                signal_id = _sha256(row.get("signal_sha256"), name="signal_sha256")
                if signal_id in existing:
                    continue
                payload_digest = _sha256(
                    row.get("outcome_payload_sha256"),
                    name="outcome_payload_sha256",
                )
                payload_body = {
                    key: value
                    for key, value in row.items()
                    if key != "outcome_payload_sha256"
                }
                if _stable_digest(payload_body) != payload_digest:
                    raise ValueError("outcome payload digest mismatch")
                chained = {**row, "previous_outcome_sha256": previous}
                chained["outcome_sha256"] = _stable_digest(chained)
                handle.write(_canonical_json(chained) + "\n")
                previous = str(chained["outcome_sha256"])
                existing.add(signal_id)
                appended += 1
            if appended:
                handle.flush()
                os.fsync(handle.fileno())
    return appended


def build_forward_scorecard(
    candidate: Mapping[str, Any],
    shadows: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    *,
    candidate_sha256: str,
    as_of_utc: datetime,
    acquisition_errors: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Evaluate one deterministic first forward cohort without optional stopping."""

    _candidate(candidate)
    candidate_sha = _sha256(candidate_sha256, name="candidate_sha256")
    as_of = _aware_utc(as_of_utc)
    tasks = _signal_tasks(candidate, shadows)
    by_signal = {str(row.get("signal_sha256") or ""): row for row in outcomes}
    expected = {str(task["signal_sha256"]) for task in tasks}
    unexpected = sorted(signal_id for signal_id in by_signal if signal_id not in expected)
    if unexpected:
        raise ValueError("outcome ledger contains signals absent from the shadow ledger")
    mature = [task for task in tasks if task["eligible_for_resolution_at"] <= as_of]
    due_missing = [task for task in mature if task["signal_sha256"] not in by_signal]
    complete_batches: list[list[dict[str, Any]]] = []
    for _decision_id, batch in _decision_batches(mature):
        if any(task["signal_sha256"] not in by_signal for task in batch):
            break
        complete_batches.append(batch)

    evaluation = candidate["forward_evaluation"]
    cohort_tasks: list[dict[str, Any]] = []
    cohort_rows: list[Mapping[str, Any]] = []
    close_reason: str | None = None
    for batch in complete_batches:
        cohort_tasks.extend(batch)
        cohort_rows.extend(by_signal[task["signal_sha256"]] for task in batch)
        metrics = _score_metrics(cohort_rows)
        floors = (
            metrics["mature_signals"] >= int(evaluation["minimum_mature_signals"])
            and metrics["mature_fills"] >= int(evaluation["minimum_mature_fills"])
            and metrics["active_days"] >= int(evaluation["minimum_active_days"])
        )
        maximum = metrics["mature_signals"] >= int(
            evaluation["maximum_mature_signals"]
        )
        if floors or maximum:
            close_reason = (
                "SAMPLE_FLOORS_REACHED" if floors else "MAXIMUM_SIGNALS_REACHED"
            )
            break
    metrics = _score_metrics(cohort_rows)
    gates = _evaluation_gates(
        evaluation,
        metrics,
        cohort_closed=close_reason is not None,
    )
    evidence_passed = close_reason is not None and all(gate["passed"] for gate in gates)
    if not tasks:
        status = "NO_FORWARD_SIGNALS"
    elif not mature:
        status = "WAITING_FOR_MATURITY"
    elif due_missing and close_reason is None:
        status = "DUE_OUTCOMES_MISSING"
    elif close_reason is None:
        status = "COLLECTING_FORWARD_EVIDENCE"
    elif evidence_passed:
        status = "FORWARD_EVIDENCE_PASSED_REVIEW_REQUIRED"
    else:
        status = "FORWARD_EVIDENCE_REJECTED_LOCKED_COHORT"
    payload = {
        "contract": FORWARD_SCORECARD_CONTRACT,
        "schema_version": 1,
        "generated_at_utc": as_of.isoformat(),
        "candidate_sha256": candidate_sha,
        "status": status,
        "cohort_selection": evaluation["cohort_selection"],
        "cohort_closed": close_reason is not None,
        "cohort_close_reason": close_reason,
        "cohort_terminal_decision_id": (
            cohort_tasks[-1]["decision_id"] if cohort_tasks else None
        ),
        "emitted_decision_count": len(shadows),
        "emitted_signal_count": len(tasks),
        "mature_signal_count": len(mature),
        "recorded_outcome_count": len(outcomes),
        "due_outcome_missing_count": len(due_missing),
        "pending_not_mature_count": len(tasks) - len(mature),
        "post_cohort_outcome_count": max(0, len(outcomes) - len(cohort_rows)),
        "metrics": metrics,
        "gates": gates,
        "forward_evidence_passed": evidence_passed,
        "promotion_review_required": evidence_passed,
        "promotion_allowed": False,
        "live_order_enabled": False,
        "broker_mutation_allowed": False,
        "order_intents": [],
        "acquisition_errors": [dict(error) for error in acquisition_errors],
    }
    return {**payload, "scorecard_sha256": _stable_digest(payload)}


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _signal_tasks(
    candidate: Mapping[str, Any],
    shadows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grace = timedelta(seconds=int(candidate["resolver"]["truth_close_grace_seconds"]))
    tasks: list[dict[str, Any]] = []
    for shadow in shadows:
        decision_at = _utc(shadow.get("decision_at_utc"), name="decision_at_utc")
        recorded_at = _utc(
            shadow.get("generated_at_utc"),
            name="generated_at_utc",
        )
        decision_id = _sha256(shadow.get("decision_id"), name="decision_id")
        signals = shadow.get("signals")
        if not isinstance(signals, list):
            raise ValueError("shadow signals must be a list")
        for signal in signals:
            if not isinstance(signal, Mapping):
                raise ValueError("shadow signal must be an object")
            _validate_signal(signal, expected_decision_id=decision_id)
            maturity = decision_at + timedelta(
                minutes=int(signal["entry_ttl_min"]) + int(signal["max_hold_min"])
            )
            tasks.append(
                {
                    "decision_id": decision_id,
                    "decision_at_utc": decision_at.isoformat(),
                    "shadow_generated_at_utc": recorded_at.isoformat(),
                    "maturity_at_utc": maturity.isoformat(),
                    "eligible_for_resolution_at_utc": (maturity + grace).isoformat(),
                    "eligible_for_resolution_at": maturity + grace,
                    "signal_sha256": str(signal["signal_sha256"]),
                    "pair": str(signal["pair"]),
                    "signal": dict(signal),
                }
            )
    tasks.sort(
        key=lambda task: (
            task["eligible_for_resolution_at"],
            task["decision_id"],
            task["pair"],
            task["signal_sha256"],
        )
    )
    return tasks


def _decision_batches(
    tasks: Sequence[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    batches: list[tuple[str, list[dict[str, Any]]]] = []
    for task in tasks:
        decision_id = str(task["decision_id"])
        if not batches or batches[-1][0] != decision_id:
            batches.append((decision_id, []))
        batches[-1][1].append(task)
    return batches


def _score_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    signal_values = [float(row["conservative_pips"]) for row in rows]
    filled = [row for row in rows if row.get("filled") is True]
    fill_values = [float(row["conservative_pips"]) for row in filled]
    resolved = [
        row
        for row in filled
        if row.get("exit_reason") in {"TAKE_PROFIT", "STOP_LOSS"}
    ]
    daily: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        daily[str(row["decision_at_utc"])[:10]].append(float(row["conservative_pips"]))
    daily_values = [sum(values) for _day, values in sorted(daily.items())]
    profit_factor, infinite = _profit_factor(fill_values)
    return {
        "mature_signals": len(rows),
        "mature_fills": len(filled),
        "resolved_fills": len(resolved),
        "unfilled_signals": len(rows) - len(filled),
        "take_profit_fills": sum(row.get("exit_reason") == "TAKE_PROFIT" for row in filled),
        "stop_loss_fills": sum(row.get("exit_reason") == "STOP_LOSS" for row in filled),
        "ambiguous_fills": sum(
            str(row.get("exit_reason") or "").startswith("AMBIGUOUS")
            for row in filled
        ),
        "open_unresolved_fills": sum(
            row.get("exit_reason") == "OPEN_UNRESOLVED" for row in filled
        ),
        "fill_rate": _ratio(len(filled), len(rows)),
        "resolved_fill_fraction": _ratio(len(resolved), len(filled)),
        "mean_conservative_pips_per_signal": _mean(signal_values),
        "mean_conservative_pips_per_fill": _mean(fill_values),
        "net_conservative_pips": round(sum(signal_values), 6),
        "profit_factor": profit_factor,
        "profit_factor_infinite": infinite,
        "one_sided_95_student_t_lower_pips_per_fill": _student_t_lower(fill_values),
        "active_days": len(daily_values),
        "positive_day_rate": _ratio(sum(value > 0.0 for value in daily_values), len(daily_values)),
        "one_sided_95_student_t_daily_lower_pips": _student_t_lower(daily_values),
        "student_t_critical_policy": "ONE_SIDED_95_DF30_CONSERVATIVE_CAP",
    }


def _evaluation_gates(
    evaluation: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    cohort_closed: bool,
) -> list[dict[str, Any]]:
    profit_factor = (
        math.inf
        if metrics.get("profit_factor_infinite") is True
        else _finite(metrics.get("profit_factor"))
    )
    checks = [
        ("COHORT_CLOSED", cohort_closed, True, "eq"),
        ("MINIMUM_MATURE_SIGNALS", metrics["mature_signals"], evaluation["minimum_mature_signals"], "gte"),
        ("MINIMUM_MATURE_FILLS", metrics["mature_fills"], evaluation["minimum_mature_fills"], "gte"),
        ("MINIMUM_RESOLVED_FILL_FRACTION", metrics["resolved_fill_fraction"], evaluation["minimum_resolved_fill_fraction"], "gte"),
        ("MINIMUM_ACTIVE_DAYS", metrics["active_days"], evaluation["minimum_active_days"], "gte"),
        ("MINIMUM_FILL_RATE", metrics["fill_rate"], evaluation["minimum_fill_rate"], "gte"),
        ("POSITIVE_MEAN_PER_SIGNAL", metrics["mean_conservative_pips_per_signal"], evaluation["minimum_mean_conservative_pips_per_signal"], "gt"),
        ("POSITIVE_MEAN_PER_FILL", metrics["mean_conservative_pips_per_fill"], evaluation["minimum_mean_conservative_pips_per_fill"], "gt"),
        ("POSITIVE_FILL_STUDENT_T_LOWER", metrics["one_sided_95_student_t_lower_pips_per_fill"], evaluation["minimum_one_sided_95_student_t_lower_pips_per_fill"], "gt"),
        ("MINIMUM_PROFIT_FACTOR", profit_factor, evaluation["minimum_profit_factor"], "gte"),
        ("MINIMUM_POSITIVE_DAY_RATE", metrics["positive_day_rate"], evaluation["minimum_positive_day_rate"], "gte"),
        ("POSITIVE_DAILY_STUDENT_T_LOWER", metrics["one_sided_95_student_t_daily_lower_pips"], evaluation["minimum_one_sided_95_student_t_daily_lower_pips"], "gt"),
    ]
    gates: list[dict[str, Any]] = []
    for name, actual, threshold, operator in checks:
        actual_infinite = (
            isinstance(actual, float) and math.isinf(actual) and actual > 0.0
        )
        if operator == "eq":
            passed = actual is threshold
        else:
            actual_number = (
                math.inf
                if actual_infinite
                else _finite(actual)
            )
            threshold_number = _finite(threshold)
            if actual_number is None or threshold_number is None:
                passed = False
            elif operator == "gte":
                passed = actual_number >= threshold_number
            elif operator == "lte":
                passed = actual_number <= threshold_number
            else:
                passed = actual_number > threshold_number
        gates.append(
            {
                "gate": name,
                "operator": operator,
                "actual": None if actual_infinite else actual,
                "actual_infinite": actual_infinite,
                "threshold": threshold,
                "passed": passed,
            }
        )
    return gates


def _attached_exit(
    *,
    side: str,
    entry: float,
    target: float,
    stop: float,
    candles: Sequence[S5BidAskCandle],
    pip: float,
    interval_seconds: int,
) -> dict[str, Any]:
    risk_pips = abs(entry - stop) / pip
    reward_pips = abs(target - entry) / pip
    for index, candle in enumerate(candles):
        if side == "LONG":
            target_hit = candle.bid_h >= target
            stop_hit = candle.bid_l <= stop
            target_close_proof = candle.bid_c >= target
            stop_open = candle.bid_o
            gap_stop = stop_open < stop
            stop_exit = stop_open if gap_stop else stop
            stop_pips = (stop_exit - entry) / pip
        else:
            target_hit = candle.ask_l <= target
            stop_hit = candle.ask_h >= stop
            target_close_proof = candle.ask_c <= target
            stop_open = candle.ask_o
            gap_stop = stop_open > stop
            stop_exit = stop_open if gap_stop else stop
            stop_pips = (entry - stop_exit) / pip
        exit_at = candle.timestamp_utc + timedelta(seconds=interval_seconds)
        if stop_hit and target_hit:
            return {
                "status": "MATURE_FILLED_AMBIGUOUS",
                "exit_reason": "AMBIGUOUS_TP_SL_ORDERING",
                "exit_at_utc": None,
                "ambiguity_at_utc": exit_at.isoformat(),
                "realized_pips": None,
                "conservative_pips": round(min(stop_pips, -risk_pips), 6),
                "gap_through_stop": gap_stop,
            }
        if index == 0 and target_hit and not target_close_proof:
            return {
                "status": "MATURE_FILLED_AMBIGUOUS",
                "exit_reason": "AMBIGUOUS_TARGET_BEFORE_FILL",
                "exit_at_utc": None,
                "ambiguity_at_utc": exit_at.isoformat(),
                "realized_pips": None,
                "conservative_pips": round(-risk_pips, 6),
                "gap_through_stop": False,
            }
        if stop_hit:
            return {
                "status": "MATURE_FILLED_RESOLVED",
                "exit_reason": "STOP_LOSS",
                "exit_at_utc": exit_at.isoformat(),
                "realized_pips": round(stop_pips, 6),
                "conservative_pips": round(stop_pips, 6),
                "gap_through_stop": gap_stop,
            }
        if target_hit:
            return {
                "status": "MATURE_FILLED_RESOLVED",
                "exit_reason": "TAKE_PROFIT",
                "exit_at_utc": exit_at.isoformat(),
                "realized_pips": round(reward_pips, 6),
                "conservative_pips": round(reward_pips, 6),
                "gap_through_stop": False,
            }
    return {
        "status": "MATURE_FILLED_OPEN",
        "exit_reason": "OPEN_UNRESOLVED",
        "exit_at_utc": None,
        "realized_pips": None,
        "conservative_pips": round(-risk_pips, 6),
        "gap_through_stop": False,
    }


def _validate_shadow_row(row: Mapping[str, Any], *, candidate_sha: str) -> None:
    if row.get("contract") != FORWARD_SHADOW_CONTRACT:
        raise ValueError("forward shadow contract mismatch")
    if row.get("schema_version") != 1 or row.get("status") != "EMITTED":
        raise ValueError("forward shadow must be one emitted schema-v1 decision")
    if row.get("candidate_sha256") != candidate_sha:
        raise ValueError("forward shadow candidate hash mismatch")
    if (
        row.get("shadow_only") is not True
        or row.get("live_ready") is not False
        or row.get("promotion_allowed") is not False
        or row.get("broker_mutation_allowed") is not False
        or row.get("order_intents") != []
    ):
        raise ValueError("forward shadow safety boundary mismatch")
    decision_id = _sha256(row.get("decision_id"), name="decision_id")
    shadow_digest = _sha256(row.get("shadow_sha256"), name="shadow_sha256")
    body = {key: value for key, value in row.items() if key != "shadow_sha256"}
    if _stable_digest(body) != shadow_digest:
        raise ValueError("forward shadow digest mismatch")
    decision_at = _utc(row.get("decision_at_utc"), name="decision_at_utc")
    closes_at = _utc(
        row.get("collection_closes_at_utc"),
        name="collection_closes_at_utc",
    )
    generated_at = _utc(row.get("generated_at_utc"), name="generated_at_utc")
    if not decision_at <= generated_at <= closes_at:
        raise ValueError("forward shadow was not recorded inside its collection window")
    signals = row.get("signals")
    if not isinstance(signals, list) or not signals:
        raise ValueError("emitted forward shadow needs signals")
    for signal in signals:
        if not isinstance(signal, Mapping):
            raise ValueError("forward signal must be an object")
        _validate_signal(signal, expected_decision_id=decision_id)
        quote_at = _utc(signal.get("quote_timestamp_utc"), name="quote_timestamp_utc")
        if not decision_at <= quote_at <= closes_at:
            raise ValueError("forward signal quote lies outside its decision window")


def _validate_outcome_shape(row: Mapping[str, Any]) -> None:
    if row.get("contract") != FORWARD_OUTCOME_CONTRACT or row.get("schema_version") != 1:
        raise ValueError("forward outcome contract is invalid")
    if (
        row.get("truth_contract") != TRUTH_CONTRACT
        or row.get("truth_source") != "OANDA_S5_BID_ASK"
        or row.get("truth_granularity") != "S5"
        or row.get("missing_no_tick_intervals_synthesized") is not False
    ):
        raise ValueError("forward outcome truth contract is invalid")
    if (
        row.get("shadow_only") is not True
        or row.get("live_ready") is not False
        or row.get("promotion_allowed") is not False
        or row.get("broker_mutation_allowed") is not False
        or row.get("order_intents") != []
    ):
        raise ValueError("forward outcome safety boundary is invalid")
    _sha256(row.get("candidate_sha256"), name="candidate_sha256")
    _sha256(row.get("decision_id"), name="decision_id")
    _sha256(row.get("signal_sha256"), name="signal_sha256")
    _sha256(row.get("truth_candles_sha256"), name="truth_candles_sha256")
    chunks = row.get("truth_chunk_sha256")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("forward outcome truth chunks are missing")
    for chunk in chunks:
        _sha256(chunk, name="truth_chunk_sha256")
    candle_count = row.get("truth_candle_count")
    if candle_count.__class__ is not int or candle_count < 0:
        raise ValueError("forward outcome truth candle count is invalid")
    decision = _utc(row.get("decision_at_utc"), name="decision_at_utc")
    recorded = _utc(
        row.get("shadow_generated_at_utc"),
        name="shadow_generated_at_utc",
    )
    quote = _utc(row.get("quote_timestamp_utc"), name="quote_timestamp_utc")
    activation = _utc(row.get("activation_at_utc"), name="activation_at_utc")
    expires = _utc(row.get("entry_expires_at_utc"), name="entry_expires_at_utc")
    maturity = _utc(row.get("maturity_at_utc"), name="maturity_at_utc")
    resolved = _utc(row.get("resolved_at_utc"), name="resolved_at_utc")
    truth_start = _utc(
        row.get("truth_interval_start_utc"),
        name="truth_interval_start_utc",
    )
    truth_end = _utc(
        row.get("truth_interval_end_utc"),
        name="truth_interval_end_utc",
    )
    if (
        truth_start != decision
        or truth_end != maturity
        or expires != decision + timedelta(minutes=5)
        or maturity != decision + timedelta(minutes=1440)
        or not decision <= quote < expires
        or not decision <= recorded < expires
        or not max(quote, recorded) <= activation < expires < maturity <= resolved
    ):
        raise ValueError("forward outcome frozen timeline is invalid")
    pair = str(row.get("pair") or "").upper()
    side = str(row.get("side") or "")
    direction = str(row.get("predicted_direction") or "")
    if not pair or pair != row.get("pair") or (direction, side) not in {
        ("UP", "LONG"),
        ("DOWN", "SHORT"),
    }:
        raise ValueError("forward outcome pair or direction is invalid")
    entry = _positive_finite(row.get("entry_price"), name="entry_price")
    target = _positive_finite(row.get("take_profit_price"), name="take_profit_price")
    stop = _positive_finite(row.get("stop_loss_price"), name="stop_loss_price")
    pip = 1.0 / instrument_pip_factor(pair)
    if side == "LONG" and not stop < entry < target:
        raise ValueError("forward outcome LONG geometry is invalid")
    if side == "SHORT" and not target < entry < stop:
        raise ValueError("forward outcome SHORT geometry is invalid")
    if not math.isclose(abs(target - entry) / pip, 15.0, abs_tol=1e-6):
        raise ValueError("forward outcome target distance is invalid")
    if not math.isclose(abs(entry - stop) / pip, 30.0, abs_tol=1e-6):
        raise ValueError("forward outcome stop distance is invalid")
    filled = row.get("filled")
    status = str(row.get("status") or "")
    reason = str(row.get("exit_reason") or "")
    realized = _finite(row.get("realized_pips"))
    conservative = _finite(row.get("conservative_pips"))
    if conservative is None:
        raise ValueError("forward outcome conservative pips are invalid")
    if filled is False:
        if (
            status != "MATURE_UNFILLED"
            or reason != "UNFILLED_EXPIRED"
            or realized != 0.0
            or conservative != 0.0
        ):
            raise ValueError("forward unfilled outcome economics are invalid")
    elif filled is True and reason == "TAKE_PROFIT":
        if (
            status != "MATURE_FILLED_RESOLVED"
            or realized != 15.0
            or conservative != 15.0
        ):
            raise ValueError("forward take-profit economics are invalid")
    elif filled is True and reason == "STOP_LOSS":
        if (
            status != "MATURE_FILLED_RESOLVED"
            or realized is None
            or realized > -30.0 + 1e-6
            or conservative != realized
        ):
            raise ValueError("forward stop-loss economics are invalid")
    elif filled is True and reason in {
        "AMBIGUOUS_TP_SL_ORDERING",
        "AMBIGUOUS_TARGET_BEFORE_FILL",
        "OPEN_UNRESOLVED",
    }:
        expected_status = (
            "MATURE_FILLED_OPEN"
            if reason == "OPEN_UNRESOLVED"
            else "MATURE_FILLED_AMBIGUOUS"
        )
        if (
            status != expected_status
            or realized is not None
            or conservative > -30.0 + 1e-6
        ):
            raise ValueError("forward conservative unresolved economics are invalid")
    else:
        raise ValueError("forward outcome fill/exit status is invalid")


def _validate_signal(signal: Mapping[str, Any], *, expected_decision_id: str) -> None:
    signal_digest = _sha256(signal.get("signal_sha256"), name="signal_sha256")
    body = {key: value for key, value in signal.items() if key != "signal_sha256"}
    if _stable_digest(body) != signal_digest:
        raise ValueError("forward signal digest mismatch")
    if signal.get("decision_id") != expected_decision_id:
        raise ValueError("forward signal decision mismatch")
    direction = str(signal.get("predicted_direction") or "")
    side = str(signal.get("side") or "")
    if (direction, side) not in {("UP", "LONG"), ("DOWN", "SHORT")}:
        raise ValueError("forward signal direction/side mismatch")
    expected = {
        "horizon_min": 1440,
        "selected_rule": "return_12",
        "orientation": "DIRECT",
        "order_type": "LIMIT",
        "entry_ttl_min": 5,
        "max_hold_min": 1435,
        "take_profit_pips": 15.0,
        "stop_loss_pips": 30.0,
        "shadow_only": True,
        "live_ready": False,
        "broker_mutation_allowed": False,
    }
    for key, required in expected.items():
        if signal.get(key) != required:
            raise ValueError(f"forward signal {key} mismatch")
    pair = str(signal.get("pair") or "").upper()
    if not pair or pair != signal.get("pair"):
        raise ValueError("forward signal pair is invalid")
    entry = _positive_finite(signal.get("entry_price"), name="entry_price")
    target = _positive_finite(signal.get("take_profit_price"), name="take_profit_price")
    stop = _positive_finite(signal.get("stop_loss_price"), name="stop_loss_price")
    pip = 1.0 / instrument_pip_factor(pair)
    if side == "LONG" and not stop < entry < target:
        raise ValueError("forward LONG geometry is invalid")
    if side == "SHORT" and not target < entry < stop:
        raise ValueError("forward SHORT geometry is invalid")
    if not math.isclose(abs(target - entry) / pip, 15.0, abs_tol=1e-6):
        raise ValueError("forward target distance mismatch")
    if not math.isclose(abs(entry - stop) / pip, 30.0, abs_tol=1e-6):
        raise ValueError("forward stop distance mismatch")
    _utc(signal.get("entry_expires_at_utc"), name="entry_expires_at_utc")
    _utc(signal.get("quote_timestamp_utc"), name="quote_timestamp_utc")


def _ceil_interval(value: datetime, *, seconds: int) -> datetime:
    timestamp = value.timestamp()
    rounded = math.ceil(timestamp / seconds - 1e-12) * seconds
    return datetime.fromtimestamp(rounded, tz=timezone.utc)


def _validated_candles(
    candles: Sequence[S5BidAskCandle],
    *,
    start: datetime,
    end: datetime,
    interval_seconds: int,
) -> list[S5BidAskCandle]:
    normalized = list(candles)
    previous: datetime | None = None
    for candle in normalized:
        timestamp = _aware_utc(candle.timestamp_utc)
        if not start <= timestamp < end:
            raise ValueError("truth candle lies outside the frozen interval")
        if int(timestamp.timestamp()) % interval_seconds != 0:
            raise ValueError("truth candle timestamp is off the S5 grid")
        if previous is not None and timestamp <= previous:
            raise ValueError("truth candles must be strictly chronological and unique")
        previous = timestamp
        bid = (candle.bid_o, candle.bid_h, candle.bid_l, candle.bid_c)
        ask = (candle.ask_o, candle.ask_h, candle.ask_l, candle.ask_c)
        if any(_finite(value) is None or float(value) <= 0.0 for value in (*bid, *ask)):
            raise ValueError("truth candle contains an invalid price")
        if not (
            candle.bid_l <= min(candle.bid_o, candle.bid_c) <= candle.bid_h
            and candle.bid_l <= max(candle.bid_o, candle.bid_c) <= candle.bid_h
            and candle.ask_l <= min(candle.ask_o, candle.ask_c) <= candle.ask_h
            and candle.ask_l <= max(candle.ask_o, candle.ask_c) <= candle.ask_h
        ):
            raise ValueError("truth candle OHLC envelope is invalid")
        if not all(bid_value < ask_value for bid_value, ask_value in zip(bid, ask)):
            raise ValueError("truth candle bid/ask ordering is invalid")
    return normalized


def _candle_payload(candle: S5BidAskCandle) -> dict[str, Any]:
    return {
        "timestamp_utc": _aware_utc(candle.timestamp_utc).isoformat(),
        "bid": [candle.bid_o, candle.bid_h, candle.bid_l, candle.bid_c],
        "ask": [candle.ask_o, candle.ask_h, candle.ask_l, candle.ask_c],
    }


def _candidate(candidate: Mapping[str, Any]) -> None:
    issues = validate_forward_candidate(candidate)
    if issues:
        raise ValueError("invalid forward candidate: " + "; ".join(issues))


def _seal_payload(body: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(body)
    return {**payload, "outcome_payload_sha256": _stable_digest(payload)}


def _student_t_lower(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    critical = _student_t_critical(len(values) - 1)
    lower = statistics.mean(values) - critical * statistics.stdev(values) / math.sqrt(
        len(values)
    )
    return round(lower, 6)


def _student_t_critical(degrees_of_freedom: int) -> float:
    table = (
        6.314,
        2.920,
        2.353,
        2.132,
        2.015,
        1.943,
        1.895,
        1.860,
        1.833,
        1.812,
        1.796,
        1.782,
        1.771,
        1.761,
        1.753,
        1.746,
        1.740,
        1.734,
        1.729,
        1.725,
        1.721,
        1.717,
        1.714,
        1.711,
        1.708,
        1.706,
        1.703,
        1.701,
        1.699,
        1.697,
    )
    if degrees_of_freedom < 1:
        raise ValueError("Student-t degrees of freedom must be positive")
    return table[min(degrees_of_freedom, len(table)) - 1]


def _profit_factor(values: Sequence[float]) -> tuple[float | None, bool]:
    profit = sum(value for value in values if value > 0.0)
    loss = -sum(value for value in values if value < 0.0)
    if loss == 0.0:
        return (None, profit > 0.0)
    return (round(profit / loss, 6), False)


def _mean(values: Sequence[float]) -> float | None:
    return round(statistics.mean(values), 6) if values else None


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _positive_finite(value: Any, *, name: str) -> float:
    parsed = _finite(value)
    if parsed is None or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def _utc(value: Any, *, name: str) -> datetime:
    if isinstance(value, datetime):
        return _aware_utc(value)
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be an aware UTC timestamp") from exc
    try:
        return _aware_utc(parsed)
    except ValueError as exc:
        raise ValueError(f"{name} must be an aware UTC timestamp") from exc


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _sha256(value: Any, *, name: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return text


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _stable_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


__all__ = [
    "FORWARD_OUTCOME_CONTRACT",
    "FORWARD_SCORECARD_CONTRACT",
    "S5BidAskCandle",
    "TRUTH_CONTRACT",
    "append_forward_outcomes",
    "build_forward_scorecard",
    "load_forward_outcomes",
    "load_forward_shadows",
    "pending_forward_signals",
    "resolve_frozen_forward_signal",
    "write_json_atomic",
]
