"""Independent forward shadow for the causal four-hour technical selector.

The module deliberately has no order-intent or gateway adapter.  It retunes a
predeclared M5 technical rule from only already-resolved bid/ask outcomes,
records market-entry shadows, and later resolves the exact S5 stop/time-close
vehicle.  Historical research and inspected holdout rows never enter the
forward scorecard.
"""

from __future__ import annotations

import concurrent.futures
import bisect
import fcntl
import hashlib
import json
import math
import os
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from quant_rabbit.broker.oanda import OandaReadOnlyClient
from quant_rabbit.forecast_passive_limit_replay import (
    MARKET_STOP_TIME_CLOSE_REPLAY_CONTRACT,
    simulate_market_stop_time_close,
)
from quant_rabbit.instruments import instrument_pip_factor


CANDIDATE_CONTRACT = "QR_CONTEXTUAL_TECHNICAL_240M_FORWARD_CANDIDATE_V1"
SHADOW_CONTRACT = "QR_CONTEXTUAL_TECHNICAL_240M_FORWARD_SHADOW_V1"
OUTCOME_CONTRACT = "QR_CONTEXTUAL_TECHNICAL_240M_FORWARD_OUTCOME_V1"
SCORECARD_CONTRACT = "QR_CONTEXTUAL_TECHNICAL_240M_FORWARD_SCORECARD_V1"
TRUTH_ADAPTER_CONTRACT = "QR_CONTEXTUAL_TECHNICAL_240M_TRUTH_ADAPTER_V1"
RULES = (
    "breakout_fast",
    "trend_fast",
    "trend_slow",
    "pullback_in_trend",
    "mean_revert_fast",
    "mean_revert_slow",
)
CONTEXT_SCOPES = (
    ("PAIR_PHASE_SESSION", ("pair", "market_phase", "utc_session_bucket")),
    ("PAIR_PHASE", ("pair", "market_phase")),
    ("PAIR_SESSION", ("pair", "utc_session_bucket")),
    ("PAIR", ("pair",)),
    ("PHASE_SESSION", ("market_phase", "utc_session_bucket")),
    ("PHASE", ("market_phase",)),
    ("GLOBAL", ()),
)
THRESHOLD_QUANTILES = (0.0, 0.50, 0.70, 0.85)


@dataclass(frozen=True)
class Ohlc:
    o: float
    h: float
    l: float
    c: float


@dataclass(frozen=True)
class BidAskCandle:
    timestamp_utc: datetime
    pair: str
    bid: Ohlc
    ask: Ohlc


@dataclass(frozen=True)
class _ForecastRow:
    source_index: int
    timestamp_utc: datetime
    pair: str
    direction: str


class _ReadOnlyClient(Protocol):
    def get_json(
        self,
        path: str,
        query: dict[str, str] | None = None,
    ) -> dict[str, Any]: ...


def load_candidate(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("contextual forward candidate must be an object")
    validate_candidate(value)
    return value


def validate_candidate(value: Mapping[str, Any]) -> None:
    if value.get("contract") != CANDIDATE_CONTRACT or value.get("schema_version") != 1:
        raise ValueError("contextual forward candidate contract is invalid")
    if value.get("shadow_enabled") is not True:
        raise ValueError("contextual forward shadow must be enabled")
    if value.get("live_order_enabled") is not False or value.get("promotion_allowed") is not False:
        raise ValueError("contextual forward candidate must remain shadow-only")
    selector = _mapping(value.get("selector"), "selector")
    pairs = selector.get("pairs")
    if not isinstance(pairs, list) or not pairs or len(set(map(str, pairs))) != len(pairs):
        raise ValueError("selector.pairs must be a non-empty unique list")
    if tuple(selector.get("rules") or ()) != RULES:
        raise ValueError("selector.rules do not match the frozen rule family")
    if tuple(float(item) for item in selector.get("threshold_quantiles") or ()) != THRESHOLD_QUANTILES:
        raise ValueError("selector.threshold_quantiles do not match the frozen grid")
    if int(selector.get("horizon_min") or 0) != 240:
        raise ValueError("selector.horizon_min must be 240")
    if int(selector.get("schedule_interval_min") or 0) != 240:
        raise ValueError("selector.schedule_interval_min must be 240")
    if int(selector.get("context_lookback_days") or 0) != 14:
        raise ValueError("selector.context_lookback_days must be 14")
    if int(selector.get("minimum_context_rows") or 0) < 12:
        raise ValueError("selector.minimum_context_rows must be at least 12")
    if selector.get("admission") != "SHRUNK_POSITIVE":
        raise ValueError("selector.admission must be SHRUNK_POSITIVE")
    vehicle = _mapping(value.get("vehicle"), "vehicle")
    if vehicle.get("entry_vehicle") != "MARKET_STOP_TIME_CLOSE":
        raise ValueError("vehicle must be MARKET_STOP_TIME_CLOSE")
    if float(vehicle.get("risk_pips") or 0.0) != 20.0:
        raise ValueError("vehicle.risk_pips must be the frozen 20-pip research geometry")
    if float(vehicle.get("max_hold_min") or 0.0) != 240.0:
        raise ValueError("vehicle.max_hold_min must be 240")
    if vehicle.get("take_profit_pips") is not None:
        raise ValueError("time-close candidate must not invent a fixed take profit")
    evaluation = _mapping(value.get("forward_evaluation"), "forward_evaluation")
    fixed = int(evaluation.get("fixed_cohort_signals") or 0)
    if fixed < 30 or int(evaluation.get("minimum_active_days") or 0) < 5:
        raise ValueError("forward cohort floors are too small")
    _utc(value.get("forward_evaluation_not_before_utc"), name="forward lock")


def decision_window(
    candidate: Mapping[str, Any],
    *,
    as_of_utc: datetime,
) -> dict[str, Any]:
    validate_candidate(candidate)
    now = _aware_utc(as_of_utc)
    selector = _mapping(candidate["selector"], "selector")
    interval = int(selector["schedule_interval_min"])
    minute = int(now.timestamp() // 60)
    decision = datetime.fromtimestamp((minute // interval) * interval * 60, tz=timezone.utc)
    closes = decision + timedelta(seconds=int(selector["collection_lateness_max_seconds"]))
    not_before = _utc(candidate["forward_evaluation_not_before_utc"], name="forward lock")
    status = "OPEN"
    if decision < not_before:
        status = "BEFORE_FORWARD_LOCK"
    elif now > closes:
        status = "OUTSIDE_COLLECTION_WINDOW"
    return {
        "status": status,
        "decision_at_utc": decision.isoformat(),
        "collection_closes_at_utc": closes.isoformat(),
        "observed_at_utc": now.isoformat(),
    }


def emit_forward_shadow_from_oanda(
    *,
    candidate_path: Path,
    shadow_path: Path,
    shadow_ledger_path: Path,
    client_factory: Callable[[], _ReadOnlyClient] = OandaReadOnlyClient,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    now = _aware_utc((clock or _utc_now)())
    candidate = load_candidate(candidate_path)
    candidate_sha = _file_sha256(candidate_path)
    window = decision_window(candidate, as_of_utc=now)
    base = {
        "contract": SHADOW_CONTRACT,
        "schema_version": 1,
        "candidate_sha256": candidate_sha,
        "shadow_only": True,
        "broker_read": False,
        "broker_mutation": False,
        "live_order_enabled": False,
        "promotion_allowed": False,
        "order_intents": [],
        **window,
    }
    if window["status"] != "OPEN":
        result = _seal({**base, "status": window["status"], "signals": []}, "shadow_sha256")
        write_json_atomic(shadow_path, result)
        return result
    decision = _utc(window["decision_at_utc"], name="decision")
    decision_id = _stable_digest({"candidate_sha256": candidate_sha, "decision_at_utc": decision.isoformat()})
    if _decision_recorded(shadow_ledger_path, decision_id=decision_id, candidate_sha=candidate_sha):
        result = _seal({**base, "status": "ALREADY_RECORDED", "decision_id": decision_id, "signals": []}, "shadow_sha256")
        write_json_atomic(shadow_path, result)
        return result
    if now < decision + timedelta(seconds=5):
        result = _seal({**base, "status": "WAITING_FOR_FIRST_COMPLETE_S5_QUOTE", "decision_id": decision_id, "signals": []}, "shadow_sha256")
        write_json_atomic(shadow_path, result)
        return result
    try:
        client = client_factory()
    except Exception as exc:
        result = _seal({**base, "status": "CLIENT_UNAVAILABLE", "decision_id": decision_id, "signals": [], "errors": [_error("CLIENT", exc)]}, "shadow_sha256")
        write_json_atomic(shadow_path, result)
        return result
    selector = _mapping(candidate["selector"], "selector")
    pairs = [str(pair) for pair in selector["pairs"]]
    history_from = decision - timedelta(days=int(selector["history_fetch_days"]))
    workers = min(len(pairs), int(_mapping(candidate["resolver"], "resolver")["max_workers"]))
    m5_by_pair: dict[str, list[BidAskCandle]] = {}
    s5_by_pair: dict[str, list[BidAskCandle]] = {}
    errors: list[dict[str, Any]] = []

    def fetch(pair: str) -> tuple[str, list[BidAskCandle], list[BidAskCandle]]:
        m5, _ = fetch_bidask_candles(
            client,
            pair=pair,
            granularity="M5",
            time_from=history_from,
            time_to=decision,
            chunk_candle_limit=int(_mapping(candidate["resolver"], "resolver")["chunk_candle_limit"]),
        )
        s5, _ = fetch_bidask_candles(
            client,
            pair=pair,
            granularity="S5",
            time_from=decision,
            time_to=now,
            chunk_candle_limit=int(_mapping(candidate["resolver"], "resolver")["chunk_candle_limit"]),
        )
        return pair, m5, s5

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch, pair): pair for pair in pairs}
        for future in concurrent.futures.as_completed(futures):
            pair = futures[future]
            try:
                key, m5, s5 = future.result()
                m5_by_pair[key] = m5
                s5_by_pair[key] = s5
            except Exception as exc:
                errors.append(_error("HISTORY_OR_ENTRY_FETCH", exc, pair=pair))
    shadow = build_forward_shadow(
        candidate,
        candidate_sha256=candidate_sha,
        decision_at_utc=decision,
        observed_at_utc=now,
        m5_by_pair=m5_by_pair,
        s5_by_pair=s5_by_pair,
        acquisition_errors=errors,
    )
    if shadow.get("status") == "EMITTED" or (
        shadow.get("status") == "NO_ADMITTED_SIGNALS"
        and not shadow.get("errors")
    ):
        append_jsonl_once(
            shadow_ledger_path,
            shadow,
            identity_key="decision_id",
            expected_identity=decision_id,
        )
    write_json_atomic(shadow_path, shadow)
    return shadow


def build_forward_shadow(
    candidate: Mapping[str, Any],
    *,
    candidate_sha256: str,
    decision_at_utc: datetime,
    observed_at_utc: datetime,
    m5_by_pair: Mapping[str, Sequence[BidAskCandle]],
    s5_by_pair: Mapping[str, Sequence[BidAskCandle]],
    acquisition_errors: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    validate_candidate(candidate)
    decision = _aware_utc(decision_at_utc)
    observed = _aware_utc(observed_at_utc)
    selector = _mapping(candidate["selector"], "selector")
    vehicle = _mapping(candidate["vehicle"], "vehicle")
    decision_id = _stable_digest({"candidate_sha256": candidate_sha256, "decision_at_utc": decision.isoformat()})
    history_rows: list[dict[str, Any]] = []
    current_rows: list[dict[str, Any]] = []
    input_provenance: dict[str, Any] = {}
    issues = [dict(item) for item in acquisition_errors]
    for pair in map(str, selector["pairs"]):
        m5 = list(m5_by_pair.get(pair) or ())
        s5 = sorted(s5_by_pair.get(pair) or (), key=lambda item: item.timestamp_utc)
        input_provenance[pair] = {
            "m5_candles": len(m5),
            "m5_sha256": _stable_digest([_candle_payload(item) for item in m5]),
            "s5_entry_candles": len(s5),
            "s5_entry_sha256": _stable_digest([_candle_payload(item) for item in s5]),
        }
        try:
            history, current = _pair_selector_rows(
                pair,
                m5,
                s5,
                decision_at_utc=decision,
                selector=selector,
            )
        except Exception as exc:
            issues.append(_error("PAIR_SELECTOR", exc, pair=pair))
            continue
        history_rows.extend(history)
        if current is not None:
            current_rows.append(current)
    signals: list[dict[str, Any]] = []
    for current in sorted(current_rows, key=lambda row: str(row["pair"])):
        context, scope = _context_history(
            history_rows,
            current,
            minimum_rows=int(selector["minimum_context_rows"]),
        )
        selected = _best_context_rule(
            context,
            current,
            minimum_qualified_rows=max(4, int(selector["minimum_context_rows"]) // 2),
        )
        if selected is None:
            continue
        shrunk = float(selected["mean_pips"]) - float(selected["standard_error_pips"])
        if shrunk <= 0.0:
            continue
        score = float(current[str(selected["rule"])]) * int(selected["orientation"])
        direction = "UP" if score >= 0.0 else "DOWN"
        pip = 1.0 / instrument_pip_factor(str(current["pair"]))
        entry = float(current["entry_ask"] if direction == "UP" else current["entry_bid"])
        risk = float(vehicle["risk_pips"])
        invalidation = entry - risk * pip if direction == "UP" else entry + risk * pip
        signal_body = {
            "decision_id": decision_id,
            "decision_at_utc": decision.isoformat(),
            "forecast_at_utc": current["forecast_at_utc"],
            "maturity_at_utc": (decision + timedelta(minutes=float(vehicle["max_hold_min"]))).isoformat(),
            "pair": str(current["pair"]),
            "direction": direction,
            "side": "LONG" if direction == "UP" else "SHORT",
            "entry_vehicle": "MARKET_STOP_TIME_CLOSE",
            "entry_quote_at_utc": current["entry_quote_at_utc"],
            "entry_bid": round(float(current["entry_bid"]), 9),
            "entry_ask": round(float(current["entry_ask"]), 9),
            "entry_price": round(entry, 9),
            "entry_spread_pips": round(float(current["entry_spread_pips"]), 6),
            "invalidation_price": round(invalidation, 9),
            "risk_pips": risk,
            "take_profit_pips": None,
            "max_hold_min": float(vehicle["max_hold_min"]),
            "technical_selection": {
                "market_phase": current["market_phase"],
                "utc_session_bucket": current["utc_session_bucket"],
                "context_scope": scope,
                "context_rows": len(context),
                "rule": selected["rule"],
                "orientation": "DIRECT" if int(selected["orientation"]) == 1 else "INVERSE",
                "minimum_absolute_score": round(float(selected["threshold"]), 9),
                "current_rule_value": round(float(current[str(selected["rule"])]), 9),
                "oriented_score": round(score, 9),
                "qualified_rows": int(selected["qualified_rows"]),
                "recent_mean_pips": round(float(selected["mean_pips"]), 6),
                "recent_standard_error_pips": round(float(selected["standard_error_pips"]), 6),
                "shrunk_recent_edge_pips": round(shrunk, 6),
            },
            "shadow_only": True,
            "live_order_enabled": False,
        }
        signals.append(_seal(signal_body, "signal_sha256"))
    body = {
        "contract": SHADOW_CONTRACT,
        "schema_version": 1,
        "candidate_sha256": candidate_sha256,
        "decision_id": decision_id,
        "decision_at_utc": decision.isoformat(),
        "observed_at_utc": observed.isoformat(),
        "status": "EMITTED" if signals else "NO_ADMITTED_SIGNALS",
        "shadow_only": True,
        "broker_read": True,
        "broker_mutation": False,
        "live_order_enabled": False,
        "promotion_allowed": False,
        "history_resolved_rows": len(history_rows),
        "current_eligible_rows": len(current_rows),
        "signals": signals,
        "input_provenance": input_provenance,
        "errors": issues[:20],
        "order_intents": [],
    }
    return _seal(body, "shadow_sha256")


def resolve_due_outcomes_from_oanda(
    *,
    candidate_path: Path,
    shadow_ledger_path: Path,
    outcome_ledger_path: Path,
    scorecard_path: Path,
    client_factory: Callable[[], _ReadOnlyClient] = OandaReadOnlyClient,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    now = _aware_utc((clock or _utc_now)())
    candidate = load_candidate(candidate_path)
    candidate_sha = _file_sha256(candidate_path)
    shadows = load_jsonl(shadow_ledger_path, contract=SHADOW_CONTRACT, candidate_sha=candidate_sha)
    outcomes = load_jsonl(outcome_ledger_path, contract=OUTCOME_CONTRACT, candidate_sha=candidate_sha)
    resolved_ids = {str(row["signal_sha256"]) for row in outcomes}
    resolver = _mapping(candidate["resolver"], "resolver")
    grace = timedelta(seconds=int(resolver["truth_close_grace_seconds"]))
    tasks = [
        signal
        for shadow in shadows
        for signal in shadow.get("signals", [])
        if str(signal.get("signal_sha256")) not in resolved_ids
        and now >= _utc(signal.get("maturity_at_utc"), name="maturity") + grace
    ]
    tasks.sort(key=lambda row: (str(row["maturity_at_utc"]), str(row["pair"]), str(row["signal_sha256"])))
    tasks = tasks[: int(resolver["max_due_signals_per_run"])]
    base = {
        "contract": TRUTH_ADAPTER_CONTRACT,
        "schema_version": 1,
        "as_of_utc": now.isoformat(),
        "shadow_only": True,
        "broker_mutation": False,
        "live_order_enabled": False,
        "promotion_allowed": False,
        "order_intents": [],
    }
    if not tasks:
        scorecard = build_forward_scorecard(candidate, shadows, outcomes, candidate_sha256=candidate_sha, as_of_utc=now)
        write_json_atomic(scorecard_path, scorecard)
        return {**base, "status": "NO_DUE_SIGNALS", "broker_read": False, "scorecard_status": scorecard["status"]}
    client = client_factory()
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    def resolve(signal: Mapping[str, Any]) -> dict[str, Any]:
        decision = _utc(signal["decision_at_utc"], name="decision")
        maturity = _utc(signal["maturity_at_utc"], name="maturity")
        candles, chunk_hashes = fetch_bidask_candles(
            client,
            pair=str(signal["pair"]),
            granularity="S5",
            time_from=decision,
            time_to=maturity + grace + timedelta(seconds=5),
            chunk_candle_limit=int(resolver["chunk_candle_limit"]),
        )
        replay = simulate_market_stop_time_close(
            _ForecastRow(
                source_index=0,
                timestamp_utc=decision,
                pair=str(signal["pair"]),
                direction=str(signal["direction"]),
            ),
            candles,
            horizon_min=float(signal["max_hold_min"]),
            risk_pips=float(signal["risk_pips"]),
            candle_interval=timedelta(seconds=5),
            candle_times=[item.timestamp_utc for item in candles],
            time_close_quote_grace=grace,
        )
        if replay.get("filled") is not True:
            raise ValueError("frozen market signal did not reproduce its recorded entry")
        if str(replay.get("fill_at_utc")) != str(signal.get("entry_quote_at_utc")):
            raise ValueError("frozen entry quote timestamp changed")
        if abs(float(replay["entry_price"]) - float(signal["entry_price"])) > 1e-9:
            raise ValueError("frozen entry price changed")
        body = {
            "contract": OUTCOME_CONTRACT,
            "schema_version": 1,
            "candidate_sha256": candidate_sha,
            "decision_id": signal["decision_id"],
            "signal_sha256": signal["signal_sha256"],
            "pair": signal["pair"],
            "direction": signal["direction"],
            "decision_at_utc": signal["decision_at_utc"],
            "maturity_at_utc": signal["maturity_at_utc"],
            "resolved_at_utc": now.isoformat(),
            "truth_contract": MARKET_STOP_TIME_CLOSE_REPLAY_CONTRACT,
            "truth_chunk_sha256": chunk_hashes,
            "status": replay["status"],
            "exit_reason": replay.get("exit_reason"),
            "exit_at_utc": replay.get("exit_at_utc"),
            "entry_price": replay.get("entry_price"),
            "exit_price": replay.get("exit_price"),
            "realized_pips": replay.get("realized_pips"),
            "conservative_pips": replay.get("conservative_pips"),
            "gap_through_stop": replay.get("gap_through_stop", False),
            "shadow_only": True,
            "live_order_enabled": False,
        }
        return _seal(body, "outcome_sha256")

    workers = min(len(tasks), int(resolver["max_workers"]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(resolve, signal): signal for signal in tasks}
        for future in concurrent.futures.as_completed(futures):
            signal = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                errors.append(_error("TRUTH_FETCH_OR_RESOLUTION", exc, pair=str(signal.get("pair") or ""), signal_sha256=str(signal.get("signal_sha256") or "")))
    appended = 0
    for row in sorted(rows, key=lambda item: (str(item["decision_at_utc"]), str(item["pair"]))):
        if append_jsonl_once(outcome_ledger_path, row, identity_key="signal_sha256", expected_identity=str(row["signal_sha256"])):
            appended += 1
    outcomes = load_jsonl(outcome_ledger_path, contract=OUTCOME_CONTRACT, candidate_sha=candidate_sha)
    scorecard = build_forward_scorecard(candidate, shadows, outcomes, candidate_sha256=candidate_sha, as_of_utc=now, acquisition_errors=errors)
    write_json_atomic(scorecard_path, scorecard)
    return {
        **base,
        "status": "RESOLVED_WITH_ERRORS" if errors else "RESOLVED",
        "broker_read": True,
        "selected_due_count": len(tasks),
        "resolved_in_memory_count": len(rows),
        "ledger_appended_count": appended,
        "scorecard_status": scorecard["status"],
        "forward_evidence_passed": scorecard["forward_evidence_passed"],
        "errors": errors[:20],
    }


def build_forward_scorecard(
    candidate: Mapping[str, Any],
    shadows: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    *,
    candidate_sha256: str,
    as_of_utc: datetime,
    acquisition_errors: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    evaluation = _mapping(candidate["forward_evaluation"], "forward_evaluation")
    fixed = int(evaluation["fixed_cohort_signals"])
    signals = sorted(
        [signal for shadow in shadows for signal in shadow.get("signals", [])],
        key=lambda row: (str(row["decision_at_utc"]), str(row["pair"]), str(row["signal_sha256"])),
    )
    cohort = signals[:fixed]
    outcome_by_id = {str(row["signal_sha256"]): row for row in outcomes}
    resolved = [outcome_by_id[str(signal["signal_sha256"])] for signal in cohort if str(signal["signal_sha256"]) in outcome_by_id]
    locked = len(cohort) == fixed and len(resolved) == fixed
    metrics = _score_metrics(resolved)
    profit_factor = (
        math.inf
        if metrics.get("profit_factor_infinite") is True
        else float(metrics.get("profit_factor") or 0.0)
    )
    gates = {
        "FIXED_COHORT_COMPLETE": locked,
        "MINIMUM_ACTIVE_DAYS": int(metrics["active_days"]) >= int(evaluation["minimum_active_days"]),
        "POSITIVE_MEAN_PIPS": _greater(metrics.get("mean_conservative_pips"), evaluation["minimum_mean_conservative_pips"]),
        "POSITIVE_T_LOWER": _greater(metrics.get("one_sided_95_student_t_lower_pips"), evaluation["minimum_one_sided_95_student_t_lower_pips"]),
        "MINIMUM_PROFIT_FACTOR": profit_factor >= float(evaluation["minimum_profit_factor"]),
        "MINIMUM_POSITIVE_DAY_RATE": float(metrics.get("positive_day_rate") or 0.0) >= float(evaluation["minimum_positive_day_rate"]),
        "POSITIVE_DAILY_T_LOWER": _greater(metrics.get("one_sided_95_daily_lower_pips"), evaluation["minimum_one_sided_95_daily_lower_pips"]),
    }
    passed = bool(locked and all(gates.values()))
    if not signals:
        status = "NO_FORWARD_SIGNALS"
    elif not locked:
        status = "COLLECTING_FIXED_FORWARD_COHORT"
    elif passed:
        status = "FORWARD_EVIDENCE_PASSED_REVIEW_REQUIRED"
    else:
        status = "FORWARD_EVIDENCE_REJECTED_FIXED_COHORT"
    body = {
        "contract": SCORECARD_CONTRACT,
        "schema_version": 1,
        "candidate_sha256": candidate_sha256,
        "generated_at_utc": _aware_utc(as_of_utc).isoformat(),
        "status": status,
        "shadow_only": True,
        "live_order_enabled": False,
        "promotion_allowed": False,
        "forward_evidence_passed": passed,
        "fixed_cohort_signals": fixed,
        "emitted_signal_count": len(signals),
        "cohort_signal_count": len(cohort),
        "cohort_resolved_count": len(resolved),
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, ok in gates.items() if not ok],
        "acquisition_errors": [dict(item) for item in acquisition_errors][:20],
        "promotion_blockers": list(candidate.get("promotion_blockers") or ()),
        "order_intents": [],
    }
    return _seal(body, "scorecard_sha256")


def fetch_bidask_candles(
    client: _ReadOnlyClient,
    *,
    pair: str,
    granularity: str,
    time_from: datetime,
    time_to: datetime,
    chunk_candle_limit: int,
) -> tuple[list[BidAskCandle], list[str]]:
    seconds = {"S5": 5, "M5": 300}.get(str(granularity))
    if seconds is None:
        raise ValueError("only S5 and M5 bid/ask truth are supported")
    start = _aware_utc(time_from)
    end = _aware_utc(time_to)
    if end <= start:
        raise ValueError("truth interval must be positive")
    if not 1 <= int(chunk_candle_limit) <= 5000:
        raise ValueError("chunk_candle_limit must be inside 1..5000")
    step = timedelta(seconds=seconds * int(chunk_candle_limit))
    cursor = start
    by_timestamp: dict[datetime, BidAskCandle] = {}
    chunk_hashes: list[str] = []
    while cursor < end:
        chunk_end = min(end, cursor + step)
        payload = client.get_json(
            f"/v3/instruments/{pair}/candles",
            {
                "granularity": granularity,
                "from": _oanda_time(cursor),
                "to": _oanda_time(chunk_end),
                "price": "BA",
                "includeFirst": "true",
                "smooth": "false",
            },
        )
        if not isinstance(payload, dict) or not isinstance(payload.get("candles"), list):
            raise ValueError("OANDA candle response is invalid")
        if payload.get("instrument") != pair or payload.get("granularity") != granularity:
            raise ValueError("OANDA candle provenance mismatch")
        if len(payload["candles"]) > int(chunk_candle_limit):
            raise ValueError("OANDA response exceeds the requested chunk bound")
        chunk_hashes.append(_stable_digest(payload))
        for item in payload["candles"]:
            if not isinstance(item, Mapping) or item.get("complete") is not True:
                continue
            candle = _parse_candle(item, pair=pair)
            if not cursor <= candle.timestamp_utc < chunk_end:
                raise ValueError("OANDA candle lies outside requested chunk")
            previous = by_timestamp.get(candle.timestamp_utc)
            if previous is not None and previous != candle:
                raise ValueError("conflicting OANDA candle timestamp")
            by_timestamp[candle.timestamp_utc] = candle
        cursor = chunk_end
    return [by_timestamp[key] for key in sorted(by_timestamp)], chunk_hashes


def load_jsonl(path: Path, *, contract: str, candidate_sha: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    seal_field = {
        SHADOW_CONTRACT: "shadow_sha256",
        OUTCOME_CONTRACT: "outcome_sha256",
    }.get(contract)
    identity_field = {
        SHADOW_CONTRACT: "decision_id",
        OUTCOME_CONTRACT: "signal_sha256",
    }.get(contract)
    if seal_field is None or identity_field is None:
        raise ValueError("unsupported contextual forward ledger contract")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or value.get("contract") != contract:
                raise ValueError(f"ledger contract mismatch at line {number}")
            if value.get("candidate_sha256") != candidate_sha:
                raise ValueError(f"ledger candidate mismatch at line {number}")
            stored_seal = value.get(seal_field)
            body = dict(value)
            body.pop(seal_field, None)
            if not isinstance(stored_seal, str) or stored_seal != _stable_digest(body):
                raise ValueError(f"ledger seal mismatch at line {number}")
            identity = str(value.get(identity_field) or "")
            if not identity or identity in seen:
                raise ValueError(f"ledger identity is missing or duplicated at line {number}")
            seen.add(identity)
            rows.append(value)
    return rows


def append_jsonl_once(
    path: Path,
    row: Mapping[str, Any],
    *,
    identity_key: str,
    expected_identity: str,
) -> bool:
    if str(row.get(identity_key)) != str(expected_identity):
        raise ValueError("ledger identity mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            with path.open(encoding="utf-8") as current:
                for line in current:
                    try:
                        previous = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if str(previous.get(identity_key)) == str(expected_identity):
                        return False
        with path.open("a", encoding="utf-8") as handle:
            handle.write(_canonical_json(dict(row)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return True


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


def _pair_selector_rows(
    pair: str,
    candles: Sequence[BidAskCandle],
    entry_candles: Sequence[BidAskCandle],
    *,
    decision_at_utc: datetime,
    selector: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    ordered = sorted(candles, key=lambda item: item.timestamp_utc)
    features = _feature_rows(pair, ordered)
    by_time = {row["timestamp_utc"]: index for index, row in enumerate(features)}
    forecast_at = decision_at_utc - timedelta(minutes=5)
    lookback_from = forecast_at - timedelta(days=int(selector["context_lookback_days"]))
    horizon = timedelta(minutes=int(selector["horizon_min"]))
    spread_cap = float(selector["spread_cap_pips"])
    history: list[dict[str, Any]] = []
    for index, row in enumerate(features):
        entry_at = row["timestamp_utc"] + timedelta(minutes=5)
        if int(entry_at.timestamp() // 60) % int(selector["schedule_interval_min"]) != 0:
            continue
        if not lookback_from <= row["timestamp_utc"] < forecast_at:
            continue
        entry_index = by_time.get(entry_at)
        exit_at = entry_at + horizon
        exit_index = by_time.get(exit_at)
        truth_bars = int(selector["horizon_min"]) // 5 + 1
        if (
            entry_index is None
            or exit_index is None
            or entry_index != index + 1
            or exit_index != index + truth_bars
            or exit_at > forecast_at
        ):
            continue
        if index < 288 or row["timestamp_utc"] - features[index - 288]["timestamp_utc"] > timedelta(hours=26):
            continue
        if not _row_complete(row):
            continue
        entry_candle = ordered[entry_index]
        exit_candle = ordered[exit_index]
        pip_factor = instrument_pip_factor(pair)
        entry_spread = (entry_candle.ask.o - entry_candle.bid.o) * pip_factor
        if entry_spread > spread_cap:
            continue
        labelled = dict(row)
        labelled.update(
            {
                "entry_timestamp_utc": entry_at.isoformat(),
                "future_timestamp_utc": exit_at.isoformat(),
                "entry_spread_pips": entry_spread,
                "long_pips": (exit_candle.bid.o - entry_candle.ask.o) * pip_factor,
                "short_pips": (entry_candle.bid.o - exit_candle.ask.o) * pip_factor,
            }
        )
        history.append(labelled)
    current_index = by_time.get(forecast_at)
    if current_index is None or not entry_candles:
        return history, None
    current = dict(features[current_index])
    if not _row_complete(current):
        return history, None
    quote = min(
        (item for item in entry_candles if item.timestamp_utc >= decision_at_utc),
        key=lambda item: item.timestamp_utc,
        default=None,
    )
    if quote is None:
        return history, None
    spread = (quote.ask.o - quote.bid.o) * instrument_pip_factor(pair)
    if spread > spread_cap:
        return history, None
    current.update(
        {
            "forecast_at_utc": forecast_at.isoformat(),
            "entry_quote_at_utc": quote.timestamp_utc.isoformat(),
            "entry_bid": quote.bid.o,
            "entry_ask": quote.ask.o,
            "entry_spread_pips": spread,
        }
    )
    return history, current


def _feature_rows(pair: str, candles: Sequence[BidAskCandle]) -> list[dict[str, Any]]:
    if not candles:
        return []
    pip_factor = instrument_pip_factor(pair)
    mid_open = [(item.bid.o + item.ask.o) / 2.0 for item in candles]
    mid_close = [(item.bid.c + item.ask.c) / 2.0 for item in candles]
    mid_high = [(item.bid.h + item.ask.h) / 2.0 for item in candles]
    mid_low = [(item.bid.l + item.ask.l) / 2.0 for item in candles]
    ema = {span: _ema(mid_close, span) for span in (5, 12, 24, 48)}
    returns = {
        lag: [None if index < lag else (mid_close[index] - mid_close[index - lag]) * pip_factor for index in range(len(candles))]
        for lag in (3, 12, 48)
    }
    delta = [None] + [(mid_close[index] - mid_close[index - 1]) * pip_factor for index in range(1, len(candles))]
    gain = [None if item is None else max(item, 0.0) for item in delta]
    loss = [None if item is None else max(-item, 0.0) for item in delta]
    gain14 = _rolling_mean(gain, 14)
    loss14 = _rolling_mean(loss, 14)
    rsi = [
        None
        if gain14[index] is None or loss14[index] in (None, 0.0)
        else 100.0 - 100.0 / (1.0 + float(gain14[index]) / float(loss14[index]))
        for index in range(len(candles))
    ]
    true_range = []
    for index in range(len(candles)):
        values = [(mid_high[index] - mid_low[index]) * pip_factor]
        if index:
            values.extend(
                [
                    abs(mid_high[index] - mid_close[index - 1]) * pip_factor,
                    abs(mid_low[index] - mid_close[index - 1]) * pip_factor,
                ]
            )
        true_range.append(max(values))
    atr14 = _rolling_mean(true_range, 14)
    atr48 = _rolling_mean(true_range, 48)
    atr288 = _rolling_mean(true_range, 288)
    location12 = _rolling_location(mid_close, 12)
    location48 = _rolling_location(mid_close, 48)
    trend_strength: list[float | None] = []
    volatility_ratio: list[float | None] = []
    alignment: list[bool] = []
    basic: list[dict[str, Any]] = []
    for index, candle in enumerate(candles):
        gap5_24 = (ema[5][index] - ema[24][index]) * pip_factor
        gap12_48 = (ema[12][index] - ema[48][index]) * pip_factor
        a14, a48, a288 = atr14[index], atr48[index], atr288[index]
        strength = None if not _positive(a48) else abs(gap12_48) / float(a48)
        ratio = None if not _positive(a288) or a14 is None else float(a14) / float(a288)
        trend_strength.append(strength)
        volatility_ratio.append(ratio)
        aligned = bool(
            returns[12][index] is not None
            and returns[48][index] is not None
            and _sign(returns[12][index]) == _sign(returns[48][index]) == _sign(gap12_48)
        )
        alignment.append(aligned)
        body_ratio = None if not _positive(a14) else ((mid_close[index] - mid_open[index]) * pip_factor) / float(a14)
        location12_scaled = None if location12[index] is None else (float(location12[index]) - 0.5) * 2.0
        location48_scaled = None if location48[index] is None else (float(location48[index]) - 0.5) * 2.0
        rsi_scaled = None if rsi[index] is None else (float(rsi[index]) - 50.0) / 50.0
        basic.append(
            {
                "pair": pair,
                "timestamp_utc": candle.timestamp_utc,
                "utc_session_bucket": _session(candle.timestamp_utc.hour),
                "breakout_fast": _mean_or_none((
                    _divide(returns[3][index], a14), body_ratio, location48_scaled
                )),
                "trend_fast": _mean_or_none((
                    _divide(returns[3][index], a14),
                    _divide(returns[12][index], a48),
                    _divide(gap5_24, a48),
                )),
                "trend_slow": _mean_or_none((
                    _divide(returns[12][index], a48),
                    _divide(returns[48][index], a288),
                    _divide(gap12_48, a48),
                )),
                "pullback_in_trend": _mean_or_none((
                    _divide(gap12_48, a48),
                    _negative(_divide(returns[3][index], a14)),
                    _negative(location12_scaled),
                )),
                "mean_revert_fast": _mean_or_none((
                    _negative(_divide(returns[3][index], a14)),
                    _negative(rsi_scaled),
                    _negative(location12_scaled),
                )),
                "mean_revert_slow": _mean_or_none((
                    _negative(_divide(returns[12][index], a48)),
                    _negative(rsi_scaled),
                    _negative(location48_scaled),
                )),
            }
        )
    trend_quantiles = _rolling_prior_quantiles(
        trend_strength,
        window=8640,
        minimum_rows=2016,
        quantiles=(0.25, 0.75),
    )
    volatility_quantiles = _rolling_prior_quantiles(
        volatility_ratio,
        window=8640,
        minimum_rows=2016,
        quantiles=(0.25,),
    )
    for index, row in enumerate(basic):
        trend_low, trend_high = trend_quantiles[index]
        (volatility_low,) = volatility_quantiles[index]
        strength = trend_strength[index]
        ratio = volatility_ratio[index]
        change = None if index < 12 or strength is None or trend_strength[index - 12] is None else strength - float(trend_strength[index - 12])
        is_trend = bool(alignment[index] and strength is not None and trend_high is not None and strength >= trend_high)
        is_pre_trend = bool(
            not is_trend
            and (
                ratio is not None and volatility_low is not None and ratio <= volatility_low
                or alignment[index] and change is not None and change > 0.0
            )
        )
        is_range = bool(not is_trend and not is_pre_trend and strength is not None and trend_low is not None and strength <= trend_low)
        row["market_phase"] = "TREND" if is_trend else "PRE_TREND" if is_pre_trend else "RANGE" if is_range else "PRE_RANGE"
    return basic


def _context_history(
    history: Sequence[Mapping[str, Any]],
    current: Mapping[str, Any],
    *,
    minimum_rows: int,
) -> tuple[list[Mapping[str, Any]], str]:
    for scope, fields in CONTEXT_SCOPES:
        rows = [row for row in history if all(row.get(field) == current.get(field) for field in fields)]
        if len(rows) >= minimum_rows:
            return rows, scope
    return [], "NONE"


def _best_context_rule(
    history: Sequence[Mapping[str, Any]],
    current: Mapping[str, Any],
    *,
    minimum_qualified_rows: int,
) -> dict[str, Any] | None:
    best_rank: tuple[float, float, int] | None = None
    best: dict[str, Any] | None = None
    for rule in RULES:
        values = [float(row[rule]) for row in history if _finite(row.get(rule)) is not None]
        if len(values) != len(history) or _finite(current.get(rule)) is None:
            continue
        absolute = [abs(value) for value in values]
        thresholds = sorted({0.0, *[round(_quantile(absolute, q), 9) for q in THRESHOLD_QUANTILES if q]})
        current_absolute = abs(float(current[rule]))
        for threshold in thresholds:
            if current_absolute < threshold:
                continue
            indices = [index for index, value in enumerate(absolute) if value >= threshold]
            if len(indices) < minimum_qualified_rows:
                continue
            for orientation in (1, -1):
                outcomes = [
                    float(history[index]["long_pips"] if values[index] * orientation >= 0.0 else history[index]["short_pips"])
                    for index in indices
                ]
                mean = statistics.mean(outcomes)
                standard_error = statistics.stdev(outcomes) / math.sqrt(len(outcomes))
                rank = (mean - standard_error, mean, len(outcomes))
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best = {
                        "rule": rule,
                        "orientation": orientation,
                        "threshold": threshold,
                        "qualified_rows": len(outcomes),
                        "mean_pips": mean,
                        "standard_error_pips": standard_error,
                    }
    return best


def _score_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [float(row["conservative_pips"]) for row in rows if _finite(row.get("conservative_pips")) is not None]
    by_day: dict[str, float] = {}
    for row in rows:
        value = _finite(row.get("conservative_pips"))
        if value is None:
            continue
        day = _utc(row["decision_at_utc"], name="decision").date().isoformat()
        by_day[day] = by_day.get(day, 0.0) + value
    daily = list(by_day.values())
    gains = sum(value for value in values if value > 0.0)
    losses = -sum(value for value in values if value < 0.0)
    return {
        "signals": len(rows),
        "active_days": len(daily),
        "mean_conservative_pips": round(statistics.mean(values), 6) if values else None,
        "net_conservative_pips": round(sum(values), 6),
        "profit_factor": round(gains / losses, 6) if losses > 0.0 else None,
        "profit_factor_infinite": bool(gains > 0.0 and losses == 0.0),
        "positive_day_rate": round(sum(value > 0.0 for value in daily) / len(daily), 6) if daily else 0.0,
        "one_sided_95_student_t_lower_pips": _student_t_lower(values),
        "one_sided_95_daily_lower_pips": _student_t_lower(daily),
        "stop_loss_count": sum(row.get("exit_reason") == "STOP_LOSS" for row in rows),
        "time_close_count": sum(row.get("exit_reason") == "TIME_CLOSE" for row in rows),
    }


def _student_t_lower(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    critical = _student_t_critical(len(values) - 1)
    return round(statistics.mean(values) - critical * statistics.stdev(values) / math.sqrt(len(values)), 6)


def _student_t_critical(df: int) -> float:
    table = {
        1: 6.3138, 2: 2.92, 3: 2.3534, 4: 2.1318, 5: 2.015, 6: 1.9432,
        7: 1.8946, 8: 1.8595, 9: 1.8331, 10: 1.8125, 12: 1.7823,
        15: 1.7531, 20: 1.7247, 25: 1.7081, 30: 1.6973, 40: 1.6839,
        60: 1.6706, 120: 1.6577,
    }
    if df in table:
        return table[df]
    lower = max(key for key in table if key < df) if df > 1 else 1
    upper_keys = [key for key in table if key > df]
    if not upper_keys:
        return 1.644854
    upper = min(upper_keys)
    ratio = (df - lower) / (upper - lower)
    return table[lower] + ratio * (table[upper] - table[lower])


def _parse_candle(value: Mapping[str, Any], *, pair: str) -> BidAskCandle:
    bid = _mapping(value.get("bid"), "bid")
    ask = _mapping(value.get("ask"), "ask")
    return BidAskCandle(
        timestamp_utc=_utc(value.get("time"), name="candle time"),
        pair=pair,
        bid=Ohlc(*[_positive_float(bid.get(key), name=f"bid.{key}") for key in ("o", "h", "l", "c")]),
        ask=Ohlc(*[_positive_float(ask.get(key), name=f"ask.{key}") for key in ("o", "h", "l", "c")]),
    )


def _candle_payload(value: BidAskCandle) -> dict[str, Any]:
    return {
        "timestamp_utc": value.timestamp_utc.isoformat(),
        "pair": value.pair,
        "bid": vars(value.bid),
        "ask": vars(value.ask),
    }


def _ema(values: Sequence[float], span: int) -> list[float]:
    alpha = 2.0 / (span + 1.0)
    output: list[float] = []
    for value in values:
        output.append(float(value) if not output else alpha * float(value) + (1.0 - alpha) * output[-1])
    return output


def _rolling_mean(values: Sequence[float | None], window: int) -> list[float | None]:
    output: list[float | None] = []
    for index in range(len(values)):
        current = values[max(0, index - window + 1):index + 1]
        finite = [float(item) for item in current if item is not None and math.isfinite(float(item))]
        output.append(statistics.mean(finite) if len(finite) >= window else None)
    return output


def _rolling_location(values: Sequence[float], window: int) -> list[float | None]:
    output: list[float | None] = []
    for index in range(len(values)):
        current = values[max(0, index - window + 1):index + 1]
        if len(current) < window or max(current) == min(current):
            output.append(None)
        else:
            output.append((float(values[index]) - min(current)) / (max(current) - min(current)))
    return output


def _rolling_prior_quantiles(
    values: Sequence[float | None],
    *,
    window: int,
    minimum_rows: int,
    quantiles: Sequence[float],
) -> list[tuple[float | None, ...]]:
    """Match ``rolling(window, min_periods).quantile(...).shift(1)``."""

    ordered: list[float] = []
    output: list[tuple[float | None, ...]] = []
    for index, value in enumerate(values):
        if len(ordered) >= minimum_rows:
            output.append(tuple(_quantile(ordered, item) for item in quantiles))
        else:
            output.append(tuple(None for _ in quantiles))
        parsed = _finite(value)
        if parsed is not None:
            bisect.insort(ordered, parsed)
        if index >= window:
            expired = _finite(values[index - window])
            if expired is not None:
                position = bisect.bisect_left(ordered, expired)
                if position >= len(ordered) or ordered[position] != expired:
                    raise ValueError("rolling quantile window lost its expired value")
                ordered.pop(position)
    return output


def _quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("quantile requires values")
    ordered = sorted(map(float, values))
    position = (len(ordered) - 1) * float(quantile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _row_complete(row: Mapping[str, Any]) -> bool:
    return all(_finite(row.get(rule)) is not None for rule in RULES) and bool(row.get("market_phase"))


def _decision_recorded(path: Path, *, decision_id: str, candidate_sha: str) -> bool:
    if not path.exists():
        return False
    for row in load_jsonl(path, contract=SHADOW_CONTRACT, candidate_sha=candidate_sha):
        if row.get("decision_id") == decision_id:
            return True
    return False


def _session(hour: int) -> str:
    return "UTC_00_08" if hour < 8 else "UTC_08_13" if hour < 13 else "UTC_13_17" if hour < 17 else "UTC_17_22" if hour < 22 else "UTC_22_24"


def _divide(numerator: Any, denominator: Any) -> float | None:
    left, right = _finite(numerator), _finite(denominator)
    return None if left is None or right is None or right == 0.0 else left / right


def _negative(value: Any) -> float | None:
    parsed = _finite(value)
    return None if parsed is None else -parsed


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    return None if any(value is None for value in values) else statistics.mean(float(value) for value in values if value is not None)


def _sign(value: Any) -> int:
    parsed = float(value)
    return 1 if parsed > 0.0 else -1 if parsed < 0.0 else 0


def _positive(value: Any) -> bool:
    parsed = _finite(value)
    return parsed is not None and parsed > 0.0


def _greater(value: Any, threshold: Any) -> bool:
    parsed = _finite(value)
    return parsed is not None and parsed > float(threshold)


def _positive_float(value: Any, *, name: str) -> float:
    parsed = _finite(value)
    if parsed is None or parsed <= 0.0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _utc(value: Any, *, name: str) -> datetime:
    text = str(value or "").strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be an aware timestamp") from exc
    return _aware_utc(parsed)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _oanda_time(value: datetime) -> str:
    return _aware_utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _stable_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _seal(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    body = dict(payload)
    body.pop(field, None)
    return {**body, field: _stable_digest(body)}


def _error(code: str, exc: Exception, **context: Any) -> dict[str, Any]:
    return {"code": code, "message": f"{exc.__class__.__name__}: {exc}"[:320], **context}


__all__ = [
    "BidAskCandle",
    "CANDIDATE_CONTRACT",
    "OUTCOME_CONTRACT",
    "Ohlc",
    "SCORECARD_CONTRACT",
    "SHADOW_CONTRACT",
    "build_forward_scorecard",
    "build_forward_shadow",
    "decision_window",
    "emit_forward_shadow_from_oanda",
    "fetch_bidask_candles",
    "load_candidate",
    "resolve_due_outcomes_from_oanda",
]
