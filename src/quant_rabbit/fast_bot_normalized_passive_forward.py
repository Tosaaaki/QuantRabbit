"""Prospective, zero-authority observer for one frozen passive-limit lead.

The historical search artifact is hypothesis-generation evidence only.  This
module records only decisions after the immutable forward lock, reproduces the
frozen hourly M1 selector, and resolves fills/time closes from exact OANDA S5
bid/ask candles.  It deliberately exposes no order intent or gateway adapter.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from quant_rabbit.broker.oanda import OandaReadOnlyClient
from quant_rabbit.contextual_technical_forward import (
    BidAskCandle,
    append_jsonl_once,
    fetch_bidask_candles,
    write_json_atomic,
)
from quant_rabbit.fast_bot_profitability_gate import (
    assess_profitability_evidence,
    build_profitability_evidence,
)
from quant_rabbit.instruments import instrument_pip_factor


POLICY_CONTRACT = "QR_FAST_BOT_NORMALIZED_PASSIVE_FORWARD_POLICY_V1"
DECISION_CONTRACT = "QR_FAST_BOT_NORMALIZED_PASSIVE_FORWARD_DECISION_V1"
OUTCOME_CONTRACT = "QR_FAST_BOT_NORMALIZED_PASSIVE_FORWARD_OUTCOME_V1"
SCORECARD_CONTRACT = "QR_FAST_BOT_NORMALIZED_PASSIVE_FORWARD_SCORECARD_V1"
TRUTH_CONTRACT = "QR_FAST_BOT_PASSIVE_LIMIT_S5_TIME_CLOSE_TRUTH_V1"


class _ReadOnlyClient(Protocol):
    def get_json(
        self,
        path: str,
        query: dict[str, str] | None = None,
    ) -> dict[str, Any]: ...


def load_policy(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("normalized passive forward policy must be an object")
    validate_policy(value)
    return value


def validate_policy(value: Mapping[str, Any]) -> None:
    if value.get("contract") != POLICY_CONTRACT or value.get("schema_version") != 1:
        raise ValueError("normalized passive forward policy contract is invalid")
    safety = {
        "shadow_enabled": True,
        "research_observation_allowed": True,
        "primary_trading_candidate_allowed": False,
        "automatic_adoption_allowed": False,
        "promotion_allowed": False,
        "live_order_enabled": False,
        "execution_authority": "NONE",
        "broker_mutation_allowed": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "manual_tagless_policy": "NO_TOUCH",
    }
    for key, expected in safety.items():
        if value.get(key) != expected:
            raise ValueError(f"policy safety invariant changed: {key}")
    if value.get("broker_http_methods_allowed") != ["GET"]:
        raise ValueError("policy broker method allowlist must contain only GET")

    selector = _mapping(value.get("selector"), "selector")
    expected_selector = {
        "pair": "EUR_USD",
        "candidate_side": "LONG",
        "orientation": "REVERSAL",
        "decision_interval_minutes": 60,
        "decision_candle_minute_utc": 0,
        "lookback_minutes": 15,
        "confirmation_lookback_minutes": 240,
        "normalized_threshold": 1.25,
        "confirmation_policy": "SAME_SIGN_COMPLETED_M1_RETURN",
        "maximum_decision_spread_pips": 1.5,
        "collection_lateness_max_seconds_after_activation": 90,
    }
    for key, expected in expected_selector.items():
        if selector.get(key) != expected:
            raise ValueError(f"frozen selector changed: {key}")

    vehicle = _mapping(value.get("vehicle"), "vehicle")
    expected_vehicle = {
        "entry_vehicle": "PASSIVE_LIMIT_TIME_CLOSE",
        "entry_spread_fraction": 0.25,
        "entry_ttl_minutes": 5,
        "holding_minutes": 240,
        "price_tick": 0.00001,
        "gap_through_fill_policy": "CONSERVATIVE_LIMIT_PRICE_NO_IMPROVEMENT",
        "take_profit_pips": None,
        "stop_loss_pips": None,
    }
    for key, expected in expected_vehicle.items():
        if vehicle.get(key) != expected:
            raise ValueError(f"frozen vehicle changed: {key}")

    resolver = _mapping(value.get("resolver"), "resolver")
    if resolver.get("truth_granularity") != "S5":
        raise ValueError("resolver truth must remain exact S5")
    if not 1 <= int(resolver.get("chunk_candle_limit") or 0) <= 5000:
        raise ValueError("resolver chunk bound is invalid")
    if int(resolver.get("truth_close_grace_seconds") or 0) < 5:
        raise ValueError("resolver close grace is too small")
    if int(resolver.get("max_due_signals_per_run") or 0) < 1:
        raise ValueError("resolver due-signal bound is invalid")

    thresholds = _mapping(value.get("forward_evaluation"), "forward_evaluation")
    expected_thresholds = {
        "minimum_samples": 100,
        "minimum_active_days": 10,
        "minimum_profit_factor": 1.25,
        "minimum_pessimistic_expectancy_pips": 0.0,
        "minimum_positive_day_rate": 2.0 / 3.0,
        "maximum_daily_sample_share": 0.70,
    }
    for key, expected in expected_thresholds.items():
        actual = thresholds.get(key)
        if isinstance(expected, float):
            if actual is None or abs(float(actual) - expected) > 1e-12:
                raise ValueError(f"forward evaluation threshold changed: {key}")
        elif actual != expected:
            raise ValueError(f"forward evaluation threshold changed: {key}")

    cutoff = _utc(value.get("forward_evaluation_not_before_utc"), name="forward lock")
    frozen = _utc(value.get("frozen_at_utc"), name="policy freeze")
    if cutoff < frozen:
        raise ValueError("forward lock predates policy freeze")

    disclosure = _mapping(value.get("selection_disclosure"), "selection_disclosure")
    if disclosure.get("historical_research_only") is not True:
        raise ValueError("historical research disclosure is missing")
    if disclosure.get("historical_holdout_inspected") is not True:
        raise ValueError("holdout disclosure is missing")
    if disclosure.get("selection_used_holdout") is not False:
        raise ValueError("holdout must not have selected the candidate")
    if disclosure.get("multiple_testing_corrected") is not False:
        raise ValueError("multiple-testing disclosure changed")
    if disclosure.get("selection_status") != "HOLDOUT_REJECT":
        raise ValueError("only the disclosed rejected research lead is allowed")
    if disclosure.get("shadow_candidate_admitted") is not False:
        raise ValueError("historical artifact must not admit the candidate")
    artifact_path = Path(str(disclosure.get("source_artifact_path") or ""))
    if not artifact_path.is_absolute() or not artifact_path.is_file():
        raise ValueError("historical source artifact is unavailable")
    artifact_sha = _file_sha256(artifact_path)
    if artifact_sha != disclosure.get("source_artifact_sha256"):
        raise ValueError("historical source artifact hash changed")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError("historical source artifact is invalid")
    if artifact.get("contract_sha256") != disclosure.get("source_contract_sha256"):
        raise ValueError("historical source contract hash changed")
    selection = _mapping(artifact.get("selection"), "historical selection")
    if selection.get("candidate_id") != value.get("candidate_id"):
        raise ValueError("historical selected candidate changed")
    if selection.get("status") != "HOLDOUT_REJECT":
        raise ValueError("historical holdout status changed")
    if selection.get("selection_used_holdout") is not False:
        raise ValueError("historical holdout contaminated selection")
    if selection.get("shadow_candidate_admitted") is not False:
        raise ValueError("historical artifact unexpectedly admitted candidate")
    required_blockers = {
        "HISTORICAL_HOLDOUT_TOO_SMALL",
        "PESSIMISTIC_EXPECTANCY_NOT_POSITIVE",
        "POSITIVE_MONTH_RATE_BELOW_FLOOR",
        "MULTIPLE_TESTING_UNCORRECTED",
        "FRESH_PROSPECTIVE_EVIDENCE_REQUIRED",
        "SEPARATE_LIVE_PROMOTION_CONTRACT_REQUIRED",
    }
    if set(value.get("promotion_blockers") or ()) != required_blockers:
        raise ValueError("policy promotion blockers changed")


def policy_sha256(path: Path) -> str:
    load_policy(path)
    return _file_sha256(path)


def decision_window(
    policy: Mapping[str, Any],
    *,
    as_of_utc: datetime,
) -> dict[str, Any]:
    validate_policy(policy)
    now = _aware_utc(as_of_utc)
    decision = now.replace(minute=0, second=0, microsecond=0)
    activation = decision + timedelta(minutes=1)
    opens = activation + timedelta(seconds=5)
    selector = _mapping(policy["selector"], "selector")
    closes = activation + timedelta(
        seconds=int(selector["collection_lateness_max_seconds_after_activation"])
    )
    cutoff = _utc(policy["forward_evaluation_not_before_utc"], name="forward lock")
    if decision < cutoff:
        status = "BEFORE_FORWARD_LOCK"
    elif now < opens:
        status = "WAITING_FOR_COMPLETE_M1"
    elif now > closes:
        status = "OUTSIDE_COLLECTION_WINDOW"
    else:
        status = "OPEN"
    return {
        "status": status,
        "decision_at_utc": decision.isoformat(),
        "activation_at_utc": activation.isoformat(),
        "collection_opens_at_utc": opens.isoformat(),
        "collection_closes_at_utc": closes.isoformat(),
        "observed_at_utc": now.isoformat(),
    }


def observe_from_oanda(
    *,
    policy_path: Path,
    latest_decision_path: Path,
    decision_ledger_path: Path,
    client_factory: Callable[[], _ReadOnlyClient] = OandaReadOnlyClient,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    now = _aware_utc((clock or _utc_now)())
    policy = load_policy(policy_path)
    policy_sha = _file_sha256(policy_path)
    window = decision_window(policy, as_of_utc=now)
    base = _zero_authority(
        {
            "policy_sha256": policy_sha,
            "candidate_id": policy["candidate_id"],
            **window,
        }
    )
    if window["status"] != "OPEN":
        result = {**base, "status": window["status"], "signals": []}
        write_json_atomic(latest_decision_path, result)
        return result
    decision_at = _utc(window["decision_at_utc"], name="decision")
    activation_at = _utc(window["activation_at_utc"], name="activation")
    decision_id = _stable_digest(
        {"policy_sha256": policy_sha, "decision_at_utc": decision_at.isoformat()}
    )
    decisions = load_decisions(decision_ledger_path, policy_sha256=policy_sha)
    if any(row.get("decision_id") == decision_id for row in decisions):
        result = {
            **base,
            "status": "ALREADY_RECORDED",
            "decision_id": decision_id,
            "signals": [],
        }
        write_json_atomic(latest_decision_path, result)
        return result

    selector = _mapping(policy["selector"], "selector")
    history_from = decision_at - timedelta(
        minutes=int(selector["confirmation_lookback_minutes"])
    )
    try:
        client = client_factory()
        candles, chunk_hashes = fetch_bidask_candles(
            client,
            pair=str(selector["pair"]),
            granularity="M1",
            time_from=history_from,
            time_to=activation_at,
            chunk_candle_limit=int(_mapping(policy["resolver"], "resolver")["chunk_candle_limit"]),
        )
        decision_row = build_decision(
            policy,
            policy_sha256=policy_sha,
            decision_at_utc=decision_at,
            observed_at_utc=now,
            candles=candles,
            prior_decisions=decisions,
            truth_chunk_sha256=chunk_hashes,
        )
    except Exception as exc:
        result = {
            **base,
            "status": "INPUT_UNAVAILABLE_RETRY_WITHIN_WINDOW",
            "decision_id": decision_id,
            "signals": [],
            "errors": [_error("M1_FETCH_OR_DECISION", exc)],
        }
        write_json_atomic(latest_decision_path, result)
        return result
    append_jsonl_once(
        decision_ledger_path,
        decision_row,
        identity_key="decision_id",
        expected_identity=decision_id,
    )
    write_json_atomic(latest_decision_path, decision_row)
    return decision_row


def build_decision(
    policy: Mapping[str, Any],
    *,
    policy_sha256: str,
    decision_at_utc: datetime,
    observed_at_utc: datetime,
    candles: Sequence[BidAskCandle],
    prior_decisions: Sequence[Mapping[str, Any]] = (),
    truth_chunk_sha256: Sequence[str] = (),
) -> dict[str, Any]:
    validate_policy(policy)
    decision = _aware_utc(decision_at_utc)
    observed = _aware_utc(observed_at_utc)
    selector = _mapping(policy["selector"], "selector")
    vehicle = _mapping(policy["vehicle"], "vehicle")
    pair = str(selector["pair"])
    ordered = sorted(candles, key=lambda item: item.timestamp_utc)
    confirmation = int(selector["confirmation_lookback_minutes"])
    expected_times = [
        decision - timedelta(minutes=confirmation - index)
        for index in range(confirmation + 1)
    ]
    if len(ordered) != len(expected_times):
        raise ValueError(
            f"M1_HISTORY_COUNT_MISMATCH expected={len(expected_times)} actual={len(ordered)}"
        )
    if [item.timestamp_utc for item in ordered] != expected_times:
        raise ValueError("M1_HISTORY_NOT_CONTIGUOUS")
    if any(item.pair != pair for item in ordered):
        raise ValueError("M1_HISTORY_PAIR_MISMATCH")

    decision_id = _stable_digest(
        {"policy_sha256": policy_sha256, "decision_at_utc": decision.isoformat()}
    )
    activation = decision + timedelta(minutes=1)
    lookback = int(selector["lookback_minutes"])
    factor = instrument_pip_factor(pair)
    mids = [(item.bid.c + item.ask.c) / 2.0 for item in ordered]
    minute_returns = [0.0] + [
        (mids[index] - mids[index - 1]) * factor
        for index in range(1, len(mids))
    ]
    short_return = (mids[-1] - mids[-1 - lookback]) * factor
    confirmation_return = (mids[-1] - mids[0]) * factor
    variation = math.sqrt(sum(value * value for value in minute_returns[-lookback:]))
    normalized = abs(short_return) / variation if variation > 0.0 else 0.0
    source_direction = "UP" if short_return > 0.0 else "DOWN" if short_return < 0.0 else "FLAT"
    qualifying = bool(
        source_direction != "FLAT"
        and short_return * confirmation_return > 0.0
        and normalized >= float(selector["normalized_threshold"])
    )
    latest = ordered[-1]
    spread_pips = (latest.ask.c - latest.bid.c) * factor
    reservation = _active_reservation(prior_decisions, activation_at_utc=activation)
    signals: list[dict[str, Any]] = []
    reservation_until: datetime | None = None
    result_status = "NO_QUALIFYING_RETURN"
    selected_side: str | None = None
    limit_price: float | None = None

    if qualifying and reservation is not None:
        result_status = "SKIPPED_OVERLAP"
    elif qualifying and (
        not latest.bid.c < latest.ask.c
        or spread_pips > float(selector["maximum_decision_spread_pips"])
    ):
        result_status = "SPREAD_CAP_REJECT"
    elif qualifying:
        oriented = -1 if source_direction == "UP" else 1
        selected_side = "LONG" if oriented > 0 else "SHORT"
        reservation_until = activation + timedelta(
            minutes=int(vehicle["entry_ttl_minutes"]) + int(vehicle["holding_minutes"])
        )
        if selected_side != selector["candidate_side"]:
            result_status = "SIDE_FILTERED_RESERVED"
        else:
            width = latest.ask.c - latest.bid.c
            raw_limit = latest.bid.c + width * float(vehicle["entry_spread_fraction"])
            tick = float(vehicle["price_tick"])
            limit_price = math.floor(raw_limit / tick + 1e-9) * tick
            fill_end = activation + timedelta(minutes=int(vehicle["entry_ttl_minutes"]))
            maturity = reservation_until
            signal_body = _zero_authority(
                {
                    "decision_id": decision_id,
                    "decision_at_utc": decision.isoformat(),
                    "activation_at_utc": activation.isoformat(),
                    "fill_window_end_exclusive_utc": fill_end.isoformat(),
                    "maturity_at_utc": maturity.isoformat(),
                    "pair": pair,
                    "side": "LONG",
                    "direction": "UP",
                    "entry_vehicle": "PASSIVE_LIMIT_TIME_CLOSE",
                    "entry_price": round(limit_price, 9),
                    "entry_ttl_minutes": int(vehicle["entry_ttl_minutes"]),
                    "holding_minutes": int(vehicle["holding_minutes"]),
                    "take_profit_pips": None,
                    "stop_loss_pips": None,
                    "order_intents": [],
                }
            )
            signals.append(_seal(signal_body, "signal_sha256"))
            result_status = "EMITTED"

    body = _zero_authority(
        {
            "contract": DECISION_CONTRACT,
            "schema_version": 1,
            "policy_sha256": policy_sha256,
            "candidate_id": policy["candidate_id"],
            "decision_id": decision_id,
            "decision_at_utc": decision.isoformat(),
            "activation_at_utc": activation.isoformat(),
            "observed_at_utc": observed.isoformat(),
            "status": result_status,
            "qualifying_return": qualifying,
            "source_direction": source_direction,
            "selected_side": selected_side,
            "short_return_pips": round(short_return, 9),
            "confirmation_return_pips": round(confirmation_return, 9),
            "path_variation_pips": round(variation, 9),
            "normalized_return": round(normalized, 9),
            "decision_bid_close": round(latest.bid.c, 9),
            "decision_ask_close": round(latest.ask.c, 9),
            "decision_spread_pips": round(spread_pips, 9),
            "limit_price": round(limit_price, 9) if limit_price is not None else None,
            "overlap_reservation_source_decision_id": (
                str(reservation.get("decision_id")) if reservation else None
            ),
            "reservation_until_utc": reservation_until.isoformat() if reservation_until else None,
            "m1_candle_count": len(ordered),
            "m1_from_utc": ordered[0].timestamp_utc.isoformat(),
            "m1_to_utc": ordered[-1].timestamp_utc.isoformat(),
            "m1_truth_chunk_sha256": list(truth_chunk_sha256),
            "m1_candles_sha256": _stable_digest([_candle_payload(item) for item in ordered]),
            "signals": signals,
            "order_intents": [],
        }
    )
    return _seal(body, "decision_sha256")


def resolve_due_outcomes_from_oanda(
    *,
    policy_path: Path,
    decision_ledger_path: Path,
    outcome_ledger_path: Path,
    scorecard_path: Path,
    client_factory: Callable[[], _ReadOnlyClient] = OandaReadOnlyClient,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    now = _aware_utc((clock or _utc_now)())
    policy = load_policy(policy_path)
    policy_sha = _file_sha256(policy_path)
    decisions = load_decisions(decision_ledger_path, policy_sha256=policy_sha)
    outcomes = load_outcomes(outcome_ledger_path, policy_sha256=policy_sha)
    resolved = {str(row["signal_sha256"]) for row in outcomes}
    resolver = _mapping(policy["resolver"], "resolver")
    grace = timedelta(seconds=int(resolver["truth_close_grace_seconds"]))
    due = [
        signal
        for decision in decisions
        for signal in decision.get("signals", [])
        if str(signal.get("signal_sha256")) not in resolved
        and now >= _utc(signal.get("maturity_at_utc"), name="maturity") + grace
    ]
    due.sort(key=lambda row: (str(row["maturity_at_utc"]), str(row["signal_sha256"])))
    due = due[: int(resolver["max_due_signals_per_run"])]
    errors: list[dict[str, Any]] = []
    appended = 0
    broker_read = False
    if due:
        client = client_factory()
        broker_read = True
        for signal in due:
            try:
                activation = _utc(signal["activation_at_utc"], name="activation")
                maturity = _utc(signal["maturity_at_utc"], name="maturity")
                candles, chunk_hashes = fetch_bidask_candles(
                    client,
                    pair=str(signal["pair"]),
                    granularity="S5",
                    time_from=activation,
                    time_to=maturity,
                    chunk_candle_limit=int(resolver["chunk_candle_limit"]),
                )
                outcome = resolve_signal(
                    signal,
                    candles,
                    policy_sha256=policy_sha,
                    resolved_at_utc=now,
                    truth_chunk_sha256=chunk_hashes,
                )
                if append_jsonl_once(
                    outcome_ledger_path,
                    outcome,
                    identity_key="signal_sha256",
                    expected_identity=str(signal["signal_sha256"]),
                ):
                    appended += 1
            except Exception as exc:
                errors.append(
                    _error(
                        "S5_FETCH_OR_RESOLUTION",
                        exc,
                        signal_sha256=str(signal.get("signal_sha256") or ""),
                    )
                )
    outcomes = load_outcomes(outcome_ledger_path, policy_sha256=policy_sha)
    scorecard = build_scorecard(
        policy,
        decisions,
        outcomes,
        policy_sha256=policy_sha,
        as_of_utc=now,
        acquisition_errors=errors,
    )
    write_json_atomic(scorecard_path, scorecard)
    return _zero_authority(
        {
            "status": (
                "NO_DUE_SIGNALS"
                if not due
                else "RESOLVED_WITH_ERRORS"
                if errors
                else "RESOLVED"
            ),
            "broker_read": broker_read,
            "selected_due_count": len(due),
            "ledger_appended_count": appended,
            "scorecard_status": scorecard["status"],
            "prospective_gate_passed": scorecard["prospective_gate_passed"],
            "errors": errors,
        }
    )


def resolve_signal(
    signal: Mapping[str, Any],
    candles: Sequence[BidAskCandle],
    *,
    policy_sha256: str,
    resolved_at_utc: datetime,
    truth_chunk_sha256: Sequence[str] = (),
) -> dict[str, Any]:
    if signal.get("side") != "LONG" or signal.get("direction") != "UP":
        raise ValueError("frozen forward signal side is invalid")
    pair = str(signal.get("pair") or "")
    activation = _utc(signal.get("activation_at_utc"), name="activation")
    fill_end = _utc(signal.get("fill_window_end_exclusive_utc"), name="fill end")
    maturity = _utc(signal.get("maturity_at_utc"), name="maturity")
    limit = float(signal.get("entry_price"))
    ordered = sorted(candles, key=lambda item: item.timestamp_utc)
    if len({item.timestamp_utc for item in ordered}) != len(ordered):
        raise ValueError("duplicate S5 truth timestamps")
    if any(item.pair != pair for item in ordered):
        raise ValueError("S5 truth pair mismatch")
    if any(int(item.timestamp_utc.timestamp()) % 5 != 0 for item in ordered):
        raise ValueError("S5 truth timestamp is off grid")
    if any(not activation <= item.timestamp_utc < maturity for item in ordered):
        raise ValueError("S5 truth lies outside frozen interval")
    touches = [
        item
        for item in ordered
        if activation <= item.timestamp_utc < fill_end and item.ask.l <= limit
    ]
    common = {
        "contract": OUTCOME_CONTRACT,
        "schema_version": 1,
        "policy_sha256": policy_sha256,
        "decision_id": signal["decision_id"],
        "signal_sha256": signal["signal_sha256"],
        "pair": pair,
        "side": "LONG",
        "decision_at_utc": signal["decision_at_utc"],
        "activation_at_utc": signal["activation_at_utc"],
        "maturity_at_utc": signal["maturity_at_utc"],
        "resolved_at_utc": _aware_utc(resolved_at_utc).isoformat(),
        "truth_contract": TRUTH_CONTRACT,
        "truth_chunk_sha256": list(truth_chunk_sha256),
        "truth_candle_count": len(ordered),
        "truth_candles_sha256": _stable_digest([_candle_payload(item) for item in ordered]),
        "entry_price": round(limit, 9),
        "order_intents": [],
    }
    if not touches:
        return _seal(
            _zero_authority(
                {
                    **common,
                    "status": "UNFILLED",
                    "filled": False,
                    "fill_at_utc": None,
                    "exit_at_utc": None,
                    "exit_price": None,
                    "realized_pips": None,
                }
            ),
            "outcome_sha256",
        )
    fill = touches[0]
    fill_minute = fill.timestamp_utc.replace(second=0, microsecond=0)
    exit_at = fill_minute + timedelta(minutes=int(signal["holding_minutes"]))
    exit_candle = next((item for item in ordered if item.timestamp_utc == exit_at), None)
    if exit_candle is None:
        raise ValueError("exact S5 time-close quote is missing")
    exit_price = exit_candle.bid.o
    realized = (exit_price - limit) * instrument_pip_factor(pair)
    return _seal(
        _zero_authority(
            {
                **common,
                "status": "FILLED_TIME_CLOSE",
                "filled": True,
                "fill_at_utc": fill.timestamp_utc.isoformat(),
                "fill_minute_at_utc": fill_minute.isoformat(),
                "exit_at_utc": exit_at.isoformat(),
                "exit_price": round(exit_price, 9),
                "realized_pips": round(realized, 6),
                "gap_through_fill": bool(fill.ask.o < limit),
            }
        ),
        "outcome_sha256",
    )


def build_scorecard(
    policy: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    *,
    policy_sha256: str,
    as_of_utc: datetime,
    acquisition_errors: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    validate_policy(policy)
    signals = sorted(
        [signal for decision in decisions for signal in decision.get("signals", [])],
        key=lambda row: (str(row["activation_at_utc"]), str(row["signal_sha256"])),
    )
    outcome_by_signal = {str(row["signal_sha256"]): row for row in outcomes}
    resolved = [
        outcome_by_signal[str(row["signal_sha256"])]
        for row in signals
        if str(row["signal_sha256"]) in outcome_by_signal
    ]
    filled = [row for row in resolved if row.get("filled") is True]
    metrics = _prospective_metrics(filled)
    disclosure = _mapping(policy["selection_disclosure"], "selection_disclosure")
    evidence = build_profitability_evidence(
        lane_id=str(policy["candidate_id"]),
        pair="EUR_USD",
        side="LONG",
        method="NORMALIZED_RETURN_REVERSAL",
        order_type="LIMIT",
        metrics=metrics,
        source_artifact_sha256=str(disclosure["source_artifact_sha256"]),
        generated_at_utc=_aware_utc(as_of_utc),
        evidence_end_utc=_aware_utc(as_of_utc),
    )
    gate = assess_profitability_evidence(
        evidence,
        thresholds=_mapping(policy["forward_evaluation"], "forward_evaluation"),
    )
    passed = gate.get("status") == "SHADOW_FORWARD_OBSERVATION_READY"
    if passed:
        status = "PROSPECTIVE_EVIDENCE_PASSED_REVIEW_REQUIRED"
    elif signals:
        status = "COLLECTING_PROSPECTIVE_EVIDENCE"
    else:
        status = "COLLECTING_NO_PROSPECTIVE_SIGNALS"
    body = _zero_authority(
        {
            "contract": SCORECARD_CONTRACT,
            "schema_version": 1,
            "policy_sha256": policy_sha256,
            "candidate_id": policy["candidate_id"],
            "generated_at_utc": _aware_utc(as_of_utc).isoformat(),
            "status": status,
            "historical_profitability_claim_allowed": False,
            "prospective_gate_passed": passed,
            "research_observation_continues": True,
            "decision_count": len(decisions),
            "emitted_signal_count": len(signals),
            "resolved_signal_count": len(resolved),
            "filled_signal_count": len(filled),
            "unfilled_signal_count": sum(row.get("filled") is False for row in resolved),
            "metrics": metrics,
            "profitability_evidence": evidence,
            "profitability_gate": gate,
            "historical_promotion_blockers": list(policy.get("promotion_blockers") or ()),
            "acquisition_errors": [dict(item) for item in acquisition_errors][:20],
            "order_intents": [],
        }
    )
    return _seal(body, "scorecard_sha256")


def load_decisions(path: Path, *, policy_sha256: str) -> list[dict[str, Any]]:
    rows = _load_sealed_jsonl(
        path,
        contract=DECISION_CONTRACT,
        policy_sha256=policy_sha256,
        seal_field="decision_sha256",
        identity_field="decision_id",
    )
    for row in rows:
        for signal in row.get("signals", []):
            if not isinstance(signal, dict):
                raise ValueError("decision signal is invalid")
            seal = signal.get("signal_sha256")
            body = {key: value for key, value in signal.items() if key != "signal_sha256"}
            if not isinstance(seal, str) or seal != _stable_digest(body):
                raise ValueError("decision signal seal mismatch")
    return rows


def load_outcomes(path: Path, *, policy_sha256: str) -> list[dict[str, Any]]:
    return _load_sealed_jsonl(
        path,
        contract=OUTCOME_CONTRACT,
        policy_sha256=policy_sha256,
        seal_field="outcome_sha256",
        identity_field="signal_sha256",
    )


def _load_sealed_jsonl(
    path: Path,
    *,
    contract: str,
    policy_sha256: str,
    seal_field: str,
    identity_field: str,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or value.get("contract") != contract:
                raise ValueError(f"ledger contract mismatch at line {number}")
            if value.get("policy_sha256") != policy_sha256:
                raise ValueError(f"ledger policy mismatch at line {number}")
            seal = value.get(seal_field)
            body = {key: item for key, item in value.items() if key != seal_field}
            if not isinstance(seal, str) or seal != _stable_digest(body):
                raise ValueError(f"ledger seal mismatch at line {number}")
            identity = str(value.get(identity_field) or "")
            if not identity or identity in seen:
                raise ValueError(f"ledger identity missing or duplicated at line {number}")
            seen.add(identity)
            rows.append(value)
    return rows


def _active_reservation(
    decisions: Sequence[Mapping[str, Any]],
    *,
    activation_at_utc: datetime,
) -> Mapping[str, Any] | None:
    active = []
    for decision in decisions:
        raw = decision.get("reservation_until_utc")
        if raw and activation_at_utc < _utc(raw, name="reservation"):
            active.append(decision)
    return max(active, key=lambda row: str(row["reservation_until_utc"])) if active else None


def _prospective_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [float(row["realized_pips"]) for row in rows]
    by_day_values: dict[str, list[float]] = {}
    for row, value in zip(rows, values):
        day = _utc(row["fill_at_utc"], name="fill").date().isoformat()
        by_day_values.setdefault(day, []).append(value)
    by_day_net = [sum(items) for items in by_day_values.values()]
    gains = sum(value for value in values if value > 0.0)
    losses = -sum(value for value in values if value < 0.0)
    if losses > 0.0:
        profit_factor: float | str = round(gains / losses, 6)
    elif gains > 0.0:
        profit_factor = "INF"
    else:
        profit_factor = 0.0
    pessimistic = _pessimistic_expectancy(values)
    sample_count = len(values)
    return {
        "sample_count": sample_count,
        "active_days": len(by_day_values),
        "wins": sum(value > 0.0 for value in values),
        "losses": sum(value < 0.0 for value in values),
        "net_pips": round(sum(values), 6),
        "expectancy_pips": round(statistics.mean(values), 6) if values else 0.0,
        "pessimistic_expectancy_pips": round(pessimistic, 6) if pessimistic is not None else None,
        "profit_factor": profit_factor,
        "positive_day_rate": (
            round(sum(value > 0.0 for value in by_day_net) / len(by_day_net), 6)
            if by_day_net
            else 0.0
        ),
        "max_daily_sample_share": (
            round(max(len(items) for items in by_day_values.values()) / sample_count, 6)
            if sample_count
            else 0.0
        ),
        "spread_included": True,
    }


def _pessimistic_expectancy(values: Sequence[float]) -> float | None:
    if not values:
        return None
    wins = [value for value in values if value > 0.0]
    losses = [-value for value in values if value < 0.0]
    observed = len(wins) / len(values)
    z = 1.96
    denominator = 1.0 + z * z / len(values)
    center = observed + z * z / (2.0 * len(values))
    margin = z * math.sqrt(
        (observed * (1.0 - observed) + z * z / (4.0 * len(values))) / len(values)
    )
    lower = max(0.0, (center - margin) / denominator)
    average_win = statistics.mean(wins) if wins else 0.0
    average_loss = statistics.mean(losses) if losses else 0.0
    return lower * average_win - (1.0 - lower) * average_loss


def _zero_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(value),
        "shadow_only": True,
        "execution_authority": "NONE",
        "broker_http_methods_allowed": ["GET"],
        "broker_mutation_allowed": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "gateway_invocations": 0,
        "live_permission": False,
        "promotion_allowed": False,
        "automatic_adoption_allowed": False,
        "primary_trading_candidate_allowed": False,
        "manual_tagless_policy": "NO_TOUCH",
    }


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != field}
    return {**body, field: _stable_digest(body)}


def _stable_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candle_payload(value: BidAskCandle) -> dict[str, Any]:
    return {
        "timestamp_utc": value.timestamp_utc.isoformat(),
        "pair": value.pair,
        "bid": vars(value.bid),
        "ask": vars(value.ask),
    }


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _utc(value: Any, *, name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} timestamp is missing")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return _aware_utc(parsed)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone aware")
    return value.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _error(scope: str, exc: Exception, **extra: str) -> dict[str, Any]:
    return {
        "scope": scope,
        "error_type": type(exc).__name__,
        "message": str(exc)[:500],
        **extra,
    }
