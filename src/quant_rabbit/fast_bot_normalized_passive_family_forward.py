"""Prospective, zero-authority observer for a frozen candidate family.

V2 keeps the rejected EUR/USD anchor visible and adds one independently
selected USD/JPY exploratory hypothesis.  Both candidates are fixed before
the new cutoff, evaluated on separate append-only ledgers, and share a
Bonferroni-corrected family decision.  Interim futility may stop research
collection, but no result can create an order intent or live permission.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import NormalDist
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


POLICY_CONTRACT = "QR_FAST_BOT_NORMALIZED_PASSIVE_FORWARD_POLICY_V2"
DECISION_CONTRACT = "QR_FAST_BOT_NORMALIZED_PASSIVE_FAMILY_DECISION_V2"
OUTCOME_CONTRACT = "QR_FAST_BOT_NORMALIZED_PASSIVE_FAMILY_OUTCOME_V2"
SCORECARD_CONTRACT = "QR_FAST_BOT_NORMALIZED_PASSIVE_FAMILY_SCORECARD_V2"
TRUTH_CONTRACT = "QR_FAST_BOT_PASSIVE_LIMIT_S5_TIME_CLOSE_TRUTH_V1"

ANCHOR_ID = (
    "EUR_USD:LONG:REVERSAL:LB_15M:CLB_240M:Z_1.25:"
    "F_0.25:TTL_5M:HOLD_240M"
)
EXPLORATORY_ID = (
    "USD_JPY:LONG:MOMENTUM:LB_60M:CLB_240M:Z_2.0:"
    "F_0.5:TTL_5M:HOLD_240M"
)
TERMINAL_CANDIDATE_STATUSES = {
    "FUTILITY_REJECTED_COLLECTION_STOPPED",
    "FINAL_EVIDENCE_REJECTED_COLLECTION_STOPPED",
    "FINAL_EVIDENCE_PASSED_REVIEW_REQUIRED",
}


class _ReadOnlyClient(Protocol):
    def get_json(
        self,
        path: str,
        query: dict[str, str] | None = None,
    ) -> dict[str, Any]: ...


def load_policy(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("normalized passive family policy must be an object")
    validate_policy(value)
    return value


def validate_policy(value: Mapping[str, Any]) -> None:
    if value.get("contract") != POLICY_CONTRACT or value.get("schema_version") != 2:
        raise ValueError("normalized passive family policy contract is invalid")
    expected_safety = {
        "shadow_enabled": True,
        "research_observation_allowed": True,
        "primary_trading_candidate_allowed": False,
        "automatic_adoption_allowed": False,
        "automatic_replacement_allowed": False,
        "promotion_allowed": False,
        "live_order_enabled": False,
        "execution_authority": "NONE",
        "broker_mutation_allowed": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "manual_tagless_policy": "NO_TOUCH",
    }
    for key, expected in expected_safety.items():
        if value.get(key) != expected:
            raise ValueError(f"family policy safety invariant changed: {key}")
    if value.get("broker_http_methods_allowed") != ["GET"]:
        raise ValueError("family policy broker method allowlist must contain only GET")
    if value.get("family_id") != "NORMALIZED_PASSIVE_TWO_CANDIDATE_FAMILY_V2":
        raise ValueError("family identity changed")

    frozen = _utc(value.get("frozen_at_utc"), name="policy freeze")
    cutoff = _utc(value.get("forward_evaluation_not_before_utc"), name="forward lock")
    if cutoff < frozen:
        raise ValueError("forward lock predates policy freeze")

    resolver = _mapping(value.get("resolver"), "resolver")
    expected_resolver = {
        "truth_granularity": "S5",
        "chunk_candle_limit": 5000,
        "truth_close_grace_seconds": 30,
        "max_due_signals_per_run": 4,
    }
    if dict(resolver) != expected_resolver:
        raise ValueError("family resolver changed")

    evaluation = _mapping(value.get("family_evaluation"), "family_evaluation")
    expected_evaluation = {
        "candidate_count": 2,
        "familywise_alpha": 0.05,
        "multiple_testing_correction": "BONFERRONI",
        "per_candidate_final_alpha": 0.025,
        "fixed_final_sample_prefix": 100,
        "interim_futility_looks": [30, 60],
        "interim_test_count": 4,
        "per_interim_test_alpha": 0.0125,
        "futility_rule": (
            "STOP_IF_BONFERRONI_WILSON_OPTIMISTIC_EXPECTANCY_NOT_POSITIVE"
        ),
        "winner_revival_after_fixed_prefix": False,
        "replacement_policy": "NEW_VERSION_AND_UNTOUCHED_FUTURE_CUTOFF_REQUIRED",
    }
    if dict(evaluation) != expected_evaluation:
        raise ValueError("family evaluation contract changed")

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

    disclosure = _mapping(value.get("selection_disclosure"), "selection_disclosure")
    expected_disclosure = {
        "historical_research_only": True,
        "historical_search_candidate_count": 3456,
        "historical_search_multiple_testing_corrected": False,
        "v2_selection_used_holdout": False,
        "strict_pre_holdout_qualified_count": 1,
    }
    for key, expected in expected_disclosure.items():
        if disclosure.get(key) != expected:
            raise ValueError(f"family selection disclosure changed: {key}")
    artifact_path = Path(str(disclosure.get("source_artifact_path") or ""))
    if not artifact_path.is_absolute() or not artifact_path.is_file():
        raise ValueError("family historical source artifact is unavailable")
    if _file_sha256(artifact_path) != disclosure.get("source_artifact_sha256"):
        raise ValueError("family historical source artifact hash changed")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError("family historical source artifact is invalid")
    if artifact.get("contract_sha256") != disclosure.get("source_contract_sha256"):
        raise ValueError("family historical source contract hash changed")
    if artifact.get("pre_holdout_qualified_count") != 1:
        raise ValueError("historical strict-qualified count changed")

    candidates = value.get("candidates")
    if not isinstance(candidates, list) or [
        row.get("candidate_id") if isinstance(row, Mapping) else None
        for row in candidates
    ] != [ANCHOR_ID, EXPLORATORY_ID]:
        raise ValueError("frozen candidate family changed")
    expected_candidates = _expected_candidates()
    historical_rows = {
        str(row.get("candidate_id")): row
        for row in artifact.get("pre_holdout_candidates", [])
        if isinstance(row, Mapping)
    }
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        expected = expected_candidates[candidate_id]
        for key in (
            "candidate_role",
            "historical_holdout_inspected",
            "historical_status",
            "strict_pre_holdout_qualified",
        ):
            if candidate.get(key) != expected[key]:
                raise ValueError(f"candidate disclosure changed: {candidate_id}:{key}")
        if dict(_mapping(candidate.get("selector"), "candidate selector")) != expected[
            "selector"
        ]:
            raise ValueError(f"candidate selector changed: {candidate_id}")
        if dict(_mapping(candidate.get("vehicle"), "candidate vehicle")) != expected[
            "vehicle"
        ]:
            raise ValueError(f"candidate vehicle changed: {candidate_id}")
        historical = historical_rows.get(candidate_id)
        if not isinstance(historical, Mapping):
            raise ValueError(f"candidate missing from historical artifact: {candidate_id}")
        if historical.get("pre_holdout_qualified") is not candidate.get(
            "strict_pre_holdout_qualified"
        ):
            raise ValueError(f"candidate qualification drifted: {candidate_id}")
        _match_metric_subset(
            candidate.get("historical_train"), historical.get("train"), "train"
        )
        _match_metric_subset(
            candidate.get("historical_validation"),
            historical.get("validation"),
            "validation",
        )

    selection = _mapping(artifact.get("selection"), "historical selection")
    anchor = candidates[0]
    if selection.get("candidate_id") != ANCHOR_ID:
        raise ValueError("historical anchor changed")
    if selection.get("status") != "HOLDOUT_REJECT":
        raise ValueError("historical anchor holdout status changed")
    _match_metric_subset(
        anchor.get("historical_holdout"), selection.get("holdout"), "holdout"
    )
    exploratory = candidates[1]
    if exploratory.get("historical_holdout") is not None:
        raise ValueError("exploratory candidate holdout must remain unopened")
    if "holdout" in historical_rows[EXPLORATORY_ID]:
        raise ValueError("public pre-holdout row unexpectedly contains holdout")
    _validate_exploratory_selection(disclosure, artifact, EXPLORATORY_ID)

    required_blockers = {
        "HISTORICAL_ANCHOR_HOLDOUT_REJECTED",
        "EXPLORATORY_CANDIDATE_BELOW_STRICT_SAMPLE_FLOOR",
        "HISTORICAL_SEARCH_MULTIPLE_TESTING_UNCORRECTED",
        "FAMILYWISE_PROSPECTIVE_EVIDENCE_REQUIRED",
        "SEPARATE_LIVE_PROMOTION_CONTRACT_REQUIRED",
    }
    if set(value.get("promotion_blockers") or ()) != required_blockers:
        raise ValueError("family promotion blockers changed")


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
    lateness = int(
        _mapping(policy["candidates"][0]["selector"], "selector")
        ["collection_lateness_max_seconds_after_activation"]
    )
    closes = activation + timedelta(seconds=lateness)
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
    outcome_ledger_path: Path,
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
            "family_id": policy["family_id"],
            **window,
        }
    )
    if window["status"] != "OPEN":
        result = {**base, "status": window["status"], "candidate_results": []}
        write_json_atomic(latest_decision_path, result)
        return result

    decisions = load_decisions(decision_ledger_path, policy_sha256=policy_sha)
    outcomes = load_outcomes(outcome_ledger_path, policy_sha256=policy_sha)
    current_scorecard = build_scorecard(
        policy,
        decisions,
        outcomes,
        policy_sha256=policy_sha,
        as_of_utc=now,
    )
    terminal = set(current_scorecard["terminal_candidate_ids"])
    decision_at = _utc(window["decision_at_utc"], name="decision")
    activation_at = _utc(window["activation_at_utc"], name="activation")
    candidate_results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    client: _ReadOnlyClient | None = None
    for candidate in policy["candidates"]:
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in terminal:
            candidate_status = next(
                row["status"]
                for row in current_scorecard["candidate_scorecards"]
                if row["candidate_id"] == candidate_id
            )
            candidate_results.append(
                {
                    "candidate_id": candidate_id,
                    "status": "COLLECTION_STOPPED",
                    "terminal_status": candidate_status,
                }
            )
            continue
        decision_id = _stable_digest(
            {
                "policy_sha256": policy_sha,
                "candidate_id": candidate_id,
                "decision_at_utc": decision_at.isoformat(),
            }
        )
        if any(row.get("decision_id") == decision_id for row in decisions):
            candidate_results.append(
                {
                    "candidate_id": candidate_id,
                    "status": "ALREADY_RECORDED",
                    "decision_id": decision_id,
                }
            )
            continue
        selector = _mapping(candidate["selector"], "selector")
        history_from = decision_at - timedelta(
            minutes=int(selector["confirmation_lookback_minutes"])
        )
        try:
            if client is None:
                client = client_factory()
            candles, chunk_hashes = fetch_bidask_candles(
                client,
                pair=str(selector["pair"]),
                granularity="M1",
                time_from=history_from,
                time_to=activation_at,
                chunk_candle_limit=int(policy["resolver"]["chunk_candle_limit"]),
            )
            row = build_decision(
                policy,
                candidate,
                policy_sha256=policy_sha,
                decision_at_utc=decision_at,
                observed_at_utc=now,
                candles=candles,
                prior_decisions=decisions,
                truth_chunk_sha256=chunk_hashes,
            )
            append_jsonl_once(
                decision_ledger_path,
                row,
                identity_key="decision_id",
                expected_identity=decision_id,
            )
            decisions.append(row)
            candidate_results.append(_bounded_candidate_decision(row))
        except Exception as exc:
            error = _error(
                "M1_FETCH_OR_DECISION",
                exc,
                candidate_id=candidate_id,
                decision_id=decision_id,
            )
            errors.append(error)
            candidate_results.append(
                {
                    "candidate_id": candidate_id,
                    "decision_id": decision_id,
                    "status": "INPUT_UNAVAILABLE_RETRY_WITHIN_WINDOW",
                    "error": error,
                }
            )
    result = _zero_authority(
        {
            **window,
            "policy_sha256": policy_sha,
            "family_id": policy["family_id"],
            "status": "RECORDED_WITH_ERRORS" if errors else "RECORDED",
            "candidate_results": candidate_results,
            "errors": errors,
        }
    )
    write_json_atomic(latest_decision_path, result)
    return result


def build_decision(
    policy: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    policy_sha256: str,
    decision_at_utc: datetime,
    observed_at_utc: datetime,
    candles: Sequence[BidAskCandle],
    prior_decisions: Sequence[Mapping[str, Any]] = (),
    truth_chunk_sha256: Sequence[str] = (),
) -> dict[str, Any]:
    validate_policy(policy)
    candidate_id = str(candidate.get("candidate_id") or "")
    if candidate_id not in {row["candidate_id"] for row in policy["candidates"]}:
        raise ValueError("candidate is outside frozen family")
    decision = _aware_utc(decision_at_utc)
    observed = _aware_utc(observed_at_utc)
    selector = _mapping(candidate["selector"], "selector")
    vehicle = _mapping(candidate["vehicle"], "vehicle")
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
        {
            "policy_sha256": policy_sha256,
            "candidate_id": candidate_id,
            "decision_at_utc": decision.isoformat(),
        }
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
    source_direction = (
        "UP" if short_return > 0.0 else "DOWN" if short_return < 0.0 else "FLAT"
    )
    qualifying = bool(
        source_direction != "FLAT"
        and short_return * confirmation_return > 0.0
        and normalized >= float(selector["normalized_threshold"])
    )
    latest = ordered[-1]
    spread_pips = (latest.ask.c - latest.bid.c) * factor
    reservation = _active_reservation(
        prior_decisions,
        candidate_id=candidate_id,
        activation_at_utc=activation,
    )
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
        source_sign = 1 if source_direction == "UP" else -1
        oriented_sign = (
            source_sign if selector["orientation"] == "MOMENTUM" else -source_sign
        )
        selected_side = "LONG" if oriented_sign > 0 else "SHORT"
        reservation_until = activation + timedelta(
            minutes=int(vehicle["entry_ttl_minutes"])
            + int(vehicle["holding_minutes"])
        )
        if selected_side != selector["candidate_side"]:
            result_status = "SIDE_FILTERED_RESERVED"
        else:
            width = latest.ask.c - latest.bid.c
            fraction = float(vehicle["entry_spread_fraction"])
            raw_limit = (
                latest.bid.c + width * fraction
                if selected_side == "LONG"
                else latest.ask.c - width * fraction
            )
            tick = float(vehicle["price_tick"])
            limit_price = (
                math.floor(raw_limit / tick + 1e-9) * tick
                if selected_side == "LONG"
                else math.ceil(raw_limit / tick - 1e-9) * tick
            )
            fill_end = activation + timedelta(minutes=int(vehicle["entry_ttl_minutes"]))
            signal_body = _zero_authority(
                {
                    "policy_sha256": policy_sha256,
                    "family_id": policy["family_id"],
                    "candidate_id": candidate_id,
                    "decision_id": decision_id,
                    "decision_at_utc": decision.isoformat(),
                    "activation_at_utc": activation.isoformat(),
                    "fill_window_end_exclusive_utc": fill_end.isoformat(),
                    "maturity_at_utc": reservation_until.isoformat(),
                    "pair": pair,
                    "side": selected_side,
                    "direction": "UP" if selected_side == "LONG" else "DOWN",
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
            "schema_version": 2,
            "policy_sha256": policy_sha256,
            "family_id": policy["family_id"],
            "candidate_id": candidate_id,
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
            "reservation_until_utc": (
                reservation_until.isoformat() if reservation_until else None
            ),
            "m1_candle_count": len(ordered),
            "m1_from_utc": ordered[0].timestamp_utc.isoformat(),
            "m1_to_utc": ordered[-1].timestamp_utc.isoformat(),
            "m1_truth_chunk_sha256": list(truth_chunk_sha256),
            "m1_candles_sha256": _stable_digest(
                [_candle_payload(item) for item in ordered]
            ),
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
    grace = timedelta(seconds=int(policy["resolver"]["truth_close_grace_seconds"]))
    due = [
        signal
        for decision in decisions
        for signal in decision.get("signals", [])
        if str(signal.get("signal_sha256")) not in resolved
        and now >= _utc(signal.get("maturity_at_utc"), name="maturity") + grace
    ]
    due.sort(key=lambda row: (str(row["maturity_at_utc"]), str(row["signal_sha256"])))
    due = due[: int(policy["resolver"]["max_due_signals_per_run"])]
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
                    chunk_candle_limit=int(policy["resolver"]["chunk_candle_limit"]),
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
                        candidate_id=str(signal.get("candidate_id") or ""),
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
            "terminal_candidate_ids": scorecard["terminal_candidate_ids"],
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
    side = str(signal.get("side") or "")
    direction = str(signal.get("direction") or "")
    if (side, direction) not in {("LONG", "UP"), ("SHORT", "DOWN")}:
        raise ValueError("family forward signal side is invalid")
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
        if activation <= item.timestamp_utc < fill_end
        and (item.ask.l <= limit if side == "LONG" else item.bid.h >= limit)
    ]
    common = {
        "contract": OUTCOME_CONTRACT,
        "schema_version": 2,
        "policy_sha256": policy_sha256,
        "family_id": signal["family_id"],
        "candidate_id": signal["candidate_id"],
        "decision_id": signal["decision_id"],
        "signal_sha256": signal["signal_sha256"],
        "pair": pair,
        "side": side,
        "decision_at_utc": signal["decision_at_utc"],
        "activation_at_utc": signal["activation_at_utc"],
        "maturity_at_utc": signal["maturity_at_utc"],
        "resolved_at_utc": _aware_utc(resolved_at_utc).isoformat(),
        "truth_contract": TRUTH_CONTRACT,
        "truth_chunk_sha256": list(truth_chunk_sha256),
        "truth_candle_count": len(ordered),
        "truth_candles_sha256": _stable_digest(
            [_candle_payload(item) for item in ordered]
        ),
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
    exit_price = exit_candle.bid.o if side == "LONG" else exit_candle.ask.o
    factor = instrument_pip_factor(pair)
    realized = (
        (exit_price - limit) * factor
        if side == "LONG"
        else (limit - exit_price) * factor
    )
    gap_through = fill.ask.o < limit if side == "LONG" else fill.bid.o > limit
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
                "gap_through_fill": bool(gap_through),
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
    outcome_by_signal = {str(row["signal_sha256"]): row for row in outcomes}
    source_sha = str(policy["selection_disclosure"]["source_artifact_sha256"])
    family_evaluation = _mapping(policy["family_evaluation"], "family evaluation")
    final_limit = int(family_evaluation["fixed_final_sample_prefix"])
    final_z = NormalDist().inv_cdf(
        1.0 - float(family_evaluation["per_candidate_final_alpha"])
    )
    candidate_scorecards: list[dict[str, Any]] = []
    for candidate in policy["candidates"]:
        candidate_id = str(candidate["candidate_id"])
        candidate_decisions = [
            row for row in decisions if row.get("candidate_id") == candidate_id
        ]
        signals = sorted(
            [signal for row in candidate_decisions for signal in row.get("signals", [])],
            key=lambda row: (str(row["activation_at_utc"]), str(row["signal_sha256"])),
        )
        resolved = [
            outcome_by_signal[str(signal["signal_sha256"])]
            for signal in signals
            if str(signal["signal_sha256"]) in outcome_by_signal
        ]
        filled = sorted(
            [row for row in resolved if row.get("filled") is True],
            key=lambda row: (str(row["fill_at_utc"]), str(row["signal_sha256"])),
        )
        fixed_filled = filled[:final_limit]
        metrics = _prospective_metrics(fixed_filled, confidence_z=final_z)
        selector = _mapping(candidate["selector"], "selector")
        evidence = build_profitability_evidence(
            lane_id=candidate_id,
            pair=str(selector["pair"]),
            side=str(selector["candidate_side"]),
            method=f"NORMALIZED_RETURN_{selector['orientation']}",
            order_type="LIMIT",
            metrics=metrics,
            source_artifact_sha256=source_sha,
            generated_at_utc=_aware_utc(as_of_utc),
            evidence_end_utc=_aware_utc(as_of_utc),
        )
        raw_gate = assess_profitability_evidence(
            evidence,
            thresholds=_mapping(policy["forward_evaluation"], "forward evaluation"),
        )
        final_passed = bool(
            len(fixed_filled) == final_limit
            and raw_gate.get("status") == "SHADOW_FORWARD_OBSERVATION_READY"
        )
        gate = _zero_authority(
            {
                "contract": "QR_FAST_BOT_NORMALIZED_PASSIVE_FAMILY_GATE_V2",
                "schema_version": 2,
                "source_gate_sha256": raw_gate.get("gate_sha256"),
                "source_shadow_status": raw_gate.get("status"),
                "blockers": list(raw_gate.get("blockers") or ()),
                "thresholds": dict(raw_gate.get("thresholds") or {}),
                "metrics": dict(raw_gate.get("metrics") or {}),
                "interpretation": "RESEARCH_REVIEW_ONLY_NOT_TRADING_ADMISSION",
            }
        )
        futility = _futility_assessment(filled, family_evaluation=family_evaluation)
        if len(fixed_filled) == final_limit:
            status = (
                "FINAL_EVIDENCE_PASSED_REVIEW_REQUIRED"
                if final_passed
                else "FINAL_EVIDENCE_REJECTED_COLLECTION_STOPPED"
            )
        elif futility["rejected"]:
            status = "FUTILITY_REJECTED_COLLECTION_STOPPED"
        elif signals:
            status = "COLLECTING_PROSPECTIVE_EVIDENCE"
        else:
            status = "COLLECTING_NO_PROSPECTIVE_SIGNALS"
        candidate_scorecards.append(
            _zero_authority(
                {
                    "candidate_id": candidate_id,
                    "candidate_role": candidate["candidate_role"],
                    "pair": selector["pair"],
                    "side": selector["candidate_side"],
                    "status": status,
                    "historical_holdout_inspected": candidate[
                        "historical_holdout_inspected"
                    ],
                    "strict_pre_holdout_qualified": candidate[
                        "strict_pre_holdout_qualified"
                    ],
                    "decision_count": len(candidate_decisions),
                    "emitted_signal_count": len(signals),
                    "resolved_signal_count": len(resolved),
                    "filled_signal_count_total": len(filled),
                    "evaluated_fixed_prefix_count": len(fixed_filled),
                    "unfilled_signal_count": sum(
                        row.get("filled") is False for row in resolved
                    ),
                    "metrics": metrics,
                    "profitability_evidence": evidence,
                    "profitability_gate": gate,
                    "interim_futility": futility,
                    "prospective_gate_passed": final_passed,
                    "collection_stopped": status in TERMINAL_CANDIDATE_STATUSES,
                    "order_intents": [],
                }
            )
        )
    terminal_ids = [
        row["candidate_id"]
        for row in candidate_scorecards
        if row["status"] in TERMINAL_CANDIDATE_STATUSES
    ]
    passed_ids = [
        row["candidate_id"]
        for row in candidate_scorecards
        if row["status"] == "FINAL_EVIDENCE_PASSED_REVIEW_REQUIRED"
    ]
    any_signal = any(row["emitted_signal_count"] for row in candidate_scorecards)
    if passed_ids:
        status = "FAMILY_EVIDENCE_PASSED_REVIEW_REQUIRED"
    elif len(terminal_ids) == len(candidate_scorecards):
        status = "FAMILY_EXHAUSTED_NO_CANDIDATE_PASSED"
    elif any_signal:
        status = "COLLECTING_FAMILY_PROSPECTIVE_EVIDENCE"
    else:
        status = "COLLECTING_NO_PROSPECTIVE_SIGNALS"
    body = _zero_authority(
        {
            "contract": SCORECARD_CONTRACT,
            "schema_version": 2,
            "policy_sha256": policy_sha256,
            "family_id": policy["family_id"],
            "generated_at_utc": _aware_utc(as_of_utc).isoformat(),
            "status": status,
            "candidate_count": len(candidate_scorecards),
            "candidate_scorecards": candidate_scorecards,
            "terminal_candidate_ids": terminal_ids,
            "passed_candidate_ids": passed_ids,
            "prospective_gate_passed": bool(passed_ids),
            "familywise_correction": {
                "method": family_evaluation["multiple_testing_correction"],
                "familywise_alpha": family_evaluation["familywise_alpha"],
                "per_candidate_final_alpha": family_evaluation[
                    "per_candidate_final_alpha"
                ],
                "final_confidence_z": round(final_z, 9),
                "interim_test_count": family_evaluation["interim_test_count"],
                "per_interim_test_alpha": family_evaluation[
                    "per_interim_test_alpha"
                ],
            },
            "historical_profitability_claim_allowed": False,
            "research_observation_continues": len(terminal_ids)
            < len(candidate_scorecards),
            "automatic_replacement_allowed": False,
            "replacement_policy": family_evaluation["replacement_policy"],
            "historical_promotion_blockers": list(
                policy.get("promotion_blockers") or ()
            ),
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
            if signal.get("candidate_id") != row.get("candidate_id"):
                raise ValueError("decision signal candidate mismatch")
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


def _futility_assessment(
    filled: Sequence[Mapping[str, Any]],
    *,
    family_evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    alpha = float(family_evaluation["per_interim_test_alpha"])
    z = NormalDist().inv_cdf(1.0 - alpha)
    looks: list[dict[str, Any]] = []
    rejected_at: int | None = None
    for raw_look in family_evaluation["interim_futility_looks"]:
        look = int(raw_look)
        if len(filled) < look:
            looks.append(
                {
                    "sample_prefix": look,
                    "status": "NOT_REACHED",
                    "optimistic_expectancy_pips": None,
                }
            )
            continue
        prefix = filled[:look]
        values = [float(row["realized_pips"]) for row in prefix]
        optimistic = _optimistic_expectancy(values, confidence_z=z)
        failed = optimistic <= 0.0
        looks.append(
            {
                "sample_prefix": look,
                "status": "FUTILITY_REJECT" if failed else "CONTINUE",
                "optimistic_expectancy_pips": round(optimistic, 6),
            }
        )
        if failed:
            rejected_at = look
            break
    return {
        "rule": family_evaluation["futility_rule"],
        "per_test_alpha": alpha,
        "confidence_z": round(z, 9),
        "looks": looks,
        "rejected": rejected_at is not None,
        "rejected_at_sample_prefix": rejected_at,
        "automatic_replacement_allowed": False,
    }


def _prospective_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    confidence_z: float,
) -> dict[str, Any]:
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
    pessimistic = _pessimistic_expectancy(values, confidence_z=confidence_z)
    sample_count = len(values)
    return {
        "sample_count": sample_count,
        "active_days": len(by_day_values),
        "wins": sum(value > 0.0 for value in values),
        "losses": sum(value < 0.0 for value in values),
        "net_pips": round(sum(values), 6),
        "expectancy_pips": round(statistics.mean(values), 6) if values else 0.0,
        "pessimistic_expectancy_pips": (
            round(pessimistic, 6) if pessimistic is not None else None
        ),
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
        "familywise_corrected_confidence_z": round(confidence_z, 9),
    }


def _pessimistic_expectancy(
    values: Sequence[float],
    *,
    confidence_z: float,
) -> float | None:
    if not values:
        return None
    wins = [value for value in values if value > 0.0]
    losses = [-value for value in values if value < 0.0]
    lower = _wilson_bound(
        success_count=len(wins),
        sample_count=len(values),
        confidence_z=confidence_z,
        upper=False,
    )
    average_win = statistics.mean(wins) if wins else 0.0
    average_loss = statistics.mean(losses) if losses else 0.0
    return lower * average_win - (1.0 - lower) * average_loss


def _optimistic_expectancy(
    values: Sequence[float],
    *,
    confidence_z: float,
) -> float:
    wins = [value for value in values if value > 0.0]
    losses = [-value for value in values if value < 0.0]
    upper = _wilson_bound(
        success_count=len(wins),
        sample_count=len(values),
        confidence_z=confidence_z,
        upper=True,
    )
    average_win = statistics.mean(wins) if wins else 0.0
    average_loss = statistics.mean(losses) if losses else 0.0
    return upper * average_win - (1.0 - upper) * average_loss


def _wilson_bound(
    *,
    success_count: int,
    sample_count: int,
    confidence_z: float,
    upper: bool,
) -> float:
    if sample_count <= 0:
        return 1.0 if upper else 0.0
    observed = success_count / sample_count
    z2 = confidence_z * confidence_z
    denominator = 1.0 + z2 / sample_count
    center = observed + z2 / (2.0 * sample_count)
    margin = confidence_z * math.sqrt(
        (observed * (1.0 - observed) + z2 / (4.0 * sample_count))
        / sample_count
    )
    numerator = center + margin if upper else center - margin
    return min(1.0, max(0.0, numerator / denominator))


def _active_reservation(
    decisions: Sequence[Mapping[str, Any]],
    *,
    candidate_id: str,
    activation_at_utc: datetime,
) -> Mapping[str, Any] | None:
    active = []
    for decision in decisions:
        if decision.get("candidate_id") != candidate_id:
            continue
        raw = decision.get("reservation_until_utc")
        if raw and activation_at_utc < _utc(raw, name="reservation"):
            active.append(decision)
    return max(active, key=lambda row: str(row["reservation_until_utc"])) if active else None


def _validate_exploratory_selection(
    disclosure: Mapping[str, Any],
    artifact: Mapping[str, Any],
    expected_candidate_id: str,
) -> None:
    rule = _mapping(
        disclosure.get("exploratory_candidate_selection"),
        "exploratory selection",
    )
    expected_rule = {
        "pair_must_differ_from_anchor": True,
        "minimum_train_samples": 40,
        "minimum_validation_samples": 25,
        "minimum_train_profit_factor": 1.0,
        "minimum_validation_profit_factor": 1.0,
        "minimum_train_positive_year_rate": 0.75,
        "minimum_validation_positive_year_rate": 1.0,
        "positive_train_and_validation_net_required": True,
        "ranking": [
            "MAXIMIZE_MIN_TRAIN_VALIDATION_PROFIT_FACTOR",
            "MAXIMIZE_VALIDATION_NET_PIPS",
            "MAXIMIZE_TRAIN_NET_PIPS",
            "LEXICOGRAPHIC_CANDIDATE_ID",
        ],
        "holdout_inspection_forbidden": True,
    }
    if dict(rule) != expected_rule:
        raise ValueError("exploratory candidate selection rule changed")
    eligible: list[Mapping[str, Any]] = []
    for row in artifact.get("pre_holdout_candidates", []):
        if not isinstance(row, Mapping) or row.get("pair") == "EUR_USD":
            continue
        train = _mapping(row.get("train"), "historical train")
        validation = _mapping(row.get("validation"), "historical validation")
        if (
            int(train.get("trades") or 0) >= 40
            and int(validation.get("trades") or 0) >= 25
            and _profit_factor_value(train.get("profit_factor")) > 1.0
            and _profit_factor_value(validation.get("profit_factor")) > 1.0
            and float(train.get("net_pips") or 0.0) > 0.0
            and float(validation.get("net_pips") or 0.0) > 0.0
            and float(train.get("positive_year_rate") or 0.0) >= 0.75
            and float(validation.get("positive_year_rate") or 0.0) == 1.0
        ):
            eligible.append(row)
    if not eligible:
        raise ValueError("exploratory selection has no eligible pre-holdout rows")
    chosen = max(
        eligible,
        key=lambda row: (
            min(
                _profit_factor_value(row["train"]["profit_factor"]),
                _profit_factor_value(row["validation"]["profit_factor"]),
            ),
            float(row["validation"]["net_pips"]),
            float(row["train"]["net_pips"]),
            str(row["candidate_id"]),
        ),
    )
    if chosen.get("candidate_id") != expected_candidate_id:
        raise ValueError("exploratory candidate no longer wins frozen pre-holdout rule")


def _expected_candidates() -> dict[str, dict[str, Any]]:
    return {
        ANCHOR_ID: {
            "candidate_role": "REJECTED_HISTORICAL_ANCHOR_PROSPECTIVE_MONITOR",
            "historical_holdout_inspected": True,
            "historical_status": "HOLDOUT_REJECT",
            "strict_pre_holdout_qualified": True,
            "selector": {
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
            },
            "vehicle": {
                "entry_vehicle": "PASSIVE_LIMIT_TIME_CLOSE",
                "entry_spread_fraction": 0.25,
                "entry_ttl_minutes": 5,
                "holding_minutes": 240,
                "price_tick": 0.00001,
                "gap_through_fill_policy": "CONSERVATIVE_LIMIT_PRICE_NO_IMPROVEMENT",
                "take_profit_pips": None,
                "stop_loss_pips": None,
            },
        },
        EXPLORATORY_ID: {
            "candidate_role": (
                "DISTINCT_PAIR_EXPLORATORY_BELOW_STRICT_SAMPLE_FLOOR"
            ),
            "historical_holdout_inspected": False,
            "historical_status": "PRE_HOLDOUT_EXPLORATORY_ONLY",
            "strict_pre_holdout_qualified": False,
            "selector": {
                "pair": "USD_JPY",
                "candidate_side": "LONG",
                "orientation": "MOMENTUM",
                "decision_interval_minutes": 60,
                "decision_candle_minute_utc": 0,
                "lookback_minutes": 60,
                "confirmation_lookback_minutes": 240,
                "normalized_threshold": 2.0,
                "confirmation_policy": "SAME_SIGN_COMPLETED_M1_RETURN",
                "maximum_decision_spread_pips": 1.5,
                "collection_lateness_max_seconds_after_activation": 90,
            },
            "vehicle": {
                "entry_vehicle": "PASSIVE_LIMIT_TIME_CLOSE",
                "entry_spread_fraction": 0.5,
                "entry_ttl_minutes": 5,
                "holding_minutes": 240,
                "price_tick": 0.001,
                "gap_through_fill_policy": "CONSERVATIVE_LIMIT_PRICE_NO_IMPROVEMENT",
                "take_profit_pips": None,
                "stop_loss_pips": None,
            },
        },
    }


def _match_metric_subset(actual: Any, source: Any, name: str) -> None:
    actual_map = _mapping(actual, f"policy {name}")
    source_map = _mapping(source, f"historical {name}")
    for key, value in actual_map.items():
        if source_map.get(key) != value:
            raise ValueError(f"historical {name} metric changed: {key}")


def _bounded_candidate_decision(value: Mapping[str, Any]) -> dict[str, Any]:
    keep = (
        "candidate_id",
        "decision_id",
        "decision_at_utc",
        "activation_at_utc",
        "status",
        "qualifying_return",
        "source_direction",
        "selected_side",
        "normalized_return",
        "decision_spread_pips",
    )
    return {key: value.get(key) for key in keep}


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
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
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
    return _aware_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone aware")
    return value.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _profit_factor_value(value: Any) -> float:
    return math.inf if value == "INF" else float(value)


def _error(scope: str, exc: Exception, **extra: str) -> dict[str, Any]:
    return {
        "scope": scope,
        "error_type": type(exc).__name__,
        "message": str(exc)[:500],
        **extra,
    }
