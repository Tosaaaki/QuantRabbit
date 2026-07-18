from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quant_rabbit.dojo_ai_forward import (
    DojoAIForwardError,
    build_cell_response_failure,
    build_cell_response_seal,
    build_day_requests,
    build_day_requests_from_capture,
    build_missing_day_seal,
    build_phase_index,
    build_precommit,
    build_start_receipt,
    build_source_capture,
    build_source_capture_from_request,
    prepare_day_request,
    validate_cell_terminal,
    validate_day_seal,
    validate_phase_index,
    validate_precommit,
    validate_start_receipt,
    validate_source_capture,
    validate_source_request,
)
from quant_rabbit.dojo_market_calendar import expected_oanda_fx_slots
from quant_rabbit.dojo_ai_truth import (
    DojoAITruthError,
    build_phase_score as build_truth_phase_score,
    build_day_score as build_truth_day_score,
    build_truth_bundle,
    build_truth_capture,
    build_truth_request,
    validate_day_score as validate_truth_day_score,
    validate_phase_score as validate_truth_phase_score,
    validate_truth_bundle_with_capture,
    validate_truth_capture,
    validate_truth_request,
)
from quant_rabbit.dojo_prompt_phase import LOCKED_VARIANT_PROMPT_SHA256


REPO = Path(__file__).resolve().parents[1]
REGISTRY = json.loads(
    (REPO / "research/registries/dojo_prompt_experiment_v1.json").read_text()
)


def utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def prompt_texts() -> dict[str, str]:
    return {
        row["variant_id"]: (REPO / row["prompt_path"]).read_text()
        for row in REGISTRY["variants"]
    }


def spec() -> dict:
    return {
        "first_cutoff_utc": "2026-07-22T15:00:00Z",
        "allocation_nonce": digest("fixed-prospective-allocation"),
        "model_policy": {
            "model_name": "gpt-5.5",
            "model_version": "gpt-5.5",
            "model_lineage": "openai-gpt-5.5",
            "reasoning_effort": "high",
        },
        "source_bindings": {
            "git_commit": "a" * 40,
            "files": {
                "src/quant_rabbit/dojo_ai_forward.py": digest("forward-module"),
                "src/quant_rabbit/dojo_ai_discretion.py": digest("packet-module"),
            },
        },
    }


def lifecycle() -> tuple[dict, dict]:
    precommit = build_precommit(
        REGISTRY,
        prompt_texts(),
        spec(),
        now_utc=utc("2026-07-19T00:00:00Z"),
    )
    start = build_start_receipt(
        precommit,
        now_utc=utc("2026-07-19T00:01:00Z"),
    )
    return precommit, start


def oanda_response(precommit: dict, ordinal: int = 1) -> dict:
    schedule = precommit["schedule"][ordinal - 1]
    window_start = utc(schedule["source_window_start_utc"])
    cutoff = utc(schedule["decision_cutoff_utc"])
    candles = []
    for cursor in expected_oanda_fx_slots(
        window_start, cutoff, step=timedelta(minutes=5)
    ):
        candles.append(
            {
                "complete": True,
                "volume": 12,
                "time": cursor.isoformat().replace("+00:00", "Z"),
                "bid": {
                    "o": "150.000",
                    "h": "150.010",
                    "l": "149.990",
                    "c": "150.005",
                },
                "ask": {
                    "o": "150.002",
                    "h": "150.012",
                    "l": "149.992",
                    "c": "150.007",
                },
            }
        )
    return {
        "instrument": "USD_JPY",
        "granularity": "M5",
        "candles": candles,
    }


def decision(day: dict, cell_index: int = 0) -> dict:
    packet = day["cells"][cell_index]["packet"]
    return {
        "trial_id": packet["trial_id"],
        "action": "FLAT",
        "pair": "USD_JPY",
        "size": "NONE",
        "confidence": 0.55,
        "evidence_refs": [packet["observations"][0]["id"]],
        "target_pips": None,
        "invalidation_pips": None,
        "strongest_counterargument": "A directional move may begin after cutoff.",
        "abstain_reason": "The sealed evidence is not decisive.",
    }


def truth_response(precommit: dict, ordinal: int = 1) -> dict:
    schedule = precommit["schedule"][ordinal - 1]
    start = utc(schedule["decision_cutoff_utc"])
    end = utc(schedule["truth_not_before_utc"]) - timedelta(minutes=2)
    candles = []
    for cursor in expected_oanda_fx_slots(start, end, step=timedelta(minutes=5)):
        candles.append(
            {
                "complete": True,
                "volume": 8,
                "time": cursor.isoformat().replace("+00:00", "Z"),
                "bid": {
                    "o": "150.000",
                    "h": "150.110",
                    "l": "149.990",
                    "c": "150.100",
                },
                "ask": {
                    "o": "150.002",
                    "h": "150.112",
                    "l": "149.992",
                    "c": "150.102",
                },
            }
        )
    return {"instrument": "USD_JPY", "granularity": "M5", "candles": candles}


def test_precommit_freezes_exact_30_day_90_cell_schedule() -> None:
    precommit, start = lifecycle()
    assert validate_precommit(precommit) == precommit
    assert validate_start_receipt(start, precommit) == start
    schedule = precommit["schedule"]
    assert len(schedule) == 30
    assert sum(len(row["cells"]) for row in schedule) == 90
    assert [cell["variant_id"] for cell in schedule[0]["cells"]] == list(
        LOCKED_VARIANT_PROMPT_SHA256
    )
    cutoffs = [utc(row["decision_cutoff_utc"]) for row in schedule]
    assert all(cutoff.weekday() in {0, 1, 2, 3} for cutoff in cutoffs)
    assert all(cutoff.hour == 15 and cutoff.minute == 0 for cutoff in cutoffs)
    assert len({row["blind_day_id"] for row in schedule}) == 30
    cells = [cell for row in schedule for cell in row["cells"]]
    assert len({cell["cell_id"] for cell in cells}) == 90
    assert len({cell["context_id"] for cell in cells}) == 90
    assert precommit["authority"]["ai_order_authority"] == "NONE"
    assert precommit["authority"]["live_permission"] is False


def test_precommit_after_first_cutoff_and_bad_schedule_anchor_are_rejected() -> None:
    with pytest.raises(DojoAIForwardError, match="precede first cutoff"):
        build_precommit(
            REGISTRY,
            prompt_texts(),
            spec(),
            now_utc=utc("2026-07-22T15:00:00Z"),
        )
    bad = spec()
    bad["first_cutoff_utc"] = "2026-07-24T15:00:00Z"  # Friday.
    with pytest.raises(DojoAIForwardError, match="weekday/time"):
        build_precommit(
            REGISTRY,
            prompt_texts(),
            bad,
            now_utc=utc("2026-07-19T00:00:00Z"),
        )


def test_day_seals_source_and_all_three_precommitted_requests_atomically() -> None:
    precommit, start = lifecycle()
    schedule = precommit["schedule"][0]
    sealed = build_day_requests(
        REGISTRY,
        precommit,
        start,
        None,
        oanda_response(precommit),
        ordinal=1,
        now_utc=utc(schedule["source_not_before_utc"]),
    )
    assert (
        validate_day_seal(
            sealed,
            REGISTRY,
            precommit,
            start,
            None,
            expected_ordinal=1,
        )
        == sealed
    )
    assert sealed["state"] == "REQUESTS_SEALED"
    assert len(sealed["cells"]) == 3
    assert {
        row["source_sha256"] for row in [cell["assignment"] for cell in sealed["cells"]]
    } == {sealed["source_sha256"]}
    assert [cell["assignment"]["cell_id"] for cell in sealed["cells"]] == [
        cell["cell_id"] for cell in schedule["cells"]
    ]
    assert all(
        cell["request_receipt"]["answer_key_present"] is False
        and cell["request_receipt"]["model_api_invoked"] is False
        and cell["request_receipt"]["authority"]["live_permission"] is False
        for cell in sealed["cells"]
    )


def test_first_source_capture_is_immutable_parent_of_day_requests() -> None:
    precommit, start = lifecycle()
    schedule = precommit["schedule"][0]
    capture = build_source_capture(
        precommit,
        start,
        None,
        oanda_response(precommit),
        ordinal=1,
        now_utc=utc(schedule["source_not_before_utc"]),
    )
    assert validate_source_capture(capture, precommit, start, None) == capture
    day = build_day_requests_from_capture(
        REGISTRY, precommit, start, None, capture
    )
    assert day["source_capture_sha256"] == capture["source_capture_sha256"]
    assert (
        day["source_provenance"]["source_capture_sha256"]
        == capture["source_capture_sha256"]
    )
    tampered = copy.deepcopy(capture)
    tampered["response"]["candles"][0]["volume"] += 1
    with pytest.raises(DojoAIForwardError, match="digest mismatch"):
        validate_source_capture(tampered, precommit, start, None)


def test_source_request_precedes_capture_and_invalid_first_response_is_retained() -> None:
    precommit, start = lifecycle()
    schedule = precommit["schedule"][0]
    requested = utc(schedule["source_not_before_utc"])
    request = prepare_day_request(
        precommit,
        start,
        None,
        ordinal=1,
        now_utc=requested,
    )
    assert validate_source_request(request, precommit, start, None) == request
    capture = build_source_capture_from_request(
        precommit,
        start,
        None,
        request,
        {"malformed": "first-response"},
        acquired_at_utc=requested + timedelta(seconds=1),
    )
    assert capture["requested_at_utc"] < capture["acquired_at_utc"]
    assert validate_source_capture(capture, precommit, start, None) == capture
    with pytest.raises(DojoAIForwardError, match="keys are not exact"):
        build_day_requests_from_capture(REGISTRY, precommit, start, None, capture)
    valid_capture = build_source_capture_from_request(
        precommit,
        start,
        None,
        request,
        oanda_response(precommit),
        acquired_at_utc=requested + timedelta(seconds=1),
    )
    day = build_day_requests_from_capture(
        REGISTRY, precommit, start, None, valid_capture
    )
    assert (
        validate_day_seal(
            day,
            REGISTRY,
            precommit,
            start,
            None,
            expected_ordinal=1,
        )
        == day
    )
    with pytest.raises(DojoAIForwardError, match="ordering"):
        build_source_capture_from_request(
            precommit,
            start,
            None,
            request,
            oanda_response(precommit),
            acquired_at_utc=requested - timedelta(seconds=1),
        )


@pytest.mark.parametrize(
    "when",
    ["2026-07-22T15:01:59Z", "2026-07-22T15:30:01Z"],
)
def test_day_source_cannot_be_backdated_or_late(when: str) -> None:
    precommit, start = lifecycle()
    with pytest.raises(DojoAIForwardError, match="causal seal window"):
        build_day_requests(
            REGISTRY,
            precommit,
            start,
            None,
            oanda_response(precommit),
            ordinal=1,
            now_utc=utc(when),
        )


def test_day_source_identity_and_future_data_cannot_be_caller_selected() -> None:
    precommit, start = lifecycle()
    schedule = precommit["schedule"][0]
    source = oanda_response(precommit)
    source["instrument"] = "EUR_USD"
    with pytest.raises(DojoAIForwardError, match="identity drifted"):
        build_day_requests(
            REGISTRY,
            precommit,
            start,
            None,
            source,
            ordinal=1,
            now_utc=utc(schedule["source_not_before_utc"]),
        )
    source = oanda_response(precommit)
    source["candles"][0]["time"] = "2026-07-22T15:00:00Z"
    with pytest.raises(DojoAIForwardError, match="causal source window"):
        build_day_requests(
            REGISTRY,
            precommit,
            start,
            None,
            source,
            ordinal=1,
            now_utc=utc(schedule["source_not_before_utc"]),
        )
    source = oanda_response(precommit)
    source["candles"][0]["future_return"] = 0.1
    with pytest.raises(DojoAIForwardError, match="keys are not exact"):
        build_day_requests(
            REGISTRY,
            precommit,
            start,
            None,
            source,
            ordinal=1,
            now_utc=utc(schedule["source_not_before_utc"]),
        )


def test_oanda_zero_nanoseconds_are_normalized_but_fractional_opens_rejected() -> None:
    precommit, start = lifecycle()
    schedule = precommit["schedule"][0]
    source = oanda_response(precommit)
    for row in source["candles"]:
        row["time"] = row["time"].replace("Z", ".000000000Z")
    sealed = build_day_requests(
        REGISTRY,
        precommit,
        start,
        None,
        source,
        ordinal=1,
        now_utc=utc(schedule["source_not_before_utc"]),
    )
    assert sealed["source"]["observations"][0]["observed_at_utc"].endswith("Z")
    source = oanda_response(precommit)
    source["candles"][0]["time"] = source["candles"][0]["time"].replace(
        "Z", ".000000001Z"
    )
    with pytest.raises(DojoAIForwardError, match="aligned OANDA UTC timestamp"):
        build_day_requests(
            REGISTRY,
            precommit,
            start,
            None,
            source,
            ordinal=1,
            now_utc=utc(schedule["source_not_before_utc"]),
        )


def test_duplicate_or_unsorted_source_rows_are_rejected() -> None:
    precommit, start = lifecycle()
    schedule = precommit["schedule"][0]
    source = oanda_response(precommit)
    source["candles"].append(copy.deepcopy(source["candles"][0]))
    with pytest.raises(DojoAIForwardError, match="duplicated or unsorted"):
        build_day_requests(
            REGISTRY,
            precommit,
            start,
            None,
            source,
            ordinal=1,
            now_utc=utc(schedule["source_not_before_utc"]),
        )


def test_market_source_subset_cannot_be_selected() -> None:
    precommit, start = lifecycle()
    schedule = precommit["schedule"][0]
    source = oanda_response(precommit)
    source["candles"] = source["candles"][-1:]
    with pytest.raises(DojoAIForwardError, match="fixed coverage floor"):
        build_day_requests(
            REGISTRY,
            precommit,
            start,
            None,
            source,
            ordinal=1,
            now_utc=utc(schedule["source_not_before_utc"]),
        )


def test_missed_day_is_permanent_three_cell_failure_and_chain_continues() -> None:
    precommit, start = lifecycle()
    day1_schedule = precommit["schedule"][0]
    with pytest.raises(DojoAIForwardError, match="before source deadline"):
        build_missing_day_seal(
            precommit,
            start,
            None,
            ordinal=1,
            now_utc=utc(day1_schedule["source_seal_deadline_utc"]),
        )
    missing = build_missing_day_seal(
        precommit,
        start,
        None,
        ordinal=1,
        now_utc=utc(day1_schedule["source_seal_deadline_utc"]) + timedelta(seconds=1),
    )
    assert missing["state"] == "MISSING_SOURCE_DEADLINE"
    assert len(missing["terminal_failures"]) == 3
    assert all(
        row["economic_fallback"] == "SYNTHETIC_FLAT_ZERO_RETURN"
        for row in missing["terminal_failures"]
    )
    day2_schedule = precommit["schedule"][1]
    day2 = build_day_requests(
        REGISTRY,
        precommit,
        start,
        missing,
        oanda_response(precommit, ordinal=2),
        ordinal=2,
        now_utc=utc(day2_schedule["source_not_before_utc"]),
    )
    assert day2["previous_receipt_sha256"] == missing["day_seal_sha256"]


def test_strict_ordinal_prevents_day_reselection() -> None:
    precommit, start = lifecycle()
    with pytest.raises(DojoAIForwardError, match="chain has a gap"):
        build_day_requests(
            REGISTRY,
            precommit,
            start,
            None,
            oanda_response(precommit, ordinal=2),
            ordinal=2,
            now_utc=utc(precommit["schedule"][1]["source_not_before_utc"]),
        )


def test_day_chain_time_cannot_move_backwards() -> None:
    precommit, start = lifecycle()
    late_day1 = build_missing_day_seal(
        precommit,
        start,
        None,
        ordinal=1,
        now_utc=utc("2026-07-23T16:02:00Z"),
    )
    assert utc(late_day1["sealed_at_utc"]) > utc(
        precommit["schedule"][1]["source_not_before_utc"]
    )
    with pytest.raises(DojoAIForwardError, match="time moved backwards"):
        build_day_requests(
            REGISTRY,
            precommit,
            start,
            late_day1,
            oanda_response(precommit, ordinal=2),
            ordinal=2,
            now_utc=utc(precommit["schedule"][1]["source_not_before_utc"]),
        )


def test_response_is_sealed_before_deadline_without_answer_key_or_authority() -> None:
    precommit, start = lifecycle()
    schedule = precommit["schedule"][0]
    day = build_day_requests(
        REGISTRY,
        precommit,
        start,
        None,
        oanda_response(precommit),
        ordinal=1,
        now_utc=utc(schedule["source_not_before_utc"]),
    )
    cell_id = schedule["cells"][0]["cell_id"]
    terminal = build_cell_response_seal(
        REGISTRY,
        precommit,
        start,
        None,
        day,
        decision(day),
        cell_id=cell_id,
        now_utc=utc(day["sealed_at_utc"]) + timedelta(minutes=1),
    )
    assert (
        validate_cell_terminal(
            terminal,
            REGISTRY,
            precommit,
            start,
            None,
            day,
        )
        == terminal
    )
    assert terminal["answer_key_opened"] is False
    assert terminal["response_selection_allowed"] is False
    assert terminal["authority"]["ai_order_authority"] == "NONE"
    assert terminal["authority"]["live_permission"] is False


def test_response_cell_and_deadline_cannot_be_selected_after_outcome() -> None:
    precommit, start = lifecycle()
    schedule = precommit["schedule"][0]
    day = build_day_requests(
        REGISTRY,
        precommit,
        start,
        None,
        oanda_response(precommit),
        ordinal=1,
        now_utc=utc(schedule["source_not_before_utc"]),
    )
    with pytest.raises(DojoAIForwardError, match="uniquely assigned"):
        build_cell_response_seal(
            REGISTRY,
            precommit,
            start,
            None,
            day,
            decision(day),
            cell_id="cell-not-scheduled",
            now_utc=utc(day["sealed_at_utc"]) + timedelta(minutes=1),
        )
    with pytest.raises(DojoAIForwardError, match="deadline"):
        build_cell_response_seal(
            REGISTRY,
            precommit,
            start,
            None,
            day,
            decision(day),
            cell_id=schedule["cells"][0]["cell_id"],
            now_utc=utc(schedule["response_deadline_utc"]) + timedelta(seconds=1),
        )


def test_missing_response_is_permanent_only_after_deadline() -> None:
    precommit, start = lifecycle()
    schedule = precommit["schedule"][0]
    day = build_day_requests(
        REGISTRY,
        precommit,
        start,
        None,
        oanda_response(precommit),
        ordinal=1,
        now_utc=utc(schedule["source_not_before_utc"]),
    )
    cell_id = schedule["cells"][0]["cell_id"]
    with pytest.raises(DojoAIForwardError, match="before deadline"):
        build_cell_response_failure(
            precommit,
            day,
            cell_id=cell_id,
            now_utc=utc(schedule["response_deadline_utc"]),
        )
    terminal = build_cell_response_failure(
        precommit,
        day,
        cell_id=cell_id,
        now_utc=utc(schedule["response_deadline_utc"]) + timedelta(seconds=1),
    )
    assert terminal["state"] == "MISSING_RESPONSE_DEADLINE"
    assert terminal["economic_fallback"] == "SYNTHETIC_FLAT_ZERO_RETURN"
    assert terminal["late_response_backfill_allowed"] is False
    assert (
        validate_cell_terminal(
            terminal,
            REGISTRY,
            precommit,
            start,
            None,
            day,
        )
        == terminal
    )


def test_phase_index_derives_exact_90_cell_denominator_after_truth_maturity() -> None:
    precommit, start = lifecycle()
    day1_schedule = precommit["schedule"][0]
    day1 = build_day_requests(
        REGISTRY,
        precommit,
        start,
        None,
        oanda_response(precommit),
        ordinal=1,
        now_utc=utc(day1_schedule["source_not_before_utc"]),
    )
    terminals = [
        build_cell_response_failure(
            precommit,
            day1,
            cell_id=cell["cell_id"],
            now_utc=utc(day1_schedule["response_deadline_utc"])
            + timedelta(seconds=1),
        )
        for cell in day1_schedule["cells"]
    ]
    days = [day1]
    previous = day1
    for ordinal in range(2, 31):
        schedule = precommit["schedule"][ordinal - 1]
        missing = build_missing_day_seal(
            precommit,
            start,
            previous,
            ordinal=ordinal,
            now_utc=utc(schedule["source_seal_deadline_utc"])
            + timedelta(seconds=1),
        )
        days.append(missing)
        previous = missing
    last_truth = utc(precommit["schedule"][-1]["truth_not_before_utc"])
    with pytest.raises(DojoAIForwardError, match="truth horizons"):
        build_phase_index(
            REGISTRY,
            precommit,
            start,
            days,
            terminals,
            now_utc=last_truth - timedelta(seconds=1),
        )
    future_last = build_missing_day_seal(
        precommit,
        start,
        days[-2],
        ordinal=30,
        now_utc=utc("2030-01-01T00:00:00Z"),
    )
    with pytest.raises(DojoAIForwardError, match="predates a day seal"):
        build_phase_index(
            REGISTRY,
            precommit,
            start,
            [*days[:-1], future_last],
            terminals,
            now_utc=last_truth,
        )
    index = build_phase_index(
        REGISTRY,
        precommit,
        start,
        days,
        terminals,
        now_utc=last_truth,
    )
    assert validate_phase_index(index, REGISTRY, precommit, start, days, terminals) == index
    assert index["allocated_cell_count"] == 90
    assert len(index["cell_index"]) == 90
    assert index["response_sealed_count"] == 0
    assert index["missing_response_cell_count"] == 3
    assert index["missing_source_cell_count"] == 87
    assert index["answer_keys_opened"] is False
    assert index["truth_scoring_present"] is False
    assert index["promotion_eligible"] is False


def test_market_truth_opens_only_after_all_responses_and_scores_from_ba() -> None:
    precommit, start = lifecycle()
    schedule = precommit["schedule"][0]
    day = build_day_requests(
        REGISTRY,
        precommit,
        start,
        None,
        oanda_response(precommit),
        ordinal=1,
        now_utc=utc(schedule["source_not_before_utc"]),
    )
    terminals = [
        build_cell_response_seal(
            REGISTRY,
            precommit,
            start,
            None,
            day,
            decision(day, index),
            cell_id=cell["assignment"]["cell_id"],
            now_utc=utc(schedule["source_not_before_utc"]),
        )
        for index, cell in enumerate(day["cells"])
    ]
    truth_time = utc(schedule["truth_not_before_utc"])
    with pytest.raises(DojoAITruthError, match="all three"):
        build_truth_request(precommit, day, terminals[:2], now_utc=truth_time)
    request = build_truth_request(precommit, day, terminals, now_utc=truth_time)
    assert validate_truth_request(request, precommit, day, terminals) == request
    capture = build_truth_capture(
        request, truth_response(precommit), acquired_at_utc=truth_time
    )
    assert validate_truth_capture(capture, request) == capture
    bundle = build_truth_bundle(
        precommit, day, terminals, capture, sealed_at_utc=truth_time
    )
    assert (
        validate_truth_bundle_with_capture(
            bundle, precommit, day, terminals, capture
        )
        == bundle
    )
    assert bundle["truth_semantics"] == "FIXED_24H_DIRECTION_AND_SIZE_ONLY"
    assert bundle["coverage"]["exact_entry_boundary_present"] is True
    assert bundle["coverage"]["exact_exit_boundary_present"] is True
    assert bundle["returns"]["FLAT"] == 0.0
    assert bundle["returns"]["LONG_FULL"] > 0.0
    assert bundle["returns"]["SHORT_FULL"] < 0.0
    assert len(bundle["answer_keys"]) == 3
    day_score = build_truth_day_score(
        precommit,
        day,
        terminals,
        capture,
        bundle,
        scored_at_utc=truth_time,
    )
    assert (
        validate_truth_day_score(
            day_score, precommit, day, terminals, capture, bundle
        )
        == day_score
    )
    assert {row["return_key"] for row in day_score["cell_results"]} == {"FLAT"}
    assert all(row["net_return"] == 0.0 for row in day_score["cell_results"])


def test_truth_first_capture_is_terminal_and_exact_boundaries_fail_closed() -> None:
    precommit, start = lifecycle()
    schedule = precommit["schedule"][0]
    day = build_day_requests(
        REGISTRY,
        precommit,
        start,
        None,
        oanda_response(precommit),
        ordinal=1,
        now_utc=utc(schedule["source_not_before_utc"]),
    )
    terminals = [
        build_cell_response_seal(
            REGISTRY,
            precommit,
            start,
            None,
            day,
            decision(day, index),
            cell_id=cell["assignment"]["cell_id"],
            now_utc=utc(schedule["source_not_before_utc"]),
        )
        for index, cell in enumerate(day["cells"])
    ]
    truth_time = utc(schedule["truth_not_before_utc"])
    request = build_truth_request(precommit, day, terminals, now_utc=truth_time)
    sparse = truth_response(precommit)
    sparse["candles"] = sparse["candles"][1:]
    capture = build_truth_capture(request, sparse, acquired_at_utc=truth_time)
    assert capture["response"] == sparse
    with pytest.raises(DojoAITruthError, match="exact entry or exit boundary"):
        build_truth_bundle(
            precommit, day, terminals, capture, sealed_at_utc=truth_time
        )


def test_truth_phase_score_keeps_exact_90_cell_denominator() -> None:
    precommit, start = lifecycle()
    first_schedule = precommit["schedule"][0]
    first_day = build_day_requests(
        REGISTRY,
        precommit,
        start,
        None,
        oanda_response(precommit),
        ordinal=1,
        now_utc=utc(first_schedule["source_not_before_utc"]),
    )
    terminals = [
        build_cell_response_seal(
            REGISTRY,
            precommit,
            start,
            None,
            first_day,
            decision(first_day, index),
            cell_id=cell["assignment"]["cell_id"],
            now_utc=utc(first_schedule["source_not_before_utc"]),
        )
        for index, cell in enumerate(first_day["cells"])
    ]
    truth_time = utc(first_schedule["truth_not_before_utc"])
    request = build_truth_request(precommit, first_day, terminals, now_utc=truth_time)
    capture = build_truth_capture(
        request, truth_response(precommit), acquired_at_utc=truth_time
    )
    bundle = build_truth_bundle(
        precommit, first_day, terminals, capture, sealed_at_utc=truth_time
    )
    day_score = build_truth_day_score(
        precommit,
        first_day,
        terminals,
        capture,
        bundle,
        scored_at_utc=truth_time,
    )
    days = [first_day]
    previous = first_day
    for ordinal in range(2, 31):
        schedule = precommit["schedule"][ordinal - 1]
        previous = build_missing_day_seal(
            precommit,
            start,
            previous,
            ordinal=ordinal,
            now_utc=utc(schedule["source_seal_deadline_utc"])
            + timedelta(seconds=1),
        )
        days.append(previous)
    phase_time = utc(precommit["schedule"][-1]["truth_not_before_utc"])
    index = build_phase_index(
        REGISTRY,
        precommit,
        start,
        days,
        terminals,
        now_utc=phase_time,
    )
    phase = build_truth_phase_score(
        precommit, index, days, [day_score], sealed_at_utc=phase_time
    )
    assert phase["allocated_cell_count"] == 90
    assert phase["valid_response_cell_count"] == 3
    assert phase["response_failure_cell_count"] == 87
    assert len(phase["cell_results"]) == 90
    assert phase["goal_status"] == "3X_NOT_REACHABLE"
    assert phase["prompt_selection_allowed"] is False
    assert phase["effective_independent_n"] == 0
    assert (
        validate_truth_phase_score(phase, precommit, index, days, [day_score])
        == phase
    )
    forged = copy.deepcopy(phase)
    forged["best_calendar_30d_multiple"] = 3.0
    with pytest.raises(DojoAITruthError, match="digest"):
        validate_truth_phase_score(forged, precommit, index, days, [day_score])
