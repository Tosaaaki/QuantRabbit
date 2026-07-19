from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest

from quant_rabbit.dojo_portfolio_allocator import (
    DojoPortfolioAllocatorError,
    build_portfolio_allocation,
    score_allocation_opportunity_cost,
)


EPOCH = 1_700_000_000


def _market_conversion(pair: str) -> tuple[float, float]:
    if pair.endswith("JPY"):
        return 150.0, 1.0
    return 1.1, 150.0


def _candidate(
    intent_id: str,
    *,
    owner_id: str | None = None,
    pair: str = "USD_JPY",
    side: str = "LONG",
    notional_jpy: float = 100_000.0,
    expected_net_edge_jpy: float = 100.0,
    holding_seconds: int = 3_600,
    sl_bound: bool = True,
    stop_distance_pips: float = 10.0,
    stress_cost_pips: float = 0.3,
) -> dict[str, object]:
    entry, jpy_per_quote_unit = _market_conversion(pair)
    pip = 0.01 if pair.endswith("JPY") else 0.0001
    units = notional_jpy / (entry * jpy_per_quote_unit)
    return {
        "intent_id": intent_id,
        "observed_epoch": EPOCH,
        "owner_id": owner_id or f"owner:{intent_id}",
        "strategy_family": "test_family",
        "pair": pair,
        "side": side,
        "order_kind": "MARKET",
        "units": units,
        "entry_price": entry,
        "jpy_per_quote_unit": jpy_per_quote_unit,
        "conversion_snapshot_id": None,
        "conversion_snapshot_sha256": None,
        "tp_price": entry + 10 * pip if side == "LONG" else entry - 10 * pip,
        "sl_price": (
            entry - stop_distance_pips * pip
            if side == "LONG" and sl_bound
            else entry + stop_distance_pips * pip
            if sl_bound
            else None
        ),
        "stress_cost_pips": stress_cost_pips,
        "expected_net_edge_jpy": expected_net_edge_jpy,
        "expected_holding_seconds": holding_seconds,
        "valid_until_epoch": EPOCH + 60,
    }


def _position(
    position_id: str,
    *,
    owner_id: str | None = None,
    pair: str = "EUR_JPY",
    side: str = "LONG",
    notional_jpy: float = 100_000.0,
    continuation_edge_jpy: float = 50.0,
    max_reduction_fraction: float = 1.0,
) -> dict[str, object]:
    mark_price, jpy_per_quote_unit = _market_conversion(pair)
    return {
        "position_id": position_id,
        "owner_id": owner_id or f"owner:{position_id}",
        "pair": pair,
        "side": side,
        "units": notional_jpy / (mark_price * jpy_per_quote_unit),
        "mark_price": mark_price,
        "jpy_per_quote_unit": jpy_per_quote_unit,
        "conversion_snapshot_id": None,
        "conversion_snapshot_sha256": None,
        "continuation_edge_jpy": continuation_edge_jpy,
        "max_reduction_fraction": max_reduction_fraction,
    }


def _pending(
    order_id: str,
    *,
    pair: str = "GBP_JPY",
    side: str = "LONG",
    notional_jpy: float = 100_000.0,
) -> dict[str, object]:
    trigger_price, jpy_per_quote_unit = _market_conversion(pair)
    return {
        "order_id": order_id,
        "owner_id": f"owner:{order_id}",
        "pair": pair,
        "side": side,
        "units": notional_jpy / (trigger_price * jpy_per_quote_unit),
        "trigger_price": trigger_price,
        "jpy_per_quote_unit": jpy_per_quote_unit,
        "conversion_snapshot_id": None,
        "conversion_snapshot_sha256": None,
    }


def _allocate(
    *,
    positions=(),
    pending=(),
    candidates=(),
    margin_cap: float = 0.5,
    currency_cap: float = 100.0,
    candidate_loss_cap: float = 0.05,
    switching_cost_jpy: float = 0.0,
    owner_caps: dict[str, tuple[int, int]] | None = None,
):
    position_rows = list(positions)
    pending_rows = list(pending)
    candidate_rows = list(candidates)
    effective_caps = owner_caps or {
        str(candidate["owner_id"]): (1, 4) for candidate in candidate_rows
    }
    return build_portfolio_allocation(
        decision_epoch=EPOCH,
        equity_jpy=100_000.0,
        leverage=10.0,
        global_margin_cap_fraction=margin_cap,
        currency_cap_fraction=currency_cap,
        max_candidate_loss_fraction=candidate_loss_cap,
        owner_concurrency_caps=[
            {
                "owner_id": owner_id,
                "max_concurrent_per_pair": caps[0],
                "global_max_concurrent": caps[1],
            }
            for owner_id, caps in effective_caps.items()
        ],
        open_positions=position_rows,
        pending_orders=pending_rows,
        candidate_intents=candidate_rows,
        switching_cost_jpy=switching_cost_jpy,
    )


def _reseal(allocation: dict[str, object]) -> dict[str, object]:
    body = {
        key: value for key, value in allocation.items() if key != "allocation_sha256"
    }
    payload = json.dumps(
        body,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    allocation["allocation_sha256"] = hashlib.sha256(payload).hexdigest()
    return allocation


def test_same_epoch_normalization_is_order_independent_and_records_all_intents():
    positions = [_position("P2"), _position("P1", side="SHORT")]
    pending = [_pending("O2"), _pending("O1", side="SHORT")]
    candidates = [
        _candidate("C2", expected_net_edge_jpy=200.0),
        _candidate("C1", expected_net_edge_jpy=100.0),
    ]

    first = _allocate(
        positions=positions,
        pending=pending,
        candidates=candidates,
        margin_cap=0.9,
    )
    second = _allocate(
        positions=reversed(positions),
        pending=reversed(pending),
        candidates=reversed(candidates),
        margin_cap=0.9,
    )

    assert first == second
    assert first["allocation_sha256"] == second["allocation_sha256"]
    assert [row["intent"]["intent_id"] for row in first["candidate_intent_log"]] == [
        "C1",
        "C2",
    ]
    assert first["candidate_count"] == 2
    assert first["all_candidate_intents_recorded"] is True
    assert first["decision"]["selected_intent_id"] == "C2"
    assert first["live_permission"] is False


def test_caller_notional_is_rejected_and_derived_notional_drives_margin():
    candidate = _candidate("C1", notional_jpy=200_000.0)
    candidate["notional_jpy"] = 1.0
    with pytest.raises(DojoPortfolioAllocatorError, match="schema mismatch"):
        _allocate(candidates=[candidate])

    allocation = _allocate(candidates=[_candidate("C2", notional_jpy=200_000.0)])
    intent = allocation["candidate_intent_log"][0]["intent"]
    assert intent["notional_jpy"] == pytest.approx(200_000.0)
    assert allocation["candidate_intent_log"][0]["initial_admission"][
        "candidate_margin_jpy"
    ] == pytest.approx(20_000.0)


def test_jpy_quote_rejects_spoofed_conversion_rate():
    candidate = _candidate("C1", pair="USD_JPY")
    candidate["jpy_per_quote_unit"] = 1e-6

    with pytest.raises(DojoPortfolioAllocatorError, match="exactly 1.0"):
        _allocate(candidates=[candidate])


@pytest.mark.parametrize(
    ("collection", "row"),
    [
        ("candidates", _candidate("C1", pair="EUR_USD")),
        ("positions", _position("P1", pair="EUR_USD")),
        ("pending", _pending("O1", pair="EUR_USD")),
    ],
)
def test_non_jpy_quotes_fail_closed_without_independent_conversion_receipt(
    collection, row
):
    with pytest.raises(
        DojoPortfolioAllocatorError, match="non-JPY quote is unsupported"
    ):
        _allocate(**{collection: [row]})


def test_self_attested_non_jpy_snapshot_reference_cannot_enable_admission():
    self_attested_source = {
        "contract": "CALLER_SELF_ATTESTED_CONVERSION_V1",
        "instrument": "USD_JPY",
        "observed_epoch": EPOCH,
        "jpy_per_quote_unit": 1e-6,
    }
    candidate = _candidate("C1", pair="EUR_USD")
    candidate.update(
        {
            "units": 1_000.0,
            "jpy_per_quote_unit": 1e-6,
            "conversion_snapshot_id": "self-attested:USD_JPY",
            "conversion_snapshot_sha256": hashlib.sha256(
                json.dumps(
                    self_attested_source,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }
    )

    with pytest.raises(
        DojoPortfolioAllocatorError, match="non-JPY quote is unsupported"
    ):
        _allocate(candidates=[candidate])


def test_pending_orders_consume_shadow_margin_before_fill():
    allocation = _allocate(
        pending=[_pending("O1", notional_jpy=400_000.0)],
        candidates=[_candidate("C1", notional_jpy=200_000.0)],
    )

    assert allocation["account"]["pending_shadow_margin_jpy"] == 40_000.0
    assert allocation["decision"]["action"] == "SKIP"
    admission = allocation["candidate_intent_log"][0]["initial_admission"]
    assert admission["projected_margin_jpy"] == 60_000.0
    assert admission["reason_codes"] == [
        "GLOBAL_MARGIN_CAP_EXCEEDED",
        "PENDING_SHADOW_MARGIN_CONSUMED",
    ]


def test_opposite_pending_orders_do_not_net_currency_shadow():
    allocation = _allocate(
        pending=[
            _pending("O1", pair="EUR_JPY", side="LONG", notional_jpy=60_000.0),
            _pending("O2", pair="EUR_JPY", side="SHORT", notional_jpy=60_000.0),
        ],
        candidates=[_candidate("C1", notional_jpy=10_000.0)],
        margin_cap=0.9,
        currency_cap=0.75,
    )

    assert allocation["account"]["pending_currency_gross_shadow"] == {
        "EUR": 1.2,
        "JPY": 1.2,
    }
    admission = allocation["candidate_intent_log"][0]["initial_admission"]
    assert admission["pending_shadow_breached_currencies"] == ["EUR", "JPY"]
    assert "PENDING_SHADOW_CURRENCY_CAP_EXCEEDED" in admission["reason_codes"]
    assert allocation["decision"]["action"] == "SKIP"


def test_currency_factor_cap_blocks_margin_feasible_candidate():
    allocation = _allocate(
        candidates=[_candidate("C1", pair="EUR_JPY", notional_jpy=100_000.0)],
        margin_cap=0.9,
        currency_cap=0.75,
    )

    assert allocation["decision"]["action"] == "SKIP"
    admission = allocation["candidate_intent_log"][0]["initial_admission"]
    assert admission["projected_margin_fraction"] == 0.1
    assert admission["currency_exposure"]["breached_currencies"] == ["EUR", "JPY"]
    assert admission["reason_codes"] == ["CURRENCY_EXPOSURE_CAP_EXCEEDED"]


def test_stopless_candidate_is_recorded_but_not_admitted():
    allocation = _allocate(candidates=[_candidate("C1", sl_bound=False)])

    row = allocation["candidate_intent_log"][0]
    assert allocation["decision"]["action"] == "SKIP"
    assert row["intent"]["finite_exit_bound"] is False
    assert row["initial_admission"]["reason_codes"] == ["FINITE_EXIT_BOUND_MISSING"]
    assert row["counterfactual_outcome_required"] is True


def test_candidate_loss_cap_uses_derived_stop_loss_bound():
    allocation = _allocate(
        candidates=[_candidate("C1", stop_distance_pips=5_000.0)],
        candidate_loss_cap=0.05,
    )

    row = allocation["candidate_intent_log"][0]
    assert row["intent"]["stop_loss_jpy"] > 5_000.0
    assert row["initial_admission"]["reason_codes"] == ["CANDIDATE_LOSS_CAP_EXCEEDED"]
    assert allocation["decision"]["action"] == "SKIP"


def test_hold_full_when_best_intent_fits_without_releasing_incumbent():
    allocation = _allocate(
        positions=[_position("P1")],
        candidates=[_candidate("C1", expected_net_edge_jpy=250.0)],
    )

    assert allocation["decision"]["action"] == "HOLD_FULL"
    assert allocation["decision"]["reason_codes"] == [
        "SELECTED_INTENT_FITS_WITHOUT_RELEASING_CAPITAL"
    ]
    assert allocation["decision"]["selected_intent_id"] == "C1"
    assert allocation["decision"]["entry_admitted"] is True
    assert allocation["decision"]["selected_plan"]["position_id"] is None
    assert allocation["decision"]["selected_plan"]["reduction_fraction"] == 0.0


def test_hold_reduce_uses_smallest_profitable_release_that_restores_capacity():
    allocation = _allocate(
        positions=[
            _position("P1", notional_jpy=400_000.0, continuation_edge_jpy=100.0)
        ],
        candidates=[
            _candidate(
                "C1",
                owner_id="owner:P1",
                notional_jpy=200_000.0,
                expected_net_edge_jpy=500.0,
            )
        ],
        switching_cost_jpy=20.0,
    )

    plan = allocation["decision"]["selected_plan"]
    assert allocation["decision"]["action"] == "HOLD_REDUCE"
    assert plan["position_id"] == "P1"
    assert plan["reduction_fraction"] == 0.25
    assert plan["post_release_admission"]["projected_margin_jpy"] == 50_000.0
    assert plan["lost_continuation_edge_jpy"] == 25.0
    assert plan["switching_cost_jpy"] == 5.0
    assert plan["incremental_expected_edge_jpy"] == 470.0


def test_cut_rotate_requires_full_release_and_positive_incremental_edge():
    allocation = _allocate(
        positions=[
            _position("P1", notional_jpy=400_000.0, continuation_edge_jpy=100.0)
        ],
        candidates=[
            _candidate(
                "C1",
                owner_id="owner:P1",
                notional_jpy=500_000.0,
                expected_net_edge_jpy=1_000.0,
            )
        ],
        switching_cost_jpy=10.0,
    )

    plan = allocation["decision"]["selected_plan"]
    assert allocation["decision"]["action"] == "CUT_ROTATE"
    assert plan["reduction_fraction"] == 1.0
    assert plan["incremental_expected_edge_jpy"] == 890.0
    assert plan["post_release_admission"]["projected_margin_jpy"] == 50_000.0


def test_cross_owner_position_cannot_be_released_for_candidate():
    allocation = _allocate(
        positions=[_position("P1", owner_id="owner:B", notional_jpy=400_000.0)],
        candidates=[_candidate("C1", owner_id="owner:A", notional_jpy=200_000.0)],
    )

    row = allocation["candidate_intent_log"][0]
    assert allocation["decision"]["action"] == "SKIP"
    assert row["owner_release_eligible_position_ids"] == []
    assert "CROSS_OWNER_RELEASE_FORBIDDEN" in row["disposition_reason_codes"]


def test_owner_at_cap_cannot_preempt_feasible_other_owner_before_ranking():
    allocation = _allocate(
        positions=[
            _position(
                "P1",
                owner_id="owner:A",
                pair="USD_JPY",
                max_reduction_fraction=0.0,
            )
        ],
        candidates=[
            _candidate(
                "C_A",
                owner_id="owner:A",
                expected_net_edge_jpy=1_000.0,
            ),
            _candidate(
                "C_B",
                owner_id="owner:B",
                expected_net_edge_jpy=100.0,
            ),
        ],
        margin_cap=0.9,
        owner_caps={"owner:A": (1, 1), "owner:B": (1, 1)},
    )

    rows = {
        row["intent"]["intent_id"]: row for row in allocation["candidate_intent_log"]
    }
    assert allocation["decision"]["selected_intent_id"] == "C_B"
    assert rows["C_A"]["initial_admission"]["reason_codes"] == [
        "OWNER_PAIR_CONCURRENCY_CAP_REACHED",
        "OWNER_GLOBAL_CONCURRENCY_CAP_REACHED",
    ]
    assert rows["C_A"]["disposition"] == "SKIPPED"


def test_skip_when_rotation_would_destroy_more_edge_than_candidate_adds():
    allocation = _allocate(
        positions=[
            _position("P1", notional_jpy=400_000.0, continuation_edge_jpy=100.0)
        ],
        candidates=[
            _candidate(
                "C1",
                owner_id="owner:P1",
                notional_jpy=500_000.0,
                expected_net_edge_jpy=50.0,
            )
        ],
        switching_cost_jpy=10.0,
    )

    assert allocation["decision"]["action"] == "SKIP"
    assert allocation["decision"]["entry_admitted"] is False
    assert allocation["candidate_intent_log"][0]["disposition"] == "SKIPPED"
    assert (
        "NO_ADMISSIBLE_BOUNDED_RELEASE"
        in allocation["candidate_intent_log"][0]["disposition_reason_codes"]
    )


def test_candidate_margin_hour_efficiency_precedes_raw_edge():
    allocation = _allocate(
        candidates=[
            _candidate("C_SLOW", expected_net_edge_jpy=101.0, holding_seconds=360_000),
            _candidate("C_FAST", expected_net_edge_jpy=100.0, holding_seconds=3_600),
        ]
    )

    assert allocation["decision"]["selected_intent_id"] == "C_FAST"
    rows = {
        row["intent"]["intent_id"]: row for row in allocation["candidate_intent_log"]
    }
    assert rows["C_SLOW"]["disposition_reason_codes"] == [
        "LOWER_PLAN_CAPITAL_EFFICIENCY_OR_TIEBREAK"
    ]


def test_release_plan_efficiency_uses_incremental_edge_after_release_cost():
    allocation = _allocate(
        positions=[
            _position(
                "P1",
                owner_id="owner:C1",
                notional_jpy=200_000.0,
                continuation_edge_jpy=490.0,
            ),
            _position(
                "P2",
                owner_id="owner:C2",
                notional_jpy=200_000.0,
                continuation_edge_jpy=0.0,
            ),
        ],
        candidates=[
            _candidate("C1", notional_jpy=300_000.0, expected_net_edge_jpy=500.0),
            _candidate("C2", notional_jpy=300_000.0, expected_net_edge_jpy=100.0),
        ],
    )

    plan = allocation["decision"]["selected_plan"]
    assert allocation["decision"]["selected_intent_id"] == "C2"
    assert plan["position_id"] == "P2"
    assert plan["incremental_expected_edge_jpy"] == 100.0


def test_opportunity_cost_waits_for_all_counterfactuals_then_scores_regret():
    allocation = _allocate(
        candidates=[
            _candidate("C1", expected_net_edge_jpy=100.0),
            _candidate("C2", expected_net_edge_jpy=200.0),
        ]
    )
    identities = {
        row["intent"]["intent_id"]: row["intent_identity_sha256"]
        for row in allocation["candidate_intent_log"]
    }
    c1 = {
        "intent_id": "C1",
        "intent_identity_sha256": identities["C1"],
        "resolved_net_pnl_jpy": 50.0,
    }
    c2 = {
        "intent_id": "C2",
        "intent_identity_sha256": identities["C2"],
        "resolved_net_pnl_jpy": -10.0,
    }

    pending = score_allocation_opportunity_cost(allocation, [c2])
    assert pending["status"] == "PENDING_COUNTERFACTUAL_OUTCOMES"
    assert pending["missing_intent_ids"] == ["C1"]
    assert pending["candidate_selection_opportunity_loss_jpy"] is None

    first = score_allocation_opportunity_cost(allocation, [c2, c1])
    second = score_allocation_opportunity_cost(allocation, [c1, c2])
    assert first == second
    assert first["status"] == "COMPLETE"
    assert first["selected_intent_id"] == "C2"
    assert first["selected_resolved_net_pnl_jpy"] == -10.0
    assert first["best_counterfactual_intent_id"] == "C1"
    assert first["best_available_net_pnl_jpy"] == 50.0
    assert first["candidate_selection_opportunity_loss_jpy"] == 60.0
    assert first["classification"] == "SELF_ATTESTED_COUNTERFACTUAL_DIAGNOSTIC"
    assert first["score_scope"] == "CANDIDATE_SELECTION_ONLY"
    assert first["incumbent_release_outcome_included"] is False
    assert first["portfolio_opportunity_cost_claim_allowed"] is False
    assert first["allocation_or_entry_admission_allowed"] is False
    assert first["proof_eligible"] is False
    assert first["promotion_eligible"] is False
    assert first["tuning_proof_eligible"] is False


def test_selected_intent_is_never_its_own_counterfactual():
    allocation = _allocate(
        candidates=[
            _candidate("C1", expected_net_edge_jpy=10.0),
            _candidate("C2", expected_net_edge_jpy=20.0),
        ]
    )
    identities = {
        row["intent"]["intent_id"]: row["intent_identity_sha256"]
        for row in allocation["candidate_intent_log"]
    }
    score = score_allocation_opportunity_cost(
        allocation,
        [
            {
                "intent_id": "C1",
                "intent_identity_sha256": identities["C1"],
                "resolved_net_pnl_jpy": 10.0,
            },
            {
                "intent_id": "C2",
                "intent_identity_sha256": identities["C2"],
                "resolved_net_pnl_jpy": 50.0,
            },
        ],
    )

    assert score["selected_intent_id"] == "C2"
    assert score["best_counterfactual_intent_id"] == "C1"
    assert score["candidate_selection_opportunity_loss_jpy"] == 0.0


def test_outcome_must_bind_exact_candidate_identity():
    allocation = _allocate(candidates=[_candidate("C1")])
    outcome = {
        "intent_id": "C1",
        "intent_identity_sha256": "0" * 64,
        "resolved_net_pnl_jpy": 1.0,
    }

    with pytest.raises(DojoPortfolioAllocatorError, match="identity mismatch"):
        score_allocation_opportunity_cost(allocation, [outcome])


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda row: row.update(all_candidate_intents_recorded=False),
            "safety boundary",
        ),
        (lambda row: row.update(schema_version=True), "receipt verification"),
        (
            lambda row: row["decision"].update(action="SKIP"),
            "canonical reconstruction mismatch",
        ),
    ],
)
def test_resealed_receipt_contradictions_fail_closed(mutation, error):
    allocation = deepcopy(_allocate(candidates=[_candidate("C1")]))
    mutation(allocation)
    _reseal(allocation)

    with pytest.raises(DojoPortfolioAllocatorError, match=error):
        score_allocation_opportunity_cost(allocation, [])


def test_resealed_duplicate_candidate_log_fails_closed():
    allocation = deepcopy(_allocate(candidates=[_candidate("C1")]))
    allocation["candidate_intent_log"].append(
        deepcopy(allocation["candidate_intent_log"][0])
    )
    allocation["candidate_count"] = 2
    _reseal(allocation)

    with pytest.raises(DojoPortfolioAllocatorError, match="duplicate candidate"):
        score_allocation_opportunity_cost(allocation, [])


def test_duplicate_candidate_id_and_cross_epoch_candidate_fail_closed():
    duplicate = _candidate("C1")
    with pytest.raises(DojoPortfolioAllocatorError, match="duplicate candidate"):
        _allocate(candidates=[duplicate, deepcopy(duplicate)])

    wrong_epoch = _candidate("C2")
    wrong_epoch["observed_epoch"] = EPOCH - 1
    with pytest.raises(DojoPortfolioAllocatorError, match="decision epoch"):
        _allocate(candidates=[wrong_epoch])
