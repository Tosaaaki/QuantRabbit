from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from quant_rabbit.dojo_bot_catalog import (
    AUTHORITY_INVARIANTS,
    bot_config_risk_vector,
    validate_bot_config,
)
from quant_rabbit.dojo_bot_trainer import (
    DojoBotTrainerError,
    PROPOSAL_CONTRACT as LEGACY_PROPOSAL_CONTRACT,
    seal_candidate_proposal as seal_legacy_candidate_proposal,
)
from quant_rabbit.dojo_g2_baseline import build_g2_baseline
from quant_rabbit.dojo_shared_worker_protocol import seal_post_exit_snapshot
from quant_rabbit.dojo_strategy_catalog_revision_v3 import (
    CATALOG_CONTRACT,
    FAMILY,
    PROPOSAL_CONTRACT,
    DojoStrategyCatalogRevisionV3Error,
    catalog_manifest,
    seal_candidate_proposal,
    session_open_range_break_risk_vector,
    validate_session_open_range_break_config,
)
from quant_rabbit.dojo_tuned_strategy_runtime import (
    build_tuned_strategy_runtime_factory,
    build_tuned_strategy_runtime_seal,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATION_SHA = "d" * 64
START_EPOCH = int(datetime(2025, 1, 15, tzinfo=timezone.utc).timestamp())


def _raw_config(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "signal": FAMILY,
        "pairs": ["USD_JPY"],
        "tp_atr": None,
        "sl_pips": None,
        "ceiling_min": 60,
        "max_concurrent_per_pair": 1,
        "global_max_concurrent": 1,
        "per_pos_lev": 5.0,
        "atr_floor_pips": 0.5,
        "exit_policy": "FIXED",
        "session_buffer_atr": 0.0,
        "session_tp_range": 1.5,
        "session_sl_range": 0.75,
        "max_initial_stop_pips": 15.0,
        **dict(AUTHORITY_INVARIANTS),
    }
    value.update(overrides)
    return value


def _proposal(**config_overrides: Any) -> dict[str, Any]:
    return seal_candidate_proposal(
        {
            "contract": PROPOSAL_CONTRACT,
            "schema_version": 3,
            "candidate_id": "session-open-range-break-bounded-v1",
            "family": FAMILY,
            "hypothesis": (
                "A causal London opening-range break may continue, provided its "
                "atomic rounded initial stop remains inside a sealed pip cap."
            ),
            "config": _raw_config(**config_overrides),
            "risk_increase": False,
        }
    )


def _runtime(**config_overrides: Any):
    seal = build_tuned_strategy_runtime_seal(
        REPO_ROOT,
        candidate_proposals=[_proposal(**config_overrides)],
        generation_ordinal=4,
        generation_binding_sha256=GENERATION_SHA,
    )
    runtime = build_tuned_strategy_runtime_factory(seal, repo_root=REPO_ROOT)(
        {"trade_pairs": ["USD_JPY"], "granularity": "M5", "bar_seconds": 300},
        seal["worker_catalog"],
        None,
    )
    return seal, runtime


def _snapshot(
    bindings: list[dict[str, str]],
    *,
    epoch: int,
    phase: str,
    mid: float,
    watermark: int,
) -> dict[str, Any]:
    timestamp = datetime.fromtimestamp(epoch, timezone.utc).isoformat()
    return seal_post_exit_snapshot(
        {
            "coordinate_id": "session-open-range-break-bounded-m5",
            "epoch": epoch,
            "phase": phase,
            "intrabar": "OHLC",
            "quote_batch_sha256": f"{watermark:064x}",
            "quote_watermark": watermark,
            "expected_quote_pairs": ["USD_JPY"],
            "active_worker_bindings": bindings,
            "account": {
                "balance_jpy": 200_000.0,
                "equity_jpy": 200_000.0,
                "margin_used_jpy": 0.0,
                "accrued_financing_jpy": 0.0,
            },
            "quotes": [
                {
                    "pair": "USD_JPY",
                    "bid": mid - 0.005,
                    "ask": mid + 0.005,
                    "timestamp": f"{timestamp}#{phase}",
                }
            ],
            "positions": [],
            "pending_orders": [],
        }
    )


def _feed_bar(
    runtime: Any,
    bindings: list[dict[str, str]],
    *,
    epoch: int,
    o: float,
    h: float,
    low: float,
    c: float,
    clock: list[int],
) -> dict[str, list[dict[str, Any]]]:
    results = {}
    for phase, mid in (("O", o), ("H", h), ("L", low), ("C", c)):
        clock[0] += 1
        results[phase] = runtime.propose(
            _snapshot(
                bindings,
                epoch=epoch,
                phase=phase,
                mid=mid,
                watermark=clock[0],
            )
        )
    return results


def _feed_opening_range(
    runtime: Any,
    bindings: list[dict[str, str]],
    *,
    high: float = 145.1,
    low: float = 144.9,
) -> list[int]:
    clock = [0]
    for index in range(96):
        _feed_bar(
            runtime,
            bindings,
            epoch=START_EPOCH + index * 300,
            o=145.0,
            h=high,
            low=low,
            c=145.0,
            clock=clock,
        )
    return clock


def test_v3_is_opt_in_and_keeps_legacy_g2_catalog_artifact_unchanged() -> None:
    bounded = validate_session_open_range_break_config(_raw_config())
    legacy_shape = dict(bounded)
    del legacy_shape["max_initial_stop_pips"]

    assert CATALOG_CONTRACT == "QR_DOJO_STRATEGY_CATALOG_REVISION_V3"
    assert catalog_manifest()["family_contract"]["bound_exceeded_action"] == "HOLD"
    assert (
        build_g2_baseline()["artifact_sha256"]
        == "7a12ece80923d3cc2eb173fc166117b9e8d2100ba5e59100af60b1c9c5bbdfbb"
    )
    assert validate_bot_config(legacy_shape)["signal"] == FAMILY
    with pytest.raises(DojoBotTrainerError, match="candidate bot config is invalid"):
        seal_legacy_candidate_proposal(
            {
                "contract": LEGACY_PROPOSAL_CONTRACT,
                "schema_version": 1,
                "candidate_id": "legacy-must-reject-bounded-key",
                "family": FAMILY,
                "hypothesis": "V1 must not silently admit the new stop contract.",
                "config": bounded,
                "risk_increase": False,
            }
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ({"max_initial_stop_pips": 0}, "positive"),
        ({"max_initial_stop_pips": 25.01}, "hard envelope"),
        ({"max_initial_stop_pips": True}, "finite"),
        ({"max_initial_stop_pips": float("inf")}, "finite"),
    ],
)
def test_v3_stop_bound_is_finite_positive_and_inside_hard_envelope(
    mutation: dict[str, Any], match: str
) -> None:
    with pytest.raises(DojoStrategyCatalogRevisionV3Error, match=match):
        validate_session_open_range_break_config(_raw_config(**mutation))

    missing = _raw_config()
    del missing["max_initial_stop_pips"]
    with pytest.raises(DojoStrategyCatalogRevisionV3Error, match="requires"):
        validate_session_open_range_break_config(missing)


def test_bounded_dynamic_stop_is_rankable_without_the_v1_dynamic_blocker() -> None:
    vector = session_open_range_break_risk_vector(
        _raw_config(), stress_slippage_pips_per_fill=0.3
    )
    legacy = _raw_config()
    del legacy["max_initial_stop_pips"]
    legacy_vector = bot_config_risk_vector(legacy, stress_slippage_pips_per_fill=0.3)

    assert vector["rankable"] is True
    assert vector["blocker_codes"] == []
    assert vector["initial_stop_bound_kind"] == (
        "DYNAMIC_CONFIG_MAX_ATOMIC_ROUNDED_PIPS"
    )
    assert vector["max_initial_stop_pips"] == 15.0
    assert vector["single_stop_risk_index"] == pytest.approx(76.5)
    assert legacy_vector["blocker_codes"] == [
        "DYNAMIC_INITIAL_STOP_BOUND_NOT_IMPLEMENTED"
    ]


def test_runtime_emits_atomic_stop_at_or_below_sealed_cap() -> None:
    seal, runtime = _runtime(max_initial_stop_pips=15.0)
    bindings = seal["worker_catalog"]
    clock = _feed_opening_range(runtime, bindings)
    result = _feed_bar(
        runtime,
        bindings,
        epoch=START_EPOCH + 96 * 300,
        o=145.0,
        h=145.35,
        low=145.0,
        c=145.30,
        clock=clock,
    )
    intents = result["C"][0]["new_risk_intents"]

    assert len(intents) == 1
    parameters = intents[0]["parameters"]
    stop_pips = abs(parameters["entry_price"] - parameters["sl_price"]) / 0.01
    assert stop_pips <= 15.0 + 1e-12
    capability = seal["workers"][0]["algorithm_capability"]
    assert capability["bounded_dynamic_initial_stop"] is True
    assert capability["atomic_rounded_stop_cap_enforced"] is True


def test_runtime_holds_and_burns_daily_attempt_when_atomic_stop_exceeds_cap() -> None:
    seal, runtime = _runtime(max_initial_stop_pips=14.9)
    bindings = seal["worker_catalog"]
    clock = _feed_opening_range(runtime, bindings)
    result = _feed_bar(
        runtime,
        bindings,
        epoch=START_EPOCH + 96 * 300,
        o=145.0,
        h=145.35,
        low=145.0,
        c=145.30,
        clock=clock,
    )

    assert result["C"][0]["new_risk_intents"] == []
    assert result["C"][0]["risk_reducing_intents"] == []
    state = runtime.export_state()
    worker = next(iter(state["workers"].values()))
    assert worker["hold_ack_count"] >= 1
    assert worker["pairs"]["USD_JPY"]["session_attempted_day"] == "2025-01-15"


def test_runtime_checks_rounded_atomic_stop_not_only_raw_range_distance() -> None:
    seal, runtime = _runtime(max_initial_stop_pips=14.96)
    bindings = seal["worker_catalog"]
    # Raw stop distance is 14.955 pips, but executable 0.001 quote rounding
    # expands the emitted geometry to 15.0 pips.  The atomic contract must HOLD.
    clock = _feed_opening_range(
        runtime,
        bindings,
        high=145.0997,
        low=144.9003,
    )
    raw_stop_pips = (145.0997 - 144.9003) * 0.75 / 0.01
    assert raw_stop_pips < 14.96

    result = _feed_bar(
        runtime,
        bindings,
        epoch=START_EPOCH + 96 * 300,
        o=145.0,
        h=145.35,
        low=145.0,
        c=145.30,
        clock=clock,
    )

    assert result["C"][0]["new_risk_intents"] == []
