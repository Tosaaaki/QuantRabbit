from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pytest

from quant_rabbit.dojo_builtin_strategy_runtime import (
    ALGORITHM_REVISION,
    DojoBuiltinStrategyRuntimeError,
    build_builtin_strategy_runtime_seal,
    builtin_strategy_runtime_factory,
    verify_builtin_strategy_runtime_seal,
)
from quant_rabbit.dojo_shared_worker_protocol import (
    seal_post_exit_snapshot,
    seal_worker_proposal,
    seal_worker_proposal_batch,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PAIRS = ("AUD_USD", "EUR_USD", "GBP_USD", "NZD_USD", "USD_JPY")
BASES = {
    "AUD_USD": 0.6500,
    "EUR_USD": 1.1000,
    "GBP_USD": 1.3000,
    "NZD_USD": 0.6100,
    "USD_JPY": 145.00,
}


def _snapshot(
    *,
    bindings: list[dict[str, str]],
    epoch: int,
    phase: str,
    watermark: int,
    offsets_pips: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    offsets = dict(offsets_pips or {})
    quotes = []
    timestamp = datetime.fromtimestamp(epoch, timezone.utc).isoformat()
    for pair in PAIRS:
        pip = 0.01 if pair.endswith("JPY") else 0.0001
        mid = BASES[pair] + offsets.get(pair, 0.0) * pip
        spread = 2.0 * pip
        quotes.append(
            {
                "pair": pair,
                "bid": mid - spread / 2.0,
                "ask": mid + spread / 2.0,
                "timestamp": f"{timestamp}#{phase}",
            }
        )
    return seal_post_exit_snapshot(
        {
            "coordinate_id": "coordinate-test",
            "epoch": epoch,
            "phase": phase,
            "intrabar": "OHLC",
            "quote_batch_sha256": f"{watermark:064x}",
            "quote_watermark": watermark,
            "expected_quote_pairs": list(PAIRS),
            "active_worker_bindings": bindings,
            "account": {
                "balance_jpy": 200_000.0,
                "equity_jpy": 200_000.0,
                "margin_used_jpy": 0.0,
                "accrued_financing_jpy": 0.0,
            },
            "quotes": quotes,
            "positions": [],
            "pending_orders": [],
        }
    )


def _runtime() -> tuple[Any, dict[str, Any]]:
    seal = build_builtin_strategy_runtime_seal(REPO_ROOT)
    runtime = builtin_strategy_runtime_factory(
        {"trade_pairs": list(PAIRS)}, seal["worker_catalog"], None
    )
    return runtime, seal


def _seed_all_family_histories(state: dict[str, Any], *, epoch: int) -> None:
    for worker in state["workers"].values():
        for pair, pair_state in worker["pairs"].items():
            pip = 0.01 if pair.endswith("JPY") else 0.0001
            base = BASES[pair]
            closes = [base - 10.0 * pip]
            closes.extend(base + 5.0 * pip for _ in range(1440))
            pair_state.update(
                {
                    "atr": 5.0 * pip,
                    "closes": closes,
                    "diffs_6h": [4.0 * pip for _ in range(360)],
                    "widths": [100.0 * pip for _ in range(360)],
                    "forming": None,
                    "last_closed_epoch": epoch - 60,
                    "day": datetime.fromtimestamp(
                        epoch, timezone.utc
                    ).date().isoformat(),
                    "day_high": base,
                    "day_low": base - 2.0 * pip,
                    "day_observation_count": 60,
                    "previous_day_high": base + 1.0 * pip,
                    "previous_day_low": base - 20.0 * pip,
                }
            )


def test_runtime_seal_binds_exact_four_workers_configs_and_dependencies() -> None:
    seal = build_builtin_strategy_runtime_seal(REPO_ROOT)
    assert verify_builtin_strategy_runtime_seal(seal, repo_root=REPO_ROOT) == seal
    assert seal["worker_count"] == 4
    assert {row["family_id"] for row in seal["worker_catalog"]} == {
        "compression_break",
        "daily_break_pullback",
        "range_fade_limit",
        "spike_fade",
    }
    assert all(row["algorithm_revision"] == ALGORITHM_REVISION for row in seal["workers"])
    assert seal["capabilities"] == {
        "arbitrary_import_allowed": False,
        "broker_handle_available": False,
        "broker_mutation_allowed": False,
        "external_code_loading_allowed": False,
        "filesystem_available_to_worker": False,
        "live_permission": False,
        "network_available_to_worker": False,
        "order_authority": "NONE",
        "proposal_only": True,
    }
    assert seal["evidence"]["official_evidence_eligible"] is False
    assert seal["evidence"]["independent_reexecution_available"] is False


def test_runtime_seal_rejects_dependency_or_config_drift() -> None:
    seal = build_builtin_strategy_runtime_seal(REPO_ROOT)
    dependency_tamper = copy.deepcopy(seal)
    dependency_tamper["dependencies"][0]["sha256"] = "f" * 64
    with pytest.raises(
        DojoBuiltinStrategyRuntimeError, match="closed dependency denominator"
    ):
        verify_builtin_strategy_runtime_seal(dependency_tamper, repo_root=REPO_ROOT)

    config_tamper = copy.deepcopy(seal)
    config_tamper["workers"][0]["config"]["ceiling_min"] += 1
    with pytest.raises(
        DojoBuiltinStrategyRuntimeError, match="closed dependency denominator"
    ):
        verify_builtin_strategy_runtime_seal(config_tamper, repo_root=REPO_ROOT)


def test_every_worker_explicitly_acknowledges_hold_on_non_close_phase() -> None:
    runtime, seal = _runtime()
    snapshot = _snapshot(
        bindings=seal["worker_catalog"],
        epoch=1_700_000_000,
        phase="O",
        watermark=1,
    )
    proposals = runtime.propose(snapshot)
    assert len(proposals) == 4
    assert all(row["risk_reducing_intents"] == [] for row in proposals)
    assert all(row["new_risk_intents"] == [] for row in proposals)
    batch = seal_worker_proposal_batch(
        snapshot, [seal_worker_proposal(snapshot, row) for row in proposals]
    )
    assert batch["proposal_count"] == 4
    assert batch["new_risk_intent_count"] == 0
    state = runtime.export_state()
    assert all(row["hold_ack_count"] == 1 for row in state["workers"].values())


def test_four_families_emit_protocol_valid_multi_pair_proposals() -> None:
    empty_runtime, seal = _runtime()
    prior = empty_runtime.export_state()
    epoch = 1_700_006_400
    _seed_all_family_histories(prior, epoch=epoch)
    runtime = builtin_strategy_runtime_factory(
        {"trade_pairs": list(PAIRS)}, seal["worker_catalog"], prior
    )
    phases = (
        ("O", 0.0),
        ("H", 30.0),
        ("L", -10.0),
        ("C", 2.0),
    )
    proposals = []
    snapshot = None
    for watermark, (phase, offset) in enumerate(phases, 1):
        snapshot = _snapshot(
            bindings=seal["worker_catalog"],
            epoch=epoch,
            phase=phase,
            watermark=watermark,
            offsets_pips={pair: offset for pair in PAIRS},
        )
        proposals = runtime.propose(snapshot)
    assert snapshot is not None
    sealed = [seal_worker_proposal(snapshot, row) for row in proposals]
    batch = seal_worker_proposal_batch(snapshot, sealed)
    by_family = {row["family_id"]: row for row in sealed}
    assert set(by_family) == {
        "compression_break",
        "daily_break_pullback",
        "range_fade_limit",
        "spike_fade",
    }
    assert all(row["new_risk_intents"] for row in by_family.values())
    assert {
        intent["parameters"]["pair"]
        for row in sealed
        for intent in row["new_risk_intents"]
    } == set(PAIRS[:4])
    assert batch["proposal_count"] == 4
    assert batch["new_risk_intent_count"] == 16
    assert {intent["action"] for intent in by_family["compression_break"]["new_risk_intents"]} == {"STOP"}
    assert {
        intent["action"]
        for family in ("daily_break_pullback", "range_fade_limit", "spike_fade")
        for intent in by_family[family]["new_risk_intents"]
    } == {"LIMIT"}


def test_mid_candle_carry_is_byte_equivalent_to_uninterrupted_runtime() -> None:
    uninterrupted, seal = _runtime()
    epoch = 1_700_012_000
    opening = _snapshot(
        bindings=seal["worker_catalog"], epoch=epoch, phase="O", watermark=1
    )
    uninterrupted.propose(opening)
    carried = builtin_strategy_runtime_factory(
        {"trade_pairs": list(PAIRS)},
        seal["worker_catalog"],
        uninterrupted.export_state(),
    )
    for watermark, phase in enumerate(("H", "L", "C"), 2):
        snapshot = _snapshot(
            bindings=seal["worker_catalog"],
            epoch=epoch,
            phase=phase,
            watermark=watermark,
        )
        assert uninterrupted.propose(snapshot) == carried.propose(snapshot)
        assert uninterrupted.export_state() == carried.export_state()


def test_prior_state_cannot_swap_a_worker_config() -> None:
    runtime, seal = _runtime()
    prior = runtime.export_state()
    worker_id = seal["worker_catalog"][0]["worker_id"]
    prior["workers"][worker_id]["config_sha256"] = "f" * 64
    with pytest.raises(
        DojoBuiltinStrategyRuntimeError, match="identity/counter binding drifted"
    ):
        builtin_strategy_runtime_factory(
            {"trade_pairs": list(PAIRS)}, seal["worker_catalog"], prior
        )


def test_phase_skip_or_backward_clock_is_rejected() -> None:
    runtime, seal = _runtime()
    epoch = 1_700_018_000
    runtime.propose(
        _snapshot(
            bindings=seal["worker_catalog"],
            epoch=epoch,
            phase="O",
            watermark=1,
        )
    )
    with pytest.raises(
        DojoBuiltinStrategyRuntimeError, match="phase sequence is non-causal"
    ):
        runtime.propose(
            _snapshot(
                bindings=seal["worker_catalog"],
                epoch=epoch,
                phase="C",
                watermark=2,
            )
        )
