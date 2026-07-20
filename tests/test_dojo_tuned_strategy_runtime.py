from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from quant_rabbit.dojo_bot_catalog import AUTHORITY_INVARIANTS, validate_bot_config
from quant_rabbit.dojo_bot_trainer import PROPOSAL_CONTRACT, seal_candidate_proposal
from quant_rabbit.dojo_shared_worker_protocol import (
    seal_post_exit_snapshot,
    seal_worker_proposal,
    seal_worker_proposal_batch,
)
from quant_rabbit.dojo_tuned_strategy_runtime import (
    DojoTunedStrategyRuntimeError,
    build_tuned_strategy_runtime_factory,
    build_tuned_strategy_runtime_seal,
    verify_tuned_strategy_runtime_seal,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATION_SHA = "a" * 64
FAMILIES = (
    "burst",
    "compression_break",
    "daily_break_pullback",
    "fade_ladder",
    "mean_revert_24h",
    "prev_day_extreme_fade",
    "pullback_limit",
    "range_fade_limit",
    "round_number_fade",
    "session_open_range_break",
    "spike_fade",
    "weekend_gap_recovery",
)


def _config(family: str, *, tp_atr: float = 3.0) -> dict[str, Any]:
    dynamic = family in {"session_open_range_break", "weekend_gap_recovery"}
    value: dict[str, Any] = {
        "signal": family,
        "pairs": ["USD_JPY"],
        "tp_atr": None if dynamic else tp_atr,
        "sl_pips": None if dynamic else 25.0,
        "ceiling_min": 60,
        "max_concurrent_per_pair": 1,
        "global_max_concurrent": 1,
        "per_pos_lev": 5.0,
        "atr_floor_pips": 0.5,
        "exit_policy": "FIXED",
        **dict(AUTHORITY_INVARIANTS),
    }
    if family == "pullback_limit":
        value["pull_atr"] = 0.6
    if family in {"mean_revert_24h", "fade_ladder", "range_fade_limit"}:
        value["fade_atr"] = 1.2
    if family in {"fade_ladder", "range_fade_limit"}:
        value["eff_max"] = 0.2
    if family == "fade_ladder":
        value["max_concurrent_per_pair"] = 2
        value["global_max_concurrent"] = 2
    if family == "session_open_range_break":
        value.update(
            {
                "session_buffer_atr": 0.2,
                "session_tp_range": 1.5,
                "session_sl_range": 0.75,
            }
        )
    if family == "weekend_gap_recovery":
        value.update(
            {
                "weekend_gap_atr": 4.0,
                "weekend_sl_gap": 1.0,
                "weekend_wait_bars": 15,
                "weekend_spread_fraction": 0.15,
            }
        )
    return validate_bot_config(value)


def _proposal(family: str, *, suffix: str = "", tp_atr: float = 3.0) -> dict[str, Any]:
    return seal_candidate_proposal(
        {
            "contract": PROPOSAL_CONTRACT,
            "schema_version": 1,
            "candidate_id": f"tuned-{family.replace('_', '-')}{suffix}",
            "family": family,
            "hypothesis": f"A sealed {family} candidate.",
            "config": _config(family, tp_atr=tp_atr),
            "risk_increase": False,
        }
    )


def _seal(*families: str) -> dict[str, Any]:
    return build_tuned_strategy_runtime_seal(
        REPO_ROOT,
        candidate_proposals=[_proposal(family) for family in families],
        generation_ordinal=2,
        generation_binding_sha256=GENERATION_SHA,
    )


def _snapshot(
    bindings: list[dict[str, str]], *, epoch: int, phase: str, watermark: int
) -> dict[str, Any]:
    timestamp = datetime.fromtimestamp(epoch, timezone.utc).isoformat()
    return seal_post_exit_snapshot(
        {
            "coordinate_id": "tuned-coordinate",
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
                    "bid": 144.99,
                    "ask": 145.01,
                    "timestamp": f"{timestamp}#{phase}",
                }
            ],
            "positions": [],
            "pending_orders": [],
        }
    )


def test_seal_accepts_every_explicitly_implemented_family_and_binds_capabilities() -> (
    None
):
    seal = _seal(*FAMILIES)

    assert verify_tuned_strategy_runtime_seal(seal, repo_root=REPO_ROOT) == seal
    assert seal["worker_count"] == len(FAMILIES)
    assert set(seal["supported_family_allowlist"]) == set(FAMILIES)
    assert {row["family_id"] for row in seal["workers"]} == set(FAMILIES)
    assert all(row["algorithm_capability_sha256"] for row in seal["workers"])
    assert seal["capabilities"]["arbitrary_import_allowed"] is False
    assert seal["capabilities"]["candidate_declared_plugin_allowed"] is False
    assert seal["capabilities"]["runtime_seal_mutation_allowed"] is False
    assert seal["capabilities"]["explicit_hold_ack_required"] is True
    assert seal["capabilities"]["order_authority"] == "NONE"


def test_timeframe_profiles_use_equal_elapsed_horizons_but_unequal_bar_counts() -> None:
    seal = _seal("burst")
    profiles = {row["granularity"]: row for row in seal["timeframe_profiles"]}

    assert profiles["M1"]["trend_horizon_seconds"] == 86_400
    assert profiles["M5"]["trend_horizon_seconds"] == 86_400
    assert profiles["M1"]["trend_close_count"] == 1_441
    assert profiles["M5"]["trend_close_count"] == 289
    assert profiles["M1"]["efficiency_diff_count"] == 360
    assert profiles["M5"]["efficiency_diff_count"] == 72
    assert profiles["M1"]["width_window_count"] == 20
    assert profiles["M5"]["width_window_count"] == 4
    assert profiles["M1"]["prior_swing_bar_count"] == 3
    assert profiles["M5"]["prior_swing_bar_count"] == 1


def test_seal_rejects_unsealed_or_normalized_equivalent_candidates() -> None:
    sealed = _proposal("burst")
    unsealed = {
        key: sealed[key]
        for key in (
            "contract",
            "schema_version",
            "candidate_id",
            "family",
            "hypothesis",
            "config",
            "risk_increase",
        )
    }
    with pytest.raises(DojoTunedStrategyRuntimeError, match="not sealed"):
        build_tuned_strategy_runtime_seal(
            REPO_ROOT,
            candidate_proposals=[unsealed],
            generation_ordinal=2,
            generation_binding_sha256=GENERATION_SHA,
        )

    equivalent = _proposal("burst", suffix="-copy")
    with pytest.raises(DojoTunedStrategyRuntimeError, match="equivalent"):
        build_tuned_strategy_runtime_seal(
            REPO_ROOT,
            candidate_proposals=[sealed, equivalent],
            generation_ordinal=2,
            generation_binding_sha256=GENERATION_SHA,
        )


def test_factory_is_immutable_and_every_worker_explicitly_acknowledges_hold() -> None:
    seal = _seal("burst", "range_fade_limit")
    factory = build_tuned_strategy_runtime_factory(seal, repo_root=REPO_ROOT)
    with pytest.raises(AttributeError, match="immutable"):
        factory.runtime_binding_sha256 = "b" * 64  # type: ignore[misc]
    runtime = factory(
        {"trade_pairs": ["USD_JPY"], "granularity": "M5", "bar_seconds": 300},
        seal["worker_catalog"],
        None,
    )
    snapshot = _snapshot(
        seal["worker_catalog"], epoch=1_700_000_100, phase="O", watermark=1
    )
    proposals = runtime.propose(snapshot)
    batch = seal_worker_proposal_batch(
        snapshot, [seal_worker_proposal(snapshot, row) for row in proposals]
    )

    assert batch["proposal_count"] == 2
    assert batch["new_risk_intent_count"] == 0
    assert all(row["risk_reducing_intents"] == [] for row in proposals)
    state = runtime.export_state()
    assert state["runtime_binding_sha256"] == seal["runtime_binding_sha256"]
    assert state["granularity"] == "M5"
    assert state["cadence_seconds"] == 300
    assert all(row["hold_ack_count"] == 1 for row in state["workers"].values())


def test_factory_rejects_cadence_drift_cross_timeframe_carry_and_seal_tamper() -> None:
    seal = _seal("burst")
    factory = build_tuned_strategy_runtime_factory(seal, repo_root=REPO_ROOT)
    with pytest.raises(DojoTunedStrategyRuntimeError, match="timeframe/cadence"):
        factory(
            {"trade_pairs": ["USD_JPY"], "granularity": "M5", "bar_seconds": 60},
            seal["worker_catalog"],
            None,
        )

    m1 = factory(
        {"trade_pairs": ["USD_JPY"], "granularity": "M1", "bar_seconds": 60},
        seal["worker_catalog"],
        None,
    )
    prior = m1.export_state()
    with pytest.raises(DojoTunedStrategyRuntimeError, match="binding drifted"):
        factory(
            {"trade_pairs": ["USD_JPY"], "granularity": "M5", "bar_seconds": 300},
            seal["worker_catalog"],
            prior,
        )

    tampered = copy.deepcopy(seal)
    tampered["workers"][0]["config"]["ceiling_min"] += 1
    with pytest.raises(DojoTunedStrategyRuntimeError, match="differs"):
        verify_tuned_strategy_runtime_seal(tampered, repo_root=REPO_ROOT)
