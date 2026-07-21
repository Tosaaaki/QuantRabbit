from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from quant_rabbit.dojo_bot_catalog import (
    AUTHORITY_INVARIANTS,
    CATALOG_CONTRACT as LEGACY_CATALOG_CONTRACT,
    DojoBotCatalogError,
    catalog_manifest as legacy_catalog_manifest,
    validate_bot_config,
)
from quant_rabbit.dojo_bot_trainer import (
    PROPOSAL_CONTRACT as LEGACY_PROPOSAL_CONTRACT,
    DojoBotTrainerError,
    seal_candidate_proposal as seal_legacy_candidate_proposal,
)
from quant_rabbit.dojo_g2_baseline import build_g2_baseline
from quant_rabbit.dojo_shared_worker_protocol import seal_post_exit_snapshot
from quant_rabbit.dojo_strategy_catalog_revision_v2 import (
    CATALOG_CONTRACT,
    ENTRY_POLICY,
    FAMILY,
    PROPOSAL_CONTRACT,
    DojoStrategyCatalogRevisionV2Error,
    catalog_manifest,
    seal_candidate_proposal,
    validate_asia_sweep_reclaim_be_config,
)
from quant_rabbit.dojo_tuned_strategy_runtime import (
    RUNTIME_SEAL_CONTRACT,
    DojoTunedStrategyRuntimeError,
    build_tuned_strategy_runtime_factory,
    build_tuned_strategy_runtime_seal,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATION_SHA = "c" * 64
START_EPOCH = int(datetime(2025, 1, 15, tzinfo=timezone.utc).timestamp())


def _config(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "signal": FAMILY,
        "pairs": ["USD_JPY"],
        "tp_atr": 1.5,
        "sl_pips": 15.0,
        "ceiling_min": 60,
        "max_concurrent_per_pair": 1,
        "global_max_concurrent": 1,
        "per_pos_lev": 2.0,
        "atr_floor_pips": 0.5,
        "exit_policy": "BREAKEVEN",
        "be_trigger_atr": 0.5,
        "be_offset_pips": 1.0,
        **dict(AUTHORITY_INVARIANTS),
    }
    value.update(overrides)
    return validate_asia_sweep_reclaim_be_config(value)


def _proposal() -> dict[str, Any]:
    return seal_candidate_proposal(
        {
            "contract": PROPOSAL_CONTRACT,
            "schema_version": 2,
            "candidate_id": "asia-sweep-reclaim-be-v1",
            "family": FAMILY,
            "hypothesis": (
                "A strict London-Asia range sweep and completed-M5 reclaim may "
                "mean-revert when entered only at the next M5 open."
            ),
            "config": _config(),
            "risk_increase": False,
        }
    )


def _runtime(*, granularity: str = "M5", bar_seconds: int = 300):
    seal = build_tuned_strategy_runtime_seal(
        REPO_ROOT,
        candidate_proposals=[_proposal()],
        generation_ordinal=3,
        generation_binding_sha256=GENERATION_SHA,
    )
    factory = build_tuned_strategy_runtime_factory(seal, repo_root=REPO_ROOT)
    runtime = factory(
        {
            "trade_pairs": ["USD_JPY"],
            "granularity": granularity,
            "bar_seconds": bar_seconds,
        },
        seal["worker_catalog"],
        None,
    )
    return seal, runtime


def _position(
    binding: dict[str, str], *, entry_price: float, opened_epoch: int
) -> dict[str, Any]:
    return {
        "position_id": "pos-asia-short-1",
        "worker_id": binding["worker_id"],
        "owner_id": binding["owner_id"],
        "family_id": binding["family_id"],
        "pair": "USD_JPY",
        "side": "SHORT",
        "units": 1_000.0,
        "entry_price": entry_price,
        "tp_price": entry_price - 0.30,
        "sl_price": entry_price + 0.15,
        "opened_epoch": opened_epoch,
        "hard_exit_epoch": opened_epoch + 3_600,
    }


def _snapshot(
    bindings: list[dict[str, str]],
    *,
    epoch: int,
    phase: str,
    intrabar: str,
    mid: float,
    watermark: int,
    positions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    timestamp = datetime.fromtimestamp(epoch, timezone.utc).isoformat()
    return seal_post_exit_snapshot(
        {
            "coordinate_id": "asia-sweep-reclaim-m5",
            "epoch": epoch,
            "phase": phase,
            "intrabar": intrabar,
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
            "positions": positions or [],
            "pending_orders": [],
        }
    )


def _feed_bar(
    runtime: Any,
    bindings: list[dict[str, str]],
    *,
    epoch: int,
    intrabar: str,
    o: float,
    h: float,
    low: float,
    c: float,
    clock: list[int],
    positions: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    values = {"O": o, "H": h, "L": low, "C": c}
    order = ("O", "H", "L", "C") if intrabar == "OHLC" else ("O", "L", "H", "C")
    results: dict[str, list[dict[str, Any]]] = {}
    for phase in order:
        clock[0] += 1
        results[phase] = runtime.propose(
            _snapshot(
                bindings,
                epoch=epoch,
                phase=phase,
                intrabar=intrabar,
                mid=values[phase],
                watermark=clock[0],
                positions=positions,
            )
        )
    return results


def _feed_complete_asia_range(
    runtime: Any,
    bindings: list[dict[str, str]],
    *,
    intrabar: str,
    clock: list[int],
    start_offset_bars: int = 0,
) -> None:
    for index in range(start_offset_bars, 96):
        _feed_bar(
            runtime,
            bindings,
            epoch=START_EPOCH + index * 300,
            intrabar=intrabar,
            o=145.10,
            h=145.20,
            low=145.00,
            c=145.10,
            clock=clock,
        )


def test_revision_is_explicit_opt_in_and_leaves_v1_and_g2_unchanged() -> None:
    config = _config()

    assert CATALOG_CONTRACT == "QR_DOJO_STRATEGY_CATALOG_REVISION_V2"
    assert catalog_manifest()["family_contract"]["entry_policy"] == ENTRY_POLICY
    assert FAMILY not in {
        row["family_id"] for row in legacy_catalog_manifest()["families"]
    }
    assert LEGACY_CATALOG_CONTRACT == "QR_DOJO_BOT_CATALOG_V1"
    assert (
        build_g2_baseline()["artifact_sha256"]
        == "7a12ece80923d3cc2eb173fc166117b9e8d2100ba5e59100af60b1c9c5bbdfbb"
    )
    with pytest.raises(DojoBotCatalogError, match="not a reviewed DOJO bot family"):
        validate_bot_config(config)
    with pytest.raises(DojoBotTrainerError, match="candidate bot config is invalid"):
        seal_legacy_candidate_proposal(
            {
                "contract": LEGACY_PROPOSAL_CONTRACT,
                "schema_version": 1,
                "candidate_id": "legacy-must-reject-asia",
                "family": FAMILY,
                "hypothesis": "The old proposal contract must not admit V2.",
                "config": config,
                "risk_increase": False,
            }
        )
    with pytest.raises(DojoStrategyCatalogRevisionV2Error, match="BREAKEVEN"):
        _config(exit_policy="FIXED", be_trigger_atr=None, be_offset_pips=None)
    with pytest.raises(DojoStrategyCatalogRevisionV2Error, match="finite initial stop"):
        _config(sl_pips=None)


def test_revision_runtime_is_m5_only() -> None:
    proposal = _proposal()
    seal = build_tuned_strategy_runtime_seal(
        REPO_ROOT,
        candidate_proposals=[proposal],
        generation_ordinal=3,
        generation_binding_sha256=GENERATION_SHA,
    )
    factory = build_tuned_strategy_runtime_factory(seal, repo_root=REPO_ROOT)

    assert seal["contract"] == RUNTIME_SEAL_CONTRACT
    capability = seal["workers"][0]["algorithm_capability"]
    assert capability["m5_close_only"] is True
    assert capability["proposal_on_next_batch_only"] is True
    assert capability["supports_breakeven_overlay"] is True
    assert capability["supports_atr_trailing_overlay"] is False
    assert seal["workers"][0]["owner_id"].startswith("dojo-long-tuned-v2:")
    with pytest.raises(DojoTunedStrategyRuntimeError, match="requires sealed M5"):
        factory(
            {"trade_pairs": ["USD_JPY"], "granularity": "M1", "bar_seconds": 60},
            seal["worker_catalog"],
            None,
        )


@pytest.mark.parametrize("intrabar", ["OHLC", "OLHC"])
def test_completed_m5_reclaim_proposes_only_at_exact_next_open(intrabar: str) -> None:
    seal, runtime = _runtime()
    bindings = seal["worker_catalog"]
    clock = [0]
    _feed_complete_asia_range(runtime, bindings, intrabar=intrabar, clock=clock)

    signal_epoch = START_EPOCH + 96 * 300
    signal_results = _feed_bar(
        runtime,
        bindings,
        epoch=signal_epoch,
        intrabar=intrabar,
        o=145.15,
        h=145.25,
        low=145.10,
        c=145.18,
        clock=clock,
    )

    assert all(
        proposal["new_risk_intents"] == []
        for proposals in signal_results.values()
        for proposal in proposals
    )
    state = runtime.export_state()
    pending = next(iter(state["workers"].values()))["pairs"]["USD_JPY"][
        "asia_sweep_reclaim_pending"
    ]
    assert pending["side"] == "SHORT"
    assert pending["signal_close_epoch"] == signal_epoch
    assert pending["entry_due_epoch"] == signal_epoch + 300

    # Chunk/cell handoff must preserve the one-shot causal signal without
    # reconstructing it from any future candle.
    factory = build_tuned_strategy_runtime_factory(seal, repo_root=REPO_ROOT)
    runtime = factory(
        {"trade_pairs": ["USD_JPY"], "granularity": "M5", "bar_seconds": 300},
        bindings,
        state,
    )

    clock[0] += 1
    next_open = runtime.propose(
        _snapshot(
            bindings,
            epoch=signal_epoch + 300,
            phase="O",
            intrabar=intrabar,
            mid=145.17,
            watermark=clock[0],
        )
    )
    intents = next_open[0]["new_risk_intents"]

    assert len(intents) == 1
    assert intents[0]["action"] == "MARKET"
    assert intents[0]["parameters"]["side"] == "SHORT"
    assert intents[0]["parameters"]["entry_price"] == 145.165
    assert intents[0]["reason_code"] == "ASIA_SWEEP_RECLAIM_BE_NEXT_M5_OPEN"
    assert (
        next(iter(runtime.export_state()["workers"].values()))["pairs"]["USD_JPY"][
            "asia_sweep_reclaim_pending"
        ]
        is None
    )


def test_downside_sweep_reclaim_proposes_long_at_next_ask() -> None:
    seal, runtime = _runtime()
    bindings = seal["worker_catalog"]
    clock = [0]
    _feed_complete_asia_range(runtime, bindings, intrabar="OHLC", clock=clock)
    signal_epoch = START_EPOCH + 96 * 300
    _feed_bar(
        runtime,
        bindings,
        epoch=signal_epoch,
        intrabar="OHLC",
        o=145.05,
        h=145.10,
        low=144.95,
        c=145.02,
        clock=clock,
    )
    clock[0] += 1
    proposals = runtime.propose(
        _snapshot(
            bindings,
            epoch=signal_epoch + 300,
            phase="O",
            intrabar="OHLC",
            mid=145.03,
            watermark=clock[0],
        )
    )
    intent = proposals[0]["new_risk_intents"][0]

    assert intent["parameters"]["side"] == "LONG"
    assert intent["parameters"]["entry_price"] == 145.035


def test_carry_rejects_rehashed_pending_with_opposite_side() -> None:
    seal, runtime = _runtime()
    bindings = seal["worker_catalog"]
    clock = [0]
    _feed_complete_asia_range(runtime, bindings, intrabar="OHLC", clock=clock)
    signal_epoch = START_EPOCH + 96 * 300
    _feed_bar(
        runtime,
        bindings,
        epoch=signal_epoch,
        intrabar="OHLC",
        o=145.15,
        h=145.25,
        low=145.10,
        c=145.18,
        clock=clock,
    )
    tampered_state = runtime.export_state()
    pending = next(iter(tampered_state["workers"].values()))["pairs"]["USD_JPY"][
        "asia_sweep_reclaim_pending"
    ]
    assert pending["side"] == "SHORT"

    # Preserve the pending row's old structural checks while reversing the
    # trade. A caller could reserialize/rehash this state before the next O;
    # restore must derive direction from the authenticated market bar instead.
    pending["side"] = "LONG"
    pending["swept_level"] = pending["range_low"]

    factory = build_tuned_strategy_runtime_factory(seal, repo_root=REPO_ROOT)
    with pytest.raises(
        DojoTunedStrategyRuntimeError,
        match="pending side differs from its signal bar",
    ):
        factory(
            {"trade_pairs": ["USD_JPY"], "granularity": "M5", "bar_seconds": 300},
            bindings,
            tampered_state,
        )


def test_skipping_the_immediate_next_open_burns_the_staged_signal() -> None:
    seal, runtime = _runtime()
    bindings = seal["worker_catalog"]
    clock = [0]
    _feed_complete_asia_range(runtime, bindings, intrabar="OHLC", clock=clock)
    signal_epoch = START_EPOCH + 96 * 300
    _feed_bar(
        runtime,
        bindings,
        epoch=signal_epoch,
        intrabar="OHLC",
        o=145.15,
        h=145.25,
        low=145.10,
        c=145.18,
        clock=clock,
    )
    clock[0] += 1
    proposals = runtime.propose(
        _snapshot(
            bindings,
            epoch=signal_epoch + 600,
            phase="O",
            intrabar="OHLC",
            mid=145.17,
            watermark=clock[0],
        )
    )

    assert proposals[0]["new_risk_intents"] == []
    pair_state = next(iter(runtime.export_state()["workers"].values()))["pairs"][
        "USD_JPY"
    ]
    assert pair_state["asia_sweep_reclaim_pending"] is None


def test_incomplete_asia_range_cannot_stage_or_enter() -> None:
    seal, runtime = _runtime()
    bindings = seal["worker_catalog"]
    clock = [0]
    _feed_complete_asia_range(
        runtime, bindings, intrabar="OHLC", clock=clock, start_offset_bars=1
    )
    signal_epoch = START_EPOCH + 96 * 300
    _feed_bar(
        runtime,
        bindings,
        epoch=signal_epoch,
        intrabar="OHLC",
        o=145.15,
        h=145.25,
        low=145.10,
        c=145.18,
        clock=clock,
    )
    clock[0] += 1
    proposals = runtime.propose(
        _snapshot(
            bindings,
            epoch=signal_epoch + 300,
            phase="O",
            intrabar="OHLC",
            mid=145.17,
            watermark=clock[0],
        )
    )

    assert proposals[0]["new_risk_intents"] == []
    pair_state = next(iter(runtime.export_state()["workers"].values()))["pairs"][
        "USD_JPY"
    ]
    assert pair_state["asia_sweep_reclaim_pending"] is None
    assert pair_state["asia_sweep_reclaim_attempted_day"] is None


def test_breakeven_overlay_is_reused_after_asia_entry() -> None:
    seal, runtime = _runtime()
    bindings = seal["worker_catalog"]
    binding = bindings[0]
    clock = [0]
    _feed_complete_asia_range(runtime, bindings, intrabar="OHLC", clock=clock)
    signal_epoch = START_EPOCH + 96 * 300
    _feed_bar(
        runtime,
        bindings,
        epoch=signal_epoch,
        intrabar="OHLC",
        o=145.15,
        h=145.25,
        low=145.10,
        c=145.18,
        clock=clock,
    )
    entry_epoch = signal_epoch + 300
    clock[0] += 1
    entry_proposals = runtime.propose(
        _snapshot(
            bindings,
            epoch=entry_epoch,
            phase="O",
            intrabar="OHLC",
            mid=145.17,
            watermark=clock[0],
        )
    )
    entry = entry_proposals[0]["new_risk_intents"][0]["parameters"]["entry_price"]
    positions = [_position(binding, entry_price=entry, opened_epoch=entry_epoch)]

    for phase, mid in (("H", 145.18), ("L", 145.12), ("C", 145.14)):
        clock[0] += 1
        first_position_bar = runtime.propose(
            _snapshot(
                bindings,
                epoch=entry_epoch,
                phase=phase,
                intrabar="OHLC",
                mid=mid,
                watermark=clock[0],
                positions=positions,
            )
        )
    assert first_position_bar[0]["risk_reducing_intents"] == []

    next_epoch = entry_epoch + 300
    second_position_bar = _feed_bar(
        runtime,
        bindings,
        epoch=next_epoch,
        intrabar="OHLC",
        o=145.13,
        h=145.14,
        low=144.90,
        c=144.95,
        clock=clock,
        positions=positions,
    )
    reductions = second_position_bar["C"][0]["risk_reducing_intents"]

    assert len(reductions) == 1
    assert reductions[0]["action"] == "TIGHTEN_STOP"
    assert reductions[0]["reason_code"] == "BREAKEVEN_OVERLAY"
    assert reductions[0]["parameters"]["sl_price"] == entry - 0.01
