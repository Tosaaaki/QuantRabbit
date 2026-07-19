from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import pytest

from quant_rabbit.dojo_allocation_execution import (
    DojoAllocationExecutionError,
    DojoAllocationExecutionSession,
)
from quant_rabbit.dojo_lab_provenance import (
    StrategyOwnershipError,
    strategy_ownership_registry,
)
from quant_rabbit.dojo_portfolio_allocator import (
    DojoPortfolioAllocatorError,
    build_portfolio_allocation,
)
from quant_rabbit.virtual_broker import VirtualBroker, VirtualBrokerError


EPOCH = 1_700_000_000
QUOTE_TS = "2023-11-14T22:13:20+00:00"
NEXT_QUOTE_TS = "2023-11-14T22:14:20+00:00"


def _candidate(
    intent_id: str,
    *,
    owner_id: str,
    pair: str = "USD_JPY",
    side: str = "LONG",
    order_kind: str = "MARKET",
    units: float = 1_000.0,
    entry_price: float = 150.0,
    expected_net_edge_jpy: float = 100.0,
    stop_pips: float = 10.0,
) -> dict[str, object]:
    pip = 0.01
    return {
        "intent_id": intent_id,
        "observed_epoch": EPOCH,
        "owner_id": owner_id,
        "strategy_family": "e2e_test",
        "pair": pair,
        "side": side,
        "order_kind": order_kind,
        "units": units,
        "entry_price": entry_price,
        "jpy_per_quote_unit": 1.0,
        "conversion_snapshot_id": None,
        "conversion_snapshot_sha256": None,
        "tp_price": (
            entry_price + stop_pips * pip
            if side == "LONG"
            else entry_price - stop_pips * pip
        ),
        "sl_price": (
            entry_price - stop_pips * pip
            if side == "LONG"
            else entry_price + stop_pips * pip
        ),
        "stress_cost_pips": 0.2,
        "expected_net_edge_jpy": expected_net_edge_jpy,
        "expected_holding_seconds": 3_600,
        "valid_until_epoch": EPOCH + 60,
    }


def _open_position(
    broker: VirtualBroker,
    trade_id: str,
    *,
    owner_id: str,
    continuation_edge_jpy: float = 10.0,
    max_reduction_fraction: float = 0.5,
) -> dict[str, object]:
    position = broker.positions[trade_id]
    quote = broker.last_quotes[position.pair]
    assert position.sl_price is not None
    return {
        "position_id": trade_id,
        "owner_id": owner_id,
        "pair": position.pair,
        "side": position.side,
        "units": position.units,
        "mark_price": (quote[0] + quote[1]) / 2.0,
        "bid_price": quote[0],
        "ask_price": quote[1],
        "quote_timestamp": quote[2],
        "quote_sequence": broker._last_quote_watermarks[position.pair],
        "sl_price": position.sl_price,
        "stress_cost_pips": 0.2,
        "jpy_per_quote_unit": 1.0,
        "conversion_snapshot_id": None,
        "conversion_snapshot_sha256": None,
        "continuation_edge_jpy": continuation_edge_jpy,
        "max_reduction_fraction": max_reduction_fraction,
    }


def _pending_order(
    broker: VirtualBroker, order_id: str, *, owner_id: str
) -> dict[str, object]:
    order = broker.orders[order_id]
    assert order.sl_pips is not None
    return {
        "order_id": order_id,
        "owner_id": owner_id,
        "pair": order.pair,
        "side": order.side,
        "units": order.units,
        "trigger_price": order.limit_price,
        "sl_pips": order.sl_pips,
        "stress_cost_pips": 0.2,
        "jpy_per_quote_unit": 1.0,
        "conversion_snapshot_id": None,
        "conversion_snapshot_sha256": None,
    }


def _allocate(
    broker: VirtualBroker,
    *,
    owner_id: str,
    pair_cap: int,
    global_cap: int,
    candidates: list[dict[str, object]],
    positions: list[dict[str, object]] | None = None,
    pending: list[dict[str, object]] | None = None,
    margin_cap: float = 0.9,
) -> dict[str, object]:
    decision_pair = str(candidates[0]["pair"])
    decision_quote = broker.last_quotes[decision_pair]
    decision_sequence = broker._last_quote_watermarks[decision_pair]
    return build_portfolio_allocation(
        decision_epoch=EPOCH,
        decision_quote_timestamp=decision_quote[2],
        decision_quote_sequence=decision_sequence,
        equity_jpy=float(broker.account()["equity_jpy"]),
        leverage=broker.leverage,
        global_margin_cap_fraction=margin_cap,
        currency_cap_fraction=10.0,
        max_candidate_loss_fraction=0.1,
        max_portfolio_loss_fraction=0.2,
        owner_concurrency_caps=[
            {
                "owner_id": owner_id,
                "max_concurrent_per_pair": pair_cap,
                "global_max_concurrent": global_cap,
            }
        ],
        open_positions=positions or [],
        pending_orders=pending or [],
        candidate_intents=candidates,
    )


def _selected(allocation: dict[str, object]) -> dict[str, object]:
    rows = allocation["candidate_intent_log"]
    assert isinstance(rows, list)
    return next(dict(row["intent"]) for row in rows if row["disposition"] == "SELECTED")


def _prepare(
    session: DojoAllocationExecutionSession,
    allocation: dict[str, object],
    *,
    run_id: str,
    nonce: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    selected = _selected(allocation)
    config = session.seal_config(run_id=run_id)
    prepared = session.prepare(
        allocation=allocation,
        selected_intent=selected,
        sealed_config=config,
        execution_nonce=nonce,
    )
    return selected, config, prepared


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _reseal_submission(submission: dict[str, object]) -> dict[str, object]:
    core = {
        key: value
        for key, value in submission.items()
        if key
        not in {
            "submission_core_sha256",
            "binding_ledger_sha256",
            "submission_sha256",
        }
    }
    submission["submission_core_sha256"] = _canonical_sha(core)
    body = {
        key: value for key, value in submission.items() if key != "submission_sha256"
    }
    submission["submission_sha256"] = _canonical_sha(body)
    return submission


def test_session_rejects_virtual_broker_subclass(tmp_path: Path) -> None:
    class BrokerLookalike(VirtualBroker):
        pass

    broker = BrokerLookalike(tmp_path / "ledger.jsonl", fast_ledger=True)
    with pytest.raises(DojoAllocationExecutionError, match="exact VirtualBroker"):
        DojoAllocationExecutionSession(
            broker,
            "e2e:lookalike",
            max_concurrent_per_pair=1,
            global_max_concurrent=1,
        )


def test_parallel_first_session_construction_shares_one_broker_lock(
    tmp_path: Path,
) -> None:
    broker = VirtualBroker(tmp_path / "parallel-session-construction.jsonl")
    barrier = threading.Barrier(8)

    def construct(index: int) -> DojoAllocationExecutionSession:
        barrier.wait()
        return DojoAllocationExecutionSession(
            broker,
            f"owner_{index}",
            max_concurrent_per_pair=1,
            global_max_concurrent=1,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        sessions = list(pool.map(construct, range(8)))

    locks = {id(session._broker_lock) for session in sessions}
    assert len(locks) == 1
    assert sessions[0]._broker_lock is getattr(
        broker, "_dojo_allocation_execution_lock"
    )


def test_selected_intent_only_is_hash_bound_and_one_shot(tmp_path: Path) -> None:
    broker = VirtualBroker(
        tmp_path / "ledger.jsonl",
        balance_jpy=200_000.0,
        leverage=10.0,
        fast_ledger=True,
    )
    broker.on_quote("USD_JPY", 149.99, 150.00, QUOTE_TS)
    owner_id = "e2e:selector"
    session = DojoAllocationExecutionSession(
        broker,
        owner_id,
        max_concurrent_per_pair=1,
        global_max_concurrent=2,
    )
    allocation = _allocate(
        broker,
        owner_id=owner_id,
        pair_cap=1,
        global_cap=2,
        candidates=[
            _candidate("C-best", owner_id=owner_id, expected_net_edge_jpy=200.0),
            _candidate("C-skip", owner_id=owner_id, expected_net_edge_jpy=100.0),
        ],
    )
    selected, config, prepared = _prepare(
        session, allocation, run_id="run:selector", nonce="nonce:selector"
    )

    skipped = next(
        dict(row["intent"])
        for row in allocation["candidate_intent_log"]
        if row["disposition"] == "SKIPPED"
    )
    with pytest.raises(DojoAllocationExecutionError, match="exact selected"):
        session.prepare(
            allocation=allocation,
            selected_intent=skipped,
            sealed_config=config,
            execution_nonce="nonce:unselected",
        )

    tampered = deepcopy(prepared)
    tampered["units"] = float(tampered["units"]) + 1
    with pytest.raises(DojoAllocationExecutionError, match="tampered"):
        session.submit(
            allocation=allocation,
            selected_intent=selected,
            sealed_config=config,
            prepared_submission=tampered,
            execution_epoch=EPOCH,
        )

    submission = session.submit(
        allocation=allocation,
        selected_intent=selected,
        sealed_config=config,
        prepared_submission=prepared,
        execution_epoch=EPOCH,
    )
    verification = session.verify(submission)
    assert verification["status"] == "FILLED"
    assert verification["owner_verified"] is True
    assert verification["owner_concurrency_within_caps"] is True
    assert verification["allocation_time_aggregate_sl_cap_enforced"] is True
    assert verification["fill_time_aggregate_sl_cap_rechecked"] is False
    assert submission["external_broker_authority"] == "NONE"
    assert submission["live_permission"] is False

    forged = _reseal_submission({**deepcopy(submission), "pair": "EUR_JPY"})
    with pytest.raises(DojoAllocationExecutionError, match="not bound"):
        session.verify(forged)

    with pytest.raises(DojoAllocationExecutionError, match="already claimed"):
        session.submit(
            allocation=allocation,
            selected_intent=selected,
            sealed_config=config,
            prepared_submission=prepared,
            execution_epoch=EPOCH,
        )


@pytest.mark.parametrize(
    ("attribute", "changed_value"),
    [("leverage", 11.0), ("slippage_pips", 0.1)],
)
def test_sealed_execution_settings_reject_broker_mutation(
    tmp_path: Path, attribute: str, changed_value: float
) -> None:
    broker = VirtualBroker(
        tmp_path / f"{attribute}.jsonl",
        balance_jpy=200_000.0,
        leverage=10.0,
        fast_ledger=True,
    )
    broker.on_quote("USD_JPY", 149.99, 150.00, QUOTE_TS)
    owner_id = f"e2e:settings-{attribute}"
    session = DojoAllocationExecutionSession(
        broker,
        owner_id,
        max_concurrent_per_pair=1,
        global_max_concurrent=1,
    )
    allocation = _allocate(
        broker,
        owner_id=owner_id,
        pair_cap=1,
        global_cap=1,
        candidates=[_candidate("C-settings", owner_id=owner_id)],
    )
    selected, config, prepared = _prepare(
        session,
        allocation,
        run_id=f"run:settings-{attribute}",
        nonce=f"nonce:settings-{attribute}",
    )
    setattr(broker, attribute, changed_value)
    with pytest.raises(DojoAllocationExecutionError, match="settings changed"):
        session.submit(
            allocation=allocation,
            selected_intent=selected,
            sealed_config=config,
            prepared_submission=prepared,
            execution_epoch=EPOCH,
        )
    assert not broker.positions


def test_tampered_config_is_rejected_even_when_prepared_was_valid(
    tmp_path: Path,
) -> None:
    broker = VirtualBroker(
        tmp_path / "ledger.jsonl", balance_jpy=200_000.0, fast_ledger=True
    )
    broker.on_quote("USD_JPY", 149.99, 150.00, QUOTE_TS)
    owner_id = "e2e:config"
    session = DojoAllocationExecutionSession(
        broker, owner_id, max_concurrent_per_pair=1, global_max_concurrent=1
    )
    allocation = _allocate(
        broker,
        owner_id=owner_id,
        pair_cap=1,
        global_cap=1,
        candidates=[_candidate("C-config", owner_id=owner_id)],
    )
    selected, config, prepared = _prepare(
        session, allocation, run_id="run:config", nonce="nonce:config"
    )
    tampered = deepcopy(config)
    tampered["expected_slippage_pips"] = 0.1
    with pytest.raises(DojoAllocationExecutionError, match="config is invalid"):
        session.submit(
            allocation=allocation,
            selected_intent=selected,
            sealed_config=tampered,
            prepared_submission=prepared,
            execution_epoch=EPOCH,
        )


def test_execution_epoch_must_exactly_match_selected_quote_epoch(
    tmp_path: Path,
) -> None:
    broker = VirtualBroker(
        tmp_path / "quote-epoch.jsonl",
        balance_jpy=200_000.0,
        leverage=10.0,
        fast_ledger=True,
    )
    broker.on_quote("USD_JPY", 149.99, 150.00, QUOTE_TS)
    owner_id = "e2e:quote-epoch"
    session = DojoAllocationExecutionSession(
        broker, owner_id, max_concurrent_per_pair=1, global_max_concurrent=1
    )
    allocation = _allocate(
        broker,
        owner_id=owner_id,
        pair_cap=1,
        global_cap=1,
        candidates=[_candidate("C-quote-epoch", owner_id=owner_id)],
    )
    selected, config, prepared = _prepare(
        session,
        allocation,
        run_id="run:quote-epoch",
        nonce="nonce:quote-epoch",
    )
    with pytest.raises(DojoAllocationExecutionError, match="exact selected quote"):
        session.submit(
            allocation=allocation,
            selected_intent=selected,
            sealed_config=config,
            prepared_submission=prepared,
            execution_epoch=EPOCH + 1,
        )
    assert not broker.positions


def test_stale_selected_quote_epoch_fails_at_allocator_boundary(tmp_path: Path) -> None:
    broker = VirtualBroker(
        tmp_path / "stale-quote-epoch.jsonl",
        balance_jpy=200_000.0,
        leverage=10.0,
        fast_ledger=True,
    )
    broker.on_quote("USD_JPY", 149.99, 150.00, NEXT_QUOTE_TS)
    owner_id = "e2e:stale-quote-epoch"
    with pytest.raises(DojoPortfolioAllocatorError, match="decision_epoch"):
        _allocate(
            broker,
            owner_id=owner_id,
            pair_cap=1,
            global_cap=1,
            candidates=[_candidate("C-stale-quote-epoch", owner_id=owner_id)],
        )


def test_open_position_sl_understatement_fails_before_claim(tmp_path: Path) -> None:
    broker = VirtualBroker(
        tmp_path / "ledger.jsonl",
        balance_jpy=300_000.0,
        leverage=10.0,
        fast_ledger=True,
    )
    broker.on_quote_batch(
        [
            ("USD_JPY", 149.99, 150.00, QUOTE_TS),
            ("EUR_JPY", 159.99, 160.00, QUOTE_TS),
        ]
    )
    owner_id = "e2e:open-sl"
    session = DojoAllocationExecutionSession(
        broker, owner_id, max_concurrent_per_pair=1, global_max_concurrent=2
    )
    first = _allocate(
        broker,
        owner_id=owner_id,
        pair_cap=1,
        global_cap=2,
        candidates=[_candidate("C-open", owner_id=owner_id)],
    )
    selected, config, prepared = _prepare(
        session, first, run_id="run:open-sl-first", nonce="nonce:open-sl-first"
    )
    submitted = session.submit(
        allocation=first,
        selected_intent=selected,
        sealed_config=config,
        prepared_submission=prepared,
        execution_epoch=EPOCH,
    )
    trade_id = str(submitted["virtual_broker_identity"])
    understated = _open_position(broker, trade_id, owner_id=owner_id)
    understated["sl_price"] = float(understated["mark_price"]) - 0.01
    allocation = _allocate(
        broker,
        owner_id=owner_id,
        pair_cap=1,
        global_cap=2,
        positions=[understated],
        candidates=[
            _candidate(
                "C-next",
                owner_id=owner_id,
                pair="EUR_JPY",
                order_kind="LIMIT",
                entry_price=159.90,
            )
        ],
    )
    selected, config, prepared = _prepare(
        session, allocation, run_id="run:open-sl-second", nonce="nonce:open-sl-second"
    )
    with pytest.raises(DojoAllocationExecutionError, match="position evidence"):
        session.submit(
            allocation=allocation,
            selected_intent=selected,
            sealed_config=config,
            prepared_submission=prepared,
            execution_epoch=EPOCH,
        )


def test_open_position_executable_quote_must_match_broker_before_claim(
    tmp_path: Path,
) -> None:
    broker = VirtualBroker(
        tmp_path / "open-executable-quote.jsonl",
        balance_jpy=400_000.0,
        leverage=10.0,
        fast_ledger=True,
    )
    broker.on_quote_batch(
        [
            ("USD_JPY", 149.90, 150.10, QUOTE_TS),
            ("EUR_JPY", 159.99, 160.00, QUOTE_TS),
        ]
    )
    owner_id = "e2e:open-executable-quote"
    session = DojoAllocationExecutionSession(
        broker, owner_id, max_concurrent_per_pair=1, global_max_concurrent=2
    )
    trade_id = session._owner_view.market_order(
        "USD_JPY", "LONG", 1_000.0, sl_pips=25.0
    )
    position = _open_position(broker, trade_id, owner_id=owner_id)
    assert (
        float(position["sl_price"])
        < float(position["bid_price"])
        < float(position["mark_price"])
    )
    position["bid_price"] = 149.96
    position["ask_price"] = 150.04
    allocation = _allocate(
        broker,
        owner_id=owner_id,
        pair_cap=1,
        global_cap=2,
        positions=[position],
        candidates=[
            _candidate(
                "C-open-executable-next",
                owner_id=owner_id,
                pair="EUR_JPY",
                order_kind="LIMIT",
                entry_price=159.90,
            )
        ],
    )
    selected, config, prepared = _prepare(
        session,
        allocation,
        run_id="run:open-executable-quote",
        nonce="nonce:open-executable-quote",
    )
    with pytest.raises(DojoAllocationExecutionError, match="position evidence"):
        session.submit(
            allocation=allocation,
            selected_intent=selected,
            sealed_config=config,
            prepared_submission=prepared,
            execution_epoch=EPOCH,
        )


def test_incumbent_open_phase_cannot_mix_with_candidate_close_phase(
    tmp_path: Path,
) -> None:
    broker = VirtualBroker(
        tmp_path / "mixed-quote-phase.jsonl",
        balance_jpy=400_000.0,
        leverage=10.0,
        fast_ledger=True,
    )
    broker.on_quote("USD_JPY", 149.99, 150.00, f"{QUOTE_TS}#OPEN")
    owner_id = "e2e:mixed-quote-phase"
    session = DojoAllocationExecutionSession(
        broker, owner_id, max_concurrent_per_pair=1, global_max_concurrent=2
    )
    trade_id = session._owner_view.market_order(
        "USD_JPY", "LONG", 1_000.0, sl_pips=10.0
    )
    broker.on_quote_batch(
        [
            ("USD_JPY", 149.99, 150.00, f"{QUOTE_TS}#OPEN"),
            ("EUR_JPY", 159.99, 160.00, f"{QUOTE_TS}#CLOSE"),
        ]
    )
    position = _open_position(broker, trade_id, owner_id=owner_id)

    with pytest.raises(DojoPortfolioAllocatorError, match="phase/watermark"):
        _allocate(
            broker,
            owner_id=owner_id,
            pair_cap=1,
            global_cap=2,
            positions=[position],
            candidates=[
                _candidate(
                    "C-mixed-quote-phase",
                    owner_id=owner_id,
                    pair="EUR_JPY",
                    order_kind="LIMIT",
                    entry_price=159.90,
                )
            ],
        )


def test_pending_sl_pips_understatement_fails_before_claim(tmp_path: Path) -> None:
    broker = VirtualBroker(
        tmp_path / "ledger.jsonl",
        balance_jpy=400_000.0,
        leverage=10.0,
        fast_ledger=True,
    )
    broker.on_quote_batch(
        [
            ("USD_JPY", 149.99, 150.01, QUOTE_TS),
            ("EUR_JPY", 159.99, 160.01, QUOTE_TS),
        ]
    )
    owner_a = "e2e:pending-sl-a"
    owner_b = "e2e:pending-sl-b"
    session_a = DojoAllocationExecutionSession(
        broker, owner_a, max_concurrent_per_pair=1, global_max_concurrent=1
    )
    session_b = DojoAllocationExecutionSession(
        broker, owner_b, max_concurrent_per_pair=1, global_max_concurrent=1
    )
    first = _allocate(
        broker,
        owner_id=owner_a,
        pair_cap=1,
        global_cap=1,
        candidates=[
            _candidate(
                "C-pending",
                owner_id=owner_a,
                order_kind="LIMIT",
                entry_price=149.90,
            )
        ],
    )
    selected, config, prepared = _prepare(
        session_a,
        first,
        run_id="run:pending-sl-first",
        nonce="nonce:pending-sl-first",
    )
    submitted = session_a.submit(
        allocation=first,
        selected_intent=selected,
        sealed_config=config,
        prepared_submission=prepared,
        execution_epoch=EPOCH,
    )
    pending = _pending_order(
        broker, str(submitted["virtual_broker_identity"]), owner_id=owner_a
    )
    pending["sl_pips"] = float(pending["sl_pips"]) / 2
    allocation = _allocate(
        broker,
        owner_id=owner_b,
        pair_cap=1,
        global_cap=1,
        pending=[pending],
        candidates=[
            _candidate(
                "C-other",
                owner_id=owner_b,
                pair="EUR_JPY",
                order_kind="LIMIT",
                entry_price=159.90,
            )
        ],
    )
    selected, config, prepared = _prepare(
        session_b,
        allocation,
        run_id="run:pending-sl-second",
        nonce="nonce:pending-sl-second",
    )
    with pytest.raises(DojoAllocationExecutionError, match="pending evidence"):
        session_b.submit(
            allocation=allocation,
            selected_intent=selected,
            sealed_config=config,
            prepared_submission=prepared,
            execution_epoch=EPOCH,
        )


def test_run_nonce_and_run_intent_are_durable_replay_keys(tmp_path: Path) -> None:
    broker = VirtualBroker(
        tmp_path / "ledger.jsonl",
        balance_jpy=300_000.0,
        leverage=10.0,
        fast_ledger=True,
    )
    broker.on_quote("USD_JPY", 149.99, 150.00, QUOTE_TS)
    owner_id = "e2e:durable-replay"
    session = DojoAllocationExecutionSession(
        broker, owner_id, max_concurrent_per_pair=1, global_max_concurrent=1
    )
    run_id = "run:durable-replay"
    original_nonce = "nonce:durable-replay"
    first = _allocate(
        broker,
        owner_id=owner_id,
        pair_cap=1,
        global_cap=1,
        candidates=[_candidate("C-repeat", owner_id=owner_id)],
    )
    selected, config, prepared = _prepare(
        session, first, run_id=run_id, nonce=original_nonce
    )
    submitted = session.submit(
        allocation=first,
        selected_intent=selected,
        sealed_config=config,
        prepared_submission=prepared,
        execution_epoch=EPOCH,
    )
    broker.close_trade(str(submitted["virtual_broker_identity"]))

    repeated_intent = _allocate(
        broker,
        owner_id=owner_id,
        pair_cap=1,
        global_cap=1,
        candidates=[_candidate("C-repeat", owner_id=owner_id)],
    )
    repeated_selected, repeated_config, repeated_prepared = _prepare(
        session,
        repeated_intent,
        run_id=run_id,
        nonce="nonce:durable-replay-new",
    )
    with pytest.raises(DojoAllocationExecutionError, match="intent was already"):
        session.submit(
            allocation=repeated_intent,
            selected_intent=repeated_selected,
            sealed_config=repeated_config,
            prepared_submission=repeated_prepared,
            execution_epoch=EPOCH,
        )

    repeated_nonce = _allocate(
        broker,
        owner_id=owner_id,
        pair_cap=1,
        global_cap=1,
        candidates=[_candidate("C-new", owner_id=owner_id)],
    )
    nonce_selected, nonce_config, nonce_prepared = _prepare(
        session,
        repeated_nonce,
        run_id=run_id,
        nonce=original_nonce,
    )
    with pytest.raises(DojoAllocationExecutionError, match="nonce was already"):
        session.submit(
            allocation=repeated_nonce,
            selected_intent=nonce_selected,
            sealed_config=nonce_config,
            prepared_submission=nonce_prepared,
            execution_epoch=EPOCH,
        )
    assert not broker.positions


def test_allocation_and_selected_intent_are_one_shot_across_run_ids(
    tmp_path: Path,
) -> None:
    broker = VirtualBroker(
        tmp_path / "cross-run-replay.jsonl",
        balance_jpy=300_000.0,
        leverage=10.0,
        fast_ledger=True,
    )
    broker.on_quote("USD_JPY", 149.99, 150.00, QUOTE_TS)
    owner_id = "e2e:cross-run-replay"
    session = DojoAllocationExecutionSession(
        broker, owner_id, max_concurrent_per_pair=1, global_max_concurrent=1
    )
    allocation = _allocate(
        broker,
        owner_id=owner_id,
        pair_cap=1,
        global_cap=1,
        candidates=[
            _candidate(
                "C-cross-run",
                owner_id=owner_id,
                order_kind="LIMIT",
                entry_price=149.90,
            )
        ],
    )
    selected, config_one, prepared_one = _prepare(
        session,
        allocation,
        run_id="run:cross-run-one",
        nonce="nonce:cross-run-one",
    )
    first = session.submit(
        allocation=allocation,
        selected_intent=selected,
        sealed_config=config_one,
        prepared_submission=prepared_one,
        execution_epoch=EPOCH,
    )
    broker.cancel_order(str(first["virtual_broker_identity"]))

    config_two = session.seal_config(run_id="run:cross-run-two")
    prepared_two = session.prepare(
        allocation=allocation,
        selected_intent=selected,
        sealed_config=config_two,
        execution_nonce="nonce:cross-run-two",
    )
    with pytest.raises(
        DojoAllocationExecutionError, match="allocation and selected intent"
    ):
        session.submit(
            allocation=allocation,
            selected_intent=selected,
            sealed_config=config_two,
            prepared_submission=prepared_two,
            execution_epoch=EPOCH,
        )

    claims = [
        record
        for record in (
            json.loads(line)
            for line in broker.ledger_path.read_text(encoding="utf-8").splitlines()
        )
        if record["event"] == "DOJO_ALLOCATION_EXECUTION_CLAIM"
    ]
    assert len(claims) == 1
    assert not broker.orders


def test_reopened_stateful_broker_requires_verified_snapshot_restore(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "historical-identity.jsonl"
    original = VirtualBroker(
        ledger_path, balance_jpy=300_000.0, leverage=10.0, fast_ledger=True
    )
    original.on_quote("USD_JPY", 149.99, 150.00, QUOTE_TS)
    assert original.market_order("USD_JPY", "LONG", 100.0) == "T000001"
    original._handle.close()

    reopened = VirtualBroker(
        ledger_path, balance_jpy=300_000.0, leverage=10.0, fast_ledger=True
    )
    owner_id = "e2e:historical-identity"
    with pytest.raises(DojoAllocationExecutionError, match="snapshot restore"):
        DojoAllocationExecutionSession(
            reopened, owner_id, max_concurrent_per_pair=1, global_max_concurrent=1
        )


def test_reopened_restored_broker_rehydrates_owner_for_durable_verification(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "rehydrated-owner.jsonl"
    original = VirtualBroker(
        ledger_path, balance_jpy=300_000.0, leverage=10.0, fast_ledger=True
    )
    original.on_quote("USD_JPY", 149.99, 150.00, QUOTE_TS)
    owner_id = "e2e:rehydrated-owner"
    original_session = DojoAllocationExecutionSession(
        original, owner_id, max_concurrent_per_pair=1, global_max_concurrent=1
    )
    allocation = _allocate(
        original,
        owner_id=owner_id,
        pair_cap=1,
        global_cap=1,
        candidates=[_candidate("C-rehydrated-owner", owner_id=owner_id)],
    )
    selected, config, prepared = _prepare(
        original_session,
        allocation,
        run_id="run:rehydrated-owner",
        nonce="nonce:rehydrated-owner",
    )
    submission = original_session.submit(
        allocation=allocation,
        selected_intent=selected,
        sealed_config=config,
        prepared_submission=prepared,
        execution_epoch=EPOCH,
    )
    snapshot = original.snapshot()
    original._handle.close()

    reopened = VirtualBroker(
        ledger_path, balance_jpy=300_000.0, leverage=10.0, fast_ledger=True
    )
    reopened.restore(snapshot)
    reopened_session = DojoAllocationExecutionSession(
        reopened, owner_id, max_concurrent_per_pair=1, global_max_concurrent=1
    )
    verification = reopened_session.verify(submission)

    assert verification["status"] == "FILLED"
    assert verification["owner_verified"] is True
    assert verification["active_owner_global_count"] == 1


def test_owner_rehydration_rejects_ledger_changed_after_broker_open(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "rehydration-tamper.jsonl"
    original = VirtualBroker(ledger_path, fast_ledger=True)
    original._log("NON_STRATEGY_EVIDENCE", {"value": 1})
    original._handle.close()
    reopened = VirtualBroker(ledger_path, fast_ledger=True)
    records = ledger_path.read_text(encoding="utf-8").splitlines()
    terminal = json.loads(records[-1])
    terminal["payload"]["value"] = 2
    records[-1] = json.dumps(terminal, ensure_ascii=False, sort_keys=True)
    ledger_path.write_text("\n".join(records) + "\n", encoding="utf-8")

    with pytest.raises(StrategyOwnershipError, match="ledger is invalid"):
        strategy_ownership_registry(reopened)


def test_stale_parallel_allocations_serialize_before_owner_submission(
    tmp_path: Path,
) -> None:
    broker = VirtualBroker(
        tmp_path / "ledger.jsonl",
        balance_jpy=500_000.0,
        leverage=10.0,
        fast_ledger=True,
    )
    broker.on_quote_batch(
        [
            ("USD_JPY", 149.99, 150.00, QUOTE_TS),
            ("EUR_JPY", 159.99, 160.00, QUOTE_TS),
        ]
    )
    session_a = DojoAllocationExecutionSession(
        broker, "e2e:race-a", max_concurrent_per_pair=1, global_max_concurrent=1
    )
    session_b = DojoAllocationExecutionSession(
        broker, "e2e:race-b", max_concurrent_per_pair=1, global_max_concurrent=1
    )
    allocation_a = _allocate(
        broker,
        owner_id="e2e:race-a",
        pair_cap=1,
        global_cap=1,
        margin_cap=0.5,
        candidates=[_candidate("C-race-a", owner_id="e2e:race-a", units=15_000.0)],
    )
    allocation_b = _allocate(
        broker,
        owner_id="e2e:race-b",
        pair_cap=1,
        global_cap=1,
        margin_cap=0.5,
        candidates=[
            _candidate(
                "C-race-b",
                owner_id="e2e:race-b",
                pair="EUR_JPY",
                units=15_000.0,
                entry_price=160.00,
            )
        ],
    )
    selected_a, config_a, prepared_a = _prepare(
        session_a, allocation_a, run_id="run:race-a", nonce="nonce:race-a"
    )
    selected_b, config_b, prepared_b = _prepare(
        session_b, allocation_b, run_id="run:race-b", nonce="nonce:race-b"
    )
    barrier = threading.Barrier(2)

    def submit_after_barrier(
        session: DojoAllocationExecutionSession,
        allocation: dict[str, object],
        selected: dict[str, object],
        config: dict[str, object],
        prepared: dict[str, object],
    ) -> object:
        barrier.wait()
        try:
            return session.submit(
                allocation=allocation,
                selected_intent=selected,
                sealed_config=config,
                prepared_submission=prepared,
                execution_epoch=EPOCH,
            )
        except DojoAllocationExecutionError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda args: submit_after_barrier(*args),
                [
                    (session_a, allocation_a, selected_a, config_a, prepared_a),
                    (session_b, allocation_b, selected_b, config_b, prepared_b),
                ],
            )
        )
    assert sum(isinstance(result, dict) for result in results) == 1
    assert (
        sum(isinstance(result, DojoAllocationExecutionError) for result in results) == 1
    )
    assert len(broker.positions) == 1
    assert broker.account()["margin_usage"] <= 0.5


def test_verify_rejects_modified_virtual_broker_ledger_chain(tmp_path: Path) -> None:
    broker = VirtualBroker(
        tmp_path / "ledger.jsonl", balance_jpy=200_000.0, fast_ledger=True
    )
    broker.on_quote("USD_JPY", 149.99, 150.00, QUOTE_TS)
    owner_id = "e2e:ledger"
    session = DojoAllocationExecutionSession(
        broker, owner_id, max_concurrent_per_pair=1, global_max_concurrent=1
    )
    allocation = _allocate(
        broker,
        owner_id=owner_id,
        pair_cap=1,
        global_cap=1,
        candidates=[_candidate("C-ledger", owner_id=owner_id)],
    )
    selected, config, prepared = _prepare(
        session, allocation, run_id="run:ledger", nonce="nonce:ledger"
    )
    submission = session.submit(
        allocation=allocation,
        selected_intent=selected,
        sealed_config=config,
        prepared_submission=prepared,
        execution_epoch=EPOCH,
    )
    lines = broker.ledger_path.read_text().splitlines()
    terminal = json.loads(lines[-1])
    terminal["sha"] = "f" * 64
    lines[-1] = json.dumps(terminal, sort_keys=True)
    broker.ledger_path.write_text("\n".join(lines) + "\n")
    with pytest.raises(DojoAllocationExecutionError, match="hash chain"):
        session.verify(submission)


def test_duplicate_json_key_cannot_preserve_a_ledger_hash(tmp_path: Path) -> None:
    ledger_path = tmp_path / "duplicate-key-ledger.jsonl"
    broker = VirtualBroker(
        ledger_path, balance_jpy=200_000.0, leverage=10.0, fast_ledger=True
    )
    broker.on_quote("USD_JPY", 149.99, 150.00, QUOTE_TS)
    owner_id = "e2e:duplicate-key-ledger"
    session = DojoAllocationExecutionSession(
        broker, owner_id, max_concurrent_per_pair=1, global_max_concurrent=1
    )
    allocation = _allocate(
        broker,
        owner_id=owner_id,
        pair_cap=1,
        global_cap=1,
        candidates=[_candidate("C-duplicate-key-ledger", owner_id=owner_id)],
    )
    selected, config, prepared = _prepare(
        session,
        allocation,
        run_id="run:duplicate-key-ledger",
        nonce="nonce:duplicate-key-ledger",
    )
    submission = session.submit(
        allocation=allocation,
        selected_intent=selected,
        sealed_config=config,
        prepared_submission=prepared,
        execution_epoch=EPOCH,
    )
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    lines[-1] = lines[-1].replace('"event":', '"event":"IGNORED_DUPLICATE","event":', 1)
    ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(DojoAllocationExecutionError, match="JSON is invalid"):
        session.verify(submission)
    broker._handle.close()
    with pytest.raises(VirtualBrokerError, match="invalid ledger JSON"):
        VirtualBroker(ledger_path, fast_ledger=True)


def test_post_claim_failure_is_audited_consumed_and_never_releases_incumbent(
    tmp_path: Path,
) -> None:
    broker = VirtualBroker(
        tmp_path / "ledger.jsonl", balance_jpy=200_000.0, fast_ledger=True
    )
    broker.on_quote("USD_JPY", 149.99, 150.00, QUOTE_TS)
    owner_id = "e2e:post-claim-failure"
    session = DojoAllocationExecutionSession(
        broker, owner_id, max_concurrent_per_pair=1, global_max_concurrent=1
    )
    allocation = _allocate(
        broker,
        owner_id=owner_id,
        pair_cap=1,
        global_cap=1,
        candidates=[_candidate("C-fail", owner_id=owner_id)],
    )
    selected, config, prepared = _prepare(
        session, allocation, run_id="run:failure", nonce="nonce:failure"
    )

    def fail_market_order(*args: object, **kwargs: object) -> str:
        del args, kwargs
        raise VirtualBrokerError("injected virtual broker failure")

    broker.market_order = fail_market_order  # type: ignore[method-assign]
    with pytest.raises(DojoAllocationExecutionError, match="claim remains consumed"):
        session.submit(
            allocation=allocation,
            selected_intent=selected,
            sealed_config=config,
            prepared_submission=prepared,
            execution_epoch=EPOCH,
        )
    assert not broker.positions
    records = [json.loads(line) for line in broker.ledger_path.read_text().splitlines()]
    assert (
        sum(record["event"] == "DOJO_ALLOCATION_EXECUTION_CLAIM" for record in records)
        == 1
    )
    failure = next(
        record
        for record in records
        if record["event"] == "DOJO_ALLOCATION_EXECUTION_FAILED"
    )
    assert failure["payload"]["incumbent_release_attempted"] is False
    assert failure["payload"]["claim_consumed"] is True
    with pytest.raises(DojoAllocationExecutionError, match="already claimed"):
        session.submit(
            allocation=allocation,
            selected_intent=selected,
            sealed_config=config,
            prepared_submission=prepared,
            execution_epoch=EPOCH,
        )


def test_post_mutation_log_failure_rolls_back_state_but_consumes_claim(
    tmp_path: Path,
) -> None:
    broker = VirtualBroker(
        tmp_path / "post-mutation-log-failure.jsonl",
        balance_jpy=300_000.0,
        leverage=10.0,
        fast_ledger=True,
    )
    broker.on_quote("USD_JPY", 149.99, 150.00, QUOTE_TS)
    owner_id = "e2e:post-mutation-log-failure"
    session = DojoAllocationExecutionSession(
        broker, owner_id, max_concurrent_per_pair=1, global_max_concurrent=1
    )
    allocation = _allocate(
        broker,
        owner_id=owner_id,
        pair_cap=1,
        global_cap=1,
        candidates=[_candidate("C-post-mutation-log-failure", owner_id=owner_id)],
    )
    selected, config, prepared = _prepare(
        session,
        allocation,
        run_id="run:post-mutation-log-failure",
        nonce="nonce:post-mutation-log-failure",
    )
    owned_log = broker._log

    def fail_fill_log(event: str, payload: dict[str, object]) -> None:
        if event == "FILL_MARKET":
            raise OSError("injected fill log failure")
        owned_log(event, payload)

    broker._log = fail_fill_log  # type: ignore[method-assign]
    with pytest.raises(DojoAllocationExecutionError, match="state was rolled back"):
        session.submit(
            allocation=allocation,
            selected_intent=selected,
            sealed_config=config,
            prepared_submission=prepared,
            execution_epoch=EPOCH,
        )

    assert not broker.positions
    assert broker._seq == 1
    records = [
        json.loads(line)
        for line in broker.ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["event"] for record in records[-2:]] == [
        "DOJO_ALLOCATION_EXECUTION_CLAIM",
        "DOJO_ALLOCATION_EXECUTION_FAILED",
    ]
    assert records[-1]["payload"]["failure_type"] == "OSError"
    assert records[-1]["payload"]["virtual_state_rolled_back"] is True
    with pytest.raises(DojoAllocationExecutionError, match="already claimed"):
        session.submit(
            allocation=allocation,
            selected_intent=selected,
            sealed_config=config,
            prepared_submission=prepared,
            execution_epoch=EPOCH,
        )
    broker._log = owned_log  # type: ignore[method-assign]
    next_allocation = _allocate(
        broker,
        owner_id=owner_id,
        pair_cap=1,
        global_cap=1,
        candidates=[_candidate("C-after-failed-claim", owner_id=owner_id)],
    )
    next_selected, next_config, next_prepared = _prepare(
        session,
        next_allocation,
        run_id="run:after-failed-claim",
        nonce="nonce:after-failed-claim",
    )
    next_submission = session.submit(
        allocation=next_allocation,
        selected_intent=next_selected,
        sealed_config=next_config,
        prepared_submission=next_prepared,
        execution_epoch=EPOCH,
    )
    assert next_submission["virtual_broker_identity"] == "T000002"


def test_two_independent_selected_pending_intents_fill_in_one_quote_batch(
    tmp_path: Path,
) -> None:
    broker = VirtualBroker(
        tmp_path / "ledger.jsonl",
        balance_jpy=500_000.0,
        leverage=10.0,
        fast_ledger=True,
    )
    broker.on_quote_batch(
        [
            ("USD_JPY", 150.00, 150.02, QUOTE_TS),
            ("EUR_JPY", 160.00, 160.02, QUOTE_TS),
        ]
    )
    owner_a = "e2e:owner-a"
    owner_b = "e2e:owner-b"
    session_a = DojoAllocationExecutionSession(
        broker, owner_a, max_concurrent_per_pair=1, global_max_concurrent=1
    )
    session_b = DojoAllocationExecutionSession(
        broker, owner_b, max_concurrent_per_pair=1, global_max_concurrent=1
    )

    allocation_a = _allocate(
        broker,
        owner_id=owner_a,
        pair_cap=1,
        global_cap=1,
        candidates=[
            _candidate(
                "C-a",
                owner_id=owner_a,
                pair="USD_JPY",
                order_kind="LIMIT",
                entry_price=149.90,
            )
        ],
    )
    selected_a, config_a, prepared_a = _prepare(
        session_a, allocation_a, run_id="run:batch-a", nonce="nonce:batch-a"
    )
    submitted_a = session_a.submit(
        allocation=allocation_a,
        selected_intent=selected_a,
        sealed_config=config_a,
        prepared_submission=prepared_a,
        execution_epoch=EPOCH,
    )

    pending_a = _pending_order(
        broker, str(submitted_a["virtual_broker_identity"]), owner_id=owner_a
    )
    allocation_b = _allocate(
        broker,
        owner_id=owner_b,
        pair_cap=1,
        global_cap=1,
        pending=[pending_a],
        candidates=[
            _candidate(
                "C-b",
                owner_id=owner_b,
                pair="EUR_JPY",
                order_kind="LIMIT",
                entry_price=159.90,
            )
        ],
    )
    selected_b, config_b, prepared_b = _prepare(
        session_b, allocation_b, run_id="run:batch-b", nonce="nonce:batch-b"
    )
    submitted_b = session_b.submit(
        allocation=allocation_b,
        selected_intent=selected_b,
        sealed_config=config_b,
        prepared_submission=prepared_b,
        execution_epoch=EPOCH,
    )

    events = broker.on_quote_batch(
        [
            ("USD_JPY", 149.88, 149.90, NEXT_QUOTE_TS),
            ("EUR_JPY", 159.88, 159.90, NEXT_QUOTE_TS),
        ]
    )
    assert sum(event["event"] == "FILL_LIMIT" for event in events) == 2
    verified_a = session_a.verify(submitted_a)
    verified_b = session_b.verify(submitted_b)
    assert verified_a["status"] == "FILLED"
    assert verified_b["status"] == "FILLED"
    assert verified_a["active_owner_global_count"] == 1
    assert verified_b["active_owner_global_count"] == 1
    assert (
        strategy_ownership_registry(broker).historical_trade_owner(
            str(verified_a["trade_id"])
        )
        == owner_a
    )
    assert (
        strategy_ownership_registry(broker).historical_trade_owner(
            str(verified_b["trade_id"])
        )
        == owner_b
    )


@pytest.mark.parametrize(
    ("order_kind", "entry_price", "fill_quote", "expected_entry"),
    [
        ("LIMIT", 149.90, (149.88, 149.89), 149.892),
        ("STOP", 150.10, (150.10, 150.11), 150.112),
    ],
)
def test_pending_fill_price_and_exits_match_sealed_execution_settings(
    tmp_path: Path,
    order_kind: str,
    entry_price: float,
    fill_quote: tuple[float, float],
    expected_entry: float,
) -> None:
    broker = VirtualBroker(
        tmp_path / f"pending-fill-{order_kind}.jsonl",
        balance_jpy=300_000.0,
        leverage=10.0,
        slippage_pips=0.2,
        fast_ledger=True,
    )
    broker.on_quote("USD_JPY", 149.99, 150.00, QUOTE_TS)
    owner_id = f"e2e:pending-fill-{order_kind}"
    session = DojoAllocationExecutionSession(
        broker, owner_id, max_concurrent_per_pair=1, global_max_concurrent=1
    )
    allocation = _allocate(
        broker,
        owner_id=owner_id,
        pair_cap=1,
        global_cap=1,
        candidates=[
            _candidate(
                f"C-pending-fill-{order_kind}",
                owner_id=owner_id,
                order_kind=order_kind,
                entry_price=entry_price,
            )
        ],
    )
    selected, config, prepared = _prepare(
        session,
        allocation,
        run_id=f"run:pending-fill-{order_kind}",
        nonce=f"nonce:pending-fill-{order_kind}",
    )
    submission = session.submit(
        allocation=allocation,
        selected_intent=selected,
        sealed_config=config,
        prepared_submission=prepared,
        execution_epoch=EPOCH,
    )
    broker.on_quote(
        "USD_JPY",
        fill_quote[0],
        fill_quote[1],
        NEXT_QUOTE_TS,
    )

    verification = session.verify(submission)
    assert verification["status"] == "FILLED"
    position = broker.positions[str(verification["trade_id"])]
    assert position.entry_price == expected_entry
    assert position.tp_price == round(expected_entry + 0.10, 3)
    assert position.sl_price == round(expected_entry - 0.10, 3)


def test_pending_fill_rejects_execution_setting_drift(tmp_path: Path) -> None:
    broker = VirtualBroker(
        tmp_path / "pending-fill-setting-drift.jsonl",
        balance_jpy=300_000.0,
        leverage=10.0,
        fast_ledger=True,
    )
    broker.on_quote("USD_JPY", 149.99, 150.00, QUOTE_TS)
    owner_id = "e2e:pending-fill-setting-drift"
    session = DojoAllocationExecutionSession(
        broker, owner_id, max_concurrent_per_pair=1, global_max_concurrent=1
    )
    allocation = _allocate(
        broker,
        owner_id=owner_id,
        pair_cap=1,
        global_cap=1,
        candidates=[
            _candidate(
                "C-pending-fill-setting-drift",
                owner_id=owner_id,
                order_kind="LIMIT",
                entry_price=149.90,
            )
        ],
    )
    selected, config, prepared = _prepare(
        session,
        allocation,
        run_id="run:pending-fill-setting-drift",
        nonce="nonce:pending-fill-setting-drift",
    )
    submission = session.submit(
        allocation=allocation,
        selected_intent=selected,
        sealed_config=config,
        prepared_submission=prepared,
        execution_epoch=EPOCH,
    )
    broker.slippage_pips = 0.1
    with pytest.raises(DojoAllocationExecutionError, match="changed before fill"):
        broker.on_quote("USD_JPY", 149.88, 149.89, NEXT_QUOTE_TS)
    assert str(submission["virtual_broker_identity"]) in broker.orders
    assert not broker.positions


def test_partial_close_keeps_selected_trade_in_owner_occupancy(
    tmp_path: Path,
) -> None:
    broker = VirtualBroker(
        tmp_path / "ledger.jsonl",
        balance_jpy=200_000.0,
        leverage=10.0,
        fast_ledger=True,
    )
    broker.on_quote_batch(
        [
            ("USD_JPY", 149.99, 150.01, QUOTE_TS),
            ("EUR_JPY", 159.99, 160.01, QUOTE_TS),
        ]
    )
    owner_id = "e2e:partial"
    session = DojoAllocationExecutionSession(
        broker,
        owner_id,
        max_concurrent_per_pair=1,
        global_max_concurrent=2,
    )

    first_allocation = _allocate(
        broker,
        owner_id=owner_id,
        pair_cap=1,
        global_cap=2,
        candidates=[
            _candidate(
                "C-first",
                owner_id=owner_id,
                pair="USD_JPY",
                entry_price=150.01,
            )
        ],
    )
    first_selected, first_config, first_prepared = _prepare(
        session,
        first_allocation,
        run_id="run:partial-first",
        nonce="nonce:partial-first",
    )
    first_submission = session.submit(
        allocation=first_allocation,
        selected_intent=first_selected,
        sealed_config=first_config,
        prepared_submission=first_prepared,
        execution_epoch=EPOCH,
    )
    first_trade_id = str(first_submission["virtual_broker_identity"])
    broker.close_trade(first_trade_id, units=250.0)
    partial_verification = session.verify(first_submission)
    assert partial_verification["selected_trade_remaining_units"] == pytest.approx(
        750.0
    )
    assert partial_verification["selected_trade_partially_closed"] is True
    assert partial_verification["partial_close_preserves_occupancy"] is True
    assert partial_verification["active_owner_global_count"] == 1

    second_allocation = _allocate(
        broker,
        owner_id=owner_id,
        pair_cap=1,
        global_cap=2,
        positions=[_open_position(broker, first_trade_id, owner_id=owner_id)],
        margin_cap=0.20,
        candidates=[
            _candidate(
                "C-second",
                owner_id=owner_id,
                pair="EUR_JPY",
                order_kind="LIMIT",
                units=500.0,
                entry_price=159.90,
                expected_net_edge_jpy=200.0,
            )
        ],
    )
    assert second_allocation["decision"]["action"] == "HOLD_FULL"
    second_selected, second_config, second_prepared = _prepare(
        session,
        second_allocation,
        run_id="run:partial-second",
        nonce="nonce:partial-second",
    )
    second_submission = session.submit(
        allocation=second_allocation,
        selected_intent=second_selected,
        sealed_config=second_config,
        prepared_submission=second_prepared,
        execution_epoch=EPOCH,
    )
    assert broker.positions[first_trade_id].units == pytest.approx(750.0)

    events = broker.on_quote_batch(
        [
            ("USD_JPY", 149.99, 150.01, NEXT_QUOTE_TS),
            ("EUR_JPY", 159.88, 159.90, NEXT_QUOTE_TS),
        ]
    )
    assert sum(event["event"] == "FILL_LIMIT" for event in events) == 1
    verification = session.verify(second_submission)
    assert verification["status"] == "FILLED"
    assert verification["active_owner_global_count"] == 2
    assert first_trade_id in verification["active_owner_trade_ids"]


def test_release_plan_fails_before_claim_or_position_mutation(tmp_path: Path) -> None:
    broker = VirtualBroker(
        tmp_path / "ledger.jsonl",
        balance_jpy=200_000.0,
        leverage=10.0,
        fast_ledger=True,
    )
    broker.on_quote_batch(
        [
            ("USD_JPY", 149.99, 150.01, QUOTE_TS),
            ("EUR_JPY", 159.99, 160.01, QUOTE_TS),
        ]
    )
    owner_id = "e2e:no-release"
    session = DojoAllocationExecutionSession(
        broker,
        owner_id,
        max_concurrent_per_pair=1,
        global_max_concurrent=2,
    )
    first = _allocate(
        broker,
        owner_id=owner_id,
        pair_cap=1,
        global_cap=2,
        candidates=[
            _candidate(
                "C-first",
                owner_id=owner_id,
                pair="USD_JPY",
                entry_price=150.01,
            )
        ],
    )
    selected, config, prepared = _prepare(
        session, first, run_id="run:no-release-first", nonce="nonce:no-release-first"
    )
    submitted = session.submit(
        allocation=first,
        selected_intent=selected,
        sealed_config=config,
        prepared_submission=prepared,
        execution_epoch=EPOCH,
    )
    trade_id = str(submitted["virtual_broker_identity"])
    units_before = broker.positions[trade_id].units
    release_allocation = _allocate(
        broker,
        owner_id=owner_id,
        pair_cap=1,
        global_cap=2,
        positions=[_open_position(broker, trade_id, owner_id=owner_id)],
        margin_cap=0.10,
        candidates=[
            _candidate(
                "C-release",
                owner_id=owner_id,
                pair="EUR_JPY",
                order_kind="LIMIT",
                units=500.0,
                entry_price=159.90,
                expected_net_edge_jpy=200.0,
            )
        ],
    )
    assert release_allocation["decision"]["action"] == "HOLD_REDUCE"
    config = session.seal_config(run_id="run:no-release-second")
    with pytest.raises(DojoAllocationExecutionError, match="release is unsupported"):
        session.prepare(
            allocation=release_allocation,
            selected_intent=_selected(release_allocation),
            sealed_config=config,
            execution_nonce="nonce:no-release-second",
        )
    assert broker.positions[trade_id].units == units_before
    assert not any(
        '"event": "DOJO_ALLOCATION_EXECUTION_CLAIM"' in line
        and '"run_id": "run:no-release-second"' in line
        for line in broker.ledger_path.read_text().splitlines()
    )
