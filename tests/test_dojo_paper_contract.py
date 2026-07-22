from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from quant_rabbit.dojo_paper_contract import (
    DojoPaperContractError,
    build_session_contract,
    canonical_sha256,
    prepare_drain_contract,
    publish_immutable_json,
)
from quant_rabbit.virtual_broker import VirtualBroker


UTC = timezone.utc


def _runtime() -> dict:
    return {"files": [], "manifest_sha256": "1" * 64}


def _bot(*tags: str) -> dict:
    payload = {
        "kind": "module",
        "config": [{"strategy_tag": tag} for tag in tags],
        "config_sha256": "2" * 64,
        "strategy_tags": list(tags),
    }
    return {**payload, "bot_contract_sha256": canonical_sha256(payload)}


def _formal_contract(*, costs_explicit: bool, tags=("W_FADE",)) -> dict:
    return build_session_contract(
        experiment_id="episode-s5",
        room_id="room-w-fade-stress",
        candidate_id="W_FADE@v1",
        room_kind="single_strategy",
        proof_mode="formal",
        feed="live",
        pairs=["USD_JPY"],
        initial_balance_jpy=200_000.0,
        slippage_pips=0.3,
        financing_pips_per_day=0.8,
        leverage=25.0,
        costs_explicit=costs_explicit,
        runtime=_runtime(),
        bot=_bot(*tags),
        source={
            "kind": "live_read_only_pricing",
            "window_start_utc": "2026-07-27T00:00:00+00:00",
            "window_end_utc": "2026-08-03T00:00:00+00:00",
        },
    )


def test_formal_room_rejects_implicit_costs_and_mixed_strategy_tags():
    with pytest.raises(DojoPaperContractError, match="explicit slippage"):
        _formal_contract(costs_explicit=False)

    with pytest.raises(DojoPaperContractError, match="exactly one"):
        _formal_contract(costs_explicit=True, tags=("W_FADE", "W_SPIKE"))

    contract = _formal_contract(costs_explicit=True)
    with pytest.raises(DojoPaperContractError, match="finite"):
        build_session_contract(
            experiment_id=contract["experiment_id"],
            room_id=contract["room_id"],
            candidate_id=contract["candidate_id"],
            room_kind=contract["room_kind"],
            proof_mode=contract["proof_mode"],
            feed=contract["feed"],
            pairs=contract["pairs"],
            initial_balance_jpy=contract["initial_balance_jpy"],
            slippage_pips=float("nan"),
            financing_pips_per_day=0.8,
            leverage=25.0,
            costs_explicit=True,
            runtime=contract["runtime"],
            bot=contract["bot"],
            source=contract["source"],
        )


def test_formal_evidence_requires_a_fixed_live_window_or_replay_manifest():
    kwargs = {
        "experiment_id": "episode-s5",
        "room_id": "room-w-fade-stress",
        "candidate_id": "W_FADE@v1",
        "room_kind": "single_strategy",
        "proof_mode": "formal",
        "pairs": ["USD_JPY"],
        "initial_balance_jpy": 200_000.0,
        "slippage_pips": 0.3,
        "financing_pips_per_day": 0.8,
        "leverage": 25.0,
        "costs_explicit": True,
        "runtime": _runtime(),
        "bot": _bot("W_FADE"),
    }
    with pytest.raises(DojoPaperContractError, match="ISO start/end"):
        build_session_contract(
            **kwargs,
            feed="live",
            source={"kind": "live_read_only_pricing"},
        )
    with pytest.raises(DojoPaperContractError, match="source manifest"):
        build_session_contract(
            **kwargs,
            feed="replay",
            source={"kind": "sealed_replay"},
        )


def test_session_contract_is_publish_once_and_content_addressed(tmp_path):
    contract = _formal_contract(costs_explicit=True)
    path = tmp_path / "session_contract.json"

    assert publish_immutable_json(path, contract) is True
    assert publish_immutable_json(path, contract) is False
    assert contract["session_contract_sha256"] == canonical_sha256(
        {
            key: value
            for key, value in contract.items()
            if key != "session_contract_sha256"
        }
    )

    changed = {**contract, "candidate_id": "silently-replaced"}
    with pytest.raises(DojoPaperContractError, match="immutable contract"):
        publish_immutable_json(path, changed)


def _stopped_session(tmp_path, *, strategy_tag: str | None):
    session = tmp_path / "session"
    (session / "inbox" / "processed").mkdir(parents=True)
    broker = VirtualBroker(session / "ledger.jsonl", balance_jpy=200_000.0)
    opened = datetime(2026, 7, 22, 0, 0, tzinfo=UTC)
    broker.on_quote("USD_JPY", 162.00, 162.01, opened.isoformat())
    trade_id = broker.market_order(
        "USD_JPY",
        "LONG",
        1_000,
        tp_pips=200,
        strategy_tag=strategy_tag,
    )
    broker._log("SESSION_STOP", {"reason": "fixed boundary"})
    (session / "broker_snapshot.json").write_text(
        json.dumps(broker.snapshot()), encoding="utf-8"
    )
    return session, broker, trade_id, opened


def test_first_drain_requires_terminal_stop_and_explicit_legacy_mode(tmp_path):
    session = tmp_path / "session"
    (session / "inbox" / "processed").mkdir(parents=True)
    broker = VirtualBroker(session / "ledger.jsonl", balance_jpy=200_000.0)
    broker.on_quote("USD_JPY", 162.00, 162.01, datetime.now(UTC).isoformat())
    broker.market_order("USD_JPY", "LONG", 1_000)
    (session / "broker_snapshot.json").write_text(json.dumps(broker.snapshot()))

    with pytest.raises(DojoPaperContractError, match="terminal SESSION_STOP"):
        prepare_drain_contract(
            session_dir=session,
            pairs=["USD_JPY"],
            ceiling_minutes=480,
            allow_legacy_untagged=True,
            slippage_pips=0.0,
            financing_pips_per_day=0.0,
            leverage=25.0,
            runtime=_runtime(),
        )

    broker._log("SESSION_STOP", {"reason": "fixed boundary"})
    (session / "broker_snapshot.json").write_text(json.dumps(broker.snapshot()))
    with pytest.raises(DojoPaperContractError, match="untagged positions"):
        prepare_drain_contract(
            session_dir=session,
            pairs=["USD_JPY"],
            ceiling_minutes=480,
            allow_legacy_untagged=False,
            slippage_pips=0.0,
            financing_pips_per_day=0.0,
            leverage=25.0,
            runtime=_runtime(),
        )


def test_drain_resume_keeps_the_original_inventory_contract(tmp_path):
    session, broker, trade_id, opened = _stopped_session(
        tmp_path, strategy_tag=None
    )
    first = prepare_drain_contract(
        session_dir=session,
        pairs=["USD_JPY"],
        ceiling_minutes=480,
        allow_legacy_untagged=True,
        slippage_pips=0.0,
        financing_pips_per_day=0.0,
        leverage=25.0,
        runtime=_runtime(),
    )

    broker._log("DRAIN_START", {"contract": first["drain_contract_sha256"]})
    broker.on_quote(
        "USD_JPY",
        162.02,
        162.03,
        (opened + timedelta(minutes=10)).isoformat(),
    )
    broker.close_trade(trade_id)
    broker._log("DRAIN_STOP", {"status": "SEALED"})
    (session / "broker_snapshot.json").write_text(json.dumps(broker.snapshot()))

    resumed = prepare_drain_contract(
        session_dir=session,
        pairs=["USD_JPY"],
        ceiling_minutes=480,
        allow_legacy_untagged=True,
        slippage_pips=0.0,
        financing_pips_per_day=0.0,
        leverage=25.0,
        runtime=_runtime(),
    )

    assert resumed == first
    assert resumed["legacy_untagged_trade_ids"] == [trade_id]
    assert resumed["source_snapshot_sha256"] != canonical_sha256(
        json.loads(
            (session / "broker_snapshot.json").read_text(encoding="utf-8")
        )
    )


def test_drain_refuses_pending_agent_actions(tmp_path):
    session, _, _, _ = _stopped_session(tmp_path, strategy_tag="W_FADE")
    (session / "inbox" / "late-order.json").write_text(
        json.dumps({"action": "MARKET"}), encoding="utf-8"
    )

    with pytest.raises(DojoPaperContractError, match="pending agent actions"):
        prepare_drain_contract(
            session_dir=session,
            pairs=["USD_JPY"],
            ceiling_minutes=480,
            allow_legacy_untagged=False,
            slippage_pips=0.0,
            financing_pips_per_day=0.0,
            leverage=25.0,
            runtime=_runtime(),
        )


def test_drain_refuses_a_rewritten_existing_contract(tmp_path):
    session, _, _, _ = _stopped_session(tmp_path, strategy_tag="W_FADE")
    prepare_drain_contract(
        session_dir=session,
        pairs=["USD_JPY"],
        ceiling_minutes=480,
        allow_legacy_untagged=False,
        slippage_pips=0.0,
        financing_pips_per_day=0.0,
        leverage=25.0,
        runtime=_runtime(),
    )
    path = session / "drain_contract.json"
    rewritten = json.loads(path.read_text(encoding="utf-8"))
    rewritten["policy"]["new_entries_allowed"] = True
    path.write_text(json.dumps(rewritten), encoding="utf-8")

    with pytest.raises(DojoPaperContractError, match="content hash mismatch"):
        prepare_drain_contract(
            session_dir=session,
            pairs=["USD_JPY"],
            ceiling_minutes=480,
            allow_legacy_untagged=False,
            slippage_pips=0.0,
            financing_pips_per_day=0.0,
            leverage=25.0,
            runtime=_runtime(),
        )
