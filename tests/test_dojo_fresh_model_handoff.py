from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quant_rabbit.dojo_fresh_model_handoff import (
    DojoFreshModelHandoffError,
    build_paper_source_packet_from_rooms,
    compile_snapshot,
    current_ready_packet,
    initial_story_content,
    initialize_handoff,
    seal_model_response,
    submit_model_response,
    validate_story_content,
    verify_handoff,
)
from quant_rabbit.dojo_paired_model_queue import canonical_sha256


def _source_packet(
    *,
    market_open: bool = True,
    position: bool = True,
    order: bool = False,
    ledger_tip: str = "a" * 64,
) -> dict:
    positions = (
        [
            {
                "position_id": "room-1:T0001",
                "room_id": "room-1",
                "trade_id": "T0001",
                "pair": "USD_JPY",
                "side": "LONG",
                "units": 2_400,
                "entry_price": 163.50,
                "current_price": 163.45,
                "unrealized_pips": -5.0,
                "opened_ts": "2026-07-27T01:00:00+00:00",
                "strategy_tag": "TEST",
                "tp_price": 163.60,
                "sl_price": 163.25,
            }
        ]
        if position
        else []
    )
    orders = (
        [
            {
                "room_id": "room-1",
                "order_id": "O0001",
                "pair": "USD_JPY",
                "side": "LONG",
                "kind": "LIMIT",
                "limit_price": 163.40,
                "units": 2_400,
                "strategy_tag": "TEST",
            }
        ]
        if order
        else []
    )
    body = {
        "contract": "QR_DOJO_PAPER_AI_SHADOW_HOURLY_V1",
        "generated_at_utc": "2026-07-27T01:32:00+00:00",
        "purpose": "SHADOW_ONLY_INVENTORY_AND_MARKET_STORY_REVIEW",
        "market_status": {
            "is_fx_open": market_open,
            "active_sessions": ["Tokyo"] if market_open else [],
            "closed_reason": None if market_open else "WEEKEND",
            "most_recent_open_utc": "2026-07-26T21:00:00+00:00",
        },
        "market_features": {
            "USD_JPY": {
                "pair": "USD_JPY",
                "last_complete_m1_utc": "2026-07-27T01:31:00+00:00",
                "last_mid": 163.455,
                "return_1h_pips": -8.2,
                "return_4h_pips": 12.4,
                "return_24h_pips": 20.0,
                "mean_m1_range_1h_pips": 1.2,
                "complete_m1_count": 1_500,
                "price_component": "BA",
                "source": "OANDA_READ_ONLY",
            }
        },
        "rooms": [
            {
                "experiment_id": "paper-test",
                "room_id": "room-1",
                "candidate_id": "candidate-1",
                "strategy_tags": ["TEST"],
                "configured_pairs": ["USD_JPY"],
                "balance_jpy": 199_900.0,
                "positions": positions,
                "orders": orders,
                "state_wall_time_utc": "2026-07-27T01:31:58+00:00",
                "state_age_seconds": 2.0,
                "ledger_tip_sha256": ledger_tip,
                "paper_only": True,
                "order_authority": "NONE",
                "live_permission": False,
            }
        ],
        "safety": {
            "paper_only": True,
            "shadow_only": True,
            "order_authority": "NONE",
            "live_permission": False,
            "broker_mutation_allowed": False,
            "recommendations_are_not_commands": True,
        },
    }
    return {**body, "packet_sha256": canonical_sha256(body)}


def _next_story(action: str = "HOLD") -> dict:
    story = initial_story_content()
    story.update(
        {
            "current_thesis": "USDJPY pullback remains bounded but unproven",
            "macro_regime": "USD_BID",
            "micro_regime": "PULLBACK",
            "evidence_for": ["4h return remains positive"],
            "evidence_against": ["1h return is negative"],
            "inventory_risk": ["OPEN_LONG_UNREALIZED_LOSS"],
            "last_action": action,
            "expected_outcome": "risk remains bounded until next causal review",
            "invalidation_conditions": ["hard risk guard breach"],
            "next_review": "HIGH_RISK_15M",
            "confidence": 0.55,
            "known_unknowns": ["margin utilization absent from source packet"],
        }
    )
    return story


def _write_active_room(
    rooms_root: Path,
    *,
    now: datetime,
    margin_usage: float = 0.1,
    state_age_seconds: int = 2,
) -> None:
    room = rooms_root / "experiment-1" / "room-1"
    room.mkdir(parents=True)
    contract = {
        "contract": "QR_VIRTUAL_MARKET_SESSION_V2",
        "experiment_id": "experiment-1",
        "room_id": "room-1",
        "candidate_id": "candidate-1",
        "proof_mode": "diagnostic",
        "proof_eligible": False,
        "authority": {
            "broker_mutation_allowed": False,
            "live_permission": False,
            "order_authority": "NONE",
        },
        "source": {
            "window_start_utc": (now - timedelta(hours=1)).isoformat(),
            "window_end_utc": (now + timedelta(hours=1)).isoformat(),
        },
        "bot": {
            "strategy_tags": ["TEST"],
            "config": {"pairs": ["USD_JPY"], "ceiling_min": 60},
        },
    }
    snapshot = {
        "seq": 2,
        "ledger_sha": "c" * 64,
        "balance_jpy": 199_900.0,
        "positions": [
            {
                "trade_id": "T0001",
                "pair": "USD_JPY",
                "side": "LONG",
                "units": 2_400,
                "entry_price": 163.50,
                "opened_ts": (now - timedelta(minutes=30)).isoformat(),
                "strategy_tag": "TEST",
                "tp_price": 163.60,
                "sl_price": 163.25,
            }
        ],
        "orders": [],
    }
    state = {
        "wall_time_utc": (now - timedelta(seconds=state_age_seconds)).isoformat(),
        "account": {
            "balance_jpy": 199_900.0,
            "equity_jpy": 199_880.0,
            "margin_usage": margin_usage,
        },
        "quotes": {
            "USD_JPY": {
                "bid": 163.45,
                "ask": 163.46,
                "ts": (now - timedelta(seconds=1)).isoformat(),
            }
        },
    }
    for name, value in (
        ("session_contract.json", contract),
        ("broker_snapshot.json", snapshot),
        ("state.json", state),
    ):
        (room / name).write_text(json.dumps(value), encoding="utf-8")
    ledger_event = {
        "event": "FILL_LIMIT",
        "ts_utc": (now - timedelta(minutes=30)).isoformat(),
        "sha": "d" * 64,
        "payload": {
            "trade_id": "T0001",
            "order_id": "O0001",
            "side": "LONG",
            "pl_jpy": None,
        },
    }
    (room / "ledger.jsonl").write_text(
        json.dumps(ledger_event, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _accept_first(root: Path) -> tuple[dict, dict]:
    packet = current_ready_packet(root)
    response = seal_model_response(
        packet=packet,
        action="HOLD",
        reason_ids=["POSITION_OPEN_REVIEW_CAUSAL"],
        next_story_content=_next_story(),
        provider_model="codex-test",
        provider_execution_id="fresh-task-test-1",
    )
    status = submit_model_response(root=root, response_value=response)
    return response, status


def test_compile_publishes_bounded_packet_without_conversation_or_wall_clock(
    tmp_path: Path,
) -> None:
    initialize_handoff(tmp_path)
    status = compile_snapshot(root=tmp_path, source_packet=_source_packet())

    assert status["state"] == "WAITING_FOR_FRESH_TASK"
    packet = current_ready_packet(tmp_path)
    serialized = json.dumps(packet, sort_keys=True)
    assert "generated_at_utc" not in serialized
    assert "state_wall_time_utc" not in serialized
    assert "state_age_seconds" not in serialized
    assert "opened_ts" not in serialized
    assert packet["conversation_history_allowed"] is False
    assert packet["fresh_codex_task_required"] is True
    assert packet["snapshot"]["future_information_visible"] is False
    assert packet["authority"]["order_authority"] == "NONE"
    assert "MARKET" not in packet["action_allowlist"]
    assert "LIMIT" not in packet["action_allowlist"]


def test_accept_updates_story_and_same_state_skips_without_tokens(
    tmp_path: Path,
) -> None:
    initialize_handoff(tmp_path)
    source = _source_packet()
    compile_snapshot(root=tmp_path, source_packet=source)
    response, accepted = _accept_first(tmp_path)

    assert accepted["state"] == "IDLE_NO_READY_PACKET"
    assert accepted["accepted_fresh_model_decision_count"] == 1
    assert accepted["current_story_sequence"] == 1
    duplicate = submit_model_response(root=tmp_path, response_value=response)
    assert duplicate == accepted

    skipped = compile_snapshot(root=tmp_path, source_packet=source)
    assert skipped["state"] == "STATE_HASH_UNCHANGED_NO_MODEL_CALL"
    assert skipped["model_tokens_used"] == 0
    assert skipped["fresh_task_created"] is False
    assert verify_handoff(tmp_path) == accepted


def test_changed_state_publishes_next_packet_with_only_previous_summary(
    tmp_path: Path,
) -> None:
    initialize_handoff(tmp_path)
    compile_snapshot(root=tmp_path, source_packet=_source_packet())
    _accept_first(tmp_path)

    changed = _source_packet(ledger_tip="b" * 64)
    status = compile_snapshot(root=tmp_path, source_packet=changed)
    packet = current_ready_packet(tmp_path)

    assert status["state"] == "WAITING_FOR_FRESH_TASK"
    assert packet["rolling_story"]["story_sequence"] == 1
    assert packet["previous_decision"]["action"] == "HOLD"
    assert set(packet["previous_decision"]) == {
        "decision_packet_sha256",
        "action",
        "reason_ids",
        "next_story_sha256",
    }


def test_flat_and_closed_idle_skip_but_closed_inventory_is_reviewed(
    tmp_path: Path,
) -> None:
    initialize_handoff(tmp_path)
    flat = compile_snapshot(
        root=tmp_path,
        source_packet=_source_packet(position=False, order=False),
    )
    assert flat["state"] == "FLAT_IDLE_NO_MODEL_CALL"
    assert flat["model_tokens_used"] == 0

    closed_root = tmp_path / "closed"
    initialize_handoff(closed_root)
    closed_flat = compile_snapshot(
        root=closed_root,
        source_packet=_source_packet(
            market_open=False,
            position=False,
            order=False,
        ),
    )
    assert closed_flat["state"] == "MARKET_CLOSED_FLAT_NO_MODEL_CALL"

    exposed_root = tmp_path / "closed-exposed"
    initialize_handoff(exposed_root)
    exposed = compile_snapshot(
        root=exposed_root,
        source_packet=_source_packet(market_open=False, position=True),
    )
    assert exposed["state"] == "WAITING_FOR_FRESH_TASK"


def test_high_risk_and_major_event_select_declared_cadence(tmp_path: Path) -> None:
    high_root = tmp_path / "high"
    initialize_handoff(high_root)
    compile_snapshot(
        root=high_root,
        source_packet=_source_packet(),
        risk_signals=["MARGIN_UTILIZATION_HIGH"],
    )
    assert current_ready_packet(high_root)["cadence_mode"] == "HIGH_RISK_15M"

    event_root = tmp_path / "event"
    initialize_handoff(event_root)
    compile_snapshot(
        root=event_root,
        source_packet=_source_packet(position=False, order=False),
        major_event_ids=["MARKET_DISLOCATION"],
    )
    assert (
        current_ready_packet(event_root)["cadence_mode"]
        == "MAJOR_EVENT_IMMEDIATE_ON_COMPILER_INVOCATION"
    )


def test_local_room_compiler_uses_no_provider_and_detects_high_risk(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 7, 27, 2, 15, tzinfo=timezone.utc)
    rooms_root = tmp_path / "rooms"
    _write_active_room(rooms_root, now=now, margin_usage=0.6)

    source, signals = build_paper_source_packet_from_rooms(
        rooms_root=rooms_root,
        now_utc=now,
    )

    assert source["local_compiler"] == {
        "network_access_used": False,
        "model_credentials_used": False,
        "broker_client_used": False,
        "source": "LOCAL_PAPER_ROOM_STATE_AND_SNAPSHOT",
    }
    assert signals == ["MARGIN_UTILIZATION_HIGH"]
    assert source["safety"]["order_authority"] == "NONE"
    initialize_handoff(tmp_path / "handoff")
    compile_snapshot(
        root=tmp_path / "handoff",
        source_packet=source,
        risk_signals=signals,
    )
    packet = current_ready_packet(tmp_path / "handoff")
    assert packet["cadence_mode"] == "HIGH_RISK_15M"
    assert packet["snapshot"]["market_features"]["USD_JPY"]["source"] == (
        "LOCAL_PAPER_ROOM_STATE"
    )
    assert packet["recent_events"] == [
        {
            "available_through_utc": (now - timedelta(minutes=30)).isoformat(),
            "event_id": f"room-1:{'d' * 64}",
            "event_type": "FILL_LIMIT",
            "summary": (
                "room=room-1;trade=T0001;order=O0001;" "side=LONG;realized_pl_jpy=None"
            ),
        }
    ]


def test_local_room_compiler_rejects_stale_active_room(tmp_path: Path) -> None:
    now = datetime(2026, 7, 27, 2, 15, tzinfo=timezone.utc)
    rooms_root = tmp_path / "rooms"
    _write_active_room(rooms_root, now=now, state_age_seconds=901)

    with pytest.raises(DojoFreshModelHandoffError, match="stale"):
        build_paper_source_packet_from_rooms(
            rooms_root=rooms_root,
            now_utc=now,
        )


def test_local_room_compiler_rejects_future_ledger_event(tmp_path: Path) -> None:
    now = datetime(2026, 7, 27, 2, 15, tzinfo=timezone.utc)
    rooms_root = tmp_path / "rooms"
    _write_active_room(rooms_root, now=now)
    ledger_path = rooms_root / "experiment-1" / "room-1" / "ledger.jsonl"
    event = json.loads(ledger_path.read_text(encoding="utf-8"))
    event["ts_utc"] = (now + timedelta(seconds=1)).isoformat()
    ledger_path.write_text(json.dumps(event) + "\n", encoding="utf-8")

    with pytest.raises(DojoFreshModelHandoffError, match="future event"):
        build_paper_source_packet_from_rooms(
            rooms_root=rooms_root,
            now_utc=now,
        )


def test_story_bounds_and_response_causal_flags_fail_closed(tmp_path: Path) -> None:
    oversized = initial_story_content()
    oversized["known_unknowns"] = [f"unknown-{index}" for index in range(9)]
    with pytest.raises(DojoFreshModelHandoffError, match="bounded text list"):
        validate_story_content(oversized)

    initialize_handoff(tmp_path)
    compile_snapshot(root=tmp_path, source_packet=_source_packet())
    packet = current_ready_packet(tmp_path)
    response = seal_model_response(
        packet=packet,
        action="HOLD",
        reason_ids=["CAUSAL_REVIEW"],
        next_story_content=_next_story(),
        provider_model="codex-test",
        provider_execution_id="fresh-task-test-1",
    )
    tampered = copy.deepcopy(response)
    tampered["future_information_used"] = True
    tampered["response_sha256"] = canonical_sha256(
        {key: item for key, item in tampered.items() if key != "response_sha256"}
    )
    with pytest.raises(DojoFreshModelHandoffError, match="response seal"):
        submit_model_response(root=tmp_path, response_value=tampered)


def test_conflicting_duplicate_and_live_action_fail_closed(tmp_path: Path) -> None:
    initialize_handoff(tmp_path)
    compile_snapshot(root=tmp_path, source_packet=_source_packet())
    response, _ = _accept_first(tmp_path)
    conflicting = copy.deepcopy(response)
    conflicting["provider_execution_id"] = "different-task"
    conflicting["response_sha256"] = canonical_sha256(
        {key: item for key, item in conflicting.items() if key != "response_sha256"}
    )
    with pytest.raises(DojoFreshModelHandoffError, match="duplicate response"):
        submit_model_response(root=tmp_path, response_value=conflicting)

    next_root = tmp_path / "live-action"
    initialize_handoff(next_root)
    compile_snapshot(root=next_root, source_packet=_source_packet())
    with pytest.raises(DojoFreshModelHandoffError, match="allowlist"):
        seal_model_response(
            packet=current_ready_packet(next_root),
            action="MARKET",
            reason_ids=["TRY_TO_TRADE"],
            next_story_content=_next_story("HOLD"),
            provider_model="codex-test",
            provider_execution_id="fresh-task-test-2",
        )
