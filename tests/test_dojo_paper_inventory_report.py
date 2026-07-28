from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from test_dojo_fresh_model_handoff import _source_packet

from quant_rabbit.dojo_fresh_model_handoff import (
    DojoFreshModelHandoffError,
    compile_snapshot,
    current_ready_packet,
    halt_for_quota,
    initial_story_content,
    initialize_handoff,
    preflight_paper_decision,
    resume_quota_halt,
    seal_model_response,
    submit_model_response,
)
from quant_rabbit.dojo_paired_model_queue import canonical_sha256
from quant_rabbit.dojo_paper_inventory_report import (
    build_paper_inventory_report,
    render_paper_inventory_report,
)


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _room(
    root: Path,
    *,
    now: datetime,
    ordinal: int,
    position: bool,
    tp_gross_measured: bool,
) -> None:
    room_id = f"room-{ordinal}"
    room = root / "experiment-active" / room_id
    room.mkdir(parents=True)
    strategy = f"STRATEGY_{ordinal}"
    contract = {
        "contract": "QR_VIRTUAL_MARKET_SESSION_V2",
        "experiment_id": "experiment-active",
        "room_id": room_id,
        "candidate_id": f"candidate-{ordinal}",
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
            "stale_quote_max_seconds": 90,
        },
        "pairs": ["USD_JPY"],
        "costs": {
            "explicit": True,
            "financing_pips_per_day": 0 if tp_gross_measured else 0.8,
            "slippage_pips_per_fill": 0 if tp_gross_measured else 0.3,
        },
        "bot": {
            "strategy_tags": [strategy],
            "config": {
                "pairs": ["USD_JPY"],
                "ceiling_min": 60,
            },
        },
    }
    event_body = {
        "event": "EXIT_TP",
        "payload": {
            "pl_jpy": 25.0,
            **({"gross_pl_jpy": 25.0} if tp_gross_measured else {}),
        },
        "prev_sha": "0" * 64,
        "ts_utc": (now - timedelta(minutes=10)).isoformat(),
    }
    event = {**event_body, "sha": canonical_sha256(event_body)}
    positions = (
        [
            {
                "trade_id": f"T{ordinal}",
                "pair": "USD_JPY",
                "side": "LONG",
                "units": 1000.0,
                "entry_price": 163.40,
                "opened_ts": (now - timedelta(minutes=20)).isoformat(),
                "strategy_tag": strategy,
                "tp_price": 163.60,
                "sl_price": 163.20,
            }
        ]
        if position
        else []
    )
    orders = [] if position else [
        {
            "order_id": f"O{ordinal}",
            "pair": "USD_JPY",
            "side": "SHORT",
            "kind": "LIMIT",
            "limit_price": 163.70,
            "units": 1000.0,
            "strategy_tag": strategy,
        }
    ]
    snapshot = {
        "seq": ordinal,
        "ledger_sha": event["sha"],
        "balance_jpy": 200_000.0 + ordinal,
        "positions": positions,
        "orders": orders,
    }
    state = {
        "wall_time_utc": (now - timedelta(seconds=2)).isoformat(),
        "account": {
            "balance_jpy": 200_000.0 + ordinal,
            "equity_jpy": 200_050.0 + ordinal,
            "margin_usage": 0.1 if position else 0.0,
            "margin_used_jpy": 1000.0 if position else 0.0,
        },
        "quotes": {
            "USD_JPY": {
                "bid": 163.45,
                "ask": 163.46,
                "ts": (now - timedelta(seconds=1)).isoformat(),
            }
        },
    }
    _write(room / "session_contract.json", contract)
    _write(room / "broker_snapshot.json", snapshot)
    _write(room / "state.json", state)
    (room / "ledger.jsonl").write_text(
        json.dumps(event, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rooms(root: Path, now: datetime) -> None:
    for ordinal in range(1, 5):
        _room(
            root,
            now=now,
            ordinal=ordinal,
            position=ordinal == 1,
            tp_gross_measured=ordinal != 2,
        )


def test_report_renders_fixed_card_and_leaves_unproved_economics_unmeasured(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 7, 27, 4, 0, tzinfo=timezone.utc)
    runtime = tmp_path / "runtime"
    rooms = tmp_path / "rooms"
    initialize_handoff(runtime)
    _rooms(rooms, now)

    report = build_paper_inventory_report(
        runtime_root=runtime,
        rooms_root=rooms,
        now_utc=now,
    )
    rendered = render_paper_inventory_report(report)

    assert report["room_count"] == 4
    assert report["authority"] == "NONE"
    assert report["runtime_health_status"] == "HEALTHY"
    assert report["profitability_status"] == "PROFITABLE"
    assert report["inventory_observation"]["open_position_count"] == 1
    assert report["inventory_observation"]["resting_order_count"] == 3
    assert report["inventory_observation"]["counter_trend_resting_order_count"] is None
    assert report["rooms"][0]["long_units"] == 1000.0
    assert report["rooms"][0]["unrealized_pl_jpy"] == pytest.approx(50.0)
    assert report["totals"]["tp_gross_jpy"] is None
    assert report["totals"]["economic_result_status"] == "UNDETERMINED"
    assert rendered.startswith("PAPER AI inventory supervisor\n")
    assert "稼働評価: HEALTHY | 収益評価: PROFITABLE" in rendered
    assert "| room / strategy |" in rendered
    assert "未計測" in rendered
    assert "shadow only, authority NONE" in rendered


def test_quota_halt_is_atomic_repeat_noop_and_explicit_resume(
    tmp_path: Path,
) -> None:
    initialize_handoff(tmp_path)
    compile_snapshot(root=tmp_path, source_packet=_source_packet())
    packet_sha = current_ready_packet(tmp_path)["decision_packet_sha256"]

    first = halt_for_quota(
        tmp_path,
        reason="weekly usage exhausted",
        observed_at_utc="2026-07-27T04:00:00+00:00",
    )
    repeated = halt_for_quota(
        tmp_path,
        reason="different text must not replace first sentinel",
        observed_at_utc="2026-07-27T04:01:00+00:00",
    )

    assert repeated == first
    assert first["current_ready_packet_sha256"] == packet_sha
    assert first["accepted_state_mutated"] is False
    with pytest.raises(DojoFreshModelHandoffError, match="withheld"):
        current_ready_packet(tmp_path)
    resumed = resume_quota_halt(tmp_path)
    assert resumed["current_ready_packet_sha256"] == packet_sha


def test_halted_preflight_reads_no_rooms_and_returns_dont_notify(tmp_path: Path) -> None:
    initialize_handoff(tmp_path)
    halt_for_quota(
        tmp_path,
        reason="model task cannot start",
        observed_at_utc="2026-07-27T04:00:00+00:00",
    )

    result = preflight_paper_decision(
        root=tmp_path,
        rooms_root=tmp_path / "does-not-exist",
        now_utc=datetime(2026, 7, 27, 4, 0, tzinfo=timezone.utc),
    )

    assert result["state"] == "HALTED_QUOTA"
    assert result["zero_work"] is True
    assert result["notification"] == "DONT_NOTIFY"
    assert result["decision_packet"] is None
    assert result["inventory_report"] is None


def test_sealed_unsubmitted_response_survives_quota_crash_boundary(
    tmp_path: Path,
) -> None:
    initialize_handoff(tmp_path)
    compile_snapshot(root=tmp_path, source_packet=_source_packet())
    packet = current_ready_packet(tmp_path)
    story = initial_story_content()
    story["last_action"] = "HOLD"
    response = seal_model_response(
        packet=packet,
        action="HOLD",
        reason_ids=["QUOTA_CRASH_BOUNDARY_TEST"],
        next_story_content=story,
        provider_model="codex-test",
        provider_execution_id="fresh-task-crash-boundary",
    )
    halt_for_quota(
        tmp_path,
        reason="429 after response seal",
        observed_at_utc="2026-07-27T04:00:00+00:00",
    )

    with pytest.raises(DojoFreshModelHandoffError, match="unaccepted response"):
        submit_model_response(root=tmp_path, response_value=response)
    resumed = resume_quota_halt(tmp_path)
    assert resumed["accepted_fresh_model_decision_count"] == 0
    accepted = submit_model_response(root=tmp_path, response_value=response)
    assert accepted["accepted_fresh_model_decision_count"] == 1
