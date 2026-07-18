from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quant_rabbit.dojo_worker_forward import (
    DojoWorkerForwardError,
    audit_lifecycle,
    build_day_seal,
    build_final_receipt,
    build_precommit,
    build_start_receipt,
    canonical_sha256,
    validate_precommit,
    write_new_json,
)


UTC = timezone.utc
WINDOW_START = datetime(2030, 7, 20, tzinfo=UTC)
WINDOW_END = WINDOW_START + timedelta(days=14)


def _spec() -> dict:
    config = {
        "signal": "burst",
        "tp_pips": 3,
        "sl_pips": None,
        "ceiling_min": 240,
    }
    return {
        "experiment_id": "dojo-worker-forward-smoke-v1",
        "window": {
            "start_utc": WINDOW_START.isoformat(),
            "end_utc": WINDOW_END.isoformat(),
        },
        "candidate_set": {
            "declared_grid_size": 1,
            "family_denominator": 1,
            "candidates": [
                {
                    "candidate_id": "burst__E1",
                    "family_id": "burst",
                    "config": config,
                    "config_sha256": canonical_sha256(config),
                }
            ],
        },
        "mechanics": {
            "pairs": ["USD_JPY"],
            "granularity": "M1",
            "bot_bar": "feed",
            "intrabar_paths": ["OHLC", "OLHC"],
            "initial_balance_jpy": 200_000,
            "slippage_pips_per_fill": 0.3,
            "financing_pips_per_day": 0.8,
            "leverage": 25,
            "period_end_settlement": "CANCEL_OWNED_ORDERS_THEN_CLOSE_OWNED_POSITIONS",
            "terminal_score_basis": "FULLY_RESOLVED_BALANCE",
        },
        "source_bindings": {
            "git_commit": "a" * 40,
            "runner_sha256": "b" * 64,
            "bot_module_sha256": "c" * 64,
            "bot_dependency_sha256": {
                "src/quant_rabbit/virtual_broker.py": "d" * 64,
            },
            "scorer_sha256": "e" * 64,
            "precommit_builder_sha256": "f" * 64,
        },
        "thresholds": {
            "minimum_calendar_30d_multiple": 3.0,
            "maximum_margin_closeouts": 0,
            "zero_trade_policy": "FAIL_CLOSED",
        },
        "daily_source_policy": {
            "source_origin": "OANDA historical bid/ask closed-candle export",
            "seal_grace_hours": 12,
            "minimum_open_market_days": 10,
            "expected_source_ids": ["USD_JPY:M1"],
            "late_backfill_allowed": False,
            "market_closed_days_require_explicit_receipt": True,
        },
    }


def _source(day_start: datetime, *, closed: bool = False) -> dict:
    if closed:
        return {
            "market_closed": True,
            "closure_reason": "PREDECLARED_MARKET_CLOSURE",
            "sources": [],
        }
    return {
        "market_closed": False,
        "closure_reason": None,
        "sources": [
            {
                "source_id": "USD_JPY:M1",
                "pair": "USD_JPY",
                "granularity": "M1",
                "content_sha256": "1" * 64,
                "size_bytes": 123,
                "row_count": 60,
                "first_event_utc": (day_start + timedelta(minutes=1)).isoformat(),
                "last_event_utc": (day_start + timedelta(hours=23)).isoformat(),
            }
        ],
    }


def _parents() -> tuple[dict, dict]:
    precommit = build_precommit(_spec(), now_utc=WINDOW_START - timedelta(days=2))
    start = build_start_receipt(precommit, now_utc=WINDOW_START - timedelta(days=1))
    return precommit, start


def _day_chain(precommit: dict, start: dict) -> list[dict]:
    days: list[dict] = []
    for ordinal in range(1, 15):
        day_start = WINDOW_START + timedelta(days=ordinal - 1)
        days.append(
            build_day_seal(
                precommit,
                start,
                days[-1] if days else None,
                _source(day_start, closed=ordinal in {6, 7, 13, 14}),
                ordinal=ordinal,
                now_utc=day_start + timedelta(days=1, hours=1),
            )
        )
    return days


def _results(precommit: dict, *, positive: bool = True) -> dict:
    rows = []
    for intrabar in ("OHLC", "OLHC"):
        rows.append(
            {
                "candidate_id": "burst__E1",
                "intrabar": intrabar,
                "status": (
                    "PASS_POSITIVE_RESOLVED_BALANCE"
                    if positive
                    else "FAIL_NON_POSITIVE_RESOLVED_BALANCE"
                ),
                "ledger_sha256": ("2" if intrabar == "OHLC" else "3") * 64,
                "score_receipt_sha256": ("4" if intrabar == "OHLC" else "5") * 64,
                "entries": 4,
                "resolved_exits": 4,
                "terminal_net_jpy": 120_000 if positive else -2_000,
                "calendar_30d_multiple": 3.2 if positive else 0.98,
                "margin_closeouts": 0,
                "terminal_resolved": True,
                "promotion_eligible": False,
            }
        )
    return {
        "precommit_sha256": precommit["precommit_sha256"],
        "window_start_utc": precommit["window"]["start_utc"],
        "window_end_utc": precommit["window"]["end_utc"],
        "results": rows,
    }


def test_precommit_is_exact_future_fixed_and_never_promotable() -> None:
    precommit = build_precommit(_spec(), now_utc=WINDOW_START - timedelta(days=2))
    assert validate_precommit(precommit) == precommit
    assert precommit["candidate_set"]["candidate_count"] == 1
    assert precommit["study"]["smoke_only"] is True
    assert precommit["study"]["proof_threshold_met_by_this_window"] is False
    assert precommit["authority"]["promotion_eligible"] is False
    assert precommit["attestations"]["self_attested_only"] is True

    with pytest.raises(DojoWorkerForwardError, match="must precede"):
        build_precommit(_spec(), now_utc=WINDOW_START)

    injected = json.loads(json.dumps(precommit))
    injected["live_permission"] = True
    body = {key: value for key, value in injected.items() if key != "precommit_sha256"}
    injected["precommit_sha256"] = canonical_sha256(body)
    with pytest.raises(DojoWorkerForwardError, match="schema mismatch"):
        validate_precommit(injected)


def test_start_must_be_after_precommit_and_before_window() -> None:
    precommit = build_precommit(_spec(), now_utc=WINDOW_START - timedelta(days=2))
    with pytest.raises(DojoWorkerForwardError, match="predates"):
        build_start_receipt(precommit, now_utc=WINDOW_START - timedelta(days=3))
    with pytest.raises(DojoWorkerForwardError, match="missed"):
        build_start_receipt(precommit, now_utc=WINDOW_START)


def test_daily_seals_require_order_coverage_and_deadline() -> None:
    precommit, start = _parents()
    day_start = WINDOW_START
    with pytest.raises(DojoWorkerForwardError, match="before day end"):
        build_day_seal(
            precommit,
            start,
            None,
            _source(day_start),
            ordinal=1,
            now_utc=day_start + timedelta(hours=1),
        )
    with pytest.raises(DojoWorkerForwardError, match="deadline"):
        build_day_seal(
            precommit,
            start,
            None,
            _source(day_start),
            ordinal=1,
            now_utc=day_start + timedelta(days=1, hours=13),
        )
    with pytest.raises(DojoWorkerForwardError, match="gap"):
        build_day_seal(
            precommit,
            start,
            None,
            _source(day_start + timedelta(days=1)),
            ordinal=2,
            now_utc=day_start + timedelta(days=2, hours=1),
        )


def test_finalize_requires_exact_two_path_denominator_and_stays_diagnostic() -> None:
    precommit, start = _parents()
    days = _day_chain(precommit, start)
    incomplete = _results(precommit)
    incomplete["results"].pop()
    with pytest.raises(DojoWorkerForwardError, match="incomplete"):
        build_final_receipt(
            precommit,
            start,
            days,
            incomplete,
            now_utc=WINDOW_END,
        )

    final = build_final_receipt(
        precommit,
        start,
        days,
        _results(precommit),
        now_utc=WINDOW_END,
    )
    assert final["state"] == "FINALIZED_PASS"
    assert final["smoke_passing_candidate_ids"] == ["burst__E1"]
    assert final["source_coverage_passed"] is True
    assert final["proof_eligible"] is False
    assert final["promotion_eligible"] is False
    assert final["effective_independent_n"] == 0
    assert final["goal_status"] == "3X_NOT_REACHABLE"


def test_local_append_only_writer_and_gap_audit_fail_closed(tmp_path: Path) -> None:
    precommit, start = _parents()
    write_new_json(tmp_path / "precommit.json", precommit)
    write_new_json(tmp_path / "start.json", start)
    with pytest.raises(FileExistsError):
        write_new_json(tmp_path / "start.json", start)
    status = audit_lifecycle(
        tmp_path,
        now_utc=WINDOW_START + timedelta(days=1, hours=13),
    )
    assert status["state"] == "GAP"
    assert status["promotion_eligible"] is False
    assert "DAY_1_SEAL_DEADLINE_MISSED" in status["blockers"]
