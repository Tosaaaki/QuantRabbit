from __future__ import annotations

import importlib.util
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

from quant_rabbit.dojo_lab_provenance import (
    DojoLabProvenanceError,
    OwnedBrokerView,
    StrategyOwnershipError,
    combine_intrabar_results,
    create_run_root,
    create_trial_dir,
    reserve_window_plan,
    score_session_ledger,
    strategy_ownership_registry,
    validate_window_plan,
    write_new_json,
)
from quant_rabbit.virtual_broker import VirtualBroker, VirtualBrokerError


BOT_PATH = Path("/tmp/dojo-lab-bot.py")
BOT_SHA = "c" * 64
CONFIG_TEXT = '{"signal":"test"}'
CONFIG_SHA = hashlib.sha256(CONFIG_TEXT.encode()).hexdigest()
OWNER_ID = "dojo:test-owner"
DEPENDENCIES = {"src/quant_rabbit/virtual_broker.py": "e" * 64}


def _strategy_event(event: str) -> bool:
    return event.startswith("EXIT") or event in {
        "ORDER_REJECTED_INSUFFICIENT_MARGIN",
        "ORDER_REJECTED_CONCURRENCY_CAP",
        "FILL_MARKET",
        "ORDER_LIMIT",
        "ORDER_STOP",
        "ORDER_CANCEL",
        "ORDER_CANCEL_CONCURRENCY_CAP",
        "LIMIT_REJECTED_INSUFFICIENT_MARGIN",
        "FILL_LIMIT",
        "CLOSE",
        "SET_EXIT",
        "MARGIN_CLOSEOUT",
    }


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _record(event: str, payload: dict) -> str:
    """Legacy helper retained only for non-collected historical assertions."""

    return json.dumps({"event": event, "payload": payload}) + "\n"


def _manifest(*, intrabar: str = "OHLC", slippage: float = 0.3) -> dict:
    body = {
        "schema": "QR_VIRTUAL_SESSION_REPRODUCIBILITY_V1",
        "source": {
            "git_head": "test-head",
            "session_script_sha256": "a" * 64,
            "virtual_broker_sha256": "b" * 64,
            "python_executable": "/python",
            "python_version": "test",
        },
        "replay": {
            "feed": "replay",
            "pairs": ["USD_JPY"],
            "time_from": "2026-01-01T00:00:00",
            "time_to": "2026-01-31T00:00:00",
            "granularity": "M1",
            "intrabar": intrabar,
            "bot_bar": "feed",
            "period_end_settlement": False,
        },
        "corpus": {},
        "costs": {
            "slippage_pips_per_fill": slippage,
            "financing_pips_per_day": 0.8,
            "leverage": 25.0,
        },
        "initial_balance_jpy": 200_000.0,
        "resume_snapshot": None,
        "bot": {
            "kind": "custom_module",
            "name": None,
            "module_path": str(BOT_PATH),
            "module_sha256": BOT_SHA,
            "class": "Bot",
            "strategy_owner_id": OWNER_ID,
            "dependency_sha256": DEPENDENCIES,
            "configuration_bindings": {
                "DOJO_BOT_CONFIG": {
                    "sha256": CONFIG_SHA,
                    "length": len(CONFIG_TEXT),
                }
            },
        },
        "pacing": {"bars_per_second": 100000.0, "step": False, "state_every": 1},
        "order_authority": "NONE",
    }
    corpus_body = {
        "root": "/corpus",
        "shards": [
            {
                "path": "oanda/USD_JPY/USD_JPY_M1_BA_2026.jsonl.gz",
                "size_bytes": 123,
                "sha256": "d" * 64,
            }
        ],
    }
    body["corpus"] = {**corpus_body, "corpus_sha256": _sha(corpus_body)}
    return {**body, "manifest_sha256": _sha(body)}


def _ledger_lines(
    events: list[tuple[str, dict]],
    *,
    intrabar: str = "OHLC",
    manifest_slippage: float = 0.3,
) -> list[str]:
    manifest = _manifest(intrabar=intrabar, slippage=manifest_slippage)
    owned_events = [
        (
            event,
            {
                **payload,
                **(
                    {"strategy_owner_id": payload.get("strategy_owner_id", OWNER_ID)}
                    if _strategy_event(event)
                    else {}
                ),
            },
        )
        for event, payload in events
    ]
    all_events = [
        (
            "SESSION_START",
            {
                "contract": "QR_VIRTUAL_MARKET_SESSION_V1",
                "feed": "replay",
                "pairs": "USD_JPY",
                "balance": 200_000.0,
                "order_authority": "NONE",
                "reproducibility_manifest": manifest,
                "reproducibility_manifest_sha256": manifest["manifest_sha256"],
            },
        ),
        *owned_events,
    ]
    previous = "0" * 64
    lines: list[str] = []
    for index, (event, payload) in enumerate(all_events):
        body = {
            "ts_utc": f"2026-01-{index + 1:02d}T00:00:00+00:00",
            "event": event,
            "payload": payload,
            "prev_sha": previous,
        }
        record = {**body, "sha": _sha(body)}
        previous = record["sha"]
        lines.append(json.dumps(record, sort_keys=True))
    return lines


def _score_kwargs(*, intrabar: str = "OHLC") -> dict:
    return {
        "start_balance_jpy": 200_000.0,
        "window_role": "VAL",
        "window": ("2026-01-01T00:00:00", "2026-01-31T00:00:00"),
        "intrabar": intrabar,
        "legacy_contaminated": False,
        "expected_pairs": ("USD_JPY",),
        "expected_granularity": "M1",
        "expected_bot_bar": "feed",
        "expected_period_end_settlement": False,
        "expected_slippage_pips": 0.3,
        "expected_financing_pips_per_day": 0.8,
        "expected_bot_module_path": BOT_PATH,
        "expected_bot_module_sha256": BOT_SHA,
        "expected_bot_dependency_sha256": DEPENDENCIES,
        "expected_strategy_owner_id": OWNER_ID,
        "expected_bot_config_sha256": CONFIG_SHA,
        "expected_bot_config_length": len(CONFIG_TEXT),
        "reservation_evidence": {
            "status": "RESERVED",
            "promotion_eligible": True,
        },
    }


def _load_bot_module(name: str):
    path = Path(__file__).resolve().parents[1] / "bots" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"dojo_test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_script_module(filename: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    module_name = "dojo_test_" + filename.replace("-", "_").replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_window_plan_is_disjoint_and_screened_holdout_fails_closed() -> None:
    windows = {
        "TRAIN": ("2024-01-01T00:00:00", "2025-01-01T00:00:00"),
        "VAL": ("2025-01-01T00:00:00", "2026-01-01T00:00:00"),
        "FINAL": ("2026-02-01T00:00:00", "2026-03-01T00:00:00"),
    }
    plan = validate_window_plan(
        windows,
        screened_windows={
            "quarantined": ("2026-01-01T00:00:00", "2026-02-01T00:00:00")
        },
    )
    assert plan["windows"]["FINAL"]["start_utc"] == "2026-02-01T00:00:00Z"

    overlapping = dict(windows)
    overlapping["VAL"] = ("2024-12-01T00:00:00", "2026-01-01T00:00:00")
    with pytest.raises(DojoLabProvenanceError, match="evaluation windows overlap"):
        validate_window_plan(overlapping)

    with pytest.raises(DojoLabProvenanceError, match="overlaps holdout FINAL"):
        validate_window_plan(
            windows,
            screened_windows={
                "used_screen": ("2026-02-15T00:00:00", "2026-02-20T00:00:00")
            },
        )

    reversed_plan = dict(windows)
    reversed_plan["TRAIN"] = ("2026-03-01T00:00:00", "2026-04-01T00:00:00")
    with pytest.raises(DojoLabProvenanceError, match="chronological"):
        validate_window_plan(reversed_plan)


def test_global_holdout_reservation_is_durable_and_reuse_fails_closed(
    tmp_path: Path,
) -> None:
    windows = {
        "TRAIN": ("2024-01-01T00:00:00", "2025-01-01T00:00:00"),
        "VAL": ("2030-01-01T00:00:00", "2030-02-01T00:00:00"),
        "FINAL": ("2031-02-01T00:00:00", "2031-03-01T00:00:00"),
    }
    plan = validate_window_plan(windows)
    absent = reserve_window_plan(
        None,
        run_id="20260719T000000.000000Z-00000000",
        experiment_id="absent",
        plan=plan,
    )
    assert absent["status"] == "GLOBAL_REGISTRY_ABSENT"
    assert absent["promotion_eligible"] is False

    registry = tmp_path / "global" / "window_reservations.jsonl"
    first = reserve_window_plan(
        registry,
        run_id="20260719T000000.000000Z-00000001",
        experiment_id="first",
        plan=plan,
    )
    assert first["status"] == "RESERVED"
    assert first["promotion_eligible"] is False
    assert first["local_reservation_verified"] is True
    assert first["external_monotonicity_attested"] is False
    assert registry.read_text().count("\n") == 1
    with pytest.raises(DojoLabProvenanceError, match="holdout already reserved"):
        reserve_window_plan(
            registry,
            run_id="20260719T000000.000000Z-00000002",
            experiment_id="reuse",
            plan=plan,
        )


@pytest.mark.parametrize("filename", ["run-dojo-lab.py", "run-pair-adaptation-lab.py"])
def test_runner_window_constants_exclude_the_legacy_screen(filename: str) -> None:
    module = _load_script_module(filename)
    plan = validate_window_plan(
        module.WINDOWS, screened_windows=module.SCREENED_WINDOWS
    )
    assert (
        plan["screened_windows"]["LEGACY_S5_SCREEN"]["end_utc"]
        == (plan["windows"]["FINAL"]["start_utc"])
    )


def test_run_and_trial_paths_are_create_once(tmp_path: Path) -> None:
    run_id = "20260719T000000.000000Z-deadbeef"
    resolved_id, run_root = create_run_root(tmp_path, run_id)
    assert resolved_id == run_id
    trial = create_trial_dir(run_root, "train__candidate__ohlc")
    assert (trial / "inbox").is_dir()
    with pytest.raises(DojoLabProvenanceError, match="trial already exists"):
        create_trial_dir(run_root, "train__candidate__ohlc")
    with pytest.raises(DojoLabProvenanceError, match="run_id already exists"):
        create_run_root(tmp_path, run_id)

    result_path = run_root / "scoreboard.json"
    write_new_json(result_path, {"run_id": run_id})
    with pytest.raises(DojoLabProvenanceError, match="result already exists"):
        write_new_json(result_path, {"run_id": run_id})


def _obsolete_terminal_mark_to_market_and_calendar_month_enter_score(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        _record("FILL_MARKET", {"trade_id": "T1"})
        + _record("CLOSE", {"trade_id": "T1", "pl_jpy": 5_000.0})
        + _record("FILL_LIMIT", {"order_id": "O2", "trade_id": "T2"})
        + _record(
            "SESSION_STOP",
            {
                "account": {
                    "balance_jpy": 205_000.0,
                    "equity_jpy": 207_500.0,
                    "open_positions": 1,
                    "resting_orders": 1,
                }
            },
        ),
        encoding="utf-8",
    )
    score = score_session_ledger(
        ledger,
        start_balance_jpy=200_000.0,
        window_role="FINAL",
        window=("2026-01-01T00:00:00", "2026-02-01T00:00:00"),
        intrabar="OHLC",
        legacy_contaminated=False,
    )
    assert score["entries"] == 2
    assert score["realized_net_jpy"] == 5_000.0
    assert score["terminal_unrealized_jpy"] == 2_500.0
    assert score["terminal_net_jpy"] == 7_500.0
    assert score["open_positions_marked"] == 1
    assert score["calendar_days"] == 31.0
    assert score["calendar_30d_multiple"] == pytest.approx(
        (207_500 / 200_000) ** (30 / 31), abs=1e-8
    )
    assert score["promotion_eligible"] is True


def _obsolete_zero_trade_is_invalid_even_when_terminal_equity_is_positive(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        _record(
            "SESSION_STOP",
            {
                "account": {
                    "balance_jpy": 200_100.0,
                    "equity_jpy": 200_100.0,
                    "open_positions": 0,
                    "resting_orders": 0,
                }
            },
        ),
        encoding="utf-8",
    )
    score = score_session_ledger(
        ledger,
        start_balance_jpy=200_000.0,
        window_role="VAL",
        window=("2026-01-01T00:00:00", "2026-01-31T00:00:00"),
        intrabar="OLHC",
        legacy_contaminated=False,
    )
    assert score["status"] == "INVALID_ZERO_TRADES"
    assert score["economic_gate_passed"] is False
    assert "ZERO_TRADES" in score["promotion_blockers"]


def test_open_position_or_order_at_stop_is_not_scoreable(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    lines = _ledger_lines(
        [
            ("FILL_MARKET", {"trade_id": "T1", "slippage_pips": 0.3}),
            (
                "CLOSE",
                {
                    "trade_id": "T1",
                    "pl_jpy": 5_000.0,
                    "financing_jpy": 12.0,
                    "slippage_pips": 0.3,
                },
            ),
            (
                "FILL_LIMIT",
                {"order_id": "O2", "trade_id": "T2", "slippage_pips": 0.3},
            ),
            (
                "SESSION_STOP",
                {
                    "account": {
                        "balance_jpy": 205_000.0,
                        "equity_jpy": 207_500.0,
                        "open_positions": 1,
                        "resting_orders": 1,
                    }
                },
            ),
        ]
    )
    ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
    score = score_session_ledger(ledger, **_score_kwargs())
    assert score["entries"] == 2
    assert score["terminal_net_jpy"] == 5_000.0
    assert score["terminal_unrealized_jpy"] == 2_500.0
    assert score["status"] == "INVALID_TERMINAL_EXPOSURE"
    assert score["economic_gate_passed"] is False
    assert score["promotion_eligible"] is False


def test_zero_trade_and_missing_global_registry_are_explicit(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    lines = _ledger_lines(
        [
            (
                "SESSION_STOP",
                {
                    "account": {
                        "balance_jpy": 200_000.0,
                        "equity_jpy": 200_000.0,
                        "open_positions": 0,
                        "resting_orders": 0,
                    }
                },
            )
        ],
        intrabar="OLHC",
    )
    ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
    kwargs = _score_kwargs(intrabar="OLHC")
    kwargs["reservation_evidence"] = None
    score = score_session_ledger(ledger, **kwargs)
    assert score["status"] == "INVALID_ZERO_TRADES"
    assert "ZERO_TRADES" in score["promotion_blockers"]
    assert "GLOBAL_WINDOW_RESERVATION_ABSENT" in score["promotion_blockers"]


def test_past_holdout_reservation_is_permanently_diagnostic(
    tmp_path: Path,
) -> None:
    plan = validate_window_plan(
        {
            "TRAIN": ("2025-01-01T00:00:00", "2026-01-01T00:00:00"),
            "VAL": ("2026-01-01T00:00:00", "2026-01-31T00:00:00"),
            "FINAL": ("2026-02-01T00:00:00", "2026-03-01T00:00:00"),
        }
    )
    reservation = reserve_window_plan(
        tmp_path / "global.jsonl",
        run_id="20260719T000000.000000Z-aabbccdd",
        experiment_id="score-test",
        plan=plan,
    )
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        "\n".join(
            _ledger_lines(
                [
                    (
                        "FILL_MARKET",
                        {"trade_id": "T1", "slippage_pips": 0.3},
                    ),
                    (
                        "CLOSE",
                        {
                            "trade_id": "T1",
                            "pl_jpy": 100.0,
                            "financing_jpy": 1.0,
                            "slippage_pips": 0.3,
                        },
                    ),
                    (
                        "SESSION_STOP",
                        {
                            "account": {
                                "balance_jpy": 200_100.0,
                                "equity_jpy": 200_100.0,
                                "open_positions": 0,
                                "resting_orders": 0,
                            }
                        },
                    ),
                ]
            )
        )
        + "\n",
        encoding="utf-8",
    )
    kwargs = _score_kwargs()
    kwargs["reservation_evidence"] = reservation
    score = score_session_ledger(ledger, **kwargs)
    assert score["economic_gate_passed"] is True
    assert reservation["status"] == "RESERVED_HISTORICAL_DIAGNOSTIC"
    assert reservation["reserved_before_every_holdout"] is False
    assert reservation["historical_diagnostic_only"] is True
    assert reservation["promotion_blocker"] == "HOLDOUT_RESERVED_AFTER_START"
    assert score["local_candidate_eligible"] is False
    assert score["promotion_eligible"] is False
    assert "GLOBAL_WINDOW_RESERVATION_UNVERIFIED" in score["promotion_blockers"]
    assert score["reservation_status"] == "RESERVED_HISTORICAL_DIAGNOSTIC"


def test_ledger_chain_manifest_cost_and_terminal_framing_are_authenticated(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    events = [
        ("FILL_MARKET", {"trade_id": "T1", "slippage_pips": 0.3}),
        (
            "CLOSE",
            {
                "trade_id": "T1",
                "pl_jpy": 100.0,
                "financing_jpy": 1.0,
                "slippage_pips": 0.3,
            },
        ),
        (
            "SESSION_STOP",
            {
                "account": {
                    "balance_jpy": 200_100.0,
                    "equity_jpy": 200_100.0,
                    "open_positions": 0,
                    "resting_orders": 0,
                }
            },
        ),
    ]
    good_lines = _ledger_lines(events)

    tampered = [json.loads(line) for line in good_lines]
    tampered[1]["payload"]["trade_id"] = "T999"
    ledger.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in tampered) + "\n"
    )
    with pytest.raises(DojoLabProvenanceError, match="record hash mismatch"):
        score_session_ledger(ledger, **_score_kwargs())

    ledger.write_text("\n".join(_ledger_lines(events, manifest_slippage=0.0)) + "\n")
    with pytest.raises(
        DojoLabProvenanceError, match="manifest hardened costs mismatch"
    ):
        score_session_ledger(ledger, **_score_kwargs())

    nonterminal_lines = _ledger_lines([events[-1], ("BOT_LOADED", {"module": "late"})])
    ledger.write_text("\n".join(nonterminal_lines) + "\n")
    with pytest.raises(DojoLabProvenanceError, match="terminal SESSION_STOP"):
        score_session_ledger(ledger, **_score_kwargs())

    ledger.write_text("\n".join(good_lines) + "\n")
    zero_cost_kwargs = _score_kwargs()
    zero_cost_kwargs["expected_slippage_pips"] = 0.0
    with pytest.raises(DojoLabProvenanceError, match="hardened slippage"):
        score_session_ledger(ledger, **zero_cost_kwargs)

    missing_exit_cost = list(events)
    missing_exit_cost[1] = (
        "CLOSE",
        {"trade_id": "T1", "pl_jpy": 100.0, "slippage_pips": 0.3},
    )
    ledger.write_text("\n".join(_ledger_lines(missing_exit_cost)) + "\n")
    with pytest.raises(DojoLabProvenanceError, match="exit cost evidence"):
        score_session_ledger(ledger, **_score_kwargs())


def test_strategy_owner_and_dependency_closure_are_fail_closed(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    events = [
        ("FILL_MARKET", {"trade_id": "T1", "slippage_pips": 0.3}),
        (
            "CLOSE",
            {
                "trade_id": "T1",
                "pl_jpy": 100.0,
                "financing_jpy": 1.0,
                "slippage_pips": 0.3,
            },
        ),
        (
            "SESSION_STOP",
            {
                "account": {
                    "balance_jpy": 200_100.0,
                    "equity_jpy": 200_100.0,
                    "open_positions": 0,
                    "resting_orders": 0,
                }
            },
        ),
    ]
    ledger.write_text("\n".join(_ledger_lines(events)) + "\n")

    owner_kwargs = _score_kwargs()
    owner_kwargs["expected_strategy_owner_id"] = "dojo:wrong-owner"
    with pytest.raises(DojoLabProvenanceError, match="mismatched owner"):
        score_session_ledger(ledger, **owner_kwargs)

    dependency_kwargs = _score_kwargs()
    dependency_kwargs["expected_bot_dependency_sha256"] = {
        "src/quant_rabbit/virtual_broker.py": "f" * 64
    }
    with pytest.raises(DojoLabProvenanceError, match="dependency closure mismatch"):
        score_session_ledger(ledger, **dependency_kwargs)


def _write_concurrency_ledger(tmp_path: Path) -> Path:
    ledger = tmp_path / "concurrency-ledger.jsonl"
    ledger.write_text(
        "\n".join(
            _ledger_lines(
                [
                    (
                        "FILL_MARKET",
                        {
                            "trade_id": "T1",
                            "pair": "USD_JPY",
                            "units": 1_000.0,
                            "slippage_pips": 0.3,
                        },
                    ),
                    (
                        "CLOSE",
                        {
                            "trade_id": "T1",
                            "units": 1_000.0,
                            "pl_jpy": 50.0,
                            "financing_jpy": 0.0,
                            "slippage_pips": 0.3,
                        },
                    ),
                    (
                        "FILL_MARKET",
                        {
                            "trade_id": "T2",
                            "pair": "USD_JPY",
                            "units": 1_000.0,
                            "slippage_pips": 0.3,
                        },
                    ),
                    (
                        "CLOSE",
                        {
                            "trade_id": "T2",
                            "units": 1_000.0,
                            "pl_jpy": 50.0,
                            "financing_jpy": 0.0,
                            "slippage_pips": 0.3,
                        },
                    ),
                    (
                        "SESSION_STOP",
                        {
                            "account": {
                                "balance_jpy": 200_100.0,
                                "equity_jpy": 200_100.0,
                                "open_positions": 0,
                                "resting_orders": 0,
                            }
                        },
                    ),
                ]
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return ledger


@pytest.mark.parametrize(
    ("pair_cap", "global_cap", "message"),
    [
        (1, None, "declared together"),
        (True, 1, "positive integers"),
        (1, 0, "positive integers"),
    ],
)
def test_score_owner_concurrency_cap_contract_is_fail_closed(
    tmp_path: Path,
    pair_cap: object,
    global_cap: object,
    message: str,
) -> None:
    ledger = _write_concurrency_ledger(tmp_path)
    kwargs = _score_kwargs()
    kwargs["expected_max_concurrent_per_pair"] = pair_cap
    kwargs["expected_global_max_concurrent"] = global_cap

    with pytest.raises(DojoLabProvenanceError, match=message):
        score_session_ledger(ledger, **kwargs)


def test_score_reconstructs_owner_concurrency_from_fill_exit_ledger(
    tmp_path: Path,
) -> None:
    ledger = _write_concurrency_ledger(tmp_path)
    score = score_session_ledger(
        ledger,
        **_score_kwargs(),
        expected_max_concurrent_per_pair=1,
        expected_global_max_concurrent=1,
    )

    assert score["status"] == "PASS_POSITIVE_RESOLVED_BALANCE"
    assert score["owner_concurrency"] == {
        "status": "VERIFIED_FROM_FILL_EXIT_LEDGER",
        "strategy_owner_id": OWNER_ID,
        "max_concurrent_per_pair": 1,
        "global_max_concurrent": 1,
        "observed_peak_global": 1,
        "observed_peak_by_pair": {"USD_JPY": 1},
        "resolved_trade_count": 2,
        "terminal_active_positions": 0,
    }


@pytest.mark.parametrize(
    ("pair_cap", "global_cap", "message"),
    [
        (1, 2, "per-pair concurrency cap exceeded"),
        (2, 1, "global concurrency cap exceeded"),
    ],
)
def test_score_rejects_ledger_that_exceeded_owner_concurrency_cap(
    tmp_path: Path,
    pair_cap: int,
    global_cap: int,
    message: str,
) -> None:
    ledger = tmp_path / "concurrency-breach-ledger.jsonl"
    ledger.write_text(
        "\n".join(
            _ledger_lines(
                [
                    (
                        "FILL_MARKET",
                        {
                            "trade_id": "T1",
                            "pair": "USD_JPY",
                            "units": 1_000.0,
                            "slippage_pips": 0.3,
                        },
                    ),
                    (
                        "FILL_MARKET",
                        {
                            "trade_id": "T2",
                            "pair": "USD_JPY",
                            "units": 1_000.0,
                            "slippage_pips": 0.3,
                        },
                    ),
                    (
                        "CLOSE",
                        {
                            "trade_id": "T1",
                            "units": 1_000.0,
                            "pl_jpy": 50.0,
                            "financing_jpy": 0.0,
                            "slippage_pips": 0.3,
                        },
                    ),
                    (
                        "CLOSE",
                        {
                            "trade_id": "T2",
                            "units": 1_000.0,
                            "pl_jpy": 50.0,
                            "financing_jpy": 0.0,
                            "slippage_pips": 0.3,
                        },
                    ),
                    (
                        "SESSION_STOP",
                        {
                            "account": {
                                "balance_jpy": 200_100.0,
                                "equity_jpy": 200_100.0,
                                "open_positions": 0,
                                "resting_orders": 0,
                            }
                        },
                    ),
                ]
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DojoLabProvenanceError, match=message):
        score_session_ledger(
            ledger,
            **_score_kwargs(),
            expected_max_concurrent_per_pair=pair_cap,
            expected_global_max_concurrent=global_cap,
        )


def test_score_keeps_partial_close_active_until_remaining_units_are_closed(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "partial-close-ledger.jsonl"
    ledger.write_text(
        "\n".join(
            _ledger_lines(
                [
                    (
                        "FILL_MARKET",
                        {
                            "trade_id": "T1",
                            "pair": "USD_JPY",
                            "units": 1_000.0,
                            "slippage_pips": 0.3,
                        },
                    ),
                    (
                        "CLOSE",
                        {
                            "trade_id": "T1",
                            "units": 400.0,
                            "pl_jpy": 40.0,
                            "financing_jpy": 0.0,
                            "slippage_pips": 0.3,
                        },
                    ),
                    (
                        "CLOSE",
                        {
                            "trade_id": "T1",
                            "units": 600.0,
                            "pl_jpy": 60.0,
                            "financing_jpy": 0.0,
                            "slippage_pips": 0.3,
                        },
                    ),
                    (
                        "SESSION_STOP",
                        {
                            "account": {
                                "balance_jpy": 200_100.0,
                                "equity_jpy": 200_100.0,
                                "open_positions": 0,
                                "resting_orders": 0,
                            }
                        },
                    ),
                ]
            )
        )
        + "\n",
        encoding="utf-8",
    )

    score = score_session_ledger(
        ledger,
        **_score_kwargs(),
        expected_max_concurrent_per_pair=1,
        expected_global_max_concurrent=1,
    )

    assert score["status"] == "PASS_POSITIVE_RESOLVED_BALANCE"
    assert score["entries"] == 1
    assert score["resolved_exits"] == 2
    assert score["owner_concurrency"]["resolved_trade_count"] == 1
    assert score["owner_concurrency"]["terminal_active_positions"] == 0


@pytest.mark.parametrize(
    ("close_units", "message"),
    [
        (0.0, "invalid close units"),
        (-1.0, "invalid close units"),
        (1_001.0, "over-closes a trade"),
    ],
)
def test_score_rejects_invalid_partial_close_units(
    tmp_path: Path,
    close_units: float,
    message: str,
) -> None:
    ledger = tmp_path / f"invalid-close-{close_units}.jsonl"
    ledger.write_text(
        "\n".join(
            _ledger_lines(
                [
                    (
                        "FILL_MARKET",
                        {
                            "trade_id": "T1",
                            "pair": "USD_JPY",
                            "units": 1_000.0,
                            "slippage_pips": 0.3,
                        },
                    ),
                    (
                        "CLOSE",
                        {
                            "trade_id": "T1",
                            "units": close_units,
                            "pl_jpy": 100.0,
                            "financing_jpy": 0.0,
                            "slippage_pips": 0.3,
                        },
                    ),
                    (
                        "SESSION_STOP",
                        {
                            "account": {
                                "balance_jpy": 200_100.0,
                                "equity_jpy": 200_100.0,
                                "open_positions": 0,
                                "resting_orders": 0,
                            }
                        },
                    ),
                ]
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DojoLabProvenanceError, match=message):
        score_session_ledger(
            ledger,
            **_score_kwargs(),
            expected_max_concurrent_per_pair=1,
            expected_global_max_concurrent=1,
        )


def test_both_intrabar_paths_are_required_and_lower_path_is_authoritative() -> None:
    common = {
        "entries": 3,
        "margin_closeouts": 0,
        "calendar_30d_multiple": 1.01,
        "promotion_eligible": True,
        "promotion_blockers": [],
        "intrabar_pair_manifest_sha256": "e" * 64,
    }
    combined = combine_intrabar_results(
        [
            {
                **common,
                "intrabar": "OHLC",
                "terminal_net_jpy": 1_000.0,
                "economic_gate_passed": True,
            },
            {
                **common,
                "intrabar": "OLHC",
                "terminal_net_jpy": -1.0,
                "economic_gate_passed": False,
            },
        ]
    )
    assert combined["pessimistic_intrabar"] == "OLHC"
    assert combined["pessimistic_terminal_net_jpy"] == -1.0
    assert combined["gate_passed"] is False
    with pytest.raises(DojoLabProvenanceError, match="exactly OHLC and OLHC"):
        combine_intrabar_results(
            [
                {
                    **common,
                    "intrabar": "OHLC",
                    "terminal_net_jpy": 1_000.0,
                    "economic_gate_passed": True,
                }
            ]
        )

    mismatched = [
        {
            **common,
            "intrabar": "OHLC",
            "terminal_net_jpy": 1.0,
            "economic_gate_passed": True,
        },
        {
            **common,
            "intrabar": "OLHC",
            "terminal_net_jpy": 1.0,
            "economic_gate_passed": True,
            "intrabar_pair_manifest_sha256": "f" * 64,
        },
    ]
    with pytest.raises(DojoLabProvenanceError, match="differ outside"):
        combine_intrabar_results(mismatched)


def test_order_owner_promotes_to_trade_and_other_owner_cannot_mutate(
    tmp_path: Path,
) -> None:
    broker = VirtualBroker(tmp_path / "ledger.jsonl", fast_ledger=True)
    broker.on_quote("USD_JPY", 150.00, 150.01, "2026-01-01T00:00:00+00:00")
    owner_a = OwnedBrokerView(broker, "combo:0:a")
    owner_b = OwnedBrokerView(broker, "combo:1:b")
    order_id = owner_a.limit_order("USD_JPY", "LONG", 1_000, price=149.95)
    with pytest.raises(StrategyOwnershipError, match="unowned order"):
        owner_b.cancel_order(order_id)

    broker.on_quote("USD_JPY", 149.93, 149.94, "2026-01-01T00:01:00+00:00")
    registry = strategy_ownership_registry(broker)
    trade_id = next(iter(broker.positions))
    assert registry.historical_order_owner(order_id) == "combo:0:a"
    assert registry.historical_trade_owner(trade_id) == "combo:0:a"
    assert owner_a.active_trade_ids() == (trade_id,)
    assert owner_b.active_trade_ids() == ()
    with pytest.raises(StrategyOwnershipError, match="unowned trade"):
        owner_b.close_trade(trade_id)
    owner_a.close_trade(trade_id)

    records = [json.loads(line) for line in broker.ledger_path.read_text().splitlines()]
    owned_events = [
        row for row in records if row["event"] in {"ORDER_LIMIT", "FILL_LIMIT", "CLOSE"}
    ]
    assert owned_events
    assert {row["payload"]["strategy_owner_id"] for row in owned_events} == {
        "combo:0:a"
    }


def test_duplicate_owner_id_fails_closed(tmp_path: Path) -> None:
    broker = VirtualBroker(tmp_path / "ledger.jsonl", fast_ledger=True)
    OwnedBrokerView(broker, "same-owner")
    with pytest.raises(StrategyOwnershipError, match="duplicate strategy owner id"):
        OwnedBrokerView(broker, "same-owner")


def test_owned_facade_hides_raw_broker_and_returns_immutable_queries(
    tmp_path: Path,
) -> None:
    broker = VirtualBroker(tmp_path / "ledger.jsonl", fast_ledger=True)
    broker.on_quote("USD_JPY", 150.00, 150.01, "2026-01-01T00:00:00+00:00")
    view = OwnedBrokerView(broker, "opaque-owner")
    with pytest.raises(AttributeError):
        getattr(view, "_broker")
    with pytest.raises(AttributeError):
        getattr(view, "positions")
    with pytest.raises(AttributeError):
        getattr(view, "_OwnedBrokerView__broker")

    account = view.account()
    with pytest.raises(TypeError):
        account["balance_jpy"] = 1  # type: ignore[index]
    trade_id = view.market_order("USD_JPY", "LONG", 1_000)
    position = view.position(trade_id)
    assert position is not None
    with pytest.raises(AttributeError):
        position.side = "SHORT"  # type: ignore[misc]


def test_owned_facade_exposes_exact_current_market_entry_for_sizing(
    tmp_path: Path,
) -> None:
    broker = VirtualBroker(
        tmp_path / "ledger.jsonl", fast_ledger=True, slippage_pips=0.3
    )
    broker.on_quote("USD_JPY", 150.00, 150.01, "2026-01-01T00:00:00+00:00")
    view = OwnedBrokerView(broker, "entry-sizing-owner")

    quote = view.executable_quote("USD_JPY")
    assert (quote.pair, quote.bid, quote.ask) == ("USD_JPY", 150.00, 150.01)
    assert view.executable_market_entry_price("USD_JPY", "LONG") == 150.013
    assert view.executable_market_entry_price("USD_JPY", "SHORT") == 149.997
    with pytest.raises(VirtualBrokerError, match="invalid side"):
        view.executable_market_entry_price("USD_JPY", "FLAT")
    with pytest.raises(VirtualBrokerError, match="no live quote"):
        view.executable_market_entry_price("EUR_USD", "LONG")
    with pytest.raises(VirtualBrokerError, match="no live quote"):
        view.executable_quote("EUR_USD")


def test_combo_hand_with_short_ceiling_cannot_adopt_or_close_sibling_trade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    combo_module = _load_bot_module("combo_bot")
    configs = [
        {
            "signal": "burst",
            "pairs": ["USD_JPY"],
            "tp_pips": 3,
            "ceiling_min": 1000,
            "max_concurrent": 3,
        },
        {
            "signal": "burst",
            "pairs": ["USD_JPY"],
            "tp_pips": 3,
            "ceiling_min": 1,
            "max_concurrent": 3,
        },
    ]
    monkeypatch.setenv("DOJO_BOT_COMBO", json.dumps(configs))
    broker = VirtualBroker(tmp_path / "ledger.jsonl", fast_ledger=True)
    broker.on_quote("USD_JPY", 150.00, 150.01, "2026-01-01T00:00:00+00:00")
    combo = combo_module.Bot(broker)
    assert combo.hands[0].owner_id != combo.hands[1].owner_id

    trade_id = combo.hands[0].broker.market_order("USD_JPY", "LONG", 1_000)
    bar = {
        "epoch": 60,
        "bid_o": 150.00,
        "bid_h": 150.00,
        "bid_l": 150.00,
        "bid_c": 150.00,
        "ask_o": 150.01,
        "ask_h": 150.01,
        "ask_l": 150.01,
        "ask_c": 150.01,
    }
    combo.on_bar_closed("USD_JPY", bar, 60)
    bar["epoch"] = 121
    combo.on_bar_closed("USD_JPY", bar, 121)

    assert trade_id in broker.positions
    assert trade_id in combo.hands[0].state["USD_JPY"].my_trades
    assert trade_id not in combo.hands[1].state["USD_JPY"].my_trades
    assert math.isfinite(broker.account()["equity_jpy"])


def test_same_bar_opposite_resting_orders_cannot_exceed_owner_pair_cap(
    tmp_path: Path,
) -> None:
    lab_module = _load_bot_module("lab_bot")
    broker = VirtualBroker(
        tmp_path / "ledger.jsonl", balance_jpy=2_000_000.0, fast_ledger=True
    )
    broker.on_quote("USD_JPY", 150.00, 150.02, "2026-01-01T00:00:00+00:00")
    hand = lab_module.Bot(
        broker,
        {
            "signal": "range_fade_limit",
            "pairs": ["USD_JPY"],
            "tp_pips": 5,
            "ceiling_min": 60,
            "max_concurrent_per_pair": 1,
            "global_max_concurrent": 4,
            "strategy_owner_id": "test:pair-cap",
        },
    )
    first_order = hand.broker.limit_order("USD_JPY", "LONG", 1_000, price=150.05)
    second_order = hand.broker.limit_order("USD_JPY", "SHORT", 1_000, price=149.95)

    events = broker.on_quote("USD_JPY", 150.00, 150.02, "2026-01-01T00:01:00+00:00")

    assert len(broker.positions) == 1
    assert not broker.orders
    assert events == []
    records = [json.loads(line) for line in broker.ledger_path.read_text().splitlines()]
    consequences = [
        record["payload"]
        for record in records
        if record["event"] in {"FILL_LIMIT", "ORDER_CANCEL_CONCURRENCY_CAP"}
    ]
    assert [payload["event"] for payload in consequences] == [
        "FILL_LIMIT",
        "ORDER_CANCEL_CONCURRENCY_CAP",
    ]
    assert consequences[0]["order_id"] == first_order
    assert consequences[1]["order_id"] == second_order
    assert consequences[1]["admission"] == {
        "scope": "PAIR",
        "reason": "OWNER_PAIR_CONCURRENCY_CAP_REACHED",
        "active_pair_positions": 1,
        "max_concurrent_per_pair": 1,
        "active_global_positions": 1,
        "global_max_concurrent": 4,
    }
    cancellation = next(
        json.loads(line)
        for line in broker.ledger_path.read_text().splitlines()
        if json.loads(line)["event"] == "ORDER_CANCEL_CONCURRENCY_CAP"
    )
    assert cancellation["payload"]["strategy_owner_id"] == "test:pair-cap"


def test_same_phase_cross_pair_fills_cannot_exceed_owner_global_cap(
    tmp_path: Path,
) -> None:
    lab_module = _load_bot_module("lab_bot")
    broker = VirtualBroker(
        tmp_path / "ledger.jsonl", balance_jpy=2_000_000.0, fast_ledger=True
    )
    pairs = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "CAD_JPY"]
    initial_quotes = [
        (pair, 150.00 + index, 150.02 + index, "2026-01-01T00:00:00+00:00#O")
        for index, pair in enumerate(pairs)
    ]
    broker.on_quote_batch(initial_quotes)
    hand = lab_module.Bot(
        broker,
        {
            "signal": "range_fade_limit",
            "pairs": pairs,
            "tp_pips": 5,
            "ceiling_min": 60,
            "max_concurrent": 2,
            "global_max_concurrent": 4,
            "strategy_owner_id": "test:global-cap",
        },
    )
    order_ids = [
        hand.broker.limit_order(pair, "LONG", 1_000, price=ask + 0.01)
        for pair, _, ask, _ in initial_quotes
    ]

    events = broker.on_quote_batch(
        [
            (pair, bid, ask, "2026-01-01T00:01:00+00:00#O")
            for pair, bid, ask, _ in initial_quotes
        ]
    )

    assert len(broker.positions) == 4
    assert not broker.orders
    assert events == []
    records = [json.loads(line) for line in broker.ledger_path.read_text().splitlines()]
    assert sum(record["event"] == "FILL_LIMIT" for record in records) == 4
    rejected = [
        record["payload"]
        for record in records
        if record["event"] == "ORDER_CANCEL_CONCURRENCY_CAP"
    ]
    assert len(rejected) == 1
    # Each order is marketable at submission and must resolve atomically at
    # that already-staged quote; the fifth submission sees the four live fills.
    assert rejected[0]["order_id"] == order_ids[-1]
    assert rejected[0]["admission"] == {
        "scope": "GLOBAL",
        "reason": "OWNER_GLOBAL_CONCURRENCY_CAP_REACHED",
        "active_pair_positions": 0,
        "max_concurrent_per_pair": 2,
        "active_global_positions": 4,
        "global_max_concurrent": 4,
    }


def test_concurrency_caps_are_owner_isolated_and_market_fills_reject_at_cap(
    tmp_path: Path,
) -> None:
    broker = VirtualBroker(
        tmp_path / "ledger.jsonl", balance_jpy=2_000_000.0, fast_ledger=True
    )
    broker.on_quote("USD_JPY", 150.00, 150.02, "2026-01-01T00:00:00+00:00")
    owner_a = OwnedBrokerView(
        broker,
        "test:owner-a",
        max_concurrent_per_pair=1,
        global_max_concurrent=1,
    )
    owner_b = OwnedBrokerView(
        broker,
        "test:owner-b",
        max_concurrent_per_pair=1,
        global_max_concurrent=1,
    )

    owner_a.market_order("USD_JPY", "LONG", 1_000)
    owner_b.market_order("USD_JPY", "SHORT", 1_000)
    with pytest.raises(VirtualBrokerError, match="concurrency cap"):
        owner_a.market_order("USD_JPY", "SHORT", 1_000)

    assert len(owner_a.active_trade_ids()) == 1
    assert len(owner_b.active_trade_ids()) == 1
    rejection = next(
        json.loads(line)
        for line in reversed(broker.ledger_path.read_text().splitlines())
        if json.loads(line)["event"] == "ORDER_REJECTED_CONCURRENCY_CAP"
    )
    assert rejection["payload"]["strategy_owner_id"] == "test:owner-a"
    assert rejection["payload"]["admission"]["scope"] == "PAIR"
