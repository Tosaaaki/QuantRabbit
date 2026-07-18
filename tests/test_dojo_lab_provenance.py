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
from quant_rabbit.virtual_broker import VirtualBroker


BOT_PATH = Path("/tmp/dojo-lab-bot.py")
BOT_SHA = "c" * 64
CONFIG_TEXT = '{"signal":"test"}'
CONFIG_SHA = hashlib.sha256(CONFIG_TEXT.encode()).hexdigest()


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
            "dependency_sha256": {},
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
        *events,
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
        "expected_slippage_pips": 0.3,
        "expected_financing_pips_per_day": 0.8,
        "expected_bot_module_path": BOT_PATH,
        "expected_bot_module_sha256": BOT_SHA,
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
        "VAL": ("2025-01-01T00:00:00", "2026-01-01T00:00:00"),
        "FINAL": ("2026-02-01T00:00:00", "2026-03-01T00:00:00"),
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


def test_positive_score_requires_reauthenticated_durable_reservation(
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
    assert score["local_candidate_eligible"] is True
    assert score["promotion_eligible"] is False
    assert (
        "EXTERNAL_MONOTONIC_RESERVATION_ATTESTATION_ABSENT"
        in score["promotion_blockers"]
    )
    assert score["reservation_status"] == "RESERVED"


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
