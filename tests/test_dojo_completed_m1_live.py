from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from quant_rabbit.dojo_completed_m1_live import (
    COMPLETED_M1_SOURCE,
    COMPLETED_M1_SOURCE_ERROR_CONTRACT,
    CompletedM1EvidenceError,
    completed_m1_bars,
    cutoff_payload,
    fetch_completed_m1_fail_closed,
    quote_response_cutoff,
    restore_consumed_bars,
    seed_completed_m1_history,
)
from quant_rabbit.virtual_broker import VirtualBroker


ROOT = Path(__file__).resolve().parents[1]
UTC = timezone.utc


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _candle(
    timestamp: str,
    *,
    bid: tuple[str, str, str, str] = (
        "1.10000",
        "1.10010",
        "1.09990",
        "1.10005",
    ),
    ask: tuple[str, str, str, str] = (
        "1.10008",
        "1.10018",
        "1.09998",
        "1.10013",
    ),
    complete: bool = True,
) -> dict:
    def block(values: tuple[str, str, str, str]) -> dict[str, str]:
        return dict(zip(("o", "h", "l", "c"), values, strict=True))

    return {
        "time": timestamp,
        "complete": complete,
        "volume": 17,
        "bid": block(bid),
        "ask": block(ask),
    }


def test_completed_m1_accepts_only_unseen_causal_complete_bid_ask_rows():
    payload = {
        "candles": [
            _candle("2026-07-23T01:00:00.000000000Z"),
            _candle("2026-07-23T01:01:00.000000000Z"),
            _candle("2026-07-23T01:02:00.000000000Z", complete=False),
            _candle("2026-07-23T01:03:00.000000000Z"),
        ]
    }

    rows = completed_m1_bars(
        payload,
        pair="EUR_USD",
        cutoff_utc="2026-07-23T01:02:30+00:00",
        after_epoch=int(
            datetime(2026, 7, 23, 1, 0, tzinfo=UTC).timestamp()
        ),
    )

    assert [row["source"]["start_utc"] for row in rows] == [
        "2026-07-23T01:01:00+00:00"
    ]
    assert rows[0]["source"]["pair"] == "EUR_USD"
    assert rows[0]["source"]["granularity"] == "M1"
    assert rows[0]["source"]["complete"] is True
    assert len(rows[0]["source_sha256"]) == 64


def test_completed_m1_refuses_conflicting_duplicate_or_crossed_bid_ask():
    first = _candle("2026-07-23T01:01:00Z")
    conflict = _candle(
        "2026-07-23T01:01:00Z",
        bid=("1.10000", "1.10015", "1.09990", "1.10007"),
    )
    with pytest.raises(CompletedM1EvidenceError, match="conflicting"):
        completed_m1_bars(
            {"candles": [first, conflict]},
            pair="EUR_USD",
            cutoff_utc="2026-07-23T01:02:30Z",
            after_epoch=0,
        )

    crossed = _candle(
        "2026-07-23T01:01:00Z",
        ask=("1.09990", "1.10000", "1.09980", "1.09995"),
    )
    with pytest.raises(CompletedM1EvidenceError, match="ask is below bid"):
        completed_m1_bars(
            {"candles": [crossed]},
            pair="EUR_USD",
            cutoff_utc="2026-07-23T01:02:30Z",
            after_epoch=0,
        )


def test_cutoff_is_paper_only_and_refuses_future_quote():
    row = completed_m1_bars(
        {"candles": [_candle("2026-07-23T01:01:00Z")]},
        pair="EUR_USD",
        cutoff_utc="2026-07-23T01:02:30Z",
        after_epoch=0,
    )[0]

    sealed = cutoff_payload(
        pair="EUR_USD",
        bar=row,
        cutoff_utc="2026-07-23T01:02:30Z",
        quote_timestamp_utc="2026-07-23T01:02:29Z",
        decision_mode="ACTION",
    )

    assert sealed["paper_only"] is True
    assert sealed["order_authority"] == "NONE"
    assert sealed["live_permission"] is False
    assert sealed["bar_source_sha256"] == row["source_sha256"]
    assert len(sealed["cutoff_sha256"]) == 64

    with pytest.raises(CompletedM1EvidenceError, match="after cutoff"):
        cutoff_payload(
            pair="EUR_USD",
            bar=row,
            cutoff_utc="2026-07-23T01:02:30Z",
            quote_timestamp_utc="2026-07-23T01:02:31Z",
            decision_mode="ACTION",
        )


def test_quote_response_cutoff_is_post_acquisition_and_refuses_clock_skew():
    cutoff = quote_response_cutoff(
        acquired_at_utc="2026-07-23T01:02:31Z",
        quote_timestamp_utc="2026-07-23T01:02:30Z",
    )

    assert cutoff == datetime(2026, 7, 23, 1, 2, 31, tzinfo=UTC)

    with pytest.raises(
        CompletedM1EvidenceError,
        match="after acquisition cutoff",
    ):
        quote_response_cutoff(
            acquired_at_utc="2026-07-23T01:02:29Z",
            quote_timestamp_utc="2026-07-23T01:02:30Z",
        )


class _SeedRecorder:
    def __init__(self) -> None:
        self.rows: list[tuple[str, dict]] = []

    def seed_bar(self, pair: str, bar: dict) -> None:
        self.rows.append((pair, bar))


def _write_seed(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def test_seed_history_is_fixed_to_window_and_content_addressed(
    tmp_path: Path,
):
    window_start = datetime(2026, 7, 23, 1, 0, tzinfo=UTC)
    first_epoch = int(window_start.timestamp()) - 1441 * 60
    rows = [
        {
            **_candle(
                datetime.fromtimestamp(
                    first_epoch + index * 60,
                    UTC,
                ).isoformat(),
            ),
            "pair": "EUR_USD",
            "granularity": "M1",
            "price": "BA",
        }
        for index in range(1441)
    ]
    source = (
        tmp_path
        / "seed"
        / "fixed"
        / "EUR_USD"
        / "EUR_USD_M1_BA_fixed.jsonl.gz"
    )
    _write_seed(source, rows)
    bot = _SeedRecorder()

    manifest = seed_completed_m1_history(
        seed_root=tmp_path / "seed",
        bot=bot,
        pairs=["EUR_USD"],
        window_start_utc=window_start,
        seed_hours=1441 / 60,
    )

    assert manifest["pair_counts"] == {"EUR_USD": 1441}
    assert manifest["pair_last_bar_epochs"] == {
        "EUR_USD": int(window_start.timestamp()) - 60
    }
    assert manifest["pair_last_bar_end_utc"] == {
        "EUR_USD": window_start.isoformat()
    }
    assert manifest["files"][0]["sha256"]
    assert manifest["window_start_utc"] == window_start.isoformat()
    assert manifest["order_authority"] == "NONE"
    assert len(manifest["seed_manifest_sha256"]) == 64
    assert len(bot.rows) == 1441

    rows.append(
        {
            **_candle(window_start.isoformat()),
            "pair": "EUR_USD",
            "granularity": "M1",
            "price": "BA",
        }
    )
    _write_seed(source, rows)
    with pytest.raises(
        CompletedM1EvidenceError,
        match="post-cutoff evidence",
    ):
        seed_completed_m1_history(
            seed_root=tmp_path / "seed",
            bot=_SeedRecorder(),
            pairs=["EUR_USD"],
            window_start_utc=window_start,
            seed_hours=1441 / 60,
        )


def _ledger_cutoff(sealed: dict) -> str:
    return json.dumps(
        {"event": "BOT_M1_CUTOFF", "payload": {"payload": sealed}},
        sort_keys=True,
    )


def test_restore_replays_cutoffs_as_seed_only_and_fails_closed_on_gap(
    tmp_path: Path,
):
    start = datetime(2026, 7, 23, 1, 0, tzinfo=UTC)
    rows = completed_m1_bars(
        {
            "candles": [
                _candle("2026-07-23T01:00:00Z"),
                _candle("2026-07-23T01:01:00Z"),
            ]
        },
        pair="EUR_USD",
        cutoff_utc="2026-07-23T01:02:30Z",
        after_epoch=int(start.timestamp()) - 60,
    )
    sealed = [
        cutoff_payload(
            pair="EUR_USD",
            bar=row,
            cutoff_utc="2026-07-23T01:02:30Z",
            quote_timestamp_utc="2026-07-23T01:02:29Z",
            decision_mode="ACTION",
        )
        for row in rows
    ]
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        "\n".join(_ledger_cutoff(row) for row in sealed) + "\n",
        encoding="utf-8",
    )
    bot = _SeedRecorder()

    cursors = restore_consumed_bars(
        ledger,
        bot=bot,
        pairs=["EUR_USD"],
        initial_epoch=int(start.timestamp()) - 60,
    )

    assert cursors == {"EUR_USD": int(start.timestamp()) + 60}
    assert [bar["epoch"] for _, bar in bot.rows] == [
        int(start.timestamp()),
        int(start.timestamp()) + 60,
    ]

    ledger.write_text(_ledger_cutoff(sealed[1]) + "\n", encoding="utf-8")
    with pytest.raises(CompletedM1EvidenceError, match="gap"):
        restore_consumed_bars(
            ledger,
            bot=_SeedRecorder(),
            pairs=["EUR_USD"],
            initial_epoch=int(start.timestamp()) - 60,
        )


def test_restore_uses_exact_per_pair_seed_cursor(tmp_path: Path):
    ledger = tmp_path / "absent-ledger.jsonl"
    cursors = restore_consumed_bars(
        ledger,
        bot=_SeedRecorder(),
        pairs=["EUR_USD", "USD_JPY"],
        initial_epochs={
            "EUR_USD": 1_700_000_000,
            "USD_JPY": 1_700_000_060,
        },
    )

    assert cursors == {
        "EUR_USD": 1_700_000_000,
        "USD_JPY": 1_700_000_060,
    }

    with pytest.raises(
        CompletedM1EvidenceError,
        match="exactly cover bot pairs",
    ):
        restore_consumed_bars(
            ledger,
            bot=_SeedRecorder(),
            pairs=["EUR_USD", "USD_JPY"],
            initial_epochs={"EUR_USD": 1_700_000_000},
        )


def test_observed_bot_records_exact_no_entry_reason(tmp_path: Path):
    observed = _load(
        ROOT / "bots/lab_bot_observed.py",
        "dojo_observed_bot_test",
    )
    broker = VirtualBroker(tmp_path / "ledger.jsonl", balance_jpy=200_000.0)
    bot = observed.Bot(
        broker,
        {
            "pairs": ["EUR_USD"],
            "strategy_tag": "W_FADE_EURUSD_DIAGNOSTIC",
            "signal": "range_fade_limit",
            "tp_pips": 8.0,
            "sl_pips": None,
            "ceiling_min": 480,
            "max_concurrent": 2,
            "per_pos_lev": 2.0,
            "atr_floor_pips": 1.0,
            "fade_atr": 1.2,
            "eff_max": 0.2,
        },
    )
    start_epoch = int(
        datetime(2026, 7, 22, 0, 0, tzinfo=UTC).timestamp()
    )
    quiet = {
        "bid_o": 1.10000,
        "bid_h": 1.10001,
        "bid_l": 1.09999,
        "bid_c": 1.10000,
        "ask_o": 1.10008,
        "ask_h": 1.10009,
        "ask_l": 1.10007,
        "ask_c": 1.10008,
    }
    for index in range(1441):
        epoch = start_epoch + index * 60
        bot.seed_bar("EUR_USD", {**quiet, "epoch": epoch})
    decision_epoch = start_epoch + 1441 * 60

    bot.on_bar_closed(
        "EUR_USD",
        {**quiet, "epoch": decision_epoch},
        decision_epoch,
    )

    decisions = [
        row
        for row in map(
            json.loads,
            broker.ledger_path.read_text(encoding="utf-8").splitlines(),
        )
        if row["event"] == "BOT_DECISION"
    ]
    assert len(decisions) == 1
    payload = decisions[0]["payload"]
    assert payload["result"]["reason"] == "ATR_BELOW_FLOOR"
    assert payload["result"]["supervision"] == "STOP"
    assert payload["inputs"]["atr_pips"] < 1.0
    assert payload["order_authority"] == "NONE"
    assert len(payload["decision_sha256"]) == 64
    assert not broker.positions
    assert not broker.orders


def test_completed_m1_source_error_cancels_stale_orders_and_retries_cursor(
    tmp_path: Path,
):
    class BrokenThenRecoveredClient:
        calls = 0

        def get_json(self, _path: str, _query: dict[str, str]) -> dict:
            self.calls += 1
            if self.calls <= 2:
                raise TimeoutError("bounded read timeout")
            return {"candles": []}

    class BotStub:
        strategy_tag = "W_FADE_EURUSD_COMPLETED_M1_DIAGNOSTIC"

    broker = VirtualBroker(tmp_path / "ledger.jsonl", balance_jpy=200_000.0)
    broker.limit_order(
        "EUR_USD",
        "LONG",
        1_000,
        1.1,
        strategy_tag=BotStub.strategy_tag,
    )
    cursor = int(datetime(2026, 7, 23, 6, 0, tzinfo=UTC).timestamp())
    fingerprints: dict[str, str] = {}
    client = BrokenThenRecoveredClient()

    first = fetch_completed_m1_fail_closed(
        client,
        broker=broker,
        bot=BotStub(),
        pair="EUR_USD",
        after_epoch=cursor,
        cutoff=datetime(2026, 7, 23, 6, 2, tzinfo=UTC),
        error_fingerprints=fingerprints,
    )
    second = fetch_completed_m1_fail_closed(
        client,
        broker=broker,
        bot=BotStub(),
        pair="EUR_USD",
        after_epoch=cursor,
        cutoff=datetime(2026, 7, 23, 6, 2, 5, tzinfo=UTC),
        error_fingerprints=fingerprints,
    )
    recovered = fetch_completed_m1_fail_closed(
        client,
        broker=broker,
        bot=BotStub(),
        pair="EUR_USD",
        after_epoch=cursor,
        cutoff=datetime(2026, 7, 23, 6, 2, 10, tzinfo=UTC),
        error_fingerprints=fingerprints,
    )

    assert first is None
    assert second is None
    assert recovered == []
    assert not broker.orders
    rows = [
        json.loads(line)
        for line in broker.ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    source_errors = [
        row for row in rows if row["event"] == "BOT_M1_SOURCE_ERROR"
    ]
    assert len(source_errors) == 1
    payload = source_errors[0]["payload"]["payload"]
    assert payload["contract"] == COMPLETED_M1_SOURCE_ERROR_CONTRACT
    assert payload["retry_same_cursor"] is True
    assert payload["new_entries_allowed"] is False
    assert payload["order_authority"] == "NONE"
    assert len(payload["source_error_sha256"]) == 64
    assert sum(
        row["event"] == "BOT_M1_SOURCE_ORDER_CANCEL" for row in rows
    ) == 1
    assert sum(row["event"] == "BOT_M1_SOURCE_RECOVERED" for row in rows) == 1
    assert fingerprints == {}


def test_completed_m1_launcher_is_diagnostic_only_and_binds_new_runtime(
    tmp_path: Path,
):
    launcher = _load(
        ROOT / "scripts/run-dojo-paper-room-completed-m1.py",
        "dojo_completed_m1_launcher_test",
    )
    source = json.loads(
        (
            ROOT
            / "config/dojo_paper_rooms_eurusd_diagnostic_20260722_v1.json"
        ).read_text(encoding="utf-8")
    )
    source["defaults"]["live_bot_bar_source"] = COMPLETED_M1_SOURCE
    source["defaults"]["bot_module"] = "bots/lab_bot_observed.py:Bot"
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(source), encoding="utf-8")

    command, env, _session_dir = launcher.build_launch(
        registry_path=registry,
        room_id=source["rooms"][0]["room_id"],
        python_executable="/fixed/python3",
    )

    assert command[1] == str(
        ROOT / "scripts/run-virtual-market-session-completed-m1.py"
    )
    assert command[command.index("--paper-proof-mode") + 1] == "diagnostic"
    assert env["QR_DOJO_LIVE_BOT_BAR_SOURCE"] == COMPLETED_M1_SOURCE
    dependencies = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--runtime-dependency"
    ]
    assert str(ROOT / "src/quant_rabbit/dojo_completed_m1_live.py") in (
        dependencies
    )
    assert str(ROOT / "bots/lab_bot.py") in dependencies

    source["proof_mode"] = "formal"
    registry.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="diagnostic-only"):
        launcher.build_launch(
            registry_path=registry,
            room_id=source["rooms"][0]["room_id"],
            python_executable="/fixed/python3",
        )
