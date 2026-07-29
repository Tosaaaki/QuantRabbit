from __future__ import annotations

import fcntl
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from quant_rabbit.crypto.ledger import CryptoLedger
from quant_rabbit.crypto.outbox import AsyncTradeOutbox, TRADE_COLUMNS
from quant_rabbit.crypto.paper import PaperEngine
from quant_rabbit.crypto.reporting import (
    IroriSlackSummarySink,
    PaperShadowReportingWriter,
)
from quant_rabbit.crypto.shadow import (
    PaperShadowAlreadyRunning,
    PaperShadowService,
    PaperShadowServiceConfig,
)


def _round_trip(
    ledger: CryptoLedger,
    *,
    sink: Any = None,
    allow_short: bool = False,
) -> None:
    engine = PaperEngine(
        ledger,
        maker_fill_fraction=Decimal("1"),
        allow_short=allow_short,
        max_leverage=Decimal("2") if allow_short else Decimal("1"),
        trade_sink=sink,
    )
    engine.process_intent(
        {
            "intent_id": "open-1",
            "run_id": "run-1",
            "pair": "btc_jpy",
            "side": "BUY",
            "position_effect": "OPEN",
            "amount": "1",
            "order_style": "PAPER_TAKER",
            "event_at_utc": "2026-07-28T00:00:00+00:00",
            "regime": "FAST",
            "signal_reason": "ENTRY",
            "authority": "NONE",
            "live_permission": False,
        },
        depth={"asks": [["100", "1"]], "bids": [["99", "1"]]},
        maker_fee_rate=Decimal("0"),
        taker_fee_rate=Decimal("0.001"),
    )
    engine.process_intent(
        {
            "intent_id": "close-1",
            "run_id": "run-1",
            "pair": "btc_jpy",
            "side": "SELL",
            "position_effect": "CLOSE",
            "amount": "1",
            "order_style": "PAPER_TAKER",
            "event_at_utc": "2026-07-28T00:00:01+00:00",
            "regime": "FAST",
            "signal_reason": "TAKE_PROFIT",
            "authority": "NONE",
            "live_permission": False,
        },
        depth={"asks": [["102", "1"]], "bids": [["101", "1"]]},
        maker_fee_rate=Decimal("0"),
        taker_fee_rate=Decimal("0.001"),
    )


def test_trade_close_writes_one_complete_outbox_row(tmp_path: Path) -> None:
    ledger = CryptoLedger(tmp_path / "spot" / "ledger.db")
    outbox = AsyncTradeOutbox(
        tmp_path / "spot" / "trade_outbox.jsonl", ledger
    )
    _round_trip(ledger, sink=outbox.enqueue)
    assert outbox.flush()
    rows = [
        json.loads(line)
        for line in outbox.path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert all(column in row for column in TRADE_COLUMNS)
    assert row["run_id"] == "run-1"
    assert row["paper_mode"] == "SPOT"
    assert row["side"] == "LONG"
    assert row["holding_ms"] == 1000
    assert row["exit_reason"] == "TAKE_PROFIT"
    assert row["authority"] == "NONE"
    assert row["live_permission"] is False
    assert row["ledger_event_hash"]
    assert ledger.verify()["valid"] is True
    assert outbox.close()


def test_outbox_recovers_from_ledger_without_duplicate(
    tmp_path: Path,
) -> None:
    ledger = CryptoLedger(tmp_path / "margin" / "ledger.db")
    _round_trip(ledger, allow_short=True)
    path = tmp_path / "margin" / "trade_outbox.jsonl"
    first = AsyncTradeOutbox(path, ledger)
    assert first.flush()
    assert first.close()
    second = AsyncTradeOutbox(path, ledger)
    assert second.flush()
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
    assert second.close()


def test_outbox_restart_recovers_only_ledger_suffix(tmp_path: Path) -> None:
    ledger = CryptoLedger(tmp_path / "margin" / "ledger.db")
    _round_trip(ledger, allow_short=True)
    path = tmp_path / "margin" / "trade_outbox.jsonl"
    first = AsyncTradeOutbox(path, ledger)
    assert first.close()
    ledger.events = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("outbox restart must not rescan the full ledger")
    )

    second = AsyncTradeOutbox(path, ledger)

    assert second.status()["known_operations"] == 1
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
    assert second.close()


class FakeSheets:
    def __init__(self) -> None:
        self.trades: dict[str, dict[str, Any]] = {}
        self.summaries: dict[str, dict[str, Any]] = {}
        self.fail_once = False
        self.trade_append_calls = 0

    def append_trade(
        self, operation_id: str, row: dict[str, Any]
    ) -> None:
        self.trade_append_calls += 1
        if self.fail_once:
            self.fail_once = False
            raise OSError("temporary sheets failure")
        self.trades.setdefault(operation_id, row)

    def append_summary(
        self, operation_id: str, row: dict[str, Any]
    ) -> None:
        self.summaries.setdefault(operation_id, row)

    def readback(self, operation_id: str) -> bool:
        return operation_id in self.trades or operation_id in self.summaries


class FakeSlack:
    def __init__(self) -> None:
        self.posts: dict[str, str] = {}

    def post_summary(
        self, operation_id: str, row: dict[str, Any]
    ) -> str:
        assert row["per_trade_slack_posts"] is False
        return self.posts.setdefault(
            operation_id, f"https://slack.invalid/{operation_id}"
        )

    def readback(self, operation_id: str, permalink: str) -> bool:
        return self.posts.get(operation_id) == permalink


def test_reporting_writer_retries_and_deduplicates_targets(
    tmp_path: Path,
) -> None:
    ledger = CryptoLedger(tmp_path / "spot" / "ledger.db")
    outbox = AsyncTradeOutbox(
        tmp_path / "spot" / "trade_outbox.jsonl", ledger
    )
    _round_trip(ledger, sink=outbox.enqueue)
    assert outbox.flush()
    assert outbox.close()
    sheets = FakeSheets()
    slack = FakeSlack()
    sheets.fail_once = True
    writer = PaperShadowReportingWriter(
        tmp_path, sheets=sheets, slack=slack
    )
    now = datetime(2026, 7, 28, 0, 30, tzinfo=timezone.utc)
    failed = writer.run_once(now)
    assert failed["sheets_trade_pending"] == 1
    retried = writer.run_once(now)
    assert retried["sheets_trade_delivered"] == 1
    again = writer.run_once(now)
    assert again["sheets_trade_delivered"] == 0
    assert len(sheets.trades) == 1
    assert len(sheets.summaries) == 2
    assert len(slack.posts) == 2
    assert len(
        (tmp_path / "summary_outbox.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 2
    assert {
        row["period_key"] for row in sheets.summaries.values()
    } == {"2026-07-27T23:00:00Z", "2026-07-27"}


def test_reporting_writer_readback_prevents_restart_duplicate(
    tmp_path: Path,
) -> None:
    ledger = CryptoLedger(tmp_path / "spot" / "ledger.db")
    outbox = AsyncTradeOutbox(
        tmp_path / "spot" / "trade_outbox.jsonl", ledger
    )
    _round_trip(ledger, sink=outbox.enqueue)
    assert outbox.close()
    row = json.loads(outbox.path.read_text(encoding="utf-8"))
    sheets = FakeSheets()
    sheets.trades[row["operation_id"]] = row

    result = PaperShadowReportingWriter(
        tmp_path, sheets=sheets
    ).run_once(datetime(2026, 7, 28, 0, 30, tzinfo=timezone.utc))

    assert result["sheets_trade_delivered"] == 1
    assert sheets.trade_append_calls == 0


def test_reporting_writer_retries_durable_summary_backlog(
    tmp_path: Path,
) -> None:
    first = datetime(2026, 7, 28, 0, 30, tzinfo=timezone.utc)
    pending = PaperShadowReportingWriter(tmp_path).run_once(first)
    assert pending["slack_summary_pending"] == 2

    slack = FakeSlack()
    second = datetime(2026, 7, 28, 1, 30, tzinfo=timezone.utc)
    delivered = PaperShadowReportingWriter(
        tmp_path, slack=slack
    ).run_once(second)

    assert delivered["local_summary_added"] == 1
    assert delivered["slack_summary_pending"] == 0
    assert len(slack.posts) == 3


def test_irori_sink_requires_verified_existing_thread_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helper = tmp_path / "post_slack.sh"
    helper.write_text("#!/bin/bash\n", encoding="utf-8")
    operation_id = "crypto-paper-summary:2026-07-28T00"
    parent_ts = "1785197873.432059"
    receipt = {
        "ok": True,
        "verified": True,
        "identity_team_id": "T_APPROVED",
        "identity_user_id": "U_IRORI",
        "channel": "C0BDKFTGBQB",
        "channel_name": "quant-rabbit",
        "parent_ts": parent_ts,
        "reply_ts": "1785197999.000001",
        "permalink": (
            "https://irori-hub.slack.com/archives/"
            "C0BDKFTGBQB/p1785197873432059"
        ),
        "operation_id": operation_id,
        "reply_only": True,
    }
    captured: dict[str, Any] = {}

    def fake_run(*args: Any, **kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(receipt),
            stderr="",
        )

    monkeypatch.setattr(
        "quant_rabbit.crypto.reporting.subprocess.run", fake_run
    )
    sink = IroriSlackSummarySink(
        helper_path=helper,
        route_ref="task:quant-rabbit",
        parent_ts=parent_ts,
    )
    permalink = sink.post_summary(
        operation_id,
        {
            "period": "hour",
            "period_key": "2026-07-28T00:00:00Z",
        },
    )

    assert captured["env"]["IRORI_REPORT_OPERATION_ID"] == operation_id
    assert captured["env"]["IRORI_REPORT_PARENT_TS"] == parent_ts
    assert operation_id in captured["input"]
    assert sink.readback(operation_id, permalink)


def test_partial_closes_emit_one_weighted_completed_trade(
    tmp_path: Path,
) -> None:
    ledger = CryptoLedger(tmp_path / "spot" / "ledger.db")
    outbox = AsyncTradeOutbox(
        tmp_path / "spot" / "trade_outbox.jsonl", ledger
    )
    engine = PaperEngine(
        ledger,
        maker_fill_fraction=Decimal("1"),
        trade_sink=outbox.enqueue,
    )
    common = {
        "run_id": "partial-run",
        "pair": "btc_jpy",
        "order_style": "PAPER_TAKER",
        "regime": "FAST",
        "authority": "NONE",
        "live_permission": False,
    }
    engine.process_intent(
        {
            **common,
            "intent_id": "partial-open",
            "side": "BUY",
            "position_effect": "OPEN",
            "amount": "1",
            "event_at_utc": "2026-07-28T00:00:00+00:00",
        },
        depth={"asks": [["100", "1"]], "bids": [["99", "1"]]},
        maker_fee_rate=Decimal("0"),
        taker_fee_rate=Decimal("0"),
    )
    for index, bid in enumerate(("99", "101"), 1):
        engine.process_intent(
            {
                **common,
                "intent_id": f"partial-close-{index}",
                "side": "SELL",
                "position_effect": "CLOSE",
                "amount": "0.5",
                "event_at_utc": (
                    f"2026-07-28T00:00:0{index}+00:00"
                ),
                "signal_reason": "PARTIAL_EXIT",
            },
            depth={"asks": [["102", "1"]], "bids": [[bid, "0.5"]]},
            maker_fee_rate=Decimal("0"),
            taker_fee_rate=Decimal("0"),
        )
    assert outbox.close()
    rows = [
        json.loads(line)
        for line in outbox.path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["quantity"] == "1.0"
    assert rows[0]["exit_price"] == "100"
    assert rows[0]["holding_ms"] == 2000


def test_shadow_service_rejects_duplicate_mode_process(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "spot"
    runtime.mkdir()
    lock = (runtime / "service.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    service = PaperShadowService(
        PaperShadowServiceConfig(mode="spot", runtime_dir=runtime),
        pairs=["btc_jpy"],
        pair_fees={"btc_jpy": (Decimal("0"), Decimal("0"))},
        daily_interest_rates={},
    )
    with pytest.raises(PaperShadowAlreadyRunning):
        service.run()
    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    lock.close()


def test_shadow_restart_verifies_only_suffix_from_last_epoch(
    tmp_path: Path,
) -> None:
    ledger = CryptoLedger(tmp_path / "ledger.db")
    ledger.append("TEST", "one", {"n": 1}, dedupe_key="one")
    checkpoint = ledger.verify()
    latest_epoch = tmp_path / "latest_epoch.json"
    latest_epoch.write_text(
        json.dumps({"ledger_integrity": checkpoint}),
        encoding="utf-8",
    )
    ledger.append("TEST", "two", {"n": 2}, dedupe_key="two")
    reopened = CryptoLedger(tmp_path / "ledger.db", verify_on_open=False)
    result = PaperShadowService._verify_ledger_for_restart(
        reopened,
        latest_epoch,
    )
    assert result["valid"] is True
    assert result["event_count"] == 2
    assert result["head_hash"] == ledger.verify()["head_hash"]
