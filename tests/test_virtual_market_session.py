import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from quant_rabbit.dojo_lab_provenance import OwnedBrokerView
from quant_rabbit.virtual_broker import VirtualBroker, VirtualBrokerError


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "run-virtual-market-session.py"
)
SPEC = importlib.util.spec_from_file_location("virtual_market_session", SCRIPT_PATH)
SESSION = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SESSION)


def _write_bar(root: Path, pair: str, stamp: str, offset: float) -> None:
    shard = root / "fixture" / pair / f"{pair}_M1_BA_2026.jsonl.gz"
    shard.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "time": stamp,
        "bid": {"o": offset + 1, "h": offset + 4, "l": offset, "c": offset + 2},
        "ask": {"o": offset + 11, "h": offset + 14, "l": offset + 10, "c": offset + 12},
    }
    with gzip.open(shard, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _write_bars(root: Path, pair: str, rows: list[tuple[str, float]]) -> None:
    shard = root / "fixture" / pair / f"{pair}_M1_BA_2026.jsonl.gz"
    shard.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(shard, "wt", encoding="utf-8") as handle:
        for stamp, offset in rows:
            row = {
                "time": stamp,
                "bid": {
                    "o": offset + 1,
                    "h": offset + 4,
                    "l": offset,
                    "c": offset + 2,
                },
                "ask": {
                    "o": offset + 1.02,
                    "h": offset + 4.02,
                    "l": offset + 0.02,
                    "c": offset + 2.02,
                },
            }
            handle.write(json.dumps(row) + "\n")


def _manifest_args(corpus: Path, **overrides):
    values = {
        "feed": "replay",
        "pairs": "USD_JPY",
        "balance": 200_000.0,
        "corpus_root": str(corpus),
        "time_from": "2026-01-01T00:00:00",
        "time_to": "2026-01-01T00:02:00",
        "granularity": "M1",
        "intrabar": "OHLC",
        "bot": None,
        "bot_module": None,
        "strategy_owner_id": None,
        "bot_dependency": [],
        "bot_bar": "feed",
        "settle_at_end": False,
        "continuous_mtm": False,
        "slippage_pips": 0.0,
        "financing_pips_day": 0.0,
        "session_dir": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_replay_orders_epoch_then_phase_then_pair_and_is_pair_order_invariant(tmp_path):
    root = tmp_path / "corpus"
    _write_bar(root, "USD_JPY", "2026-01-01T00:00:00.000000000Z", 100)
    _write_bar(root, "EUR_USD", "2026-01-01T00:00:00.000000000Z", 200)

    forward = list(
        SESSION._iter_replay_quotes(
            root,
            ["USD_JPY", "EUR_USD"],
            "2026-01-01T00:00:00",
            "2026-01-01T00:01:00",
        )
    )
    reverse = list(
        SESSION._iter_replay_quotes(
            root,
            ["EUR_USD", "USD_JPY"],
            "2026-01-01T00:00:00",
            "2026-01-01T00:01:00",
        )
    )

    assert forward == reverse
    assert [(pair, phase) for _, pair, _, _, phase in forward] == [
        ("EUR_USD", "O"),
        ("USD_JPY", "O"),
        ("EUR_USD", "H"),
        ("USD_JPY", "H"),
        ("EUR_USD", "L"),
        ("USD_JPY", "L"),
        ("EUR_USD", "C"),
        ("USD_JPY", "C"),
    ]


def test_reproducibility_manifest_binds_sources_costs_and_bot(tmp_path):
    corpus = tmp_path / "corpus"
    _write_bar(corpus, "USD_JPY", "2026-01-01T00:00:00.000000000Z", 100)
    bot_module = tmp_path / "bot.py"
    bot_module.write_text("class Worker:\n    pass\n", encoding="utf-8")
    args = argparse.Namespace(
        feed="replay",
        pairs="USD_JPY",
        balance=200_000.0,
        corpus_root=str(corpus),
        time_from="2026-01-01T00:00:00",
        time_to="2026-01-01T00:01:00",
        granularity="M1",
        intrabar="OHLC",
        bot=None,
        bot_module=f"{bot_module}:Worker",
        strategy_owner_id="test:worker",
        bot_dependency=["src/quant_rabbit/virtual_broker.py"],
        bot_bar="feed",
        settle_at_end=False,
        continuous_mtm=True,
        slippage_pips=0.7,
        financing_pips_day=0.2,
    )

    manifest = SESSION._build_reproducibility_manifest(args)

    assert manifest["schema"] == "QR_VIRTUAL_SESSION_REPRODUCIBILITY_V1"
    assert manifest["replay"]["pairs"] == ["USD_JPY"]
    assert manifest["replay"]["intrabar"] == "OHLC"
    assert manifest["replay"]["period_end_settlement"] is False
    assert manifest["replay"]["continuous_mtm"] is True
    assert manifest["costs"] == {
        "slippage_pips_per_fill": 0.7,
        "financing_pips_per_day": 0.2,
        "leverage": 25.0,
    }
    assert manifest["bot"]["class"] == "Worker"
    assert manifest["bot"]["strategy_owner_id"] == "test:worker"
    assert set(manifest["bot"]["dependency_sha256"]) == {
        "src/quant_rabbit/virtual_broker.py"
    }
    assert len(manifest["bot"]["module_sha256"]) == 64
    assert len(manifest["corpus"]["shards"]) == 1
    assert len(manifest["corpus"]["shards"][0]["sha256"]) == 64
    assert len(manifest["source"]["git_head"]) == 40
    assert len(manifest["manifest_sha256"]) == 64
    assert (
        manifest["replay"]["mtm_coordinate_contract"] == "QR_REPLAY_MTM_COORDINATES_V1"
    )
    assert manifest["replay"]["phase_order"] == ["O", "H", "L", "C"]
    assert manifest["replay"]["feed_pairs"] == ["USD_JPY"]
    assert manifest["replay"]["expected_phase_mark_count"] == 4
    assert manifest["replay"]["full_pair_phase_coverage"] is True
    assert manifest["replay"]["partial_phase_count"] == 0
    assert len(manifest["replay"]["expected_batch_chain_terminal_sha256"]) == 64

    body = dict(manifest)
    digest = body.pop("manifest_sha256")
    assert digest == SESSION._canonical_sha256(body)


def test_manifest_precommits_full_two_pair_mtm_coordinate_and_batch_chain(tmp_path):
    corpus = tmp_path / "corpus"
    rows = [
        ("2026-01-01T00:00:00.000000000Z", 100.0),
        ("2026-01-01T00:01:00.000000000Z", 110.0),
    ]
    _write_bars(corpus, "USD_JPY", rows)
    _write_bars(corpus, "EUR_USD", [(stamp, value + 100) for stamp, value in rows])

    manifest = SESSION._build_reproducibility_manifest(
        _manifest_args(corpus, pairs="USD_JPY,EUR_USD", continuous_mtm=True)
    )
    replay = manifest["replay"]

    assert replay["mtm_coordinate_contract"] == "QR_REPLAY_MTM_COORDINATES_V1"
    assert replay["phase_order"] == ["O", "H", "L", "C"]
    assert replay["feed_pairs"] == ["EUR_USD", "USD_JPY"]
    assert replay["expected_phase_mark_count"] == 8
    assert replay["full_pair_phase_coverage"] is True
    assert replay["partial_phase_count"] == 0
    assert replay["first_coordinate"]["phase"] == "O"
    assert replay["last_coordinate"]["phase"] == "C"

    alternate = SESSION._build_reproducibility_manifest(
        _manifest_args(
            corpus,
            pairs="EUR_USD,USD_JPY",
            intrabar="OLHC",
            continuous_mtm=True,
        )
    )["replay"]
    assert alternate["phase_order"] == ["O", "L", "H", "C"]
    assert alternate["expected_phase_mark_count"] == 8
    assert (
        alternate["expected_batch_chain_terminal_sha256"]
        != replay["expected_batch_chain_terminal_sha256"]
    )


def test_manifest_rejects_partial_feed_pair_phase_coverage(tmp_path):
    corpus = tmp_path / "corpus"
    _write_bars(
        corpus,
        "USD_JPY",
        [
            ("2026-01-01T00:00:00.000000000Z", 100.0),
            ("2026-01-01T00:01:00.000000000Z", 110.0),
        ],
    )
    _write_bars(
        corpus,
        "EUR_USD",
        [("2026-01-01T00:00:00.000000000Z", 200.0)],
    )

    with pytest.raises(ValueError, match="full feed-pair coverage"):
        SESSION._build_reproducibility_manifest(
            _manifest_args(
                corpus, pairs="USD_JPY,EUR_USD", continuous_mtm=True
            )
        )


def test_manifest_rejects_resume_snapshot_as_fresh_mtm_evidence(tmp_path):
    corpus = tmp_path / "corpus"
    _write_bars(
        corpus,
        "USD_JPY",
        [("2026-01-01T00:00:00.000000000Z", 100.0)],
    )
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "broker_snapshot.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="resumed replay cannot attest"):
        SESSION._build_reproducibility_manifest(
            _manifest_args(
                corpus,
                session_dir=session_dir,
                time_to="2026-01-01T00:01:00",
                continuous_mtm=True,
            )
        )


def test_manifest_preserves_generic_replay_snapshot_resume_without_mtm(tmp_path):
    corpus = tmp_path / "corpus"
    _write_bars(
        corpus,
        "USD_JPY",
        [("2026-01-01T00:00:00.000000000Z", 100.0)],
    )
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    snapshot = session_dir / "broker_snapshot.json"
    snapshot.write_text("{}", encoding="utf-8")

    manifest = SESSION._build_reproducibility_manifest(
        _manifest_args(
            corpus,
            session_dir=session_dir,
            time_to="2026-01-01T00:01:00",
        )
    )

    assert manifest["replay"]["continuous_mtm"] is False
    assert "mtm_coordinate_contract" not in manifest["replay"]
    assert manifest["resume_snapshot"]["path"] == str(snapshot.resolve())


def test_session_start_ledger_binds_reproducibility_manifest(tmp_path):
    corpus = tmp_path / "corpus"
    session_dir = tmp_path / "session"
    _write_bar(corpus, "USD_JPY", "2026-01-01T00:00:00.000000000Z", 100)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--feed",
            "replay",
            "--session-dir",
            str(session_dir),
            "--pairs",
            "USD_JPY",
            "--corpus-root",
            str(corpus),
            "--from",
            "2026-01-01T00:00:00",
            "--to",
            "2026-01-01T00:01:00",
            "--bars-per-second",
            "10000",
            "--fast-ledger",
        ],
        cwd=SCRIPT_PATH.parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    records = [
        json.loads(line)
        for line in (session_dir / "ledger.jsonl").read_text().splitlines()
    ]
    start = records[0]
    assert start["event"] == "SESSION_START"
    assert all(row["event"] != "BOT_LOADED" for row in records)
    marks = [row for row in records if row["event"] == "ACCOUNT_MARK"]
    assert marks == []
    manifest = start["payload"]["reproducibility_manifest"]
    assert manifest["replay"]["continuous_mtm"] is False
    assert "mtm_coordinate_contract" not in manifest["replay"]
    assert (
        start["payload"]["reproducibility_manifest_sha256"]
        == manifest["manifest_sha256"]
    )
    body = dict(manifest)
    digest = body.pop("manifest_sha256")
    assert digest == SESSION._canonical_sha256(body)


def test_two_pair_replay_emits_complete_continuous_mtm_evidence(tmp_path):
    corpus = tmp_path / "corpus"
    session_dir = tmp_path / "session"
    rows = [
        ("2026-01-01T00:00:00.000000000Z", 100.0),
        ("2026-01-01T00:01:00.000000000Z", 110.0),
    ]
    _write_bars(corpus, "USD_JPY", rows)
    _write_bars(corpus, "EUR_USD", [(stamp, value + 100) for stamp, value in rows])
    bot_module = tmp_path / "bot.py"
    bot_module.write_text(
        "class Bot:\n"
        "    def __init__(self, broker):\n"
        "        self.broker = broker\n"
        "    def on_bar_closed(self, pair, bar, epoch):\n"
        "        pass\n",
        encoding="utf-8",
    )
    (session_dir / "inbox").mkdir(parents=True)
    (session_dir / "inbox" / "market.json").write_text(
        json.dumps(
            {
                "action": "MARKET",
                "pair": "USD_JPY",
                "side": "LONG",
                "units": 1,
            }
        ),
        encoding="utf-8",
    )
    os.utime(session_dir / "inbox" / "market.json", (0, 0))

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--feed",
            "replay",
            "--session-dir",
            str(session_dir),
            "--pairs",
            "USD_JPY,EUR_USD",
            "--corpus-root",
            str(corpus),
            "--from",
            "2026-01-01T00:00:00",
            "--to",
            "2026-01-01T00:02:00",
            "--bars-per-second",
            "10000",
            "--fast-ledger",
            "--continuous-mtm",
            "--bot-module",
            f"{bot_module}:Bot",
            "--strategy-owner-id",
            "test:mtm-worker",
            "--bot-dependency",
            "src/quant_rabbit/virtual_broker.py",
        ],
        cwd=SCRIPT_PATH.parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    records = [
        json.loads(line)
        for line in (session_dir / "ledger.jsonl").read_text().splitlines()
    ]
    indexed = list(enumerate(records))
    start_index, session_start = next(
        (index, row) for index, row in indexed if row["event"] == "SESSION_START"
    )
    loaded_index, _ = next(
        (index, row) for index, row in indexed if row["event"] == "BOT_LOADED"
    )
    marks = [(index, row) for index, row in indexed if row["event"] == "ACCOUNT_MARK"]
    batches = [
        (index, row) for index, row in indexed if row["event"] == "QUOTE_BATCH_BEGIN"
    ]
    stop_index, session_stop = next(
        (index, row) for index, row in indexed if row["event"] == "SESSION_STOP"
    )

    assert [row["payload"]["kind"] for _, row in marks] == [
        "START",
        *("PHASE" for _ in range(8)),
        "TERMINAL",
    ]
    assert start_index < loaded_index < marks[0][0]
    assert len(batches) == 8
    phase_marks = marks[1:-1]
    assert [row["payload"]["coordinate"] for _, row in batches] == [
        row["payload"]["coordinate"] for _, row in phase_marks
    ]
    for (batch_index, batch), (mark_index, mark) in zip(
        batches, phase_marks, strict=True
    ):
        assert batch_index < mark_index
        assert batch["payload"]["batch_sha256"] == mark["payload"]["batch_sha256"]
    second_open_batch_index, _ = batches[4]
    second_open_mark_index, _ = phase_marks[4]
    action_fill_index = next(
        index for index, row in indexed if row["event"] == "FILL_MARKET"
    )
    assert second_open_batch_index < action_fill_index < second_open_mark_index
    terminal_index, terminal = marks[-1]
    assert terminal_index < stop_index
    assert terminal["payload"]["coordinate"] is None
    assert terminal["payload"]["feed_cursor"]["completed"] is True

    manifest_replay = session_start["payload"]["reproducibility_manifest"]["replay"]
    assert manifest_replay["expected_phase_mark_count"] == 8
    assert (
        manifest_replay["expected_batch_chain_terminal_sha256"]
        == batches[-1][1]["payload"]["batch_sha256"]
    )
    stop = session_stop["payload"]
    assert stop["mtm_complete"] is True
    assert stop["mtm_mark_count"] == 10
    assert stop["mtm_terminal_mark_sha256"] == terminal["payload"]["mark_sha256"]
    assert stop["account"] == terminal["payload"]["account"]


def test_period_end_settlement_resolves_only_declared_owner(tmp_path: Path) -> None:
    broker = VirtualBroker(tmp_path / "ledger.jsonl", balance_jpy=2_000_000.0)
    broker.on_quote("USD_JPY", 150.0, 150.02, "2026-01-01T00:00:00+00:00")
    owner = OwnedBrokerView(broker, "dojo:settle")
    sibling = OwnedBrokerView(broker, "dojo:sibling")
    own_trade = owner.market_order("USD_JPY", "LONG", 100)
    own_order = owner.limit_order("USD_JPY", "LONG", 100, price=149.0)
    sibling_trade = sibling.market_order("USD_JPY", "SHORT", 100)

    SESSION._settle_custom_bot_at_end(broker, "dojo:settle")

    assert own_trade not in broker.positions
    assert own_order not in broker.orders
    assert sibling_trade in broker.positions
    records = [json.loads(line) for line in broker.ledger_path.read_text().splitlines()]
    settlement = next(
        row["payload"] for row in records if row["event"] == "PERIOD_END_SETTLEMENT"
    )
    assert settlement["strategy_owner_id"] == "dojo:settle"
    assert settlement["complete"] is True
    assert settlement["errors"] == []


def _replay_args(corpus: Path, **overrides):
    values = {
        "pairs": "USD_JPY",
        "corpus_root": str(corpus),
        "bars_per_second": 10000.0,
        "state_every": 1,
        "step": False,
        "bot_bar": "feed",
        "granularity": "M1",
        "time_from": "2026-01-01T00:00:00",
        "time_to": "2026-01-01T00:03:00",
        "intrabar": "OHLC",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_closed_bar_bot_market_order_executes_at_next_open_quote(tmp_path):
    corpus = tmp_path / "corpus"
    _write_bars(
        corpus,
        "USD_JPY",
        [
            ("2026-01-01T00:00:00.000000000Z", 100.0),
            ("2026-01-01T00:01:00.000000000Z", 110.0),
        ],
    )
    session = tmp_path / "session"
    (session / "inbox" / "processed").mkdir(parents=True)
    broker = VirtualBroker(
        session / "ledger.jsonl", balance_jpy=2_000_000.0, fast_ledger=True
    )

    class Bot:
        placed = False

        def on_bar_closed(self, pair, bar, epoch):
            if not self.placed:
                self.placed = True
                broker.market_order(pair, "LONG", 1)

    SESSION.run_replay(_replay_args(corpus), broker, session, bot=Bot())
    fill = next(
        json.loads(line)["payload"]
        for line in (session / "ledger.jsonl").read_text().splitlines()
        if json.loads(line)["event"] == "FILL_MARKET"
    )
    assert fill["quote"]["ts"].endswith("00:01:00+00:00#O")
    assert fill["entry"] == pytest.approx(111.02)


def test_replay_resume_skips_all_quotes_at_or_before_cursor(tmp_path):
    corpus = tmp_path / "corpus"
    _write_bars(
        corpus,
        "USD_JPY",
        [
            ("2026-01-01T00:00:00.000000000Z", 100.0),
            ("2026-01-01T00:01:00.000000000Z", 110.0),
            ("2026-01-01T00:02:00.000000000Z", 120.0),
        ],
    )
    session = tmp_path / "session"
    (session / "inbox" / "processed").mkdir(parents=True)
    broker = VirtualBroker(
        session / "ledger.jsonl", balance_jpy=2_000_000.0, fast_ledger=True
    )
    broker.on_quote("USD_JPY", 112.0, 112.02, "2026-01-01T00:01:00+00:00#C")
    trade_id = broker.market_order("USD_JPY", "LONG", 1, sl_pips=5)
    broker.feed_cursor = {
        "mode": "replay",
        "epoch": 1767225660,
        "phase": "C",
        "bar_count": 2,
        "completed": False,
        "replay_identity_sha256": "sealed-id",
    }

    SESSION.run_replay(
        _replay_args(corpus),
        broker,
        session,
        replay_identity_sha256="sealed-id",
    )
    assert trade_id in broker.positions
    assert all(
        not (
            json.loads(line)["event"] == "EXIT_SL"
            and json.loads(line)["payload"]["quote"]["ts"] < "2026-01-01T00:01:00"
        )
        for line in (session / "ledger.jsonl").read_text().splitlines()
    )


def test_replay_resume_with_exposure_and_no_cursor_fails_closed(tmp_path):
    corpus = tmp_path / "corpus"
    _write_bars(
        corpus,
        "USD_JPY",
        [("2026-01-01T00:00:00.000000000Z", 100.0)],
    )
    session = tmp_path / "session"
    (session / "inbox" / "processed").mkdir(parents=True)
    broker = VirtualBroker(session / "ledger.jsonl", balance_jpy=2_000_000.0)
    broker.on_quote("USD_JPY", 110.0, 110.02, "2026-01-01T00:01:00+00:00#C")
    broker.market_order("USD_JPY", "LONG", 1)

    with pytest.raises(VirtualBrokerError, match="no causal feed cursor"):
        SESSION.run_replay(_replay_args(corpus), broker, session)


def test_inbox_rejects_non_finite_json_before_broker_mutation(tmp_path):
    session = tmp_path / "session"
    inbox = session / "inbox"
    (inbox / "processed").mkdir(parents=True)
    broker = VirtualBroker(session / "ledger.jsonl", balance_jpy=2_000_000.0)
    broker.on_quote("USD_JPY", 150.0, 150.02, "2026-01-01T00:00:00+00:00")
    action = inbox / "nan.json"
    action.write_text('{"action":"MARKET","pair":"USD_JPY","side":"LONG","units":NaN}')
    os.utime(action, (0, 0))

    assert SESSION._process_inbox(session, broker) == 1
    assert not broker.positions
    rejected = [
        json.loads(line)
        for line in broker.ledger_path.read_text().splitlines()
        if json.loads(line)["event"] == "AGENT_ACTION_REJECTED"
    ]
    assert rejected
    assert "non-finite JSON constant" in rejected[-1]["payload"]["error"]


def test_manifest_requires_every_pair_and_period_and_replay_rechecks_seal(tmp_path):
    corpus = tmp_path / "corpus"
    _write_bar(corpus, "USD_JPY", "2026-01-01T00:00:00.000000000Z", 100)
    args = argparse.Namespace(
        feed="replay",
        pairs="USD_JPY,EUR_USD",
        balance=200_000.0,
        corpus_root=str(corpus),
        time_from="2026-01-01T00:00:00",
        time_to="2026-01-01T00:01:00",
        granularity="M1",
        intrabar="OHLC",
        bot=None,
        bot_module=None,
        bot_bar="feed",
        slippage_pips=0.0,
        financing_pips_day=0.0,
    )
    with pytest.raises(ValueError, match="missing requested pair shards"):
        SESSION._build_reproducibility_manifest(args)

    args.pairs = "USD_JPY"
    manifest = SESSION._build_reproducibility_manifest(args)
    _write_bar(corpus, "USD_JPY", "2026-01-01T00:00:00.000000000Z", 101)
    with pytest.raises(ValueError, match="changed"):
        list(
            SESSION._iter_replay_quotes(
                corpus,
                ["USD_JPY"],
                args.time_from,
                args.time_to,
                expected_shards=manifest["corpus"]["shards"],
            )
        )


def test_invalid_corpus_geometry_fails_closed(tmp_path):
    corpus = tmp_path / "corpus"
    shard = corpus / "fixture" / "USD_JPY" / "USD_JPY_M1_BA_2026.jsonl.gz"
    shard.parent.mkdir(parents=True)
    row = {
        "time": "2026-01-01T00:00:00Z",
        "bid": {"o": 100, "h": 99, "l": 98, "c": 100},
        "ask": {"o": 100.02, "h": 100.04, "l": 100.00, "c": 100.02},
    }
    with gzip.open(shard, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="high geometry"):
        list(
            SESSION._iter_replay_quotes(
                corpus,
                ["USD_JPY"],
                "2026-01-01T00:00:00",
                "2026-01-01T00:01:00",
            )
        )


def test_live_incomplete_quote_batch_never_processes_actions(tmp_path, monkeypatch):
    import quant_rabbit.broker.oanda as oanda

    session = tmp_path / "session"
    inbox = session / "inbox"
    (inbox / "processed").mkdir(parents=True)
    broker = VirtualBroker(session / "ledger.jsonl", balance_jpy=2_000_000.0)
    broker.on_quote("USD_JPY", 150.0, 150.02, "2026-01-01T00:00:00+00:00")
    action = inbox / "market.json"
    action.write_text('{"action":"MARKET","pair":"USD_JPY","side":"LONG","units":1}')
    os.utime(action, (0, 0))

    class EmptyClient:
        def quotes(self, pairs):
            return {}

    monkeypatch.setattr(oanda, "OandaReadOnlyClient", lambda: EmptyClient())
    monkeypatch.setattr(
        SESSION, "compute_market_status", lambda now: SimpleNamespace(is_fx_open=True)
    )
    times = iter((0.0, 0.0, 100.0))
    monkeypatch.setattr(SESSION.time_mod, "time", lambda: next(times))
    monkeypatch.setattr(SESSION.time_mod, "sleep", lambda seconds: None)
    args = argparse.Namespace(pairs="USD_JPY", minutes=1.0)

    SESSION.run_live(args, broker, session)
    assert not broker.positions
    assert action.exists()
    assert any(
        json.loads(line)["event"] == "QUOTE_BATCH_REJECTED"
        for line in broker.ledger_path.read_text().splitlines()
    )
