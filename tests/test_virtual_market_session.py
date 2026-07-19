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


def _write_bars(
    root: Path,
    pair: str,
    rows: list[tuple[str, float]],
    *,
    granularity: str = "M1",
) -> None:
    shard = root / "fixture" / pair / f"{pair}_{granularity}_BA_2026.jsonl.gz"
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
        manifest["replay"]["mtm_coordinate_contract"]
        == "QR_REPLAY_ASYNC_MTM_COORDINATES_V2"
    )
    assert manifest["replay"]["coordinate_schedule"] == "SEALED_CORPUS_EPOCH_UNION"
    assert (
        manifest["replay"]["quote_policy"] == "OBSERVED_ONLY_NO_SYNTHETIC_CARRY_QUOTES"
    )
    assert manifest["replay"]["phase_order"] == ["O", "H", "L", "C"]
    assert manifest["replay"]["feed_pairs"] == ["USD_JPY"]
    assert manifest["replay"]["expected_union_epoch_count"] == 1
    assert manifest["replay"]["expected_partial_epoch_count"] == 0
    assert manifest["replay"]["expected_phase_mark_count"] == 4
    assert manifest["replay"]["expected_partial_phase_count"] == 0
    assert manifest["replay"]["pair_row_counts"] == {"USD_JPY": 1}
    assert manifest["replay"]["expected_quote_count"] == 4
    assert manifest["replay"]["max_carried_quote_age_seconds"] == 900
    assert manifest["replay"]["synthetic_quote_count"] == 0
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

    assert replay["mtm_coordinate_contract"] == "QR_REPLAY_ASYNC_MTM_COORDINATES_V2"
    assert replay["phase_order"] == ["O", "H", "L", "C"]
    assert replay["feed_pairs"] == ["EUR_USD", "USD_JPY"]
    assert replay["expected_phase_mark_count"] == 8
    assert replay["expected_union_epoch_count"] == 2
    assert replay["expected_full_epoch_count"] == 2
    assert replay["expected_partial_epoch_count"] == 0
    assert replay["expected_partial_phase_count"] == 0
    assert replay["pair_row_counts"] == {"EUR_USD": 2, "USD_JPY": 2}
    assert replay["expected_quote_count"] == 16
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


def test_manifest_seals_sparse_observed_pair_phase_coverage(tmp_path):
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

    replay = SESSION._build_reproducibility_manifest(
        _manifest_args(corpus, pairs="USD_JPY,EUR_USD", continuous_mtm=True)
    )["replay"]

    assert replay["expected_union_epoch_count"] == 2
    assert replay["expected_full_epoch_count"] == 1
    assert replay["expected_partial_epoch_count"] == 1
    assert replay["expected_partial_phase_count"] == 4
    assert replay["pair_row_counts"] == {"EUR_USD": 1, "USD_JPY": 2}
    assert replay["expected_quote_count"] == 12
    assert len(replay["availability_mask_sha256"]) == 64


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
    terminal_ts = "2026-01-01T00:00:00+00:00#C"
    broker.on_quote("USD_JPY", 150.0, 150.02, terminal_ts)
    owner = OwnedBrokerView(broker, "dojo:settle")
    sibling = OwnedBrokerView(broker, "dojo:sibling")
    own_trade = owner.market_order("USD_JPY", "LONG", 100)
    own_order = owner.limit_order("USD_JPY", "LONG", 100, price=149.0)
    sibling_trade = sibling.market_order("USD_JPY", "SHORT", 100)

    SESSION._settle_custom_bot_at_end(
        broker,
        "dojo:settle",
        {
            "coordinate": {"epoch": 1767225600, "phase": "C"},
            "batch_pairs": ["USD_JPY"],
        },
    )

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


def test_period_end_settlement_refuses_stale_pair_absent_from_terminal_batch(
    tmp_path: Path,
) -> None:
    broker = VirtualBroker(tmp_path / "ledger.jsonl", balance_jpy=2_000_000.0)
    broker.on_quote("USD_JPY", 150.0, 150.02, "2026-01-02T21:59:00+00:00#C")
    owner = OwnedBrokerView(broker, "dojo:stale-settle")
    trade_id = owner.market_order("USD_JPY", "LONG", 100)
    broker.on_quote("EUR_USD", 1.10, 1.1002, "2026-01-04T22:05:00+00:00#C")

    SESSION._settle_custom_bot_at_end(
        broker,
        "dojo:stale-settle",
        {
            "coordinate": {"epoch": 1767564300, "phase": "C"},
            "batch_pairs": ["EUR_USD"],
        },
    )

    assert trade_id in broker.positions
    settlement = next(
        json.loads(line)["payload"]
        for line in broker.ledger_path.read_text().splitlines()
        if json.loads(line)["event"] == "PERIOD_END_SETTLEMENT"
    )
    assert settlement["complete"] is False
    assert settlement["errors"] == [f"close:{trade_id}:TERMINAL_PAIR_QUOTE_UNAVAILABLE"]


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


def test_stale_gap_skips_actions_and_clock_uses_current_observed_open_epoch(tmp_path):
    corpus = tmp_path / "corpus"
    _write_bars(
        corpus,
        "USD_JPY",
        [
            ("2026-01-01T00:00:00.000000000Z", 100.0),
            ("2026-01-01T02:00:00.000000000Z", 110.0),
            ("2026-01-01T02:01:00.000000000Z", 111.0),
            ("2026-01-01T03:00:00.000000000Z", 112.0),
            ("2026-01-01T03:01:00.000000000Z", 113.0),
        ],
    )
    session = tmp_path / "session"
    (session / "inbox" / "processed").mkdir(parents=True)
    broker = VirtualBroker(session / "ledger.jsonl", balance_jpy=2_000_000.0)

    class Bot:
        def __init__(self):
            self.trade_id = None
            self.entry_decision_epoch = None
            self.decisions = []

        def on_bar_closed(self, pair, bar, epoch):
            self.decisions.append((bar["epoch"], epoch))
            if self.trade_id is None:
                self.trade_id = broker.market_order(pair, "LONG", 1)
                self.entry_decision_epoch = epoch
            elif (
                self.trade_id in broker.positions
                and epoch - self.entry_decision_epoch >= 3600
            ):
                broker.close_trade(self.trade_id)

    bot = Bot()
    SESSION.run_replay(
        _replay_args(corpus, time_to="2026-01-01T03:02:00"),
        broker,
        session,
        bot=bot,
    )
    records = [json.loads(line) for line in broker.ledger_path.read_text().splitlines()]
    fill = next(row["payload"] for row in records if row["event"] == "FILL_MARKET")
    close = next(row["payload"] for row in records if row["event"] == "CLOSE")

    assert bot.decisions == [
        (1767225600, 1767232800),
        (1767232800, 1767232860),
        (1767232860, 1767236400),
        (1767236400, 1767236460),
    ]
    assert bot.entry_decision_epoch == 1767232860
    assert fill["quote"]["ts"] == "2026-01-01T02:01:00+00:00#O"
    assert close["quote"]["ts"] == "2026-01-01T03:01:00+00:00#O"


def test_sparse_pair_callbacks_close_once_at_pair_local_next_open(tmp_path):
    corpus = tmp_path / "corpus"
    _write_bars(
        corpus,
        "USD_JPY",
        [
            ("2026-01-01T00:00:00.000000000Z", 100.0),
            ("2026-01-01T00:01:00.000000000Z", 101.0),
            ("2026-01-01T00:03:00.000000000Z", 103.0),
        ],
    )
    _write_bars(
        corpus,
        "EUR_USD",
        [
            ("2026-01-01T00:00:00.000000000Z", 200.0),
            ("2026-01-01T00:02:00.000000000Z", 202.0),
            ("2026-01-01T00:03:00.000000000Z", 203.0),
        ],
    )
    session = tmp_path / "session"
    (session / "inbox" / "processed").mkdir(parents=True)
    broker = VirtualBroker(session / "ledger.jsonl", balance_jpy=2_000_000.0)

    class Bot:
        def __init__(self):
            self.closed = []

        def on_bar_closed(self, pair, bar, epoch):
            self.closed.append((pair, bar["epoch"], epoch, broker.last_quotes[pair][2]))

    bot = Bot()
    SESSION.run_replay(
        _replay_args(
            corpus,
            pairs="USD_JPY,EUR_USD",
            time_to="2026-01-01T00:04:00",
        ),
        broker,
        session,
        bot=bot,
    )

    assert bot.closed == [
        ("USD_JPY", 1767225600, 1767225660, "2026-01-01T00:01:00+00:00#O"),
        ("EUR_USD", 1767225600, 1767225720, "2026-01-01T00:02:00+00:00#O"),
        ("EUR_USD", 1767225720, 1767225780, "2026-01-01T00:03:00+00:00#O"),
        ("USD_JPY", 1767225660, 1767225780, "2026-01-01T00:03:00+00:00#O"),
    ]


def test_bot_callback_feed_gate_requires_each_pair_to_initialize_once(tmp_path):
    broker = VirtualBroker(tmp_path / "ledger.jsonl", balance_jpy=2_000_000.0)
    broker.on_quote("USD_JPY", 150.0, 150.02, "2026-01-01T00:15:00+00:00#O")
    assert (
        SESSION._bot_callbacks_have_initialized_feed(broker, ["EUR_USD", "USD_JPY"])
        is False
    )
    broker.on_quote("EUR_USD", 1.1, 1.1002, "2026-01-01T00:00:00+00:00#O")
    assert SESSION._bot_callbacks_have_initialized_feed(broker, ["EUR_USD", "USD_JPY"])


@pytest.mark.parametrize(
    ("decision_epoch", "expected"),
    [
        (60, True),
        (900, True),
        (901, False),
    ],
)
def test_bot_callback_causal_gap_boundary(decision_epoch, expected):
    assert (
        SESSION._bot_callback_is_within_causal_gap({"epoch": 0}, decision_epoch)
        is expected
    )


def test_sunday_stale_bars_update_state_without_new_risk_and_callbacks_continue(
    tmp_path,
):
    corpus = tmp_path / "corpus"
    _write_bars(
        corpus,
        "USD_JPY",
        [
            ("2026-01-02T21:59:00.000000000Z", 100.0),
            ("2026-01-04T22:05:00.000000000Z", 101.0),
            ("2026-01-04T22:10:00.000000000Z", 102.0),
            ("2026-01-04T22:11:00.000000000Z", 103.0),
        ],
    )
    _write_bars(
        corpus,
        "EUR_USD",
        [
            ("2026-01-02T21:59:00.000000000Z", 200.0),
            ("2026-01-04T22:10:00.000000000Z", 202.0),
            ("2026-01-04T22:11:00.000000000Z", 203.0),
        ],
    )
    session = tmp_path / "session"
    (session / "inbox" / "processed").mkdir(parents=True)
    broker = VirtualBroker(session / "ledger.jsonl", balance_jpy=2_000_000.0)

    class Bot:
        def __init__(self):
            self.closed = []

        def on_bar_closed(self, pair, bar, epoch):
            self.closed.append((pair, bar["epoch"], epoch))

    bot = Bot()
    SESSION.run_replay(
        _replay_args(
            corpus,
            pairs="USD_JPY,EUR_USD",
            time_from="2026-01-02T21:59:00",
            time_to="2026-01-04T22:12:00",
        ),
        broker,
        session,
        bot=bot,
    )

    assert bot.closed == [
        ("USD_JPY", 1767391140, 1767564300),
        ("EUR_USD", 1767391140, 1767564600),
        ("USD_JPY", 1767564300, 1767564600),
        ("EUR_USD", 1767564600, 1767564660),
        ("USD_JPY", 1767564600, 1767564660),
    ]


def test_sparse_s5_m1_aggregate_closes_on_pair_local_next_minute(tmp_path):
    corpus = tmp_path / "corpus"
    _write_bars(
        corpus,
        "USD_JPY",
        [
            ("2026-01-01T00:00:00.000000000Z", 100.0),
            ("2026-01-01T00:00:05.000000000Z", 101.0),
            ("2026-01-01T00:01:00.000000000Z", 102.0),
        ],
        granularity="S5",
    )
    _write_bars(
        corpus,
        "EUR_USD",
        [
            ("2026-01-01T00:00:00.000000000Z", 200.0),
            ("2026-01-01T00:00:10.000000000Z", 201.0),
            ("2026-01-01T00:01:05.000000000Z", 202.0),
        ],
        granularity="S5",
    )
    session = tmp_path / "session"
    (session / "inbox" / "processed").mkdir(parents=True)
    broker = VirtualBroker(session / "ledger.jsonl", balance_jpy=2_000_000.0)

    class Bot:
        def __init__(self):
            self.closed = []

        def on_bar_closed(self, pair, bar, epoch):
            self.closed.append((pair, bar["epoch"], epoch, broker.last_quotes[pair][2]))

    bot = Bot()
    SESSION.run_replay(
        _replay_args(
            corpus,
            pairs="USD_JPY,EUR_USD",
            granularity="S5",
            bot_bar="M1",
            time_to="2026-01-01T00:02:00",
        ),
        broker,
        session,
        bot=bot,
    )

    assert bot.closed == [
        ("USD_JPY", 1767225600, 1767225660, "2026-01-01T00:01:00+00:00#O"),
        ("EUR_USD", 1767225600, 1767225665, "2026-01-01T00:01:05+00:00#O"),
    ]


def test_sparse_mtm_runtime_matches_union_commitment_and_detects_tamper(tmp_path):
    corpus = tmp_path / "corpus"
    _write_bars(
        corpus,
        "USD_JPY",
        [
            ("2026-01-01T00:00:00.000000000Z", 100.0),
            ("2026-01-01T00:01:00.000000000Z", 101.0),
        ],
    )
    _write_bars(
        corpus,
        "EUR_USD",
        [("2026-01-01T00:01:00.000000000Z", 200.0)],
    )
    args = _replay_args(
        corpus,
        pairs="USD_JPY,EUR_USD",
        time_to="2026-01-01T00:02:00",
    )
    shards = SESSION._selected_corpus_shards(
        corpus,
        ["EUR_USD", "USD_JPY"],
        args.time_from,
        args.time_to,
        "M1",
    )
    shard_rows = [
        {
            "path": path.relative_to(corpus).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": SESSION._file_sha256(path),
        }
        for path in shards
    ]
    commitment = SESSION._build_replay_mtm_commitment(
        root=corpus,
        pairs=["EUR_USD", "USD_JPY"],
        time_from=args.time_from,
        time_to=args.time_to,
        intrabar="OHLC",
        granularity="M1",
        expected_shards=shard_rows,
    )

    session = tmp_path / "valid"
    (session / "inbox" / "processed").mkdir(parents=True)
    broker = VirtualBroker(session / "ledger.jsonl", balance_jpy=2_000_000.0)
    broker.account_mark("START")
    runtime = SESSION.run_replay(
        args,
        broker,
        session,
        expected_shards=shard_rows,
        mtm_contract=commitment,
    )
    assert runtime["phase_mark_count"] == 8
    assert commitment["expected_partial_epoch_count"] == 1
    assert commitment["expected_partial_phase_count"] == 4
    batches = [
        json.loads(line)["payload"]
        for line in (session / "ledger.jsonl").read_text().splitlines()
        if json.loads(line)["event"] == "QUOTE_BATCH_BEGIN"
    ]
    assert [row["batch_pairs"] for row in batches[:4]] == [["USD_JPY"]] * 4
    assert [row["batch_pairs"] for row in batches[4:]] == [["EUR_USD", "USD_JPY"]] * 4

    tampered_session = tmp_path / "tampered"
    (tampered_session / "inbox" / "processed").mkdir(parents=True)
    tampered_broker = VirtualBroker(
        tampered_session / "ledger.jsonl", balance_jpy=2_000_000.0
    )
    tampered_broker.account_mark("START")
    tampered = {**commitment, "availability_mask_sha256": "0" * 64}
    with pytest.raises(VirtualBrokerError, match="availability_mask_sha256"):
        SESSION.run_replay(
            args,
            tampered_broker,
            tampered_session,
            expected_shards=shard_rows,
            mtm_contract=tampered,
        )


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
