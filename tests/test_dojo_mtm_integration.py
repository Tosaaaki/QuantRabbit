from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from quant_rabbit.dojo_bot_trainer import score_ledger_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]
SESSION_SCRIPT = REPO_ROOT / "scripts" / "run-virtual-market-session.py"
OWNER_ID = "mtm:integration-owner"
PAIRS = ["CAD_JPY", "USD_JPY"]


def _write_synchronized_corpus(root: Path) -> None:
    for pair_index, pair in enumerate(PAIRS):
        shard = root / "fixture" / pair / f"{pair}_M1_BA_2026.jsonl.gz"
        shard.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(shard, "wt", encoding="utf-8") as handle:
            for minute in range(3):
                base = 80.0 + pair_index * 70.0 + minute * 0.01
                handle.write(
                    json.dumps(
                        {
                            "time": f"2026-01-01T00:0{minute}:00Z",
                            "bid": {
                                "o": base,
                                "h": base + 0.02,
                                "l": base - 0.02,
                                "c": base + 0.01,
                            },
                            "ask": {
                                "o": base + 0.01,
                                "h": base + 0.03,
                                "l": base - 0.01,
                                "c": base + 0.02,
                            },
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )


@pytest.mark.parametrize(
    ("entry_call", "expected_entry_events", "followup_call", "followup_events"),
    [
        (
            "self.trade_id = self.broker.market_order(pair, 'LONG', 100, sl_pips=25)",
            ["FILL_MARKET"],
            None,
            [],
        ),
        (
            "self.trade_id = self.broker.limit_order(pair, 'LONG', 100, price=200, sl_pips=25)",
            ["ORDER_LIMIT", "FILL_LIMIT"],
            None,
            [],
        ),
        (
            "self.trade_id = self.broker.market_order(pair, 'LONG', 100)",
            ["FILL_MARKET"],
            "self.broker.set_exit(self.trade_id, sl_price=999)",
            ["SET_EXIT", "EXIT_SL"],
        ),
    ],
)
def test_real_runner_and_real_scorer_verify_complete_mtm_chain(
    tmp_path: Path,
    entry_call: str,
    expected_entry_events: list[str],
    followup_call: str | None,
    followup_events: list[str],
) -> None:
    corpus = tmp_path / "corpus"
    session = tmp_path / "session"
    module = tmp_path / "one_trade_bot.py"
    _write_synchronized_corpus(corpus)
    module_source = (
        "from quant_rabbit.dojo_lab_provenance import OwnedBrokerView\n"
        "class Bot:\n"
        "    def __init__(self, broker):\n"
        f"        self.broker = OwnedBrokerView(broker, {OWNER_ID!r}, "
        "max_concurrent_per_pair=1, global_max_concurrent=2)\n"
        "        self.entered = False\n"
        "        self.followed = False\n"
        "        self.trade_id = None\n"
        "    def on_bar_closed(self, pair, bar, epoch):\n"
        "        if pair == 'USD_JPY' and not self.entered:\n"
        f"            {entry_call}\n"
        "            self.entered = True\n"
    )
    if followup_call is not None:
        module_source += (
            "        elif pair == 'USD_JPY' and not self.followed:\n"
            f"            {followup_call}\n"
            "            self.followed = True\n"
        )
    module.write_text(module_source, encoding="utf-8")
    config_json = "{}"
    command = [
        sys.executable,
        str(SESSION_SCRIPT),
        "--feed",
        "replay",
        "--session-dir",
        str(session),
        "--pairs",
        ",".join(PAIRS),
        "--corpus-root",
        str(corpus),
        "--from",
        "2026-01-01T00:00:00Z",
        "--to",
        "2026-01-01T00:03:00Z",
        "--intrabar",
        "OHLC",
        "--bars-per-second",
        "10000",
        "--fast-ledger",
        "--continuous-mtm",
        "--bot-module",
        f"{module}:Bot",
        "--strategy-owner-id",
        OWNER_ID,
        "--bot-dependency",
        "src/quant_rabbit/dojo_lab_provenance.py",
        "--bot-dependency",
        "src/quant_rabbit/virtual_broker.py",
        "--settle-at-end",
    ]
    env = dict(os.environ)
    env["DOJO_BOT_CONFIG"] = config_json

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    ledger = session / "ledger.jsonl"
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    events = [row["event"] for row in records]
    assert events.count("QUOTE_BATCH_BEGIN") == 12
    assert events.count("ACCOUNT_MARK") == 14
    entry_event_indices = [
        events.index(event_name) for event_name in expected_entry_events
    ]
    assert entry_event_indices == list(
        range(entry_event_indices[0], entry_event_indices[0] + len(entry_event_indices))
    )
    if followup_events:
        followup_indices = [events.index(event_name) for event_name in followup_events]
        assert followup_indices == list(
            range(followup_indices[0], followup_indices[0] + len(followup_indices))
        )
    manifest = records[0]["payload"]["reproducibility_manifest"]
    dependencies = manifest["bot"]["dependency_sha256"]

    metrics = score_ledger_metrics(
        ledger,
        200_000.0,
        PAIRS,
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:03:00Z",
        expected_intrabar="OHLC",
        expected_slippage_pips_per_fill=0.0,
        expected_financing_pips_per_day=0.0,
        expected_corpus_sha256=manifest["corpus"]["corpus_sha256"],
        expected_bot_config_sha256=hashlib.sha256(config_json.encode()).hexdigest(),
        expected_strategy_owner_id=OWNER_ID,
        expected_bot_module_sha256=manifest["bot"]["module_sha256"],
        expected_bot_dependency_sha256=dependencies,
        expected_feed_pairs=PAIRS,
        expected_max_concurrent_per_pair=1,
        expected_global_max_concurrent=2,
    )

    assert metrics["mtm_complete"] is True
    assert metrics["mtm_evidence_status"] == (
        "VERIFIED_COORDINATE_COMPLETE_ACCOUNT_MARK_CHAIN"
    )
    assert metrics["mtm_mark_count"] == 14
    assert metrics["fill_count"] == 1
    assert metrics["resolved_exit_slices"] == 1
    assert metrics["mtm_max_drawdown_fraction"] is not None
    assert metrics["terminal_flat"] is True
