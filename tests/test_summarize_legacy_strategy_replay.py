from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize-legacy-strategy-replay.py"
SPEC = importlib.util.spec_from_file_location("legacy_replay_summary", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_metrics_distinguish_no_trades_from_zero_pnl() -> None:
    metrics = MODULE._metrics([])
    assert metrics["net_pnl_jpy"] is None
    assert metrics["profit_factor"] is None
    assert metrics["status"] == "insufficient_no_trades"


def test_metrics_compute_drawdown_and_expectancy() -> None:
    metrics = MODULE._metrics([{"pnl_jpy": 100}, {"pnl_jpy": -40}, {"pnl_jpy": -20}])
    assert metrics["net_pnl_jpy"] == 40.0
    assert metrics["expectancy_jpy"] == 13.33
    assert metrics["max_drawdown_jpy"] == 60.0
    assert metrics["profit_factor"] == 1.6667
    assert metrics["profit_giveback_rate"] == 0.6
