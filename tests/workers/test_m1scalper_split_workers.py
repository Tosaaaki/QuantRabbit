from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _reload(module_name: str):
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def _module_source(module_name: str) -> str:
    module = _reload(module_name)
    return Path(str(module.__file__)).read_text(encoding="utf-8")


def test_scenario_workers_do_not_import_m1scalper_entry_engine():
    assert "workers.scalp_m1scalper" not in _module_source("workers.scalp_trend_breakout.worker")
    assert "workers.scalp_m1scalper" not in _module_source("workers.scalp_pullback_continuation.worker")
    assert "workers.scalp_m1scalper" not in _module_source("workers.scalp_failed_break_reverse.worker")


def test_scenario_configs_have_strategy_specific_defaults(monkeypatch):
    monkeypatch.setenv("QUANTRABBIT_ENV_FILE", "/tmp/does-not-exist-qr-env")
    monkeypatch.delenv("M1SCALP_SIGNAL_TAG_CONTAINS", raising=False)
    monkeypatch.delenv("M1SCALP_STRATEGY_TAG_OVERRIDE", raising=False)
    monkeypatch.delenv("M1SCALP_ALLOW_REVERSION", raising=False)
    monkeypatch.delenv("M1SCALP_ALLOW_TREND", raising=False)

    trend = _reload("workers.scalp_trend_breakout.config")
    assert trend.LOG_PREFIX == "[TrendBreakout]"
    assert trend.SIGNAL_TAG_CONTAINS == {"breakout-retest"}
    assert trend.STRATEGY_TAG_OVERRIDE == "TrendBreakout"
    assert trend.ALLOW_REVERSION is False
    assert trend.ALLOW_TREND is True

    pullback = _reload("workers.scalp_pullback_continuation.config")
    assert pullback.LOG_PREFIX == "[PullbackContinuation]"
    assert pullback.SIGNAL_TAG_CONTAINS == {"buy-dip", "sell-rally"}
    assert pullback.STRATEGY_TAG_OVERRIDE == "PullbackContinuation"
    assert pullback.ALLOW_REVERSION is True
    assert pullback.ALLOW_TREND is True

    failed = _reload("workers.scalp_failed_break_reverse.config")
    assert failed.LOG_PREFIX == "[FailedBreakReverse]"
    assert failed.SIGNAL_TAG_CONTAINS == {"vshape-rebound"}
    assert failed.STRATEGY_TAG_OVERRIDE == "FailedBreakReverse"
    assert failed.ALLOW_REVERSION is True
    assert failed.ALLOW_TREND is False


def test_scenario_exit_workers_have_strategy_specific_default_allowlists(monkeypatch):
    monkeypatch.delenv("M1SCALP_EXIT_TAG_ALLOWLIST", raising=False)
    trend_exit = _reload("workers.scalp_trend_breakout.exit_worker")
    assert trend_exit.ALLOWED_TAGS == {"TrendBreakout"}

    monkeypatch.delenv("M1SCALP_EXIT_TAG_ALLOWLIST", raising=False)
    pullback_exit = _reload("workers.scalp_pullback_continuation.exit_worker")
    assert pullback_exit.ALLOWED_TAGS == {"PullbackContinuation"}

    monkeypatch.delenv("M1SCALP_EXIT_TAG_ALLOWLIST", raising=False)
    failed_exit = _reload("workers.scalp_failed_break_reverse.exit_worker")
    assert failed_exit.ALLOWED_TAGS == {"FailedBreakReverse"}
