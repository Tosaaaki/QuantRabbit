from __future__ import annotations

import importlib
import os
import sys


def _reload(module_name: str):
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def test_trend_breakout_worker_sets_entry_defaults(monkeypatch):
    monkeypatch.delenv("M1SCALP_SIGNAL_TAG_CONTAINS", raising=False)
    monkeypatch.delenv("M1SCALP_STRATEGY_TAG_OVERRIDE", raising=False)
    monkeypatch.delenv("M1SCALP_ALLOW_REVERSION", raising=False)
    monkeypatch.delenv("M1SCALP_ALLOW_TREND", raising=False)

    _reload("workers.scalp_trend_breakout.worker")

    assert os.getenv("M1SCALP_SIGNAL_TAG_CONTAINS") == "breakout-retest"
    assert os.getenv("M1SCALP_STRATEGY_TAG_OVERRIDE") == "TrendBreakout"
    assert os.getenv("M1SCALP_ALLOW_REVERSION") == "0"
    assert os.getenv("M1SCALP_ALLOW_TREND") == "1"


def test_pullback_continuation_worker_sets_entry_defaults(monkeypatch):
    monkeypatch.delenv("M1SCALP_SIGNAL_TAG_CONTAINS", raising=False)
    monkeypatch.delenv("M1SCALP_STRATEGY_TAG_OVERRIDE", raising=False)

    _reload("workers.scalp_pullback_continuation.worker")

    assert os.getenv("M1SCALP_SIGNAL_TAG_CONTAINS") == "buy-dip,sell-rally"
    assert os.getenv("M1SCALP_STRATEGY_TAG_OVERRIDE") == "PullbackContinuation"


def test_failed_break_reverse_worker_sets_entry_defaults(monkeypatch):
    monkeypatch.delenv("M1SCALP_SIGNAL_TAG_CONTAINS", raising=False)
    monkeypatch.delenv("M1SCALP_STRATEGY_TAG_OVERRIDE", raising=False)
    monkeypatch.delenv("M1SCALP_ALLOW_TREND", raising=False)

    _reload("workers.scalp_failed_break_reverse.worker")

    assert os.getenv("M1SCALP_SIGNAL_TAG_CONTAINS") == "vshape-rebound"
    assert os.getenv("M1SCALP_STRATEGY_TAG_OVERRIDE") == "FailedBreakReverse"
    assert os.getenv("M1SCALP_ALLOW_TREND") == "0"


def test_m1scalper_exit_allowlist_can_be_overridden(monkeypatch):
    monkeypatch.setenv("M1SCALP_EXIT_TAG_ALLOWLIST", "TrendBreakout,PullbackContinuation")
    mod = _reload("workers.scalp_m1scalper.exit_worker")
    assert mod.ALLOWED_TAGS == {"TrendBreakout", "PullbackContinuation"}


def test_trend_breakout_exit_wrapper_sets_tag_allowlist(monkeypatch):
    monkeypatch.delenv("M1SCALP_EXIT_TAG_ALLOWLIST", raising=False)
    _reload("workers.scalp_trend_breakout.exit_worker")
    assert os.getenv("M1SCALP_EXIT_TAG_ALLOWLIST") == "TrendBreakout"
