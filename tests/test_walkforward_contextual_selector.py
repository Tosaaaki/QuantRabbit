from __future__ import annotations

import unittest

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as exc:  # optional research dependency
    raise unittest.SkipTest(f"walk-forward research dependencies unavailable: {exc.name}") from exc

from scripts.audit_walkforward_technical_selector import (
    CONTEXT_RULES,
    _best_context_rule,
    _context_history,
    _parse_pairs,
    _scheduled_rows,
)


def _history() -> pd.DataFrame:
    rows = []
    for pair, phase, session, outcome in (
        ("EUR_USD", "TREND", "UTC_08_13", 3.0),
        ("EUR_USD", "TREND", "UTC_08_13", 2.0),
        ("EUR_USD", "TREND", "UTC_13_17", 1.0),
        ("EUR_USD", "RANGE", "UTC_08_13", -2.0),
        ("GBP_USD", "TREND", "UTC_08_13", -3.0),
    ):
        row = {
            "pair": pair,
            "market_phase": phase,
            "utc_session_bucket": session,
            "long_pips": outcome,
            "short_pips": -outcome,
        }
        row.update({rule: 1.0 for rule in CONTEXT_RULES})
        rows.append(row)
    return pd.DataFrame(rows)


def test_context_uses_most_specific_scope_with_enough_resolved_rows() -> None:
    history = _history()
    current = history.iloc[0]
    exact, exact_scope = _context_history(
        history,
        current,
        minimum_context_rows=2,
    )
    assert exact_scope == "PAIR_PHASE_SESSION"
    assert len(exact) == 2

    fallback, fallback_scope = _context_history(
        history,
        current,
        minimum_context_rows=3,
    )
    assert fallback_scope == "PAIR_PHASE"
    assert len(fallback) == 3


def test_rule_selection_uses_only_supplied_resolved_context() -> None:
    history = pd.concat([_history()] * 3, ignore_index=True)
    current = history.iloc[0]
    selected = _best_context_rule(
        history,
        current,
        minimum_qualified_rows=4,
        np=np,
    )
    assert selected is not None
    assert selected["orientation"] == 1
    assert selected["mean_pips"] > 0.0
    assert selected["qualified_rows"] >= 4


def test_schedule_uses_next_executable_open_for_short_horizons() -> None:
    index = pd.date_range("2026-07-01T00:00:00Z", periods=12, freq="5min")
    dataset = pd.DataFrame(
        {"entry_timestamp_utc": index + pd.Timedelta(minutes=5)},
        index=index,
    )

    scheduled = _scheduled_rows(dataset, entry_interval_minutes=15)

    assert list(scheduled["entry_timestamp_utc"].dt.minute) == [15, 30, 45, 0]


def test_pair_filter_normalizes_slash_pairs() -> None:
    assert _parse_pairs("aud/chf,USD_CHF") == {"AUD_CHF", "USD_CHF"}
