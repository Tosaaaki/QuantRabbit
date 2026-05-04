"""Tests for CFTC COT positioning analytics."""

from __future__ import annotations

import unittest
from datetime import date, timedelta

from quant_rabbit.analysis.cot_signals import (
    COTReport,
    cot_commercial_extreme,
    cot_net_pct,
    cot_week_delta,
    cot_z_score,
)


def _report(
    week_index: int,
    *,
    leveraged_long: int,
    leveraged_short: int,
    commercial_long: int = 0,
    commercial_short: int = 0,
    open_interest: int = 1_000_000,
) -> COTReport:
    return COTReport(
        currency="USD",
        report_date=date(2026, 1, 6) + timedelta(weeks=week_index),
        leveraged_funds_long=leveraged_long,
        leveraged_funds_short=leveraged_short,
        commercial_long=commercial_long,
        commercial_short=commercial_short,
        open_interest=open_interest,
    )


class COTNetPctTests(unittest.TestCase):
    def test_zero_open_interest_returns_none(self) -> None:
        r = _report(0, leveraged_long=10, leveraged_short=5, open_interest=0)
        self.assertIsNone(cot_net_pct(r))

    def test_basic_pct(self) -> None:
        r = _report(0, leveraged_long=20_000, leveraged_short=10_000, open_interest=100_000)
        self.assertAlmostEqual(cot_net_pct(r), 10.0, places=6)


class COTZScoreTests(unittest.TestCase):
    def test_too_few_reports_returns_none(self) -> None:
        history = [_report(0, leveraged_long=100, leveraged_short=50)]
        self.assertIsNone(cot_z_score(history))

    def test_z_score_signals_extreme(self) -> None:
        # 51 noisy stable weeks then a large spike on the 52nd.
        history: list[COTReport] = []
        for i in range(51):
            jitter = 100 if i % 2 == 0 else -100
            history.append(
                _report(
                    i,
                    leveraged_long=100_000 + jitter,
                    leveraged_short=90_000,
                )
            )
        history.append(_report(51, leveraged_long=300_000, leveraged_short=10_000))
        z = cot_z_score(history)
        self.assertIsNotNone(z)
        self.assertGreater(z, 5.0)


class COTWeekDeltaTests(unittest.TestCase):
    def test_delta_change(self) -> None:
        history = [
            _report(0, leveraged_long=110_000, leveraged_short=100_000),
            _report(1, leveraged_long=130_000, leveraged_short=100_000),
        ]
        delta = cot_week_delta(history)
        self.assertIsNotNone(delta)
        # Net pct went from 1% to 3%.
        self.assertAlmostEqual(delta, 2.0, places=6)


class COTCommercialExtremeTests(unittest.TestCase):
    def test_no_extreme_when_same_side(self) -> None:
        history: list[COTReport] = []
        for i in range(52):
            history.append(
                _report(
                    i,
                    leveraged_long=100_000 + i * 100,
                    leveraged_short=90_000,
                    commercial_long=120_000 + i * 100,
                    commercial_short=80_000,
                )
            )
        # Both leveraged and commercial trending net-long: not contrarian.
        self.assertFalse(cot_commercial_extreme(history))

    def test_extreme_when_opposing_extremes(self) -> None:
        history: list[COTReport] = []
        for i in range(51):
            jitter = 100 if i % 2 == 0 else -100
            history.append(
                _report(
                    i,
                    leveraged_long=100_000 + jitter,
                    leveraged_short=90_000,
                    commercial_long=90_000 - jitter,
                    commercial_short=100_000,
                )
            )
        # Final week: leveraged super long, commercial super short → contrarian.
        history.append(
            _report(
                51,
                leveraged_long=300_000,
                leveraged_short=10_000,
                commercial_long=10_000,
                commercial_short=300_000,
            )
        )
        self.assertTrue(cot_commercial_extreme(history))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
