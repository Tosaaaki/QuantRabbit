"""CFTC Commitments-of-Traders positioning analytics.

The trader uses Leveraged-Funds net positioning (TFF report) as a contrarian
crowding signal: when leveraged-funds net % is at a multi-year extreme and
commercials sit on the opposite side, the bias is fading the speculator.

This module computes:
  * ``cot_net_pct(report)`` — leveraged-funds net % of open interest
  * ``cot_z_score(history, lookback_years)`` — z-score of latest net %
  * ``cot_week_delta(history)`` — week-on-week delta in net %
  * ``cot_commercial_extreme(report)`` — True when commercials sit at
    a contrarian extreme (≥ 1σ) opposite the leveraged funds.

All helpers operate on ``COTReport`` records. The CFTC publish format
(weekly, Friday close-of-business, downloadable as CSV/text) is parsed by
the caller — this module deliberately does not fetch the file so the test
suite stays offline.

Contract:
  ``docs/AGENT_CONTRACT.md`` §3.5 — every numeric constant must carry an
  (a) what / (b) why-constant / (c) replace-with comment.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence


# ---------------------------------------------------------------------------
# §3.5 documented constants
# ---------------------------------------------------------------------------

# (a) Default lookback for the leveraged-funds z-score (one calendar year of
#     weekly observations ≈ 52 reports).
# (b) Constant because §3.5 forbids stacking ad-hoc gates; a fixed annual
#     lookback is the operator-agreed window.
# (c) Replace if the research note retunes the percentile window — pass
#     ``lookback_years`` to ``cot_z_score`` rather than mutating this.
_COT_DEFAULT_LOOKBACK_YEARS = 1

# (a) Number of weekly reports CFTC publishes per calendar year.
# (b) Constant because the publication cadence is set by CFTC, not by the
#     operator; missing weeks (holidays) shorten the window naturally.
# (c) Only update if CFTC moves to a different cadence (very unlikely).
_COT_REPORTS_PER_YEAR = 52

# (a) Z-score threshold above which commercials are treated as "extreme".
# (b) Constant because the contrarian gate is a coarse 1σ filter, not a
#     tunable risk parameter; the trader uses it as one input among many.
# (c) Replace via a function argument if the strategy ever needs a 2σ
#     stricter gate; keep this as the conservative default.
_COMMERCIAL_EXTREME_Z = 1.0


@dataclass(frozen=True)
class COTReport:
    """Single weekly CFTC TFF (Traders in Financial Futures) record."""

    currency: str  # e.g. "USD", "JPY", "EUR"
    report_date: date
    leveraged_funds_long: int
    leveraged_funds_short: int
    commercial_long: int
    commercial_short: int
    open_interest: int

    @property
    def leveraged_net(self) -> int:
        return self.leveraged_funds_long - self.leveraged_funds_short

    @property
    def commercial_net(self) -> int:
        return self.commercial_long - self.commercial_short


def cot_net_pct(report: COTReport) -> float | None:
    """Return leveraged-funds net positioning as % of open interest."""

    if report.open_interest <= 0:
        return None
    return report.leveraged_net / report.open_interest * 100.0


def _net_pct_history(history: Sequence[COTReport]) -> list[float]:
    pcts: list[float] = []
    for report in history:
        pct = cot_net_pct(report)
        if pct is not None:
            pcts.append(pct)
    return pcts


def cot_z_score(
    history: Sequence[COTReport],
    *,
    lookback_years: int = _COT_DEFAULT_LOOKBACK_YEARS,
) -> float | None:
    """Z-score of the most recent leveraged-funds net % vs trailing window."""

    pcts = _net_pct_history(history)
    if len(pcts) < 2:
        return None
    window = max(2, lookback_years * _COT_REPORTS_PER_YEAR)
    sample = pcts[-window:]
    if len(sample) < 2:
        return None
    last = sample[-1]
    base = sample[:-1]
    n = len(base)
    mean = sum(base) / n
    var = sum((v - mean) ** 2 for v in base) / (n - 1) if n > 1 else 0.0
    sd = var ** 0.5
    if sd == 0:
        return None
    return (last - mean) / sd


def cot_week_delta(history: Sequence[COTReport]) -> float | None:
    """Week-on-week change in leveraged-funds net %."""

    pcts = _net_pct_history(history)
    if len(pcts) < 2:
        return None
    return pcts[-1] - pcts[-2]


def cot_commercial_extreme(
    history: Sequence[COTReport],
    *,
    lookback_years: int = _COT_DEFAULT_LOOKBACK_YEARS,
) -> bool:
    """True when commercials sit at a contrarian extreme to leveraged funds.

    Computes both leveraged and commercial net % z-scores; flags True when
    they are on opposite sides AND ``|commercial_z| ≥ 1`` AND
    ``|leveraged_z| ≥ 1``.
    """

    if not history:
        return False
    last = history[-1]
    if last.open_interest <= 0:
        return False

    lev_pcts = _net_pct_history(history)
    com_pcts: list[float] = []
    for report in history:
        if report.open_interest <= 0:
            continue
        com_pcts.append(report.commercial_net / report.open_interest * 100.0)

    if len(lev_pcts) < 2 or len(com_pcts) < 2:
        return False

    window = max(2, lookback_years * _COT_REPORTS_PER_YEAR)

    def _z(values: Sequence[float]) -> float | None:
        sample = values[-window:]
        if len(sample) < 2:
            return None
        last_v = sample[-1]
        base = sample[:-1]
        n = len(base)
        m = sum(base) / n
        v = sum((x - m) ** 2 for x in base) / (n - 1) if n > 1 else 0.0
        sd = v ** 0.5
        if sd == 0:
            return None
        return (last_v - m) / sd

    lev_z = _z(lev_pcts)
    com_z = _z(com_pcts)
    if lev_z is None or com_z is None:
        return False
    if (lev_z * com_z) >= 0:
        return False  # same side, no contrarian setup
    return abs(lev_z) >= _COMMERCIAL_EXTREME_Z and abs(com_z) >= _COMMERCIAL_EXTREME_Z


__all__ = [
    "COTReport",
    "cot_net_pct",
    "cot_z_score",
    "cot_week_delta",
    "cot_commercial_extreme",
]
