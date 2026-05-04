"""Generic Z-score / percentile-rank normalization layer.

The naive failure mode this module corrects: citing raw indicator values
("RSI 70 is overbought") without context. RSI 70 on EUR_USD M5 in a quiet
Tokyo session is a different signal than RSI 70 on GBP_JPY M5 during
London volatility. Per `docs/research/03-quant-ensemble-regime.md` §2 step 2,
the highest-leverage cheap fix is converting every indicator value to a
rolling Z-score and rolling percentile rank per-pair per-timeframe, so the
trader reasons about *deviation from this market's recent self*, not about
absolute thresholds memorized from textbooks.

Pure-stdlib, no numpy/pandas. Pure functions, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


# ---------------------------------------------------------------------------
# Window defaults — per AGENT_CONTRACT §3.5 (a)/(b)/(c)
# ---------------------------------------------------------------------------
#
# DEFAULT_Z_WINDOW = 200
#   (a) Market reality: ~one trading day of M5 bars (288), reduced to 200 to
#       stay reactive when the prior day's regime decays. Long enough for the
#       sample mean / stdev to stabilize beyond intraday noise, short enough
#       that a Z-score reflects "current regime", not "year-old regime".
#   (b) Constant rather than market-derived: this is operator policy
#       (responsiveness vs stability), not a market-measurable quantity. The
#       trader chooses how much history "counts as recent" — the market does
#       not tell us.
#   (c) Replace via: explicit `window=` kwarg on `rolling_z` /
#       `normalize_indicator`. For H1 timeframe, pass window=120 (~5 days);
#       for D1, pass window=60 (~3 months).
DEFAULT_Z_WINDOW: int = 200

# DEFAULT_PCT_WINDOW = 500
#   (a) Market reality: ~1.7 trading days of M5 bars. A wider lookback than
#       the Z-window because percentile rank is more sensitive to outlier
#       influence at small-N (a 50-bar window can place "current value" at
#       the 100th percentile from a single outlier two days ago). 500 buys
#       statistical stability while still tracking the evolving distribution.
#   (b) Constant rather than market-derived: same reason as DEFAULT_Z_WINDOW.
#       The percentile horizon is policy.
#   (c) Replace via: explicit `window=` kwarg on `rolling_percentile_rank` /
#       `normalize_indicator`. Macrosynergy's Hurst study uses 252 days for
#       D1 — pass window=252 when working on D1 series.
DEFAULT_PCT_WINDOW: int = 500


@dataclass(frozen=True)
class NormalizedValue:
    """Normalized view of a single indicator observation.

    Attributes:
        raw: Original value (may be None if the underlying indicator was None).
        z_score: Rolling Z-score over the trailing `window_z` valid samples.
            None if the window is underfilled.
        percentile: 0..100 rank of `raw` within the trailing `window_pct`
            valid samples. None if the window is underfilled.
    """

    raw: float | None
    z_score: float | None
    percentile: float | None


def _trailing_valid(values: Sequence[float | None], end_idx: int, window: int) -> list[float]:
    """Collect up to `window` non-None values ending at `end_idx` (inclusive).

    Walks backwards from `end_idx`, skipping None entries, and stops once
    `window` valid samples have been collected. Returns the slice in
    chronological order (oldest -> newest).
    """
    out: list[float] = []
    i = end_idx
    while i >= 0 and len(out) < window:
        v = values[i]
        if v is not None:
            out.append(float(v))
        i -= 1
    out.reverse()
    return out


def _z_from_window(window_values: list[float], current: float) -> float | None:
    """Z-score of `current` against the supplied window. None if degenerate."""
    n = len(window_values)
    if n < 2:
        return None
    m = sum(window_values) / n
    var = sum((v - m) ** 2 for v in window_values) / n
    if var <= 0.0:
        return 0.0
    sd = var ** 0.5
    return (current - m) / sd


def _percentile_from_window(window_values: list[float], current: float) -> float | None:
    """Percentile rank (0..100) of `current` against the window.

    Uses the "fraction at-or-below current" convention (`<=`), so the lowest
    sample yields >0 and the highest sample yields 100. None if window empty.
    """
    n = len(window_values)
    if n == 0:
        return None
    below_or_eq = sum(1 for v in window_values if v <= current)
    return 100.0 * below_or_eq / n


def rolling_z(
    series: Sequence[float | None],
    *,
    window: int = DEFAULT_Z_WINDOW,
) -> list[float | None]:
    """Per-bar rolling Z-score over the trailing `window` non-None samples.

    Returns a list the same length as `series`. Each entry is the Z-score of
    the value at that index against the window of valid samples ending at
    that index. Entries are None when:
      - the value at that index is None,
      - fewer than 2 valid samples are available in the trailing window,
      - the window stdev is exactly 0 with `current != mean` (degenerate).

    Skips None values cleanly: the trailing window is built from the most
    recent `window` non-None samples, regardless of how many None entries
    intervene.
    """
    if window < 2:
        # Need at least 2 samples for a defined Z; smaller windows are
        # nonsensical. Fail loud rather than silently returning all-None.
        raise ValueError(f"rolling_z requires window >= 2, got {window}")
    n = len(series)
    out: list[float | None] = [None] * n
    for i in range(n):
        cur = series[i]
        if cur is None:
            continue
        # Use the trailing window ending one bar earlier as the reference
        # distribution; this avoids leaking the current value into its own
        # mean/stdev (which would shrink Z toward zero).
        ref = _trailing_valid(series, i - 1, window)
        out[i] = _z_from_window(ref, float(cur))
    return out


def rolling_percentile_rank(
    series: Sequence[float | None],
    *,
    window: int = DEFAULT_PCT_WINDOW,
) -> list[float | None]:
    """Per-bar rolling percentile rank (0..100) over trailing `window` samples.

    See class docstring for window semantics. Returns same-length list as
    `series` with None at indices where:
      - the value at that index is None, or
      - fewer than 2 valid samples are available in the trailing window
        (a single-sample percentile is meaningless).

    Includes the current bar in the reference window (this matches the
    Macrosynergy convention and is consistent with how operators describe
    "current ATR is in the 80th percentile of the last 500 bars").
    """
    if window < 2:
        raise ValueError(f"rolling_percentile_rank requires window >= 2, got {window}")
    n = len(series)
    out: list[float | None] = [None] * n
    for i in range(n):
        cur = series[i]
        if cur is None:
            continue
        ref = _trailing_valid(series, i, window)
        if len(ref) < 2:
            continue
        out[i] = _percentile_from_window(ref, float(cur))
    return out


def normalize_indicator(
    values: Sequence[float | None],
    *,
    window_z: int = DEFAULT_Z_WINDOW,
    window_pct: int = DEFAULT_PCT_WINDOW,
) -> list[NormalizedValue]:
    """Normalize an indicator series into per-bar `NormalizedValue` records.

    Combines `rolling_z` and `rolling_percentile_rank` so callers can read
    both views in a single pass. Returns a list of length `len(values)`;
    each entry preserves the raw input even when normalization is None
    (window underfilled), so downstream code can distinguish "value
    missing" from "context missing".
    """
    zs = rolling_z(values, window=window_z)
    pcts = rolling_percentile_rank(values, window=window_pct)
    out: list[NormalizedValue] = []
    for raw, z, p in zip(values, zs, pcts):
        out.append(
            NormalizedValue(
                raw=None if raw is None else float(raw),
                z_score=z,
                percentile=p,
            )
        )
    return out
