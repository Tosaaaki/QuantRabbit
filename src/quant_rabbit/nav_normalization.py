"""NAV-fraction normalization for scorecards (weakness ledger W13).

Pips aggregated across pairs distort selection: one JPY pip is not one USD
pip.  These helpers convert realized pips into NAV fractions through
explicit, caller-supplied conversion rates — no hidden market data, no
guessing.  Selection keys move to the NAV side; pips stay for pair-local
diagnostics.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from quant_rabbit.instruments import instrument_pip_factor


def _finite(value: Any, label: str, *, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return number


def pips_to_nav_fraction(
    *,
    pair: str,
    realized_pips: float,
    units: float,
    quote_to_account_rate: float,
    nav_account_currency: float,
) -> float:
    """Convert one trade's realized pips into a NAV fraction.

    ``quote_to_account_rate`` converts the pair's quote currency into the
    account currency (1.0 when they are the same); the caller must source it
    from the same sealed data as the trade itself.
    """

    pip_size = 1.0 / float(instrument_pip_factor(pair))
    pips = _finite(realized_pips, "realized_pips", positive=False)
    size = _finite(units, "units")
    rate = _finite(quote_to_account_rate, "quote_to_account_rate")
    nav = _finite(nav_account_currency, "nav_account_currency")
    quote_pnl = pips * pip_size * size
    return round(quote_pnl * rate / nav, 12)


def nav_columns_for_metrics(
    *,
    pair: str,
    net_pips: float,
    trade_count: int,
    units: float,
    quote_to_account_rate: float,
    nav_account_currency: float,
) -> Mapping[str, float]:
    """NAV-side companion columns for one pair's pip metrics."""

    if isinstance(trade_count, bool) or not isinstance(trade_count, int) or trade_count < 0:
        raise ValueError("trade_count must be a non-negative integer")
    total = pips_to_nav_fraction(
        pair=pair,
        realized_pips=net_pips,
        units=units,
        quote_to_account_rate=quote_to_account_rate,
        nav_account_currency=nav_account_currency,
    )
    return {
        "net_nav_fraction": total,
        "mean_nav_fraction_per_trade": round(total / trade_count, 12)
        if trade_count
        else 0.0,
    }
