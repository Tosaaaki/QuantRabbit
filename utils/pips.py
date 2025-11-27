"""
Utility helpers for working with pip / price conversions and trade costs.

At the moment QuantRabbit only trades USD/JPY, but the helpers are written in a
way that keeps the logic explicit and easy to extend to other instruments later
on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

_PIP_SIZE_BY_INSTRUMENT: dict[str, float] = {
    # 1 pip for USD/JPY is 0.01 JPY
    "USD_JPY": 0.01,
}


def pip_size(instrument: str) -> float:
    """
    Return the pip size (price delta that equals 1 pip) for the given instrument.
    Defaults to 0.0001 if the instrument is unknown.
    """
    return _PIP_SIZE_BY_INSTRUMENT.get(instrument, 0.0001)


def to_pips(instrument: str, price_delta: float) -> float:
    """
    Convert a price delta into pips for the instrument.
    """
    size = pip_size(instrument)
    if size == 0:
        raise ValueError(f"pip size for instrument {instrument!r} is zero")
    return price_delta / size


def from_pips(instrument: str, pips: float) -> float:
    """
    Convert a pip delta into a price delta for the instrument.
    """
    return pips * pip_size(instrument)


def pip_value(instrument: str, units: int) -> float:
    """
    Return the monetary value (in quote currency) of one pip for the given
    instrument and position size.
    """
    return abs(units) * pip_size(instrument)


@dataclass
class CostBreakdown:
    """
    Detailed breakdown of round-trip trading costs expressed in pips.
    """

    entry_spread_pips: float = 0.0
    exit_spread_pips: Optional[float] = None
    slippage_pips: float = 0.0
    commission_currency: float = 0.0

    def total_cost_pips(self, instrument: str, units: Optional[int] = None) -> float:
        """
        Calculate the total cost in pips. Commission is converted to pips if
        both `units` and a non-zero commission are provided.
        """
        entry = max(self.entry_spread_pips, 0.0)
        exit_spread = self.exit_spread_pips
        if exit_spread is None:
            exit_spread = entry
        else:
            exit_spread = max(exit_spread, 0.0)
        slip = max(self.slippage_pips, 0.0)
        commission_pips = 0.0
        if units and self.commission_currency:
            pip_val = pip_value(instrument, units)
            if pip_val:
                commission_pips = self.commission_currency / pip_val
        return entry + exit_spread + slip + commission_pips


def apply_costs(
    instrument: str,
    gross_pips: float,
    *,
    costs: Optional[CostBreakdown] = None,
    units: Optional[int] = None,
) -> tuple[float, float]:
    """
    Apply the provided cost profile to the gross profit (in pips) and return the
    net profit together with the total cost in pips.
    """
    if costs is None:
        return gross_pips, 0.0
    total_cost = costs.total_cost_pips(instrument, units=units)
    net = gross_pips - total_cost
    return net, total_cost

