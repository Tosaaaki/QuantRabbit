"""Per-currency net-exposure guard (weakness ledger W12).

Twenty-eight pair cells are not independent: one USD theme moves most of
them together, so per-pair caps understate real portfolio risk, while a
deliberate same-theme basket (an operator-proven pattern) has no first-class
representation.  This module aggregates any position set into signed
per-currency exposure fractions and admits a candidate order only when
every touched currency stays inside its declared cap.  Pure functions,
no broker surface, no order authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

GUARD_POLICY = "PER_CURRENCY_NET_NAV_EXPOSURE_CAP_V1"
DEFAULT_CURRENCY_CAP_FRACTION = 0.5
G8_CURRENCIES = ("AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NZD", "USD")


@dataclass(frozen=True, slots=True)
class ExposureDecision:
    admitted: bool
    reason: str
    breached_currencies: tuple[str, ...]
    projected_exposure: Mapping[str, float]

    def payload(self) -> dict[str, Any]:
        return {
            "policy": GUARD_POLICY,
            "admitted": self.admitted,
            "reason": self.reason,
            "breached_currencies": list(self.breached_currencies),
            "projected_exposure": dict(self.projected_exposure),
        }


def _split_pair(pair: str) -> tuple[str, str]:
    parts = str(pair).upper().split("_")
    if len(parts) != 2 or not all(len(p) == 3 and p.isalpha() for p in parts):
        raise ValueError(f"pair identity is invalid: {pair!r}")
    base, quote = parts
    if base == quote:
        raise ValueError(f"pair identity is degenerate: {pair!r}")
    return base, quote


def _validated_fraction(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{label} must be finite")
    return number


def net_currency_exposure(
    positions: Iterable[Mapping[str, Any]],
) -> dict[str, float]:
    """Sum signed NAV-fraction exposure per currency.

    Each position needs ``pair`` (``AAA_BBB``), ``side`` (LONG/SHORT), and
    ``nav_exposure_fraction`` (position notional as a NAV fraction, > 0).
    A LONG on AAA_BBB is +AAA/-BBB; a SHORT is the mirror.
    """

    totals: dict[str, float] = {}
    for position in positions:
        base, quote = _split_pair(str(position.get("pair")))
        side = str(position.get("side", "")).upper()
        if side not in {"LONG", "SHORT"}:
            raise ValueError("position side must be LONG or SHORT")
        fraction = _validated_fraction(
            position.get("nav_exposure_fraction"), "nav_exposure_fraction"
        )
        if fraction <= 0:
            raise ValueError("nav_exposure_fraction must be positive")
        sign = 1.0 if side == "LONG" else -1.0
        totals[base] = totals.get(base, 0.0) + sign * fraction
        totals[quote] = totals.get(quote, 0.0) - sign * fraction
    return {currency: round(value, 12) for currency, value in totals.items()}


def evaluate_currency_exposure(
    open_positions: Iterable[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    *,
    currency_cap_fraction: float = DEFAULT_CURRENCY_CAP_FRACTION,
) -> ExposureDecision:
    """Admit the candidate only if no currency's |net| exceeds the cap."""

    cap = _validated_fraction(currency_cap_fraction, "currency_cap_fraction")
    if cap <= 0:
        raise ValueError("currency_cap_fraction must be positive")
    projected = net_currency_exposure([*open_positions, candidate])
    breached = tuple(
        sorted(
            currency
            for currency, value in projected.items()
            if abs(value) > cap + 1e-12
        )
    )
    if breached:
        return ExposureDecision(
            admitted=False,
            reason="CURRENCY_EXPOSURE_CAP_EXCEEDED",
            breached_currencies=breached,
            projected_exposure=projected,
        )
    return ExposureDecision(
        admitted=True,
        reason="WITHIN_PER_CURRENCY_CAP",
        breached_currencies=(),
        projected_exposure=projected,
    )
