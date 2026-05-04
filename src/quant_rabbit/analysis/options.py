"""FX option-skew adapter — risk reversals + ATM implied vol.

Real-time implied vol and risk-reversal data for FX option markets is sold
by Bloomberg (BVOL/RR), Refinitiv, and ICAP. There is no high-quality free
public feed. To stay honest about that constraint, this module ships a
clean adapter shell with two responsibilities:

1. If `QR_OPTION_SKEW_PROVIDER` env var is set to a recognized value, attempt
   to call the configured provider plug-in. None are bundled by default —
   wire one up when a vendor account becomes available.
2. Otherwise emit `MISSING_OPTION_SKEW_FEED` with severity `BLOCK` per
   §3.5 of the agent contract. The trader treats option-skew data as
   unavailable rather than guessed.

The adapter format is fixed so that any future implementation only has to
return `OptionSkewSnapshot` records.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Sequence


@dataclass(frozen=True)
class OptionSkewReading:
    pair: str
    tenor: str  # e.g. "1W" / "1M" / "3M"
    atm_iv: float | None  # at-the-money implied vol (annualized, %)
    rr_25d: float | None  # 25-delta risk reversal (call IV - put IV)
    bf_25d: float | None  # 25-delta butterfly
    timestamp_utc: str | None
    source: str | None
    issue: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "pair": self.pair,
            "tenor": self.tenor,
            "atm_iv": self.atm_iv,
            "rr_25d": self.rr_25d,
            "bf_25d": self.bf_25d,
            "timestamp_utc": self.timestamp_utc,
            "source": self.source,
            "issue": self.issue,
        }


@dataclass(frozen=True)
class OptionSkewSnapshot:
    generated_at_utc: str
    provider: str | None
    readings: tuple[OptionSkewReading, ...] = field(default_factory=tuple)
    issues: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at_utc": self.generated_at_utc,
            "provider": self.provider,
            "readings": [r.to_dict() for r in self.readings],
            "issues": list(self.issues),
        }


def build_option_skew_snapshot(*, pairs: Sequence[str], tenors: Sequence[str] = ("1W", "1M", "3M")) -> OptionSkewSnapshot:
    """Try the configured provider; otherwise return a MISSING-gated snapshot.

    To plug in a real provider, set `QR_OPTION_SKEW_PROVIDER=<key>` and
    register a callable in `_PROVIDERS`. The callable must accept (pairs,
    tenors) and return `Iterable[OptionSkewReading]`.
    """

    provider_key = os.environ.get("QR_OPTION_SKEW_PROVIDER", "").strip().lower() or None
    issues: list[str] = []
    readings: list[OptionSkewReading] = []

    if provider_key and provider_key in _PROVIDERS:
        try:
            for r in _PROVIDERS[provider_key](pairs, tenors):
                readings.append(r)
        except Exception as exc:
            issues.append(f"OPTION_SKEW_PROVIDER_ERROR ({provider_key}): {exc}")
    else:
        issues.append(
            "MISSING_OPTION_SKEW_FEED: no provider configured. Set "
            "QR_OPTION_SKEW_PROVIDER=<key> and register a feed adapter "
            "in src/quant_rabbit/analysis/options.py:_PROVIDERS."
        )
        for pair in pairs:
            for tenor in tenors:
                readings.append(OptionSkewReading(
                    pair=pair, tenor=tenor, atm_iv=None, rr_25d=None, bf_25d=None,
                    timestamp_utc=None, source=None,
                    issue="MISSING_OPTION_SKEW_FEED",
                ))

    return OptionSkewSnapshot(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        provider=provider_key,
        readings=tuple(readings),
        issues=tuple(issues),
    )


# Provider registry. Real implementations should validate credentials, fetch
# from the vendor API, normalize timezones, and emit `OptionSkewReading`.
# Example signature:
#
#     def _bloomberg_provider(pairs, tenors):
#         for pair in pairs:
#             for tenor in tenors:
#                 yield OptionSkewReading(...)
#
_PROVIDERS: dict[str, object] = {}
