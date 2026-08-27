"""Timestamp-only compatibility wrapper for the frozen V26 runner.

This module does not change the preregistered signal, execution, cost,
inventory, evaluation, or admission logic.  It only replaces the frozen
runner's timestamp parser so Python 3.10 can consume the source's nine-digit
UTC fractional seconds when the sub-microsecond digits are exactly zero.

The wrapper is preparation only.  Its existence does not authorize another
official execution.
"""

from __future__ import annotations

import re
from datetime import datetime

import run_causal_min_spread_representative_v26 as frozen_v26


_UTC_TIMESTAMP = re.compile(
    r"^(?P<head>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?Z$"
)


def parse_v26_utc_timestamp(value: str) -> datetime:
    """Parse V26 UTC timestamps without silently losing nonzero precision."""
    match = _UTC_TIMESTAMP.fullmatch(value)
    if match is None:
        raise ValueError(f"V26 timestamp is not canonical UTC: {value}")
    fraction = match.group("fraction") or ""
    if len(fraction) > 6 and any(character != "0" for character in fraction[6:]):
        raise ValueError(f"V26 timestamp has nonzero sub-microsecond precision: {value}")
    microseconds = fraction[:6].ljust(6, "0")
    normalized = f"{match.group('head')}.{microseconds}+00:00"
    return datetime.fromisoformat(normalized)


def install_timestamp_compatibility() -> None:
    """Install the sole recovery change into the frozen V26 module."""
    frozen_v26.parse_time = parse_v26_utc_timestamp


def main() -> int:
    raise RuntimeError(
        "V26 recovery is not executable until a separate explicit authorization "
        "artifact and one-shot launcher are registered"
    )


if __name__ == "__main__":
    raise SystemExit(main())
