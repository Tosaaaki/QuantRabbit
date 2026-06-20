"""Forecast precision helpers for live-entry gating.

The projection ledger can show very high raw hit rates for targets that are
too small to monetize after spread. Live gates therefore need both statistical
confidence and an execution-aware target-width check.
"""

from __future__ import annotations

import math
import re
from typing import Any


_PIP_RE = re.compile(r"(?P<pips>\d+(?:\.\d+)?)\s*pip", re.IGNORECASE)


def wilson_lower_bound(successes: int, trials: int, *, z: float = 1.96) -> float:
    """Return the Wilson lower confidence bound for a binomial hit rate."""
    if trials <= 0:
        return 0.0
    successes = max(0, min(int(successes), int(trials)))
    p_hat = successes / float(trials)
    denom = 1.0 + (z * z / trials)
    centre = p_hat + (z * z / (2.0 * trials))
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + (z * z / (4.0 * trials))) / trials)
    return max(0.0, min(1.0, (centre - margin) / denom))


def successes_from_hit_rate(hit_rate: float | None, samples: int | None) -> int | None:
    """Convert a rounded hit-rate bucket back to an integer success count."""
    if hit_rate is None or samples is None or samples <= 0:
        return None
    bounded = max(0.0, min(1.0, float(hit_rate)))
    return max(0, min(int(samples), int(round(bounded * int(samples)))))


def hit_rate_wilson_lower(hit_rate: float | None, samples: int | None) -> float | None:
    successes = successes_from_hit_rate(hit_rate, samples)
    if successes is None or samples is None or samples <= 0:
        return None
    return wilson_lower_bound(successes, int(samples))


def target_pips_from_text(text: str | None) -> float | None:
    """Extract the first '<number>pip' distance from rationale text."""
    if not text:
        return None
    match = _PIP_RE.search(str(text))
    if match is None:
        return None
    try:
        return float(match.group("pips"))
    except (TypeError, ValueError):
        return None


def target_pips_from_payload(payload: Any) -> float | None:
    """Read target-pip distance from a support signal payload."""
    if not isinstance(payload, dict):
        return None
    for key in ("target_pips", "target_distance_pips", "reward_pips"):
        try:
            value = payload.get(key)
        except AttributeError:
            value = None
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed >= 0.0:
            return parsed
    return target_pips_from_text(str(payload.get("rationale") or ""))


def support_signal_clears_live_precision(
    payload: dict[str, Any],
    *,
    min_wilson_lower: float,
    min_samples: int,
    min_target_pips: float,
) -> bool:
    """Return whether a support signal is statistically and economically usable.

    `liquidity_sweep_*` is target-distance based. If its target-pip width is
    missing or inside the configured floor, the signal is treated as unproven
    for live support even if its raw touch hit-rate is high.
    """
    try:
        samples = int(payload.get("samples", 0) or 0)
    except (TypeError, ValueError):
        return False
    if samples < int(min_samples):
        return False
    try:
        hit_rate = float(payload.get("hit_rate"))
    except (TypeError, ValueError):
        return False
    lower = hit_rate_wilson_lower(hit_rate, samples)
    if lower is None or lower < float(min_wilson_lower):
        return False
    name = str(payload.get("name") or payload.get("calibration_name") or "").lower()
    target_pips = target_pips_from_payload(payload)
    if "liquidity_sweep" in name:
        if target_pips is None:
            return False
        return target_pips >= float(min_target_pips)
    if target_pips is not None and target_pips < float(min_target_pips):
        return False
    return True
