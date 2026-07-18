"""Panel-wide market conditions reader (the operator's "read the market").

Per-pair regime labels are not a market read.  A pro reads the whole board:
how many pairs trend together (theme breadth), which currency is driving
(dominant theme), how much of the board is in high volatility, and whether
volatility is expanding.  This module aggregates per-pair classifications
and per-currency signed momentum into one sealed market-conditions
snapshot — the machine pre-read that feeds the AI trader's layer-2 read,
the family router, and state-conditional tactics.  Closed candles only; no
order authority.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Mapping, Sequence

from quant_rabbit.regime_classifier_shadow import (
    RegimeClassifierError,
    classify_regime,
)

CONTRACT = "QR_MARKET_CONDITIONS_SNAPSHOT_V1"
MOMENTUM_LOOKBACK = 60  # minutes of closes for the currency momentum read
THEME_DOMINANCE_FLOOR = 1.5  # dominant currency must lead 2nd by this ratio


class MarketConditionsError(ValueError):
    """Raised when panel inputs are malformed."""


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_market_conditions(
    panel: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    as_of_utc: datetime,
    high_impact_event_active: bool = False,
) -> dict[str, Any]:
    """Read the whole board: per-pair cells plus market-wide aggregates."""

    if not panel:
        raise MarketConditionsError("a non-empty pair panel is required")

    pair_states: dict[str, dict[str, Any]] = {}
    regime_counts: dict[str, int] = {}
    high_vol = 0
    classified = 0
    currency_momentum: dict[str, list[float]] = {}
    for pair, candles in sorted(panel.items()):
        parts = str(pair).upper().split("_")
        if len(parts) != 2:
            raise MarketConditionsError(f"pair identity is invalid: {pair!r}")
        try:
            state = classify_regime(
                candles, as_of_utc=as_of_utc,
                high_impact_event_active=high_impact_event_active,
            )
        except RegimeClassifierError:
            pair_states[pair.upper()] = {"regime": None, "vol_state": None}
            continue
        classified += 1
        regime = state["regime"]
        pair_states[pair.upper()] = {
            "regime": regime,
            "vol_state": state["vol_state"],
            "confidence": state["confidence"],
        }
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
        high_vol += int(state["vol_state"] == "HIGH")
        closes = [float(c["close"]) for c in candles]
        window = closes[-MOMENTUM_LOOKBACK:]
        if len(window) >= 2 and window[0] > 0:
            signed = (window[-1] / window[0]) - 1.0
            base, quote = parts
            currency_momentum.setdefault(base, []).append(signed)
            currency_momentum.setdefault(quote, []).append(-signed)

    if classified == 0:
        raise MarketConditionsError("no pair could be classified")

    momentum = {
        currency: round(sum(values) / len(values), 9)
        for currency, values in sorted(currency_momentum.items())
        if values
    }
    # Theme score SUMS signed contributions: a currency confirmed across many
    # pairs outranks one seen once with the same per-pair move — breadth is
    # what makes a theme, not a single loud pair.
    theme_score = {
        currency: round(sum(values), 9)
        for currency, values in sorted(currency_momentum.items())
        if values
    }
    ranked = sorted(theme_score.items(), key=lambda item: -abs(item[1]))
    dominant = None
    if ranked:
        leader = ranked[0]
        runner_up_abs = abs(ranked[1][1]) if len(ranked) > 1 else 0.0
        if runner_up_abs == 0.0 or abs(leader[1]) / max(runner_up_abs, 1e-12) >= THEME_DOMINANCE_FLOOR:
            dominant = {
                "currency": leader[0],
                "direction": "STRONG" if leader[1] > 0 else "WEAK",
                "theme_score": leader[1],
                "confirming_pairs": len(currency_momentum.get(leader[0], [])),
            }

    trend_breadth = regime_counts.get("TREND", 0) / classified
    dominant_regime = max(regime_counts, key=regime_counts.get)
    board_reading = (
        "EVENT_BOARD"
        if high_impact_event_active
        else "THEMED_TREND_BOARD"
        if dominant is not None and trend_breadth >= 0.3
        else "BROAD_TREND_BOARD"
        if trend_breadth >= 0.5
        else "COMPRESSED_BOARD"
        if regime_counts.get("SQUEEZE", 0) / classified >= 0.4
        else "MIXED_RANGE_BOARD"
    )

    body: dict[str, Any] = {
        "contract": CONTRACT,
        "schema_version": 1,
        "as_of_utc": as_of_utc.isoformat(),
        "classified_pairs": classified,
        "pair_states": pair_states,
        "regime_counts": dict(sorted(regime_counts.items())),
        "dominant_regime": dominant_regime,
        "trend_breadth": round(trend_breadth, 9),
        "high_vol_share": round(high_vol / classified, 9),
        "currency_momentum": momentum,
        "dominant_theme": dominant,
        "board_reading": board_reading,
        "feeds": [
            "CODEX_AI_TRADER layer-2 read (machine pre-read)",
            "regime_family_router cell selection",
            "state-conditional tactic switching",
        ],
        "order_authority": "NONE",
        "live_permission": False,
    }
    return {**body, "snapshot_sha256": _canonical_sha(body)}
