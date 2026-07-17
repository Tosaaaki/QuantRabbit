"""Deterministic regime + volatility classifier (completeness gap #1).

The whole all-weather grid (W2 family GO/CAUTION, W21's regime x vol cells)
is conditioned on a regime label — yet nothing measured it; the brain
consumed a *declared* regime.  This closes that hole: from closed candles
only, it emits {TREND / RANGE / SQUEEZE / EVENT, vol_state, confidence}
with sealed component values.  It is the operator's "measure the state,
not the clock" rule in code.

Causality: every candle must close strictly before the decision clock;
timestamps must be chronological and unique.  No indicator reads a bar the
decision could not yet see.  Classification grants no order authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from statistics import fmean, pstdev
from typing import Any, Mapping, Sequence

CONTRACT = "QR_REGIME_VOL_CLASSIFIER_V1"
# Pre-declared thresholds (not fit to any window's outcome).
SHORT_WINDOW = 20
VOL_HISTORY_WINDOW = 120
BB_K = 2.0
SQUEEZE_BB_WIDTH_FLOOR = 0.0025  # band width as a fraction of price
TREND_EFFICIENCY_FLOOR = 0.35  # Kaufman efficiency ratio
HIGH_VOL_PERCENTILE = 0.60
REGIMES = ("TREND", "RANGE", "SQUEEZE", "EVENT")
VOL_STATES = ("LOW", "HIGH")


class RegimeClassifierError(ValueError):
    """Raised when classifier inputs are malformed or non-causal."""


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _closes(candles: Sequence[Mapping[str, Any]], as_of: datetime) -> list[float]:
    closes: list[float] = []
    previous: datetime | None = None
    for candle in candles:
        stamp = candle.get("time")
        if not isinstance(stamp, datetime) or stamp.tzinfo is None:
            raise RegimeClassifierError("candle time must be timezone-aware")
        stamp = stamp.astimezone(timezone.utc)
        if stamp >= as_of:
            raise RegimeClassifierError("candle must close strictly before decision")
        if previous is not None and stamp <= previous:
            raise RegimeClassifierError("candles must be chronological and unique")
        previous = stamp
        close = candle.get("close")
        if isinstance(close, bool) or not isinstance(close, (int, float)):
            raise RegimeClassifierError("candle close must be a number")
        value = float(close)
        if not math.isfinite(value) or value <= 0.0:
            raise RegimeClassifierError("candle close must be a positive finite price")
        closes.append(value)
    return closes


def _efficiency_ratio(closes: Sequence[float]) -> float:
    net = abs(closes[-1] - closes[0])
    path = sum(abs(closes[i] - closes[i - 1]) for i in range(1, len(closes)))
    if path <= 0.0:
        return 0.0
    return net / path


def _bb_width(closes: Sequence[float]) -> float:
    mean = fmean(closes)
    if mean <= 0.0:
        return 0.0
    return 2.0 * BB_K * pstdev(closes) / mean


def _realized_vol(closes: Sequence[float]) -> float:
    returns = [
        (closes[i] / closes[i - 1]) - 1.0 for i in range(1, len(closes))
    ]
    return pstdev(returns) if len(returns) >= 2 else 0.0


def classify_regime(
    candles: Sequence[Mapping[str, Any]],
    *,
    as_of_utc: datetime,
    high_impact_event_active: bool = False,
) -> dict[str, Any]:
    """Classify the current regime and volatility from closed candles only."""

    if as_of_utc.tzinfo is None:
        raise RegimeClassifierError("decision clock must be timezone-aware")
    as_of = as_of_utc.astimezone(timezone.utc)
    closes = _closes(candles, as_of)
    if len(closes) < VOL_HISTORY_WINDOW:
        raise RegimeClassifierError(
            "insufficient closed-candle history for classification"
        )

    window = closes[-SHORT_WINDOW:]
    efficiency = _efficiency_ratio(window)
    bb_width = _bb_width(window)
    current_vol = _realized_vol(window)

    # Volatility percentile: rank the current short-window vol against the
    # rolling distribution of the same measure across recent history.
    vol_samples: list[float] = []
    for end in range(SHORT_WINDOW, len(closes) + 1):
        vol_samples.append(_realized_vol(closes[end - SHORT_WINDOW:end]))
    ranked = sorted(vol_samples)
    below = sum(1 for value in ranked if value < current_vol)
    percentile = below / len(ranked) if ranked else 0.0
    vol_state = "HIGH" if percentile >= HIGH_VOL_PERCENTILE else "LOW"

    if high_impact_event_active:
        regime = "EVENT"
        confidence = 1.0
    elif bb_width < SQUEEZE_BB_WIDTH_FLOOR:
        regime = "SQUEEZE"
        confidence = round(
            min(1.0, (SQUEEZE_BB_WIDTH_FLOOR - bb_width) / SQUEEZE_BB_WIDTH_FLOOR), 9
        )
    elif efficiency >= TREND_EFFICIENCY_FLOOR:
        regime = "TREND"
        confidence = round(
            min(1.0, (efficiency - TREND_EFFICIENCY_FLOOR) / (1.0 - TREND_EFFICIENCY_FLOOR)),
            9,
        )
    else:
        regime = "RANGE"
        confidence = round(
            min(1.0, (TREND_EFFICIENCY_FLOOR - efficiency) / TREND_EFFICIENCY_FLOOR), 9
        )

    body: dict[str, Any] = {
        "contract": CONTRACT,
        "schema_version": 1,
        "as_of_utc": as_of.isoformat(),
        "regime": regime,
        "vol_state": vol_state,
        "confidence": confidence,
        "components": {
            "efficiency_ratio": round(efficiency, 9),
            "bb_width_fraction": round(bb_width, 12),
            "short_window_realized_vol": round(current_vol, 12),
            "vol_percentile": round(percentile, 9),
        },
        "thresholds": {
            "trend_efficiency_floor": TREND_EFFICIENCY_FLOOR,
            "squeeze_bb_width_floor": SQUEEZE_BB_WIDTH_FLOOR,
            "high_vol_percentile": HIGH_VOL_PERCENTILE,
            "short_window": SHORT_WINDOW,
            "vol_history_window": VOL_HISTORY_WINDOW,
        },
        "measured_from_candles_not_clock": True,
        "uses_post_decision_information": False,
        "order_authority": "NONE",
        "live_permission": False,
    }
    return {**body, "classification_sha256": _canonical_sha(body)}
