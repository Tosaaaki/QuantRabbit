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
# Pre-declared thresholds.  All boundaries are scale-free (the 0..1 Kaufman
# efficiency ratio) or RELATIVE (percentile of the instrument's own recent
# distribution) — never an absolute pip magic number, which is the same
# non-generalizing mistake as a time-of-day rule.
SHORT_WINDOW = 20
VOL_HISTORY_WINDOW = 120
BB_K = 2.0
TREND_EFFICIENCY_FLOOR = 0.35  # Kaufman efficiency ratio (scale-free)
SQUEEZE_BB_WIDTH_PERCENTILE = 0.20  # compressed relative to own history
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


def _percentile_rank(samples: Sequence[float], current: float) -> float:
    if not samples:
        return 0.0
    below = sum(1 for value in samples if value < current)
    return below / len(samples)


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

    # Both percentiles rank the current short-window measure against the
    # rolling distribution of the SAME measure over recent history — relative,
    # so the boundaries generalize across instruments and timeframes.
    vol_samples: list[float] = []
    bb_samples: list[float] = []
    for end in range(SHORT_WINDOW, len(closes) + 1):
        segment = closes[end - SHORT_WINDOW:end]
        vol_samples.append(_realized_vol(segment))
        bb_samples.append(_bb_width(segment))
    vol_percentile = _percentile_rank(vol_samples, current_vol)
    bb_percentile = _percentile_rank(bb_samples, bb_width)
    vol_state = "HIGH" if vol_percentile >= HIGH_VOL_PERCENTILE else "LOW"

    # Directional strength (scale-free) decides trend first; among non-trending
    # states, band compression relative to own history splits squeeze vs range.
    if high_impact_event_active:
        regime = "EVENT"
        confidence = 1.0
    elif efficiency >= TREND_EFFICIENCY_FLOOR:
        regime = "TREND"
        confidence = round(
            min(1.0, (efficiency - TREND_EFFICIENCY_FLOOR) / (1.0 - TREND_EFFICIENCY_FLOOR)),
            9,
        )
    elif bb_percentile < SQUEEZE_BB_WIDTH_PERCENTILE:
        regime = "SQUEEZE"
        confidence = round(
            min(1.0, (SQUEEZE_BB_WIDTH_PERCENTILE - bb_percentile) / SQUEEZE_BB_WIDTH_PERCENTILE),
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
            "bb_width_percentile": round(bb_percentile, 9),
            "short_window_realized_vol": round(current_vol, 12),
            "vol_percentile": round(vol_percentile, 9),
        },
        "thresholds": {
            "trend_efficiency_floor": TREND_EFFICIENCY_FLOOR,
            "squeeze_bb_width_percentile": SQUEEZE_BB_WIDTH_PERCENTILE,
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
