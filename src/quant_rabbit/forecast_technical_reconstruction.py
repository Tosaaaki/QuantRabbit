"""Causal technical-feature reconstruction for legacy forecast learning.

New forecasts freeze their signed point-in-time technical context.  Older
forecast rows predate that contract, but can still be evaluated against local
OANDA bid/ask candles.  This module derives a deliberately small set of model
features from *complete candles whose close is no later than the forecast
timestamp*.  Reconstructed features are labelled as such and never overwrite
any point-in-time technical evidence already present on a scored row.

The reconstruction is training evidence, not live entry permission.  Its
thresholds classify descriptive cohorts; they do not authorize an order.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from quant_rabbit.analysis.sessions import tag_bar
from quant_rabbit.instruments import instrument_pip_factor
from quant_rabbit.strategy.tf_weights import classify_situation


FORECAST_TECHNICAL_RECONSTRUCTION_CONTRACT = (
    "QR_FORECAST_TECHNICAL_RECONSTRUCTION_M1_V1"
)
FORECAST_TECHNICAL_RECONSTRUCTION_BASIS = (
    "PAST_COMPLETE_OANDA_BID_ASK_CANDLES_ONLY"
)
TECHNICAL_FEATURE_FIELDS = (
    "technical_regime",
    "technical_atr_band",
    "technical_spread_band",
    "technical_range_location_24h",
    "technical_structure_alignment",
    "technical_situation",
    "technical_selected_method",
    "technical_family_direction_alignment",
)
SUPPORTED_GRANULARITY = "M1"
# Data-quality floor, not a market threshold. OANDA omits no-tick buckets, so
# demanding all 1,440 nominal minutes would reject valid liquid FX history.
MIN_LOOKBACK_COVERAGE = 0.80
MAX_RECENT_CANDLE_LAG = timedelta(minutes=5)


@dataclass(frozen=True)
class _Bar:
    timestamp_utc: datetime
    o: float
    h: float
    l: float  # noqa: E741 - mirrors the OANDA OHLC low field
    c: float


def reconstruct_missing_technical_features(
    scored_rows: Sequence[Mapping[str, Any]],
    candles_by_pair: Mapping[str, Sequence[Any]],
    *,
    granularity: str = SUPPORTED_GRANULARITY,
    lookback_hours: float = 24.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fill wholly missing legacy technical fields from causal bid/ask bars.

    A row with any retained technical field is preserved in full.  This avoids
    silently producing a hybrid of signed point-in-time evidence and a later
    reconstruction.  The caller receives new dictionaries; input rows are not
    mutated.
    """

    normalized_granularity = str(granularity or "").strip().upper()
    if normalized_granularity != SUPPORTED_GRANULARITY:
        raise ValueError(
            f"technical reconstruction requires {SUPPORTED_GRANULARITY} candles"
        )
    try:
        lookback = timedelta(hours=float(lookback_hours))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("technical lookback hours must be numeric") from exc
    if lookback < timedelta(hours=24):
        raise ValueError("technical reconstruction requires at least 24 hours")

    candle_delta = timedelta(minutes=1)
    expected_candles = int(lookback.total_seconds() / candle_delta.total_seconds())
    minimum_candles = int(math.ceil(expected_candles * MIN_LOOKBACK_COVERAGE))
    indexed: dict[str, tuple[list[datetime], list[Any]]] = {}
    for pair, raw_candles in candles_by_pair.items():
        ordered = sorted(
            (
                candle
                for candle in raw_candles
                if isinstance(getattr(candle, "timestamp_utc", None), datetime)
            ),
            key=lambda candle: candle.timestamp_utc,
        )
        indexed[str(pair).upper()] = (
            [candle.timestamp_utc for candle in ordered],
            ordered,
        )

    counters = {
        "input_rows": len(scored_rows),
        "exact_or_partial_context_rows_preserved": 0,
        "reconstructed_rows": 0,
        "skipped_invalid_rows": 0,
        "skipped_missing_pair_history_rows": 0,
        "skipped_incomplete_lookback_rows": 0,
    }
    output: list[dict[str, Any]] = []
    for raw_row in scored_rows:
        row = dict(raw_row)
        if _has_retained_technical_evidence(row):
            counters["exact_or_partial_context_rows_preserved"] += 1
            output.append(row)
            continue
        pair = str(row.get("pair") or "").strip().upper()
        direction = str(
            row.get("forecast_direction") or row.get("direction") or ""
        ).strip().upper()
        forecast_at = _parse_utc(row.get("timestamp_utc"))
        if not pair or direction not in {"UP", "DOWN"} or forecast_at is None:
            counters["skipped_invalid_rows"] += 1
            output.append(row)
            continue
        times_and_candles = indexed.get(pair)
        if times_and_candles is None:
            counters["skipped_missing_pair_history_rows"] += 1
            output.append(row)
            continue
        times, pair_candles = times_and_candles
        # A candle is eligible only when its close is <= the forecast time.
        # For M1 data, an open at 10:29 closes at 10:30.
        end = bisect.bisect_right(times, forecast_at - candle_delta)
        start_at = forecast_at - lookback
        start = bisect.bisect_left(times, start_at)
        window = pair_candles[start:end]
        if not _lookback_is_complete(
            window,
            start_at=start_at,
            forecast_at=forecast_at,
            candle_delta=candle_delta,
            minimum_candles=minimum_candles,
        ):
            counters["skipped_incomplete_lookback_rows"] += 1
            output.append(row)
            continue
        features = _features_from_window(
            window,
            pair=pair,
            forecast_direction=direction,
            forecast_at=forecast_at,
            candle_delta=candle_delta,
        )
        reconstruction_body = {
            "contract": FORECAST_TECHNICAL_RECONSTRUCTION_CONTRACT,
            "basis": FORECAST_TECHNICAL_RECONSTRUCTION_BASIS,
            "pair": pair,
            "forecast_timestamp_utc": forecast_at.isoformat(),
            "granularity": normalized_granularity,
            "lookback_hours": float(lookback_hours),
            "first_candle_open_utc": window[0].timestamp_utc.isoformat(),
            "last_candle_close_utc": (
                window[-1].timestamp_utc + candle_delta
            ).isoformat(),
            "candle_count": len(window),
            "features": features,
        }
        row.update(features)
        row.update(
            {
                "technical_feature_source": "HISTORICAL_BID_ASK_RECONSTRUCTION",
                "technical_reconstruction_contract": (
                    FORECAST_TECHNICAL_RECONSTRUCTION_CONTRACT
                ),
                "technical_reconstruction_sha256": _digest(reconstruction_body),
                "technical_reconstruction_last_candle_close_utc": (
                    window[-1].timestamp_utc + candle_delta
                ).isoformat(),
                "technical_reconstruction_candle_count": len(window),
            }
        )
        counters["reconstructed_rows"] += 1
        output.append(row)

    coverage = {
        field: round(
            sum(
                1
                for row in output
                if str(row.get(field) or "").strip().upper() != "MISSING"
            )
            / len(output),
            6,
        )
        if output
        else 0.0
        for field in TECHNICAL_FEATURE_FIELDS
    }
    return output, {
        "technical_reconstruction_contract": (
            FORECAST_TECHNICAL_RECONSTRUCTION_CONTRACT
        ),
        "technical_reconstruction_basis": FORECAST_TECHNICAL_RECONSTRUCTION_BASIS,
        "technical_reconstruction_granularity": normalized_granularity,
        "technical_reconstruction_lookback_hours": float(lookback_hours),
        "technical_reconstruction_minimum_candles": minimum_candles,
        **counters,
        "technical_feature_coverage_after_reconstruction": coverage,
    }


def _has_retained_technical_evidence(row: Mapping[str, Any]) -> bool:
    if str(row.get("technical_context_sha256") or "").strip():
        return True
    return any(
        str(row.get(field) or "").strip().upper() not in {"", "MISSING"}
        for field in TECHNICAL_FEATURE_FIELDS
    )


def _lookback_is_complete(
    window: Sequence[Any],
    *,
    start_at: datetime,
    forecast_at: datetime,
    candle_delta: timedelta,
    minimum_candles: int,
) -> bool:
    if len(window) < minimum_candles:
        return False
    first_open = window[0].timestamp_utc
    last_close = window[-1].timestamp_utc + candle_delta
    if first_open > start_at + candle_delta:
        return False
    if last_close > forecast_at:
        return False
    if forecast_at - last_close > MAX_RECENT_CANDLE_LAG:
        return False
    return True


def _features_from_window(
    candles: Sequence[Any],
    *,
    pair: str,
    forecast_direction: str,
    forecast_at: datetime,
    candle_delta: timedelta,
) -> dict[str, str]:
    minute_bars = [_mid_bar(candle) for candle in candles]
    m5 = _aggregate(minute_bars, minutes=5)
    m15 = _aggregate(minute_bars, minutes=15)
    h1 = _aggregate(minute_bars, minutes=60)
    h4 = _aggregate(minute_bars, minutes=240)

    price = minute_bars[-1].c
    range_low = min(bar.l for bar in minute_bars)
    range_high = max(bar.h for bar in minute_bars)
    location_value = (
        (price - range_low) / (range_high - range_low)
        if range_high > range_low
        else 0.5
    )
    location = _range_location(location_value)
    atr_series = _rolling_atr(m15, period=14)
    atr_percentile = _percentile_rank(atr_series, atr_series[-1]) if atr_series else None
    atr_band = _atr_band(atr_percentile)

    m5_atr = _atr(m5[-14:])
    pip_factor = instrument_pip_factor(pair)
    last = candles[-1]
    spread_pips = max(0.0, (float(last.ask.c) - float(last.bid.c)) * pip_factor)
    m5_atr_pips = m5_atr * pip_factor if m5_atr is not None else None
    spread_ratio = (
        spread_pips / m5_atr_pips
        if m5_atr_pips is not None and m5_atr_pips > 0.0
        else None
    )
    spread_band = _spread_band(spread_ratio)

    regime, trend_direction = _regime_and_direction(m15, h1)
    structure_direction = _structure_direction(m15, trend_direction=trend_direction)
    structure_alignment = (
        "ALIGNED"
        if structure_direction == forecast_direction
        else "OPPOSED"
        if structure_direction in {"UP", "DOWN"}
        else "MISSING"
    )
    failure_direction = _failed_break_direction(m5)
    selected_method, family_direction = _method_and_family_direction(
        regime=regime,
        trend_direction=trend_direction,
        structure_direction=structure_direction,
        failed_break_direction=failure_direction,
        location=location,
    )
    family_alignment = (
        "ALIGNED"
        if family_direction == forecast_direction
        else "CONTRADICTED"
        if family_direction in {"UP", "DOWN"}
        else "NON_DIRECTIONAL"
    )

    session = tag_bar(forecast_at).tag.value
    chart_story = " ".join(
        _classifier_story_block(timeframe, bars)
        for timeframe, bars in (("H4", h4), ("H1", h1), ("M15", m15), ("M5", m5))
    )
    situation = classify_situation(
        session=session,
        chart_story=chart_story,
        dominant_regime=regime,
    )
    # Assert the temporal contract at the last possible point before return.
    if candles[-1].timestamp_utc + candle_delta > forecast_at:
        raise ValueError("technical reconstruction received a future candle")
    return {
        "technical_regime": regime,
        "technical_atr_band": atr_band,
        "technical_spread_band": spread_band,
        "technical_range_location_24h": location,
        "technical_structure_alignment": structure_alignment,
        "technical_situation": situation,
        "technical_selected_method": selected_method,
        "technical_family_direction_alignment": family_alignment,
    }


def _mid_bar(candle: Any) -> _Bar:
    return _Bar(
        timestamp_utc=candle.timestamp_utc.astimezone(timezone.utc),
        o=(float(candle.bid.o) + float(candle.ask.o)) / 2.0,
        h=(float(candle.bid.h) + float(candle.ask.h)) / 2.0,
        l=(float(candle.bid.l) + float(candle.ask.l)) / 2.0,
        c=(float(candle.bid.c) + float(candle.ask.c)) / 2.0,
    )


def _aggregate(bars: Sequence[_Bar], *, minutes: int) -> list[_Bar]:
    grouped: dict[datetime, list[_Bar]] = {}
    for bar in bars:
        timestamp = bar.timestamp_utc.astimezone(timezone.utc)
        epoch_minute = int(timestamp.timestamp() // 60)
        bucket_epoch_minute = epoch_minute - (epoch_minute % minutes)
        bucket = datetime.fromtimestamp(bucket_epoch_minute * 60, tz=timezone.utc)
        grouped.setdefault(bucket, []).append(bar)
    output: list[_Bar] = []
    for bucket in sorted(grouped):
        items = grouped[bucket]
        output.append(
            _Bar(
                timestamp_utc=bucket,
                o=items[0].o,
                h=max(item.h for item in items),
                l=min(item.l for item in items),
                c=items[-1].c,
            )
        )
    return output


def _true_ranges(bars: Sequence[_Bar]) -> list[float]:
    values: list[float] = []
    previous_close: float | None = None
    for bar in bars:
        value = bar.h - bar.l
        if previous_close is not None:
            value = max(value, abs(bar.h - previous_close), abs(bar.l - previous_close))
        values.append(max(0.0, value))
        previous_close = bar.c
    return values


def _atr(bars: Sequence[_Bar]) -> float | None:
    ranges = _true_ranges(bars)
    return sum(ranges) / len(ranges) if ranges else None


def _rolling_atr(bars: Sequence[_Bar], *, period: int) -> list[float]:
    ranges = _true_ranges(bars)
    if len(ranges) < period:
        return []
    return [
        sum(ranges[index - period + 1 : index + 1]) / period
        for index in range(period - 1, len(ranges))
    ]


def _trend_stats(bars: Sequence[_Bar], *, count: int) -> tuple[str | None, float, float]:
    window = list(bars[-count:])
    if len(window) < max(4, count // 2):
        return None, 0.0, 0.0
    delta = window[-1].c - window[0].c
    direction = "UP" if delta > 0.0 else "DOWN" if delta < 0.0 else None
    path = sum(abs(window[index].c - window[index - 1].c) for index in range(1, len(window)))
    efficiency = abs(delta) / path if path > 0.0 else 0.0
    atr = _atr(window)
    normalized_move = abs(delta) / atr if atr is not None and atr > 0.0 else 0.0
    return direction, efficiency, normalized_move


def _regime_and_direction(
    m15: Sequence[_Bar],
    h1: Sequence[_Bar],
) -> tuple[str, str | None]:
    m15_direction, m15_efficiency, m15_move = _trend_stats(m15, count=16)
    h1_direction, h1_efficiency, h1_move = _trend_stats(h1, count=12)
    aligned = m15_direction is not None and m15_direction == h1_direction
    direction = m15_direction if aligned or h1_direction is None else h1_direction
    if (
        aligned
        and m15_efficiency >= 0.35
        and h1_efficiency >= 0.25
        and m15_move >= 2.0
        and h1_move >= 2.0
    ):
        return "TREND_STRONG", direction
    if (
        (aligned and m15_efficiency >= 0.20 and max(m15_move, h1_move) >= 1.0)
        or (m15_efficiency >= 0.30 and m15_move >= 1.5)
    ):
        return "TREND_WEAK", direction
    if m15_efficiency <= 0.22 and h1_efficiency <= 0.28:
        return "RANGE", None
    return "TRANSITION", direction


def _structure_direction(
    m15: Sequence[_Bar],
    *,
    trend_direction: str | None,
) -> str | None:
    if len(m15) < 32:
        return None
    previous = m15[-32:-16]
    recent = m15[-16:]
    recent_higher = (
        max(bar.h for bar in recent) > max(bar.h for bar in previous)
        and min(bar.l for bar in recent) > min(bar.l for bar in previous)
    )
    recent_lower = (
        max(bar.h for bar in recent) < max(bar.h for bar in previous)
        and min(bar.l for bar in recent) < min(bar.l for bar in previous)
    )
    if recent_higher and recent[-1].c > previous[-1].c:
        return "UP"
    if recent_lower and recent[-1].c < previous[-1].c:
        return "DOWN"
    direction, efficiency, normalized_move = _trend_stats(m15, count=16)
    if direction == trend_direction and efficiency >= 0.25 and normalized_move >= 1.0:
        return direction
    return None


def _failed_break_direction(m5: Sequence[_Bar]) -> str | None:
    if len(m5) < 30:
        return None
    reference = m5[-30:-3]
    recent = m5[-3:]
    rail_high = max(bar.h for bar in reference)
    rail_low = min(bar.l for bar in reference)
    atr = _atr(reference[-14:])
    if atr is None or atr <= 0.0:
        return None
    if max(bar.h for bar in recent) > rail_high + atr * 0.10 and recent[-1].c < rail_high:
        return "DOWN"
    if min(bar.l for bar in recent) < rail_low - atr * 0.10 and recent[-1].c > rail_low:
        return "UP"
    return None


def _method_and_family_direction(
    *,
    regime: str,
    trend_direction: str | None,
    structure_direction: str | None,
    failed_break_direction: str | None,
    location: str,
) -> tuple[str, str | None]:
    if failed_break_direction in {"UP", "DOWN"}:
        return "BREAKOUT_FAILURE", failed_break_direction
    if regime in {"TREND_STRONG", "TREND_WEAK"}:
        return "TREND_CONTINUATION", structure_direction or trend_direction
    if regime == "RANGE":
        inward = "UP" if location == "LOWER" else "DOWN" if location == "UPPER" else None
        return "RANGE_ROTATION", inward
    return "NONE", None


def _classifier_story_block(timeframe: str, bars: Sequence[_Bar]) -> str:
    count = {"H4": 6, "H1": 12, "M15": 16, "M5": 24}.get(timeframe, 12)
    direction, efficiency, normalized_move = _trend_stats(bars, count=count)
    trending = direction is not None and efficiency >= 0.20 and normalized_move >= 1.0
    regime = "TREND_WEAK" if trending else "RANGE"
    # A bounded directional-strength proxy is used only to choose the existing
    # situation vocabulary. It is not represented as historical ADX evidence.
    strength_proxy = min(100.0, max(0.0, efficiency * 100.0))
    return f"{timeframe}({regime} ADX={strength_proxy:.2f})"


def _percentile_rank(values: Sequence[float], current: float) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    if not finite or not math.isfinite(current):
        return None
    return 100.0 * sum(value <= current for value in finite) / len(finite)


def _atr_band(value: float | None) -> str:
    if value is None:
        return "UNKNOWN"
    if value <= 25.0:
        return "LOW"
    if value >= 75.0:
        return "HIGH"
    return "NORMAL"


def _spread_band(value: float | None) -> str:
    if value is None:
        return "UNKNOWN"
    if value <= 0.25:
        return "TIGHT"
    if value <= 0.75:
        return "NORMAL"
    return "WIDE"


def _range_location(value: float) -> str:
    if value <= 1.0 / 3.0:
        return "LOWER"
    if value >= 2.0 / 3.0:
        return "UPPER"
    return "MIDDLE"


def _parse_utc(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
