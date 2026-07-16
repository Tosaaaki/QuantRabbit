"""Causal bid/ask replay for forecast-directed passive LIMIT entries.

The vehicle joins the executable near side at the first complete candle after
forecast emission (LONG at bid, SHORT at ask), then keeps the forecast's
already-frozen target and invalidation as broker-attached TP/SL geometry.
There is no synthetic market entry and no observation-end close.

This module is research-only.  Broad M1 results can rank technical cohorts,
but cannot grant live permission; any candidate still needs exact S5 replay,
forward evidence, and the normal verifier/risk/gateway checks.
"""

from __future__ import annotations

import bisect
import math
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence

from quant_rabbit.instruments import instrument_pip_factor


PASSIVE_LIMIT_REPLAY_CONTRACT = "QR_FORECAST_PASSIVE_LIMIT_BID_ASK_V1"
MARKET_BRACKET_REPLAY_CONTRACT = "QR_FORECAST_MARKET_BRACKET_BID_ASK_V1"
MARKET_STOP_TIME_CLOSE_REPLAY_CONTRACT = (
    "QR_FORECAST_MARKET_STOP_TIME_CLOSE_BID_ASK_V1"
)
DIRECTIONAL = {"UP", "DOWN"}


def ceil_time(value: datetime, interval: timedelta) -> datetime:
    """Return the first exact interval boundary at or after ``value``."""

    seconds = interval.total_seconds()
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError("candle interval must be finite and positive")
    timestamp = value.timestamp()
    rounded = math.ceil(timestamp / seconds - 1e-12) * seconds
    return datetime.fromtimestamp(rounded, tz=value.tzinfo)


def select_independent_forecasts(
    rows: Sequence[Any],
    *,
    horizon_min: float,
) -> list[Any]:
    """Select pair-local, non-overlapping trials for one fixed horizon."""

    if not math.isfinite(float(horizon_min)) or float(horizon_min) <= 0.0:
        raise ValueError("horizon_min must be finite and positive")
    horizon = timedelta(minutes=float(horizon_min))
    selected: list[Any] = []
    accepted_until: dict[str, datetime] = {}
    for row in sorted(
        rows,
        key=lambda item: (
            item.timestamp_utc,
            str(item.pair),
            int(getattr(item, "source_index", 0)),
        ),
    ):
        pair = str(row.pair)
        if row.timestamp_utc < accepted_until.get(pair, row.timestamp_utc):
            continue
        selected.append(row)
        accepted_until[pair] = row.timestamp_utc + horizon
    return selected


def simulate_passive_limit(
    row: Any,
    candles: Sequence[Any],
    *,
    horizon_min: float,
    entry_ttl_min: float | None = None,
    max_hold_min: float | None = None,
    reward_pips: float | None = None,
    risk_pips: float | None = None,
    candle_interval: timedelta = timedelta(minutes=1),
    candle_times: Sequence[datetime] | None = None,
    technical_features: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay one passive LIMIT plus an attached TP/SL bracket.

    The entry may rest for ``entry_ttl_min`` and, once filled, the bracket is
    observed for ``max_hold_min``. Missing/no-tick bars are not synthesized.
    Optional fixed-pip geometry translates only the forecast direction; it
    never changes or re-labels that prediction.
    """

    direction = str(getattr(row, "direction", "") or "").upper()
    pair = str(getattr(row, "pair", "") or "").upper()
    source_index = int(getattr(row, "source_index", 0))
    timestamp = getattr(row, "timestamp_utc", None)
    current = _finite(getattr(row, "current_price", None))
    forecast_target = _finite(getattr(row, "target_price", None))
    forecast_invalidation = _finite(getattr(row, "invalidation_price", None))
    fixed_reward = _positive_finite(reward_pips)
    fixed_risk = _positive_finite(risk_pips)
    fixed_geometry = reward_pips is not None or risk_pips is not None
    if fixed_geometry and (fixed_reward is None or fixed_risk is None):
        raise ValueError("reward_pips and risk_pips must both be finite and positive")
    entry_ttl = _positive_finite(
        horizon_min if entry_ttl_min is None else entry_ttl_min
    )
    max_hold = _positive_finite(
        horizon_min if max_hold_min is None else max_hold_min
    )
    if entry_ttl is None or max_hold is None:
        raise ValueError("entry TTL and maximum hold must be finite and positive")
    base = {
        "source_index": source_index,
        "timestamp_utc": _iso(timestamp),
        "pair": pair,
        "direction": direction,
        "side": "LONG" if direction == "UP" else "SHORT" if direction == "DOWN" else None,
        "horizon_min": float(horizon_min),
        "entry_ttl_min": entry_ttl,
        "max_hold_min": max_hold,
        "geometry_source": "FIXED_PIPS" if fixed_geometry else "FROZEN_FORECAST",
        "technical_features": dict(technical_features or {}),
        "contract": PASSIVE_LIMIT_REPLAY_CONTRACT,
    }
    if (
        direction not in DIRECTIONAL
        or not pair
        or not isinstance(timestamp, datetime)
        or (not fixed_geometry and current is None)
        or (not fixed_geometry and forecast_target is None)
        or (not fixed_geometry and forecast_invalidation is None)
    ):
        return {**base, "status": "INVALID_FORECAST_GEOMETRY", "filled": False}
    if (
        not fixed_geometry
        and direction == "UP"
        and not forecast_invalidation < current < forecast_target
    ):
        return {**base, "status": "INVALID_FORECAST_GEOMETRY", "filled": False}
    if (
        not fixed_geometry
        and direction == "DOWN"
        and not forecast_target < current < forecast_invalidation
    ):
        return {**base, "status": "INVALID_FORECAST_GEOMETRY", "filled": False}
    if not math.isfinite(float(horizon_min)) or float(horizon_min) <= 0.0:
        raise ValueError("horizon_min must be finite and positive")

    activation = ceil_time(timestamp, candle_interval)
    entry_end = activation + timedelta(minutes=entry_ttl)
    available_end = entry_end + timedelta(minutes=max_hold)
    times = (
        candle_times
        if candle_times is not None
        else [item.timestamp_utc for item in candles]
    )
    if len(times) != len(candles):
        raise ValueError("candle_times must align one-to-one with candles")
    start = bisect.bisect_left(times, activation)
    end = bisect.bisect_right(times, available_end - candle_interval)
    if end <= start:
        return {
            **base,
            "status": "NO_EXECUTABLE_QUOTE_WINDOW",
            "filled": False,
            "activation_at_utc": _iso(activation),
            "entry_expire_at_utc": _iso(entry_end),
            "observation_available_to_utc": _iso(available_end),
        }
    window = candles[start:end]
    first = window[0]
    pip = 1.0 / instrument_pip_factor(pair)
    if direction == "UP":
        entry = float(first.bid.o)
        executable_open = float(first.ask.o)
        passive = entry < executable_open
    else:
        entry = float(first.ask.o)
        executable_open = float(first.bid.o)
        passive = entry > executable_open
    if fixed_geometry and direction == "UP":
        target = entry + fixed_reward * pip
        invalidation = entry - fixed_risk * pip
    elif fixed_geometry:
        target = entry - fixed_reward * pip
        invalidation = entry + fixed_risk * pip
    else:
        target = forecast_target
        invalidation = forecast_invalidation
    if not passive:
        return {
            **base,
            "status": "NON_POSITIVE_SPREAD_AT_ACTIVATION",
            "filled": False,
            "activation_at_utc": _iso(activation),
        }
    if direction == "UP" and not invalidation < entry < target:
        return {
            **base,
            "status": "PASSIVE_ENTRY_BREAKS_FORECAST_GEOMETRY",
            "filled": False,
            "entry_price": entry,
        }
    if direction == "DOWN" and not target < entry < invalidation:
        return {
            **base,
            "status": "PASSIVE_ENTRY_BREAKS_FORECAST_GEOMETRY",
            "filled": False,
            "entry_price": entry,
        }

    spread_pips = (float(first.ask.o) - float(first.bid.o)) / pip
    entry_window_end = bisect.bisect_right(times, entry_end - candle_interval)
    entry_window = candles[start:entry_window_end]
    fill_index = next(
        (
            index
            for index, candle in enumerate(entry_window)
            if (direction == "UP" and float(candle.ask.l) <= entry)
            or (direction == "DOWN" and float(candle.bid.h) >= entry)
        ),
        None,
    )
    geometry = {
        "activation_at_utc": _iso(activation),
        "entry_expire_at_utc": _iso(entry_end),
        "observation_available_to_utc": _iso(available_end),
        "entry_price": entry,
        "target_price": target,
        "invalidation_price": invalidation,
        "activation_spread_pips": round(spread_pips, 6),
        "reward_pips": round(abs(target - entry) / pip, 6),
        "risk_pips": round(abs(entry - invalidation) / pip, 6),
    }
    if fill_index is None:
        return {
            **base,
            **geometry,
            "status": "UNFILLED_EXPIRED",
            "filled": False,
        }

    fill_at = entry_window[fill_index].timestamp_utc
    fill_end = fill_at + timedelta(minutes=max_hold)
    fill_end_index = bisect.bisect_right(times, fill_end - candle_interval)
    fill_candles = candles[start + fill_index : fill_end_index]
    exit_result = _attached_exit(
        direction=direction,
        entry=entry,
        target=target,
        invalidation=invalidation,
        candles=fill_candles,
        pip=pip,
        candle_interval=candle_interval,
    )
    return {
        **base,
        **geometry,
        "filled": True,
        "fill_at_utc": _iso(fill_at),
        **exit_result,
    }


def simulate_market_stop_time_close(
    row: Any,
    candles: Sequence[Any],
    *,
    horizon_min: float,
    risk_pips: float,
    candle_interval: timedelta = timedelta(minutes=1),
    candle_times: Sequence[datetime] | None = None,
    technical_features: Mapping[str, Any] | None = None,
    time_close_quote_grace: timedelta = timedelta(seconds=60),
) -> dict[str, Any]:
    """Enter at market, cap downside with SL, and let profit run to horizon."""

    direction = str(getattr(row, "direction", "") or "").upper()
    pair = str(getattr(row, "pair", "") or "").upper()
    source_index = int(getattr(row, "source_index", 0))
    timestamp = getattr(row, "timestamp_utc", None)
    risk = _positive_finite(risk_pips)
    horizon = _positive_finite(horizon_min)
    base = {
        "source_index": source_index,
        "timestamp_utc": _iso(timestamp),
        "pair": pair,
        "direction": direction,
        "side": (
            "LONG"
            if direction == "UP"
            else "SHORT"
            if direction == "DOWN"
            else None
        ),
        "horizon_min": float(horizon_min),
        "entry_ttl_min": 0.0,
        "max_hold_min": float(horizon_min),
        "geometry_source": "FIXED_STOP_TIME_CLOSE",
        "entry_vehicle": "MARKET_STOP_TIME_CLOSE",
        "technical_features": dict(technical_features or {}),
        "contract": MARKET_STOP_TIME_CLOSE_REPLAY_CONTRACT,
    }
    if (
        direction not in DIRECTIONAL
        or not pair
        or not isinstance(timestamp, datetime)
        or risk is None
        or horizon is None
    ):
        return {**base, "status": "INVALID_FORECAST_GEOMETRY", "filled": False}
    activation = ceil_time(timestamp, candle_interval)
    available_end = activation + timedelta(minutes=horizon)
    times = (
        candle_times
        if candle_times is not None
        else [item.timestamp_utc for item in candles]
    )
    if len(times) != len(candles):
        raise ValueError("candle_times must align one-to-one with candles")
    start = bisect.bisect_left(times, activation)
    end = bisect.bisect_right(times, available_end - candle_interval)
    if end <= start:
        return {
            **base,
            "status": "NO_EXECUTABLE_QUOTE_WINDOW",
            "filled": False,
            "activation_at_utc": _iso(activation),
            "observation_available_to_utc": _iso(available_end),
        }
    window = candles[start:end]
    first = window[0]
    pip = 1.0 / instrument_pip_factor(pair)
    if direction == "UP":
        entry = float(first.ask.o)
        invalidation = entry - risk * pip
    else:
        entry = float(first.bid.o)
        invalidation = entry + risk * pip
    spread_pips = (float(first.ask.o) - float(first.bid.o)) / pip
    geometry = {
        "filled": True,
        "activation_at_utc": _iso(activation),
        "observation_available_to_utc": _iso(available_end),
        "fill_at_utc": _iso(first.timestamp_utc),
        "entry_price": entry,
        "target_price": None,
        "invalidation_price": invalidation,
        "activation_spread_pips": round(spread_pips, 6),
        "reward_pips": None,
        "risk_pips": round(risk, 6),
    }
    for candle in window:
        if direction == "UP":
            stop_open = float(candle.bid.o)
            stop_hit = float(candle.bid.l) <= invalidation
            stop_exit = stop_open if stop_open < invalidation else invalidation
            realized = (stop_exit - entry) / pip
        else:
            stop_open = float(candle.ask.o)
            stop_hit = float(candle.ask.h) >= invalidation
            stop_exit = stop_open if stop_open > invalidation else invalidation
            realized = (entry - stop_exit) / pip
        if stop_hit:
            value = round(realized, 6)
            return {
                **base,
                **geometry,
                "status": "FILLED_RESOLVED",
                "exit_reason": "STOP_LOSS",
                "exit_at_utc": _iso(candle.timestamp_utc + candle_interval),
                "realized_pips": value,
                "conservative_pips": value,
                "gap_through_stop": stop_exit != invalidation,
            }
    time_close_index = bisect.bisect_left(times, available_end)
    if (
        time_close_index >= len(candles)
        or times[time_close_index] > available_end + time_close_quote_grace
    ):
        return {
            **base,
            **geometry,
            "status": "FILLED_OPEN",
            "exit_reason": "OPEN_UNRESOLVED",
            "exit_at_utc": None,
            "realized_pips": None,
            "conservative_pips": round(-risk, 6),
            "gap_through_stop": False,
        }
    time_close_candle = candles[time_close_index]
    exit_price = (
        float(time_close_candle.bid.o)
        if direction == "UP"
        else float(time_close_candle.ask.o)
    )
    realized = (
        (exit_price - entry) / pip
        if direction == "UP"
        else (entry - exit_price) / pip
    )
    value = round(realized, 6)
    return {
        **base,
        **geometry,
        "status": "FILLED_RESOLVED",
        "exit_reason": "TIME_CLOSE",
        "exit_at_utc": _iso(time_close_candle.timestamp_utc),
        "exit_price": exit_price,
        "realized_pips": value,
        "conservative_pips": value,
        "gap_through_stop": False,
    }


def simulate_market_bracket(
    row: Any,
    candles: Sequence[Any],
    *,
    horizon_min: float,
    reward_pips: float,
    risk_pips: float,
    candle_interval: timedelta = timedelta(minutes=1),
    candle_times: Sequence[datetime] | None = None,
    technical_features: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay an executable market entry with an attached TP/SL bracket."""

    direction = str(getattr(row, "direction", "") or "").upper()
    pair = str(getattr(row, "pair", "") or "").upper()
    source_index = int(getattr(row, "source_index", 0))
    timestamp = getattr(row, "timestamp_utc", None)
    reward = _positive_finite(reward_pips)
    risk = _positive_finite(risk_pips)
    horizon = _positive_finite(horizon_min)
    base = {
        "source_index": source_index,
        "timestamp_utc": _iso(timestamp),
        "pair": pair,
        "direction": direction,
        "side": (
            "LONG"
            if direction == "UP"
            else "SHORT"
            if direction == "DOWN"
            else None
        ),
        "horizon_min": float(horizon_min),
        "entry_ttl_min": 0.0,
        "max_hold_min": float(horizon_min),
        "geometry_source": "FIXED_PIPS",
        "entry_vehicle": "MARKET",
        "technical_features": dict(technical_features or {}),
        "contract": MARKET_BRACKET_REPLAY_CONTRACT,
    }
    if (
        direction not in DIRECTIONAL
        or not pair
        or not isinstance(timestamp, datetime)
        or reward is None
        or risk is None
        or horizon is None
    ):
        return {**base, "status": "INVALID_FORECAST_GEOMETRY", "filled": False}
    activation = ceil_time(timestamp, candle_interval)
    available_end = activation + timedelta(minutes=horizon)
    times = (
        candle_times
        if candle_times is not None
        else [item.timestamp_utc for item in candles]
    )
    if len(times) != len(candles):
        raise ValueError("candle_times must align one-to-one with candles")
    start = bisect.bisect_left(times, activation)
    end = bisect.bisect_right(times, available_end - candle_interval)
    if end <= start:
        return {
            **base,
            "status": "NO_EXECUTABLE_QUOTE_WINDOW",
            "filled": False,
            "activation_at_utc": _iso(activation),
            "observation_available_to_utc": _iso(available_end),
        }
    window = candles[start:end]
    first = window[0]
    pip = 1.0 / instrument_pip_factor(pair)
    if direction == "UP":
        entry = float(first.ask.o)
        target = entry + reward * pip
        invalidation = entry - risk * pip
    else:
        entry = float(first.bid.o)
        target = entry - reward * pip
        invalidation = entry + risk * pip
    spread_pips = (float(first.ask.o) - float(first.bid.o)) / pip
    exit_result = _attached_exit(
        direction=direction,
        entry=entry,
        target=target,
        invalidation=invalidation,
        candles=window,
        pip=pip,
        candle_interval=candle_interval,
        entry_at_candle_open=True,
    )
    return {
        **base,
        "filled": True,
        "activation_at_utc": _iso(activation),
        "observation_available_to_utc": _iso(available_end),
        "fill_at_utc": _iso(first.timestamp_utc),
        "entry_price": entry,
        "target_price": target,
        "invalidation_price": invalidation,
        "activation_spread_pips": round(spread_pips, 6),
        "reward_pips": round(reward, 6),
        "risk_pips": round(risk, 6),
        **exit_result,
    }


def replay_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize resolved and conservative all-filled economics."""

    signals = len(rows)
    filled = [row for row in rows if row.get("filled") is True]
    resolved = [row for row in filled if _finite(row.get("realized_pips")) is not None]
    conservative = [
        float(row["conservative_pips"])
        for row in filled
        if _finite(row.get("conservative_pips")) is not None
    ]
    resolved_pips = [float(row["realized_pips"]) for row in resolved]
    by_day: dict[str, list[float]] = defaultdict(list)
    for row in filled:
        value = _finite(row.get("conservative_pips"))
        if value is None:
            continue
        day = str(row.get("timestamp_utc") or "")[:10]
        if day:
            by_day[day].append(value)
    daily = [sum(values) for values in by_day.values()]
    return {
        "signals": signals,
        "fills": len(filled),
        "fill_rate": round(len(filled) / signals, 6) if signals else 0.0,
        "resolved_fills": len(resolved),
        "resolution_rate": round(len(resolved) / len(filled), 6) if filled else 0.0,
        "take_profit_fills": sum(row.get("exit_reason") == "TAKE_PROFIT" for row in filled),
        "stop_loss_fills": sum(row.get("exit_reason") == "STOP_LOSS" for row in filled),
        "ambiguous_fills": sum(str(row.get("exit_reason") or "").startswith("AMBIGUOUS") for row in filled),
        "open_unresolved_fills": sum(row.get("exit_reason") == "OPEN_UNRESOLVED" for row in filled),
        "mean_resolved_pips": _mean(resolved_pips),
        "mean_conservative_pips": _mean(conservative),
        "net_conservative_pips": round(sum(conservative), 6) if conservative else 0.0,
        "conservative_profit_factor": _profit_factor(conservative),
        "conservative_win_rate": round(sum(value > 0.0 for value in conservative) / len(conservative), 6)
        if conservative
        else 0.0,
        "active_days": len(by_day),
        "positive_day_rate": round(sum(value > 0.0 for value in daily) / len(daily), 6)
        if daily
        else 0.0,
        "one_sided_95_mean_lower_pips": _one_sided_lower(conservative),
        "one_sided_95_daily_lower_pips": _one_sided_lower(daily),
    }


def _attached_exit(
    *,
    direction: str,
    entry: float,
    target: float,
    invalidation: float,
    candles: Sequence[Any],
    pip: float,
    candle_interval: timedelta,
    entry_at_candle_open: bool = False,
) -> dict[str, Any]:
    risk_pips = abs(entry - invalidation) / pip
    reward_pips = abs(target - entry) / pip
    for index, candle in enumerate(candles):
        if direction == "UP":
            target_hit = float(candle.bid.h) >= target
            stop_hit = float(candle.bid.l) <= invalidation
            target_close_proof = float(candle.bid.c) >= target
            stop_open = float(candle.bid.o)
            gap_stop = stop_open < invalidation
            stop_exit = stop_open if gap_stop else invalidation
            stop_pips = (stop_exit - entry) / pip
        else:
            target_hit = float(candle.ask.l) <= target
            stop_hit = float(candle.ask.h) >= invalidation
            target_close_proof = float(candle.ask.c) <= target
            stop_open = float(candle.ask.o)
            gap_stop = stop_open > invalidation
            stop_exit = stop_open if gap_stop else invalidation
            stop_pips = (entry - stop_exit) / pip
        exit_at = candle.timestamp_utc + candle_interval
        if stop_hit and target_hit:
            return {
                "status": "FILLED_AMBIGUOUS",
                "exit_reason": "AMBIGUOUS_TP_SL_ORDERING",
                "exit_at_utc": None,
                "ambiguity_at_utc": _iso(exit_at),
                "realized_pips": None,
                "conservative_pips": round(min(stop_pips, -risk_pips), 6),
                "gap_through_stop": gap_stop,
            }
        if (
            index == 0
            and not entry_at_candle_open
            and target_hit
            and not target_close_proof
        ):
            return {
                "status": "FILLED_AMBIGUOUS",
                "exit_reason": "AMBIGUOUS_TARGET_BEFORE_FILL",
                "exit_at_utc": None,
                "ambiguity_at_utc": _iso(exit_at),
                "realized_pips": None,
                "conservative_pips": round(-risk_pips, 6),
                "gap_through_stop": False,
            }
        if stop_hit:
            return {
                "status": "FILLED_RESOLVED",
                "exit_reason": "STOP_LOSS",
                "exit_at_utc": _iso(exit_at),
                "realized_pips": round(stop_pips, 6),
                "conservative_pips": round(stop_pips, 6),
                "gap_through_stop": gap_stop,
            }
        if target_hit:
            return {
                "status": "FILLED_RESOLVED",
                "exit_reason": "TAKE_PROFIT",
                "exit_at_utc": _iso(exit_at),
                "realized_pips": round(reward_pips, 6),
                "conservative_pips": round(reward_pips, 6),
                "gap_through_stop": False,
            }
    return {
        "status": "FILLED_OPEN",
        "exit_reason": "OPEN_UNRESOLVED",
        "exit_at_utc": None,
        "realized_pips": None,
        "conservative_pips": round(-risk_pips, 6),
        "gap_through_stop": False,
    }


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _positive_finite(value: Any) -> float | None:
    result = _finite(value)
    return result if result is not None and result > 0.0 else None


def _iso(value: Any) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None


def _mean(values: Sequence[float]) -> float | None:
    return round(statistics.mean(values), 6) if values else None


def _profit_factor(values: Sequence[float]) -> float | None:
    profit = sum(value for value in values if value > 0.0)
    loss = -sum(value for value in values if value < 0.0)
    if loss == 0.0:
        return math.inf if profit > 0.0 else 0.0
    return round(profit / loss, 6)


def _one_sided_lower(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    # 1.645 is the predeclared one-sided 95% normal critical value.  This is a
    # broad-screen statistic; exact S5 promotion uses the stricter Student-t
    # and daily-stability contract in the canonical RANGE replay validator.
    lower = statistics.mean(values) - 1.645 * statistics.stdev(values) / math.sqrt(len(values))
    return round(lower, 6)
