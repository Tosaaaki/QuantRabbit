"""Deterministic, zero-order FX paper signal and accounting primitives.

The module has no network or broker imports.  It turns completed, contiguous M5
bars into a cost-independent directional proposal and prices that same proposal
under several declared virtual execution arms.
"""
from __future__ import annotations

import math
from datetime import timedelta
from typing import Any

from shadow_runtime import parse_utc


class PaperConfigError(ValueError):
    pass


def _finite_number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PaperConfigError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise PaperConfigError(f"{name} invalid")
    return number


def validate_paper_config(config: dict[str, Any]) -> None:
    required = {
        "enabled",
        "strategy_id",
        "timeframe",
        "fast_ema_bars",
        "slow_ema_bars",
        "momentum_bars",
        "atr_bars",
        "tp_atr_multiple",
        "tp_spread_multiple_floor",
        "max_age_bars",
        "expected_order_ttl_bars",
        "virtual_units",
        "hard_max_open_positions_per_instrument",
        "hard_max_open_positions_total",
        "entry_cost_gate_used",
        "require_tradeable_bbo",
        "require_side_liquidity",
        "persist_latency_event_consumption",
        "llm_unwind_semantics",
        "jpy_conversion_quote_max_age_seconds",
        "arms",
    }
    missing = required - set(config)
    if missing:
        raise PaperConfigError(f"paper config missing: {sorted(missing)}")
    if config["enabled"] is not True or config["timeframe"] != "M5":
        raise PaperConfigError("paper strategy must be enabled on M5")
    if config["entry_cost_gate_used"] is not False:
        raise PaperConfigError("cost must not suppress raw signals")
    if (
        config["require_tradeable_bbo"] is not True
        or config["require_side_liquidity"] is not True
        or config["persist_latency_event_consumption"] is not True
        or config["llm_unwind_semantics"] != "ONE_OLDEST_POSITION_PER_POLICY_DECISION"
    ):
        raise PaperConfigError("paper execution safety contract mismatch")
    if not isinstance(config["strategy_id"], str) or not config["strategy_id"]:
        raise PaperConfigError("strategy_id invalid")
    integer_fields = (
        "fast_ema_bars",
        "slow_ema_bars",
        "momentum_bars",
        "atr_bars",
        "max_age_bars",
        "expected_order_ttl_bars",
        "virtual_units",
        "hard_max_open_positions_per_instrument",
        "hard_max_open_positions_total",
        "jpy_conversion_quote_max_age_seconds",
    )
    for name in integer_fields:
        if type(config[name]) is not int or config[name] <= 0:
            raise PaperConfigError(f"{name} invalid")
    if config["fast_ema_bars"] >= config["slow_ema_bars"]:
        raise PaperConfigError("fast EMA must be shorter than slow EMA")
    _finite_number(config["tp_atr_multiple"], "tp_atr_multiple", positive=True)
    _finite_number(
        config["tp_spread_multiple_floor"],
        "tp_spread_multiple_floor",
        positive=True,
    )
    required_arms = {
        "RAW_SIGNAL",
        "EXECUTABLE_BASE",
        "ADVERSE_STRESS",
        "ACTUAL_LLM_INVENTORY",
    }
    if set(config["arms"]) != required_arms:
        raise PaperConfigError("paper execution arm set mismatch")
    for arm, values in config["arms"].items():
        if set(values) != {"price_mode", "slippage_pips_per_side", "entry_latency_events"}:
            raise PaperConfigError(f"{arm} schema mismatch")
        if values["price_mode"] not in {"MID", "BID_ASK"}:
            raise PaperConfigError(f"{arm} price mode invalid")
        _finite_number(values["slippage_pips_per_side"], f"{arm}.slippage")
        if type(values["entry_latency_events"]) is not int or values["entry_latency_events"] < 0:
            raise PaperConfigError(f"{arm} entry latency invalid")


def pip_size(instrument: str) -> float:
    """OANDA FX pip convention: JPY quote pairs use 0.01, others 0.0001."""
    if not isinstance(instrument, str) or "_" not in instrument:
        raise ValueError("invalid instrument")
    return 0.01 if instrument.endswith("_JPY") else 0.0001


def _ema(values: list[float], period: int) -> float:
    if not values:
        raise ValueError("EMA requires values")
    # Standard EMA smoothing represents a period-N rolling memory.  The period
    # is preregistered in the runtime contract rather than tuned in this module.
    alpha = 2.0 / (period + 1.0)
    result = values[0]
    for value in values[1:]:
        result = alpha * value + (1.0 - alpha) * result
    return result


def _mid(bar: dict[str, Any], field: str) -> float:
    return (float(bar[f"bid_{field}"]) + float(bar[f"ask_{field}"])) / 2.0


def completed_bar_input_window(
    bars: list[dict[str, Any]],
    minimum: int,
) -> list[dict[str, Any]] | None:
    """Return one causal M5 feature window ending in LIVE evidence.

    Historical OANDA BID/ASK candles may prefix the window as feature-only
    warmup.  They may never follow a LIVE row, overlap it, bridge a missing M5
    interval, or become the final decision row.  LIVE rows must additionally
    remain inside one source-attested feed segment.
    """
    if type(minimum) is not int or minimum <= 0:
        raise ValueError("minimum input window invalid")
    if len(bars) < minimum:
        return None
    tail = bars[-minimum:]
    instrument = tail[0].get("instrument")
    if not instrument:
        return None
    saw_live = False
    live_segment_id: str | None = None
    prior_start = None
    for prior, current in zip(tail, tail[1:]):
        if current.get("instrument") != instrument:
            return None
        if parse_utc(current["start_utc"]) - parse_utc(prior["start_utc"]) != timedelta(minutes=5):
            return None
    for bar in tail:
        start = parse_utc(bar["start_utc"])
        end = parse_utc(bar["end_utc"])
        if end - start != timedelta(minutes=5) or (prior_start is not None and start <= prior_start):
            return None
        prior_start = start
        source = bar.get("feature_source", "LIVE_ATTESTED_M5")
        if source == "OANDA_HISTORICAL_M5_WARMUP":
            if (
                saw_live
                or bar.get("warmup_only") is not True
                or bar.get("excluded_from_forward_pnl") is not True
            ):
                return None
            continue
        if source != "LIVE_ATTESTED_M5":
            return None
        segment_id = bar.get("segment_id")
        if not isinstance(segment_id, str) or not segment_id:
            return None
        if live_segment_id is None:
            live_segment_id = segment_id
        elif segment_id != live_segment_id:
            return None
        saw_live = True
    if not saw_live or tail[-1].get("feature_source", "LIVE_ATTESTED_M5") != "LIVE_ATTESTED_M5":
        return None
    return tail


def _contiguous_tail(
    bars: list[dict[str, Any]],
    minimum: int,
) -> list[dict[str, Any]] | None:
    return completed_bar_input_window(bars, minimum)


def evaluate_completed_bar_signal(
    bars: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    """Return one cost-independent direction proposal from completed bars only."""
    validate_paper_config(config)
    minimum = max(
        config["slow_ema_bars"] + 1,
        config["momentum_bars"] + 1,
        config["atr_bars"] + 1,
    )
    tail = _contiguous_tail(bars, minimum)
    if tail is None:
        return None
    closes = [_mid(bar, "c") for bar in tail]
    fast = _ema(closes, config["fast_ema_bars"])
    slow = _ema(closes, config["slow_ema_bars"])
    prior_slow = _ema(closes[:-1], config["slow_ema_bars"])
    direction = 1 if fast > slow else -1 if fast < slow else 0
    momentum = closes[-1] - closes[-1 - config["momentum_bars"]]
    slow_slope = slow - prior_slow
    if direction == 0 or direction * momentum <= 0 or direction * slow_slope <= 0:
        return None

    true_ranges: list[float] = []
    for index in range(1, len(tail)):
        high = _mid(tail[index], "h")
        low = _mid(tail[index], "l")
        previous_close = _mid(tail[index - 1], "c")
        true_ranges.append(max(high - low, abs(high - previous_close), abs(low - previous_close)))
    atr = sum(true_ranges[-config["atr_bars"] :]) / config["atr_bars"]
    spread = float(tail[-1]["ask_c"]) - float(tail[-1]["bid_c"])
    if not math.isfinite(atr) or atr <= 0 or not math.isfinite(spread) or spread <= 0:
        return None
    tp_distance = max(
        atr * float(config["tp_atr_multiple"]),
        spread * float(config["tp_spread_multiple_floor"]),
    )
    return {
        "direction": direction,
        "direction_label": "LONG" if direction > 0 else "SHORT",
        "decision_mid": closes[-1],
        "fast_ema": fast,
        "slow_ema": slow,
        "slow_ema_slope": slow_slope,
        "momentum_price": momentum,
        "atr_price": atr,
        "observed_spread_price": spread,
        "tp_distance_price": tp_distance,
        "entry_cost_gate_used": False,
        "completed_bar_count_used": len(tail),
        "historical_warmup_bar_count_used": sum(
            bar.get("feature_source") == "OANDA_HISTORICAL_M5_WARMUP" for bar in tail
        ),
        "segment_id": tail[-1]["segment_id"],
    }


def executable_bbo_available(
    event: dict[str, Any],
    direction: int,
    *,
    entry: bool,
    required_units: int,
) -> bool:
    """Prove the selected virtual side is currently tradeable and deep enough."""
    if direction not in {-1, 1} or type(required_units) is not int or required_units <= 0:
        return False
    if event.get("tradeable") is not True:
        return False
    liquidity_field = (
        "ask_liquidity"
        if (entry and direction > 0) or (not entry and direction < 0)
        else "bid_liquidity"
    )
    liquidity = event.get(liquidity_field)
    if isinstance(liquidity, bool) or not isinstance(liquidity, (int, float)):
        return False
    amount = float(liquidity)
    return math.isfinite(amount) and amount >= required_units


def virtual_price(
    event: dict[str, Any],
    direction: int,
    arm_config: dict[str, Any],
    *,
    entry: bool,
    required_units: int | None = None,
) -> float:
    bid = _finite_number(event["bid"], "bid", positive=True)
    ask = _finite_number(event["ask"], "ask", positive=True)
    if ask <= bid or direction not in {-1, 1}:
        raise ValueError("invalid executable BBO")
    if required_units is not None and not executable_bbo_available(
        event,
        direction,
        entry=entry,
        required_units=required_units,
    ):
        raise ValueError("selected BBO side is not executable")
    if arm_config["price_mode"] == "MID":
        return (bid + ask) / 2.0
    slip = float(arm_config["slippage_pips_per_side"]) * pip_size(event["instrument"])
    if entry:
        return ask + slip if direction > 0 else bid - slip
    return bid - slip if direction > 0 else ask + slip


def quote_pnl(entry_price: float, exit_price: float, direction: int, units: int) -> float:
    if type(units) is not int or units <= 0 or direction not in {-1, 1}:
        raise ValueError("invalid paper position")
    result = direction * (float(exit_price) - float(entry_price)) * units
    if not math.isfinite(result):
        raise ValueError("non-finite paper PnL")
    return result


def pnl_pips(entry_price: float, exit_price: float, direction: int, instrument: str) -> float:
    return direction * (float(exit_price) - float(entry_price)) / pip_size(instrument)
