from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path


V2_DIR = Path(__file__).resolve().parents[1] / "2026-08-25-v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from fx_original_indicators import Bar  # noqa: E402


CURRENT_FEATURES = (
    "seismic_shock_index",
    "directional_impulse_short",
    "directional_impulse_long",
    "semivariance_alignment",
    "auction_close_location",
    "escape_range_fraction",
    "continuation_coherence",
    "graph_escape_alignment",
    "dynamic_shape_energy",
)

DELAYED_FEATURES = (
    "post_break_reentry_depth",
    "post_break_failure_velocity",
    "post_break_spread_relaxation",
    "failed_auction_score",
    "sweep_reversal_pressure",
)

ALL_FEATURES = CURRENT_FEATURES + DELAYED_FEATURES


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _directional_impulse(returns: list[float], side: int) -> float:
    energy = math.sqrt(sum(value * value for value in returns))
    return side * sum(returns) / energy if energy > 0 else 0.0


def enrich_event(event: dict, bars: list[Bar], lookback: int) -> dict:
    """Add fixed causal composites to an existing breakout event.

    CURRENT_FEATURES use history plus the completed breakout bar.  DELAYED_FEATURES
    additionally use the next completed bar and are therefore forbidden for the
    immediate worker by the factory's feature-time authority table.
    """
    i = int(event["breakout_index"])
    if i < lookback or i + 1 >= len(bars):
        raise ValueError("event lacks required completed context")
    side = int(event["escape_side"])
    window = bars[i - lookback:i + 1]
    log_returns = [
        math.log(current.mid_c / previous.mid_c)
        for previous, current in zip(window, window[1:])
    ]
    historical = log_returns[:-1] or [0.0]
    current_return = log_returns[-1]
    previous_return = historical[-1]
    median_energy = _median([value * value for value in historical]) or 1e-16
    diff_scale = math.sqrt(_mean([
        (right - left) ** 2 for left, right in zip(historical, historical[1:])
    ])) or math.sqrt(median_energy)
    energy_ratio = current_return * current_return / median_energy
    jerk = abs(current_return - previous_return) / max(diff_scale, 1e-12)

    short_impulse = _directional_impulse(log_returns[-3:], side)
    long_impulse = _directional_impulse(log_returns[-12:], side)
    upside = _mean([max(value, 0.0) ** 2 for value in historical])
    downside = _mean([min(value, 0.0) ** 2 for value in historical])
    semivariance_alignment = side * (upside - downside) / max(upside + downside, 1e-16)

    bar = bars[i]
    next_bar = bars[i + 1]
    bar_range = max(bar.mid_h - bar.mid_l, float(event["scale"]) * 0.05)
    close_location = (
        (bar.mid_c - bar.mid_l) / bar_range
        if side > 0 else (bar.mid_h - bar.mid_c) / bar_range
    )
    escape_range_fraction = max(0.0, float(event["rail_escape_energy"])) * float(event["scale"]) / bar_range
    spread_strain = max(float(event["session_spread_strain"]), 0.0)
    graph_alignment = max(0.0, float(event.get("currency_propagation") or 0.0))
    breadth = float(event.get("currency_breadth") or 0.5)
    concentration = max(float(event.get("currency_propagation_concentration") or 0.0), 0.0)

    seismic = (
        math.log1p(energy_ratio)
        + math.log1p(jerk)
        + math.log1p(spread_strain)
        + math.log1p(concentration * 8.0)
    )
    continuation = (
        max(short_impulse, 0.0) * max(long_impulse, 0.0)
        * (0.5 + breadth) * max(close_location, 0.0)
        / (1.0 + spread_strain)
    )
    graph_escape = (
        graph_alignment * breadth * max(float(event["rail_escape_energy"]), 0.0)
        / (1.0 + concentration)
    )
    dynamic_shape = (
        abs(float(event["price_spread_loop_area"]))
        * (1.0 + jerk) * (1.0 + abs(short_impulse - long_impulse))
    )

    reentry_depth = max(0.0, -float(event["next_boundary_distance"]))
    failure_velocity = max(
        0.0,
        -side * math.log(next_bar.mid_c / bar.mid_c) / max(math.sqrt(median_energy), 1e-12),
    )
    spread_relaxation = bar.spread_c / max(next_bar.spread_c, 1e-12)
    failed_auction = (
        reentry_depth * failure_velocity * (1.0 + float(event["wick_rejection_ratio"]))
        / (1.0 + max(float(event["boundary_acceptance"]), 0.0))
    )
    sweep_reversal = (
        failed_auction
        * (1.0 + float(event["boundary_crowding"]))
        * (1.0 + max(float(event["tick_volume_shock"]), 0.0))
        * spread_relaxation
    )

    values = {
        "seismic_shock_index": seismic,
        "directional_impulse_short": short_impulse,
        "directional_impulse_long": long_impulse,
        "semivariance_alignment": semivariance_alignment,
        "auction_close_location": close_location,
        "escape_range_fraction": escape_range_fraction,
        "continuation_coherence": continuation,
        "graph_escape_alignment": graph_escape,
        "dynamic_shape_energy": dynamic_shape,
        "post_break_reentry_depth": reentry_depth,
        "post_break_failure_velocity": failure_velocity,
        "post_break_spread_relaxation": spread_relaxation,
        "failed_auction_score": failed_auction,
        "sweep_reversal_pressure": sweep_reversal,
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("non-finite composite feature")
    return {**event, **values}
