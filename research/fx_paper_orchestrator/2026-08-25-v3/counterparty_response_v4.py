from __future__ import annotations

import math
import statistics

from causal_composite_indicators_v3 import Bar


FEATURES = (
    "crs_excursion_area",
    "crs_opposing_wick_absorption",
    "crs_reclaim_velocity",
    "crs_reclaim_curvature",
    "crs_spread_shock",
    "crs_spread_recovery",
    "crs_graph_propagation",
    "ptr_arc_chord_ratio",
    "ptr_signed_triangle_area",
    "ptr_turning_angle",
    "ptr_session_centroid_distance",
    "lsr_volume_anomaly",
    "lsr_range_energy",
    "lsr_jerk",
    "counterparty_response_surface",
)


def _median(values: list[float], floor: float = 1e-12) -> float:
    return max(float(statistics.median(values)), floor)


def _turning_angle(left: tuple[float, float], right: tuple[float, float]) -> float:
    dot = left[0] * right[0] + left[1] * right[1]
    cross = left[0] * right[1] - left[1] * right[0]
    return math.atan2(cross, dot)


def counterparty_features(event: dict, bars: list[Bar], lookback: int = 24) -> dict:
    """Compute a delayed, completed-bar response surface.

    If the excursion bar is i, the decision is allowed only after i+1 is
    completed. No bar after i+1 is read. The first executable fill therefore
    belongs strictly after the timestamp of i+1.
    """
    i = int(event["breakout_index"])
    if i < max(lookback, 3) or i + 1 >= len(bars):
        raise ValueError("event lacks completed pre-break and response context")
    side = int(event["escape_side"])
    if side not in (-1, 1):
        raise ValueError("escape_side must be -1 or 1")
    scale = max(float(event["scale"]), 1e-12)
    history = bars[i - lookback:i]
    breakout, response = bars[i], bars[i + 1]
    price_scale = scale / max(breakout.mid_c, 1e-12)

    log_prices = [math.log(bars[j].mid_c) for j in range(i - 2, i + 2)]
    normalized = [(value - log_prices[0]) / max(price_scale, 1e-12) for value in log_prices]
    segments = [normalized[j + 1] - normalized[j] for j in range(3)]
    arc = sum(math.hypot(1.0, change) for change in segments)
    chord = math.hypot(3.0, normalized[-1] - normalized[0])
    arc_chord = arc / max(chord, 1e-12)
    triangle = side * 0.5 * (
        normalized[-3] - 2.0 * normalized[-2] + normalized[-1]
    )
    turn = side * _turning_angle((1.0, segments[-2]), (1.0, segments[-1]))

    hist_returns = [
        math.log(right.mid_c / left.mid_c)
        for left, right in zip(history, history[1:])
    ]
    vol = math.sqrt(statistics.fmean(value * value for value in hist_returns)) if hist_returns else 1e-12
    vol = max(vol, 1e-12)
    break_return = math.log(breakout.mid_c / bars[i - 1].mid_c)
    response_return = math.log(response.mid_c / breakout.mid_c)
    reclaim_velocity = -side * response_return / vol
    reclaim_curvature = -side * (response_return - break_return) / vol

    breakout_range = max(breakout.mid_h - breakout.mid_l, scale * 0.05)
    opposing_wick = (
        breakout.mid_h - max(breakout.mid_o, breakout.mid_c)
        if side > 0 else min(breakout.mid_o, breakout.mid_c) - breakout.mid_l
    ) / breakout_range
    volume_anomaly = breakout.volume / _median([bar.volume for bar in history], 1.0)
    spread_shock = breakout.spread_c / _median([bar.spread_c for bar in history])
    spread_recovery = breakout.spread_c / max(response.spread_c, 1e-12)
    ranges = [(bar.mid_h - bar.mid_l) / max(bar.mid_c, 1e-12) for bar in history]
    range_energy = ((breakout.mid_h - breakout.mid_l) / max(breakout.mid_c, 1e-12)) ** 2
    range_energy /= _median([value * value for value in ranges])
    jerk = abs(response_return - break_return) / vol

    excursion = max(float(event.get("rail_escape_energy", 0.0)), 0.0)
    reentry = max(-float(event.get("next_boundary_distance", 0.0)), 0.0)
    excursion_area = excursion + 0.5 * max(excursion - reentry, 0.0)
    graph = max(float(event.get("currency_propagation", 0.0) or 0.0), 0.0)
    centroid = statistics.fmean(bar.mid_c for bar in history)
    centroid_distance = side * (response.mid_c - centroid) / scale

    absorption = max(opposing_wick, 0.0) * max(volume_anomaly, 0.0)
    response_surface = math.tanh(
        absorption
        + max(reclaim_velocity, 0.0)
        + 0.5 * max(reclaim_curvature, 0.0)
        + 0.25 * max(spread_recovery - 1.0, 0.0)
        + 0.25 * graph
        - 0.5 * max(centroid_distance, 0.0)
    )
    values = {
        "crs_excursion_area": excursion_area,
        "crs_opposing_wick_absorption": absorption,
        "crs_reclaim_velocity": reclaim_velocity,
        "crs_reclaim_curvature": reclaim_curvature,
        "crs_spread_shock": spread_shock,
        "crs_spread_recovery": spread_recovery,
        "crs_graph_propagation": graph,
        "ptr_arc_chord_ratio": arc_chord,
        "ptr_signed_triangle_area": triangle,
        "ptr_turning_angle": turn,
        "ptr_session_centroid_distance": centroid_distance,
        "lsr_volume_anomaly": volume_anomaly,
        "lsr_range_energy": range_energy,
        "lsr_jerk": jerk,
        "counterparty_response_surface": response_surface,
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("non-finite counterparty response feature")
    return {**event, **values, "response_completed_index": i + 1}
