from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utils.env_utils import env_bool, env_float


@dataclass(frozen=True, slots=True)
class ExitForecastAdjustment:
    enabled: bool
    contra_score: float
    direction_prob: Optional[float]
    against_prob: Optional[float]
    edge_strength: float
    reason: str
    profit_take_mult: float
    trail_start_mult: float
    trail_backoff_mult: float
    lock_buffer_mult: float
    loss_cut_mult: float
    max_hold_mult: float
    price_hint_pips: Optional[float]
    range_span_pips: Optional[float]
    target_reach_prob: Optional[float]


def _safe_float(value: object) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num != num:
        return None
    return num


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _thesis_forecast(entry_thesis: Optional[dict]) -> tuple[dict, dict]:
    if not isinstance(entry_thesis, dict):
        return {}, {}
    forecast = entry_thesis.get("forecast")
    fusion = entry_thesis.get("forecast_fusion")
    return (
        dict(forecast) if isinstance(forecast, dict) else {},
        dict(fusion) if isinstance(fusion, dict) else {},
    )


def _resolve_direction_prob(side: str, forecast: dict, fusion: dict) -> Optional[float]:
    direct = _safe_float(fusion.get("direction_prob"))
    if direct is not None:
        return _clamp(direct, 0.0, 1.0)
    p_up = _safe_float(forecast.get("p_up"))
    if p_up is None:
        return None
    if str(side).strip().lower() == "long":
        return _clamp(p_up, 0.0, 1.0)
    return _clamp(1.0 - p_up, 0.0, 1.0)


def _resolve_edge_strength(forecast: dict, direction_prob: Optional[float]) -> float:
    edge_raw = _safe_float(forecast.get("edge"))
    if edge_raw is not None:
        edge_strength = _clamp((edge_raw - 0.5) / 0.5, 0.0, 1.0)
    else:
        edge_strength = 0.0
    if direction_prob is not None:
        bias_strength = _clamp(abs(direction_prob - 0.5) * 2.0, 0.0, 1.0)
        edge_strength = max(edge_strength, bias_strength)
    return edge_strength


def _resolve_price_hints(
    forecast: dict,
    *,
    env_prefix: Optional[str],
) -> tuple[Optional[float], Optional[float]]:
    pip_size = env_float("EXIT_FORECAST_PIP_SIZE", 0.01, prefix=env_prefix)
    pip_size = max(1e-6, float(pip_size))

    target_price = _safe_float(forecast.get("target_price"))
    anchor_price = _safe_float(forecast.get("anchor_price"))
    tp_hint = _safe_float(forecast.get("tp_pips_hint"))
    range_low_price = _safe_float(forecast.get("range_low_price"))
    range_high_price = _safe_float(forecast.get("range_high_price"))

    price_hint_pips: Optional[float] = None
    if target_price is not None and anchor_price is not None:
        price_hint_pips = abs(float(target_price) - float(anchor_price)) / pip_size
    elif tp_hint is not None and tp_hint > 0.0:
        price_hint_pips = abs(float(tp_hint))

    range_span_pips: Optional[float] = None
    if range_low_price is not None and range_high_price is not None:
        range_span_pips = abs(float(range_high_price) - float(range_low_price)) / pip_size

    if price_hint_pips is not None and price_hint_pips <= 0.0:
        price_hint_pips = None
    if range_span_pips is not None and range_span_pips <= 0.0:
        range_span_pips = None
    return price_hint_pips, range_span_pips


def _resolve_target_reach_prob(forecast: dict) -> Optional[float]:
    for key in ("target_reach_prob", "tp_reach_prob", "target_prob"):
        value = _safe_float(forecast.get(key))
        if value is None:
            continue
        return _clamp(value, 0.0, 1.0)
    return None


def build_exit_forecast_adjustment(
    *,
    side: str,
    entry_thesis: Optional[dict],
    env_prefix: Optional[str] = None,
) -> ExitForecastAdjustment:
    enabled = env_bool("EXIT_FORECAST_ENABLED", True, prefix=env_prefix)
    if not enabled:
        return ExitForecastAdjustment(
            enabled=False,
            contra_score=0.0,
            direction_prob=None,
            against_prob=None,
            edge_strength=0.0,
            reason="disabled",
            profit_take_mult=1.0,
            trail_start_mult=1.0,
            trail_backoff_mult=1.0,
            lock_buffer_mult=1.0,
            loss_cut_mult=1.0,
            max_hold_mult=1.0,
            price_hint_pips=None,
            range_span_pips=None,
            target_reach_prob=None,
        )

    forecast, fusion = _thesis_forecast(entry_thesis)
    direction_prob = _resolve_direction_prob(side, forecast, fusion)
    if direction_prob is None:
        return ExitForecastAdjustment(
            enabled=False,
            contra_score=0.0,
            direction_prob=None,
            against_prob=None,
            edge_strength=0.0,
            reason="missing_forecast",
            profit_take_mult=1.0,
            trail_start_mult=1.0,
            trail_backoff_mult=1.0,
            lock_buffer_mult=1.0,
            loss_cut_mult=1.0,
            max_hold_mult=1.0,
            price_hint_pips=None,
            range_span_pips=None,
            target_reach_prob=None,
        )

    against_prob = _clamp(1.0 - direction_prob, 0.0, 1.0)
    min_against = _clamp(
        env_float("EXIT_FORECAST_MIN_AGAINST_PROB", 0.56, prefix=env_prefix),
        0.5,
        0.95,
    )
    edge_strength = _resolve_edge_strength(forecast, direction_prob)
    contra_prob = _clamp((against_prob - min_against) / max(1e-6, (1.0 - min_against)), 0.0, 1.0)
    contra_score = _clamp(contra_prob * (0.35 + 0.65 * edge_strength), 0.0, 1.0)

    allowed = forecast.get("allowed")
    if allowed is False:
        disallow_floor = _clamp(
            env_float("EXIT_FORECAST_DISALLOW_CONTRA_FLOOR", 0.75, prefix=env_prefix),
            0.0,
            1.0,
        )
        contra_score = max(contra_score, disallow_floor)
    target_reach_prob = _resolve_target_reach_prob(forecast)
    if (
        target_reach_prob is not None
        and env_bool("EXIT_FORECAST_TARGET_REACH_ENABLED", True, prefix=env_prefix)
    ):
        reach_weight = _clamp(
            env_float("EXIT_FORECAST_TARGET_REACH_WEIGHT_MAX", 0.28, prefix=env_prefix),
            0.0,
            0.8,
        )
        low_prob = _clamp(
            env_float("EXIT_FORECAST_TARGET_REACH_LOW", 0.40, prefix=env_prefix),
            0.05,
            0.5,
        )
        high_prob = _clamp(
            env_float("EXIT_FORECAST_TARGET_REACH_HIGH", 0.62, prefix=env_prefix),
            0.5,
            0.95,
        )
        if high_prob <= low_prob + 0.01:
            high_prob = min(0.95, low_prob + 0.22)

        if target_reach_prob <= low_prob:
            shortfall = _clamp((low_prob - target_reach_prob) / max(1e-6, low_prob), 0.0, 1.0)
            contra_score = _clamp(
                contra_score + reach_weight * shortfall * (0.45 + 0.55 * edge_strength),
                0.0,
                1.0,
            )
        elif target_reach_prob >= high_prob and allowed is not False:
            surplus = _clamp(
                (target_reach_prob - high_prob) / max(1e-6, (1.0 - high_prob)),
                0.0,
                1.0,
            )
            relax = reach_weight * surplus * (0.30 + 0.70 * direction_prob)
            contra_score = _clamp(contra_score * (1.0 - relax), 0.0, 1.0)

    price_hint_pips, range_span_pips = _resolve_price_hints(
        forecast,
        env_prefix=env_prefix,
    )
    if env_bool("EXIT_FORECAST_PRICE_HINT_ENABLED", True, prefix=env_prefix):
        hint_weight = _clamp(
            env_float("EXIT_FORECAST_PRICE_HINT_WEIGHT_MAX", 0.20, prefix=env_prefix),
            0.0,
            0.6,
        )
        if price_hint_pips is not None:
            hint_min = max(
                0.1,
                env_float("EXIT_FORECAST_PRICE_HINT_MIN_PIPS", 0.6, prefix=env_prefix),
            )
            hint_max = max(
                hint_min + 0.1,
                env_float("EXIT_FORECAST_PRICE_HINT_MAX_PIPS", 8.0, prefix=env_prefix),
            )
            distance_norm = _clamp(
                (float(price_hint_pips) - hint_min) / max(1e-6, (hint_max - hint_min)),
                0.0,
                1.0,
            )
            # Smaller target distance -> tighten exits sooner.
            tightness = 1.0 - distance_norm
            contra_score = _clamp(
                contra_score * (1.0 + hint_weight * tightness),
                0.0,
                1.0,
            )
            # Larger projected move with aligned direction -> keep some room.
            if direction_prob >= 0.60 and allowed is not False:
                relax = hint_weight * distance_norm * 0.35
                contra_score = _clamp(contra_score * (1.0 - relax), 0.0, 1.0)
        if range_span_pips is not None:
            narrow_pips = max(
                0.1,
                env_float("EXIT_FORECAST_RANGE_HINT_NARROW_PIPS", 3.5, prefix=env_prefix),
            )
            wide_pips = max(
                narrow_pips + 0.1,
                env_float("EXIT_FORECAST_RANGE_HINT_WIDE_PIPS", 18.0, prefix=env_prefix),
            )
            if range_span_pips <= narrow_pips:
                contra_score = _clamp(
                    contra_score + hint_weight * 0.25 * (0.5 + 0.5 * edge_strength),
                    0.0,
                    1.0,
                )
            elif range_span_pips >= wide_pips and direction_prob >= 0.58:
                contra_score = _clamp(contra_score * (1.0 - hint_weight * 0.10), 0.0, 1.0)

    profit_tighten = _clamp(
        env_float("EXIT_FORECAST_PROFIT_TIGHTEN_MAX", 0.30, prefix=env_prefix),
        0.0,
        0.8,
    )
    trail_start_tighten = _clamp(
        env_float("EXIT_FORECAST_TRAIL_START_TIGHTEN_MAX", 0.34, prefix=env_prefix),
        0.0,
        0.8,
    )
    trail_backoff_tighten = _clamp(
        env_float("EXIT_FORECAST_TRAIL_BACKOFF_TIGHTEN_MAX", 0.25, prefix=env_prefix),
        0.0,
        0.8,
    )
    lock_boost = _clamp(
        env_float("EXIT_FORECAST_LOCK_BUFFER_BOOST_MAX", 0.30, prefix=env_prefix),
        0.0,
        1.5,
    )
    loss_tighten = _clamp(
        env_float("EXIT_FORECAST_LOSSCUT_TIGHTEN_MAX", 0.35, prefix=env_prefix),
        0.0,
        0.85,
    )
    hold_tighten = _clamp(
        env_float("EXIT_FORECAST_MAX_HOLD_TIGHTEN_MAX", 0.30, prefix=env_prefix),
        0.0,
        0.85,
    )

    return ExitForecastAdjustment(
        enabled=contra_score > 0.0,
        contra_score=round(contra_score, 6),
        direction_prob=round(direction_prob, 6),
        against_prob=round(against_prob, 6),
        edge_strength=round(edge_strength, 6),
        reason=str(forecast.get("reason") or ""),
        profit_take_mult=_clamp(1.0 - profit_tighten * contra_score, 0.30, 1.0),
        trail_start_mult=_clamp(1.0 - trail_start_tighten * contra_score, 0.30, 1.0),
        trail_backoff_mult=_clamp(1.0 - trail_backoff_tighten * contra_score, 0.20, 1.0),
        lock_buffer_mult=_clamp(1.0 + lock_boost * contra_score, 1.0, 2.5),
        loss_cut_mult=_clamp(1.0 - loss_tighten * contra_score, 0.20, 1.0),
        max_hold_mult=_clamp(1.0 - hold_tighten * contra_score, 0.25, 1.0),
        price_hint_pips=(
            round(float(price_hint_pips), 6) if price_hint_pips is not None else None
        ),
        range_span_pips=(
            round(float(range_span_pips), 6) if range_span_pips is not None else None
        ),
        target_reach_prob=(
            round(float(target_reach_prob), 6) if target_reach_prob is not None else None
        ),
    )


def apply_exit_forecast_to_targets(
    *,
    profit_take: float,
    trail_start: float,
    trail_backoff: float,
    lock_buffer: float,
    adjustment: ExitForecastAdjustment,
    profit_take_floor: float = 0.1,
    trail_start_floor: float = 0.1,
    trail_backoff_floor: float = 0.05,
    lock_buffer_floor: float = 0.05,
) -> tuple[float, float, float, float]:
    if not adjustment.enabled:
        return profit_take, trail_start, trail_backoff, lock_buffer
    adjusted_profit = max(float(profit_take_floor), float(profit_take) * adjustment.profit_take_mult)
    adjusted_trail_start = max(float(trail_start_floor), float(trail_start) * adjustment.trail_start_mult)
    adjusted_trail_backoff = max(float(trail_backoff_floor), float(trail_backoff) * adjustment.trail_backoff_mult)
    adjusted_lock_buffer = max(float(lock_buffer_floor), float(lock_buffer) * adjustment.lock_buffer_mult)
    return adjusted_profit, adjusted_trail_start, adjusted_trail_backoff, adjusted_lock_buffer


def apply_exit_forecast_to_loss_cut(
    *,
    soft_pips: float,
    hard_pips: float,
    max_hold_sec: Optional[float],
    adjustment: ExitForecastAdjustment,
    floor_pips: float = 0.1,
) -> tuple[float, float, Optional[float]]:
    if not adjustment.enabled:
        return soft_pips, hard_pips, max_hold_sec
    floor = float(floor_pips)
    soft_adj = max(floor, float(soft_pips) * adjustment.loss_cut_mult)
    hard_adj = max(soft_adj, float(hard_pips) * adjustment.loss_cut_mult)
    if max_hold_sec is not None and max_hold_sec > 0.0:
        hold_adj = max(1.0, float(max_hold_sec) * adjustment.max_hold_mult)
    else:
        hold_adj = max_hold_sec
    return soft_adj, hard_adj, hold_adj
