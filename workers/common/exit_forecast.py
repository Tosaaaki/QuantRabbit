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
