"""
Helpers to derive scalp signals from recent tick data.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Optional

from market_data import tick_window

from . import config


@dataclass(frozen=True)
class SignalFeatures:
    latest_mid: float
    spread_pips: float
    momentum_pips: float
    short_momentum_pips: float
    range_pips: float
    tick_count: int
    span_seconds: float
    impulse_pips: float = 0.0
    impulse_span_sec: float = 0.0
    impulse_direction: int = 0
    consolidation_range_pips: float = 0.0
    consolidation_span_sec: float = 0.0
    consolidation_ok: bool = False
    rsi: Optional[float] = None
    atr_pips: Optional[float] = None
    rsi_source: str = "tick"
    atr_source: str = "tick"
    pattern_tag: str = "unknown"
    pattern_features: Optional[tuple[float, ...]] = None


def _as_pips(delta: float) -> float:
    return delta / config.PIP_VALUE


def _window_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return mean(values)


def span_requirement_ok(span_seconds: float, tick_count: int) -> bool:
    span_required = config.MIN_ENTRY_TICK_SPAN_SEC
    span_ok = span_seconds >= span_required
    if span_ok:
        return True
    relaxed_floor = span_required * config.MIN_SPAN_RELAX_RATIO
    tick_buffer = config.MIN_SPAN_RELAX_TICK_BUFFER
    if (
        span_seconds >= relaxed_floor
        and tick_count >= config.MIN_ENTRY_TICK_COUNT + tick_buffer
    ):
        return True
    return False


def _compute_rsi(prices: list[float], period: int) -> Optional[float]:
    if len(prices) < 2:
        return None
    effective_period = min(period, len(prices) - 1)
    gains = []
    losses = []
    for i in range(1, len(prices)):
        diff = prices[i] - prices[i - 1]
        if diff >= 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-diff)
    if not gains:
        return 50.0
    slice_gains = gains[-effective_period:]
    slice_losses = losses[-effective_period:]
    avg_gain = sum(slice_gains) / effective_period
    avg_loss = sum(slice_losses) / effective_period
    if avg_gain == 0.0 and avg_loss == 0.0:
        return 50.0
    if avg_loss == 0.0:
        return 100.0
    if avg_gain == 0.0:
        return 0.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_atr(prices: list[float], period: int) -> Optional[float]:
    if len(prices) <= 1:
        return None
    true_ranges = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
    if not true_ranges:
        return 0.0
    effective_period = min(period, len(true_ranges))
    return mean(true_ranges[-effective_period:]) / config.PIP_VALUE


def _resolve_indicators(mids: list[float]) -> tuple[Optional[float], str, Optional[float], str]:
    """
    Resolve RSI / ATR from tick mids, optionally extending the window when the
    short sample is too thin. Returns both the value and the source label to
    aid downstream logging.
    """
    rsi = _compute_rsi(mids, config.RSI_PERIOD)
    atr_pips = _compute_atr(mids, config.ATR_PERIOD)
    rsi_source = "tick" if rsi is not None else "missing"
    atr_source = "tick" if atr_pips is not None else "missing"

    if (rsi is None or atr_pips is None) and len(mids) < config.RSI_PERIOD + 1:
        extended_ticks = tick_window.recent_ticks(
            seconds=min(config.LONG_WINDOW_SEC * 3, 30.0), limit=360
        )
        extended_mids = [float(t["mid"]) for t in extended_ticks] if extended_ticks else []
        if len(extended_mids) >= 2:
            if rsi is None:
                rsi_ext = _compute_rsi(extended_mids, config.RSI_PERIOD)
                if rsi_ext is not None:
                    rsi = rsi_ext
                    rsi_source = "extended"
            if atr_pips is None:
                atr_ext = _compute_atr(extended_mids, config.ATR_PERIOD)
                if atr_ext is not None:
                    atr_pips = atr_ext
                    atr_source = "extended"

    if rsi is None:
        rsi_source = "missing"
    if atr_pips is None:
        atr_source = "missing"

    return rsi, rsi_source, atr_pips, atr_source


def _classify_pattern(mids: list[float]) -> str:
    n = len(mids)
    if n < 6:
        return "thin_data"

    pip = config.PIP_VALUE
    latest = mids[-1]
    first = mids[0]
    max_mid = max(mids)
    min_mid = min(mids)
    range_pips = (max_mid - min_mid) / pip if pip else 0.0

    short_window = min(5, n - 1)
    long_mom = (latest - first) / pip if pip else 0.0
    short_mom = (
        (latest - mids[-short_window - 1]) / pip if short_window > 0 and pip else long_mom
    )

    avg_abs_step = sum(abs(mids[i] - mids[i - 1]) for i in range(1, n))
    avg_abs_step = avg_abs_step / ((n - 1) * pip) if pip and n > 1 else 0.0

    max_idx = max(range(n), key=lambda i: (mids[i], i))
    min_idx = min(range(n), key=lambda i: (mids[i], -i))

    recent_high = max_idx >= n - 3
    recent_low = min_idx >= n - 3

    if range_pips < 0.25 and avg_abs_step < 0.05:
        return "flat_chop"
    if range_pips < 0.45 and abs(long_mom) < 0.2:
        return "narrow_range"
    if recent_high and short_mom < -0.2:
        return "spike_reversal_down"
    if recent_low and short_mom > 0.2:
        return "spike_reversal_up"
    if long_mom > 0.6 and short_mom > 0.3:
        return "impulse_up"
    if long_mom < -0.6 and short_mom < -0.3:
        return "impulse_down"
    if long_mom > 0.6 and short_mom < -0.1:
        return "pullback_after_impulse_up"
    if long_mom < -0.6 and short_mom > 0.1:
        return "pullback_after_impulse_down"
    if range_pips >= 1.0 and abs(short_mom) < 0.2:
        return "broad_range_stall"
    return "neutral"


def _pattern_feature_vector(ticks: list[dict[str, float]], mids: list[float]) -> tuple[float, ...]:
    pip = config.PIP_VALUE if config.PIP_VALUE else 0.01
    latest_epoch = float(ticks[-1]["epoch"])

    def window_stats(seconds: float) -> tuple[float, float, float, float]:
        subset = [float(t["mid"]) for t in ticks if latest_epoch - float(t["epoch"]) <= seconds]
        if len(subset) < 2:
            subset = mids[-min(len(mids), 2):]
        slope = (subset[-1] - subset[0]) / pip if len(subset) >= 2 else 0.0
        rng = (max(subset) - min(subset)) / pip if subset else 0.0
        std = pstdev(subset) / pip if len(subset) >= 2 else 0.0
        density = len(subset) / max(seconds, 1.0)
        return slope, rng, std, density

    features: list[float] = []
    for window in (3.0, 6.0, 15.0, 30.0):
        features.extend(window_stats(window))

    overall_slope = (mids[-1] - mids[0]) / pip if len(mids) >= 2 else 0.0
    overall_range = (max(mids) - min(mids)) / pip if mids else 0.0
    features.append(overall_slope)
    features.append(overall_range)
    features.append(float(len(mids)))
    span = float(ticks[-1]["epoch"] - ticks[0]["epoch"]) if len(ticks) >= 2 else 0.0
    features.append(span)
    return tuple(features)


def extract_features(
    spread_pips: float, *, ticks: Optional[list[dict[str, float]]] = None
) -> Optional[SignalFeatures]:
    if ticks is None:
        ticks = tick_window.recent_ticks(seconds=config.LONG_WINDOW_SEC, limit=180)
    if len(ticks) < config.MIN_TICK_COUNT:
        return None

    mids = [float(t["mid"]) for t in ticks]
    epochs = [float(t.get("epoch", 0.0)) for t in ticks]
    latest_mid = mids[-1]
    latest_epoch = epochs[-1]
    long_mean = _window_mean(mids)

    short_window = max(5, int(len(mids) * config.SHORT_WINDOW_SEC / config.LONG_WINDOW_SEC))
    short_slice = mids[-short_window:]
    short_mean = _window_mean(short_slice)

    high_mid = max(mids)
    low_mid = min(mids)

    span_seconds = float(latest_epoch - epochs[0])

    momentum = _as_pips(latest_mid - long_mean)
    short_momentum = _as_pips(latest_mid - short_mean)
    range_pips = _as_pips(high_mid - low_mid)
    rsi, rsi_source, atr_pips, atr_source = _resolve_indicators(mids)
    pattern_tag = _classify_pattern(mids)
    pattern_features = _pattern_feature_vector(ticks, mids)

    impulse_cutoff = latest_epoch - config.IMPULSE_LOOKBACK_SEC
    impulse_indices = [i for i, epoch in enumerate(epochs) if epoch >= impulse_cutoff]
    impulse_pips = 0.0
    impulse_span_sec = 0.0
    impulse_direction = 0
    if len(impulse_indices) >= config.IMPULSE_MIN_TICKS:
        start_idx = impulse_indices[0]
        impulse_delta = mids[-1] - mids[start_idx]
        impulse_pips = abs(_as_pips(impulse_delta))
        impulse_span_sec = max(0.0, latest_epoch - epochs[start_idx])
        if impulse_delta > 0:
            impulse_direction = 1
        elif impulse_delta < 0:
            impulse_direction = -1

    consolidation_cutoff = latest_epoch - config.CONSOLIDATION_WINDOW_SEC
    consolidation_indices = [
        i for i, epoch in enumerate(epochs) if epoch >= consolidation_cutoff
    ]
    consolidation_range_pips = 0.0
    consolidation_span_sec = 0.0
    consolidation_ok = False
    if len(consolidation_indices) >= config.CONSOLIDATION_MIN_TICKS:
        subset = [mids[i] for i in consolidation_indices]
        consolidation_range_pips = _as_pips(max(subset) - min(subset))
        consolidation_span_sec = max(
            0.0,
            epochs[consolidation_indices[-1]] - epochs[consolidation_indices[0]],
        )
        consolidation_ok = consolidation_range_pips <= config.CONSOLIDATION_MAX_RANGE_PIPS

    return SignalFeatures(
        latest_mid=latest_mid,
        spread_pips=spread_pips,
        momentum_pips=momentum,
        short_momentum_pips=short_momentum,
        range_pips=range_pips,
        tick_count=len(ticks),
        span_seconds=span_seconds,
        impulse_pips=impulse_pips,
        impulse_span_sec=impulse_span_sec,
        impulse_direction=impulse_direction,
        consolidation_range_pips=consolidation_range_pips,
        consolidation_span_sec=consolidation_span_sec,
        consolidation_ok=consolidation_ok,
        rsi=rsi,
        atr_pips=atr_pips,
        rsi_source=rsi_source,
        atr_source=atr_source,
        pattern_tag=pattern_tag,
        pattern_features=pattern_features,
    )


def evaluate_signal(
    features: SignalFeatures,
    *,
    m1_rsi: Optional[float] = None,
    range_active: bool = False,
) -> Optional[str]:
    # 動的なレンジ/モメンタム閾値を ATR/スプレッドから導出
    atr = features.atr_pips or 0.0
    spread = max(0.0, float(features.spread_pips))
    dyn_range_floor = max(
        config.ENTRY_RANGE_FLOOR_PIPS,
        spread * config.ENTRY_RANGE_SPREAD_COEF,
    )
    dyn_mom = max(
        config.ENTRY_THRESHOLD_PIPS,
        spread * config.ENTRY_MOM_SPREAD_COEF,
        atr * config.ENTRY_MOM_ATR_COEF,
    )
    dyn_short_mom = max(config.ENTRY_SHORT_THRESHOLD_PIPS, dyn_mom * 0.8)

    if range_active:
        dyn_range_floor *= 0.8
        dyn_mom *= 0.85
        dyn_short_mom *= 0.75

    if features.atr_pips is None or features.atr_pips < config.MIN_ENTRY_ATR_PIPS:
        return None
    if features.tick_count < config.MIN_ENTRY_TICK_COUNT:
        return None
    if not span_requirement_ok(features.span_seconds, features.tick_count):
        return None
    if features.range_pips < dyn_range_floor:
        return None
    if abs(features.momentum_pips) < dyn_mom:
        return None
    if abs(features.short_momentum_pips) < dyn_short_mom:
        return None
    if features.tick_count < config.MIN_TICK_COUNT:
        return None
    if features.span_seconds <= 0.0:
        return None

    if not config.FORCE_ENTRIES:
        if config.MIN_IMPULSE_PIPS > 0.0 and features.impulse_pips < config.MIN_IMPULSE_PIPS:
            return None
        if features.impulse_direction == 0:
            return None
        if config.REQUIRE_CONSOLIDATION and not features.consolidation_ok:
            return None

    momentum = features.momentum_pips
    short_momentum = features.short_momentum_pips
    abs_momentum = abs(momentum)
    action: Optional[str] = None

    if features.impulse_direction > 0 and momentum < 0:
        return None
    if features.impulse_direction < 0 and momentum > 0:
        return None

    tick_rsi = features.rsi
    effective_rsi = tick_rsi
    if effective_rsi is None:
        effective_rsi = m1_rsi if m1_rsi is not None else 50.0

    m1_bias: Optional[str] = None
    if m1_rsi is not None:
        if m1_rsi >= config.M1_RSI_SHORT_MAX:
            m1_bias = "long"
        elif m1_rsi <= config.M1_RSI_LONG_MIN:
            m1_bias = "short"

    candidate_long = momentum > 0 and abs_momentum >= dyn_mom
    candidate_long = candidate_long and short_momentum >= dyn_short_mom
    candidate_short = momentum < 0 and abs_momentum >= dyn_mom
    candidate_short = candidate_short and short_momentum <= -dyn_short_mom

    # Determine base direction by stronger impulse.
    if candidate_long and candidate_short:
        action = "OPEN_LONG" if abs_momentum >= abs(short_momentum) else "OPEN_SHORT"
    elif candidate_long:
        action = "OPEN_LONG"
    elif candidate_short:
        action = "OPEN_SHORT"
    else:
        return None

    # Apply RSI gating with fallback.
    if action == "OPEN_LONG" and effective_rsi >= config.RSI_SHORT_MAX and short_momentum <= -config.ENTRY_SHORT_THRESHOLD_PIPS:
        action = "REVERSAL_SHORT"
    elif action == "OPEN_SHORT" and effective_rsi <= config.RSI_LONG_MIN and short_momentum >= config.ENTRY_SHORT_THRESHOLD_PIPS:
        action = "REVERSAL_LONG"

    # Incorporate M1 bias: if conflicting strongly, mark as reversal.
    if m1_bias == "long" and action.startswith("OPEN_SHORT") and short_momentum >= config.ENTRY_SHORT_THRESHOLD_PIPS:
        action = "REVERSAL_LONG"
    elif m1_bias == "short" and action.startswith("OPEN_LONG") and short_momentum <= -config.ENTRY_SHORT_THRESHOLD_PIPS:
        action = "REVERSAL_SHORT"

    # If both reversal flags triggered, pick based on RSI bias.
    if action.startswith("REVERSAL"):
        # Ensure reversal still has adequate opposite momentum.
        if action.endswith("LONG") and short_momentum < config.ENTRY_SHORT_THRESHOLD_PIPS:
            action = "OPEN_LONG" if candidate_long else None
        elif action.endswith("SHORT") and short_momentum > -config.ENTRY_SHORT_THRESHOLD_PIPS:
            action = "OPEN_SHORT" if candidate_short else None

    if range_active and action in {"OPEN_LONG", "OPEN_SHORT"}:
        # Rangeモードでは、直近モメンタムが逆向きに強い場合は素直に逆張り扱いに切り替える
        if action == "OPEN_LONG" and short_momentum <= -dyn_short_mom:
            action = "REVERSAL_SHORT"
        elif action == "OPEN_SHORT" and short_momentum >= dyn_short_mom:
            action = "REVERSAL_LONG"

    valid_actions = {"OPEN_LONG", "OPEN_SHORT", "REVERSAL_LONG", "REVERSAL_SHORT"}
    if action in {"OPEN_LONG", "REVERSAL_LONG"} and tick_rsi is not None and tick_rsi >= config.RSI_ENTRY_OVERBOUGHT:
        return None
    if action in {"OPEN_SHORT", "REVERSAL_SHORT"} and tick_rsi is not None and tick_rsi <= config.RSI_ENTRY_OVERSOLD:
        return None

    if action not in valid_actions:
        return None
    return action
