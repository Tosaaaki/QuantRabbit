"""Pure historical exact-S5 multi-timeframe technical research grid.

The module deliberately has no file, network, broker, runtime, or promotion
surface.  It consumes one immutable chronological S5 bid/ask stream, freezes
signals only from completed UTC buckets, and evaluates fixed diagnostic arms.
"""

from __future__ import annotations

import math
import hashlib
import json
import random
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence

from quant_rabbit.instruments import instrument_pip_factor
from quant_rabbit.technical_forecast_forward_outcome import S5BidAskCandle


GRID_CONTRACT = "QR_CAUSAL_MULTITF_S5_GRID_V1"
GLOBAL_CONTRACT = "QR_CAUSAL_MULTITF_S5_GRID_GLOBAL_V1"
TIMEFRAMES_SECONDS = {"M1": 60, "M5": 300, "M15": 900, "H1": 3600}
ORIENTATIONS = ("DIRECT", "INVERSE")

# These are preregistered diagnostic research choices, not production market
# constants.  A later study must add a versioned policy rather than retune V1.
BASE_TP_ATR = 1.0
BASE_SL_ATR = 1.0
BASE_HOLD_SECONDS = 900
BASE_TTL_SECONDS = 90
ATR_PERIOD = 14
ADX_PERIOD = 14
RSI_PERIOD = 14
EMA_FAST_PERIOD = 12
EMA_SLOW_PERIOD = 26
EMA_TREND_PERIOD = 50
BREAKOUT_LOOKBACK = 20
MIN_VALIDATION_CLUSTER_DAYS = 8
SIGN_FLIP_EXACT_MAX_DAYS = 16
SIGN_FLIP_MONTE_CARLO_REPETITIONS = 65536
BOOTSTRAP_INTERVAL_REPETITIONS = 4096


_AUTHORITY: dict[str, Any] = {
    "historical_only": True,
    "diagnostic_only": True,
    "shadow_only": True,
    "order_authority": "NONE",
    "live_permission": False,
    "live_order_enabled": False,
    "promotion_allowed": False,
    "automatic_promotion_allowed": False,
    "broker_mutation_allowed": False,
}


@dataclass(frozen=True)
class UtcSplit:
    """One explicit half-open UTC research partition."""

    name: str
    from_utc: datetime
    to_utc: datetime


@dataclass(frozen=True)
class ArmSpec:
    """One-factor-at-a-time exit/activation variant."""

    arm_id: str
    tp_atr_multiple: float = BASE_TP_ATR
    sl_atr_multiple: float = BASE_SL_ATR
    hold_seconds: int = BASE_HOLD_SECONDS
    ttl_seconds: int = BASE_TTL_SECONDS
    complexity: int = 0

    @property
    def name(self) -> str:
        return self.arm_id


@dataclass(frozen=True)
class TechnicalSpec:
    """A role-separated technical rule; regime inputs never choose direction."""

    hypothesis_id: str
    family: str
    regime_role: str
    direction_role: str
    trigger_role: str
    complexity: int
    no_trade_control: bool = False

    @property
    def name(self) -> str:
        return self.hypothesis_id


def build_predeclared_catalog_v1() -> tuple[TechnicalSpec, ...]:
    """Return the immutable H01-H08 role-based research catalog."""

    return (
        TechnicalSpec(
            "H01",
            "TREND_CONTINUATION",
            "M15_H1_ATR_ADX_TREND",
            "M5_DI_EMA",
            "M1_RSI_MACD",
            1,
        ),
        TechnicalSpec(
            "H02",
            "TREND_PULLBACK",
            "M15_H1_ATR_ADX_TREND",
            "M5_DI_EMA",
            "M1_RSI50_RECLAIM",
            2,
        ),
        TechnicalSpec(
            "H03",
            "RANGE_ROTATION",
            "M5_M15_LOW_ADX_RANGE",
            "M1_RANGE_LOCATION",
            "M1_RSI_REENTRY",
            2,
        ),
        TechnicalSpec(
            "H04",
            "PRETREND_BREAKOUT",
            "M15_ATR_COMPRESSION",
            "M5_DI_EMA",
            "M1_BREAKOUT",
            3,
        ),
        TechnicalSpec(
            "H05",
            "BREAKOUT_FAILURE",
            "M5_M15_RANGE",
            "M1_FAILED_BREAK_SIDE",
            "M1_REACCEPTANCE",
            3,
        ),
        TechnicalSpec(
            "H06",
            "EXHAUSTION_REVERSAL",
            "M15_H1_ATR_ADX_DECELERATION",
            "M5_DI_EMA_OPPOSITE",
            "M1_RSI_MACD_REVERSAL",
            4,
        ),
        TechnicalSpec(
            "H07",
            "SESSION_OPEN_EXPANSION",
            "UTC_SESSION_ATR_EXPANSION",
            "M5_DI_EMA",
            "M1_BREAKOUT_MACD",
            4,
        ),
        TechnicalSpec("H08", "NO_TRADE_CONTROL", "CONTROL", "EITHER", "NONE", 0, True),
    )


def build_predeclared_arms_v1() -> tuple[ArmSpec, ...]:
    """Return the immutable 13-arm OFAT vehicle grid."""

    return (
        ArmSpec("BASE"),
        ArmSpec("TP050", tp_atr_multiple=0.50, complexity=1),
        ArmSpec("TP075", tp_atr_multiple=0.75, complexity=1),
        ArmSpec("TP125", tp_atr_multiple=1.25, complexity=1),
        ArmSpec("SL075", sl_atr_multiple=0.75, complexity=1),
        ArmSpec("SL125", sl_atr_multiple=1.25, complexity=1),
        ArmSpec("SL150", sl_atr_multiple=1.50, complexity=1),
        ArmSpec("HOLD180", hold_seconds=180, complexity=1),
        ArmSpec("HOLD300", hold_seconds=300, complexity=1),
        ArmSpec("HOLD600", hold_seconds=600, complexity=1),
        ArmSpec("HOLD1800", hold_seconds=1800, complexity=1),
        ArmSpec("TTL45", ttl_seconds=45, complexity=1),
        ArmSpec("TTL180", ttl_seconds=180, complexity=1),
    )


@dataclass(frozen=True)
class _Bar:
    timeframe: str
    start: datetime
    end: datetime
    open: float
    high: float
    low: float
    close: float
    ticks: int


@dataclass(frozen=True)
class _Feature:
    timeframe: str
    completed_at: datetime
    close: float
    atr: float | None
    adx: float | None
    plus_di: float | None
    minus_di: float | None
    ema_fast: float | None
    ema_slow: float | None
    ema_trend: float | None
    rsi: float | None
    macd_hist: float | None
    previous_rsi: float | None
    previous_macd_hist: float | None
    previous_atr: float | None
    previous_adx: float | None
    prior_high: float | None
    prior_low: float | None
    current_high: float
    current_low: float
    current_open: float


@dataclass
class _Bucket:
    timeframe: str
    start: datetime
    end: datetime
    open: float
    high: float
    low: float
    close: float
    ticks: int = 1

    def add(self, candle: S5BidAskCandle) -> None:
        mid_h = (candle.bid_h + candle.ask_h) / 2.0
        mid_l = (candle.bid_l + candle.ask_l) / 2.0
        self.high = max(self.high, mid_h)
        self.low = min(self.low, mid_l)
        self.close = (candle.bid_c + candle.ask_c) / 2.0
        self.ticks += 1

    def finish(self) -> _Bar:
        return _Bar(
            timeframe=self.timeframe,
            start=self.start,
            end=self.end,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            ticks=self.ticks,
        )


@dataclass
class _Position:
    candidate_id: str
    hypothesis_id: str
    orientation: str
    arm: ArmSpec
    side: str
    split_name: str
    activation: datetime
    expiry: datetime
    atr: float
    signal_mid: float
    filled: bool = False
    entry_at: datetime | None = None
    entry_exec: float | None = None
    entry_mid: float | None = None
    take_profit: float | None = None
    stop_loss: float | None = None
    hold_at: datetime | None = None


def _candidate_rows() -> tuple[tuple[TechnicalSpec, str, ArmSpec, str], ...]:
    rows: list[tuple[TechnicalSpec, str, ArmSpec, str]] = []
    for spec in build_predeclared_catalog_v1():
        if spec.no_trade_control:
            continue
        for orientation in ORIENTATIONS:
            for arm in build_predeclared_arms_v1():
                candidate_id = f"{spec.hypothesis_id}:{orientation}:{arm.arm_id}"
                rows.append((spec, orientation, arm, candidate_id))
    return tuple(rows)


def _simplicity_key(orientation: str, arm: ArmSpec) -> tuple[int, int, float, str]:
    """Frozen one-SE tie-break: direction, base, geometry distance, arm id."""

    orientation_rank = 0 if orientation == "DIRECT" else 1
    if arm.arm_id == "BASE":
        return orientation_rank, 0, 0.0, arm.arm_id
    if arm.arm_id.startswith(("TP", "SL")):
        distance = max(
            abs(arm.tp_atr_multiple - BASE_TP_ATR),
            abs(arm.sl_atr_multiple - BASE_SL_ATR),
        )
        return orientation_rank, 1, distance, arm.arm_id
    return orientation_rank, 2, 0.0, arm.arm_id


def _canonical_sha(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _blank_stat() -> dict[str, Any]:
    return {
        "raw_signal_count": 0,
        "signal_count": 0,
        "deoverlap_count": 0,
        "embargoed_signal_count": 0,
        "expired_unfilled_count": 0,
        "filled_count": 0,
        "resolved_count": 0,
        "purged_count": 0,
        "unresolved_count": 0,
        "win_count": 0,
        "loss_count": 0,
        "flat_count": 0,
        "gross_mid_pips": 0.0,
        "spread_drag_pips": 0.0,
        "exact_net_pips": 0.0,
        "gross_mid_r": 0.0,
        "spread_drag_r": 0.0,
        "exact_net_r": 0.0,
        "gross_profit_pips": 0.0,
        "gross_loss_pips": 0.0,
        "gross_profit_r": 0.0,
        "gross_loss_r": 0.0,
        "equity_pips": 0.0,
        "peak_equity_pips": 0.0,
        "max_drawdown_pips": 0.0,
        "reason_counts": defaultdict(int),
    }


def _normalise_splits(splits: Sequence[UtcSplit]) -> tuple[UtcSplit, ...]:
    if isinstance(splits, (str, bytes)) or not isinstance(splits, Sequence):
        raise ValueError("splits must be a sequence of UtcSplit values")
    normalized = tuple(splits)
    if not normalized:
        raise ValueError("at least one UTC split is required")
    names: set[str] = set()
    previous_end: datetime | None = None
    for split in normalized:
        if split.__class__ is not UtcSplit:
            raise ValueError("split class is invalid")
        name = str(split.name).strip().upper()
        if not name or name in names:
            raise ValueError("split names must be non-empty and unique")
        names.add(name)
        start = _aware_utc(split.from_utc, "split.from_utc")
        end = _aware_utc(split.to_utc, "split.to_utc")
        if start >= end:
            raise ValueError("split interval must be positive")
        if previous_end is not None and start < previous_end:
            raise ValueError(
                "split intervals must be chronological and non-overlapping"
            )
        previous_end = end
    return normalized


def _split_for(instant: datetime, splits: Sequence[UtcSplit]) -> UtcSplit | None:
    for split in splits:
        if split.from_utc <= instant < split.to_utc:
            return split
    return None


def _aware_utc(value: datetime, name: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{name} must be timezone-aware UTC")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must use UTC offset zero")
    return value.astimezone(timezone.utc)


def _validate_candle(candle: S5BidAskCandle, previous: datetime | None) -> datetime:
    if candle.__class__ is not S5BidAskCandle:
        raise ValueError("candles must contain exact S5BidAskCandle values")
    stamp = _aware_utc(candle.timestamp_utc, "candle.timestamp_utc")
    if stamp.microsecond or int(stamp.timestamp()) % 5:
        raise ValueError(
            "S5 candle timestamp must lie on the exact five-second UTC grid"
        )
    if previous is not None and stamp <= previous:
        raise ValueError("S5 candles must be chronological and unique")
    prices = (
        candle.bid_o,
        candle.bid_h,
        candle.bid_l,
        candle.bid_c,
        candle.ask_o,
        candle.ask_h,
        candle.ask_l,
        candle.ask_c,
    )
    if any(
        isinstance(value, bool) or not math.isfinite(value) or value <= 0.0
        for value in prices
    ):
        raise ValueError("S5 candle prices must be finite and positive")
    if not (
        candle.bid_l
        <= min(candle.bid_o, candle.bid_c)
        <= max(candle.bid_o, candle.bid_c)
        <= candle.bid_h
        and candle.ask_l
        <= min(candle.ask_o, candle.ask_c)
        <= max(candle.ask_o, candle.ask_c)
        <= candle.ask_h
    ):
        raise ValueError("S5 candle OHLC geometry is invalid")
    if not all(
        bid <= ask
        for bid, ask in (
            (candle.bid_o, candle.ask_o),
            (candle.bid_h, candle.ask_h),
            (candle.bid_l, candle.ask_l),
            (candle.bid_c, candle.ask_c),
        )
    ):
        raise ValueError("S5 bid/ask candle is crossed")
    complete = getattr(candle, "complete", True)
    if complete is not True:
        raise ValueError("S5 candle must be complete")
    return stamp


def _bucket_start(stamp: datetime, seconds: int) -> datetime:
    epoch = int(stamp.timestamp())
    return datetime.fromtimestamp(epoch - epoch % seconds, tz=timezone.utc)


def _new_bucket(timeframe: str, candle: S5BidAskCandle) -> _Bucket:
    seconds = TIMEFRAMES_SECONDS[timeframe]
    start = _bucket_start(candle.timestamp_utc, seconds)
    mid_o = (candle.bid_o + candle.ask_o) / 2.0
    return _Bucket(
        timeframe=timeframe,
        start=start,
        end=start + timedelta(seconds=seconds),
        open=mid_o,
        high=(candle.bid_h + candle.ask_h) / 2.0,
        low=(candle.bid_l + candle.ask_l) / 2.0,
        close=(candle.bid_c + candle.ask_c) / 2.0,
    )


def _ema_series(values: Sequence[float], period: int) -> list[float | None]:
    """Return a standard, SMA-seeded EMA in one linear pass."""

    result: list[float | None] = [None] * len(values)
    if period <= 0 or len(values) < period:
        return result
    value = statistics.fmean(float(item) for item in values[:period])
    result[period - 1] = value
    alpha = 2.0 / (period + 1.0)
    for index, item in enumerate(values[period:], start=period):
        value += alpha * (float(item) - value)
        result[index] = value
    return result


def _ema(values: Sequence[float], period: int) -> float | None:
    series = _ema_series(values, period)
    return series[-1] if series else None


def _rsi(closes: Sequence[float], period: int = RSI_PERIOD) -> float | None:
    if len(closes) <= period:
        return None
    changes = [float(b) - float(a) for a, b in zip(closes, closes[1:])]
    average_gain = statistics.fmean(max(0.0, item) for item in changes[:period])
    average_loss = statistics.fmean(max(0.0, -item) for item in changes[:period])
    for item in changes[period:]:
        average_gain = (average_gain * (period - 1) + max(0.0, item)) / period
        average_loss = (average_loss * (period - 1) + max(0.0, -item)) / period
    if average_gain == 0.0 and average_loss == 0.0:
        return 50.0
    if average_loss == 0.0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + average_gain / average_loss)


def _atr_di_adx(
    bars: Sequence[_Bar],
) -> tuple[float | None, float | None, float | None, float | None]:
    if len(bars) <= ADX_PERIOD * 2:
        return None, None, None, None
    trs: list[float] = []
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    for previous, current in zip(bars, bars[1:]):
        trs.append(
            max(
                current.high - current.low,
                abs(current.high - previous.close),
                abs(current.low - previous.close),
            )
        )
        up = current.high - previous.high
        down = previous.low - current.low
        plus_dm.append(up if up > down and up > 0.0 else 0.0)
        minus_dm.append(down if down > up and down > 0.0 else 0.0)
    tr_smooth = sum(trs[:ADX_PERIOD])
    plus_smooth = sum(plus_dm[:ADX_PERIOD])
    minus_smooth = sum(minus_dm[:ADX_PERIOD])
    dx_values: list[float] = []
    plus_di = minus_di = 0.0
    for index in range(ADX_PERIOD, len(trs)):
        tr_smooth = tr_smooth - tr_smooth / ADX_PERIOD + trs[index]
        plus_smooth = plus_smooth - plus_smooth / ADX_PERIOD + plus_dm[index]
        minus_smooth = minus_smooth - minus_smooth / ADX_PERIOD + minus_dm[index]
        plus_di = 100.0 * plus_smooth / tr_smooth if tr_smooth else 0.0
        minus_di = 100.0 * minus_smooth / tr_smooth if tr_smooth else 0.0
        denominator = plus_di + minus_di
        dx_values.append(
            100.0 * abs(plus_di - minus_di) / denominator if denominator else 0.0
        )
    if len(dx_values) < ADX_PERIOD:
        return None, None, None, None
    adx = statistics.fmean(dx_values[:ADX_PERIOD])
    for value in dx_values[ADX_PERIOD:]:
        adx = (adx * (ADX_PERIOD - 1) + value) / ADX_PERIOD
    atr = tr_smooth / ADX_PERIOD
    return atr, plus_di, minus_di, adx


def _feature(bars: Sequence[_Bar], prior: _Feature | None) -> _Feature:
    current = bars[-1]
    closes = [item.close for item in bars]
    atr, plus_di, minus_di, adx = _atr_di_adx(bars)
    ema_fast = _ema(closes, EMA_FAST_PERIOD)
    ema_slow = _ema(closes, EMA_SLOW_PERIOD)
    ema_trend = _ema(closes, EMA_TREND_PERIOD)
    fast_series = _ema_series(closes, EMA_FAST_PERIOD)
    slow_series = _ema_series(closes, EMA_SLOW_PERIOD)
    macd_values = [
        float(fast) - float(slow)
        for fast, slow in zip(fast_series, slow_series)
        if fast is not None and slow is not None
    ]
    macd = macd_values[-1] if macd_values else None
    macd_signal = _ema(macd_values, 9)
    lookback = bars[-(BREAKOUT_LOOKBACK + 1) : -1]
    return _Feature(
        timeframe=current.timeframe,
        completed_at=current.end,
        close=current.close,
        atr=atr,
        adx=adx,
        plus_di=plus_di,
        minus_di=minus_di,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        ema_trend=ema_trend,
        rsi=_rsi(closes),
        macd_hist=(macd - macd_signal)
        if macd is not None and macd_signal is not None
        else None,
        previous_rsi=prior.rsi if prior else None,
        previous_macd_hist=prior.macd_hist if prior else None,
        previous_atr=prior.atr if prior else None,
        previous_adx=prior.adx if prior else None,
        prior_high=max((item.high for item in lookback), default=None)
        if len(lookback) >= BREAKOUT_LOOKBACK
        else None,
        prior_low=min((item.low for item in lookback), default=None)
        if len(lookback) >= BREAKOUT_LOOKBACK
        else None,
        current_high=current.high,
        current_low=current.low,
        current_open=current.open,
    )


def _trend_side(feature: _Feature | None) -> str | None:
    if (
        feature is None
        or feature.plus_di is None
        or feature.minus_di is None
        or feature.ema_fast is None
        or feature.ema_trend is None
    ):
        return None
    if feature.plus_di > feature.minus_di and feature.ema_fast > feature.ema_trend:
        return "LONG"
    if feature.minus_di > feature.plus_di and feature.ema_fast < feature.ema_trend:
        return "SHORT"
    return None


def _momentum_allows(feature: _Feature | None, side: str) -> bool:
    if feature is None:
        return False
    rsi_ok = feature.rsi is not None and (
        feature.rsi >= 50.0 if side == "LONG" else feature.rsi <= 50.0
    )
    macd_ok = feature.macd_hist is not None and (
        feature.macd_hist >= 0.0 if side == "LONG" else feature.macd_hist <= 0.0
    )
    return rsi_ok and macd_ok


def _crossed_zero(current: float | None, previous: float | None, side: str) -> bool:
    if current is None or previous is None:
        return False
    return previous <= 0.0 < current if side == "LONG" else previous >= 0.0 > current


def _crossed_fifty(current: float | None, previous: float | None, side: str) -> bool:
    if current is None or previous is None:
        return False
    return previous <= 50.0 < current if side == "LONG" else previous >= 50.0 > current


def _signal_for(
    spec: TechnicalSpec,
    features: Mapping[str, _Feature],
    *,
    activation: datetime,
) -> str | None:
    """Evaluate one predeclared rule from immutable completed features only."""

    m1 = features.get("M1")
    m5 = features.get("M5")
    m15 = features.get("M15")
    h1 = features.get("H1")
    if m1 is None:
        return None
    direction = _trend_side(m5)
    regime = h1 or m15 or m5
    trend_regime = bool(
        regime is not None
        and regime.atr is not None
        and regime.adx is not None
        and regime.adx >= 18.0
    )
    range_regime = bool(
        (m15 or m5) is not None
        and (m15 or m5).atr is not None
        and (m15 or m5).adx is not None
        and (m15 or m5).adx < 25.0
    )

    if spec.hypothesis_id == "H01":
        return (
            direction
            if trend_regime and direction and _momentum_allows(m1, direction)
            else None
        )
    if spec.hypothesis_id == "H02":
        if not trend_regime or direction is None:
            return None
        reclaimed = _crossed_fifty(m1.rsi, m1.previous_rsi, direction)
        return direction if reclaimed else None
    if spec.hypothesis_id == "H03":
        if not range_regime or m1.prior_high is None or m1.prior_low is None:
            return None
        width = m1.prior_high - m1.prior_low
        if width <= 0.0:
            return None
        location = (m1.close - m1.prior_low) / width
        long_reentry = bool(
            m1.previous_rsi is not None
            and m1.rsi is not None
            and m1.previous_rsi < 35.0 <= m1.rsi
        )
        short_reentry = bool(
            m1.previous_rsi is not None
            and m1.rsi is not None
            and m1.previous_rsi > 65.0 >= m1.rsi
        )
        if location <= 0.25 and long_reentry:
            return "LONG"
        if location >= 0.75 and short_reentry:
            return "SHORT"
        return None
    if spec.hypothesis_id == "H04":
        if m1.prior_high is None or m1.prior_low is None or direction is None:
            return None
        compressed = bool(
            m15 is not None
            and m15.atr is not None
            and m15.previous_atr is not None
            and m15.atr <= m15.previous_atr
        )
        broke = (
            m1.close > m1.prior_high if direction == "LONG" else m1.close < m1.prior_low
        )
        return direction if compressed and broke else None
    if spec.hypothesis_id == "H05":
        if not range_regime or m1.prior_high is None or m1.prior_low is None:
            return None
        if m1.current_high > m1.prior_high and m1.close < m1.prior_high:
            return "SHORT"
        if m1.current_low < m1.prior_low and m1.close > m1.prior_low:
            return "LONG"
        return None
    if spec.hypothesis_id == "H06":
        old_side = _trend_side(m15 or h1)
        if old_side is None or regime is None:
            return None
        decelerating = bool(
            (
                regime.adx is not None
                and regime.previous_adx is not None
                and regime.adx < regime.previous_adx
            )
            or (
                regime.atr is not None
                and regime.previous_atr is not None
                and regime.atr < regime.previous_atr
            )
        )
        reverse = "SHORT" if old_side == "LONG" else "LONG"
        reversal_trigger = _crossed_fifty(
            m1.rsi, m1.previous_rsi, reverse
        ) and _crossed_zero(m1.macd_hist, m1.previous_macd_hist, reverse)
        return reverse if decelerating and reversal_trigger else None
    if spec.hypothesis_id == "H07":
        # The fixed hours are UTC session landmarks for diagnostic segmentation;
        # a later version should replace them with a timezone/session calendar.
        session_open = activation.minute < 15 and activation.hour in {7, 8, 12, 13}
        if (
            not session_open
            or direction is None
            or m1.prior_high is None
            or m1.prior_low is None
        ):
            return None
        broke = (
            m1.close > m1.prior_high if direction == "LONG" else m1.close < m1.prior_low
        )
        expanding = bool(m5 and m5.atr and m5.previous_atr and m5.atr > m5.previous_atr)
        macd_confirms = bool(
            m1.macd_hist is not None
            and (m1.macd_hist > 0.0 if direction == "LONG" else m1.macd_hist < 0.0)
        )
        return direction if broke and expanding and macd_confirms else None
    return None


def _opposite(side: str) -> str:
    return "SHORT" if side == "LONG" else "LONG"


def _open_prices(candle: S5BidAskCandle, side: str) -> tuple[float, float]:
    mid = (candle.bid_o + candle.ask_o) / 2.0
    return (candle.ask_o, mid) if side == "LONG" else (candle.bid_o, mid)


def _exit_open(candle: S5BidAskCandle, side: str) -> tuple[float, float]:
    mid = (candle.bid_o + candle.ask_o) / 2.0
    return (candle.bid_o, mid) if side == "LONG" else (candle.ask_o, mid)


def _finish_position(
    position: _Position,
    candle: S5BidAskCandle,
    *,
    reason: str,
    exit_exec: float,
    exit_mid: float,
    pip_factor: float,
) -> dict[str, Any]:
    assert position.entry_at is not None
    assert position.entry_exec is not None
    assert position.entry_mid is not None
    direction = 1.0 if position.side == "LONG" else -1.0
    gross_mid = direction * (exit_mid - position.entry_mid) * pip_factor
    exact_net = direction * (exit_exec - position.entry_exec) * pip_factor
    spread_drag = gross_mid - exact_net
    assert position.stop_loss is not None
    initial_risk_pips = abs(position.entry_exec - position.stop_loss) * pip_factor
    if initial_risk_pips <= 0.0:
        raise ValueError("initial risk must be positive")
    gross_mid_r = gross_mid / initial_risk_pips
    exact_net_r = exact_net / initial_risk_pips
    spread_drag_r = spread_drag / initial_risk_pips
    if abs(gross_mid) < 1e-12:
        gross_mid = 0.0
    if abs(exact_net) < 1e-12:
        exact_net = 0.0
    if abs(spread_drag) < 1e-12:
        spread_drag = 0.0
    return {
        "candidate_id": position.candidate_id,
        "hypothesis_id": position.hypothesis_id,
        "orientation": position.orientation,
        "arm_id": position.arm.arm_id,
        "side": position.side,
        "split": position.split_name,
        "activation_at_utc": position.activation.isoformat(),
        "entry_at_utc": position.entry_at.isoformat(),
        "exit_at_utc": candle.timestamp_utc.isoformat(),
        "reason": reason,
        "gross_mid_pips": gross_mid,
        "gross_mid_is_proxy": reason not in {"HOLD_EXIT", "SL_GAP_AT_TIME_CLOSE"}
        and not reason.endswith("_GAP"),
        "spread_drag_pips": spread_drag,
        "exact_net_pips": exact_net,
        "initial_risk_pips": initial_risk_pips,
        "gross_mid_r": gross_mid_r,
        "spread_drag_r": spread_drag_r,
        "exact_net_r": exact_net_r,
    }


def _resolve_on_candle(
    position: _Position,
    candle: S5BidAskCandle,
    *,
    pip_factor: float,
) -> dict[str, Any] | None:
    if not position.filled:
        return None
    assert position.hold_at is not None
    assert position.take_profit is not None
    assert position.stop_loss is not None
    if candle.timestamp_utc >= position.hold_at:
        executable, mid = _exit_open(candle, position.side)
        stop_gap = (
            executable <= position.stop_loss
            if position.side == "LONG"
            else executable >= position.stop_loss
        )
        return _finish_position(
            position,
            candle,
            reason="SL_GAP_AT_TIME_CLOSE" if stop_gap else "HOLD_EXIT",
            exit_exec=executable,
            exit_mid=mid,
            pip_factor=pip_factor,
        )
    tp = position.take_profit
    sl = position.stop_loss
    if position.side == "LONG":
        if candle.bid_o <= sl:
            executable = candle.bid_o
            return _finish_position(
                position,
                candle,
                reason="STOP_LOSS_GAP",
                exit_exec=executable,
                exit_mid=(candle.bid_o + candle.ask_o) / 2.0,
                pip_factor=pip_factor,
            )
        if candle.bid_o >= tp:
            executable = candle.bid_o
            return _finish_position(
                position,
                candle,
                reason="TAKE_PROFIT_GAP",
                exit_exec=executable,
                exit_mid=(candle.bid_o + candle.ask_o) / 2.0,
                pip_factor=pip_factor,
            )
        stop_hit = candle.bid_l <= sl
        take_hit = candle.bid_h >= tp
        if stop_hit:
            executable = sl
            reason = "STOP_LOSS"
            if take_hit:
                reason = "STOP_LOSS_SAME_S5"
            return _finish_position(
                position,
                candle,
                reason=reason,
                exit_exec=executable,
                exit_mid=sl,
                pip_factor=pip_factor,
            )
        if take_hit:
            return _finish_position(
                position,
                candle,
                reason="TAKE_PROFIT",
                exit_exec=tp,
                exit_mid=tp,
                pip_factor=pip_factor,
            )
    else:
        if candle.ask_o >= sl:
            executable = candle.ask_o
            return _finish_position(
                position,
                candle,
                reason="STOP_LOSS_GAP",
                exit_exec=executable,
                exit_mid=(candle.bid_o + candle.ask_o) / 2.0,
                pip_factor=pip_factor,
            )
        if candle.ask_o <= tp:
            executable = candle.ask_o
            return _finish_position(
                position,
                candle,
                reason="TAKE_PROFIT_GAP",
                exit_exec=executable,
                exit_mid=(candle.bid_o + candle.ask_o) / 2.0,
                pip_factor=pip_factor,
            )
        stop_hit = candle.ask_h >= sl
        take_hit = candle.ask_l <= tp
        if stop_hit:
            executable = sl
            reason = "STOP_LOSS"
            if take_hit:
                reason = "STOP_LOSS_SAME_S5"
            return _finish_position(
                position,
                candle,
                reason=reason,
                exit_exec=executable,
                exit_mid=sl,
                pip_factor=pip_factor,
            )
        if take_hit:
            return _finish_position(
                position,
                candle,
                reason="TAKE_PROFIT",
                exit_exec=tp,
                exit_mid=tp,
                pip_factor=pip_factor,
            )
    return None


def _record_trade(stat: dict[str, Any], trade: Mapping[str, Any]) -> None:
    gross = float(trade["gross_mid_pips"])
    drag = float(trade["spread_drag_pips"])
    net = float(trade["exact_net_pips"])
    gross_r = float(trade["gross_mid_r"])
    drag_r = float(trade["spread_drag_r"])
    net_r = float(trade["exact_net_r"])
    stat["resolved_count"] += 1
    stat["gross_mid_pips"] += gross
    stat["spread_drag_pips"] += drag
    stat["exact_net_pips"] += net
    stat["gross_mid_r"] += gross_r
    stat["spread_drag_r"] += drag_r
    stat["exact_net_r"] += net_r
    if net > 0.0:
        stat["win_count"] += 1
        stat["gross_profit_pips"] += net
        stat["gross_profit_r"] += net_r
    elif net < 0.0:
        stat["loss_count"] += 1
        stat["gross_loss_pips"] += -net
        stat["gross_loss_r"] += -net_r
    else:
        stat["flat_count"] += 1
    stat["equity_pips"] += net
    stat["peak_equity_pips"] = max(stat["peak_equity_pips"], stat["equity_pips"])
    stat["max_drawdown_pips"] = max(
        stat["max_drawdown_pips"],
        stat["peak_equity_pips"] - stat["equity_pips"],
    )
    stat["reason_counts"][str(trade["reason"])] += 1


def _metric_from_stat(stat: Mapping[str, Any]) -> dict[str, Any]:
    resolved = int(stat["resolved_count"])
    profit = float(stat["gross_profit_pips"])
    loss = float(stat["gross_loss_pips"])
    profit_r = float(stat["gross_profit_r"])
    loss_r = float(stat["gross_loss_r"])
    pf: float | None
    if loss > 0.0:
        pf = profit / loss
    else:
        pf = None
    return {
        "raw_signal_count": int(stat["raw_signal_count"]),
        "signal_count": int(stat["signal_count"]),
        "deoverlap_count": int(stat["deoverlap_count"]),
        "embargoed_signal_count": int(stat["embargoed_signal_count"]),
        "expired_unfilled_count": int(stat["expired_unfilled_count"]),
        "filled_count": int(stat["filled_count"]),
        "resolved_count": resolved,
        "purged_count": int(stat["purged_count"]),
        "unresolved_count": int(stat["unresolved_count"]),
        "win_count": int(stat["win_count"]),
        "loss_count": int(stat["loss_count"]),
        "flat_count": int(stat["flat_count"]),
        "gross_mid_pips": float(stat["gross_mid_pips"]),
        "spread_drag_pips": float(stat["spread_drag_pips"]),
        "exact_net_pips": float(stat["exact_net_pips"]),
        "average_net_pips": float(stat["exact_net_pips"]) / resolved
        if resolved
        else None,
        "gross_mid_r": float(stat["gross_mid_r"]),
        "spread_drag_r": float(stat["spread_drag_r"]),
        "exact_net_r": float(stat["exact_net_r"]),
        "average_net_r": float(stat["exact_net_r"]) / resolved if resolved else None,
        "profit_factor_r": profit_r / loss_r if loss_r > 0.0 else None,
        "profit_factor": pf,
        "win_rate": int(stat["win_count"]) / resolved if resolved else None,
        "max_drawdown_pips": float(stat["max_drawdown_pips"]),
        "gross_profit_pips": profit,
        "gross_loss_pips": loss,
        "gross_profit_r": profit_r,
        "gross_loss_r": loss_r,
        "reason_counts": dict(sorted(stat["reason_counts"].items())),
    }


def run_causal_multitf_s5_grid(
    pair: str,
    candles: Iterable[S5BidAskCandle],
    splits: Sequence[UtcSplit],
    unavailable_pairs: Sequence[str] = (),
) -> dict[str, Any]:
    """Run one pair through the causal grid while consuming its S5 stream once."""

    pair_name = str(pair).strip().upper()
    if not pair_name:
        raise ValueError("pair is required")
    normalized_splits = _normalise_splits(splits)
    split_by_name = {split.name: split for split in normalized_splits}
    unavailable = tuple(str(item).strip().upper() for item in unavailable_pairs)
    if len(set(unavailable)) != len(unavailable):
        raise ValueError("unavailable_pairs must be unique")
    pip_factor = float(instrument_pip_factor(pair_name))
    catalog = build_predeclared_catalog_v1()
    arms = build_predeclared_arms_v1()
    max_candidate_horizon_seconds = max(
        arm.ttl_seconds + arm.hold_seconds for arm in arms
    )
    candidates = _candidate_rows()
    stats: dict[tuple[str, str], dict[str, Any]] = {
        (candidate_id, split.name): _blank_stat()
        for _spec, _orientation, _arm, candidate_id in candidates
        for split in normalized_splits
    }
    daily: dict[tuple[str, str, str], dict[str, Any]] = {}
    if pair_name in unavailable:
        return _build_pair_result(
            pair_name,
            normalized_splits,
            candidates,
            stats,
            daily,
            status="UNAVAILABLE",
            trade_rows=[],
            signal_rows=[],
            aggregation={
                "source_candle_count": 0,
                "observed_missing_s5_slots": 0,
                "synthetic_s5_count": 0,
                "completed_bucket_counts": {key: 0 for key in TIMEFRAMES_SECONDS},
                "completed_bucket_clocks": {key: [] for key in TIMEFRAMES_SECONDS},
                "partial_bucket_counts": {key: 0 for key in TIMEFRAMES_SECONDS},
                "observed_utc_days_by_split": {
                    split.name: [] for split in normalized_splits
                },
            },
        )

    if isinstance(candles, (str, bytes)):
        raise ValueError("candles must be an iterable of S5BidAskCandle")
    try:
        iterator = iter(candles)
    except TypeError as error:
        raise ValueError("candles must be iterable") from error

    bars: dict[str, deque[_Bar]] = {
        timeframe: deque(maxlen=EMA_TREND_PERIOD + BREAKOUT_LOOKBACK + 4)
        for timeframe in TIMEFRAMES_SECONDS
    }
    features: dict[str, _Feature] = {}
    buckets: dict[str, _Bucket] = {}
    completed_counts = {key: 0 for key in TIMEFRAMES_SECONDS}
    completed_clocks: dict[str, list[str]] = {key: [] for key in TIMEFRAMES_SECONDS}
    active: dict[str, _Position] = {}
    trade_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    trade_row_total = 0
    previous_stamp: datetime | None = None
    first_stamp: datetime | None = None
    last_stamp: datetime | None = None
    source_count = 0
    missing_slots = 0
    observed_days: dict[str, set[str]] = {
        split.name: set() for split in normalized_splits
    }

    for candle in iterator:
        prior_stamp = previous_stamp
        stamp = _validate_candle(candle, previous_stamp)
        source_count += 1
        if first_stamp is None:
            first_stamp = stamp
        if previous_stamp is not None:
            missing_slots += max(
                0, int((stamp - previous_stamp).total_seconds() // 5) - 1
            )
        previous_stamp = stamp
        last_stamp = stamp
        observed_split = _split_for(stamp, normalized_splits)
        if observed_split is not None:
            observed_days[observed_split.name].add(stamp.date().isoformat())

        # A position from an earlier partition must not suppress the first
        # signal of the next partition. Purge it before indicators/signals at
        # the crossed boundary are processed.
        crossed_split_names = {
            split.name
            for split in normalized_splits
            if stamp >= split.to_utc
            and (prior_stamp is None or prior_stamp < split.to_utc)
        }
        if crossed_split_names:
            purge_ids: list[str] = []
            for candidate_id, position in active.items():
                if position.split_name not in crossed_split_names:
                    continue
                stat = stats[(candidate_id, position.split_name)]
                stat["purged_count"] += 1
                stat["unresolved_count"] += int(position.filled)
                stat["reason_counts"]["SPLIT_END_PURGE"] += 1
                purge_ids.append(candidate_id)
            for candidate_id in purge_ids:
                active.pop(candidate_id, None)

        completed_events: list[_Bar] = []
        for timeframe in TIMEFRAMES_SECONDS:
            bucket = buckets.get(timeframe)
            if bucket is not None and bucket.end <= stamp:
                completed_events.append(bucket.finish())
                del buckets[timeframe]

        # A no-tick gap may close multiple timeframe buckets at the same
        # observed S5. Process those close events in clock order and freeze the
        # exact feature view at each M1 decision clock.
        completed_m1: list[tuple[_Bar, dict[str, _Feature]]] = []
        completed_events.sort(key=lambda bar: (bar.end, bar.timeframe))
        event_index = 0
        while event_index < len(completed_events):
            event_end = completed_events[event_index].end
            same_clock: list[_Bar] = []
            while (
                event_index < len(completed_events)
                and completed_events[event_index].end == event_end
            ):
                same_clock.append(completed_events[event_index])
                event_index += 1
            for bar in same_clock:
                timeframe = bar.timeframe
                bars[timeframe].append(bar)
                features[timeframe] = _feature(
                    tuple(bars[timeframe]), features.get(timeframe)
                )
                completed_counts[timeframe] += 1
                if len(completed_clocks[timeframe]) < 512:
                    completed_clocks[timeframe].append(bar.end.isoformat())
            for bar in same_clock:
                if bar.timeframe == "M1":
                    completed_m1.append((bar, dict(features)))

        # Emit candidate signals only after every timeframe ending at this same
        # boundary has been finalized.  The current S5 candle is still unseen by
        # every indicator and therefore cannot leak into the frozen decision.
        for m1_bar, frozen_features in completed_m1:
            split = _split_for(m1_bar.end, normalized_splits)
            if split is None:
                continue
            frozen_atr = (
                frozen_features.get("M5").atr
                if frozen_features.get("M5") is not None
                and frozen_features["M5"].atr is not None
                else frozen_features["M1"].atr
            )
            if frozen_atr is None or frozen_atr <= 0.0:
                continue
            for spec in catalog:
                if spec.no_trade_control:
                    continue
                raw_side = _signal_for(spec, frozen_features, activation=m1_bar.end)
                if raw_side is None:
                    continue
                embargoed = (
                    m1_bar.end + timedelta(seconds=max_candidate_horizon_seconds)
                    >= split.to_utc
                )
                if len(signal_rows) < 512:
                    signal_rows.append(
                        {
                            "hypothesis_id": spec.hypothesis_id,
                            "activation_at_utc": m1_bar.end.isoformat(),
                            "direct_side": raw_side,
                            "atr_price": frozen_atr,
                            "split": split.name,
                            "split_end_embargoed": embargoed,
                            "feature_completed_at_utc": {
                                key: feature.completed_at.isoformat()
                                for key, feature in sorted(frozen_features.items())
                            },
                        }
                    )
                for orientation in ORIENTATIONS:
                    side = raw_side if orientation == "DIRECT" else _opposite(raw_side)
                    for arm in arms:
                        candidate_id = (
                            f"{spec.hypothesis_id}:{orientation}:{arm.arm_id}"
                        )
                        stat = stats[(candidate_id, split.name)]
                        stat["raw_signal_count"] += 1
                        if embargoed:
                            stat["embargoed_signal_count"] += 1
                            stat["reason_counts"]["SPLIT_END_EMBARGO"] += 1
                            continue
                        if candidate_id in active:
                            stat["deoverlap_count"] += 1
                            stat["reason_counts"]["DEOVERLAP_ACTIVE_POSITION"] += 1
                            continue
                        stat["signal_count"] += 1
                        active[candidate_id] = _Position(
                            candidate_id=candidate_id,
                            hypothesis_id=spec.hypothesis_id,
                            orientation=orientation,
                            arm=arm,
                            side=side,
                            split_name=split.name,
                            activation=m1_bar.end,
                            expiry=m1_bar.end + timedelta(seconds=arm.ttl_seconds),
                            atr=frozen_atr,
                            signal_mid=m1_bar.close,
                        )

        resolved_ids: list[str] = []
        for candidate_id, position in active.items():
            split = split_by_name[position.split_name]
            stat = stats[(candidate_id, position.split_name)]
            if stamp >= split.to_utc:
                stat["purged_count"] += 1
                stat["unresolved_count"] += int(position.filled)
                stat["reason_counts"]["SPLIT_END_PURGE"] += 1
                resolved_ids.append(candidate_id)
                continue
            if not position.filled:
                if stamp > position.expiry:
                    stat["expired_unfilled_count"] += 1
                    stat["reason_counts"]["ENTRY_TTL_EXPIRED_NO_REAL_S5"] += 1
                    resolved_ids.append(candidate_id)
                    continue
                if stamp < position.activation:
                    continue
                executable, mid = _open_prices(candle, position.side)
                position.filled = True
                position.entry_at = stamp
                position.entry_exec = executable
                position.entry_mid = mid
                position.hold_at = stamp + timedelta(seconds=position.arm.hold_seconds)
                if position.side == "LONG":
                    position.take_profit = (
                        mid + position.arm.tp_atr_multiple * position.atr
                    )
                    position.stop_loss = (
                        mid - position.arm.sl_atr_multiple * position.atr
                    )
                else:
                    position.take_profit = (
                        mid - position.arm.tp_atr_multiple * position.atr
                    )
                    position.stop_loss = (
                        mid + position.arm.sl_atr_multiple * position.atr
                    )
                stat["filled_count"] += 1
            trade = _resolve_on_candle(position, candle, pip_factor=pip_factor)
            if trade is None:
                continue
            _record_trade(stat, trade)
            day = (
                position.entry_at.date().isoformat()
                if position.entry_at
                else stamp.date().isoformat()
            )
            day_stat = daily.setdefault(
                (candidate_id, position.split_name, day), _blank_stat()
            )
            day_stat["filled_count"] += 1
            _record_trade(day_stat, trade)
            trade_row_total += 1
            if len(trade_rows) < 512:
                trade_rows.append(trade)
            resolved_ids.append(candidate_id)

        for candidate_id in resolved_ids:
            active.pop(candidate_id, None)

        for timeframe in TIMEFRAMES_SECONDS:
            bucket = buckets.get(timeframe)
            expected_start = _bucket_start(stamp, TIMEFRAMES_SECONDS[timeframe])
            if bucket is None:
                buckets[timeframe] = _new_bucket(timeframe, candle)
            elif bucket.start == expected_start:
                bucket.add(candle)
            else:
                # Any earlier non-empty bucket was finalized above.  Empty UTC
                # buckets crossed by a genuine no-tick gap are never synthesized.
                buckets[timeframe] = _new_bucket(timeframe, candle)

    for candidate_id, position in tuple(active.items()):
        stat = stats[(candidate_id, position.split_name)]
        stat["purged_count"] += 1
        stat["unresolved_count"] += int(position.filled)
        stat["reason_counts"]["SOURCE_END_BEFORE_SPLIT_RESOLUTION"] += 1

    aggregation = {
        "source_candle_count": source_count,
        "first_s5_at_utc": first_stamp.isoformat() if first_stamp else None,
        "last_s5_at_utc": last_stamp.isoformat() if last_stamp else None,
        "observed_missing_s5_slots": missing_slots,
        "synthetic_s5_count": 0,
        "completed_bucket_counts": completed_counts,
        "completed_bucket_clocks": completed_clocks,
        "partial_bucket_counts": {
            timeframe: int(timeframe in buckets) for timeframe in TIMEFRAMES_SECONDS
        },
        "observed_utc_days_by_split": {
            split.name: sorted(observed_days[split.name]) for split in normalized_splits
        },
        "trade_row_count": trade_row_total,
        "trade_rows_returned": len(trade_rows),
        "trade_rows_omitted": max(0, trade_row_total - len(trade_rows)),
    }
    return _build_pair_result(
        pair_name,
        normalized_splits,
        candidates,
        stats,
        daily,
        status="OK",
        trade_rows=trade_rows,
        signal_rows=signal_rows,
        aggregation=aggregation,
    )


def _daily_rows(
    pair: str,
    daily: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (candidate_id, split_name, day), stat in sorted(daily.items()):
        metric = _metric_from_stat(stat)
        rows.append(
            {
                "pair": pair,
                "candidate_id": candidate_id,
                "split": split_name,
                "utc_day": day,
                "resolved_count": metric["resolved_count"],
                "win_count": metric["win_count"],
                "loss_count": metric["loss_count"],
                "flat_count": metric["flat_count"],
                "gross_mid_pips": metric["gross_mid_pips"],
                "spread_drag_pips": metric["spread_drag_pips"],
                "exact_net_pips": metric["exact_net_pips"],
                "gross_mid_r": metric["gross_mid_r"],
                "spread_drag_r": metric["spread_drag_r"],
                "exact_net_r": metric["exact_net_r"],
                "gross_profit_pips": metric["gross_profit_pips"],
                "gross_loss_pips": metric["gross_loss_pips"],
                "gross_profit_r": metric["gross_profit_r"],
                "gross_loss_r": metric["gross_loss_r"],
                "max_drawdown_input_pips": max(0.0, -metric["exact_net_pips"]),
                "reason_counts": metric["reason_counts"],
            }
        )
    return rows


def _build_pair_result(
    pair: str,
    splits: Sequence[UtcSplit],
    candidates: Sequence[tuple[TechnicalSpec, str, ArmSpec, str]],
    stats: Mapping[tuple[str, str], Mapping[str, Any]],
    daily: Mapping[tuple[str, str, str], Mapping[str, Any]],
    *,
    status: str,
    trade_rows: Sequence[Mapping[str, Any]],
    signal_rows: Sequence[Mapping[str, Any]],
    aggregation: Mapping[str, Any],
) -> dict[str, Any]:
    daily_rows = _daily_rows(pair, daily)
    day_count: dict[tuple[str, str], int] = defaultdict(int)
    for row in daily_rows:
        day_count[(str(row["candidate_id"]), str(row["split"]))] += 1
    candidate_metrics: list[dict[str, Any]] = []
    for spec, orientation, arm, candidate_id in candidates:
        metrics_by_split: dict[str, Any] = {}
        for split in splits:
            metric = _metric_from_stat(stats[(candidate_id, split.name)])
            metric["active_day_count"] = day_count[(candidate_id, split.name)]
            metrics_by_split[split.name] = metric
        candidate_metrics.append(
            {
                "candidate_id": candidate_id,
                "hypothesis_id": spec.hypothesis_id,
                "family": spec.family,
                "orientation": orientation,
                "arm_id": arm.arm_id,
                "complexity": spec.complexity
                + arm.complexity
                + int(orientation == "INVERSE"),
                "simplicity_key": list(_simplicity_key(orientation, arm)),
                "metrics_by_split": metrics_by_split,
            }
        )
    validation_name = _named_split(splits, "VALIDATION")
    holdout_name = _named_split(splits, "HOLDOUT")
    observed_days_by_split = aggregation.get("observed_utc_days_by_split") or {}
    if not isinstance(observed_days_by_split, Mapping):
        raise ValueError("aggregation observed UTC days are invalid")
    selection = _select_validation(
        candidate_metrics,
        daily_rows,
        validation_name,
        observed_days_by_split,
    )
    winners = list(selection["winner_arm_ids"])
    holdout_evaluated = winners if holdout_name is not None else []
    total_reasons: dict[str, int] = defaultdict(int)
    raw_count = signal_count = deoverlap = 0
    for row in candidate_metrics:
        for metric in row["metrics_by_split"].values():
            raw_count += int(metric["raw_signal_count"])
            signal_count += int(metric["signal_count"])
            deoverlap += int(metric["deoverlap_count"])
            for reason, count in metric["reason_counts"].items():
                total_reasons[str(reason)] += int(count)
    return {
        "contract": GRID_CONTRACT,
        "schema_version": 1,
        "status": status,
        "pair": pair,
        "catalog": [dict(item.__dict__) for item in build_predeclared_catalog_v1()],
        "arms": [dict(item.__dict__) for item in build_predeclared_arms_v1()],
        "candidate_count": len(candidate_metrics),
        "candidate_metrics": candidate_metrics,
        "daily_aggregates": daily_rows,
        "trade_rows": [dict(item) for item in trade_rows],
        "signal_rows": [dict(item) for item in signal_rows],
        "aggregation": dict(aggregation),
        "raw_signal_count": raw_count,
        "accepted_signal_count": signal_count,
        "deoverlap_count": deoverlap,
        "reason_counts": dict(sorted(total_reasons.items())),
        "validation": {
            "split": validation_name,
            "selected_arm_ids": winners,
            "validation_winner_arm_ids": winners,
            "multiple_testing": selection,
            "selection_uses_holdout": False,
        },
        "holdout": {
            "split": holdout_name,
            "evaluated_arm_ids": holdout_evaluated,
            "reselection_performed": False,
            "selection_unchanged": True,
        },
        **_AUTHORITY,
    }


def _named_split(splits: Sequence[UtcSplit], wanted: str) -> str | None:
    for split in splits:
        if str(split.name).upper() == wanted:
            return split.name
    return None


def _daily_samples(
    daily_rows: Sequence[Mapping[str, Any]],
    candidate_id: str,
    split_name: str | None,
    observed_days: Sequence[str],
    *,
    value_key: str,
) -> list[float]:
    if split_name is None or not observed_days:
        return []
    totals: dict[str, float] = defaultdict(float)
    for row in daily_rows:
        if row.get("candidate_id") != candidate_id or row.get("split") != split_name:
            continue
        totals[str(row["utc_day"])] += float(row.get(value_key, 0.0))
    return [totals.get(day, 0.0) for day in observed_days]


def _one_sided_p_and_se(
    samples: Sequence[float], *, seed_key: str
) -> tuple[float, float, str]:
    """One-sided UTC-day sign-flip test under a symmetric zero-return null."""

    if len(samples) < MIN_VALIDATION_CLUSTER_DAYS:
        return 1.0, 0.0, "INSUFFICIENT_UTC_DAY_CLUSTERS"
    mean = statistics.fmean(samples)
    se = statistics.stdev(samples) / math.sqrt(len(samples))
    if mean <= 0.0:
        return 1.0, se, "NON_POSITIVE_DAILY_MEAN"
    values = [float(item) for item in samples]
    observed_sum = sum(values)
    size = len(values)
    tolerance = max(1e-15, abs(observed_sum) * 1e-12)
    if size <= SIGN_FLIP_EXACT_MAX_DAYS:
        total = 1 << size
        extreme = 0
        for mask in range(total):
            signed_sum = sum(
                value if mask & (1 << index) else -value
                for index, value in enumerate(values)
            )
            if signed_sum >= observed_sum - tolerance:
                extreme += 1
        return extreme / total, se, "UTC_DAY_EXACT_SIGN_FLIP_V1"
    seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:16], 16)
    generator = random.Random(seed)
    extreme = 0
    for _ in range(SIGN_FLIP_MONTE_CARLO_REPETITIONS):
        signed_sum = sum(
            value if generator.getrandbits(1) else -value for value in values
        )
        if signed_sum >= observed_sum - tolerance:
            extreme += 1
    p_value = (extreme + 1.0) / (SIGN_FLIP_MONTE_CARLO_REPETITIONS + 1.0)
    return p_value, se, "UTC_DAY_MONTE_CARLO_SIGN_FLIP_V1"


def _select_validation(
    candidate_metrics: Sequence[Mapping[str, Any]],
    daily_rows: Sequence[Mapping[str, Any]],
    validation_name: str | None,
    observed_days_by_split: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    tests: list[dict[str, Any]] = []
    for candidate in candidate_metrics:
        candidate_id = str(candidate["candidate_id"])
        split_metrics = candidate.get("metrics_by_split", {})
        metric = (
            split_metrics.get(validation_name, {})
            if isinstance(split_metrics, Mapping)
            else {}
        )
        observed_days = (
            tuple(observed_days_by_split.get(validation_name, ()))
            if validation_name is not None
            else ()
        )
        samples_r = _daily_samples(
            daily_rows,
            candidate_id,
            validation_name,
            observed_days,
            value_key="exact_net_r",
        )
        samples_pips = _daily_samples(
            daily_rows,
            candidate_id,
            validation_name,
            observed_days,
            value_key="exact_net_pips",
        )
        p_value, se, evidence_status = _one_sided_p_and_se(
            samples_r, seed_key=f"{validation_name}:{candidate_id}:R"
        )
        average_daily_r = statistics.fmean(samples_r) if samples_r else None
        average_daily_pips = statistics.fmean(samples_pips) if samples_pips else None
        average_pips = (
            metric.get("average_net_pips") if isinstance(metric, Mapping) else None
        )
        average_r = metric.get("average_net_r") if isinstance(metric, Mapping) else None
        profit_r = (
            float(metric.get("gross_profit_r", 0.0))
            if isinstance(metric, Mapping)
            else 0.0
        )
        loss_r = (
            float(metric.get("gross_loss_r", 0.0))
            if isinstance(metric, Mapping)
            else 0.0
        )
        pf_pass = profit_r > 0.0 and (loss_r == 0.0 or profit_r / loss_r > 1.0)
        complete_outcomes_pass = bool(
            isinstance(metric, Mapping)
            and int(metric.get("unresolved_count", 0)) == 0
            and int(metric.get("purged_count", 0)) == 0
        )
        tests.append(
            {
                "candidate_id": candidate_id,
                "hypothesis_id": candidate["hypothesis_id"],
                "complexity": int(candidate["complexity"]),
                "simplicity_key": list(candidate.get("simplicity_key") or ()),
                "average_net_pips": average_pips,
                "average_daily_net_pips": average_daily_pips,
                "average_net_r": average_r,
                "average_daily_net_r": average_daily_r,
                "standard_error_r": se,
                "raw_one_sided_p": p_value,
                "holm_adjusted_p": 1.0,
                "profit_factor_pass": pf_pass,
                "complete_outcomes_pass": complete_outcomes_pass,
                "positive_mean_pass": average_r is not None and float(average_r) > 0.0,
                "utc_day_cluster_count": len(samples_r),
                "evidence_status": evidence_status,
            }
        )
    ordered = sorted(
        tests, key=lambda item: (item["raw_one_sided_p"], item["candidate_id"])
    )
    running = 0.0
    total = len(ordered)
    for index, item in enumerate(ordered):
        adjusted = min(1.0, float(item["raw_one_sided_p"]) * (total - index))
        running = max(running, adjusted)
        item["holm_adjusted_p"] = running
    winners: list[str] = []
    for hypothesis in (f"H{number:02d}" for number in range(1, 8)):
        eligible = [
            item
            for item in tests
            if item["hypothesis_id"] == hypothesis
            and item["positive_mean_pass"]
            and item["profit_factor_pass"]
            and item["complete_outcomes_pass"]
            and float(item["holm_adjusted_p"]) <= 0.05
        ]
        if not eligible:
            continue
        best = min(
            eligible,
            key=lambda item: (
                -float(item["average_daily_net_r"]),
                float(item["standard_error_r"]),
                tuple(item["simplicity_key"]),
                item["candidate_id"],
            ),
        )
        threshold = float(best["average_daily_net_r"]) - float(best["standard_error_r"])
        within_one_se = [
            item for item in eligible if float(item["average_daily_net_r"]) >= threshold
        ]
        chosen = min(
            within_one_se,
            key=lambda item: (
                tuple(item["simplicity_key"]),
                -float(item["average_daily_net_r"]),
                item["candidate_id"],
            ),
        )
        winners.append(str(chosen["candidate_id"]))
    return {
        "policy": "HOLM_182_THEN_POSITIVE_MEAN_PF_GT_ONE_THEN_ONE_SE_SIMPLEST_PER_H_FAMILY",
        "familywise_alpha": 0.05,
        "daily_cluster_policy": "FIXED_OBSERVED_UTC_DAYS_ZERO_FILLED",
        "selection_return_unit": "EQUAL_INITIAL_RISK_R",
        "daily_hypothesis_test": "EXACT_OR_MONTE_CARLO_SIGN_FLIP_V1",
        "sign_flip_exact_max_days": SIGN_FLIP_EXACT_MAX_DAYS,
        "sign_flip_monte_carlo_repetitions": SIGN_FLIP_MONTE_CARLO_REPETITIONS,
        "minimum_cluster_days": MIN_VALIDATION_CLUSTER_DAYS,
        "tested_candidate_count": len(tests),
        "no_signal_candidates_use_p_one": True,
        "winner_arm_ids": winners[:7],
        "candidate_tests": sorted(tests, key=lambda item: item["candidate_id"]),
    }


def _merge_metric_into_stat(stat: dict[str, Any], metric: Mapping[str, Any]) -> None:
    for key in (
        "raw_signal_count",
        "signal_count",
        "deoverlap_count",
        "embargoed_signal_count",
        "expired_unfilled_count",
        "filled_count",
        "resolved_count",
        "purged_count",
        "unresolved_count",
        "win_count",
        "loss_count",
        "flat_count",
    ):
        stat[key] += int(metric.get(key, 0))
    for key in (
        "gross_mid_pips",
        "spread_drag_pips",
        "exact_net_pips",
        "gross_mid_r",
        "spread_drag_r",
        "exact_net_r",
        "gross_profit_pips",
        "gross_loss_pips",
        "gross_profit_r",
        "gross_loss_r",
    ):
        stat[key] += float(metric.get(key, 0.0))
    for reason, count in (metric.get("reason_counts") or {}).items():
        stat["reason_counts"][str(reason)] += int(count)


def _bootstrap_lower_bound(samples: Sequence[float], *, seed_key: str) -> float | None:
    if len(samples) < MIN_VALIDATION_CLUSTER_DAYS:
        return None
    if len(set(float(item) for item in samples)) == 1:
        return float(samples[0])
    generator = random.Random(
        int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:16], 16)
    )
    size = len(samples)
    means = sorted(
        sum(float(samples[generator.randrange(size)]) for _ in range(size)) / size
        for _ in range(BOOTSTRAP_INTERVAL_REPETITIONS)
    )
    return means[max(0, int(0.05 * len(means)) - 1)]


def _portfolio_result(
    candidate_ids: Sequence[str],
    *,
    split_name: str | None,
    metrics_by_id: Mapping[str, Mapping[str, Any]],
    daily_rows: Sequence[Mapping[str, Any]],
    observed_days: Sequence[str],
) -> dict[str, Any]:
    stat = _blank_stat()
    selected = tuple(candidate_ids)
    for candidate_id in selected:
        candidate = metrics_by_id[candidate_id]
        split_metrics = candidate.get("metrics_by_split")
        if not isinstance(split_metrics, Mapping) or split_name not in split_metrics:
            continue
        metric = split_metrics[split_name]
        if isinstance(metric, Mapping):
            _merge_metric_into_stat(stat, metric)
    daily_pips: dict[str, float] = defaultdict(float)
    daily_r: dict[str, float] = defaultdict(float)
    selected_set = set(selected)
    if split_name is not None:
        for row in daily_rows:
            if (
                row.get("split") == split_name
                and row.get("candidate_id") in selected_set
            ):
                daily_pips[str(row["utc_day"])] += float(row.get("exact_net_pips", 0.0))
                daily_r[str(row["utc_day"])] += float(row.get("exact_net_r", 0.0))
    samples_pips = [daily_pips.get(day, 0.0) for day in observed_days]
    samples_r = [daily_r.get(day, 0.0) for day in observed_days]
    pips_equity = pips_peak = pips_drawdown = 0.0
    r_equity = r_peak = r_drawdown = 0.0
    for pips_value, r_value in zip(samples_pips, samples_r):
        pips_equity += pips_value
        pips_peak = max(pips_peak, pips_equity)
        pips_drawdown = max(pips_drawdown, pips_peak - pips_equity)
        r_equity += r_value
        r_peak = max(r_peak, r_equity)
        r_drawdown = max(r_drawdown, r_peak - r_equity)
    metric = _metric_from_stat(stat)
    metric["max_drawdown_pips"] = None
    metric["max_daily_close_drawdown_pips"] = pips_drawdown
    metric["max_daily_close_drawdown_r"] = r_drawdown
    metric["closed_trade_drawdown_available"] = False
    average_daily_pips = statistics.fmean(samples_pips) if samples_pips else None
    average_daily_r = statistics.fmean(samples_r) if samples_r else None
    lower_bound_pips = _bootstrap_lower_bound(
        samples_pips,
        seed_key=f"portfolio:{split_name}:{','.join(selected)}:PIPS",
    )
    lower_bound_r = _bootstrap_lower_bound(
        samples_r,
        seed_key=f"portfolio:{split_name}:{','.join(selected)}:R",
    )
    average_trade_r = metric["average_net_r"]
    profit_r = float(metric["gross_profit_r"])
    loss_r = float(metric["gross_loss_r"])
    checks = {
        "resolved_trade_count_positive": int(metric["resolved_count"]) > 0,
        "unresolved_count_zero": int(metric["unresolved_count"]) == 0,
        "purged_count_zero": int(metric["purged_count"]) == 0,
        "average_net_r_per_trade_positive": average_trade_r is not None
        and float(average_trade_r) > 0.0,
        "average_daily_net_r_positive": average_daily_r is not None
        and average_daily_r > 0.0,
        "one_sided_95_daily_cluster_lower_bound_r_positive": lower_bound_r is not None
        and lower_bound_r > 0.0,
        "profit_factor_r_above_one": profit_r > 0.0
        and (loss_r == 0.0 or profit_r / loss_r > 1.0),
    }
    return {
        "candidate_ids": list(selected),
        "split": split_name,
        "metrics": metric,
        "return_unit_for_pass": "EQUAL_INITIAL_RISK_R",
        "utc_day_cluster_count": len(samples_r),
        "average_daily_net_pips": average_daily_pips,
        "one_sided_95_lower_bound_daily_net_pips": lower_bound_pips,
        "average_daily_net_r": average_daily_r,
        "one_sided_95_lower_bound_daily_net_r": lower_bound_r,
        "lower_bound_method": "UTC_DAY_CLUSTER_BOOTSTRAP_PERCENTILE_V1",
        "pass_checks": checks,
        "passed": bool(selected) and all(checks.values()),
    }


def combine_causal_multitf_s5_grid_runs(
    pair_runs: Sequence[Mapping[str, Any]],
    splits: Sequence[UtcSplit],
) -> dict[str, Any]:
    """Pool pair/day aggregates, select on validation, then seal holdout use."""

    normalized_splits = _normalise_splits(splits)
    if isinstance(pair_runs, (str, bytes)) or not isinstance(pair_runs, Sequence):
        raise ValueError("pair_runs must be a sequence")
    seen_pairs: set[str] = set()
    pooled_stats: dict[tuple[str, str], dict[str, Any]] = {
        (candidate_id, split.name): _blank_stat()
        for _spec, _orientation, _arm, candidate_id in _candidate_rows()
        for split in normalized_splits
    }
    pooled_daily: dict[tuple[str, str, str], dict[str, Any]] = {}
    evaluated_pairs: list[str] = []
    unavailable_pairs: list[str] = []
    observed_days: dict[str, set[str]] = {
        split.name: set() for split in normalized_splits
    }
    for result in pair_runs:
        if not isinstance(result, Mapping):
            raise ValueError("each pair run must be a mapping")
        pair = str(result.get("pair") or "")
        if not pair or pair in seen_pairs:
            raise ValueError("pair runs must identify unique non-empty pairs")
        seen_pairs.add(pair)
        if result.get("status") == "UNAVAILABLE":
            unavailable_pairs.append(pair)
            continue
        if (
            result.get("contract") != GRID_CONTRACT
            or int(result.get("candidate_count", -1)) != 182
        ):
            raise ValueError("pair run does not satisfy causal grid V1")
        evaluated_pairs.append(pair)
        aggregation = result.get("aggregation")
        if not isinstance(aggregation, Mapping):
            raise ValueError("pair run aggregation is invalid")
        pair_observed = aggregation.get("observed_utc_days_by_split")
        if not isinstance(pair_observed, Mapping):
            raise ValueError("pair run observed UTC day calendar is invalid")
        for split in normalized_splits:
            values = pair_observed.get(split.name)
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise ValueError("pair run observed UTC days are invalid")
            for value in values:
                day = str(value)
                try:
                    datetime.fromisoformat(day)
                except ValueError as error:
                    raise ValueError("pair run observed UTC day is invalid") from error
                observed_days[split.name].add(day)
        metrics = result.get("candidate_metrics")
        if (
            not isinstance(metrics, Sequence)
            or isinstance(metrics, (str, bytes))
            or len(metrics) != 182
        ):
            raise ValueError("pair run candidate metrics must contain all 182 rows")
        metric_ids: set[str] = set()
        for candidate in metrics:
            if not isinstance(candidate, Mapping):
                raise ValueError("pair candidate metric must be a mapping")
            candidate_id = str(candidate.get("candidate_id") or "")
            if candidate_id in metric_ids or not candidate_id:
                raise ValueError("pair candidate ids must be unique")
            metric_ids.add(candidate_id)
            split_metrics = candidate.get("metrics_by_split")
            if not isinstance(split_metrics, Mapping):
                raise ValueError("pair candidate split metrics are invalid")
            for split in normalized_splits:
                metric = split_metrics.get(split.name)
                if not isinstance(metric, Mapping):
                    raise ValueError("pair candidate is missing a declared split")
                _merge_metric_into_stat(
                    pooled_stats[(candidate_id, split.name)], metric
                )
        rows = result.get("daily_aggregates")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise ValueError("pair daily aggregates are invalid")
        for row in rows:
            if not isinstance(row, Mapping) or row.get("pair") != pair:
                raise ValueError("pair daily aggregate binding is invalid")
            candidate_id = str(row.get("candidate_id") or "")
            split_name = str(row.get("split") or "")
            day = str(row.get("utc_day") or "")
            if (candidate_id, split_name) not in pooled_stats:
                raise ValueError("pair daily aggregate scope is invalid")
            try:
                datetime.fromisoformat(day)
            except ValueError as error:
                raise ValueError("pair daily aggregate UTC day is invalid") from error
            target = pooled_daily.setdefault(
                (candidate_id, split_name, day), _blank_stat()
            )
            target["resolved_count"] += int(row.get("resolved_count", 0))
            target["win_count"] += int(row.get("win_count", 0))
            target["loss_count"] += int(row.get("loss_count", 0))
            target["flat_count"] += int(row.get("flat_count", 0))
            for key in (
                "gross_mid_pips",
                "spread_drag_pips",
                "exact_net_pips",
                "gross_mid_r",
                "spread_drag_r",
                "exact_net_r",
                "gross_profit_pips",
                "gross_loss_pips",
                "gross_profit_r",
                "gross_loss_r",
            ):
                target[key] += float(row.get(key, 0.0))
            for reason, count in (row.get("reason_counts") or {}).items():
                target["reason_counts"][str(reason)] += int(count)

    global_daily = _daily_rows("GLOBAL", pooled_daily)
    daily_by_candidate_split: dict[tuple[str, str], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in global_daily:
        daily_by_candidate_split[(str(row["candidate_id"]), str(row["split"]))].append(
            row
        )
    daily_drawdown_pips: dict[tuple[str, str], float] = {}
    daily_drawdown_r: dict[tuple[str, str], float] = {}
    for key, rows in daily_by_candidate_split.items():
        pips_equity = pips_peak = pips_drawdown = 0.0
        r_equity = r_peak = r_drawdown = 0.0
        for row in sorted(rows, key=lambda item: str(item["utc_day"])):
            pips_equity += float(row["exact_net_pips"])
            pips_peak = max(pips_peak, pips_equity)
            pips_drawdown = max(pips_drawdown, pips_peak - pips_equity)
            r_equity += float(row["exact_net_r"])
            r_peak = max(r_peak, r_equity)
            r_drawdown = max(r_drawdown, r_peak - r_equity)
        daily_drawdown_pips[key] = pips_drawdown
        daily_drawdown_r[key] = r_drawdown

    global_metrics: list[dict[str, Any]] = []
    for spec, orientation, arm, candidate_id in _candidate_rows():
        by_split: dict[str, Any] = {}
        for split in normalized_splits:
            metric = _metric_from_stat(pooled_stats[(candidate_id, split.name)])
            metric["max_drawdown_pips"] = None
            metric["max_daily_close_drawdown_pips"] = daily_drawdown_pips.get(
                (candidate_id, split.name), 0.0
            )
            metric["max_daily_close_drawdown_r"] = daily_drawdown_r.get(
                (candidate_id, split.name), 0.0
            )
            metric["closed_trade_drawdown_available"] = False
            metric["active_day_count"] = len(
                daily_by_candidate_split[(candidate_id, split.name)]
            )
            by_split[split.name] = metric
        global_metrics.append(
            {
                "candidate_id": candidate_id,
                "hypothesis_id": spec.hypothesis_id,
                "family": spec.family,
                "orientation": orientation,
                "arm_id": arm.arm_id,
                "complexity": spec.complexity
                + arm.complexity
                + int(orientation == "INVERSE"),
                "simplicity_key": list(_simplicity_key(orientation, arm)),
                "metrics_by_split": by_split,
            }
        )
    validation_name = _named_split(normalized_splits, "VALIDATION")
    holdout_name = _named_split(normalized_splits, "HOLDOUT")
    observed_days_by_split = {
        split.name: sorted(observed_days[split.name]) for split in normalized_splits
    }
    selection = _select_validation(
        global_metrics,
        global_daily,
        validation_name,
        observed_days_by_split,
    )
    winners = list(selection["winner_arm_ids"])
    by_id = {str(row["candidate_id"]): row for row in global_metrics}
    winner_set = set(winners)
    selection_receipt_body = {
        "contract": "QR_CAUSAL_MULTITF_S5_GRID_SELECTION_RECEIPT_V1",
        "validation_split": validation_name,
        "observed_validation_utc_days": observed_days_by_split.get(validation_name, [])
        if validation_name is not None
        else [],
        "selected_arm_ids": winners,
        "multiple_testing": selection,
        "selection_uses_holdout": False,
    }
    selection_receipt_sha256 = _canonical_sha(selection_receipt_body)
    holdout_metrics = [
        {
            "candidate_id": candidate_id,
            "metrics": dict(by_id[candidate_id]["metrics_by_split"][holdout_name]),
        }
        for candidate_id in winners
        if holdout_name is not None
    ]
    redacted_metrics: list[dict[str, Any]] = []
    for row in global_metrics:
        candidate_id = str(row["candidate_id"])
        visible_splits = {
            split_name: dict(metric)
            for split_name, metric in row["metrics_by_split"].items()
            if split_name != holdout_name or candidate_id in winner_set
        }
        redacted_metrics.append({**row, "metrics_by_split": visible_splits})
    visible_daily = [
        row
        for row in global_daily
        if row.get("split") != holdout_name
        or str(row.get("candidate_id")) in winner_set
    ]
    validation_portfolio = _portfolio_result(
        winners,
        split_name=validation_name,
        metrics_by_id=by_id,
        daily_rows=global_daily,
        observed_days=observed_days_by_split.get(validation_name, ())
        if validation_name is not None
        else (),
    )
    holdout_portfolio = _portfolio_result(
        winners,
        split_name=holdout_name,
        metrics_by_id=by_id,
        daily_rows=global_daily,
        observed_days=observed_days_by_split.get(holdout_name, ())
        if holdout_name is not None
        else (),
    )
    return {
        "contract": GLOBAL_CONTRACT,
        "schema_version": 1,
        "status": "OK" if evaluated_pairs else "NO_AVAILABLE_PAIR_RUNS",
        "evaluated_pairs": evaluated_pairs,
        "unavailable_pairs": unavailable_pairs,
        "candidate_count": len(global_metrics),
        "candidate_metrics": redacted_metrics,
        "daily_aggregates": visible_daily,
        "observed_utc_days_by_split": observed_days_by_split,
        "selected_arm_ids": winners,
        "validation_winner_arm_ids": winners,
        "selection_receipt": selection_receipt_body,
        "selection_receipt_sha256": selection_receipt_sha256,
        "validation": {
            "split": validation_name,
            "selected_arm_ids": winners,
            "multiple_testing": selection,
            "selection_uses_holdout": False,
        },
        "holdout_evaluated_arm_ids": winners if holdout_name is not None else [],
        "holdout_metrics": holdout_metrics,
        "validation_portfolio": validation_portfolio,
        "holdout_portfolio": holdout_portfolio,
        "holdout_selection_unchanged": True,
        "holdout": {
            "split": holdout_name,
            "evaluated_arm_ids": winners if holdout_name is not None else [],
            "reselection_performed": False,
            "selection_unchanged": True,
        },
        **_AUTHORITY,
    }


__all__ = [
    "ArmSpec",
    "TechnicalSpec",
    "UtcSplit",
    "build_predeclared_arms_v1",
    "build_predeclared_catalog_v1",
    "combine_causal_multitf_s5_grid_runs",
    "run_causal_multitf_s5_grid",
]
