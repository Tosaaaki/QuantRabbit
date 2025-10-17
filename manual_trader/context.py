"""Utilities for gathering market context for the manual trading assistant."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import pandas as pd

from analysis.regime_classifier import classify
from analysis.summary_ingestor import check_event_soon, get_latest_news
from indicators.calc_core import IndicatorEngine
from market_data.candle_fetcher import Candle, fetch_historical_candles

logger = logging.getLogger(__name__)


@dataclass
class FrameContext:
    timeframe: str
    regime: str
    factors: Dict[str, float]
    price_snapshot: Dict[str, float | str]
    recent_ohlc: List[Dict[str, float]]


@dataclass
class ManualContext:
    instrument: str
    timestamp: str
    event_window: bool
    macro: FrameContext
    micro: FrameContext
    latest_news: Dict[str, Any]


def _candles_to_df(candles: Iterable[Candle]) -> pd.DataFrame:
    rows = [
        {
            "open": float(c["open"]),
            "high": float(c["high"]),
            "low": float(c["low"]),
            "close": float(c["close"]),
        }
        for c in candles
    ]
    return pd.DataFrame(rows, dtype=float)


def _price_snapshot(candles: List[Candle], tail: int) -> Dict[str, float | str]:
    if not candles:
        return {}
    tail_candles = candles[-tail:]
    latest = tail_candles[-1]
    highs = [c["high"] for c in tail_candles]
    lows = [c["low"] for c in tail_candles]
    closes = [c["close"] for c in tail_candles]
    first_close = closes[0]
    last_close = closes[-1]
    range_high = max(highs)
    range_low = min(lows)
    return {
        "last_time": latest["time"].isoformat(timespec="seconds"),
        "last_close": round(last_close, 5),
        "range_high": round(range_high, 5),
        "range_low": round(range_low, 5),
        "range_span": round(range_high - range_low, 5),
        "change_pct": round(((last_close - first_close) / first_close) * 100.0, 3)
        if first_close
        else 0.0,
    }


def _build_frame(tf: str, candles: List[Candle]) -> FrameContext:
    if len(candles) < 20:
        raise ValueError(f"Not enough candles for timeframe {tf}")
    df = _candles_to_df(candles)
    factors = IndicatorEngine.compute(df)
    latest = candles[-1]
    factors["open"] = float(latest["open"])
    factors["high"] = float(latest["high"])
    factors["low"] = float(latest["low"])
    factors["close"] = float(latest["close"])
    regime = classify(factors, tf, event_mode=False)
    price_snapshot = _price_snapshot(candles, tail=10 if tf == "M1" else 6)
    recent = [
        {
            "time": c["time"].isoformat(timespec="seconds"),
            "open": float(c["open"]),
            "high": float(c["high"]),
            "low": float(c["low"]),
            "close": float(c["close"]),
        }
        for c in candles[-5:]
    ]
    return FrameContext(
        timeframe=tf,
        regime=regime,
        factors={k: float(v) for k, v in factors.items()},
        price_snapshot=price_snapshot,
        recent_ohlc=recent,
    )


async def gather_context(
    *,
    instrument: str = "USD_JPY",
    m1_count: int = 180,
    h4_count: int = 90,
) -> ManualContext:
    """Collect technical context, regime, and news for the manual trader."""

    try:
        m1_candles, h4_candles = await asyncio.gather(
            fetch_historical_candles(instrument, "M1", m1_count),
            fetch_historical_candles(instrument, "H4", h4_count),
        )
    except Exception as exc:  # pragma: no cover - network/stateful dependency
        logger.error("Failed to fetch historical candles: %s", exc)
        raise

    if len(m1_candles) < 20 or len(h4_candles) < 20:
        raise RuntimeError("Insufficient historical data for manual context")

    event_window = check_event_soon(within_minutes=30, min_impact=3)
    macro_frame = _build_frame("H4", h4_candles)
    micro_frame = _build_frame("M1", m1_candles)

    # Re-classify with event flag to bubble up "Event" regime when applicable
    if event_window:
        macro_frame.regime = classify(macro_frame.factors, "H4", event_mode=True)
        micro_frame.regime = classify(micro_frame.factors, "M1", event_mode=True)

    news = get_latest_news()

    latest_time = micro_frame.price_snapshot.get("last_time") or macro_frame.price_snapshot.get(
        "last_time"
    )
    timestamp = str(latest_time)

    return ManualContext(
        instrument=instrument,
        timestamp=timestamp,
        event_window=event_window,
        macro=macro_frame,
        micro=micro_frame,
        latest_news=news,
    )
