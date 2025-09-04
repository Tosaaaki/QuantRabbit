"""
analysis.instrument_selector
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Rank instruments by recent volatility (ATR on H4) and return top-K.
Focus on JPY crosses to keep pip handling consistent in current engine.
"""

from __future__ import annotations

import asyncio
from typing import List, Tuple

from indicators.calc_core import IndicatorEngine
from market_data.candle_fetcher import fetch_historical_candles
import pandas as pd


DEFAULT_WATCHLIST = [
    "USD_JPY",
    "GBP_JPY",
    "AUD_JPY",
    "NZD_JPY",
]


async def _atr_for(instrument: str) -> Tuple[str, float]:
    try:
        candles = await fetch_historical_candles(instrument, "H4", 60)
        if len(candles) < 20:
            return instrument, 0.0
        df = pd.DataFrame(candles)[["open", "high", "low", "close"]]
        fac = IndicatorEngine.compute(df)
        # ATR in price units -> convert to pips for JPY cross (x100)
        atr_pips = float(fac.get("atr", 0.0) or 0.0) * 100.0
        return instrument, atr_pips
    except Exception:
        return instrument, 0.0


async def rank_by_atr(watchlist: List[str] | None = None, top_k: int = 2) -> List[str]:
    wl = watchlist or DEFAULT_WATCHLIST
    tasks = [asyncio.create_task(_atr_for(ins)) for ins in wl]
    results = await asyncio.gather(*tasks)
    ranked = sorted(results, key=lambda x: x[1], reverse=True)
    return [ins for ins, _ in ranked[:top_k]]


def is_resource_currency_pair(instrument: str) -> bool:
    # Simple heuristic: AUD/JPY, NZD/JPY and e.g., USD/ZAR, USD/TRY (not JPY cross) if added later
    return instrument in ("AUD_JPY", "NZD_JPY")

