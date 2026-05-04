"""Indicator and chart-reading toolkit for the discretionary trader.

The trader (Codex) is the decision maker; this package builds the objective
material the trader looks at — OHLC candles fetched from OANDA, classical
technical indicators, and a regime/score view per pair-timeframe. Everything is
pure-Python, no third-party dependencies.
"""

from quant_rabbit.analysis.candles import Candle, fetch_candles, fetch_candles_via_client
from quant_rabbit.analysis.indicators import IndicatorSet, compute_indicators
from quant_rabbit.analysis.chart_reader import ChartView, PairChart, build_pair_chart

__all__ = [
    "Candle",
    "fetch_candles",
    "fetch_candles_via_client",
    "IndicatorSet",
    "compute_indicators",
    "ChartView",
    "PairChart",
    "build_pair_chart",
]
