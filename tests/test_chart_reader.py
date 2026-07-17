from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from quant_rabbit.analysis.candles import (
    TECHNICAL_CANDLE_PROVENANCE_INVALID,
    TECHNICAL_CANDLE_SPREAD_CONTAMINATED,
    Candle,
)
from quant_rabbit.analysis.chart_reader import DEFAULT_TIMEFRAMES, build_pair_chart
from quant_rabbit.instruments import NORMAL_SPREAD_PIPS
from quant_rabbit.risk import RiskPolicy
from quant_rabbit.strategy.directional_forecaster import synthesize_forecast


def _series(start: float, step: float, n: int = 100) -> list[Candle]:
    base = datetime(2026, 5, 4, tzinfo=timezone.utc)
    out: list[Candle] = []
    prev = start
    for i in range(n):
        c = start + step * i
        h = max(prev, c) + abs(step) * 0.5 + 0.01
        low_value = min(prev, c) - abs(step) * 0.5 - 0.01
        out.append(Candle(base + timedelta(minutes=5 * i), prev, h, low_value, c, 1000, True))
        prev = c
    return out


def _steady_h1_trend(start: float, step: float, n: int = 100) -> list[Candle]:
    base = datetime(2026, 5, 4, tzinfo=timezone.utc)
    out: list[Candle] = []
    previous = start
    for index in range(n):
        close = previous + step
        out.append(Candle(
            base + timedelta(hours=index),
            previous,
            max(previous, close) + abs(step) * 0.25,
            min(previous, close) - abs(step) * 0.25,
            close,
            1000,
            True,
        ))
        previous = close
    return out


def _mba_entry(
    index: int,
    *,
    granularity: str = "M1",
    ask_widening: float = 0.0,
) -> dict[str, object]:
    step = {
        "M1": timedelta(minutes=1),
        "M5": timedelta(minutes=5),
        "D": timedelta(days=1),
    }[granularity]
    base_time = {
        "M1": datetime(2026, 7, 14, tzinfo=timezone.utc),
        "M5": datetime(2026, 7, 13, 21, 5, tzinfo=timezone.utc),
        "D": datetime(2026, 6, 1, tzinfo=timezone.utc),
    }[granularity]
    timestamp = base_time + step * index
    base = 1.1000 + index * 0.00001
    reference_mid = {"o": base, "h": base + 0.0002, "l": base - 0.0002, "c": base + 0.00005}
    bid = {key: value - 0.00003 for key, value in reference_mid.items()}
    ask = {key: value + 0.00003 + ask_widening for key, value in reference_mid.items()}
    mid = {key: (bid[key] + ask[key]) / 2.0 for key in reference_mid}
    return {
        "time": timestamp.isoformat().replace("+00:00", "Z"),
        "complete": True,
        "mid": {key: f"{value:.5f}" for key, value in mid.items()},
        "bid": {key: f"{value:.5f}" for key, value in bid.items()},
        "ask": {key: f"{value:.5f}" for key, value in ask.items()},
        "volume": 100,
    }


def _above_eur_usd_execution_cap_widening() -> float:
    cap_pips = NORMAL_SPREAD_PIPS["EUR_USD"] * RiskPolicy().max_spread_multiple
    return (cap_pips + 1.0 - 0.6) / 10000.0


def _fresh_fast_mba_entries(
    *,
    granularity: str,
    count: int,
    contaminated_indices: set[int],
) -> list[dict[str, object]]:
    step = timedelta(minutes=1 if granularity == "M1" else 5)
    latest = datetime(
        2026,
        7,
        14,
        0,
        44 if granularity == "M1" else 40,
        tzinfo=timezone.utc,
    )
    first = latest - step * (count - 1)
    entries: list[dict[str, object]] = []
    for index in range(count):
        entry = _mba_entry(
            index,
            granularity=granularity,
            ask_widening=(
                _above_eur_usd_execution_cap_widening()
                if index in contaminated_indices
                else 0.0
            ),
        )
        entry["time"] = (first + step * index).isoformat().replace("+00:00", "Z")
        entries.append(entry)
    return entries


class ChartReaderTest(unittest.TestCase):
    def test_default_pair_chart_reads_full_timeframe_stack(self) -> None:
        candles_by_tf = {
            tf: _series(156.0, 0.02, n=220)
            for tf in DEFAULT_TIMEFRAMES
        }

        chart = build_pair_chart(
            "USD_JPY",
            client=None,  # type: ignore[arg-type]
            candles_by_tf=candles_by_tf,
        )

        self.assertEqual(tuple(view.granularity for view in chart.views), DEFAULT_TIMEFRAMES)
        self.assertTrue(all(view.indicator_series["adx_14"] for view in chart.views))
        self.assertTrue(all(view.indicator_series["atr_pips"] for view in chart.views))
        self.assertTrue(
            all(view.indicator_series["ema_12_minus_50_pips"] for view in chart.views)
        )
        self.assertTrue(
            all(
                len(view.indicator_series["adx_14"]) <= 30
                for view in chart.views
            )
        )
        self.assertEqual(chart.technical_candle_integrity["evaluation_status"], "NOT_EVALUATED")
        self.assertEqual(chart.technical_candle_integrity["source"], "INJECTED")
        self.assertFalse(chart.technical_candle_integrity["forecast_blocking"])

    def test_production_chart_quarantines_rollover_mid_bar_using_one_mba_packet(self) -> None:
        class _Client:
            def __init__(self) -> None:
                self.calls: list[tuple[str, dict[str, str]]] = []

            def get_json(self, path: str, params: dict[str, str]) -> dict[str, object]:
                self.calls.append((path, params))
                entries = [_mba_entry(index) for index in range(40)]
                entries.append(_mba_entry(
                    40,
                    ask_widening=_above_eur_usd_execution_cap_widening(),
                ))
                return {
                    "instrument": "EUR_USD",
                    "granularity": params["granularity"],
                    "candles": entries,
                }

        client = _Client()
        chart = build_pair_chart(  # type: ignore[arg-type]
            "EUR_USD", client=client, timeframes=("M1",), count=41
        )

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0][1]["price"], "MBA")
        self.assertEqual(chart.views[0].indicators.candles_count, 0)
        self.assertEqual(len(chart.views[0].recent_candles), 0)
        self.assertTrue(chart.technical_candle_integrity["forecast_blocking"])
        self.assertIn(
            TECHNICAL_CANDLE_SPREAD_CONTAMINATED,
            chart.technical_candle_integrity["blocking_codes"],
        )
        self.assertIn("technical candle integrity BLOCKED", " ".join(chart.warnings))

    def test_generated_m1_m5_one_clean_of_200_forces_unclear_zero(self) -> None:
        class _Client:
            def get_json(self, _path: str, params: dict[str, str]) -> dict[str, object]:
                granularity = params["granularity"]
                return {
                    "instrument": "EUR_USD",
                    "granularity": granularity,
                    "candles": _fresh_fast_mba_entries(
                        granularity=granularity,
                        count=200,
                        contaminated_indices=set(range(199)),
                    ),
                }

        chart = build_pair_chart(  # type: ignore[arg-type]
            "EUR_USD",
            client=_Client(),
            timeframes=("M1", "M5"),
            count=200,
        )
        pair_chart = chart.to_dict()
        pair_chart["generated_at_utc"] = "2026-07-14T00:45:00+00:00"
        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[SimpleNamespace(
                direction="UP",
                bonus_magnitude=100.0,
                confidence=1.0,
                rationale="strong up detector",
            )],
            correlation_signals=[],
            paths=[],
            now_utc=datetime(2026, 7, 14, 0, 45, tzinfo=timezone.utc),
        )

        for timeframe in ("M1", "M5"):
            receipt = chart.technical_candle_integrity["timeframes"][timeframe]
            self.assertEqual(receipt["raw_entry_count"], 200)
            self.assertEqual(receipt["contaminated_count"], 199)
            self.assertEqual(receipt["clean_count"], 1)
            self.assertEqual(receipt["recent_clean_tail_count"], 1)
            self.assertTrue(receipt["forecast_blocking"])
        self.assertEqual([view.indicators.candles_count for view in chart.views], [1, 1])
        self.assertEqual(forecast.direction, "UNCLEAR")
        self.assertEqual(forecast.confidence, 0.0)

    def test_generated_older_contamination_with_30_clean_tail_remains_tradeable(self) -> None:
        class _Client:
            def get_json(self, _path: str, params: dict[str, str]) -> dict[str, object]:
                granularity = params["granularity"]
                return {
                    "instrument": "EUR_USD",
                    "granularity": granularity,
                    "candles": _fresh_fast_mba_entries(
                        granularity=granularity,
                        count=70,
                        contaminated_indices=set(range(40)),
                    ),
                }

        chart = build_pair_chart(  # type: ignore[arg-type]
            "EUR_USD",
            client=_Client(),
            timeframes=("M1", "M5"),
            count=70,
        )
        pair_chart = chart.to_dict()
        pair_chart["generated_at_utc"] = "2026-07-14T00:45:00+00:00"
        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[SimpleNamespace(
                direction="UP",
                bonus_magnitude=100.0,
                confidence=1.0,
                rationale="strong up detector",
            )],
            correlation_signals=[],
            paths=[],
            now_utc=datetime(2026, 7, 14, 0, 45, tzinfo=timezone.utc),
        )

        self.assertEqual(chart.technical_candle_integrity["evaluation_status"], "DEGRADED")
        self.assertFalse(chart.technical_candle_integrity["forecast_blocking"])
        for timeframe in ("M1", "M5"):
            receipt = chart.technical_candle_integrity["timeframes"][timeframe]
            details = receipt["quarantine_details"]
            window = receipt["quarantine_details_window"]
            self.assertEqual(receipt["contaminated_count"], 40)
            self.assertEqual(len(details), 8)
            self.assertEqual(receipt["quarantine_details_truncated"], 32)
            self.assertEqual(window["start_index"], 32)
            self.assertEqual(window["total_count"], 40)
            self.assertEqual(
                window["total_code_counts"],
                {
                    TECHNICAL_CANDLE_SPREAD_CONTAMINATED: 40,
                    TECHNICAL_CANDLE_PROVENANCE_INVALID: 0,
                },
            )
            self.assertEqual(
                window["published_code_counts"],
                {
                    TECHNICAL_CANDLE_SPREAD_CONTAMINATED: 8,
                    TECHNICAL_CANDLE_PROVENANCE_INVALID: 0,
                },
            )
            self.assertEqual(
                window["omitted_code_counts"],
                {
                    TECHNICAL_CANDLE_SPREAD_CONTAMINATED: 32,
                    TECHNICAL_CANDLE_PROVENANCE_INVALID: 0,
                },
            )
            self.assertEqual(window["total_timestamped_count"], 40)
            self.assertEqual(window["published_timestamped_count"], 8)
            self.assertEqual(window["omitted_timestamped_count"], 32)
            self.assertEqual(
                window["latest_timestamp_utc"],
                details[-1]["timestamp_utc"],
            )
        self.assertEqual([view.indicators.candles_count for view in chart.views], [30, 30])
        self.assertEqual(forecast.direction, "UP")
        self.assertGreater(forecast.confidence, 0.0)

    def test_wide_daily_mba_does_not_block_clean_execution_timeframes(self) -> None:
        class _Client:
            def get_json(self, _path: str, params: dict[str, str]) -> dict[str, object]:
                widening = (
                    _above_eur_usd_execution_cap_widening()
                    if params["granularity"] == "D"
                    else 0.0
                )
                return {
                    "instrument": "EUR_USD",
                    "granularity": params["granularity"],
                    "candles": [
                        _mba_entry(
                            index,
                            granularity=params["granularity"],
                            ask_widening=widening,
                        )
                        for index in range(44)
                    ]
                }

        chart = build_pair_chart(  # type: ignore[arg-type]
            "EUR_USD",
            client=_Client(),
            timeframes=("M1", "M5", "D"),
            count=44,
        )
        pair_chart = chart.to_dict()
        pair_chart["generated_at_utc"] = "2026-07-14T00:45:00+00:00"
        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[SimpleNamespace(
                direction="UP",
                bonus_magnitude=100.0,
                confidence=1.0,
                rationale="strong up detector",
            )],
            correlation_signals=[],
            paths=[],
            now_utc=datetime(2026, 7, 14, 0, 45, tzinfo=timezone.utc),
        )

        integrity = chart.technical_candle_integrity
        self.assertEqual(integrity["evaluation_status"], "PASS")
        self.assertEqual(
            integrity["timeframes"]["D"]["spread_evaluation_mode"],
            "PROVENANCE_ONLY_HIGHER_TIMEFRAME",
        )
        self.assertEqual(integrity["timeframes"]["D"]["contaminated_count"], 0)
        self.assertEqual(forecast.direction, "UP")
        self.assertGreater(forecast.confidence, 0.0)

    def test_malformed_daily_mba_still_blocks_forecast(self) -> None:
        class _Client:
            def get_json(self, _path: str, params: dict[str, str]) -> dict[str, object]:
                entries = [
                    _mba_entry(
                        index,
                        granularity=params["granularity"],
                        ask_widening=(
                            _above_eur_usd_execution_cap_widening()
                            if params["granularity"] == "D"
                            else 0.0
                        ),
                    )
                    for index in range(44)
                ]
                if params["granularity"] == "D":
                    entries[-1].pop("ask")
                return {
                    "instrument": "EUR_USD",
                    "granularity": params["granularity"],
                    "candles": entries,
                }

        chart = build_pair_chart(  # type: ignore[arg-type]
            "EUR_USD",
            client=_Client(),
            timeframes=("M1", "M5", "D"),
            count=44,
        )
        pair_chart = chart.to_dict()
        pair_chart["generated_at_utc"] = "2026-07-14T00:45:00+00:00"
        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[SimpleNamespace(
                direction="UP",
                bonus_magnitude=100.0,
                confidence=1.0,
                rationale="strong up detector",
            )],
            correlation_signals=[],
            paths=[],
            now_utc=datetime(2026, 7, 14, 0, 45, tzinfo=timezone.utc),
        )

        self.assertTrue(chart.technical_candle_integrity["forecast_blocking"])
        self.assertIn(
            TECHNICAL_CANDLE_PROVENANCE_INVALID,
            chart.technical_candle_integrity["blocking_codes"],
        )
        self.assertEqual(forecast.direction, "UNCLEAR")
        self.assertEqual(forecast.confidence, 0.0)

    def test_uptrend_pair_scores_long_above_short(self) -> None:
        candles_by_tf = {
            "M5": _series(156.0, 0.04),
            "M15": _series(156.0, 0.05),
            "H1": _series(156.0, 0.08),
        }
        chart = build_pair_chart(
            "USD_JPY",
            client=None,  # type: ignore[arg-type]
            timeframes=tuple(candles_by_tf),
            candles_by_tf=candles_by_tf,
        )
        self.assertGreater(chart.long_score, chart.short_score)
        self.assertIn("USD_JPY", chart.chart_story)
        self.assertEqual(len(chart.views), 3)

    def test_downtrend_pair_scores_short_above_long(self) -> None:
        candles_by_tf = {
            "M5": _series(160.0, -0.04),
            "M15": _series(160.0, -0.05),
            "H1": _series(160.0, -0.08),
        }
        chart = build_pair_chart(
            "USD_JPY",
            client=None,  # type: ignore[arg-type]
            timeframes=tuple(candles_by_tf),
            candles_by_tf=candles_by_tf,
        )
        self.assertGreater(chart.short_score, chart.long_score)
        self.assertIn("TREND_DOWN", chart.dominant_regime)

    def test_chart_story_includes_indicator_fragments(self) -> None:
        candles_by_tf = {"M5": _series(157.0, 0.03)}
        chart = build_pair_chart(
            "USD_JPY",
            client=None,  # type: ignore[arg-type]
            timeframes=("M5",),
            candles_by_tf=candles_by_tf,
        )
        self.assertIn("ADX=", chart.chart_story)
        self.assertIn("ATR=", chart.chart_story)
        self.assertIn("RSI=", chart.chart_story)

    def test_short_history_pair_chart_uses_available_regime_reading(self) -> None:
        candles_by_tf = {"M5": _series(1.17, -0.00005, n=200)}

        chart = build_pair_chart(
            "EUR_USD",
            client=None,  # type: ignore[arg-type]
            timeframes=("M5",),
            candles_by_tf=candles_by_tf,
        )

        reading = chart.views[0].regime_reading
        self.assertIsNotNone(reading)
        assert reading is not None
        self.assertNotEqual(reading.state, "UNKNOWN")
        self.assertIn(reading.source, {"ohlc_dfa_atr_percentile", "indicator_set_M5"})
        self.assertEqual(reading.lookback_bars, 200)
        self.assertIn("Read=", chart.chart_story)

    def test_extended_confluence_publishes_price_range_bounds(self) -> None:
        candles_by_tf = {
            "H1": _series(1.1800, -0.0004, n=80),
            "H4": _series(1.1900, -0.0003, n=80),
        }

        chart = build_pair_chart(
            "EUR_USD",
            client=None,  # type: ignore[arg-type]
            timeframes=("H1", "H4"),
            candles_by_tf=candles_by_tf,
        )

        self.assertIsNotNone(chart.confluence["price_range_24h_low"])
        self.assertIsNotNone(chart.confluence["price_range_24h_high"])
        self.assertLess(chart.confluence["price_range_24h_low"], chart.confluence["price_range_24h_high"])
        self.assertIsNotNone(chart.confluence["price_range_7d_low"])
        self.assertIsNotNone(chart.confluence["price_range_7d_high"])
        self.assertLess(chart.confluence["price_range_7d_low"], chart.confluence["price_range_7d_high"])

    def test_raw_24h_to_h1_ratio_above_two_is_not_automatically_an_outlier(self) -> None:
        candles_by_tf = {"H1": _steady_h1_trend(1.1000, 0.0002, n=100)}

        chart = build_pair_chart(
            "EUR_USD",
            client=None,  # type: ignore[arg-type]
            timeframes=("H1",),
            candles_by_tf=candles_by_tf,
        )

        self.assertGreater(chart.confluence["range_24h_expansion_ratio"], 2.0)
        self.assertFalse(chart.confluence["range_24h_expansion_outlier"])
        self.assertEqual(
            chart.confluence["range_24h_sigma_multiple"],
            chart.confluence["range_24h_expansion_ratio"],
        )

    def test_24h_expansion_outlier_is_calculated_from_prior_rolling_windows(self) -> None:
        candles = _steady_h1_trend(1.1000, 0.0002, n=100)
        last = candles[-1]
        candles[-1] = Candle(
            last.timestamp_utc,
            last.open,
            last.high + 0.0200,
            last.low,
            last.close,
            last.volume,
            last.complete,
        )

        chart = build_pair_chart(
            "EUR_USD",
            client=None,  # type: ignore[arg-type]
            timeframes=("H1",),
            candles_by_tf={"H1": candles},
        )

        self.assertTrue(chart.confluence["range_24h_expansion_outlier"])
        self.assertGreater(
            chart.confluence["range_24h_expansion_ratio"],
            chart.confluence["range_24h_expansion_upper_fence"],
        )
        self.assertEqual(chart.confluence["range_24h_expansion_percentile"], 1.0)
        self.assertEqual(chart.confluence["range_24h_expansion_sample_count"], 76)


if __name__ == "__main__":
    unittest.main()
