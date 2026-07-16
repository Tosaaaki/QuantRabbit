from __future__ import annotations

import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from quant_rabbit.forecast_passive_limit_replay import (
    ceil_time,
    replay_metrics,
    select_independent_forecasts,
    simulate_market_bracket,
    simulate_market_stop_time_close,
    simulate_passive_limit,
)


@dataclass(frozen=True)
class _Ohlc:
    o: float
    h: float
    l: float  # noqa: E741 - mirrors the production OANDA shape
    c: float


@dataclass(frozen=True)
class _Candle:
    timestamp_utc: datetime
    bid: _Ohlc
    ask: _Ohlc


@dataclass(frozen=True)
class _Forecast:
    source_index: int
    timestamp_utc: datetime
    pair: str
    direction: str
    current_price: float
    target_price: float
    invalidation_price: float


def _candle(ts: datetime, *, bid: tuple[float, float, float, float], spread: float = 0.0002) -> _Candle:
    bid_ohlc = _Ohlc(*bid)
    return _Candle(
        timestamp_utc=ts,
        bid=bid_ohlc,
        ask=_Ohlc(*(value + spread for value in bid)),
    )


class ForecastPassiveLimitReplayTest(unittest.TestCase):
    def test_ceil_time_uses_first_boundary_after_subminute_forecast(self) -> None:
        value = datetime(2026, 7, 15, 0, 0, 12, tzinfo=timezone.utc)
        self.assertEqual(
            ceil_time(value, timedelta(minutes=1)),
            datetime(2026, 7, 15, 0, 1, tzinfo=timezone.utc),
        )

    def test_long_joins_bid_and_exits_take_profit_on_bid(self) -> None:
        start = datetime(2026, 7, 15, 0, 1, tzinfo=timezone.utc)
        row = _Forecast(1, start - timedelta(seconds=30), "EUR_USD", "UP", 1.1001, 1.1006, 1.0997)
        candles = [
            _candle(start, bid=(1.1000, 1.1002, 1.0998, 1.1001)),
            _candle(start + timedelta(minutes=1), bid=(1.1001, 1.1007, 1.1000, 1.1006)),
        ]
        result = simulate_passive_limit(row, candles, horizon_min=5)
        self.assertTrue(result["filled"])
        self.assertEqual(result["entry_price"], 1.1)
        self.assertEqual(result["exit_reason"], "TAKE_PROFIT")
        self.assertAlmostEqual(result["realized_pips"], 6.0)

    def test_short_uses_bid_to_fill_and_ask_to_stop(self) -> None:
        start = datetime(2026, 7, 15, 0, 1, tzinfo=timezone.utc)
        row = _Forecast(2, start, "EUR_USD", "DOWN", 1.1001, 1.0995, 1.1006)
        candles = [
            _candle(start, bid=(1.1000, 1.1003, 1.0999, 1.1002)),
            _candle(start + timedelta(minutes=1), bid=(1.1002, 1.1005, 1.1000, 1.1004)),
        ]
        result = simulate_passive_limit(row, candles, horizon_min=5)
        self.assertTrue(result["filled"])
        self.assertEqual(result["entry_price"], 1.1002)
        self.assertEqual(result["exit_reason"], "STOP_LOSS")
        self.assertAlmostEqual(result["realized_pips"], -4.0)

    def test_unfilled_limit_is_not_a_trade(self) -> None:
        start = datetime(2026, 7, 15, 0, 1, tzinfo=timezone.utc)
        row = _Forecast(3, start, "EUR_USD", "UP", 1.1001, 1.1007, 1.0996)
        candles = [
            _candle(start, bid=(1.1000, 1.1005, 1.1000, 1.1004)),
            _candle(start + timedelta(minutes=1), bid=(1.1004, 1.1008, 1.1003, 1.1007)),
        ]
        result = simulate_passive_limit(row, candles, horizon_min=2)
        self.assertFalse(result["filled"])
        self.assertEqual(result["status"], "UNFILLED_EXPIRED")

    def test_fixed_pip_vehicle_separates_entry_ttl_from_hold(self) -> None:
        start = datetime(2026, 7, 15, 0, 1, tzinfo=timezone.utc)
        row = _Forecast(30, start, "EUR_USD", "UP", 1.1001, None, None)  # type: ignore[arg-type]
        candles = [
            _candle(start, bid=(1.1000, 1.1002, 1.1001, 1.1001)),
            _candle(start + timedelta(minutes=1), bid=(1.1001, 1.1002, 1.0998, 1.1001)),
            _candle(start + timedelta(minutes=2), bid=(1.1001, 1.1004, 1.1000, 1.1003)),
        ]
        result = simulate_passive_limit(
            row,
            candles,
            horizon_min=2,
            entry_ttl_min=2,
            max_hold_min=3,
            reward_pips=3,
            risk_pips=5,
        )
        self.assertTrue(result["filled"])
        self.assertEqual(result["geometry_source"], "FIXED_PIPS")
        self.assertEqual(result["exit_reason"], "TAKE_PROFIT")
        self.assertAlmostEqual(result["realized_pips"], 3.0)

    def test_limit_cannot_fill_after_entry_ttl(self) -> None:
        start = datetime(2026, 7, 15, 0, 1, tzinfo=timezone.utc)
        row = _Forecast(31, start, "EUR_USD", "UP", 1.1001, None, None)  # type: ignore[arg-type]
        candles = [
            _candle(start, bid=(1.1000, 1.1002, 1.1001, 1.1001)),
            _candle(start + timedelta(minutes=1), bid=(1.1001, 1.1002, 1.0997, 1.1000)),
        ]
        result = simulate_passive_limit(
            row,
            candles,
            horizon_min=1,
            entry_ttl_min=1,
            max_hold_min=5,
            reward_pips=3,
            risk_pips=5,
        )
        self.assertFalse(result["filled"])
        self.assertEqual(result["status"], "UNFILLED_EXPIRED")

    def test_market_long_enters_at_ask_and_exits_tp_on_bid(self) -> None:
        start = datetime(2026, 7, 15, 0, 1, tzinfo=timezone.utc)
        row = _Forecast(32, start, "EUR_USD", "UP", 1.1001, None, None)  # type: ignore[arg-type]
        candles = [
            _candle(start, bid=(1.1000, 1.1002, 1.0999, 1.1001)),
            _candle(
                start + timedelta(minutes=1),
                bid=(1.1001, 1.1006, 1.1000, 1.1005),
            ),
        ]
        result = simulate_market_bracket(
            row,
            candles,
            horizon_min=2,
            reward_pips=3,
            risk_pips=6,
        )
        self.assertTrue(result["filled"])
        self.assertEqual(result["entry_vehicle"], "MARKET")
        self.assertAlmostEqual(result["entry_price"], 1.1002)
        self.assertEqual(result["exit_reason"], "TAKE_PROFIT")
        self.assertAlmostEqual(result["realized_pips"], 3.0)

    def test_market_first_bar_target_is_not_passive_fill_ambiguity(self) -> None:
        start = datetime(2026, 7, 15, 0, 1, tzinfo=timezone.utc)
        row = _Forecast(33, start, "EUR_USD", "UP", 1.1001, None, None)  # type: ignore[arg-type]
        candles = [_candle(start, bid=(1.1000, 1.1006, 1.0999, 1.1001))]
        result = simulate_market_bracket(
            row,
            candles,
            horizon_min=1,
            reward_pips=3,
            risk_pips=6,
        )
        self.assertEqual(result["exit_reason"], "TAKE_PROFIT")
        self.assertAlmostEqual(result["realized_pips"], 3.0)

    def test_market_stop_time_close_lets_profit_run_to_horizon(self) -> None:
        start = datetime(2026, 7, 15, 0, 1, tzinfo=timezone.utc)
        row = _Forecast(34, start, "EUR_USD", "UP", 1.1001, None, None)  # type: ignore[arg-type]
        candles = [
            _candle(start, bid=(1.1000, 1.1003, 1.0999, 1.1002)),
            _candle(
                start + timedelta(minutes=1),
                bid=(1.1002, 1.1009, 1.1001, 1.1008),
            ),
            _candle(
                start + timedelta(minutes=2),
                bid=(1.1008, 1.1010, 1.1007, 1.1009),
            ),
        ]
        result = simulate_market_stop_time_close(
            row,
            candles,
            horizon_min=2,
            risk_pips=6,
        )
        self.assertEqual(result["exit_reason"], "TIME_CLOSE")
        self.assertAlmostEqual(result["realized_pips"], 6.0)

    def test_market_time_close_uses_next_real_quote_inside_grace(self) -> None:
        start = datetime(2026, 7, 15, 0, 1, tzinfo=timezone.utc)
        row = _Forecast(35, start, "EUR_USD", "UP", 1.1001, None, None)  # type: ignore[arg-type]
        candles = [
            _candle(start, bid=(1.1000, 1.1002, 1.0999, 1.1001)),
            _candle(
                start + timedelta(minutes=1, seconds=15),
                bid=(1.1005, 1.1007, 1.1004, 1.1006),
            ),
        ]
        result = simulate_market_stop_time_close(
            row,
            candles,
            horizon_min=1,
            risk_pips=6,
            time_close_quote_grace=timedelta(seconds=30),
        )
        self.assertEqual(result["exit_reason"], "TIME_CLOSE")
        self.assertEqual(
            result["exit_at_utc"],
            (start + timedelta(minutes=1, seconds=15)).isoformat(),
        )

    def test_fill_bar_target_without_close_proof_is_conservative_loss(self) -> None:
        start = datetime(2026, 7, 15, 0, 1, tzinfo=timezone.utc)
        row = _Forecast(4, start, "EUR_USD", "UP", 1.1001, 1.1005, 1.0995)
        candles = [_candle(start, bid=(1.1000, 1.1006, 1.0997, 1.1002))]
        result = simulate_passive_limit(row, candles, horizon_min=1)
        self.assertEqual(result["exit_reason"], "AMBIGUOUS_TARGET_BEFORE_FILL")
        self.assertIsNone(result["realized_pips"])
        self.assertAlmostEqual(result["conservative_pips"], -5.0)

    def test_fixed_horizon_selection_is_pair_local_and_non_overlapping(self) -> None:
        start = datetime(2026, 7, 15, tzinfo=timezone.utc)
        rows = [
            _Forecast(1, start, "EUR_USD", "UP", 1.0, 1.1, 0.9),
            _Forecast(2, start + timedelta(minutes=5), "EUR_USD", "UP", 1.0, 1.1, 0.9),
            _Forecast(3, start + timedelta(minutes=5), "USD_JPY", "UP", 100.0, 101.0, 99.0),
            _Forecast(4, start + timedelta(minutes=15), "EUR_USD", "UP", 1.0, 1.1, 0.9),
        ]
        selected = select_independent_forecasts(rows, horizon_min=15)
        self.assertEqual([row.source_index for row in selected], [1, 3, 4])

    def test_metrics_charge_unresolved_fills_as_full_risk(self) -> None:
        metrics = replay_metrics(
            [
                {"timestamp_utc": "2026-07-15T00:00:00Z", "filled": True, "exit_reason": "TAKE_PROFIT", "realized_pips": 3.0, "conservative_pips": 3.0},
                {"timestamp_utc": "2026-07-15T01:00:00Z", "filled": True, "exit_reason": "OPEN_UNRESOLVED", "realized_pips": None, "conservative_pips": -2.0},
                {"timestamp_utc": "2026-07-15T02:00:00Z", "filled": False},
            ]
        )
        self.assertEqual(metrics["fills"], 2)
        self.assertEqual(metrics["open_unresolved_fills"], 1)
        self.assertEqual(metrics["mean_conservative_pips"], 0.5)


if __name__ == "__main__":
    unittest.main()
