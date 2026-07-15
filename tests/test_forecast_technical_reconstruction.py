from __future__ import annotations

import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from quant_rabbit.forecast_technical_reconstruction import (
    TECHNICAL_FEATURE_FIELDS,
    reconstruct_missing_technical_features,
)


@dataclass(frozen=True)
class _Ohlc:
    o: float
    h: float
    l: float  # noqa: E741 - mirrors the production OHLC fixture shape
    c: float


@dataclass(frozen=True)
class _Candle:
    timestamp_utc: datetime
    bid: _Ohlc
    ask: _Ohlc


class ForecastTechnicalReconstructionTest(unittest.TestCase):
    def _candles(
        self,
        *,
        start: datetime,
        count: int,
        future_jump_at: int | None = None,
    ) -> list[_Candle]:
        output: list[_Candle] = []
        price = 1.1000
        for index in range(count):
            drift = 0.000002
            if future_jump_at is not None and index >= future_jump_at:
                drift = 0.01
            opened = price
            price += drift
            bid = _Ohlc(
                o=opened,
                h=max(opened, price) + 0.00001,
                l=min(opened, price) - 0.00001,
                c=price,
            )
            ask = _Ohlc(
                o=bid.o + 0.0001,
                h=bid.h + 0.0001,
                l=bid.l + 0.0001,
                c=bid.c + 0.0001,
            )
            output.append(
                _Candle(
                    timestamp_utc=start + timedelta(minutes=index),
                    bid=bid,
                    ask=ask,
                )
            )
        return output

    def _legacy_row(self, forecast_at: datetime) -> dict[str, object]:
        return {
            "source_index": 1,
            "timestamp_utc": forecast_at.isoformat(),
            "pair": "EUR_USD",
            "direction": "UP",
            "forecast_direction": "UP",
            **{field: "MISSING" for field in TECHNICAL_FEATURE_FIELDS},
        }

    def test_future_candle_cannot_change_reconstructed_features(self) -> None:
        start = datetime(2026, 7, 1, tzinfo=timezone.utc)
        forecast_at = start + timedelta(hours=24, minutes=1)
        causal = self._candles(start=start, count=24 * 60 + 1)
        with_future = causal + self._candles(
            start=forecast_at,
            count=60,
            future_jump_at=0,
        )

        first, first_stats = reconstruct_missing_technical_features(
            [self._legacy_row(forecast_at)],
            {"EUR_USD": causal},
        )
        second, second_stats = reconstruct_missing_technical_features(
            [self._legacy_row(forecast_at)],
            {"EUR_USD": with_future},
        )

        self.assertEqual(first_stats["reconstructed_rows"], 1)
        self.assertEqual(second_stats["reconstructed_rows"], 1)
        self.assertEqual(
            {field: first[0][field] for field in TECHNICAL_FEATURE_FIELDS},
            {field: second[0][field] for field in TECHNICAL_FEATURE_FIELDS},
        )
        self.assertEqual(
            first[0]["technical_reconstruction_sha256"],
            second[0]["technical_reconstruction_sha256"],
        )

    def test_point_in_time_features_are_never_overwritten(self) -> None:
        start = datetime(2026, 7, 1, tzinfo=timezone.utc)
        forecast_at = start + timedelta(hours=24, minutes=1)
        row = self._legacy_row(forecast_at)
        row.update(
            {
                "technical_context_sha256": "a" * 64,
                "technical_regime": "RANGE",
                "technical_selected_method": "RANGE_ROTATION",
            }
        )

        output, stats = reconstruct_missing_technical_features(
            [row],
            {"EUR_USD": self._candles(start=start, count=24 * 60 + 1)},
        )

        self.assertEqual(stats["reconstructed_rows"], 0)
        self.assertEqual(stats["exact_or_partial_context_rows_preserved"], 1)
        self.assertEqual(output[0], row)

    def test_complete_history_produces_nonzero_technical_coverage(self) -> None:
        start = datetime(2026, 7, 1, tzinfo=timezone.utc)
        forecast_at = start + timedelta(hours=24, minutes=1)

        output, stats = reconstruct_missing_technical_features(
            [self._legacy_row(forecast_at)],
            {"EUR_USD": self._candles(start=start, count=24 * 60 + 1)},
        )

        self.assertEqual(stats["reconstructed_rows"], 1)
        self.assertEqual(output[0]["technical_feature_source"], "HISTORICAL_BID_ASK_RECONSTRUCTION")
        for field in TECHNICAL_FEATURE_FIELDS:
            self.assertNotEqual(output[0][field], "MISSING")
            self.assertEqual(
                stats["technical_feature_coverage_after_reconstruction"][field],
                1.0,
            )

    def test_incomplete_lookback_remains_missing(self) -> None:
        start = datetime(2026, 7, 1, tzinfo=timezone.utc)
        forecast_at = start + timedelta(hours=12)
        row = self._legacy_row(forecast_at)

        output, stats = reconstruct_missing_technical_features(
            [row],
            {"EUR_USD": self._candles(start=start, count=12 * 60)},
        )

        self.assertEqual(stats["reconstructed_rows"], 0)
        self.assertEqual(stats["skipped_incomplete_lookback_rows"], 1)
        for field in TECHNICAL_FEATURE_FIELDS:
            self.assertEqual(output[0][field], "MISSING")


if __name__ == "__main__":
    unittest.main()
