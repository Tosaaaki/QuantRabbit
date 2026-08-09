from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
import unittest

from hypothesis import given, strategies as st


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("admission", ROOT / "run_admission.py")
ADMISSION = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(ADMISSION)


class AdmissionProperties(unittest.TestCase):
    @given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), max_size=100))
    def test_drawdown_is_nonnegative(self, values: list[float]) -> None:
        self.assertGreaterEqual(ADMISSION.drawdown(values), 0.0)

    def test_prior_forecast_never_joins_future(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        index = {"EUR_USD": ([start, start + timedelta(minutes=2)], [{"v": 1}, {"v": 2}])}
        row, joined = ADMISSION.prior_forecast(index, "EUR_USD", start + timedelta(minutes=1))
        self.assertEqual(row, {"v": 1})
        self.assertLessEqual(joined, start + timedelta(minutes=1))

    def test_purge_removes_train_labels_overlapping_validation(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rows = []
        for i in range(10):
            rows.append({
                "feature_at_utc": (start + timedelta(hours=i)).isoformat(),
                "close_at_utc": (start + timedelta(hours=i + (10 if i == 5 else 0), minutes=1)).isoformat(),
            })
        train, validation, purged = ADMISSION.split_rows(rows)
        self.assertEqual(len(validation), 4)
        self.assertGreaterEqual(purged, 1)
        validation_start = ADMISSION.parse_time(validation[0]["feature_at_utc"])
        self.assertTrue(all(ADMISSION.parse_time(row["close_at_utc"]) < validation_start - ADMISSION.EMBARGO for row in train))

    def test_features_exclude_future_labels(self) -> None:
        row = {
            "feature_at_utc": "2026-01-01T00:00:00Z", "intended_price": 1.1, "pair": "EUR_USD",
            "side": "LONG", "lane_id": "x", "units": 1000, "tp": 1.2, "sl": 1.0,
            "forecast_direction": "UP", "forecast_confidence": .7, "forecast_at_utc": "2025-12-31T23:59:00Z",
            "forecast_horizon_min": 60, "net_jpy": 999999, "exit_reason": "TAKE_PROFIT_ORDER",
        }
        names = set(ADMISSION.features(row))
        self.assertFalse(names & ADMISSION.FORBIDDEN_FEATURES)


if __name__ == "__main__":
    unittest.main()
