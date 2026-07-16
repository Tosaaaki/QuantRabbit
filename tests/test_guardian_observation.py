from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.analysis.candles import _technical_candles_from_payload
from quant_rabbit.guardian_observation import (
    ObservationContractError,
    observation_refresh_due,
    publish_blocked_current_m1,
    publish_current_m1,
    update_slow_retention,
    validate_current_m1_contract,
    validate_slow_retention_contract,
)
from quant_rabbit.instruments import (
    DEFAULT_TRADER_PAIRS,
    NORMAL_SPREAD_PIPS,
    OANDA_SPREAD_CALIBRATION_V1,
    instrument_pip_factor,
)
from quant_rabbit.risk import RiskPolicy


NOW = datetime(2026, 7, 16, 12, 0, 7, tzinfo=timezone.utc)
TF_SECONDS = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D": 86400,
}


def _view(pair: str, timeframe: str) -> dict:
    duration = TF_SECONDS[timeframe]
    latest_epoch = int(NOW.timestamp())
    latest_epoch -= latest_epoch % duration
    latest = datetime.fromtimestamp(latest_epoch - duration, tz=timezone.utc)
    factor = instrument_pip_factor(pair)
    decimals = 3 if factor == 100 else 5
    base = 100.0 if factor == 100 else 1.0
    half_spread = NORMAL_SPREAD_PIPS[pair] / factor / 2.0
    candles = []
    for index in range(120):
        started = latest - timedelta(seconds=duration * (119 - index))
        candles.append(
            {
                "time": started.isoformat().replace("+00:00", "Z"),
                "complete": True,
                "volume": 1,
                "mid": {
                    "o": f"{base:.{decimals}f}",
                    "h": f"{base:.{decimals}f}",
                    "l": f"{base:.{decimals}f}",
                    "c": f"{base:.{decimals}f}",
                },
                "bid": {
                    "o": f"{base - half_spread:.{decimals}f}",
                    "h": f"{base - half_spread:.{decimals}f}",
                    "l": f"{base - half_spread:.{decimals}f}",
                    "c": f"{base - half_spread:.{decimals}f}",
                },
                "ask": {
                    "o": f"{base + half_spread:.{decimals}f}",
                    "h": f"{base + half_spread:.{decimals}f}",
                    "l": f"{base + half_spread:.{decimals}f}",
                    "c": f"{base + half_spread:.{decimals}f}",
                },
            }
        )
    batch = _technical_candles_from_payload(
        {"instrument": pair, "granularity": timeframe, "candles": candles},
        pair=pair,
        granularity=timeframe,
        requested_count=120,
        pip_factor=factor,
        normal_spread_pips=NORMAL_SPREAD_PIPS[pair],
        max_spread_multiple=RiskPolicy().max_spread_multiple,
        spread_anomaly_cap_pips=OANDA_SPREAD_CALIBRATION_V1.pairs[pair].max_pips,
        spread_calibration_sha256=OANDA_SPREAD_CALIBRATION_V1.calibration_sha256,
    )
    tail = batch.integrity["recent_clean_tail_count"]
    recent = [
        {
            "t": candle.timestamp_utc.isoformat(),
            "complete": candle.complete,
            "o": candle.open,
            "h": candle.high,
            "l": candle.low,
            "c": candle.close,
            "v": candle.volume,
        }
        for candle in list(batch.candles[-tail:])[-30:]
    ]
    return {
        "granularity": timeframe,
        "recent_candles": recent,
        "indicators": {"candles_count": tail, "atr_pips": 5.0},
        "market_state": {
            "direction": "UP",
            "phase": "TREND",
            "readiness": "TRIGGERED",
            "trigger": "BREAKOUT_CLOSE",
            "structure": "BREAKOUT_ACTIVE",
            "location": "MIDDLE_THIRD",
            "value_zone": "FAIR_VALUE",
            "extension": "BALANCED",
            "evidence_complete": True,
        },
        "candle_integrity": dict(batch.integrity),
    }


def _source(pairs: tuple[str, ...], timeframes: tuple[str, ...]) -> dict:
    return {
        "generated_at_utc": NOW.isoformat(),
        "timeframes": list(timeframes),
        "pairs_requested": len(pairs),
        "pairs_succeeded": len(pairs),
        "pairs_failed": 0,
        "partial": False,
        "failures": [],
        "charts": [
            {"pair": pair, "views": [_view(pair, timeframe) for timeframe in timeframes]}
            for pair in pairs
        ],
    }


def _snapshot() -> dict:
    return {
        "fetched_at_utc": NOW.isoformat(),
        "positions": [],
        "orders": [],
        "quotes": {
            pair: {
                "bid": 100.0 if pair.endswith("_JPY") else 1.0,
                "ask": 100.01 if pair.endswith("_JPY") else 1.0001,
                "timestamp_utc": NOW.isoformat(),
            }
            for pair in DEFAULT_TRADER_PAIRS
        },
    }


class GuardianObservationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.all_pair_m1_source = _source(DEFAULT_TRADER_PAIRS, ("M1",))

    def test_publish_current_m1_is_exact_28_sealed_and_metrics_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "raw.json"
            snapshot_path = root / "snapshot.json"
            output_path = root / "current.json"
            source_path.write_text(json.dumps(self.all_pair_m1_source))
            snapshot_path.write_text(json.dumps(_snapshot()))

            value = publish_current_m1(
                source_path=source_path,
                snapshot_path=snapshot_path,
                output_path=output_path,
                active_pair_count=6,
                active_chart_wall_time_seconds=4.25,
                all_pair_m1_wall_time_seconds=8.5,
                post_chart_snapshot_wall_time_seconds=0.5,
                now_utc=NOW,
            )

            self.assertTrue(validate_current_m1_contract(value, now_utc=NOW))
            self.assertEqual([row["pair"] for row in value["charts"]], list(DEFAULT_TRADER_PAIRS))
            self.assertTrue(all([view["granularity"] for view in row["views"]] == ["M1"] for row in value["charts"]))
            self.assertEqual(value["request_metrics"]["active_rotation_candle_requests"], 42)
            self.assertEqual(value["request_metrics"]["all_pair_m1_candle_requests"], 28)
            self.assertEqual(value["request_metrics"]["total_candle_requests"], 70)
            self.assertEqual(value["request_metrics"]["quote_pairs_succeeded"], 28)
            self.assertFalse(value["live_permission"])
            self.assertFalse(value["broker_mutation_allowed"])
            self.assertFalse(observation_refresh_due(output_path, now_utc=NOW))

    def test_invalid_current_source_does_not_replace_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "raw.json"
            snapshot_path = root / "snapshot.json"
            output_path = root / "current.json"
            output_path.write_text('{"sentinel":"previous"}\n')
            invalid = json.loads(json.dumps(self.all_pair_m1_source))
            invalid["charts"] = invalid["charts"][:-1]
            invalid["pairs_succeeded"] -= 1
            invalid["pairs_failed"] = 1
            invalid["partial"] = True
            source_path.write_text(json.dumps(invalid))
            snapshot_path.write_text(json.dumps(_snapshot()))

            with self.assertRaises(ObservationContractError):
                publish_current_m1(
                    source_path=source_path,
                    snapshot_path=snapshot_path,
                    output_path=output_path,
                    active_pair_count=6,
                    active_chart_wall_time_seconds=0,
                    all_pair_m1_wall_time_seconds=0,
                    post_chart_snapshot_wall_time_seconds=0,
                    now_utc=NOW,
                )

            self.assertEqual(json.loads(output_path.read_text()), {"sentinel": "previous"})

    def test_slow_retention_merges_rotation_and_rejects_bad_source_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "active.json"
            output_path = root / "retained.json"
            first_pairs = DEFAULT_TRADER_PAIRS[:2]
            second_pairs = DEFAULT_TRADER_PAIRS[2:4]
            source_path.write_text(json.dumps(_source(first_pairs, ("M1", *(
                "M5", "M15", "M30", "H1", "H4", "D"
            )))))
            first = update_slow_retention(
                source_path=source_path,
                output_path=output_path,
                source_pairs=first_pairs,
                now_utc=NOW,
            )
            self.assertTrue(validate_slow_retention_contract(first))
            self.assertEqual(len(first["charts"]), 28)
            self.assertTrue(first["charts"][0]["coverage_complete"])
            self.assertFalse(first["charts"][2]["coverage_complete"])

            source_path.write_text(json.dumps(_source(second_pairs, ("M1", *(
                "M5", "M15", "M30", "H1", "H4", "D"
            )))))
            second = update_slow_retention(
                source_path=source_path,
                output_path=output_path,
                source_pairs=second_pairs,
                now_utc=NOW + timedelta(minutes=1),
            )
            self.assertTrue(validate_slow_retention_contract(second))
            self.assertTrue(all(second["charts"][index]["coverage_complete"] for index in range(4)))
            before_bad_update = output_path.read_bytes()
            bad_source = _source((DEFAULT_TRADER_PAIRS[4],), ("M1", "M5", "M15", "M30", "H1", "H4", "D"))
            bad_source["charts"][0]["views"][1]["candle_integrity"]["pair"] = "EUR_USD"
            source_path.write_text(json.dumps(bad_source))
            with self.assertRaises(ObservationContractError):
                update_slow_retention(
                    source_path=source_path,
                    output_path=output_path,
                    source_pairs=(DEFAULT_TRADER_PAIRS[4],),
                    now_utc=NOW + timedelta(minutes=2),
                )
            self.assertEqual(output_path.read_bytes(), before_bad_update)

    def test_blocked_current_is_empty_without_touching_slow_retention(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            current_path = root / "current.json"
            slow_path = root / "slow.json"
            slow_path.write_text('{"sentinel":"last-good"}\n')
            value = publish_blocked_current_m1(
                output_path=current_path,
                reason="M1_REFRESH_FAILED",
                active_pair_count=6,
                active_chart_wall_time_seconds=1,
                all_pair_m1_wall_time_seconds=2,
                post_chart_snapshot_wall_time_seconds=1,
                now_utc=NOW,
            )
            self.assertEqual(value["status"], "BLOCKED")
            self.assertEqual(value["charts"], [])
            self.assertEqual(json.loads(slow_path.read_text()), {"sentinel": "last-good"})
            self.assertFalse(observation_refresh_due(current_path, now_utc=NOW))
            self.assertTrue(
                observation_refresh_due(
                    current_path,
                    now_utc=NOW + timedelta(minutes=2),
                )
            )


if __name__ == "__main__":
    unittest.main()
