"""Unit tests for strategy/projection_ledger.py."""

from __future__ import annotations

import tempfile
import threading
import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from quant_rabbit.strategy.projection_ledger import (
    CONFIDENCE_MAX_MULTIPLIER,
    CONFIDENCE_MIN_MULTIPLIER,
    IMMEDIATE_PROJECTION_RESOLUTION_WINDOW_MIN,
    LedgerEntry,
    compute_hit_rates,
    confidence_calibration,
    load_ledger,
    record_directional_forecast,
    record_projections,
    select_calibration_signal_name,
    setup_grade,
    verify_pending,
)


@dataclass
class _Sig:
    name: str
    direction: str
    lead_time_min: float
    confidence: float
    bonus_magnitude: float
    rationale: str = ""


class RecordTest(unittest.TestCase):
    def test_record_writes_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sigs = [
                _Sig("bb_squeeze", "EITHER", 75, 0.6, 8.0),
                _Sig("liquidity_sweep_high", "DOWN", 15, 0.9, 12.0, "M5 equal-highs at 1.17150 (1.0pip up)"),
            ]
            n = record_projections(sigs, pair="EUR_USD", current_price=1.17140,
                                    data_root=root)
            self.assertEqual(n, 2)
            entries = load_ledger(root)
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0].signal_name, "bb_squeeze")
            self.assertEqual(entries[0].resolution_status, "PENDING")
            self.assertEqual(entries[1].predicted_target_price, 1.17150)

    def test_record_immediate_followthrough_gets_resolution_window(self) -> None:
        from quant_rabbit.strategy.intent_generator import _expired_pending_projection_count

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime(2026, 6, 7, 14, 29, tzinfo=timezone.utc)
            sig = _Sig(
                "news_theme_followthrough",
                "DOWN",
                0.0,
                0.75,
                10.0,
                "EUR_USD SHORT news-theme follow-through",
            )
            n = record_projections(
                [sig],
                pair="EUR_USD",
                current_price=1.15225,
                data_root=root,
                now=now,
            )

            self.assertEqual(n, 1)
            entries = load_ledger(root)
            self.assertEqual(entries[0].lead_time_min, 0.0)
            self.assertAlmostEqual(
                entries[0].resolution_window_min,
                IMMEDIATE_PROJECTION_RESOLUTION_WINDOW_MIN,
            )
            self.assertEqual(
                _expired_pending_projection_count(
                    data_root=root,
                    validation_time_utc=now + timedelta(minutes=IMMEDIATE_PROJECTION_RESOLUTION_WINDOW_MIN / 2.0),
                ),
                0,
            )

    def test_record_empty_signals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            n = record_projections([], pair="EUR_USD", current_price=1.0, data_root=Path(tmp))
            self.assertEqual(n, 0)

    def test_record_dedupes_same_cycle_pair_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sig = _Sig("liquidity_sweep_low", "UP", 15, 0.9, 12.0, "M5 equal-lows at 1.16800 (4.0pip down)")
            first = record_projections(
                [sig],
                pair="EUR_USD",
                current_price=1.16840,
                data_root=root,
                cycle_id="2026-05-15T00:00:00Z",
            )
            second = record_projections(
                [sig],
                pair="EUR_USD",
                current_price=1.16840,
                data_root=root,
                cycle_id="2026-05-15T00:00:00Z",
            )
            self.assertEqual(first, 1)
            self.assertEqual(second, 0)
            entries = load_ledger(root)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].cycle_id, "2026-05-15T00:00:00Z")

    def test_record_projections_dedupes_same_cycle_pair_signal_race(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sig = _Sig("liquidity_sweep_low", "UP", 15, 0.9, 12.0, "M5 equal-lows at 1.16800 (4.0pip down)")
            thread_count = 6
            barrier = threading.Barrier(thread_count)

            def stale_precheck(*_args: object, **_kwargs: object) -> set[tuple]:
                barrier.wait(timeout=5)
                return set()

            results: list[int] = []
            errors: list[BaseException] = []

            def worker() -> None:
                try:
                    results.append(
                        record_projections(
                            [sig],
                            pair="EUR_USD",
                            current_price=1.16840,
                            data_root=root,
                            cycle_id="cycle-race",
                        )
                    )
                except BaseException as exc:
                    errors.append(exc)

            with mock.patch(
                "quant_rabbit.strategy.projection_ledger._existing_projection_keys",
                side_effect=stale_precheck,
            ):
                threads = [threading.Thread(target=worker) for _ in range(thread_count)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=5)

            self.assertEqual(errors, [])
            self.assertEqual(sum(results), 1)
            entries = load_ledger(root)
            self.assertEqual(len(entries), 1)

    def test_record_directional_forecast_feeds_calibration_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            forecast = type(
                "Forecast",
                (),
                {
                    "direction": "DOWN",
                    "confidence": 0.73,
                    "current_price": 1.1000,
                    "target_price": 1.0950,
                    "invalidation_price": 1.1030,
                    "horizon_min": 60,
                },
            )()

            first = record_directional_forecast(
                forecast,
                pair="EUR_USD",
                current_price=None,
                data_root=root,
                regime_at_emission="TREND",
                cycle_id="cycle-1",
            )
            second = record_directional_forecast(
                forecast,
                pair="EUR_USD",
                current_price=None,
                data_root=root,
                regime_at_emission="TREND",
                cycle_id="cycle-1",
            )

            self.assertEqual(first, 1)
            self.assertEqual(second, 0)
            entries = load_ledger(root)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].signal_name, "directional_forecast")
            self.assertEqual(entries[0].direction, "DOWN")
            self.assertEqual(entries[0].entry_price, 1.1000)
            self.assertEqual(entries[0].predicted_target_price, 1.0950)
            self.assertEqual(entries[0].predicted_invalidation_price, 1.1030)

    def test_record_directional_forecast_dedupes_same_cycle_race(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            forecast = type(
                "Forecast",
                (),
                {
                    "direction": "DOWN",
                    "confidence": 0.73,
                    "current_price": 1.1000,
                    "target_price": 1.0950,
                    "invalidation_price": 1.1030,
                    "horizon_min": 60,
                },
            )()
            thread_count = 6
            barrier = threading.Barrier(thread_count)

            def stale_precheck(*_args: object, **_kwargs: object) -> set[tuple]:
                barrier.wait(timeout=5)
                return set()

            results: list[int] = []
            errors: list[BaseException] = []

            def worker() -> None:
                try:
                    results.append(
                        record_directional_forecast(
                            forecast,
                            pair="EUR_USD",
                            current_price=None,
                            data_root=root,
                            regime_at_emission="TREND",
                            cycle_id="cycle-race",
                        )
                    )
                except BaseException as exc:
                    errors.append(exc)

            with mock.patch(
                "quant_rabbit.strategy.projection_ledger._existing_projection_keys",
                side_effect=stale_precheck,
            ):
                threads = [threading.Thread(target=worker) for _ in range(thread_count)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=5)

            self.assertEqual(errors, [])
            self.assertEqual(sum(results), 1)
            entries = load_ledger(root)
            self.assertEqual(len(entries), 1)


class VerifyTest(unittest.TestCase):
    def _setup(self, signal, ts_offset_min=30):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        emitted = datetime.now(timezone.utc) - timedelta(minutes=ts_offset_min)
        entry = LedgerEntry(
            timestamp_emitted_utc=emitted.isoformat().replace("+00:00", "Z"),
            pair="EUR_USD",
            signal_name=signal.get("name", "test"),
            direction=signal.get("direction", "UP"),
            lead_time_min=signal.get("lead_time_min", 10),
            confidence=0.7,
            entry_price=signal.get("entry_price", 1.1700),
            predicted_target_price=signal.get("target"),
            resolution_window_min=signal.get("window", 20),
            resolution_status="PENDING",
            predicted_invalidation_price=signal.get("invalidation"),
        )
        from quant_rabbit.strategy.projection_ledger import write_ledger
        write_ledger([entry], root)
        return tmp, root

    def test_directional_up_hit(self) -> None:
        tmp, root = self._setup({"direction": "UP", "lead_time_min": 5, "window": 10, "entry_price": 1.1700})
        with tmp:
            counts = verify_pending(
                root,
                quotes_by_pair={"EUR_USD": {"bid": 1.1715, "ask": 1.1716}},
                atr_pips_by_pair={"EUR_USD": 10.0},  # threshold = 5 pip
            )
            self.assertEqual(counts["HIT"], 1)
            entries = load_ledger(root)
            self.assertEqual(entries[0].resolution_status, "HIT")

    def test_directional_up_miss(self) -> None:
        tmp, root = self._setup({"direction": "UP", "lead_time_min": 5, "window": 10, "entry_price": 1.1700})
        with tmp:
            counts = verify_pending(
                root,
                quotes_by_pair={"EUR_USD": {"bid": 1.1700, "ask": 1.1701}},  # no move
                atr_pips_by_pair={"EUR_USD": 10.0},
            )
            self.assertEqual(counts["MISS"], 1)

    def test_liquidity_target_hit(self) -> None:
        tmp, root = self._setup({
            "direction": "UP", "lead_time_min": 5, "window": 10,
            "entry_price": 1.1700, "target": 1.1720,
        })
        with tmp:
            counts = verify_pending(
                root,
                quotes_by_pair={"EUR_USD": {"bid": 1.1719, "ask": 1.1721}},
                atr_pips_by_pair={"EUR_USD": 10.0},
            )
            # Mid 1.1720 reaches target → HIT
            self.assertEqual(counts["HIT"], 1)

    def test_liquidity_target_does_not_require_atr(self) -> None:
        tmp, root = self._setup({
            "direction": "UP", "lead_time_min": 5, "window": 10,
            "entry_price": 1.1700, "target": 1.1720,
        })
        with tmp:
            counts = verify_pending(
                root,
                quotes_by_pair={"EUR_USD": {"bid": 1.1719, "ask": 1.1721}},
                atr_pips_by_pair={},
            )
            self.assertEqual(counts["HIT"], 1)

    def test_liquidity_target_uses_window_high_not_current_quote(self) -> None:
        emitted = datetime.now(timezone.utc) - timedelta(minutes=30)
        tmp, root = self._setup({
            "direction": "UP", "lead_time_min": 5, "window": 10,
            "entry_price": 1.1700, "target": 1.1720,
        })
        with tmp:
            from quant_rabbit.strategy.projection_ledger import write_ledger
            entries = load_ledger(root)
            entries[0].timestamp_emitted_utc = emitted.isoformat().replace("+00:00", "Z")
            write_ledger(entries, root)
            counts = verify_pending(
                root,
                quotes_by_pair={"EUR_USD": {"bid": 1.1701, "ask": 1.1702}},
                atr_pips_by_pair={"EUR_USD": 10.0},
                candles_by_pair={
                    "EUR_USD": [
                        {
                            "timestamp": (emitted + timedelta(minutes=1)).isoformat(),
                            "high": 1.1721,
                            "low": 1.1698,
                            "close": 1.1702,
                        }
                    ]
                },
            )
            self.assertEqual(counts["HIT"], 1)
            self.assertEqual(load_ledger(root)[0].resolution_status, "HIT")

    def test_liquidity_sweep_high_uses_signal_name_not_entry_direction(self) -> None:
        tmp, root = self._setup({
            "name": "liquidity_sweep_high",
            "direction": "DOWN",
            "lead_time_min": 5,
            "window": 10,
            "entry_price": 1.1700,
            "target": 1.1720,
        })
        with tmp:
            counts = verify_pending(
                root,
                quotes_by_pair={"EUR_USD": {"bid": 1.1719, "ask": 1.1721}},
                atr_pips_by_pair={},
            )
            self.assertEqual(counts["HIT"], 1)
            entries = load_ledger(root)
            self.assertEqual(entries[0].resolution_status, "HIT")
            self.assertIn("sweep-high target", entries[0].resolution_evidence)

    def test_liquidity_sweep_low_uses_signal_name_not_entry_direction(self) -> None:
        tmp, root = self._setup({
            "name": "liquidity_sweep_low",
            "direction": "UP",
            "lead_time_min": 5,
            "window": 10,
            "entry_price": 1.1700,
            "target": 1.1680,
        })
        with tmp:
            counts = verify_pending(
                root,
                quotes_by_pair={"EUR_USD": {"bid": 1.1679, "ask": 1.1681}},
                atr_pips_by_pair={},
            )
            self.assertEqual(counts["HIT"], 1)
            entries = load_ledger(root)
            self.assertEqual(entries[0].resolution_status, "HIT")
            self.assertIn("sweep-low target", entries[0].resolution_evidence)

    def test_candle_mode_times_out_when_window_truth_missing(self) -> None:
        tmp, root = self._setup({
            "direction": "UP", "lead_time_min": 5, "window": 10,
            "entry_price": 1.1700,
        })
        with tmp:
            counts = verify_pending(
                root,
                quotes_by_pair={"EUR_USD": {"bid": 1.1800, "ask": 1.1801}},
                atr_pips_by_pair={"EUR_USD": 10.0},
                candles_by_pair={},
            )
            self.assertEqual(counts["TIMEOUT"], 1)
            self.assertEqual(load_ledger(root)[0].resolution_status, "TIMEOUT")

    def test_directional_forecast_invalidation_before_target_is_miss(self) -> None:
        emitted = datetime.now(timezone.utc) - timedelta(minutes=30)
        tmp, root = self._setup({
            "name": "directional_forecast",
            "direction": "UP",
            "lead_time_min": 5,
            "window": 10,
            "entry_price": 1.1000,
            "target": 1.1020,
            "invalidation": 1.0990,
        })
        with tmp:
            from quant_rabbit.strategy.projection_ledger import write_ledger
            entries = load_ledger(root)
            entries[0].timestamp_emitted_utc = emitted.isoformat().replace("+00:00", "Z")
            write_ledger(entries, root)

            counts = verify_pending(
                root,
                candles_by_pair={
                    "EUR_USD": [
                        {
                            "timestamp": (emitted + timedelta(minutes=1)).isoformat(),
                            "high": 1.1005,
                            "low": 1.0988,
                            "close": 1.0992,
                        },
                        {
                            "timestamp": (emitted + timedelta(minutes=2)).isoformat(),
                            "high": 1.1022,
                            "low": 1.0992,
                            "close": 1.1015,
                        },
                    ]
                },
            )

            self.assertEqual(counts["MISS"], 1)
            entry = load_ledger(root)[0]
            self.assertEqual(entry.resolution_status, "MISS")
            self.assertIn("invalidation 1.09900 touched before target 1.10200", entry.resolution_evidence)

    def test_directional_forecast_target_before_invalidation_is_hit(self) -> None:
        emitted = datetime.now(timezone.utc) - timedelta(minutes=30)
        tmp, root = self._setup({
            "name": "directional_forecast",
            "direction": "DOWN",
            "lead_time_min": 5,
            "window": 10,
            "entry_price": 1.1000,
            "target": 1.0980,
            "invalidation": 1.1010,
        })
        with tmp:
            from quant_rabbit.strategy.projection_ledger import write_ledger
            entries = load_ledger(root)
            entries[0].timestamp_emitted_utc = emitted.isoformat().replace("+00:00", "Z")
            write_ledger(entries, root)

            counts = verify_pending(
                root,
                candles_by_pair={
                    "EUR_USD": [
                        {
                            "timestamp": (emitted + timedelta(minutes=1)).isoformat(),
                            "high": 1.1003,
                            "low": 1.0978,
                            "close": 1.0985,
                        },
                        {
                            "timestamp": (emitted + timedelta(minutes=2)).isoformat(),
                            "high": 1.1012,
                            "low": 1.0983,
                            "close": 1.1008,
                        },
                    ]
                },
            )

            self.assertEqual(counts["HIT"], 1)
            entry = load_ledger(root)[0]
            self.assertEqual(entry.resolution_status, "HIT")
            self.assertIn("target 1.09800 touched before invalidation 1.10100", entry.resolution_evidence)

    def test_directional_forecast_same_candle_target_and_invalidation_is_miss(self) -> None:
        emitted = datetime.now(timezone.utc) - timedelta(minutes=30)
        tmp, root = self._setup({
            "name": "directional_forecast",
            "direction": "UP",
            "lead_time_min": 5,
            "window": 10,
            "entry_price": 1.1000,
            "target": 1.1020,
            "invalidation": 1.0990,
        })
        with tmp:
            from quant_rabbit.strategy.projection_ledger import write_ledger
            entries = load_ledger(root)
            entries[0].timestamp_emitted_utc = emitted.isoformat().replace("+00:00", "Z")
            write_ledger(entries, root)

            counts = verify_pending(
                root,
                candles_by_pair={
                    "EUR_USD": [
                        {
                            "timestamp": (emitted + timedelta(minutes=1)).isoformat(),
                            "high": 1.1022,
                            "low": 1.0988,
                            "close": 1.1005,
                        }
                    ]
                },
            )

            self.assertEqual(counts["MISS"], 1)
            evidence = load_ledger(root)[0].resolution_evidence
            self.assertIn("price candle touched", evidence)
            self.assertIn("ordering ambiguous", evidence)

    def test_directional_forecast_prefers_m1_over_coarser_ambiguous_fallback(self) -> None:
        emitted = datetime.now(timezone.utc) - timedelta(minutes=30)
        tmp, root = self._setup({
            "name": "directional_forecast",
            "direction": "UP",
            "lead_time_min": 5,
            "window": 10,
            "entry_price": 1.1000,
            "target": 1.1020,
            "invalidation": 1.0990,
        })
        with tmp:
            from quant_rabbit.strategy.projection_ledger import write_ledger
            entries = load_ledger(root)
            entries[0].timestamp_emitted_utc = emitted.isoformat().replace("+00:00", "Z")
            write_ledger(entries, root)

            counts = verify_pending(
                root,
                candles_by_pair={
                    "EUR_USD": {
                        "M1": [
                            {
                                "timestamp": (emitted + timedelta(minutes=1)).isoformat(),
                                "high": 1.1022,
                                "low": 1.1002,
                                "close": 1.1018,
                            }
                        ],
                        "M5": [
                            {
                                "timestamp": (emitted + timedelta(minutes=1)).isoformat(),
                                "high": 1.1024,
                                "low": 1.0988,
                                "close": 1.1004,
                            }
                        ],
                    }
                },
            )

            self.assertEqual(counts["HIT"], 1)
            evidence = load_ledger(root)[0].resolution_evidence
            self.assertIn("target 1.10200 touched before invalidation 1.09900", evidence)
            self.assertNotIn("ordering ambiguous", evidence)

    def test_directional_forecast_uses_m5_when_m1_window_missing(self) -> None:
        emitted = datetime.now(timezone.utc) - timedelta(minutes=30)
        tmp, root = self._setup({
            "name": "directional_forecast",
            "direction": "DOWN",
            "lead_time_min": 5,
            "window": 10,
            "entry_price": 1.1000,
            "target": 1.0980,
            "invalidation": 1.1010,
        })
        with tmp:
            from quant_rabbit.strategy.projection_ledger import write_ledger
            entries = load_ledger(root)
            entries[0].timestamp_emitted_utc = emitted.isoformat().replace("+00:00", "Z")
            write_ledger(entries, root)

            counts = verify_pending(
                root,
                candles_by_pair={
                    "EUR_USD": {
                        "M1": [
                            {
                                "timestamp": (emitted + timedelta(minutes=20)).isoformat(),
                                "high": 1.1012,
                                "low": 1.0978,
                                "close": 1.0990,
                            }
                        ],
                        "M5": [
                            {
                                "timestamp": (emitted + timedelta(minutes=5)).isoformat(),
                                "high": 1.1005,
                                "low": 1.0978,
                                "close": 1.0986,
                            }
                        ],
                    }
                },
            )

            self.assertEqual(counts["HIT"], 1)
            self.assertIn("target 1.09800 touched before invalidation 1.10100", load_ledger(root)[0].resolution_evidence)

    def test_pending_within_window(self) -> None:
        # Emitted just now with 60-min window → still pending
        tmp, root = self._setup({"window": 60}, ts_offset_min=10)
        with tmp:
            counts = verify_pending(
                root,
                quotes_by_pair={"EUR_USD": {"bid": 1.0, "ask": 1.0}},
            )
            self.assertEqual(counts["PENDING"], 1)

    def test_verify_pending_preserves_append_started_mid_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            emitted = datetime.now(timezone.utc) - timedelta(minutes=30)
            from quant_rabbit.strategy import projection_ledger as ledger_mod

            ledger_mod.write_ledger(
                [
                    LedgerEntry(
                        timestamp_emitted_utc=emitted.isoformat().replace("+00:00", "Z"),
                        pair="EUR_USD",
                        signal_name="directional_probe",
                        direction="UP",
                        lead_time_min=5,
                        confidence=0.7,
                        entry_price=1.1700,
                        predicted_target_price=None,
                        resolution_window_min=10,
                        resolution_status="PENDING",
                        cycle_id="verify-cycle",
                    )
                ],
                root,
            )

            verify_inside = threading.Event()
            release_verify = threading.Event()
            append_entered_critical = threading.Event()
            errors: list[BaseException] = []
            original_quote_point_path = ledger_mod._quote_point_path
            original_existing_keys = ledger_mod._existing_projection_keys_from_handle

            def paused_quote_point_path(entry, *, quotes_by_pair):
                verify_inside.set()
                if not release_verify.wait(timeout=5):
                    raise AssertionError("verification did not resume")
                return original_quote_point_path(entry, quotes_by_pair=quotes_by_pair)

            def observed_existing_keys(handle, *, cycle_id, pair):
                if cycle_id == "append-cycle":
                    append_entered_critical.set()
                return original_existing_keys(handle, cycle_id=cycle_id, pair=pair)

            def run_verify() -> None:
                try:
                    verify_pending(
                        root,
                        quotes_by_pair={"EUR_USD": {"bid": 1.1715, "ask": 1.1715}},
                        atr_pips_by_pair={"EUR_USD": 10.0},
                        now=emitted + timedelta(minutes=20),
                    )
                except BaseException as exc:
                    errors.append(exc)

            def run_append() -> None:
                try:
                    record_projections(
                        [_Sig("liquidity_sweep_low", "UP", 15, 0.9, 12.0, "M5 equal-lows at 1.16800 (4.0pip down)")],
                        pair="EUR_USD",
                        current_price=1.16840,
                        data_root=root,
                        cycle_id="append-cycle",
                    )
                except BaseException as exc:
                    errors.append(exc)

            with mock.patch.object(ledger_mod, "_quote_point_path", side_effect=paused_quote_point_path), \
                 mock.patch.object(ledger_mod, "_existing_projection_keys_from_handle", side_effect=observed_existing_keys):
                verify_thread = threading.Thread(target=run_verify)
                verify_thread.start()
                self.assertTrue(verify_inside.wait(timeout=5))

                append_thread = threading.Thread(target=run_append)
                append_thread.start()
                self.assertFalse(append_entered_critical.wait(timeout=0.2))

                release_verify.set()
                verify_thread.join(timeout=5)
                append_thread.join(timeout=5)

            self.assertFalse(verify_thread.is_alive())
            self.assertFalse(append_thread.is_alive())
            self.assertEqual(errors, [])
            self.assertTrue(append_entered_critical.is_set())

            entries = load_ledger(root)
            self.assertEqual(len(entries), 2)
            by_cycle = {entry.cycle_id: entry for entry in entries}
            self.assertEqual(by_cycle["verify-cycle"].resolution_status, "HIT")
            self.assertEqual(by_cycle["append-cycle"].resolution_status, "PENDING")


class HitRatesTest(unittest.TestCase):
    def test_compute_hit_rates_per_pair_regime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger
            entries = []
            # 3 HITs and 1 MISS for bb_squeeze on EUR_USD in TREND regime
            for status in ["HIT", "HIT", "HIT", "MISS"]:
                entries.append(LedgerEntry(
                    timestamp_emitted_utc="2026-05-14T00:00:00Z",
                    pair="EUR_USD", signal_name="bb_squeeze", direction="EITHER",
                    lead_time_min=10, confidence=0.7,
                    entry_price=1.0, predicted_target_price=None,
                    resolution_window_min=20, resolution_status=status,
                    regime_at_emission="TREND",
                ))
            write_ledger(entries, root)
            hr = compute_hit_rates(root)
            self.assertIn("bb_squeeze", hr)
            # New schema: "<pair>:<regime>" keying
            self.assertIn("EUR_USD:TREND", hr["bb_squeeze"])
            self.assertEqual(hr["bb_squeeze"]["EUR_USD:TREND"]["samples"], 4)
            self.assertAlmostEqual(hr["bb_squeeze"]["EUR_USD:TREND"]["hit_rate"], 0.75)
            # Aggregate buckets also present
            self.assertIn("EUR_USD:_all_regimes", hr["bb_squeeze"])
            self.assertIn("_all_pairs:_all_regimes", hr["bb_squeeze"])

    def test_compute_hit_rates_dedupes_historical_cycle_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            duplicate_hit = LedgerEntry(
                timestamp_emitted_utc="2026-05-14T00:00:00Z",
                pair="EUR_USD",
                signal_name="liquidity_sweep_low",
                direction="UP",
                lead_time_min=15,
                confidence=0.9,
                entry_price=1.1684,
                predicted_target_price=1.1680,
                resolution_window_min=30,
                resolution_status="HIT",
                regime_at_emission="RANGE",
                cycle_id="cycle-race",
            )
            entries = [
                duplicate_hit,
                duplicate_hit,
                LedgerEntry(
                    timestamp_emitted_utc="2026-05-14T00:30:00Z",
                    pair="EUR_USD",
                    signal_name="liquidity_sweep_low",
                    direction="UP",
                    lead_time_min=15,
                    confidence=0.9,
                    entry_price=1.1685,
                    predicted_target_price=1.1681,
                    resolution_window_min=30,
                    resolution_status="MISS",
                    regime_at_emission="RANGE",
                    cycle_id="cycle-next",
                ),
            ]
            write_ledger(entries, root)

            hr = compute_hit_rates(root)

            bucket = hr["liquidity_sweep_low"]["EUR_USD:RANGE"]
            self.assertEqual(bucket["samples"], 2)
            self.assertAlmostEqual(bucket["hit_rate"], 0.5)

    def test_regime_segmented_separates_by_regime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger
            entries = []
            # TREND: 4 HITs
            for _ in range(4):
                entries.append(LedgerEntry(
                    timestamp_emitted_utc="2026-05-14T00:00:00Z",
                    pair="EUR_USD", signal_name="sig_x", direction="UP",
                    lead_time_min=10, confidence=0.7,
                    entry_price=1.0, predicted_target_price=None,
                    resolution_window_min=20, resolution_status="HIT",
                    regime_at_emission="TREND",
                ))
            # RANGE: 4 MISSes
            for _ in range(4):
                entries.append(LedgerEntry(
                    timestamp_emitted_utc="2026-05-14T00:00:00Z",
                    pair="EUR_USD", signal_name="sig_x", direction="UP",
                    lead_time_min=10, confidence=0.7,
                    entry_price=1.0, predicted_target_price=None,
                    resolution_window_min=20, resolution_status="MISS",
                    regime_at_emission="RANGE",
                ))
            write_ledger(entries, root)
            hr = compute_hit_rates(root)
            self.assertAlmostEqual(hr["sig_x"]["EUR_USD:TREND"]["hit_rate"], 1.0)
            self.assertAlmostEqual(hr["sig_x"]["EUR_USD:RANGE"]["hit_rate"], 0.0)
            self.assertAlmostEqual(hr["sig_x"]["EUR_USD:_all_regimes"]["hit_rate"], 0.5)

    def test_directional_forecast_hit_rates_are_split_by_direction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger
            entries = []
            for status, direction in [("MISS", "UP"), ("HIT", "DOWN")]:
                entries.append(LedgerEntry(
                    timestamp_emitted_utc="2026-05-14T00:00:00Z",
                    pair="EUR_USD", signal_name="directional_forecast", direction=direction,
                    lead_time_min=60, confidence=0.7,
                    entry_price=1.0, predicted_target_price=None,
                    resolution_window_min=60, resolution_status=status,
                    regime_at_emission="TREND",
                ))
            write_ledger(entries, root)

            hr = compute_hit_rates(root)

            self.assertAlmostEqual(hr["directional_forecast"]["EUR_USD:TREND"]["hit_rate"], 0.5)
            self.assertAlmostEqual(hr["directional_forecast_up"]["EUR_USD:TREND"]["hit_rate"], 0.0)
            self.assertAlmostEqual(hr["directional_forecast_down"]["EUR_USD:TREND"]["hit_rate"], 1.0)

    def test_legacy_directional_forecast_without_invalidation_is_not_calibration_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger
            write_ledger([
                LedgerEntry(
                    timestamp_emitted_utc="2026-05-14T00:00:00Z",
                    pair="EUR_USD", signal_name="directional_forecast", direction="UP",
                    lead_time_min=60, confidence=0.7,
                    entry_price=1.0, predicted_target_price=1.01,
                    resolution_window_min=60, resolution_status="HIT",
                    regime_at_emission="TREND",
                )
            ], root)

            self.assertEqual(compute_hit_rates(root), {})

    def test_detector_hit_rates_are_split_by_direction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger
            entries = []
            for status, direction in [("MISS", "UP"), ("HIT", "DOWN")]:
                entries.append(LedgerEntry(
                    timestamp_emitted_utc="2026-05-14T00:00:00Z",
                    pair="EUR_USD",
                    signal_name="liquidity_sweep_high",
                    direction=direction,
                    lead_time_min=10, confidence=0.7,
                    entry_price=1.0, predicted_target_price=None,
                    resolution_window_min=20, resolution_status=status,
                    regime_at_emission="TREND",
                ))
            write_ledger(entries, root)

            hr = compute_hit_rates(root)

            self.assertAlmostEqual(
                hr["liquidity_sweep_high"]["EUR_USD:TREND"]["hit_rate"], 0.5,
            )
            self.assertAlmostEqual(
                hr["liquidity_sweep_high_up"]["EUR_USD:TREND"]["hit_rate"], 0.0,
            )
            self.assertAlmostEqual(
                hr["liquidity_sweep_high_down"]["EUR_USD:TREND"]["hit_rate"], 1.0,
            )


class CalibrationTest(unittest.TestCase):
    def test_high_hit_rate_boosts_confidence(self) -> None:
        # 100% hit but only 20 samples — Bayesian pessimism applies
        # mu = 21/22 ≈ 0.955, sigma ≈ 0.044, lower bound ≈ 0.90
        # So mult should be > 1.0 but not at CAP
        hr = {"sig_x": {
            "EUR_USD:TREND": {"hit_rate": 1.0, "samples": 20},
            "_all_pairs:_all_regimes": {"hit_rate": 1.0, "samples": 20},
        }}
        mult = confidence_calibration("sig_x", "EUR_USD", hit_rates=hr, regime="TREND")
        self.assertGreater(mult, 1.2)
        self.assertLessEqual(mult, CONFIDENCE_MAX_MULTIPLIER)

    def test_high_hit_rate_with_large_samples_approaches_max(self) -> None:
        # 100% with 100 samples → Bayesian bound much tighter
        hr = {"sig_x": {
            "EUR_USD:TREND": {"hit_rate": 1.0, "samples": 100},
        }}
        mult = confidence_calibration("sig_x", "EUR_USD", hit_rates=hr, regime="TREND")
        self.assertGreater(mult, 1.35)  # close to CAP

    def test_zero_hit_rate_dampens_confidence(self) -> None:
        hr = {"sig_x": {
            "EUR_USD:_all_regimes": {"hit_rate": 0.0, "samples": 100},
        }}
        mult = confidence_calibration("sig_x", "EUR_USD", hit_rates=hr)
        # 100 misses → posterior mean ≈ 0.01, lower bound ≈ 0
        # Should dampen significantly
        self.assertLess(mult, 0.5)

    def test_few_samples_returns_1_0(self) -> None:
        hr = {"sig_x": {"EUR_USD:_all_regimes": {"hit_rate": 0.5, "samples": 3}}}
        mult = confidence_calibration("sig_x", "EUR_USD", hit_rates=hr)
        self.assertEqual(mult, 1.0)

    def test_bayesian_pessimism_small_perfect_sample(self) -> None:
        """3 HITs / 3 trials → Bayesian posterior mean is 0.8, NOT 1.0.
        Lower bound even lower → mild boost only. This is the
        statistically-correct improvement over linear interp."""
        hr = {"sig_x": {"EUR_USD:_all_regimes": {"hit_rate": 1.0, "samples": 10}}}
        mult = confidence_calibration("sig_x", "EUR_USD", hit_rates=hr)
        # With 10 samples all HIT, mu ≈ 0.917, sigma ≈ 0.078, lower ≈ 0.82
        self.assertGreater(mult, 1.0)
        self.assertLess(mult, CONFIDENCE_MAX_MULTIPLIER)

    def test_unknown_signal_returns_1_0(self) -> None:
        hr = {}
        mult = confidence_calibration("unknown", "EUR_USD", hit_rates=hr)
        self.assertEqual(mult, 1.0)

    def test_regime_specific_overrides_all_regimes(self) -> None:
        """Same signal x pair: TREND regime hit_rate 0.9 should win
        over _all_regimes 0.3 when both have ≥10 samples and regime is TREND."""
        hr = {"sig_x": {
            "EUR_USD:TREND": {"hit_rate": 0.9, "samples": 15},
            "EUR_USD:_all_regimes": {"hit_rate": 0.3, "samples": 50},
        }}
        mult = confidence_calibration("sig_x", "EUR_USD", hit_rates=hr, regime="TREND")
        self.assertGreater(mult, 1.0)
        # And in RANGE, the _all_regimes 0.3 → < 1.0
        mult_range = confidence_calibration("sig_x", "EUR_USD", hit_rates=hr, regime="RANGE")
        self.assertLess(mult_range, 1.0)

    def test_falls_back_through_hierarchy(self) -> None:
        """With no pair:regime specific data, falls back to pair:_all_regimes."""
        hr = {"sig_x": {
            "EUR_USD:_all_regimes": {"hit_rate": 0.8, "samples": 15},
        }}
        mult = confidence_calibration("sig_x", "EUR_USD", hit_rates=hr, regime="TREND")
        self.assertGreater(mult, 1.0)

    def test_select_calibration_signal_name_prefers_directional_bucket(self) -> None:
        hr = {
            "sig_x": {"EUR_USD:TREND": {"hit_rate": 0.8, "samples": 100}},
            "sig_x_up": {"EUR_USD:TREND": {"hit_rate": 0.0, "samples": 100}},
        }

        selected = select_calibration_signal_name(
            "sig_x", "UP", "EUR_USD", hit_rates=hr, regime="TREND",
        )

        self.assertEqual(selected, "sig_x_up")

    def test_select_calibration_signal_name_keeps_base_for_thin_directional_sample(self) -> None:
        hr = {
            "sig_x": {"EUR_USD:TREND": {"hit_rate": 0.8, "samples": 100}},
            "sig_x_up": {"EUR_USD:TREND": {"hit_rate": 0.0, "samples": 3}},
        }

        selected = select_calibration_signal_name(
            "sig_x", "UP", "EUR_USD", hit_rates=hr, regime="TREND",
        )

        self.assertEqual(selected, "sig_x")


class SetupGradeTest(unittest.TestCase):
    def test_grade_a_strong_confluence_no_news(self) -> None:
        self.assertEqual(setup_grade(aligned_signal_count=4, has_news_block=False,
                                       confluence_score=27), "A")

    def test_grade_d_when_news_blocks(self) -> None:
        self.assertEqual(setup_grade(aligned_signal_count=10, has_news_block=True,
                                       confluence_score=50), "D")

    def test_grade_b_medium(self) -> None:
        self.assertEqual(setup_grade(aligned_signal_count=3, has_news_block=False,
                                       confluence_score=18), "B")

    def test_grade_c_minimal_confluence(self) -> None:
        self.assertEqual(setup_grade(aligned_signal_count=2, has_news_block=False,
                                       confluence_score=8), "C")


if __name__ == "__main__":
    unittest.main()
