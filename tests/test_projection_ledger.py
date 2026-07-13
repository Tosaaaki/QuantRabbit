"""Unit tests for strategy/projection_ledger.py."""

from __future__ import annotations

import os
import json
import tempfile
import threading
import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

from quant_rabbit.strategy.forecast_technical_context import build_forecast_technical_context

from quant_rabbit.strategy.projection_ledger import (
    CONFIDENCE_MAX_MULTIPLIER,
    CONFIDENCE_MIN_MULTIPLIER,
    DIRECTIONAL_FORECAST_MAX_RESOLUTION_WINDOW_MIN,
    IMMEDIATE_PROJECTION_RESOLUTION_WINDOW_MIN,
    LedgerEntry,
    compute_hit_rates,
    confidence_calibration,
    load_ledger,
    record_directional_forecast,
    record_projections,
    retryable_truth_timeout_pairs,
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


def _range_emission_context(
    *,
    pair: str = "EUR_USD",
    current_price: float = 1.1050,
    h1_atr_pips: object = 20.0,
) -> dict[str, object]:
    return build_forecast_technical_context(
        {
            "confluence": {
                "dominant_regime": "RANGE",
                "price_percentile_24h": 0.5,
            },
            "views": [
                {
                    "granularity": "M5",
                    "regime_reading": {"state": "RANGE", "atr_percentile": 50},
                    "indicators": {"atr_pips": 5.0},
                },
                {
                    "granularity": "M15",
                    "regime_reading": {"state": "RANGE", "atr_percentile": 50},
                    "structure": {
                        "structure_events": [
                            {"kind": "BOS_UP", "index": 5, "close_confirmed": True}
                        ]
                    },
                },
                {
                    "granularity": "H1",
                    "regime_reading": {"state": "RANGE", "atr_percentile": 50},
                    "indicators": {"atr_pips": h1_atr_pips},
                },
            ],
        },
        pair=pair,
        current_price=current_price,
        spread_pips=0.5,
    )


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

    def test_append_recorders_skip_nonobject_json_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "projection_ledger.jsonl"
            path.write_text("[]\nnull\n1\n{malformed-json\n", encoding="utf-8")
            sig = _Sig("bb_squeeze", "EITHER", 15, 0.7, 8.0)
            forecast = SimpleNamespace(
                direction="UP",
                confidence=0.24,
                raw_confidence=0.80,
                calibration_multiplier=0.30,
                current_price=1.1000,
                target_price=1.1020,
                invalidation_price=1.0990,
                horizon_min=60,
            )

            self.assertEqual(
                record_projections(
                    [sig],
                    pair="EUR_USD",
                    current_price=1.1000,
                    data_root=root,
                    cycle_id="nonobject-projection",
                ),
                1,
            )
            self.assertEqual(
                record_directional_forecast(
                    forecast,
                    pair="EUR_USD",
                    current_price=None,
                    data_root=root,
                    cycle_id="nonobject-directional",
                ),
                1,
            )
            self.assertEqual(len(load_ledger(root)), 2)

    def test_record_directional_forecast_normalises_naive_now_to_utc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            forecast = SimpleNamespace(
                direction="UP",
                confidence=0.24,
                raw_confidence=0.80,
                calibration_multiplier=0.30,
                current_price=1.1000,
                target_price=1.1020,
                invalidation_price=1.0990,
                horizon_min=60,
            )

            self.assertEqual(
                record_directional_forecast(
                    forecast,
                    pair="EUR_USD",
                    current_price=None,
                    data_root=root,
                    cycle_id="naive-now",
                    now=datetime(2026, 7, 13, 12, 34, 56),
                ),
                1,
            )

            self.assertEqual(
                load_ledger(root)[0].timestamp_emitted_utc,
                "2026-07-13T12:34:56Z",
            )

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

    def test_record_projections_caches_same_cycle_pair_key_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sig = _Sig("liquidity_sweep_low", "UP", 15, 0.9, 12.0, "M5 equal-lows at 1.16800 (4.0pip down)")

            from quant_rabbit.strategy import projection_ledger as ledger_mod

            ledger_mod._PROJECTION_KEY_CACHE.clear()
            original_scan = ledger_mod._scan_existing_projection_keys_from_handle
            scan_calls = 0

            def observed_scan(*args: object, **kwargs: object) -> set[tuple]:
                nonlocal scan_calls
                scan_calls += 1
                return original_scan(*args, **kwargs)

            with mock.patch.object(ledger_mod, "_scan_existing_projection_keys_from_handle", side_effect=observed_scan):
                first = record_projections(
                    [sig],
                    pair="EUR_USD",
                    current_price=1.16840,
                    data_root=root,
                    cycle_id="cycle-cache",
                )
                second = record_projections(
                    [sig],
                    pair="EUR_USD",
                    current_price=1.16840,
                    data_root=root,
                    cycle_id="cycle-cache",
                )

            self.assertEqual(first, 1)
            self.assertEqual(second, 0)
            self.assertEqual(scan_calls, 1)
            self.assertEqual(len(load_ledger(root)), 1)

    def test_record_projections_caches_one_key_scan_across_pairs_in_same_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eur_sig = _Sig(
                "liquidity_sweep_low",
                "UP",
                15,
                0.9,
                12.0,
                "M5 equal-lows at 1.16800 (4.0pip down)",
            )
            jpy_sig = _Sig(
                "liquidity_sweep_high",
                "DOWN",
                15,
                0.8,
                10.0,
                "M5 equal-highs at 158.400 (4.0pip up)",
            )

            from quant_rabbit.strategy import projection_ledger as ledger_mod

            ledger_mod._PROJECTION_KEY_CACHE.clear()
            original_scan = ledger_mod._scan_existing_projection_keys_from_handle
            scan_calls = 0

            def observed_scan(*args: object, **kwargs: object) -> set[tuple]:
                nonlocal scan_calls
                scan_calls += 1
                return original_scan(*args, **kwargs)

            with mock.patch.object(ledger_mod, "_scan_existing_projection_keys_from_handle", side_effect=observed_scan):
                first = record_projections(
                    [eur_sig],
                    pair="EUR_USD",
                    current_price=1.16840,
                    data_root=root,
                    cycle_id="cycle-cache-all-pairs",
                )
                second = record_projections(
                    [jpy_sig],
                    pair="USD_JPY",
                    current_price=158.350,
                    data_root=root,
                    cycle_id="cycle-cache-all-pairs",
                )
                duplicate = record_projections(
                    [eur_sig],
                    pair="EUR_USD",
                    current_price=1.16840,
                    data_root=root,
                    cycle_id="cycle-cache-all-pairs",
                )

            self.assertEqual((first, second, duplicate), (1, 1, 0))
            self.assertEqual(scan_calls, 1)
            self.assertEqual(len(load_ledger(root)), 2)

    def test_projection_key_cache_invalidates_when_ledger_inode_is_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sig = _Sig(
                "liquidity_sweep_low",
                "UP",
                15,
                0.9,
                12.0,
                "M5 equal-lows at 1.16800 (4.0pip down)",
            )
            cycle_id = "cycle-inode-replacement"

            from quant_rabbit.strategy import projection_ledger as ledger_mod

            ledger_mod._PROJECTION_KEY_CACHE.clear()
            self.assertEqual(
                record_projections(
                    [sig],
                    pair="EUR_USD",
                    current_price=1.16840,
                    data_root=root,
                    cycle_id=cycle_id,
                ),
                1,
            )
            path = root / ledger_mod.LEDGER_FILENAME
            original_stat = path.stat()
            original_text = path.read_text()
            replacement_text = original_text.replace("liquidity_sweep_low", "liquidity_sweep_alt")
            self.assertEqual(len(replacement_text), len(original_text))
            replacement = root / "replacement.jsonl"
            replacement.write_text(replacement_text)
            os.utime(replacement, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            os.replace(replacement, path)

            self.assertEqual(path.stat().st_size, original_stat.st_size)
            self.assertEqual(path.stat().st_mtime_ns, original_stat.st_mtime_ns)
            self.assertEqual(
                record_projections(
                    [sig],
                    pair="EUR_USD",
                    current_price=1.16840,
                    data_root=root,
                    cycle_id=cycle_id,
                ),
                1,
            )
            self.assertEqual(len(load_ledger(root)), 2)

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
                    "raw_confidence": 0.81,
                    "calibration_multiplier": 0.901234,
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
            self.assertEqual(entries[0].confidence, 0.73)
            self.assertEqual(entries[0].raw_confidence, 0.81)
            self.assertEqual(entries[0].calibration_multiplier, 0.901234)

    def test_record_directional_forecast_missing_raw_schema_stays_explicitly_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            forecast = type(
                "Forecast",
                (),
                {
                    "direction": "UP",
                    "confidence": 0.99,
                    "current_price": 1.1000,
                    "target_price": 1.1020,
                    "invalidation_price": 1.0990,
                    "horizon_min": 60,
                },
            )()

            self.assertEqual(
                record_directional_forecast(
                    forecast,
                    pair="EUR_USD",
                    current_price=None,
                    data_root=root,
                    regime_at_emission="TREND",
                    cycle_id="missing-raw-schema",
                ),
                1,
            )
            payload = json.loads((root / "projection_ledger.jsonl").read_text())
            self.assertIn("raw_confidence", payload)
            self.assertIn("calibration_multiplier", payload)
            self.assertIsNone(payload["raw_confidence"])
            self.assertIsNone(payload["calibration_multiplier"])

            entries = load_ledger(root)
            entries[0].resolution_status = "HIT"
            entries[0].resolution_evidence = "target touched before invalidation"
            from quant_rabbit.strategy.projection_ledger import write_ledger

            write_ledger(entries, root)
            self.assertNotIn("directional_forecast_up", compute_hit_rates(root))

    def test_directional_forecast_context_survives_ledger_resolution_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = build_forecast_technical_context(
                {
                    "confluence": {
                        "dominant_regime": "TREND_UP",
                        "price_percentile_24h": 0.7,
                    },
                    "views": [
                        {
                            "granularity": "M5",
                            "regime_reading": {"state": "TREND_STRONG", "atr_percentile": 80},
                            "indicators": {"atr_pips": 2.0},
                            "structure": {
                                "structure_events": [
                                    {"kind": "BOS_UP", "index": 5, "close_confirmed": True}
                                ]
                            },
                        },
                        {
                            "granularity": "M15",
                            "regime_reading": {"state": "TREND_WEAK", "atr_percentile": 50},
                            "structure": {
                                "structure_events": [
                                    {"kind": "BOS_UP", "index": 4, "close_confirmed": True}
                                ]
                            },
                        },
                    ],
                },
                pair="EUR_USD",
                current_price=1.1,
                spread_pips=0.5,
            )
            emitted = datetime(2026, 7, 13, 0, 0, tzinfo=timezone.utc)
            forecast = SimpleNamespace(
                direction="UP",
                confidence=0.7,
                current_price=1.1,
                target_price=1.101,
                invalidation_price=1.099,
                horizon_min=60,
                technical_context_v1=context,
            )
            self.assertEqual(
                record_directional_forecast(
                    forecast,
                    pair="EUR_USD",
                    current_price=1.1,
                    data_root=root,
                    cycle_id="context-cycle",
                    now=emitted,
                ),
                1,
            )

            verify_pending(
                root,
                quotes_by_pair={"EUR_USD": {"bid": 1.1012, "ask": 1.1013}},
                atr_pips_by_pair={"EUR_USD": 2.0},
                now=emitted + timedelta(minutes=61),
            )

            entry = load_ledger(root)[0]
            self.assertEqual(entry.technical_context_v1, context)
            raw = json.loads((root / "projection_ledger.jsonl").read_text())
            self.assertEqual(raw["technical_context_v1"]["context_sha256"], context["context_sha256"])

    def test_record_range_forecast_feeds_calibration_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            forecast = type(
                "Forecast",
                (),
                {
                    "direction": "RANGE",
                    "confidence": 0.67,
                    "current_price": 1.1050,
                    "target_price": 1.1120,
                    "invalidation_price": 1.0980,
                    "range_low_price": 1.1000,
                    "range_high_price": 1.1100,
                    "range_width_pips": 100.0,
                    "horizon_min": 120,
                },
            )()

            written = record_directional_forecast(
                forecast,
                pair="EUR_USD",
                current_price=None,
                data_root=root,
                regime_at_emission="RANGE",
                cycle_id="range-cycle",
            )

            self.assertEqual(written, 1)
            entries = load_ledger(root)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].signal_name, "directional_forecast")
            self.assertEqual(entries[0].direction, "RANGE")
            self.assertIsNone(entries[0].predicted_target_price)
            self.assertIsNone(entries[0].predicted_invalidation_price)
            self.assertEqual(entries[0].predicted_range_low_price, 1.1000)
            self.assertEqual(entries[0].predicted_range_high_price, 1.1100)
            self.assertEqual(entries[0].pre_emission_range_pips, 100.0)

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
    def setUp(self) -> None:
        self._market_open_patch = mock.patch(
            "quant_rabbit.strategy.projection_ledger.projection_telemetry_market_open",
            return_value=True,
        )
        self._market_open_patch.start()

    def tearDown(self) -> None:
        self._market_open_patch.stop()

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

    def _mark_new_schema_directional(
        self,
        root: Path,
        *,
        emitted: datetime,
        direction: str = "UP",
        window_min: float = 10.0,
        target: float | None = 1.1020,
        invalidation: float | None = 1.0990,
        range_low: float | None = None,
        range_high: float | None = None,
        technical_context_v1: dict[str, object] | None = None,
    ) -> None:
        from quant_rabbit.strategy.projection_ledger import write_ledger

        entry = load_ledger(root)[0]
        entry.timestamp_emitted_utc = emitted.isoformat().replace("+00:00", "Z")
        entry.signal_name = "directional_forecast"
        entry.direction = direction
        entry.lead_time_min = window_min
        entry.resolution_window_min = window_min
        entry.confidence = 0.24
        entry.raw_confidence = 0.80
        entry.calibration_multiplier = 0.30
        entry.entry_price = 1.1000 if direction != "RANGE" else 1.1050
        entry.predicted_target_price = target if direction != "RANGE" else None
        entry.predicted_invalidation_price = invalidation if direction != "RANGE" else None
        entry.predicted_range_low_price = range_low
        entry.predicted_range_high_price = range_high
        entry.pre_emission_range_pips = (
            (range_high - range_low) * 10_000
            if range_low is not None and range_high is not None
            else None
        )
        entry.technical_context_v1 = technical_context_v1
        write_ledger([entry], root)

    @staticmethod
    def _closed_candles(
        emitted: datetime,
        *,
        count: int,
        step_min: int = 1,
        high: float = 1.1010,
        low: float = 1.0995,
        close: float = 1.1002,
    ) -> list[dict[str, object]]:
        return [
            {
                "timestamp": (emitted + timedelta(minutes=step_min * index)).isoformat(),
                "high": high,
                "low": low,
                "close": close,
                "complete": True,
            }
            for index in range(count)
        ]

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

    def test_unexpired_pending_verification_does_not_rewrite_ledger(self) -> None:
        tmp, root = self._setup(
            {"direction": "UP", "lead_time_min": 10, "window": 60, "entry_price": 1.1700},
            ts_offset_min=1,
        )
        with tmp:
            from quant_rabbit.strategy import projection_ledger as ledger_mod

            path = root / ledger_mod.LEDGER_FILENAME
            before = path.read_bytes()
            with mock.patch.object(
                ledger_mod,
                "_write_ledger_to_handle",
                wraps=ledger_mod._write_ledger_to_handle,
            ) as write_ledger:
                counts = verify_pending(
                    root,
                    quotes_by_pair={"EUR_USD": {"bid": 1.1700, "ask": 1.1701}},
                    atr_pips_by_pair={"EUR_USD": 10.0},
                )

            self.assertEqual(counts["PENDING"], 1)
            write_ledger.assert_not_called()
            self.assertEqual(path.read_bytes(), before)

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

    def test_new_directional_quote_only_never_scores_target_miss(self) -> None:
        emitted = datetime.now(timezone.utc) - timedelta(minutes=30)
        tmp, root = self._setup(
            {
                "name": "directional_forecast",
                "direction": "UP",
                "window": 10,
                "entry_price": 1.1000,
                "target": 1.1020,
                "invalidation": 1.0990,
            }
        )
        with tmp:
            self._mark_new_schema_directional(root, emitted=emitted)

            counts = verify_pending(
                root,
                quotes_by_pair={"EUR_USD": {"bid": 1.1001, "ask": 1.1002}},
                now=emitted + timedelta(minutes=20),
            )

            entry = load_ledger(root)[0]
            self.assertEqual(counts["TIMEOUT"], 1)
            self.assertEqual(counts["MISS"], 0)
            self.assertEqual(entry.resolution_status, "TIMEOUT")
            self.assertIn("incomplete closed candle truth", entry.resolution_evidence)
            self.assertNotIn("directional_forecast_up", compute_hit_rates(root))

    def test_new_directional_partial_candle_windows_are_retryable_not_scored(self) -> None:
        emitted = datetime.now(timezone.utc) - timedelta(minutes=30)
        complete = self._closed_candles(emitted, count=10)
        variants = {
            "single": complete[:1],
            "leading_gap": complete[1:],
            "trailing_gap": complete[:-1],
            "internal_gap": complete[:5] + complete[6:],
            "explicit_incomplete": [
                {**item, "complete": False if index == 5 else True}
                for index, item in enumerate(complete)
            ],
            "high_below_low": [
                {**item, "high": 1.0990 if index == 5 else item["high"]}
                for index, item in enumerate(complete)
            ],
            "close_above_high": [
                {**item, "close": 1.1020 if index == 5 else item["close"]}
                for index, item in enumerate(complete)
            ],
            "close_below_low": [
                {**item, "close": 1.0990 if index == 5 else item["close"]}
                for index, item in enumerate(complete)
            ],
            "boolean_high": [
                {**item, "high": True if index == 5 else item["high"]}
                for index, item in enumerate(complete)
            ],
            "boolean_low": [
                {**item, "low": True if index == 5 else item["low"]}
                for index, item in enumerate(complete)
            ],
            "boolean_close": [
                {**item, "close": True if index == 5 else item["close"]}
                for index, item in enumerate(complete)
            ],
        }
        for label, candles in variants.items():
            with self.subTest(label=label):
                tmp, root = self._setup(
                    {
                        "name": "directional_forecast",
                        "direction": "UP",
                        "window": 10,
                        "entry_price": 1.1000,
                        "target": 1.1020,
                        "invalidation": 1.0990,
                    }
                )
                with tmp:
                    self._mark_new_schema_directional(root, emitted=emitted)
                    counts = verify_pending(
                        root,
                        candles_by_pair={"EUR_USD": {"M1": candles}},
                        now=emitted + timedelta(minutes=20),
                    )
                    entry = load_ledger(root)[0]

                self.assertEqual(counts["TIMEOUT"], 1)
                self.assertEqual(counts["HIT"], 0)
                self.assertEqual(counts["MISS"], 0)
                self.assertIn("incomplete closed candle truth", entry.resolution_evidence)
                self.assertNotIn("directional_forecast_up", compute_hit_rates(root))

    def test_new_directional_noniterable_candle_bucket_fails_closed(self) -> None:
        emitted = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
        for bucket in (7, True):
            with self.subTest(bucket=bucket):
                tmp, root = self._setup(
                    {
                        "name": "directional_forecast",
                        "direction": "UP",
                        "window": 10,
                        "entry_price": 1.1000,
                        "target": 1.1020,
                        "invalidation": 1.0990,
                    }
                )
                with tmp:
                    self._mark_new_schema_directional(root, emitted=emitted)
                    counts = verify_pending(
                        root,
                        candles_by_pair={"EUR_USD": {"M1": bucket}},
                        now=emitted + timedelta(minutes=20),
                    )
                    entry = load_ledger(root)[0]

                self.assertEqual(counts["TIMEOUT"], 1)
                self.assertIn("incomplete closed candle truth", entry.resolution_evidence)
                self.assertNotIn("directional_forecast_up", compute_hit_rates(root))

    def test_new_directional_complete_m1_scores_target_invalidation_and_no_touch(self) -> None:
        emitted = datetime.now(timezone.utc) - timedelta(minutes=30)
        cases = (
            ("target", "HIT"),
            ("invalidation", "MISS"),
            ("no_touch", "TIMEOUT"),
        )
        for label, expected_status in cases:
            with self.subTest(label=label):
                candles = self._closed_candles(emitted, count=10)
                if label == "target":
                    candles[3]["high"] = 1.1022
                elif label == "invalidation":
                    candles[3]["low"] = 1.0988
                tmp, root = self._setup(
                    {
                        "name": "directional_forecast",
                        "direction": "UP",
                        "window": 10,
                        "entry_price": 1.1000,
                        "target": 1.1020,
                        "invalidation": 1.0990,
                    }
                )
                with tmp:
                    self._mark_new_schema_directional(root, emitted=emitted)
                    counts = verify_pending(
                        root,
                        candles_by_pair={"EUR_USD": {"M1": candles}},
                        now=emitted + timedelta(minutes=20),
                    )
                    entry = load_ledger(root)[0]

                self.assertEqual(counts[expected_status], 1)
                self.assertEqual(entry.resolution_status, expected_status)
                if label == "no_touch":
                    self.assertIn("both untouched", entry.resolution_evidence)

    def test_new_directional_partial_m1_falls_back_to_complete_m5(self) -> None:
        emitted = datetime.now(timezone.utc) - timedelta(minutes=30)
        partial_m1 = self._closed_candles(emitted, count=5)
        partial_m1[2]["low"] = 1.0988
        complete_m5 = self._closed_candles(
            emitted,
            count=2,
            step_min=5,
            high=1.1010,
            low=1.0995,
        )
        complete_m5[1]["high"] = 1.1022
        tmp, root = self._setup(
            {
                "name": "directional_forecast",
                "direction": "UP",
                "window": 10,
                "entry_price": 1.1000,
                "target": 1.1020,
                "invalidation": 1.0990,
            }
        )
        with tmp:
            self._mark_new_schema_directional(root, emitted=emitted)
            counts = verify_pending(
                root,
                candles_by_pair={
                    "EUR_USD": {"M1": partial_m1, "M5": complete_m5}
                },
                now=emitted + timedelta(minutes=20),
            )
            entry = load_ledger(root)[0]

        self.assertEqual(counts["HIT"], 1)
        self.assertEqual(entry.resolution_status, "HIT")
        self.assertNotIn("invalidation", entry.resolution_evidence.split(" touched before ")[0])

    def test_new_directional_rejects_coarse_bars_straddling_truth_boundaries(self) -> None:
        emitted = datetime(2026, 7, 13, 12, 4, tzinfo=timezone.utc)
        partial_m1 = self._closed_candles(emitted, count=5)
        variants = {
            "M5": [
                {
                    "timestamp": (datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc) + timedelta(minutes=5 * index)).isoformat(),
                    "high": 1.1025,
                    "low": 1.0995,
                    "close": 1.1002,
                    "complete": True,
                }
                for index in range(3)
            ],
            "H1": [
                {
                    "timestamp": datetime(2026, 7, 13, hour, 0, tzinfo=timezone.utc).isoformat(),
                    "high": 1.1025,
                    "low": 1.0995,
                    "close": 1.1002,
                    "complete": True,
                }
                for hour in (12, 13)
            ],
        }
        for label, coarse in variants.items():
            with self.subTest(label=label):
                window_min = 10 if label == "M5" else 60
                tmp, root = self._setup(
                    {
                        "name": "directional_forecast",
                        "direction": "UP",
                        "window": window_min,
                        "entry_price": 1.1000,
                        "target": 1.1020,
                        "invalidation": 1.0990,
                    }
                )
                with tmp:
                    self._mark_new_schema_directional(
                        root,
                        emitted=emitted,
                        window_min=window_min,
                    )
                    counts = verify_pending(
                        root,
                        candles_by_pair={
                            "EUR_USD": {"M1": partial_m1, label: coarse},
                        },
                        now=emitted + timedelta(minutes=130),
                    )
                    entry = load_ledger(root)[0]

                self.assertEqual(counts["TIMEOUT"], 1)
                self.assertEqual(counts["HIT"], 0)
                self.assertEqual(counts["MISS"], 0)
                self.assertEqual(entry.resolution_status, "TIMEOUT")
                self.assertIn("incomplete closed candle truth", entry.resolution_evidence)

    def test_new_directional_accepts_second_stamped_complete_m1_truth(self) -> None:
        emitted = datetime(2026, 7, 13, 12, 4, 24, tzinfo=timezone.utc)
        first_open = emitted.replace(second=0)
        candles = self._closed_candles(first_open, count=11)
        candles[4]["high"] = 1.1025
        tmp, root = self._setup(
            {
                "name": "directional_forecast",
                "direction": "UP",
                "window": 10,
                "entry_price": 1.1000,
                "target": 1.1020,
                "invalidation": 1.0990,
            }
        )
        with tmp:
            self._mark_new_schema_directional(root, emitted=emitted)
            counts = verify_pending(
                root,
                candles_by_pair={"EUR_USD": {"M1": candles}},
                now=emitted + timedelta(minutes=20),
            )

        self.assertEqual(counts["HIT"], 1)

    def test_new_directional_accepts_coarse_truth_with_subminute_endpoint_overhang(self) -> None:
        emitted = datetime(2026, 7, 13, 12, 0, 30, tzinfo=timezone.utc)
        candles = [
            {
                "timestamp": emitted.replace(second=0).isoformat(),
                "high": 1.1025,
                "low": 1.0995,
                "close": 1.1002,
                "complete": True,
            }
        ]
        tmp, root = self._setup(
            {
                "name": "directional_forecast",
                "direction": "UP",
                "window": 4,
                "entry_price": 1.1000,
                "target": 1.1020,
                "invalidation": 1.0990,
            }
        )
        with tmp:
            self._mark_new_schema_directional(root, emitted=emitted, window_min=4)
            counts = verify_pending(
                root,
                candles_by_pair={"EUR_USD": {"M5": candles}},
                now=emitted + timedelta(minutes=10),
            )

        self.assertEqual(counts["HIT"], 1)

    def test_new_directional_rejects_duplicate_and_overlapping_candles(self) -> None:
        emitted = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
        complete = self._closed_candles(emitted, count=10)
        variants = {
            "duplicate": complete + [{**complete[4]}],
            "overlap": complete[:5]
            + [{**complete[5], "timestamp": (emitted + timedelta(minutes=4, seconds=30)).isoformat()}]
            + complete[5:],
        }
        for label, candles in variants.items():
            with self.subTest(label=label):
                tmp, root = self._setup(
                    {
                        "name": "directional_forecast",
                        "direction": "UP",
                        "window": 10,
                        "entry_price": 1.1000,
                        "target": 1.1020,
                        "invalidation": 1.0990,
                    }
                )
                with tmp:
                    self._mark_new_schema_directional(root, emitted=emitted)
                    counts = verify_pending(
                        root,
                        candles_by_pair={"EUR_USD": {"M1": candles}},
                        now=emitted + timedelta(minutes=20),
                    )
                    entry = load_ledger(root)[0]

                self.assertEqual(counts["TIMEOUT"], 1)
                self.assertIn("incomplete closed candle truth", entry.resolution_evidence)

    def test_new_directional_retry_resolves_only_after_complete_truth_arrives(self) -> None:
        emitted = datetime.now(timezone.utc) - timedelta(minutes=30)
        partial = self._closed_candles(emitted, count=4)
        complete = self._closed_candles(emitted, count=10)
        complete[4]["high"] = 1.1022
        tmp, root = self._setup(
            {
                "name": "directional_forecast",
                "direction": "UP",
                "window": 10,
                "entry_price": 1.1000,
                "target": 1.1020,
                "invalidation": 1.0990,
            }
        )
        with tmp:
            self._mark_new_schema_directional(root, emitted=emitted)
            first = verify_pending(
                root,
                candles_by_pair={"EUR_USD": {"M1": partial}},
                now=emitted + timedelta(minutes=20),
            )
            second = verify_pending(
                root,
                candles_by_pair={"EUR_USD": {"M1": complete}},
                now=emitted + timedelta(minutes=21),
            )
            entry = load_ledger(root)[0]

        self.assertEqual(first["TIMEOUT"], 1)
        self.assertEqual(second["HIT"], 1)
        self.assertEqual(entry.resolution_status, "HIT")

    def test_verify_pending_normalises_naive_now_to_utc(self) -> None:
        emitted = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
        candles = self._closed_candles(emitted, count=10)
        candles[4]["high"] = 1.1022
        tmp, root = self._setup(
            {
                "name": "directional_forecast",
                "direction": "UP",
                "window": 10,
                "entry_price": 1.1000,
                "target": 1.1020,
                "invalidation": 1.0990,
            }
        )
        with tmp:
            self._mark_new_schema_directional(root, emitted=emitted)
            counts = verify_pending(
                root,
                candles_by_pair={"EUR_USD": {"M1": candles}},
                now=datetime(2026, 7, 13, 12, 20),
            )

        self.assertEqual(counts["HIT"], 1)

    def test_new_directional_expiry_open_bar_is_outside_truth_window(self) -> None:
        emitted = datetime.now(timezone.utc) - timedelta(minutes=30)
        candles = self._closed_candles(emitted, count=11)
        candles[10]["high"] = 1.1025
        tmp, root = self._setup(
            {
                "name": "directional_forecast",
                "direction": "UP",
                "window": 10,
                "entry_price": 1.1000,
                "target": 1.1020,
                "invalidation": 1.0990,
            }
        )
        with tmp:
            self._mark_new_schema_directional(root, emitted=emitted)
            counts = verify_pending(
                root,
                candles_by_pair={"EUR_USD": {"M1": candles}},
                now=emitted + timedelta(minutes=20),
            )
            entry = load_ledger(root)[0]

        self.assertEqual(counts["TIMEOUT"], 1)
        self.assertEqual(entry.resolution_status, "TIMEOUT")
        self.assertIn("both untouched", entry.resolution_evidence)

    def test_new_directional_invalid_time_window_and_pair_fail_closed_not_retryable(self) -> None:
        now = datetime.now(timezone.utc)
        variants = (
            ("huge_window", "resolution window exceeds"),
            ("future_timestamp", "future emission timestamp"),
            ("invalid_timestamp", "invalid emission timestamp"),
            ("unsupported_pair", "unsupported or noncanonical pair"),
        )
        for label, evidence_fragment in variants:
            with self.subTest(label=label):
                tmp, root = self._setup(
                    {
                        "name": "directional_forecast",
                        "direction": "UP",
                        "window": 10,
                        "entry_price": 1.1000,
                        "target": 1.1020,
                        "invalidation": 1.0990,
                    }
                )
                with tmp:
                    self._mark_new_schema_directional(
                        root,
                        emitted=now - timedelta(minutes=30),
                    )
                    entry = load_ledger(root)[0]
                    if label == "huge_window":
                        entry.resolution_window_min = 1e300
                    elif label == "future_timestamp":
                        entry.timestamp_emitted_utc = (
                            now + timedelta(days=1)
                        ).isoformat()
                    elif label == "invalid_timestamp":
                        entry.timestamp_emitted_utc = "not-a-timestamp"
                    else:
                        entry.pair = "XAU_USD"
                    from quant_rabbit.strategy.projection_ledger import write_ledger

                    write_ledger([entry], root)
                    counts = verify_pending(
                        root,
                        candles_by_pair={},
                        now=now,
                    )
                    resolved = load_ledger(root)[0]

                self.assertEqual(counts["TIMEOUT"], 1)
                self.assertEqual(counts["PENDING"], 0)
                self.assertIn(evidence_fragment, resolved.resolution_evidence)
                self.assertEqual(retryable_truth_timeout_pairs([resolved]), set())

    def test_malicious_optional_numeric_and_candle_do_not_stop_other_verification(self) -> None:
        emitted = datetime.now(timezone.utc) - timedelta(minutes=30)
        now = emitted + timedelta(minutes=20)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "projection_ledger.jsonl"

            def payload(pair: str, cycle_id: str) -> dict[str, object]:
                return {
                    "timestamp_emitted_utc": emitted.isoformat(),
                    "pair": pair,
                    "signal_name": "directional_forecast",
                    "direction": "UP",
                    "lead_time_min": 10.0,
                    "confidence": 0.24,
                    "raw_confidence": 0.80,
                    "calibration_multiplier": 0.30,
                    "entry_price": 1.100,
                    "predicted_target_price": 1.102,
                    "predicted_invalidation_price": 1.099,
                    "resolution_window_min": 10.0,
                    "resolution_status": "PENDING",
                    "cycle_id": cycle_id,
                }

            bad_optional = {
                **payload("AUD_USD", "bad-optional-target"),
                "predicted_target_price": 10**400,
            }
            path.write_text(
                "\n".join(
                    json.dumps(item)
                    for item in (
                        payload("EUR_USD", "bad-candle"),
                        bad_optional,
                        payload("GBP_USD", "valid-candle"),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            bad_candles = self._closed_candles(emitted, count=10)
            bad_candles[5]["high"] = 10**400
            valid_candles = self._closed_candles(emitted, count=10)
            valid_candles[5]["high"] = 1.1022

            counts = verify_pending(
                root,
                candles_by_pair={
                    "EUR_USD": {"M1": bad_candles},
                    "AUD_USD": {"M1": self._closed_candles(emitted, count=10)},
                    "GBP_USD": {"M1": valid_candles},
                },
                now=now,
            )
            by_cycle = {entry.cycle_id: entry for entry in load_ledger(root)}

        self.assertEqual(counts["HIT"], 1)
        self.assertEqual(by_cycle["valid-candle"].resolution_status, "HIT")
        self.assertEqual(by_cycle["bad-candle"].resolution_status, "TIMEOUT")
        self.assertIn(
            "incomplete closed candle truth",
            by_cycle["bad-candle"].resolution_evidence,
        )
        self.assertEqual(by_cycle["bad-optional-target"].resolution_status, "TIMEOUT")

    def test_malformed_atr_is_local_timeout_and_does_not_stop_verification(self) -> None:
        emitted = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
        pairs = ("EUR_USD", "AUD_USD", "NZD_USD", "GBP_USD")
        atr_by_pair = {
            "EUR_USD": "bad",
            "AUD_USD": float("nan"),
            "NZD_USD": 10**400,
            "GBP_USD": 10.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            write_ledger(
                [
                    LedgerEntry(
                        timestamp_emitted_utc=emitted.isoformat(),
                        pair=pair,
                        signal_name="legacy_movement",
                        direction="UP",
                        lead_time_min=10,
                        confidence=0.7,
                        entry_price=1.1000,
                        predicted_target_price=None,
                        resolution_window_min=10,
                        resolution_status="PENDING",
                        cycle_id=f"atr-{pair}",
                    )
                    for pair in pairs
                ],
                root,
            )
            counts = verify_pending(
                root,
                quotes_by_pair={
                    pair: {"bid": 1.1009, "ask": 1.1011}
                    for pair in pairs
                },
                atr_pips_by_pair=atr_by_pair,
                now=emitted + timedelta(minutes=20),
            )
            statuses = {
                entry.pair: entry.resolution_status
                for entry in load_ledger(root)
            }

        self.assertEqual(counts["TIMEOUT"], 3)
        self.assertEqual(counts["HIT"], 1)
        self.assertEqual(statuses["GBP_USD"], "HIT")

    def test_new_range_requires_complete_truth_for_hold_and_break(self) -> None:
        emitted = datetime.now(timezone.utc) - timedelta(minutes=150)
        for label, high in (("hold", 1.1090), ("break", 1.1125)):
            with self.subTest(label=label):
                tmp, root = self._setup(
                    {
                        "name": "directional_forecast",
                        "direction": "RANGE",
                        "window": 120,
                        "entry_price": 1.1050,
                    }
                )
                with tmp:
                    self._mark_new_schema_directional(
                        root,
                        emitted=emitted,
                        direction="RANGE",
                        window_min=120,
                        target=None,
                        invalidation=None,
                        range_low=1.1000,
                        range_high=1.1100,
                        technical_context_v1=_range_emission_context(
                            current_price=1.1050,
                            h1_atr_pips=4.0,
                        ),
                    )
                    partial = [
                        {
                            "timestamp": (emitted + timedelta(minutes=30)).isoformat(),
                            "high": high,
                            "low": 1.1005,
                            "close": 1.1050,
                            "complete": True,
                        }
                    ]
                    counts = verify_pending(
                        root,
                        candles_by_pair={"EUR_USD": {"M1": partial}},
                        now=emitted + timedelta(minutes=140),
                    )
                    entry = load_ledger(root)[0]

                self.assertEqual(counts["TIMEOUT"], 1)
                self.assertEqual(entry.resolution_status, "TIMEOUT")
                self.assertIn("incomplete closed candle truth", entry.resolution_evidence)

    def test_new_range_complete_window_scores_hold_and_break(self) -> None:
        emitted = datetime.now(timezone.utc) - timedelta(minutes=150)
        for label, expected_status in (("hold", "HIT"), ("break", "MISS")):
            with self.subTest(label=label):
                candles = self._closed_candles(
                    emitted,
                    count=24,
                    step_min=5,
                    high=1.1090,
                    low=1.1005,
                    close=1.1050,
                )
                if label == "break":
                    candles[10]["high"] = 1.1125
                tmp, root = self._setup(
                    {
                        "name": "directional_forecast",
                        "direction": "RANGE",
                        "window": 120,
                        "entry_price": 1.1050,
                    }
                )
                with tmp:
                    self._mark_new_schema_directional(
                        root,
                        emitted=emitted,
                        direction="RANGE",
                        window_min=120,
                        target=None,
                        invalidation=None,
                        range_low=1.1000,
                        range_high=1.1100,
                        technical_context_v1=_range_emission_context(
                            current_price=1.1050,
                            h1_atr_pips=4.0,
                        ),
                    )
                    counts = verify_pending(
                        root,
                        atr_pips_by_pair={"EUR_USD": 4.0},
                        candles_by_pair={"EUR_USD": {"M5": candles}},
                        now=emitted + timedelta(minutes=140),
                    )
                    entry = load_ledger(root)[0]

                self.assertEqual(counts[expected_status], 1)
                self.assertEqual(entry.resolution_status, expected_status)

    def test_new_range_uses_only_emission_h1_atr_for_reproducible_outcome(self) -> None:
        emitted = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
        external_atr_variants = (None, 10.0, 20.0, 50.0, float("nan"))
        for external_atr in external_atr_variants:
            with self.subTest(external_atr=external_atr):
                candles = self._closed_candles(
                    emitted,
                    count=10,
                    high=1.1090,
                    low=1.1005,
                    close=1.1050,
                )
                # 15 pips above the emitted box: MISS with the immutable
                # emission H1 ATR=20 (10-pip tolerance), but the old verifier
                # flipped this to HIT when the later ATR map contained 50.
                candles[4]["high"] = 1.1115
                tmp, root = self._setup(
                    {
                        "name": "directional_forecast",
                        "direction": "RANGE",
                        "window": 10,
                        "entry_price": 1.1050,
                    }
                )
                with tmp:
                    self._mark_new_schema_directional(
                        root,
                        emitted=emitted,
                        direction="RANGE",
                        window_min=10,
                        target=None,
                        invalidation=None,
                        range_low=1.1000,
                        range_high=1.1100,
                        technical_context_v1=_range_emission_context(
                            current_price=1.1050,
                            h1_atr_pips=20.0,
                        ),
                    )
                    external = (
                        {}
                        if external_atr is None
                        else {"EUR_USD": external_atr}
                    )
                    counts = verify_pending(
                        root,
                        atr_pips_by_pair=external,
                        candles_by_pair={"EUR_USD": {"M1": candles}},
                        now=emitted + timedelta(minutes=20),
                    )
                    entry = load_ledger(root)[0]

                self.assertEqual(counts["MISS"], 1)
                self.assertEqual(entry.resolution_status, "MISS")
                self.assertIn("plus 10.0pip ATR tolerance", entry.resolution_evidence)

    def test_new_range_invalid_emission_atr_context_is_retryable_zero_learning(self) -> None:
        emitted = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
        valid_context = _range_emission_context(
            current_price=1.1050,
            h1_atr_pips=20.0,
        )
        invalid_parent = json.loads(json.dumps(valid_context))
        invalid_parent["identity"]["pair"] = "GBP_USD"
        contexts = {
            "missing": None,
            "nonfinite": _range_emission_context(
                current_price=1.1050,
                h1_atr_pips=float("nan"),
            ),
            "nonpositive": _range_emission_context(
                current_price=1.1050,
                h1_atr_pips=0.0,
            ),
            "invalid_parent": invalid_parent,
        }
        for label, context in contexts.items():
            with self.subTest(label=label):
                candles = self._closed_candles(
                    emitted,
                    count=10,
                    high=1.1090,
                    low=1.1005,
                    close=1.1050,
                )
                tmp, root = self._setup(
                    {
                        "name": "directional_forecast",
                        "direction": "RANGE",
                        "window": 10,
                        "entry_price": 1.1050,
                    }
                )
                with tmp:
                    self._mark_new_schema_directional(
                        root,
                        emitted=emitted,
                        direction="RANGE",
                        window_min=10,
                        target=None,
                        invalidation=None,
                        range_low=1.1000,
                        range_high=1.1100,
                        technical_context_v1=context,
                    )
                    counts = verify_pending(
                        root,
                        atr_pips_by_pair={"EUR_USD": 50.0},
                        candles_by_pair={"EUR_USD": {"M1": candles}},
                        now=emitted + timedelta(minutes=20),
                    )
                    entry = load_ledger(root)[0]
                    hit_rates = compute_hit_rates(root)

                self.assertEqual(counts["TIMEOUT"], 1)
                self.assertEqual(entry.resolution_status, "TIMEOUT")
                self.assertIn("range emission ATR truth unavailable", entry.resolution_evidence)
                self.assertEqual(retryable_truth_timeout_pairs([entry]), {"EUR_USD"})
                self.assertNotIn("directional_forecast_range", hit_rates)

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

    def test_directional_forecast_no_touch_is_timeout_not_miss(self) -> None:
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
                            "high": 1.1008,
                            "low": 1.0993,
                            "close": 1.1002,
                        },
                        {
                            "timestamp": (emitted + timedelta(minutes=4)).isoformat(),
                            "high": 1.1011,
                            "low": 1.0994,
                            "close": 1.1007,
                        },
                    ]
                },
            )

            self.assertEqual(counts["TIMEOUT"], 1)
            entry = load_ledger(root)[0]
            self.assertEqual(entry.resolution_status, "TIMEOUT")
            self.assertIn("both untouched in forecast window", entry.resolution_evidence)

    def test_range_forecast_box_hold_is_hit(self) -> None:
        emitted = datetime.now(timezone.utc) - timedelta(minutes=150)
        tmp, root = self._setup({
            "name": "directional_forecast",
            "direction": "RANGE",
            "lead_time_min": 120,
            "window": 120,
            "entry_price": 1.1050,
        })
        with tmp:
            from quant_rabbit.strategy.projection_ledger import write_ledger
            entries = load_ledger(root)
            entries[0].timestamp_emitted_utc = emitted.isoformat().replace("+00:00", "Z")
            entries[0].predicted_range_low_price = 1.1000
            entries[0].predicted_range_high_price = 1.1100
            entries[0].pre_emission_range_pips = 100.0
            write_ledger(entries, root)

            counts = verify_pending(
                root,
                atr_pips_by_pair={"EUR_USD": 4.0},
                candles_by_pair={
                    "EUR_USD": [
                        {
                            "timestamp": (emitted + timedelta(minutes=30)).isoformat(),
                            "high": 1.1098,
                            "low": 1.1003,
                            "close": 1.1055,
                        }
                    ]
                },
            )

            entry = load_ledger(root)[0]
            self.assertEqual(counts["HIT"], 1)
            self.assertEqual(entry.resolution_status, "HIT")
            self.assertIn("range held", entry.resolution_evidence)

    def test_range_forecast_box_break_is_miss(self) -> None:
        emitted = datetime.now(timezone.utc) - timedelta(minutes=150)
        tmp, root = self._setup({
            "name": "directional_forecast",
            "direction": "RANGE",
            "lead_time_min": 120,
            "window": 120,
            "entry_price": 1.1050,
        })
        with tmp:
            from quant_rabbit.strategy.projection_ledger import write_ledger
            entries = load_ledger(root)
            entries[0].timestamp_emitted_utc = emitted.isoformat().replace("+00:00", "Z")
            entries[0].predicted_range_low_price = 1.1000
            entries[0].predicted_range_high_price = 1.1100
            entries[0].pre_emission_range_pips = 100.0
            write_ledger(entries, root)

            counts = verify_pending(
                root,
                atr_pips_by_pair={"EUR_USD": 4.0},
                candles_by_pair={
                    "EUR_USD": [
                        {
                            "timestamp": (emitted + timedelta(minutes=30)).isoformat(),
                            "high": 1.1125,
                            "low": 1.1004,
                            "close": 1.1118,
                        }
                    ]
                },
            )

            entry = load_ledger(root)[0]
            self.assertEqual(counts["MISS"], 1)
            self.assertEqual(entry.resolution_status, "MISS")
            self.assertIn("range broke", entry.resolution_evidence)

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

    def test_truth_missing_timeout_is_retried_with_later_m5_candle_truth(self) -> None:
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
            entries[0].resolution_status = "TIMEOUT"
            entries[0].resolution_evidence = "no M1 candle truth for projection window"
            entries[0].resolved_at_utc = (emitted + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
            write_ledger(entries, root)

            counts = verify_pending(
                root,
                candles_by_pair={
                    "EUR_USD": {
                        "M1": [],
                        "M5": [
                            {
                                "timestamp": (emitted + timedelta(minutes=5)).isoformat(),
                                "high": 1.1004,
                                "low": 1.0978,
                                "close": 1.0984,
                            }
                        ],
                    }
                },
            )

            resolved = load_ledger(root)[0]

        self.assertEqual(counts["HIT"], 1)
        self.assertEqual(resolved.resolution_status, "HIT")
        self.assertIn("target 1.09800 touched before invalidation 1.10100", resolved.resolution_evidence)

    def test_closed_market_timeout_is_not_retried(self) -> None:
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
            entries[0].resolution_status = "TIMEOUT"
            entries[0].resolution_evidence = "market closed at projection emission; excluded from calibration"
            entries[0].resolved_at_utc = (emitted + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
            write_ledger(entries, root)

            counts = verify_pending(
                root,
                candles_by_pair={
                    "EUR_USD": [
                        {
                            "timestamp": (emitted + timedelta(minutes=1)).isoformat(),
                            "high": 1.1025,
                            "low": 1.1000,
                            "close": 1.1020,
                        }
                    ]
                },
            )

            resolved = load_ledger(root)[0]

        self.assertEqual(counts["HIT"], 0)
        self.assertEqual(counts["MISS"], 0)
        self.assertEqual(counts["TIMEOUT"], 0)
        self.assertEqual(resolved.resolution_status, "TIMEOUT")
        self.assertIn("market closed", resolved.resolution_evidence)

    def test_pending_within_window(self) -> None:
        # Emitted just now with 60-min window → still pending
        tmp, root = self._setup({"window": 60}, ts_offset_min=10)
        with tmp:
            counts = verify_pending(
                root,
                quotes_by_pair={"EUR_USD": {"bid": 1.0, "ask": 1.0}},
            )
            self.assertEqual(counts["PENDING"], 1)

    def test_closed_market_projection_times_out_without_calibration(self) -> None:
        self._market_open_patch.stop()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                emitted = datetime(2026, 6, 7, 14, 0, tzinfo=timezone.utc)
                entry = LedgerEntry(
                    timestamp_emitted_utc=emitted.isoformat().replace("+00:00", "Z"),
                    pair="EUR_USD",
                    signal_name="directional_forecast",
                    direction="UP",
                    lead_time_min=60,
                    confidence=0.8,
                    entry_price=1.1700,
                    predicted_target_price=1.1720,
                    predicted_invalidation_price=1.1690,
                    resolution_window_min=60,
                    resolution_status="PENDING",
                    cycle_id="closed-market-cycle",
                )
                from quant_rabbit.strategy.projection_ledger import write_ledger

                write_ledger([entry], root)

                counts = verify_pending(
                    root,
                    now=emitted + timedelta(minutes=10),
                    quotes_by_pair={"EUR_USD": {"bid": 1.1730, "ask": 1.1732}},
                    atr_pips_by_pair={"EUR_USD": 10.0},
                )

                resolved = load_ledger(root)[0]
        finally:
            self._market_open_patch.start()

        self.assertEqual(counts["TIMEOUT"], 1)
        self.assertEqual(resolved.resolution_status, "TIMEOUT")
        self.assertIn("market closed at projection emission", resolved.resolution_evidence)

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
    @staticmethod
    def _point_in_time_directional_trial(
        *,
        emitted_at: datetime,
        resolved_at: datetime,
        status: str,
        cycle_id: str,
        window_min: float = 60.0,
    ) -> LedgerEntry:
        return LedgerEntry(
            timestamp_emitted_utc=emitted_at.isoformat(),
            pair="EUR_USD",
            signal_name="directional_forecast",
            direction="UP",
            lead_time_min=60,
            confidence=0.24,
            raw_confidence=0.80,
            calibration_multiplier=0.30,
            entry_price=1.1000,
            predicted_target_price=1.1020,
            predicted_invalidation_price=1.0990,
            resolution_window_min=window_min,
            resolution_status=status,
            resolved_at_utc=resolved_at.isoformat(),
            resolution_evidence=(
                "target touched before invalidation"
                if status == "HIT"
                else "invalidation touched before target"
            ),
            regime_at_emission="TREND",
            cycle_id=cycle_id,
        )

    def test_compute_hit_rates_per_pair_regime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger
            entries = []
            # 3 HITs and 1 MISS for bb_squeeze on EUR_USD in TREND regime
            for index, status in enumerate(["HIT", "HIT", "HIT", "MISS"]):
                entries.append(LedgerEntry(
                    timestamp_emitted_utc=(
                        datetime(2026, 5, 14, tzinfo=timezone.utc)
                        + timedelta(minutes=index * 20)
                    ).isoformat(),
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

    def test_compute_hit_rates_tracks_timeouts_as_economic_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            entries = []
            for idx, status in enumerate(["HIT"] * 9 + ["MISS"] + ["TIMEOUT"] * 10):
                entries.append(LedgerEntry(
                    timestamp_emitted_utc=(
                        datetime(2026, 5, 14, tzinfo=timezone.utc)
                        + timedelta(minutes=idx * 20)
                    ).isoformat(),
                    pair="EUR_USD",
                    signal_name="session_expansion_ny",
                    direction="EITHER",
                    lead_time_min=10,
                    confidence=0.7,
                    entry_price=1.0,
                    predicted_target_price=None,
                    resolution_window_min=20,
                    resolution_status=status,
                    resolution_evidence=(
                        "window expired before tradable expansion"
                        if status == "TIMEOUT"
                        else f"resolved {status}"
                    ),
                    regime_at_emission="TREND",
                    cycle_id=f"cycle-{idx}",
                ))
            entries.append(LedgerEntry(
                timestamp_emitted_utc=(
                    datetime(2026, 5, 14, tzinfo=timezone.utc)
                    + timedelta(minutes=20 * 20)
                ).isoformat(),
                pair="EUR_USD",
                signal_name="session_expansion_ny",
                direction="EITHER",
                lead_time_min=10,
                confidence=0.7,
                entry_price=1.0,
                predicted_target_price=None,
                resolution_window_min=20,
                resolution_status="TIMEOUT",
                resolution_evidence="market closed at projection emission; excluded from calibration",
                regime_at_emission="TREND",
                cycle_id="market-closed",
            ))
            write_ledger(entries, root)

            hr = compute_hit_rates(root)
            bucket = hr["session_expansion_ny"]["EUR_USD:TREND"]

            self.assertEqual(bucket["samples"], 10)
            self.assertAlmostEqual(bucket["hit_rate"], 0.9)
            self.assertEqual(bucket["calibration_samples"], 20)
            self.assertEqual(bucket["economic_samples"], 20)
            self.assertAlmostEqual(bucket["economic_hit_rate"], 0.45)
            self.assertEqual(bucket["timeout_count"], 10)
            self.assertAlmostEqual(bucket["timeout_rate"], 0.5)

    def test_range_directional_forecast_gets_direction_specific_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            statuses = ["HIT", "HIT", "MISS", "HIT"]
            entries = [
                LedgerEntry(
                    timestamp_emitted_utc=(
                        datetime(2026, 6, 16, tzinfo=timezone.utc)
                        + timedelta(minutes=index * 120)
                    ).isoformat(),
                    pair="EUR_USD",
                    signal_name="directional_forecast",
                    direction="RANGE",
                    lead_time_min=120,
                    confidence=0.7,
                    entry_price=1.1050,
                    predicted_target_price=None,
                    predicted_invalidation_price=None,
                    predicted_range_low_price=1.1000,
                    predicted_range_high_price=1.1100,
                    pre_emission_range_pips=100.0,
                    resolution_window_min=120,
                    resolution_status=status,
                    regime_at_emission="RANGE",
                )
                for index, status in enumerate(statuses)
            ]
            write_ledger(entries, root)

            hit_rates = compute_hit_rates(root)

            self.assertIn("directional_forecast", hit_rates)
            self.assertIn("directional_forecast_range", hit_rates)
            self.assertEqual(
                hit_rates["directional_forecast_range"]["EUR_USD:RANGE"]["samples"],
                4,
            )
            self.assertEqual(
                hit_rates["directional_forecast_range"]["EUR_USD:RANGE"]["hit_rate"],
                0.75,
            )

    def test_compute_hit_rates_caches_until_ledger_stat_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy import projection_ledger as ledger_mod
            from quant_rabbit.strategy.projection_ledger import write_ledger

            def entry(status: str, cycle_id: str) -> LedgerEntry:
                minute = 0 if cycle_id == "cycle-1" else 20
                return LedgerEntry(
                    timestamp_emitted_utc=(
                        datetime(2026, 5, 14, tzinfo=timezone.utc)
                        + timedelta(minutes=minute)
                    ).isoformat(),
                    pair="EUR_USD",
                    signal_name="bb_squeeze",
                    direction="EITHER",
                    lead_time_min=10,
                    confidence=0.7,
                    entry_price=1.0,
                    predicted_target_price=None,
                    resolution_window_min=20,
                    resolution_status=status,
                    regime_at_emission="TREND",
                    cycle_id=cycle_id,
                )

            write_ledger([entry("HIT", "cycle-1")], root)
            ledger_mod._HIT_RATE_CACHE.clear()
            original_load = ledger_mod.load_ledger
            load_calls = 0

            def observed_load(*args: object, **kwargs: object) -> list[LedgerEntry]:
                nonlocal load_calls
                load_calls += 1
                return original_load(*args, **kwargs)

            with mock.patch.object(ledger_mod, "load_ledger", side_effect=observed_load):
                first = compute_hit_rates(root)
                first["bb_squeeze"]["EUR_USD:TREND"]["samples"] = 99
                second = compute_hit_rates(root)
                write_ledger([entry("HIT", "cycle-1"), entry("MISS", "cycle-2")], root)
                third = compute_hit_rates(root)

            self.assertEqual(load_calls, 2)
            self.assertEqual(second["bb_squeeze"]["EUR_USD:TREND"]["samples"], 1)
            self.assertEqual(third["bb_squeeze"]["EUR_USD:TREND"]["samples"], 2)
            self.assertAlmostEqual(third["bb_squeeze"]["EUR_USD:TREND"]["hit_rate"], 0.5)

    def test_as_of_excludes_future_emissions_before_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            as_of = datetime(2026, 7, 13, 12, tzinfo=timezone.utc)
            observed = self._point_in_time_directional_trial(
                emitted_at=as_of - timedelta(hours=2),
                resolved_at=as_of - timedelta(hours=1),
                status="MISS",
                cycle_id="observed-miss",
            )
            future = self._point_in_time_directional_trial(
                emitted_at=as_of + timedelta(minutes=1),
                resolved_at=as_of + timedelta(hours=1, minutes=1),
                status="HIT",
                cycle_id="future-hit",
            )
            write_ledger([observed, future], root)

            bucket = compute_hit_rates(root, as_of=as_of)[
                "directional_forecast_up"
            ]["EUR_USD:TREND"]

            self.assertEqual(bucket["samples"], 1)
            self.assertEqual(bucket["hit_rate"], 0.0)

    def test_as_of_requires_resolution_and_full_window_to_be_observable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            as_of = datetime(2026, 7, 13, 12, tzinfo=timezone.utc)
            base = as_of - timedelta(hours=4)
            eligible = self._point_in_time_directional_trial(
                emitted_at=base,
                resolved_at=base + timedelta(hours=1),
                status="MISS",
                cycle_id="eligible-miss",
            )
            resolved_in_future = self._point_in_time_directional_trial(
                emitted_at=base + timedelta(hours=1),
                resolved_at=as_of + timedelta(seconds=1),
                status="HIT",
                cycle_id="future-resolution-hit",
            )
            incomplete_window = self._point_in_time_directional_trial(
                emitted_at=as_of - timedelta(minutes=30),
                resolved_at=as_of - timedelta(minutes=1),
                status="HIT",
                cycle_id="incomplete-window-hit",
            )
            prematurely_resolved = self._point_in_time_directional_trial(
                emitted_at=base + timedelta(hours=2),
                resolved_at=base + timedelta(hours=2, minutes=30),
                status="HIT",
                cycle_id="premature-resolution-hit",
            )
            write_ledger(
                [
                    eligible,
                    resolved_in_future,
                    incomplete_window,
                    prematurely_resolved,
                ],
                root,
            )

            bucket = compute_hit_rates(root, as_of=as_of)[
                "directional_forecast_up"
            ]["EUR_USD:TREND"]

            self.assertEqual(bucket["samples"], 1)
            self.assertEqual(bucket["hit_rate"], 0.0)

    def test_as_of_boundary_recomputes_without_ledger_stat_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            emitted_at = datetime(2026, 7, 13, 10, tzinfo=timezone.utc)
            resolved_at = emitted_at + timedelta(hours=1)
            write_ledger(
                [
                    self._point_in_time_directional_trial(
                        emitted_at=emitted_at,
                        resolved_at=resolved_at,
                        status="HIT",
                        cycle_id="boundary-hit",
                    )
                ],
                root,
            )
            ledger_stat = (root / "projection_ledger.jsonl").stat()

            before = compute_hit_rates(
                root,
                as_of=resolved_at - timedelta(microseconds=1),
            )
            at_boundary = compute_hit_rates(root, as_of=resolved_at)
            after_stat = (root / "projection_ledger.jsonl").stat()

            self.assertNotIn("directional_forecast_up", before)
            self.assertEqual(
                at_boundary["directional_forecast_up"]["EUR_USD:TREND"]["samples"],
                1,
            )
            self.assertEqual(ledger_stat.st_size, after_stat.st_size)
            self.assertEqual(ledger_stat.st_mtime_ns, after_stat.st_mtime_ns)

    def test_strict_snapshot_reports_non_object_and_core_invalid_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "projection_ledger.jsonl").write_text(
                "[]\n{}\n",
                encoding="utf-8",
            )

            result = compute_hit_rates(
                root,
                as_of=datetime(2026, 7, 13, tzinfo=timezone.utc),
                include_parse_integrity=True,
            )

            self.assertIsInstance(result, tuple)
            hit_rates, parse_integrity = result
            self.assertEqual(hit_rates, {})
            self.assertEqual(parse_integrity["status"], "INVALID")
            self.assertEqual(parse_integrity["non_object_rows"], 1)
            self.assertEqual(parse_integrity["unloadable_object_rows"], 1)
            self.assertEqual(parse_integrity["invalid_nonblank_rows"], 2)

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
            for index in range(4):
                entries.append(LedgerEntry(
                    timestamp_emitted_utc=f"2026-05-14T0{index // 3}:{(index * 20) % 60:02d}:00Z",
                    pair="EUR_USD", signal_name="sig_x", direction="UP",
                    lead_time_min=10, confidence=0.7,
                    entry_price=1.0, predicted_target_price=None,
                    resolution_window_min=20, resolution_status="HIT",
                    regime_at_emission="TREND",
                ))
            # RANGE: 4 MISSes
            for index in range(4):
                minute = 80 + index * 20
                entries.append(LedgerEntry(
                    timestamp_emitted_utc=(
                        datetime(2026, 5, 14, tzinfo=timezone.utc)
                        + timedelta(minutes=minute)
                    ).isoformat(),
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

    def test_no_cycle_detector_rows_use_outcome_blind_non_overlap_trials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            base = datetime(2026, 5, 14, tzinfo=timezone.utc)

            def trial(*, second: int, status: str, regime: str) -> LedgerEntry:
                return LedgerEntry(
                    timestamp_emitted_utc=(base + timedelta(seconds=second)).isoformat(),
                    pair="EUR_USD",
                    signal_name="liquidity_sweep_low",
                    direction="DOWN",
                    lead_time_min=15,
                    confidence=0.9,
                    entry_price=1.17087,
                    predicted_target_price=1.17085,
                    resolution_window_min=30,
                    resolution_status=status,
                    regime_at_emission=regime,
                )

            entries = [
                trial(second=index, status="HIT", regime="TREND")
                for index in range(100)
            ]
            entries.append(
                trial(second=30 * 60, status="MISS", regime="UNCLEAR")
            )
            write_ledger(entries, root)

            hit_rates = compute_hit_rates(root)
            all_regimes = hit_rates["liquidity_sweep_low_down"][
                "EUR_USD:_all_regimes"
            ]

            self.assertEqual(all_regimes["samples"], 2)
            self.assertEqual(all_regimes["hit_rate"], 0.5)
            self.assertEqual(
                hit_rates["liquidity_sweep_low_down"]["EUR_USD:TREND"]["samples"],
                1,
            )
            self.assertEqual(
                hit_rates["liquidity_sweep_low_down"]["EUR_USD:UNCLEAR"]["samples"],
                1,
            )

    def test_pending_cycle_trial_owns_window_before_outcome_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            base = datetime(2026, 5, 14, tzinfo=timezone.utc)

            def trial(
                *,
                minute: int,
                status: str,
                regime: str,
                cycle_id: str,
            ) -> LedgerEntry:
                return LedgerEntry(
                    timestamp_emitted_utc=(base + timedelta(minutes=minute)).isoformat(),
                    pair="EUR_USD",
                    signal_name="liquidity_sweep_low",
                    direction="DOWN",
                    lead_time_min=15,
                    confidence=0.9,
                    entry_price=1.17087,
                    predicted_target_price=1.17085,
                    resolution_window_min=30,
                    resolution_status=status,
                    regime_at_emission=regime,
                    cycle_id=cycle_id,
                )

            write_ledger(
                [
                    trial(
                        minute=0,
                        status="PENDING",
                        regime="UNCLEAR",
                        cycle_id="pending-owner",
                    ),
                    trial(
                        minute=5,
                        status="HIT",
                        regime="TREND",
                        cycle_id="overlapping-hit",
                    ),
                    trial(
                        minute=30,
                        status="MISS",
                        regime="TREND",
                        cycle_id="boundary-miss",
                    ),
                ],
                root,
            )

            hit_rates = compute_hit_rates(root)

            self.assertNotIn("EUR_USD:UNCLEAR", hit_rates["liquidity_sweep_low_down"])
            self.assertEqual(
                hit_rates["liquidity_sweep_low_down"]["EUR_USD:TREND"]["samples"],
                1,
            )
            self.assertEqual(
                hit_rates["liquidity_sweep_low_down"]["EUR_USD:TREND"]["hit_rate"],
                0.0,
            )

    def test_cycle_rows_with_distinct_targets_share_one_non_overlap_clock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            base = datetime(2026, 5, 14, tzinfo=timezone.utc)

            def trial(
                *,
                minute: int,
                status: str,
                cycle_id: str,
                target: float,
            ) -> LedgerEntry:
                return LedgerEntry(
                    timestamp_emitted_utc=(base + timedelta(minutes=minute)).isoformat(),
                    pair="EUR_USD",
                    signal_name="liquidity_sweep_low",
                    direction="DOWN",
                    lead_time_min=15,
                    confidence=0.9,
                    entry_price=1.17087,
                    predicted_target_price=target,
                    resolution_window_min=30,
                    resolution_status=status,
                    regime_at_emission="TREND",
                    cycle_id=cycle_id,
                )

            write_ledger(
                [
                    trial(
                        minute=0,
                        status="HIT",
                        cycle_id="same-cycle",
                        target=1.17085,
                    ),
                    # Exact-cycle dedupe intentionally keeps a distinct target,
                    # but the shared truth window must still suppress it.
                    trial(
                        minute=0,
                        status="HIT",
                        cycle_id="same-cycle",
                        target=1.17080,
                    ),
                    trial(
                        minute=5,
                        status="HIT",
                        cycle_id="different-cycle-same-window",
                        target=1.17075,
                    ),
                    trial(
                        minute=30,
                        status="MISS",
                        cycle_id="boundary-cycle",
                        target=1.17070,
                    ),
                ],
                root,
            )

            bucket = compute_hit_rates(root)["liquidity_sweep_low_down"][
                "EUR_USD:TREND"
            ]
            self.assertEqual(bucket["samples"], 2)
            self.assertEqual(bucket["hit_rate"], 0.5)

    def test_generic_window_over_24_hours_remains_a_valid_trial_clock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            base = datetime(2026, 5, 14, tzinfo=timezone.utc)

            def trial(*, hour: int, status: str, cycle_id: str) -> LedgerEntry:
                return LedgerEntry(
                    timestamp_emitted_utc=(base + timedelta(hours=hour)).isoformat(),
                    pair="EUR_USD",
                    signal_name="macro_event_nowcast_consumption",
                    direction="UP",
                    lead_time_min=24 * 60,
                    confidence=0.8,
                    entry_price=1.17,
                    predicted_target_price=None,
                    resolution_window_min=48 * 60,
                    resolution_status=status,
                    regime_at_emission="TREND",
                    cycle_id=cycle_id,
                )

            write_ledger(
                [
                    trial(hour=0, status="HIT", cycle_id="long-window-owner"),
                    trial(hour=24, status="HIT", cycle_id="inside-long-window"),
                    trial(hour=48, status="MISS", cycle_id="long-window-boundary"),
                ],
                root,
            )

            bucket = compute_hit_rates(root)["macro_event_nowcast_consumption"][
                "EUR_USD:TREND"
            ]
            self.assertEqual(bucket["samples"], 2)
            self.assertEqual(bucket["hit_rate"], 0.5)

    def test_directional_forecast_hit_rates_are_split_by_direction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger
            entries = []
            for status, direction in [("MISS", "UP"), ("HIT", "DOWN")]:
                target = 1.01 if direction == "UP" else 0.99
                invalidation = 0.99 if direction == "UP" else 1.01
                entries.append(LedgerEntry(
                    timestamp_emitted_utc="2026-05-14T00:00:00Z",
                    pair="EUR_USD", signal_name="directional_forecast", direction=direction,
                    lead_time_min=60, confidence=0.7,
                    entry_price=1.0,
                    predicted_target_price=target,
                    predicted_invalidation_price=invalidation,
                    resolution_window_min=60, resolution_status=status,
                    regime_at_emission="TREND",
                ))
            write_ledger(entries, root)

            hr = compute_hit_rates(root)

            self.assertAlmostEqual(hr["directional_forecast"]["EUR_USD:TREND"]["hit_rate"], 0.5)
            self.assertAlmostEqual(hr["directional_forecast_up"]["EUR_USD:TREND"]["hit_rate"], 0.0)
            self.assertAlmostEqual(hr["directional_forecast_down"]["EUR_USD:TREND"]["hit_rate"], 1.0)

    def test_compute_hit_rates_tracks_directional_invalidation_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            entries = []
            for index, status in enumerate(["MISS", "MISS", "MISS", "HIT"]):
                evidence = (
                    "invalidation 1.09900 touched before target 1.10200"
                    if status == "MISS"
                    else "target 1.10200 touched before invalidation 1.09900"
                )
                entries.append(LedgerEntry(
                    timestamp_emitted_utc=(
                        datetime(2026, 6, 16, tzinfo=timezone.utc)
                        + timedelta(hours=index)
                    ).isoformat(),
                    pair="EUR_USD",
                    signal_name="directional_forecast",
                    direction="UP",
                    lead_time_min=60,
                    confidence=0.7,
                    entry_price=1.1,
                    predicted_target_price=1.102,
                    predicted_invalidation_price=1.099,
                    resolution_window_min=60,
                    resolution_status=status,
                    resolution_evidence=evidence,
                    regime_at_emission="TREND",
                    cycle_id=f"cycle-{index}",
                ))
            write_ledger(entries, root)

            hr = compute_hit_rates(root)

            bucket = hr["directional_forecast_up"]["EUR_USD:TREND"]
            self.assertEqual(bucket["samples"], 4)
            self.assertAlmostEqual(bucket["hit_rate"], 0.25)
            self.assertEqual(bucket["invalidation_first_count"], 3)
            self.assertAlmostEqual(bucket["invalidation_first_rate"], 0.75)

    def test_raw_confidence_keeps_calibration_learning_after_multiplier_dampening(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            base = datetime(2026, 7, 13, tzinfo=timezone.utc)
            entries = []
            for index in range(12):
                entries.append(
                    LedgerEntry(
                        timestamp_emitted_utc=(base + timedelta(hours=index)).isoformat(),
                        pair="EUR_USD",
                        signal_name="directional_forecast",
                        direction="UP",
                        lead_time_min=60,
                        confidence=0.24,
                        raw_confidence=0.80,
                        calibration_multiplier=0.30,
                        entry_price=1.1,
                        predicted_target_price=1.102,
                        predicted_invalidation_price=1.099,
                        resolution_window_min=60,
                        resolution_status="HIT",
                        resolution_evidence="target 1.10200 touched before invalidation 1.09900",
                        regime_at_emission="TREND",
                        cycle_id=f"raw-grade-{index}",
                    )
                )
            # A high calibrated number cannot admit genuinely weak raw evidence.
            entries.append(
                LedgerEntry(
                    timestamp_emitted_utc=(base + timedelta(hours=12)).isoformat(),
                    pair="EUR_USD",
                    signal_name="directional_forecast",
                    direction="UP",
                    lead_time_min=60,
                    confidence=0.30,
                    raw_confidence=0.20,
                    calibration_multiplier=1.50,
                    entry_price=1.1,
                    predicted_target_price=1.102,
                    predicted_invalidation_price=1.099,
                    resolution_window_min=60,
                    resolution_status="MISS",
                    resolution_evidence="invalidation 1.09900 touched before target 1.10200",
                    regime_at_emission="TREND",
                    cycle_id="low-raw",
                )
            )
            write_ledger(entries, root)

            bucket = compute_hit_rates(root)["directional_forecast_up"]["EUR_USD:TREND"]

            self.assertEqual(bucket["samples"], 12)
            self.assertEqual(bucket["calibration_samples"], 12)
            self.assertEqual(bucket["hit_rate"], 1.0)

    def test_raw_schema_directional_calibration_uses_pair_wide_non_overlap_clock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            base = datetime(2026, 7, 13, tzinfo=timezone.utc)

            def directional(
                *,
                minute: int,
                direction: str,
                status: str,
                cycle_id: str,
                pair: str = "EUR_USD",
            ) -> LedgerEntry:
                is_range = direction == "RANGE"
                return LedgerEntry(
                    timestamp_emitted_utc=(base + timedelta(minutes=minute)).isoformat(),
                    pair=pair,
                    signal_name="directional_forecast",
                    direction=direction,
                    lead_time_min=60,
                    confidence=0.25,
                    raw_confidence=0.80,
                    calibration_multiplier=0.3125,
                    entry_price=1.1,
                    predicted_target_price=None if is_range else (1.102 if direction == "UP" else 1.098),
                    predicted_invalidation_price=None if is_range else (1.099 if direction == "UP" else 1.101),
                    predicted_range_low_price=1.098 if is_range else None,
                    predicted_range_high_price=1.102 if is_range else None,
                    resolution_window_min=60,
                    resolution_status=status,
                    resolution_evidence=(
                        "range held"
                        if is_range
                        else (
                            "target touched before invalidation"
                            if status == "HIT"
                            else "invalidation touched before target"
                        )
                    ),
                    regime_at_emission="RANGE" if is_range else "TREND",
                    cycle_id=cycle_id,
                    technical_context_v1=(
                        _range_emission_context(
                            current_price=1.1,
                            h1_atr_pips=20.0,
                        )
                        if is_range
                        else None
                    ),
                )

            write_ledger(
                [
                    directional(minute=0, direction="RANGE", status="HIT", cycle_id="range-zero"),
                    # Shares the RANGE truth window and must not become a second trial.
                    directional(minute=5, direction="UP", status="HIT", cycle_id="up-overlap"),
                    # Same timestamp conflict is also one trial, never two.
                    directional(minute=60, direction="DOWN", status="MISS", cycle_id="down-boundary-a"),
                    directional(minute=60, direction="UP", status="HIT", cycle_id="down-boundary-b"),
                    # Exact prior truth-end is an admissible new boundary.
                    directional(minute=120, direction="UP", status="HIT", cycle_id="up-next-boundary"),
                    # Pair clocks are independent.
                    directional(
                        minute=5,
                        direction="UP",
                        status="HIT",
                        cycle_id="gbp-independent",
                        pair="GBP_USD",
                    ),
                ],
                root,
            )

            hit_rates = compute_hit_rates(root)

            self.assertEqual(hit_rates["directional_forecast"]["EUR_USD:_all_regimes"]["samples"], 3)
            self.assertEqual(hit_rates["directional_forecast_range"]["EUR_USD:RANGE"]["samples"], 1)
            self.assertEqual(hit_rates["directional_forecast_down"]["EUR_USD:TREND"]["samples"], 1)
            self.assertEqual(hit_rates["directional_forecast_up"]["EUR_USD:TREND"]["samples"], 1)
            self.assertEqual(hit_rates["directional_forecast_up"]["GBP_USD:TREND"]["samples"], 1)

    def test_pending_first_raw_trial_suppresses_overlapping_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            base = datetime(2026, 7, 13, tzinfo=timezone.utc)

            def trial(*, minute: int, status: str, cycle_id: str) -> LedgerEntry:
                return LedgerEntry(
                    timestamp_emitted_utc=(base + timedelta(minutes=minute)).isoformat(),
                    pair="EUR_USD",
                    signal_name="directional_forecast",
                    direction="UP",
                    lead_time_min=60,
                    confidence=0.24,
                    raw_confidence=0.80,
                    calibration_multiplier=0.30,
                    entry_price=1.1,
                    predicted_target_price=1.102,
                    predicted_invalidation_price=1.099,
                    resolution_window_min=60,
                    resolution_status=status,
                    resolution_evidence=(
                        "target touched before invalidation" if status == "HIT" else ""
                    ),
                    regime_at_emission="TREND",
                    cycle_id=cycle_id,
                )

            write_ledger(
                [
                    trial(minute=0, status="PENDING", cycle_id="pending-first"),
                    trial(minute=5, status="HIT", cycle_id="overlapping-hit"),
                ],
                root,
            )

            self.assertNotIn("directional_forecast_up", compute_hit_rates(root))

    def test_truth_missing_timeout_first_raw_trial_suppresses_overlapping_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            base = datetime(2026, 7, 13, tzinfo=timezone.utc)
            common = {
                "pair": "EUR_USD",
                "signal_name": "directional_forecast",
                "direction": "UP",
                "lead_time_min": 60,
                "confidence": 0.24,
                "raw_confidence": 0.80,
                "calibration_multiplier": 0.30,
                "entry_price": 1.1,
                "predicted_target_price": 1.102,
                "predicted_invalidation_price": 1.099,
                "resolution_window_min": 60,
                "regime_at_emission": "TREND",
            }
            write_ledger(
                [
                    LedgerEntry(
                        timestamp_emitted_utc=base.isoformat(),
                        resolution_status="TIMEOUT",
                        resolution_evidence="no candle truth for projection window",
                        cycle_id="truth-missing-first",
                        **common,
                    ),
                    LedgerEntry(
                        timestamp_emitted_utc=(base + timedelta(minutes=5)).isoformat(),
                        resolution_status="HIT",
                        resolution_evidence="target touched before invalidation",
                        cycle_id="overlapping-hit",
                        **common,
                    ),
                ],
                root,
            )

            self.assertNotIn("directional_forecast_up", compute_hit_rates(root))

    def test_low_raw_first_trial_does_not_suppress_overlapping_entry_grade_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            base = datetime(2026, 7, 13, tzinfo=timezone.utc)

            def trial(*, minute: int, raw_confidence: float, status: str, cycle_id: str) -> LedgerEntry:
                return LedgerEntry(
                    timestamp_emitted_utc=(base + timedelta(minutes=minute)).isoformat(),
                    pair="EUR_USD",
                    signal_name="directional_forecast",
                    direction="UP",
                    lead_time_min=60,
                    confidence=raw_confidence * 0.30,
                    raw_confidence=raw_confidence,
                    calibration_multiplier=0.30,
                    entry_price=1.1,
                    predicted_target_price=1.102,
                    predicted_invalidation_price=1.099,
                    resolution_window_min=60,
                    resolution_status=status,
                    resolution_evidence="target touched before invalidation",
                    regime_at_emission="TREND",
                    cycle_id=cycle_id,
                )

            write_ledger(
                [
                    trial(
                        minute=0,
                        raw_confidence=0.20,
                        status="HIT",
                        cycle_id="low-raw-first",
                    ),
                    trial(
                        minute=5,
                        raw_confidence=0.80,
                        status="HIT",
                        cycle_id="entry-grade-overlap",
                    ),
                ],
                root,
            )

            bucket = compute_hit_rates(root)["directional_forecast_up"]["EUR_USD:TREND"]
            self.assertEqual(bucket["samples"], 1)
            self.assertEqual(bucket["hit_rate"], 1.0)

    def test_low_raw_range_does_not_suppress_overlapping_entry_grade_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            base = datetime(2026, 7, 13, tzinfo=timezone.utc)
            write_ledger(
                [
                    LedgerEntry(
                        timestamp_emitted_utc=base.isoformat(),
                        pair="EUR_USD",
                        signal_name="directional_forecast",
                        direction="RANGE",
                        lead_time_min=60,
                        confidence=0.06,
                        raw_confidence=0.20,
                        calibration_multiplier=0.30,
                        entry_price=1.100,
                        predicted_target_price=None,
                        predicted_range_low_price=1.098,
                        predicted_range_high_price=1.102,
                        resolution_window_min=60,
                        resolution_status="HIT",
                        regime_at_emission="RANGE",
                        cycle_id="low-raw-range-first",
                    ),
                    LedgerEntry(
                        timestamp_emitted_utc=(base + timedelta(minutes=5)).isoformat(),
                        pair="EUR_USD",
                        signal_name="directional_forecast",
                        direction="UP",
                        lead_time_min=60,
                        confidence=0.24,
                        raw_confidence=0.80,
                        calibration_multiplier=0.30,
                        entry_price=1.100,
                        predicted_target_price=1.102,
                        predicted_invalidation_price=1.099,
                        resolution_window_min=60,
                        resolution_status="HIT",
                        resolution_evidence="target touched before invalidation",
                        regime_at_emission="TREND",
                        cycle_id="entry-grade-up-overlap",
                    ),
                ],
                root,
            )

            hit_rates = compute_hit_rates(root)
            self.assertNotIn("directional_forecast_range", hit_rates)
            bucket = hit_rates["directional_forecast_up"]["EUR_USD:TREND"]
            self.assertEqual(bucket["samples"], 1)
            self.assertEqual(bucket["hit_rate"], 1.0)

    def test_malformed_ex_ante_geometry_does_not_occupy_pair_clock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            base = datetime(2026, 7, 13, tzinfo=timezone.utc)

            def trial(
                *,
                minute: int,
                direction: str,
                entry_price: float | None,
                target: float,
                invalidation: float,
                cycle_id: str,
            ) -> LedgerEntry:
                return LedgerEntry(
                    timestamp_emitted_utc=(base + timedelta(minutes=minute)).isoformat(),
                    pair="EUR_USD",
                    signal_name="directional_forecast",
                    direction=direction,
                    lead_time_min=60,
                    confidence=0.24,
                    raw_confidence=0.80,
                    calibration_multiplier=0.30,
                    entry_price=entry_price,
                    predicted_target_price=target,
                    predicted_invalidation_price=invalidation,
                    resolution_window_min=60,
                    resolution_status="HIT",
                    resolution_evidence="target touched before invalidation",
                    regime_at_emission="TREND",
                    cycle_id=cycle_id,
                )

            write_ledger(
                [
                    trial(
                        minute=0,
                        direction="UP",
                        entry_price=None,
                        target=1.102,
                        invalidation=1.099,
                        cycle_id="missing-entry",
                    ),
                    trial(
                        minute=1,
                        direction="UP",
                        entry_price=0.0,
                        target=1.102,
                        invalidation=1.099,
                        cycle_id="nonpositive-entry",
                    ),
                    trial(
                        minute=2,
                        direction="UP",
                        entry_price=1.100,
                        target=1.098,
                        invalidation=1.099,
                        cycle_id="up-wrong-side",
                    ),
                    trial(
                        minute=3,
                        direction="DOWN",
                        entry_price=1.100,
                        target=1.102,
                        invalidation=1.099,
                        cycle_id="down-wrong-side",
                    ),
                    trial(
                        minute=5,
                        direction="UP",
                        entry_price=1.100,
                        target=1.102,
                        invalidation=1.099,
                        cycle_id="valid-overlap",
                    ),
                ],
                root,
            )

            bucket = compute_hit_rates(root)["directional_forecast_up"]["EUR_USD:TREND"]
            self.assertEqual(bucket["samples"], 1)
            self.assertEqual(bucket["hit_rate"], 1.0)

    def test_invalid_or_overflowing_horizons_do_not_occupy_pair_clock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            base = datetime(2026, 7, 13, tzinfo=timezone.utc)

            def trial(
                *,
                emitted_at: str,
                pair: str,
                window_min: float,
                status: str,
                cycle_id: str,
            ) -> LedgerEntry:
                return LedgerEntry(
                    timestamp_emitted_utc=emitted_at,
                    pair=pair,
                    signal_name="directional_forecast",
                    direction="UP",
                    lead_time_min=min(window_min, 60.0),
                    confidence=0.24,
                    raw_confidence=0.80,
                    calibration_multiplier=0.30,
                    entry_price=1.100,
                    predicted_target_price=1.102,
                    predicted_invalidation_price=1.099,
                    resolution_window_min=window_min,
                    resolution_status=status,
                    resolution_evidence=(
                        "target touched before invalidation"
                        if status == "HIT"
                        else "invalidation touched before target"
                    ),
                    regime_at_emission="TREND",
                    cycle_id=cycle_id,
                )

            write_ledger(
                [
                    trial(
                        emitted_at=base.isoformat(),
                        pair="EUR_USD",
                        window_min=1e300,
                        status="HIT",
                        cycle_id="huge-window",
                    ),
                    trial(
                        emitted_at="9999-12-31T23:59:59+00:00",
                        pair="EUR_USD",
                        window_min=60.0,
                        status="HIT",
                        cycle_id="year-9999-overflow",
                    ),
                    trial(
                        emitted_at=(base + timedelta(minutes=5)).isoformat(),
                        pair="EUR_USD",
                        window_min=60.0,
                        status="HIT",
                        cycle_id="valid-after-invalid",
                    ),
                    trial(
                        emitted_at=base.isoformat(),
                        pair="GBP_USD",
                        window_min=DIRECTIONAL_FORECAST_MAX_RESOLUTION_WINDOW_MIN,
                        status="MISS",
                        cycle_id="max-boundary-a",
                    ),
                    trial(
                        emitted_at=(
                            base
                            + timedelta(
                                minutes=DIRECTIONAL_FORECAST_MAX_RESOLUTION_WINDOW_MIN
                            )
                        ).isoformat(),
                        pair="GBP_USD",
                        window_min=DIRECTIONAL_FORECAST_MAX_RESOLUTION_WINDOW_MIN,
                        status="HIT",
                        cycle_id="max-boundary-b",
                    ),
                ],
                root,
            )

            hit_rates = compute_hit_rates(root)
            self.assertEqual(
                hit_rates["directional_forecast_up"]["EUR_USD:TREND"]["samples"],
                1,
            )
            self.assertEqual(
                hit_rates["directional_forecast_up"]["EUR_USD:TREND"]["hit_rate"],
                1.0,
            )
            self.assertEqual(
                hit_rates["directional_forecast_up"]["GBP_USD:TREND"]["samples"],
                2,
            )

    def test_invalid_pairs_cannot_poison_global_directional_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            base = datetime(2026, 7, 1, tzinfo=timezone.utc)
            entries: list[LedgerEntry] = []

            def trial(*, index: int, pair: str, status: str, cycle_id: str) -> LedgerEntry:
                return LedgerEntry(
                    timestamp_emitted_utc=(base + timedelta(hours=index)).isoformat(),
                    pair=pair,
                    signal_name="directional_forecast",
                    direction="UP",
                    lead_time_min=60,
                    confidence=0.24,
                    raw_confidence=0.80,
                    calibration_multiplier=0.30,
                    entry_price=1.100,
                    predicted_target_price=1.102,
                    predicted_invalidation_price=1.099,
                    resolution_window_min=60,
                    resolution_status=status,
                    resolution_evidence=(
                        "target touched before invalidation"
                        if status == "HIT"
                        else "invalidation touched before target"
                    ),
                    regime_at_emission="TREND",
                    cycle_id=cycle_id,
                )

            for index in range(30):
                entries.append(
                    trial(
                        index=index,
                        pair="EUR_USD",
                        status="MISS",
                        cycle_id=f"valid-miss-{index}",
                    )
                )
            invalid_pairs = ("", " ", "eur_usd", "XAU_USD")
            for index in range(40):
                entries.append(
                    trial(
                        index=100 + index,
                        pair=invalid_pairs[index % len(invalid_pairs)],
                        status="HIT",
                        cycle_id=f"invalid-pair-hit-{index}",
                    )
                )
            write_ledger(entries, root)

            hit_rates = compute_hit_rates(root)
            global_bucket = hit_rates["directional_forecast_up"][
                "_all_pairs:TREND"
            ]
            multiplier = confidence_calibration(
                "directional_forecast_up",
                "GBP_USD",
                hit_rates=hit_rates,
                regime="TREND",
            )

            self.assertEqual(global_bucket["samples"], 30)
            self.assertEqual(global_bucket["hit_rate"], 0.0)
            self.assertEqual(multiplier, CONFIDENCE_MIN_MULTIPLIER)

    def test_backfilled_old_new_schema_row_does_not_steal_lookback_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            base = datetime(2026, 7, 13, tzinfo=timezone.utc)

            def trial(*, minute: int, status: str, cycle_id: str) -> LedgerEntry:
                return LedgerEntry(
                    timestamp_emitted_utc=(base + timedelta(minutes=minute)).isoformat(),
                    pair="EUR_USD",
                    signal_name="directional_forecast",
                    direction="UP",
                    lead_time_min=60,
                    confidence=0.24,
                    raw_confidence=0.80,
                    calibration_multiplier=0.30,
                    entry_price=1.100,
                    predicted_target_price=1.102,
                    predicted_invalidation_price=1.099,
                    resolution_window_min=60,
                    resolution_status=status,
                    resolution_evidence=(
                        "target touched before invalidation"
                        if status == "HIT"
                        else "invalidation touched before target"
                    ),
                    regime_at_emission="TREND",
                    cycle_id=cycle_id,
                )

            # The older MISS arrived late and is physically last in JSONL.
            write_ledger(
                [
                    trial(minute=60, status="HIT", cycle_id="newer-first-write"),
                    trial(minute=0, status="MISS", cycle_id="older-backfill"),
                ],
                root,
            )

            bucket = compute_hit_rates(root, lookback=1)["directional_forecast_up"][
                "EUR_USD:TREND"
            ]
            self.assertEqual(bucket["samples"], 1)
            self.assertEqual(bucket["hit_rate"], 1.0)

    def test_equal_timestamp_new_schema_lookback_keeps_source_tie_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            emitted_at = "2026-07-13T00:00:00+00:00"

            def trial(*, pair: str, status: str, cycle_id: str) -> LedgerEntry:
                return LedgerEntry(
                    timestamp_emitted_utc=emitted_at,
                    pair=pair,
                    signal_name="directional_forecast",
                    direction="UP",
                    lead_time_min=60,
                    confidence=0.24,
                    raw_confidence=0.80,
                    calibration_multiplier=0.30,
                    entry_price=1.100,
                    predicted_target_price=1.102,
                    predicted_invalidation_price=1.099,
                    resolution_window_min=60,
                    resolution_status=status,
                    resolution_evidence=(
                        "target touched before invalidation"
                        if status == "HIT"
                        else "invalidation touched before target"
                    ),
                    regime_at_emission="TREND",
                    cycle_id=cycle_id,
                )

            write_ledger(
                [
                    trial(pair="EUR_USD", status="HIT", cycle_id="tie-first"),
                    trial(pair="GBP_USD", status="MISS", cycle_id="tie-second"),
                ],
                root,
            )

            bucket = compute_hit_rates(root, lookback=1)["directional_forecast_up"][
                "_all_pairs:TREND"
            ]
            self.assertEqual(bucket["samples"], 1)
            self.assertEqual(bucket["hit_rate"], 0.0)

    def test_mixed_legacy_and_new_rows_share_chronological_lookback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            base = datetime(2026, 7, 13, tzinfo=timezone.utc)

            def trial(
                *,
                minute: int,
                status: str,
                cycle_id: str,
                new_schema: bool,
            ) -> LedgerEntry:
                return LedgerEntry(
                    timestamp_emitted_utc=(base + timedelta(minutes=minute)).isoformat(),
                    pair="EUR_USD",
                    signal_name="directional_forecast",
                    direction="UP",
                    lead_time_min=60,
                    confidence=0.80,
                    raw_confidence=0.80 if new_schema else None,
                    calibration_multiplier=1.0 if new_schema else None,
                    entry_price=1.100,
                    predicted_target_price=1.102,
                    predicted_invalidation_price=1.099,
                    resolution_window_min=60,
                    resolution_status=status,
                    resolution_evidence=(
                        "target touched before invalidation"
                        if status == "HIT"
                        else "invalidation touched before target"
                    ),
                    regime_at_emission="TREND",
                    cycle_id=cycle_id,
                )

            write_ledger(
                [
                    trial(
                        minute=60,
                        status="HIT",
                        cycle_id="new-newer-source-slot-0",
                        new_schema=True,
                    ),
                    trial(
                        minute=180,
                        status="MISS",
                        cycle_id="legacy-late-source-slot-1",
                        new_schema=False,
                    ),
                    trial(
                        minute=0,
                        status="MISS",
                        cycle_id="new-older-source-slot-2",
                        new_schema=True,
                    ),
                    trial(
                        minute=-60,
                        status="HIT",
                        cycle_id="legacy-early-source-slot-3",
                        new_schema=False,
                    ),
                ],
                root,
            )

            bucket = compute_hit_rates(root, lookback=2)["directional_forecast_up"][
                "EUR_USD:TREND"
            ]
            self.assertEqual(bucket["samples"], 2)
            self.assertEqual(bucket["hit_rate"], 0.5)

    def test_legacy_and_non_directional_backfills_use_emission_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            entries = [
                LedgerEntry(
                    timestamp_emitted_utc="2026-07-13T01:00:00+00:00",
                    pair="EUR_USD",
                    signal_name="directional_forecast",
                    direction="UP",
                    lead_time_min=60,
                    confidence=0.80,
                    entry_price=1.100,
                    predicted_target_price=1.102,
                    predicted_invalidation_price=1.099,
                    resolution_window_min=60,
                    resolution_status="HIT",
                    regime_at_emission="TREND",
                    cycle_id="legacy-newer-first",
                ),
                LedgerEntry(
                    timestamp_emitted_utc="2026-07-13T00:00:00+00:00",
                    pair="EUR_USD",
                    signal_name="directional_forecast",
                    direction="UP",
                    lead_time_min=60,
                    confidence=0.80,
                    entry_price=1.100,
                    predicted_target_price=1.102,
                    predicted_invalidation_price=1.099,
                    resolution_window_min=60,
                    resolution_status="MISS",
                    regime_at_emission="TREND",
                    cycle_id="legacy-older-last",
                ),
                LedgerEntry(
                    timestamp_emitted_utc="2026-07-13T01:00:00+00:00",
                    pair="EUR_USD",
                    signal_name="sig_x",
                    direction="EITHER",
                    lead_time_min=60,
                    confidence=0.80,
                    entry_price=1.100,
                    predicted_target_price=None,
                    resolution_window_min=60,
                    resolution_status="HIT",
                    regime_at_emission="TREND",
                    cycle_id="signal-newer-first",
                ),
                LedgerEntry(
                    timestamp_emitted_utc="2026-07-13T00:00:00+00:00",
                    pair="EUR_USD",
                    signal_name="sig_x",
                    direction="EITHER",
                    lead_time_min=60,
                    confidence=0.80,
                    entry_price=1.100,
                    predicted_target_price=None,
                    resolution_window_min=60,
                    resolution_status="MISS",
                    regime_at_emission="TREND",
                    cycle_id="signal-older-last",
                ),
            ]
            write_ledger(entries, root)

            hit_rates = compute_hit_rates(root, lookback=1)
            self.assertEqual(
                hit_rates["directional_forecast_up"]["EUR_USD:TREND"]["hit_rate"],
                1.0,
            )
            self.assertEqual(hit_rates["sig_x"]["EUR_USD:TREND"]["hit_rate"], 1.0)

    def test_invalid_or_partial_raw_schema_never_falls_back_to_legacy_confidence(self) -> None:
        base_payload = {
            "timestamp_emitted_utc": "2026-07-13T00:00:00+00:00",
            "pair": "EUR_USD",
            "signal_name": "directional_forecast",
            "direction": "UP",
            "lead_time_min": 60,
            "confidence": 0.99,
            "entry_price": 1.1,
            "predicted_target_price": 1.102,
            "predicted_invalidation_price": 1.099,
            "resolution_window_min": 60,
            "resolution_status": "HIT",
            "resolution_evidence": "target touched before invalidation",
            "regime_at_emission": "TREND",
            "cycle_id": "invalid-schema",
        }
        variants = (
            {"raw_confidence": "corrupt", "calibration_multiplier": 0.30},
            {"raw_confidence": 0.80},
            {"raw_confidence": 0.80, "calibration_multiplier": "corrupt"},
        )
        for variant in variants:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                ledger_path = root / "projection_ledger.jsonl"
                ledger_path.write_text(
                    json.dumps({**base_payload, **variant}) + "\n",
                    encoding="utf-8",
                )

                loaded = load_ledger(root)[0]
                self.assertTrue(loaded.raw_confidence_field_present)
                self.assertIn("raw_confidence", loaded.to_dict())
                self.assertNotIn("directional_forecast_up", compute_hit_rates(root))

    def test_forged_confidence_triplet_does_not_occupy_pair_clock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            base = datetime(2026, 7, 13, tzinfo=timezone.utc)

            def trial(
                *,
                minute: int,
                confidence: float,
                raw_confidence: float,
                multiplier: float,
                status: str,
                cycle_id: str,
            ) -> LedgerEntry:
                return LedgerEntry(
                    timestamp_emitted_utc=(base + timedelta(minutes=minute)).isoformat(),
                    pair="EUR_USD",
                    signal_name="directional_forecast",
                    direction="UP",
                    lead_time_min=60,
                    confidence=confidence,
                    raw_confidence=raw_confidence,
                    calibration_multiplier=multiplier,
                    entry_price=1.100,
                    predicted_target_price=1.102,
                    predicted_invalidation_price=1.099,
                    resolution_window_min=60,
                    resolution_status=status,
                    resolution_evidence=(
                        "target touched before invalidation"
                        if status == "HIT"
                        else "invalidation touched before target"
                    ),
                    regime_at_emission="TREND",
                    cycle_id=cycle_id,
                )

            write_ledger(
                [
                    trial(
                        minute=0,
                        confidence=0.06,
                        raw_confidence=0.80,
                        multiplier=0.30,
                        status="MISS",
                        cycle_id="forged-triplet",
                    ),
                    trial(
                        minute=1,
                        confidence=0.04,
                        raw_confidence=0.80,
                        multiplier=0.05,
                        status="MISS",
                        cycle_id="multiplier-below-contract",
                    ),
                    trial(
                        minute=2,
                        confidence=1.0,
                        raw_confidence=0.80,
                        multiplier=1.60,
                        status="MISS",
                        cycle_id="multiplier-above-contract",
                    ),
                    trial(
                        minute=5,
                        confidence=0.24,
                        raw_confidence=0.80,
                        multiplier=0.30,
                        status="HIT",
                        cycle_id="valid-after-forgeries",
                    ),
                ],
                root,
            )

            bucket = compute_hit_rates(root)["directional_forecast_up"]["EUR_USD:TREND"]
            self.assertEqual(bucket["samples"], 1)
            self.assertEqual(bucket["hit_rate"], 1.0)

    def test_malformed_json_and_numeric_rows_do_not_stop_later_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "projection_ledger.jsonl"
            base = datetime(2026, 7, 13, tzinfo=timezone.utc)

            def payload(*, minute: int, status: str, cycle_id: str) -> dict[str, object]:
                return {
                    "timestamp_emitted_utc": (base + timedelta(minutes=minute)).isoformat(),
                    "pair": "EUR_USD",
                    "signal_name": "directional_forecast",
                    "direction": "UP",
                    "lead_time_min": 60.0,
                    "confidence": 0.24,
                    "raw_confidence": 0.80,
                    "calibration_multiplier": 0.30,
                    "entry_price": 1.100,
                    "predicted_target_price": 1.102,
                    "predicted_invalidation_price": 1.099,
                    "resolution_window_min": 60.0,
                    "resolution_status": status,
                    "resolution_evidence": (
                        "target touched before invalidation"
                        if status == "HIT"
                        else "invalidation touched before target"
                    ),
                    "regime_at_emission": "TREND",
                    "cycle_id": cycle_id,
                }

            bad_lead = {**payload(minute=5, status="HIT", cycle_id="bad-lead"), "lead_time_min": "bad"}
            bad_confidence = {
                **payload(minute=6, status="HIT", cycle_id="bad-confidence"),
                "confidence": float("inf"),
            }
            bad_window = {
                **payload(minute=7, status="HIT", cycle_id="bad-window"),
                "resolution_window_min": 10**400,
            }
            bad_entry = {
                **payload(minute=8, status="HIT", cycle_id="bad-entry"),
                "entry_price": ["not", "numeric"],
            }
            bad_target = {
                **payload(minute=9, status="HIT", cycle_id="bad-target"),
                "predicted_target_price": 10**400,
            }
            lines = [
                json.dumps(payload(minute=0, status="MISS", cycle_id="valid-miss")),
                json.dumps(7),
                json.dumps([{"not": "an object row"}]),
                "{malformed-json",
                *(json.dumps(item) for item in (
                    bad_lead,
                    bad_confidence,
                    bad_window,
                    bad_entry,
                    bad_target,
                )),
                json.dumps(payload(minute=60, status="HIT", cycle_id="valid-hit")),
            ]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            entries = load_ledger(root)
            first_bucket = compute_hit_rates(root)["directional_forecast_up"][
                "EUR_USD:TREND"
            ]
            self.assertEqual(first_bucket["samples"], 2)
            self.assertEqual(first_bucket["hit_rate"], 0.5)

            from quant_rabbit.strategy.projection_ledger import write_ledger

            write_ledger(entries, root)
            rewritten = load_ledger(root)
            second_bucket = compute_hit_rates(root)["directional_forecast_up"][
                "EUR_USD:TREND"
            ]
            rewritten_bad_lead = next(
                entry for entry in rewritten if entry.cycle_id == "bad-lead"
            )

            self.assertEqual(second_bucket["samples"], 2)
            self.assertEqual(second_bucket["hit_rate"], 0.5)
            self.assertFalse(rewritten_bad_lead.lead_time_min_valid)
            self.assertIsNone(rewritten_bad_lead.to_dict()["lead_time_min"])
            self.assertIn("raw_confidence", rewritten_bad_lead.to_dict())
            self.assertIn("calibration_multiplier", rewritten_bad_lead.to_dict())

    def test_directional_timeout_penalty_requires_scored_no_touch_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            base = datetime(2026, 7, 13, tzinfo=timezone.utc)
            common = {
                "pair": "EUR_USD",
                "signal_name": "directional_forecast",
                "direction": "UP",
                "lead_time_min": 60,
                "confidence": 0.22,
                "raw_confidence": 0.80,
                "calibration_multiplier": 0.275,
                "entry_price": 1.1,
                "predicted_target_price": 1.102,
                "predicted_invalidation_price": 1.099,
                "resolution_window_min": 60,
                "resolution_status": "TIMEOUT",
                "regime_at_emission": "TREND",
            }
            write_ledger(
                [
                    LedgerEntry(
                        timestamp_emitted_utc=base.isoformat(),
                        resolution_evidence=(
                            "target 1.10200 and invalidation 1.09900 both untouched in forecast window"
                        ),
                        cycle_id="scored-no-touch",
                        **common,
                    ),
                    LedgerEntry(
                        timestamp_emitted_utc=(base + timedelta(hours=1)).isoformat(),
                        resolution_evidence="no candle truth for projection window",
                        cycle_id="truth-missing",
                        **common,
                    ),
                ],
                root,
            )

            bucket = compute_hit_rates(root)["directional_forecast_up"]["EUR_USD:TREND"]

            self.assertEqual(bucket["samples"], 0)
            self.assertEqual(bucket["calibration_samples"], 1)
            self.assertEqual(bucket["target_timeout_count"], 1)

    def test_new_range_truth_gap_timeouts_contribute_zero_learning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            base = datetime(2026, 7, 13, tzinfo=timezone.utc)
            evidence_variants = (
                "incomplete closed candle truth for full projection window; retryable",
                "no candle truth for projection window",
                "market closed at projection emission; excluded from calibration",
                "malformed candle truth",
            )
            entries = [
                LedgerEntry(
                    timestamp_emitted_utc=(base + timedelta(hours=2 * index)).isoformat(),
                    pair="EUR_USD",
                    signal_name="directional_forecast",
                    direction="RANGE",
                    lead_time_min=60,
                    confidence=0.24,
                    raw_confidence=0.80,
                    calibration_multiplier=0.30,
                    entry_price=1.1000,
                    predicted_target_price=None,
                    predicted_invalidation_price=None,
                    predicted_range_low_price=1.0990,
                    predicted_range_high_price=1.1010,
                    resolution_window_min=60,
                    resolution_status="TIMEOUT",
                    resolution_evidence=evidence,
                    regime_at_emission="RANGE",
                    cycle_id=f"range-timeout-{index}",
                )
                for index, evidence in enumerate(evidence_variants)
            ]
            entries.append(
                LedgerEntry(
                    timestamp_emitted_utc=(base + timedelta(hours=10)).isoformat(),
                    pair="EUR_USD",
                    signal_name="directional_forecast",
                    direction="RANGE",
                    lead_time_min=60,
                    confidence=0.24,
                    raw_confidence=0.80,
                    calibration_multiplier=0.30,
                    entry_price=1.1000,
                    predicted_target_price=None,
                    predicted_invalidation_price=None,
                    predicted_range_low_price=1.0990,
                    predicted_range_high_price=1.1010,
                    resolution_window_min=60,
                    resolution_status="HIT",
                    resolution_evidence="range held inside forecast box",
                    regime_at_emission="RANGE",
                    cycle_id="range-valid-hit",
                    technical_context_v1=_range_emission_context(
                        current_price=1.1000,
                        h1_atr_pips=20.0,
                    ),
                )
            )
            write_ledger(entries, root)

            bucket = compute_hit_rates(root)["directional_forecast_range"][
                "EUR_USD:RANGE"
            ]

        self.assertEqual(bucket["samples"], 1)
        self.assertEqual(bucket["calibration_samples"], 1)
        self.assertEqual(bucket["target_timeout_count"], 0)
        self.assertEqual(bucket["economic_hit_rate"], 1.0)

    def test_low_confidence_directional_forecasts_do_not_train_entry_grade_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            entries = []
            for index in range(12):
                entries.append(
                    LedgerEntry(
                        timestamp_emitted_utc=(
                            datetime(2026, 6, 16, tzinfo=timezone.utc)
                            + timedelta(hours=index)
                        ).isoformat(),
                        pair="EUR_USD",
                        signal_name="directional_forecast",
                        direction="UP",
                        lead_time_min=60,
                        confidence=0.12,
                        entry_price=1.1,
                        predicted_target_price=1.102,
                        predicted_invalidation_price=1.099,
                        resolution_window_min=60,
                        resolution_status="MISS",
                        resolution_evidence="invalidation 1.09900 touched before target 1.10200",
                        regime_at_emission="TREND",
                        cycle_id=f"watch-only-{index}",
                    )
                )
            for index in range(4):
                entries.append(
                    LedgerEntry(
                        timestamp_emitted_utc=(
                            datetime(2026, 6, 16, tzinfo=timezone.utc)
                            + timedelta(hours=12 + index)
                        ).isoformat(),
                        pair="EUR_USD",
                        signal_name="directional_forecast",
                        direction="UP",
                        lead_time_min=60,
                        confidence=0.7,
                        entry_price=1.1,
                        predicted_target_price=1.102,
                        predicted_invalidation_price=1.099,
                        resolution_window_min=60,
                        resolution_status="HIT",
                        resolution_evidence="target 1.10200 touched before invalidation 1.09900",
                        regime_at_emission="TREND",
                        cycle_id=f"entry-grade-{index}",
                    )
                )
            write_ledger(entries, root)

            hr = compute_hit_rates(root)

            bucket = hr["directional_forecast_up"]["EUR_USD:TREND"]
            self.assertEqual(bucket["samples"], 4)
            self.assertEqual(bucket["calibration_samples"], 4)
            self.assertAlmostEqual(bucket["hit_rate"], 1.0)
            self.assertEqual(bucket["invalidation_first_count"], 0)

    def test_low_confidence_directional_timeouts_do_not_bypass_entry_grade_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            entries = []
            for index in range(12):
                entries.append(
                    LedgerEntry(
                        timestamp_emitted_utc=(
                            datetime(2026, 6, 16, tzinfo=timezone.utc)
                            + timedelta(hours=index)
                        ).isoformat(),
                        pair="EUR_USD",
                        signal_name="directional_forecast",
                        direction="UP",
                        lead_time_min=60,
                        confidence=0.35,
                        entry_price=1.1,
                        predicted_target_price=1.102,
                        predicted_invalidation_price=1.099,
                        resolution_window_min=60,
                        resolution_status="TIMEOUT",
                        resolution_evidence="target and invalidation both untouched in forecast window",
                        regime_at_emission="TREND",
                        cycle_id=f"watch-only-timeout-{index}",
                    )
                )
            for index in range(4):
                entries.append(
                    LedgerEntry(
                        timestamp_emitted_utc=(
                            datetime(2026, 6, 16, tzinfo=timezone.utc)
                            + timedelta(hours=12 + index)
                        ).isoformat(),
                        pair="EUR_USD",
                        signal_name="directional_forecast",
                        direction="UP",
                        lead_time_min=60,
                        confidence=0.7,
                        entry_price=1.1,
                        predicted_target_price=1.102,
                        predicted_invalidation_price=1.099,
                        resolution_window_min=60,
                        resolution_status="HIT",
                        resolution_evidence="target 1.10200 touched before invalidation 1.09900",
                        regime_at_emission="TREND",
                        cycle_id=f"entry-grade-hit-{index}",
                    )
                )
            write_ledger(entries, root)

            hr = compute_hit_rates(root)

            bucket = hr["directional_forecast_up"]["EUR_USD:TREND"]
            self.assertEqual(bucket["samples"], 4)
            self.assertEqual(bucket["calibration_samples"], 4)
            self.assertEqual(bucket["target_timeout_count"], 0)
            self.assertAlmostEqual(bucket["hit_rate"], 1.0)

    def test_compute_hit_rates_excludes_no_touch_miss_from_direction_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            entries = [
                LedgerEntry(
                    timestamp_emitted_utc="2026-06-16T00:00:00Z",
                    pair="EUR_USD",
                    signal_name="directional_forecast",
                    direction="UP",
                    lead_time_min=60,
                    confidence=0.7,
                    entry_price=1.1,
                    predicted_target_price=1.102,
                    predicted_invalidation_price=1.099,
                    resolution_window_min=60,
                    resolution_status="MISS",
                    resolution_evidence=(
                        "target 1.10200 not reached; invalidation 1.09900 also untouched in forecast window"
                    ),
                    regime_at_emission="TREND",
                    cycle_id="cycle-no-touch",
                ),
                LedgerEntry(
                    timestamp_emitted_utc="2026-06-16T01:00:00Z",
                    pair="EUR_USD",
                    signal_name="directional_forecast",
                    direction="UP",
                    lead_time_min=60,
                    confidence=0.7,
                    entry_price=1.1,
                    predicted_target_price=1.102,
                    predicted_invalidation_price=1.099,
                    resolution_window_min=60,
                    resolution_status="MISS",
                    resolution_evidence="2026-06-16T00:10:00Z invalidation 1.09900 touched before target 1.10200",
                    regime_at_emission="TREND",
                    cycle_id="cycle-invalidation-first",
                ),
            ]
            write_ledger(entries, root)

            hr = compute_hit_rates(root)

            bucket = hr["directional_forecast_up"]["EUR_USD:TREND"]
            self.assertEqual(bucket["samples"], 1)
            self.assertEqual(bucket["calibration_samples"], 2)
            self.assertEqual(bucket["target_timeout_count"], 1)
            self.assertAlmostEqual(bucket["target_timeout_rate"], 0.5)
            self.assertEqual(bucket["invalidation_first_count"], 1)
            self.assertAlmostEqual(bucket["invalidation_first_rate"], 1.0)

    def test_directional_forecast_timeouts_dampen_confidence_without_polluting_hit_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            entries = [
                LedgerEntry(
                    timestamp_emitted_utc=(
                        datetime(2026, 6, 16, tzinfo=timezone.utc)
                        + timedelta(hours=idx)
                    ).isoformat(),
                    pair="EUR_CHF",
                    signal_name="directional_forecast",
                    direction="DOWN",
                    lead_time_min=60,
                    confidence=0.7,
                    entry_price=0.94,
                    predicted_target_price=0.936,
                    predicted_invalidation_price=0.943,
                    resolution_window_min=60,
                    resolution_status="TIMEOUT",
                    resolution_evidence="target and invalidation both untouched in forecast window",
                    regime_at_emission="TREND",
                    cycle_id=f"timeout-{idx}",
                )
                for idx in range(12)
            ]
            write_ledger(entries, root)

            hr = compute_hit_rates(root)
            bucket = hr["directional_forecast_down"]["EUR_CHF:TREND"]

            self.assertEqual(bucket["samples"], 0)
            self.assertEqual(bucket["calibration_samples"], 12)
            self.assertEqual(bucket["target_timeout_count"], 12)
            self.assertEqual(bucket["target_timeout_rate"], 1.0)
            self.assertEqual(bucket["hit_rate"], 0.0)
            mult = confidence_calibration(
                "directional_forecast_down",
                "EUR_CHF",
                hit_rates=hr,
                regime="TREND",
            )
            self.assertLess(mult, 1.0)
            self.assertGreater(mult, CONFIDENCE_MIN_MULTIPLIER)

    def test_pair_regime_bucket_survives_global_multi_pair_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from quant_rabbit.strategy.projection_ledger import write_ledger

            entries = []
            for idx in range(10):
                entries.append(LedgerEntry(
                    timestamp_emitted_utc=(
                        datetime(2026, 5, 14, tzinfo=timezone.utc)
                        + timedelta(hours=idx)
                    ).isoformat(),
                    pair="GBP_CHF",
                    signal_name="directional_forecast",
                    direction="DOWN",
                    lead_time_min=60,
                    confidence=0.7,
                    entry_price=1.10,
                    predicted_target_price=1.09,
                    predicted_invalidation_price=1.11,
                    resolution_window_min=60,
                    resolution_status="MISS",
                    regime_at_emission="UNCLEAR",
                    cycle_id=f"bad-gbp-chf-{idx}",
                ))
            for idx in range(1100):
                entries.append(LedgerEntry(
                    timestamp_emitted_utc=f"2026-05-15T00:{idx % 60:02d}:00Z",
                    pair=f"PAIR_{idx}",
                    signal_name="directional_forecast",
                    direction="UP",
                    lead_time_min=60,
                    confidence=0.7,
                    entry_price=1.10,
                    predicted_target_price=1.11,
                    predicted_invalidation_price=1.09,
                    resolution_window_min=60,
                    resolution_status="HIT",
                    regime_at_emission="TREND",
                    cycle_id=f"other-{idx}",
                ))
            write_ledger(entries, root)

            hr = compute_hit_rates(root)

            bucket = hr["directional_forecast_down"]["GBP_CHF:UNCLEAR"]
            self.assertEqual(bucket["samples"], 10)
            self.assertEqual(bucket["hit_rate"], 0.0)
            mult = confidence_calibration(
                "directional_forecast_down",
                "GBP_CHF",
                hit_rates=hr,
                regime="UNCLEAR",
            )
            self.assertLess(mult, 0.5)

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
                ),
                LedgerEntry(
                    timestamp_emitted_utc="2026-05-14T00:01:00Z",
                    pair="EUR_USD", signal_name="directional_forecast", direction="UP",
                    lead_time_min=60, confidence=0.7,
                    entry_price=1.0, predicted_target_price=None,
                    predicted_invalidation_price=None,
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

    def test_select_calibration_signal_name_uses_directional_timeout_evidence(self) -> None:
        hr = {
            "sig_x": {"EUR_USD:TREND": {"hit_rate": 0.8, "samples": 100}},
            "sig_x_up": {
                "EUR_USD:TREND": {
                    "hit_rate": 0.0,
                    "samples": 0,
                    "calibration_samples": 30,
                    "target_timeout_rate": 1.0,
                }
            },
        }

        selected = select_calibration_signal_name(
            "sig_x", "UP", "EUR_USD", hit_rates=hr, regime="TREND",
        )

        self.assertEqual(selected, "sig_x_up")
        self.assertLess(
            confidence_calibration(selected, "EUR_USD", hit_rates=hr, regime="TREND"),
            1.0,
        )

    def test_select_calibration_signal_name_keeps_directional_forecast_alias_even_when_thin(self) -> None:
        hr = {
            "directional_forecast": {"EUR_USD:TREND": {"hit_rate": 0.8, "samples": 100}},
            "directional_forecast_down": {
                "_all_pairs:_all_regimes": {"hit_rate": 0.0, "samples": 10},
            },
        }

        selected = select_calibration_signal_name(
            "directional_forecast", "DOWN", "EUR_USD", hit_rates=hr, regime="TREND",
        )

        self.assertEqual(selected, "directional_forecast_down")

    def test_select_calibration_signal_name_does_not_mix_range_base_into_directional_forecast(self) -> None:
        hr = {
            "directional_forecast": {"EUR_USD:TREND": {"hit_rate": 1.0, "samples": 100}},
            "directional_forecast_up": {"EUR_USD:TREND": {"hit_rate": 0.0, "samples": 3}},
        }

        selected = select_calibration_signal_name(
            "directional_forecast", "UP", "EUR_USD", hit_rates=hr, regime="TREND",
        )

        self.assertEqual(selected, "directional_forecast_up")
        self.assertEqual(
            confidence_calibration(selected, "EUR_USD", hit_rates=hr, regime="TREND"),
            1.0,
        )

    def test_confidence_calibration_ignores_thin_global_bucket(self) -> None:
        hr = {
            "directional_forecast_down": {
                "_all_pairs:_all_regimes": {"hit_rate": 0.0, "samples": 10},
            },
        }

        mult = confidence_calibration(
            "directional_forecast_down",
            "EUR_USD",
            hit_rates=hr,
            regime="TREND",
        )

        self.assertEqual(mult, 1.0)

    def test_confidence_calibration_uses_broad_global_bucket(self) -> None:
        hr = {
            "directional_forecast_down": {
                "_all_pairs:_all_regimes": {"hit_rate": 0.0, "samples": 30},
            },
        }

        mult = confidence_calibration(
            "directional_forecast_down",
            "EUR_USD",
            hit_rates=hr,
            regime="TREND",
        )

        self.assertLess(mult, 1.0)

    def test_select_calibration_signal_name_keeps_range_bucket_when_thin(self) -> None:
        hr = {
            "directional_forecast": {"EUR_USD:RANGE": {"hit_rate": 0.1, "samples": 100}},
            "directional_forecast_range": {"EUR_USD:RANGE": {"hit_rate": 0.0, "samples": 3}},
        }

        selected = select_calibration_signal_name(
            "directional_forecast", "RANGE", "EUR_USD", hit_rates=hr, regime="RANGE",
        )

        self.assertEqual(selected, "directional_forecast_range")
        self.assertEqual(
            confidence_calibration(selected, "EUR_USD", hit_rates=hr, regime="RANGE"),
            1.0,
        )


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
