from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


replay = _load_script("oanda_history_replay_validate")
range_replay = _load_script("oanda_range_scout_replay_validate")
fetch = sys.modules["oanda_range_scout_truth_fetch"]


class _PayloadResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _row(*, current: float = 1.105, low: float = 1.100, high: float = 1.110):
    return range_replay.RangeForecastRow(
        source_index=1,
        timestamp_utc=_dt("2026-07-01T00:00:00Z"),
        pair="EUR_USD",
        confidence=0.8,
        current_price=current,
        range_low_price=low,
        range_high_price=high,
        horizon_min=120.0,
        cycle_id="cycle-1",
    )


def _candle(
    *,
    bid_o: float,
    bid_h: float,
    bid_l: float,
    bid_c: float,
    ask_o: float,
    ask_h: float,
    ask_l: float,
    ask_c: float,
    timestamp: str = "2026-07-01T00:00:05Z",
):
    return replay.QuoteCandle(
        timestamp_utc=_dt(timestamp),
        pair="EUR_USD",
        bid=replay.Ohlc(bid_o, bid_h, bid_l, bid_c),
        ask=replay.Ohlc(ask_o, ask_h, ask_l, ask_c),
    )


def _write_sparse_receipted_history(
    root: Path,
    *,
    empty: bool = False,
) -> tuple[Path, datetime, datetime]:
    start = _dt("2026-07-01T00:00:00Z")
    end = _dt("2026-07-01T00:01:00Z")
    rows = []
    for timestamp, bid_c, ask_c in (
        ("2026-07-01T00:00:00Z", 1.1001, 1.1003),
        ("2026-07-01T00:00:15Z", 1.1002, 1.1004),
        ("2026-07-01T00:00:55Z", 1.1003, 1.1005),
    ):
        rows.append(
            {
                "time": timestamp,
                "complete": True,
                "volume": 1,
                "bid": {
                    "o": str(bid_c),
                    "h": str(bid_c + 0.0002),
                    "l": str(bid_c - 0.0004),
                    "c": str(bid_c),
                },
                "ask": {
                    "o": str(ask_c),
                    "h": str(ask_c + 0.0002),
                    "l": str(ask_c - 0.0003),
                    "c": str(ask_c),
                },
            }
        )

    root = root.resolve()
    run_dir = root / "run"
    task = fetch.FetchTask(
        pair="EUR_USD",
        granularity="S5",
        start=start,
        end=end,
        price="BA",
    )
    client = fetch.OandaReadOnlyClient(
        token="test-token",
        account_id="test-account",
        base_url=fetch.PRODUCTION_OANDA_BASE_URL,
        env_file=Path("/definitely/missing/qr-test-env"),
    )
    payload = {
        "instrument": "EUR_USD",
        "granularity": "S5",
        "candles": [] if empty else rows,
    }
    with mock.patch.object(
        fetch.oanda_module.urllib.request,
        "urlopen",
        return_value=_PayloadResponse(payload),
    ):
        task_summary = fetch._fetch_task(
            client,
            task,
            run_dir=run_dir,
            receipt_root=root,
            max_candles_per_request=4500,
            sleep_seconds=0.0,
            retries=1,
            compress=False,
            dry_run=False,
        )
    summary = {
        "schema_version": fetch.RANGE_FETCH_SUMMARY_SCHEMA,
        "generated_at_utc": fetch.base_fetch._iso(datetime.now(timezone.utc)),
        "window": {
            "from": fetch.base_fetch._iso(start),
            "to": fetch.base_fetch._iso(end),
        },
        "pairs": ["EUR_USD"],
        "granularities": ["S5"],
        "price": "BA",
        "max_candles_per_request": 4500,
        "output_root": str(root),
        "output_dir": str(run_dir),
        "tasks": [task_summary],
        "errors": [],
        "total_rows": task_summary["rows"],
        "total_requests": task_summary["requests"],
        "dry_run": False,
        "include_incomplete": False,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return run_dir, start, end


def _resign_single_receipt(run_dir: Path, mutate) -> None:
    root = run_dir.parent
    receipt_path = root / fetch.RANGE_TRUTH_RECEIPT_FILE
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mutate(receipt)
    receipt["task_manifest_sha256"] = fetch.task_manifest_sha256(
        summary["tasks"][0]
    )
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = fetch._content_sha256(body)
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    summary["tasks"][0]["range_truth_acquisition_receipt_sha256"] = receipt[
        "receipt_sha256"
    ]
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


class OandaRangeScoutReplayValidateTest(unittest.TestCase):
    def test_default_history_selection_uses_latest_successful_run_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            earlier = root / "20260701T000000000000Z"
            latest = root / "20260702T000000000000Z"
            failed = root / "20260703T000000000000Z"
            for run_dir in (earlier, latest, failed):
                run_dir.mkdir()
                (run_dir / "summary.json").write_text("{}", encoding="utf-8")
            (failed / "EUR_USD.partial").write_text("", encoding="utf-8")
            (root / "latest_summary.json").write_text(
                json.dumps({"output_dir": str(latest)}),
                encoding="utf-8",
            )

            selected = range_replay._range_history_dirs(
                None,
                granularity="S5",
                auto_min_days=30.0,
                default_root=root,
            )

            self.assertEqual(selected, [latest])

    def test_loader_dedupes_cycle_and_overlapping_persistence(self) -> None:
        rows = [
            {
                "timestamp_utc": "2026-07-01T00:00:00Z",
                "cycle_id": "cycle-a",
                "pair": "EUR_USD",
                "direction": "RANGE",
                "confidence": 0.8,
                "current_price": 1.105,
                "range_low_price": 1.100,
                "range_high_price": 1.110,
                "horizon_min": 120,
            },
            {
                "timestamp_utc": "2026-07-01T00:00:10Z",
                "cycle_id": "cycle-a",
                "pair": "EUR_USD",
                "direction": "RANGE",
                "confidence": 0.8,
                "current_price": 1.105,
                "range_low_price": 1.100,
                "range_high_price": 1.110,
                "horizon_min": 120,
            },
            {
                "timestamp_utc": "2026-07-01T00:30:00Z",
                "cycle_id": "cycle-b",
                "pair": "EUR_USD",
                "direction": "RANGE",
                "confidence": 0.8,
                "current_price": 1.105,
                "range_low_price": 1.100,
                "range_high_price": 1.110,
                "horizon_min": 120,
            },
            {
                "timestamp_utc": "2026-07-01T01:40:00Z",
                "cycle_id": "cycle-c",
                "pair": "EUR_USD",
                "direction": "RANGE",
                "confidence": 0.8,
                "current_price": 1.106,
                "range_low_price": 1.101,
                "range_high_price": 1.111,
                "horizon_min": 120,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "forecast.jsonl"
            path.write_text("\n".join(json.dumps(item) for item in rows) + "\n")
            loaded, stats = range_replay.load_range_forecasts(path, dedupe_minutes=90)

        self.assertEqual([item.cycle_id for item in loaded], ["cycle-a", "cycle-c"])
        self.assertEqual(stats["skipped_duplicate_rows"], 1)
        self.assertEqual(stats["skipped_persistence_rows"], 1)

    def test_nearest_rail_selects_one_side(self) -> None:
        lower = range_replay.range_signal(_row(current=1.103))
        upper = range_replay.range_signal(_row(current=1.108))

        self.assertEqual((lower.side, lower.entry), ("LONG", 1.100))
        self.assertEqual((upper.side, upper.entry), ("SHORT", 1.110))

    def test_long_same_candle_target_and_stop_is_stop_first(self) -> None:
        signal = range_replay.range_signal(_row(current=1.103))
        candle = _candle(
            bid_o=1.1000,
            bid_h=1.1012,
            bid_l=1.0994,
            bid_c=1.1005,
            ask_o=1.1002,
            ask_h=1.1014,
            ask_l=1.0996,
            ask_c=1.1007,
        )

        result = range_replay.simulate_filled_signal(
            signal,
            [candle],
            take_profit_pips=10.0,
            stop_loss_pips=5.0,
            pip=0.0001,
        )

        self.assertEqual(result["exit_reason"], "STOP_LOSS")
        self.assertEqual(result["realized_pips"], -5.0)
        self.assertTrue(result["same_candle_stop_first"])

    def test_fill_candle_target_only_uses_adverse_ordering_without_deleting_sample(self) -> None:
        signal = range_replay.range_signal(_row(current=1.103))
        candle = _candle(
            bid_o=1.1000,
            bid_h=1.1012,
            bid_l=1.0995,
            bid_c=1.1004,
            ask_o=1.1002,
            ask_h=1.1014,
            ask_l=1.0997,
            ask_c=1.1006,
        )

        result = range_replay.simulate_filled_signal(
            signal,
            [candle],
            take_profit_pips=10.0,
            stop_loss_pips=10.0,
            pip=0.0001,
        )

        self.assertEqual(result["exit_reason"], "TTL_CLOSE")
        self.assertAlmostEqual(result["realized_pips"], 4.0)
        self.assertTrue(result["scorable"])
        self.assertTrue(result["fill_bar_target_ambiguous"])

    def test_fill_candle_close_beyond_target_proves_post_fill_take_profit(self) -> None:
        signal = range_replay.range_signal(_row(current=1.103))
        candle = _candle(
            bid_o=1.1000,
            bid_h=1.1012,
            bid_l=1.0995,
            bid_c=1.1011,
            ask_o=1.1002,
            ask_h=1.1014,
            ask_l=1.0997,
            ask_c=1.1013,
        )

        result = range_replay.simulate_filled_signal(
            signal,
            [candle],
            take_profit_pips=10.0,
            stop_loss_pips=10.0,
            pip=0.0001,
        )

        self.assertEqual(result["exit_reason"], "TAKE_PROFIT")
        self.assertEqual(result["realized_pips"], 10.0)
        self.assertTrue(result["fill_bar_target_ambiguous"])

    def test_timeout_close_uses_executable_bid_for_long(self) -> None:
        signal = range_replay.range_signal(_row(current=1.103))
        candle = _candle(
            bid_o=1.1000,
            bid_h=1.1004,
            bid_l=1.0998,
            bid_c=1.1003,
            ask_o=1.1002,
            ask_h=1.1006,
            ask_l=1.1000,
            ask_c=1.1005,
        )

        result = range_replay.simulate_filled_signal(
            signal,
            [candle],
            take_profit_pips=10.0,
            stop_loss_pips=10.0,
            pip=0.0001,
        )

        self.assertEqual(result["exit_reason"], "TTL_CLOSE")
        self.assertAlmostEqual(result["realized_pips"], 3.0)

    def test_timeout_close_uses_executable_ask_for_short(self) -> None:
        signal = range_replay.range_signal(_row(current=1.108))
        candle = _candle(
            bid_o=1.1098,
            bid_h=1.1100,
            bid_l=1.1094,
            bid_c=1.1095,
            ask_o=1.1100,
            ask_h=1.1102,
            ask_l=1.1096,
            ask_c=1.1097,
        )

        result = range_replay.simulate_filled_signal(
            signal,
            [candle],
            take_profit_pips=10.0,
            stop_loss_pips=10.0,
            pip=0.0001,
        )

        self.assertEqual(result["exit_reason"], "TTL_CLOSE")
        self.assertAlmostEqual(result["realized_pips"], 3.0)

    def test_score_does_not_read_candle_that_starts_at_ttl_boundary(self) -> None:
        row = _row(current=1.103)
        pre_ttl_candles = [
            _candle(
                timestamp=f"2026-07-01T00:00:{second:02d}Z",
                bid_o=1.1000,
                bid_h=1.1004,
                bid_l=1.0998,
                bid_c=1.1003,
                ask_o=1.1002,
                ask_h=1.1006,
                ask_l=1.1000,
                ask_c=1.1005,
            )
            for second in range(0, 60, 5)
        ]
        after_ttl_target = _candle(
            timestamp="2026-07-01T00:01:00Z",
            bid_o=1.1003,
            bid_h=1.1012,
            bid_l=1.1002,
            bid_c=1.1011,
            ask_o=1.1005,
            ask_h=1.1014,
            ask_l=1.1004,
            ask_c=1.1013,
        )

        results, stats = range_replay.score_range_forecasts(
            [row],
            {"EUR_USD": [*pre_ttl_candles, after_ttl_target]},
            ttl_minutes=1,
            tp_grid=(10.0,),
            sl_grid=(10.0,),
            candle_interval=timedelta(seconds=5),
        )

        self.assertEqual(stats["filled_signals"], 1)
        self.assertEqual(results[0]["exit_reason"], "TTL_CLOSE")
        self.assertAlmostEqual(results[0]["realized_pips"], 3.0)

    def test_microsecond_forecast_uses_explicit_non_lookahead_s5_clock(self) -> None:
        row = replace(
            _row(current=1.103),
            timestamp_utc=_dt("2026-07-01T00:00:02.123456Z"),
        )
        pre_ttl = [
            _candle(
                timestamp=f"2026-07-01T00:00:{second:02d}Z",
                bid_o=1.1000,
                bid_h=1.1004,
                bid_l=1.0998,
                bid_c=1.1003,
                ask_o=1.1002,
                ask_h=1.1006,
                ask_l=1.1000,
                ask_c=1.1005,
            )
            for second in range(5, 60, 5)
        ]
        post_ttl_target = _candle(
            timestamp="2026-07-01T00:01:00Z",
            bid_o=1.1003,
            bid_h=1.1012,
            bid_l=1.1002,
            bid_c=1.1011,
            ask_o=1.1005,
            ask_h=1.1014,
            ask_l=1.1004,
            ask_c=1.1013,
        )

        results, stats = range_replay.score_range_forecasts(
            [row],
            {"EUR_USD": [*pre_ttl, post_ttl_target]},
            ttl_minutes=1,
            tp_grid=(10.0,),
            sl_grid=(10.0,),
            candle_interval=timedelta(seconds=5),
        )

        self.assertEqual(
            range_replay.range_order_activation_at(row),
            _dt("2026-07-01T00:00:05Z"),
        )
        self.assertEqual(
            range_replay.truth_end(row, ttl_minutes=1),
            _dt("2026-07-01T00:01:00Z"),
        )
        self.assertEqual(stats["complete_truth_windows"], 1)
        self.assertEqual(results[0]["exit_reason"], "TTL_CLOSE")
        self.assertEqual(results[0]["exit_at_utc"], "2026-07-01T00:01:00Z")
        self.assertAlmostEqual(results[0]["realized_pips"], 3.0)

    def test_incomplete_truth_tail_is_not_scored_as_ttl_close(self) -> None:
        row = _row(current=1.103)
        lone_fill = _candle(
            bid_o=1.1000,
            bid_h=1.1004,
            bid_l=1.0998,
            bid_c=1.1003,
            ask_o=1.1002,
            ask_h=1.1006,
            ask_l=1.1000,
            ask_c=1.1005,
        )

        results, stats = range_replay.score_range_forecasts(
            [row],
            {"EUR_USD": [lone_fill]},
            ttl_minutes=90,
            tp_grid=(10.0,),
            sl_grid=(10.0,),
            candle_interval=timedelta(seconds=5),
        )

        self.assertEqual(results, [])
        self.assertEqual(stats["skipped_incomplete_truth_window"], 1)

    def test_truth_window_rejects_a_full_missing_interval_at_start(self) -> None:
        first = _candle(timestamp="2026-07-01T00:00:05Z", **{
            "bid_o": 1.1000,
            "bid_h": 1.1004,
            "bid_l": 1.0998,
            "bid_c": 1.1003,
            "ask_o": 1.1002,
            "ask_h": 1.1006,
            "ask_l": 1.1000,
            "ask_c": 1.1005,
        })

        self.assertFalse(
            range_replay.complete_truth_window(
                [first],
                forecast_start=_dt("2026-07-01T00:00:00Z"),
                forecast_end=_dt("2026-07-01T00:00:10Z"),
                candle_interval=timedelta(seconds=5),
            )
        )

    def test_receipted_full_coverage_accepts_sparse_s5_as_natural_no_tick(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, start, end = _write_sparse_receipted_history(Path(tmp))
            verified = range_replay.validate_sparse_s5_truth_provenance(
                [run_dir],
                required_windows_by_pair={"EUR_USD": [(start, end)]},
                granularity="S5",
            )
            candles, candle_stats = replay._load_candles(
                [run_dir],
                granularity="S5",
                windows_by_pair={"EUR_USD": [(start, end)]},
            )
            range_replay._require_lossless_verified_candle_load(candle_stats, verified)
            results, sparse_stats = range_replay.score_range_forecasts(
                [_row(current=1.103)],
                candles,
                ttl_minutes=1,
                tp_grid=(10.0,),
                sl_grid=(10.0,),
                candle_interval=timedelta(seconds=5),
                verified_sparse_s5_coverage_by_pair=verified.coverage_by_pair,
            )
            strict_results, strict_stats = range_replay.score_range_forecasts(
                [_row(current=1.103)],
                candles,
                ttl_minutes=1,
                tp_grid=(10.0,),
                sl_grid=(10.0,),
                candle_interval=timedelta(seconds=5),
            )

            self.assertEqual(verified.file_count, 1)
            self.assertEqual(verified.row_count, 3)
            self.assertEqual(sparse_stats["complete_truth_windows"], 1)
            self.assertEqual(results[0]["exit_reason"], "TTL_CLOSE")
            self.assertEqual(results[0]["exit_at_utc"], "2026-07-01T00:01:00Z")
            self.assertEqual(strict_results, [])
            self.assertEqual(strict_stats["skipped_incomplete_truth_window"], 1)

    def test_receipted_sparse_tail_without_ttl_boundary_bar_is_unscorable(self) -> None:
        row = _row(current=1.103)
        candles = [
            _candle(
                timestamp=timestamp,
                bid_o=1.1000,
                bid_h=1.1004,
                bid_l=1.0998,
                bid_c=1.1003,
                ask_o=1.1002,
                ask_h=1.1006,
                ask_l=1.1000,
                ask_c=1.1005,
            )
            for timestamp in (
                "2026-07-01T00:00:00Z",
                "2026-07-01T00:00:15Z",
            )
        ]

        results, stats = range_replay.score_range_forecasts(
            [row],
            {"EUR_USD": candles},
            ttl_minutes=1,
            tp_grid=(10.0,),
            sl_grid=(10.0,),
            candle_interval=timedelta(seconds=5),
            verified_sparse_s5_coverage_by_pair={
                "EUR_USD": [
                    (
                        _dt("2026-07-01T00:00:00Z"),
                        _dt("2026-07-01T00:01:00Z"),
                    )
                ]
            },
        )

        self.assertEqual(results, [])
        self.assertEqual(stats["skipped_incomplete_truth_window"], 1)

    def test_receipted_leading_no_tick_gap_keeps_later_loss_evidence(self) -> None:
        row = _row(current=1.103)
        candles = [
            _candle(
                timestamp="2026-07-01T00:00:15Z",
                bid_o=1.1000,
                bid_h=1.1003,
                bid_l=1.0998,
                bid_c=1.1001,
                ask_o=1.1002,
                ask_h=1.1005,
                ask_l=1.0999,
                ask_c=1.1003,
            ),
            _candle(
                timestamp="2026-07-01T00:00:55Z",
                bid_o=1.1001,
                bid_h=1.1002,
                bid_l=1.0994,
                bid_c=1.0995,
                ask_o=1.1003,
                ask_h=1.1004,
                ask_l=1.0996,
                ask_c=1.0997,
            ),
        ]

        results, stats = range_replay.score_range_forecasts(
            [row],
            {"EUR_USD": candles},
            ttl_minutes=1,
            tp_grid=(10.0,),
            sl_grid=(5.0,),
            candle_interval=timedelta(seconds=5),
            verified_sparse_s5_coverage_by_pair={
                "EUR_USD": [
                    (
                        _dt("2026-07-01T00:00:00Z"),
                        _dt("2026-07-01T00:01:00Z"),
                    )
                ]
            },
        )

        self.assertEqual(stats["complete_truth_windows"], 1)
        self.assertEqual(stats["filled_signals"], 1)
        self.assertEqual(results[0]["exit_reason"], "STOP_LOSS")
        self.assertEqual(results[0]["realized_pips"], -5.0)

    def test_receipted_leading_no_tick_gap_fills_first_quote_that_crossed_limit(self) -> None:
        cases = (
            (
                "LONG",
                _row(current=1.101),
                _candle(
                    timestamp="2026-07-01T00:00:15Z",
                    bid_o=1.0996,
                    bid_h=1.0998,
                    bid_l=1.0994,
                    bid_c=1.0995,
                    ask_o=1.0998,
                    ask_h=1.1000,
                    ask_l=1.0996,
                    ask_c=1.0997,
                ),
            ),
            (
                "SHORT",
                _row(current=1.109),
                _candle(
                    timestamp="2026-07-01T00:00:15Z",
                    bid_o=1.1102,
                    bid_h=1.1104,
                    bid_l=1.1100,
                    bid_c=1.1103,
                    ask_o=1.1104,
                    ask_h=1.1106,
                    ask_l=1.1102,
                    ask_c=1.1105,
                ),
            ),
        )
        ttl_boundary = _candle(
            timestamp="2026-07-01T00:00:55Z",
            bid_o=1.1000,
            bid_h=1.1002,
            bid_l=1.0998,
            bid_c=1.1001,
            ask_o=1.1002,
            ask_h=1.1004,
            ask_l=1.1000,
            ask_c=1.1003,
        )

        for side, row, crossed_first_quote in cases:
            with self.subTest(side=side):
                results, stats = range_replay.score_range_forecasts(
                    [row],
                    {"EUR_USD": [crossed_first_quote, ttl_boundary]},
                    ttl_minutes=1,
                    tp_grid=(10.0,),
                    sl_grid=(5.0,),
                    candle_interval=timedelta(seconds=5),
                    verified_sparse_s5_coverage_by_pair={
                        "EUR_USD": [
                            (
                                _dt("2026-07-01T00:00:00Z"),
                                _dt("2026-07-01T00:01:00Z"),
                            )
                        ]
                    },
                )

                self.assertEqual(stats["complete_truth_windows"], 1)
                self.assertEqual(stats["skipped_not_passive"], 0)
                self.assertEqual(stats["filled_signals"], 1)
                self.assertEqual(results[0]["side"], side)
                self.assertEqual(results[0]["fill_at_utc"], "2026-07-01T00:00:15Z")
                self.assertEqual(results[0]["exit_reason"], "STOP_LOSS")
                self.assertEqual(results[0]["realized_pips"], -5.0)

    def test_sparse_s5_provenance_rejects_legacy_hash_and_coverage_gaps(self) -> None:
        cases = ("missing_receipt", "file_hash_drift", "receipt_chain_tamper", "coverage_outside")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                run_dir, start, end = _write_sparse_receipted_history(Path(tmp))
                required = {"EUR_USD": [(start, end)]}
                if case == "missing_receipt":
                    (run_dir.parent / fetch.RANGE_TRUTH_RECEIPT_FILE).unlink()
                elif case == "file_hash_drift":
                    candle_path = Path(
                        json.loads((run_dir / "summary.json").read_text())["tasks"][0]["path"]
                    )
                    candle_path.write_bytes(candle_path.read_bytes() + b"\n")
                elif case == "receipt_chain_tamper":
                    receipt_path = run_dir.parent / fetch.RANGE_TRUTH_RECEIPT_FILE
                    receipt = json.loads(receipt_path.read_text())
                    receipt["rows"] += 1
                    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
                else:
                    required = {"EUR_USD": [(start - timedelta(seconds=5), end)]}

                with self.assertRaises(ValueError):
                    range_replay.validate_sparse_s5_truth_provenance(
                        [run_dir],
                        required_windows_by_pair=required,
                        granularity="S5",
                    )

    def test_overlapping_empty_receipted_windows_do_not_merge_into_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, start, end = _write_sparse_receipted_history(
                root / "first",
                empty=True,
            )
            second, _, _ = _write_sparse_receipted_history(
                root / "second",
                empty=True,
            )

            with self.assertRaises(ValueError):
                range_replay.validate_sparse_s5_truth_provenance(
                    [first, second],
                    required_windows_by_pair={"EUR_USD": [(start, end)]},
                    granularity="S5",
                )

    def test_sparse_s5_provenance_rejects_summary_contract_tampering(self) -> None:
        mutations = {
            "published": lambda value, _root: value["tasks"][0].__setitem__("published", False),
            "dry_run": lambda value, _root: value["tasks"][0].__setitem__("dry_run", True),
            "partial": lambda value, _root: value["tasks"][0].__setitem__("partial_path", "/tmp/x"),
            "errors": lambda value, _root: value["tasks"][0].__setitem__("errors", [{"error": "x"}]),
            "requests": lambda value, _root: value["tasks"][0].__setitem__("requests", 0),
            "windows": lambda value, _root: value["tasks"][0].__setitem__("windows", 0),
            "max_request": lambda value, _root: value.__setitem__(
                "max_candles_per_request", fetch.OANDA_MAX_CANDLES_PER_REQUEST + 1
            ),
            "rows": lambda value, _root: value["tasks"][0].__setitem__(
                "rows", value["tasks"][0]["rows"] + 1
            ),
            "pair": lambda value, _root: value["tasks"][0].__setitem__("pair", "GBP_USD"),
            "granularity": lambda value, _root: value["tasks"][0].__setitem__(
                "granularity", "M1"
            ),
            "price": lambda value, _root: value["tasks"][0].__setitem__("price", "M"),
            "path_containment": lambda value, root: value["tasks"][0].__setitem__(
                "path", str(root / "outside.jsonl")
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                run_dir, start, end = _write_sparse_receipted_history(Path(tmp))
                summary_path = run_dir / "summary.json"
                summary = copy.deepcopy(json.loads(summary_path.read_text(encoding="utf-8")))
                if label == "path_containment":
                    source = Path(summary["tasks"][0]["path"])
                    (run_dir.parent / "outside.jsonl").write_bytes(source.read_bytes())
                mutate(summary, run_dir.parent)
                summary_path.write_text(
                    json.dumps(summary, indent=2, sort_keys=True),
                    encoding="utf-8",
                )

                with self.assertRaises(ValueError):
                    range_replay.validate_sparse_s5_truth_provenance(
                        [run_dir],
                        required_windows_by_pair={"EUR_USD": [(start, end)]},
                        granularity="S5",
                    )

    def test_sparse_s5_provenance_rejects_resigned_immature_and_unparseable_truth(self) -> None:
        for case in (
            "recorded_before_maturity",
            "dependency_hash_drift",
            "parser_skip",
            "duplicate",
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                run_dir, start, end = _write_sparse_receipted_history(Path(tmp))
                summary_path = run_dir / "summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                candle_path = Path(summary["tasks"][0]["path"])
                if case == "recorded_before_maturity":
                    _resign_single_receipt(
                        run_dir,
                        lambda receipt: receipt.__setitem__(
                            "recorded_at_utc", "2026-06-30T23:59:59Z"
                        ),
                    )
                elif case == "dependency_hash_drift":
                    _resign_single_receipt(
                        run_dir,
                        lambda receipt: receipt["dependencies"][0].__setitem__(
                            "sha256", "0" * 64
                        ),
                    )
                else:
                    lines = candle_path.read_text(encoding="utf-8").splitlines()
                    if case == "parser_skip":
                        malformed = json.loads(lines[1])
                        malformed.pop("ask")
                        lines[1] = json.dumps(malformed, sort_keys=True)
                    else:
                        lines.append(lines[-1])
                        summary["tasks"][0]["rows"] += 1
                        summary["total_rows"] += 1
                        summary_path.write_text(
                            json.dumps(summary, indent=2, sort_keys=True),
                            encoding="utf-8",
                        )
                    candle_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

                    def resign_file(receipt):
                        receipt["candle_sha256"] = fetch._sha256_bytes(candle_path.read_bytes())
                        receipt["rows"] = len(lines)

                    _resign_single_receipt(run_dir, resign_file)

                with self.assertRaises(ValueError):
                    range_replay.validate_sparse_s5_truth_provenance(
                        [run_dir],
                        required_windows_by_pair={"EUR_USD": [(start, end)]},
                        granularity="S5",
                    )

    def test_friday_no_market_classification_uses_ttl_not_forecast_horizon(self) -> None:
        friday_16_ny = replace(
            _row(),
            timestamp_utc=_dt("2026-07-03T20:00:00Z"),
            horizon_min=360.0,
        )

        self.assertFalse(
            range_replay.range_truth_is_no_market(friday_16_ny, ttl_minutes=60)
        )
        self.assertTrue(
            range_replay.range_truth_is_no_market(friday_16_ny, ttl_minutes=90)
        )

    def test_nonpositive_grid_distance_is_rejected(self) -> None:
        signal = range_replay.range_signal(_row(current=1.103))
        candle = _candle(
            bid_o=1.1000,
            bid_h=1.1004,
            bid_l=1.0998,
            bid_c=1.1003,
            ask_o=1.1002,
            ask_h=1.1006,
            ask_l=1.1000,
            ask_c=1.1005,
        )

        with self.assertRaises(ValueError):
            range_replay.simulate_filled_signal(
                signal,
                [candle],
                take_profit_pips=10.0,
                stop_loss_pips=-5.0,
                pip=0.0001,
            )

    def test_oos_status_is_insufficient_when_no_family_can_be_evaluated(self) -> None:
        self.assertEqual(
            range_replay._report_status([]),
            "INSUFFICIENT_OOS_SAMPLE_UNDER_TTL_CLOSE_RESEARCH_CONTRACT",
        )

    def test_student_t_lower_bound_does_not_use_optimistic_normal_cutoff(self) -> None:
        items = []
        for idx, pnl in enumerate([2.7] * 5 + [-0.7] * 5):
            day = f"2026-07-{idx + 1:02d}"
            items.append(
                {
                    "timestamp_utc": f"{day}T00:00:00Z",
                    "resolved_day_utc": day,
                    "realized_pips": pnl,
                    "exit_reason": "TAKE_PROFIT" if pnl > 0 else "STOP_LOSS",
                }
            )

        result = range_replay.metrics(items)

        self.assertGreater(result["mean_pips"], 0.0)
        self.assertLess(result["one_sided_95_mean_lower_pips"], 0.0)

    def test_single_resolved_day_validation_is_underpowered(self) -> None:
        items = []
        for idx in range(40):
            forecast_day = idx // 4 + 1
            resolved_day = (
                f"2026-07-{forecast_day:02d}"
                if forecast_day <= 7
                else "2026-07-10"
            )
            items.append(
                {
                    "source_index": idx,
                    "timestamp_utc": f"2026-07-{forecast_day:02d}T00:00:00Z",
                    "resolved_day_utc": resolved_day,
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "take_profit_pips": 5.0,
                    "stop_loss_pips": 5.0,
                    "realized_pips": 5.0,
                    "exit_reason": "TAKE_PROFIT",
                }
            )

        selected = range_replay.train_validation_selections(
            items,
            train_fraction=0.7,
            min_train_fills=20,
            min_validation_fills=10,
            min_total_fills=30,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["validation"]["active_days"], 1)
        self.assertFalse(selected[0]["oos_powered"])
        self.assertFalse(selected[0]["hypothesis_candidate"])
        self.assertEqual(
            range_replay._report_status(selected),
            "INSUFFICIENT_OOS_SAMPLE_UNDER_TTL_CLOSE_RESEARCH_CONTRACT",
        )

    def test_validation_edge_concentrated_in_one_day_is_not_candidate(self) -> None:
        items = []
        source_index = 0
        for day in range(1, 11):
            for sample in range(3):
                items.append(
                    {
                        "source_index": source_index,
                        "timestamp_utc": f"2026-06-{day:02d}T00:00:{sample:02d}Z",
                        "resolved_day_utc": f"2026-06-{day:02d}",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit_pips": 5.0,
                        "stop_loss_pips": 5.0,
                        "realized_pips": 5.0,
                        "exit_reason": "TAKE_PROFIT",
                    }
                )
                source_index += 1
        for sample in range(16):
            items.append(
                {
                    "source_index": source_index,
                    "timestamp_utc": f"2026-06-11T00:00:{sample:02d}Z",
                    "resolved_day_utc": "2026-06-11",
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "take_profit_pips": 5.0,
                    "stop_loss_pips": 5.0,
                    "realized_pips": 5.0,
                    "exit_reason": "TAKE_PROFIT",
                }
            )
            source_index += 1
        for day in range(12, 16):
            items.append(
                {
                    "source_index": source_index,
                    "timestamp_utc": f"2026-06-{day:02d}T00:00:00Z",
                    "resolved_day_utc": f"2026-06-{day:02d}",
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "take_profit_pips": 5.0,
                    "stop_loss_pips": 5.0,
                    "realized_pips": -5.0,
                    "exit_reason": "STOP_LOSS",
                }
            )
            source_index += 1

        selected = range_replay.train_validation_selections(
            items,
            train_fraction=0.7,
            min_train_fills=20,
            min_validation_fills=10,
            min_total_fills=30,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["validation"]["positive_day_rate"], 0.2)
        self.assertEqual(selected[0]["validation"]["max_daily_sample_share"], 0.8)
        self.assertFalse(selected[0]["hypothesis_candidate"])

    def test_low_pair_side_truth_coverage_is_not_powered_or_candidate(self) -> None:
        items = []
        for day in range(1, 16):
            for sample in range(3):
                items.append(
                    {
                        "source_index": len(items),
                        "timestamp_utc": f"2026-06-{day:02d}T00:00:{sample:02d}Z",
                        "resolved_day_utc": f"2026-06-{day:02d}",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit_pips": 5.0,
                        "stop_loss_pips": 5.0,
                        "realized_pips": 5.0,
                        "exit_reason": "TAKE_PROFIT",
                    }
                )

        selected = range_replay.train_validation_selections(
            items,
            train_fraction=0.7,
            min_train_fills=20,
            min_validation_fills=10,
            min_total_fills=30,
            family_truth_coverage={
                ("EUR_USD", "LONG"): {
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "forecast_rows": 450,
                    "complete_truth_windows": 45,
                    "coverage_rate": 0.1,
                }
            },
        )

        self.assertEqual(len(selected), 1)
        self.assertFalse(selected[0]["oos_powered"])
        self.assertFalse(selected[0]["hypothesis_candidate"])
        self.assertEqual(
            range_replay._report_status(selected),
            "INSUFFICIENT_OOS_SAMPLE_UNDER_TTL_CLOSE_RESEARCH_CONTRACT",
        )

    def test_validation_slice_truth_coverage_cannot_hide_behind_full_period(self) -> None:
        items = []
        for day in range(1, 16):
            for sample in range(3):
                items.append(
                    {
                        "source_index": len(items),
                        "timestamp_utc": f"2026-06-{day:02d}T00:00:{sample:02d}Z",
                        "resolved_day_utc": f"2026-06-{day:02d}",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit_pips": 5.0,
                        "stop_loss_pips": 5.0,
                        "realized_pips": 5.0,
                        "exit_reason": "TAKE_PROFIT",
                    }
                )
        forecast_by_day = {
            f"2026-06-{day:02d}": 100 if day <= 10 else 10
            for day in range(1, 16)
        }
        complete_by_day = {
            f"2026-06-{day:02d}": 100 if day <= 10 else 3
            for day in range(1, 16)
        }

        selected = range_replay.train_validation_selections(
            items,
            train_fraction=0.7,
            min_train_fills=20,
            min_validation_fills=10,
            min_total_fills=30,
            family_truth_coverage={
                ("EUR_USD", "LONG"): {
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "forecast_rows": 1050,
                    "complete_truth_windows": 1015,
                    "coverage_rate": round(1015 / 1050, 6),
                    "forecast_rows_by_day": forecast_by_day,
                    "complete_truth_windows_by_day": complete_by_day,
                }
            },
        )

        self.assertEqual(len(selected), 1)
        coverage = selected[0]["truth_coverage"]
        self.assertGreaterEqual(
            coverage["coverage_rate"],
            range_replay.MIN_TRUTH_WINDOW_COVERAGE_RATE,
        )
        self.assertEqual(coverage["train"]["coverage_rate"], 1.0)
        self.assertEqual(coverage["validation"]["coverage_rate"], 0.3)
        self.assertFalse(selected[0]["oos_powered"])
        self.assertFalse(selected[0]["hypothesis_candidate"])

    def test_trade_level_edge_with_negative_daily_t_lower_bound_is_not_candidate(self) -> None:
        positive_days = {1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 15, 16, 17, 18}
        items = []
        for day in range(1, 21):
            pnls = [3.0] * 7 if day in positive_days else [-15.0]
            for slot, pnl in enumerate(pnls):
                items.append(
                    {
                        "source_index": len(items),
                        "timestamp_utc": (
                            f"2026-06-{day:02d}T{slot * 2:02d}:00:00Z"
                        ),
                        "resolved_day_utc": f"2026-06-{day:02d}",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit_pips": 3.0,
                        "stop_loss_pips": 15.0,
                        "realized_pips": pnl,
                        "exit_reason": "TAKE_PROFIT" if pnl > 0 else "STOP_LOSS",
                    }
                )

        selected = range_replay.train_validation_selections(
            items,
            train_fraction=0.7,
            min_train_fills=20,
            min_validation_fills=10,
            min_total_fills=30,
        )

        self.assertEqual(len(selected), 1)
        validation = selected[0]["validation"]
        self.assertGreater(validation["one_sided_95_mean_lower_pips"], 0.0)
        self.assertLess(validation["one_sided_95_daily_mean_lower_pips"], 0.0)
        self.assertFalse(selected[0]["hypothesis_candidate"])

    def test_train_only_edge_is_not_candidate(self) -> None:
        items = []
        for idx in range(40):
            pnl = 5.0 if idx < 28 else -5.0
            items.append(
                {
                    "source_index": idx,
                    "timestamp_utc": f"2026-07-{idx // 4 + 1:02d}T00:00:00Z",
                    "campaign_day_jst": f"2026-07-{idx // 4 + 1:02d}",
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "take_profit_pips": 5.0,
                    "stop_loss_pips": 5.0,
                    "realized_pips": pnl,
                    "exit_reason": "TAKE_PROFIT" if pnl > 0 else "STOP_LOSS",
                }
            )

        selected = range_replay.train_validation_selections(
            items,
            train_fraction=0.7,
            min_train_fills=20,
            min_validation_fills=10,
            min_total_fills=30,
        )

        self.assertEqual(len(selected), 1)
        self.assertGreater(selected[0]["train"]["mean_pips"], 0)
        self.assertLess(selected[0]["validation"]["mean_pips"], 0)
        self.assertFalse(selected[0]["hypothesis_candidate"])


if __name__ == "__main__":
    unittest.main()
