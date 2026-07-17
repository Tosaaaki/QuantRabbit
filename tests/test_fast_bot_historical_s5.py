from __future__ import annotations

import gzip
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import quant_rabbit.fast_bot_historical_s5 as historical_s5

from quant_rabbit.fast_bot_historical_s5 import (
    HistoricalS5CacheError,
    HistoricalS5SliceRequest,
    build_historical_s5_manifest,
    load_historical_s5_manifest,
    load_historical_s5_slice,
    load_historical_s5_slices,
    write_historical_s5_manifest,
)


UTC = timezone.utc


def _row(pair: str, timestamp: datetime, *, bid: float = 1.1000) -> dict:
    ask = bid + 0.0002
    return {
        "ask": {"c": ask, "h": ask, "l": ask, "o": ask},
        "bid": {"c": bid, "h": bid, "l": bid, "o": bid},
        "complete": True,
        "granularity": "S5",
        "pair": pair,
        "price": "BA",
        "time": timestamp.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
        "volume": 1,
    }


def _write_run(
    root: Path,
    *,
    run_id: str,
    pair: str,
    declared_from: datetime,
    declared_to: datetime,
    rows: list[dict],
    declared_rows: int | None = None,
) -> Path:
    pair_dir = root / run_id / pair
    pair_dir.mkdir(parents=True)
    name = (
        f"{pair}_S5_BA_{declared_from.strftime('%Y%m%dT%H%M%SZ')}_"
        f"{declared_to.strftime('%Y%m%dT%H%M%SZ')}.jsonl.gz"
    )
    candle_path = pair_dir / name
    with gzip.open(candle_path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    relative = candle_path.relative_to(root)
    task = {
        "compressed": True,
        "dry_run": False,
        "errors": [],
        "from": declared_from.isoformat().replace("+00:00", "Z"),
        "granularity": "S5",
        "pair": pair,
        "partial_path": None,
        "path": f"logs/replay/oanda_history/{relative.as_posix()}",
        "price": "BA",
        "published": True,
        "requests": 1,
        "rows": len(rows) if declared_rows is None else declared_rows,
        "to": declared_to.isoformat().replace("+00:00", "Z"),
        "windows": 1,
    }
    summary = {
        "dry_run": False,
        "errors": [],
        "generated_at_utc": declared_to.isoformat(),
        "granularities": ["S5"],
        "pairs": [pair],
        "price": "BA",
        "tasks": [task],
        "total_rows": task["rows"],
        "window": {"from": task["from"], "to": task["to"]},
    }
    (root / run_id / "summary.json").write_text(
        json.dumps(summary, sort_keys=True),
        encoding="utf-8",
    )
    return candle_path


class HistoricalS5ManifestTest(unittest.TestCase):
    def test_explicit_run_scope_is_canonical_and_excludes_other_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            start = datetime(2026, 5, 1, tzinfo=UTC)
            allowed_run = "20260501T010000Z"
            excluded_run = "20260502T010000Z"
            selected = _write_run(
                root,
                run_id=allowed_run,
                pair="EUR_USD",
                declared_from=start,
                declared_to=start + timedelta(hours=1),
                rows=[_row("EUR_USD", start)],
            )
            orphan_dir = root / allowed_run / "GBP_USD"
            orphan_dir.mkdir()
            orphan = (
                orphan_dir / "GBP_USD_S5_BA_20260501T000000Z_20260501T010000Z.jsonl.gz"
            )
            with gzip.open(orphan, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(_row("GBP_USD", start), sort_keys=True) + "\n")
            _write_run(
                root,
                run_id=excluded_run,
                pair="EUR_USD",
                declared_from=start,
                declared_to=start + timedelta(hours=2),
                rows=[_row("EUR_USD", start, bid=9.0)],
                declared_rows=2,
            )

            manifest = build_historical_s5_manifest(
                root,
                pairs=("EUR_USD",),
                allowed_run_ids=(allowed_run,),
            )

            self.assertEqual(
                manifest["selection_policy"],
                historical_s5.EXPLICIT_RUN_SELECTION_POLICY,
            )
            self.assertEqual(
                manifest["run_scope_policy"],
                historical_s5.EXPLICIT_RUN_SCOPE_POLICY,
            )
            self.assertEqual(manifest["allowed_run_ids"], [allowed_run])
            self.assertEqual(manifest["summary_run_ids"], [allowed_run])
            self.assertTrue(manifest["summary_run_ids_exact_set_proved"])
            self.assertTrue(manifest["run_scope_is_outcome_blind"])
            self.assertEqual(
                manifest["selected_sources"][0]["relative_path"],
                selected.relative_to(root).as_posix(),
            )
            self.assertEqual(
                manifest["unadmitted_files"], [orphan.relative_to(root).as_posix()]
            )
            self.assertNotIn(excluded_run, json.dumps(manifest, sort_keys=True))
            with self.assertRaisesRegex(HistoricalS5CacheError, "row count"):
                build_historical_s5_manifest(root, pairs=("EUR_USD",))

    def test_explicit_run_scope_order_canonicalizes_to_the_same_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            start = datetime(2026, 5, 1, tzinfo=UTC)
            run_ids = ("20260501T010000Z", "20260502T010000Z")
            _write_run(
                root,
                run_id=run_ids[0],
                pair="EUR_USD",
                declared_from=start,
                declared_to=start + timedelta(hours=2),
                rows=[_row("EUR_USD", start)],
            )
            _write_run(
                root,
                run_id=run_ids[1],
                pair="EUR_USD",
                declared_from=start + timedelta(hours=1),
                declared_to=start + timedelta(hours=2),
                rows=[_row("EUR_USD", start + timedelta(hours=1))],
            )

            forward = build_historical_s5_manifest(
                root,
                pairs=("EUR_USD",),
                allowed_run_ids=run_ids,
            )
            reverse = build_historical_s5_manifest(
                root,
                pairs=("EUR_USD",),
                allowed_run_ids=tuple(reversed(run_ids)),
            )

            self.assertEqual(forward, reverse)
            self.assertEqual(forward["allowed_run_ids"], list(run_ids))
            self.assertEqual(forward["manifest_sha256"], reverse["manifest_sha256"])

    def test_explicit_run_scope_rejects_unsafe_missing_and_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            no_summary = root / "20260501T010000Z"
            no_summary.mkdir()
            cases = (
                ((), "non-empty"),
                (("20260501T010000Z", "20260501T010000Z"), "unique"),
                (("../20260501T010000Z",), "unsafe"),
                (("20261301T010000Z",), "clock is invalid"),
                (("20260502T010000Z",), "does not exist"),
                (("20260501T010000Z",), "no regular summary"),
            )
            for run_ids, message in cases:
                with (
                    self.subTest(run_ids=run_ids),
                    self.assertRaisesRegex(HistoricalS5CacheError, message),
                ):
                    build_historical_s5_manifest(
                        root,
                        pairs=("EUR_USD",),
                        allowed_run_ids=run_ids,
                    )

    def test_explicit_run_scope_semantic_tamper_fails_even_when_resealed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            start = datetime(2026, 5, 1, tzinfo=UTC)
            run_id = "20260501T010000Z"
            _write_run(
                root,
                run_id=run_id,
                pair="EUR_USD",
                declared_from=start,
                declared_to=start + timedelta(hours=1),
                rows=[_row("EUR_USD", start)],
            )
            manifest = build_historical_s5_manifest(
                root,
                pairs=("EUR_USD",),
                allowed_run_ids=(run_id,),
            )
            manifest["summary_run_ids"] = ["20260502T010000Z"]
            body = {
                key: value
                for key, value in manifest.items()
                if key != "manifest_sha256"
            }
            manifest["manifest_sha256"] = historical_s5._canonical_sha(body)

            with self.assertRaisesRegex(HistoricalS5CacheError, "run scope binding"):
                historical_s5._validate_manifest(manifest)

    def test_parallel_and_serial_manifest_scans_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            start = datetime(2026, 5, 1, tzinfo=UTC)
            pairs = (
                "EUR_USD",
                "GBP_USD",
                "AUD_USD",
                "NZD_USD",
                "USD_JPY",
                "USD_CAD",
            )
            run_ids: list[str] = []
            for index, pair in enumerate(pairs):
                run_id = f"20260501T{index + 1:02d}0000Z"
                run_ids.append(run_id)
                row_count = 2_000 if index == 0 else 1
                rows = [
                    _row(pair, start + timedelta(seconds=offset * 5))
                    for offset in range(row_count)
                ]
                _write_run(
                    root,
                    run_id=run_id,
                    pair=pair,
                    declared_from=start,
                    declared_to=start + timedelta(hours=4),
                    rows=rows,
                )

            serial = build_historical_s5_manifest(
                root,
                pairs=pairs,
                allowed_run_ids=tuple(reversed(run_ids)),
                scan_workers=1,
            )
            with mock.patch.object(historical_s5.os, "cpu_count", return_value=8):
                parallel = build_historical_s5_manifest(
                    root,
                    pairs=pairs,
                    allowed_run_ids=tuple(run_ids),
                    scan_workers=6,
                )

            self.assertEqual(serial, parallel)
            self.assertEqual(serial["manifest_sha256"], parallel["manifest_sha256"])
            self.assertEqual(
                json.dumps(serial, sort_keys=True, separators=(",", ":")),
                json.dumps(parallel, sort_keys=True, separators=(",", ":")),
            )
            serial_path = root / "serial-manifest.json"
            parallel_path = root / "parallel-manifest.json"
            write_historical_s5_manifest(serial_path, serial)
            write_historical_s5_manifest(parallel_path, parallel)
            self.assertEqual(serial_path.read_bytes(), parallel_path.read_bytes())

    def test_parallel_scan_reports_the_same_first_input_order_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            start = datetime(2026, 5, 1, tzinfo=UTC)
            pairs = (
                "EUR_USD",
                "GBP_USD",
                "AUD_USD",
                "NZD_USD",
                "USD_JPY",
                "USD_CAD",
            )
            run_ids: list[str] = []
            for index, pair in enumerate(pairs):
                run_id = f"20260501T{index + 1:02d}0000Z"
                run_ids.append(run_id)
                rows = [_row(pair, start)]
                declared_rows = 2 if index == 1 else None
                if index == 2:
                    rows.append(_row(pair, start))
                _write_run(
                    root,
                    run_id=run_id,
                    pair=pair,
                    declared_from=start,
                    declared_to=start + timedelta(hours=1),
                    rows=rows,
                    declared_rows=declared_rows,
                )

            errors: list[str] = []
            with mock.patch.object(historical_s5.os, "cpu_count", return_value=8):
                for workers in (1, 6):
                    with self.assertRaises(HistoricalS5CacheError) as raised:
                        build_historical_s5_manifest(
                            root,
                            pairs=pairs,
                            allowed_run_ids=tuple(run_ids),
                            scan_workers=workers,
                        )
                    errors.append(str(raised.exception))

            self.assertEqual(errors[0], errors[1])
            self.assertIn("row count", errors[0])

    def test_manifest_scan_worker_count_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(
                historical_s5._resolved_manifest_scan_workers(None, task_count=99),
                1,
            )
            for workers in (True, 0, 7):
                with (
                    self.subTest(workers=workers),
                    self.assertRaisesRegex(HistoricalS5CacheError, "from 1 to 6"),
                ):
                    build_historical_s5_manifest(
                        root,
                        pairs=("EUR_USD",),
                        scan_workers=workers,
                    )

    def test_outcome_blind_duplicate_selection_and_missing_pair_are_explicit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            early = datetime(2026, 5, 1, tzinfo=UTC)
            late = datetime(2026, 5, 2, tzinfo=UTC)
            end_early = early + timedelta(days=2)
            end_late = late + timedelta(days=2)
            first = _write_run(
                root,
                run_id="20260503T000000Z",
                pair="AUD_JPY",
                declared_from=early,
                declared_to=end_early,
                rows=[_row("AUD_JPY", early, bid=100.0)],
            )
            _write_run(
                root,
                run_id="20260504T000000Z",
                pair="AUD_JPY",
                declared_from=late,
                declared_to=end_late,
                rows=[_row("AUD_JPY", late, bid=999.0)],
            )
            _write_run(
                root,
                run_id="20260503T010000Z",
                pair="EUR_USD",
                declared_from=early,
                declared_to=end_early,
                rows=[_row("EUR_USD", early)],
            )

            manifest = build_historical_s5_manifest(
                root,
                pairs=("AUD_JPY", "EUR_USD", "NZD_CHF"),
            )

            by_pair = {row["pair"]: row for row in manifest["selected_sources"]}
            self.assertEqual(
                by_pair["AUD_JPY"]["relative_path"], str(first.relative_to(root))
            )
            self.assertEqual(by_pair["AUD_JPY"]["candidate_count_for_pair"], 2)
            self.assertEqual(manifest["missing_pairs"], ["NZD_CHF"])
            self.assertFalse(manifest["complete_pair_coverage"])
            self.assertEqual(len(manifest["duplicate_candidates"]), 1)
            self.assertFalse(
                manifest["duplicate_candidates"][0]["outcome_data_used_for_selection"]
            )
            self.assertFalse(manifest["forward_proof_eligible"])
            self.assertFalse(manifest["live_permission"])
            self.assertFalse(manifest["broker_mutation_allowed"])
            self.assertFalse(manifest["all_selected_sources_acquisition_receipted"])

    def test_orphan_cache_file_is_visible_but_never_admitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            start = datetime(2026, 5, 1, tzinfo=UTC)
            pair_dir = root / "20260501T010000Z" / "EUR_USD"
            pair_dir.mkdir(parents=True)
            orphan = (
                pair_dir / "EUR_USD_S5_BA_20260501T000000Z_20260501T010000Z.jsonl.gz"
            )
            with gzip.open(orphan, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(_row("EUR_USD", start)) + "\n")

            manifest = build_historical_s5_manifest(root, pairs=("EUR_USD",))

            self.assertEqual(manifest["selected_sources"], [])
            self.assertEqual(manifest["missing_pairs"], ["EUR_USD"])
            self.assertEqual(
                manifest["unadmitted_files"], [str(orphan.relative_to(root))]
            )

    def test_summary_row_count_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            start = datetime(2026, 5, 1, tzinfo=UTC)
            _write_run(
                root,
                run_id="20260501T010000Z",
                pair="EUR_USD",
                declared_from=start,
                declared_to=start + timedelta(hours=1),
                rows=[_row("EUR_USD", start)],
                declared_rows=2,
            )

            with self.assertRaisesRegex(HistoricalS5CacheError, "row count"):
                build_historical_s5_manifest(root, pairs=("EUR_USD",))

    def test_duplicate_timestamp_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            start = datetime(2026, 5, 1, tzinfo=UTC)
            _write_run(
                root,
                run_id="20260501T010000Z",
                pair="EUR_USD",
                declared_from=start,
                declared_to=start + timedelta(hours=1),
                rows=[_row("EUR_USD", start), _row("EUR_USD", start)],
            )

            with self.assertRaisesRegex(HistoricalS5CacheError, "strictly increasing"):
                build_historical_s5_manifest(root, pairs=("EUR_USD",))

    def test_nonzero_nanosecond_tail_is_rejected_instead_of_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            start = datetime(2026, 5, 1, tzinfo=UTC)
            row = _row("EUR_USD", start)
            row["time"] = "2026-05-01T00:00:00.000000001Z"
            _write_run(
                root,
                run_id="20260501T010000Z",
                pair="EUR_USD",
                declared_from=start,
                declared_to=start + timedelta(hours=1),
                rows=[row],
            )

            with self.assertRaisesRegex(HistoricalS5CacheError, "off-grid"):
                build_historical_s5_manifest(root, pairs=("EUR_USD",))

    def test_manifest_scan_rejects_same_inode_mutation_between_hash_and_parse(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            start = datetime(2026, 5, 1, tzinfo=UTC)
            source = _write_run(
                root,
                run_id="20260501T010000Z",
                pair="EUR_USD",
                declared_from=start,
                declared_to=start + timedelta(hours=1),
                rows=[_row("EUR_USD", start)],
            )
            original_hash = historical_s5._sha256_handle

            def hash_then_touch(handle):
                digest = original_hash(handle)
                current = source.stat()
                os.utime(
                    source,
                    ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000_000),
                )
                return digest

            with (
                mock.patch.object(
                    historical_s5,
                    "_sha256_handle",
                    side_effect=hash_then_touch,
                ),
                self.assertRaisesRegex(
                    HistoricalS5CacheError, "changed during stable read"
                ),
            ):
                build_historical_s5_manifest(root, pairs=("EUR_USD",))

    def test_sealed_manifest_round_trip_does_not_rescan_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            root.mkdir()
            start = datetime(2026, 5, 1, tzinfo=UTC)
            candle_path = _write_run(
                root,
                run_id="20260501T010000Z",
                pair="EUR_USD",
                declared_from=start,
                declared_to=start + timedelta(hours=1),
                rows=[_row("EUR_USD", start)],
            )
            manifest = build_historical_s5_manifest(root, pairs=("EUR_USD",))
            manifest_path = Path(tmp) / "manifest.json"
            write_historical_s5_manifest(manifest_path, manifest)
            candle_path.unlink()

            loaded = load_historical_s5_manifest(manifest_path)

            self.assertEqual(loaded, manifest)

    def test_legacy_partial_boundary_candle_is_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            start = datetime(2026, 5, 1, tzinfo=UTC)
            _write_run(
                root,
                run_id="20260501T000012Z",
                pair="EUR_USD",
                declared_from=start,
                declared_to=start + timedelta(seconds=12),
                rows=[
                    _row("EUR_USD", start),
                    _row("EUR_USD", start + timedelta(seconds=5)),
                    _row("EUR_USD", start + timedelta(seconds=10)),
                ],
            )

            manifest = build_historical_s5_manifest(root, pairs=("EUR_USD",))
            source = manifest["selected_sources"][0]
            loaded = load_historical_s5_slice(
                manifest,
                pair="EUR_USD",
                time_from=start,
                time_to=start + timedelta(seconds=12),
            )

            self.assertEqual(source["declared_rows"], 3)
            self.assertEqual(source["usable_rows"], 2)
            self.assertEqual(source["quarantined_boundary_rows"], 1)
            self.assertEqual(
                [item.timestamp_utc for item in loaded.candles],
                [start, start + timedelta(seconds=5)],
            )


class HistoricalS5SliceTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[dict, Path, datetime]:
        start = datetime(2026, 5, 1, tzinfo=UTC)
        rows = [
            _row(
                "EUR_USD", start + timedelta(seconds=offset), bid=1.1 + offset / 100000
            )
            for offset in (0, 5, 10, 15)
        ]
        path = _write_run(
            root,
            run_id="20260501T010000Z",
            pair="EUR_USD",
            declared_from=start,
            declared_to=start + timedelta(hours=1),
            rows=rows,
        )
        return build_historical_s5_manifest(root, pairs=("EUR_USD",)), path, start

    def test_slice_aligns_inward_and_never_includes_horizon_candle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _path, start = self._fixture(Path(tmp))

            loaded = load_historical_s5_slice(
                manifest,
                pair="EUR_USD",
                time_from=start + timedelta(seconds=2),
                time_to=start + timedelta(seconds=14),
            )

            self.assertEqual(loaded.aligned_from_utc, start + timedelta(seconds=5))
            self.assertEqual(loaded.aligned_to_utc, start + timedelta(seconds=10))
            self.assertEqual(
                [item.timestamp_utc for item in loaded.candles],
                [start + timedelta(seconds=5)],
            )
            self.assertEqual(loaded.grid_slot_count, 1)
            self.assertEqual(loaded.no_tick_slot_count, 0)
            receipt = loaded.receipt()
            self.assertTrue(receipt["exact_interval_membership_proved"])
            self.assertTrue(receipt["historical_only"])
            self.assertFalse(receipt["forward_proof_eligible"])
            self.assertFalse(receipt["promotion_allowed"])
            self.assertFalse(receipt["live_permission"])

    def test_streamed_truth_path_hash_matches_legacy_canonical_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _path, start = self._fixture(Path(tmp))
            loaded = load_historical_s5_slice(
                manifest,
                pair="EUR_USD",
                time_from=start,
                time_to=start + timedelta(seconds=20),
            )
            legacy_body = [
                {
                    "timestamp_utc": candle.timestamp_utc.isoformat(),
                    "bid": [
                        candle.bid_o,
                        candle.bid_h,
                        candle.bid_l,
                        candle.bid_c,
                    ],
                    "ask": [
                        candle.ask_o,
                        candle.ask_h,
                        candle.ask_l,
                        candle.ask_c,
                    ],
                }
                for candle in loaded.candles
            ]

            self.assertEqual(
                loaded.receipt()["truth_path_sha256"],
                historical_s5._canonical_sha(legacy_body),
            )

    def test_batch_hashes_and_decompresses_once_per_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _path, start = self._fixture(Path(tmp))
            requests = (
                HistoricalS5SliceRequest(
                    pair="EUR_USD",
                    time_from=start + timedelta(seconds=2),
                    time_to=start + timedelta(seconds=16),
                ),
                HistoricalS5SliceRequest(
                    pair="EUR_USD",
                    time_from=start + timedelta(seconds=8),
                    time_to=start + timedelta(seconds=21),
                ),
                HistoricalS5SliceRequest(
                    pair="EUR_USD",
                    time_from=start,
                    time_to=start + timedelta(seconds=6),
                ),
            )
            with (
                mock.patch.object(
                    historical_s5,
                    "_sha256_handle",
                    wraps=historical_s5._sha256_handle,
                ) as hash_spy,
                mock.patch.object(
                    historical_s5,
                    "_cache_binary_stream",
                    wraps=historical_s5._cache_binary_stream,
                ) as stream_spy,
            ):
                loaded = load_historical_s5_slices(manifest, requests=requests)

            self.assertEqual(hash_spy.call_count, 1)
            self.assertEqual(stream_spy.call_count, 1)
            self.assertEqual(
                [[c.timestamp_utc for c in item.candles] for item in loaded],
                [
                    [start + timedelta(seconds=5), start + timedelta(seconds=10)],
                    [start + timedelta(seconds=10), start + timedelta(seconds=15)],
                    [start],
                ],
            )

    def test_stable_fd_rejects_path_replacement_after_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest, path, start = self._fixture(Path(tmp))
            replacement = path.with_name(path.name + ".replacement")
            replacement.write_bytes(path.read_bytes())
            original_hash = historical_s5._sha256_handle

            def hash_then_replace(handle):
                digest = original_hash(handle)
                os.replace(replacement, path)
                return digest

            with (
                mock.patch.object(
                    historical_s5,
                    "_sha256_handle",
                    side_effect=hash_then_replace,
                ),
                self.assertRaisesRegex(
                    HistoricalS5CacheError,
                    "changed during stable read|path identity changed",
                ),
            ):
                load_historical_s5_slice(
                    manifest,
                    pair="EUR_USD",
                    time_from=start,
                    time_to=start + timedelta(seconds=15),
                )

    def test_missing_grid_slots_remain_no_tick_and_are_not_synthesized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            start = datetime(2026, 5, 1, tzinfo=UTC)
            _write_run(
                root,
                run_id="20260501T010000Z",
                pair="EUR_USD",
                declared_from=start,
                declared_to=start + timedelta(hours=1),
                rows=[
                    _row("EUR_USD", start),
                    _row("EUR_USD", start + timedelta(seconds=10)),
                ],
            )
            manifest = build_historical_s5_manifest(root, pairs=("EUR_USD",))

            loaded = load_historical_s5_slice(
                manifest,
                pair="EUR_USD",
                time_from=start,
                time_to=start + timedelta(seconds=15),
            )

            self.assertEqual(len(loaded.candles), 2)
            self.assertEqual(loaded.grid_slot_count, 3)
            self.assertEqual(loaded.no_tick_slot_count, 1)

    def test_interval_outside_declared_coverage_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _path, start = self._fixture(Path(tmp))

            with self.assertRaisesRegex(HistoricalS5CacheError, "exceeds declared"):
                load_historical_s5_slice(
                    manifest,
                    pair="EUR_USD",
                    time_from=start - timedelta(seconds=5),
                    time_to=start + timedelta(seconds=10),
                )

    def test_file_mutation_after_manifest_fails_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest, path, start = self._fixture(Path(tmp))
            original = path.read_bytes()
            path.write_bytes(original + b"changed")

            with self.assertRaisesRegex(HistoricalS5CacheError, "size changed"):
                load_historical_s5_slice(
                    manifest,
                    pair="EUR_USD",
                    time_from=start,
                    time_to=start + timedelta(seconds=15),
                )

    def test_manifest_tamper_fails_before_file_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _path, start = self._fixture(Path(tmp))
            manifest["selection_is_outcome_blind"] = False

            with self.assertRaisesRegex(HistoricalS5CacheError, "digest mismatch"):
                load_historical_s5_slice(
                    manifest,
                    pair="EUR_USD",
                    time_from=start,
                    time_to=start + timedelta(seconds=15),
                )


if __name__ == "__main__":
    unittest.main()
