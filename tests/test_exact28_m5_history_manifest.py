from __future__ import annotations

import copy
import gzip
import hashlib
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from quant_rabbit import fast_bot_historical_s5 as historical_s5
from quant_rabbit.exact28_m5_history_manifest import (
    DEFAULT_PERIOD_FROM_UTC,
    DEFAULT_PERIOD_TO_UTC,
    HistoricalM5ManifestError,
    _expected_shards,
    build_exact28_m5_history_manifest,
    load_exact28_m5_history_manifest,
    validate_exact28_m5_history_manifest,
    write_exact28_m5_history_manifest,
)
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS


UTC = timezone.utc
RowMutator = Callable[[str, list[dict]], None]


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _row(pair: str, timestamp: datetime, *, bid: float = 1.1) -> dict:
    ask = bid + 0.0002
    return {
        "ask": {"c": ask, "h": ask, "l": ask, "o": ask},
        "bid": {"c": bid, "h": bid, "l": bid, "o": bid},
        "complete": True,
        "granularity": "M5",
        "pair": pair,
        "price": "BA",
        "time": timestamp.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
        "volume": 1,
    }


def _make_exact28_root(
    root: Path,
    *,
    row_mutator: RowMutator | None = None,
    fetch_script_path: Path | None = None,
) -> tuple[datetime, datetime, Path]:
    time_from = datetime(2020, 1, 1, tzinfo=UTC)
    time_to = time_from + timedelta(minutes=15)
    run_id = "20260717T010000Z"
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    fetch_script = fetch_script_path or (
        Path(__file__).resolve().parents[1] / "scripts" / "oanda_history_fetch.py"
    )
    fetch_script = fetch_script.resolve(strict=True)
    fetch_sha = hashlib.sha256(fetch_script.read_bytes()).hexdigest()
    recorded_at = _iso(datetime.now(UTC))
    previous_receipt_sha: str | None = None
    receipt_rows: list[dict] = []
    tasks: list[dict] = []

    for pair_index, pair in enumerate(DEFAULT_TRADER_PAIRS):
        rows = [
            _row(
                pair,
                time_from + timedelta(minutes=5 * offset),
                bid=1.1 + pair_index / 10_000 + offset / 100_000,
            )
            for offset in range(3)
        ]
        if row_mutator is not None:
            row_mutator(pair, rows)
        pair_dir = run_dir / pair
        pair_dir.mkdir()
        name = (
            f"{pair}_M5_BA_{time_from.strftime('%Y%m%dT%H%M%SZ')}_"
            f"{time_to.strftime('%Y%m%dT%H%M%SZ')}.jsonl.gz"
        )
        candle_path = pair_dir / name
        with gzip.open(candle_path, "wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        receipt_body = {
            "schema_version": "QR_OANDA_TRUTH_ACQUISITION_RECEIPT_V1",
            "sequence": len(receipt_rows) + 1,
            "recorded_at_utc": recorded_at,
            "output_root": str(root.resolve()),
            "candle_path": str(candle_path.resolve()),
            "candle_sha256": hashlib.sha256(candle_path.read_bytes()).hexdigest(),
            "pair": pair,
            "granularity": "M5",
            "price_component": "BA",
            "window": {"from_utc": _iso(time_from), "to_utc": _iso(time_to)},
            "rows": len(rows),
            "fetch_script_path": str(fetch_script),
            "fetch_script_sha256": fetch_sha,
            "previous_receipt_sha256": previous_receipt_sha,
        }
        receipt = {
            **receipt_body,
            "receipt_sha256": historical_s5._canonical_sha(receipt_body),
        }
        previous_receipt_sha = receipt["receipt_sha256"]
        receipt_rows.append(receipt)
        tasks.append(
            {
                "compressed": True,
                "dry_run": False,
                "errors": [],
                "from": _iso(time_from),
                "granularity": "M5",
                "pair": pair,
                "partial_path": None,
                "path": str(candle_path),
                "price": "BA",
                "published": True,
                "requests": 3,
                "rows": len(rows),
                "to": _iso(time_to),
                "truth_acquisition_receipt_sha256": receipt["receipt_sha256"],
                "windows": 3,
            }
        )

    receipt_path = root / "truth_acquisition_receipts.jsonl"
    receipt_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in receipt_rows
        ),
        encoding="utf-8",
    )
    summary = {
        "dry_run": False,
        "errors": [],
        "generated_at_utc": recorded_at,
        "granularities": ["M5"],
        "max_candles_per_request": 2,
        "output_dir": str(run_dir),
        "pairs": list(DEFAULT_TRADER_PAIRS),
        "price": "BA",
        "tasks": tasks,
        "total_requests": 3 * len(tasks),
        "total_rows": sum(task["rows"] for task in tasks),
        "window": {"from": _iso(time_from), "to": _iso(time_to)},
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return time_from, time_to, summary_path


class Exact28M5HistoryManifestTest(unittest.TestCase):
    def test_default_period_is_the_pre_holdout_2020_2026_scope(self) -> None:
        self.assertEqual(DEFAULT_PERIOD_FROM_UTC, datetime(2020, 1, 1, tzinfo=UTC))
        self.assertEqual(DEFAULT_PERIOD_TO_UTC, datetime(2026, 7, 10, tzinfo=UTC))
        shards = _expected_shards(DEFAULT_PERIOD_FROM_UTC, DEFAULT_PERIOD_TO_UTC)
        self.assertEqual([shard.shard_id for shard in shards], [str(year) for year in range(2020, 2027)])
        self.assertFalse(any(shard.terminal_partial_year for shard in shards[:-1]))
        self.assertTrue(shards[-1].terminal_partial_year)

    def test_builds_compact_exact28_receipted_manifest_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "history"
            root.mkdir()
            time_from, time_to, _summary = _make_exact28_root(root)

            manifest = build_exact28_m5_history_manifest(
                root,
                period_from_utc=time_from,
                period_to_utc=time_to,
            )

            self.assertEqual(manifest["expected_pair_count"], 28)
            self.assertEqual(manifest["expected_shard_count"], 1)
            self.assertEqual(manifest["selected_pair_shard_count"], 28)
            self.assertEqual(
                [(row["pair"], row["shard_id"]) for row in manifest["sources"]],
                [(pair, "2020") for pair in DEFAULT_TRADER_PAIRS],
            )
            self.assertTrue(manifest["complete_exact28_annual_shard_coverage"])
            self.assertTrue(manifest["receipt_ledger"]["hash_chain_proved"])
            self.assertTrue(manifest["source_root_clean"])
            self.assertFalse(manifest["raw_candles_embedded"])
            self.assertFalse(manifest["live_permission"])
            self.assertFalse(manifest["broker_mutation_allowed"])
            self.assertEqual(manifest["order_authority"], "NONE")
            self.assertFalse(
                manifest["endpoint_identity"]["response_top_level_instrument_receipted"]
            )
            self.assertTrue(
                all("candles" not in source and "rows" not in source for source in manifest["sources"])
            )

            manifest_path = Path(tmp) / "manifest.json"
            write_exact28_m5_history_manifest(manifest_path, manifest)
            self.assertEqual(load_exact28_m5_history_manifest(manifest_path), manifest)

    def test_rejects_summary_that_is_not_exact_configured_28(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "history"
            root.mkdir()
            time_from, time_to, summary_path = _make_exact28_root(root)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["pairs"] = summary["pairs"][:-1]
            summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")

            with self.assertRaisesRegex(HistoricalM5ManifestError, "exact configured 28"):
                build_exact28_m5_history_manifest(
                    root,
                    period_from_utc=time_from,
                    period_to_utc=time_to,
                )

    def test_rejects_temporary_or_partial_debris_before_admission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "history"
            root.mkdir()
            time_from, time_to, _summary = _make_exact28_root(root)
            (root / "abandoned.jsonl.gz.partial").write_bytes(b"partial")

            with self.assertRaisesRegex(HistoricalM5ManifestError, "temporary/partial"):
                build_exact28_m5_history_manifest(
                    root,
                    period_from_utc=time_from,
                    period_to_utc=time_to,
                )

    def test_rejects_off_grid_m5_row_even_when_summary_and_receipt_match(self) -> None:
        def mutate(pair: str, rows: list[dict]) -> None:
            if pair == DEFAULT_TRADER_PAIRS[0]:
                rows[1]["time"] = "2020-01-01T00:04:59.000000000Z"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "history"
            root.mkdir()
            time_from, time_to, _summary = _make_exact28_root(root, row_mutator=mutate)

            with self.assertRaisesRegex(HistoricalM5ManifestError, "off-grid"):
                build_exact28_m5_history_manifest(
                    root,
                    period_from_utc=time_from,
                    period_to_utc=time_to,
                )

    def test_rejects_duplicate_timestamp_even_when_summary_and_receipt_match(
        self,
    ) -> None:
        def mutate(pair: str, rows: list[dict]) -> None:
            if pair == DEFAULT_TRADER_PAIRS[0]:
                rows[1]["time"] = rows[0]["time"]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "history"
            root.mkdir()
            time_from, time_to, _summary = _make_exact28_root(root, row_mutator=mutate)

            with self.assertRaisesRegex(
                HistoricalM5ManifestError, "strictly increasing and unique"
            ):
                build_exact28_m5_history_manifest(
                    root,
                    period_from_utc=time_from,
                    period_to_utc=time_to,
                )

    def test_rejects_row_landing_exactly_on_the_exclusive_shard_end(self) -> None:
        # OANDA may return a candle opening exactly at the request `to`;
        # the half-open shard contract must refuse it instead of admitting
        # a row that also belongs to the next shard.
        def mutate(pair: str, rows: list[dict]) -> None:
            if pair == DEFAULT_TRADER_PAIRS[0]:
                boundary = datetime(2020, 1, 1, 0, 15, tzinfo=UTC)
                rows[2]["time"] = boundary.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "history"
            root.mkdir()
            time_from, time_to, _summary = _make_exact28_root(root, row_mutator=mutate)

            with self.assertRaisesRegex(
                HistoricalM5ManifestError, "outside its exact half-open shard"
            ):
                build_exact28_m5_history_manifest(
                    root,
                    period_from_utc=time_from,
                    period_to_utc=time_to,
                )

    def test_rejects_implausibly_low_internal_slot_coverage(self) -> None:
        def mutate(pair: str, rows: list[dict]) -> None:
            if pair == DEFAULT_TRADER_PAIRS[0]:
                del rows[1:]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "history"
            root.mkdir()
            time_from, time_to, _summary = _make_exact28_root(root, row_mutator=mutate)

            with self.assertRaisesRegex(
                HistoricalM5ManifestError, "coverage is implausibly low"
            ):
                build_exact28_m5_history_manifest(
                    root,
                    period_from_utc=time_from,
                    period_to_utc=time_to,
                )

    def test_rejects_truncated_gzip_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "history"
            root.mkdir()
            time_from, time_to, _summary = _make_exact28_root(root)
            victim = next(
                (root / "20260717T010000Z" / DEFAULT_TRADER_PAIRS[0]).glob(
                    "*.jsonl.gz"
                )
            )
            payload = victim.read_bytes()
            victim.write_bytes(payload[: len(payload) // 2])

            with self.assertRaisesRegex(
                HistoricalM5ManifestError, "gzip stream is invalid"
            ):
                build_exact28_m5_history_manifest(
                    root,
                    period_from_utc=time_from,
                    period_to_utc=time_to,
                )

    def test_rejects_crossed_high_low_even_when_open_close_are_passive(self) -> None:
        def mutate(pair: str, rows: list[dict]) -> None:
            if pair == DEFAULT_TRADER_PAIRS[0]:
                # Open/close stay uncrossed; only the bid high escapes above
                # the ask high, which is impossible if bid<=ask at every tick.
                rows[1]["bid"]["h"] = rows[1]["ask"]["h"] + 0.0001

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "history"
            root.mkdir()
            time_from, time_to, _summary = _make_exact28_root(root, row_mutator=mutate)

            with self.assertRaisesRegex(HistoricalM5ManifestError, "crossed"):
                build_exact28_m5_history_manifest(
                    root,
                    period_from_utc=time_from,
                    period_to_utc=time_to,
                )

    def test_rejects_incomplete_request_window_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "history"
            root.mkdir()
            time_from, time_to, summary_path = _make_exact28_root(root)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["tasks"][0]["requests"] = 2
            summary["total_requests"] -= 1
            summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")

            with self.assertRaisesRegex(HistoricalM5ManifestError, "request/window"):
                build_exact28_m5_history_manifest(
                    root,
                    period_from_utc=time_from,
                    period_to_utc=time_to,
                )

    def test_accepts_staged_fetch_script_copy_with_identical_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            original = Path(__file__).resolve().parents[1] / "scripts/oanda_history_fetch.py"
            staged_dir = base / "staged"
            staged_dir.mkdir()
            staged = staged_dir / "oanda_history_fetch.py"
            shutil.copyfile(original, staged)
            root = base / "history"
            root.mkdir()
            time_from, time_to, _summary = _make_exact28_root(
                root,
                fetch_script_path=staged,
            )

            manifest = build_exact28_m5_history_manifest(
                root,
                period_from_utc=time_from,
                period_to_utc=time_to,
            )

            self.assertTrue(manifest["complete_exact28_annual_shard_coverage"])
            self.assertEqual(
                manifest["endpoint_identity"]["fetch_script_sha256"],
                hashlib.sha256(original.read_bytes()).hexdigest(),
            )

    def test_rejects_staged_fetch_script_with_different_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            original = Path(__file__).resolve().parents[1] / "scripts/oanda_history_fetch.py"
            staged_dir = base / "staged"
            staged_dir.mkdir()
            staged = staged_dir / "oanda_history_fetch.py"
            staged.write_bytes(original.read_bytes() + b"\n# modified staged fetcher\n")
            root = base / "history"
            root.mkdir()
            time_from, time_to, _summary = _make_exact28_root(
                root,
                fetch_script_path=staged,
            )

            with self.assertRaisesRegex(HistoricalM5ManifestError, "digest drifted"):
                build_exact28_m5_history_manifest(
                    root,
                    period_from_utc=time_from,
                    period_to_utc=time_to,
                )

    def test_resealed_authority_tamper_still_fails_semantic_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "history"
            root.mkdir()
            time_from, time_to, _summary = _make_exact28_root(root)
            manifest = build_exact28_m5_history_manifest(
                root,
                period_from_utc=time_from,
                period_to_utc=time_to,
            )
            tampered = copy.deepcopy(manifest)
            tampered["live_permission"] = True
            body = {
                key: value
                for key, value in tampered.items()
                if key != "manifest_sha256"
            }
            tampered["manifest_sha256"] = historical_s5._canonical_sha(body)

            with self.assertRaisesRegex(HistoricalM5ManifestError, "unsafe flag"):
                validate_exact28_m5_history_manifest(tampered)

    def test_manifest_output_inside_source_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "history"
            root.mkdir()
            time_from, time_to, _summary = _make_exact28_root(root)
            manifest = build_exact28_m5_history_manifest(
                root,
                period_from_utc=time_from,
                period_to_utc=time_to,
            )

            with self.assertRaisesRegex(HistoricalM5ManifestError, "outside"):
                write_exact28_m5_history_manifest(root / "manifest.json", manifest)


if __name__ == "__main__":
    unittest.main()
