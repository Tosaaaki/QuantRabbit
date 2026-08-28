from __future__ import annotations

import ast
import json
import os
import tempfile
import unittest
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from oanda_history_capture import (
    BAR_SECONDS,
    CONTRACT_PATH,
    HTTP_METHOD,
    LOOKBACK_DAYS,
    MAX_CANDLES_PER_GET,
    PRICE_COMPONENT,
    REQUEST_SPACING_SECONDS,
    REST_HOST,
    ROW_SCHEMA,
    SYMBOLS,
    CaptureError,
    RequestPacer,
    Window,
    _fetch_or_resume_window,
    _load_contract,
    _validate_run_receipt_chain,
    analyze_gaps,
    build_plan,
    capture,
    canonical_bytes,
    floor_completed_m5,
    parse_oanda_time,
    plan_document,
    sha256_bytes,
    utc_text,
    validate_payload,
    verify_run,
)


HERE = Path(__file__).resolve().parent


def candle(time_text: str, *, complete: bool = True, bid: str = "1.1000", ask: str = "1.1002") -> dict:
    return {
        "time": time_text,
        "complete": complete,
        "volume": 7,
        "bid": {"o": bid, "h": bid, "l": bid, "c": bid},
        "ask": {"o": ask, "h": ask, "l": ask, "c": ask},
    }


class FakeResponse:
    status = 200

    def __init__(self, payload: dict):
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload, sort_keys=True).encode("utf-8")


class FakeConnection:
    requests: list[tuple[str, str, dict]] = []

    def __init__(self, host: str, **_kwargs):
        self.host = host
        self.request_data: tuple[str, str, dict] | None = None

    def request(self, method: str, target: str, headers: dict) -> None:
        self.request_data = (method, target, headers)
        self.__class__.requests.append(self.request_data)

    def getresponse(self) -> FakeResponse:
        assert self.request_data is not None
        _method, target, _headers = self.request_data
        parsed = urllib.parse.urlsplit(target)
        query = urllib.parse.parse_qs(parsed.query)
        instrument = parsed.path.split("/instruments/", 1)[1].split("/", 1)[0]
        start = query["from"][0]
        return FakeResponse(
            {
                "instrument": instrument,
                "granularity": "M5",
                "candles": [candle(start)],
            }
        )

    def close(self) -> None:
        return None


class ForbiddenConnection:
    def __init__(self, *_args, **_kwargs):
        raise AssertionError("network must not be used for a resumed or published window")


class CaptureTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeConnection.requests = []
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        # Published runs are intentionally read-only. Restore test-owned modes
        # so TemporaryDirectory can remove only its own fixture tree.
        for path in sorted(self.root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if path.is_symlink():
                continue
            try:
                path.chmod(0o700 if path.is_dir() else 0o600)
            except FileNotFoundError:
                pass
        self.temp.cleanup()

    def test_contract_and_two_year_chunk_plan_are_exact(self) -> None:
        contract = _load_contract()
        self.assertEqual(contract["http_method_allowlist"], ["GET"])
        self.assertEqual(contract["symbols"], list(SYMBOLS))
        self.assertEqual(contract["lookback_days"], LOOKBACK_DAYS)
        self.assertEqual(contract["request_spacing_seconds"], REQUEST_SPACING_SECONDS)
        end = datetime(2026, 8, 28, 4, 0, tzinfo=timezone.utc)
        plan = build_plan(end)
        self.assertEqual(plan.end - plan.start, timedelta(days=730))
        self.assertEqual(plan_document(plan)["plan_sha256"], plan.plan_sha256)
        self.assertEqual({window.instrument for window in plan.windows}, set(SYMBOLS))
        self.assertTrue(
            all(
                0 < int((window.end - window.start).total_seconds()) // BAR_SECONDS
                < MAX_CANDLES_PER_GET
                for window in plan.windows
            )
        )
        for symbol in SYMBOLS:
            symbol_windows = [window for window in plan.windows if window.instrument == symbol]
            self.assertEqual(symbol_windows[0].start, plan.start)
            self.assertEqual(symbol_windows[-1].end, plan.end)
            self.assertTrue(all(left.end == right.start for left, right in zip(symbol_windows, symbol_windows[1:])))

    def test_completed_bid_ask_validation_and_fail_closed_cases(self) -> None:
        start = datetime(2026, 8, 28, 0, 0, tzinfo=timezone.utc)
        window = Window("EUR_USD", 0, start, start + timedelta(minutes=10))
        payload = {
            "instrument": "EUR_USD",
            "granularity": "M5",
            "candles": [candle("2026-08-28T00:00:00.000000000Z")],
        }
        rows = validate_payload(payload, window)
        self.assertEqual(rows[0]["schema"], ROW_SCHEMA)
        self.assertEqual(rows[0]["bid"]["o"], "1.1")
        self.assertEqual(rows[0]["ask"]["o"], "1.1002")
        self.assertEqual(rows[0]["volume_semantics"], "OANDA_PRICE_COUNT_NOT_TRADED_VOLUME")
        for mutation in (
            {**payload, "candles": [candle("2026-08-28T00:00:00Z", complete=False)]},
            {**payload, "candles": [{**candle("2026-08-28T00:00:00Z"), "ask": None}]},
            {**payload, "candles": [candle("2026-08-28T00:10:00Z")]},
            {**payload, "candles": [candle("2026-08-28T00:01:00Z")]},
            {**payload, "candles": [candle("2026-08-28T00:00:00Z", bid="nan")]},
            {**payload, "candles": [candle("2026-08-28T00:00:00Z", bid="1.2", ask="1.1")]},
        ):
            with self.subTest(mutation=mutation):
                with self.assertRaises(CaptureError):
                    validate_payload(mutation, window)

    def test_window_cache_resumes_without_a_second_request(self) -> None:
        partial = self.root / "one.partial"
        start = datetime(2026, 8, 28, 0, 0, tzinfo=timezone.utc)
        window = Window("EUR_USD", 0, start, start + timedelta(minutes=10))
        attempts = []
        rows, meta, reused = _fetch_or_resume_window(
            "fixture-account",
            "fixture-token",
            partial,
            window,
            connection_factory=FakeConnection,
            pacer=RequestPacer(lambda _seconds: None),
            retries=1,
            progress_callback=lambda: attempts.append(1),
        )
        self.assertFalse(reused)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(meta["credential_values_persisted"], 0)
        rows2, meta2, reused2 = _fetch_or_resume_window(
            "different-account-that-must-not-be-used",
            "different-token-that-must-not-be-used",
            partial,
            window,
            connection_factory=ForbiddenConnection,
            pacer=RequestPacer(lambda _seconds: None),
            retries=1,
            progress_callback=lambda: attempts.append(1),
        )
        self.assertTrue(reused2)
        self.assertEqual(rows2, rows)
        self.assertEqual(meta2, meta)
        self.assertEqual(len(attempts), 1)

    def test_gap_classes_are_separate_and_never_synthesize_prices(self) -> None:
        start = datetime(2026, 12, 25, 0, 0, tzinfo=timezone.utc)
        end = start + timedelta(days=2, minutes=5)
        rows = [
            {
                "time_utc": utc_text(start + timedelta(days=2)),
            }
        ]
        report = analyze_gaps(rows, start, end)
        counts = report["missing_slot_counts"]
        self.assertGreater(counts["KNOWN_HOLIDAY"], 0)
        self.assertGreater(counts["WEEKEND_CLOSED"], 0)
        self.assertGreaterEqual(counts["UNEXPLAINED_WEEKDAY"], 0)
        self.assertEqual(report["missing_prices_synthesized"], 0)
        self.assertEqual(report["expected_grid_slots"], report["observed_rows"] + report["missing_slots"])

    def test_full_fixture_publish_is_atomic_immutable_idempotent_and_secret_free(self) -> None:
        end = datetime(2026, 8, 28, 4, 0, tzinfo=timezone.utc)
        plan = build_plan(end)
        result = capture(
            "fixture-account-sensitive",
            "fixture-token-sensitive",
            self.root / "runs",
            plan,
            connection_factory=FakeConnection,
            sleeper=lambda _seconds: None,
            retries=1,
        )
        self.assertEqual(result["status"], "PUBLISHED")
        final = self.root / "runs" / plan.run_id
        self.assertTrue(final.is_dir())
        self.assertFalse((self.root / "runs" / f"{plan.run_id}.partial").exists())
        manifest = verify_run(final)
        self.assertEqual(manifest["successful_windows"], len(plan.windows))
        self.assertEqual(manifest["network_attempts"], len(plan.windows))
        self.assertEqual(manifest["credential_values_persisted"], 0)
        self.assertEqual(manifest["external_orders"], 0)
        self.assertFalse(manifest["forward_pnl_included"])
        first_request = FakeConnection.requests[0]
        self.assertEqual(first_request[0], HTTP_METHOD)
        self.assertIn("from=", first_request[1])
        self.assertIn("to=", first_request[1])
        self.assertIn("price=BA", first_request[1])
        request_count = len(FakeConnection.requests)
        second = capture(
            "unused-account-sensitive",
            "unused-token-sensitive",
            self.root / "runs",
            plan,
            connection_factory=ForbiddenConnection,
            sleeper=lambda _seconds: None,
            retries=1,
        )
        self.assertEqual(second["status"], "VERIFIED_EXISTING")
        self.assertEqual(len(FakeConnection.requests), request_count)
        receipt_rows = _validate_run_receipt_chain((self.root / "runs" / "run_receipts.jsonl").read_bytes())
        self.assertEqual(len(receipt_rows), 1)
        secret_needles = (
            b"fixture-account-sensitive",
            b"fixture-token-sensitive",
            b"unused-account-sensitive",
            b"unused-token-sensitive",
        )
        for path in (self.root / "runs").rglob("*"):
            if path.is_file() and not path.is_symlink():
                payload = path.read_bytes()
                self.assertTrue(all(needle not in payload for needle in secret_needles), path)
        self.assertEqual(final.stat().st_mode & 0o222, 0)
        self.assertTrue(all(path.stat().st_mode & 0o222 == 0 for path in final.rglob("*") if not path.is_symlink()))

    def test_tamper_is_rejected(self) -> None:
        end = datetime(2026, 8, 28, 4, 0, tzinfo=timezone.utc)
        plan = build_plan(end)
        result = capture(
            "fixture-account",
            "fixture-token",
            self.root / "runs",
            plan,
            connection_factory=FakeConnection,
            sleeper=lambda _seconds: None,
            retries=1,
        )
        final = self.root / "runs" / result["manifest"]["run_id"]
        data_path = final / result["manifest"]["files"][0]["path"]
        data_path.chmod(0o600)
        with data_path.open("ab") as handle:
            handle.write(b"{}\n")
        with self.assertRaises(CaptureError):
            verify_run(final)

    def test_source_has_only_get_surface_and_imports_approved_loader(self) -> None:
        source = (HERE / "oanda_history_capture.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        http_literals = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in {"GET", "POST", "PUT", "PATCH", "DELETE"}
        }
        self.assertEqual(http_literals, {"GET"})
        lowered = source.lower()
        for forbidden_path in ("/orders", "/trades", "/positions"):
            self.assertNotIn(forbidden_path, lowered)
        self.assertNotIn("oandareadonlyclient", lowered)
        self.assertIn("from oanda_live_feed import load_approved_live_credentials", source)
        self.assertEqual(REST_HOST, "https://api-fxtrade.oanda.com")
        self.assertEqual(PRICE_COMPONENT, "BA")
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        self.assertFalse(contract["live_order_authority"])
        self.assertEqual(contract["external_orders"], 0)

    def test_nanosecond_parser_and_completed_boundary(self) -> None:
        parsed = parse_oanda_time("2026-08-28T00:00:00.123456789Z")
        self.assertEqual(parsed.microsecond, 123456)
        value = datetime(2026, 8, 28, 0, 7, 59, tzinfo=timezone.utc)
        self.assertEqual(floor_completed_m5(value), datetime(2026, 8, 28, 0, 5, tzinfo=timezone.utc))
        self.assertEqual(sha256_bytes(canonical_bytes({"a": 1})), sha256_bytes(b'{"a":1}'))


if __name__ == "__main__":
    unittest.main()
