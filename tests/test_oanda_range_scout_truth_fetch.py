from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "oanda_range_scout_truth_fetch.py"
    )
    spec = importlib.util.spec_from_file_location("oanda_range_scout_truth_fetch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


truth_fetch = _load_module()


class _PayloadResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class OandaRangeScoutTruthFetchTest(unittest.TestCase):
    def _task(self):
        start = datetime(2026, 6, 1, tzinfo=timezone.utc)
        return truth_fetch.FetchTask(
            pair="EUR_USD",
            granularity="S5",
            start=start,
            end=start + timedelta(seconds=5),
            price="BA",
        )

    def _client(self, *, base_url: str | None = None):
        return truth_fetch.OandaReadOnlyClient(
            token="test-token",
            account_id="test-account",
            base_url=base_url or truth_fetch.PRODUCTION_OANDA_BASE_URL,
            env_file=Path("/definitely/missing/qr-test-env"),
        )

    def _fetch(self, payload: object, root: Path):
        with mock.patch.object(
            truth_fetch.oanda_module.urllib.request,
            "urlopen",
            return_value=_PayloadResponse(payload),
        ):
            return truth_fetch._fetch_task(
                self._client(),
                self._task(),
                run_dir=root / "run",
                receipt_root=root,
                max_candles_per_request=2,
                sleep_seconds=0.0,
                retries=1,
                compress=False,
                dry_run=False,
            )

    def test_default_window_stops_at_latest_completed_s5_boundary(self) -> None:
        args = truth_fetch.argparse.Namespace(
            time_from=None,
            time_to=None,
            days=1.0,
        )

        start, end = truth_fetch._resolve_acquisition_window(args)
        observed_after = datetime.now(timezone.utc)

        self.assertEqual(start.microsecond, 0)
        self.assertEqual(end.microsecond, 0)
        self.assertEqual(start.timestamp() % 5, 0)
        self.assertEqual(end.timestamp() % 5, 0)
        self.assertLessEqual(end, observed_after)
        self.assertLess((observed_after - end).total_seconds(), 5.1)
        self.assertEqual(end - start, timedelta(days=1))

    def test_help_runs_without_pythonpath(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        script = repo / "scripts/oanda_range_scout_truth_fetch.py"
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=repo,
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("RANGE scout replay", result.stdout)

    def test_pythonpath_shadow_cannot_replace_canonical_oanda_module(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        script = repo / "scripts/oanda_range_scout_truth_fetch.py"
        with tempfile.TemporaryDirectory() as tmp:
            evil_root = Path(tmp) / "evil"
            evil_broker = evil_root / "quant_rabbit" / "broker"
            evil_broker.mkdir(parents=True)
            (evil_root / "quant_rabbit" / "__init__.py").write_text(
                "", encoding="utf-8"
            )
            (evil_broker / "__init__.py").write_text("", encoding="utf-8")
            (evil_broker / "oanda.py").write_text(
                'raise RuntimeError("PYTHONPATH shadow loaded")\n',
                encoding="utf-8",
            )
            env = os.environ.copy()
            # Reproduce the vulnerable ordering: an untrusted package precedes
            # the canonical src root, while that canonical root is already in
            # sys.path and therefore would not have been moved to the front.
            env["PYTHONPATH"] = os.pathsep.join(
                (str(evil_root), str(repo / "src"))
            )
            result = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=repo,
                capture_output=True,
                text=True,
                env=env,
                timeout=10,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("PYTHONPATH shadow loaded", result.stderr)

    def test_invalid_success_payloads_are_partial_and_never_receipted(self) -> None:
        valid_candle = {
            "time": "2026-06-01T00:00:00Z",
            "complete": True,
            "volume": 1,
            "bid": {"o": "1.0", "h": "1.1", "l": "0.9", "c": "1.0"},
            "ask": {"o": "1.1", "h": "1.2", "l": "1.0", "c": "1.1"},
        }
        payloads = {
            "wrong_instrument": {
                "instrument": "GBP_USD",
                "granularity": "S5",
                "candles": [],
            },
            "wrong_granularity": {
                "instrument": "EUR_USD",
                "granularity": "M1",
                "candles": [],
            },
            "missing_candles": {"instrument": "EUR_USD", "granularity": "S5"},
            "incomplete_candle": {
                "instrument": "EUR_USD",
                "granularity": "S5",
                "candles": [{**valid_candle, "complete": False}],
            },
            "malformed_candle": {
                "instrument": "EUR_USD",
                "granularity": "S5",
                "candles": [{key: value for key, value in valid_candle.items() if key != "ask"}],
            },
        }
        for label, payload in payloads.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                summary = self._fetch(payload, root)

                self.assertFalse(summary["published"])
                self.assertTrue(summary["errors"])
                self.assertIsNotNone(summary["partial_path"])
                self.assertFalse(
                    (root / truth_fetch.RANGE_TRUTH_RECEIPT_FILE).exists()
                )

    def test_adapter_rejects_cross_request_duplicate_and_conflicting_timestamp(self) -> None:
        candle = {
            "time": "2026-06-01T00:00:00Z",
            "complete": True,
            "volume": 1,
            "bid": {"o": "1.0", "h": "1.1", "l": "0.9", "c": "1.0"},
            "ask": {"o": "1.1", "h": "1.2", "l": "1.0", "c": "1.1"},
        }
        query = {
            "granularity": "S5",
            "from": "2026-06-01T00:00:00Z",
            "to": "2026-06-01T00:00:05Z",
            "price": "BA",
            "includeFirst": "true",
        }
        path = "/v3/instruments/EUR_USD/candles"
        for label, repeated in {
            "duplicate": candle,
            "conflict": {
                **candle,
                "bid": {**candle["bid"], "c": "1.01"},
            },
        }.items():
            with self.subTest(label=label):
                adapter = truth_fetch._ValidatedCandleClient(
                    self._client(),
                    task=self._task(),
                )
                payloads = (
                    {
                        "instrument": "EUR_USD",
                        "granularity": "S5",
                        "candles": [candle],
                    },
                    {
                        "instrument": "EUR_USD",
                        "granularity": "S5",
                        "candles": [repeated],
                    },
                )
                with mock.patch.object(
                    truth_fetch.oanda_module.urllib.request,
                    "urlopen",
                    side_effect=[_PayloadResponse(item) for item in payloads],
                ):
                    adapter.get_json(path, query)
                    with self.assertRaises(ValueError):
                        adapter.get_json(path, query)

    def test_correct_metadata_with_empty_candles_is_published_and_receipted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            summary = self._fetch(
                {
                    "instrument": "EUR_USD",
                    "granularity": "S5",
                    "candles": [],
                },
                root,
            )

            self.assertTrue(summary["published"])
            self.assertEqual(summary["rows"], 0)
            self.assertEqual(summary["errors"], [])
            receipts = truth_fetch._validate_range_truth_receipt_chain(
                (root / truth_fetch.RANGE_TRUTH_RECEIPT_FILE).read_bytes()
            )
            self.assertEqual(receipts[0]["rows"], 0)
            self.assertEqual(
                receipts[0]["dependencies"],
                truth_fetch.expected_dependency_records(),
            )
            self.assertEqual(
                receipts[0]["source_base_url"],
                truth_fetch.PRODUCTION_OANDA_BASE_URL,
            )

    def test_receipted_network_request_is_exact_get_without_body(self) -> None:
        observed = []

        def respond(request, *, timeout):
            observed.append((request, timeout))
            return _PayloadResponse(
                {
                    "instrument": "EUR_USD",
                    "granularity": "S5",
                    "candles": [],
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with mock.patch.object(
                truth_fetch.oanda_module.urllib.request,
                "urlopen",
                side_effect=respond,
            ):
                summary = truth_fetch._fetch_task(
                    self._client(),
                    self._task(),
                    run_dir=root / "run",
                    receipt_root=root,
                    max_candles_per_request=2,
                    sleep_seconds=0.0,
                    retries=1,
                    compress=False,
                    dry_run=False,
                )

        self.assertTrue(summary["published"])
        self.assertEqual(len(observed), 1)
        request, timeout = observed[0]
        self.assertEqual(request.get_method(), "GET")
        self.assertIsNone(request.data)
        self.assertEqual(timeout, self._client().http_timeout_seconds)
        self.assertTrue(
            request.full_url.startswith(
                "https://api-fxtrade.oanda.com/v3/instruments/EUR_USD/candles?"
            )
        )

    def test_noncanonical_pair_is_rejected_before_network_or_file_write(self) -> None:
        task = truth_fetch.FetchTask(
            pair="../EUR_USD",
            granularity="S5",
            start=self._task().start,
            end=self._task().end,
            price="BA",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with mock.patch.object(
                truth_fetch.oanda_module.urllib.request,
                "urlopen",
            ) as urlopen:
                with self.assertRaisesRegex(ValueError, "canonical FX instrument"):
                    truth_fetch._fetch_task(
                        self._client(),
                        task,
                        run_dir=root / "run",
                        receipt_root=root,
                        max_candles_per_request=2,
                        sleep_seconds=0.0,
                        retries=1,
                        compress=False,
                        dry_run=False,
                    )

            urlopen.assert_not_called()
            self.assertFalse((root / "run").exists())

    def test_receipt_path_rejects_noncanonical_or_nonproduction_client(self) -> None:
        class FakeClient:
            base_url = truth_fetch.PRODUCTION_OANDA_BASE_URL

            def get_json(self, _path, _query):
                return {}

        clients = {
            "fake": FakeClient(),
            "nonproduction": self._client(base_url="https://example.invalid"),
            "instance_override": self._client(),
        }
        clients["instance_override"].get_json = lambda _path, _query: {}
        for label, client in clients.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                with self.assertRaises(ValueError):
                    truth_fetch._fetch_task(
                        client,
                        self._task(),
                        run_dir=root / "run",
                        receipt_root=root,
                        max_candles_per_request=2,
                        sleep_seconds=0.0,
                        retries=1,
                        compress=False,
                        dry_run=False,
                    )
                self.assertFalse(
                    (root / truth_fetch.RANGE_TRUTH_RECEIPT_FILE).exists()
                )

    def test_future_window_and_oversized_request_are_rejected_before_fetch(self) -> None:
        future_start = datetime.now(timezone.utc) + timedelta(minutes=1)
        future_task = truth_fetch.FetchTask(
            pair="EUR_USD",
            granularity="S5",
            start=future_start,
            end=future_start + timedelta(seconds=5),
            price="BA",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with self.assertRaises(ValueError):
                truth_fetch._fetch_task(
                    self._client(),
                    future_task,
                    run_dir=root / "future",
                    receipt_root=root,
                    max_candles_per_request=2,
                    sleep_seconds=0.0,
                    retries=1,
                    compress=False,
                    dry_run=False,
                )
            with self.assertRaises(ValueError):
                truth_fetch._fetch_task(
                    self._client(),
                    self._task(),
                    run_dir=root / "oversized",
                    receipt_root=root,
                    max_candles_per_request=(
                        truth_fetch.OANDA_MAX_CANDLES_PER_REQUEST + 1
                    ),
                    sleep_seconds=0.0,
                    retries=1,
                    compress=False,
                    dry_run=False,
                )
            self.assertFalse((root / truth_fetch.RANGE_TRUTH_RECEIPT_FILE).exists())

    def test_code_drift_after_fetch_demotes_file_to_partial_without_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with mock.patch.object(
                truth_fetch,
                "_code_snapshot_unchanged",
                return_value=False,
            ):
                summary = self._fetch(
                    {
                        "instrument": "EUR_USD",
                        "granularity": "S5",
                        "candles": [],
                    },
                    root,
                )

            self.assertFalse(summary["published"])
            self.assertTrue(summary["errors"])
            self.assertEqual(
                summary["errors"][-1]["error"],
                "acquisition_code_or_dependency_sha_drift",
            )
            self.assertFalse(Path(summary["path"]).exists())
            self.assertTrue(Path(summary["partial_path"]).exists())
            self.assertFalse((root / truth_fetch.RANGE_TRUTH_RECEIPT_FILE).exists())


if __name__ == "__main__":
    unittest.main()
