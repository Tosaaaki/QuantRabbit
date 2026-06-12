from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from quant_rabbit.broker.oanda import DEFAULT_OANDA_HTTP_TIMEOUT_SECONDS, OandaReadOnlyClient
from quant_rabbit.cli import main


class OandaClientTest(unittest.TestCase):
    def test_requires_explicit_qr_oanda_credentials(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "OANDA_TOKEN": "legacy-token",
                "OANDA_ACCOUNT_ID": "legacy-account",
                "QR_OANDA_ENV_FILE": "/tmp/quant-rabbit-no-such-env",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "QR_OANDA_TOKEN"):
                OandaReadOnlyClient()

    def test_uses_qr_oanda_credentials(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "QR_OANDA_TOKEN": "qr-token",
                "QR_OANDA_ACCOUNT_ID": "qr-account",
                "QR_OANDA_BASE_URL": "https://example.invalid/",
            },
            clear=True,
        ):
            client = OandaReadOnlyClient()

        self.assertEqual(client.token, "qr-token")
        self.assertEqual(client.account_id, "qr-account")
        self.assertEqual(client.base_url, "https://example.invalid")

    def test_get_json_uses_named_http_timeout(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, _exc_type, _exc, _tb) -> None:
                return None

            def read(self) -> bytes:
                return b"{}"

        with patch.dict(
            "os.environ",
            {
                "QR_OANDA_TOKEN": "qr-token",
                "QR_OANDA_ACCOUNT_ID": "qr-account",
                "QR_OANDA_BASE_URL": "https://example.invalid/",
            },
            clear=True,
        ):
            client = OandaReadOnlyClient()

        with patch("quant_rabbit.broker.oanda.urllib.request.urlopen", return_value=FakeResponse()) as urlopen:
            client.get_json("/v3/test")

        self.assertEqual(urlopen.call_args.kwargs["timeout"], DEFAULT_OANDA_HTTP_TIMEOUT_SECONDS)

    def test_falls_back_to_project_env_local_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env.local"
            env_file.write_text(
                "\n".join(
                    [
                        "QR_OANDA_ACCOUNT_ID=qr-account-from-file",
                        "QR_OANDA_TOKEN=qr-token-from-file",
                        "QR_OANDA_BASE_URL=https://example.invalid/",
                    ]
                )
                + "\n"
            )

            with patch.dict("os.environ", {"QR_OANDA_ENV_FILE": str(env_file)}, clear=True):
                client = OandaReadOnlyClient()

        self.assertEqual(client.token, "qr-token-from-file")
        self.assertEqual(client.account_id, "qr-account-from-file")
        self.assertEqual(client.base_url, "https://example.invalid")

    def test_broker_snapshot_cli_reports_missing_qr_credentials_as_json(self) -> None:
        output = io.StringIO()
        with patch.dict("os.environ", {"QR_OANDA_ENV_FILE": "/tmp/quant-rabbit-no-such-env"}, clear=True), redirect_stdout(output):
            exit_code = main(["broker-snapshot"])

        self.assertEqual(exit_code, 2)
        payload = json.loads(output.getvalue())
        self.assertIn("QR_OANDA_TOKEN", payload["error"])


class OandaAccountSummaryTest(unittest.TestCase):
    def test_account_summary_parser_reads_oanda_payload(self) -> None:
        from datetime import datetime, timezone

        from quant_rabbit.broker.oanda import _account_summary_from_payload

        payload = {
            "account": {
                "NAV": "210266.0364",
                "balance": "210106.6870",
                "unrealizedPL": "159.3494",
                "marginUsed": "22042.32",
                "marginAvailable": "188242.8203",
                "pl": "-247578.3592",
                "financing": "-122416.9538",
                "lastTransactionID": "470126",
                "hedgingEnabled": True,
            }
        }
        now = datetime(2026, 5, 4, 4, 0, tzinfo=timezone.utc)

        summary = _account_summary_from_payload(payload, now_utc=now)

        self.assertAlmostEqual(summary.nav_jpy, 210266.0364)
        self.assertAlmostEqual(summary.balance_jpy, 210106.687)
        self.assertAlmostEqual(summary.unrealized_pl_jpy, 159.3494)
        self.assertAlmostEqual(summary.margin_used_jpy, 22042.32)
        self.assertEqual(summary.last_transaction_id, "470126")
        self.assertTrue(summary.hedging_enabled)
        self.assertEqual(summary.fetched_at_utc, now)

    def test_snapshot_continues_when_summary_call_fails(self) -> None:
        from datetime import datetime, timezone
        from unittest.mock import patch

        from quant_rabbit.models import BrokerSnapshot

        with patch.dict(
            "os.environ",
            {
                "QR_OANDA_TOKEN": "qr-token",
                "QR_OANDA_ACCOUNT_ID": "qr-account",
                "QR_OANDA_BASE_URL": "https://example.invalid/",
            },
            clear=True,
        ):
            client = OandaReadOnlyClient()

        def _raise_on_summary(path, query=None):
            if path.endswith("/summary"):
                raise RuntimeError("simulated summary outage")
            if path.endswith("/openTrades"):
                return {"trades": []}
            if path.endswith("/pendingOrders"):
                return {"orders": []}
            if "/pricing" in path:
                now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                return {
                    "prices": [{"instrument": "USD_JPY", "bids": [{"price": "157.0"}], "asks": [{"price": "157.01"}], "time": now}],
                    "homeConversions": [{"currency": "USD", "accountGain": "156.9", "accountLoss": "157.1", "positionValue": "157.0"}],
                }
            return {}

        with patch.object(client, "get_json", side_effect=_raise_on_summary):
            snapshot = client.snapshot(["USD_JPY"])

        self.assertIsInstance(snapshot, BrokerSnapshot)
        self.assertIsNone(snapshot.account)
        self.assertEqual(len(snapshot.quotes), 1)
        self.assertAlmostEqual(snapshot.home_conversions["USD"], 157.1)

    def test_snapshot_backfills_take_profit_from_pending_orders(self) -> None:
        # Regression for the 2026-05-13 EUR_USD 470960/470948 case where
        # openTrades returned takeProfitOrder=null but pendingOrders still
        # carried the TP order with tradeID matching the open position.
        # The position must come back protected (take_profit set) so
        # PositionManager does not queue a redundant PROTECT repair.
        from datetime import datetime, timezone
        from unittest.mock import patch

        with patch.dict(
            "os.environ",
            {
                "QR_OANDA_TOKEN": "qr-token",
                "QR_OANDA_ACCOUNT_ID": "qr-account",
                "QR_OANDA_BASE_URL": "https://example.invalid/",
            },
            clear=True,
        ):
            client = OandaReadOnlyClient()

        def _route(path, query=None):
            if path.endswith("/openTrades"):
                return {
                    "trades": [
                        {
                            "id": "470960",
                            "instrument": "EUR_USD",
                            "currentUnits": "-8000",
                            "price": "1.17340",
                            "unrealizedPL": "-630.0",
                            # takeProfitOrder intentionally absent
                        }
                    ]
                }
            if path.endswith("/pendingOrders"):
                return {
                    "orders": [
                        {
                            "id": "470967",
                            "type": "TAKE_PROFIT",
                            "tradeID": "470960",
                            "price": "1.17291",
                            "state": "PENDING",
                            "instrument": None,
                        }
                    ]
                }
            if "/pricing" in path:
                now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                return {
                    "prices": [
                        {
                            "instrument": "EUR_USD",
                            "bids": [{"price": "1.1733"}],
                            "asks": [{"price": "1.1734"}],
                            "time": now,
                        }
                    ],
                    "homeConversions": [],
                }
            if path.endswith("/summary"):
                return {
                    "account": {
                        "NAV": "207000",
                        "balance": "207000",
                        "marginUsed": "100000",
                        "marginAvailable": "107000",
                        "lastTransactionID": "470967",
                        "hedgingEnabled": True,
                    }
                }
            return {}

        with patch.object(client, "get_json", side_effect=_route):
            snapshot = client.snapshot(["EUR_USD"])

        self.assertEqual(len(snapshot.positions), 1)
        pos = snapshot.positions[0]
        self.assertEqual(pos.trade_id, "470960")
        # take_profit should be backfilled from the pending TP order.
        self.assertIsNotNone(pos.take_profit)
        self.assertAlmostEqual(pos.take_profit, 1.17291, places=5)
        # stop_loss must stay None — SL-free positions must not pick up
        # a stray pending SL via the same path.
        self.assertIsNone(pos.stop_loss)

    def test_snapshot_does_not_backfill_stop_loss(self) -> None:
        # SL-free design invariant: a position with stop_loss=None must
        # not be silently joined to a pending STOP_LOSS order even if one
        # exists with the matching tradeID. Only TP is backfilled.
        from datetime import datetime, timezone
        from unittest.mock import patch

        with patch.dict(
            "os.environ",
            {
                "QR_OANDA_TOKEN": "qr-token",
                "QR_OANDA_ACCOUNT_ID": "qr-account",
                "QR_OANDA_BASE_URL": "https://example.invalid/",
            },
            clear=True,
        ):
            client = OandaReadOnlyClient()

        def _route(path, query=None):
            if path.endswith("/openTrades"):
                return {
                    "trades": [
                        {
                            "id": "470960",
                            "instrument": "EUR_USD",
                            "currentUnits": "-8000",
                            "price": "1.17340",
                            "unrealizedPL": "-630.0",
                        }
                    ]
                }
            if path.endswith("/pendingOrders"):
                return {
                    "orders": [
                        {
                            "id": "470968",
                            "type": "STOP_LOSS",
                            "tradeID": "470960",
                            "price": "1.18000",
                            "state": "PENDING",
                            "instrument": None,
                        }
                    ]
                }
            if "/pricing" in path:
                now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                return {
                    "prices": [
                        {
                            "instrument": "EUR_USD",
                            "bids": [{"price": "1.1733"}],
                            "asks": [{"price": "1.1734"}],
                            "time": now,
                        }
                    ],
                    "homeConversions": [],
                }
            if path.endswith("/summary"):
                return {
                    "account": {
                        "NAV": "207000",
                        "balance": "207000",
                        "marginUsed": "100000",
                        "marginAvailable": "107000",
                        "lastTransactionID": "470968",
                        "hedgingEnabled": True,
                    }
                }
            return {}

        with patch.object(client, "get_json", side_effect=_route):
            snapshot = client.snapshot(["EUR_USD"])

        self.assertEqual(len(snapshot.positions), 1)
        pos = snapshot.positions[0]
        # SL stays None (SL-free invariant).
        self.assertIsNone(pos.stop_loss)
        # TP also stays None (no pending TP available).
        self.assertIsNone(pos.take_profit)


if __name__ == "__main__":
    unittest.main()
