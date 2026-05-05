from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from quant_rabbit.broker.oanda import OandaReadOnlyClient
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


if __name__ == "__main__":
    unittest.main()
