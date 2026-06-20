from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from quant_rabbit.broker.webull import (
    WebullConfig,
    WebullStockOrder,
    WebullStockOrderGateway,
    WEBULL_ORDER_ID_PREFIX,
)
from quant_rabbit.cli import main


class WebullConfigTest(unittest.TestCase):
    def test_loads_only_qr_webull_credentials_from_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env.local"
            env_file.write_text(
                "\n".join(
                    [
                        "QR_WEBULL_APP_KEY=app-key",
                        "QR_WEBULL_APP_SECRET=app-secret",
                        "QR_WEBULL_ACCOUNT_ID=account-id",
                        "QR_WEBULL_ENV=production",
                        "QR_WEBULL_ENDPOINT=https://api.webull.com/",
                        "WEBULL_PASSWORD=must-not-load",
                    ]
                )
                + "\n"
            )

            with patch.dict("os.environ", {"QR_WEBULL_ENV_FILE": str(env_file)}, clear=True):
                config = WebullConfig.from_env()

        self.assertEqual(config.app_key, "app-key")
        self.assertEqual(config.app_secret, "app-secret")
        self.assertEqual(config.account_id, "account-id")
        self.assertEqual(config.environment, "production")
        self.assertEqual(config.resolved_endpoint, "api.webull.com")
        self.assertNotIn("must-not-load", json.dumps(config.safe_status()))

    def test_requires_app_key_and_secret_for_api_client(self) -> None:
        config = WebullConfig(app_key=None, app_secret=None)

        with self.assertRaisesRegex(RuntimeError, "QR_WEBULL_APP_KEY"):
            config.require_credentials()


class WebullStockOrderTest(unittest.TestCase):
    def test_limit_order_payload_uses_official_stock_fields(self) -> None:
        order = WebullStockOrder(symbol="aapl", side="BUY", quantity="1.0", limit_price="180.00")

        payload = order.to_webull_payload()

        self.assertEqual(payload["combo_type"], "NORMAL")
        self.assertTrue(payload["client_order_id"].startswith(WEBULL_ORDER_ID_PREFIX))
        self.assertLessEqual(len(payload["client_order_id"]), 32)
        self.assertEqual(payload["symbol"], "AAPL")
        self.assertEqual(payload["instrument_type"], "EQUITY")
        self.assertEqual(payload["market"], "US")
        self.assertEqual(payload["order_type"], "LIMIT")
        self.assertEqual(payload["quantity"], "1")
        self.assertEqual(payload["limit_price"], "180")
        self.assertEqual(payload["support_trading_session"], "CORE")
        self.assertEqual(payload["time_in_force"], "DAY")

    def test_limit_price_is_required_for_limit_orders(self) -> None:
        order = WebullStockOrder(symbol="AAPL", side="BUY", quantity="1", order_type="LIMIT")

        issues = order.validate()

        self.assertTrue(any(issue["code"] == "WEBULL_LIMIT_PRICE_REQUIRED" for issue in issues))
        with self.assertRaisesRegex(ValueError, "WEBULL_LIMIT_PRICE_REQUIRED"):
            order.to_webull_payload()


class WebullStockOrderGatewayTest(unittest.TestCase):
    def test_stages_without_credentials_when_preview_and_send_are_not_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "request.json"
            report = Path(tmp) / "report.md"
            with patch.dict("os.environ", {"QR_WEBULL_ENV_FILE": str(Path(tmp) / "missing.env")}, clear=True):
                summary = WebullStockOrderGateway(output_path=output, report_path=report).run(
                    order=WebullStockOrder(symbol="AAPL", side="BUY", quantity="1", limit_price="180"),
                )

            saved = json.loads(output.read_text())
            report_exists = report.exists()

        self.assertEqual(summary.status, "STAGED")
        self.assertFalse(summary.sent)
        self.assertEqual(saved["status"], "STAGED")
        self.assertEqual(saved["order_request"]["new_orders"][0]["symbol"], "AAPL")
        self.assertTrue(report_exists)

    def test_send_is_blocked_without_live_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "request.json"
            report = Path(tmp) / "report.md"
            with patch.dict(
                "os.environ",
                {
                    "QR_WEBULL_APP_KEY": "app-key",
                    "QR_WEBULL_APP_SECRET": "app-secret",
                    "QR_WEBULL_ACCOUNT_ID": "account-id",
                },
                clear=True,
            ):
                summary = WebullStockOrderGateway(output_path=output, report_path=report).run(
                    order=WebullStockOrder(symbol="AAPL", side="BUY", quantity="1", limit_price="180"),
                    send=True,
                    confirm_live=True,
                )

        self.assertEqual(summary.status, "BLOCKED")
        self.assertTrue(any(issue["code"] == "WEBULL_LIVE_DISABLED" for issue in summary.issues))


class WebullCliTest(unittest.TestCase):
    def test_env_check_does_not_print_secret_values(self) -> None:
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "env.json"
            report = Path(tmp) / "env.md"
            with patch.dict(
                "os.environ",
                {
                    "QR_WEBULL_APP_KEY": "visible-key-value",
                    "QR_WEBULL_APP_SECRET": "visible-secret-value",
                    "QR_WEBULL_ACCOUNT_ID": "visible-account-id",
                    "QR_WEBULL_ENV_FILE": str(Path(tmp) / "missing.env"),
                },
                clear=True,
            ), redirect_stdout(stdout):
                code = main(["webull-env-check", "--output", str(output), "--report", str(report)])

        self.assertEqual(code, 0)
        rendered = stdout.getvalue()
        self.assertNotIn("visible-key-value", rendered)
        self.assertNotIn("visible-secret-value", rendered)
        self.assertNotIn("visible-account-id", rendered)
        self.assertIn("app_key_present", rendered)

    def test_stage_stock_order_cli_stages_by_default(self) -> None:
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "order.json"
            report = Path(tmp) / "order.md"
            with patch.dict("os.environ", {"QR_WEBULL_ENV_FILE": str(Path(tmp) / "missing.env")}, clear=True), redirect_stdout(stdout):
                code = main(
                    [
                        "webull-stage-stock-order",
                        "--symbol",
                        "AAPL",
                        "--side",
                        "BUY",
                        "--quantity",
                        "1",
                        "--order-type",
                        "LIMIT",
                        "--limit-price",
                        "180",
                        "--output",
                        str(output),
                        "--report",
                        str(report),
                    ]
                )

            payload = json.loads(stdout.getvalue())
            output_exists = output.exists()
            report_exists = report.exists()

        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "STAGED")
        self.assertFalse(payload["sent"])
        self.assertTrue(output_exists)
        self.assertTrue(report_exists)
