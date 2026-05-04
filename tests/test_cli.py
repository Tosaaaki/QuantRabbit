from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from quant_rabbit.cli import main


class CliHelpTest(unittest.TestCase):
    def test_top_level_help_renders_daily_target_percent_text(self) -> None:
        stdout = io.StringIO()

        with self.assertRaises(SystemExit) as raised, redirect_stdout(stdout):
            main(["--help"])

        self.assertEqual(raised.exception.code, 0)
        help_text = stdout.getvalue()
        self.assertIn("daily-target-state", help_text)
        self.assertIn("10% target", help_text)

    def test_autotrade_missing_gpt_decision_response_returns_json_error(self) -> None:
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            code = main(
                [
                    "autotrade-cycle",
                    "--use-gpt-trader",
                    "--gpt-decision-response",
                    "/tmp/qr-missing-gpt-decision-response.json",
                ]
            )

        self.assertEqual(code, 2)
        payload = json.loads(stdout.getvalue())
        self.assertIn("qr-missing-gpt-decision-response.json", payload["error"])

    def test_gpt_trader_requires_codex_decision_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot.json"
            snapshot.write_text(json.dumps({"positions": [], "orders": [], "quotes": {}}))
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main(["gpt-trader-decision", "--snapshot", str(snapshot)])

        self.assertEqual(code, 2)
        payload = json.loads(stdout.getvalue())
        self.assertIn("--decision-response", payload["error"])

    def test_replay_execution_missing_prices_returns_json_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            intents.write_text(json.dumps({"results": []}))
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main(
                    [
                        "replay-execution",
                        "--intents",
                        str(intents),
                        "--prices",
                        str(root / "missing_prices.json"),
                    ]
                )

        self.assertEqual(code, 2)
        payload = json.loads(stdout.getvalue())
        self.assertIn("missing_prices.json", payload["error"])


if __name__ == "__main__":
    unittest.main()
