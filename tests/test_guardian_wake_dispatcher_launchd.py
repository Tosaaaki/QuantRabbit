from __future__ import annotations

import plistlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLIST = ROOT / "scripts" / "guardian" / "com.quantrabbit.guardian-wake-dispatcher.plist"


class GuardianWakeDispatcherLaunchdTest(unittest.TestCase):
    def test_plist_runs_live_dispatcher_with_read_only_safety_defaults(self) -> None:
        payload = plistlib.loads(PLIST.read_bytes())

        self.assertEqual(payload["Label"], "com.quantrabbit.guardian-wake-dispatcher")
        self.assertEqual(payload["WorkingDirectory"], "/Users/tossaki/App/QuantRabbit-live")
        self.assertIn(payload["StartInterval"], {30, 60})
        self.assertEqual(
            payload["StandardOutPath"],
            "/Users/tossaki/App/QuantRabbit-live/logs/guardian_wake_dispatcher.launchd.log",
        )
        self.assertEqual(
            payload["StandardErrorPath"],
            "/Users/tossaki/App/QuantRabbit-live/logs/guardian_wake_dispatcher.launchd.err",
        )

        command = " ".join(payload["ProgramArguments"])
        self.assertIn("/Users/tossaki/App/QuantRabbit-live", command)
        self.assertIn("tools/guardian_wake_dispatcher.py", command)

        env = payload["EnvironmentVariables"]
        self.assertEqual(
            env["QR_GUARDIAN_WAKE_CODEX_BIN"],
            "/Applications/ChatGPT.app/Contents/Resources/codex",
        )
        self.assertEqual(env["QR_GUARDIAN_WAKE_CODEX_PREFLIGHT"], "1")
        self.assertEqual(env["QR_GUARDIAN_WAKE_GATEWAY_HANDOFF"], "0")
        self.assertEqual(env["QR_GUARDIAN_ACTION_EXECUTE"], "0")
        self.assertEqual(env["CODEX_DISABLE_UPDATE_CHECK"], "1")


if __name__ == "__main__":
    unittest.main()
