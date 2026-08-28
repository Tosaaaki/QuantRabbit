from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "owner_forward_shadow_runtime",
    ROOT / "tools" / "owner_forward_shadow_runtime.py",
)
assert SPEC and SPEC.loader
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class OwnerForwardShadowRuntimeTests(unittest.TestCase):
    def test_child_environment_forces_zero_authority(self) -> None:
        with patch.dict(
            os.environ,
            {
                "QR_LIVE_ENABLED": "1",
                "AI_ORDER_AUTHORITY": "LIVE",
                "QR_AUTOTRADE_LOCK_OWNER_TOKEN": "secret-token",
            },
            clear=False,
        ):
            env = runtime.child_environment(Path("/approved/.env.local"))
        self.assertEqual(env["QR_LIVE_ENABLED"], "0")
        self.assertEqual(env["AI_ORDER_AUTHORITY"], "NONE")
        self.assertEqual(env["QR_AUTOTRADE_LOCK_HELD"], "0")
        self.assertNotIn("QR_AUTOTRADE_LOCK_OWNER_TOKEN", env)

    def test_cohort_is_content_addressed(self) -> None:
        root = runtime.cohort_root(Path("/state"), "a" * 40, "b" * 64)
        self.assertEqual(root, Path("/state") / ("a" * 40 + "-" + "b" * 16))

    def test_plist_is_single_keepalive_zero_authority_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(
                expected_commit="a" * 40,
                expected_source_sha256="b" * 64,
                state_root=Path(temp),
                oanda_env_file=Path("/approved/.env.local"),
                interval_seconds=30.0,
            )
            plist = runtime.desired_plist(args)
        self.assertEqual(plist["Label"], runtime.LABEL)
        self.assertTrue(plist["KeepAlive"])
        self.assertTrue(plist["RunAtLoad"])
        argv = plist["ProgramArguments"]
        self.assertEqual(argv[2], "run")
        joined = " ".join(argv).lower()
        self.assertNotIn("--send", joined)
        self.assertNotIn("--confirm-live", joined)
        self.assertNotIn("position-execution", joined)

    def test_shadow_universe_is_initial_two_pairs(self) -> None:
        self.assertEqual(runtime.SHADOW_PAIRS, ("EUR_USD", "USD_JPY"))

    def test_run_cycle_uses_only_read_and_shadow_commands(self) -> None:
        calls = []

        def fake_runner(argv, *, env, **_kwargs):
            calls.append(list(argv))
            if "broker-snapshot" in argv:
                output = Path(argv[argv.index("--output") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps({"fetched_at_utc": "2026-08-28T00:00:00+00:00", "quotes": {}}))
                return {"stdout_tail": "{}", "stderr_tail": "", "returncode": 0, "wall_seconds": 0.1}
            if "resolve-fast-bot-shadow-outcomes.py" in " ".join(argv):
                scorecard = Path(argv[argv.index("--scorecard") + 1])
                scorecard.parent.mkdir(parents=True, exist_ok=True)
                scorecard.write_text(json.dumps({"filled_signals": 0}))
                return {"stdout_tail": json.dumps({"status": "NO_DUE_SIGNALS"}), "stderr_tail": "", "returncode": 0, "wall_seconds": 0.0}
            if "run_fast_bot_corrective_challenger.py" in " ".join(argv):
                scorecard = Path(argv[argv.index("--scorecard") + 1])
                scorecard.parent.mkdir(parents=True, exist_ok=True)
                scorecard.write_text(json.dumps({"external_order_attempts": 0, "external_orders": 0}))
                return {
                    "stdout_tail": json.dumps(
                        {
                            "status": "NO_DUE_SIGNALS",
                            "execution_authority": "NONE",
                            "external_order_attempts": 0,
                            "external_orders": 0,
                        }
                    ),
                    "stderr_tail": "",
                    "returncode": 0,
                    "wall_seconds": 0.0,
                }
            raise AssertionError(argv)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime._write_zero_authority_inputs(root)
            with patch.object(runtime, "_pair_charts_refresh_due", return_value=False):
                state = runtime.run_cycle(
                    root=root,
                    env=runtime.child_environment(Path("/approved/.env.local")),
                    state={},
                    command_runner=fake_runner,
                )
        joined = "\n".join(" ".join(row) for row in calls).lower()
        self.assertIn("broker-snapshot", joined)
        self.assertIn("resolve-fast-bot-shadow-outcomes.py", joined)
        self.assertIn("run_fast_bot_corrective_challenger.py", joined)
        self.assertNotIn("--send", joined)
        self.assertNotIn("--confirm-live", joined)
        self.assertNotIn("position-execution", joined)
        self.assertEqual(state["event_count"], 1)
        self.assertEqual(state["proposal_count"], 0)
        self.assertEqual(state["virtual_fill_count"], 0)
        self.assertEqual(state["last_corrective_challenger_result"]["external_orders"], 0)

    def test_status_hard_codes_zero_external_orders(self) -> None:
        manifest = {
            "python_executable": "/python",
            "python_executable_sha256": "a" * 64,
            "commit": "b" * 40,
            "source_bundle_sha256": "c" * 64,
        }
        status = runtime._base_status(
            manifest=manifest,
            root=Path("/state/cohort"),
            started_at="2026-08-28T00:00:00+00:00",
            counters={"event_count": 0, "proposal_count": 0, "virtual_fill_count": 0},
            restart_count=1,
        )
        self.assertEqual(status["execution_authority"], "NONE")
        self.assertFalse(status["broker_mutation_allowed"])
        self.assertEqual(status["external_order_attempts"], 0)
        self.assertEqual(status["external_orders"], 0)
        self.assertEqual(status["manual_tagless_positions_policy"], "NO_TOUCH")
        self.assertEqual(status["existing_tp_sl_policy"], "NO_TOUCH")


if __name__ == "__main__":
    unittest.main()
