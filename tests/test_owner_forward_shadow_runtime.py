from __future__ import annotations

import importlib.util
import json
import os
import subprocess
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

    def test_source_bundle_preserves_history_and_activates_v3_holdout_policy(self) -> None:
        self.assertIn(
            Path("config/fast_bot_profit_holdout_v1.json"),
            runtime.SOURCE_BUNDLE_PATHS,
        )
        self.assertIn(
            Path("config/fast_bot_profit_holdout_v2.json"),
            runtime.SOURCE_BUNDLE_PATHS,
        )
        self.assertEqual(
            runtime.PROFIT_HOLDOUT_POLICY_PATH,
            Path("config/fast_bot_profit_holdout_v3.json"),
        )
        self.assertIn(
            runtime.PROFIT_HOLDOUT_POLICY_PATH,
            runtime.SOURCE_BUNDLE_PATHS,
        )
        self.assertIn(
            Path("tools/audit_fast_bot_resident_profit_candidates.py"),
            runtime.SOURCE_BUNDLE_PATHS,
        )
        self.assertIn(
            Path("tools/run_fast_bot_normalized_passive_forward.py"),
            runtime.SOURCE_BUNDLE_PATHS,
        )
        self.assertIn(
            Path("src/quant_rabbit/fast_bot_normalized_passive_forward.py"),
            runtime.SOURCE_BUNDLE_PATHS,
        )
        self.assertIn(
            Path("src/quant_rabbit/fast_bot_normalized_passive_family_forward.py"),
            runtime.SOURCE_BUNDLE_PATHS,
        )
        self.assertIn(
            Path("config/fast_bot_normalized_passive_forward_v1.json"),
            runtime.SOURCE_BUNDLE_PATHS,
        )
        self.assertIn(
            Path("config/fast_bot_normalized_passive_forward_v2.json"),
            runtime.SOURCE_BUNDLE_PATHS,
        )

    def test_pair_chart_fetches_latency_sensitive_m1_last(self) -> None:
        self.assertEqual(runtime.SLOW_TIMEFRAMES.split(",")[-1], "M1")
        self.assertEqual(
            set(runtime.SLOW_TIMEFRAMES.split(",")),
            {"M1", "M5", "M15", "M30", "H1", "H4", "D"},
        )

    def test_pair_chart_timeout_budget_covers_exact_get_scope(self) -> None:
        self.assertEqual(runtime._pair_chart_timeout_seconds(), 450.0)
        self.assertEqual(
            runtime._pair_chart_timeout_seconds(
                pairs=("EUR_USD",),
                timeframes="M1,M5,M15,H1",
            ),
            150.0,
        )

    def test_subprocess_timeout_is_reported_as_bounded_pair_chart_failure(self) -> None:
        argv = ["python", "-m", "quant_rabbit.cli", "pair-charts"]
        with patch.object(
            runtime.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(argv, 450.0),
        ):
            with self.assertRaisesRegex(
                runtime.RuntimeBlocked,
                r"COMMAND_TIMEOUT:quant_rabbit\.cli:pair-charts:budget_seconds=450",
            ):
                runtime.run_command(argv, env={}, timeout=450.0)

    def test_run_cycle_uses_only_read_and_shadow_commands(self) -> None:
        calls = []

        def fake_runner(argv, *, env, **_kwargs):
            calls.append(list(argv))
            if "broker-snapshot" in argv:
                output = Path(argv[argv.index("--output") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps({"fetched_at_utc": "2026-08-28T00:00:00+00:00", "quotes": {}}))
                return {"stdout_tail": "{}", "stderr_tail": "", "returncode": 0, "wall_seconds": 0.1}
            if "pair-charts" in argv:
                output = Path(argv[argv.index("--output") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(
                        {
                            "generated_at_utc": "2026-08-28T00:00:00+00:00",
                            "charts": [],
                        }
                    )
                )
                return {"stdout_tail": "{}", "stderr_tail": "", "returncode": 0, "wall_seconds": 0.1}
            if "run-fast-bot-shadow.py" in " ".join(argv):
                output = Path(argv[argv.index("--output") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps({"signals": []}))
                return {
                    "stdout_tail": json.dumps({"signal_count": 0, "shadow_output": str(output)}),
                    "stderr_tail": "",
                    "returncode": 0,
                    "wall_seconds": 0.0,
                }
            if "run_fast_bot_shock_guard.py" in " ".join(argv):
                output = Path(argv[argv.index("--output") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps({"signals": []}))
                return {
                    "stdout_tail": json.dumps(
                        {
                            "shadow_output": str(output),
                            "execution_authority": "NONE",
                            "external_order_attempts": 0,
                            "external_orders": 0,
                        }
                    ),
                    "stderr_tail": "",
                    "returncode": 0,
                    "wall_seconds": 0.0,
                }
            if "resolve-fast-bot-shadow-outcomes.py" in " ".join(argv):
                scorecard = Path(argv[argv.index("--scorecard") + 1])
                scorecard.parent.mkdir(parents=True, exist_ok=True)
                scorecard.write_text(json.dumps({"filled_signals": 0}))
                return {"stdout_tail": json.dumps({"status": "NO_DUE_SIGNALS"}), "stderr_tail": "", "returncode": 0, "wall_seconds": 0.0}
            if "run_fast_bot_profit_holdout.py" in " ".join(argv):
                output = Path(argv[argv.index("--output") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(
                        {
                            "status": "COLLECT_MORE_INDEPENDENT_DAYS",
                            "execution_authority": "NONE",
                            "broker_mutation_allowed": False,
                            "live_permission": False,
                            "external_order_attempts": 0,
                            "external_orders": 0,
                        }
                    )
                )
                return {
                    "stdout_tail": output.read_text(),
                    "stderr_tail": "",
                    "returncode": 0,
                    "wall_seconds": 0.0,
                }
            if "run_fast_bot_pair_side_quarantine.py" in " ".join(argv):
                output = Path(argv[argv.index("--output") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(
                        {
                            "status": "WAITING_FOR_FORWARD_SIGNALS",
                            "execution_authority": "NONE",
                            "broker_mutation_allowed": False,
                            "live_permission": False,
                            "external_order_attempts": 0,
                            "external_orders": 0,
                        }
                    )
                )
                return {
                    "stdout_tail": output.read_text(),
                    "stderr_tail": "",
                    "returncode": 0,
                    "wall_seconds": 0.0,
                }
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
            if "run_fast_bot_knowledge.py" in " ".join(argv):
                scorecard = Path(argv[argv.index("--scorecard") + 1])
                scorecard.parent.mkdir(parents=True, exist_ok=True)
                scorecard.write_text(
                    json.dumps(
                        {
                            "assessment_status": "COLLECTING_FORWARD_EVIDENCE",
                            "external_order_attempts": 0,
                            "external_orders": 0,
                        }
                    )
                )
                return {
                    "stdout_tail": json.dumps(
                        {
                            "status": "COLLECTING_FORWARD_EVIDENCE",
                            "execution_authority": "NONE",
                            "external_order_attempts": 0,
                            "external_orders": 0,
                        }
                    ),
                    "stderr_tail": "",
                    "returncode": 0,
                    "wall_seconds": 0.0,
                }
            if "run_autonomous_shadow_nervous_system.py" in " ".join(argv):
                output = Path(argv[argv.index("--output") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(
                        {
                            "status": "NO_SIGNALS",
                            "execution_authority": "NONE",
                            "broker_mutation_allowed": False,
                            "external_order_attempts": 0,
                            "external_orders": 0,
                            "human_approval_required": False,
                        }
                    )
                )
                return {
                    "stdout_tail": output.read_text(),
                    "stderr_tail": "",
                    "returncode": 0,
                    "wall_seconds": 0.0,
                }
            if "run_fast_bot_shock_follow.py" in " ".join(argv):
                scorecard = Path(argv[argv.index("--scorecard") + 1])
                scorecard.parent.mkdir(parents=True, exist_ok=True)
                scorecard.write_text(json.dumps({"external_order_attempts": 0, "external_orders": 0}))
                return {
                    "stdout_tail": json.dumps(
                        {
                            "shadow_status": "NO_CONFIRMED_SIGNAL",
                            "execution_authority": "NONE",
                            "external_order_attempts": 0,
                            "external_orders": 0,
                        }
                    ),
                    "stderr_tail": "",
                    "returncode": 0,
                    "wall_seconds": 0.0,
                }
            if "run_eurusd_outcome_learning.py" in " ".join(argv):
                scorecard = Path(argv[argv.index("--prospective-scorecard") + 1])
                scorecard.parent.mkdir(parents=True, exist_ok=True)
                scorecard.write_text(json.dumps({"external_order_attempts": 0, "external_orders": 0}))
                return {
                    "stdout_tail": json.dumps(
                        {
                            "status": "MARKET_CLOSED_NO_OBSERVATION",
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
            with patch.object(runtime, "_pair_charts_refresh_due", return_value=True):
                state = runtime.run_cycle(
                    root=root,
                    env=runtime.child_environment(Path("/approved/.env.local")),
                    state={},
                    command_runner=fake_runner,
                )
        joined = "\n".join(" ".join(row) for row in calls).lower()
        self.assertIn("broker-snapshot", joined)
        self.assertIn("resolve-fast-bot-shadow-outcomes.py", joined)
        self.assertIn("run_fast_bot_profit_holdout.py select", joined)
        self.assertIn("run_fast_bot_profit_holdout.py evaluate", joined)
        self.assertIn("run_fast_bot_pair_side_quarantine.py", joined)
        holdout_calls = [
            " ".join(row)
            for row in calls
            if "run_fast_bot_profit_holdout.py" in " ".join(row)
        ]
        self.assertEqual(len(holdout_calls), 2)
        self.assertTrue(
            all("config/fast_bot_profit_holdout_v3.json" in row for row in holdout_calls)
        )
        self.assertTrue(
            all("config/fast_bot_profit_holdout_v1.json" not in row for row in holdout_calls)
        )
        self.assertTrue(
            all("config/fast_bot_profit_holdout_v2.json" not in row for row in holdout_calls)
        )
        self.assertIn("run_fast_bot_corrective_challenger.py", joined)
        self.assertIn("run_fast_bot_knowledge.py", joined)
        self.assertIn("run_autonomous_shadow_nervous_system.py", joined)
        command_lines = [" ".join(row) for row in calls]
        nervous_index = next(
            index
            for index, line in enumerate(command_lines)
            if "run_autonomous_shadow_nervous_system.py" in line
        )
        knowledge_index = next(
            index
            for index, line in enumerate(command_lines)
            if "run_fast_bot_knowledge.py" in line
        )
        self.assertLess(nervous_index, knowledge_index)
        self.assertIn("run_fast_bot_shock_follow.py", joined)
        self.assertIn("run_eurusd_outcome_learning.py", joined)
        self.assertNotIn("--send", joined)
        self.assertNotIn("--confirm-live", joined)
        self.assertNotIn("position-execution", joined)
        self.assertEqual(state["event_count"], 1)
        self.assertEqual(state["proposal_count"], 0)
        self.assertEqual(state["virtual_fill_count"], 0)
        self.assertEqual(state["last_corrective_challenger_result"]["external_orders"], 0)
        self.assertEqual(state["last_profit_holdout_scorecard_result"]["external_orders"], 0)
        self.assertEqual(state["last_pair_side_quarantine_selection_result"]["external_orders"], 0)
        self.assertEqual(state["last_pair_side_quarantine_outcome_result"]["status"], "NO_DUE_SIGNALS")
        self.assertEqual(state["last_knowledge_result"]["external_orders"], 0)
        self.assertEqual(state["last_autonomous_shadow_result"]["external_orders"], 0)
        self.assertEqual(state["last_shock_follow_result"]["external_orders"], 0)
        self.assertEqual(state["last_eurusd_learning_result"]["external_orders"], 0)
        self.assertEqual(state["last_normalized_passive_forward_result"], {})

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
        self.assertEqual(status["gateway_invocations"], 0)
        self.assertEqual(status["manual_tagless_positions_policy"], "NO_TOUCH")
        self.assertEqual(status["existing_tp_sl_policy"], "NO_TOUCH")
        self.assertEqual(status["last_profit_holdout_selection_result"], {})
        self.assertEqual(status["last_profit_holdout_outcome_result"], {})
        self.assertEqual(status["last_profit_holdout_scorecard_result"], {})
        self.assertEqual(status["last_normalized_passive_forward_result"], {})
        self.assertEqual(status["last_autonomous_shadow_result"], {})
        self.assertEqual(
            status["normalized_passive_forward_label"],
            "com.quantrabbit.normalized-passive-forward",
        )


if __name__ == "__main__":
    unittest.main()
