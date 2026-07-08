from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


class ScriptBootstrapTest(unittest.TestCase):
    def test_core_scripts_run_help_without_pythonpath(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        scripts = (
            "forecast_direction_candle_truth.py",
            "generate_intents.py",
            "mine_market_stories.py",
            "mine_strategy.py",
            "oanda_history_fetch.py",
            "oanda_history_replay_validate.py",
            "oanda_universal_rotation_miner.py",
            "package_bidask_replay_precision_rules.py",
            "plan_campaign.py",
            "qr_disk_maintenance.py",
            "qr_weekend_task_switch.py",
            "technical_entry_miner.py",
        )

        for script_name in scripts:
            with self.subTest(script=script_name):
                result = subprocess.run(
                    [sys.executable, str(repo / "scripts" / script_name), "--help"],
                    cwd=repo,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=15,
                )

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout.lower())
