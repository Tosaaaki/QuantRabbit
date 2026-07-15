from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SYNC = ROOT / "scripts" / "sync-live-runtime.sh"
INSTALL_HOOKS = ROOT / "scripts" / "install-live-runtime-hooks.sh"


class LiveRuntimeSyncTest(unittest.TestCase):
    def test_installed_post_commit_hook_binds_current_worktree_as_sync_dev_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            linked = root / "linked"
            live = root / "live"
            _init_repo(repo)
            scripts = repo / "scripts"
            scripts.mkdir()
            installer = scripts / "install-live-runtime-hooks.sh"
            installer.write_text(INSTALL_HOOKS.read_text())
            installer.chmod(0o755)
            sync = scripts / "sync-live-runtime.sh"
            sync.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$QR_SYNC_DEV_ROOT"\n')
            sync.chmod(0o755)
            _run(["git", "add", "scripts"], cwd=repo)
            _run(["git", "commit", "-m", "install hook fixture"], cwd=repo)
            _run(["git", "branch", "-m", "main"], cwd=repo)
            _run(["git", "worktree", "add", "-b", "feature", str(linked), "main"], cwd=repo)

            env = os.environ.copy()
            env.update(
                {
                    "QR_SYNC_DEV_ROOT": str(repo),
                    "QR_SYNC_LIVE_ROOT": str(live),
                }
            )
            install_result = subprocess.run(
                ["bash", str(installer)],
                cwd=repo,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(install_result.returncode, 0, install_result.stderr)

            hook = Path(
                _git(
                    repo,
                    "rev-parse",
                    "--path-format=absolute",
                    "--git-path",
                    "hooks/post-commit",
                )
            )
            hook_result = subprocess.run(
                ["bash", str(hook)],
                cwd=linked,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(hook_result.returncode, 0, hook_result.stderr)
            self.assertEqual(Path(hook_result.stdout.strip()).resolve(), linked.resolve())

    def test_promotes_from_clean_detached_linked_development_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            source = Path(tmp) / "source"
            live = Path(tmp) / "live"
            _init_repo(repo)
            _commit_file(repo, "src/app.py", "print('v1')\n", "initial")
            _run(["git", "branch", "-m", "main"], cwd=repo)
            _run(["git", "checkout", "-b", "feature"], cwd=repo)
            _commit_file(repo, "src/app.py", "print('v2')\n", "feature")
            _run(["git", "worktree", "add", "-b", "runtime", str(live), "main"], cwd=repo)
            _run(["git", "worktree", "add", "--detach", str(source), "feature"], cwd=repo)

            result = _sync(source, live, source_branch="feature")

            self.assertTrue((source / ".git").is_file())
            self.assertEqual(result.returncode, 0, result.stderr)
            feature_head = _git(repo, "rev-parse", "feature")
            self.assertEqual(_git(repo, "rev-parse", "main"), feature_head)
            self.assertEqual(_git(live, "rev-parse", "HEAD"), feature_head)

    def test_rejects_unrelated_development_clone_before_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            clone = root / "clone"
            live = root / "live"
            _init_repo(repo)
            _commit_file(repo, "src/app.py", "print('v1')\n", "initial")
            _run(["git", "branch", "-m", "main"], cwd=repo)
            _run(["git", "checkout", "-b", "feature"], cwd=repo)
            _commit_file(repo, "src/app.py", "print('v2')\n", "feature")
            _run(["git", "worktree", "add", "-b", "runtime", str(live), "main"], cwd=repo)
            _run(["git", "clone", "--local", str(repo), str(clone)], cwd=root)
            main_before = _git(repo, "rev-parse", "main")

            result = _sync(clone, live, source_branch="feature")

            self.assertEqual(result.returncode, 2)
            self.assertIn("do not share the same git common directory", result.stderr)
            self.assertEqual(_git(repo, "rev-parse", "main"), main_before)
            self.assertEqual(_git(live, "rev-parse", "HEAD"), main_before)

    def test_rejects_post_merge_live_target_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            live = Path(tmp) / "live"
            _init_repo(repo)
            _commit_file(repo, "src/app.py", "print('v1')\n", "initial")
            _run(["git", "branch", "-m", "main"], cwd=repo)
            _run(["git", "checkout", "-b", "feature"], cwd=repo)
            _commit_file(repo, "src/app.py", "print('v2')\n", "feature")
            _run(["git", "worktree", "add", "-b", "runtime", str(live), "main"], cwd=repo)
            live_before = _git(live, "rev-parse", "HEAD")
            post_merge = repo / ".git" / "hooks" / "post-merge"
            post_merge.write_text(
                "#!/bin/sh\n"
                f"exec git reset --hard {live_before} >/dev/null\n"
            )
            post_merge.chmod(0o755)

            result = _sync(repo, live, source_branch="feature")

            self.assertEqual(result.returncode, 7)
            self.assertIn("live sync mismatch", result.stderr)
            self.assertEqual(_git(repo, "rev-parse", "main"), _git(repo, "rev-parse", "feature"))
            self.assertEqual(_git(live, "rev-parse", "HEAD"), live_before)

    def test_promotes_source_to_main_and_live_after_preserving_report_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            live = Path(tmp) / "live"
            _init_repo(repo)
            _commit_file(repo, "src/app.py", "print('v1')\n", "initial")
            _run(["git", "branch", "-m", "main"], cwd=repo)
            _commit_file(repo, "docs/cycle_report.md", "tracked report\n", "track report")
            _run(["git", "checkout", "-b", "feature"], cwd=repo)
            _commit_file(repo, "src/app.py", "print('v2')\n", "feature")
            _run(["git", "worktree", "add", "-b", "runtime", str(live), "main"], cwd=repo)
            (live / "docs" / "cycle_report.md").write_text("new runtime drift\n")

            result = _sync(repo, live, source_branch="feature")

            self.assertEqual(result.returncode, 0, result.stderr)
            feature_head = _git(repo, "rev-parse", "feature")
            self.assertEqual(_git(repo, "rev-parse", "main"), feature_head)
            self.assertEqual(_git(live, "rev-parse", "HEAD"), feature_head)
            self.assertEqual((live / "docs" / "cycle_report.md").read_text(), "new runtime drift\n")
            self.assertEqual(_git(live, "status", "--short"), "M docs/cycle_report.md")

    def test_promotes_after_preserving_guardian_trigger_contract_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            live = Path(tmp) / "live"
            _init_repo(repo)
            _commit_file(repo, "src/app.py", "print('v1')\n", "initial")
            _run(["git", "branch", "-m", "main"], cwd=repo)
            _commit_file(repo, "data/guardian_trigger_contract.json", '{"generated_at_utc":"old"}\n', "track contract")
            _run(["git", "checkout", "-b", "feature"], cwd=repo)
            _commit_file(repo, "src/app.py", "print('v2')\n", "feature")
            _run(["git", "worktree", "add", "-b", "runtime", str(live), "main"], cwd=repo)
            (live / "data" / "guardian_trigger_contract.json").write_text('{"generated_at_utc":"runtime"}\n')

            result = _sync(repo, live, source_branch="feature")

            self.assertEqual(result.returncode, 0, result.stderr)
            feature_head = _git(repo, "rev-parse", "feature")
            self.assertEqual(_git(repo, "rev-parse", "main"), feature_head)
            self.assertEqual(_git(live, "rev-parse", "HEAD"), feature_head)
            self.assertEqual((live / "data" / "guardian_trigger_contract.json").read_text(), '{"generated_at_utc":"runtime"}\n')
            self.assertEqual(_git(live, "status", "--short"), "M data/guardian_trigger_contract.json")

    def test_promotes_after_preserving_payoff_shape_diagnosis_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            live = Path(tmp) / "live"
            _init_repo(repo)
            _commit_file(repo, "src/app.py", "print('v1')\n", "initial")
            _run(["git", "branch", "-m", "main"], cwd=repo)
            _commit_file(repo, "data/payoff_shape_diagnosis.json", '{"generated_at_utc":"old"}\n', "track payoff diagnosis")
            _run(["git", "checkout", "-b", "feature"], cwd=repo)
            _commit_file(repo, "src/app.py", "print('v2')\n", "feature")
            _run(["git", "worktree", "add", "-b", "runtime", str(live), "main"], cwd=repo)
            (live / "data" / "payoff_shape_diagnosis.json").write_text('{"generated_at_utc":"runtime"}\n')

            result = _sync(repo, live, source_branch="feature")

            self.assertEqual(result.returncode, 0, result.stderr)
            feature_head = _git(repo, "rev-parse", "feature")
            self.assertEqual(_git(repo, "rev-parse", "main"), feature_head)
            self.assertEqual(_git(live, "rev-parse", "HEAD"), feature_head)
            self.assertEqual((live / "data" / "payoff_shape_diagnosis.json").read_text(), '{"generated_at_utc":"runtime"}\n')
            self.assertEqual(_git(live, "status", "--short"), "M data/payoff_shape_diagnosis.json")

    def test_promotes_after_preserving_as_proof_evidence_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            live = Path(tmp) / "live"
            _init_repo(repo)
            _commit_file(repo, "src/app.py", "print('v1')\n", "initial")
            _run(["git", "branch", "-m", "main"], cwd=repo)
            _commit_file(repo, "data/as_proof_pack_queue.json", '{"generated_at":"old"}\n', "track proof queue")
            _commit_file(repo, "data/harvest_live_grade_path.json", '{"generated_at":"old"}\n', "track harvest path")
            _commit_file(repo, "data/portfolio_4x_path_planner.json", '{"generated_at":"old"}\n', "track planner")
            _commit_file(repo, "docs/as_proof_pack_queue.md", "old proof report\n", "track proof report")
            _run(["git", "checkout", "-b", "feature"], cwd=repo)
            _commit_file(repo, "src/app.py", "print('v2')\n", "feature")
            _run(["git", "worktree", "add", "-b", "runtime", str(live), "main"], cwd=repo)
            (live / "data" / "as_proof_pack_queue.json").write_text('{"generated_at":"runtime"}\n')
            (live / "data" / "harvest_live_grade_path.json").write_text('{"generated_at":"runtime"}\n')
            (live / "data" / "portfolio_4x_path_planner.json").write_text('{"generated_at":"runtime"}\n')
            (live / "docs" / "as_proof_pack_queue.md").write_text("runtime proof report\n")

            result = _sync(repo, live, source_branch="feature")

            self.assertEqual(result.returncode, 0, result.stderr)
            feature_head = _git(repo, "rev-parse", "feature")
            self.assertEqual(_git(repo, "rev-parse", "main"), feature_head)
            self.assertEqual(_git(live, "rev-parse", "HEAD"), feature_head)
            self.assertEqual((live / "data" / "as_proof_pack_queue.json").read_text(), '{"generated_at":"runtime"}\n')
            self.assertEqual((live / "data" / "harvest_live_grade_path.json").read_text(), '{"generated_at":"runtime"}\n')
            self.assertEqual((live / "data" / "portfolio_4x_path_planner.json").read_text(), '{"generated_at":"runtime"}\n')
            self.assertEqual((live / "docs" / "as_proof_pack_queue.md").read_text(), "runtime proof report\n")
            self.assertEqual(
                {line.strip() for line in _git(live, "status", "--short").splitlines()},
                {
                    "M data/as_proof_pack_queue.json",
                    "M data/harvest_live_grade_path.json",
                    "M data/portfolio_4x_path_planner.json",
                    "M docs/as_proof_pack_queue.md",
                },
            )

    def test_promotes_after_preserving_active_contract_and_eurusd_evidence_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            live = Path(tmp) / "live"
            _init_repo(repo)
            _commit_file(repo, "src/app.py", "print('v1')\n", "initial")
            _run(["git", "branch", "-m", "main"], cwd=repo)
            _commit_file(repo, "data/active_trader_contract.json", '{"status":"old"}\n', "track active contract")
            _commit_file(repo, "docs/active_trader_contract.md", "old active contract\n", "track active report")
            _commit_file(repo, "data/active_opportunity_board.json", '{"status":"old"}\n', "track active board")
            _commit_file(repo, "docs/active_opportunity_board.md", "old active board\n", "track active board report")
            _commit_file(
                repo,
                "data/eurusd_short_breakout_failure_limit_s5_bidask_replay.json",
                '{"status":"old"}\n',
                "track eurusd replay",
            )
            _commit_file(
                repo,
                "data/eurusd_short_breakout_failure_limit_sample_mining.json",
                '{"status":"old"}\n',
                "track eurusd limit sample mining",
            )
            _commit_file(
                repo,
                "data/eurusd_short_breakout_failure_market_stop_vehicle_diagnosis.json",
                '{"status":"old"}\n',
                "track eurusd market stop diagnosis",
            )
            _commit_file(
                repo,
                "data/eurusd_short_breakout_failure_stop_harvest_replay.json",
                '{"status":"old"}\n',
                "track eurusd stop harvest replay",
            )
            _commit_file(
                repo,
                "docs/eurusd_short_breakout_failure_limit_s5_bidask_replay.md",
                "old eurusd replay\n",
                "track eurusd replay report",
            )
            _commit_file(
                repo,
                "docs/eurusd_short_breakout_failure_limit_sample_mining.md",
                "old eurusd limit sample mining\n",
                "track eurusd limit sample mining report",
            )
            _commit_file(
                repo,
                "docs/eurusd_short_breakout_failure_market_stop_vehicle_diagnosis.md",
                "old eurusd market stop diagnosis\n",
                "track eurusd market stop diagnosis report",
            )
            _commit_file(
                repo,
                "docs/eurusd_short_breakout_failure_stop_harvest_replay.md",
                "old eurusd stop harvest replay\n",
                "track eurusd stop harvest replay report",
            )
            _run(["git", "checkout", "-b", "feature"], cwd=repo)
            _commit_file(repo, "src/app.py", "print('v2')\n", "feature")
            _run(["git", "worktree", "add", "-b", "runtime", str(live), "main"], cwd=repo)
            (live / "data" / "active_trader_contract.json").write_text('{"status":"runtime"}\n')
            (live / "docs" / "active_trader_contract.md").write_text("runtime active contract\n")
            (live / "data" / "active_opportunity_board.json").write_text('{"status":"runtime"}\n')
            (live / "docs" / "active_opportunity_board.md").write_text("runtime active board\n")
            (live / "data" / "eurusd_short_breakout_failure_limit_s5_bidask_replay.json").write_text(
                '{"status":"runtime"}\n'
            )
            (live / "data" / "eurusd_short_breakout_failure_limit_sample_mining.json").write_text(
                '{"status":"runtime"}\n'
            )
            (live / "data" / "eurusd_short_breakout_failure_market_stop_vehicle_diagnosis.json").write_text(
                '{"status":"runtime"}\n'
            )
            (live / "data" / "eurusd_short_breakout_failure_stop_harvest_replay.json").write_text(
                '{"status":"runtime"}\n'
            )
            (live / "docs" / "eurusd_short_breakout_failure_limit_s5_bidask_replay.md").write_text(
                "runtime eurusd replay\n"
            )
            (live / "docs" / "eurusd_short_breakout_failure_limit_sample_mining.md").write_text(
                "runtime eurusd limit sample mining\n"
            )
            (live / "docs" / "eurusd_short_breakout_failure_market_stop_vehicle_diagnosis.md").write_text(
                "runtime eurusd market stop diagnosis\n"
            )
            (live / "docs" / "eurusd_short_breakout_failure_stop_harvest_replay.md").write_text(
                "runtime eurusd stop harvest replay\n"
            )

            result = _sync(repo, live, source_branch="feature")

            self.assertEqual(result.returncode, 0, result.stderr)
            feature_head = _git(repo, "rev-parse", "feature")
            self.assertEqual(_git(repo, "rev-parse", "main"), feature_head)
            self.assertEqual(_git(live, "rev-parse", "HEAD"), feature_head)
            self.assertEqual((live / "data" / "active_trader_contract.json").read_text(), '{"status":"runtime"}\n')
            self.assertEqual((live / "docs" / "active_trader_contract.md").read_text(), "runtime active contract\n")
            self.assertEqual((live / "data" / "active_opportunity_board.json").read_text(), '{"status":"runtime"}\n')
            self.assertEqual((live / "docs" / "active_opportunity_board.md").read_text(), "runtime active board\n")
            self.assertEqual(
                (live / "data" / "eurusd_short_breakout_failure_limit_s5_bidask_replay.json").read_text(),
                '{"status":"runtime"}\n',
            )
            self.assertEqual(
                (live / "data" / "eurusd_short_breakout_failure_limit_sample_mining.json").read_text(),
                '{"status":"runtime"}\n',
            )
            self.assertEqual(
                (live / "data" / "eurusd_short_breakout_failure_market_stop_vehicle_diagnosis.json").read_text(),
                '{"status":"runtime"}\n',
            )
            self.assertEqual(
                (live / "data" / "eurusd_short_breakout_failure_stop_harvest_replay.json").read_text(),
                '{"status":"runtime"}\n',
            )
            self.assertEqual(
                (live / "docs" / "eurusd_short_breakout_failure_limit_s5_bidask_replay.md").read_text(),
                "runtime eurusd replay\n",
            )
            self.assertEqual(
                (live / "docs" / "eurusd_short_breakout_failure_limit_sample_mining.md").read_text(),
                "runtime eurusd limit sample mining\n",
            )
            self.assertEqual(
                (live / "docs" / "eurusd_short_breakout_failure_market_stop_vehicle_diagnosis.md").read_text(),
                "runtime eurusd market stop diagnosis\n",
            )
            self.assertEqual(
                (live / "docs" / "eurusd_short_breakout_failure_stop_harvest_replay.md").read_text(),
                "runtime eurusd stop harvest replay\n",
            )
            self.assertEqual(
                {line.strip() for line in _git(live, "status", "--short").splitlines()},
                {
                    "M data/active_trader_contract.json",
                    "M docs/active_trader_contract.md",
                    "M data/active_opportunity_board.json",
                    "M docs/active_opportunity_board.md",
                    "M data/eurusd_short_breakout_failure_limit_s5_bidask_replay.json",
                    "M data/eurusd_short_breakout_failure_limit_sample_mining.json",
                    "M data/eurusd_short_breakout_failure_market_stop_vehicle_diagnosis.json",
                    "M data/eurusd_short_breakout_failure_stop_harvest_replay.json",
                    "M docs/eurusd_short_breakout_failure_limit_s5_bidask_replay.md",
                    "M docs/eurusd_short_breakout_failure_limit_sample_mining.md",
                    "M docs/eurusd_short_breakout_failure_market_stop_vehicle_diagnosis.md",
                    "M docs/eurusd_short_breakout_failure_stop_harvest_replay.md",
                },
            )

    def test_blocks_when_development_has_source_dirty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            live = Path(tmp) / "live"
            _init_repo(repo)
            _commit_file(repo, "src/app.py", "print('v1')\n", "initial")
            _run(["git", "branch", "-m", "main"], cwd=repo)
            _run(["git", "checkout", "-b", "feature"], cwd=repo)
            (repo / "src" / "app.py").write_text("print('dirty')\n")
            _run(["git", "worktree", "add", "-b", "runtime", str(live), "main"], cwd=repo)

            result = _sync(repo, live, source_branch="feature")

            self.assertEqual(result.returncode, 3)
            self.assertIn("blocking dirty development path", result.stderr)
            self.assertEqual(_git(repo, "rev-parse", "main"), _git(live, "rev-parse", "HEAD"))

    def test_blocks_live_source_dirty_before_advancing_main(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            live = Path(tmp) / "live"
            _init_repo(repo)
            _commit_file(repo, "src/app.py", "print('v1')\n", "initial")
            _run(["git", "branch", "-m", "main"], cwd=repo)
            main_before = _git(repo, "rev-parse", "main")
            _run(["git", "checkout", "-b", "feature"], cwd=repo)
            _commit_file(repo, "src/app.py", "print('v2')\n", "feature")
            _run(["git", "worktree", "add", "-b", "runtime", str(live), "main"], cwd=repo)
            (live / "src" / "app.py").write_text("print('live dirty')\n")

            result = _sync(repo, live, source_branch="feature")

            self.assertEqual(result.returncode, 3)
            self.assertIn("blocking dirty live path", result.stderr)
            self.assertEqual(_git(repo, "rev-parse", "main"), main_before)

    def test_live_only_syncs_runtime_from_main_without_source_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            live = Path(tmp) / "live"
            _init_repo(repo)
            _commit_file(repo, "src/app.py", "print('v1')\n", "initial")
            _run(["git", "branch", "-m", "main"], cwd=repo)
            _run(["git", "worktree", "add", "-b", "runtime", str(live), "main"], cwd=repo)
            _commit_file(repo, "src/app.py", "print('v2')\n", "main update")

            result = _sync(repo, live, live_only=True)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(_git(live, "rev-parse", "HEAD"), _git(repo, "rev-parse", "main"))

    def test_live_only_preserves_report_drift_when_already_at_main(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            live = Path(tmp) / "live"
            _init_repo(repo)
            _commit_file(repo, "src/app.py", "print('v1')\n", "initial")
            _run(["git", "branch", "-m", "main"], cwd=repo)
            _commit_file(repo, "docs/cycle_report.md", "tracked report\n", "track report")
            _run(["git", "worktree", "add", "-b", "runtime", str(live), "main"], cwd=repo)
            (live / "docs" / "cycle_report.md").write_text("new runtime drift\n")

            result = _sync(repo, live, live_only=True)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(_git(live, "rev-parse", "HEAD"), _git(repo, "rev-parse", "main"))
            self.assertEqual((live / "docs" / "cycle_report.md").read_text(), "new runtime drift\n")
            self.assertEqual(_git(live, "status", "--short"), "M docs/cycle_report.md")

    def test_preserves_close_reentry_report_archive_as_runtime_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            live = Path(tmp) / "live"
            _init_repo(repo)
            _commit_file(repo, "src/app.py", "print('v1')\n", "initial")
            _run(["git", "branch", "-m", "main"], cwd=repo)
            _run(["git", "checkout", "-b", "feature"], cwd=repo)
            _commit_file(repo, "src/app.py", "print('v2')\n", "feature")
            _run(["git", "worktree", "add", "-b", "runtime", str(live), "main"], cwd=repo)
            archive = live / "docs" / "gpt_trader_decision_report.close_reentry.md"
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_text("archived close receipt report\n")

            result = _sync(repo, live, source_branch="feature")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(_git(live, "rev-parse", "HEAD"), _git(repo, "rev-parse", "feature"))
            self.assertEqual(archive.read_text(), "archived close receipt report\n")
            self.assertEqual(
                _git(live, "status", "--short", "--untracked-files=all"),
                "?? docs/gpt_trader_decision_report.close_reentry.md",
            )

    def test_live_only_removes_empty_verdict_marker_before_clean_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            live = Path(tmp) / "live"
            _init_repo(repo)
            _commit_file(repo, "src/app.py", "print('v1')\n", "initial")
            _run(["git", "branch", "-m", "main"], cwd=repo)
            _run(["git", "worktree", "add", "-b", "runtime", str(live), "main"], cwd=repo)
            (live / "EXTEND").write_text("")

            result = _sync(repo, live, live_only=True)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((live / "EXTEND").exists())
            self.assertIn("removed empty live verdict marker: EXTEND", result.stderr)

    def test_live_only_blocks_nonempty_verdict_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            live = Path(tmp) / "live"
            _init_repo(repo)
            _commit_file(repo, "src/app.py", "print('v1')\n", "initial")
            _run(["git", "branch", "-m", "main"], cwd=repo)
            _run(["git", "worktree", "add", "-b", "runtime", str(live), "main"], cwd=repo)
            (live / "EXTEND").write_text("operator note\n")

            result = _sync(repo, live, live_only=True)

            self.assertEqual(result.returncode, 3)
            self.assertTrue((live / "EXTEND").exists())
            self.assertIn("blocking dirty live path: ?? EXTEND", result.stderr)

    def test_live_only_allows_weekend_guard_paused_automation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            live = Path(tmp) / "live"
            _init_repo(repo)
            _commit_file(repo, "src/app.py", "print('v1')\n", "initial")
            _run(["git", "branch", "-m", "main"], cwd=repo)
            _run(["git", "worktree", "add", "-b", "runtime", str(live), "main"], cwd=repo)
            automation_file = Path(tmp) / "automation.toml"
            state_file = Path(tmp) / "weekend-state.json"
            _write_automation(automation_file, live, status="PAUSED")
            state_file.write_text(
                '{"mode": "paused", "tasks": {"codex:qr-trader": {"status": "ACTIVE"}}}\n'
            )

            result = _sync(
                repo,
                live,
                live_only=True,
                skip_automation_check=False,
                automation_file=automation_file,
                weekend_state_file=state_file,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("automation is PAUSED by weekend task guard", result.stderr)

    def test_live_only_blocks_paused_automation_without_weekend_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            live = Path(tmp) / "live"
            _init_repo(repo)
            _commit_file(repo, "src/app.py", "print('v1')\n", "initial")
            _run(["git", "branch", "-m", "main"], cwd=repo)
            _run(["git", "worktree", "add", "-b", "runtime", str(live), "main"], cwd=repo)
            automation_file = Path(tmp) / "automation.toml"
            _write_automation(automation_file, live, status="PAUSED")

            result = _sync(
                repo,
                live,
                live_only=True,
                skip_automation_check=False,
                automation_file=automation_file,
                weekend_state_file=Path(tmp) / "missing-weekend-state.json",
            )

            self.assertEqual(result.returncode, 6)
            self.assertIn("QR vNext Trader automation is not ACTIVE", result.stderr)

    def test_live_only_blocks_stale_accepted_only_automation_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            live = Path(tmp) / "live"
            _init_repo(repo)
            _commit_file(repo, "src/app.py", "print('v1')\n", "initial")
            _run(["git", "branch", "-m", "main"], cwd=repo)
            _run(["git", "worktree", "add", "-b", "runtime", str(live), "main"], cwd=repo)
            automation_file = Path(tmp) / "automation.toml"
            _write_automation(
                automation_file,
                live,
                status="ACTIVE",
                prompt="\n".join(
                    [
                        _current_trader_prompt_sentinel(),
                        "After the receipt is ACCEPTED by `gpt-trader-decision`, stop here.",
                    ]
                ),
            )

            result = _sync(
                repo,
                live,
                live_only=True,
                skip_automation_check=False,
                automation_file=automation_file,
            )

            self.assertEqual(result.returncode, 6)
            self.assertIn("stale ACCEPTED-only gateway handoff text", result.stderr)

    def test_live_only_blocks_stale_recent_receipt_stop_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            live = Path(tmp) / "live"
            _init_repo(repo)
            _commit_file(repo, "src/app.py", "print('v1')\n", "initial")
            _run(["git", "branch", "-m", "main"], cwd=repo)
            _run(["git", "worktree", "add", "-b", "runtime", str(live), "main"], cwd=repo)
            automation_file = Path(tmp) / "automation.toml"
            _write_automation(
                automation_file,
                live,
                status="ACTIVE",
                prompt="\n".join(
                    [
                        _current_trader_prompt_sentinel(),
                        "STOP if `data/codex_trader_decision_response.json` was written very recently by another cycle.",
                    ]
                ),
            )

            result = _sync(
                repo,
                live,
                live_only=True,
                skip_automation_check=False,
                automation_file=automation_file,
            )

            self.assertEqual(result.returncode, 6)
            self.assertIn("stale recent-receipt STOP text", result.stderr)

    def test_live_only_blocks_stale_clean_tree_runtime_drift_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            live = Path(tmp) / "live"
            _init_repo(repo)
            _commit_file(repo, "src/app.py", "print('v1')\n", "initial")
            _run(["git", "branch", "-m", "main"], cwd=repo)
            _run(["git", "worktree", "add", "-b", "runtime", str(live), "main"], cwd=repo)
            automation_file = Path(tmp) / "automation.toml"
            _write_automation(
                automation_file,
                live,
                status="ACTIVE",
                prompt=_current_trader_prompt_sentinel(
                    include_current_runtime_drift=False
                ),
            )

            result = _sync(
                repo,
                live,
                live_only=True,
                skip_automation_check=False,
                automation_file=automation_file,
            )

            self.assertEqual(result.returncode, 6)
            self.assertIn("clean-tree runtime drift allow-list is stale", result.stderr)

    def test_live_only_blocks_too_fast_trader_cadence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            live = Path(tmp) / "live"
            _init_repo(repo)
            _commit_file(repo, "src/app.py", "print('v1')\n", "initial")
            _run(["git", "branch", "-m", "main"], cwd=repo)
            _run(["git", "worktree", "add", "-b", "runtime", str(live), "main"], cwd=repo)
            automation_file = Path(tmp) / "automation.toml"
            _write_automation(
                automation_file,
                live,
                status="ACTIVE",
                rrule="FREQ=MINUTELY;INTERVAL=5;BYDAY=SU,MO,TU,WE,TH,FR,SA",
            )

            result = _sync(
                repo,
                live,
                live_only=True,
                skip_automation_check=False,
                automation_file=automation_file,
            )

            self.assertEqual(result.returncode, 6)
            self.assertIn("cadence must be 60 minutes", result.stderr)


def _sync(
    repo: Path,
    live: Path,
    *,
    source_branch: str = "feature",
    live_only: bool = False,
    skip_automation_check: bool = True,
    automation_file: Path | None = None,
    weekend_state_file: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "QR_SYNC_DEV_ROOT": str(repo),
            "QR_SYNC_LIVE_ROOT": str(live),
            "QR_SYNC_SOURCE_BRANCH": source_branch,
            "QR_SYNC_MAIN_BRANCH": "main",
            "QR_SYNC_LIVE_BRANCH": "runtime",
        }
    )
    if skip_automation_check:
        env["QR_SYNC_SKIP_AUTOMATION_CHECK"] = "1"
    if automation_file is not None:
        env["QR_SYNC_AUTOMATION_FILE"] = str(automation_file)
    if weekend_state_file is not None:
        env["QR_WEEKEND_TASK_STATE_FILE"] = str(weekend_state_file)
    args = ["bash", str(SYNC), "--skip-tests"]
    if live_only:
        args.append("--live-only")
        env.pop("QR_SYNC_SOURCE_BRANCH", None)
    return subprocess.run(args, cwd=repo, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def _init_repo(repo: Path) -> None:
    repo.mkdir()
    _run(["git", "init"], cwd=repo)
    _run(["git", "config", "user.email", "test@example.invalid"], cwd=repo)
    _run(["git", "config", "user.name", "Test User"], cwd=repo)


def _commit_file(repo: Path, path: str, contents: str, message: str) -> None:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(contents)
    _run(["git", "add", path], cwd=repo)
    _run(["git", "commit", "-m", message], cwd=repo)


def _git(repo: Path, *args: str) -> str:
    return _run(["git", *args], cwd=repo).stdout.strip()


def _run(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def _write_automation(
    path: Path,
    live: Path,
    *,
    status: str,
    prompt: str | None = None,
    rrule: str = "FREQ=MINUTELY;INTERVAL=60;BYDAY=SU,MO,TU,WE,TH,FR,SA",
) -> None:
    prompt_text = prompt if prompt is not None else _current_trader_prompt_sentinel()
    path.write_text(
        "\n".join(
            [
                "version = 1",
                'id = "qr-trader"',
                'kind = "cron"',
                'name = "QR vNext Trader"',
                "prompt = '''" + prompt_text + "'''",
                f'status = "{status}"',
                f'rrule = "{rrule}"',
                'model = "gpt-5.5"',
                'reasoning_effort = "high"',
                'execution_environment = "local"',
                f'cwds = ["{live}"]',
            ]
        )
        + "\n"
    )


def _current_trader_prompt_sentinel(
    *,
    include_current_runtime_drift: bool = True,
) -> str:
    lines = [
        "In this workflow, the AI trader is this scheduled GPT-5.5/Codex role; there is no second AI decision-maker.",
        "The deterministic draft is never the final AI decision.",
        "Write `data/trader_decision_baseline.json` and `data/market_read_evidence_packet.json` before the GPT market read.",
        "Author `data/codex_market_read_overlay.json`, run `trader-apply-market-read`, and never replace it downstream with deterministic output.",
        "Run the locked handoff with QR_LIVE_WRAPPER_FINALIZE_CODEX_MARKET_READ=1.",
        "Strict economics split must preserve lifetime, recent, prior, and historical results without claiming proof from a small sample.",
        "Every insufficient-evidence tuning review must include structured `evidence_acquisition`.",
        "Run exactly one gateway cycle after every completed `gpt-trader-decision` verification result, including REJECTED.",
        "Do **not** stop solely because `data/codex_trader_decision_response.json` was written recently; route it through `trader-prompt-route`.",
    ]
    if include_current_runtime_drift:
        lines.append(
            "Tracked `docs/*_report.md`, `docs/guardian_action_review.md`, `data/guardian_trigger_contract.json`, receipt-state drift (`data/guardian_receipt_consumption.json`, `data/guardian_receipt_operator_review.json`), named proof/acceptance evidence diffs, `data/trader_goal_loop_orchestrator.json`, `data/active_trader_contract.json`, `data/active_opportunity_board.json`, `docs/active_opportunity_board.md`, and `eurusd_short_breakout_failure_*` diffs are runtime drift and **do not** block the run."
        )
    else:
        lines.append(
            "Tracked `docs/*_report.md`, `docs/guardian_action_review.md`, and `data/guardian_trigger_contract.json` diffs are runtime drift and **do not** block the run."
        )
    return "\n".join(lines)


if __name__ == "__main__":
    unittest.main()
