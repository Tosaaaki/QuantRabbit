from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SYNC = ROOT / "scripts" / "sync-live-runtime.sh"


class LiveRuntimeSyncTest(unittest.TestCase):
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
            )

            self.assertEqual(result.returncode, 6)
            self.assertIn("QR vNext Trader automation is not ACTIVE", result.stderr)


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


def _write_automation(path: Path, live: Path, *, status: str) -> None:
    path.write_text(
        "\n".join(
            [
                "version = 1",
                'id = "qr-trader"',
                'kind = "cron"',
                'name = "QR vNext Trader"',
                'prompt = "test"',
                f'status = "{status}"',
                'rrule = "FREQ=WEEKLY;BYDAY=MO;BYHOUR=7;BYMINUTE=0"',
                'model = "gpt-5.5"',
                'reasoning_effort = "medium"',
                'execution_environment = "local"',
                f'cwds = ["{live}"]',
            ]
        )
        + "\n"
    )


if __name__ == "__main__":
    unittest.main()
