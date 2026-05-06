from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SYNC = ROOT / "scripts" / "sync-live-runtime.sh"


class LiveRuntimeSyncTest(unittest.TestCase):
    def test_promotes_source_to_main_and_live_after_clearing_report_drift(self) -> None:
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
            self.assertEqual(_git(live, "status", "--short"), "")

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


def _sync(repo: Path, live: Path, *, source_branch: str = "feature", live_only: bool = False) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "QR_SYNC_DEV_ROOT": str(repo),
            "QR_SYNC_LIVE_ROOT": str(live),
            "QR_SYNC_SOURCE_BRANCH": source_branch,
            "QR_SYNC_MAIN_BRANCH": "main",
            "QR_SYNC_LIVE_BRANCH": "runtime",
            "QR_SYNC_SKIP_AUTOMATION_CHECK": "1",
        }
    )
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


if __name__ == "__main__":
    unittest.main()
