"""Scalp Squeeze Pulse Break dedicated EXIT worker."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)


def _run_exit_worker() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) if not existing_pythonpath else f"{repo_root}{os.pathsep}{existing_pythonpath}"

    subprocess.run(
        [sys.executable, "-m", "workers.scalp_precision.exit_worker"],
        check=True,
        cwd=str(repo_root),
        env=env,
    )


if __name__ == "__main__":
    _configure_logging()
    _run_exit_worker()
