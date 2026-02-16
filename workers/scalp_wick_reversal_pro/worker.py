"""Scalp Wick Reversal Pro dedicated ENTRY worker."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

_STRATEGY_MODE = "wick_reversal_pro"


def _configure_strategy_env() -> None:
    os.environ["SCALP_PRECISION_ENABLED"] = "1"
    os.environ["SCALP_PRECISION_MODE"] = _STRATEGY_MODE
    os.environ["SCALP_PRECISION_ALLOWLIST"] = _STRATEGY_MODE
    os.environ["SCALP_PRECISION_UNIT_ALLOWLIST"] = _STRATEGY_MODE
    os.environ["SCALP_PRECISION_MODE_FILTER_ALLOWLIST"] = "1"
    os.environ["SCALP_PRECISION_LOG_PREFIX"] = "[Scalp:WickRevPro]"


def _run_worker() -> None:
    _configure_strategy_env()
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) if not existing_pythonpath else f"{repo_root}{os.pathsep}{existing_pythonpath}"

    subprocess.run(
        [sys.executable, "-m", "workers.scalp_precision.worker"],
        check=True,
        cwd=str(repo_root),
        env=env,
    )


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)


if __name__ == "__main__":
    _configure_logging()
    _run_worker()
