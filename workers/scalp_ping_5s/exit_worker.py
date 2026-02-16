"""Scalp Ping 5s dedicated EXIT worker.

Maps SCALP_PING_5S_* settings to precision-exit settings and runs the
scalp_precision exit flow in-process after prefix mapping.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path


def _apply_prefix_map(prefix: str) -> None:
    prefix = prefix.rstrip("_")
    source = f"{prefix}_"
    target = "SCALP_PRECISION_"
    for key, value in list(os.environ.items()):
        key = str(key)
        if not key.startswith(source):
            continue
        os.environ[f"{target}{key[len(source):]}"] = str(value)

    os.environ.setdefault(
        "SCALP_PRECISION_EXIT_TAGS",
        os.getenv(f"{prefix}_EXIT_TAGS", os.getenv(f"{prefix}_STRATEGY_TAG", "scalp_ping_5s_live")),
    )
    os.environ.setdefault("SCALP_PRECISION_POCKET", os.getenv(f"{prefix}_POCKET", "scalp_fast"))
    os.environ.setdefault(
        "SCALP_PRECISION_EXIT_LOG_PREFIX",
        os.getenv(f"{prefix}_EXIT_LOG_PREFIX", "[ScalpPing5S:Exit]"),
    )
    os.environ.setdefault("SCALP_PRECISION_EXIT_PROFILE_ENABLED", os.getenv(f"{prefix}_EXIT_PROFILE_ENABLED", "1"))


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )


def _run_exit_worker() -> None:
    _ENV_PREFIX = "SCALP_PING_5S"
    _apply_prefix_map(_ENV_PREFIX)
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
