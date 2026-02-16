"""Scalp Ping 5s B版専用 EXIT worker.

B版の環境変数を precision-exit 形式へマッピングして、専用 worker の
in-process 実行でエグジットを起動します。
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
        os.getenv(f"{prefix}_EXIT_TAGS", os.getenv(f"{prefix}_STRATEGY_TAG", "scalp_ping_5s_b_live")),
    )
    os.environ.setdefault("SCALP_PRECISION_POCKET", os.getenv(f"{prefix}_POCKET", "scalp_fast"))
    os.environ.setdefault(
        "SCALP_PRECISION_EXIT_LOG_PREFIX",
        os.getenv(f"{prefix}_EXIT_LOG_PREFIX", "[ScalpPing5S_B:Exit]"),
    )
    os.environ.setdefault("SCALP_PRECISION_EXIT_PROFILE_ENABLED", os.getenv(f"{prefix}_EXIT_PROFILE_ENABLED", "1"))
    os.environ.setdefault("SCALP_PRECISION_EXIT_PROFILE_PATH", os.getenv("STRATEGY_PROTECTION_PATH", "config/strategy_exit_protections.yaml"))
    os.environ.setdefault("SCALP_PRECISION_EXIT_PROFILE_TTL_SEC", os.getenv(f"{prefix}_EXIT_PROFILE_TTL_SEC", "12.0"))


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )


if __name__ == "__main__":
    _configure_logging()
    prefix = "SCALP_PING_5S_B"
    _apply_prefix_map(prefix)
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
