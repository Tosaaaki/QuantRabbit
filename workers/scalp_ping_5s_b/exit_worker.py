"""Scalp Ping 5s B版専用 EXIT worker.

B版の環境変数を precision-exit 形式へマッピングして、専用 worker の
in-process 実行でエグジットを起動します。
"""

from __future__ import annotations

import asyncio
import logging
import os
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
    if __package__ in (None, ""):
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
    from workers.scalp_precision import exit_worker

    asyncio.run(exit_worker.scalp_precision_exit_worker())
