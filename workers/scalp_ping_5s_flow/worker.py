"""Entry worker entrypoint for the flow-only 5-second scalp variant."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path


def _apply_alt_env(prefix: str, *, fallback_tag: str, fallback_log_prefix: str) -> None:
    base_prefix = "SCALP_PING_5S"
    source_prefix = str(prefix).rstrip("_")
    source = f"{source_prefix}_"

    # Keep B/other clones isolated from the base SCALP_PING_5S_* values.
    for key in list(os.environ):
        if (
            str(key).startswith(f"{base_prefix}_")
            and not str(key).startswith(source)
        ):
            del os.environ[key]

    alt_items = list(os.environ.items())
    for key, value in alt_items:
        if not str(key).startswith(source):
            continue
        mapped_key = f"{base_prefix}_{key[len(source):]}"
        os.environ[mapped_key] = str(value)

    os.environ[f"{base_prefix}_ENV_PREFIX"] = source_prefix
    os.environ[f"{base_prefix}_ENABLED"] = str(os.getenv(f"{source_prefix}_ENABLED", "0"))
    os.environ[f"{base_prefix}_STRATEGY_TAG"] = os.getenv(
        f"{source_prefix}_STRATEGY_TAG", fallback_tag
    )
    os.environ[f"{base_prefix}_LOG_PREFIX"] = os.getenv(
        f"{source_prefix}_LOG_PREFIX", fallback_log_prefix
    )

    logger = logging.getLogger(__name__)
    logger.info(
        "[SCALP_PING_5S_FLOW] env mapped: source=%s enabled=%s strategy=%s",
        source_prefix,
        os.getenv(f"{base_prefix}_ENABLED", "0"),
        os.getenv(f"{base_prefix}_STRATEGY_TAG", fallback_tag),
    )


def _run_worker() -> None:
    _apply_alt_env(
        "SCALP_PING_5S_FLOW",
        fallback_tag="scalp_ping_5s_flow_live",
        fallback_log_prefix="[SCALP_PING_5S_FLOW]",
    )
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) if not existing_pythonpath else f"{repo_root}{os.pathsep}{existing_pythonpath}"
    subprocess.run(
        [sys.executable, "-m", "workers.scalp_ping_5s.worker"],
        check=True,
        cwd=str(repo_root),
        env=env,
    )


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )


if __name__ == "__main__":
    _configure_logging()
    _run_worker()
