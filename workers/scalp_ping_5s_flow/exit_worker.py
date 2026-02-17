"""Exit worker entrypoint for the flow-only 5-second scalp variant."""

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

    # Remove stale envs from other strategy branches so flow route is isolated.
    for key in list(os.environ):
        str_key = str(key)
        if str_key.startswith(f"{base_prefix}_") and not str_key.startswith(source):
            del os.environ[key]
        if str_key.startswith("SCALP_PRECISION_"):
            del os.environ[key]

    alt_items = list(os.environ.items())
    for key, value in alt_items:
        if not str(key).startswith(source):
            continue
        mapped_key = f"{base_prefix}_{key[len(source):]}"
        os.environ[mapped_key] = str(value)

    # Keep env-prefix identity aligned with flow clone for downstream config reads.
    os.environ[f"{base_prefix}_ENV_PREFIX"] = source_prefix
    os.environ[f"{base_prefix}_ENABLED"] = str(os.getenv(f"{source_prefix}_ENABLED", "0"))
    os.environ[f"{base_prefix}_STRATEGY_TAG"] = os.getenv(
        f"{source_prefix}_STRATEGY_TAG", fallback_tag
    )
    os.environ[f"{base_prefix}_LOG_PREFIX"] = os.getenv(
        f"{source_prefix}_LOG_PREFIX", fallback_log_prefix
    )

    # SCALP_PING_5S_*-style exit keys are translated into SCALP_PRECISION_*
    # namespace consumed by the shared exit worker.
    source_exit_tags = os.getenv(f"{source_prefix}_EXIT_TAGS", fallback_tag)
    source_exit_pocket = os.getenv(f"{source_prefix}_POCKET", "scalp_fast")
    source_exit_profile_enabled = os.getenv(f"{source_prefix}_EXIT_PROFILE_ENABLED", "1")
    source_exit_log_prefix = os.getenv(
        f"{source_prefix}_EXIT_LOG_PREFIX",
        os.getenv(f"{source_prefix}_LOG_PREFIX", fallback_log_prefix),
    )
    os.environ["SCALP_PRECISION_EXIT_TAGS"] = source_exit_tags
    os.environ[f"{base_prefix}_EXIT_TAGS"] = source_exit_tags
    os.environ["SCALP_PRECISION_POCKET"] = source_exit_pocket
    os.environ["SCALP_PRECISION_EXIT_PROFILE_ENABLED"] = source_exit_profile_enabled
    os.environ[f"{base_prefix}_EXIT_PROFILE_ENABLED"] = source_exit_profile_enabled
    os.environ["SCALP_PRECISION_EXIT_LOG_PREFIX"] = source_exit_log_prefix
    os.environ[f"{base_prefix}_EXIT_LOG_PREFIX"] = source_exit_log_prefix

    logger = logging.getLogger(__name__)
    logger.info(
        "[SCALP_PING_5S_FLOW_EXIT] env mapped: source=%s enabled=%s strategy=%s tags=%s",
        source_prefix,
        os.getenv(f"{base_prefix}_ENABLED", "0"),
        os.getenv(f"{base_prefix}_STRATEGY_TAG", fallback_tag),
        os.getenv("SCALP_PRECISION_EXIT_TAGS", ""),
    )


def _run_worker() -> None:
    _apply_alt_env(
        "SCALP_PING_5S_FLOW",
        fallback_tag="scalp_ping_5s_flow_live",
        fallback_log_prefix="[SCALP_PING_5S_FLOW:Exit]",
    )
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) if not existing_pythonpath else f"{repo_root}{os.pathsep}{existing_pythonpath}"
    logging.getLogger(__name__).info("Application started!")
    subprocess.run(
        [sys.executable, "-m", "workers.scalp_ping_5s.exit_worker"],
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
