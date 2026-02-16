"""MACD RSI division entry worker micro variant (B).

This module maps strategy-specific SCALP_MACD_RSI_DIV_B_* environment knobs
onto SCALP_PRECISION_* before loading the base MACD RSI strategy worker.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path


def _apply_alt_env(prefix: str, *, fallback_tag: str, fallback_log_prefix: str) -> None:
    base_prefix = "SCALP_PRECISION"
    prefix = str(prefix).rstrip("_")
    source = f"{prefix}_"

    # Keep only B-prefixed variables as the source of truth for this process.
    for key in list(os.environ):
        if str(key).startswith(f"{base_prefix}_") and not str(key).startswith(source):
            del os.environ[key]

    # Capture the source environment once cleanup is complete so only B-prefixed
    # variables are used for projection.
    source_items = list(os.environ.items())
    for key, value in source_items:
        if not str(key).startswith(source):
            continue
        mapped_key = f"{base_prefix}_{str(key)[len(source):]}"
        os.environ[mapped_key] = str(value)

    os.environ[f"{base_prefix}_ENABLED"] = str(os.getenv(f"{prefix}_ENABLED", "0"))
    os.environ[f"{base_prefix}_STRATEGY_TAG"] = os.getenv(
        f"{prefix}_STRATEGY_TAG", fallback_tag
    )
    os.environ[f"{base_prefix}_LOG_PREFIX"] = os.getenv(
        f"{prefix}_LOG_PREFIX", fallback_log_prefix
    )
    os.environ[f"{base_prefix}_ENV_PREFIX"] = prefix

    logger = logging.getLogger(__name__)
    logger.info(
        "[MACD_RSIS_B] env mapped: source=%s enabled=%s strategy=%s log_prefix=%s",
        prefix,
        os.getenv(f"{base_prefix}_ENABLED", "0"),
        os.getenv(f"{base_prefix}_STRATEGY_TAG", fallback_tag),
        os.getenv(f"{base_prefix}_LOG_PREFIX", fallback_log_prefix),
    )


def _run_worker() -> None:
    _apply_alt_env(
        "SCALP_MACD_RSI_DIV_B",
        fallback_tag="scalp_macd_rsi_div_b_live",
        fallback_log_prefix="[MACD_RSI_B]",
    )

    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) if not existing_pythonpath else f"{repo_root}{os.pathsep}{existing_pythonpath}"

    logging.getLogger(__name__).info("Application started!")
    subprocess.run(
        [sys.executable, "-m", "workers.scalp_macd_rsi_div.worker"],
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
