"""MACD RSI division exit worker micro variant (B).

This wrapper maps SCALP_MACD_RSI_DIV_B_* keys onto SCALP_PRECISION_* and
runs the shared MACD RSI DIV exit loop with B-specific tags.
"""

from __future__ import annotations

import asyncio
import logging
import os


def _apply_alt_env(prefix: str, *, fallback_tag: str, fallback_log_prefix: str) -> None:
    base_prefix = "SCALP_PRECISION"
    prefix = str(prefix).rstrip("_")
    source = f"{prefix}_"

    for key in list(os.environ):
        if str(key).startswith(f"{base_prefix}_") and not str(key).startswith(source):
            del os.environ[key]

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

    exit_tags = os.getenv(
        f"{prefix}_EXIT_TAGS",
        os.getenv(f"{prefix}_STRATEGY_TAG", fallback_tag),
    ).strip()
    if exit_tags:
        os.environ[f"{base_prefix}_EXIT_TAGS"] = exit_tags
        os.environ["SCALP_PRECISION_EXIT_TAGS"] = exit_tags

    exit_pocket = os.getenv(f"{prefix}_POCKET", "").strip()
    if exit_pocket:
        os.environ[f"{base_prefix}_POCKET"] = exit_pocket
        os.environ["SCALP_PRECISION_POCKET"] = exit_pocket

    logging.getLogger(__name__).info(
        "[MACD_RSIS_B_EXIT] env mapped: source=%s enabled=%s strategy=%s tags=%s",
        prefix,
        os.getenv(f"{base_prefix}_ENABLED", "0"),
        os.getenv(f"{base_prefix}_STRATEGY_TAG", fallback_tag),
        os.getenv("SCALP_PRECISION_EXIT_TAGS", ""),
    )


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )


async def scalp_macd_rsi_div_b_exit_worker() -> None:
    _apply_alt_env(
        "SCALP_MACD_RSI_DIV_B",
        fallback_tag="scalp_macd_rsi_div_b_live",
        fallback_log_prefix="[MACD_RSI_B:Exit]",
    )
    from workers.scalp_macd_rsi_div.exit_worker import (  # noqa: WPS433
        scalp_macd_rsi_div_exit_worker,
    )

    await scalp_macd_rsi_div_exit_worker()


if __name__ == "__main__":
    _configure_logging()
    asyncio.run(scalp_macd_rsi_div_b_exit_worker())
