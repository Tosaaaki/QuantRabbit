"""Experimental 5-second ping scalp worker (B版).

この実装は既存 ``workers.scalp_ping_5s`` を再利用しつつ、環境変数プレフィックスを
``SCALP_PING_5S_B_*`` に切り替えて独立した戦略タグで運用するための薄いラッパです。
"""

from __future__ import annotations

import asyncio
import logging
import os


def _apply_alt_env(prefix: str, *, fallback_tag: str, fallback_log_prefix: str) -> None:
    base_prefix = "SCALP_PING_5S"
    prefix = str(prefix).rstrip("_")
    source = f"{prefix}_"

    alt_items = list(os.environ.items())
    for key, value in alt_items:
        if not str(key).startswith(source):
            continue
        mapped_key = f"{base_prefix}_{key[len(source):]}"
        os.environ[mapped_key] = str(value)

    # Keep this clone disabled by default unless explicitly enabled.
    os.environ[f"{base_prefix}_ENABLED"] = os.getenv(
        f"{prefix}_ENABLED", "0"
    )
    os.environ[f"{base_prefix}_STRATEGY_TAG"] = os.getenv(
        f"{prefix}_STRATEGY_TAG", fallback_tag
    )
    os.environ[f"{base_prefix}_LOG_PREFIX"] = os.getenv(
        f"{prefix}_LOG_PREFIX", fallback_log_prefix
    )


_apply_alt_env(
    "SCALP_PING_5S_B",
    fallback_tag="scalp_ping_5s_b",
    fallback_log_prefix="[SCALP_PING_5S_B]",
)

# Import existing worker after env remap so config loads B-specific env.
from workers.scalp_ping_5s.worker import scalp_ping_5s_worker as scalp_ping_5s_b_worker


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )


if __name__ == "__main__":
    _configure_logging()
    asyncio.run(scalp_ping_5s_b_worker())
