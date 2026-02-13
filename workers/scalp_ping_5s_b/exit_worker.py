"""専用 5秒ping B版の EXIT ワーカー.

`workers.scalp_precision.exit_worker` を再利用しつつ、B版の環境変数を
このワーカー単体で解決できるように環境を再マッピングして起動する。
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Iterable


def _apply_alt_env(prefix: str, *, fallback_tag: str, fallback_log_prefix: str) -> None:
    prefix = prefix.rstrip("_")
    source = f"{prefix}_"
    target = "SCALP_PRECISION_"

    for key, value in list(os.environ.items()):
        key = str(key)
        if not key.startswith(source):
            continue
        suffix = key[len(source) :]
        os.environ[f"{target}{suffix}"] = str(value)

    os.environ.setdefault(
        "SCALP_PRECISION_EXIT_TAGS",
        os.getenv(f"{prefix}_EXIT_TAGS", fallback_tag),
    )
    os.environ.setdefault(
        "SCALP_PRECISION_POCKET",
        os.getenv(f"{prefix}_POCKET", "scalp_fast"),
    )
    os.environ.setdefault(
        "SCALP_PRECISION_EXIT_LOG_PREFIX",
        os.getenv(
            f"{prefix}_EXIT_LOG_PREFIX",
            fallback_log_prefix,
        ),
    )
    os.environ.setdefault(
        "SCALP_PRECISION_EXIT_PROFILE_ENABLED",
        os.getenv(f"{prefix}_EXIT_PROFILE_ENABLED", "1"),
    )

    # Keep legacy and explicit overrides available.
    exit_override_map = {
        "SCALP_PING_5S_B_EXIT_TAGS": "SCALP_PRECISION_EXIT_TAGS",
        "SCALP_PING_5S_B_POCKET": "SCALP_PRECISION_POCKET",
        "SCALP_PING_5S_B_EXIT_LOG_PREFIX": "SCALP_PRECISION_EXIT_LOG_PREFIX",
        "SCALP_PING_5S_B_EXIT_PROFILE_ENABLED": "SCALP_PRECISION_EXIT_PROFILE_ENABLED",
        "SCALP_PING_5S_B_EXIT_PROFILE_PATH": "SCALP_PRECISION_EXIT_PROFILE_PATH",
        "SCALP_PING_5S_B_EXIT_PROFILE_TTL_SEC": "SCALP_PRECISION_EXIT_PROFILE_TTL_SEC",
    }
    for src, dst in exit_override_map.items():
        value = os.getenv(src)
        if value is not None:
            os.environ[dst] = value


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )


_ENV_PREFIX = "SCALP_PING_5S_B"
_apply_alt_env(
    _ENV_PREFIX,
    fallback_tag="scalp_ping_5s_b",
    fallback_log_prefix="[ScalpPing5S_B:Exit]",
)

from workers.scalp_precision.exit_worker import scalp_precision_exit_worker


if __name__ == "__main__":
    _configure_logging()
    asyncio.run(scalp_precision_exit_worker())
