"""Experimental 5-second ping scalp worker (B版).

この実装は既存 ``workers.scalp_ping_5s`` を再利用しつつ、環境変数プレフィックスを
``SCALP_PING_5S_B_*`` に切り替えて独立した戦略タグで運用するための薄いラッパです。
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

_FALSEY = {"", "0", "false", "no", "off"}


def _env_truthy(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in _FALSEY


def _apply_alt_env(prefix: str, *, fallback_tag: str, fallback_log_prefix: str) -> None:
    base_prefix = "SCALP_PING_5S"
    prefix = str(prefix).rstrip("_")
    source = f"{prefix}_"

    # B is a clone of SCALP_PING_5S. Keep only the B-prefixed overrides and
    # remove non-B SCALP_PING_5S_* variables so stale A-layer values are not mixed.
    # Keep B variables intact during cleanup so they can be copied down below.
    for key in list(os.environ):
        if (
            str(key).startswith(f"{base_prefix}_")
            and not str(key).startswith(source)
        ):
            del os.environ[key]

    # Capture all variables after cleanup before mutating so we don't lose B keys
    # needed for the source->base prefix projection.
    alt_items = list(os.environ.items())

    for key, value in alt_items:
        if not str(key).startswith(source):
            continue
        mapped_key = f"{base_prefix}_{key[len(source):]}"
        os.environ[mapped_key] = str(value)

    # Keep env-prefix identity aligned with B strategy so downstream guards use B
    # namespace settings (e.g., SCALP_PING_5S_B_PERF_GUARD_* / B-specific
    # tunables).
    os.environ[f"{base_prefix}_ENV_PREFIX"] = prefix

    # Keep this clone disabled by default unless explicitly enabled.
    os.environ[f"{base_prefix}_ENABLED"] = str(os.getenv(f"{prefix}_ENABLED", "0"))
    os.environ[f"{base_prefix}_STRATEGY_TAG"] = os.getenv(
        f"{prefix}_STRATEGY_TAG", fallback_tag
    )
    os.environ[f"{base_prefix}_LOG_PREFIX"] = os.getenv(
        f"{prefix}_LOG_PREFIX", fallback_log_prefix
    )

    logger = logging.getLogger(__name__)

    # Fail closed for B variant direction guard:
    # if side filter is missing/invalid, enforce sell-only so the worker does not
    # drift into unintended long entries on stale env composition.
    allow_no_side_filter = _env_truthy(f"{prefix}_ALLOW_NO_SIDE_FILTER", False)
    mapped_side_filter_key = f"{base_prefix}_SIDE_FILTER"
    allowed_side_filters = {
        "sell",
        "short",
        "open_short",
        "buy",
        "long",
        "open_long",
    }
    no_side_filter_aliases = {"", "none", "off", "disabled"}

    def _normalize_side_filter(raw: str) -> str | None:
        normalized = str(raw).strip().lower()
        if normalized in allowed_side_filters:
            return normalized
        if allow_no_side_filter and normalized in no_side_filter_aliases:
            return ""
        return None

    mapped_side_filter = str(os.getenv(mapped_side_filter_key, "")).strip().lower()
    normalized_side_filter = _normalize_side_filter(mapped_side_filter)
    if normalized_side_filter is None:
        source_side_filter = str(os.getenv(f"{prefix}_SIDE_FILTER", "")).strip().lower()
        normalized_side_filter = _normalize_side_filter(source_side_filter)
        if normalized_side_filter is None:
            normalized_side_filter = "sell"
        os.environ[mapped_side_filter_key] = normalized_side_filter
        if normalized_side_filter == "":
            logger.warning(
                "[SCALP5S_B] allowing %s to be empty (%s_ALLOW_NO_SIDE_FILTER=1)",
                mapped_side_filter_key,
                prefix,
            )
        else:
            logger.warning(
                "[SCALP5S_B] forcing %s=%s (fail-closed direction guard)",
                mapped_side_filter_key,
                normalized_side_filter,
            )
    else:
        os.environ[mapped_side_filter_key] = normalized_side_filter
        if normalized_side_filter == "":
            logger.warning(
                "[SCALP5S_B] allowing %s to be empty (%s_ALLOW_NO_SIDE_FILTER=1)",
                mapped_side_filter_key,
                prefix,
            )
    mapped_side_filter = normalized_side_filter

    # Safety baseline for B variant: keep entry protection on by default.
    # Explicitly opt out only for temporary experiments.
    allow_unprotected = _env_truthy(f"{prefix}_ALLOW_UNPROTECTED_ENTRY", False)
    mapped_use_sl_key = f"{base_prefix}_USE_SL"
    mapped_disable_hard_stop_key = f"{base_prefix}_DISABLE_ENTRY_HARD_STOP"
    if not allow_unprotected:
        if not _env_truthy(mapped_use_sl_key, False):
            logger.warning(
                "[SCALP5S_B] forcing %s=1 (set %s_ALLOW_UNPROTECTED_ENTRY=1 to bypass)",
                mapped_use_sl_key,
                prefix,
            )
            os.environ[mapped_use_sl_key] = "1"
        if _env_truthy(mapped_disable_hard_stop_key, False):
            logger.warning(
                "[SCALP5S_B] forcing %s=0 (set %s_ALLOW_UNPROTECTED_ENTRY=1 to bypass)",
                mapped_disable_hard_stop_key,
                prefix,
            )
            os.environ[mapped_disable_hard_stop_key] = "0"

    # Guardrails for mixed-prefix risk: explicitly surface effective runtime knobs
    # used by entry logic so missing/shifted B-side toggles are visible in startup
    # logs before any order path is evaluated.
    base_enabled = os.getenv(f"{base_prefix}_ENABLED", "0")
    mapped_revert_enabled = os.getenv(f"{base_prefix}_REVERT_ENABLED", "1")
    base_env_prefix = os.getenv(f"{base_prefix}_ENV_PREFIX", "")
    base_strategy = os.getenv(f"{base_prefix}_STRATEGY_TAG", fallback_tag)
    base_log_prefix = os.getenv(f"{base_prefix}_LOG_PREFIX", fallback_log_prefix)
    logger.info(
        "[SCALP5S_B] env mapped: source=%s mapped_prefix=%s enabled=%s revert_enabled=%s env_prefix=%s strategy=%s log_prefix=%s side_filter=%s",
        prefix,
        base_prefix,
        base_enabled,
        mapped_revert_enabled,
        base_env_prefix,
        base_strategy,
        base_log_prefix,
        mapped_side_filter or "(unset)",
    )
    if str(base_env_prefix).strip().upper() != prefix:
        logger.warning(
            "[SCALP5S_B] SCALP_PING_5S_ENV_PREFIX=%s but expected=%s",
            base_env_prefix or "(unset)",
            prefix,
        )
    if str(mapped_revert_enabled).strip().lower() in {"0", "false", "no", "off"}:
        logger.warning(
            "[SCALP5S_B] SCALP_PING_5S_REVERT_ENABLED is OFF. "
            "This causes no_signal:revert_disabled-like drops unless intentionally disabled."
        )


def _run_worker() -> None:
    _apply_alt_env(
        "SCALP_PING_5S_B",
        fallback_tag="scalp_ping_5s_b",
        fallback_log_prefix="[SCALP_PING_5S_B]",
    )
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) if not existing_pythonpath else f"{repo_root}{os.pathsep}{existing_pythonpath}"
    logging.getLogger(__name__).info("Application started!")
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
