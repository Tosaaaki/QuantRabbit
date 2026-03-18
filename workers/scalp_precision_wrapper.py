from __future__ import annotations

import os

_FALSEY = {"", "0", "false", "no", "off"}
_ENTRY_BASE_PREFIX = "SCALP_PRECISION"
_EXIT_BASE_PREFIX = "SCALP_PRECISION_EXIT"


def _clear_projected_env(*, base_prefix: str, source_prefix: str) -> None:
    keep = f"{source_prefix}_"
    base = f"{base_prefix}_"
    for key in list(os.environ):
        if str(key).startswith(base) and not str(key).startswith(keep):
            del os.environ[key]


def _project_prefixed_env(*, base_prefix: str, source_prefix: str) -> None:
    _clear_projected_env(base_prefix=base_prefix, source_prefix=source_prefix)
    source = f"{source_prefix}_"
    source_len = len(source)
    for key, value in list(os.environ.items()):
        if not str(key).startswith(source):
            continue
        mapped_key = f"{base_prefix}_{key[source_len:]}"
        os.environ[mapped_key] = str(value)


def _env_truthy(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in _FALSEY


def apply_precision_mode_env(
    source_prefix: str,
    *,
    mode: str,
    fallback_log_prefix: str,
    fallback_allowlist: str | None = None,
    fallback_pocket: str = "scalp",
) -> None:
    _project_prefixed_env(base_prefix=_ENTRY_BASE_PREFIX, source_prefix=source_prefix)
    fallback_allow = fallback_allowlist or mode
    os.environ[f"{_ENTRY_BASE_PREFIX}_ENABLED"] = str(
        os.getenv(f"{source_prefix}_ENABLED", "1")
    )
    os.environ[f"{_ENTRY_BASE_PREFIX}_MODE"] = os.getenv(f"{source_prefix}_MODE", mode)
    os.environ[f"{_ENTRY_BASE_PREFIX}_MODE_FILTER_ALLOWLIST"] = os.getenv(
        f"{source_prefix}_MODE_FILTER_ALLOWLIST", "1"
    )
    allowlist = os.getenv(f"{source_prefix}_UNIT_ALLOWLIST", fallback_allow)
    os.environ[f"{_ENTRY_BASE_PREFIX}_UNIT_ALLOWLIST"] = allowlist
    os.environ[f"{_ENTRY_BASE_PREFIX}_ALLOWLIST"] = allowlist
    os.environ[f"{_ENTRY_BASE_PREFIX}_LOG_PREFIX"] = os.getenv(
        f"{source_prefix}_LOG_PREFIX", fallback_log_prefix
    )
    os.environ[f"{_ENTRY_BASE_PREFIX}_POCKET"] = os.getenv(
        f"{source_prefix}_POCKET", fallback_pocket
    )


def apply_precision_exit_env(
    source_prefix: str,
    *,
    exit_tags: str,
    fallback_log_prefix: str,
    fallback_pocket: str = "scalp",
) -> None:
    exit_source_prefix = f"{source_prefix}_EXIT"
    _project_prefixed_env(
        base_prefix=_EXIT_BASE_PREFIX, source_prefix=exit_source_prefix
    )
    os.environ[f"{_ENTRY_BASE_PREFIX}_POCKET"] = os.getenv(
        f"{exit_source_prefix}_POCKET", fallback_pocket
    )
    os.environ[f"{_EXIT_BASE_PREFIX}_TAGS"] = os.getenv(
        f"{exit_source_prefix}_TAGS", exit_tags
    )
    os.environ[f"{_EXIT_BASE_PREFIX}_LOG_PREFIX"] = os.getenv(
        f"{exit_source_prefix}_LOG_PREFIX", fallback_log_prefix
    )
    os.environ[f"{_EXIT_BASE_PREFIX}_PROFILE_ENABLED"] = os.getenv(
        f"{exit_source_prefix}_PROFILE_ENABLED", "1"
    )
    os.environ[f"{_EXIT_BASE_PREFIX}_PROFILE_TTL_SEC"] = os.getenv(
        f"{exit_source_prefix}_PROFILE_TTL_SEC", "12.0"
    )


def protection_enabled(source_prefix: str, *, default: bool = True) -> bool:
    return _env_truthy(f"{source_prefix}_ALLOW_UNPROTECTED_ENTRY", not default)
