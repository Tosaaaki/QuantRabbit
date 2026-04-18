#!/usr/bin/env python3
"""Shared config/env.toml loader for QuantRabbit tools."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "env.toml"

_ALIASES: dict[str, tuple[str, ...]] = {
    "oanda_token": ("oanda_token", "OANDA_TOKEN", "OANDA_API_KEY", "api_key"),
    "oanda_account_id": ("oanda_account_id", "OANDA_ACCOUNT_ID", "account_id"),
    "oanda_practice": ("oanda_practice", "OANDA_PRACTICE"),
    "oanda_hedging_enabled": ("oanda_hedging_enabled", "OANDA_HEDGING_ENABLED"),
}


def load_env_toml(path: Path = CONFIG_PATH) -> dict:
    """Load env.toml with TOML support when available, plain fallback otherwise."""
    try:
        import tomllib  # Python 3.11+
    except ImportError:  # pragma: no cover - Python <3.11
        tomllib = None

    if tomllib is not None:
        with open(path, "rb") as fh:
            return tomllib.load(fh)

    cfg: dict[str, str] = {}
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            cfg[key.strip()] = value.strip().strip('"').strip("'")
    return cfg


def _normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_config_value(cfg: dict, canonical_key: str, default=None):
    aliases = _ALIASES.get(canonical_key, (canonical_key,))
    for key in aliases:
        value = cfg.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    if default is not None:
        return default
    alias_list = ", ".join(aliases)
    raise KeyError(f"Missing {canonical_key} in {CONFIG_PATH} (accepted keys: {alias_list})")


def get_oanda_config(path: Path = CONFIG_PATH) -> dict[str, object]:
    cfg = load_env_toml(path)
    practice = _normalize_bool(get_config_value(cfg, "oanda_practice", default=False))
    base_url = "https://api-fxpractice.oanda.com" if practice else "https://api-fxtrade.oanda.com"
    return {
        "oanda_token": str(get_config_value(cfg, "oanda_token")),
        "oanda_account_id": str(get_config_value(cfg, "oanda_account_id")),
        "oanda_practice": practice,
        "oanda_hedging_enabled": _normalize_bool(
            get_config_value(cfg, "oanda_hedging_enabled", default=False)
        ),
        "oanda_base_url": base_url,
    }
