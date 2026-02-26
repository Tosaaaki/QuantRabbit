from __future__ import annotations

import importlib
import os
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workers.scalp_ping_5s_b import worker as b_worker
from workers.scalp_ping_5s_c import worker as c_worker
from workers.scalp_ping_5s_d import worker as d_worker


def _clear_scalp_ping_env(monkeypatch) -> None:
    for key in list(os.environ):
        if key.startswith("SCALP_PING_5S"):
            monkeypatch.delenv(key, raising=False)


def _reload_ping_config():
    from workers.scalp_ping_5s import config as config_mod

    return importlib.reload(config_mod)


def test_apply_alt_env_forces_protected_entry_by_default(monkeypatch) -> None:
    _clear_scalp_ping_env(monkeypatch)
    monkeypatch.setenv("SCALP_PING_5S_B_ENABLED", "1")
    monkeypatch.setenv("SCALP_PING_5S_B_USE_SL", "0")
    monkeypatch.setenv("SCALP_PING_5S_B_DISABLE_ENTRY_HARD_STOP", "1")

    b_worker._apply_alt_env(
        "SCALP_PING_5S_B",
        fallback_tag="scalp_ping_5s_b_live",
        fallback_log_prefix="[SCALP_PING_5S_B]",
    )

    assert os.getenv("SCALP_PING_5S_USE_SL") == "1"
    assert os.getenv("SCALP_PING_5S_DISABLE_ENTRY_HARD_STOP") == "0"


def test_apply_alt_env_b_forces_side_filter_sell_when_missing(monkeypatch) -> None:
    _clear_scalp_ping_env(monkeypatch)
    monkeypatch.setenv("SCALP_PING_5S_B_ENABLED", "1")
    monkeypatch.delenv("SCALP_PING_5S_B_SIDE_FILTER", raising=False)

    b_worker._apply_alt_env(
        "SCALP_PING_5S_B",
        fallback_tag="scalp_ping_5s_b_live",
        fallback_log_prefix="[SCALP_PING_5S_B]",
    )

    assert os.getenv("SCALP_PING_5S_SIDE_FILTER") == "sell"


def test_apply_alt_env_b_forces_side_filter_sell_when_invalid(monkeypatch) -> None:
    _clear_scalp_ping_env(monkeypatch)
    monkeypatch.setenv("SCALP_PING_5S_B_ENABLED", "1")
    monkeypatch.setenv("SCALP_PING_5S_B_SIDE_FILTER", "invalid")

    b_worker._apply_alt_env(
        "SCALP_PING_5S_B",
        fallback_tag="scalp_ping_5s_b_live",
        fallback_log_prefix="[SCALP_PING_5S_B]",
    )

    assert os.getenv("SCALP_PING_5S_SIDE_FILTER") == "sell"


def test_apply_alt_env_b_keeps_valid_explicit_side_filter(monkeypatch) -> None:
    _clear_scalp_ping_env(monkeypatch)
    monkeypatch.setenv("SCALP_PING_5S_B_ENABLED", "1")
    monkeypatch.setenv("SCALP_PING_5S_B_SIDE_FILTER", "buy")

    b_worker._apply_alt_env(
        "SCALP_PING_5S_B",
        fallback_tag="scalp_ping_5s_b_live",
        fallback_log_prefix="[SCALP_PING_5S_B]",
    )

    assert os.getenv("SCALP_PING_5S_SIDE_FILTER") == "buy"


def test_apply_alt_env_c_forces_side_filter_sell_when_missing(monkeypatch) -> None:
    _clear_scalp_ping_env(monkeypatch)
    monkeypatch.setenv("SCALP_PING_5S_C_ENABLED", "1")
    monkeypatch.delenv("SCALP_PING_5S_C_SIDE_FILTER", raising=False)

    c_worker._apply_alt_env(
        "SCALP_PING_5S_C",
        fallback_tag="scalp_ping_5s_c_live",
        fallback_log_prefix="[SCALP_PING_5S_C]",
    )

    assert os.getenv("SCALP_PING_5S_SIDE_FILTER") == "sell"


def test_apply_alt_env_c_forces_side_filter_sell_when_invalid(monkeypatch) -> None:
    _clear_scalp_ping_env(monkeypatch)
    monkeypatch.setenv("SCALP_PING_5S_C_ENABLED", "1")
    monkeypatch.setenv("SCALP_PING_5S_C_SIDE_FILTER", "invalid")

    c_worker._apply_alt_env(
        "SCALP_PING_5S_C",
        fallback_tag="scalp_ping_5s_c_live",
        fallback_log_prefix="[SCALP_PING_5S_C]",
    )

    assert os.getenv("SCALP_PING_5S_SIDE_FILTER") == "sell"


def test_apply_alt_env_c_keeps_valid_explicit_side_filter(monkeypatch) -> None:
    _clear_scalp_ping_env(monkeypatch)
    monkeypatch.setenv("SCALP_PING_5S_C_ENABLED", "1")
    monkeypatch.setenv("SCALP_PING_5S_C_SIDE_FILTER", "buy")

    c_worker._apply_alt_env(
        "SCALP_PING_5S_C",
        fallback_tag="scalp_ping_5s_c_live",
        fallback_log_prefix="[SCALP_PING_5S_C]",
    )

    assert os.getenv("SCALP_PING_5S_SIDE_FILTER") == "buy"


def test_apply_alt_env_c_ignores_no_side_filter_override_and_forces_sell(monkeypatch) -> None:
    _clear_scalp_ping_env(monkeypatch)
    monkeypatch.setenv("SCALP_PING_5S_C_ENABLED", "1")
    monkeypatch.setenv("SCALP_PING_5S_C_SIDE_FILTER", "none")
    monkeypatch.setenv("SCALP_PING_5S_C_ALLOW_NO_SIDE_FILTER", "1")

    c_worker._apply_alt_env(
        "SCALP_PING_5S_C",
        fallback_tag="scalp_ping_5s_c_live",
        fallback_log_prefix="[SCALP_PING_5S_C]",
    )

    assert os.getenv("SCALP_PING_5S_SIDE_FILTER") == "sell"


def test_apply_alt_env_keeps_unprotected_when_explicitly_enabled(monkeypatch) -> None:
    _clear_scalp_ping_env(monkeypatch)
    monkeypatch.setenv("SCALP_PING_5S_B_ENABLED", "1")
    monkeypatch.setenv("SCALP_PING_5S_B_USE_SL", "0")
    monkeypatch.setenv("SCALP_PING_5S_B_DISABLE_ENTRY_HARD_STOP", "1")
    monkeypatch.setenv("SCALP_PING_5S_B_ALLOW_UNPROTECTED_ENTRY", "1")

    b_worker._apply_alt_env(
        "SCALP_PING_5S_B",
        fallback_tag="scalp_ping_5s_b_live",
        fallback_log_prefix="[SCALP_PING_5S_B]",
    )

    assert os.getenv("SCALP_PING_5S_USE_SL") == "0"
    assert os.getenv("SCALP_PING_5S_DISABLE_ENTRY_HARD_STOP") == "1"


def test_apply_alt_env_c_enables_force_exit_by_default(monkeypatch) -> None:
    _clear_scalp_ping_env(monkeypatch)
    monkeypatch.setenv("SCALP_PING_5S_C_ENABLED", "1")
    monkeypatch.delenv("SCALP_PING_5S_C_FORCE_EXIT_MAX_ACTIONS", raising=False)

    c_worker._apply_alt_env(
        "SCALP_PING_5S_C",
        fallback_tag="scalp_ping_5s_c_live",
        fallback_log_prefix="[SCALP_PING_5S_C]",
    )
    cfg = _reload_ping_config()

    assert cfg.ENV_PREFIX == "SCALP_PING_5S_C"
    assert cfg.FORCE_EXIT_MAX_ACTIONS == 2
    assert cfg.FORCE_EXIT_ACTIVE is True


def test_apply_alt_env_c_can_disable_force_exit_with_explicit_zero(monkeypatch) -> None:
    _clear_scalp_ping_env(monkeypatch)
    monkeypatch.setenv("SCALP_PING_5S_C_ENABLED", "1")
    monkeypatch.setenv("SCALP_PING_5S_C_FORCE_EXIT_MAX_ACTIONS", "0")

    c_worker._apply_alt_env(
        "SCALP_PING_5S_C",
        fallback_tag="scalp_ping_5s_c_live",
        fallback_log_prefix="[SCALP_PING_5S_C]",
    )
    cfg = _reload_ping_config()

    assert cfg.ENV_PREFIX == "SCALP_PING_5S_C"
    assert cfg.FORCE_EXIT_MAX_ACTIONS == 0
    assert cfg.FORCE_EXIT_ACTIVE is False


def test_apply_alt_env_d_enables_force_exit_by_default(monkeypatch) -> None:
    _clear_scalp_ping_env(monkeypatch)
    monkeypatch.setenv("SCALP_PING_5S_D_ENABLED", "1")
    monkeypatch.delenv("SCALP_PING_5S_D_FORCE_EXIT_MAX_ACTIONS", raising=False)

    d_worker._apply_alt_env(
        "SCALP_PING_5S_D",
        fallback_tag="scalp_ping_5s_d_live",
        fallback_log_prefix="[SCALP_PING_5S_D]",
    )
    cfg = _reload_ping_config()

    assert cfg.ENV_PREFIX == "SCALP_PING_5S_D"
    assert cfg.FORCE_EXIT_MAX_ACTIONS == 2
    assert cfg.FORCE_EXIT_ACTIVE is True


def test_apply_alt_env_d_can_disable_force_exit_with_explicit_zero(monkeypatch) -> None:
    _clear_scalp_ping_env(monkeypatch)
    monkeypatch.setenv("SCALP_PING_5S_D_ENABLED", "1")
    monkeypatch.setenv("SCALP_PING_5S_D_FORCE_EXIT_MAX_ACTIONS", "0")

    d_worker._apply_alt_env(
        "SCALP_PING_5S_D",
        fallback_tag="scalp_ping_5s_d_live",
        fallback_log_prefix="[SCALP_PING_5S_D]",
    )
    cfg = _reload_ping_config()

    assert cfg.ENV_PREFIX == "SCALP_PING_5S_D"
    assert cfg.FORCE_EXIT_MAX_ACTIONS == 0
    assert cfg.FORCE_EXIT_ACTIVE is False


def test_apply_alt_env_b_maps_block_hours_jst(monkeypatch) -> None:
    _clear_scalp_ping_env(monkeypatch)
    monkeypatch.setenv("SCALP_PING_5S_B_ENABLED", "1")
    monkeypatch.setenv("SCALP_PING_5S_B_BLOCK_HOURS_JST", "1,2,3,25")

    b_worker._apply_alt_env(
        "SCALP_PING_5S_B",
        fallback_tag="scalp_ping_5s_b_live",
        fallback_log_prefix="[SCALP_PING_5S_B]",
    )
    cfg = _reload_ping_config()

    assert cfg.ENV_PREFIX == "SCALP_PING_5S_B"
    # 25 is normalized to 1 and deduplicated.
    assert cfg.BLOCK_HOURS_JST == (1, 2, 3)


def test_apply_alt_env_d_maps_allow_hours_jst(monkeypatch) -> None:
    _clear_scalp_ping_env(monkeypatch)
    monkeypatch.setenv("SCALP_PING_5S_D_ENABLED", "1")
    monkeypatch.setenv("SCALP_PING_5S_D_ALLOW_HOURS_JST", "1,10,25")

    d_worker._apply_alt_env(
        "SCALP_PING_5S_D",
        fallback_tag="scalp_ping_5s_d_live",
        fallback_log_prefix="[SCALP_PING_5S_D]",
    )
    cfg = _reload_ping_config()

    assert cfg.ENV_PREFIX == "SCALP_PING_5S_D"
    # 25 is normalized to 1 and deduplicated.
    assert cfg.ALLOW_HOURS_JST == (1, 10)
