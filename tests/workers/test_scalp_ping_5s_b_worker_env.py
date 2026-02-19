from __future__ import annotations

import os
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workers.scalp_ping_5s_b import worker as b_worker


def _clear_scalp_ping_env(monkeypatch) -> None:
    for key in list(os.environ):
        if key.startswith("SCALP_PING_5S"):
            monkeypatch.delenv(key, raising=False)


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
