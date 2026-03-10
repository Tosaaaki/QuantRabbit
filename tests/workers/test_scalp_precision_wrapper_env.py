from __future__ import annotations

import os
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workers import scalp_precision_wrapper as precision_wrapper
from workers.scalp_drought_revert import exit_worker as drought_exit_worker
from workers.scalp_drought_revert import worker as drought_worker
from workers.scalp_precision_lowvol import exit_worker as lowvol_exit_worker
from workers.scalp_precision_lowvol import worker as lowvol_worker
from workers.scalp_vwap_revert import exit_worker as vwap_exit_worker
from workers.scalp_vwap_revert import worker as vwap_worker


def _clear_precision_env(monkeypatch) -> None:
    for key in list(os.environ):
        if key.startswith("SCALP_PRECISION"):
            monkeypatch.delenv(key, raising=False)


def test_apply_precision_mode_env_projects_prefixed_values(monkeypatch) -> None:
    _clear_precision_env(monkeypatch)
    monkeypatch.setenv("SCALP_PRECISION_LOWVOL_ENABLED", "1")
    monkeypatch.setenv("SCALP_PRECISION_LOWVOL_LOG_PREFIX", "[Scalp:PLV]")
    monkeypatch.setenv("SCALP_PRECISION_LOWVOL_LOOP_INTERVAL_SEC", "1.8")
    monkeypatch.setenv("SCALP_PRECISION_ALLOWLIST", "stale")

    precision_wrapper.apply_precision_mode_env(
        "SCALP_PRECISION_LOWVOL",
        mode="precision_lowvol",
        fallback_log_prefix="[Scalp:PrecisionLowVol]",
    )

    assert os.getenv("SCALP_PRECISION_ENABLED") == "1"
    assert os.getenv("SCALP_PRECISION_MODE") == "precision_lowvol"
    assert os.getenv("SCALP_PRECISION_ALLOWLIST") == "precision_lowvol"
    assert os.getenv("SCALP_PRECISION_UNIT_ALLOWLIST") == "precision_lowvol"
    assert os.getenv("SCALP_PRECISION_LOG_PREFIX") == "[Scalp:PLV]"
    assert os.getenv("SCALP_PRECISION_LOOP_INTERVAL_SEC") == "1.8"


def test_apply_precision_exit_env_projects_prefixed_values(monkeypatch) -> None:
    _clear_precision_env(monkeypatch)
    monkeypatch.setenv("SCALP_PRECISION_VWAP_REVERT_EXIT_TAGS", "VwapRevertS")
    monkeypatch.setenv("SCALP_PRECISION_VWAP_REVERT_EXIT_LOG_PREFIX", "[ScalpExit:VWAP]")
    monkeypatch.setenv("SCALP_PRECISION_VWAP_REVERT_EXIT_PROFILE_ENABLED", "0")
    monkeypatch.setenv("SCALP_PRECISION_VWAP_REVERT_EXIT_POCKET", "scalp_fast")

    precision_wrapper.apply_precision_exit_env(
        "SCALP_PRECISION_VWAP_REVERT",
        exit_tags="VwapRevertS",
        fallback_log_prefix="[ScalpExit:VWAPRevert]",
    )

    assert os.getenv("SCALP_PRECISION_EXIT_TAGS") == "VwapRevertS"
    assert os.getenv("SCALP_PRECISION_EXIT_LOG_PREFIX") == "[ScalpExit:VWAP]"
    assert os.getenv("SCALP_PRECISION_EXIT_PROFILE_ENABLED") == "0"
    assert os.getenv("SCALP_PRECISION_POCKET") == "scalp_fast"


def test_lowvol_wrapper_sets_precision_lowvol_mode(monkeypatch) -> None:
    _clear_precision_env(monkeypatch)
    lowvol_worker._apply_alt_env()

    assert os.getenv("SCALP_PRECISION_MODE") == "precision_lowvol"
    assert os.getenv("SCALP_PRECISION_UNIT_ALLOWLIST") == "precision_lowvol"
    assert os.getenv("SCALP_PRECISION_LOG_PREFIX") == "[Scalp:PrecisionLowVol]"


def test_lowvol_exit_wrapper_sets_precision_lowvol_tag(monkeypatch) -> None:
    _clear_precision_env(monkeypatch)
    lowvol_exit_worker._apply_exit_env()

    assert os.getenv("SCALP_PRECISION_EXIT_TAGS") == "PrecisionLowVol"
    assert os.getenv("SCALP_PRECISION_EXIT_LOG_PREFIX") == "[ScalpExit:PrecisionLowVol]"


def test_vwap_wrapper_sets_vwap_revert_mode(monkeypatch) -> None:
    _clear_precision_env(monkeypatch)
    vwap_worker._apply_alt_env()

    assert os.getenv("SCALP_PRECISION_MODE") == "vwap_revert"
    assert os.getenv("SCALP_PRECISION_UNIT_ALLOWLIST") == "vwap_revert"
    assert os.getenv("SCALP_PRECISION_LOG_PREFIX") == "[Scalp:VWAPRevert]"


def test_vwap_exit_wrapper_sets_vwap_tag(monkeypatch) -> None:
    _clear_precision_env(monkeypatch)
    vwap_exit_worker._apply_exit_env()

    assert os.getenv("SCALP_PRECISION_EXIT_TAGS") == "VwapRevertS"
    assert os.getenv("SCALP_PRECISION_EXIT_LOG_PREFIX") == "[ScalpExit:VWAPRevert]"


def test_drought_wrapper_sets_drought_revert_mode(monkeypatch) -> None:
    _clear_precision_env(monkeypatch)
    drought_worker._apply_alt_env()

    assert os.getenv("SCALP_PRECISION_MODE") == "drought_revert"
    assert os.getenv("SCALP_PRECISION_UNIT_ALLOWLIST") == "drought_revert"
    assert os.getenv("SCALP_PRECISION_LOG_PREFIX") == "[Scalp:DroughtRevert]"


def test_drought_exit_wrapper_sets_drought_tag(monkeypatch) -> None:
    _clear_precision_env(monkeypatch)
    drought_exit_worker._apply_exit_env()

    assert os.getenv("SCALP_PRECISION_EXIT_TAGS") == "DroughtRevert"
    assert os.getenv("SCALP_PRECISION_EXIT_LOG_PREFIX") == "[ScalpExit:DroughtRevert]"
