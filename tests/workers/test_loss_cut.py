from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def test_resolve_loss_cut_disabled() -> None:
    from workers.common.loss_cut import pick_loss_cut_reason, resolve_loss_cut

    params = resolve_loss_cut({"loss_cut_enabled": False, "loss_cut_hard_pips": 5.0}, sl_pips=5.0)
    assert params.enabled is False
    assert (
        pick_loss_cut_reason(pnl_pips=-10.0, hold_sec=999.0, params=params, has_stop_loss=False) is None
    )


def test_resolve_loss_cut_requires_sl() -> None:
    from workers.common.loss_cut import pick_loss_cut_reason, resolve_loss_cut

    params = resolve_loss_cut(
        {
            "loss_cut_enabled": True,
            "loss_cut_require_sl": True,
            "loss_cut_hard_pips": 3.0,
        },
        sl_pips=5.0,
    )
    assert params.enabled is True
    assert (
        pick_loss_cut_reason(pnl_pips=-3.1, hold_sec=10.0, params=params, has_stop_loss=False) is None
    )
    assert (
        pick_loss_cut_reason(pnl_pips=-3.1, hold_sec=10.0, params=params, has_stop_loss=True)
        == "max_adverse"
    )


def test_resolve_loss_cut_derived_from_sl_mult() -> None:
    from workers.common.loss_cut import pick_loss_cut_reason, resolve_loss_cut

    params = resolve_loss_cut(
        {
            "loss_cut_enabled": True,
            "loss_cut_require_sl": False,
            "loss_cut_hard_pips": 0.0,  # derive
            "loss_cut_hard_sl_mult": 1.2,
            "loss_cut_cooldown_sec": 0.0,  # should default when enabled
        },
        sl_pips=5.0,
    )
    assert params.enabled is True
    assert params.hard_pips == pytest.approx(6.0)
    assert params.cooldown_sec == pytest.approx(6.0)
    assert (
        pick_loss_cut_reason(pnl_pips=-5.9, hold_sec=10.0, params=params, has_stop_loss=False) is None
    )
    assert (
        pick_loss_cut_reason(pnl_pips=-6.0, hold_sec=10.0, params=params, has_stop_loss=False)
        == "max_adverse"
    )


def test_strategy_exit_profile_merge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "strategy_exit_protections.yaml"
    cfg.write_text(
        "\n".join(
            [
                "defaults:",
                "  exit_profile:",
                "    loss_cut_enabled: false",
                "    loss_cut_hard_pips: 30",
                "strategies:",
                "  Foo:",
                "    exit_profile:",
                "      loss_cut_enabled: true",
                "      loss_cut_hard_pips: 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("STRATEGY_PROTECTION_ENABLED", "1")
    monkeypatch.setenv("STRATEGY_PROTECTION_PATH", str(cfg))
    monkeypatch.setenv("STRATEGY_PROTECTION_TTL_SEC", "1")

    import utils.strategy_protection as sp

    importlib.reload(sp)

    prof = sp.exit_profile_for_tag("Foo-bar")
    assert prof.get("loss_cut_enabled") is True
    assert float(prof.get("loss_cut_hard_pips") or 0) == 5.0

