from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import execution.order_manager as order_manager


def test_allow_stop_loss_on_fill_strategy_override_for_scalp_ping_5s_b(monkeypatch) -> None:
    monkeypatch.setattr(order_manager, "fixed_sl_mode", lambda: None)
    monkeypatch.delenv("ORDER_ALLOW_STOP_LOSS_ON_FILL_SCALP_PING_5S_B", raising=False)

    assert (
        order_manager._allow_stop_loss_on_fill(
            "scalp_fast",
            strategy_tag="scalp_ping_5s_b_live",
        )
        is True
    )


def test_allow_stop_loss_on_fill_keeps_default_off_for_non_override(monkeypatch) -> None:
    monkeypatch.setattr(order_manager, "fixed_sl_mode", lambda: None)
    monkeypatch.delenv("ORDER_ALLOW_STOP_LOSS_ON_FILL_SCALP_PING_5S_B", raising=False)

    assert (
        order_manager._allow_stop_loss_on_fill(
            "scalp_fast",
            strategy_tag="scalp_ping_5s_live",
        )
        is False
    )


def test_allow_stop_loss_on_fill_b_variant_can_override_global_off(monkeypatch) -> None:
    monkeypatch.setattr(order_manager, "fixed_sl_mode", lambda: False)
    monkeypatch.setenv("ORDER_ALLOW_STOP_LOSS_ON_FILL_SCALP_PING_5S_B", "1")

    assert (
        order_manager._allow_stop_loss_on_fill(
            "scalp_fast",
            strategy_tag="scalp_ping_5s_b_live",
        )
        is True
    )


def test_disable_hard_stop_b_variant_enabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("ORDER_DISABLE_ENTRY_HARD_STOP_SCALP_PING_5S_B", raising=False)
    assert (
        order_manager._disable_hard_stop_by_strategy(
            "scalp_ping_5s_b_live", "scalp_fast", {}
        )
        is False
    )


def test_disable_hard_stop_respects_explicit_thesis_override(monkeypatch) -> None:
    assert (
        order_manager._disable_hard_stop_by_strategy(
            "scalp_ping_5s_b_live",
            "scalp_fast",
            {"disable_entry_hard_stop": True},
        )
        is True
    )


def test_disable_hard_stop_legacy_ping_keeps_default_disabled(monkeypatch) -> None:
    monkeypatch.delenv("ORDER_DISABLE_ENTRY_HARD_STOP_SCALP_PING_5", raising=False)
    assert (
        order_manager._disable_hard_stop_by_strategy(
            "scalp_ping_5s_live", "scalp_fast", {}
        )
        is True
    )
