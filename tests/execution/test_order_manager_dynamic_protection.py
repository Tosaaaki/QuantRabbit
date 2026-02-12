from __future__ import annotations

from datetime import datetime, timezone
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import execution.order_manager as order_manager


def test_dynamic_protection_uses_scalp_defaults_for_scalp_fast(monkeypatch) -> None:
    captured: list[tuple[str, float, float | None, str, float]] = []

    monkeypatch.setattr(order_manager.policy_bus, "latest", lambda: None)
    monkeypatch.setattr(
        order_manager,
        "_load_strategy_protection_config",
        lambda: {
            "defaults": {
                "be_profile": {
                    "macro": {"trigger_pips": 6.8, "lock_ratio": 0.55, "min_lock_pips": 2.6, "cooldown_sec": 90},
                    "micro": {"trigger_pips": 2.2, "lock_ratio": 0.5, "min_lock_pips": 0.6, "cooldown_sec": 45},
                    "scalp": {"trigger_pips": 1.6, "lock_ratio": 0.35, "min_lock_pips": 0.5, "cooldown_sec": 20},
                },
                "start_delay_sec": {"micro": 25, "scalp": 12},
                "max_delay_sec": {"micro": 70, "scalp": 35},
                "tp_move": {
                    "enabled": True,
                    "macro": {"trigger_pips": 6.0, "buffer_pips": 2.5},
                    "micro": {"trigger_pips": 2.0, "buffer_pips": 1.0},
                    "scalp": {"trigger_pips": 1.0, "buffer_pips": 0.8},
                },
                "tp_move_min_gap_pips": 0.3,
            },
            "strategies": {},
        },
    )
    monkeypatch.setattr(
        order_manager,
        "_maybe_update_protections",
        lambda trade_id, sl, tp, *, context, ref_price: captured.append((trade_id, sl, tp, context, ref_price)),
    )
    order_manager._LAST_PROTECTIONS.clear()

    open_positions = {
        "scalp_fast": {
            "open_trades": [
                {
                    "trade_id": "T-1",
                    "price": 153.000,
                    "side": "long",
                    "client_id": "qr-test-1",
                    "open_time": "2000-01-01T00:00:00Z",
                    "stop_loss": {"price": 152.950},
                    "take_profit": {"price": 153.050},
                    "entry_thesis": {"strategy_tag": "unknown_scalp_fast"},
                }
            ]
        }
    }
    fac_m1 = {"close": 153.020, "atr_pips": 0.4, "vol_5m": 1.0}

    order_manager._apply_dynamic_protections_v2(open_positions, fac_m1, {})

    assert len(captured) == 1
    trade_id, sl_price, tp_price, context, ref_price = captured[0]
    assert trade_id == "T-1"
    assert context == "dynamic_protection_v2"
    assert ref_price == 153.020
    assert sl_price > 153.000
    assert tp_price is not None
    # TP should be pulled near current price for scalp-style protection, not left at far value.
    assert tp_price < 153.050


def test_dynamic_protection_honors_scalp_ping_5s_live_override(monkeypatch) -> None:
    captured: list[tuple[str, float, float | None, str, float]] = []

    monkeypatch.setattr(order_manager.policy_bus, "latest", lambda: None)
    monkeypatch.setattr(
        order_manager,
        "_load_strategy_protection_config",
        lambda: {
            "defaults": {
                "be_profile": {
                    "macro": {"trigger_pips": 6.8, "lock_ratio": 0.55, "min_lock_pips": 2.6, "cooldown_sec": 90},
                    "micro": {"trigger_pips": 2.2, "lock_ratio": 0.5, "min_lock_pips": 0.6, "cooldown_sec": 45},
                    "scalp": {"trigger_pips": 1.6, "lock_ratio": 0.35, "min_lock_pips": 0.5, "cooldown_sec": 20},
                },
                "start_delay_sec": {"micro": 25, "scalp": 12},
                "max_delay_sec": {"micro": 70, "scalp": 35},
                "tp_move": {
                    "enabled": True,
                    "macro": {"trigger_pips": 6.0, "buffer_pips": 2.5},
                    "micro": {"trigger_pips": 2.0, "buffer_pips": 1.0},
                    "scalp": {"trigger_pips": 1.0, "buffer_pips": 0.8},
                },
                "tp_move_min_gap_pips": 0.3,
            },
            "strategies": {
                "scalp_ping_5s_live": {
                    "start_delay_sec": 3,
                    "max_delay_sec": 10,
                    "be_profile": {
                        "trigger_pips": 0.65,
                        "lock_ratio": 0.52,
                        "min_lock_pips": 0.18,
                        "cooldown_sec": 3,
                    },
                    "tp_move": {"enabled": True, "trigger_pips": 0.75, "buffer_pips": 0.25},
                }
            },
        },
    )
    monkeypatch.setattr(
        order_manager,
        "_maybe_update_protections",
        lambda trade_id, sl, tp, *, context, ref_price: captured.append((trade_id, sl, tp, context, ref_price)),
    )
    order_manager._LAST_PROTECTIONS.clear()

    open_positions = {
        "scalp_fast": {
            "open_trades": [
                {
                    "trade_id": "T-2",
                    "price": 153.000,
                    "side": "long",
                    "client_id": "qr-test-2",
                    "open_time": "2000-01-01T00:00:00Z",
                    "stop_loss": {"price": 152.950},
                    "take_profit": {"price": 153.030},
                    "entry_thesis": {"strategy_tag": "scalp_ping_5s_live"},
                }
            ]
        }
    }
    fac_m1 = {"close": 153.012, "atr_pips": 0.2, "vol_5m": 1.0}

    order_manager._apply_dynamic_protections_v2(open_positions, fac_m1, {})

    assert len(captured) == 1
    trade_id, sl_price, tp_price, context, ref_price = captured[0]
    assert trade_id == "T-2"
    assert context == "dynamic_protection_v2"
    assert ref_price == 153.012
    assert sl_price > 153.000
    assert tp_price is not None
    # Override buffer is 0.25 pips, so TP should be close to current price.
    assert 153.012 <= tp_price <= 153.016


def test_rollover_sl_strip_cancels_carry_trade(monkeypatch) -> None:
    now_utc = datetime(2026, 2, 12, 22, 15, tzinfo=timezone.utc)
    now_jst = now_utc.astimezone(order_manager._JST)
    cutoff_jst = now_jst.replace(hour=7, minute=0, second=0, microsecond=0)
    ctx = {
        "active": True,
        "now_utc": now_utc,
        "now_jst": now_jst,
        "cutoff_jst": cutoff_jst,
    }

    cancelled: list[dict[str, object]] = []
    monkeypatch.setattr(
        order_manager,
        "_cancel_order_sync",
        lambda **kwargs: cancelled.append(kwargs) or True,
    )
    monkeypatch.setattr(order_manager, "_ROLLOVER_SL_STRIP_COOLDOWN_SEC", 0.0, raising=False)
    monkeypatch.setattr(order_manager, "_ROLLOVER_SL_STRIP_MAX_ACTIONS", 4, raising=False)
    monkeypatch.setattr(order_manager, "_ROLLOVER_SL_STRIP_INCLUDE_MANUAL", False, raising=False)
    monkeypatch.setattr(order_manager, "_ROLLOVER_SL_STRIP_REQUIRE_CARRYOVER", True, raising=False)
    order_manager._LAST_ROLLOVER_SL_STRIP_TS.clear()
    order_manager._LAST_PROTECTIONS.clear()

    open_positions = {
        "scalp_fast": {
            "open_trades": [
                {
                    "trade_id": "R-1",
                    "client_id": "qr-test-r1",
                    "open_time": "2026-02-12T20:00:00Z",  # 05:00 JST (< 07:00 cutoff)
                    "stop_loss": {"price": 152.900, "order_id": "SL-ORDER-1"},
                }
            ]
        }
    }

    count = order_manager._strip_rollover_stop_losses(open_positions, ctx)
    assert count == 1
    assert len(cancelled) == 1
    assert cancelled[0]["order_id"] == "SL-ORDER-1"
    assert cancelled[0]["reason"] == "rollover_sl_strip"


def test_dynamic_protection_skips_trade_during_rollover_sl_strip(monkeypatch) -> None:
    captured: list[tuple[str, float, float | None, str, float]] = []
    now_utc = datetime(2026, 2, 12, 22, 20, tzinfo=timezone.utc)
    now_jst = now_utc.astimezone(order_manager._JST)
    cutoff_jst = now_jst.replace(hour=7, minute=0, second=0, microsecond=0)
    active_ctx = {
        "active": True,
        "now_utc": now_utc,
        "now_jst": now_jst,
        "cutoff_jst": cutoff_jst,
    }

    monkeypatch.setattr(order_manager.policy_bus, "latest", lambda: None)
    monkeypatch.setattr(
        order_manager,
        "_load_strategy_protection_config",
        lambda: {
            "defaults": {
                "be_profile": {
                    "macro": {"trigger_pips": 6.8, "lock_ratio": 0.55, "min_lock_pips": 2.6, "cooldown_sec": 90},
                    "micro": {"trigger_pips": 2.2, "lock_ratio": 0.5, "min_lock_pips": 0.6, "cooldown_sec": 45},
                    "scalp": {"trigger_pips": 1.6, "lock_ratio": 0.35, "min_lock_pips": 0.5, "cooldown_sec": 20},
                },
                "start_delay_sec": {"micro": 25, "scalp": 12},
                "max_delay_sec": {"micro": 70, "scalp": 35},
                "tp_move": {
                    "enabled": True,
                    "macro": {"trigger_pips": 6.0, "buffer_pips": 2.5},
                    "micro": {"trigger_pips": 2.0, "buffer_pips": 1.0},
                    "scalp": {"trigger_pips": 1.0, "buffer_pips": 0.8},
                },
                "tp_move_min_gap_pips": 0.3,
            },
            "strategies": {},
        },
    )
    monkeypatch.setattr(
        order_manager,
        "_maybe_update_protections",
        lambda trade_id, sl, tp, *, context, ref_price: captured.append((trade_id, sl, tp, context, ref_price)),
    )
    monkeypatch.setattr(order_manager, "_ROLLOVER_SL_STRIP_INCLUDE_MANUAL", False, raising=False)
    monkeypatch.setattr(order_manager, "_ROLLOVER_SL_STRIP_REQUIRE_CARRYOVER", True, raising=False)
    order_manager._LAST_PROTECTIONS.clear()

    open_positions = {
        "scalp_fast": {
            "open_trades": [
                {
                    "trade_id": "R-2",
                    "price": 153.000,
                    "side": "long",
                    "client_id": "qr-test-r2",
                    "open_time": "2026-02-12T20:10:00Z",  # 05:10 JST (< 07:00 cutoff)
                    "stop_loss": {"price": 152.950, "order_id": "SL-ORDER-2"},
                    "take_profit": {"price": 153.040},
                    "entry_thesis": {"strategy_tag": "unknown_scalp_fast"},
                }
            ]
        }
    }
    fac_m1 = {"close": 153.020, "atr_pips": 0.4, "vol_5m": 1.0}

    order_manager._apply_dynamic_protections_v2(open_positions, fac_m1, {}, rollover_ctx=active_ctx)
    assert captured == []

    order_manager._apply_dynamic_protections_v2(open_positions, fac_m1, {}, rollover_ctx={"active": False})
    assert len(captured) == 1
