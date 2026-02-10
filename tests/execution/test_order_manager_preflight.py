from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import execution.order_manager as order_manager
from execution.order_manager import (
    _dynamic_entry_sl_target_pips,
    _entry_quality_gate,
    _preflight_units,
    _projected_usage_with_netting,
)


class _Snapshot:
    def __init__(self, margin_available: float, margin_used: float, margin_rate: float):
        self.margin_available = margin_available
        self.margin_used = margin_used
        self.margin_rate = margin_rate


def test_preflight_allows_margin_reduction_on_hedge(monkeypatch):
    """Opposite-direction entry that shrinks net margin should be allowed even when free margin is low."""
    snap = _Snapshot(margin_available=50_000.0, margin_used=480_000.0, margin_rate=0.04)
    monkeypatch.setattr("utils.oanda_account.get_account_snapshot", lambda: snap)
    # Net short 80k; buy 50k reduces net to 30k.
    monkeypatch.setattr("utils.oanda_account.get_position_summary", lambda timeout=3.0: (0.0, 80_000.0))

    units, req_margin = _preflight_units(estimated_price=150.0, requested_units=50_000)

    assert units == 50_000
    # margin requirement after hedge (30k * 150 * 0.04 = 180k) should drop vs current usage
    assert req_margin < snap.margin_used
    assert req_margin > 0


def test_preflight_scales_down_with_budget(monkeypatch):
    snap = _Snapshot(margin_available=60_000.0, margin_used=300_000.0, margin_rate=0.04)
    monkeypatch.setattr("utils.oanda_account.get_account_snapshot", lambda: snap)
    monkeypatch.setattr("utils.oanda_account.get_position_summary", lambda timeout=3.0: (50_000.0, 0.0))

    units, req_margin = _preflight_units(estimated_price=150.0, requested_units=100_000)

    # Budget = 300k + 60k*0.92 = 355,200 JPY => max net after trade ≈ 59,200 units
    assert 0 < units < 100_000
    per_unit_margin = 150.0 * 0.04
    net_after = 50_000.0 + units
    budget = snap.margin_used + snap.margin_available * 0.92
    assert req_margin == abs(net_after) * per_unit_margin
    assert req_margin <= budget + 1.0  # small slack for rounding


def test_projected_usage_with_netting_allows_reduction(monkeypatch):
    nav = 300_000.0
    margin_rate = 0.04
    meta = {"entry_price": 156.8}

    # Current net: 80k short. Buying 50k should bring net to 30k -> usage drops.
    monkeypatch.setattr("utils.oanda_account.get_position_summary", lambda: (0.0, 80_000.0))
    usage = _projected_usage_with_netting(
        nav,
        margin_rate,
        side_label="buy",
        units=50_000,
        margin_used=0.0,
        meta=meta,
    )

    assert usage is not None
    # margin_used after hedge ≈ 30k * 156.8 * 0.04
    expected_usage = (30_000 * 156.8 * margin_rate) / nav
    assert abs(usage - expected_usage) < 1e-6


def test_dynamic_entry_sl_target_pips_reflects_market_volatility(monkeypatch) -> None:
    monkeypatch.setattr(
        order_manager,
        "_tp_cap_factors",
        lambda pocket, entry_thesis: (
            "M5",
            {"atr_pips": 3.4, "vol_5m": 1.9, "adx": 29.0},
        ),
    )
    target, meta = _dynamic_entry_sl_target_pips(
        "micro",
        entry_thesis={"strategy_tag": "MicroRangeBreak"},
        quote={"spread_pips": 1.1},
        sl_hint_pips=1.6,
        loss_guard_pips=None,
    )
    assert target is not None
    assert target > 2.8
    assert target >= 1.6
    assert meta.get("spread_pips") == 1.1


def test_entry_quality_gate_blocks_low_confidence_on_wide_spread(monkeypatch) -> None:
    monkeypatch.setattr(
        order_manager,
        "_tp_cap_factors",
        lambda pocket, entry_thesis: (
            "M5",
            {"atr_pips": 3.1, "vol_5m": 1.75},
        ),
    )
    allowed, reason, details = _entry_quality_gate(
        "micro",
        confidence=56.0,
        strategy_tag="MicroRangeBreak",
        entry_thesis={"strategy_tag": "MicroRangeBreak"},
        quote={"spread_pips": 1.0},
        sl_pips=2.0,
        tp_pips=4.0,
    )
    assert allowed is False
    assert reason == "entry_quality_spread_sl"
    assert details.get("spread_sl_ratio") is not None


def test_entry_quality_gate_allows_high_confidence_bypass(monkeypatch) -> None:
    monkeypatch.setattr(
        order_manager,
        "_tp_cap_factors",
        lambda pocket, entry_thesis: (
            "M1",
            {"atr_pips": 2.8, "vol_5m": 1.6},
        ),
    )
    allowed, reason, details = _entry_quality_gate(
        "micro",
        confidence=95.0,
        strategy_tag="MicroRangeBreak",
        entry_thesis={"strategy_tag": "MicroRangeBreak"},
        quote={"spread_pips": 1.0},
        sl_pips=2.0,
        tp_pips=4.0,
    )
    assert allowed is True
    assert reason is None
    assert details.get("confidence") == 95.0


def test_entry_quality_gate_strategy_penalty_blocks_low_quality_tag(monkeypatch) -> None:
    monkeypatch.setattr(
        order_manager,
        "_tp_cap_factors",
        lambda pocket, entry_thesis: (
            "M1",
            {"atr_pips": 2.0, "vol_5m": 1.1},
        ),
    )
    monkeypatch.setattr(
        order_manager,
        "_entry_strategy_quality_snapshot",
        lambda strategy_tag, pocket: {
            "sample": 80.0,
            "pf": 0.62,
            "win_rate": 0.73,
            "avg_pips": -0.8,
            "avg_win_pips": 1.2,
            "avg_loss_pips": 3.8,
            "payoff": 0.32,
        },
    )

    allowed, reason, details = _entry_quality_gate(
        "micro",
        confidence=60.0,
        strategy_tag="TickImbalance",
        entry_thesis={"strategy_tag": "TickImbalance"},
        quote={"spread_pips": 0.3},
        sl_pips=2.2,
        tp_pips=4.2,
    )
    assert allowed is False
    assert reason == "entry_quality_strategy_confidence"
    assert float(details.get("strategy_penalty") or 0.0) > 0.0


def test_entry_quality_gate_strategy_penalty_skips_warmup(monkeypatch) -> None:
    monkeypatch.setattr(
        order_manager,
        "_tp_cap_factors",
        lambda pocket, entry_thesis: (
            "M1",
            {"atr_pips": 2.0, "vol_5m": 1.1},
        ),
    )
    monkeypatch.setattr(
        order_manager,
        "_entry_strategy_quality_snapshot",
        lambda strategy_tag, pocket: {
            "sample": 6.0,
            "pf": 0.5,
            "win_rate": 0.4,
            "avg_pips": -1.0,
            "avg_win_pips": 0.8,
            "avg_loss_pips": 2.0,
            "payoff": 0.4,
        },
    )

    allowed, reason, details = _entry_quality_gate(
        "micro",
        confidence=60.0,
        strategy_tag="TickImbalance",
        entry_thesis={"strategy_tag": "TickImbalance"},
        quote={"spread_pips": 0.3},
        sl_pips=2.2,
        tp_pips=4.2,
    )
    assert allowed is True
    assert reason is None
    assert float(details.get("strategy_penalty") or 0.0) == 0.0
