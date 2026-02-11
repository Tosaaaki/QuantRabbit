from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import execution.order_manager as order_manager
from execution.order_manager import (
    _augment_entry_thesis_policy_generation,
    _dynamic_entry_sl_target_pips,
    _entry_loss_cap_jpy,
    _entry_quality_gate,
    _entry_quality_microstructure_gate_decision,
    _entry_quality_regime_gate_decision,
    _loss_cap_units_from_sl,
    _order_spread_block_pips,
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

def test_entry_quality_regime_gate_blocks_on_mismatch_low_conf(monkeypatch):
    monkeypatch.setenv("ORDER_ENTRY_QUALITY_REGIME_PENALTY_ENABLED", "1")
    monkeypatch.setenv("ORDER_ENTRY_QUALITY_REGIME_MISMATCH_MIN_CONF_MICRO", "70")

    entry_thesis = {
        "confidence": 60,
        "micro_regime": "Range",
    }
    allowed, reason, details = _entry_quality_regime_gate_decision(
        pocket="micro",
        strategy_tag="micro_trendmomentum",
        entry_thesis=entry_thesis,
        confidence=None,
    )

    assert allowed is False
    assert reason == "entry_quality_regime_confidence"
    assert details.get("mismatch_reason") == "trend_in_range"


def test_entry_quality_microstructure_gate_blocks_on_low_density(monkeypatch):
    import time

    class _TickWindow:
        def __init__(self, ticks):
            self._ticks = ticks

        def recent_ticks(self, seconds=60.0, *, limit=None):
            return self._ticks

    monkeypatch.setenv("ORDER_ENTRY_QUALITY_MICROSTRUCTURE_ENABLED", "1")
    monkeypatch.setenv("ORDER_ENTRY_QUALITY_MICROSTRUCTURE_WINDOW_SEC", "60")
    monkeypatch.setenv("ORDER_ENTRY_QUALITY_MICROSTRUCTURE_MAX_AGE_MS", "10000")
    monkeypatch.setenv("ORDER_ENTRY_QUALITY_MICROSTRUCTURE_MIN_SPAN_RATIO", "0.7")
    monkeypatch.setenv("ORDER_ENTRY_QUALITY_MICROSTRUCTURE_MIN_TICK_DENSITY_SCALP", "1.0")

    now = time.time()
    ticks = []
    # 10 ticks over ~60s -> density ~0.166 tick/sec (should block)
    for i in range(10):
        epoch = now - 59.0 + i * (59.0 / 9.0)
        ticks.append({"epoch": epoch, "bid": 150.0, "ask": 150.01, "mid": 150.005})
    monkeypatch.setattr("execution.order_manager.tick_window", _TickWindow(ticks))

    allowed, reason, details = _entry_quality_microstructure_gate_decision(pocket="scalp")

    assert allowed is False
    assert reason == "entry_quality_microstructure_density"
    assert details.get("tick_count") == 10


def test_loss_cap_units_from_sl_uses_jpy_per_pip_formula() -> None:
    units = _loss_cap_units_from_sl(loss_cap_jpy=150.0, sl_pips=2.5)
    assert units == 6000


def test_entry_loss_cap_jpy_uses_strategy_override(monkeypatch) -> None:
    monkeypatch.setenv("ORDER_ENTRY_LOSS_CAP_JPY_STRATEGY_SCALP_PING_5S_LIVE", "150")
    cap = _entry_loss_cap_jpy(
        "scalp_fast",
        strategy_tag="scalp_ping_5s_live",
        entry_thesis=None,
    )
    assert cap == 150.0


def test_order_spread_block_pips_prefers_strategy_override(monkeypatch) -> None:
    monkeypatch.setattr(order_manager, "_ORDER_SPREAD_BLOCK_PIPS", 1.6)
    monkeypatch.setenv("ORDER_SPREAD_BLOCK_PIPS_STRATEGY_SCALP_PING_5S_LIVE", "2.4")
    threshold = _order_spread_block_pips(
        "scalp_fast",
        strategy_tag="scalp_ping_5s_live",
        entry_thesis=None,
    )
    assert threshold == 2.4


def test_policy_generation_marker_applies_only_to_new_entries(monkeypatch) -> None:
    monkeypatch.setattr(order_manager, "_ENTRY_POLICY_GENERATION", "2026-02-11-losscap-v1")
    thesis = {"strategy_tag": "scalp_ping_5s_live"}
    augmented = _augment_entry_thesis_policy_generation(thesis, reduce_only=False)
    assert augmented is not None
    assert augmented.get("policy_generation") == "2026-02-11-losscap-v1"
    assert augmented.get("policy_scope") == "new_entries_only"

    reduced = _augment_entry_thesis_policy_generation(thesis, reduce_only=True)
    assert reduced == thesis
