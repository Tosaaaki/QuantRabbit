from __future__ import annotations

from execution import risk_guard


def test_build_exposure_state_handles_manual_and_bot_units(monkeypatch):
    positions = {
        "manual": {"open_trades": [{"units": 15000}]},
        "macro": {"open_trades": [{"units": 25000}, {"units": -5000}]},
        "micro": {"units": 8000},  # fallback to net units
        "__net__": {"units": 0},
    }

    # Avoid noisy metric logging during unit tests.
    monkeypatch.setattr(
        risk_guard, "log_metric", lambda *args, **kwargs: None, raising=False
    )

    state = risk_guard.build_exposure_state(
        positions,
        equity=5_000_000.0,  # JPY
        price=150.0,  # USDJPY
        cap_ratio=0.9,
    )
    assert state is not None
    assert state.manual_units == 15000.0
    # bot units counts absolute trades (25000 + 5000 + 8000)
    assert state.bot_units == 38000.0
    assert state.limit_units() > 0
    assert state.available_units() > 0
    assert not state.would_exceed(1000)
    # Consuming almost all remaining capacity should trip the guard.
    assert state.would_exceed(int(state.available_units()) + 1000)


def test_build_exposure_state_returns_none_when_equity_missing(monkeypatch):
    monkeypatch.setattr(
        risk_guard, "log_metric", lambda *args, **kwargs: None, raising=False
    )
    state = risk_guard.build_exposure_state(
        {"macro": {"units": 10000}},
        equity=None,
        price=150.0,
    )
    assert state is None


def test_exposure_state_uses_margin_pool(monkeypatch):
    monkeypatch.setattr(
        risk_guard, "log_metric", lambda *args, **kwargs: None, raising=False
    )
    positions = {
        "manual": {"units": 10000},
        "micro": {"open_trades": [{"units": 5000}]},
    }
    price = 155.0
    margin_rate = 0.03
    unit_margin = price * margin_rate
    margin_used = 120000.0
    margin_available = 80000.0
    cap = 0.9

    state = risk_guard.build_exposure_state(
        positions,
        equity=5_000_000.0,
        price=price,
        margin_used=margin_used,
        margin_available=margin_available,
        margin_rate=margin_rate,
        cap_ratio=cap,
    )
    assert state is not None
    expected_pool = cap * (margin_used + margin_available)
    assert state.margin_pool == expected_pool
    expected_manual_margin = 10000 * unit_margin
    expected_bot_margin = 5000 * unit_margin
    assert state.manual_margin == expected_manual_margin
    assert state.bot_margin == expected_bot_margin
    ratio = state.ratio()
    assert 0 < ratio < 1
    assert abs(
        ratio
        - ((expected_manual_margin + expected_bot_margin) / expected_pool)
    ) < 1e-6


def test_perf_multiplier_pf_gates_win_score(monkeypatch):
    # Ensure a high win rate does not keep risk near 1.0 when PF is clearly bad.
    monkeypatch.setattr(risk_guard, "_RISK_MULT_ENABLED", True, raising=False)
    monkeypatch.setattr(risk_guard, "_RISK_PERF_ENABLED", True, raising=False)
    monkeypatch.setattr(risk_guard, "_RISK_PERF_MIN_TRADES", 30, raising=False)
    monkeypatch.setattr(risk_guard, "_RISK_PERF_PF_BAD", 0.95, raising=False)
    monkeypatch.setattr(risk_guard, "_RISK_PERF_PF_REF", 1.30, raising=False)
    monkeypatch.setattr(risk_guard, "_RISK_PERF_WIN_BAD", 0.47, raising=False)
    monkeypatch.setattr(risk_guard, "_RISK_PERF_WIN_REF", 0.58, raising=False)
    monkeypatch.setattr(risk_guard, "_RISK_PERF_MIN_MULT", 0.25, raising=False)
    monkeypatch.setattr(risk_guard, "_RISK_PERF_MAX_MULT", 2.0, raising=False)

    # Typical failure mode: win rate looks great but PF is terrible because losses are larger.
    monkeypatch.setattr(
        risk_guard,
        "_query_perf_stats",
        lambda tag, pocket: (0.40, 0.76, 500),
        raising=False,
    )
    mult, details = risk_guard._perf_multiplier("TickImbalance", "scalp_fast")
    assert abs(mult - 0.25) < 1e-9
    assert abs(details["pf_score"] - 0.0) < 1e-9
    assert abs(details["win_score"] - 1.0) < 1e-9


def test_perf_multiplier_rewards_good_pf(monkeypatch):
    monkeypatch.setattr(risk_guard, "_RISK_MULT_ENABLED", True, raising=False)
    monkeypatch.setattr(risk_guard, "_RISK_PERF_ENABLED", True, raising=False)
    monkeypatch.setattr(risk_guard, "_RISK_PERF_MIN_TRADES", 30, raising=False)
    monkeypatch.setattr(risk_guard, "_RISK_PERF_PF_BAD", 0.95, raising=False)
    monkeypatch.setattr(risk_guard, "_RISK_PERF_PF_REF", 1.30, raising=False)
    monkeypatch.setattr(risk_guard, "_RISK_PERF_WIN_BAD", 0.47, raising=False)
    monkeypatch.setattr(risk_guard, "_RISK_PERF_WIN_REF", 0.58, raising=False)
    monkeypatch.setattr(risk_guard, "_RISK_PERF_MIN_MULT", 0.25, raising=False)
    monkeypatch.setattr(risk_guard, "_RISK_PERF_MAX_MULT", 2.0, raising=False)

    monkeypatch.setattr(
        risk_guard,
        "_query_perf_stats",
        lambda tag, pocket: (1.50, 0.60, 500),
        raising=False,
    )
    mult, _ = risk_guard._perf_multiplier("TrendMA", "macro")
    assert abs(mult - 2.0) < 1e-9


def test_allowed_lot_allows_flatten_when_risk_overshoots(monkeypatch):
    """
    When margin usage is already above the cap, opposite-direction orders that *reduce net exposure*
    should still be allowed up to the flatten amount (even if risk-based sizing would overshoot).
    """
    # Avoid dynamic multipliers / DB access in this unit test.
    monkeypatch.setattr(risk_guard, "_risk_multiplier", lambda **_: (1.0, {}), raising=False)

    # Keep safety scaling out of the way for deterministic assertions.
    monkeypatch.setenv("MARGIN_SAFETY_FACTOR", "1.0")
    monkeypatch.setattr(risk_guard, "MAX_MARGIN_USAGE", 0.85, raising=False)
    monkeypatch.setattr(risk_guard, "MAX_MARGIN_USAGE_HARD", 0.90, raising=False)

    equity = 200_000.0
    margin_available = 5_000.0  # ~2.5% free
    margin_used = 195_000.0
    price = 154.0
    margin_rate = 0.04

    # Net long 30k units (~0.30 lot). Request a short that would reduce net exposure.
    lot = risk_guard.allowed_lot(
        equity,
        sl_pips=3.0,  # tight -> risk sizing would overshoot beyond flatten
        margin_available=margin_available,
        margin_used=margin_used,
        price=price,
        margin_rate=margin_rate,
        risk_pct_override=0.04,
        pocket="micro",
        side="short",
        open_long_units=30_000.0,
        open_short_units=0.0,
    )
    assert abs(lot - 0.300) < 1e-9


def test_clamp_sl_tp_applies_rr_floor(monkeypatch):
    monkeypatch.setattr(risk_guard, "_RR_NORMALIZE_ENABLED", True, raising=False)
    monkeypatch.setattr(risk_guard, "_RR_MIN_RATIO", 1.20, raising=False)
    monkeypatch.setattr(risk_guard, "_RR_MAX_RATIO", 2.40, raising=False)
    monkeypatch.setattr(risk_guard, "_RR_MIN_SAMPLES", 999, raising=False)
    monkeypatch.setattr(
        risk_guard,
        "_query_rr_outcome_stats",
        lambda tag, pocket: {},
        raising=False,
    )

    sl, tp = risk_guard.clamp_sl_tp(
        price=150.0,
        sl=149.9,   # 10 pips
        tp=150.05,  # 5 pips -> RR 0.5
        is_buy=True,
    )
    assert sl == 149.9
    assert tp == 150.12  # 10 pips * RR min 1.2


def test_clamp_sl_tp_adapts_for_low_tp_rate(monkeypatch):
    monkeypatch.setattr(risk_guard, "_RR_NORMALIZE_ENABLED", True, raising=False)
    monkeypatch.setattr(risk_guard, "_RR_MIN_RATIO", 1.00, raising=False)
    monkeypatch.setattr(risk_guard, "_RR_MAX_RATIO", 3.00, raising=False)
    monkeypatch.setattr(risk_guard, "_RR_TARGET_TP_RATE", 0.54, raising=False)
    monkeypatch.setattr(risk_guard, "_RR_ADAPT_GAIN", 1.0, raising=False)
    monkeypatch.setattr(risk_guard, "_RR_TP_SHRINK_MAX", 0.25, raising=False)
    monkeypatch.setattr(risk_guard, "_RR_SL_EXPAND_MAX", 0.10, raising=False)
    monkeypatch.setattr(risk_guard, "_RR_ALLOW_SL_EXPAND_PF", 1.05, raising=False)
    monkeypatch.setattr(risk_guard, "_RR_PF_HARD_GUARD", 0.95, raising=False)
    monkeypatch.setattr(risk_guard, "_RR_PF_HARD_GUARD_TP_SHRINK_CAP", 0.10, raising=False)
    monkeypatch.setattr(risk_guard, "_RR_MIN_SAMPLES", 40, raising=False)
    monkeypatch.setattr(
        risk_guard,
        "_query_rr_outcome_stats",
        lambda tag, pocket: {"n": 120.0, "tp_rate": 0.30, "pf": 1.20},
        raising=False,
    )

    sl, tp = risk_guard.clamp_sl_tp(
        price=150.0,
        sl=149.9,  # 10 pips
        tp=150.2,  # 20 pips
        is_buy=True,
    )
    assert sl == 149.89  # SL 10% expansion when PF is healthy
    assert tp == 150.15  # TP shrinks by capped 25%


def test_clamp_sl_tp_limits_tp_shrink_when_pf_is_bad(monkeypatch):
    monkeypatch.setattr(risk_guard, "_RR_NORMALIZE_ENABLED", True, raising=False)
    monkeypatch.setattr(risk_guard, "_RR_MIN_RATIO", 1.00, raising=False)
    monkeypatch.setattr(risk_guard, "_RR_MAX_RATIO", 3.00, raising=False)
    monkeypatch.setattr(risk_guard, "_RR_TARGET_TP_RATE", 0.54, raising=False)
    monkeypatch.setattr(risk_guard, "_RR_ADAPT_GAIN", 1.0, raising=False)
    monkeypatch.setattr(risk_guard, "_RR_TP_SHRINK_MAX", 0.25, raising=False)
    monkeypatch.setattr(risk_guard, "_RR_SL_EXPAND_MAX", 0.10, raising=False)
    monkeypatch.setattr(risk_guard, "_RR_ALLOW_SL_EXPAND_PF", 1.05, raising=False)
    monkeypatch.setattr(risk_guard, "_RR_PF_HARD_GUARD", 0.95, raising=False)
    monkeypatch.setattr(risk_guard, "_RR_PF_HARD_GUARD_TP_SHRINK_CAP", 0.10, raising=False)
    monkeypatch.setattr(risk_guard, "_RR_MIN_SAMPLES", 40, raising=False)
    monkeypatch.setattr(
        risk_guard,
        "_query_rr_outcome_stats",
        lambda tag, pocket: {"n": 120.0, "tp_rate": 0.20, "pf": 0.80},
        raising=False,
    )

    sl, tp = risk_guard.clamp_sl_tp(
        price=150.0,
        sl=149.9,  # 10 pips
        tp=150.2,  # 20 pips
        is_buy=True,
    )
    # PF bad: TP shrink is capped at 10% and SL is not expanded.
    assert sl == 149.9
    assert tp == 150.18


def test_clamp_sl_tp_infers_strategy_and_pocket_from_caller(monkeypatch):
    monkeypatch.setattr(risk_guard, "_RR_NORMALIZE_ENABLED", True, raising=False)
    monkeypatch.setattr(risk_guard, "_RR_MIN_RATIO", 1.0, raising=False)
    monkeypatch.setattr(risk_guard, "_RR_MAX_RATIO", 3.0, raising=False)
    monkeypatch.setattr(risk_guard, "_RR_MIN_SAMPLES", 999, raising=False)

    captured: dict[str, object] = {}

    def _capture(tag, pocket):
        captured["tag"] = tag
        captured["pocket"] = pocket
        return {}

    monkeypatch.setattr(risk_guard, "_query_rr_outcome_stats", _capture, raising=False)

    def _caller() -> tuple[float | None, float | None]:
        strategy_tag = "MyEdge"
        pocket = "scalp"
        return risk_guard.clamp_sl_tp(
            price=150.0,
            sl=149.9,
            tp=150.2,
            is_buy=True,
        )

    _caller()
    assert captured.get("tag") == "MyEdge"
    assert captured.get("pocket") == "scalp"
