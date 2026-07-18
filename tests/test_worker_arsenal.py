from datetime import datetime, timedelta, timezone

import pytest

from quant_rabbit.worker_arsenal import (
    DEADMAN_TTL,
    ArmDecision,
    WorkerArsenal,
    WorkerArsenalError,
    WorkerSpec,
)

UTC = timezone.utc
NOW = datetime(2026, 7, 20, 9, 0, tzinfo=UTC)


def _spec(**kw) -> WorkerSpec:
    base = dict(
        worker_id="W_BURST",
        cell="TREND_HIGH",
        pairs=("USD_JPY",),
        max_spread_pips=1.8,
        entry_style="MARKET",
        max_concurrent=3,
        time_stop_minutes=60,
        kill_switch="trend-flip flush",
        per_position_leverage=4.3,
    )
    base.update(kw)
    return WorkerSpec(**base)


def _arsenal() -> WorkerArsenal:
    a = WorkerArsenal()
    a.register(_spec())
    a.register(_spec(worker_id="W_GRID", cell="RANGE_LOW",
                     pairs=("EUR_USD", "AUD_USD"), max_spread_pips=1.6,
                     entry_style="LIMIT_PASSIVE", max_concurrent=5,
                     time_stop_minutes=240, per_position_leverage=0.9,
                     kill_switch="1.5x span escape"))
    a.ai_heartbeat(NOW)
    return a


def test_bleed_cell_binding_refused():
    with pytest.raises(WorkerArsenalError, match="NEVER_ARM_CELL_BINDING"):
        _spec(cell="RANGE_HIGH")


def test_unknown_cell_binding_refused():
    with pytest.raises(WorkerArsenalError, match="UNKNOWN_CELL_BINDING"):
        _spec(cell="BULL_MODE")


def test_duplicate_worker_refused():
    a = _arsenal()
    with pytest.raises(WorkerArsenalError, match="DUPLICATE_WORKER"):
        a.register(_spec())


def test_arm_only_matching_cell_and_habitat():
    a = _arsenal()
    decisions = a.arm_cycle(
        now=NOW, measured_cell="TREND_HIGH",
        spreads_pips={"USD_JPY": 1.6, "EUR_USD": 1.5, "AUD_USD": 1.3},
    )
    by = {(d.worker_id, d.pair): d for d in decisions}
    assert by[("W_BURST", "USD_JPY")].armed
    assert not by[("W_GRID", "EUR_USD")].armed
    assert by[("W_GRID", "EUR_USD")].reason == "CELL_MISMATCH:TREND_HIGH"


def test_spread_over_cap_disarms():
    a = _arsenal()
    decisions = a.arm_cycle(
        now=NOW, measured_cell="TREND_HIGH", spreads_pips={"USD_JPY": 2.4},
    )
    d = next(x for x in decisions if x.worker_id == "W_BURST")
    assert not d.armed and d.reason.startswith("SPREAD_OVER_CAP")


def test_unmeasured_spread_disarms():
    a = _arsenal()
    decisions = a.arm_cycle(now=NOW, measured_cell="TREND_HIGH", spreads_pips={})
    assert all(not d.armed for d in decisions)
    d = next(x for x in decisions if x.worker_id == "W_BURST")
    assert d.reason == "SPREAD_UNMEASURED"


def test_never_arm_cells_disarm_everything():
    a = _arsenal()
    for cell in ("RANGE_HIGH", "UNCLEAR", "EVENT"):
        decisions = a.arm_cycle(
            now=NOW, measured_cell=cell,
            spreads_pips={"USD_JPY": 1.0, "EUR_USD": 1.0, "AUD_USD": 1.0},
        )
        assert all(not d.armed for d in decisions), cell


def test_deadman_stale_eye_disarms_everything():
    a = _arsenal()
    later = NOW + DEADMAN_TTL + timedelta(minutes=1)
    decisions = a.arm_cycle(
        now=later, measured_cell="TREND_HIGH", spreads_pips={"USD_JPY": 1.0},
    )
    assert all(d.reason == "DEADMAN_EYE_STALE" for d in decisions)


def test_no_heartbeat_ever_means_dead():
    a = WorkerArsenal()
    a.register(_spec())
    assert not a.ai_alive(NOW)


def test_heartbeat_clock_regression_refused():
    a = _arsenal()
    with pytest.raises(WorkerArsenalError, match="HEARTBEAT_CLOCK_REGRESSION"):
        a.ai_heartbeat(NOW - timedelta(minutes=5))


def test_inventory_ai_decisions_route():
    a = _arsenal()
    assert a.inventory_action(
        now=NOW, worker_id="W_BURST", minutes_held=30, ai_decision="HOLD",
    ) == "HOLD_AI"
    assert a.inventory_action(
        now=NOW, worker_id="W_BURST", minutes_held=90, ai_decision="CUT",
    ) == "CUT_AI"


def test_inventory_mechanical_fallback_when_ai_silent():
    a = _arsenal()
    assert a.inventory_action(
        now=NOW, worker_id="W_BURST", minutes_held=30, ai_decision=None,
    ) == "HOLD_MECHANICAL"
    assert a.inventory_action(
        now=NOW, worker_id="W_BURST", minutes_held=61, ai_decision=None,
    ) == "CUT_MECHANICAL_TIME_STOP"


def test_inventory_stale_heartbeat_ignores_ai_answer():
    a = _arsenal()
    later = NOW + DEADMAN_TTL + timedelta(minutes=1)
    assert a.inventory_action(
        now=later, worker_id="W_BURST", minutes_held=61, ai_decision="HOLD",
    ) == "CUT_MECHANICAL_TIME_STOP"


def test_inventory_hard_ceiling_beats_ai_hold():
    a = _arsenal()
    assert a.inventory_action(
        now=NOW, worker_id="W_BURST", minutes_held=240, ai_decision="HOLD",
    ) == "CUT_HARD_CEILING"


def test_inventory_invalid_ai_decision_refused():
    a = _arsenal()
    with pytest.raises(WorkerArsenalError, match="INVALID_AI_DECISION"):
        a.inventory_action(
            now=NOW, worker_id="W_BURST", minutes_held=10, ai_decision="MAYBE",
        )


def test_live_promotion_needs_both_digests():
    a = _arsenal()
    assert not a.is_live("W_BURST")
    with pytest.raises(WorkerArsenalError, match="INVALID_PROMOTION_DIGESTS"):
        a.grant_live("W_BURST", operator_approval_sha256="short",
                     prospective_proof_sha256="x" * 64)
    a.grant_live("W_BURST", operator_approval_sha256="a" * 64,
                 prospective_proof_sha256="b" * 64)
    assert a.is_live("W_BURST")
    assert not a.is_live("W_GRID")
