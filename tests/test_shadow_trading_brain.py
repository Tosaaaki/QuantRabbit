from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quant_rabbit.gate_throughput_slo import _canonical_sha as _slo_sha  # noqa: F401
from quant_rabbit.shadow_trading_brain import (
    LAYER2_LIVE_SOURCE,
    LAYER2_SHADOW_SOURCE,
    ShadowBrainError,
    run_shadow_brain_cycle,
    validate_market_read_handoff,
)
from quant_rabbit.supervision_outcome_scorer import build_supervision_scorecard

UTC = timezone.utc
MIDWEEK = datetime(2026, 7, 15, 9, 0, tzinfo=UTC)
FRIDAY_LATE = datetime(2026, 7, 17, 20, 0, tzinfo=UTC)
SHA = "a" * 64


def _read(action="GO", side="LONG", regime="TREND", source=LAYER2_LIVE_SOURCE, pair="GBP_USD"):
    return {
        "read_source": source,
        "declared_regime": regime,
        "pair_reads": [
            {
                "pair": pair,
                "action": action,
                "side": side,
                "narrative_sha256": SHA,
                "predicted_direction": side,
                "conviction_conditions": [
                    ["REGIME_ALIGNED", True],
                    ["SESSION_FAVORABLE", True],
                    ["NO_EVENT_WINDOW", True],
                ],
            }
        ],
    }


def _cycle(**overrides):
    base = {
        "cycle_id": "20260715T090000Z",
        "decision_utc": MIDWEEK,
        "evidence_packet_sha256": SHA,
        "proprietary_indicator_sha256": SHA,
        "broker_positions": [
            {"position_id": "p1", "pair": "EUR_USD", "side": "LONG", "nav_exposure_fraction": 0.2}
        ],
        "ledger_open_positions": [
            {"position_id": "p1", "lane_id": "S5_SURVIVOR", "thesis_state": "STILL_VALID"}
        ],
        "manual_no_touch_ids": [],
        "nav_account_currency": 1_000_000.0,
        "broker_snapshot_sha256": SHA,
        "ledger_tip_sha256": "b" * 64,
        "market_read": _read(),
        "supervision_scorecard": None,
        "candidates": [
            {"pair": "GBP_USD", "side": "LONG", "hold_minutes": 180, "family_id": "S5_SURVIVOR", "nav_exposure_fraction": 0.2}
        ],
    }
    base.update(overrides)
    return run_shadow_brain_cycle(**base)


def test_full_cycle_admits_go_read_and_never_grants_authority() -> None:
    cycle = _cycle()

    assert cycle["inventory_status"] == "RECONCILED"
    assert cycle["admitted_candidate_count"] == 1
    row = cycle["candidate_rows"][0]
    assert row["admitted"] is True
    assert row["shadow_risk_fraction"] > 0.0
    assert cycle["order_intents"] == []
    assert cycle["go_risk_jpy"] == 0.0
    assert cycle["order_authority"] == "NONE"
    assert cycle["live_permission"] is False
    assert cycle["layer2_live_decision_owner"] == LAYER2_LIVE_SOURCE


def test_layer2_is_reserved_for_codex_ai_trader() -> None:
    live = validate_market_read_handoff(_read(source=LAYER2_LIVE_SOURCE))
    assert live["authored_by_brain"] is False
    assert live["live_authoring_reserved_for"] == LAYER2_LIVE_SOURCE

    placeholder = validate_market_read_handoff(_read(source=LAYER2_SHADOW_SOURCE))
    assert placeholder["authored_by_brain"] is True

    # A GO read whose prediction contradicts the traded side is refused.
    with pytest.raises(ShadowBrainError, match="prediction must match"):
        validate_market_read_handoff(
            {
                "read_source": LAYER2_LIVE_SOURCE,
                "declared_regime": "TREND",
                "pair_reads": [
                    {
                        "pair": "EUR_USD",
                        "action": "GO",
                        "side": "LONG",
                        "narrative_sha256": SHA,
                        "predicted_direction": "SHORT",
                        "conviction_conditions": [],
                    }
                ],
            }
        )


def test_no_go_read_and_side_mismatch_refuse_entry() -> None:
    no_go = _cycle(market_read=_read(action="STOP", side="LONG"))
    assert no_go["candidate_rows"][0]["admitted"] is False
    assert "NO_GO_READ" in no_go["candidate_rows"][0]["refusal_reasons"]

    mismatch = _cycle(
        market_read=_read(action="GO", side="SHORT"),
        candidates=[
            {"pair": "GBP_USD", "side": "LONG", "hold_minutes": 180, "family_id": "F", "nav_exposure_fraction": 0.1}
        ],
    )
    assert "READ_SIDE_MISMATCH" in mismatch["candidate_rows"][0]["refusal_reasons"]


def test_unreconciled_inventory_fails_the_whole_cycle_closed() -> None:
    cycle = _cycle(
        broker_positions=[
            {"position_id": "p1", "pair": "EUR_USD", "side": "LONG", "nav_exposure_fraction": 0.2},
            {"position_id": "ghost", "pair": "AUD_USD", "side": "LONG", "nav_exposure_fraction": 0.1},
        ]
    )
    assert cycle["inventory_status"] == "UNRECONCILED_FAIL_CLOSED"
    assert cycle["admitted_candidate_count"] == 0
    assert "INVENTORY_UNRECONCILED" in cycle["candidate_rows"][0]["refusal_reasons"]


def test_metacognition_demotes_unreliable_supervisor_families() -> None:
    unreliable = [
        {"family_id": "S5_SURVIVOR", "action": "GO", "realized_stressed_pips": -5.0}
    ] * 8 + [
        {"family_id": "S5_SURVIVOR", "action": "GO", "realized_stressed_pips": 5.0}
    ] * 4
    scorecard = build_supervision_scorecard(unreliable, window_label="2026-07")
    assert scorecard["supervision_auto_caution_required"] == ["S5_SURVIVOR"]

    cycle = _cycle(supervision_scorecard=scorecard)
    row = cycle["candidate_rows"][0]
    assert row["admitted"] is False
    assert "FAMILY_AUTO_CAUTION_UNRELIABLE_SUPERVISOR" in row["refusal_reasons"]
    assert cycle["layers"]["3_metacognition"]["auto_caution_families"] == ["S5_SURVIVOR"]


def test_close_crossing_and_daily_stop_block_entries() -> None:
    crossing = _cycle(decision_utc=FRIDAY_LATE, market_read=_read(), candidates=[
        {"pair": "GBP_USD", "side": "LONG", "hold_minutes": 720, "family_id": "F", "nav_exposure_fraction": 0.1}
    ])
    assert "HOLD_WOULD_CROSS_NEXT_FX_CLOSE" in crossing["candidate_rows"][0]["refusal_reasons"]

    stopped = _cycle(today_nav_return_fraction=-0.05)
    assert stopped["candidate_rows"][0]["admitted"] is False
    assert "DAILY_STOP_ENGAGED" in stopped["candidate_rows"][0]["refusal_reasons"]


def test_missing_or_zero_exposure_fails_closed_not_coerced() -> None:
    # Regression: a candidate omitting/zeroing nav_exposure_fraction must be
    # refused, never scored as a negligible 0.0001 that bypasses the cap.
    for bad in (None, 0.0, -0.1, "big"):
        candidate = {"pair": "GBP_USD", "side": "LONG", "hold_minutes": 180, "family_id": "F"}
        if bad != "MISSING":
            candidate["nav_exposure_fraction"] = bad
        cycle = _cycle(
            market_read=_read(),
            candidates=[candidate],
        )
        row = cycle["candidate_rows"][0]
        assert row["admitted"] is False
        assert "INVALID_NAV_EXPOSURE_FRACTION" in row["refusal_reasons"]

    omitted = _cycle(
        market_read=_read(),
        candidates=[{"pair": "GBP_USD", "side": "LONG", "hold_minutes": 180, "family_id": "F"}],
    )
    assert "INVALID_NAV_EXPOSURE_FRACTION" in omitted["candidate_rows"][0]["refusal_reasons"]


def test_missing_family_id_cannot_evade_auto_caution() -> None:
    # Regression: an omitted family_id must refuse (UNKNOWN_FAMILY), so a
    # candidate cannot dodge supervisor demotion by not declaring its family.
    cycle = _cycle(
        candidates=[
            {"pair": "GBP_USD", "side": "LONG", "hold_minutes": 180, "nav_exposure_fraction": 0.2}
        ]
    )
    row = cycle["candidate_rows"][0]
    assert row["admitted"] is False
    assert "UNKNOWN_FAMILY" in row["refusal_reasons"]


def test_measured_regime_cell_gates_family_eligibility() -> None:
    from quant_rabbit.regime_classifier_shadow import classify_regime
    from quant_rabbit.regime_family_router import build_family_catalog

    catalog = build_family_catalog(
        [
            {
                "family_id": "S5_SURVIVOR",
                "regime_affinity": ["TREND"],
                "vol_affinity": ["HIGH"],
                "promotion_state": "VALIDATION_REPLICATED",
            }
        ]
    )
    # A steadily trending, volatile series measures TREND/HIGH.
    from datetime import timedelta

    start = MIDWEEK - timedelta(minutes=200)
    closes = [1.10 + i * 0.0004 * (1.5 if i % 2 else 1.0) for i in range(140)]
    candles = [{"time": start + timedelta(minutes=i), "close": c} for i, c in enumerate(closes)]
    measured = classify_regime(candles, as_of_utc=MIDWEEK)

    # Family catalogued for TREND/HIGH: if measurement matches, it may trade.
    matched = _cycle(measured_regime=measured, family_catalog=catalog)
    meta = matched["layers"]["3_metacognition"]
    assert meta["measured_regime"] == measured["regime"]
    assert meta["measured_cell_eligible_families"] is not None

    # A candidate from a family NOT eligible for the measured cell is refused.
    foreign = _cycle(
        measured_regime=measured,
        family_catalog=catalog,
        candidates=[
            {"pair": "GBP_USD", "side": "LONG", "hold_minutes": 180, "family_id": "RANGE_RAIL", "nav_exposure_fraction": 0.2}
        ],
    )
    assert "FAMILY_NOT_ELIGIBLE_FOR_MEASURED_CELL" in foreign["candidate_rows"][0]["refusal_reasons"]


def test_cycle_seal_is_deterministic_and_tamper_evident() -> None:
    from quant_rabbit.shadow_trading_brain import _canonical_sha

    cycle = _cycle()
    body = {k: v for k, v in cycle.items() if k != "cycle_sha256"}
    assert cycle["cycle_sha256"] == _canonical_sha(body)
