from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from quant_rabbit.fast_bot_shadow_orchestrator import (
    CURRENT_CELL_STATE,
    EXPECTED_MATRIX_CARDINALITY,
    MAX_SHADOW_ORCHESTRATION_BYTES,
    PairCausalShadowInput,
    build_fast_bot_shadow_orchestration_v1,
    build_fast_bot_shadow_risk_sizing_identity_v1,
    fast_bot_shadow_orchestration_valid_v1,
)
from quant_rabbit.fast_bot_technical_grid_backtest import (
    CausalTimeframeFeature,
    build_causal_technical_feature_snapshot_v1,
)
from quant_rabbit.fast_bot_technical_hypotheses import (
    build_fast_bot_technical_hypotheses,
)
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS


UTC = timezone.utc
DECISION = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)
CYCLE_ID = "cycle-20260717T120000Z"
TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D")


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _reseal(value: dict[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return {**body, "contract_sha256": _sha(body)}


def _features(*, trend: bool = False) -> list[CausalTimeframeFeature]:
    rows: list[CausalTimeframeFeature] = []
    for timeframe in TIMEFRAMES:
        if trend:
            market = {
                "direction": "UP",
                "phase": "TREND",
                "readiness": "TRIGGERED",
                "location": "MIDDLE_THIRD",
                "value_zone": "FAIR_VALUE",
                "extension": "BALANCED",
                "evidence_complete": True,
            }
            indicators = {
                "close": 101.5,
                "ema_20": 101.0,
                "ema_50": 100.0,
                "ema_slope_20": 3.0,
                "plus_di_14": 31.0,
                "minus_di_14": 14.0,
                "supertrend_dir": 1,
                "rsi_14": 63.0,
                "macd_hist": 0.2,
                "roc_5": 0.4,
                "adx_14": 28.0,
                "atr_pips": 5.0,
                "z_score_20": 0.0,
            }
            series = {
                "rsi_14": [47.0, 49.0, 51.0, 56.0, 63.0],
                "macd_hist": [-0.2, -0.1, 0.0, 0.1, 0.2],
                "adx_14": [20.0, 22.0, 24.0, 26.0, 28.0],
                "atr_pips": [3.0, 3.4, 3.8, 4.4, 5.0],
                "ema_12_minus_50_pips": [-0.2, -0.1, 0.1, 0.5, 1.0],
            }
        else:
            market = {}
            indicators = {"atr_pips": 5.0} if timeframe == "M5" else {}
            series = {}
        rows.append(
            CausalTimeframeFeature(
                timeframe=timeframe,
                complete_candle_close_utc=DECISION,
                market_state=market,
                indicators=indicators,
                indicator_series=series,
            )
        )
    return rows


def _pair_input(pair: str, *, trend: bool = False) -> PairCausalShadowInput:
    snapshot = build_causal_technical_feature_snapshot_v1(
        pair=pair,
        decision_at_utc=DECISION,
        timeframes=_features(trend=trend),
    )
    shadow = build_fast_bot_technical_hypotheses(
        snapshot,
        attempt_direction="UP",
        branch_outcome="ACCEPTED",
        route_family="BREAKOUT_CONTINUATION",
        spread_pips=0.5,
        m5_atr_pips=5.0,
        spread_to_m5_atr=0.1,
    )
    return PairCausalShadowInput(
        feature_snapshot=snapshot,
        hypothesis_shadow=shadow,
        attempt_direction="UP",
        branch_outcome="ACCEPTED",
        route_family="BREAKOUT_CONTINUATION",
        spread_pips=0.5,
        m5_atr_pips=5.0,
        spread_to_m5_atr=0.1,
    )


def _pair_inputs() -> list[PairCausalShadowInput]:
    return [_pair_input(pair) for pair in DEFAULT_TRADER_PAIRS]


def _risk(*, cycle_id: str = CYCLE_ID) -> dict[str, Any]:
    return build_fast_bot_shadow_risk_sizing_identity_v1(
        cycle_id=cycle_id,
        as_of_utc=DECISION + timedelta(seconds=5),
        sizing_nav_jpy=250_000.0,
        daily_loss_capacity_before_open_jpy=25_000.0,
        fresh_open_risk_jpy=3_000.0,
        pending_risk_jpy=2_000.0,
        per_trade_risk_cap_jpy=1_000.0,
        broker_snapshot_sha256="a" * 64,
        daily_target_state_sha256="b" * 64,
        execution_ledger_tip_sha256="c" * 64,
    )


def _build(
    *,
    pair_inputs: list[PairCausalShadowInput] | None = None,
    risk: dict[str, Any] | None = None,
    observed: datetime = DECISION + timedelta(seconds=10),
    deadline: datetime = DECISION + timedelta(minutes=1),
) -> dict[str, Any]:
    return build_fast_bot_shadow_orchestration_v1(
        cycle_id=CYCLE_ID,
        cycle_started_at_utc=DECISION - timedelta(seconds=5),
        decision_at_utc=DECISION,
        freshness_deadline_utc=deadline,
        observed_at_utc=observed,
        pair_inputs=pair_inputs or _pair_inputs(),
        risk_sizing_identity=risk or _risk(),
    )


def test_exact_28_by_182_matrix_is_compact_h08_shadow_with_zero_go() -> None:
    started = time.perf_counter()
    artifact = _build()
    elapsed = time.perf_counter() - started

    assert artifact["matrix"]["pair_count"] == 28
    assert artifact["matrix"]["candidate_count_per_pair"] == 182
    assert (
        artifact["matrix"]["matrix_cardinality"] == EXPECTED_MATRIX_CARDINALITY == 5096
    )
    assert artifact["matrix"]["state_runs"] == [
        {
            "flat_start_inclusive": 0,
            "flat_end_exclusive": 5096,
            "length": 5096,
            "state": CURRENT_CELL_STATE,
        }
    ]
    assert [row["pair"] for row in artifact["pair_routes"]] == list(
        DEFAULT_TRADER_PAIRS
    )
    assert all(
        row["selected_hypothesis_id"] == "H08" for row in artifact["pair_routes"]
    )
    assert all(
        row["deep_validation"] == "DETERMINISTIC_REBUILD_EXACT_MATCH"
        for row in artifact["pair_routes"]
    )
    assert artifact["go_count"] == 0
    assert artifact["go_candidate_refs"] == []
    assert artifact["go_risk_jpy"] == 0.0
    assert artifact["order_intents"] == []
    assert artifact["order_authority"] == "NONE"
    assert artifact["economic_status"] == "UNPROVEN_NO_PROFIT_CLAIM"
    assert artifact["profitability_proven"] is False
    assert (
        len(json.dumps(artifact, separators=(",", ":")).encode())
        < MAX_SHADOW_ORCHESTRATION_BYTES
    )
    assert elapsed < 5.0
    assert fast_bot_shadow_orchestration_valid_v1(
        artifact, now_utc=DECISION + timedelta(seconds=20)
    )


def test_pair_input_order_does_not_change_canonical_artifact() -> None:
    inputs = _pair_inputs()
    assert _build(pair_inputs=inputs) == _build(pair_inputs=list(reversed(inputs)))


def test_deep_validation_rejects_resealed_hypothesis_tampering() -> None:
    inputs = _pair_inputs()
    target = inputs[0]
    tampered_shadow = deepcopy(target.hypothesis_shadow)
    h08 = next(
        row for row in tampered_shadow["hypotheses"] if row["hypothesis_id"] == "H08"
    )
    h08["evidence"] = ["FORGED_H08_CAUSE"]
    row_body = {key: item for key, item in h08.items() if key != "hypothesis_sha256"}
    h08["hypothesis_sha256"] = _sha(row_body)
    tampered_shadow = _reseal(tampered_shadow)
    inputs[0] = PairCausalShadowInput(
        feature_snapshot=target.feature_snapshot,
        hypothesis_shadow=tampered_shadow,
        attempt_direction=target.attempt_direction,
        branch_outcome=target.branch_outcome,
        route_family=target.route_family,
        spread_pips=target.spread_pips,
        m5_atr_pips=target.m5_atr_pips,
        spread_to_m5_atr=target.spread_to_m5_atr,
    )

    with pytest.raises(ValueError, match="deep validation"):
        _build(pair_inputs=inputs)


def test_active_directional_route_cannot_be_mislabeled_as_h08() -> None:
    inputs = _pair_inputs()
    trend_input = _pair_input(DEFAULT_TRADER_PAIRS[0], trend=True)
    rows = {
        row["hypothesis_id"]: row for row in trend_input.hypothesis_shadow["hypotheses"]
    }
    assert rows["H01"]["status"] == "ACTIVE_SHADOW"
    assert rows["H08"]["status"] == "INACTIVE_SHADOW"
    inputs[0] = trend_input

    with pytest.raises(ValueError, match="not the active H08"):
        _build(pair_inputs=inputs)


def test_incomplete_or_duplicate_pair_universe_fails_closed() -> None:
    inputs = _pair_inputs()
    with pytest.raises(ValueError, match="exactly all 28"):
        _build(pair_inputs=inputs[:-1])

    inputs[-1] = inputs[0]
    with pytest.raises(ValueError, match="unknown, duplicated"):
        _build(pair_inputs=inputs)


def test_cycle_and_risk_freshness_are_content_bound() -> None:
    with pytest.raises(ValueError, match="different cycle"):
        _build(risk=_risk(cycle_id="other-cycle"))
    with pytest.raises(ValueError, match="started <= decision"):
        _build(
            observed=DECISION + timedelta(minutes=1),
            deadline=DECISION + timedelta(minutes=1),
        )

    artifact = _build()
    assert not fast_bot_shadow_orchestration_valid_v1(
        artifact, now_utc=DECISION + timedelta(minutes=1)
    )
    assert not fast_bot_shadow_orchestration_valid_v1(
        artifact, now_utc=DECISION + timedelta(seconds=9)
    )


def test_risk_identity_subtracts_open_and_pending_exactly_once_and_rejects_tamper() -> (
    None
):
    risk = _risk()
    assert risk["available_capacity_after_open_and_pending_jpy"] == 20_000.0
    assert risk["candidate_risk_allocations"] == []
    assert risk["go_risk_jpy"] == 0.0

    tampered = deepcopy(risk)
    tampered["available_capacity_after_open_and_pending_jpy"] = 25_000.0
    tampered = _reseal(tampered)
    with pytest.raises(ValueError, match="identity digest"):
        _build(risk=tampered)

    with pytest.raises(ValueError, match="exceeds current available capacity"):
        build_fast_bot_shadow_risk_sizing_identity_v1(
            cycle_id=CYCLE_ID,
            as_of_utc=DECISION + timedelta(seconds=5),
            sizing_nav_jpy=250_000.0,
            daily_loss_capacity_before_open_jpy=4_000.0,
            fresh_open_risk_jpy=3_000.0,
            pending_risk_jpy=1_000.0,
            per_trade_risk_cap_jpy=1.0,
            broker_snapshot_sha256="a" * 64,
            daily_target_state_sha256="b" * 64,
            execution_ledger_tip_sha256="c" * 64,
        )


def test_resealed_go_or_candidate_order_forgery_is_rejected() -> None:
    artifact = _build()
    forged_go = deepcopy(artifact)
    forged_go["go_count"] = 1
    forged_go["go_candidate_refs"] = ["EUR_USD:H01:DIRECT:BASE"]
    forged_go = _reseal(forged_go)
    assert not fast_bot_shadow_orchestration_valid_v1(
        forged_go, now_utc=DECISION + timedelta(seconds=20)
    )

    forged_order = deepcopy(artifact)
    forged_order["matrix"]["candidate_order"][0] = "FORGED"
    forged_order = _reseal(forged_order)
    assert not fast_bot_shadow_orchestration_valid_v1(
        forged_order, now_utc=DECISION + timedelta(seconds=20)
    )
