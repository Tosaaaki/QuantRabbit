from __future__ import annotations

import io
import hashlib
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from quant_rabbit.cli import main
from quant_rabbit.execution_timing_contracts import (
    TP_PROGRESS_REPAIR_LIVE_EVIDENCE_BOUNDARY_UTC,
    TP_PROGRESS_REPAIR_REPLAY_CONTRACT,
    TP_PROGRESS_REPAIR_REPLAY_FIELD,
)
from quant_rabbit.forecast_precision import (
    bidask_replay_precision_rule_digest,
    canonical_bidask_replay_precision_rule,
)
from quant_rabbit.models import (
    MarketContext,
    OrderIntent,
    OrderType,
    Owner,
    Side,
    TradeMethod,
)
from quant_rabbit.strategy.forecast_technical_context import (
    build_forecast_technical_context,
)
from quant_rabbit.strategy.intent_generator import (
    SIZING_ACTUAL_ANCHOR_VERSION,
    SIZING_ACTUAL_RECEIPT_DIRECTORY,
    sizing_actual_anchor_receipt,
    sizing_conversion_snapshot_receipt_from_payload,
)
from quant_rabbit.self_improvement_audit import (
    PROFITABILITY_DISCIPLINE_CODES,
    PROJECTION_PENDING_EXPIRY_GRACE_SECONDS,
    STATUS_ACTION_REQUIRED,
    STATUS_BLOCKED,
    SelfImprovementAuditor,
    _chronological_directional_calibration_rows,
    _coverage_market_evidence_refresh,
    _deduped_directional_calibration_rows,
    _effect_metrics,
    _directional_forecast_invalidation_first_like,
    _directional_forecast_no_touch_timeout_like,
    _directional_forecast_target_timeout_like,
    _gateway_close_recovery_observation,
    _intent_findings,
    _intent_artifact_freshness,
    _intent_live_readiness_family_breakdown,
    _no_live_ready_next_action,
    _normalized_pending_order_type,
    _profit_capture_miss_findings,
    _profitability_findings,
    _position_guardian_runtime_status,
    _projection_expired,
    _report_perspective_alignment_text,
    _root_cause_focus,
    _select_independent_directional_calibration_rows,
    _select_independent_legacy_directional_calibration_rows,
    _top_intent_blockers,
    _top_intent_live_readiness_blockers,
)
from quant_rabbit.paths import DEFAULT_EXECUTION_LEDGER_DB, DEFAULT_SELF_IMPROVEMENT_HISTORY_DB


_REPLAY_NOW = datetime(2026, 7, 14, 0, 0, tzinfo=timezone.utc)
_NEGATIVE_TP_ROTATION_BLOCKER = "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"
_TP_SIZING_RECEIPT_TEMP = tempfile.TemporaryDirectory()
_TP_SIZING_RECEIPT_ROOT = (
    Path(_TP_SIZING_RECEIPT_TEMP.name) / SIZING_ACTUAL_RECEIPT_DIRECTORY
)
_TP_SIZING_RECEIPT_ROOT.mkdir(parents=True, exist_ok=True)


def _tp_acquisition_broker_snapshot() -> dict[str, object]:
    timestamp = _REPLAY_NOW.isoformat().replace("+00:00", "Z")
    return {
        "fetched_at_utc": timestamp,
        "quotes": {
            "EUR_USD": {
                "bid": 1.0999,
                "ask": 1.1001,
                "timestamp_utc": timestamp,
            },
            "USD_CAD": {
                "bid": 1.3499,
                "ask": 1.3501,
                "timestamp_utc": timestamp,
            },
        },
        "home_conversions": {"USD": 100.0, "CAD": 100.0},
    }


def _tp_acquisition_intents(results: list[dict[str, object]]) -> dict[str, object]:
    return {
        "generated_at_utc": _REPLAY_NOW.isoformat().replace("+00:00", "Z"),
        "results": results,
    }


def _tp_acquisition_route_result(
    *,
    lane_id: str,
    metadata: dict[str, object] | None = None,
    blocker_codes: list[str] | None = None,
    authoritative_codes: list[str] | None = None,
    include_authoritative_codes: bool = True,
    estimated_margin_jpy_override: float | None = None,
) -> dict[str, object]:
    codes = blocker_codes or [_NEGATIVE_TP_ROTATION_BLOCKER]
    payload_metadata = dict(metadata or {})
    payload_metadata.setdefault("desk", "failure_trader")
    predictive_scout = (
        payload_metadata.get("positive_rotation_mode")
        == "PREDICTIVE_SCOUT_FORWARD_EVIDENCE"
    )
    pair = "USD_CAD" if predictive_scout else "EUR_USD"
    side = "LONG" if predictive_scout else "SHORT"
    order_type = "LIMIT" if predictive_scout else "STOP-ENTRY"
    entry = 1.3500 if predictive_scout else 1.1000
    take_profit = 1.3510 if predictive_scout else 1.0990
    stop_loss = 1.3493 if predictive_scout else 1.1010
    pip_factor = 10_000.0
    loss_pips = abs(stop_loss - entry) * pip_factor
    reward_pips = abs(take_profit - entry) * pip_factor
    jpy_per_pip = 10.0
    quote_to_jpy = jpy_per_pip * pip_factor / 1000.0
    margin_rate = 0.04
    risk_jpy = loss_pips * jpy_per_pip
    reward_jpy = reward_pips * jpy_per_pip
    estimated_margin_jpy = 1000.0 * entry * quote_to_jpy * margin_rate
    if estimated_margin_jpy_override is not None:
        estimated_margin_jpy = estimated_margin_jpy_override
    effective_cap = float(
        payload_metadata.setdefault(
            "loss_asymmetry_guard_effective_max_loss_jpy",
            500.0,
        )
    )
    payload_metadata.setdefault("loss_asymmetry_guard_loss_cap_jpy", 500.0)
    payload_metadata.setdefault("loss_asymmetry_guard_base_max_loss_jpy", 500.0)
    payload_metadata.setdefault("max_loss_jpy", 500.0)
    payload_metadata.update(
        {
            "sizing_actual_risk_jpy": round(risk_jpy, 4),
            "sizing_actual_reward_jpy": round(reward_jpy, 4),
            "sizing_actual_loss_pips": round(loss_pips, 4),
            "sizing_actual_reward_pips": round(reward_pips, 4),
            "sizing_actual_units": 1000,
            "sizing_actual_jpy_per_pip": round(jpy_per_pip, 8),
            "sizing_actual_quote_to_jpy": round(quote_to_jpy, 8),
            "sizing_actual_margin_rate": round(margin_rate, 8),
            "sizing_actual_estimated_margin_jpy": round(
                estimated_margin_jpy,
                4,
            ),
            "sizing_actual_risk_basis": "RISK_ENGINE_VALIDATED_ORDER",
            "sizing_actual_risk_cap_jpy": round(effective_cap, 4),
            "sizing_actual_risk_cap_utilization": round(
                risk_jpy / effective_cap,
                6,
            ),
        }
    )
    conversion_receipt = sizing_conversion_snapshot_receipt_from_payload(
        pair,
        _tp_acquisition_broker_snapshot(),
    )
    if conversion_receipt is None:
        raise AssertionError("test broker snapshot must produce conversion receipt")
    payload_metadata["sizing_actual_conversion_receipt"] = conversion_receipt
    result: dict[str, object] = {
        "lane_id": lane_id,
        "status": "DRY_RUN_BLOCKED",
        "intent": {
            "pair": pair,
            "side": side,
            "order_type": order_type,
            "owner": "trader",
            "units": 1000,
            "entry": entry,
            "tp": take_profit,
            "sl": stop_loss,
            "market_context": {"method": "BREAKOUT_FAILURE"},
            "metadata": payload_metadata,
        },
        "risk_metrics": {
            "entry_price": entry,
            "loss_pips": loss_pips,
            "reward_pips": reward_pips,
            "risk_jpy": risk_jpy,
            "reward_jpy": reward_jpy,
            "reward_risk": reward_pips / loss_pips,
            "jpy_per_pip": jpy_per_pip,
            "estimated_margin_jpy": estimated_margin_jpy,
        },
        "risk_issues": [
            {
                "code": code,
                "message": f"blocking issue {code}",
                "severity": "BLOCK",
            }
            for code in codes
        ],
        "live_strategy_issues": [],
        "live_blockers": [],
    }
    if include_authoritative_codes:
        result["live_blocker_codes"] = list(
            authoritative_codes if authoritative_codes is not None else codes
        )
    typed_intent = OrderIntent(
        pair=pair,
        side=Side.parse(side),
        order_type=OrderType.parse(order_type),
        units=1000,
        entry=entry,
        tp=take_profit,
        sl=stop_loss,
        thesis="test sizing anchor",
        owner=Owner.TRADER,
        market_context=MarketContext(
            regime="",
            narrative="",
            chart_story="",
            method=TradeMethod.BREAKOUT_FAILURE,
            invalidation="",
        ),
        metadata=payload_metadata,
    )
    sizing_anchor = sizing_actual_anchor_receipt(
        lane_id=lane_id,
        intent=typed_intent,
        risk_metrics=result["risk_metrics"],
        sizing_metadata=payload_metadata,
    )
    if sizing_anchor is not None:
        digest = sizing_anchor["receipt_sha256"]
        payload_metadata.update(
            {
                "sizing_actual_anchor_contract": SIZING_ACTUAL_ANCHOR_VERSION,
                "sizing_actual_anchor_sha256": digest,
            }
        )
        path = _TP_SIZING_RECEIPT_ROOT / f"{digest}.json"
        path.write_text(
            json.dumps(
                sizing_anchor,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return result


def _valid_tp_collection_metadata() -> dict[str, object]:
    return {
        "desk": "failure_trader",
        "loss_asymmetry_guard_active": True,
        "capture_economics_status": "NEGATIVE_EXPECTANCY",
        "loss_asymmetry_guard_mode": "CAP_AVG_WIN",
        "loss_asymmetry_guard_relaxed": False,
        "loss_asymmetry_guard_loss_cap_jpy": 500.0,
        "loss_asymmetry_guard_base_max_loss_jpy": 500.0,
        "loss_asymmetry_guard_effective_max_loss_jpy": 500.0,
        "max_loss_jpy": 500.0,
        "attach_take_profit_on_fill": True,
        "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
        "tp_target_intent": "HARVEST",
        "opportunity_mode": "HARVEST",
        "capture_take_profit_exact_vehicle_required": True,
        "capture_take_profit_scope": "PAIR_SIDE_METHOD_VEHICLE",
        "capture_take_profit_scope_key": (
            "EUR_USD|SHORT|BREAKOUT_FAILURE|STOP|TAKE_PROFIT_ORDER"
        ),
        "capture_take_profit_vehicle": "STOP",
        "capture_take_profit_metrics_source": (
            "data/execution_ledger.db:exact_vehicle_take_profit"
        ),
        "positive_rotation_mode": "TP_PROOF_COLLECTION_HARVEST",
        "positive_rotation_proof_collection_ready": True,
        "positive_rotation_proof_collection_mode": "TP_PROOF_COLLECTION_HARVEST",
        "positive_rotation_proof_collection_min_trades": 5,
        "positive_rotation_proof_collection_target_trades": 20,
        "positive_rotation_proof_collection_gap_trades": 13,
        "capture_take_profit_trades": 7,
        "capture_take_profit_wins": 7,
        "capture_take_profit_losses": 0,
        "capture_take_profit_expectancy_jpy": 918.2276,
        "capture_take_profit_net_jpy": 6427.5931,
        "capture_take_profit_avg_win_jpy": 918.2276,
        "capture_take_profit_avg_loss_jpy": 0.0,
        "capture_avg_win_jpy": 500.0,
        "capture_avg_loss_jpy": 1042.0,
        "capture_market_close_expectancy_jpy": -756.6,
        "positive_rotation_confidence_method": (
            "WILSON_LOWER_BOUND_STRESS_EXPECTANCY"
        ),
        "positive_rotation_confidence_z": 1.96,
        "positive_rotation_tp_wins": 7,
        "positive_rotation_tp_trades": 7,
        "positive_rotation_tp_win_rate_lower": 0.645661,
        "positive_rotation_loss_proxy_jpy": 1042.0,
        "positive_rotation_pessimistic_expectancy_jpy": 223.6428,
    }


def _valid_predictive_scout_metadata() -> dict[str, object]:
    rule_name = (
        "USD_CAD_DOWN_H31_60m_C0p50_0p65_FADE_TO_UP_S5_"
        "BIDASK_CONTRARIAN_HARVEST_TP10_SL7"
    )
    rule = canonical_bidask_replay_precision_rule(rule_name)
    if rule is None:
        raise AssertionError("canonical predictive-SCOUT rule is missing")
    return {
        "loss_asymmetry_guard_active": True,
        "capture_economics_status": "NEGATIVE_EXPECTANCY",
        "attach_take_profit_on_fill": True,
        "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
        "tp_target_intent": "HARVEST",
        "opportunity_mode": "HARVEST",
        "broker_stop_loss_mode": "INTENT_SL",
        "desk": "failure_trader",
        "campaign_role": "BIDASK_REPLAY_CONTRARIAN_SCOUT",
        "forecast_direction": "DOWN",
        "forecast_confidence": 0.60,
        "forecast_horizon_min": 45,
        "forecast_cycle_id": "test-usdcad-down-c050-065-h31-60",
        "predictive_scout": True,
        "predictive_scout_source": "BIDASK_REPLAY_PRECISION",
        "predictive_scout_rule_name": rule_name,
        "predictive_scout_rule_digest": bidask_replay_precision_rule_digest(rule),
        "predictive_scout_rule_is_vehicle_proof": False,
        "predictive_scout_vehicle_proof_status": "UNPROVEN_PASSIVE_LIMIT",
        "predictive_scout_hypothesis": (
            "REPRODUCIBLE_FORECAST_FAILURE_CONTRARIAN"
        ),
        "predictive_scout_promotion_allowed": False,
        "bidask_replay_precision_seed_rule": rule,
        "positive_rotation_mode": "PREDICTIVE_SCOUT_FORWARD_EVIDENCE",
        "positive_rotation_live_ready": False,
        "bidask_replay_precision_seed": True,
    }


def _valid_tp_proven_metadata() -> dict[str, object]:
    return {
        "desk": "failure_trader",
        "loss_asymmetry_guard_active": True,
        "capture_economics_status": "NEGATIVE_EXPECTANCY",
        "loss_asymmetry_guard_mode": "TP_PROVEN_RELAXED",
        "loss_asymmetry_guard_relaxed": True,
        "loss_asymmetry_guard_loss_cap_jpy": 100.0,
        "loss_asymmetry_guard_base_max_loss_jpy": 500.0,
        "loss_asymmetry_guard_effective_max_loss_jpy": 500.0,
        "max_loss_jpy": 500.0,
        "attach_take_profit_on_fill": True,
        "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
        "tp_target_intent": "HARVEST",
        "opportunity_mode": "HARVEST",
        "capture_take_profit_exact_vehicle_required": True,
        "capture_take_profit_scope": "PAIR_SIDE_METHOD_VEHICLE",
        "capture_take_profit_scope_key": (
            "EUR_USD|SHORT|BREAKOUT_FAILURE|STOP|TAKE_PROFIT_ORDER"
        ),
        "capture_take_profit_vehicle": "STOP",
        "capture_take_profit_metrics_source": (
            "data/execution_ledger.db:exact_vehicle_take_profit"
        ),
        "positive_rotation_mode": "TP_PROVEN_HARVEST",
        "positive_rotation_live_ready": True,
        "capture_take_profit_trades": 20,
        "capture_take_profit_wins": 20,
        "capture_take_profit_losses": 0,
        "capture_take_profit_expectancy_jpy": 500.0,
        "capture_take_profit_net_jpy": 10000.0,
        "capture_take_profit_avg_win_jpy": 500.0,
        "capture_take_profit_avg_loss_jpy": 0.0,
        "capture_avg_win_jpy": 100.0,
        "capture_avg_loss_jpy": 100.0,
        "capture_market_close_expectancy_jpy": -100.0,
        "positive_rotation_confidence_method": (
            "WILSON_LOWER_BOUND_STRESS_EXPECTANCY"
        ),
        "positive_rotation_confidence_z": 1.96,
        "positive_rotation_tp_wins": 20,
        "positive_rotation_tp_trades": 20,
        "positive_rotation_tp_win_rate_lower": 0.83887,
        "positive_rotation_loss_proxy_jpy": 100.0,
        "positive_rotation_pessimistic_expectancy_jpy": 403.3219,
    }


def _replay_window(now: datetime) -> dict[str, object]:
    cutoff = now - timedelta(hours=744)
    return {
        "from_utc": cutoff.isoformat(),
        "canceled_from_utc": cutoff.isoformat(),
        "market_close_from_utc": cutoff.isoformat(),
        "market_close_anchor_utc": None,
        "to_utc": now.isoformat(),
        "lookback_hours": 744.0,
        "post_close_hours": 6.0,
        "max_events": 80,
    }


def _loss_close_replay_rows(
    *,
    audited: int = 28,
    pre_repair_samples: int = 23,
    raw_misses: int = 8,
    replay_triggers: int = 7,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(audited):
        close_at = (
            "2026-06-22T17:00:00+00:00"
            if index < pre_repair_samples
            else "2026-06-22T18:00:00+00:00"
        )
        fill_at = (
            "2026-06-22T16:00:00+00:00"
            if index < pre_repair_samples
            else "2026-06-22T17:55:00+00:00"
        )
        raw_miss = index < raw_misses
        replay_trigger = index < replay_triggers
        rows.append(
            {
                "trade_id": f"loss-{index}",
                "pair": "EUR_USD",
                "side": "LONG",
                "units": 1000,
                "entry": 1.1,
                "close_price": 1.09,
                "realized_pl_jpy": -10.0,
                "exit_reason": "STOP_LOSS_ORDER",
                "candles_available": 1,
                "fill_at_utc": fill_at,
                "close_at_utc": close_at,
                "mfe_at_utc": close_at if raw_miss else None,
                "repair_replay_trigger_at_utc": (
                    close_at if replay_trigger else None
                ),
                "profit_capture_missed_before_loss_close": raw_miss,
                "repair_replay_triggered_before_loss_close": replay_trigger,
            }
        )
    return rows


class SelfImprovementAuditorTest(unittest.TestCase):
    def test_perspective_alignment_report_keeps_later_opposite_rail_view(self) -> None:
        text = _report_perspective_alignment_text(
            {
                "status": "RANGE_METHOD_MISMATCH_REPAIR_REQUIRED",
                "range_forecast_method_mismatch_groups": 4,
                "range_forecast_method_mismatch_lanes": 9,
                "range_forecast_method_mismatch_top": [
                    {"pair": "AUD_JPY", "direction": "LONG", "method_mismatch_lanes": 3, "range_rotation_lanes": 0},
                    {"pair": "AUD_JPY", "direction": "SHORT", "method_mismatch_lanes": 3, "range_rotation_lanes": 0},
                    {"pair": "USD_CHF", "direction": "LONG", "method_mismatch_lanes": 2, "range_rotation_lanes": 1},
                    {
                        "pair": "USD_CAD",
                        "direction": "LONG",
                        "method_mismatch_lanes": 1,
                        "range_rotation_lanes": 1,
                        "range_rotation_other_side_lanes": 2,
                        "range_rotation_other_side_directions": [{"code": "SHORT", "count": 2}],
                        "range_rotation_other_side_top_live_blocker_codes": [
                            {"code": "SPREAD_TOO_WIDE", "count": 2}
                        ],
                    },
                ],
            }
        )

        self.assertIn("USD_CAD LONG mismatch=1", text)
        self.assertIn("other_rail=SHORT", text)
        self.assertIn("other_blockers=SPREAD_TOO_WIDE", text)

    def test_default_history_db_is_separate_from_execution_ledger(self) -> None:
        auditor = SelfImprovementAuditor()

        self.assertEqual(auditor.db_path, DEFAULT_EXECUTION_LEDGER_DB)
        self.assertEqual(auditor.history_db_path, DEFAULT_SELF_IMPROVEMENT_HISTORY_DB)
        self.assertNotEqual(auditor.history_db_path, auditor.db_path)

    def test_pending_order_type_normalization_matches_broker_and_intents(self) -> None:
        self.assertEqual(_normalized_pending_order_type("LIMIT_ORDER"), "LIMIT")
        self.assertEqual(_normalized_pending_order_type("STOP"), "STOP-ENTRY")
        self.assertEqual(_normalized_pending_order_type("STOP_ORDER"), "STOP-ENTRY")
        self.assertEqual(_normalized_pending_order_type("STOP_ENTRY"), "STOP-ENTRY")
        self.assertEqual(_normalized_pending_order_type("MARKET_IF_TOUCHED_ORDER"), "MARKET-IF-TOUCHED")

    def test_profit_capture_miss_becomes_p0_while_target_open(self) -> None:
        findings = _profit_capture_miss_findings(
            run_id=_REPLAY_NOW.isoformat(),
            target_open=True,
            timing_payload={
                "generated_at_utc": _REPLAY_NOW.isoformat(),
                "status": "OK",
                "fetch_errors": [],
                "precision": {
                    TP_PROGRESS_REPAIR_REPLAY_FIELD: TP_PROGRESS_REPAIR_REPLAY_CONTRACT,
                    "price_basis": "OANDA_M1_BID_ASK_CANDLES",
                    "granularity": "M1",
                },
                "window": _replay_window(_REPLAY_NOW),
                "summary": {
                    "loss_closes_audited": 1,
                    "loss_closes_profit_capture_missed": 1,
                    "loss_closes_profit_capture_missed_rate": 1.0,
                    "loss_closes_repair_replay_triggered": 1,
                    "loss_closes_repair_replay_triggered_rate": 1.0,
                    "loss_close_repair_replay_profit_capture_jpy": 126.0,
                    "loss_close_repair_replay_delta_jpy": 466.2,
                    "stop_loss_closes_profit_capture_missed": 1,
                    "loss_close_estimated_capture_gap_jpy": 302.4,
                    "loss_close_actual_pl_jpy": -340.2,
                    "loss_close_counterfactual_profit_capture_pl_jpy": 105.84,
                    "loss_close_counterfactual_profit_capture_delta_jpy": 446.04,
                    "loss_close_counterfactual_profit_capture_jpy": 105.84,
                    "tp_progress_repair_live_evidence_boundary_utc": (
                        TP_PROGRESS_REPAIR_LIVE_EVIDENCE_BOUNDARY_UTC
                    ),
                    "tp_progress_repair_live_evidence_status": (
                        "POST_REPAIR_REPLAY_FAILURES_PRESENT"
                    ),
                    "pre_repair_historical_loss_closes_audited": 0,
                    "pre_repair_historical_loss_closes_profit_capture_missed": 0,
                    "pre_repair_historical_loss_closes_repair_replay_triggered": 0,
                    "post_repair_live_evidence_loss_closes_audited": 1,
                    "post_repair_live_evidence_loss_closes_profit_capture_missed": 1,
                    "post_repair_live_evidence_loss_closes_repair_replay_triggered": 1,
                },
                "loss_close_regrets": [
                    {
                        "trade_id": "472792",
                        "lane_id": "range_trader:USD_JPY:SHORT:RANGE_ROTATION",
                        "pair": "USD_JPY",
                        "side": "SHORT",
                        "exit_reason": "STOP_LOSS_ORDER",
                        "realized_pl_jpy": -340.2,
                        "units": 1000,
                        "entry": 150.0,
                        "close_price": 149.9,
                        "candles_available": 20,
                        "fill_at_utc": "2026-07-13T11:00:00+00:00",
                        "close_at_utc": "2026-07-13T12:00:00+00:00",
                        "mfe_pips_before_loss_close": 4.8,
                        "mfe_at_utc": "2026-07-13T11:50:00+00:00",
                        "tp_progress_before_loss_close": 0.86,
                        "estimated_mfe_jpy_before_loss_close": 302.4,
                        "profit_capture_missed_before_loss_close": True,
                        "profit_capture_counterfactual_exit": "TP_PROGRESS_CAPTURE",
                        "profit_capture_counterfactual_pips": 3.0,
                        "profit_capture_counterfactual_jpy": 105.84,
                        "profit_capture_counterfactual_net_improvement_jpy": 446.04,
                        "repair_replay_triggered_before_loss_close": True,
                        "repair_replay_exit": "TP_PROGRESS_PRODUCTION_GATE_REPLAY",
                        "repair_replay_trigger_at_utc": (
                            "2026-07-13T11:55:00+00:00"
                        ),
                        "repair_replay_profit_pips": 2.0,
                        "repair_replay_noise_floor_pips": 1.6,
                        "repair_replay_counterfactual_jpy": 126.0,
                        "repair_replay_counterfactual_net_improvement_jpy": 466.2,
                    }
                ],
            },
        )

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["priority"], "P0")
        self.assertEqual(findings[0]["code"], "LOSS_CLOSE_PROFIT_CAPTURE_MISSED")
        self.assertEqual(
            findings[0]["evidence"]["top_profit_capture_misses"][0]["trade_id"],
            "472792",
        )
        self.assertEqual(
            findings[0]["evidence"]["loss_close_counterfactual_profit_capture_delta_jpy"],
            446.04,
        )
        self.assertEqual(
            findings[0]["evidence"]["top_profit_capture_misses"][0][
                "profit_capture_counterfactual_net_improvement_jpy"
            ],
            446.04,
        )
        self.assertEqual(findings[0]["evidence"]["loss_closes_repair_replay_triggered"], 1)
        self.assertEqual(
            findings[0]["evidence"]["top_repair_replay_triggers"][0]["trade_id"],
            "472792",
        )

    def test_pre_repair_tp_progress_replay_history_is_p1_until_post_repair_sample(self) -> None:
        findings = _profit_capture_miss_findings(
            run_id=_REPLAY_NOW.isoformat(),
            target_open=True,
            timing_payload={
                "generated_at_utc": _REPLAY_NOW.isoformat(),
                "status": "OK",
                "fetch_errors": [],
                "precision": {
                    TP_PROGRESS_REPAIR_REPLAY_FIELD: TP_PROGRESS_REPAIR_REPLAY_CONTRACT,
                    "price_basis": "OANDA_M1_BID_ASK_CANDLES",
                    "granularity": "M1",
                },
                "window": _replay_window(_REPLAY_NOW),
                "summary": {
                    "loss_closes_audited": 34,
                    "loss_closes_profit_capture_missed": 14,
                    "loss_closes_profit_capture_missed_rate": 0.4118,
                    "loss_closes_repair_replay_triggered": 13,
                    "loss_closes_repair_replay_triggered_rate": 0.3824,
                    "tp_progress_repair_live_evidence_boundary_utc": (
                        TP_PROGRESS_REPAIR_LIVE_EVIDENCE_BOUNDARY_UTC
                    ),
                    "tp_progress_repair_live_evidence_status": "WAITING_FOR_POST_REPAIR_SAMPLE",
                    "pre_repair_historical_loss_closes_audited": 34,
                    "pre_repair_historical_loss_closes_profit_capture_missed": 14,
                    "pre_repair_historical_loss_closes_repair_replay_triggered": 13,
                    "post_repair_live_evidence_loss_closes_audited": 0,
                    "post_repair_live_evidence_loss_closes_profit_capture_missed": 0,
                    "post_repair_live_evidence_loss_closes_repair_replay_triggered": 0,
                },
                "loss_close_regrets": _loss_close_replay_rows(
                    audited=34,
                    pre_repair_samples=34,
                    raw_misses=14,
                    replay_triggers=13,
                ),
            },
        )

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["priority"], "P1")
        self.assertIn("pre-repair replay trigger", findings[0]["message"])
        self.assertEqual(
            findings[0]["evidence"]["post_repair_live_evidence_loss_closes_audited"],
            0,
        )
        self.assertEqual(
            findings[0]["evidence"][
                "post_repair_live_evidence_loss_closes_repair_replay_triggered"
            ],
            0,
        )
        self.assertEqual(
            findings[0]["evidence"]["pre_repair_historical_loss_closes_repair_replay_triggered"],
            13,
        )

    def test_post_repair_clean_replay_suppresses_historical_profit_capture_p1(self) -> None:
        findings = _profit_capture_miss_findings(
            run_id=_REPLAY_NOW.isoformat(),
            target_open=True,
            timing_payload={
                "generated_at_utc": _REPLAY_NOW.isoformat(),
                "status": "OK",
                "fetch_errors": [],
                "precision": {
                    TP_PROGRESS_REPAIR_REPLAY_FIELD: TP_PROGRESS_REPAIR_REPLAY_CONTRACT,
                    "price_basis": "OANDA_M1_BID_ASK_CANDLES",
                    "granularity": "M1",
                },
                "window": _replay_window(_REPLAY_NOW),
                "summary": {
                    "loss_closes_audited": 28,
                    "loss_closes_profit_capture_missed": 8,
                    "loss_closes_repair_replay_triggered": 7,
                    "tp_progress_repair_live_evidence_boundary_utc": (
                        TP_PROGRESS_REPAIR_LIVE_EVIDENCE_BOUNDARY_UTC
                    ),
                    "tp_progress_repair_live_evidence_status": "POST_REPAIR_REPLAY_CLEAN",
                    "pre_repair_historical_loss_closes_audited": 23,
                    "pre_repair_historical_loss_closes_profit_capture_missed": 8,
                    "pre_repair_historical_loss_closes_repair_replay_triggered": 7,
                    "post_repair_live_evidence_loss_closes_audited": 5,
                    "post_repair_live_evidence_loss_closes_profit_capture_missed": 0,
                    "post_repair_live_evidence_loss_closes_repair_replay_triggered": 0,
                },
                "loss_close_regrets": _loss_close_replay_rows(),
            },
        )

        self.assertEqual(findings, [])

    def test_post_repair_clean_values_without_contract_remain_fail_closed(self) -> None:
        findings = _profit_capture_miss_findings(
            run_id=_NOW.isoformat(),
            target_open=True,
            timing_payload={
                "generated_at_utc": _NOW.isoformat(),
                "precision": {},
                "summary": {
                    "loss_closes_audited": 28,
                    "loss_closes_profit_capture_missed": 8,
                    "loss_closes_repair_replay_triggered": 7,
                    "tp_progress_repair_live_evidence_status": "POST_REPAIR_REPLAY_CLEAN",
                    "pre_repair_historical_loss_closes_audited": 23,
                    "pre_repair_historical_loss_closes_profit_capture_missed": 8,
                    "pre_repair_historical_loss_closes_repair_replay_triggered": 7,
                    "post_repair_live_evidence_loss_closes_audited": 5,
                    "post_repair_live_evidence_loss_closes_profit_capture_missed": 0,
                    "post_repair_live_evidence_loss_closes_repair_replay_triggered": 0,
                },
                "loss_close_regrets": [],
            },
        )

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["priority"], "P1")
        self.assertIn("lacks current production-gate replay proof", findings[0]["message"])

    def test_post_repair_clean_values_with_status_mismatch_remain_fail_closed(self) -> None:
        for status in (
            "WAITING_FOR_POST_REPAIR_SAMPLE",
            "POST_REPAIR_REPLAY_FAILURES_PRESENT",
        ):
            with self.subTest(status=status):
                findings = _profit_capture_miss_findings(
                    run_id=_REPLAY_NOW.isoformat(),
                    target_open=True,
                    timing_payload={
                        "generated_at_utc": _REPLAY_NOW.isoformat(),
                        "status": "OK",
                        "fetch_errors": [],
                        "precision": {
                            TP_PROGRESS_REPAIR_REPLAY_FIELD: TP_PROGRESS_REPAIR_REPLAY_CONTRACT,
                            "price_basis": "OANDA_M1_BID_ASK_CANDLES",
                            "granularity": "M1",
                        },
                        "window": _replay_window(_REPLAY_NOW),
                        "summary": {
                            "loss_closes_audited": 28,
                            "loss_closes_profit_capture_missed": 8,
                            "loss_closes_repair_replay_triggered": 7,
                            "tp_progress_repair_live_evidence_boundary_utc": (
                                TP_PROGRESS_REPAIR_LIVE_EVIDENCE_BOUNDARY_UTC
                            ),
                            "tp_progress_repair_live_evidence_status": status,
                            "pre_repair_historical_loss_closes_audited": 23,
                            "pre_repair_historical_loss_closes_profit_capture_missed": 8,
                            "pre_repair_historical_loss_closes_repair_replay_triggered": 7,
                            "post_repair_live_evidence_loss_closes_audited": 5,
                            "post_repair_live_evidence_loss_closes_profit_capture_missed": 0,
                            "post_repair_live_evidence_loss_closes_repair_replay_triggered": 0,
                        },
                        "loss_close_regrets": _loss_close_replay_rows(),
                    },
                )

                self.assertEqual(len(findings), 1)
                self.assertEqual(findings[0]["priority"], "P1")
                self.assertEqual(
                    findings[0]["code"],
                    "LOSS_CLOSE_PROFIT_CAPTURE_MISSED",
                )

    def test_post_repair_clean_status_with_incomplete_or_inconsistent_split_is_fail_closed(
        self,
    ) -> None:
        base_summary = {
            "loss_closes_audited": 28,
            "loss_closes_profit_capture_missed": 8,
            "loss_closes_repair_replay_triggered": 7,
            "tp_progress_repair_live_evidence_boundary_utc": (
                TP_PROGRESS_REPAIR_LIVE_EVIDENCE_BOUNDARY_UTC
            ),
            "tp_progress_repair_live_evidence_status": "POST_REPAIR_REPLAY_CLEAN",
            "pre_repair_historical_loss_closes_audited": 23,
            "pre_repair_historical_loss_closes_profit_capture_missed": 8,
            "pre_repair_historical_loss_closes_repair_replay_triggered": 7,
            "post_repair_live_evidence_loss_closes_audited": 5,
            "post_repair_live_evidence_loss_closes_profit_capture_missed": 0,
            "post_repair_live_evidence_loss_closes_repair_replay_triggered": 0,
        }
        variants = []
        incomplete = dict(base_summary)
        incomplete.pop("post_repair_live_evidence_loss_closes_profit_capture_missed")
        variants.append(incomplete)
        all_counts_missing = dict(base_summary)
        for field in (
            "loss_closes_audited",
            "loss_closes_profit_capture_missed",
            "loss_closes_repair_replay_triggered",
            "pre_repair_historical_loss_closes_audited",
            "pre_repair_historical_loss_closes_profit_capture_missed",
            "pre_repair_historical_loss_closes_repair_replay_triggered",
            "post_repair_live_evidence_loss_closes_audited",
            "post_repair_live_evidence_loss_closes_profit_capture_missed",
            "post_repair_live_evidence_loss_closes_repair_replay_triggered",
        ):
            all_counts_missing.pop(field)
        variants.append(all_counts_missing)
        inconsistent = dict(base_summary)
        inconsistent["pre_repair_historical_loss_closes_audited"] = 22
        variants.append(inconsistent)
        non_integer = dict(base_summary)
        non_integer["post_repair_live_evidence_loss_closes_profit_capture_missed"] = 0.5
        variants.append(non_integer)
        corrupt_total = dict(base_summary)
        corrupt_total["loss_closes_profit_capture_missed"] = "corrupt"
        variants.append(corrupt_total)
        impossible = dict(base_summary)
        impossible["loss_closes_audited"] = 5
        impossible["pre_repair_historical_loss_closes_audited"] = 0
        variants.append(impossible)
        missing_declared_status = dict(base_summary)
        missing_declared_status.pop("tp_progress_repair_live_evidence_status")
        variants.append(missing_declared_status)
        missing_boundary = dict(base_summary)
        missing_boundary.pop("tp_progress_repair_live_evidence_boundary_utc")
        variants.append(missing_boundary)
        wrong_boundary = dict(base_summary)
        wrong_boundary["tp_progress_repair_live_evidence_boundary_utc"] = (
            "2026-06-22T17:54:25Z"
        )
        variants.append(wrong_boundary)

        for summary in variants:
            with self.subTest(summary=summary):
                findings = _profit_capture_miss_findings(
                    run_id=_REPLAY_NOW.isoformat(),
                    target_open=True,
                    timing_payload={
                        "generated_at_utc": _REPLAY_NOW.isoformat(),
                        "status": "OK",
                        "fetch_errors": [],
                        "precision": {
                            TP_PROGRESS_REPAIR_REPLAY_FIELD: TP_PROGRESS_REPAIR_REPLAY_CONTRACT,
                            "price_basis": "OANDA_M1_BID_ASK_CANDLES",
                            "granularity": "M1",
                        },
                        "window": _replay_window(_REPLAY_NOW),
                        "summary": summary,
                        "loss_close_regrets": _loss_close_replay_rows(),
                    },
                )

                self.assertEqual(len(findings), 1)
                self.assertEqual(findings[0]["code"], "LOSS_CLOSE_PROFIT_CAPTURE_MISSED")

    def test_post_repair_clean_counts_with_partial_source_remain_fail_closed(self) -> None:
        base_payload = {
            "generated_at_utc": _REPLAY_NOW.isoformat(),
            "status": "OK",
            "fetch_errors": [],
            "precision": {
                TP_PROGRESS_REPAIR_REPLAY_FIELD: TP_PROGRESS_REPAIR_REPLAY_CONTRACT,
                "price_basis": "OANDA_M1_BID_ASK_CANDLES",
                "granularity": "M1",
            },
            "window": _replay_window(_REPLAY_NOW),
            "summary": {
                "loss_closes_audited": 28,
                "loss_closes_profit_capture_missed": 8,
                "loss_closes_repair_replay_triggered": 7,
                "tp_progress_repair_live_evidence_boundary_utc": (
                    TP_PROGRESS_REPAIR_LIVE_EVIDENCE_BOUNDARY_UTC
                ),
                "tp_progress_repair_live_evidence_status": "POST_REPAIR_REPLAY_CLEAN",
                "pre_repair_historical_loss_closes_audited": 23,
                "pre_repair_historical_loss_closes_profit_capture_missed": 8,
                "pre_repair_historical_loss_closes_repair_replay_triggered": 7,
                "post_repair_live_evidence_loss_closes_audited": 5,
                "post_repair_live_evidence_loss_closes_profit_capture_missed": 0,
                "post_repair_live_evidence_loss_closes_repair_replay_triggered": 0,
            },
            "loss_close_regrets": _loss_close_replay_rows(),
        }
        variants = []
        partial = json.loads(json.dumps(base_payload))
        partial["status"] = "PARTIAL_DATA"
        partial["fetch_errors"] = [{"pair": "EUR_USD", "error": "missing candle"}]
        variants.append(partial)
        missing_status = json.loads(json.dumps(base_payload))
        missing_status.pop("status")
        variants.append(missing_status)
        short_window = json.loads(json.dumps(base_payload))
        short_window["window"]["lookback_hours"] = 168.0
        variants.append(short_window)
        wrong_granularity = json.loads(json.dumps(base_payload))
        wrong_granularity["precision"]["granularity"] = "M5"
        variants.append(wrong_granularity)
        unknown_cap = json.loads(json.dumps(base_payload))
        unknown_cap["window"].pop("max_events")
        variants.append(unknown_cap)
        cap_reached = json.loads(json.dumps(base_payload))
        cap_reached["summary"]["loss_closes_audited"] = 80
        cap_reached["summary"]["pre_repair_historical_loss_closes_audited"] = 75
        variants.append(cap_reached)
        stale_clock = json.loads(json.dumps(base_payload))
        stale_at = (_REPLAY_NOW - timedelta(hours=2)).isoformat()
        stale_clock["generated_at_utc"] = stale_at
        stale_clock["window"]["to_utc"] = stale_at
        variants.append(stale_clock)
        mismatched_clock = json.loads(json.dumps(base_payload))
        mismatched_clock["window"]["to_utc"] = (
            _REPLAY_NOW - timedelta(seconds=1)
        ).isoformat()
        variants.append(mismatched_clock)
        naive_clock = json.loads(json.dumps(base_payload))
        naive_clock["generated_at_utc"] = _REPLAY_NOW.replace(
            tzinfo=None
        ).isoformat()
        naive_clock["window"]["to_utc"] = _REPLAY_NOW.replace(
            tzinfo=None
        ).isoformat()
        variants.append(naive_clock)
        float_cap = json.loads(json.dumps(base_payload))
        float_cap["window"]["max_events"] = 80.0
        variants.append(float_cap)
        zero_candle_row = json.loads(json.dumps(base_payload))
        zero_candle_row["loss_close_regrets"][0]["candles_available"] = 0
        variants.append(zero_candle_row)
        missing_loss_row = json.loads(json.dumps(base_payload))
        missing_loss_row["loss_close_regrets"].pop()
        variants.append(missing_loss_row)
        duplicate_trade_row = json.loads(json.dumps(base_payload))
        duplicate_trade_row["loss_close_regrets"][1] = json.loads(
            json.dumps(duplicate_trade_row["loss_close_regrets"][0])
        )
        variants.append(duplicate_trade_row)
        naive_row_clock = json.loads(json.dumps(base_payload))
        naive_row_clock["loss_close_regrets"][0]["close_at_utc"] = (
            "2026-06-22T17:00:00"
        )
        variants.append(naive_row_clock)
        row_count_mismatch = json.loads(json.dumps(base_payload))
        row_count_mismatch["loss_close_regrets"][0][
            "profit_capture_missed_before_loss_close"
        ] = False
        row_count_mismatch["loss_close_regrets"][0][
            "repair_replay_triggered_before_loss_close"
        ] = False
        variants.append(row_count_mismatch)
        row_split_mismatch = json.loads(json.dumps(base_payload))
        row_split_mismatch["loss_close_regrets"][0]["mfe_at_utc"] = (
            "2026-06-22T18:00:00+00:00"
        )
        row_split_mismatch["loss_close_regrets"][0][
            "repair_replay_trigger_at_utc"
        ] = "2026-06-22T18:00:00+00:00"
        variants.append(row_split_mismatch)
        future_row = json.loads(json.dumps(base_payload))
        future_row["loss_close_regrets"][0]["close_at_utc"] = (
            _REPLAY_NOW + timedelta(minutes=1)
        ).isoformat()
        variants.append(future_row)
        fill_order_mismatch = json.loads(json.dumps(base_payload))
        fill_order_mismatch["loss_close_regrets"][0]["fill_at_utc"] = (
            "2026-06-22T17:30:00+00:00"
        )
        variants.append(fill_order_mismatch)
        hidden_trigger = json.loads(json.dumps(base_payload))
        hidden_trigger["loss_close_regrets"][7][
            "repair_replay_trigger_at_utc"
        ] = "2026-06-22T17:00:00+00:00"
        variants.append(hidden_trigger)
        corrupt_sort_value = json.loads(json.dumps(partial))
        corrupt_sort_value["loss_close_regrets"][0][
            "profit_capture_counterfactual_net_improvement_jpy"
        ] = "corrupt"
        variants.append(corrupt_sort_value)
        non_list_rows = json.loads(json.dumps(base_payload))
        non_list_rows["loss_close_regrets"] = 42
        variants.append(non_list_rows)
        overflow_lookback = json.loads(json.dumps(base_payload))
        overflow_lookback["window"]["lookback_hours"] = 1e308
        variants.append(overflow_lookback)
        for field, invalid_value in (
            ("realized_pl_jpy", None),
            ("realized_pl_jpy", 100.0),
            ("realized_pl_jpy", "corrupt"),
            ("pair", ""),
            ("side", "BUY"),
            ("side", []),
            ("side", {}),
            ("units", 0),
            ("entry", "corrupt"),
            ("close_price", 0.0),
            ("exit_reason", ""),
        ):
            invalid_core_row = json.loads(json.dumps(base_payload))
            invalid_core_row["loss_close_regrets"][0][field] = invalid_value
            variants.append(invalid_core_row)

        for payload in variants:
            with self.subTest(payload=payload):
                findings = _profit_capture_miss_findings(
                    run_id=_REPLAY_NOW.isoformat(),
                    target_open=True,
                    timing_payload=payload,
                )

                self.assertEqual(len(findings), 1)
                self.assertEqual(findings[0]["code"], "LOSS_CLOSE_PROFIT_CAPTURE_MISSED")

    def test_zero_miss_clearance_requires_complete_current_replay_source(self) -> None:
        base_payload = {
            "generated_at_utc": _REPLAY_NOW.isoformat(),
            "status": "OK",
            "fetch_errors": [],
            "precision": {
                TP_PROGRESS_REPAIR_REPLAY_FIELD: TP_PROGRESS_REPAIR_REPLAY_CONTRACT,
                "price_basis": "OANDA_M1_BID_ASK_CANDLES",
                "granularity": "M1",
            },
            "window": _replay_window(_REPLAY_NOW),
            "summary": {
                "loss_closes_audited": 5,
                "loss_closes_profit_capture_missed": 0,
                "loss_closes_repair_replay_triggered": 0,
                "tp_progress_repair_live_evidence_boundary_utc": (
                    TP_PROGRESS_REPAIR_LIVE_EVIDENCE_BOUNDARY_UTC
                ),
                "tp_progress_repair_live_evidence_status": (
                    "POST_REPAIR_REPLAY_CLEAN"
                ),
                "pre_repair_historical_loss_closes_audited": 0,
                "pre_repair_historical_loss_closes_profit_capture_missed": 0,
                "pre_repair_historical_loss_closes_repair_replay_triggered": 0,
                "post_repair_live_evidence_loss_closes_audited": 5,
                "post_repair_live_evidence_loss_closes_profit_capture_missed": 0,
                "post_repair_live_evidence_loss_closes_repair_replay_triggered": 0,
            },
            "loss_close_regrets": _loss_close_replay_rows(
                audited=5,
                pre_repair_samples=0,
                raw_misses=0,
                replay_triggers=0,
            ),
        }
        self.assertEqual(
            _profit_capture_miss_findings(
                run_id=_REPLAY_NOW.isoformat(),
                target_open=True,
                timing_payload=base_payload,
            ),
            [],
        )

        variants: list[dict[str, object]] = [{}, {"summary": {}, "loss_close_regrets": []}]
        partial = json.loads(json.dumps(base_payload))
        partial["status"] = "PARTIAL_DATA"
        partial["fetch_errors"] = [{"pair": "EUR_USD", "error": "missing"}]
        variants.append(partial)
        stale = json.loads(json.dumps(base_payload))
        stale_at = _REPLAY_NOW - timedelta(hours=24)
        stale["generated_at_utc"] = stale_at.isoformat()
        stale["window"] = _replay_window(stale_at)
        variants.append(stale)
        short = json.loads(json.dumps(base_payload))
        short["window"]["lookback_hours"] = 1.0
        variants.append(short)
        missing_rows = json.loads(json.dumps(base_payload))
        missing_rows["loss_close_regrets"] = []
        variants.append(missing_rows)
        missing_status = json.loads(json.dumps(base_payload))
        missing_status.pop("status")
        variants.append(missing_status)
        cap_reached = json.loads(json.dumps(base_payload))
        cap_reached["summary"]["loss_closes_audited"] = 80
        cap_reached["summary"]["post_repair_live_evidence_loss_closes_audited"] = 80
        cap_reached["loss_close_regrets"] = _loss_close_replay_rows(
            audited=80,
            pre_repair_samples=0,
            raw_misses=0,
            replay_triggers=0,
        )
        variants.append(cap_reached)

        for payload in variants:
            with self.subTest(payload=payload):
                findings = _profit_capture_miss_findings(
                    run_id=_REPLAY_NOW.isoformat(),
                    target_open=True,
                    timing_payload=payload,
                )

                self.assertEqual(len(findings), 1)
                self.assertEqual(findings[0]["priority"], "P1")
                self.assertEqual(
                    findings[0]["code"],
                    "LOSS_CLOSE_PROFIT_CAPTURE_MISSED",
                )

    def test_contradictory_zero_total_does_not_hide_post_repair_replay_failure(self) -> None:
        findings = _profit_capture_miss_findings(
            run_id=_REPLAY_NOW.isoformat(),
            target_open=True,
            timing_payload={
                "generated_at_utc": _REPLAY_NOW.isoformat(),
                "status": "OK",
                "fetch_errors": [],
                "precision": {
                    TP_PROGRESS_REPAIR_REPLAY_FIELD: TP_PROGRESS_REPAIR_REPLAY_CONTRACT,
                    "price_basis": "OANDA_M1_BID_ASK_CANDLES",
                    "granularity": "M1",
                },
                "window": _replay_window(_REPLAY_NOW),
                "summary": {
                    "loss_closes_audited": 1,
                    "loss_closes_profit_capture_missed": 0,
                    "loss_closes_repair_replay_triggered": 0,
                    "tp_progress_repair_live_evidence_boundary_utc": (
                        TP_PROGRESS_REPAIR_LIVE_EVIDENCE_BOUNDARY_UTC
                    ),
                    "tp_progress_repair_live_evidence_status": (
                        "POST_REPAIR_REPLAY_FAILURES_PRESENT"
                    ),
                    "pre_repair_historical_loss_closes_audited": 0,
                    "pre_repair_historical_loss_closes_profit_capture_missed": 0,
                    "pre_repair_historical_loss_closes_repair_replay_triggered": 0,
                    "post_repair_live_evidence_loss_closes_audited": 1,
                    "post_repair_live_evidence_loss_closes_profit_capture_missed": 1,
                    "post_repair_live_evidence_loss_closes_repair_replay_triggered": 1,
                },
                "loss_close_regrets": _loss_close_replay_rows(
                    audited=1,
                    pre_repair_samples=0,
                    raw_misses=1,
                    replay_triggers=1,
                ),
            },
        )

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["priority"], "P0")
        self.assertEqual(findings[0]["code"], "LOSS_CLOSE_PROFIT_CAPTURE_MISSED")

    def test_raw_profit_capture_miss_without_repair_replay_is_p1_diagnostic(self) -> None:
        findings = _profit_capture_miss_findings(
            run_id=_NOW.isoformat(),
            target_open=True,
            timing_payload={
                "generated_at_utc": _NOW.isoformat(),
                "precision": {
                    TP_PROGRESS_REPAIR_REPLAY_FIELD: TP_PROGRESS_REPAIR_REPLAY_CONTRACT,
                },
                "summary": {
                    "loss_closes_audited": 1,
                    "loss_closes_profit_capture_missed": 1,
                    "loss_closes_profit_capture_missed_rate": 1.0,
                    "loss_closes_repair_replay_triggered": 0,
                    "loss_closes_repair_replay_triggered_rate": 0.0,
                    "loss_close_counterfactual_profit_capture_delta_jpy": 607.5,
                },
                "loss_close_regrets": [
                    {
                        "trade_id": "raw-only-noise",
                        "lane_id": "range_trader:AUD_NZD:SHORT:RANGE_ROTATION",
                        "pair": "AUD_NZD",
                        "side": "SHORT",
                        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
                        "profit_capture_missed_before_loss_close": True,
                        "profit_capture_counterfactual_net_improvement_jpy": 607.5,
                        "repair_replay_triggered_before_loss_close": False,
                    }
                ],
            },
        )

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["priority"], "P1")
        self.assertEqual(findings[0]["code"], "LOSS_CLOSE_PROFIT_CAPTURE_MISSED")
        self.assertEqual(findings[0]["evidence"]["loss_closes_repair_replay_triggered"], 0)
        self.assertEqual(findings[0]["evidence"]["top_repair_replay_triggers"], [])

    def test_inactive_position_guardian_is_p0_for_target_open_profit_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(120.0, 90.0, 60.0),
            )
            with mock.patch.dict(
                os.environ,
                {
                    "QR_REQUIRE_POSITION_GUARDIAN_ACTIVE": "1",
                    "QR_POSITION_GUARDIAN_ACTIVE": "0",
                },
                clear=False,
            ):
                summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertIn("POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE", codes)
        finding = codes["POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE"]
        self.assertEqual(finding["priority"], "P0")
        self.assertEqual(finding["layer"], "execution_quality")
        self.assertEqual(finding["evidence"]["live_ready_lanes"], 1)
        self.assertFalse(finding["evidence"]["guardian"]["active"])
        self.assertTrue(finding["evidence"]["guardian"]["required"])
        self.assertFalse(payload["runtime"]["position_guardian"]["active"])

    def test_position_guardian_operator_override_suppresses_inactive_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(120.0, 90.0, 60.0),
            )
            with mock.patch.dict(
                os.environ,
                {
                    "QR_REQUIRE_POSITION_GUARDIAN_ACTIVE": "0",
                    "QR_POSITION_GUARDIAN_ACTIVE": "0",
                },
                clear=False,
            ):
                _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertNotIn("POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE", codes)
        self.assertFalse(payload["runtime"]["position_guardian"]["required"])

    def test_loaded_position_guardian_with_stale_heartbeat_is_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(120.0, 90.0, 60.0),
            )
            plist = root / "com.quantrabbit.position-guardian.plist"
            plist.write_text("<plist/>")
            heartbeat = root / "position_guardian_execution.json"
            heartbeat.write_text(
                json.dumps(
                    {
                        "generated_at_utc": (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(),
                        "status": "NO_ACTION",
                    }
                )
            )
            with mock.patch.dict(
                os.environ,
                {
                    "QR_REQUIRE_POSITION_GUARDIAN_ACTIVE": "1",
                    "QR_POSITION_GUARDIAN_PLIST": str(plist),
                    "QR_POSITION_GUARDIAN_EXECUTION": str(heartbeat),
                    "QR_POSITION_GUARDIAN_HEARTBEAT": str(root / "missing_position_guardian.json"),
                    "QR_POSITION_GUARDIAN_INTERVAL": "30",
                    "QR_POSITION_GUARDIAN_HEARTBEAT_MAX_AGE_SECONDS": "120",
                },
                clear=False,
            ), mock.patch(
                "quant_rabbit.self_improvement_audit.subprocess.run",
                return_value=mock.Mock(returncode=0),
            ):
                summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        finding = codes["POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE"]
        guardian = finding["evidence"]["guardian"]
        self.assertTrue(guardian["launchd_loaded"])
        self.assertFalse(guardian["heartbeat_fresh"])
        self.assertEqual(guardian["active_source"], "stale_heartbeat")
        self.assertFalse(guardian["active"])

    def test_loaded_position_guardian_stale_heartbeat_during_live_lock_suppresses_inactive_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(120.0, 90.0, 60.0),
            )
            plist = root / "com.quantrabbit.position-guardian.plist"
            plist.write_text("<plist/>")
            heartbeat = root / "position_guardian_execution.json"
            heartbeat.write_text(
                json.dumps(
                    {
                        "generated_at_utc": (_NOW - timedelta(minutes=10)).isoformat(),
                        "status": "NO_ACTION",
                    }
                )
            )
            lock_dir = root / ".quant_rabbit_live.lock"
            lock_dir.mkdir()
            (lock_dir / "pid").write_text(str(os.getpid()))
            (lock_dir / "command").write_text("cycle-refresh")
            (lock_dir / "started_at_utc").write_text((_NOW - timedelta(minutes=5)).isoformat())
            with mock.patch.dict(
                os.environ,
                {
                    "QR_REQUIRE_POSITION_GUARDIAN_ACTIVE": "1",
                    "QR_POSITION_GUARDIAN_PLIST": str(plist),
                    "QR_POSITION_GUARDIAN_EXECUTION": str(heartbeat),
                    "QR_POSITION_GUARDIAN_HEARTBEAT": str(root / "missing_position_guardian.json"),
                    "QR_POSITION_GUARDIAN_INTERVAL": "30",
                    "QR_POSITION_GUARDIAN_HEARTBEAT_MAX_AGE_SECONDS": "120",
                    "QR_AUTOTRADE_LOCK_HELD": "1",
                    "QR_AUTOTRADE_LOCK_DIR": str(lock_dir),
                },
                clear=False,
            ), mock.patch(
                "quant_rabbit.self_improvement_audit.subprocess.run",
                return_value=mock.Mock(returncode=0),
            ):
                os.environ.pop("QR_POSITION_GUARDIAN_ACTIVE", None)
                _run(files, now=_NOW)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertNotIn("POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE", codes)
        guardian = payload["runtime"]["position_guardian"]
        self.assertTrue(guardian["launchd_loaded"])
        self.assertFalse(guardian["heartbeat_fresh"])
        self.assertFalse(guardian["active"])
        self.assertEqual(guardian["active_source"], "live_runtime_lock_busy")
        self.assertTrue(guardian["live_runtime_lock_active"])
        self.assertTrue(guardian["live_runtime_lock_held_by_current_process"])
        self.assertEqual(guardian["live_runtime_lock_command"], "cycle-refresh")

    def test_env_active_guardian_stale_heartbeat_during_live_lock_suppresses_inactive_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(120.0, 90.0, 60.0),
            )
            heartbeat = root / "position_guardian_execution.json"
            heartbeat.write_text(
                json.dumps(
                    {
                        "generated_at_utc": (_NOW - timedelta(minutes=10)).isoformat(),
                        "status": "NO_ACTION",
                    }
                )
            )
            lock_dir = root / ".quant_rabbit_live.lock"
            lock_dir.mkdir()
            (lock_dir / "pid").write_text(str(os.getpid()))
            (lock_dir / "command").write_text("run-autotrade-live")
            (lock_dir / "started_at_utc").write_text((_NOW - timedelta(minutes=5)).isoformat())
            with mock.patch.dict(
                os.environ,
                {
                    "QR_REQUIRE_POSITION_GUARDIAN_ACTIVE": "1",
                    "QR_POSITION_GUARDIAN_ACTIVE": "1",
                    "QR_POSITION_GUARDIAN_EXECUTION": str(heartbeat),
                    "QR_POSITION_GUARDIAN_HEARTBEAT": str(root / "missing_position_guardian.json"),
                    "QR_POSITION_GUARDIAN_INTERVAL": "30",
                    "QR_POSITION_GUARDIAN_HEARTBEAT_MAX_AGE_SECONDS": "120",
                    "QR_AUTOTRADE_LOCK_HELD": "1",
                    "QR_AUTOTRADE_LOCK_DIR": str(lock_dir),
                },
                clear=False,
            ):
                _run(files, now=_NOW)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertNotIn("POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE", codes)
        guardian = payload["runtime"]["position_guardian"]
        self.assertFalse(guardian["heartbeat_fresh"])
        self.assertFalse(guardian["active"])
        self.assertEqual(guardian["env_active"], "1")
        self.assertEqual(guardian["active_source"], "live_runtime_lock_busy")
        self.assertTrue(guardian["live_runtime_lock_active"])
        self.assertTrue(guardian["live_runtime_lock_held_by_current_process"])
        self.assertEqual(guardian["live_runtime_lock_command"], "run-autotrade-live")

    def test_loaded_position_guardian_with_fresh_heartbeat_suppresses_inactive_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(120.0, 90.0, 60.0),
            )
            plist = root / "com.quantrabbit.position-guardian.plist"
            plist.write_text("<plist/>")
            heartbeat = root / "position_guardian_execution.json"
            heartbeat.write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "NO_ACTION",
                    }
                )
            )
            with mock.patch.dict(
                os.environ,
                {
                    "QR_REQUIRE_POSITION_GUARDIAN_ACTIVE": "1",
                    "QR_POSITION_GUARDIAN_PLIST": str(plist),
                    "QR_POSITION_GUARDIAN_EXECUTION": str(heartbeat),
                    "QR_POSITION_GUARDIAN_HEARTBEAT": str(root / "missing_position_guardian.json"),
                    "QR_POSITION_GUARDIAN_INTERVAL": "30",
                    "QR_POSITION_GUARDIAN_HEARTBEAT_MAX_AGE_SECONDS": "120",
                },
                clear=False,
            ), mock.patch(
                "quant_rabbit.self_improvement_audit.subprocess.run",
                return_value=mock.Mock(returncode=0),
            ):
                _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertNotIn("POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE", codes)
        guardian = payload["runtime"]["position_guardian"]
        self.assertTrue(guardian["launchd_loaded"])
        self.assertTrue(guardian["heartbeat_fresh"])
        self.assertTrue(guardian["active"])

    def test_position_guardian_default_source_paths_read_live_heartbeat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_data = root / "source" / "data"
            live_data = root / "live" / "data"
            live_data.mkdir(parents=True)
            live_heartbeat = live_data / "position_guardian.json"
            live_heartbeat.write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "NO_POSITION",
                    }
                )
            )
            source_execution = source_data / "position_guardian_execution.json"
            source_heartbeat = source_data / "position_guardian.json"
            plist = root / "com.quantrabbit.position-guardian.plist"
            plist.write_text("<plist/>")

            with (
                mock.patch(
                    "quant_rabbit.self_improvement_audit.DEFAULT_POSITION_GUARDIAN_EXECUTION",
                    source_execution,
                ),
                mock.patch(
                    "quant_rabbit.self_improvement_audit.DEFAULT_POSITION_GUARDIAN_HEARTBEAT",
                    source_heartbeat,
                ),
                mock.patch.dict(
                    os.environ,
                    {
                        "QR_SYNC_LIVE_ROOT": str(root / "live"),
                        "QR_POSITION_GUARDIAN_PLIST": str(plist),
                        "QR_REQUIRE_POSITION_GUARDIAN_ACTIVE": "1",
                        "QR_POSITION_GUARDIAN_INTERVAL": "30",
                        "QR_POSITION_GUARDIAN_HEARTBEAT_MAX_AGE_SECONDS": "120",
                    },
                    clear=False,
                ),
                mock.patch(
                    "quant_rabbit.self_improvement_audit.subprocess.run",
                    return_value=mock.Mock(returncode=0),
                ),
            ):
                os.environ.pop("QR_POSITION_GUARDIAN_EXECUTION", None)
                os.environ.pop("QR_POSITION_GUARDIAN_HEARTBEAT", None)
                guardian = _position_guardian_runtime_status()

        self.assertTrue(guardian["active"])
        self.assertEqual(guardian["active_source"], "launchd+heartbeat")
        self.assertTrue(guardian["heartbeat_fresh"])
        self.assertEqual(guardian["heartbeat_path"], str(live_heartbeat))
        self.assertIn(str(live_heartbeat), guardian["heartbeat_candidates"])

    def test_position_guardian_custom_paths_do_not_fall_back_to_live_heartbeat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live_data = root / "live" / "data"
            live_data.mkdir(parents=True)
            (live_data / "position_guardian.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "NO_POSITION",
                    }
                )
            )
            custom_execution = root / "custom" / "position_guardian_execution.json"
            custom_heartbeat = root / "custom" / "position_guardian.json"
            plist = root / "com.quantrabbit.position-guardian.plist"
            plist.write_text("<plist/>")

            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "QR_SYNC_LIVE_ROOT": str(root / "live"),
                        "QR_POSITION_GUARDIAN_PLIST": str(plist),
                        "QR_REQUIRE_POSITION_GUARDIAN_ACTIVE": "1",
                        "QR_POSITION_GUARDIAN_EXECUTION": str(custom_execution),
                        "QR_POSITION_GUARDIAN_HEARTBEAT": str(custom_heartbeat),
                        "QR_POSITION_GUARDIAN_INTERVAL": "30",
                        "QR_POSITION_GUARDIAN_HEARTBEAT_MAX_AGE_SECONDS": "120",
                    },
                    clear=False,
                ),
                mock.patch(
                    "quant_rabbit.self_improvement_audit.subprocess.run",
                    return_value=mock.Mock(returncode=0),
                ),
            ):
                guardian = _position_guardian_runtime_status()

        self.assertFalse(guardian["active"])
        self.assertEqual(guardian["active_source"], "stale_heartbeat")
        self.assertIsNone(guardian["heartbeat_path"])
        self.assertNotIn(str(live_data / "position_guardian.json"), guardian["heartbeat_candidates"])

    def test_projection_expiry_uses_live_telemetry_grace(self) -> None:
        grace = timedelta(seconds=PROJECTION_PENDING_EXPIRY_GRACE_SECONDS)
        row = {
            "timestamp_emitted_utc": (
                _NOW - timedelta(minutes=30) - grace + timedelta(seconds=1)
            ).isoformat(),
            "resolution_window_min": 30.0,
        }
        self.assertFalse(_projection_expired(row, now=_NOW))

        row["timestamp_emitted_utc"] = (
            _NOW - timedelta(minutes=30) - grace - timedelta(seconds=1)
        ).isoformat()
        self.assertTrue(_projection_expired(row, now=_NOW))

    def test_no_touch_directional_miss_is_not_invalidation_first(self) -> None:
        no_touch = {
            "direction": "UP",
            "resolution_status": "MISS",
            "resolution_evidence": (
                "target 1.10200 not reached; invalidation 1.09900 also untouched in forecast window"
            ),
        }
        invalidation_first = {
            "direction": "UP",
            "resolution_status": "MISS",
            "resolution_evidence": "2026-06-16T00:10:00Z invalidation 1.09900 touched before target 1.10200",
        }

        self.assertFalse(_directional_forecast_invalidation_first_like(no_touch))
        self.assertTrue(_directional_forecast_target_timeout_like(no_touch))
        self.assertTrue(_directional_forecast_invalidation_first_like(invalidation_first))
        self.assertFalse(_directional_forecast_target_timeout_like(invalidation_first))

    def test_incomplete_truth_timeout_is_not_classified_as_scored_no_touch(self) -> None:
        row = {
            "direction": "UP",
            "resolution_status": "TIMEOUT",
            "predicted_target_price": 1.1020,
            "predicted_invalidation_price": 1.0990,
            "resolution_evidence": (
                "target and invalidation both untouched; incomplete closed candle truth "
                "for full projection window"
            ),
        }

        self.assertFalse(_directional_forecast_no_touch_timeout_like(row))
        self.assertFalse(
            _directional_forecast_no_touch_timeout_like(
                {
                    **row,
                    "direction": "RANGE",
                    "predicted_range_low_price": 1.0990,
                    "predicted_range_high_price": 1.1010,
                }
            )
        )

    def test_top_intent_blockers_ignore_dry_run_strategy_warnings(self) -> None:
        blockers = _top_intent_blockers(
            {
                "results": [
                    {
                        "lane_id": "trend_trader:EUR_AUD:LONG:TREND_CONTINUATION",
                        "status": "LIVE_READY",
                        "risk_issues": [],
                        "strategy_issues": [
                            {
                                "code": "STRATEGY_PROFILE_MISSING",
                                "message": "forecast-seeded advisory profile warning",
                                "severity": "WARN",
                            }
                        ],
                        "live_strategy_issues": [
                            {
                                "code": "STRATEGY_PROFILE_MISSING",
                                "message": "forecast-seeded advisory profile warning",
                                "severity": "WARN",
                            }
                        ],
                        "live_blockers": [],
                    },
                    {
                        "lane_id": "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
                        "status": "DRY_RUN_BLOCKED",
                        "risk_issues": [
                            {
                                "code": "REWARD_RISK_TOO_LOW",
                                "message": "reward/risk below floor",
                                "severity": "BLOCK",
                            }
                        ],
                        "strategy_issues": [],
                        "live_strategy_issues": [],
                        "live_blockers": ["EUR_USD SHORT forecast confidence below live floor"],
                    },
                ]
            }
        )

        messages = {item["message"] for item in blockers}
        self.assertNotIn("STRATEGY_PROFILE_MISSING", messages)
        self.assertIn("REWARD_RISK_TOO_LOW", messages)
        self.assertIn("EUR_USD SHORT forecast confidence below live floor", messages)

    def test_live_readiness_blockers_include_warn_live_gates_for_dry_run_passed(self) -> None:
        blockers = _top_intent_live_readiness_blockers(
            {
                "results": [
                    {
                        "lane_id": "failure_trader:AUD_CAD:SHORT:BREAKOUT_FAILURE:LIMIT",
                        "status": "DRY_RUN_PASSED",
                        "risk_issues": [
                            {
                                "code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
                                "message": "forecast confidence below live floor",
                                "severity": "WARN",
                            },
                            {
                                "code": "SPREAD_ADVISORY",
                                "message": "spread is elevated but still below block floor",
                                "severity": "WARN",
                            },
                        ],
                        "strategy_issues": [
                            {
                                "code": "STRATEGY_PROFILE_MISSING",
                                "message": "dry-run advisory profile warning",
                                "severity": "WARN",
                            }
                        ],
                        "live_strategy_issues": [
                            {
                                "code": "STRATEGY_NOT_ELIGIBLE",
                                "message": "strategy profile is not live eligible",
                                "severity": "WARN",
                                "strategy_profile_evidence": {
                                    "profile_status": "BLOCK_UNTIL_NEW_EVIDENCE",
                                    "required_fix": "both live and pretrade feedback are negative",
                                    "live_net_jpy": -1501.57,
                                    "pretrade_net_jpy": -837.76,
                                },
                            }
                        ],
                        "live_blockers": [],
                    },
                    {
                        "lane_id": "failure_trader:GBP_USD:SHORT:BREAKOUT_FAILURE:LIMIT",
                        "status": "DRY_RUN_PASSED",
                        "risk_issues": [],
                        "strategy_issues": [],
                        "live_strategy_issues": [],
                        "live_blockers": ["legacy live blocker fallback"],
                    },
                    {
                        "lane_id": "range_trader:EUR_CHF:SHORT:RANGE_ROTATION",
                        "status": "LIVE_READY",
                        "risk_issues": [
                            {
                                "code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
                                "message": "live-ready advisory must not be counted",
                                "severity": "WARN",
                            }
                        ],
                        "live_strategy_issues": [],
                        "live_blockers": [],
                    },
                ]
            },
            statuses={"DRY_RUN_PASSED"},
        )

        messages = {item["message"]: item for item in blockers}
        self.assertEqual(messages["FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE"]["count"], 1)
        self.assertEqual(messages["STRATEGY_NOT_ELIGIBLE"]["count"], 1)
        self.assertEqual(messages["legacy live blocker fallback"]["count"], 1)
        self.assertNotIn("SPREAD_ADVISORY", messages)
        self.assertNotIn("STRATEGY_PROFILE_MISSING", messages)

    def test_live_readiness_family_breakdown_separates_repair_surfaces(self) -> None:
        breakdown = _intent_live_readiness_family_breakdown(
            {
                "results": [
                    {
                        "lane_id": "failure_trader:AUD_CAD:SHORT:BREAKOUT_FAILURE:LIMIT",
                        "status": "DRY_RUN_PASSED",
                        "intent": {
                            "pair": "AUD_CAD",
                            "side": "SHORT",
                            "order_type": "LIMIT",
                            "market_context": {"method": "BREAKOUT_FAILURE"},
                            "metadata": {"forecast_direction": "DOWN", "forecast_confidence": 0.49},
                        },
                        "risk_metrics": {"reward_risk": 1.4},
                        "risk_issues": [
                            {
                                "code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
                                "message": "forecast confidence below live floor",
                                "severity": "WARN",
                            },
                            {
                                "code": "CHART_DIRECTION_CONFLICT",
                                "message": "chart direction conflicts with lane side",
                                "severity": "WARN",
                            },
                        ],
                        "live_strategy_issues": [
                            {
                                "code": "STRATEGY_NOT_ELIGIBLE",
                                "message": "strategy profile is not live eligible",
                                "severity": "WARN",
                                "strategy_profile_evidence": {
                                    "profile_status": "BLOCK_UNTIL_NEW_EVIDENCE",
                                    "required_fix": "both live and pretrade feedback are negative",
                                    "live_net_jpy": -1501.57,
                                    "pretrade_net_jpy": -837.76,
                                },
                            }
                        ],
                        "live_blockers": [],
                    },
                    {
                        "lane_id": "range_trader:NZD_USD:SHORT:RANGE_ROTATION:MARKET",
                        "status": "DRY_RUN_PASSED",
                        "intent": {
                            "pair": "NZD_USD",
                            "side": "SHORT",
                            "order_type": "MARKET",
                            "market_context": {"method": "RANGE_ROTATION"},
                            "metadata": {"forecast_direction": "RANGE", "forecast_confidence": 0.63},
                        },
                        "risk_metrics": {"reward_risk": 2.85},
                        "risk_issues": [],
                        "live_strategy_issues": [
                            {
                                "code": "STRATEGY_NOT_ELIGIBLE",
                                "message": "strategy profile is not live eligible",
                                "severity": "WARN",
                                "strategy_profile_evidence": {
                                    "profile_status": "BLOCK_UNTIL_NEW_EVIDENCE",
                                    "required_fix": "both live and pretrade feedback are negative",
                                    "live_net_jpy": -1501.57,
                                    "pretrade_net_jpy": -837.76,
                                },
                            }
                        ],
                        "live_blockers": [],
                    },
                    {
                        "lane_id": "range_trader:NZD_USD:SHORT:RANGE_ROTATION:MARKET",
                        "status": "DRY_RUN_PASSED",
                        "intent": {
                            "pair": "NZD_USD",
                            "side": "SHORT",
                            "order_type": "MARKET",
                            "market_context": {"method": "RANGE_ROTATION"},
                            "metadata": {"forecast_direction": "RANGE", "forecast_confidence": 0.61},
                        },
                        "risk_metrics": {"reward_risk": 1.25},
                        "risk_issues": [],
                        "live_strategy_issues": [
                            {
                                "code": "STRATEGY_NOT_ELIGIBLE",
                                "message": "strategy profile is not live eligible",
                                "severity": "WARN",
                            }
                        ],
                        "live_blockers": [],
                    },
                    {
                        "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:STOP",
                        "status": "DRY_RUN_BLOCKED",
                        "risk_issues": [
                            {
                                "code": "REWARD_RISK_TOO_LOW",
                                "message": "reward/risk below floor",
                                "severity": "BLOCK",
                            }
                        ],
                        "live_strategy_issues": [],
                        "live_blockers": [],
                    },
                    {
                        "lane_id": "scalper:USD_JPY:LONG:EXECUTION:MARKET",
                        "status": "DRY_RUN_BLOCKED",
                        "risk_issues": [],
                        "live_strategy_issues": [],
                        "live_blockers": ["broker liquidity unavailable for order"],
                    },
                    {
                        "lane_id": "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION:MARKET",
                        "status": "LIVE_READY",
                        "risk_issues": [
                            {
                                "code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
                                "message": "live-ready advisory must not be counted",
                                "severity": "WARN",
                            }
                        ],
                        "live_strategy_issues": [],
                        "live_blockers": [],
                    },
                ]
            }
        )

        all_families = {item["family"]: item for item in breakdown["all_non_live_ready"]}
        dry_run_families = {item["family"]: item for item in breakdown["dry_run_passed"]}
        self.assertEqual(all_families["forecast"]["lane_count"], 1)
        self.assertEqual(all_families["strategy_profile"]["lane_count"], 2)
        self.assertEqual(all_families["market_structure"]["lane_count"], 1)
        self.assertEqual(all_families["risk_geometry"]["lane_count"], 1)
        self.assertEqual(all_families["execution_liquidity"]["lane_count"], 1)
        self.assertNotIn("risk_geometry", dry_run_families)
        self.assertEqual(dry_run_families["strategy_profile"]["lane_count"], 2)
        dry_candidates = breakdown["nearest_live_ready_candidates"]
        contradictory = next(
            item
            for item in dry_candidates
            if item["lane_id"]
            == "range_trader:NZD_USD:SHORT:RANGE_ROTATION:MARKET"
        )
        self.assertFalse(contradictory["candidate_contract_valid"])
        self.assertEqual(
            contradictory["candidate_integrity_status"],
            "DUPLICATE_CONTRADICTION",
        )
        self.assertEqual(contradictory["duplicate_lane_row_count"], 2)
        nearest_unique = dry_candidates[0]
        self.assertEqual(
            nearest_unique["lane_id"],
            "failure_trader:AUD_CAD:SHORT:BREAKOUT_FAILURE:LIMIT",
        )
        self.assertEqual(
            nearest_unique["blocker_families"],
            ["forecast", "market_structure", "strategy_profile"],
        )
        self.assertEqual(nearest_unique["reward_risk"], 1.4)
        blocker_evidence = next(
            blocker["strategy_profile_evidence"]
            for blocker in nearest_unique["blockers"]
            if blocker.get("code") == "STRATEGY_NOT_ELIGIBLE"
        )
        self.assertEqual(
            next(
                blocker["code"]
                for blocker in nearest_unique["blockers"]
                if blocker.get("code") == "STRATEGY_NOT_ELIGIBLE"
            ),
            "STRATEGY_NOT_ELIGIBLE",
        )
        self.assertEqual(blocker_evidence["profile_status"], "BLOCK_UNTIL_NEW_EVIDENCE")
        self.assertEqual(blocker_evidence["live_net_jpy"], -1501.57)
        candidate_lane_ids = [item["lane_id"] for item in breakdown["nearest_live_ready_candidates"]]
        self.assertEqual(len(candidate_lane_ids), len(set(candidate_lane_ids)))

    def test_no_live_ready_next_action_reports_unreachable_tp_proof_route(self) -> None:
        intents = {
            "generated_at_utc": _NOW.isoformat(),
            "results": [
                {
                    "lane_id": "range_trader:EUR_JPY:LONG:RANGE_ROTATION",
                    "status": "DRY_RUN_BLOCKED",
                    "intent": {
                        "pair": "EUR_JPY",
                        "side": "LONG",
                        "order_type": "LIMIT",
                        "market_context": {"method": "RANGE_ROTATION"},
                        "metadata": {
                            "forecast_direction": "RANGE",
                            "forecast_confidence": 1.0,
                        },
                    },
                    "risk_metrics": {"reward_risk": 1.62, "reward_jpy": 698.0},
                    "risk_issues": [
                        {
                            "code": "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                            "message": "exact TP evidence is not mature",
                            "severity": "BLOCK",
                        },
                        {
                            "code": "CHART_DIRECTION_CONFLICT",
                            "message": "advisory chart disagreement",
                            "severity": "WARN",
                        },
                    ],
                    "live_strategy_issues": [],
                    "live_blockers": [],
                    "live_blocker_codes": [
                        "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"
                    ],
                }
            ],
        }

        findings = _intent_findings(
            run_id=_NOW.isoformat(),
            intents=intents,
            target_state={},
            target_open=True,
            live_ready=[],
            active_positions=[],
            pending_entry_orders=[],
            coverage_optimization={
                "generated_at_utc": _NOW.isoformat(),
                "artifact_diagnostics": {
                    "intents_generated_at_utc": _NOW.isoformat(),
                    "intents_age_seconds": 0.0,
                    "intents_artifact_stale": False,
                    "intents_stale_after_seconds": 3600.0,
                },
            },
        )

        finding = next(
            item
            for item in findings
            if item["code"] == "TARGET_OPEN_NO_LIVE_READY_LANES"
        )
        self.assertIn(
            "TP_PROOF_ACQUISITION_ROUTE_UNREACHABLE",
            finding["next_action"],
        )
        self.assertNotIn("Narrow the next repair", finding["next_action"])
        self.assertIn("OANDA campaign firepower is audit-only", finding["next_action"])
        self.assertIn("Do not wait for receipts", finding["next_action"])

    def test_no_live_ready_snapshot_mismatch_routes_to_data_freshness_repair(self) -> None:
        broker_snapshot = _tp_acquisition_broker_snapshot()
        broker_snapshot["home_conversions"] = {"USD": 101.0, "CAD": 100.0}
        breakdown = _intent_live_readiness_family_breakdown(
            _tp_acquisition_intents(
                [
                    _tp_acquisition_route_result(
                        lane_id="failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
                        metadata=_valid_tp_collection_metadata(),
                    )
                ]
            ),
            broker_snapshot=broker_snapshot,
            sizing_receipt_root=_TP_SIZING_RECEIPT_ROOT,
        )

        candidate = breakdown["nearest_all_non_live_ready_candidates"][0]
        self.assertFalse(candidate["tp_proof_acquisition_route_reachable"])
        self.assertIn(
            "sizing conversion receipt does not match independent broker snapshot",
            candidate["tp_proof_acquisition_route_reason"],
        )
        action = _no_live_ready_next_action(
            coverage_refresh=None,
            intent_evidence_fresh=True,
            live_readiness_breakdown=breakdown,
        )
        self.assertIn("REFRESH_INTENT_SIZING_BROKER_SNAPSHOT", action)
        self.assertNotIn("Build or validate one exact non-market", action)

    def test_no_live_ready_positive_rotation_bad_units_routes_to_sizing_repair(self) -> None:
        result = _tp_acquisition_route_result(
            lane_id="failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
            metadata=_valid_tp_collection_metadata(),
            blocker_codes=[
                _NEGATIVE_TP_ROTATION_BLOCKER,
                "BAD_UNITS",
                "MARGIN_TOO_THIN_FOR_MIN_LOT",
            ],
        )
        intent = result["intent"]
        self.assertIsInstance(intent, dict)
        intent["units"] = 0
        breakdown = _intent_live_readiness_family_breakdown(
            _tp_acquisition_intents([result]),
            broker_snapshot=_tp_acquisition_broker_snapshot(),
            sizing_receipt_root=_TP_SIZING_RECEIPT_ROOT,
        )

        candidate = breakdown["nearest_all_non_live_ready_candidates"][0]
        self.assertFalse(candidate["tp_proof_acquisition_route_reachable"])
        self.assertIn(
            "intent units are not a signed-64-bit positive integer",
            candidate["tp_proof_acquisition_route_reason"],
        )
        action = _no_live_ready_next_action(
            coverage_refresh=None,
            intent_evidence_fresh=True,
            live_readiness_breakdown=breakdown,
        )
        self.assertIn("REPAIR_TP_PROOF_ACQUISITION_SIZING_CONTRACT", action)
        self.assertIn("RiskEngine sizing/allocation", action)
        self.assertNotIn("Build or validate one exact non-market", action)

    def test_root_cause_focus_treats_tp_route_snapshot_mismatch_as_data_freshness(self) -> None:
        broker_snapshot = _tp_acquisition_broker_snapshot()
        broker_snapshot["home_conversions"] = {"USD": 101.0, "CAD": 100.0}
        breakdown = _intent_live_readiness_family_breakdown(
            _tp_acquisition_intents(
                [
                    _tp_acquisition_route_result(
                        lane_id="failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
                        metadata=_valid_tp_collection_metadata(),
                    )
                ]
            ),
            broker_snapshot=broker_snapshot,
            sizing_receipt_root=_TP_SIZING_RECEIPT_ROOT,
        )

        root_focus = _root_cause_focus(
            findings=[
                {
                    "priority": "P0",
                    "layer": "opportunity",
                    "code": "TARGET_OPEN_NO_LIVE_READY_LANES",
                    "message": "daily target is open but order_intents has no LIVE_READY lanes",
                    "next_action": "stale sizing receipt",
                    "evidence": {
                        "live_readiness_blocker_families": breakdown,
                    },
                }
            ],
            runtime={"live_ready_lanes": 0},
            effect_metrics={"window": {"net_jpy": 100.0, "profit_factor": None}},
            execution_quality={},
        )

        self.assertEqual(root_focus["primary"]["family"], "DATA_FRESHNESS")
        self.assertEqual(root_focus["primary"]["priority"], "P0")
        self.assertIn(
            "TARGET_OPEN_NO_LIVE_READY_LANES",
            root_focus["primary"]["supporting_codes"],
        )

    def test_no_live_ready_next_action_names_reachable_tp_collection_lane(self) -> None:
        breakdown = _intent_live_readiness_family_breakdown(
            _tp_acquisition_intents(
                [
                    _tp_acquisition_route_result(
                        lane_id="failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
                        metadata=_valid_tp_collection_metadata(),
                    )
                ]
            ),
            broker_snapshot=_tp_acquisition_broker_snapshot(),
            sizing_receipt_root=_TP_SIZING_RECEIPT_ROOT,
        )

        candidate = breakdown["nearest_all_non_live_ready_candidates"][0]
        self.assertTrue(candidate["tp_proof_acquisition_required"])
        self.assertTrue(candidate["tp_proof_acquisition_route_reachable"])
        action = _no_live_ready_next_action(
            coverage_refresh=None,
            intent_evidence_fresh=True,
            live_readiness_breakdown=breakdown,
        )
        self.assertIn(
            "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
            action,
        )
        self.assertIn(_NEGATIVE_TP_ROTATION_BLOCKER, action)
        self.assertIn("do not repeat a broad market refresh", action.lower())

    def test_reachable_tp_route_sorts_ahead_of_smaller_dead_end(self) -> None:
        breakdown = _intent_live_readiness_family_breakdown(
            _tp_acquisition_intents(
                [
                    _tp_acquisition_route_result(lane_id="one-blocker-dead-end"),
                    _tp_acquisition_route_result(
                        lane_id="failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
                        metadata=_valid_tp_collection_metadata(),
                        blocker_codes=[
                            _NEGATIVE_TP_ROTATION_BLOCKER,
                            "REWARD_RISK_TOO_LOW",
                        ],
                    ),
                ]
            ),
            broker_snapshot=_tp_acquisition_broker_snapshot(),
            sizing_receipt_root=_TP_SIZING_RECEIPT_ROOT,
        )

        candidates = breakdown["nearest_all_non_live_ready_candidates"]
        self.assertEqual(
            candidates[0]["lane_id"],
            "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
        )
        self.assertTrue(candidates[0]["tp_proof_acquisition_route_reachable"])
        self.assertFalse(candidates[1]["tp_proof_acquisition_route_reachable"])

    def test_candidate_limit_does_not_truncate_the_only_reachable_tp_route(
        self,
    ) -> None:
        results = [
            _tp_acquisition_route_result(lane_id=f"dead-end-{index}")
            for index in range(9)
        ]
        results.append(
            _tp_acquisition_route_result(
                lane_id="failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
                metadata=_valid_tp_collection_metadata(),
                blocker_codes=[
                    _NEGATIVE_TP_ROTATION_BLOCKER,
                    "REWARD_RISK_TOO_LOW",
                ],
            )
        )
        breakdown = _intent_live_readiness_family_breakdown(
            _tp_acquisition_intents(results),
            broker_snapshot=_tp_acquisition_broker_snapshot(),
            sizing_receipt_root=_TP_SIZING_RECEIPT_ROOT,
        )

        candidates = breakdown["nearest_all_non_live_ready_candidates"]
        self.assertEqual(len(candidates), 8)
        self.assertEqual(
            candidates[0]["lane_id"],
            "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
        )
        action = _no_live_ready_next_action(
            coverage_refresh=None,
            intent_evidence_fresh=True,
            live_readiness_breakdown=breakdown,
        )
        self.assertIn("failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE", action)
        self.assertNotIn("TP_PROOF_ACQUISITION_ROUTE_UNREACHABLE", action)

    def test_all_approved_tp_acquisition_modes_require_self_consistent_types(
        self,
    ) -> None:
        modes = (
            _valid_tp_collection_metadata(),
            _valid_predictive_scout_metadata(),
            _valid_tp_proven_metadata(),
        )
        for metadata in modes:
            with self.subTest(mode=metadata["positive_rotation_mode"]):
                lane_id = (
                    "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT"
                    if metadata["positive_rotation_mode"]
                    == "PREDICTIVE_SCOUT_FORWARD_EVIDENCE"
                    else "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE"
                )
                breakdown = _intent_live_readiness_family_breakdown(
                    _tp_acquisition_intents(
                        [
                            _tp_acquisition_route_result(
                                lane_id=lane_id,
                                metadata=metadata,
                            )
                        ]
                    ),
                    broker_snapshot=_tp_acquisition_broker_snapshot(),
                    sizing_receipt_root=_TP_SIZING_RECEIPT_ROOT,
                )
                candidate = breakdown["nearest_all_non_live_ready_candidates"][0]
                self.assertTrue(candidate["tp_proof_acquisition_route_reachable"])
                self.assertEqual(
                    candidate["tp_proof_acquisition_route_status"],
                    "TP_PROOF_ACQUISITION_ROUTE_REACHABLE",
                )

    def test_zero_incremental_margin_hedge_receipt_remains_reachable(self) -> None:
        snapshot = _tp_acquisition_broker_snapshot()
        snapshot["account"] = {"hedging_enabled": True}
        snapshot["positions"] = [
            {
                "trade_id": "existing-long",
                "pair": "EUR_USD",
                "side": "LONG",
                "units": 1000,
                "entry_price": 1.1000,
                "owner": "trader",
            }
        ]
        result = _tp_acquisition_route_result(
            lane_id="failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
            metadata=_valid_tp_collection_metadata(),
            estimated_margin_jpy_override=0.0,
        )

        breakdown = _intent_live_readiness_family_breakdown(
            _tp_acquisition_intents([result]),
            broker_snapshot=snapshot,
            sizing_receipt_root=_TP_SIZING_RECEIPT_ROOT,
        )

        candidate = breakdown["nearest_all_non_live_ready_candidates"][0]
        self.assertTrue(candidate["tp_proof_acquisition_route_reachable"])
        self.assertEqual(
            candidate["tp_proof_acquisition_route_status"],
            "TP_PROOF_ACQUISITION_ROUTE_REACHABLE",
        )

    def test_tp_collection_route_rejects_coercible_or_non_finite_proof_fields(
        self,
    ) -> None:
        variants = (
            ("positive_rotation_mode", 1),
            ("positive_rotation_mode", "tp_proof_collection_harvest"),
            ("positive_rotation_proof_collection_ready", 1),
            ("positive_rotation_proof_collection_min_trades", "5"),
            ("positive_rotation_proof_collection_target_trades", 20.0),
            ("capture_take_profit_trades", "7"),
            ("capture_take_profit_trades", 7.0),
            ("capture_take_profit_losses", False),
            ("capture_take_profit_expectancy_jpy", "918.2276"),
            ("capture_take_profit_expectancy_jpy", float("inf")),
            ("positive_rotation_pessimistic_expectancy_jpy", "223.6428"),
            ("positive_rotation_pessimistic_expectancy_jpy", float("nan")),
            ("positive_rotation_live_ready", "false"),
        )
        for field, malformed in variants:
            with self.subTest(field=field, malformed=malformed):
                metadata = _valid_tp_collection_metadata()
                metadata[field] = malformed
                breakdown = _intent_live_readiness_family_breakdown(
                    {
                        "results": [
                            _tp_acquisition_route_result(
                                lane_id=f"malformed-{field}",
                                metadata=metadata,
                            )
                        ]
                    }
                )
                candidate = breakdown["nearest_all_non_live_ready_candidates"][0]
                self.assertFalse(candidate["tp_proof_acquisition_route_reachable"])
                self.assertEqual(
                    candidate["tp_proof_acquisition_route_status"],
                    "TP_PROOF_ACQUISITION_ROUTE_UNREACHABLE",
                )

    def test_predictive_scout_requires_exact_seed_and_exact_false_live_ready(
        self,
    ) -> None:
        variants = (
            ("bidask_replay_precision_seed", "true"),
            ("bidask_replay_precision_seed", 1),
            ("positive_rotation_live_ready", "false"),
            ("positive_rotation_live_ready", 0),
            ("positive_rotation_live_ready", None),
        )
        for field, malformed in variants:
            with self.subTest(field=field, malformed=malformed):
                metadata = _valid_predictive_scout_metadata()
                metadata[field] = malformed
                breakdown = _intent_live_readiness_family_breakdown(
                    {
                        "results": [
                            _tp_acquisition_route_result(
                                lane_id=f"malformed-scout-{field}",
                                metadata=metadata,
                            )
                        ]
                    }
                )
                candidate = breakdown["nearest_all_non_live_ready_candidates"][0]
                self.assertFalse(candidate["tp_proof_acquisition_route_reachable"])

    def test_oanda_firepower_remains_audit_only_despite_positive_metrics(self) -> None:
        metadata = _valid_tp_proven_metadata()
        metadata["positive_rotation_mode"] = "OANDA_CAMPAIGN_FIREPOWER_HARVEST"
        breakdown = _intent_live_readiness_family_breakdown(
            {
                "results": [
                    _tp_acquisition_route_result(
                        lane_id="oanda-audit-only",
                        metadata=metadata,
                    )
                ]
            }
        )

        candidate = breakdown["nearest_all_non_live_ready_candidates"][0]
        self.assertFalse(candidate["tp_proof_acquisition_route_reachable"])
        self.assertIn(
            "audit-only",
            candidate["tp_proof_acquisition_route_reason"],
        )

    def test_oanda_actual_audit_only_blocker_requires_unreachable_route(self) -> None:
        code = "OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED"
        breakdown = _intent_live_readiness_family_breakdown(
            {
                "results": [
                    _tp_acquisition_route_result(
                        lane_id="oanda-actual-producer-shape",
                        metadata={
                            "positive_rotation_mode": (
                                "OANDA_CAMPAIGN_FIREPOWER_HARVEST"
                            ),
                            "positive_rotation_live_ready": False,
                            "positive_rotation_oanda_campaign_audit_only": True,
                            "positive_rotation_oanda_campaign_live_permission": False,
                            "positive_rotation_oanda_campaign_local_tp_proof_required": True,
                        },
                        blocker_codes=[code],
                        authoritative_codes=[code],
                    )
                ]
            }
        )

        candidate = breakdown["nearest_all_non_live_ready_candidates"][0]
        self.assertTrue(candidate["tp_proof_acquisition_required"])
        self.assertFalse(candidate["tp_proof_acquisition_route_reachable"])
        self.assertIn("audit-only", candidate["tp_proof_acquisition_route_reason"])

    def test_predictive_scout_route_rejects_market_or_unsigned_rule(self) -> None:
        valid = _valid_predictive_scout_metadata()
        unsigned = dict(valid)
        unsigned.pop("predictive_scout_rule_digest")
        rows = [
            _tp_acquisition_route_result(
                lane_id="unsigned-scout",
                metadata=unsigned,
            ),
            _tp_acquisition_route_result(
                lane_id="market-scout",
                metadata=valid,
            ),
        ]
        market_intent = rows[1]["intent"]
        assert isinstance(market_intent, dict)
        market_intent["order_type"] = "MARKET"
        breakdown = _intent_live_readiness_family_breakdown({"results": rows})

        candidates = {
            item["lane_id"]: item
            for item in breakdown["nearest_all_non_live_ready_candidates"]
        }
        self.assertFalse(
            candidates["unsigned-scout"]["tp_proof_acquisition_route_reachable"]
        )
        self.assertFalse(
            candidates["market-scout"]["tp_proof_acquisition_route_reachable"]
        )

    def test_tp_collection_route_revalidates_shape_scope_guard_and_wilson(self) -> None:
        variants = (
            ("attach_take_profit_on_fill", False),
            ("capture_take_profit_scope_key", "EUR_USD|SHORT|WRONG"),
            ("loss_asymmetry_guard_relaxed", True),
            ("capture_take_profit_avg_win_jpy", 0.0),
            ("capture_market_close_expectancy_jpy", 1.0),
            ("positive_rotation_pessimistic_expectancy_jpy", 224.0),
        )
        for field, value in variants:
            with self.subTest(field=field):
                metadata = _valid_tp_collection_metadata()
                metadata[field] = value
                breakdown = _intent_live_readiness_family_breakdown(
                    {
                        "results": [
                            _tp_acquisition_route_result(
                                lane_id=f"tampered-{field}",
                                metadata=metadata,
                            )
                        ]
                    }
                )
                candidate = breakdown["nearest_all_non_live_ready_candidates"][0]
                self.assertFalse(
                    candidate["tp_proof_acquisition_route_reachable"]
                )

    def test_tp_routes_reject_impossible_arithmetic_and_non_exact_receipts(
        self,
    ) -> None:
        for valid_metadata in (
            _valid_tp_collection_metadata(),
            _valid_tp_proven_metadata(),
        ):
            trades = int(valid_metadata["capture_take_profit_trades"])
            variants = (
                ("wins-exceed-trades", {"capture_take_profit_wins": trades + 1}),
                ("counts-do-not-sum", {"capture_take_profit_wins": trades - 1}),
                ("expectancy-not-net", {"capture_take_profit_expectancy_jpy": 1.0}),
                ("net-not-outcomes", {"capture_take_profit_net_jpy": 1.0}),
                ("negative-average-loss", {"capture_take_profit_avg_loss_jpy": -1.0}),
                (
                    "broad-scope",
                    {
                        "capture_take_profit_exact_vehicle_required": False,
                        "capture_take_profit_scope": "PAIR_SIDE_METHOD",
                        "capture_take_profit_scope_key": (
                            "EUR_USD|SHORT|BREAKOUT_FAILURE|TAKE_PROFIT_ORDER"
                        ),
                    },
                ),
                ("wrong-vehicle", {"capture_take_profit_vehicle": "LIMIT"}),
                (
                    "wrong-source",
                    {"capture_take_profit_metrics_source": "data/capture_economics.json"},
                ),
                ("wrong-scope-key", {"capture_take_profit_scope_key": "garbage"}),
            )
            for label, changes in variants:
                with self.subTest(
                    mode=valid_metadata["positive_rotation_mode"],
                    variant=label,
                ):
                    metadata = dict(valid_metadata)
                    metadata.update(changes)
                    breakdown = _intent_live_readiness_family_breakdown(
                        {
                            "results": [
                                _tp_acquisition_route_result(
                                    lane_id=f"invalid-{label}",
                                    metadata=metadata,
                                )
                            ]
                        }
                    )
                    candidate = breakdown["nearest_all_non_live_ready_candidates"][0]
                    self.assertFalse(
                        candidate["tp_proof_acquisition_route_reachable"]
                    )

    def test_contradictory_duplicate_lane_cannot_hide_tp_dead_end(self) -> None:
        lane_id = "duplicate-lane"
        dead_end = _tp_acquisition_route_result(lane_id=lane_id)
        forecast_only = _tp_acquisition_route_result(
            lane_id=lane_id,
            blocker_codes=["FORECAST_WATCH_ONLY"],
            authoritative_codes=["FORECAST_WATCH_ONLY"],
        )
        breakdown = _intent_live_readiness_family_breakdown(
            {"results": [dead_end, forecast_only]}
        )

        candidate = breakdown["nearest_all_non_live_ready_candidates"][0]
        self.assertFalse(candidate["candidate_contract_valid"])
        self.assertEqual(
            candidate["candidate_integrity_status"],
            "DUPLICATE_CONTRADICTION",
        )
        self.assertTrue(candidate["tp_proof_acquisition_required"])
        self.assertFalse(candidate["tp_proof_acquisition_route_reachable"])
        action = _no_live_ready_next_action(
            coverage_refresh=None,
            intent_evidence_fresh=True,
            live_readiness_breakdown=breakdown,
        )
        self.assertIn("INTENT_CANDIDATE_CONTRACT_UNTRUSTED", action)

    def test_identical_duplicate_lane_collapses_without_changing_contract(self) -> None:
        row = _tp_acquisition_route_result(
            lane_id="failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
            metadata=_valid_tp_collection_metadata(),
        )
        duplicate = json.loads(json.dumps(row))
        breakdown = _intent_live_readiness_family_breakdown(
            _tp_acquisition_intents([row, duplicate]),
            broker_snapshot=_tp_acquisition_broker_snapshot(),
            sizing_receipt_root=_TP_SIZING_RECEIPT_ROOT,
        )

        candidate = breakdown["nearest_all_non_live_ready_candidates"][0]
        self.assertTrue(candidate["candidate_contract_valid"])
        self.assertEqual(
            candidate["candidate_integrity_status"],
            "IDENTICAL_DUPLICATE_COLLAPSED",
        )
        self.assertTrue(candidate["tp_proof_acquisition_route_reachable"])

    def test_tp_route_requires_exact_producer_identity_guard_sizing_and_lane(
        self,
    ) -> None:
        row = _tp_acquisition_route_result(
            lane_id="failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
            metadata=_valid_tp_collection_metadata(),
        )
        variants: list[tuple[str, dict[str, object]]] = []

        def mutated(label: str) -> dict[str, object]:
            duplicate = json.loads(json.dumps(row))
            variants.append((label, duplicate))
            return duplicate

        for owner in ("operator_manual", None):
            duplicate = mutated(f"owner-{owner}")
            intent = duplicate["intent"]
            assert isinstance(intent, dict)
            if owner is None:
                intent.pop("owner")
            else:
                intent["owner"] = owner

        for field, value in (
            ("pair", " eur_usd "),
            ("order_type", "STOP"),
            ("order_type", " STOP-ENTRY "),
        ):
            duplicate = mutated(f"{field}-{value}")
            intent = duplicate["intent"]
            assert isinstance(intent, dict)
            intent[field] = value

        duplicate = mutated("method-alias")
        market_context = duplicate["intent"]["market_context"]  # type: ignore[index]
        assert isinstance(market_context, dict)
        market_context["method"] = "breakout-failure"

        for lane_id in (
            "failure_trader:GBP_USD:LONG:RANGE_ROTATION",
            "garbage",
            123,
        ):
            duplicate = mutated(f"lane-{lane_id}")
            duplicate["lane_id"] = lane_id

        for field, value in (
            ("loss_asymmetry_guard_loss_cap_jpy", None),
            ("loss_asymmetry_guard_base_max_loss_jpy", "500"),
            ("loss_asymmetry_guard_effective_max_loss_jpy", -1.0),
        ):
            duplicate = mutated(f"guard-{field}")
            metadata = duplicate["intent"]["metadata"]  # type: ignore[index]
            assert isinstance(metadata, dict)
            if value is None:
                metadata.pop(field)
            else:
                metadata[field] = value

        for units in (1, 1001, 10**12, 2**63 - 1, 10**30):
            duplicate = mutated(f"units-{units}")
            intent = duplicate["intent"]
            assert isinstance(intent, dict)
            intent["units"] = units
            metadata = intent["metadata"]
            assert isinstance(metadata, dict)
            metadata["sizing_actual_units"] = units

        duplicate = mutated("guard-cap-not-capture-average-winner")
        intent = duplicate["intent"]
        assert isinstance(intent, dict)
        metadata = intent["metadata"]
        assert isinstance(metadata, dict)
        metadata["loss_asymmetry_guard_loss_cap_jpy"] = 600.0
        metadata["loss_asymmetry_guard_effective_max_loss_jpy"] = 600.0
        metadata["max_loss_jpy"] = 600.0
        metadata["sizing_actual_risk_cap_jpy"] = 600.0
        risk_jpy = float(metadata["sizing_actual_risk_jpy"])
        metadata["sizing_actual_risk_cap_utilization"] = round(
            risk_jpy / 600.0,
            6,
        )

        duplicate = mutated("risk-receipt-mismatch")
        risk_metrics = duplicate["risk_metrics"]
        assert isinstance(risk_metrics, dict)
        risk_metrics["risk_jpy"] = 1.0

        for label, duplicate in variants:
            with self.subTest(variant=label):
                breakdown = _intent_live_readiness_family_breakdown(
                    _tp_acquisition_intents([duplicate]),
                    broker_snapshot=_tp_acquisition_broker_snapshot(),
                    sizing_receipt_root=_TP_SIZING_RECEIPT_ROOT,
                )
                candidate = breakdown["nearest_all_non_live_ready_candidates"][0]
                self.assertFalse(
                    candidate["tp_proof_acquisition_route_reachable"]
                )

    def test_tp_route_rejects_joint_units_and_conversion_self_consistency_tamper(
        self,
    ) -> None:
        row = _tp_acquisition_route_result(
            lane_id="failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
            metadata=_valid_tp_collection_metadata(),
        )
        intent = row["intent"]
        assert isinstance(intent, dict)
        metadata = intent["metadata"]
        assert isinstance(metadata, dict)

        # Keep JPY-per-pip and margin numerically unchanged by scaling units up
        # and quote conversion down.  This passed the former circular check.
        intent["units"] = 1_000_000
        metadata["sizing_actual_units"] = 1_000_000
        metadata["sizing_actual_quote_to_jpy"] = 0.1

        breakdown = _intent_live_readiness_family_breakdown(
            _tp_acquisition_intents([row]),
            broker_snapshot=_tp_acquisition_broker_snapshot(),
            sizing_receipt_root=_TP_SIZING_RECEIPT_ROOT,
        )
        candidate = breakdown["nearest_all_non_live_ready_candidates"][0]

        self.assertFalse(candidate["tp_proof_acquisition_route_reachable"])
        self.assertIn(
            "independent broker snapshot",
            candidate["tp_proof_acquisition_route_reason"],
        )

        producer_choice_tamper = _tp_acquisition_route_result(
            lane_id="failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
            metadata=_valid_tp_collection_metadata(),
        )
        tampered_intent = producer_choice_tamper["intent"]
        assert isinstance(tampered_intent, dict)
        tampered_metadata = tampered_intent["metadata"]
        assert isinstance(tampered_metadata, dict)
        tampered_metrics = producer_choice_tamper["risk_metrics"]
        assert isinstance(tampered_metrics, dict)
        tampered_intent["units"] = 2000
        tampered_metrics.update(
            {
                "risk_jpy": 200.0,
                "reward_jpy": 200.0,
                "jpy_per_pip": 20.0,
                "estimated_margin_jpy": 8800.0,
            }
        )
        tampered_metadata.update(
            {
                "sizing_actual_units": 2000,
                "sizing_actual_risk_jpy": 200.0,
                "sizing_actual_reward_jpy": 200.0,
                "sizing_actual_jpy_per_pip": 20.0,
                "sizing_actual_estimated_margin_jpy": 8800.0,
                "sizing_actual_risk_cap_utilization": 0.4,
            }
        )
        producer_choice_breakdown = _intent_live_readiness_family_breakdown(
            _tp_acquisition_intents([producer_choice_tamper]),
            broker_snapshot=_tp_acquisition_broker_snapshot(),
            sizing_receipt_root=_TP_SIZING_RECEIPT_ROOT,
        )
        producer_choice_candidate = producer_choice_breakdown[
            "nearest_all_non_live_ready_candidates"
        ][0]
        self.assertFalse(
            producer_choice_candidate["tp_proof_acquisition_route_reachable"]
        )
        self.assertIn(
            "current sizing does not match independent producer receipt",
            producer_choice_candidate["tp_proof_acquisition_route_reason"],
        )

    def test_tp_route_rejects_rehashed_metadata_receipt_and_stale_snapshot(
        self,
    ) -> None:
        row = _tp_acquisition_route_result(
            lane_id="failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
            metadata=_valid_tp_collection_metadata(),
        )
        intent = row["intent"]
        assert isinstance(intent, dict)
        metadata = intent["metadata"]
        assert isinstance(metadata, dict)
        receipt = metadata["sizing_actual_conversion_receipt"]
        assert isinstance(receipt, dict)
        source = receipt["conversion_source"]
        assert isinstance(source, dict)
        source["home_conversion"] = 0.1
        source["selected_quote_to_jpy"] = 0.1
        unsigned = {
            key: value
            for key, value in receipt.items()
            if key != "snapshot_conversion_sha256"
        }
        receipt["snapshot_conversion_sha256"] = hashlib.sha256(
            json.dumps(
                unsigned,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

        breakdown = _intent_live_readiness_family_breakdown(
            _tp_acquisition_intents([row]),
            broker_snapshot=_tp_acquisition_broker_snapshot(),
            sizing_receipt_root=_TP_SIZING_RECEIPT_ROOT,
        )
        candidate = breakdown["nearest_all_non_live_ready_candidates"][0]
        self.assertFalse(candidate["tp_proof_acquisition_route_reachable"])
        self.assertIn(
            "does not match independent broker snapshot",
            candidate["tp_proof_acquisition_route_reason"],
        )

        fresh_row = _tp_acquisition_route_result(
            lane_id="failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
            metadata=_valid_tp_collection_metadata(),
        )
        stale_intents = _tp_acquisition_intents([fresh_row])
        stale_intents["generated_at_utc"] = (
            _REPLAY_NOW + timedelta(seconds=21)
        ).isoformat().replace("+00:00", "Z")
        stale_breakdown = _intent_live_readiness_family_breakdown(
            stale_intents,
            broker_snapshot=_tp_acquisition_broker_snapshot(),
            sizing_receipt_root=_TP_SIZING_RECEIPT_ROOT,
        )
        stale_candidate = stale_breakdown[
            "nearest_all_non_live_ready_candidates"
        ][0]
        self.assertFalse(stale_candidate["tp_proof_acquisition_route_reachable"])
        self.assertIn(
            "was not fresh at intent generation",
            stale_candidate["tp_proof_acquisition_route_reason"],
        )

    def test_unique_noncanonical_raw_contract_is_not_actionable(self) -> None:
        row = _tp_acquisition_route_result(
            lane_id="failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
            metadata=_valid_tp_collection_metadata(),
        )
        row["future_unprojected_field"] = float("nan")
        breakdown = _intent_live_readiness_family_breakdown({"results": [row]})

        candidate = breakdown["nearest_all_non_live_ready_candidates"][0]
        self.assertFalse(candidate["candidate_contract_valid"])
        self.assertEqual(
            candidate["candidate_integrity_status"],
            "RAW_CONTRACT_INVALID",
        )
        self.assertIsNone(
            breakdown["nearest_all_non_live_ready_actionable_candidate"]
        )

    def test_duplicate_lane_requires_identical_raw_execution_contract(self) -> None:
        row = _tp_acquisition_route_result(
            lane_id="failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
            metadata=_valid_tp_collection_metadata(),
        )
        variants: list[tuple[str, dict[str, object]]] = []
        for field, value in (
            ("pair", "GBP_USD"),
            ("side", "LONG"),
            ("order_type", "LIMIT"),
            ("units", 2000),
            ("entry", 1.1001),
            ("tp", 1.0989),
            ("sl", 1.1011),
        ):
            duplicate = json.loads(json.dumps(row))
            intent = duplicate["intent"]
            assert isinstance(intent, dict)
            intent[field] = value
            variants.append((field, duplicate))

        method_duplicate = json.loads(json.dumps(row))
        method_intent = method_duplicate["intent"]
        assert isinstance(method_intent, dict)
        market_context = method_intent["market_context"]
        assert isinstance(market_context, dict)
        market_context["method"] = "RANGE_ROTATION"
        variants.append(("market_context.method", method_duplicate))

        evidence_duplicate = json.loads(json.dumps(row))
        evidence_issues = evidence_duplicate["risk_issues"]
        assert isinstance(evidence_issues, list)
        assert isinstance(evidence_issues[0], dict)
        evidence_issues[0]["authoritative_evidence"] = {"source": "different"}
        variants.append(("authoritative_evidence", evidence_duplicate))

        receipt_duplicate = json.loads(json.dumps(row))
        receipt_intent = receipt_duplicate["intent"]
        assert isinstance(receipt_intent, dict)
        receipt_metadata = receipt_intent["metadata"]
        assert isinstance(receipt_metadata, dict)
        receipt_metadata["capture_take_profit_net_jpy"] = 6427.6031
        variants.append(("acquisition_receipt", receipt_duplicate))

        risk_allowed_duplicate = json.loads(json.dumps(row))
        risk_allowed_duplicate["risk_allowed"] = False
        variants.append(("risk_allowed", risk_allowed_duplicate))

        note_duplicate = json.loads(json.dumps(row))
        note_duplicate["note"] = "different execution-source contract"
        variants.append(("note", note_duplicate))

        for label, duplicate in variants:
            with self.subTest(field=label):
                breakdown = _intent_live_readiness_family_breakdown(
                    {"results": [row, duplicate]}
                )
                candidate = breakdown["nearest_all_non_live_ready_candidates"][0]
                self.assertFalse(candidate["candidate_contract_valid"])
                self.assertEqual(
                    candidate["candidate_integrity_status"],
                    "DUPLICATE_CONTRADICTION",
                )
                self.assertFalse(
                    candidate["tp_proof_acquisition_route_reachable"]
                )

    def test_next_action_skips_incomplete_candidate_for_later_complete_one(self) -> None:
        action = _no_live_ready_next_action(
            coverage_refresh=None,
            intent_evidence_fresh=True,
            live_readiness_breakdown={
                "nearest_live_ready_candidates": [
                    {
                        "lane_id": "incomplete-first",
                        "candidate_contract_valid": True,
                        "authoritative_live_blocker_codes": ["FORECAST_WATCH_ONLY"],
                        "authoritative_live_blocker_codes_complete": False,
                    }
                ],
                "nearest_all_non_live_ready_candidates": [
                    {
                        "lane_id": "complete-second",
                        "candidate_contract_valid": True,
                        "authoritative_live_blocker_codes": ["REWARD_RISK_TOO_LOW"],
                        "authoritative_live_blocker_codes_complete": True,
                    }
                ],
            },
        )

        self.assertIn("complete-second", action)
        self.assertNotIn("without a named gate", action)

    def test_actionable_candidate_is_computed_before_top_eight_slice(self) -> None:
        rows = [
            _tp_acquisition_route_result(
                lane_id=f"incomplete-{index}",
                blocker_codes=["FORECAST_WATCH_ONLY"],
                authoritative_codes=["DIFFERENT_CODE"],
            )
            for index in range(9)
        ]
        rows.append(
            _tp_acquisition_route_result(
                lane_id="complete-after-cutoff",
                blocker_codes=["FORECAST_WATCH_ONLY", "REWARD_RISK_TOO_LOW"],
                authoritative_codes=[
                    "FORECAST_WATCH_ONLY",
                    "REWARD_RISK_TOO_LOW",
                ],
            )
        )
        breakdown = _intent_live_readiness_family_breakdown({"results": rows})

        visible_ids = {
            item["lane_id"]
            for item in breakdown["nearest_all_non_live_ready_candidates"]
        }
        self.assertNotIn("complete-after-cutoff", visible_ids)
        self.assertEqual(
            breakdown["nearest_all_non_live_ready_actionable_candidate"]["lane_id"],
            "complete-after-cutoff",
        )
        action = _no_live_ready_next_action(
            coverage_refresh=None,
            intent_evidence_fresh=True,
            live_readiness_breakdown=breakdown,
        )
        self.assertIn("complete-after-cutoff", action)

    def test_tp_route_not_required_ignores_malformed_acquisition_metadata(self) -> None:
        breakdown = _intent_live_readiness_family_breakdown(
            {
                "results": [
                    _tp_acquisition_route_result(
                        lane_id="forecast-only",
                        metadata={
                            "positive_rotation_mode": 1,
                            "capture_take_profit_trades": "7",
                            "bidask_replay_precision_seed": "true",
                        },
                        blocker_codes=["FORECAST_WATCH_ONLY"],
                    )
                ]
            }
        )

        candidate = breakdown["nearest_all_non_live_ready_candidates"][0]
        self.assertFalse(candidate["tp_proof_acquisition_required"])
        self.assertIsNone(candidate["tp_proof_acquisition_route_reachable"])
        self.assertEqual(
            candidate["tp_proof_acquisition_route_status"],
            "TP_PROOF_ACQUISITION_ROUTE_NOT_REQUIRED",
        )

    def test_structured_negative_blocker_requires_route_without_authoritative_codes(
        self,
    ) -> None:
        variants = (
            {"include_authoritative_codes": False},
            {
                "include_authoritative_codes": True,
                "authoritative_codes": ["UNRELATED_BLOCKER"],
            },
        )
        for variant in variants:
            with self.subTest(variant=variant):
                breakdown = _intent_live_readiness_family_breakdown(
                    {
                        "results": [
                            _tp_acquisition_route_result(
                                lane_id="structured-negative",
                                **variant,
                            )
                        ]
                    }
                )
                candidate = breakdown["nearest_all_non_live_ready_candidates"][0]
                self.assertTrue(candidate["tp_proof_acquisition_required"])
                self.assertFalse(candidate["tp_proof_acquisition_route_reachable"])

    def test_authoritative_negative_code_requires_route_when_structured_code_differs(
        self,
    ) -> None:
        breakdown = _intent_live_readiness_family_breakdown(
            {
                "results": [
                    _tp_acquisition_route_result(
                        lane_id="authoritative-negative",
                        blocker_codes=["REWARD_RISK_TOO_LOW"],
                        authoritative_codes=[_NEGATIVE_TP_ROTATION_BLOCKER],
                    )
                ]
            }
        )

        candidate = breakdown["nearest_all_non_live_ready_candidates"][0]
        self.assertFalse(candidate["authoritative_live_blocker_codes_complete"])
        self.assertTrue(candidate["tp_proof_acquisition_required"])
        self.assertFalse(candidate["tp_proof_acquisition_route_reachable"])

    def test_next_action_rejects_legacy_negative_candidate_without_route_fields(
        self,
    ) -> None:
        action = _no_live_ready_next_action(
            coverage_refresh=None,
            intent_evidence_fresh=True,
            live_readiness_breakdown={
                "nearest_live_ready_candidates": [
                    {
                        "lane_id": "legacy-dead-end",
                        "blockers": [{"code": _NEGATIVE_TP_ROTATION_BLOCKER}],
                        "authoritative_live_blocker_codes": [
                            _NEGATIVE_TP_ROTATION_BLOCKER
                        ],
                        "authoritative_live_blocker_codes_complete": True,
                    }
                ],
                "nearest_all_non_live_ready_candidates": [],
            },
        )

        self.assertIn("TP_PROOF_ACQUISITION_ROUTE_UNREACHABLE", action)
        self.assertNotIn("legacy-dead-end", action)

    def test_tp_proven_mode_rejects_non_boolean_live_ready_claim(self) -> None:
        metadata = _valid_tp_proven_metadata()
        metadata["positive_rotation_live_ready"] = 1
        breakdown = _intent_live_readiness_family_breakdown(
            {
                "results": [
                    _tp_acquisition_route_result(
                        lane_id="malformed-tp-proven",
                        metadata=metadata,
                    )
                ]
            }
        )

        candidate = breakdown["nearest_all_non_live_ready_candidates"][0]
        self.assertFalse(candidate["tp_proof_acquisition_route_reachable"])

    def test_no_live_ready_next_action_prefers_dry_run_passed_candidate_like_report(
        self,
    ) -> None:
        action = _no_live_ready_next_action(
            coverage_refresh=None,
            intent_evidence_fresh=True,
            live_readiness_breakdown={
                "nearest_live_ready_candidates": [
                    {
                        "lane_id": "dry-run-passed",
                        "blocker_count": 1,
                        "blockers": [{"code": "FORECAST_WATCH_ONLY"}],
                        "authoritative_live_blocker_codes": ["FORECAST_WATCH_ONLY"],
                        "authoritative_live_blocker_codes_complete": True,
                    }
                ],
                "nearest_all_non_live_ready_candidates": [
                    {
                        "lane_id": "dry-run-blocked",
                        "blocker_count": 1,
                        "blockers": [{"code": "RISK_BLOCK"}],
                        "authoritative_live_blocker_codes": ["RISK_BLOCK"],
                        "authoritative_live_blocker_codes_complete": True,
                    }
                ],
            },
        )

        self.assertIn("dry-run-passed", action)
        self.assertIn("FORECAST_WATCH_ONLY", action)
        self.assertNotIn("dry-run-blocked", action)

    def test_market_evidence_refresh_still_outranks_lane_specific_repair(self) -> None:
        action = _no_live_ready_next_action(
            coverage_refresh={"requires_market_evidence_refresh": True},
            intent_evidence_fresh=True,
            live_readiness_breakdown={
                "nearest_all_non_live_ready_candidates": [
                    {
                        "lane_id": "range_trader:EUR_JPY:LONG:RANGE_ROTATION",
                        "blockers": [
                            {
                                "code": "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"
                            }
                        ],
                    }
                ]
            },
        )

        self.assertIn("Refresh broker truth", action)
        self.assertNotIn("Narrow the next repair", action)

    def test_missing_market_matrix_alone_is_not_a_broker_quote_refresh(self) -> None:
        refresh = _coverage_market_evidence_refresh(
            {
                "artifact_diagnostics": {
                    "requires_market_evidence_refresh": True,
                    "market_context_matrix_missing": True,
                    "all_lanes_spread_blocked": False,
                    "all_lanes_quote_stale": False,
                }
            }
        )

        self.assertIsNone(refresh)

    def test_truthy_string_market_flags_do_not_trigger_broker_refresh(self) -> None:
        for diagnostics in (
            {
                "requires_market_evidence_refresh": "true",
                "all_lanes_spread_blocked": True,
                "all_lanes_quote_stale": False,
            },
            {
                "requires_market_evidence_refresh": True,
                "all_lanes_spread_blocked": "false",
                "all_lanes_quote_stale": "false",
            },
        ):
            with self.subTest(diagnostics=diagnostics):
                self.assertIsNone(
                    _coverage_market_evidence_refresh(
                        {"artifact_diagnostics": diagnostics}
                    )
                )

    def test_stale_intent_evidence_cannot_name_a_lane_specific_repair(self) -> None:
        action = _no_live_ready_next_action(
            coverage_refresh={"requires_market_evidence_refresh": True},
            intent_evidence_fresh=False,
            live_readiness_breakdown={
                "nearest_all_non_live_ready_candidates": [
                    {
                        "lane_id": "stale-lane",
                        "authoritative_live_blocker_codes": ["STALE_BLOCKER"],
                        "authoritative_live_blocker_codes_complete": True,
                    }
                ]
            },
        )

        self.assertIn("Regenerate order intents", action)
        self.assertNotIn("stale-lane", action)

    def test_intent_freshness_rejects_self_reported_unbounded_threshold(self) -> None:
        old = _NOW - timedelta(days=13)
        freshness = _intent_artifact_freshness(
            run_id=_NOW.isoformat(),
            coverage_optimization={
                "generated_at_utc": old.isoformat(),
                "artifact_diagnostics": {
                    "intents_generated_at_utc": old.isoformat(),
                    "intents_age_seconds": 0.0,
                    "intents_artifact_stale": False,
                    "intents_stale_after_seconds": 999999999.0,
                },
            },
            intents={"generated_at_utc": old.isoformat()},
        )

        self.assertFalse(freshness["fresh"])

    def test_intent_freshness_rejects_timezone_less_proof_clocks(self) -> None:
        aware = _NOW.isoformat()
        naive = _NOW.replace(tzinfo=None).isoformat()
        variants = (
            (naive, aware, aware, aware),
            (aware, naive, aware, aware),
            (aware, aware, naive, aware),
            (aware, aware, aware, naive),
        )
        for run_id, intents_at, diagnostics_at, coverage_at in variants:
            with self.subTest(
                run_id=run_id,
                intents_at=intents_at,
                diagnostics_at=diagnostics_at,
                coverage_at=coverage_at,
            ):
                freshness = _intent_artifact_freshness(
                    run_id=run_id,
                    coverage_optimization={
                        "generated_at_utc": coverage_at,
                        "artifact_diagnostics": {
                            "intents_generated_at_utc": diagnostics_at,
                            "intents_age_seconds": 0.0,
                            "intents_artifact_stale": False,
                            "intents_stale_after_seconds": 3600.0,
                        },
                    },
                    intents={"generated_at_utc": intents_at},
                )

                self.assertFalse(freshness["fresh"])

    def test_no_live_ready_next_action_does_not_mislabel_family_as_exact_blocker(self) -> None:
        action = _no_live_ready_next_action(
            coverage_refresh=None,
            intent_evidence_fresh=True,
            live_readiness_breakdown={
                "nearest_all_non_live_ready_candidates": [
                    {
                        "lane_id": "legacy-lane",
                        "blocker_count": 1,
                        "blockers": [
                            {
                                "family": "risk_geometry",
                                "message": "legacy unstructured blocker",
                            }
                        ],
                    }
                ]
            },
        )

        self.assertIn("inspect top live blockers", action)
        self.assertNotIn("risk_geometry", action)

    def test_no_live_ready_next_action_does_not_hide_truncated_blockers(self) -> None:
        action = _no_live_ready_next_action(
            coverage_refresh=None,
            intent_evidence_fresh=True,
            live_readiness_breakdown={
                "nearest_all_non_live_ready_candidates": [
                    {
                        "lane_id": "many-blockers",
                        "blocker_count": 7,
                        "blockers": [
                            {"code": f"BLOCKER_{index}"} for index in range(6)
                        ],
                        "authoritative_live_blocker_codes": [
                            f"BLOCKER_{index}" for index in range(6)
                        ],
                        "authoritative_live_blocker_codes_complete": False,
                    }
                ]
            },
        )

        self.assertIn("inspect top live blockers", action)
        self.assertNotIn("BLOCKER_0", action)

    def test_incomplete_authoritative_blocker_codes_keep_action_generic(self) -> None:
        breakdown = _intent_live_readiness_family_breakdown(
            {
                "results": [
                    {
                        "lane_id": "incomplete-codes",
                        "status": "DRY_RUN_BLOCKED",
                        "intent": {},
                        "risk_issues": [
                            {"code": "BLOCK_A", "message": "a", "severity": "BLOCK"},
                            {"code": "BLOCK_B", "message": "b", "severity": "BLOCK"},
                        ],
                        "live_strategy_issues": [],
                        "live_blockers": ["a", "b"],
                        "live_blocker_codes": ["BLOCK_A"],
                    }
                ]
            }
        )

        candidate = breakdown["nearest_all_non_live_ready_candidates"][0]
        self.assertFalse(candidate["authoritative_live_blocker_codes_complete"])
        action = _no_live_ready_next_action(
            coverage_refresh=None,
            intent_evidence_fresh=True,
            live_readiness_breakdown=breakdown,
        )
        self.assertIn("inspect top live blockers", action)
        self.assertNotIn("BLOCK_A", action)

    def test_unmapped_legacy_live_blocker_keeps_action_generic(self) -> None:
        breakdown = _intent_live_readiness_family_breakdown(
            {
                "results": [
                    {
                        "lane_id": "legacy-extra",
                        "status": "DRY_RUN_BLOCKED",
                        "intent": {},
                        "risk_issues": [
                            {
                                "code": "EXACT",
                                "message": "exact message",
                                "severity": "BLOCK",
                            }
                        ],
                        "live_strategy_issues": [],
                        "live_blockers": ["exact message", "legacy extra blocker"],
                        "live_blocker_codes": ["EXACT"],
                    }
                ]
            }
        )

        candidate = breakdown["nearest_all_non_live_ready_candidates"][0]
        self.assertFalse(candidate["authoritative_live_blocker_codes_complete"])

    def test_extra_or_duplicate_authoritative_blocker_evidence_stays_generic(
        self,
    ) -> None:
        variants = (
            {
                "live_blockers": ["exact message"],
                "live_blocker_codes": ["EXACT", "FAKE"],
            },
            {
                "live_blockers": ["exact message", "exact message"],
                "live_blocker_codes": ["EXACT"],
            },
            {
                "live_blockers": ["exact message"],
                "live_blocker_codes": ["EXACT"],
                "strategy_issues": 42,
            },
        )
        for variant in variants:
            with self.subTest(variant=variant):
                breakdown = _intent_live_readiness_family_breakdown(
                    {
                        "results": [
                            {
                                "lane_id": "untrusted-proof",
                                "status": "DRY_RUN_BLOCKED",
                                "intent": {},
                                "risk_issues": [
                                    {
                                        "code": "EXACT",
                                        "message": "exact message",
                                        "severity": "BLOCK",
                                    }
                                ],
                                "live_strategy_issues": [],
                                **variant,
                            }
                        ]
                    }
                )

                candidate = breakdown["nearest_all_non_live_ready_candidates"][0]
                self.assertFalse(
                    candidate["authoritative_live_blocker_codes_complete"]
                )
                action = _no_live_ready_next_action(
                    coverage_refresh=None,
                    intent_evidence_fresh=True,
                    live_readiness_breakdown=breakdown,
                )
                self.assertIn("inspect top live blockers", action)
                self.assertNotIn("EXACT", action)

    def test_blocks_missing_memory_projection_and_entry_thesis_holes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=True, write_memory=False, projection_expired=True)

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

            codes = {item["code"] for item in payload["findings"]}
            self.assertEqual(summary.status, STATUS_BLOCKED)
            self.assertGreaterEqual(summary.p0_findings, 3)
            self.assertEqual(payload["findings_count"], summary.findings)
            self.assertEqual(payload["p0_findings"], summary.p0_findings)
            self.assertEqual(payload["p1_findings"], summary.p1_findings)
            self.assertEqual(payload["p2_findings"], summary.p2_findings)
            self.assertIn("MEMORY_HEALTH_UNREADABLE", codes)
            self.assertIn("PROJECTION_LEDGER_EXPIRED_PENDING", codes)
            self.assertIn("ENTRY_THESIS_MISSING_FOR_OPEN_TRADES", codes)
            with sqlite3.connect(files["history_db"]) as conn:
                run_count = conn.execute("SELECT COUNT(*) FROM self_improvement_audit_runs").fetchone()[0]
                finding_count = conn.execute("SELECT COUNT(*) FROM self_improvement_findings").fetchone()[0]
                self.assertEqual(run_count, 1)
                self.assertEqual(finding_count, summary.findings)

    def test_missing_execution_timing_audit_remains_visible_as_p1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False)
            files["execution_timing"].unlink()

            _run(files)
            payload = json.loads(files["output"].read_text())
            findings = {
                item["code"]: item for item in payload["findings"]
            }

            self.assertEqual(
                findings["EXECUTION_TIMING_AUDIT_UNREADABLE"]["priority"],
                "P1",
            )

    def test_stale_memory_health_routes_to_refresh_before_old_blocker_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, live_ready_market_rr=1.4, closed_pls=(100.0, 80.0, -50.0))
            files["memory"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": (_NOW - timedelta(minutes=5)).isoformat(),
                        "status": "MEMORY_HEALTH_BLOCKED",
                        "issues": [{"code": "SHORT_FORECAST_PAIR_STALE", "severity": "BLOCK"}],
                        "blockers": ["old forecast row predates old broker snapshot"],
                        "warnings": [],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertIn("MEMORY_HEALTH_STALE", codes)
        self.assertNotIn("MEMORY_HEALTH_BLOCKED", codes)
        evidence = codes["MEMORY_HEALTH_STALE"]["evidence"]
        self.assertEqual(evidence["memory_health_generated_at_utc"], (_NOW - timedelta(minutes=5)).isoformat())
        self.assertEqual(evidence["stale_against"][0]["label"], "broker_snapshot")

    def test_learning_audit_quarantine_is_p1_not_global_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, 50.0),
            )
            files["learning"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "status": "LEARNING_AUDIT_BLOCKED",
                        "blockers": [
                            "risk-increasing learning influence is active while recent effect window is negative"
                        ],
                        "warnings": [],
                        "checks": [
                            {
                                "check_name": "learning_influence_recent_outcome",
                                "status": "BLOCK",
                                "severity": "BLOCK",
                                "message": (
                                    "risk-increasing learning influence is active while recent effect "
                                    "window is negative"
                                ),
                            }
                        ],
                        "learning_influence": {
                            "influenced_lanes": 1,
                            "risk_increasing_lanes": 1,
                            "total_learning_score_delta": 8.0,
                            "lanes": [
                                {
                                    "lane_id": "trend_trader:AUD_USD:LONG:TREND_CONTINUATION",
                                    "learning_influences": ["ai_backtest_research_positive_edge"],
                                    "learning_score_delta": 8.0,
                                }
                            ],
                        },
                        "effect_metrics": {"closed_trades": 30, "net_jpy": -100.0, "profit_factor": 0.9},
                        "min_effect_sample": 3,
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertNotIn("LEARNING_AUDIT_BLOCKED", codes)
        self.assertEqual(codes["LEARNING_AUDIT_INFLUENCED_LANES_QUARANTINED"]["priority"], "P1")

    def test_memory_health_audited_snapshot_time_prevents_false_stale_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, closed_pls=(100.0, 80.0, -50.0))
            snapshot_ts = _NOW + timedelta(minutes=2)
            intents_ts = _NOW + timedelta(minutes=1)

            snapshot = json.loads(files["snapshot"].read_text())
            snapshot["fetched_at_utc"] = snapshot_ts.isoformat()
            snapshot["account"]["fetched_at_utc"] = snapshot_ts.isoformat()
            files["snapshot"].write_text(json.dumps(snapshot))

            intents = json.loads(files["intents"].read_text())
            intents["generated_at_utc"] = intents_ts.isoformat()
            files["intents"].write_text(json.dumps(intents))

            memory = json.loads(files["memory"].read_text())
            memory["generated_at_utc"] = _NOW.isoformat()
            memory["metrics"] = {
                "runtime": {
                    "snapshot_fetched_at_utc": snapshot_ts.isoformat(),
                    "order_intents_generated_at_utc": intents_ts.isoformat(),
                }
            }
            files["memory"].write_text(json.dumps(memory))

            _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertNotIn("MEMORY_HEALTH_STALE", codes)

    def test_memory_health_stale_when_audited_capture_economics_predates_current_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, live_ready_market_rr=1.4, closed_pls=(100.0, 80.0, -50.0))
            current_capture_ts = _NOW + timedelta(minutes=1)
            stale_capture_ts = _NOW - timedelta(minutes=10)

            intents = json.loads(files["intents"].read_text())
            intents["generated_at_utc"] = _NOW.isoformat()
            files["intents"].write_text(json.dumps(intents))

            capture = json.loads(files["capture_economics"].read_text())
            capture["generated_at_utc"] = current_capture_ts.isoformat()
            files["capture_economics"].write_text(json.dumps(capture))

            files["memory"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "status": "MEMORY_HEALTH_BLOCKED",
                        "metrics": {
                            "runtime": {
                                "snapshot_fetched_at_utc": _NOW.isoformat(),
                                "order_intents_generated_at_utc": _NOW.isoformat(),
                                "capture_economics_generated_at_utc": stale_capture_ts.isoformat(),
                            }
                        },
                        "issues": [{"code": "SHORT_ORDER_INTENTS_MEMORY_BLOCKERS", "severity": "BLOCK"}],
                        "blockers": ["old capture economics capped the current short lane"],
                        "warnings": [],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertIn("MEMORY_HEALTH_STALE", codes)
        self.assertNotIn("MEMORY_HEALTH_BLOCKED", codes)
        evidence = codes["MEMORY_HEALTH_STALE"]["evidence"]
        self.assertEqual(evidence["stale_against"][0]["label"], "capture_economics")
        self.assertEqual(evidence["stale_against"][0]["timestamp_utc"], current_capture_ts.isoformat())
        self.assertEqual(evidence["stale_against"][0]["audited_timestamp_utc"], stale_capture_ts.isoformat())

    def test_external_live_lock_defers_mid_refresh_memory_stale_judgment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, live_ready_market_rr=1.4, closed_pls=(100.0, 80.0, -50.0))
            files["memory"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": (_NOW - timedelta(minutes=5)).isoformat(),
                        "status": "MEMORY_HEALTH_PASS",
                        "issues": [],
                        "blockers": [],
                        "warnings": [],
                    }
                )
            )
            lock_dir = root / ".quant_rabbit_live.lock"
            lock_dir.mkdir()
            (lock_dir / "pid").write_text(str(os.getpid()), encoding="utf-8")
            (lock_dir / "command").write_text("cycle-refresh", encoding="utf-8")
            (lock_dir / "started_at_utc").write_text(
                (_NOW - timedelta(minutes=3)).isoformat(),
                encoding="utf-8",
            )

            with mock.patch.dict(
                os.environ,
                {"QR_AUTOTRADE_LOCK_DIR": str(lock_dir), "QR_AUTOTRADE_LOCK_HELD": ""},
                clear=False,
            ):
                summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertIn("LIVE_RUNTIME_UPDATE_IN_PROGRESS", codes)
        self.assertNotIn("MEMORY_HEALTH_STALE", codes)
        self.assertNotIn("TARGET_OPEN_NO_LIVE_READY_LANES", codes)
        evidence = codes["LIVE_RUNTIME_UPDATE_IN_PROGRESS"]["evidence"]
        self.assertEqual(evidence["pid"], os.getpid())
        self.assertEqual(evidence["command"], "cycle-refresh")
        self.assertEqual(evidence["started_at_utc"], (_NOW - timedelta(minutes=3)).isoformat())
        self.assertGreaterEqual(evidence["lock_age_seconds"], 0.0)

    def test_external_live_lock_defers_mid_refresh_sidecar_stale_judgment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=True, closed_pls=(100.0, 80.0, -50.0))
            for key, list_key in (
                ("position_management", "positions"),
                ("thesis_evolution", "evolutions"),
                ("position_thesis", "assessments"),
                ("forecast_persistence", "verdicts"),
            ):
                files[key].write_text(
                    json.dumps(
                        {
                            "generated_at_utc": (_NOW - timedelta(minutes=5)).isoformat(),
                            "snapshot_fetched_at_utc": (_NOW - timedelta(minutes=5)).isoformat(),
                            list_key: [],
                        }
                    )
                )
            lock_dir = root / ".quant_rabbit_live.lock"
            lock_dir.mkdir()
            (lock_dir / "pid").write_text(str(os.getpid()), encoding="utf-8")
            (lock_dir / "command").write_text("cycle-refresh", encoding="utf-8")

            with mock.patch.dict(
                os.environ,
                {"QR_AUTOTRADE_LOCK_DIR": str(lock_dir), "QR_AUTOTRADE_LOCK_HELD": ""},
                clear=False,
            ):
                summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertIn("LIVE_RUNTIME_UPDATE_IN_PROGRESS", codes)
        self.assertNotIn("POSITION_MANAGEMENT_STALE", codes)
        self.assertNotIn("THESIS_EVOLUTION_STALE", codes)
        self.assertNotIn("POSITION_THESIS_STALE", codes)
        self.assertNotIn("FORECAST_PERSISTENCE_STALE", codes)

    def test_position_guardian_lock_does_not_defer_self_improvement_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, closed_pls=(100.0, 80.0, -50.0))
            lock_dir = root / ".quant_rabbit_live.lock"
            lock_dir.mkdir()
            (lock_dir / "pid").write_text(str(os.getpid()), encoding="utf-8")
            (lock_dir / "command").write_text("run-position-guardian-live", encoding="utf-8")
            (lock_dir / "started_at_utc").write_text(
                (_NOW - timedelta(seconds=2)).isoformat(),
                encoding="utf-8",
            )

            with mock.patch.dict(
                os.environ,
                {"QR_AUTOTRADE_LOCK_DIR": str(lock_dir), "QR_AUTOTRADE_LOCK_HELD": ""},
                clear=False,
            ):
                summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertNotIn("LIVE_RUNTIME_UPDATE_IN_PROGRESS", codes)
        self.assertIn("TARGET_OPEN_NO_LIVE_READY_LANES", codes)

    def test_external_live_lock_still_surfaces_coverage_perspective_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, live_ready_market_rr=1.4, closed_pls=(100.0, 80.0, -50.0))
            files["coverage"].write_text(
                json.dumps(
                    {
                        "artifact_diagnostics": {},
                        "perspective_alignment_diagnostics": {
                            "status": "RANGE_METHOD_MISMATCH_REPAIR_REQUIRED",
                            "pair_direction_groups": 2,
                            "range_forecast_method_mismatch_groups": 1,
                            "range_forecast_method_mismatch_lanes": 3,
                            "range_forecast_method_mismatch_top": [
                                {
                                    "pair": "AUD_JPY",
                                    "direction": "SHORT",
                                    "method_mismatch_lanes": 3,
                                    "range_rotation_lanes": 0,
                                    "range_rotation_live_ready_lanes": 0,
                                    "top_live_blocker_codes": [
                                        {"code": "RANGE_FORECAST_REQUIRES_RANGE_ROTATION", "count": 3}
                                    ],
                                }
                            ],
                        },
                    }
                )
            )
            lock_dir = root / ".quant_rabbit_live.lock"
            lock_dir.mkdir()
            (lock_dir / "pid").write_text(str(os.getpid()), encoding="utf-8")

            with mock.patch.dict(
                os.environ,
                {"QR_AUTOTRADE_LOCK_DIR": str(lock_dir), "QR_AUTOTRADE_LOCK_HELD": ""},
                clear=False,
            ):
                summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertIn("LIVE_RUNTIME_UPDATE_IN_PROGRESS", codes)
        self.assertNotIn("TARGET_OPEN_NO_LIVE_READY_LANES", codes)
        finding = codes["RANGE_FORECAST_METHOD_MISMATCH_REPAIR_REQUIRED"]
        self.assertEqual(finding["priority"], "P1")
        perspective = finding["evidence"]["perspective_alignment_diagnostics"]
        self.assertEqual(perspective["range_forecast_method_mismatch_lanes"], 3)
        self.assertEqual(perspective["range_forecast_method_mismatch_top"][0]["pair"], "AUD_JPY")

    def test_wrapper_owned_live_lock_still_allows_memory_stale_judgment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, live_ready_market_rr=1.4, closed_pls=(100.0, 80.0, -50.0))
            files["memory"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": (_NOW - timedelta(minutes=5)).isoformat(),
                        "status": "MEMORY_HEALTH_PASS",
                        "issues": [],
                        "blockers": [],
                        "warnings": [],
                    }
                )
            )
            lock_dir = root / ".quant_rabbit_live.lock"
            lock_dir.mkdir()
            (lock_dir / "pid").write_text(str(os.getpid()), encoding="utf-8")

            with mock.patch.dict(
                os.environ,
                {"QR_AUTOTRADE_LOCK_DIR": str(lock_dir), "QR_AUTOTRADE_LOCK_HELD": "1"},
                clear=False,
            ):
                summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertNotIn("LIVE_RUNTIME_UPDATE_IN_PROGRESS", codes)
        self.assertIn("MEMORY_HEALTH_STALE", codes)

    def test_missing_entry_thesis_ledger_without_open_trades_is_not_a_finding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["entry_thesis"].unlink()

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertNotIn("ENTRY_THESIS_LEDGER_UNREADABLE", codes)
        self.assertEqual(summary.p0_findings, 0)

    def test_history_dedupes_identical_retry_inside_operational_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=True, write_memory=False)

            first = _run(files, now=_NOW)
            second = _run(files, now=_NOW + timedelta(seconds=30))

            self.assertEqual(first.status, STATUS_BLOCKED)
            self.assertEqual(second.status, STATUS_BLOCKED)
            with sqlite3.connect(files["history_db"]) as conn:
                run_count = conn.execute("SELECT COUNT(*) FROM self_improvement_audit_runs").fetchone()[0]
                finding_count = conn.execute("SELECT COUNT(*) FROM self_improvement_findings").fetchone()[0]
            self.assertEqual(run_count, 1)
            self.assertEqual(finding_count, first.findings)

    def test_history_dedupes_stale_gpt_retry_ignoring_streak_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "generated_at_utc": (_NOW - timedelta(minutes=1)).isoformat(),
                        "decision": {"action": "WAIT"},
                        "verification_issues": [],
                    }
                )
            )

            first = _run(files, now=_NOW)
            _run(files, now=_NOW + timedelta(seconds=30))

            self.assertEqual(first.status, STATUS_BLOCKED)
            with sqlite3.connect(files["history_db"]) as conn:
                run_count = conn.execute("SELECT COUNT(*) FROM self_improvement_audit_runs").fetchone()[0]
                finding_count = conn.execute("SELECT COUNT(*) FROM self_improvement_findings").fetchone()[0]
                stale_streaks = [
                    json.loads(row[0]).get("current_streak")
                    for row in conn.execute(
                        """
                        SELECT evidence_json
                        FROM self_improvement_findings
                        WHERE code = 'LATEST_GPT_DECISION_STALE'
                        """
                    )
                ]
            self.assertEqual(run_count, 1)
            self.assertEqual(finding_count, first.findings)
            self.assertEqual(stale_streaks, [1])

    def test_history_dedupes_repeated_loop_retry_ignoring_nested_streak_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                closed_pls=(100.0, 80.0, 50.0),
            )

            _run(files, now=_NOW)
            _run(files, now=_NOW + timedelta(minutes=3))
            third = _run(files, now=_NOW + timedelta(minutes=6))
            retry = _run(files, now=_NOW + timedelta(minutes=6, seconds=30))

            self.assertEqual(third.status, STATUS_BLOCKED)
            self.assertEqual(retry.status, STATUS_BLOCKED)
            with sqlite3.connect(files["history_db"]) as conn:
                run_count = conn.execute("SELECT COUNT(*) FROM self_improvement_audit_runs").fetchone()[0]
                loop_streaks = [
                    json.loads(row[0]).get("current_streak")
                    for row in conn.execute(
                        """
                        SELECT evidence_json
                        FROM self_improvement_findings
                        WHERE code = 'REPEATED_SELF_IMPROVEMENT_LOOP'
                        ORDER BY ts_utc
                        """
                    )
                ]
            self.assertEqual(run_count, 3)
            self.assertEqual(loop_streaks, [3])

    def test_history_dedupes_live_lock_retry_ignoring_lock_age(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            lock_dir = root / ".quant_rabbit_live.lock"
            lock_dir.mkdir()
            (lock_dir / "pid").write_text(str(os.getpid()), encoding="utf-8")
            (lock_dir / "command").write_text("cycle-refresh", encoding="utf-8")
            (lock_dir / "started_at_utc").write_text(
                (_NOW - timedelta(minutes=3)).isoformat(),
                encoding="utf-8",
            )

            with mock.patch.dict(
                os.environ,
                {"QR_AUTOTRADE_LOCK_DIR": str(lock_dir), "QR_AUTOTRADE_LOCK_HELD": ""},
                clear=False,
            ):
                first = _run(files, now=_NOW)
                second = _run(files, now=_NOW + timedelta(seconds=30))

            self.assertEqual(first.status, STATUS_BLOCKED)
            self.assertEqual(second.status, STATUS_BLOCKED)
            with sqlite3.connect(files["history_db"]) as conn:
                run_count = conn.execute("SELECT COUNT(*) FROM self_improvement_audit_runs").fetchone()[0]
                lock_ages = [
                    json.loads(row[0]).get("lock_age_seconds")
                    for row in conn.execute(
                        """
                        SELECT evidence_json
                        FROM self_improvement_findings
                        WHERE code = 'LIVE_RUNTIME_UPDATE_IN_PROGRESS'
                        """
                    )
                ]
            self.assertEqual(run_count, 1)
            self.assertEqual(len(lock_ages), 1)

    def test_history_dedupes_live_lock_retry_while_downstream_artifacts_move(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            lock_dir = root / ".quant_rabbit_live.lock"
            lock_dir.mkdir()
            (lock_dir / "pid").write_text(str(os.getpid()), encoding="utf-8")
            (lock_dir / "command").write_text("run-autotrade-live", encoding="utf-8")
            (lock_dir / "started_at_utc").write_text(
                (_NOW - timedelta(minutes=3)).isoformat(),
                encoding="utf-8",
            )

            with mock.patch.dict(
                os.environ,
                {"QR_AUTOTRADE_LOCK_DIR": str(lock_dir), "QR_AUTOTRADE_LOCK_HELD": ""},
                clear=False,
            ):
                first = _run(files, now=_NOW)
                files["forecast_history"].write_text(
                    "\n".join(
                        json.dumps(
                            {
                                "timestamp_utc": _NOW.isoformat(),
                                "pair": "EUR_USD",
                                "direction": "UP",
                                "confidence": 0.62,
                            }
                        )
                        for _idx in range(3)
                    )
                    + "\n"
                )
                second = _run(files, now=_NOW + timedelta(seconds=30))

            self.assertEqual(first.status, STATUS_BLOCKED)
            self.assertEqual(second.status, STATUS_BLOCKED)
            payload = json.loads(files["output"].read_text())
            self.assertIn("FORECAST_HISTORY_LEGACY_PHANTOM_CLUSTERS", {item["code"] for item in payload["findings"]})
            with sqlite3.connect(files["history_db"]) as conn:
                run_count = conn.execute("SELECT COUNT(*) FROM self_improvement_audit_runs").fetchone()[0]
                lock_count = conn.execute(
                    "SELECT COUNT(*) FROM self_improvement_findings WHERE code = 'LIVE_RUNTIME_UPDATE_IN_PROGRESS'"
                ).fetchone()[0]
            self.assertEqual(run_count, 1)
            self.assertEqual(lock_count, 1)

    def test_pending_entry_lifecycle_flags_cancel_before_fill_churn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, 50.0),
            )
            _write_pending_cancel_churn_ledger(files["execution_db"])

            summary = _run(files)
            payload = json.loads(files["output"].read_text())
            report = files["report"].read_text()

        codes = {item["code"]: item for item in payload["findings"]}
        lifecycle = payload["execution_quality"]["pending_entry_lifecycle"]
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertIn("PENDING_ENTRY_FILL_RATE_WEAK", codes)
        self.assertEqual(codes["PENDING_ENTRY_FILL_RATE_WEAK"]["priority"], "P1")
        self.assertEqual(lifecycle["accepted_entry_orders"], 3)
        self.assertEqual(lifecycle["filled_entry_orders"], 0)
        self.assertEqual(lifecycle["canceled_before_fill_orders"], 3)
        self.assertEqual(lifecycle["canceled_before_fill_replaced_orders"], 0)
        self.assertEqual(lifecycle["canceled_before_fill_orphan_orders"], 3)
        self.assertEqual(lifecycle["cancel_before_fill_rate"], 1.0)
        self.assertEqual(lifecycle["cancel_replacement_rate"], 0.0)
        self.assertIn("Pending entry lifecycle", report)

    def test_pending_entry_lifecycle_flags_high_cancel_rate_even_with_fills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, 50.0),
            )
            _write_pending_mixed_cancel_churn_ledger(files["execution_db"])
            files["coverage"].write_text(
                json.dumps(
                    {
                        "status": "COVERAGE_OK",
                        "remaining_target_jpy": 1000.0,
                        "live_ready_reward_jpy": 1000.0,
                        "artifact_diagnostics": {},
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        lifecycle = payload["execution_quality"]["pending_entry_lifecycle"]
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertIn("PENDING_ENTRY_CANCEL_RATE_HIGH", codes)
        self.assertEqual(codes["PENDING_ENTRY_CANCEL_RATE_HIGH"]["priority"], "P1")
        self.assertEqual(lifecycle["accepted_entry_orders"], 5)
        self.assertEqual(lifecycle["filled_entry_orders"], 2)
        self.assertEqual(lifecycle["canceled_before_fill_orders"], 3)
        self.assertEqual(lifecycle["canceled_before_fill_replaced_orders"], 0)
        self.assertEqual(lifecycle["canceled_before_fill_orphan_orders"], 3)
        self.assertAlmostEqual(lifecycle["cancel_before_fill_rate"], 0.6)
        self.assertEqual(lifecycle["cancel_replacement_rate"], 0.0)
        evidence = codes["PENDING_ENTRY_CANCEL_RATE_HIGH"]["evidence"]
        orphan_group_keys = {
            (item["pair"], item["side"], item["method"])
            for item in evidence["canceled_before_fill_orphan_groups"]
        }
        self.assertIn(("EUR_CAD", "LONG", "RANGE_ROTATION"), orphan_group_keys)
        self.assertIn(("NZD_CHF", "LONG", "RANGE_ROTATION"), orphan_group_keys)
        self.assertIn(("NZD_JPY", "LONG", "RANGE_ROTATION"), orphan_group_keys)
        self.assertEqual(payload["root_cause_focus"]["primary"]["family"], "EXECUTION_LIFECYCLE")

    def test_pending_entry_lifecycle_includes_cancel_timing_regret(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, 50.0),
            )
            _write_pending_mixed_cancel_churn_ledger(files["execution_db"])
            files["coverage"].write_text(
                json.dumps(
                    {
                        "status": "COVERAGE_OK",
                        "remaining_target_jpy": 1000.0,
                        "live_ready_reward_jpy": 1000.0,
                        "artifact_diagnostics": {},
                    }
                )
            )
            files["execution_timing"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "status": "OK",
                        "fetch_errors": [],
                        "precision": {
                            TP_PROGRESS_REPAIR_REPLAY_FIELD: TP_PROGRESS_REPAIR_REPLAY_CONTRACT,
                            "price_basis": "OANDA_M1_BID_ASK_CANDLES",
                            "granularity": "M1",
                        },
                        "window": _replay_window(_NOW),
                        "summary": {
                            "canceled_orders_audited": 3,
                            "canceled_entry_touched_after_cancel": 2,
                            "canceled_entry_touched_after_cancel_rate": 0.667,
                            "canceled_positive_after_cancel_entry": 2,
                            "canceled_tp_touched_after_cancel": 1,
                            "canceled_estimated_missed_mfe_jpy": 815.5,
                            "loss_closes_audited": 0,
                            "loss_closes_profit_capture_missed": 0,
                            "loss_closes_repair_replay_triggered": 0,
                            "tp_progress_repair_live_evidence_boundary_utc": (
                                TP_PROGRESS_REPAIR_LIVE_EVIDENCE_BOUNDARY_UTC
                            ),
                            "tp_progress_repair_live_evidence_status": (
                                "WAITING_FOR_POST_REPAIR_SAMPLE"
                            ),
                            "pre_repair_historical_loss_closes_audited": 0,
                            "pre_repair_historical_loss_closes_profit_capture_missed": 0,
                            "pre_repair_historical_loss_closes_repair_replay_triggered": 0,
                            "post_repair_live_evidence_loss_closes_audited": 0,
                            "post_repair_live_evidence_loss_closes_profit_capture_missed": 0,
                            "post_repair_live_evidence_loss_closes_repair_replay_triggered": 0,
                        },
                        "loss_close_regrets": [],
                        "canceled_order_regrets": [
                            {
                                "order_id": "O-mixed-3",
                                "lane_id": "range_trader:EUR_CAD:LONG:RANGE_ROTATION",
                                "pair": "EUR_CAD",
                                "side": "LONG",
                                "order_type": "LIMIT_ORDER",
                                "entry_touched_after_cancel": True,
                                "tp_touched_after_cancel": True,
                                "sl_touched_after_cancel": False,
                                "entry_touch_after_cancel_minutes": 68.5,
                                "mfe_pips_after_cancel_entry": 12.4,
                                "estimated_missed_mfe_jpy": 815.5,
                            }
                        ],
                        "canceled_order_regret_by_shape": {
                            "basis": "canceled pending orders grouped by pair|side|method|order_type",
                            "total_shapes": 1,
                            "items": [
                                {
                                    "evidence_ref": "timing:canceled_shape:EUR_CAD:LONG:RANGE_ROTATION:LIMIT_ORDER",
                                    "pair": "EUR_CAD",
                                    "side": "LONG",
                                    "method": "RANGE_ROTATION",
                                    "order_type": "LIMIT_ORDER",
                                    "priority_class": "PRESERVE_PENDING_THESIS_TP_TOUCHED",
                                    "next_action": "review cancel rule/TTL before canceling this pending shape",
                                    "orders": 3,
                                    "entry_touched_after_cancel": 2,
                                    "entry_touch_after_cancel_rate": 0.667,
                                    "positive_after_cancel_entry": 2,
                                    "positive_after_cancel_entry_rate": 0.667,
                                    "tp_touched_after_cancel": 1,
                                    "tp_touched_after_cancel_rate": 0.333,
                                    "estimated_missed_mfe_jpy": 815.5,
                                }
                            ],
                        },
                    }
                )
            )

            _run(files)
            payload = json.loads(files["output"].read_text())
            report = files["report"].read_text()

        codes = {item["code"]: item for item in payload["findings"]}
        timing_regret = codes["PENDING_ENTRY_CANCEL_RATE_HIGH"]["evidence"]["timing_regret"]
        self.assertEqual(timing_regret["canceled_entry_touched_after_cancel"], 2)
        self.assertEqual(timing_regret["canceled_tp_touched_after_cancel"], 1)
        self.assertAlmostEqual(timing_regret["canceled_estimated_missed_mfe_jpy"], 815.5)
        self.assertEqual(timing_regret["top_regretted_cancels"][0]["order_id"], "O-mixed-3")
        self.assertEqual(
            timing_regret["top_regretted_shapes"][0]["evidence_ref"],
            "timing:canceled_shape:EUR_CAD:LONG:RANGE_ROTATION:LIMIT_ORDER",
        )
        self.assertEqual(
            timing_regret["top_regretted_shapes"][0]["priority_class"],
            "PRESERVE_PENDING_THESIS_TP_TOUCHED",
        )
        self.assertIn("Pending cancel timing regret", report)
        self.assertIn("timing:canceled_shape:EUR_CAD:LONG:RANGE_ROTATION:LIMIT_ORDER", report)

    def test_pending_entry_lifecycle_distinguishes_replaced_from_orphan_cancels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, 50.0),
            )
            _write_pending_replacement_churn_ledger(files["execution_db"])

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        lifecycle = payload["execution_quality"]["pending_entry_lifecycle"]
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertIn("PENDING_ENTRY_FILL_RATE_WEAK", codes)
        self.assertEqual(lifecycle["accepted_entry_orders"], 4)
        self.assertEqual(lifecycle["canceled_before_fill_orders"], 3)
        self.assertEqual(lifecycle["canceled_before_fill_replaced_orders"], 1)
        self.assertEqual(lifecycle["canceled_before_fill_orphan_orders"], 2)
        self.assertAlmostEqual(lifecycle["cancel_replacement_rate"], 1 / 3)
        samples = codes["PENDING_ENTRY_FILL_RATE_WEAK"]["evidence"]["samples"]
        replaced = next(item for item in samples if item["order_id"] == "replace-source")
        self.assertEqual(replaced["replaced_with_order_id"], "replace-next")
        self.assertAlmostEqual(replaced["replacement_after_min"], 5.0)
        groups = lifecycle["canceled_before_fill_orphan_groups"]
        self.assertEqual(groups[0]["pair"], "CAD_CHF")
        self.assertEqual(groups[0]["side"], "LONG")
        self.assertEqual(groups[0]["method"], "RANGE_ROTATION")
        self.assertEqual(groups[0]["count"], 1)

    def test_current_pending_reconcile_flags_cap_candidate_and_advice_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                closed_pls=(100.0, 80.0, 50.0),
            )
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot["home_conversions"] = {"USD": 150.0}
            snapshot["orders"] = [
                {
                    "order_id": "risk-pending",
                    "pair": "EUR_USD",
                    "order_type": "STOP",
                    "state": "PENDING",
                    "units": 2000,
                    "price": 1.171,
                    "owner": "trader",
                    "trade_id": None,
                    "raw": {
                        "createTime": (_NOW - timedelta(minutes=40)).isoformat(),
                        "clientExtensions": {
                            "comment": (
                                "qr-vnext lane=trend_trader:EUR_USD:LONG:TREND_CONTINUATION "
                                "desk=trend_trader"
                            ),
                            "tag": "trader",
                        },
                        "stopLossOnFill": {"price": "1.1600"},
                    },
                }
            ]
            files["snapshot"].write_text(json.dumps(snapshot))
            files["target"].write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_target_jpy": 1000.0,
                        "per_trade_risk_budget_jpy": 100.0,
                    }
                )
            )
            files["intents"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "results": [
                            {
                                "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                                "status": "DRY_RUN_BLOCKED",
                                "intent": {
                                    "pair": "EUR_USD",
                                    "side": "LONG",
                                    "order_type": "STOP-ENTRY",
                                },
                                "risk_issues": [
                                    {
                                        "code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
                                        "severity": "BLOCK",
                                    }
                                ],
                                "live_blockers": ["forecast confidence below entry grade"],
                            }
                        ],
                    }
                )
            )
            files["attack_advice"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "status": "ATTACK_ADVICE_READY",
                        "recommended_now_lane_ids": ["trend_trader:EUR_USD:SHORT:TREND_CONTINUATION"],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())
            report = files["report"].read_text()

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertIn("PENDING_ENTRY_CANCEL_REVIEW_REQUIRED", codes)
        review = payload["execution_quality"]["pending_entry_reconcile"]
        self.assertEqual(review["cancel_review_order_ids"], ["risk-pending"])
        order_review = review["orders"][0]
        reason_codes = {item["code"] for item in order_review["review_reasons"]}
        monitor_codes = {item["code"] for item in order_review["monitor_reasons"]}
        self.assertIn("PENDING_ATTACHED_SL_RISK_EXCEEDS_CAP", reason_codes)
        self.assertIn("PENDING_CURRENT_CANDIDATE_NOT_LIVE_READY", monitor_codes)
        self.assertIn("PENDING_ATTACK_ADVICE_NOT_CURRENT", monitor_codes)
        self.assertEqual(order_review["reconcile_status"], "CANCEL_REVIEW_REQUIRED")
        self.assertGreater(order_review["attached_sl_risk_jpy"], 100.0)
        self.assertIn("Pending entry reconcile", report)

    def test_current_pending_reconcile_preserves_transiently_blocked_broker_thesis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                closed_pls=(100.0, 80.0, 50.0),
            )
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot["home_conversions"] = {"USD": 150.0}
            snapshot["orders"] = [
                {
                    "order_id": "safe-pending",
                    "pair": "EUR_USD",
                    "order_type": "STOP",
                    "state": "PENDING",
                    "units": 2000,
                    "price": 1.171,
                    "owner": "trader",
                    "trade_id": None,
                    "raw": {
                        "createTime": (_NOW - timedelta(minutes=40)).isoformat(),
                        "clientExtensions": {
                            "comment": (
                                "qr-vnext lane=trend_trader:EUR_USD:LONG:TREND_CONTINUATION "
                                "desk=trend_trader"
                            ),
                            "tag": "trader",
                        },
                        "stopLossOnFill": {"price": "1.1705"},
                    },
                }
            ]
            files["snapshot"].write_text(json.dumps(snapshot))
            files["target"].write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_target_jpy": 1000.0,
                        "per_trade_risk_budget_jpy": 1000.0,
                        "remaining_risk_budget_jpy": 10_000.0,
                    }
                )
            )
            files["intents"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "results": [
                            {
                                "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                                "status": "DRY_RUN_BLOCKED",
                                "intent": {
                                    "pair": "EUR_USD",
                                    "side": "LONG",
                                    "order_type": "STOP-ENTRY",
                                },
                                "risk_issues": [
                                    {
                                        "code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
                                        "severity": "BLOCK",
                                    }
                                ],
                                "live_blockers": [
                                    "temporary forecast confidence below entry grade"
                                ],
                            }
                        ],
                    }
                )
            )
            files["attack_advice"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "status": "ATTACK_ADVICE_READY",
                        "recommended_now_lane_ids": [
                            "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION"
                        ],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())
            report = files["report"].read_text()

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertNotIn("PENDING_ENTRY_CANCEL_REVIEW_REQUIRED", codes)
        review = payload["execution_quality"]["pending_entry_reconcile"]
        self.assertEqual(review["cancel_review_order_ids"], [])
        self.assertEqual(review["preserved_order_ids"], ["safe-pending"])
        preserved = review["monitored_orders"][0]
        reason_codes = {item["code"] for item in preserved["monitor_reasons"]}
        self.assertIn("PENDING_CURRENT_CANDIDATE_NOT_LIVE_READY", reason_codes)
        self.assertIn("PENDING_ATTACK_ADVICE_NOT_CURRENT", reason_codes)
        self.assertEqual(
            preserved["reconcile_status"],
            "PRESERVE_BROKER_ANCHORED_THESIS",
        )
        self.assertIn("preserved `1`", report)

    def test_gateway_pending_tail_risk_uses_portfolio_cap_not_per_trade_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                closed_pls=(100.0, 80.0, 50.0),
            )
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot["home_conversions"] = {"USD": 150.0}
            snapshot["orders"] = [
                {
                    "order_id": "gateway-tail",
                    "pair": "EUR_USD",
                    "order_type": "STOP",
                    "state": "PENDING",
                    "units": 2000,
                    "price": 1.171,
                    "owner": "trader",
                    "trade_id": None,
                    "raw": {
                        "createTime": (_NOW - timedelta(minutes=10)).isoformat(),
                        "clientExtensions": {
                            "comment": (
                                "qr-vnext lane=trend_trader:EUR_USD:LONG:TREND_CONTINUATION "
                                "desk=trend_trader"
                            ),
                            "tag": "trader",
                        },
                        "takeProfitOnFill": {"price": "1.1730"},
                        "stopLossOnFill": {"price": "1.1600"},
                    },
                }
            ]
            files["snapshot"].write_text(json.dumps(snapshot))
            files["target"].write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_target_jpy": 1000.0,
                        "per_trade_risk_budget_jpy": 100.0,
                        "daily_risk_budget_jpy": 10_000.0,
                        "open_risk_jpy": 0.0,
                        "remaining_risk_budget_jpy": 10_000.0,
                    }
                )
            )
            files["intents"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "results": [
                            {
                                "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                                "status": "LIVE_READY",
                                "intent": {
                                    "pair": "EUR_USD",
                                    "side": "LONG",
                                    "order_type": "STOP-ENTRY",
                                },
                                "risk_issues": [],
                                "live_blockers": [],
                            }
                        ],
                    }
                )
            )
            files["attack_advice"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "status": "ATTACK_PARTIAL",
                        "recommended_now_lane_ids": ["trend_trader:EUR_USD:LONG:TREND_CONTINUATION"],
                    }
                )
            )

            _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        review = payload["execution_quality"]["pending_entry_reconcile"]
        self.assertNotIn("PENDING_ENTRY_CANCEL_REVIEW_REQUIRED", codes)
        self.assertEqual(review["cancel_review_orders"], 0)
        self.assertEqual(review["cancel_review_order_ids"], [])

    def test_gateway_pending_tail_risk_blocks_when_portfolio_capacity_exceeded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                closed_pls=(100.0, 80.0, 50.0),
            )
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot["home_conversions"] = {"USD": 150.0}
            snapshot["orders"] = [
                {
                    "order_id": "gateway-tail",
                    "pair": "EUR_USD",
                    "order_type": "STOP",
                    "state": "PENDING",
                    "units": 2000,
                    "price": 1.171,
                    "owner": "trader",
                    "trade_id": None,
                    "raw": {
                        "createTime": (_NOW - timedelta(minutes=10)).isoformat(),
                        "clientExtensions": {
                            "comment": (
                                "qr-vnext lane=trend_trader:EUR_USD:LONG:TREND_CONTINUATION "
                                "desk=trend_trader"
                            ),
                            "tag": "trader",
                        },
                        "takeProfitOnFill": {"price": "1.1730"},
                        "stopLossOnFill": {"price": "1.1600"},
                    },
                }
            ]
            files["snapshot"].write_text(json.dumps(snapshot))
            files["target"].write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_target_jpy": 1000.0,
                        "per_trade_risk_budget_jpy": 100.0,
                        "daily_risk_budget_jpy": 1_000.0,
                        "open_risk_jpy": 0.0,
                        "remaining_risk_budget_jpy": 1_000.0,
                    }
                )
            )
            files["intents"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "results": [
                            {
                                "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                                "status": "LIVE_READY",
                                "intent": {
                                    "pair": "EUR_USD",
                                    "side": "LONG",
                                    "order_type": "STOP-ENTRY",
                                },
                                "risk_issues": [],
                                "live_blockers": [],
                            }
                        ],
                    }
                )
            )
            files["attack_advice"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "status": "ATTACK_PARTIAL",
                        "recommended_now_lane_ids": ["trend_trader:EUR_USD:LONG:TREND_CONTINUATION"],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertIn("PENDING_ENTRY_CANCEL_REVIEW_REQUIRED", codes)
        review = payload["execution_quality"]["pending_entry_reconcile"]
        reason_codes = {item["code"] for item in review["orders"][0]["review_reasons"]}
        self.assertIn("PENDING_ATTACHED_SL_PORTFOLIO_RISK_EXCEEDS_CAP", reason_codes)
        self.assertNotIn("PENDING_ATTACHED_SL_RISK_EXCEEDS_CAP", reason_codes)
        self.assertEqual(review["orders"][0]["attached_sl_risk_cap_basis"], "PORTFOLIO_CAP")

    def test_repeated_self_improvement_finding_surfaces_anti_loop_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                closed_pls=(100.0, 80.0, 50.0),
            )

            _run(files, now=_NOW)
            _run(files, now=_NOW + timedelta(minutes=3))
            third = _run(files, now=_NOW + timedelta(minutes=6))
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(third.status, STATUS_BLOCKED)
        self.assertIn("REPEATED_SELF_IMPROVEMENT_LOOP", codes)
        loop = codes["REPEATED_SELF_IMPROVEMENT_LOOP"]
        self.assertEqual(loop["priority"], "P1")
        self.assertEqual(loop["evidence"]["current_streak"], 3)
        self.assertEqual(loop["evidence"]["previous_streak"], 2)
        self.assertEqual(loop["evidence"]["repeated_code"], "TARGET_OPEN_NO_LIVE_READY_LANES")

    def test_root_cause_focus_prioritizes_forecast_adverse_path_over_process_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, 50.0),
            )
            rows = []
            for idx in range(10):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 2)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "lead_time_min": 60.0,
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": "MISS",
                        "resolution_evidence": "2026-06-16T04:44:00Z invalidation 1.16800 touched before target 1.17200",
                        "cycle_id": f"invalidation-cycle-{idx}",
                    }
                )
            for idx in range(2):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 14)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "lead_time_min": 60.0,
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": "HIT",
                        "resolution_evidence": "2026-06-16T04:24:00Z target 1.17200 touched before invalidation 1.16800",
                        "cycle_id": f"hit-cycle-{idx}",
                    }
                )
            files["projection_ledger"].write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            _run(files, now=_NOW)
            _run(files, now=_NOW + timedelta(minutes=3))
            _run(files, now=_NOW + timedelta(minutes=6))
            payload = json.loads(files["output"].read_text())

        root_focus = payload["root_cause_focus"]
        self.assertEqual(root_focus["status"], "FOCUSED")
        self.assertEqual(root_focus["primary"]["family"], "FORECAST_ADVERSE_PATH")
        self.assertIn(
            "DIRECTIONAL_FORECAST_INVALIDATION_FIRST_DOMINANT",
            root_focus["primary"]["supporting_codes"],
        )
        self.assertEqual(payload["next_actions"][0]["code"], "ROOT_CAUSE_FOCUS:FORECAST_ADVERSE_PATH")
        self.assertIn("REPEATED_SELF_IMPROVEMENT_LOOP", {item["code"] for item in payload["findings"]})

    def test_root_cause_focus_keeps_secondary_repeated_pending_churn(self) -> None:
        findings = [
            {
                "priority": "P0",
                "layer": "decision_history",
                "code": "LATEST_GPT_DECISION_HAS_BLOCKING_ISSUES",
                "message": "latest GPT CLOSE rejected by spread gates",
                "next_action": "verify a fresh decision",
                "evidence": {},
            },
            {
                "priority": "P1",
                "layer": "execution_quality",
                "code": "PENDING_ENTRY_CANCEL_RATE_HIGH",
                "message": "pending entries are canceled before fill",
                "next_action": "separate thesis invalidation from entry-distance/TTL failures",
                "evidence": {
                    "cancel_before_fill_rate": 0.67,
                    "fill_rate": 0.33,
                },
            },
            {
                "priority": "P1",
                "layer": "process",
                "code": "REPEATED_SELF_IMPROVEMENT_LOOP",
                "message": "same self-improvement finding persisted",
                "next_action": "execute a narrow repair",
                "evidence": {
                    "repeated_code": "LATEST_GPT_DECISION_HAS_BLOCKING_ISSUES",
                    "repeated_priority": "P0",
                    "current_streak": 4,
                    "previous_streak": 3,
                    "repeated_findings": [
                        {
                            "code": "LATEST_GPT_DECISION_HAS_BLOCKING_ISSUES",
                            "priority": "P0",
                            "layer": "decision_history",
                            "current_streak": 4,
                            "previous_streak": 3,
                            "message": "latest GPT CLOSE rejected by spread gates",
                            "next_action": "verify a fresh decision",
                        },
                        {
                            "code": "PENDING_ENTRY_CANCEL_RATE_HIGH",
                            "priority": "P1",
                            "layer": "execution_quality",
                            "current_streak": 21,
                            "previous_streak": 20,
                            "message": "pending entries are canceled before fill",
                            "next_action": "separate thesis invalidation from entry-distance/TTL failures",
                        },
                    ],
                },
            },
        ]

        root_focus = _root_cause_focus(
            findings=findings,
            runtime={"live_ready_lanes": 0},
            effect_metrics={"window": {"net_jpy": -1084.8, "profit_factor": 0.868}},
            execution_quality={
                "pending_entry_lifecycle": {
                    "cancel_before_fill_rate": 0.67,
                    "fill_rate": 0.33,
                }
            },
        )

        execution = next(
            item for item in root_focus["candidates"] if item["family"] == "EXECUTION_LIFECYCLE"
        )
        self.assertEqual(execution["process_loop_streak"], 21)
        self.assertEqual(execution["confidence"], "HIGH")
        self.assertIn("PENDING_ENTRY_CANCEL_RATE_HIGH", execution["supporting_codes"])
        self.assertEqual(execution["metrics"]["pending_cancel_before_fill_rate"], 0.67)
        self.assertEqual(execution["metrics"]["pending_fill_rate"], 0.33)
        self.assertIn("pending_cancel_before_fill_rate=0.670", execution["why"])

    def test_root_cause_focus_repeated_loop_does_not_erase_guardian_metrics(self) -> None:
        findings = [
            {
                "priority": "P0",
                "layer": "execution_quality",
                "code": "POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE",
                "message": "position guardian is required but inactive",
                "next_action": "load only with explicit operator approval",
                "evidence": {
                    "guardian": {
                        "required": True,
                        "active": False,
                        "active_source": "plist_missing",
                    },
                    "profit_capture_miss_active": True,
                },
            },
            {
                "priority": "P1",
                "layer": "process",
                "code": "REPEATED_SELF_IMPROVEMENT_LOOP",
                "message": "same self-improvement finding persisted",
                "next_action": "execute a narrow repair",
                "evidence": {
                    "repeated_code": "POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE",
                    "repeated_layer": "execution_quality",
                    "current_streak": 7,
                    "previous_streak": 6,
                },
            },
        ]

        root_focus = _root_cause_focus(
            findings=findings,
            runtime={"live_ready_lanes": 0},
            effect_metrics={"window": {"net_jpy": -277.61, "profit_factor": 0.907}},
            execution_quality={},
        )

        primary = root_focus["primary"]
        self.assertEqual(primary["family"], "EXIT_AND_PROFIT_CAPTURE")
        self.assertEqual(primary["process_loop_streak"], 7)
        self.assertTrue(primary["metrics"]["position_guardian_required"])
        self.assertFalse(primary["metrics"]["position_guardian_active"])
        self.assertEqual(primary["metrics"]["position_guardian_active_source"], "plist_missing")
        self.assertTrue(primary["metrics"]["profit_capture_miss_active"])
        self.assertIn("position_guardian_active=False source=plist_missing", primary["why"])
        self.assertIn("profit_capture_miss_active=True", primary["why"])

    def test_action_required_for_hidden_open_loss_and_low_market_rr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=0.8,
                unrealized_pl_jpy=-300.0,
                closed_pls=(100.0, 80.0, -50.0),
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertEqual(summary.p0_findings, 0)
        self.assertIn("OPEN_LOSS_EXCEEDS_24H_REALIZED_GAIN", codes)
        self.assertIn("LIVE_READY_MARKET_RR_BELOW_ONE", codes)
        self.assertEqual(summary.live_ready_lanes, 1)

    def test_directional_forecast_timeout_only_samples_are_p1_learning_hole(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["projection_ledger"].write_text(
                "\n".join(
                    json.dumps(
                        {
                            "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 2)).isoformat(),
                            "pair": "EUR_USD",
                            "direction": "UP",
                            "signal_name": "directional_forecast",
                            "lead_time_min": 60.0,
                            "predicted_target_price": 1.1720,
                            "predicted_invalidation_price": 1.1680,
                            "resolution_window_min": 60.0,
                            "resolution_status": "TIMEOUT",
                            "resolution_evidence": "no candle truth for projection window",
                            "cycle_id": f"cycle-{idx}",
                        }
                    )
                    for idx in range(10)
                )
                + "\n"
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertEqual(summary.p0_findings, 0)
        self.assertIn("DIRECTIONAL_FORECAST_CALIBRATION_UNRESOLVED", codes)
        self.assertEqual(codes["DIRECTIONAL_FORECAST_CALIBRATION_UNRESOLVED"]["priority"], "P1")
        self.assertEqual(codes["DIRECTIONAL_FORECAST_CALIBRATION_UNRESOLVED"]["evidence"]["status_counts"]["TIMEOUT"], 10)

    def test_directional_forecast_low_hit_rate_is_p1_forecast_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            rows = []
            for idx in range(10):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 2)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "lead_time_min": 60.0,
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": "HIT" if idx == 0 else "MISS",
                        "cycle_id": f"cycle-{idx}",
                    }
                )
            files["projection_ledger"].write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertEqual(summary.p0_findings, 0)
        self.assertIn("DIRECTIONAL_FORECAST_HIT_RATE_WEAK", codes)
        evidence = codes["DIRECTIONAL_FORECAST_HIT_RATE_WEAK"]["evidence"]
        self.assertEqual(evidence["samples"], 10)
        self.assertAlmostEqual(evidence["hit_rate"], 0.1)
        self.assertTrue(evidence["worst_buckets"])

    def test_projection_headline_precision_gap_is_p1_forecast_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            rows = []
            for idx in range(50):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 2)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "EITHER",
                        "regime_at_emission": "TREND",
                        "signal_name": "bb_squeeze_expansion_imminent",
                        "resolution_window_min": 60.0,
                        "resolution_status": "HIT",
                        "cycle_id": f"hit-cycle-{idx}",
                    }
                )
            for idx in range(50):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 80)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "EITHER",
                        "regime_at_emission": "TREND",
                        "signal_name": "bb_squeeze_expansion_imminent",
                        "resolution_window_min": 60.0,
                        "resolution_status": "TIMEOUT",
                        "resolution_evidence": "no tradable expansion before projection expiry",
                        "cycle_id": f"timeout-cycle-{idx}",
                    }
                )
            for idx in range(96):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 140)).isoformat(),
                        "pair": "GBP_USD",
                        "direction": "EITHER",
                        "regime_at_emission": "TREND",
                        "signal_name": "session_expansion_london",
                        "resolution_window_min": 60.0,
                        "resolution_status": "HIT",
                        "cycle_id": f"edge-hit-cycle-{idx}",
                    }
                )
            for idx in range(4):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 240)).isoformat(),
                        "pair": "GBP_USD",
                        "direction": "EITHER",
                        "regime_at_emission": "TREND",
                        "signal_name": "session_expansion_london",
                        "resolution_window_min": 60.0,
                        "resolution_status": "MISS",
                        "resolution_evidence": "expansion threshold not reached",
                        "cycle_id": f"edge-miss-cycle-{idx}",
                    }
                )
            files["projection_ledger"].write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertEqual(summary.p0_findings, 0)
        self.assertIn("PROJECTION_ECONOMIC_PRECISION_WEAK", codes)
        finding = codes["PROJECTION_ECONOMIC_PRECISION_WEAK"]
        self.assertEqual(finding["priority"], "P1")
        weak = finding["evidence"]["weak_buckets"][0]
        self.assertEqual(weak["signal_name"], "bb_squeeze_expansion_imminent")
        self.assertEqual(weak["pair"], "EUR_USD")
        self.assertEqual(weak["samples"], 50)
        self.assertEqual(weak["economic_samples"], 100)
        self.assertAlmostEqual(weak["hit_rate"], 1.0)
        self.assertAlmostEqual(weak["economic_hit_rate"], 0.5)
        self.assertGreaterEqual(weak["hit_rate_wilson_lower"], 0.90)
        self.assertLess(weak["economic_hit_rate_wilson_lower"], 0.90)
        self.assertTrue(
            any(
                item["signal_name"] == "session_expansion_london"
                and item["pair"] == "GBP_USD"
                and item["passes_economic_precision"]
                for item in finding["evidence"]["usable_edges"]
            )
        )
        root_focus = payload["root_cause_focus"]
        forecast_candidates = [
            item
            for item in root_focus["candidates"]
            if "PROJECTION_ECONOMIC_PRECISION_WEAK" in item.get("supporting_codes", [])
        ]
        self.assertTrue(forecast_candidates)
        self.assertIn("projection_economic_precision_gap_count", forecast_candidates[0]["why"])
        self.assertIn("projection_economic_precision_edge_count", forecast_candidates[0]["why"])

    def test_legacy_watch_only_samples_keep_sample_shortfall_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            rows = []
            for idx in range(12):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 2)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "confidence": 0.18,
                        "entry_price": 1.1700,
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "lead_time_min": 60.0,
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": "MISS",
                        "resolution_evidence": (
                            "2026-06-16T04:44:00Z invalidation 1.16800 touched before target 1.17200"
                        ),
                        "cycle_id": f"watch-only-cycle-{idx}",
                    }
                )
            files["projection_ledger"].write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertIn("DIRECTIONAL_FORECAST_ENTRY_GRADE_SAMPLE_SHORTFALL", codes)
        self.assertNotIn("DIRECTIONAL_FORECAST_HIT_RATE_WEAK", codes)
        self.assertNotIn("DIRECTIONAL_FORECAST_INVALIDATION_FIRST_DOMINANT", codes)
        finding = codes["DIRECTIONAL_FORECAST_ENTRY_GRADE_SAMPLE_SHORTFALL"]
        self.assertEqual(finding["priority"], "P1")
        evidence = finding["evidence"]
        self.assertEqual(evidence["entry_grade_samples"], 0)
        self.assertEqual(evidence["watch_only_movement_samples"], 12)
        self.assertEqual(evidence["movement_samples"], 12)
        self.assertEqual(
            evidence["entry_grade_confidence_basis"],
            "raw_confidence_when_present_else_legacy_confidence",
        )

    def test_directional_forecast_raw_grade_survives_low_calibrated_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            rows = []
            for idx in range(10):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 2)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "confidence": 0.20,
                        "raw_confidence": 0.80,
                        "calibration_multiplier": 0.25,
                        "entry_price": 1.1700,
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "lead_time_min": 60.0,
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": "HIT" if idx == 0 else "MISS",
                        "resolution_evidence": (
                            "target 1.17200 touched before invalidation 1.16800"
                            if idx == 0
                            else "invalidation 1.16800 touched before target 1.17200"
                        ),
                        "cycle_id": f"raw-grade-cycle-{idx}",
                    }
                )
            files["projection_ledger"].write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n"
            )

            _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        evidence = codes["DIRECTIONAL_FORECAST_HIT_RATE_WEAK"]["evidence"]
        self.assertEqual(evidence["samples"], 10)
        self.assertAlmostEqual(evidence["hit_rate"], 0.1)
        self.assertEqual(evidence["raw_schema_input_rows"], 10)
        self.assertEqual(evidence["raw_schema_selected_rows"], 10)
        self.assertEqual(evidence["skipped_overlapping_raw_schema_rows"], 0)

    def test_directional_forecast_audit_excludes_pair_overlap_after_range_clock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            base = _NOW - timedelta(days=3)
            rows = []
            for idx in range(10):
                block = base + timedelta(minutes=120 * idx)
                rows.extend(
                    [
                        {
                            "timestamp_emitted_utc": block.isoformat(),
                            "pair": "EUR_USD",
                            "direction": "RANGE",
                            "confidence": 0.25,
                            "raw_confidence": 0.80,
                            "calibration_multiplier": 0.3125,
                            "entry_price": 1.1700,
                            "regime_at_emission": "RANGE",
                            "signal_name": "directional_forecast",
                            "lead_time_min": 60.0,
                            "predicted_range_low_price": 1.1680,
                            "predicted_range_high_price": 1.1720,
                            "technical_context_v1": _range_emission_context(
                                current_price=1.1700,
                            ),
                            "resolution_window_min": 60.0,
                            "resolution_status": "HIT",
                            "resolution_evidence": "range held",
                            "cycle_id": f"range-clock-{idx}",
                        },
                        {
                            "timestamp_emitted_utc": (block + timedelta(minutes=5)).isoformat(),
                            "pair": "EUR_USD",
                            "direction": "DOWN" if idx % 2 else "UP",
                            "confidence": 0.20,
                            "raw_confidence": 0.80,
                            "calibration_multiplier": 0.25,
                            "entry_price": 1.1700,
                            "regime_at_emission": "TREND",
                            "signal_name": "directional_forecast",
                            "lead_time_min": 60.0,
                            "predicted_target_price": 1.1680 if idx % 2 else 1.1720,
                            "predicted_invalidation_price": 1.1720 if idx % 2 else 1.1680,
                            "resolution_window_min": 60.0,
                            "resolution_status": "HIT",
                            "resolution_evidence": "target touched before invalidation",
                            "cycle_id": f"overlap-flip-{idx}",
                        },
                        {
                            "timestamp_emitted_utc": (block + timedelta(minutes=60)).isoformat(),
                            "pair": "EUR_USD",
                            "direction": "UP",
                            "confidence": 0.20,
                            "raw_confidence": 0.80,
                            "calibration_multiplier": 0.25,
                            "entry_price": 1.1700,
                            "regime_at_emission": "TREND",
                            "signal_name": "directional_forecast",
                            "lead_time_min": 60.0,
                            "predicted_target_price": 1.1720,
                            "predicted_invalidation_price": 1.1680,
                            "resolution_window_min": 60.0,
                            "resolution_status": "MISS",
                            "resolution_evidence": "invalidation touched before target",
                            "cycle_id": f"independent-miss-{idx}",
                        },
                    ]
                )
            files["projection_ledger"].write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n"
            )

            _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        evidence = codes["DIRECTIONAL_FORECAST_HIT_RATE_WEAK"]["evidence"]
        self.assertEqual(evidence["samples"], 10)
        self.assertEqual(evidence["hit_count"], 0)
        self.assertEqual(evidence["raw_schema_input_rows"], 30)
        self.assertEqual(evidence["raw_schema_selected_rows"], 20)
        self.assertEqual(evidence["skipped_overlapping_raw_schema_rows"], 10)
        self.assertTrue(evidence["independent_non_overlap"])

    def test_directional_forecast_same_timestamp_tie_uses_source_order(self) -> None:
        emitted_at = (_NOW - timedelta(hours=2)).isoformat()
        range_row = {
            "timestamp_emitted_utc": emitted_at,
            "pair": "EUR_USD",
            "direction": "RANGE",
            "signal_name": "directional_forecast",
            "lead_time_min": 60.0,
            "confidence": 0.20,
            "raw_confidence": 0.80,
            "calibration_multiplier": 0.25,
            "entry_price": 1.1700,
            "predicted_range_low_price": 1.1680,
            "predicted_range_high_price": 1.1720,
            "resolution_window_min": 60.0,
            "technical_context_v1": _range_emission_context(
                current_price=1.1700,
                h1_atr_pips=20.0,
            ),
        }
        up_row = {
            "timestamp_emitted_utc": emitted_at,
            "pair": "EUR_USD",
            "direction": "UP",
            "signal_name": "directional_forecast",
            "lead_time_min": 60.0,
            "confidence": 0.20,
            "raw_confidence": 0.80,
            "calibration_multiplier": 0.25,
            "entry_price": 1.1700,
            "predicted_target_price": 1.1720,
            "predicted_invalidation_price": 1.1680,
            "resolution_window_min": 60.0,
        }

        range_first, range_first_stats = _select_independent_directional_calibration_rows(
            [range_row, up_row]
        )
        up_first, up_first_stats = _select_independent_directional_calibration_rows(
            [up_row, range_row]
        )

        self.assertEqual(range_first, [range_row])
        self.assertEqual(up_first, [up_row])
        self.assertEqual(range_first_stats["skipped_overlapping_raw_schema_rows"], 1)
        self.assertEqual(up_first_stats["skipped_overlapping_raw_schema_rows"], 1)

    def test_directional_forecast_selector_bounds_horizon_and_datetime(self) -> None:
        base = _NOW - timedelta(days=3)

        def row(
            *,
            timestamp: str,
            pair: str,
            window_min: float,
            cycle_id: str,
        ) -> dict[str, object]:
            return {
                "timestamp_emitted_utc": timestamp,
                "pair": pair,
                "direction": "UP",
                "confidence": 0.24,
                "raw_confidence": 0.80,
                "calibration_multiplier": 0.30,
                "entry_price": 1.1700,
                "signal_name": "directional_forecast",
                "lead_time_min": 60.0,
                "predicted_target_price": 1.1720,
                "predicted_invalidation_price": 1.1680,
                "resolution_window_min": window_min,
                "resolution_status": "HIT",
                "cycle_id": cycle_id,
            }

        rows = [
            row(
                timestamp=base.isoformat(),
                pair="EUR_USD",
                window_min=1e300,
                cycle_id="huge-window",
            ),
            row(
                timestamp="9999-12-31T23:59:59+00:00",
                pair="EUR_USD",
                window_min=60.0,
                cycle_id="datetime-overflow",
            ),
            row(
                timestamp=(base + timedelta(minutes=5)).isoformat(),
                pair="EUR_USD",
                window_min=60.0,
                cycle_id="valid-after-invalid",
            ),
            row(
                timestamp=base.isoformat(),
                pair="GBP_USD",
                window_min=24.0 * 60.0,
                cycle_id="max-window-a",
            ),
            row(
                timestamp=(base + timedelta(days=1)).isoformat(),
                pair="GBP_USD",
                window_min=24.0 * 60.0,
                cycle_id="max-window-b",
            ),
        ]

        selected, stats = _select_independent_directional_calibration_rows(rows)

        self.assertEqual(
            [item["cycle_id"] for item in selected],
            ["valid-after-invalid", "max-window-a", "max-window-b"],
        )
        self.assertEqual(stats["raw_schema_selected_rows"], 3)
        self.assertEqual(stats["ex_ante_ineligible_raw_schema_rows_excluded"], 1)
        self.assertEqual(stats["invalid_raw_schema_rows_excluded"], 1)

    def test_directional_forecast_selector_rejects_noncanonical_or_unsupported_pairs(self) -> None:
        emitted_at = (_NOW - timedelta(hours=2)).isoformat()

        def row(pair: str, cycle_id: str) -> dict[str, object]:
            return {
                "timestamp_emitted_utc": emitted_at,
                "pair": pair,
                "direction": "UP",
                "confidence": 0.24,
                "raw_confidence": 0.80,
                "calibration_multiplier": 0.30,
                "entry_price": 1.1700,
                "signal_name": "directional_forecast",
                "lead_time_min": 60.0,
                "predicted_target_price": 1.1720,
                "predicted_invalidation_price": 1.1680,
                "resolution_window_min": 60.0,
                "resolution_status": "HIT",
                "cycle_id": cycle_id,
            }

        rows = [
            row("", "empty"),
            row(" ", "whitespace"),
            row("eur_usd", "noncanonical"),
            row("XAU_USD", "unsupported"),
            row("EUR_USD", "valid"),
        ]

        selected, stats = _select_independent_directional_calibration_rows(rows)

        self.assertEqual([item["cycle_id"] for item in selected], ["valid"])
        self.assertEqual(stats["raw_schema_selected_rows"], 1)
        self.assertEqual(stats["ex_ante_ineligible_raw_schema_rows_excluded"], 4)

    def test_legacy_directional_hundred_overlaps_collapse_before_quality_audit(self) -> None:
        base = _NOW - timedelta(hours=2)
        rows: list[dict[str, object]] = []
        for index in range(100):
            rows.append(
                {
                    "timestamp_emitted_utc": (
                        base + timedelta(seconds=index)
                    ).isoformat(),
                    "pair": "EUR_USD",
                    "direction": "UP",
                    "confidence": 0.80,
                    "entry_price": 1.1700,
                    "regime_at_emission": "TREND",
                    "signal_name": "directional_forecast",
                    "lead_time_min": 60.0,
                    "predicted_target_price": 1.1720,
                    "predicted_invalidation_price": 1.1680,
                    "resolution_window_min": 60.0,
                    "resolution_status": "PENDING" if index == 0 else "HIT",
                    "resolution_evidence": (
                        "" if index == 0 else "target touched before invalidation"
                    ),
                }
            )

        deduped = _deduped_directional_calibration_rows(rows)
        selected, stats = (
            _select_independent_legacy_directional_calibration_rows(deduped)
        )

        self.assertEqual(len(deduped), 100)
        self.assertEqual(selected, [rows[0]])
        self.assertEqual(stats["legacy_selected_rows"], 1)
        self.assertEqual(stats["skipped_overlapping_legacy_rows"], 99)

        codes = self._directional_audit_codes(rows)
        finding = codes["DIRECTIONAL_FORECAST_INDEPENDENT_SAMPLE_SHORTFALL"]
        evidence = finding["evidence"]
        self.assertEqual(evidence["rows"], 100)
        self.assertEqual(evidence["selected_trial_rows"], 1)
        self.assertEqual(evidence["skipped_overlapping_legacy_rows"], 99)

    def test_legacy_exact_dedupe_then_global_chronology_matches_projection_ledger(self) -> None:
        base = _NOW - timedelta(hours=4)

        def row(timestamp: datetime, *, status: str) -> dict[str, object]:
            return {
                "timestamp_emitted_utc": timestamp.isoformat(),
                "pair": "EUR_USD",
                "direction": "UP",
                "confidence": 0.80,
                "entry_price": 1.1700,
                "regime_at_emission": "TREND",
                "signal_name": "directional_forecast",
                "lead_time_min": 60.0,
                "predicted_target_price": 1.1720,
                "predicted_invalidation_price": 1.1680,
                "resolution_window_min": 60.0,
                "resolution_status": status,
                "resolution_evidence": "target touched before invalidation",
            }

        newer = row(base + timedelta(hours=2), status="MISS")
        duplicate_old_pending = row(
            base + timedelta(microseconds=100_000),
            status="PENDING",
        )
        duplicate_old_resolved = row(
            base + timedelta(microseconds=900_000),
            status="HIT",
        )
        source = [newer, duplicate_old_pending, duplicate_old_resolved]

        deduped = _deduped_directional_calibration_rows(source)
        selected, stats = (
            _select_independent_legacy_directional_calibration_rows(deduped)
        )
        chronological = _chronological_directional_calibration_rows(
            selected,
            source_index_by_id={id(item): index for index, item in enumerate(source)},
        )

        self.assertEqual(deduped, [newer, duplicate_old_resolved])
        self.assertEqual(stats["legacy_selected_rows"], 2)
        self.assertEqual(chronological, [duplicate_old_resolved, newer])

    def _directional_audit_codes(
        self,
        rows: list[dict[str, object]],
    ) -> dict[str, dict[str, object]]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["projection_ledger"].write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n"
            )
            _run(files)
            payload = json.loads(files["output"].read_text())
        return {item["code"]: item for item in payload["findings"]}

    def test_directional_forecast_audit_clocks_unscored_first_trial_before_outcome(self) -> None:
        for first_status, first_evidence in (
            ("PENDING", ""),
            ("TIMEOUT", "no candle truth for projection window"),
        ):
            with self.subTest(first_status=first_status):
                base = _NOW - timedelta(days=3)
                rows: list[dict[str, object]] = []
                for idx in range(10):
                    block = base + timedelta(minutes=120 * idx)

                    def row(
                        *,
                        minute: int,
                        status: str,
                        evidence: str,
                        suffix: str,
                    ) -> dict[str, object]:
                        return {
                            "timestamp_emitted_utc": (block + timedelta(minutes=minute)).isoformat(),
                            "pair": "EUR_USD",
                            "direction": "UP",
                            "confidence": 0.20,
                            "raw_confidence": 0.80,
                            "calibration_multiplier": 0.25,
                            "entry_price": 1.1700,
                            "regime_at_emission": "TREND",
                            "signal_name": "directional_forecast",
                            "lead_time_min": 60.0,
                            "predicted_target_price": 1.1720,
                            "predicted_invalidation_price": 1.1680,
                            "resolution_window_min": 60.0,
                            "resolution_status": status,
                            "resolution_evidence": evidence,
                            "cycle_id": f"{first_status.lower()}-{idx}-{suffix}",
                        }

                    rows.extend(
                        [
                            row(
                                minute=0,
                                status=first_status,
                                evidence=first_evidence,
                                suffix="first",
                            ),
                            row(
                                minute=5,
                                status="HIT",
                                evidence="target touched before invalidation",
                                suffix="overlap-hit",
                            ),
                            row(
                                minute=60,
                                status="MISS",
                                evidence="invalidation touched before target",
                                suffix="boundary-miss",
                            ),
                        ]
                    )

                evidence = self._directional_audit_codes(rows)[
                    "DIRECTIONAL_FORECAST_HIT_RATE_WEAK"
                ]["evidence"]
                self.assertEqual(evidence["samples"], 10)
                self.assertEqual(evidence["hit_count"], 0)
                self.assertEqual(evidence["raw_schema_input_rows"], 30)
                self.assertEqual(evidence["raw_schema_selected_rows"], 20)
                self.assertEqual(evidence["skipped_overlapping_raw_schema_rows"], 10)

    def test_unscored_clock_owner_blocks_later_hit_from_coverage_and_matches_ledger(self) -> None:
        from quant_rabbit.strategy.projection_ledger import compute_hit_rates

        base = _NOW - timedelta(days=3)
        rows: list[dict[str, object]] = []
        for idx in range(10):
            block = base + timedelta(minutes=120 * idx)

            def row(*, minute: int, status: str, evidence: str, suffix: str) -> dict[str, object]:
                return {
                    "timestamp_emitted_utc": (
                        block + timedelta(minutes=minute)
                    ).isoformat(),
                    "pair": "EUR_USD",
                    "direction": "UP",
                    "confidence": 0.20,
                    "raw_confidence": 0.80,
                    "calibration_multiplier": 0.25,
                    "entry_price": 1.1700,
                    "regime_at_emission": "TREND",
                    "signal_name": "directional_forecast",
                    "lead_time_min": 60.0,
                    "predicted_target_price": 1.1720,
                    "predicted_invalidation_price": 1.1680,
                    "resolution_window_min": 60.0,
                    "resolution_status": status,
                    "resolution_evidence": evidence,
                    "cycle_id": f"truth-gap-clock-{idx}-{suffix}",
                }

            rows.extend(
                [
                    row(
                        minute=0,
                        status="TIMEOUT",
                        evidence="no candle truth for projection window",
                        suffix="clock-owner",
                    ),
                    row(
                        minute=5,
                        status="HIT",
                        evidence="target touched before invalidation",
                        suffix="overlap-hit",
                    ),
                ]
            )

        codes = self._directional_audit_codes(rows)
        unresolved = codes["DIRECTIONAL_FORECAST_CALIBRATION_UNRESOLVED"]
        evidence = unresolved["evidence"]
        self.assertEqual(evidence["selected_trial_rows"], 10)
        self.assertEqual(evidence["scored_outcome_samples"], 0)
        self.assertEqual(evidence["calibration_coverage"], 0.0)
        self.assertEqual(evidence["status_counts"], {"TIMEOUT": 10})
        self.assertEqual(evidence["raw_schema_input_rows"], 20)
        self.assertEqual(evidence["raw_schema_selected_rows"], 10)
        self.assertEqual(evidence["skipped_overlapping_raw_schema_rows"], 10)
        self.assertNotIn("DIRECTIONAL_FORECAST_HIT_RATE_WEAK", codes)

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            (data_root / "projection_ledger.jsonl").write_text(
                "\n".join(json.dumps(item) for item in rows) + "\n"
            )
            hit_rates = compute_hit_rates(data_root)
        self.assertNotIn("directional_forecast", hit_rates)

    def test_directional_forecast_audit_excludes_low_raw_before_pair_clock(self) -> None:
        base = _NOW - timedelta(days=3)
        rows: list[dict[str, object]] = []
        for idx in range(10):
            block = base + timedelta(minutes=180 * idx)
            for minute, raw_confidence, status, suffix in (
                (0, 0.20, "MISS", "low-raw-first"),
                (5, 0.80, "HIT", "entry-grade-hit"),
                (65, 0.80, "MISS", "entry-grade-miss-a"),
                (125, 0.80, "MISS", "entry-grade-miss-b"),
            ):
                rows.append(
                    {
                        "timestamp_emitted_utc": (block + timedelta(minutes=minute)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "confidence": raw_confidence * 0.25,
                        "raw_confidence": raw_confidence,
                        "calibration_multiplier": 0.25,
                        "entry_price": 1.1700,
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "lead_time_min": 60.0,
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": status,
                        "resolution_evidence": (
                            "target touched before invalidation"
                            if status == "HIT"
                            else "invalidation touched before target"
                        ),
                        "cycle_id": f"low-raw-order-{idx}-{suffix}",
                    }
                )

        evidence = self._directional_audit_codes(rows)[
            "DIRECTIONAL_FORECAST_HIT_RATE_WEAK"
        ]["evidence"]
        self.assertEqual(evidence["samples"], 30)
        self.assertEqual(evidence["hit_count"], 10)
        self.assertEqual(evidence["raw_schema_input_rows"], 40)
        self.assertEqual(evidence["raw_schema_selected_rows"], 30)
        self.assertEqual(evidence["ex_ante_ineligible_raw_schema_rows_excluded"], 10)

    def test_directional_forecast_audit_excludes_low_raw_range_before_pair_clock(self) -> None:
        base = _NOW - timedelta(days=3)
        rows: list[dict[str, object]] = []
        for idx in range(10):
            block = base + timedelta(minutes=180 * idx)
            rows.append(
                {
                    "timestamp_emitted_utc": block.isoformat(),
                    "pair": "EUR_USD",
                    "direction": "RANGE",
                    "confidence": 0.06,
                    "raw_confidence": 0.20,
                    "calibration_multiplier": 0.30,
                    "entry_price": 1.1700,
                    "regime_at_emission": "RANGE",
                    "signal_name": "directional_forecast",
                    "lead_time_min": 60.0,
                    "predicted_range_low_price": 1.1680,
                    "predicted_range_high_price": 1.1720,
                    "resolution_window_min": 60.0,
                    "resolution_status": "HIT",
                    "resolution_evidence": "range held",
                    "cycle_id": f"low-raw-range-{idx}",
                }
            )
            for minute, status, suffix in (
                (5, "HIT", "entry-grade-hit"),
                (65, "MISS", "entry-grade-miss-a"),
                (125, "MISS", "entry-grade-miss-b"),
            ):
                rows.append(
                    {
                        "timestamp_emitted_utc": (block + timedelta(minutes=minute)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "confidence": 0.24,
                        "raw_confidence": 0.80,
                        "calibration_multiplier": 0.30,
                        "entry_price": 1.1700,
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "lead_time_min": 60.0,
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": status,
                        "resolution_evidence": (
                            "target touched before invalidation"
                            if status == "HIT"
                            else "invalidation touched before target"
                        ),
                        "cycle_id": f"low-raw-range-order-{idx}-{suffix}",
                    }
                )

        evidence = self._directional_audit_codes(rows)[
            "DIRECTIONAL_FORECAST_HIT_RATE_WEAK"
        ]["evidence"]
        self.assertEqual(evidence["samples"], 30)
        self.assertEqual(evidence["hit_count"], 10)
        self.assertEqual(evidence["raw_schema_selected_rows"], 30)
        self.assertEqual(evidence["ex_ante_ineligible_raw_schema_rows_excluded"], 10)

    def test_directional_forecast_audit_excludes_bad_geometry_before_pair_clock(self) -> None:
        base = _NOW - timedelta(days=3)
        rows: list[dict[str, object]] = []
        for idx in range(10):
            block = base + timedelta(minutes=180 * idx)
            for minute, target, invalidation, status, suffix in (
                (0, 1.1680, 1.1690, "HIT", "bad-up-geometry"),
                (5, 1.1720, 1.1680, "HIT", "valid-hit"),
                (65, 1.1720, 1.1680, "MISS", "valid-miss-a"),
                (125, 1.1720, 1.1680, "MISS", "valid-miss-b"),
            ):
                rows.append(
                    {
                        "timestamp_emitted_utc": (block + timedelta(minutes=minute)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "confidence": 0.20,
                        "raw_confidence": 0.80,
                        "calibration_multiplier": 0.25,
                        "entry_price": 1.1700,
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "lead_time_min": 60.0,
                        "predicted_target_price": target,
                        "predicted_invalidation_price": invalidation,
                        "resolution_window_min": 60.0,
                        "resolution_status": status,
                        "resolution_evidence": (
                            "target touched before invalidation"
                            if status == "HIT"
                            else "invalidation touched before target"
                        ),
                        "cycle_id": f"geometry-order-{idx}-{suffix}",
                    }
                )

        evidence = self._directional_audit_codes(rows)[
            "DIRECTIONAL_FORECAST_HIT_RATE_WEAK"
        ]["evidence"]
        self.assertEqual(evidence["samples"], 30)
        self.assertEqual(evidence["hit_count"], 10)
        self.assertEqual(evidence["raw_schema_selected_rows"], 30)
        self.assertEqual(evidence["ex_ante_ineligible_raw_schema_rows_excluded"], 10)

    def test_directional_forecast_audit_rejects_forged_confidence_triplets(self) -> None:
        base = _NOW - timedelta(days=3)
        rows: list[dict[str, object]] = []
        for idx in range(10):
            block = base + timedelta(minutes=180 * idx)
            rows.append(
                {
                    "timestamp_emitted_utc": block.isoformat(),
                    "pair": "EUR_USD",
                    "direction": "UP",
                    "confidence": 0.06,
                    "raw_confidence": 0.80,
                    "calibration_multiplier": 0.30,
                    "entry_price": 1.1700,
                    "regime_at_emission": "TREND",
                    "signal_name": "directional_forecast",
                    "lead_time_min": 60.0,
                    "predicted_target_price": 1.1720,
                    "predicted_invalidation_price": 1.1680,
                    "resolution_window_min": 60.0,
                    "resolution_status": "MISS",
                    "resolution_evidence": "invalidation touched before target",
                    "cycle_id": f"forged-triplet-{idx}",
                }
            )
            for minute, status, suffix in (
                (5, "HIT", "valid-hit"),
                (65, "MISS", "valid-miss-a"),
                (125, "MISS", "valid-miss-b"),
            ):
                rows.append(
                    {
                        "timestamp_emitted_utc": (block + timedelta(minutes=minute)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "confidence": 0.24,
                        "raw_confidence": 0.80,
                        "calibration_multiplier": 0.30,
                        "entry_price": 1.1700,
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "lead_time_min": 60.0,
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": status,
                        "resolution_evidence": (
                            "target touched before invalidation"
                            if status == "HIT"
                            else "invalidation touched before target"
                        ),
                        "cycle_id": f"triplet-order-{idx}-{suffix}",
                    }
                )

        evidence = self._directional_audit_codes(rows)[
            "DIRECTIONAL_FORECAST_HIT_RATE_WEAK"
        ]["evidence"]
        self.assertEqual(evidence["samples"], 30)
        self.assertEqual(evidence["hit_count"], 10)
        self.assertEqual(evidence["raw_schema_selected_rows"], 30)
        self.assertEqual(evidence["invalid_raw_schema_rows_excluded"], 10)

    def test_directional_forecast_audit_rejects_partial_or_invalid_raw_schema(self) -> None:
        base = _NOW - timedelta(days=3)
        rows: list[dict[str, object]] = []
        invalid_variants = (
            {"raw_confidence": "corrupt", "calibration_multiplier": 0.25},
            {"raw_confidence": 0.80},
            {"raw_confidence": 0.80, "calibration_multiplier": "corrupt"},
        )
        for idx in range(10):
            block = base + timedelta(minutes=180 * idx)
            invalid_row: dict[str, object] = {
                "timestamp_emitted_utc": block.isoformat(),
                "pair": "EUR_USD",
                "direction": "UP",
                "confidence": 0.99,
                "entry_price": 1.1700,
                "regime_at_emission": "TREND",
                "signal_name": "directional_forecast",
                "lead_time_min": 60.0,
                "predicted_target_price": 1.1720,
                "predicted_invalidation_price": 1.1680,
                "resolution_window_min": 60.0,
                "resolution_status": "HIT",
                "resolution_evidence": "target touched before invalidation",
                "cycle_id": f"invalid-schema-{idx}",
                **invalid_variants[idx % len(invalid_variants)],
            }
            rows.append(invalid_row)
            for minute, status, suffix in (
                (5, "HIT", "valid-hit"),
                (65, "MISS", "valid-miss-a"),
                (125, "MISS", "valid-miss-b"),
            ):
                rows.append(
                    {
                        **invalid_row,
                        "timestamp_emitted_utc": (block + timedelta(minutes=minute)).isoformat(),
                        "confidence": 0.20,
                        "raw_confidence": 0.80,
                        "calibration_multiplier": 0.25,
                        "resolution_status": status,
                        "resolution_evidence": (
                            "target touched before invalidation"
                            if status == "HIT"
                            else "invalidation touched before target"
                        ),
                        "cycle_id": f"invalid-schema-{idx}-{suffix}",
                    }
                )

        evidence = self._directional_audit_codes(rows)[
            "DIRECTIONAL_FORECAST_HIT_RATE_WEAK"
        ]["evidence"]
        self.assertEqual(evidence["samples"], 30)
        self.assertEqual(evidence["hit_count"], 10)
        self.assertEqual(evidence["raw_schema_input_rows"], 40)
        self.assertEqual(evidence["raw_schema_selected_rows"], 30)
        self.assertEqual(evidence["invalid_raw_schema_rows_excluded"], 10)

    def test_directional_forecast_audit_survives_nonobject_and_bad_numeric_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            rows: list[dict[str, object]] = []
            for idx in range(10):
                status = "HIT" if idx == 0 else "MISS"
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 2)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "confidence": 0.24,
                        "raw_confidence": 0.80,
                        "calibration_multiplier": 0.30,
                        "entry_price": 1.1700,
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "lead_time_min": 60.0,
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": status,
                        "resolution_evidence": (
                            "target touched before invalidation"
                            if status == "HIT"
                            else "invalidation touched before target"
                        ),
                        "cycle_id": f"valid-numeric-{idx}",
                    }
                )
            bad_base = {**rows[-1], "resolution_status": "HIT"}
            bad_rows = [
                {**bad_base, "cycle_id": "bad-lead", "lead_time_min": "bad"},
                {**bad_base, "cycle_id": "bad-confidence", "confidence": float("inf")},
                {
                    **bad_base,
                    "cycle_id": "bad-window",
                    "resolution_window_min": 10**400,
                },
                {**bad_base, "cycle_id": "bad-entry", "entry_price": ["bad"]},
            ]
            lines = [json.dumps(7), json.dumps([{"bad": "shape"}])]
            lines.extend(json.dumps(row) for row in rows + bad_rows)
            files["projection_ledger"].write_text("\n".join(lines) + "\n")

            _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertIn("PROJECTION_LEDGER_MALFORMED_ROWS", codes)
        self.assertEqual(
            codes["PROJECTION_LEDGER_MALFORMED_ROWS"]["evidence"]["malformed_rows"],
            2,
        )
        evidence = codes["DIRECTIONAL_FORECAST_HIT_RATE_WEAK"]["evidence"]
        self.assertEqual(evidence["samples"], 10)
        self.assertEqual(evidence["hit_count"], 1)
        self.assertEqual(evidence["invalid_raw_schema_rows_excluded"], 3)
        self.assertEqual(evidence["ex_ante_ineligible_raw_schema_rows_excluded"], 1)

    def test_directional_forecast_audit_counts_only_scored_no_touch_timeouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            rows = []
            for idx in range(10):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 2)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "confidence": 0.20,
                        "raw_confidence": 0.80,
                        "calibration_multiplier": 0.25,
                        "entry_price": 1.1700,
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "lead_time_min": 60.0,
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": "TIMEOUT",
                        "resolution_evidence": (
                            "target 1.17200 and invalidation 1.16800 both untouched in forecast window"
                        ),
                        "cycle_id": f"no-touch-{idx}",
                    }
                )
            rows.append(
                {
                    **rows[-1],
                    "timestamp_emitted_utc": (_NOW - timedelta(hours=13)).isoformat(),
                    "resolution_evidence": "no candle truth for projection window",
                    "cycle_id": "truth-missing",
                }
            )
            files["projection_ledger"].write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n"
            )

            _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        evidence = codes["DIRECTIONAL_FORECAST_TARGET_TIMEOUT_DOMINANT"]["evidence"]
        self.assertEqual(evidence["samples"], 10)
        self.assertEqual(evidence["target_timeout_samples"], 10)
        self.assertEqual(evidence["raw_schema_input_rows"], 11)
        self.assertEqual(evidence["raw_schema_selected_rows"], 11)

    def test_directional_forecast_audit_range_truth_gaps_are_zero_learning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            evidence_variants = (
                "incomplete closed candle truth for full projection window; retryable",
                "no candle truth for projection window",
                "market closed at projection emission; excluded from calibration",
                "malformed candle truth",
            )
            rows = [
                {
                    "timestamp_emitted_utc": (
                        _NOW - timedelta(hours=2 * index + 2)
                    ).isoformat(),
                    "pair": "EUR_USD",
                    "direction": "RANGE",
                    "confidence": 0.24,
                    "raw_confidence": 0.80,
                    "calibration_multiplier": 0.30,
                    "entry_price": 1.1000,
                    "regime_at_emission": "RANGE",
                    "signal_name": "directional_forecast",
                    "lead_time_min": 60.0,
                    "predicted_range_low_price": 1.0990,
                    "predicted_range_high_price": 1.1010,
                    "resolution_window_min": 60.0,
                    "resolution_status": "TIMEOUT",
                    "resolution_evidence": evidence_variants[index % len(evidence_variants)],
                    "cycle_id": f"range-truth-gap-{index}",
                }
                for index in range(12)
            ]
            files["projection_ledger"].write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n"
            )

            _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertNotIn("DIRECTIONAL_FORECAST_TARGET_TIMEOUT_DOMINANT", codes)
        unresolved = codes["DIRECTIONAL_FORECAST_CALIBRATION_UNRESOLVED"]
        self.assertEqual(unresolved["evidence"]["target_timeout_samples"], 0)
        self.assertEqual(
            unresolved["evidence"][
                "invalid_range_emission_atr_context_rows_excluded"
            ],
            12,
        )

    def test_directional_forecast_low_hit_rate_excludes_range_box_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            rows = []
            for idx in range(10):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 2)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": "MISS",
                        "cycle_id": f"movement-cycle-{idx}",
                    }
                )
            for idx in range(20):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 2)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "RANGE",
                        "regime_at_emission": "RANGE",
                        "signal_name": "directional_forecast",
                        "predicted_range_low_price": 1.1680,
                        "predicted_range_high_price": 1.1720,
                        "resolution_window_min": 120.0,
                        "resolution_status": "HIT",
                        "cycle_id": f"range-cycle-{idx}",
                    }
                )
            files["projection_ledger"].write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        hit_rate = codes["DIRECTIONAL_FORECAST_HIT_RATE_WEAK"]
        evidence = hit_rate["evidence"]
        self.assertEqual(evidence["samples"], 10)
        self.assertEqual(evidence["hit_count"], 0)
        self.assertAlmostEqual(evidence["hit_rate"], 0.0)
        # Legacy RANGE rows are emitted hourly with a two-hour truth window,
        # so only the ten exact-boundary trials are independent.
        self.assertEqual(evidence["range_samples_excluded"], 10)
        self.assertEqual(evidence["skipped_overlapping_legacy_rows"], 10)
        self.assertEqual(evidence["total_calibrated_samples"], 20)

    def test_directional_forecast_no_touch_misses_are_target_timeout_not_hit_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            rows = []
            for idx in range(15):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 2)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": "MISS",
                        "resolution_evidence": "target 1.17200 not reached before invalidation 1.16800",
                        "cycle_id": f"target-timeout-cycle-{idx}",
                    }
                )
            for idx in range(10):
                status = "HIT" if idx < 5 else "MISS"
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 20)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": status,
                        "resolution_evidence": (
                            "2026-06-16T04:24:00Z target 1.17200 touched before invalidation 1.16800"
                            if status == "HIT"
                            else "2026-06-16T04:44:00Z invalidation 1.16800 touched before target 1.17200"
                        ),
                        "cycle_id": f"touch-cycle-{idx}",
                    }
                )
            files["projection_ledger"].write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertIn("DIRECTIONAL_FORECAST_TARGET_TIMEOUT_DOMINANT", codes)
        self.assertNotIn("DIRECTIONAL_FORECAST_HIT_RATE_WEAK", codes)
        evidence = codes["DIRECTIONAL_FORECAST_TARGET_TIMEOUT_DOMINANT"]["evidence"]
        self.assertEqual(evidence["target_timeout_samples"], 15)
        self.assertEqual(evidence["touch_calibrated_samples"], 10)
        self.assertAlmostEqual(evidence["target_timeout_rate"], 0.6)

    def test_directional_forecast_invalidation_first_dominant_is_p1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            rows = []
            for idx in range(10):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 2)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": "MISS",
                        "resolution_evidence": "2026-06-16T04:44:00Z invalidation 1.16800 touched before target 1.17200",
                        "cycle_id": f"invalidation-cycle-{idx}",
                    }
                )
            for idx in range(2):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 14)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": "HIT",
                        "resolution_evidence": "2026-06-16T04:24:00Z target 1.17200 touched before invalidation 1.16800",
                        "cycle_id": f"hit-cycle-{idx}",
                    }
                )
            files["projection_ledger"].write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        finding = codes["DIRECTIONAL_FORECAST_INVALIDATION_FIRST_DOMINANT"]
        self.assertEqual(finding["priority"], "P1")
        evidence = finding["evidence"]
        self.assertEqual(evidence["samples"], 12)
        self.assertEqual(evidence["invalidation_first_count"], 10)
        self.assertAlmostEqual(evidence["invalidation_first_rate"], 0.8333)
        self.assertEqual(evidence["worst_buckets"][0]["pair"], "EUR_USD")

    def test_directional_forecast_historical_weakness_is_p2_when_recent_window_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            rows = []
            for idx in range(10):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(days=8, hours=idx)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": "MISS",
                        "cycle_id": f"old-cycle-{idx}",
                    }
                )
            for idx in range(10):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 2)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": "HIT" if idx < 6 else "MISS",
                        "cycle_id": f"recent-cycle-{idx}",
                    }
                )
            files["projection_ledger"].write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        hit_rate = codes["DIRECTIONAL_FORECAST_HIT_RATE_WEAK"]
        self.assertEqual(hit_rate["priority"], "P2")
        self.assertTrue(hit_rate["evidence"]["recent_recovered"])
        self.assertAlmostEqual(hit_rate["evidence"]["window_hit_rates"]["7d"]["hit_rate"], 0.6)
        bucket = codes["DIRECTIONAL_FORECAST_BUCKET_HIT_RATE_WEAK"]
        self.assertEqual(bucket["priority"], "P2")
        self.assertTrue(bucket["evidence"]["recent_recovered"])

    def test_directional_forecast_timeout_dominant_is_p1_calibration_hole(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            rows = []
            for idx in range(2):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 2)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": "HIT" if idx == 0 else "MISS",
                        "cycle_id": f"cycle-hitmiss-{idx}",
                    }
                )
            for idx in range(10):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 4)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": "TIMEOUT",
                        "cycle_id": f"cycle-timeout-{idx}",
                    }
                )
            files["projection_ledger"].write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertEqual(summary.p0_findings, 0)
        self.assertIn("DIRECTIONAL_FORECAST_CALIBRATION_TIMEOUT_DOMINANT", codes)
        evidence = codes["DIRECTIONAL_FORECAST_CALIBRATION_TIMEOUT_DOMINANT"]["evidence"]
        self.assertEqual(evidence["calibrated_samples"], 2)
        self.assertEqual(evidence["status_counts"]["TIMEOUT"], 10)
        self.assertLess(evidence["calibration_coverage"], evidence["min_coverage"])

    def test_directional_forecast_missing_geometry_is_p1_calibration_hole(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            rows = []
            for idx in range(2):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 2)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": "HIT" if idx == 0 else "MISS",
                        "cycle_id": f"cycle-hitmiss-{idx}",
                    }
                )
            for idx in range(10):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 4)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "predicted_target_price": None,
                        "predicted_invalidation_price": None,
                        "resolution_window_min": 60.0,
                        "resolution_status": "HIT" if idx < 6 else "MISS",
                        "cycle_id": f"cycle-legacy-{idx}",
                    }
                )
            files["projection_ledger"].write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertEqual(summary.p0_findings, 0)
        self.assertIn("DIRECTIONAL_FORECAST_CALIBRATION_GEOMETRY_MISSING", codes)
        self.assertNotIn("DIRECTIONAL_FORECAST_CALIBRATION_TIMEOUT_DOMINANT", codes)
        evidence = codes["DIRECTIONAL_FORECAST_CALIBRATION_GEOMETRY_MISSING"]["evidence"]
        self.assertEqual(evidence["calibrated_samples"], 2)
        self.assertEqual(evidence["missing_geometry_samples"], 10)
        self.assertLess(evidence["calibration_coverage"], evidence["min_coverage"])

    def test_directional_forecast_old_missing_geometry_is_p2_when_recent_geometry_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            rows = []
            for idx in range(14):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(days=8, hours=idx)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "predicted_target_price": None,
                        "predicted_invalidation_price": None,
                        "resolution_window_min": 60.0,
                        "resolution_status": "MISS",
                        "cycle_id": f"old-legacy-{idx}",
                    }
                )
            for idx in range(12):
                rows.append(
                    {
                        "timestamp_emitted_utc": (_NOW - timedelta(hours=idx + 2)).isoformat(),
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "regime_at_emission": "TREND",
                        "signal_name": "directional_forecast",
                        "predicted_target_price": 1.1720,
                        "predicted_invalidation_price": 1.1680,
                        "resolution_window_min": 60.0,
                        "resolution_status": "HIT" if idx < 7 else "MISS",
                        "cycle_id": f"recent-geometry-{idx}",
                    }
                )
            files["projection_ledger"].write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        finding = codes["DIRECTIONAL_FORECAST_CALIBRATION_GEOMETRY_MISSING"]
        self.assertEqual(finding["priority"], "P2")
        evidence = finding["evidence"]
        self.assertTrue(evidence["recent_recovered"])
        self.assertEqual(evidence["recent_24h_rows"], 12)
        self.assertEqual(evidence["recent_24h_calibrated_samples"], 12)
        self.assertGreaterEqual(evidence["recent_24h_calibration_coverage"], evidence["min_coverage"])

    def test_order_intents_without_market_context_refs_are_p1_when_matrix_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["market_context_matrix"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "pairs": {"EUR_USD": {"LONG": {"support": []}}},
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertIn("ORDER_INTENTS_MARKET_CONTEXT_EVIDENCE_MISSING", codes)
        finding = codes["ORDER_INTENTS_MARKET_CONTEXT_EVIDENCE_MISSING"]
        self.assertEqual(finding["priority"], "P1")
        self.assertEqual(finding["evidence"]["candidate_count"], 1)
        self.assertEqual(finding["evidence"]["with_context_refs"], 0)

    def test_forecast_history_duplicate_cycle_pair_is_p1_measurement_hole(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["forecast_history"].write_text(
                "\n".join(
                    json.dumps(
                        {
                            "timestamp_utc": (_NOW + timedelta(seconds=idx)).isoformat(),
                            "cycle_id": "cycle-dup",
                            "pair": "EUR_USD",
                            "direction": "UP",
                            "confidence": 0.62,
                        }
                    )
                    for idx in range(2)
                )
                + "\n"
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertIn("FORECAST_HISTORY_DUPLICATE_CYCLE_PAIR", codes)
        finding = codes["FORECAST_HISTORY_DUPLICATE_CYCLE_PAIR"]
        self.assertEqual(finding["priority"], "P1")
        self.assertEqual(finding["evidence"]["duplicate_cycle_pair_groups"], 1)
        self.assertEqual(finding["evidence"]["examples"][0]["pair"], "EUR_USD")
        self.assertEqual(finding["evidence"]["examples"][0]["count"], 2)

    def test_legacy_forecast_history_phantom_clusters_are_p2_dedupe_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["forecast_history"].write_text(
                "\n".join(
                    json.dumps(
                        {
                            "timestamp_utc": _NOW.isoformat(),
                            "pair": "EUR_USD",
                            "direction": "UP",
                            "confidence": 0.62,
                        }
                    )
                    for _idx in range(3)
                )
                + "\n"
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertIn("FORECAST_HISTORY_LEGACY_PHANTOM_CLUSTERS", codes)
        finding = codes["FORECAST_HISTORY_LEGACY_PHANTOM_CLUSTERS"]
        self.assertEqual(finding["priority"], "P2")
        self.assertEqual(finding["evidence"]["phantom_clusters"], 1)

    def test_market_context_ref_on_intent_satisfies_context_evidence_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["market_context_matrix"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "pairs": {"EUR_USD": {"LONG": {"support": []}}},
                    }
                )
            )
            intents = json.loads(files["intents"].read_text())
            intents["results"][0]["intent"]["metadata"]["market_context_matrix_ref"] = "matrix:EUR_USD:LONG"
            files["intents"].write_text(json.dumps(intents))

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertNotIn("ORDER_INTENTS_MARKET_CONTEXT_EVIDENCE_MISSING", codes)

    def test_order_intents_predating_matrix_with_live_ready_lane_is_p0_stale_context_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["market_context_matrix"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "pairs": {"EUR_USD": {"LONG": {"support": []}}},
                    }
                )
            )
            intents = json.loads(files["intents"].read_text())
            intents["generated_at_utc"] = (_NOW - timedelta(hours=1)).isoformat()
            intents["results"][0]["intent"]["metadata"]["market_context_matrix_ref"] = "matrix:EUR_USD:LONG"
            files["intents"].write_text(json.dumps(intents))

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertEqual(summary.p0_findings, 1)
        self.assertIn("ORDER_INTENTS_MARKET_CONTEXT_EVIDENCE_STALE", codes)
        self.assertNotIn("ORDER_INTENTS_MARKET_CONTEXT_EVIDENCE_MISSING", codes)
        finding = codes["ORDER_INTENTS_MARKET_CONTEXT_EVIDENCE_STALE"]
        self.assertEqual(finding["priority"], "P0")
        self.assertEqual(finding["evidence"]["candidate_count"], 1)
        self.assertEqual(finding["evidence"]["live_ready_lanes"], 1)
        self.assertEqual(finding["evidence"]["with_context_refs"], 1)

    def test_order_intents_predating_matrix_without_live_ready_lane_stays_p1_context_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
                pending_entry=True,
            )
            files["market_context_matrix"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "pairs": {"EUR_USD": {"LONG": {"support": []}}},
                    }
                )
            )
            intents = json.loads(files["intents"].read_text())
            intents["generated_at_utc"] = (_NOW - timedelta(hours=1)).isoformat()
            intents["results"][0]["status"] = "DRY_RUN_BLOCKED"
            intents["results"][0]["intent"]["metadata"]["market_context_matrix_ref"] = "matrix:EUR_USD:LONG"
            files["intents"].write_text(json.dumps(intents))

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertIn("ORDER_INTENTS_MARKET_CONTEXT_EVIDENCE_STALE", codes)
        finding = codes["ORDER_INTENTS_MARKET_CONTEXT_EVIDENCE_STALE"]
        self.assertEqual(finding["priority"], "P1")
        self.assertEqual(finding["evidence"]["live_ready_lanes"], 0)

    def test_unattributable_close_gate_ablation_is_p1_assumption_hole(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["ai_backtest"].write_text(
                json.dumps(
                    {
                        "mechanism_ablation": {
                            "close_gate_ab": {
                                "status": "MEASURED",
                                "close_events": 8,
                                "bot_attributed_close_events": 0,
                                "gateway_close_sent_events": 0,
                                "loss_side_market_close_count": 5,
                                "loss_side_market_close_net_jpy": -1200.0,
                                "broker_accepted_without_gateway_loss_side_market_close_count": 2,
                                "broker_accepted_without_gateway_loss_side_market_close_source_counts": {
                                    "DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE": 2
                                },
                                "broker_accepted_without_gateway_loss_side_market_close_evidence_counts": {
                                    "NO_LOCAL_GATEWAY_CLOSE_RECEIPT": 2,
                                    "NO_CLIENT_EXTENSION": 2,
                                },
                                "unattributed_loss_side_market_close_count": 5,
                            }
                        }
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertIn("CLOSE_GATE_ABLATION_NOT_ATTRIBUTABLE", codes)
        finding = codes["CLOSE_GATE_ABLATION_NOT_ATTRIBUTABLE"]
        self.assertEqual(finding["priority"], "P1")
        self.assertEqual(finding["evidence"]["gateway_close_sent_events"], 0)
        self.assertIn("direct/manual", finding["next_action"])
        self.assertEqual(
            finding["evidence"]["broker_accepted_without_gateway_loss_side_market_close_source_counts"],
            {"DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE": 2},
        )
        self.assertEqual(
            finding["evidence"]["broker_accepted_without_gateway_loss_side_market_close_evidence_counts"],
            {"NO_LOCAL_GATEWAY_CLOSE_RECEIPT": 2, "NO_CLIENT_EXTENSION": 2},
        )

    def test_close_gate_ablation_with_trader_entry_source_requests_receipt_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["ai_backtest"].write_text(
                json.dumps(
                    {
                        "mechanism_ablation": {
                            "close_gate_ab": {
                                "status": "MEASURED",
                                "close_events": 8,
                                "bot_attributed_close_events": 8,
                                "gateway_close_sent_events": 0,
                                "loss_side_market_close_count": 5,
                                "loss_side_market_close_net_jpy": -1200.0,
                                "broker_accepted_without_gateway_loss_side_market_close_count": 3,
                                "broker_accepted_without_gateway_loss_side_market_close_source_counts": {
                                    "TRADER_ENTRY_LANE_ID": 2,
                                    "DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE": 1,
                                },
                                "broker_accepted_without_gateway_loss_side_market_close_evidence_counts": {
                                    "NO_LOCAL_GATEWAY_CLOSE_RECEIPT": 3,
                                    "TRADER_ENTRY_LANE_ID": 2,
                                    "NO_CLIENT_EXTENSION": 1,
                                },
                                "unattributed_loss_side_market_close_count": 0,
                            }
                        }
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        finding = codes["CLOSE_GATE_ABLATION_NOT_ATTRIBUTABLE"]
        self.assertIn("trader-owned entries", finding["next_action"])
        self.assertIn("GATEWAY_TRADE_CLOSE_SENT", finding["next_action"])
        self.assertIn("1 residual direct/manual close", finding["next_action"])

    def test_external_manual_close_residual_does_not_keep_close_gate_ablation_p1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["ai_backtest"].write_text(
                json.dumps(
                    {
                        "mechanism_ablation": {
                            "close_gate_ab": {
                                "status": "MEASURED",
                                "close_events": 8,
                                "bot_attributed_close_events": 8,
                                "gateway_close_sent_events": 4,
                                "broker_trade_close_accept_events": 4,
                                "loss_side_market_close_count": 5,
                                "loss_side_market_close_net_jpy": -1200.0,
                                "broker_accepted_without_gateway_loss_side_market_close_count": 1,
                                "broker_accepted_without_gateway_loss_side_market_close_source_counts": {
                                    "DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE": 1,
                                },
                                "broker_accepted_without_gateway_loss_side_market_close_evidence_counts": {
                                    "NO_LOCAL_GATEWAY_CLOSE_RECEIPT": 1,
                                    "NO_CLIENT_EXTENSION": 1,
                                },
                                "broker_accepted_without_gateway_policy_gap_loss_side_market_close_count": 0,
                                "broker_accepted_without_gateway_policy_gap_loss_side_market_close_source_counts": {},
                                "broker_accepted_without_gateway_policy_gap_loss_side_market_close_evidence_counts": {},
                                "broker_accepted_without_gateway_external_loss_side_market_close_count": 1,
                                "broker_accepted_without_gateway_external_loss_side_market_close_source_counts": {
                                    "DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE": 1,
                                },
                                "unattributed_loss_side_market_close_count": 0,
                            }
                        }
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertNotIn("CLOSE_GATE_ABLATION_NOT_ATTRIBUTABLE", codes)

    def test_legacy_review_exit_close_ablation_remains_p1_assumption_hole(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["ai_backtest"].write_text(
                json.dumps(
                    {
                        "mechanism_ablation": {
                            "close_gate_ab": {
                                "status": "MEASURED",
                                "close_events": 8,
                                "bot_attributed_close_events": 8,
                                "gateway_close_sent_events": 3,
                                "broker_trade_close_accept_events": 3,
                                "loss_side_market_close_count": 3,
                                "loss_side_market_close_net_jpy": -900.0,
                                "gateway_gpt_close_loss_side_market_close_count": 0,
                                "gateway_review_exit_loss_side_market_close_count": 3,
                                "gateway_review_exit_loss_side_market_close_net_jpy": -900.0,
                                "broker_accepted_without_gateway_loss_side_market_close_count": 0,
                                "unattributed_loss_side_market_close_count": 0,
                            }
                        }
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertIn("LEGACY_REVIEW_EXIT_CLOSE_DRAG", codes)
        finding = codes["LEGACY_REVIEW_EXIT_CLOSE_DRAG"]
        self.assertIn("legacy REVIEW_EXIT", finding["next_action"])
        self.assertEqual(finding["evidence"]["gateway_review_exit_loss_side_market_close_count"], 3)

    def test_historical_review_exit_close_drag_is_p2_when_no_24h_losses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["ai_backtest"].write_text(
                json.dumps(
                    {
                        "mechanism_ablation": {
                            "close_gate_ab": {
                                "status": "MEASURED",
                                "close_events": 8,
                                "bot_attributed_close_events": 8,
                                "gateway_close_sent_events": 3,
                                "broker_trade_close_accept_events": 3,
                                "loss_side_market_close_count": 3,
                                "loss_side_market_close_net_jpy": -900.0,
                                "gateway_gpt_close_loss_side_market_close_count": 0,
                                "gateway_review_exit_loss_side_market_close_count": 3,
                                "gateway_review_exit_loss_side_market_close_net_jpy": -900.0,
                                "gateway_review_exit_recent_24h_loss_side_market_close_count": 0,
                                "gateway_review_exit_recent_24h_loss_side_market_close_net_jpy": 0.0,
                                "gateway_review_exit_recent_7d_loss_side_market_close_count": 1,
                                "gateway_review_exit_recent_7d_loss_side_market_close_net_jpy": -172.0,
                                "gateway_review_exit_latest_loss_side_market_close_ts_utc": (
                                    "2026-05-14T14:44:38+00:00"
                                ),
                                "broker_accepted_without_gateway_loss_side_market_close_count": 0,
                                "unattributed_loss_side_market_close_count": 0,
                            }
                        }
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertNotIn("LEGACY_REVIEW_EXIT_CLOSE_DRAG", codes)
        finding = codes["LEGACY_REVIEW_EXIT_HISTORICAL_DRAG"]
        self.assertEqual(finding["priority"], "P2")
        self.assertEqual(finding["evidence"]["gateway_review_exit_recent_24h_loss_side_market_close_count"], 0)
        self.assertEqual(finding["evidence"]["gateway_review_exit_recent_7d_loss_side_market_close_count"], 1)

    def test_profitable_backtest_edges_missing_from_coverage_are_p1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["coverage"].write_text(
                json.dumps(
                    {
                        "artifact_diagnostics": {
                            "profitable_bucket_coverage": {
                                "source_status": "RESEARCH_PROFITABLE_NOT_CERTIFIED",
                                "live_permission": False,
                                "positive_pair_directions": 2,
                                "positive_managed_net_jpy": 1500.0,
                                "state_counts": {
                                    "NO_CURRENT_LANE": 1,
                                    "SPREAD_NORMALIZED_LIVE_BLOCKED": 1,
                                },
                                "blocked_or_missing_top": [
                                    {
                                        "pair": "USD_CAD",
                                        "direction": "LONG",
                                        "coverage_state": "NO_CURRENT_LANE",
                                        "managed_net_jpy": 900.0,
                                        "raw_net_jpy": 800.0,
                                        "trades": 10,
                                        "days": 3,
                                        "current_lane_count": 0,
                                        "spread_normalized_candidate_count": 0,
                                        "spread_normalized_no_live_blocker_count": 0,
                                        "top_blockers": [],
                                        "strategy_profile_status": "MINE_MISSED_EDGE",
                                        "strategy_profile_required_fix": "candidate not surfaced",
                                        "strategy_profile_blocks_live": True,
                                        "matrix_support_count": 8,
                                        "matrix_reject_count": 1,
                                        "matrix_warning_count": 2,
                                        "matrix_strongest_support": "USD_CAD DXY and oil context align LONG",
                                        "matrix_strongest_reject": "spread stressed",
                                        "matrix_cross_asset_context": [
                                            "GOLD_CONTEXT_TECHNICAL_DIRECTION: XAU_USD maps to LONG",
                                            "OIL_CONTEXT_TECHNICAL_DIRECTION: WTICO_USD maps to LONG",
                                        ],
                                        "matrix_support_context": [
                                            "GOLD_CONTEXT_TECHNICAL_DIRECTION: XAU_USD maps to LONG",
                                            "OIL_CONTEXT_TECHNICAL_DIRECTION: WTICO_USD maps to LONG",
                                        ],
                                        "matrix_reject_context": [],
                                        "same_side_matrix_context_supported": True,
                                    },
                                    {
                                        "pair": "EUR_USD",
                                        "direction": "SHORT",
                                        "coverage_state": "SPREAD_NORMALIZED_NO_LIVE_BLOCKER",
                                        "managed_net_jpy": 600.0,
                                        "current_lane_count": 1,
                                    },
                                ],
                            }
                        }
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertIn("PROFITABLE_BACKTEST_EDGE_COVERAGE_GAP", codes)
        self.assertIn("MARKET_CONTEXT_SUPPORTED_EDGE_NOT_ACTIONABLE", codes)
        gap = codes["PROFITABLE_BACKTEST_EDGE_COVERAGE_GAP"]
        self.assertEqual(gap["priority"], "P1")
        self.assertEqual(gap["evidence"]["blocked_edges"][0]["pair"], "USD_CAD")
        supported = codes["MARKET_CONTEXT_SUPPORTED_EDGE_NOT_ACTIONABLE"]
        self.assertIn("GOLD_CONTEXT_TECHNICAL_DIRECTION", supported["evidence"]["supported_edges"][0]["matrix_cross_asset_context"][0])

    def test_forecast_gated_profitable_edges_do_not_become_coverage_p1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["coverage"].write_text(
                json.dumps(
                    {
                        "artifact_diagnostics": {
                            "profitable_bucket_coverage": {
                                "source_status": "RESEARCH_PROFITABLE_NOT_CERTIFIED",
                                "live_permission": False,
                                "positive_pair_directions": 1,
                                "positive_managed_net_jpy": 900.0,
                                "state_counts": {"SURFACED_BUT_BLOCKED": 1},
                                "blocked_or_missing_top": [
                                    {
                                        "pair": "EUR_USD",
                                        "direction": "LONG",
                                        "coverage_state": "SURFACED_BUT_BLOCKED",
                                        "managed_net_jpy": 900.0,
                                        "raw_net_jpy": 800.0,
                                        "trades": 10,
                                        "days": 3,
                                        "current_lane_count": 4,
                                        "spread_normalized_candidate_count": 0,
                                        "spread_normalized_no_live_blocker_count": 0,
                                        "top_blockers": [
                                            "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
                                            "EUR_USD LONG forecast RANGE confidence 0.38 < 0.55",
                                        ],
                                        "strategy_profile_status": "CANDIDATE",
                                        "strategy_profile_required_fix": "eligible but forecast blocked",
                                        "strategy_profile_blocks_live": False,
                                        "matrix_support_count": 8,
                                        "matrix_reject_count": 1,
                                        "matrix_support_context": [
                                            "GOLD_CONTEXT_TECHNICAL_DIRECTION: XAU_USD maps to LONG",
                                        ],
                                        "same_side_matrix_context_supported": True,
                                    }
                                ],
                            }
                        }
                    }
                )
            )

            _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertNotIn("PROFITABLE_BACKTEST_EDGE_COVERAGE_GAP", codes)
        self.assertNotIn("MARKET_CONTEXT_SUPPORTED_EDGE_NOT_ACTIONABLE", codes)
        finding = codes["PROFITABLE_BACKTEST_EDGE_FORECAST_GATED"]
        self.assertEqual(finding["priority"], "P2")
        self.assertEqual(finding["evidence"]["forecast_gated_edges"][0]["pair"], "EUR_USD")

    def test_strategy_gated_profitable_edges_do_not_become_coverage_p1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["coverage"].write_text(
                json.dumps(
                    {
                        "artifact_diagnostics": {
                            "profitable_bucket_coverage": {
                                "source_status": "RESEARCH_PROFITABLE_NOT_CERTIFIED",
                                "live_permission": False,
                                "positive_pair_directions": 1,
                                "positive_managed_net_jpy": 700.0,
                                "state_counts": {"NO_CURRENT_LANE": 1},
                                "blocked_or_missing_top": [
                                    {
                                        "pair": "USD_JPY",
                                        "direction": "LONG",
                                        "coverage_state": "NO_CURRENT_LANE",
                                        "managed_net_jpy": 700.0,
                                        "raw_net_jpy": 650.0,
                                        "trades": 7,
                                        "days": 2,
                                        "current_lane_count": 0,
                                        "spread_normalized_candidate_count": 0,
                                        "spread_normalized_no_live_blocker_count": 0,
                                        "top_blockers": [],
                                        "strategy_profile_status": "BLOCK_UNTIL_NEW_EVIDENCE",
                                        "strategy_profile_required_fix": "current evidence required",
                                        "strategy_profile_blocks_live": True,
                                        "matrix_support_count": 6,
                                        "matrix_reject_count": 0,
                                        "matrix_support_context": [
                                            "DXY_CONTEXT_TECHNICAL_DIRECTION: DXY maps to LONG",
                                        ],
                                        "same_side_matrix_context_supported": True,
                                    }
                                ],
                            }
                        }
                    }
                )
            )

            _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertNotIn("PROFITABLE_BACKTEST_EDGE_COVERAGE_GAP", codes)
        self.assertNotIn("MARKET_CONTEXT_SUPPORTED_EDGE_NOT_ACTIONABLE", codes)
        finding = codes["PROFITABLE_BACKTEST_EDGE_STRATEGY_GATED"]
        self.assertEqual(finding["priority"], "P2")
        self.assertEqual(finding["evidence"]["strategy_gated_edges"][0]["pair"], "USD_JPY")

    def test_lane_only_verification_blockers_do_not_mask_opportunity_hole(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, verification_lane_blockers=True)

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertEqual(summary.p0_findings, 1)
        self.assertIn("TARGET_OPEN_NO_LIVE_READY_LANES", codes)
        self.assertIn("VERIFICATION_LEDGER_LANE_BLOCKERS_RECORDED", codes)
        self.assertNotIn("VERIFICATION_LEDGER_BLOCKED", codes)

    def test_pending_entry_downgrades_no_live_ready_hole_from_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, pending_entry=True)

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertEqual(summary.p0_findings, 0)
        self.assertEqual(codes["TARGET_OPEN_NO_LIVE_READY_LANES"]["priority"], "P1")
        self.assertEqual(
            codes["TARGET_OPEN_NO_LIVE_READY_LANES"]["evidence"]["trader_pending_entry_orders"][0]["order_id"],
            "P1",
        )
        self.assertEqual(payload["runtime"]["open_trader_pending_entries"], 1)

    def test_market_evidence_refresh_downgrades_no_live_ready_hole_from_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False)
            files["coverage"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "status": "COVERAGE_GAP",
                        "artifact_diagnostics": {
                            "requires_market_evidence_refresh": True,
                            "all_lanes_spread_blocked": True,
                            "all_lanes_quote_stale": False,
                            "quote_stale_result_count": 8,
                            "spread_normalized_candidate_count": 2,
                            "spread_normalized_candidate_reward_jpy": 2376.0,
                        },
                        "action_items": [
                            "refresh broker-snapshot and generate-intents after quotes and spreads are tradable",
                        ],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        finding = codes["TARGET_OPEN_NO_LIVE_READY_LANES"]
        evidence = finding["evidence"]["coverage_market_evidence_refresh"]
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertEqual(summary.p0_findings, 0)
        self.assertEqual(finding["priority"], "P1")
        self.assertTrue(evidence["requires_market_evidence_refresh"])
        self.assertTrue(evidence["all_lanes_spread_blocked"])
        self.assertEqual(evidence["spread_normalized_candidate_count"], 2)

    def test_no_live_ready_evidence_names_dry_run_passed_live_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False)
            files["intents"].write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "lane_id": "failure_trader:AUD_CAD:SHORT:BREAKOUT_FAILURE:LIMIT",
                                "status": "DRY_RUN_PASSED",
                                "intent": {
                                    "side": "SHORT",
                                    "order_type": "LIMIT",
                                    "metadata": {
                                        "opportunity_mode": "HARVEST",
                                        "opportunity_mode_reason": "tp_target_intent=HARVEST",
                                        "opportunity_mode_reward_risk": 1.18,
                                        "tp_target_intent": "HARVEST",
                                        "forecast_direction": "DOWN",
                                        "forecast_confidence": 0.311,
                                        "forecast_raw_confidence": 0.5244919527711897,
                                        "chart_direction_bias": "LONG",
                                        "forecast_market_support": {
                                            "aligned_projection_count": 1,
                                            "best_hit_rate": 0.82,
                                            "best_samples": 100,
                                            "direction": "DOWN",
                                            "ok": True,
                                            "reason": (
                                                "liquidity_sweep_high DOWN hit_rate=0.82 "
                                                "samples=100 supports weak calibrated forecast"
                                            ),
                                            "signals": [
                                                {
                                                    "confidence": 0.9922,
                                                    "direction": "DOWN",
                                                    "hit_rate": 0.82,
                                                    "name": "liquidity_sweep_high",
                                                    "samples": 100,
                                                    "timeframe": "M15",
                                                }
                                            ],
                                            "timing_projection_count": 0,
                                            "unselected_reason": "",
                                        },
                                    },
                                },
                                "risk_issues": [
                                    {
                                        "code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
                                        "message": "forecast confidence below live floor",
                                        "severity": "WARN",
                                    }
                                ],
                                "strategy_issues": [
                                    {
                                        "code": "STRATEGY_PROFILE_MISSING",
                                        "message": "dry-run advisory profile warning",
                                        "severity": "WARN",
                                    }
                                ],
                                "live_strategy_issues": [
                                    {
                                        "code": "STRATEGY_NOT_ELIGIBLE",
                                        "message": "strategy profile is not live eligible",
                                        "severity": "WARN",
                                    }
                                ],
                                "live_blockers": [],
                                "risk_metrics": {
                                    "reward_jpy": 420.0,
                                    "reward_risk": 1.18,
                                },
                            }
                        ]
                    }
                )
            )
            files["coverage"].write_text(
                json.dumps(
                    {
                        "opportunity_modes": {
                            "HARVEST": {
                                "lanes": 1,
                                "live_ready_lanes": 0,
                                "promotion_candidate_lanes": 0,
                                "reward_jpy": 420.0,
                                "live_ready_reward_jpy": 0.0,
                                "potential_reward_jpy": 0.0,
                                "top_issue_codes": [{"code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", "count": 1}],
                                "top_live_blocker_codes": [
                                    {"code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", "count": 1}
                                ],
                                "top_blockers": [{"label": "forecast confidence below live floor", "count": 1}],
                            },
                            "RUNNER": {
                                "lanes": 0,
                                "live_ready_lanes": 0,
                                "promotion_candidate_lanes": 0,
                                "reward_jpy": 0.0,
                            },
                        },
                        "runner_candidate_diagnostics": {
                            "status": "RUNNER_CANDIDATES_DEMOTED_TO_HARVEST",
                            "trend_candidate_lanes": 4,
                            "runner_qualified_lanes": 0,
                            "attached_harvest_lanes": 4,
                            "top_demotion_reasons": [
                                {
                                    "reason": "RANGE regime is not a clean runner trend",
                                    "count": 3,
                                }
                            ],
                            "top_issue_codes": [
                                {
                                    "code": "FORECAST_WATCH_ONLY",
                                    "count": 1,
                                }
                            ],
                            "top_live_blocker_codes": [
                                {
                                    "code": "TREND_MARKET_NOT_OPERATING_TREND",
                                    "count": 2,
                                }
                            ],
                        },
                        "perspective_alignment_diagnostics": {
                            "status": "RANGE_METHOD_MISMATCH_REPAIR_REQUIRED",
                            "pair_direction_groups": 4,
                            "range_forecast_method_mismatch_groups": 1,
                            "range_forecast_method_mismatch_lanes": 2,
                            "range_forecast_method_mismatch_top": [
                                {
                                    "pair": "EUR_USD",
                                    "direction": "SHORT",
                                    "method_mismatch_lanes": 2,
                                    "method_mismatch_reward_jpy": 2800.0,
                                    "range_rotation_lanes": 1,
                                    "range_rotation_live_ready_lanes": 0,
                                    "range_rotation_top_live_blocker_codes": [
                                        {"code": "RANGE_ROTATION_BROADER_LOCATION_CHASE", "count": 1}
                                    ],
                                    "top_live_blocker_codes": [
                                        {"code": "RANGE_FORECAST_REQUIRES_RANGE_ROTATION", "count": 2}
                                    ],
                                }
                            ],
                        },
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())
            report_text = files["report"].read_text()

        codes = {item["code"]: item for item in payload["findings"]}
        evidence = codes["TARGET_OPEN_NO_LIVE_READY_LANES"]["evidence"]
        dry_run_blockers = {item["message"]: item for item in evidence["dry_run_passed_live_readiness_blockers"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertEqual(evidence["status_counts"]["DRY_RUN_PASSED"], 1)
        self.assertEqual(evidence["opportunity_modes"]["HARVEST"]["lanes"], 1)
        self.assertEqual(evidence["opportunity_modes"]["HARVEST"]["reward_jpy"], 420.0)
        self.assertEqual(evidence["opportunity_modes"]["RUNNER"]["top_issue_codes"][0]["code"], "FORECAST_WATCH_ONLY")
        self.assertEqual(
            evidence["opportunity_modes"]["RUNNER"]["top_live_blocker_codes"][0]["code"],
            "TREND_MARKET_NOT_OPERATING_TREND",
        )
        runner_diagnostics = evidence["runner_candidate_diagnostics"]
        self.assertEqual(runner_diagnostics["status"], "RUNNER_CANDIDATES_DEMOTED_TO_HARVEST")
        self.assertEqual(runner_diagnostics["trend_candidate_lanes"], 4)
        self.assertEqual(runner_diagnostics["runner_qualified_lanes"], 0)
        self.assertEqual(runner_diagnostics["top_demotion_reasons"][0]["reason"], "RANGE regime is not a clean runner trend")
        self.assertEqual(runner_diagnostics["top_issue_codes"][0]["code"], "FORECAST_WATCH_ONLY")
        self.assertEqual(
            runner_diagnostics["top_live_blocker_codes"][0]["code"],
            "TREND_MARKET_NOT_OPERATING_TREND",
        )
        perspective_diagnostics = evidence["perspective_alignment_diagnostics"]
        self.assertEqual(perspective_diagnostics["status"], "RANGE_METHOD_MISMATCH_REPAIR_REQUIRED")
        self.assertEqual(perspective_diagnostics["range_forecast_method_mismatch_lanes"], 2)
        self.assertEqual(
            perspective_diagnostics["range_forecast_method_mismatch_top"][0]["range_rotation_top_live_blocker_codes"][0]["code"],
            "RANGE_ROTATION_BROADER_LOCATION_CHASE",
        )
        self.assertIn("runner candidates", report_text)
        self.assertIn("perspective alignment", report_text)
        self.assertIn("opportunity modes", report_text)
        self.assertIn("reward=`420.0`", report_text)
        self.assertIn("live_codes=`TREND_MARKET_NOT_OPERATING_TREND`", report_text)
        self.assertIn("RANGE_METHOD_MISMATCH_REPAIR_REQUIRED", report_text)
        self.assertIn("EUR_USD SHORT mismatch=2", report_text)
        self.assertIn("dry-run blocker families", report_text)
        self.assertIn("nearest dry-run lanes", report_text)
        self.assertIn("forecast gate reasons", report_text)
        self.assertIn("RUNNER_CANDIDATES_DEMOTED_TO_HARVEST", report_text)
        self.assertIn("RANGE regime is not a clean runner trend=3", report_text)
        self.assertIn("failure_trader:AUD_CAD:SHORT:BREAKOUT_FAILURE:LIMIT", report_text)
        self.assertIn("HARVEST", report_text)
        self.assertIn("forecast=1", report_text)
        self.assertIn("rr=`1.180`", report_text)
        self.assertIn("reward=`420.000`", report_text)
        self.assertIn("liquidity_sweep_high DOWN", report_text)
        self.assertEqual(dry_run_blockers["FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE"]["count"], 1)
        self.assertEqual(dry_run_blockers["STRATEGY_NOT_ELIGIBLE"]["count"], 1)
        self.assertNotIn("STRATEGY_PROFILE_MISSING", dry_run_blockers)
        forecast_diagnostics = evidence["dry_run_passed_forecast_gate_diagnostics"]
        self.assertEqual(forecast_diagnostics["reason_counts"][0]["count"], 1)
        self.assertIn("liquidity_sweep_high DOWN", forecast_diagnostics["reason_counts"][0]["reason"])
        lane_diagnostic = forecast_diagnostics["lanes"][0]
        self.assertEqual(lane_diagnostic["lane_id"], "failure_trader:AUD_CAD:SHORT:BREAKOUT_FAILURE:LIMIT")
        self.assertEqual(lane_diagnostic["opportunity_mode"], "HARVEST")
        self.assertEqual(lane_diagnostic["opportunity_mode_reward_risk"], 1.18)
        self.assertEqual(lane_diagnostic["tp_target_intent"], "HARVEST")
        self.assertEqual(lane_diagnostic["chart_direction_bias"], "LONG")
        self.assertEqual(lane_diagnostic["forecast_confidence"], 0.311)
        self.assertTrue(lane_diagnostic["forecast_market_support_ok"])
        self.assertEqual(lane_diagnostic["forecast_market_support_best_hit_rate"], 0.82)
        self.assertEqual(lane_diagnostic["forecast_market_support_top_signal"]["name"], "liquidity_sweep_high")

    def test_no_live_ready_strips_stale_coverage_blocker_codes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, closed_pls=(100.0, 80.0, -50.0))
            coverage_ts = _NOW - timedelta(minutes=30)
            files["intents"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "results": [
                            {
                                "lane_id": "range_trader:EUR_USD:SHORT:RANGE_ROTATION",
                                "status": "DRY_RUN_BLOCKED",
                                "intent": {
                                    "pair": "EUR_USD",
                                    "side": "SHORT",
                                    "order_type": "LIMIT",
                                    "metadata": {"opportunity_mode": "HARVEST"},
                                },
                                "risk_metrics": {"reward_jpy": 420.0, "reward_risk": 1.18},
                                "risk_issues": [
                                    {
                                        "code": "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
                                        "severity": "BLOCK",
                                        "message": "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
                                    }
                                ],
                                "live_strategy_issues": [],
                                "live_blockers": ["BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE"],
                            }
                        ],
                    }
                )
            )
            files["coverage"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": coverage_ts.isoformat(),
                        "opportunity_modes": {
                            "HARVEST": {
                                "lanes": 1,
                                "live_ready_lanes": 0,
                                "reward_jpy": 420.0,
                                "top_issue_codes": [
                                    {"code": "POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE", "count": 1}
                                ],
                                "top_live_blocker_codes": [
                                    {"code": "POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE", "count": 1}
                                ],
                                "top_blockers": [
                                    {
                                        "label": "position guardian inactive P0 blocks LIVE_READY intent generation",
                                        "count": 1,
                                    }
                                ],
                            }
                        },
                        "runner_candidate_diagnostics": {
                            "status": "RUNNER_CANDIDATES_DEMOTED_TO_HARVEST",
                            "trend_candidate_lanes": 1,
                            "runner_qualified_lanes": 0,
                            "top_issue_codes": [
                                {"code": "POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE", "count": 1}
                            ],
                            "top_live_blocker_codes": [
                                {"code": "POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE", "count": 1}
                            ],
                            "top_blockers": [
                                {
                                    "label": "position guardian inactive P0 blocks LIVE_READY intent generation",
                                    "count": 1,
                                }
                            ],
                        },
                        "artifact_diagnostics": {
                            "profitable_bucket_coverage": {
                                "source_status": "OK",
                                "live_permission": False,
                                "positive_pair_directions": 1,
                                "positive_managed_net_jpy": 120.0,
                                "state_counts": {"BLOCKED": 1},
                                "blocked_or_missing_top": [
                                    {
                                        "pair": "EUR_USD",
                                        "direction": "SHORT",
                                        "coverage_state": "BLOCKED",
                                        "managed_net_jpy": 120.0,
                                        "top_blockers": [
                                            "POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE",
                                            "position guardian inactive P0 blocks LIVE_READY intent generation",
                                        ],
                                        "matrix_support_count": 1,
                                        "matrix_reject_count": 0,
                                        "matrix_support_context": ["EUR_USD short_score leads"],
                                    }
                                ],
                            }
                        },
                    }
                )
            )

            with mock.patch(
                "quant_rabbit.self_improvement_audit._position_guardian_runtime_status",
                return_value={"required": False, "active": False},
            ):
                summary = _run(files)
            payload = json.loads(files["output"].read_text())
            report_text = files["report"].read_text()

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        evidence = codes["TARGET_OPEN_NO_LIVE_READY_LANES"]["evidence"]
        self.assertEqual(
            evidence["coverage_source_freshness"]["status"],
            "STALE_AGAINST_ORDER_INTENTS",
        )
        self.assertEqual(evidence["opportunity_modes"]["HARVEST"]["top_issue_codes"], [])
        self.assertEqual(evidence["opportunity_modes"]["HARVEST"]["top_live_blocker_codes"], [])
        self.assertEqual(evidence["opportunity_modes"]["HARVEST"]["top_blockers"], [])
        self.assertEqual(evidence["runner_candidate_diagnostics"]["top_issue_codes"], [])
        self.assertEqual(evidence["runner_candidate_diagnostics"]["top_live_blocker_codes"], [])
        self.assertEqual(
            evidence["non_live_ready_live_readiness_blockers"][0]["message"],
            "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
        )
        coverage_gap = codes["PROFITABLE_BACKTEST_EDGE_COVERAGE_GAP"]
        self.assertEqual(
            coverage_gap["evidence"]["coverage_source_freshness"]["status"],
            "STALE_AGAINST_ORDER_INTENTS",
        )
        self.assertEqual(coverage_gap["evidence"]["blocked_edges"][0]["top_blockers"], [])
        self.assertEqual(
            coverage_gap["evidence"]["blocked_edges"][0]["top_blockers_status"],
            "STALE_AGAINST_ORDER_INTENTS",
        )
        payload_text = json.dumps(payload)
        self.assertNotIn("POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE", payload_text)
        self.assertNotIn("position guardian inactive", payload_text)
        self.assertNotIn("POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE", report_text)

    def test_same_side_unselected_projection_arbitration_becomes_forecast_repair_finding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False)
            files["intents"].write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "lane_id": "range_trader:EUR_JPY:LONG:RANGE_ROTATION",
                                "status": "DRY_RUN_PASSED",
                                "intent": {
                                    "pair": "EUR_JPY",
                                    "side": "LONG",
                                    "order_type": "LIMIT",
                                    "metadata": {
                                        "forecast_direction": "RANGE",
                                        "forecast_confidence": 0.944,
                                        "forecast_raw_confidence": 0.821,
                                        "chart_direction_bias": "LONG",
                                        "forecast_market_support": {
                                            "ok": False,
                                            "direction": "RANGE",
                                            "reason": "forecast RANGE has no executable direction; audited projection unselected",
                                            "unselected_projection_count": 2,
                                            "unselected_reason": (
                                                "liquidity_sweep_low UP audited hit_rate=1.00 "
                                                "samples=40 was unselected because forecast=RANGE"
                                            ),
                                            "unselected_signals": [
                                                {
                                                    "confidence": 0.9918,
                                                    "direction": "UP",
                                                    "hit_rate": 1.0,
                                                    "economic_hit_rate": 1.0,
                                                    "economic_samples": 40,
                                                    "name": "liquidity_sweep_low",
                                                    "samples": 40,
                                                    "target_pips": 5.0,
                                                    "timeframe": "M15",
                                                    "rationale": "sell-side sweep target, fade LONG",
                                                },
                                                {
                                                    "confidence": 0.8123,
                                                    "direction": "UP",
                                                    "hit_rate": 0.78,
                                                    "name": "liquidity_sweep_low",
                                                    "samples": 18,
                                                    "timeframe": "M30",
                                                    "rationale": "higher-timeframe sell-side sweep target",
                                                }
                                            ],
                                        },
                                    },
                                },
                                "risk_issues": [],
                                "strategy_issues": [],
                                "live_strategy_issues": [],
                                "live_blockers": [],
                                "risk_metrics": {
                                    "reward_jpy": 1200.0,
                                    "reward_risk": 1.2,
                                },
                            }
                        ]
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())
            report_text = files["report"].read_text()

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertIn("TARGET_OPEN_NO_LIVE_READY_LANES", codes)
        finding = codes["FORECAST_ARBITRATION_UNSELECTED_PROJECTION_REPAIR_REQUIRED"]
        self.assertEqual(finding["priority"], "P1")
        diagnostics = finding["evidence"]["forecast_arbitration_diagnostics"]
        self.assertEqual(diagnostics["lane_count"], 1)
        self.assertEqual(diagnostics["same_side_lane_count"], 1)
        self.assertEqual(diagnostics["same_side_actionable_repair_lane_count"], 1)
        self.assertEqual(diagnostics["same_side_context_blocked_lane_count"], 0)
        self.assertEqual(diagnostics["opposite_side_lane_count"], 0)
        self.assertEqual(diagnostics["relation_counts"][0]["relation"], "same_side")
        self.assertEqual(diagnostics["direction_counts"][0]["direction"], "UP")
        self.assertEqual(diagnostics["signal_counts"][0]["signal"], "liquidity_sweep_low:UP")
        self.assertEqual(diagnostics["lanes"][0]["pair"], "EUR_JPY")
        self.assertEqual(diagnostics["lanes"][0]["top_unselected_signal_side"], "LONG")
        self.assertEqual(diagnostics["lanes"][0]["top_unselected_signal_relation"], "same_side")
        self.assertEqual(diagnostics["lanes"][0]["top_unselected_signal"]["hit_rate"], 1.0)
        no_live_ready_evidence = codes["TARGET_OPEN_NO_LIVE_READY_LANES"]["evidence"]
        self.assertEqual(no_live_ready_evidence["forecast_arbitration_diagnostics"]["lane_count"], 1)
        self.assertIn("forecast arbitration", report_text)
        self.assertIn("relations=`same_side=1`", report_text)
        self.assertIn("EUR_JPY LONG->liquidity_sweep_low UP", report_text)

    def test_same_side_unselected_projection_below_live_precision_waits_for_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False)
            files["intents"].write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "lane_id": "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT",
                                "status": "DRY_RUN_PASSED",
                                "live_blocker_codes": [
                                    "FORECAST_NOT_EXECUTABLE_FOR_LIVE",
                                ],
                                "intent": {
                                    "pair": "EUR_USD",
                                    "side": "SHORT",
                                    "order_type": "LIMIT",
                                    "metadata": {
                                        "forecast_direction": "UNCLEAR",
                                        "forecast_confidence": 0.1792,
                                        "forecast_raw_confidence": 0.1792,
                                        "chart_direction_bias": "SHORT",
                                        "forecast_market_support": {
                                            "ok": False,
                                            "direction": "UNCLEAR",
                                            "reason": (
                                                "forecast UNCLEAR has no executable direction; "
                                                "audited projection unselected"
                                            ),
                                            "unselected_projection_count": 1,
                                            "unselected_reason": (
                                                "macro_event_nowcast_inflation DOWN audited "
                                                "hit_rate=0.75 samples=100 was unselected because "
                                                "forecast=UNCLEAR"
                                            ),
                                            "unselected_signals": [
                                                {
                                                    "calibration_name": "macro_event_nowcast_inflation_down",
                                                    "confidence": 0.79,
                                                    "direction": "DOWN",
                                                    "economic_hit_rate": 0.75,
                                                    "economic_samples": 100,
                                                    "hit_rate": 0.75,
                                                    "name": "macro_event_nowcast_inflation",
                                                    "samples": 100,
                                                    "timeout_rate": 0.0,
                                                }
                                            ],
                                        },
                                    },
                                },
                                "risk_issues": [
                                    {
                                        "code": "FORECAST_NOT_EXECUTABLE_FOR_LIVE",
                                        "message": "forecast UNCLEAR remains non-executable",
                                        "severity": "BLOCK",
                                    }
                                ],
                                "strategy_issues": [],
                                "live_strategy_issues": [
                                    {
                                        "code": "STRATEGY_NOT_ELIGIBLE",
                                        "message": "risk-resized dry-run receipt keeps strategy profile as advisory only",
                                        "severity": "WARN",
                                    }
                                ],
                                "live_blockers": [],
                                "risk_metrics": {
                                    "reward_jpy": 408.5,
                                    "reward_risk": 1.008,
                                },
                            }
                        ]
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())
            report_text = files["report"].read_text()

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertNotIn("FORECAST_ARBITRATION_UNSELECTED_PROJECTION_REPAIR_REQUIRED", codes)
        finding = codes["FORECAST_ARBITRATION_SAME_SIDE_PRECISION_WAIT"]
        self.assertEqual(finding["priority"], "P2")
        diagnostics = finding["evidence"]["forecast_arbitration_diagnostics"]
        self.assertEqual(diagnostics["same_side_lane_count"], 1)
        self.assertEqual(diagnostics["same_side_actionable_repair_lane_count"], 0)
        self.assertEqual(diagnostics["same_side_context_blocked_lane_count"], 0)
        self.assertEqual(diagnostics["same_side_precision_wait_lane_count"], 1)
        self.assertFalse(
            diagnostics["same_side_precision_wait_lanes"][0]["top_unselected_signal"][
                "live_precision_ok"
            ]
        )
        self.assertIn("same_side_precision_wait=`1`", report_text)

    def test_same_side_unselected_projection_with_context_blockers_is_not_actionable_repair(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False)
            files["intents"].write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "lane_id": "range_trader:EUR_JPY:LONG:RANGE_ROTATION",
                                "status": "DRY_RUN_PASSED",
                                "intent": {
                                    "pair": "EUR_JPY",
                                    "side": "LONG",
                                    "order_type": "LIMIT",
                                    "metadata": {
                                        "forecast_direction": "RANGE",
                                        "forecast_confidence": 0.7,
                                        "forecast_raw_confidence": 0.82,
                                        "chart_direction_bias": "SHORT",
                                        "forecast_market_support": {
                                            "ok": False,
                                            "direction": "RANGE",
                                            "reason": "forecast RANGE has no executable direction; audited projection unselected",
                                            "unselected_projection_count": 1,
                                            "unselected_reason": (
                                                "liquidity_sweep_low UP audited hit_rate=0.78 "
                                                "samples=18 was unselected because forecast=RANGE"
                                            ),
                                            "unselected_signals": [
                                                {
                                                    "confidence": 0.8123,
                                                    "direction": "UP",
                                                    "hit_rate": 0.78,
                                                    "name": "liquidity_sweep_low",
                                                    "samples": 18,
                                                    "timeframe": "M30",
                                                    "rationale": "higher-timeframe sell-side sweep target",
                                                }
                                            ],
                                        },
                                    },
                                },
                                "risk_issues": [
                                    {
                                        "code": "CHART_DIRECTION_CONFLICT",
                                        "message": "chart direction conflicts with entry side",
                                        "severity": "WARN",
                                    }
                                ],
                                "strategy_issues": [],
                                "live_strategy_issues": [
                                    {
                                        "code": "STRATEGY_PROFILE_MISSING",
                                        "message": "EUR_JPY LONG is absent from mined strategy profile",
                                        "severity": "WARN",
                                    }
                                ],
                                "live_blockers": [],
                                "risk_metrics": {
                                    "reward_jpy": 1200.0,
                                    "reward_risk": 1.2,
                                },
                            }
                        ]
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())
            report_text = files["report"].read_text()

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertNotIn("FORECAST_ARBITRATION_UNSELECTED_PROJECTION_REPAIR_REQUIRED", codes)
        finding = codes["FORECAST_ARBITRATION_SAME_SIDE_CONTEXT_BLOCKED"]
        self.assertEqual(finding["priority"], "P2")
        diagnostics = finding["evidence"]["forecast_arbitration_diagnostics"]
        self.assertEqual(diagnostics["same_side_lane_count"], 1)
        self.assertEqual(diagnostics["same_side_actionable_repair_lane_count"], 0)
        self.assertEqual(diagnostics["same_side_context_blocked_lane_count"], 1)
        self.assertEqual(
            diagnostics["same_side_context_blocked_lanes"][0]["context_blocker_families"],
            ["market_structure", "strategy_profile"],
        )
        self.assertEqual(
            diagnostics["same_side_context_blocker_counts"],
            [
                {"family": "market_structure", "count": 1},
                {"family": "strategy_profile", "count": 1},
            ],
        )
        self.assertIn("same_side_context_blocked=`1`", report_text)

    def test_opposite_unselected_projection_arbitration_is_enforced_not_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False)
            files["intents"].write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "lane_id": "range_trader:EUR_JPY:SHORT:RANGE_ROTATION",
                                "status": "DRY_RUN_PASSED",
                                "intent": {
                                    "pair": "EUR_JPY",
                                    "side": "SHORT",
                                    "order_type": "LIMIT",
                                    "metadata": {
                                        "forecast_direction": "RANGE",
                                        "forecast_confidence": 0.944,
                                        "forecast_raw_confidence": 0.821,
                                        "chart_direction_bias": "LONG",
                                        "forecast_market_support": {
                                            "ok": False,
                                            "direction": "RANGE",
                                            "reason": "forecast RANGE has no executable direction; audited projection unselected",
                                            "unselected_projection_count": 1,
                                            "unselected_reason": (
                                                "liquidity_sweep_low UP audited hit_rate=1.00 "
                                                "samples=40 was unselected because forecast=RANGE"
                                            ),
                                            "unselected_signals": [
                                                {
                                                    "confidence": 0.9918,
                                                    "direction": "UP",
                                                    "hit_rate": 1.0,
                                                    "name": "liquidity_sweep_low",
                                                    "samples": 40,
                                                    "timeframe": "M15",
                                                    "rationale": "sell-side sweep target, fade LONG",
                                                }
                                            ],
                                        },
                                    },
                                },
                                "risk_issues": [
                                    {
                                        "code": "FORECAST_RANGE_UNSELECTED_DIRECTION_CONFLICT",
                                        "message": "audited projection conflicts with RANGE lane",
                                        "severity": "WARN",
                                    }
                                ],
                                "strategy_issues": [],
                                "live_strategy_issues": [],
                                "live_blockers": [
                                    {
                                        "code": "FORECAST_RANGE_UNSELECTED_DIRECTION_CONFLICT",
                                        "message": "audited projection conflicts with RANGE lane",
                                        "severity": "BLOCK",
                                    }
                                ],
                                "risk_metrics": {
                                    "reward_jpy": 1200.0,
                                    "reward_risk": 1.2,
                                },
                            }
                        ]
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())
            report_text = files["report"].read_text()

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertNotIn("FORECAST_ARBITRATION_UNSELECTED_PROJECTION_REPAIR_REQUIRED", codes)
        finding = codes["FORECAST_ARBITRATION_OPPOSITE_PROJECTION_CONFLICTS_ENFORCED"]
        self.assertEqual(finding["priority"], "P2")
        diagnostics = finding["evidence"]["forecast_arbitration_diagnostics"]
        self.assertEqual(diagnostics["lane_count"], 1)
        self.assertEqual(diagnostics["same_side_lane_count"], 0)
        self.assertEqual(diagnostics["opposite_side_lane_count"], 1)
        self.assertEqual(diagnostics["relation_counts"][0]["relation"], "opposite_side")
        self.assertEqual(diagnostics["lanes"][0]["top_unselected_signal_relation"], "opposite_side")
        self.assertIn("relations=`opposite_side=1`", report_text)

    def test_mixed_unselected_projection_arbitration_is_not_same_side_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False)
            files["intents"].write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "lane_id": "range_trader:GBP_JPY:SHORT:RANGE_ROTATION",
                                "status": "DRY_RUN_PASSED",
                                "intent": {
                                    "pair": "GBP_JPY",
                                    "side": "SHORT",
                                    "order_type": "LIMIT",
                                    "metadata": {
                                        "forecast_direction": "RANGE",
                                        "forecast_confidence": 0.985,
                                        "forecast_raw_confidence": 0.9,
                                        "chart_direction_bias": "LONG",
                                        "forecast_market_support": {
                                            "ok": False,
                                            "direction": "RANGE",
                                            "reason": "forecast RANGE has no executable direction; audited projection unselected",
                                            "unselected_projection_count": 2,
                                            "unselected_reason": "mixed sweep projections were unselected because forecast=RANGE",
                                            "unselected_signals": [
                                                {
                                                    "confidence": 0.68,
                                                    "direction": "DOWN",
                                                    "hit_rate": 0.825,
                                                    "name": "liquidity_sweep_high",
                                                    "samples": 40,
                                                    "timeframe": "M5",
                                                    "rationale": "buy-side sweep target, fade SHORT",
                                                },
                                                {
                                                    "confidence": 0.875,
                                                    "direction": "UP",
                                                    "hit_rate": 0.667,
                                                    "name": "liquidity_sweep_low",
                                                    "samples": 24,
                                                    "timeframe": "M15",
                                                    "rationale": "sell-side sweep target, fade LONG",
                                                },
                                            ],
                                        },
                                    },
                                },
                                "risk_issues": [
                                    {
                                        "code": "FORECAST_RANGE_UNSELECTED_DIRECTION_CONFLICT",
                                        "message": "audited projection conflicts with RANGE lane",
                                        "severity": "WARN",
                                    }
                                ],
                                "strategy_issues": [],
                                "live_strategy_issues": [],
                                "live_blockers": [
                                    {
                                        "code": "FORECAST_RANGE_UNSELECTED_DIRECTION_CONFLICT",
                                        "message": "audited projection conflicts with RANGE lane",
                                        "severity": "BLOCK",
                                    }
                                ],
                                "risk_metrics": {
                                    "reward_jpy": 1200.0,
                                    "reward_risk": 1.2,
                                },
                            }
                        ]
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())
            report_text = files["report"].read_text()

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertNotIn("FORECAST_ARBITRATION_UNSELECTED_PROJECTION_REPAIR_REQUIRED", codes)
        finding = codes["FORECAST_ARBITRATION_OPPOSITE_PROJECTION_CONFLICTS_ENFORCED"]
        diagnostics = finding["evidence"]["forecast_arbitration_diagnostics"]
        self.assertEqual(diagnostics["lane_count"], 1)
        self.assertEqual(diagnostics["same_side_lane_count"], 0)
        self.assertEqual(diagnostics["opposite_side_lane_count"], 0)
        self.assertEqual(diagnostics["mixed_relation_lane_count"], 1)
        self.assertEqual(diagnostics["opposite_conflict_lane_count"], 1)
        self.assertEqual(diagnostics["relation_counts"][0]["relation"], "mixed_with_opposite")
        self.assertEqual(diagnostics["lanes"][0]["top_unselected_signal_relation"], "mixed_with_opposite")
        self.assertEqual(diagnostics["lanes"][0]["same_side_unselected_signal"]["direction"], "DOWN")
        self.assertEqual(diagnostics["lanes"][0]["opposite_side_unselected_signal"]["direction"], "UP")
        self.assertIn("relations=`mixed_with_opposite=1`", report_text)

    def test_coverage_perspective_mismatch_becomes_self_improvement_finding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False)
            files["coverage"].write_text(
                json.dumps(
                    {
                        "status": "COVERAGE_GAP",
                        "perspective_alignment_diagnostics": {
                            "status": "RANGE_METHOD_MISMATCH_REPAIR_REQUIRED",
                            "pair_direction_groups": 3,
                            "range_forecast_method_mismatch_groups": 1,
                            "range_forecast_method_mismatch_lanes": 5,
                            "range_forecast_method_mismatch_top": [
                                {
                                    "pair": "EUR_USD",
                                    "direction": "SHORT",
                                    "method_mismatch_lanes": 5,
                                    "method_mismatch_reward_jpy": 6500.0,
                                    "range_rotation_lanes": 2,
                                    "range_rotation_live_ready_lanes": 0,
                                    "range_rotation_top_live_blocker_codes": [
                                        {"code": "FORECAST_RANGE_UNSELECTED_DIRECTION_CONFLICT", "count": 2},
                                        {"code": "RANGE_ROTATION_BROADER_LOCATION_CHASE", "count": 2},
                                    ],
                                    "range_rotation_absence_reason": "OPPOSITE_RAIL_SIDE_SURFACED",
                                    "range_rotation_other_side_lanes": 1,
                                    "range_rotation_other_side_directions": [
                                        {"code": "LONG", "count": 1}
                                    ],
                                    "range_rotation_other_side_top_live_blocker_codes": [
                                        {"code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", "count": 1}
                                    ],
                                    "range_rotation_other_side_top_blockers": [
                                        {"label": "opposite rail confidence still below live floor", "count": 1}
                                    ],
                                    "top_live_blocker_codes": [
                                        {"code": "RANGE_FORECAST_REQUIRES_RANGE_ROTATION", "count": 5}
                                    ],
                                }
                            ],
                        },
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())
            report_text = files["report"].read_text()

        codes = {item["code"]: item for item in payload["findings"]}
        finding = codes["RANGE_FORECAST_METHOD_MISMATCH_REPAIR_REQUIRED"]
        perspective = finding["evidence"]["perspective_alignment_diagnostics"]
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertEqual(finding["priority"], "P1")
        self.assertEqual(finding["layer"], "forecast")
        self.assertEqual(perspective["range_forecast_method_mismatch_lanes"], 5)
        self.assertEqual(perspective["range_forecast_method_mismatch_top"][0]["pair"], "EUR_USD")
        self.assertEqual(
            perspective["range_forecast_method_mismatch_top"][0]["range_rotation_top_live_blocker_codes"][0]["code"],
            "FORECAST_RANGE_UNSELECTED_DIRECTION_CONFLICT",
        )
        self.assertEqual(
            perspective["range_forecast_method_mismatch_top"][0]["range_rotation_other_side_directions"][0]["code"],
            "LONG",
        )
        self.assertEqual(
            perspective["range_forecast_method_mismatch_top"][0]["range_rotation_other_side_top_live_blocker_codes"][0]["code"],
            "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
        )
        self.assertIn("RANGE_FORECAST_METHOD_MISMATCH_REPAIR_REQUIRED", report_text)
        self.assertIn("perspective alignment", report_text)
        self.assertIn("EUR_USD SHORT mismatch=5", report_text)
        self.assertIn("other_rail=LONG", report_text)
        self.assertIn("other_blockers=FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", report_text)

    def test_partial_live_ready_coverage_still_names_target_shortfall(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, live_ready_market_rr=1.5, pending_entry=True)
            files["target"].write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_target_jpy": 1000.0,
                        "remaining_minimum_jpy": 500.0,
                    }
                )
            )
            files["coverage"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "status": "COVERAGE_GAP",
                        "remaining_target_jpy": 1000.0,
                        "live_ready_reward_jpy": 120.0,
                        "opportunity_modes": {
                            "HARVEST": {
                                "lanes": 4,
                                "live_ready_lanes": 1,
                                "live_ready_reward_jpy": 120.0,
                                "promotion_candidate_lanes": 0,
                                "top_issue_codes": [
                                    {"code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", "count": 3}
                                ],
                                "top_blockers": [
                                    {"label": "forecast confidence below live floor", "count": 3}
                                ],
                            },
                            "RUNNER": {
                                "lanes": 0,
                                "live_ready_lanes": 0,
                                "live_ready_reward_jpy": 0.0,
                                "promotion_candidate_lanes": 0,
                            },
                        },
                        "runner_candidate_diagnostics": {
                            "status": "RUNNER_CANDIDATES_DEMOTED_TO_HARVEST",
                            "trend_candidate_lanes": 3,
                            "runner_qualified_lanes": 0,
                            "attached_harvest_lanes": 3,
                            "top_demotion_reasons": [
                                {"reason": "RANGE regime is not a clean runner trend", "count": 2}
                            ],
                        },
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())
            report_text = files["report"].read_text()

        codes = {item["code"]: item for item in payload["findings"]}
        finding = codes["TARGET_OPEN_LIVE_READY_COVERAGE_SHORTFALL"]
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertEqual(finding["priority"], "P1")
        self.assertEqual(finding["evidence"]["live_ready_lanes"], 1)
        self.assertEqual(finding["evidence"]["live_ready_reward_jpy"], 120.0)
        self.assertEqual(finding["evidence"]["required_additional_reward_jpy"], 880.0)
        self.assertEqual(finding["evidence"]["minimum_floor_shortfall_jpy"], 380.0)
        self.assertEqual(finding["evidence"]["opportunity_modes"]["HARVEST"]["live_ready_lanes"], 1)
        self.assertEqual(
            finding["evidence"]["runner_candidate_diagnostics"]["status"],
            "RUNNER_CANDIDATES_DEMOTED_TO_HARVEST",
        )
        self.assertIn("live coverage", report_text)
        self.assertIn("runner candidates", report_text)

    def test_lane_only_verification_blockers_are_not_p0_with_live_ready_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                verification_lane_blockers=True,
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertIn("VERIFICATION_LEDGER_LANE_BLOCKERS_RECORDED", codes)
        self.assertNotIn("VERIFICATION_LEDGER_BLOCKED", codes)

    def test_persistent_profitability_discipline_escalates_to_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, -400.0, 50.0, -300.0),
            )
            files["ai_backtest"].write_text(
                json.dumps(
                    {
                        "mechanism_ablation": {
                            "close_gate_ab": {
                                "status": "MEASURED",
                                "loss_side_market_close_count": 2,
                                "loss_side_market_close_net_jpy": -700.0,
                                "broker_trade_close_loss_side_market_close_count": 2,
                                "broker_trade_close_loss_side_market_close_source_counts": {
                                    "DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE": 2
                                },
                                "broker_accepted_without_gateway_loss_side_market_close_count": 2,
                                "broker_accepted_without_gateway_loss_side_market_close_net_jpy": -700.0,
                                "broker_accepted_without_gateway_loss_side_market_close_source_counts": {
                                    "DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE": 2
                                },
                                "broker_accepted_without_gateway_loss_side_market_close_evidence_counts": {
                                    "NO_LOCAL_GATEWAY_CLOSE_RECEIPT": 2,
                                    "NO_CLIENT_EXTENSION": 2,
                                },
                            }
                        }
                    }
                )
            )

            first = _run(files, now=_NOW)
            retry = _run(files, now=_NOW + timedelta(seconds=30))
            second = _run(files, now=_NOW + timedelta(minutes=3))
            third = _run(files, now=_NOW + timedelta(minutes=6))
            payload = json.loads(files["output"].read_text())
            with sqlite3.connect(files["history_db"]) as conn:
                run_count = conn.execute("SELECT COUNT(*) FROM self_improvement_audit_runs").fetchone()[0]

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(first.status, STATUS_ACTION_REQUIRED)
        self.assertEqual(retry.status, STATUS_ACTION_REQUIRED)
        self.assertEqual(second.status, STATUS_ACTION_REQUIRED)
        self.assertEqual(third.status, STATUS_BLOCKED)
        self.assertEqual(run_count, 3)
        self.assertIn("NEGATIVE_RECENT_EXPECTANCY", codes)
        self.assertIn("SMALL_WIN_LARGE_LOSS_ASYMMETRY", codes)
        self.assertIn("PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED", codes)
        self.assertEqual(codes["PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED"]["priority"], "P0")
        self.assertEqual(codes["PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED"]["evidence"]["current_streak"], 3)
        close_evidence = codes["PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED"]["evidence"][
            "system_defect_evidence"
        ]["ai_backtest_close_gate_loss_evidence"]
        self.assertEqual(close_evidence["loss_side_market_close_count"], 2)
        self.assertEqual(close_evidence["broker_accepted_without_gateway_loss_side_market_close_count"], 2)
        self.assertEqual(
            close_evidence["broker_accepted_without_gateway_loss_side_market_close_evidence_counts"],
            {"NO_LOCAL_GATEWAY_CLOSE_RECEIPT": 2, "NO_CLIENT_EXTENSION": 2},
        )

    def test_direct_manual_close_dominated_repaired_profitability_does_not_escalate_to_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, live_ready_market_rr=1.4)
            _write_direct_manual_close_recovery_ledger(files["execution_db"])

            first = _run(files, now=_NOW)
            second = _run(files, now=_NOW + timedelta(minutes=3))
            third = _run(files, now=_NOW + timedelta(minutes=6))
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(first.status, STATUS_ACTION_REQUIRED)
        self.assertEqual(second.status, STATUS_ACTION_REQUIRED)
        self.assertEqual(third.status, STATUS_ACTION_REQUIRED)
        self.assertIn("NEGATIVE_RECENT_EXPECTANCY", codes)
        self.assertIn("SMALL_WIN_LARGE_LOSS_ASYMMETRY", codes)
        self.assertIn("DIRECT_OR_MANUAL_CLOSE_DOMINATED_PROFITABILITY_DRAG", codes)
        self.assertNotIn("PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED", codes)
        repair = codes["DIRECT_OR_MANUAL_CLOSE_DOMINATED_PROFITABILITY_DRAG"]["evidence"]
        self.assertEqual(repair["non_gateway_close_drag_metric"]["net_jpy"], -700.0)
        self.assertEqual(repair["net_without_non_gateway_close_drag_jpy"], 150.0)
        self.assertEqual(repair["last_24h_non_gateway_market_close_loss_trades"], 0)

    @staticmethod
    def _failed_trailing_effect() -> dict:
        return {
            "closed_trades": 20,
            "net_jpy": -5000.0,
            "profit_factor": 0.4,
            "expectancy_jpy": -250.0,
            "avg_win_jpy": 300.0,
            "avg_loss_jpy_abs": 1200.0,
            "worst_segments": [],
            "close_provenance_metrics": {},
            "market_order_trade_close_loss_provenance_metrics": {},
        }

    @staticmethod
    def _positive_24h_gateway_effect() -> dict:
        # Mirrors the live 2026-06-11 deadlock evidence: trailing window still
        # negative, but the last-24h gateway-attributable closes are net
        # positive without loss asymmetry.
        return {
            "closed_trades": 4,
            "net_jpy": 366.1,
            "gross_profit_jpy": 788.5,
            "gross_loss_jpy": 422.4,
            "profit_factor": 1.87,
            "expectancy_jpy": 91.5,
            "close_provenance_metrics": {
                "GATEWAY_TRADE_CLOSE_SENT": {
                    "trades": 3,
                    "net_jpy": -282.7,
                    "gross_profit_jpy": 139.7,
                    "gross_loss_jpy": 422.4,
                    "win_trades": 1,
                    "loss_trades": 2,
                },
                "TAKE_PROFIT_ORDER": {
                    "trades": 1,
                    "net_jpy": 648.8,
                    "gross_profit_jpy": 648.8,
                    "gross_loss_jpy": 0.0,
                    "win_trades": 1,
                    "loss_trades": 0,
                },
            },
            "market_order_trade_close_loss_provenance_metrics": {},
        }

    @staticmethod
    def _bleeding_24h_gateway_effect() -> dict:
        return {
            "closed_trades": 3,
            "net_jpy": -7157.0,
            "gross_profit_jpy": 0.0,
            "gross_loss_jpy": 7157.0,
            "profit_factor": 0.0,
            "expectancy_jpy": -2385.7,
            "close_provenance_metrics": {
                "GATEWAY_TRADE_CLOSE_SENT": {
                    "trades": 3,
                    "net_jpy": -7157.0,
                    "gross_profit_jpy": 0.0,
                    "gross_loss_jpy": 7157.0,
                    "win_trades": 0,
                    "loss_trades": 3,
                },
            },
            "market_order_trade_close_loss_provenance_metrics": {},
        }

    def test_gateway_close_recovery_observation_conditions(self) -> None:
        observation = _gateway_close_recovery_observation(self._positive_24h_gateway_effect())
        self.assertIsNotNone(observation)
        self.assertEqual(observation["gateway_win_trades"], 2)
        self.assertEqual(observation["gateway_loss_trades"], 2)
        self.assertAlmostEqual(observation["gateway_net_jpy"], 366.1, places=1)

        self.assertIsNone(_gateway_close_recovery_observation(self._bleeding_24h_gateway_effect()))
        self.assertIsNone(_gateway_close_recovery_observation({"error": "missing"}))
        self.assertIsNone(_gateway_close_recovery_observation({"close_provenance_metrics": {}}))

        manual_only = {
            "close_provenance_metrics": {
                "NON_TRADER_CLIENT_EXTENSION": {
                    "trades": 1,
                    "net_jpy": 22000.0,
                    "gross_profit_jpy": 22000.0,
                    "gross_loss_jpy": 0.0,
                    "win_trades": 1,
                    "loss_trades": 0,
                },
            },
        }
        self.assertIsNone(_gateway_close_recovery_observation(manual_only))

        asymmetric_but_positive = {
            "close_provenance_metrics": {
                "GATEWAY_TRADE_CLOSE_SENT": {
                    "trades": 4,
                    "net_jpy": 100.0,
                    "gross_profit_jpy": 1500.0,
                    "gross_loss_jpy": 1400.0,
                    "win_trades": 3,
                    "loss_trades": 1,
                },
            },
        }
        self.assertIsNone(_gateway_close_recovery_observation(asymmetric_but_positive))

    def test_persistent_profitability_downgrades_to_recovery_on_clean_24h_gateway_window(self) -> None:
        findings = _profitability_findings(
            run_id="run-recovery",
            effect=self._failed_trailing_effect(),
            effect_24h=self._positive_24h_gateway_effect(),
            snapshot={},
            min_sample=3,
            close_gate_loss_evidence=None,
            previous_discipline_streak=5,
        )
        codes = {item["code"]: item for item in findings}
        self.assertNotIn("PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED", codes)
        self.assertIn("PERSISTENT_PROFITABILITY_DISCIPLINE_RECOVERY", codes)
        recovery = codes["PERSISTENT_PROFITABILITY_DISCIPLINE_RECOVERY"]
        self.assertEqual(recovery["priority"], "P1")
        self.assertEqual(recovery["evidence"]["current_streak"], 6)
        self.assertEqual(recovery["evidence"]["recovery_observation"]["gateway_win_trades"], 2)

    def test_persistent_profitability_stays_p0_when_24h_gateway_window_bleeds(self) -> None:
        findings = _profitability_findings(
            run_id="run-bleeding",
            effect=self._failed_trailing_effect(),
            effect_24h=self._bleeding_24h_gateway_effect(),
            snapshot={},
            min_sample=3,
            close_gate_loss_evidence=None,
            previous_discipline_streak=5,
        )
        codes = {item["code"]: item for item in findings}
        self.assertIn("PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED", codes)
        self.assertNotIn("PERSISTENT_PROFITABILITY_DISCIPLINE_RECOVERY", codes)
        self.assertEqual(codes["PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED"]["priority"], "P0")

    def test_persistent_profitability_stays_p0_without_material_gateway_recovery_proof(self) -> None:
        effect_24h = {
            "closed_trades": 1,
            "net_jpy": -661.5,
            "gross_profit_jpy": 0.0,
            "gross_loss_jpy": 661.5,
            "profit_factor": 0.0,
            "expectancy_jpy": -661.5,
            "close_provenance_metrics": {
                "GATEWAY_TRADE_CLOSE_SENT": {
                    "trades": 1,
                    "net_jpy": -661.5,
                    "gross_profit_jpy": 0.0,
                    "gross_loss_jpy": 661.5,
                    "win_trades": 0,
                    "loss_trades": 1,
                    "loss_containment_trades": 1,
                    "loss_containment_net_jpy": -661.5,
                    "loss_containment_avoided_loss_jpy": 800.0,
                },
            },
            "market_order_trade_close_loss_provenance_metrics": {
                "GATEWAY_TRADE_CLOSE_SENT": {
                    "trades": 1,
                    "net_jpy": -661.5,
                    "gross_profit_jpy": 0.0,
                    "gross_loss_jpy": 661.5,
                    "win_trades": 0,
                    "loss_trades": 1,
                }
            },
        }

        findings = _profitability_findings(
            run_id="run-no-recovery-proof",
            effect={
                **self._failed_trailing_effect(),
                "profit_factor": 0.891,
                "expectancy_jpy": -29.04,
                "avg_win_jpy": 509.21,
                "avg_loss_jpy_abs": 500.0,
            },
            effect_24h=effect_24h,
            snapshot={},
            min_sample=3,
            close_gate_loss_evidence=None,
            previous_discipline_streak=5,
        )

        codes = {item["code"]: item for item in findings}
        blocked = codes["PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED"]
        self.assertEqual(blocked["priority"], "P0")
        self.assertNotIn("PERSISTENT_PROFITABILITY_DISCIPLINE_RECOVERY", codes)
        evidence = blocked["evidence"]["system_defect_evidence"]
        self.assertIn("persistent_negative_expectancy_without_recovery", evidence)
        self.assertFalse(
            evidence["persistent_negative_expectancy_without_recovery"][
                "last_24h_gateway_recovery_proven"
            ]
        )

    def test_persistent_profitability_recovers_when_gateway_close_only_materially_contained_loss(
        self,
    ) -> None:
        effect_24h = {
            "closed_trades": 1,
            "net_jpy": -661.5,
            "gross_profit_jpy": 0.0,
            "gross_loss_jpy": 661.5,
            "profit_factor": 0.0,
            "expectancy_jpy": -661.5,
            "close_provenance_metrics": {
                "GATEWAY_TRADE_CLOSE_SENT": {
                    "trades": 1,
                    "net_jpy": -661.5,
                    "gross_profit_jpy": 0.0,
                    "gross_loss_jpy": 661.5,
                    "win_trades": 0,
                    "loss_trades": 1,
                    "loss_containment_trades": 1,
                    "loss_containment_net_jpy": -661.5,
                    "loss_containment_avoided_loss_jpy": 5782.5,
                },
            },
            "market_order_trade_close_loss_provenance_metrics": {
                "GATEWAY_TRADE_CLOSE_SENT": {
                    "trades": 1,
                    "net_jpy": -661.5,
                    "gross_profit_jpy": 0.0,
                    "gross_loss_jpy": 661.5,
                    "win_trades": 0,
                    "loss_trades": 1,
                }
            },
        }

        findings = _profitability_findings(
            run_id="run-contained-only",
            effect={
                **self._failed_trailing_effect(),
                "profit_factor": 0.891,
                "expectancy_jpy": -29.04,
                "avg_win_jpy": 509.21,
                "avg_loss_jpy_abs": 500.0,
            },
            effect_24h=effect_24h,
            snapshot={},
            min_sample=3,
            close_gate_loss_evidence=None,
            previous_discipline_streak=5,
        )

        codes = {item["code"]: item for item in findings}
        self.assertNotIn("PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED", codes)
        recovery = codes["PERSISTENT_PROFITABILITY_DISCIPLINE_RECOVERY"]
        observation = recovery["evidence"]["recovery_observation"]
        self.assertEqual(recovery["priority"], "P1")
        self.assertEqual(observation["recovery_basis"], "material_loss_containment")
        self.assertEqual(observation["gateway_win_trades"], 0)
        self.assertEqual(observation["gateway_loss_trades"], 0)
        self.assertEqual(observation["gateway_net_jpy"], 0.0)
        self.assertEqual(observation["gateway_raw_net_jpy"], -661.5)
        self.assertEqual(observation["loss_containment_trades"], 1)
        self.assertGreater(observation["loss_containment_avoided_loss_jpy"], 1323.0)

    def test_persistent_profitability_stays_p0_when_contained_loss_erases_tp_win(self) -> None:
        effect_24h = {
            "closed_trades": 2,
            "net_jpy": -200.0,
            "gross_profit_jpy": 800.0,
            "gross_loss_jpy": 1000.0,
            "profit_factor": 0.8,
            "expectancy_jpy": -100.0,
            "close_provenance_metrics": {
                "GATEWAY_TRADE_CLOSE_SENT": {
                    "trades": 1,
                    "net_jpy": -1000.0,
                    "gross_profit_jpy": 0.0,
                    "gross_loss_jpy": 1000.0,
                    "win_trades": 0,
                    "loss_trades": 1,
                    "loss_containment_trades": 1,
                    "loss_containment_net_jpy": -1000.0,
                    "loss_containment_avoided_loss_jpy": 2400.0,
                },
                "TAKE_PROFIT_ORDER": {
                    "trades": 1,
                    "net_jpy": 800.0,
                    "gross_profit_jpy": 800.0,
                    "gross_loss_jpy": 0.0,
                    "win_trades": 1,
                    "loss_trades": 0,
                },
            },
            "market_order_trade_close_loss_provenance_metrics": {},
        }

        findings = _profitability_findings(
            run_id="run-contained-close",
            effect=self._failed_trailing_effect(),
            effect_24h=effect_24h,
            snapshot={},
            min_sample=3,
            close_gate_loss_evidence=None,
            previous_discipline_streak=5,
        )

        codes = {item["code"]: item for item in findings}
        self.assertIn("PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED", codes)
        self.assertNotIn("PERSISTENT_PROFITABILITY_DISCIPLINE_RECOVERY", codes)
        blocked = codes["PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED"]
        self.assertEqual(blocked["priority"], "P0")
        self.assertIn("24h_gateway_raw_net=-200.00 JPY", blocked["message"])
        observation = blocked["evidence"]["system_defect_evidence"]["gateway_close_bleed_observation"]
        self.assertEqual(observation["gateway_bleed_basis"], "contained_loss_erased_wins")
        self.assertEqual(observation["gateway_raw_net_jpy"], -200.0)
        self.assertEqual(observation["gateway_net_jpy"], 800.0)
        self.assertEqual(observation["gateway_loss_trades"], 0)
        self.assertEqual(observation["loss_containment_trades"], 1)

    def test_persistent_profitability_stays_p0_when_containment_window_is_still_net_negative(
        self,
    ) -> None:
        effect_24h = {
            "closed_trades": 5,
            "net_jpy": -239.93,
            "gross_profit_jpy": 96.16,
            "gross_loss_jpy": 336.09,
            "profit_factor": 0.286,
            "expectancy_jpy": -47.99,
            "close_provenance_metrics": {
                "GATEWAY_TRADE_CLOSE_SENT": {
                    "trades": 5,
                    "net_jpy": -239.93,
                    "gross_profit_jpy": 96.16,
                    "gross_loss_jpy": 336.09,
                    "win_trades": 3,
                    "loss_trades": 2,
                    "loss_containment_trades": 2,
                    "loss_containment_net_jpy": -336.09,
                    "loss_containment_avoided_loss_jpy": 6204.61,
                },
            },
            "market_order_trade_close_loss_provenance_metrics": {
                "GATEWAY_TRADE_CLOSE_SENT": {
                    "trades": 2,
                    "net_jpy": -336.09,
                    "gross_profit_jpy": 0.0,
                    "gross_loss_jpy": 336.09,
                    "win_trades": 0,
                    "loss_trades": 2,
                }
            },
        }

        findings = _profitability_findings(
            run_id="run-contained-but-negative",
            effect={
                **self._failed_trailing_effect(),
                "profit_factor": 0.789,
                "expectancy_jpy": -51.76,
                "avg_win_jpy": 411.02,
                "avg_loss_jpy_abs": 463.12,
                "worst_segments": [
                    {
                        "pair": "AUD_NZD",
                        "side": "SHORT",
                        "method": "RANGE_ROTATION",
                        "trades": 2,
                        "net_jpy": -169.0777,
                        "trade_ids": ["472632", "472655"],
                    }
                ],
            },
            effect_24h=effect_24h,
            snapshot={},
            min_sample=3,
            close_gate_loss_evidence=None,
            previous_discipline_streak=53,
        )

        codes = {item["code"]: item for item in findings}
        self.assertIn("NEGATIVE_RECENT_EXPECTANCY", codes)
        self.assertIn("PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED", codes)
        self.assertNotIn("PERSISTENT_PROFITABILITY_DISCIPLINE_RECOVERY", codes)
        blocked = codes["PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED"]
        self.assertEqual(blocked["priority"], "P0")
        self.assertIn("data/execution_ledger.db", blocked["message"])
        self.assertIn("pair=AUD_NZD", blocked["message"])
        self.assertIn("trade_ids=472632,472655", blocked["message"])
        self.assertIn("Inspect data/execution_ledger.db", blocked["next_action"])
        bleed = blocked["evidence"]["system_defect_evidence"]["gateway_close_bleed_observation"]
        self.assertAlmostEqual(bleed["gateway_raw_net_jpy"], -239.93, places=2)
        self.assertEqual(bleed["gateway_loss_trades"], 0)
        self.assertEqual(bleed["loss_containment_trades"], 2)

    def test_persistent_profitability_escalates_when_gateway_close_bleeds_without_loss_asymmetry(
        self,
    ) -> None:
        effect = {
            "closed_trades": 29,
            "net_jpy": -209.58,
            "profit_factor": 0.971,
            "expectancy_jpy": -7.23,
            "avg_win_jpy": 509.21,
            "avg_loss_jpy_abs": 489.24,
            "worst_segments": [],
            "market_order_trade_close_loss_provenance_metrics": {
                "GATEWAY_TRADE_CLOSE_SENT": {
                    "trades": 15,
                    "net_jpy": -7338.58,
                    "gross_profit_jpy": 0.0,
                    "gross_loss_jpy": 7338.58,
                    "win_trades": 0,
                    "loss_trades": 15,
                }
            },
        }
        effect_24h = {
            "closed_trades": 6,
            "net_jpy": -673.02,
            "gross_profit_jpy": 1515.79,
            "gross_loss_jpy": 2188.81,
            "profit_factor": 0.693,
            "expectancy_jpy": -112.17,
            "close_provenance_metrics": {
                "GATEWAY_TRADE_CLOSE_SENT": {
                    "trades": 5,
                    "net_jpy": -1331.96,
                    "gross_profit_jpy": 856.85,
                    "gross_loss_jpy": 2188.81,
                    "win_trades": 2,
                    "loss_trades": 3,
                },
                "TAKE_PROFIT_ORDER": {
                    "trades": 1,
                    "net_jpy": 658.94,
                    "gross_profit_jpy": 658.94,
                    "gross_loss_jpy": 0.0,
                    "win_trades": 1,
                    "loss_trades": 0,
                },
            },
            "market_order_trade_close_loss_provenance_metrics": {
                "GATEWAY_TRADE_CLOSE_SENT": {
                    "trades": 3,
                    "net_jpy": -2188.81,
                    "gross_profit_jpy": 0.0,
                    "gross_loss_jpy": 2188.81,
                    "win_trades": 0,
                    "loss_trades": 3,
                }
            },
        }

        findings = _profitability_findings(
            run_id="run-gateway-bleed",
            effect=effect,
            effect_24h=effect_24h,
            snapshot={},
            min_sample=3,
            close_gate_loss_evidence=None,
            previous_discipline_streak=2,
        )

        codes = {item["code"]: item for item in findings}
        self.assertIn("NEGATIVE_RECENT_EXPECTANCY", codes)
        self.assertNotIn("SMALL_WIN_LARGE_LOSS_ASYMMETRY", codes)
        blocked = codes["PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED"]
        self.assertEqual(blocked["priority"], "P0")
        bleed = blocked["evidence"]["system_defect_evidence"]["gateway_close_bleed_observation"]
        self.assertAlmostEqual(bleed["gateway_net_jpy"], -673.02, places=2)
        self.assertEqual(bleed["gateway_loss_trades"], 3)

    def test_profitability_streak_counts_negative_expectancy_without_loss_asymmetry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "history.db"
            with sqlite3.connect(db_path) as conn:
                conn.executescript(
                    """
                    CREATE TABLE self_improvement_audit_runs (
                        run_uid TEXT PRIMARY KEY,
                        ts_utc TEXT NOT NULL,
                        status TEXT NOT NULL,
                        output_path TEXT NOT NULL,
                        report_path TEXT NOT NULL,
                        window_hours REAL NOT NULL,
                        findings INTEGER NOT NULL,
                        p0_findings INTEGER NOT NULL,
                        p1_findings INTEGER NOT NULL,
                        p2_findings INTEGER NOT NULL,
                        closed_trades INTEGER NOT NULL,
                        net_jpy REAL NOT NULL,
                        profit_factor REAL,
                        expectancy_jpy REAL,
                        live_ready_lanes INTEGER NOT NULL,
                        open_trader_positions INTEGER NOT NULL,
                        inserted_at_utc TEXT NOT NULL
                    );
                    CREATE TABLE self_improvement_findings (
                        finding_uid TEXT PRIMARY KEY,
                        run_uid TEXT NOT NULL,
                        ts_utc TEXT NOT NULL,
                        priority TEXT NOT NULL,
                        layer TEXT NOT NULL,
                        code TEXT NOT NULL,
                        message TEXT NOT NULL,
                        next_action TEXT NOT NULL,
                        evidence_json TEXT NOT NULL,
                        inserted_at_utc TEXT NOT NULL
                    );
                    """
                )
                for index in range(3):
                    ts = (_NOW - timedelta(minutes=index * 3)).isoformat()
                    run_uid = f"run-{index}"
                    conn.execute(
                        """
                        INSERT INTO self_improvement_audit_runs(
                            run_uid, ts_utc, status, output_path, report_path, window_hours,
                            findings, p0_findings, p1_findings, p2_findings, closed_trades,
                            net_jpy, profit_factor, expectancy_jpy, live_ready_lanes,
                            open_trader_positions, inserted_at_utc
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_uid,
                            ts,
                            STATUS_ACTION_REQUIRED,
                            "out.json",
                            "report.md",
                            168.0,
                            1,
                            0,
                            1,
                            0,
                            29,
                            -209.58,
                            0.971,
                            -7.23,
                            1,
                            0,
                            ts,
                        ),
                    )
                    conn.execute(
                        """
                        INSERT INTO self_improvement_findings(
                            finding_uid, run_uid, ts_utc, priority, layer, code,
                            message, next_action, evidence_json, inserted_at_utc
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            f"finding-{index}",
                            run_uid,
                            ts,
                            "P1",
                            "profitability",
                            "NEGATIVE_RECENT_EXPECTANCY",
                            "recent outcome window is not profitable",
                            "repair profitability discipline",
                            "{}",
                            ts,
                        ),
                    )

            auditor = SelfImprovementAuditor(history_db_path=db_path)

            self.assertEqual(auditor._history_code_streak(PROFITABILITY_DISCIPLINE_CODES), 3)
            self.assertEqual(
                auditor._history_code_streak(
                    ("NEGATIVE_RECENT_EXPECTANCY", "SMALL_WIN_LARGE_LOSS_ASYMMETRY")
                ),
                0,
            )

    def test_effect_metrics_attributes_closed_pl_to_opening_lane_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "execution_ledger.db"
            _write_method_attribution_ledger(db_path)

            effect = _effect_metrics(db_path, window_hours=168.0, now=_NOW)

        by_method = {
            (item["pair"], item["side"], item["method"]): item
            for item in effect["worst_segments"]
        }
        rotation = by_method[("EUR_USD", "SHORT", "RANGE_ROTATION")]
        self.assertEqual(rotation["trades"], 2)
        self.assertAlmostEqual(rotation["net_jpy"], -700.0)
        self.assertEqual(
            rotation["lane_ids"],
            ["range_trader:EUR_USD:SHORT:RANGE_ROTATION"],
        )
        self.assertEqual(rotation["trade_ids"], ["T1", "T2"])
        trend = by_method[("EUR_USD", "SHORT", "TREND_CONTINUATION")]
        self.assertEqual(trend["trades"], 1)
        self.assertAlmostEqual(trend["net_jpy"], 120.0)

    def test_effect_metrics_classifies_trader_entry_market_order_loss_close_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "execution_ledger.db"
            _write_market_close_attribution_ledger(db_path, include_gateway_close=False)
            with sqlite3.connect(db_path) as conn:
                conn.execute("ALTER TABLE execution_events ADD COLUMN raw_json TEXT")
                conn.execute(
                    """
                    INSERT INTO execution_events(
                        event_uid, ts_utc, event_type, lane_id, order_id, trade_id,
                        pair, side, units, realized_pl_jpy, exit_reason, raw_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "broker-close-accept",
                        (_NOW - timedelta(hours=1, minutes=10)).isoformat(),
                        "ORDER_ACCEPTED",
                        "",
                        "C42",
                        "",
                        "EUR_USD",
                        "SHORT",
                        1000,
                        None,
                        "TRADE_CLOSE",
                        json.dumps(
                            {
                                "id": "C42",
                                "reason": "TRADE_CLOSE",
                                "tradeClose": {"tradeID": "T42", "units": "ALL"},
                            }
                        ),
                    ),
                )

            effect = _effect_metrics(db_path, window_hours=168.0, now=_NOW)

        market_loss = effect["market_order_trade_close_loss_provenance_metrics"]
        self.assertEqual(market_loss["TRADER_ENTRY_LANE_ID"]["trades"], 1)
        self.assertAlmostEqual(market_loss["TRADER_ENTRY_LANE_ID"]["net_jpy"], -500.0)
        segment = effect["worst_segments"][0]
        self.assertEqual(
            segment["close_provenance_counts"],
            {"TRADER_ENTRY_LANE_ID": 1},
        )
        self.assertEqual(
            segment["close_provenance_net_jpy"],
            {"TRADER_ENTRY_LANE_ID": -500.0},
        )

    def test_effect_metrics_classifies_stale_gpt_close_satisfied_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "execution_ledger.db"
            _write_market_close_attribution_ledger(db_path, include_gateway_close=False)
            with sqlite3.connect(db_path) as conn:
                _add_raw_json_column(conn)
                _insert_broker_trade_close_accept(conn)
                _insert_stale_gpt_close_satisfied(conn)

            effect = _effect_metrics(db_path, window_hours=168.0, now=_NOW)

        market_loss = effect["market_order_trade_close_loss_provenance_metrics"]
        self.assertEqual(market_loss["STALE_GPT_CLOSE_SATISFIED"]["trades"], 1)
        self.assertAlmostEqual(market_loss["STALE_GPT_CLOSE_SATISFIED"]["net_jpy"], -500.0)
        self.assertNotIn("DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE", market_loss)
        segment = effect["worst_segments"][0]
        self.assertEqual(segment["close_provenance_counts"], {"STALE_GPT_CLOSE_SATISFIED": 1})

    def test_effect_metrics_classifies_gateway_market_order_loss_close_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "execution_ledger.db"
            _write_market_close_attribution_ledger(db_path, include_gateway_close=True)

            effect = _effect_metrics(db_path, window_hours=168.0, now=_NOW)

        market_loss = effect["market_order_trade_close_loss_provenance_metrics"]
        self.assertEqual(market_loss["GATEWAY_TRADE_CLOSE_SENT"]["trades"], 1)
        self.assertAlmostEqual(market_loss["GATEWAY_TRADE_CLOSE_SENT"]["net_jpy"], -500.0)

    def test_effect_metrics_marks_gateway_loss_close_contained_before_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "execution_ledger.db"
            _write_market_close_attribution_ledger(db_path, include_gateway_close=True)
            with sqlite3.connect(db_path) as conn:
                conn.execute("UPDATE execution_events SET price = 1.1000 WHERE event_uid = 'fill'")
                conn.execute("UPDATE execution_events SET price = 1.0980 WHERE event_uid = 'close'")
                _insert_close_gate_pass_observation(
                    conn,
                    trade_id="T42",
                    ts_utc=(_NOW - timedelta(hours=1, minutes=5)).isoformat(),
                )
                conn.execute(
                    """
                    INSERT INTO execution_events(
                        event_uid, ts_utc, event_type, lane_id, order_id, trade_id,
                        pair, side, units, realized_pl_jpy, exit_reason, price, sl
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "protection-sl",
                        (_NOW - timedelta(hours=2, minutes=50)).isoformat(),
                        "PROTECTION_CREATED",
                        "",
                        "SL42",
                        "T42",
                        "EUR_USD",
                        "",
                        None,
                        None,
                        "ON_FILL",
                        1.0950,
                        1.0950,
                    ),
                )

            effect = _effect_metrics(db_path, window_hours=168.0, now=_NOW)

        market_loss = effect["market_order_trade_close_loss_provenance_metrics"]["GATEWAY_TRADE_CLOSE_SENT"]
        close_metric = effect["close_provenance_metrics"]["GATEWAY_TRADE_CLOSE_SENT"]
        self.assertEqual(market_loss["trades"], 1)
        self.assertAlmostEqual(market_loss["net_jpy"], -500.0)
        self.assertEqual(market_loss["loss_containment_trades"], 1)
        self.assertGreater(market_loss["loss_containment_avoided_loss_jpy"], 0.0)
        self.assertEqual(close_metric["loss_containment_trades"], 1)

    def test_effect_metrics_does_not_count_gateway_loss_containment_without_close_gate_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "execution_ledger.db"
            _write_market_close_attribution_ledger(db_path, include_gateway_close=True)
            with sqlite3.connect(db_path) as conn:
                conn.execute("UPDATE execution_events SET price = 1.1000 WHERE event_uid = 'fill'")
                conn.execute("UPDATE execution_events SET price = 1.0980 WHERE event_uid = 'close'")
                conn.execute(
                    """
                    INSERT INTO execution_events(
                        event_uid, ts_utc, event_type, lane_id, order_id, trade_id,
                        pair, side, units, realized_pl_jpy, exit_reason, price, sl
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "protection-sl",
                        (_NOW - timedelta(hours=2, minutes=50)).isoformat(),
                        "PROTECTION_CREATED",
                        "",
                        "SL42",
                        "T42",
                        "EUR_USD",
                        "",
                        None,
                        None,
                        "ON_FILL",
                        1.0950,
                        1.0950,
                    ),
                )

            effect = _effect_metrics(db_path, window_hours=168.0, now=_NOW)

        market_loss = effect["market_order_trade_close_loss_provenance_metrics"]["GATEWAY_TRADE_CLOSE_SENT"]
        self.assertEqual(market_loss["trades"], 1)
        self.assertAlmostEqual(market_loss["net_jpy"], -500.0)
        self.assertNotIn("loss_containment_trades", market_loss)
        self.assertNotIn("loss_containment_avoided_loss_jpy", market_loss)

    def test_effect_metrics_matches_gateway_market_order_loss_close_by_order_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "execution_ledger.db"
            _write_market_close_attribution_ledger(db_path, include_gateway_close=False)
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO execution_events(
                        event_uid, ts_utc, event_type, lane_id, order_id, trade_id,
                        pair, side, units, realized_pl_jpy, exit_reason
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "gw-close-order-only",
                        (_NOW - timedelta(hours=1, minutes=5)).isoformat(),
                        "GATEWAY_TRADE_CLOSE_SENT",
                        "",
                        "C42",
                        "",
                        "EUR_USD",
                        "",
                        None,
                        None,
                        "REVIEW_EXIT",
                    ),
                )

            effect = _effect_metrics(db_path, window_hours=168.0, now=_NOW)

        market_loss = effect["market_order_trade_close_loss_provenance_metrics"]
        self.assertEqual(market_loss["GATEWAY_TRADE_CLOSE_SENT"]["trades"], 1)
        self.assertAlmostEqual(market_loss["GATEWAY_TRADE_CLOSE_SENT"]["net_jpy"], -500.0)
        self.assertNotIn("NO_LOCAL_CLOSE_PROVENANCE", market_loss)

    def test_unattributed_market_order_close_is_p1_execution_hole(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, live_ready_market_rr=1.4)
            _write_market_close_attribution_ledger(files["execution_db"], include_gateway_close=False)

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertIn("UNATTRIBUTED_MARKET_ORDER_CLOSES", codes)
        finding = codes["UNATTRIBUTED_MARKET_ORDER_CLOSES"]
        self.assertEqual(finding["priority"], "P1")
        self.assertEqual(finding["evidence"]["unattributed_loss_count"], 1)
        self.assertEqual(finding["evidence"]["examples"][0]["trade_id"], "T42")

    def test_stale_gpt_close_satisfied_is_separate_from_unattributed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, live_ready_market_rr=1.4)
            _write_market_close_attribution_ledger(files["execution_db"], include_gateway_close=False)
            with sqlite3.connect(files["execution_db"]) as conn:
                _add_raw_json_column(conn)
                _insert_broker_trade_close_accept(conn)
                _insert_stale_gpt_close_satisfied(conn)

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertNotIn("UNATTRIBUTED_MARKET_ORDER_CLOSES", codes)
        finding = codes["STALE_GPT_CLOSE_SATISFIED_AFTER_BROKER_CLOSE"]
        self.assertEqual(finding["priority"], "P1")
        self.assertEqual(finding["evidence"]["stale_gpt_close_satisfied_count"], 1)
        self.assertEqual(finding["evidence"]["examples"][0]["trade_id"], "T42")

    def test_unattributed_market_order_close_reports_broker_accept_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, live_ready_market_rr=1.4)
            _write_market_close_attribution_ledger(files["execution_db"], include_gateway_close=False)
            with sqlite3.connect(files["execution_db"]) as conn:
                conn.execute("ALTER TABLE execution_events ADD COLUMN raw_json TEXT")
                conn.execute(
                    """
                    INSERT INTO execution_events(
                        event_uid, ts_utc, event_type, lane_id, order_id, trade_id,
                        pair, side, units, realized_pl_jpy, exit_reason, raw_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "broker-close-accept",
                        (_NOW - timedelta(hours=1, minutes=10)).isoformat(),
                        "ORDER_ACCEPTED",
                        "",
                        "C42",
                        "",
                        "EUR_USD",
                        "SHORT",
                        1000,
                        None,
                        "TRADE_CLOSE",
                        json.dumps(
                            {
                                "id": "C42",
                                "type": "MARKET_ORDER",
                                "reason": "TRADE_CLOSE",
                                "tradeClose": {"tradeID": "T42", "units": "ALL"},
                            }
                        ),
                    ),
                )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        finding = codes["UNATTRIBUTED_MARKET_ORDER_CLOSES"]
        self.assertEqual(finding["evidence"]["broker_trade_close_accept_count"], 1)
        self.assertEqual(
            finding["evidence"]["broker_trade_close_accept_source_counts"],
            {"TRADER_ENTRY_LANE_ID": 1},
        )
        self.assertEqual(
            finding["evidence"]["examples"][0]["broker_trade_close_accept_sources"],
            ["TRADER_ENTRY_LANE_ID"],
        )

    def test_broker_trade_close_accept_uses_entry_lane_source_when_close_has_no_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, live_ready_market_rr=1.4)
            _write_market_close_attribution_ledger(files["execution_db"], include_gateway_close=False)
            with sqlite3.connect(files["execution_db"]) as conn:
                _add_raw_json_column(conn)
                conn.execute(
                    """
                    UPDATE execution_events
                    SET lane_id = ?
                    WHERE event_uid = 'fill'
                    """,
                    ("trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET",),
                )
                _insert_broker_trade_close_accept(conn)

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        finding = codes["UNATTRIBUTED_MARKET_ORDER_CLOSES"]
        self.assertEqual(
            finding["evidence"]["broker_trade_close_accept_source_counts"],
            {"TRADER_ENTRY_LANE_ID": 1},
        )
        self.assertEqual(
            finding["evidence"]["examples"][0]["broker_trade_close_accept_sources"],
            ["TRADER_ENTRY_LANE_ID"],
        )
        market_loss = payload["effect_metrics"]["window"]["market_order_trade_close_loss_provenance_metrics"]
        self.assertEqual(market_loss["TRADER_ENTRY_LANE_ID"]["trades"], 1)
        self.assertNotIn("DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE", market_loss)

    def test_broker_trade_close_accept_uses_gateway_entry_receipt_source_when_fill_lane_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, live_ready_market_rr=1.4)
            _write_market_close_attribution_ledger(files["execution_db"], include_gateway_close=False)
            with sqlite3.connect(files["execution_db"]) as conn:
                _add_raw_json_column(conn)
                _insert_broker_trade_close_accept(conn)

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        finding = codes["UNATTRIBUTED_MARKET_ORDER_CLOSES"]
        self.assertEqual(
            finding["evidence"]["broker_trade_close_accept_source_counts"],
            {"TRADER_ENTRY_LANE_ID": 1},
        )
        market_loss = payload["effect_metrics"]["window"]["market_order_trade_close_loss_provenance_metrics"]
        self.assertEqual(market_loss["TRADER_ENTRY_LANE_ID"]["trades"], 1)
        self.assertNotIn("DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE", market_loss)

    def test_gateway_close_receipt_satisfies_market_order_close_attribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, live_ready_market_rr=1.4)
            _write_market_close_attribution_ledger(files["execution_db"], include_gateway_close=True)

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertNotIn("UNATTRIBUTED_MARKET_ORDER_CLOSES", codes)

    def test_reconciled_gpt_close_satisfies_market_order_close_attribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, live_ready_market_rr=1.4)
            _write_market_close_attribution_ledger(
                files["execution_db"],
                include_gateway_close=False,
                include_reconciled_close=True,
            )
            with sqlite3.connect(files["execution_db"]) as conn:
                conn.execute(
                    """
                    INSERT INTO execution_events(
                        event_uid, ts_utc, event_type, lane_id, order_id, trade_id,
                        pair, side, units, realized_pl_jpy, exit_reason
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "gpt-close-accepted",
                        (_NOW - timedelta(hours=1, minutes=10)).isoformat(),
                        "GATEWAY_GPT_CLOSE_ACCEPTED",
                        "",
                        "",
                        "T42",
                        "EUR_USD",
                        "",
                        None,
                        None,
                        "GPT_CLOSE_ACCEPTED",
                    ),
                )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        effect = payload["effect_metrics"]["window"]
        self.assertEqual(summary.p0_findings, 0)
        self.assertNotIn("ACCEPTED_GPT_CLOSE_WITHOUT_POSITION_GATEWAY_RECEIPT", codes)
        self.assertNotIn("UNATTRIBUTED_MARKET_ORDER_CLOSES", codes)
        self.assertEqual(
            effect["market_order_trade_close_loss_provenance_metrics"][
                "GATEWAY_GPT_CLOSE_ACCEPTED"
            ]["trades"],
            1,
        )

    def test_accepted_gpt_close_without_position_gateway_is_separate_from_unattributed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=False, live_ready_market_rr=1.4)
            _write_market_close_attribution_ledger(files["execution_db"], include_gateway_close=False)
            with sqlite3.connect(files["execution_db"]) as conn:
                conn.execute(
                    """
                    INSERT INTO execution_events(
                        event_uid, ts_utc, event_type, lane_id, order_id, trade_id,
                        pair, side, units, realized_pl_jpy, exit_reason
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "gpt-close-accepted",
                        (_NOW - timedelta(hours=1, minutes=10)).isoformat(),
                        "GATEWAY_GPT_CLOSE_ACCEPTED",
                        "",
                        "",
                        "T42",
                        "EUR_USD",
                        "",
                        None,
                        None,
                        "GPT_CLOSE_ACCEPTED",
                    ),
                )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertNotIn("UNATTRIBUTED_MARKET_ORDER_CLOSES", codes)
        finding = codes["ACCEPTED_GPT_CLOSE_WITHOUT_POSITION_GATEWAY_RECEIPT"]
        self.assertEqual(finding["priority"], "P1")
        self.assertEqual(finding["evidence"]["accepted_gpt_close_without_position_gateway_count"], 1)
        self.assertEqual(finding["evidence"]["examples"][0]["trade_id"], "T42")

    def test_rejected_close_for_closed_trades_is_not_current_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "REJECTED",
                        "decision": {"action": "CLOSE", "close_trade_ids": ["T1"]},
                        "verification_issues": [
                            {
                                "severity": "BLOCK",
                                "code": "CLOSE_THESIS_STILL_VALID",
                                "message": "CLOSE rejected for stale fixture trade",
                            }
                        ],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertNotIn("LATEST_GPT_DECISION_HAS_BLOCKING_ISSUES", codes)
        self.assertIn("STALE_GPT_CLOSE_BLOCKERS_FOR_CLOSED_TRADES", codes)
        self.assertEqual(codes["STALE_GPT_CLOSE_BLOCKERS_FOR_CLOSED_TRADES"]["priority"], "P1")

    def test_missing_legacy_trader_decision_is_not_p1_when_gpt_decision_readable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["trader"].unlink()

            _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertNotIn("TRADER_DECISION_UNREADABLE", codes)
        self.assertNotIn("GPT_DECISION_UNREADABLE", codes)

    def test_missing_legacy_trader_decision_is_reported_when_gpt_decision_unreadable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["gpt"].unlink()
            files["trader"].unlink()

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertEqual(codes["GPT_DECISION_UNREADABLE"]["priority"], "P0")
        self.assertEqual(codes["TRADER_DECISION_UNREADABLE"]["priority"], "P1")

    def test_rejected_non_close_receipt_blockers_are_not_current_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "REJECTED",
                        "decision": {"action": "REQUEST_EVIDENCE", "close_trade_ids": []},
                        "verification_issues": [
                            {
                                "severity": "BLOCK",
                                "code": "UNKNOWN_EVIDENCE_REF",
                                "message": "unknown evidence refs: option:skew:unknown",
                            }
                        ],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertNotIn("LATEST_GPT_DECISION_HAS_BLOCKING_ISSUES", codes)
        self.assertIn("LATEST_GPT_DECISION_REJECTED_WITH_BLOCKERS", codes)
        self.assertEqual(codes["LATEST_GPT_DECISION_REJECTED_WITH_BLOCKERS"]["priority"], "P1")
        self.assertEqual(codes["LATEST_GPT_DECISION_REJECTED_WITH_BLOCKERS"]["evidence"]["action"], "REQUEST_EVIDENCE")

    def test_rejected_stale_trade_receipt_is_not_stale_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "REJECTED",
                        "generated_at_utc": (_NOW - timedelta(minutes=1)).isoformat(),
                        "decision": {"action": "TRADE", "close_trade_ids": []},
                        "verification_issues": [
                            {
                                "severity": "BLOCK",
                                "code": "SELF_IMPROVEMENT_P0_BLOCKS_TRADE",
                                "message": "trade rejected by prior audit P0",
                            }
                        ],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertNotIn("LATEST_GPT_DECISION_STALE", codes)
        self.assertIn("LATEST_GPT_DECISION_REJECTED_WITH_BLOCKERS", codes)
        self.assertEqual(codes["LATEST_GPT_DECISION_REJECTED_WITH_BLOCKERS"]["priority"], "P1")

    def test_rejected_close_for_active_trade_remains_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=True)
            files["entry_thesis"].write_text(
                json.dumps(
                    {
                        "trade_id": "T1",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "filled_at_utc": _NOW.isoformat(),
                    }
                )
                + "\n"
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "REJECTED",
                        "decision": {"action": "CLOSE", "close_trade_ids": ["T1"]},
                        "verification_issues": [
                            {
                                "severity": "BLOCK",
                                "code": "CLOSE_THESIS_STILL_VALID",
                                "message": "CLOSE rejected for active fixture trade",
                            }
                        ],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertGreaterEqual(summary.p0_findings, 1)
        self.assertIn("LATEST_GPT_DECISION_HAS_BLOCKING_ISSUES", codes)
        self.assertEqual(
            codes["LATEST_GPT_DECISION_HAS_BLOCKING_ISSUES"]["evidence"]["active_close_trade_ids"],
            ["T1"],
        )

    def test_rejected_close_deferred_only_by_spread_is_not_current_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=True)
            files["entry_thesis"].write_text(
                json.dumps(
                    {
                        "trade_id": "T1",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "filled_at_utc": _NOW.isoformat(),
                    }
                )
                + "\n"
            )
            files["position_thesis"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "assessments": [
                            {
                                "trade_id": "T1",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "verdict": "REVIEW_CLOSE",
                                "context_notes": [
                                    "invalidation hit with technical invalidation confirmed against LONG"
                                ],
                            }
                        ],
                    }
                )
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "REJECTED",
                        "decision": {"action": "CLOSE", "close_trade_ids": ["T1"]},
                        "verification_issues": [
                            {
                                "severity": "BLOCK",
                                "code": "POSITION_CLOSE_SPREAD_TOO_WIDE",
                                "message": "close spread is above the deterministic cap",
                            },
                            {
                                "severity": "BLOCK",
                                "code": "POSITION_CLOSE_FLOW_SPREAD_TOO_WIDE",
                                "message": "flow spread is above the deterministic cap",
                            },
                        ],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertNotIn("LATEST_GPT_DECISION_HAS_BLOCKING_ISSUES", codes)
        finding = codes["LATEST_GPT_CLOSE_DEFERRED_BY_LIQUIDITY"]
        self.assertEqual(finding["priority"], "P1")
        self.assertEqual(finding["evidence"]["active_close_trade_ids"], ["T1"])
        self.assertEqual(
            finding["evidence"]["codes"],
            ["POSITION_CLOSE_SPREAD_TOO_WIDE", "POSITION_CLOSE_FLOW_SPREAD_TOO_WIDE"],
        )

    def test_operator_auth_required_close_is_not_reported_as_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=True)
            files["position_thesis"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "assessments": [
                            {
                                "trade_id": "T1",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "verdict": "REVIEW_CLOSE",
                                "context_notes": [
                                    "invalidation hit with technical invalidation confirmed against LONG"
                                ],
                            }
                        ],
                    }
                )
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "REJECTED",
                        "decision": {"action": "CLOSE", "close_trade_ids": ["T1"]},
                        "verification_issues": [
                            {
                                "severity": "BLOCK",
                                "code": "CLOSE_OPERATOR_AUTH_REQUIRED",
                                "message": "explicit Gate B is still missing for T1",
                            }
                        ],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertGreaterEqual(summary.p0_findings, 1)
        self.assertIn("LATEST_GPT_DECISION_HAS_BLOCKING_ISSUES", codes)
        self.assertNotIn("OPEN_POSITION_CLOSE_EVIDENCE_UNRESOLVED", codes)
        finding = codes["OPEN_POSITION_CLOSE_OPERATOR_AUTH_REQUIRED"]
        self.assertEqual(finding["priority"], "P1")
        self.assertEqual(finding["evidence"]["active_close_trade_ids"], ["T1"])

    def test_guardian_close_review_is_reported_as_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=True)
            files["gpt"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "status": "ACCEPTED",
                        "decision": {"action": "TRADE"},
                        "verification_issues": [],
                    }
                )
            )
            files["entry_thesis"].write_text(
                json.dumps(
                    {
                        "trade_id": "T1",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "filled_at_utc": _NOW.isoformat(),
                    }
                )
                + "\n"
            )
            files["position_guardian_management"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "action": "HOLD_PROTECTED",
                        "positions": [
                            {
                                "trade_id": "T1",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "action": "HOLD_PROTECTED",
                                "close_review_action": "REVIEW_EXIT",
                                "reasons": [
                                    "loss-cut: structural OB broken across 2 TFs (M15@1.17000, H1@1.17100) (-120 JPY)",
                                    "QR_DISABLE_AUTO_CLOSE=1 -> REVIEW_EXIT demoted to HOLD_PROTECTED",
                                ],
                            }
                        ],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertGreaterEqual(summary.p1_findings, 1)
        self.assertIn("OPEN_POSITION_CLOSE_EVIDENCE_UNRESOLVED", codes)
        signals = codes["OPEN_POSITION_CLOSE_EVIDENCE_UNRESOLVED"]["evidence"]["signals"]
        self.assertEqual(signals[0]["source"], "position_guardian_management")
        self.assertEqual(signals[0]["trade_id"], "T1")

    def test_operator_auth_required_close_with_hold_sidecars_is_not_decision_history_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=True)
            files["entry_thesis"].write_text(
                json.dumps(
                    {
                        "trade_id": "T1",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "filled_at_utc": _NOW.isoformat(),
                    }
                )
                + "\n"
            )
            files["position_thesis"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "assessments": [
                            {
                                "trade_id": "T1",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "verdict": "REVIEW_CLOSE",
                                "context_notes": [
                                    "invalidation hit with technical invalidation confirmed against LONG"
                                ],
                            }
                        ],
                    }
                )
            )
            files["thesis_evolution"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "evolutions": [
                            {
                                "trade_id": "T1",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "status": "WEAKENED",
                                "verdict": "HOLD",
                                "rationale": "current forecast still supports the open position side",
                            }
                        ],
                    }
                )
            )
            files["forecast_persistence"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "verdicts": [
                            {
                                "trade_id": "T1",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "verdict": "EXTEND",
                                "reason": "recent forecasts still support the open position side",
                            }
                        ],
                    }
                )
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "REJECTED",
                        "decision": {"action": "CLOSE", "close_trade_ids": ["T1"]},
                        "verification_issues": [
                            {
                                "severity": "BLOCK",
                                "code": "CLOSE_OPERATOR_AUTH_REQUIRED",
                                "message": "explicit Gate B is still missing for T1",
                            }
                        ],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.p0_findings, 0)
        self.assertNotIn("LATEST_GPT_DECISION_HAS_BLOCKING_ISSUES", codes)
        finding = codes["LATEST_GPT_DECISION_SOFT_CLOSE_ADVISORY_REJECTED"]
        self.assertEqual(finding["priority"], "P1")
        self.assertEqual(finding["evidence"]["active_close_trade_ids"], ["T1"])
        self.assertIn("OPEN_POSITION_CLOSE_OPERATOR_AUTH_REQUIRED", codes)

    def test_operator_auth_required_close_with_opposite_side_sidecars_stays_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=True)
            files["position_thesis"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "assessments": [
                            {
                                "trade_id": "T1",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "verdict": "REVIEW_CLOSE",
                                "context_notes": [
                                    "invalidation hit with technical invalidation confirmed against LONG"
                                ],
                            }
                        ],
                    }
                )
            )
            files["thesis_evolution"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "evolutions": [
                            {
                                "trade_id": "T1",
                                "pair": "EUR_USD",
                                "side": "SHORT",
                                "verdict": "HOLD",
                                "rationale": "opposite side support must not protect the active LONG",
                            }
                        ],
                    }
                )
            )
            files["forecast_persistence"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": _NOW.isoformat(),
                        "verdicts": [
                            {
                                "trade_id": "T1",
                                "pair": "EUR_USD",
                                "side": "SHORT",
                                "verdict": "EXTEND",
                                "reason": "opposite side support must not protect the active LONG",
                            }
                        ],
                    }
                )
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "REJECTED",
                        "decision": {"action": "CLOSE", "close_trade_ids": ["T1"]},
                        "verification_issues": [
                            {
                                "severity": "BLOCK",
                                "code": "CLOSE_OPERATOR_AUTH_REQUIRED",
                                "message": "explicit Gate B is still missing for T1",
                            }
                        ],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertGreaterEqual(summary.p0_findings, 1)
        self.assertIn("LATEST_GPT_DECISION_HAS_BLOCKING_ISSUES", codes)
        self.assertNotIn("LATEST_GPT_DECISION_SOFT_CLOSE_ADVISORY_REJECTED", codes)

    def test_accepted_wait_predating_snapshot_without_risk_is_p1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "generated_at_utc": (_NOW - timedelta(minutes=1)).isoformat(),
                        "decision": {"action": "WAIT"},
                        "verification_issues": [],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertIn("LATEST_GPT_DECISION_STALE", codes)
        self.assertEqual(codes["LATEST_GPT_DECISION_STALE"]["priority"], "P1")
        self.assertEqual(
            codes["LATEST_GPT_DECISION_STALE"]["evidence"]["snapshot_fetched_at_utc"],
            _NOW.isoformat(),
        )
        self.assertEqual(codes["LATEST_GPT_DECISION_STALE"]["evidence"]["current_streak"], 1)

    def test_accepted_wait_predating_snapshot_with_open_position_stays_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=True,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["entry_thesis"].write_text(
                json.dumps(
                    {
                        "trade_id": "T1",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "filled_at_utc": _NOW.isoformat(),
                    }
                )
                + "\n"
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "generated_at_utc": (_NOW - timedelta(minutes=1)).isoformat(),
                        "decision": {"action": "WAIT"},
                        "verification_issues": [],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertIn("LATEST_GPT_DECISION_STALE", codes)
        self.assertEqual(codes["LATEST_GPT_DECISION_STALE"]["priority"], "P0")

    def test_consumed_wait_decision_predating_snapshot_is_not_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "generated_at_utc": (_NOW - timedelta(minutes=1)).isoformat(),
                        "decision": {"action": "WAIT"},
                        "verification_issues": [],
                    }
                )
            )
            with sqlite3.connect(files["execution_db"]) as conn:
                conn.execute(
                    """
                    INSERT INTO execution_events(event_uid, ts_utc, event_type, pair, side, realized_pl_jpy)
                    VALUES (?, ?, 'GATEWAY_ORDER_NO_ACTION', NULL, NULL, NULL)
                    """,
                    ("consumed-wait", (_NOW - timedelta(seconds=30)).isoformat()),
                )

            _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertNotIn("LATEST_GPT_DECISION_STALE", codes)

    def test_consumed_trade_decision_predating_snapshot_is_not_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "generated_at_utc": (_NOW - timedelta(minutes=1)).isoformat(),
                        "decision": {
                            "action": "TRADE",
                            "selected_lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET",
                            "selected_lane_ids": ["trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET"],
                        },
                        "verification_issues": [],
                    }
                )
            )
            with sqlite3.connect(files["execution_db"]) as conn:
                conn.execute(
                    """
                    INSERT INTO execution_events(event_uid, ts_utc, event_type, pair, side, realized_pl_jpy)
                    VALUES (?, ?, 'GATEWAY_ORDER_SENT', 'EUR_USD', 'LONG', NULL)
                    """,
                    ("consumed-trade", (_NOW - timedelta(seconds=30)).isoformat()),
                )

            _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertNotIn("LATEST_GPT_DECISION_STALE", codes)

    def test_accepted_trade_consumed_by_current_pending_entry_is_not_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=True,
                pending_entry=True,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "generated_at_utc": (_NOW - timedelta(minutes=1)).isoformat(),
                        "decision": {
                            "action": "TRADE",
                            "selected_lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                            "selected_lane_ids": ["trend_trader:EUR_USD:LONG:TREND_CONTINUATION"],
                        },
                        "verification_issues": [],
                    }
                )
            )

            _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"] for item in payload["findings"]}
        self.assertNotIn("LATEST_GPT_DECISION_STALE", codes)

    def test_accepted_request_evidence_predating_snapshot_without_risk_is_p1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "generated_at_utc": (_NOW - timedelta(minutes=1)).isoformat(),
                        "decision": {"action": "REQUEST_EVIDENCE"},
                        "verification_issues": [],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertIn("LATEST_GPT_DECISION_STALE", codes)
        self.assertEqual(codes["LATEST_GPT_DECISION_STALE"]["priority"], "P1")
        self.assertEqual(codes["LATEST_GPT_DECISION_STALE"]["evidence"]["live_ready_lanes"], 0)
        self.assertEqual(codes["LATEST_GPT_DECISION_STALE"]["evidence"]["pending_entry_orders"], 0)

    def test_stale_gpt_decision_finding_records_history_streak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "generated_at_utc": (_NOW - timedelta(minutes=1)).isoformat(),
                        "decision": {"action": "WAIT"},
                        "verification_issues": [],
                    }
                )
            )

            _run(files, now=_NOW)
            _run(files, now=_NOW + timedelta(minutes=1))
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(codes["LATEST_GPT_DECISION_STALE"]["evidence"]["current_streak"], 2)
        self.assertEqual(codes["LATEST_GPT_DECISION_STALE"]["evidence"]["previous_streak"], 1)

    def test_stale_gpt_decision_is_not_p0_when_live_ready_entry_needs_fresh_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=False,
                live_ready_market_rr=1.4,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["gpt"].write_text(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "generated_at_utc": (_NOW - timedelta(minutes=1)).isoformat(),
                        "decision": {"action": "WAIT"},
                        "verification_issues": [],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_ACTION_REQUIRED)
        self.assertIn("LATEST_GPT_DECISION_STALE", codes)
        self.assertEqual(codes["LATEST_GPT_DECISION_STALE"]["priority"], "P1")
        self.assertEqual(codes["LATEST_GPT_DECISION_STALE"]["evidence"]["live_ready_lanes"], 1)

    def test_position_management_stale_uses_source_snapshot_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                active_position=True,
                closed_pls=(100.0, 80.0, -50.0),
            )
            files["position_management"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": (_NOW + timedelta(seconds=5)).isoformat(),
                        "snapshot_fetched_at_utc": (_NOW - timedelta(minutes=5)).isoformat(),
                        "positions": [],
                    }
                )
            )

            summary = _run(files)
            payload = json.loads(files["output"].read_text())

        codes = {item["code"]: item for item in payload["findings"]}
        self.assertEqual(summary.status, STATUS_BLOCKED)
        self.assertIn("POSITION_MANAGEMENT_STALE", codes)
        self.assertEqual(codes["POSITION_MANAGEMENT_STALE"]["priority"], "P0")
        self.assertEqual(
            codes["POSITION_MANAGEMENT_STALE"]["evidence"]["sidecar_snapshot_fetched_at_utc"],
            (_NOW - timedelta(minutes=5)).isoformat(),
        )

    def test_cli_writes_audit_and_returns_blocked_code_for_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, active_position=True, write_memory=False)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main(
                    [
                        "self-improvement-audit",
                        "--db",
                        str(files["execution_db"]),
                        "--history-db",
                        str(files["history_db"]),
                        "--output",
                        str(files["output"]),
                        "--report",
                        str(files["report"]),
                        "--snapshot",
                        str(files["snapshot"]),
                        "--target-state",
                        str(files["target"]),
                        "--order-intents",
                        str(files["intents"]),
                        "--market-context-matrix",
                        str(files["market_context_matrix"]),
                        "--memory-health",
                        str(files["memory"]),
                        "--capture-economics",
                        str(files["capture_economics"]),
                        "--learning-audit",
                        str(files["learning"]),
                        "--ai-test-bot-backtest",
                        str(files["ai_backtest"]),
                        "--verification-ledger",
                        str(files["verification"]),
                        "--attack-advice",
                        str(files["attack_advice"]),
                        "--forecast-history",
                        str(files["forecast_history"]),
                        "--projection-ledger",
                        str(files["projection_ledger"]),
                        "--entry-thesis-ledger",
                        str(files["entry_thesis"]),
                        "--gpt-decision",
                        str(files["gpt"]),
                        "--trader-decision",
                        str(files["trader"]),
                        "--position-management",
                        str(files["position_management"]),
                        "--thesis-evolution",
                        str(files["thesis_evolution"]),
                        "--position-thesis",
                        str(files["position_thesis"]),
                        "--forecast-persistence",
                        str(files["forecast_persistence"]),
                        "--coverage-optimization",
                        str(files["coverage"]),
                    ]
                )

            result = json.loads(stdout.getvalue())
            self.assertEqual(code, 2)
            self.assertEqual(result["status"], STATUS_BLOCKED)
            self.assertTrue(files["output"].exists())
            self.assertTrue(files["report"].exists())
            payload = json.loads(files["output"].read_text())
            self.assertEqual(payload["artifact_paths"]["ai_attack_advice"], str(files["attack_advice"]))
            self.assertEqual(payload["artifact_paths"]["capture_economics"], str(files["capture_economics"]))


_NOW = datetime(2026, 6, 5, 0, 0, tzinfo=timezone.utc)


def _range_emission_context(
    *,
    pair: str = "EUR_USD",
    current_price: float = 1.1050,
    h1_atr_pips: object = 20.0,
) -> dict[str, object]:
    return build_forecast_technical_context(
        {
            "confluence": {
                "dominant_regime": "RANGE",
                "price_percentile_24h": 0.5,
            },
            "views": [
                {
                    "granularity": "M5",
                    "regime_reading": {"state": "RANGE", "atr_percentile": 50},
                    "indicators": {"atr_pips": 5.0},
                },
                {
                    "granularity": "M15",
                    "regime_reading": {"state": "RANGE", "atr_percentile": 50},
                    "structure": {
                        "structure_events": [
                            {"kind": "BOS_UP", "index": 5, "close_confirmed": True}
                        ]
                    },
                },
                {
                    "granularity": "H1",
                    "regime_reading": {"state": "RANGE", "atr_percentile": 50},
                    "indicators": {"atr_pips": h1_atr_pips},
                },
            ],
        },
        pair=pair,
        current_price=current_price,
        spread_pips=0.5,
    )


def _run(files: dict[str, Path], *, now: datetime = _NOW):
    return SelfImprovementAuditor(
        db_path=files["execution_db"],
        history_db_path=files["history_db"],
        output_path=files["output"],
        report_path=files["report"],
    ).run(
        snapshot_path=files["snapshot"],
        target_state_path=files["target"],
        order_intents_path=files["intents"],
        market_context_matrix_path=files["market_context_matrix"],
        memory_health_path=files["memory"],
        capture_economics_path=files["capture_economics"],
        learning_audit_path=files["learning"],
        ai_test_bot_backtest_path=files["ai_backtest"],
        verification_ledger_path=files["verification"],
        execution_timing_audit_path=files["execution_timing"],
        forecast_history_path=files["forecast_history"],
        projection_ledger_path=files["projection_ledger"],
        entry_thesis_ledger_path=files["entry_thesis"],
        gpt_decision_path=files["gpt"],
        trader_decision_path=files["trader"],
        position_management_path=files["position_management"],
        position_guardian_management_path=files["position_guardian_management"],
        thesis_evolution_path=files["thesis_evolution"],
        position_thesis_path=files["position_thesis"],
        forecast_persistence_path=files["forecast_persistence"],
        coverage_optimization_path=files["coverage"],
        attack_advice_path=files["attack_advice"],
        now=now,
    )


def _fixtures(
    root: Path,
    *,
    active_position: bool,
    write_memory: bool = True,
    projection_expired: bool = False,
    live_ready_market_rr: float | None = None,
    unrealized_pl_jpy: float = 0.0,
    closed_pls: tuple[float, ...] = (100.0, -250.0, 50.0),
    verification_lane_blockers: bool = False,
    pending_entry: bool = False,
) -> dict[str, Path]:
    files = {
        "execution_db": root / "execution_ledger.db",
        "history_db": root / "self_improvement_history.db",
        "output": root / "self_improvement.json",
        "report": root / "self_improvement.md",
        "snapshot": root / "broker_snapshot.json",
        "target": root / "daily_target_state.json",
        "intents": root / "order_intents.json",
        "market_context_matrix": root / "market_context_matrix.json",
        "memory": root / "memory_health.json",
        "capture_economics": root / "capture_economics.json",
        "learning": root / "learning_audit.json",
        "ai_backtest": root / "ai_test_bot_backtest.json",
        "verification": root / "verification_ledger.json",
        "execution_timing": root / "execution_timing_audit.json",
        "forecast_history": root / "forecast_history.jsonl",
        "projection_ledger": root / "projection_ledger.jsonl",
        "entry_thesis": root / "entry_thesis_ledger.jsonl",
        "gpt": root / "gpt_trader_decision.json",
        "trader": root / "trader_decision.json",
        "position_management": root / "position_management.json",
        "position_guardian_management": root / "position_guardian_management.json",
        "thesis_evolution": root / "thesis_evolution_report.json",
        "position_thesis": root / "position_thesis_report.json",
        "forecast_persistence": root / "forecast_persistence_report.json",
        "coverage": root / "coverage_optimization.json",
        "attack_advice": root / "ai_attack_advice.json",
    }
    positions = []
    if active_position:
        positions.append(
            {
                "trade_id": "T1",
                "pair": "EUR_USD",
                "side": "LONG",
                "owner": "trader",
                "units": 1000,
                "entry_price": 1.17,
                "take_profit": 1.18,
                "unrealized_pl_jpy": -120.0,
            }
        )
    orders = []
    if pending_entry:
        orders.append(
            {
                "order_id": "P1",
                "pair": "EUR_USD",
                "order_type": "STOP",
                "state": "PENDING",
                "units": 1000,
                "price": 1.171,
                "owner": "trader",
                "trade_id": None,
            }
        )
    files["snapshot"].write_text(
        json.dumps(
            {
                "fetched_at_utc": _NOW.isoformat(),
                "account": {
                    "fetched_at_utc": _NOW.isoformat(),
                    "last_transaction_id": "100",
                    "unrealized_pl_jpy": unrealized_pl_jpy,
                },
                "positions": positions,
                "orders": orders,
                "quotes": {"EUR_USD": {"bid": 1.1701, "ask": 1.1702, "timestamp_utc": _NOW.isoformat()}},
            }
        )
    )
    files["target"].write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 1000.0}))
    files["capture_economics"].write_text(
        json.dumps(
            {
                "generated_at_utc": _NOW.isoformat(),
                "status": "OK",
                "overall": {"trades": len(closed_pls)},
            }
        )
    )
    results = []
    if live_ready_market_rr is not None:
        results.append(
            {
                "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET",
                "status": "LIVE_READY",
                "intent": {
                    "pair": "EUR_USD",
                    "order_type": "MARKET",
                    "metadata": {"order_timing": "NOW_MARKET"},
                },
                "risk_metrics": {"reward_risk": live_ready_market_rr},
                "risk_issues": [],
                "strategy_issues": [],
                "live_blockers": [],
            }
        )
    files["intents"].write_text(json.dumps({"results": results}))
    if write_memory:
        files["memory"].write_text(
            json.dumps(
                {
                    "generated_at_utc": _NOW.isoformat(),
                    "status": "MEMORY_HEALTH_PASS",
                    "issues": [],
                    "blockers": [],
                    "warnings": [],
                }
            )
        )
    files["learning"].write_text(
        json.dumps(
            {
                "status": "LEARNING_AUDIT_PASS",
                "blockers": [],
                "warnings": [],
                "learning_influence": {"influenced_lanes": 0},
                "effect_metrics": {"closed_trades": len(closed_pls)},
                "min_effect_sample": 3,
            }
        )
    )
    verification_payload: dict[str, object] = {"status": "OK", "blocking_observations": 0, "blocking_evidence": []}
    if verification_lane_blockers:
        verification_payload = {
            "status": "BLOCKED",
            "blocking_observations": 1,
            "blocking_evidence": [
                {
                    "source": "order_intents",
                    "source_path": str(files["intents"]),
                    "subject_type": "lane",
                    "subject_id": "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                    "check_name": "lane_blockers",
                    "status": "BLOCK",
                    "severity": "BLOCK",
                    "evidence": {
                        "blockers": [
                            {
                                "code": "FRESH_ENTRY_REWARD_RISK_NOT_POSITIVE",
                                "message": "fresh entry reward/risk does not exceed 1.00x",
                                "severity": "BLOCK",
                            }
                        ]
                    },
                }
            ],
        }
    files["verification"].write_text(json.dumps(verification_payload))
    files["execution_timing"].write_text(
        json.dumps(
            {
                "generated_at_utc": _NOW.isoformat(),
                "status": "OK",
                "fetch_errors": [],
                "precision": {
                    TP_PROGRESS_REPAIR_REPLAY_FIELD: TP_PROGRESS_REPAIR_REPLAY_CONTRACT,
                    "price_basis": "OANDA_M1_BID_ASK_CANDLES",
                    "granularity": "M1",
                },
                "window": _replay_window(_NOW),
                "summary": {
                    "canceled_orders_audited": 0,
                    "canceled_entry_touched_after_cancel": 0,
                    "canceled_entry_touched_after_cancel_rate": None,
                    "canceled_positive_after_cancel_entry": 0,
                    "canceled_tp_touched_after_cancel": 0,
                    "canceled_estimated_missed_mfe_jpy": 0.0,
                    "loss_closes_audited": 0,
                    "loss_closes_profit_capture_missed": 0,
                    "loss_closes_repair_replay_triggered": 0,
                    "tp_progress_repair_live_evidence_boundary_utc": (
                        TP_PROGRESS_REPAIR_LIVE_EVIDENCE_BOUNDARY_UTC
                    ),
                    "tp_progress_repair_live_evidence_status": (
                        "WAITING_FOR_POST_REPAIR_SAMPLE"
                    ),
                    "pre_repair_historical_loss_closes_audited": 0,
                    "pre_repair_historical_loss_closes_profit_capture_missed": 0,
                    "pre_repair_historical_loss_closes_repair_replay_triggered": 0,
                    "post_repair_live_evidence_loss_closes_audited": 0,
                    "post_repair_live_evidence_loss_closes_profit_capture_missed": 0,
                    "post_repair_live_evidence_loss_closes_repair_replay_triggered": 0,
                },
                "loss_close_regrets": [],
                "canceled_order_regrets": [],
            }
        )
    )
    files["forecast_history"].write_text(
        json.dumps({"timestamp_utc": _NOW.isoformat(), "cycle_id": "cycle-1", "pair": "EUR_USD", "direction": "UP"})
        + "\n"
    )
    emitted = _NOW - timedelta(minutes=90 if projection_expired else 1)
    files["projection_ledger"].write_text(
        json.dumps(
            {
                "timestamp_emitted_utc": emitted.isoformat(),
                "pair": "EUR_USD",
                "signal_name": "directional_forecast",
                "resolution_window_min": 30.0,
                "resolution_status": "PENDING" if projection_expired else "HIT",
                "cycle_id": "cycle-1",
            }
        )
        + "\n"
    )
    if active_position:
        files["entry_thesis"].write_text("")
    else:
        files["entry_thesis"].write_text("")
    files["gpt"].write_text(json.dumps({"status": "ACCEPTED", "decision": {"action": "TRADE"}, "verification_issues": []}))
    files["trader"].write_text(json.dumps({"action": "SEND_ENTRY", "generated_at_utc": _NOW.isoformat()}))
    for key, list_key in (
        ("position_management", "positions"),
        ("thesis_evolution", "evolutions"),
        ("position_thesis", "assessments"),
        ("forecast_persistence", "verdicts"),
    ):
        files[key].write_text(json.dumps({"generated_at_utc": _NOW.isoformat(), list_key: []}))
    files["coverage"].write_text(json.dumps({"artifact_diagnostics": {}}))
    files["attack_advice"].write_text(
        json.dumps(
            {
                "generated_at_utc": _NOW.isoformat(),
                "status": "NO_ATTACK_ADVICE",
                "recommended_now_lane_ids": [],
            }
        )
    )
    _write_execution_ledger(files["execution_db"], closed_pls=closed_pls, last_transaction_id="100")
    return files


def _write_execution_ledger(path: Path, *, closed_pls: tuple[float, ...], last_transaction_id: str) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE sync_state (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at_utc TEXT NOT NULL);
            CREATE TABLE execution_events (
                event_uid TEXT PRIMARY KEY,
                ts_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                pair TEXT,
                side TEXT,
                realized_pl_jpy REAL
            );
            """
        )
        conn.execute(
            "INSERT INTO sync_state(key, value, updated_at_utc) VALUES (?, ?, ?)",
            ("last_oanda_transaction_id", last_transaction_id, _NOW.isoformat()),
        )
        for idx, pl in enumerate(closed_pls):
            conn.execute(
                """
                INSERT INTO execution_events(event_uid, ts_utc, event_type, pair, side, realized_pl_jpy)
                VALUES (?, ?, 'TRADE_CLOSED', 'EUR_USD', 'LONG', ?)
                """,
                (f"closed-{idx}", (_NOW - timedelta(hours=idx + 1)).isoformat(), pl),
            )


def _write_pending_cancel_churn_ledger(path: Path) -> None:
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE sync_state (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at_utc TEXT NOT NULL);
            CREATE TABLE execution_events (
                event_uid TEXT PRIMARY KEY,
                ts_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                lane_id TEXT,
                order_id TEXT,
                trade_id TEXT,
                pair TEXT,
                side TEXT,
                units INTEGER,
                price REAL,
                realized_pl_jpy REAL,
                exit_reason TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO sync_state(key, value, updated_at_utc) VALUES (?, ?, ?)",
            ("last_oanda_transaction_id", "100", _NOW.isoformat()),
        )
        for idx, pair in enumerate(("AUD_CAD", "CAD_CHF", "NZD_CHF")):
            accepted_ts = _NOW - timedelta(minutes=30 - idx * 5)
            canceled_ts = accepted_ts + timedelta(minutes=8 + idx)
            order_id = f"O-pending-{idx}"
            lane_id = f"range_trader:{pair}:LONG:RANGE_ROTATION"
            conn.execute(
                """
                INSERT INTO execution_events(
                    event_uid, ts_utc, event_type, lane_id, order_id,
                    pair, side, units, price, exit_reason
                )
                VALUES (?, ?, 'ORDER_ACCEPTED', ?, ?, ?, 'LONG', 1000, ?, NULL)
                """,
                (f"accepted-{idx}", accepted_ts.isoformat(), lane_id, order_id, pair, 1.1 + idx / 1000),
            )
            conn.execute(
                """
                INSERT INTO execution_events(
                    event_uid, ts_utc, event_type, lane_id, order_id,
                    pair, side, units, price, exit_reason
                )
                VALUES (?, ?, 'ORDER_CANCELED', ?, ?, ?, 'LONG', 1000, ?, NULL)
                """,
                (f"canceled-{idx}", canceled_ts.isoformat(), lane_id, order_id, pair, 1.1 + idx / 1000),
            )


def _write_pending_mixed_cancel_churn_ledger(path: Path) -> None:
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE sync_state (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at_utc TEXT NOT NULL);
            CREATE TABLE execution_events (
                event_uid TEXT PRIMARY KEY,
                ts_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                lane_id TEXT,
                order_id TEXT,
                trade_id TEXT,
                pair TEXT,
                side TEXT,
                units INTEGER,
                price REAL,
                realized_pl_jpy REAL,
                exit_reason TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO sync_state(key, value, updated_at_utc) VALUES (?, ?, ?)",
            ("last_oanda_transaction_id", "100", _NOW.isoformat()),
        )
        for idx, pair in enumerate(("AUD_CAD", "CAD_CHF", "NZD_CHF", "EUR_CAD", "NZD_JPY")):
            accepted_ts = _NOW - timedelta(minutes=60 - idx * 5)
            resolved_ts = accepted_ts + timedelta(minutes=6 + idx)
            order_id = f"O-mixed-{idx}"
            lane_id = f"range_trader:{pair}:LONG:RANGE_ROTATION"
            conn.execute(
                """
                INSERT INTO execution_events(
                    event_uid, ts_utc, event_type, lane_id, order_id,
                    pair, side, units, price, exit_reason
                )
                VALUES (?, ?, 'ORDER_ACCEPTED', ?, ?, ?, 'LONG', 1000, ?, NULL)
                """,
                (f"accepted-mixed-{idx}", accepted_ts.isoformat(), lane_id, order_id, pair, 1.2 + idx / 1000),
            )
            event_type = "ORDER_FILLED" if idx < 2 else "ORDER_CANCELED"
            exit_reason = "LIMIT_ORDER" if event_type == "ORDER_FILLED" else None
            conn.execute(
                """
                INSERT INTO execution_events(
                    event_uid, ts_utc, event_type, lane_id, order_id,
                    pair, side, units, price, exit_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, 'LONG', 1000, ?, ?)
                """,
                (
                    f"resolved-mixed-{idx}",
                    resolved_ts.isoformat(),
                    event_type,
                    lane_id,
                    order_id,
                    pair,
                    1.2 + idx / 1000,
                    exit_reason,
                ),
            )


def _write_pending_replacement_churn_ledger(path: Path) -> None:
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE sync_state (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at_utc TEXT NOT NULL);
            CREATE TABLE execution_events (
                event_uid TEXT PRIMARY KEY,
                ts_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                lane_id TEXT,
                order_id TEXT,
                trade_id TEXT,
                pair TEXT,
                side TEXT,
                units INTEGER,
                price REAL,
                realized_pl_jpy REAL,
                exit_reason TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO sync_state(key, value, updated_at_utc) VALUES (?, ?, ?)",
            ("last_oanda_transaction_id", "100", _NOW.isoformat()),
        )
        rows = [
            (
                "accepted-source",
                _NOW - timedelta(minutes=55),
                "ORDER_ACCEPTED",
                "range_trader:AUD_CAD:SHORT:RANGE_ROTATION",
                "replace-source",
                "AUD_CAD",
                "SHORT",
                -1000,
                0.9898,
            ),
            (
                "canceled-source",
                _NOW - timedelta(minutes=50),
                "ORDER_CANCELED",
                "range_trader:AUD_CAD:SHORT:RANGE_ROTATION",
                "replace-source",
                "AUD_CAD",
                "SHORT",
                -1000,
                0.9898,
            ),
            (
                "accepted-next",
                _NOW - timedelta(minutes=45),
                "ORDER_ACCEPTED",
                "range_trader:AUD_CAD:SHORT:RANGE_ROTATION",
                "replace-next",
                "AUD_CAD",
                "SHORT",
                -1000,
                0.9897,
            ),
            (
                "accepted-orphan-1",
                _NOW - timedelta(minutes=40),
                "ORDER_ACCEPTED",
                "range_trader:CAD_CHF:LONG:RANGE_ROTATION",
                "orphan-1",
                "CAD_CHF",
                "LONG",
                1000,
                0.5665,
            ),
            (
                "canceled-orphan-1",
                _NOW - timedelta(minutes=32),
                "ORDER_CANCELED",
                "range_trader:CAD_CHF:LONG:RANGE_ROTATION",
                "orphan-1",
                "CAD_CHF",
                "LONG",
                1000,
                0.5665,
            ),
            (
                "accepted-orphan-2",
                _NOW - timedelta(minutes=25),
                "ORDER_ACCEPTED",
                "range_trader:NZD_JPY:SHORT:RANGE_ROTATION",
                "orphan-2",
                "NZD_JPY",
                "SHORT",
                -1000,
                93.56,
            ),
            (
                "canceled-orphan-2",
                _NOW - timedelta(minutes=18),
                "ORDER_CANCELED",
                "range_trader:NZD_JPY:SHORT:RANGE_ROTATION",
                "orphan-2",
                "NZD_JPY",
                "SHORT",
                -1000,
                93.56,
            ),
        ]
        for event_uid, ts, event_type, lane_id, order_id, pair, side, units, price in rows:
            conn.execute(
                """
                INSERT INTO execution_events(
                    event_uid, ts_utc, event_type, lane_id, order_id,
                    pair, side, units, price, exit_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (event_uid, ts.isoformat(), event_type, lane_id, order_id, pair, side, units, price),
            )


def _write_direct_manual_close_recovery_ledger(path: Path) -> None:
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE sync_state (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at_utc TEXT NOT NULL);
            CREATE TABLE execution_events (
                event_uid TEXT PRIMARY KEY,
                ts_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                lane_id TEXT,
                order_id TEXT,
                trade_id TEXT,
                pair TEXT,
                side TEXT,
                units INTEGER,
                price REAL,
                tp REAL,
                sl REAL,
                realized_pl_jpy REAL,
                exit_reason TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO sync_state(key, value, updated_at_utc) VALUES (?, ?, ?)",
            ("last_oanda_transaction_id", "100", _NOW.isoformat()),
        )
        rows = [
            (
                "fill-old-direct",
                (_NOW - timedelta(hours=49)).isoformat(),
                "ORDER_FILLED",
                "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE",
                "O-old",
                "T-old",
                "EUR_USD",
                "LONG",
                1000,
                None,
                None,
            ),
            (
                "close-old-direct",
                (_NOW - timedelta(hours=48)).isoformat(),
                "TRADE_CLOSED",
                "",
                "C-old",
                "T-old",
                "EUR_USD",
                "LONG",
                1000,
                -700.0,
                "MARKET_ORDER_TRADE_CLOSE",
            ),
            (
                "fill-gateway-loss",
                (_NOW - timedelta(hours=3)).isoformat(),
                "ORDER_FILLED",
                "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE",
                "O-gw",
                "T-gw",
                "USD_CAD",
                "LONG",
                1000,
                None,
                None,
            ),
            (
                "gateway-close-sent",
                (_NOW - timedelta(hours=2, minutes=5)).isoformat(),
                "GATEWAY_TRADE_CLOSE_SENT",
                "",
                "",
                "T-gw",
                "USD_CAD",
                "",
                None,
                None,
                "GPT_CLOSE",
            ),
            (
                "close-gateway-loss",
                (_NOW - timedelta(hours=2)).isoformat(),
                "TRADE_CLOSED",
                "",
                "C-gw",
                "T-gw",
                "USD_CAD",
                "LONG",
                1000,
                -50.0,
                "MARKET_ORDER_TRADE_CLOSE",
            ),
            (
                "close-tp-1",
                (_NOW - timedelta(hours=1)).isoformat(),
                "TRADE_CLOSED",
                "range_trader:EUR_CHF:LONG:RANGE_ROTATION",
                "TP-1",
                "T-tp-1",
                "EUR_CHF",
                "LONG",
                1000,
                100.0,
                "TAKE_PROFIT_ORDER",
            ),
            (
                "close-tp-2",
                (_NOW - timedelta(minutes=30)).isoformat(),
                "TRADE_CLOSED",
                "range_trader:EUR_GBP:SHORT:RANGE_ROTATION",
                "TP-2",
                "T-tp-2",
                "EUR_GBP",
                "SHORT",
                1000,
                100.0,
                "TAKE_PROFIT_ORDER",
            ),
        ]
        conn.executemany(
            """
            INSERT INTO execution_events(
                event_uid, ts_utc, event_type, lane_id, order_id, trade_id,
                pair, side, units, realized_pl_jpy, exit_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def _write_method_attribution_ledger(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE execution_events (
                event_uid TEXT PRIMARY KEY,
                ts_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                lane_id TEXT,
                order_id TEXT,
                trade_id TEXT,
                pair TEXT,
                side TEXT,
                realized_pl_jpy REAL
            );
            """
        )
        rows = [
            (
                "fill-range-1",
                (_NOW - timedelta(hours=6)).isoformat(),
                "ORDER_FILLED",
                "range_trader:EUR_USD:SHORT:RANGE_ROTATION",
                "O1",
                "T1",
                "EUR_USD",
                "SHORT",
                0.0,
            ),
            (
                "close-range-1",
                (_NOW - timedelta(hours=5)).isoformat(),
                "TRADE_CLOSED",
                "",
                "C1",
                "T1",
                "EUR_USD",
                "SHORT",
                -500.0,
            ),
            (
                "fill-range-2",
                (_NOW - timedelta(hours=4)).isoformat(),
                "ORDER_FILLED",
                "range_trader:EUR_USD:SHORT:RANGE_ROTATION",
                "O2",
                "T2",
                "EUR_USD",
                "SHORT",
                0.0,
            ),
            (
                "close-range-2",
                (_NOW - timedelta(hours=3)).isoformat(),
                "TRADE_CLOSED",
                "",
                "C2",
                "T2",
                "EUR_USD",
                "SHORT",
                -200.0,
            ),
            (
                "fill-trend",
                (_NOW - timedelta(hours=2)).isoformat(),
                "ORDER_FILLED",
                "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION",
                "O3",
                "T3",
                "EUR_USD",
                "SHORT",
                0.0,
            ),
            (
                "close-trend",
                (_NOW - timedelta(hours=1)).isoformat(),
                "TRADE_CLOSED",
                "",
                "C3",
                "T3",
                "EUR_USD",
                "SHORT",
                120.0,
            ),
        ]
        conn.executemany(
            """
            INSERT INTO execution_events(
                event_uid, ts_utc, event_type, lane_id, order_id, trade_id,
                pair, side, realized_pl_jpy
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def _write_market_close_attribution_ledger(
    path: Path,
    *,
    include_gateway_close: bool,
    include_reconciled_close: bool = False,
) -> None:
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE sync_state (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at_utc TEXT NOT NULL);
            CREATE TABLE execution_events (
                event_uid TEXT PRIMARY KEY,
                ts_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                lane_id TEXT,
                order_id TEXT,
                trade_id TEXT,
                pair TEXT,
                side TEXT,
                units INTEGER,
                price REAL,
                tp REAL,
                sl REAL,
                realized_pl_jpy REAL,
                exit_reason TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO sync_state(key, value, updated_at_utc) VALUES (?, ?, ?)",
            ("last_oanda_transaction_id", "100", _NOW.isoformat()),
        )
        rows = [
            (
                "gw-entry",
                (_NOW - timedelta(hours=3)).isoformat(),
                "GATEWAY_ORDER_SENT",
                "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET",
                "O42",
                "",
                "EUR_USD",
                "LONG",
                1000,
                None,
                None,
            ),
            (
                "fill",
                (_NOW - timedelta(hours=2, minutes=55)).isoformat(),
                "ORDER_FILLED",
                "",
                "O42",
                "T42",
                "EUR_USD",
                "LONG",
                1000,
                None,
                None,
            ),
        ]
        if include_gateway_close:
            rows.append(
                (
                    "gw-close",
                    (_NOW - timedelta(hours=1, minutes=5)).isoformat(),
                    "GATEWAY_TRADE_CLOSE_SENT",
                    "",
                    "",
                    "T42",
                    "EUR_USD",
                    "",
                    None,
                    None,
                    "GPT_CLOSE",
                )
            )
        if include_reconciled_close:
            rows.append(
                (
                    "gw-close-reconciled",
                    (_NOW - timedelta(hours=1, minutes=5)).isoformat(),
                    "GATEWAY_TRADE_CLOSE_RECONCILED",
                    "",
                    "C42",
                    "T42",
                    "EUR_USD",
                    "",
                    None,
                    None,
                    "GPT_CLOSE_RECONCILED",
                )
            )
        rows.append(
            (
                "close",
                (_NOW - timedelta(hours=1)).isoformat(),
                "TRADE_CLOSED",
                "",
                "C42",
                "T42",
                "EUR_USD",
                "LONG",
                1000,
                -500.0,
                "MARKET_ORDER_TRADE_CLOSE",
            )
        )
        conn.executemany(
            """
            INSERT INTO execution_events(
                event_uid, ts_utc, event_type, lane_id, order_id, trade_id,
                pair, side, units, realized_pl_jpy, exit_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def _insert_close_gate_pass_observation(
    conn: sqlite3.Connection,
    *,
    trade_id: str,
    ts_utc: str,
) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS verification_observations (
            ts_utc TEXT NOT NULL,
            subject_id TEXT,
            check_name TEXT NOT NULL,
            status TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO verification_observations(ts_utc, subject_id, check_name, status)
        VALUES (?, ?, ?, ?)
        """,
        (ts_utc, trade_id, "close_gate_evidence", "PASS"),
    )


def _add_raw_json_column(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("ALTER TABLE execution_events ADD COLUMN raw_json TEXT")
    except sqlite3.OperationalError as exc:
        if "duplicate column" not in str(exc).lower():
            raise


def _insert_broker_trade_close_accept(
    conn: sqlite3.Connection,
    *,
    trade_id: str = "T42",
    order_id: str = "C42",
) -> None:
    conn.execute(
        """
        INSERT INTO execution_events(
            event_uid, ts_utc, event_type, lane_id, order_id, trade_id,
            pair, side, units, realized_pl_jpy, exit_reason, raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            f"broker-close-accept-{trade_id}",
            (_NOW - timedelta(hours=1, minutes=10)).isoformat(),
            "ORDER_ACCEPTED",
            "",
            order_id,
            "",
            "EUR_USD",
            "SHORT",
            1000,
            None,
            "TRADE_CLOSE",
            json.dumps(
                {
                    "id": order_id,
                    "reason": "TRADE_CLOSE",
                    "tradeClose": {"tradeID": trade_id, "units": "ALL"},
                }
            ),
        ),
    )


def _insert_stale_gpt_close_satisfied(
    conn: sqlite3.Connection,
    *,
    trade_id: str = "T42",
) -> None:
    conn.execute(
        """
        INSERT INTO execution_events(
            event_uid, ts_utc, event_type, lane_id, order_id, trade_id,
            pair, side, units, realized_pl_jpy, exit_reason, raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            f"stale-gpt-close-satisfied-{trade_id}",
            (_NOW - timedelta(minutes=55)).isoformat(),
            "GATEWAY_POSITION_NO_ACTION",
            "",
            "",
            trade_id,
            "EUR_USD",
            "",
            None,
            None,
            "GPT_CLOSE",
            json.dumps(
                {
                    "management_action": "GPT_CLOSE",
                    "sent": False,
                    "request": None,
                    "issues": [
                        {
                            "severity": "INFO",
                            "code": "STALE_CLOSE_ALREADY_ABSENT",
                            "message": "accepted CLOSE receipt named a trade id that is already absent",
                        }
                    ],
                }
            ),
        ),
    )
