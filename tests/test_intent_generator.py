from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from quant_rabbit.market_close_leak_gate import MARKET_CLOSE_LEAK_FAMILY_BLOCK_CODE
from quant_rabbit.execution_ledger import ExecutionLedger
from quant_rabbit.models import (
    AccountSummary,
    BrokerPosition,
    BrokerSnapshot,
    MarketContext,
    OrderIntent,
    OrderType,
    Owner,
    Quote,
    Side,
    TradeMethod,
)
from quant_rabbit.risk import MARGIN_AWARE_BASKET_BUFFER, RiskIssue
from quant_rabbit.strategy.intent_generator import (
    IntentGenerator,
    MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCK_CODE,
    MONTH_SCALE_RESIDUAL_LOSS_REPAIR_BLOCK_CODE,
    OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED_CODE,
    POSITIVE_ROTATION_OANDA_CAMPAIGN_FIREPOWER_MODE,
    POSITIVE_ROTATION_FIREPOWER_BLOCK_CODE,
    POSITIVE_ROTATION_LIVE_BLOCK_CODE,
    POSITIVE_ROTATION_PROOF_COLLECTION_WARN_CODE,
    RANGE_TARGET_SPREAD_CUSHION_MULT,
    RANGE_WATCH_MATRIX_MAX_AGE_SECONDS,
    SELF_IMPROVEMENT_PROFITABILITY_P0_REPAIR_RECENT_LOSS_CODE,
    _annotate_oanda_campaign_current_risk_firepower,
    _append_current_range_phase_lanes,
    _append_forecast_seed_lanes,
    _capture_positive_rotation_live_issue,
    _exact_vehicle_net_metrics,
    _exact_vehicle_take_profit_metrics,
    _forecast_context_payload,
    _forecast_learning_scout_seed_lane,
    _forecast_learning_scout_seed_lanes,
    _forecast_live_readiness_issue,
    _forecast_watch_only_issue,
    _forecast_seed_lane,
    _geometry,
    _geometry_metadata,
    _intent_from_lane,
    _m15_recovery_micro_live_issue,
    _m15_recovery_chart_context_for,
    _minimum_range_target_pips,
    _oanda_campaign_firepower_path_for_data_root,
    _oanda_campaign_firepower_shape_matches_method,
    _oanda_campaign_vehicle_shape_reprice_metadata,
    _oanda_m5_rotation_state_for,
    _order_variants_for,
    _predictive_scout_nav_risk_metadata,
    _profitability_acceptance_month_residual_issue,
    _range_rail_limit_watch_only_metadata_can_trade,
    _range_indicators_for_lane,
    _range_seed_direction,
    _same_day_loss_streak_issues,
    _session_bucket_from_tag,
    _risk_budgeted_units,
    sizing_conversion_snapshot_receipt_from_payload,
)
from quant_rabbit.strategy.directional_forecaster import (
    M15_RECOVERY_MICRO_MAX_UNITS,
    build_m15_recovery_micro_receipt,
)
from quant_rabbit.strategy.lane_history_ledger import SameDayLaneLossStreak, SameDayLossStreak
from quant_rabbit.strategy.forecast_technical_context import (
    CONFIDENCE_SEMANTICS,
    build_forecast_technical_context,
    verify_forecast_technical_context_evidence,
)
from tests.support_bidask_rules import (
    bidask_rules_env,
    write_bidask_replay_fixture_rules,
    write_nonmatching_bidask_rules,
)


def _high_precision_market_support(direction: str) -> dict:
    return {
        "ok": True,
        "direction": direction,
        "aligned_projection_count": 1,
        "timing_projection_count": 0,
        "best_hit_rate": 1.0,
        "best_samples": 40,
        "best_aligned_hit_rate": 1.0,
        "best_aligned_samples": 40,
        "directional_calibration_name": f"directional_forecast_{direction.lower()}",
        "directional_hit_rate": 1.0,
        "directional_samples": 40,
        "reason": f"macro_event_nowcast_central_bank {direction} hit_rate=1.00 samples=40 supports forecast",
        "signals": [
            {
                "name": "macro_event_nowcast_central_bank",
                "direction": direction,
                "confidence": 0.9,
                "hit_rate": 1.0,
                "samples": 40,
            }
        ],
    }


def _capture_scoped_tp_payload(
    *,
    pair: str = "EUR_USD",
    side: str = "LONG",
    method: str = "RANGE_ROTATION",
    trades: int = 93,
    wins: int = 93,
    losses: int = 0,
    avg_win_jpy: float = 504.0,
    avg_loss_jpy: float = 0.0,
    expectancy_jpy_per_trade: float = 504.0,
) -> dict:
    metrics = {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "avg_win_jpy": avg_win_jpy,
        "avg_loss_jpy": avg_loss_jpy,
        "expectancy_jpy_per_trade": expectancy_jpy_per_trade,
    }
    return {
        "generated_at_utc": "2026-07-02T00:00:00+00:00",
        "by_pair_side_exit_reason": {
            pair: {
                side: {
                    "TAKE_PROFIT_ORDER": dict(metrics),
                }
            }
        },
        "by_pair_side_method_exit_reason": {
            pair: {
                side: {
                    method: {
                        "TAKE_PROFIT_ORDER": dict(metrics),
                    }
                }
            }
        },
    }


def _exact_vehicle_rotation_metadata(*, trades: int) -> dict[str, object]:
    avg_win_jpy = 500.0
    proven = trades >= 20
    return {
        "loss_asymmetry_guard_active": True,
        "capture_economics_status": "NEGATIVE_EXPECTANCY",
        "loss_asymmetry_guard_mode": (
            "TP_PROVEN_RELAXED" if proven else "CAP_AVG_WIN"
        ),
        "loss_asymmetry_guard_relaxed": proven,
        "loss_asymmetry_guard_loss_cap_jpy": 500.0,
        "loss_asymmetry_guard_base_max_loss_jpy": 1000.0,
        "loss_asymmetry_guard_effective_max_loss_jpy": (
            1000.0 if proven else 500.0
        ),
        "max_loss_jpy": 1000.0 if proven else 500.0,
        "attach_take_profit_on_fill": True,
        "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
        "tp_target_intent": "HARVEST",
        "opportunity_mode": "HARVEST",
        "capture_take_profit_exact_vehicle_required": True,
        "capture_take_profit_scope": "PAIR_SIDE_METHOD_VEHICLE",
        "capture_take_profit_scope_key": (
            "EUR_USD|LONG|RANGE_ROTATION|LIMIT|TAKE_PROFIT_ORDER"
        ),
        "capture_take_profit_vehicle": "LIMIT",
        "capture_take_profit_metrics_source": (
            "data/execution_ledger.db:exact_vehicle_take_profit"
        ),
        "capture_take_profit_trades": trades,
        "capture_take_profit_wins": trades,
        "capture_take_profit_losses": 0,
        "capture_take_profit_expectancy_jpy": avg_win_jpy,
        "capture_take_profit_net_jpy": trades * avg_win_jpy,
        "capture_take_profit_avg_win_jpy": avg_win_jpy,
        "capture_take_profit_avg_loss_jpy": 0.0,
        "capture_avg_win_jpy": 500.0,
        "capture_avg_loss_jpy": 100.0,
        "capture_market_close_expectancy_jpy": -100.0,
    }


def _write_exact_vehicle_rotation_wins(
    root: Path,
    *,
    method: str = "RANGE_ROTATION",
    count: int = 93,
    realized_pl_jpy: float = 504.0,
) -> None:
    desk = "range_trader" if method == "RANGE_ROTATION" else "failure_trader"
    _write_exact_vehicle_take_profit_closes(
        root,
        lane_id=f"{desk}:EUR_USD:LONG:{method}:LIMIT",
        pair="EUR_USD",
        side="LONG",
        entry_reason="LIMIT_ORDER",
        count=count,
        realized_pl_jpy=realized_pl_jpy,
    )


def _write_oanda_campaign_firepower_report(
    root: Path,
    *,
    status: str = "VERIFIED_TARGET_10_ROUTE_ESTIMATED",
    pair: str = "EUR_USD",
    side: str = "LONG",
    shape: str = "range_reversion",
    exit_shape: str = "tp1_sl1",
    aggregate_return_pct: float = 14.08,
    matching_return_pct: float = 2.8,
    weighted_return_pct: float = 0.64,
    observed_attempts_per_active_day: float = 22.0,
) -> Path:
    path = (
        root
        / "logs"
        / "reports"
        / "forecast_improvement"
        / "oanda_universal_rotation_mining_latest.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-06-21T00:00:00Z",
                "campaign_firepower": {
                    "contract": "audit-only firepower estimate; live gates still decide",
                    "per_trade_risk_pct_lens": 1.0,
                    "minimum_return_pct": 5.0,
                    "target_return_pct": 10.0,
                    "status": status,
                    "high_precision": {
                        "unique_vehicle_count": 5,
                        "pair_count": 5,
                        "observed_attempts_per_active_day": observed_attempts_per_active_day,
                        "weighted_return_pct_per_trade_at_risk_lens": weighted_return_pct,
                        "estimated_return_pct_per_active_day_at_observed_frequency": aggregate_return_pct,
                        "trades_needed_for_minimum_5pct_at_weighted_expectancy": 8,
                        "trades_needed_for_target_10pct_at_weighted_expectancy": 16,
                        "top_vehicles": [
                            {
                                "vehicle_key": f"{pair}|{side}|{shape}|{exit_shape}",
                                "exit_shape": exit_shape,
                                "evidence_status": "HIGH_PRECISION_VALIDATED",
                                "pair": pair,
                                "shape": shape,
                                "firepower_side": side,
                                "validation_n": 80,
                                "active_days": 20,
                                "estimated_return_pct_per_active_day_at_observed_frequency": matching_return_pct,
                                "live_permission": False,
                            }
                        ],
                    },
                    "evidence_queue": {
                        "unique_vehicle_count": 0,
                        "pair_count": 0,
                        "observed_attempts_per_active_day": 0.0,
                        "weighted_return_pct_per_trade_at_risk_lens": 0.0,
                        "estimated_return_pct_per_active_day_at_observed_frequency": 0.0,
                        "trades_needed_for_minimum_5pct_at_weighted_expectancy": None,
                        "trades_needed_for_target_10pct_at_weighted_expectancy": None,
                        "top_vehicles": [],
                    },
                },
            }
        )
    )
    return path


def _write_profitability_p0_and_negative_capture(root: Path) -> None:
    (root / "self_improvement_audit.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "findings": [
                    {
                        "priority": "P0",
                        "layer": "profitability",
                        "code": "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
                        "message": "market-close leakage is still negative",
                        "evidence": {
                            "current_streak": 65,
                            "system_defect_evidence": {
                                "profit_factor": 0.788,
                                "expectancy_jpy": -54.04,
                            },
                        },
                    }
                ],
            }
        )
    )
    (root / "capture_economics.json").write_text(
        json.dumps(
            {
                "status": "NEGATIVE_EXPECTANCY",
                "overall": {
                    "trades": 210,
                    "avg_win_jpy": 600.0,
                    "avg_loss_jpy": 1100.0,
                    "payoff_ratio": 0.545,
                    "breakeven_payoff_at_win_rate": 0.7,
                },
                "by_exit_reason": {
                    "TAKE_PROFIT_ORDER": {
                        "trades": 93,
                        "wins": 93,
                        "losses": 0,
                        "avg_win_jpy": 504.0,
                        "avg_loss_jpy": 0.0,
                        "expectancy_jpy_per_trade": 504.0,
                    },
                    "MARKET_ORDER_TRADE_CLOSE": {
                        "trades": 84,
                        "wins": 13,
                        "losses": 71,
                        "avg_win_jpy": 218.4,
                        "avg_loss_jpy": 1095.5,
                        "expectancy_jpy_per_trade": -892.1,
                    },
                },
            }
        )
    )


def _write_month_scale_residual_acceptance(
    root: Path,
    *,
    pair: str = "EUR_USD",
    side: str = "LONG",
    method: str = "RANGE_ROTATION",
    repair_replay_pl_jpy: float = -2333.8215,
    residual_scope: str | None = None,
    extra_groups: list[dict] | None = None,
) -> None:
    primary_group = {
        "pair": pair,
        "side": side,
        "method": method,
        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
        "loss_closes": 1,
        "repair_replay_pl_jpy": repair_replay_pl_jpy,
        "block_reasons": {"BELOW_TP_PROGRESS_GATE": 1},
        "examples": [
            {
                "trade_id": "472071",
                "lane_id": f"test:{pair}:{side}:{method}",
                "repair_replay_pl_jpy": repair_replay_pl_jpy,
            }
        ],
    }
    if residual_scope is not None:
        primary_group["residual_scope"] = residual_scope
    groups = [*(extra_groups or []), primary_group]
    (root / "profitability_acceptance.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "PROFITABILITY_ACCEPTANCE_BLOCKED",
                "findings": [
                    {
                        "priority": "P0",
                        "code": "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE",
                        "message": "month-scale replay remains negative",
                        "evidence": {
                            "window_lookback_hours": 744.0,
                            "repair_replay_counterfactual_pl_jpy": -13824.5957,
                            "top_repair_replay_residual_groups": groups,
                        },
                    }
                ],
                "metrics": {
                    "profit_capture_replay_repair": {
                        "top_repair_replay_residual_groups": groups
                    }
                },
            }
        )
    )


def _write_month_scale_residual_metrics_only(
    root: Path,
    *,
    pair: str = "EUR_USD",
    side: str = "LONG",
    method: str = "RANGE_ROTATION",
    repair_replay_pl_jpy: float = -2333.8215,
) -> None:
    (root / "profitability_acceptance.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "PROFITABILITY_ACCEPTANCE_PASS",
                "findings": [],
                "metrics": {
                    "profit_capture_replay_repair": {
                        "top_repair_replay_residual_groups": [
                            {
                                "pair": pair,
                                "side": side,
                                "method": method,
                                "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
                                "loss_closes": 1,
                                "repair_replay_pl_jpy": repair_replay_pl_jpy,
                            }
                        ]
                    }
                },
            }
        )
    )


def _write_month_scale_residual_timing_audit(
    root: Path,
    *,
    pair: str = "EUR_USD",
    side: str = "LONG",
    method: str = "RANGE_ROTATION",
    repair_replay_pl_jpy: float = -2333.8215,
    lookback_hours: float = 744.0,
) -> None:
    lane_id = f"range_trader:{pair}:{side}:{method}"
    (root / "execution_timing_audit.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "OK",
                "window": {"lookback_hours": lookback_hours},
                "precision": {
                    "profit_capture_repair_replay_contract": {
                        "name": "TP_PROGRESS_PRODUCTION_GATE_REPLAY"
                    }
                },
                "summary": {
                    "loss_close_repair_replay_counterfactual_pl_jpy": -13824.5957,
                    "top_repair_replay_residual_groups": [
                        {
                            "pair": pair,
                            "side": side,
                            "method": method,
                            "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
                            "loss_closes": 1,
                            "repair_replay_pl_jpy": repair_replay_pl_jpy,
                            "block_reasons": {"BELOW_TP_PROGRESS_GATE": 1},
                            "examples": [
                                {
                                    "trade_id": "472071",
                                    "lane_id": lane_id,
                                    "repair_replay_pl_jpy": repair_replay_pl_jpy,
                                }
                            ],
                        }
                    ],
                },
                "loss_close_regrets": [
                    {
                        "trade_id": "472071",
                        "lane_id": lane_id,
                        "pair": pair,
                        "side": side,
                        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
                        "realized_pl_jpy": -2981.8961,
                        "repair_replay_counterfactual_pl_jpy": repair_replay_pl_jpy,
                        "repair_replay_triggered_before_loss_close": False,
                        "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
                    }
                ],
            }
        )
    )


def _write_acceptance_style_p0_and_negative_capture(root: Path) -> None:
    """Mirror current live P0 shape: acceptance red, no legacy profitability P0."""

    (root / "self_improvement_audit.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "findings": [
                    {
                        "priority": "P0",
                        "layer": "execution_quality",
                        "code": "LOSS_CLOSE_PROFIT_CAPTURE_MISSED",
                        "message": "loss closes missed available TP-progress profit capture",
                        "evidence": {"estimated_gap_jpy": 646.508},
                    },
                    {
                        "priority": "P0",
                        "layer": "execution_quality",
                        "code": "POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE",
                        "message": "position guardian is inactive",
                        "evidence": {
                            "live_ready_lanes": 1,
                            "guardian": {
                                "active": False,
                                "active_source": "plist_missing",
                                "launchd_loaded": False,
                            },
                        },
                    },
                    {
                        "priority": "P0",
                        "layer": "target",
                        "code": "TARGET_OPEN_NO_LIVE_READY_LANES",
                        "message": "target is open but no executable lanes were visible",
                        "evidence": {"live_ready_lanes": 0},
                    },
                ],
            }
        )
    )
    (root / "capture_economics.json").write_text(
        json.dumps(
            {
                "status": "NEGATIVE_EXPECTANCY",
                "overall": {
                    "trades": 210,
                    "avg_win_jpy": 600.0,
                    "avg_loss_jpy": 1100.0,
                    "payoff_ratio": 0.545,
                    "breakeven_payoff_at_win_rate": 0.7,
                },
                "by_exit_reason": {
                    "TAKE_PROFIT_ORDER": {
                        "trades": 93,
                        "wins": 93,
                        "losses": 0,
                        "avg_win_jpy": 504.0,
                        "avg_loss_jpy": 0.0,
                        "expectancy_jpy_per_trade": 504.0,
                    },
                    "MARKET_ORDER_TRADE_CLOSE": {
                        "trades": 84,
                        "wins": 13,
                        "losses": 71,
                        "avg_win_jpy": 218.4,
                        "avg_loss_jpy": 1095.5,
                        "expectancy_jpy_per_trade": -892.1,
                    },
                },
            }
        )
    )


def _write_profitability_p0_with_matching_worst_segment(
    root: Path,
    *,
    close_provenance_net_jpy: dict[str, float],
) -> None:
    (root / "self_improvement_audit.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "findings": [
                    {
                        "priority": "P0",
                        "layer": "profitability",
                        "code": "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
                        "message": "market-close leakage is still negative",
                        "evidence": {
                            "current_streak": 65,
                            "system_defect_evidence": {
                                "profit_factor": 0.536,
                                "expectancy_jpy": -117.4,
                                "worst_segments": [
                                    {
                                        "pair": "EUR_USD",
                                        "side": "LONG",
                                        "method": "RANGE_ROTATION",
                                        "trades": 2,
                                        "net_jpy": sum(close_provenance_net_jpy.values()),
                                        "trade_ids": ["T-stop", "T-gateway"],
                                        "close_provenance_counts": {
                                            key: 1 for key in close_provenance_net_jpy
                                        },
                                        "close_provenance_net_jpy": close_provenance_net_jpy,
                                    }
                                ],
                            },
                        },
                    }
                ],
            }
        )
    )
    (root / "capture_economics.json").write_text(
        json.dumps(
            {
                "status": "NEGATIVE_EXPECTANCY",
                "overall": {
                    "trades": 210,
                    "avg_win_jpy": 600.0,
                    "avg_loss_jpy": 1100.0,
                    "payoff_ratio": 0.545,
                    "breakeven_payoff_at_win_rate": 0.7,
                },
                "by_exit_reason": {
                    "TAKE_PROFIT_ORDER": {
                        "trades": 93,
                        "wins": 93,
                        "losses": 0,
                        "avg_win_jpy": 504.0,
                        "avg_loss_jpy": 0.0,
                        "expectancy_jpy_per_trade": 504.0,
                    },
                    "MARKET_ORDER_TRADE_CLOSE": {
                        "trades": 84,
                        "wins": 13,
                        "losses": 71,
                        "avg_win_jpy": 218.4,
                        "avg_loss_jpy": 1095.5,
                        "expectancy_jpy_per_trade": -892.1,
                    },
                },
            }
        )
    )


def _oanda_seed_range_campaign(root: Path, *, side: str = "LONG") -> Path:
    path = root / "oanda_range_campaign.json"
    path.write_text(
        json.dumps(
            {
                "lanes": [
                    {
                        "desk": "range_trader",
                        "pair": "EUR_USD",
                        "direction": side,
                        "method": "RANGE_ROTATION",
                        "adoption": "ORDER_INTENT_REQUIRED",
                        "campaign_role": "OANDA_FIREPOWER_ROUTE",
                        "reason": "OANDA high precision range vehicle",
                        "required_receipt": "Build current non-market order intent.",
                        "target_reward_risk": 2.0,
                        "blockers": [],
                        "story_examples": ["OANDA campaign firepower fixture"],
                        "oanda_campaign_firepower_seed": True,
                        "oanda_campaign_vehicle_key": f"EUR_USD|{side}|range_reversion|tp2_sl1",
                        "oanda_campaign_vehicle_count": 1,
                        "oanda_campaign_vehicle_keys": [f"EUR_USD|{side}|range_reversion|tp2_sl1"],
                        "oanda_campaign_firepower_status": "VERIFIED_TARGET_10_ROUTE_ESTIMATED",
                        "oanda_campaign_exit_shape": "tp2_sl1",
                        "oanda_campaign_exit_shapes": ["tp2_sl1"],
                        "oanda_campaign_estimated_return_pct_per_active_day": 2.8,
                        "oanda_campaign_live_permission": False,
                    }
                ]
            }
        )
    )
    return path


class IntentGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self._default_root_tmp = tempfile.TemporaryDirectory()
        self._default_root_patch = patch(
            "quant_rabbit.strategy.intent_generator.ROOT",
            Path(self._default_root_tmp.name),
        )
        self._default_root_patch.start()
        self._prior_require_forecast_for_live = os.environ.pop(
            "QR_REQUIRE_FORECAST_FOR_LIVE",
            None,
        )
        self._prior_require_telemetry_for_live = os.environ.pop(
            "QR_REQUIRE_TELEMETRY_FOR_LIVE",
            None,
        )
        self._prior_bidask_rules = os.environ.get("QR_BIDASK_REPLAY_PRECISION_RULES")
        os.environ["QR_BIDASK_REPLAY_PRECISION_RULES"] = str(
            write_nonmatching_bidask_rules(Path(self._default_root_tmp.name))
        )

    def tearDown(self) -> None:
        self._default_root_patch.stop()
        self._default_root_tmp.cleanup()
        if self._prior_require_forecast_for_live is None:
            os.environ.pop("QR_REQUIRE_FORECAST_FOR_LIVE", None)
        else:
            os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = self._prior_require_forecast_for_live
        if self._prior_require_telemetry_for_live is None:
            os.environ.pop("QR_REQUIRE_TELEMETRY_FOR_LIVE", None)
        else:
            os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = self._prior_require_telemetry_for_live
        if self._prior_bidask_rules is None:
            os.environ.pop("QR_BIDASK_REPLAY_PRECISION_RULES", None)
        else:
            os.environ["QR_BIDASK_REPLAY_PRECISION_RULES"] = self._prior_bidask_rules

    def test_forecast_context_payload_preserves_verified_technical_context(self) -> None:
        context = build_forecast_technical_context(
            {
                "confluence": {
                    "dominant_regime": "TREND_UP",
                    "price_percentile_24h": 0.7,
                    "price_percentile_7d": 0.6,
                },
                "views": [
                    {
                        "granularity": "M5",
                        "regime_reading": {
                            "state": "TREND_STRONG",
                            "atr_percentile": 60.0,
                        },
                        "indicators": {"atr_pips": 2.0},
                        "structure": {
                            "structure_events": [
                                {
                                    "kind": "BOS_UP",
                                    "index": 1,
                                    "close_confirmed": True,
                                }
                            ]
                        },
                    }
                ],
            },
            pair="EUR_USD",
            current_price=1.1001,
            spread_pips=0.2,
        )
        payload = _forecast_context_payload(
            SimpleNamespace(
                pair="EUR_USD",
                direction="UP",
                confidence=0.72,
                current_price=1.1001,
                technical_context_v1=context,
            ),
            cycle_id="cycle-context",
        )

        evidence = payload["forecast_technical_context"]
        self.assertEqual(evidence["status"], "VALID")
        self.assertEqual(evidence["confidence_semantics"], CONFIDENCE_SEMANTICS)
        self.assertEqual(evidence["technical_context_v1"], context)
        self.assertEqual(evidence["context_sha256"], context["context_sha256"])
        receipt = context["regime_family_weighting"]
        self.assertEqual(
            payload["forecast_regime_family_weighting_sha256"],
            receipt["receipt_sha256"],
        )
        self.assertEqual(
            payload["forecast_regime_family_selected_method"],
            receipt["source_identity"]["selected_method"],
        )
        self.assertEqual(
            payload["forecast_regime_family_direction"],
            receipt["aggregate"]["direction"],
        )
        self.assertEqual(
            verify_forecast_technical_context_evidence(
                evidence,
                pair="EUR_USD",
                current_price=1.1001,
            ),
            (True, None),
        )

    def test_session_bucket_maps_judas_window_to_london_context(self) -> None:
        self.assertEqual(_session_bucket_from_tag("JUDAS_WINDOW"), "LONDON")

    def test_oanda_m5_rotation_state_exposes_mined_candle_features(self) -> None:
        prior_candles = [
            {"o": 100.40, "h": 101.00, "l": 100.00, "c": 100.50, "complete": True}
            for _ in range(21)
        ]
        current = {
            "o": 100.95,
            "h": 101.20,
            "l": 100.80,
            "c": 100.90,
            "complete": True,
        }

        state = _oanda_m5_rotation_state_for(
            "GBP_JPY",
            {
                "M5": {"atr_pips": 10.0},
                "M5__recent_candles": [*prior_candles, current],
            },
        )

        self.assertEqual(state["oanda_m5_bar_range"], "wide")
        self.assertEqual(state["oanda_m5_bar_range_atr"], 4.0)
        self.assertEqual(state["oanda_m5_range_pos_bucket"], "high")
        self.assertEqual(state["oanda_m5_wick_reject_short"], True)
        self.assertEqual(state["oanda_m5_wick_reject_long"], False)
        self.assertEqual(state["oanda_m5_failed_break_short"], True)
        self.assertEqual(state["oanda_m5_failed_break_long"], False)
        self.assertAlmostEqual(state["oanda_m5_upper_wick"], 0.625)
        self.assertAlmostEqual(state["oanda_m5_lower_wick"], 0.25)

    def test_side_matched_failed_break_is_persisted_as_entry_thesis_signal(self) -> None:
        from quant_rabbit.strategy.entry_thesis_ledger import (
            record_entry_thesis_from_response_result,
        )
        from quant_rabbit.strategy.intent_generator import _intent_from_lane

        prior_candles = [
            {"o": 1.1720, "h": 1.1750, "l": 1.1700, "c": 1.1720, "complete": True}
            for _ in range(21)
        ]
        current = {
            "o": 1.1698,
            "h": 1.1720,
            "l": 1.1690,
            "c": 1.1710,
            "complete": True,
        }
        chart_context = _oanda_m5_rotation_state_for(
            "EUR_USD",
            {
                "M5": {"atr_pips": 8.0},
                "M5__recent_candles": [*prior_candles, current],
            },
        )
        now = datetime(2026, 7, 13, 1, 0, tzinfo=timezone.utc)
        quote = Quote("EUR_USD", bid=1.1709, ask=1.1711, timestamp_utc=now)
        snapshot = BrokerSnapshot(
            fetched_at_utc=now,
            quotes={"EUR_USD": quote},
            account=AccountSummary(
                nav_jpy=200_000.0,
                balance_jpy=200_000.0,
                margin_available_jpy=200_000.0,
                fetched_at_utc=now,
            ),
            home_conversions={"USD": 156.0},
        )
        intent = _intent_from_lane(
            {
                "desk": "failure_trader",
                "pair": "EUR_USD",
                "direction": "LONG",
                "method": "BREAKOUT_FAILURE",
                "adoption": "ORDER_INTENT_REQUIRED",
                "campaign_role": "NOW",
                "reason": "failed downside break reclaimed support",
                "required_receipt": "wait for failed-break retest",
                "target_reward_risk": 2.0,
                "blockers": [],
                "story_examples": [],
            },
            quote,
            snapshot,
            max_loss_jpy=500.0,
            atr_pips=8.0,
            parent_lane_id="failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE",
            chart_context=chart_context,
        )

        self.assertIs(intent.metadata["failed_acceptance"], True)
        self.assertEqual(intent.metadata["failed_acceptance_side"], "LONG")
        self.assertEqual(intent.metadata["acceptance_zone"], 1.17)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "forecast_history.jsonl").write_text(
                json.dumps(
                    {
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "confidence": 0.72,
                        "timestamp_utc": "2026-07-13T00:55:00Z",
                        "cycle_id": "cycle-failed-acceptance",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = record_entry_thesis_from_response_result(
                response={
                    "orderFillTransaction": {
                        "id": "81002",
                        "orderID": "81001",
                        "time": "2026-07-13T01:00:03Z",
                        "type": "ORDER_FILL",
                        "reason": "MARKET_ORDER",
                        "tradeOpened": {"tradeID": "81003"},
                        "price": "1.17100",
                    }
                },
                intent=intent,
                data_root=root,
                now=now,
            )

        self.assertEqual(result.status, "RECORDED")
        assert result.thesis is not None
        signal = result.thesis.context_evidence["guardian_tuning_signal_state"]
        self.assertIs(signal["failed_acceptance"], True)
        self.assertEqual(signal["acceptance_zone"], 1.17)

    def test_oanda_pullback_continuation_shape_matches_trend_campaign_method(self) -> None:
        self.assertTrue(
            _oanda_campaign_firepower_shape_matches_method(
                "pullback_continuation",
                TradeMethod.TREND_CONTINUATION,
            )
        )
        self.assertFalse(
            _oanda_campaign_firepower_shape_matches_method(
                "pullback_continuation",
                TradeMethod.RANGE_ROTATION,
            )
        )

    def test_requires_snapshot_before_pricing_intents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = _campaign(root)
            strategy = _strategy(root)
            output = root / "intents.json"
            report = root / "intents.md"

            summary = IntentGenerator(
                campaign_plan=campaign,
                strategy_profile=strategy,
                output_path=output,
                report_path=report,
                max_loss_pct=0.5,
                risk_equity_jpy=200_000.0,
            ).run()

            self.assertEqual(summary.generated, 0)
            self.assertEqual(summary.needs_snapshot, 1)
            self.assertIn("NEEDS_BROKER_SNAPSHOT", report.read_text())

    def test_dedupes_duplicate_lane_keys_before_order_variant_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = root / "campaign.json"
            duplicate_lane = {
                "desk": "trend_trader",
                "pair": "EUR_USD",
                "direction": "LONG",
                "method": "TREND_CONTINUATION",
                "adoption": "RISK_REPAIR_DRY_RUN",
                "campaign_role": "NOW_IF_REPAIRED",
                "reason": "duplicate trend pressure",
                "required_receipt": "dry-run under loss cap",
                "blockers": [],
                "story_examples": ["same setup surfaced twice"],
            }
            campaign.write_text(json.dumps({"lanes": [duplicate_lane, dict(duplicate_lane)]}))
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=campaign,
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            lane_ids = [item["lane_id"] for item in payload["results"]]

            self.assertEqual(summary.generated, 2)
            self.assertEqual(len(lane_ids), len(set(lane_ids)))
            self.assertEqual(
                set(lane_ids),
                {
                    "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                    "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET",
                },
            )

    def test_forecast_seed_does_not_replace_oanda_firepower_lane_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lane = {
                "desk": "range_trader",
                "pair": "GBP_CHF",
                "direction": "SHORT",
                "method": TradeMethod.RANGE_ROTATION.value,
                "adoption": "ORDER_INTENT_REQUIRED",
                "campaign_role": "OANDA_FIREPOWER_ROUTE",
                "reason": "OANDA firepower route",
                "required_receipt": "build a current non-market receipt",
                "blockers": [],
                "oanda_campaign_firepower_seed": True,
                "oanda_campaign_vehicle_key": "GBP_CHF|SHORT|range_reversion|tp1.25_sl1",
                "oanda_campaign_vehicle_keys": ["GBP_CHF|SHORT|range_reversion|tp1.25_sl1"],
                "oanda_campaign_exit_shape": "tp1.25_sl1",
                "oanda_campaign_exit_shapes": ["tp1.25_sl1"],
            }
            forecast = SimpleNamespace(
                direction="DOWN",
                confidence=0.1,
                raw_confidence=0.1,
                calibration_multiplier=1.0,
                current_price=1.1111,
                target_price=1.1000,
                invalidation_price=1.1200,
                range_low_price=None,
                range_high_price=None,
                range_width_pips=None,
                horizon_min=60,
                rationale_summary="weak forecast should not erase firepower route",
                drivers_for=[],
                drivers_against=[],
                component_scores={},
                market_support={},
            )
            snapshot = BrokerSnapshot(
                fetched_at_utc=datetime(2026, 6, 22, tzinfo=timezone.utc),
                quotes={"GBP_CHF": Quote("GBP_CHF", bid=1.1110, ask=1.1112)},
            )

            with (
                patch(
                    "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                    return_value=forecast,
                ),
                patch(
                    "quant_rabbit.strategy.intent_generator._forecast_seed_methods",
                    return_value=[TradeMethod.RANGE_ROTATION.value],
                ),
                patch(
                    "quant_rabbit.strategy.intent_generator._forecast_seed_min_confidence_for_direction",
                    return_value=0.62,
                ),
                patch(
                    "quant_rabbit.strategy.intent_generator._forecast_market_support_allows_side",
                    return_value=False,
                ),
                patch(
                    "quant_rabbit.strategy.intent_generator._forecast_watch_candidate_reason",
                    return_value="below live forecast floor",
                ),
            ):
                lanes = _append_forecast_seed_lanes(
                    [lane],
                    {"GBP_CHF": {}},
                    snapshot,
                    data_root=root,
                    forecast_cycle_id="cycle-1",
                )

            self.assertEqual(len(lanes), 1)
            kept = lanes[0]
            self.assertEqual(kept["campaign_role"], "OANDA_FIREPOWER_ROUTE")
            self.assertTrue(kept["oanda_campaign_firepower_seed"])
            self.assertEqual(
                kept["oanda_campaign_vehicle_key"],
                "GBP_CHF|SHORT|range_reversion|tp1.25_sl1",
            )
            self.assertEqual(kept["forecast_direction"], "DOWN")
            self.assertNotIn("forecast_watch_only", kept)

    def test_forecast_seed_lanes_share_one_hit_rate_snapshot_across_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = BrokerSnapshot(
                fetched_at_utc=datetime(2026, 7, 10, tzinfo=timezone.utc),
                quotes={
                    "EUR_USD": Quote("EUR_USD", bid=1.1683, ask=1.1685),
                    "USD_JPY": Quote("USD_JPY", bid=158.34, ask=158.36),
                },
            )
            charts = {"EUR_USD": {}, "USD_JPY": {}}
            hit_rates = {
                "directional_forecast_up": {
                    "EUR_USD:TREND": {"hit_rate": 0.7, "samples": 20}
                }
            }

            with patch(
                "quant_rabbit.strategy.projection_ledger.compute_hit_rates",
                return_value=hit_rates,
            ) as compute_hit_rates, patch(
                "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                return_value=None,
            ) as forecast_for_pair:
                result = _append_forecast_seed_lanes(
                    [],
                    charts,
                    snapshot,
                    data_root=root,
                    forecast_cycle_id="cycle-shared-calibration",
                )

            self.assertEqual(result, [])
            compute_hit_rates.assert_called_once_with(root)
            self.assertEqual(forecast_for_pair.call_count, 2)
            for call in forecast_for_pair.call_args_list:
                self.assertIs(call.kwargs["hit_rates"], hit_rates)

    def test_forecast_learning_scout_selects_top_non_manual_pair_and_limit_only(self) -> None:
        now = datetime(2026, 7, 15, 1, 0, tzinfo=timezone.utc)

        def forecast(pair: str, probability: float, rank_direction: str) -> SimpleNamespace:
            return SimpleNamespace(
                pair=pair,
                direction="UP",
                confidence=0.71,
                raw_confidence=0.73,
                calibration_multiplier=1.0,
                current_price=1.0,
                target_price=1.01,
                invalidation_price=0.99,
                range_low_price=None,
                range_high_price=None,
                range_width_pips=None,
                horizon_min=0,
                rationale_summary="current technical forecast",
                drivers_for=[],
                drivers_against=[],
                component_scores={"UP": 70.0, "DOWN": 30.0, "RANGE": 20.0},
                market_support={},
                technical_context_v1={},
                forecast_learning_v1={
                    "model_status": "RANK_ONLY",
                    "original_direction": "UP",
                    "rank_direction": rank_direction,
                    "orientation": "DIRECT" if rank_direction == "UP" else "INVERSE",
                    "selected_orientation_probability": probability,
                    "ranking_horizon_min": 180,
                    "features": {
                        "technical_selected_method": "TREND_CONTINUATION",
                        "technical_family_direction_alignment": (
                            "ALIGNED" if rank_direction == "UP" else "CONTRADICTED"
                        ),
                    },
                },
            )

        source = {
            "desk": "trend_trader",
            "pair": "AUD_JPY",
            "direction": "LONG",
            "method": TradeMethod.TREND_CONTINUATION.value,
            "reason": "source context",
            "blockers": ["stale source blocker"],
        }
        snapshot = BrokerSnapshot(
            fetched_at_utc=now,
            positions=(
                BrokerPosition(
                    trade_id="manual-eurusd",
                    pair="EUR_USD",
                    side=Side.SHORT,
                    units=1000,
                    entry_price=1.17,
                    owner=Owner.OPERATOR_MANUAL,
                ),
            ),
        )
        seed = _forecast_learning_scout_seed_lane(
            [source],
            snapshot,
            {
                "EUR_USD": forecast("EUR_USD", 0.99, "DOWN"),
                "AUD_JPY": forecast("AUD_JPY", 0.74, "DOWN"),
            },
            source_by_pair={"AUD_JPY": source},
            existing_by_key={
                (
                    source["desk"],
                    source["pair"],
                    source["direction"],
                    source["method"],
                ): source
            },
            cycle_id="learning-cycle",
        )

        self.assertIsNotNone(seed)
        assert seed is not None
        self.assertEqual(seed["pair"], "AUD_JPY")
        self.assertEqual(seed["direction"], "SHORT")
        self.assertEqual(seed["method"], "TREND_CONTINUATION")
        self.assertEqual(seed["desk"], "trend_trader")
        self.assertEqual(seed["campaign_role"], "FORECAST_LEARNING_SCOUT")
        self.assertEqual(seed["predictive_scout_source"], "FORECAST_ORIENTATION_LEARNING")
        self.assertEqual(seed["forecast_horizon_min"], 180)
        self.assertEqual(seed["blockers"], [])
        self.assertEqual(_order_variants_for(seed), (OrderType.LIMIT,))

    def test_forecast_learning_scout_keeps_ranked_fallback_pairs(self) -> None:
        now = datetime(2026, 7, 15, 1, 0, tzinfo=timezone.utc)

        def forecast(
            probability: float,
            rank_direction: str,
            method: str = "TREND_CONTINUATION",
        ) -> SimpleNamespace:
            return SimpleNamespace(
                direction="UP",
                confidence=0.71,
                raw_confidence=0.73,
                calibration_multiplier=1.0,
                current_price=1.0,
                target_price=1.01,
                invalidation_price=0.99,
                range_low_price=None,
                range_high_price=None,
                range_width_pips=None,
                horizon_min=60,
                rationale_summary="current technical forecast",
                drivers_for=[],
                drivers_against=[],
                component_scores={"UP": 70.0, "DOWN": 30.0, "RANGE": 20.0},
                market_support={},
                technical_context_v1={},
                forecast_learning_v1={
                    "model_status": "RANK_ONLY",
                    "original_direction": "UP",
                    "rank_direction": rank_direction,
                    "orientation": "DIRECT" if rank_direction == "UP" else "INVERSE",
                    "selected_orientation_probability": probability,
                    "features": {
                        "technical_selected_method": method,
                        "technical_family_direction_alignment": (
                            "ALIGNED" if rank_direction == "UP" else "CONTRADICTED"
                        ),
                    },
                },
            )

        snapshot = BrokerSnapshot(
            fetched_at_utc=now,
            positions=(
                BrokerPosition(
                    trade_id="manual-eurusd",
                    pair="EUR_USD",
                    side=Side.SHORT,
                    units=1000,
                    entry_price=1.17,
                    owner=Owner.OPERATOR_MANUAL,
                ),
            ),
        )
        forecasts = {
            "EUR_USD": forecast(0.99, "DOWN"),
            "GBP_USD": forecast(0.95, "UP", "NONE"),
            "GBP_CAD": forecast(0.88, "UP"),
            "AUD_JPY": forecast(0.79, "DOWN"),
            "NZD_USD": forecast(0.68, "UP"),
        }

        with patch(
            "quant_rabbit.strategy.intent_generator.FORECAST_LEARNING_SCOUT_MAX_SEEDS",
            2,
        ):
            seeds = _forecast_learning_scout_seed_lanes(
                [],
                snapshot,
                forecasts,
                source_by_pair={},
                existing_by_key={},
                cycle_id="learning-fallback-cycle",
            )

        self.assertEqual(
            [(lane["pair"], lane["direction"]) for lane in seeds],
            [("GBP_CAD", "LONG"), ("AUD_JPY", "SHORT")],
        )
        self.assertTrue(all(_order_variants_for(lane) == (OrderType.LIMIT,) for lane in seeds))

    def test_forecast_learning_scout_skips_inverse_rank_that_contradicts_trend_method(self) -> None:
        now = datetime(2026, 7, 15, 1, 0, tzinfo=timezone.utc)

        def forecast(
            pair: str,
            probability: float,
            rank_direction: str,
            family_alignment: str,
        ) -> SimpleNamespace:
            return SimpleNamespace(
                pair=pair,
                direction="UP",
                confidence=0.71,
                raw_confidence=0.73,
                calibration_multiplier=1.0,
                current_price=1.0,
                target_price=1.01,
                invalidation_price=0.99,
                range_low_price=None,
                range_high_price=None,
                range_width_pips=None,
                horizon_min=60,
                rationale_summary="current technical forecast",
                drivers_for=[],
                drivers_against=[],
                component_scores={"UP": 70.0, "DOWN": 30.0, "RANGE": 20.0},
                market_support={},
                technical_context_v1={},
                forecast_learning_v1={
                    "model_status": "RANK_ONLY",
                    "original_direction": "UP",
                    "rank_direction": rank_direction,
                    "orientation": "DIRECT" if rank_direction == "UP" else "INVERSE",
                    "selected_orientation_probability": probability,
                    "features": {
                        "technical_selected_method": "TREND_CONTINUATION",
                        "technical_family_direction_alignment": family_alignment,
                    },
                },
            )

        snapshot = BrokerSnapshot(fetched_at_utc=now)
        with patch(
            "quant_rabbit.strategy.intent_generator.FORECAST_LEARNING_SCOUT_MAX_SEEDS",
            1,
        ):
            seeds = _forecast_learning_scout_seed_lanes(
                [],
                snapshot,
                {
                    "GBP_AUD": forecast("GBP_AUD", 0.91, "DOWN", "ALIGNED"),
                    "AUD_JPY": forecast("AUD_JPY", 0.72, "UP", "ALIGNED"),
                },
                source_by_pair={},
                existing_by_key={},
                cycle_id="learning-method-compatibility-cycle",
            )

        self.assertEqual(
            [(lane["pair"], lane["direction"], lane["method"]) for lane in seeds],
            [("AUD_JPY", "LONG", "TREND_CONTINUATION")],
        )

    def test_forecast_learning_scout_bypasses_only_broad_replay_negative(self) -> None:
        metadata = {
            "forecast_direction": "UP",
            "forecast_confidence": 0.70,
            "predictive_scout_source": "FORECAST_ORIENTATION_LEARNING",
        }
        intent = OrderIntent(
            pair="AUD_JPY",
            side=Side.LONG,
            order_type=OrderType.LIMIT,
            units=1,
            entry=114.20,
            tp=114.25,
            sl=114.15,
            thesis="bounded forecast-learning forward evidence",
            market_context=MarketContext(
                regime="BREAKOUT_FAILURE retest",
                narrative="ranked historical-tick forecast",
                chart_story="passive retest",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="technical stop",
            ),
            metadata=metadata,
        )
        broad_negative = {
            "code": "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
            "message": "broad pair-direction history is negative",
            "severity": "BLOCK",
        }

        with patch(
            "quant_rabbit.strategy.intent_generator._require_forecast_for_live_active",
            return_value=True,
        ), patch(
            "quant_rabbit.strategy.intent_generator._technical_harvest_negative_precision_issue_for_intent",
            return_value=None,
        ), patch(
            "quant_rabbit.strategy.intent_generator._predictive_scout_forward_evidence_allowed",
            return_value=True,
        ), patch(
            "quant_rabbit.strategy.intent_generator._bidask_replay_negative_precision_issue_for_intent",
            return_value=broad_negative,
        ) as broad_check:
            issue = _forecast_live_readiness_issue(
                intent,
                metadata,
                TradeMethod.BREAKOUT_FAILURE,
            )

        self.assertIsNone(issue)
        broad_check.assert_not_called()
        self.assertEqual(
            metadata["forecast_precision_basis"],
            "CURRENT_BIDASK_ORIENTATION_LEARNING_SCOUT",
        )

        technical_negative = {
            "code": "TECHNICAL_HARVEST_NEGATIVE_BUCKET_FOR_LIVE",
            "message": "exact technical state is negative",
            "severity": "BLOCK",
        }
        with patch(
            "quant_rabbit.strategy.intent_generator._require_forecast_for_live_active",
            return_value=True,
        ), patch(
            "quant_rabbit.strategy.intent_generator._technical_harvest_negative_precision_issue_for_intent",
            return_value=technical_negative,
        ), patch(
            "quant_rabbit.strategy.intent_generator._predictive_scout_forward_evidence_allowed",
            return_value=True,
        ):
            issue = _forecast_live_readiness_issue(
                intent,
                metadata,
                TradeMethod.BREAKOUT_FAILURE,
            )

        self.assertEqual(issue, technical_negative)

    def test_jsonl_dict_reader_streams_valid_dict_rows(self) -> None:
        from quant_rabbit.strategy.intent_generator import _iter_jsonl_dicts

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.jsonl"
            path.write_text('{"row": 1}\nnot-json\n[2]\n{"row": 3}\n')

            rows = _iter_jsonl_dicts(path)

            self.assertIs(iter(rows), rows)
            self.assertEqual(list(rows), [{"row": 1}, {"row": 3}])

    def test_bidask_replay_precision_seed_adds_exact_limit_harvest_lane(self) -> None:
        from quant_rabbit.strategy.intent_generator import _order_variants_for

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rules_path = root / "bidask_live_grade.json"
            rules_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "generated_at_utc": "2026-06-22T00:00:00Z",
                        "generated_from": "unit-test",
                        "edge_rules": [],
                        "negative_rules": [],
                        "contrarian_edge_rules": [
                            {
                                "name": "AUD_JPY_UP_FADE_TO_DOWN_S5_BIDASK_CONTRARIAN_HARVEST_TP10_SL7",
                                "pair": "AUD_JPY",
                                "side": "SHORT",
                                "direction": "DOWN",
                                "forecast_direction": "UP",
                                "faded_direction": "UP",
                                "contrarian_edge": True,
                                "confidence_bucket": "0.75-0.90",
                                "horizon_bucket": "61-240m",
                                "granularity": "S5",
                                "samples": 124,
                                "directional_hit_rate": 0.76,
                                "avg_final_pips": 5.8,
                                "avg_mfe_pips": 12.0,
                                "avg_mae_pips": 4.5,
                                "optimized_take_profit_pips": 10.0,
                                "optimized_stop_loss_pips": 7.0,
                                "optimized_avg_realized_pips": 2.4,
                                "optimized_win_rate": 0.70,
                                "optimized_profit_factor": 2.5,
                                "daily_stability_status": "DAILY_STABLE",
                                "active_days": 4,
                                "positive_day_rate": 0.75,
                                "min_target_pips": 9.8,
                                "max_target_pips": 10.5,
                                "max_stop_pips": 7.2,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            source = {
                "desk": "range_trader",
                "pair": "AUD_JPY",
                "direction": "LONG",
                "method": TradeMethod.RANGE_ROTATION.value,
                "adoption": "ORDER_INTENT_REQUIRED",
                "campaign_role": "BASE_ROUTE",
                "reason": "source lane",
                "required_receipt": "build a current non-market receipt",
                "blockers": [],
            }
            forecast = SimpleNamespace(
                direction="UP",
                confidence=0.80,
                raw_confidence=0.80,
                calibration_multiplier=1.0,
                current_price=114.289,
                target_price=114.389,
                invalidation_price=114.189,
                range_low_price=None,
                range_high_price=None,
                range_width_pips=None,
                horizon_min=90,
                rationale_summary="UP forecast bucket has audited fade edge",
                drivers_for=[],
                drivers_against=[],
                component_scores={},
                market_support={},
            )
            snapshot = BrokerSnapshot(
                fetched_at_utc=datetime(2026, 6, 22, tzinfo=timezone.utc),
                quotes={"AUD_JPY": Quote("AUD_JPY", bid=114.288, ask=114.290)},
            )
            charts = {"AUD_JPY": {"confluence": {"score_balance": "LONG_LEAN"}}}

            with (
                bidask_rules_env(rules_path),
                patch(
                    "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                    return_value=forecast,
                ),
                patch(
                    "quant_rabbit.strategy.intent_generator._forecast_seed_methods",
                    return_value=[],
                ),
                patch(
                    "quant_rabbit.strategy.intent_generator._record_forecast_seed_telemetry",
                ),
            ):
                lanes = _append_forecast_seed_lanes(
                    [source],
                    charts,
                    snapshot,
                    data_root=root,
                    forecast_cycle_id="cycle-1",
                )

        seed = lanes[0]
        self.assertTrue(seed["bidask_replay_precision_seed"])
        self.assertEqual(seed["desk"], "failure_trader")
        self.assertEqual(seed["pair"], "AUD_JPY")
        self.assertEqual(seed["direction"], "SHORT")
        self.assertEqual(seed["method"], TradeMethod.BREAKOUT_FAILURE.value)
        self.assertEqual(seed["campaign_role"], "BIDASK_REPLAY_CONTRARIAN_SCOUT")
        self.assertTrue(seed["predictive_scout"])
        self.assertEqual(seed["predictive_scout_source"], "BIDASK_REPLAY_PRECISION")
        self.assertEqual(_order_variants_for(seed), (OrderType.LIMIT,))
        self.assertEqual(seed["forecast_direction"], "UP")
        self.assertIn("attached broker TP and SL", seed["required_receipt"])
        self.assertIn("not this LIMIT fill vehicle", seed["required_receipt"])
        self.assertEqual(seed["bidask_replay_precision_seed_rule"]["scalp_tp_pips"], 10.0)
        self.assertEqual(seed["bidask_replay_precision_seed_rule"]["scalp_stop_pips"], 7.0)

    def test_predictive_scout_discovery_units_derive_from_current_nav_and_stop_distance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            config_root = root / "config"
            data_root.mkdir()
            config_root.mkdir()
            policy = json.loads(
                (Path(__file__).parents[1] / "config" / "predictive_scout_policy.json").read_text()
            )
            (config_root / "predictive_scout_policy.json").write_text(
                json.dumps(policy),
                encoding="utf-8",
            )
            ExecutionLedger(
                db_path=data_root / "execution_ledger.db",
                report_path=root / "execution_ledger.md",
            )._init_db()
            now = datetime(2026, 7, 10, 3, 0, tzinfo=timezone.utc)
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                quotes={
                    "USD_CAD": Quote("USD_CAD", bid=1.4158, ask=1.4159, timestamp_utc=now),
                },
                account=AccountSummary(
                    nav_jpy=200_000.0,
                    balance_jpy=200_000.0,
                    margin_available_jpy=200_000.0,
                    last_transaction_id="100",
                    fetched_at_utc=now,
                ),
                home_conversions={"CAD": 108.0},
            )
            lane = {
                "predictive_scout": True,
                "desk": "failure_trader",
                "campaign_role": "BIDASK_REPLAY_CONTRARIAN_SCOUT",
                "forecast_cycle_id": "cycle-current-nav",
                "forecast_direction": "DOWN",
                "forecast_confidence": 0.55,
                "forecast_horizon_min": 45,
                "bidask_replay_precision_seed_rule": {
                    "forecast_direction": "DOWN",
                    "faded_direction": "DOWN",
                    "horizon_bucket": "31-60m",
                    "confidence_bucket": "0.50-0.65",
                    "granularity": "S5",
                },
            }
            runtime_metadata = {
                "predictive_scout": True,
                "predictive_scout_source": "BIDASK_REPLAY_PRECISION",
                "predictive_scout_rule_name": "unit-test-rule",
                "predictive_scout_rule_digest": "unit-test-digest",
            }

            metadata, max_loss_jpy = _predictive_scout_nav_risk_metadata(
                lane,
                pair="USD_CAD",
                side=Side.LONG,
                order_type=OrderType.LIMIT,
                method=TradeMethod.BREAKOUT_FAILURE,
                entry=1.4159,
                tp=1.4169,
                sl=1.4152,
                snapshot=snapshot,
                runtime_metadata=runtime_metadata,
                data_root=data_root,
            )
            units = _risk_budgeted_units(
                "USD_CAD",
                1.4159,
                1.4152,
                max_loss_jpy=float(max_loss_jpy or 0.0),
                snapshot=snapshot,
                side=Side.LONG,
                loss_budget_target=True,
            )

        self.assertEqual(metadata["predictive_scout_nav_risk_plan_status"], "READY")
        self.assertEqual(metadata["predictive_scout_risk_tier"], "DISCOVERY")
        self.assertEqual(metadata["predictive_scout_nav_jpy_at_sizing"], 200_000.0)
        self.assertEqual(max_loss_jpy, 200.0)
        risk_per_unit_jpy = abs(1.4159 - 1.4152) * 108.0
        self.assertEqual(units, 2645)
        self.assertLessEqual(units * risk_per_unit_jpy, max_loss_jpy)
        self.assertGreater((units + 1) * risk_per_unit_jpy, max_loss_jpy)

    def test_predictive_scout_loss_warn_uses_exact_vehicle_gate_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime(2026, 7, 10, 3, 0, tzinfo=timezone.utc)
            lane = {
                "desk": "failure_trader",
                "pair": "USD_CAD",
                "direction": "LONG",
                "method": TradeMethod.BREAKOUT_FAILURE.value,
                "campaign_role": "BIDASK_REPLAY_CONTRARIAN_SCOUT",
            }
            intent = OrderIntent(
                pair="USD_CAD",
                side=Side.LONG,
                order_type=OrderType.LIMIT,
                units=1000,
                entry=1.3500,
                tp=1.3510,
                sl=1.3493,
                thesis="bounded forward evidence",
                owner=Owner.TRADER,
                market_context=MarketContext(
                    regime="range",
                    narrative="reproducible forecast failure bucket",
                    chart_story="passive retest",
                    method=TradeMethod.BREAKOUT_FAILURE,
                    invalidation="broker stop",
                ),
                metadata={"predictive_scout": True},
            )
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                quotes={
                    "USD_CAD": Quote(
                        "USD_CAD",
                        bid=1.3501,
                        ask=1.3503,
                        timestamp_utc=now,
                    )
                },
            )
            loss_streak = SameDayLaneLossStreak(
                pair="USD_CAD",
                side="LONG",
                method=TradeMethod.BREAKOUT_FAILURE.value,
                consecutive_losses=1,
                net_loss_jpy=-300.0,
                last_loss_ts_utc="2026-07-10T01:00:00+00:00",
            )
            p0_issue = {
                "code": "SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE",
                "message": "profitability P0 remains active",
                "severity": "BLOCK",
            }
            clean_risk = SimpleNamespace(allowed=True, issues=(), metrics=None)
            generator = IntentGenerator(
                campaign_plan=root / "campaign.json",
                strategy_profile=root / "strategy.json",
                output_path=root / "intents.json",
                report_path=root / "intents.md",
                data_root=root,
                max_loss_jpy=500.0,
            )

            with (
                patch(
                    "quant_rabbit.strategy.intent_generator._intent_from_lane",
                    return_value=intent,
                ),
                patch(
                    "quant_rabbit.strategy.intent_generator.RiskEngine.validate",
                    return_value=clean_risk,
                ),
                patch(
                    "quant_rabbit.strategy.intent_generator._atr_pips_for",
                    return_value=10.0,
                ),
                patch(
                    "quant_rabbit.strategy.intent_generator._predictive_scout_p0_forward_evidence_allowed",
                    return_value=True,
                ),
                patch(
                    "quant_rabbit.strategy.intent_generator._self_improvement_profitability_p0_repair_allowed",
                    return_value=False,
                ),
                patch(
                    "quant_rabbit.strategy.intent_generator.predictive_scout_intent_issues",
                    return_value=[],
                ),
            ):
                result = generator._build_for_lane(
                    lane,
                    snapshot,
                    None,
                    max_loss_jpy=500.0,
                    pair_charts={"USD_CAD": {}},
                    validation_time_utc=now,
                    data_root=root,
                    repair_loss_streaks={
                        ("USD_CAD", "LONG", TradeMethod.BREAKOUT_FAILURE.value): loss_streak
                    },
                    self_improvement_profitability_issue=p0_issue,
                )

            issue_codes = {issue["code"] for issue in result.risk_issues}
            metadata = result.intent["metadata"] if result.intent is not None else {}

            self.assertEqual(result.status, "LIVE_READY")
            self.assertTrue(result.risk_allowed)
            self.assertEqual(result.live_blockers, ())
            self.assertNotIn(
                SELF_IMPROVEMENT_PROFITABILITY_P0_REPAIR_RECENT_LOSS_CODE,
                issue_codes,
            )
            self.assertNotIn("self_improvement_p0_repair_recent_lane_loss", metadata)
            self.assertIn("PREDICTIVE_SCOUT_RECENT_LANE_LOSS_RECORDED", issue_codes)
            self.assertNotIn(
                "PREDICTIVE_SCOUT_RECENT_LANE_LOSS_RECORDED",
                result.live_blocker_codes,
            )

    def test_unclear_zero_forecast_seed_is_kept_for_auditable_blocker(self) -> None:
        from quant_rabbit.strategy.directional_forecaster import DirectionalForecast
        from quant_rabbit.strategy.intent_generator import _forecast_seed_for_pair

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_chart = {
                "pair": "EUR_USD",
                "confluence": {"dominant_regime": "UNCLEAR"},
                "views": [
                    {
                        "granularity": "M5",
                        "regime_reading": {"state": "TREND_WEAK"},
                        "family_scores": {"trend_score": 0.1},
                        "indicators": {"close": 1.1733, "atr_pips": 8.0},
                    },
                    {
                        "granularity": "M15",
                        "regime_reading": {"state": "RANGE"},
                        "family_scores": {"mean_rev_score": 0.1},
                        "indicators": {"close": 1.1733, "atr_pips": 12.0},
                    },
                ],
            }
            charts = {"EUR_USD": {"__raw_chart": raw_chart}}
            snapshot = BrokerSnapshot(
                fetched_at_utc=datetime(2026, 6, 22, tzinfo=timezone.utc),
                quotes={
                    "EUR_USD": Quote(
                        "EUR_USD",
                        bid=1.1732,
                        ask=1.1734,
                        timestamp_utc=datetime(2026, 6, 22, tzinfo=timezone.utc),
                    )
                },
            )
            unclear = DirectionalForecast(
                pair="EUR_USD",
                direction="UNCLEAR",
                confidence=0.0,
                invalidation_price=None,
                target_price=None,
                horizon_min=0,
                drivers_for=("UP and DOWN are tied",),
                drivers_against=("no executable edge",),
                rationale_summary="contested: UP=25.0 vs DOWN=25.0",
                current_price=1.1733,
                component_scores={"UP": 25.0, "DOWN": 25.0, "RANGE": 0.0, "EITHER": 0.0},
            )

            with (
                patch("quant_rabbit.strategy.pattern_signals.detect_pattern_signals", return_value=[]),
                patch("quant_rabbit.strategy.forward_projection.detect_forward_projections", return_value=[]),
                patch("quant_rabbit.strategy.correlation_predictor.detect_correlation_lag", return_value=[]),
                patch("quant_rabbit.strategy.path_projection.detect_paths", return_value=[]),
                patch("quant_rabbit.strategy.reversal_signal.detect_reversal", return_value=None),
                patch("quant_rabbit.strategy.projection_ledger.compute_hit_rates", return_value={}),
                patch("quant_rabbit.strategy.directional_forecaster.synthesize_forecast", return_value=unclear),
            ):
                seed = _forecast_seed_for_pair("EUR_USD", charts, snapshot, data_root=root)

            self.assertIsNotNone(seed)
            self.assertEqual(getattr(seed, "direction", None), "UNCLEAR")
            self.assertEqual(getattr(seed, "confidence", None), 0.0)

    def test_refuses_stale_campaign_plan_when_target_state_is_newer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            old_ts = datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat()
            current_ts = datetime(2026, 1, 2, tzinfo=timezone.utc).isoformat()
            campaign = data_root / "daily_campaign_plan.json"
            campaign.write_text(
                json.dumps(
                    {
                        "generated_at_utc": old_ts,
                        "start_balance_jpy": 200000.0,
                        "target_jpy": 20000.0,
                        "lanes": [
                            {
                                "desk": "trend_trader",
                                "pair": "EUR_USD",
                                "direction": "LONG",
                                "method": "TREND_CONTINUATION",
                                "adoption": "ORDER_INTENT_REQUIRED",
                                "campaign_role": "NOW",
                                "reason": "stale fixture",
                                "required_receipt": "must not be reused",
                                "blockers": [],
                            }
                        ],
                    }
                )
            )
            (data_root / "daily_target_state.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": current_ts,
                        "status": "PURSUE_TARGET",
                        "start_balance_jpy": 100000.0,
                        "target_jpy": 10000.0,
                        "daily_risk_budget_jpy": 10000.0,
                        "per_trade_risk_budget_jpy": 1000.0,
                    }
                )
            )
            snapshot = _snapshot(root, fetched_at_utc=current_ts, quote_timestamp_utc=current_ts)

            with self.assertRaisesRegex(RuntimeError, "campaign plan stale"):
                IntentGenerator(
                    campaign_plan=campaign,
                    strategy_profile=_strategy(root),
                    output_path=root / "intents.json",
                    report_path=root / "intents.md",
                    pair_charts_path=_pair_charts(root),
                    data_root=data_root,
                ).run(snapshot_path=snapshot)

    def test_generates_and_risk_checks_priced_intent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = _campaign(root)
            strategy = _strategy(root)
            snapshot = _snapshot(root)
            output = root / "intents.json"
            report = root / "intents.md"

            cap_jpy = 500.0
            summary = IntentGenerator(
                campaign_plan=campaign,
                strategy_profile=strategy,
                output_path=output,
                report_path=report,
                pair_charts_path=_pair_charts(root),
                market_context_matrix_path=root / "market_context_matrix.json",
                data_root=root,
                max_loss_jpy=cap_jpy,
            ).run(snapshot_path=snapshot)

            self.assertEqual(summary.generated, 2)
            self.assertEqual(summary.dry_run_passed, 2)
            self.assertEqual(summary.live_ready, 0)
            payload = json.loads(output.read_text())
            order_types = {item["intent"]["order_type"] for item in payload["results"]}
            self.assertEqual(order_types, {"STOP-ENTRY", "MARKET"})
            result = next(item for item in payload["results"] if item["intent"]["order_type"] == "STOP-ENTRY")
            self.assertEqual(result["status"], "DRY_RUN_PASSED")
            self.assertEqual(result["intent"]["pair"], "EUR_USD")
            self.assertEqual(result["intent"]["market_context"]["method"], "TREND_CONTINUATION")
            self.assertLessEqual(result["risk_metrics"]["risk_jpy"], cap_jpy)
            self.assertGreater(result["risk_metrics"]["spread_pips"], 0.0)
            self.assertTrue(result["live_blockers"])
            self.assertTrue(result["live_blocker_codes"])

            market = next(item for item in payload["results"] if item["intent"]["order_type"] == "MARKET")
            self.assertEqual(market["lane_id"], "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET")
            self.assertEqual(market["intent"]["metadata"]["parent_lane_id"], "trend_trader:EUR_USD:LONG:TREND_CONTINUATION")
            self.assertEqual(market["intent"]["metadata"]["order_timing"], "NOW_MARKET")

    def test_capture_loss_asymmetry_caps_generated_new_entry_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 30,
                            "avg_win_jpy": 600.0,
                            "avg_loss_jpy": 1100.0,
                            "payoff_ratio": 0.545,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                    }
                )
            )
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(item for item in payload["results"] if item["intent"]["order_type"] == "STOP-ENTRY")
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertGreater(summary.generated, 0)
            self.assertTrue(metadata["loss_asymmetry_guard_active"])
            self.assertEqual(metadata["capture_economics_status"], "NEGATIVE_EXPECTANCY")
            self.assertEqual(metadata["loss_asymmetry_guard_loss_cap_jpy"], 600.0)
            self.assertEqual(metadata["max_loss_jpy"], 600.0)
            self.assertLessEqual(result["risk_metrics"]["risk_jpy"], 600.0)
            self.assertEqual(result["status"], "DRY_RUN_BLOCKED")
            self.assertIn(POSITIVE_ROTATION_LIVE_BLOCK_CODE, issue_codes)
            self.assertIn(POSITIVE_ROTATION_LIVE_BLOCK_CODE, result["live_blocker_codes"])

    def test_positive_rotation_live_gate_rejects_impossible_or_broad_tp_proof(
        self,
    ) -> None:
        def intent_for(metadata: dict[str, object]) -> OrderIntent:
            return OrderIntent(
                pair="EUR_USD",
                side=Side.LONG,
                order_type=OrderType.LIMIT,
                units=1000,
                entry=1.1000,
                tp=1.1010,
                sl=1.0990,
                thesis="exact vehicle TP proof",
                market_context=MarketContext(
                    regime="RANGE",
                    narrative="",
                    chart_story="",
                    method=TradeMethod.RANGE_ROTATION,
                    invalidation="",
                ),
                metadata=metadata,
            )

        proven_metadata = _exact_vehicle_rotation_metadata(trades=20)
        self.assertIsNone(
            _capture_positive_rotation_live_issue(intent_for(proven_metadata))
        )
        self.assertTrue(proven_metadata["positive_rotation_live_ready"])
        self.assertEqual(
            proven_metadata["positive_rotation_mode"],
            "TP_PROVEN_HARVEST",
        )

        collection_metadata = _exact_vehicle_rotation_metadata(trades=7)
        collection_issue = _capture_positive_rotation_live_issue(
            intent_for(collection_metadata)
        )
        self.assertIsNotNone(collection_issue)
        assert collection_issue is not None
        self.assertEqual(
            collection_issue["code"],
            POSITIVE_ROTATION_PROOF_COLLECTION_WARN_CODE,
        )
        self.assertEqual(
            collection_metadata["positive_rotation_mode"],
            "TP_PROOF_COLLECTION_HARVEST",
        )

        for trades in (7, 20):
            variants = (
                ("float-trades", {"capture_take_profit_trades": float(trades)}),
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
                            "EUR_USD|LONG|RANGE_ROTATION|TAKE_PROFIT_ORDER"
                        ),
                    },
                ),
                ("wrong-vehicle", {"capture_take_profit_vehicle": "STOP"}),
                (
                    "wrong-source",
                    {"capture_take_profit_metrics_source": "data/capture_economics.json"},
                ),
                ("wrong-scope-key", {"capture_take_profit_scope_key": "garbage"}),
            )
            for label, changes in variants:
                with self.subTest(trades=trades, variant=label):
                    metadata = _exact_vehicle_rotation_metadata(trades=trades)
                    metadata.update(changes)
                    issue = _capture_positive_rotation_live_issue(intent_for(metadata))
                    self.assertIsNotNone(issue)
                    assert issue is not None
                    self.assertEqual(issue["code"], POSITIVE_ROTATION_LIVE_BLOCK_CODE)
                    self.assertNotIn("positive_rotation_live_ready", metadata)

    def test_sub1000_sizing_uses_loss_asymmetry_effective_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 30,
                            "avg_win_jpy": 50.0,
                            "avg_loss_jpy": 1100.0,
                            "payoff_ratio": 0.045,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                    }
                )
            )
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root), max_candidates=1)

            result = json.loads(output.read_text())["results"][0]
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

        self.assertEqual(result["intent"]["units"], 125)
        self.assertEqual(metadata["max_loss_jpy"], 50.0)
        self.assertNotIn("LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT", issue_codes)
        self.assertNotIn("MIN_LOT_SIZE_UNAVAILABLE", issue_codes)
        self.assertIn(POSITIVE_ROTATION_LIVE_BLOCK_CODE, issue_codes)
        self.assertIn(POSITIVE_ROTATION_LIVE_BLOCK_CODE, result["live_blocker_codes"])
        self.assertNotIn("BAD_UNITS", result["live_blocker_codes"])

    def test_capture_loss_asymmetry_relaxes_tp_proven_harvest_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_exact_vehicle_rotation_wins(root)
            (root / "daily_target_state.json").write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_risk_budget_jpy": 100000.0,
                        "remaining_minimum_jpy": 8000.0,
                        "remaining_target_jpy": 17000.0,
                        "target_trades_per_day": 30,
                    }
                )
            )
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 210,
                            "avg_win_jpy": 600.0,
                            "avg_loss_jpy": 1100.0,
                            "payoff_ratio": 0.545,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                        "by_exit_reason": {
                            "TAKE_PROFIT_ORDER": {
                                "trades": 93,
                                "wins": 93,
                                "losses": 0,
                                "avg_win_jpy": 504.0,
                                "avg_loss_jpy": 0.0,
                                "expectancy_jpy_per_trade": 504.0,
                            },
                            "MARKET_ORDER_TRADE_CLOSE": {
                                "trades": 84,
                                "wins": 13,
                                "losses": 71,
                                "avg_win_jpy": 218.4,
                                "avg_loss_jpy": 1095.5,
                                "expectancy_jpy_per_trade": -892.1,
                            },
                        },
                        **_capture_scoped_tp_payload(),
                    }
                )
            )
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_stamp_campaign_generated_at(_range_campaign(root)),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertGreater(summary.generated, 0)
            self.assertTrue(metadata["loss_asymmetry_guard_active"])
            self.assertTrue(metadata["loss_asymmetry_guard_relaxed"])
            self.assertEqual(metadata["loss_asymmetry_guard_mode"], "TP_PROVEN_RELAXED")
            self.assertEqual(metadata["loss_asymmetry_guard_loss_cap_jpy"], 600.0)
            self.assertEqual(metadata["loss_asymmetry_guard_effective_max_loss_jpy"], 1000.0)
            self.assertEqual(metadata["max_loss_jpy"], 1000.0)
            self.assertEqual(metadata["tp_execution_mode"], "ATTACHED_TECHNICAL_TP")
            self.assertEqual(metadata["tp_target_intent"], "HARVEST")
            self.assertEqual(
                metadata["capture_take_profit_scope"],
                "PAIR_SIDE_METHOD_VEHICLE",
            )
            self.assertEqual(metadata["capture_take_profit_vehicle"], "LIMIT")
            self.assertEqual(metadata["positive_rotation_mode"], "TP_PROVEN_HARVEST")
            self.assertEqual(
                metadata["positive_rotation_confidence_method"],
                "WILSON_LOWER_BOUND_STRESS_EXPECTANCY",
            )
            self.assertGreater(metadata["positive_rotation_pessimistic_expectancy_jpy"], 0.0)
            self.assertEqual(result["status"], "LIVE_READY")
            self.assertNotIn(POSITIVE_ROTATION_LIVE_BLOCK_CODE, issue_codes)
            self.assertNotIn(POSITIVE_ROTATION_FIREPOWER_BLOCK_CODE, issue_codes)
            self.assertTrue(metadata["positive_rotation_minimum_floor_reachable"])
            self.assertLessEqual(metadata["positive_rotation_required_minimum_trades"], 30)
            self.assertLessEqual(result["risk_metrics"]["risk_jpy"], 1000.0)

    def test_global_tp_profit_is_not_enough_to_relax_unproven_pair_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 210,
                            "avg_win_jpy": 600.0,
                            "avg_loss_jpy": 1100.0,
                            "payoff_ratio": 0.545,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                        "by_exit_reason": {
                            "TAKE_PROFIT_ORDER": {
                                "trades": 93,
                                "wins": 93,
                                "losses": 0,
                                "avg_win_jpy": 504.0,
                                "avg_loss_jpy": 0.0,
                                "expectancy_jpy_per_trade": 504.0,
                            },
                            "MARKET_ORDER_TRADE_CLOSE": {
                                "trades": 84,
                                "wins": 13,
                                "losses": 71,
                                "avg_win_jpy": 218.4,
                                "avg_loss_jpy": 1095.5,
                                "expectancy_jpy_per_trade": -892.1,
                            },
                        },
                    }
                )
            )
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_stamp_campaign_generated_at(_range_campaign(root)),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertEqual(metadata["loss_asymmetry_guard_mode"], "CAP_AVG_WIN")
            self.assertEqual(metadata["capture_take_profit_scope"], "MISSING_SCOPED")
            self.assertNotEqual(metadata.get("positive_rotation_mode"), "TP_PROVEN_HARVEST")
            self.assertIn(POSITIVE_ROTATION_LIVE_BLOCK_CODE, issue_codes)

    def test_pair_side_tp_profit_does_not_relax_method_without_tp_proof(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 210,
                            "avg_win_jpy": 600.0,
                            "avg_loss_jpy": 1100.0,
                            "payoff_ratio": 0.545,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                        "by_exit_reason": {
                            "TAKE_PROFIT_ORDER": {
                                "trades": 93,
                                "wins": 93,
                                "losses": 0,
                                "avg_win_jpy": 504.0,
                                "avg_loss_jpy": 0.0,
                                "expectancy_jpy_per_trade": 504.0,
                            },
                            "MARKET_ORDER_TRADE_CLOSE": {
                                "trades": 84,
                                "wins": 13,
                                "losses": 71,
                                "avg_win_jpy": 218.4,
                                "avg_loss_jpy": 1095.5,
                                "expectancy_jpy_per_trade": -892.1,
                            },
                        },
                        "by_pair_side_exit_reason": {
                            "EUR_USD": {
                                "LONG": {
                                    "TAKE_PROFIT_ORDER": {
                                        "trades": 25,
                                        "wins": 25,
                                        "losses": 0,
                                        "avg_win_jpy": 505.4,
                                        "avg_loss_jpy": 0.0,
                                        "expectancy_jpy_per_trade": 505.4,
                                    }
                                }
                            }
                        },
                        "by_pair_side_method_exit_reason": {
                            "EUR_USD": {
                                "LONG": {
                                    "RANGE_ROTATION": {
                                        "MARKET_ORDER_TRADE_CLOSE": {
                                            "trades": 2,
                                            "wins": 0,
                                            "losses": 2,
                                            "avg_win_jpy": 0.0,
                                            "avg_loss_jpy": 1316.8,
                                            "expectancy_jpy_per_trade": -1316.8,
                                        }
                                    }
                                }
                            }
                        },
                    }
                )
            )
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_stamp_campaign_generated_at(_range_campaign(root)),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]

            self.assertEqual(metadata["loss_asymmetry_guard_mode"], "CAP_AVG_WIN")
            self.assertEqual(metadata["capture_take_profit_scope"], "MISSING_METHOD_EXIT")
            self.assertNotEqual(metadata.get("positive_rotation_mode"), "TP_PROVEN_HARVEST")

    def test_stale_capture_economics_blocks_tp_proven_rotation_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "daily_target_state.json").write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_risk_budget_jpy": 100000.0,
                        "remaining_minimum_jpy": 8000.0,
                        "remaining_target_jpy": 17000.0,
                        "target_trades_per_day": 30,
                    }
                )
            )
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 210,
                            "avg_win_jpy": 600.0,
                            "avg_loss_jpy": 1100.0,
                            "payoff_ratio": 0.545,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                        "by_exit_reason": {
                            "TAKE_PROFIT_ORDER": {
                                "trades": 93,
                                "wins": 93,
                                "losses": 0,
                                "avg_win_jpy": 504.0,
                                "avg_loss_jpy": 0.0,
                                "expectancy_jpy_per_trade": 504.0,
                            },
                            "MARKET_ORDER_TRADE_CLOSE": {
                                "trades": 84,
                                "wins": 13,
                                "losses": 71,
                                "avg_win_jpy": 218.4,
                                "avg_loss_jpy": 1095.5,
                                "expectancy_jpy_per_trade": -892.1,
                            },
                        },
                        **_capture_scoped_tp_payload(),
                        "generated_at_utc": "2026-06-17T14:05:31+00:00",
                    }
                )
            )
            with sqlite3.connect(root / "execution_ledger.db") as conn:
                conn.execute(
                    """
                    CREATE TABLE execution_events (
                        ts_utc TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        realized_pl_jpy REAL
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO execution_events(ts_utc, event_type, realized_pl_jpy)
                    VALUES ('2026-06-19T01:02:03.123456789Z', 'TRADE_CLOSED', -900.0)
                    """
                )
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_stamp_campaign_generated_at(_range_campaign(root)),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertGreater(summary.generated, 0)
            self.assertTrue(metadata["loss_asymmetry_guard_active"])
            self.assertEqual(metadata["loss_asymmetry_guard_mode"], "CAPTURE_ECONOMICS_STALE")
            self.assertTrue(metadata["capture_economics_stale"])
            self.assertEqual(
                metadata["capture_economics_latest_realized_ts_utc"],
                "2026-06-19T01:02:03.123456+00:00",
            )
            self.assertEqual(result["status"], "DRY_RUN_BLOCKED")
            self.assertIn("CAPTURE_ECONOMICS_STALE", issue_codes)
            self.assertIn("CAPTURE_ECONOMICS_STALE", result["live_blocker_codes"])
            self.assertNotEqual(metadata.get("positive_rotation_mode"), "TP_PROVEN_HARVEST")

    def test_capture_tp_proven_but_daily_firepower_short_warns_live_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_exact_vehicle_rotation_wins(root)
            (root / "daily_target_state.json").write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_risk_budget_jpy": 100000.0,
                        "remaining_minimum_jpy": 20000.0,
                        "remaining_target_jpy": 30000.0,
                        "target_trades_per_day": 30,
                    }
                )
            )
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 210,
                            "avg_win_jpy": 600.0,
                            "avg_loss_jpy": 1100.0,
                            "payoff_ratio": 0.545,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                        "by_exit_reason": {
                            "TAKE_PROFIT_ORDER": {
                                "trades": 93,
                                "wins": 93,
                                "losses": 0,
                                "avg_win_jpy": 504.0,
                                "avg_loss_jpy": 0.0,
                                "expectancy_jpy_per_trade": 504.0,
                            },
                            "MARKET_ORDER_TRADE_CLOSE": {
                                "trades": 84,
                                "wins": 13,
                                "losses": 71,
                                "avg_win_jpy": 218.4,
                                "avg_loss_jpy": 1095.5,
                                "expectancy_jpy_per_trade": -892.1,
                            },
                        },
                        **_capture_scoped_tp_payload(),
                    }
                )
            )
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_stamp_campaign_generated_at(_range_campaign(root)),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertGreater(summary.generated, 0)
            self.assertEqual(metadata["positive_rotation_mode"], "TP_PROVEN_HARVEST")
            self.assertFalse(metadata["positive_rotation_minimum_floor_reachable"])
            self.assertGreater(metadata["positive_rotation_required_minimum_trades"], 30)
            self.assertEqual(result["status"], "LIVE_READY")
            self.assertIn(POSITIVE_ROTATION_FIREPOWER_BLOCK_CODE, issue_codes)
            firepower_issue = next(
                issue
                for issue in result["risk_issues"]
                if issue["code"] == POSITIVE_ROTATION_FIREPOWER_BLOCK_CODE
            )
            self.assertEqual(firepower_issue["severity"], "WARN")
            self.assertNotIn(POSITIVE_ROTATION_FIREPOWER_BLOCK_CODE, result["live_blocker_codes"])

    def test_capture_tp_proven_uses_matching_oanda_campaign_firepower_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_exact_vehicle_rotation_wins(root)
            _write_oanda_campaign_firepower_report(root, exit_shape="tp2_sl1")
            (root / "daily_target_state.json").write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_risk_budget_jpy": 100000.0,
                        "remaining_minimum_jpy": 20000.0,
                        "remaining_target_jpy": 30000.0,
                        "target_trades_per_day": 30,
                    }
                )
            )
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 210,
                            "avg_win_jpy": 600.0,
                            "avg_loss_jpy": 1100.0,
                            "payoff_ratio": 0.545,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                        "by_exit_reason": {
                            "TAKE_PROFIT_ORDER": {
                                "trades": 93,
                                "wins": 93,
                                "losses": 0,
                                "avg_win_jpy": 504.0,
                                "avg_loss_jpy": 0.0,
                                "expectancy_jpy_per_trade": 504.0,
                            },
                            "MARKET_ORDER_TRADE_CLOSE": {
                                "trades": 84,
                                "wins": 13,
                                "losses": 71,
                                "avg_win_jpy": 218.4,
                                "avg_loss_jpy": 1095.5,
                                "expectancy_jpy_per_trade": -892.1,
                            },
                        },
                        **_capture_scoped_tp_payload(),
                    }
                )
            )
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_stamp_campaign_generated_at(_range_campaign(root)),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertGreater(summary.generated, 0)
            self.assertEqual(result["status"], "LIVE_READY")
            self.assertEqual(metadata["positive_rotation_mode"], "TP_PROVEN_HARVEST")
            self.assertFalse(metadata["positive_rotation_capture_minimum_floor_reachable"])
            self.assertTrue(metadata["positive_rotation_minimum_floor_reachable"])
            self.assertEqual(
                metadata["positive_rotation_minimum_floor_reach_basis"],
                "OANDA_CAMPAIGN_FIREPOWER_MATCHING_VEHICLE",
            )
            self.assertTrue(metadata["positive_rotation_oanda_campaign_firepower_vehicle_match"])
            self.assertEqual(
                metadata["positive_rotation_oanda_campaign_firepower_status"],
                "VERIFIED_TARGET_10_ROUTE_ESTIMATED",
            )
            self.assertFalse(metadata["positive_rotation_oanda_campaign_live_permission"])
            self.assertNotIn(POSITIVE_ROTATION_FIREPOWER_BLOCK_CODE, issue_codes)

    def test_oanda_campaign_firepower_path_falls_back_to_packaged_runtime_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packaged = root / "src" / "quant_rabbit" / "oanda_universal_rotation_precision_rules.json"
            packaged.parent.mkdir(parents=True, exist_ok=True)
            packaged.write_text(json.dumps({"campaign_firepower": {"status": "VERIFIED_TARGET_10_ROUTE_ESTIMATED"}}))

            path = _oanda_campaign_firepower_path_for_data_root(root)

            self.assertEqual(path, packaged)

    def test_oanda_campaign_firepower_path_prefers_fresh_preserved_packaged_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latest = (
                root
                / "logs"
                / "reports"
                / "forecast_improvement"
                / "oanda_universal_rotation_mining_latest.json"
            )
            packaged = root / "src" / "quant_rabbit" / "oanda_universal_rotation_precision_rules.json"
            latest.parent.mkdir(parents=True, exist_ok=True)
            packaged.parent.mkdir(parents=True, exist_ok=True)
            latest.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-23T10:09:16Z",
                        "campaign_firepower": {
                            "status": "VERIFIED_MINIMUM_5_ROUTE_ESTIMATED",
                        },
                    }
                )
            )
            packaged.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-23T10:09:16Z",
                        "source_report": "logs/reports/forecast_improvement/oanda_universal_rotation_mining_latest.json",
                        "campaign_firepower_preserved_from_existing": True,
                        "campaign_firepower": {
                            "status": "VERIFIED_TARGET_10_ROUTE_ESTIMATED",
                        },
                    }
                )
            )

            path = _oanda_campaign_firepower_path_for_data_root(root)

            self.assertEqual(path, packaged)

    def test_matching_oanda_campaign_firepower_cannot_create_one_unit_floor_lift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_oanda_campaign_firepower_report(root, exit_shape="tp2_sl1")
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 210,
                            "avg_win_jpy": 0.05,
                            "avg_loss_jpy": 1100.0,
                            "payoff_ratio": 0.045,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                        "by_exit_reason": {
                            "TAKE_PROFIT_ORDER": {
                                "trades": 93,
                                "wins": 93,
                                "losses": 0,
                                "avg_win_jpy": 504.0,
                                "avg_loss_jpy": 0.0,
                                "expectancy_jpy_per_trade": 504.0,
                            },
                            "MARKET_ORDER_TRADE_CLOSE": {
                                "trades": 84,
                                "wins": 13,
                                "losses": 71,
                                "avg_win_jpy": 218.4,
                                "avg_loss_jpy": 1095.5,
                                "expectancy_jpy_per_trade": -892.1,
                            },
                        },
                    }
                )
            )
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_stamp_campaign_generated_at(_oanda_seed_range_campaign(root)),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertEqual(result["status"], "DRY_RUN_BLOCKED")
            self.assertEqual(result["intent"]["units"], 0)
            self.assertEqual(metadata["loss_asymmetry_guard_loss_cap_jpy"], 0.05)
            self.assertEqual(metadata["loss_asymmetry_guard_mode"], "CAP_AVG_WIN")
            self.assertEqual(metadata["loss_asymmetry_guard_effective_max_loss_jpy"], 0.05)
            self.assertNotIn("positive_rotation_oanda_campaign_min_lot_sizing", metadata)
            self.assertEqual(
                metadata["positive_rotation_mode"],
                POSITIVE_ROTATION_OANDA_CAMPAIGN_FIREPOWER_MODE,
            )
            self.assertTrue(metadata["positive_rotation_oanda_campaign_firepower_vehicle_match"])
            self.assertIn("LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT", issue_codes)
            self.assertIn(OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED_CODE, issue_codes)
            self.assertIn(
                OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED_CODE,
                result["live_blocker_codes"],
            )

    def test_non_matching_oanda_campaign_firepower_keeps_avg_win_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_oanda_campaign_firepower_report(root, pair="GBP_USD", side="SHORT")
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 210,
                            "avg_win_jpy": 50.0,
                            "avg_loss_jpy": 1100.0,
                            "payoff_ratio": 0.045,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                    }
                )
            )
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_stamp_campaign_generated_at(_oanda_seed_range_campaign(root)),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertEqual(result["intent"]["units"], 398)
            self.assertEqual(metadata["loss_asymmetry_guard_mode"], "CAP_AVG_WIN")
            self.assertNotIn("positive_rotation_oanda_campaign_min_lot_sizing", metadata)
            self.assertNotIn("LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT", issue_codes)
            self.assertIn(POSITIVE_ROTATION_LIVE_BLOCK_CODE, issue_codes)

    def test_oanda_campaign_firepower_requires_current_exit_shape_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_oanda_campaign_firepower_report(root, exit_shape="tp1_sl1")
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 210,
                            "avg_win_jpy": 50.0,
                            "avg_loss_jpy": 1100.0,
                            "payoff_ratio": 0.045,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                    }
                )
            )
            output = root / "intents.json"
            campaign = _oanda_seed_range_campaign(root)
            campaign_payload = json.loads(campaign.read_text())
            campaign_payload["lanes"][0].update(
                {
                    "oanda_campaign_vehicle_key": "EUR_USD|LONG|range_reversion|tp1_sl1",
                    "oanda_campaign_vehicle_keys": ["EUR_USD|LONG|range_reversion|tp1_sl1"],
                    "oanda_campaign_exit_shape": "tp1_sl1",
                    "oanda_campaign_exit_shapes": ["tp1_sl1"],
                }
            )
            campaign.write_text(json.dumps(campaign_payload))

            IntentGenerator(
                campaign_plan=_stamp_campaign_generated_at(campaign),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertEqual(result["intent"]["units"], 398)
            self.assertEqual(metadata["oanda_campaign_exit_shape"], "tp1_sl1")
            self.assertFalse(metadata["positive_rotation_oanda_campaign_firepower_vehicle_match"])
            self.assertEqual(metadata["positive_rotation_oanda_campaign_current_reward_risk"], 2.0)
            self.assertEqual(metadata["positive_rotation_oanda_campaign_candidate_vehicle_count"], 1)
            self.assertEqual(metadata["positive_rotation_oanda_campaign_available_vehicle_count"], 1)
            self.assertEqual(metadata["positive_rotation_oanda_campaign_closest_vehicle_exit_shape"], "tp1_sl1")
            self.assertEqual(
                metadata["positive_rotation_oanda_campaign_closest_vehicle_expected_reward_risk"],
                1.0,
            )
            self.assertTrue(metadata["positive_rotation_oanda_campaign_closest_vehicle_identity_allowed"])
            self.assertIn(
                "expects 1.000R but current intent is 2.000R",
                metadata["positive_rotation_oanda_campaign_vehicle_mismatch_reason"],
            )
            self.assertNotIn("positive_rotation_oanda_campaign_min_lot_sizing", metadata)
            self.assertNotIn("LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT", issue_codes)
            self.assertIn(POSITIVE_ROTATION_LIVE_BLOCK_CODE, issue_codes)

    def test_oanda_campaign_firepower_requires_lane_vehicle_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_oanda_campaign_firepower_report(root, exit_shape="tp1_sl1")
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 210,
                            "avg_win_jpy": 50.0,
                            "avg_loss_jpy": 1100.0,
                            "payoff_ratio": 0.045,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                    }
                )
            )
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_stamp_campaign_generated_at(_oanda_seed_range_campaign(root)),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertEqual(metadata["oanda_campaign_exit_shape"], "tp2_sl1")
            self.assertFalse(metadata["positive_rotation_oanda_campaign_firepower_vehicle_match"])
            self.assertEqual(metadata["positive_rotation_oanda_campaign_candidate_vehicle_count"], 0)
            self.assertEqual(metadata["positive_rotation_oanda_campaign_available_vehicle_count"], 1)
            self.assertFalse(metadata["positive_rotation_oanda_campaign_closest_vehicle_identity_allowed"])
            self.assertIn(
                "not allowed by the lane vehicle identity",
                metadata["positive_rotation_oanda_campaign_vehicle_mismatch_reason"],
            )
            self.assertEqual(result["intent"]["units"], 398)
            self.assertNotIn("positive_rotation_oanda_campaign_min_lot_sizing", metadata)
            self.assertNotIn("LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT", issue_codes)
            self.assertIn(POSITIVE_ROTATION_LIVE_BLOCK_CODE, issue_codes)

    def test_oanda_campaign_vehicle_reprice_reports_deeper_limit_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_oanda_campaign_firepower_report(
                root,
                pair="USD_JPY",
                side="LONG",
                exit_shape="tp1_sl1",
            )
            now = datetime.now(timezone.utc)
            quote = Quote(pair="USD_JPY", bid=161.27, ask=161.35, timestamp_utc=now)
            metadata = _oanda_campaign_vehicle_shape_reprice_metadata(
                lane={
                    "oanda_campaign_firepower_seed": True,
                    "oanda_campaign_vehicle_key": "USD_JPY|LONG|range_reversion|tp1_sl1",
                    "oanda_campaign_vehicle_keys": ["USD_JPY|LONG|range_reversion|tp1_sl1"],
                    "oanda_campaign_exit_shape": "tp1_sl1",
                    "oanda_campaign_exit_shapes": ["tp1_sl1"],
                },
                pair="USD_JPY",
                side=Side.LONG,
                method=TradeMethod.RANGE_ROTATION,
                order_type=OrderType.LIMIT,
                quote=quote,
                entry=161.19,
                tp=161.358,
                sl=160.566,
                data_root=root,
            )

            self.assertTrue(metadata["oanda_campaign_vehicle_reprice_checked"])
            self.assertEqual(
                metadata["oanda_campaign_vehicle_reprice_status"],
                "ENTRY_REPRICE_POSSIBLE",
            )
            self.assertEqual(
                metadata["oanda_campaign_vehicle_reprice_vehicle_key"],
                "USD_JPY|LONG|range_reversion|tp1_sl1",
            )
            self.assertEqual(metadata["oanda_campaign_vehicle_reprice_exit_shape"], "tp1_sl1")
            self.assertEqual(metadata["oanda_campaign_vehicle_reprice_expected_reward_risk"], 1.0)
            self.assertAlmostEqual(
                metadata["oanda_campaign_vehicle_reprice_current_reward_risk"],
                0.269231,
                places=6,
            )
            self.assertEqual(metadata["oanda_campaign_vehicle_reprice_required_entry"], 160.962)
            self.assertAlmostEqual(
                metadata["oanda_campaign_vehicle_reprice_entry_improvement_pips"],
                22.8,
                places=1,
            )
            self.assertEqual(metadata["oanda_campaign_vehicle_reprice_required_reward_risk"], 1.0)

    def test_oanda_campaign_vehicle_reprice_does_not_degrade_richer_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_oanda_campaign_firepower_report(root, exit_shape="tp1_sl1")
            now = datetime.now(timezone.utc)
            quote = Quote(pair="EUR_USD", bid=1.10000, ask=1.10008, timestamp_utc=now)
            metadata = _oanda_campaign_vehicle_shape_reprice_metadata(
                lane={
                    "oanda_campaign_firepower_seed": True,
                    "oanda_campaign_vehicle_key": "EUR_USD|LONG|range_reversion|tp1_sl1",
                    "oanda_campaign_vehicle_keys": ["EUR_USD|LONG|range_reversion|tp1_sl1"],
                    "oanda_campaign_exit_shape": "tp1_sl1",
                    "oanda_campaign_exit_shapes": ["tp1_sl1"],
                },
                pair="EUR_USD",
                side=Side.LONG,
                method=TradeMethod.RANGE_ROTATION,
                order_type=OrderType.LIMIT,
                quote=quote,
                entry=1.10000,
                tp=1.10200,
                sl=1.09900,
                data_root=root,
            )

            self.assertTrue(metadata["oanda_campaign_vehicle_reprice_checked"])
            self.assertEqual(
                metadata["oanda_campaign_vehicle_reprice_status"],
                "ENTRY_REPRICE_NOT_NEEDED_OR_DEGRADES",
            )
            self.assertEqual(metadata["oanda_campaign_vehicle_reprice_expected_reward_risk"], 1.0)
            self.assertEqual(metadata["oanda_campaign_vehicle_reprice_current_reward_risk"], 2.0)
            self.assertNotIn("oanda_campaign_vehicle_reprice_required_entry", metadata)

    def test_oanda_campaign_vehicle_reprice_restores_expected_rr_inside_match_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_oanda_campaign_firepower_report(root, exit_shape="tp1_sl1")
            now = datetime.now(timezone.utc)
            quote = Quote(pair="EUR_USD", bid=1.10000, ask=1.10008, timestamp_utc=now)
            metadata = _oanda_campaign_vehicle_shape_reprice_metadata(
                lane={
                    "oanda_campaign_firepower_seed": True,
                    "oanda_campaign_vehicle_key": "EUR_USD|LONG|range_reversion|tp1_sl1",
                    "oanda_campaign_vehicle_keys": ["EUR_USD|LONG|range_reversion|tp1_sl1"],
                    "oanda_campaign_exit_shape": "tp1_sl1",
                    "oanda_campaign_exit_shapes": ["tp1_sl1"],
                },
                pair="EUR_USD",
                side=Side.LONG,
                method=TradeMethod.RANGE_ROTATION,
                order_type=OrderType.LIMIT,
                quote=quote,
                entry=1.10000,
                tp=1.10095,
                sl=1.09900,
                data_root=root,
            )

            self.assertTrue(metadata["oanda_campaign_vehicle_reprice_checked"])
            self.assertEqual(
                metadata["oanda_campaign_vehicle_reprice_status"],
                "ENTRY_REPRICE_POSSIBLE",
            )
            self.assertEqual(metadata["oanda_campaign_vehicle_reprice_expected_reward_risk"], 1.0)
            self.assertEqual(metadata["oanda_campaign_vehicle_reprice_current_reward_risk"], 0.95)
            self.assertEqual(metadata["oanda_campaign_vehicle_reprice_required_entry"], 1.09997)
            self.assertGreaterEqual(
                metadata["oanda_campaign_vehicle_reprice_required_reward_risk"],
                1.0,
            )

    def test_oanda_campaign_vehicle_reprice_applies_before_sizing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_oanda_campaign_firepower_report(root, exit_shape="tp3_sl1")
            campaign = _oanda_seed_range_campaign(root)
            campaign_payload = json.loads(campaign.read_text())
            campaign_payload["lanes"][0].update(
                {
                    "oanda_campaign_vehicle_key": "EUR_USD|LONG|range_reversion|tp3_sl1",
                    "oanda_campaign_vehicle_keys": ["EUR_USD|LONG|range_reversion|tp3_sl1"],
                    "oanda_campaign_exit_shape": "tp3_sl1",
                    "oanda_campaign_exit_shapes": ["tp3_sl1"],
                }
            )
            campaign.write_text(json.dumps(campaign_payload))
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_stamp_campaign_generated_at(campaign),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            intent = result["intent"]
            metadata = intent["metadata"]

            self.assertEqual(intent["entry"], 1.17084)
            self.assertEqual(metadata["oanda_campaign_vehicle_reprice_applied"], True)
            self.assertEqual(metadata["oanda_campaign_vehicle_reprice_original_entry"], 1.17104)
            self.assertEqual(metadata["oanda_campaign_vehicle_reprice_applied_entry"], 1.17084)
            self.assertEqual(
                metadata["oanda_campaign_vehicle_reprice_status"],
                "MATCHED_CURRENT_GEOMETRY",
            )
            self.assertEqual(metadata["oanda_campaign_vehicle_reprice_current_reward_risk"], 3.0)
            self.assertEqual(metadata["virtual_take_profit_reward_risk"], 3.0)
            self.assertGreater(intent["units"], 0)

    def test_oanda_campaign_vehicle_reprice_rejects_range_box_break(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_oanda_campaign_firepower_report(root, exit_shape="tp3_sl1")
            campaign = _oanda_seed_range_campaign(root)
            campaign_payload = json.loads(campaign.read_text())
            campaign_payload["lanes"][0].update(
                {
                    "oanda_campaign_vehicle_key": "EUR_USD|LONG|range_reversion|tp3_sl1",
                    "oanda_campaign_vehicle_keys": ["EUR_USD|LONG|range_reversion|tp3_sl1"],
                    "oanda_campaign_exit_shape": "tp3_sl1",
                    "oanda_campaign_exit_shapes": ["tp3_sl1"],
                }
            )
            campaign.write_text(json.dumps(campaign_payload))
            pair_charts = root / "pair_charts_narrow.json"
            pair_charts.write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "charts": [
                            {
                                "pair": "EUR_USD",
                                "views": [
                                    {
                                        "granularity": "M5",
                                        "indicators": {
                                            "atr_pips": 8.0,
                                            "bb_lower": 1.1710,
                                            "bb_upper": 1.1720,
                                            "bb_middle": 1.1715,
                                            "donchian_low": 1.1707,
                                            "donchian_high": 1.1720,
                                            "vwap": 1.1715,
                                            "avwap_anchor": 1.1714,
                                            "avwap_lower_1sd": 1.1712,
                                            "avwap_upper_1sd": 1.1719,
                                            "linreg_channel_lower": 1.1709,
                                            "linreg_channel_upper": 1.1720,
                                            "swing_low": 1.1705,
                                            "swing_high": 1.1720,
                                        },
                                    }
                                ],
                            }
                        ]
                    }
                )
            )
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_stamp_campaign_generated_at(campaign),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=pair_charts,
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            intent = result["intent"]
            metadata = intent["metadata"]

            self.assertEqual(intent["entry"], 1.17306)
            self.assertFalse(metadata["oanda_campaign_vehicle_reprice_applied"])
            self.assertEqual(
                metadata["oanda_campaign_vehicle_reprice_apply_rejected_status"],
                "RANGE_BOX_CONTRACT_BROKEN",
            )
            self.assertNotIn("oanda_campaign_vehicle_reprice_applied_entry", metadata)

    def test_capture_tp_proven_keeps_firepower_warn_for_non_matching_oanda_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_exact_vehicle_rotation_wins(root)
            _write_oanda_campaign_firepower_report(
                root,
                pair="GBP_USD",
                side="SHORT",
                shape="pullback_continuation",
            )
            (root / "daily_target_state.json").write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_risk_budget_jpy": 100000.0,
                        "remaining_minimum_jpy": 20000.0,
                        "remaining_target_jpy": 30000.0,
                        "target_trades_per_day": 30,
                    }
                )
            )
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 210,
                            "avg_win_jpy": 600.0,
                            "avg_loss_jpy": 1100.0,
                            "payoff_ratio": 0.545,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                        "by_exit_reason": {
                            "TAKE_PROFIT_ORDER": {
                                "trades": 93,
                                "wins": 93,
                                "losses": 0,
                                "avg_win_jpy": 504.0,
                                "avg_loss_jpy": 0.0,
                                "expectancy_jpy_per_trade": 504.0,
                            },
                            "MARKET_ORDER_TRADE_CLOSE": {
                                "trades": 84,
                                "wins": 13,
                                "losses": 71,
                                "avg_win_jpy": 218.4,
                                "avg_loss_jpy": 1095.5,
                                "expectancy_jpy_per_trade": -892.1,
                            },
                        },
                        **_capture_scoped_tp_payload(),
                    }
                )
            )
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_stamp_campaign_generated_at(_range_campaign(root)),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertFalse(metadata["positive_rotation_oanda_campaign_firepower_vehicle_match"])
            self.assertFalse(metadata["positive_rotation_minimum_floor_reachable"])
            self.assertIn(POSITIVE_ROTATION_FIREPOWER_BLOCK_CODE, issue_codes)
            self.assertNotIn(POSITIVE_ROTATION_FIREPOWER_BLOCK_CODE, result["live_blocker_codes"])

    def test_thin_exact_tp_collection_warns_without_claiming_tp_proven_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_exact_vehicle_rotation_wins(
                root,
                count=10,
                realized_pl_jpy=600.0,
            )
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 210,
                            "avg_win_jpy": 600.0,
                            "avg_loss_jpy": 1100.0,
                            "payoff_ratio": 0.545,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                        "by_exit_reason": {
                            "TAKE_PROFIT_ORDER": {
                                "trades": 93,
                                "wins": 93,
                                "losses": 0,
                                "avg_win_jpy": 504.0,
                                "avg_loss_jpy": 0.0,
                                "expectancy_jpy_per_trade": 504.0,
                            },
                            "MARKET_ORDER_TRADE_CLOSE": {
                                "trades": 84,
                                "wins": 13,
                                "losses": 71,
                                "avg_win_jpy": 218.4,
                                "avg_loss_jpy": 1100.0,
                                "expectancy_jpy_per_trade": -892.1,
                            },
                        },
                        **_capture_scoped_tp_payload(
                            trades=10,
                            wins=10,
                            losses=0,
                            avg_win_jpy=600.0,
                            avg_loss_jpy=0.0,
                            expectancy_jpy_per_trade=600.0,
                        ),
                    }
                )
            )
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_range_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_by_code = {issue["code"]: issue for issue in result["risk_issues"]}

            self.assertGreaterEqual(summary.live_ready, 1)
            self.assertEqual(result["status"], "LIVE_READY")
            self.assertEqual(metadata["loss_asymmetry_guard_mode"], "CAP_AVG_WIN")
            self.assertFalse(metadata["loss_asymmetry_guard_relaxed"])
            self.assertEqual(metadata["loss_asymmetry_guard_effective_max_loss_jpy"], 600.0)
            self.assertEqual(metadata["max_loss_jpy"], 600.0)
            self.assertEqual(metadata["positive_rotation_mode"], "TP_PROOF_COLLECTION_HARVEST")
            self.assertTrue(metadata["positive_rotation_proof_collection_ready"])
            self.assertEqual(metadata["positive_rotation_proof_collection_gap_trades"], 10)
            self.assertNotIn("positive_rotation_live_ready", metadata)
            self.assertGreater(metadata["positive_rotation_pessimistic_expectancy_jpy"], 0.0)
            self.assertIn(POSITIVE_ROTATION_PROOF_COLLECTION_WARN_CODE, issue_by_code)
            self.assertEqual(
                issue_by_code[POSITIVE_ROTATION_PROOF_COLLECTION_WARN_CODE]["severity"],
                "WARN",
            )
            self.assertNotIn(POSITIVE_ROTATION_LIVE_BLOCK_CODE, issue_by_code)
            self.assertNotIn(POSITIVE_ROTATION_PROOF_COLLECTION_WARN_CODE, result["live_blocker_codes"])

    def test_thin_tp_collection_uses_sub1000_units_without_floor_lift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_exact_vehicle_rotation_wins(
                root,
                count=10,
                realized_pl_jpy=600.0,
            )
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 210,
                            "avg_win_jpy": 50.0,
                            "avg_loss_jpy": 1100.0,
                            "payoff_ratio": 0.045,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                        "by_exit_reason": {
                            "TAKE_PROFIT_ORDER": {
                                "trades": 93,
                                "wins": 93,
                                "losses": 0,
                                "avg_win_jpy": 504.0,
                                "avg_loss_jpy": 0.0,
                                "expectancy_jpy_per_trade": 504.0,
                            },
                            "MARKET_ORDER_TRADE_CLOSE": {
                                "trades": 84,
                                "wins": 13,
                                "losses": 71,
                                "avg_win_jpy": 218.4,
                                "avg_loss_jpy": 1100.0,
                                "expectancy_jpy_per_trade": -892.1,
                            },
                        },
                        **_capture_scoped_tp_payload(
                            trades=10,
                            wins=10,
                            losses=0,
                            avg_win_jpy=600.0,
                            avg_loss_jpy=0.0,
                            expectancy_jpy_per_trade=600.0,
                        ),
                    }
                )
            )
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_range_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_by_code = {issue["code"]: issue for issue in result["risk_issues"]}

            self.assertEqual(result["intent"]["units"], 398)
            self.assertEqual(metadata["loss_asymmetry_guard_loss_cap_jpy"], 50.0)
            self.assertEqual(metadata["loss_asymmetry_guard_mode"], "CAP_AVG_WIN")
            self.assertFalse(metadata["loss_asymmetry_guard_relaxed"])
            self.assertEqual(metadata["loss_asymmetry_guard_effective_max_loss_jpy"], 50.0)
            self.assertNotIn("positive_rotation_proof_collection_min_lot_sizing", metadata)
            self.assertEqual(metadata["positive_rotation_mode"], "TP_PROOF_COLLECTION_HARVEST")
            self.assertTrue(metadata["positive_rotation_proof_collection_ready"])
            self.assertNotIn("LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT", issue_by_code)
            self.assertNotIn("LOSS_ASYMMETRY_GUARD_EXCEEDED", issue_by_code)
            self.assertNotIn(POSITIVE_ROTATION_LIVE_BLOCK_CODE, issue_by_code)

    def test_thin_exact_tp_collection_survives_stale_chart_bias_as_separate_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_exact_vehicle_rotation_wins(
                root,
                method="BREAKOUT_FAILURE",
                count=10,
                realized_pl_jpy=600.0,
            )
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 210,
                            "avg_win_jpy": 600.0,
                            "avg_loss_jpy": 1100.0,
                            "payoff_ratio": 0.545,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                        "by_exit_reason": {
                            "TAKE_PROFIT_ORDER": {
                                "trades": 93,
                                "wins": 93,
                                "losses": 0,
                                "avg_win_jpy": 504.0,
                                "avg_loss_jpy": 0.0,
                                "expectancy_jpy_per_trade": 504.0,
                            },
                            "MARKET_ORDER_TRADE_CLOSE": {
                                "trades": 84,
                                "wins": 13,
                                "losses": 71,
                                "avg_win_jpy": 218.4,
                                "avg_loss_jpy": 1100.0,
                                "expectancy_jpy_per_trade": -892.1,
                            },
                        },
                        **_capture_scoped_tp_payload(
                            method="BREAKOUT_FAILURE",
                            trades=10,
                            wins=10,
                            losses=0,
                            avg_win_jpy=600.0,
                            avg_loss_jpy=0.0,
                            expectancy_jpy_per_trade=600.0,
                        ),
                    }
                )
            )
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_breakout_failure_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts_with_direction(
                    root,
                    long_score=0.30,
                    short_score=0.70,
                    dominant_regime="BREAKOUT_FAILURE",
                    m5_regime="BREAKOUT_FAILURE",
                ),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertEqual(metadata["chart_direction_bias"], "SHORT")
            self.assertEqual(metadata["positive_rotation_mode"], "TP_PROOF_COLLECTION_HARVEST")
            self.assertTrue(metadata["positive_rotation_proof_collection_ready"])
            self.assertTrue(
                metadata["positive_rotation_proof_collection_direction_bias_conflict_observed"]
            )
            self.assertIn(POSITIVE_ROTATION_PROOF_COLLECTION_WARN_CODE, issue_codes)
            self.assertNotIn(POSITIVE_ROTATION_LIVE_BLOCK_CODE, issue_codes)
            self.assertIn("CHART_DIRECTION_CONFLICT", issue_codes)
            self.assertIn(MARKET_CLOSE_LEAK_FAMILY_BLOCK_CODE, result["live_blocker_codes"])
            self.assertNotEqual(result["status"], "LIVE_READY")

    def test_thin_tp_collection_allows_self_improvement_p0_repair_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_exact_vehicle_rotation_wins(
                root,
                count=10,
                realized_pl_jpy=600.0,
            )
            (root / "self_improvement_audit.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "findings": [
                            {
                                "priority": "P0",
                                "layer": "profitability",
                                "code": "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
                                "message": "market-close leakage is still negative",
                                "evidence": {
                                    "current_streak": 65,
                                    "system_defect_evidence": {
                                        "profit_factor": 0.788,
                                        "expectancy_jpy": -54.04,
                                        "gateway_close_bleed_observation": {
                                            "gateway_net_jpy": -1200.0,
                                        },
                                    },
                                },
                            }
                        ],
                    }
                )
            )
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 210,
                            "avg_win_jpy": 600.0,
                            "avg_loss_jpy": 1100.0,
                            "payoff_ratio": 0.545,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                        "by_exit_reason": {
                            "TAKE_PROFIT_ORDER": {
                                "trades": 93,
                                "wins": 93,
                                "losses": 0,
                                "avg_win_jpy": 504.0,
                                "avg_loss_jpy": 0.0,
                                "expectancy_jpy_per_trade": 504.0,
                            },
                            "MARKET_ORDER_TRADE_CLOSE": {
                                "trades": 84,
                                "wins": 13,
                                "losses": 71,
                                "avg_win_jpy": 218.4,
                                "avg_loss_jpy": 1100.0,
                                "expectancy_jpy_per_trade": -892.1,
                            },
                        },
                        **_capture_scoped_tp_payload(
                            trades=10,
                            wins=10,
                            losses=0,
                            avg_win_jpy=600.0,
                            avg_loss_jpy=0.0,
                            expectancy_jpy_per_trade=600.0,
                        ),
                    }
                )
            )
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_range_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertGreaterEqual(summary.live_ready, 1)
            self.assertEqual(result["status"], "LIVE_READY")
            self.assertEqual(metadata["positive_rotation_mode"], "TP_PROOF_COLLECTION_HARVEST")
            self.assertTrue(metadata["positive_rotation_proof_collection_ready"])
            self.assertTrue(metadata["self_improvement_p0_repair_live_ready"])
            self.assertEqual(metadata["self_improvement_p0_repair_mode"], "TP_HARVEST_REPAIR")
            self.assertIn(POSITIVE_ROTATION_PROOF_COLLECTION_WARN_CODE, issue_codes)
            self.assertIn("SELF_IMPROVEMENT_P0_PROFITABILITY_REPAIR_MODE", issue_codes)
            self.assertNotIn("SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE", issue_codes)
            self.assertEqual(result["live_blockers"], [])

    def test_tp_proven_exact_vehicle_rotation_survives_stale_chart_bias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_exact_vehicle_rotation_wins(
                root,
                method="BREAKOUT_FAILURE",
            )
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 213,
                            "avg_win_jpy": 415.5,
                            "avg_loss_jpy": 1061.9,
                            "payoff_ratio": 0.391,
                            "breakeven_payoff_at_win_rate": 0.651,
                        },
                        "by_exit_reason": {
                            "TAKE_PROFIT_ORDER": {
                                "trades": 93,
                                "wins": 93,
                                "losses": 0,
                                "avg_win_jpy": 504.0,
                                "avg_loss_jpy": 0.0,
                                "expectancy_jpy_per_trade": 504.0,
                            },
                            "MARKET_ORDER_TRADE_CLOSE": {
                                "trades": 92,
                                "wins": 19,
                                "losses": 73,
                                "avg_win_jpy": 220.0,
                                "avg_loss_jpy": 1095.5,
                                "expectancy_jpy_per_trade": -809.8,
                            },
                        },
                        **_capture_scoped_tp_payload(method="BREAKOUT_FAILURE"),
                    }
                )
            )
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_breakout_failure_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts_with_direction(
                    root,
                    long_score=0.30,
                    short_score=0.70,
                    dominant_regime="BREAKOUT_FAILURE",
                    m5_regime="BREAKOUT_FAILURE",
                ),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertEqual(metadata["chart_direction_bias"], "SHORT")
            self.assertEqual(
                metadata["capture_take_profit_scope"],
                "PAIR_SIDE_METHOD_VEHICLE",
            )
            self.assertEqual(
                metadata["capture_take_profit_scope_key"],
                "EUR_USD|LONG|BREAKOUT_FAILURE|LIMIT|TAKE_PROFIT_ORDER",
            )
            self.assertEqual(metadata["positive_rotation_mode"], "TP_PROVEN_HARVEST")
            self.assertTrue(metadata["positive_rotation_direction_bias_conflict_overridden"])
            self.assertGreater(metadata["positive_rotation_pessimistic_expectancy_jpy"], 0.0)
            self.assertNotIn(POSITIVE_ROTATION_LIVE_BLOCK_CODE, issue_codes)
            self.assertIn("CHART_DIRECTION_CONFLICT", issue_codes)
            self.assertNotEqual(result["status"], "LIVE_READY")

    def test_capture_tp_positive_but_stress_negative_blocks_live_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_exact_vehicle_rotation_wins(
                root,
                count=20,
                realized_pl_jpy=20.0,
            )
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 210,
                            "avg_win_jpy": 600.0,
                            "avg_loss_jpy": 1100.0,
                            "payoff_ratio": 0.545,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                        "by_exit_reason": {
                            "TAKE_PROFIT_ORDER": {
                                "trades": 20,
                                "wins": 20,
                                "losses": 0,
                                "avg_win_jpy": 20.0,
                                "avg_loss_jpy": 0.0,
                                "expectancy_jpy_per_trade": 20.0,
                            },
                            "MARKET_ORDER_TRADE_CLOSE": {
                                "trades": 84,
                                "wins": 13,
                                "losses": 71,
                                "avg_win_jpy": 218.4,
                                "avg_loss_jpy": 1095.5,
                                "expectancy_jpy_per_trade": -892.1,
                            },
                        },
                        **_capture_scoped_tp_payload(
                            trades=20,
                            wins=20,
                            losses=0,
                            avg_win_jpy=20.0,
                            avg_loss_jpy=0.0,
                            expectancy_jpy_per_trade=20.0,
                        ),
                    }
                )
            )
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_range_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertGreater(summary.generated, 0)
            self.assertTrue(metadata["loss_asymmetry_guard_relaxed"])
            self.assertEqual(metadata["loss_asymmetry_guard_mode"], "TP_PROVEN_RELAXED")
            self.assertLess(metadata["positive_rotation_pessimistic_expectancy_jpy"], 0.0)
            self.assertEqual(result["status"], "DRY_RUN_BLOCKED")
            self.assertIn(POSITIVE_ROTATION_LIVE_BLOCK_CODE, issue_codes)
            self.assertIn(POSITIVE_ROTATION_LIVE_BLOCK_CODE, result["live_blocker_codes"])

    def test_sub1000_sizing_uses_loss_streak_adjusted_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            loss_streak = SameDayLossStreak(
                pair="EUR_USD",
                consecutive_losses=1,
                net_loss_jpy=-100.0,
                last_loss_ts_utc="2026-06-15T00:00:00Z",
            )

            with patch(
                "quant_rabbit.strategy.intent_generator.compute_same_day_loss_streaks",
                return_value={"EUR_USD": loss_streak},
            ):
                IntentGenerator(
                    campaign_plan=_campaign(root),
                    strategy_profile=_strategy(root, status="CANDIDATE"),
                    output_path=root / "intents.json",
                    report_path=root / "intents.md",
                    pair_charts_path=_pair_charts(root),
                    data_root=data_root,
                    max_loss_jpy=500.0,
                ).run(snapshot_path=_snapshot(root), max_candidates=1)

            payload = json.loads((root / "intents.json").read_text())
            result = payload["results"][0]

        issue_codes = {item["code"] for item in result["risk_issues"]}
        self.assertEqual(result["intent"]["units"], 628)
        self.assertNotIn("LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT", issue_codes)
        self.assertNotIn("MIN_LOT_SIZE_UNAVAILABLE", issue_codes)
        self.assertIn("SAME_DAY_LOSS_STREAK", issue_codes)
        self.assertNotIn("LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT", result["live_blocker_codes"])
        self.assertNotIn("BAD_UNITS", result["live_blocker_codes"])
        self.assertFalse(any("loss budget can only fund" in blocker for blocker in result["live_blockers"]))
        self.assertFalse(any("units must be positive" in blocker for blocker in result["live_blockers"]))

    def test_forecast_seed_telemetry_skips_stale_quote(self) -> None:
        from quant_rabbit.models import Quote
        from quant_rabbit.strategy.intent_generator import _record_forecast_seed_telemetry

        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        validation_time = datetime(2026, 6, 5, 3, 0, tzinfo=timezone.utc)
        stale_quote_time = validation_time - timedelta(seconds=120)
        forecast = SimpleNamespace(
            direction="UP",
            confidence=0.72,
            current_price=1.1,
            target_price=1.105,
            invalidation_price=1.098,
            horizon_min=60,
            projection_signals=(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            recorded = _record_forecast_seed_telemetry(
                forecast,
                pair="EUR_USD",
                quote=Quote("EUR_USD", 1.0999, 1.1001, stale_quote_time),
                pair_chart={},
                data_root=data_root,
                cycle_id="stale-cycle",
                validation_time_utc=validation_time,
            )

            self.assertFalse(recorded)
            self.assertFalse((data_root / "forecast_history.jsonl").exists())
            self.assertFalse((data_root / "projection_ledger.jsonl").exists())

    def test_forecast_seed_lanes_skip_context_when_required_telemetry_is_stale(self) -> None:
        prior = os.environ.get("QR_REQUIRE_TELEMETRY_FOR_LIVE")
        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        try:
            validation_time = datetime(2026, 6, 5, 3, 0, tzinfo=timezone.utc)
            stale_quote_time = validation_time - timedelta(seconds=120)
            forecast = SimpleNamespace(
                pair="EUR_USD",
                direction="UP",
                confidence=0.91,
                current_price=1.1,
                target_price=1.105,
                invalidation_price=1.098,
                horizon_min=60,
                projection_signals=(),
                market_support=_high_precision_market_support("UP"),
                rationale_summary="UP forecast",
                drivers_for=("test",),
                drivers_against=(),
            )
            source = {
                "desk": "trend_trader",
                "pair": "EUR_USD",
                "direction": "LONG",
                "method": TradeMethod.TREND_CONTINUATION.value,
            }
            snapshot = BrokerSnapshot(
                fetched_at_utc=validation_time,
                quotes={"EUR_USD": Quote("EUR_USD", 1.0999, 1.1001, stale_quote_time)},
            )

            with tempfile.TemporaryDirectory() as tmp, patch(
                "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                return_value=forecast,
            ):
                data_root = Path(tmp)
                lanes = _append_forecast_seed_lanes(
                    [source],
                    {"EUR_USD": {}},
                    snapshot,
                    data_root=data_root,
                    forecast_cycle_id="cycle-stale",
                )

            self.assertEqual(lanes, [source])
            self.assertNotIn("forecast_cycle_id", lanes[0])
            self.assertNotIn("forecast_direction", lanes[0])
        finally:
            if prior is None:
                os.environ.pop("QR_REQUIRE_TELEMETRY_FOR_LIVE", None)
            else:
                os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = prior

    def test_forecast_seed_telemetry_rejects_inexact_or_far_future_clock(self) -> None:
        from quant_rabbit.models import Quote
        from quant_rabbit.strategy.intent_generator import (
            _record_forecast_seed_telemetry,
            _snapshot_from_json,
        )

        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        validation_time = datetime(2026, 6, 5, 3, 0, tzinfo=timezone.utc)
        missing_clock_quote = _snapshot_from_json(
            {
                "fetched_at_utc": validation_time.isoformat(),
                "positions": [],
                "orders": [],
                "quotes": {"EUR_USD": {"bid": 1.0999, "ask": 1.1001}},
            }
        ).quotes["EUR_USD"]
        canonical_z_quote = _snapshot_from_json(
            {
                "fetched_at_utc": "2026-06-05T03:00:00Z",
                "positions": [],
                "orders": [],
                "quotes": {
                    "EUR_USD": {
                        "bid": 1.0999,
                        "ask": 1.1001,
                        "timestamp_utc": "2026-06-05T03:00:00Z",
                    }
                },
            }
        ).quotes["EUR_USD"]
        self.assertEqual(canonical_z_quote.timestamp_utc, validation_time)
        quotes = {
            "missing": missing_clock_quote,
            "naive": Quote(
                "EUR_USD",
                1.0999,
                1.1001,
                datetime(2026, 6, 5, 3, 0),
            ),
            "far-future": Quote(
                "EUR_USD",
                1.0999,
                1.1001,
                validation_time + timedelta(minutes=1),
            ),
        }
        forecast = SimpleNamespace(
            direction="UP",
            confidence=0.72,
            current_price=1.1,
            target_price=1.105,
            invalidation_price=1.098,
            horizon_min=60,
            projection_signals=(),
        )

        for label, quote in quotes.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                data_root = Path(tmp)
                recorded = _record_forecast_seed_telemetry(
                    forecast,
                    pair="EUR_USD",
                    quote=quote,
                    pair_chart={},
                    data_root=data_root,
                    cycle_id=f"{label}-clock-cycle",
                    validation_time_utc=validation_time,
                )

                self.assertFalse(recorded)
                self.assertFalse((data_root / "forecast_history.jsonl").exists())
                self.assertFalse((data_root / "projection_ledger.jsonl").exists())
                self.assertFalse(
                    (
                        data_root
                        / "regime_family_contradiction_shadow_ledger.jsonl"
                    ).exists()
                )

    def test_market_context_matrix_metadata_is_advisory_on_intent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                market_context_matrix_path=_market_context_matrix(root),
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(item for item in payload["results"] if item["intent"]["order_type"] == "MARKET")
            metadata = result["intent"]["metadata"]

            self.assertGreater(summary.generated, 0)
            self.assertEqual(metadata["market_context_matrix_ref"], "matrix:EUR_USD:LONG")
            self.assertEqual(metadata["matrix_support_count"], 4)
            self.assertEqual(metadata["matrix_reject_count"], 1)
            self.assertEqual(metadata["matrix_support_layers"], ["context_asset_chart"])
            self.assertIn("GOLD_CONTEXT_TECHNICAL_DIRECTION", metadata["matrix_support_context"][0])
            self.assertEqual(metadata["matrix_support_refs"], ["context_asset:XAU_USD"])
            self.assertIn("context_asset:XAU_USD", metadata["matrix_context_refs"])
            self.assertIn("cot:EUR", metadata["matrix_context_refs"])
            self.assertIn("matrix matrix:EUR_USD:LONG", result["intent"]["market_context"]["chart_story"])
            self.assertIn("XAU_USD pressure maps to EUR_USD LONG", result["intent"]["market_context"]["chart_story"])

    def test_news_artifact_refs_are_persisted_on_intent_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            logs_root = root / "logs"
            logs_root.mkdir()
            (logs_root / "news_digest.md").write_text(
                "\n".join(
                    [
                        "# FX News Digest",
                        "## Pair-Specific Notes",
                        "- EUR/USD: USD data risk keeps the pair headline-sensitive today.",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (data_root / "news_items.json").write_text(
                json.dumps({"generated_at_utc": datetime.now(timezone.utc).isoformat(), "items": []}) + "\n",
                encoding="utf-8",
            )
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                market_context_matrix_path=_market_context_matrix(root),
                data_root=data_root,
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(item for item in payload["results"] if item["intent"]["order_type"] == "MARKET")
            metadata = result["intent"]["metadata"]

            self.assertGreater(summary.generated, 0)
            self.assertEqual(metadata["news_refs"], ["news:digest", "news:items"])
            self.assertEqual(metadata["news_digest_ref"], "news:digest")
            self.assertEqual(metadata["news_items_ref"], "news:items")
            self.assertEqual(metadata["news_signal_names"], ["market_story_news_artifact"])
            self.assertIn("EUR/USD", metadata["news_pair_context"][0])

    def test_matrix_supported_profile_edge_seeds_pending_repair_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            strategy = root / "strategy.json"
            strategy.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "pair": "USD_JPY",
                                "direction": "LONG",
                                "status": "BLOCK_UNTIL_NEW_EVIDENCE",
                                "required_fix": "risk-resized receipt required",
                                "target_reward_risk": 2.4,
                                "positive_best_jpy": 1220.0,
                                "positive_tail_jpy": 15.0,
                                "positive_evidence_n": 1722,
                            }
                        ]
                    }
                )
            )
            charts = root / "pair_charts.json"
            charts.write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "charts": [
                            {
                                "pair": "USD_JPY",
                                "dominant_regime": "TREND_UP",
                                "long_score": 0.78,
                                "short_score": 0.12,
                                "confluence": {
                                    "score_balance": "LONG_LEAN",
                                    "score_gap": 0.66,
                                    "tf_agreement_score": 0.8,
                                    "range_24h_sigma_multiple": 1.0,
                                },
                                "views": [
                                    {
                                        "granularity": "M5",
                                        "regime": "TREND_UP",
                                        "long_bias": 0.75,
                                        "short_bias": 0.15,
                                        "indicators": {
                                            "atr_pips": 8.0,
                                            "bb_lower": 156.20,
                                            "bb_upper": 157.10,
                                            "donchian_low": 156.10,
                                            "donchian_high": 157.20,
                                            "swing_low": 156.05,
                                            "swing_high": 157.25,
                                        },
                                    }
                                ],
                            }
                        ]
                    }
                )
            )
            matrix = root / "matrix.json"
            matrix.write_text(
                json.dumps(
                    {
                        "pairs": {
                            "USD_JPY": {
                                "LONG": {
                                    "evidence_ref": "matrix:USD_JPY:LONG",
                                    "support_count": 4,
                                    "reject_count": 0,
                                    "supports": [
                                        {
                                            "code": "CHART_CONFLUENCE_LONG_LEAN",
                                            "layer": "chart",
                                            "message": "USD_JPY confluence score_balance=LONG_LEAN",
                                            "evidence_refs": ["chart:USD_JPY:structure"],
                                        },
                                        {
                                            "code": "BASE_STRENGTH_EXCEEDS_QUOTE",
                                            "layer": "strength",
                                            "message": "USD strength exceeds JPY",
                                            "evidence_refs": ["strength:USD_JPY"],
                                        },
                                        {
                                            "code": "DXY_24H_DIRECTION",
                                            "layer": "cross_asset",
                                            "message": "DXY maps to USD_JPY LONG",
                                            "evidence_refs": ["cross:dxy"],
                                        },
                                        {
                                            "code": "FLOW_SPREAD_EXECUTABLE",
                                            "layer": "flow",
                                            "message": "USD_JPY spread stress=NORMAL",
                                            "evidence_refs": ["flow:USD_JPY"],
                                        },
                                    ],
                                }
                            }
                        },
                        "issues": [],
                    }
                )
            )
            output = root / "intents.json"

            prior_sl_free = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                IntentGenerator(
                    campaign_plan=_campaign(root),
                    strategy_profile=strategy,
                    output_path=output,
                    report_path=root / "intents.md",
                    pair_charts_path=charts,
                    market_context_matrix_path=matrix,
                    max_loss_jpy=500.0,
                ).run(snapshot_path=_snapshot(root))
            finally:
                if prior_sl_free is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl_free

            payload = json.loads(output.read_text())
            usd_rows = [
                item
                for item in payload["results"]
                if (item.get("intent") or {}).get("pair") == "USD_JPY"
                and (item.get("intent") or {}).get("side") == "LONG"
            ]
            mirrored_seed_rows = [
                item
                for item in payload["results"]
                if (item.get("intent") or {}).get("pair") == "USD_JPY"
                and (item.get("intent") or {}).get("side") == "SHORT"
                and ((item.get("intent") or {}).get("metadata") or {}).get("matrix_repair_seed")
            ]

            self.assertGreaterEqual(len(usd_rows), 2)
            self.assertEqual(mirrored_seed_rows, [])
            self.assertFalse(any(row["intent"]["order_type"] == "MARKET" for row in usd_rows))
            metadata = usd_rows[0]["intent"]["metadata"]
            self.assertTrue(metadata["matrix_repair_seed"])
            self.assertEqual(metadata["matrix_repair_profile_status"], "BLOCK_UNTIL_NEW_EVIDENCE")
            self.assertEqual(metadata["market_context_matrix_ref"], "matrix:USD_JPY:LONG")
            self.assertEqual(metadata["base_target_reward_risk"], 2.4)
            self.assertGreaterEqual(metadata["target_reward_risk"], 2.4)
            self.assertTrue(any("BLOCK_UNTIL_NEW_EVIDENCE" in blocker for row in usd_rows for blocker in row["live_blockers"]))

    def test_oanda_firepower_seed_is_not_mirrored_to_opposite_direction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = root / "campaign.json"
            campaign.write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "lanes": [
                            {
                                "desk": "range_trader",
                                "pair": "EUR_USD",
                                "direction": "SHORT",
                                "method": "RANGE_ROTATION",
                                "adoption": "ORDER_INTENT_REQUIRED",
                                "campaign_role": "OANDA_FIREPOWER_ROUTE",
                                "reason": "OANDA high precision SHORT range vehicle",
                                "required_receipt": "Build current non-market order intent.",
                                "target_reward_risk": 1.0,
                                "oanda_campaign_firepower_seed": True,
                                "oanda_campaign_vehicle_key": "EUR_USD|SHORT|range_reversion|tp1_sl1",
                                "oanda_campaign_firepower_status": "VERIFIED_TARGET_10_ROUTE_ESTIMATED",
                                "oanda_campaign_exit_shape": "tp1_sl1",
                                "oanda_campaign_estimated_return_pct_per_active_day": 1.4,
                                "oanda_campaign_live_permission": False,
                            }
                        ],
                    }
                )
            )
            output = root / "intents.json"
            prior_sl_free = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                IntentGenerator(
                    campaign_plan=campaign,
                    strategy_profile=_strategy(root, status="CANDIDATE", direction="SHORT"),
                    output_path=output,
                    report_path=root / "intents.md",
                    pair_charts_path=_pair_charts(root),
                    max_loss_jpy=500.0,
                ).run(snapshot_path=_snapshot(root))
            finally:
                if prior_sl_free is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl_free

            payload = json.loads(output.read_text())
            oanda_rows = [
                item
                for item in payload["results"]
                if ((item.get("intent") or {}).get("metadata") or {}).get("oanda_campaign_firepower_seed")
            ]

            self.assertGreaterEqual(len(oanda_rows), 1)
            self.assertTrue(all(row["intent"]["side"] == "SHORT" for row in oanda_rows))
            self.assertFalse(
                any(
                    row["intent"]["side"] == "LONG"
                    and row["intent"]["metadata"].get("oanda_campaign_vehicle_key")
                    for row in payload["results"]
                )
            )

    def test_oanda_firepower_seed_does_not_spawn_current_range_derivative(self) -> None:
        source = {
            "desk": "trend_trader",
            "pair": "EUR_USD",
            "direction": "SHORT",
            "method": "TREND_CONTINUATION",
            "adoption": "ORDER_INTENT_REQUIRED",
            "campaign_role": "OANDA_FIREPOWER_ROUTE",
            "reason": "OANDA high precision SHORT pullback vehicle",
            "required_receipt": "Build current non-market order intent.",
            "target_reward_risk": 1.25,
            "oanda_campaign_firepower_seed": True,
            "oanda_campaign_vehicle_key": "EUR_USD|SHORT|pullback_continuation|tp1.25_sl1",
            "oanda_campaign_firepower_status": "VERIFIED_TARGET_10_ROUTE_ESTIMATED",
            "oanda_campaign_exit_shape": "tp1.25_sl1",
            "oanda_campaign_estimated_return_pct_per_active_day": 1.8,
            "oanda_campaign_live_permission": False,
        }
        charts = {
            "EUR_USD": {
                "dominant_regime": "RANGE",
                "M5__regime_reading": {"state": "RANGE"},
                "M5__regime": "RANGE",
                "M5": {
                    "bb_lower": 1.1710,
                    "bb_upper": 1.1760,
                    "donchian_low": 1.1707,
                    "donchian_high": 1.1764,
                    "adx_14": 15.0,
                    "choppiness_14": 70.0,
                },
            }
        }

        lanes = _append_current_range_phase_lanes([source], charts)

        self.assertEqual(len(lanes), 1)
        self.assertTrue(lanes[0]["oanda_campaign_firepower_seed"])
        self.assertEqual(lanes[0]["method"], "TREND_CONTINUATION")

    def test_forecast_seed_lane_strips_oanda_firepower_identity_from_source(self) -> None:
        source = {
            "desk": "range_trader",
            "pair": "EUR_USD",
            "direction": "SHORT",
            "method": "RANGE_ROTATION",
            "adoption": "ORDER_INTENT_REQUIRED",
            "campaign_role": "OANDA_FIREPOWER_ROUTE",
            "reason": "OANDA high precision SHORT range vehicle",
            "required_receipt": "Build current non-market order intent.",
            "target_reward_risk": 1.0,
            "oanda_campaign_firepower_seed": True,
            "oanda_campaign_vehicle_key": "EUR_USD|SHORT|range_reversion|tp1_sl1",
            "oanda_campaign_firepower_status": "VERIFIED_TARGET_10_ROUTE_ESTIMATED",
            "oanda_campaign_exit_shape": "tp1_sl1",
            "oanda_campaign_estimated_return_pct_per_active_day": 1.4,
            "oanda_campaign_live_permission": False,
        }
        forecast = SimpleNamespace(
            direction="UP",
            confidence=0.92,
            raw_confidence=0.92,
            calibration_multiplier=1.0,
            current_price=1.1730,
            target_price=1.1810,
            invalidation_price=1.1680,
            range_low_price=None,
            range_high_price=None,
            range_width_pips=None,
            horizon_min=60,
            rationale_summary="fresh forecast supports long continuation",
            drivers_for=("projection support",),
            drivers_against=(),
            component_scores={},
            market_support=None,
        )

        lane = _forecast_seed_lane(
            source,
            pair="EUR_USD",
            side="LONG",
            method=TradeMethod.TREND_CONTINUATION.value,
            forecast=forecast,
            cycle_id="cycle-1",
        )

        self.assertTrue(lane["forecast_seed"])
        self.assertEqual(lane["direction"], "LONG")
        self.assertNotIn("oanda_campaign_firepower_seed", lane)
        self.assertNotIn("oanda_campaign_vehicle_key", lane)

    def test_matrix_supported_watch_only_edge_gets_diagnostic_dry_run_lanes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            strategy = root / "strategy.json"
            strategy.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "pair": "AUD_JPY",
                                "direction": "LONG",
                                "status": "WATCH_ONLY",
                                "required_fix": (
                                    "missed seats were directionally correct, but realized seat net is negative; "
                                    "repair discovery filters before mining this edge"
                                ),
                                "target_reward_risk": 2.2,
                                "positive_best_jpy": 1180.0,
                                "positive_tail_jpy": 240.0,
                                "positive_evidence_n": 12,
                                "pretrade_net_jpy": 1408.0,
                            }
                        ]
                    }
                )
            )
            charts = root / "pair_charts.json"
            charts.write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "charts": [
                            {
                                "pair": "AUD_JPY",
                                "dominant_regime": "TREND_UP",
                                "long_score": 0.61,
                                "short_score": 0.34,
                                "confluence": {
                                    "score_balance": "LONG_LEAN",
                                    "score_gap": 0.27,
                                    "tf_agreement_score": 0.8,
                                    "range_24h_sigma_multiple": 1.0,
                                    "price_percentile_24h": 0.58,
                                    "price_percentile_7d": 0.62,
                                },
                                "views": [
                                    {
                                        "granularity": "M5",
                                        "regime": "TREND_UP",
                                        "long_bias": 0.66,
                                        "short_bias": 0.24,
                                        "indicators": {
                                            "atr_pips": 8.0,
                                            "bb_lower": 113.10,
                                            "bb_upper": 113.90,
                                            "bb_middle": 113.50,
                                            "donchian_low": 113.05,
                                            "donchian_high": 113.95,
                                            "swing_low": 113.00,
                                            "swing_high": 114.00,
                                        },
                                    },
                                    {
                                        "granularity": "H1",
                                        "regime": "TREND_UP",
                                        "indicators": {"atr_pips": 24.0, "adx_14": 31.0},
                                    },
                                    {
                                        "granularity": "H4",
                                        "regime": "TREND_UP",
                                        "indicators": {"atr_pips": 42.0, "adx_14": 31.0},
                                    },
                                ],
                            }
                        ]
                    }
                )
            )
            matrix = root / "matrix.json"
            matrix.write_text(
                json.dumps(
                    {
                        "pairs": {
                            "AUD_JPY": {
                                "LONG": {
                                    "evidence_ref": "matrix:AUD_JPY:LONG",
                                    "support_count": 4,
                                    "reject_count": 0,
                                    "supports": [
                                        {
                                            "code": "CHART_CONFLUENCE_LONG_LEAN",
                                            "layer": "chart",
                                            "message": "AUD_JPY confluence score_balance=LONG_LEAN",
                                            "evidence_refs": ["chart:AUD_JPY:structure"],
                                        },
                                        {
                                            "code": "BASE_STRENGTH_EXCEEDS_QUOTE",
                                            "layer": "strength",
                                            "message": "AUD strength exceeds JPY",
                                            "evidence_refs": ["strength:AUD_JPY"],
                                        },
                                        {
                                            "code": "RISK_ASSET_JPY_CROSS_DIRECTION",
                                            "layer": "cross_asset",
                                            "message": "SPX risk context maps to AUD_JPY LONG",
                                            "evidence_refs": ["cross:spx"],
                                        },
                                        {
                                            "code": "FLOW_SPREAD_EXECUTABLE",
                                            "layer": "flow",
                                            "message": "AUD_JPY spread stress=NORMAL",
                                            "evidence_refs": ["flow:AUD_JPY"],
                                        },
                                    ],
                                }
                            }
                        },
                        "issues": [],
                    }
                )
            )
            snapshot = root / "snapshot.json"
            now = datetime.now(timezone.utc).isoformat()
            snapshot.write_text(
                json.dumps(
                    {
                        "fetched_at_utc": now,
                        "positions": [],
                        "orders": [],
                        "quotes": {
                            "EUR_USD": {"bid": 1.17322, "ask": 1.17330, "timestamp_utc": now},
                            "AUD_JPY": {"bid": 113.36, "ask": 113.376, "timestamp_utc": now},
                        },
                        "account": {
                            "nav_jpy": 200000.0,
                            "balance_jpy": 200000.0,
                            "margin_used_jpy": 0.0,
                            "margin_available_jpy": 200000.0,
                            "fetched_at_utc": now,
                        },
                    }
                )
            )
            output = root / "intents.json"

            prior_sl_free = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                with patch(
                    "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                    return_value=None,
                ):
                    IntentGenerator(
                        campaign_plan=_campaign(root),
                        strategy_profile=strategy,
                        output_path=output,
                        report_path=root / "intents.md",
                        pair_charts_path=charts,
                        market_context_matrix_path=matrix,
                        max_loss_jpy=500.0,
                    ).run(snapshot_path=snapshot)
            finally:
                if prior_sl_free is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl_free

            payload = json.loads(output.read_text())
            aud_rows = [
                item
                for item in payload["results"]
                if (item.get("intent") or {}).get("pair") == "AUD_JPY"
                and (item.get("intent") or {}).get("side") == "LONG"
            ]

            self.assertGreaterEqual(len(aud_rows), 2)
            self.assertFalse(any(row["status"] == "LIVE_READY" for row in aud_rows))
            self.assertFalse(any(row["intent"]["order_type"] == "MARKET" for row in aud_rows))
            methods = {row["intent"]["market_context"]["method"] for row in aud_rows}
            self.assertIn("BREAKOUT_FAILURE", methods)
            self.assertIn("TREND_CONTINUATION", methods)
            metadata = aud_rows[0]["intent"]["metadata"]
            self.assertTrue(metadata["matrix_repair_seed"])
            self.assertTrue(metadata["matrix_watch_only_seed"])
            self.assertEqual(metadata["matrix_repair_profile_status"], "WATCH_ONLY")
            self.assertTrue(any("WATCH_ONLY" in blocker for row in aud_rows for blocker in row["live_blockers"]))
            self.assertTrue(
                any(
                    issue["code"] == "STRATEGY_NOT_ELIGIBLE" and issue["severity"] == "BLOCK"
                    for row in aud_rows
                    for issue in row["live_strategy_issues"]
                )
            )

    def test_contested_matrix_supported_edge_gets_blocked_diagnostic_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            strategy = root / "strategy.json"
            strategy.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "pair": "USD_JPY",
                                "direction": "LONG",
                                "status": "RISK_REPAIR_CANDIDATE",
                                "required_fix": "risk-resized receipt required",
                                "target_reward_risk": 2.1,
                                "positive_best_jpy": 910.0,
                                "positive_evidence_n": 11,
                            }
                        ]
                    }
                )
            )
            charts = root / "pair_charts.json"
            charts.write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "charts": [
                            {
                                "pair": "USD_JPY",
                                "dominant_regime": "TREND_UP",
                                "long_score": 0.71,
                                "short_score": 0.18,
                                "confluence": {
                                    "score_balance": "LONG_LEAN",
                                    "score_gap": 0.53,
                                    "tf_agreement_score": 0.8,
                                    "range_24h_sigma_multiple": 1.0,
                                },
                                "views": [
                                    {
                                        "granularity": "M5",
                                        "regime": "TREND_UP",
                                        "long_bias": 0.69,
                                        "short_bias": 0.16,
                                        "indicators": {
                                            "atr_pips": 8.0,
                                            "bb_lower": 156.20,
                                            "bb_upper": 157.10,
                                            "donchian_low": 156.10,
                                            "donchian_high": 157.20,
                                            "swing_low": 156.05,
                                            "swing_high": 157.25,
                                        },
                                    }
                                ],
                            }
                        ]
                    }
                )
            )
            matrix = root / "matrix.json"
            matrix.write_text(
                json.dumps(
                    {
                        "pairs": {
                            "USD_JPY": {
                                "LONG": {
                                    "evidence_ref": "matrix:USD_JPY:LONG",
                                    "support_count": 4,
                                    "reject_count": 1,
                                    "supports": [
                                        {
                                            "code": "CHART_CONFLUENCE_LONG_LEAN",
                                            "layer": "chart",
                                            "message": "USD_JPY confluence score_balance=LONG_LEAN",
                                        },
                                        {
                                            "code": "BASE_STRENGTH_EXCEEDS_QUOTE",
                                            "layer": "strength",
                                            "message": "USD strength exceeds JPY",
                                        },
                                        {
                                            "code": "DXY_24H_DIRECTION",
                                            "layer": "cross_asset",
                                            "message": "DXY maps to USD_JPY LONG",
                                        },
                                        {
                                            "code": "FLOW_SPREAD_EXECUTABLE",
                                            "layer": "flow",
                                            "message": "USD_JPY spread stress=NORMAL",
                                        },
                                    ],
                                    "rejects": [
                                        {
                                            "code": "CALENDAR_RISK_WINDOW",
                                            "layer": "calendar",
                                            "message": "USD_JPY high-impact event window still active",
                                        }
                                    ],
                                }
                            }
                        },
                        "issues": [],
                    }
                )
            )
            output = root / "intents.json"

            prior_sl_free = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                with patch(
                    "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                    return_value=None,
                ):
                    IntentGenerator(
                        campaign_plan=_campaign(root),
                        strategy_profile=strategy,
                        output_path=output,
                        report_path=root / "intents.md",
                        pair_charts_path=charts,
                        market_context_matrix_path=matrix,
                        max_loss_jpy=500.0,
                    ).run(snapshot_path=_snapshot(root))
            finally:
                if prior_sl_free is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl_free

            payload = json.loads(output.read_text())
            contested_rows = [
                item
                for item in payload["results"]
                if ((item.get("intent") or {}).get("metadata") or {}).get("matrix_repair_reject_blocked")
            ]

            self.assertGreaterEqual(len(contested_rows), 1)
            self.assertFalse(any(row["status"] == "LIVE_READY" for row in contested_rows))
            self.assertTrue(
                any(
                    issue["code"] == "MATRIX_REPAIR_REJECT_CONTEXT"
                    and issue["severity"] == "BLOCK"
                    for row in contested_rows
                    for issue in row["risk_issues"]
                )
            )
            metadata = contested_rows[0]["intent"]["metadata"]
            self.assertTrue(metadata["matrix_repair_seed"])
            self.assertEqual(metadata["matrix_repair_profile_status"], "RISK_REPAIR_CANDIDATE")
            self.assertEqual(metadata["matrix_repair_reject_reasons"], ["USD_JPY high-impact event window still active"])
            self.assertTrue(
                any(
                    "current reject context" in blocker
                    for row in contested_rows
                    for blocker in row["live_blockers"]
                )
            )

    def test_post_harvest_profit_take_seeds_pullback_limit_reentry_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            close_ts = datetime.now(timezone.utc) - timedelta(minutes=12)
            _write_post_harvest_close(
                data_root,
                ts_utc=close_ts.isoformat().replace("+00:00", "Z"),
                pair="EUR_USD",
                closed_units=-5000,
            )
            forecast = SimpleNamespace(
                pair="EUR_USD",
                direction="UP",
                confidence=0.84,
                current_price=1.17326,
                target_price=1.1762,
                invalidation_price=1.1718,
                horizon_min=60,
                rationale_summary="UP forecast after post-harvest pullback",
                drivers_for=("fresh pullback retest",),
                drivers_against=("wait for rail fill",),
            )
            output = root / "intents.json"

            with patch(
                "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                return_value=forecast,
            ):
                IntentGenerator(
                    campaign_plan=_campaign(root),
                    strategy_profile=_strategy(root, status="CANDIDATE"),
                    pair_charts_path=_pair_charts_with_direction(
                        root,
                        long_score=0.58,
                        short_score=0.42,
                        dominant_regime="RANGE",
                        m5_regime="RANGE",
                        m5_long_bias=0.54,
                        m5_short_bias=0.30,
                        adx=12.0,
                        choppiness=68.0,
                        close=1.1733,
                    ),
                    output_path=output,
                    report_path=root / "intents.md",
                    data_root=data_root,
                    max_loss_jpy=500.0,
                ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            reentry_rows = [
                item
                for item in payload["results"]
                if ((item.get("intent") or {}).get("metadata") or {}).get("post_harvest_reentry_seed")
            ]

            self.assertEqual(len(reentry_rows), 1)
            row = reentry_rows[0]
            self.assertEqual(row["lane_id"], "post_harvest_trader:EUR_USD:LONG:RANGE_ROTATION")
            self.assertEqual(row["intent"]["order_type"], "LIMIT")
            self.assertEqual(row["intent"]["metadata"]["forecast_direction"], "UP")
            self.assertEqual(row["intent"]["metadata"]["post_harvest_trade_id"], "harvest-1")
            self.assertIn("Post-harvest re-entry lane", row["intent"]["metadata"]["required_receipt"])
            post_harvest_lane_ids = [
                item["lane_id"]
                for item in payload["results"]
                if ((item.get("intent") or {}).get("metadata") or {}).get("post_harvest_reentry_seed")
            ]
            self.assertFalse(any(lane_id.endswith(":MARKET") for lane_id in post_harvest_lane_ids))

    def test_attached_take_profit_order_seeds_pullback_limit_reentry_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            close_ts = datetime.now(timezone.utc) - timedelta(minutes=10)
            _write_broker_take_profit_close(
                data_root,
                ts_utc=close_ts.isoformat().replace("+00:00", "Z"),
                pair="EUR_USD",
                side="LONG",
                lane_id="range_trader:EUR_USD:LONG:RANGE_ROTATION",
                realized_pl_jpy=620.5,
            )
            forecast = SimpleNamespace(
                pair="EUR_USD",
                direction="UP",
                confidence=0.84,
                current_price=1.17326,
                target_price=1.1762,
                invalidation_price=1.1718,
                horizon_min=60,
                rationale_summary="UP forecast after broker TP pullback",
                drivers_for=("fresh range retest",),
                drivers_against=("wait for rail fill",),
            )
            output = root / "intents.json"

            with patch(
                "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                return_value=forecast,
            ):
                IntentGenerator(
                    campaign_plan=_campaign(root),
                    strategy_profile=_strategy(root, status="CANDIDATE"),
                    pair_charts_path=_pair_charts_with_direction(
                        root,
                        long_score=0.58,
                        short_score=0.42,
                        dominant_regime="RANGE",
                        m5_regime="RANGE",
                        m5_long_bias=0.54,
                        m5_short_bias=0.30,
                        adx=12.0,
                        choppiness=68.0,
                        close=1.1733,
                    ),
                    output_path=output,
                    report_path=root / "intents.md",
                    data_root=data_root,
                    max_loss_jpy=500.0,
                ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            reentry_rows = [
                item
                for item in payload["results"]
                if ((item.get("intent") or {}).get("metadata") or {}).get("post_harvest_reentry_seed")
            ]

            self.assertEqual(len(reentry_rows), 1)
            row = reentry_rows[0]
            metadata = row["intent"]["metadata"]
            self.assertEqual(row["lane_id"], "post_harvest_trader:EUR_USD:LONG:RANGE_ROTATION")
            self.assertEqual(row["intent"]["order_type"], "LIMIT")
            self.assertEqual(metadata["post_harvest_source"], "TAKE_PROFIT_ORDER")
            self.assertEqual(metadata["post_harvest_trade_id"], "tp-1")
            self.assertIn("Post-harvest re-entry lane", metadata["required_receipt"])
            self.assertFalse(row["lane_id"].endswith(":MARKET"))

    def test_post_harvest_reentry_seed_skips_when_same_pair_position_still_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            _write_post_harvest_close(
                data_root,
                ts_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                pair="EUR_USD",
                closed_units=-5000,
            )

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                pair_charts_path=_pair_charts_with_direction(
                    root,
                    long_score=0.58,
                    short_score=0.42,
                    dominant_regime="RANGE",
                    m5_regime="RANGE",
                    adx=12.0,
                    choppiness=68.0,
                ),
                output_path=root / "intents.json",
                report_path=root / "intents.md",
                data_root=data_root,
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot_with_position(root))

            payload = json.loads((root / "intents.json").read_text())
            self.assertFalse(
                any(
                    ((item.get("intent") or {}).get("metadata") or {}).get("post_harvest_reentry_seed")
                    for item in payload["results"]
                )
            )

    def test_live_entry_requires_fresh_forecast_when_live_default_active(self) -> None:
        prior = os.environ.get("QR_REQUIRE_FORECAST_FOR_LIVE")
        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"

                summary = IntentGenerator(
                    campaign_plan=_campaign(root),
                    strategy_profile=_strategy(root, status="CANDIDATE"),
                    output_path=output,
                    report_path=root / "intents.md",
                    pair_charts_path=_pair_charts(root),
                    max_loss_jpy=500.0,
                ).run(snapshot_path=_snapshot(root))

                payload = json.loads(output.read_text())
        finally:
            if prior is None:
                os.environ.pop("QR_REQUIRE_FORECAST_FOR_LIVE", None)
            else:
                os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = prior

        self.assertEqual(summary.live_ready, 0)
        self.assertGreater(summary.dry_run_passed, 0)
        issue_codes = {issue["code"] for item in payload["results"] for issue in item["risk_issues"]}
        blockers = [blocker for item in payload["results"] for blocker in item["live_blockers"]]
        self.assertIn("FORECAST_CONTEXT_REQUIRED_FOR_LIVE", issue_codes)
        self.assertTrue(any("no fresh executable pair forecast" in blocker for blocker in blockers))

    def test_unclear_forecast_context_is_attached_without_live_permission(self) -> None:
        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"
            forecast = SimpleNamespace(
                pair="EUR_USD",
                direction="UNCLEAR",
                confidence=0.11,
                raw_confidence=0.11,
                calibration_multiplier=1.0,
                current_price=1.17326,
                target_price=None,
                invalidation_price=None,
                horizon_min=0,
                rationale_summary="contested: DOWN=93.3 vs UP=83.3",
                drivers_for=("pair_chart SHORT_LEAN",),
                drivers_against=("mean reversion bounce risk",),
                component_scores={"UP": 83.3, "DOWN": 93.3, "RANGE": 0.0, "EITHER": 0.0},
            )

            with (
                patch("quant_rabbit.strategy.intent_generator.ROOT", root),
                patch(
                    "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                    return_value=forecast,
                ),
            ):
                IntentGenerator(
                    campaign_plan=_range_campaign(root, direction="LONG"),
                    strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                    pair_charts_path=_pair_charts_with_direction(
                        root,
                        long_score=0.46,
                        short_score=0.54,
                        dominant_regime="RANGE",
                        m5_regime="RANGE",
                        m5_long_bias=0.58,
                        m5_short_bias=0.22,
                        regime_state="RANGE",
                    ),
                    output_path=output,
                    report_path=root / "intents.md",
                    max_loss_jpy=500.0,
                ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())

        results = [item for item in payload["results"] if item["intent"] and item["intent"]["pair"] == "EUR_USD"]
        issue_codes = {issue["code"] for item in results for issue in item["risk_issues"]}
        blockers = [blocker for item in results for blocker in item["live_blockers"]]
        metadata = results[0]["intent"]["metadata"]

        self.assertTrue(results)
        self.assertEqual(metadata["forecast_direction"], "UNCLEAR")
        self.assertEqual(metadata["forecast_confidence"], 0.11)
        self.assertIsNotNone(metadata["forecast_cycle_id"])
        self.assertIn("FORECAST_NOT_EXECUTABLE_FOR_LIVE", issue_codes)
        self.assertIn("TELEMETRY_FORECAST_NOT_EXECUTABLE_FOR_LIVE", issue_codes)
        self.assertNotIn("TELEMETRY_FORECAST_CONTEXT_REQUIRED_FOR_LIVE", issue_codes)
        self.assertTrue(any("current pair forecast is UNCLEAR" in blocker for blocker in blockers))
        self.assertFalse(any("no executable forecast metadata" in blocker for blocker in blockers))

    def test_live_entry_requires_forecast_telemetry_when_live_default_active(self) -> None:
        prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"
                forecast = SimpleNamespace(
                    direction="DOWN",
                    confidence=0.91,
                    current_price=1.17326,
                    target_price=1.1712,
                    invalidation_price=1.1742,
                    horizon_min=60,
                    rationale_summary="DOWN forecast from current tape",
                    drivers_for=("pair_chart SHORT_LEAN",),
                    drivers_against=("range bounce risk",),
                    market_support=_high_precision_market_support("DOWN"),
                )

                with (
                    patch("quant_rabbit.strategy.intent_generator.ROOT", root),
                    patch(
                        "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                        return_value=forecast,
                    ),
                    patch(
                        "quant_rabbit.strategy.intent_generator._record_forecast_seed_telemetry",
                        return_value=False,
                    ),
                ):
                    summary = IntentGenerator(
                        campaign_plan=_campaign(root, direction="LONG"),
                        strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                        pair_charts_path=_pair_charts_with_direction(
                            root,
                            long_score=0.46,
                            short_score=0.54,
                            dominant_regime="RANGE",
                            m5_regime="RANGE",
                        ),
                        output_path=output,
                        report_path=root / "intents.md",
                        max_loss_jpy=500.0,
                    ).run(snapshot_path=_snapshot(root))

                payload = json.loads(output.read_text())
        finally:
            if prior_sl is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl

        short_results = [item for item in payload["results"] if item["intent"]["side"] == "SHORT"]
        issue_codes = {issue["code"] for item in short_results for issue in item["risk_issues"]}
        self.assertEqual(summary.live_ready, 0)
        self.assertTrue(short_results)
        self.assertTrue(any(item["status"] == "DRY_RUN_PASSED" for item in short_results))
        self.assertTrue(all(item["status"] != "LIVE_READY" for item in short_results))
        self.assertIn("TELEMETRY_FORECAST_HISTORY_REQUIRED_FOR_LIVE", issue_codes)

    def test_generate_intents_records_pre_entry_forecast_telemetry_before_live_validation(self) -> None:
        prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"
                open_ts = "2026-06-08T12:00:00+00:00"
                snapshot = _snapshot(root, fetched_at_utc=open_ts, quote_timestamp_utc=open_ts)
                projection_signal = SimpleNamespace(
                    name="bb_squeeze_expansion_imminent",
                    timeframe="H1",
                    direction="EITHER",
                    lead_time_min=300,
                    confidence=0.74,
                    bonus_magnitude=8.0,
                    rationale="H1 squeeze expansion timing",
                )
                forecast = SimpleNamespace(
                    pair="EUR_USD",
                    direction="DOWN",
                    confidence=0.91,
                    current_price=1.17326,
                    target_price=1.1712,
                    invalidation_price=1.1742,
                    horizon_min=60,
                    rationale_summary="DOWN forecast from current tape",
                    drivers_for=("pair_chart SHORT_LEAN",),
                    drivers_against=("range bounce risk",),
                    projection_signals=(projection_signal,),
                    market_support=_high_precision_market_support("DOWN"),
                )

                with (
                    patch("quant_rabbit.strategy.intent_generator.ROOT", root),
                    patch(
                        "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                        return_value=forecast,
                    ),
                ):
                    IntentGenerator(
                        campaign_plan=_campaign(root, direction="LONG"),
                        strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                        pair_charts_path=_pair_charts_with_direction(
                            root,
                            long_score=0.46,
                            short_score=0.54,
                            dominant_regime="RANGE",
                            m5_regime="RANGE",
                        ),
                        output_path=output,
                        report_path=root / "intents.md",
                        max_loss_jpy=500.0,
                    ).run(snapshot_path=snapshot)

                payload = json.loads(output.read_text())
                rows = (root / "data" / "forecast_history.jsonl").read_text().splitlines()
                projection_rows = [
                    json.loads(line)
                    for line in (root / "data" / "projection_ledger.jsonl").read_text().splitlines()
                    if line.strip()
                ]
        finally:
            if prior_sl is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl

        self.assertTrue(rows)
        latest = json.loads(rows[-1])
        self.assertEqual(latest["pair"], "EUR_USD")
        self.assertTrue(str(latest["cycle_id"]).startswith("pre-entry-forecast-refresh:"))
        signal_names = {row["signal_name"] for row in projection_rows}
        self.assertIn("directional_forecast", signal_names)
        self.assertIn("bb_squeeze_expansion_imminent", signal_names)

        short_results = [item for item in payload["results"] if item["intent"]["side"] == "SHORT"]
        issue_codes = {
            issue["code"]
            for item in short_results
            for issue in item["risk_issues"]
        }
        self.assertNotIn("TELEMETRY_FORECAST_HISTORY_REQUIRED_FOR_LIVE", issue_codes)
        self.assertNotIn("TELEMETRY_DIRECTIONAL_PROJECTION_REQUIRED_FOR_LIVE", issue_codes)

    def test_generate_intents_replaces_stale_same_cycle_forecast_telemetry(self) -> None:
        from quant_rabbit.strategy.intent_generator import _pre_entry_forecast_cycle_id, _snapshot_from_json

        prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"
                open_ts = "2026-06-08T12:00:00+00:00"
                snapshot_path = _snapshot(root, fetched_at_utc=open_ts, quote_timestamp_utc=open_ts)
                pair_charts_path = _pair_charts_with_direction(
                    root,
                    long_score=0.46,
                    short_score=0.54,
                    dominant_regime="RANGE",
                    m5_regime="RANGE",
                )
                snapshot = _snapshot_from_json(json.loads(snapshot_path.read_text()))
                cycle_id = _pre_entry_forecast_cycle_id(snapshot, pair_charts_path=pair_charts_path)
                data_root = root / "data"
                data_root.mkdir()
                (data_root / "forecast_history.jsonl").write_text(
                    json.dumps(
                        {
                            "timestamp_utc": "2026-06-08T12:00:00Z",
                            "cycle_id": cycle_id,
                            "pair": "EUR_USD",
                            "direction": "UP",
                            "confidence": 0.34,
                            "current_price": 1.17326,
                            "target_price": 1.1762,
                            "invalidation_price": 1.1718,
                            "horizon_min": 60,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                forecast = SimpleNamespace(
                    pair="EUR_USD",
                    direction="DOWN",
                    confidence=0.91,
                    current_price=1.17326,
                    target_price=1.1712,
                    invalidation_price=1.1742,
                    horizon_min=60,
                    rationale_summary="DOWN forecast from current tape",
                    drivers_for=("pair_chart SHORT_LEAN",),
                    drivers_against=("range bounce risk",),
                    market_support=_high_precision_market_support("DOWN"),
                )

                with (
                    patch("quant_rabbit.strategy.intent_generator.ROOT", root),
                    patch(
                        "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                        return_value=forecast,
                    ),
                ):
                    summary = IntentGenerator(
                        campaign_plan=_campaign(root, direction="LONG"),
                        strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                        pair_charts_path=pair_charts_path,
                        output_path=output,
                        report_path=root / "intents.md",
                        max_loss_jpy=500.0,
                    ).run(snapshot_path=snapshot_path)

                payload = json.loads(output.read_text())
                rows = [
                    json.loads(line)
                    for line in (data_root / "forecast_history.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
        finally:
            if prior_sl is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl

        same_cycle_rows = [
            row
            for row in rows
            if row.get("cycle_id") == cycle_id and row.get("pair") == "EUR_USD"
        ]
        short_results = [item for item in payload["results"] if item["intent"]["side"] == "SHORT"]
        issue_codes = {
            issue["code"]
            for item in short_results
            for issue in item["risk_issues"]
        }

        self.assertEqual(len(same_cycle_rows), 1)
        self.assertEqual(same_cycle_rows[0]["direction"], "DOWN")
        self.assertEqual(same_cycle_rows[0]["confidence"], 0.91)
        self.assertGreater(summary.live_ready, 0)
        self.assertNotIn("TELEMETRY_FORECAST_HISTORY_MISMATCH_FOR_LIVE", issue_codes)

    def test_generate_intents_skips_projection_telemetry_when_market_closed(self) -> None:
        prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"
                closed_ts = "2026-06-07T14:00:00+00:00"
                snapshot = _snapshot(root, fetched_at_utc=closed_ts, quote_timestamp_utc=closed_ts)
                projection_signal = SimpleNamespace(
                    name="bb_squeeze_expansion_imminent",
                    timeframe="H1",
                    direction="EITHER",
                    lead_time_min=300,
                    confidence=0.74,
                    bonus_magnitude=8.0,
                    rationale="H1 squeeze expansion timing",
                )
                forecast = SimpleNamespace(
                    pair="EUR_USD",
                    direction="DOWN",
                    confidence=0.91,
                    current_price=1.17326,
                    target_price=1.1712,
                    invalidation_price=1.1742,
                    horizon_min=60,
                    rationale_summary="DOWN forecast from closed-market snapshot",
                    drivers_for=("pair_chart SHORT_LEAN",),
                    drivers_against=("range bounce risk",),
                    projection_signals=(projection_signal,),
                )

                with (
                    patch("quant_rabbit.strategy.intent_generator.ROOT", root),
                    patch(
                        "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                        return_value=forecast,
                    ),
                ):
                    IntentGenerator(
                        campaign_plan=_campaign(root, direction="LONG"),
                        strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                        pair_charts_path=_pair_charts_with_direction(
                            root,
                            long_score=0.46,
                            short_score=0.54,
                            dominant_regime="RANGE",
                            m5_regime="RANGE",
                        ),
                        output_path=output,
                        report_path=root / "intents.md",
                        max_loss_jpy=500.0,
                    ).run(snapshot_path=snapshot)

                forecast_rows = (root / "data" / "forecast_history.jsonl").read_text().splitlines()
                projection_path = root / "data" / "projection_ledger.jsonl"
        finally:
            if prior_sl is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl

        self.assertTrue(forecast_rows)
        latest = json.loads(forecast_rows[-1])
        self.assertEqual(latest["pair"], "EUR_USD")
        self.assertEqual(latest["timestamp_utc"], "2026-06-07T14:00:00Z")
        self.assertFalse(projection_path.exists())

    def test_matching_forecast_telemetry_allows_live_ready(self) -> None:
        prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"
                open_ts = "2026-06-08T12:00:00+00:00"
                snapshot = _snapshot(root, fetched_at_utc=open_ts, quote_timestamp_utc=open_ts)
                _write_forecast_telemetry(
                    root,
                    direction="DOWN",
                    confidence=0.91,
                    timestamp_utc="2026-06-08T12:00:00Z",
                )
                forecast = SimpleNamespace(
                    direction="DOWN",
                    confidence=0.91,
                    current_price=1.17326,
                    target_price=1.1712,
                    invalidation_price=1.1742,
                    horizon_min=60,
                    rationale_summary="DOWN forecast from current tape",
                    drivers_for=("pair_chart SHORT_LEAN",),
                    drivers_against=("range bounce risk",),
                    market_support=_high_precision_market_support("DOWN"),
                )

                with (
                    patch("quant_rabbit.strategy.intent_generator.ROOT", root),
                    patch(
                        "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                        return_value=forecast,
                    ),
                ):
                    summary = IntentGenerator(
                        campaign_plan=_campaign(root, direction="LONG"),
                        strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                        pair_charts_path=_pair_charts_with_direction(
                            root,
                            long_score=0.46,
                            short_score=0.54,
                            dominant_regime="RANGE",
                            m5_regime="RANGE",
                        ),
                        output_path=output,
                        report_path=root / "intents.md",
                        max_loss_jpy=500.0,
                    ).run(snapshot_path=snapshot)

                payload = json.loads(output.read_text())
        finally:
            if prior_sl is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl

        short_results = [item for item in payload["results"] if item["intent"]["side"] == "SHORT"]
        self.assertGreater(summary.live_ready, 0)
        self.assertTrue(any(item["status"] == "LIVE_READY" for item in short_results))
        self.assertTrue(
            all(
                "TELEMETRY_" not in issue["code"]
                for item in short_results
                for issue in item["risk_issues"]
            )
        )

    def test_same_cycle_forecast_telemetry_allows_snapshot_timestamp_ordering_skew(self) -> None:
        prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"
                fetched_at = datetime(2026, 6, 8, 12, 5, tzinfo=timezone.utc)
                snapshot = _snapshot(
                    root,
                    fetched_at_utc=fetched_at.isoformat(),
                    quote_timestamp_utc=fetched_at.isoformat(),
                )
                forecast = SimpleNamespace(
                    pair="EUR_USD",
                    direction="DOWN",
                    confidence=0.91,
                    current_price=1.17326,
                    target_price=1.1712,
                    invalidation_price=1.1742,
                    horizon_min=60,
                    rationale_summary="DOWN forecast from current tape",
                    drivers_for=("pair_chart SHORT_LEAN",),
                    drivers_against=("range bounce risk",),
                    market_support=_high_precision_market_support("DOWN"),
                )

                with (
                    patch("quant_rabbit.strategy.intent_generator.ROOT", root),
                    patch(
                        "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                        return_value=forecast,
                    ),
                ):
                    summary = IntentGenerator(
                        campaign_plan=_campaign(root, direction="LONG"),
                        strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                        pair_charts_path=_pair_charts_with_direction(
                            root,
                            long_score=0.46,
                            short_score=0.54,
                            dominant_regime="RANGE",
                            m5_regime="RANGE",
                        ),
                        output_path=output,
                        report_path=root / "intents.md",
                        max_loss_jpy=500.0,
                    ).run(snapshot_path=snapshot)

                payload = json.loads(output.read_text())
                forecast_rows = (root / "data" / "forecast_history.jsonl").read_text().splitlines()
        finally:
            if prior_sl is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl

        latest = json.loads(forecast_rows[-1])
        short_results = [item for item in payload["results"] if item["intent"]["side"] == "SHORT"]
        issue_codes = {
            issue["code"]
            for item in short_results
            for issue in item["risk_issues"]
        }
        self.assertGreater(summary.live_ready, 0)
        self.assertEqual(latest["pair"], "EUR_USD")
        self.assertEqual(latest["cycle_id"], short_results[0]["intent"]["metadata"]["forecast_cycle_id"])
        self.assertNotIn("TELEMETRY_FORECAST_HISTORY_STALE_FOR_LIVE", issue_codes)

    def test_forecast_telemetry_from_different_cycle_cannot_be_live_ready(self) -> None:
        prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"
                fetched_at = datetime.now(timezone.utc) - timedelta(minutes=2)
                snapshot = _snapshot(
                    root,
                    fetched_at_utc=fetched_at.isoformat(),
                    quote_timestamp_utc=fetched_at.isoformat(),
                )
                _write_forecast_telemetry(root, direction="DOWN", confidence=0.91, cycle_id="older-cycle")
                forecast = SimpleNamespace(
                    pair="EUR_USD",
                    direction="DOWN",
                    confidence=0.91,
                    current_price=1.17326,
                    target_price=1.1712,
                    invalidation_price=1.1742,
                    horizon_min=60,
                    rationale_summary="DOWN forecast from current tape",
                    drivers_for=("pair_chart SHORT_LEAN",),
                    drivers_against=("range bounce risk",),
                    market_support=_high_precision_market_support("DOWN"),
                )

                with (
                    patch("quant_rabbit.strategy.intent_generator.ROOT", root),
                    patch(
                        "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                        return_value=forecast,
                    ),
                    patch(
                        "quant_rabbit.strategy.intent_generator._record_forecast_seed_telemetry",
                        return_value=None,
                    ),
                ):
                    summary = IntentGenerator(
                        campaign_plan=_campaign(root, direction="LONG"),
                        strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                        pair_charts_path=_pair_charts_with_direction(
                            root,
                            long_score=0.46,
                            short_score=0.54,
                            dominant_regime="RANGE",
                            m5_regime="RANGE",
                        ),
                        output_path=output,
                        report_path=root / "intents.md",
                        max_loss_jpy=500.0,
                    ).run(snapshot_path=snapshot)

                payload = json.loads(output.read_text())
        finally:
            if prior_sl is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl

        short_results = [item for item in payload["results"] if item["intent"]["side"] == "SHORT"]
        issue_codes = {
            issue["code"]
            for item in short_results
            for issue in item["risk_issues"]
        }
        self.assertEqual(summary.live_ready, 0)
        self.assertIn("TELEMETRY_FORECAST_CONTEXT_REQUIRED_FOR_LIVE", issue_codes)
        self.assertNotIn("TELEMETRY_FORECAST_HISTORY_STALE_FOR_LIVE", issue_codes)

    def test_snapshot_packet_time_prevents_self_stale_intents(self) -> None:
        prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"
                fetched_at = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)
                quote_at = fetched_at - timedelta(seconds=10)
                snapshot = _snapshot(
                    root,
                    fetched_at_utc=fetched_at.isoformat(),
                    quote_timestamp_utc=quote_at.isoformat(),
                )
                _write_forecast_telemetry(
                    root,
                    direction="DOWN",
                    confidence=0.91,
                    timestamp_utc=quote_at.isoformat().replace("+00:00", "Z"),
                )
                forecast = SimpleNamespace(
                    direction="DOWN",
                    confidence=0.91,
                    current_price=1.17326,
                    target_price=1.1712,
                    invalidation_price=1.1742,
                    horizon_min=60,
                    rationale_summary="DOWN forecast from current tape",
                    drivers_for=("pair_chart SHORT_LEAN",),
                    drivers_against=("range bounce risk",),
                    market_support=_high_precision_market_support("DOWN"),
                )

                with (
                    patch("quant_rabbit.strategy.intent_generator.ROOT", root),
                    patch(
                        "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                        return_value=forecast,
                    ),
                ):
                    summary = IntentGenerator(
                        campaign_plan=_campaign(root, direction="LONG"),
                        strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                        pair_charts_path=_pair_charts_with_direction(
                            root,
                            long_score=0.46,
                            short_score=0.54,
                            dominant_regime="RANGE",
                            m5_regime="RANGE",
                        ),
                        output_path=output,
                        report_path=root / "intents.md",
                        max_loss_jpy=500.0,
                    ).run(snapshot_path=snapshot)

                payload = json.loads(output.read_text())
        finally:
            if prior_sl is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl

        short_results = [item for item in payload["results"] if item["intent"]["side"] == "SHORT"]
        stale_codes = {
            issue["code"]
            for item in short_results
            for issue in item["risk_issues"]
            if issue["code"] == "STALE_QUOTE"
        }
        self.assertGreater(summary.live_ready, 0)
        self.assertFalse(stale_codes)

    def test_watch_only_strategy_profile_cannot_become_live_ready_under_sl_free(self) -> None:
        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"

                IntentGenerator(
                    campaign_plan=_campaign(root),
                    strategy_profile=_strategy(root, status="WATCH_ONLY"),
                    output_path=output,
                    report_path=root / "intents.md",
                    pair_charts_path=_pair_charts(root),
                    max_loss_jpy=500.0,
                ).run(snapshot_path=_snapshot(root))

                payload = json.loads(output.read_text())
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        statuses = {item["status"] for item in payload["results"]}
        blockers = [blocker for item in payload["results"] for blocker in item["live_blockers"]]
        self.assertNotIn("LIVE_READY", statuses)
        self.assertIn("DRY_RUN_PASSED", statuses)
        self.assertTrue(any("WATCH_ONLY" in blocker for blocker in blockers))
        live_strategy_issues = [
            issue
            for item in payload["results"]
            for issue in item["live_strategy_issues"]
        ]
        self.assertTrue(
            any(
                issue["code"] == "STRATEGY_NOT_ELIGIBLE" and issue["severity"] == "BLOCK"
                for issue in live_strategy_issues
            )
        )
        strategy_block = next(
            issue
            for issue in live_strategy_issues
            if issue["code"] == "STRATEGY_NOT_ELIGIBLE" and issue["severity"] == "BLOCK"
        )
        self.assertEqual(
            strategy_block["strategy_profile_evidence"]["profile_status"],
            "WATCH_ONLY",
        )
        self.assertEqual(
            strategy_block["strategy_profile_evidence"]["required_fix"],
            "edge exists but old sizing broke the loss cap",
        )

    def test_forecast_seed_pending_trigger_can_repair_watch_only_profile_under_sl_free(self) -> None:
        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"
                technical_context = build_forecast_technical_context(
                    {
                        "confluence": {
                            "dominant_regime": "TREND_UP",
                            "price_percentile_24h": 0.7,
                            "price_percentile_7d": 0.6,
                        },
                        "views": [
                            {
                                "granularity": "M5",
                                "regime_reading": {
                                    "state": "TREND_WEAK",
                                    "atr_percentile": 50.0,
                                },
                                "indicators": {"atr_pips": 8.0},
                                "structure": {
                                    "structure_events": [
                                        {
                                            "kind": "BOS_UP",
                                            "index": 1,
                                            "close_confirmed": True,
                                        }
                                    ]
                                },
                            }
                        ],
                    },
                    pair="EUR_USD",
                    current_price=1.17326,
                    spread_pips=0.8,
                )
                forecast = SimpleNamespace(
                    pair="EUR_USD",
                    direction="UP",
                    confidence=0.91,
                    current_price=1.17326,
                    target_price=1.1762,
                    invalidation_price=1.1718,
                    horizon_min=60,
                    rationale_summary="UP forecast from current tape",
                    drivers_for=("sell-side sweep fade",),
                    drivers_against=("old profile is watch-only",),
                    market_support=_high_precision_market_support("UP"),
                    technical_context_v1=technical_context,
                )

                with patch(
                    "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                    return_value=forecast,
                ):
                    IntentGenerator(
                        campaign_plan=_campaign(root, direction="SHORT"),
                        strategy_profile=_strategy(root, status="WATCH_ONLY", direction="LONG"),
                        pair_charts_path=_pair_charts_with_direction(
                            root,
                            long_score=0.82,
                            short_score=0.18,
                            dominant_regime="TREND_UP",
                            m5_regime="TREND_UP",
                        ),
                        output_path=output,
                        report_path=root / "intents.md",
                        max_loss_jpy=500.0,
                    ).run(snapshot_path=_snapshot(root))

                payload = json.loads(output.read_text())
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        trigger = next(
            item
            for item in payload["results"]
            if item["lane_id"] == "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE"
        )
        market = next(
            item
            for item in payload["results"]
            if item["lane_id"] == "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:MARKET"
        )
        self.assertTrue(trigger["intent"]["metadata"]["forecast_seed"])
        forecast_context = trigger["intent"]["metadata"][
            "forecast_technical_context"
        ]
        self.assertEqual(forecast_context["status"], "VALID")
        self.assertEqual(
            verify_forecast_technical_context_evidence(
                forecast_context,
                pair="EUR_USD",
                current_price=1.17326,
            ),
            (True, None),
        )
        self.assertEqual(trigger["status"], "DRY_RUN_PASSED")
        self.assertIn(MARKET_CLOSE_LEAK_FAMILY_BLOCK_CODE, trigger["live_blocker_codes"])
        self.assertTrue(
            any(
                issue["code"] == "STRATEGY_NOT_ELIGIBLE" and issue["severity"] == "WARN"
                for issue in trigger["live_strategy_issues"]
            )
        )
        self.assertEqual(market["status"], "DRY_RUN_PASSED")
        self.assertTrue(any("WATCH_ONLY" in blocker for blocker in market["live_blockers"]))

    def test_forecast_seed_watch_only_trigger_with_opposing_chart_bias_stays_dry_run(self) -> None:
        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"
                forecast = SimpleNamespace(
                    direction="DOWN",
                    confidence=0.91,
                    current_price=1.17326,
                    target_price=1.1702,
                    invalidation_price=1.1748,
                    horizon_min=60,
                    rationale_summary="DOWN forecast from current tape",
                    drivers_for=("buy-side sweep fade",),
                    drivers_against=("old profile is watch-only",),
                )

                with patch(
                    "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                    return_value=forecast,
                ):
                    IntentGenerator(
                        campaign_plan=_campaign(root, direction="LONG"),
                        strategy_profile=_strategy(root, status="WATCH_ONLY", direction="SHORT"),
                        pair_charts_path=_pair_charts_with_direction(
                            root,
                            long_score=0.82,
                            short_score=0.18,
                            dominant_regime="RANGE",
                            m5_regime="RANGE",
                            m5_long_bias=0.86,
                            m5_short_bias=0.14,
                        ),
                        output_path=output,
                        report_path=root / "intents.md",
                        max_loss_jpy=500.0,
                    ).run(snapshot_path=_snapshot(root))

                payload = json.loads(output.read_text())
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        trigger = next(
            item
            for item in payload["results"]
            if item["lane_id"] == "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE"
        )
        self.assertTrue(trigger["intent"]["metadata"]["forecast_seed"])
        self.assertNotEqual(trigger["status"], "LIVE_READY")
        self.assertTrue(any("WATCH_ONLY" in blocker for blocker in trigger["live_blockers"]))
        self.assertTrue(
            any(
                issue["code"] == "STRATEGY_NOT_ELIGIBLE" and issue["severity"] == "BLOCK"
                for issue in trigger["live_strategy_issues"]
            )
        )

    def test_carries_current_regime_and_session_bucket_from_pair_charts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root),
                pair_charts_path=_pair_charts_with_context(root),
                output_path=output,
                report_path=root / "intents.md",
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            intent = payload["results"][0]["intent"]
            self.assertEqual(intent["metadata"]["regime_state"], "TREND_DOWN")
            self.assertEqual(intent["metadata"]["session_bucket"], "NY")
            self.assertEqual(intent["market_context"]["session"], "NY")
            self.assertIn("TREND_DOWN current", intent["market_context"]["regime"])

    def test_carries_tf_location_and_nearby_level_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root),
                pair_charts_path=_pair_charts_location_map(root),
                levels_path=_levels_snapshot(root),
                output_path=output,
                report_path=root / "intents.md",
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            intent = payload["results"][0]["intent"]
            metadata = intent["metadata"]

            self.assertEqual(metadata["tf_regime_map"]["M5"]["classification"], "RANGE")
            self.assertEqual(metadata["tf_regime_map"]["H1"]["classification"], "TREND_UP")
            self.assertIn("M5", metadata["range_timeframes"])
            self.assertIn("H1:TREND_UP", metadata["trend_timeframes"])
            self.assertTrue(metadata["nearest_levels_below"])
            self.assertTrue(metadata["nearest_levels_above"])
            self.assertTrue(metadata["level_clusters_near"])
            self.assertTrue(all(item["count"] > 1 for item in metadata["level_clusters_near"]))
            self.assertTrue(
                any(
                    str(item["source"]).startswith("levels:")
                    for item in metadata["nearest_levels_below"] + metadata["nearest_levels_above"]
                )
            )
            self.assertIn("M5 RANGE", metadata["market_location_story"])
            self.assertIn("H1 TREND_UP", metadata["market_location_story"])
            self.assertIn(metadata["market_location_story"], intent["market_context"]["chart_story"])

    def test_forecast_first_seed_adds_current_direction_before_mirror_candidates(self) -> None:
        import os

        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"
                forecast = SimpleNamespace(
                    direction="DOWN",
                    confidence=0.91,
                    current_price=1.17326,
                    target_price=1.1688,
                    invalidation_price=1.1742,
                    horizon_min=60,
                    rationale_summary="DOWN forecast from current tape",
                    drivers_for=("pair_chart SHORT_LEAN",),
                    drivers_against=("range bounce risk",),
                    market_support=_high_precision_market_support("DOWN"),
                )

                with patch(
                    "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                    return_value=forecast,
                ):
                    IntentGenerator(
                        campaign_plan=_campaign(root, direction="LONG"),
                        strategy_profile=_strategy(root, status="CANDIDATE", direction="SHORT"),
                        pair_charts_path=_pair_charts_with_direction(
                            root,
                            long_score=0.18,
                            short_score=0.82,
                            dominant_regime="TREND_DOWN",
                            m5_regime="TREND_DOWN",
                        ),
                        output_path=output,
                        report_path=root / "intents.md",
                        max_loss_jpy=500.0,
                    ).run(snapshot_path=_snapshot(root))

                payload = json.loads(output.read_text())
                results = payload["results"]
                seed = next(
                    item
                    for item in results
                    if item["lane_id"] == "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION"
                )

                self.assertTrue(seed["intent"]["metadata"]["forecast_seed"])
                self.assertEqual(seed["intent"]["metadata"]["forecast_direction"], "DOWN")
                self.assertEqual(seed["intent"]["metadata"]["forecast_confidence"], 0.91)
                self.assertEqual(seed["intent"]["tp"], 1.1688)
                self.assertEqual(seed["intent"]["metadata"]["tp_target_source"], "FORECAST_CAPPED_ATR_RR")
                self.assertIn("forecast-first", seed["intent"]["market_context"]["narrative"])
                self.assertEqual(
                    sum(
                        1
                        for item in results
                        if item["lane_id"] == "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION"
                    ),
                    1,
                )
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

    def test_forecast_first_blocks_opposite_direction_candidates(self) -> None:
        import os

        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"
                forecast = SimpleNamespace(
                    direction="DOWN",
                    confidence=0.91,
                    current_price=1.17326,
                    target_price=1.1712,
                    invalidation_price=1.1742,
                    horizon_min=60,
                    rationale_summary="DOWN forecast from current tape",
                    drivers_for=("pair_chart SHORT_LEAN",),
                    drivers_against=("range bounce risk",),
                    market_support=_high_precision_market_support("DOWN"),
                )

                with patch(
                    "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                    return_value=forecast,
                ):
                    summary = IntentGenerator(
                        campaign_plan=_campaign(root, direction="LONG"),
                        strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                        pair_charts_path=_pair_charts_with_direction(
                            root,
                            long_score=0.46,
                            short_score=0.54,
                            dominant_regime="RANGE",
                            m5_regime="RANGE",
                        ),
                        output_path=output,
                        report_path=root / "intents.md",
                        max_loss_jpy=500.0,
                    ).run(snapshot_path=_snapshot(root))

                payload = json.loads(output.read_text())
                long_results = [item for item in payload["results"] if item["intent"]["side"] == "LONG"]
                short_results = [item for item in payload["results"] if item["intent"]["side"] == "SHORT"]
                long_issue_codes = {
                    issue["code"]
                    for item in long_results
                    for issue in item["risk_issues"]
                    if issue["severity"] == "BLOCK"
                }

                self.assertGreater(summary.live_ready, 0)
                self.assertTrue(short_results)
                self.assertTrue(any(item["status"] == "LIVE_READY" for item in short_results))
                self.assertTrue(long_results)
                self.assertTrue(all(item["status"] == "DRY_RUN_BLOCKED" for item in long_results))
                self.assertIn("FORECAST_DIRECTION_CONFLICT", long_issue_codes)
                self.assertTrue(
                    all(
                        item["intent"]["metadata"]["forecast_direction"] == "DOWN"
                        and item["intent"]["metadata"]["forecast_confidence"] == 0.91
                        for item in payload["results"]
                    )
                )
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

    def test_forecast_tp_does_not_cap_when_target_fails_execution_floor(self) -> None:
        import os

        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"
                forecast = SimpleNamespace(
                    direction="DOWN",
                    confidence=0.91,
                    current_price=1.17326,
                    target_price=1.1708,
                    invalidation_price=1.1742,
                    horizon_min=60,
                    rationale_summary="DOWN forecast from current tape",
                    drivers_for=("pair_chart SHORT_LEAN",),
                    drivers_against=("range bounce risk",),
                )

                with patch(
                    "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                    return_value=forecast,
                ):
                    IntentGenerator(
                        campaign_plan=_campaign(root, direction="LONG"),
                        strategy_profile=_strategy(root, status="CANDIDATE", direction="SHORT"),
                        pair_charts_path=_pair_charts_with_direction(
                            root,
                            long_score=0.18,
                            short_score=0.82,
                            dominant_regime="TREND_DOWN",
                            m5_regime="TREND_DOWN",
                        ),
                        output_path=output,
                        report_path=root / "intents.md",
                        max_loss_jpy=500.0,
                    ).run(snapshot_path=_snapshot(root))

                payload = json.loads(output.read_text())
                seed = next(
                    item
                    for item in payload["results"]
                    if item["lane_id"] == "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION"
                )

                self.assertTrue(seed["intent"]["metadata"]["forecast_seed"])
                self.assertEqual(seed["intent"]["metadata"]["forecast_target_price"], 1.1708)
                self.assertNotEqual(seed["intent"]["tp"], 1.1708)
                self.assertEqual(seed["intent"]["metadata"]["tp_target_source"], "ATR_RR")
                self.assertIn(
                    "forecast target skipped: forecast TP RR",
                    seed["intent"]["metadata"]["tp_target_reason"],
                )
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

    def test_below_entry_forecast_still_marks_existing_lanes_as_forecast_context(self) -> None:
        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"
            forecast = SimpleNamespace(
                direction="UP",
                confidence=0.51,
                current_price=1.17326,
                target_price=1.1762,
                invalidation_price=1.1718,
                horizon_min=60,
                rationale_summary="UP forecast below fresh-entry threshold",
                drivers_for=("pair_chart LONG_LEAN",),
                drivers_against=("confidence below entry gate",),
            )

            with patch(
                "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                return_value=forecast,
            ):
                summary = IntentGenerator(
                    campaign_plan=_campaign(root, direction="LONG"),
                    strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                    pair_charts_path=_pair_charts_with_direction(
                        root,
                        long_score=0.72,
                        short_score=0.28,
                        dominant_regime="TREND_UP",
                        m5_regime="TREND_UP",
                    ),
                    output_path=output,
                    report_path=root / "intents.md",
                    max_loss_jpy=500.0,
                ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
        issue_codes = {issue["code"] for item in payload["results"] for issue in item["risk_issues"]}

        self.assertEqual(summary.live_ready, 0)
        self.assertIn("FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", issue_codes)
        self.assertNotIn("FORECAST_CONTEXT_REQUIRED_FOR_LIVE", issue_codes)
        self.assertTrue(
            all(
                item["intent"]["metadata"]["forecast_direction"] == "UP"
                and item["intent"]["metadata"]["forecast_confidence"] == 0.51
                and not item["intent"]["metadata"]["forecast_seed"]
                for item in payload["results"]
                if item["intent"]["pair"] == "EUR_USD"
            )
        )

    def test_audited_projection_support_can_clear_near_miss_forecast_floor(self) -> None:
        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"
            forecast = SimpleNamespace(
                direction="UP",
                confidence=0.58,
                raw_confidence=0.67,
                calibration_multiplier=0.87,
                current_price=1.17326,
                target_price=1.1762,
                invalidation_price=1.1718,
                horizon_min=60,
                rationale_summary="UP forecast damped by broad directional calibration",
                drivers_for=("M5 liquidity sweep low fade",),
                drivers_against=("directional forecast calibration below entry gate",),
                component_scores={"UP": 110.2, "DOWN": 49.8, "RANGE": 14.0, "EITHER": 9.5},
                market_support={
                    "ok": True,
                    "direction": "UP",
                    "aligned_projection_count": 1,
                    "timing_projection_count": 0,
                    "best_hit_rate": 1.0,
                    "best_samples": 40,
                    "best_aligned_hit_rate": 1.0,
                    "best_aligned_samples": 40,
                    "directional_calibration_name": "directional_forecast_up",
                    "directional_hit_rate": 1.0,
                    "directional_samples": 40,
                    "reason": "liquidity_sweep_low UP hit_rate=1.00 samples=40 supports weak calibrated forecast",
                    "signals": [
                        {
                            "name": "liquidity_sweep_low",
                            "direction": "UP",
                            "confidence": 0.88,
                            "hit_rate": 1.0,
                            "samples": 40,
                            "target_pips": 6.0,
                        }
                    ],
                },
            )

            with patch(
                "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                return_value=forecast,
            ):
                summary = IntentGenerator(
                    campaign_plan=_campaign(root, direction="LONG"),
                    strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                    pair_charts_path=_pair_charts_with_direction(
                        root,
                        long_score=0.72,
                        short_score=0.28,
                        dominant_regime="TREND_UP",
                        m5_regime="TREND_UP",
                    ),
                    output_path=output,
                    report_path=root / "intents.md",
                    max_loss_jpy=500.0,
                ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())

        issue_codes = {issue["code"] for item in payload["results"] for issue in item["risk_issues"]}
        live_ready = [item for item in payload["results"] if item["status"] == "LIVE_READY"]
        blocked_family = [
            item
            for item in payload["results"]
            if item["intent"]["market_context"]["method"] == "BREAKOUT_FAILURE"
            and MARKET_CLOSE_LEAK_FAMILY_BLOCK_CODE in item.get("live_blocker_codes", [])
        ]

        self.assertEqual(summary.live_ready, 0)
        self.assertFalse(live_ready)
        self.assertNotIn("FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", issue_codes)
        self.assertIn(MARKET_CLOSE_LEAK_FAMILY_BLOCK_CODE, issue_codes)
        self.assertTrue(blocked_family)
        self.assertTrue(all(item["intent"]["metadata"]["forecast_market_support_ok"] for item in blocked_family))
        self.assertTrue(
            all(
                item["intent"]["metadata"]["forecast_directional_calibration_name"]
                == "directional_forecast_up"
                and item["intent"]["metadata"]["forecast_directional_hit_rate"] == 1.0
                and item["intent"]["metadata"]["forecast_directional_samples"] == 40
                for item in blocked_family
            )
        )

    def test_strong_directional_watch_override_rewrites_receipt_for_live_ready_stop(self) -> None:
        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"
            campaign = root / "campaign.json"
            campaign.write_text(json.dumps({"lanes": []}))
            forecast = SimpleNamespace(
                direction="UP",
                confidence=0.44,
                raw_confidence=0.63,
                calibration_multiplier=0.70,
                current_price=1.17326,
                target_price=1.1762,
                invalidation_price=1.1718,
                horizon_min=60,
                rationale_summary="raw forecast near floor with strong audited event support",
                drivers_for=("macro_event_nowcast_central_bank UP",),
                drivers_against=("calibrated confidence below entry gate",),
                component_scores={"UP": 142.0, "DOWN": 63.0, "RANGE": 0.0, "EITHER": 12.0},
                market_support={
                    "ok": True,
                    "direction": "UP",
                    "aligned_projection_count": 1,
                    "timing_projection_count": 1,
                    "best_hit_rate": 1.0,
                    "best_samples": 40,
                    "reason": "macro_event_nowcast_central_bank UP hit_rate=1.00 samples=40 supports weak calibrated forecast",
                    "signals": [
                        {
                            "name": "macro_event_nowcast_central_bank",
                            "direction": "UP",
                            "confidence": 0.79,
                            "hit_rate": 1.0,
                            "samples": 40,
                        },
                        {
                            "name": "bb_squeeze_expansion_imminent",
                            "direction": "EITHER",
                            "confidence": 0.63,
                            "hit_rate": 1.0,
                            "samples": 40,
                        },
                    ],
                },
            )

            with patch(
                "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                return_value=forecast,
            ):
                IntentGenerator(
                    campaign_plan=campaign,
                    strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                    pair_charts_path=_pair_charts_with_direction(
                        root,
                        long_score=0.72,
                        short_score=0.28,
                        dominant_regime="TREND_UP",
                        m5_regime="TREND_UP",
                    ),
                    output_path=output,
                    report_path=root / "intents.md",
                    max_loss_jpy=500.0,
                ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())

        blocked_ready_shape = [
            item
            for item in payload["results"]
            if item["intent"]["order_type"] == OrderType.STOP_ENTRY.value
            and item["intent"]["metadata"].get("forecast_watch_only_live_override")
            and MARKET_CLOSE_LEAK_FAMILY_BLOCK_CODE in item.get("live_blocker_codes", [])
        ]

        self.assertTrue(blocked_ready_shape)
        metadata = blocked_ready_shape[0]["intent"]["metadata"]
        receipt = metadata["required_receipt"]
        event_risk = blocked_ready_shape[0]["intent"]["market_context"]["event_risk"]
        self.assertTrue(metadata["forecast_watch_only"])
        self.assertTrue(metadata["forecast_watch_only_live_override"])
        self.assertIn("Forecast support override", receipt)
        self.assertNotIn("Watch-only forecast-first lane", receipt)
        self.assertNotIn("Do not send live until", receipt)
        self.assertIn("forecast support override", event_risk.lower())
        self.assertNotIn("watch-only forecast candidate", event_risk.lower())

    def test_forecast_context_payload_persists_news_refs(self) -> None:
        forecast = SimpleNamespace(
            direction="UP",
            confidence=0.57,
            raw_confidence=0.61,
            calibration_multiplier=0.93,
            current_price=1.17326,
            target_price=1.1762,
            invalidation_price=1.1718,
            horizon_min=60,
            rationale_summary="UP forecast has fresh news-theme follow-through",
            drivers_for=("news_theme_followthrough USD soft",),
            drivers_against=(),
            component_scores={"UP": 96.0, "DOWN": 41.0, "RANGE": 8.0, "EITHER": 4.0},
            market_support={
                "ok": True,
                "direction": "UP",
                "aligned_projection_count": 1,
                "timing_projection_count": 0,
                "best_hit_rate": 0.6,
                "best_samples": 32,
                "reason": "news_theme_followthrough UP hit_rate=0.60 samples=32 supports forecast",
                "signals": [
                    {
                        "name": "news_theme_followthrough",
                        "direction": "UP",
                        "confidence": 0.73,
                        "hit_rate": 0.6,
                        "samples": 32,
                    }
                ],
            },
        )

        metadata = _forecast_context_payload(forecast, cycle_id="cycle-news")

        self.assertEqual(metadata["forecast_cycle_id"], "cycle-news")
        self.assertEqual(metadata["news_refs"], ["news:digest", "news:items"])
        self.assertEqual(metadata["news_digest_ref"], "news:digest")
        self.assertEqual(metadata["news_signal_names"], ["news_theme_followthrough"])

    def test_forecast_context_payload_persists_directional_calibration(self) -> None:
        forecast = SimpleNamespace(
            direction="UP",
            confidence=0.82,
            raw_confidence=0.91,
            rationale_summary="high-confidence directional forecast",
            drivers_for=(),
            drivers_against=(),
            component_scores={"UP": 91.0, "DOWN": 12.0},
            market_support={
                "ok": False,
                "direction": "UP",
                "directional_calibration_name": "directional_forecast_up",
                "directional_hit_rate": 0.1,
                "directional_samples": 12,
                "directional_economic_hit_rate": 0.08,
                "directional_economic_samples": 20,
                "directional_timeout_rate": 0.4,
                "directional_timeout_count": 8,
                "reason": "directional forecast bucket is weak",
            },
        )

        metadata = _forecast_context_payload(forecast)

        self.assertEqual(
            metadata["forecast_directional_calibration_name"],
            "directional_forecast_up",
        )
        self.assertAlmostEqual(metadata["forecast_directional_hit_rate"], 0.1)
        self.assertEqual(metadata["forecast_directional_samples"], 12)
        self.assertAlmostEqual(metadata["forecast_directional_economic_hit_rate"], 0.08)
        self.assertEqual(metadata["forecast_directional_economic_samples"], 20)
        self.assertAlmostEqual(metadata["forecast_directional_timeout_rate"], 0.4)
        self.assertEqual(metadata["forecast_directional_timeout_count"], 8)
        support = metadata["forecast_market_support"]
        self.assertEqual(support["directional_calibration_name"], "directional_forecast_up")
        self.assertAlmostEqual(support["directional_hit_rate"], 0.1)
        self.assertEqual(support["directional_samples"], 12)
        self.assertAlmostEqual(support["directional_economic_hit_rate"], 0.08)
        self.assertEqual(support["directional_economic_samples"], 20)
        self.assertAlmostEqual(support["directional_timeout_rate"], 0.4)
        self.assertEqual(support["directional_timeout_count"], 8)

    def test_same_cycle_projection_bootstrap_can_clear_near_miss_forecast_floor(self) -> None:
        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"
            forecast = SimpleNamespace(
                direction="UP",
                confidence=0.58,
                raw_confidence=0.67,
                calibration_multiplier=0.87,
                current_price=1.17326,
                target_price=1.1762,
                invalidation_price=1.1718,
                horizon_min=60,
                rationale_summary="UP forecast damped while same-cycle projection is strong",
                drivers_for=("event surprise follow-through UP",),
                drivers_against=("directional forecast calibration below entry gate",),
                component_scores={"UP": 116.0, "DOWN": 44.0, "RANGE": 8.0, "EITHER": 4.0},
                market_support={
                    "ok": True,
                    "direction": "UP",
                    "aligned_projection_count": 1,
                    "timing_projection_count": 0,
                    "best_hit_rate": None,
                    "best_samples": 0,
                    "bootstrap_projection_support": True,
                    "reason": (
                        "event_surprise_followthrough UP same-cycle bootstrap: "
                        "signal_conf=0.86, raw_forecast_conf=0.66, calibrated_conf=0.49; "
                        "ledger samples pending"
                    ),
                    "signals": [
                        {
                            "name": "event_surprise_followthrough",
                            "direction": "UP",
                            "confidence": 0.86,
                            "samples": 0,
                            "bootstrap_projection_support": True,
                        }
                    ],
                },
            )

            with patch(
                "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                return_value=forecast,
            ):
                summary = IntentGenerator(
                    campaign_plan=_campaign(root, direction="LONG"),
                    strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                    pair_charts_path=_pair_charts_with_direction(
                        root,
                        long_score=0.72,
                        short_score=0.28,
                        dominant_regime="TREND_UP",
                        m5_regime="TREND_UP",
                    ),
                    output_path=output,
                    report_path=root / "intents.md",
                    max_loss_jpy=500.0,
                ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())

        live_ready = [item for item in payload["results"] if item["status"] == "LIVE_READY"]
        issue_codes = {issue["code"] for item in payload["results"] for issue in item["risk_issues"]}
        blocked_family = [
            item
            for item in payload["results"]
            if item["intent"]["market_context"]["method"] == "BREAKOUT_FAILURE"
            and MARKET_CLOSE_LEAK_FAMILY_BLOCK_CODE in item.get("live_blocker_codes", [])
        ]

        self.assertEqual(summary.live_ready, 0)
        self.assertFalse(live_ready)
        self.assertNotIn("FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", issue_codes)
        self.assertIn(MARKET_CLOSE_LEAK_FAMILY_BLOCK_CODE, issue_codes)
        self.assertTrue(blocked_family)
        self.assertTrue(all(item["intent"]["metadata"]["forecast_market_support_ok"] for item in blocked_family))
        self.assertTrue(
            all(
                item["intent"]["metadata"]["forecast_market_support"]["bootstrap_projection_support"]
                for item in blocked_family
            )
        )

    def test_same_cycle_projection_bootstrap_requires_raw_forecast_above_live_floor(self) -> None:
        from quant_rabbit.models import MarketContext, TradeMethod
        from quant_rabbit.strategy.intent_generator import _forecast_live_readiness_issue

        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        metadata = {
            "forecast_direction": "UP",
            "forecast_confidence": 0.58,
            "forecast_raw_confidence": 0.64,
            "chart_direction_bias": "LONG",
            "forecast_market_support": {
                "ok": True,
                "direction": "UP",
                "aligned_projection_count": 1,
                "timing_projection_count": 0,
                "best_hit_rate": None,
                "best_samples": 0,
                "bootstrap_projection_support": True,
                "reason": "same-cycle bootstrap should not bypass the live directional floor",
            },
        }
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.STOP_ENTRY,
            units=5000,
            entry=1.1734,
            tp=1.1762,
            sl=1.1718,
            thesis="near-miss bootstrap with raw forecast still below live floor",
            market_context=MarketContext(
                regime="TREND_UP current; TREND_CONTINUATION campaign lane",
                narrative="",
                chart_story="",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="",
            ),
            metadata=metadata,
        )

        issue = _forecast_live_readiness_issue(intent, metadata, TradeMethod.TREND_CONTINUATION)

        self.assertEqual(issue["code"], "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE")

    def test_unclear_forecast_limit_allows_same_side_unselected_projection_support(self) -> None:
        from quant_rabbit.models import MarketContext, TradeMethod
        from quant_rabbit.strategy.intent_generator import _forecast_live_readiness_issue

        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        metadata = {
            "forecast_direction": "UNCLEAR",
            "forecast_confidence": 0.21,
            "forecast_horizon_min": 60,
            "chart_direction_bias": "LONG",
            "forecast_market_support": {
                "ok": False,
                "direction": "UNCLEAR",
                "reason": "forecast UNCLEAR has no executable direction; audited projection unselected",
                "unselected_projection_count": 1,
                "unselected_signals": [
                    {
                        "name": "macro_event_nowcast_central_bank",
                        "direction": "UP",
                        "confidence": 0.74,
                        "hit_rate": 1.0,
                        "samples": 40,
                    }
                ],
            },
        }
        intent = OrderIntent(
            pair="EUR_JPY",
            side=Side.LONG,
            order_type=OrderType.LIMIT,
            units=5000,
            entry=168.4,
            tp=168.9,
            sl=168.0,
            thesis="passive failed-break retest with audited same-side projection",
            market_context=MarketContext(
                regime="BREAKOUT_FAILURE retest",
                narrative="",
                chart_story="",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="",
            ),
            metadata=metadata,
        )

        issue = _forecast_live_readiness_issue(intent, metadata, TradeMethod.BREAKOUT_FAILURE)

        self.assertIsNone(issue)

    def test_unclear_forecast_market_keeps_same_side_unselected_projection_blocked(self) -> None:
        from quant_rabbit.models import MarketContext, TradeMethod
        from quant_rabbit.strategy.intent_generator import _forecast_live_readiness_issue

        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        metadata = {
            "forecast_direction": "UNCLEAR",
            "forecast_confidence": 0.21,
            "chart_direction_bias": "LONG",
            "forecast_market_support": {
                "ok": False,
                "direction": "UNCLEAR",
                "unselected_projection_count": 1,
                "unselected_signals": [
                    {
                        "name": "macro_event_nowcast_central_bank",
                        "direction": "UP",
                        "confidence": 0.74,
                        "hit_rate": 0.562,
                        "samples": 16,
                    }
                ],
            },
        }
        intent = OrderIntent(
            pair="EUR_JPY",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=5000,
            entry=None,
            tp=168.9,
            sl=168.0,
            thesis="market chase must wait for executable pair forecast",
            market_context=MarketContext(
                regime="BREAKOUT_FAILURE retest",
                narrative="",
                chart_story="",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="",
            ),
            metadata=metadata,
        )

        issue = _forecast_live_readiness_issue(intent, metadata, TradeMethod.BREAKOUT_FAILURE)

        self.assertEqual(issue["code"], "FORECAST_NOT_EXECUTABLE_FOR_LIVE")

    def test_weak_forecast_trend_continuation_needs_two_higher_tf_confirmations(self) -> None:
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _forecast_live_readiness_issue

        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        metadata = {
            "forecast_seed": True,
            "forecast_direction": "UP",
            "forecast_confidence": 0.45,
            "forecast_raw_confidence": 0.72,
            "chart_direction_bias": "LONG",
            "trend_timeframes": ["H4:TREND_UP"],
            "tf_regime_map": {
                "H1": {"classification": "RANGE"},
                "H4": {"classification": "TREND_UP"},
                "D": {"classification": "RANGE"},
            },
            "forecast_market_support": {
                "ok": True,
                "direction": "UP",
                "aligned_projection_count": 1,
                "timing_projection_count": 0,
                "best_hit_rate": 0.88,
                "best_samples": 75,
                "reason": "macro_event_nowcast_central_bank UP supports weak calibrated forecast",
                "signals": [
                    {
                        "name": "macro_event_nowcast_central_bank",
                        "direction": "UP",
                        "confidence": 0.79,
                        "hit_rate": 0.88,
                        "samples": 75,
                    }
                ],
            },
        }
        intent = OrderIntent(
            pair="EUR_CHF",
            side=Side.LONG,
            order_type=OrderType.STOP_ENTRY,
            units=6300,
            entry=0.92177,
            tp=0.92317,
            sl=0.91909,
            thesis="weak forecast-first continuation in range",
            market_context=MarketContext(
                regime="RANGE current; TREND_CONTINUATION campaign lane",
                narrative="",
                chart_story="",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="",
            ),
            metadata=metadata,
        )

        issue = _forecast_live_readiness_issue(intent, metadata, TradeMethod.TREND_CONTINUATION)

        self.assertEqual(issue["code"], "FORECAST_TREND_CONTINUATION_HIGHER_TF_REQUIRED_FOR_LIVE")

    def test_weak_forecast_trend_continuation_allows_two_higher_tf_confirmations(self) -> None:
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _forecast_live_readiness_issue

        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        metadata = {
            "forecast_seed": True,
            "forecast_direction": "DOWN",
            "forecast_confidence": 0.56,
            "forecast_raw_confidence": 0.80,
            "chart_direction_bias": "SHORT",
            "trend_timeframes": ["H4:TREND_DOWN", "D:TREND_DOWN"],
            "forecast_market_support": {
                "ok": True,
                "direction": "DOWN",
                "aligned_projection_count": 1,
                "timing_projection_count": 0,
                "best_hit_rate": 1.0,
                "best_samples": 40,
                "reason": "macro_event_nowcast_central_bank DOWN supports weak calibrated forecast",
                "signals": [
                    {
                        "name": "macro_event_nowcast_central_bank",
                        "direction": "DOWN",
                        "confidence": 0.79,
                        "hit_rate": 1.0,
                        "samples": 40,
                    }
                ],
            },
        }
        intent = OrderIntent(
            pair="CAD_JPY",
            side=Side.SHORT,
            order_type=OrderType.STOP_ENTRY,
            units=11000,
            entry=114.525,
            tp=114.14,
            sl=114.673,
            thesis="weak forecast-first continuation with higher-TF confirmation",
            market_context=MarketContext(
                regime="TREND_DOWN current; TREND_CONTINUATION campaign lane",
                narrative="",
                chart_story="",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="",
            ),
            metadata=metadata,
        )

        issue = _forecast_live_readiness_issue(intent, metadata, TradeMethod.TREND_CONTINUATION)

        self.assertIsNone(issue)

    def test_same_cycle_projection_bootstrap_is_built_without_ledger_samples(self) -> None:
        from quant_rabbit.strategy.intent_generator import _forecast_market_support_for_forecast

        forecast = SimpleNamespace(
            direction="DOWN",
            confidence=0.49,
            raw_confidence=0.66,
        )
        projection_signal = SimpleNamespace(
            name="event_surprise_followthrough",
            timeframe=None,
            direction="DOWN",
            confidence=0.86,
            bonus_magnitude=16.0,
            rationale="USD high-impact payroll beat -> EUR_USD DOWN",
        )

        support = _forecast_market_support_for_forecast(
            pair="EUR_USD",
            forecast=forecast,
            projection_signals=[projection_signal],
            hit_rates=None,
            regime="TREND",
        )

        self.assertFalse(support["ok"])
        self.assertFalse(support["bootstrap_projection_support"])
        self.assertEqual(support["best_samples"], 0)
        self.assertEqual(support["reason"], "no directional audited projection support")

    def test_same_cycle_projection_bootstrap_rejects_audited_bad_signal(self) -> None:
        from quant_rabbit.strategy.intent_generator import _forecast_market_support_for_forecast

        forecast = SimpleNamespace(
            direction="UP",
            confidence=0.49,
            raw_confidence=0.66,
        )
        projection_signal = SimpleNamespace(
            name="cross_asset_dxy_lag",
            timeframe="H1",
            direction="UP",
            confidence=0.86,
            bonus_magnitude=16.0,
            rationale="DXY lag maps to USD_CHF UP",
        )
        hit_rates = {
            "cross_asset_dxy_lag": {
                "_all_pairs:_all_regimes": {"hit_rate": 0.0, "samples": 14},
            }
        }

        support = _forecast_market_support_for_forecast(
            pair="USD_CHF",
            forecast=forecast,
            projection_signals=[projection_signal],
            hit_rates=hit_rates,
            regime="TREND",
        )

        self.assertFalse(support["ok"])
        self.assertFalse(support["bootstrap_projection_support"])
        self.assertEqual(support["signals"][0]["name"], "cross_asset_dxy_lag")
        self.assertEqual(support["signals"][0]["samples"], 14)
        self.assertEqual(support["signals"][0]["hit_rate"], 0.0)

    def test_forecast_seed_uses_supplied_data_root_for_calibration_and_context(self) -> None:
        from quant_rabbit.strategy.intent_generator import (
            _forecast_seed_for_pair,
            _load_pair_charts,
            _snapshot_from_json,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            (root / "logs").mkdir()
            charts = _load_pair_charts(
                _pair_charts_with_direction(
                    root,
                    long_score=0.72,
                    short_score=0.28,
                    dominant_regime="TREND_UP",
                    m5_regime="TREND_UP",
                )
            )
            snapshot = _snapshot_from_json(json.loads(_snapshot(root).read_text()))
            forecast = SimpleNamespace(
                direction="UP",
                confidence=0.58,
                raw_confidence=0.58,
                current_price=1.17326,
                target_price=1.1762,
                invalidation_price=1.1718,
                horizon_min=60,
                rationale_summary="UP forecast",
                drivers_for=(),
                drivers_against=(),
                component_scores={"UP": 80.0, "DOWN": 30.0},
            )
            projection_signal = SimpleNamespace(
                name="liquidity_sweep_low",
                timeframe="M5",
                direction="UP",
                confidence=0.86,
                bonus_magnitude=8.0,
                rationale="M5 equal-lows 6.0pip sweep low fade",
            )
            hit_rates = {
                "liquidity_sweep_low_up": {
                    "EUR_USD:TREND": {"hit_rate": 1.0, "samples": 40},
                    "EUR_USD:_all_regimes": {"hit_rate": 1.0, "samples": 40},
                }
            }

            with patch(
                "quant_rabbit.strategy.intent_generator._forecast_seed_has_rich_chart_context",
                return_value=True,
            ), patch(
                "quant_rabbit.strategy.pattern_signals.detect_pattern_signals",
                return_value=[],
            ), patch(
                "quant_rabbit.strategy.forward_projection.detect_forward_projections",
                return_value=[projection_signal],
            ) as detect_forward, patch(
                "quant_rabbit.strategy.correlation_predictor.detect_correlation_lag",
                return_value=[],
            ), patch(
                "quant_rabbit.strategy.path_projection.detect_paths",
                return_value=[],
            ), patch(
                "quant_rabbit.strategy.reversal_signal.detect_reversal",
                return_value=None,
            ), patch(
                "quant_rabbit.strategy.projection_ledger.compute_hit_rates",
                return_value=hit_rates,
            ) as compute_hit_rates, patch(
                "quant_rabbit.strategy.directional_forecaster.synthesize_forecast",
                return_value=forecast,
            ) as synthesize_forecast:
                seed = _forecast_seed_for_pair("EUR_USD", charts or {}, snapshot, data_root=data_root)

            self.assertIsNotNone(seed)
            compute_hit_rates.assert_called_once_with(data_root)
            forecast_kwargs = synthesize_forecast.call_args.kwargs
            self.assertAlmostEqual(forecast_kwargs["spread_pips"], 0.8)
            self.assertIs(forecast_kwargs["require_technical_candle_integrity"], True)
            self.assertEqual(
                forecast_kwargs["pair_chart"]["generated_at_utc"],
                charts["EUR_USD"]["generated_at_utc"],
            )
            self.assertNotEqual(forecast_kwargs["pair_chart"]["generated_at_utc"], "")
            kwargs = detect_forward.call_args.kwargs
            self.assertEqual(kwargs["calendar_path"], data_root / "economic_calendar.json")
            self.assertEqual(kwargs["news_digest_path"], root / "logs" / "news_digest.md")
            self.assertEqual(kwargs["news_items_path"], data_root / "news_items.json")
            self.assertEqual(kwargs["cross_asset_path"], data_root / "cross_asset_snapshot.json")
            self.assertTrue(seed.market_support["ok"])

    def test_event_surprise_forecast_seed_gets_macro_size_up(self) -> None:
        prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"
                forecast = SimpleNamespace(
                    direction="DOWN",
                    confidence=0.88,
                    raw_confidence=0.88,
                    current_price=1.17326,
                    target_price=1.1682,
                    invalidation_price=1.1742,
                    horizon_min=240,
                    rationale_summary="NFP beat creates USD follow-through",
                    drivers_for=("event surprise follow-through DOWN",),
                    drivers_against=(),
                    component_scores={"UP": 12.0, "DOWN": 95.0, "RANGE": 4.0, "EITHER": 0.0},
                    market_support={
                        "ok": True,
                        "direction": "DOWN",
                        "aligned_projection_count": 1,
                        "timing_projection_count": 0,
                        "best_hit_rate": None,
                        "best_samples": 0,
                        "bootstrap_projection_support": True,
                        "reason": "event_surprise_followthrough DOWN same-cycle bootstrap",
                        "signals": [
                            {
                                "name": "event_surprise_followthrough",
                                "direction": "DOWN",
                                "confidence": 0.9,
                                "samples": 0,
                                "bootstrap_projection_support": True,
                            }
                        ],
                    },
                )

                with patch(
                    "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                    return_value=forecast,
                ):
                    IntentGenerator(
                        campaign_plan=_campaign(root, direction="LONG"),
                        strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                        pair_charts_path=_pair_charts_with_direction(
                            root,
                            long_score=0.22,
                            short_score=0.78,
                            dominant_regime="TREND_DOWN",
                            m5_regime="TREND_DOWN",
                            atr_pips=3.0,
                        ),
                        output_path=output,
                        report_path=root / "intents.md",
                        max_loss_jpy=500.0,
                    ).run(snapshot_path=_snapshot(root))

                payload = json.loads(output.read_text())
        finally:
            if prior_sl is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl

        event_items = [
            item for item in payload["results"]
            if item["intent"]
            and item["intent"]["side"] == "SHORT"
            and item["intent"]["metadata"].get("macro_event_size_up")
        ]
        self.assertTrue(event_items)
        event = event_items[0]
        metadata = event["intent"]["metadata"]
        self.assertEqual(metadata["macro_event_signal_name"], "event_surprise_followthrough")
        self.assertEqual(metadata["max_loss_jpy"], 500.0)
        self.assertEqual(metadata["macro_event_confidence_band"], "HIGH")
        self.assertEqual(metadata["macro_event_risk_fraction"], 1.0)
        self.assertLessEqual(metadata["max_loss_jpy"], metadata["macro_event_fresh_absolute_cap_jpy"])
        self.assertTrue(metadata["macro_event_loss_budget_target"])
        self.assertLessEqual(event["risk_metrics"]["risk_jpy"], metadata["max_loss_jpy"])

    def test_below_entry_forecast_for_missing_strong_chart_pair_becomes_watch_only_lane(self) -> None:
        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"
            charts = root / "pair_charts_multi_pair.json"
            charts.write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "charts": [
                            {
                                "pair": "EUR_USD",
                                "dominant_regime": "TREND_UP",
                                "long_score": 0.72,
                                "short_score": 0.28,
                                "views": [
                                    {
                                        "granularity": "M5",
                                        "regime": "TREND_UP",
                                        "long_bias": 0.72,
                                        "short_bias": 0.28,
                                        "regime_reading": {"state": "TREND_UP", "confidence": 0.6},
                                        "family_scores": {"trend_score": 0.8, "mean_rev_score": 0.1},
                                        "indicators": {
                                            "atr_pips": 8.0,
                                            "bb_lower": 1.1710,
                                            "bb_upper": 1.1760,
                                            "donchian_low": 1.1707,
                                            "donchian_high": 1.1764,
                                            "swing_low": 1.1705,
                                            "swing_high": 1.1767,
                                            "adx_14": 27.0,
                                        },
                                    }
                                ],
                            },
                            {
                                "pair": "USD_JPY",
                                "dominant_regime": "TREND_UP",
                                "long_score": 0.82,
                                "short_score": 0.12,
                                "views": [
                                    {
                                        "granularity": "M5",
                                        "regime": "TREND_UP",
                                        "long_bias": 0.82,
                                        "short_bias": 0.12,
                                        "regime_reading": {"state": "TREND_UP", "confidence": 0.72},
                                        "family_scores": {"trend_score": 0.9, "mean_rev_score": 0.1},
                                        "indicators": {
                                            "atr_pips": 8.0,
                                            "bb_lower": 156.10,
                                            "bb_upper": 157.20,
                                            "donchian_low": 156.05,
                                            "donchian_high": 157.25,
                                            "swing_low": 155.90,
                                            "swing_high": 157.40,
                                            "adx_14": 29.0,
                                        },
                                    }
                                ],
                            },
                        ]
                    }
                )
            )
            forecast = SimpleNamespace(
                direction="UP",
                confidence=0.51,
                raw_confidence=0.70,
                current_price=156.644,
                target_price=157.20,
                invalidation_price=156.10,
                horizon_min=60,
                rationale_summary="raw chart strong but calibrated below live floor",
                drivers_for=("USD_JPY LONG_LEAN chart score",),
                drivers_against=("calibration below entry gate",),
            )

            with patch(
                "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                return_value=forecast,
            ):
                summary = IntentGenerator(
                    campaign_plan=_campaign(root, direction="LONG"),
                    strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                    pair_charts_path=charts,
                    output_path=output,
                    report_path=root / "intents.md",
                    max_loss_jpy=500.0,
                ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())

        usd_jpy_results = [item for item in payload["results"] if item["intent"]["pair"] == "USD_JPY"]
        eur_usd_results = [item for item in payload["results"] if item["intent"]["pair"] == "EUR_USD"]
        watch_issue_codes = {
            issue["code"]
            for item in usd_jpy_results
            for issue in item["risk_issues"]
        }

        self.assertEqual(summary.live_ready, 0)
        self.assertTrue(usd_jpy_results)
        self.assertIn("FORECAST_WATCH_ONLY", watch_issue_codes)
        self.assertTrue(all(item["status"] != "LIVE_READY" for item in usd_jpy_results))
        self.assertTrue(all(item["intent"]["metadata"]["forecast_seed"] for item in usd_jpy_results))
        self.assertTrue(all(item["intent"]["metadata"]["forecast_watch_only"] for item in usd_jpy_results))
        self.assertTrue(
            all(
                not item["intent"]["metadata"]["forecast_seed"]
                and not item["intent"]["metadata"]["forecast_watch_only"]
                for item in eur_usd_results
            )
        )

    def test_range_forecast_seed_uses_range_rotation_floor(self) -> None:
        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"
            forecast = SimpleNamespace(
                direction="RANGE",
                confidence=0.52,
                current_price=1.17326,
                target_price=None,
                invalidation_price=None,
                range_low_price=1.1724,
                range_high_price=1.1748,
                range_width_pips=24.0,
                horizon_min=60,
                rationale_summary="RANGE forecast still supports box rotation",
                drivers_for=("M5 range rail holds",),
                drivers_against=("limited directional extension",),
            )

            with (
                patch(
                    "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                    return_value=forecast,
                ),
                patch(
                    "quant_rabbit.strategy.intent_generator.record_range_vehicle_candidates",
                    return_value=0,
                ),
            ):
                IntentGenerator(
                    campaign_plan=_range_campaign(root, direction="LONG"),
                    strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                    pair_charts_path=_pair_charts_with_direction(
                        root,
                        long_score=0.52,
                        short_score=0.48,
                        dominant_regime="RANGE",
                        m5_regime="RANGE",
                        m5_long_bias=0.58,
                        m5_short_bias=0.22,
                        regime_state="RANGE",
                        adx=17.0,
                        choppiness=65.0,
                    ),
                    output_path=output,
                    report_path=root / "intents.md",
                    max_loss_jpy=500.0,
                ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())

        seed = next(
            item
            for item in payload["results"]
            if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
        )
        seed_issue_codes = {issue["code"] for issue in seed["risk_issues"]}

        self.assertTrue(seed["intent"]["metadata"]["forecast_seed"])
        self.assertEqual(seed["intent"]["metadata"]["forecast_direction"], "RANGE")
        self.assertEqual(seed["intent"]["metadata"]["forecast_confidence"], 0.52)
        self.assertEqual(seed["intent"]["metadata"]["forecast_range_low_price"], 1.1724)
        self.assertEqual(seed["intent"]["metadata"]["forecast_range_high_price"], 1.1748)
        self.assertEqual(seed["intent"]["metadata"]["forecast_range_width_pips"], 24.0)
        self.assertEqual(seed["intent"]["metadata"]["geometry_model"], "RANGE_RAIL_LIMIT")
        self.assertNotIn("FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", seed_issue_codes)

    def test_failed_intent_publication_receipts_forecast_cycle_as_aborted(self) -> None:
        prior_telemetry = os.environ.get("QR_REQUIRE_TELEMETRY_FOR_LIVE")
        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data_root = root / "data"
                output = root / "intents.json"
                forecast = SimpleNamespace(
                    pair="EUR_USD",
                    direction="RANGE",
                    confidence=0.72,
                    current_price=1.17326,
                    target_price=None,
                    invalidation_price=None,
                    range_low_price=1.1724,
                    range_high_price=1.1748,
                    range_width_pips=24.0,
                    horizon_min=60,
                    rationale_summary="RANGE forecast for failed publication test",
                    drivers_for=("M5 range rail holds",),
                    drivers_against=("limited directional extension",),
                )
                with (
                    patch(
                        "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                        return_value=forecast,
                    ),
                    patch(
                        "quant_rabbit.strategy.intent_generator.record_range_vehicle_candidates",
                        side_effect=ValueError("invalid candidate payload"),
                    ),
                    self.assertRaisesRegex(ValueError, "invalid candidate payload"),
                ):
                    IntentGenerator(
                        campaign_plan=_range_campaign(root, direction="LONG"),
                        strategy_profile=_strategy(
                            root,
                            status="CANDIDATE",
                            direction="LONG",
                        ),
                        pair_charts_path=_pair_charts_with_direction(
                            root,
                            long_score=0.52,
                            short_score=0.48,
                            dominant_regime="RANGE",
                            m5_regime="RANGE",
                            m5_long_bias=0.58,
                            m5_short_bias=0.22,
                            regime_state="RANGE",
                            adx=17.0,
                            choppiness=65.0,
                        ),
                        output_path=output,
                        report_path=root / "intents.md",
                        data_root=data_root,
                        max_loss_jpy=500.0,
                    ).run(snapshot_path=_snapshot(root))

                receipts = [
                    json.loads(line)
                    for line in (
                        data_root / "forecast_generation_receipts.jsonl"
                    ).read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                self.assertFalse(output.exists())
                self.assertEqual(receipts[-1]["status"], "ABORTED")
                self.assertFalse(receipts[-1]["learning_eligible"])
                self.assertEqual(receipts[-1]["error_type"], "ValueError")
                self.assertFalse(receipts[-1]["live_permission"])
        finally:
            if prior_telemetry is None:
                os.environ.pop("QR_REQUIRE_TELEMETRY_FOR_LIVE", None)
            else:
                os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = prior_telemetry

    def test_range_target_floor_keeps_send_time_spread_cushion(self) -> None:
        target_pips = _minimum_range_target_pips(stop_pips=10.0, spread_pips=2.0)

        self.assertAlmostEqual(target_pips, 10.0 * RANGE_TARGET_SPREAD_CUSHION_MULT)

    def test_range_forecast_box_keeps_breakout_pending_rotation_blocked(self) -> None:
        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"
            charts = root / "pair_charts.json"
            charts.write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "charts": [
                            {
                                "pair": "EUR_USD",
                                "dominant_regime": "RANGE",
                                "long_score": 0.58,
                                "short_score": 0.42,
                                "confluence": {
                                    "score_balance": "LONG_LEAN",
                                    "score_gap": 0.16,
                                    "dominant_regime": "RANGE",
                                },
                                "views": [
                                    {
                                        "granularity": "M5",
                                        "regime": "RANGE",
                                        "long_bias": 0.58,
                                        "short_bias": 0.42,
                                        "regime_reading": {"state": "TREND_WEAK", "confidence": 0.62},
                                        "family_scores": {"mean_rev_score": 1.2, "trend_score": 0.1, "breakout_score": 0.0},
                                        "indicators": {
                                            "close": 1.1732,
                                            "atr_pips": 8.0,
                                            "bb_lower": 1.1710,
                                            "bb_upper": 1.1760,
                                            "donchian_low": 1.1707,
                                            "donchian_high": 1.1764,
                                            "swing_low": 1.1705,
                                            "swing_high": 1.1767,
                                            "linreg_channel_lower": 1.1709,
                                            "linreg_channel_upper": 1.1761,
                                            "adx_14": 17.0,
                                            "choppiness_14": 58.0,
                                        },
                                    },
                                    {
                                        "granularity": "M15",
                                        "regime": "RANGE",
                                        "regime_reading": {"state": "TRANSITION", "confidence": 0.52},
                                        "family_scores": {"mean_rev_score": 1.0, "trend_score": 0.0, "breakout_score": 0.0},
                                        "indicators": {
                                            "close": 1.1732,
                                            "atr_pips": 11.0,
                                            "bb_lower": 1.1708,
                                            "bb_upper": 1.1763,
                                            "donchian_low": 1.1705,
                                            "donchian_high": 1.1766,
                                            "adx_14": 18.0,
                                            "choppiness_14": 59.0,
                                        },
                                    },
                                    {
                                        "granularity": "M30",
                                        "regime": "UNCLEAR",
                                        "regime_reading": {"state": "BREAKOUT_PENDING", "confidence": 0.61},
                                        "family_scores": {"mean_rev_score": 0.7, "trend_score": 0.2, "breakout_score": 0.3},
                                        "indicators": {
                                            "close": 1.1732,
                                            "atr_pips": 15.0,
                                            "bb_lower": 1.1700,
                                            "bb_upper": 1.1770,
                                            "donchian_low": 1.1698,
                                            "donchian_high": 1.1772,
                                            "adx_14": 21.0,
                                            "choppiness_14": 42.0,
                                        },
                                    },
                                ],
                            }
                        ]
                    }
                )
            )
            forecast = SimpleNamespace(
                direction="RANGE",
                confidence=0.56,
                raw_confidence=0.76,
                current_price=1.17326,
                target_price=None,
                invalidation_price=None,
                range_low_price=1.1710,
                range_high_price=1.1760,
                range_width_pips=50.0,
                horizon_min=120,
                rationale_summary="RANGE forecast has a measurable box, but higher timeframe is pending",
                drivers_for=("M5/M15 rails still define the box",),
                drivers_against=("M30 breakout pending",),
            )

            with (
                patch("quant_rabbit.strategy.intent_generator._forecast_seed_for_pair", return_value=forecast),
                patch("quant_rabbit.strategy.intent_generator._record_forecast_seed_telemetry", return_value=None),
            ):
                IntentGenerator(
                    campaign_plan=_campaign(root),
                    strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                    pair_charts_path=charts,
                    output_path=output,
                    report_path=root / "intents.md",
                    max_loss_jpy=500.0,
                ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())

        seed = next(
            item
            for item in payload["results"]
            if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
        )
        metadata = seed["intent"]["metadata"]
        issue_codes = {issue["code"] for issue in seed["risk_issues"]}

        self.assertTrue(metadata["forecast_seed"])
        self.assertTrue(metadata["forecast_watch_only"])
        self.assertIn("current range-rotation edge is not confirmed", metadata["forecast_watch_only_reason"])
        self.assertEqual(metadata["forecast_direction"], "RANGE")
        self.assertEqual(metadata["forecast_range_low_price"], 1.1710)
        self.assertEqual(metadata["forecast_range_high_price"], 1.1760)
        self.assertEqual(metadata["geometry_model"], "RANGE_RAIL_LIMIT")
        self.assertIn("FORECAST_WATCH_ONLY", issue_codes)
        self.assertIn("RANGE_PHASE_NOT_ROTATION", issue_codes)

    def test_range_rail_watch_only_override_requires_current_scoped_matrix(self) -> None:
        generated_at = datetime.now(timezone.utc)
        metadata = {
            "forecast_direction": "RANGE",
            "forecast_confidence": 0.56,
            "forecast_range_low_price": 1.1724,
            "forecast_range_high_price": 1.1748,
            "geometry_model": "RANGE_RAIL_LIMIT",
            "range_entry_side": "support",
            "range_tp_is_inside_box": True,
            "range_sl_outside_box": True,
            "market_context_matrix_ref": "matrix:EUR_USD:LONG",
            "market_context_matrix_pair": "EUR_USD",
            "market_context_matrix_side": "LONG",
            "market_context_matrix_generated_at_utc": generated_at.isoformat(),
            "matrix_support_count": 1,
            "matrix_reject_count": 2,
        }

        self.assertFalse(
            _range_rail_limit_watch_only_metadata_can_trade(
                pair="EUR_USD",
                side=Side.LONG,
                order_type=OrderType.LIMIT,
                metadata=metadata,
                validation_time_utc=generated_at,
            )
        )
        self.assertTrue(
            _range_rail_limit_watch_only_metadata_can_trade(
                pair="EUR_USD",
                side=Side.LONG,
                order_type=OrderType.LIMIT,
                metadata={**metadata, "matrix_reject_count": 1},
                validation_time_utc=generated_at,
            )
        )
        missing_binding = {
            key: value
            for key, value in metadata.items()
            if not key.startswith("market_context_matrix_")
        }
        malformed_count = {**metadata, "matrix_support_count": "1"}
        stale = {
            **metadata,
            "market_context_matrix_generated_at_utc": (
                generated_at
                - timedelta(seconds=RANGE_WATCH_MATRIX_MAX_AGE_SECONDS + 1)
            ).isoformat(),
        }
        unscoped = {**metadata, "market_context_matrix_ref": "matrix:GBP_USD:LONG"}
        missing_count = {
            key: value
            for key, value in metadata.items()
            if key != "matrix_support_count"
        }
        malformed_side = {**metadata, "market_context_matrix_side": "long"}
        for candidate in (
            missing_binding,
            missing_count,
            malformed_count,
            malformed_side,
            stale,
            unscoped,
        ):
            with self.subTest(candidate=candidate):
                self.assertFalse(
                    _range_rail_limit_watch_only_metadata_can_trade(
                        pair="EUR_USD",
                        side=Side.LONG,
                        order_type=OrderType.LIMIT,
                        metadata=candidate,
                        validation_time_utc=generated_at,
                    )
                )
        self.assertTrue(
            _range_rail_limit_watch_only_metadata_can_trade(
                pair="EUR_USD",
                side=Side.LONG,
                order_type=OrderType.LIMIT,
                metadata={**metadata, "matrix_reject_count": 1},
                # Production refresh writes the matrix before refreshing the
                # broker snapshot; the reverse order is also valid on reuse.
                validation_time_utc=generated_at - timedelta(seconds=30),
            )
        )

    def test_range_forecast_box_watch_only_override_obeys_matrix_balance(self) -> None:
        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"
            charts = root / "pair_charts.json"
            matrix = _market_context_matrix(root)
            charts.write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "charts": [
                            {
                                "pair": "EUR_USD",
                                "dominant_regime": "UNCLEAR",
                                "long_score": 0.54,
                                "short_score": 0.46,
                                "confluence": {
                                    "score_balance": "LONG_LEAN",
                                    "score_gap": 0.08,
                                    "dominant_regime": "UNCLEAR",
                                },
                                "views": [
                                    {
                                        "granularity": "M5",
                                        "regime": "TREND_WEAK",
                                        "long_bias": 0.54,
                                        "short_bias": 0.46,
                                        "regime_reading": {"state": "TREND_WEAK", "confidence": 0.6},
                                        "family_scores": {
                                            "mean_rev_score": 0.8,
                                            "trend_score": 0.2,
                                            "breakout_score": 0.0,
                                        },
                                        "indicators": {
                                            "close": 1.17326,
                                            "atr_pips": 8.0,
                                            "adx_14": 22.0,
                                            "choppiness_14": 49.0,
                                        },
                                    }
                                ],
                            }
                        ]
                    }
                )
            )
            forecast = SimpleNamespace(
                direction="RANGE",
                confidence=0.56,
                raw_confidence=0.74,
                current_price=1.17326,
                target_price=None,
                invalidation_price=None,
                range_low_price=1.1724,
                range_high_price=1.1748,
                range_width_pips=24.0,
                horizon_min=120,
                rationale_summary="RANGE forecast box survives but chart rail keys are missing",
                drivers_for=("forecast range low/high define executable rails",),
                drivers_against=("chart packet lacks explicit M5 support/resistance keys",),
            )

            with (
                patch("quant_rabbit.strategy.intent_generator._forecast_seed_for_pair", return_value=forecast),
                patch("quant_rabbit.strategy.intent_generator._record_forecast_seed_telemetry", return_value=None),
                patch(
                    "quant_rabbit.strategy.intent_generator.record_range_vehicle_candidates",
                    return_value=0,
                ),
            ):
                IntentGenerator(
                    campaign_plan=_campaign(root),
                    strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                    pair_charts_path=charts,
                    market_context_matrix_path=matrix,
                    output_path=output,
                    report_path=root / "intents.md",
                    max_loss_jpy=500.0,
                ).run(snapshot_path=_snapshot(root))

                supported_payload = json.loads(output.read_text())
                matrix_payload = json.loads(matrix.read_text())
                long_matrix = matrix_payload["pairs"]["EUR_USD"]["LONG"]
                long_matrix["support_count"] = 1
                long_matrix["reject_count"] = 2
                long_matrix["strongest_reject"] = "directional counterevidence dominates the rail"
                matrix.write_text(json.dumps(matrix_payload))
                IntentGenerator(
                    campaign_plan=_campaign(root),
                    strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                    pair_charts_path=charts,
                    market_context_matrix_path=matrix,
                    output_path=output,
                    report_path=root / "intents.md",
                    max_loss_jpy=500.0,
                ).run(snapshot_path=_snapshot(root))
                rejected_payload = json.loads(output.read_text())

        supported_seed = next(
            item
            for item in supported_payload["results"]
            if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
        )
        supported_metadata = supported_seed["intent"]["metadata"]
        supported_issue_codes = {issue["code"] for issue in supported_seed["risk_issues"]}

        self.assertTrue(supported_metadata["forecast_seed"])
        self.assertTrue(supported_metadata["forecast_watch_only"])
        self.assertEqual(supported_metadata["forecast_direction"], "RANGE")
        self.assertEqual(supported_metadata["geometry_model"], "RANGE_RAIL_LIMIT")
        self.assertEqual(supported_metadata["range_indicator_source"], "forecast_range_box")
        self.assertEqual(supported_metadata["range_support"], 1.1724)
        self.assertEqual(supported_metadata["range_resistance"], 1.1748)
        self.assertEqual(supported_metadata["range_entry_side"], "support")
        self.assertTrue(supported_metadata["range_tp_is_inside_box"])
        self.assertTrue(supported_metadata["range_sl_outside_box"])
        self.assertEqual(supported_metadata["matrix_support_count"], 4)
        self.assertEqual(supported_metadata["matrix_reject_count"], 1)
        self.assertTrue(supported_metadata["forecast_watch_only_live_override"])
        self.assertIn("Range rail override", supported_metadata["required_receipt"])
        self.assertIn(
            "range rail override",
            supported_seed["intent"]["market_context"]["event_risk"],
        )
        self.assertNotIn("FORECAST_WATCH_ONLY", supported_issue_codes)
        self.assertEqual(supported_seed["status"], "LIVE_READY")

        rejected_seed = next(
            item
            for item in rejected_payload["results"]
            if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
        )
        rejected_metadata = rejected_seed["intent"]["metadata"]
        rejected_issue_codes = {issue["code"] for issue in rejected_seed["risk_issues"]}

        self.assertEqual(rejected_metadata["matrix_support_count"], 1)
        self.assertEqual(rejected_metadata["matrix_reject_count"], 2)
        self.assertFalse(rejected_metadata["forecast_watch_only_live_override"])
        self.assertNotIn("Range rail override", rejected_metadata["required_receipt"])
        self.assertIn("FORECAST_WATCH_ONLY", rejected_issue_codes)
        self.assertNotEqual(rejected_seed["status"], "LIVE_READY")

    def test_range_forecast_box_adds_existing_opposite_side_rotation_counterpart(self) -> None:
        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"
            charts = root / "pair_charts.json"
            charts.write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "charts": [
                            {
                                "pair": "EUR_USD",
                                "dominant_regime": "UNCLEAR",
                                "long_score": 0.50,
                                "short_score": 0.50,
                                "views": [
                                    {
                                        "granularity": "M5",
                                        "regime": "TREND_WEAK",
                                        "regime_reading": {"state": "TREND_WEAK", "confidence": 0.6},
                                        "family_scores": {"mean_rev_score": 0.8, "trend_score": 0.1},
                                        "indicators": {
                                            "close": 1.17326,
                                            "atr_pips": 8.0,
                                            "adx_14": 21.0,
                                            "choppiness_14": 52.0,
                                        },
                                    }
                                ],
                            }
                        ]
                    }
                )
            )
            forecast = SimpleNamespace(
                direction="RANGE",
                confidence=0.72,
                raw_confidence=0.80,
                current_price=1.17326,
                target_price=None,
                invalidation_price=None,
                range_low_price=1.1710,
                range_high_price=1.1740,
                range_width_pips=30.0,
                horizon_min=60,
                rationale_summary="measured RANGE box supports rail rotation",
                drivers_for=("forecast range box defines both support and resistance rails",),
                drivers_against=("current price is in upper half, so immediate side is SHORT",),
            )

            with (
                patch("quant_rabbit.strategy.intent_generator._forecast_seed_for_pair", return_value=forecast),
                patch("quant_rabbit.strategy.intent_generator._record_forecast_seed_telemetry", return_value=None),
            ):
                IntentGenerator(
                    campaign_plan=_campaign(root, direction="LONG"),
                    strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                    pair_charts_path=charts,
                    output_path=output,
                    report_path=root / "intents.md",
                    max_loss_jpy=500.0,
                ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())

        lane_ids = {item["lane_id"] for item in payload["results"]}
        long_seed = next(
            item
            for item in payload["results"]
            if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
        )
        short_seed = next(
            item
            for item in payload["results"]
            if item["lane_id"] == "range_trader:EUR_USD:SHORT:RANGE_ROTATION"
        )

        self.assertIn("range_trader:EUR_USD:LONG:RANGE_ROTATION", lane_ids)
        self.assertIn("range_trader:EUR_USD:SHORT:RANGE_ROTATION", lane_ids)
        self.assertEqual(long_seed["intent"]["metadata"]["range_entry_side"], "support")
        self.assertEqual(short_seed["intent"]["metadata"]["range_entry_side"], "resistance")
        self.assertEqual(long_seed["intent"]["metadata"]["range_indicator_source"], "forecast_range_box")

    def test_range_forecast_non_rotation_lane_is_dry_run_blocked(self) -> None:
        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        forecast = SimpleNamespace(
            pair="EUR_USD",
            direction="RANGE",
            confidence=0.72,
            raw_confidence=0.78,
            current_price=1.17326,
            target_price=None,
            invalidation_price=None,
            range_low_price=1.1724,
            range_high_price=1.1748,
            horizon_min=60,
            rationale_summary="two-way range, not directional continuation",
            drivers_for=("M5 box still active",),
            drivers_against=("no breakout confirmation",),
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"
            with (
                patch(
                    "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                    return_value=forecast,
                ),
                patch(
                    "quant_rabbit.strategy.intent_generator.record_range_vehicle_candidates",
                    return_value=0,
                ),
            ):
                IntentGenerator(
                    campaign_plan=_campaign(root, direction="LONG"),
                    strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                    pair_charts_path=_pair_charts_with_direction(
                        root,
                        long_score=0.54,
                        short_score=0.46,
                        dominant_regime="RANGE",
                        m5_regime="RANGE",
                    ),
                    output_path=output,
                    report_path=root / "intents.md",
                    max_loss_jpy=500.0,
                ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())

        trend = next(
            item
            for item in payload["results"]
            if item["lane_id"] == "trend_trader:EUR_USD:LONG:TREND_CONTINUATION"
        )
        issue_codes = {issue["code"] for issue in trend["risk_issues"]}

        self.assertEqual(trend["status"], "DRY_RUN_BLOCKED")
        self.assertFalse(trend["risk_allowed"])
        self.assertIn("RANGE_FORECAST_REQUIRES_RANGE_ROTATION", issue_codes)

    def test_weak_range_forecast_non_rotation_lane_is_dry_run_blocked(self) -> None:
        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        forecast = SimpleNamespace(
            pair="EUR_USD",
            direction="RANGE",
            confidence=0.31,
            raw_confidence=0.78,
            current_price=1.17326,
            target_price=None,
            invalidation_price=None,
            range_low_price=1.1724,
            range_high_price=1.1748,
            horizon_min=60,
            rationale_summary="weak two-way range, not directional continuation",
            drivers_for=("M5 box still active",),
            drivers_against=("no breakout confirmation",),
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"
            with patch(
                "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                return_value=forecast,
            ):
                IntentGenerator(
                    campaign_plan=_campaign(root, direction="LONG"),
                    strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                    pair_charts_path=_pair_charts_with_direction(
                        root,
                        long_score=0.54,
                        short_score=0.46,
                        dominant_regime="RANGE",
                        m5_regime="RANGE",
                    ),
                    output_path=output,
                    report_path=root / "intents.md",
                    max_loss_jpy=500.0,
                ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())

        trend = next(
            item
            for item in payload["results"]
            if item["lane_id"] == "trend_trader:EUR_USD:LONG:TREND_CONTINUATION"
        )
        issue_codes = {issue["code"] for issue in trend["risk_issues"]}

        self.assertEqual(trend["status"], "DRY_RUN_BLOCKED")
        self.assertFalse(trend["risk_allowed"])
        self.assertIn("RANGE_FORECAST_REQUIRES_RANGE_ROTATION", issue_codes)

    def test_reversal_recovery_hedge_uses_recovery_forecast_floor_for_live_context(self) -> None:
        from quant_rabbit.models import BrokerSnapshot, MarketContext, OrderIntent, OrderType, Quote, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import (
            _forecast_live_readiness_issue,
            _method_context_issues,
            _telemetry_live_readiness_issues,
        )

        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        recovery_metadata = {
            "position_intent": "HEDGE",
            "hedge_recovery": True,
            "hedge_timing_class": "REVERSAL",
            "forecast_direction": "UP",
            "forecast_confidence": 0.51,
            "forecast_target_price": 1.1762,
            "forecast_invalidation_price": 1.1718,
        }
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.STOP_ENTRY,
            units=5000,
            entry=1.1734,
            tp=1.1762,
            sl=1.1718,
            thesis="reversal recovery hedge",
            market_context=MarketContext(
                regime="UNCLEAR current; TREND_CONTINUATION campaign lane",
                narrative="",
                chart_story="",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="",
            ),
            metadata=recovery_metadata,
        )

        fresh_entry_issue = _forecast_live_readiness_issue(
            OrderIntent(
                pair="EUR_USD",
                side=Side.LONG,
                order_type=OrderType.STOP_ENTRY,
                units=5000,
                entry=1.1734,
                tp=1.1762,
                sl=1.1718,
                thesis="fresh entry",
                market_context=intent.market_context,
                metadata={**recovery_metadata, "position_intent": "NEW", "hedge_recovery": False},
            ),
            {**recovery_metadata, "position_intent": "NEW", "hedge_recovery": False},
            TradeMethod.TREND_CONTINUATION,
        )

        self.assertIsNone(_forecast_live_readiness_issue(intent, recovery_metadata, TradeMethod.TREND_CONTINUATION))
        self.assertEqual(fresh_entry_issue["code"], "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE")

        range_recovery_metadata = {
            **recovery_metadata,
            "forecast_direction": "RANGE",
            "forecast_confidence": 0.82,
        }
        self.assertIsNone(
            _forecast_live_readiness_issue(intent, range_recovery_metadata, TradeMethod.TREND_CONTINUATION)
        )
        range_fresh_issue = _forecast_live_readiness_issue(
            OrderIntent(
                pair="EUR_USD",
                side=Side.LONG,
                order_type=OrderType.MARKET,
                units=5000,
                entry=1.1734,
                tp=1.1762,
                sl=1.1718,
                thesis="fresh range trend entry",
                market_context=intent.market_context,
                metadata={**range_recovery_metadata, "position_intent": "NEW", "hedge_recovery": False},
            ),
            {**range_recovery_metadata, "position_intent": "NEW", "hedge_recovery": False},
            TradeMethod.TREND_CONTINUATION,
        )
        self.assertEqual(range_fresh_issue["code"], "RANGE_FORECAST_REQUIRES_RANGE_ROTATION")
        self.assertEqual(range_fresh_issue["severity"], "BLOCK")

        chart_reversal_metadata = {
            "position_intent": "HEDGE",
            "hedge_recovery": True,
            "hedge_timing_class": "REVERSAL",
            "chart_score_balance": "LONG_LEAN",
            "chart_score_gap": 0.183,
            "pattern_reversal_dominant_side": "LONG",
            "pattern_reversal_weight_long": 44.25,
            "pattern_reversal_weight_short": 30.42,
            "trend_timeframes": ["M1:TREND_UP", "M5:TREND_UP"],
        }
        market_recovery = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=5000,
            entry=1.1734,
            tp=1.1762,
            sl=1.1718,
            thesis="chart-confirmed reversal recovery hedge",
            market_context=intent.market_context,
            metadata=chart_reversal_metadata,
        )
        self.assertIsNone(
            _forecast_live_readiness_issue(market_recovery, chart_reversal_metadata, TradeMethod.TREND_CONTINUATION)
        )

        now = datetime.now(timezone.utc)
        with patch(
            "quant_rabbit.strategy.intent_generator._latest_forecast_history_for_pair",
            return_value={
                "timestamp_utc": now.isoformat().replace("+00:00", "Z"),
                "direction": "UNCLEAR",
                "confidence": 0.18,
                "cycle_id": "cycle",
            },
        ), patch(
            "quant_rabbit.strategy.intent_generator._expired_pending_projection_count",
            return_value=0,
        ), patch(
            "quant_rabbit.strategy.intent_generator._execution_ledger_sync_live_issue",
            return_value=None,
        ):
            telemetry_codes = {
                issue["code"]
                for issue in _telemetry_live_readiness_issues(
                    market_recovery,
                    chart_reversal_metadata,
                    BrokerSnapshot(
                        fetched_at_utc=now,
                        quotes={"EUR_USD": Quote("EUR_USD", 1.1733, 1.1735, now)},
                    ),
                    now,
                )
            }
        self.assertNotIn("TELEMETRY_FORECAST_CONTEXT_REQUIRED_FOR_LIVE", telemetry_codes)

        weak_chart_reversal = {
            **chart_reversal_metadata,
            "pattern_reversal_dominant_side": "SHORT",
            "pattern_reversal_weight_long": 33.75,
            "pattern_reversal_weight_short": 36.08,
        }
        weak_issue = _forecast_live_readiness_issue(
            OrderIntent(
                pair="EUR_USD",
                side=Side.LONG,
                order_type=OrderType.MARKET,
                units=5000,
                entry=1.1734,
                tp=1.1762,
                sl=1.1718,
                thesis="weak recovery hedge",
                market_context=intent.market_context,
                metadata=weak_chart_reversal,
            ),
            weak_chart_reversal,
            TradeMethod.TREND_CONTINUATION,
        )
        self.assertEqual(weak_issue["code"], "FORECAST_CONTEXT_REQUIRED_FOR_LIVE")

        tolerated_confidence_metadata = {
            **chart_reversal_metadata,
            "forecast_direction": "UP",
            "forecast_confidence": 0.6133,
            "forecast_cycle_id": "cycle",
        }
        tolerated_confidence_intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=5000,
            entry=1.1734,
            tp=1.1762,
            sl=1.1718,
            thesis="rounded confidence recovery hedge",
            market_context=intent.market_context,
            metadata=tolerated_confidence_metadata,
        )

        def telemetry_codes_for_latest_confidence(latest_confidence: float) -> set[str]:
            with patch(
                "quant_rabbit.strategy.intent_generator._latest_forecast_history_for_pair",
                return_value={
                    "timestamp_utc": now.isoformat().replace("+00:00", "Z"),
                    "direction": "UP",
                    "confidence": latest_confidence,
                    "cycle_id": "cycle",
                },
            ), patch(
                "quant_rabbit.strategy.intent_generator._directional_projection_recorded",
                return_value=True,
            ), patch(
                "quant_rabbit.strategy.intent_generator._expired_pending_projection_count",
                return_value=0,
            ), patch(
                "quant_rabbit.strategy.intent_generator._execution_ledger_sync_live_issue",
                return_value=None,
            ):
                return {
                    issue["code"]
                    for issue in _telemetry_live_readiness_issues(
                        tolerated_confidence_intent,
                        tolerated_confidence_metadata,
                        BrokerSnapshot(
                            fetched_at_utc=now,
                            quotes={"EUR_USD": Quote("EUR_USD", 1.1733, 1.1735, now)},
                        ),
                        now,
                    )
                }

        self.assertNotIn(
            "TELEMETRY_FORECAST_HISTORY_MISMATCH_FOR_LIVE",
            telemetry_codes_for_latest_confidence(0.6134),
        )
        self.assertIn(
            "TELEMETRY_FORECAST_HISTORY_MISMATCH_FOR_LIVE",
            telemetry_codes_for_latest_confidence(0.616),
        )

        opposed = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.STOP_ENTRY,
            units=5000,
            entry=1.1730,
            tp=1.1700,
            sl=1.1750,
            thesis="opposed reversal recovery hedge",
            market_context=intent.market_context,
            metadata=recovery_metadata,
        )
        opposed_codes = {issue["code"] for issue in _method_context_issues(opposed)}
        self.assertIn("FORECAST_DIRECTION_CONFLICT", opposed_codes)

    def test_weak_range_forecast_box_seeds_range_rotation_watch_lane(self) -> None:
        forecast = SimpleNamespace(
            direction="RANGE",
            confidence=0.30,
            raw_confidence=0.73,
            calibration_multiplier=0.41,
            current_price=1.17326,
            target_price=None,
            invalidation_price=None,
            range_low_price=1.1710,
            range_high_price=1.1760,
            range_width_pips=50.0,
            horizon_min=120,
            rationale_summary="weak calibrated directional forecast inside measured range box",
            drivers_for=("range forming",),
            drivers_against=("directional bucket weak",),
            component_scores={"RANGE": 10.0, "UP": 8.0, "DOWN": 12.0},
            market_support={"ok": False, "direction": "RANGE", "reason": "forecast RANGE has no executable direction"},
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"
            with patch(
                "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                return_value=forecast,
            ):
                IntentGenerator(
                    campaign_plan=_campaign(root, direction="LONG"),
                    strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                    output_path=output,
                    report_path=root / "intents.md",
                    pair_charts_path=_pair_charts_with_direction(
                        root,
                        long_score=0.54,
                        short_score=0.46,
                        dominant_regime="TREND_UP",
                        m5_regime="TREND_UP",
                        adx=34.0,
                        choppiness=42.0,
                    ),
                    max_loss_jpy=500.0,
                ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())

        range_lane = next(
            item
            for item in payload["results"]
            if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
        )
        metadata = range_lane["intent"]["metadata"]

        self.assertTrue(metadata["forecast_seed"])
        self.assertTrue(metadata["forecast_watch_only"])
        self.assertEqual(metadata["forecast_direction"], "RANGE")
        self.assertEqual(metadata["forecast_range_low_price"], 1.1710)
        self.assertEqual(metadata["forecast_range_high_price"], 1.1760)
        self.assertIn("watch-only forecast candidate", range_lane["intent"]["market_context"]["event_risk"])

    def test_same_forecast_from_later_cycle_does_not_stale_current_intent(self) -> None:
        from quant_rabbit.models import BrokerSnapshot, MarketContext, OrderIntent, OrderType, Quote, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _telemetry_live_readiness_issues

        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        now = datetime.now(timezone.utc)
        metadata = {
            "forecast_direction": "UP",
            "forecast_confidence": 0.6133,
            "forecast_cycle_id": "pre-entry-cycle",
        }
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=5000,
            entry=1.1734,
            tp=1.1762,
            sl=1.1718,
            thesis="current-cycle intent",
            market_context=MarketContext(
                regime="TREND_UP",
                narrative="",
                chart_story="",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="",
            ),
            metadata=metadata,
        )
        current_cycle = {
            "timestamp_utc": now.isoformat().replace("+00:00", "Z"),
            "direction": "UP",
            "confidence": 0.6133,
            "cycle_id": "pre-entry-cycle",
        }
        later_same_forecast = {
            "timestamp_utc": (now + timedelta(seconds=10)).isoformat().replace("+00:00", "Z"),
            "direction": "UP",
            "confidence": 0.6134,
            "cycle_id": "position-forecast-cycle",
        }

        with patch(
            "quant_rabbit.strategy.intent_generator._forecast_history_for_pair_cycle",
            return_value=current_cycle,
        ), patch(
            "quant_rabbit.strategy.intent_generator._latest_forecast_history_for_pair",
            return_value=later_same_forecast,
        ), patch(
            "quant_rabbit.strategy.intent_generator._directional_projection_recorded",
            return_value=True,
        ), patch(
            "quant_rabbit.strategy.intent_generator._expired_pending_projection_count",
            return_value=0,
        ), patch(
            "quant_rabbit.strategy.intent_generator._execution_ledger_sync_live_issue",
            return_value=None,
        ):
            codes = {
                issue["code"]
                for issue in _telemetry_live_readiness_issues(
                    intent,
                    metadata,
                    BrokerSnapshot(
                        fetched_at_utc=now,
                        quotes={"EUR_USD": Quote("EUR_USD", 1.1733, 1.1735, now)},
                    ),
                    now,
                )
            }

        self.assertNotIn("TELEMETRY_FORECAST_HISTORY_STALE_FOR_LIVE", codes)
        self.assertNotIn("TELEMETRY_FORECAST_HISTORY_MISMATCH_FOR_LIVE", codes)

    def test_learning_scout_telemetry_accepts_same_cycle_unclear_source(self) -> None:
        from quant_rabbit.models import BrokerSnapshot, MarketContext, OrderIntent, OrderType, Quote, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _telemetry_live_readiness_issues

        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        now = datetime(2026, 7, 15, 17, 30, tzinfo=timezone.utc)
        metadata = {
            "predictive_scout_source": "FORECAST_ORIENTATION_LEARNING",
            "forecast_direction": "UNCLEAR",
            "forecast_confidence": 0.0,
            "forecast_cycle_id": "contradiction-cycle",
        }
        intent = OrderIntent(
            pair="CAD_JPY",
            side=Side.LONG,
            order_type=OrderType.LIMIT,
            units=313,
            entry=117.10,
            tp=117.20,
            sl=117.00,
            thesis="bounded contradiction learning scout",
            market_context=MarketContext(
                regime="TREND_UP",
                narrative="",
                chart_story="",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="",
            ),
            metadata=metadata,
        )
        source_row = {
            "timestamp_utc": now.isoformat().replace("+00:00", "Z"),
            "direction": "UNCLEAR",
            "confidence": 0.0,
            "cycle_id": "contradiction-cycle",
        }

        with patch(
            "quant_rabbit.strategy.intent_generator."
            "_predictive_scout_forward_evidence_allowed",
            return_value=True,
        ), patch(
            "quant_rabbit.strategy.intent_generator._forecast_history_for_pair_cycle",
            return_value=source_row,
        ), patch(
            "quant_rabbit.strategy.intent_generator._latest_forecast_history_for_pair",
            return_value=source_row,
        ), patch(
            "quant_rabbit.strategy.intent_generator._expired_pending_projection_count",
            return_value=0,
        ), patch(
            "quant_rabbit.strategy.intent_generator._execution_ledger_sync_live_issue",
            return_value=None,
        ):
            codes = {
                issue["code"]
                for issue in _telemetry_live_readiness_issues(
                    intent,
                    metadata,
                    BrokerSnapshot(
                        fetched_at_utc=now,
                        quotes={"CAD_JPY": Quote("CAD_JPY", 117.10, 117.12, now)},
                    ),
                    now,
                )
            }

        self.assertNotIn("TELEMETRY_FORECAST_NOT_EXECUTABLE_FOR_LIVE", codes)
        self.assertNotIn("TELEMETRY_FORECAST_HISTORY_MISMATCH_FOR_LIVE", codes)

    def test_stale_quote_skips_forecast_history_mismatch_checks(self) -> None:
        from quant_rabbit.models import BrokerSnapshot, MarketContext, OrderIntent, OrderType, Quote, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _telemetry_live_readiness_issues

        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        now = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)
        metadata = {
            "forecast_direction": "UP",
            "forecast_confidence": 0.72,
            "forecast_cycle_id": "fresh-cycle",
        }
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=5000,
            entry=1.1734,
            tp=1.1762,
            sl=1.1718,
            thesis="stale quote intent",
            market_context=MarketContext(
                regime="TREND_UP",
                narrative="",
                chart_story="",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="",
            ),
            metadata=metadata,
        )

        with patch(
            "quant_rabbit.strategy.intent_generator._latest_forecast_history_for_pair",
            side_effect=AssertionError("stale quote should not scan latest forecast_history"),
        ), patch(
            "quant_rabbit.strategy.intent_generator._forecast_history_for_pair_cycle",
            side_effect=AssertionError("stale quote should not scan cycle forecast_history"),
        ), patch(
            "quant_rabbit.strategy.intent_generator._expired_pending_projection_count",
            return_value=0,
        ), patch(
            "quant_rabbit.strategy.intent_generator._execution_ledger_sync_live_issue",
            return_value=None,
        ):
            codes = {
                issue["code"]
                for issue in _telemetry_live_readiness_issues(
                    intent,
                    metadata,
                    BrokerSnapshot(
                        fetched_at_utc=now,
                        quotes={
                            "EUR_USD": Quote(
                                "EUR_USD",
                                1.1733,
                                1.1735,
                                now - timedelta(seconds=120),
                            )
                        },
                    ),
                    now,
                )
            }

        self.assertIn("TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE", codes)
        self.assertNotIn("TELEMETRY_FORECAST_HISTORY_STALE_FOR_LIVE", codes)
        self.assertNotIn("TELEMETRY_FORECAST_HISTORY_MISMATCH_FOR_LIVE", codes)

    def test_telemetry_cache_supplies_forecast_and_projection_checks(self) -> None:
        from quant_rabbit.models import BrokerSnapshot, MarketContext, OrderIntent, OrderType, Quote, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import (
            _build_telemetry_live_readiness_cache,
            _telemetry_live_readiness_issues,
        )

        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        now = datetime(2026, 6, 5, 2, 40, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data"
            data_root.mkdir()
            current_ts = now.isoformat().replace("+00:00", "Z")
            expired_ts = (now - timedelta(minutes=90)).isoformat().replace("+00:00", "Z")
            (data_root / "forecast_history.jsonl").write_text(
                json.dumps(
                    {
                        "timestamp_utc": current_ts,
                        "cycle_id": "cycle-1",
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "confidence": 0.6133,
                    }
                )
                + "\n"
            )
            (data_root / "projection_ledger.jsonl").write_text(
                json.dumps(
                    {
                        "timestamp_emitted_utc": current_ts,
                        "pair": "EUR_USD",
                        "signal_name": "directional_forecast",
                        "direction": "UP",
                        "resolution_window_min": 120,
                        "resolution_status": "PENDING",
                        "cycle_id": "cycle-1",
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "timestamp_emitted_utc": expired_ts,
                        "pair": "GBP_USD",
                        "signal_name": "directional_forecast",
                        "direction": "DOWN",
                        "resolution_window_min": 30,
                        "resolution_status": "PENDING",
                        "cycle_id": "expired-cycle",
                    }
                )
                + "\n"
            )

            cache = _build_telemetry_live_readiness_cache(
                data_root=data_root,
                validation_time_utc=now,
            )

        self.assertIn(("EUR_USD", "cycle-1"), cache.directional_projection_keys)
        self.assertIn(("EUR_USD", "cycle-1", "directional_forecast"), cache.projection_signal_keys)
        self.assertEqual(cache.expired_pending_projection_count, 1)
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=5000,
            entry=1.1734,
            tp=1.1762,
            sl=1.1718,
            thesis="cached telemetry intent",
            market_context=MarketContext(
                regime="TREND_UP",
                narrative="",
                chart_story="",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="",
            ),
            metadata={
                "forecast_direction": "UP",
                "forecast_confidence": 0.6133,
                "forecast_cycle_id": "cycle-1",
            },
        )

        with patch(
            "quant_rabbit.strategy.intent_generator._latest_forecast_history_for_pair",
            side_effect=AssertionError("cache should avoid forecast_history latest scan"),
        ), patch(
            "quant_rabbit.strategy.intent_generator._forecast_history_for_pair_cycle",
            side_effect=AssertionError("cache should avoid forecast_history cycle scan"),
        ), patch(
            "quant_rabbit.strategy.intent_generator._directional_projection_recorded",
            side_effect=AssertionError("cache should avoid projection_ledger directional scan"),
        ), patch(
            "quant_rabbit.strategy.intent_generator._expired_pending_projection_count",
            side_effect=AssertionError("cache should avoid projection_ledger expiry scan"),
        ), patch(
            "quant_rabbit.strategy.intent_generator._execution_ledger_sync_live_issue",
            return_value=None,
        ):
            codes = {
                issue["code"]
                for issue in _telemetry_live_readiness_issues(
                    intent,
                    intent.metadata,
                    BrokerSnapshot(
                        fetched_at_utc=now,
                        quotes={"EUR_USD": Quote("EUR_USD", 1.1733, 1.1735, now)},
                    ),
                    now,
                    cache=cache,
                )
            }

        self.assertNotIn("TELEMETRY_FORECAST_HISTORY_REQUIRED_FOR_LIVE", codes)
        self.assertNotIn("TELEMETRY_DIRECTIONAL_PROJECTION_REQUIRED_FOR_LIVE", codes)
        self.assertIn("TELEMETRY_PROJECTION_PENDING_EXPIRED_FOR_LIVE", codes)

    def test_market_support_projection_must_be_same_cycle_telemetry(self) -> None:
        from quant_rabbit.models import BrokerSnapshot, MarketContext, OrderIntent, OrderType, Quote, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import (
            _build_telemetry_live_readiness_cache,
            _telemetry_live_readiness_issues,
        )

        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        now = datetime(2026, 6, 5, 3, 10, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data"
            data_root.mkdir()
            ts = now.isoformat().replace("+00:00", "Z")
            (data_root / "forecast_history.jsonl").write_text(
                json.dumps(
                    {
                        "timestamp_utc": ts,
                        "cycle_id": "cycle-2",
                        "pair": "EUR_USD",
                        "direction": "UP",
                        "confidence": 0.49,
                    }
                )
                + "\n"
            )
            directional_row = {
                "timestamp_emitted_utc": ts,
                "pair": "EUR_USD",
                "signal_name": "directional_forecast",
                "direction": "UP",
                "resolution_window_min": 120,
                "resolution_status": "PENDING",
                "cycle_id": "cycle-2",
            }
            support_row = {
                **directional_row,
                "signal_name": "bb_squeeze_expansion_imminent",
                "direction": "EITHER",
            }
            metadata = {
                "forecast_direction": "UP",
                "forecast_confidence": 0.49,
                "forecast_cycle_id": "cycle-2",
                "forecast_market_support": {
                    "ok": True,
                    "direction": "UP",
                    "aligned_projection_count": 0,
                    "timing_projection_count": 1,
                    "best_hit_rate": 0.92,
                    "best_samples": 98,
                    "reason": "bb squeeze timing supports raw forecast",
                    "signals": [{"name": "bb_squeeze_expansion_imminent", "direction": "EITHER"}],
                },
            }
            intent = OrderIntent(
                pair="EUR_USD",
                side=Side.LONG,
                order_type=OrderType.LIMIT,
                units=5000,
                entry=1.1734,
                tp=1.1762,
                sl=1.1718,
                thesis="support telemetry intent",
                market_context=MarketContext(
                    regime="TREND_UP",
                    narrative="",
                    chart_story="",
                    method=TradeMethod.BREAKOUT_FAILURE,
                    invalidation="",
                ),
                metadata=metadata,
            )
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                quotes={"EUR_USD": Quote("EUR_USD", 1.1733, 1.1735, now)},
            )

            (data_root / "projection_ledger.jsonl").write_text(json.dumps(directional_row) + "\n")
            missing_cache = _build_telemetry_live_readiness_cache(
                data_root=data_root,
                validation_time_utc=now,
            )
            missing_codes = {
                issue["code"]
                for issue in _telemetry_live_readiness_issues(
                    intent,
                    metadata,
                    snapshot,
                    now,
                    cache=missing_cache,
                )
            }

            (data_root / "projection_ledger.jsonl").write_text(
                json.dumps(directional_row) + "\n" + json.dumps(support_row) + "\n"
            )
            complete_cache = _build_telemetry_live_readiness_cache(
                data_root=data_root,
                validation_time_utc=now,
            )
            complete_codes = {
                issue["code"]
                for issue in _telemetry_live_readiness_issues(
                    intent,
                    metadata,
                    snapshot,
                    now,
                    cache=complete_cache,
                )
            }

        self.assertIn("TELEMETRY_MARKET_SUPPORT_PROJECTION_REQUIRED_FOR_LIVE", missing_codes)
        self.assertNotIn("TELEMETRY_MARKET_SUPPORT_PROJECTION_REQUIRED_FOR_LIVE", complete_codes)

    def test_fresh_entry_live_rr_issue_only_blocks_non_positive_reward(self) -> None:
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _fresh_entry_live_reward_risk_issue

        intent = OrderIntent(
            pair="AUD_NZD",
            side=Side.LONG,
            order_type=OrderType.LIMIT,
            units=12000,
            entry=1.2145,
            tp=1.21575,
            sl=1.213,
            thesis="fresh failed-break limit",
            market_context=MarketContext(
                regime="RANGE",
                narrative="",
                chart_story="",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="",
            ),
            metadata={"position_intent": "NEW"},
        )

        self.assertIsNone(_fresh_entry_live_reward_risk_issue(intent, SimpleNamespace(reward_risk=0.83)))
        issue = _fresh_entry_live_reward_risk_issue(intent, SimpleNamespace(reward_risk=0.0))
        self.assertIsNotNone(issue)
        self.assertEqual(issue["code"], "FRESH_ENTRY_REWARD_RISK_NOT_POSITIVE")
        self.assertEqual(issue["severity"], "BLOCK")
        self.assertIsNone(_fresh_entry_live_reward_risk_issue(intent, SimpleNamespace(reward_risk=1.01)))
        hedge_intent = OrderIntent(
            pair="AUD_NZD",
            side=Side.LONG,
            order_type=OrderType.LIMIT,
            units=12000,
            entry=1.2145,
            tp=1.21575,
            sl=1.213,
            thesis="recovery hedge",
            market_context=intent.market_context,
            metadata={"position_intent": "HEDGE", "hedge_recovery": True},
        )
        self.assertIsNone(_fresh_entry_live_reward_risk_issue(hedge_intent, SimpleNamespace(reward_risk=0.83)))

    def test_forecast_first_trend_continuation_needs_live_reward_risk_floor(self) -> None:
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _fresh_entry_live_reward_risk_issue

        intent = OrderIntent(
            pair="EUR_CHF",
            side=Side.LONG,
            order_type=OrderType.STOP_ENTRY,
            units=6300,
            entry=0.92177,
            tp=0.92317,
            sl=0.91909,
            thesis="forecast-first continuation with thin payoff",
            market_context=MarketContext(
                regime="RANGE current; TREND_CONTINUATION campaign lane",
                narrative="",
                chart_story="",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="",
            ),
            metadata={"forecast_seed": True},
        )

        issue = _fresh_entry_live_reward_risk_issue(intent, SimpleNamespace(reward_risk=1.196))

        self.assertIsNotNone(issue)
        self.assertEqual(issue["code"], "FORECAST_TREND_CONTINUATION_REWARD_RISK_TOO_LOW")
        self.assertEqual(issue["severity"], "WARN")
        self.assertIsNone(_fresh_entry_live_reward_risk_issue(intent, SimpleNamespace(reward_risk=1.5)))

    def test_range_rotation_uses_range_probability_floor_for_live_context(self) -> None:
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _forecast_live_readiness_issue

        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        metadata = {
            "forecast_direction": "RANGE",
            "forecast_confidence": 0.51,
            "geometry_model": "RANGE_RAIL_LIMIT",
            "range_support": 1.1618,
            "range_resistance": 1.1650,
            "range_tp_is_inside_box": True,
            "range_sl_outside_box": True,
        }
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.LIMIT,
            units=4000,
            entry=1.1619,
            tp=1.1633,
            sl=1.1601,
            thesis="range rail rotation",
            market_context=MarketContext(
                regime="RANGE current; RANGE_ROTATION campaign lane",
                narrative="",
                chart_story="",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="",
            ),
            metadata=metadata,
        )

        self.assertIsNone(_forecast_live_readiness_issue(intent, metadata, TradeMethod.RANGE_ROTATION))

        weak_metadata = {**metadata, "forecast_confidence": 0.49}
        weak_issue = _forecast_live_readiness_issue(intent, weak_metadata, TradeMethod.RANGE_ROTATION)
        self.assertEqual(weak_issue["code"], "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE")

    def test_range_forecast_allows_tp_proven_breakout_failure_limit_for_live_context(self) -> None:
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _forecast_live_readiness_issue

        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        metadata = {
            "forecast_direction": "RANGE",
            "forecast_confidence": 0.54,
            "forecast_range_low_price": 1.1618,
            "forecast_range_high_price": 1.1650,
            "geometry_model": "RANGE_RAIL_LIMIT",
            "range_indicator_source": "forecast_range_box",
            "range_support": 1.1618,
            "range_resistance": 1.1650,
            "range_entry_side": "support",
            "range_tp_is_inside_box": True,
            "range_sl_outside_box": True,
            "attach_take_profit_on_fill": True,
            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
            "tp_target_intent": "HARVEST",
            "opportunity_mode": "HARVEST",
            "positive_rotation_mode": "TP_PROVEN_HARVEST",
            "positive_rotation_live_ready": True,
            "positive_rotation_pessimistic_expectancy_jpy": 180.0,
            "capture_take_profit_scope": "PAIR_SIDE_METHOD",
            "capture_take_profit_scope_key": "EUR_USD|LONG|BREAKOUT_FAILURE|TAKE_PROFIT_ORDER",
            "capture_take_profit_trades": 20,
            "capture_take_profit_losses": 0,
            "capture_take_profit_expectancy_jpy": 591.5,
        }
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.LIMIT,
            units=4000,
            entry=1.1619,
            tp=1.1633,
            sl=1.1601,
            thesis="tp-proven failed-break fade",
            market_context=MarketContext(
                regime="RANGE current; BREAKOUT_FAILURE campaign lane",
                narrative="failed-break fade has exact local broker TP proof",
                chart_story="support reclaim inside the box",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="support fails",
            ),
            metadata=metadata,
        )

        self.assertIsNone(
            _forecast_live_readiness_issue(intent, metadata, TradeMethod.BREAKOUT_FAILURE)
        )
        stop_issue = _forecast_live_readiness_issue(
            replace(intent, order_type=OrderType.STOP_ENTRY),
            metadata,
            TradeMethod.BREAKOUT_FAILURE,
        )
        self.assertIsNotNone(stop_issue)
        self.assertEqual(
            stop_issue["code"],
            "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
        )

    def test_range_limit_uses_same_side_unselected_projection_support(self) -> None:
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _forecast_live_readiness_issue

        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        metadata = {
            "forecast_direction": "RANGE",
            "forecast_confidence": 0.43,
            "forecast_horizon_min": 120,
            "geometry_model": "RANGE_RAIL_LIMIT",
            "range_tp_is_inside_box": True,
            "range_sl_outside_box": True,
            "forecast_market_support": {
                "ok": False,
                "direction": "RANGE",
                "unselected_projection_count": 1,
                "unselected_signals": [
                    {
                        "name": "liquidity_sweep_high",
                        "direction": "DOWN",
                        "confidence": 0.7034,
                        "hit_rate": 1.0,
                        "samples": 40,
                        "target_pips": 6.0,
                        "lead_time_min": 15.0,
                    }
                ],
            },
        }
        intent = OrderIntent(
            pair="USD_CAD",
            side=Side.SHORT,
            order_type=OrderType.LIMIT,
            units=8000,
            entry=1.40019,
            tp=1.39933,
            sl=1.40121,
            thesis="upper-rail sweep fade",
            market_context=MarketContext(
                regime="RANGE current; RANGE_ROTATION campaign lane",
                narrative="audited sweep supports the upper-rail fade",
                chart_story="range rail geometry waits above current price",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="SL trades",
            ),
            metadata=metadata,
        )

        self.assertIsNone(_forecast_live_readiness_issue(intent, metadata, TradeMethod.RANGE_ROTATION))

        market_intent = OrderIntent(
            pair="USD_CAD",
            side=Side.SHORT,
            order_type=OrderType.MARKET,
            units=8000,
            entry=None,
            tp=1.39933,
            sl=1.40121,
            thesis="do not market-chase weak range forecast",
            market_context=intent.market_context,
            metadata=metadata,
        )
        market_issue = _forecast_live_readiness_issue(
            market_intent,
            metadata,
            TradeMethod.RANGE_ROTATION,
        )

        self.assertEqual(market_issue["code"], "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE")

    def test_technical_harvest_precision_allows_only_audited_short_scalp_shape(self) -> None:
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _forecast_live_readiness_issue

        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        metadata = {
            "forecast_direction": "DOWN",
            "forecast_confidence": 0.23,
            "chart_direction_bias": "SHORT",
            "m1_atr_percentile_100": 0.10,
            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
            "tp_target_intent": "HARVEST",
            "opportunity_mode": "HARVEST",
            "forecast_market_support": {
                "ok": False,
                "direction": "DOWN",
                "aligned_projection_count": 0,
            },
        }
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.LIMIT,
            units=1000,
            entry=1.17330,
            tp=1.17280,
            sl=1.17370,
            thesis="audited EUR_USD short low-ATR scalp",
            market_context=MarketContext(
                regime="BREAKOUT_FAILURE rejection retest",
                narrative="low-ATR M1 short harvest",
                chart_story="M1 ATR low",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="4 pip stop",
            ),
            metadata=metadata,
        )

        issue = _forecast_live_readiness_issue(intent, metadata, TradeMethod.BREAKOUT_FAILURE)

        self.assertIsNone(issue)
        self.assertTrue(metadata["technical_harvest_precision_live_ready"])
        self.assertEqual(
            metadata["technical_harvest_precision_support"]["name"],
            "EUR_USD_DOWN_M1_ATR_LOW_TP5_SL4",
        )

        stale_metadata = {**metadata, "m1_atr_percentile_100": 0.50}
        stale_metadata.pop("technical_harvest_precision_live_ready", None)
        stale_metadata.pop("technical_harvest_precision_support", None)
        stale_intent = replace(intent, tp=1.17265, metadata=stale_metadata)

        blocked = _forecast_live_readiness_issue(
            stale_intent,
            stale_metadata,
            TradeMethod.BREAKOUT_FAILURE,
        )

        self.assertEqual(blocked["code"], "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE")

        negative_metadata = {**metadata, "m5_ema_slope_5": 0.20}
        negative_metadata.pop("technical_harvest_precision_live_ready", None)
        negative_metadata.pop("technical_harvest_precision_support", None)
        negative_intent = replace(intent, metadata=negative_metadata)

        negative = _forecast_live_readiness_issue(
            negative_intent,
            negative_metadata,
            TradeMethod.BREAKOUT_FAILURE,
        )

        self.assertEqual(negative["code"], "TECHNICAL_HARVEST_NEGATIVE_BUCKET_FOR_LIVE")
        self.assertEqual(
            negative_metadata["technical_harvest_precision_negative"]["name"],
            "EUR_USD_DOWN_M5_EMA_SLOPE5_OPPOSED_TP5_SL4",
        )

    def test_intent_generation_emits_audited_technical_harvest_scalp_geometry(self) -> None:
        from quant_rabbit.models import AccountSummary, BrokerSnapshot, OrderType, Quote, TradeMethod
        from quant_rabbit.strategy.intent_generator import (
            _forecast_live_readiness_issue,
            _intent_from_lane,
            _method_context_issues,
        )

        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        now = datetime.now(timezone.utc)
        quote = Quote(pair="EUR_USD", bid=1.16264, ask=1.16272, timestamp_utc=now)
        snapshot = BrokerSnapshot(
            fetched_at_utc=now,
            positions=(),
            orders=(),
            quotes={
                "EUR_USD": quote,
                "USD_JPY": Quote(pair="USD_JPY", bid=156.64, ask=156.648, timestamp_utc=now),
            },
            account=AccountSummary(
                nav_jpy=200_000.0,
                balance_jpy=200_000.0,
                margin_used_jpy=0.0,
                margin_available_jpy=200_000.0,
                hedging_enabled=True,
                fetched_at_utc=now,
            ),
            home_conversions={"USD": 156.64},
        )

        intent = _intent_from_lane(
            {
                "desk": "failure_trader",
                "pair": "EUR_USD",
                "direction": "SHORT",
                "method": "BREAKOUT_FAILURE",
                "adoption": "RISK_REPAIR_DRY_RUN",
                "campaign_role": "NOW_IF_REPAIRED",
                "reason": "audited low-ATR short harvest",
                "required_receipt": "dry-run under loss cap",
                "forecast_direction": "DOWN",
                "forecast_confidence": 0.23,
                "forecast_target_price": 1.15500,
            },
            quote,
            snapshot,
            max_loss_jpy=1000.0,
            atr_pips=2.0,
            order_type_override=OrderType.LIMIT,
            chart_context={
                "chart_direction_bias": "SHORT",
                "m1_atr_percentile_100": 0.10,
            },
            pair_chart={
                "pair": "EUR_USD",
                "views": [{"granularity": "M1", "indicators": {"atr_pips": 2.0}}],
            },
        )

        self.assertAlmostEqual((intent.entry - intent.tp) * 10000, 5.0, places=3)
        self.assertAlmostEqual((intent.sl - intent.entry) * 10000, 4.0, places=3)
        self.assertEqual(intent.metadata["tp_target_source"], "TECHNICAL_HARVEST_PRECISION")
        self.assertEqual(
            intent.metadata["technical_harvest_precision_geometry_rule"],
            "EUR_USD_DOWN_M1_ATR_LOW_TP5_SL4",
        )
        self.assertNotIn(
            "HARVEST_TP_STRUCTURE_MISSING",
            {issue["code"] for issue in _method_context_issues(intent)},
        )

        self.assertIsNone(
            _forecast_live_readiness_issue(intent, intent.metadata, TradeMethod.BREAKOUT_FAILURE)
        )
        self.assertTrue(intent.metadata["technical_harvest_precision_live_ready"])

    def test_bidask_replay_rank_only_does_not_clear_live_forecast_gates(self) -> None:
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _forecast_live_readiness_issue

        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        rules_path = write_bidask_replay_fixture_rules(Path(self._default_root_tmp.name))
        support_metadata = {
            "forecast_direction": "DOWN",
            "forecast_confidence": 0.23,
            "chart_direction_bias": "SHORT",
            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
            "tp_target_intent": "HARVEST",
            "opportunity_mode": "HARVEST",
            "forecast_market_support": {
                "ok": False,
                "direction": "DOWN",
                "aligned_projection_count": 0,
            },
        }
        support_intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.LIMIT,
            units=1000,
            entry=1.17330,
            tp=1.17280,
            sl=1.17400,
            thesis="S5 bid/ask replay-backed EUR_USD short harvest",
            market_context=MarketContext(
                regime="BREAKOUT_FAILURE rejection retest",
                narrative="S5 bid/ask replay supports EUR_USD DOWN attached harvest",
                chart_story="retest below resistance",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="7 pip stop",
            ),
            metadata=support_metadata,
        )

        with bidask_rules_env(rules_path):
            support_blocked = _forecast_live_readiness_issue(
                support_intent,
                support_metadata,
                TradeMethod.BREAKOUT_FAILURE,
            )
        self.assertEqual(support_blocked["code"], "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE")
        self.assertNotIn("bidask_replay_precision_live_ready", support_metadata)
        self.assertNotIn("bidask_replay_precision_support", support_metadata)

        block_metadata = {
            "forecast_direction": "UP",
            "forecast_confidence": 0.87,
            "chart_direction_bias": "LONG",
        }
        block_intent = OrderIntent(
            pair="AUD_JPY",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=1000,
            entry=114.289,
            tp=114.338,
            sl=114.250,
            thesis="AUD_JPY high-confidence long must not replay the losing bucket",
            market_context=MarketContext(
                regime="TREND_CONTINUATION campaign lane",
                narrative="forecast points up",
                chart_story="old high-confidence AUD_JPY UP bucket",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="forecast invalidation",
            ),
            metadata=block_metadata,
        )

        with bidask_rules_env(rules_path):
            blocked = _forecast_live_readiness_issue(
                block_intent,
                block_metadata,
                TradeMethod.TREND_CONTINUATION,
            )

        self.assertEqual(blocked["code"], "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE")
        self.assertEqual(
            block_metadata["bidask_replay_precision_negative"]["name"],
            "AUD_JPY_UP_S5_BIDASK_NEGATIVE_EXPECTANCY",
        )

        contrarian_metadata = {
            "forecast_direction": "UP",
            "forecast_confidence": 0.80,
            "forecast_horizon_min": 60,
            "chart_direction_bias": "LONG",
            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
            "tp_target_intent": "HARVEST",
            "opportunity_mode": "HARVEST",
            "forecast_market_support": {
                "ok": False,
                "direction": "UP",
                "aligned_projection_count": 0,
            },
        }
        contrarian_intent = OrderIntent(
            pair="AUD_JPY",
            side=Side.SHORT,
            order_type=OrderType.LIMIT,
            units=1000,
            entry=114.289,
            tp=114.189,
            sl=114.359,
            thesis="AUD_JPY UP 0.75-0.90 forecast bucket is faded by S5 bid/ask replay",
            market_context=MarketContext(
                regime="BREAKOUT_FAILURE rejection retest",
                narrative="audited contrarian replay supports SHORT",
                chart_story="fade only after retest geometry",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="7 pip stop",
            ),
            metadata=contrarian_metadata,
        )

        with bidask_rules_env(rules_path):
            self.assertIsNone(
                _forecast_live_readiness_issue(
                    contrarian_intent,
                    contrarian_metadata,
                    TradeMethod.BREAKOUT_FAILURE,
                )
            )
        self.assertNotIn("bidask_replay_precision_live_ready", contrarian_metadata)
        self.assertNotIn("bidask_replay_precision_support", contrarian_metadata)

    def test_bidask_replay_geometry_plan_emits_audited_limit_tp_sl_grid(self) -> None:
        from quant_rabbit.models import OrderType, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _bidask_replay_precision_geometry_plan

        with tempfile.TemporaryDirectory() as tmp:
            rules_path = Path(tmp) / "bidask_live_grade.json"
            rules_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "generated_at_utc": "2026-06-22T00:00:00Z",
                        "generated_from": "unit-test",
                        "edge_rules": [],
                        "negative_rules": [],
                        "contrarian_edge_rules": [
                            {
                                "name": "AUD_JPY_UP_FADE_TO_DOWN_S5_BIDASK_CONTRARIAN_HARVEST_TP10_SL7",
                                "pair": "AUD_JPY",
                                "side": "SHORT",
                                "direction": "DOWN",
                                "forecast_direction": "UP",
                                "faded_direction": "UP",
                                "contrarian_edge": True,
                                "confidence_bucket": "0.75-0.90",
                                "horizon_bucket": "61-240m",
                                "granularity": "S5",
                                "samples": 124,
                                "directional_hit_rate": 0.76,
                                "avg_final_pips": 5.8,
                                "avg_mfe_pips": 12.0,
                                "avg_mae_pips": 4.5,
                                "optimized_take_profit_pips": 10.0,
                                "optimized_stop_loss_pips": 7.0,
                                "optimized_avg_realized_pips": 2.4,
                                "optimized_win_rate": 0.70,
                                "optimized_profit_factor": 2.5,
                                "daily_stability_status": "DAILY_STABLE",
                                "active_days": 4,
                                "positive_day_rate": 0.75,
                                "min_target_pips": 9.8,
                                "max_target_pips": 10.5,
                                "max_stop_pips": 7.2,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with bidask_rules_env(rules_path):
                tp, sl, metadata = _bidask_replay_precision_geometry_plan(
                    pair="AUD_JPY",
                    side=Side.SHORT,
                    method=TradeMethod.BREAKOUT_FAILURE,
                    order_type=OrderType.LIMIT,
                    entry=114.289,
                    tp=114.239,
                    sl=114.389,
                    chart_context={"chart_direction_bias": "LONG"},
                    tp_execution_metadata={
                        "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                        "tp_target_intent": "HARVEST",
                        "opportunity_mode": "HARVEST",
                    },
                    forecast_direction="UP",
                    forecast_confidence=0.80,
                    forecast_horizon_min=90,
                )

        self.assertEqual(tp, 114.189)
        self.assertEqual(sl, 114.359)
        self.assertTrue(metadata["bidask_replay_precision_geometry"])
        self.assertEqual(metadata["tp_target_source"], "BIDASK_REPLAY_PRECISION")
        self.assertEqual(metadata["bidask_replay_precision_geometry_tp_pips"], 10.0)
        self.assertEqual(metadata["bidask_replay_precision_geometry_stop_pips"], 7.0)

    def test_directional_forecast_weak_hit_rate_blocks_live_readiness(self) -> None:
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _forecast_live_readiness_issue

        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        metadata = {
            "forecast_direction": "UP",
            "forecast_confidence": 0.82,
            "forecast_raw_confidence": 0.91,
            "forecast_directional_calibration_name": "directional_forecast_up",
            "forecast_directional_hit_rate": 0.1,
            "forecast_directional_samples": 12,
        }
        intent = OrderIntent(
            pair="USD_CAD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=5000,
            entry=None,
            tp=1.4050,
            sl=1.3900,
            thesis="high-confidence forecast from a weak realized bucket",
            market_context=MarketContext(
                regime="TREND_UP current; BREAKOUT_FAILURE campaign lane",
                narrative="",
                chart_story="",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="",
            ),
            metadata=metadata,
        )

        issue = _forecast_live_readiness_issue(intent, metadata, TradeMethod.BREAKOUT_FAILURE)

        self.assertEqual(issue["code"], "FORECAST_DIRECTIONAL_HIT_RATE_WEAK_FOR_LIVE")
        self.assertIn("hit_rate=0.10", issue["message"])

    def test_directional_forecast_invalidation_first_blocks_live_readiness(self) -> None:
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _forecast_live_readiness_issue

        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        metadata = {
            "forecast_direction": "UP",
            "forecast_confidence": 0.82,
            "forecast_raw_confidence": 0.91,
            "forecast_directional_calibration_name": "directional_forecast_up",
            "forecast_directional_hit_rate": 0.72,
            "forecast_directional_samples": 20,
            "forecast_directional_invalidation_first_rate": 0.75,
            "forecast_directional_invalidation_first_count": 15,
        }
        intent = OrderIntent(
            pair="USD_CAD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=5000,
            entry=None,
            tp=1.4050,
            sl=1.3900,
            thesis="high-confidence forecast that still touches invalidation first",
            market_context=MarketContext(
                regime="TREND_UP current; TREND_CONTINUATION campaign lane",
                narrative="",
                chart_story="",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="",
            ),
            metadata=metadata,
        )

        issue = _forecast_live_readiness_issue(intent, metadata, TradeMethod.TREND_CONTINUATION)

        self.assertEqual(issue["code"], "FORECAST_DIRECTIONAL_INVALIDATION_FIRST_FOR_LIVE")
        self.assertIn("15/20", issue["message"])

    def test_directional_forecast_weak_hit_rate_falls_back_to_market_support(self) -> None:
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _forecast_live_readiness_issue

        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        metadata = {
            "forecast_direction": "UP",
            "forecast_confidence": 0.82,
            "forecast_raw_confidence": 0.91,
            "forecast_market_support": {
                "directional_calibration_name": "directional_forecast_up",
                "directional_hit_rate": 0.1,
                "directional_samples": 12,
            },
        }
        intent = OrderIntent(
            pair="AUD_CAD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=5000,
            entry=None,
            tp=0.9910,
            sl=0.9860,
            thesis="high-confidence forecast with weak nested directional calibration",
            market_context=MarketContext(
                regime="TREND_UP current; BREAKOUT_FAILURE campaign lane",
                narrative="",
                chart_story="",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="",
            ),
            metadata=metadata,
        )

        issue = _forecast_live_readiness_issue(intent, metadata, TradeMethod.BREAKOUT_FAILURE)

        self.assertEqual(issue["code"], "FORECAST_DIRECTIONAL_HIT_RATE_WEAK_FOR_LIVE")
        self.assertIn("directional_forecast_up", issue["message"])

    def test_projection_support_does_not_clear_opposing_chart_bias(self) -> None:
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _forecast_live_readiness_issue

        os.environ["QR_REQUIRE_FORECAST_FOR_LIVE"] = "1"
        metadata = {
            "forecast_direction": "UP",
            "forecast_confidence": 0.49,
            "forecast_raw_confidence": 0.63,
            "chart_direction_bias": "SHORT",
            "forecast_market_support": {
                "ok": True,
                "direction": "UP",
                "aligned_projection_count": 1,
                "timing_projection_count": 0,
                "best_hit_rate": 0.62,
                "best_samples": 48,
                "reason": "liquidity_sweep_low supports UP",
            },
        }
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.STOP_ENTRY,
            units=5000,
            entry=1.1734,
            tp=1.1762,
            sl=1.1718,
            thesis="near-miss forecast with opposing chart bias",
            market_context=MarketContext(
                regime="TREND_UP current; TREND_CONTINUATION campaign lane",
                narrative="",
                chart_story="",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="",
            ),
            metadata=metadata,
        )

        issue = _forecast_live_readiness_issue(intent, metadata, TradeMethod.TREND_CONTINUATION)

        self.assertEqual(issue["code"], "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE")

    def test_projection_support_falls_back_when_pair_bucket_is_thin(self) -> None:
        from quant_rabbit.strategy.intent_generator import _forecast_market_support_for_forecast

        forecast = SimpleNamespace(direction="UP", raw_confidence=0.63)
        signal = SimpleNamespace(
            name="liquidity_sweep_low",
            direction="UP",
            confidence=0.88,
            bonus_magnitude=12.0,
            timeframe="M5",
            rationale="M5 equal-lows sweep target, fade LONG",
        )
        hit_rates = {
            "liquidity_sweep_low": {
                "EUR_USD:RANGE": {"hit_rate": 1.0, "samples": 2},
                "_all_pairs:_all_regimes": {"hit_rate": 0.62, "samples": 48},
            }
        }

        support = _forecast_market_support_for_forecast(
            pair="EUR_USD",
            forecast=forecast,
            projection_signals=[signal],
            hit_rates=hit_rates,
            regime="RANGE",
        )

        self.assertFalse(support["ok"])
        self.assertEqual(support["signals"][0]["samples"], 48)
        self.assertFalse(support["signals"][0]["live_precision_ok"])

    def test_forecast_market_support_exposes_directional_forecast_calibration(self) -> None:
        from quant_rabbit.strategy.intent_generator import _forecast_market_support_for_forecast

        forecast = SimpleNamespace(direction="UP", raw_confidence=0.91)
        hit_rates = {
            "directional_forecast_up": {
                "USD_CAD:TREND": {"hit_rate": 0.1, "samples": 12},
                "_all_pairs:_all_regimes": {"hit_rate": 0.62, "samples": 100},
            }
        }

        support = _forecast_market_support_for_forecast(
            pair="USD_CAD",
            forecast=forecast,
            projection_signals=[],
            hit_rates=hit_rates,
            regime="TREND",
        )

        self.assertFalse(support["ok"])
        self.assertEqual(support["directional_calibration_name"], "directional_forecast_up")
        self.assertAlmostEqual(support["directional_hit_rate"], 0.1)
        self.assertEqual(support["directional_samples"], 12)

    def test_range_forecast_exposes_its_own_box_hold_calibration(self) -> None:
        from quant_rabbit.strategy.intent_generator import _forecast_market_support_for_forecast

        forecast = SimpleNamespace(direction="RANGE", raw_confidence=0.68)
        hit_rates = {
            "directional_forecast_range": {
                "EUR_USD:RANGE": {
                    "hit_rate": 0.64,
                    "samples": 40,
                    "economic_hit_rate": 0.55,
                    "economic_samples": 40,
                    "timeout_rate": 0.0,
                    "timeout_count": 0,
                    "invalidation_first_rate": 0.45,
                    "invalidation_first_count": 18,
                },
            },
        }

        support = _forecast_market_support_for_forecast(
            pair="EUR_USD",
            forecast=forecast,
            projection_signals=[],
            hit_rates=hit_rates,
            regime="RANGE",
        )

        self.assertFalse(support["ok"])
        self.assertEqual(
            support["directional_calibration_name"],
            "directional_forecast_range",
        )
        self.assertAlmostEqual(support["directional_hit_rate"], 0.64)
        self.assertEqual(support["directional_samples"], 40)
        self.assertAlmostEqual(support["directional_economic_hit_rate"], 0.55)
        self.assertEqual(support["directional_economic_samples"], 40)

    def test_thin_range_forecast_preserves_range_calibration_identity(self) -> None:
        from quant_rabbit.strategy.intent_generator import _forecast_market_support_for_forecast

        forecast = SimpleNamespace(direction="RANGE", raw_confidence=0.68)
        hit_rates = {
            "directional_forecast_range": {
                "_all_pairs:_all_regimes": {
                    "hit_rate": 0.75,
                    "samples": 29,
                    "economic_hit_rate": 0.70,
                    "economic_samples": 29,
                },
            },
        }

        support = _forecast_market_support_for_forecast(
            pair="EUR_USD",
            forecast=forecast,
            projection_signals=[],
            hit_rates=hit_rates,
            regime="RANGE",
        )

        self.assertEqual(
            support["directional_calibration_name"],
            "directional_forecast_range",
        )
        self.assertIsNone(support["directional_hit_rate"])
        self.assertEqual(support["directional_samples"], 0)
        self.assertIsNone(support["directional_economic_hit_rate"])
        self.assertEqual(support["directional_economic_samples"], 0)

    def test_missing_range_alias_does_not_fall_back_to_mixed_base_bucket(self) -> None:
        from quant_rabbit.strategy.intent_generator import _forecast_market_support_for_forecast

        forecast = SimpleNamespace(direction="RANGE", raw_confidence=0.68)
        hit_rates = {
            "directional_forecast": {
                "EUR_USD:RANGE": {
                    "hit_rate": 0.99,
                    "samples": 100,
                    "economic_hit_rate": 0.99,
                    "economic_samples": 100,
                },
            },
        }

        support = _forecast_market_support_for_forecast(
            pair="EUR_USD",
            forecast=forecast,
            projection_signals=[],
            hit_rates=hit_rates,
            regime="RANGE",
        )

        self.assertEqual(
            support["directional_calibration_name"],
            "directional_forecast_range",
        )
        self.assertIsNone(support["directional_hit_rate"])
        self.assertEqual(support["directional_samples"], 0)
        self.assertIsNone(support["directional_economic_hit_rate"])
        self.assertEqual(support["directional_economic_samples"], 0)

    def test_forecast_directional_calibration_ignores_thin_global_bucket(self) -> None:
        from quant_rabbit.strategy.intent_generator import _forecast_market_support_for_forecast

        forecast = SimpleNamespace(direction="UP", raw_confidence=0.91)
        hit_rates = {
            "directional_forecast": {
                "_all_pairs:_all_regimes": {"hit_rate": 0.34, "samples": 29},
            }
        }

        support = _forecast_market_support_for_forecast(
            pair="AUD_JPY",
            forecast=forecast,
            projection_signals=[],
            hit_rates=hit_rates,
            regime="UNCLEAR",
        )

        self.assertIsNone(support["directional_hit_rate"])
        self.assertEqual(support["directional_samples"], 0)
        self.assertEqual(
            support["directional_calibration_name"],
            "directional_forecast_up",
        )

    def test_forecast_directional_calibration_uses_broad_directional_global_bucket(self) -> None:
        from quant_rabbit.strategy.intent_generator import _forecast_market_support_for_forecast

        forecast = SimpleNamespace(direction="UP", raw_confidence=0.91)
        hit_rates = {
            "directional_forecast_up": {
                "_all_pairs:_all_regimes": {"hit_rate": 0.34, "samples": 30},
            }
        }

        support = _forecast_market_support_for_forecast(
            pair="AUD_JPY",
            forecast=forecast,
            projection_signals=[],
            hit_rates=hit_rates,
            regime="UNCLEAR",
        )

        self.assertEqual(support["directional_calibration_name"], "directional_forecast_up")
        self.assertAlmostEqual(support["directional_hit_rate"], 0.34)
        self.assertEqual(support["directional_samples"], 30)

    def test_projection_support_dedupes_repeated_calibration_and_cites_best(self) -> None:
        from quant_rabbit.strategy.intent_generator import _forecast_market_support_for_forecast

        forecast = SimpleNamespace(direction="UP", confidence=0.6326, raw_confidence=0.6913)
        signals = [
            SimpleNamespace(
                name="liquidity_sweep_low",
                direction="UP",
                confidence=0.8976,
                bonus_magnitude=12.0,
                timeframe="M15",
                rationale="M15 equal-lows sweep target, fade LONG",
            ),
            SimpleNamespace(
                name="macro_event_nowcast_central_bank",
                direction="UP",
                confidence=0.79,
                bonus_magnitude=10.0,
                timeframe=None,
                rationale="FOMC statement nowcast supports USD_CAD LONG",
            ),
            SimpleNamespace(
                name="macro_event_nowcast_central_bank",
                direction="UP",
                confidence=0.79,
                bonus_magnitude=10.0,
                timeframe=None,
                rationale="FOMC press conference nowcast supports USD_CAD LONG",
            ),
        ]
        hit_rates = {
            "liquidity_sweep_low_up": {
                "_all_pairs:_all_regimes": {"hit_rate": 0.66, "samples": 100},
            },
            "macro_event_nowcast_central_bank_up": {
                "_all_pairs:_all_regimes": {"hit_rate": 1.0, "samples": 40},
            },
        }

        support = _forecast_market_support_for_forecast(
            pair="USD_CAD",
            forecast=forecast,
            projection_signals=signals,
            hit_rates=hit_rates,
            regime="RANGE",
        )

        self.assertTrue(support["ok"])
        self.assertEqual(support["aligned_projection_count"], 1)
        self.assertEqual(support["best_samples"], 40)
        self.assertAlmostEqual(support["best_hit_rate"], 1.0)
        self.assertEqual(support["signals"][0]["name"], "macro_event_nowcast_central_bank")
        self.assertEqual(
            sum(
                1
                for item in support["signals"]
                if item["calibration_name"] == "macro_event_nowcast_central_bank_up"
            ),
            1,
        )
        self.assertIn("macro_event_nowcast_central_bank UP hit_rate=1.00 samples=40", support["reason"])

    def test_projection_support_ignores_macro_event_beyond_forecast_horizon(self) -> None:
        from quant_rabbit.strategy.intent_generator import _forecast_market_support_for_forecast

        forecast = SimpleNamespace(direction="UP", confidence=0.4588, raw_confidence=0.66, horizon_min=180)
        signals = [
            SimpleNamespace(
                name="macro_event_nowcast_central_bank",
                direction="UP",
                confidence=0.79,
                bonus_magnitude=10.0,
                lead_time_min=3797.0,
                timeframe=None,
                rationale="FOMC statement nowcast is days away, not this entry horizon",
            )
        ]
        hit_rates = {
            "macro_event_nowcast_central_bank_up": {
                "_all_pairs:_all_regimes": {"hit_rate": 0.88, "samples": 75},
            },
        }

        support = _forecast_market_support_for_forecast(
            pair="EUR_CHF",
            forecast=forecast,
            projection_signals=signals,
            hit_rates=hit_rates,
            regime="RANGE",
        )

        self.assertFalse(support["ok"])
        self.assertEqual(support["aligned_projection_count"], 0)
        self.assertEqual(support["signals"], [])
        self.assertEqual(support["reason"], "no current projection clears audited support floors")

    def test_projection_support_keeps_directional_and_timing_hit_rates_separate(self) -> None:
        from quant_rabbit.strategy.intent_generator import _forecast_market_support_for_forecast

        forecast = SimpleNamespace(direction="UP", confidence=0.2437, raw_confidence=0.544)
        signals = [
            SimpleNamespace(
                name="bb_squeeze_expansion_imminent",
                direction="EITHER",
                confidence=0.60,
                bonus_magnitude=8.0,
                timeframe="M15",
                rationale="M15 squeeze timing, direction must be supplied elsewhere",
            ),
            SimpleNamespace(
                name="liquidity_sweep_low",
                direction="UP",
                confidence=0.93,
                bonus_magnitude=12.0,
                timeframe="M5",
                rationale="M5 equal-lows 6.0pip sweep target, fade LONG",
            ),
        ]
        hit_rates = {
            "bb_squeeze_expansion_imminent": {
                "_all_pairs:_all_regimes": {"hit_rate": 1.0, "samples": 40},
            },
            "liquidity_sweep_low_up": {
                "_all_pairs:_all_regimes": {"hit_rate": 1.0, "samples": 40},
            },
        }

        support = _forecast_market_support_for_forecast(
            pair="AUD_JPY",
            forecast=forecast,
            projection_signals=signals,
            hit_rates=hit_rates,
            regime="TREND",
        )

        self.assertTrue(support["ok"])
        self.assertEqual(support["aligned_projection_count"], 1)
        self.assertEqual(support["timing_projection_count"], 1)
        self.assertAlmostEqual(support["best_aligned_hit_rate"], 1.0)
        self.assertEqual(support["best_aligned_samples"], 40)
        self.assertAlmostEqual(support["best_timing_hit_rate"], 1.0)
        self.assertEqual(support["best_timing_samples"], 40)
        self.assertAlmostEqual(support["best_hit_rate"], 1.0)
        self.assertIn("liquidity_sweep_low UP hit_rate=1.00 samples=40", support["reason"])
        self.assertEqual(support["signals"][0]["direction"], "UP")

    def test_supported_weak_opposite_forecast_blocks_this_side(self) -> None:
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.LIMIT,
            units=5000,
            entry=1.1730,
            tp=1.1760,
            sl=1.1710,
            thesis="long despite supported down forecast",
            market_context=MarketContext(
                regime="TREND_DOWN",
                narrative="",
                chart_story="",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="",
            ),
            metadata={
                "forecast_direction": "DOWN",
                "forecast_confidence": 0.18,
                "forecast_raw_confidence": 0.55,
                "chart_direction_bias": "SHORT",
                "forecast_market_support": {
                    "ok": True,
                    "direction": "DOWN",
                    "aligned_projection_count": 1,
                    "timing_projection_count": 0,
                    "best_hit_rate": 1.0,
                    "best_samples": 40,
                    "reason": (
                        "news_theme_followthrough DOWN hit_rate=1.00 "
                        "samples=40 supports weak calibrated forecast"
                    ),
                    "signals": [
                        {
                            "name": "news_theme_followthrough",
                            "direction": "DOWN",
                            "confidence": 0.8,
                            "hit_rate": 1.0,
                            "samples": 40,
                        }
                    ],
                },
            },
        )

        codes = {
            issue["code"]: issue["severity"]
            for issue in _method_context_issues(intent)
        }

        self.assertEqual(codes["FORECAST_DIRECTION_CONFLICT"], "BLOCK")

    def test_unsupported_weak_directional_bucket_does_not_veto_opposite_side(self) -> None:
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = OrderIntent(
            pair="EUR_JPY",
            side=Side.SHORT,
            order_type=OrderType.LIMIT,
            units=5000,
            entry=185.60,
            tp=185.10,
            sl=185.95,
            thesis="short_retest_while_up_forecast_bucket_is_weak",
            market_context=MarketContext(
                regime="BREAKOUT_FAILURE retest",
                narrative="upside break failed and retest is selling",
                chart_story="failed break retest near resistance",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="resistance reclaims on M5 bodies",
            ),
            metadata={
                "forecast_direction": "UP",
                "forecast_confidence": 0.82,
                "forecast_raw_confidence": 0.91,
                "forecast_directional_calibration_name": "directional_forecast_up",
                "forecast_directional_hit_rate": 0.10,
                "forecast_directional_samples": 30,
                "forecast_market_support": {
                    "ok": False,
                    "direction": "UP",
                    "aligned_projection_count": 0,
                    "best_hit_rate": None,
                    "best_samples": 0,
                    "reason": "no current projection clears audited support floors",
                },
            },
        )

        codes = {issue["code"]: issue["severity"] for issue in _method_context_issues(intent)}

        self.assertNotIn("FORECAST_DIRECTION_CONFLICT", codes)

    def test_unclear_forecast_records_unselected_audited_news_projection(self) -> None:
        from quant_rabbit.strategy.intent_generator import _forecast_market_support_for_forecast

        forecast = SimpleNamespace(direction="UNCLEAR", raw_confidence=0.18)
        signal = SimpleNamespace(
            name="news_theme_followthrough",
            direction="DOWN",
            confidence=0.8,
            bonus_magnitude=10.0,
            timeframe="H1",
            rationale="risk-off news theme follows through lower",
        )
        hit_rates = {
            "news_theme_followthrough": {
                "_all_pairs:TREND": {"hit_rate": 0.651, "samples": 63},
                "_all_pairs:_all_regimes": {"hit_rate": 0.692, "samples": 91},
            }
        }

        support = _forecast_market_support_for_forecast(
            pair="EUR_USD",
            forecast=forecast,
            projection_signals=[signal],
            hit_rates=hit_rates,
            regime="TREND",
        )

        self.assertFalse(support["ok"])
        self.assertEqual(support["unselected_projection_count"], 1)
        self.assertEqual(support["best_unselected_samples"], 63)
        self.assertAlmostEqual(support["best_unselected_hit_rate"], 0.651)
        self.assertEqual(support["unselected_signals"][0]["name"], "news_theme_followthrough")
        self.assertIn("forecast=UNCLEAR", support["unselected_reason"])

    def test_forecast_context_payload_persists_unselected_news_signal_refs(self) -> None:
        forecast = SimpleNamespace(
            direction="UNCLEAR",
            confidence=0.18,
            raw_confidence=0.18,
            rationale_summary="contested technical forecast",
            drivers_for=(),
            drivers_against=("range prior",),
            component_scores={"UP": 41.0, "DOWN": 45.0, "RANGE": 43.0},
            market_support={
                "ok": False,
                "direction": "UNCLEAR",
                "reason": "forecast UNCLEAR has no executable direction; audited projection unselected",
                "signals": [],
                "unselected_projection_count": 1,
                "best_unselected_hit_rate": 0.651,
                "best_unselected_samples": 63,
                "unselected_reason": (
                    "news_theme_followthrough DOWN audited hit_rate=0.65 "
                    "samples=63 was unselected because forecast=UNCLEAR"
                ),
                "unselected_signals": [
                    {
                        "name": "news_theme_followthrough",
                        "direction": "DOWN",
                        "confidence": 0.8,
                        "hit_rate": 0.651,
                        "samples": 63,
                    }
                ],
            },
        )

        metadata = _forecast_context_payload(forecast)

        self.assertEqual(metadata["news_refs"], ["news:digest", "news:items"])
        self.assertEqual(metadata["news_signal_names"], ["news_theme_followthrough"])
        self.assertEqual(
            metadata["forecast_market_support"]["unselected_signals"][0]["name"],
            "news_theme_followthrough",
        )

    def test_unselected_range_projection_conflict_prevents_live_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"
            forecast = SimpleNamespace(
                direction="RANGE",
                confidence=0.59,
                raw_confidence=0.65,
                calibration_multiplier=0.9,
                current_price=1.17326,
                target_price=1.1748,
                invalidation_price=1.1710,
                horizon_min=60,
                rationale_summary="range rotation, but audited projection points lower",
                drivers_for=("range box intact",),
                drivers_against=("news followthrough lower",),
                market_support={
                    "ok": False,
                    "direction": "RANGE",
                    "reason": "RANGE forecast left a directional projection unselected",
                    "unselected_projection_count": 1,
                    "best_unselected_hit_rate": 0.56,
                    "best_unselected_samples": 27,
                    "unselected_signals": [
                        {
                            "name": "news_theme_followthrough",
                            "direction": "DOWN",
                            "confidence": 0.741,
                            "economic_hit_rate": 1.0,
                            "economic_samples": 40,
                            "hit_rate": 1.0,
                            "samples": 40,
                        }
                    ],
                },
            )

            with patch(
                "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                return_value=forecast,
            ):
                summary = IntentGenerator(
                    campaign_plan=_range_campaign(root),
                    strategy_profile=_strategy(root, status="CANDIDATE"),
                    output_path=output,
                    report_path=root / "intents.md",
                    pair_charts_path=_pair_charts(root),
                    max_loss_jpy=500.0,
                ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            range_limit = next(
                item
                for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
                and item["intent"]["order_type"] == "LIMIT"
            )
            issue = next(
                issue
                for issue in range_limit["risk_issues"]
                if issue["code"] == "FORECAST_RANGE_UNSELECTED_DIRECTION_CONFLICT"
            )

            self.assertEqual(summary.live_ready, 0)
            self.assertEqual(range_limit["status"], "DRY_RUN_PASSED")
            self.assertEqual(issue["severity"], "BLOCK")
            self.assertTrue(range_limit["live_blockers"])

    def test_projection_expiry_grace_avoids_same_cycle_false_blocker(self) -> None:
        from quant_rabbit.strategy.intent_generator import _expired_pending_projection_count

        emitted_at = datetime(2026, 6, 1, 6, 56, 21, tzinfo=timezone.utc)
        row = {
            "timestamp_emitted_utc": emitted_at.isoformat().replace("+00:00", "Z"),
            "pair": "EUR_USD",
            "signal_name": "bb_squeeze_expansion_imminent",
            "direction": "EITHER",
            "resolution_window_min": 150.0,
            "resolution_status": "PENDING",
        }
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            (data_root / "projection_ledger.jsonl").write_text(json.dumps(row) + "\n")
            with patch(
                "quant_rabbit.strategy.intent_generator.PROJECTION_PENDING_EXPIRY_GRACE_SECONDS",
                300.0,
            ):
                just_after_expiry = emitted_at + timedelta(minutes=150, seconds=60)
                stale_after_grace = emitted_at + timedelta(minutes=150, seconds=301)

                self.assertEqual(
                    _expired_pending_projection_count(
                        data_root=data_root,
                        validation_time_utc=just_after_expiry,
                    ),
                    0,
                )
                self.assertEqual(
                    _expired_pending_projection_count(
                        data_root=data_root,
                        validation_time_utc=stale_after_grace,
                    ),
                    1,
                )

    def test_projection_expiry_ignores_rows_emitted_after_validation_time(self) -> None:
        from quant_rabbit.strategy.intent_generator import (
            _build_telemetry_live_readiness_cache,
            _expired_pending_projection_count,
        )

        validation_time = datetime(2026, 6, 1, 6, 56, 21, tzinfo=timezone.utc)
        row = {
            "timestamp_emitted_utc": (validation_time + timedelta(seconds=5)).isoformat().replace("+00:00", "Z"),
            "pair": "EUR_USD",
            "signal_name": "news_theme_followthrough",
            "direction": "DOWN",
            "resolution_window_min": 0.0,
            "resolution_status": "PENDING",
            "cycle_id": "same-generate-intents-cycle",
        }
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            (data_root / "projection_ledger.jsonl").write_text(json.dumps(row) + "\n")

            self.assertEqual(
                _expired_pending_projection_count(
                    data_root=data_root,
                    validation_time_utc=validation_time,
                ),
                0,
            )
            self.assertEqual(
                _build_telemetry_live_readiness_cache(
                    data_root=data_root,
                    validation_time_utc=validation_time,
                ).expired_pending_projection_count,
                0,
            )

    def test_blocks_chart_direction_conflict_before_live_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_campaign(root, direction="SHORT"),
                strategy_profile=_strategy(root, status="CANDIDATE", direction="SHORT"),
                pair_charts_path=_pair_charts_with_direction(
                    root,
                    long_score=0.76,
                    short_score=0.19,
                    dominant_regime="TREND_UP",
                    m5_regime="TREND_UP",
                ),
                output_path=output,
                report_path=root / "intents.md",
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            issue_codes = {issue["code"] for item in payload["results"] for issue in item["risk_issues"]}

            self.assertEqual(summary.live_ready, 0)
            self.assertIn("CHART_DIRECTION_CONFLICT", issue_codes)
            self.assertTrue(all(item["status"] == "DRY_RUN_BLOCKED" for item in payload["results"]))

    def test_tied_chart_score_gap_does_not_emit_direction_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root, direction="LONG"),
                strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                pair_charts_path=_pair_charts_with_direction(
                    root,
                    long_score=0.48,
                    short_score=0.52,
                    dominant_regime="RANGE",
                    m5_regime="RANGE",
                    m5_long_bias=0.48,
                    m5_short_bias=0.52,
                ),
                output_path=output,
                report_path=root / "intents.md",
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            issue_codes = {issue["code"] for item in payload["results"] for issue in item["risk_issues"]}
            metadata = payload["results"][0]["intent"]["metadata"]

            self.assertEqual(metadata["chart_score_balance"], "TIED")
            self.assertIsNone(metadata["chart_direction_bias"])
            self.assertNotIn("CHART_DIRECTION_CONFLICT", issue_codes)

    def test_range_rotation_m5_tie_does_not_fallback_to_broader_chart_lean(self) -> None:
        # Live 2026-06-18 GBP_CHF: the broader confluence packet leaned SHORT,
        # but the operating M5 range read was exactly tied. RANGE_ROTATION
        # direction conflict must use the operating M5 bias; a tie is neutral,
        # not a reason to inherit the broader lean and starve the retest lane.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_range_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE", direction="LONG"),
                pair_charts_path=_pair_charts_with_direction(
                    root,
                    long_score=0.3049,
                    short_score=0.5771,
                    dominant_regime="RANGE",
                    m5_regime="RANGE",
                    m5_long_bias=0.4286,
                    m5_short_bias=0.4286,
                ),
                output_path=output,
                report_path=root / "intents.md",
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            issue_codes = {issue["code"] for issue in result["risk_issues"]}
            metadata = result["intent"]["metadata"]

            self.assertEqual(metadata["chart_direction_bias"], "SHORT")
            self.assertEqual(metadata["m5_long_bias"], 0.4286)
            self.assertEqual(metadata["m5_short_bias"], 0.4286)
            self.assertNotIn("CHART_DIRECTION_CONFLICT", issue_codes)

    def test_sl_free_still_blocks_decisive_trend_continuation_conflict(self) -> None:
        import os

        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"

                summary = IntentGenerator(
                    campaign_plan=_campaign(root),
                    strategy_profile=_strategy(root, status="CANDIDATE"),
                    pair_charts_path=_pair_charts_with_direction(
                        root,
                        long_score=0.0484,
                        short_score=0.9456,
                        dominant_regime="TREND_DOWN",
                        m5_regime="TREND_DOWN",
                    ),
                    output_path=output,
                    report_path=root / "intents.md",
                    max_loss_jpy=500.0,
                ).run(snapshot_path=_snapshot(root))

                payload = json.loads(output.read_text())
                issues = [issue for item in payload["results"] for issue in item["risk_issues"]]
                long_results = [item for item in payload["results"] if item["intent"]["side"] == "LONG"]
                short_results = [item for item in payload["results"] if item["intent"]["side"] == "SHORT"]

                self.assertGreater(summary.live_ready, 0)
                self.assertTrue(all(item["status"] == "DRY_RUN_BLOCKED" for item in long_results))
                self.assertTrue(any(item["status"] == "LIVE_READY" for item in short_results))
                self.assertTrue(
                    any(
                        issue["code"] == "CHART_DIRECTION_CONFLICT"
                        and issue["severity"] == "BLOCK"
                        and "trend-continuation hard gate" in issue["message"]
                        for issue in issues
                    ),
                    f"expected hard trend conflict blocker, got {issues}",
                )
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

    def test_sl_free_preserves_range_and_failure_entries_when_trend_lane_blocks(self) -> None:
        import os

        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)

                range_output = root / "range_intents.json"
                IntentGenerator(
                    campaign_plan=_range_campaign(root),
                    strategy_profile=_strategy(root, status="CANDIDATE"),
                    output_path=range_output,
                    report_path=root / "range_intents.md",
                    pair_charts_path=_pair_charts_with_direction(
                        root,
                        long_score=0.0484,
                        short_score=0.9456,
                        dominant_regime="TREND_DOWN",
                        m5_regime="RANGE",
                        m5_long_bias=0.76,
                        m5_short_bias=0.12,
                        regime_quantile="QUIET",
                        atr_pips=3.2,
                    ),
                    max_loss_jpy=140.0,
                ).run(snapshot_path=_snapshot(root, eur_bid=1.17110, eur_ask=1.17118))

                range_results = json.loads(range_output.read_text())["results"]
                self.assertTrue(
                    any(item["status"] == "LIVE_READY" for item in range_results),
                    f"range/fade entries should remain available, got {range_results}",
                )

                failure_output = root / "failure_intents.json"
                IntentGenerator(
                    campaign_plan=_trigger_campaign(root),
                    strategy_profile=_strategy(root, status="CANDIDATE"),
                    output_path=failure_output,
                    report_path=root / "failure_intents.md",
                    pair_charts_path=_pair_charts_with_direction(
                        root,
                        long_score=0.0484,
                        short_score=0.9456,
                        dominant_regime="TREND_DOWN",
                        m5_regime="TREND_DOWN",
                    ),
                    max_loss_jpy=500.0,
                ).run(snapshot_path=_snapshot(root))

                failure_results = json.loads(failure_output.read_text())["results"]
                self.assertEqual({item["intent"]["order_type"] for item in failure_results}, {"LIMIT", "STOP-ENTRY"})
                self.assertTrue(
                    any(item["status"] == "LIVE_READY" for item in failure_results),
                    f"breakout-failure trigger entries should remain available, got {failure_results}",
                )
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

    def test_blocks_trend_market_chase_when_m5_is_not_trending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                pair_charts_path=_pair_charts_with_direction(
                    root,
                    long_score=0.76,
                    short_score=0.19,
                    dominant_regime="TREND_UP",
                    m5_regime="RANGE",
                ),
                output_path=output,
                report_path=root / "intents.md",
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            stop_entry = next(item for item in payload["results"] if item["intent"]["order_type"] == "STOP-ENTRY")
            market = next(item for item in payload["results"] if item["intent"]["order_type"] == "MARKET")
            market_issue_codes = {issue["code"] for issue in market["risk_issues"]}

            self.assertEqual(stop_entry["status"], "LIVE_READY")
            self.assertEqual(market["status"], "DRY_RUN_BLOCKED")
            self.assertIn("TREND_MARKET_NOT_OPERATING_TREND", market_issue_codes)

    def test_trigger_receipt_required_does_not_create_market_chase_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_trigger_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            lane_ids = {item["lane_id"] for item in payload["results"]}
            order_types = {item["intent"]["order_type"] for item in payload["results"]}

            self.assertEqual(summary.generated, 2)
            self.assertEqual(order_types, {"LIMIT", "STOP-ENTRY"})
            self.assertEqual(
                lane_ids,
                {
                    "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE",
                    "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                },
            )
            self.assertFalse(any(lane_id.endswith(":MARKET") for lane_id in lane_ids))

    def test_breakout_failure_generates_retest_limit_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_breakout_failure_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            by_lane_id = {item["lane_id"]: item for item in payload["results"]}
            parent = "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE"

            self.assertEqual(summary.generated, 3)
            self.assertEqual(
                {lane_id: item["intent"]["order_type"] for lane_id, item in by_lane_id.items()},
                {
                    f"{parent}:LIMIT": "LIMIT",
                    parent: "STOP-ENTRY",
                    f"{parent}:MARKET": "MARKET",
                },
            )
            limit = by_lane_id[f"{parent}:LIMIT"]
            self.assertEqual(limit["intent"]["metadata"]["parent_lane_id"], parent)
            self.assertEqual(limit["intent"]["metadata"]["order_timing"], "PENDING_TRIGGER")

    def test_range_forecast_rebinds_breakout_failure_limit_to_forecast_box(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"
            forecast = SimpleNamespace(
                pair="EUR_USD",
                direction="RANGE",
                confidence=0.91,
                raw_confidence=0.93,
                current_price=1.17326,
                target_price=1.17480,
                invalidation_price=1.17200,
                range_low_price=1.17200,
                range_high_price=1.17480,
                range_width_pips=28.0,
                horizon_min=60,
                rationale_summary="measured range box",
                drivers_for=("M5 range compression",),
                drivers_against=("rail may fail",),
                component_scores={"UP": 0.1, "DOWN": 0.1, "RANGE": 0.8},
                market_support={},
                technical_context_v1=None,
            )

            with patch(
                "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                return_value=forecast,
            ):
                IntentGenerator(
                    campaign_plan=_breakout_failure_campaign(root),
                    strategy_profile=_strategy(root, status="CANDIDATE"),
                    output_path=output,
                    report_path=root / "intents.md",
                    pair_charts_path=_pair_charts(root),
                    max_loss_jpy=500.0,
                ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item
                for item in payload["results"]
                if item["lane_id"]
                == "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT"
            )
            intent = result["intent"]
            metadata = intent["metadata"]

            self.assertEqual(metadata["forecast_direction"], "RANGE")
            self.assertEqual(metadata["geometry_model"], "RANGE_RAIL_LIMIT")
            self.assertEqual(
                metadata["range_indicator_source"],
                "forecast_range_box",
            )
            self.assertAlmostEqual(metadata["range_support"], 1.17200)
            self.assertAlmostEqual(metadata["range_resistance"], 1.17480)
            self.assertGreaterEqual(intent["entry"], 1.17200)
            self.assertLess(abs(intent["entry"] - 1.17200), 0.0002)
            self.assertLess(intent["sl"], metadata["range_support"])
            self.assertGreater(intent["tp"], intent["entry"])
            self.assertLess(intent["tp"], metadata["range_resistance"])
            self.assertTrue(metadata["range_tp_is_inside_box"])
            self.assertTrue(metadata["range_sl_outside_box"])
            for lane_id in (
                "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE",
                "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:MARKET",
            ):
                alternate = next(
                    item for item in payload["results"] if item["lane_id"] == lane_id
                )
                alternate_metadata = alternate["intent"]["metadata"]
                self.assertEqual(
                    alternate_metadata["geometry_model"],
                    "ATR_SPREAD_STRUCTURE",
                )
                self.assertNotIn("range_support", alternate_metadata)
                self.assertNotIn("range_resistance", alternate_metadata)

    def test_sizes_repair_receipt_to_use_loss_budget_without_breaking_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = _campaign(root)
            strategy = _strategy(root)
            snapshot = _snapshot(root)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=campaign,
                strategy_profile=strategy,
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                max_loss_jpy=500.0,
            ).run(snapshot_path=snapshot)

            payload = json.loads(output.read_text())
            result = payload["results"][0]
            intent = result["intent"]
            # With ATR-derived geometry the SL distance is the larger of
            # 1*ATR(M5) and 6*spread, so unit count is bounded by the new
            # geometry; assert risk fits the cap rather than a fixed unit.
            self.assertLessEqual(result["risk_metrics"]["risk_jpy"], 500.0)
            self.assertGreater(intent["units"], 0)

    def test_generic_stop_sits_beyond_current_adverse_wick_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            result = next(item for item in json.loads(output.read_text())["results"] if item["intent"]["order_type"] == "STOP-ENTRY")
            intent = result["intent"]
            metadata = intent["metadata"]

            self.assertEqual(metadata["geometry_model"], "ATR_SPREAD_STRUCTURE")
            self.assertTrue(metadata["structural_stop_outside_level"])
            self.assertLess(intent["sl"], metadata["structural_stop_level"])
            self.assertLessEqual(result["risk_metrics"]["risk_jpy"], 500.0)

    def test_sizes_usd_quote_pair_from_snapshot_conversion_not_static_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                max_loss_jpy=1_000.0,
            ).run(snapshot_path=_snapshot(root, usd_jpy=300.0))

            result = json.loads(output.read_text())["results"][0]
            self.assertEqual(result["status"], "DRY_RUN_PASSED")
            # Unit count depends on conversion rate AND the (now ATR-derived)
            # SL distance — assert risk fits cap rather than fixed unit count.
            self.assertGreater(result["intent"]["units"], 0)
            self.assertLessEqual(result["risk_metrics"]["risk_jpy"], 1_000.0)

    def test_uses_campaign_runner_reward_risk_for_tp_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root, target_reward_risk=4.0),
                strategy_profile=_strategy(root),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                market_context_matrix_path=root / "market_context_matrix.json",
                data_root=root,
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            result = json.loads(output.read_text())["results"][0]
            self.assertEqual(result["intent"]["metadata"]["target_reward_risk"], 4.0)
            self.assertAlmostEqual(result["risk_metrics"]["reward_risk"], 4.0)

    def test_small_wave_attaches_structural_harvest_tp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts_tp_mode(root, adx=16.0, atr_percentile=0.2),
                market_context_matrix_path=root / "market_context_matrix.json",
                data_root=root,
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            result = json.loads(output.read_text())["results"][0]
            metadata = result["intent"]["metadata"]
            self.assertTrue(metadata["attach_take_profit_on_fill"])
            self.assertEqual(metadata["tp_execution_mode"], "ATTACHED_TECHNICAL_TP")
            self.assertEqual(metadata["tp_target_source"], "STRUCTURAL_HARVEST")
            self.assertIn("ADX", metadata["tp_attach_reason"])

    def test_directional_range_market_scalp_requires_edge_aligned_position(self) -> None:
        from quant_rabbit.models import Quote, Side
        from quant_rabbit.strategy.intent_generator import _directional_range_market_geometry

        quote = Quote(pair="EUR_USD", bid=1.15960, ask=1.15968)
        base_context = {
            "m5_regime": "RANGE",
            "m5_regime_quantile": "QUIET",
            "m5_long_bias": 0.2,
            "m5_short_bias": 0.7,
            "tf_regime_map": {"M5": {"range_position": 0.05}},
        }

        self.assertIsNone(
            _directional_range_market_geometry(
                "EUR_USD",
                Side.SHORT,
                quote,
                reward_risk=1.0,
                atr_pips=2.0,
                spread_pips=0.8,
                chart_context=base_context,
            )
        )
        upper_edge_short = {
            **base_context,
            "tf_regime_map": {"M5": {"range_position": 0.90}},
        }
        self.assertIsNotNone(
            _directional_range_market_geometry(
                "EUR_USD",
                Side.SHORT,
                quote,
                reward_risk=1.0,
                atr_pips=2.0,
                spread_pips=0.8,
                chart_context=upper_edge_short,
            )
        )
        lower_edge_long = {
            **base_context,
            "m5_long_bias": 0.7,
            "m5_short_bias": 0.2,
        }
        self.assertIsNotNone(
            _directional_range_market_geometry(
                "EUR_USD",
                Side.LONG,
                quote,
                reward_risk=1.0,
                atr_pips=2.0,
                spread_pips=0.8,
                chart_context=lower_edge_long,
            )
        )

    def test_attached_harvest_missing_structure_uses_fresh_live_floor_tp(self) -> None:
        from quant_rabbit.models import OrderType, Quote, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _take_profit_execution_plan

        tp, metadata = _take_profit_execution_plan(
            pair="EUR_USD",
            side=Side.SHORT,
            method=TradeMethod.BREAKOUT_FAILURE,
            order_type=OrderType.MARKET,
            quote=Quote(pair="EUR_USD", bid=1.16264, ask=1.16272),
            entry=1.16272,
            tp=1.14355,
            sl=1.16728,
            reward_risk=4.2,
            execution_regime="UNCLEAR",
            chart_context={"range_24h_sigma_multiple": 6.5},
            pair_chart={"pair": "EUR_USD", "views": [{"granularity": "M5", "indicators": {"atr_pips": 6.8}}]},
            atr_pips=6.8,
        )

        self.assertEqual(tp, 1.15814)
        self.assertEqual(metadata["tp_target_source"], "OPERATING_HARVEST_FLOOR")
        self.assertEqual(metadata["opportunity_mode"], "HARVEST")
        self.assertEqual(metadata["opportunity_mode_reward_risk"], metadata["virtual_take_profit_reward_risk"])
        self.assertEqual(metadata["tp_target_distance_pips"], 45.8)
        self.assertGreater(metadata["virtual_take_profit_reward_risk"], 1.0)
        self.assertIn("structural anchor missing", metadata["tp_target_reason"])
        self.assertIn("fresh_live_rr_floor", metadata["tp_target_reason"])

    def test_attached_harvest_fallback_skip_reason_surfaces_when_spread_floor_exceeds_atr_cap(self) -> None:
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Quote, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _method_context_issues, _take_profit_execution_plan

        tp, metadata = _take_profit_execution_plan(
            pair="EUR_USD",
            side=Side.SHORT,
            method=TradeMethod.BREAKOUT_FAILURE,
            order_type=OrderType.LIMIT,
            quote=Quote(pair="EUR_USD", bid=1.16250, ask=1.16550),
            entry=1.16400,
            tp=1.15000,
            sl=1.16900,
            reward_risk=2.8,
            execution_regime="UNCLEAR",
            chart_context={"range_24h_sigma_multiple": 4.0},
            pair_chart={"pair": "EUR_USD", "views": [{"granularity": "M5", "indicators": {"atr_pips": 6.0}}]},
            atr_pips=6.0,
        )

        self.assertEqual(tp, 1.15)
        self.assertEqual(metadata["tp_target_source"], "ATR_RR")
        self.assertIn("attached HARVEST fallback skipped", metadata["tp_target_reason"])
        self.assertIn("exceeds", metadata["tp_target_reason"])

        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.LIMIT,
            units=1000,
            entry=1.16400,
            tp=tp,
            sl=1.16900,
            thesis="fallback skip reason must be actionable",
            market_context=MarketContext(
                regime="UNCLEAR current; BREAKOUT_FAILURE campaign lane",
                narrative="",
                chart_story="",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="",
            ),
            metadata=metadata,
        )
        issue = next(
            issue
            for issue in _method_context_issues(intent)
            if issue["code"] == "HARVEST_TP_STRUCTURE_MISSING"
        )
        self.assertIn("fallback skipped", issue["message"])

    def test_range_rotation_far_rail_uses_operating_harvest_floor_tp(self) -> None:
        from quant_rabbit.models import OrderType, Quote, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _take_profit_execution_plan

        tp, metadata = _take_profit_execution_plan(
            pair="EUR_USD",
            side=Side.LONG,
            method=TradeMethod.RANGE_ROTATION,
            order_type=OrderType.LIMIT,
            quote=Quote(pair="EUR_USD", bid=1.17000, ask=1.17008),
            entry=1.17000,
            tp=1.18000,
            sl=1.16800,
            reward_risk=1.2,
            execution_regime="RANGE",
            chart_context={"m5_regime": "RANGE", "m5_regime_quantile": "QUIET"},
            pair_chart=None,
            atr_pips=3.0,
        )

        self.assertEqual(tp, 1.1712)
        self.assertEqual(metadata["tp_target_source"], "OPERATING_RANGE_HARVEST_FLOOR")
        self.assertEqual(metadata["opportunity_mode"], "HARVEST")
        self.assertEqual(metadata["tp_target_distance_pips"], 12.0)
        self.assertAlmostEqual(metadata["virtual_take_profit_reward_risk"], 0.6)
        self.assertIn("range rail target too far", metadata["tp_target_reason"])
        self.assertIn("range_rr_floor", metadata["tp_target_reason"])

    def test_recovery_hedge_missing_harvest_structure_uses_operating_floor_tp(self) -> None:
        from quant_rabbit.models import OrderType, Quote, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _take_profit_execution_plan

        tp, metadata = _take_profit_execution_plan(
            pair="EUR_USD",
            side=Side.LONG,
            method=TradeMethod.TREND_CONTINUATION,
            order_type=OrderType.STOP_ENTRY,
            quote=Quote(pair="EUR_USD", bid=1.16264, ask=1.16272),
            entry=1.16288,
            tp=1.17200,
            sl=1.16220,
            reward_risk=4.2,
            execution_regime="UNCLEAR",
            chart_context={"range_24h_sigma_multiple": 2.8},
            pair_chart={
                "pair": "EUR_USD",
                "views": [{"granularity": "M5", "indicators": {"atr_pips": 6.8}}],
            },
            atr_pips=6.8,
            hedge_recovery=True,
        )

        self.assertEqual(tp, 1.16329)
        self.assertEqual(metadata["tp_target_source"], "OPERATING_HARVEST_FLOOR")
        self.assertLessEqual(metadata["tp_target_distance_pips"], 27.2)
        self.assertIn("structural anchor missing", metadata["tp_target_reason"])

    def test_recovery_hedge_intent_passes_recovery_state_into_tp_plan(self) -> None:
        from quant_rabbit.models import (
            AccountSummary,
            BrokerPosition,
            BrokerSnapshot,
            OrderType,
            Owner,
            Quote,
            Side,
        )
        from quant_rabbit.strategy.intent_generator import _intent_from_lane, _method_context_issues

        now = datetime.now(timezone.utc)
        quote = Quote(pair="EUR_USD", bid=1.16264, ask=1.16272, timestamp_utc=now)
        snapshot = BrokerSnapshot(
            fetched_at_utc=now,
            positions=(
                BrokerPosition(
                    trade_id="trapped-short",
                    pair="EUR_USD",
                    side=Side.SHORT,
                    units=22_000,
                    entry_price=1.15800,
                    unrealized_pl_jpy=-23_000.0,
                    owner=Owner.TRADER,
                ),
            ),
            orders=(),
            quotes={
                "EUR_USD": quote,
                "USD_JPY": Quote(pair="USD_JPY", bid=156.64, ask=156.648, timestamp_utc=now),
            },
            account=AccountSummary(
                nav_jpy=200_000.0,
                balance_jpy=220_000.0,
                margin_used_jpy=120_000.0,
                margin_available_jpy=80_000.0,
                hedging_enabled=True,
                fetched_at_utc=now,
            ),
            home_conversions={"USD": 156.64},
        )
        intent = _intent_from_lane(
            {
                "desk": "trend_trader",
                "pair": "EUR_USD",
                "direction": "LONG",
                "method": "TREND_CONTINUATION",
                "adoption": "RISK_REPAIR_DRY_RUN",
                "campaign_role": "NOW_IF_REPAIRED",
                "reason": "reversal recovery hedge",
                "required_receipt": "dry-run under loss cap",
                "forecast_direction": "UP",
                "forecast_confidence": 0.51,
                "forecast_target_price": 1.1662,
                "forecast_invalidation_price": 1.1618,
            },
            quote,
            snapshot,
            max_loss_jpy=500.0,
            atr_pips=6.8,
            order_type_override=OrderType.STOP_ENTRY,
            regime_state="UNCLEAR",
            chart_context={"range_24h_sigma_multiple": 2.8},
            pair_chart={
                "pair": "EUR_USD",
                "views": [{"granularity": "M5", "indicators": {"atr_pips": 6.8}}],
            },
        )

        self.assertEqual(intent.metadata["position_intent"], "HEDGE")
        self.assertTrue(intent.metadata["hedge_recovery"])
        self.assertEqual(intent.metadata["hedge_timing_class"], "REVERSAL")
        self.assertEqual(intent.metadata["tp_target_source"], "OPERATING_HARVEST_FLOOR")
        self.assertNotIn(
            "HARVEST_TP_STRUCTURE_MISSING",
            {issue["code"] for issue in _method_context_issues(intent)},
        )

    def test_harvest_target_intent_overrides_high_reward_risk_opportunity_mode(self) -> None:
        from quant_rabbit.models import TradeMethod
        from quant_rabbit.strategy.intent_generator import _opportunity_mode_from_execution_plan

        mode, reason = _opportunity_mode_from_execution_plan(
            method=TradeMethod.BREAKOUT_FAILURE,
            target_intent="HARVEST",
            reward_risk=2.6,
        )

        self.assertEqual(mode, "HARVEST")
        self.assertEqual(reason, "tp_target_intent=HARVEST")

    def test_strong_trend_uses_runner_no_broker_tp_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts_tp_mode(root, adx=31.0, atr_percentile=0.8),
                market_context_matrix_path=root / "market_context_matrix.json",
                data_root=root,
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            result = json.loads(output.read_text())["results"][0]
            metadata = result["intent"]["metadata"]
            self.assertFalse(metadata["attach_take_profit_on_fill"])
            self.assertEqual(metadata["opportunity_mode"], "RUNNER")
            self.assertEqual(metadata["tp_execution_mode"], "RUNNER_NO_BROKER_TP")
            self.assertEqual(metadata["tp_target_source"], "STRUCTURAL_EXTEND")
            self.assertIn("qualifies as runner", metadata["tp_attach_reason"])

    def test_range_rotation_uses_rail_limit_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_range_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            result = json.loads(output.read_text())["results"][0]
            intent = result["intent"]
            metadata = intent["metadata"]
            self.assertEqual(result["status"], "LIVE_READY")
            self.assertEqual(intent["order_type"], "LIMIT")
            self.assertEqual(metadata["geometry_model"], "RANGE_RAIL_LIMIT")
            self.assertAlmostEqual(metadata["range_support"], 1.17100)
            self.assertAlmostEqual(metadata["range_resistance"], 1.17600)
            self.assertLess(abs(intent["entry"] - metadata["range_support"]), 0.0002)
            self.assertLess(intent["sl"], metadata["range_support"])
            self.assertTrue(metadata["range_tp_is_inside_box"])

    def test_range_rotation_limit_entry_stays_inside_box_when_spread_offset_is_wide(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_range_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root, eur_bid=1.17115, eur_ask=1.17300))

            result = json.loads(output.read_text())["results"][0]
            intent = result["intent"]
            metadata = intent["metadata"]
            support = metadata["range_support"]
            resistance = metadata["range_resistance"]
            entry_position = (intent["entry"] - support) / (resistance - support)
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertEqual(intent["order_type"], "LIMIT")
            self.assertEqual(metadata["geometry_model"], "RANGE_RAIL_LIMIT")
            self.assertGreaterEqual(intent["entry"], support)
            self.assertLess(intent["entry"], resistance)
            self.assertLessEqual(entry_position, 0.30)
            self.assertLess(intent["sl"], support)
            self.assertTrue(metadata["range_tp_is_inside_box"])
            self.assertIn("SPREAD_TOO_WIDE", issue_codes)

    def test_range_rotation_adds_market_reclaim_when_quote_is_at_rail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_range_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root, eur_bid=1.17110, eur_ask=1.17118))

            payload = json.loads(output.read_text())
            market = next(item for item in payload["results"] if item["lane_id"].endswith(":MARKET"))

            self.assertEqual(
                market["status"],
                "LIVE_READY",
                json.dumps(market, sort_keys=True),
            )
            self.assertEqual(market["intent"]["order_type"], "MARKET")
            self.assertEqual(market["intent"]["metadata"]["geometry_model"], "RANGE_RAIL_MARKET")
            self.assertTrue(market["intent"]["metadata"]["range_tp_is_inside_box"])

    def test_range_rotation_market_variant_blocks_when_quote_is_mid_box(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_range_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            market = next(item for item in payload["results"] if item["lane_id"].endswith(":MARKET"))
            issue_codes = {issue["code"] for issue in market["risk_issues"]}

            self.assertEqual(market["status"], "DRY_RUN_BLOCKED")
            self.assertIn("RANGE_MARKET_NOT_AT_RAIL", issue_codes)

    def test_stable_current_range_synthesizes_range_rotation_from_trend_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts_with_direction(
                    root,
                    long_score=0.55,
                    short_score=0.45,
                    dominant_regime="RANGE",
                    m5_regime="RANGE",
                    m5_long_bias=0.56,
                    m5_short_bias=0.44,
                    regime_quantile="QUIET",
                    atr_pips=3.2,
                    regime_state="RANGE",
                ),
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            range_limit = next(
                item
                for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )

            self.assertEqual(range_limit["status"], "LIVE_READY")
            self.assertEqual(range_limit["intent"]["order_type"], "LIMIT")
            self.assertEqual(range_limit["intent"]["metadata"]["geometry_model"], "RANGE_RAIL_LIMIT")

    def test_breakout_pending_range_does_not_synthesize_rotation_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts_with_direction(
                    root,
                    long_score=0.55,
                    short_score=0.45,
                    dominant_regime="RANGE",
                    m5_regime="RANGE",
                    m5_long_bias=0.56,
                    m5_short_bias=0.44,
                    regime_quantile="QUIET",
                    atr_pips=3.2,
                    regime_state="BREAKOUT_PENDING",
                ),
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            lane_ids = {item["lane_id"] for item in payload["results"]}

            self.assertNotIn("range_trader:EUR_USD:LONG:RANGE_ROTATION", lane_ids)

    def test_squeeze_range_does_not_synthesize_rotation_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts_with_direction(
                    root,
                    long_score=0.55,
                    short_score=0.45,
                    dominant_regime="RANGE",
                    m5_regime="RANGE",
                    m5_long_bias=0.56,
                    m5_short_bias=0.44,
                    regime_quantile="QUIET",
                    atr_pips=3.2,
                    regime_state="RANGE",
                    bb_squeeze=True,
                    atr_percentile=0.12,
                ),
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            lane_ids = {item["lane_id"] for item in payload["results"]}

            self.assertNotIn("range_trader:EUR_USD:LONG:RANGE_ROTATION", lane_ids)

    def test_range_forming_synthesizes_limit_only_rotation_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts_with_direction(
                    root,
                    long_score=0.55,
                    short_score=0.45,
                    dominant_regime="UNCLEAR",
                    m5_regime="TREND_WEAK",
                    m5_long_bias=0.56,
                    m5_short_bias=0.44,
                    regime_quantile="QUIET",
                    atr_pips=3.2,
                    regime_state="TREND_WEAK",
                    adx=18.0,
                    choppiness=55.0,
                    bb_width_percentile=0.30,
                ),
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            lane_ids = {item["lane_id"] for item in payload["results"]}
            range_limit = next(
                item
                for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )

            self.assertIn("range_trader:EUR_USD:LONG:RANGE_ROTATION", lane_ids)
            self.assertNotIn("range_trader:EUR_USD:LONG:RANGE_ROTATION:MARKET", lane_ids)
            self.assertEqual(range_limit["intent"]["order_type"], "LIMIT")
            self.assertEqual(range_limit["intent"]["metadata"]["range_phase"], "RANGE_FORMING")

    def test_confirmed_range_breakout_synthesizes_breakout_stop_entry_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts_with_direction(
                    root,
                    long_score=0.64,
                    short_score=0.36,
                    dominant_regime="RANGE",
                    m5_regime="RANGE",
                    m5_long_bias=0.66,
                    m5_short_bias=0.34,
                    regime_quantile="NORMAL",
                    atr_pips=3.2,
                    regime_state="BREAKOUT_PENDING",
                    close=1.17645,
                    trend_score=0.75,
                    breakout_score=0.85,
                ),
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            breakout = next(
                item
                for item in payload["results"]
                if item["lane_id"] == "trend_trader:EUR_USD:LONG:TREND_CONTINUATION"
            )

            self.assertEqual(breakout["intent"]["order_type"], "STOP-ENTRY")
            self.assertEqual(breakout["intent"]["metadata"]["range_phase"], "BREAKOUT_UP")
            self.assertEqual(breakout["intent"]["metadata"]["range_breakout_direction"], "UP")

    def test_existing_range_rotation_blocks_during_breakout_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_range_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts_with_direction(
                    root,
                    long_score=0.55,
                    short_score=0.45,
                    dominant_regime="RANGE",
                    m5_regime="RANGE",
                    m5_long_bias=0.56,
                    m5_short_bias=0.44,
                    regime_quantile="QUIET",
                    atr_pips=3.2,
                    regime_state="BREAKOUT_PENDING",
                ),
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            range_limit = next(
                item
                for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            issue_codes = {issue["code"] for issue in range_limit["risk_issues"]}

            self.assertEqual(range_limit["status"], "DRY_RUN_BLOCKED")
            self.assertIn("RANGE_PHASE_NOT_ROTATION", issue_codes)

    def test_low_vol_directional_range_market_uses_tight_risk_budgeted_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_range_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts_with_direction(
                    root,
                    long_score=0.76,
                    short_score=0.19,
                    dominant_regime="UNCLEAR",
                    m5_regime="RANGE",
                    m5_long_bias=0.72,
                    m5_short_bias=0.18,
                    regime_quantile="QUIET",
                    atr_pips=3.2,
                    range_support=1.1728,
                    range_resistance=1.1768,
                ),
                max_loss_jpy=140.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            market = next(item for item in payload["results"] if item["lane_id"].endswith(":MARKET"))
            issue_codes = {issue["code"] for issue in market["risk_issues"]}

            self.assertEqual(market["status"], "LIVE_READY")
            self.assertEqual(market["intent"]["metadata"]["geometry_model"], "RANGE_DIRECTIONAL_MARKET")
            self.assertEqual(market["intent"]["metadata"]["regime_state"], "RANGE")
            self.assertEqual(market["intent"]["metadata"]["regime_stop_widen_mult"], 1.0)
            self.assertGreaterEqual(market["intent"]["units"], 2000)
            self.assertLessEqual(market["risk_metrics"]["risk_jpy"], 140.0)
            self.assertNotIn("RANGE_MARKET_NOT_AT_RAIL", issue_codes)
            self.assertNotIn("FRESH_ENTRY_REWARD_RISK_NOT_POSITIVE", issue_codes)

    def test_range_direction_conflict_uses_m5_bias_not_aggregate_bias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_range_campaign(root, direction="SHORT"),
                strategy_profile=_strategy(root, status="CANDIDATE", direction="SHORT"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts_with_direction(
                    root,
                    long_score=0.82,
                    short_score=0.11,
                    dominant_regime="UNCLEAR",
                    m5_regime="RANGE",
                    m5_long_bias=0.12,
                    m5_short_bias=0.76,
                    regime_quantile="QUIET",
                    atr_pips=3.2,
                    range_support=1.1680,
                    range_resistance=1.1736,
                ),
                max_loss_jpy=140.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            issue_codes = {issue["code"] for item in payload["results"] for issue in item["risk_issues"]}
            market = next(item for item in payload["results"] if item["lane_id"].endswith(":MARKET"))

            self.assertEqual(market["status"], "LIVE_READY")
            self.assertNotIn("CHART_DIRECTION_CONFLICT", issue_codes)
            self.assertNotIn("FRESH_ENTRY_REWARD_RISK_NOT_POSITIVE", issue_codes)

    def test_open_position_with_per_trade_sized_risk_does_not_block_new_entry(self) -> None:
        # AGENT_CONTRACT §3.5 regression: portfolio cap is the WHOLE-DAY risk
        # budget, not the per-trade slice. A previous bug fed `max_loss_jpy`
        # (per-trade cap, e.g. 1050 JPY) into `max_portfolio_loss_jpy` so the
        # second any position opened, every fresh-entry candidate failed
        # `open_risk + candidate_risk > 1051` and the trader fell back to WAIT
        # for the rest of the day. With the fix, when no whole-day cap is
        # available (no daily_target_state.json on disk) the portfolio gate is
        # a no-op — it does NOT silently inherit the per-trade cap.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = _campaign(root)
            strategy = _strategy(root)
            snapshot = _snapshot_with_position(root)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=campaign,
                strategy_profile=strategy,
                output_path=output,
                report_path=root / "intents.md",
                max_loss_jpy=1050.0,
            ).run(snapshot_path=snapshot)

            payload = json.loads(output.read_text())
            result = payload["results"][0]
            issue_codes = {i["code"] for i in result["risk_issues"]}
            # Per-trade cap (1050) is still enforced on the new candidate via
            # `loss_cap`; portfolio cap (whole day) is absent in this test
            # because no ledger exists, so the portfolio gate is a no-op.
            self.assertNotIn("PORTFOLIO_LOSS_CAP_EXCEEDED", issue_codes)

    def test_remaining_daily_risk_budget_exhaustion_blocks_fresh_live_ready_entries(self) -> None:
        # Regression from 2026-06-12 live run: after a USD_CHF entry filled,
        # daily_target_state moved to RISK_BUDGET_EXHAUSTED, but
        # generate-intents still emitted fresh LIVE_READY lanes because it
        # sized from per_trade_risk_budget_jpy and ignored the remaining-day
        # circuit breaker.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            (data_root / "daily_target_state.json").write_text(
                json.dumps(
                    {
                        "status": "RISK_BUDGET_EXHAUSTED",
                        "daily_risk_budget_jpy": 10_000.0,
                        "per_trade_risk_budget_jpy": 1_000.0,
                        "remaining_risk_budget_jpy": 0.0,
                    }
                )
            )
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=data_root,
                max_loss_pct=0.5,
                risk_equity_jpy=200_000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            issue_codes = {
                issue["code"]
                for item in payload["results"]
                for issue in item["risk_issues"]
            }

            self.assertEqual(summary.live_ready, 0)
            self.assertTrue(payload["results"])
            self.assertTrue(all(item["status"] == "DRY_RUN_BLOCKED" for item in payload["results"]))
            self.assertIn("DAILY_RISK_BUDGET_EXHAUSTED", issue_codes)

    def test_self_improvement_profitability_p0_blocks_fresh_live_ready_generation(self) -> None:
        # Regression from qr-self-improvement-watch 2026-06-16: the verifier
        # and gateway rejected trades under the persistent profitability P0,
        # but generate-intents could still advertise fresh LIVE_READY lanes.
        # The dry-run layer should surface the same new-risk block directly.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            (data_root / "self_improvement_audit.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "findings": [
                            {
                                "priority": "P0",
                                "layer": "profitability",
                                "code": "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
                                "message": "profitability discipline has failed for 16 consecutive audit run(s)",
                                "evidence": {
                                    "current_streak": 16,
                                    "system_defect_evidence": {
                                        "profit_factor": 0.971,
                                        "expectancy_jpy": -7.22,
                                        "gateway_close_bleed_observation": {
                                            "gateway_bleed_basis": "contained_loss_erased_wins",
                                            "gateway_raw_net_jpy": -239.9306,
                                            "gateway_net_jpy": 96.157,
                                        },
                                    },
                                },
                            }
                        ],
                    }
                )
            )
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=data_root,
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            issue_codes = {
                issue["code"]
                for item in payload["results"]
                for issue in item["risk_issues"]
            }

            self.assertEqual(summary.live_ready, 0)
            self.assertTrue(payload["results"])
            self.assertTrue(all(item["status"] == "DRY_RUN_BLOCKED" for item in payload["results"]))
            self.assertIn("SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE", issue_codes)
            self.assertTrue(
                any(
                    "self-improvement profitability P0 blocks LIVE_READY" in blocker
                    for item in payload["results"]
                    for blocker in item["live_blockers"]
                )
            )
            self.assertTrue(
                any(
                    "24h_gateway_raw_net=-239.9306" in blocker
                    and "24h_gateway_net_ex_containment=96.157" in blocker
                    for item in payload["results"]
                    for blocker in item["live_blockers"]
                )
            )
            shadow = payload["self_improvement_p0_shadow_live_ready"]
            self.assertGreaterEqual(shadow["count"], 1)
            self.assertTrue(shadow["send_blocked"])
            self.assertTrue(
                any(
                    (item["intent"]["metadata"] or {}).get("self_improvement_p0_shadow_live_ready") is True
                    for item in payload["results"]
                    if item.get("intent")
                )
            )

    def test_inactive_position_guardian_p0_blocks_fresh_live_ready_generation(self) -> None:
        # Regression from USD_JPY 472792: once self-improvement proves the fast
        # profit-capture guardian is inactive, generate-intents must stop
        # advertising fresh LIVE_READY risk before GPT/gateway have to reject it.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            (data_root / "self_improvement_audit.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "findings": [
                            {
                                "priority": "P0",
                                "layer": "execution_quality",
                                "code": "POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE",
                                "message": (
                                    "position guardian is required but inactive; TP-progress profit "
                                    "cannot be captured between full trader cycles"
                                ),
                                "evidence": {
                                    "target_open": True,
                                    "live_ready_lanes": 1,
                                    "guardian": {
                                        "required": True,
                                        "active": False,
                                        "active_source": "plist_missing",
                                        "launchd_loaded": False,
                                    },
                                },
                            }
                        ],
                    }
                )
            )
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=data_root,
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            issue_codes = {
                issue["code"]
                for item in payload["results"]
                for issue in item["risk_issues"]
            }

            self.assertEqual(summary.live_ready, 0)
            self.assertTrue(payload["results"])
            self.assertTrue(all(item["status"] == "DRY_RUN_BLOCKED" for item in payload["results"]))
            self.assertIn("POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE", issue_codes)
            self.assertTrue(
                any(
                    "position guardian inactive P0 blocks LIVE_READY" in blocker
                    and "profit-capture monitoring must be active" in blocker
                    for item in payload["results"]
                    for blocker in item["live_blockers"]
                )
            )

    def test_recovered_position_guardian_heartbeat_clears_stale_self_improvement_p0(self) -> None:
        # A refreshed guardian heartbeat must clear stale self-improvement P0
        # contamination inside intent diagnostics. Live sends still re-check
        # guardian health in the gateway; this only keeps old support evidence
        # from masking the real lane blockers after the guardian has recovered.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            (data_root / "self_improvement_audit.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": (
                            datetime.now(timezone.utc) - timedelta(minutes=20)
                        ).isoformat(),
                        "findings": [
                            {
                                "priority": "P0",
                                "layer": "execution_quality",
                                "code": "POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE",
                                "message": (
                                    "position guardian is required but inactive; TP-progress profit "
                                    "cannot be captured between full trader cycles"
                                ),
                                "evidence": {
                                    "target_open": True,
                                    "live_ready_lanes": 1,
                                    "guardian": {
                                        "required": True,
                                        "active": False,
                                        "active_source": "env+heartbeat",
                                        "env_active": "0",
                                        "heartbeat_fresh": False,
                                        "launchd_loaded": None,
                                    },
                                },
                            }
                        ],
                    }
                )
            )
            heartbeat = data_root / "position_guardian.json"
            heartbeat.write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "NO_POSITION",
                        "sent": False,
                    }
                )
            )
            execution = data_root / "position_guardian_execution.json"
            output = root / "intents.json"

            with patch.dict(
                os.environ,
                {
                    "QR_REQUIRE_POSITION_GUARDIAN_ACTIVE": "1",
                    "QR_POSITION_GUARDIAN_ACTIVE": "1",
                    "QR_POSITION_GUARDIAN_REQUIRE_HEARTBEAT": "1",
                    "QR_POSITION_GUARDIAN_HEARTBEAT": str(heartbeat),
                    "QR_POSITION_GUARDIAN_EXECUTION": str(execution),
                    "QR_POSITION_GUARDIAN_HEARTBEAT_MAX_AGE_SECONDS": "120",
                },
                clear=False,
            ):
                summary = IntentGenerator(
                    campaign_plan=_range_campaign(root),
                    strategy_profile=_strategy(root, status="CANDIDATE"),
                    output_path=output,
                    report_path=root / "intents.md",
                    pair_charts_path=_pair_charts(root),
                    data_root=data_root,
                    max_loss_jpy=500.0,
                ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            issue_codes = {
                issue["code"]
                for item in payload["results"]
                for issue in item["risk_issues"]
            }

            self.assertGreaterEqual(summary.live_ready, 1)
            self.assertNotIn("POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE", issue_codes)
            self.assertFalse(
                any(
                    "position guardian inactive P0 blocks LIVE_READY" in blocker
                    for item in payload["results"]
                    for blocker in item["live_blockers"]
                )
            )

    def test_self_improvement_profitability_p0_allows_tp_proven_harvest_repair_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_exact_vehicle_rotation_wins(root)
            (root / "self_improvement_audit.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "findings": [
                            {
                                "priority": "P0",
                                "layer": "profitability",
                                "code": "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
                                "message": "market-close leakage is still negative",
                                "evidence": {
                                    "current_streak": 65,
                                    "system_defect_evidence": {
                                        "profit_factor": 0.788,
                                        "expectancy_jpy": -54.04,
                                    },
                                },
                            }
                        ],
                    }
                )
            )
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 210,
                            "avg_win_jpy": 600.0,
                            "avg_loss_jpy": 1100.0,
                            "payoff_ratio": 0.545,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                        "by_exit_reason": {
                            "TAKE_PROFIT_ORDER": {
                                "trades": 93,
                                "wins": 93,
                                "losses": 0,
                                "avg_win_jpy": 504.0,
                                "avg_loss_jpy": 0.0,
                                "expectancy_jpy_per_trade": 504.0,
                            },
                            "MARKET_ORDER_TRADE_CLOSE": {
                                "trades": 84,
                                "wins": 13,
                                "losses": 71,
                                "avg_win_jpy": 218.4,
                                "avg_loss_jpy": 1095.5,
                                "expectancy_jpy_per_trade": -892.1,
                            },
                        },
                        **_capture_scoped_tp_payload(),
                    }
                )
            )
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_range_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertGreaterEqual(summary.live_ready, 1)
            self.assertEqual(result["status"], "LIVE_READY")
            self.assertTrue(metadata["self_improvement_p0_repair_live_ready"])
            self.assertEqual(metadata["self_improvement_p0_repair_mode"], "TP_HARVEST_REPAIR")
            self.assertIn("SELF_IMPROVEMENT_P0_PROFITABILITY_REPAIR_MODE", issue_codes)
            self.assertNotIn("SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE", issue_codes)
            self.assertEqual(result["live_blockers"], [])

    def test_limit_live_ready_requires_limit_vehicle_tp_proof_when_ledger_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-07-01T02:30:00+00:00",
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 210,
                            "avg_win_jpy": 600.0,
                            "avg_loss_jpy": 1100.0,
                            "payoff_ratio": 0.545,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                        "by_exit_reason": {
                            "TAKE_PROFIT_ORDER": {
                                "trades": 93,
                                "wins": 93,
                                "losses": 0,
                                "avg_win_jpy": 504.0,
                                "avg_loss_jpy": 0.0,
                                "expectancy_jpy_per_trade": 504.0,
                            },
                            "MARKET_ORDER_TRADE_CLOSE": {
                                "trades": 84,
                                "wins": 13,
                                "losses": 71,
                                "avg_win_jpy": 218.4,
                                "avg_loss_jpy": 1095.5,
                                "expectancy_jpy_per_trade": -892.1,
                            },
                        },
                        **_capture_scoped_tp_payload(
                            method="BREAKOUT_FAILURE",
                            trades=20,
                            wins=20,
                            losses=0,
                            avg_win_jpy=591.5,
                            avg_loss_jpy=0.0,
                            expectancy_jpy_per_trade=591.5,
                        ),
                    }
                )
            )
            _write_exact_vehicle_take_profit_closes(
                root,
                lane_id="failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:STOP",
                pair="EUR_USD",
                side="LONG",
                entry_reason="STOP_ORDER",
                count=20,
                realized_pl_jpy=591.5,
            )
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_breakout_failure_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts_with_direction(
                    root,
                    long_score=0.70,
                    short_score=0.30,
                    dominant_regime="BREAKOUT_FAILURE",
                    m5_regime="BREAKOUT_FAILURE",
                ),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertEqual(result["status"], "DRY_RUN_BLOCKED")
            self.assertEqual(metadata["capture_take_profit_scope"], "PAIR_SIDE_METHOD_VEHICLE")
            self.assertEqual(
                metadata["capture_take_profit_scope_key"],
                "EUR_USD|LONG|BREAKOUT_FAILURE|LIMIT|TAKE_PROFIT_ORDER",
            )
            self.assertEqual(metadata["capture_take_profit_trades"], 0)
            self.assertEqual(metadata["broad_capture_take_profit_trades"], 20)
            self.assertTrue(metadata["broad_capture_take_profit_not_used_as_exact_vehicle_proof"])
            self.assertNotIn("self_improvement_p0_repair_live_ready", metadata)
            self.assertIn(POSITIVE_ROTATION_LIVE_BLOCK_CODE, issue_codes)

    def test_exact_vehicle_tp_metrics_use_net_pl_including_financing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_exact_vehicle_metric_trade(
                root,
                trade_id="financing-loss",
                fill_lane_id="failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                realized_pl_jpy=100.0,
                standalone_financing_jpy=-200.0,
            )

            metrics = _exact_vehicle_take_profit_metrics(root / "execution_ledger.db")

            self.assertIsNotNone(metrics)
            assert metrics is not None
            row = metrics[("EUR_USD", "LONG", "BREAKOUT_FAILURE", "LIMIT")]
            self.assertEqual(row["trades"], 1)
            self.assertEqual(row["wins"], 0)
            self.assertEqual(row["losses"], 1)
            self.assertEqual(row["net_jpy"], -100.0)
            self.assertEqual(row["expectancy_jpy_per_trade"], -100.0)
            self.assertEqual(row["avg_loss_jpy"], 100.0)

    def test_intent_persists_exact_nondivisible_tp_net_before_rounded_averages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 30,
                            "avg_win_jpy": 100.0,
                            "avg_loss_jpy": 500.0,
                            "payoff_ratio": 0.2,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                    }
                )
            )
            _write_exact_vehicle_take_profit_closes(
                root,
                lane_id="failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:STOP",
                pair="EUR_USD",
                side="LONG",
                entry_reason="STOP_ORDER",
                count=6,
                realized_pl_jpy=[100.0] * 5 + [100.0001],
            )
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_breakout_failure_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts_with_direction(
                    root,
                    long_score=0.70,
                    short_score=0.30,
                    dominant_regime="BREAKOUT_FAILURE",
                    m5_regime="BREAKOUT_FAILURE",
                ),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            metrics = _exact_vehicle_take_profit_metrics(
                root / "execution_ledger.db"
            )
            self.assertIsNotNone(metrics)
            assert metrics is not None
            exact = metrics[("EUR_USD", "LONG", "BREAKOUT_FAILURE", "STOP")]
            payload = json.loads(output.read_text())
            result = next(
                item
                for item in payload["results"]
                if item.get("intent", {}).get("order_type") == "STOP-ENTRY"
            )
            metadata = result["intent"]["metadata"]

            self.assertEqual(exact["net_jpy"], 600.0001)
            self.assertEqual(exact["expectancy_jpy_per_trade"], 100.0)
            self.assertEqual(metadata["capture_take_profit_net_jpy"], 600.0001)
            self.assertEqual(
                metadata["capture_take_profit_net_jpy"],
                exact["net_jpy"],
            )

    def test_exact_vehicle_net_metrics_include_non_tp_exit_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_exact_vehicle_metric_trade(
                root,
                trade_id="mixed-exits",
                fill_lane_id="failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                realized_pl_jpy=200.0,
                partial_realized_pl_jpy=-50.0,
                partial_exit_reason="MARKET_ORDER_TRADE_CLOSE",
            )

            metrics = _exact_vehicle_net_metrics(root / "execution_ledger.db")

            self.assertIsNotNone(metrics)
            assert metrics is not None
            row = metrics[("EUR_USD", "LONG", "BREAKOUT_FAILURE", "LIMIT")]
            self.assertEqual(row["exit_scope"], "ALL_AUDITED_EXITS")
            self.assertEqual(row["trades"], 1)
            self.assertEqual(row["wins"], 1)
            self.assertEqual(row["net_jpy"], 150.0)
            self.assertEqual(row["expectancy_jpy_per_trade"], 150.0)

    def test_positive_capture_still_attaches_exact_vehicle_all_exit_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "POSITIVE_EXPECTANCY",
                        "overall": {
                            "trades": 20,
                            "avg_win_jpy": 100.0,
                            "avg_loss_jpy": 0.0,
                        },
                    }
                )
            )
            _write_exact_vehicle_take_profit_closes(
                root,
                lane_id="failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:STOP",
                pair="EUR_USD",
                side="LONG",
                entry_reason="STOP_ORDER",
                count=20,
                realized_pl_jpy=100.0,
            )
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_breakout_failure_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts_with_direction(
                    root,
                    long_score=0.70,
                    short_score=0.30,
                    dominant_regime="BREAKOUT_FAILURE",
                    m5_regime="BREAKOUT_FAILURE",
                ),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item
                for item in payload["results"]
                if item.get("intent", {}).get("order_type") == "STOP-ENTRY"
            )
            metadata = result["intent"]["metadata"]
            self.assertEqual(
                metadata["capture_exact_vehicle_net_scope_key"],
                "EUR_USD|LONG|BREAKOUT_FAILURE|STOP|ALL_AUDITED_EXITS",
            )
            self.assertEqual(metadata["capture_exact_vehicle_net_trades"], 20)
            self.assertEqual(metadata["capture_exact_vehicle_net_wins"], 20)
            self.assertEqual(metadata["capture_exact_vehicle_net_jpy"], 2000.0)
            self.assertEqual(
                metadata["capture_exact_vehicle_net_metrics_source"],
                "data/execution_ledger.db:exact_vehicle_net",
            )

    def test_exact_vehicle_tp_metrics_never_use_close_lane_for_manual_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_exact_vehicle_metric_trade(
                root,
                trade_id="manual-entry",
                fill_lane_id="",
                close_lane_id="failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                realized_pl_jpy=300.0,
            )

            metrics = _exact_vehicle_take_profit_metrics(root / "execution_ledger.db")

            self.assertEqual(metrics, {})

    def test_exact_vehicle_tp_metrics_allow_matching_entry_gateway_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_exact_vehicle_metric_trade(
                root,
                trade_id="gateway-attributed-entry",
                fill_lane_id="",
                gateway_lane_id="failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                realized_pl_jpy=250.0,
                standalone_financing_jpy=-10.0,
            )

            metrics = _exact_vehicle_take_profit_metrics(root / "execution_ledger.db")

            self.assertIsNotNone(metrics)
            assert metrics is not None
            row = metrics[("EUR_USD", "LONG", "BREAKOUT_FAILURE", "LIMIT")]
            self.assertEqual(row["trades"], 1)
            self.assertEqual(row["wins"], 1)
            self.assertEqual(row["net_jpy"], 240.0)

    def test_exact_vehicle_tp_metrics_reject_entry_truth_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lane_id = "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT"
            _write_exact_vehicle_metric_trade(
                root,
                trade_id="pair-mismatch",
                fill_lane_id=lane_id,
                fill_pair="GBP_USD",
            )
            _write_exact_vehicle_metric_trade(
                root,
                trade_id="side-mismatch",
                fill_lane_id=lane_id,
                fill_side="SHORT",
            )
            _write_exact_vehicle_metric_trade(
                root,
                trade_id="vehicle-mismatch",
                fill_lane_id=lane_id,
                fill_reason="STOP_ORDER",
            )
            _write_exact_vehicle_metric_trade(
                root,
                trade_id="gateway-pair-mismatch",
                fill_lane_id=lane_id,
                gateway_lane_id=lane_id,
                gateway_pair="GBP_USD",
            )

            metrics = _exact_vehicle_take_profit_metrics(root / "execution_ledger.db")

            self.assertIsNone(metrics)

    def test_exact_vehicle_tp_metrics_exclude_partial_only_trade(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_exact_vehicle_metric_trade(
                root,
                trade_id="partial-only",
                fill_lane_id="failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                outcome_event_type="TRADE_REDUCED",
                realized_pl_jpy=150.0,
            )

            metrics = _exact_vehicle_take_profit_metrics(root / "execution_ledger.db")

            self.assertEqual(metrics, {})

    def test_exact_vehicle_tp_metrics_fail_closed_on_invalid_financing_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_exact_vehicle_metric_trade(
                root,
                trade_id="valid-close",
                fill_lane_id="failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
            )
            with sqlite3.connect(root / "execution_ledger.db") as conn:
                conn.execute(
                    """
                    INSERT INTO execution_events (
                      ts_utc, event_type, financing_jpy, raw_json
                    ) VALUES (?, 'OANDA_TRANSACTION', ?, ?)
                    """,
                    ("2026-07-01T00:03:00+00:00", -5.0, "{not-json"),
                )

            metrics = _exact_vehicle_take_profit_metrics(root / "execution_ledger.db")

            self.assertIsNone(metrics)

    def test_exact_vehicle_tp_metrics_require_financing_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "execution_ledger.db"
            with sqlite3.connect(db) as conn:
                conn.execute(
                    """
                    CREATE TABLE execution_events (
                      ts_utc TEXT,
                      event_type TEXT,
                      lane_id TEXT,
                      pair TEXT,
                      side TEXT,
                      order_id TEXT,
                      trade_id TEXT,
                      exit_reason TEXT,
                      realized_pl_jpy REAL,
                      raw_json TEXT
                    )
                    """
                )

            metrics = _exact_vehicle_take_profit_metrics(db)

            self.assertIsNone(metrics)

    def test_exact_vehicle_tp_metrics_allow_fill_vehicle_without_lane_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_exact_vehicle_metric_trade(
                root,
                trade_id="legacy-lane-limit",
                fill_lane_id="failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE",
                fill_reason="LIMIT_ORDER",
                realized_pl_jpy=200.0,
            )

            metrics = _exact_vehicle_take_profit_metrics(root / "execution_ledger.db")

            self.assertIsNotNone(metrics)
            assert metrics is not None
            self.assertEqual(
                metrics[("EUR_USD", "LONG", "BREAKOUT_FAILURE", "LIMIT")]["trades"],
                1,
            )

    def test_exact_vehicle_tp_metrics_exclude_operator_manual_and_late_gateway(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_exact_vehicle_metric_trade(
                root,
                trade_id="explicit-manual",
                fill_lane_id="operator_manual:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                realized_pl_jpy=300.0,
            )
            _write_exact_vehicle_metric_trade(
                root,
                trade_id="late-gateway",
                fill_lane_id="",
                realized_pl_jpy=300.0,
            )
            with sqlite3.connect(root / "execution_ledger.db") as conn:
                conn.execute(
                    """
                    INSERT INTO execution_events (
                      ts_utc, event_type, lane_id, pair, side, units, order_id, trade_id,
                      exit_reason, realized_pl_jpy, financing_jpy, raw_json
                    ) VALUES (?, 'ORDER_ACCEPTED', ?, ?, ?, ?, ?, ?, ?, NULL, 0.0, ?)
                    """,
                    (
                        "2026-07-01T02:00:00+00:00",
                        "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                        "EUR_USD",
                        "LONG",
                        1000,
                        "entry-late-gateway",
                        "late-gateway",
                        "LIMIT_ORDER",
                        json.dumps({"type": "LIMIT_ORDER"}),
                    ),
                )

            self.assertEqual(
                _exact_vehicle_take_profit_metrics(root / "execution_ledger.db"),
                {},
            )

    def test_exact_vehicle_tp_metrics_require_pure_tp_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_exact_vehicle_metric_trade(
                root,
                trade_id="market-reduction-then-tp",
                fill_lane_id="failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                realized_pl_jpy=200.0,
                partial_realized_pl_jpy=50.0,
                partial_exit_reason="MARKET_ORDER_TRADE_CLOSE",
            )

            self.assertEqual(
                _exact_vehicle_take_profit_metrics(root / "execution_ledger.db"),
                {},
            )

    def test_exact_vehicle_tp_metrics_allocate_offsetting_zero_total_financing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_exact_vehicle_metric_trade(
                root,
                trade_id="system-financed",
                fill_lane_id="failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                realized_pl_jpy=100.0,
                standalone_financing_jpy=-200.0,
            )
            with sqlite3.connect(root / "execution_ledger.db") as conn:
                financing = {
                    "type": "DAILY_FINANCING",
                    "financing": "0.0",
                    "positionFinancings": [
                        {
                            "openTradeFinancings": [
                                {"tradeID": "system-financed", "financing": "-200.0"},
                                {"tradeID": "manual-financed", "financing": "200.0"},
                            ]
                        }
                    ],
                }
                conn.execute(
                    """
                    UPDATE execution_events
                    SET financing_jpy=0.0, raw_json=?
                    WHERE event_type='OANDA_TRANSACTION'
                    """,
                    (json.dumps(financing),),
                )

            metrics = _exact_vehicle_take_profit_metrics(root / "execution_ledger.db")

            self.assertIsNotNone(metrics)
            assert metrics is not None
            row = metrics[("EUR_USD", "LONG", "BREAKOUT_FAILURE", "LIMIT")]
            self.assertEqual(row["trades"], 1)
            self.assertEqual(row["wins"], 0)
            self.assertEqual(row["losses"], 1)
            self.assertEqual(row["net_jpy"], -100.0)

    def test_month_scale_residual_group_blocks_matching_harvest_repair_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_profitability_p0_and_negative_capture(root)
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 210,
                            "avg_win_jpy": 600.0,
                            "avg_loss_jpy": 1100.0,
                            "payoff_ratio": 0.545,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                        "by_exit_reason": {
                            "TAKE_PROFIT_ORDER": {
                                "trades": 93,
                                "wins": 93,
                                "losses": 0,
                                "avg_win_jpy": 504.0,
                                "avg_loss_jpy": 0.0,
                                "expectancy_jpy_per_trade": 504.0,
                            },
                            "MARKET_ORDER_TRADE_CLOSE": {
                                "trades": 84,
                                "wins": 13,
                                "losses": 71,
                                "avg_win_jpy": 218.4,
                                "avg_loss_jpy": 1095.5,
                                "expectancy_jpy_per_trade": -892.1,
                            },
                        },
                        **_capture_scoped_tp_payload(),
                    }
                )
            )
            _write_month_scale_residual_acceptance(
                root,
                pair="EUR_USD",
                side="LONG",
                method="RANGE_ROTATION",
            )
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_range_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertEqual(summary.live_ready, 0)
            self.assertEqual(result["status"], "DRY_RUN_BLOCKED")
            self.assertTrue(metadata["month_scale_residual_loss_repair_blocked"])
            self.assertEqual(
                metadata["month_scale_residual_loss_group"]["repair_replay_pl_jpy"],
                -2333.8215,
            )
            self.assertNotIn("self_improvement_p0_repair_live_ready", metadata)
            self.assertIn(MONTH_SCALE_RESIDUAL_LOSS_REPAIR_BLOCK_CODE, issue_codes)
            self.assertIn("SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE", issue_codes)
            self.assertIn(
                MONTH_SCALE_RESIDUAL_LOSS_REPAIR_BLOCK_CODE,
                result["live_blocker_codes"],
            )

    def test_month_scale_residual_group_blocks_matching_normal_live_ready_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_month_scale_residual_acceptance(
                root,
                pair="EUR_USD",
                side="LONG",
                method="RANGE_ROTATION",
            )
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_range_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item
                for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertEqual(summary.live_ready, 0)
            self.assertEqual(result["status"], "DRY_RUN_BLOCKED")
            self.assertTrue(metadata["month_scale_residual_loss_repair_blocked"])
            self.assertNotIn("self_improvement_p0_repair_live_ready", metadata)
            self.assertIn(MONTH_SCALE_RESIDUAL_LOSS_REPAIR_BLOCK_CODE, issue_codes)
            self.assertIn(
                MONTH_SCALE_RESIDUAL_LOSS_REPAIR_BLOCK_CODE,
                result["live_blocker_codes"],
            )

    def test_month_scale_entry_quality_residual_group_uses_specific_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_month_scale_residual_acceptance(
                root,
                pair="EUR_USD",
                side="LONG",
                method="RANGE_ROTATION",
                residual_scope="ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
            )
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_range_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item
                for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}
            issue = next(
                item
                for item in result["risk_issues"]
                if item["code"] == MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCK_CODE
            )

            self.assertEqual(summary.live_ready, 0)
            self.assertEqual(result["status"], "DRY_RUN_BLOCKED")
            self.assertEqual(
                metadata["month_scale_residual_loss_group"]["residual_scope"],
                "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
            )
            self.assertIn(MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCK_CODE, issue_codes)
            self.assertIn(
                MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCK_CODE,
                result["live_blocker_codes"],
            )
            self.assertNotIn(MONTH_SCALE_RESIDUAL_LOSS_REPAIR_BLOCK_CODE, issue_codes)
            self.assertIn(
                "without a TP-progress production-gate profit candidate",
                issue["message"],
            )

    def test_month_scale_gate_uses_canonical_global_top_ten_before_typed_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical_pairs = [
                "AUD_USD",
                "NZD_CAD",
                "NZD_USD",
                "EUR_CHF",
                "GBP_USD",
                "GBP_CHF",
                "EUR_GBP",
                "EUR_JPY",
                "USD_JPY",
                "EUR_USD",
            ]
            canonical_groups = [
                {
                    "pair": pair,
                    "side": "LONG",
                    "method": "RANGE_ROTATION",
                    "exit_reason": "STOP_LOSS_ORDER",
                    "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
                    "loss_closes": 1,
                    "actual_pl_jpy": float(-2000 + index * 100),
                    "repair_replay_pl_jpy": float(-2000 + index * 100),
                }
                for index, pair in enumerate(canonical_pairs)
            ]
            typed_tp_diagnostic = {
                "pair": "AUD_NZD",
                "side": "SHORT",
                "method": "RANGE_ROTATION",
                "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
                "residual_scope": "TP_PROGRESS_DIAGNOSTIC_BLOCKED",
                "loss_closes": 1,
                "actual_pl_jpy": -50.0,
                "repair_replay_pl_jpy": -50.0,
            }
            replay_metrics = {
                "top_repair_replay_residual_groups": canonical_groups,
                "top_tp_progress_repair_residual_groups": [typed_tp_diagnostic],
                "top_entry_quality_residual_groups": canonical_groups,
            }
            (root / "profitability_acceptance.json").write_text(
                json.dumps(
                    {
                        "status": "PROFITABILITY_ACCEPTANCE_BLOCKED",
                        "findings": [
                            {
                                "priority": "P0",
                                "code": "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE",
                                "message": "month-scale replay remains negative",
                                "evidence": replay_metrics,
                            }
                        ],
                        "metrics": {"profit_capture_replay_repair": replay_metrics},
                    }
                )
            )
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_range_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item
                for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertTrue(metadata["month_scale_residual_loss_repair_blocked"])
            self.assertEqual(
                metadata["month_scale_residual_loss_group"]["repair_replay_pl_jpy"],
                -1100.0,
            )
            self.assertIn(MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCK_CODE, issue_codes)

    def test_month_scale_gate_deduplicates_shape_and_keeps_worst_source_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            finding_group = {
                "pair": "EUR_USD",
                "side": "LONG",
                "method": "RANGE_ROTATION",
                "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
                "repair_replay_pl_jpy": -100.0,
            }
            metrics_group = {
                **finding_group,
                "repair_replay_pl_jpy": -200.0,
            }
            (root / "profitability_acceptance.json").write_text(
                json.dumps(
                    {
                        "status": "PROFITABILITY_ACCEPTANCE_BLOCKED",
                        "findings": [
                            {
                                "priority": "P0",
                                "code": "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE",
                                "message": "month-scale replay remains negative",
                                "evidence": {
                                    "top_repair_replay_residual_groups": [finding_group]
                                },
                            }
                        ],
                        "metrics": {
                            "profit_capture_replay_repair": {
                                "top_repair_replay_residual_groups": [metrics_group]
                            }
                        },
                    }
                )
            )

            issue = _profitability_acceptance_month_residual_issue(root)

            self.assertIsNotNone(issue)
            assert issue is not None
            segments = issue["blocked_profitability_segments"]
            self.assertEqual(len(segments), 1)
            self.assertEqual(segments[0]["pair"], "EUR_USD")
            self.assertEqual(segments[0]["net_jpy"], -200.0)

    def test_month_scale_timing_artifact_blocks_matching_harvest_repair_before_acceptance_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_profitability_p0_and_negative_capture(root)
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 210,
                            "avg_win_jpy": 600.0,
                            "avg_loss_jpy": 1100.0,
                            "payoff_ratio": 0.545,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                        "by_exit_reason": {
                            "TAKE_PROFIT_ORDER": {
                                "trades": 93,
                                "wins": 93,
                                "losses": 0,
                                "avg_win_jpy": 504.0,
                                "avg_loss_jpy": 0.0,
                                "expectancy_jpy_per_trade": 504.0,
                            },
                            "MARKET_ORDER_TRADE_CLOSE": {
                                "trades": 84,
                                "wins": 13,
                                "losses": 71,
                                "avg_win_jpy": 218.4,
                                "avg_loss_jpy": 1095.5,
                                "expectancy_jpy_per_trade": -892.1,
                            },
                        },
                        **_capture_scoped_tp_payload(),
                    }
                )
            )
            _write_month_scale_residual_timing_audit(
                root,
                pair="EUR_USD",
                side="LONG",
                method="RANGE_ROTATION",
            )
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_range_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertEqual(summary.live_ready, 0)
            self.assertEqual(result["status"], "DRY_RUN_BLOCKED")
            self.assertTrue(metadata["month_scale_residual_loss_repair_blocked"])
            self.assertEqual(
                metadata["month_scale_residual_loss_group"]["repair_replay_pl_jpy"],
                -2333.8215,
            )
            self.assertNotIn("self_improvement_p0_repair_live_ready", metadata)
            self.assertIn(MONTH_SCALE_RESIDUAL_LOSS_REPAIR_BLOCK_CODE, issue_codes)
            self.assertIn(
                MONTH_SCALE_RESIDUAL_LOSS_REPAIR_BLOCK_CODE,
                result["live_blocker_codes"],
            )

    def test_month_scale_residual_group_does_not_block_different_harvest_repair_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_exact_vehicle_rotation_wins(root)
            _write_profitability_p0_and_negative_capture(root)
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 210,
                            "avg_win_jpy": 600.0,
                            "avg_loss_jpy": 1100.0,
                            "payoff_ratio": 0.545,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                        "by_exit_reason": {
                            "TAKE_PROFIT_ORDER": {
                                "trades": 93,
                                "wins": 93,
                                "losses": 0,
                                "avg_win_jpy": 504.0,
                                "avg_loss_jpy": 0.0,
                                "expectancy_jpy_per_trade": 504.0,
                            },
                            "MARKET_ORDER_TRADE_CLOSE": {
                                "trades": 84,
                                "wins": 13,
                                "losses": 71,
                                "avg_win_jpy": 218.4,
                                "avg_loss_jpy": 1095.5,
                                "expectancy_jpy_per_trade": -892.1,
                            },
                        },
                        **_capture_scoped_tp_payload(),
                    }
                )
            )
            _write_month_scale_residual_acceptance(
                root,
                pair="GBP_USD",
                side="LONG",
                method="BREAKOUT_FAILURE",
                repair_replay_pl_jpy=-2981.8961,
            )
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_range_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertGreaterEqual(summary.live_ready, 1)
            self.assertEqual(result["status"], "LIVE_READY")
            self.assertNotIn("month_scale_residual_loss_repair_blocked", metadata)
            self.assertTrue(metadata["self_improvement_p0_repair_live_ready"])
            self.assertNotIn(MONTH_SCALE_RESIDUAL_LOSS_REPAIR_BLOCK_CODE, issue_codes)
            self.assertNotIn("SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE", issue_codes)

    def test_month_scale_residual_group_message_names_matching_segment_not_worst_global_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_month_scale_residual_acceptance(
                root,
                pair="EUR_USD",
                side="LONG",
                method="RANGE_ROTATION",
                repair_replay_pl_jpy=-2333.8215,
                extra_groups=[
                    {
                        "pair": "GBP_USD",
                        "side": "LONG",
                        "method": "BREAKOUT_FAILURE",
                        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
                        "loss_closes": 1,
                        "repair_replay_pl_jpy": -2981.8961,
                        "block_reasons": {"BELOW_TP_PROGRESS_GATE": 1},
                        "examples": [
                            {
                                "trade_id": "472070",
                                "lane_id": "test:GBP_USD:LONG:BREAKOUT_FAILURE",
                                "repair_replay_pl_jpy": -2981.8961,
                            }
                        ],
                    }
                ],
            )
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_range_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item
                for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            issue = next(
                item
                for item in result["risk_issues"]
                if item["code"] == MONTH_SCALE_RESIDUAL_LOSS_REPAIR_BLOCK_CODE
            )

            self.assertIn("pair=EUR_USD", issue["message"])
            self.assertIn("method=RANGE_ROTATION", issue["message"])
            self.assertNotIn("pair=GBP_USD", issue["message"])

    def test_month_scale_residual_metrics_without_p0_does_not_block_harvest_repair_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_exact_vehicle_rotation_wins(root)
            _write_profitability_p0_and_negative_capture(root)
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 210,
                            "avg_win_jpy": 600.0,
                            "avg_loss_jpy": 1100.0,
                            "payoff_ratio": 0.545,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                        "by_exit_reason": {
                            "TAKE_PROFIT_ORDER": {
                                "trades": 93,
                                "wins": 93,
                                "losses": 0,
                                "avg_win_jpy": 504.0,
                                "avg_loss_jpy": 0.0,
                                "expectancy_jpy_per_trade": 504.0,
                            },
                            "MARKET_ORDER_TRADE_CLOSE": {
                                "trades": 84,
                                "wins": 13,
                                "losses": 71,
                                "avg_win_jpy": 218.4,
                                "avg_loss_jpy": 1095.5,
                                "expectancy_jpy_per_trade": -892.1,
                            },
                        },
                        **_capture_scoped_tp_payload(),
                    }
                )
            )
            _write_month_scale_residual_metrics_only(
                root,
                pair="EUR_USD",
                side="LONG",
                method="RANGE_ROTATION",
            )
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_range_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertGreaterEqual(summary.live_ready, 1)
            self.assertEqual(result["status"], "LIVE_READY")
            self.assertNotIn("month_scale_residual_loss_repair_blocked", metadata)
            self.assertTrue(metadata["self_improvement_p0_repair_live_ready"])
            self.assertNotIn(MONTH_SCALE_RESIDUAL_LOSS_REPAIR_BLOCK_CODE, issue_codes)
            self.assertNotIn("SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE", issue_codes)

    def test_oanda_firepower_seed_requires_local_tp_scope_for_profitability_p0_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_profitability_p0_and_negative_capture(root)
            _write_oanda_campaign_firepower_report(
                root,
                pair="EUR_USD",
                side="LONG",
                exit_shape="tp2_sl1",
            )
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_oanda_seed_range_campaign(root, side="LONG"),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            market_result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION:MARKET"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}
            market_issue_codes = {issue["code"] for issue in market_result["risk_issues"]}

            self.assertEqual(summary.live_ready, 0)
            self.assertEqual(result["status"], "DRY_RUN_BLOCKED")
            self.assertEqual(metadata["capture_take_profit_scope"], "MISSING_SCOPED")
            self.assertFalse(metadata["positive_rotation_live_ready"])
            self.assertEqual(
                metadata["positive_rotation_mode"],
                POSITIVE_ROTATION_OANDA_CAMPAIGN_FIREPOWER_MODE,
            )
            self.assertTrue(metadata["positive_rotation_oanda_campaign_firepower_vehicle_match"])
            self.assertTrue(metadata["positive_rotation_oanda_campaign_audit_only"])
            self.assertTrue(metadata["positive_rotation_oanda_campaign_local_tp_proof_required"])
            self.assertTrue(metadata["positive_rotation_minimum_floor_reachable"])
            self.assertEqual(
                metadata["positive_rotation_minimum_floor_reach_basis"],
                "OANDA_CAMPAIGN_FIREPOWER_MATCHING_VEHICLE",
            )
            self.assertFalse(metadata["positive_rotation_oanda_campaign_live_permission"])
            self.assertNotIn("self_improvement_p0_repair_live_ready", metadata)
            self.assertIn(OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED_CODE, issue_codes)
            self.assertIn("SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE", issue_codes)
            self.assertIn(
                OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED_CODE,
                result["live_blocker_codes"],
            )
            self.assertEqual(market_result["status"], "DRY_RUN_BLOCKED")
            self.assertIn(POSITIVE_ROTATION_LIVE_BLOCK_CODE, market_issue_codes)
            self.assertIn("SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE", market_issue_codes)

    def test_oanda_firepower_seed_stays_capacity_only_under_guardian_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_acceptance_style_p0_and_negative_capture(root)
            _write_oanda_campaign_firepower_report(
                root,
                pair="EUR_USD",
                side="LONG",
                exit_shape="tp2_sl1",
                weighted_return_pct=0.64,
                observed_attempts_per_active_day=22.0,
            )
            (root / "daily_target_state.json").write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "start_balance_jpy": 100000.0,
                        "remaining_minimum_jpy": 4000.0,
                        "remaining_target_jpy": 9000.0,
                        "target_trades_per_day": 10,
                    }
                )
            )
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_stamp_campaign_generated_at(
                    _oanda_seed_range_campaign(root, side="LONG")
                ),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertEqual(summary.live_ready, 0)
            self.assertEqual(result["status"], "DRY_RUN_BLOCKED")
            self.assertEqual(metadata["loss_asymmetry_guard_mode"], "CAP_AVG_WIN")
            self.assertFalse(metadata["loss_asymmetry_guard_relaxed"])
            self.assertEqual(metadata["loss_asymmetry_guard_effective_max_loss_jpy"], 600.0)
            self.assertFalse(metadata["positive_rotation_live_ready"])
            self.assertEqual(
                metadata["positive_rotation_mode"],
                POSITIVE_ROTATION_OANDA_CAMPAIGN_FIREPOWER_MODE,
            )
            self.assertTrue(metadata["positive_rotation_oanda_campaign_audit_only"])
            self.assertFalse(metadata["positive_rotation_oanda_campaign_live_permission"])
            self.assertTrue(metadata["positive_rotation_oanda_campaign_local_tp_proof_required"])
            self.assertEqual(
                metadata["positive_rotation_minimum_floor_reach_basis"],
                "OANDA_CAMPAIGN_FIREPOWER_CURRENT_RISK_UNDERPOWERED",
            )
            self.assertNotIn("self_improvement_p0_repair_live_ready", metadata)
            self.assertIn(OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED_CODE, issue_codes)
            self.assertIn("POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE", issue_codes)
            self.assertIn(
                "POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE",
                result["live_blocker_codes"],
            )

    def test_oanda_firepower_seed_cannot_use_normal_cap_when_scaled_pace_reaches_floor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_profitability_p0_and_negative_capture(root)
            _write_oanda_campaign_firepower_report(
                root,
                pair="EUR_USD",
                side="LONG",
                exit_shape="tp2_sl1",
                weighted_return_pct=0.64,
                observed_attempts_per_active_day=22.0,
            )
            (root / "daily_target_state.json").write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "start_balance_jpy": 100000.0,
                        "remaining_minimum_jpy": 4000.0,
                        "remaining_target_jpy": 9000.0,
                        "target_trades_per_day": 10,
                    }
                )
            )
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_stamp_campaign_generated_at(
                    _oanda_seed_range_campaign(root, side="LONG")
                ),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertEqual(summary.live_ready, 0)
            self.assertEqual(result["status"], "DRY_RUN_BLOCKED")
            self.assertEqual(metadata["loss_asymmetry_guard_mode"], "CAP_AVG_WIN")
            self.assertFalse(metadata["loss_asymmetry_guard_relaxed"])
            self.assertEqual(metadata["loss_asymmetry_guard_loss_cap_jpy"], 600.0)
            self.assertEqual(metadata["loss_asymmetry_guard_effective_max_loss_jpy"], 600.0)
            self.assertLessEqual(
                result["risk_metrics"]["risk_jpy"],
                metadata["loss_asymmetry_guard_loss_cap_jpy"],
            )
            self.assertNotIn("positive_rotation_oanda_campaign_normal_cap_relaxed", metadata)
            self.assertFalse(metadata["positive_rotation_live_ready"])
            self.assertTrue(metadata["positive_rotation_oanda_campaign_audit_only"])
            self.assertFalse(metadata["positive_rotation_oanda_campaign_live_permission"])
            self.assertTrue(metadata["positive_rotation_oanda_campaign_local_tp_proof_required"])
            self.assertNotIn("LOSS_ASYMMETRY_GUARD_EXCEEDED", issue_codes)
            self.assertEqual(
                metadata["positive_rotation_minimum_floor_reach_basis"],
                "OANDA_CAMPAIGN_FIREPOWER_CURRENT_RISK_UNDERPOWERED",
            )
            self.assertNotIn("self_improvement_p0_repair_live_ready", metadata)
            self.assertIn(OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED_CODE, issue_codes)
            self.assertIn("SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE", issue_codes)

    def test_oanda_firepower_seed_does_not_relax_when_daily_floor_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_profitability_p0_and_negative_capture(root)
            _write_oanda_campaign_firepower_report(
                root,
                pair="EUR_USD",
                side="LONG",
                exit_shape="tp2_sl1",
                weighted_return_pct=0.64,
                observed_attempts_per_active_day=22.0,
            )
            (root / "daily_target_state.json").write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "start_balance_jpy": 100000.0,
                        "remaining_target_jpy": 9000.0,
                        "target_trades_per_day": 10,
                    }
                )
            )
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_stamp_campaign_generated_at(
                    _oanda_seed_range_campaign(root, side="LONG")
                ),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]

            self.assertEqual(metadata["loss_asymmetry_guard_mode"], "CAP_AVG_WIN")
            self.assertFalse(metadata["loss_asymmetry_guard_relaxed"])
            self.assertEqual(metadata["loss_asymmetry_guard_effective_max_loss_jpy"], 600.0)
            self.assertNotIn(
                "positive_rotation_oanda_campaign_normal_cap_relaxed",
                metadata,
            )

    def test_oanda_firepower_seed_scales_daily_floor_to_current_risk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_profitability_p0_and_negative_capture(root)
            _write_oanda_campaign_firepower_report(
                root,
                pair="EUR_USD",
                side="LONG",
                exit_shape="tp2_sl1",
            )
            (root / "daily_target_state.json").write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "start_balance_jpy": 173958.1237,
                        "remaining_minimum_jpy": 10655.9438,
                        "remaining_target_jpy": 19353.8438,
                        "target_trades_per_day": 30,
                    }
                )
            )
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_stamp_campaign_generated_at(
                    _oanda_seed_range_campaign(root, side="LONG")
                ),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=417.4,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}
            firepower_issue = next(
                issue
                for issue in result["risk_issues"]
                if issue["code"] == OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED_CODE
            )

            self.assertEqual(summary.live_ready, 0)
            self.assertEqual(result["status"], "DRY_RUN_BLOCKED")
            self.assertEqual(
                metadata["positive_rotation_mode"],
                POSITIVE_ROTATION_OANDA_CAMPAIGN_FIREPOWER_MODE,
            )
            expected_current_risk_pct = result["risk_metrics"]["risk_jpy"] / 173958.1237 * 100.0
            self.assertEqual(
                metadata["positive_rotation_oanda_campaign_current_risk_basis"],
                "ACTUAL_ORDER_RISK",
            )
            self.assertAlmostEqual(
                metadata["positive_rotation_oanda_campaign_current_risk_jpy"],
                result["risk_metrics"]["risk_jpy"],
                places=4,
            )
            self.assertAlmostEqual(
                metadata["sizing_actual_risk_jpy"],
                result["risk_metrics"]["risk_jpy"],
                places=4,
            )
            self.assertLess(metadata["sizing_actual_risk_cap_utilization"], 1.0)
            self.assertEqual(
                metadata["sizing_actual_anchor_contract"],
                "ACTUAL_SIZING_ANCHOR_V1",
            )
            sizing_anchor_path = (
                root
                / "sizing_actual_receipts"
                / f"{metadata['sizing_actual_anchor_sha256']}.json"
            )
            self.assertTrue(sizing_anchor_path.exists())
            sizing_anchor = json.loads(sizing_anchor_path.read_text())
            self.assertEqual(
                sizing_anchor["intent"]["units"],
                result["intent"]["units"],
            )
            self.assertEqual(
                sizing_anchor["risk_metrics"]["risk_jpy"],
                result["risk_metrics"]["risk_jpy"],
            )
            self.assertAlmostEqual(
                metadata["positive_rotation_oanda_campaign_current_risk_pct"],
                expected_current_risk_pct,
                places=5,
            )
            self.assertLess(
                metadata["positive_rotation_oanda_campaign_current_risk_pct"],
                0.239943,
            )
            self.assertEqual(
                metadata["positive_rotation_oanda_campaign_current_risk_estimated_return_basis"],
                "MATCHING_VEHICLE",
            )
            self.assertAlmostEqual(
                metadata[
                    "positive_rotation_oanda_campaign_current_risk_estimated_return_pct_per_active_day"
                ],
                metadata[
                    "positive_rotation_oanda_campaign_current_risk_matching_vehicle_estimated_return_pct_per_active_day"
                ],
                places=6,
            )
            self.assertGreater(
                metadata[
                    "positive_rotation_oanda_campaign_current_risk_aggregate_estimated_return_pct_per_active_day"
                ],
                metadata[
                    "positive_rotation_oanda_campaign_current_risk_estimated_return_pct_per_active_day"
                ],
            )
            self.assertFalse(
                metadata["positive_rotation_oanda_campaign_current_risk_minimum_floor_reachable"]
            )
            self.assertLess(
                metadata["positive_rotation_oanda_campaign_current_risk_estimated_return_pct_per_active_day"],
                metadata["positive_rotation_oanda_campaign_remaining_minimum_pct"],
            )
            self.assertGreater(
                metadata["positive_rotation_oanda_campaign_current_risk_required_minimum_trades"],
                30,
            )
            self.assertFalse(metadata["positive_rotation_minimum_floor_reachable"])
            self.assertEqual(
                metadata["positive_rotation_minimum_floor_reach_basis"],
                "OANDA_CAMPAIGN_FIREPOWER_CURRENT_RISK_UNDERPOWERED",
            )
            self.assertIn(OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED_CODE, issue_codes)
            self.assertEqual(firepower_issue["severity"], "BLOCK")
            self.assertIn(
                OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED_CODE,
                result["live_blocker_codes"],
            )
            self.assertIn("SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE", issue_codes)
            self.assertNotIn("self_improvement_p0_repair_live_ready", metadata)

    def test_oanda_firepower_current_risk_uses_matching_vehicle_not_aggregate_route(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_profitability_p0_and_negative_capture(root)
            _write_oanda_campaign_firepower_report(
                root,
                pair="EUR_USD",
                side="LONG",
                exit_shape="tp2_sl1",
                aggregate_return_pct=30.0,
                matching_return_pct=2.0,
                weighted_return_pct=0.64,
            )
            (root / "daily_target_state.json").write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "start_balance_jpy": 173958.1237,
                        "remaining_minimum_jpy": 3000.0,
                        "remaining_target_jpy": 19353.8438,
                        "target_trades_per_day": 30,
                    }
                )
            )
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_stamp_campaign_generated_at(
                    _oanda_seed_range_campaign(root, side="LONG")
                ),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=417.4,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]

            self.assertEqual(result["status"], "DRY_RUN_BLOCKED")
            self.assertEqual(
                metadata["positive_rotation_oanda_campaign_current_risk_estimated_return_basis"],
                "MATCHING_VEHICLE",
            )
            self.assertGreater(
                metadata[
                    "positive_rotation_oanda_campaign_current_risk_aggregate_estimated_return_pct_per_active_day"
                ],
                metadata["positive_rotation_oanda_campaign_remaining_minimum_pct"],
            )
            self.assertLess(
                metadata[
                    "positive_rotation_oanda_campaign_current_risk_estimated_return_pct_per_active_day"
                ],
                metadata["positive_rotation_oanda_campaign_remaining_minimum_pct"],
            )
            self.assertFalse(
                metadata["positive_rotation_oanda_campaign_current_risk_minimum_floor_reachable"]
            )
            self.assertFalse(metadata["positive_rotation_minimum_floor_reachable"])
            self.assertEqual(
                metadata["positive_rotation_minimum_floor_reach_basis"],
                "OANDA_CAMPAIGN_FIREPOWER_CURRENT_RISK_UNDERPOWERED",
            )
            issue_codes = {issue["code"] for issue in result["risk_issues"]}
            self.assertIn(OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED_CODE, issue_codes)
            self.assertIn(
                OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED_CODE,
                result["live_blocker_codes"],
            )

    def test_oanda_firepower_current_risk_rejects_aggregate_without_matching_vehicle(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "daily_target_state.json").write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "start_balance_jpy": 173958.1237,
                        "remaining_minimum_jpy": 3000.0,
                        "remaining_target_jpy": 19353.8438,
                        "target_trades_per_day": 30,
                    }
                )
            )
            metadata = {
                "sizing_actual_risk_jpy": 1000.0,
                "max_loss_jpy": 1000.0,
                "positive_rotation_oanda_campaign_per_trade_risk_pct_lens": 1.0,
                "positive_rotation_oanda_campaign_estimated_return_pct_per_active_day": 30.0,
                "positive_rotation_oanda_campaign_matching_vehicle_estimated_return_pct_per_active_day": None,
                "positive_rotation_oanda_campaign_weighted_return_pct_per_trade_at_risk_lens": 0.64,
                "positive_rotation_oanda_campaign_firepower_vehicle_match": False,
            }

            metrics = _annotate_oanda_campaign_current_risk_firepower(
                metadata,
                data_root=root,
            )

            self.assertIsNotNone(metrics)
            assert metrics is not None
            self.assertEqual(metrics["estimated_return_basis"], "NO_MATCHING_VEHICLE")
            self.assertIsNone(metrics["estimated_return_pct_per_active_day"])
            self.assertGreater(metrics["aggregate_estimated_return_pct_per_active_day"], 0)
            self.assertFalse(metrics["minimum_floor_reachable"])
            self.assertFalse(
                metadata["positive_rotation_oanda_campaign_current_risk_minimum_floor_reachable"]
            )

    def test_oanda_repair_not_blocked_by_same_segment_stop_loss_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_profitability_p0_with_matching_worst_segment(
                root,
                close_provenance_net_jpy={
                    "STOP_LOSS_ORDER": -280.8,
                    "GATEWAY_TRADE_CLOSE_SENT": 7.2,
                },
            )
            _write_oanda_campaign_firepower_report(
                root,
                pair="EUR_USD",
                side="LONG",
                exit_shape="tp2_sl1",
            )
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_stamp_campaign_generated_at(
                    _oanda_seed_range_campaign(root, side="LONG")
                ),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertEqual(result["status"], "DRY_RUN_BLOCKED")
            self.assertNotIn("self_improvement_p0_repair_live_ready", metadata)
            self.assertTrue(metadata["positive_rotation_oanda_campaign_firepower_vehicle_match"])
            self.assertIn(OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED_CODE, issue_codes)
            self.assertIn("SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE", issue_codes)
            self.assertIn("SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE", result["live_blocker_codes"])

    def test_oanda_repair_blocks_same_day_same_lane_loss_recycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_profitability_p0_with_matching_worst_segment(
                root,
                close_provenance_net_jpy={"STOP_LOSS_ORDER": -280.8},
            )
            _write_oanda_campaign_firepower_report(
                root,
                pair="EUR_USD",
                side="LONG",
                exit_shape="tp2_sl1",
            )
            _write_same_day_lane_outcomes(
                root,
                outcomes=(("2026-06-22T05:00:00+00:00", -280.8),),
            )
            _stamp_capture_economics(root, "2026-06-22T05:45:00+00:00")
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_stamp_campaign_generated_at(
                    _oanda_seed_range_campaign(root, side="LONG")
                ),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(
                snapshot_path=_snapshot(
                    root,
                    fetched_at_utc="2026-06-22T06:00:00+00:00",
                    quote_timestamp_utc="2026-06-22T06:00:00+00:00",
                )
            )

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertEqual(result["status"], "DRY_RUN_BLOCKED")
            self.assertNotIn("self_improvement_p0_repair_live_ready", metadata)
            self.assertNotIn("self_improvement_p0_repair_recent_lane_loss", metadata)
            self.assertNotIn(SELF_IMPROVEMENT_PROFITABILITY_P0_REPAIR_RECENT_LOSS_CODE, issue_codes)
            self.assertIn(OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED_CODE, issue_codes)
            self.assertIn("SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE", issue_codes)
            self.assertIn(
                OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED_CODE,
                result["live_blocker_codes"],
            )

    def test_oanda_repair_same_day_lane_win_resets_recent_loss_recycle_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_profitability_p0_with_matching_worst_segment(
                root,
                close_provenance_net_jpy={"STOP_LOSS_ORDER": -280.8},
            )
            _write_oanda_campaign_firepower_report(
                root,
                pair="EUR_USD",
                side="LONG",
                exit_shape="tp2_sl1",
            )
            _write_same_day_lane_outcomes(
                root,
                outcomes=(
                    ("2026-06-22T05:00:00+00:00", -280.8),
                    ("2026-06-22T05:30:00+00:00", 385.0),
                ),
            )
            _stamp_capture_economics(root, "2026-06-22T05:45:00+00:00")
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_stamp_campaign_generated_at(
                    _oanda_seed_range_campaign(root, side="LONG")
                ),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(
                snapshot_path=_snapshot(
                    root,
                    fetched_at_utc="2026-06-22T06:00:00+00:00",
                    quote_timestamp_utc="2026-06-22T06:00:00+00:00",
                )
            )

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertEqual(result["status"], "DRY_RUN_BLOCKED")
            self.assertNotIn("self_improvement_p0_repair_live_ready", metadata)
            self.assertNotIn("self_improvement_p0_repair_recent_lane_loss", metadata)
            self.assertNotIn(SELF_IMPROVEMENT_PROFITABILITY_P0_REPAIR_RECENT_LOSS_CODE, issue_codes)
            self.assertIn(OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED_CODE, issue_codes)
            self.assertIn("SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE", issue_codes)

    def test_oanda_repair_still_blocked_by_same_segment_gateway_close_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_profitability_p0_with_matching_worst_segment(
                root,
                close_provenance_net_jpy={"GATEWAY_TRADE_CLOSE_SENT": -280.8},
            )
            _write_oanda_campaign_firepower_report(
                root,
                pair="EUR_USD",
                side="LONG",
                exit_shape="tp2_sl1",
            )
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_stamp_campaign_generated_at(
                    _oanda_seed_range_campaign(root, side="LONG")
                ),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertEqual(result["status"], "DRY_RUN_BLOCKED")
            self.assertNotIn("self_improvement_p0_repair_live_ready", metadata)
            self.assertNotIn("self_improvement_p0_shadow_live_ready", metadata)
            self.assertIn(OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED_CODE, issue_codes)
            self.assertIn("SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE", issue_codes)
            self.assertIn("SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE", result["live_blocker_codes"])

    def test_oanda_firepower_seed_repair_requires_matching_vehicle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_profitability_p0_and_negative_capture(root)
            _write_oanda_campaign_firepower_report(root, pair="EUR_USD", side="SHORT")
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_oanda_seed_range_campaign(root, side="LONG"),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertEqual(summary.live_ready, 0)
            self.assertEqual(result["status"], "DRY_RUN_BLOCKED")
            self.assertFalse(metadata["positive_rotation_oanda_campaign_firepower_vehicle_match"])
            self.assertNotIn("positive_rotation_live_ready", metadata)
            self.assertNotIn("self_improvement_p0_repair_live_ready", metadata)
            self.assertIn(POSITIVE_ROTATION_LIVE_BLOCK_CODE, issue_codes)
            self.assertIn("SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE", issue_codes)

    def test_self_improvement_profitability_p0_repair_allows_range_m5_tie_against_broader_lean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_exact_vehicle_rotation_wins(root)
            (root / "self_improvement_audit.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "findings": [
                            {
                                "priority": "P0",
                                "layer": "profitability",
                                "code": "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
                                "message": "market-close leakage is still negative",
                                "evidence": {
                                    "current_streak": 65,
                                    "system_defect_evidence": {
                                        "profit_factor": 0.788,
                                        "expectancy_jpy": -54.04,
                                        "worst_segments": [
                                            {
                                                "pair": "NZD_CAD",
                                                "side": "SHORT",
                                                "method": "RANGE_ROTATION",
                                                "trades": 2,
                                                "net_jpy": -2044.45,
                                                "trade_ids": ["472312", "472380"],
                                            }
                                        ],
                                    },
                                },
                            }
                        ],
                    }
                )
            )
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 210,
                            "avg_win_jpy": 600.0,
                            "avg_loss_jpy": 1100.0,
                            "payoff_ratio": 0.545,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                        "by_exit_reason": {
                            "TAKE_PROFIT_ORDER": {
                                "trades": 93,
                                "wins": 93,
                                "losses": 0,
                                "avg_win_jpy": 504.0,
                                "avg_loss_jpy": 0.0,
                                "expectancy_jpy_per_trade": 504.0,
                            },
                            "MARKET_ORDER_TRADE_CLOSE": {
                                "trades": 84,
                                "wins": 13,
                                "losses": 71,
                                "avg_win_jpy": 218.4,
                                "avg_loss_jpy": 1095.5,
                                "expectancy_jpy_per_trade": -892.1,
                            },
                        },
                        **_capture_scoped_tp_payload(),
                    }
                )
            )
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_range_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts_with_direction(
                    root,
                    long_score=0.3049,
                    short_score=0.5771,
                    dominant_regime="RANGE",
                    m5_regime="RANGE",
                    m5_long_bias=0.4286,
                    m5_short_bias=0.4286,
                ),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertGreaterEqual(summary.live_ready, 1)
            self.assertEqual(result["status"], "LIVE_READY")
            self.assertEqual(metadata["chart_direction_bias"], "SHORT")
            self.assertEqual(metadata["m5_long_bias"], 0.4286)
            self.assertEqual(metadata["m5_short_bias"], 0.4286)
            self.assertTrue(metadata["self_improvement_p0_repair_live_ready"])
            self.assertNotIn("CHART_DIRECTION_CONFLICT", issue_codes)
            self.assertNotIn("SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE", issue_codes)
            self.assertIn("SELF_IMPROVEMENT_P0_PROFITABILITY_REPAIR_MODE", issue_codes)

    def test_self_improvement_profitability_p0_blocks_harvest_repair_on_named_worst_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "self_improvement_audit.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "findings": [
                            {
                                "priority": "P0",
                                "layer": "profitability",
                                "code": "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
                                "message": "market-close leakage is still negative",
                                "evidence": {
                                    "current_streak": 60,
                                    "system_defect_evidence": {
                                        "profit_factor": 0.833,
                                        "expectancy_jpy": -40.10,
                                        "worst_segments": [
                                            {
                                                "pair": "EUR_USD",
                                                "side": "LONG",
                                                "method": "RANGE_ROTATION",
                                                "trades": 2,
                                                "net_jpy": -2044.45,
                                                "trade_ids": ["472312", "472380"],
                                            }
                                        ],
                                    },
                                },
                            }
                        ],
                    }
                )
            )
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 210,
                            "avg_win_jpy": 600.0,
                            "avg_loss_jpy": 1100.0,
                            "payoff_ratio": 0.545,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                        "by_exit_reason": {
                            "TAKE_PROFIT_ORDER": {
                                "trades": 93,
                                "wins": 93,
                                "losses": 0,
                                "avg_win_jpy": 504.0,
                                "avg_loss_jpy": 0.0,
                                "expectancy_jpy_per_trade": 504.0,
                            },
                            "MARKET_ORDER_TRADE_CLOSE": {
                                "trades": 84,
                                "wins": 13,
                                "losses": 71,
                                "avg_win_jpy": 218.4,
                                "avg_loss_jpy": 1095.5,
                                "expectancy_jpy_per_trade": -892.1,
                            },
                        },
                    }
                )
            )
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_range_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertEqual(summary.live_ready, 0)
            self.assertEqual(result["status"], "DRY_RUN_BLOCKED")
            self.assertNotIn("self_improvement_p0_repair_live_ready", metadata)
            self.assertIn("SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE", issue_codes)
            self.assertTrue(
                any(
                    "inspect=data/execution_ledger.db worst_segment[pair=EUR_USD" in blocker
                    for blocker in result["live_blockers"]
                )
            )

    def test_self_improvement_profitability_p0_blocks_harvest_repair_on_any_loss_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "self_improvement_audit.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "findings": [
                            {
                                "priority": "P0",
                                "layer": "profitability",
                                "code": "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
                                "message": "market-close leakage is still negative",
                                "evidence": {
                                    "current_streak": 60,
                                    "system_defect_evidence": {
                                        "profit_factor": 0.833,
                                        "expectancy_jpy": -40.10,
                                        "worst_segments": [
                                            {
                                                "pair": "NZD_USD",
                                                "side": "LONG",
                                                "method": "RANGE_ROTATION",
                                                "trades": 1,
                                                "net_jpy": -1380.80,
                                                "trade_ids": ["472743"],
                                            },
                                            {
                                                "pair": "EUR_USD",
                                                "side": "LONG",
                                                "method": "RANGE_ROTATION",
                                                "trades": 2,
                                                "net_jpy": -2044.45,
                                                "trade_ids": ["472312", "472380"],
                                            },
                                        ],
                                    },
                                },
                            }
                        ],
                    }
                )
            )
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 210,
                            "avg_win_jpy": 600.0,
                            "avg_loss_jpy": 1100.0,
                            "payoff_ratio": 0.545,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                        "by_exit_reason": {
                            "TAKE_PROFIT_ORDER": {
                                "trades": 93,
                                "wins": 93,
                                "losses": 0,
                                "avg_win_jpy": 504.0,
                                "avg_loss_jpy": 0.0,
                                "expectancy_jpy_per_trade": 504.0,
                            },
                            "MARKET_ORDER_TRADE_CLOSE": {
                                "trades": 84,
                                "wins": 13,
                                "losses": 71,
                                "avg_win_jpy": 218.4,
                                "avg_loss_jpy": 1095.5,
                                "expectancy_jpy_per_trade": -892.1,
                            },
                        },
                    }
                )
            )

            IntentGenerator(
                campaign_plan=_range_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=root / "intents.json",
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads((root / "intents.json").read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertEqual(result["status"], "DRY_RUN_BLOCKED")
            self.assertNotIn(
                "self_improvement_p0_repair_live_ready",
                result["intent"]["metadata"],
            )
            self.assertIn("SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE", issue_codes)

    def test_self_improvement_profitability_p0_repair_requires_direction_alignment(self) -> None:
        import os

        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "self_improvement_audit.json").write_text(
                    json.dumps(
                        {
                            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                            "findings": [
                                {
                                    "priority": "P0",
                                    "layer": "profitability",
                                    "code": "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
                                    "message": "market-close leakage is still negative",
                                    "evidence": {
                                        "current_streak": 65,
                                        "system_defect_evidence": {
                                            "profit_factor": 0.788,
                                            "expectancy_jpy": -54.04,
                                        },
                                    },
                                }
                            ],
                        }
                    )
                )
                (root / "capture_economics.json").write_text(
                    json.dumps(
                        {
                            "status": "NEGATIVE_EXPECTANCY",
                            "overall": {
                                "trades": 210,
                                "avg_win_jpy": 600.0,
                                "avg_loss_jpy": 1100.0,
                                "payoff_ratio": 0.545,
                                "breakeven_payoff_at_win_rate": 0.7,
                            },
                            "by_exit_reason": {
                                "TAKE_PROFIT_ORDER": {
                                    "trades": 93,
                                    "wins": 93,
                                    "losses": 0,
                                    "avg_win_jpy": 504.0,
                                    "avg_loss_jpy": 0.0,
                                    "expectancy_jpy_per_trade": 504.0,
                                },
                                "MARKET_ORDER_TRADE_CLOSE": {
                                    "trades": 84,
                                    "wins": 13,
                                    "losses": 71,
                                    "avg_win_jpy": 218.4,
                                    "avg_loss_jpy": 1095.5,
                                    "expectancy_jpy_per_trade": -892.1,
                                },
                            },
                        }
                    )
                )
                output = root / "intents.json"

                IntentGenerator(
                    campaign_plan=_range_campaign(root),
                    strategy_profile=_strategy(root, status="CANDIDATE"),
                    output_path=output,
                    report_path=root / "intents.md",
                    pair_charts_path=_pair_charts_with_direction(
                        root,
                        long_score=0.15,
                        short_score=0.85,
                        dominant_regime="RANGE",
                        m5_regime="RANGE",
                        m5_long_bias=0.15,
                        m5_short_bias=0.85,
                    ),
                    data_root=root,
                    max_loss_jpy=1000.0,
                ).run(snapshot_path=_snapshot(root))

                payload = json.loads(output.read_text())
                result = next(
                    item for item in payload["results"]
                    if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
                )
                metadata = result["intent"]["metadata"]
                issue_codes = {issue["code"] for issue in result["risk_issues"]}

                self.assertEqual(result["status"], "DRY_RUN_BLOCKED")
                self.assertNotIn("self_improvement_p0_repair_live_ready", metadata)
                self.assertIn("CHART_DIRECTION_CONFLICT", issue_codes)
                self.assertIn("SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE", issue_codes)
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

    def test_self_improvement_forecast_adverse_path_blocks_fresh_live_ready_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            (data_root / "self_improvement_audit.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "root_cause_focus": {
                            "primary": {
                                "family": "FORECAST_ADVERSE_PATH",
                                "confidence": "HIGH",
                                "priority": "P1",
                                "process_loop_streak": 16,
                                "supporting_codes": [
                                    "DIRECTIONAL_FORECAST_HIT_RATE_WEAK",
                                    "DIRECTIONAL_FORECAST_INVALIDATION_FIRST_DOMINANT",
                                ],
                                "metrics": {
                                    "directional_hit_rate": 0.261,
                                    "invalidation_first_rate": 0.739,
                                    "profit_factor": 0.891,
                                },
                            }
                        },
                        "findings": [
                            {
                                "priority": "P1",
                                "layer": "forecast",
                                "code": "DIRECTIONAL_FORECAST_HIT_RATE_WEAK",
                                "message": "directional_forecast HIT rate is weak",
                            }
                        ],
                    }
                )
            )
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=data_root,
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            issue_codes = {
                issue["code"]
                for item in payload["results"]
                for issue in item["risk_issues"]
            }

            self.assertEqual(summary.live_ready, 0)
            self.assertTrue(payload["results"])
            self.assertTrue(all(item["status"] == "DRY_RUN_BLOCKED" for item in payload["results"]))
            self.assertIn("SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH", issue_codes)
            self.assertTrue(
                any(
                    "persistent high-confidence forecast adverse path" in blocker
                    for item in payload["results"]
                    for blocker in item["live_blockers"]
                )
            )

    def test_stale_self_improvement_forecast_adverse_path_is_not_reused_after_memory_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            old_audit_at = datetime.now(timezone.utc) - timedelta(hours=2)
            (data_root / "self_improvement_audit.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": old_audit_at.isoformat(),
                        "root_cause_focus": {
                            "primary": {
                                "family": "FORECAST_ADVERSE_PATH",
                                "confidence": "HIGH",
                                "priority": "P1",
                                "process_loop_streak": 16,
                                "supporting_codes": [
                                    "DIRECTIONAL_FORECAST_HIT_RATE_WEAK",
                                    "DIRECTIONAL_FORECAST_INVALIDATION_FIRST_DOMINANT",
                                ],
                                "metrics": {
                                    "directional_hit_rate": 0.261,
                                    "invalidation_first_rate": 0.739,
                                    "profit_factor": 0.891,
                                },
                            }
                        },
                    }
                )
            )
            (data_root / "memory_health.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": (old_audit_at + timedelta(hours=1)).isoformat(),
                        "status": "MEMORY_HEALTH_PASS",
                    }
                )
            )
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=data_root,
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            issue_codes = {
                issue["code"]
                for item in payload["results"]
                for issue in item["risk_issues"]
            }

            self.assertNotIn("SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH", issue_codes)

    def test_forecast_adverse_path_allows_tp_proven_breakout_failure_repair_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_exact_vehicle_rotation_wins(
                root,
                method="BREAKOUT_FAILURE",
                count=20,
            )
            (root / "self_improvement_audit.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "root_cause_focus": {
                            "primary": {
                                "family": "FORECAST_ADVERSE_PATH",
                                "confidence": "HIGH",
                                "priority": "P1",
                                "process_loop_streak": 16,
                                "supporting_codes": [
                                    "DIRECTIONAL_FORECAST_HIT_RATE_WEAK",
                                    "DIRECTIONAL_FORECAST_INVALIDATION_FIRST_DOMINANT",
                                ],
                                "metrics": {
                                    "directional_hit_rate": 0.261,
                                    "invalidation_first_rate": 0.739,
                                    "profit_factor": 0.891,
                                },
                            }
                        },
                    }
                )
            )
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 213,
                            "avg_win_jpy": 415.5,
                            "avg_loss_jpy": 1061.9,
                            "payoff_ratio": 0.391,
                            "breakeven_payoff_at_win_rate": 0.651,
                        },
                        "by_exit_reason": {
                            "TAKE_PROFIT_ORDER": {
                                "trades": 93,
                                "wins": 93,
                                "losses": 0,
                                "avg_win_jpy": 504.0,
                                "avg_loss_jpy": 0.0,
                                "expectancy_jpy_per_trade": 504.0,
                            },
                            "MARKET_ORDER_TRADE_CLOSE": {
                                "trades": 92,
                                "wins": 19,
                                "losses": 73,
                                "avg_win_jpy": 220.0,
                                "avg_loss_jpy": 1095.5,
                                "expectancy_jpy_per_trade": -809.8,
                            },
                        },
                        **_capture_scoped_tp_payload(method="BREAKOUT_FAILURE", trades=20),
                    }
                )
            )
            campaign = _breakout_failure_campaign(root)
            campaign_payload = json.loads(campaign.read_text())
            campaign_payload["lanes"][0].update(
                {
                    "forecast_direction": "RANGE",
                    "forecast_confidence": 0.62,
                    "forecast_target_price": 1.17480,
                    "forecast_invalidation_price": 1.17200,
                    "forecast_range_low_price": 1.17200,
                    "forecast_range_high_price": 1.17480,
                }
            )
            campaign.write_text(json.dumps(campaign_payload))
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=campaign,
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts_with_direction(
                    root,
                    long_score=0.70,
                    short_score=0.30,
                    dominant_regime="BREAKOUT_FAILURE",
                    m5_regime="BREAKOUT_FAILURE",
                ),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertGreaterEqual(summary.live_ready, 1)
            self.assertEqual(result["status"], "LIVE_READY")
            self.assertEqual(metadata["geometry_model"], "RANGE_RAIL_LIMIT")
            self.assertEqual(
                metadata["range_indicator_source"],
                "forecast_range_box",
            )
            self.assertAlmostEqual(metadata["range_support"], 1.17200)
            self.assertAlmostEqual(metadata["range_resistance"], 1.17480)
            self.assertTrue(metadata["range_tp_is_inside_box"])
            self.assertTrue(metadata["range_sl_outside_box"])
            self.assertTrue(metadata["positive_rotation_live_ready"])
            self.assertEqual(metadata["positive_rotation_mode"], "TP_PROVEN_HARVEST")
            self.assertTrue(metadata["self_improvement_forecast_adverse_path_repair_live_ready"])
            self.assertEqual(
                metadata["self_improvement_forecast_adverse_path_repair_mode"],
                "TP_HARVEST_REPAIR",
            )
            self.assertIn("SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH_REPAIR_MODE", issue_codes)
            self.assertNotIn("SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH", issue_codes)
            self.assertNotIn("SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH", result["live_blocker_codes"])

    def test_projection_economic_precision_gap_blocks_persistent_forecast_path(self) -> None:
        # The operator goal needs forecast precision that remains true after
        # TIMEOUT/no-touch outcomes. A repeated headline-only projection bucket
        # must therefore become an executable no-new-risk blocker, not just an
        # audit line.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            (data_root / "self_improvement_audit.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "root_cause_focus": {
                            "primary": {
                                "family": "FORECAST_ADVERSE_PATH",
                                "confidence": "HIGH",
                                "priority": "P1",
                                "process_loop_streak": 9,
                                "supporting_codes": [
                                    "PROJECTION_ECONOMIC_PRECISION_WEAK",
                                ],
                                "metrics": {
                                    "projection_economic_precision_gap_count": 2,
                                    "projection_worst_economic_wilson_lower": 0.8882,
                                    "projection_worst_timeout_rate": 0.02,
                                    "profit_factor": 0.626,
                                },
                            }
                        },
                        "findings": [
                            {
                                "priority": "P1",
                                "layer": "forecast",
                                "code": "PROJECTION_ECONOMIC_PRECISION_WEAK",
                                "message": (
                                    "2 projection bucket(s) clear headline Wilson 90% precision "
                                    "but fail economic precision after TIMEOUT/no-touch penalties"
                                ),
                            }
                        ],
                    }
                )
            )
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=data_root,
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            issue_codes = {
                issue["code"]
                for item in payload["results"]
                for issue in item["risk_issues"]
            }

            self.assertEqual(summary.live_ready, 0)
            self.assertIn("SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH", issue_codes)
            self.assertTrue(
                any(
                    "projection_economic_precision_gap_count=2" in blocker
                    and "projection_worst_economic_wilson_lower=0.888" in blocker
                    for item in payload["results"]
                    for blocker in item["live_blockers"]
                )
            )

    def test_persistent_pending_churn_blocks_fresh_pending_intents_only(self) -> None:
        # Regression from qr-self-improvement-watch 2026-06-17: pending entries
        # were accepted, then repeatedly canceled before fill. Keep MARKET
        # variants available for separate gates, but stop arming new pending
        # orders until entry-distance/TTL is separated from thesis invalidation.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            (data_root / "self_improvement_audit.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "root_cause_focus": {
                            "primary": {
                                "family": "DATA_FRESHNESS",
                                "confidence": "HIGH",
                                "priority": "P0",
                                "supporting_codes": ["MEMORY_HEALTH_STALE"],
                            },
                            "candidates": [
                                {
                                    "family": "EXECUTION_LIFECYCLE",
                                    "confidence": "HIGH",
                                    "priority": "P0",
                                    "process_loop_streak": 14,
                                    "supporting_codes": [
                                        "PENDING_ENTRY_CANCEL_REVIEW_REQUIRED",
                                        "PENDING_ENTRY_CANCEL_RATE_HIGH",
                                    ],
                                    "metrics": {
                                        "pending_cancel_before_fill_rate": 0.656,
                                        "pending_fill_rate": 0.333,
                                        "profit_factor": 0.894,
                                    },
                                }
                            ],
                        },
                    }
                )
            )
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=data_root,
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            by_order_type = {
                item["intent"]["order_type"]: {issue["code"] for issue in item["risk_issues"]}
                for item in payload["results"]
                if item.get("intent")
            }

            self.assertGreaterEqual(summary.live_ready, 1)
            self.assertIn("STOP-ENTRY", by_order_type)
            self.assertIn("MARKET", by_order_type)
            self.assertIn(
                "SELF_IMPROVEMENT_PENDING_EXECUTION_LIFECYCLE",
                by_order_type["STOP-ENTRY"],
            )
            self.assertNotIn(
                "SELF_IMPROVEMENT_PENDING_EXECUTION_LIFECYCLE",
                by_order_type["MARKET"],
            )

    def test_pending_churn_allows_tp_proven_harvest_repair_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_exact_vehicle_rotation_wins(root)
            (root / "self_improvement_audit.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "root_cause_focus": {
                            "primary": {
                                "family": "EXECUTION_LIFECYCLE",
                                "confidence": "HIGH",
                                "priority": "P0",
                                "process_loop_streak": 7,
                                "supporting_codes": [
                                    "PENDING_ENTRY_CANCEL_RATE_HIGH",
                                    "PENDING_ENTRY_FILL_RATE_WEAK",
                                ],
                                "metrics": {
                                    "pending_cancel_before_fill_rate": 0.687,
                                    "pending_fill_rate": 0.313,
                                    "profit_factor": 0.813,
                                },
                            },
                        },
                    }
                )
            )
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 210,
                            "avg_win_jpy": 600.0,
                            "avg_loss_jpy": 1100.0,
                            "payoff_ratio": 0.545,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                        "by_exit_reason": {
                            "TAKE_PROFIT_ORDER": {
                                "trades": 93,
                                "wins": 93,
                                "losses": 0,
                                "avg_win_jpy": 504.0,
                                "avg_loss_jpy": 0.0,
                                "expectancy_jpy_per_trade": 504.0,
                            },
                            "MARKET_ORDER_TRADE_CLOSE": {
                                "trades": 84,
                                "wins": 13,
                                "losses": 71,
                                "avg_win_jpy": 218.4,
                                "avg_loss_jpy": 1095.5,
                                "expectancy_jpy_per_trade": -892.1,
                            },
                        },
                        **_capture_scoped_tp_payload(),
                    }
                )
            )
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_range_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertGreaterEqual(summary.live_ready, 1)
            self.assertEqual(result["status"], "LIVE_READY")
            self.assertTrue(metadata["self_improvement_pending_execution_repair_live_ready"])
            self.assertEqual(
                metadata["self_improvement_pending_execution_repair_mode"],
                "TP_HARVEST_REPAIR",
            )
            self.assertIn("SELF_IMPROVEMENT_PENDING_EXECUTION_REPAIR_MODE", issue_codes)
            self.assertNotIn("SELF_IMPROVEMENT_PENDING_EXECUTION_LIFECYCLE", issue_codes)
            self.assertEqual(result["live_blockers"], [])

    def test_pending_churn_group_blocks_matching_repair_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "self_improvement_audit.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "root_cause_focus": {
                            "primary": {
                                "family": "EXECUTION_LIFECYCLE",
                                "confidence": "HIGH",
                                "priority": "P0",
                                "process_loop_streak": 7,
                                "supporting_codes": [
                                    "PENDING_ENTRY_CANCEL_RATE_HIGH",
                                    "PENDING_ENTRY_FILL_RATE_WEAK",
                                ],
                                "metrics": {
                                    "pending_cancel_before_fill_rate": 0.687,
                                    "pending_fill_rate": 0.313,
                                    "profit_factor": 0.813,
                                },
                            },
                        },
                        "findings": [
                            {
                                "priority": "P1",
                                "layer": "execution_quality",
                                "code": "PENDING_ENTRY_CANCEL_RATE_HIGH",
                                "message": "pending entries are canceled before fill",
                                "next_action": (
                                    "separate thesis invalidation from entry-distance/TTL failures"
                                ),
                                "evidence": {
                                    "canceled_before_fill_orphan_groups": [
                                        {
                                            "pair": "EUR_USD",
                                            "side": "LONG",
                                            "method": "RANGE_ROTATION",
                                            "count": 4,
                                            "order_ids": ["472100", "472101"],
                                        }
                                    ]
                                },
                            }
                        ],
                    }
                )
            )
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 210,
                            "avg_win_jpy": 600.0,
                            "avg_loss_jpy": 1100.0,
                            "payoff_ratio": 0.545,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                        "by_exit_reason": {
                            "TAKE_PROFIT_ORDER": {
                                "trades": 93,
                                "wins": 93,
                                "losses": 0,
                                "avg_win_jpy": 504.0,
                                "avg_loss_jpy": 0.0,
                                "expectancy_jpy_per_trade": 504.0,
                            },
                            "MARKET_ORDER_TRADE_CLOSE": {
                                "trades": 84,
                                "wins": 13,
                                "losses": 71,
                                "avg_win_jpy": 218.4,
                                "avg_loss_jpy": 1095.5,
                                "expectancy_jpy_per_trade": -892.1,
                            },
                        },
                    }
                )
            )
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_range_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertEqual(result["status"], "DRY_RUN_BLOCKED")
            self.assertIn("SELF_IMPROVEMENT_PENDING_EXECUTION_LIFECYCLE", issue_codes)
            self.assertNotIn("SELF_IMPROVEMENT_PENDING_EXECUTION_REPAIR_MODE", issue_codes)
            self.assertNotIn("self_improvement_pending_execution_repair_live_ready", metadata)

    def test_pending_churn_group_with_thin_pair_sample_allows_repair_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_exact_vehicle_rotation_wins(
                root,
                count=10,
                realized_pl_jpy=600.0,
            )
            (root / "self_improvement_audit.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "root_cause_focus": {
                            "primary": {
                                "family": "EXECUTION_LIFECYCLE",
                                "confidence": "HIGH",
                                "priority": "P0",
                                "process_loop_streak": 7,
                                "supporting_codes": [
                                    "PENDING_ENTRY_CANCEL_RATE_HIGH",
                                    "PENDING_ENTRY_FILL_RATE_WEAK",
                                ],
                                "metrics": {
                                    "pending_cancel_before_fill_rate": 0.687,
                                    "pending_fill_rate": 0.313,
                                    "profit_factor": 0.813,
                                },
                            },
                        },
                        "findings": [
                            {
                                "priority": "P1",
                                "layer": "execution_quality",
                                "code": "PENDING_ENTRY_CANCEL_RATE_HIGH",
                                "message": "pending entries are canceled before fill",
                                "next_action": (
                                    "separate thesis invalidation from entry-distance/TTL failures"
                                ),
                                "evidence": {
                                    "canceled_before_fill_orphan_groups": [
                                        {
                                            "pair": "EUR_USD",
                                            "side": "LONG",
                                            "method": "RANGE_ROTATION",
                                            "count": 2,
                                            "order_ids": ["472100", "472101"],
                                        }
                                    ]
                                },
                            }
                        ],
                    }
                )
            )
            (root / "capture_economics.json").write_text(
                json.dumps(
                    {
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 210,
                            "avg_win_jpy": 600.0,
                            "avg_loss_jpy": 1100.0,
                            "payoff_ratio": 0.545,
                            "breakeven_payoff_at_win_rate": 0.7,
                        },
                        "by_exit_reason": {
                            "TAKE_PROFIT_ORDER": {
                                "trades": 93,
                                "wins": 93,
                                "losses": 0,
                                "avg_win_jpy": 504.0,
                                "avg_loss_jpy": 0.0,
                                "expectancy_jpy_per_trade": 504.0,
                            },
                            "MARKET_ORDER_TRADE_CLOSE": {
                                "trades": 84,
                                "wins": 13,
                                "losses": 71,
                                "avg_win_jpy": 218.4,
                                "avg_loss_jpy": 1095.5,
                                "expectancy_jpy_per_trade": -892.1,
                            },
                        },
                        **_capture_scoped_tp_payload(),
                    }
                )
            )
            output = root / "intents.json"

            summary = IntentGenerator(
                campaign_plan=_range_campaign(root),
                strategy_profile=_strategy(root, status="CANDIDATE"),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                data_root=root,
                max_loss_jpy=1000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = next(
                item for item in payload["results"]
                if item["lane_id"] == "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            )
            metadata = result["intent"]["metadata"]
            issue_codes = {issue["code"] for issue in result["risk_issues"]}

            self.assertGreaterEqual(summary.live_ready, 1)
            self.assertEqual(result["status"], "LIVE_READY")
            self.assertTrue(metadata["self_improvement_pending_execution_repair_live_ready"])
            self.assertIn("SELF_IMPROVEMENT_PENDING_EXECUTION_REPAIR_MODE", issue_codes)
            self.assertNotIn("SELF_IMPROVEMENT_PENDING_EXECUTION_LIFECYCLE", issue_codes)

    def test_sizes_units_with_percentage_risk_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                max_loss_pct=1.0,
                risk_equity_jpy=100_000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = payload["results"][0]
            self.assertEqual(result["status"], "DRY_RUN_PASSED")
            # 1% of 100,000 = 1000 JPY cap. Unit count is derived from cap and
            # the (now ATR-aware) SL distance; assert risk respects the cap.
            self.assertGreater(result["intent"]["units"], 0)
            self.assertLessEqual(result["risk_metrics"]["risk_jpy"], 1000.0)

    def test_sizes_units_to_margin_budget_before_live_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root),
                output_path=output,
                report_path=root / "intents.md",
                pair_charts_path=_pair_charts(root),
                max_loss_jpy=5_000.0,
            ).run(
                snapshot_path=_snapshot(
                    root,
                    nav_jpy=220_145.7765,
                    balance_jpy=208_945.7765,
                    margin_used_jpy=156_414.0,
                    margin_available_jpy=63_831.7765,
                )
            )

            payload = json.loads(output.read_text())
            for result in payload["results"]:
                self.assertGreater(result["intent"]["units"], 0)
                self.assertEqual(result["risk_metrics"]["max_margin_utilization_pct"], 95.0)
                self.assertLessEqual(result["risk_metrics"]["margin_utilization_after_pct"], 95.0)
                buffered_budget = result["risk_metrics"]["margin_budget_jpy"] * MARGIN_AWARE_BASKET_BUFFER
                self.assertLessEqual(result["risk_metrics"]["estimated_margin_jpy"], buffered_budget + 1e-6)
                issue_codes = {issue["code"] for issue in result["risk_issues"]}
                self.assertNotIn("MARGIN_UTILIZATION_CAP_EXCEEDED", issue_codes)
                self.assertNotIn("MARGIN_AVAILABLE_EXCEEDED", issue_codes)

    def test_recovery_hedge_uses_conviction_scaled_tranche_under_sl_free(self) -> None:
        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "intents.json"

                IntentGenerator(
                    campaign_plan=_campaign(root, direction="SHORT"),
                    strategy_profile=_strategy(root, status="MINE_MISSED_EDGE", direction="SHORT"),
                    output_path=output,
                    report_path=root / "intents.md",
                    pair_charts_path=_pair_charts_with_direction(
                        root,
                        long_score=0.25,
                        short_score=0.75,
                        dominant_regime="TREND_DOWN",
                        m5_regime="TREND_DOWN",
                    ),
                    max_loss_jpy=20_000.0,
                ).run(snapshot_path=_snapshot_with_underwater_long(root))

                payload = json.loads(output.read_text())
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        result = next(
            item
            for item in payload["results"]
            if item["lane_id"] == "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION"
        )
        intent = result["intent"]
        metadata = intent["metadata"]

        self.assertEqual(metadata["position_intent"], "HEDGE")
        self.assertTrue(metadata["hedge_recovery"])
        self.assertEqual(metadata["hedge_recovery_reference_units"], 22_000)
        self.assertLess(intent["units"], 22_000)
        self.assertEqual(intent["units"], metadata["hedge_recovery_units"])
        self.assertGreaterEqual(intent["units"], 1_000)
        self.assertLess(metadata["hedge_recovery_size_scale"], 1.0)
        self.assertEqual(metadata["hedge_timing_class"], "CONTINUATION")
        self.assertTrue(metadata["hedge_unwind_plan_required"])

    def test_recovery_hedge_reversal_confirmation_allows_larger_time_efficient_tranche(self) -> None:
        from quant_rabbit.models import Side
        from quant_rabbit.strategy.intent_generator import (
            RECOVERY_HEDGE_CONTINUATION_MAX_SCALE,
            _hedge_timing_metadata,
            _recovery_hedge_sizing_metadata,
        )

        metadata = {
            "position_intent": "HEDGE",
            "hedge_recovery": True,
            "hedge_recovery_units": 22_000,
            "hedge_existing_unrealized_pl_jpy": -23_000.0,
        }
        chart_context = {
            "chart_short_score": 0.75,
            "chart_long_score": 0.25,
            "chart_score_gap": -0.50,
            "tf_agreement_score": 0.80,
            "chart_story_structural": "M15 BOS_DOWN close_confirmed after failed reclaim",
        }

        metadata.update(_hedge_timing_metadata(Side.SHORT, metadata, chart_context, {}))
        sizing = _recovery_hedge_sizing_metadata(Side.SHORT, metadata, chart_context, {})

        self.assertEqual(metadata["hedge_timing_class"], "REVERSAL")
        self.assertGreater(sizing["hedge_recovery_size_scale"], RECOVERY_HEDGE_CONTINUATION_MAX_SCALE)
        self.assertGreater(sizing["hedge_recovery_units"], 7_000)

    def test_recovery_hedge_m5_opposition_prevents_reversal_classification(self) -> None:
        from quant_rabbit.models import Side
        from quant_rabbit.strategy.intent_generator import _hedge_timing_metadata

        metadata = {
            "position_intent": "HEDGE",
            "hedge_recovery": True,
            "hedge_recovery_units": 22_000,
            "hedge_existing_unrealized_pl_jpy": -23_000.0,
        }
        chart_context = {
            "chart_short_score": 0.75,
            "chart_long_score": 0.25,
            "chart_score_gap": -0.50,
            "tf_agreement_score": 0.80,
            "m5_long_bias": 0.625,
            "m5_short_bias": 0.25,
            "chart_story_structural": (
                "M5(FAILURE_RISK struct=CHOCH_UP@1.1593); "
                "H1(TREND_DOWN struct=BOS_DOWN@1.1594)"
            ),
        }

        metadata.update(_hedge_timing_metadata(Side.SHORT, metadata, chart_context, {}))

        self.assertEqual(metadata["hedge_timing_class"], "CONTINUATION")

    def test_recovery_hedge_market_blocks_when_m5_opposes_side(self) -> None:
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.MARKET,
            units=6000,
            tp=1.15181,
            sl=1.16131,
            thesis="market recovery hedge while M5 opposes",
            market_context=MarketContext(
                regime="BREAKOUT_FAILURE campaign lane",
                narrative="",
                chart_story="",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="",
            ),
            metadata={
                "position_intent": "HEDGE",
                "hedge_recovery": True,
                "chart_direction_bias": Side.SHORT.value,
                "chart_score_gap": -0.42,
                "m5_long_bias": 0.625,
                "m5_short_bias": 0.25,
                "chart_story_structural": (
                    "M5(FAILURE_RISK struct=CHOCH_UP@1.1593); "
                    "H1(TREND_DOWN struct=BOS_DOWN@1.1594)"
                ),
            },
        )

        issues = _method_context_issues(intent)
        issue_by_code = {issue["code"]: issue for issue in issues}

        self.assertEqual(issue_by_code["RECOVERY_HEDGE_MARKET_OPPOSED_BY_M5"]["severity"], "BLOCK")

    def test_breakout_failure_market_blocks_short_before_upper_retest(self) -> None:
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.MARKET,
            units=1000,
            entry=1.16136,
            tp=1.15971,
            sl=1.16198,
            thesis="short failed-break market chase near support",
            market_context=MarketContext(
                regime="UNCLEAR current; BREAKOUT_FAILURE campaign lane",
                narrative="",
                chart_story="",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="",
            ),
            metadata={
                "tf_regime_map": {
                    "M5": {
                        "classification": "TREND_WEAK",
                        "range_position": 0.226,
                        "nearest_support_distance_pips": -0.74,
                        "nearest_resistance_distance_pips": 2.54,
                    },
                    "M15": {
                        "classification": "RANGE",
                        "range_position": 0.041,
                        "nearest_support_distance_pips": -0.25,
                        "nearest_resistance_distance_pips": 5.75,
                    },
                }
            },
        )

        issues = _method_context_issues(intent)
        issue_by_code = {issue["code"]: issue for issue in issues}

        self.assertEqual(issue_by_code["BREAKOUT_FAILURE_MARKET_NOT_RETESTED"]["severity"], "BLOCK")

    def test_breakout_failure_market_allows_short_after_upper_retest(self) -> None:
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.MARKET,
            units=1000,
            entry=1.16176,
            tp=1.15971,
            sl=1.16224,
            thesis="short failed-break after resistance retest",
            market_context=MarketContext(
                regime="RANGE current; BREAKOUT_FAILURE campaign lane",
                narrative="",
                chart_story="",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="",
            ),
            metadata={
                "tf_regime_map": {
                    "M5": {
                        "classification": "RANGE",
                        "range_position": 0.82,
                        "nearest_support_distance_pips": -4.2,
                        "nearest_resistance_distance_pips": 0.6,
                    },
                    "M15": {
                        "classification": "RANGE",
                        "range_position": 0.71,
                        "nearest_support_distance_pips": -6.4,
                        "nearest_resistance_distance_pips": 1.8,
                    },
                }
            },
        )

        issues = _method_context_issues(intent)
        issue_codes = {issue["code"] for issue in issues}

        self.assertNotIn("BREAKOUT_FAILURE_MARKET_NOT_RETESTED", issue_codes)

    def test_breakout_failure_blocks_distant_atr_rr_when_harvest_tp_missing(self) -> None:
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Side, TradeMethod
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.SHORT,
            order_type=OrderType.STOP_ENTRY,
            units=3000,
            entry=1.15964,
            tp=1.15017,
            sl=1.16172,
            thesis="failed break must not use far ATR/RR fallback",
            market_context=MarketContext(
                regime="TREND_DOWN current; BREAKOUT_FAILURE campaign lane",
                narrative="",
                chart_story="",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="",
            ),
            metadata={
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "tp_target_source": "ATR_RR",
                "tp_target_reason": "structural target too close; using ATR/RR virtual target",
                "tp_target_distance_pips": 94.7,
                "tp_atr_pips": 3.0,
                "position_intent": "HEDGE",
                "hedge_recovery": True,
            },
        )

        issues = _method_context_issues(intent)
        issue_by_code = {issue["code"]: issue for issue in issues}

        self.assertEqual(issue_by_code["HARVEST_TP_STRUCTURE_MISSING"]["severity"], "BLOCK")

    def test_recovery_hedge_without_reversal_confirmation_caps_continuation_chase(self) -> None:
        from quant_rabbit.models import Side
        from quant_rabbit.strategy.intent_generator import (
            RECOVERY_HEDGE_CONTINUATION_MAX_SCALE,
            _hedge_timing_metadata,
            _recovery_hedge_sizing_metadata,
        )

        metadata = {
            "position_intent": "HEDGE",
            "hedge_recovery": True,
            "hedge_recovery_units": 22_000,
            "hedge_existing_unrealized_pl_jpy": -23_000.0,
        }
        chart_context = {
            "chart_short_score": 0.52,
            "chart_long_score": 0.48,
            "chart_score_gap": -0.04,
            "tf_agreement_score": 0.34,
            "chart_story_structural": "M15 range churn without close-confirmed BOS",
        }

        metadata.update(_hedge_timing_metadata(Side.SHORT, metadata, chart_context, {}))
        sizing = _recovery_hedge_sizing_metadata(Side.SHORT, metadata, chart_context, {})

        self.assertEqual(metadata["hedge_timing_class"], "CONTINUATION")
        self.assertLessEqual(sizing["hedge_recovery_size_scale"], RECOVERY_HEDGE_CONTINUATION_MAX_SCALE)
        self.assertLess(sizing["hedge_recovery_units"], 11_000)


def _campaign(root: Path, *, target_reward_risk: float | None = None, direction: str = "LONG") -> Path:
    path = root / "campaign.json"
    path.write_text(
        json.dumps(
            {
                "lanes": [
                    {
                        "desk": "trend_trader",
                        "pair": "EUR_USD",
                        "direction": direction,
                        "method": "TREND_CONTINUATION",
                        "adoption": "RISK_REPAIR_DRY_RUN",
                        "campaign_role": "NOW_IF_REPAIRED",
                        "reason": "trend continuation pressure",
                        "required_receipt": "dry-run under loss cap",
                        **({"target_reward_risk": target_reward_risk} if target_reward_risk is not None else {}),
                        "blockers": ["old sizing broke the loss cap"],
                        "story_examples": ["quality_audit: green staircase into upper band"],
                    },
                    {
                        "desk": "event_risk_trader",
                        "pair": "EUR_USD",
                        "direction": "BOTH",
                        "method": "EVENT_RISK",
                        "adoption": "RISK_OVERLAY",
                    },
                ]
            }
        )
    )
    return path


def _range_campaign(root: Path, *, direction: str = "LONG") -> Path:
    path = root / "range_campaign.json"
    path.write_text(
        json.dumps(
            {
                "lanes": [
                    {
                        "desk": "range_trader",
                        "pair": "EUR_USD",
                        "direction": direction,
                        "method": "RANGE_ROTATION",
                        "adoption": "ORDER_INTENT_REQUIRED",
                        "campaign_role": "NOW",
                        "reason": "range rail rotation pressure",
                        "required_receipt": "enter only at lower rail and rotate toward box interior",
                        "target_reward_risk": 2.0,
                        "blockers": [],
                        "story_examples": ["quality_audit: lower rail box reclaim into midpoint"],
                    }
                ]
            }
        )
    )
    return path


def _stamp_campaign_generated_at(path: Path) -> Path:
    payload = json.loads(path.read_text())
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload))
    return path


def _trigger_campaign(root: Path) -> Path:
    path = root / "trigger_campaign.json"
    path.write_text(
        json.dumps(
            {
                "lanes": [
                    {
                        "desk": "failure_trader",
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "method": "BREAKOUT_FAILURE",
                        "adoption": "TRIGGER_RECEIPT_REQUIRED",
                        "campaign_role": "BACKUP_OR_RELOAD",
                        "reason": "missed-edge trigger pressure",
                        "required_receipt": "Arm only a trigger/pending-entry receipt; no market chase.",
                        "target_reward_risk": 2.0,
                        "blockers": [],
                        "story_examples": ["quality_audit: failed downside break reclaimed the box"],
                    }
                ]
            }
        )
    )
    return path


def _breakout_failure_campaign(root: Path) -> Path:
    path = root / "breakout_failure_campaign.json"
    path.write_text(
        json.dumps(
            {
                "lanes": [
                    {
                        "desk": "failure_trader",
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "method": "BREAKOUT_FAILURE",
                        "adoption": "ORDER_INTENT_REQUIRED",
                        "campaign_role": "NOW",
                        "reason": "failed downside break reclaimed support",
                        "required_receipt": "wait for failed-break retest or confirmed trigger; no blind chase.",
                        "target_reward_risk": 2.0,
                        "blockers": [],
                        "story_examples": ["quality_audit: failed downside break reclaimed the box"],
                    }
                ]
            }
        )
    )
    return path


def _strategy(root: Path, *, status: str = "RISK_REPAIR_CANDIDATE", direction: str = "LONG") -> Path:
    path = root / "strategy.json"
    path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "pair": "EUR_USD",
                        "direction": direction,
                        "status": status,
                        "required_fix": "edge exists but old sizing broke the loss cap",
                    }
                ]
            }
        )
    )
    return path


def _write_post_harvest_close(
    data_root: Path,
    *,
    ts_utc: str,
    pair: str,
    closed_units: int,
) -> None:
    db = data_root / "execution_ledger.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE execution_events (
              ts_utc TEXT,
              event_type TEXT,
              pair TEXT,
              side TEXT,
              units INTEGER,
              order_id TEXT,
              trade_id TEXT,
              exit_reason TEXT,
              raw_json TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO execution_events (
              ts_utc, event_type, pair, side, units, order_id, trade_id, exit_reason, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts_utc,
                "GATEWAY_TRADE_CLOSE_SENT",
                pair,
                "",
                None,
                "close-order-1",
                "harvest-1",
                "TAKE_PROFIT_MARKET",
                json.dumps(
                    {
                        "sent": True,
                        "management_action": "TAKE_PROFIT_MARKET",
                        "owner": "trader",
                        "pair": pair,
                        "trade_id": "harvest-1",
                        "reasons": [
                            "temporary top profit-take: profit 5.5pip, top pullback confirmed",
                            "post-close re-entry discipline: refresh broker truth and require a fresh LIVE_READY pullback/retest lane before re-entering",
                        ],
                        "response": {
                            "orderCreateTransaction": {
                                "units": str(closed_units),
                            }
                        },
                    }
                ),
            ),
        )


def _write_broker_take_profit_close(
    data_root: Path,
    *,
    ts_utc: str,
    pair: str,
    side: str,
    lane_id: str,
    realized_pl_jpy: float,
) -> None:
    db = data_root / "execution_ledger.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE execution_events (
              ts_utc TEXT,
              event_type TEXT,
              lane_id TEXT,
              pair TEXT,
              side TEXT,
              units INTEGER,
              order_id TEXT,
              trade_id TEXT,
              exit_reason TEXT,
              realized_pl_jpy REAL,
              raw_json TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO execution_events (
              ts_utc, event_type, lane_id, pair, side, units, order_id, trade_id,
              exit_reason, realized_pl_jpy, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts_utc,
                "ORDER_FILLED",
                lane_id,
                pair,
                side,
                5000,
                "entry-order-1",
                "tp-1",
                None,
                None,
                json.dumps({"client_order_id": "qrv1-test"}),
            ),
        )
        conn.execute(
            """
            INSERT INTO execution_events (
              ts_utc, event_type, lane_id, pair, side, units, order_id, trade_id,
              exit_reason, realized_pl_jpy, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts_utc,
                "TRADE_CLOSED",
                "",
                pair,
                side,
                5000,
                "tp-order-1",
                "tp-1",
                "TAKE_PROFIT_ORDER",
                realized_pl_jpy,
                json.dumps({"reason": "TAKE_PROFIT_ORDER"}),
            ),
        )


def _write_exact_vehicle_take_profit_closes(
    data_root: Path,
    *,
    lane_id: str,
    pair: str,
    side: str,
    entry_reason: str,
    count: int,
    realized_pl_jpy: float | list[float],
    start_day: str = "2026-07-01",
) -> None:
    outcomes = (
        [float(realized_pl_jpy)] * count
        if isinstance(realized_pl_jpy, (int, float))
        else [float(value) for value in realized_pl_jpy]
    )
    if len(outcomes) != count:
        raise AssertionError("exact-vehicle fixture outcome count mismatch")
    db = data_root / "execution_ledger.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS execution_events (
              ts_utc TEXT,
              event_type TEXT,
              lane_id TEXT,
              pair TEXT,
              side TEXT,
              units INTEGER,
              order_id TEXT,
              trade_id TEXT,
              exit_reason TEXT,
              realized_pl_jpy REAL,
              financing_jpy REAL,
              raw_json TEXT
            )
            """
        )
        _ensure_execution_coverage(conn)
        for index in range(count):
            realized = outcomes[index]
            trade_id = f"exact-vehicle-{index}"
            order_id = f"entry-{index}"
            entry_ts = f"{start_day}T00:{index % 60:02d}:00+00:00"
            close_ts = f"{start_day}T01:{index % 60:02d}:00+00:00"
            signed_units = 1000 if side == "LONG" else -1000
            conn.execute(
                """
                INSERT INTO execution_events (
                  ts_utc, event_type, lane_id, pair, side, units, order_id, trade_id,
                  exit_reason, realized_pl_jpy, financing_jpy, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry_ts,
                    "ORDER_FILLED",
                    lane_id,
                    pair,
                    side,
                    1000,
                    order_id,
                    trade_id,
                    entry_reason,
                    None,
                    0.0,
                    json.dumps(
                        {
                            "type": "ORDER_FILL",
                            "time": entry_ts,
                            "instrument": pair,
                            "orderID": order_id,
                            "units": str(signed_units),
                            "reason": entry_reason,
                            "client_order_id": f"qrv1-exact-{index}",
                            "tradeOpened": {
                                "tradeID": trade_id,
                                "units": str(signed_units),
                            },
                        }
                    ),
                ),
            )
            conn.execute(
                """
                INSERT INTO execution_events (
                  ts_utc, event_type, lane_id, pair, side, units, order_id, trade_id,
                  exit_reason, realized_pl_jpy, financing_jpy, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    close_ts,
                    "TRADE_CLOSED",
                    "",
                    pair,
                    side,
                    1000,
                    f"tp-{index}",
                    trade_id,
                    "TAKE_PROFIT_ORDER",
                    realized,
                    0.0,
                    json.dumps(
                        {
                            "type": "ORDER_FILL",
                            "time": close_ts,
                            "reason": "TAKE_PROFIT_ORDER",
                            "commission": "0.0",
                            "guaranteedExecutionFee": "0.0",
                            "tradesClosed": [
                                {
                                    "tradeID": trade_id,
                                    "realizedPL": str(realized),
                                    "financing": "0.0",
                                }
                            ],
                        }
                    ),
                ),
            )
        conn.commit()


def _write_exact_vehicle_metric_trade(
    data_root: Path,
    *,
    trade_id: str,
    fill_lane_id: str,
    gateway_lane_id: str | None = None,
    close_lane_id: str = "",
    fill_pair: str = "EUR_USD",
    fill_side: str = "LONG",
    fill_reason: str = "LIMIT_ORDER",
    gateway_pair: str | None = None,
    gateway_side: str | None = None,
    outcome_event_type: str = "TRADE_CLOSED",
    realized_pl_jpy: float = 100.0,
    close_financing_jpy: float = 0.0,
    standalone_financing_jpy: float | None = None,
    partial_realized_pl_jpy: float | None = None,
    partial_exit_reason: str = "MARKET_ORDER_TRADE_CLOSE",
) -> None:
    db = data_root / "execution_ledger.db"
    order_id = f"entry-{trade_id}"
    fill_ts = "2026-07-01T00:01:00+00:00"
    close_ts = "2026-07-01T01:00:00+00:00"
    signed_units = 1000 if fill_side == "LONG" else -1000
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS execution_events (
              ts_utc TEXT,
              event_type TEXT,
              lane_id TEXT,
              pair TEXT,
              side TEXT,
              units INTEGER,
              order_id TEXT,
              trade_id TEXT,
              exit_reason TEXT,
              realized_pl_jpy REAL,
              financing_jpy REAL,
              raw_json TEXT
            )
            """
        )
        _ensure_execution_coverage(conn)
        if gateway_lane_id is not None:
            conn.execute(
                """
                INSERT INTO execution_events (
                  ts_utc, event_type, lane_id, pair, side, units, order_id, trade_id,
                  exit_reason, realized_pl_jpy, financing_jpy, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "2026-07-01T00:00:00+00:00",
                    "ORDER_ACCEPTED",
                    gateway_lane_id,
                    gateway_pair or fill_pair,
                    gateway_side or fill_side,
                    1000,
                    order_id,
                    trade_id,
                    "CLIENT_ORDER",
                    None,
                    None,
                    json.dumps({"type": "LIMIT"}),
                ),
            )
        conn.execute(
            """
            INSERT INTO execution_events (
              ts_utc, event_type, lane_id, pair, side, units, order_id, trade_id,
              exit_reason, realized_pl_jpy, financing_jpy, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fill_ts,
                "ORDER_FILLED",
                fill_lane_id,
                fill_pair,
                fill_side,
                1000,
                order_id,
                trade_id,
                    fill_reason,
                    0.0,
                    0.0,
                    json.dumps(
                        {
                            "type": "ORDER_FILL",
                            "time": fill_ts,
                            "instrument": fill_pair,
                            "orderID": order_id,
                            "units": str(signed_units),
                            "reason": fill_reason,
                            "tradeOpened": {
                                "tradeID": trade_id,
                                "units": str(signed_units),
                            },
                        }
                    ),
            ),
        )
        if standalone_financing_jpy is not None:
            financing_raw = {
                "type": "DAILY_FINANCING",
                "financing": str(standalone_financing_jpy),
                "positionFinancings": [
                    {
                        "openTradeFinancings": [
                            {
                                "tradeID": trade_id,
                                "financing": str(standalone_financing_jpy),
                                "homeConversionCost": "-1.25",
                            }
                        ]
                    }
                ],
            }
            conn.execute(
                """
                INSERT INTO execution_events (
                  ts_utc, event_type, lane_id, pair, side, units, order_id, trade_id,
                  exit_reason, realized_pl_jpy, financing_jpy, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "2026-07-01T00:02:00+00:00",
                    "OANDA_TRANSACTION",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    standalone_financing_jpy,
                    json.dumps(financing_raw),
                ),
            )
        if partial_realized_pl_jpy is not None:
            conn.execute(
                """
                INSERT INTO execution_events (
                  ts_utc, event_type, lane_id, pair, side, units, order_id, trade_id,
                  exit_reason, realized_pl_jpy, financing_jpy, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "2026-07-01T00:30:00+00:00",
                    "TRADE_REDUCED",
                    "",
                    fill_pair,
                    fill_side,
                    500,
                    f"reduce-{trade_id}",
                    trade_id,
                    partial_exit_reason,
                    partial_realized_pl_jpy,
                    0.0,
                    json.dumps(
                        {
                            "type": "ORDER_FILL",
                            "time": "2026-07-01T00:30:00+00:00",
                            "reason": partial_exit_reason,
                            "commission": "0.0",
                            "guaranteedExecutionFee": "0.0",
                            "tradeReduced": {
                                "tradeID": trade_id,
                                "realizedPL": str(partial_realized_pl_jpy),
                                "financing": "0.0",
                            },
                        }
                    ),
                ),
            )
        conn.execute(
            """
            INSERT INTO execution_events (
              ts_utc, event_type, lane_id, pair, side, units, order_id, trade_id,
              exit_reason, realized_pl_jpy, financing_jpy, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                close_ts,
                outcome_event_type,
                close_lane_id,
                fill_pair,
                fill_side,
                1000,
                f"close-{trade_id}",
                trade_id,
                "TAKE_PROFIT_ORDER",
                realized_pl_jpy,
                close_financing_jpy,
                json.dumps(
                    {
                        "type": "ORDER_FILL",
                        "time": close_ts,
                        "reason": "TAKE_PROFIT_ORDER",
                        "commission": "0.0",
                        "guaranteedExecutionFee": "0.0",
                        (
                            "tradesClosed"
                            if outcome_event_type == "TRADE_CLOSED"
                            else "tradeReduced"
                        ): (
                            [
                                {
                                    "tradeID": trade_id,
                                    "realizedPL": str(realized_pl_jpy),
                                    "financing": str(close_financing_jpy),
                                }
                            ]
                            if outcome_event_type == "TRADE_CLOSED"
                            else {
                                "tradeID": trade_id,
                                "realizedPL": str(realized_pl_jpy),
                                "financing": str(close_financing_jpy),
                            }
                        ),
                    }
                ),
            ),
        )
        conn.commit()


def _ensure_execution_coverage(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sync_state (
          key TEXT PRIMARY KEY,
          value TEXT,
          updated_at_utc TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO sync_state(key, value, updated_at_utc)
        VALUES ('oanda_transaction_coverage_start_utc', ?, ?)
        """,
        ("2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00"),
    )


def _write_same_day_lane_outcomes(
    data_root: Path,
    *,
    lane_id: str = "range_trader:EUR_USD:LONG:RANGE_ROTATION",
    pair: str = "EUR_USD",
    side: str = "LONG",
    outcomes: tuple[tuple[str, float], ...] = (("2026-06-22T05:00:00+00:00", -280.8),),
) -> None:
    db = data_root / "execution_ledger.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS execution_events (
              ts_utc TEXT,
              event_type TEXT,
              trade_id TEXT,
              order_id TEXT,
              lane_id TEXT,
              pair TEXT,
              side TEXT,
              units INTEGER,
              realized_pl_jpy REAL
            )
            """
        )
        for index, (ts_utc, realized_pl_jpy) in enumerate(outcomes, start=1):
            trade_id = f"T-lane-{index}"
            order_id = f"O-lane-{index}"
            conn.execute(
                """
                INSERT INTO execution_events(
                    ts_utc, event_type, trade_id, order_id, lane_id, pair, side, units, realized_pl_jpy
                ) VALUES (?, 'ORDER_ACCEPTED', ?, ?, ?, ?, ?, ?, NULL)
                """,
                (ts_utc, trade_id, order_id, lane_id, pair, side, 1000),
            )
            conn.execute(
                """
                INSERT INTO execution_events(
                    ts_utc, event_type, trade_id, order_id, lane_id, pair, side, units, realized_pl_jpy
                ) VALUES (?, 'ORDER_FILLED', ?, ?, ?, ?, ?, ?, NULL)
                """,
                (ts_utc, trade_id, order_id, lane_id, pair, side, 1000),
            )
            conn.execute(
                """
                INSERT INTO execution_events(
                    ts_utc, event_type, trade_id, order_id, lane_id, pair, side, units, realized_pl_jpy
                ) VALUES (?, 'TRADE_CLOSED', ?, ?, NULL, ?, NULL, NULL, ?)
                """,
                (ts_utc, trade_id, order_id, pair, realized_pl_jpy),
            )
        conn.commit()
    finally:
        conn.close()


def _stamp_capture_economics(root: Path, generated_at_utc: str) -> None:
    path = root / "capture_economics.json"
    payload = json.loads(path.read_text())
    payload["generated_at_utc"] = generated_at_utc
    path.write_text(json.dumps(payload))


def _pair_charts(root: Path) -> Path:
    path = root / "pair_charts.json"
    path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "charts": [
                    {
                        "pair": "EUR_USD",
                        "views": [
                            {
                                "granularity": "M5",
                                "indicators": {
                                    "atr_pips": 8.0,
                                    "bb_lower": 1.1710,
                                    "bb_upper": 1.1760,
                                    "bb_middle": 1.1735,
                                    "donchian_low": 1.1707,
                                    "donchian_high": 1.1764,
                                    "vwap": 1.1738,
                                    "avwap_anchor": 1.1734,
                                    "avwap_lower_1sd": 1.1712,
                                    "avwap_upper_1sd": 1.1758,
                                    "linreg_channel_lower": 1.1709,
                                    "linreg_channel_upper": 1.1761,
                                    "swing_low": 1.1705,
                                    "swing_high": 1.1767,
                                },
                            }
                        ],
                    }
                ]
            }
        )
    )
    return path


def _pair_charts_with_context(root: Path) -> Path:
    path = root / "pair_charts_context.json"
    path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "charts": [
                    {
                        "pair": "EUR_USD",
                        "dominant_regime": "TREND_DOWN",
                        "session": {"current_tag": "NY_AM_KILLZONE"},
                        "views": [
                            {
                                "granularity": "M5",
                                "regime_reading": {"state": "TREND_WEAK", "confidence": 0.5, "atr_percentile": 80.0},
                                "indicators": {
                                    "atr_pips": 8.0,
                                    "bb_lower": 1.1710,
                                    "bb_upper": 1.1760,
                                    "bb_middle": 1.1735,
                                    "donchian_low": 1.1707,
                                    "donchian_high": 1.1764,
                                    "vwap": 1.1738,
                                    "avwap_anchor": 1.1734,
                                    "avwap_lower_1sd": 1.1712,
                                    "avwap_upper_1sd": 1.1758,
                                    "linreg_channel_lower": 1.1709,
                                    "linreg_channel_upper": 1.1761,
                                    "swing_low": 1.1705,
                                    "swing_high": 1.1767,
                                },
                            }
                        ],
                    }
                ]
            }
        )
    )
    return path


def _pair_charts_location_map(root: Path) -> Path:
    path = root / "pair_charts_location_map.json"
    path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "charts": [
                    {
                        "pair": "EUR_USD",
                        "dominant_regime": "TREND_UP",
                        "long_score": 0.66,
                        "short_score": 0.22,
                        "session": {"current_tag": "NY_AM_KILLZONE"},
                        "views": [
                            {
                                "granularity": "M5",
                                "regime": "RANGE",
                                "long_bias": 0.54,
                                "short_bias": 0.46,
                                "regime_reading": {"state": "RANGE", "confidence": 0.72, "atr_percentile": 28.0},
                                "indicators": {
                                    "close": 1.17325,
                                    "atr_pips": 6.0,
                                    "adx_14": 16.0,
                                    "choppiness_14": 64.0,
                                    "linreg_slope_20": 0.01,
                                    "bb_lower": 1.1710,
                                    "bb_upper": 1.1760,
                                    "bb_middle": 1.1735,
                                    "donchian_low": 1.1707,
                                    "donchian_high": 1.1764,
                                    "vwap": 1.1732,
                                    "avwap_anchor": 1.1731,
                                    "avwap_lower_1sd": 1.1712,
                                    "avwap_upper_1sd": 1.1758,
                                    "linreg_channel_lower": 1.1709,
                                    "linreg_channel_upper": 1.1761,
                                    "swing_low": 1.1705,
                                    "swing_high": 1.1767,
                                },
                            },
                            {
                                "granularity": "H1",
                                "regime": "TREND_UP",
                                "long_bias": 0.72,
                                "short_bias": 0.18,
                                "regime_reading": {"state": "TREND_UP", "confidence": 0.77, "atr_percentile": 68.0},
                                "indicators": {
                                    "close": 1.17325,
                                    "atr_pips": 24.0,
                                    "adx_14": 31.0,
                                    "choppiness_14": 38.0,
                                    "linreg_slope_20": 0.52,
                                    "bb_lower": 1.1688,
                                    "bb_upper": 1.1802,
                                    "bb_middle": 1.1745,
                                    "donchian_low": 1.1684,
                                    "donchian_high": 1.1810,
                                    "vwap": 1.1729,
                                    "avwap_anchor": 1.1719,
                                    "linreg_channel_lower": 1.1700,
                                    "linreg_channel_upper": 1.1790,
                                    "swing_low": 1.1678,
                                    "swing_high": 1.1820,
                                },
                            },
                        ],
                    }
                ]
            }
        )
    )
    return path


def _levels_snapshot(root: Path) -> Path:
    path = root / "levels_snapshot.json"
    path.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "pair": "EUR_USD",
                        "pdh": 1.1762,
                        "pdl": 1.1691,
                        "pdc": 1.1728,
                        "daily_open": 1.1718,
                        "weekly_open": 1.1700,
                        "monthly_open": 1.1650,
                        "last_close": 1.1732,
                        "pivots": [
                            {
                                "style": "STANDARD",
                                "pp": 1.1727,
                                "r1": 1.1763,
                                "r2": 1.1792,
                                "s1": 1.1692,
                                "s2": 1.1664,
                            }
                        ],
                        "sessions": [
                            {"name": "ASIA", "high": 1.1742, "low": 1.1711, "range_pips": 31.0},
                            {"name": "LONDON", "high": 1.1764, "low": 1.1708, "range_pips": 56.0},
                        ],
                        "round_numbers": [
                            {"price": 1.17, "distance_pips": -33.0},
                            {"price": 1.175, "distance_pips": 17.0},
                        ],
                    }
                ],
                "issues": [],
            }
        )
    )
    return path


def _market_context_matrix(root: Path) -> Path:
    path = root / "market_context_matrix.json"
    path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "trade_count_policy": "ADVISORY_ONLY_DOES_NOT_BLOCK_OR_DEMOTE_LANES",
                "pairs": {
                    "EUR_USD": {
                        "LONG": {
                            "evidence_ref": "matrix:EUR_USD:LONG",
                            "support_count": 4,
                            "reject_count": 1,
                            "warning_count": 2,
                            "missing_count": 1,
                            "strongest_support": "chart and strength align EUR_USD LONG",
                            "strongest_reject": "COT longer-term conflicts EUR_USD LONG",
                            "supports": [
                                {
                                    "code": "GOLD_CONTEXT_TECHNICAL_DIRECTION",
                                    "layer": "context_asset_chart",
                                    "message": "XAU_USD pressure maps to EUR_USD LONG",
                                    "evidence_refs": ["context_asset:XAU_USD"],
                                }
                            ],
                            "rejects": [
                                {
                                    "code": "COT_CONFLICT",
                                    "layer": "cot",
                                    "message": "COT longer-term conflicts EUR_USD LONG",
                                    "evidence_refs": ["cot:EUR"],
                                }
                            ],
                        },
                        "SHORT": {
                            "evidence_ref": "matrix:EUR_USD:SHORT",
                            "support_count": 1,
                            "reject_count": 4,
                            "warning_count": 2,
                            "missing_count": 1,
                            "strongest_support": "COT longer-term aligns EUR_USD SHORT",
                            "strongest_reject": "chart and strength reject EUR_USD SHORT",
                        },
                    }
                },
                "issues": [],
            }
        )
    )
    return path


def _pair_charts_tp_mode(root: Path, *, adx: float, atr_percentile: float) -> Path:
    path = root / "pair_charts_tp_mode.json"
    path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "charts": [
                    {
                        "pair": "EUR_USD",
                        "dominant_regime": "TREND_UP",
                        "long_score": 0.78,
                        "short_score": 0.12,
                        "confluence": {
                            "score_balance": "LONG_LEAN",
                            "score_gap": 0.66,
                            "dominant_regime": "TREND_UP",
                            "tf_agreement_score": 1.0,
                            "atr_percentile_24h": atr_percentile,
                            "range_24h_sigma_multiple": 1.1,
                        },
                        "session": {"current_tag": "LONDON_NY_OVERLAP"},
                        "views": [
                            {
                                "granularity": "M5",
                                "regime": "TREND_UP",
                                "long_bias": 0.78,
                                "short_bias": 0.12,
                                "regime_reading": {"state": "TREND_UP", "confidence": 0.8, "atr_percentile": 70.0},
                                "indicators": {
                                    "atr_pips": 8.0,
                                    "bb_lower": 1.1710,
                                    "bb_upper": 1.1760,
                                    "donchian_low": 1.1707,
                                    "donchian_high": 1.1764,
                                    "linreg_channel_lower": 1.1709,
                                    "linreg_channel_upper": 1.1761,
                                    "swing_low": 1.1705,
                                    "swing_high": 1.1767,
                                },
                            },
                            {
                                "granularity": "H1",
                                "regime": "TREND_UP",
                                "indicators": {"atr_pips": 28.0, "adx_14": adx},
                                "smc": {"dealing_range": {"swing_high": 1.2300, "swing_low": 1.1500}},
                            },
                            {
                                "granularity": "H4",
                                "regime": "TREND_UP",
                                "indicators": {"atr_pips": 42.0, "adx_14": adx},
                                "smc": {"dealing_range": {"swing_high": 1.2300, "swing_low": 1.1500}},
                            },
                        ],
                    }
                ]
            }
        )
    )
    return path


def _pair_charts_with_direction(
    root: Path,
    *,
    long_score: float,
    short_score: float,
    dominant_regime: str,
    m5_regime: str,
    m5_long_bias: float | None = None,
    m5_short_bias: float | None = None,
    regime_quantile: str = "NORMAL",
    atr_pips: float = 8.0,
    regime_state: str = "TREND_WEAK",
    bb_squeeze: bool = False,
    atr_percentile: float = 50.0,
    adx: float = 22.0,
    choppiness: float = 50.0,
    bb_width_percentile: float = 50.0,
    close: float = 1.1735,
    trend_score: float = 0.2,
    breakout_score: float = 0.1,
    range_support: float = 1.1710,
    range_resistance: float = 1.1760,
) -> Path:
    path = root / "pair_charts_direction.json"
    path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "charts": [
                    {
                        "pair": "EUR_USD",
                        "generated_at_utc": "",
                        "dominant_regime": dominant_regime,
                        "long_score": long_score,
                        "short_score": short_score,
                        "confluence": {
                            "score_balance": (
                                "TIED"
                                if abs(round(long_score - short_score, 4)) <= 0.05
                                else ("LONG_LEAN" if long_score > short_score else "SHORT_LEAN")
                            ),
                            "score_gap": round(long_score - short_score, 4),
                            "dominant_regime": dominant_regime,
                        },
                        "session": {"current_tag": "NY_AM_KILLZONE"},
                        "views": [
                            {
                                "granularity": "M5",
                                "regime": m5_regime,
                                "long_bias": long_score if m5_long_bias is None else m5_long_bias,
                                "short_bias": short_score if m5_short_bias is None else m5_short_bias,
                                "regime_reading": {"state": regime_state, "confidence": 0.6, "atr_percentile": atr_percentile},
                                "family_scores": {
                                    "mean_rev_score": 1.1,
                                    "trend_score": trend_score,
                                    "breakout_score": breakout_score,
                                    "disagreement": 0.2,
                                },
                                "indicators": {
                                    "close": close,
                                    "atr_pips": atr_pips,
                                    "regime_quantile": regime_quantile,
                                    "bb_lower": range_support,
                                    "bb_upper": range_resistance,
                                    "bb_middle": 1.1735,
                                    "donchian_low": range_support - 0.0003,
                                    "donchian_high": range_resistance + 0.0004,
                                    "vwap": 1.1738,
                                    "avwap_anchor": 1.1734,
                                    "avwap_lower_1sd": 1.1712,
                                    "avwap_upper_1sd": 1.1758,
                                    "linreg_channel_lower": range_support - 0.0001,
                                    "linreg_channel_upper": range_resistance + 0.0001,
                                    "swing_low": range_support - 0.0005,
                                    "swing_high": range_resistance + 0.0007,
                                    "adx_14": adx,
                                    "choppiness_14": choppiness,
                                    "atr_percentile_100": atr_percentile,
                                    "bb_width_percentile_100": bb_width_percentile,
                                    "bb_squeeze": bb_squeeze,
                                },
                            }
                        ],
                    }
                ]
            }
        )
    )
    return path


def _snapshot(
    root: Path,
    *,
    usd_jpy: float = 156.64,
    eur_bid: float = 1.17322,
    eur_ask: float = 1.17330,
    nav_jpy: float = 200_000.0,
    balance_jpy: float = 200_000.0,
    margin_used_jpy: float = 0.0,
    margin_available_jpy: float = 200_000.0,
    fetched_at_utc: str | None = None,
    quote_timestamp_utc: str | None = None,
) -> Path:
    path = root / "snapshot.json"
    now = fetched_at_utc or datetime.now(timezone.utc).isoformat()
    quote_time = quote_timestamp_utc or now
    path.write_text(
        json.dumps(
            {
                "fetched_at_utc": now,
                "positions": [],
                "orders": [],
                "quotes": {
                    "EUR_USD": {"bid": eur_bid, "ask": eur_ask, "timestamp_utc": quote_time},
                    "USD_JPY": {"bid": usd_jpy, "ask": usd_jpy + 0.008, "timestamp_utc": quote_time},
                },
                "account": {
                    "nav_jpy": nav_jpy,
                    "balance_jpy": balance_jpy,
                    "margin_used_jpy": margin_used_jpy,
                    "margin_available_jpy": margin_available_jpy,
                    "fetched_at_utc": now,
                },
            }
        )
    )
    return path


def _write_forecast_telemetry(
    root: Path,
    *,
    direction: str,
    confidence: float,
    cycle_id: str = "test-cycle",
    timestamp_utc: str | None = None,
) -> None:
    data_root = root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    ts = timestamp_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    (data_root / "forecast_history.jsonl").write_text(
        json.dumps(
            {
                "timestamp_utc": ts,
                "cycle_id": cycle_id,
                "pair": "EUR_USD",
                "direction": direction,
                "confidence": confidence,
                "current_price": 1.17326,
                "target_price": 1.1712,
                "invalidation_price": 1.1742,
                "horizon_min": 60,
            }
        )
        + "\n"
    )
    (data_root / "projection_ledger.jsonl").write_text(
        json.dumps(
            {
                "timestamp_emitted_utc": ts,
                "pair": "EUR_USD",
                "signal_name": "directional_forecast",
                "direction": direction,
                "lead_time_min": 60,
                "confidence": confidence,
                "entry_price": 1.17326,
                "predicted_target_price": 1.1712,
                "predicted_invalidation_price": 1.1742,
                "resolution_window_min": 120,
                "resolution_status": "PENDING",
                "cycle_id": cycle_id,
            }
        )
        + "\n"
    )


def _snapshot_with_position(root: Path) -> Path:
    """Snapshot carrying an open trader position whose worst-case loss is
    near the per-trade cap (≈1000 JPY @ 5000u EUR_USD with a 20-pip stop)."""
    path = root / "snapshot.json"
    now = datetime.now(timezone.utc).isoformat()
    path.write_text(
        json.dumps(
            {
                "fetched_at_utc": now,
                "positions": [
                    {
                        "trade_id": "101",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "units": 5000,
                        "entry_price": 1.1710,
                        "take_profit": 1.1750,
                        "stop_loss": 1.1690,
                        "owner": "trader",
                    }
                ],
                "orders": [],
                "quotes": {
                    "EUR_USD": {"bid": 1.17322, "ask": 1.17330, "timestamp_utc": now},
                    "USD_JPY": {"bid": 156.64, "ask": 156.648, "timestamp_utc": now},
                },
                "account": {
                    "nav_jpy": 200_000.0,
                    "balance_jpy": 200_000.0,
                    "margin_used_jpy": 0.0,
                    "margin_available_jpy": 200_000.0,
                    "fetched_at_utc": now,
                },
            }
        )
    )
    return path


def _snapshot_with_underwater_long(root: Path) -> Path:
    path = root / "snapshot.json"
    now = datetime.now(timezone.utc).isoformat()
    path.write_text(
        json.dumps(
            {
                "fetched_at_utc": now,
                "positions": [
                    {
                        "trade_id": "101",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "units": 22_000,
                        "entry_price": 1.16688,
                        "unrealized_pl_jpy": -23_000.0,
                        "take_profit": None,
                        "stop_loss": None,
                        "owner": "trader",
                    }
                ],
                "orders": [],
                "quotes": {
                    "EUR_USD": {"bid": 1.16042, "ask": 1.16050, "timestamp_utc": now},
                    "USD_JPY": {"bid": 156.64, "ask": 156.648, "timestamp_utc": now},
                },
                "account": {
                    "nav_jpy": 170_000.0,
                    "balance_jpy": 192_000.0,
                    "margin_used_jpy": 162_000.0,
                    "margin_available_jpy": 7_000.0,
                    "unrealized_pl_jpy": -23_000.0,
                    "hedging_enabled": True,
                    "fetched_at_utc": now,
                },
                "home_conversions": {"USD": 156.64, "JPY": 1.0},
            }
        )
    )
    return path


class RegimeAwareGeometryHelpersTest(unittest.TestCase):
    """Unit tests for regime-derived reward_risk and SL widening helpers.

    Per AGENT_CONTRACT §3.5: TP/SL must be regime-derived. These tests pin the
    multiplier mapping so a future refactor cannot silently revert to a single
    fixed reward_risk floor.
    """

    def test_range_regime_shortens_target(self) -> None:
        from quant_rabbit.strategy.intent_generator import (
            REGIME_REWARD_RISK_RANGE_MULT,
            _regime_reward_risk_multiplier,
        )

        self.assertEqual(_regime_reward_risk_multiplier("RANGE"), REGIME_REWARD_RISK_RANGE_MULT)
        self.assertLess(REGIME_REWARD_RISK_RANGE_MULT, 1.0)

    def test_trend_regime_widens_target(self) -> None:
        from quant_rabbit.strategy.intent_generator import (
            REGIME_REWARD_RISK_TREND_MULT,
            _regime_reward_risk_multiplier,
        )

        self.assertEqual(_regime_reward_risk_multiplier("TREND_UP"), REGIME_REWARD_RISK_TREND_MULT)
        self.assertEqual(_regime_reward_risk_multiplier("TREND_DOWN"), REGIME_REWARD_RISK_TREND_MULT)
        self.assertGreater(REGIME_REWARD_RISK_TREND_MULT, 1.0)

    def test_impulse_regime_extends_target_furthest(self) -> None:
        from quant_rabbit.strategy.intent_generator import (
            REGIME_REWARD_RISK_IMPULSE_MULT,
            REGIME_REWARD_RISK_TREND_MULT,
            _regime_reward_risk_multiplier,
        )

        self.assertEqual(_regime_reward_risk_multiplier("IMPULSE_UP"), REGIME_REWARD_RISK_IMPULSE_MULT)
        self.assertGreaterEqual(REGIME_REWARD_RISK_IMPULSE_MULT, REGIME_REWARD_RISK_TREND_MULT)

    def test_unknown_or_unclear_regime_returns_unchanged(self) -> None:
        from quant_rabbit.strategy.intent_generator import _regime_reward_risk_multiplier

        self.assertEqual(_regime_reward_risk_multiplier(None), 1.0)
        self.assertEqual(_regime_reward_risk_multiplier(""), 1.0)
        self.assertEqual(_regime_reward_risk_multiplier("UNCLEAR"), 1.0)
        self.assertEqual(_regime_reward_risk_multiplier("FAILURE_RISK"), 1.0)

    def test_low_confidence_widens_stop(self) -> None:
        from quant_rabbit.strategy.intent_generator import (
            REGIME_LOW_CONFIDENCE_STOP_MULT,
            _regime_stop_widening_multiplier,
        )

        reading = {"confidence": 0.2, "atr_percentile": 0.5}
        self.assertEqual(_regime_stop_widening_multiplier(reading), REGIME_LOW_CONFIDENCE_STOP_MULT)

    def test_high_volatility_widens_stop(self) -> None:
        from quant_rabbit.strategy.intent_generator import (
            REGIME_HIGH_VOL_STOP_MULT,
            _regime_stop_widening_multiplier,
        )

        reading = {"confidence": 0.9, "atr_percentile": 0.95}
        self.assertEqual(_regime_stop_widening_multiplier(reading), REGIME_HIGH_VOL_STOP_MULT)

    def test_percent_scale_atr_percentile_does_not_turn_quiet_tape_into_high_vol(self) -> None:
        from quant_rabbit.strategy.intent_generator import (
            REGIME_HIGH_VOL_STOP_MULT,
            _regime_stop_widening_multiplier,
        )

        quiet_percent_scale = {"confidence": 0.9, "atr_percentile": 35.0}
        hot_percent_scale = {"confidence": 0.9, "atr_percentile": 95.0}

        self.assertEqual(_regime_stop_widening_multiplier(quiet_percent_scale), 1.0)
        self.assertEqual(_regime_stop_widening_multiplier(hot_percent_scale), REGIME_HIGH_VOL_STOP_MULT)

    def test_widening_is_clamped_at_max(self) -> None:
        from quant_rabbit.strategy.intent_generator import (
            REGIME_MAX_STOP_WIDEN,
            _regime_stop_widening_multiplier,
        )

        # Both signals trigger; result should not exceed the documented ceiling.
        reading = {"confidence": 0.1, "atr_percentile": 0.99}
        result = _regime_stop_widening_multiplier(reading)
        self.assertLessEqual(result, REGIME_MAX_STOP_WIDEN)

    def test_missing_reading_does_not_widen(self) -> None:
        from quant_rabbit.strategy.intent_generator import _regime_stop_widening_multiplier

        self.assertEqual(_regime_stop_widening_multiplier(None), 1.0)
        self.assertEqual(_regime_stop_widening_multiplier({}), 1.0)


class MacroEventSizingPlanTest(unittest.TestCase):
    @staticmethod
    def _lane(confidence: float) -> dict[str, object]:
        return {
            "forecast_direction": "UP",
            "forecast_market_support": {
                "ok": True,
                "direction": "UP",
                "signals": [
                    {
                        "name": "event_surprise_followthrough",
                        "direction": "UP",
                        "confidence": confidence,
                    }
                ],
            },
        }

    def test_same_direction_event_surprise_uses_high_band_inside_fresh_cap(self) -> None:
        from quant_rabbit.models import Side
        from quant_rabbit.strategy.intent_generator import _macro_event_sizing_plan

        effective, metadata = _macro_event_sizing_plan(
            self._lane(0.9),
            side=Side.LONG,
            base_max_loss_jpy=100.0,
            portfolio_loss_cap=400.0,
            position_metadata={},
            sizing_nav_jpy=20_000.0,
        )

        self.assertEqual(effective, 100.0)
        self.assertTrue(metadata["macro_event_size_up"])
        self.assertTrue(metadata["macro_event_confidence_sizing"])
        self.assertEqual(metadata["macro_event_confidence_band"], "HIGH")
        self.assertEqual(metadata["macro_event_risk_fraction"], 1.0)
        self.assertEqual(metadata["macro_event_fresh_absolute_cap_jpy"], 100.0)
        self.assertEqual(metadata["macro_event_signal_name"], "event_surprise_followthrough")

    def test_low_medium_high_confidence_increases_cap_and_units_without_exceeding_fresh_policy(self) -> None:
        from quant_rabbit.models import Side
        from quant_rabbit.strategy.intent_generator import (
            _macro_event_sizing_plan,
            _risk_budgeted_units,
        )

        now = datetime.now(timezone.utc)
        snapshot = BrokerSnapshot(
            fetched_at_utc=now,
            positions=(),
            orders=(),
            quotes={
                "EUR_USD": Quote("EUR_USD", bid=1.17322, ask=1.17330, timestamp_utc=now),
                "USD_JPY": Quote("USD_JPY", bid=156.64, ask=156.648, timestamp_utc=now),
            },
            account=AccountSummary(
                nav_jpy=50_000.0,
                balance_jpy=50_000.0,
                margin_used_jpy=0.0,
                margin_available_jpy=50_000.0,
                fetched_at_utc=now,
            ),
        )
        results: list[tuple[str, float, int]] = []
        for confidence in (0.79, 0.82, 0.87, 0.92):
            effective, metadata = _macro_event_sizing_plan(
                self._lane(confidence),
                side=Side.LONG,
                base_max_loss_jpy=500.0,
                portfolio_loss_cap=1_000.0,
                position_metadata={},
                sizing_nav_jpy=snapshot.account.nav_jpy,
            )
            units = _risk_budgeted_units(
                "EUR_USD",
                1.17330,
                1.17230,
                max_loss_jpy=effective,
                snapshot=snapshot,
                side=Side.LONG,
                loss_budget_target=True,
            )
            results.append((metadata["macro_event_confidence_band"], effective, units))

        self.assertEqual(
            [item[0] for item in results],
            ["SUB_THRESHOLD", "LOW", "MEDIUM", "HIGH"],
        )
        self.assertEqual([item[1] for item in results], [50.0, 125.0, 250.0, 500.0])
        self.assertLess(results[0][2], results[1][2])
        self.assertLess(results[1][2], results[2][2])
        self.assertLess(results[2][2], results[3][2])
        self.assertLessEqual(results[3][1], 500.0)

    def test_fresh_nav_and_non_refill_capacity_only_tighten_macro_absolute_cap(self) -> None:
        from quant_rabbit.models import Side
        from quant_rabbit.strategy.intent_generator import _macro_event_sizing_plan

        nav_capped, nav_metadata = _macro_event_sizing_plan(
            self._lane(0.95),
            side=Side.LONG,
            base_max_loss_jpy=3_000.0,
            portfolio_loss_cap=10_000.0,
            position_metadata={},
            sizing_nav_jpy=200_000.0,
        )
        loss_capped, loss_metadata = _macro_event_sizing_plan(
            self._lane(0.95),
            side=Side.LONG,
            base_max_loss_jpy=3_000.0,
            portfolio_loss_cap=300.0,
            position_metadata={},
            sizing_nav_jpy=200_000.0,
        )

        self.assertEqual(nav_capped, 2_000.0)
        self.assertEqual(nav_metadata["macro_event_nav_absolute_cap_jpy"], 2_000.0)
        self.assertEqual(loss_capped, 300.0)
        self.assertEqual(loss_metadata["macro_event_daily_loss_capacity_cap_jpy"], 300.0)

    def test_macro_event_sizing_does_not_expand_hedge_or_opposed_signal(self) -> None:
        from quant_rabbit.models import Side
        from quant_rabbit.strategy.intent_generator import _macro_event_sizing_plan

        lane = {
            "forecast_direction": "DOWN",
            "forecast_market_support": {
                "ok": True,
                "direction": "DOWN",
                "signals": [
                    {
                        "name": "event_surprise_followthrough",
                        "direction": "DOWN",
                        "confidence": 0.95,
                    }
                ],
            },
        }

        hedge_effective, hedge_metadata = _macro_event_sizing_plan(
            lane,
            side=Side.SHORT,
            base_max_loss_jpy=100.0,
            portfolio_loss_cap=1000.0,
            position_metadata={"position_intent": "HEDGE"},
        )
        opposed_effective, opposed_metadata = _macro_event_sizing_plan(
            lane,
            side=Side.LONG,
            base_max_loss_jpy=100.0,
            portfolio_loss_cap=1000.0,
            position_metadata={},
        )

        self.assertEqual(hedge_effective, 100.0)
        self.assertEqual(hedge_metadata, {})
        self.assertEqual(opposed_effective, 100.0)
        self.assertEqual(opposed_metadata, {})


class CanonicalRangeBoxTest(unittest.TestCase):
    @staticmethod
    def _charts() -> dict[str, dict[str, object]]:
        return {
            "EUR_USD": {
                "M5": {
                    "atr_pips": 2.0,
                    # A narrow micro box puts 1.17320 in its lower half.
                    "bb_lower": 1.17300,
                    "bb_upper": 1.17380,
                }
            }
        }

    def test_range_forecast_box_precedes_covering_chart_micro_box(self) -> None:
        selected = _range_indicators_for_lane(
            "EUR_USD",
            self._charts(),
            {
                "forecast_direction": "RANGE",
                "forecast_range_low_price": 1.17000,
                "forecast_range_high_price": 1.17400,
            },
            current_price=1.17320,
            method=TradeMethod.RANGE_ROTATION,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["range_indicator_source"], "forecast_range_box")
        self.assertEqual(selected["bb_lower"], 1.17000)
        self.assertEqual(selected["bb_upper"], 1.17400)

    def test_range_seed_side_uses_forecast_box_not_opposite_micro_box(self) -> None:
        forecast = SimpleNamespace(
            direction="RANGE",
            range_low_price=1.17000,
            range_high_price=1.17400,
        )

        # 1.17320 is in the upper half of the forecast box (SHORT), while the
        # covering M5 micro box above puts it in the lower half (LONG).
        self.assertEqual(
            _range_seed_direction(
                "EUR_USD",
                self._charts(),
                1.17320,
                forecast=forecast,
            ),
            "SHORT",
        )

    def test_forecast_box_drives_generated_range_metadata(self) -> None:
        quote = Quote("EUR_USD", bid=1.17316, ask=1.17324)
        selected = _range_indicators_for_lane(
            "EUR_USD",
            self._charts(),
            {
                "forecast_direction": "RANGE",
                "forecast_range_low_price": 1.17000,
                "forecast_range_high_price": 1.17400,
            },
            current_price=quote.mid,
            method=TradeMethod.RANGE_ROTATION,
        )
        self.assertIsNotNone(selected)
        entry, tp, sl = _geometry(
            "EUR_USD",
            Side.SHORT,
            OrderType.LIMIT,
            quote,
            reward_risk=1.0,
            atr_pips=2.0,
            range_indicators=selected,
            chart_indicators=selected,
        )
        metadata = _geometry_metadata(
            "EUR_USD",
            Side.SHORT,
            OrderType.LIMIT,
            quote,
            entry=entry,
            tp=tp,
            sl=sl,
            range_indicators=selected,
            chart_indicators=selected,
            atr_pips=2.0,
        )

        self.assertEqual(metadata["range_indicator_source"], "forecast_range_box")
        self.assertEqual(metadata["range_support"], 1.17000)
        self.assertEqual(metadata["range_resistance"], 1.17400)
        self.assertTrue(metadata["range_tp_is_inside_box"])
        self.assertTrue(metadata["range_sl_outside_box"])
        self.assertLessEqual(tp, 1.17400)
        self.assertGreater(sl, 1.17400)

    def test_price_outside_forecast_box_does_not_fallback_to_micro_box(self) -> None:
        quote = Quote("EUR_USD", bid=1.17316, ask=1.17324)
        selected = _range_indicators_for_lane(
            "EUR_USD",
            self._charts(),
            {
                "forecast_direction": "RANGE",
                "forecast_range_low_price": 1.17000,
                "forecast_range_high_price": 1.17200,
            },
            current_price=quote.mid,
            method=TradeMethod.RANGE_ROTATION,
        )
        self.assertIsNotNone(selected)
        entry, tp, sl = _geometry(
            "EUR_USD",
            Side.LONG,
            OrderType.LIMIT,
            quote,
            reward_risk=1.0,
            atr_pips=2.0,
            range_indicators=selected,
            chart_indicators=selected,
        )
        metadata = _geometry_metadata(
            "EUR_USD",
            Side.LONG,
            OrderType.LIMIT,
            quote,
            entry=entry,
            tp=tp,
            sl=sl,
            range_indicators=selected,
            chart_indicators=selected,
            atr_pips=2.0,
        )

        self.assertEqual(selected["range_indicator_source"], "forecast_range_box")
        self.assertNotEqual(metadata["geometry_model"], "RANGE_RAIL_LIMIT")
        self.assertNotIn("range_tp_is_inside_box", metadata)
        self.assertNotIn("range_sl_outside_box", metadata)

    def test_non_range_and_invalid_forecast_keep_chart_rails(self) -> None:
        chart = self._charts()["EUR_USD"]["M5"]
        non_range = _range_indicators_for_lane(
            "EUR_USD",
            self._charts(),
            {
                "forecast_direction": "UP",
                "forecast_range_low_price": 1.17000,
                "forecast_range_high_price": 1.17400,
            },
            current_price=1.17320,
            method=TradeMethod.BREAKOUT_FAILURE,
        )
        invalid_range = _range_indicators_for_lane(
            "EUR_USD",
            self._charts(),
            {
                "forecast_direction": "RANGE",
                "forecast_range_low_price": 1.17400,
                "forecast_range_high_price": 1.17000,
            },
            current_price=1.17320,
            method=TradeMethod.RANGE_ROTATION,
        )

        self.assertEqual(non_range, chart)
        self.assertEqual(invalid_range, chart)


class RangeRewardRiskFloorTest(unittest.TestCase):
    """Risk policy must allow rr < min_reward_risk for RANGE entries.

    Regression: before the regime-aware floor, range_trader rotations were
    forced to ≥1.2R even when the opposing rail capped TP closer. The
    risk validator must read `intent.metadata['regime_state']` and apply
    `policy.range_min_reward_risk` instead of the default floor.
    """

    def test_range_state_uses_range_min_reward_risk(self) -> None:
        from quant_rabbit.risk import RiskPolicy

        policy = RiskPolicy()
        self.assertLess(policy.range_min_reward_risk, policy.min_reward_risk)

    def test_default_min_reward_risk_unchanged_for_non_range(self) -> None:
        from quant_rabbit.risk import RiskPolicy

        # The default floor for trend/breakout/unclear remains conservative.
        self.assertGreaterEqual(RiskPolicy().min_reward_risk, 1.2)


class PerTradeFloorTest(unittest.TestCase):
    """Per-trade risk must be floored when pace×budget shrinks below an
    equity-derived minimum, but only when pace was not explicitly set by
    operator CLI. Per feedback_high_conviction_execution.md.
    """

    def test_floor_applied_when_pace_is_derived(self) -> None:
        from quant_rabbit.risk import RiskPolicy

        policy = RiskPolicy()
        self.assertIsNotNone(policy.min_per_trade_risk_pct)
        self.assertGreater(policy.min_per_trade_risk_pct, 0.0)


class NavPctSizingTest(unittest.TestCase):
    """`_nav_pct_position_units` and the SL-free sizing precedence path.

    User directive 2026-05-08「BaseUnitを決めると、資産が増えたときに追従
    できないよ。％で決めないといけなくない？」: position size must be
    NAV-relative so it auto-scales with equity.
    """

    def _account(self, nav_jpy: float = 227000.0):
        from quant_rabbit.models import AccountSummary
        return AccountSummary(
            balance_jpy=nav_jpy,
            nav_jpy=nav_jpy,
            margin_used_jpy=0.0,
            margin_available_jpy=nav_jpy,
            unrealized_pl_jpy=0.0,
            financing_jpy=0.0,
            pl_jpy=0.0,
            fetched_at_utc=datetime.now(timezone.utc),
            hedging_enabled=True,
            last_transaction_id="0",
        )

    def _snapshot(self, nav_jpy: float = 227000.0):
        from quant_rabbit.models import BrokerSnapshot, Quote
        return BrokerSnapshot(
            fetched_at_utc=datetime.now(timezone.utc),
            positions=(),
            orders=(),
            quotes={
                "EUR_USD": Quote(
                    pair="EUR_USD",
                    bid=1.17280,
                    ask=1.17290,
                    timestamp_utc=datetime.now(timezone.utc),
                ),
                "USD_JPY": Quote(
                    pair="USD_JPY",
                    bid=156.886,
                    ask=156.894,
                    timestamp_utc=datetime.now(timezone.utc),
                ),
            },
            account=self._account(nav_jpy),
            home_conversions={"USD": 157.0, "JPY": 1.0},
        )

    def test_nav_pct_returns_none_when_env_unset(self) -> None:
        import os
        from quant_rabbit.strategy.intent_generator import _nav_pct_position_units
        prior = os.environ.pop("QR_TRADER_POSITION_NAV_PCT", None)
        try:
            result = _nav_pct_position_units("EUR_USD", 1.17290, self._snapshot())
            self.assertIsNone(result)
        finally:
            if prior is not None:
                os.environ["QR_TRADER_POSITION_NAV_PCT"] = prior

    def test_nav_pct_30_yields_about_10000u_for_eur_usd_at_227k_nav(self) -> None:
        # 30% × 227,000 JPY = 68,100 JPY margin.
        # EUR_USD at 1.17290 with USDJPY=157 → margin/u ≈ 1.17290 × 157 × 0.04
        # = 7.366 JPY/u. → 68,100 / 7.366 ≈ 9,245 units.
        import os
        from quant_rabbit.strategy.intent_generator import _nav_pct_position_units
        os.environ["QR_TRADER_POSITION_NAV_PCT"] = "30"
        try:
            result = _nav_pct_position_units("EUR_USD", 1.17290, self._snapshot())
            self.assertIsNotNone(result)
            assert result is not None
            self.assertGreater(result, 8500.0)
            self.assertLess(result, 10500.0)
        finally:
            os.environ.pop("QR_TRADER_POSITION_NAV_PCT", None)

    def test_nav_pct_auto_scales_with_higher_nav(self) -> None:
        # Bumping NAV from 227k to 250k should grow units proportionally.
        import os
        from quant_rabbit.strategy.intent_generator import _nav_pct_position_units
        os.environ["QR_TRADER_POSITION_NAV_PCT"] = "30"
        try:
            small = _nav_pct_position_units("EUR_USD", 1.17290, self._snapshot(nav_jpy=227000.0))
            big = _nav_pct_position_units("EUR_USD", 1.17290, self._snapshot(nav_jpy=250000.0))
            self.assertIsNotNone(small)
            self.assertIsNotNone(big)
            assert small is not None and big is not None
            # 250/227 × small ≈ big within 5% tolerance.
            expected_ratio = 250000.0 / 227000.0
            actual_ratio = big / small
            self.assertAlmostEqual(actual_ratio, expected_ratio, delta=0.02)
        finally:
            os.environ.pop("QR_TRADER_POSITION_NAV_PCT", None)

    def test_nav_pct_auto_scales_down_with_lower_nav(self) -> None:
        # Drawdown to 200k should shrink units proportionally.
        import os
        from quant_rabbit.strategy.intent_generator import _nav_pct_position_units
        os.environ["QR_TRADER_POSITION_NAV_PCT"] = "30"
        try:
            normal = _nav_pct_position_units("EUR_USD", 1.17290, self._snapshot(nav_jpy=227000.0))
            shrunk = _nav_pct_position_units("EUR_USD", 1.17290, self._snapshot(nav_jpy=200000.0))
            self.assertIsNotNone(normal)
            self.assertIsNotNone(shrunk)
            assert normal is not None and shrunk is not None
            self.assertLess(shrunk, normal)
            expected_ratio = 200000.0 / 227000.0
            actual_ratio = shrunk / normal
            self.assertAlmostEqual(actual_ratio, expected_ratio, delta=0.02)
        finally:
            os.environ.pop("QR_TRADER_POSITION_NAV_PCT", None)

    def test_nav_pct_invalid_value_returns_none(self) -> None:
        import os
        from quant_rabbit.strategy.intent_generator import _nav_pct_position_units
        for bad in ("abc", "0", "-5", "  "):
            os.environ["QR_TRADER_POSITION_NAV_PCT"] = bad
            try:
                self.assertIsNone(_nav_pct_position_units("EUR_USD", 1.17290, self._snapshot()))
            finally:
                os.environ.pop("QR_TRADER_POSITION_NAV_PCT", None)


class IntegerUnitSizingIntentTest(unittest.TestCase):
    """Coverage for integer-unit sizing and the true one-unit budget floor.

    OANDA accepts whole FX units, so viable sub-1000u orders must retain their
    loss- and margin-bounded size. Zero units are reserved for geometry whose
    effective loss or margin budget cannot fund even one whole unit.
    """

    def _stub_snapshot(self, margin_used: float = 0.0, margin_available: float = 200000.0, positions=()):
        from quant_rabbit.models import AccountSummary, BrokerSnapshot, Quote
        now = datetime.now(timezone.utc)
        return BrokerSnapshot(
            fetched_at_utc=now,
            positions=tuple(positions),
            orders=(),
            quotes={
                "EUR_USD": Quote(pair="EUR_USD", bid=1.17280, ask=1.17290, timestamp_utc=now),
                "USD_JPY": Quote(pair="USD_JPY", bid=157.0, ask=157.01, timestamp_utc=now),
            },
            home_conversions={"USD": 157.005},
            account=AccountSummary(
                balance_jpy=227000.0,
                nav_jpy=227000.0,
                margin_used_jpy=margin_used,
                margin_available_jpy=margin_available,
                unrealized_pl_jpy=0.0,
                financing_jpy=0.0,
                pl_jpy=0.0,
                fetched_at_utc=now,
                hedging_enabled=True,
                last_transaction_id="0",
            ),
        )

    def test_risk_budgeted_units_returns_sub1000_integer_when_budget_supports_it(self) -> None:
        # max_loss_jpy=50 JPY, stop ≈ 20 pip → loss budget ≈ 159 units.
        from quant_rabbit.strategy.intent_generator import _risk_budgeted_units
        units = _risk_budgeted_units(
            "EUR_USD",
            entry=1.17290,
            sl=1.17490,
            max_loss_jpy=50.0,
            snapshot=self._stub_snapshot(),
        )
        self.assertEqual(units, 159)

    def test_risk_budgeted_units_returns_zero_when_loss_budget_is_below_one_unit(self) -> None:
        from quant_rabbit.strategy.intent_generator import _risk_budgeted_units
        units = _risk_budgeted_units(
            "EUR_USD",
            entry=1.17290,
            sl=1.17490,
            max_loss_jpy=0.1,
            snapshot=self._stub_snapshot(),
        )
        self.assertEqual(units, 0)

    def test_risk_budgeted_units_remain_positive_at_92_point_5_percent_existing_utilization(self) -> None:
        nav_jpy = 227_000.0
        units = _risk_budgeted_units(
            "EUR_USD",
            entry=1.17290,
            sl=1.17490,
            max_loss_jpy=2_000.0,
            snapshot=self._stub_snapshot(
                margin_used=nav_jpy * 0.925,
                margin_available=nav_jpy * 0.075,
            ),
        )

        self.assertEqual(units, 693)

    def test_unit_floor_block_reports_loss_budget_below_one_separately_from_margin(self) -> None:
        from quant_rabbit.models import Side
        from quant_rabbit.strategy.intent_generator import _min_lot_block_issue

        issue = _min_lot_block_issue(
            pair="EUR_USD",
            entry=1.17290,
            sl=1.17490,
            max_loss_jpy=0.1,
            snapshot=self._stub_snapshot(margin_available=200000.0),
            side=Side.SHORT,
        )

        self.assertEqual(issue["code"], "LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT")
        self.assertIn("loss budget", issue["message"])
        self.assertNotIn("margin headroom", issue["message"])

    def test_unit_floor_block_reports_margin_below_one_when_margin_is_the_cap(self) -> None:
        from quant_rabbit.models import Side
        from quant_rabbit.strategy.intent_generator import _min_lot_block_issue

        issue = _min_lot_block_issue(
            pair="EUR_USD",
            entry=1.17290,
            sl=1.17490,
            max_loss_jpy=2000.0,
            snapshot=self._stub_snapshot(margin_available=1.0),
            side=Side.SHORT,
        )

        self.assertEqual(issue["code"], "MARGIN_TOO_THIN_FOR_MIN_LOT")
        self.assertIn("margin headroom", issue["message"])

    def test_unit_floor_block_reports_combined_loss_and_margin_below_one(self) -> None:
        from quant_rabbit.models import Side
        from quant_rabbit.strategy.intent_generator import _min_lot_block_issue

        issue = _min_lot_block_issue(
            pair="EUR_USD",
            entry=1.17290,
            sl=1.17490,
            max_loss_jpy=0.1,
            snapshot=self._stub_snapshot(margin_available=1.0),
            side=Side.SHORT,
        )

        self.assertEqual(issue["code"], "LOSS_AND_MARGIN_TOO_THIN_FOR_MIN_LOT")
        self.assertIn("loss budget", issue["message"])
        self.assertIn("margin headroom", issue["message"])
        self.assertIn("Free margin alone is insufficient", issue["message"])

    def test_risk_budgeted_units_caps_same_pair_hedge_by_loss_budget(self) -> None:
        import os

        from quant_rabbit.models import BrokerPosition, Owner, Side
        from quant_rabbit.strategy.intent_generator import _risk_budgeted_units

        prior_sl_free = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        open_long = BrokerPosition(
            trade_id="101",
            pair="EUR_USD",
            side=Side.LONG,
            units=22_000,
            entry_price=1.16688,
            owner=Owner.TRADER,
        )
        try:
            units = _risk_budgeted_units(
                "EUR_USD",
                entry=1.17290,
                sl=1.17490,
                max_loss_jpy=2_000.0,
                snapshot=self._stub_snapshot(
                    margin_used=209_000.0,
                    margin_available=18_000.0,
                    positions=(open_long,),
                ),
                side=Side.SHORT,
                position_intent="HEDGE",
            )
        finally:
            if prior_sl_free is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl_free

        self.assertEqual(units, 6_369)

    def test_risk_budgeted_units_ignores_manual_opposing_units_for_hedge_target(self) -> None:
        import os

        from quant_rabbit.models import BrokerPosition, Owner, Side
        from quant_rabbit.strategy.intent_generator import _risk_budgeted_units

        prior_sl_free = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        trader_long = BrokerPosition(
            trade_id="101",
            pair="EUR_USD",
            side=Side.LONG,
            units=7_000,
            entry_price=1.16688,
            owner=Owner.TRADER,
        )
        manual_long = BrokerPosition(
            trade_id="manual",
            pair="EUR_USD",
            side=Side.LONG,
            units=15_000,
            entry_price=1.16688,
            owner=Owner.UNKNOWN,
        )
        existing_short = BrokerPosition(
            trade_id="102",
            pair="EUR_USD",
            side=Side.SHORT,
            units=5_700,
            entry_price=1.16013,
            owner=Owner.TRADER,
        )
        try:
            units = _risk_budgeted_units(
                "EUR_USD",
                entry=1.16013,
                sl=1.16317,
                max_loss_jpy=10_000.0,
                snapshot=self._stub_snapshot(
                    margin_used=209_000.0,
                    margin_available=18_000.0,
                    positions=(trader_long, manual_long, existing_short),
                ),
                side=Side.SHORT,
                position_intent="HEDGE",
            )
        finally:
            if prior_sl_free is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl_free

        self.assertEqual(units, 1_300)

    def test_risk_budgeted_units_uses_manual_exposure_for_broker_margin_offset(self) -> None:
        import os

        from quant_rabbit.models import BrokerPosition, Owner, Side
        from quant_rabbit.strategy.intent_generator import _risk_budgeted_units

        prior_sl_free = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        prior_nav_pct = os.environ.pop("QR_TRADER_POSITION_NAV_PCT", None)
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        trader_long = BrokerPosition(
            trade_id="101",
            pair="EUR_USD",
            side=Side.LONG,
            units=7_000,
            entry_price=1.16688,
            owner=Owner.TRADER,
        )
        manual_long = BrokerPosition(
            trade_id="manual",
            pair="EUR_USD",
            side=Side.LONG,
            units=15_000,
            entry_price=1.16688,
            owner=Owner.UNKNOWN,
        )
        existing_short = BrokerPosition(
            trade_id="102",
            pair="EUR_USD",
            side=Side.SHORT,
            units=8_400,
            entry_price=1.16013,
            owner=Owner.TRADER,
        )
        try:
            units = _risk_budgeted_units(
                "EUR_USD",
                entry=1.16013,
                sl=1.16317,
                max_loss_jpy=10_000.0,
                snapshot=self._stub_snapshot(
                    margin_used=209_000.0,
                    margin_available=18_000.0,
                    positions=(trader_long, manual_long, existing_short),
                ),
                side=Side.SHORT,
                position_intent="PYRAMID",
            )
        finally:
            if prior_sl_free is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl_free
            if prior_nav_pct is not None:
                os.environ["QR_TRADER_POSITION_NAV_PCT"] = prior_nav_pct

        # The 22,000u opposing broker exposure offsets the existing 8,400u
        # short, and the default 95% cap contributes its remaining buffered
        # margin room, so sizing permits 14,421u. A stale 3,000u base-unit
        # target must not suppress that current loss/margin-budgeted result.
        self.assertEqual(units, 14_421)

    def test_position_intent_metadata_marks_underwater_opposite_side_as_recovery_hedge(self) -> None:
        from quant_rabbit.models import BrokerPosition, Owner, Side
        from quant_rabbit.strategy.intent_generator import _position_intent_metadata

        trader_long = BrokerPosition(
            trade_id="101",
            pair="EUR_USD",
            side=Side.LONG,
            units=7000,
            entry_price=1.16688,
            unrealized_pl_jpy=-500.0,
            owner=Owner.TRADER,
        )
        manual_long = BrokerPosition(
            trade_id="102",
            pair="EUR_USD",
            side=Side.LONG,
            units=15_000,
            entry_price=1.16688,
            unrealized_pl_jpy=-1000.0,
            owner=Owner.UNKNOWN,
        )

        metadata = _position_intent_metadata("EUR_USD", Side.SHORT, self._stub_snapshot(positions=(trader_long, manual_long)))

        self.assertEqual(metadata["position_intent"], "HEDGE")
        self.assertTrue(metadata["hedge_recovery"])
        self.assertEqual(metadata["hedge_reference_scope"], "trader_owned_only")
        self.assertEqual(metadata["hedge_recovery_units"], 7_000)
        self.assertEqual(metadata["hedge_recovery_unrealized_pl_jpy"], -500.0)
        self.assertEqual(metadata["hedge_non_trader_opposing_units_ignored"], 15_000)
        self.assertEqual(metadata["hedge_non_trader_opposing_unrealized_pl_jpy_ignored"], -1000.0)

    def test_position_intent_metadata_caps_recovery_hedge_to_uncovered_opposite_units(self) -> None:
        from quant_rabbit.models import BrokerPosition, Owner, Side
        from quant_rabbit.strategy.intent_generator import _position_intent_metadata

        trader_long = BrokerPosition(
            trade_id="101",
            pair="EUR_USD",
            side=Side.LONG,
            units=22_000,
            entry_price=1.16688,
            unrealized_pl_jpy=-1500.0,
            owner=Owner.TRADER,
        )
        existing_short = BrokerPosition(
            trade_id="102",
            pair="EUR_USD",
            side=Side.SHORT,
            units=5_000,
            entry_price=1.17000,
            unrealized_pl_jpy=400.0,
            owner=Owner.TRADER,
        )

        metadata = _position_intent_metadata("EUR_USD", Side.SHORT, self._stub_snapshot(positions=(trader_long, existing_short)))

        self.assertEqual(metadata["position_intent"], "HEDGE")
        self.assertTrue(metadata["hedge_recovery"])
        self.assertEqual(metadata["hedge_gross_opposing_units"], 22_000)
        self.assertEqual(metadata["hedge_existing_same_side_units"], 5_000)
        self.assertEqual(metadata["hedge_recovery_units"], 17_000)

    def test_position_intent_metadata_suppresses_covered_recovery_hedge(self) -> None:
        from quant_rabbit.models import BrokerPosition, Owner, Side
        from quant_rabbit.strategy.intent_generator import _position_intent_metadata

        trader_long = BrokerPosition(
            trade_id="101",
            pair="EUR_USD",
            side=Side.LONG,
            units=22_000,
            entry_price=1.16688,
            unrealized_pl_jpy=-1500.0,
            owner=Owner.TRADER,
        )
        existing_short = BrokerPosition(
            trade_id="102",
            pair="EUR_USD",
            side=Side.SHORT,
            units=22_000,
            entry_price=1.17000,
            unrealized_pl_jpy=900.0,
            owner=Owner.TRADER,
        )

        metadata = _position_intent_metadata("EUR_USD", Side.SHORT, self._stub_snapshot(positions=(trader_long, existing_short)))

        self.assertEqual(metadata["position_intent"], "PYRAMID")
        self.assertEqual(metadata["hedge_reference_units"], 0)
        self.assertEqual(metadata["hedge_existing_same_side_units"], 22_000)
        self.assertEqual(metadata["hedge_suppressed_reason"], "opposite_exposure_already_covered")

    def test_position_intent_metadata_labels_long_adverse_add(self) -> None:
        from quant_rabbit.models import BrokerPosition, Owner, Side
        from quant_rabbit.strategy.intent_generator import _position_intent_metadata

        existing_long = BrokerPosition(
            trade_id="101",
            pair="EUR_USD",
            side=Side.LONG,
            units=10_000,
            entry_price=1.10000,
            owner=Owner.TRADER,
        )

        metadata = _position_intent_metadata(
            "EUR_USD",
            Side.LONG,
            self._stub_snapshot(positions=(existing_long,)),
            entry=1.09930,
        )

        self.assertEqual(metadata["position_intent"], "PYRAMID")
        self.assertEqual(metadata["same_pair_add_type"], "AVERAGE_INTO_ADVERSE")
        self.assertEqual(metadata["same_pair_existing_entries"], 1)
        self.assertEqual(metadata["same_pair_existing_units"], 10_000)
        self.assertEqual(metadata["same_pair_existing_avg_entry"], 1.1)
        self.assertEqual(metadata["same_pair_add_distance_from_avg_pips"], -7.0)
        self.assertEqual(metadata["same_pair_adverse_add_pips"], 7.0)
        self.assertEqual(metadata["same_pair_with_move_add_pips"], 0.0)

    def test_position_intent_metadata_labels_long_with_move_add(self) -> None:
        from quant_rabbit.models import BrokerPosition, Owner, Side
        from quant_rabbit.strategy.intent_generator import _position_intent_metadata

        existing_long = BrokerPosition(
            trade_id="101",
            pair="EUR_USD",
            side=Side.LONG,
            units=10_000,
            entry_price=1.10000,
            owner=Owner.TRADER,
        )

        metadata = _position_intent_metadata(
            "EUR_USD",
            Side.LONG,
            self._stub_snapshot(positions=(existing_long,)),
            entry=1.10080,
        )

        self.assertEqual(metadata["same_pair_add_type"], "PYRAMID_WITH_MOVE")
        self.assertEqual(metadata["same_pair_adverse_add_pips"], 0.0)
        self.assertEqual(metadata["same_pair_with_move_add_pips"], 8.0)

    def test_position_intent_metadata_labels_short_adverse_and_with_move_adds(self) -> None:
        from quant_rabbit.models import BrokerPosition, Owner, Side
        from quant_rabbit.strategy.intent_generator import _position_intent_metadata

        existing_short = BrokerPosition(
            trade_id="101",
            pair="EUR_USD",
            side=Side.SHORT,
            units=10_000,
            entry_price=1.10000,
            owner=Owner.TRADER,
        )

        adverse = _position_intent_metadata(
            "EUR_USD",
            Side.SHORT,
            self._stub_snapshot(positions=(existing_short,)),
            entry=1.10070,
        )
        with_move = _position_intent_metadata(
            "EUR_USD",
            Side.SHORT,
            self._stub_snapshot(positions=(existing_short,)),
            entry=1.09920,
        )

        self.assertEqual(adverse["same_pair_add_type"], "AVERAGE_INTO_ADVERSE")
        self.assertEqual(adverse["same_pair_adverse_add_pips"], 7.0)
        self.assertEqual(with_move["same_pair_add_type"], "PYRAMID_WITH_MOVE")
        self.assertEqual(with_move["same_pair_with_move_add_pips"], 8.0)

    def test_risk_budgeted_units_preserves_integer_precision_near_1000(self) -> None:
        # max_loss_jpy ~315 JPY → loss budget ≈ 1003 whole units.
        from quant_rabbit.strategy.intent_generator import _risk_budgeted_units
        units = _risk_budgeted_units(
            "EUR_USD",
            entry=1.17290,
            sl=1.17490,
            max_loss_jpy=315.0,
            snapshot=self._stub_snapshot(),
        )
        self.assertEqual(units, 1003)

    def test_risk_budgeted_units_preserves_integer_precision_for_clear_budget(self) -> None:
        from quant_rabbit.strategy.intent_generator import _risk_budgeted_units
        units = _risk_budgeted_units(
            "EUR_USD",
            entry=1.17290,
            sl=1.17490,
            max_loss_jpy=2000.0,
            snapshot=self._stub_snapshot(),
        )
        self.assertEqual(units, 6369)

    def test_sl_free_without_nav_pct_uses_current_loss_budget(self) -> None:
        import os
        from quant_rabbit.strategy.intent_generator import _risk_budgeted_units

        prior_sl_free = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        prior_nav_pct = os.environ.pop("QR_TRADER_POSITION_NAV_PCT", None)
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            normal_units = _risk_budgeted_units(
                "EUR_USD",
                entry=1.17290,
                sl=1.17490,
                max_loss_jpy=2000.0,
                snapshot=self._stub_snapshot(),
            )
            event_units = _risk_budgeted_units(
                "EUR_USD",
                entry=1.17290,
                sl=1.17490,
                max_loss_jpy=2000.0,
                snapshot=self._stub_snapshot(),
                loss_budget_target=True,
            )
        finally:
            if prior_sl_free is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl_free
            if prior_nav_pct is not None:
                os.environ["QR_TRADER_POSITION_NAV_PCT"] = prior_nav_pct

        self.assertEqual(normal_units, 6369)
        self.assertEqual(event_units, 6369)

    def test_sl_free_ignores_legacy_base_units_when_nav_pct_is_unavailable(self) -> None:
        import os
        from quant_rabbit.strategy.intent_generator import _risk_budgeted_units

        for legacy_units in ("3000", "9999"):
            with self.subTest(legacy_units=legacy_units):
                with patch.dict(
                    os.environ,
                    {
                        "QR_TRADER_DISABLE_SL_REPAIR": "1",
                        "QR_TRADER_BASE_UNITS": legacy_units,
                    },
                ):
                    os.environ.pop("QR_TRADER_POSITION_NAV_PCT", None)
                    units = _risk_budgeted_units(
                        "EUR_USD",
                        entry=1.17290,
                        sl=1.17490,
                        max_loss_jpy=2000.0,
                        snapshot=self._stub_snapshot(),
                    )

                self.assertEqual(units, 6369)

    def test_legacy_micro_lot_env_does_not_change_integer_sizing(self) -> None:
        import os
        from quant_rabbit.strategy.intent_generator import _risk_budgeted_units
        with patch.dict(os.environ, {"QR_ALLOW_TEST_MICRO_LOT": "1"}):
            units = _risk_budgeted_units(
                "EUR_USD",
                entry=1.17290,
                sl=1.17490,
                max_loss_jpy=50.0,
                snapshot=self._stub_snapshot(),
            )
        self.assertEqual(units, 159)

    def test_account_none_keeps_legacy_fallback_for_test_fixtures(self) -> None:
        # Fixture-style snapshot without an account must not trigger the
        # production floor — many legacy test fixtures construct snapshots
        # without an `AccountSummary` and rely on the historical
        # micro-unit fallback.
        from quant_rabbit.models import BrokerSnapshot, Quote
        from quant_rabbit.strategy.intent_generator import _risk_budgeted_units
        now = datetime.now(timezone.utc)
        no_account = BrokerSnapshot(
            fetched_at_utc=now,
            positions=(),
            orders=(),
            quotes={
                "EUR_USD": Quote(pair="EUR_USD", bid=1.17280, ask=1.17290, timestamp_utc=now),
                "USD_JPY": Quote(pair="USD_JPY", bid=157.0, ask=157.01, timestamp_utc=now),
            },
            home_conversions={"USD": 157.005},
            account=None,
        )
        units = _risk_budgeted_units(
            "EUR_USD",
            entry=1.17290,
            sl=1.17490,
            max_loss_jpy=50.0,
            snapshot=no_account,
        )
        self.assertGreater(units, 0)


class ExhaustionRangeChaseTest(unittest.TestCase):
    """Coverage for 2026-05-13 filter C: refuse same-direction entries
    after a 2σ-equivalent 24h range extension. Operates via
    `_method_context_issues` against the intent's metadata, so it
    fires at intent-generation time without touching open positions.
    """

    def _intent(
        self,
        *,
        side,
        sigma_mult,
        price_pct_24h,
        pair: str = "EUR_USD",
        metadata_extra: dict | None = None,
        order_type: str = "MARKET",
        method: str = "TREND_CONTINUATION",
        entry: float | None = None,
    ):
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Owner, Side, TradeMethod
        metadata = {
            "range_24h_sigma_multiple": sigma_mult,
            "price_percentile_24h": price_pct_24h,
        }
        if metadata_extra:
            metadata.update(metadata_extra)
        return OrderIntent(
            pair=pair,
            side=Side.LONG if side == "LONG" else Side.SHORT,
            order_type=OrderType.parse(order_type),
            units=5000,
            entry=entry,
            tp=1.18 if side == "LONG" else 1.17,
            sl=1.17 if side == "LONG" else 1.18,
            thesis="test thesis",
            owner=Owner.TRADER,
            market_context=MarketContext(
                regime="TREND_UP",
                narrative="test",
                chart_story="test",
                method=TradeMethod.parse(method),
                invalidation="sl trades",
            ),
            metadata=metadata,
        )

    def test_long_at_top_after_2sigma_range_blocks(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues
        intent = self._intent(side="LONG", sigma_mult=2.5, price_pct_24h=0.92)
        codes = {issue["code"] for issue in _method_context_issues(intent)}
        self.assertIn("EXHAUSTION_RANGE_CHASE", codes)

    def test_short_at_bottom_after_2sigma_range_blocks(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues
        intent = self._intent(side="SHORT", sigma_mult=2.5, price_pct_24h=0.08)
        codes = {issue["code"] for issue in _method_context_issues(intent)}
        self.assertIn("EXHAUSTION_RANGE_CHASE", codes)

    def test_recovery_hedge_market_at_extended_side_blocks(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues
        intent = self._intent(
            side="SHORT",
            sigma_mult=2.5,
            price_pct_24h=0.08,
            metadata_extra={"position_intent": "HEDGE", "hedge_recovery": True},
        )
        issue = next(
            issue for issue in _method_context_issues(intent) if issue["code"] == "EXHAUSTION_RANGE_CHASE"
        )
        self.assertEqual(issue["severity"], "BLOCK")

    def test_recovery_hedge_stop_at_extended_side_warns_instead_of_blocks(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues
        intent = self._intent(
            side="SHORT",
            sigma_mult=2.5,
            price_pct_24h=0.08,
            order_type="STOP-ENTRY",
            metadata_extra={"position_intent": "HEDGE", "hedge_recovery": True},
        )
        issue = next(
            issue for issue in _method_context_issues(intent) if issue["code"] == "EXHAUSTION_RANGE_CHASE"
        )
        self.assertEqual(issue["severity"], "WARN")

    def test_long_at_bottom_after_2sigma_range_passes(self) -> None:
        # LONG mean-reversion entry at low — not a chase.
        from quant_rabbit.strategy.intent_generator import _method_context_issues
        intent = self._intent(side="LONG", sigma_mult=2.5, price_pct_24h=0.10)
        codes = {issue["code"] for issue in _method_context_issues(intent)}
        self.assertNotIn("EXHAUSTION_RANGE_CHASE", codes)

    def test_sigma_below_threshold_passes(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues
        intent = self._intent(side="LONG", sigma_mult=1.5, price_pct_24h=0.97)
        codes = {issue["code"] for issue in _method_context_issues(intent)}
        self.assertNotIn("EXHAUSTION_RANGE_CHASE", codes)

    def test_missing_sigma_no_block(self) -> None:
        # AGENT_CONTRACT §3.5: no data → no filter.
        from quant_rabbit.strategy.intent_generator import _method_context_issues
        intent = self._intent(side="LONG", sigma_mult=None, price_pct_24h=0.97)
        codes = {issue["code"] for issue in _method_context_issues(intent)}
        self.assertNotIn("EXHAUSTION_RANGE_CHASE", codes)

    def test_failed_break_short_market_at_upper_retest_not_chase(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            side="SHORT",
            sigma_mult=3.0,
            price_pct_24h=0.25,
            method="BREAKOUT_FAILURE",
            metadata_extra={
                "tf_regime_map": {
                    "M5": {"range_position": 0.80},
                    "M15": {"range_position": 0.78},
                },
            },
        )
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertNotIn("EXHAUSTION_RANGE_CHASE", codes)

    def test_failed_break_short_stop_near_lower_side_still_chase(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            side="SHORT",
            sigma_mult=3.0,
            price_pct_24h=0.25,
            method="BREAKOUT_FAILURE",
            order_type="STOP-ENTRY",
            entry=1.16080,
            metadata_extra={
                "tf_regime_map": {
                    "M5": {"nearest_support": 1.16000, "nearest_resistance": 1.16400},
                    "M15": {"nearest_support": 1.15900, "nearest_resistance": 1.16500},
                },
            },
        )
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertIn("EXHAUSTION_RANGE_CHASE", codes)

    def test_failed_break_short_limit_at_upper_retest_not_chase(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            side="SHORT",
            sigma_mult=3.0,
            price_pct_24h=0.25,
            method="BREAKOUT_FAILURE",
            order_type="LIMIT",
            entry=1.16360,
            metadata_extra={
                "tf_regime_map": {
                    "M5": {"nearest_support": 1.16000, "nearest_resistance": 1.16400},
                    "M15": {"nearest_support": 1.15900, "nearest_resistance": 1.16500},
                },
            },
        )
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertNotIn("EXHAUSTION_RANGE_CHASE", codes)

    def test_failed_break_short_limit_at_broader_extreme_still_chases(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            side="SHORT",
            sigma_mult=12.7,
            price_pct_24h=0.022,
            method="BREAKOUT_FAILURE",
            order_type="LIMIT",
            entry=1.14672,
            metadata_extra={
                "price_percentile_7d": 0.0,
                "entry_price_percentile_24h": 0.0531,
                "entry_price_percentile_7d": 0.0288,
                "tf_regime_map": {
                    "M5": {"nearest_support": 1.14610, "nearest_resistance": 1.14677},
                    "M15": {"nearest_support": 1.14612, "nearest_resistance": 1.14669},
                },
            },
        )
        issues = _method_context_issues(intent)
        issue = next(issue for issue in issues if issue["code"] == "EXHAUSTION_RANGE_CHASE")

        self.assertEqual(issue["severity"], "BLOCK")
        self.assertIn("p7d=0.03", issue["message"])

    def test_range_rotation_short_at_tiny_upper_rail_but_broader_discount_still_chases(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            side="SHORT",
            sigma_mult=8.5,
            price_pct_24h=0.09,
            method="RANGE_ROTATION",
            order_type="LIMIT",
            entry=1.16089,
            metadata_extra={
                "price_percentile_7d": 0.04,
                "tf_regime_map": {
                    "M5": {"nearest_support": 1.15900, "nearest_resistance": 1.16100},
                    "M15": {"nearest_support": 1.15880, "nearest_resistance": 1.16120},
                },
            },
        )
        issues = _method_context_issues(intent)
        issue = next(issue for issue in issues if issue["code"] == "EXHAUSTION_RANGE_CHASE")

        self.assertEqual(issue["severity"], "BLOCK")

    def test_range_rotation_short_limit_uses_entry_percentile_for_retest(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            side="SHORT",
            sigma_mult=8.5,
            price_pct_24h=0.09,
            method="RANGE_ROTATION",
            order_type="LIMIT",
            entry=1.16089,
            metadata_extra={
                "price_percentile_7d": 0.04,
                "entry_price_percentile_24h": 0.72,
                "entry_price_percentile_7d": 0.66,
                "tf_regime_map": {
                    "M5": {"nearest_support": 1.15900, "nearest_resistance": 1.16100},
                    "M15": {"nearest_support": 1.15880, "nearest_resistance": 1.16120},
                },
            },
        )
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertNotIn("RANGE_ROTATION_BROADER_LOCATION_CHASE", codes)
        self.assertNotIn("EXHAUSTION_RANGE_CHASE", codes)

    def test_oanda_firepower_short_limit_can_use_one_broader_edge_horizon(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            side="SHORT",
            sigma_mult=None,
            price_pct_24h=0.80,
            method="RANGE_ROTATION",
            order_type="LIMIT",
            entry=1.16500,
            metadata_extra={
                "price_percentile_7d": 0.29,
                "entry_price_percentile_24h": 0.88,
                "entry_price_percentile_7d": 0.32,
                "forecast_direction": "RANGE",
                "forecast_confidence": 0.76,
                "forecast_raw_confidence": 0.90,
                "geometry_model": "RANGE_RAIL_LIMIT",
                "forecast_range_low_price": 1.15820,
                "forecast_range_high_price": 1.16620,
                "range_entry_side": "resistance",
                "range_tp_is_inside_box": True,
                "range_sl_outside_box": True,
                "range_breakout_direction": None,
                "attach_take_profit_on_fill": True,
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "opportunity_mode": "HARVEST",
                "positive_rotation_live_ready": True,
                "positive_rotation_mode": POSITIVE_ROTATION_OANDA_CAMPAIGN_FIREPOWER_MODE,
                "positive_rotation_oanda_campaign_firepower_vehicle_match": True,
                "positive_rotation_oanda_campaign_minimum_floor_reachable": True,
                "tf_regime_map": {
                    "M5": {"nearest_support": 1.16270, "nearest_resistance": 1.16520},
                    "M15": {"nearest_support": 1.16190, "nearest_resistance": 1.16580},
                },
            },
        )
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertNotIn("RANGE_ROTATION_BROADER_LOCATION_CHASE", codes)
        self.assertEqual(
            intent.metadata["range_rotation_broader_location_override"],
            "OANDA_CAMPAIGN_FIREPOWER_SINGLE_HORIZON_EDGE",
        )

    def test_oanda_firepower_short_limit_at_opposite_extreme_still_blocks(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            side="SHORT",
            sigma_mult=None,
            price_pct_24h=0.80,
            method="RANGE_ROTATION",
            order_type="LIMIT",
            entry=1.16500,
            metadata_extra={
                "price_percentile_7d": 0.02,
                "entry_price_percentile_24h": 0.88,
                "entry_price_percentile_7d": 0.03,
                "forecast_direction": "RANGE",
                "forecast_confidence": 0.76,
                "forecast_raw_confidence": 0.90,
                "geometry_model": "RANGE_RAIL_LIMIT",
                "forecast_range_low_price": 1.15820,
                "forecast_range_high_price": 1.16620,
                "range_entry_side": "resistance",
                "range_tp_is_inside_box": True,
                "range_sl_outside_box": True,
                "range_breakout_direction": None,
                "attach_take_profit_on_fill": True,
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "opportunity_mode": "HARVEST",
                "positive_rotation_live_ready": True,
                "positive_rotation_mode": POSITIVE_ROTATION_OANDA_CAMPAIGN_FIREPOWER_MODE,
                "positive_rotation_oanda_campaign_firepower_vehicle_match": True,
                "positive_rotation_oanda_campaign_minimum_floor_reachable": True,
                "tf_regime_map": {
                    "M5": {"nearest_support": 1.16270, "nearest_resistance": 1.16520},
                    "M15": {"nearest_support": 1.16190, "nearest_resistance": 1.16580},
                },
            },
        )
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertIn("RANGE_ROTATION_BROADER_LOCATION_CHASE", codes)
        self.assertNotIn("range_rotation_broader_location_override", intent.metadata)

    def test_range_rotation_long_at_broader_premium_blocks(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            side="LONG",
            sigma_mult=None,
            price_pct_24h=None,
            method="RANGE_ROTATION",
            order_type="LIMIT",
            entry=1.16670,
            metadata_extra={
                "price_percentile_7d": 0.92,
                "tf_regime_map": {
                    "M5": {"nearest_support": 1.16650, "nearest_resistance": 1.16740},
                    "M15": {"nearest_support": 1.16580, "nearest_resistance": 1.16820},
                },
            },
        )
        issues = _method_context_issues(intent)
        issue = next(issue for issue in issues if issue["code"] == "RANGE_ROTATION_BROADER_LOCATION_CHASE")

        self.assertEqual(issue["severity"], "BLOCK")

    def test_range_rotation_short_at_broader_premium_keeps_retest_carveout(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            side="SHORT",
            sigma_mult=7.0,
            price_pct_24h=0.86,
            method="RANGE_ROTATION",
            order_type="LIMIT",
            entry=1.16500,
            metadata_extra={
                "price_percentile_7d": 0.83,
                "tf_regime_map": {
                    "M5": {"nearest_support": 1.16380, "nearest_resistance": 1.16520},
                    "M15": {"nearest_support": 1.16270, "nearest_resistance": 1.16580},
                },
            },
        )
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertNotIn("EXHAUSTION_RANGE_CHASE", codes)

    def test_range_forming_long_against_strong_higher_tf_downtrend_blocks(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            side="LONG",
            sigma_mult=None,
            price_pct_24h=0.24,
            method="RANGE_ROTATION",
            order_type="LIMIT",
            entry=1.16030,
            metadata_extra={
                "range_phase": "RANGE_FORMING",
                "chart_direction_bias": "SHORT",
                "matrix_reject_count": 3,
                "forecast_direction": "RANGE",
                "forecast_confidence": 0.55,
                "forecast_raw_confidence": 0.55,
                "forecast_component_scores": {"UP": 55.0, "DOWN": 95.0, "RANGE": 7.7, "EITHER": 3.8},
                "geometry_model": "RANGE_RAIL_LIMIT",
                "forecast_range_low_price": 1.15980,
                "forecast_range_high_price": 1.16150,
                "range_entry_side": "support",
                "range_tp_is_inside_box": True,
                "range_sl_outside_box": True,
                "range_breakout_direction": None,
                "tf_regime_map": {
                    "H1": {"classification": "TREND_DOWN", "adx": 37.3},
                    "H4": {"classification": "TREND_DOWN", "adx": 26.8},
                    "M5": {"nearest_support": 1.16010, "nearest_resistance": 1.16110},
                    "M15": {"nearest_support": 1.15980, "nearest_resistance": 1.16150},
                },
            },
        )
        issue = next(
            issue for issue in _method_context_issues(intent) if issue["code"] == "RANGE_FORMING_HTF_TREND_CONFLICT"
        )

        self.assertEqual(issue["severity"], "BLOCK")
        self.assertIn("H1 TREND_DOWN", issue["message"])

    def test_range_forming_predictive_box_bypasses_htf_conflict_guard(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            side="LONG",
            sigma_mult=None,
            price_pct_24h=0.24,
            method="RANGE_ROTATION",
            order_type="LIMIT",
            entry=1.16030,
            metadata_extra={
                "range_phase": "RANGE_FORMING",
                "chart_direction_bias": "SHORT",
                "matrix_reject_count": 3,
                "forecast_direction": "RANGE",
                "forecast_confidence": 0.65,
                "forecast_raw_confidence": 0.65,
                "forecast_component_scores": {"UP": 42.0, "DOWN": 45.0, "RANGE": 8.0, "EITHER": 0.0},
                "geometry_model": "RANGE_RAIL_LIMIT",
                "forecast_range_low_price": 1.15980,
                "forecast_range_high_price": 1.16150,
                "range_entry_side": "support",
                "range_tp_is_inside_box": True,
                "range_sl_outside_box": True,
                "range_breakout_direction": None,
                "tf_regime_map": {
                    "H1": {"classification": "TREND_DOWN", "adx": 37.3},
                    "H4": {"classification": "TREND_DOWN", "adx": 26.8},
                    "M5": {"nearest_support": 1.16010, "nearest_resistance": 1.16110},
                    "M15": {"nearest_support": 1.15980, "nearest_resistance": 1.16150},
                },
            },
        )
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertNotIn("RANGE_FORMING_HTF_TREND_CONFLICT", codes)

    def test_range_forming_directional_margin_still_blocks_weak_range_thesis(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            side="LONG",
            sigma_mult=None,
            price_pct_24h=0.24,
            method="RANGE_ROTATION",
            order_type="LIMIT",
            entry=1.16030,
            metadata_extra={
                "range_phase": "RANGE_FORMING",
                "chart_direction_bias": "SHORT",
                "matrix_reject_count": 3,
                "forecast_direction": "RANGE",
                "forecast_confidence": 0.65,
                "forecast_raw_confidence": 0.65,
                "forecast_component_scores": {"UP": 45.0, "DOWN": 75.0, "RANGE": 12.0, "EITHER": 0.0},
                "geometry_model": "RANGE_RAIL_LIMIT",
                "forecast_range_low_price": 1.15980,
                "forecast_range_high_price": 1.16150,
                "range_entry_side": "support",
                "range_tp_is_inside_box": True,
                "range_sl_outside_box": True,
                "range_breakout_direction": None,
                "tf_regime_map": {
                    "H1": {"classification": "TREND_DOWN", "adx": 37.3},
                    "H4": {"classification": "TREND_DOWN", "adx": 26.8},
                    "M5": {"nearest_support": 1.16010, "nearest_resistance": 1.16110},
                    "M15": {"nearest_support": 1.15980, "nearest_resistance": 1.16150},
                },
            },
        )
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertIn("RANGE_FORMING_HTF_TREND_CONFLICT", codes)

    def test_range_forming_oanda_firepower_edge_bypasses_htf_conflict_guard(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            side="SHORT",
            sigma_mult=None,
            price_pct_24h=0.96,
            method="RANGE_ROTATION",
            order_type="LIMIT",
            entry=1.16100,
            metadata_extra={
                "range_phase": "RANGE_FORMING",
                "chart_direction_bias": "LONG",
                "matrix_reject_count": 4,
                "forecast_direction": "RANGE",
                "forecast_confidence": 0.55,
                "forecast_raw_confidence": 0.91,
                "forecast_component_scores": {"UP": 83.0, "DOWN": 48.0, "RANGE": 26.5, "EITHER": 0.0},
                "geometry_model": "RANGE_RAIL_LIMIT",
                "forecast_range_low_price": 1.15820,
                "forecast_range_high_price": 1.16120,
                "range_entry_side": "resistance",
                "range_tp_is_inside_box": True,
                "range_sl_outside_box": True,
                "range_breakout_direction": None,
                "entry_price_percentile_24h": 0.96,
                "entry_price_percentile_7d": 1.0,
                "attach_take_profit_on_fill": True,
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "opportunity_mode": "HARVEST",
                "positive_rotation_live_ready": True,
                "positive_rotation_mode": POSITIVE_ROTATION_OANDA_CAMPAIGN_FIREPOWER_MODE,
                "positive_rotation_oanda_campaign_firepower_vehicle_match": True,
                "positive_rotation_oanda_campaign_minimum_floor_reachable": True,
                "tf_regime_map": {
                    "H1": {"classification": "TREND_UP", "adx": 22.3},
                    "H4": {"classification": "TREND_UP", "adx": 37.3},
                    "M5": {"nearest_support": 1.15980, "nearest_resistance": 1.16120},
                    "M15": {"nearest_support": 1.15890, "nearest_resistance": 1.16130},
                },
            },
        )
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertNotIn("RANGE_FORMING_HTF_TREND_CONFLICT", codes)
        self.assertEqual(
            intent.metadata["range_forming_htf_conflict_override"],
            "OANDA_CAMPAIGN_FIREPOWER_RANGE_EDGE",
        )

    def test_range_forming_oanda_firepower_without_broader_edge_still_blocks(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            side="SHORT",
            sigma_mult=None,
            price_pct_24h=0.56,
            method="RANGE_ROTATION",
            order_type="LIMIT",
            entry=1.16100,
            metadata_extra={
                "range_phase": "RANGE_FORMING",
                "chart_direction_bias": "LONG",
                "matrix_reject_count": 4,
                "forecast_direction": "RANGE",
                "forecast_confidence": 0.55,
                "forecast_raw_confidence": 0.91,
                "geometry_model": "RANGE_RAIL_LIMIT",
                "forecast_range_low_price": 1.15820,
                "forecast_range_high_price": 1.16120,
                "range_entry_side": "resistance",
                "range_tp_is_inside_box": True,
                "range_sl_outside_box": True,
                "range_breakout_direction": None,
                "entry_price_percentile_24h": 0.56,
                "entry_price_percentile_7d": 0.58,
                "attach_take_profit_on_fill": True,
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "opportunity_mode": "HARVEST",
                "positive_rotation_live_ready": True,
                "positive_rotation_mode": POSITIVE_ROTATION_OANDA_CAMPAIGN_FIREPOWER_MODE,
                "positive_rotation_oanda_campaign_firepower_vehicle_match": True,
                "positive_rotation_oanda_campaign_minimum_floor_reachable": True,
                "tf_regime_map": {
                    "H1": {"classification": "TREND_UP", "adx": 22.3},
                    "H4": {"classification": "TREND_UP", "adx": 37.3},
                    "M5": {"nearest_support": 1.15980, "nearest_resistance": 1.16120},
                    "M15": {"nearest_support": 1.15890, "nearest_resistance": 1.16130},
                },
            },
        )
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertIn("RANGE_FORMING_HTF_TREND_CONFLICT", codes)
        self.assertNotIn("range_forming_htf_conflict_override", intent.metadata)

    def test_range_forming_trend_aligned_rotation_passes_htf_conflict_guard(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            side="SHORT",
            sigma_mult=None,
            price_pct_24h=0.72,
            method="RANGE_ROTATION",
            order_type="LIMIT",
            entry=1.16100,
            metadata_extra={
                "range_phase": "RANGE_FORMING",
                "chart_direction_bias": "SHORT",
                "matrix_reject_count": 0,
                "tf_regime_map": {
                    "H1": {"classification": "TREND_DOWN", "adx": 37.3},
                    "H4": {"classification": "TREND_DOWN", "adx": 26.8},
                    "M5": {"nearest_support": 1.16010, "nearest_resistance": 1.16110},
                    "M15": {"nearest_support": 1.15980, "nearest_resistance": 1.16150},
                },
            },
        )
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertNotIn("RANGE_FORMING_HTF_TREND_CONFLICT", codes)

    def test_stable_range_keeps_rotation_despite_higher_tf_trend(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            side="LONG",
            sigma_mult=None,
            price_pct_24h=0.24,
            method="RANGE_ROTATION",
            order_type="LIMIT",
            entry=1.16030,
            metadata_extra={
                "range_phase": "IN_RANGE",
                "chart_direction_bias": "SHORT",
                "matrix_reject_count": 3,
                "tf_regime_map": {
                    "H1": {"classification": "TREND_DOWN", "adx": 37.3},
                    "H4": {"classification": "TREND_DOWN", "adx": 26.8},
                    "M5": {"nearest_support": 1.16010, "nearest_resistance": 1.16110},
                    "M15": {"nearest_support": 1.15980, "nearest_resistance": 1.16150},
                },
            },
        )
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertNotIn("RANGE_FORMING_HTF_TREND_CONFLICT", codes)

    def test_geometry_metadata_publishes_limit_entry_percentiles(self) -> None:
        from quant_rabbit.models import OrderType, Quote, Side
        from quant_rabbit.strategy.intent_generator import _geometry_metadata

        quote = Quote(
            pair="EUR_USD",
            bid=1.15220,
            ask=1.15228,
            timestamp_utc=datetime.now(timezone.utc),
        )
        metadata = _geometry_metadata(
            "EUR_USD",
            Side.SHORT,
            OrderType.LIMIT,
            quote,
            entry=1.16000,
            tp=1.15400,
            sl=1.16200,
            range_indicators={
                "donchian_low": 1.15000,
                "donchian_high": 1.16100,
            },
            chart_indicators={
                "donchian_low": 1.15000,
                "donchian_high": 1.16100,
            },
            chart_context={
                "price_range_24h_low": 1.15000,
                "price_range_24h_high": 1.17000,
                "price_range_24h_source": "confluence_24h",
                "price_range_7d_low": 1.14000,
                "price_range_7d_high": 1.18000,
                "price_range_7d_source": "confluence_7d",
            },
            atr_pips=8.0,
        )

        self.assertEqual(metadata["entry_price_percentile_24h"], 0.5)
        self.assertEqual(metadata["entry_price_percentile_7d"], 0.5)
        self.assertEqual(metadata["entry_price_percentile_24h_source"], "confluence_24h")


class BreakoutFailureStopChaseTest(unittest.TestCase):
    def _intent(self, *, side: str, order_type: str, entry: float):
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Owner, Side, TradeMethod

        return OrderIntent(
            pair="EUR_USD",
            side=Side.LONG if side == "LONG" else Side.SHORT,
            order_type=OrderType.parse(order_type),
            units=3000,
            entry=entry,
            tp=entry + 0.0018 if side == "LONG" else entry - 0.0018,
            sl=entry - 0.0012 if side == "LONG" else entry + 0.0012,
            thesis="failed-break retest timing test",
            owner=Owner.TRADER,
            market_context=MarketContext(
                regime="FAILURE_RISK",
                narrative="breakout failure lane",
                chart_story="failed break and retest map",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="sl trades",
            ),
            metadata={
                "tf_regime_map": {
                    "M5": {
                        "nearest_support": 1.16000,
                        "nearest_resistance": 1.16400,
                        "nearest_support_distance_pips": -20.0,
                        "nearest_resistance_distance_pips": 20.0,
                    },
                    "M15": {
                        "nearest_support": 1.15900,
                        "nearest_resistance": 1.16500,
                        "nearest_support_distance_pips": -30.0,
                        "nearest_resistance_distance_pips": 30.0,
                    },
                },
            },
        )

    def test_short_stop_entry_on_lower_half_blocks(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(side="SHORT", order_type="STOP-ENTRY", entry=1.16080)
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertIn("BREAKOUT_FAILURE_STOP_CHASES_FAILED_SIDE", codes)

    def test_long_stop_entry_on_upper_half_blocks(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(side="LONG", order_type="STOP-ENTRY", entry=1.16320)
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertIn("BREAKOUT_FAILURE_STOP_CHASES_FAILED_SIDE", codes)

    def test_short_limit_at_resistance_does_not_get_stop_chase_block(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(side="SHORT", order_type="LIMIT", entry=1.16360)
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertNotIn("BREAKOUT_FAILURE_STOP_CHASES_FAILED_SIDE", codes)


class PatternReversalChaseTest(unittest.TestCase):
    def _intent(
        self,
        *,
        side: str = "SHORT",
        order_type: str = "STOP-ENTRY",
        method: str = "TREND_CONTINUATION",
        metadata_extra: dict | None = None,
    ):
        from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Owner, Side, TradeMethod

        entry = 1.16080
        metadata = {
            "pattern_reversal_dominant_side": "LONG",
            "pattern_reversal_weight_long": 22.5,
            "pattern_reversal_weight_short": 6.0,
            "pattern_signals": [
                {
                    "name": "failed_breakout",
                    "timeframe": "M15",
                    "direction": "UP",
                    "side": "LONG",
                    "weight": 11.25,
                    "chase_block_evidence": True,
                    "rationale": "M15 BOS_DOWN wick-only (close_confirmed=False) -> trap fade UP",
                },
                {
                    "name": "hammer",
                    "timeframe": "M5",
                    "direction": "UP",
                    "side": "LONG",
                    "weight": 11.25,
                    "chase_block_evidence": True,
                    "rationale": "M5 hammer at lower rail -> UP",
                },
                {
                    "name": "aroon_strong_down",
                    "timeframe": "M15",
                    "direction": "DOWN",
                    "side": "SHORT",
                    "weight": 6.0,
                    "chase_block_evidence": False,
                    "rationale": "M15 aroon momentum DOWN",
                },
            ],
        }
        if metadata_extra:
            metadata.update(metadata_extra)
        return OrderIntent(
            pair="EUR_USD",
            side=Side.LONG if side == "LONG" else Side.SHORT,
            order_type=OrderType.parse(order_type),
            units=3000,
            entry=entry,
            tp=entry + 0.0018 if side == "LONG" else entry - 0.0018,
            sl=entry - 0.0012 if side == "LONG" else entry + 0.0012,
            thesis="pattern reversal chase test",
            owner=Owner.TRADER,
            market_context=MarketContext(
                regime="TREND_DOWN",
                narrative="trend lane",
                chart_story="failed break / candle-shape evidence",
                method=TradeMethod.parse(method),
                invalidation="sl trades",
            ),
            metadata=metadata,
        )

    def test_short_stop_entry_against_failed_break_reversal_blocks(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent()
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertIn("PATTERN_REVERSAL_CHASE", codes)

    def test_breakout_failure_market_against_reversal_blocks(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(order_type="MARKET", method="BREAKOUT_FAILURE")
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertIn("PATTERN_REVERSAL_CHASE", codes)

    def test_retest_limit_does_not_get_pattern_chase_block(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(order_type="LIMIT", method="BREAKOUT_FAILURE")
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertNotIn("PATTERN_REVERSAL_CHASE", codes)

    def test_close_confirmed_operating_break_allows_continuation(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            metadata_extra={
                "chart_story_structural": (
                    "EUR_USD TREND_DOWN; "
                    "M5(TREND_DOWN struct=BOS_DOWN@1.1580); "
                    "M15(TREND_DOWN struct=CHOCH_DOWN@1.1575)"
                ),
                "m5_long_bias": 0.30,
                "m5_short_bias": 0.70,
                "m15_long_bias": 0.35,
                "m15_short_bias": 0.65,
            }
        )
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertNotIn("PATTERN_REVERSAL_CHASE", codes)

    def test_recovery_hedge_against_reversal_warns(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(metadata_extra={"position_intent": "HEDGE", "hedge_recovery": True})
        issue = next(issue for issue in _method_context_issues(intent) if issue["code"] == "PATTERN_REVERSAL_CHASE")

        self.assertEqual(issue["severity"], "WARN")

    def test_recovery_hedge_market_against_reversal_blocks(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            order_type="MARKET",
            metadata_extra={"position_intent": "HEDGE", "hedge_recovery": True},
        )
        issue = next(issue for issue in _method_context_issues(intent) if issue["code"] == "PATTERN_REVERSAL_CHASE")

        self.assertEqual(issue["severity"], "BLOCK")

    def test_recovery_hedge_market_buy_into_short_reversal_and_exhaustion_blocks(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            side="LONG",
            order_type="MARKET",
            metadata_extra={
                "position_intent": "HEDGE",
                "hedge_recovery": True,
                "hedge_timing_class": "REVERSAL",
                "pattern_reversal_dominant_side": "SHORT",
                "pattern_reversal_weight_long": 33.75,
                "pattern_reversal_weight_short": 39.11,
                "pattern_signals": [
                    {
                        "name": "rsi_extreme_top",
                        "timeframe": "M5",
                        "direction": "DOWN",
                        "side": "SHORT",
                        "weight": 11.25,
                        "chase_block_evidence": True,
                        "rationale": "M5 top exhaustion -> DOWN",
                    },
                ],
                "range_24h_sigma_multiple": 8.657,
                "price_percentile_24h": 0.83,
            },
        )
        issues = {issue["code"]: issue for issue in _method_context_issues(intent)}

        self.assertEqual(issues["PATTERN_REVERSAL_CHASE"]["severity"], "BLOCK")
        self.assertEqual(issues["EXHAUSTION_RANGE_CHASE"]["severity"], "BLOCK")

    def test_aligned_pattern_dominance_passes(self) -> None:
        from quant_rabbit.strategy.intent_generator import _method_context_issues

        intent = self._intent(
            metadata_extra={
                "pattern_reversal_dominant_side": "SHORT",
                "pattern_reversal_weight_long": 6.0,
                "pattern_reversal_weight_short": 22.5,
                "pattern_signals": [
                    {
                        "name": "shooting_star",
                        "timeframe": "M15",
                        "direction": "DOWN",
                        "side": "SHORT",
                        "weight": 11.25,
                        "chase_block_evidence": True,
                        "rationale": "M15 shooting star -> DOWN",
                    }
                ],
            }
        )
        codes = {issue["code"] for issue in _method_context_issues(intent)}

        self.assertNotIn("PATTERN_REVERSAL_CHASE", codes)


class TimingEvidenceBreakoutStopTest(unittest.TestCase):
    """_forecast_market_support_allows_side breakout-proof path (§8, 2026-06-10).

    Projection-ledger truth: timing detectors (squeeze/session expansion) hit
    77-88% while the aggregate directional forecast reaches its target only
    ~43-50%. A resting STOP-ENTRY beyond the rail only fills when the market
    itself breaks, so it may bypass the confidence floor when the audited
    timing evidence and the structural lean both agree with the lane side.
    """

    @staticmethod
    def _metadata(
        *,
        direction: str = "UP",
        confidence: float = 0.47,
        raw_confidence: float | None = None,
        bias: str = "LONG",
        timing_count: int = 1,
        hit_rate: float = 1.0,
        samples: int = 40,
        support_ok: bool = False,
    ) -> dict:
        return {
            "forecast_direction": direction,
            "forecast_confidence": confidence,
            "forecast_raw_confidence": confidence if raw_confidence is None else raw_confidence,
            "chart_direction_bias": bias,
            "forecast_market_support": {
                "ok": support_ok,
                "aligned_projection_count": 0,
                "timing_projection_count": timing_count,
                "best_hit_rate": hit_rate,
                "best_samples": samples,
                "bootstrap_projection_support": False,
                "signals": [
                    {
                        "name": "session_expansion_london",
                        "direction": "EITHER",
                        "confidence": 0.91,
                        "hit_rate": hit_rate,
                        "samples": samples,
                    }
                ],
            },
        }

    @staticmethod
    def _strong_directional_metadata(
        *,
        confidence: float = 0.45,
        raw_confidence: float = 0.63,
    ) -> dict:
        metadata = TimingEvidenceBreakoutStopTest._metadata(
            direction="DOWN",
            confidence=confidence,
            raw_confidence=raw_confidence,
            bias="SHORT",
            timing_count=2,
            hit_rate=1.0,
            samples=37,
            support_ok=True,
        )
        metadata["forecast_market_support"]["aligned_projection_count"] = 2
        metadata["forecast_market_support"]["direction"] = "DOWN"
        metadata["forecast_market_support"]["signals"] = [
            {
                "name": "macro_event_nowcast_central_bank",
                "calibration_name": "macro_event_nowcast_central_bank_down",
                "direction": "DOWN",
                "confidence": 0.79,
                "hit_rate": 1.0,
                "samples": 37,
                "timeframe": None,
            },
            {
                "name": "bb_squeeze_expansion_imminent",
                "calibration_name": "bb_squeeze_expansion_imminent",
                "direction": "EITHER",
                "confidence": 0.63,
                "hit_rate": 0.92,
                "samples": 100,
                "timeframe": "M30",
            },
        ]
        return metadata

    def _allows(
        self,
        metadata: dict,
        *,
        side: str = "LONG",
        order_type: OrderType | None = OrderType.STOP_ENTRY,
        method: TradeMethod | None = None,
        min_confidence: float = 0.55,
    ) -> bool:
        from quant_rabbit.strategy.intent_generator import _forecast_market_support_allows_side

        return _forecast_market_support_allows_side(
            side,
            metadata,
            min_confidence=min_confidence,
            order_type=order_type,
            method=method,
        )

    def test_stop_entry_with_timing_evidence_and_lean_bypasses_floor(self) -> None:
        self.assertTrue(self._allows(self._metadata()))

    def test_deep_weak_timing_stop_remains_watch_only(self) -> None:
        # Regression for a live CAD_JPY shape: a strong EITHER/timing signal
        # predicts expansion, but not direction. Deeply weak calibrated direction
        # must stay watch-only instead of becoming LIVE_READY.
        self.assertFalse(
            self._allows(
                self._metadata(
                    confidence=0.38,
                    raw_confidence=0.57,
                    timing_count=2,
                    hit_rate=0.92,
                    samples=100,
                )
            )
        )

    def test_timing_only_stop_does_not_rescue_known_weak_direction_bucket(self) -> None:
        metadata = self._metadata(
            confidence=0.58,
            raw_confidence=0.66,
            timing_count=1,
            hit_rate=0.88,
            samples=500,
            support_ok=True,
        )
        metadata["forecast_directional_calibration_name"] = "directional_forecast_up"
        metadata["forecast_directional_hit_rate"] = 0.12
        metadata["forecast_directional_samples"] = 18

        self.assertFalse(self._allows(metadata, min_confidence=0.65))

    def test_strong_directional_support_rescues_stop_entry_below_near_miss_floor(self) -> None:
        # CAD_JPY live shape: final calibration is below the 0.10 near-miss
        # band, but raw forecast remains near the floor and audited
        # same-direction macro support is strong. Only STOP-ENTRY gets the
        # confirmation lift.
        self.assertTrue(
            self._allows(
                self._strong_directional_metadata(),
                side="SHORT",
                order_type=OrderType.STOP_ENTRY,
                min_confidence=0.65,
            )
        )

    def test_strong_directional_support_does_not_rescue_known_weak_direction_bucket(self) -> None:
        metadata = self._strong_directional_metadata()
        metadata["forecast_directional_calibration_name"] = "directional_forecast_down"
        metadata["forecast_directional_hit_rate"] = 0.12
        metadata["forecast_directional_samples"] = 37

        self.assertFalse(
            self._allows(
                metadata,
                side="SHORT",
                order_type=OrderType.STOP_ENTRY,
                min_confidence=0.65,
            )
        )

    def test_range_edge_stop_entry_does_not_get_support_override(self) -> None:
        # EUR_CHF live-loss shape: a weak calibrated LONG forecast had strong
        # macro support, but M5/M15 were boxed near the upper rail. A buy-stop
        # there is not breakout proof; it is a range-edge fakeout candidate.
        metadata = self._strong_directional_metadata(
            confidence=0.45,
            raw_confidence=0.63,
        )
        metadata.update(
            {
                "forecast_direction": "UP",
                "chart_direction_bias": "LONG",
                "m5_regime": "RANGE",
                "range_phase": "RANGE_STABLE",
                "tf_regime_map": {
                    "M5": {"classification": "RANGE", "range_position": 0.95},
                    "M15": {"classification": "RANGE", "range_position": 0.81},
                },
            }
        )
        metadata["forecast_market_support"]["direction"] = "UP"
        metadata["forecast_market_support"]["signals"][0].update(
            {
                "direction": "UP",
                "calibration_name": "macro_event_nowcast_central_bank_up",
            }
        )

        self.assertFalse(
            self._allows(
                metadata,
                side="LONG",
                order_type=OrderType.STOP_ENTRY,
                min_confidence=0.65,
            )
        )

    def test_strong_directional_support_must_be_inside_forecast_horizon(self) -> None:
        # EUR_CHF 472445 shape: a multi-day macro event nowcast must not rescue
        # a weak intraday forecast-first STOP entry.
        metadata = self._strong_directional_metadata(
            confidence=0.46,
            raw_confidence=0.66,
        )
        metadata.update(
            {
                "forecast_direction": "UP",
                "forecast_horizon_min": 180,
                "chart_direction_bias": "LONG",
                "m5_regime": "TREND_UP",
            }
        )
        metadata["forecast_market_support"]["direction"] = "UP"
        metadata["forecast_market_support"]["signals"][0].update(
            {
                "direction": "UP",
                "calibration_name": "macro_event_nowcast_central_bank_up",
                "lead_time_min": 3797.0,
            }
        )

        self.assertFalse(
            self._allows(
                metadata,
                side="LONG",
                order_type=OrderType.STOP_ENTRY,
                min_confidence=0.65,
            )
        )

    def test_strong_directional_support_requires_raw_forecast_near_floor(self) -> None:
        self.assertFalse(
            self._allows(
                self._strong_directional_metadata(raw_confidence=0.59),
                side="SHORT",
                order_type=OrderType.STOP_ENTRY,
                min_confidence=0.65,
            )
        )

    def test_strong_directional_support_does_not_rescue_market_chase(self) -> None:
        self.assertFalse(
            self._allows(
                self._strong_directional_metadata(),
                side="SHORT",
                order_type=OrderType.MARKET,
                min_confidence=0.65,
            )
        )

    def test_strong_directional_support_does_not_rescue_deeply_weak_forecast(self) -> None:
        self.assertFalse(
            self._allows(
                self._strong_directional_metadata(confidence=0.26, raw_confidence=0.67),
                side="SHORT",
                order_type=OrderType.STOP_ENTRY,
                min_confidence=0.65,
            )
        )

    def test_either_best_signal_does_not_count_as_strong_directional_support(self) -> None:
        metadata = self._strong_directional_metadata()
        metadata["forecast_market_support"]["signals"][0]["hit_rate"] = 0.57
        self.assertFalse(
            self._allows(
                metadata,
                side="SHORT",
                order_type=OrderType.STOP_ENTRY,
                min_confidence=0.65,
            )
        )

    def test_either_timing_hit_rate_cannot_launder_weak_aligned_market_support(self) -> None:
        metadata = self._metadata(
            direction="UP",
            confidence=0.60,
            raw_confidence=0.60,
            bias="LONG",
            timing_count=1,
            hit_rate=0.90,
            samples=100,
            support_ok=True,
        )
        metadata["forecast_market_support"].update(
            {
                "aligned_projection_count": 1,
                "best_aligned_hit_rate": 0.40,
                "best_aligned_samples": 100,
                "best_timing_hit_rate": 0.90,
                "best_timing_samples": 100,
            }
        )

        self.assertFalse(
            self._allows(
                metadata,
                side="LONG",
                order_type=OrderType.MARKET,
                min_confidence=0.65,
            )
        )

    def test_market_order_keeps_confidence_floor(self) -> None:
        self.assertFalse(self._allows(self._metadata(), order_type=OrderType.MARKET))

    def test_limit_order_keeps_confidence_floor(self) -> None:
        self.assertFalse(self._allows(self._metadata(), order_type=OrderType.LIMIT))

    def test_failed_break_limit_with_strong_directional_support_bypasses_weak_floor(self) -> None:
        metadata = self._strong_directional_metadata(
            confidence=0.23,
            raw_confidence=0.53,
        )
        metadata.update(
            {
                "chart_direction_bias": "SHORT",
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "opportunity_mode": "HARVEST",
            }
        )

        self.assertTrue(
            self._allows(
                metadata,
                side="SHORT",
                order_type=OrderType.LIMIT,
                method=TradeMethod.BREAKOUT_FAILURE,
                min_confidence=0.65,
            )
        )

    def test_failed_break_limit_requires_raw_directional_edge(self) -> None:
        metadata = self._strong_directional_metadata(
            confidence=0.23,
            raw_confidence=0.51,
        )
        metadata.update(
            {
                "chart_direction_bias": "SHORT",
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "opportunity_mode": "HARVEST",
            }
        )

        self.assertFalse(
            self._allows(
                metadata,
                side="SHORT",
                order_type=OrderType.LIMIT,
                method=TradeMethod.BREAKOUT_FAILURE,
                min_confidence=0.65,
            )
        )

    def test_failed_break_market_still_keeps_confidence_floor(self) -> None:
        metadata = self._strong_directional_metadata(
            confidence=0.23,
            raw_confidence=0.53,
        )
        metadata.update(
            {
                "chart_direction_bias": "SHORT",
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "opportunity_mode": "HARVEST",
            }
        )

        self.assertFalse(
            self._allows(
                metadata,
                side="SHORT",
                order_type=OrderType.MARKET,
                method=TradeMethod.BREAKOUT_FAILURE,
                min_confidence=0.65,
            )
        )

    def test_missing_structural_lean_fails_closed(self) -> None:
        self.assertFalse(self._allows(self._metadata(bias="")))

    def test_opposing_structural_lean_blocks(self) -> None:
        self.assertFalse(self._allows(self._metadata(bias="SHORT")))

    def test_side_must_still_match_forecast_direction(self) -> None:
        # §5 alignment preserved: a SHORT lane cannot use the breakout path
        # against an UP forecast even with strong timing evidence.
        self.assertFalse(
            self._allows(self._metadata(direction="UP", bias="SHORT"), side="SHORT")
        )

    def test_weak_timing_hit_rate_blocks(self) -> None:
        self.assertFalse(self._allows(self._metadata(hit_rate=0.6)))

    def test_thin_samples_block(self) -> None:
        self.assertFalse(self._allows(self._metadata(samples=2)))

    def test_no_timing_signal_blocks(self) -> None:
        self.assertFalse(self._allows(self._metadata(timing_count=0)))

    def test_near_miss_path_unchanged_for_market_orders(self) -> None:
        # Live support now requires signal-level Wilson>=90% proof; aggregate
        # support.ok alone must not rescue MARKET orders.
        metadata = self._metadata(confidence=0.47, support_ok=True, timing_count=0)
        metadata["forecast_market_support"]["aligned_projection_count"] = 1
        metadata["forecast_market_support"]["best_hit_rate"] = 0.62
        self.assertFalse(self._allows(metadata, order_type=OrderType.MARKET))


class DisasterSlMetadataTest(unittest.TestCase):
    """_disaster_sl_metadata — §3.5-K catastrophe bound (2026-06-11)."""

    def setUp(self) -> None:
        self._prior = {k: os.environ.get(k) for k in ("QR_DISASTER_SL",)}
        os.environ["QR_DISASTER_SL"] = "1"

    def tearDown(self) -> None:
        for k, v in self._prior.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    @staticmethod
    def _compute(side: Side, *, chart_context: dict | None, entry=1.1500, expected_sl=None):
        from quant_rabbit.strategy.intent_generator import _disaster_sl_metadata

        if expected_sl is None:
            expected_sl = 1.1470 if side == Side.LONG else 1.1530
        return _disaster_sl_metadata(
            "EUR_USD", side, entry=entry, expected_sl=expected_sl, chart_context=chart_context
        )

    def test_long_disaster_below_entry_at_h4_atr_multiple(self) -> None:
        meta = self._compute(
            Side.LONG,
            chart_context={"h4_atr_pips": 30.0, "session_current_tag": "LONDON"},
        )
        # 30 × 2.5 × 1.0 = 75 pips below entry 1.1500 → 1.14250
        self.assertAlmostEqual(meta["disaster_sl_pips"], 75.0)
        self.assertAlmostEqual(meta["disaster_sl"], 1.1425)

    def test_short_disaster_above_entry_with_thin_session_widening(self) -> None:
        meta = self._compute(
            Side.SHORT,
            chart_context={"h4_atr_pips": 30.0, "session_current_tag": "TOKYO"},
        )
        # 30 × 2.5 × 1.3 = 97.5 pips above entry → 1.15975
        self.assertAlmostEqual(meta["disaster_sl_pips"], 97.5)
        self.assertAlmostEqual(meta["disaster_sl"], 1.15975)

    def test_missing_h4_atr_fails_loud_not_silent(self) -> None:
        meta = self._compute(Side.LONG, chart_context={})
        self.assertEqual(meta, {"disaster_sl_missing": "H4_ATR_MISSING"})

    def test_disaster_always_beyond_expected_stop(self) -> None:
        # Tiny H4 ATR (5 pips → 12.5p disaster) vs a 30-pip expected stop:
        # the strict-ordering buffer lifts the disaster to 30 × 1.25 = 37.5p.
        meta = self._compute(
            Side.LONG,
            chart_context={"h4_atr_pips": 5.0, "session_current_tag": "LONDON"},
        )
        self.assertAlmostEqual(meta["disaster_sl_pips"], 37.5)
        self.assertLess(meta["disaster_sl"], 1.1470)

    def test_env_off_returns_empty(self) -> None:
        os.environ["QR_DISASTER_SL"] = "0"
        meta = self._compute(
            Side.LONG,
            chart_context={"h4_atr_pips": 30.0, "session_current_tag": "LONDON"},
        )
        self.assertEqual(meta, {})


class SizingConversionSnapshotReceiptTest(unittest.TestCase):
    @staticmethod
    def _snapshot() -> dict[str, object]:
        timestamp = "2026-07-14T00:00:00Z"
        return {
            "fetched_at_utc": timestamp,
            "quotes": {
                "EUR_USD": {
                    "bid": 1.0999,
                    "ask": 1.1001,
                    "timestamp_utc": timestamp,
                },
                "USD_JPY": {
                    "bid": 157.2,
                    "ask": 157.21,
                    "timestamp_utc": timestamp,
                },
            },
            "home_conversions": {},
        }

    def test_receipt_uses_canonical_broker_conversion_leg(self) -> None:
        receipt = sizing_conversion_snapshot_receipt_from_payload(
            "EUR_USD",
            self._snapshot(),
        )

        self.assertIsNotNone(receipt)
        assert receipt is not None
        source = receipt["conversion_source"]
        self.assertEqual(source["kind"], "BROKER_CONVERSION_QUOTE")
        self.assertEqual(source["instrument"], "USD_JPY")
        self.assertEqual(source["orientation"], "QUOTE_CURRENCY_TO_JPY")
        self.assertEqual(source["selected_price_rule"], "MAX_BID_ASK")
        self.assertEqual(source["selected_quote_to_jpy"], 157.21)
        self.assertEqual(len(receipt["snapshot_conversion_sha256"]), 64)

    def test_raw_snapshot_parser_rejects_coercible_or_defaultable_evidence(
        self,
    ) -> None:
        variants: list[tuple[str, dict[str, object]]] = []

        def mutated(label: str) -> dict[str, object]:
            payload = json.loads(json.dumps(self._snapshot()))
            variants.append((label, payload))
            return payload

        missing_fetched = mutated("missing-fetched-at")
        missing_fetched.pop("fetched_at_utc")

        naive_fetched = mutated("naive-fetched-at")
        naive_fetched["fetched_at_utc"] = "2026-07-14T00:00:00"

        string_bid = mutated("string-bid")
        string_bid["quotes"]["EUR_USD"]["bid"] = "1.0999"  # type: ignore[index]

        bool_bid = mutated("bool-bid")
        bool_bid["quotes"]["EUR_USD"]["bid"] = True  # type: ignore[index]

        missing_timestamp = mutated("missing-quote-timestamp")
        missing_timestamp["quotes"]["EUR_USD"].pop("timestamp_utc")  # type: ignore[index]

        string_home = mutated("string-home-conversion")
        string_home["home_conversions"] = {"USD": "157.2"}

        wrong_embedded_pair = mutated("wrong-embedded-pair")
        wrong_embedded_pair["quotes"]["USD_JPY"]["pair"] = "JPY_USD"  # type: ignore[index]

        for label, payload in variants:
            with self.subTest(variant=label):
                self.assertIsNone(
                    sizing_conversion_snapshot_receipt_from_payload(
                        "EUR_USD",
                        payload,
                    )
                )


class M15RecoveryMicroIntentTest(unittest.TestCase):
    @staticmethod
    def _evidence():
        from tests.test_directional_forecaster import (
            TechnicalCandleIntegrityForecastGateTest,
        )

        now = datetime(2026, 7, 14, 0, 45, tzinfo=timezone.utc)
        chart = TechnicalCandleIntegrityForecastGateTest._m15_recovery_chart()
        receipt = build_m15_recovery_micro_receipt(
            chart,
            expected_pair="EUR_USD",
            now_utc=now,
            current_spread_pips=0.8,
        )
        assert isinstance(receipt, dict)
        m15_view = next(
            view
            for view in chart["views"]
            if isinstance(view, dict) and view.get("granularity") == "M15"
        )
        return now, chart, receipt, m15_view

    @staticmethod
    def _lane(receipt: dict[str, object]) -> dict[str, object]:
        from quant_rabbit.strategy.m15_recovery_contract import (
            build_forecast_binding,
            canonical_sha256,
        )

        cycle_id = "m15-recovery-intent-test"
        scores = {"UP": 26.4, "DOWN": 8.4, "RANGE": 5.2}
        evidence_body = {
            "contract": "QR_M15_RECOVERY_FORECAST_V1",
            "source_recovery_receipt_sha256": receipt["receipt_sha256"],
            "pair": "EUR_USD",
            "chart_generated_at_utc": receipt["chart_generated_at_utc"],
            "forecast_current_price": 1.10004,
            "forecast_spread_pips": 0.8,
            "filtered_input_sha256": "f" * 64,
            "raw_winner": "UP",
            "component_scores": scores,
            "final_direction": "UP",
            "raw_confidence": 0.6516,
            "calibration_multiplier": 0.64,
            "calibration_scope": "M15_RECOVERY_CONSERVATIVE_DIRECTIONAL_PRIOR",
            "confidence": 0.417,
            "target_price": 1.102,
            "invalidation_price": 1.0995,
            "horizon_min": 60,
            "geometry_source_timeframe": "M15",
            "live_permission": False,
        }
        evidence = {
            **evidence_body,
            "evidence_sha256": canonical_sha256(evidence_body),
        }
        binding = build_forecast_binding(evidence, cycle_id=cycle_id)
        assert isinstance(binding, dict)
        return {
            "desk": "failure_trader",
            "pair": "EUR_USD",
            "direction": "LONG",
            "method": "BREAKOUT_FAILURE",
            "adoption": "ORDER_INTENT_REQUIRED",
            "campaign_role": "NOW",
            "reason": "M15 recovery proof candidate",
            "required_receipt": "M15 close-confirmed continuation",
            "target_reward_risk": 2.0,
            "blockers": [],
            "story_examples": [],
            "forecast_cycle_id": cycle_id,
            "forecast_direction": "UP",
            "forecast_confidence": 0.417,
            "forecast_raw_confidence": 0.6516,
            "forecast_calibration_multiplier": 0.64,
            "forecast_current_price": 1.10004,
            "forecast_target_price": 1.102,
            "forecast_invalidation_price": 1.0995,
            "forecast_horizon_min": 60,
            "forecast_component_scores": scores,
            "forecast_m15_recovery_receipt": receipt,
            "forecast_m15_recovery_mode": "M15_RECOVERY_MICRO",
            "forecast_m15_recovery_live_permission": False,
            "forecast_m15_recovery_evidence": evidence,
            "forecast_m15_recovery_binding": binding,
            "forecast_m15_recovery_binding_sha256": binding["binding_sha256"],
        }

    @staticmethod
    def _snapshot(now: datetime, *, spread_pips: float = 0.8) -> BrokerSnapshot:
        bid = 1.10000
        ask = bid + spread_pips / 10_000.0
        return BrokerSnapshot(
            fetched_at_utc=now,
            quotes={
                "EUR_USD": Quote(
                    "EUR_USD",
                    bid=bid,
                    ask=ask,
                    timestamp_utc=now,
                )
            },
            account=AccountSummary(
                nav_jpy=200_000.0,
                balance_jpy=200_000.0,
                margin_available_jpy=200_000.0,
                fetched_at_utc=now,
            ),
            home_conversions={"USD": 156.0},
        )

    @staticmethod
    def _indexed_chart(chart: dict[str, object], m15_view: dict[str, object]):
        views = chart["views"]
        assert isinstance(views, list)
        m1_view = next(
            view
            for view in views
            if isinstance(view, dict) and view.get("granularity") == "M1"
        )
        m5_view = next(
            view
            for view in views
            if isinstance(view, dict) and view.get("granularity") == "M5"
        )
        return {
            "EUR_USD": {
                "__raw_chart": chart,
                "generated_at_utc": chart["generated_at_utc"],
                "dominant_regime": "TREND_UP",
                "M1": m1_view["indicators"],
                "M5": m5_view["indicators"],
                "M15": m15_view["indicators"],
                "M15__regime": "TREND_UP",
                "M15__regime_reading": m15_view["regime_reading"],
                "session": {},
                "confluence": {},
            }
        }

    def test_recovery_cap_is_999_and_does_not_raise_five_units(self) -> None:
        now, chart, receipt, m15_view = self._evidence()
        payload = _forecast_context_payload(
            SimpleNamespace(
                pair="EUR_USD",
                direction="UP",
                confidence=0.8,
                current_price=1.1,
                m15_recovery_receipt=receipt,
                technical_context_v1={},
            )
        )
        self.assertEqual(payload["forecast_m15_recovery_receipt"], receipt)
        self.assertIs(payload["forecast_m15_recovery_live_permission"], False)
        lane = self._lane(receipt)
        snapshot = self._snapshot(now)
        quote = snapshot.quotes["EUR_USD"]
        for budgeted_units, expected_units in ((1500, 999), (5, 5)):
            with self.subTest(budgeted_units=budgeted_units):
                with patch(
                    "quant_rabbit.strategy.intent_generator._risk_budgeted_units",
                    return_value=budgeted_units,
                ):
                    intent = _intent_from_lane(
                        lane,
                        quote,
                        snapshot,
                        max_loss_jpy=500.0,
                        # A conflicting caller value must never replace the
                        # receipt-bound M15 ATR on the recovery path.
                        atr_pips=999.0,
                        range_indicators=m15_view["indicators"],
                        pair_chart=chart,
                        validation_time_utc=now,
                        m15_recovery_receipt=receipt,
                    )
                self.assertEqual(intent.units, expected_units)
                self.assertEqual(
                    intent.metadata["m15_recovery_micro_pre_cap_units"],
                    budgeted_units,
                )
                self.assertEqual(
                    intent.metadata["m15_recovery_micro_max_units"],
                    M15_RECOVERY_MICRO_MAX_UNITS,
                )
                self.assertIs(
                    intent.metadata["m15_recovery_micro_full_size_allowed"],
                    False,
                )
                self.assertEqual(
                    intent.metadata["geometry_atr_source_timeframe"], "M15"
                )
                self.assertEqual(intent.metadata["geometry_atr_pips"], 5.0)
                self.assertEqual(intent.metadata["forecast_confidence"], 0.417)
                self.assertEqual(
                    (intent.entry, intent.tp, intent.sl),
                    (1.10009, 1.102, 1.0995),
                )
                self.assertEqual(
                    intent.metadata["geometry_model"],
                    "QR_M15_RECOVERY_GEOMETRY_V1",
                )
                self.assertEqual(
                    intent.metadata["tp_target_source"],
                    "M15_RECOVERY_FORECAST_BOUND",
                )
                self.assertNotIn(
                    "technical_harvest_precision_applied", intent.metadata
                )
                self.assertNotIn("bidask_replay_precision_applied", intent.metadata)
                lane_binding = intent.metadata["m15_recovery_lane_binding"]
                self.assertEqual(
                    lane_binding["geometry_binding"]["atr_pips"], 5.0
                )
                self.assertEqual(
                    lane_binding["geometry_binding"]["forecast_target_price"],
                    1.102,
                )
                self.assertEqual(
                    lane_binding["geometry_binding"][
                        "forecast_invalidation_price"
                    ],
                    1.0995,
                )
                self.assertIs(
                    intent.metadata[
                        "m15_recovery_micro_manual_position_mutation_allowed"
                    ],
                    False,
                )

    def test_recovery_stop_uses_forecast_invalidation_not_global_atr_floor(self) -> None:
        now, chart, receipt, m15_view = self._evidence()
        lane = self._lane(receipt)
        snapshot = self._snapshot(now)
        quote = snapshot.quotes["EUR_USD"]

        with (
            patch(
                "quant_rabbit.strategy.intent_generator.GEOMETRY_ATR_MULT",
                5.0,
            ),
            patch(
                "quant_rabbit.strategy.intent_generator._risk_budgeted_units",
                return_value=5,
            ),
        ):
            intent = _intent_from_lane(
                lane,
                quote,
                snapshot,
                max_loss_jpy=500.0,
                atr_pips=999.0,
                range_indicators=m15_view["indicators"],
                pair_chart=chart,
                validation_time_utc=now,
                m15_recovery_receipt=receipt,
            )

        self.assertEqual(intent.sl, 1.0995)
        self.assertEqual(
            intent.sl,
            intent.metadata["geometry_forecast_invalidation_price"],
        )
        self.assertGreater(
            (intent.entry - intent.sl) * 10_000,
            0.8 * 5.0,
        )

    def test_recovery_variant_generation_is_stop_entry_only(self) -> None:
        from quant_rabbit.strategy.intent_generator import _order_variants_for

        _now, _chart, receipt, _m15_view = self._evidence()
        recovery_lane = self._lane(receipt)
        self.assertEqual(
            _order_variants_for(recovery_lane),
            (OrderType.STOP_ENTRY,),
        )

        ordinary_failed_break = {
            "desk": "failure_trader",
            "pair": "EUR_USD",
            "direction": "LONG",
            "method": "BREAKOUT_FAILURE",
        }
        self.assertEqual(
            _order_variants_for(ordinary_failed_break),
            (OrderType.LIMIT, OrderType.STOP_ENTRY, OrderType.MARKET),
        )
        ordinary_range = {
            **ordinary_failed_break,
            "desk": "range_trader",
            "method": "RANGE_ROTATION",
        }
        self.assertEqual(
            _order_variants_for(ordinary_range),
            (OrderType.LIMIT, OrderType.MARKET),
        )

    def test_bound_exact_tp_recovery_uses_lane_proof_not_global_confidence_floor(self) -> None:
        from tests.test_risk_engine import _m15_recovery_fixture

        with tempfile.TemporaryDirectory() as tmp:
            _now, _chart_path, intent, _broker = _m15_recovery_fixture(
                Path(tmp)
            )
            self.assertLess(intent.metadata["forecast_confidence"], 0.55)
            with patch.dict(
                os.environ,
                {"QR_REQUIRE_FORECAST_FOR_LIVE": "1"},
                clear=False,
            ):
                self.assertIsNone(
                    _forecast_live_readiness_issue(
                        intent,
                        intent.metadata,
                        TradeMethod.BREAKOUT_FAILURE,
                    )
                )
                watch_metadata = dict(intent.metadata)
                watch_metadata["forecast_watch_only"] = True
                watch_intent = replace(intent, metadata=watch_metadata)
                self.assertIsNone(
                    _forecast_watch_only_issue(watch_intent, watch_metadata)
                )

                tampered_metadata = json.loads(json.dumps(intent.metadata))
                tampered_metadata["forecast_confidence"] = 0.1
                tampered_intent = replace(intent, metadata=tampered_metadata)
                issue = _forecast_live_readiness_issue(
                    tampered_intent,
                    tampered_metadata,
                    TradeMethod.BREAKOUT_FAILURE,
                )
                self.assertIsNotNone(issue)
                assert issue is not None
                self.assertEqual(
                    issue["code"],
                    "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
                )

    def test_recovery_context_removes_all_m1_m5_and_aggregate_inputs(self) -> None:
        now, chart, _receipt, m15_view = self._evidence()
        indexed = self._indexed_chart(chart, m15_view)
        source = indexed["EUR_USD"]
        source.update(
            {
                "long_score": 999.0,
                "short_score": -999.0,
                "chart_story": "M1/M5 aggregate must not leak",
                "confluence": {
                    "score_balance": "LONG_LEAN",
                    "score_gap": 999.0,
                },
                "M1__family_scores": {"trend_score": 999.0},
                "M5__family_scores": {"trend_score": 999.0},
            }
        )

        context, recovery_charts = _m15_recovery_chart_context_for(
            "EUR_USD",
            indexed,
            current_price=1.1,
        )

        self.assertEqual(
            context["m15_recovery_context_timeframes"],
            ["M15", "M30", "H1", "H4", "D"],
        )
        self.assertFalse(
            any(
                key.lower().startswith(("m1_", "m5_", "oanda_m5_"))
                for key in context
            )
        )
        for forbidden in (
            "chart_long_score",
            "chart_short_score",
            "chart_direction_bias",
            "chart_score_balance",
            "chart_score_gap",
            "confluence",
            "chart_story_structural",
            "tf_agreement_score",
        ):
            self.assertNotIn(forbidden, context)
        assert recovery_charts is not None
        recovery_source = recovery_charts["EUR_USD"]
        self.assertNotIn("M1", recovery_source)
        self.assertNotIn("M5", recovery_source)
        raw = recovery_source["__raw_chart"]
        self.assertEqual(
            {
                view["granularity"]
                for view in raw["views"]
                if isinstance(view, dict)
            },
            {"M15"},
        )

    def test_recovery_market_support_cannot_reintroduce_m5_projection(self) -> None:
        import quant_rabbit.strategy.intent_generator as intent_generator_module

        from quant_rabbit.strategy.intent_generator import (
            _forecast_seed_for_pair,
            _load_pair_charts,
            _snapshot_from_json,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            (root / "logs").mkdir()
            charts = _load_pair_charts(
                _pair_charts_with_direction(
                    root,
                    long_score=0.3,
                    short_score=0.7,
                    dominant_regime="TREND_DOWN",
                    m5_regime="TREND_DOWN",
                )
            )
            snapshot = _snapshot_from_json(
                json.loads(_snapshot(root).read_text())
            )
            m5_signal = SimpleNamespace(
                name="m5_only_signal",
                timeframe="M5",
                direction="DOWN",
                confidence=0.99,
                bonus_magnitude=99.0,
                rationale="M5 must not leak",
            )
            m15_signal = SimpleNamespace(
                name="m15_bound_signal",
                timeframe="M15",
                direction="DOWN",
                confidence=0.8,
                bonus_magnitude=10.0,
                rationale="M15 is allowed",
            )
            forecast = SimpleNamespace(
                pair="EUR_USD",
                direction="DOWN",
                confidence=0.485,
                raw_confidence=0.758,
                current_price=1.17326,
                target_price=1.17,
                invalidation_price=1.175,
                horizon_min=60,
                rationale_summary="M15 recovery DOWN",
                drivers_for=(),
                drivers_against=(),
                component_scores={"UP": 8.0, "DOWN": 26.0, "RANGE": 5.0},
                m15_recovery_receipt={
                    "contract": "QR_M15_RECOVERY_MICRO_V1",
                    "mode": "M15_RECOVERY_MICRO",
                },
            )
            support_regimes: list[object] = []
            original_support = (
                intent_generator_module._forecast_market_support_for_forecast
            )

            def capture_support_regime(**kwargs):
                support_regimes.append(kwargs.get("regime"))
                return original_support(**kwargs)

            with patch(
                "quant_rabbit.strategy.intent_generator._forecast_seed_has_rich_chart_context",
                return_value=True,
            ), patch(
                "quant_rabbit.strategy.pattern_signals.detect_pattern_signals",
                return_value=[],
            ), patch(
                "quant_rabbit.strategy.forward_projection.detect_forward_projections",
                return_value=[m5_signal, m15_signal],
            ), patch(
                "quant_rabbit.strategy.correlation_predictor.detect_correlation_lag",
                return_value=[],
            ), patch(
                "quant_rabbit.strategy.path_projection.detect_paths",
                return_value=[],
            ), patch(
                "quant_rabbit.strategy.reversal_signal.detect_reversal",
                return_value=None,
            ), patch(
                "quant_rabbit.strategy.directional_forecaster.synthesize_forecast",
                return_value=forecast,
            ), patch(
                "quant_rabbit.strategy.intent_generator._forecast_market_support_for_forecast",
                side_effect=capture_support_regime,
            ):
                seed = _forecast_seed_for_pair(
                    "EUR_USD",
                    charts or {},
                    snapshot,
                    data_root=data_root,
                    hit_rates=None,
                )

        self.assertIsNotNone(seed)
        assert seed is not None
        self.assertEqual(seed.projection_signals, (m15_signal,))
        support_names = {
            item["name"]
            for item in seed.market_support.get("signals", [])
            if isinstance(item, dict)
        }
        self.assertNotIn("m5_only_signal", support_names)
        self.assertEqual(support_regimes, [None])

    def test_build_revalidates_m15_atr_and_rejects_tamper_or_current_spread(self) -> None:
        now, chart, receipt, m15_view = self._evidence()
        indexed = self._indexed_chart(chart, m15_view)
        clean_risk = SimpleNamespace(allowed=True, issues=(), metrics=None)
        dummy_intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.STOP_ENTRY,
            units=5,
            entry=1.1002,
            tp=1.1012,
            sl=1.0997,
            thesis="recovery test",
            owner=Owner.TRADER,
            market_context=MarketContext(
                regime="TREND_UP",
                narrative="M15 recovery",
                chart_story="M15 only",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="M15 structure",
            ),
            metadata={},
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generator = IntentGenerator(
                campaign_plan=root / "campaign.json",
                strategy_profile=root / "strategy.json",
                output_path=root / "intents.json",
                report_path=root / "intents.md",
                data_root=root,
                max_loss_jpy=500.0,
            )
            cases = []
            cases.append(("valid", receipt, 0.8, True))
            tampered = json.loads(json.dumps(receipt))
            tampered["sizing"]["max_units"] = 1000
            cases.append(("tampered", tampered, 0.8, False))
            cases.append(("wide_current_spread", receipt, 99.0, False))
            for name, candidate, spread, expected_valid in cases:
                with self.subTest(name=name):
                    lane = self._lane(candidate)
                    snapshot = self._snapshot(now, spread_pips=spread)
                    with (
                        patch(
                            "quant_rabbit.strategy.intent_generator._intent_from_lane",
                            return_value=replace(dummy_intent, metadata={}),
                        ) as build_intent,
                        patch(
                            "quant_rabbit.strategy.intent_generator.RiskEngine.validate",
                            return_value=clean_risk,
                        ),
                    ):
                        result = generator._build_for_lane(
                            lane,
                            snapshot,
                            None,
                            max_loss_jpy=500.0,
                            pair_charts=indexed,
                            validation_time_utc=now,
                            data_root=root,
                        )
                    kwargs = build_intent.call_args.kwargs
                    issue_codes = {issue["code"] for issue in result.risk_issues}
                    if expected_valid:
                        self.assertEqual(kwargs["atr_pips"], 5.0)
                        self.assertEqual(kwargs["m15_recovery_receipt"], receipt)
                        self.assertNotIn("MISSING_ATR_DATA", issue_codes)
                        self.assertNotIn("M15_RECOVERY_RECEIPT_INVALID", issue_codes)
                    else:
                        self.assertIsNone(kwargs["atr_pips"])
                        self.assertIsNone(kwargs["m15_recovery_receipt"])
                        self.assertIn("MISSING_ATR_DATA", issue_codes)
                        self.assertIn("M15_RECOVERY_RECEIPT_INVALID", issue_codes)

    def test_only_stop_entry_exact_tp_proof_micro_shape_is_candidate(self) -> None:
        _now, _chart, receipt, _m15_view = self._evidence()
        metadata = _exact_vehicle_rotation_metadata(trades=7)
        metadata.update(
            {
                "forecast_direction": "UP",
                "m15_recovery_micro_receipt": receipt,
                "m15_recovery_micro_live_permission": False,
                "capture_take_profit_scope_key": (
                    "EUR_USD|LONG|BREAKOUT_FAILURE|STOP|TAKE_PROFIT_ORDER"
                ),
                "capture_take_profit_vehicle": "STOP",
            }
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.STOP_ENTRY,
            units=5,
            entry=1.1002,
            tp=1.1012,
            sl=1.0995,
            thesis="exact TP proof recovery",
            owner=Owner.TRADER,
            market_context=MarketContext(
                regime="TREND_UP",
                narrative="exact TP proof collection",
                chart_story="M15 recovery",
                method=TradeMethod.BREAKOUT_FAILURE,
                invalidation="failed-break structure",
            ),
            metadata=metadata,
        )
        capture_issue = _capture_positive_rotation_live_issue(intent)
        self.assertEqual(
            capture_issue["code"], POSITIVE_ROTATION_PROOF_COLLECTION_WARN_CODE
        )
        issue = _m15_recovery_micro_live_issue(intent)
        self.assertEqual(issue["code"], "M15_RECOVERY_LIVE_REVALIDATION_REQUIRED")
        self.assertIs(intent.metadata["m15_recovery_micro_shape_eligible"], True)
        self.assertIs(intent.metadata["m15_recovery_micro_live_permission"], False)

        limit_rejected = replace(
            intent,
            order_type=OrderType.LIMIT,
            metadata=dict(intent.metadata),
        )
        limit_issue = _m15_recovery_micro_live_issue(limit_rejected)
        self.assertEqual(
            limit_issue["code"],
            "M15_RECOVERY_NON_TP_HARVEST_BLOCKED",
        )
        self.assertIs(
            limit_rejected.metadata["m15_recovery_micro_shape_eligible"],
            False,
        )

        revalidated_issue = _m15_recovery_micro_live_issue(
            intent,
            SimpleNamespace(
                issues=(
                    RiskIssue(
                        "M15_RECOVERY_RISK_REVALIDATED",
                        "current canonical source passed",
                        "WARN",
                    ),
                )
            ),
        )
        self.assertEqual(
            revalidated_issue,
            {
                "code": "M15_RECOVERY_RISK_REVALIDATED",
                "message": (
                    "M15 recovery micro candidate passed independent RiskEngine "
                    "source/spread/time/shape validation; the final gateway must "
                    "still re-read and match the same receipt before POST"
                ),
                "severity": "WARN",
            },
        )
        self.assertIs(
            intent.metadata["m15_recovery_micro_risk_revalidated"], True
        )
        self.assertIs(intent.metadata["m15_recovery_micro_live_permission"], False)

        for name, rejected in (
            (
                "non_tp",
                replace(intent, metadata={**intent.metadata, "positive_rotation_mode": None}),
            ),
            ("full_size", replace(intent, units=1000, metadata=dict(intent.metadata))),
        ):
            with self.subTest(name=name):
                rejected_issue = _m15_recovery_micro_live_issue(rejected)
                self.assertEqual(
                    rejected_issue["code"],
                    "M15_RECOVERY_NON_TP_HARVEST_BLOCKED",
                )
                self.assertIs(
                    rejected.metadata["m15_recovery_micro_shape_eligible"], False
                )


class SameDayLossStreakIssueTest(unittest.TestCase):
    """_same_day_loss_streak_issues — §8 re-entry discipline (2026-06-10)."""

    @staticmethod
    def _intent(order_type: OrderType, metadata: dict | None = None) -> OrderIntent:
        return OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=order_type,
            units=5000,
            tp=1.1600,
            sl=1.1400,
            thesis="test",
            entry=1.1500,
            metadata=metadata or {},
        )

    @staticmethod
    def _streak(count: int) -> SameDayLossStreak:
        return SameDayLossStreak(
            pair="EUR_USD",
            consecutive_losses=count,
            net_loss_jpy=-2000.0 * count,
            last_loss_ts_utc="2026-06-04T14:44:00Z",
        )

    def _issues(self, order_type: OrderType, count: int, metadata: dict | None = None):
        intent = self._intent(order_type, metadata)
        issues = _same_day_loss_streak_issues(
            intent,
            self._streak(count) if count else None,
            base_max_loss_jpy=2000.0,
            effective_max_loss_jpy=2000.0 * (0.5**count),
        )
        return intent, issues

    def test_no_streak_emits_nothing(self) -> None:
        _, issues = self._issues(OrderType.MARKET, 0)
        self.assertEqual(issues, [])

    def test_streak_at_threshold_blocks_market_chase(self) -> None:
        intent, issues = self._issues(OrderType.MARKET, 2)
        self.assertEqual([i["code"] for i in issues], ["SAME_DAY_LOSS_STREAK_CHASE"])
        self.assertEqual(issues[0]["severity"], "BLOCK")
        self.assertEqual(intent.metadata["same_day_loss_streak"], 2)
        self.assertAlmostEqual(intent.metadata["loss_streak_max_loss_scale"], 0.25)

    def test_streak_at_threshold_blocks_stop_entry_chase(self) -> None:
        _, issues = self._issues(OrderType.STOP_ENTRY, 2)
        self.assertEqual(issues[0]["severity"], "BLOCK")

    def test_streak_at_threshold_keeps_limit_retest_tradable(self) -> None:
        _, issues = self._issues(OrderType.LIMIT, 2)
        self.assertEqual([i["code"] for i in issues], ["SAME_DAY_LOSS_STREAK"])
        self.assertEqual(issues[0]["severity"], "WARN")

    def test_single_loss_warns_only(self) -> None:
        _, issues = self._issues(OrderType.MARKET, 1)
        self.assertEqual(issues[0]["severity"], "WARN")

    def test_recovery_hedge_is_never_hard_blocked(self) -> None:
        _, issues = self._issues(OrderType.MARKET, 3, metadata={"position_intent": "HEDGE"})
        self.assertEqual(issues[0]["severity"], "WARN")

    def test_env_zero_threshold_disables_gate(self) -> None:
        with patch("quant_rabbit.strategy.intent_generator.LOSS_STREAK_BLOCK_THRESHOLD", 0):
            _, issues = self._issues(OrderType.MARKET, 5)
        self.assertEqual(issues, [])


if __name__ == "__main__":
    unittest.main()
