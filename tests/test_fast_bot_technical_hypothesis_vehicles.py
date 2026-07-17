from __future__ import annotations

import hashlib
import inspect
import json
import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any

import quant_rabbit.fast_bot_technical_hypothesis_vehicles as vehicle_module
from quant_rabbit.fast_bot_learning import build_fast_bot_learning_shadow
from quant_rabbit.fast_bot_technical_hypotheses import (
    build_fast_bot_technical_hypotheses,
)
from quant_rabbit.fast_bot_technical_hypothesis_vehicles import (
    CATALOG_CONTRACT_V2,
    NATIVE_MARKET_UNIT_COHORT_POLICY_V2,
    TECHNICAL_HYPOTHESIS_VEHICLE_CATALOG_V2,
    build_fast_bot_technical_hypothesis_vehicles_v2,
    technical_hypothesis_vehicle_catalog_v2,
    technical_hypothesis_vehicle_shadow_v2_valid,
)


PAIR = "EUR_JPY"
CYCLE = datetime(2026, 7, 17, 7, 35, 10, tzinfo=timezone.utc)
CANDLE_CLOSE = CYCLE.replace(second=0, microsecond=0)
TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D")
METHODS = ("BREAKOUT_FAILURE", "RANGE_ROTATION", "TREND_CONTINUATION")
SIDES = ("LONG", "SHORT")


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "contract_sha256": _sha(value)}


def _seal_feature_row(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "feature_sha256": _sha(value)}


def _reseal_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for row in snapshot["timeframes"]:
        body = {key: item for key, item in row.items() if key != "feature_sha256"}
        rows.append(_seal_feature_row(body))
    body = {key: item for key, item in snapshot.items() if key != "contract_sha256"}
    body["timeframes"] = rows
    return _seal(body)


def _reseal_vehicle_shadow(shadow: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for row in shadow["vehicles"]:
        body = {key: item for key, item in row.items() if key != "vehicle_sha256"}
        rows.append({**body, "vehicle_sha256": _sha(body)})
    body = {key: item for key, item in shadow.items() if key != "contract_sha256"}
    body["vehicles"] = rows
    return _seal(body)


def _frame(snapshot: dict[str, Any], timeframe: str) -> dict[str, Any]:
    return next(row for row in snapshot["timeframes"] if row["timeframe"] == timeframe)


def _feature_snapshot(*, mode: str = "trend") -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for timeframe in TIMEFRAMES:
        if mode == "range":
            market = {
                "direction": "UP",
                "phase": "RANGE",
                "readiness": "TRIGGERED",
                "location": "LOWER_THIRD",
                "value_zone": "DEEP_DISCOUNT",
                "extension": "OVERSOLD",
                "evidence_complete": True,
            }
            indicators = {
                "close": 150.005,
                "rsi_14": 32.0,
                "adx_14": 18.0,
                "choppiness_14": 62.0,
                "hurst_100": 0.4,
                "z_score_20": -1.4,
                "bb_position": 0.1,
                "donchian_position": 0.1,
                "donchian_high": 150.012,
                "donchian_low": 149.980,
                "bb_middle": 150.000,
                "atr_pips": 5.0,
            }
            rsi_series = [45.0, 36.0, 28.0, 29.0, 32.0]
            atr_series = [5.4, 5.3, 5.2, 5.1, 5.0]
        elif mode == "quiet":
            market = {
                "direction": "EITHER",
                "phase": "PRE_RANGE",
                "readiness": "FORMING",
                "location": "MIDDLE_THIRD",
                "value_zone": "FAIR_VALUE",
                "extension": "BALANCED",
                "evidence_complete": True,
            }
            indicators = {
                "close": 150.005,
                "atr_pips": 5.0,
            }
            rsi_series = [50.0, 50.0]
            atr_series = [5.0, 5.0]
        else:
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
                "close": 150.005,
                "ema_20": 149.990,
                "ema_50": 149.980,
                "ema_slope_20": 0.01,
                "plus_di_14": 31.0,
                "minus_di_14": 14.0,
                "supertrend_dir": 1,
                "rsi_14": 63.0,
                "macd_hist": 0.2,
                "roc_5": 0.4,
                "adx_14": 28.0,
                "atr_pips": 5.0,
                "z_score_20": 0.0,
                "donchian_high": 150.012,
                "donchian_low": 149.980,
                "bb_middle": 149.950,
            }
            rsi_series = [47.0, 49.0, 51.0, 56.0, 63.0]
            atr_series = [3.0, 3.4, 3.8, 4.4, 5.0]
        rows.append(
            _seal_feature_row(
                {
                    "timeframe": timeframe,
                    "complete_candle_close_utc": CANDLE_CLOSE.isoformat(),
                    "market_state": market,
                    "indicators": indicators,
                    "indicator_series": {
                        "rsi_14": rsi_series,
                        "macd_hist": [-0.2, -0.1, 0.0, 0.1, 0.2],
                        "adx_14": [20.0, 22.0, 24.0, 26.0, 28.0],
                        "atr_pips": atr_series,
                        "ema_12_minus_50_pips": [-0.2, -0.1, 0.1, 0.5, 1.0],
                    },
                }
            )
        )
    snapshot = _seal(
        {
            "contract": "QR_FAST_BOT_EPISODE_TECHNICAL_FEATURE_SNAPSHOT_V1",
            "schema_version": 1,
            "pair": PAIR,
            "handoff_cycle_generated_at_utc": CYCLE.isoformat(),
            "feature_allowlist_version": 1,
            "timeframes": rows,
            "hypothesis_families": [
                "TREND",
                "PULLBACK",
                "BREAKOUT",
                "BREAKOUT_FAILURE",
                "RANGE",
                "EXHAUSTION",
            ],
            "raw_chart_packet_embedded": False,
            "diagnostic_only": True,
            "order_authority": "NONE",
            "live_permission": False,
            "broker_mutation_allowed": False,
        }
    )
    if mode in {"trend", "trend_down"}:
        _frame(snapshot, "M5")["indicators"]["donchian_high"] = 150.020
        _frame(snapshot, "M5")["indicators"]["donchian_low"] = 149.970
        _frame(snapshot, "M15")["indicators"]["atr_pips"] = 8.0
        _frame(snapshot, "M15")["indicators"]["bb_squeeze"] = 1
        if mode == "trend_down":
            for timeframe in TIMEFRAMES:
                row = _frame(snapshot, timeframe)
                row["market_state"]["direction"] = "DOWN"
                row["indicators"].update(
                    {
                        "ema_20": 150.020,
                        "ema_50": 150.030,
                        "ema_slope_20": -0.01,
                        "plus_di_14": 14.0,
                        "minus_di_14": 31.0,
                        "supertrend_dir": -1,
                        "rsi_14": 40.0,
                        "macd_hist": -0.2,
                        "roc_5": -0.4,
                    }
                )
                row["indicator_series"]["rsi_14"] = [53.0, 51.0, 49.0, 46.0, 40.0]
                row["indicator_series"]["macd_hist"] = [0.2, 0.1, 0.0, -0.1, -0.2]
                row["indicator_series"]["ema_12_minus_50_pips"] = [
                    0.2,
                    0.1,
                    -0.1,
                    -0.5,
                    -1.0,
                ]
        snapshot = _reseal_snapshot(snapshot)
    elif mode == "exhaustion":
        for timeframe in ("M5", "M15", "H1"):
            _frame(snapshot, timeframe)["market_state"]["extension"] = "OVERBOUGHT"
        m5 = _frame(snapshot, "M5")
        m5["indicators"]["rsi_14"] = 75.0
        m5["indicator_series"]["adx_14"] = [36.0, 34.0, 32.0, 30.0, 28.0]
        m5["indicator_series"]["macd_hist"] = [0.6, 0.5, 0.4, 0.3, 0.2]
        m5["indicator_series"]["atr_pips"] = [8.0, 7.0, 6.0, 5.5, 5.0]
        m1 = _frame(snapshot, "M1")
        m1["market_state"]["direction"] = "DOWN"
        m1["market_state"]["readiness"] = "TRIGGERED"
        m1["indicators"]["donchian_high"] = 150.050
        m1["indicators"]["donchian_low"] = 149.970
        m1["indicators"]["bb_middle"] = 149.950
        _frame(snapshot, "M15")["indicators"]["atr_pips"] = 8.0
        snapshot = _reseal_snapshot(snapshot)
    return snapshot


def _spread_pips(bid: float, ask: float) -> float:
    return round((ask - bid) * 100, 6)


def _regime_row(
    side: str,
    method: str,
    *,
    spread_pips: float,
    m5_atr_pips: float,
) -> dict[str, Any]:
    return {
        "pair": PAIR,
        "side": side,
        "method": method,
        "state": "GO",
        "score": 1.0,
        "execution_enabled": True,
        "hard_blockers": [],
        "caution_reasons": [],
        "m1_closed_candle_utc": CANDLE_CLOSE.isoformat(),
        "m5_atr_pips": m5_atr_pips,
        "spread_pips": spread_pips,
        "spread_to_m5_atr": round(spread_pips / m5_atr_pips, 6),
        "failed_break_direction": "NONE",
        "ai_supervision": {"mode": "UNSUPERVISED", "reason": "TEST"},
        "timeframe_votes": {
            timeframe: {
                "evidence_complete": True,
                "direction_score": 1 if side == "LONG" else -1,
                "observed_direction": "UP" if side == "LONG" else "DOWN",
                "phase": "TREND",
                "readiness": "ACTIVE",
                "location": "MIDDLE_THIRD",
                "structure": "BREAKOUT_ACTIVE",
                "trigger": "BREAKOUT_CLOSE",
                "extension": "BALANCED",
                "value_zone": "EQUILIBRIUM",
            }
            for timeframe in TIMEFRAMES
        },
    }


def _learning_seat(
    feature_snapshot: dict[str, Any],
    *,
    bid: float = 150.000,
    ask: float = 150.008,
    input_blocked: bool = False,
) -> dict[str, Any]:
    m5_atr = float(_frame(feature_snapshot, "M5")["indicators"]["atr_pips"])
    snapshot = {
        "fetched_at_utc": CYCLE.isoformat(),
        "quotes": {
            PAIR: {
                "bid": bid,
                "ask": ask,
                "timestamp_utc": CYCLE.isoformat(),
            }
        },
        "positions": [],
        "orders": [],
    }
    spread = _spread_pips(bid, ask)
    rows = [
        _regime_row(
            side,
            method,
            spread_pips=spread,
            m5_atr_pips=m5_atr,
        )
        for method in METHODS
        for side in SIDES
    ]
    if input_blocked:
        for row in rows:
            row["state"] = "STOP"
            row["execution_enabled"] = False
            row["hard_blockers"] = [
                "FAST_CHART_PACKET_STALE",
                "FAST_TIMEFRAME_EVIDENCE_MISSING:M1,M5",
            ]
            row["timeframe_votes"]["M1"]["evidence_complete"] = False
            row["timeframe_votes"]["M5"]["evidence_complete"] = False
    regime = _seal(
        {
            "contract": "QR_HIERARCHICAL_BOT_REGIME_V1",
            "schema_version": 1,
            "generated_at_utc": CYCLE.isoformat(),
            "rows": rows,
            "sources": {"broker_snapshot_sha256": _sha(snapshot)},
        }
    )
    shadow = build_fast_bot_learning_shadow(regime, snapshot, now_utc=CYCLE)
    if shadow["seat_count"] != 1:
        raise AssertionError(shadow)
    return shadow["seats"][0]


def _anchor_and_route(
    branch: str, *, attempt_direction: str = "UP"
) -> tuple[dict[str, Any], dict[str, Any]]:
    rail = (
        {
            "upper": 150.010,
            "lower": 149.910,
            "width": 0.100,
            "buffer": 0.005,
            "buffer_ratio": 0.05,
        }
        if attempt_direction == "UP"
        else {
            "upper": 150.090,
            "lower": 149.990,
            "width": 0.100,
            "buffer": 0.005,
            "buffer_ratio": 0.05,
        }
    )
    anchor = {
        "episode_kind": "M5_RANGE_BREAK_ATTEMPT",
        "episode_rule_version": "M5_RANGE_BREAK_ATTEMPT_V1",
        "horizon_lane": "M1_EXECUTION_15M_HOLD",
        "confirmation_ttl_seconds": 15 * 60,
        "setup_candle_utc": (CANDLE_CLOSE - timedelta(minutes=15)).isoformat(),
        "attempt_candle_utc": (CANDLE_CLOSE - timedelta(minutes=10)).isoformat(),
        "attempt_direction": attempt_direction,
        "rail": rail,
        "source_evidence_sha256": "a" * 64,
    }
    if branch == "ACCEPTED":
        accepted_side = "LONG" if attempt_direction == "UP" else "SHORT"
        route = {
            "branch_outcome": "ACCEPTED",
            "trade_side": accepted_side,
            "candidate_methods": ["TREND_CONTINUATION"],
            "route_family": "BREAKOUT_CONTINUATION",
            "branch_candle_utc": (CANDLE_CLOSE - timedelta(minutes=5)).isoformat(),
            "branch_close": 150.015,
            "selection_status": "UNSELECTED_COUNTERFACTUAL_PATHS_PRESERVED",
        }
    else:
        rejected_side = "SHORT" if attempt_direction == "UP" else "LONG"
        route = {
            "branch_outcome": "REJECTED",
            "trade_side": rejected_side,
            "candidate_methods": ["BREAKOUT_FAILURE", "RANGE_ROTATION"],
            "route_family": "RANGE_RECLAIM_OR_BREAKOUT_FAILURE",
            "branch_candle_utc": (CANDLE_CLOSE - timedelta(minutes=5)).isoformat(),
            "branch_close": 150.000,
            "selection_status": "UNSELECTED_COUNTERFACTUAL_PATHS_PRESERVED",
        }
    return anchor, route


def _sources(
    *,
    mode: str,
    branch: str,
    bid: float = 150.000,
    ask: float = 150.008,
    attempt_direction: str = "UP",
    input_blocked: bool = False,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    feature_snapshot = _feature_snapshot(mode=mode)
    seat = _learning_seat(
        feature_snapshot,
        bid=bid,
        ask=ask,
        input_blocked=input_blocked,
    )
    anchor, route = _anchor_and_route(branch, attempt_direction=attempt_direction)
    technical_shadow = build_fast_bot_technical_hypotheses(
        feature_snapshot,
        attempt_direction=anchor["attempt_direction"],
        branch_outcome=route["branch_outcome"],
        route_family=route["route_family"],
        spread_pips=seat["executable_spread_pips"],
        m5_atr_pips=seat["m5_atr_pips"],
        spread_to_m5_atr=seat["spread_to_m5_atr"],
    )
    return feature_snapshot, technical_shadow, anchor, route, seat


def _build(
    sources: tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ],
) -> dict[str, Any]:
    feature_snapshot, technical_shadow, anchor, route, seat = sources
    return build_fast_bot_technical_hypothesis_vehicles_v2(
        technical_feature_snapshot=feature_snapshot,
        technical_hypothesis_shadow=technical_shadow,
        episode_anchor=anchor,
        episode_route=route,
        learning_seat=seat,
        confirmed_at_utc=CANDLE_CLOSE.isoformat(),
    )


def _vehicle(result: dict[str, Any], hypothesis_id: str) -> dict[str, Any]:
    return next(
        row for row in result["vehicles"] if row["hypothesis_id"] == hypothesis_id
    )


class FastBotTechnicalHypothesisVehiclesTest(unittest.TestCase):
    def test_catalog_is_sealed_complete_and_has_zero_authority(self) -> None:
        catalog = technical_hypothesis_vehicle_catalog_v2()

        self.assertEqual(catalog["contract"], CATALOG_CONTRACT_V2)
        self.assertEqual(catalog["schema_version"], 2)
        self.assertEqual(len(TECHNICAL_HYPOTHESIS_VEHICLE_CATALOG_V2), 8)
        self.assertEqual(
            [row["hypothesis_id"] for row in catalog["vehicles"]],
            [f"H{number:02d}" for number in range(1, 9)],
        )
        for row in catalog["vehicles"]:
            self.assertIn("entry_formula", row)
            self.assertIn("entry_ttl_policy", row)
            self.assertIn("take_profit_formula", row)
            self.assertIn("stop_loss_formula", row)
            self.assertIn("max_hold_policy", row)
            self.assertIn("gap_policy", row)
            self.assertIn("intrabar_policy", row)
            self.assertIn("applicability", row)
        self.assertEqual(catalog["order_authority"], "NONE")
        self.assertFalse(catalog["live_permission"])
        self.assertFalse(catalog["promotion_allowed"])
        self.assertFalse(catalog["broker_mutation_allowed"])
        self.assertEqual(
            catalog["cost_policy"],
            "OBSERVE_FROZEN_COST_COHORT_NEVER_EXCLUDE_BY_SPREAD_TO_TP",
        )
        self.assertEqual(
            catalog["native_market_unit_cohort_policy"],
            NATIVE_MARKET_UNIT_COHORT_POLICY_V2,
        )
        self.assertFalse(catalog["tuned_geometry_multipliers_allowed"])
        self.assertFalse(catalog["fixed_pip_geometry_allowed"])
        self.assertIn("used directly", catalog["native_market_unit_cohort_definition"])

    def test_trend_breakout_and_session_rows_freeze_exact_causal_geometry(self) -> None:
        sources = _sources(mode="trend", branch="ACCEPTED")
        result = _build(sources)

        self.assertEqual(result["status"], "EMITTED")
        expected = {
            "H01": (150.013, 150.093, 149.963),
            "H02": (150.009, 150.089, 149.959),
            "H04": (150.016, 150.116, 150.005),
            "H07": (150.021, 150.121, 150.005),
        }
        for hypothesis_id, prices in expected.items():
            row = _vehicle(result, hypothesis_id)
            self.assertEqual(row["status"], "EXACT_STOP_READY")
            execution = row["execution"]
            self.assertEqual(
                (
                    execution["entry_price"],
                    execution["take_profit_price"],
                    execution["stop_loss_price"],
                ),
                prices,
            )
            self.assertEqual(execution["order_type"], "STOP")
            self.assertEqual(execution["time_in_force"], "GTD")
            self.assertEqual(execution["natural_trigger_component"], "ASK")
            self.assertEqual(execution["exit_component"], "BID")
            self.assertIsNone(execution["price_bound"])
            self.assertEqual(execution["activation_at_utc"], CYCLE.isoformat())
            self.assertFalse(row["cost_observation"]["cost_is_applicability_gate"])
            self.assertEqual(
                execution["native_market_unit_cohort_policy"],
                NATIVE_MARKET_UNIT_COHORT_POLICY_V2,
            )
            self.assertIsNone(
                execution["geometry_distance_cohort"]["tuned_numeric_multiplier"]
            )
            self.assertIsNone(
                execution["geometry_distance_cohort"]["fixed_pip_distance"]
            )
        self.assertEqual(
            _vehicle(result, "H01")["execution"]["geometry_distance_cohort"][
                "take_profit_measure"
            ],
            "FROZEN_M15_ATR_PIPS",
        )
        self.assertEqual(
            _vehicle(result, "H04")["execution"]["geometry_distance_cohort"][
                "take_profit_measure"
            ],
            "EXACT_FROZEN_EPISODE_RAIL_WIDTH",
        )

        feature_snapshot, technical_shadow, anchor, route, seat = sources
        self.assertTrue(
            technical_hypothesis_vehicle_shadow_v2_valid(
                result,
                technical_feature_snapshot=feature_snapshot,
                technical_hypothesis_shadow=technical_shadow,
                episode_anchor=anchor,
                episode_route=route,
                learning_seat=seat,
                confirmed_at_utc=CANDLE_CLOSE.isoformat(),
            )
        )

    def test_spread_larger_than_tp_remains_a_scored_cost_cohort(self) -> None:
        sources = _sources(
            mode="trend",
            branch="ACCEPTED",
            bid=150.000,
            ask=150.060,
        )
        feature_snapshot = sources[0]
        _frame(feature_snapshot, "M15")["indicators"]["atr_pips"] = 5.0
        feature_snapshot = _reseal_snapshot(feature_snapshot)
        _, _, anchor, route, seat = sources
        technical_shadow = build_fast_bot_technical_hypotheses(
            feature_snapshot,
            attempt_direction=anchor["attempt_direction"],
            branch_outcome=route["branch_outcome"],
            route_family=route["route_family"],
            spread_pips=seat["executable_spread_pips"],
            m5_atr_pips=seat["m5_atr_pips"],
            spread_to_m5_atr=seat["spread_to_m5_atr"],
        )

        result = _build((feature_snapshot, technical_shadow, anchor, route, seat))
        h01 = _vehicle(result, "H01")

        self.assertEqual(h01["status"], "EXACT_STOP_READY")
        self.assertGreater(
            h01["cost_observation"]["executable_spread_pips"],
            h01["cost_observation"]["gross_take_profit_pips"],
        )
        self.assertFalse(h01["cost_observation"]["cost_is_applicability_gate"])

    def test_short_stop_formulas_use_bid_and_downside_rail_geometry(self) -> None:
        result = _build(
            _sources(
                mode="trend_down",
                branch="ACCEPTED",
                attempt_direction="DOWN",
            )
        )

        expected = {
            "H01": (149.979, 149.899, 150.029),
            "H02": (149.999, 149.919, 150.049),
            "H04": (149.979, 149.879, 149.995),
            "H07": (149.969, 149.869, 149.995),
        }
        for hypothesis_id, prices in expected.items():
            row = _vehicle(result, hypothesis_id)
            self.assertEqual(row["status"], "EXACT_STOP_READY")
            execution = row["execution"]
            self.assertEqual(row["predicted_side"], "SHORT")
            self.assertEqual(
                (
                    execution["entry_price"],
                    execution["take_profit_price"],
                    execution["stop_loss_price"],
                ),
                prices,
            )
            self.assertEqual(execution["natural_trigger_component"], "BID")
            self.assertEqual(execution["exit_component"], "ASK")

    def test_exhaustion_uses_mean_and_opposite_donchian_without_fallback(self) -> None:
        sources = _sources(mode="exhaustion", branch="ACCEPTED")
        result = _build(sources)
        h06 = _vehicle(result, "H06")

        self.assertEqual(h06["status"], "EXACT_STOP_READY")
        self.assertEqual(h06["predicted_side"], "SHORT")
        self.assertEqual(h06["execution"]["entry_price"], 149.999)
        self.assertEqual(h06["execution"]["take_profit_price"], 149.950)
        self.assertEqual(h06["execution"]["stop_loss_price"], 150.051)

        feature_snapshot, _, anchor, route, seat = sources
        invalid = deepcopy(feature_snapshot)
        _frame(invalid, "M1")["indicators"]["bb_middle"] = 150.020
        invalid = _reseal_snapshot(invalid)
        shadow = build_fast_bot_technical_hypotheses(
            invalid,
            attempt_direction=anchor["attempt_direction"],
            branch_outcome=route["branch_outcome"],
            route_family=route["route_family"],
            spread_pips=seat["executable_spread_pips"],
            m5_atr_pips=seat["m5_atr_pips"],
            spread_to_m5_atr=seat["spread_to_m5_atr"],
        )
        blocked = _build((invalid, shadow, anchor, route, seat))
        blocked_h06 = _vehicle(blocked, "H06")

        self.assertEqual(blocked_h06["status"], "UNSCORABLE_CAUSAL_INPUT")
        self.assertIn("TAKE_PROFIT_NOT_FAVORABLE", blocked_h06["reasons"])
        self.assertIsNone(blocked_h06["execution"])

    def test_h03_h05_bind_exact_base_arm_without_copying_its_numbers(self) -> None:
        sources = _sources(mode="range", branch="REJECTED")
        result = _build(sources)
        seat = sources[-1]

        for hypothesis_id, side, method in (
            ("H03", "LONG", "RANGE_ROTATION"),
            ("H05", "SHORT", "BREAKOUT_FAILURE"),
        ):
            row = _vehicle(result, hypothesis_id)
            candidate = next(
                item
                for item in seat["candidates"]
                if item["side"] == side and item["method"] == method
            )
            base_arm = next(arm for arm in candidate["arms"] if arm["arm_id"] == "BASE")

            self.assertEqual(row["status"], "FROZEN_BASE_PROXY_READY")
            self.assertIsNone(row["execution"])
            self.assertEqual(
                row["proxy_binding"]["candidate_id"], candidate["candidate_id"]
            )
            self.assertEqual(row["proxy_binding"]["arm_id"], "BASE")
            self.assertEqual(row["proxy_binding"]["arm_sha256"], _sha(base_arm))
            self.assertEqual(
                row["proxy_binding"]["activation_at_utc"], CYCLE.isoformat()
            )
            self.assertEqual(
                row["proxy_binding"]["scoring_start_at_utc"], CYCLE.isoformat()
            )
            self.assertFalse(row["proxy_binding"]["pre_activation_price_path_allowed"])
            self.assertEqual(
                row["proxy_binding"]["path_lower_bound_policy"],
                "S5_INTERVAL_START_GTE_SCORING_START_AT_UTC",
            )
            serialized = json.dumps(row["proxy_binding"], sort_keys=True)
            for forbidden in (
                "entry_price",
                "take_profit_price",
                "stop_loss_price",
                "entry_ttl_seconds",
                "max_hold_seconds",
            ):
                self.assertNotIn(forbidden, serialized)

    def test_input_blocked_seat_keeps_geometry_but_never_enters_scorecard(self) -> None:
        sources = _sources(
            mode="trend",
            branch="ACCEPTED",
            input_blocked=True,
        )
        result = _build(sources)

        self.assertEqual(result["status"], "EMITTED_DIAGNOSTIC_ONLY")
        self.assertFalse(result["causal_input_proof_eligible"])
        self.assertFalse(result["paired_direction_proof_eligible"])
        self.assertFalse(result["technical_hypothesis_proof_eligible"])
        self.assertFalse(result["scorecard_eligible"])
        self.assertIn(
            "INPUT_BLOCKED_SHADOW_DIAGNOSTIC_ONLY",
            result["scorecard_ineligibility_reasons"],
        )
        self.assertGreater(result["diagnostic_vehicle_count"], 0)
        self.assertEqual(
            result["diagnostic_evaluable_vehicle_count"],
            result["diagnostic_vehicle_count"],
        )
        self.assertEqual(result["scoring_vehicle_count"], 0)

        h01 = _vehicle(result, "H01")
        self.assertEqual(h01["status"], "EXACT_STOP_DIAGNOSTIC_ONLY")
        self.assertIsNotNone(h01["execution"])
        self.assertTrue(h01["diagnostic_vehicle_available"])
        self.assertFalse(h01["scorecard_eligible"])
        self.assertFalse(h01["scoring_vehicle_available"])
        self.assertIn(
            "INPUT_BLOCKED_SHADOW_DIAGNOSTIC_ONLY",
            h01["scorecard_ineligibility_reasons"],
        )
        for row in result["vehicles"]:
            self.assertIn("causal_input_proof_eligible", row)
            self.assertIn("paired_direction_proof_eligible", row)
            self.assertIn("technical_hypothesis_proof_eligible", row)
            self.assertIn("diagnostic_vehicle_available", row)
            self.assertIn("diagnostic_evaluable", row)
            self.assertIn("scorecard_eligible", row)
            self.assertIn("scorecard_ineligibility_reasons", row)
            self.assertFalse(row["causal_input_proof_eligible"])
            self.assertFalse(row["paired_direction_proof_eligible"])
            self.assertFalse(row["technical_hypothesis_proof_eligible"])
            self.assertFalse(row["scorecard_eligible"])
            self.assertFalse(row["scoring_vehicle"])
            self.assertFalse(row["scoring_vehicle_available"])

        proxy_result = _build(
            _sources(
                mode="range",
                branch="REJECTED",
                input_blocked=True,
            )
        )
        for hypothesis_id in ("H03", "H05"):
            proxy = _vehicle(proxy_result, hypothesis_id)
            self.assertEqual(proxy["status"], "FROZEN_BASE_PROXY_DIAGNOSTIC_ONLY")
            self.assertIsNotNone(proxy["proxy_binding"])
            self.assertTrue(proxy["diagnostic_vehicle_available"])
            self.assertFalse(proxy["scorecard_eligible"])
            self.assertFalse(proxy["scoring_vehicle_available"])

        feature_snapshot, technical_shadow, anchor, route, seat = sources
        self.assertTrue(
            technical_hypothesis_vehicle_shadow_v2_valid(
                result,
                technical_feature_snapshot=feature_snapshot,
                technical_hypothesis_shadow=technical_shadow,
                episode_anchor=anchor,
                episode_route=route,
                learning_seat=seat,
                confirmed_at_utc=CANDLE_CLOSE.isoformat(),
            )
        )

    def test_validator_recomputes_top_and_row_scorecard_eligibility(self) -> None:
        sources = _sources(mode="trend", branch="ACCEPTED")
        result = _build(sources)
        feature_snapshot, technical_shadow, anchor, route, seat = sources

        tampered_top = deepcopy(result)
        tampered_top["scorecard_eligible"] = False
        tampered_top = _reseal_vehicle_shadow(tampered_top)
        self.assertFalse(
            technical_hypothesis_vehicle_shadow_v2_valid(
                tampered_top,
                technical_feature_snapshot=feature_snapshot,
                technical_hypothesis_shadow=technical_shadow,
                episode_anchor=anchor,
                episode_route=route,
                learning_seat=seat,
                confirmed_at_utc=CANDLE_CLOSE.isoformat(),
            )
        )

        tampered_row = deepcopy(result)
        _vehicle(tampered_row, "H01")["scorecard_eligible"] = False
        tampered_row = _reseal_vehicle_shadow(tampered_row)
        self.assertFalse(
            technical_hypothesis_vehicle_shadow_v2_valid(
                tampered_row,
                technical_feature_snapshot=feature_snapshot,
                technical_hypothesis_shadow=technical_shadow,
                episode_anchor=anchor,
                episode_route=route,
                learning_seat=seat,
                confirmed_at_utc=CANDLE_CLOSE.isoformat(),
            )
        )

    def test_m1_confirmation_clock_must_be_frozen_before_seat_generation(self) -> None:
        sources = _sources(mode="trend", branch="ACCEPTED")
        feature_snapshot, _, anchor, route, seat = sources

        mismatched_snapshot = deepcopy(feature_snapshot)
        _frame(mismatched_snapshot, "M1")["complete_candle_close_utc"] = (
            CANDLE_CLOSE - timedelta(minutes=1)
        ).isoformat()
        mismatched_snapshot = _reseal_snapshot(mismatched_snapshot)
        shadow = build_fast_bot_technical_hypotheses(
            mismatched_snapshot,
            attempt_direction=anchor["attempt_direction"],
            branch_outcome=route["branch_outcome"],
            route_family=route["route_family"],
            spread_pips=seat["executable_spread_pips"],
            m5_atr_pips=seat["m5_atr_pips"],
            spread_to_m5_atr=seat["spread_to_m5_atr"],
        )
        mismatched = _build((mismatched_snapshot, shadow, anchor, route, seat))
        self.assertEqual(mismatched["status"], "INVALID_CAUSAL_INPUT")
        self.assertIn(
            "FEATURE_M1_CLOSE_CONFIRMED_AT_BINDING_INVALID",
            mismatched["causal_input_errors"],
        )

        after_generation = build_fast_bot_technical_hypothesis_vehicles_v2(
            technical_feature_snapshot=feature_snapshot,
            technical_hypothesis_shadow=sources[1],
            episode_anchor=anchor,
            episode_route=route,
            learning_seat=seat,
            confirmed_at_utc=(CYCLE + timedelta(seconds=1)).isoformat(),
        )
        self.assertEqual(after_generation["status"], "INVALID_CAUSAL_INPUT")
        self.assertIn(
            "CONFIRMED_AT_AFTER_SEAT_GENERATION",
            after_generation["causal_input_errors"],
        )

        early_seat = deepcopy(seat)
        before_close = CANDLE_CLOSE - timedelta(seconds=1)
        early_seat["generated_at_utc"] = before_close.isoformat()
        early_seat["quote_timestamp_utc"] = before_close.isoformat()
        early_seat = _seal(
            {key: item for key, item in early_seat.items() if key != "contract_sha256"}
        )
        m1_after_generation = build_fast_bot_technical_hypothesis_vehicles_v2(
            technical_feature_snapshot=feature_snapshot,
            technical_hypothesis_shadow=sources[1],
            episode_anchor=anchor,
            episode_route=route,
            learning_seat=early_seat,
            confirmed_at_utc=CANDLE_CLOSE.isoformat(),
        )
        self.assertEqual(m1_after_generation["status"], "INVALID_CAUSAL_INPUT")
        self.assertIn(
            "LEARNING_SEAT_M1_CLOSE_AFTER_GENERATION",
            m1_after_generation["causal_input_errors"],
        )

    def test_h08_is_an_explicit_zero_control_only_when_active(self) -> None:
        result = _build(_sources(mode="quiet", branch="ACCEPTED"))
        h08 = _vehicle(result, "H08")

        self.assertEqual(h08["status"], "ZERO_PNL_CONTROL_READY")
        self.assertTrue(h08["scoring_vehicle_available"])
        self.assertEqual(
            h08["control"],
            {
                "control_type": "NO_TRADE_ZERO_PNL",
                "net_pips": 0.0,
                "filled": False,
            },
        )
        self.assertIsNone(h08["execution"])

    def test_module_has_no_outcome_reader_or_broker_authority_dependency(self) -> None:
        source = inspect.getsource(vehicle_module)

        for forbidden in (
            "fast_bot_episode_truth",
            "fast_bot_learning_truth",
            "OandaReadOnlyClient",
            "OandaExecutionClient",
            "close_trade",
            "create_order",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
