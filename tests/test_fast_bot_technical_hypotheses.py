from __future__ import annotations

import hashlib
import json
import unittest
from copy import deepcopy
from typing import Any

from quant_rabbit.fast_bot_technical_hypotheses import (
    CATALOG_CONTRACT,
    build_fast_bot_technical_hypotheses,
    technical_hypothesis_catalog,
    technical_hypothesis_shadow_valid,
)


TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D")
ROUTES = {
    "ACCEPTED": "BREAKOUT_CONTINUATION",
    "REJECTED": "RANGE_RECLAIM_OR_BREAKOUT_FAILURE",
}


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


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "contract_sha256": _sha(value)}


def _seal_row(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "feature_sha256": _sha(value)}


def _reseal_snapshot(
    value: dict[str, Any], *, reseal_rows: bool = True
) -> dict[str, Any]:
    if reseal_rows:
        value["timeframes"] = [
            _seal_row(
                {
                    key: item
                    for key, item in row.items()
                    if key != "feature_sha256"
                }
            )
            for row in value["timeframes"]
        ]
    return _seal(
        {key: item for key, item in value.items() if key != "contract_sha256"}
    )


def _row(value: dict[str, Any], timeframe: str) -> dict[str, Any]:
    return next(
        item for item in value["timeframes"] if item["timeframe"] == timeframe
    )


def _snapshot(
    *, mode: str = "trend", cycle: str = "2026-07-17T00:35:10+00:00"
) -> dict[str, Any]:
    close = cycle.replace(":10+00:00", ":00+00:00")
    rows = []
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
                "close": 100.0,
                "rsi_14": 32.0,
                "adx_14": 18.0,
                "choppiness_14": 62.0,
                "hurst_100": 0.4,
                "z_score_20": -1.4,
                "atr_pips": 4.0,
            }
            rsi_series = [45.0, 36.0, 28.0, 29.0, 32.0]
            atr_series = [4.4, 4.3, 4.2, 4.1, 4.0]
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
            rsi_series = [47.0, 49.0, 51.0, 56.0, 63.0]
            atr_series = [3.0, 3.4, 3.8, 4.4, 5.0]
        rows.append(
            _seal_row(
                {
                    "timeframe": timeframe,
                    "complete_candle_close_utc": close,
                    "market_state": market,
                    "indicators": indicators,
                    "indicator_series": {
                        "rsi_14": rsi_series,
                        "macd_hist": [-0.2, -0.1, 0.0, 0.1, 0.2],
                        "adx_14": [20.0, 22.0, 24.0, 26.0, 28.0],
                        "atr_pips": atr_series,
                        "ema_12_minus_50_pips": [
                            -0.2,
                            -0.1,
                            0.1,
                            0.5,
                            1.0,
                        ],
                    },
                }
            )
        )
    return _seal(
        {
            "contract": "QR_FAST_BOT_EPISODE_TECHNICAL_FEATURE_SNAPSHOT_V1",
            "schema_version": 1,
            "pair": "EUR_JPY",
            "handoff_cycle_generated_at_utc": cycle,
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


def _build(
    snapshot: dict[str, Any],
    *,
    attempt: str = "UP",
    branch: str = "ACCEPTED",
    route: str | None = None,
    spread: float = 0.8,
    atr: float | None = None,
    ratio: float | None = None,
) -> dict[str, Any]:
    m5_atr = (
        float(_row(snapshot, "M5")["indicators"]["atr_pips"])
        if atr is None
        else atr
    )
    return build_fast_bot_technical_hypotheses(
        snapshot,
        attempt_direction=attempt,
        branch_outcome=branch,
        route_family=route or ROUTES.get(branch, "AMBIGUOUS"),
        spread_pips=spread,
        m5_atr_pips=m5_atr,
        spread_to_m5_atr=spread / m5_atr if ratio is None else ratio,
    )


def _exhaustion_snapshot(*, correct_extension: bool) -> dict[str, Any]:
    snapshot = deepcopy(_snapshot())
    extension = "OVERBOUGHT" if correct_extension else "OVERSOLD"
    for timeframe in ("M5", "M15", "H1"):
        _row(snapshot, timeframe)["market_state"]["extension"] = extension
    m5 = _row(snapshot, "M5")
    m5["indicators"]["rsi_14"] = 75.0 if correct_extension else 25.0
    m5["indicators"]["adx_14"] = 28.0
    m5["indicators"]["atr_pips"] = 5.0
    m5["indicator_series"]["adx_14"] = [36.0, 34.0, 32.0, 30.0, 28.0]
    m5["indicator_series"]["macd_hist"] = [0.6, 0.5, 0.4, 0.3, 0.2]
    m5["indicator_series"]["atr_pips"] = [8.0, 7.0, 6.0, 5.5, 5.0]
    m1 = _row(snapshot, "M1")
    m1["market_state"]["direction"] = "DOWN"
    m1["market_state"]["readiness"] = "TRIGGERED"
    return _reseal_snapshot(snapshot)


class FastBotTechnicalHypothesesTest(unittest.TestCase):
    def test_catalog_is_fixed_and_zero_authority(self) -> None:
        catalog = technical_hypothesis_catalog()
        self.assertEqual(catalog["contract"], CATALOG_CONTRACT)
        self.assertEqual(len(catalog["hypotheses"]), 8)
        self.assertEqual(
            catalog["evaluator_policy"],
            "QR_FAST_BOT_TECHNICAL_HYPOTHESIS_EVALUATOR_V1",
        )
        policies = {
            row["hypothesis_id"]: row["arm_truth_join_policy"]
            for row in catalog["hypotheses"]
        }
        self.assertEqual(policies["H03"], "EXISTING_PASSIVE_QUOTE_PROXY")
        self.assertEqual(
            policies["H01"], "REQUIRES_HYPOTHESIS_SPECIFIC_VEHICLE"
        )
        self.assertFalse(catalog["promotion_allowed"])
        self.assertEqual(catalog["order_authority"], "NONE")

    def test_trend_context_uses_directionless_strength_and_fresh_cross(self) -> None:
        result = _build(_snapshot())
        rows = {row["hypothesis_id"]: row for row in result["hypotheses"]}
        self.assertEqual(result["status"], "EMITTED")
        self.assertEqual(rows["H01"]["status"], "ACTIVE_SHADOW")
        self.assertEqual(rows["H01"]["predicted_side"], "LONG")
        self.assertIn(
            "M5_ADX_STRENGTHENING_DIRECTIONLESS", rows["H01"]["evidence"]
        )
        self.assertEqual(rows["H01"]["raw_confluence_score"], 4)
        self.assertEqual(rows["H02"]["status"], "ACTIVE_SHADOW")
        self.assertIn(
            "M1_FRESH_EMA12_50_CROSS_AGE_2", rows["H02"]["evidence"]
        )
        self.assertFalse(rows["H01"]["raw_confluence_score_is_probability"])
        self.assertEqual(rows["H08"]["status"], "INACTIVE_SHADOW")
        self.assertEqual(
            result["cost_state"]["cost_role"],
            "STATE_VARIABLE_AND_REQUIRED_EDGE_THRESHOLD_NOT_DIRECTION",
        )

    def test_range_context_requires_edge_group_and_fresh_reentry(self) -> None:
        result = _build(
            _snapshot(mode="range"),
            attempt="DOWN",
            branch="REJECTED",
        )
        rows = {row["hypothesis_id"]: row for row in result["hypotheses"]}
        self.assertEqual(rows["H03"]["status"], "ACTIVE_SHADOW")
        self.assertEqual(rows["H03"]["predicted_side"], "LONG")
        self.assertEqual(rows["H03"]["raw_confluence_score"], 4)
        self.assertIn(
            "M5_ZSCORE_BB_DONCHIAN_EDGE_GROUP_UP", rows["H03"]["evidence"]
        )
        self.assertEqual(rows["H05"]["status"], "ACTIVE_SHADOW")
        self.assertEqual(rows["H05"]["predicted_side"], "LONG")

    def test_future_complete_candle_is_rejected_even_when_resealed(self) -> None:
        snapshot = deepcopy(_snapshot())
        _row(snapshot, "M1")["complete_candle_close_utc"] = (
            "2026-07-17T00:36:00+00:00"
        )
        result = _build(_reseal_snapshot(snapshot))
        self.assertEqual(result["status"], "INVALID_FEATURE_SNAPSHOT")
        self.assertEqual(result["hypotheses"], [])

    def test_tampered_row_sha_is_rejected_even_when_outer_snapshot_is_resealed(
        self,
    ) -> None:
        snapshot = deepcopy(_snapshot())
        _row(snapshot, "M1")["indicators"]["rsi_14"] = 99.0
        result = _build(_reseal_snapshot(snapshot, reseal_rows=False))
        self.assertEqual(result["status"], "INVALID_FEATURE_SNAPSHOT")

    def test_nonzero_authority_or_unallowlisted_feature_is_rejected(self) -> None:
        authority = deepcopy(_snapshot())
        authority["order_authority"] = "BOT"
        self.assertEqual(
            _build(_reseal_snapshot(authority))["status"],
            "INVALID_FEATURE_SNAPSHOT",
        )

        unexpected = deepcopy(_snapshot())
        _row(unexpected, "M5")["indicators"]["secret_vote"] = 1.0
        self.assertEqual(
            _build(_reseal_snapshot(unexpected))["status"],
            "INVALID_FEATURE_SNAPSHOT",
        )

    def test_nonfinite_or_out_of_bounds_series_is_rejected(self) -> None:
        snapshot = deepcopy(_snapshot())
        _row(snapshot, "M5")["indicator_series"]["adx_14"][-1] = 1.0e13
        self.assertEqual(
            _build(_reseal_snapshot(snapshot))["status"],
            "INVALID_FEATURE_SNAPSHOT",
        )

    def test_stale_crosses_do_not_activate_pullback(self) -> None:
        snapshot = deepcopy(_snapshot())
        m1_series = _row(snapshot, "M1")["indicator_series"]
        m1_series["rsi_14"] = [49.0, 51.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0]
        m1_series["macd_hist"] = [-0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        m1_series["ema_12_minus_50_pips"] = [-0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        rows = {
            row["hypothesis_id"]: row
            for row in _build(_reseal_snapshot(snapshot))["hypotheses"]
        }
        self.assertEqual(rows["H02"]["status"], "INACTIVE_SHADOW")
        self.assertIn("M1_REACCELERATION_NOT_CONFIRMED", rows["H02"]["blockers"])

    def test_missing_adx_series_fails_closed_for_trend_strength(self) -> None:
        snapshot = deepcopy(_snapshot())
        del _row(snapshot, "M5")["indicator_series"]["adx_14"]
        rows = {
            row["hypothesis_id"]: row
            for row in _build(_reseal_snapshot(snapshot))["hypotheses"]
        }
        self.assertEqual(rows["H01"]["status"], "INACTIVE_SHADOW")
        self.assertIn("M5_ADX_WEAK_OR_FALLING", rows["H01"]["blockers"])

    def test_range_rotation_fails_without_zscore_bb_or_donchian_location(self) -> None:
        snapshot = deepcopy(_snapshot(mode="range"))
        del _row(snapshot, "M5")["indicators"]["z_score_20"]
        rows = {
            row["hypothesis_id"]: row
            for row in _build(
                _reseal_snapshot(snapshot), attempt="DOWN", branch="REJECTED"
            )["hypotheses"]
        }
        self.assertEqual(rows["H03"]["status"], "INACTIVE_SHADOW")
        self.assertIn(
            "RANGE_EDGE_OR_RAW_LOCATION_GROUP_UNAVAILABLE", rows["H03"]["blockers"]
        )

    def test_pretrend_breakout_requires_atr_expansion(self) -> None:
        snapshot = deepcopy(_snapshot())
        _row(snapshot, "M15")["market_state"]["phase"] = "PRE_TREND"
        m5 = _row(snapshot, "M5")
        m5["indicator_series"]["atr_pips"] = [8.0, 7.0, 6.0, 5.5, 5.0]
        rows = {
            row["hypothesis_id"]: row
            for row in _build(_reseal_snapshot(snapshot))["hypotheses"]
        }
        self.assertEqual(rows["H04"]["status"], "INACTIVE_SHADOW")
        self.assertIn("M5_ATR_NOT_EXPANDING", rows["H04"]["blockers"])

    def test_exhaustion_reversal_rejects_wrong_direction_extension(self) -> None:
        rows = {
            row["hypothesis_id"]: row
            for row in _build(_exhaustion_snapshot(correct_extension=False))[
                "hypotheses"
            ]
        }
        self.assertEqual(rows["H06"]["status"], "INACTIVE_SHADOW")
        self.assertIn(
            "DIRECTIONAL_EXTENSION_NOT_CONFIRMED_UP", rows["H06"]["blockers"]
        )

    def test_exhaustion_requires_atr_deceleration_and_closed_m1_reversal(self) -> None:
        snapshot = _exhaustion_snapshot(correct_extension=True)
        rows = {
            row["hypothesis_id"]: row for row in _build(snapshot)["hypotheses"]
        }
        self.assertEqual(rows["H06"]["status"], "ACTIVE_SHADOW")
        self.assertEqual(rows["H06"]["predicted_side"], "SHORT")
        self.assertIn(
            "M5_ATR_DECELERATING_DIRECTIONLESS", rows["H06"]["evidence"]
        )
        self.assertEqual(rows["H06"]["raw_confluence_score"], 4)
        self.assertEqual(rows["H08"]["status"], "ACTIVE_SHADOW")
        self.assertIn(
            "ACTIVE_DIRECTIONAL_HYPOTHESIS_CONFLICT", rows["H08"]["evidence"]
        )

        no_atr_deceleration = deepcopy(snapshot)
        _row(no_atr_deceleration, "M5")["indicator_series"]["atr_pips"] = [
            3.0,
            3.4,
            3.8,
            4.4,
            5.0,
        ]
        rows = {
            row["hypothesis_id"]: row
            for row in _build(_reseal_snapshot(no_atr_deceleration))["hypotheses"]
        }
        self.assertEqual(rows["H06"]["status"], "INACTIVE_SHADOW")

        armed_only = deepcopy(snapshot)
        _row(armed_only, "M1")["market_state"]["readiness"] = "ARMED"
        rows = {
            row["hypothesis_id"]: row
            for row in _build(_reseal_snapshot(armed_only))["hypotheses"]
        }
        self.assertEqual(rows["H06"]["status"], "INACTIVE_SHADOW")

    def test_rejected_branch_cannot_activate_session_expansion(self) -> None:
        snapshot = deepcopy(
            _snapshot(cycle="2026-07-17T07:35:10+00:00")
        )
        _row(snapshot, "M15")["indicators"]["bb_squeeze"] = 1
        rows = {
            row["hypothesis_id"]: row
            for row in _build(
                _reseal_snapshot(snapshot), branch="REJECTED", attempt="UP"
            )["hypotheses"]
        }
        self.assertEqual(rows["H07"]["status"], "INACTIVE_SHADOW")
        self.assertIn("SEALED_BREAK_NOT_ACCEPTED", rows["H07"]["blockers"])
        self.assertNotIn("M15_PRE_SESSION_COMPRESSION", rows["H07"]["evidence"])

    def test_invalid_cost_and_incoherent_route_fail_closed(self) -> None:
        snapshot = _snapshot()
        self.assertEqual(
            _build(snapshot, ratio=0.5)["status"], "INVALID_COST_STATE"
        )
        self.assertEqual(
            _build(snapshot, atr=6.0, ratio=0.8 / 6.0)["status"],
            "INVALID_COST_STATE",
        )
        self.assertEqual(
            _build(
                snapshot,
                branch="ACCEPTED",
                route="RANGE_RECLAIM_OR_BREAKOUT_FAILURE",
            )["status"],
            "INVALID_EPISODE_CONTEXT",
        )
        self.assertEqual(
            _build(snapshot, attempt="SIDEWAYS")["status"],
            "INVALID_EPISODE_CONTEXT",
        )

    def test_shadow_validator_rebuilds_every_nested_row(self) -> None:
        snapshot = _snapshot()
        shadow = _build(snapshot)
        kwargs = {
            "feature_snapshot": snapshot,
            "attempt_direction": "UP",
            "branch_outcome": "ACCEPTED",
            "route_family": "BREAKOUT_CONTINUATION",
            "spread_pips": 0.8,
            "m5_atr_pips": 5.0,
            "spread_to_m5_atr": 0.16,
        }
        self.assertTrue(technical_hypothesis_shadow_valid(shadow, **kwargs))

        tampered = deepcopy(shadow)
        tampered["hypotheses"][0]["raw_confluence_score"] = 99
        tampered = _seal(
            {
                key: item
                for key, item in tampered.items()
                if key != "contract_sha256"
            }
        )
        self.assertFalse(technical_hypothesis_shadow_valid(tampered, **kwargs))
        unknown_policy = deepcopy(shadow)
        unknown_policy["evaluator_policy"] = (
            "QR_FAST_BOT_TECHNICAL_HYPOTHESIS_EVALUATOR_V2"
        )
        unknown_policy = _seal(
            {
                key: item
                for key, item in unknown_policy.items()
                if key != "contract_sha256"
            }
        )
        self.assertFalse(
            technical_hypothesis_shadow_valid(unknown_policy, **kwargs)
        )
        self.assertFalse(
            technical_hypothesis_shadow_valid(
                shadow,
                **{**kwargs, "spread_to_m5_atr": 0.5},
            )
        )

    def test_output_is_deterministic_for_same_frozen_input(self) -> None:
        snapshot = _snapshot()
        self.assertEqual(_build(snapshot), _build(snapshot))


if __name__ == "__main__":
    unittest.main()
