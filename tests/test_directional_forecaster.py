"""Unit tests for strategy/directional_forecaster.py."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from quant_rabbit.analysis.candles import (
    PAIR_TECHNICAL_CANDLE_INTEGRITY_SCHEMA,
    TECHNICAL_CANDLE_INDICATOR_WARMUP_MIN_CLEAN_COUNT,
    TECHNICAL_CANDLE_INTEGRITY_SCHEMA,
    TECHNICAL_CANDLE_PROVENANCE_INVALID,
    TECHNICAL_CANDLE_QUARANTINE_DETAIL_LIMIT,
    TECHNICAL_CANDLE_QUARANTINE_DETAILS_ORDER,
    TECHNICAL_CANDLE_QUARANTINE_DETAILS_SELECTION,
    TECHNICAL_CANDLE_SPREAD_PROVENANCE_ONLY_MODE,
    TECHNICAL_CANDLE_SPREAD_CONTAMINATED,
)
from quant_rabbit.instruments import (
    NORMAL_SPREAD_PIPS,
    OANDA_SPREAD_CALIBRATION_V1,
)
from quant_rabbit.risk import RiskPolicy
from quant_rabbit.strategy.directional_forecaster import (
    FORECAST_D_ANCHOR_HORIZON_MIN,
    FORECAST_EXECUTION_HORIZON_MIN,
    FORECAST_OPERATING_SWING_HORIZON_MIN,
    M15_RECOVERY_MICRO_CONTRACT,
    M15_RECOVERY_MICRO_MAX_UNITS,
    TECHNICAL_CANDLE_FORECAST_MAX_AGE_SECONDS,
    TECHNICAL_CANDLE_FORECAST_MAX_FUTURE_SKEW_SECONDS,
    _valid_mba_integrity,
    build_m15_recovery_micro_receipt,
    synthesize_forecast as _synthesize_forecast,
    validate_m15_recovery_micro_receipt,
)


def synthesize_forecast(**kwargs):
    """Hand-built unit charts opt out; production-contract tests opt back in."""

    kwargs.setdefault("require_technical_candle_integrity", False)
    return _synthesize_forecast(**kwargs)


@dataclass
class _Sig:
    direction: str
    bonus_magnitude: float
    confidence: float
    rationale: str = ""
    timeframe: str | None = None


@dataclass
class _NamedSig:
    name: str
    direction: str
    bonus_magnitude: float
    confidence: float
    rationale: str = ""


def _production_mba_integrity(*, recovered: bool) -> dict[str, object]:
    normal_spread_pips = NORMAL_SPREAD_PIPS["EUR_USD"]
    max_spread_multiple = RiskPolicy().max_spread_multiple
    execution_spread_cap_pips = round(
        normal_spread_pips * max_spread_multiple,
        6,
    )
    spread_cap_pips = round(
        OANDA_SPREAD_CALIBRATION_V1.pairs["EUR_USD"].max_pips,
        6,
    )
    blocking_codes = [] if recovered else [TECHNICAL_CANDLE_SPREAD_CONTAMINATED]
    latest_complete = "2026-07-14T00:43:00+00:00" if recovered else "2026-07-14T00:44:00+00:00"
    latest_clean = "2026-07-14T00:43:00+00:00"
    quarantine_timestamp = "2026-07-14T00:13:00+00:00" if recovered else latest_complete
    requested_count = 70 if recovered else 41
    clean_count = 69 if recovered else 40
    recent_clean_tail_count = (
        TECHNICAL_CANDLE_INDICATOR_WARMUP_MIN_CLEAN_COUNT if recovered else 0
    )
    item = {
        "schema": TECHNICAL_CANDLE_INTEGRITY_SCHEMA,
        "pair": "EUR_USD",
        "granularity": "M1",
        "payload_instrument": "EUR_USD",
        "payload_granularity": "M1",
        "source": "OANDA_MBA",
        "requested_price": "MBA",
        "spread_evaluation_mode": "EXECUTION_ENDPOINT_CAP",
        "evaluation_status": "DEGRADED" if recovered else "BLOCKED",
        "policy_source": "OANDA_SPREAD_CALIBRATION_V1.pairs.max_pips",
        "spread_calibration_sha256": OANDA_SPREAD_CALIBRATION_V1.calibration_sha256,
        "normal_spread_pips": normal_spread_pips,
        "max_spread_multiple": max_spread_multiple,
        "execution_spread_cap_pips": execution_spread_cap_pips,
        "spread_cap_pips": spread_cap_pips,
        "requested_count": requested_count,
        "raw_entry_count": requested_count,
        "coverage_complete": True,
        "complete_entry_count": requested_count,
        "clean_count": clean_count,
        "indicator_warmup_min_clean_count": TECHNICAL_CANDLE_INDICATOR_WARMUP_MIN_CLEAN_COUNT,
        "recent_clean_tail_count": recent_clean_tail_count,
        "indicator_warmup_complete": True,
        "recent_clean_coverage_complete": recovered,
        "contaminated_count": 1,
        "malformed_count": 0,
        "quarantined_count": 1,
        "recent_tail_state": "CLEAN" if recovered else "SPREAD_CONTAMINATED",
        "recent_tail_contaminated": not recovered,
        "recent_tail_invalid": False,
        "provenance_complete": True,
        "forecast_blocking": not recovered,
        "codes": [TECHNICAL_CANDLE_SPREAD_CONTAMINATED],
        "blocking_codes": blocking_codes,
        "latest_complete_timestamp_utc": latest_complete,
        "latest_clean_timestamp_utc": latest_clean,
        "quarantine_details": [{
            "timestamp_utc": quarantine_timestamp,
            "code": TECHNICAL_CANDLE_SPREAD_CONTAMINATED,
            "max_spread_pips": spread_cap_pips + 1.0,
            "spread_cap_pips": spread_cap_pips,
        }],
        "quarantine_details_truncated": 0,
        "quarantine_details_window": {
            "selection": TECHNICAL_CANDLE_QUARANTINE_DETAILS_SELECTION,
            "order": TECHNICAL_CANDLE_QUARANTINE_DETAILS_ORDER,
            "limit": TECHNICAL_CANDLE_QUARANTINE_DETAIL_LIMIT,
            "start_index": 0,
            "end_index_exclusive": 1,
            "total_count": 1,
            "total_code_counts": {
                TECHNICAL_CANDLE_SPREAD_CONTAMINATED: 1,
                TECHNICAL_CANDLE_PROVENANCE_INVALID: 0,
            },
            "published_code_counts": {
                TECHNICAL_CANDLE_SPREAD_CONTAMINATED: 1,
                TECHNICAL_CANDLE_PROVENANCE_INVALID: 0,
            },
            "omitted_code_counts": {
                TECHNICAL_CANDLE_SPREAD_CONTAMINATED: 0,
                TECHNICAL_CANDLE_PROVENANCE_INVALID: 0,
            },
            "total_timestamped_count": 1,
            "published_timestamped_count": 1,
            "omitted_timestamped_count": 0,
            "latest_timestamp_utc": quarantine_timestamp,
        },
    }
    m5_item = copy.deepcopy(item)
    m5_item.update({
        "granularity": "M5",
        "payload_granularity": "M5",
        "evaluation_status": "PASS",
        "requested_count": 41,
        "raw_entry_count": 41,
        "coverage_complete": True,
        "complete_entry_count": 41,
        "clean_count": 41,
        "indicator_warmup_min_clean_count": TECHNICAL_CANDLE_INDICATOR_WARMUP_MIN_CLEAN_COUNT,
        "recent_clean_tail_count": 41,
        "indicator_warmup_complete": True,
        "recent_clean_coverage_complete": True,
        "contaminated_count": 0,
        "quarantined_count": 0,
        "recent_tail_state": "CLEAN",
        "recent_tail_contaminated": False,
        "forecast_blocking": False,
        "codes": [],
        "blocking_codes": [],
        "latest_complete_timestamp_utc": "2026-07-14T00:40:00+00:00",
        "latest_clean_timestamp_utc": "2026-07-14T00:40:00+00:00",
        "quarantine_details": [],
        "quarantine_details_window": {
            "selection": TECHNICAL_CANDLE_QUARANTINE_DETAILS_SELECTION,
            "order": TECHNICAL_CANDLE_QUARANTINE_DETAILS_ORDER,
            "limit": TECHNICAL_CANDLE_QUARANTINE_DETAIL_LIMIT,
            "start_index": 0,
            "end_index_exclusive": 0,
            "total_count": 0,
            "total_code_counts": {
                TECHNICAL_CANDLE_SPREAD_CONTAMINATED: 0,
                TECHNICAL_CANDLE_PROVENANCE_INVALID: 0,
            },
            "published_code_counts": {
                TECHNICAL_CANDLE_SPREAD_CONTAMINATED: 0,
                TECHNICAL_CANDLE_PROVENANCE_INVALID: 0,
            },
            "omitted_code_counts": {
                TECHNICAL_CANDLE_SPREAD_CONTAMINATED: 0,
                TECHNICAL_CANDLE_PROVENANCE_INVALID: 0,
            },
            "total_timestamped_count": 0,
            "published_timestamped_count": 0,
            "omitted_timestamped_count": 0,
            "latest_timestamp_utc": None,
        },
    })
    return {
        "schema": PAIR_TECHNICAL_CANDLE_INTEGRITY_SCHEMA,
        "pair": "EUR_USD",
        "source": "OANDA_MBA",
        "evaluation_status": "DEGRADED" if recovered else "BLOCKED",
        "forecast_blocking": not recovered,
        "codes": [TECHNICAL_CANDLE_SPREAD_CONTAMINATED],
        "blocking_codes": blocking_codes,
        "requested_timeframes": ["M1", "M5"],
        "evaluated_timeframe_count": 2,
        "timeframes": {"M1": item, "M5": m5_item},
    }


class ForecastGeometryTest(unittest.TestCase):
    def test_down_forecast_uses_only_below_current_targets(self) -> None:
        pair_chart = {
            "views": [
                {
                    "granularity": "M15",
                    "indicators": {"pip_size": 0.0001},
                    "structure": {
                        "liquidity": [
                            {"side": "EQ_LOW", "price": 1.1015, "indices": [1, 2, 3, 4]},
                            {"side": "EQ_HIGH", "price": 1.1020, "indices": [1, 2, 3, 4]},
                        ],
                        "swings": [
                            {"side": "HIGH", "price": 1.1020},
                            {"side": "LOW", "price": 1.1015},
                        ],
                    },
                },
                {
                    "granularity": "H1",
                    "indicators": {"pip_size": 0.0001},
                    "structure": {
                        "liquidity": [
                            {"side": "EQ_LOW", "price": 1.0980, "indices": [1, 2, 3, 4]},
                        ],
                        "swings": [
                            {"side": "LOW", "price": 1.0980},
                        ],
                    },
                },
            ]
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[_Sig("DOWN", 30.0, 1.0, "breakdown")],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "DOWN")
        self.assertIsNotNone(forecast.target_price)
        self.assertLess(forecast.target_price, 1.1000)
        self.assertIsNotNone(forecast.invalidation_price)
        self.assertGreater(forecast.invalidation_price, 1.1000)
        self.assertIn("breakdown", " ".join(forecast.drivers_for))
        self.assertEqual(forecast.current_price, 1.1)
        self.assertGreater(forecast.down_score, 0.0)

    def test_forecast_target_prefers_nearest_executable_level_over_distant_structural_tp(self) -> None:
        pair_chart = {
            "views": [
                {
                    "granularity": "M5",
                    "indicators": {
                        "pip_size": 0.0001,
                        "atr_pips": 2.0,
                    },
                    "structure": {
                        "swings": [
                            {"side": "LOW", "price": 1.0990},
                            {"side": "HIGH", "price": 1.1005},
                        ],
                    },
                }
            ]
        }

        with patch(
            "quant_rabbit.strategy.directional_forecaster.structural_tp_target",
            return_value=(1.0950, "far H1 structural target"),
        ):
            forecast = synthesize_forecast(
                pair="EUR_USD",
                pair_chart=pair_chart,
                current_price=1.1000,
                pattern_signals=[],
                projection_signals=[_Sig("DOWN", 60.0, 1.0, "breakdown")],
                correlation_signals=[],
                paths=[],
            )

        self.assertEqual(forecast.direction, "DOWN")
        self.assertEqual(forecast.target_price, 1.099)
        self.assertEqual(forecast.invalidation_price, 1.1005)

    def test_forecast_invalidation_ignores_levels_inside_operating_noise(self) -> None:
        pair_chart = {
            "views": [
                {
                    "granularity": "M5",
                    "indicators": {
                        "pip_size": 0.0001,
                        "atr_pips": 2.0,
                        "donchian_high": 1.10004,
                        "donchian_low": 1.09965,
                    },
                    "structure": {
                        "swings": [
                            {"side": "HIGH", "price": 1.10004},
                            {"side": "HIGH", "price": 1.10040},
                            {"side": "LOW", "price": 1.09965},
                        ],
                    },
                }
            ]
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[_Sig("DOWN", 60.0, 1.0, "breakdown")],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "DOWN")
        self.assertEqual(forecast.invalidation_price, 1.1004)
        self.assertLess(forecast.target_price, 1.1000)

    def test_incomplete_noise_cleared_geometry_reduces_forecast_confidence(self) -> None:
        pair_chart = {
            "views": [
                {
                    "granularity": "M5",
                    "indicators": {
                        "pip_size": 0.0001,
                        "atr_pips": 2.0,
                        "donchian_high": 1.10040,
                        "donchian_low": 1.09992,
                    },
                    "structure": {
                        "swings": [
                            {"side": "HIGH", "price": 1.10040},
                            {"side": "LOW", "price": 1.09992},
                        ],
                    },
                }
            ]
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[_Sig("DOWN", 60.0, 1.0, "breakdown")],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "DOWN")
        self.assertIsNone(forecast.target_price)
        self.assertEqual(forecast.invalidation_price, 1.1004)
        self.assertLess(forecast.confidence, forecast.raw_confidence)
        self.assertIn("robust forecast geometry missing target", forecast.rationale_summary)

    def test_rollover_spread_does_not_move_forecast_geometry(self) -> None:
        pair_chart = {
            "views": [
                {
                    "granularity": "M5",
                    "indicators": {
                        "pip_size": 0.0001,
                        "atr_pips": 2.0,
                        "donchian_high": 1.10120,
                        "donchian_low": 1.09880,
                    },
                    "structure": {
                        "swings": [
                            {"side": "HIGH", "price": 1.10120},
                            {"side": "LOW", "price": 1.09880},
                        ],
                    },
                }
            ]
        }

        normal_spread = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[_Sig("DOWN", 60.0, 1.0, "breakdown")],
            correlation_signals=[],
            paths=[],
            spread_pips=0.8,
        )
        rollover_spread = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[_Sig("DOWN", 60.0, 1.0, "breakdown")],
            correlation_signals=[],
            paths=[],
            spread_pips=20.0,
        )

        self.assertEqual(normal_spread.direction, "DOWN")
        self.assertEqual(normal_spread.target_price, 1.0988)
        self.assertEqual(normal_spread.invalidation_price, 1.1012)
        self.assertEqual(rollover_spread.target_price, normal_spread.target_price)
        self.assertEqual(rollover_spread.invalidation_price, normal_spread.invalidation_price)
        self.assertEqual(rollover_spread.confidence, normal_spread.confidence)

    def test_lagging_htf_downtrend_dampens_micro_up_signal_to_unclear(self) -> None:
        pair_chart = {
            "confluence": {
                "score_balance": "SHORT_LEAN",
                "score_gap": -0.75,
                "tf_agreement_score": 1.0,
                "dominant_regime": "TREND_DOWN",
            },
            "views": [
                {
                    "granularity": "H1",
                    "regime": "TREND_DOWN",
                    "indicators": {"pip_size": 0.0001, "adx_14": 48.0},
                    "structure": {"structure_events": []},
                },
                {
                    "granularity": "H4",
                    "regime": "TREND_DOWN",
                    "indicators": {"pip_size": 0.0001, "adx_14": 35.0},
                    "structure": {"structure_events": []},
                },
                {
                    "granularity": "M1",
                    "regime": "RANGE",
                    "indicators": {"pip_size": 0.0001, "adx_14": 12.0},
                    "structure": {
                        "structure_events": [
                            {"kind": "BOS_UP", "close_confirmed": True, "broken_pivot_price": 1.1001}
                        ]
                    },
                },
            ],
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1000,
            pattern_signals=[_Sig("UP", 40.0, 1.0, "M1-only bottom bounce")],
            projection_signals=[],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "UNCLEAR")
        self.assertEqual(forecast.up_score, 0.0)
        self.assertEqual(forecast.down_score, 0.0)
        self.assertIn("lagging indicator bias requires M15/H1 close-confirmed reversal", forecast.rationale_summary)
        self.assertIn("SHORT_LEAN", forecast.rationale_summary)

    def test_countertrend_confirmation_ignores_stale_m15_h1_structure(self) -> None:
        pair_chart = {
            "confluence": {
                "score_balance": "SHORT_LEAN",
                "score_gap": -0.75,
                "tf_agreement_score": 1.0,
                "dominant_regime": "TREND_DOWN",
            },
            "views": [
                {
                    "granularity": "M15",
                    "regime": "TREND_DOWN",
                    "indicators": {"pip_size": 0.0001, "adx_14": 38.0},
                    "structure": {
                        "structure_events": [
                            {"kind": "CHOCH_UP", "close_confirmed": True, "index": 100},
                            {"kind": "BOS_DOWN", "close_confirmed": True, "index": 194},
                        ]
                    },
                },
                {
                    "granularity": "H1",
                    "regime": "TREND_DOWN",
                    "indicators": {"pip_size": 0.0001, "adx_14": 52.0},
                    "structure": {
                        "structure_events": [
                            {"kind": "BOS_UP", "close_confirmed": True, "index": 109},
                            {"kind": "BOS_DOWN", "close_confirmed": True, "index": 153},
                        ]
                    },
                },
                {
                    "granularity": "H4",
                    "regime": "TREND_DOWN",
                    "indicators": {"pip_size": 0.0001, "adx_14": 35.0},
                    "structure": {"structure_events": []},
                },
            ],
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1000,
            pattern_signals=[_Sig("UP", 90.0, 1.0, "stale reversal should not dominate")],
            projection_signals=[],
            correlation_signals=[],
            paths=[],
        )

        self.assertNotEqual(forecast.direction, "UP")
        self.assertLess(forecast.up_score, 90.0)

    def test_countertrend_score_gap_damps_to_unclear_without_structure(self) -> None:
        pair_chart = {
            "confluence": {
                "score_balance": "LONG_LEAN",
                "score_gap": 0.28,
                "tf_agreement_score": 1.0 / 3.0,
                "dominant_regime": "TREND_UP",
            },
            "views": [
                {
                    "granularity": "H1",
                    "regime": "TREND_UP",
                    "indicators": {"pip_size": 0.0001, "adx_14": 28.0},
                    "structure": {"structure_events": []},
                },
                {
                    "granularity": "H4",
                    "regime": "TREND_UP",
                    "indicators": {"pip_size": 0.0001, "adx_14": 30.0},
                    "structure": {"structure_events": []},
                },
            ],
        }

        forecast = synthesize_forecast(
            pair="AUD_JPY",
            pair_chart=pair_chart,
            current_price=113.40,
            pattern_signals=[_Sig("DOWN", 45.0, 1.0, "unconfirmed top fade")],
            projection_signals=[],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "UNCLEAR")
        self.assertEqual(forecast.up_score, 0.0)
        self.assertEqual(forecast.down_score, 0.0)
        self.assertIn("countertrend DOWN damped", forecast.rationale_summary)

    def test_countertrend_confirmation_accepts_latest_h1_reversal(self) -> None:
        pair_chart = {
            "confluence": {
                "score_balance": "SHORT_LEAN",
                "score_gap": -0.75,
                "tf_agreement_score": 1.0,
                "dominant_regime": "TREND_DOWN",
            },
            "views": [
                {
                    "granularity": "H1",
                    "regime": "TREND_DOWN",
                    "indicators": {"pip_size": 0.0001, "adx_14": 52.0},
                    "structure": {
                        "structure_events": [
                            {"kind": "BOS_DOWN", "close_confirmed": True, "index": 153},
                            {"kind": "CHOCH_UP", "close_confirmed": True, "index": 168},
                        ]
                    },
                },
                {
                    "granularity": "H4",
                    "regime": "TREND_DOWN",
                    "indicators": {"pip_size": 0.0001, "adx_14": 35.0},
                    "structure": {"structure_events": []},
                },
            ],
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1000,
            pattern_signals=[_Sig("UP", 90.0, 1.0, "fresh H1 reversal")],
            projection_signals=[],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "UP")
        self.assertIn("countertrend UP allowed", forecast.rationale_summary)

    def test_score_momentum_does_not_confirm_early_turn_without_structure(self) -> None:
        pair_chart = {
            "confluence": {
                "score_balance": "SHORT_LEAN",
                "score_gap": -0.30,
                "tf_agreement_score": 1.0,
                "dominant_regime": "TREND_DOWN",
                "price_percentile_24h": 0.04,
                "price_percentile_7d": 0.06,
                "score_momentum": {
                    "direction": "UP",
                    "elapsed_min": 60.0,
                    "long_score_delta": 0.24,
                    "short_score_delta": -0.26,
                    "score_gap_delta": 0.40,
                    "score_gap_slope_per_hour": 0.40,
                    "previous_score_gap": -0.70,
                    "current_score_gap": -0.30,
                },
            },
            "views": [
                {
                    "granularity": "H1",
                    "regime": "TREND_DOWN",
                    "indicators": {"pip_size": 0.0001, "adx_14": 32.0},
                    "structure": {"structure_events": []},
                },
                {
                    "granularity": "H4",
                    "regime": "TREND_DOWN",
                    "indicators": {"pip_size": 0.0001, "adx_14": 31.0},
                    "structure": {"structure_events": []},
                },
            ],
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1000,
            pattern_signals=[_Sig("UP", 20.0, 1.0, "fresh lower sweep")],
            projection_signals=[],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "UNCLEAR")
        self.assertEqual(forecast.up_score, 0.0)
        self.assertEqual(forecast.down_score, 0.0)
        self.assertIn("market location", " ".join(forecast.drivers_for))
        self.assertNotIn("score momentum", " ".join(forecast.drivers_for))
        self.assertNotIn("score momentum", forecast.rationale_summary)
        self.assertIn("lagging indicator bias requires M15/H1 close-confirmed reversal", forecast.rationale_summary)

    def test_technical_family_scores_do_not_create_forecast_before_structure(self) -> None:
        pair_chart = {
            "confluence": {
                "score_balance": "TIED",
                "score_gap": 0.0,
                "dominant_regime": "TREND_UP",
            },
            "views": [
                {
                    "granularity": "M15",
                    "regime": "TREND_UP",
                    "indicators": {"pip_size": 0.0001},
                    "family_scores": {
                        "trend_score": 0.82,
                        "mean_rev_score": 0.05,
                        "breakout_score": 0.10,
                        "disagreement": 0.0,
                    },
                },
                {
                    "granularity": "H1",
                    "regime": "TREND_UP",
                    "indicators": {"pip_size": 0.0001},
                    "family_scores": {
                        "trend_score": 0.74,
                        "mean_rev_score": -0.10,
                        "breakout_score": 0.05,
                        "disagreement": 0.0,
                    },
                },
            ],
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "UNCLEAR")
        self.assertEqual(forecast.up_score, 0.0)
        self.assertEqual(forecast.down_score, 0.0)
        self.assertNotIn("technical trend family", " ".join(forecast.drivers_for))

    def test_stable_range_phase_forecasts_range(self) -> None:
        pair_chart = {
            "confluence": {
                "score_balance": "TIED",
                "score_gap": 0.0,
                "dominant_regime": "UNCLEAR",
            },
            "views": [
                {
                    "granularity": "M15",
                    "regime": "RANGE",
                    "regime_reading": {"state": "RANGE", "confidence": 0.9, "atr_percentile": 30.0},
                    "indicators": {
                        "pip_size": 0.0001,
                        "adx_14": 14.0,
                        "choppiness_14": 66.0,
                        "close": 1.1050,
                        "donchian_low": 1.1000,
                        "donchian_high": 1.1100,
                    },
                },
                {
                    "granularity": "H1",
                    "regime": "RANGE",
                    "regime_reading": {"state": "RANGE", "confidence": 0.9, "atr_percentile": 35.0},
                    "indicators": {
                        "pip_size": 0.0001,
                        "adx_14": 16.0,
                        "choppiness_14": 64.0,
                        "close": 1.1050,
                        "donchian_low": 1.0990,
                        "donchian_high": 1.1110,
                    },
                },
            ],
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1050,
            pattern_signals=[],
            projection_signals=[],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "RANGE")
        self.assertGreater(forecast.range_score, forecast.up_score)
        self.assertIn("inside stable range", " ".join(forecast.drivers_for))
        self.assertEqual(forecast.range_low_price, 1.1000)
        self.assertEqual(forecast.range_high_price, 1.1100)
        self.assertAlmostEqual(forecast.range_width_pips or 0.0, 100.0)

    def test_weak_calibrated_direction_inside_range_falls_back_to_range(self) -> None:
        pair_chart = {
            "confluence": {
                "score_balance": "TIED",
                "score_gap": 0.0,
                "dominant_regime": "UNCLEAR",
            },
            "views": [
                {
                    "granularity": "M15",
                    "regime": "RANGE",
                    "regime_reading": {"state": "RANGE", "confidence": 0.9},
                    "indicators": {
                        "pip_size": 0.0001,
                        "adx_14": 14.0,
                        "choppiness_14": 66.0,
                        "close": 1.1050,
                        "donchian_low": 1.1000,
                        "donchian_high": 1.1100,
                    },
                },
                {
                    "granularity": "H1",
                    "regime": "RANGE",
                    "regime_reading": {"state": "RANGE", "confidence": 0.9},
                    "indicators": {
                        "pip_size": 0.0001,
                        "adx_14": 16.0,
                        "choppiness_14": 64.0,
                        "close": 1.1050,
                        "donchian_low": 1.0990,
                        "donchian_high": 1.1110,
                    },
                },
            ],
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1050,
            pattern_signals=[],
            projection_signals=[_Sig("DOWN", 120.0, 1.0, "raw downside pressure")],
            correlation_signals=[],
            paths=[],
            hit_rates={
                "directional_forecast_down": {
                    "EUR_USD:RANGE": {"hit_rate": 0.0, "samples": 30},
                },
            },
            regime="RANGE",
        )

        self.assertEqual(forecast.direction, "RANGE")
        self.assertEqual(forecast.range_low_price, 1.1000)
        self.assertEqual(forecast.range_high_price, 1.1100)
        self.assertIn("weak calibrated DOWN forecast inside IN_RANGE", forecast.rationale_summary)
        self.assertIn("raw downside pressure", " ".join(forecast.drivers_against))

    def test_strong_direction_inside_range_stays_directional(self) -> None:
        pair_chart = {
            "views": [
                {
                    "granularity": "M15",
                    "regime": "RANGE",
                    "regime_reading": {"state": "RANGE", "confidence": 0.9},
                    "indicators": {
                        "pip_size": 0.0001,
                        "adx_14": 14.0,
                        "choppiness_14": 66.0,
                        "close": 1.1050,
                        "donchian_low": 1.1000,
                        "donchian_high": 1.1100,
                    },
                    "structure": {
                        "swings": [
                            {"side": "HIGH", "price": 1.1110},
                            {"side": "LOW", "price": 1.0990},
                        ],
                    },
                }
            ],
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1050,
            pattern_signals=[],
            projection_signals=[_Sig("DOWN", 120.0, 1.0, "audited downside pressure")],
            correlation_signals=[],
            paths=[],
            hit_rates={
                "directional_forecast_down": {
                    "EUR_USD:RANGE": {"hit_rate": 0.8, "samples": 30},
                },
            },
            regime="RANGE",
        )

        self.assertEqual(forecast.direction, "DOWN")
        self.assertGreaterEqual(forecast.confidence, 0.55)
        self.assertIn("audited downside pressure", " ".join(forecast.drivers_for))

    def test_contested_direction_inside_range_forecasts_range(self) -> None:
        pair_chart = {
            "confluence": {
                "score_balance": "TIED",
                "score_gap": 0.0,
                "dominant_regime": "RANGE",
            },
            "views": [
                {
                    "granularity": "M5",
                    "regime": "UNCLEAR",
                    "regime_reading": {"state": "TREND_WEAK", "confidence": 0.6},
                    "indicators": {
                        "pip_size": 0.0001,
                        "adx_14": 18.0,
                        "choppiness_14": 56.0,
                        "close": 1.1050,
                        "donchian_low": 1.1000,
                        "donchian_high": 1.1100,
                    },
                },
                {
                    "granularity": "M15",
                    "regime": "UNCLEAR",
                    "regime_reading": {"state": "TREND_WEAK", "confidence": 0.6},
                    "indicators": {
                        "pip_size": 0.0001,
                        "adx_14": 19.0,
                        "choppiness_14": 55.0,
                        "close": 1.1050,
                        "donchian_low": 1.1000,
                        "donchian_high": 1.1100,
                    },
                },
                {
                    "granularity": "H1",
                    "regime": "UNCLEAR",
                    "regime_reading": {"state": "TRANSITION", "confidence": 0.6},
                    "indicators": {
                        "pip_size": 0.0001,
                        "adx_14": 20.0,
                        "choppiness_14": 50.0,
                        "close": 1.1050,
                        "donchian_low": 1.0990,
                        "donchian_high": 1.1110,
                    },
                },
            ],
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1050,
            pattern_signals=[_Sig("UP", 55.0, 1.0, "up evidence")],
            projection_signals=[_Sig("DOWN", 53.0, 1.0, "down evidence")],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "RANGE")
        self.assertGreaterEqual(forecast.confidence, 0.55)
        self.assertIn("contested direction inside RANGE_FORMING", forecast.rationale_summary)
        self.assertIn("range forming", " ".join(forecast.drivers_for))

    def test_contested_direction_without_range_evidence_stays_unclear(self) -> None:
        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart={"views": [{"granularity": "M15", "indicators": {"pip_size": 0.0001}}]},
            current_price=1.1050,
            pattern_signals=[_Sig("UP", 55.0, 1.0, "up evidence")],
            projection_signals=[_Sig("DOWN", 53.0, 1.0, "down evidence")],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "UNCLEAR")
        self.assertIn("contested", forecast.rationale_summary)

    def test_lower_half_range_location_contests_downside_chase_forecast(self) -> None:
        pair_chart = {
            "confluence": {
                "score_balance": "SHORT_LEAN",
                "score_gap": -0.4,
                "dominant_regime": "RANGE",
            },
            "views": [
                {
                    "granularity": "M5",
                    "regime": "RANGE",
                    "regime_reading": {"state": "RANGE", "confidence": 0.9},
                    "indicators": {
                        "pip_size": 0.0001,
                        "adx_14": 14.0,
                        "choppiness_14": 66.0,
                        "close": 1.1008,
                        "donchian_low": 1.1000,
                        "donchian_high": 1.1100,
                    },
                },
                {
                    "granularity": "M15",
                    "regime": "RANGE",
                    "regime_reading": {"state": "RANGE", "confidence": 0.9},
                    "indicators": {
                        "pip_size": 0.0001,
                        "adx_14": 16.0,
                        "choppiness_14": 64.0,
                        "close": 1.1010,
                        "donchian_low": 1.1000,
                        "donchian_high": 1.1120,
                    },
                },
                {
                    "granularity": "H1",
                    "regime": "RANGE",
                    "regime_reading": {"state": "RANGE", "confidence": 0.9},
                    "indicators": {
                        "pip_size": 0.0001,
                        "adx_14": 17.0,
                        "choppiness_14": 63.0,
                        "close": 1.1010,
                        "donchian_low": 1.0990,
                        "donchian_high": 1.1130,
                    },
                },
            ],
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1010,
            pattern_signals=[],
            projection_signals=[_Sig("DOWN", 35.0, 1.0, "downside pressure")],
            correlation_signals=[],
            paths=[],
        )

        self.assertNotEqual(forecast.direction, "DOWN")
        self.assertGreater(forecast.up_score, 0.0)
        self.assertIn("contested", forecast.rationale_summary)

    def test_breakout_pending_range_phase_blocks_range_rotation(self) -> None:
        pair_chart = {
            "confluence": {
                "score_balance": "TIED",
                "score_gap": 0.0,
                "dominant_regime": "RANGE",
                "range_24h_sigma_multiple": 0.7,
            },
            "views": [
                {
                    "granularity": "M15",
                    "regime": "RANGE",
                    "regime_reading": {"state": "BREAKOUT_PENDING", "confidence": 0.8, "atr_percentile": 12.0},
                    "indicators": {
                        "pip_size": 0.0001,
                        "adx_14": 22.0,
                        "choppiness_14": 50.0,
                        "bb_squeeze": 1,
                        "bb_width_percentile_100": 0.12,
                        "close": 1.1050,
                        "donchian_low": 1.1000,
                        "donchian_high": 1.1100,
                    },
                },
                {
                    "granularity": "H1",
                    "regime": "RANGE",
                    "regime_reading": {"state": "BREAKOUT_PENDING", "confidence": 0.8, "atr_percentile": 14.0},
                    "indicators": {
                        "pip_size": 0.0001,
                        "adx_14": 23.0,
                        "choppiness_14": 52.0,
                        "bb_squeeze": 1,
                        "bb_width_percentile_100": 0.15,
                        "close": 1.1050,
                        "donchian_low": 1.0990,
                        "donchian_high": 1.1110,
                    },
                },
            ],
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1050,
            pattern_signals=[],
            projection_signals=[],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "UNCLEAR")
        self.assertIn("breakout pending blocks RANGE rotation", forecast.rationale_summary)
        self.assertIn("BREAKOUT_PENDING", " ".join(forecast.drivers_for))

    def test_confirmed_range_breakout_can_forecast_direction(self) -> None:
        pair_chart = {
            "confluence": {
                "score_balance": "TIED",
                "score_gap": 0.0,
                "dominant_regime": "TRANSITION",
            },
            "views": [
                {
                    "granularity": "M15",
                    "regime": "RANGE",
                    "regime_reading": {"state": "BREAKOUT_PENDING", "confidence": 0.8, "atr_percentile": 12.0},
                    "indicators": {
                        "pip_size": 0.0001,
                        "close": 1.1101,
                        "donchian_low": 1.1000,
                        "donchian_high": 1.1100,
                        "linreg_slope_20": 0.4,
                    },
                    "family_scores": {"trend_score": 0.55, "breakout_score": 0.75, "disagreement": 0.1},
                },
                {
                    "granularity": "H1",
                    "regime": "RANGE",
                    "regime_reading": {"state": "BREAKOUT_PENDING", "confidence": 0.8, "atr_percentile": 15.0},
                    "indicators": {
                        "pip_size": 0.0001,
                        "close": 1.1101,
                        "donchian_low": 1.0990,
                        "donchian_high": 1.1100,
                        "linreg_slope_20": 0.3,
                    },
                    "family_scores": {"trend_score": 0.50, "breakout_score": 0.70, "disagreement": 0.1},
                },
            ],
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1101,
            pattern_signals=[],
            projection_signals=[],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "UP")
        self.assertIn("range breakout confirmed UP", " ".join(forecast.drivers_for))

    def test_forecast_calibration_prefers_direction_specific_history(self) -> None:
        hit_rates = {
            "directional_forecast": {
                "EUR_USD:TREND": {"hit_rate": 1.0, "samples": 100},
            },
            "directional_forecast_up": {
                "EUR_USD:TREND": {"hit_rate": 0.0, "samples": 100},
            },
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart={"views": [{"granularity": "M15", "indicators": {"pip_size": 0.0001}}]},
            current_price=1.1000,
            pattern_signals=[_Sig("UP", 60.0, 1.0, "up setup")],
            projection_signals=[],
            correlation_signals=[],
            paths=[],
            hit_rates=hit_rates,
            regime="TREND",
        )

        self.assertEqual(forecast.direction, "UP")
        self.assertLess(forecast.calibration_multiplier, 0.5)
        self.assertLess(forecast.confidence, 0.5)

    def test_projection_signal_calibration_happens_before_forecast_selection(self) -> None:
        hit_rates = {
            "liquidity_sweep_low_up": {
                "EUR_USD:TREND": {"hit_rate": 0.46, "samples": 100},
            },
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart={"views": [{"granularity": "M15", "indicators": {"pip_size": 0.0001}}]},
            current_price=1.1000,
            pattern_signals=[_Sig("DOWN", 60.0, 1.0, "down setup")],
            projection_signals=[
                _NamedSig(
                    "liquidity_sweep_low",
                    "UP",
                    45.0,
                    1.0,
                    "weak historical sweep-low UP detector",
                )
            ],
            correlation_signals=[],
            paths=[],
            hit_rates=hit_rates,
            regime="TREND",
        )

        self.assertEqual(forecast.direction, "DOWN")
        self.assertGreater(forecast.down_score, forecast.up_score)
        self.assertIn("[cal×", " ".join(forecast.drivers_against))

    def test_audited_subrandom_projection_signal_cannot_create_forecast_side(self) -> None:
        hit_rates = {
            "news_theme_followthrough_up": {
                "_all_pairs:_all_regimes": {"hit_rate": 0.39, "samples": 1000},
            },
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart={"views": [{"granularity": "M15", "indicators": {"pip_size": 0.0001}}]},
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[
                _NamedSig(
                    "news_theme_followthrough",
                    "UP",
                    100.0,
                    1.0,
                    "bad audited news follow-through",
                )
            ],
            correlation_signals=[],
            paths=[],
            hit_rates=hit_rates,
            regime="TREND",
        )

        self.assertEqual(forecast.direction, "UNCLEAR")
        self.assertEqual(forecast.up_score, 0.0)
        self.assertEqual(forecast.down_score, 0.0)

    def test_thin_bad_projection_sample_does_not_hard_exclude_signal(self) -> None:
        hit_rates = {
            "news_theme_followthrough_up": {
                "EUR_USD:TREND": {"hit_rate": 0.0, "samples": 2},
            },
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart={"views": [{"granularity": "M15", "indicators": {"pip_size": 0.0001}}]},
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[
                _NamedSig(
                    "news_theme_followthrough",
                    "UP",
                    100.0,
                    1.0,
                    "thin news sample",
                )
            ],
            correlation_signals=[],
            paths=[],
            hit_rates=hit_rates,
            regime="TREND",
        )

        self.assertEqual(forecast.direction, "UP")
        self.assertGreater(forecast.up_score, 0.0)

    def test_d_h4_anchor_extends_directional_forecast_horizon(self) -> None:
        pair_chart = {
            "views": [
                {
                    "granularity": "M5",
                    "indicators": {"pip_size": 0.0001, "atr_pips": 2.0},
                    "structure": {
                        "swings": [
                            {"side": "HIGH", "price": 1.1080},
                            {"side": "LOW", "price": 1.0975},
                        ]
                    },
                },
                {
                    "granularity": "H4",
                    "regime": "TREND_UP",
                    "indicators": {"pip_size": 0.0001, "adx_14": 34.0},
                    "family_scores": {"trend_score": 0.72, "disagreement": 0.0},
                },
                {
                    "granularity": "D",
                    "regime": "TREND_UP",
                    "indicators": {"pip_size": 0.0001, "adx_14": 31.0},
                    "family_scores": {"trend_score": 0.68, "disagreement": 0.0},
                },
            ]
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[_Sig("UP", 35.0, 1.0, "daily continuation")],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "UP")
        self.assertEqual(forecast.horizon_min, FORECAST_D_ANCHOR_HORIZON_MIN)
        self.assertIn("D/H4 UP anchor", forecast.rationale_summary)
        self.assertIn("D/H4 UP anchor", " ".join(forecast.drivers_for))

    def test_h1_m30_anchor_uses_operating_swing_horizon(self) -> None:
        pair_chart = {
            "views": [
                {
                    "granularity": "M5",
                    "indicators": {"pip_size": 0.0001, "atr_pips": 2.0},
                    "structure": {
                        "swings": [
                            {"side": "HIGH", "price": 1.1025},
                            {"side": "LOW", "price": 1.0940},
                        ]
                    },
                },
                {
                    "granularity": "M30",
                    "regime": "TREND_DOWN",
                    "indicators": {"pip_size": 0.0001},
                    "family_scores": {"trend_score": -0.70, "disagreement": 0.0},
                },
                {
                    "granularity": "H1",
                    "regime": "TREND_DOWN",
                    "indicators": {"pip_size": 0.0001},
                    "family_scores": {"trend_score": -0.74, "disagreement": 0.0},
                },
            ]
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[_Sig("DOWN", 35.0, 1.0, "operating swing breakdown")],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "DOWN")
        self.assertEqual(forecast.horizon_min, FORECAST_OPERATING_SWING_HORIZON_MIN)
        self.assertIn("H1 operating-swing DOWN", forecast.rationale_summary)
        self.assertIn("H1 operating-swing DOWN", " ".join(forecast.drivers_for))

    def test_micro_only_forecast_keeps_execution_horizon(self) -> None:
        pair_chart = {
            "views": [
                {
                    "granularity": "M5",
                    "regime": "TREND_UP",
                    "indicators": {"pip_size": 0.0001, "atr_pips": 2.0, "adx_14": 31.0},
                    "family_scores": {"trend_score": 0.80, "disagreement": 0.0},
                    "structure": {
                        "swings": [
                            {"side": "HIGH", "price": 1.1040},
                            {"side": "LOW", "price": 1.0975},
                        ]
                    },
                }
            ]
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[_Sig("UP", 35.0, 1.0, "M5 impulse")],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "UP")
        self.assertEqual(forecast.horizon_min, FORECAST_EXECUTION_HORIZON_MIN)
        self.assertNotIn("anchor horizon", forecast.rationale_summary)


class RegimeFamilyForecastGateTest(unittest.TestCase):
    @staticmethod
    def _chart(trend_score: float) -> dict:
        return {
            "pair": "EUR_USD",
            "session": {"current_tag": "LONDON"},
            "confluence": {
                "dominant_regime": "TREND_STRONG",
                "price_percentile_24h": 0.5,
            },
            "views": [
                {
                    "granularity": "M15",
                    "regime_reading": {
                        "state": "TREND_STRONG",
                        "atr_percentile": 50.0,
                    },
                    "family_scores": {
                        "trend_score": trend_score,
                        "mean_rev_score": 0.0,
                        "breakout_score": 0.0,
                        "disagreement": 0.0,
                    },
                    "indicators": {"pip_size": 0.0001, "atr_pips": 2.0},
                    "structure": {
                        "structure_events": [
                            {
                                "kind": "BOS_UP",
                                "close_confirmed": True,
                                "index": 1,
                                "timestamp_utc": "2026-07-14T00:00:00Z",
                            }
                        ]
                    },
                },
                {
                    "granularity": "M5",
                    "regime_reading": {
                        "state": "TREND_STRONG",
                        "atr_percentile": 50.0,
                    },
                    "family_scores": {
                        "trend_score": trend_score,
                        "mean_rev_score": 0.0,
                        "breakout_score": 0.0,
                        "disagreement": 0.0,
                    },
                    "indicators": {"pip_size": 0.0001, "atr_pips": 2.0},
                },
            ],
        }

    def test_aligned_receipt_is_not_reported_as_an_extra_support_driver(self) -> None:
        from quant_rabbit.strategy.directional_forecaster import (
            _regime_family_adjustment,
        )
        from quant_rabbit.strategy.forecast_technical_context import (
            build_forecast_technical_context,
        )

        context = build_forecast_technical_context(
            self._chart(1.0),
            pair="EUR_USD",
            current_price=1.1,
            spread_pips=0.2,
        )
        score, reason = _regime_family_adjustment(context, "UP", 30.0, 10.0)

        self.assertEqual(score, 30.0)
        self.assertIsNone(reason)

    def test_opposite_receipt_vetoes_to_unclear_without_sort_direction_flip(self) -> None:
        with patch(
            "quant_rabbit.strategy.directional_forecaster."
            "_technical_candle_integrity_forecast_blockers",
            return_value=(),
        ):
            forecast = _synthesize_forecast(
                pair="EUR_USD",
                pair_chart=self._chart(-1.0),
                current_price=1.1000,
                pattern_signals=[],
                projection_signals=[_Sig("UP", 100.0, 1.0, "strong up detector")],
                correlation_signals=[],
                paths=[],
                spread_pips=0.2,
                entry_bid=1.09999,
                entry_ask=1.10001,
            )

        self.assertEqual(forecast.direction, "UNCLEAR")
        self.assertEqual(forecast.confidence, 0.0)
        self.assertEqual(forecast.horizon_min, 0)
        self.assertIn("contradiction vetoed UP", forecast.rationale_summary)
        self.assertTrue(
            any("CONTRADICTS_FORECAST" in reason for reason in forecast.drivers_against)
        )
        shadow = forecast.regime_family_contradiction_shadow
        self.assertEqual(shadow["entry_quote_source"], "EXPLICIT_BID_ASK")
        self.assertEqual(shadow["entry_bid"], 1.09999)
        self.assertEqual(shadow["entry_ask"], 1.10001)
        self.assertEqual(shadow["detector_arm"]["direction"], "UP")
        self.assertEqual(shadow["family_arm"]["direction"], "DOWN")
        self.assertEqual(shadow["detector_scores"]["UP"], 100.0)
        self.assertTrue(shadow["read_only"])
        self.assertFalse(shadow["live_permission"])
        self.assertFalse(shadow["sizing_permission"])
        learning = forecast.forecast_learning_v1
        self.assertEqual(learning["model_status"], "RANK_ONLY")
        self.assertEqual(learning["original_direction"], "UP")
        self.assertIn(learning["rank_direction"], {"UP", "DOWN"})
        self.assertTrue(learning["always_returns_direction"])
        self.assertEqual(
            learning["probability_semantics"],
            "ORIENTATION_RANK_NOT_TRADE_WIN_PROBABILITY",
        )
        self.assertNotEqual(
            learning["features"]["score_margin_bucket"],
            "ZERO",
        )
        self.assertNotIn(
            "regime_family_contradiction_shadow",
            forecast.to_dict(),
        )
        self.assertIn("forecast_learning_v1", forecast.to_dict())
        from quant_rabbit.strategy.intent_generator import _forecast_context_payload

        self.assertNotIn(
            "regime_family_contradiction_shadow",
            json.dumps(
                _forecast_context_payload(
                    forecast,
                    cycle_id="contradiction-cycle",
                ),
                sort_keys=True,
            ),
        )
        from quant_rabbit.strategy.forecast_persistence_tracker import record_forecast
        from quant_rabbit.strategy.regime_family_contradiction_shadow import (
            LEDGER_FILENAME,
        )

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            # The shadow ledger intentionally uses the unique minimal RFC3339
            # representation accepted by its 1-9 digit nanosecond contract.
            # A Python datetime whose microsecond ends in zero therefore seals
            # as five fractional digits (``.02343Z``), which Python 3.10's
            # ``datetime.fromisoformat`` cannot parse even though it is valid
            # RFC3339.  Keep both clocks deterministic and assert the ledger's
            # canonical bytes directly so this integration check does not
            # depend on either wall-clock precision or that parser limitation.
            emitted_at = datetime(
                2026,
                7,
                15,
                0,
                0,
                30,
                23_430,
                tzinfo=timezone.utc,
            )
            history_at = emitted_at + timedelta(seconds=10)
            with patch(
                "quant_rabbit.strategy.regime_family_contradiction_shadow."
                "_ledger_recorded_at_utc",
                return_value=emitted_at + timedelta(seconds=1),
            ):
                self.assertTrue(
                    record_forecast(
                        forecast,
                        data_root=data_root,
                        cycle_id="contradiction-cycle",
                        now=history_at,
                        shadow_emitted_at_utc=emitted_at,
                    )
                )
            history_row = json.loads(
                (data_root / "forecast_history.jsonl").read_text()
            )
            self.assertNotIn("regime_family_contradiction_shadow", history_row)
            self.assertEqual(
                history_row["timestamp_utc"],
                history_at.isoformat().replace("+00:00", "Z"),
            )
            shadow_events = [
                json.loads(line)
                for line in (data_root / LEDGER_FILENAME).read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(len(shadow_events), 1)
            self.assertEqual(
                shadow_events[0]["payload"]["emitted_at_utc"],
                "2026-07-15T00:00:30.02343Z",
            )
            self.assertFalse((data_root / "projection_ledger.jsonl").exists())

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            self.assertTrue(
                record_forecast(
                    forecast,
                    data_root=data_root,
                    cycle_id="missing-shadow-clock",
                    now=history_at,
                )
            )
            self.assertTrue((data_root / "forecast_history.jsonl").exists())
            self.assertFalse((data_root / LEDGER_FILENAME).exists())

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            self.assertTrue(
                record_forecast(
                    forecast,
                    data_root=data_root,
                    cycle_id="stale-shadow-clock",
                    now=datetime.now(timezone.utc),
                    shadow_emitted_at_utc=datetime.now(timezone.utc)
                    - timedelta(minutes=2),
                )
            )
            self.assertTrue((data_root / "forecast_history.jsonl").exists())
            self.assertFalse((data_root / LEDGER_FILENAME).exists())

    def test_bad_entry_quote_drops_only_shadow_not_the_fail_closed_veto(self) -> None:
        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=self._chart(-1.0),
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[_Sig("UP", 100.0, 1.0, "strong up detector")],
            correlation_signals=[],
            paths=[],
            spread_pips=0.2,
            entry_bid=1.09990,
            entry_ask=1.10010,
        )

        self.assertEqual(forecast.direction, "UNCLEAR")
        self.assertEqual(forecast.confidence, 0.0)
        self.assertEqual(forecast.horizon_min, 0)
        self.assertEqual(forecast.regime_family_contradiction_shadow, {})


class ContestedRangePhaseEvidenceTest(unittest.TestCase):
    """_contested_range_raw_confidence phase-evidence path (2026-06-11).

    Live funnel regression: 19/28 pairs sat in measured boxes while the
    production signal book inflated directional scores enough that
    range_score < margin, so 60+ rotation lanes per cycle died as UNCLEAR.
    The phase detector's own confidence (>= 0.5 == its documented
    minimum-evidence bar) now qualifies RANGE resolution directly.
    """

    @staticmethod
    def _compute(phase: str, phase_conf: float, *, range_score: float, margin_pair=(100.0, 92.0)):
        from quant_rabbit.strategy.directional_forecaster import (
            _RangePhase,
            _contested_range_raw_confidence,
        )

        winner, runner_up = margin_pair
        return _contested_range_raw_confidence(
            phase=_RangePhase(
                phase=phase, confidence=phase_conf, direction=None,
                rationale="test", evidence=("e1",),
            ),
            winner_score=winner,
            runner_up_score=runner_up,
            range_score=range_score,
            margin=winner - runner_up,
        )

    def test_confident_box_qualifies_even_when_range_echo_lost_to_noise(self) -> None:
        # margin 8 > range_score 5, but the box itself measured 0.66.
        conf = self._compute("IN_RANGE", 0.66, range_score=5.0)
        self.assertIsNotNone(conf)
        # (directional_uncertainty 0.92 + phase 0.66) / 2 = 0.79
        self.assertAlmostEqual(conf, 0.79, places=2)

    def test_weak_box_below_min_evidence_stays_unclear(self) -> None:
        self.assertIsNone(self._compute("RANGE_FORMING", 0.49, range_score=5.0))

    def test_breakout_pending_box_never_qualifies(self) -> None:
        self.assertIsNone(self._compute("BREAKOUT_PENDING", 1.0, range_score=5.0))

    def test_strong_range_echo_path_unchanged(self) -> None:
        # range_score 20 >= margin 8 → original blend, phase conf irrelevant.
        conf = self._compute("IN_RANGE", 0.0, range_score=20.0)
        self.assertIsNotNone(conf)
        # (0.92 + min(1, 20/92)) / 2 ≈ 0.5687
        self.assertAlmostEqual(conf, (0.92 + 20.0 / 92.0) / 2.0, places=3)


class TechnicalCandleIntegrityForecastGateTest(unittest.TestCase):
    @staticmethod
    def _chart(*, recovered: bool) -> dict[str, object]:
        integrity = _production_mba_integrity(recovered=recovered)
        timeframes = integrity["timeframes"]
        assert isinstance(timeframes, dict)
        item = timeframes["M1"]
        m5_item = timeframes["M5"]
        assert isinstance(item, dict)
        assert isinstance(m5_item, dict)
        latest_clean = item["latest_clean_timestamp_utc"]
        latest_clean_dt = datetime.fromisoformat(str(latest_clean))
        m5_latest_clean = str(m5_item["latest_clean_timestamp_utc"])
        m5_latest_clean_dt = datetime.fromisoformat(m5_latest_clean)
        m1_recent = (
            [
                {
                    "t": (
                        latest_clean_dt
                        - timedelta(
                            minutes=TECHNICAL_CANDLE_INDICATOR_WARMUP_MIN_CLEAN_COUNT - 1 - index
                        )
                    ).isoformat()
                }
                for index in range(TECHNICAL_CANDLE_INDICATOR_WARMUP_MIN_CLEAN_COUNT)
            ]
            if recovered
            else []
        )
        m5_recent = [
            {
                "t": (
                    m5_latest_clean_dt
                    - timedelta(
                        minutes=5
                        * (TECHNICAL_CANDLE_INDICATOR_WARMUP_MIN_CLEAN_COUNT - 1 - index)
                    )
                ).isoformat()
            }
            for index in range(TECHNICAL_CANDLE_INDICATOR_WARMUP_MIN_CLEAN_COUNT)
        ]
        return {
            "pair": "EUR_USD",
            "generated_at_utc": "2026-07-14T00:45:00Z",
            "technical_candle_integrity": integrity,
            "views": [
                {
                    "granularity": "M1",
                    "indicators": {
                        "candles_count": item["recent_clean_tail_count"],
                        "pip_size": 0.0001,
                        "atr_pips": 2.0,
                        "donchian_high": 1.1010,
                        "donchian_low": 1.0990,
                    },
                    "recent_candles": m1_recent,
                    "candle_integrity": item,
                },
                {
                    "granularity": "M5",
                    "indicators": {
                        "candles_count": 41,
                        "pip_size": 0.0001,
                        "atr_pips": 4.0,
                        "donchian_high": 1.1020,
                        "donchian_low": 1.0980,
                    },
                    "recent_candles": m5_recent,
                    "candle_integrity": m5_item,
                },
            ],
        }

    @classmethod
    def _m15_recovery_chart(cls) -> dict[str, object]:
        chart = cls._chart(recovered=True)
        aggregate = chart["technical_candle_integrity"]
        assert isinstance(aggregate, dict)
        timeframes = aggregate["timeframes"]
        assert isinstance(timeframes, dict)
        views = chart["views"]
        assert isinstance(views, list)
        view_by_tf = {
            str(view["granularity"]): view
            for view in views
            if isinstance(view, dict)
        }

        # Preserve a clean producer-owned M5 receipt as the M15 template before
        # turning both fast timeframes into historical-quarantine warmup state.
        m15_item = copy.deepcopy(timeframes["M5"])
        for tf, latest, quarantine_at, cadence in (
            ("M1", "2026-07-14T00:43:00+00:00", "2026-07-14T00:13:00+00:00", 1),
            ("M5", "2026-07-14T00:40:00+00:00", "2026-07-14T00:25:00+00:00", 5),
        ):
            item = timeframes[tf]
            assert isinstance(item, dict)
            cap = item["spread_cap_pips"]
            item.update(
                {
                    "evaluation_status": "BLOCKED",
                    "forecast_blocking": True,
                    "codes": [
                        TECHNICAL_CANDLE_SPREAD_CONTAMINATED,
                        TECHNICAL_CANDLE_PROVENANCE_INVALID,
                    ],
                    "blocking_codes": [TECHNICAL_CANDLE_PROVENANCE_INVALID],
                    "recent_clean_tail_count": 2,
                    "recent_clean_coverage_complete": False,
                    "recent_tail_state": "CLEAN",
                    "recent_tail_contaminated": False,
                    "recent_tail_invalid": False,
                    "contaminated_count": 1,
                    "malformed_count": 0,
                    "quarantined_count": 1,
                    "latest_complete_timestamp_utc": latest,
                    "latest_clean_timestamp_utc": latest,
                    "quarantine_details": [
                        {
                            "timestamp_utc": quarantine_at,
                            "code": TECHNICAL_CANDLE_SPREAD_CONTAMINATED,
                            "max_spread_pips": float(cap) + 1.0,
                            "spread_cap_pips": cap,
                        }
                    ],
                }
            )
            if tf == "M5":
                item.update(
                    {
                        "requested_count": 42,
                        "raw_entry_count": 42,
                        "complete_entry_count": 42,
                        "clean_count": 41,
                    }
                )
            window = item["quarantine_details_window"]
            assert isinstance(window, dict)
            window.update(
                {
                    "start_index": 0,
                    "end_index_exclusive": 1,
                    "total_count": 1,
                    "total_code_counts": {
                        TECHNICAL_CANDLE_SPREAD_CONTAMINATED: 1,
                        TECHNICAL_CANDLE_PROVENANCE_INVALID: 0,
                    },
                    "published_code_counts": {
                        TECHNICAL_CANDLE_SPREAD_CONTAMINATED: 1,
                        TECHNICAL_CANDLE_PROVENANCE_INVALID: 0,
                    },
                    "omitted_code_counts": {
                        TECHNICAL_CANDLE_SPREAD_CONTAMINATED: 0,
                        TECHNICAL_CANDLE_PROVENANCE_INVALID: 0,
                    },
                    "total_timestamped_count": 1,
                    "published_timestamped_count": 1,
                    "omitted_timestamped_count": 0,
                    "latest_timestamp_utc": quarantine_at,
                }
            )
            latest_dt = datetime.fromisoformat(latest)
            view = view_by_tf[tf]
            assert isinstance(view, dict)
            view["indicators"] = {
                "valid": False,
                "candles_count": 2,
                "pip_size": 0.0001,
                "atr_pips": None,
            }
            view["recent_candles"] = [
                {"t": (latest_dt - timedelta(minutes=cadence)).isoformat()},
                {"t": latest_dt.isoformat()},
            ]

        m15_latest = datetime(2026, 7, 14, 0, 30, tzinfo=timezone.utc)
        m15_item.update(
            {
                "granularity": "M15",
                "payload_granularity": "M15",
                "spread_evaluation_mode": TECHNICAL_CANDLE_SPREAD_PROVENANCE_ONLY_MODE,
                "indicator_warmup_min_clean_count": 1,
                "latest_complete_timestamp_utc": m15_latest.isoformat(),
                "latest_clean_timestamp_utc": m15_latest.isoformat(),
            }
        )
        m15_view = {
            "granularity": "M15",
            "regime": "TREND_UP",
            "regime_reading": {"state": "TREND_UP", "confidence": 0.8},
            "family_scores": {"trend_score": 0.7},
            "indicators": {
                "valid": True,
                "candles_count": 41,
                "pip_size": 0.0001,
                "atr_pips": 5.0,
                "adx_14": 30.0,
            },
            "structure": {
                "swings": [
                    {"side": "HIGH", "price": 1.1020},
                    {"side": "LOW", "price": 1.0980},
                ]
            },
            "recent_candles": [
                {
                    "t": (
                        m15_latest
                        - timedelta(minutes=15 * (29 - index))
                    ).isoformat()
                }
                for index in range(30)
            ],
            "candle_integrity": m15_item,
        }
        timeframes["M15"] = m15_item
        views.append(m15_view)
        aggregate.update(
            {
                "evaluation_status": "BLOCKED",
                "forecast_blocking": True,
                "codes": [
                    TECHNICAL_CANDLE_SPREAD_CONTAMINATED,
                    TECHNICAL_CANDLE_PROVENANCE_INVALID,
                ],
                "blocking_codes": [TECHNICAL_CANDLE_PROVENANCE_INVALID],
                "requested_timeframes": ["M1", "M5", "M15"],
                "evaluated_timeframe_count": 3,
            }
        )
        return chart

    @staticmethod
    def _production_forecast(
        chart: dict[str, object],
        *,
        pair: str = "EUR_USD",
        now_utc: object = datetime(2026, 7, 14, 0, 45, tzinfo=timezone.utc),
    ):
        return synthesize_forecast(
            pair=pair,
            pair_chart=chart,
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[_Sig("UP", 100.0, 1.0, "strong up detector")],
            correlation_signals=[],
            paths=[],
            now_utc=now_utc,
            require_technical_candle_integrity=True,
        )

    def test_recent_rollover_contamination_forces_unclear_zero_confidence(self) -> None:
        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=self._chart(recovered=False),
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[_Sig("UP", 100.0, 1.0, "strong up detector")],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "UNCLEAR")
        self.assertEqual(forecast.confidence, 0.0)
        self.assertEqual(forecast.raw_confidence, 0.0)
        self.assertIn(TECHNICAL_CANDLE_SPREAD_CONTAMINATED, forecast.drivers_against)
        self.assertIn(TECHNICAL_CANDLE_SPREAD_CONTAMINATED, forecast.rationale_summary)

    def test_clean_tail_reenables_directional_forecast_after_quarantine(self) -> None:
        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=self._chart(recovered=True),
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[_Sig("UP", 100.0, 1.0, "strong up detector")],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "UP")
        self.assertGreater(forecast.confidence, 0.0)
        self.assertNotIn(TECHNICAL_CANDLE_SPREAD_CONTAMINATED, forecast.drivers_against)

    def test_m15_recovery_builds_content_addressed_non_live_micro_candidate(self) -> None:
        chart = self._m15_recovery_chart()
        now = datetime(2026, 7, 14, 0, 45, tzinfo=timezone.utc)
        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=chart,
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[
                _Sig("UP", 100.0, 1.0, "M15 up detector", timeframe="M15")
            ],
            correlation_signals=[],
            paths=[],
            spread_pips=0.8,
            now_utc=now,
            require_technical_candle_integrity=True,
        )

        self.assertEqual(forecast.direction, "UP")
        receipt = forecast.m15_recovery_receipt
        self.assertEqual(receipt["contract"], M15_RECOVERY_MICRO_CONTRACT)
        self.assertEqual(receipt["geometry_source"]["timeframe"], "M15")
        self.assertEqual(receipt["sizing"]["max_units"], M15_RECOVERY_MICRO_MAX_UNITS)
        self.assertEqual(len(receipt["m15_plus_scoring_input_sha256"]), 64)
        self.assertIs(receipt["live_permission"], False)
        self.assertIs(receipt["manual_position_mutation_allowed"], False)
        self.assertTrue(
            validate_m15_recovery_micro_receipt(
                receipt,
                pair_chart=chart,
                expected_pair="EUR_USD",
                now_utc=now,
                current_spread_pips=0.8,
            )
        )
        tampered = copy.deepcopy(receipt)
        tampered["sizing"]["max_units"] = 1000
        self.assertFalse(
            validate_m15_recovery_micro_receipt(
                tampered,
                pair_chart=chart,
                expected_pair="EUR_USD",
                now_utc=now,
                current_spread_pips=0.8,
            )
        )
        changed_scoring_chart = copy.deepcopy(chart)
        changed_views = changed_scoring_chart["views"]
        assert isinstance(changed_views, list)
        changed_m15 = next(
            view
            for view in changed_views
            if isinstance(view, dict) and view.get("granularity") == "M15"
        )
        changed_m15["regime_reading"] = {
            "state": "BREAKOUT_PENDING",
            "confidence": 0.8,
        }
        changed_m15["family_scores"] = {
            "trend_score": 0.7,
            "breakout_score": 9.99,
        }
        self.assertFalse(
            validate_m15_recovery_micro_receipt(
                receipt,
                pair_chart=changed_scoring_chart,
                expected_pair="EUR_USD",
                now_utc=now,
                current_spread_pips=0.8,
            )
        )

    def test_m15_recovery_accepts_one_recovered_and_one_rebuilding_fast_tail(self) -> None:
        chart = self._m15_recovery_chart()
        clean = self._chart(recovered=True)
        aggregate = chart["technical_candle_integrity"]
        clean_aggregate = clean["technical_candle_integrity"]
        assert isinstance(aggregate, dict)
        assert isinstance(clean_aggregate, dict)
        timeframes = aggregate["timeframes"]
        clean_timeframes = clean_aggregate["timeframes"]
        assert isinstance(timeframes, dict)
        assert isinstance(clean_timeframes, dict)
        timeframes["M1"] = copy.deepcopy(clean_timeframes["M1"])
        views = chart["views"]
        clean_views = clean["views"]
        assert isinstance(views, list)
        assert isinstance(clean_views, list)
        clean_m1 = next(
            view
            for view in clean_views
            if isinstance(view, dict) and view.get("granularity") == "M1"
        )
        m1_index = next(
            index
            for index, view in enumerate(views)
            if isinstance(view, dict) and view.get("granularity") == "M1"
        )
        views[m1_index] = copy.deepcopy(clean_m1)

        receipt = build_m15_recovery_micro_receipt(
            chart,
            expected_pair="EUR_USD",
            now_utc=datetime(2026, 7, 14, 0, 45, tzinfo=timezone.utc),
            current_spread_pips=0.8,
        )

        self.assertIsNotNone(receipt)
        assert receipt is not None
        self.assertEqual(
            receipt["fast_timeframe_receipts"]["M1"]["recovery_state"],
            "RECOVERED",
        )
        self.assertEqual(
            receipt["fast_timeframe_receipts"]["M5"]["recovery_state"],
            "RECOVERING",
        )

    def test_m15_recovery_keeps_raw_direction_when_calibration_is_below_live_floor(self) -> None:
        chart = self._m15_recovery_chart()
        now = datetime(2026, 7, 14, 0, 45, tzinfo=timezone.utc)

        with patch(
            "quant_rabbit.strategy.directional_forecaster._calibrated_confidence",
            side_effect=lambda **kwargs: (
                kwargs["raw_confidence"] * 0.64,
                0.64,
            ),
        ):
            forecast = synthesize_forecast(
                pair="EUR_USD",
                pair_chart=chart,
                current_price=1.1000,
                pattern_signals=[],
                projection_signals=[
                    _Sig(
                        "DOWN",
                        100.0,
                        1.0,
                        "M15 downside detector",
                        timeframe="M15",
                    )
                ],
                correlation_signals=[
                    _Sig(
                        "UP",
                        30.0,
                        1.0,
                        "M15 opposing context",
                        timeframe="M15",
                    )
                ],
                paths=[],
                spread_pips=0.8,
                now_utc=now,
                require_technical_candle_integrity=True,
            )

        self.assertEqual(forecast.direction, "DOWN")
        self.assertLess(forecast.confidence, 0.55)
        self.assertEqual(forecast.calibration_multiplier, 0.64)
        self.assertEqual(
            forecast.m15_recovery_forecast_evidence["raw_winner"],
            "DOWN",
        )
        self.assertEqual(
            forecast.m15_recovery_forecast_evidence["final_direction"],
            "DOWN",
        )

    def test_m15_recovery_calibration_excludes_mixed_fast_regime(self) -> None:
        chart = self._m15_recovery_chart()
        now = datetime(2026, 7, 14, 0, 45, tzinfo=timezone.utc)
        observed_regimes: list[tuple[str, object]] = []

        def known_weak(**kwargs):
            observed_regimes.append(("known_weak", kwargs["regime"]))
            return None

        def projection_multiplier(**kwargs):
            observed_regimes.append(("projection", kwargs["regime"]))
            return 1.0

        def forecast_calibration(**kwargs):
            observed_regimes.append(("forecast", kwargs["regime"]))
            return kwargs["raw_confidence"], 1.0

        forecasts = []
        with (
            patch(
                "quant_rabbit.strategy.directional_forecaster._projection_signal_known_weak_reason",
                side_effect=known_weak,
            ),
            patch(
                "quant_rabbit.strategy.directional_forecaster._projection_signal_calibration_multiplier",
                side_effect=projection_multiplier,
            ),
            patch(
                "quant_rabbit.strategy.directional_forecaster._calibrated_confidence",
                side_effect=forecast_calibration,
            ),
        ):
            for raw_regime in ("TREND", "RANGE"):
                forecasts.append(
                    synthesize_forecast(
                        pair="EUR_USD",
                        pair_chart=copy.deepcopy(chart),
                        current_price=1.1000,
                        pattern_signals=[],
                        projection_signals=[
                            _Sig(
                                "DOWN",
                                100.0,
                                1.0,
                                "M15 downside detector",
                                timeframe="M15",
                            )
                        ],
                        correlation_signals=[],
                        paths=[],
                        hit_rates={},
                        regime=raw_regime,
                        spread_pips=0.8,
                        now_utc=now,
                        require_technical_candle_integrity=True,
                    )
                )

        self.assertTrue(observed_regimes)
        self.assertEqual(
            {regime for _stage, regime in observed_regimes},
            {None},
        )
        self.assertEqual(
            [
                (
                    forecast.direction,
                    forecast.confidence,
                    forecast.raw_confidence,
                    forecast.calibration_multiplier,
                    forecast.component_scores,
                )
                for forecast in forecasts
            ],
            [
                (
                    forecasts[0].direction,
                    forecasts[0].confidence,
                    forecasts[0].raw_confidence,
                    forecasts[0].calibration_multiplier,
                    forecasts[0].component_scores,
                )
            ]
            * 2,
        )

    def test_m15_recovery_rejects_wide_current_spread_and_malformed_fast_receipt(self) -> None:
        for mutation, spread in (
            ("wide_spread", 99.0),
            ("malformed", 0.8),
            ("stale_m15", 0.8),
        ):
            with self.subTest(mutation=mutation):
                chart = self._m15_recovery_chart()
                if mutation == "malformed":
                    aggregate = chart["technical_candle_integrity"]
                    assert isinstance(aggregate, dict)
                    timeframes = aggregate["timeframes"]
                    assert isinstance(timeframes, dict)
                    item = timeframes["M1"]
                    assert isinstance(item, dict)
                    item["malformed_count"] = 1
                elif mutation == "stale_m15":
                    aggregate = chart["technical_candle_integrity"]
                    assert isinstance(aggregate, dict)
                    timeframes = aggregate["timeframes"]
                    assert isinstance(timeframes, dict)
                    item = timeframes["M15"]
                    assert isinstance(item, dict)
                    stale_latest = datetime(
                        2026, 7, 14, 0, 0, tzinfo=timezone.utc
                    )
                    item["latest_complete_timestamp_utc"] = stale_latest.isoformat()
                    item["latest_clean_timestamp_utc"] = stale_latest.isoformat()
                    views = chart["views"]
                    assert isinstance(views, list)
                    m15_view = next(
                        view
                        for view in views
                        if isinstance(view, dict)
                        and view.get("granularity") == "M15"
                    )
                    m15_view["recent_candles"] = [
                        {
                            "t": (
                                stale_latest
                                - timedelta(minutes=15 * (29 - index))
                            ).isoformat()
                        }
                        for index in range(30)
                    ]
                forecast = synthesize_forecast(
                    pair="EUR_USD",
                    pair_chart=chart,
                    current_price=1.1000,
                    pattern_signals=[],
                    projection_signals=[
                        _Sig("UP", 100.0, 1.0, "M15 up detector", timeframe="M15")
                    ],
                    correlation_signals=[],
                    paths=[],
                    spread_pips=spread,
                    now_utc=datetime(2026, 7, 14, 0, 45, tzinfo=timezone.utc),
                    require_technical_candle_integrity=True,
                )
                self.assertEqual(forecast.direction, "UNCLEAR")
                self.assertEqual(forecast.m15_recovery_receipt, {})

    def test_normal_clean_tail_does_not_emit_recovery_receipt(self) -> None:
        forecast = self._production_forecast(self._chart(recovered=True))
        self.assertEqual(forecast.m15_recovery_receipt, {})

    def test_reordered_0042_then_0013_quarantines_cannot_fake_a_clean_tail(self) -> None:
        chart = self._chart(recovered=True)
        integrity = chart["technical_candle_integrity"]
        assert isinstance(integrity, dict)
        timeframes = integrity["timeframes"]
        assert isinstance(timeframes, dict)
        item = timeframes["M1"]
        assert isinstance(item, dict)
        details = item["quarantine_details"]
        assert isinstance(details, list) and isinstance(details[-1], dict)
        older = copy.deepcopy(details[-1])
        recent = copy.deepcopy(details[-1])
        recent["timestamp_utc"] = "2026-07-14T00:42:00+00:00"
        older["timestamp_utc"] = "2026-07-14T00:13:00+00:00"
        # Reproduce the bypass exactly: a quarantine inside the published
        # clean window is placed before an older detail so a last-row-only
        # verifier sees 00:13 and misses 00:42.
        details[:] = [recent, older]
        item["requested_count"] = 71
        item["raw_entry_count"] = 71
        item["complete_entry_count"] = 71
        item["contaminated_count"] = 2
        item["quarantined_count"] = 2
        window = item["quarantine_details_window"]
        assert isinstance(window, dict)
        window.update({
            "start_index": 0,
            "end_index_exclusive": 2,
            "total_count": 2,
            "total_code_counts": {
                TECHNICAL_CANDLE_SPREAD_CONTAMINATED: 2,
                TECHNICAL_CANDLE_PROVENANCE_INVALID: 0,
            },
            "published_code_counts": {
                TECHNICAL_CANDLE_SPREAD_CONTAMINATED: 2,
                TECHNICAL_CANDLE_PROVENANCE_INVALID: 0,
            },
            "omitted_code_counts": {
                TECHNICAL_CANDLE_SPREAD_CONTAMINATED: 0,
                TECHNICAL_CANDLE_PROVENANCE_INVALID: 0,
            },
            "total_timestamped_count": 2,
            "published_timestamped_count": 2,
            "omitted_timestamped_count": 0,
            "latest_timestamp_utc": "2026-07-14T00:13:00+00:00",
        })

        forecast = self._production_forecast(chart)

        self.assertEqual(forecast.direction, "UNCLEAR")
        self.assertEqual(forecast.confidence, 0.0)
        self.assertIn(TECHNICAL_CANDLE_PROVENANCE_INVALID, forecast.drivers_against)

    def test_equal_time_reorder_or_duplicate_quarantine_is_rejected(self) -> None:
        for case in (
            "equal_time_reordered",
            "equal_time_canonical",
            "offset_equivalent",
            "exact_duplicate",
        ):
            with self.subTest(case=case):
                chart = self._chart(recovered=True)
                integrity = chart["technical_candle_integrity"]
                assert isinstance(integrity, dict)
                timeframes = integrity["timeframes"]
                assert isinstance(timeframes, dict)
                item = timeframes["M1"]
                assert isinstance(item, dict)
                details = item["quarantine_details"]
                assert isinstance(details, list) and isinstance(details[0], dict)
                first = copy.deepcopy(details[0])
                second = copy.deepcopy(details[0])
                if case == "equal_time_reordered":
                    first["max_spread_pips"] = second["max_spread_pips"] + 1.0
                elif case == "equal_time_canonical":
                    second["max_spread_pips"] = first["max_spread_pips"] + 1.0
                elif case == "offset_equivalent":
                    second["timestamp_utc"] = "2026-07-14T00:13:00Z"
                details[:] = [first, second]
                item.update(
                    {
                        "requested_count": 71,
                        "raw_entry_count": 71,
                        "complete_entry_count": 71,
                        "contaminated_count": 2,
                        "quarantined_count": 2,
                    }
                )
                window = item["quarantine_details_window"]
                assert isinstance(window, dict)
                window.update(
                    {
                        "start_index": 0,
                        "end_index_exclusive": 2,
                        "total_count": 2,
                        "total_code_counts": {
                            TECHNICAL_CANDLE_SPREAD_CONTAMINATED: 2,
                            TECHNICAL_CANDLE_PROVENANCE_INVALID: 0,
                        },
                        "published_code_counts": {
                            TECHNICAL_CANDLE_SPREAD_CONTAMINATED: 2,
                            TECHNICAL_CANDLE_PROVENANCE_INVALID: 0,
                        },
                        "omitted_code_counts": {
                            TECHNICAL_CANDLE_SPREAD_CONTAMINATED: 0,
                            TECHNICAL_CANDLE_PROVENANCE_INVALID: 0,
                        },
                        "total_timestamped_count": 2,
                        "published_timestamped_count": 2,
                        "omitted_timestamped_count": 0,
                    }
                )

                forecast = self._production_forecast(chart)

                self.assertEqual(forecast.direction, "UNCLEAR")
                self.assertEqual(forecast.confidence, 0.0)
                self.assertIn(
                    TECHNICAL_CANDLE_PROVENANCE_INVALID,
                    forecast.drivers_against,
                )

    def test_huge_integrity_integer_fails_closed_without_overflow(self) -> None:
        for field in ("normal_spread_pips", "quarantine_max_spread_pips"):
            with self.subTest(field=field):
                chart = self._chart(recovered=True)
                integrity = chart["technical_candle_integrity"]
                assert isinstance(integrity, dict)
                timeframes = integrity["timeframes"]
                assert isinstance(timeframes, dict)
                item = timeframes["M1"]
                assert isinstance(item, dict)
                if field == "normal_spread_pips":
                    item["normal_spread_pips"] = 10**1000
                else:
                    details = item["quarantine_details"]
                    assert isinstance(details, list) and isinstance(details[0], dict)
                    details[0]["max_spread_pips"] = 10**1000

                forecast = self._production_forecast(chart)

                self.assertEqual(forecast.direction, "UNCLEAR")
                self.assertEqual(forecast.confidence, 0.0)
                self.assertIn(
                    TECHNICAL_CANDLE_PROVENANCE_INVALID,
                    forecast.drivers_against,
                )

    def test_noncanonical_latest_quarantine_timestamp_is_rejected(self) -> None:
        chart = self._chart(recovered=True)
        integrity = chart["technical_candle_integrity"]
        assert isinstance(integrity, dict)
        timeframes = integrity["timeframes"]
        assert isinstance(timeframes, dict)
        item = timeframes["M1"]
        assert isinstance(item, dict)
        window = item["quarantine_details_window"]
        assert isinstance(window, dict)
        window["latest_timestamp_utc"] = "2026-07-14T09:13:00+09:00"

        forecast = self._production_forecast(chart)

        self.assertEqual(forecast.direction, "UNCLEAR")
        self.assertEqual(forecast.confidence, 0.0)
        self.assertIn(
            TECHNICAL_CANDLE_PROVENANCE_INVALID,
            forecast.drivers_against,
        )

    def test_null_timestamp_spread_detail_cannot_fake_a_clean_tail(self) -> None:
        chart = self._chart(recovered=True)
        integrity = chart["technical_candle_integrity"]
        assert isinstance(integrity, dict)
        timeframes = integrity["timeframes"]
        assert isinstance(timeframes, dict)
        item = timeframes["M1"]
        assert isinstance(item, dict)
        details = item["quarantine_details"]
        assert isinstance(details, list) and isinstance(details[0], dict)
        older = copy.deepcopy(details[0])
        hidden = copy.deepcopy(details[0])
        hidden["timestamp_utc"] = None
        details[:] = [hidden, older]
        item.update({
            "requested_count": 71,
            "raw_entry_count": 71,
            "complete_entry_count": 71,
            "contaminated_count": 2,
            "quarantined_count": 2,
        })
        window = item["quarantine_details_window"]
        assert isinstance(window, dict)
        window.update({
            "start_index": 0,
            "end_index_exclusive": 2,
            "total_count": 2,
            "total_code_counts": {
                TECHNICAL_CANDLE_SPREAD_CONTAMINATED: 2,
                TECHNICAL_CANDLE_PROVENANCE_INVALID: 0,
            },
            "published_code_counts": {
                TECHNICAL_CANDLE_SPREAD_CONTAMINATED: 2,
                TECHNICAL_CANDLE_PROVENANCE_INVALID: 0,
            },
            "omitted_code_counts": {
                TECHNICAL_CANDLE_SPREAD_CONTAMINATED: 0,
                TECHNICAL_CANDLE_PROVENANCE_INVALID: 0,
            },
            "total_timestamped_count": 1,
            "published_timestamped_count": 1,
            "omitted_timestamped_count": 0,
            "latest_timestamp_utc": older["timestamp_utc"],
        })

        forecast = self._production_forecast(chart)

        self.assertEqual(forecast.direction, "UNCLEAR")
        self.assertEqual(forecast.confidence, 0.0)
        self.assertIn(TECHNICAL_CANDLE_PROVENANCE_INVALID, forecast.drivers_against)

    def test_relabelled_quarantine_code_cannot_escape_malformed_count(self) -> None:
        chart = self._chart(recovered=True)
        integrity = chart["technical_candle_integrity"]
        assert isinstance(integrity, dict)
        timeframes = integrity["timeframes"]
        assert isinstance(timeframes, dict)
        item = timeframes["M1"]
        assert isinstance(item, dict)
        details = item["quarantine_details"]
        assert isinstance(details, list) and isinstance(details[0], dict)
        older = copy.deepcopy(details[0])
        details[:] = [
            {
                "timestamp_utc": None,
                "code": TECHNICAL_CANDLE_PROVENANCE_INVALID,
                "reason": "timestamp missing or malformed",
            },
            older,
        ]
        item.update({
            "requested_count": 71,
            "raw_entry_count": 71,
            "complete_entry_count": 71,
            "contaminated_count": 2,
            "quarantined_count": 2,
        })
        window = item["quarantine_details_window"]
        assert isinstance(window, dict)
        window.update({
            "start_index": 0,
            "end_index_exclusive": 2,
            "total_count": 2,
            "total_code_counts": {
                TECHNICAL_CANDLE_SPREAD_CONTAMINATED: 2,
                TECHNICAL_CANDLE_PROVENANCE_INVALID: 0,
            },
            "published_code_counts": {
                TECHNICAL_CANDLE_SPREAD_CONTAMINATED: 1,
                TECHNICAL_CANDLE_PROVENANCE_INVALID: 1,
            },
            "omitted_code_counts": {
                TECHNICAL_CANDLE_SPREAD_CONTAMINATED: 0,
                TECHNICAL_CANDLE_PROVENANCE_INVALID: 0,
            },
            "total_timestamped_count": 1,
            "published_timestamped_count": 1,
            "omitted_timestamped_count": 0,
            "latest_timestamp_utc": older["timestamp_utc"],
        })

        forecast = self._production_forecast(chart)

        self.assertEqual(forecast.direction, "UNCLEAR")
        self.assertEqual(forecast.confidence, 0.0)
        self.assertIn(TECHNICAL_CANDLE_PROVENANCE_INVALID, forecast.drivers_against)

    def test_valid_truncated_mixed_window_matches_producer_contract(self) -> None:
        chart = self._chart(recovered=True)
        integrity = chart["technical_candle_integrity"]
        views = chart["views"]
        assert isinstance(integrity, dict) and isinstance(views, list)
        timeframes = integrity["timeframes"]
        assert isinstance(timeframes, dict)
        item = timeframes["M1"]
        assert isinstance(item, dict)
        spread_cap = item["spread_cap_pips"]
        spread_details = [
            {
                "timestamp_utc": f"2026-07-14T00:{minute:02d}:00+00:00",
                "code": TECHNICAL_CANDLE_SPREAD_CONTAMINATED,
                "max_spread_pips": float(spread_cap) + 1.0,
                "spread_cap_pips": spread_cap,
            }
            for minute in range(7, 14)
        ]
        item.update({
            "requested_count": 78,
            "raw_entry_count": 78,
            "complete_entry_count": 78,
            "contaminated_count": 7,
            "malformed_count": 2,
            "quarantined_count": 9,
            "provenance_complete": False,
            "forecast_blocking": True,
            "codes": [
                TECHNICAL_CANDLE_SPREAD_CONTAMINATED,
                TECHNICAL_CANDLE_PROVENANCE_INVALID,
            ],
            "blocking_codes": [TECHNICAL_CANDLE_PROVENANCE_INVALID],
            "evaluation_status": "BLOCKED",
            "quarantine_details": [
                {
                    "timestamp_utc": None,
                    "code": TECHNICAL_CANDLE_PROVENANCE_INVALID,
                    "reason": "candles[1] is not an object",
                },
                *spread_details,
            ],
            "quarantine_details_truncated": 1,
            "quarantine_details_window": {
                "selection": TECHNICAL_CANDLE_QUARANTINE_DETAILS_SELECTION,
                "order": TECHNICAL_CANDLE_QUARANTINE_DETAILS_ORDER,
                "limit": TECHNICAL_CANDLE_QUARANTINE_DETAIL_LIMIT,
                "start_index": 1,
                "end_index_exclusive": 9,
                "total_count": 9,
                "total_code_counts": {
                    TECHNICAL_CANDLE_SPREAD_CONTAMINATED: 7,
                    TECHNICAL_CANDLE_PROVENANCE_INVALID: 2,
                },
                "published_code_counts": {
                    TECHNICAL_CANDLE_SPREAD_CONTAMINATED: 7,
                    TECHNICAL_CANDLE_PROVENANCE_INVALID: 1,
                },
                "omitted_code_counts": {
                    TECHNICAL_CANDLE_SPREAD_CONTAMINATED: 0,
                    TECHNICAL_CANDLE_PROVENANCE_INVALID: 1,
                },
                "total_timestamped_count": 7,
                "published_timestamped_count": 7,
                "omitted_timestamped_count": 0,
                "latest_timestamp_utc": "2026-07-14T00:13:00+00:00",
            },
        })
        view = next(
            candidate
            for candidate in views
            if isinstance(candidate, dict) and candidate.get("granularity") == "M1"
        )

        self.assertTrue(
            _valid_mba_integrity(
                item,
                chart_generated_at=datetime(2026, 7, 14, 0, 45, tzinfo=timezone.utc),
                view=view,
                forecast_now=datetime(2026, 7, 14, 0, 45, tzinfo=timezone.utc),
            )
        )

    def test_quarantine_window_boolean_counts_fail_closed(self) -> None:
        for path in (
            ("limit",),
            ("start_index",),
            ("end_index_exclusive",),
            ("total_count",),
            ("total_code_counts", TECHNICAL_CANDLE_SPREAD_CONTAMINATED),
            ("published_code_counts", TECHNICAL_CANDLE_SPREAD_CONTAMINATED),
            ("omitted_code_counts", TECHNICAL_CANDLE_PROVENANCE_INVALID),
            ("total_timestamped_count",),
            ("published_timestamped_count",),
            ("omitted_timestamped_count",),
        ):
            with self.subTest(path=path):
                chart = self._chart(recovered=True)
                integrity = chart["technical_candle_integrity"]
                assert isinstance(integrity, dict)
                timeframes = integrity["timeframes"]
                assert isinstance(timeframes, dict)
                item = timeframes["M1"]
                assert isinstance(item, dict)
                window = item["quarantine_details_window"]
                assert isinstance(window, dict)
                if len(path) == 1:
                    window[path[0]] = bool(window[path[0]])
                else:
                    nested = window[path[0]]
                    assert isinstance(nested, dict)
                    nested[path[1]] = bool(nested[path[1]])

                forecast = self._production_forecast(chart)

                self.assertEqual(forecast.direction, "UNCLEAR")
                self.assertEqual(forecast.confidence, 0.0)
                self.assertIn(
                    TECHNICAL_CANDLE_PROVENANCE_INVALID,
                    forecast.drivers_against,
                )

    def test_quarantine_window_float_counts_fail_closed(self) -> None:
        for path in (
            ("limit",),
            ("start_index",),
            ("end_index_exclusive",),
            ("total_count",),
            ("total_code_counts", TECHNICAL_CANDLE_SPREAD_CONTAMINATED),
            ("published_code_counts", TECHNICAL_CANDLE_SPREAD_CONTAMINATED),
            ("omitted_code_counts", TECHNICAL_CANDLE_PROVENANCE_INVALID),
            ("total_timestamped_count",),
            ("published_timestamped_count",),
            ("omitted_timestamped_count",),
        ):
            with self.subTest(path=path):
                chart = self._chart(recovered=True)
                integrity = chart["technical_candle_integrity"]
                assert isinstance(integrity, dict)
                timeframes = integrity["timeframes"]
                assert isinstance(timeframes, dict)
                item = timeframes["M1"]
                assert isinstance(item, dict)
                window = item["quarantine_details_window"]
                assert isinstance(window, dict)
                if len(path) == 1:
                    window[path[0]] = float(window[path[0]])
                else:
                    nested = window[path[0]]
                    assert isinstance(nested, dict)
                    nested[path[1]] = float(nested[path[1]])

                forecast = self._production_forecast(chart)

                self.assertEqual(forecast.direction, "UNCLEAR")
                self.assertEqual(forecast.confidence, 0.0)
                self.assertIn(
                    TECHNICAL_CANDLE_PROVENANCE_INVALID,
                    forecast.drivers_against,
                )

    def test_count_and_recent_cadence_tampering_fail_closed(self) -> None:
        count_tamper = self._chart(recovered=True)
        count_integrity = count_tamper["technical_candle_integrity"]
        assert isinstance(count_integrity, dict)
        count_timeframes = count_integrity["timeframes"]
        assert isinstance(count_timeframes, dict)
        count_item = count_timeframes["M1"]
        assert isinstance(count_item, dict)
        count_item["requested_count"] = int(count_item["raw_entry_count"]) + 1
        count_item["coverage_complete"] = False

        cadence_tamper = self._chart(recovered=True)
        cadence_views = cadence_tamper["views"]
        assert isinstance(cadence_views, list)
        cadence_recent = cadence_views[0]["recent_candles"]
        assert isinstance(cadence_recent, list)
        tail = datetime.fromisoformat(str(cadence_recent[-1]["t"]))
        cadence_recent[0]["t"] = (tail - timedelta(seconds=1)).isoformat()

        for chart in (count_tamper, cadence_tamper):
            with self.subTest(chart=chart):
                forecast = self._production_forecast(chart)
                self.assertEqual(forecast.direction, "UNCLEAR")
                self.assertEqual(forecast.confidence, 0.0)
                self.assertIn(
                    TECHNICAL_CANDLE_PROVENANCE_INVALID,
                    forecast.drivers_against,
                )

    def test_persisted_chart_without_mba_integrity_fails_closed(self) -> None:
        chart = self._chart(recovered=True)
        chart.pop("technical_candle_integrity")

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=chart,
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[_Sig("UP", 100.0, 1.0, "strong up detector")],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "UNCLEAR")
        self.assertEqual(forecast.confidence, 0.0)
        self.assertIn(TECHNICAL_CANDLE_PROVENANCE_INVALID, forecast.drivers_against)

    def test_contradictory_aggregate_cannot_hide_blocking_tail(self) -> None:
        chart = self._chart(recovered=False)
        integrity = chart["technical_candle_integrity"]
        assert isinstance(integrity, dict)
        integrity["forecast_blocking"] = False
        integrity["blocking_codes"] = []
        integrity["evaluation_status"] = "DEGRADED"

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=chart,
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[_Sig("UP", 100.0, 1.0, "strong up detector")],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "UNCLEAR")
        self.assertEqual(forecast.confidence, 0.0)
        self.assertIn(TECHNICAL_CANDLE_PROVENANCE_INVALID, forecast.drivers_against)

    def test_self_consistent_noncanonical_spread_policy_tamper_fails_closed(self) -> None:
        chart = self._chart(recovered=True)
        integrity = chart["technical_candle_integrity"]
        assert isinstance(integrity, dict)
        timeframes = integrity["timeframes"]
        assert isinstance(timeframes, dict)
        item = timeframes["M1"]
        assert isinstance(item, dict)
        tampered_normal = float(item["normal_spread_pips"]) + 0.1
        tampered_multiple = float(item["max_spread_multiple"]) + 0.1
        item["normal_spread_pips"] = tampered_normal
        item["max_spread_multiple"] = tampered_multiple
        item["spread_cap_pips"] = round(tampered_normal * tampered_multiple, 6)

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=chart,
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[_Sig("UP", 100.0, 1.0, "strong up detector")],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "UNCLEAR")
        self.assertEqual(forecast.confidence, 0.0)
        self.assertIn(TECHNICAL_CANDLE_PROVENANCE_INVALID, forecast.drivers_against)

    def test_spread_policy_receipt_rejects_string_and_bool_numbers(self) -> None:
        for field in (
            "normal_spread_pips",
            "max_spread_multiple",
            "spread_cap_pips",
        ):
            with self.subTest(field=field):
                chart = self._chart(recovered=True)
                integrity = chart["technical_candle_integrity"]
                assert isinstance(integrity, dict)
                timeframes = integrity["timeframes"]
                assert isinstance(timeframes, dict)
                item = timeframes["M1"]
                assert isinstance(item, dict)
                item[field] = True if field == "max_spread_multiple" else str(item[field])

                forecast = synthesize_forecast(
                    pair="EUR_USD",
                    pair_chart=chart,
                    current_price=1.1000,
                    pattern_signals=[],
                    projection_signals=[_Sig("UP", 100.0, 1.0, "strong up detector")],
                    correlation_signals=[],
                    paths=[],
                )

                self.assertEqual(forecast.direction, "UNCLEAR")
                self.assertEqual(forecast.confidence, 0.0)
                self.assertIn(TECHNICAL_CANDLE_PROVENANCE_INVALID, forecast.drivers_against)

    def test_execution_timeframe_cannot_claim_provenance_only_spread_mode(self) -> None:
        chart = self._chart(recovered=True)
        integrity = chart["technical_candle_integrity"]
        assert isinstance(integrity, dict)
        timeframes = integrity["timeframes"]
        assert isinstance(timeframes, dict)
        item = timeframes["M1"]
        assert isinstance(item, dict)
        item["spread_evaluation_mode"] = "PROVENANCE_ONLY_HIGHER_TIMEFRAME"

        forecast = self._production_forecast(chart)

        self.assertEqual(forecast.direction, "UNCLEAR")
        self.assertEqual(forecast.confidence, 0.0)
        self.assertIn(TECHNICAL_CANDLE_PROVENANCE_INVALID, forecast.drivers_against)

    def test_payload_identity_receipt_must_match_pair_and_timeframe(self) -> None:
        for field, value in (
            ("payload_instrument", "GBP_USD"),
            ("payload_granularity", "H1"),
        ):
            with self.subTest(field=field):
                chart = self._chart(recovered=True)
                integrity = chart["technical_candle_integrity"]
                assert isinstance(integrity, dict)
                timeframes = integrity["timeframes"]
                assert isinstance(timeframes, dict)
                item = timeframes["M1"]
                assert isinstance(item, dict)
                item[field] = value

                forecast = self._production_forecast(chart)

                self.assertEqual(forecast.direction, "UNCLEAR")
                self.assertEqual(forecast.confidence, 0.0)
                self.assertIn(TECHNICAL_CANDLE_PROVENANCE_INVALID, forecast.drivers_against)

    def test_explicit_production_contract_blocks_when_timestamp_and_receipt_are_both_removed(self) -> None:
        chart = self._chart(recovered=True)
        chart.pop("generated_at_utc")
        chart.pop("technical_candle_integrity")

        forecast = self._production_forecast(chart)

        self.assertEqual(forecast.direction, "UNCLEAR")
        self.assertEqual(forecast.confidence, 0.0)
        self.assertIn(TECHNICAL_CANDLE_PROVENANCE_INVALID, forecast.drivers_against)

    def test_default_forecast_contract_fails_closed_without_integrity(self) -> None:
        chart = self._chart(recovered=True)
        chart.pop("generated_at_utc")
        chart.pop("technical_candle_integrity")

        forecast = _synthesize_forecast(
            pair="EUR_USD",
            pair_chart=chart,
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[_Sig("UP", 100.0, 1.0, "strong up detector")],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "UNCLEAR")
        self.assertEqual(forecast.confidence, 0.0)
        self.assertIn(TECHNICAL_CANDLE_PROVENANCE_INVALID, forecast.drivers_against)

    def test_production_generated_at_requires_exact_timezone_aware_iso_string(self) -> None:
        for value in (None, True, 1720917900, "garbage", "2026-07-14T00:45:00"):
            with self.subTest(value=value):
                chart = self._chart(recovered=True)
                if value is None:
                    chart.pop("generated_at_utc")
                else:
                    chart["generated_at_utc"] = value

                forecast = self._production_forecast(chart)

                self.assertEqual(forecast.direction, "UNCLEAR")
                self.assertEqual(forecast.confidence, 0.0)
                self.assertIn(TECHNICAL_CANDLE_PROVENANCE_INVALID, forecast.drivers_against)

    def test_production_forecast_requires_aware_now_datetime(self) -> None:
        for value in (
            None,
            True,
            1720917900,
            "2026-07-14T00:45:00Z",
            datetime(2026, 7, 14, 0, 45),
        ):
            with self.subTest(value=value):
                forecast = self._production_forecast(
                    self._chart(recovered=True),
                    now_utc=value,
                )

                self.assertEqual(forecast.direction, "UNCLEAR")
                self.assertEqual(forecast.confidence, 0.0)
                self.assertIn(TECHNICAL_CANDLE_PROVENANCE_INVALID, forecast.drivers_against)

    def test_production_chart_freshness_blocks_stale_and_future_artifacts(self) -> None:
        generated = datetime(2026, 7, 14, 0, 45, tzinfo=timezone.utc)
        stale = self._production_forecast(
            self._chart(recovered=True),
            now_utc=generated + timedelta(
                seconds=TECHNICAL_CANDLE_FORECAST_MAX_AGE_SECONDS + 1
            ),
        )
        future = self._production_forecast(
            self._chart(recovered=True),
            now_utc=generated - timedelta(
                seconds=TECHNICAL_CANDLE_FORECAST_MAX_FUTURE_SKEW_SECONDS + 1
            ),
        )

        for forecast in (stale, future):
            self.assertEqual(forecast.direction, "UNCLEAR")
            self.assertEqual(forecast.confidence, 0.0)
            self.assertIn(TECHNICAL_CANDLE_PROVENANCE_INVALID, forecast.drivers_against)

    def test_production_fast_freshness_boundary_remains_reachable(self) -> None:
        generated = datetime(2026, 7, 14, 0, 45, tzinfo=timezone.utc)
        forecast = self._production_forecast(
            self._chart(recovered=True),
            # M1 latest complete is 00:43, exactly three minutes old here.
            now_utc=generated + timedelta(minutes=1),
        )

        self.assertEqual(forecast.direction, "UP")
        self.assertGreater(forecast.confidence, 0.0)

    def test_production_fast_candle_freshness_blocks_stale_m1_or_m5(self) -> None:
        for timeframe, stale_tail, quarantine_tail in (
            ("M1", "2026-07-14T00:41:59+00:00", "2026-07-14T00:41:00+00:00"),
            ("M5", "2026-07-14T00:34:59+00:00", None),
        ):
            with self.subTest(timeframe=timeframe):
                chart = self._chart(recovered=True)
                integrity = chart["technical_candle_integrity"]
                views = chart["views"]
                assert isinstance(integrity, dict) and isinstance(views, list)
                timeframes = integrity["timeframes"]
                assert isinstance(timeframes, dict)
                item = timeframes[timeframe]
                assert isinstance(item, dict)
                item["latest_complete_timestamp_utc"] = stale_tail
                item["latest_clean_timestamp_utc"] = stale_tail
                if quarantine_tail is not None:
                    details = item["quarantine_details"]
                    assert isinstance(details, list) and isinstance(details[-1], dict)
                    details[-1]["timestamp_utc"] = quarantine_tail
                view = next(
                    candidate
                    for candidate in views
                    if isinstance(candidate, dict)
                    and candidate.get("granularity") == timeframe
                )
                recent = view["recent_candles"]
                assert isinstance(recent, list) and isinstance(recent[-1], dict)
                recent[-1]["t"] = stale_tail

                forecast = self._production_forecast(chart)

                self.assertEqual(forecast.direction, "UNCLEAR")
                self.assertEqual(forecast.confidence, 0.0)
                self.assertIn(TECHNICAL_CANDLE_PROVENANCE_INVALID, forecast.drivers_against)

    def test_production_requires_both_m1_and_m5_receipts(self) -> None:
        for removed in (("M1",), ("M5",), ("M1", "M5")):
            with self.subTest(removed=removed):
                chart = self._chart(recovered=True)
                integrity = chart["technical_candle_integrity"]
                views = chart["views"]
                assert isinstance(integrity, dict) and isinstance(views, list)
                timeframes = integrity["timeframes"]
                requested = integrity["requested_timeframes"]
                assert isinstance(timeframes, dict) and isinstance(requested, list)
                for timeframe in removed:
                    timeframes.pop(timeframe)
                    requested.remove(timeframe)
                chart["views"] = [
                    view
                    for view in views
                    if isinstance(view, dict)
                    and view.get("granularity") not in removed
                ]
                remaining = list(timeframes.values())
                integrity["evaluated_timeframe_count"] = len(remaining)
                integrity["codes"] = list(dict.fromkeys(
                    code
                    for item in remaining
                    if isinstance(item, dict)
                    for code in item.get("codes", [])
                ))
                integrity["blocking_codes"] = list(dict.fromkeys(
                    code
                    for item in remaining
                    if isinstance(item, dict)
                    for code in item.get("blocking_codes", [])
                ))
                integrity["forecast_blocking"] = bool(integrity["blocking_codes"])
                integrity["evaluation_status"] = (
                    "BLOCKED"
                    if integrity["blocking_codes"]
                    else "DEGRADED"
                    if integrity["codes"]
                    else "PASS"
                )

                forecast = self._production_forecast(chart)

                self.assertEqual(forecast.direction, "UNCLEAR")
                self.assertEqual(forecast.confidence, 0.0)
                self.assertIn(TECHNICAL_CANDLE_PROVENANCE_INVALID, forecast.drivers_against)

    def test_view_and_top_level_timeframe_receipts_must_match_one_to_one(self) -> None:
        variants: list[tuple[str, dict[str, object]]] = []

        missing_view = copy.deepcopy(self._chart(recovered=True))
        missing_view["views"] = []
        variants.append(("missing_view", missing_view))

        duplicate_view = copy.deepcopy(self._chart(recovered=True))
        views = duplicate_view["views"]
        assert isinstance(views, list)
        views.append(copy.deepcopy(views[0]))
        variants.append(("duplicate_view", duplicate_view))

        mismatched_receipt = copy.deepcopy(self._chart(recovered=True))
        views = mismatched_receipt["views"]
        assert isinstance(views, list) and isinstance(views[0], dict)
        view_receipt = views[0]["candle_integrity"]
        assert isinstance(view_receipt, dict)
        view_receipt["clean_count"] = 40
        variants.append(("mismatched_receipt", mismatched_receipt))

        noncanonical_tf = copy.deepcopy(self._chart(recovered=True))
        integrity = noncanonical_tf["technical_candle_integrity"]
        views = noncanonical_tf["views"]
        assert isinstance(integrity, dict) and isinstance(views, list)
        timeframes = integrity["timeframes"]
        assert isinstance(timeframes, dict) and isinstance(views[0], dict)
        item = timeframes.pop("M1")
        assert isinstance(item, dict)
        item["granularity"] = "m1"
        timeframes["m1"] = item
        integrity["requested_timeframes"] = ["m1"]
        views[0]["granularity"] = "m1"
        variants.append(("noncanonical_tf", noncanonical_tf))

        for name, chart in variants:
            with self.subTest(name=name):
                forecast = self._production_forecast(chart)
                self.assertEqual(forecast.direction, "UNCLEAR")
                self.assertEqual(forecast.confidence, 0.0)
                self.assertIn(TECHNICAL_CANDLE_PROVENANCE_INVALID, forecast.drivers_against)

    def test_forecast_pair_must_match_chart_aggregate_and_child_pair_exactly(self) -> None:
        argument_mismatch = self._production_forecast(
            self._chart(recovered=True),
            pair="GBP_USD",
        )
        self.assertIn(TECHNICAL_CANDLE_PROVENANCE_INVALID, argument_mismatch.drivers_against)

        for level in ("chart", "aggregate", "child"):
            with self.subTest(level=level):
                chart = copy.deepcopy(self._chart(recovered=True))
                integrity = chart["technical_candle_integrity"]
                assert isinstance(integrity, dict)
                if level == "chart":
                    chart["pair"] = "GBP_USD"
                elif level == "aggregate":
                    integrity["pair"] = "GBP_USD"
                else:
                    timeframes = integrity["timeframes"]
                    assert isinstance(timeframes, dict)
                    child = timeframes["M1"]
                    assert isinstance(child, dict)
                    child["pair"] = "GBP_USD"

                forecast = self._production_forecast(chart)
                self.assertEqual(forecast.direction, "UNCLEAR")
                self.assertIn(TECHNICAL_CANDLE_PROVENANCE_INVALID, forecast.drivers_against)


class ProductionPairChartLoaderIntegrityTest(unittest.TestCase):
    @staticmethod
    def _write_packet(root: Path, generated_at: object = "2026-07-14T00:45:00+00:00") -> Path:
        path = root / "pair_charts.json"
        path.write_text(json.dumps({
            "generated_at_utc": generated_at,
            "charts": [{
                "pair": "EUR_USD",
                "generated_at_utc": "",
                "views": [],
            }],
        }))
        return path

    def test_loaders_unconditionally_replace_row_timestamp_with_trusted_outer_timestamp(self) -> None:
        from quant_rabbit.strategy.intent_generator import _load_pair_charts
        from quant_rabbit.strategy.trader_brain import _load_full_pair_charts_for_brain

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_packet(Path(tmp))

            brain_charts = _load_full_pair_charts_for_brain(path)
            intent_charts = _load_pair_charts(path)

        self.assertEqual(
            brain_charts["EUR_USD"]["generated_at_utc"],
            "2026-07-14T00:45:00+00:00",
        )
        assert intent_charts is not None
        self.assertEqual(
            intent_charts["EUR_USD"]["__raw_chart"]["generated_at_utc"],
            "2026-07-14T00:45:00+00:00",
        )

    def test_loaders_fail_closed_on_missing_or_malformed_outer_timestamp(self) -> None:
        from quant_rabbit.strategy.intent_generator import _load_pair_charts
        from quant_rabbit.strategy.trader_brain import _load_full_pair_charts_for_brain

        for value in (None, True, 1720917900, "garbage", "2026-07-14T00:45:00"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                path = self._write_packet(Path(tmp), generated_at=value)

                self.assertEqual(_load_full_pair_charts_for_brain(path), {})
                self.assertIsNone(_load_pair_charts(path))

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_packet(Path(tmp))
            payload = json.loads(path.read_text())
            payload.pop("generated_at_utc")
            path.write_text(json.dumps(payload))

            self.assertEqual(_load_full_pair_charts_for_brain(path), {})
            self.assertIsNone(_load_pair_charts(path))

    def test_loaders_fail_closed_on_duplicate_pair_rows(self) -> None:
        from quant_rabbit.strategy.intent_generator import _load_pair_charts
        from quant_rabbit.strategy.trader_brain import _load_full_pair_charts_for_brain

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_packet(Path(tmp))
            payload = json.loads(path.read_text())
            payload["charts"].append(copy.deepcopy(payload["charts"][0]))
            payload["charts"][1]["generated_at_utc"] = "2099-01-01T00:00:00+00:00"
            path.write_text(json.dumps(payload))

            self.assertEqual(_load_full_pair_charts_for_brain(path), {})
            self.assertIsNone(_load_pair_charts(path))

    def test_loaders_fail_closed_on_noncanonical_pair_rows(self) -> None:
        from quant_rabbit.strategy.intent_generator import _load_pair_charts
        from quant_rabbit.strategy.trader_brain import _load_full_pair_charts_for_brain

        for pair in (None, True, 123, "", "eur_usd", "EURUSD", "XAU_USD"):
            with self.subTest(pair=pair), tempfile.TemporaryDirectory() as tmp:
                path = self._write_packet(Path(tmp))
                payload = json.loads(path.read_text())
                payload["charts"][0]["pair"] = pair
                path.write_text(json.dumps(payload))

                self.assertEqual(_load_full_pair_charts_for_brain(path), {})
                self.assertIsNone(_load_pair_charts(path))


class TechnicalCandleIntegrityReceiptBindingTest(unittest.TestCase):
    """Tamper tests that bind aggregate receipts to their chart evidence."""

    _chart = staticmethod(TechnicalCandleIntegrityForecastGateTest._chart)
    _production_forecast = staticmethod(
        TechnicalCandleIntegrityForecastGateTest._production_forecast
    )

    def test_self_consistent_clean_state_cannot_hide_latest_contaminated_tail(self) -> None:
        chart = copy.deepcopy(self._chart(recovered=False))
        integrity = chart["technical_candle_integrity"]
        assert isinstance(integrity, dict)
        timeframes = integrity["timeframes"]
        assert isinstance(timeframes, dict)
        item = timeframes["M1"]
        assert isinstance(item, dict)
        item.update({
            "recent_tail_state": "CLEAN",
            "recent_tail_contaminated": False,
            "forecast_blocking": False,
            "blocking_codes": [],
            "evaluation_status": "DEGRADED",
        })
        integrity.update({
            "forecast_blocking": False,
            "blocking_codes": [],
            "evaluation_status": "DEGRADED",
        })

        forecast = self._production_forecast(chart)

        self.assertEqual(forecast.direction, "UNCLEAR")
        self.assertIn(TECHNICAL_CANDLE_PROVENANCE_INVALID, forecast.drivers_against)

    def test_impossible_clean_complete_counts_cannot_hide_contamination(self) -> None:
        chart = copy.deepcopy(self._chart(recovered=False))
        integrity = chart["technical_candle_integrity"]
        assert isinstance(integrity, dict)
        timeframes = integrity["timeframes"]
        assert isinstance(timeframes, dict)
        item = timeframes["M1"]
        assert isinstance(item, dict)
        item.update({
            "contaminated_count": 0,
            "quarantined_count": 0,
            "recent_tail_state": "CLEAN",
            "recent_tail_contaminated": False,
            "forecast_blocking": False,
            "codes": [],
            "blocking_codes": [],
            "evaluation_status": "PASS",
            "latest_complete_timestamp_utc": "2026-07-14T00:39:00+00:00",
            "quarantine_details": [],
        })
        integrity.update({
            "forecast_blocking": False,
            "codes": [],
            "blocking_codes": [],
            "evaluation_status": "PASS",
        })

        forecast = self._production_forecast(chart)

        self.assertEqual(forecast.direction, "UNCLEAR")
        self.assertIn(TECHNICAL_CANDLE_PROVENANCE_INVALID, forecast.drivers_against)

    def test_aggregate_evaluated_timeframe_count_rejects_bool(self) -> None:
        chart = self._chart(recovered=True)
        integrity = chart["technical_candle_integrity"]
        assert isinstance(integrity, dict)
        integrity["evaluated_timeframe_count"] = True

        forecast = self._production_forecast(chart)

        self.assertEqual(forecast.direction, "UNCLEAR")
        self.assertIn(TECHNICAL_CANDLE_PROVENANCE_INVALID, forecast.drivers_against)

    def test_view_indicator_count_and_recent_tail_must_bind_to_receipt(self) -> None:
        for field in ("candles_count", "recent_tail"):
            with self.subTest(field=field):
                chart = copy.deepcopy(self._chart(recovered=True))
                views = chart["views"]
                assert isinstance(views, list) and isinstance(views[0], dict)
                if field == "candles_count":
                    indicators = views[0]["indicators"]
                    assert isinstance(indicators, dict)
                    indicators["candles_count"] = 40
                else:
                    recent = views[0]["recent_candles"]
                    assert isinstance(recent, list) and isinstance(recent[-1], dict)
                    recent[-1]["t"] = "2026-07-14T00:40:00+00:00"

                forecast = self._production_forecast(chart)
                self.assertEqual(forecast.direction, "UNCLEAR")
                self.assertIn(TECHNICAL_CANDLE_PROVENANCE_INVALID, forecast.drivers_against)
