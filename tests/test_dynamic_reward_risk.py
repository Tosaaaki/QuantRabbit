"""Unit tests for the market-derived reward_risk computation in
intent_generator. AGENT_CONTRACT §3.5 mandates reward_risk be
regime-derived, not a fixed literal — these tests defend that contract."""

from __future__ import annotations

import unittest

from quant_rabbit.strategy.intent_generator import (
    DYNAMIC_RR_BASE,
    DYNAMIC_RR_CEILING,
    DYNAMIC_RR_FLOOR,
    _market_derived_reward_risk,
)


class MarketDerivedRewardRiskTest(unittest.TestCase):
    def test_empty_context_returns_base(self) -> None:
        rr, rats = _market_derived_reward_risk(None)
        self.assertAlmostEqual(rr, DYNAMIC_RR_BASE)
        self.assertEqual(len(rats), 1)
        rr2, _ = _market_derived_reward_risk({})
        self.assertAlmostEqual(rr2, DYNAMIC_RR_BASE)

    def test_high_atr_percentile_adds_bonus(self) -> None:
        ctx = {"confluence": {"atr_percentile_24h": 0.85}}
        rr, rats = _market_derived_reward_risk(ctx)
        self.assertGreater(rr, DYNAMIC_RR_BASE)
        self.assertTrue(any("ATR %ile" in r for r in rats))

    def test_low_atr_percentile_subtracts_penalty(self) -> None:
        ctx = {"confluence": {"atr_percentile_24h": 0.15}}
        rr, _ = _market_derived_reward_risk(ctx)
        self.assertLess(rr, DYNAMIC_RR_BASE)

    def test_trending_adx_adds_bonus(self) -> None:
        ctx = {"h1_adx": 30}
        rr, _ = _market_derived_reward_risk(ctx)
        self.assertGreater(rr, DYNAMIC_RR_BASE)

    def test_ranging_adx_subtracts_penalty(self) -> None:
        ctx = {"h1_adx": 12}
        rr, _ = _market_derived_reward_risk(ctx)
        self.assertLess(rr, DYNAMIC_RR_BASE)

    def test_session_overlap_bonus(self) -> None:
        ctx = {"session_current_tag": "LONDON_NY_OVERLAP"}
        rr, _ = _market_derived_reward_risk(ctx)
        self.assertGreater(rr, DYNAMIC_RR_BASE)

    def test_off_hours_session_penalty(self) -> None:
        ctx = {"session_current_tag": "OFF_HOURS"}
        rr, _ = _market_derived_reward_risk(ctx)
        self.assertLess(rr, DYNAMIC_RR_BASE)

    def test_pair_relative_24h_expansion_outlier_subtracts_penalty(self) -> None:
        ctx = {"confluence": {
            "range_24h_expansion_ratio": 9.0,
            "range_24h_expansion_upper_fence": 8.0,
            "range_24h_expansion_outlier": True,
        }}
        rr, _ = _market_derived_reward_risk(ctx)
        self.assertLess(rr, DYNAMIC_RR_BASE)

    def test_clamped_to_floor(self) -> None:
        # Triple negative: low ATR + low ADX + off hours + exhausted
        ctx = {
            "confluence": {"atr_percentile_24h": 0.1, "range_24h_sigma_multiple": 3.0},
            "h1_adx": 10,
            "session_current_tag": "OFF_HOURS",
        }
        rr, _ = _market_derived_reward_risk(ctx)
        self.assertGreaterEqual(rr, DYNAMIC_RR_FLOOR)
        self.assertAlmostEqual(rr, DYNAMIC_RR_FLOOR, places=2)

    def test_clamped_to_ceiling(self) -> None:
        # Triple positive: high ATR + trending ADX + LDN/NY overlap, no exhaustion
        ctx = {
            "confluence": {"atr_percentile_24h": 0.95, "range_24h_sigma_multiple": 1.2},
            "h1_adx": 35,
            "session_current_tag": "LONDON_NY_OVERLAP",
        }
        rr, _ = _market_derived_reward_risk(ctx)
        self.assertLessEqual(rr, DYNAMIC_RR_CEILING)
        self.assertAlmostEqual(rr, DYNAMIC_RR_CEILING, places=2)

    def test_h4_adx_fallback_when_h1_missing(self) -> None:
        ctx = {"h4_adx": 30}
        rr, _ = _market_derived_reward_risk(ctx)
        self.assertGreater(rr, DYNAMIC_RR_BASE)

    def test_session_bucket_fallback(self) -> None:
        """When `session_current_tag` is absent, `session_bucket` is used."""
        ctx = {"session_bucket": "LONDON_NY_OVERLAP"}
        rr, _ = _market_derived_reward_risk(ctx)
        self.assertGreater(rr, DYNAMIC_RR_BASE)


if __name__ == "__main__":
    unittest.main()
