"""Unit tests for strategy/tp_rebalancer.py."""

from __future__ import annotations

import os
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.strategy.tp_rebalancer import (
    HYSTERESIS_PIPS,
    TPAdjustment,
    _chart_context_from_chart,
    _extract_atr_pips,
    apply_tp_adjustments,
    compute_tp_adjustment,
    load_close_review_trade_ids,
    load_entry_thesis_blocker_trade_ids,
)


class ComputeTPAdjustmentTest(unittest.TestCase):
    def _kill_switch_off(self) -> None:
        if "QR_DISABLE_TP_REBALANCE" in os.environ:
            del os.environ["QR_DISABLE_TP_REBALANCE"]
        if "QR_ENABLE_MISSING_TP_REPAIR" in os.environ:
            del os.environ["QR_ENABLE_MISSING_TP_REPAIR"]
        if "QR_DISABLE_INSURANCE_TP" in os.environ:
            del os.environ["QR_DISABLE_INSURANCE_TP"]

    def test_long_expanded_tp(self) -> None:
        self._kill_switch_off()
        # Entry 1.3000, current price 1.3050 (50 pip in profit),
        # current TP 1.3050 (50 pip from entry), reward_risk=3.0, ATR=30
        # → desired distance = 90 pip → new TP 1.3090 (40 pip wider)
        adj = compute_tp_adjustment(
            trade_id="t1", pair="EUR_USD", side="LONG",
            entry_price=1.3000, current_tp=1.3050,
            current_price=1.3050, atr_pips=30, reward_risk=3.0,
        )
        self.assertIsNotNone(adj)
        self.assertGreater(adj.new_tp, adj.current_tp)  # expanded
        self.assertGreater(adj.distance_pips_new, adj.distance_pips_old)

    def test_short_expanded_tp(self) -> None:
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="t2", pair="USD_JPY", side="SHORT",
            entry_price=160.00, current_tp=159.70,
            current_price=159.70, atr_pips=40, reward_risk=2.5,
        )
        self.assertIsNotNone(adj)
        self.assertLess(adj.new_tp, adj.current_tp)  # TP moves DOWN for SHORT = wider
        self.assertGreater(adj.distance_pips_new, adj.distance_pips_old)

    def test_close_review_blocks_underwater_short_tp_expansion(self) -> None:
        """A close-reviewed loser must not be reclassified as a runner.

        Regression for 2026-05-28: EUR/USD SHORTs had fresh
        REVIEW_CLOSE/RECOMMEND_CLOSE sidecars, then the post-gateway TP pass
        widened their take-profits while Gate B close authorization was absent.
        """
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="471720", pair="EUR_USD", side="SHORT",
            entry_price=1.16090, current_tp=1.16004,
            current_price=1.16528,
            atr_pips=21.3, reward_risk=2.7,
            close_review_active=True,
        )

        self.assertIsNone(adj)

    def test_close_review_allows_underwater_short_tp_contraction(self) -> None:
        """Close review should freeze runner expansion, not freeze escape TPs.

        A stale underwater SHORT with a far TP must be allowed to move its TP
        back toward entry so a same-side recovery move can release inventory.
        """
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="471720", pair="EUR_USD", side="SHORT",
            entry_price=1.16090, current_tp=1.15515,
            current_price=1.16480,
            atr_pips=20.9, reward_risk=2.7,
            is_reversal_firing=True,
            close_review_active=True,
        )

        self.assertIsNotNone(adj)
        self.assertGreater(adj.new_tp, adj.current_tp)
        self.assertLess(adj.new_tp, 1.16090)
        self.assertLess(adj.distance_pips_new, adj.distance_pips_old)

    def test_entry_thesis_blocker_freezes_tp_adjustment(self) -> None:
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="471720", pair="EUR_USD", side="SHORT",
            entry_price=1.16090, current_tp=1.16004,
            current_price=1.15970,
            atr_pips=21.3, reward_risk=2.7,
            entry_thesis_block_active=True,
        )

        self.assertIsNone(adj)

    def test_expand_only_mode_blocks_contraction_when_in_profit(self) -> None:
        """LONG in profit (current > entry), small desired distance
        from contracted regime. Must NOT contract — expand-only rule."""
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="t3", pair="EUR_USD", side="LONG",
            entry_price=1.3000, current_tp=1.3100,
            current_price=1.3010,  # in slight profit
            atr_pips=15, reward_risk=1.5,
        )
        self.assertIsNone(adj)

    def test_regression_471029_in_profit_no_contraction(self) -> None:
        """Exact 471029 incident: entry 1.35077, TP 1.35956 (88pip),
        price 1.35100 (slight profit), low reward_risk. Position is
        NOT significantly adverse → expand_only mode → no contraction."""
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="471029", pair="GBP_USD", side="LONG",
            entry_price=1.35077, current_tp=1.35956,
            current_price=1.35100,
            atr_pips=2.0, reward_risk=1.5,
        )
        self.assertIsNone(adj)

    def test_profitable_short_existing_tp_harvests_when_forecast_and_technicals_fade(self) -> None:
        """Profitable existing TP may contract only when forecast drag and
        technical MFE-risk agree, and the new TP is structurally anchored."""
        self._kill_switch_off()
        pair_chart = {
            "pair": "EUR_USD",
            "views": [
                {
                    "granularity": "M15",
                    "structure": {
                        "liquidity": [
                            {
                                "side": "EQ_LOW",
                                "price": 1.15875,
                                "indices": [1, 2, 3, 4],
                            }
                        ]
                    },
                }
            ],
        }
        adj = compute_tp_adjustment(
            trade_id="471292", pair="EUR_USD", side="SHORT",
            entry_price=1.16077, current_tp=1.15640,
            current_price=1.15947,
            atr_pips=19.2, reward_risk=2.2,
            latest_forecast={"direction": "UNCLEAR", "confidence": 0.24, "horizon_min": 0},
            chart_context={
                "confluence": {
                    "price_percentile_7d": 0.0,
                    "tf_agreement_score": 0.33,
                    "range_24h_sigma_multiple": 6.7,
                },
                "indicators_by_tf": {
                    "M15": {
                        "stoch_rsi": 0.13,
                        "williams_r_14": -85.0,
                        "close": 1.15970,
                        "bb_lower": 1.15972,
                    }
                },
            },
            pair_chart=pair_chart,
        )
        self.assertIsNotNone(adj)
        self.assertLess(adj.new_tp, 1.15947)
        self.assertGreater(adj.new_tp, 1.15640)
        self.assertLess(adj.distance_pips_new, adj.distance_pips_old)
        self.assertIn("forecast_harvest", adj.rationale)
        self.assertIn("forecast UNCLEAR", adj.rationale)
        self.assertIn("HARVEST", adj.rationale)

    def test_profitable_short_existing_tp_keeps_running_without_forecast_drag(self) -> None:
        self._kill_switch_off()
        pair_chart = {
            "pair": "EUR_USD",
            "views": [
                {
                    "granularity": "M15",
                    "structure": {
                        "liquidity": [
                            {
                                "side": "EQ_LOW",
                                "price": 1.15875,
                                "indices": [1, 2, 3, 4],
                            }
                        ]
                    },
                }
            ],
        }
        adj = compute_tp_adjustment(
            trade_id="runner-short", pair="EUR_USD", side="SHORT",
            entry_price=1.16077, current_tp=1.15640,
            current_price=1.15947,
            atr_pips=19.2, reward_risk=2.2,
            latest_forecast={"direction": "DOWN", "confidence": 0.8, "horizon_min": 180},
            chart_context={
                "confluence": {
                    "price_percentile_7d": 0.0,
                    "tf_agreement_score": 0.33,
                    "range_24h_sigma_multiple": 6.7,
                },
                "indicators_by_tf": {"M15": {"stoch_rsi": 0.13}},
            },
            pair_chart=pair_chart,
        )
        self.assertIsNone(adj)

    def test_profitable_short_reversal_keeps_reachable_harvest_tp_under_mfe_risk(self) -> None:
        """A reversal print must not expand a profitable HARVEST TP when the
        same packet says the runner edge is gone.

        Regression for 2026-06-05 AUD_NZD: PositionManager staged HARVEST_TP
        near 1.21364, then tp_rebalancer's expand_reversal path wanted to
        stretch it back to a 29pip runner.
        """
        self._kill_switch_off()
        pair_chart = {
            "pair": "AUD_NZD",
            "views": [
                {
                    "granularity": "M15",
                    "structure": {
                        "liquidity": [
                            {
                                "side": "EQ_LOW",
                                "price": 1.21364,
                                "indices": list(range(21)),
                            }
                        ]
                    },
                }
            ],
        }
        harvest_context = {
            "confluence": {
                "tf_agreement_score": 0.33,
                "range_24h_sigma_multiple": 4.84,
            },
            "indicators_by_tf": {
                "M5": {"atr_pips": 2.1},
                "M30": {"stoch_rsi": 0.20, "williams_r_14": -80.7},
            },
        }

        adj = compute_tp_adjustment(
            trade_id="472037", pair="AUD_NZD", side="SHORT",
            entry_price=1.21445, current_tp=1.21364,
            current_price=1.21420,
            atr_pips=19.5, reward_risk=1.5,
            is_reversal_firing=True,
            latest_forecast={"direction": "UNCLEAR", "confidence": 0.20, "horizon_min": 0},
            chart_context=harvest_context,
            pair_chart=pair_chart,
        )

        self.assertIsNone(adj)

    def test_profitable_short_reversal_contracts_stale_tp_to_harvest_anchor(self) -> None:
        """If the broker TP is already stale-wide, forecast+technical MFE-risk
        should contract it even when a side-aligned reversal signal exists."""
        self._kill_switch_off()
        pair_chart = {
            "pair": "AUD_NZD",
            "views": [
                {
                    "granularity": "M15",
                    "structure": {
                        "liquidity": [
                            {
                                "side": "EQ_LOW",
                                "price": 1.21364,
                                "indices": list(range(21)),
                            }
                        ]
                    },
                }
            ],
        }
        harvest_context = {
            "confluence": {
                "tf_agreement_score": 0.33,
                "range_24h_sigma_multiple": 4.84,
            },
            "indicators_by_tf": {
                "M5": {"atr_pips": 2.1},
                "M30": {"stoch_rsi": 0.20, "williams_r_14": -80.7},
            },
        }

        adj = compute_tp_adjustment(
            trade_id="472037", pair="AUD_NZD", side="SHORT",
            entry_price=1.21445, current_tp=1.21152,
            current_price=1.21420,
            atr_pips=19.5, reward_risk=1.5,
            is_reversal_firing=True,
            latest_forecast={"direction": "UNCLEAR", "confidence": 0.20, "horizon_min": 0},
            chart_context=harvest_context,
            pair_chart=pair_chart,
        )

        self.assertIsNotNone(adj)
        self.assertEqual(adj.new_tp, 1.21364)
        self.assertLess(adj.distance_pips_new, adj.distance_pips_old)
        self.assertIn("forecast_harvest", adj.rationale)
        self.assertIn("forecast UNCLEAR", adj.rationale)

    def test_profitable_short_existing_tp_skips_nearest_anchor_inside_market_safety(self) -> None:
        self._kill_switch_off()
        pair_chart = {
            "pair": "EUR_USD",
            "views": [
                {
                    "granularity": "M15",
                    "structure": {
                        "liquidity": [
                            {
                                "side": "EQ_LOW",
                                "price": 1.15922,
                                "indices": [1, 2, 3, 4],
                            }
                        ]
                    },
                },
                {
                    "granularity": "H1",
                    "structure": {
                        "liquidity": [
                            {
                                "side": "EQ_LOW",
                                "price": 1.15789,
                                "indices": [1, 2, 3, 4],
                            }
                        ]
                    },
                },
            ],
        }
        adj = compute_tp_adjustment(
            trade_id="471292", pair="EUR_USD", side="SHORT",
            entry_price=1.16077, current_tp=1.15640,
            current_price=1.15947,
            atr_pips=19.2, reward_risk=2.2,
            latest_forecast={"direction": "UNCLEAR", "confidence": 0.24, "horizon_min": 0},
            chart_context={
                "confluence": {
                    "price_percentile_7d": 0.0,
                    "tf_agreement_score": 0.33,
                    "range_24h_sigma_multiple": 6.7,
                },
                "indicators_by_tf": {"M15": {"stoch_rsi": 0.13}},
            },
            pair_chart=pair_chart,
        )
        self.assertIsNotNone(adj)
        self.assertEqual(adj.new_tp, 1.15789)
        self.assertIn("next HARVEST anchor", adj.rationale)

    def test_contract_adverse_mode_long_underwater(self) -> None:
        """LONG significantly underwater (≥ 1 ATR) without reversal:
        contract TP to a small lock-in profit on the bounce."""
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="t-adv-long", pair="EUR_USD", side="LONG",
            entry_price=1.3000, current_tp=1.3100,  # 100 pip TP
            current_price=1.2960,  # 40 pip underwater
            atr_pips=20, reward_risk=1.5,
            is_reversal_firing=False,
        )
        self.assertIsNotNone(adj)
        # New TP should be CLOSER to entry than original
        self.assertLess(adj.distance_pips_new, adj.distance_pips_old)
        # But still above entry (lock-in profit)
        self.assertGreater(adj.new_tp, 1.3000)
        self.assertIn("contract_adverse", adj.rationale)

    def test_contract_adverse_mode_short_underwater(self) -> None:
        """SHORT significantly underwater (price ABOVE entry, ≥ 1 ATR)
        without reversal: contract TP toward entry-lock_in."""
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="t-adv-short", pair="USD_JPY", side="SHORT",
            entry_price=160.0, current_tp=159.50,  # 50 pip TP
            current_price=160.30,  # 30 pip underwater
            atr_pips=25, reward_risk=1.5,
            is_reversal_firing=False,
        )
        self.assertIsNotNone(adj)
        self.assertLess(adj.distance_pips_new, adj.distance_pips_old)
        self.assertLess(adj.new_tp, 160.0)  # below entry, SHORT TP

    def test_contracts_stale_short_harvest_tp_when_operating_atr_makes_target_unreachable(self) -> None:
        """A failed-break/recovery SHORT should not keep a 26×M5-ATR TP
        just because the position is only slightly underwater."""
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="471306", pair="EUR_USD", side="SHORT",
            entry_price=1.15960, current_tp=1.15181,
            current_price=1.15974,
            atr_pips=19.7, reward_risk=3.0,
            is_reversal_firing=False,
            chart_context={
                "confluence": {
                    "price_percentile_7d": 0.03,
                    "range_24h_sigma_multiple": 3.57,
                },
                "indicators_by_tf": {
                    "M5": {"atr_pips": 3.0},
                    "M15": {"atr_pips": 5.1},
                },
            },
        )

        self.assertIsNotNone(adj)
        self.assertEqual(adj.new_tp, 1.1588)
        self.assertLess(adj.distance_pips_new, adj.distance_pips_old)
        self.assertIn("contract_stale_harvest", adj.rationale)
        self.assertIn("operating ATR", adj.rationale)

    def test_contracts_profitable_stale_short_tp_to_operating_reward_anchor(self) -> None:
        """A profitable SHORT with a stale full-distance TP re-anchors to a
        reachable reward target on the operating timeframe — NOT to a 5.1-pip
        near-market snap.

        Capture-economics repair (2026-06-10): the old near-market snap
        harvested profitable runners at noise distance, producing the
        +376 JPY average win vs -1,437 JPY average loss ledger asymmetry.
        New candidate = entry ± min(reward_risk × operating_ATR,
        MAX_TP_DISTANCE_ATR_MULT × operating_ATR), floored at the adverse
        lock-in distance. Operating ATR resolves to the M5 reading (6.6),
        so the target is 3.0 × 6.6 = 19.8 pips from entry.
        """
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="471328", pair="EUR_USD", side="SHORT",
            entry_price=1.16272, current_tp=1.14355,
            current_price=1.16209,
            atr_pips=22.7, reward_risk=3.0,
            is_reversal_firing=False,
            chart_context={
                "confluence": {
                    "range_24h_sigma_multiple": 6.43,
                    "score_balance": "TIED",
                },
                "indicators_by_tf": {
                    "M5": {"atr_pips": 6.6},
                    "M15": {"atr_pips": 9.4},
                },
            },
        )

        self.assertIsNotNone(adj)
        self.assertEqual(adj.new_tp, 1.16074)
        self.assertAlmostEqual(adj.distance_pips_new, 19.8, places=1)
        self.assertLess(adj.distance_pips_new, adj.distance_pips_old)
        self.assertIn("contract_profitable_stale_harvest", adj.rationale)
        self.assertIn("profit 6.3pip", adj.rationale)

    def test_profitable_stale_tp_already_past_reanchor_holds_instead_of_market_snap(self) -> None:
        """When profit has already consumed the re-anchored target (within the
        market-safety margin), the sidecar HOLDs the existing TP rather than
        snapping it to market; profit capture belongs to the structural
        forecast_harvest / partial-close paths in that state."""
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="471329", pair="EUR_USD", side="SHORT",
            entry_price=1.16272, current_tp=1.14355,
            # 26 pips in profit with 2+ technical pressure readings → the old
            # code snapped TP to market+5.1; reanchor at 9.4 pips (1.0 rr ×
            # operating ATR... using rr=1.0 → max(lock_in 8, 9.4)=9.4) is
            # already consumed, so the new code returns None.
            current_price=1.16012,
            atr_pips=22.7, reward_risk=1.0,
            is_reversal_firing=False,
            chart_context={
                "confluence": {
                    "range_24h_sigma_multiple": 6.43,
                    "score_balance": "TIED",
                    "price_percentile_24h": 0.02,
                    "tf_agreement_score": 0.33,
                },
                "indicators_by_tf": {
                    "M5": {"atr_pips": 6.6},
                    "M15": {"atr_pips": 9.4},
                },
            },
        )

        self.assertIsNone(adj)

    def test_keeps_reachable_profitable_harvest_tp_instead_of_expanding_again(self) -> None:
        """A near structural harvest TP must not be expanded back into a runner
        while the position has not yet cleared operating ATR."""
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="471328", pair="EUR_USD", side="SHORT",
            entry_price=1.16272, current_tp=1.16163,
            current_price=1.16218,
            atr_pips=22.7, reward_risk=4.2,
            is_reversal_firing=False,
            chart_context={
                "confluence": {"range_24h_sigma_multiple": 6.50},
                "indicators_by_tf": {
                    "M5": {"atr_pips": 6.8},
                    "M15": {"atr_pips": 9.3},
                },
            },
        )

        self.assertIsNone(adj)

    def test_keeps_reachable_harvest_tp_after_micro_adverse_flip(self) -> None:
        """Regression for 471345: PositionManager set a near HARVEST TP, then
        post-gateway tp-rebalance widened it when the next quote tick put the
        short slightly underwater. A reachable harvest TP must survive that
        micro flip; otherwise one cycle can tighten and immediately stretch."""
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="471345", pair="EUR_USD", side="SHORT",
            entry_price=1.16244, current_tp=1.16174,
            current_price=1.16260,
            atr_pips=21.3, reward_risk=2.7,
            is_reversal_firing=False,
            chart_context={
                "confluence": {"range_24h_sigma_multiple": 6.43},
                "indicators_by_tf": {
                    "M5": {"atr_pips": 1.1},
                    "M15": {"atr_pips": 3.8},
                },
            },
        )

        self.assertIsNone(adj)

    def test_keeps_reachable_harvest_tp_after_profit_progress(self) -> None:
        """Regression for 471353→471355: a later TP pass tightened the short to
        a reachable harvest target, then expanded it again after the position
        had made some progress. HARVEST TP means bank, not reclassify as runner."""
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="471345", pair="EUR_USD", side="SHORT",
            entry_price=1.16244, current_tp=1.16164,
            current_price=1.16208,
            atr_pips=21.3, reward_risk=2.7,
            is_reversal_firing=False,
            chart_context={
                "confluence": {"range_24h_sigma_multiple": 6.50},
                "indicators_by_tf": {
                    "M5": {"atr_pips": 1.4},
                    "M15": {"atr_pips": 3.8},
                },
            },
        )

        self.assertIsNone(adj)

    def test_keeps_entry_lock_harvest_tp_after_small_profit_flip(self) -> None:
        """Regression for 471492/471495: a failed-break SHORT was contracted
        to the 8pip lock-in TP while adverse, then a small profit flip expanded
        it back into a distant runner. Entry-lock harvest means bank."""
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="471492", pair="EUR_USD", side="SHORT",
            entry_price=1.16013, current_tp=1.15933,
            current_price=1.15990,
            atr_pips=18.5, reward_risk=2.7,
            is_reversal_firing=False,
            chart_context=None,
        )

        self.assertIsNone(adj)

    def test_trailing_mode_pushes_tp_ahead_of_price(self) -> None:
        """LONG in profit ≥ TRAILING_TRIGGER_ATR_MULT × ATR triggers
        trailing branch: new TP anchored on current_price + lock-behind."""
        self._kill_switch_off()
        # LONG entry 1.3000, current 1.3050 (50 pip profit), ATR 25
        # entry_anchored = entry + rr*atr = 1.3000 + 2.0*25*pip = 1.3050
        # trailing eligible (50 ≥ 1.0 × 25): trailing = current + 1.5*25 = 1.3050 + 37.5pip = 1.30875
        # trailing > entry_anchored → mode = trailing
        adj = compute_tp_adjustment(
            trade_id="t-trail", pair="EUR_USD", side="LONG",
            entry_price=1.3000, current_tp=1.3045,
            current_price=1.3050,
            atr_pips=25, reward_risk=2.0,
        )
        self.assertIsNotNone(adj)
        self.assertGreater(adj.new_tp, adj.current_tp)
        self.assertIn("trailing", adj.rationale)

    def test_trailing_mode_not_eligible_when_profit_below_atr(self) -> None:
        """Profit < 1×ATR → trailing branch disabled, fall through to
        entry-anchored expand_only."""
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="t-no-trail", pair="EUR_USD", side="LONG",
            entry_price=1.3000, current_tp=1.3040,
            current_price=1.3005,  # 5 pip profit
            atr_pips=25, reward_risk=2.0,  # entry_anchored 50pip
        )
        # New TP should be entry-anchored = 1.3050, NOT current+trailing
        self.assertIsNotNone(adj)
        # entry_anchored distance = 50, trailing would be 5 + 37 = 42, so entry wins
        # Just check rationale mode = expand_only (not trailing)
        self.assertNotIn("trailing", adj.rationale)

    def test_reversal_firing_expands_even_when_underwater(self) -> None:
        """LONG underwater BUT reversal signal fires: expand mode wins,
        contraction is skipped. 'Let the bounce run further.'"""
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="t-rev", pair="GBP_USD", side="LONG",
            entry_price=1.3500, current_tp=1.3550,  # 50 pip TP
            current_price=1.3460,  # 40 pip underwater
            atr_pips=25, reward_risk=4.0,  # desired 100 pip
            is_reversal_firing=True,
        )
        self.assertIsNotNone(adj)
        # Expanded, NOT contracted
        self.assertGreater(adj.distance_pips_new, adj.distance_pips_old)
        self.assertIn("expand_reversal", adj.rationale)

    def test_hysteresis_blocks_small_change(self) -> None:
        self._kill_switch_off()
        # Existing 50pip TP, new candidate 51pip → 1pip change < HYSTERESIS
        adj = compute_tp_adjustment(
            trade_id="t4", pair="EUR_USD", side="LONG",
            entry_price=1.3000, current_tp=1.3050,
            current_price=1.3010, atr_pips=20, reward_risk=2.55,  # 51 pip
        )
        self.assertIsNone(adj)

    def test_manual_owned_can_adjust_take_profit(self) -> None:
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="t5", pair="EUR_USD", side="LONG",
            entry_price=1.30, current_tp=1.305,
            current_price=1.305, atr_pips=30, reward_risk=3.0,
            owner="manual",
        )
        self.assertIsNotNone(adj)
        self.assertGreater(adj.new_tp, 1.305)

    def test_external_owned_is_not_adjusted(self) -> None:
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="t5-external", pair="EUR_USD", side="LONG",
            entry_price=1.30, current_tp=1.31,
            current_price=1.305, atr_pips=30, reward_risk=3.0,
            owner="external",
        )
        self.assertIsNone(adj)

    def test_skips_position_without_tp(self) -> None:
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="t6", pair="EUR_USD", side="LONG",
            entry_price=1.30, current_tp=None,
            current_price=1.305, atr_pips=30, reward_risk=3.0,
        )
        self.assertIsNone(adj)

    def test_manual_position_without_tp_preserves_no_broker_tp_by_default(self) -> None:
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="manual-no-tp", pair="EUR_USD", side="LONG",
            entry_price=1.3000, current_tp=None,
            current_price=1.3010, atr_pips=20, reward_risk=2.0,
            owner="unknown",
        )
        self.assertIsNone(adj)

    def test_manual_position_without_tp_gets_take_profit_repair_when_enabled(self) -> None:
        self._kill_switch_off()
        os.environ["QR_ENABLE_MISSING_TP_REPAIR"] = "1"
        adj = compute_tp_adjustment(
            trade_id="manual-no-tp", pair="EUR_USD", side="LONG",
            entry_price=1.3000, current_tp=None,
            current_price=1.3010, atr_pips=20, reward_risk=2.0,
            owner="unknown",
        )
        self.assertIsNotNone(adj)
        self.assertIsNone(adj.current_tp)
        self.assertGreater(adj.new_tp, 1.3010)
        self.assertIn("manual_tp_repair", adj.rationale)

    def test_profit_runner_without_tp_gets_insurance_when_forecast_cannot_reach_next_session(self) -> None:
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="runner-low-forecast", pair="EUR_USD", side="LONG",
            entry_price=1.3000, current_tp=None,
            current_price=1.3050, atr_pips=20, reward_risk=3.0,
            owner="trader",
            latest_forecast={"direction": "UNCLEAR", "confidence": 0.1, "horizon_min": 0},
            chart_context={
                "confluence": {"tf_agreement_score": 1.0},
                "session": {"minutes_to_next_killzone": 45},
                "indicators_by_tf": {},
            },
        )
        self.assertIsNotNone(adj)
        self.assertEqual(adj.current_tp, None)
        self.assertGreater(adj.new_tp, 1.3050)
        self.assertIn("insurance_tp", adj.rationale)
        self.assertIn("forecast UNCLEAR", adj.rationale)

    def test_profit_runner_without_tp_keeps_running_when_forecast_is_strong_and_no_pressure(self) -> None:
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="runner-strong", pair="EUR_USD", side="LONG",
            entry_price=1.3000, current_tp=None,
            current_price=1.3050, atr_pips=20, reward_risk=3.0,
            owner="trader",
            latest_forecast={"direction": "UP", "confidence": 0.8, "horizon_min": 180},
            chart_context={
                "confluence": {"tf_agreement_score": 1.0, "range_24h_sigma_multiple": 1.0},
                "session": {"minutes_to_next_killzone": 45},
                "indicators_by_tf": {"M15": {"rsi_14": 55, "stoch_rsi": 0.5}},
            },
        )
        self.assertIsNone(adj)

    def test_profit_runner_without_tp_ignores_single_technical_warning(self) -> None:
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="runner-one-warning", pair="EUR_USD", side="LONG",
            entry_price=1.3000, current_tp=None,
            current_price=1.3050, atr_pips=20, reward_risk=3.0,
            owner="trader",
            latest_forecast={"direction": "UP", "confidence": 0.8, "horizon_min": 180},
            chart_context={
                "confluence": {"price_percentile_24h": 0.96, "tf_agreement_score": 1.0},
                "session": {"minutes_to_next_killzone": 45},
                "indicators_by_tf": {"M15": {"rsi_14": 55, "stoch_rsi": 0.5}},
            },
        )
        self.assertIsNone(adj)

    def test_profit_runner_without_tp_gets_insurance_on_technical_exhaustion(self) -> None:
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="runner-exhausted", pair="EUR_USD", side="LONG",
            entry_price=1.3000, current_tp=None,
            current_price=1.3050, atr_pips=20, reward_risk=3.0,
            owner="trader",
            latest_forecast={"direction": "UP", "confidence": 0.8, "horizon_min": 180},
            chart_context={
                "confluence": {"price_percentile_24h": 0.96, "tf_agreement_score": 1.0},
                "session": {"minutes_to_next_killzone": 45},
                "indicators_by_tf": {"M15": {"rsi_14": 72}},
            },
        )
        self.assertIsNotNone(adj)
        self.assertIn("insurance_tp", adj.rationale)
        self.assertIn("price percentile", adj.rationale)

    def test_long_tp_never_below_entry(self) -> None:
        """Even with extreme contraction, LONG TP must stay > entry."""
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="t7", pair="EUR_USD", side="LONG",
            entry_price=1.3000, current_tp=1.3050,
            current_price=1.2950,  # underwater
            atr_pips=5, reward_risk=1.5,  # 7.5 pip desired
        )
        if adj is not None:
            self.assertGreater(adj.new_tp, 1.3000)  # above entry

    def test_short_tp_never_above_entry(self) -> None:
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="t8", pair="USD_JPY", side="SHORT",
            entry_price=160.00, current_tp=159.50,
            current_price=160.50,  # underwater for SHORT
            atr_pips=10, reward_risk=1.5,
        )
        if adj is not None:
            self.assertLess(adj.new_tp, 160.00)

    def test_safety_margin_from_current_price_long(self) -> None:
        """LONG TP must stay above current price + safety margin so it
        doesn't fire on the same tick."""
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="t9", pair="EUR_USD", side="LONG",
            entry_price=1.3000, current_tp=1.3100,
            current_price=1.3080,  # close to current TP
            atr_pips=10, reward_risk=1.5,  # 15 pip → candidate 1.3015 (below current!)
        )
        # candidate below current would be raised to current+safety, but
        # then it must stay > entry (already true here). Either we get
        # adj with new_tp >= current+safety, or None.
        if adj is not None:
            self.assertGreaterEqual(adj.new_tp, 1.3080)

    def test_kill_switch_disables(self) -> None:
        os.environ["QR_DISABLE_TP_REBALANCE"] = "1"
        try:
            adj = compute_tp_adjustment(
                trade_id="t-k", pair="EUR_USD", side="LONG",
                entry_price=1.30, current_tp=1.31,
                current_price=1.305, atr_pips=30, reward_risk=3.0,
            )
            self.assertIsNone(adj)
        finally:
            del os.environ["QR_DISABLE_TP_REBALANCE"]

    def test_zero_atr_returns_none(self) -> None:
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="t-z", pair="EUR_USD", side="LONG",
            entry_price=1.30, current_tp=1.31,
            current_price=1.305, atr_pips=0, reward_risk=3.0,
        )
        self.assertIsNone(adj)


class ChartContextExtractionTest(unittest.TestCase):
    def test_atr_from_confluence_h4(self) -> None:
        chart = {"pair": "EUR_USD", "confluence": {"h4_atr_pips": 25}}
        self.assertEqual(_extract_atr_pips(chart, "EUR_USD"), 25.0)

    def test_atr_fallback_to_views_indicators(self) -> None:
        chart = {
            "pair": "EUR_USD",
            "confluence": {},
            "views": [
                {"granularity": "H1", "indicators": {"atr_pips": 18}},
                {"granularity": "H4", "indicators": {"atr_pips": 30}},
            ],
        }
        # Preference order picks H4 first.
        self.assertEqual(_extract_atr_pips(chart, "EUR_USD"), 30.0)

    def test_chart_context_extracts_h1_adx(self) -> None:
        chart = {
            "pair": "EUR_USD",
            "confluence": {"atr_percentile_24h": 0.6},
            "views": [{"granularity": "H1", "indicators": {"adx_14": 28}}],
        }
        ctx = _chart_context_from_chart(chart)
        self.assertEqual(ctx["h1_adx"], 28)
        self.assertEqual(ctx["confluence"]["atr_percentile_24h"], 0.6)


class CloseReviewSidecarTest(unittest.TestCase):
    def test_loads_recent_review_close_trade_ids(self) -> None:
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "position_thesis_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": now.isoformat(),
                        "assessments": [
                            {"trade_id": "471720", "verdict": "REVIEW_CLOSE"},
                            {"trade_id": "471232", "verdict": "HOLD"},
                        ],
                    }
                )
            )
            (root / "thesis_evolution_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": now.isoformat(),
                        "evolutions": [
                            {"trade_id": "471717", "status": "BROKEN", "verdict": "RECOMMEND_CLOSE"}
                        ],
                    }
                )
            )

            self.assertEqual(
                load_close_review_trade_ids(root, now=now),
                {"471720", "471717"},
            )

    def test_loads_recent_entry_thesis_blocker_trade_ids(self) -> None:
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "thesis_evolution_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": now.isoformat(),
                        "evolutions": [
                            {
                                "trade_id": "471910",
                                "status": "UNVERIFIABLE",
                                "verdict": "REQUIRE_THESIS_REPAIR",
                            },
                            {"trade_id": "471817", "status": "WEAKENED", "verdict": "HOLD"},
                        ],
                    }
                )
            )

            self.assertEqual(load_entry_thesis_blocker_trade_ids(root, now=now), {"471910"})

    def test_ignores_stale_review_close_sidecars(self) -> None:
        now = datetime(2026, 5, 28, 14, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "position_thesis_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-05-28T12:00:00+00:00",
                        "assessments": [{"trade_id": "471720", "verdict": "REVIEW_CLOSE"}],
                    }
                )
            )

            self.assertEqual(load_close_review_trade_ids(root, now=now), set())


class ApplyTPAdjustmentsTest(unittest.TestCase):
    def test_dry_run_does_not_call_broker(self) -> None:
        adj = TPAdjustment(
            trade_id="t1", pair="EUR_USD", side="LONG",
            entry_price=1.30, current_tp=1.31, new_tp=1.32,
            distance_pips_old=100, distance_pips_new=200, rationale="test",
        )

        class MockClient:
            calls = []
            def replace_trade_dependent_orders(self, *a, **kw):
                self.calls.append((a, kw))

        client = MockClient()
        results = apply_tp_adjustments([adj], client, dry_run=True)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["sent"])
        self.assertEqual(client.calls, [])

    def test_apply_calls_broker(self) -> None:
        adj = TPAdjustment(
            trade_id="t2", pair="USD_JPY", side="SHORT",
            entry_price=160.0, current_tp=159.7, new_tp=159.5,
            distance_pips_old=30, distance_pips_new=50, rationale="test",
        )

        class MockClient:
            calls = []
            def replace_trade_dependent_orders(self, trade_id, payload):
                self.calls.append((trade_id, payload))
                return {"ok": True}

        client = MockClient()
        results = apply_tp_adjustments([adj], client, dry_run=False)
        self.assertTrue(results[0]["sent"])
        self.assertIsNone(results[0]["error"])
        self.assertEqual(client.calls[0][0], "t2")
        self.assertIn("takeProfit", client.calls[0][1])

    def test_broker_exception_captured_per_adjustment(self) -> None:
        adj = TPAdjustment(
            trade_id="t3", pair="EUR_USD", side="LONG",
            entry_price=1.30, current_tp=1.31, new_tp=1.32,
            distance_pips_old=100, distance_pips_new=200, rationale="test",
        )

        class FlakyClient:
            def replace_trade_dependent_orders(self, *a, **kw):
                raise RuntimeError("broker timeout")

        client = FlakyClient()
        results = apply_tp_adjustments([adj], client, dry_run=False)
        self.assertFalse(results[0]["sent"])
        self.assertEqual(results[0]["error"], "broker timeout")


if __name__ == "__main__":
    unittest.main()
