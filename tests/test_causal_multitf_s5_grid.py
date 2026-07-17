from __future__ import annotations

import copy
import re
import unittest
from datetime import datetime, timedelta, timezone

from quant_rabbit import causal_multitf_s5_grid as grid_core
from quant_rabbit.causal_multitf_s5_grid import (
    UtcSplit,
    build_predeclared_arms_v1,
    build_predeclared_catalog_v1,
    combine_causal_multitf_s5_grid_runs,
    run_causal_multitf_s5_grid,
)
from quant_rabbit.technical_forecast_forward_outcome import S5BidAskCandle


UTC = timezone.utc
# H05's declared regime role uses the latest M15 feature.  Thirty completed
# M15 buckets give ATR/ADX their full 14+14 Wilder warm-up before the signal.
FAILURE_MINUTE = 450
ENTRY_MINUTE = FAILURE_MINUTE + 1


def _candle(
    timestamp: datetime,
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
    spread: float = 0.0002,
) -> S5BidAskCandle:
    half = spread / 2.0
    return S5BidAskCandle(
        timestamp_utc=timestamp,
        bid_o=open_ - half,
        bid_h=high - half,
        bid_l=low - half,
        bid_c=close - half,
        ask_o=open_ + half,
        ask_h=high + half,
        ask_l=low + half,
        ask_c=close + half,
    )


def _failure_path(
    start: datetime,
    *,
    spread: float = 0.0002,
    entry_candle: S5BidAskCandle | None = None,
) -> list[S5BidAskCandle]:
    rows = [
        _candle(
            start + timedelta(minutes=index),
            open_=1.0,
            high=1.0005,
            low=0.9995,
            close=1.0,
            spread=spread,
        )
        for index in range(FAILURE_MINUTE)
    ]
    # This completed M1 sweeps the preceding range high and closes back inside.
    rows.append(
        _candle(
            start + timedelta(minutes=FAILURE_MINUTE),
            open_=1.0,
            high=1.0008,
            low=0.9997,
            close=1.0002,
            spread=spread,
        )
    )
    rows.append(
        entry_candle
        or _candle(
            start + timedelta(minutes=ENTRY_MINUTE),
            open_=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            spread=spread,
        )
    )
    return rows


def _splits(start: datetime) -> tuple[UtcSplit, ...]:
    return (
        UtcSplit("VALIDATION", start, start + timedelta(hours=12)),
        UtcSplit(
            "HOLDOUT",
            start + timedelta(hours=12),
            start + timedelta(hours=24),
        ),
    )


def _candidate(result: dict, candidate_id: str) -> dict:
    return next(
        row
        for row in result["candidate_metrics"]
        if row["candidate_id"] == candidate_id
    )


class CausalMultiTfS5GridTest(unittest.TestCase):
    def test_exact_sign_flip_does_not_overstate_eight_day_evidence(self) -> None:
        p_value, _se, method = grid_core._one_sided_p_and_se(
            [1.0, 9.0] * 4,
            seed_key="eight-day-support",
        )

        self.assertEqual(method, "UTC_DAY_EXACT_SIGN_FLIP_V1")
        self.assertEqual(p_value, 1.0 / 256.0)
        self.assertGreater(p_value * 182, 0.05)

    def test_one_se_best_mean_tie_uses_lower_se_before_candidate_id(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=UTC)
        days = [
            (start + timedelta(days=index)).date().isoformat() for index in range(8)
        ]
        candidates = []
        daily_rows = []
        fixtures = {
            "H01:DIRECT:BASE": ([4.4, 4.6] * 4, [0, 0, 0, 0]),
            "H01:DIRECT:TP075": ([4.9, 5.1] * 4, [0, 1, 0.25, "TP075"]),
            "H01:DIRECT:TP125": ([1.0, 9.0] * 4, [0, 1, 0.25, "TP125"]),
        }
        for candidate_id, (returns, simplicity_key) in fixtures.items():
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "hypothesis_id": "H01",
                    "complexity": 1,
                    "simplicity_key": simplicity_key,
                    "metrics_by_split": {
                        "VALIDATION": {
                            "average_net_pips": sum(returns) / len(returns),
                            "average_net_r": sum(returns) / len(returns),
                            "gross_profit_r": sum(returns),
                            "gross_loss_r": 0.0,
                            "unresolved_count": 0,
                            "purged_count": 0,
                        }
                    },
                }
            )
            daily_rows.extend(
                {
                    "candidate_id": candidate_id,
                    "split": "VALIDATION",
                    "utc_day": day,
                    "exact_net_pips": value,
                    "exact_net_r": value,
                }
                for day, value in zip(days, returns)
            )

        selection = grid_core._select_validation(
            candidates,
            daily_rows,
            "VALIDATION",
            {"VALIDATION": days},
        )

        self.assertEqual(selection["winner_arm_ids"], ["H01:DIRECT:TP075"])

    def test_catalog_and_ofat_grid_are_exactly_182_candidates(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=UTC)
        result = run_causal_multitf_s5_grid("EUR_USD", [], _splits(start))

        self.assertEqual(
            [item.hypothesis_id for item in build_predeclared_catalog_v1()],
            [f"H{i:02d}" for i in range(1, 9)],
        )
        self.assertEqual(len(build_predeclared_arms_v1()), 13)
        self.assertEqual(result["candidate_count"], 182)
        self.assertEqual(
            len({row["candidate_id"] for row in result["candidate_metrics"]}),
            182,
        )

    def test_partial_bucket_and_future_s5_mutation_do_not_change_frozen_signal(
        self,
    ) -> None:
        start = datetime(2026, 1, 1, tzinfo=UTC)
        baseline = _failure_path(start)
        mutated = list(baseline)
        mutated[-1] = _candle(
            start + timedelta(minutes=ENTRY_MINUTE),
            open_=1.0,
            high=1.01,
            low=0.99,
            close=1.007,
        )

        first = run_causal_multitf_s5_grid("EUR_USD", baseline, _splits(start))
        second = run_causal_multitf_s5_grid("EUR_USD", mutated, _splits(start))

        frozen_first = [
            row
            for row in first["signal_rows"]
            if row["activation_at_utc"]
            <= (start + timedelta(minutes=ENTRY_MINUTE)).isoformat()
        ]
        frozen_second = [
            row
            for row in second["signal_rows"]
            if row["activation_at_utc"]
            <= (start + timedelta(minutes=ENTRY_MINUTE)).isoformat()
        ]
        self.assertEqual(frozen_first, frozen_second)
        self.assertTrue(any(row["hypothesis_id"] == "H05" for row in frozen_first))
        self.assertEqual(
            first["aggregation"]["completed_bucket_counts"]["M5"], FAILURE_MINUTE // 5
        )
        self.assertNotIn(
            (start + timedelta(minutes=FAILURE_MINUTE + 5)).isoformat(),
            first["aggregation"]["completed_bucket_clocks"]["M5"],
        )
        self.assertEqual(first["aggregation"]["synthetic_s5_count"], 0)

    def test_no_tick_gap_never_exposes_future_completed_feature_to_signal(
        self,
    ) -> None:
        start = datetime(2026, 1, 1, tzinfo=UTC)
        rows = _failure_path(start)[:-1]
        # The H05 M1 failure bar completes at 07:31.  The next real S5 does not
        # arrive until 07:36, after the containing M5 bucket's 07:35 boundary.
        # A causal evaluator must still use only features complete by 07:31.
        rows.append(
            _candle(
                start + timedelta(minutes=ENTRY_MINUTE + 5),
                open_=1.0,
                high=1.0,
                low=1.0,
                close=1.0,
            )
        )

        result = run_causal_multitf_s5_grid("EUR_USD", rows, _splits(start))
        signal = next(
            row
            for row in result["signal_rows"]
            if row["hypothesis_id"] == "H05"
            and row["activation_at_utc"]
            == (start + timedelta(minutes=ENTRY_MINUTE)).isoformat()
        )
        activation = datetime.fromisoformat(signal["activation_at_utc"])

        self.assertTrue(signal["feature_completed_at_utc"])
        self.assertTrue(
            all(
                datetime.fromisoformat(completed_at) <= activation
                for completed_at in signal["feature_completed_at_utc"].values()
            ),
            signal["feature_completed_at_utc"],
        )

    def test_split_end_embargo_prevents_outcome_dependent_late_censoring(
        self,
    ) -> None:
        start = datetime(2026, 1, 1, tzinfo=UTC)
        rows = _failure_path(start)
        rows.append(
            _candle(
                start + timedelta(minutes=ENTRY_MINUTE + 1),
                open_=1.01,
                high=1.011,
                low=0.999,
                close=1.01,
            )
        )
        truncated_splits = (
            UtcSplit(
                "VALIDATION",
                start,
                start + timedelta(minutes=ENTRY_MINUTE + 1),
            ),
            UtcSplit(
                "HOLDOUT",
                start + timedelta(minutes=ENTRY_MINUTE + 2),
                start + timedelta(hours=24),
            ),
        )

        truncated = run_causal_multitf_s5_grid("EUR_USD", rows, truncated_splits)
        extended = run_causal_multitf_s5_grid("EUR_USD", rows, _splits(start))
        truncated_metric = _candidate(truncated, "H05:DIRECT:BASE")["metrics_by_split"][
            "VALIDATION"
        ]
        extended_metric = _candidate(extended, "H05:DIRECT:BASE")["metrics_by_split"][
            "VALIDATION"
        ]

        # Every arm shares the preregistered maximum horizon embargo:
        # max(TTL + HOLD) = 180 + 1800 = 1980 seconds.  The raw opportunity
        # remains auditable, but it must never fill and disappear as a
        # zero-P/L split purge merely because its losing outcome is too late.
        self.assertEqual(truncated_metric["raw_signal_count"], 1)
        self.assertEqual(truncated_metric["signal_count"], 0)
        self.assertEqual(truncated_metric["filled_count"], 0)
        self.assertEqual(truncated_metric["resolved_count"], 0)
        self.assertEqual(truncated_metric["purged_count"], 0)
        self.assertEqual(truncated_metric["reason_counts"].get("SPLIT_END_EMBARGO"), 1)
        self.assertEqual(extended_metric["filled_count"], 1)
        self.assertEqual(extended_metric["resolved_count"], 1)
        self.assertEqual(extended_metric["purged_count"], 0)
        self.assertEqual(extended_metric["reason_counts"]["STOP_LOSS_GAP"], 1)
        self.assertLess(extended_metric["exact_net_pips"], 0.0)

    def test_open_gap_is_resolved_before_later_same_s5_dual_touch(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=UTC)
        rows = _failure_path(start)
        # H05 DIRECT is SHORT.  The next real S5 opens beyond its TP, then its
        # range spans both barriers.  Chronology requires the open gap to win;
        # SL-first applies only after no barrier was already crossed at open.
        rows.append(
            _candle(
                start + timedelta(minutes=ENTRY_MINUTE, seconds=5),
                open_=0.9985,
                high=1.003,
                low=0.997,
                close=1.0,
            )
        )

        result = run_causal_multitf_s5_grid("EUR_USD", rows, _splits(start))
        trade = next(
            row
            for row in result["trade_rows"]
            if row["candidate_id"] == "H05:DIRECT:BASE"
        )
        self.assertRegex(trade["reason"], r"^TAKE_PROFIT.*GAP")
        self.assertGreater(trade["exact_net_pips"], 0.0)

    def test_expired_old_split_position_cannot_deoverlap_new_split_signal(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=UTC)
        rows = _failure_path(start)
        rows.append(
            _candle(
                start + timedelta(minutes=ENTRY_MINUTE + 1),
                open_=1.0002,
                high=1.0009,
                low=0.9998,
                close=1.0003,
            )
        )
        rows.append(
            _candle(
                start + timedelta(minutes=ENTRY_MINUTE + 2),
                open_=1.0003,
                high=1.0003,
                low=1.0003,
                close=1.0003,
            )
        )
        splits = (
            UtcSplit("TRAIN", start, start + timedelta(minutes=ENTRY_MINUTE + 2)),
            UtcSplit(
                "VALIDATION",
                start + timedelta(minutes=ENTRY_MINUTE + 2),
                start + timedelta(hours=12),
            ),
            UtcSplit(
                "HOLDOUT", start + timedelta(hours=12), start + timedelta(hours=24)
            ),
        )

        result = run_causal_multitf_s5_grid("EUR_USD", rows, splits)
        metric = _candidate(result, "H05:DIRECT:SL150")["metrics_by_split"][
            "VALIDATION"
        ]
        self.assertEqual(metric["raw_signal_count"], 1)
        self.assertEqual(metric["signal_count"], 1)
        self.assertEqual(metric["deoverlap_count"], 0)

    def test_holdout_mutation_cannot_change_validation_selection_receipt(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=UTC)
        splits = (
            UtcSplit("VALIDATION", start, start + timedelta(days=30)),
            UtcSplit("HOLDOUT", start + timedelta(days=30), start + timedelta(days=40)),
        )
        positive = self._synthetic_pair_result(start, splits, holdout_net=100.0)
        negative = self._synthetic_pair_result(start, splits, holdout_net=-100.0)

        first = combine_causal_multitf_s5_grid_runs([positive], splits)
        second = combine_causal_multitf_s5_grid_runs([negative], splits)

        self.assertEqual(
            first["validation_winner_arm_ids"], second["validation_winner_arm_ids"]
        )
        self.assertEqual(
            first["selection_receipt_sha256"], second["selection_receipt_sha256"]
        )
        self.assertTrue(first["holdout_selection_unchanged"])
        self.assertTrue(second["holdout_selection_unchanged"])
        self.assertRegex(
            first["selection_receipt_sha256"], re.compile(r"^[0-9a-f]{64}$")
        )

    def test_global_holdout_is_redacted_to_winners_and_has_portfolio_gate(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=UTC)
        splits = (
            UtcSplit("VALIDATION", start, start + timedelta(days=30)),
            UtcSplit("HOLDOUT", start + timedelta(days=30), start + timedelta(days=40)),
        )
        result = combine_causal_multitf_s5_grid_runs(
            [self._synthetic_pair_result(start, splits, holdout_net=5.0)],
            splits,
        )
        winners = set(result["validation_winner_arm_ids"])
        self.assertTrue(winners)
        for row in result["candidate_metrics"]:
            has_holdout = "HOLDOUT" in row["metrics_by_split"]
            self.assertEqual(has_holdout, row["candidate_id"] in winners)
        self.assertTrue(
            all(
                row["split"] != "HOLDOUT" or row["candidate_id"] in winners
                for row in result["daily_aggregates"]
            )
        )
        for key in ("validation_portfolio", "holdout_portfolio"):
            portfolio = result[key]
            self.assertEqual(set(portfolio["candidate_ids"]), winners)
            self.assertIn("average_daily_net_pips", portfolio)
            self.assertIn("one_sided_95_lower_bound_daily_net_pips", portfolio)
            self.assertIsInstance(portfolio["pass_checks"], dict)
            self.assertIsInstance(portfolio["passed"], bool)

    def test_mixed_pair_selection_and_portfolio_use_equal_risk_r(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=UTC)
        splits = (
            UtcSplit("VALIDATION", start, start + timedelta(days=20)),
            UtcSplit(
                "HOLDOUT",
                start + timedelta(days=20),
                start + timedelta(days=40),
            ),
        )
        usd_jpy_pips = [9.95 if index % 2 == 0 else 10.05 for index in range(20)]
        eur_usd_pips = [-9.0] * 20

        # Raw pips say +1/day, while equal-initial-risk returns say -0.5R/day:
        # the winning JPY leg risks 20 pips and the losing EUR/USD leg risks
        # only 9 pips.  A cross-pair selector must not treat those pips as one
        # homogeneous return unit.
        rejected = combine_causal_multitf_s5_grid_runs(
            [
                self._synthetic_mixed_pair_result(
                    "USD_JPY",
                    splits,
                    validation_pips=usd_jpy_pips,
                    validation_r=[value / 20.0 for value in usd_jpy_pips],
                    holdout_pips=usd_jpy_pips,
                    holdout_r=[value / 20.0 for value in usd_jpy_pips],
                ),
                self._synthetic_mixed_pair_result(
                    "EUR_USD",
                    splits,
                    validation_pips=eur_usd_pips,
                    validation_r=[-1.0] * 20,
                    holdout_pips=eur_usd_pips,
                    holdout_r=[-1.0] * 20,
                ),
            ],
            splits,
        )
        candidate_id = "H01:DIRECT:BASE"
        rejected_test = next(
            row
            for row in rejected["validation"]["multiple_testing"]["candidate_tests"]
            if row["candidate_id"] == candidate_id
        )
        self.assertAlmostEqual(rejected_test["average_daily_net_pips"], 1.0)
        self.assertAlmostEqual(rejected_test["average_daily_net_r"], -0.5)
        self.assertFalse(rejected_test["positive_mean_pass"])
        self.assertNotIn(candidate_id, rejected["selected_arm_ids"])
        self.assertEqual(
            rejected["validation"]["multiple_testing"]["selection_return_unit"],
            "EQUAL_INITIAL_RISK_R",
        )

        # Seal the same candidate on genuinely positive validation R, then make
        # only its holdout R negative while raw pips remain positive.  This
        # isolates the portfolio gate from selection and proves it also uses R.
        sealed = combine_causal_multitf_s5_grid_runs(
            [
                self._synthetic_mixed_pair_result(
                    "USD_JPY",
                    splits,
                    validation_pips=usd_jpy_pips,
                    validation_r=[value / 10.0 for value in usd_jpy_pips],
                    holdout_pips=usd_jpy_pips,
                    holdout_r=[value / 20.0 for value in usd_jpy_pips],
                ),
                self._synthetic_mixed_pair_result(
                    "EUR_USD",
                    splits,
                    validation_pips=eur_usd_pips,
                    validation_r=[-0.4] * 20,
                    holdout_pips=eur_usd_pips,
                    holdout_r=[-1.0] * 20,
                ),
            ],
            splits,
        )
        self.assertIn(candidate_id, sealed["selected_arm_ids"])
        holdout = sealed["holdout_portfolio"]
        self.assertEqual(holdout["return_unit_for_pass"], "EQUAL_INITIAL_RISK_R")
        self.assertAlmostEqual(holdout["average_daily_net_pips"], 1.0)
        self.assertGreater(holdout["one_sided_95_lower_bound_daily_net_pips"], 0.0)
        self.assertAlmostEqual(holdout["average_daily_net_r"], -0.5)
        self.assertLess(holdout["one_sided_95_lower_bound_daily_net_r"], 0.0)
        self.assertFalse(holdout["passed"])

    def _synthetic_mixed_pair_result(
        self,
        pair: str,
        splits: tuple[UtcSplit, ...],
        *,
        validation_pips: list[float],
        validation_r: list[float],
        holdout_pips: list[float],
        holdout_r: list[float],
    ) -> dict:
        if not (
            len(validation_pips) == len(validation_r)
            and len(holdout_pips) == len(holdout_r)
        ):
            raise ValueError("synthetic pips and R rows must align")
        result = run_causal_multitf_s5_grid(pair, [], splits)
        candidate_id = "H01:DIRECT:BASE"
        candidate = _candidate(result, candidate_id)
        daily_rows: list[dict] = []
        observed_days: dict[str, list[str]] = {}

        for split_name, pips_values, r_values in (
            ("VALIDATION", validation_pips, validation_r),
            ("HOLDOUT", holdout_pips, holdout_r),
        ):
            split = next(item for item in splits if item.name == split_name)
            days = [
                (split.from_utc + timedelta(days=index)).date().isoformat()
                for index in range(len(pips_values))
            ]
            observed_days[split_name] = days
            gross_profit_pips = sum(max(0.0, value) for value in pips_values)
            gross_loss_pips = sum(max(0.0, -value) for value in pips_values)
            gross_profit_r = sum(max(0.0, value) for value in r_values)
            gross_loss_r = sum(max(0.0, -value) for value in r_values)
            metric = candidate["metrics_by_split"][split_name]
            metric.update(
                raw_signal_count=len(pips_values),
                signal_count=len(pips_values),
                filled_count=len(pips_values),
                resolved_count=len(pips_values),
                win_count=sum(value > 0.0 for value in pips_values),
                loss_count=sum(value < 0.0 for value in pips_values),
                flat_count=sum(value == 0.0 for value in pips_values),
                gross_mid_pips=sum(pips_values),
                spread_drag_pips=0.0,
                exact_net_pips=sum(pips_values),
                average_net_pips=(sum(pips_values) / len(pips_values)),
                gross_mid_r=sum(r_values),
                spread_drag_r=0.0,
                exact_net_r=sum(r_values),
                average_net_r=(sum(r_values) / len(r_values)),
                gross_profit_pips=gross_profit_pips,
                gross_loss_pips=gross_loss_pips,
                gross_profit_r=gross_profit_r,
                gross_loss_r=gross_loss_r,
                profit_factor=(
                    gross_profit_pips / gross_loss_pips
                    if gross_loss_pips > 0.0
                    else None
                ),
                profit_factor_r=(
                    gross_profit_r / gross_loss_r if gross_loss_r > 0.0 else None
                ),
                win_rate=(sum(value > 0.0 for value in pips_values) / len(pips_values)),
                active_day_count=len(pips_values),
                reason_counts={"SYNTHETIC_MIXED_PAIR": len(pips_values)},
            )
            daily_rows.extend(
                {
                    "pair": pair,
                    "candidate_id": candidate_id,
                    "split": split_name,
                    "utc_day": day,
                    "resolved_count": 1,
                    "win_count": int(pips_value > 0.0),
                    "loss_count": int(pips_value < 0.0),
                    "flat_count": int(pips_value == 0.0),
                    "gross_mid_pips": pips_value,
                    "spread_drag_pips": 0.0,
                    "exact_net_pips": pips_value,
                    "gross_mid_r": r_value,
                    "spread_drag_r": 0.0,
                    "exact_net_r": r_value,
                    "gross_profit_pips": max(0.0, pips_value),
                    "gross_loss_pips": max(0.0, -pips_value),
                    "gross_profit_r": max(0.0, r_value),
                    "gross_loss_r": max(0.0, -r_value),
                    "max_drawdown_input_pips": max(0.0, -pips_value),
                    "reason_counts": {"SYNTHETIC_MIXED_PAIR": 1},
                }
                for day, pips_value, r_value in zip(days, pips_values, r_values)
            )
        result["daily_aggregates"] = daily_rows
        result["aggregation"]["observed_utc_days_by_split"] = observed_days
        return copy.deepcopy(result)

    def _synthetic_pair_result(
        self,
        start: datetime,
        splits: tuple[UtcSplit, ...],
        *,
        holdout_net: float,
    ) -> dict:
        result = run_causal_multitf_s5_grid("EUR_USD", [], splits)
        candidate_id = "H01:DIRECT:BASE"
        candidate = _candidate(result, candidate_id)
        validation = candidate["metrics_by_split"]["VALIDATION"]
        validation.update(
            raw_signal_count=20,
            signal_count=20,
            resolved_count=20,
            filled_count=20,
            win_count=20,
            loss_count=0,
            exact_net_pips=210.0,
            gross_mid_pips=230.0,
            spread_drag_pips=20.0,
            average_net_pips=10.5,
            exact_net_r=210.0,
            gross_mid_r=230.0,
            spread_drag_r=20.0,
            average_net_r=10.5,
            gross_profit_pips=210.0,
            gross_loss_pips=0.0,
            gross_profit_r=210.0,
            gross_loss_r=0.0,
            profit_factor=None,
            profit_factor_r=None,
            win_rate=1.0,
            active_day_count=20,
            max_drawdown_pips=0.0,
            reason_counts={"SYNTHETIC_VALIDATION": 20},
        )
        holdout = candidate["metrics_by_split"]["HOLDOUT"]
        holdout.update(
            raw_signal_count=1,
            signal_count=1,
            resolved_count=1,
            filled_count=1,
            win_count=int(holdout_net > 0),
            loss_count=int(holdout_net < 0),
            exact_net_pips=holdout_net,
            gross_mid_pips=holdout_net + 1.0,
            spread_drag_pips=1.0,
            average_net_pips=holdout_net,
            exact_net_r=holdout_net,
            gross_mid_r=holdout_net + 1.0,
            spread_drag_r=1.0,
            average_net_r=holdout_net,
            gross_profit_pips=max(0.0, holdout_net),
            gross_loss_pips=max(0.0, -holdout_net),
            gross_profit_r=max(0.0, holdout_net),
            gross_loss_r=max(0.0, -holdout_net),
            profit_factor=None if holdout_net >= 0 else 0.0,
            profit_factor_r=None if holdout_net >= 0 else 0.0,
            win_rate=float(holdout_net > 0),
            active_day_count=1,
            max_drawdown_pips=max(0.0, -holdout_net),
            reason_counts={"SYNTHETIC_HOLDOUT": 1},
        )
        result["daily_aggregates"] = [
            {
                "pair": "EUR_USD",
                "candidate_id": candidate_id,
                "split": "VALIDATION",
                "utc_day": (start + timedelta(days=index)).date().isoformat(),
                "resolved_count": 1,
                "win_count": 1,
                "loss_count": 0,
                "flat_count": 0,
                "gross_mid_pips": float(index + 2),
                "spread_drag_pips": 1.0,
                "exact_net_pips": float(index + 1),
                "gross_mid_r": float(index + 2),
                "spread_drag_r": 1.0,
                "exact_net_r": float(index + 1),
                "gross_profit_pips": float(index + 1),
                "gross_loss_pips": 0.0,
                "gross_profit_r": float(index + 1),
                "gross_loss_r": 0.0,
                "max_drawdown_input_pips": 0.0,
                "reason_counts": {"SYNTHETIC_VALIDATION": 1},
            }
            for index in range(20)
        ]
        result["daily_aggregates"].append(
            {
                "pair": "EUR_USD",
                "candidate_id": candidate_id,
                "split": "HOLDOUT",
                "utc_day": (start + timedelta(days=30)).date().isoformat(),
                "resolved_count": 1,
                "win_count": int(holdout_net > 0),
                "loss_count": int(holdout_net < 0),
                "flat_count": int(holdout_net == 0),
                "gross_mid_pips": holdout_net + 1.0,
                "spread_drag_pips": 1.0,
                "exact_net_pips": holdout_net,
                "gross_mid_r": holdout_net + 1.0,
                "spread_drag_r": 1.0,
                "exact_net_r": holdout_net,
                "gross_profit_pips": max(0.0, holdout_net),
                "gross_loss_pips": max(0.0, -holdout_net),
                "gross_profit_r": max(0.0, holdout_net),
                "gross_loss_r": max(0.0, -holdout_net),
                "max_drawdown_input_pips": max(0.0, -holdout_net),
                "reason_counts": {"SYNTHETIC_HOLDOUT": 1},
            }
        )
        result["aggregation"]["observed_utc_days_by_split"] = {
            "VALIDATION": [
                (start + timedelta(days=index)).date().isoformat()
                for index in range(20)
            ],
            "HOLDOUT": [(start + timedelta(days=30)).date().isoformat()],
        }
        return copy.deepcopy(result)


if __name__ == "__main__":
    unittest.main()
