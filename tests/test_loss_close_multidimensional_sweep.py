from __future__ import annotations

from dataclasses import replace
import unittest

from quant_rabbit.loss_close_multidimensional_sweep import (
    STAGE_2,
    STAGE_3,
    SweepContract,
    build_local_refinement_grid,
    build_stage1_price_action_grid,
    evaluate_multidimensional_plateau,
)


def _arm(net: float, dd: float = 10.0) -> dict[str, float]:
    return {
        "mean_net_jpy": net,
        "max_drawdown_jpy": dd,
        "ruin_floor_breach_count": 0,
        "margin_closeout_breach_count": 0,
        "incomplete_unwind_count": 0,
        "unresolved_fill_order_count": 0,
    }


def _row(point, split: str, pa: float, *, inv: float = 0.0, candle: float = 0.5):
    return {
        "config_id": point.config_id,
        "config": point.config,
        "split": split,
        "cohort_sha256": ("a" if split == "TRAIN" else "b") * 64,
        "cost_model_sha256": "c" * 64,
        "event_count": 30,
        "arms": {
            "INVENTORY_ONLY": _arm(inv, 12.0),
            "CANDLE_1_2": _arm(candle, 11.0),
            "PRICE_ACTION_MULTI_BAR": _arm(pa, 10.0),
        },
    }


class LossCloseMultidimensionalSweepTest(unittest.TestCase):
    def assert_read_only(self, result):
        self.assertIs(result["read_only"], True)
        self.assertIs(result["paper_permission_allowed"], False)
        self.assertIs(result["live_permission_allowed"], False)
        self.assertIs(result["broker_order_allowed"], False)
        self.assertIs(result["deployment_allowed"], False)
        self.assertIs(result["holdout_used"], False)
        self.assertIs(result["always_profit_claim_allowed"], False)

    def test_stage1_is_bounded_coupled_geometry_not_full_cartesian(self):
        grid = build_stage1_price_action_grid()
        self.assertEqual(len(grid), 27)
        self.assertEqual(len({point.config_id for point in grid}), 27)
        self.assertTrue(all(point.feature_spec.attack_tolerance_ratio == 0.08 for point in grid))

    def test_local_refinement_changes_only_near_train_centres(self):
        centre = build_stage1_price_action_grid()[13].feature_spec
        stage2 = build_local_refinement_grid((centre,), stage=STAGE_2)
        stage3 = build_local_refinement_grid((centre,), stage=STAGE_3)
        self.assertLess(len(stage2), 12)
        self.assertEqual(
            {point.feature_spec.attack_tolerance_ratio for point in stage3},
            {0.04, 0.08, 0.12},
        )

    def test_isolated_maximum_is_rejected(self):
        points = build_stage1_price_action_grid()[:3]
        rows = []
        for index, point in enumerate(points):
            rows.extend(
                (
                    _row(point, "TRAIN", 10.0 if index == 1 else 0.0),
                    _row(point, "VALIDATION", 10.0),
                )
            )
        result = evaluate_multidimensional_plateau(rows)
        self.assertEqual(result["status"], "REJECTED_ISOLATED_TRAIN_PEAK")
        self.assertFalse(result["hypothesis_survives_pre_holdout"])
        self.assert_read_only(result)

    def test_connected_train_plateau_must_also_be_connected_on_validation(self):
        centre = build_stage1_price_action_grid()[13].feature_spec
        points = build_local_refinement_grid((centre,), stage=STAGE_3)
        rows = []
        for point in points:
            rows.extend((_row(point, "TRAIN", 3.0), _row(point, "VALIDATION", 2.0)))
        result = evaluate_multidimensional_plateau(rows)
        self.assertEqual(result["status"], "PRE_HOLDOUT_PLATEAU_SURVIVES_VALIDATION")
        self.assertEqual(result["selected_train_plateau_size"], 3)
        self.assertTrue(result["hypothesis_survives_pre_holdout"])
        self.assertFalse(result["selection_used_validation"])
        self.assertFalse(result["single_best_cell_adoption_allowed"])

    def test_validation_rejects_if_only_one_cell_remains_positive(self):
        centre = build_stage1_price_action_grid()[13].feature_spec
        points = build_local_refinement_grid((centre,), stage=STAGE_3)
        rows = []
        for index, point in enumerate(points):
            rows.extend(
                (
                    _row(point, "TRAIN", 3.0),
                    _row(point, "VALIDATION", 2.0 if index == 1 else 0.0),
                )
            )
        result = evaluate_multidimensional_plateau(rows)
        self.assertEqual(result["status"], "REJECTED_ON_VALIDATION")
        self.assertFalse(result["hypothesis_survives_pre_holdout"])

    def test_holdout_cohort_mismatch_and_short_embargo_fail_closed(self):
        points = build_stage1_price_action_grid()[:3]
        rows = [
            _row(point, split, 3.0)
            for point in points
            for split in ("TRAIN", "VALIDATION")
        ]
        rows[1]["cohort_sha256"] = "d" * 64
        mismatch = evaluate_multidimensional_plateau(rows)
        self.assertIn("SPLIT_COHORT_OR_COST_MISMATCH:VALIDATION", mismatch["blockers"])

        rows[1]["cohort_sha256"] = "b" * 64
        for row in rows:
            if row["split"] == "VALIDATION":
                row["cost_model_sha256"] = "d" * 64
        cost_mismatch = evaluate_multidimensional_plateau(rows)
        self.assertIn("COST_MODEL_MISMATCH_BETWEEN_SPLITS", cost_mismatch["blockers"])

        holdout = evaluate_multidimensional_plateau(rows, holdout_used=True)
        self.assertIn("HOLDOUT_USE_FORBIDDEN", holdout["blockers"])

        short = evaluate_multidimensional_plateau(
            rows,
            contract=replace(SweepContract(), embargo_seconds=300),
        )
        self.assertIn("EMBARGO_SHORTER_THAN_MAX_UNWIND", short["blockers"])

    def test_unresolved_fill_or_unwind_cannot_form_plateau(self):
        centre = build_stage1_price_action_grid()[13].feature_spec
        points = build_local_refinement_grid((centre,), stage=STAGE_3)
        rows = [
            _row(point, split, 3.0)
            for point in points
            for split in ("TRAIN", "VALIDATION")
        ]
        for row in rows:
            row["arms"]["PRICE_ACTION_MULTI_BAR"]["unresolved_fill_order_count"] = 1
        result = evaluate_multidimensional_plateau(rows)
        self.assertEqual(result["status"], "REJECTED_NO_TRAIN_INCREMENT")
        self.assertFalse(result["hypothesis_survives_pre_holdout"])


if __name__ == "__main__":
    unittest.main()
