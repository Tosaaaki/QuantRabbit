from __future__ import annotations

import copy
import math
import unittest

from quant_rabbit.dojo_replay_gates import (
    PROOF_MANIFEST_CONTRACT,
    canonical_proof_manifest_sha256,
    evaluate_inventory_release_proof_ladder,
)


def _metrics(
    *,
    settlements: int = 40,
    active_days: int = 24,
    net_jpy: float = 400.0,
    profit_factor: float = 1.5,
    worst_day_jpy: float = -100.0,
    realized_drawdown_jpy: float = 200.0,
    margin_events: int = 0,
    ruin_events: int = 0,
    unresolved_positions: int = 0,
    unresolved_orders: int = 0,
    end_of_replay_forced_close_count: int = 0,
    end_of_replay_forced_close_net_jpy: float = 0.0,
) -> dict[str, object]:
    return {
        "settlements": settlements,
        "active_days": active_days,
        "net_jpy": net_jpy,
        "profit_factor": profit_factor,
        "expectancy_jpy": net_jpy / settlements if settlements else 0.0,
        "worst_day_jpy": worst_day_jpy,
        "realized_drawdown_jpy": realized_drawdown_jpy,
        "margin_events": margin_events,
        "ruin_events": ruin_events,
        "unresolved_positions": unresolved_positions,
        "unresolved_orders": unresolved_orders,
        "end_of_replay_forced_close_count": (end_of_replay_forced_close_count),
        "end_of_replay_forced_close_net_jpy": (end_of_replay_forced_close_net_jpy),
    }


def _passing_arms() -> list[dict[str, object]]:
    arms: list[dict[str, object]] = []
    for window in ("TRAIN", "VAL", "S5"):
        for policy in ("BASELINE", "CANDIDATE"):
            for cost in ("BASE", "STRESS"):
                for intrabar in ("OHLC", "OLHC"):
                    if window == "TRAIN":
                        metrics = (
                            _metrics(
                                net_jpy=600.0,
                                profit_factor=1.6,
                                worst_day_jpy=-90.0,
                                realized_drawdown_jpy=180.0,
                            )
                            if policy == "CANDIDATE"
                            else _metrics(
                                net_jpy=400.0,
                                profit_factor=1.4,
                            )
                        )
                    else:
                        metrics = (
                            _metrics(
                                net_jpy=500.0,
                                profit_factor=1.4,
                                worst_day_jpy=-90.0,
                                realized_drawdown_jpy=180.0,
                            )
                            if policy == "CANDIDATE"
                            else _metrics(
                                net_jpy=300.0,
                                profit_factor=1.2,
                            )
                        )
                    arms.append(
                        {
                            "window": window,
                            "policy": policy,
                            "cost": cost,
                            "intrabar": intrabar,
                            "metrics": metrics,
                        }
                    )
    return arms


def _sealed_manifest(
    arms: list[dict[str, object]],
) -> dict[str, object]:
    body: dict[str, object] = {
        "contract": PROOF_MANIFEST_CONTRACT,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "candidate_id": "a" * 64,
        "spec_sha256": "b" * 64,
        "policy_sha256": "c" * 64,
        "artifact_manifest_sha256": "d" * 64,
        "windows": {
            "TRAIN": {
                "from_utc": "2026-01-01T00:00:00+00:00",
                "to_utc": "2026-03-01T00:00:00+00:00",
                "source_sha256": "e" * 64,
            },
            "VAL": {
                "from_utc": "2026-03-01T00:00:00+00:00",
                "to_utc": "2026-05-01T00:00:00+00:00",
                "source_sha256": "e" * 64,
            },
            "S5": {
                "from_utc": "2026-05-10T00:00:00Z",
                "to_utc": "2026-07-17T00:00:00Z",
                "source_sha256": "f" * 64,
            },
        },
        "arms": arms,
    }
    body["manifest_sha256"] = canonical_proof_manifest_sha256(body)
    return body


def _evaluate(
    arms: list[dict[str, object]],
) -> dict[str, object]:
    manifest = _sealed_manifest(arms)
    digest = manifest["manifest_sha256"]
    assert isinstance(digest, str)
    return evaluate_inventory_release_proof_ladder(
        manifest,
        expected_manifest_sha256=digest,
    )


def _find_arm(
    arms: list[dict[str, object]],
    window: str,
    policy: str,
    cost: str,
    intrabar: str,
) -> dict[str, object]:
    return next(
        arm
        for arm in arms
        if (
            arm["window"],
            arm["policy"],
            arm["cost"],
            arm["intrabar"],
        )
        == (window, policy, cost, intrabar)
    )


class DojoReplayGateTests(unittest.TestCase):
    def test_complete_independent_proof_passes(self) -> None:
        result = _evaluate(_passing_arms())

        self.assertEqual(result["decision"], "PROOF_ELIGIBLE")
        self.assertEqual(result["contract"], "QR_DOJO_REPLAY_GATE_DECISION_V2")
        self.assertTrue(result["manifest_authenticated"])
        self.assertTrue(result["train_eligible"])
        self.assertTrue(result["independent_proof_eligible"])
        self.assertTrue(result["proof_eligible"])
        self.assertFalse(result["artifact_provenance_authenticated"])
        self.assertFalse(result["paper_eligible"])
        self.assertIsNone(result["launch_preflight_token_sha256"])
        self.assertIsNone(result["death_code"])
        self.assertEqual(result["reasons"], [])

    def test_input_is_not_mutated_and_result_is_deterministic(self) -> None:
        arms = _passing_arms()
        before = copy.deepcopy(arms)

        first = _evaluate(arms)
        second = _evaluate(arms)

        self.assertEqual(arms, before)
        self.assertEqual(first, second)

    def test_naked_arms_or_missing_trusted_digest_fail_closed(self) -> None:
        arms = _passing_arms()
        naked = evaluate_inventory_release_proof_ladder(arms)
        manifest = _sealed_manifest(arms)
        untrusted = evaluate_inventory_release_proof_ladder(manifest)

        for result in (naked, untrusted):
            self.assertEqual(result["decision"], "MEASUREMENT_BLOCKED")
            self.assertEqual(result["death_code"], "MEASUREMENT")
            self.assertFalse(result["manifest_authenticated"])
            self.assertFalse(result["proof_eligible"])

    def test_canonical_and_trusted_manifest_digests_are_both_required(self) -> None:
        manifest = _sealed_manifest(_passing_arms())
        trusted_digest = manifest["manifest_sha256"]
        assert isinstance(trusted_digest, str)

        manifest["artifact_manifest_sha256"] = "1" * 64
        stale_self_digest = evaluate_inventory_release_proof_ladder(
            manifest,
            expected_manifest_sha256=trusted_digest,
        )
        self.assertEqual(stale_self_digest["decision"], "MEASUREMENT_BLOCKED")
        stale_messages = {reason["message"] for reason in stale_self_digest["reasons"]}
        self.assertIn(
            "sealed proof manifest canonical digest mismatch",
            stale_messages,
        )

        manifest["manifest_sha256"] = canonical_proof_manifest_sha256(manifest)
        resealed_spoof = evaluate_inventory_release_proof_ladder(
            manifest,
            expected_manifest_sha256=trusted_digest,
        )
        self.assertEqual(resealed_spoof["decision"], "MEASUREMENT_BLOCKED")
        self.assertIn(
            "sealed proof manifest does not match trusted digest",
            {reason["message"] for reason in resealed_spoof["reasons"]},
        )

    def test_spoofed_window_or_source_cannot_reuse_trusted_binding(self) -> None:
        for mutation in ("window", "source"):
            with self.subTest(mutation=mutation):
                manifest = _sealed_manifest(_passing_arms())
                trusted_digest = manifest["manifest_sha256"]
                assert isinstance(trusted_digest, str)
                windows = manifest["windows"]
                assert isinstance(windows, dict)
                train = windows["TRAIN"]
                assert isinstance(train, dict)
                if mutation == "window":
                    train["to_utc"] = "2026-02-01T00:00:00+00:00"
                else:
                    train["source_sha256"] = "1" * 64
                manifest["manifest_sha256"] = canonical_proof_manifest_sha256(manifest)

                result = evaluate_inventory_release_proof_ladder(
                    manifest,
                    expected_manifest_sha256=trusted_digest,
                )

                self.assertEqual(result["decision"], "MEASUREMENT_BLOCKED")
                self.assertFalse(result["manifest_authenticated"])
                self.assertIn(
                    "sealed proof manifest does not match trusted digest",
                    {reason["message"] for reason in result["reasons"]},
                )

    def test_missing_train_or_duplicate_arm_is_measurement_block(self) -> None:
        for mutation in ("missing_train", "duplicate"):
            with self.subTest(mutation=mutation):
                arms = _passing_arms()
                if mutation == "missing_train":
                    arms.pop(0)
                else:
                    arms.append(copy.deepcopy(arms[0]))
                result = _evaluate(arms)

                self.assertEqual(result["decision"], "MEASUREMENT_BLOCKED")
                self.assertEqual(result["death_code"], "MEASUREMENT")
                self.assertFalse(result["proof_eligible"])

    def test_rejected_train_does_not_require_unrun_val_or_s5(self) -> None:
        arms = [arm for arm in _passing_arms() if arm["window"] == "TRAIN"]
        metrics = _find_arm(arms, "TRAIN", "CANDIDATE", "BASE", "OHLC")["metrics"]
        assert isinstance(metrics, dict)
        metrics["net_jpy"] = 200.0
        metrics["expectancy_jpy"] = 5.0

        result = _evaluate(arms)

        self.assertEqual(result["decision"], "TRAIN_REJECTED")
        self.assertEqual(result["death_code"], "INVENTORY")

    def test_selected_train_requires_complete_val_and_s5_matrix(self) -> None:
        arms = _passing_arms()
        arms.pop()

        result = _evaluate(arms)

        self.assertEqual(result["decision"], "MEASUREMENT_BLOCKED")
        self.assertTrue(result["train_eligible"])
        self.assertEqual(result["death_code"], "MEASUREMENT")

    def test_all_train_arms_need_the_fixed_sample_floors(self) -> None:
        arms = _passing_arms()
        metrics = _find_arm(arms, "TRAIN", "BASELINE", "BASE", "OHLC")["metrics"]
        assert isinstance(metrics, dict)
        metrics["settlements"] = 29
        metrics["expectancy_jpy"] = metrics["net_jpy"] / 29

        result = _evaluate(arms)

        self.assertEqual(result["decision"], "TRAIN_REJECTED")
        self.assertEqual(result["death_code"], "MEASUREMENT")
        self.assertIn("settlements are below", result["reasons"][0]["message"])

    def test_train_requires_strict_net_and_expectancy_improvement(self) -> None:
        arms = _passing_arms()
        baseline = _find_arm(arms, "TRAIN", "BASELINE", "STRESS", "OLHC")["metrics"]
        candidate = _find_arm(arms, "TRAIN", "CANDIDATE", "STRESS", "OLHC")["metrics"]
        assert isinstance(baseline, dict)
        assert isinstance(candidate, dict)
        candidate["net_jpy"] = baseline["net_jpy"]
        candidate["expectancy_jpy"] = baseline["expectancy_jpy"]

        result = _evaluate(arms)

        self.assertEqual(result["decision"], "TRAIN_REJECTED")
        self.assertEqual(result["death_code"], "INVENTORY")
        messages = {reason["message"] for reason in result["reasons"]}
        self.assertIn("candidate net_jpy did not strictly improve baseline", messages)
        self.assertIn(
            "candidate expectancy_jpy did not strictly improve baseline",
            messages,
        )

    def test_train_rejects_each_worse_risk_dimension(self) -> None:
        mutations = {
            "worst_day_jpy": -101.0,
            "realized_drawdown_jpy": 201.0,
            "margin_events": 1,
            "ruin_events": 1,
            "unresolved_positions": 1,
            "unresolved_orders": 1,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                arms = _passing_arms()
                metrics = _find_arm(arms, "TRAIN", "CANDIDATE", "BASE", "OHLC")[
                    "metrics"
                ]
                assert isinstance(metrics, dict)
                metrics[field] = value

                result = _evaluate(arms)

                self.assertEqual(result["decision"], "TRAIN_REJECTED")
                self.assertEqual(result["death_code"], "RISK")

    def test_train_failure_never_advances_on_good_validation(self) -> None:
        arms = _passing_arms()
        metrics = _find_arm(arms, "TRAIN", "CANDIDATE", "BASE", "OHLC")["metrics"]
        assert isinstance(metrics, dict)
        metrics["net_jpy"] = -400.0
        metrics["expectancy_jpy"] = -10.0

        result = _evaluate(arms)

        self.assertEqual(result["decision"], "TRAIN_REJECTED")
        self.assertFalse(result["independent_proof_eligible"])

    def test_val_and_s5_both_paths_need_independent_stress_gate(self) -> None:
        for window in ("VAL", "S5"):
            for intrabar in ("OHLC", "OLHC"):
                with self.subTest(window=window, intrabar=intrabar):
                    arms = _passing_arms()
                    metrics = _find_arm(
                        arms,
                        window,
                        "CANDIDATE",
                        "STRESS",
                        intrabar,
                    )["metrics"]
                    assert isinstance(metrics, dict)
                    metrics["profit_factor"] = 1.249999

                    result = _evaluate(arms)

                    self.assertEqual(result["decision"], "PROOF_REJECTED")
                    self.assertTrue(result["train_eligible"])
                    self.assertEqual(result["death_code"], "OVERFIT")
                    self.assertFalse(result["proof_eligible"])

    def test_all_independent_base_and_stress_arms_need_sample_floors(self) -> None:
        for cost in ("BASE", "STRESS"):
            for policy in ("BASELINE", "CANDIDATE"):
                with self.subTest(cost=cost, policy=policy):
                    arms = _passing_arms()
                    metrics = _find_arm(arms, "VAL", policy, cost, "OHLC")["metrics"]
                    assert isinstance(metrics, dict)
                    metrics["active_days"] = 19

                    result = _evaluate(arms)

                    self.assertEqual(result["decision"], "PROOF_REJECTED")
                    self.assertEqual(result["death_code"], "MEASUREMENT")

    def test_independent_base_cannot_be_negative_or_worse_risk(self) -> None:
        for mutation in ("negative", "risk"):
            with self.subTest(mutation=mutation):
                arms = _passing_arms()
                metrics = _find_arm(arms, "S5", "CANDIDATE", "BASE", "OLHC")["metrics"]
                assert isinstance(metrics, dict)
                if mutation == "negative":
                    metrics["net_jpy"] = -40.0
                    metrics["expectancy_jpy"] = -1.0
                    metrics["profit_factor"] = 0.8
                else:
                    metrics["worst_day_jpy"] = -101.0

                result = _evaluate(arms)

                self.assertEqual(result["decision"], "PROOF_REJECTED")
                self.assertEqual(
                    result["death_code"],
                    "OVERFIT" if mutation == "negative" else "RISK",
                )

    def test_independent_stress_requires_positive_net_and_expectancy(self) -> None:
        arms = _passing_arms()
        metrics = _find_arm(arms, "VAL", "CANDIDATE", "STRESS", "OHLC")["metrics"]
        assert isinstance(metrics, dict)
        metrics["net_jpy"] = 0.0
        metrics["expectancy_jpy"] = 0.0

        result = _evaluate(arms)

        self.assertEqual(result["decision"], "PROOF_REJECTED")
        self.assertEqual(result["death_code"], "OVERFIT")
        messages = {reason["message"] for reason in result["reasons"]}
        self.assertIn("candidate STRESS net_jpy is not positive", messages)
        self.assertIn("candidate STRESS expectancy_jpy is not positive", messages)

    def test_independent_stress_rejects_worse_risk(self) -> None:
        arms = _passing_arms()
        metrics = _find_arm(arms, "S5", "CANDIDATE", "STRESS", "OLHC")["metrics"]
        assert isinstance(metrics, dict)
        metrics["realized_drawdown_jpy"] = 201.0

        result = _evaluate(arms)

        self.assertEqual(result["decision"], "PROOF_REJECTED")
        self.assertEqual(result["death_code"], "RISK")

    def test_equal_nonzero_candidate_unresolved_end_is_rejected(self) -> None:
        for window in ("TRAIN", "VAL"):
            with self.subTest(window=window):
                arms = _passing_arms()
                for policy in ("BASELINE", "CANDIDATE"):
                    metrics = _find_arm(arms, window, policy, "BASE", "OHLC")["metrics"]
                    assert isinstance(metrics, dict)
                    metrics["unresolved_positions"] = 1
                    metrics["unresolved_orders"] = 1

                result = _evaluate(arms)

                self.assertEqual(
                    result["decision"],
                    "TRAIN_REJECTED" if window == "TRAIN" else "PROOF_REJECTED",
                )
                self.assertEqual(result["death_code"], "RISK")
                messages = {reason["message"] for reason in result["reasons"]}
                self.assertIn(
                    "candidate unresolved_positions must be zero at replay end",
                    messages,
                )
                self.assertIn(
                    "candidate unresolved_orders must be zero at replay end",
                    messages,
                )

    def test_forced_close_is_zero_on_every_baseline_and_candidate_arm(self) -> None:
        arms = _passing_arms()
        metrics = _find_arm(arms, "VAL", "BASELINE", "BASE", "OHLC")["metrics"]
        assert isinstance(metrics, dict)
        metrics["end_of_replay_forced_close_count"] = 1
        metrics["end_of_replay_forced_close_net_jpy"] = -10.0

        result = _evaluate(arms)

        self.assertEqual(result["decision"], "PROOF_REJECTED")
        self.assertEqual(result["death_code"], "MEASUREMENT")
        self.assertIn(
            "end-of-replay forced-close activity is forbidden",
            {reason["message"] for reason in result["reasons"]},
        )

    def test_missing_safety_field_and_nonfinite_value_fail_closed(self) -> None:
        for mutation in ("missing_ruin", "nan_net", "boolean_count"):
            with self.subTest(mutation=mutation):
                arms = _passing_arms()
                metrics = arms[0]["metrics"]
                assert isinstance(metrics, dict)
                if mutation == "missing_ruin":
                    metrics.pop("ruin_events")
                elif mutation == "nan_net":
                    metrics["net_jpy"] = math.nan
                else:
                    metrics["margin_events"] = True

                result = _evaluate(arms)

                self.assertEqual(result["decision"], "MEASUREMENT_BLOCKED")
                self.assertEqual(result["death_code"], "MEASUREMENT")

    def test_expectancy_must_reconcile_to_net_and_count(self) -> None:
        arms = _passing_arms()
        metrics = arms[0]["metrics"]
        assert isinstance(metrics, dict)
        metrics["expectancy_jpy"] = 999.0

        result = _evaluate(arms)

        self.assertEqual(result["decision"], "MEASUREMENT_BLOCKED")
        self.assertIn(
            "expectancy_jpy does not reconcile",
            result["reasons"][0]["message"],
        )

    def test_legitimate_infinite_profit_factor_is_supported(self) -> None:
        arms = _passing_arms()
        metrics = _find_arm(arms, "VAL", "CANDIDATE", "STRESS", "OHLC")["metrics"]
        assert isinstance(metrics, dict)
        metrics["profit_factor"] = math.inf

        result = _evaluate(arms)

        self.assertEqual(result["decision"], "PROOF_ELIGIBLE")

    def test_infinite_profit_factor_without_positive_economics_is_invalid(self) -> None:
        arms = _passing_arms()
        metrics = arms[0]["metrics"]
        assert isinstance(metrics, dict)
        metrics["profit_factor"] = math.inf
        metrics["net_jpy"] = 0.0
        metrics["expectancy_jpy"] = 0.0

        result = _evaluate(arms)

        self.assertEqual(result["decision"], "MEASUREMENT_BLOCKED")
        self.assertEqual(result["death_code"], "MEASUREMENT")


if __name__ == "__main__":
    unittest.main()
