from __future__ import annotations

import copy
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.eurusd_outcome_learning import (
    DECISION_FEATURES,
    DIAGNOSTIC_ROW_CONTRACT,
    DIAGNOSTIC_SCORECARD_CONTRACT,
    MANIFEST_CONTRACT,
    POLICY_CONTRACT,
    TRAINING_RECEIPT_CONTRACT,
    TRAINING_ROW_CONTRACT,
    build_diagnostic_rows,
    build_diagnostic_scorecard,
    build_prospective_decisions,
    build_prospective_outcomes,
    build_training_rows,
    canonical_sha,
    closed_market_observation,
    load_config,
    observe_prospective,
    retraining_status,
    route_state,
    seal,
    sealed_valid,
    train_policy,
    verify_manifest,
    write_json_atomic,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "eurusd_learned_policy_v1.json"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _signal(index: int, *, generated: datetime, method: str = "RANGE_ROTATION", side: str = "LONG") -> dict:
    body = {
        "contract": "QR_FAST_BOT_SHADOW_SIGNAL_V1",
        "schema_version": 3,
        "signal_id": f"signal-{index:04d}",
        "pair": "EUR_USD",
        "side": side,
        "method": method,
        "strategy_id": method.lower(),
        "generated_at_utc": generated.isoformat(),
        "m1_closed_candle_utc": generated.replace(second=0, microsecond=0).isoformat(),
        "quote_timestamp_utc": (generated - timedelta(seconds=1)).isoformat(),
        "m5_atr_pips": 5.0,
        "spread_pips": 0.8,
        "regime_score": -5.0,
        "shadow_only": True,
        "live_permission": False,
        "broker_mutation_allowed": False,
    }
    return {**body, "signal_sha256": canonical_sha(body)}


def _outcome(signal: dict, *, resolved: datetime, realized: float = -4.0, filled: bool = True) -> dict:
    generated = datetime.fromisoformat(signal["generated_at_utc"])
    body = {
        "contract": "QR_FAST_BOT_S5_BID_ASK_OUTCOME_V1",
        "schema_version": 1,
        "signal_id": signal["signal_id"],
        "signal_sha256": signal["signal_sha256"],
        "pair": "EUR_USD",
        "side": signal["side"],
        "method": signal["method"],
        "signal_generated_at_utc": signal["generated_at_utc"],
        "maturity_at_utc": (generated + timedelta(seconds=990)).isoformat(),
        "resolved_at_utc": resolved.isoformat(),
        "filled": filled,
        "fill_at_utc": (generated + timedelta(seconds=30)).isoformat() if filled else None,
        "exit_reason": "STOP_LOSS" if filled else "UNFILLED",
        "exit_at_utc": (generated + timedelta(seconds=120)).isoformat() if filled else None,
        "realized_pips": realized if filled else 0.0,
        "truth_source": "OANDA_S5_BID_ASK",
        "truth_request_coverage_proved": True,
        "truth_chunk_sha256": ["a" * 64],
        "broker_mutation": False,
        "shadow_only": True,
        "live_permission": False,
    }
    return seal(body)


def _baseline(signal: dict, outcome: dict, *, evaluated: datetime, regime: str = "REGIME_NEGATIVE") -> dict:
    body = {
        "contract": "QR_FAST_BOT_CORRECTIVE_CHALLENGER_ROW_V1",
        "schema_version": 1,
        "arm_id": "BASELINE",
        "row_identity": canonical_sha([signal["signal_id"], "BASELINE"]),
        "signal_id": signal["signal_id"],
        "signal_sha256": signal["signal_sha256"],
        "outcome_sha256": outcome["contract_sha256"],
        "evaluated_at_utc": evaluated.isoformat(),
        "generated_at_utc": signal["generated_at_utc"],
        "pair": "EUR_USD",
        "strategy": signal["method"],
        "side": signal["side"],
        "m5_atr_pips": signal["m5_atr_pips"],
        "atr_bucket": "ATR_GE_5",
        "spread_pips": signal["spread_pips"],
        "spread_bucket": "0.8P",
        "regime_score": signal["regime_score"],
        "regime_bucket": regime,
        "prior_atr_observations": 9,
        "prior_atr_median_pips": 2.5,
        "causal_atr_ratio": 2.0,
        "vol_shock": True,
        "vol_shock_reasons": ["CAUSAL_ATR_RATIO"],
        "rapid_time_bucket_utc": "2026-08-28T13:00:00+00:00",
        "vetoed": False,
        "veto_reason": None,
        "filled": outcome["filled"],
        "fill_at_utc": outcome["fill_at_utc"],
        "exit_reason": outcome["exit_reason"],
        "exit_at_utc": outcome["exit_at_utc"],
        "realized_pips": outcome["realized_pips"],
        "after_cost_net_pips": outcome["realized_pips"],
        "mfe_pips": 0.5,
        "mae_pips": 4.0,
        "time_to_stop_seconds": 90.0,
        "truth_source": "OANDA_S5_BID_ASK",
        "truth_chunk_sha256": ["a" * 64],
        "truth_hash_match": True,
        "execution_authority": "NONE",
        "broker_http_methods_used": ["GET"],
        "broker_mutation": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "live_permission": False,
        "automatic_parameter_change_allowed": False,
    }
    return seal(body)


def _ledger_fixture(root: Path, *, count: int = 30) -> tuple[Path, Path, Path]:
    start = datetime(2026, 8, 28, 13, 0, tzinfo=timezone.utc)
    signals: list[dict] = []
    outcomes: list[dict] = []
    baselines: list[dict] = []
    for index in range(count):
        signal = _signal(index, generated=start + timedelta(minutes=index))
        resolved = datetime.fromisoformat(signal["generated_at_utc"]) + timedelta(seconds=1000)
        outcome = _outcome(signal, resolved=resolved)
        baseline = _baseline(signal, outcome, evaluated=resolved + timedelta(seconds=1))
        signals.append(signal)
        outcomes.append(outcome)
        baselines.append(baseline)
    signal_path = root / "signals.jsonl"
    outcome_path = root / "outcomes.jsonl"
    baseline_path = root / "corrective.jsonl"
    _write_jsonl(signal_path, signals)
    _write_jsonl(outcome_path, outcomes)
    _write_jsonl(baseline_path, baselines)
    return signal_path, outcome_path, baseline_path


class EurUsdOutcomeLearningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config, self.config_sha = load_config(CONFIG)
        self.now = datetime(2026, 8, 29, 1, 0, tzinfo=timezone.utc)

    def _training(self, root: Path) -> tuple[list[dict], dict[str, str]]:
        signals, outcomes, corrective = _ledger_fixture(root)
        return build_training_rows(
            signal_ledger_path=signals,
            outcome_ledger_path=outcomes,
            corrective_ledger_path=corrective,
            cutoff_at_utc=datetime(2026, 8, 28, 20, 0, tzinfo=timezone.utc),
            now_utc=self.now,
            maximum_resolution_lag_seconds=3600,
        )

    def test_config_freezes_bounded_router_and_zero_authority(self) -> None:
        self.assertEqual(self.config["router"]["choices"], list(("NO_TRADE", "SHOCK_BREAKOUT_FOLLOW", "SHOCK_PULLBACK_CONTINUATION", "TREND_CONTINUATION")))
        self.assertEqual(self.config["authority"]["execution_authority"], "NONE")
        self.assertFalse(self.config["authority"]["automatic_adoption_allowed"])
        self.assertFalse(self.config["evidence"]["retrospective_reinterpretation_allowed"])

    def test_training_uses_resolved_rows_after_outcome_and_does_not_leak_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows, hashes = self._training(Path(directory))
        self.assertEqual(len(rows), 30)
        self.assertEqual(set(rows[0]["features"]), set(DECISION_FEATURES))
        self.assertEqual(rows[0]["outcome_fields_used_as_decision_features"], [])
        self.assertNotIn("realized_pips", rows[0]["features"])
        self.assertIn("realized_pips", rows[0]["labels"])
        self.assertLess(rows[0]["resolved_at_utc"], rows[0]["entered_training_at_utc"])
        self.assertEqual(len(hashes["input_ledger_sha256"]), 64)
        self.assertTrue(all(sealed_valid(row, TRAINING_ROW_CONTRACT) for row in rows))

    def test_unresolved_future_stale_and_out_of_order_rows_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            signals, outcomes, corrective = _ledger_fixture(root, count=3)
            outcome_rows = [json.loads(line) for line in outcomes.read_text().splitlines()]
            _write_jsonl(outcomes, outcome_rows[:-1])
            with self.assertRaisesRegex(ValueError, "unresolved"):
                build_training_rows(
                    signal_ledger_path=signals,
                    outcome_ledger_path=outcomes,
                    corrective_ledger_path=corrective,
                    cutoff_at_utc=datetime(2026, 8, 28, 20, 0, tzinfo=timezone.utc),
                    now_utc=self.now,
                    maximum_resolution_lag_seconds=3600,
                )
            signals, outcomes, corrective = _ledger_fixture(root, count=3)
            outcome_rows = [json.loads(line) for line in outcomes.read_text().splitlines()]
            outcome_rows[1]["resolved_at_utc"] = "2026-08-30T00:00:00+00:00"
            outcome_rows[1] = seal({k: v for k, v in outcome_rows[1].items() if k != "contract_sha256"})
            _write_jsonl(outcomes, outcome_rows)
            with self.assertRaisesRegex(ValueError, "unresolved|future"):
                build_training_rows(
                    signal_ledger_path=signals,
                    outcome_ledger_path=outcomes,
                    corrective_ledger_path=corrective,
                    cutoff_at_utc=datetime(2026, 8, 30, 1, 0, tzinfo=timezone.utc),
                    now_utc=self.now,
                    maximum_resolution_lag_seconds=3600,
                )

    def test_policy_quarantines_loss_cell_and_activation_is_strictly_later(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows, hashes = self._training(Path(directory))
        activation = datetime(2026, 8, 29, 0, 30, tzinfo=timezone.utc)
        trained = train_policy(
            training_rows=rows,
            input_hashes=hashes,
            config=self.config,
            config_sha256=self.config_sha,
            activation_at_utc=activation,
        )
        policy = trained["policy"]
        self.assertTrue(sealed_valid(policy, POLICY_CONTRACT))
        self.assertTrue(sealed_valid(trained["training_receipt"], TRAINING_RECEIPT_CONTRACT))
        self.assertTrue(sealed_valid(trained["manifest"], MANIFEST_CONTRACT))
        self.assertEqual(policy["status"], "TEST_REQUIRED")
        self.assertEqual(policy["quarantined_cells"][0]["strategy"], "RANGE_ROTATION")
        with self.assertRaisesRegex(ValueError, "strictly after"):
            train_policy(
                training_rows=rows,
                input_hashes=hashes,
                config=self.config,
                config_sha256=self.config_sha,
                activation_at_utc=datetime(2026, 8, 28, 19, 0, tzinfo=timezone.utc),
            )

    def test_manifest_hash_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows, hashes = self._training(Path(directory))
        trained = train_policy(
            training_rows=rows,
            input_hashes=hashes,
            config=self.config,
            config_sha256=self.config_sha,
            activation_at_utc=datetime(2026, 8, 29, 0, 30, tzinfo=timezone.utc),
        )
        tampered = copy.deepcopy(trained["manifest"])
        tampered["policy"]["quarantined_cells"] = []
        with self.assertRaisesRegex(ValueError, "manifest seal|policy seal"):
            verify_manifest(tampered, config=self.config, config_sha256=self.config_sha, training_rows=rows)

    def test_router_forbids_range_long_and_uses_only_bounded_shock_choices(self) -> None:
        policy = {
            "activation_at_utc": "2026-08-29T00:00:00+00:00",
            "quarantined_cells": [],
            "router_thresholds": self.config["router"]["selected_thresholds"],
        }
        features = {
            "pair": "EUR_USD",
            "side": "LONG",
            "strategy": "RANGE_ROTATION",
            "regime": "REGIME_NEGATIVE",
            "atr_ratio": 2.0,
            "atr_bucket": "ATR_GE_5",
            "impulse_direction": "LONG",
            "impulse_magnitude_atr": 1.3,
            "spread_to_atr": 0.1,
            "session": "NEW_YORK_UTC",
            "higher_timeframe_alignment": "ALIGNED",
        }
        decision = route_state(
            features,
            policy=policy,
            observed_at_utc=datetime(2026, 8, 29, 0, 30, tzinfo=timezone.utc),
            now_utc=datetime(2026, 8, 29, 0, 30, 30, tzinfo=timezone.utc),
        )
        self.assertEqual(decision["choice"], "NO_TRADE")
        self.assertEqual(decision["reason"], "RANGE_ROTATION_LONG_FORBIDDEN")
        features["strategy"] = "SHOCK_BREAKOUT_FOLLOW"
        features["regime"] = "REGIME_POSITIVE"
        decision = route_state(
            features,
            policy=policy,
            observed_at_utc=datetime(2026, 8, 29, 0, 30, tzinfo=timezone.utc),
            now_utc=datetime(2026, 8, 29, 0, 30, 30, tzinfo=timezone.utc),
        )
        self.assertEqual(decision["choice"], "SHOCK_BREAKOUT_FOLLOW")

    def test_threshold_outside_allowlist_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            value = copy.deepcopy(self.config)
            value["router"]["selected_thresholds"]["max_spread_to_atr"] = 0.99
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside allowlist"):
                load_config(path)

    def test_pre_activation_diagnostic_is_separate_from_prospective(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows, hashes = self._training(Path(directory))
        trained = train_policy(
            training_rows=rows,
            input_hashes=hashes,
            config=self.config,
            config_sha256=self.config_sha,
            activation_at_utc=datetime(2026, 8, 29, 0, 30, tzinfo=timezone.utc),
        )
        diagnostic = build_diagnostic_rows(rows, policy=trained["policy"])
        scorecard = build_diagnostic_scorecard(diagnostic, policy=trained["policy"], generated_at_utc=self.now)
        self.assertTrue(all(sealed_valid(row, DIAGNOSTIC_ROW_CONTRACT) for row in diagnostic))
        self.assertTrue(sealed_valid(scorecard, DIAGNOSTIC_SCORECARD_CONTRACT))
        self.assertFalse(scorecard["counts_as_forward_evidence"])
        self.assertEqual(scorecard["post_activation_prospective_row_count"], 0)
        shock = next(row for row in scorecard["comparison"] if row["arm_id"] == "EXISTING_SHOCK_FOLLOW")
        self.assertEqual(shock["eligible_count"], 0)

    def test_post_activation_shock_route_uses_only_frozen_ex_ante_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows, hashes = self._training(Path(directory))
        trained = train_policy(
            training_rows=rows,
            input_hashes=hashes,
            config=self.config,
            config_sha256=self.config_sha,
            activation_at_utc=datetime(2026, 8, 29, 0, 30, tzinfo=timezone.utc),
        )
        generated = datetime(2026, 8, 29, 0, 31, tzinfo=timezone.utc)
        signal = seal(
            {
                "contract": "QR_FAST_BOT_SHOCK_FOLLOW_SIGNAL_V1",
                "signal_id": "shock-1",
                "pair": "EUR_USD",
                "side": "LONG",
                "strategy_id": "SHOCK_BREAKOUT_FOLLOW",
                "generated_at_utc": generated.isoformat(),
                "m1_atr_expansion_ratio": 2.0,
                "shock_bucket": "SHOCK_1.8_TO_LT_2.5",
                "direction": "LONG",
                "m1_impulse_body_atr_ratio": 1.3,
                "spread_to_m1_atr": 0.1,
                "m5_direction": "UP",
                "execution_authority": "NONE",
                "external_order_attempts": 0,
                "external_orders": 0,
                "live_permission": False,
            }
        )
        decisions = build_prospective_decisions(
            [signal],
            policy=trained["policy"],
            now_utc=generated + timedelta(seconds=30),
        )
        outcome = seal(
            {
                "contract": "QR_FAST_BOT_SHOCK_FOLLOW_S5_OUTCOME_V1",
                "signal_id": "shock-1",
                "signal_generated_at_utc": generated.isoformat(),
                "resolved_at_utc": (generated + timedelta(minutes=20)).isoformat(),
                "filled": True,
                "fill_at_utc": (generated + timedelta(seconds=10)).isoformat(),
                "exit_reason": "TAKE_PROFIT",
                "exit_at_utc": (generated + timedelta(minutes=5)).isoformat(),
                "realized_pips": 2.0,
                "mfe_pips": 2.4,
                "mae_pips": 0.4,
                "entry_slippage_pips": 0.1,
                "truth_chunk_sha256": ["b" * 64],
                "execution_authority": "NONE",
                "external_order_attempts": 0,
                "external_orders": 0,
                "live_permission": False,
            }
        )
        prospective = build_prospective_outcomes(decisions, [outcome], policy=trained["policy"])
        self.assertEqual(decisions[0]["choice"], "SHOCK_BREAKOUT_FOLLOW")
        self.assertEqual(decisions[0]["order_fields_authored"], [])
        self.assertTrue(prospective[0]["counts_as_forward_evidence"])
        self.assertEqual(prospective[0]["truth_source"], "OANDA_S5_BID_ASK")

    def test_retraining_governance_never_auto_adopts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows, hashes = self._training(Path(directory))
        trained = train_policy(
            training_rows=rows,
            input_hashes=hashes,
            config=self.config,
            config_sha256=self.config_sha,
            activation_at_utc=datetime(2026, 8, 29, 0, 30, tzinfo=timezone.utc),
        )
        result = retraining_status(previous_receipt=trained["training_receipt"], candidate_rows=rows, config=self.config)
        self.assertEqual(result["status"], "NO_CHANGE_INSUFFICIENT_EVIDENCE")
        self.assertFalse(result["automatic_adoption_allowed"])

    def test_closed_market_observation_requests_no_restart_and_keeps_orders_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, hashes = self._training(root)
            trained = train_policy(
                training_rows=rows,
                input_hashes=hashes,
                config=self.config,
                config_sha256=self.config_sha,
                activation_at_utc=datetime(2026, 8, 29, 0, 30, tzinfo=timezone.utc),
            )
            manifest = root / "manifest.json"
            write_json_atomic(manifest, trained["manifest"])
            result = closed_market_observation(
                manifest_path=manifest,
                config_path=CONFIG,
                now_utc=datetime(2026, 8, 29, 1, 0, tzinfo=timezone.utc),
            )
        self.assertEqual(result["status"], "MARKET_CLOSED_NO_OBSERVATION")
        self.assertFalse(result["launchagent_restart_requested"])
        self.assertEqual(result["execution_authority"], "NONE")
        self.assertEqual(result["external_order_attempts"], 0)
        self.assertEqual(result["external_orders"], 0)
        self.assertEqual(result["manual_tagless_positions_policy"], "NO_TOUCH")

    def test_closed_market_keeps_empty_post_activation_ledgers_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, hashes = self._training(root)
            trained = train_policy(
                training_rows=rows,
                input_hashes=hashes,
                config=self.config,
                config_sha256=self.config_sha,
                activation_at_utc=datetime(2026, 8, 29, 0, 30, tzinfo=timezone.utc),
            )
            manifest = root / "manifest.json"
            write_json_atomic(manifest, trained["manifest"])
            decisions = root / "post_decisions.jsonl"
            outcomes = root / "post_outcomes.jsonl"
            scorecard = root / "post_scorecard.json"
            result = observe_prospective(
                manifest_path=manifest,
                config_path=CONFIG,
                shock_signal_ledger_path=root / "shock_signals.jsonl",
                shock_outcome_ledger_path=root / "shock_outcomes.jsonl",
                decision_ledger_path=decisions,
                prospective_outcome_ledger_path=outcomes,
                prospective_scorecard_path=scorecard,
                now_utc=datetime(2026, 8, 29, 1, 0, tzinfo=timezone.utc),
            )
            stored = json.loads(scorecard.read_text())
            self.assertEqual(decisions.read_text(), "")
            self.assertEqual(outcomes.read_text(), "")
        self.assertEqual(result["prospective_sample_count"], 0)
        self.assertEqual(stored["pre_activation_diagnostic_rows_included"], 0)
        self.assertTrue(stored["counts_as_forward_evidence"])


if __name__ == "__main__":
    unittest.main()
