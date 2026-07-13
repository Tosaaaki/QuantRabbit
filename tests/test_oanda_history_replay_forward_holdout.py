from __future__ import annotations

import gzip
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from quant_rabbit.strategy.forecast_persistence_tracker import record_forecast
from quant_rabbit.strategy.forecast_technical_context import build_forecast_technical_context


def _load_module():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    path = scripts / "oanda_history_replay_forward_holdout.py"
    spec = importlib.util.spec_from_file_location("oanda_history_replay_forward_holdout", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


forward = _load_module()
import oanda_history_fetch as history_fetch  # noqa: E402


def _fake_locked_truth_fetch(command: list[str]) -> None:
    def argument(name: str) -> str:
        return command[command.index(name) + 1]

    root = Path(argument("--output-dir"))
    pair = argument("--pairs")
    granularity = argument("--granularities")
    start = forward._required_time(argument("--from"), "--from")
    end = forward._required_time(argument("--to"), "--to")
    pair_dir = root / "mock-run" / pair
    pair_dir.mkdir(parents=True)
    rows = []
    timestamp = start
    index = 0
    while timestamp < end:
        bid = 1.1000 + index * 0.00004
        ask = bid + 0.0001
        rows.append(
            {
                "pair": pair,
                "granularity": granularity,
                "price": "BA",
                "complete": True,
                "time": forward._iso(timestamp),
                "bid": {
                    "o": bid,
                    "h": bid + 0.00006,
                    "l": bid - 0.00001,
                    "c": bid + 0.00004,
                },
                "ask": {
                    "o": ask,
                    "h": ask + 0.00006,
                    "l": ask - 0.00001,
                    "c": ask + 0.00004,
                },
            }
        )
        timestamp += forward._granularity_delta(granularity)
        index += 1
    candle_path = pair_dir / (
        f"{pair}_{granularity}_BA_"
        f"{start.strftime('%Y%m%dT%H%M%SZ')}_{end.strftime('%Y%m%dT%H%M%SZ')}.jsonl.gz"
    )
    payload = "\n".join(json.dumps(item) for item in rows) + "\n"
    candle_path.write_bytes(gzip.compress(payload.encode("utf-8")))
    history_fetch._append_truth_acquisition_receipt(
        output_root=root,
        task=history_fetch.FetchTask(
            pair=pair,
            granularity=granularity,
            start=start,
            end=end,
            price="BA",
        ),
        candle_path=candle_path,
        rows=len(rows),
    )


def _candidate() -> dict:
    return {
        "contract_kind": forward.CANDIDATE_KIND,
        "candidate_id": "TEST_EUR_USD_UP",
        "pair": "EUR_USD",
        "direction": "UP",
        "confidence_policy": {"field": "calibrated", "minimum": 0.0},
        "technical_selector": {key: "ANY" for key in forward.SELECTOR_FIELDS},
        "max_horizon_min": 1,
        "exit_policy": {"take_profit_pips": 1.0, "stop_loss_pips": 1.0},
        "acceptance": {
            "min_samples": 30,
            "min_active_days": 5,
            "max_daily_sample_share": 0.35,
            "min_positive_day_rate": 0.6,
            "min_directional_hit_wilson95_lower": 0.55,
            "min_avg_final_pips_exclusive": 0.0,
            "min_avg_realized_r_exclusive": 0.0,
            "min_fixed_exit_avg_realized_pips_exclusive": 0.5,
            "min_fixed_exit_win_wilson95_lower": 0.5,
            "min_fixed_exit_profit_factor": 1.5,
        },
    }


def _training_report() -> dict:
    return {
        "selection_contract": {"forecast_to_utc_exclusive": "2026-07-13T00:00:00Z"},
        "experiment": {"experiment_id": "training-experiment"},
        "max_evaluated_horizon_min": 60,
        "segments": {
            "by_horizon": [
                {"horizon_bucket": "31-60m", "n": 30}
            ]
        },
    }


def _context() -> dict:
    return build_forecast_technical_context(
        {
            "confluence": {
                "dominant_regime": "TREND_UP",
                "price_percentile_24h": 0.5,
                "price_percentile_7d": 0.5,
            },
            "views": [
                {
                    "granularity": "M5",
                    "regime_reading": {"state": "TREND_STRONG", "atr_percentile": 60},
                    "indicators": {"atr_pips": 2.0},
                    "structure": {
                        "structure_events": [
                            {"kind": "BOS_UP", "index": 3, "close_confirmed": True}
                        ]
                    },
                },
                {
                    "granularity": "M15",
                    "regime_reading": {"state": "TREND_WEAK", "atr_percentile": 50},
                    "structure": {
                        "structure_events": [
                            {"kind": "BOS_UP", "index": 2, "close_confirmed": True}
                        ]
                    },
                },
            ],
        },
        pair="EUR_USD",
        current_price=1.1001,
        spread_pips=1.0,
    )


class ForwardHoldoutTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path, Path]:
        candidate = root / "candidate.json"
        training = root / "training.json"
        forecast = root / "forecast_history.jsonl"
        output = root / "reports"
        candidate.write_text(json.dumps(_candidate()), encoding="utf-8")
        training.write_text(json.dumps(_training_report()), encoding="utf-8")
        forecast.write_text("", encoding="utf-8")
        return candidate, training, forecast, output

    def test_locked_truth_environment_forces_production_oanda(self) -> None:
        with patch.dict(
            os.environ,
            {
                "QR_OANDA_BASE_URL": "http://127.0.0.1:9999",
                "QR_OANDA_ENV_FILE": "/tmp/fake.env",
                "PYTHONPATH": "/tmp/fake-python",
                "QR_OANDA_TOKEN": "secret",
                "QR_OANDA_ACCOUNT_ID": "account",
            },
            clear=False,
        ):
            env = forward._locked_truth_subprocess_env()

        self.assertEqual(env["QR_OANDA_BASE_URL"], forward.PRODUCTION_OANDA_BASE_URL)
        self.assertNotIn("QR_OANDA_ENV_FILE", env)
        self.assertNotIn("PYTHONPATH", env)
        self.assertEqual(env["QR_OANDA_TOKEN"], "secret")

    def test_lock_is_created_before_window_and_evaluation_cannot_run_early(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate, training, forecast, output = self._fixture(root)
            start = datetime(2026, 7, 14, tzinfo=timezone.utc)
            end = start + timedelta(minutes=1)
            locked = forward.create_lock(
                candidate_path=candidate,
                training_report_path=training,
                forecast_history_path=forecast,
                holdout_from=start,
                holdout_to=end,
                granularity="S5",
                truth_roots=[root / "history"],
                output_dir=output,
                now_utc=start - timedelta(days=1),
            )

            lock_path = Path(locked["lock_path"])
            lock = json.loads(lock_path.read_text())
            self.assertEqual(lock_path.stem, lock["lock_sha256"])
            pending = forward.evaluate_lock(
                lock_path=lock_path,
                output_dir=output,
                now_utc=end,
            )
            self.assertEqual(pending["status"], "PENDING_NOT_MATURE")

    def test_lock_rejects_existing_holdout_rows_and_unknown_candidate_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate, training, forecast, output = self._fixture(root)
            start = datetime(2026, 7, 14, tzinfo=timezone.utc)
            forecast.write_text(json.dumps({"timestamp_utc": forward._iso(start)}) + "\n")
            with self.assertRaisesRegex(ValueError, "already contains"):
                forward.create_lock(
                    candidate_path=candidate,
                    training_report_path=training,
                    forecast_history_path=forecast,
                    holdout_from=start,
                    holdout_to=start + timedelta(days=1),
                    granularity="S5",
                    truth_roots=[root / "history"],
                    output_dir=output,
                    now_utc=start - timedelta(days=1),
                )

            value = _candidate()
            value["after_the_fact_threshold"] = 1
            with self.assertRaisesRegex(ValueError, "fields invalid"):
                forward._validate_candidate(value)

            weak = _candidate()
            weak["acceptance"]["min_samples"] = 1
            with self.assertRaisesRegex(ValueError, "weaker than proof hard floor"):
                forward._validate_candidate(weak)

    def test_lock_requires_one_dedicated_empty_truth_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate, training, forecast, output = self._fixture(root)
            start = datetime(2026, 7, 14, tzinfo=timezone.utc)
            truth_root = root / "history"
            truth_root.mkdir()
            (truth_root / "preexisting.jsonl").write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must be empty"):
                forward.create_lock(
                    candidate_path=candidate,
                    training_report_path=training,
                    forecast_history_path=forecast,
                    holdout_from=start,
                    holdout_to=start + timedelta(minutes=1),
                    granularity="S5",
                    truth_roots=[truth_root],
                    output_dir=output,
                    now_utc=start - timedelta(days=1),
                )
            (truth_root / "preexisting.jsonl").unlink()
            with self.assertRaisesRegex(ValueError, "exactly one"):
                forward.create_lock(
                    candidate_path=candidate,
                    training_report_path=training,
                    forecast_history_path=forecast,
                    holdout_from=start,
                    holdout_to=start + timedelta(minutes=1),
                    granularity="S5",
                    truth_roots=[truth_root, root / "other-history"],
                    output_dir=output,
                    now_utc=start - timedelta(days=1),
                )

    def test_unregistered_lock_and_invalid_suffix_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate, training, forecast, output = self._fixture(root)
            start = datetime(2026, 7, 14, tzinfo=timezone.utc)
            locked = forward.create_lock(
                candidate_path=candidate,
                training_report_path=training,
                forecast_history_path=forecast,
                holdout_from=start,
                holdout_to=start + timedelta(minutes=1),
                granularity="S5",
                truth_roots=[root / "history"],
                output_dir=output,
                now_utc=start - timedelta(days=1),
            )
            (output / "cohort_registry.jsonl").unlink()
            with self.assertRaisesRegex(ValueError, "registry missing"):
                forward.evaluate_lock(
                    lock_path=Path(locked["lock_path"]),
                    output_dir=output,
                    now_utc=start,
                )
            with self.assertRaisesRegex(ValueError, "non-object"):
                forward._validate_append_order(b"[]\n")

    def test_receipt_rejects_impossible_emission_time(self) -> None:
        start = datetime(2026, 7, 14, tzinfo=timezone.utc)
        blockers = forward._forecast_receipt_blockers(
            forecast_path=Path(__file__),
            selected=[],
            receipts=[
                {
                    "pair": "EUR_USD",
                    "forecast_timestamp_utc": forward._iso(start),
                    "recorded_at_utc": forward._iso(start - timedelta(minutes=2)),
                    "cycle_id": "cycle-1",
                }
            ],
            pair="EUR_USD",
            forecast_from=start,
            forecast_to=start + timedelta(minutes=1),
        )

        self.assertIn("FORECAST_EMISSION_TIME_INCONSISTENT", blockers)

    def test_evaluate_scores_only_locked_fixed_exit_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate, training, forecast, output = self._fixture(root)
            start = datetime(2026, 7, 14, tzinfo=timezone.utc)
            end = start + timedelta(minutes=1)
            locked = forward.create_lock(
                candidate_path=candidate,
                training_report_path=training,
                forecast_history_path=forecast,
                holdout_from=start,
                holdout_to=end,
                granularity="S5",
                truth_roots=[root / "history"],
                output_dir=output,
                now_utc=start - timedelta(days=1),
            )
            forecast_row = SimpleNamespace(
                pair="EUR_USD",
                direction="UP",
                confidence=0.8,
                raw_confidence=0.8,
                calibration_multiplier=1.0,
                current_price=1.1001,
                target_price=1.1004,
                invalidation_price=1.0998,
                horizon_min=1,
                up_score=50,
                down_score=10,
                range_score=5,
                drivers_for=("test",),
                drivers_against=(),
                rationale_summary="test",
                technical_context_v1=_context(),
            )
            with patch(
                "quant_rabbit.strategy.forecast_persistence_tracker._receipt_recorded_at_utc",
                return_value=start + timedelta(minutes=1),
            ):
                self.assertTrue(
                    record_forecast(
                        forecast_row,
                        data_root=root,
                        cycle_id="forward-cycle",
                        now=start,
                    )
                )
            mature = end + timedelta(minutes=1, seconds=5)

            with patch.object(
                forward,
                "_run_locked_truth_fetch_subprocess",
                side_effect=_fake_locked_truth_fetch,
            ) as fetch:
                first = forward.evaluate_lock(
                    lock_path=Path(locked["lock_path"]),
                    output_dir=output,
                    now_utc=mature,
                )
                second = forward.evaluate_lock(
                    lock_path=Path(locked["lock_path"]),
                    output_dir=output,
                    now_utc=mature + timedelta(hours=1),
                )
            fetch.assert_called_once()
            command = fetch.call_args.args[0]
            self.assertEqual(command[command.index("--pairs") + 1], "EUR_USD")
            self.assertEqual(command[command.index("--granularities") + 1], "S5")
            self.assertEqual(command[command.index("--price") + 1], "BA")
            self.assertEqual(command[command.index("--from") + 1], forward._iso(start))
            self.assertEqual(command[command.index("--to") + 1], forward._iso(mature))

            self.assertEqual(first["status"], "COMPLETE")
            self.assertEqual(first["result_sha256"], second["result_sha256"])
            report = json.loads(Path(first["result_path"]).read_text())
            self.assertEqual(report["validation"]["fixed_exit"]["take_profit_pips"], 1.0)
            self.assertEqual(report["validation"]["fixed_exit"]["stop_loss_pips"], 1.0)
            self.assertFalse(report["live_permission_granted"])
            self.assertEqual(
                report["cohort"]["truth_acquisition_receipts"]["verified_files"],
                1,
            )
            self.assertNotIn("TRUTH_ACQUISITION_UNPROVEN_FILE", report["proof_blockers"])

    def test_evaluate_rejects_truth_root_prepopulation_without_running_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate, training, forecast, output = self._fixture(root)
            start = datetime(2026, 7, 14, tzinfo=timezone.utc)
            end = start + timedelta(minutes=1)
            truth_root = root / "history"
            locked = forward.create_lock(
                candidate_path=candidate,
                training_report_path=training,
                forecast_history_path=forecast,
                holdout_from=start,
                holdout_to=end,
                granularity="S5",
                truth_roots=[truth_root],
                output_dir=output,
                now_utc=start - timedelta(days=1),
            )
            (truth_root / "operator_supplied.jsonl").write_text("{}\n", encoding="utf-8")
            mature = end + timedelta(minutes=1, seconds=5)
            with patch.object(
                forward,
                "_run_locked_truth_fetch_subprocess",
                side_effect=_fake_locked_truth_fetch,
            ) as fetch:
                with self.assertRaisesRegex(ValueError, "prepopulated"):
                    forward.evaluate_lock(
                        lock_path=Path(locked["lock_path"]),
                        output_dir=output,
                        now_utc=mature,
                    )
            fetch.assert_not_called()

    def test_evaluate_rejects_tampered_truth_acquisition_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate, training, forecast, output = self._fixture(root)
            start = datetime(2026, 7, 14, tzinfo=timezone.utc)
            end = start + timedelta(minutes=1)
            truth_root = root / "history"
            locked = forward.create_lock(
                candidate_path=candidate,
                training_report_path=training,
                forecast_history_path=forecast,
                holdout_from=start,
                holdout_to=end,
                granularity="S5",
                truth_roots=[truth_root],
                output_dir=output,
                now_utc=start - timedelta(days=1),
            )
            mature = end + timedelta(minutes=1, seconds=5)
            with patch.object(
                forward,
                "_run_locked_truth_fetch_subprocess",
                side_effect=_fake_locked_truth_fetch,
            ) as fetch:
                forward.evaluate_lock(
                    lock_path=Path(locked["lock_path"]),
                    output_dir=output,
                    now_utc=mature,
                )
                marker_path = next(truth_root.glob(forward.TRUTH_MARKER_GLOB))
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                marker["command"][marker["command"].index("BA")] = "M"
                marker_path.write_text(json.dumps(marker), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "marker digest mismatch"):
                    forward.evaluate_lock(
                        lock_path=Path(locked["lock_path"]),
                        output_dir=output,
                        now_utc=mature + timedelta(hours=1),
                    )
            fetch.assert_called_once()

    def test_authoritative_head_revokes_prior_result_after_material_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate, training, forecast, output = self._fixture(root)
            start = datetime(2026, 7, 14, tzinfo=timezone.utc)
            locked = forward.create_lock(
                candidate_path=candidate,
                training_report_path=training,
                forecast_history_path=forecast,
                holdout_from=start,
                holdout_to=start + timedelta(minutes=1),
                granularity="S5",
                truth_roots=[root / "history"],
                output_dir=output,
                now_utc=start - timedelta(days=1),
            )
            lock_path = Path(locked["lock_path"])
            lock = json.loads(lock_path.read_text())
            lock_sha = str(lock["lock_sha256"])
            result_dir = output / "forward_results" / lock_sha
            result_dir.mkdir(parents=True)
            first_body = {
                "schema_version": forward.RESULT_SCHEMA_VERSION,
                "contract_kind": forward.RESULT_KIND,
                "status": "COMPLETE",
                "semantics_version": forward.SEMANTICS_VERSION,
                "lock_ref": f"{lock_path.resolve()}#sha256={lock_sha}",
                "lock_sha256": lock_sha,
                "experiment_id": "0" * 64,
                "cohort": {
                    "forecast_from_utc_inclusive": lock["cohort"]["forecast_from_utc_inclusive"],
                    "forecast_to_utc_exclusive": lock["cohort"]["forecast_to_utc_exclusive"],
                    "matures_at_utc": lock["cohort"]["matures_at_utc"],
                },
                "candidate": lock["candidate"],
                "validation": {},
                "acceptance_checks": [
                    {"name": name, "passed": True}
                    for name in sorted(forward.ACCEPTANCE_KEYS)
                ],
                "proof_eligible": True,
                "proof_blockers": [],
                "authoritative_head_required": True,
                "live_permission_granted": False,
            }
            first_sha = forward._content_sha256(first_body)
            first = {**first_body, "result_sha256": first_sha}
            first_path = result_dir / f"{first_sha}.json"
            first_path.write_text(forward._canonical_pretty(first))
            forward._register_result(
                result_dir / "result_registry.jsonl",
                lock_sha256=lock_sha,
                result_sha256=first_sha,
            )
            forward._write_result_head(
                result_dir=result_dir,
                lock_sha256=lock_sha,
                current_result_sha256=first_sha,
                current_report=first,
            )
            first_valid, first_blockers = forward.verify_forward_result(first_path)
            self.assertFalse(first_valid)
            self.assertIn("FORWARD_RESULT_RECOMPUTE_MISMATCH", first_blockers)

            copied_dir = root / "copied_result"
            copied_dir.mkdir()
            copied_path = copied_dir / first_path.name
            copied_path.write_text(first_path.read_text())
            with self.assertRaisesRegex(ValueError, "canonical lock event"):
                forward._register_result(
                    copied_dir / "result_registry.jsonl",
                    lock_sha256=lock_sha,
                    result_sha256=first_sha,
                )

            second_body = {
                "schema_version": forward.RESULT_SCHEMA_VERSION,
                "contract_kind": forward.RESULT_KIND,
                "status": "COMPLETE",
                "semantics_version": forward.SEMANTICS_VERSION,
                "lock_ref": f"{lock_path.resolve()}#sha256={lock_sha}",
                "lock_sha256": lock_sha,
                "cohort": {
                    "forecast_from_utc_inclusive": lock["cohort"]["forecast_from_utc_inclusive"],
                    "forecast_to_utc_exclusive": lock["cohort"]["forecast_to_utc_exclusive"],
                    "matures_at_utc": lock["cohort"]["matures_at_utc"],
                },
                "candidate": lock["candidate"],
                "proof_eligible": False,
                "authoritative_head_required": True,
                "live_permission_granted": False,
                "proof_blockers": ["HOLDOUT_BACKFILL_CHANGED_RESULT"],
            }
            second_sha = forward._content_sha256(second_body)
            second = {**second_body, "result_sha256": second_sha}
            (result_dir / f"{second_sha}.json").write_text(forward._canonical_pretty(second))
            forward._register_result(
                result_dir / "result_registry.jsonl",
                lock_sha256=lock_sha,
                result_sha256=second_sha,
            )
            forward._write_result_head(
                result_dir=result_dir,
                lock_sha256=lock_sha,
                current_result_sha256=second_sha,
                current_report=second,
            )

            valid, blockers = forward.verify_forward_result(first_path)
            self.assertFalse(valid)
            self.assertIn("FORWARD_RESULT_NOT_AUTHORITATIVE_HEAD", blockers)
            self.assertIn("FORWARD_RESULT_MULTIPLE_MATERIAL_RESULTS", blockers)

            (result_dir / "result_registry.jsonl").unlink()
            (result_dir / "head.json").unlink()
            (result_dir / f"{second_sha}.json").unlink()
            with self.assertRaisesRegex(ValueError, "global result registry binding conflict"):
                forward._register_result(
                    result_dir / "result_registry.jsonl",
                    lock_sha256=lock_sha,
                    result_sha256=first_sha,
                )

    def test_evaluate_rejects_locked_prefix_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate, training, forecast, output = self._fixture(root)
            forecast.write_text("{}\n", encoding="utf-8")
            start = datetime(2026, 7, 14, tzinfo=timezone.utc)
            locked = forward.create_lock(
                candidate_path=candidate,
                training_report_path=training,
                forecast_history_path=forecast,
                holdout_from=start,
                holdout_to=start + timedelta(minutes=1),
                granularity="S5",
                truth_roots=[root / "history"],
                output_dir=output,
                now_utc=start - timedelta(days=1),
            )
            forecast.write_text("{ }\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "locked prefix changed"):
                forward.evaluate_lock(
                    lock_path=Path(locked["lock_path"]),
                    output_dir=output,
                    now_utc=start + timedelta(days=1),
                )


if __name__ == "__main__":
    unittest.main()
