from __future__ import annotations

import contextlib
import copy
import gzip
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run-adaptive-story-s5-grid.py"
UTC = timezone.utc


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _load_script():
    name = "test_run_adaptive_story_s5_grid_script"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("adaptive story CLI could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _row(pair: str, timestamp: datetime, *, bid: float) -> dict:
    ask = bid + (0.02 if pair.endswith("JPY") else 0.0002)
    return {
        "ask": {"c": ask, "h": ask, "l": ask, "o": ask},
        "bid": {"c": bid, "h": bid, "l": bid, "o": bid},
        "complete": True,
        "granularity": "S5",
        "pair": pair,
        "price": "BA",
        "time": timestamp.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
        "volume": 1,
    }


def _write_pair_cache(
    root: Path,
    *,
    pair: str,
    run_id: str,
    declared_from: datetime,
    declared_to: datetime,
) -> None:
    pair_dir = root / run_id / pair
    pair_dir.mkdir(parents=True)
    filename = (
        f"{pair}_S5_BA_{declared_from.strftime('%Y%m%dT%H%M%SZ')}_"
        f"{declared_to.strftime('%Y%m%dT%H%M%SZ')}.jsonl.gz"
    )
    candle_path = pair_dir / filename
    rows = [
        _row(pair, declared_from, bid=150.0 if pair.endswith("JPY") else 1.1),
        _row(
            pair,
            declared_from + timedelta(minutes=50),
            bid=150.01 if pair.endswith("JPY") else 1.101,
        ),
    ]
    with gzip.open(candle_path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    task = {
        "compressed": True,
        "dry_run": False,
        "errors": [],
        "from": declared_from.isoformat().replace("+00:00", "Z"),
        "granularity": "S5",
        "pair": pair,
        "partial_path": None,
        "path": f"history/{run_id}/{pair}/{filename}",
        "price": "BA",
        "published": True,
        "requests": 1,
        "rows": len(rows),
        "to": declared_to.isoformat().replace("+00:00", "Z"),
        "windows": 1,
    }
    summary = {
        "dry_run": False,
        "errors": [],
        "generated_at_utc": declared_to.isoformat(),
        "granularities": ["S5"],
        "pairs": [pair],
        "price": "BA",
        "tasks": [task],
        "total_rows": len(rows),
        "window": {"from": task["from"], "to": task["to"]},
    }
    (root / run_id / "summary.json").write_text(
        json.dumps(summary, sort_keys=True),
        encoding="utf-8",
    )
    receipt_path = root / "truth_acquisition_receipts.jsonl"
    prior = [
        json.loads(line)
        for line in (
            receipt_path.read_text(encoding="utf-8").splitlines()
            if receipt_path.exists()
            else ()
        )
        if line.strip()
    ]
    fetch_script = ROOT / "scripts" / "oanda_history_fetch.py"
    body = {
        "schema_version": "QR_OANDA_TRUTH_ACQUISITION_RECEIPT_V1",
        "sequence": len(prior) + 1,
        "recorded_at_utc": declared_to.isoformat().replace("+00:00", "Z"),
        "output_root": str(root.resolve()),
        "candle_path": str(candle_path.resolve(strict=True)),
        "candle_sha256": hashlib.sha256(candle_path.read_bytes()).hexdigest(),
        "pair": pair,
        "granularity": "S5",
        "price_component": "BA",
        "window": {
            "from_utc": task["from"],
            "to_utc": task["to"],
        },
        "rows": len(rows),
        "fetch_script_path": str(fetch_script.resolve(strict=True)),
        "fetch_script_sha256": hashlib.sha256(fetch_script.read_bytes()).hexdigest(),
        "previous_receipt_sha256": (prior[-1]["receipt_sha256"] if prior else None),
    }
    receipt = {**body, "receipt_sha256": _canonical_sha(body)}
    with receipt_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")


def _train_argv(history: Path, output: Path) -> list[str]:
    return [
        "train",
        "--history-root",
        str(history),
        "--history-run-ids",
        "20260501T010000Z,20260501T010100Z",
        "--pairs",
        "EUR_USD,GBP_JPY",
        "--train-from",
        "2026-05-01T00:00:00Z",
        "--train-to",
        "2026-05-01T00:10:00Z",
        "--validation-from",
        "2026-05-01T00:15:00Z",
        "--validation-to",
        "2026-05-01T00:25:00Z",
        "--holdout-from",
        "2026-05-01T00:30:00Z",
        "--holdout-to",
        "2026-05-01T00:40:00Z",
        "--workers",
        "1",
        "--output",
        str(output),
    ]


class AdaptiveStoryS5GridCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cli = _load_script()

    def _history(self, parent: Path) -> Path:
        root = parent / "history"
        root.mkdir()
        start = datetime(2026, 5, 1, tzinfo=UTC)
        end = start + timedelta(hours=1)
        _write_pair_cache(
            root,
            pair="EUR_USD",
            run_id="20260501T010000Z",
            declared_from=start,
            declared_to=end,
        )
        _write_pair_cache(
            root,
            pair="GBP_JPY",
            run_id="20260501T010100Z",
            declared_from=start,
            declared_to=end,
        )
        return root

    def _drifted_evaluator_seal(self, value: dict) -> dict:
        drifted = copy.deepcopy(value)
        drifted["python"]["version"] = "0.0.0-test-drift"
        body = dict(drifted)
        body.pop("evaluator_dependency_sha256", None)
        drifted["evaluator_dependency_sha256"] = self.cli._canonical_sha(body)
        return drifted

    @staticmethod
    def _set_workers(argv: list[str], workers: int) -> list[str]:
        result = list(argv)
        index = result.index("--workers")
        result[index + 1] = str(workers)
        return result

    def _fake_run(self, pair, _candles, splits, **kwargs):
        phase = splits[0].name
        split = splits[0]
        requested = kwargs.get("candidate_ids")
        vehicles = self.cli.story_core.build_story_vehicle_catalog_v2()
        if requested is None:
            selected = [item for item in vehicles if not item.no_trade_control]
        else:
            by_id = {item.candidate_id: item for item in vehicles}
            selected = [by_id[item] for item in requested]
        trials = []
        for vehicle in selected:
            if phase == "TRAIN":
                if vehicle.candidate_id == "H21:PROFIT_FIRST_24H":
                    mean = 1.1 if pair == "EUR_USD" else 0.9
                elif vehicle.candidate_id == "H21:TIME_1H":
                    mean = 0.95
                elif vehicle.exit_policy_id == "TIME_1H":
                    mean = 0.2
                else:
                    mean = 0.1
            else:
                mean = 0.25
            trials.append(
                {
                    "candidate_id": vehicle.candidate_id,
                    "hypothesis_id": vehicle.hypothesis_id,
                    "story_name": vehicle.story_name,
                    "exit_policy_id": vehicle.exit_policy_id,
                    "contextual_order_policy": vehicle.contextual_order_policy,
                    "allowed_order_modes": list(vehicle.allowed_order_modes),
                    "max_hold_seconds": vehicle.max_hold_seconds,
                    "profit_target_r": vehicle.profit_target_r,
                    "trailing_structural": vehicle.trailing_structural,
                    "complexity": vehicle.complexity,
                    "no_trade_control": False,
                    "scorecard_eligible": True,
                    "by_split": {
                        phase: {
                            "resolved_count": 10,
                            "exact_net_r": mean * 10,
                        }
                    },
                    "daily_aggregates_by_split": {
                        phase: [
                            {
                                "utc_date": utc_date,
                                "filled_count": 0,
                                "resolved_count": 0,
                                "exact_net_r": 0.0,
                                "gross_profit_r": 0.0,
                                "gross_loss_r": 0.0,
                            }
                            for utc_date in self.cli._expected_utc_day_labels(split)
                        ]
                    },
                }
            )
        candidate_ids = [vehicle.candidate_id for vehicle in selected]
        split_rows = self.cli._expected_split_rows(split)
        body = {
            "contract": self.cli.story_core.STORY_GRID_CONTRACT_V2,
            "schema_version": 2,
            "status": "COMPLETE",
            "pair": pair,
            "story_catalog_policy": self.cli.story_core.STORY_CATALOG_POLICY_V2,
            "truth_policy": self.cli.story_core.STORY_TRUTH_POLICY_V2,
            "story_catalog_sha256": self.cli._canonical_sha(
                self.cli.story_core._story_catalog_receipt_v2()
            ),
            "truth_evaluator_sha256": self.cli._canonical_sha(
                self.cli.story_core._truth_evaluator_receipt_v2()
            ),
            "price_precision_policy": (self.cli.story_core.PRICE_PRECISION_POLICY_V2),
            "price_cost_scope": dict(self.cli.story_core.PRICE_COST_SCOPE_V2),
            "entry_ttl_boundary": "EXCLUSIVE",
            "intrabar_resting_fill_s5_policy": (
                "NO_TARGET;STOP_RANGE_CHARGED_CONSERVATIVELY;NO_PREFILL_OPEN_GAP"
            ),
            "entry_gap_invalid_geometry_policy": (
                "BROKER_ON_FILL_DEPENDENT_ORDER_LOSS_CANCEL_NO_FILL"
            ),
            "requested_candidate_ids": candidate_ids,
            "evaluated_candidate_ids": candidate_ids,
            "requested_control_candidate_ids": [],
            "candidate_whitelist_sha256": self.cli._canonical_sha(candidate_ids),
            "split_receipt": split_rows,
            "split_digest": self.cli._canonical_sha(split_rows),
            "daily_aggregates_complete": True,
            "daily_cluster_basis": "ENTRY_UTC_DATE",
            "exit_day_or_mark_to_market_used_for_selection": False,
            "candidate_count": len(candidate_ids),
            "contextual_order_cross_product_forbidden": True,
            "setup_trigger_entry_policy": "T_SETUP_LT_T_TRIGGER_LT_T_ENTRY",
            "quote_observation_policy": (
                "FIRST_REAL_S5_AFTER_TRIGGER_OBSERVES_ONLY;"
                "FOLLOWING_REAL_S5_IS_EARLIEST_FILL"
            ),
            "all_trials": trials,
            **self.cli._AUTHORITY,
        }
        return {**body, "result_sha256": self.cli._canonical_sha(body)}

    def _fake_combine(self, pair_runs, splits, *, candidate_ids):
        split = splits[0]
        dates = self.cli._expected_utc_day_labels(split)
        rows = []
        for index, candidate_id in enumerate(candidate_ids):
            hypothesis_id, exit_policy_id = candidate_id.split(":", 1)
            daily = []
            for day_index, utc_date in enumerate(dates):
                if hypothesis_id == "H21" and exit_policy_id == "PROFIT_FIRST_24H":
                    exact_net_r = 1.1 if (day_index // 2) % 2 == 0 else 0.9
                elif hypothesis_id == "H21" and exit_policy_id == "TIME_1H":
                    exact_net_r = 0.95
                elif (
                    split.name == "TRAIN"
                    and hypothesis_id == "H22"
                    and exit_policy_id == "TIME_1H"
                ):
                    exact_net_r = -0.2
                elif exit_policy_id == "TIME_1H":
                    exact_net_r = 0.2
                else:
                    exact_net_r = 0.1
                daily.append(
                    {
                        "utc_date": utc_date,
                        "exact_net_r": exact_net_r,
                        "resolved_count": 31,
                    }
                )
            eligible = index < 2
            gates = {
                "resolved_trade_floor_passed": eligible,
                "active_entry_day_floor_passed": eligible,
                "contributing_pair_floor_passed": eligible,
                "average_net_r_positive": eligible,
                "average_daily_net_r_positive": eligible,
                "profit_factor_r_above_one": eligible,
                "loocv_each_day_removed_total_r_positive": eligible,
                "no_unresolved_or_purged": eligible,
            }
            total_r = sum(item["exact_net_r"] for item in daily)
            resolved = sum(item["resolved_count"] for item in daily)
            vehicle = next(
                item
                for item in self.cli.story_core.build_story_vehicle_catalog_v2()
                if item.candidate_id == candidate_id
            )
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "hypothesis_id": vehicle.hypothesis_id,
                    "story_name": vehicle.story_name,
                    "exit_policy_id": vehicle.exit_policy_id,
                    "no_trade_control": False,
                    "economic_screen_by_split": {
                        split.name: {
                            "resolved_count": resolved,
                            "active_entry_day_count": 8,
                            "contributing_pair_count": 4,
                            "resolved_pair_count": 4,
                            "gates": gates,
                            "eligible": eligible,
                            "screen_is_statistical_proof": False,
                        }
                    },
                    "by_split": {
                        split.name: {
                            "resolved_count": resolved,
                            "unresolved_or_purged_count": 0,
                            "active_entry_day_count": 8,
                            "contributing_pair_count": 4,
                            "exact_net_r": total_r,
                            "gross_profit_r": max(total_r, 0.0),
                            "gross_loss_r": max(-total_r, 0.0),
                            "average_daily_net_r": total_r / len(daily),
                            "daily_net_r": daily,
                        }
                    },
                }
            )
        split_rows = self.cli._expected_split_rows(split)
        pairs = sorted(str(row.get("pair") or "") for row in pair_runs)
        body = {
            "contract": self.cli.story_core.STORY_GRID_COMBINED_CONTRACT_V2,
            "schema_version": 2,
            "status": "COMPLETE",
            "pair_count": len(pairs),
            "pairs": pairs,
            "requested_candidate_ids": list(candidate_ids),
            "evaluated_candidate_ids": list(candidate_ids),
            "candidate_whitelist_sha256": self.cli._canonical_sha(list(candidate_ids)),
            "story_catalog_policy": self.cli.story_core.STORY_CATALOG_POLICY_V2,
            "truth_policy": self.cli.story_core.STORY_TRUTH_POLICY_V2,
            "story_catalog_sha256": self.cli._canonical_sha(
                self.cli.story_core._story_catalog_receipt_v2()
            ),
            "truth_evaluator_sha256": self.cli._canonical_sha(
                self.cli.story_core._truth_evaluator_receipt_v2()
            ),
            "accepted_pair_run_statuses": list(
                self.cli.story_core.PAIR_RUN_ALLOWED_STATUSES
            ),
            "price_precision_policy": (self.cli.story_core.PRICE_PRECISION_POLICY_V2),
            "price_cost_scope": dict(self.cli.story_core.PRICE_COST_SCOPE_V2),
            "split_receipt": split_rows,
            "split_digest": self.cli._canonical_sha(split_rows),
            "daily_cluster_basis": "ENTRY_UTC_DATE",
            "daily_zero_fill_policy": "ALL_SPLIT_CALENDAR_UTC_DAYS",
            "daily_aggregates_source": ("COMPLETE_PAIR_RUN_AGGREGATES_NOT_AUDIT_ROWS"),
            "candidate_metrics": rows,
            "economic_survivor_ids": list(candidate_ids[:2]),
            "economic_screen_is_statistical_proof": False,
            **self.cli._AUTHORITY,
        }
        return {**body, "result_sha256": self.cli._canonical_sha(body)}

    def test_train_one_day_high_mean_defaults_to_time_1h(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            history = self._history(parent)
            output = parent / "train.json"
            with (
                mock.patch.object(
                    self.cli.story_core,
                    "run_adaptive_story_s5_grid",
                    side_effect=self._fake_run,
                ) as runner,
                mock.patch.object(
                    self.cli,
                    "load_historical_s5_slice",
                    wraps=self.cli.load_historical_s5_slice,
                ) as loader,
                mock.patch.object(
                    self.cli.story_core,
                    "combine_adaptive_story_s5_grid_runs",
                    side_effect=self._fake_combine,
                    create=True,
                ) as combiner,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                status = self.cli.main(_train_argv(history, output))

            self.assertEqual(status, 0)
            self.assertEqual(runner.call_count, 2)
            expected_ids = tuple(
                item.candidate_id
                for item in self.cli.story_core.build_story_vehicle_catalog_v2()
                if not item.no_trade_control
            )
            self.assertEqual(
                [tuple(call.kwargs["candidate_ids"]) for call in runner.call_args_list],
                [expected_ids, expected_ids],
            )
            self.assertEqual(loader.call_count, 2)
            self.assertEqual(combiner.call_count, 1)
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(receipt["executed_candidate_ids"]), 50)
            self.assertEqual(len(receipt["next_phase_candidate_ids"]), 10)
            self.assertIsNone(
                receipt["selection"]["story_selections"][0]["best_candidate_id"],
            )
            self.assertEqual(
                receipt["selection"]["story_selections"][0][
                    "observed_mean_leader_candidate_id"
                ],
                "H21:PROFIT_FIRST_24H",
            )
            self.assertEqual(
                receipt["selection"]["story_selections"][0]["selected_candidate_id"],
                "H21:TIME_1H",
            )
            self.assertEqual(
                receipt["selection"]["story_selections"][0]["selection_basis"],
                "TRAIN_EXIT_DEFAULT_INSUFFICIENT_CLUSTERS",
            )
            body = dict(receipt)
            digest = body.pop("receipt_sha256")
            self.assertEqual(digest, self.cli._canonical_sha(body))
            self.assertEqual(receipt["order_authority"], "NONE")
            self.assertFalse(receipt["live_permission"])
            self.assertFalse(receipt["fully_loaded_net_economics"])
            self.assertTrue(receipt["integrity_evidence_not_external_authentication"])
            self.assertFalse(
                receipt["price_only_cost_scope"]["order_book_vwap_modeled"]
            )
            self.assertFalse(receipt["price_only_cost_scope"]["latency_modeled"])
            self.assertFalse(receipt["price_only_cost_scope"]["financing_modeled"])
            self.assertFalse(receipt["price_only_cost_scope"]["commission_modeled"])
            dependency = receipt["evaluator_dependency_seal"]
            self.assertEqual(
                [row["relative_path"] for row in dependency["source_dependencies"]],
                list(self.cli._DEPENDENCY_RELATIVE_PATHS),
            )
            self.assertEqual(
                dependency["selection_semantics"]["selectable_candidate_count"],
                50,
            )
            self.assertEqual(
                dependency["dst_schedule"]["timezone_order"],
                ["Europe/London", "America/New_York"],
            )
            self.assertEqual(receipt["frozen_manifest"]["missing_pairs"], [])
            self.assertTrue(
                receipt["frozen_manifest"]["all_selected_sources_acquisition_receipted"]
            )

    def test_real_core_phases_preserve_whitelists_and_empty_portfolio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            history = self._history(parent)
            output = parent / "real-train.json"
            with contextlib.redirect_stdout(io.StringIO()):
                status = self.cli.main(_train_argv(history, output))

            self.assertEqual(status, 0)
            receipt = json.loads(output.read_text(encoding="utf-8"))
            expected_ids = receipt["executed_candidate_ids"]
            self.assertEqual(len(expected_ids), 50)
            self.assertNotIn("H31:NO_TRADE_CONTROL", expected_ids)
            self.assertEqual(
                receipt["global_result"]["requested_candidate_ids"],
                expected_ids,
            )
            for artifact in receipt["pair_artifacts"]:
                self.assertEqual(
                    artifact["result"]["requested_candidate_ids"],
                    expected_ids,
                )
            self.assertTrue(
                all(
                    row["selection_basis"] == "TRAIN_EXIT_DEFAULT_INSUFFICIENT_CLUSTERS"
                    for row in receipt["selection"]["story_selections"]
                )
            )
            validation_output = parent / "real-validation.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    self.cli.main(
                        [
                            "validation",
                            "--history-root",
                            str(history),
                            "--train-receipt",
                            str(output),
                            "--workers",
                            "1",
                            "--output",
                            str(validation_output),
                        ]
                    ),
                    0,
                )
            validation = json.loads(validation_output.read_text(encoding="utf-8"))
            self.assertEqual(validation["next_phase_candidate_ids"], [])
            self.assertEqual(
                validation["fixed_portfolio"]["portfolio_spec"]["candidate_ids"],
                [],
            )
            holdout_output = parent / "real-holdout.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    self.cli.main(
                        [
                            "holdout",
                            "--history-root",
                            str(history),
                            "--train-receipt",
                            str(output),
                            "--validation-receipt",
                            str(validation_output),
                            "--workers",
                            "1",
                            "--output",
                            str(holdout_output),
                        ]
                    ),
                    0,
                )
            holdout = json.loads(holdout_output.read_text(encoding="utf-8"))
            self.assertEqual(holdout["status"], "NO_VALIDATION_SURVIVORS")
            self.assertEqual(holdout["executed_candidate_ids"], [])
            self.assertEqual(holdout["primary_result"]["candidate_count"], 0)
            self.assertEqual(
                holdout["primary_result"]["zero_day_count"],
                holdout["primary_result"]["utc_calendar_day_count"],
            )

    def test_train_selection_uses_two_day_block_one_se(self) -> None:
        split = self.cli.story_core.UtcSplit(
            name="TRAIN",
            from_utc=datetime(2026, 5, 1, tzinfo=UTC),
            to_utc=datetime(2026, 5, 4, tzinfo=UTC),
        )
        candidate_ids = tuple(
            item.candidate_id
            for item in self.cli.story_core.build_story_vehicle_catalog_v2()
            if not item.no_trade_control
        )
        global_result = self._fake_combine((), (split,), candidate_ids=candidate_ids)
        pair_results = [
            self._fake_run("EUR_USD", (), (split,)),
            self._fake_run("GBP_JPY", (), (split,)),
        ]
        selection = self.cli._train_selection(
            global_result,
            pair_results,
            split,
        )
        first = selection["story_selections"][0]
        self.assertEqual(first["best_candidate_id"], "H21:PROFIT_FIRST_24H")
        self.assertEqual(first["selected_candidate_id"], "H21:TIME_1H")
        self.assertEqual(first["selection_basis"], "OBSERVED_ONE_SE")
        metric = selection["candidate_metrics"][0]
        self.assertEqual(metric["utc_calendar_day_count"], 3)
        self.assertEqual(
            metric["non_overlapping_two_day_block_cluster_count"],
            2,
        )
        self.assertEqual(metric["two_day_block_sizes"], [2, 1])
        self.assertAlmostEqual(metric["mean_daily_net_r"], 3.1 / 3.0)
        self.assertAlmostEqual(
            metric["two_day_block_cluster_robust_standard_error_daily_r"],
            4.0 / 45.0,
        )

    def test_train_daily_vector_integrity_fails_closed(self) -> None:
        split = self.cli.story_core.UtcSplit(
            name="TRAIN",
            from_utc=datetime(2026, 5, 1, tzinfo=UTC),
            to_utc=datetime(2026, 5, 4, tzinfo=UTC),
        )
        candidate_ids = tuple(
            item.candidate_id
            for item in self.cli.story_core.build_story_vehicle_catalog_v2()
            if not item.no_trade_control
        )
        original = self._fake_combine((), (split,), candidate_ids=candidate_ids)

        def daily(result):
            return result["candidate_metrics"][0]["by_split"]["TRAIN"]["daily_net_r"]

        cases = {}
        missing = json.loads(json.dumps(original))
        daily(missing).pop()
        cases["missing"] = missing
        duplicate = json.loads(json.dumps(original))
        daily(duplicate)[-1] = dict(daily(duplicate)[0])
        cases["duplicate"] = duplicate
        unordered = json.loads(json.dumps(original))
        daily(unordered)[0], daily(unordered)[1] = (
            daily(unordered)[1],
            daily(unordered)[0],
        )
        cases["unordered"] = unordered
        non_finite = json.loads(json.dumps(original))
        daily(non_finite)[0]["exact_net_r"] = float("inf")
        cases["non_finite"] = non_finite

        for label, result in cases.items():
            with self.subTest(label=label), self.assertRaises(
                self.cli.AdaptiveStoryCliError
            ):
                self.cli._daily_train_metrics(result, split)

    def test_validation_and_holdout_execute_only_prior_sealed_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            history = self._history(parent)
            train_path = parent / "train.json"
            validation_path = parent / "validation.json"
            holdout_path = parent / "holdout.json"
            with (
                mock.patch.object(
                    self.cli.story_core,
                    "run_adaptive_story_s5_grid",
                    side_effect=self._fake_run,
                ),
                mock.patch.object(
                    self.cli.story_core,
                    "combine_adaptive_story_s5_grid_runs",
                    side_effect=self._fake_combine,
                    create=True,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(self.cli.main(_train_argv(history, train_path)), 0)

            train = json.loads(train_path.read_text(encoding="utf-8"))
            validation_calls: list[tuple[str, ...]] = []

            def validation_run(*args, **kwargs):
                validation_calls.append(tuple(kwargs["candidate_ids"]))
                return self._fake_run(*args, **kwargs)

            with (
                mock.patch.object(
                    self.cli.story_core,
                    "run_adaptive_story_s5_grid",
                    side_effect=validation_run,
                ),
                mock.patch.object(
                    self.cli.story_core,
                    "combine_adaptive_story_s5_grid_runs",
                    side_effect=self._fake_combine,
                    create=True,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    self.cli.main(
                        [
                            "validation",
                            "--history-root",
                            str(history),
                            "--train-receipt",
                            str(train_path),
                            "--workers",
                            "1",
                            "--output",
                            str(validation_path),
                        ]
                    ),
                    0,
                )

            self.assertEqual(
                validation_calls, [tuple(train["next_phase_candidate_ids"])] * 2
            )
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            survivors = validation["next_phase_candidate_ids"]
            self.assertEqual(survivors, train["next_phase_candidate_ids"][:1])
            self.assertEqual(
                validation["selection"]["core_validation_economic_survivor_ids"],
                train["next_phase_candidate_ids"][:2],
            )
            second_gate = validation["selection"]["same_sign_gate_rows"][1]
            self.assertTrue(second_gate["core_validation_economic_screen_eligible"])
            self.assertFalse(
                second_gate["train_validation_same_positive_sign_gate_passed"]
            )
            self.assertEqual(
                second_gate["rejection_reason"],
                "TRAIN_VALIDATION_MEAN_DAILY_R_NOT_BOTH_POSITIVE",
            )
            self.assertEqual(
                validation["selection"]["fixed_portfolio_candidate_ids"],
                survivors,
            )
            validation_portfolio = validation["fixed_portfolio"]
            self.assertEqual(
                validation_portfolio["portfolio_spec"]["candidate_ids"],
                survivors,
            )
            self.assertEqual(
                validation_portfolio["portfolio_spec"]["allocation_label"],
                self.cli.PORTFOLIO_ALLOCATION,
            )
            self.assertEqual(validation_portfolio["candidate_count"], 1)
            self.assertAlmostEqual(
                validation_portfolio["total_exact_net_r"],
                0.95,
            )
            self.assertFalse(
                validation["selection"]["subset_or_weight_search_performed"]
            )
            self.assertEqual(len(validation["pair_artifacts"]), 2)

            holdout_calls: list[tuple[str, ...]] = []

            def holdout_run(*args, **kwargs):
                holdout_calls.append(tuple(kwargs["candidate_ids"]))
                return self._fake_run(*args, **kwargs)

            with (
                mock.patch.object(
                    self.cli.story_core,
                    "run_adaptive_story_s5_grid",
                    side_effect=holdout_run,
                ),
                mock.patch.object(
                    self.cli.story_core,
                    "combine_adaptive_story_s5_grid_runs",
                    side_effect=self._fake_combine,
                    create=True,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    self.cli.main(
                        [
                            "holdout",
                            "--history-root",
                            str(history),
                            "--train-receipt",
                            str(train_path),
                            "--validation-receipt",
                            str(validation_path),
                            "--workers",
                            "1",
                            "--output",
                            str(holdout_path),
                        ]
                    ),
                    0,
                )

            self.assertEqual(holdout_calls, [tuple(survivors)] * 2)
            holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
            self.assertEqual(holdout["executed_candidate_ids"], survivors)
            self.assertEqual(
                [
                    row["candidate_id"]
                    for row in holdout["global_result"]["candidate_metrics"]
                ],
                survivors,
            )
            self.assertFalse(holdout["selection"]["reselection_performed"])
            self.assertFalse(holdout["selection"]["non_winner_results_calculated"])
            self.assertFalse(holdout["selection"]["non_winner_results_published"])
            self.assertTrue(holdout["selection"]["fixed_portfolio_is_primary_result"])
            self.assertEqual(
                holdout["primary_result"]["portfolio_spec_sha256"],
                validation_portfolio["portfolio_spec_sha256"],
            )
            self.assertEqual(
                holdout["primary_result"]["portfolio_spec"]["candidate_ids"],
                survivors,
            )
            self.assertFalse(
                holdout["primary_result"]["subset_or_weight_search_performed"]
            )

    def test_downstream_parser_has_no_pair_or_split_override(self) -> None:
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as missing_train_error,
        ):
            self.cli._parse_args(
                [
                    "holdout",
                    "--history-root",
                    "history",
                    "--validation-receipt",
                    "validation.json",
                    "--output",
                    "holdout.json",
                ]
            )
        self.assertEqual(missing_train_error.exception.code, 2)
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as validation_error,
        ):
            self.cli._parse_args(
                [
                    "validation",
                    "--history-root",
                    "history",
                    "--train-receipt",
                    "train.json",
                    "--pairs",
                    "EUR_USD",
                    "--output",
                    "validation.json",
                ]
            )
        self.assertEqual(validation_error.exception.code, 2)
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as run_scope_error,
        ):
            self.cli._parse_args(
                [
                    "validation",
                    "--history-root",
                    "history",
                    "--train-receipt",
                    "train.json",
                    "--history-run-ids",
                    "20260501T010000Z",
                    "--output",
                    "validation.json",
                ]
            )
        self.assertEqual(run_scope_error.exception.code, 2)
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as holdout_error,
        ):
            self.cli._parse_args(
                [
                    "holdout",
                    "--history-root",
                    "history",
                    "--train-receipt",
                    "train.json",
                    "--validation-receipt",
                    "validation.json",
                    "--holdout-from",
                    "2026-05-02T00:00:00Z",
                    "--output",
                    "holdout.json",
                ]
            )
        self.assertEqual(holdout_error.exception.code, 2)

    def test_tampered_train_receipt_fails_before_core_or_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            history = self._history(parent)
            train_path = parent / "train.json"
            output = parent / "validation.json"
            with (
                mock.patch.object(
                    self.cli.story_core,
                    "run_adaptive_story_s5_grid",
                    side_effect=self._fake_run,
                ),
                mock.patch.object(
                    self.cli.story_core,
                    "combine_adaptive_story_s5_grid_runs",
                    side_effect=self._fake_combine,
                    create=True,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(self.cli.main(_train_argv(history, train_path)), 0)
            train = json.loads(train_path.read_text(encoding="utf-8"))
            train["next_phase_candidate_ids"] = list(
                reversed(train["next_phase_candidate_ids"])
            )
            train_path.write_text(json.dumps(train), encoding="utf-8")
            with (
                mock.patch.object(
                    self.cli.story_core,
                    "run_adaptive_story_s5_grid",
                ) as runner,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                status = self.cli.main(
                    [
                        "validation",
                        "--history-root",
                        str(history),
                        "--train-receipt",
                        str(train_path),
                        "--workers",
                        "1",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(status, 1)
            runner.assert_not_called()
            self.assertFalse(output.exists())

    def test_resealed_validation_cannot_forge_train_mean_to_hide_winner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            history = self._history(parent)
            train_path = parent / "train.json"
            validation_path = parent / "validation.json"
            holdout_path = parent / "holdout.json"
            with (
                mock.patch.object(
                    self.cli.story_core,
                    "run_adaptive_story_s5_grid",
                    side_effect=self._fake_run,
                ),
                mock.patch.object(
                    self.cli.story_core,
                    "combine_adaptive_story_s5_grid_runs",
                    side_effect=self._fake_combine,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(self.cli.main(_train_argv(history, train_path)), 0)
                self.assertEqual(
                    self.cli.main(
                        [
                            "validation",
                            "--history-root",
                            str(history),
                            "--train-receipt",
                            str(train_path),
                            "--workers",
                            "1",
                            "--output",
                            str(validation_path),
                        ]
                    ),
                    0,
                )

            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            self.assertTrue(validation["next_phase_candidate_ids"])
            selected_gate_rows = [
                row
                for row in validation["selection"]["same_sign_gate_rows"]
                if row["selected_as_final_survivor"]
            ]
            self.assertTrue(selected_gate_rows)
            for row in selected_gate_rows:
                row["train_mean_daily_net_r"] = -1.0
                row["train_validation_same_positive_sign_gate_passed"] = False
                row["selected_as_final_survivor"] = False
                row["rejection_reason"] = (
                    "TRAIN_VALIDATION_MEAN_DAILY_R_NOT_BOTH_POSITIVE"
                )
            scope = self.cli._scope_from_receipt(validation)
            split = self.cli._split_from_scope(scope, "VALIDATION")
            empty_spec = self.cli._new_portfolio_spec([])
            validation["fixed_portfolio"] = self.cli._aggregate_fixed_portfolio(
                validation["global_result"],
                split,
                empty_spec,
            )
            validation["selection"]["selected_candidate_ids"] = []
            validation["selection"]["fixed_portfolio_candidate_ids"] = []
            validation["selection"]["fixed_portfolio_spec_sha256"] = empty_spec[
                "portfolio_spec_sha256"
            ]
            validation["next_phase_candidate_ids"] = []
            body = dict(validation)
            body.pop("receipt_sha256")
            validation["receipt_sha256"] = self.cli._canonical_sha(body)
            validation_path.write_text(json.dumps(validation), encoding="utf-8")

            with (
                mock.patch.object(
                    self.cli.story_core,
                    "combine_adaptive_story_s5_grid_runs",
                    side_effect=self._fake_combine,
                ),
                mock.patch.object(self.cli, "load_historical_s5_slice") as loader,
                mock.patch.object(
                    self.cli.story_core,
                    "run_adaptive_story_s5_grid",
                ) as runner,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                status = self.cli.main(
                    [
                        "holdout",
                        "--history-root",
                        str(history),
                        "--train-receipt",
                        str(train_path),
                        "--validation-receipt",
                        str(validation_path),
                        "--workers",
                        "1",
                        "--output",
                        str(holdout_path),
                    ]
                )
            self.assertEqual(status, 1)
            loader.assert_not_called()
            runner.assert_not_called()
            self.assertFalse(holdout_path.exists())

    def test_holdout_rejects_resealed_train_global_with_rebound_validation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            history = self._history(parent)
            train_path = parent / "train.json"
            validation_path = parent / "validation.json"
            holdout_path = parent / "holdout.json"
            with (
                mock.patch.object(
                    self.cli.story_core,
                    "run_adaptive_story_s5_grid",
                    side_effect=self._fake_run,
                ),
                mock.patch.object(
                    self.cli.story_core,
                    "combine_adaptive_story_s5_grid_runs",
                    side_effect=self._fake_combine,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(self.cli.main(_train_argv(history, train_path)), 0)
                self.assertEqual(
                    self.cli.main(
                        [
                            "validation",
                            "--history-root",
                            str(history),
                            "--train-receipt",
                            str(train_path),
                            "--workers",
                            "1",
                            "--output",
                            str(validation_path),
                        ]
                    ),
                    0,
                )

            train = json.loads(train_path.read_text(encoding="utf-8"))
            train_metric = train["global_result"]["candidate_metrics"][0]["by_split"][
                "TRAIN"
            ]
            train_metric["exact_net_r"] += 0.25
            global_body = dict(train["global_result"])
            global_body.pop("result_sha256")
            train["global_result"]["result_sha256"] = self.cli._canonical_sha(
                global_body
            )
            train_body = dict(train)
            train_body.pop("receipt_sha256")
            train["receipt_sha256"] = self.cli._canonical_sha(train_body)
            train_path.write_text(json.dumps(train), encoding="utf-8")

            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            validation["train_receipt_sha256"] = train["receipt_sha256"]
            validation_body = dict(validation)
            validation_body.pop("receipt_sha256")
            validation["receipt_sha256"] = self.cli._canonical_sha(validation_body)
            validation_path.write_text(json.dumps(validation), encoding="utf-8")

            with (
                mock.patch.object(
                    self.cli.story_core,
                    "combine_adaptive_story_s5_grid_runs",
                    side_effect=self._fake_combine,
                ),
                mock.patch.object(self.cli, "load_historical_s5_slice") as loader,
                mock.patch.object(
                    self.cli.story_core,
                    "run_adaptive_story_s5_grid",
                ) as runner,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                status = self.cli.main(
                    [
                        "holdout",
                        "--history-root",
                        str(history),
                        "--train-receipt",
                        str(train_path),
                        "--validation-receipt",
                        str(validation_path),
                        "--workers",
                        "1",
                        "--output",
                        str(holdout_path),
                    ]
                )
            self.assertEqual(status, 1)
            loader.assert_not_called()
            runner.assert_not_called()
            self.assertFalse(holdout_path.exists())

    def test_holdout_rejects_train_validation_path_alias_before_core(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            history = self._history(parent)
            train_path = parent / "train.json"
            validation_path = parent / "validation.json"
            with (
                mock.patch.object(
                    self.cli.story_core,
                    "run_adaptive_story_s5_grid",
                    side_effect=self._fake_run,
                ),
                mock.patch.object(
                    self.cli.story_core,
                    "combine_adaptive_story_s5_grid_runs",
                    side_effect=self._fake_combine,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(self.cli.main(_train_argv(history, train_path)), 0)
                self.assertEqual(
                    self.cli.main(
                        [
                            "validation",
                            "--history-root",
                            str(history),
                            "--train-receipt",
                            str(train_path),
                            "--workers",
                            "1",
                            "--output",
                            str(validation_path),
                        ]
                    ),
                    0,
                )

            hardlink_path = parent / "validation-hardlink.json"
            hardlink_path.hardlink_to(validation_path)
            aliases = {
                "same_path": validation_path,
                "hardlink": hardlink_path,
            }
            for label, alias in aliases.items():
                with self.subTest(label=label):
                    holdout_path = parent / f"holdout-{label}.json"
                    with (
                        mock.patch.object(
                            self.cli,
                            "_load_receipt",
                        ) as receipt_loader,
                        mock.patch.object(
                            self.cli.story_core,
                            "run_adaptive_story_s5_grid",
                        ) as runner,
                        contextlib.redirect_stderr(io.StringIO()),
                    ):
                        status = self.cli.main(
                            [
                                "holdout",
                                "--history-root",
                                str(history),
                                "--train-receipt",
                                str(alias),
                                "--validation-receipt",
                                str(validation_path),
                                "--workers",
                                "1",
                                "--output",
                                str(holdout_path),
                            ]
                        )
                    self.assertEqual(status, 1)
                    receipt_loader.assert_not_called()
                    runner.assert_not_called()
                    self.assertFalse(holdout_path.exists())

    def test_validation_fails_closed_when_global_combine_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            history = self._history(parent)
            train_path = parent / "train.json"
            output = parent / "validation.json"
            with (
                mock.patch.object(
                    self.cli.story_core,
                    "run_adaptive_story_s5_grid",
                    side_effect=self._fake_run,
                ),
                mock.patch.object(
                    self.cli.story_core,
                    "combine_adaptive_story_s5_grid_runs",
                    side_effect=self._fake_combine,
                    create=True,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(self.cli.main(_train_argv(history, train_path)), 0)
            with (
                mock.patch.object(
                    self.cli.story_core,
                    "run_adaptive_story_s5_grid",
                    side_effect=self._fake_run,
                ),
                mock.patch.object(
                    self.cli.story_core,
                    "combine_adaptive_story_s5_grid_runs",
                    None,
                    create=True,
                ),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                status = self.cli.main(
                    [
                        "validation",
                        "--history-root",
                        str(history),
                        "--train-receipt",
                        str(train_path),
                        "--workers",
                        "1",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(status, 1)
            self.assertFalse(output.exists())

    def test_dependency_drift_fails_before_any_output_publication(self) -> None:
        for label, seal_sequence in (
            ("before_return", 2),
            ("before_publish", 3),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                parent = Path(tmp)
                history = self._history(parent)
                output = parent / f"{label}.json"
                initial = self.cli._evaluator_dependency_seal()
                drifted = self._drifted_evaluator_seal(initial)
                side_effect = (
                    [initial, drifted]
                    if seal_sequence == 2
                    else [initial, initial, drifted]
                )
                with (
                    mock.patch.object(
                        self.cli,
                        "_evaluator_dependency_seal",
                        side_effect=side_effect,
                    ),
                    mock.patch.object(
                        self.cli.story_core,
                        "run_adaptive_story_s5_grid",
                        side_effect=self._fake_run,
                    ),
                    mock.patch.object(
                        self.cli.story_core,
                        "combine_adaptive_story_s5_grid_runs",
                        side_effect=self._fake_combine,
                    ),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    status = self.cli.main(_train_argv(history, output))
                self.assertEqual(status, 1)
                self.assertFalse(output.exists())

    def test_downstream_dependency_mismatch_fails_before_data_or_core(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            history = self._history(parent)
            train_path = parent / "train.json"
            output = parent / "validation.json"
            with (
                mock.patch.object(
                    self.cli.story_core,
                    "run_adaptive_story_s5_grid",
                    side_effect=self._fake_run,
                ),
                mock.patch.object(
                    self.cli.story_core,
                    "combine_adaptive_story_s5_grid_runs",
                    side_effect=self._fake_combine,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(self.cli.main(_train_argv(history, train_path)), 0)
            train = json.loads(train_path.read_text(encoding="utf-8"))
            train["evaluator_dependency_seal"] = self._drifted_evaluator_seal(
                train["evaluator_dependency_seal"]
            )
            body = dict(train)
            body.pop("receipt_sha256")
            train["receipt_sha256"] = self.cli._canonical_sha(body)
            train_path.write_text(json.dumps(train), encoding="utf-8")
            with (
                mock.patch.object(
                    self.cli,
                    "load_historical_s5_slice",
                ) as loader,
                mock.patch.object(
                    self.cli.story_core,
                    "run_adaptive_story_s5_grid",
                ) as runner,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                status = self.cli.main(
                    [
                        "validation",
                        "--history-root",
                        str(history),
                        "--train-receipt",
                        str(train_path),
                        "--workers",
                        "1",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(status, 1)
            loader.assert_not_called()
            runner.assert_not_called()
            self.assertFalse(output.exists())

    def test_frozen_manifest_ignores_later_unrelated_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            history = self._history(parent)
            train_path = parent / "train.json"
            output = parent / "validation.json"
            with (
                mock.patch.object(
                    self.cli.story_core,
                    "run_adaptive_story_s5_grid",
                    side_effect=self._fake_run,
                ),
                mock.patch.object(
                    self.cli.story_core,
                    "combine_adaptive_story_s5_grid_runs",
                    side_effect=self._fake_combine,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(self.cli.main(_train_argv(history, train_path)), 0)
            start = datetime(2026, 5, 1, tzinfo=UTC)
            _write_pair_cache(
                history,
                pair="USD_CAD",
                run_id="20260501T010200Z",
                declared_from=start,
                declared_to=start + timedelta(hours=1),
            )
            with (
                mock.patch.object(
                    self.cli,
                    "build_historical_s5_manifest",
                    side_effect=AssertionError("downstream must not rescan"),
                ) as builder,
                mock.patch.object(
                    self.cli.story_core,
                    "run_adaptive_story_s5_grid",
                    side_effect=self._fake_run,
                ),
                mock.patch.object(
                    self.cli.story_core,
                    "combine_adaptive_story_s5_grid_runs",
                    side_effect=self._fake_combine,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                status = self.cli.main(
                    [
                        "validation",
                        "--history-root",
                        str(history),
                        "--train-receipt",
                        str(train_path),
                        "--workers",
                        "1",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(status, 0)
            builder.assert_not_called()
            validation = json.loads(output.read_text(encoding="utf-8"))
            train = json.loads(train_path.read_text(encoding="utf-8"))
            self.assertEqual(
                validation["frozen_manifest"],
                train["frozen_manifest"],
            )

    def test_resealed_manifest_source_outside_allowed_run_scope_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            history = self._history(parent)
            train_path = parent / "train.json"
            output = parent / "validation.json"
            with (
                mock.patch.object(
                    self.cli.story_core,
                    "run_adaptive_story_s5_grid",
                    side_effect=self._fake_run,
                ),
                mock.patch.object(
                    self.cli.story_core,
                    "combine_adaptive_story_s5_grid_runs",
                    side_effect=self._fake_combine,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(self.cli.main(_train_argv(history, train_path)), 0)

            train = json.loads(train_path.read_text(encoding="utf-8"))
            manifest = train["frozen_manifest"]
            source = manifest["selected_sources"][0]
            source_path = Path(source["relative_path"])
            source["relative_path"] = str(
                Path("20260501T010200Z", *source_path.parts[1:])
            )
            source_body = dict(source)
            source_body.pop("source_sha256")
            source["source_sha256"] = self.cli._canonical_sha(source_body)
            manifest_body = dict(manifest)
            manifest_body.pop("manifest_sha256")
            manifest["manifest_sha256"] = self.cli._canonical_sha(manifest_body)
            train["manifest_receipt"]["manifest_sha256"] = manifest["manifest_sha256"]
            receipt_body = dict(train)
            receipt_body.pop("receipt_sha256")
            train["receipt_sha256"] = self.cli._canonical_sha(receipt_body)
            train_path.write_text(json.dumps(train), encoding="utf-8")

            with (
                mock.patch.object(self.cli, "load_historical_s5_slice") as loader,
                mock.patch.object(
                    self.cli.story_core,
                    "run_adaptive_story_s5_grid",
                ) as runner,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                status = self.cli.main(
                    [
                        "validation",
                        "--history-root",
                        str(history),
                        "--train-receipt",
                        str(train_path),
                        "--workers",
                        "1",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(status, 1)
            loader.assert_not_called()
            runner.assert_not_called()
            self.assertFalse(output.exists())

    def test_frozen_selected_file_tamper_fails_before_core(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            history = self._history(parent)
            train_path = parent / "train.json"
            output = parent / "validation.json"
            with (
                mock.patch.object(
                    self.cli.story_core,
                    "run_adaptive_story_s5_grid",
                    side_effect=self._fake_run,
                ),
                mock.patch.object(
                    self.cli.story_core,
                    "combine_adaptive_story_s5_grid_runs",
                    side_effect=self._fake_combine,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(self.cli.main(_train_argv(history, train_path)), 0)
            train = json.loads(train_path.read_text(encoding="utf-8"))
            selected = train["frozen_manifest"]["selected_sources"][0]
            with (history / selected["relative_path"]).open("ab") as handle:
                handle.write(b"tamper")
            with (
                mock.patch.object(
                    self.cli.story_core,
                    "run_adaptive_story_s5_grid",
                ) as runner,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                status = self.cli.main(
                    [
                        "validation",
                        "--history-root",
                        str(history),
                        "--train-receipt",
                        str(train_path),
                        "--workers",
                        "1",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(status, 1)
            runner.assert_not_called()
            self.assertFalse(output.exists())

    def test_downstream_history_root_must_match_frozen_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            history = self._history(parent)
            other_history = parent / "other-history"
            other_history.mkdir()
            train_path = parent / "train.json"
            output = parent / "validation.json"
            with (
                mock.patch.object(
                    self.cli.story_core,
                    "run_adaptive_story_s5_grid",
                    side_effect=self._fake_run,
                ),
                mock.patch.object(
                    self.cli.story_core,
                    "combine_adaptive_story_s5_grid_runs",
                    side_effect=self._fake_combine,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(self.cli.main(_train_argv(history, train_path)), 0)
            with (
                mock.patch.object(
                    self.cli,
                    "load_historical_s5_slice",
                ) as loader,
                mock.patch.object(
                    self.cli.story_core,
                    "run_adaptive_story_s5_grid",
                ) as runner,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                status = self.cli.main(
                    [
                        "validation",
                        "--history-root",
                        str(other_history),
                        "--train-receipt",
                        str(train_path),
                        "--workers",
                        "1",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(status, 1)
            loader.assert_not_called()
            runner.assert_not_called()
            self.assertFalse(output.exists())

    def test_train_requires_complete_receipted_pair_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            history = parent / "history"
            history.mkdir()
            start = datetime(2026, 5, 1, tzinfo=UTC)
            end = start + timedelta(hours=1)
            _write_pair_cache(
                history,
                pair="EUR_USD",
                run_id="20260501T010000Z",
                declared_from=start,
                declared_to=end,
            )
            _write_pair_cache(
                history,
                pair="USD_JPY",
                run_id="20260501T010100Z",
                declared_from=start,
                declared_to=end,
            )
            output = parent / "missing.json"
            with (
                mock.patch.object(
                    self.cli.story_core,
                    "run_adaptive_story_s5_grid",
                ) as runner,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                status = self.cli.main(_train_argv(history, output))
            self.assertEqual(status, 1)
            runner.assert_not_called()
            self.assertFalse(output.exists())

        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            history = self._history(parent)
            (history / "truth_acquisition_receipts.jsonl").unlink()
            output = parent / "unreceipted.json"
            with (
                mock.patch.object(
                    self.cli.story_core,
                    "run_adaptive_story_s5_grid",
                ) as runner,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                status = self.cli.main(_train_argv(history, output))
            self.assertEqual(status, 1)
            runner.assert_not_called()
            self.assertFalse(output.exists())

    def test_core_pair_and_global_seals_and_policies_are_verified(self) -> None:
        split = self.cli.story_core.UtcSplit(
            name="TRAIN",
            from_utc=datetime(2026, 5, 1, tzinfo=UTC),
            to_utc=datetime(2026, 5, 2, tzinfo=UTC),
        )
        candidate_ids = ("H21:TIME_1H",)
        valid_pair = self._fake_run(
            "EUR_USD",
            (),
            (split,),
            candidate_ids=candidate_ids,
        )
        unsealed = dict(valid_pair)
        unsealed.pop("result_sha256")
        with self.assertRaises(self.cli.AdaptiveStoryCliError):
            self.cli._validate_pair_result_contract(
                unsealed,
                pair="EUR_USD",
                split=split,
                candidate_ids=candidate_ids,
                expected_statuses=("COMPLETE",),
            )
        wrong_pair_policy = dict(valid_pair)
        wrong_pair_policy["truth_policy"] = "WRONG_TRUTH_POLICY"
        body = dict(wrong_pair_policy)
        body.pop("result_sha256")
        wrong_pair_policy["result_sha256"] = self.cli._canonical_sha(body)
        with self.assertRaises(self.cli.AdaptiveStoryCliError):
            self.cli._validate_pair_result_contract(
                wrong_pair_policy,
                pair="EUR_USD",
                split=split,
                candidate_ids=candidate_ids,
                expected_statuses=("COMPLETE",),
            )

        valid_global = self._fake_combine(
            (valid_pair,),
            (split,),
            candidate_ids=candidate_ids,
        )
        wrong_global_scope = dict(valid_global)
        wrong_global_scope["pairs"] = ["GBP_USD"]
        body = dict(wrong_global_scope)
        body.pop("result_sha256")
        wrong_global_scope["result_sha256"] = self.cli._canonical_sha(body)
        with self.assertRaises(self.cli.AdaptiveStoryCliError):
            self.cli._validate_global_result_contract(
                wrong_global_scope,
                pair_results=(valid_pair,),
                split=split,
                candidate_ids=candidate_ids,
            )

    def test_stable_read_rejects_a_b_a_path_substitution_around_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            target = parent / "receipt.json"
            replacement = parent / "replacement.json"
            original_backup = parent / "original.json"
            target.write_bytes(b'{"identity":"A"}')
            replacement.write_bytes(b'{"identity":"B"}')
            real_open = self.cli.os.open
            swapped = False

            def substitute_around_open(path, flags, *args, **kwargs):
                nonlocal swapped
                if Path(path) == target and not swapped:
                    swapped = True
                    target.rename(original_backup)
                    replacement.rename(target)
                    descriptor = real_open(path, flags, *args, **kwargs)
                    target.rename(replacement)
                    original_backup.rename(target)
                    return descriptor
                return real_open(path, flags, *args, **kwargs)

            with (
                mock.patch.object(
                    self.cli.os,
                    "open",
                    side_effect=substitute_around_open,
                ),
                self.assertRaises(self.cli.AdaptiveStoryCliError),
            ):
                self.cli._read_stable_bytes(target)
            self.assertTrue(swapped)
            self.assertEqual(target.read_bytes(), b'{"identity":"A"}')
            self.assertEqual(replacement.read_bytes(), b'{"identity":"B"}')

    def test_serial_and_parallel_receipts_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            history = self._history(parent)
            serial_output = parent / "serial.json"
            parallel_output = parent / "parallel.json"
            serial_argv = _train_argv(history, serial_output)
            parallel_argv = self._set_workers(
                _train_argv(history, parallel_output),
                2,
            )
            serial = subprocess.run(
                [sys.executable, str(SCRIPT), *serial_argv],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(serial.returncode, 0, serial.stderr)
            parallel = subprocess.run(
                [sys.executable, str(SCRIPT), *parallel_argv],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(parallel.returncode, 0, parallel.stderr)
            self.assertEqual(serial_output.read_bytes(), parallel_output.read_bytes())


if __name__ == "__main__":
    unittest.main()
