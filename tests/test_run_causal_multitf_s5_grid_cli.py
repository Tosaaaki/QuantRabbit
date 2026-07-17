from __future__ import annotations

import contextlib
import gzip
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run-causal-multitf-s5-grid.py"
UTC = timezone.utc


def _load_script():
    module_name = "test_run_causal_multitf_s5_grid_script"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("causal grid CLI could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
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
            declared_from + timedelta(minutes=35),
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


def _argv(history_root: Path, output: Path, *, pairs: str) -> list[str]:
    return [
        "--history-root",
        str(history_root),
        "--pairs",
        pairs,
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
        "--output",
        str(output),
    ]


def _pair_result(pair: str, *_args, **_kwargs) -> dict:
    if pair == "NZD_CHF":
        return {"pair": pair, "status": "UNAVAILABLE"}
    return {
        "pair": pair,
        "validation": {"selected_arm_ids": ["H01:DIRECT:BASE"]},
        "holdout": {
            "evaluated_arm_ids": ["H01:DIRECT:BASE"],
            "reselection_performed": False,
        },
    }


def _global_result(_pair_results, _splits, **_kwargs) -> dict:
    return {
        "selected_arm_ids": ["H01:DIRECT:BASE"],
        "validation_winner_arm_ids": ["H01:DIRECT:BASE"],
        "holdout_evaluated_arm_ids": ["H01:DIRECT:BASE"],
        "holdout_selection_unchanged": True,
        "selection_receipt_sha256": "a" * 64,
    }


class CausalMultiTfS5GridCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cli = _load_script()

    def test_each_available_pair_is_loaded_once_and_missing_pair_stays_explicit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "history"
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
            output = Path(tmp) / "report.json"
            stdout = io.StringIO()
            with (
                mock.patch.object(
                    self.cli,
                    "load_historical_s5_slice",
                    wraps=self.cli.load_historical_s5_slice,
                ) as loader,
                mock.patch.object(
                    self.cli.grid_core,
                    "run_causal_multitf_s5_grid",
                    side_effect=_pair_result,
                ) as runner,
                mock.patch.object(
                    self.cli.grid_core,
                    "combine_causal_multitf_s5_grid_runs",
                    side_effect=_global_result,
                ),
                contextlib.redirect_stdout(stdout),
            ):
                status = self.cli.main(
                    _argv(
                        root,
                        output,
                        pairs="EUR_USD,GBP_JPY,NZD_CHF",
                    )
                )

            self.assertEqual(status, 0)
            self.assertEqual(loader.call_count, 2)
            self.assertEqual(runner.call_count, 3)
            self.assertTrue(
                all(
                    call.kwargs["unavailable_pairs"] == ("NZD_CHF",)
                    for call in runner.call_args_list
                )
            )
            self.assertEqual(
                [call.kwargs["pair"] for call in loader.call_args_list],
                ["EUR_USD", "GBP_JPY"],
            )
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["unavailable_pairs"], ["NZD_CHF"])
            self.assertEqual(
                [item["pair"] for item in report["pair_runs"]],
                ["EUR_USD", "GBP_JPY", "NZD_CHF"],
            )
            for item in report["pair_runs"][:2]:
                self.assertRegex(
                    item["source_receipt"]["source_sha256"], r"^[0-9a-f]{64}$"
                )
                self.assertRegex(
                    item["slice_receipt"]["slice_sha256"], r"^[0-9a-f]{64}$"
                )
                self.assertEqual(
                    item["slice_receipt"]["source_manifest_sha256"],
                    report["manifest_receipt"]["manifest_sha256"],
                )
            self.assertEqual(report["pair_runs"][2]["source_status"], "UNAVAILABLE")
            self.assertIsNone(report["pair_runs"][2]["source_receipt"])
            self.assertIsNone(report["pair_runs"][2]["slice_receipt"])
            self.assertEqual(
                report["pair_runs"][2]["result"]["holdout"]["evaluated_arm_ids"],
                [],
            )
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["evaluated_pair_count"], 2)
            self.assertEqual(summary["requested_pair_run_count"], 3)
            self.assertEqual(summary["unavailable_pairs"], ["NZD_CHF"])
            self.assertNotIn("pair_runs", summary)

    def test_pair_report_redaction_removes_nonwinner_holdout_outcomes(self) -> None:
        result = {
            "pair": "EUR_USD",
            "status": "OK",
            "candidate_metrics": [
                {
                    "candidate_id": "H01:DIRECT:BASE",
                    "metrics_by_split": {
                        "VALIDATION": {"reason_counts": {"VISIBLE": 1}},
                        "HOLDOUT": {"reason_counts": {"WINNER_HOLDOUT": 1}},
                    },
                },
                {
                    "candidate_id": "H02:DIRECT:BASE",
                    "metrics_by_split": {
                        "VALIDATION": {"reason_counts": {"VISIBLE": 2}},
                        "HOLDOUT": {"reason_counts": {"SECRET_STOP": 99}},
                    },
                },
            ],
            "daily_aggregates": [
                {
                    "candidate_id": "H02:DIRECT:BASE",
                    "split": "HOLDOUT",
                    "reason_counts": {"SECRET_STOP": 99},
                }
            ],
            "trade_rows": [{"candidate_id": "H02:DIRECT:BASE", "split": "HOLDOUT"}],
            "signal_rows": [{"direct_side": "SHORT"}],
            "reason_counts": {"SECRET_STOP": 99},
            "aggregation": {"trade_row_count": 99, "source_candle_count": 10},
        }

        visible = self.cli._redact_pair_holdout_for_report(
            result,
            global_winners=["H01:DIRECT:BASE"],
            selection_receipt_sha256="a" * 64,
        )

        nonwinner = visible["candidate_metrics"][1]
        self.assertNotIn("HOLDOUT", nonwinner["metrics_by_split"])
        self.assertEqual(visible["daily_aggregates"], [])
        self.assertEqual(visible["trade_rows"], [])
        self.assertEqual(visible["signal_rows"], [])
        self.assertNotIn("SECRET_STOP", visible["reason_counts"])
        self.assertNotIn("trade_row_count", visible["aggregation"])

    def test_report_keeps_validation_winner_sealed_for_holdout_and_has_no_authority(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "history"
            root.mkdir()
            start = datetime(2026, 5, 1, tzinfo=UTC)
            _write_pair_cache(
                root,
                pair="EUR_USD",
                run_id="20260501T010000Z",
                declared_from=start,
                declared_to=start + timedelta(hours=1),
            )
            output = Path(tmp) / "report.json"
            with (
                mock.patch.object(
                    self.cli.grid_core,
                    "run_causal_multitf_s5_grid",
                    side_effect=_pair_result,
                ),
                mock.patch.object(
                    self.cli.grid_core,
                    "combine_causal_multitf_s5_grid_runs",
                    side_effect=_global_result,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    self.cli.main(_argv(root, output, pairs="EUR_USD")),
                    0,
                )

            report = json.loads(output.read_text(encoding="utf-8"))
            global_result = report["global_result"]
            self.assertEqual(
                global_result["validation_winner_arm_ids"],
                global_result["holdout_evaluated_arm_ids"],
            )
            self.assertTrue(global_result["holdout_selection_unchanged"])
            self.assertEqual(
                report["selection_contract"]["selection_source"],
                "VALIDATION_ONLY",
            )
            self.assertTrue(
                report["selection_contract"]["holdout_uses_sealed_validation_selection"]
            )
            self.assertFalse(
                report["selection_contract"]["holdout_reselection_allowed"]
            )
            self.assertEqual(report["order_authority"], "NONE")
            self.assertTrue(report["historical_only"])
            self.assertTrue(report["diagnostic_only"])
            self.assertTrue(report["shadow_only"])
            self.assertFalse(report["live_permission"])
            self.assertFalse(report["live_order_enabled"])
            self.assertFalse(report["promotion_allowed"])
            self.assertFalse(report["broker_mutation_allowed"])
            scope = dict(report["research_scope_receipt"])
            scope_digest = scope.pop("scope_sha256")
            self.assertEqual(scope_digest, self.cli._canonical_sha(scope))
            self.assertFalse(scope["implicit_default_pair_universe_used"])
            body = dict(report)
            digest = body.pop("report_sha256")
            self.assertEqual(digest, self.cli._canonical_sha(body))

    def test_tiny_cache_runs_through_the_real_core_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "history"
            root.mkdir()
            start = datetime(2026, 5, 1, tzinfo=UTC)
            _write_pair_cache(
                root,
                pair="EUR_USD",
                run_id="20260501T010000Z",
                declared_from=start,
                declared_to=start + timedelta(hours=1),
            )
            output = Path(tmp) / "report.json"
            with contextlib.redirect_stdout(io.StringIO()):
                status = self.cli.main(_argv(root, output, pairs="EUR_USD,NZD_CHF"))

            self.assertEqual(status, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["global_result"]["candidate_count"], 182)
            self.assertEqual(
                report["global_result"]["selected_arm_ids"],
                report["global_result"]["validation_winner_arm_ids"],
            )
            self.assertEqual(
                report["global_result"]["validation_winner_arm_ids"],
                report["global_result"]["holdout_evaluated_arm_ids"],
            )
            self.assertTrue(report["global_result"]["holdout_selection_unchanged"])
            self.assertEqual(report["global_result"]["unavailable_pairs"], ["NZD_CHF"])
            self.assertEqual(report["order_authority"], "NONE")
            self.assertFalse(report["live_permission"])

    def test_overlapping_split_boundaries_fail_before_loading_or_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "history"
            root.mkdir()
            output = Path(tmp) / "report.json"
            argv = _argv(root, output, pairs="EUR_USD")
            argv[argv.index("--validation-from") + 1] = "2026-05-01T00:09:00Z"
            with (
                mock.patch.object(
                    self.cli,
                    "build_historical_s5_manifest",
                ) as manifest_builder,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                status = self.cli.main(argv)

            self.assertEqual(status, 1)
            manifest_builder.assert_not_called()
            self.assertFalse(output.exists())

    def test_pairs_are_required_and_have_no_default_universe(self) -> None:
        argv = _argv(Path("history"), Path("report.json"), pairs="EUR_USD")
        pair_index = argv.index("--pairs")
        del argv[pair_index : pair_index + 2]
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            self.cli._parse_args(argv)
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
