from __future__ import annotations

import gzip
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import quant_rabbit.dojo_worker_execution as worker_execution

from quant_rabbit.dojo_worker_execution import evaluate_derived_run, verify_derived_run
from quant_rabbit.dojo_worker_forward import (
    build_day_seal,
    build_precommit,
    build_start_receipt,
    canonical_sha256,
    write_new_json,
)


REPO = Path(__file__).resolve().parents[1]


def utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_operational_worker_result_is_derived_from_virtual_ledgers(
    tmp_path: Path, monkeypatch
) -> None:
    candidate_config = {
        "signal": "burst",
        "tp_pips": 3,
        "sl_pips": 12,
        "ceiling_min": 240,
        "max_concurrent": 1,
        "per_pos_lev": 1.0,
        "atr_floor_pips": 0.1,
        "pull_atr": 0.6,
        "fade_atr": 1.2,
        "eff_max": 0.2,
    }
    dependency_paths = [
        "src/quant_rabbit/dojo_lab_provenance.py",
        "src/quant_rabbit/virtual_broker.py",
        "src/quant_rabbit/dojo_worker_source.py",
        "src/quant_rabbit/dojo_market_calendar.py",
        "src/quant_rabbit/broker/oanda.py",
        "src/quant_rabbit/analysis/market_status.py",
        "src/quant_rabbit/instruments.py",
        "src/quant_rabbit/models.py",
        "src/quant_rabbit/operator_manual.py",
        "src/quant_rabbit/paths.py",
        "src/quant_rabbit/dojo_worker_execution.py",
        "scripts/collect-dojo-worker-day.py",
        "scripts/oanda_history_fetch.py",
        "scripts/run-dojo-worker-forward.py",
    ]
    candidates = []
    for index in range(12):
        config = {**candidate_config, "tp_pips": 3 + index}
        candidates.append(
            {
                "candidate_id": f"burst-test-{index + 1:02d}",
                "family_id": f"burst-family-{index % 3 + 1}",
                "config": config,
                "config_sha256": canonical_sha256(config),
            }
        )
    python_executable = Path(sys.executable).resolve()
    spec = {
        "experiment_id": "dojo-worker-derived-test",
        "window": {
            "start_utc": "2026-07-20T00:00:00Z",
            "end_utc": "2026-08-03T00:00:00Z",
        },
        "candidate_set": {
            "declared_grid_size": 12,
            "family_denominator": 3,
            "candidates": candidates,
        },
        "mechanics": {
            "pairs": ["USD_JPY"],
            "granularity": "M1",
            "bot_bar": "feed",
            "intrabar_paths": ["OHLC", "OLHC"],
            "initial_balance_jpy": 200000.0,
            "slippage_pips_per_fill": 0.3,
            "financing_pips_per_day": 0.8,
            "leverage": 25.0,
            "period_end_settlement": "CANCEL_OWNED_ORDERS_THEN_CLOSE_OWNED_POSITIONS",
            "terminal_score_basis": "FULLY_RESOLVED_BALANCE",
        },
        "source_bindings": {
            "git_commit": "a" * 40,
            "runner_sha256": file_sha(REPO / "scripts/run-virtual-market-session.py"),
            "bot_module_sha256": file_sha(REPO / "bots/lab_bot.py"),
            "bot_dependency_sha256": {
                path: file_sha(REPO / path) for path in dependency_paths
            },
            "scorer_sha256": file_sha(
                REPO / "src/quant_rabbit/dojo_lab_provenance.py"
            ),
            "precommit_builder_sha256": file_sha(
                REPO / "src/quant_rabbit/dojo_worker_forward.py"
            ),
            "python_executable_path": str(python_executable),
            "python_executable_sha256": file_sha(python_executable),
            "python_version": sys.version,
        },
        "thresholds": {
            "minimum_calendar_30d_multiple": 3.0,
            "maximum_margin_closeouts": 0,
            "zero_trade_policy": "FAIL_CLOSED",
        },
        "daily_source_policy": {
            "expected_source_ids": ["USD_JPY:M1"],
            "source_origin": "test fixed corpus",
            "seal_grace_hours": 12,
            "minimum_open_market_days": 10,
            "late_backfill_allowed": False,
            "market_closed_days_require_explicit_receipt": True,
        },
    }
    precommit = build_precommit(spec, now_utc=utc("2026-07-19T00:00:00Z"))
    start = build_start_receipt(precommit, now_utc=utc("2026-07-19T00:01:00Z"))
    run_dir = (tmp_path / "run").resolve()
    run_dir.mkdir()
    write_new_json(run_dir / "precommit.json", precommit)
    write_new_json(run_dir / "start.json", start)
    previous = None
    window_start = utc(precommit["window"]["start_utc"])
    for ordinal in range(1, 15):
        day_start = window_start + timedelta(days=ordinal - 1)
        manifest = {
            "market_closed": False,
            "closure_reason": None,
            "sources": [
                {
                    "source_id": "USD_JPY:M1",
                    "pair": "USD_JPY",
                    "granularity": "M1",
                    "content_sha256": hashlib.sha256(
                        f"day-{ordinal}".encode()
                    ).hexdigest(),
                    "size_bytes": 1,
                    "row_count": 1,
                    "first_event_utc": day_start.isoformat().replace("+00:00", "Z"),
                    "last_event_utc": day_start.isoformat().replace("+00:00", "Z"),
                }
            ],
        }
        previous = build_day_seal(
            precommit,
            start,
            previous,
            manifest,
            ordinal=ordinal,
            now_utc=day_start + timedelta(days=1, hours=1),
        )
        write_new_json(run_dir / "days" / f"day-{ordinal:03d}.json", previous)

    shard = run_dir / "corpus/day-001/USD_JPY/USD_JPY_M1_BA_2026-test.jsonl.gz"
    shard.parent.mkdir(parents=True)
    rows = []
    for index in range(1_600):
        stamp = window_start + timedelta(minutes=index)
        base = 150 + index * 0.0001
        rows.append(
            {
                "time": stamp.isoformat().replace("+00:00", "Z"),
                "bid": {
                    "o": f"{base:.5f}",
                    "h": f"{base + 0.002:.5f}",
                    "l": f"{base - 0.001:.5f}",
                    "c": f"{base + 0.001:.5f}",
                },
                "ask": {
                    "o": f"{base + 0.002:.5f}",
                    "h": f"{base + 0.004:.5f}",
                    "l": f"{base + 0.001:.5f}",
                    "c": f"{base + 0.003:.5f}",
                },
            }
        )
    with gzip.open(shard, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    source_rows = [
        {
            "ordinal": ordinal,
            "day_seal_sha256": json.loads(
                (run_dir / "days" / f"day-{ordinal:03d}.json").read_text()
            )["day_seal_sha256"],
            "acquisition_receipt_sha256": hashlib.sha256(
                f"receipt-{ordinal}".encode()
            ).hexdigest(),
            "source_manifest_sha256": hashlib.sha256(
                f"manifest-{ordinal}".encode()
            ).hexdigest(),
            "response_sha256": hashlib.sha256(
                f"response-{ordinal}".encode()
            ).hexdigest(),
            "source_content_sha256": (
                file_sha(shard)
                if ordinal == 1
                else hashlib.sha256(f"source-{ordinal}".encode()).hexdigest()
            ),
            "source_relpath": (
                str(shard.relative_to(run_dir)) if ordinal == 1 else None
            ),
            "source_size_bytes": shard.stat().st_size if ordinal == 1 else None,
            "market_closed": ordinal != 1,
        }
        for ordinal in range(1, 15)
    ]
    monkeypatch.setattr(worker_execution, "_verify_pinned_sources", lambda *args: None)
    monkeypatch.setattr(
        worker_execution,
        "_verify_all_source_bundles",
        lambda *args: source_rows,
    )
    result = evaluate_derived_run(
        run_dir,
        repo_root=REPO,
        now_utc=utc("2026-08-03T00:00:00Z"),
    )
    assert result["cell_count"] == 24
    assert result["promotion_eligible"] is False
    assert result["smoke_gate_passed"] is False
    manifest = json.loads((run_dir / "result-manifest.json").read_text())
    assert len(manifest["results"]) == 24
    assert {row["intrabar"] for row in manifest["results"]} == {"OHLC", "OLHC"}
    assert all(row["promotion_eligible"] is False for row in manifest["results"])
    assert {row["status"] for row in manifest["results"]} == {
        "INVALID_UNSCOREABLE_TRIAL"
    }

    first_terminal = next((run_dir / "execution/cells").glob("*/OHLC/terminal.json"))
    first_terminal.unlink()
    (run_dir / "result-manifest.json").unlink()
    (run_dir / "final.json").unlink()
    monkeypatch.setattr(
        worker_execution.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("a second selectable cell run was launched")
        ),
    )
    recovered = evaluate_derived_run(
        run_dir,
        repo_root=REPO,
        now_utc=utc("2026-08-03T00:01:00Z"),
    )
    assert recovered["cell_count"] == 24
    assert verify_derived_run(run_dir, repo_root=REPO)["cell_count"] == 24

    extra = run_dir / "corpus/day-999/USD_JPY/USD_JPY_M1_BA_2026-extra.jsonl.gz"
    extra.parent.mkdir(parents=True)
    extra.write_bytes(shard.read_bytes())
    try:
        verify_derived_run(run_dir, repo_root=REPO)
    except worker_execution.DojoWorkerExecutionError as exc:
        assert "file set" in str(exc)
    else:  # pragma: no cover - the test must fail closed.
        raise AssertionError("an extra unsealed corpus shard was accepted")
    extra.unlink()

    attempt_path = first_terminal.parent / "attempt.json"
    attempt_bytes = attempt_path.read_bytes()
    attempt = json.loads(attempt_bytes)
    attempt["state"] = "REWRITTEN"
    attempt_path.write_text(json.dumps(attempt))
    try:
        verify_derived_run(run_dir, repo_root=REPO)
    except worker_execution.DojoWorkerExecutionError as exc:
        assert "attempt" in str(exc)
    else:  # pragma: no cover - the test must fail closed.
        raise AssertionError("a rewritten attempt was accepted")
    attempt_path.write_bytes(attempt_bytes)

    manifest = json.loads((run_dir / "result-manifest.json").read_text())
    manifest["results"][0]["terminal_net_jpy"] = 999_999_999.0
    (run_dir / "result-manifest.json").write_text(json.dumps(manifest))
    try:
        verify_derived_run(run_dir, repo_root=REPO)
    except worker_execution.DojoWorkerExecutionError as exc:
        assert "not derived" in str(exc)
    else:  # pragma: no cover - the test must fail closed.
        raise AssertionError("fabricated result manifest was accepted")
