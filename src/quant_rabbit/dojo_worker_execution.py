"""Derive worker-forward results only from sealed source and VirtualBroker ledgers.

This is the operational bridge between the prospective OANDA collector and the
existing market-faithful virtual session.  Callers cannot submit performance
numbers: every result is recomputed from the fixed corpus, a hash-chained
ledger, and the precommitted code/config bindings.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.dojo_lab_provenance import (
    DojoLabProvenanceError,
    canonical_strategy_owner_id,
    combine_intrabar_results,
    owner_concurrency_caps_from_config,
    score_session_ledger,
)
from quant_rabbit.dojo_worker_forward import (
    INTRABAR_PATHS,
    build_final_receipt,
    canonical_sha256,
    validate_day_seal,
    validate_precommit,
    validate_start_receipt,
    write_new_json,
)
from quant_rabbit.dojo_worker_source import verify_collected_day


CELL_ATTEMPT_CONTRACT = "QR_DOJO_WORKER_CELL_ATTEMPT_V1"
CELL_TERMINAL_CONTRACT = "QR_DOJO_WORKER_CELL_TERMINAL_V1"
DERIVED_MANIFEST_CONTRACT = "QR_DOJO_WORKER_DERIVED_RESULT_MANIFEST_V1"
EXECUTION_CONTRACT = "QR_DOJO_WORKER_DERIVED_EXECUTION_V1"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
REQUIRED_DEPENDENCIES = frozenset(
    {
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
    }
)


class DojoWorkerExecutionError(ValueError):
    """The derived worker execution is incomplete, mutable, or inconsistent."""


def evaluate_derived_run(
    run_dir: Path,
    *,
    repo_root: Path,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Run/verify the exact 12x2 cells and derive the only operational final."""

    run_dir = _real_directory(run_dir, "run_dir")
    repo_root = _real_directory(repo_root, "repo_root")
    now = _utc(now_utc or datetime.now(timezone.utc), "now_utc")
    lock_path = run_dir / ".worker-execution.lock"
    descriptor = _open_lock(lock_path)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DojoWorkerExecutionError(
                "another derived worker evaluator owns the run"
            ) from exc
        precommit, start, days = _load_lifecycle(run_dir)
        end = _parse_utc(precommit["window"]["end_utc"], "window.end_utc")
        if now < end:
            raise DojoWorkerExecutionError(
                "worker evaluation is before window maturity"
            )
        _verify_pinned_sources(precommit, repo_root)
        source_rows = _verify_all_source_bundles(run_dir, precommit, start, days)
        source_bundle_sha = canonical_sha256(source_rows)
        expected_corpus = _expected_corpus_manifest(run_dir, source_rows)
        _verify_execution_directory_set(run_dir, precommit, require_all=False)
        terminals: list[dict[str, Any]] = []
        executable = _bound_python_executable(precommit)
        for candidate in precommit["candidate_set"]["candidates"]:
            for intrabar in INTRABAR_PATHS:
                terminals.append(
                    _run_or_verify_cell(
                        run_dir,
                        repo_root=repo_root,
                        executable=executable,
                        precommit=precommit,
                        candidate=candidate,
                        intrabar=intrabar,
                        source_bundle_sha256=source_bundle_sha,
                        expected_corpus=expected_corpus,
                    )
                )
        _verify_execution_directory_set(run_dir, precommit, require_all=True)
        result_manifest = _derive_result_manifest(
            precommit, terminals, source_bundle_sha256=source_bundle_sha
        )
        manifest_path = run_dir / "result-manifest.json"
        if manifest_path.exists():
            if _strict_json(manifest_path) != result_manifest:
                raise DojoWorkerExecutionError("derived result manifest drifted")
        else:
            write_new_json(manifest_path, result_manifest)
        final = build_final_receipt(
            precommit,
            start,
            days,
            {
                key: result_manifest[key]
                for key in (
                    "precommit_sha256",
                    "window_start_utc",
                    "window_end_utc",
                    "results",
                )
            },
            now_utc=now,
        )
        final_path = run_dir / "final.json"
        if final_path.exists():
            if _strict_json(final_path) != final:
                raise DojoWorkerExecutionError("derived final receipt drifted")
        else:
            write_new_json(final_path, final)
        return {
            "contract": EXECUTION_CONTRACT,
            "state": final["state"],
            "cell_count": len(terminals),
            "source_day_count": len(source_rows),
            "source_bundle_sha256": source_bundle_sha,
            "result_manifest_sha256": result_manifest["derived_manifest_sha256"],
            "final_receipt_sha256": final["final_receipt_sha256"],
            "smoke_gate_passed": final["smoke_gate_passed"],
            "promotion_eligible": False,
            "live_permission": False,
        }
    finally:
        os.close(descriptor)


def verify_derived_run(run_dir: Path, *, repo_root: Path) -> dict[str, Any]:
    """Re-open every source, ledger, score receipt, manifest, and final."""

    run_dir = _real_directory(run_dir, "run_dir")
    repo_root = _real_directory(repo_root, "repo_root")
    precommit, start, days = _load_lifecycle(run_dir)
    _verify_pinned_sources(precommit, repo_root)
    source_rows = _verify_all_source_bundles(run_dir, precommit, start, days)
    source_bundle_sha = canonical_sha256(source_rows)
    expected_corpus = _expected_corpus_manifest(run_dir, source_rows)
    _verify_execution_directory_set(run_dir, precommit, require_all=True)
    terminals = [
        _verify_cell_terminal(
            run_dir,
            repo_root=repo_root,
            precommit=precommit,
            candidate=candidate,
            intrabar=intrabar,
            source_bundle_sha256=source_bundle_sha,
            expected_corpus=expected_corpus,
        )
        for candidate in precommit["candidate_set"]["candidates"]
        for intrabar in INTRABAR_PATHS
    ]
    derived = _derive_result_manifest(
        precommit, terminals, source_bundle_sha256=source_bundle_sha
    )
    if _strict_json(run_dir / "result-manifest.json") != derived:
        raise DojoWorkerExecutionError("persisted result manifest is not derived")
    final = _strict_json(run_dir / "final.json")
    expected = build_final_receipt(
        precommit,
        start,
        days,
        {
            key: derived[key]
            for key in (
                "precommit_sha256",
                "window_start_utc",
                "window_end_utc",
                "results",
            )
        },
        now_utc=_parse_utc(final["finalized_at_utc"], "finalized_at_utc"),
    )
    if final != expected:
        raise DojoWorkerExecutionError("persisted final is not ledger-derived")
    return {
        "contract": EXECUTION_CONTRACT,
        "state": final["state"],
        "cell_count": len(terminals),
        "source_day_count": len(source_rows),
        "source_bundle_sha256": source_bundle_sha,
        "result_manifest_sha256": derived["derived_manifest_sha256"],
        "final_receipt_sha256": final["final_receipt_sha256"],
        "smoke_gate_passed": final["smoke_gate_passed"],
        "promotion_eligible": False,
        "live_permission": False,
    }


def verify_source_bindings(precommit: Mapping[str, Any], repo_root: Path) -> None:
    """Public preflight used before sealing a fresh operational precommit."""

    _verify_pinned_sources(validate_precommit(precommit), repo_root.resolve())


def _load_lifecycle(
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    precommit = validate_precommit(_strict_json(run_dir / "precommit.json"))
    if (
        precommit["candidate_set"]["candidate_count"] != 12
        or precommit["candidate_set"]["family_denominator"] != 3
    ):
        raise DojoWorkerExecutionError(
            "operational worker smoke requires exactly 12 candidates in 3 families"
        )
    start = validate_start_receipt(_strict_json(run_dir / "start.json"), precommit)
    paths = sorted((run_dir / "days").glob("day-*.json"))
    if len(paths) != precommit["window"]["calendar_days"]:
        raise DojoWorkerExecutionError("all exact daily seals are required")
    days: list[dict[str, Any]] = []
    for ordinal, path in enumerate(paths, start=1):
        if path.name != f"day-{ordinal:03d}.json":
            raise DojoWorkerExecutionError("daily seal set contains a gap or extra")
        seal = validate_day_seal(
            _strict_json(path), precommit, start, expected_ordinal=ordinal
        )
        expected_parent = (
            start["start_receipt_sha256"]
            if ordinal == 1
            else days[-1]["day_seal_sha256"]
        )
        if seal["previous_receipt_sha256"] != expected_parent:
            raise DojoWorkerExecutionError("daily seal chain is broken")
        days.append(seal)
    return precommit, start, days


def _verify_all_source_bundles(
    run_dir: Path,
    precommit: Mapping[str, Any],
    start: Mapping[str, Any],
    days: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected_names = {f"day-{ordinal:03d}" for ordinal in range(1, len(days) + 1)}
    evidence_root = _real_directory(run_dir / "source-evidence", "source evidence")
    actual_names = {path.name for path in evidence_root.iterdir() if path.is_dir()}
    if actual_names != expected_names:
        raise DojoWorkerExecutionError("source evidence day set is not exact")
    rows: list[dict[str, Any]] = []
    for ordinal, day in enumerate(days, start=1):
        verified = verify_collected_day(run_dir, ordinal=ordinal)
        if verified["day_seal_sha256"] != day["day_seal_sha256"]:
            raise DojoWorkerExecutionError("source bundle/day seal binding drifted")
        receipt = _strict_json(
            run_dir
            / "source-evidence"
            / f"day-{ordinal:03d}"
            / "acquisition-receipt.json"
        )
        rows.append(
            {
                "ordinal": ordinal,
                "day_seal_sha256": day["day_seal_sha256"],
                "acquisition_receipt_sha256": verified["acquisition_receipt_sha256"],
                "source_manifest_sha256": verified["source_manifest_sha256"],
                "response_sha256": receipt["response_sha256"],
                "source_content_sha256": receipt["source_content_sha256"],
                "source_relpath": receipt["source_relpath"],
                "source_size_bytes": receipt["source_size_bytes"],
                "market_closed": verified["market_closed"],
            }
        )
    if rows[0]["ordinal"] != 1 or rows[-1]["ordinal"] != len(days):
        raise DojoWorkerExecutionError("source bundle ordinal coverage is incomplete")
    return rows


def _expected_corpus_manifest(
    run_dir: Path, source_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    corpus_root = _real_directory(run_dir / "corpus", "corpus root")
    shards: list[dict[str, Any]] = []
    expected_paths: set[str] = set()
    for row in source_rows:
        relative = row.get("source_relpath")
        if relative is None:
            if not row.get("market_closed"):
                raise DojoWorkerExecutionError("open source day has no corpus shard")
            continue
        if row.get("market_closed"):
            raise DojoWorkerExecutionError("closed source day has a corpus shard")
        rel = Path(relative)
        if rel.is_absolute() or len(rel.parts) < 2 or rel.parts[0] != "corpus":
            raise DojoWorkerExecutionError("source receipt corpus path is unsafe")
        corpus_relative = Path(*rel.parts[1:]).as_posix()
        if corpus_relative in expected_paths:
            raise DojoWorkerExecutionError("source receipt corpus shard is duplicated")
        expected_paths.add(corpus_relative)
        path = _safe_file(run_dir, rel)
        expected_size = row.get("source_size_bytes")
        expected_sha = row.get("source_content_sha256")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size <= 0
            or path.stat().st_size != expected_size
            or not isinstance(expected_sha, str)
            or _file_sha256(path) != expected_sha
        ):
            raise DojoWorkerExecutionError("source receipt corpus bytes drifted")
        shards.append(
            {
                "path": corpus_relative,
                "size_bytes": expected_size,
                "sha256": expected_sha,
            }
        )
    if not shards:
        raise DojoWorkerExecutionError("worker corpus contains no open-market shard")
    shards.sort(key=lambda row: row["path"])
    body = {"root": str(corpus_root), "shards": shards}
    expected = {**body, "corpus_sha256": canonical_sha256(body)}
    _verify_exact_corpus(corpus_root, expected)
    return expected


def _verify_exact_corpus(corpus_root: Path, expected_corpus: Mapping[str, Any]) -> None:
    corpus_root = _real_directory(corpus_root, "corpus root")
    expected_shards = expected_corpus.get("shards")
    if expected_corpus.get("root") != str(corpus_root) or not isinstance(
        expected_shards, list
    ):
        raise DojoWorkerExecutionError("expected corpus identity drifted")
    body = {"root": str(corpus_root), "shards": expected_shards}
    if expected_corpus.get("corpus_sha256") != canonical_sha256(body):
        raise DojoWorkerExecutionError("expected corpus manifest digest drifted")
    expected_by_path = {row["path"]: row for row in expected_shards}
    if len(expected_by_path) != len(expected_shards):
        raise DojoWorkerExecutionError("expected corpus paths are duplicated")
    actual_by_path: dict[str, Path] = {}
    for candidate in corpus_root.rglob("*"):
        if candidate.is_symlink():
            raise DojoWorkerExecutionError("corpus contains a symlink")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise DojoWorkerExecutionError("corpus contains a non-file entry")
        actual_by_path[candidate.relative_to(corpus_root).as_posix()] = candidate
    if set(actual_by_path) != set(expected_by_path):
        raise DojoWorkerExecutionError("corpus file set differs from sealed receipts")
    for relative, path in actual_by_path.items():
        expected = expected_by_path[relative]
        if (
            path.stat().st_size != expected["size_bytes"]
            or _file_sha256(path) != expected["sha256"]
        ):
            raise DojoWorkerExecutionError("corpus bytes differ from sealed receipts")


def _cell_command(
    run_dir: Path,
    *,
    repo_root: Path,
    executable: Path,
    precommit: Mapping[str, Any],
    candidate: Mapping[str, Any],
    intrabar: str,
    config: Mapping[str, Any],
) -> list[str]:
    mechanics = precommit["mechanics"]
    command = [
        str(executable),
        "-I",
        str(repo_root / "scripts/run-virtual-market-session.py"),
        "--feed",
        "replay",
        "--session-dir",
        str(_cell_dir(run_dir, candidate["candidate_id"], intrabar) / "session"),
        "--pairs",
        ",".join(mechanics["pairs"]),
        "--balance",
        str(mechanics["initial_balance_jpy"]),
        "--corpus-root",
        str(run_dir / "corpus"),
        "--from",
        precommit["window"]["start_utc"],
        "--to",
        precommit["window"]["end_utc"],
        "--bars-per-second",
        "100000",
        "--state-every",
        "1000",
        "--bot-module",
        f"{repo_root / 'bots/lab_bot.py'}:Bot",
        "--granularity",
        mechanics["granularity"],
        "--bot-bar",
        mechanics["bot_bar"],
        "--slippage-pips",
        str(mechanics["slippage_pips_per_fill"]),
        "--financing-pips-day",
        str(mechanics["financing_pips_per_day"]),
        "--intrabar",
        intrabar,
        "--strategy-owner-id",
        str(config["strategy_owner_id"]),
        "--settle-at-end",
    ]
    for dependency in sorted(precommit["source_bindings"]["bot_dependency_sha256"]):
        command.extend(["--bot-dependency", dependency])
    return command


def _run_or_verify_cell(
    run_dir: Path,
    *,
    repo_root: Path,
    executable: Path,
    precommit: Mapping[str, Any],
    candidate: Mapping[str, Any],
    intrabar: str,
    source_bundle_sha256: str,
    expected_corpus: Mapping[str, Any],
) -> dict[str, Any]:
    _verify_exact_corpus(run_dir / "corpus", expected_corpus)
    cell_dir = _cell_dir(run_dir, candidate["candidate_id"], intrabar)
    terminal_path = cell_dir / "terminal.json"
    if terminal_path.exists():
        return _verify_cell_terminal(
            run_dir,
            repo_root=repo_root,
            precommit=precommit,
            candidate=candidate,
            intrabar=intrabar,
            source_bundle_sha256=source_bundle_sha256,
            expected_corpus=expected_corpus,
        )
    cell_dir.mkdir(parents=True, exist_ok=True)
    session_dir = cell_dir / "session"
    config = {
        **candidate["config"],
        "strategy_owner_id": canonical_strategy_owner_id(
            candidate["config"], namespace="dojo-worker-forward"
        ),
    }
    config_text = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    config_sha = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
    command = _cell_command(
        run_dir,
        repo_root=repo_root,
        executable=executable,
        precommit=precommit,
        candidate=candidate,
        intrabar=intrabar,
        config=config,
    )
    attempt_path = cell_dir / "attempt.json"
    attempt_existed = attempt_path.exists()
    if not attempt_existed:
        attempt_body = {
            "contract": CELL_ATTEMPT_CONTRACT,
            "schema_version": 1,
            "candidate_id": candidate["candidate_id"],
            "intrabar": intrabar,
            "precommit_sha256": precommit["precommit_sha256"],
            "source_bundle_sha256": source_bundle_sha256,
            "command_sha256": canonical_sha256(command),
            "bot_config_sha256": config_sha,
            "bot_config_length": len(config_text),
            "state": "FIRST_ATTEMPT_FIXED",
            "authority": _authority(),
        }
        write_new_json(
            attempt_path,
            {**attempt_body, "attempt_sha256": canonical_sha256(attempt_body)},
        )
    attempt = _strict_json(attempt_path)
    _validate_attempt(
        attempt,
        candidate_id=candidate["candidate_id"],
        intrabar=intrabar,
        precommit_sha256=precommit["precommit_sha256"],
        source_bundle_sha256=source_bundle_sha256,
        command_sha256=canonical_sha256(command),
        config_sha256=config_sha,
        config_length=len(config_text),
    )
    if attempt_existed:
        # A process died after fixing the first attempt but before publishing
        # its terminal.  Never launch a second selectable run.  A fully
        # scoreable existing ledger may be terminalized; anything else is the
        # permanent failure for this cell.
        ledger = session_dir / "ledger.jsonl"
        score: dict[str, Any] | None = None
        if ledger.is_file():
            try:
                score = _score_cell(
                    ledger,
                    repo_root=repo_root,
                    precommit=precommit,
                    candidate=candidate,
                    intrabar=intrabar,
                    config_sha256=config_sha,
                    config_length=len(config_text),
                    owner_id=config["strategy_owner_id"],
                    expected_corpus=expected_corpus,
                )
            except DojoWorkerExecutionError:
                score = None
        recovery_body = {
            "contract": CELL_TERMINAL_CONTRACT,
            "schema_version": 1,
            "candidate_id": candidate["candidate_id"],
            "intrabar": intrabar,
            "precommit_sha256": precommit["precommit_sha256"],
            "source_bundle_sha256": source_bundle_sha256,
            "attempt_sha256": attempt["attempt_sha256"],
            "outcome": "SCORED" if score is not None else "RUNNER_FAILURE",
            "runner_returncode": 0 if score is not None else -1,
            "runner_stdout_sha256": EMPTY_SHA256,
            "runner_stderr_sha256": EMPTY_SHA256,
            "ledger_relpath": (
                str(ledger.relative_to(run_dir)) if ledger.is_file() else None
            ),
            "ledger_size_bytes": ledger.stat().st_size if ledger.is_file() else 0,
            "ledger_sha256": _file_sha256(ledger) if ledger.is_file() else EMPTY_SHA256,
            "score": score,
            "authority": _authority(),
        }
        terminal = {
            **recovery_body,
            "score_receipt_sha256": canonical_sha256(recovery_body),
        }
        write_new_json(terminal_path, terminal)
        return _verify_cell_terminal(
            run_dir,
            repo_root=repo_root,
            precommit=precommit,
            candidate=candidate,
            intrabar=intrabar,
            source_bundle_sha256=source_bundle_sha256,
            expected_corpus=expected_corpus,
        )
    env = {
        "DOJO_BOT_CONFIG": config_text,
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "TZ": "UTC",
    }
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    _verify_exact_corpus(run_dir / "corpus", expected_corpus)
    if completed.returncode != 0:
        ledger = session_dir / "ledger.jsonl"
        terminal_body = {
            "contract": CELL_TERMINAL_CONTRACT,
            "schema_version": 1,
            "candidate_id": candidate["candidate_id"],
            "intrabar": intrabar,
            "precommit_sha256": precommit["precommit_sha256"],
            "source_bundle_sha256": source_bundle_sha256,
            "attempt_sha256": attempt["attempt_sha256"],
            "outcome": "RUNNER_FAILURE",
            "runner_returncode": completed.returncode,
            "runner_stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
            "runner_stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
            "ledger_relpath": (
                str(ledger.relative_to(run_dir)) if ledger.is_file() else None
            ),
            "ledger_size_bytes": ledger.stat().st_size if ledger.is_file() else 0,
            "ledger_sha256": _file_sha256(ledger) if ledger.is_file() else EMPTY_SHA256,
            "score": None,
            "authority": _authority(),
        }
    else:
        ledger = session_dir / "ledger.jsonl"
        score = _score_cell(
            ledger,
            repo_root=repo_root,
            precommit=precommit,
            candidate=candidate,
            intrabar=intrabar,
            config_sha256=config_sha,
            config_length=len(config_text),
            owner_id=config["strategy_owner_id"],
            expected_corpus=expected_corpus,
        )
        terminal_body = {
            "contract": CELL_TERMINAL_CONTRACT,
            "schema_version": 1,
            "candidate_id": candidate["candidate_id"],
            "intrabar": intrabar,
            "precommit_sha256": precommit["precommit_sha256"],
            "source_bundle_sha256": source_bundle_sha256,
            "attempt_sha256": attempt["attempt_sha256"],
            "outcome": "SCORED",
            "runner_returncode": 0,
            "runner_stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
            "runner_stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
            "ledger_relpath": str(ledger.relative_to(run_dir)),
            "ledger_size_bytes": ledger.stat().st_size,
            "ledger_sha256": _file_sha256(ledger),
            "score": score,
            "authority": _authority(),
        }
    terminal = {
        **terminal_body,
        "score_receipt_sha256": canonical_sha256(terminal_body),
    }
    write_new_json(terminal_path, terminal)
    return _verify_cell_terminal(
        run_dir,
        repo_root=repo_root,
        precommit=precommit,
        candidate=candidate,
        intrabar=intrabar,
        source_bundle_sha256=source_bundle_sha256,
        expected_corpus=expected_corpus,
    )


def _verify_cell_terminal(
    run_dir: Path,
    *,
    repo_root: Path,
    precommit: Mapping[str, Any],
    candidate: Mapping[str, Any],
    intrabar: str,
    source_bundle_sha256: str,
    expected_corpus: Mapping[str, Any],
) -> dict[str, Any]:
    _verify_exact_corpus(run_dir / "corpus", expected_corpus)
    cell_dir = _cell_dir(run_dir, candidate["candidate_id"], intrabar)
    terminal = _strict_json(cell_dir / "terminal.json")
    body = {
        key: value for key, value in terminal.items() if key != "score_receipt_sha256"
    }
    if (
        terminal.get("contract") != CELL_TERMINAL_CONTRACT
        or terminal.get("schema_version") != 1
        or terminal.get("candidate_id") != candidate["candidate_id"]
        or terminal.get("intrabar") != intrabar
        or terminal.get("precommit_sha256") != precommit["precommit_sha256"]
        or terminal.get("source_bundle_sha256") != source_bundle_sha256
        or terminal.get("authority") != _authority()
        or terminal.get("score_receipt_sha256") != canonical_sha256(body)
    ):
        raise DojoWorkerExecutionError("cell terminal identity or digest drifted")
    attempt = _strict_json(cell_dir / "attempt.json")
    config = {
        **candidate["config"],
        "strategy_owner_id": canonical_strategy_owner_id(
            candidate["config"], namespace="dojo-worker-forward"
        ),
    }
    config_text = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    command = _cell_command(
        run_dir,
        repo_root=repo_root,
        executable=_bound_python_executable(precommit),
        precommit=precommit,
        candidate=candidate,
        intrabar=intrabar,
        config=config,
    )
    _validate_attempt(
        attempt,
        candidate_id=candidate["candidate_id"],
        intrabar=intrabar,
        precommit_sha256=precommit["precommit_sha256"],
        source_bundle_sha256=source_bundle_sha256,
        command_sha256=canonical_sha256(command),
        config_sha256=hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
        config_length=len(config_text),
    )
    if terminal["attempt_sha256"] != attempt.get("attempt_sha256"):
        raise DojoWorkerExecutionError("cell terminal attempt parent drifted")
    ledger_relpath = terminal["ledger_relpath"]
    if ledger_relpath is None:
        if (
            terminal["outcome"] != "RUNNER_FAILURE"
            or terminal["ledger_sha256"] != EMPTY_SHA256
        ):
            raise DojoWorkerExecutionError(
                "ledger-free terminal is not a runner failure"
            )
        return terminal
    ledger = _safe_file(run_dir, ledger_relpath)
    if (
        ledger.stat().st_size != terminal["ledger_size_bytes"]
        or _file_sha256(ledger) != terminal["ledger_sha256"]
    ):
        raise DojoWorkerExecutionError("cell ledger bytes drifted")
    if terminal["outcome"] == "RUNNER_FAILURE":
        if terminal["score"] is not None:
            raise DojoWorkerExecutionError("runner failure cannot contain a score")
        return terminal
    if terminal["outcome"] != "SCORED" or terminal["runner_returncode"] != 0:
        raise DojoWorkerExecutionError("cell terminal outcome is invalid")
    recomputed = _score_cell(
        ledger,
        repo_root=repo_root,
        precommit=precommit,
        candidate=candidate,
        intrabar=intrabar,
        config_sha256=hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
        config_length=len(config_text),
        owner_id=config["strategy_owner_id"],
        expected_corpus=expected_corpus,
    )
    if terminal["score"] != recomputed:
        raise DojoWorkerExecutionError("persisted cell score is not ledger-derived")
    return terminal


def _score_cell(
    ledger: Path,
    *,
    repo_root: Path,
    precommit: Mapping[str, Any],
    candidate: Mapping[str, Any],
    intrabar: str,
    config_sha256: str,
    config_length: int,
    owner_id: str,
    expected_corpus: Mapping[str, Any],
) -> dict[str, Any]:
    mechanics = precommit["mechanics"]
    bindings = precommit["source_bindings"]
    expected_owner_id = canonical_strategy_owner_id(
        candidate["config"], namespace="dojo-worker-forward"
    )
    expected_config = {
        **candidate["config"],
        "strategy_owner_id": expected_owner_id,
    }
    expected_config_text = json.dumps(
        expected_config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if (
        owner_id != expected_owner_id
        or config_sha256
        != hashlib.sha256(expected_config_text.encode("utf-8")).hexdigest()
        or config_length != len(expected_config_text)
    ):
        raise DojoWorkerExecutionError(
            "score inputs differ from the sealed candidate config"
        )
    try:
        pair_cap, global_cap = owner_concurrency_caps_from_config(expected_config)
        score = score_session_ledger(
            ledger,
            start_balance_jpy=mechanics["initial_balance_jpy"],
            window_role="FORWARD",
            window=(
                precommit["window"]["start_utc"],
                precommit["window"]["end_utc"],
            ),
            intrabar=intrabar,
            legacy_contaminated=False,
            expected_pairs=mechanics["pairs"],
            expected_granularity=mechanics["granularity"],
            expected_bot_bar=mechanics["bot_bar"],
            expected_period_end_settlement=True,
            expected_slippage_pips=mechanics["slippage_pips_per_fill"],
            expected_financing_pips_per_day=mechanics["financing_pips_per_day"],
            expected_bot_module_path=repo_root / "bots/lab_bot.py",
            expected_bot_module_sha256=bindings["bot_module_sha256"],
            expected_bot_dependency_sha256=bindings["bot_dependency_sha256"],
            expected_strategy_owner_id=owner_id,
            expected_bot_config_sha256=config_sha256,
            expected_bot_config_length=config_length,
            reservation_evidence=None,
            expected_max_concurrent_per_pair=pair_cap,
            expected_global_max_concurrent=global_cap,
        )
    except DojoLabProvenanceError as exc:
        raise DojoWorkerExecutionError(f"ledger scoring failed: {exc}") from exc
    owner_concurrency = score.get("owner_concurrency")
    if (
        not isinstance(owner_concurrency, Mapping)
        or owner_concurrency.get("status") != "VERIFIED_FROM_FILL_EXIT_LEDGER"
        or owner_concurrency.get("strategy_owner_id") != expected_owner_id
        or owner_concurrency.get("max_concurrent_per_pair") != pair_cap
        or owner_concurrency.get("global_max_concurrent") != global_cap
    ):
        raise DojoWorkerExecutionError(
            "ledger score did not verify the sealed owner concurrency caps"
        )
    manifest = _ledger_reproducibility_manifest(ledger)
    runtime = manifest.get("source")
    bindings = precommit["source_bindings"]
    if (
        manifest.get("corpus") != expected_corpus
        or not isinstance(runtime, Mapping)
        or runtime.get("python_executable") != bindings["python_executable_path"]
        or runtime.get("python_version") != bindings["python_version"]
    ):
        raise DojoWorkerExecutionError(
            "ledger runtime or corpus differs from the precommit"
        )
    if score["hardened_costs"]["leverage"] != mechanics["leverage"]:
        raise DojoWorkerExecutionError("ledger leverage differs from precommit")
    if score["corpus_manifest_sha256"] != canonical_sha256(expected_corpus) or score[
        "corpus_shard_count"
    ] != len(expected_corpus["shards"]):
        raise DojoWorkerExecutionError(
            "ledger corpus manifest differs from sealed source receipts"
        )
    if score["promotion_eligible"] is not False:
        raise DojoWorkerExecutionError("worker scorer unexpectedly granted promotion")
    return score


def _ledger_reproducibility_manifest(ledger: Path) -> dict[str, Any]:
    try:
        with ledger.open("r", encoding="utf-8") as handle:
            first = handle.readline()
    except OSError as exc:
        raise DojoWorkerExecutionError("cannot read ledger SESSION_START") from exc

    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DojoWorkerExecutionError(
                    f"duplicate ledger SESSION_START key: {key}"
                )
            result[key] = value
        return result

    try:
        record = json.loads(
            first,
            object_pairs_hook=reject,
            parse_constant=lambda token: (_ for _ in ()).throw(
                DojoWorkerExecutionError(
                    f"non-finite ledger SESSION_START number: {token}"
                )
            ),
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DojoWorkerExecutionError("ledger SESSION_START is invalid JSON") from exc
    if not isinstance(record, Mapping) or record.get("event") != "SESSION_START":
        raise DojoWorkerExecutionError("ledger does not begin with SESSION_START")
    payload = record.get("payload")
    manifest = (
        payload.get("reproducibility_manifest")
        if isinstance(payload, Mapping)
        else None
    )
    if not isinstance(manifest, dict):
        raise DojoWorkerExecutionError("ledger SESSION_START manifest is absent")
    return manifest


def _derive_result_manifest(
    precommit: Mapping[str, Any],
    terminals: Sequence[Mapping[str, Any]],
    *,
    source_bundle_sha256: str,
) -> dict[str, Any]:
    expected = {
        (candidate["candidate_id"], intrabar)
        for candidate in precommit["candidate_set"]["candidates"]
        for intrabar in INTRABAR_PATHS
    }
    if len(expected) != 24:
        raise DojoWorkerExecutionError("worker terminal denominator is not exact 12x2")
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    results: list[dict[str, Any]] = []
    for terminal in terminals:
        key = (terminal["candidate_id"], terminal["intrabar"])
        if key not in expected or key in by_key:
            raise DojoWorkerExecutionError("cell terminal set is unknown or duplicated")
        by_key[key] = terminal
        score = terminal["score"]
        if terminal["outcome"] == "SCORED":
            row = {
                "candidate_id": key[0],
                "intrabar": key[1],
                "status": "INVALID_UNSCOREABLE_TRIAL",
                "ledger_sha256": terminal["ledger_sha256"],
                "score_receipt_sha256": terminal["score_receipt_sha256"],
                "entries": score["entries"],
                "resolved_exits": score["resolved_exits"],
                "terminal_net_jpy": score["terminal_net_jpy"],
                "calendar_30d_multiple": score["calendar_30d_multiple"],
                "margin_closeouts": score["margin_closeouts"],
                "terminal_resolved": score["terminal_resolution_verified"],
                "promotion_eligible": False,
            }
        else:
            row = {
                "candidate_id": key[0],
                "intrabar": key[1],
                "status": "INVALID_RUNNER_FAILURE",
                "ledger_sha256": terminal["ledger_sha256"],
                "score_receipt_sha256": terminal["score_receipt_sha256"],
                "entries": 0,
                "resolved_exits": 0,
                "terminal_net_jpy": 0.0,
                "calendar_30d_multiple": 0.0,
                "margin_closeouts": 0,
                "terminal_resolved": False,
                "promotion_eligible": False,
            }
        results.append(row)
    if set(by_key) != expected:
        raise DojoWorkerExecutionError("exact 12x2 terminal denominator is incomplete")
    for candidate in precommit["candidate_set"]["candidates"]:
        scored = [
            terminal["score"]
            for terminal in terminals
            if terminal["candidate_id"] == candidate["candidate_id"]
            and terminal["outcome"] == "SCORED"
        ]
        if len(scored) == 2:
            combine_intrabar_results(scored)
    results.sort(key=lambda row: (row["candidate_id"], row["intrabar"]))
    body = {
        "contract": DERIVED_MANIFEST_CONTRACT,
        "schema_version": 1,
        "precommit_sha256": precommit["precommit_sha256"],
        "window_start_utc": precommit["window"]["start_utc"],
        "window_end_utc": precommit["window"]["end_utc"],
        "source_bundle_sha256": source_bundle_sha256,
        "result_derivation": (
            "EXACT_12X2_PINNED_RUNNER_LEDGER_CONSISTENCY_ONLY;"
            "ECONOMIC_EVENT_REPLAY_OR_EXTERNAL_WITNESS_ABSENT"
        ),
        "results": results,
        "authority": _authority(),
    }
    return {**body, "derived_manifest_sha256": canonical_sha256(body)}


def _verify_pinned_sources(precommit: Mapping[str, Any], repo_root: Path) -> None:
    bindings = precommit["source_bindings"]
    _bound_python_executable(precommit)
    dependencies = bindings["bot_dependency_sha256"]
    missing = REQUIRED_DEPENDENCIES - set(dependencies)
    if missing:
        raise DojoWorkerExecutionError(
            "operational dependency closure is incomplete: " + ",".join(sorted(missing))
        )
    checks = {
        "bots/lab_bot.py": bindings["bot_module_sha256"],
        "scripts/run-virtual-market-session.py": bindings["runner_sha256"],
        "src/quant_rabbit/dojo_lab_provenance.py": bindings["scorer_sha256"],
        "src/quant_rabbit/dojo_worker_forward.py": bindings["precommit_builder_sha256"],
        **dependencies,
    }
    commit = bindings["git_commit"]
    for relative, expected in sorted(checks.items()):
        path = _safe_file(repo_root, relative)
        if _file_sha256(path) != expected:
            raise DojoWorkerExecutionError(
                f"current source binding drifted: {relative}"
            )
        completed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if (
            completed.returncode != 0
            or hashlib.sha256(completed.stdout).hexdigest() != expected
        ):
            raise DojoWorkerExecutionError(f"commit source binding drifted: {relative}")


def _bound_python_executable(precommit: Mapping[str, Any]) -> Path:
    bindings = precommit["source_bindings"]
    required = {
        "python_executable_path",
        "python_executable_sha256",
        "python_version",
    }
    if not required.issubset(bindings):
        raise DojoWorkerExecutionError(
            "operational worker precommit lacks a Python runtime binding"
        )
    executable = Path(sys.executable).resolve()
    if (
        bindings["python_executable_path"] != str(executable)
        or not executable.is_file()
        or executable.is_symlink()
        or _file_sha256(executable) != bindings["python_executable_sha256"]
        or sys.version != bindings["python_version"]
    ):
        raise DojoWorkerExecutionError("Python runtime binding drifted")
    return executable


def _validate_attempt(
    attempt: Mapping[str, Any],
    *,
    candidate_id: str,
    intrabar: str,
    precommit_sha256: str,
    source_bundle_sha256: str,
    command_sha256: str,
    config_sha256: str,
    config_length: int,
) -> None:
    body = {key: value for key, value in attempt.items() if key != "attempt_sha256"}
    if (
        attempt.get("contract") != CELL_ATTEMPT_CONTRACT
        or attempt.get("schema_version") != 1
        or attempt.get("candidate_id") != candidate_id
        or attempt.get("intrabar") != intrabar
        or attempt.get("precommit_sha256") != precommit_sha256
        or attempt.get("source_bundle_sha256") != source_bundle_sha256
        or attempt.get("command_sha256") != command_sha256
        or attempt.get("bot_config_sha256") != config_sha256
        or attempt.get("bot_config_length") != config_length
        or attempt.get("state") != "FIRST_ATTEMPT_FIXED"
        or attempt.get("authority") != _authority()
        or attempt.get("attempt_sha256") != canonical_sha256(body)
    ):
        raise DojoWorkerExecutionError("cell attempt identity or digest drifted")


def _cell_dir(run_dir: Path, candidate_id: str, intrabar: str) -> Path:
    if Path(candidate_id).name != candidate_id or intrabar not in INTRABAR_PATHS:
        raise DojoWorkerExecutionError("unsafe cell identity")
    path = run_dir / "execution" / "cells" / candidate_id / intrabar
    if path.exists() and (path.is_symlink() or not path.is_dir()):
        raise DojoWorkerExecutionError("cell path is not a real directory")
    return path


def _verify_execution_directory_set(
    run_dir: Path, precommit: Mapping[str, Any], *, require_all: bool
) -> None:
    root = run_dir / "execution" / "cells"
    expected = {
        (candidate["candidate_id"], intrabar)
        for candidate in precommit["candidate_set"]["candidates"]
        for intrabar in INTRABAR_PATHS
    }
    actual: set[tuple[str, str]] = set()
    if root.exists() or root.is_symlink():
        root = _real_directory(root, "worker execution cells")
        for candidate_dir in root.iterdir():
            if candidate_dir.is_symlink() or not candidate_dir.is_dir():
                raise DojoWorkerExecutionError("execution contains an unsafe candidate")
            for intrabar_dir in candidate_dir.iterdir():
                if intrabar_dir.is_symlink() or not intrabar_dir.is_dir():
                    raise DojoWorkerExecutionError("execution contains an unsafe cell")
                actual.add((candidate_dir.name, intrabar_dir.name))
    if not actual.issubset(expected) or (require_all and actual != expected):
        raise DojoWorkerExecutionError("execution cell directory set is not exact")


def _safe_file(root: Path, relative: str | Path) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise DojoWorkerExecutionError("artifact path is unsafe")
    path = root / rel
    if (
        not path.is_file()
        or path.is_symlink()
        or root.resolve() not in path.resolve().parents
    ):
        raise DojoWorkerExecutionError(f"artifact is missing or unsafe: {rel}")
    return path


def _real_directory(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not path.is_dir() or path.is_symlink() or resolved != path.absolute():
        raise DojoWorkerExecutionError(f"{label} must be a real resolved directory")
    return resolved


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DojoWorkerExecutionError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject,
            parse_constant=lambda token: (_ for _ in ()).throw(
                DojoWorkerExecutionError(f"non-finite JSON number: {token}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise DojoWorkerExecutionError(f"cannot read {path.name}") from exc
    if not isinstance(value, dict):
        raise DojoWorkerExecutionError(f"{path.name} must contain an object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_lock(path: Path) -> int:
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(path, flags, 0o600)


def _parse_utc(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise DojoWorkerExecutionError(f"{field} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DojoWorkerExecutionError(f"{field} is invalid") from exc
    return _utc(parsed, field)


def _utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise DojoWorkerExecutionError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _authority() -> dict[str, Any]:
    return {
        "read_only": True,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
        "live_permission": False,
        "promotion_eligible": False,
    }
