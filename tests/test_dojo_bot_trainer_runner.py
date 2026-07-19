from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

import quant_rabbit.dojo_bot_trainer as trainer_module
from quant_rabbit.dojo_bot_trainer import (
    PROPOSAL_CONTRACT,
    STUDY_CONTRACT,
    DojoBotTrainerError,
    seal_candidate_proposal,
    seal_study,
)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run-dojo-bot-trainer.py"
SPEC = importlib.util.spec_from_file_location("run_dojo_bot_trainer", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

PAIRS = ["EUR_USD", "GBP_USD"]
FEED_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY"]


def _write(path: Path, data: bytes = b"source-v1\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _repo(tmp_path: Path, monkeypatch) -> Path:
    repo = tmp_path / "repo"
    for relative in sorted(runner.MANDATORY_SOURCE_PATHS):
        _write(repo / relative, f"{relative}\n".encode())
    monkeypatch.setattr(runner, "REPO_ROOT", repo)
    return repo


def _corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    for pair in FEED_PAIRS:
        _write_pair_rows(root, pair, (0, 1))
    return root


def _write_pair_rows(root: Path, pair: str, minutes: tuple[int, ...]) -> None:
    path = root / "archive" / pair / f"{pair}_M1_BA_2025.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for minute in minutes:
            handle.write(
                json.dumps(
                    {
                        "time": f"2025-03-01T00:{minute:02d}:00Z",
                        "bid": {"o": 1, "h": 1, "l": 1, "c": 1},
                        "ask": {"o": 2, "h": 2, "l": 2, "c": 2},
                    }
                )
                + "\n"
            )


def _config(*, tp_atr: float = 3.0) -> dict:
    return {
        "signal": "spike_fade",
        "pairs": PAIRS,
        "tp_atr": tp_atr,
        "sl_pips": 25.0,
        "ceiling_min": 60,
        "max_concurrent_per_pair": 1,
        "global_max_concurrent": 2,
        "per_pos_lev": 3.0,
        "atr_floor_pips": 0.5,
    }


def _proposal(candidate_id: str, *, tp_atr: float = 3.0) -> dict:
    return seal_candidate_proposal(
        {
            "contract": PROPOSAL_CONTRACT,
            "schema_version": 1,
            "candidate_id": candidate_id,
            "family": "spike_fade",
            "hypothesis": "A preregistered worn-TRAIN candidate.",
            "config": _config(tp_atr=tp_atr),
            "risk_increase": False,
        }
    )


def _study(corpus_sha256: str, *, two_candidates: bool = False) -> dict:
    candidates = [_proposal("C1")]
    if two_candidates:
        candidates.append(_proposal("C2", tp_atr=4.0))
    return {
        "contract": STUDY_CONTRACT,
        "schema_version": 1,
        "study_id": "runner_train_v1",
        "window_role": "TRAIN",
        "initial_balance_jpy": 200_000.0,
        "trade_pairs": PAIRS,
        "feed_pairs": FEED_PAIRS,
        "candidates": candidates,
        "window": {
            "start_utc": "2025-03-01T00:00:00Z",
            "end_utc": "2025-04-01T00:00:00Z",
            "corpus_id": "runner_fixture_m1",
            "corpus_sha256": corpus_sha256,
            "evidence_tier": "WORN_TRAIN",
        },
        "cost_arms": {
            "BASE": {
                "slippage_pips_per_fill": 0.0,
                "financing_pips_per_day": 0.0,
                "recorded_spread_multiplier": 1.0,
            },
            "STRESS": {
                "slippage_pips_per_fill": 0.3,
                "financing_pips_per_day": 0.8,
                "recorded_spread_multiplier": 1.0,
            },
        },
        "proposer_evidence": {
            "prompt_sha256": "1" * 64,
            "input_sha256": "2" * 64,
            "raw_response_sha256": "3" * 64,
            "model_claim": "test-model",
            "provider_attestation": "UNVERIFIED",
        },
        "search_budget": {
            "attempt_ordinal": 1,
            "total_attempts_in_lineage": 2,
            "max_candidates": 2,
        },
        "thresholds": {
            "normal_mtm_drawdown_max": 0.10,
            "stress_mtm_drawdown_max": 0.15,
            "peak_margin_usage_max": 0.45,
            "margin_reject_rate_max": 0.10,
            "cost_retention_min": 0.50,
            "pair_positive_share_max": 0.50,
            "pair_hhi_max": 0.50,
        },
    }


def _sealed(
    repo: Path,
    corpus: Path,
    *,
    two_candidates: bool = False,
    stress_spread_multiplier: float = 1.0,
) -> dict:
    sources = runner._source_digests(sorted(runner.MANDATORY_SOURCE_PATHS))
    corpus_manifest = runner._corpus_manifest(
        corpus,
        FEED_PAIRS,
        "2025-03-01T00:00:00+00:00",
        "2025-04-01T00:00:00+00:00",
    )
    study = _study(corpus_manifest["corpus_sha256"], two_candidates=two_candidates)
    study["cost_arms"]["STRESS"]["recorded_spread_multiplier"] = (
        stress_spread_multiplier
    )
    return seal_study(study, sources)


def _write_sealed(tmp_path: Path, sealed: dict) -> Path:
    path = tmp_path / "sealed.json"
    path.write_text(json.dumps(sealed, allow_nan=False))
    return path


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _install_successful_replay_mocks(monkeypatch, sealed: dict) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        command = list(command)
        calls.append(command)
        assert "DOJO_BOT_COMBO" not in kwargs["env"]
        session_dir = Path(_option(command, "--session-dir"))
        session_dir.mkdir(parents=True, exist_ok=True)
        feed_pairs = _option(command, "--pairs").split(",")
        corpus = runner._corpus_manifest(
            Path(_option(command, "--corpus-root")),
            feed_pairs,
            _option(command, "--from"),
            _option(command, "--to"),
        )
        config_json = kwargs["env"]["DOJO_BOT_CONFIG"]
        payload = {
            "order_authority": "NONE",
            "reproducibility_manifest": {
                "replay": {
                    "feed": "replay",
                    "pairs": feed_pairs,
                    "time_from": _option(command, "--from"),
                    "time_to": _option(command, "--to"),
                    "granularity": "M1",
                    "intrabar": _option(command, "--intrabar"),
                    "bot_bar": "feed",
                    "period_end_settlement": True,
                    "continuous_mtm": True,
                },
                "costs": {
                    "slippage_pips_per_fill": float(
                        _option(command, "--slippage-pips")
                    ),
                    "financing_pips_per_day": float(
                        _option(command, "--financing-pips-day")
                    ),
                    "leverage": 25.0,
                },
                "corpus": corpus,
                "bot": {
                    "kind": "custom_module",
                    "class": "Bot",
                    "strategy_owner_id": _option(command, "--strategy-owner-id"),
                    "module_sha256": sealed["source_digests"]["bots/lab_bot.py"],
                    "configuration_bindings": {
                        "DOJO_BOT_CONFIG": {
                            "sha256": hashlib.sha256(config_json.encode()).hexdigest(),
                            "length": len(config_json),
                        }
                    },
                    "dependency_sha256": sealed["source_digests"],
                },
            },
        }
        row = {"event": "SESSION_START", "payload": payload}
        (session_dir / "ledger.jsonl").write_text(json.dumps(row) + "\n")
        return subprocess.CompletedProcess(command, 0, stdout="done", stderr="")

    def fake_score(
        ledger_path,
        start_balance_jpy,
        expected_pairs,
        window_start,
        window_end,
        *args,
        **kwargs,
    ):
        if hasattr(ledger_path, "read"):
            raw = ledger_path.read()
        else:
            raw = Path(ledger_path).read_bytes()
        pairs = list(expected_pairs)
        net = 100.0 if len(pairs) == len(PAIRS) else 60.0
        terminal = hashlib.sha256(b"terminal:" + raw).hexdigest()
        metrics_body = {
            "terminal_net_jpy": net,
            "terminal_flat": True,
            "margin_closeouts": 0,
            "realized_max_drawdown_fraction": 0.02,
            "mtm_complete": True,
            "mtm_max_drawdown_fraction": 0.03,
            "peak_entry_margin_estimate_fraction": 0.10,
            "fill_count": 8,
            "margin_reject_count": 0,
            "capital_lock_margin_jpy_hours": 10_000.0,
            "pair_pnl_jpy": {pair: net / len(pairs) for pair in pairs},
            "ledger_size_bytes": len(raw),
            "ledger_file_sha256": hashlib.sha256(raw).hexdigest(),
            "ledger_terminal_sha256": terminal,
            "corpus_sha256": kwargs["expected_corpus_sha256"],
        }
        metrics_body["metrics_sha256"] = hashlib.sha256(
            json.dumps(metrics_body, sort_keys=True).encode()
        ).hexdigest()
        return metrics_body

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner, "score_ledger_metrics", fake_score)
    monkeypatch.setattr(trainer_module, "score_ledger_metrics", fake_score)
    return calls


def _run_args(sealed_path: Path, corpus: Path, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        sealed_study=sealed_path,
        corpus_root=corpus,
        output_dir=output,
    )


def test_seal_hashes_mandatory_sources_and_exclusive_creates_output(
    tmp_path, monkeypatch
) -> None:
    repo = _repo(tmp_path, monkeypatch)
    corpus = _corpus(tmp_path)
    corpus_sha = runner._corpus_manifest(
        corpus,
        FEED_PAIRS,
        "2025-03-01T00:00:00Z",
        "2025-04-01T00:00:00Z",
    )["corpus_sha256"]
    study_path = tmp_path / "study.json"
    study_path.write_text(json.dumps(_study(corpus_sha), allow_nan=False))
    output = tmp_path / "sealed.json"
    args = argparse.Namespace(
        study=study_path,
        source_path=sorted(runner.MANDATORY_SOURCE_PATHS),
        corpus_root=corpus,
        output=output,
    )

    assert runner._seal_command(args) == 0
    sealed = json.loads(output.read_text())
    assert sealed["source_digests"] == {
        relative: hashlib.sha256((repo / relative).read_bytes()).hexdigest()
        for relative in sorted(runner.MANDATORY_SOURCE_PATHS)
    }
    with pytest.raises(FileExistsError):
        runner._seal_command(args)


def test_strict_json_rejects_duplicate_keys(tmp_path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"contract":"first","contract":"second"}')
    with pytest.raises(runner.TrainerRunnerError, match="duplicate JSON key"):
        runner._load_json(path, field="study")


def test_run_refuses_source_drift_before_any_subprocess(tmp_path, monkeypatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    corpus = _corpus(tmp_path)
    sealed = _sealed(repo, corpus)
    sealed_path = _write_sealed(tmp_path, sealed)
    (repo / "bots/lab_bot.py").write_text("drift\n")
    calls: list = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: calls.append(a))

    output = tmp_path / "out"
    assert runner._run_command(_run_args(sealed_path, corpus, output)) == 2
    failure = json.loads((output / "run_failure.json").read_text())
    assert "source digest drift" in failure["error"]
    assert failure["fixed_denominator"]["expected_cell_count"] == 4
    assert calls == []


def test_seal_refuses_unimplemented_recorded_spread_multiplier(
    tmp_path, monkeypatch
) -> None:
    repo = _repo(tmp_path, monkeypatch)
    corpus = _corpus(tmp_path)
    with pytest.raises(DojoBotTrainerError, match="must equal 1"):
        _sealed(repo, corpus, stress_spread_multiplier=1.5)


def test_run_executes_fixed_denominator_and_true_lopo_replays(
    tmp_path, monkeypatch
) -> None:
    repo = _repo(tmp_path, monkeypatch)
    corpus = _corpus(tmp_path)
    sealed = _sealed(repo, corpus, two_candidates=True)
    sealed_path = _write_sealed(tmp_path, sealed)
    calls = _install_successful_replay_mocks(monkeypatch, sealed)

    output = tmp_path / "out"
    assert runner._run_command(_run_args(sealed_path, corpus, output)) == 0
    receipt = json.loads((output / "run.json").read_text())
    cells = json.loads((output / "cells.json").read_text())
    evaluation = json.loads((output / "evaluation.json").read_text())

    assert receipt["fixed_denominator"] == {
        "coordinate_receipts_complete": True,
        "execution_success_complete": True,
        "dropped_cell_count": 0,
        "expected_cell_count": 8,
        "failed_cell_count": 0,
        "observed_cell_count": 8,
    }
    assert len(cells) == 8
    assert evaluation["fixed_denominator"]["observed_cell_count"] == 8
    assert len(calls) == 24  # 8 main + 8 * 2 true leave-one-pair-out replays
    assert all(command[0] == runner.sys.executable for command in calls)
    assert all("--settle-at-end" in command for command in calls)
    assert all("--continuous-mtm" in command for command in calls)
    assert all("--strategy-owner-id" in command for command in calls)
    assert all(
        _option(command, "--pairs").split(",") == FEED_PAIRS for command in calls
    )
    assert all(cell["metrics"]["lopo_replay_complete"] for cell in cells)


def test_lopo_removes_only_trade_pair_and_preserves_conversion_feed(
    tmp_path, monkeypatch
) -> None:
    repo = _repo(tmp_path, monkeypatch)
    corpus = _corpus(tmp_path)
    sealed = _sealed(repo, corpus)
    sealed_path = _write_sealed(tmp_path, sealed)
    calls = _install_successful_replay_mocks(monkeypatch, sealed)
    observed_trade_pairs: list[list[str]] = []
    original_runtime_config = runner._runtime_config

    def traced_runtime_config(config, *, pairs, owner_id):
        observed_trade_pairs.append(list(pairs))
        return original_runtime_config(config, pairs=pairs, owner_id=owner_id)

    monkeypatch.setattr(runner, "_runtime_config", traced_runtime_config)
    output = tmp_path / "out"
    assert runner._run_command(_run_args(sealed_path, corpus, output)) == 0

    assert len(calls) == 12
    assert all(
        _option(command, "--pairs").split(",") == FEED_PAIRS for command in calls
    )
    assert observed_trade_pairs.count(PAIRS) == 4
    assert observed_trade_pairs.count(["EUR_USD"]) == 4
    assert observed_trade_pairs.count(["GBP_USD"]) == 4


def test_preflight_rejects_partial_pair_epoch_coverage_before_replay(
    tmp_path, monkeypatch
) -> None:
    repo = _repo(tmp_path, monkeypatch)
    corpus = _corpus(tmp_path)
    sealed = _sealed(repo, corpus)
    sealed_path = _write_sealed(tmp_path, sealed)
    _write_pair_rows(corpus, "USD_JPY", (0, 2))
    calls: list = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: calls.append(a))

    output = tmp_path / "out"
    assert runner._run_command(_run_args(sealed_path, corpus, output)) == 2
    failure = json.loads((output / "run_failure.json").read_text())
    assert "partial-pair epoch coverage" in failure["error"]
    assert calls == []


def test_corpus_rejects_single_shared_row_and_admits_simultaneous_gap(
    tmp_path,
) -> None:
    corpus = _corpus(tmp_path)
    for pair in FEED_PAIRS:
        _write_pair_rows(corpus, pair, (0,))
    with pytest.raises(runner.TrainerRunnerError, match="at least two rows"):
        runner._corpus_manifest(
            corpus,
            FEED_PAIRS,
            "2025-03-01T00:00:00Z",
            "2025-04-01T00:00:00Z",
        )

    for pair in FEED_PAIRS:
        _write_pair_rows(corpus, pair, (0, 2))
    manifest = runner._corpus_manifest(
        corpus,
        FEED_PAIRS,
        "2025-03-01T00:00:00Z",
        "2025-04-01T00:00:00Z",
    )
    coverage = manifest["synchronized_m1_coverage"]
    assert coverage["row_count_per_pair"] == 2
    assert coverage["simultaneous_missing_minutes_between_rows"] == 1
    assert coverage["partial_pair_missing_epochs"] == 0


def test_lopo_failure_is_explicit_and_never_uses_additive_substitute(
    tmp_path, monkeypatch
) -> None:
    repo = _repo(tmp_path, monkeypatch)
    corpus = _corpus(tmp_path)
    sealed = _sealed(repo, corpus)
    sealed_path = _write_sealed(tmp_path, sealed)
    calls = _install_successful_replay_mocks(monkeypatch, sealed)
    successful_fake = runner.subprocess.run
    failed = False

    def fail_one_lopo(command, **kwargs):
        nonlocal failed
        trade_pairs = json.loads(kwargs["env"]["DOJO_BOT_CONFIG"])["pairs"]
        if len(trade_pairs) == 1 and not failed:
            failed = True
            calls.append(list(command))
            return subprocess.CompletedProcess(command, 9, stdout="", stderr="boom")
        return successful_fake(command, **kwargs)

    monkeypatch.setattr(runner.subprocess, "run", fail_one_lopo)
    output = tmp_path / "out"
    assert runner._run_command(_run_args(sealed_path, corpus, output)) == 2
    receipt = json.loads((output / "run.json").read_text())
    cells = json.loads((output / "cells.json").read_text())
    evaluation = json.loads((output / "evaluation.json").read_text())

    failed_rows = [
        row
        for row in receipt["coordinates"]
        if row["status"] == "LOPO_INCOMPLETE_NO_ADDITIVE_SUBSTITUTE"
    ]
    assert len(failed_rows) == 1
    assert receipt["fixed_denominator"]["coordinate_receipts_complete"] is True
    assert receipt["fixed_denominator"]["execution_success_complete"] is False
    failed_lopo = [
        row for row in failed_rows[0]["lopo"] if row["status"].startswith("FAILED")
    ]
    assert len(failed_lopo) == 1
    assert failed_lopo[0]["terminal_net_jpy"] is None
    failed_cell = next(
        cell
        for cell in cells
        if cell["candidate_id"] == failed_rows[0]["candidate_id"]
        and cell["intrabar"] == failed_rows[0]["intrabar"]
        and cell["cost_arm"] == failed_rows[0]["cost_arm"]
    )
    assert failed_cell["metrics"]["lopo_replay_complete"] is False
    assert failed_cell["execution_status"] == "FAILED"
    assert failed_cell["failure_code"] == "COUNTERFACTUAL_LOPO_INCOMPLETE"
    assert 0.0 in failed_cell["metrics"]["leave_one_pair_out_net_jpy"].values()
    candidate = evaluation["candidate_evaluations"][0]
    assert "RUNNER_CELL_FAILURE" in candidate["gate_blockers"]
    assert "COUNTERFACTUAL_LOPO_INCOMPLETE" in candidate["gate_blockers"]
    assert candidate["diagnostic_rank_eligible"] is False


def test_seal_rejects_repo_path_escape(tmp_path, monkeypatch) -> None:
    _repo(tmp_path, monkeypatch)
    try:
        runner._source_digests(["../outside.py"])
    except runner.TrainerRunnerError as exc:
        assert "repo-relative" in str(exc)
    else:
        raise AssertionError("path escape unexpectedly accepted")
