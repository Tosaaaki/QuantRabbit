from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from quant_rabbit.dojo_ai_discretion import canonical_sha256


REPO = Path(__file__).resolve().parents[1]


def load_cli() -> ModuleType:
    path = REPO / "scripts/run-dojo-ai-forward.py"
    spec = importlib.util.spec_from_file_location("run_dojo_ai_forward", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def superseded_run(tmp_path: Path) -> tuple[ModuleType, Path, Path]:
    cli = load_cli()
    registry_path = REPO / "research/registries/dojo_prompt_experiment_v1.json"
    registry = cli._strict_json(registry_path)
    prompt_texts = {
        row["variant_id"]: (REPO / row["prompt_path"]).read_text(encoding="utf-8")
        for row in registry["variants"]
    }
    precommit = cli.build_precommit(
        registry,
        prompt_texts,
        {
            "first_cutoff_utc": "2026-07-22T15:00:00Z",
            "allocation_nonce": "a" * 64,
            "model_policy": {
                "model_name": "gpt-5.5",
                "model_version": "test-snapshot",
                "model_lineage": "test-lineage",
                "reasoning_effort": "high",
            },
            "source_bindings": {
                "git_commit": "b" * 40,
                "files": {"test-only": "c" * 64},
            },
        },
        now_utc=datetime(2026, 7, 19, tzinfo=timezone.utc),
    )
    start = cli.build_start_receipt(
        precommit,
        now_utc=datetime(2026, 7, 19, 0, 1, tzinfo=timezone.utc),
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cli._write_json_new_or_same(run_dir / "precommit.json", precommit, root=run_dir)
    cli._write_json_new_or_same(run_dir / "start.json", start, root=run_dir)
    cli.initialize_registry(
        run_dir,
        precommit_path=run_dir / "precommit.json",
        start_path=run_dir / "start.json",
        created_at_utc=datetime(2026, 7, 19, 0, 1, tzinfo=timezone.utc),
    )
    body = {
        "allocated_cell_count": 90,
        "authority": {
            "broker_mutation_allowed": False,
            "live_permission": False,
            "order_authority": "NONE",
            "promotion_eligible": False,
        },
        "contract": "QR_DOJO_AI_FORWARD_SUPERSESSION_V1",
        "evidence": [
            {
                "artifact_sha256": "d" * 64,
                "path": "/archive/chart-only-null.json",
                "result": "STRUCTURAL_CHART_READING_ALONE_HAS_NO_EDGE",
            }
        ],
        "executed_cell_count": 0,
        "market_source_acquisition_count": 0,
        "precommit_sha256": precommit["precommit_sha256"],
        "reason_code": "REDUNDANT_CANDLE_ONLY_INPUT_CLASS",
        "schema_version": 1,
        "start_receipt_sha256": start["start_receipt_sha256"],
        "state": "SUPERSEDED_BEFORE_SOURCE",
        "successor_policy": {
            "entry_full_context_requires_causal_multisource_contract": True,
            "next_experiment_class": "AI_EXIT_JUDGMENT",
            "one_judgment_per_fresh_context": True,
        },
        "superseded_at_utc": "2026-07-19T00:02:00Z",
    }
    supersession = {
        **body,
        "supersession_sha256": canonical_sha256(body),
    }
    cli._write_json_new_or_same(
        run_dir / "supersession.json", supersession, root=run_dir
    )
    cli.append_artifact_commit(
        run_dir,
        logical_id="run/supersession",
        artifact_path=run_dir / "supersession.json",
        parent_logical_ids=("precommit", "start"),
        committed_at_utc=datetime(2026, 7, 19, 0, 2, tzinfo=timezone.utc),
    )
    cli.append_invalidation(
        run_dir,
        logical_id="start",
        reason_code=supersession["reason_code"],
        evidence_sha256=supersession["supersession_sha256"],
        invalidated_at_utc=datetime(2026, 7, 19, 0, 2, tzinfo=timezone.utc),
    )
    return cli, run_dir, registry_path


def test_atomic_writer_rejects_symlink_parent_escape(tmp_path: Path) -> None:
    cli = load_cli()
    run_dir = tmp_path / "run"
    outside = tmp_path / "outside"
    run_dir.mkdir()
    outside.mkdir()
    (run_dir / "responses").symlink_to(outside, target_is_directory=True)
    target = run_dir / "responses/day-001/cell-a.json"
    with pytest.raises(RuntimeError, match="parent is unsafe"):
        cli._write_json_new_or_same(target, {"safe": True}, root=run_dir)
    assert not (outside / "day-001/cell-a.json").exists()


def test_atomic_writer_recovers_orphan_pending_file(tmp_path: Path) -> None:
    cli = load_cli()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    target = run_dir / "days/day-001.json"
    target.parent.mkdir()
    orphan = target.parent / f".{target.name}.pending-crashed"
    orphan.write_bytes(b"partial")
    cli._write_json_new_or_same(target, {"ordinal": 1}, root=run_dir)
    assert target.is_file()
    assert not orphan.exists()
    first = target.read_bytes()
    cli._write_json_new_or_same(target, {"ordinal": 1}, root=run_dir)
    assert target.read_bytes() == first


def test_strict_reader_rejects_direct_and_ancestor_symlinks(tmp_path: Path) -> None:
    cli = load_cli()
    outside = tmp_path / "outside"
    outside.mkdir()
    artifact = outside / "artifact.json"
    artifact.write_text("{}")
    direct = tmp_path / "direct.json"
    direct.symlink_to(artifact)
    with pytest.raises(RuntimeError, match="contains a symlink"):
        cli._strict_json(direct)
    parent = tmp_path / "linked-parent"
    parent.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="contains a symlink"):
        cli._strict_json(parent / "artifact.json")


def test_status_reports_valid_terminal_supersession(tmp_path: Path) -> None:
    cli, run_dir, registry_path = superseded_run(tmp_path)

    status = cli._status(SimpleNamespace(run_dir=run_dir, registry=registry_path))

    assert status["state"] == "SUPERSEDED_BEFORE_SOURCE"
    assert status["sealed_day_count"] == 0
    assert status["fixed_cell_count"] == 0
    assert status["next_ordinal"] is None
    assert status["promotion_eligible"] is False
    assert status["live_permission"] is False
    assert status["validity_registry"]["invalidated_logical_ids"] == [
        "run/supersession",
        "start",
    ]


def test_supersession_rejects_boolean_number_type_confusion(tmp_path: Path) -> None:
    cli, run_dir, _ = superseded_run(tmp_path)
    precommit = cli._strict_json(run_dir / "precommit.json")
    start = cli._strict_json(run_dir / "start.json")
    supersession = cli._strict_json(run_dir / "supersession.json")
    supersession["authority"]["live_permission"] = 0
    body = {
        key: value
        for key, value in supersession.items()
        if key != "supersession_sha256"
    }
    supersession["supersession_sha256"] = canonical_sha256(body)

    with pytest.raises(cli.DojoAIForwardError, match="identity drifted"):
        cli.validate_supersession(supersession, precommit, start)


def test_superseded_status_rejects_uncommitted_day_artifact(tmp_path: Path) -> None:
    cli, run_dir, registry_path = superseded_run(tmp_path)
    foreign = run_dir / "days" / "day-001.json"
    foreign.parent.mkdir()
    foreign.write_text('{"state":"UNCOMMITTED"}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="unexpected artifact"):
        cli._status(SimpleNamespace(run_dir=run_dir, registry=registry_path))


def test_superseded_run_rejects_collection_and_mutation_commands(
    tmp_path: Path,
) -> None:
    cli, run_dir, _ = superseded_run(tmp_path)
    with pytest.raises(RuntimeError, match="terminally superseded"):
        cli._load_parents(run_dir)

    with pytest.raises(RuntimeError, match="terminally superseded"):
        cli._start(SimpleNamespace(run_dir=run_dir))
