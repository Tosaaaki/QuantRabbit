from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


REPO = Path(__file__).resolve().parents[1]


def load_cli() -> ModuleType:
    path = REPO / "scripts/run-dojo-ai-forward.py"
    spec = importlib.util.spec_from_file_location("run_dojo_ai_forward", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
