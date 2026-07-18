from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from quant_rabbit.dojo_materialization import (
    MaterializationError,
    build_materialization_receipt,
    load_materialization_receipt,
    publish_receipt_exclusive,
    verify_materialized_archive,
    verify_materialization_receipt,
)


def _trees(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source"
    materialized = tmp_path / "materialized"
    external = tmp_path / "external" / "task.output"
    (source / "scratchpad").mkdir(parents=True)
    (source / "tasks").mkdir()
    (materialized / "scratchpad").mkdir(parents=True)
    (materialized / "tasks").mkdir()
    external.parent.mkdir()
    external.write_bytes(b"external task bytes\n")
    regular_bytes = b"regular packet bytes\n"
    (source / "scratchpad/packet.json").write_bytes(regular_bytes)
    (materialized / "scratchpad/packet.json").write_bytes(regular_bytes)
    os.symlink(external.resolve(), source / "tasks/task.output")
    (materialized / "tasks/task.output").write_bytes(external.read_bytes())
    return source, materialized, external


def test_builds_and_verifies_regular_only_materialization(tmp_path: Path) -> None:
    source, materialized, external = _trees(tmp_path)

    receipt = build_materialization_receipt(
        source_root=source, materialized_root=materialized
    )

    assert receipt["contract"] == "QR_DOJO_MATERIALIZATION_RECEIPT_V2"
    assert receipt["file_count"] == 2
    assert receipt["source_regular_count"] == 1
    assert receipt["source_symlink_count"] == 1
    link_row = next(
        row for row in receipt["files"] if row["path"] == "tasks/task.output"
    )
    assert link_row["source_entry_type"] == "SYMLINK"
    assert link_row["source_link_target"] == str(external.resolve())
    assert link_row["source_resolved_target_sha256"] == link_row["materialized_sha256"]
    assert receipt["live_permission"] is False

    path = tmp_path / "receipt.json"
    publish_receipt_exclusive(path, receipt)
    assert path.stat().st_mode & 0o777 == 0o600
    assert load_materialization_receipt(path) == receipt
    assert (
        verify_materialization_receipt(
            receipt_path=path,
            source_root=source,
            materialized_root=materialized,
        )
        == receipt
    )


def test_publish_never_overwrites_existing_receipt(tmp_path: Path) -> None:
    source, materialized, _ = _trees(tmp_path)
    receipt = build_materialization_receipt(
        source_root=source, materialized_root=materialized
    )
    path = tmp_path / "receipt.json"
    publish_receipt_exclusive(path, receipt)
    original = path.read_bytes()

    with pytest.raises(MaterializationError, match="already exists"):
        publish_receipt_exclusive(path, receipt)

    assert path.read_bytes() == original


def test_durable_archive_verifies_after_source_targets_are_retired(
    tmp_path: Path,
) -> None:
    source, materialized, external = _trees(tmp_path)
    receipt = build_materialization_receipt(
        source_root=source, materialized_root=materialized
    )
    path = tmp_path / "receipt.json"
    publish_receipt_exclusive(path, receipt)
    external.unlink()

    assert (
        verify_materialized_archive(
            receipt_path=path,
            materialized_root=materialized,
        )
        == receipt
    )
    with pytest.raises(MaterializationError, match="broken or cyclic"):
        verify_materialization_receipt(
            receipt_path=path,
            source_root=source,
            materialized_root=materialized,
        )


def test_rejects_materialized_byte_drift(tmp_path: Path) -> None:
    source, materialized, _ = _trees(tmp_path)
    receipt = build_materialization_receipt(
        source_root=source, materialized_root=materialized
    )
    path = tmp_path / "receipt.json"
    publish_receipt_exclusive(path, receipt)
    (materialized / "tasks/task.output").write_bytes(b"changed\n")

    with pytest.raises(MaterializationError, match="bytes differ"):
        verify_materialization_receipt(
            receipt_path=path,
            source_root=source,
            materialized_root=materialized,
        )


def test_rejects_broken_source_symlink(tmp_path: Path) -> None:
    source, materialized, external = _trees(tmp_path)
    external.unlink()

    with pytest.raises(MaterializationError, match="broken or cyclic"):
        build_materialization_receipt(
            source_root=source, materialized_root=materialized
        )


def test_rejects_symlink_in_materialized_tree(tmp_path: Path) -> None:
    source, materialized, external = _trees(tmp_path)
    materialized_link = materialized / "tasks/task.output"
    materialized_link.unlink()
    os.symlink(external.resolve(), materialized_link)

    with pytest.raises(
        MaterializationError, match="materialized tree contains a symlink"
    ):
        build_materialization_receipt(
            source_root=source, materialized_root=materialized
        )


def test_rejects_path_set_drift(tmp_path: Path) -> None:
    source, materialized, _ = _trees(tmp_path)
    (materialized / "extra.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(MaterializationError, match="path set mismatch"):
        build_materialization_receipt(
            source_root=source, materialized_root=materialized
        )


def test_rejects_tampered_receipt(tmp_path: Path) -> None:
    source, materialized, _ = _trees(tmp_path)
    receipt = build_materialization_receipt(
        source_root=source, materialized_root=materialized
    )
    receipt["file_count"] = 3
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(MaterializationError, match="canonical SHA-256"):
        load_materialization_receipt(path)
