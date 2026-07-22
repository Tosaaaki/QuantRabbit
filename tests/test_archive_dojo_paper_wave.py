from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from quant_rabbit.dojo_paper_contract import canonical_sha256, file_sha256
from quant_rabbit.virtual_broker import VirtualBroker


ROOT = Path(__file__).resolve().parents[1]
UTC = timezone.utc


def _load_archiver():
    path = ROOT / "scripts/archive-dojo-paper-wave.py"
    spec = importlib.util.spec_from_file_location("dojo_paper_wave_archiver_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_archive_ledger_verifier_recomputes_the_full_hash_chain(tmp_path):
    archiver = _load_archiver()
    broker = VirtualBroker(tmp_path / "ledger.jsonl", balance_jpy=200_000.0)
    broker.on_quote(
        "USD_JPY", 162.00, 162.01, datetime.now(UTC).isoformat()
    )
    broker._log("SESSION_STOP", {"proof_eligible": True})

    records, terminal = archiver.verify_ledger(broker.ledger_path)
    assert records[-1]["event"] == "SESSION_STOP"
    assert terminal == records[-1]["sha"]

    rows = broker.ledger_path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(rows[0])
    tampered["payload"]["ask"] = 999
    rows[0] = json.dumps(tampered)
    broker.ledger_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(archiver.PaperWaveArchiveError, match="ledger sha mismatch"):
        archiver.verify_ledger(broker.ledger_path)


def _manifest(tmp_path):
    registry = tmp_path / "registry.json"
    registry.write_text('{"registry":"sealed"}\n', encoding="utf-8")
    evidence = tmp_path / "ledger.jsonl"
    evidence.write_text('{"evidence":"sealed"}\n', encoding="utf-8")
    body = {
        "contract": "QR_DOJO_PAPER_WAVE_ARCHIVE_V1",
        "schema_version": 1,
        "experiment_id": "paper-wave-test",
        "registry_path": str(registry),
        "registry_sha256": file_sha256(registry),
        "window": {"start_utc": "a", "end_utc": "b"},
        "room_count": 1,
        "source_size_bytes": evidence.stat().st_size,
        "rooms": [
            {
                "room_id": "room-01",
                "candidate_id": "candidate",
                "arm": "BASE",
                "session_contract_sha256": "1" * 64,
                "terminal_ledger_sha256": "2" * 64,
                "terminal_event": "SESSION_STOP",
                "artifacts": [
                    {
                        "archive_path": "rooms/room-01/ledger.jsonl",
                        "source_path": str(evidence),
                        "size_bytes": evidence.stat().st_size,
                        "sha256": file_sha256(evidence),
                    }
                ],
            }
        ],
        "source_deleted": False,
        "remote_verified": False,
        "proof_claim": "SEALED_FORWARD_PAPER_SOURCE_ARCHIVE_ONLY",
        "live_permission": False,
        "order_authority": "NONE",
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def test_drive_mount_archive_is_deterministic_idempotent_and_not_remote_proof(
    tmp_path, monkeypatch
):
    archiver = _load_archiver()
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(archiver, "build_archive_manifest", lambda _: manifest)
    drive = tmp_path / "drive"

    first_archive, first_receipt, first = archiver.stage_archive(
        registry_path=tmp_path / "registry.json", drive_root=drive
    )
    second_archive, second_receipt, second = archiver.stage_archive(
        registry_path=tmp_path / "registry.json", drive_root=drive
    )

    assert first_archive == second_archive
    assert first_receipt == second_receipt
    assert first == second
    assert first["drive_mount_staged"] is True
    assert first["remote_verified"] is False
    assert first["source_deleted"] is False
    assert first["local_payload_verified"] is True
    with tarfile.open(first_archive, "r:gz") as archive:
        assert archive.getnames() == [
            "manifest.json",
            "registry.json",
            "rooms/room-01/ledger.jsonl",
        ]

    first_archive.write_bytes(b"conflicting archive")
    with pytest.raises(archiver.PaperWaveArchiveError, match="conflicts"):
        archiver.stage_archive(
            registry_path=tmp_path / "registry.json", drive_root=drive
        )
