#!/usr/bin/env python3
"""Stage a sealed DOJO paper wave as a deterministic Google Drive archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from quant_rabbit.dojo_paper_contract import (
    canonical_bytes,
    canonical_sha256,
    file_sha256,
    publish_immutable_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_CONTRACT = "QR_DOJO_PAPER_WAVE_ARCHIVE_V1"
RECEIPT_CONTRACT = "QR_DOJO_PAPER_WAVE_DRIVE_MOUNT_RECEIPT_V1"
DEFAULT_DRIVE_ROOT = Path(
    "/Users/tossaki/Library/CloudStorage/"
    "GoogleDrive-www.tosakiweb.net@gmail.com/マイドライブ/"
    "QuantRabbit DOJO Archives/completed-runs"
)
MAX_SOURCE_BYTES = 1024 * 1024 * 1024


class PaperWaveArchiveError(ValueError):
    pass


def _room_launcher_module():
    path = REPO_ROOT / "scripts/run-dojo-paper-room.py"
    spec = importlib.util.spec_from_file_location("dojo_paper_room_launcher", path)
    if spec is None or spec.loader is None:
        raise PaperWaveArchiveError(f"cannot load room launcher: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ledger_sha_body(record: dict[str, Any]) -> str:
    try:
        body = {
            key: record[key] for key in ("ts_utc", "event", "payload", "prev_sha")
        }
    except KeyError as exc:
        raise PaperWaveArchiveError("ledger record schema mismatch") from exc
    return hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def verify_ledger(path: Path) -> tuple[list[dict[str, Any]], str]:
    if not path.is_file():
        raise PaperWaveArchiveError(f"missing ledger: {path}")
    records = []
    previous = "0" * 64
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PaperWaveArchiveError(
                f"invalid ledger JSON at line {line_number}"
            ) from exc
        if record.get("prev_sha") != previous:
            raise PaperWaveArchiveError(
                f"ledger prev_sha mismatch at line {line_number}"
            )
        if record.get("sha") != _ledger_sha_body(record):
            raise PaperWaveArchiveError(
                f"ledger sha mismatch at line {line_number}"
            )
        previous = record["sha"]
        records.append(record)
    if not records:
        raise PaperWaveArchiveError("paper room ledger is empty")
    return records, previous


def _verify_content_hash(payload: dict[str, Any], digest_field: str) -> None:
    digest = payload.get(digest_field)
    body = {key: value for key, value in payload.items() if key != digest_field}
    if not isinstance(digest, str) or digest != canonical_sha256(body):
        raise PaperWaveArchiveError(f"{digest_field} mismatch")


def verify_room(
    *, registry_path: Path, registry: dict[str, Any], room: dict[str, Any]
) -> dict[str, Any]:
    launcher = _room_launcher_module()
    room_id = room["room_id"]
    _, _, session_dir = launcher.build_launch(
        registry_path=registry_path,
        room_id=room_id,
        python_executable="python3",
    )
    contract_path = session_dir / "session_contract.json"
    snapshot_path = session_dir / "broker_snapshot.json"
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PaperWaveArchiveError(f"room evidence is incomplete: {room_id}") from exc
    _verify_content_hash(contract, "session_contract_sha256")
    if contract.get("room_id") != room_id:
        raise PaperWaveArchiveError(f"session room id mismatch: {room_id}")
    if contract.get("experiment_id") != registry.get("experiment_id"):
        raise PaperWaveArchiveError(f"session experiment mismatch: {room_id}")
    if contract.get("candidate_id") != room.get("candidate_id"):
        raise PaperWaveArchiveError(f"session candidate mismatch: {room_id}")
    if contract.get("proof_mode") != "formal" or not contract.get("proof_eligible"):
        raise PaperWaveArchiveError(f"room is not formal proof-eligible: {room_id}")

    records, terminal_sha = verify_ledger(session_dir / "ledger.jsonl")
    if snapshot.get("ledger_sha") != terminal_sha:
        raise PaperWaveArchiveError(f"snapshot/ledger mismatch: {room_id}")
    if snapshot.get("positions") or snapshot.get("orders"):
        raise PaperWaveArchiveError(f"room has unresolved exposure: {room_id}")
    latest = records[-1]
    if latest.get("event") == "DRAIN_STOP":
        if latest.get("payload", {}).get("status") != "SEALED":
            raise PaperWaveArchiveError(f"room drain is not sealed: {room_id}")
    elif latest.get("event") != "SESSION_STOP":
        raise PaperWaveArchiveError(f"room terminal event is not sealed: {room_id}")

    source_paths = [
        contract_path,
        session_dir / "ledger.jsonl",
        snapshot_path,
    ]
    drain_path = session_dir / "drain_contract.json"
    if drain_path.exists():
        drain = json.loads(drain_path.read_text(encoding="utf-8"))
        _verify_content_hash(drain, "drain_contract_sha256")
        source_paths.append(drain_path)
    artifacts = [
        {
            "archive_path": f"rooms/{room_id}/{path.name}",
            "source_path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in source_paths
    ]
    return {
        "room_id": room_id,
        "candidate_id": room["candidate_id"],
        "arm": room["arm"],
        "session_contract_sha256": contract["session_contract_sha256"],
        "terminal_ledger_sha256": terminal_sha,
        "terminal_event": latest["event"],
        "artifacts": artifacts,
    }


def build_archive_manifest(registry_path: Path) -> dict[str, Any]:
    registry_path = registry_path.resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("contract") != "QR_DOJO_PAPER_ROOM_REGISTRY_V1":
        raise PaperWaveArchiveError("unsupported paper room registry")
    rooms = [
        verify_room(registry_path=registry_path, registry=registry, room=room)
        for room in registry.get("rooms") or []
    ]
    if len(rooms) != 4 or len({room["room_id"] for room in rooms}) != 4:
        raise PaperWaveArchiveError("paper wave must contain exact four rooms")
    total = sum(
        artifact["size_bytes"]
        for room in rooms
        for artifact in room["artifacts"]
    )
    if total > MAX_SOURCE_BYTES:
        raise PaperWaveArchiveError("paper wave source exceeds archive byte budget")
    body = {
        "contract": ARCHIVE_CONTRACT,
        "schema_version": 1,
        "experiment_id": registry["experiment_id"],
        "registry_path": str(registry_path),
        "registry_sha256": file_sha256(registry_path),
        "window": registry["window"],
        "room_count": len(rooms),
        "source_size_bytes": total,
        "rooms": rooms,
        "source_deleted": False,
        "remote_verified": False,
        "proof_claim": "SEALED_FORWARD_PAPER_SOURCE_ARCHIVE_ONLY",
        "live_permission": False,
        "order_authority": "NONE",
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def _add_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    tar.addfile(info, io.BytesIO(payload))


def _write_deterministic_archive(path: Path, manifest: dict[str, Any]) -> None:
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                _add_bytes(
                    archive,
                    "manifest.json",
                    canonical_bytes(manifest) + b"\n",
                )
                registry_path = Path(manifest["registry_path"])
                _add_bytes(
                    archive,
                    "registry.json",
                    registry_path.read_bytes(),
                )
                artifacts = sorted(
                    (
                        artifact
                        for room in manifest["rooms"]
                        for artifact in room["artifacts"]
                    ),
                    key=lambda row: row["archive_path"],
                )
                for artifact in artifacts:
                    source = Path(artifact["source_path"])
                    payload = source.read_bytes()
                    if hashlib.sha256(payload).hexdigest() != artifact["sha256"]:
                        raise PaperWaveArchiveError(
                            f"source changed during archive: {source}"
                        )
                    _add_bytes(archive, artifact["archive_path"], payload)
        raw.flush()
        os.fsync(raw.fileno())


def stage_archive(
    *, registry_path: Path, drive_root: Path
) -> tuple[Path, Path, dict[str, Any]]:
    manifest = build_archive_manifest(registry_path)
    destination = drive_root / manifest["experiment_id"]
    destination.mkdir(parents=True, exist_ok=True)
    archive_name = f"paper-wave-{manifest['manifest_sha256']}.tar.gz"
    archive_path = destination / archive_name
    descriptor, tmp_name = tempfile.mkstemp(
        prefix=f".{archive_name}.", suffix=".tmp", dir=destination
    )
    os.close(descriptor)
    Path(tmp_name).unlink()
    tmp_path = Path(tmp_name)
    try:
        _write_deterministic_archive(tmp_path, manifest)
        archive_sha = file_sha256(tmp_path)
        try:
            os.link(tmp_path, archive_path)
        except FileExistsError:
            if file_sha256(archive_path) != archive_sha:
                raise PaperWaveArchiveError("existing Drive archive conflicts")
        receipt_body = {
            "contract": RECEIPT_CONTRACT,
            "schema_version": 1,
            "experiment_id": manifest["experiment_id"],
            "manifest_sha256": manifest["manifest_sha256"],
            "archive_name": archive_name,
            "archive_path": str(archive_path),
            "archive_sha256": archive_sha,
            "archive_size_bytes": archive_path.stat().st_size,
            "local_payload_verified": file_sha256(archive_path) == archive_sha,
            "drive_mount_staged": True,
            "remote_verified": False,
            "source_deleted": False,
            "live_permission": False,
            "order_authority": "NONE",
        }
        receipt = {
            **receipt_body,
            "receipt_sha256": canonical_sha256(receipt_body),
        }
        receipt_path = destination / f"receipt-{receipt['receipt_sha256']}.json"
        publish_immutable_json(receipt_path, receipt)
        return archive_path, receipt_path, receipt
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=REPO_ROOT / "config/dojo_paper_rooms_v1.json",
    )
    parser.add_argument("--drive-root", type=Path, default=DEFAULT_DRIVE_ROOT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = build_archive_manifest(args.registry)
    if args.verify_only:
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    archive, receipt, payload = stage_archive(
        registry_path=args.registry, drive_root=args.drive_root
    )
    print(
        json.dumps(
            {
                "status": "DRIVE_MOUNT_STAGED_NOT_REMOTE_VERIFIED",
                "archive": str(archive),
                "receipt": str(receipt),
                "archive_sha256": payload["archive_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
