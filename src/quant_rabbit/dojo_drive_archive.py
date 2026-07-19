"""Fail-closed local archive staging for terminal DOJO trainer runs.

This module does not call Google Drive.  It creates deterministic ``tar.zst``
chunks plus local receipts which an external uploader can bind to Drive
metadata later.  A local finalization receipt always states that remote
verification has not happened.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tarfile
import fcntl
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Final


PLAN_CONTRACT: Final = "QR_DOJO_DRIVE_ARCHIVE_PLAN_V1"
FINALIZATION_CONTRACT: Final = "QR_DOJO_DRIVE_ARCHIVE_FINALIZATION_V1"
RUN_CONTRACT: Final = "QR_DOJO_BOT_TRAINER_RUN_V1"
EVALUATION_CONTRACT: Final = "QR_DOJO_BOT_TRAINER_EVALUATION_V1"
CELL_CONTRACT: Final = "QR_DOJO_BOT_TRAINER_CELL_V1"
MAX_JSON_BYTES: Final = 16 * 1024 * 1024
HASH_CHUNK_BYTES: Final = 1024 * 1024
MAX_FILES: Final = 1_000_000
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_CHUNK_ID_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._|=+-]{0,199}\Z")
_MONTH_ID_RE: Final = re.compile(r"\d{4}-(?:0[1-9]|1[0-2])\Z")
_PLAN_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "source_run_root",
        "destination_root",
        "chunk_kind",
        "chunk_id",
        "terminal_run",
        "file_count",
        "total_source_bytes",
        "content_tree_sha256",
        "files",
        "archive_format",
        "archive_member_prefix",
        "source_deletion_allowed",
        "source_deleted",
        "remote_verification",
        "proof_eligible",
        "promotion_eligible",
        "live_permission",
        "order_authority",
        "broker_mutation_allowed",
        "plan_sha256",
    }
)
_PLAN_TERMINAL_KEYS: Final = frozenset(
    {
        "contract",
        "status",
        "run_sha256",
        "study_sha256",
        "evaluation_sha256",
        "classification",
        "fixed_denominator",
    }
)
_PLAN_FILE_KEYS: Final = frozenset({"path", "size_bytes", "sha256"})
_REMOTE_KEYS: Final = frozenset(
    {"status", "remote_verified", "metadata_receipt_sha256"}
)
_CELL_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "study_sha256",
        "candidate_id",
        "proposal_sha256",
        "intrabar",
        "cost_arm",
        "execution_status",
        "failure_code",
        "metrics",
        "ledger_evidence",
        "cell_sha256",
    }
)
_COORDINATE_KEYS: Final = frozenset(
    {
        "candidate_id",
        "intrabar",
        "cost_arm",
        "status",
        "main_session_dir",
        "main_error",
        "lopo_replay_complete",
        "lopo",
        "cell_sha256",
    }
)
_RUN_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "study_sha256",
        "status",
        "corpus",
        "fixed_denominator",
        "coordinates",
        "cells_path",
        "evaluation_path",
        "evaluation_sha256",
        "classification",
        "proof_eligible",
        "promotion_eligible",
        "live_permission",
        "order_authority",
        "broker_mutation_allowed",
        "run_sha256",
    }
)
_FINALIZATION_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "plan_path",
        "plan_sha256",
        "content_tree_sha256",
        "chunk_kind",
        "chunk_id",
        "archive_path",
        "archive_sha256",
        "archive_size_bytes",
        "file_count",
        "total_source_bytes",
        "local_payload_verified",
        "atomic_publish_complete",
        "source_deletion_allowed",
        "source_deleted",
        "remote_verification",
        "proof_eligible",
        "promotion_eligible",
        "live_permission",
        "order_authority",
        "broker_mutation_allowed",
        "finalization_sha256",
    }
)


class DojoDriveArchiveError(ValueError):
    """Raised when an archive cannot be built without weakening provenance."""


def canonical_bytes(value: Any) -> bytes:
    """Return strict canonical JSON bytes without a trailing newline."""

    _validate_json(value, field="value")
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DojoDriveArchiveError(f"value is not canonical JSON: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _validate_json(value: Any, *, field: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DojoDriveArchiveError(f"{field} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise DojoDriveArchiveError(f"{field} contains a non-string key")
            _validate_json(item, field=f"{field}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _validate_json(item, field=f"{field}[{index}]")
        return
    raise DojoDriveArchiveError(f"{field} contains a non-JSON value")


def _reject_constant(value: str) -> None:
    raise DojoDriveArchiveError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DojoDriveArchiveError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _require_exact_keys(
    value: Any, *, expected: frozenset[str], field: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DojoDriveArchiveError(f"{field} must be a JSON object")
    actual = set(value)
    if actual != expected:
        raise DojoDriveArchiveError(
            f"{field} schema mismatch: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _stable_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _real_directory(path: Path | str, *, label: str) -> Path:
    raw = Path(path).expanduser()
    try:
        before = raw.lstat()
    except OSError as exc:
        raise DojoDriveArchiveError(f"{label} is unavailable: {raw}: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise DojoDriveArchiveError(f"{label} must be one real directory: {raw}")
    try:
        resolved = raw.resolve(strict=True)
        after = raw.lstat()
    except (OSError, RuntimeError) as exc:
        raise DojoDriveArchiveError(f"cannot resolve {label}: {raw}: {exc}") from exc
    if _stable_identity(before) != _stable_identity(after):
        raise DojoDriveArchiveError(f"{label} changed while resolving: {raw}")
    return resolved


def _assert_real_tree_path(
    root: Path, relative: str, *, expect_directory: bool | None = None
) -> Path:
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise DojoDriveArchiveError("tree path is not canonical and relative")
    current = root
    for index, part in enumerate(pure.parts):
        current = current / part
        try:
            state = current.lstat()
        except OSError as exc:
            raise DojoDriveArchiveError(
                f"tree path is unavailable: {pure.as_posix()}"
            ) from exc
        if stat.S_ISLNK(state.st_mode):
            raise DojoDriveArchiveError(
                f"tree path contains an intermediate symlink: {pure.as_posix()}"
            )
        final = index == len(pure.parts) - 1
        if not final and not stat.S_ISDIR(state.st_mode):
            raise DojoDriveArchiveError(
                f"tree path ancestor is not a directory: {pure.as_posix()}"
            )
        if final and expect_directory is True and not stat.S_ISDIR(state.st_mode):
            raise DojoDriveArchiveError(
                f"tree path is not a directory: {pure.as_posix()}"
            )
        if final and expect_directory is False and not stat.S_ISREG(state.st_mode):
            raise DojoDriveArchiveError(
                f"tree path is not a regular file: {pure.as_posix()}"
            )
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise DojoDriveArchiveError(
            f"tree path escapes the source root: {pure.as_posix()}"
        ) from exc
    return current


def _ensure_destination(path: Path | str) -> Path:
    raw = Path(path).expanduser().absolute()
    missing: list[Path] = []
    cursor = raw
    while not cursor.exists():
        missing.append(cursor)
        if cursor.parent == cursor:
            raise DojoDriveArchiveError("destination has no existing parent")
        cursor = cursor.parent
    if cursor.is_symlink() or not cursor.is_dir():
        raise DojoDriveArchiveError("destination ancestor must be a real directory")
    for directory in reversed(missing):
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            pass
        state = directory.lstat()
        if stat.S_ISLNK(state.st_mode) or not stat.S_ISDIR(state.st_mode):
            raise DojoDriveArchiveError(
                f"destination component is not a real directory: {directory}"
            )
    return _real_directory(raw, label="destination root")


def _assert_separate_roots(source: Path, destination: Path) -> None:
    if (
        source == destination
        or source in destination.parents
        or destination in source.parents
    ):
        raise DojoDriveArchiveError("source and destination trees must not overlap")


def _hash_regular_file(path: Path) -> tuple[str, int]:
    try:
        before_path = path.lstat()
    except OSError as exc:
        raise DojoDriveArchiveError(f"cannot stat file {path}: {exc}") from exc
    if not stat.S_ISREG(before_path.st_mode):
        raise DojoDriveArchiveError(f"archive input is not a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    digest = hashlib.sha256()
    observed = 0
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            before_fd = os.fstat(handle.fileno())
            if _stable_identity(before_fd) != _stable_identity(before_path):
                raise DojoDriveArchiveError(f"file changed before read: {path}")
            while chunk := handle.read(HASH_CHUNK_BYTES):
                digest.update(chunk)
                observed += len(chunk)
            after_fd = os.fstat(handle.fileno())
    except DojoDriveArchiveError:
        raise
    except OSError as exc:
        raise DojoDriveArchiveError(f"cannot read file {path}: {exc}") from exc
    try:
        after_path = path.lstat()
    except OSError as exc:
        raise DojoDriveArchiveError(f"file changed after read: {path}") from exc
    if (
        _stable_identity(before_fd) != _stable_identity(after_fd)
        or _stable_identity(before_path) != _stable_identity(after_path)
        or observed != before_fd.st_size
    ):
        raise DojoDriveArchiveError(f"file changed while hashing: {path}")
    return digest.hexdigest(), observed


def _load_json(path: Path, *, field: str) -> Any:
    digest, size = _hash_regular_file(path)
    del digest
    if size > MAX_JSON_BYTES:
        raise DojoDriveArchiveError(f"{field} exceeds JSON size limit")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            raw = handle.read(MAX_JSON_BYTES + 1)
    except OSError as exc:
        raise DojoDriveArchiveError(f"cannot read {field}: {path}: {exc}") from exc
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DojoDriveArchiveError(f"invalid {field}: {exc}") from exc
    _validate_json(value, field=field)
    return value


def _sealed_body(value: Mapping[str, Any], field: str, label: str) -> dict[str, Any]:
    digest = value.get(field)
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise DojoDriveArchiveError(f"{label} is missing {field}")
    body = {key: item for key, item in value.items() if key != field}
    if canonical_sha256(body) != digest:
        raise DojoDriveArchiveError(f"{label} {field} mismatch")
    return body


def _declared_artifact(root: Path, value: Any, expected_name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise DojoDriveArchiveError(f"declared {expected_name} path is invalid")
    declared = Path(value)
    candidate = declared if declared.is_absolute() else root / declared
    normalized = Path(os.path.abspath(candidate))
    expected = root / expected_name
    if normalized != expected:
        raise DojoDriveArchiveError(
            f"declared {expected_name} path does not bind the terminal run root"
        )
    return _assert_real_tree_path(root, expected_name, expect_directory=False)


def _coordinate_identity(
    value: Mapping[str, Any], *, field: str
) -> tuple[str, str, str]:
    result: list[str] = []
    for name in ("candidate_id", "intrabar", "cost_arm"):
        item = value.get(name)
        if not isinstance(item, str) or not item:
            raise DojoDriveArchiveError(f"{field}.{name} must be a non-empty string")
        result.append(item)
    return result[0], result[1], result[2]


def _validate_cells_and_coordinates(
    *, root: Path, run: Mapping[str, Any], cells: Sequence[Any]
) -> None:
    study_sha256 = run.get("study_sha256")
    if not isinstance(study_sha256, str) or not _SHA256_RE.fullmatch(study_sha256):
        raise DojoDriveArchiveError("run study SHA-256 is invalid")
    corpus = run.get("corpus")
    coverage = corpus.get("sparse_m1_coverage") if isinstance(corpus, Mapping) else None
    pairs = coverage.get("feed_pairs") if isinstance(coverage, Mapping) else None
    if (
        not isinstance(pairs, list)
        or not pairs
        or any(not isinstance(pair, str) or not pair for pair in pairs)
        or len(pairs) != len(set(pairs))
    ):
        raise DojoDriveArchiveError("run corpus feed-pair identity is invalid")
    feed_pairs = set(pairs)
    expected_pairs: set[str] | None = None

    cells_by_coordinate: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for index, raw in enumerate(cells):
        cell = _require_exact_keys(raw, expected=_CELL_KEYS, field=f"cells[{index}]")
        metrics = cell.get("metrics")
        if (
            cell.get("contract") != CELL_CONTRACT
            or cell.get("schema_version") != 1
            or cell.get("study_sha256") != study_sha256
            or not isinstance(cell.get("proposal_sha256"), str)
            or not _SHA256_RE.fullmatch(str(cell.get("proposal_sha256")))
            or cell.get("execution_status") not in {"SUCCESS", "FAILED"}
            or not isinstance(metrics, Mapping)
            or not isinstance(cell.get("ledger_evidence"), Mapping)
        ):
            raise DojoDriveArchiveError(f"cells[{index}] contract is invalid")
        pair_pnl = metrics.get("pair_pnl_jpy")
        lopo_net = metrics.get("leave_one_pair_out_net_jpy")
        if (
            not isinstance(pair_pnl, Mapping)
            or not isinstance(lopo_net, Mapping)
            or not pair_pnl
            or set(pair_pnl) != set(lopo_net)
            or not set(pair_pnl).issubset(feed_pairs)
            or any(not isinstance(pair, str) or not pair for pair in pair_pnl)
        ):
            raise DojoDriveArchiveError(f"cells[{index}] pair grid is invalid")
        cell_pairs = set(pair_pnl)
        if expected_pairs is None:
            expected_pairs = cell_pairs
        elif cell_pairs != expected_pairs:
            raise DojoDriveArchiveError("cells use inconsistent pair grids")
        _sealed_body(cell, "cell_sha256", f"cells[{index}]")
        identity = _coordinate_identity(cell, field=f"cells[{index}]")
        if identity in cells_by_coordinate:
            raise DojoDriveArchiveError("cells contain a duplicate coordinate")
        cells_by_coordinate[identity] = cell

    if expected_pairs is None:
        raise DojoDriveArchiveError("cells do not define a trade-pair grid")

    coordinates_by_identity: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    failed_coordinates = 0
    for index, raw in enumerate(run["coordinates"]):
        coordinate = _require_exact_keys(
            raw, expected=_COORDINATE_KEYS, field=f"coordinates[{index}]"
        )
        identity = _coordinate_identity(coordinate, field=f"coordinates[{index}]")
        if identity in coordinates_by_identity:
            raise DojoDriveArchiveError("run contains a duplicate coordinate")
        coordinates_by_identity[identity] = coordinate
        status = coordinate.get("status")
        if status not in {
            "COMPLETE",
            "MAIN_REPLAY_FAILED_SENTINEL",
            "LOPO_INCOMPLETE_NO_ADDITIVE_SUBSTITUTE",
        }:
            raise DojoDriveArchiveError("run coordinate status is unsupported")
        if status != "COMPLETE":
            failed_coordinates += 1
        main_relative = _safe_relative(
            root,
            coordinate.get("main_session_dir"),
            field=f"coordinates[{index}].main_session_dir",
        )
        _assert_real_tree_path(root, main_relative, expect_directory=True)
        cell = cells_by_coordinate.get(identity)
        if cell is None or cell.get("cell_sha256") != coordinate.get("cell_sha256"):
            raise DojoDriveArchiveError("coordinate does not bind its sealed cell")
        lopo = coordinate.get("lopo")
        if not isinstance(lopo, list):
            raise DojoDriveArchiveError("coordinate LOPO receipts must be a list")
        if status == "MAIN_REPLAY_FAILED_SENTINEL":
            if (
                coordinate.get("lopo_replay_complete") is not False
                or lopo
                or cell.get("execution_status") != "FAILED"
                or cell.get("failure_code") != "MAIN_REPLAY_FAILED"
            ):
                raise DojoDriveArchiveError("main-failure coordinate is inconsistent")
            continue
        if len(lopo) != len(expected_pairs):
            raise DojoDriveArchiveError("coordinate LOPO denominator is incomplete")
        held_out: set[str] = set()
        lopo_failed = False
        for lopo_index, raw_lopo in enumerate(lopo):
            if not isinstance(raw_lopo, Mapping):
                raise DojoDriveArchiveError("coordinate LOPO receipt is invalid")
            pair = raw_lopo.get("held_out_pair")
            lopo_status = raw_lopo.get("status")
            if (
                not isinstance(pair, str)
                or pair not in expected_pairs
                or pair in held_out
                or lopo_status
                not in {
                    "VALID_COUNTERFACTUAL_REPLAY",
                    "FAILED_NO_ADDITIVE_SUBSTITUTE",
                }
            ):
                raise DojoDriveArchiveError("coordinate LOPO identity is invalid")
            held_out.add(pair)
            session_relative = _safe_relative(
                root,
                raw_lopo.get("session_dir"),
                field=f"coordinates[{index}].lopo[{lopo_index}].session_dir",
            )
            _assert_real_tree_path(root, session_relative, expect_directory=True)
            if lopo_status == "VALID_COUNTERFACTUAL_REPLAY":
                ledger_relative = _safe_relative(
                    root,
                    raw_lopo.get("ledger_path"),
                    field=f"coordinates[{index}].lopo[{lopo_index}].ledger_path",
                )
                _assert_real_tree_path(root, ledger_relative, expect_directory=False)
                if (
                    PurePosixPath(session_relative)
                    not in PurePosixPath(ledger_relative).parents
                ):
                    raise DojoDriveArchiveError("LOPO ledger escapes its session")
                if raw_lopo.get("corpus_sha256") != corpus.get("corpus_sha256"):
                    raise DojoDriveArchiveError("LOPO corpus identity drifted")
                terminal_net = raw_lopo.get("terminal_net_jpy")
                if (
                    not isinstance(terminal_net, (int, float))
                    or isinstance(terminal_net, bool)
                    or not math.isfinite(float(terminal_net))
                ):
                    raise DojoDriveArchiveError("LOPO terminal net is invalid")
            else:
                lopo_failed = True
        if held_out != expected_pairs:
            raise DojoDriveArchiveError("coordinate LOPO pair set is incomplete")
        if status == "COMPLETE":
            if (
                lopo_failed
                or coordinate.get("lopo_replay_complete") is not True
                or cell.get("execution_status") != "SUCCESS"
                or cell.get("failure_code") is not None
            ):
                raise DojoDriveArchiveError("complete coordinate is inconsistent")
        elif (
            not lopo_failed
            or coordinate.get("lopo_replay_complete") is not False
            or cell.get("execution_status") != "FAILED"
            or cell.get("failure_code") != "COUNTERFACTUAL_LOPO_INCOMPLETE"
        ):
            raise DojoDriveArchiveError("LOPO-failure coordinate is inconsistent")
    if set(coordinates_by_identity) != set(cells_by_coordinate):
        raise DojoDriveArchiveError("coordinate and cell grids differ")
    if failed_coordinates != run["fixed_denominator"].get("failed_cell_count"):
        raise DojoDriveArchiveError(
            "failed coordinate count disagrees with denominator"
        )


def validate_terminal_run(source_run: Path | str) -> dict[str, Any]:
    """Validate the trainer's complete, fixed-denominator terminal receipt."""

    root = _real_directory(source_run, label="source run")
    failure_path = root / "run_failure.json"
    if failure_path.exists() or failure_path.is_symlink():
        raise DojoDriveArchiveError(
            "run_failure.json is present; successful terminal archive is ambiguous"
        )
    run = _load_json(root / "run.json", field="run receipt")
    run = _require_exact_keys(run, expected=_RUN_KEYS, field="run receipt")
    _sealed_body(run, "run_sha256", "run receipt")
    if run.get("contract") != RUN_CONTRACT or run.get("schema_version") != 1:
        raise DojoDriveArchiveError("unsupported run receipt contract")
    if run.get("status") not in {"COMPLETE", "COMPLETE_WITH_FAILED_CELLS"}:
        raise DojoDriveArchiveError("run is not terminal")
    if (
        run.get("live_permission") is not False
        or run.get("promotion_eligible") is not False
        or run.get("proof_eligible") is not False
        or run.get("broker_mutation_allowed") is not False
        or run.get("order_authority") != "NONE"
    ):
        raise DojoDriveArchiveError("terminal run authority boundary is invalid")
    denominator = run.get("fixed_denominator")
    if not isinstance(denominator, Mapping):
        raise DojoDriveArchiveError("run fixed denominator is missing")
    expected = denominator.get("expected_cell_count")
    observed = denominator.get("observed_cell_count")
    failed = denominator.get("failed_cell_count")
    if (
        not isinstance(expected, int)
        or isinstance(expected, bool)
        or expected < 1
        or observed != expected
        or not isinstance(failed, int)
        or isinstance(failed, bool)
        or not 0 <= failed <= expected
        or denominator.get("dropped_cell_count") != 0
        or denominator.get("coordinate_receipts_complete") is not True
    ):
        raise DojoDriveArchiveError(
            "run fixed denominator is not terminal and complete"
        )
    if (run.get("status") == "COMPLETE") != (failed == 0):
        raise DojoDriveArchiveError("run status and failed-cell count disagree")
    coordinates = run.get("coordinates")
    if not isinstance(coordinates, list) or len(coordinates) != expected:
        raise DojoDriveArchiveError("run coordinate receipts are incomplete")

    evaluation_path = _declared_artifact(
        root, run.get("evaluation_path"), "evaluation.json"
    )
    cells_path = _declared_artifact(root, run.get("cells_path"), "cells.json")
    evaluation = _load_json(evaluation_path, field="evaluation")
    if not isinstance(evaluation, Mapping):
        raise DojoDriveArchiveError("evaluation must be a JSON object")
    _sealed_body(evaluation, "evaluation_sha256", "evaluation")
    if (
        evaluation.get("contract") != EVALUATION_CONTRACT
        or evaluation.get("schema_version") != 1
        or evaluation.get("evaluation_sha256") != run.get("evaluation_sha256")
        or evaluation.get("study_sha256") != run.get("study_sha256")
        or evaluation.get("live_permission") is not False
        or evaluation.get("promotion_eligible") is not False
        or evaluation.get("proof_eligible") is not False
        or evaluation.get("broker_mutation_allowed") is not False
        or evaluation.get("order_authority") != "NONE"
    ):
        raise DojoDriveArchiveError("evaluation does not bind the terminal run")
    evaluation_denominator = evaluation.get("fixed_denominator")
    if (
        not isinstance(evaluation_denominator, Mapping)
        or evaluation_denominator.get("expected_cell_count") != expected
        or evaluation_denominator.get("observed_cell_count") != expected
        or evaluation_denominator.get("coordinate_receipts_complete") is not True
    ):
        raise DojoDriveArchiveError("evaluation fixed denominator is incomplete")
    cells = _load_json(cells_path, field="cells")
    if not isinstance(cells, list) or len(cells) != expected:
        raise DojoDriveArchiveError("cells artifact does not match fixed denominator")
    _validate_cells_and_coordinates(root=root, run=run, cells=cells)
    return dict(run)


def _safe_relative(root: Path, declared: Any, *, field: str) -> str:
    if not isinstance(declared, str) or not declared:
        raise DojoDriveArchiveError(f"{field} is not a path")
    candidate = Path(declared)
    absolute = Path(
        os.path.abspath(candidate if candidate.is_absolute() else root / candidate)
    )
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise DojoDriveArchiveError(f"{field} escapes the source run") from exc
    pure = PurePosixPath(relative.as_posix())
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise DojoDriveArchiveError(f"{field} is not canonical and relative")
    relative_text = pure.as_posix()
    _assert_real_tree_path(root, relative_text)
    return relative_text


def _walk_regular_files(root: Path, starts: Sequence[str]) -> list[str]:
    files: set[str] = set()
    stack: list[Path] = []
    for relative in starts:
        path = _assert_real_tree_path(root, relative)
        try:
            state = path.lstat()
        except OSError as exc:
            raise DojoDriveArchiveError(
                f"archive input is unavailable: {relative}"
            ) from exc
        if stat.S_ISLNK(state.st_mode):
            raise DojoDriveArchiveError(f"archive input contains a symlink: {relative}")
        if stat.S_ISREG(state.st_mode):
            files.add(relative)
        elif stat.S_ISDIR(state.st_mode):
            stack.append(path)
        else:
            raise DojoDriveArchiveError(f"archive input is a special file: {relative}")
    while stack:
        directory = stack.pop()
        directory_relative = directory.relative_to(root).as_posix()
        _assert_real_tree_path(root, directory_relative, expect_directory=True)
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise DojoDriveArchiveError(
                f"cannot scan archive input: {directory}"
            ) from exc
        for child in children:
            path = Path(child.path)
            relative = path.relative_to(root).as_posix()
            try:
                if child.is_symlink():
                    raise DojoDriveArchiveError(
                        f"archive input contains a symlink: {relative}"
                    )
                if child.is_file(follow_symlinks=False):
                    files.add(relative)
                elif child.is_dir(follow_symlinks=False):
                    stack.append(path)
                else:
                    raise DojoDriveArchiveError(
                        f"archive input is a special file: {relative}"
                    )
            except OSError as exc:
                raise DojoDriveArchiveError(
                    f"cannot inspect archive input: {relative}"
                ) from exc
            _assert_real_tree_path(root, relative)
            if len(files) + len(stack) > MAX_FILES:
                raise DojoDriveArchiveError("archive input exceeds file-count limit")
    if not files:
        raise DojoDriveArchiveError("archive chunk contains no regular files")
    return sorted(files)


def _cell_starts(root: Path, run: Mapping[str, Any], chunk_id: str) -> list[str]:
    matches: list[Mapping[str, Any]] = []
    for raw in run["coordinates"]:
        if not isinstance(raw, Mapping):
            raise DojoDriveArchiveError("run coordinate is not a JSON object")
        identity = "|".join(
            str(raw.get(field, ""))
            for field in ("candidate_id", "intrabar", "cost_arm")
        )
        if identity == chunk_id:
            matches.append(raw)
    if len(matches) != 1:
        raise DojoDriveArchiveError(
            "cell chunk_id must exactly match candidate_id|intrabar|cost_arm"
        )
    coordinate = matches[0]
    starts = ["run.json", "evaluation.json", "cells.json"]
    starts.append(
        _safe_relative(
            root, coordinate.get("main_session_dir"), field="main_session_dir"
        )
    )
    lopo = coordinate.get("lopo")
    if not isinstance(lopo, list):
        raise DojoDriveArchiveError("cell coordinate LOPO receipts are missing")
    for index, row in enumerate(lopo):
        if not isinstance(row, Mapping):
            raise DojoDriveArchiveError("cell LOPO receipt is not a JSON object")
        starts.append(
            _safe_relative(
                root, row.get("session_dir"), field=f"lopo[{index}].session_dir"
            )
        )
    return starts


def _inventory(root: Path, paths: Sequence[str]) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    total = 0
    for relative in paths:
        pure = PurePosixPath(relative)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise DojoDriveArchiveError("inventory path is not canonical and relative")
        source = _assert_real_tree_path(root, pure.as_posix(), expect_directory=False)
        digest, size = _hash_regular_file(source)
        rows.append({"path": pure.as_posix(), "size_bytes": size, "sha256": digest})
        total += size
    return rows, total


def _validate_month_binding(run: Mapping[str, Any], chunk_id: str) -> None:
    corpus = run.get("corpus")
    coverage = corpus.get("sparse_m1_coverage") if isinstance(corpus, Mapping) else None
    if not isinstance(coverage, Mapping):
        raise DojoDriveArchiveError("month chunk requires sealed corpus coverage")
    first_epoch = coverage.get("first_epoch")
    last_epoch = coverage.get("last_epoch")
    if (
        not isinstance(first_epoch, int)
        or isinstance(first_epoch, bool)
        or not isinstance(last_epoch, int)
        or isinstance(last_epoch, bool)
        or first_epoch < 0
        or last_epoch < first_epoch
    ):
        raise DojoDriveArchiveError("month chunk corpus epoch bounds are invalid")
    try:
        first_month = datetime.fromtimestamp(first_epoch, tz=timezone.utc).strftime(
            "%Y-%m"
        )
        last_month = datetime.fromtimestamp(last_epoch, tz=timezone.utc).strftime(
            "%Y-%m"
        )
    except (OverflowError, OSError, ValueError) as exc:
        raise DojoDriveArchiveError(
            "month chunk corpus epoch bounds are invalid"
        ) from exc
    if first_month != chunk_id or last_month != chunk_id:
        raise DojoDriveArchiveError(
            "month chunk_id must match the complete corpus UTC month"
        )


def _expected_inventory(
    *, root: Path, run: Mapping[str, Any], chunk_kind: str, chunk_id: str
) -> tuple[list[dict[str, Any]], int]:
    if chunk_kind == "cell":
        starts = _cell_starts(root, run, chunk_id)
    else:
        _validate_month_binding(run, chunk_id)
        starts = [
            child.name for child in sorted(os.scandir(root), key=lambda item: item.name)
        ]
    paths = _walk_regular_files(root, starts)
    return _inventory(root, paths)


def _atomic_json(path: Path, value: Any) -> None:
    payload = canonical_bytes(value) + b"\n"
    if path.exists() or path.is_symlink():
        existing = _load_json(path, field=path.name)
        if existing != value:
            raise DojoDriveArchiveError(f"existing receipt conflicts: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            if temporary.exists() and not temporary.is_symlink():
                temporary.unlink()
        except OSError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _finalize_lock(destination: Path, plan_sha256: str):
    lock_dir = destination / ".locks"
    lock_dir.mkdir(mode=0o700, exist_ok=True)
    if lock_dir.is_symlink() or not lock_dir.is_dir():
        raise DojoDriveArchiveError("archive lock directory must be real")
    lock_path = lock_dir / f"{plan_sha256}.lock"
    if lock_path.is_symlink():
        raise DojoDriveArchiveError("archive lock path must not be a symlink")
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        state = os.fstat(descriptor)
        if not stat.S_ISREG(state.st_mode):
            raise DojoDriveArchiveError("archive lock must be a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def plan_archive(
    *,
    source_run: Path | str,
    destination: Path | str,
    chunk_kind: str,
    chunk_id: str,
) -> dict[str, Any]:
    """Seal a content-addressed local plan for one cell or one monthly run."""

    if chunk_kind not in {"cell", "month"}:
        raise DojoDriveArchiveError("chunk_kind must be cell or month")
    if not _CHUNK_ID_RE.fullmatch(chunk_id) or chunk_id in {".", ".."}:
        raise DojoDriveArchiveError("chunk_id is not a safe identifier")
    if chunk_kind == "month" and not _MONTH_ID_RE.fullmatch(chunk_id):
        raise DojoDriveArchiveError("month chunk_id must use YYYY-MM")
    root = _real_directory(source_run, label="source run")
    destination_root = _ensure_destination(destination)
    _assert_separate_roots(root, destination_root)
    run = validate_terminal_run(root)
    inventory, total = _expected_inventory(
        root=root, run=run, chunk_kind=chunk_kind, chunk_id=chunk_id
    )
    body = {
        "contract": PLAN_CONTRACT,
        "schema_version": 1,
        "source_run_root": os.fspath(root),
        "destination_root": os.fspath(destination_root),
        "chunk_kind": chunk_kind,
        "chunk_id": chunk_id,
        "terminal_run": {
            "contract": run["contract"],
            "status": run["status"],
            "run_sha256": run["run_sha256"],
            "study_sha256": run.get("study_sha256"),
            "evaluation_sha256": run["evaluation_sha256"],
            "classification": run.get("classification"),
            "fixed_denominator": dict(run["fixed_denominator"]),
        },
        "file_count": len(inventory),
        "total_source_bytes": total,
        "content_tree_sha256": canonical_sha256(inventory),
        "files": inventory,
        "archive_format": "POSIX_PAX_TAR_ZSTD",
        "archive_member_prefix": "run/",
        "source_deletion_allowed": False,
        "source_deleted": False,
        "remote_verification": {
            "status": "NOT_REQUESTED",
            "remote_verified": False,
            "metadata_receipt_sha256": None,
        },
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    plan = {**body, "plan_sha256": canonical_sha256(body)}
    plan_dir = destination_root / "plans"
    plan_dir.mkdir(mode=0o700, exist_ok=True)
    if plan_dir.is_symlink():
        raise DojoDriveArchiveError("plans directory must not be a symlink")
    plan_path = plan_dir / f"{chunk_kind}-{chunk_id}-{plan['plan_sha256']}.json"
    _atomic_json(plan_path, plan)
    return {**plan, "plan_path": os.fspath(plan_path)}


def _validate_plan(plan_path: Path | str) -> tuple[dict[str, Any], Path, Path]:
    path = Path(plan_path).expanduser()
    plan = _load_json(path, field="archive plan")
    plan = _require_exact_keys(plan, expected=_PLAN_KEYS, field="archive plan")
    _sealed_body(plan, "plan_sha256", "archive plan")
    if plan.get("contract") != PLAN_CONTRACT or plan.get("schema_version") != 1:
        raise DojoDriveArchiveError("unsupported archive plan contract")
    if (
        plan.get("source_deletion_allowed") is not False
        or plan.get("source_deleted") is not False
        or plan.get("proof_eligible") is not False
        or plan.get("promotion_eligible") is not False
        or plan.get("live_permission") is not False
        or plan.get("broker_mutation_allowed") is not False
        or plan.get("order_authority") != "NONE"
    ):
        raise DojoDriveArchiveError("archive plan authority boundary is invalid")
    remote = plan.get("remote_verification")
    _require_exact_keys(remote, expected=_REMOTE_KEYS, field="remote verification")
    if remote != {
        "status": "NOT_REQUESTED",
        "remote_verified": False,
        "metadata_receipt_sha256": None,
    }:
        raise DojoDriveArchiveError(
            "archive plan cannot self-assert remote verification"
        )
    chunk_kind = plan.get("chunk_kind")
    chunk_id = plan.get("chunk_id")
    if (
        chunk_kind not in {"cell", "month"}
        or not isinstance(chunk_id, str)
        or not _CHUNK_ID_RE.fullmatch(chunk_id)
        or chunk_id in {".", ".."}
        or (chunk_kind == "month" and not _MONTH_ID_RE.fullmatch(chunk_id))
        or plan.get("archive_format") != "POSIX_PAX_TAR_ZSTD"
        or plan.get("archive_member_prefix") != "run/"
    ):
        raise DojoDriveArchiveError("archive plan chunk contract is invalid")
    root = _real_directory(str(plan.get("source_run_root", "")), label="source run")
    destination = _real_directory(
        str(plan.get("destination_root", "")), label="destination root"
    )
    _assert_separate_roots(root, destination)
    run = validate_terminal_run(root)
    terminal = plan.get("terminal_run")
    terminal = _require_exact_keys(
        terminal, expected=_PLAN_TERMINAL_KEYS, field="plan terminal run"
    )
    if (
        terminal.get("run_sha256") != run.get("run_sha256")
        or terminal.get("evaluation_sha256") != run.get("evaluation_sha256")
        or terminal.get("status") != run.get("status")
        or terminal.get("contract") != run.get("contract")
        or terminal.get("study_sha256") != run.get("study_sha256")
        or terminal.get("classification") != run.get("classification")
        or terminal.get("fixed_denominator") != run.get("fixed_denominator")
    ):
        raise DojoDriveArchiveError("terminal run changed after archive planning")
    rows = plan.get("files")
    if not isinstance(rows, list) or len(rows) != plan.get("file_count") or not rows:
        raise DojoDriveArchiveError("archive plan inventory is invalid")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    total = 0
    for raw in rows:
        raw = _require_exact_keys(
            raw, expected=_PLAN_FILE_KEYS, field="archive inventory row"
        )
        relative = raw.get("path")
        size = raw.get("size_bytes")
        digest = raw.get("sha256")
        if (
            not isinstance(relative, str)
            or relative in seen
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or not _SHA256_RE.fullmatch(digest)
        ):
            raise DojoDriveArchiveError("archive inventory row is invalid")
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or not pure.parts
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise DojoDriveArchiveError("archive inventory path is unsafe")
        seen.add(relative)
        normalized.append({"path": relative, "size_bytes": size, "sha256": digest})
        total += size
    if normalized != sorted(normalized, key=lambda row: row["path"]):
        raise DojoDriveArchiveError("archive inventory must be sorted")
    if total != plan.get("total_source_bytes") or canonical_sha256(
        normalized
    ) != plan.get("content_tree_sha256"):
        raise DojoDriveArchiveError("archive inventory summary mismatch")
    expected_inventory, expected_total = _expected_inventory(
        root=root,
        run=run,
        chunk_kind=str(chunk_kind),
        chunk_id=chunk_id,
    )
    if normalized != expected_inventory or total != expected_total:
        raise DojoDriveArchiveError(
            "archive plan inventory differs from the source-derived chunk"
        )
    expected_path = (
        destination / "plans" / f"{chunk_kind}-{chunk_id}-{plan['plan_sha256']}.json"
    )
    if Path(os.path.abspath(path)) != expected_path:
        raise DojoDriveArchiveError(
            "archive plan path is not its canonical destination"
        )
    _real_directory(expected_path.parent, label="plans directory")
    return dict(plan), root, destination


def _verify_sources(plan: Mapping[str, Any], root: Path) -> None:
    for row in plan["files"]:
        source = _assert_real_tree_path(root, row["path"], expect_directory=False)
        digest, size = _hash_regular_file(source)
        if digest != row["sha256"] or size != row["size_bytes"]:
            raise DojoDriveArchiveError(
                f"source file changed after planning: {row['path']}"
            )


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mode = 0o444
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def _open_stable_source(
    path: Path,
) -> tuple[BinaryIO, tuple[int, int, int, int, int, int]]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise DojoDriveArchiveError(f"source is no longer a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    handle = os.fdopen(descriptor, "rb", closefd=True)
    identity = _stable_identity(os.fstat(handle.fileno()))
    if identity != _stable_identity(before):
        handle.close()
        raise DojoDriveArchiveError(f"source changed before archive read: {path}")
    return handle, identity


def _resolve_zstd(zstd_bin: str) -> str:
    resolved = shutil.which(zstd_bin)
    if resolved is None:
        raise DojoDriveArchiveError(f"zstd executable is unavailable: {zstd_bin}")
    return resolved


def _write_archive(
    path: Path, plan: Mapping[str, Any], root: Path, *, zstd_bin: str
) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    output = os.fdopen(descriptor, "wb", closefd=True)
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            [_resolve_zstd(zstd_bin), "-q", "-T0", "-c"],
            stdin=subprocess.PIPE,
            stdout=output,
            stderr=subprocess.PIPE,
        )
        if process.stdin is None:
            raise DojoDriveArchiveError("zstd stdin was not created")
        with tarfile.open(
            fileobj=process.stdin, mode="w|", format=tarfile.PAX_FORMAT
        ) as archive:
            plan_payload = canonical_bytes(plan) + b"\n"
            archive.addfile(
                _tar_info(".dojo/archive-plan.json", len(plan_payload)),
                _BytesReader(plan_payload),
            )
            for row in plan["files"]:
                source = _assert_real_tree_path(
                    root, row["path"], expect_directory=False
                )
                handle, identity = _open_stable_source(source)
                try:
                    archive.addfile(
                        _tar_info(f"run/{row['path']}", row["size_bytes"]), handle
                    )
                    if _stable_identity(os.fstat(handle.fileno())) != identity:
                        raise DojoDriveArchiveError(
                            f"source changed during archive read: {row['path']}"
                        )
                finally:
                    handle.close()
        process.stdin.close()
        return_code = process.wait()
        error = (
            process.stderr.read().decode("utf-8", errors="replace")
            if process.stderr
            else ""
        )
        if return_code != 0:
            raise DojoDriveArchiveError(f"zstd failed: {error[:1000]}")
        output.flush()
        os.fsync(output.fileno())
    except Exception:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        raise
    finally:
        output.close()


class _BytesReader:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._payload) - self._offset
        start = self._offset
        self._offset = min(len(self._payload), self._offset + size)
        return self._payload[start : self._offset]


def _verify_archive_payload(
    archive_path: Path, plan: Mapping[str, Any], *, zstd_bin: str
) -> None:
    state = archive_path.lstat()
    if not stat.S_ISREG(state.st_mode):
        raise DojoDriveArchiveError("archive output must be a regular file")
    process = subprocess.Popen(
        [_resolve_zstd(zstd_bin), "-q", "-d", "-c", os.fspath(archive_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    observed: dict[str, tuple[str, int]] = {}
    observed_plan: dict[str, Any] | None = None
    try:
        if process.stdout is None:
            raise DojoDriveArchiveError("zstd stdout was not created")
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                name = PurePosixPath(member.name)
                if (
                    member.name.startswith("/")
                    or any(part in {"", ".", ".."} for part in name.parts)
                    or not member.isfile()
                ):
                    raise DojoDriveArchiveError("archive contains an unsafe member")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise DojoDriveArchiveError("archive member cannot be read")
                digest = hashlib.sha256()
                size = 0
                payload = (
                    bytearray() if member.name == ".dojo/archive-plan.json" else None
                )
                while chunk := extracted.read(HASH_CHUNK_BYTES):
                    digest.update(chunk)
                    size += len(chunk)
                    if payload is not None:
                        payload.extend(chunk)
                        if len(payload) > MAX_JSON_BYTES:
                            raise DojoDriveArchiveError(
                                "embedded plan exceeds size limit"
                            )
                if member.name == ".dojo/archive-plan.json":
                    try:
                        embedded = json.loads(
                            bytes(payload).decode("utf-8"),
                            object_pairs_hook=_reject_duplicates,
                            parse_constant=_reject_constant,
                        )
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise DojoDriveArchiveError("embedded plan is invalid") from exc
                    if not isinstance(embedded, dict) or observed_plan is not None:
                        raise DojoDriveArchiveError(
                            "embedded plan is missing or duplicated"
                        )
                    observed_plan = embedded
                    continue
                if len(name.parts) < 2 or name.parts[0] != "run":
                    raise DojoDriveArchiveError(
                        "archive member is outside the run prefix"
                    )
                relative = PurePosixPath(*name.parts[1:]).as_posix()
                if relative in observed:
                    raise DojoDriveArchiveError("archive contains a duplicate member")
                observed[relative] = (digest.hexdigest(), size)
        return_code = process.wait()
        error = (
            process.stderr.read().decode("utf-8", errors="replace")
            if process.stderr
            else ""
        )
        if return_code != 0:
            raise DojoDriveArchiveError(f"zstd verification failed: {error[:1000]}")
    except DojoDriveArchiveError:
        if process.poll() is None:
            process.terminate()
            process.wait()
        raise
    except (OSError, tarfile.TarError) as exc:
        if process.poll() is None:
            process.terminate()
            process.wait()
        raise DojoDriveArchiveError(
            f"archive payload is incomplete or invalid: {exc}"
        ) from exc
    expected = {
        row["path"]: (row["sha256"], row["size_bytes"]) for row in plan["files"]
    }
    if observed_plan != dict(plan) or observed != expected:
        raise DojoDriveArchiveError("archive payload does not match its sealed plan")


def _finalize_archive_locked(
    *,
    plan_path: Path | str,
    plan: Mapping[str, Any],
    root: Path,
    destination: Path,
    zstd_bin: str,
) -> dict[str, Any]:
    _verify_sources(plan, root)
    archive_dir = destination / "archives"
    receipt_dir = destination / "receipts"
    archive_dir.mkdir(mode=0o700, exist_ok=True)
    receipt_dir.mkdir(mode=0o700, exist_ok=True)
    if archive_dir.is_symlink() or receipt_dir.is_symlink():
        raise DojoDriveArchiveError("archive output directories must not be symlinks")
    stem = f"{plan['chunk_kind']}-{plan['chunk_id']}-{plan['plan_sha256']}"
    final_path = archive_dir / f"{stem}.tar.zst"
    part_path = archive_dir / f"{stem}.tar.zst.part"
    if final_path.is_symlink() or part_path.is_symlink():
        raise DojoDriveArchiveError("archive output path must not be a symlink")
    if final_path.exists():
        _verify_archive_payload(final_path, plan, zstd_bin=zstd_bin)
    else:
        part_ready = False
        if part_path.exists():
            try:
                _verify_archive_payload(part_path, plan, zstd_bin=zstd_bin)
                part_ready = True
            except DojoDriveArchiveError:
                rebuild = archive_dir / f".{stem}.{os.getpid()}.part"
                _write_archive(rebuild, plan, root, zstd_bin=zstd_bin)
                _verify_archive_payload(rebuild, plan, zstd_bin=zstd_bin)
                os.replace(rebuild, part_path)
                _fsync_directory(archive_dir)
                part_ready = True
        if not part_ready:
            _write_archive(part_path, plan, root, zstd_bin=zstd_bin)
            _verify_archive_payload(part_path, plan, zstd_bin=zstd_bin)
        os.replace(part_path, final_path)
        _fsync_directory(archive_dir)
    archive_sha256, archive_size = _hash_regular_file(final_path)
    body = {
        "contract": FINALIZATION_CONTRACT,
        "schema_version": 1,
        "plan_path": os.fspath(Path(plan_path).expanduser().resolve(strict=True)),
        "plan_sha256": plan["plan_sha256"],
        "content_tree_sha256": plan["content_tree_sha256"],
        "chunk_kind": plan["chunk_kind"],
        "chunk_id": plan["chunk_id"],
        "archive_path": os.fspath(final_path),
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_size,
        "file_count": plan["file_count"],
        "total_source_bytes": plan["total_source_bytes"],
        "local_payload_verified": True,
        "atomic_publish_complete": True,
        "source_deletion_allowed": False,
        "source_deleted": False,
        "remote_verification": {
            "status": "NOT_REQUESTED",
            "remote_verified": False,
            "metadata_receipt_sha256": None,
        },
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    receipt = {**body, "finalization_sha256": canonical_sha256(body)}
    receipt_path = receipt_dir / f"{stem}.json"
    _atomic_json(receipt_path, receipt)
    return {**receipt, "receipt_path": os.fspath(receipt_path)}


def finalize_archive(
    *, plan_path: Path | str, zstd_bin: str = "zstd"
) -> dict[str, Any]:
    """Build, verify, and atomically publish one planned archive chunk."""

    plan, _, destination = _validate_plan(plan_path)
    with _finalize_lock(destination, plan["plan_sha256"]):
        locked_plan, root, locked_destination = _validate_plan(plan_path)
        if (
            locked_plan["plan_sha256"] != plan["plan_sha256"]
            or locked_destination != destination
        ):
            raise DojoDriveArchiveError("archive plan changed while acquiring lock")
        return _finalize_archive_locked(
            plan_path=plan_path,
            plan=locked_plan,
            root=root,
            destination=locked_destination,
            zstd_bin=zstd_bin,
        )


def verify_finalized_archive(
    *, plan_path: Path | str, zstd_bin: str = "zstd"
) -> dict[str, Any]:
    """Re-open a finalized local archive and its receipt without changing it."""

    plan, _, destination = _validate_plan(plan_path)
    stem = f"{plan['chunk_kind']}-{plan['chunk_id']}-{plan['plan_sha256']}"
    archive_path = destination / "archives" / f"{stem}.tar.zst"
    receipt_path = destination / "receipts" / f"{stem}.json"
    receipt = _load_json(receipt_path, field="finalization receipt")
    receipt = _require_exact_keys(
        receipt, expected=_FINALIZATION_KEYS, field="finalization receipt"
    )
    _sealed_body(receipt, "finalization_sha256", "finalization receipt")
    remote = _require_exact_keys(
        receipt.get("remote_verification"),
        expected=_REMOTE_KEYS,
        field="finalization remote verification",
    )
    expected_plan_path = Path(plan_path).expanduser().resolve(strict=True)
    if (
        receipt.get("contract") != FINALIZATION_CONTRACT
        or receipt.get("schema_version") != 1
        or receipt.get("plan_path") != os.fspath(expected_plan_path)
        or receipt.get("plan_sha256") != plan["plan_sha256"]
        or receipt.get("content_tree_sha256") != plan["content_tree_sha256"]
        or receipt.get("chunk_kind") != plan["chunk_kind"]
        or receipt.get("chunk_id") != plan["chunk_id"]
        or receipt.get("archive_path") != os.fspath(archive_path)
        or receipt.get("file_count") != plan["file_count"]
        or receipt.get("total_source_bytes") != plan["total_source_bytes"]
        or receipt.get("local_payload_verified") is not True
        or receipt.get("atomic_publish_complete") is not True
        or receipt.get("source_deletion_allowed") is not False
        or receipt.get("source_deleted") is not False
        or receipt.get("proof_eligible") is not False
        or receipt.get("promotion_eligible") is not False
        or receipt.get("live_permission") is not False
        or receipt.get("order_authority") != "NONE"
        or receipt.get("broker_mutation_allowed") is not False
        or remote
        != {
            "status": "NOT_REQUESTED",
            "remote_verified": False,
            "metadata_receipt_sha256": None,
        }
    ):
        raise DojoDriveArchiveError("finalization receipt binding is invalid")
    _verify_archive_payload(archive_path, plan, zstd_bin=zstd_bin)
    digest, size = _hash_regular_file(archive_path)
    if digest != receipt.get("archive_sha256") or size != receipt.get(
        "archive_size_bytes"
    ):
        raise DojoDriveArchiveError("finalized archive hash or size drifted")
    return dict(receipt)


__all__ = [
    "DojoDriveArchiveError",
    "FINALIZATION_CONTRACT",
    "PLAN_CONTRACT",
    "canonical_sha256",
    "finalize_archive",
    "plan_archive",
    "validate_terminal_run",
    "verify_finalized_archive",
]
