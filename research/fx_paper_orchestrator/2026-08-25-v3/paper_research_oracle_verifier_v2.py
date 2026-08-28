#!/usr/bin/env python3
"""Independent accounting-only verifier for the JPY Oracle V2.

The authoritative sealed entrypoint consumes only immutable canonical request,
artifact, release, and frozen reference-result bytes.  It imports or calls no
Oracle, reference engine, strategy runner, shared accounting implementation, or
publisher capability, and returns canonical receipt and COMMIT bytes to its
launcher.  The path/FD publication adapter retained below is non-release test
scaffolding only.  Neither route can admit a trading strategy because detector
directions are never regenerated or statistically certified here.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from typing import Any, Callable, Iterable, Mapping, Sequence


try:
    _INJECTED_RUNTIME_CODE_BYTES = _SEALED_RUNTIME_CODE_BYTES
except NameError:
    _SEALED_RUNTIME = False
else:
    _SEALED_RUNTIME = True

if not _SEALED_RUNTIME:
    import argparse
    import fcntl
    import os
    import stat
    import sys
    from pathlib import Path


VERIFIER_NAME = "INDEPENDENT_JPY_ORACLE_VERIFIER_V2"
VERIFIER_SCHEMA_NAME = "paper_research_oracle_verifier_schema_v2.json"
ORACLE_NAME = "INDEPENDENT_JPY_ORACLE_V2"
ARMS = ("RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS")
ZERO_SHA = "0" * 64
DAY_NS = 86_400_000_000_000
JPY_MICROS_PER_YEN = 1_000_000
BASE_MICROUNITS_PER_UNIT = 1_000_000
RATIO_DECIMAL_SCALE = 10**18
PRICE_SUBPIP_SCALE = 1_000_000
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024
MAX_SOURCE_ROWS = 5_000_000
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PAIR_RE = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")
SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
FORBIDDEN_PROPOSAL_TOKENS = {
    "signalid", "fill", "fillprice", "path", "mfe", "mae", "pnl", "cost",
    "equity", "drawdown", "dd", "cvar", "profit", "return",
}


def _authority_items() -> tuple[tuple[str, bool | int], ...]:
    """Return the literal paper-only authority contract used for decisions."""
    return (
        ("paper_only", True),
        ("live_authority", False),
        ("broker_account_access", False),
        ("credential_access", False),
        ("order_endpoint", False),
        ("external_orders", 0),
        ("deploy", False),
        ("external_config_mutation", False),
    )


# Compatibility/display surface only.  Authoritative validation and receipts
# reconstruct the literal tuple above and never consume this mutable mapping.
AUTHORITY = dict(_authority_items())
CLASSIFICATION = "FUTURE_ONLY_ACCOUNTING_ONLY_LOCAL_UNANCHORED_NOT_ADMISSIBLE"
ANCHOR_STATUS = "LOCAL_UNANCHORED"
EXECUTION_PROVENANCE_SCOPE = (
    "LOCAL_CALLER_ASSERTED_CONTENT_BINDING_NOT_EXECUTION_ATTESTATION_"
    "NOT_EXTERNALLY_ANCHORED"
)
REFERENCE_ENGINE_ID = "EVENT_SOURCED_DOUBLE_ENTRY_REFERENCE_V1"
REFERENCE_INPUT_LABELS = (
    "source_blob",
    "source_manifest",
    "proposal",
    "execution_policy",
    "inventory_policy",
    "accounting_policy",
    "evaluation_policy",
    "instrument_registry",
    "authority_policy",
)
REFERENCE_RESULT_KEYS = frozenset({
    "engine_id",
    "input_root_sha256",
    "ledger_bytes",
    "ledger_row_count",
    "ledger_terminal_hash",
    "oracle_metrics",
    "proposal_provenance_root_sha256",
    "journal_root_sha256",
    "journal_transaction_count",
    "all_transactions_balanced",
    "economic_projection_sha256",
})
REFERENCE_RESULT_SNAPSHOT_KEYS = frozenset(
    (REFERENCE_RESULT_KEYS - {"ledger_bytes"}) | {"ledger_bytes_base64"}
)
SEALED_ARTIFACT_ROLES = tuple(sorted((
    "source_blob",
    "source_manifest",
    "proposal",
    "execution_policy",
    "inventory_policy",
    "accounting_policy",
    "evaluation_policy",
    "instrument_registry",
    "authority_policy",
    "oracle_request",
    "oracle_code_snapshot",
    "oracle_contract_snapshot",
    "oracle_schema_snapshot",
    "reference_code_snapshot",
    "reference_contract_snapshot",
    "oracle_intent",
    "oracle_commit",
    "oracle_ledger",
    "oracle_manifest",
)))
SEALED_ORACLE_RELEASE_ROLES = (
    "code_bytes",
    "contract_bytes",
    "schema_bytes",
)
REFERENCE_ATTESTATION_KEYS = (
    "reference_code_sha256",
    "reference_contract_sha256",
    "reference_result_sha256",
)
VERIFIER_RELEASE_BINDING_KEYS = frozenset({
    "code_sha256",
    "schema_sha256",
    "launcher_sha256",
    "snapshot_mode",
    "reference_code_sha256",
    "reference_contract_sha256",
    "reference_result_sha256",
})
VERIFIER_RECEIPT_KEYS = frozenset({
    "schema_version",
    "verifier_implementation",
    "status",
    "classification",
    "causal_signal_admission",
    "release_evidence_eligible",
    "admission_eligible",
    "detector_replay_receipt_required",
    "authority",
    "oracle_root_sha256",
    "oracle_manifest_sha256",
    "oracle_manifest_size_bytes",
    "oracle_ledger_sha256",
    "oracle_ledger_size_bytes",
    "expected_canonical_ledger_sha256",
    "oracle_ledger_terminal_hash",
    "raw_source_manifest_sha256",
    "oracle_request_sha256",
    "oracle_release_content_binding",
    "oracle_execution_provenance_scope",
    "verifier_release_content_binding",
    "verifier_execution_provenance_scope",
    "input_artifact_sha256",
    "independently_rebuilt_ledger",
    "independently_rebuilt_metrics",
    "producer_result_or_metrics_used",
    "verified_oracle_metrics",
    "reference_engine_id",
    "reference_code_sha256",
    "reference_contract_sha256",
    "reference_result_sha256",
    "reference_input_root_sha256",
    "reference_journal_root_sha256",
    "reference_journal_transaction_count",
    "reference_all_transactions_balanced",
    "reference_economic_projection_sha256",
    "reference_accounting_diagnostics_only",
    "reference_n_eff_statistical_admission_allowed",
    "reference_direction_accuracy_profit_gate_allowed",
    "terminal_inventory_mtm_jpy_micros",
    "external_orders",
    "anchor_status",
    "verifier_receipt_sha256",
})
REFERENCE_METRICS_KEYS = {
    "schema_version",
    "initial_equity_jpy_micros",
    "same_signal_ids_all_arms",
    "all_proposals_have_all_arm_dispositions",
    "common_gross_reference_shared",
    "arms",
    "external_orders",
    "terminal_inventory_mtm_jpy_micros",
    "metrics_sha256",
}
REFERENCE_ARM_METRIC_KEYS = frozenset({
    "proposal_count",
    "executed_count",
    "disposition_counts",
    "signal_id_set_sha256",
    "common_gross_pnl_jpy_micros",
    "realized_cost_jpy_micros",
    "fill_sizing_drag_jpy_micros",
    "latency_spread_slippage_drag_jpy_micros",
    "direct_commission_financing_cost_jpy_micros",
    "admission_opportunity_drag_jpy_micros",
    "total_execution_and_admission_drag_jpy_micros",
    "net_pnl_jpy_micros",
    "ending_equity_jpy_micros",
    "ending_equity_multiple",
    "direction_accuracy",
    "max_drawdown_jpy_micros",
    "max_drawdown_ratio",
    "cvar_tail_bps",
    "cluster_cvar_jpy_micros",
    "cluster_cvar_return",
    "currency_time_cluster_n_eff",
    "currency_time_cluster_observations",
    "monthly",
    "max_gross_notional_jpy_micros",
    "minimum_marked_equity_jpy_micros",
    "maximum_required_margin_jpy_micros",
    "minimum_free_margin_jpy_micros",
    "margin_guard_pass",
    "terminal_open_positions",
    "terminal_inventory_mtm_jpy_micros",
})
REFERENCE_CLUSTER_OBSERVATION_KEYS = frozenset({
    "cluster_id",
    "time_bucket",
    "currency_nodes",
    "source_signal_set_sha256",
    "ledger_net_pnl_jpy_micros",
    "cluster_risk_net_pnl_jpy_micros",
    "signed_return",
})
REFERENCE_MONTHLY_OBSERVATION_KEYS = frozenset({
    "month_id",
    "comparable_full_month",
    "segment_start_ts_ns",
    "segment_end_ts_ns",
    "start_equity_jpy_micros",
    "end_equity_jpy_micros",
    "equity_multiple",
    "equity_multiple_status",
    "ruin_observed",
})
REFERENCE_ARM_INTEGER_FIELDS = (
    "common_gross_pnl_jpy_micros",
    "realized_cost_jpy_micros",
    "fill_sizing_drag_jpy_micros",
    "latency_spread_slippage_drag_jpy_micros",
    "direct_commission_financing_cost_jpy_micros",
    "admission_opportunity_drag_jpy_micros",
    "total_execution_and_admission_drag_jpy_micros",
    "net_pnl_jpy_micros",
    "ending_equity_jpy_micros",
    "max_drawdown_jpy_micros",
    "cvar_tail_bps",
    "cluster_cvar_jpy_micros",
    "max_gross_notional_jpy_micros",
    "minimum_marked_equity_jpy_micros",
    "maximum_required_margin_jpy_micros",
    "minimum_free_margin_jpy_micros",
)
RATIO_TEXT_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)\.[0-9]{18}$")
MONTH_ID_RE = re.compile(r"^[0-9]{4}-(?:0[1-9]|1[0-2])$")
CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
SUPPORTED_REFERENCE_RELEASE = {
    "code_sha256": "cbac8e308bc11cd334f1cd23d23e4e75019074c1bdcfb66873b9254e3d6d520f",
    "contract_sha256": "276c34f4174a15d188406ef870d86a8d0bcbbc1b64b1f45381a033e20eb5d8f5",
}


class VerificationError(RuntimeError):
    """Fail-closed verification failure."""


class LockIdentityError(VerificationError):
    """The live lock pathname no longer names the held locked inode."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _snapshot_regular_file(path: Path) -> bytes:
    parent_fd = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        descriptor = os.open(
            path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise VerificationError("runtime snapshot is not regular")
            if before.st_size > MAX_JSON_BYTES:
                raise VerificationError("runtime snapshot exceeds fixed byte limit")
            chunks: list[bytes] = []
            offset = 0
            while offset < before.st_size:
                chunk = os.pread(
                    descriptor, min(1024 * 1024, before.st_size - offset), offset
                )
                if not chunk:
                    raise VerificationError("runtime snapshot truncated")
                chunks.append(chunk)
                offset += len(chunk)
            if os.pread(descriptor, 1, before.st_size):
                raise VerificationError("runtime snapshot grew")
            after = os.fstat(descriptor)
            identity = lambda item: (
                item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns,
                item.st_ctime_ns, item.st_nlink
            )
            if identity(before) != identity(after):
                raise VerificationError("runtime snapshot changed during read")
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)


if _SEALED_RUNTIME:
    try:
        _MODULE_CODE_BYTES = _INJECTED_RUNTIME_CODE_BYTES
        _SCHEMA_BYTES = _SEALED_SCHEMA_BYTES
        _LAUNCHER_SHA256 = _SEALED_LAUNCHER_SHA256
    except NameError as error:
        raise VerificationError("sealed verifier runtime injection is incomplete") from error
    if not all(type(value) is bytes for value in (_MODULE_CODE_BYTES, _SCHEMA_BYTES)) \
            or type(_LAUNCHER_SHA256) is not str \
            or SHA256_RE.fullmatch(_LAUNCHER_SHA256) is None:
        raise VerificationError("sealed verifier runtime injection is malformed")
    _MODULE_PATH = None
    _EXECUTION_SNAPSHOT_MODE = "SEALED_FD_COMPILE_EXEC_V2"
else:
    _MODULE_PATH = Path(__file__).resolve()
    _MODULE_CODE_BYTES = _snapshot_regular_file(_MODULE_PATH)
    _SCHEMA_BYTES = _snapshot_regular_file(_MODULE_PATH.parent / VERIFIER_SCHEMA_NAME)
    _LAUNCHER_SHA256 = None
    _EXECUTION_SNAPSHOT_MODE = "PATH_LOADED_TEST_ADAPTER_NOT_RELEASE_EVIDENCE"
_RENAME_EXCLUSIVE = None
_MODULE_CODE_SHA256 = sha256_bytes(_MODULE_CODE_BYTES)
_SCHEMA_SHA256 = sha256_bytes(_SCHEMA_BYTES)
SUPPORTED_ORACLE_RELEASE = {
    "code_sha256": "3c7a059576714e67cbf92e5689c2e88e1f7c600c62dde686a6fa63fb2b7a82c5",
    "contract_sha256": "abbdd484354f86a48c8c001fc8521f0cd96ab8aec6b46023f15bcb4502cf4467",
    "schema_sha256": "641c1b8ee69827e078fbcb49cb8f1bc54d59d03fa33f56b42550b8587f99e841",
}
def _assert_canonical_value(value: Any, location: str = "root") -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _assert_canonical_value(item, f"{location}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise VerificationError(f"non-text key at {location}")
            _assert_canonical_value(item, f"{location}.{key}")
        return
    raise VerificationError(f"non-canonical JSON type at {location}")


def canonical_bytes(value: Any) -> bytes:
    _assert_canonical_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def embedded_hash(payload: Mapping[str, Any], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return sha256_bytes(canonical_bytes(unsigned))


def _reject_float(_: str) -> Any:
    raise VerificationError("floating-point JSON number forbidden")


def _strict_int(token: str) -> int:
    if token == "-0":
        raise VerificationError("negative zero forbidden")
    return int(token)


def _pairs_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"duplicate JSON key forbidden: {key}")
        result[key] = value
    return result


def strict_json(data: bytes, label: str, *, require_lf: bool = True) -> dict[str, Any]:
    if len(data) > MAX_JSON_BYTES:
        raise VerificationError(f"{label} exceeds byte limit")
    if data.startswith(b"\xef\xbb\xbf"):
        raise VerificationError(f"{label} BOM forbidden")
    body = data
    if require_lf:
        if not data.endswith(b"\n") or data.endswith(b"\n\n"):
            raise VerificationError(f"{label} must have one terminal LF")
        body = data[:-1]
    try:
        value = json.loads(
            body.decode("utf-8", errors="strict"),
            object_pairs_hook=_pairs_object,
            parse_int=_strict_int,
            parse_float=_reject_float,
            parse_constant=_reject_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"invalid {label} JSON") from error
    if type(value) is not dict or canonical_bytes(value) != body:
        raise VerificationError(f"{label} is not one canonical JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise VerificationError(
            f"{label} schema mismatch missing={sorted(expected - set(value))} "
            f"extra={sorted(set(value) - expected)}"
        )


def _integer(value: Any, label: str, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise VerificationError(f"{label} must be integer")
    if minimum is not None and value < minimum:
        raise VerificationError(f"{label} below minimum")
    return value


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise VerificationError(f"{label} must be boolean")
    return value


def _validate_authority_exact(value: Any, label: str) -> None:
    if type(value) is not dict:
        raise VerificationError(f"{label} must be object")
    expected_items = _authority_items()
    _exact_keys(value, {key for key, _ in expected_items}, label)
    for key, expected in expected_items:
        actual = value[key]
        if type(expected) is bool:
            if type(actual) is not bool or actual is not expected:
                raise VerificationError(f"{label}.{key} exact boolean mismatch")
        elif type(expected) is int:
            if type(actual) is not int or actual != expected:
                raise VerificationError(f"{label}.{key} exact integer mismatch")
        else:  # pragma: no cover - fixed constant schema guard
            raise VerificationError("unsupported authority contract type")


def _text(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise VerificationError(f"{label} must be nonempty text")
    return value


def _digest(value: Any, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise VerificationError(f"{label} must be SHA-256")
    return value


def _validate_embedded(payload: Mapping[str, Any], field: str, label: str) -> None:
    if _digest(payload.get(field), f"{label}.{field}") != embedded_hash(payload, field):
        raise VerificationError(f"{label} embedded hash mismatch")


def _relative_parts(value: Any, label: str) -> tuple[str, ...]:
    if type(value) is not str or not value or len(value) > 512 \
            or value.startswith("/") or "//" in value or value.endswith("/"):
        raise VerificationError(f"{label} relative path invalid")
    parts = tuple(value.split("/"))
    if any(part in {"", ".", ".."} or SAFE_COMPONENT_RE.fullmatch(part) is None for part in parts):
        raise VerificationError(f"{label} relative path unsafe")
    return parts


def _validate_dirfd(directory_fd: int, label: str) -> os.stat_result:
    info = os.fstat(directory_fd)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o022:
        raise VerificationError(f"trusted {label} directory invalid")
    return info


def _assert_named_lock_identity(
    output_root_fd: int,
    lock_name: str,
    lock_fd: int,
) -> None:
    try:
        held = os.fstat(lock_fd)
        named = os.stat(lock_name, dir_fd=output_root_fd, follow_symlinks=False)
        access_mode = fcntl.fcntl(lock_fd, fcntl.F_GETFL) & os.O_ACCMODE
    except OSError as error:
        raise LockIdentityError("verifier lock pathname identity changed") from error
    for info in (held, named):
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() \
                or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600:
            raise LockIdentityError("verifier lock pathname identity changed")
    if access_mode != os.O_RDWR or (held.st_dev, held.st_ino) != (
        named.st_dev,
        named.st_ino,
    ):
        raise LockIdentityError("verifier lock pathname identity changed")


def _read_fd_snapshot(
    descriptor: int,
    label: str,
    *,
    allow_unlinked_sealed_runtime: bool = False,
) -> bytes:
    # The launcher can safely retain a pinned runtime inode after its pathname
    # is replaced.  This exception is passed only for code/contract/schema FDs
    # whose bytes are independently matched to the frozen release; evidence
    # inputs still require one live link.
    before = os.fstat(descriptor)
    allowed_link_counts = {0, 1} if allow_unlinked_sealed_runtime else {1}
    if not stat.S_ISREG(before.st_mode) or before.st_nlink not in allowed_link_counts:
        raise VerificationError(f"{label} FD is not regular")
    if before.st_size > MAX_ARTIFACT_BYTES:
        raise VerificationError(f"{label} FD exceeds fixed byte limit")
    if fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY:
        raise VerificationError(f"{label} FD must be read-only")
    chunks: list[bytes] = []
    offset = 0
    while offset < before.st_size:
        chunk = os.pread(
            descriptor, min(1024 * 1024, before.st_size - offset), offset
        )
        if not chunk:
            raise VerificationError(f"{label} FD truncated")
        chunks.append(chunk)
        offset += len(chunk)
    if os.pread(descriptor, 1, before.st_size):
        raise VerificationError(f"{label} FD grew")
    after = os.fstat(descriptor)
    identity = lambda item: (
        item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns,
        item.st_ctime_ns, item.st_nlink
    )
    if identity(before) != identity(after):
        raise VerificationError(f"{label} FD changed")
    return b"".join(chunks)


def _read_relative(
    root_fd: int,
    relative_path: str,
    label: str,
    *,
    expected_size: int | None = None,
    max_bytes: int = MAX_ARTIFACT_BYTES,
) -> bytes:
    _validate_dirfd(root_fd, "input root")
    parts = _relative_parts(relative_path, label)
    current = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            next_fd = os.open(
                part,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=current,
            )
            info = os.fstat(next_fd)
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o022:
                os.close(next_fd)
                raise VerificationError(f"{label} parent invalid")
            os.close(current)
            current = next_fd
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
            dir_fd=current,
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid() \
                    or before.st_mode & 0o022 or before.st_nlink != 1:
                raise VerificationError(f"{label} artifact invalid")
            if before.st_size > max_bytes:
                raise VerificationError(f"{label} artifact exceeds fixed byte limit")
            if expected_size is not None and before.st_size != expected_size:
                raise VerificationError(f"{label} declared size differs before read")
            chunks: list[bytes] = []
            offset = 0
            while offset < before.st_size:
                chunk = os.pread(
                    descriptor, min(1024 * 1024, before.st_size - offset), offset
                )
                if not chunk:
                    raise VerificationError(f"{label} artifact truncated")
                chunks.append(chunk)
                offset += len(chunk)
            if os.pread(descriptor, 1, before.st_size):
                raise VerificationError(f"{label} artifact grew")
            after = os.fstat(descriptor)
            identity = lambda item: (
                item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns,
                item.st_ctime_ns, item.st_nlink
            )
            if identity(before) != identity(after):
                raise VerificationError(f"{label} artifact changed")
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    finally:
        os.close(current)


def _artifact_bytes(spec: Mapping[str, Any], label: str, root_fd: int) -> bytes:
    _exact_keys(spec, {"artifact_id", "relative_path", "sha256", "size_bytes"}, label)
    if spec.get("artifact_id") != label:
        raise VerificationError(f"{label} artifact identity mismatch")
    expected_size = _integer(spec.get("size_bytes"), f"{label}.size_bytes", 0)
    expected_hash = _digest(spec.get("sha256"), f"{label}.sha256")
    data = _read_relative(
        root_fd,
        spec.get("relative_path"),
        label,
        expected_size=expected_size,
        max_bytes=(
            MAX_ARTIFACT_BYTES
            if label in {"source_blob", "oracle_ledger"}
            else MAX_JSON_BYTES
        ),
    )
    if len(data) != expected_size or sha256_bytes(data) != expected_hash:
        raise VerificationError(f"{label} exact-byte binding mismatch")
    return data


def _validate_bound_artifact_blob(
    spec: Any,
    label: str,
    data: Any,
) -> bytes:
    if type(spec) is not dict:
        raise VerificationError(f"{label} artifact specification invalid")
    _exact_keys(spec, {"artifact_id", "relative_path", "sha256", "size_bytes"}, label)
    if spec.get("artifact_id") != label:
        raise VerificationError(f"{label} artifact identity mismatch")
    _relative_parts(spec.get("relative_path"), f"{label}.relative_path")
    expected_size = _integer(spec.get("size_bytes"), f"{label}.size_bytes", 0)
    expected_hash = _digest(spec.get("sha256"), f"{label}.sha256")
    if type(data) is not bytes:
        raise VerificationError(f"{label} artifact blob must be exact bytes")
    maximum = (
        MAX_ARTIFACT_BYTES
        if label in {"source_blob", "oracle_ledger"}
        else MAX_JSON_BYTES
    )
    if len(data) > maximum:
        raise VerificationError(f"{label} artifact exceeds fixed byte limit")
    if len(data) != expected_size or sha256_bytes(data) != expected_hash:
        raise VerificationError(f"{label} exact-byte binding mismatch")
    return data


def _decode_exact_tuple_values(
    value: Any,
    expected_keys: tuple[str, ...],
    expected_value_type: type,
    label: str,
) -> dict[str, Any]:
    if type(value) is not tuple or len(value) != len(expected_keys):
        raise VerificationError(f"{label} must be exact fixed tuple")
    decoded: dict[str, Any] = {}
    observed_keys: list[str] = []
    for index, item in enumerate(value):
        if type(item) is not tuple or len(item) != 2:
            raise VerificationError(f"{label}[{index}] must be exact key/value tuple")
        key, item_value = item
        if type(key) is not str or type(item_value) is not expected_value_type:
            raise VerificationError(f"{label}[{index}] type mismatch")
        observed_keys.append(key)
        if key in decoded:
            raise VerificationError(f"{label} duplicate key")
        decoded[key] = item_value
    if tuple(observed_keys) != expected_keys:
        raise VerificationError(f"{label} keys or ordering mismatch")
    return decoded


def _pair(instrument: Any) -> tuple[str, str]:
    if type(instrument) is not str or PAIR_RE.fullmatch(instrument) is None:
        raise VerificationError("invalid FX instrument")
    base, quote = instrument.split("_", 1)
    if base == quote:
        raise VerificationError("FX base and quote currencies must differ")
    return base, quote


def _validate_registry(payload: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    _exact_keys(payload, {"schema_version", "registry_id", "instruments", "registry_sha256"}, "registry")
    if _integer(payload["schema_version"], "registry schema") != 1 \
            or payload["registry_id"] != "FROZEN_FX_INSTRUMENT_REGISTRY_V1":
        raise VerificationError("instrument registry identity mismatch")
    _validate_embedded(payload, "registry_sha256", "registry")
    instruments = payload["instruments"]
    if type(instruments) is not dict or not instruments or list(instruments) != sorted(instruments):
        raise VerificationError("instrument registry ordering invalid")
    result: dict[str, dict[str, int]] = {}
    economic_pairs: set[tuple[str, str]] = set()
    for instrument, spec in instruments.items():
        base, quote = _pair(instrument)
        economic_pair = tuple(sorted((base, quote)))
        if economic_pair in economic_pairs:
            raise VerificationError("registry contains inverse duplicate pair")
        economic_pairs.add(economic_pair)
        if type(spec) is not dict:
            raise VerificationError("instrument specification is not an object")
        _exact_keys(spec, {"price_scale", "pip_ticks"}, f"instrument {instrument}")
        scale = _integer(spec["price_scale"], f"{instrument}.price_scale", 1)
        pip_ticks = _integer(spec["pip_ticks"], f"{instrument}.pip_ticks", 1)
        if pip_ticks >= scale:
            raise VerificationError("invalid pip convention")
        result[instrument] = {"price_scale": scale, "pip_ticks": pip_ticks}
    return result


def _parse_source(
    blob: bytes,
    manifest: Mapping[str, Any],
    registry_payload: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, int]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    _exact_keys(
        manifest,
        {
            "schema_version", "source_bytes_sha256", "source_size_bytes", "event_count",
            "first_source_ts_ns", "last_source_ts_ns", "provider_allowlist",
            "instrument_registry_sha256", "stream_policies", "lossless", "manifest_sha256",
        },
        "source manifest",
    )
    if _integer(manifest["schema_version"], "source manifest schema") != 2:
        raise VerificationError("source manifest version mismatch")
    _validate_embedded(manifest, "manifest_sha256", "source manifest")
    if manifest["source_bytes_sha256"] != sha256_bytes(blob) \
            or _integer(manifest["source_size_bytes"], "source size", 0) != len(blob) \
            or manifest["instrument_registry_sha256"] != registry_payload["registry_sha256"]:
        raise VerificationError("source manifest exact-byte/registry binding mismatch")
    providers = manifest["provider_allowlist"]
    if type(providers) is not list or not providers or providers != sorted(set(providers)) \
            or any(type(provider) is not str or not provider for provider in providers):
        raise VerificationError("provider allowlist invalid")
    if _boolean(manifest["lossless"], "lossless") is not True:
        raise VerificationError("source must be lossless")
    raw_policies = manifest["stream_policies"]
    if type(raw_policies) is not list or not raw_policies:
        raise VerificationError("stream policies missing")
    policies: dict[tuple[str, str], dict[str, Any]] = {}
    policy_order: list[tuple[str, str]] = []
    for policy in raw_policies:
        if type(policy) is not dict:
            raise VerificationError("stream policy must be object")
        _exact_keys(
            policy,
            {
                "provider_id", "instrument", "sequence_required", "first_sequence",
                "last_sequence", "event_count", "max_source_gap_ns", "max_arrival_gap_ns",
            },
            "stream policy",
        )
        provider = _text(policy["provider_id"], "stream provider")
        instrument = _text(policy["instrument"], "stream instrument")
        _pair(instrument)
        if provider not in providers or instrument not in registry \
                or _boolean(policy["sequence_required"], "sequence_required") is not True:
            raise VerificationError("stream policy outside frozen feed contract")
        for field in ("first_sequence", "last_sequence", "event_count", "max_source_gap_ns", "max_arrival_gap_ns"):
            _integer(policy[field], f"stream.{field}", 1)
        key = (provider, instrument)
        if key in policies:
            raise VerificationError("duplicate stream policy")
        policies[key] = dict(policy)
        policy_order.append(key)
    if policy_order != sorted(policy_order):
        raise VerificationError("stream policy ordering invalid")
    if not blob or not blob.endswith(b"\n"):
        raise VerificationError("empty or truncated source blob")
    raw_lines = blob.splitlines(keepends=True)
    if len(raw_lines) > MAX_SOURCE_ROWS:
        raise VerificationError("source row limit exceeded")
    event_keys = {
        "schema_version", "provider_id", "instrument", "bid_ticks", "ask_ticks",
        "tick_scale", "source_ts_ns", "arrival_ts_ns", "provider_event_id", "sequence",
        "heartbeat", "quality_flags",
    }
    rows: list[dict[str, Any]] = []
    prefix = ZERO_SHA
    last_global: tuple[int, int, str, str, int] | None = None
    last_stream: dict[tuple[str, str], tuple[int, int, int]] = {}
    counts: Counter[tuple[str, str]] = Counter()
    for raw_line in raw_lines:
        row = strict_json(raw_line, "source event")
        _exact_keys(row, event_keys, "source event")
        if _integer(row["schema_version"], "source schema") != 1:
            raise VerificationError("source event version mismatch")
        provider = _text(row["provider_id"], "source provider")
        instrument = _text(row["instrument"], "source instrument")
        key = (provider, instrument)
        if key not in policies:
            raise VerificationError("source stream not allowlisted")
        _pair(instrument)
        for field in ("bid_ticks", "ask_ticks", "tick_scale", "source_ts_ns", "arrival_ts_ns", "sequence"):
            _integer(row[field], f"source.{field}", 1)
        if row["ask_ticks"] <= row["bid_ticks"] or row["arrival_ts_ns"] < row["source_ts_ns"] \
                or row["tick_scale"] != registry[instrument]["price_scale"]:
            raise VerificationError("invalid BBO/scale")
        if row["provider_event_id"] is not None and type(row["provider_event_id"]) is not str:
            raise VerificationError("provider event identity invalid")
        if _boolean(row["heartbeat"], "heartbeat") is not False \
                or type(row["quality_flags"]) is not list or row["quality_flags"]:
            raise VerificationError("non-executable source event")
        order = (row["arrival_ts_ns"], row["source_ts_ns"], provider, instrument, row["sequence"])
        if last_global is not None and order <= last_global:
            raise VerificationError("global arrival order reversal")
        last_global = order
        prior = last_stream.get(key)
        if prior is not None:
            if row["source_ts_ns"] <= prior[0] or row["arrival_ts_ns"] <= prior[1] \
                    or row["sequence"] != prior[2] + 1:
                raise VerificationError("stream chronology/sequence violation")
            frozen = policies[key]
            if row["source_ts_ns"] - prior[0] > frozen["max_source_gap_ns"] \
                    or row["arrival_ts_ns"] - prior[1] > frozen["max_arrival_gap_ns"]:
                raise VerificationError("stream gap violation")
        last_stream[key] = (row["source_ts_ns"], row["arrival_ts_ns"], row["sequence"])
        counts[key] += 1
        event_hash = sha256_bytes(raw_line)
        prefix = sha256_bytes(canonical_bytes({
            "previous_hash": prefix,
            "source_event_sha256": event_hash,
        }))
        rows.append({**row, "source_event_sha256": event_hash, "source_prefix_root_sha256": prefix})
    first_source_ts_ns = _integer(
        manifest["first_source_ts_ns"], "first source timestamp", 1
    )
    last_source_ts_ns = _integer(
        manifest["last_source_ts_ns"], "last source timestamp", 1
    )
    if _integer(manifest["event_count"], "event count", 1) != len(rows) \
            or first_source_ts_ns != min(row["source_ts_ns"] for row in rows) \
            or last_source_ts_ns != max(row["source_ts_ns"] for row in rows) \
            or set(policies) != set(counts):
        raise VerificationError("source manifest semantic mismatch")
    providers_by_instrument: defaultdict[str, set[str]] = defaultdict(set)
    for provider, instrument in policies:
        providers_by_instrument[instrument].add(provider)
    if any(len(items) != 1 for items in providers_by_instrument.values()):
        raise VerificationError("multiple providers for one instrument are ambiguous")
    for key, policy in policies.items():
        stream = [row for row in rows if (row["provider_id"], row["instrument"]) == key]
        if policy["first_sequence"] != stream[0]["sequence"] \
                or policy["last_sequence"] != stream[-1]["sequence"] \
                or policy["event_count"] != len(stream):
            raise VerificationError("stream manifest count/sequence mismatch")
    books: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        books[row["instrument"]].append(row)
    return rows, dict(books)


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _reject_producer_fields(value: Any, location: str = "proposal") -> None:
    if type(value) is dict:
        for key, item in value.items():
            if _normalize_key(key) in FORBIDDEN_PROPOSAL_TOKENS:
                raise VerificationError(f"producer outcome field forbidden at {location}.{key}")
            _reject_producer_fields(item, f"{location}.{key}")
    elif type(value) is list:
        for index, item in enumerate(value):
            _reject_producer_fields(item, f"{location}[{index}]")


def _validate_proposal(payload: Mapping[str, Any], source_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _reject_producer_fields(payload)
    _exact_keys(payload, {"schema_version", "candidate_key", "provenance", "rows", "proposal_sha256"}, "proposal")
    if _integer(payload["schema_version"], "proposal schema") != 2:
        raise VerificationError("proposal version mismatch")
    _validate_embedded(payload, "proposal_sha256", "proposal")
    _text(payload["candidate_key"], "candidate key")
    provenance = payload["provenance"]
    if type(provenance) is not dict:
        raise VerificationError("proposal provenance invalid")
    _exact_keys(
        provenance,
        {"detector_code_sha256", "detector_policy_sha256", "generator_policy_sha256", "source_acquisition_contract_sha256"},
        "proposal provenance",
    )
    for key, digest in provenance.items():
        _digest(digest, f"proposal provenance {key}")
    raw_rows = payload["rows"]
    if type(raw_rows) is not list or not raw_rows:
        raise VerificationError("proposal rows missing")
    row_keys = {
        "proposal_ordinal", "decision_source_ts_ns", "decision_arrival_ts_ns", "available_at_ns",
        "decision_source_event_sha256", "completed_data_watermark_source_ts_ns",
        "completed_data_prefix_root_sha256", "instrument", "direction", "notional_jpy_micros",
        "max_age_ns", "worker_key", "action",
    }
    by_hash = {row["source_event_sha256"]: row for row in source_rows}
    validated: list[dict[str, Any]] = []
    economic_lot_keys: set[str] = set()
    prior_order: tuple[int, int, int] | None = None
    for expected_ordinal, row in enumerate(raw_rows, 1):
        if type(row) is not dict:
            raise VerificationError("proposal row invalid")
        _exact_keys(row, row_keys, "proposal row")
        for field in (
            "proposal_ordinal", "decision_source_ts_ns", "decision_arrival_ts_ns", "available_at_ns",
            "completed_data_watermark_source_ts_ns", "direction", "notional_jpy_micros", "max_age_ns",
        ):
            _integer(row[field], f"proposal.{field}")
        if row["proposal_ordinal"] != expected_ordinal or row["direction"] not in {-1, 1} \
                or row["notional_jpy_micros"] <= 0 or row["max_age_ns"] <= 0 \
                or row["action"] != "ENTER" or row["available_at_ns"] != row["decision_arrival_ts_ns"] \
                or row["decision_arrival_ts_ns"] < row["decision_source_ts_ns"]:
            raise VerificationError("proposal chronology/value invalid")
        instrument = _text(row["instrument"], "proposal instrument")
        _pair(instrument)
        _text(row["worker_key"], "worker key")
        decision_hash = _digest(row["decision_source_event_sha256"], "decision event hash")
        prefix_hash = _digest(row["completed_data_prefix_root_sha256"], "proposal prefix hash")
        available = [event for event in source_rows if event["arrival_ts_ns"] <= row["decision_arrival_ts_ns"]]
        if not available:
            raise VerificationError("proposal has no available prefix")
        watermark = max(event["source_ts_ns"] for event in available)
        decision = by_hash.get(decision_hash)
        if row["completed_data_watermark_source_ts_ns"] != watermark \
                or prefix_hash != available[-1]["source_prefix_root_sha256"] \
                or decision is None or decision not in available \
                or decision["source_ts_ns"] != row["decision_source_ts_ns"] \
                or decision["instrument"] != instrument or row["decision_source_ts_ns"] > watermark:
            raise VerificationError("proposal prefix/decision binding mismatch")
        order = (row["decision_arrival_ts_ns"], row["decision_source_ts_ns"], row["proposal_ordinal"])
        if prior_order is not None and order <= prior_order:
            raise VerificationError("proposal input order reversal")
        prior_order = order
        economic_lot_key = sha256_bytes(canonical_bytes({
            key: row[key]
            for key in sorted(row_keys - {"proposal_ordinal"})
        }))
        if economic_lot_key in economic_lot_keys:
            raise VerificationError("duplicate economic-lot ticket partition forbidden")
        economic_lot_keys.add(economic_lot_key)
        validated.append(dict(row))
    return {**dict(payload), "rows": validated}


def _validate_policy(payload: Mapping[str, Any], policy_id: str, hash_field: str) -> None:
    if _integer(payload.get("schema_version"), f"{policy_id} schema") != 2 \
            or payload.get("policy_id") != policy_id:
        raise VerificationError(f"{policy_id} identity mismatch")
    _validate_embedded(payload, hash_field, policy_id)


def _month_bounds_ns(month_id: str) -> tuple[int, int]:
    if type(month_id) is not str or re.fullmatch(r"[0-9]{4}-(?:0[1-9]|1[0-2])", month_id) is None:
        raise VerificationError("month identifier invalid")
    start = datetime(
        int(month_id[:4]), int(month_id[5:]), 1, tzinfo=timezone.utc
    )
    end = start.replace(year=start.year + 1, month=1) if start.month == 12 else start.replace(month=start.month + 1)
    return int(start.timestamp()) * 1_000_000_000, int(end.timestamp()) * 1_000_000_000


def _all_months(start_ns: int, end_ns: int) -> list[str]:
    if start_ns >= end_ns:
        return []
    current = datetime.fromtimestamp(start_ns // 1_000_000_000, tz=timezone.utc).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )
    result: list[str] = []
    while int(current.timestamp()) * 1_000_000_000 < end_ns:
        result.append(f"{current.year:04d}-{current.month:02d}")
        current = current.replace(year=current.year + 1, month=1) if current.month == 12 else current.replace(month=current.month + 1)
    return result


def _complete_months(start_ns: int, end_ns: int) -> list[str]:
    return [
        month for month in _all_months(start_ns, end_ns)
        if _month_bounds_ns(month)[0] >= start_ns and _month_bounds_ns(month)[1] <= end_ns
    ]


def _validate_policies(
    execution: Mapping[str, Any],
    inventory: Mapping[str, Any],
    accounting: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> None:
    _validate_policy(execution, "FROZEN_EXECUTION_POLICY_V2", "execution_policy_sha256")
    _exact_keys(execution, {"schema_version", "policy_id", "arms", "max_trade_quote_staleness_ns", "execution_policy_sha256"}, "execution policy")
    _integer(execution["max_trade_quote_staleness_ns"], "trade staleness", 1)
    arms = execution["arms"]
    if type(arms) is not dict or set(arms) != set(ARMS):
        raise VerificationError("execution arms mismatch")
    cost_fields = ("latency_ns", "slippage_micropips_per_side", "commission_ppm_per_side", "financing_ppm_per_day")
    for arm in ARMS:
        spec = arms[arm]
        if type(spec) is not dict:
            raise VerificationError("execution arm invalid")
        _exact_keys(spec, {*cost_fields, "raw_mid"}, f"execution arm {arm}")
        for field in cost_fields:
            _integer(spec[field], f"{arm}.{field}", 0)
        _boolean(spec["raw_mid"], f"{arm}.raw_mid")
    raw, base, adverse = arms[ARMS[0]], arms[ARMS[1]], arms[ARMS[2]]
    if raw["raw_mid"] is not True or any(raw[field] != 0 for field in cost_fields) \
            or base["raw_mid"] is not False or adverse["raw_mid"] is not False \
            or any(adverse[field] < base[field] for field in cost_fields) \
            or not any(adverse[field] > base[field] for field in cost_fields):
        raise VerificationError("RAW/BASE/ADVERSE ordering invalid")
    _validate_policy(inventory, "FROZEN_INVENTORY_POLICY_V2", "inventory_policy_sha256")
    _exact_keys(
        inventory,
        {"schema_version", "policy_id", "max_gross_notional_jpy_micros", "max_currency_notional_jpy_micros", "max_open_positions", "same_pair_collision", "terminal_liquidation", "inventory_policy_sha256"},
        "inventory policy",
    )
    for field in ("max_gross_notional_jpy_micros", "max_currency_notional_jpy_micros", "max_open_positions"):
        _integer(inventory[field], f"inventory.{field}", 1)
    if inventory["same_pair_collision"] != "REJECT_NEW" \
            or _boolean(inventory["terminal_liquidation"], "terminal liquidation") is not True:
        raise VerificationError("inventory policy invalid")
    _validate_policy(accounting, "FROZEN_ACCOUNTING_POLICY_V2", "accounting_policy_sha256")
    _exact_keys(
        accounting,
        {"schema_version", "policy_id", "jpy_micros_per_yen", "base_microunits_per_unit", "max_conversion_staleness_ns", "supported_quote_currencies", "asset_conversion_side", "liability_conversion_side", "positive_cost_rounding", "accounting_policy_sha256"},
        "accounting policy",
    )
    if _integer(accounting["jpy_micros_per_yen"], "JPY micros") != JPY_MICROS_PER_YEN \
            or _integer(accounting["base_microunits_per_unit"], "base microunits") != BASE_MICROUNITS_PER_UNIT \
            or _integer(accounting["max_conversion_staleness_ns"], "conversion staleness", 1) <= 0 \
            or accounting["supported_quote_currencies"] != ["CAD", "CHF", "JPY", "USD"] \
            or accounting["asset_conversion_side"] != "BID" \
            or accounting["liability_conversion_side"] != "ASK" \
            or accounting["positive_cost_rounding"] != "CEILING":
        raise VerificationError("accounting policy invalid")
    _validate_policy(evaluation, "FROZEN_EVALUATION_POLICY_V2", "evaluation_policy_sha256")
    _exact_keys(
        evaluation,
        {"schema_version", "policy_id", "period_start_ts_ns", "period_end_ts_ns", "initial_equity_jpy_micros", "margin_notional_cap_jpy_micros", "margin_rate_bps", "max_gross_to_equity_bps", "cvar_tail_bps", "cluster_window_ns", "full_month_ids", "holdout_state", "evaluation_policy_sha256"},
        "evaluation policy",
    )
    start = _integer(evaluation["period_start_ts_ns"], "period start", 1)
    end = _integer(evaluation["period_end_ts_ns"], "period end", 1)
    if start >= end:
        raise VerificationError("evaluation period invalid")
    for field in ("initial_equity_jpy_micros", "margin_notional_cap_jpy_micros", "margin_rate_bps", "max_gross_to_equity_bps", "cvar_tail_bps", "cluster_window_ns"):
        _integer(evaluation[field], f"evaluation.{field}", 1)
    if evaluation["margin_rate_bps"] > 10_000 or evaluation["cvar_tail_bps"] > 10_000 \
            or evaluation["holdout_state"] != "UNOPENED" \
            or type(evaluation["full_month_ids"]) is not list \
            or evaluation["full_month_ids"] != _complete_months(start, end):
        raise VerificationError("evaluation month/risk/holdout policy invalid")
    _validate_policy(authority, "FROZEN_PAPER_AUTHORITY_V1", "authority_policy_sha256")
    authority_keys = {key for key, _ in _authority_items()}
    _exact_keys(
        authority,
        {"schema_version", "policy_id", *authority_keys, "authority_policy_sha256"},
        "authority policy",
    )
    _validate_authority_exact(
        {key: authority[key] for key in authority_keys}, "paper authority"
    )


def _expected_oracle_intent(
    request_sha256: str,
    code_sha256: str,
    contract_sha256: str,
    schema_sha256: str,
) -> dict[str, Any]:
    transaction_id = sha256_bytes(canonical_bytes({
        "request_sha256": request_sha256,
        "code_sha256": code_sha256,
        "contract_sha256": contract_sha256,
        "schema_sha256": schema_sha256,
    }))
    return {
        "schema_version": 1,
        "transaction_id": transaction_id,
        "request_sha256": request_sha256,
        "code_sha256": code_sha256,
        "contract_sha256": contract_sha256,
        "schema_sha256": schema_sha256,
    }


def _validate_oracle_transaction(
    blobs: Mapping[str, bytes],
    hashes: Mapping[str, str],
    release: Mapping[str, Any],
) -> None:
    request_sha = hashes["oracle_request"]
    expected_intent = _expected_oracle_intent(
        request_sha,
        release["hashes"]["code_sha256"],
        release["hashes"]["contract_sha256"],
        release["hashes"]["schema_sha256"],
    )
    intent_bytes = blobs["oracle_intent"]
    if intent_bytes != canonical_bytes(expected_intent) + b"\n":
        raise VerificationError("oracle intent is detached from frozen request/release")
    intent = strict_json(intent_bytes, "oracle intent")
    if intent != expected_intent:
        raise VerificationError("oracle intent semantic mismatch")
    commit = strict_json(blobs["oracle_commit"], "oracle commit")
    _exact_keys(
        commit,
        {
            "schema_version", "transaction_id", "request_sha256", "intent_sha256",
            "ledger_sha256", "ledger_size_bytes", "manifest_sha256",
            "manifest_size_bytes", "terminal_hash",
        },
        "oracle commit",
    )
    manifest = strict_json(blobs["oracle_manifest"], "oracle manifest")
    if _integer(commit["schema_version"], "oracle commit schema") != 1 \
            or commit["transaction_id"] != expected_intent["transaction_id"] \
            or commit["request_sha256"] != request_sha \
            or commit["intent_sha256"] != sha256_bytes(intent_bytes) \
            or commit["ledger_sha256"] != sha256_bytes(blobs["oracle_ledger"]) \
            or commit["ledger_size_bytes"] != len(blobs["oracle_ledger"]) \
            or commit["manifest_sha256"] != sha256_bytes(blobs["oracle_manifest"]) \
            or commit["manifest_size_bytes"] != len(blobs["oracle_manifest"]) \
            or commit["terminal_hash"] != manifest.get("oracle_ledger_terminal_hash"):
        raise VerificationError("oracle COMMIT is detached or incomplete")


def _load_request(
    request: Mapping[str, Any],
    root_fd: int,
    trusted_release: Mapping[str, Any],
    trusted_reference_release: Mapping[str, Any],
) -> dict[str, Any]:
    _exact_keys(
        request,
        {"schema_version", *SEALED_ARTIFACT_ROLES, "output_directory"},
        "verifier request",
    )
    if _integer(request["schema_version"], "verifier request schema") != 2:
        raise VerificationError("verifier request version mismatch")
    blobs: dict[str, bytes] = {}
    for label in SEALED_ARTIFACT_ROLES:
        spec = request[label]
        if type(spec) is not dict:
            raise VerificationError(f"{label} artifact specification invalid")
        blobs[label] = _artifact_bytes(spec, label, root_fd)
    return _load_request_from_bound_blobs(
        request,
        blobs,
        trusted_release,
        trusted_reference_release,
    )


def _load_request_from_bound_blobs(
    request: Mapping[str, Any],
    supplied_blobs: Mapping[str, Any],
    trusted_release: Mapping[str, Any],
    trusted_reference_release: Mapping[str, Any],
) -> dict[str, Any]:
    _exact_keys(
        request,
        {"schema_version", *SEALED_ARTIFACT_ROLES, "output_directory"},
        "verifier request",
    )
    if _integer(request["schema_version"], "verifier request schema") != 2:
        raise VerificationError("verifier request version mismatch")
    if type(supplied_blobs) is not dict \
            or set(supplied_blobs) != set(SEALED_ARTIFACT_ROLES):
        raise VerificationError("sealed artifact blob role set mismatch")
    blobs: dict[str, bytes] = {}
    hashes: dict[str, str] = {}
    for label in SEALED_ARTIFACT_ROLES:
        blobs[label] = _validate_bound_artifact_blob(
            request[label],
            label,
            supplied_blobs[label],
        )
        hashes[label] = request[label]["sha256"]
    if type(trusted_release) is not dict:
        raise VerificationError("trusted Oracle release bundle invalid")
    _exact_keys(
        trusted_release,
        {"code_bytes", "contract_bytes", "schema_bytes", "hashes"},
        "trusted Oracle release",
    )
    if type(trusted_reference_release) is not dict:
        raise VerificationError("trusted reference release bundle invalid")
    _exact_keys(
        trusted_reference_release,
        {"code_bytes", "contract_bytes", "hashes"},
        "trusted reference release",
    )
    release_bindings = {
        "oracle_code_snapshot": "code_bytes",
        "oracle_contract_snapshot": "contract_bytes",
        "oracle_schema_snapshot": "schema_bytes",
    }
    for artifact_label, release_label in release_bindings.items():
        if blobs[artifact_label] != trusted_release[release_label]:
            raise VerificationError(
                f"{artifact_label} differs from trusted Oracle release FD"
            )
    if trusted_release["hashes"] != SUPPORTED_ORACLE_RELEASE:
        raise VerificationError("trusted Oracle release is not the verifier-pinned triplet")
    reference_release_bindings = {
        "reference_code_snapshot": "code_bytes",
        "reference_contract_snapshot": "contract_bytes",
    }
    for artifact_label, release_label in reference_release_bindings.items():
        if blobs[artifact_label] != trusted_reference_release[release_label]:
            raise VerificationError(
                f"{artifact_label} differs from trusted reference release FD"
            )
    if trusted_reference_release["hashes"] != SUPPORTED_REFERENCE_RELEASE:
        raise VerificationError("trusted reference release is not pinned")
    _validate_oracle_transaction(blobs, hashes, trusted_release)
    output_name = request["output_directory"]
    if type(output_name) is not str or SAFE_COMPONENT_RE.fullmatch(output_name) is None:
        raise VerificationError("verifier output directory invalid")
    oracle_request = strict_json(blobs["oracle_request"], "oracle request snapshot")
    shared = REFERENCE_INPUT_LABELS
    _exact_keys(oracle_request, {"schema_version", *shared, "output_directory"}, "oracle request snapshot")
    if _integer(oracle_request["schema_version"], "oracle request schema") != 2:
        raise VerificationError("oracle request snapshot version mismatch")
    for label in shared:
        if oracle_request[label] != request[label]:
            raise VerificationError(f"oracle/verifier input artifact mismatch: {label}")
    registry_payload = strict_json(blobs["instrument_registry"], "instrument registry")
    registry = _validate_registry(registry_payload)
    source_manifest = strict_json(blobs["source_manifest"], "source manifest")
    source_rows, books = _parse_source(blobs["source_blob"], source_manifest, registry_payload, registry)
    proposal = _validate_proposal(strict_json(blobs["proposal"], "proposal"), source_rows)
    execution = strict_json(blobs["execution_policy"], "execution policy")
    inventory = strict_json(blobs["inventory_policy"], "inventory policy")
    accounting = strict_json(blobs["accounting_policy"], "accounting policy")
    evaluation = strict_json(blobs["evaluation_policy"], "evaluation policy")
    authority = strict_json(blobs["authority_policy"], "authority policy")
    _validate_policies(execution, inventory, accounting, evaluation, authority)
    if any(row["instrument"] not in registry for row in proposal["rows"]):
        raise VerificationError("proposal instrument outside registry")
    if any(
        row["decision_arrival_ts_ns"] < evaluation["period_start_ts_ns"]
        or row["decision_arrival_ts_ns"] >= evaluation["period_end_ts_ns"]
        for row in proposal["rows"]
    ):
        raise VerificationError("proposal outside evaluation period")
    return {
        "blobs": blobs,
        "hashes": hashes,
        "output_name": output_name,
        "oracle_request": oracle_request,
        "source_rows": source_rows,
        "books": books,
        "source_manifest": source_manifest,
        "registry_payload": registry_payload,
        "registry": registry,
        "proposal": proposal,
        "execution": execution,
        "inventory": inventory,
        "accounting": accounting,
        "evaluation": evaluation,
        "authority": authority,
        "oracle_release": trusted_release,
        "reference_release": trusted_reference_release,
    }


def _market_price(event: Mapping[str, Any], side: str) -> Fraction:
    scale = event["tick_scale"]
    if side == "bid":
        return Fraction(event["bid_ticks"], scale)
    if side == "ask":
        return Fraction(event["ask_ticks"], scale)
    if side == "mid":
        return Fraction(event["bid_ticks"] + event["ask_ticks"], scale * 2)
    raise VerificationError("unknown executable side")


def _execution_price(
    event: Mapping[str, Any],
    direction: int,
    opening: bool,
    policy: Mapping[str, Any],
    instrument_spec: Mapping[str, int],
) -> tuple[Fraction, int, int]:
    if policy["raw_mid"] is True:
        numerator = (event["bid_ticks"] + event["ask_ticks"]) * PRICE_SUBPIP_SCALE
        denominator = 2 * event["tick_scale"] * PRICE_SUBPIP_SCALE
    else:
        is_buy = (opening and direction > 0) or (not opening and direction < 0)
        ticks = event["ask_ticks"] if is_buy else event["bid_ticks"]
        slippage = policy["slippage_micropips_per_side"] * instrument_spec["pip_ticks"]
        numerator = ticks * PRICE_SUBPIP_SCALE + (slippage if is_buy else -slippage)
        denominator = event["tick_scale"] * PRICE_SUBPIP_SCALE
    if numerator <= 0:
        raise VerificationError("execution price nonpositive")
    return Fraction(numerator, denominator), numerator, denominator


def _latest_causal(
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    instrument: str,
    source_watermark_ns: int,
    arrival_cutoff_ns: int,
    max_staleness_ns: int,
) -> Mapping[str, Any]:
    candidates = [
        event for event in books.get(instrument, ())
        if event["source_ts_ns"] <= source_watermark_ns and event["arrival_ts_ns"] <= arrival_cutoff_ns
    ]
    if not candidates:
        raise VerificationError(f"missing causal quote for {instrument}")
    event = candidates[-1]
    if source_watermark_ns - event["source_ts_ns"] > max_staleness_ns \
            or arrival_cutoff_ns - event["arrival_ts_ns"] > max_staleness_ns \
            or arrival_cutoff_ns - event["source_ts_ns"] > max_staleness_ns:
        raise VerificationError(f"stale causal quote for {instrument}")
    return event


def _arrival_watermark_from_books(
    books: Mapping[str, Sequence[Mapping[str, Any]]], arrival_cutoff_ns: int
) -> int:
    available = [
        event["source_ts_ns"]
        for stream in books.values()
        for event in stream
        if event["arrival_ts_ns"] <= arrival_cutoff_ns
    ]
    if not available:
        raise VerificationError("no causal BBO at valuation arrival")
    return max(available)


def _jpy_value(
    amount: Fraction,
    currency: str,
    source_watermark_ns: int,
    arrival_cutoff_ns: int,
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    max_staleness_ns: int,
) -> Fraction:
    if amount == 0 or currency == "JPY":
        return amount
    if currency == "USD":
        quote = _latest_causal(books, "USD_JPY", source_watermark_ns, arrival_cutoff_ns, max_staleness_ns)
        return amount * _market_price(quote, "bid" if amount > 0 else "ask")
    if currency in {"CAD", "CHF"}:
        quote = _latest_causal(books, f"USD_{currency}", source_watermark_ns, arrival_cutoff_ns, max_staleness_ns)
        usd = amount / _market_price(quote, "ask" if amount > 0 else "bid")
        return _jpy_value(usd, "USD", source_watermark_ns, arrival_cutoff_ns, books, max_staleness_ns)
    raise VerificationError("unsupported quote currency")


def _currency_node_jpy_value(
    amount: Fraction,
    currency: str,
    source_watermark_ns: int,
    arrival_cutoff_ns: int,
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    max_staleness_ns: int,
    registry: Mapping[str, Mapping[str, int]],
) -> Fraction:
    if amount == 0 or currency == "JPY":
        return amount
    adjacency: defaultdict[tuple[str, str], list[tuple[str, str, Mapping[str, Any]]]] \
        = defaultdict(list)
    for instrument in sorted(registry):
        if instrument not in books:
            continue
        try:
            event = _latest_causal(
                books,
                instrument,
                source_watermark_ns,
                arrival_cutoff_ns,
                max_staleness_ns,
            )
        except VerificationError:
            continue
        base, quote = _pair(instrument)
        adjacency[(base, "OUT")].append((quote, "MULTIPLY", event))
        adjacency[(quote, "OUT")].append((base, "DIVIDE", event))
    completed: list[tuple[tuple[str, Mapping[str, Any]], ...]] = []
    pending = [(currency, frozenset({currency}), ())]
    while pending:
        node, visited, steps = pending.pop()
        if node == "JPY":
            completed.append(steps)
            continue
        for destination, operation, event in reversed(
            adjacency.get((node, "OUT"), ())
        ):
            if destination not in visited:
                pending.append((
                    destination,
                    visited | {destination},
                    steps + ((operation, event),),
                ))
    if len(completed) != 1:
        raise VerificationError("currency node JPY path must be uniquely causal")
    result = amount
    for operation, event in completed[0]:
        if operation == "MULTIPLY":
            result *= _market_price(event, "bid" if result > 0 else "ask")
        else:
            result /= _market_price(event, "ask" if result > 0 else "bid")
    return result


def _asset_micros(value_yen: Fraction) -> int:
    scaled = value_yen * JPY_MICROS_PER_YEN
    return scaled.numerator // scaled.denominator


def _cost_micros(value_micros: Fraction) -> int:
    if value_micros < 0:
        raise VerificationError("negative cost")
    return (value_micros.numerator + value_micros.denominator - 1) // value_micros.denominator


def _outward_currency_micros(value_yen: Fraction) -> int:
    scaled = value_yen * JPY_MICROS_PER_YEN
    if scaled >= 0:
        return (
            scaled.numerator + scaled.denominator - 1
        ) // scaled.denominator
    return scaled.numerator // scaled.denominator


def _scaled_ratio_text(scaled: int) -> str:
    prefix = "-" if scaled < 0 else ""
    magnitude = abs(scaled)
    return (
        f"{prefix}{magnitude // RATIO_DECIMAL_SCALE}."
        f"{magnitude % RATIO_DECIMAL_SCALE:018d}"
    )


def _ratio_text(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        raise VerificationError("ratio denominator nonpositive")
    return _scaled_ratio_text((numerator * RATIO_DECIMAL_SCALE) // denominator)


def _signed_ratio_text(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        raise VerificationError("signed ratio denominator invalid")
    return _scaled_ratio_text((numerator * RATIO_DECIMAL_SCALE) // denominator)


def _nonnegative_ratio_ceiling_text(numerator: int, denominator: int) -> str:
    if numerator < 0 or denominator <= 0:
        raise VerificationError("nonnegative ratio inputs invalid")
    scaled = (
        numerator * RATIO_DECIMAL_SCALE + denominator - 1
    ) // denominator
    return _scaled_ratio_text(scaled)


def _fresh(event: Mapping[str, Any], due_arrival_ns: int, max_staleness_ns: int) -> bool:
    return event["arrival_ts_ns"] >= due_arrival_ns \
        and event["arrival_ts_ns"] - event["source_ts_ns"] <= max_staleness_ns \
        and event["arrival_ts_ns"] - due_arrival_ns <= max_staleness_ns


def _entry_event(
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    proposal: Mapping[str, Any],
    latency_ns: int,
    period_end_ns: int,
    max_staleness_ns: int,
) -> Mapping[str, Any] | None:
    due = proposal["decision_arrival_ts_ns"] + latency_ns
    for event in books.get(proposal["instrument"], ()):
        if event["source_ts_ns"] <= proposal["decision_source_ts_ns"] or event["arrival_ts_ns"] < due:
            continue
        if event["arrival_ts_ns"] >= period_end_ns:
            return None
        if not _fresh(event, due, max_staleness_ns):
            raise VerificationError("first entry quote stale")
        return event
    return None


def _exit_event(
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    instrument: str,
    entry: Mapping[str, Any],
    due_arrival_ns: int,
    period_end_ns: int,
    max_staleness_ns: int,
) -> Mapping[str, Any] | None:
    for event in books.get(instrument, ()):
        if event["source_ts_ns"] < entry["source_ts_ns"] or event["arrival_ts_ns"] < due_arrival_ns:
            continue
        if event["arrival_ts_ns"] >= period_end_ns:
            return None
        if not _fresh(event, due_arrival_ns, max_staleness_ns):
            raise VerificationError("first exit quote stale")
        return event
    return None


def _terminal_event(
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    instrument: str,
    period_end_ns: int,
    max_staleness_ns: int,
) -> Mapping[str, Any]:
    candidates = [
        event for event in books.get(instrument, ())
        if event["source_ts_ns"] < period_end_ns and event["arrival_ts_ns"] < period_end_ns
    ]
    if not candidates:
        raise VerificationError("terminal quote missing")
    event = candidates[-1]
    cutoff = period_end_ns - 1
    if cutoff - event["source_ts_ns"] > max_staleness_ns \
            or cutoff - event["arrival_ts_ns"] > max_staleness_ns:
        raise VerificationError("terminal quote stale")
    return event


def _signal_id(candidate_key: str, proposal: Mapping[str, Any], provenance: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_bytes({
        "candidate_key": candidate_key,
        "proposal_ordinal": proposal["proposal_ordinal"],
        "decision_source_ts_ns": proposal["decision_source_ts_ns"],
        "decision_arrival_ts_ns": proposal["decision_arrival_ts_ns"],
        "decision_source_event_sha256": proposal["decision_source_event_sha256"],
        "completed_data_prefix_root_sha256": proposal["completed_data_prefix_root_sha256"],
        "instrument": proposal["instrument"],
        "direction": proposal["direction"],
        "notional_jpy_micros": proposal["notional_jpy_micros"],
        "max_age_ns": proposal["max_age_ns"],
        "worker_key": proposal["worker_key"],
        "detector_code_sha256": provenance["detector_code_sha256"],
        "detector_policy_sha256": provenance["detector_policy_sha256"],
        "generator_policy_sha256": provenance["generator_policy_sha256"],
    }))


def _economic_lot_id(
    candidate_key: str,
    proposal: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> str:
    return sha256_bytes(canonical_bytes({
        "candidate_key": candidate_key,
        "decision_source_ts_ns": proposal["decision_source_ts_ns"],
        "decision_arrival_ts_ns": proposal["decision_arrival_ts_ns"],
        "decision_source_event_sha256": proposal["decision_source_event_sha256"],
        "completed_data_prefix_root_sha256": proposal["completed_data_prefix_root_sha256"],
        "instrument": proposal["instrument"],
        "direction": proposal["direction"],
        "target_notional_jpy_micros": proposal["notional_jpy_micros"],
        "max_age_ns": proposal["max_age_ns"],
        "worker_key": proposal["worker_key"],
        "detector_code_sha256": provenance["detector_code_sha256"],
        "detector_policy_sha256": provenance["detector_policy_sha256"],
        "generator_policy_sha256": provenance["generator_policy_sha256"],
    }))


def _exact_position_risk_micros(
    *,
    direction: int,
    units_micros: int,
    price: Fraction,
    quote_currency: str,
    source_watermark_ns: int,
    arrival_cutoff_ns: int,
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    accounting: Mapping[str, Any],
    opening: bool,
) -> Fraction:
    cash_sign = -direction if opening else direction
    signed_quote = Fraction(
        cash_sign * units_micros, BASE_MICROUNITS_PER_UNIT
    ) * price
    signed_jpy = _jpy_value(
        signed_quote,
        quote_currency,
        source_watermark_ns,
        arrival_cutoff_ns,
        books,
        accounting["max_conversion_staleness_ns"],
    )
    return abs(signed_jpy * JPY_MICROS_PER_YEN)


def _outward_risk_micros(exact_jpy_micros: Fraction) -> int:
    return _cost_micros(exact_jpy_micros)


def _actual_fill_units(
    proposal: Mapping[str, Any],
    entry: Mapping[str, Any],
    entry_price: Fraction,
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    accounting: Mapping[str, Any],
) -> int:
    _, quote = _pair(proposal["instrument"])
    source_watermark = _arrival_watermark_from_books(books, entry["arrival_ts_ns"])
    one_base_micros = _exact_position_risk_micros(
        direction=proposal["direction"],
        units_micros=BASE_MICROUNITS_PER_UNIT,
        price=entry_price,
        quote_currency=quote,
        source_watermark_ns=source_watermark,
        arrival_cutoff_ns=entry["arrival_ts_ns"],
        books=books,
        accounting=accounting,
        opening=True,
    )
    if one_base_micros <= 0:
        raise VerificationError("position sizing conversion invalid")
    exact_units_micros = (
        Fraction(proposal["notional_jpy_micros"], 1)
        * BASE_MICROUNITS_PER_UNIT
        / one_base_micros
    )
    return max(0, exact_units_micros.numerator // exact_units_micros.denominator)


def _sized_units(
    proposal: Mapping[str, Any],
    entry: Mapping[str, Any],
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    accounting: Mapping[str, Any],
) -> int:
    _, quote = _pair(proposal["instrument"])
    common_price = _market_price(entry, "mid")
    entry_watermark = _arrival_watermark_from_books(books, entry["arrival_ts_ns"])
    jpy_per_base_micros = _exact_position_risk_micros(
        direction=proposal["direction"],
        units_micros=BASE_MICROUNITS_PER_UNIT,
        price=common_price,
        quote_currency=quote,
        source_watermark_ns=entry_watermark,
        arrival_cutoff_ns=entry["arrival_ts_ns"],
        books=books,
        accounting=accounting,
        opening=True,
    )
    if jpy_per_base_micros <= 0:
        raise VerificationError("position sizing conversion invalid")
    exact_units = (
        Fraction(proposal["notional_jpy_micros"], 1)
        * BASE_MICROUNITS_PER_UNIT
        / jpy_per_base_micros
    )
    units = exact_units.numerator // exact_units.denominator
    if units <= 0:
        return 0
    return units


def _common_path_gross_for_units(
    proposal: Mapping[str, Any],
    common: Mapping[str, Any],
    units_micros: int,
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    accounting: Mapping[str, Any],
) -> int:
    if units_micros < 0:
        raise VerificationError("common path units negative")
    if units_micros == 0:
        return 0
    valuation_arrival_ns = common.get("exit_valuation_arrival_ns")
    if type(valuation_arrival_ns) is not int:
        raise VerificationError("common path valuation clock missing")
    _, quote = _pair(proposal["instrument"])
    quote_pnl = (
        Fraction(
            proposal["direction"] * units_micros,
            BASE_MICROUNITS_PER_UNIT,
        )
        * (
            _market_price(common["exit"], "mid")
            - _market_price(common["entry"], "mid")
        )
    )
    source_watermark = _arrival_watermark_from_books(
        books, valuation_arrival_ns
    )
    return _asset_micros(
        _jpy_value(
            quote_pnl,
            quote,
            source_watermark,
            valuation_arrival_ns,
            books,
            accounting["max_conversion_staleness_ns"],
        )
    )


def _common_path(
    proposal: Mapping[str, Any],
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    execution: Mapping[str, Any],
    accounting: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> dict[str, Any] | None:
    entry = _entry_event(
        books, proposal, 0, evaluation["period_end_ts_ns"], execution["max_trade_quote_staleness_ns"]
    )
    if entry is None:
        return None
    due = entry["arrival_ts_ns"] + proposal["max_age_ns"]
    exit_event = _exit_event(
        books, proposal["instrument"], entry, due, evaluation["period_end_ts_ns"],
        execution["max_trade_quote_staleness_ns"],
    )
    reason = "FINITE_MAX_AGE"
    exit_valuation_arrival_ns = exit_event["arrival_ts_ns"] if exit_event is not None else None
    if exit_event is None:
        exit_event = _terminal_event(
            books, proposal["instrument"], evaluation["period_end_ts_ns"],
            execution["max_trade_quote_staleness_ns"],
        )
        if exit_event["arrival_ts_ns"] < entry["arrival_ts_ns"]:
            raise VerificationError("terminal common path precedes entry")
        reason = "TERMINAL_LIQUIDATION"
        exit_valuation_arrival_ns = evaluation["period_end_ts_ns"] - 1
    units = _sized_units(proposal, entry, books, accounting)
    if units == 0:
        return {
            "entry": entry,
            "exit": exit_event,
            "exit_reason": reason,
            "exit_valuation_arrival_ns": exit_valuation_arrival_ns,
            "units_micros": 0,
            "gross_pnl_jpy_micros": 0,
        }
    _, quote = _pair(proposal["instrument"])
    quote_pnl = Fraction(proposal["direction"] * units, BASE_MICROUNITS_PER_UNIT) \
        * (_market_price(exit_event, "mid") - _market_price(entry, "mid"))
    if exit_valuation_arrival_ns is None:
        raise VerificationError("common exit valuation clock missing")
    exit_watermark = _arrival_watermark_from_books(books, exit_valuation_arrival_ns)
    gross = _asset_micros(_jpy_value(
        quote_pnl,
        quote,
        exit_watermark,
        exit_valuation_arrival_ns,
        books,
        accounting["max_conversion_staleness_ns"],
    ))
    return {
        "entry": entry,
        "exit": exit_event,
        "exit_reason": reason,
        "exit_valuation_arrival_ns": exit_valuation_arrival_ns,
        "units_micros": units,
        "gross_pnl_jpy_micros": gross,
    }


def _signed_exposure(
    positions: Sequence[Mapping[str, Any]],
    marked: Sequence[Mapping[str, Any]],
    source_watermark_ns: int,
    arrival_cutoff_ns: int,
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    accounting: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    if len(positions) != len(marked):
        raise VerificationError("currency exposure position/mark count mismatch")
    native_legs: list[tuple[str, Fraction]] = []
    for position, mark in zip(positions, marked):
        base, quote = _pair(position["proposal"]["instrument"])
        units = position.get("units_micros")
        mark_price = mark.get("mark_price")
        if type(units) is not int or units < 0 \
                or not isinstance(mark_price, Fraction) or mark_price <= 0:
            raise VerificationError("currency exposure native leg invalid")
        signed_units = Fraction(
            position["proposal"]["direction"] * units,
            BASE_MICROUNITS_PER_UNIT,
        )
        native_legs.append((base, signed_units))
        native_legs.append((quote, -signed_units * mark_price))
    currencies = sorted({currency for currency, _ in native_legs})
    return {
        currency: amount_micros
        for currency in currencies
        if (amount_micros := _outward_currency_micros(sum((
            _currency_node_jpy_value(
                native_amount,
                node,
                source_watermark_ns,
                arrival_cutoff_ns,
                books,
                accounting["max_conversion_staleness_ns"],
                registry,
            )
            for node, native_amount in native_legs
            if node == currency
        ), Fraction()))) != 0
    }


def _valuation(
    position: Mapping[str, Any],
    mark_event: Mapping[str, Any],
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    accounting: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, int]],
    *,
    valuation_source_watermark_ns: int | None = None,
    valuation_arrival_ns: int | None = None,
) -> dict[str, int]:
    proposal = position["proposal"]
    policy = position["policy"]
    exit_price, _, _ = _execution_price(
        mark_event, proposal["direction"], False, policy, registry[proposal["instrument"]]
    )
    quote_pnl = Fraction(
        proposal["direction"] * position["units_micros"], BASE_MICROUNITS_PER_UNIT
    ) \
        * (exit_price - position["entry_price"])
    arrival_cutoff = mark_event["arrival_ts_ns"] if valuation_arrival_ns is None else valuation_arrival_ns
    source_watermark = (
        _arrival_watermark_from_books(books, arrival_cutoff)
        if valuation_source_watermark_ns is None
        else valuation_source_watermark_ns
    )
    _, quote = _pair(proposal["instrument"])
    executable_exact_micros = _jpy_value(
        quote_pnl,
        quote,
        source_watermark,
        arrival_cutoff,
        books,
        accounting["max_conversion_staleness_ns"],
    ) * JPY_MICROS_PER_YEN
    executable = (
        executable_exact_micros.numerator
        // executable_exact_micros.denominator
    )
    elapsed = arrival_cutoff - position["entry"]["arrival_ts_ns"]
    if elapsed < 0:
        raise VerificationError("valuation precedes entry")
    marked_notional_exact = _exact_position_risk_micros(
        direction=proposal["direction"],
        units_micros=position["units_micros"],
        price=exit_price,
        quote_currency=quote,
        source_watermark_ns=source_watermark,
        arrival_cutoff_ns=arrival_cutoff,
        books=books,
        accounting=accounting,
        opening=False,
    )
    entry_notional_exact = position["entry_notional_exact_jpy_micros"]
    entry_commission_exact = (
        entry_notional_exact * policy["commission_ppm_per_side"] / 1_000_000
    )
    exit_commission_exact = (
        marked_notional_exact * policy["commission_ppm_per_side"] / 1_000_000
    )
    entry_commission = _cost_micros(entry_commission_exact)
    exit_commission = _cost_micros(exit_commission_exact)
    commission = entry_commission + exit_commission
    financing_exact = (
        entry_notional_exact
        * policy["financing_ppm_per_day"]
        * elapsed
        / (DAY_NS * 1_000_000)
    )
    financing = _cost_micros(financing_exact)
    economic_net_exact = (
        executable_exact_micros
        - entry_commission_exact
        - exit_commission_exact
        - financing_exact
    )
    return {
        "mark_price": exit_price,
        "executable_pnl_jpy_micros": executable,
        "commission_jpy_micros": commission,
        "financing_jpy_micros": financing,
        "net_pnl_jpy_micros": executable - commission - financing,
        "elapsed_ns": elapsed,
        "marked_notional_jpy_micros": _outward_risk_micros(marked_notional_exact),
        "financing_basis_notional_jpy_micros": _outward_risk_micros(
            entry_notional_exact
        ),
        "economic_net_pnl_jpy_micros_numerator": economic_net_exact.numerator,
        "economic_net_pnl_jpy_micros_denominator": economic_net_exact.denominator,
    }


def _filled_record(
    position: dict[str, Any],
    exit_event: Mapping[str, Any],
    exit_reason: str,
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    accounting: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, int]],
    execution_hash: str,
    *,
    valuation_source_watermark_ns: int | None = None,
    valuation_arrival_ns: int | None = None,
) -> dict[str, Any]:
    values = _valuation(
        position,
        exit_event,
        books,
        accounting,
        registry,
        valuation_source_watermark_ns=valuation_source_watermark_ns,
        valuation_arrival_ns=valuation_arrival_ns,
    )
    proposal = position["proposal"]
    common = position["common"]
    _, exit_num, exit_den = _execution_price(
        exit_event,
        proposal["direction"],
        False,
        position["policy"],
        registry[proposal["instrument"]],
    )
    common_gross = common["gross_pnl_jpy_micros"]
    net = values["net_pnl_jpy_micros"]
    executable = values["executable_pnl_jpy_micros"]
    arm_units_common_gross = _common_path_gross_for_units(
        proposal,
        common,
        position["units_micros"],
        books,
        accounting,
    )
    fill_sizing_drag = common_gross - arm_units_common_gross
    execution_drag = arm_units_common_gross - executable
    return {
        "record_type": "ORACLE_DISPOSITION",
        "arm": position["arm"],
        "signal_id": position["signal_id"],
        "proposal_ordinal": proposal["proposal_ordinal"],
        "instrument": proposal["instrument"],
        "direction": proposal["direction"],
        "status": "FILLED_CLOSED",
        "entry_disposition": "FILLED",
        "exit_disposition": exit_reason,
        "action_transitions": ["ENTER", "EXIT"],
        "notional_jpy_micros": proposal["notional_jpy_micros"],
        "target_notional_jpy_micros": proposal["notional_jpy_micros"],
        "filled_notional_jpy_micros": position["entry_notional_jpy_micros"],
        "financing_basis_notional_jpy_micros": values[
            "financing_basis_notional_jpy_micros"
        ],
        "marked_or_exit_notional_jpy_micros": values[
            "marked_notional_jpy_micros"
        ],
        "exit_notional_jpy_micros": values["marked_notional_jpy_micros"],
        "units_micros": position["units_micros"],
        "economic_lot_id": position["economic_lot_id"],
        "common_entry_source_event_sha256": common["entry"]["source_event_sha256"],
        "common_exit_source_event_sha256": common["exit"]["source_event_sha256"],
        "common_gross_pnl_jpy_micros": common_gross,
        "arm_units_common_gross_pnl_jpy_micros": arm_units_common_gross,
        "entry_price_numerator": position["entry_price_numerator"],
        "entry_price_denominator": position["entry_price_denominator"],
        "exit_price_numerator": exit_num,
        "exit_price_denominator": exit_den,
        "entry_source_event_sha256": position["entry"]["source_event_sha256"],
        "entry_source_ts_ns": position["entry"]["source_ts_ns"],
        "entry_arrival_ts_ns": position["entry"]["arrival_ts_ns"],
        "exit_source_event_sha256": exit_event["source_event_sha256"],
        "exit_source_ts_ns": exit_event["source_ts_ns"],
        "exit_arrival_ts_ns": (
            exit_event["arrival_ts_ns"]
            if valuation_arrival_ns is None
            else valuation_arrival_ns
        ),
        "elapsed_ns": values["elapsed_ns"],
        "executable_pnl_before_direct_cost_jpy_micros": executable,
        "fill_sizing_drag_jpy_micros": fill_sizing_drag,
        "latency_spread_slippage_drag_jpy_micros": execution_drag,
        "commission_jpy_micros": values["commission_jpy_micros"],
        "financing_jpy_micros": values["financing_jpy_micros"],
        "realized_cost_jpy_micros": common_gross - net,
        "admission_opportunity_drag_jpy_micros": 0,
        "net_pnl_jpy_micros": net,
        "economic_net_pnl_jpy_micros_numerator": values[
            "economic_net_pnl_jpy_micros_numerator"
        ],
        "economic_net_pnl_jpy_micros_denominator": values[
            "economic_net_pnl_jpy_micros_denominator"
        ],
        "signed_currency_exposure_after_entry_jpy_micros": position["signed_exposure_after_entry"],
        "gross_open_notional_after_entry_jpy_micros": position["gross_after_entry"],
        "marked_equity_after_entry_jpy_micros": position["marked_equity_after_entry"],
        "required_margin_after_entry_jpy_micros": position["required_margin_after_entry"],
        "free_margin_after_entry_jpy_micros": position["free_margin_after_entry"],
        "entry_source_reference": {
            "provider_id": position["entry"]["provider_id"],
            "source_event_sha256": position["entry"]["source_event_sha256"],
            "source_ts_ns": position["entry"]["source_ts_ns"],
            "arrival_ts_ns": position["entry"]["arrival_ts_ns"],
            "execution_policy_sha256": execution_hash,
        },
        "exit_source_reference": {
            "provider_id": exit_event["provider_id"],
            "source_event_sha256": exit_event["source_event_sha256"],
            "source_ts_ns": exit_event["source_ts_ns"],
            "arrival_ts_ns": exit_event["arrival_ts_ns"],
            "execution_policy_sha256": execution_hash,
        },
        "terminal_inventory_mtm_jpy_micros": 0,
        "external_order_count": 0,
    }


def _rejected_record(
    proposal: Mapping[str, Any],
    signal_id: str,
    arm: str,
    reason: str,
    common: Mapping[str, Any] | None,
) -> dict[str, Any]:
    gross = 0 if common is None else common["gross_pnl_jpy_micros"]
    sizing_drag = gross if reason == "SIZE_ROUNDED_TO_ZERO" else 0
    latency_drag = gross if reason == "NO_CAUSAL_FILL" else 0
    admission_drag = gross if reason in {
        "SAME_PAIR_COLLISION_REJECTED",
        "GROSS_CAP_REJECTED",
        "POSITION_CAP_REJECTED",
        "CURRENCY_CAP_REJECTED",
        "MARGIN_ENTRY_REJECTED",
        "ACCOUNT_HALTED",
    } else 0
    return {
        "record_type": "ORACLE_DISPOSITION",
        "arm": arm,
        "signal_id": signal_id,
        "proposal_ordinal": proposal["proposal_ordinal"],
        "instrument": proposal["instrument"],
        "direction": proposal["direction"],
        "status": reason,
        "entry_disposition": reason,
        "exit_disposition": "NOT_APPLICABLE",
        "action_transitions": ["NO_ENTRY"],
        "notional_jpy_micros": proposal["notional_jpy_micros"],
        "target_notional_jpy_micros": proposal["notional_jpy_micros"],
        "filled_notional_jpy_micros": 0,
        "financing_basis_notional_jpy_micros": 0,
        "marked_or_exit_notional_jpy_micros": 0,
        "exit_notional_jpy_micros": 0,
        "units_micros": 0,
        "economic_lot_id": signal_id,
        "common_entry_source_event_sha256": None if common is None else common["entry"]["source_event_sha256"],
        "common_exit_source_event_sha256": None if common is None else common["exit"]["source_event_sha256"],
        "common_gross_pnl_jpy_micros": gross,
        "arm_units_common_gross_pnl_jpy_micros": (
            0 if reason in {"NO_CAUSAL_FILL", "SIZE_ROUNDED_TO_ZERO"} else gross
        ),
        "executable_pnl_before_direct_cost_jpy_micros": 0,
        "fill_sizing_drag_jpy_micros": sizing_drag,
        "latency_spread_slippage_drag_jpy_micros": latency_drag,
        "commission_jpy_micros": 0,
        "financing_jpy_micros": 0,
        "realized_cost_jpy_micros": 0,
        "admission_opportunity_drag_jpy_micros": admission_drag,
        "net_pnl_jpy_micros": 0,
        "economic_net_pnl_jpy_micros_numerator": 0,
        "economic_net_pnl_jpy_micros_denominator": 1,
        "signed_currency_exposure_after_entry_jpy_micros": {},
        "gross_open_notional_after_entry_jpy_micros": 0,
        "marked_equity_after_entry_jpy_micros": None,
        "required_margin_after_entry_jpy_micros": 0,
        "free_margin_after_entry_jpy_micros": None,
        "terminal_inventory_mtm_jpy_micros": 0,
        "external_order_count": 0,
    }


def _watermark(source_rows: Sequence[Mapping[str, Any]], arrival_ns: int) -> int:
    values = [row["source_ts_ns"] for row in source_rows if row["arrival_ts_ns"] <= arrival_ns]
    if not values:
        raise VerificationError("no source watermark")
    return max(values)


def _mark(
    active: Sequence[Mapping[str, Any]],
    closed: Sequence[Mapping[str, Any]],
    arrival_ns: int,
    source_rows: Sequence[Mapping[str, Any]],
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    accounting: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, int]],
    evaluation: Mapping[str, Any],
    trade_staleness_ns: int,
) -> dict[str, Any]:
    source_watermark = _watermark(source_rows, arrival_ns)
    realized = sum(record["net_pnl_jpy_micros"] for record in closed)
    unrealized = 0
    marked_positions: list[dict[str, Any]] = []
    for position in active:
        mark_event = _latest_causal(
            books,
            position["proposal"]["instrument"],
            source_watermark,
            arrival_ns,
            trade_staleness_ns,
        )
        values = _valuation(
            position,
            mark_event,
            books,
            accounting,
            registry,
            valuation_source_watermark_ns=source_watermark,
            valuation_arrival_ns=arrival_ns,
        )
        unrealized += values["net_pnl_jpy_micros"]
        marked_positions.append({
            "risk_notional_jpy_micros": values["marked_notional_jpy_micros"],
            "mark_price": values["mark_price"],
        })
    equity = evaluation["initial_equity_jpy_micros"] + realized + unrealized
    gross = sum(item["risk_notional_jpy_micros"] for item in marked_positions)
    required = _cost_micros(Fraction(gross * evaluation["margin_rate_bps"], 10_000))
    free = equity - required
    ratio_ok = equity > 0 and gross * 10_000 <= equity * evaluation["max_gross_to_equity_bps"]
    return {
        "arrival_ts_ns": arrival_ns,
        "source_watermark_ts_ns": source_watermark,
        "marked_equity_jpy_micros": equity,
        "gross_notional_jpy_micros": gross,
        "required_margin_jpy_micros": required,
        "free_margin_jpy_micros": free,
        "signed_currency_exposure_jpy_micros": _signed_exposure(
            active,
            marked_positions,
            source_watermark,
            arrival_ns,
            books,
            accounting,
            registry,
        ),
        "margin_ratio_pass": ratio_ok,
    }


def _risk_closeout_reason(
    mark: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> str | None:
    if mark["marked_equity_jpy_micros"] <= 0 \
            or mark["free_margin_jpy_micros"] < 0 \
            or mark["margin_ratio_pass"] is not True \
            or mark["gross_notional_jpy_micros"] \
                > evaluation["margin_notional_cap_jpy_micros"]:
        return "MARGIN_CLOSEOUT"
    if mark["gross_notional_jpy_micros"] \
            > inventory["max_gross_notional_jpy_micros"] \
            or max((
                abs(value)
                for value in mark["signed_currency_exposure_jpy_micros"].values()
            ), default=0) > inventory["max_currency_notional_jpy_micros"]:
        return "INVENTORY_CAP_CLOSEOUT"
    return None


def _replay_arm(
    arm: str,
    source_rows: Sequence[Mapping[str, Any]],
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    proposal_root: Mapping[str, Any],
    common: Mapping[int, Mapping[str, Any] | None],
    execution: Mapping[str, Any],
    inventory: Mapping[str, Any],
    accounting: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    policy = execution["arms"][arm]
    max_trade_stale = execution["max_trade_quote_staleness_ns"]
    plans: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    results: dict[int, dict[str, Any]] = {}
    signal_ids: dict[int, str] = {}
    for proposal in proposal_root["rows"]:
        ordinal = proposal["proposal_ordinal"]
        signal_id = _signal_id(proposal_root["candidate_key"], proposal, proposal_root["provenance"])
        signal_ids[ordinal] = signal_id
        common_item = common[ordinal]
        if common_item is None:
            results[ordinal] = _rejected_record(proposal, signal_id, arm, "NO_COMMON_CAUSAL_PATH", None)
            continue
        entry = _entry_event(
            books, proposal, policy["latency_ns"], evaluation["period_end_ts_ns"], max_trade_stale
        )
        if entry is None:
            results[ordinal] = _rejected_record(proposal, signal_id, arm, "NO_CAUSAL_FILL", common_item)
            continue
        plans[entry["source_event_sha256"]].append({
            "proposal": proposal,
            "signal_id": signal_id,
            "common": common_item,
            "entry": entry,
        })
    active: list[dict[str, Any]] = []
    closed: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    risk_timeline: list[dict[str, Any]] = []
    halted = False

    def close_due(event: Mapping[str, Any]) -> None:
        due = [
            position for position in active
            if position["proposal"]["instrument"] == event["instrument"]
            and event["arrival_ts_ns"] >= position["due_arrival_ns"]
            and event["source_ts_ns"] >= position["entry"]["source_ts_ns"]
        ]
        for position in sorted(due, key=lambda item: item["proposal"]["proposal_ordinal"]):
            if not _fresh(event, position["due_arrival_ns"], max_trade_stale):
                raise VerificationError("scheduled exit quote stale")
            record = _filled_record(
                position, event, "FINITE_MAX_AGE", books, accounting, registry,
                execution["execution_policy_sha256"],
            )
            position["closed_record"] = record
            results[position["proposal"]["proposal_ordinal"]] = record
            closed.append(record)
            active.remove(position)

    def closeout_all(arrival_ns: int, reason: str) -> None:
        source_watermark = _watermark(source_rows, arrival_ns)
        for position in sorted(list(active), key=lambda item: item["proposal"]["proposal_ordinal"]):
            quote = _latest_causal(
                books,
                position["proposal"]["instrument"],
                source_watermark,
                arrival_ns,
                max_trade_stale,
            )
            record = _filled_record(
                position, quote, reason, books, accounting, registry,
                execution["execution_policy_sha256"],
                valuation_source_watermark_ns=source_watermark,
                valuation_arrival_ns=arrival_ns,
            )
            position["closed_record"] = record
            results[position["proposal"]["proposal_ordinal"]] = record
            closed.append(record)
            active.remove(position)

    period_start = evaluation["period_start_ts_ns"]
    period_end = evaluation["period_end_ts_ns"]
    terminal_arrival_ns = period_end - 1
    events_by_arrival: defaultdict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for event in source_rows:
        if period_start <= event["arrival_ts_ns"] < period_end:
            events_by_arrival[event["arrival_ts_ns"]].append(event)
    boundary_clocks = {terminal_arrival_ns}
    for month_id in _all_months(period_start, period_end):
        _, month_end = _month_bounds_ns(month_id)
        checkpoint = min(period_end, month_end) - 1
        if checkpoint >= period_start:
            boundary_clocks.add(checkpoint)
    attribution_clocks = {
        item["exit_valuation_arrival_ns"]
        for item in common.values()
        if item is not None
        and period_start <= item["exit_valuation_arrival_ns"] < period_end
    }
    risk_clocks = sorted({
        *events_by_arrival,
        *boundary_clocks,
        *attribution_clocks,
    })
    for arrival_ns in risk_clocks:
        batch = sorted(
            events_by_arrival.get(arrival_ns, ()),
            key=lambda event: (
                event["source_ts_ns"], event["provider_id"], event["sequence"]
            ),
        )
        for event in batch:
            close_due(event)
        current_mark = _mark(
            active, closed, arrival_ns, source_rows, books, accounting, registry,
            evaluation, max_trade_stale,
        )
        risk_timeline.append(current_mark)
        closeout_reason = _risk_closeout_reason(
            current_mark, evaluation, inventory
        )
        if closeout_reason is not None:
            if active:
                closeout_all(arrival_ns, closeout_reason)
            halted = True
            risk_timeline.append(_mark(
                active, closed, arrival_ns, source_rows, books, accounting,
                registry, evaluation, max_trade_stale,
            ))
        batch_plans = [
            plan for event in batch for plan in plans.get(event["source_event_sha256"], ())
        ]
        for plan in sorted(batch_plans, key=lambda item: item["proposal"]["proposal_ordinal"]):
            proposal = plan["proposal"]
            ordinal = proposal["proposal_ordinal"]
            event = plan["entry"]
            if halted:
                results[ordinal] = _rejected_record(proposal, plan["signal_id"], arm, "ACCOUNT_HALTED", plan["common"])
                continue
            if any(position["proposal"]["instrument"] == proposal["instrument"] for position in active):
                results[ordinal] = _rejected_record(proposal, plan["signal_id"], arm, "SAME_PAIR_COLLISION_REJECTED", plan["common"])
                continue
            entry_price, entry_num, entry_den = _execution_price(
                event, proposal["direction"], True, policy, registry[proposal["instrument"]]
            )
            units_micros = _actual_fill_units(
                proposal, event, entry_price, books, accounting
            )
            if units_micros == 0:
                results[ordinal] = _rejected_record(
                    proposal, plan["signal_id"], arm, "SIZE_ROUNDED_TO_ZERO", plan["common"]
                )
                continue
            _, quote_currency = _pair(proposal["instrument"])
            entry_watermark = _arrival_watermark_from_books(
                books, event["arrival_ts_ns"]
            )
            entry_notional_exact = _exact_position_risk_micros(
                direction=proposal["direction"],
                units_micros=units_micros,
                price=entry_price,
                quote_currency=quote_currency,
                source_watermark_ns=entry_watermark,
                arrival_cutoff_ns=event["arrival_ts_ns"],
                books=books,
                accounting=accounting,
                opening=True,
            )
            position = {
                "arm": arm,
                "proposal": proposal,
                "signal_id": plan["signal_id"],
                "common": plan["common"],
                "entry": event,
                "entry_price": entry_price,
                "entry_price_numerator": entry_num,
                "entry_price_denominator": entry_den,
                "units_micros": units_micros,
                "economic_lot_id": _economic_lot_id(
                    proposal_root["candidate_key"], proposal, proposal_root["provenance"]
                ),
                "entry_notional_exact_jpy_micros": entry_notional_exact,
                "entry_notional_jpy_micros": _outward_risk_micros(
                    entry_notional_exact
                ),
                "policy": policy,
                # Independently enforce the frozen fill-arrival max-age.  The
                # entry latency is already reflected by the chosen entry event.
                "due_arrival_ns": event["arrival_ts_ns"] + proposal["max_age_ns"],
            }
            tentative = [*active, position]
            if len(tentative) > inventory["max_open_positions"]:
                results[ordinal] = _rejected_record(proposal, plan["signal_id"], arm, "POSITION_CAP_REJECTED", plan["common"])
                continue
            entry_mark = _mark(
                tentative, closed, arrival_ns, source_rows, books, accounting,
                registry, evaluation, max_trade_stale,
            )
            gross = entry_mark["gross_notional_jpy_micros"]
            exposure = entry_mark["signed_currency_exposure_jpy_micros"]
            if gross > inventory["max_gross_notional_jpy_micros"]:
                results[ordinal] = _rejected_record(proposal, plan["signal_id"], arm, "GROSS_CAP_REJECTED", plan["common"])
                continue
            if max((abs(value) for value in exposure.values()), default=0) > inventory["max_currency_notional_jpy_micros"]:
                results[ordinal] = _rejected_record(proposal, plan["signal_id"], arm, "CURRENCY_CAP_REJECTED", plan["common"])
                continue
            if _risk_closeout_reason(entry_mark, evaluation, inventory) \
                    == "MARGIN_CLOSEOUT":
                results[ordinal] = _rejected_record(proposal, plan["signal_id"], arm, "MARGIN_ENTRY_REJECTED", plan["common"])
                continue
            position["signed_exposure_after_entry"] = exposure
            position["gross_after_entry"] = gross
            position["marked_equity_after_entry"] = entry_mark["marked_equity_jpy_micros"]
            position["required_margin_after_entry"] = entry_mark["required_margin_jpy_micros"]
            position["free_margin_after_entry"] = entry_mark["free_margin_jpy_micros"]
            active.append(position)
            positions.append(position)
            risk_timeline.append(entry_mark)
    for ordinal, signal_id in signal_ids.items():
        if ordinal not in results and not any(position["proposal"]["proposal_ordinal"] == ordinal for position in active):
            proposal = proposal_root["rows"][ordinal - 1]
            results[ordinal] = _rejected_record(proposal, signal_id, arm, "NO_CAUSAL_FILL", common[ordinal])
    if active:
        terminal_watermark = _watermark(source_rows, terminal_arrival_ns)
        preterminal_mark = _mark(
            active,
            closed,
            terminal_arrival_ns,
            source_rows,
            books,
            accounting,
            registry,
            evaluation,
            max_trade_stale,
        )
        risk_timeline.append(preterminal_mark)
        terminal_reason = _risk_closeout_reason(
            preterminal_mark, evaluation, inventory
        ) or "TERMINAL_LIQUIDATION"
        for position in sorted(list(active), key=lambda item: item["proposal"]["proposal_ordinal"]):
            terminal = _terminal_event(
                books, position["proposal"]["instrument"], evaluation["period_end_ts_ns"], max_trade_stale
            )
            if terminal["arrival_ts_ns"] < position["entry"]["arrival_ts_ns"]:
                raise VerificationError("terminal quote precedes entry")
            record = _filled_record(
                position, terminal, terminal_reason, books, accounting,
                registry, execution["execution_policy_sha256"],
                valuation_source_watermark_ns=terminal_watermark,
                valuation_arrival_ns=terminal_arrival_ns,
            )
            position["closed_record"] = record
            results[position["proposal"]["proposal_ordinal"]] = record
            closed.append(record)
            active.remove(position)
        last_arrival = max(record["exit_arrival_ts_ns"] for record in closed)
        risk_timeline.append(_mark(
            [], closed, last_arrival, source_rows, books, accounting, registry,
            evaluation, max_trade_stale,
        ))
    if set(results) != {row["proposal_ordinal"] for row in proposal_root["rows"]}:
        raise VerificationError("incomplete arm disposition set")
    return [results[index] for index in sorted(results)], positions, risk_timeline


def _equity_at(
    positions: Sequence[Mapping[str, Any]],
    cutoff_ns: int,
    source_rows: Sequence[Mapping[str, Any]],
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    accounting: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, int]],
    evaluation: Mapping[str, Any],
    trade_staleness_ns: int,
) -> int:
    equity = evaluation["initial_equity_jpy_micros"]
    if not positions:
        return equity
    available = [row for row in source_rows if row["arrival_ts_ns"] <= cutoff_ns]
    if not available:
        return equity
    source_watermark = max(row["source_ts_ns"] for row in available)
    for position in positions:
        if position["entry"]["arrival_ts_ns"] > cutoff_ns:
            continue
        closed = position.get("closed_record")
        if closed is not None and closed["exit_arrival_ts_ns"] <= cutoff_ns:
            equity += closed["net_pnl_jpy_micros"]
            continue
        mark_event = _latest_causal(
            books,
            position["proposal"]["instrument"],
            source_watermark,
            cutoff_ns,
            trade_staleness_ns,
        )
        equity += _valuation(
            position,
            mark_event,
            books,
            accounting,
            registry,
            valuation_source_watermark_ns=source_watermark,
            valuation_arrival_ns=cutoff_ns,
        )["net_pnl_jpy_micros"]
    return equity


def _cluster_metrics(
    records: Sequence[Mapping[str, Any]], evaluation: Mapping[str, Any]
) -> tuple[int, int, str, list[dict[str, Any]]]:
    buckets: defaultdict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record["status"] == "FILLED_CLOSED":
            economic_lot_id = record.get("economic_lot_id")
            if type(economic_lot_id) is not str or SHA256_RE.fullmatch(economic_lot_id) is None:
                raise VerificationError("cluster economic-lot identity invalid")
            numerator = record.get("economic_net_pnl_jpy_micros_numerator")
            denominator = record.get("economic_net_pnl_jpy_micros_denominator")
            if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
                raise VerificationError("cluster exact economic return fraction invalid")
            buckets[record["entry_arrival_ts_ns"] // evaluation["cluster_window_ns"]].append(record)
    observations: list[dict[str, Any]] = []
    exact_observations: list[tuple[Fraction, str]] = []
    initial = evaluation["initial_equity_jpy_micros"]
    for bucket, bucket_records in sorted(buckets.items()):
        adjacency: defaultdict[str, set[str]] = defaultdict(set)
        for record in bucket_records:
            left, right = _pair(record["instrument"])
            adjacency[left].add(right)
            adjacency[right].add(left)
        unvisited = set(adjacency)
        node_components: list[list[str]] = []
        while unvisited:
            pending = [min(unvisited)]
            component_nodes: set[str] = set()
            while pending:
                node = pending.pop()
                if node in component_nodes:
                    continue
                component_nodes.add(node)
                pending.extend(sorted(adjacency[node] - component_nodes, reverse=True))
            unvisited -= component_nodes
            node_components.append(sorted(component_nodes))
        for nodes in sorted(node_components):
            node_set = set(nodes)
            component = [
                record
                for record in bucket_records
                if set(_pair(record["instrument"])).issubset(node_set)
            ]
            exact_pnl = sum(
                (
                    Fraction(
                        record["economic_net_pnl_jpy_micros_numerator"],
                        record["economic_net_pnl_jpy_micros_denominator"],
                    )
                    for record in component
                ),
                Fraction(0, 1),
            )
            risk_pnl = exact_pnl.numerator // exact_pnl.denominator
            ledger_pnl = sum(
                record["net_pnl_jpy_micros"] for record in component
            )
            identity = {"time_bucket": bucket, "currency_nodes": nodes}
            cluster_id = sha256_bytes(canonical_bytes(identity))
            observations.append({
                "cluster_id": cluster_id,
                "time_bucket": bucket,
                "currency_nodes": nodes,
                "source_signal_set_sha256": sha256_bytes(canonical_bytes(sorted({record["economic_lot_id"] for record in component}))),
                "ledger_net_pnl_jpy_micros": ledger_pnl,
                "cluster_risk_net_pnl_jpy_micros": risk_pnl,
                "signed_return": _signed_ratio_text(risk_pnl, initial),
            })
            exact_observations.append((exact_pnl, cluster_id))
    ordered = sorted(exact_observations, key=lambda item: (item[0], item[1]))
    tail_count = max(1, (len(ordered) * evaluation["cvar_tail_bps"] + 9_999) // 10_000) if ordered else 0
    tail = ordered[:tail_count]
    tail_exact = sum((item[0] for item in tail), Fraction(0, 1))
    cvar_exact = tail_exact / tail_count if tail_count else Fraction(0, 1)
    cvar_jpy = cvar_exact.numerator // cvar_exact.denominator if tail_count else 0
    cvar_return = _signed_ratio_text(
        cvar_exact.numerator,
        cvar_exact.denominator * initial,
    ) if tail_count else "0.000000000000000000"
    return len(observations), cvar_jpy, cvar_return, observations


def _arm_metrics(
    records: Sequence[Mapping[str, Any]],
    positions: Sequence[Mapping[str, Any]],
    risk_timeline: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    accounting: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, int]],
    evaluation: Mapping[str, Any],
    trade_staleness_ns: int,
) -> dict[str, Any]:
    initial = evaluation["initial_equity_jpy_micros"]
    filled = [record for record in records if record["status"] == "FILLED_CLOSED"]
    rejected = [record for record in records if record["status"] != "FILLED_CLOSED"]
    net = sum(record["net_pnl_jpy_micros"] for record in filled)
    gross = sum(record["common_gross_pnl_jpy_micros"] for record in records)
    cost = sum(record["realized_cost_jpy_micros"] for record in filled)
    sizing_drag = sum(record["fill_sizing_drag_jpy_micros"] for record in records)
    latency_drag = sum(record["latency_spread_slippage_drag_jpy_micros"] for record in records)
    direct_cost = sum(
        record["commission_jpy_micros"] + record["financing_jpy_micros"]
        for record in filled
    )
    admission_drag = sum(record["admission_opportunity_drag_jpy_micros"] for record in rejected)
    decomposed_drag = sizing_drag + latency_drag + direct_cost + admission_drag
    if decomposed_drag != gross - net:
        raise VerificationError("arm drag attribution does not reconcile")
    monthly: list[dict[str, Any]] = []
    boundary_equities: list[tuple[int, int]] = []
    for month in _all_months(evaluation["period_start_ts_ns"], evaluation["period_end_ts_ns"]):
        month_start, month_end = _month_bounds_ns(month)
        segment_start = max(month_start, evaluation["period_start_ts_ns"])
        segment_end = min(month_end, evaluation["period_end_ts_ns"])
        start_equity = _equity_at(
            positions, segment_start - 1, source_rows, books, accounting, registry,
            evaluation, trade_staleness_ns,
        )
        end_equity = _equity_at(
            positions, segment_end - 1, source_rows, books, accounting, registry,
            evaluation, trade_staleness_ns,
        )
        multiple_defined = start_equity > 0
        boundary_equities.extend((
            (segment_start - 1, start_equity),
            (segment_end - 1, end_equity),
        ))
        monthly.append({
            "month_id": month,
            "comparable_full_month": month_start >= evaluation["period_start_ts_ns"] and month_end <= evaluation["period_end_ts_ns"],
            "segment_start_ts_ns": segment_start,
            "segment_end_ts_ns": segment_end,
            "start_equity_jpy_micros": start_equity,
            "end_equity_jpy_micros": end_equity,
            "equity_multiple": _ratio_text(end_equity, start_equity) if multiple_defined else None,
            "equity_multiple_status": (
                "DEFINED" if multiple_defined else "UNDEFINED_NONPOSITIVE_START_EQUITY"
            ),
            "ruin_observed": start_equity <= 0 or end_equity <= 0,
        })
    peak = initial
    max_drawdown = 0
    max_ratio = Fraction(0, 1)
    drawdown_observations = [
        (mark["arrival_ts_ns"], index, mark["marked_equity_jpy_micros"])
        for index, mark in enumerate(risk_timeline)
    ]
    drawdown_observations.extend(
        (timestamp, len(risk_timeline) + index, equity)
        for index, (timestamp, equity) in enumerate(boundary_equities)
    )
    for _, _, equity in sorted(drawdown_observations):
        peak = max(peak, equity)
        drawdown = peak - equity
        ratio = Fraction(drawdown, peak) if peak > 0 else Fraction(1, 1)
        if drawdown > max_drawdown:
            max_drawdown = drawdown
        if ratio > max_ratio:
            max_ratio = ratio
    n_eff, cvar, cvar_return, clusters = _cluster_metrics(records, evaluation)
    max_gross = max((mark["gross_notional_jpy_micros"] for mark in risk_timeline), default=0)
    min_equity = min(
        [initial, *(mark["marked_equity_jpy_micros"] for mark in risk_timeline),
         *(equity for _, equity in boundary_equities)]
    )
    max_required = max((mark["required_margin_jpy_micros"] for mark in risk_timeline), default=0)
    min_free = min((mark["free_margin_jpy_micros"] for mark in risk_timeline), default=initial)
    return {
        "proposal_count": len(records),
        "executed_count": len(filled),
        "disposition_counts": dict(sorted(Counter(record["status"] for record in records).items())),
        "signal_id_set_sha256": sha256_bytes(canonical_bytes(sorted(record["signal_id"] for record in records))),
        "common_gross_pnl_jpy_micros": gross,
        "realized_cost_jpy_micros": cost,
        "fill_sizing_drag_jpy_micros": sizing_drag,
        "latency_spread_slippage_drag_jpy_micros": latency_drag,
        "direct_commission_financing_cost_jpy_micros": direct_cost,
        "admission_opportunity_drag_jpy_micros": admission_drag,
        "total_execution_and_admission_drag_jpy_micros": decomposed_drag,
        "net_pnl_jpy_micros": net,
        "ending_equity_jpy_micros": initial + net,
        "ending_equity_multiple": _ratio_text(initial + net, initial),
        "direction_accuracy": _ratio_text(sum(record["common_gross_pnl_jpy_micros"] > 0 for record in filled), len(filled)) if filled else "0.000000000000000000",
        "max_drawdown_jpy_micros": max_drawdown,
        "max_drawdown_ratio": _nonnegative_ratio_ceiling_text(
            max_ratio.numerator, max_ratio.denominator
        ),
        "cvar_tail_bps": evaluation["cvar_tail_bps"],
        "cluster_cvar_jpy_micros": cvar,
        "cluster_cvar_return": cvar_return,
        "currency_time_cluster_n_eff": n_eff,
        "currency_time_cluster_observations": clusters,
        "monthly": monthly,
        "max_gross_notional_jpy_micros": max_gross,
        "minimum_marked_equity_jpy_micros": min_equity,
        "maximum_required_margin_jpy_micros": max_required,
        "minimum_free_margin_jpy_micros": min_free,
        "margin_guard_pass": (
            min_equity > 0
            and min_free >= 0
            and max_gross <= evaluation["margin_notional_cap_jpy_micros"]
            and all(mark["margin_ratio_pass"] is True for mark in risk_timeline)
            and all(
                record.get("exit_disposition")
                not in {"MARGIN_CLOSEOUT", "INVENTORY_CAP_CLOSEOUT"}
                for record in records
            )
        ),
        "terminal_open_positions": 0,
        "terminal_inventory_mtm_jpy_micros": 0,
    }


def _hash_chain(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    previous = ZERO_SHA
    for sequence, row in enumerate(rows, 1):
        payload = {"ledger_schema_version": 2, "ledger_sequence": sequence, "previous_hash": previous, **dict(row)}
        payload["record_hash"] = embedded_hash(payload, "record_hash")
        result.append(payload)
        previous = payload["record_hash"]
    return result


def _expected_evidence(state: Mapping[str, Any]) -> tuple[bytes, dict[str, Any]]:
    source_rows = state["source_rows"]
    books = state["books"]
    proposal = state["proposal"]
    execution = state["execution"]
    inventory = state["inventory"]
    accounting = state["accounting"]
    evaluation = state["evaluation"]
    registry = state["registry"]
    common = {
        row["proposal_ordinal"]: _common_path(row, books, execution, accounting, evaluation)
        for row in proposal["rows"]
    }
    all_records: list[dict[str, Any]] = []
    arm_metrics: dict[str, Any] = {}
    signal_sets: dict[str, list[str]] = {}
    for arm in ARMS:
        records, positions, risk = _replay_arm(
            arm, source_rows, books, proposal, common, execution, inventory,
            accounting, evaluation, registry,
        )
        all_records.extend(records)
        signal_sets[arm] = sorted(record["signal_id"] for record in records)
        arm_metrics[arm] = _arm_metrics(
            records, positions, risk, source_rows, books, accounting, registry,
            evaluation, execution["max_trade_quote_staleness_ns"],
        )
    if len({tuple(signal_sets[arm]) for arm in ARMS}) != 1:
        raise VerificationError("arm signal sets diverged")
    all_records.sort(key=lambda row: (row["proposal_ordinal"], ARMS.index(row["arm"])))
    ledger_rows = _hash_chain(all_records)
    ledger_bytes = b"".join(canonical_bytes(row) + b"\n" for row in ledger_rows)
    metrics: dict[str, Any] = {
        "schema_version": 2,
        "initial_equity_jpy_micros": evaluation["initial_equity_jpy_micros"],
        "same_signal_ids_all_arms": True,
        "all_proposals_have_all_arm_dispositions": True,
        "common_gross_reference_shared": all(
            len({
                record["common_gross_pnl_jpy_micros"]
                for record in all_records if record["proposal_ordinal"] == ordinal
            }) == 1
            for ordinal in range(1, len(proposal["rows"]) + 1)
        ),
        "arms": arm_metrics,
        "external_orders": 0,
        "terminal_inventory_mtm_jpy_micros": 0,
    }
    metrics["metrics_sha256"] = embedded_hash(metrics, "metrics_sha256")
    provenance_root = sha256_bytes(canonical_bytes({
        "provenance": proposal["provenance"],
        "rows": [
            {
                "proposal_ordinal": row["proposal_ordinal"],
                "decision_source_event_sha256": row["decision_source_event_sha256"],
                "completed_data_watermark_source_ts_ns": row["completed_data_watermark_source_ts_ns"],
                "completed_data_prefix_root_sha256": row["completed_data_prefix_root_sha256"],
            }
            for row in proposal["rows"]
        ],
    }))
    input_hashes = {
        label: state["hashes"][label]
        for label in (
            "source_blob", "source_manifest", "proposal", "execution_policy",
            "inventory_policy", "accounting_policy", "evaluation_policy",
            "instrument_registry", "authority_policy",
        )
    }
    request_sha = sha256_bytes(state["blobs"]["oracle_request"])
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "oracle_implementation": ORACLE_NAME,
        "status": "COMPLETE",
        "classification": CLASSIFICATION,
        "causal_signal_admission": False,
        "release_evidence_eligible": False,
        "detector_replay_receipt_required": True,
        "authority": dict(AUTHORITY),
        "oracle_release_content_binding": {
            "code_sha256": state["oracle_release"]["hashes"]["code_sha256"],
            "contract_sha256": state["oracle_release"]["hashes"]["contract_sha256"],
            "schema_sha256": state["oracle_release"]["hashes"]["schema_sha256"],
            "launcher_sha256": _LAUNCHER_SHA256,
            "snapshot_mode": _EXECUTION_SNAPSHOT_MODE,
        },
        "oracle_execution_provenance_scope": EXECUTION_PROVENANCE_SCOPE,
        "request_sha256": request_sha,
        "input_artifact_sha256": dict(sorted(input_hashes.items())),
        "raw_source_manifest_sha256": state["hashes"]["source_manifest"],
        "proposal_provenance_root_sha256": provenance_root,
        "producer_result_or_metrics_used": False,
        "proposal_identity_generated_by_oracle": True,
        "oracle_ledger_file": "oracle_ledger.jsonl",
        "oracle_ledger_sha256": sha256_bytes(ledger_bytes),
        "oracle_ledger_size_bytes": len(ledger_bytes),
        "oracle_ledger_row_count": len(ledger_rows),
        "oracle_ledger_terminal_hash": ledger_rows[-1]["record_hash"] if ledger_rows else ZERO_SHA,
        "oracle_metrics": metrics,
        "terminal_inventory_mtm_jpy_micros": 0,
        "external_orders": 0,
        "anchor_status": ANCHOR_STATUS,
    }
    manifest["oracle_root_sha256"] = embedded_hash(manifest, "oracle_root_sha256")
    return ledger_bytes, manifest


def _reference_input_root(state: Mapping[str, Any]) -> str:
    artifact_hashes = {
        label: sha256_bytes(state["blobs"][label])
        for label in REFERENCE_INPUT_LABELS
    }
    return sha256_bytes(canonical_bytes({
        "artifact_sha256": dict(sorted(artifact_hashes.items())),
    }))


def _validate_reference_ledger(
    ledger_bytes: bytes,
    row_count: int,
    terminal_hash: str,
    proposal_count: int,
) -> list[dict[str, Any]]:
    if row_count == 0:
        if ledger_bytes != b"" or terminal_hash != ZERO_SHA \
                or proposal_count != 0:
            raise VerificationError("empty reference ledger chain mismatch")
        return []
    if row_count != proposal_count * len(ARMS):
        raise VerificationError("reference ledger disposition coverage mismatch")
    if not ledger_bytes.endswith(b"\n") or ledger_bytes.endswith(b"\n\n"):
        raise VerificationError("reference ledger canonical JSONL framing invalid")
    previous = ZERO_SHA
    rows = ledger_bytes.splitlines(keepends=True)
    if len(rows) != row_count:
        raise VerificationError("reference ledger row count mismatch")
    decoded_rows: list[dict[str, Any]] = []
    for sequence, line in enumerate(rows, 1):
        if not line.endswith(b"\n") or line == b"\n":
            raise VerificationError("reference ledger row framing invalid")
        row = strict_json(line, f"reference ledger row {sequence}")
        if _integer(
            row.get("ledger_schema_version"),
            f"reference ledger row {sequence} schema",
        ) != 2 \
                or _integer(
                    row.get("ledger_sequence"),
                    f"reference ledger row {sequence} sequence",
                    1,
                ) != sequence \
                or _digest(
                    row.get("previous_hash"),
                    f"reference ledger row {sequence} previous hash",
                ) != previous \
                or _digest(
                    row.get("record_hash"),
                    f"reference ledger row {sequence} record hash",
                ) != embedded_hash(row, "record_hash"):
            raise VerificationError("reference ledger hash chain mismatch")
        label = f"reference ledger row {sequence}"
        if row.get("record_type") != "ORACLE_DISPOSITION" \
                or row.get("arm") not in ARMS \
                or type(row.get("status")) is not str \
                or not row["status"]:
            raise VerificationError(f"{label} disposition identity mismatch")
        _digest(row.get("signal_id"), f"{label}.signal_id")
        proposal_ordinal = _integer(
            row.get("proposal_ordinal"),
            f"{label}.proposal_ordinal",
            1,
        )
        if proposal_ordinal > proposal_count:
            raise VerificationError(f"{label} proposal ordinal outside request")
        for field in (
            "common_gross_pnl_jpy_micros",
            "realized_cost_jpy_micros",
            "fill_sizing_drag_jpy_micros",
            "latency_spread_slippage_drag_jpy_micros",
            "commission_jpy_micros",
            "financing_jpy_micros",
            "admission_opportunity_drag_jpy_micros",
            "net_pnl_jpy_micros",
        ):
            _integer(row.get(field), f"{label}.{field}")
        if type(row.get("terminal_inventory_mtm_jpy_micros")) is not int \
                or row["terminal_inventory_mtm_jpy_micros"] != 0 \
                or type(row.get("external_order_count")) is not int \
                or row["external_order_count"] != 0:
            raise VerificationError(f"{label} authority invariant mismatch")
        previous = row["record_hash"]
        decoded_rows.append(row)
    if previous != terminal_hash:
        raise VerificationError("reference ledger terminal hash mismatch")
    expected_coverage = [
        (proposal_ordinal, arm)
        for proposal_ordinal in range(1, proposal_count + 1)
        for arm in ARMS
    ]
    actual_coverage = [
        (row["proposal_ordinal"], row["arm"])
        for row in decoded_rows
    ]
    if actual_coverage != expected_coverage:
        raise VerificationError("reference ledger arm/proposal coverage mismatch")
    return decoded_rows


def _reference_ratio_text(value: Any, label: str) -> str:
    if type(value) is not str or RATIO_TEXT_RE.fullmatch(value) is None \
            or value == "-0.000000000000000000":
        raise VerificationError(f"{label} must be exact 18-place ratio text")
    return value


def _validate_reference_cluster(
    value: Any,
    label: str,
    initial_equity: int,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise VerificationError(f"{label} must be exact object")
    _exact_keys(value, REFERENCE_CLUSTER_OBSERVATION_KEYS, label)
    _digest(value["cluster_id"], f"{label}.cluster_id")
    _digest(
        value["source_signal_set_sha256"],
        f"{label}.source_signal_set_sha256",
    )
    _integer(value["time_bucket"], f"{label}.time_bucket")
    ledger_pnl = _integer(
        value["ledger_net_pnl_jpy_micros"],
        f"{label}.ledger_net_pnl_jpy_micros",
    )
    risk_pnl = _integer(
        value["cluster_risk_net_pnl_jpy_micros"],
        f"{label}.cluster_risk_net_pnl_jpy_micros",
    )
    nodes = value["currency_nodes"]
    if type(nodes) is not list or len(nodes) < 2 \
            or any(
                type(node) is not str or CURRENCY_RE.fullmatch(node) is None
                for node in nodes
            ) \
            or nodes != sorted(set(nodes)):
        raise VerificationError(f"{label}.currency_nodes is not canonical")
    signed_return = _reference_ratio_text(
        value["signed_return"],
        f"{label}.signed_return",
    )
    if initial_equity <= 0 \
            or signed_return != _signed_ratio_text(risk_pnl, initial_equity):
        raise VerificationError(f"{label}.signed_return does not reconcile")
    # Both are independently typed even though exact-rational cluster rounding
    # can legitimately make them differ after economic-lot aggregation.
    _integer(ledger_pnl, f"{label}.ledger_net_pnl_jpy_micros")
    return value


def _validate_reference_month(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise VerificationError(f"{label} must be exact object")
    _exact_keys(value, REFERENCE_MONTHLY_OBSERVATION_KEYS, label)
    month_id = value["month_id"]
    if type(month_id) is not str or MONTH_ID_RE.fullmatch(month_id) is None:
        raise VerificationError(f"{label}.month_id is invalid")
    comparable = value["comparable_full_month"]
    ruin = value["ruin_observed"]
    if type(comparable) is not bool or type(ruin) is not bool:
        raise VerificationError(f"{label} boolean field type mismatch")
    segment_start = _integer(
        value["segment_start_ts_ns"], f"{label}.segment_start_ts_ns"
    )
    segment_end = _integer(
        value["segment_end_ts_ns"], f"{label}.segment_end_ts_ns"
    )
    start_equity = _integer(
        value["start_equity_jpy_micros"],
        f"{label}.start_equity_jpy_micros",
    )
    end_equity = _integer(
        value["end_equity_jpy_micros"],
        f"{label}.end_equity_jpy_micros",
    )
    if segment_start >= segment_end:
        raise VerificationError(f"{label} segment is empty or reversed")
    status = value["equity_multiple_status"]
    multiple = value["equity_multiple"]
    if start_equity > 0:
        if status != "DEFINED" \
                or _reference_ratio_text(
                    multiple, f"{label}.equity_multiple"
                ) != _ratio_text(end_equity, start_equity):
            raise VerificationError(f"{label} defined equity multiple mismatch")
    elif status != "UNDEFINED_NONPOSITIVE_START_EQUITY" or multiple is not None:
        raise VerificationError(f"{label} undefined equity multiple mismatch")
    if ruin is not (start_equity <= 0 or end_equity <= 0):
        raise VerificationError(f"{label}.ruin_observed does not reconcile")
    return value


def _validate_reference_arm_metrics(
    value: Any,
    arm: str,
    initial_equity: int,
    expected_cvar_tail_bps: int | None,
) -> dict[str, Any]:
    label = f"reference oracle_metrics.arms.{arm}"
    if type(value) is not dict:
        raise VerificationError(f"{label} must be exact object")
    _exact_keys(value, REFERENCE_ARM_METRIC_KEYS, label)
    proposal_count = _integer(
        value["proposal_count"], f"{label}.proposal_count", 0
    )
    executed_count = _integer(
        value["executed_count"], f"{label}.executed_count", 0
    )
    if executed_count > proposal_count:
        raise VerificationError(f"{label}.executed_count exceeds proposals")
    disposition_counts = value["disposition_counts"]
    if type(disposition_counts) is not dict \
            or any(
                type(status) is not str or not status
                or type(count) is not int or count < 0
                for status, count in disposition_counts.items()
            ) \
            or sum(disposition_counts.values()) != proposal_count \
            or disposition_counts.get("FILLED_CLOSED", 0) != executed_count:
        raise VerificationError(f"{label}.disposition_counts is invalid")
    _digest(value["signal_id_set_sha256"], f"{label}.signal_id_set_sha256")
    for field in REFERENCE_ARM_INTEGER_FIELDS:
        _integer(value[field], f"{label}.{field}")
    if value["max_drawdown_jpy_micros"] < 0 \
            or value["max_gross_notional_jpy_micros"] < 0 \
            or value["maximum_required_margin_jpy_micros"] < 0 \
            or not 0 <= value["cvar_tail_bps"] <= 10_000:
        raise VerificationError(f"{label} nonnegative metric invariant mismatch")
    if expected_cvar_tail_bps is not None \
            and value["cvar_tail_bps"] != expected_cvar_tail_bps:
        raise VerificationError(f"{label}.cvar_tail_bps policy mismatch")
    for field in (
        "ending_equity_multiple",
        "direction_accuracy",
        "max_drawdown_ratio",
        "cluster_cvar_return",
    ):
        _reference_ratio_text(value[field], f"{label}.{field}")
    if value["ending_equity_jpy_micros"] \
            != initial_equity + value["net_pnl_jpy_micros"] \
            or value["ending_equity_multiple"] != _ratio_text(
                value["ending_equity_jpy_micros"], initial_equity
            ):
        raise VerificationError(f"{label} ending equity does not reconcile")
    decomposed_drag = (
        value["fill_sizing_drag_jpy_micros"]
        + value["latency_spread_slippage_drag_jpy_micros"]
        + value["direct_commission_financing_cost_jpy_micros"]
        + value["admission_opportunity_drag_jpy_micros"]
    )
    if value["total_execution_and_admission_drag_jpy_micros"] \
            != decomposed_drag \
            or value["common_gross_pnl_jpy_micros"] \
            - value["net_pnl_jpy_micros"] != decomposed_drag:
        raise VerificationError(f"{label} drag attribution does not reconcile")
    if type(value["margin_guard_pass"]) is not bool:
        raise VerificationError(f"{label}.margin_guard_pass must be exact boolean")
    if type(value["terminal_open_positions"]) is not int \
            or value["terminal_open_positions"] != 0 \
            or type(value["terminal_inventory_mtm_jpy_micros"]) is not int \
            or value["terminal_inventory_mtm_jpy_micros"] != 0:
        raise VerificationError(f"{label} terminal authority invariant mismatch")
    observations = value["currency_time_cluster_observations"]
    n_eff = _integer(
        value["currency_time_cluster_n_eff"],
        f"{label}.currency_time_cluster_n_eff",
        0,
    )
    if type(observations) is not list or len(observations) != n_eff:
        raise VerificationError(f"{label} cluster observation count mismatch")
    for index, observation in enumerate(observations):
        _validate_reference_cluster(
            observation,
            f"{label}.currency_time_cluster_observations[{index}]",
            initial_equity,
        )
    cluster_ids = [observation["cluster_id"] for observation in observations]
    if cluster_ids != sorted(cluster_ids) or len(cluster_ids) != len(set(cluster_ids)):
        raise VerificationError(f"{label} cluster ordering is not canonical")
    monthly = value["monthly"]
    if type(monthly) is not list:
        raise VerificationError(f"{label}.monthly must be exact list")
    for index, observation in enumerate(monthly):
        _validate_reference_month(
            observation,
            f"{label}.monthly[{index}]",
        )
    month_ids = [observation["month_id"] for observation in monthly]
    if month_ids != sorted(month_ids) or len(month_ids) != len(set(month_ids)):
        raise VerificationError(f"{label} month ordering is not canonical")
    for index in range(1, len(monthly)):
        previous = monthly[index - 1]
        current = monthly[index]
        if previous["segment_end_ts_ns"] != current["segment_start_ts_ns"] \
                or previous["end_equity_jpy_micros"] \
                != current["start_equity_jpy_micros"]:
            raise VerificationError(f"{label} month boundary is discontinuous")
    return value


def _validate_reference_metrics(
    metrics: Any,
    *,
    expected_initial_equity: int | None = None,
    expected_cvar_tail_bps: int | None = None,
) -> dict[str, Any]:
    if type(metrics) is not dict:
        raise VerificationError("reference oracle_metrics must be exact dict")
    _exact_keys(metrics, REFERENCE_METRICS_KEYS, "reference oracle_metrics")
    _assert_canonical_value(metrics, "reference oracle_metrics")
    if _integer(metrics["schema_version"], "reference metrics schema") != 2:
        raise VerificationError("reference metrics version mismatch")
    initial_equity = _integer(
        metrics["initial_equity_jpy_micros"],
        "reference metrics initial equity",
        1,
    )
    if expected_initial_equity is not None \
            and initial_equity != expected_initial_equity:
        raise VerificationError("reference metrics initial equity policy mismatch")
    for field in (
        "same_signal_ids_all_arms",
        "all_proposals_have_all_arm_dispositions",
        "common_gross_reference_shared",
    ):
        if type(metrics[field]) is not bool or metrics[field] is not True:
            raise VerificationError(f"reference metrics {field} must be exact true")
    arms = metrics["arms"]
    if type(arms) is not dict or set(arms) != set(ARMS):
        raise VerificationError("reference metrics arm schema mismatch")
    for arm in ARMS:
        _validate_reference_arm_metrics(
            arms[arm],
            arm,
            initial_equity,
            expected_cvar_tail_bps,
        )
    if len({arms[arm]["proposal_count"] for arm in ARMS}) != 1 \
            or len({arms[arm]["signal_id_set_sha256"] for arm in ARMS}) != 1 \
            or len({arms[arm]["common_gross_pnl_jpy_micros"] for arm in ARMS}) \
                != 1:
        raise VerificationError("reference metrics cross-arm stream mismatch")
    month_shapes = {
        tuple(
            (
                row["month_id"],
                row["comparable_full_month"],
                row["segment_start_ts_ns"],
                row["segment_end_ts_ns"],
            )
            for row in arms[arm]["monthly"]
        )
        for arm in ARMS
    }
    if len(month_shapes) != 1:
        raise VerificationError("reference metrics cross-arm month grid mismatch")
    if type(metrics["external_orders"]) is not int \
            or metrics["external_orders"] != 0 \
            or type(metrics["terminal_inventory_mtm_jpy_micros"]) is not int \
            or metrics["terminal_inventory_mtm_jpy_micros"] != 0:
        raise VerificationError("reference metrics authority invariant mismatch")
    _validate_embedded(metrics, "metrics_sha256", "reference oracle_metrics")
    return metrics


def _validate_reference_metrics_against_ledger(
    metrics: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    proposal_count: int,
) -> None:
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        arm_metrics = metrics["arms"][arm]
        statuses = Counter(row["status"] for row in arm_rows)
        executed = sum(row["status"] == "FILLED_CLOSED" for row in arm_rows)
        expected_values = {
            "proposal_count": proposal_count,
            "executed_count": executed,
            "disposition_counts": dict(sorted(statuses.items())),
            "signal_id_set_sha256": sha256_bytes(canonical_bytes(sorted(
                row["signal_id"] for row in arm_rows
            ))),
            "common_gross_pnl_jpy_micros": sum(
                row["common_gross_pnl_jpy_micros"] for row in arm_rows
            ),
            "realized_cost_jpy_micros": sum(
                row["realized_cost_jpy_micros"] for row in arm_rows
            ),
            "fill_sizing_drag_jpy_micros": sum(
                row["fill_sizing_drag_jpy_micros"] for row in arm_rows
            ),
            "latency_spread_slippage_drag_jpy_micros": sum(
                row["latency_spread_slippage_drag_jpy_micros"]
                for row in arm_rows
            ),
            "direct_commission_financing_cost_jpy_micros": sum(
                row["commission_jpy_micros"] + row["financing_jpy_micros"]
                for row in arm_rows
            ),
            "admission_opportunity_drag_jpy_micros": sum(
                row["admission_opportunity_drag_jpy_micros"]
                for row in arm_rows
            ),
            "net_pnl_jpy_micros": sum(
                row["net_pnl_jpy_micros"] for row in arm_rows
            ),
        }
        if any(
            arm_metrics[field] != expected
            for field, expected in expected_values.items()
        ):
            raise VerificationError(
                f"reference metrics {arm} does not reconcile to ledger"
            )


def _reference_proposal_provenance_root(
    proposal: Mapping[str, Any],
) -> str:
    return sha256_bytes(canonical_bytes({
        "provenance": proposal["provenance"],
        "rows": [
            {
                "proposal_ordinal": row["proposal_ordinal"],
                "decision_source_event_sha256": row[
                    "decision_source_event_sha256"
                ],
                "completed_data_watermark_source_ts_ns": row[
                    "completed_data_watermark_source_ts_ns"
                ],
                "completed_data_prefix_root_sha256": row[
                    "completed_data_prefix_root_sha256"
                ],
            }
            for row in proposal["rows"]
        ],
    }))


def _reference_result_snapshot_bytes(result: Mapping[str, Any]) -> bytes:
    if type(result) is not dict:
        raise VerificationError("reference result must be exact dict")
    _exact_keys(result, REFERENCE_RESULT_KEYS, "reference result")
    ledger_bytes = result["ledger_bytes"]
    if type(ledger_bytes) is not bytes:
        raise VerificationError("reference ledger_bytes must be exact bytes")
    snapshot = {
        key: result[key]
        for key in REFERENCE_RESULT_KEYS
        if key != "ledger_bytes"
    }
    snapshot["ledger_bytes_base64"] = base64.b64encode(ledger_bytes).decode("ascii")
    return canonical_bytes(snapshot) + b"\n"


def _decode_reference_result_snapshot(data: Any) -> dict[str, Any]:
    if type(data) is not bytes:
        raise VerificationError("reference result snapshot must be exact bytes")
    snapshot = strict_json(data, "reference result snapshot")
    _exact_keys(
        snapshot,
        REFERENCE_RESULT_SNAPSHOT_KEYS,
        "reference result snapshot",
    )
    encoded = snapshot["ledger_bytes_base64"]
    if type(encoded) is not str:
        raise VerificationError("reference ledger base64 must be exact text")
    try:
        encoded_bytes = encoded.encode("ascii", errors="strict")
        ledger_bytes = base64.b64decode(encoded_bytes, validate=True)
    except (UnicodeEncodeError, ValueError) as error:
        raise VerificationError("reference ledger base64 is invalid") from error
    if base64.b64encode(ledger_bytes) != encoded_bytes:
        raise VerificationError("reference ledger base64 is not canonical")
    return {
        key: snapshot[key]
        for key in REFERENCE_RESULT_KEYS
        if key != "ledger_bytes"
    } | {"ledger_bytes": ledger_bytes}


def _validate_reference_result_value(
    state: Mapping[str, Any],
    result: Any,
) -> dict[str, Any]:
    if type(result) is not dict:
        raise VerificationError("reference result must be exact dict")
    _exact_keys(result, REFERENCE_RESULT_KEYS, "reference result")
    if type(result["engine_id"]) is not str \
            or result["engine_id"] != REFERENCE_ENGINE_ID:
        raise VerificationError("reference engine identity mismatch")
    for field in (
        "input_root_sha256",
        "ledger_terminal_hash",
        "proposal_provenance_root_sha256",
        "journal_root_sha256",
        "economic_projection_sha256",
    ):
        _digest(result[field], f"reference result {field}")
    if type(result["ledger_bytes"]) is not bytes:
        raise VerificationError("reference ledger_bytes must be exact bytes")
    ledger_row_count = _integer(
        result["ledger_row_count"], "reference ledger_row_count", 0
    )
    journal_count = _integer(
        result["journal_transaction_count"],
        "reference journal_transaction_count",
        0,
    )
    if type(result["all_transactions_balanced"]) is not bool \
            or result["all_transactions_balanced"] is not True:
        raise VerificationError("reference journal is not exactly balanced")
    metrics = _validate_reference_metrics(
        result["oracle_metrics"],
        expected_initial_equity=state["evaluation"][
            "initial_equity_jpy_micros"
        ],
        expected_cvar_tail_bps=state["evaluation"]["cvar_tail_bps"],
    )
    independent_input_root = _reference_input_root(state)
    if result["input_root_sha256"] != independent_input_root:
        raise VerificationError("reference input root mismatch")
    proposal_count = len(state["proposal"]["rows"])
    ledger_rows = _validate_reference_ledger(
        result["ledger_bytes"],
        ledger_row_count,
        result["ledger_terminal_hash"],
        proposal_count,
    )
    _validate_reference_metrics_against_ledger(
        metrics,
        ledger_rows,
        proposal_count,
    )
    expected_provenance_root = _reference_proposal_provenance_root(
        state["proposal"]
    )
    if result["proposal_provenance_root_sha256"] \
            != expected_provenance_root:
        raise VerificationError("reference proposal provenance root mismatch")
    projection = {
        "all_transactions_balanced": True,
        "engine_id": REFERENCE_ENGINE_ID,
        "input_root_sha256": independent_input_root,
        "journal_root_sha256": result["journal_root_sha256"],
        "journal_transaction_count": journal_count,
        "ledger_row_count": ledger_row_count,
        "ledger_sha256": sha256_bytes(result["ledger_bytes"]),
        "ledger_terminal_hash": result["ledger_terminal_hash"],
        "oracle_metrics_sha256": metrics["metrics_sha256"],
        "proposal_provenance_root_sha256": result[
            "proposal_provenance_root_sha256"
        ],
    }
    if result["economic_projection_sha256"] != sha256_bytes(
        canonical_bytes(projection)
    ):
        raise VerificationError("reference economic projection hash mismatch")
    return result


def _validate_reference_result(
    state: Mapping[str, Any],
    reference_replay: Callable[[Mapping[str, bytes]], Any],
) -> dict[str, Any]:
    """Path-loaded test adapter; sealed verification never calls a callable."""
    artifacts = {
        label: state["blobs"][label]
        for label in REFERENCE_INPUT_LABELS
    }
    if set(artifacts) != set(REFERENCE_INPUT_LABELS) \
            or any(type(value) is not bytes for value in artifacts.values()):
        raise VerificationError("internal reference artifact boundary invalid")
    try:
        result = reference_replay(artifacts)
    except BaseException as error:
        raise VerificationError("test reference replay failed") from error
    return _validate_reference_result_value(state, result)


def _expected_manifest_from_reference(
    state: Mapping[str, Any],
    reference_result: Mapping[str, Any],
) -> dict[str, Any]:
    input_hashes = {
        label: state["hashes"][label]
        for label in REFERENCE_INPUT_LABELS
    }
    ledger_bytes = reference_result["ledger_bytes"]
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "oracle_implementation": ORACLE_NAME,
        "status": "COMPLETE",
        "classification": CLASSIFICATION,
        "causal_signal_admission": False,
        "release_evidence_eligible": False,
        "detector_replay_receipt_required": True,
        "authority": dict(_authority_items()),
        "oracle_release_content_binding": {
            "code_sha256": state["oracle_release"]["hashes"]["code_sha256"],
            "contract_sha256": state["oracle_release"]["hashes"]["contract_sha256"],
            "schema_sha256": state["oracle_release"]["hashes"]["schema_sha256"],
            "launcher_sha256": _LAUNCHER_SHA256,
            "snapshot_mode": _EXECUTION_SNAPSHOT_MODE,
        },
        "oracle_execution_provenance_scope": EXECUTION_PROVENANCE_SCOPE,
        "request_sha256": sha256_bytes(state["blobs"]["oracle_request"]),
        "input_artifact_sha256": dict(sorted(input_hashes.items())),
        "raw_source_manifest_sha256": state["hashes"]["source_manifest"],
        "proposal_provenance_root_sha256": reference_result[
            "proposal_provenance_root_sha256"
        ],
        "producer_result_or_metrics_used": False,
        "proposal_identity_generated_by_oracle": True,
        "oracle_ledger_file": "oracle_ledger.jsonl",
        "oracle_ledger_sha256": sha256_bytes(ledger_bytes),
        "oracle_ledger_size_bytes": len(ledger_bytes),
        "oracle_ledger_row_count": reference_result["ledger_row_count"],
        "oracle_ledger_terminal_hash": reference_result["ledger_terminal_hash"],
        "oracle_metrics": reference_result["oracle_metrics"],
        "terminal_inventory_mtm_jpy_micros": 0,
        "external_orders": 0,
        "anchor_status": ANCHOR_STATUS,
    }
    manifest["oracle_root_sha256"] = embedded_hash(manifest, "oracle_root_sha256")
    return manifest


def _verify_actual_evidence_from_reference(
    state: Mapping[str, Any],
    reference_result: Mapping[str, Any],
    reference_result_sha256: str,
) -> dict[str, Any]:
    _digest(reference_result_sha256, "reference result snapshot hash")
    expected_ledger = reference_result["ledger_bytes"]
    expected_manifest = _expected_manifest_from_reference(state, reference_result)
    actual_ledger = state["blobs"]["oracle_ledger"]
    actual_manifest_bytes = state["blobs"]["oracle_manifest"]
    actual_manifest = strict_json(actual_manifest_bytes, "oracle manifest")
    if actual_ledger != expected_ledger:
        raise VerificationError("oracle ledger differs from independent canonical replay")
    if actual_manifest.get("oracle_metrics") != reference_result["oracle_metrics"]:
        raise VerificationError("oracle metrics differ from independent reference replay")
    if actual_manifest.get("proposal_provenance_root_sha256") != reference_result[
        "proposal_provenance_root_sha256"
    ]:
        raise VerificationError("oracle proposal provenance differs from reference replay")
    expected_manifest_bytes = canonical_bytes(expected_manifest) + b"\n"
    if actual_manifest_bytes != expected_manifest_bytes:
        raise VerificationError("oracle manifest differs from independent canonical replay")
    _validate_authority_exact(actual_manifest.get("authority"), "oracle manifest authority")
    if actual_manifest["classification"] != CLASSIFICATION \
            or type(actual_manifest["causal_signal_admission"]) is not bool \
            or actual_manifest["causal_signal_admission"] is not False \
            or type(actual_manifest["external_orders"]) is not int \
            or actual_manifest["external_orders"] != 0 \
            or type(actual_manifest["terminal_inventory_mtm_jpy_micros"]) is not int \
            or actual_manifest["terminal_inventory_mtm_jpy_micros"] != 0:
        raise VerificationError("oracle manifest authority/classification mismatch")
    receipt: dict[str, Any] = {
        "schema_version": 2,
        "verifier_implementation": VERIFIER_NAME,
        "status": "VERIFIED_ACCOUNTING_ONLY",
        "classification": CLASSIFICATION,
        "causal_signal_admission": False,
        "release_evidence_eligible": False,
        "admission_eligible": False,
        "detector_replay_receipt_required": True,
        "authority": dict(_authority_items()),
        "oracle_root_sha256": actual_manifest["oracle_root_sha256"],
        "oracle_manifest_sha256": sha256_bytes(actual_manifest_bytes),
        "oracle_manifest_size_bytes": len(actual_manifest_bytes),
        "oracle_ledger_sha256": sha256_bytes(actual_ledger),
        "oracle_ledger_size_bytes": len(actual_ledger),
        "expected_canonical_ledger_sha256": sha256_bytes(expected_ledger),
        "oracle_ledger_terminal_hash": actual_manifest["oracle_ledger_terminal_hash"],
        "raw_source_manifest_sha256": state["hashes"]["source_manifest"],
        "oracle_request_sha256": state["hashes"]["oracle_request"],
        "oracle_release_content_binding": dict(
            actual_manifest["oracle_release_content_binding"]
        ),
        "oracle_execution_provenance_scope": actual_manifest[
            "oracle_execution_provenance_scope"
        ],
        "verifier_release_content_binding": {
            "code_sha256": _MODULE_CODE_SHA256,
            "schema_sha256": _SCHEMA_SHA256,
            "launcher_sha256": _LAUNCHER_SHA256,
            "snapshot_mode": _EXECUTION_SNAPSHOT_MODE,
            "reference_code_sha256": state["reference_release"]["hashes"][
                "code_sha256"
            ],
            "reference_contract_sha256": state["reference_release"]["hashes"][
                "contract_sha256"
            ],
            "reference_result_sha256": reference_result_sha256,
        },
        "verifier_execution_provenance_scope": EXECUTION_PROVENANCE_SCOPE,
        "input_artifact_sha256": dict(sorted(state["hashes"].items())),
        "independently_rebuilt_ledger": True,
        "independently_rebuilt_metrics": True,
        "producer_result_or_metrics_used": False,
        "verified_oracle_metrics": reference_result["oracle_metrics"],
        "reference_engine_id": reference_result["engine_id"],
        "reference_code_sha256": state["reference_release"]["hashes"][
            "code_sha256"
        ],
        "reference_contract_sha256": state["reference_release"]["hashes"][
            "contract_sha256"
        ],
        "reference_result_sha256": reference_result_sha256,
        "reference_input_root_sha256": reference_result["input_root_sha256"],
        "reference_journal_root_sha256": reference_result["journal_root_sha256"],
        "reference_journal_transaction_count": reference_result[
            "journal_transaction_count"
        ],
        "reference_all_transactions_balanced": reference_result[
            "all_transactions_balanced"
        ],
        "reference_economic_projection_sha256": reference_result[
            "economic_projection_sha256"
        ],
        "reference_accounting_diagnostics_only": True,
        "reference_n_eff_statistical_admission_allowed": False,
        "reference_direction_accuracy_profit_gate_allowed": False,
        "terminal_inventory_mtm_jpy_micros": 0,
        "external_orders": 0,
        "anchor_status": ANCHOR_STATUS,
    }
    receipt["verifier_receipt_sha256"] = embedded_hash(receipt, "verifier_receipt_sha256")
    return receipt


def _verify_actual_evidence(
    state: Mapping[str, Any],
    reference_replay: Callable[[Mapping[str, bytes]], Any],
) -> dict[str, Any]:
    """Path-loaded test adapter; authoritative sealed verification is byte-only."""
    reference_result = _validate_reference_result(state, reference_replay)
    result_bytes = _reference_result_snapshot_bytes(reference_result)
    return _verify_actual_evidence_from_reference(
        state,
        reference_result,
        sha256_bytes(result_bytes),
    )


def _validate_pure_output_bytes(
    request_bytes: bytes,
    expected_receipt: Mapping[str, Any],
    receipt_bytes: Any,
    commit_bytes: Any,
) -> None:
    if type(receipt_bytes) is not bytes or type(commit_bytes) is not bytes:
        raise VerificationError("verifier output must be exact immutable bytes")
    receipt = strict_json(receipt_bytes, "verifier receipt")
    commit = strict_json(commit_bytes, "verifier commit")
    _exact_keys(receipt, VERIFIER_RECEIPT_KEYS, "verifier receipt")
    _exact_keys(
        commit,
        {
            "schema_version",
            "request_sha256",
            "receipt_sha256",
            "receipt_size_bytes",
            "verifier_receipt_sha256",
        },
        "verifier commit",
    )
    if receipt != expected_receipt:
        raise VerificationError("verifier receipt semantic output mismatch")
    _validate_authority_exact(receipt["authority"], "verifier receipt authority")
    _validate_embedded(receipt, "verifier_receipt_sha256", "verifier receipt")
    binding = receipt["verifier_release_content_binding"]
    if type(binding) is not dict:
        raise VerificationError("verifier release binding must be exact object")
    _exact_keys(binding, VERIFIER_RELEASE_BINDING_KEYS, "verifier release binding")
    for field in (
        "code_sha256",
        "schema_sha256",
        "launcher_sha256",
        "reference_code_sha256",
        "reference_contract_sha256",
        "reference_result_sha256",
    ):
        _digest(binding[field], f"verifier release binding {field}")
    if binding["snapshot_mode"] != "SEALED_FD_COMPILE_EXEC_V2" \
            or binding["reference_code_sha256"] != receipt["reference_code_sha256"] \
            or binding["reference_contract_sha256"] != receipt["reference_contract_sha256"] \
            or binding["reference_result_sha256"] != receipt["reference_result_sha256"]:
        raise VerificationError("verifier release binding link mismatch")
    for field in (
        "oracle_root_sha256",
        "oracle_manifest_sha256",
        "oracle_ledger_sha256",
        "expected_canonical_ledger_sha256",
        "oracle_ledger_terminal_hash",
        "raw_source_manifest_sha256",
        "oracle_request_sha256",
        "reference_code_sha256",
        "reference_contract_sha256",
        "reference_result_sha256",
        "reference_input_root_sha256",
        "reference_journal_root_sha256",
        "reference_economic_projection_sha256",
        "verifier_receipt_sha256",
    ):
        _digest(receipt[field], f"verifier receipt {field}")
    for field in (
        "causal_signal_admission",
        "release_evidence_eligible",
        "admission_eligible",
        "producer_result_or_metrics_used",
        "reference_n_eff_statistical_admission_allowed",
        "reference_direction_accuracy_profit_gate_allowed",
    ):
        if type(receipt[field]) is not bool or receipt[field] is not False:
            raise VerificationError(f"verifier receipt {field} must be exact false")
    for field in (
        "detector_replay_receipt_required",
        "independently_rebuilt_ledger",
        "independently_rebuilt_metrics",
        "reference_all_transactions_balanced",
        "reference_accounting_diagnostics_only",
    ):
        if type(receipt[field]) is not bool or receipt[field] is not True:
            raise VerificationError(f"verifier receipt {field} must be exact true")
    if _integer(receipt["schema_version"], "verifier receipt schema") != 2 \
            or _integer(
                receipt["oracle_manifest_size_bytes"],
                "verifier Oracle manifest size",
                0,
            ) < 0 \
            or _integer(
                receipt["oracle_ledger_size_bytes"],
                "verifier Oracle ledger size",
                0,
            ) < 0 \
            or _integer(
                receipt["reference_journal_transaction_count"],
                "verifier reference journal count",
                0,
            ) < 0 \
            or type(receipt["terminal_inventory_mtm_jpy_micros"]) is not int \
            or receipt["terminal_inventory_mtm_jpy_micros"] != 0 \
            or type(receipt["external_orders"]) is not int \
            or receipt["external_orders"] != 0:
        raise VerificationError("verifier receipt integer invariant mismatch")
    if receipt["verifier_implementation"] != VERIFIER_NAME \
            or receipt["status"] != "VERIFIED_ACCOUNTING_ONLY" \
            or receipt["classification"] != CLASSIFICATION \
            or receipt["reference_engine_id"] != REFERENCE_ENGINE_ID \
            or receipt["anchor_status"] != ANCHOR_STATUS:
        raise VerificationError("verifier receipt identity/classification mismatch")
    if type(receipt["input_artifact_sha256"]) is not dict \
            or set(receipt["input_artifact_sha256"]) != set(SEALED_ARTIFACT_ROLES):
        raise VerificationError("verifier receipt artifact hash map mismatch")
    for role in SEALED_ARTIFACT_ROLES:
        _digest(
            receipt["input_artifact_sha256"][role],
            f"verifier receipt artifact hash {role}",
        )
    _validate_reference_metrics(receipt["verified_oracle_metrics"])
    if _integer(commit["schema_version"], "verifier commit schema") != 2 \
            or commit["request_sha256"] != sha256_bytes(request_bytes) \
            or commit["receipt_sha256"] != sha256_bytes(receipt_bytes) \
            or _integer(
                commit["receipt_size_bytes"],
                "verifier commit receipt size",
                0,
            ) != len(receipt_bytes) \
            or commit["verifier_receipt_sha256"] != receipt[
                "verifier_receipt_sha256"
            ]:
        raise VerificationError("verifier COMMIT output binding mismatch")


def verify_sealed_bytes(
    request_bytes: bytes,
    artifact_blobs: tuple[tuple[str, bytes], ...],
    oracle_release_blobs: tuple[tuple[str, bytes], ...],
    reference_result_bytes: bytes,
    reference_attestation: tuple[tuple[str, str], ...],
) -> tuple[bytes, bytes]:
    """Pure authoritative verifier: immutable values in, canonical bytes out."""
    if _SEALED_RUNTIME is not True:
        raise VerificationError("pure production entrypoint requires sealed runtime")
    if type(request_bytes) is not bytes:
        raise VerificationError("verifier request must be exact bytes")
    request = strict_json(request_bytes, "verifier request")
    blobs = _decode_exact_tuple_values(
        artifact_blobs,
        SEALED_ARTIFACT_ROLES,
        bytes,
        "sealed artifact blobs",
    )
    oracle_release_values = _decode_exact_tuple_values(
        oracle_release_blobs,
        SEALED_ORACLE_RELEASE_ROLES,
        bytes,
        "sealed Oracle release blobs",
    )
    attestation = _decode_exact_tuple_values(
        reference_attestation,
        REFERENCE_ATTESTATION_KEYS,
        str,
        "sealed reference attestation",
    )
    for field in REFERENCE_ATTESTATION_KEYS:
        _digest(attestation[field], f"sealed reference attestation {field}")
    oracle_release = {
        **oracle_release_values,
        "hashes": {
            "code_sha256": sha256_bytes(oracle_release_values["code_bytes"]),
            "contract_sha256": sha256_bytes(
                oracle_release_values["contract_bytes"]
            ),
            "schema_sha256": sha256_bytes(oracle_release_values["schema_bytes"]),
        },
    }
    reference_release = {
        "code_bytes": blobs["reference_code_snapshot"],
        "contract_bytes": blobs["reference_contract_snapshot"],
        "hashes": {
            "code_sha256": sha256_bytes(blobs["reference_code_snapshot"]),
            "contract_sha256": sha256_bytes(
                blobs["reference_contract_snapshot"]
            ),
        },
    }
    if reference_release["hashes"] != {
        "code_sha256": attestation["reference_code_sha256"],
        "contract_sha256": attestation["reference_contract_sha256"],
    } or reference_release["hashes"] != SUPPORTED_REFERENCE_RELEASE:
        raise VerificationError("reference attestation release binding mismatch")
    if sha256_bytes(reference_result_bytes) != attestation[
        "reference_result_sha256"
    ]:
        raise VerificationError("reference result snapshot hash mismatch")
    state = _load_request_from_bound_blobs(
        request,
        blobs,
        oracle_release,
        reference_release,
    )
    reference_result = _validate_reference_result_value(
        state,
        _decode_reference_result_snapshot(reference_result_bytes),
    )
    receipt = _verify_actual_evidence_from_reference(
        state,
        reference_result,
        attestation["reference_result_sha256"],
    )
    receipt_bytes, commit_bytes = _receipt_output_bytes(
        receipt,
        sha256_bytes(request_bytes),
    )
    _validate_pure_output_bytes(
        request_bytes,
        receipt,
        receipt_bytes,
        commit_bytes,
    )
    return receipt_bytes, commit_bytes


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        try:
            written = os.write(descriptor, view)
        except InterruptedError:
            continue
        if written <= 0:
            raise VerificationError("short immutable output write")
        view = view[written:]


def _write_file_at(directory_fd: int, name: str, data: bytes) -> None:
    if SAFE_COMPONENT_RE.fullmatch(name) is None:
        raise VerificationError("unsafe receipt filename")
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() \
                or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600:
            raise VerificationError("receipt output is not owned regular file")
        if len(data) > MAX_ARTIFACT_BYTES:
            raise VerificationError("receipt output exceeds fixed byte limit")
        _write_all(descriptor, data)
        os.fsync(descriptor)
        final = os.fstat(descriptor)
        if not stat.S_ISREG(final.st_mode) or final.st_uid != os.geteuid() \
                or final.st_nlink != 1 or stat.S_IMODE(final.st_mode) != 0o600 \
                or final.st_size != len(data):
            raise VerificationError("receipt output changed while writing")
    finally:
        os.close(descriptor)


def _lstat_at(directory_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _child_file_set(directory_fd: int, child: str) -> set[str]:
    child_fd = os.open(
        child,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        info = os.fstat(child_fd)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o022:
            raise VerificationError("receipt child directory ownership/mode invalid")
        return set(os.listdir(child_fd))
    finally:
        os.close(child_fd)


def _read_child_file(directory_fd: int, child: str, filename: str) -> bytes:
    child_fd = os.open(
        child,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        info = os.fstat(child_fd)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o022:
            raise VerificationError("receipt child directory ownership/mode invalid")
        return _read_relative(child_fd, filename, f"receipt {filename}")
    finally:
        os.close(child_fd)


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_nlink,
        info.st_uid,
    )


def _open_bound_file_set(
    directory_fd: int,
    expected_names: set[str],
    label: str,
) -> dict[str, dict[str, Any]]:
    if set(os.listdir(directory_fd)) != expected_names:
        raise VerificationError(f"{label} file set mismatch")
    held: dict[str, dict[str, Any]] = {}
    try:
        for name in sorted(expected_names):
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=directory_fd,
            )
            info = os.fstat(descriptor)
            path_info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() \
                    or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1 \
                    or _stat_identity(info) != _stat_identity(path_info):
                os.close(descriptor)
                raise VerificationError(f"{label} child is not a private bound regular file")
            data = _read_fd_snapshot(descriptor, f"{label} {name}")
            held[name] = {
                "fd": descriptor,
                "identity": _stat_identity(info),
                "sha256": sha256_bytes(data),
                "bytes": data,
            }
        _revalidate_bound_file_set(directory_fd, expected_names, held, label)
        return held
    except BaseException:
        for item in held.values():
            os.close(item["fd"])
        raise


def _revalidate_bound_file_set(
    directory_fd: int,
    expected_names: set[str],
    held: Mapping[str, Mapping[str, Any]],
    label: str,
) -> None:
    if set(os.listdir(directory_fd)) != expected_names or set(held) != expected_names:
        raise VerificationError(f"{label} file set changed")
    for name in sorted(expected_names):
        item = held[name]
        descriptor = item["fd"]
        fd_info = os.fstat(descriptor)
        path_info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if _stat_identity(fd_info) != item["identity"] \
                or _stat_identity(path_info) != item["identity"] \
                or fd_info.st_nlink != 1:
            raise VerificationError(f"{label} child identity changed: {name}")
        data = _read_fd_snapshot(descriptor, f"{label} {name} revalidation")
        if sha256_bytes(data) != item["sha256"] or data != item["bytes"]:
            raise VerificationError(f"{label} child bytes changed: {name}")


def _close_bound_file_set(held: Mapping[str, Mapping[str, Any]]) -> None:
    for name in sorted(held, reverse=True):
        os.close(held[name]["fd"])


def _receipt_output_bytes(
    receipt: Mapping[str, Any], request_sha256: str
) -> tuple[bytes, bytes]:
    receipt_bytes = canonical_bytes(receipt) + b"\n"
    commit = {
        "schema_version": 2,
        "request_sha256": request_sha256,
        "receipt_sha256": sha256_bytes(receipt_bytes),
        "receipt_size_bytes": len(receipt_bytes),
        "verifier_receipt_sha256": receipt["verifier_receipt_sha256"],
    }
    return receipt_bytes, canonical_bytes(commit) + b"\n"


def _validate_receipt_output(
    output_root_fd: int,
    output_name: str,
    expected_receipt: Mapping[str, Any],
    request_sha256: str,
) -> dict[str, Any]:
    info = _lstat_at(output_root_fd, output_name)
    if info is None or not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o022:
        raise VerificationError("receipt output directory invalid")
    child_fd = os.open(
        output_name,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=output_root_fd,
    )
    try:
        return _validate_receipt_output_fd(
            child_fd, expected_receipt, request_sha256
        )
    finally:
        os.close(child_fd)


def _validate_receipt_output_fd(
    child_fd: int,
    expected_receipt: Mapping[str, Any],
    request_sha256: str,
) -> dict[str, Any]:
    info = os.fstat(child_fd)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o022:
        raise VerificationError("receipt output FD is not trusted directory")
    expected_names = {"verifier_receipt.json", "COMMIT.json"}
    held = _open_bound_file_set(child_fd, expected_names, "receipt output")
    try:
        receipt_bytes = held["verifier_receipt.json"]["bytes"]
        commit_bytes = held["COMMIT.json"]["bytes"]
        receipt = strict_json(receipt_bytes, "verifier receipt")
        commit = strict_json(commit_bytes, "verifier commit")
        _exact_keys(
            commit,
            {"schema_version", "request_sha256", "receipt_sha256", "receipt_size_bytes", "verifier_receipt_sha256"},
            "verifier commit",
        )
        if _integer(commit["schema_version"], "verifier commit schema") != 2 \
                or commit["request_sha256"] != request_sha256 \
                or commit["receipt_sha256"] != sha256_bytes(receipt_bytes) \
                or _integer(
                    commit["receipt_size_bytes"], "verifier receipt size", 0
                ) != len(receipt_bytes) \
                or commit["verifier_receipt_sha256"] != receipt.get("verifier_receipt_sha256") \
                or receipt_bytes != canonical_bytes(expected_receipt) + b"\n":
            raise VerificationError("verifier output binding mismatch")
        _revalidate_bound_file_set(
            child_fd, expected_names, held, "receipt output"
        )
        return receipt
    finally:
        _close_bound_file_set(held)


def _complete_or_recover_receipt_stage(
    output_root_fd: int,
    stage_name: str,
    receipt: Mapping[str, Any],
    request_sha256: str,
    assert_lock: Callable[[], None],
) -> tuple[int, os.stat_result]:
    stage_fd = os.open(
        stage_name,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=output_root_fd,
    )
    try:
        assert_lock()
        stage_info = os.fstat(stage_fd)
        if not stat.S_ISDIR(stage_info.st_mode) or stage_info.st_uid != os.geteuid() \
                or stage_info.st_mode & 0o022:
            raise VerificationError("verifier staging directory ownership/mode invalid")
        present = set(os.listdir(stage_fd))
        allowed = {"verifier_receipt.json", "COMMIT.json"}
        if not present <= allowed:
            raise VerificationError("verifier staging contains unexpected file")
        receipt_bytes, commit_bytes = _receipt_output_bytes(receipt, request_sha256)
        for filename, data in (
            ("verifier_receipt.json", receipt_bytes),
            ("COMMIT.json", commit_bytes),
        ):
            if filename in present:
                try:
                    assert_lock()
                    existing = _read_relative(
                        stage_fd, filename, f"verifier staging {filename}"
                    )
                    assert_lock()
                except LockIdentityError:
                    raise
                except OSError as error:
                    raise VerificationError(
                        f"verifier staging {filename} cannot be opened safely"
                    ) from error
                if existing != data:
                    raise VerificationError(f"verifier staging {filename} is partial or mismatched")
            else:
                assert_lock()
                _write_file_at(stage_fd, filename, data)
                assert_lock()
        assert_lock()
        os.fsync(stage_fd)
        assert_lock()
        return stage_fd, stage_info
    except BaseException:
        os.close(stage_fd)
        raise


def _materialize_final_receipt(
    output_root_fd: int,
    output_name: str,
    receipt: Mapping[str, Any],
    request_sha256: str,
    assert_lock: Callable[[], None],
    *,
    existing: bool,
) -> dict[str, Any]:
    final_fd = os.open(
        output_name,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=output_root_fd,
    )
    try:
        info = os.fstat(final_fd)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o022:
            raise VerificationError("verifier final directory invalid")
        receipt_bytes, commit_bytes = _receipt_output_bytes(receipt, request_sha256)
        expected = (
            ("verifier_receipt.json", receipt_bytes),
            ("COMMIT.json", commit_bytes),
        )
        present = set(os.listdir(final_fd))
        allowed = {name for name, _ in expected}
        if not present <= allowed:
            raise VerificationError("receipt output file set mismatch")
        if existing and "verifier_receipt.json" not in present:
            raise VerificationError("FAILED_VISIBLE_FINAL_WITHOUT_VERIFIER_RECEIPT")
        if "COMMIT.json" in present and present != allowed:
            raise VerificationError("verifier COMMIT exists with incomplete artifact set")
        for filename, data in expected:
            if filename in present:
                if _read_relative(final_fd, filename, f"verifier final {filename}") != data:
                    raise VerificationError(f"verifier final {filename} differs")
            else:
                assert_lock()
                _write_file_at(final_fd, filename, data)
                assert_lock()
        assert_lock()
        os.fsync(final_fd)
        assert_lock()
        result = _validate_receipt_output_fd(final_fd, receipt, request_sha256)
        assert_lock()
        return result
    finally:
        os.close(final_fd)


def _commit_completed_receipt_stage(
    output_root_fd: int,
    stage_fd: int,
    stage_info: os.stat_result,
    stage_name: str,
    output_name: str,
    receipt: Mapping[str, Any],
    request_sha256: str,
    assert_lock: Callable[[], None],
) -> dict[str, Any]:
    expected_names = {"verifier_receipt.json", "COMMIT.json"}
    held: dict[str, dict[str, Any]] | None = None
    try:
        assert_lock()
        _validate_receipt_output_fd(stage_fd, receipt, request_sha256)
        held = _open_bound_file_set(
            stage_fd, expected_names, "verifier publish stage"
        )
        _revalidate_bound_file_set(
            stage_fd,
            expected_names,
            held,
            "verifier publish stage before pathname check",
        )
        stage_dirent = _lstat_at(output_root_fd, stage_name)
        if stage_dirent is None or (stage_dirent.st_dev, stage_dirent.st_ino) != (
            stage_info.st_dev, stage_info.st_ino
        ):
            raise VerificationError("verifier stage pathname no longer names held FD")
        if _RENAME_EXCLUSIVE is None:
            _close_bound_file_set(held)
            held = None
            try:
                assert_lock()
                os.mkdir(output_name, 0o700, dir_fd=output_root_fd)
            except FileExistsError as error:
                raise VerificationError(
                    "verifier output leaf appeared during exclusive commit"
                ) from error
            os.fsync(output_root_fd)
            assert_lock()
            result = _materialize_final_receipt(
                output_root_fd,
                output_name,
                receipt,
                request_sha256,
                assert_lock,
                existing=False,
            )
            for filename in ("verifier_receipt.json", "COMMIT.json"):
                assert_lock()
                os.unlink(filename, dir_fd=stage_fd)
                assert_lock()
            assert_lock()
            os.fsync(stage_fd)
            assert_lock()
            os.rmdir(stage_name, dir_fd=output_root_fd)
            os.fsync(output_root_fd)
            assert_lock()
            return result
        _revalidate_bound_file_set(
            stage_fd,
            expected_names,
            held,
            "verifier publish stage immediately before rename",
        )
        assert_lock()
        _RENAME_EXCLUSIVE(output_root_fd, stage_name, output_name)
        os.fsync(output_root_fd)
        assert_lock()
        final_fd = os.open(
            output_name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=output_root_fd,
        )
        try:
            final_info = os.fstat(final_fd)
            if (final_info.st_dev, final_info.st_ino) != (stage_info.st_dev, stage_info.st_ino):
                raise VerificationError("atomically published verifier inode mismatch")
            _revalidate_bound_file_set(
                final_fd,
                expected_names,
                held,
                "published verifier held-file fence",
            )
            result = _validate_receipt_output_fd(final_fd, receipt, request_sha256)
            final_dirent = _lstat_at(output_root_fd, output_name)
            if final_dirent is None or (final_dirent.st_dev, final_dirent.st_ino) != (
                final_info.st_dev, final_info.st_ino
            ):
                raise VerificationError("published verifier pathname changed during validation")
            assert_lock()
            return result
        finally:
            os.close(final_fd)
    finally:
        if held is not None:
            _close_bound_file_set(held)
        os.close(stage_fd)


def _publish_receipt(
    output_root_fd: int,
    output_name: str,
    receipt: Mapping[str, Any],
    request_sha256: str,
) -> dict[str, Any]:
    _validate_dirfd(output_root_fd, "output root")
    lock_name = f".{output_name}.lock"
    lock_fd = os.open(
        lock_name,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=output_root_fd,
    )
    locked = False

    def assert_lock() -> None:
        _assert_named_lock_identity(output_root_fd, lock_name, lock_fd)

    try:
        info = os.fstat(lock_fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() \
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1:
            raise VerificationError("verifier lock file invalid")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise VerificationError("concurrent verifier publication") from error
        locked = True
        assert_lock()
        os.ftruncate(lock_fd, 0)
        os.lseek(lock_fd, 0, os.SEEK_SET)
        _write_all(lock_fd, canonical_bytes({
            "pid": os.getpid(),
            "request_sha256": request_sha256,
        }) + b"\n")
        os.fsync(lock_fd)
        os.fsync(output_root_fd)
        assert_lock()
        existing = _lstat_at(output_root_fd, output_name)
        if existing is not None:
            if not stat.S_ISDIR(existing.st_mode):
                raise VerificationError("verifier output leaf is not directory")
            assert_lock()
            result = _validate_receipt_output(
                output_root_fd, output_name, receipt, request_sha256
            )
            assert_lock()
            return result
        transaction = sha256_bytes(canonical_bytes({
            "request_sha256": request_sha256,
            "verifier_code_sha256": _MODULE_CODE_SHA256,
            "verifier_schema_sha256": _SCHEMA_SHA256,
        }))
        stage_name = f".{output_name}.{transaction[:16]}.stage"
        failed_name = f".{output_name}.{transaction[:16]}.failed"
        stage_info = _lstat_at(output_root_fd, stage_name)
        if stage_info is not None:
            if not stat.S_ISDIR(stage_info.st_mode):
                raise VerificationError("verifier staging leaf invalid")
            try:
                stage_fd, recovered_stage_stat = _complete_or_recover_receipt_stage(
                    output_root_fd,
                    stage_name,
                    receipt,
                    request_sha256,
                    assert_lock,
                )
                return _commit_completed_receipt_stage(
                    output_root_fd,
                    stage_fd,
                    recovered_stage_stat,
                    stage_name,
                    output_name,
                    receipt,
                    request_sha256,
                    assert_lock,
                )
            except LockIdentityError:
                raise
            except VerificationError as error:
                assert_lock()
                current_stage = _lstat_at(output_root_fd, stage_name)
                if current_stage is None or (current_stage.st_dev, current_stage.st_ino) != (
                    stage_info.st_dev, stage_info.st_ino
                ):
                    raise VerificationError("VERIFIER_STAGE_PATH_SUBSTITUTED") from error
                if _lstat_at(output_root_fd, failed_name) is not None:
                    raise VerificationError(
                        "incomplete verifier staging and failure evidence already exist"
                    ) from error
                assert_lock()
                if _RENAME_EXCLUSIVE is not None:
                    _RENAME_EXCLUSIVE(output_root_fd, stage_name, failed_name)
                else:
                    os.rename(
                        stage_name, failed_name,
                        src_dir_fd=output_root_fd, dst_dir_fd=output_root_fd,
                    )
                os.fsync(output_root_fd)
                assert_lock()
                raise VerificationError("FAILED_VISIBLE_PARTIAL_VERIFIER_OUTPUT") from error
        if _lstat_at(output_root_fd, failed_name) is not None:
            raise VerificationError("prior partial verifier output failure is preserved")
        assert_lock()
        os.mkdir(stage_name, 0o700, dir_fd=output_root_fd)
        os.fsync(output_root_fd)
        assert_lock()
        try:
            stage_fd, stage_stat = _complete_or_recover_receipt_stage(
                output_root_fd,
                stage_name,
                receipt,
                request_sha256,
                assert_lock,
            )
        except BaseException:
            # Preserve a private partial stage for deterministic verify-only
            # recovery. A mismatched byte is quarantined on the next call.
            os.fsync(output_root_fd)
            raise
        return _commit_completed_receipt_stage(
            output_root_fd,
            stage_fd,
            stage_stat,
            stage_name,
            output_name,
            receipt,
            request_sha256,
            assert_lock,
        )
    finally:
        lock_error: BaseException | None = None
        if locked:
            try:
                assert_lock()
            except BaseException as error:
                lock_error = error
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
        if lock_error is not None:
            raise lock_error


def verify_from_fds(
    request_bytes: bytes,
    *,
    input_root_fd: int,
    output_root_fd: int,
    code_fd: int | None = None,
    schema_fd: int | None = None,
    oracle_code_fd: int | None = None,
    oracle_contract_fd: int | None = None,
    oracle_schema_fd: int | None = None,
    reference_code_fd: int | None = None,
    reference_contract_fd: int | None = None,
    _test_reference_replay: Callable[[Mapping[str, bytes]], Any] | None = None,
    _test_reference_code_bytes: bytes | None = None,
    _test_reference_contract_bytes: bytes | None = None,
) -> dict[str, Any]:
    if _SEALED_RUNTIME:
        raise VerificationError("FD verifier adapter is forbidden in sealed runtime")
    input_root = _validate_dirfd(input_root_fd, "input root")
    output_root = _validate_dirfd(output_root_fd, "output root")
    if (input_root.st_dev, input_root.st_ino) == (output_root.st_dev, output_root.st_ino):
        raise VerificationError("input and output roots must be distinct directory inodes")
    if code_fd is None or _read_fd_snapshot(
        code_fd,
        "verifier code",
        allow_unlinked_sealed_runtime=True,
    ) != _MODULE_CODE_BYTES:
        raise VerificationError("launcher verifier code FD mismatch or missing")
    if schema_fd is None or _read_fd_snapshot(
        schema_fd,
        "verifier schema",
        allow_unlinked_sealed_runtime=True,
    ) != _SCHEMA_BYTES:
        raise VerificationError("launcher verifier schema FD mismatch or missing")
    if oracle_code_fd is None or oracle_contract_fd is None or oracle_schema_fd is None:
        raise VerificationError("trusted Oracle release FDs are required")
    test_reference_values = (
        _test_reference_replay,
        _test_reference_code_bytes,
        _test_reference_contract_bytes,
    )
    if not all(value is not None for value in test_reference_values) \
            or _EXECUTION_SNAPSHOT_MODE != "PATH_LOADED_TEST_ADAPTER_NOT_RELEASE_EVIDENCE" \
            or not callable(_test_reference_replay) \
            or type(_test_reference_code_bytes) is not bytes \
            or type(_test_reference_contract_bytes) is not bytes:
        raise VerificationError("path-loaded test reference injection is incomplete")
    reference_replay = _test_reference_replay
    reference_code_bytes = _test_reference_code_bytes
    reference_contract_bytes = _test_reference_contract_bytes
    if reference_code_fd is None or reference_contract_fd is None:
        raise VerificationError("trusted reference release FDs are required")
    reference_release = {
        "code_bytes": _read_fd_snapshot(
            reference_code_fd,
            "trusted reference code",
            allow_unlinked_sealed_runtime=True,
        ),
        "contract_bytes": _read_fd_snapshot(
            reference_contract_fd,
            "trusted reference contract",
            allow_unlinked_sealed_runtime=True,
        ),
    }
    if reference_release["code_bytes"] != reference_code_bytes \
            or reference_release["contract_bytes"] != reference_contract_bytes:
        raise VerificationError("trusted reference release FDs differ from injection")
    reference_release["hashes"] = {
        "code_sha256": sha256_bytes(reference_release["code_bytes"]),
        "contract_sha256": sha256_bytes(reference_release["contract_bytes"]),
    }
    if reference_release["hashes"] != SUPPORTED_REFERENCE_RELEASE:
        raise VerificationError("trusted reference release FD pair is not pinned")
    release = {
        "code_bytes": _read_fd_snapshot(
            oracle_code_fd,
            "trusted Oracle code",
            allow_unlinked_sealed_runtime=True,
        ),
        "contract_bytes": _read_fd_snapshot(
            oracle_contract_fd,
            "trusted Oracle contract",
            allow_unlinked_sealed_runtime=True,
        ),
        "schema_bytes": _read_fd_snapshot(
            oracle_schema_fd,
            "trusted Oracle schema",
            allow_unlinked_sealed_runtime=True,
        ),
    }
    release["hashes"] = {
        "code_sha256": sha256_bytes(release["code_bytes"]),
        "contract_sha256": sha256_bytes(release["contract_bytes"]),
        "schema_sha256": sha256_bytes(release["schema_bytes"]),
    }
    if release["hashes"] != SUPPORTED_ORACLE_RELEASE:
        raise VerificationError("trusted Oracle release FD triplet is not pinned")
    request = strict_json(request_bytes, "verifier request")
    state = _load_request(request, input_root_fd, release, reference_release)
    receipt = _verify_actual_evidence(state, reference_replay)
    published = _publish_receipt(
        output_root_fd,
        state["output_name"],
        receipt,
        sha256_bytes(request_bytes),
    )
    return {
        "output_directory": state["output_name"],
        "receipt_relative_path": f"{state['output_name']}/verifier_receipt.json",
        "receipt": published,
    }


def _open_trusted_directory(path: Path, label: str) -> int:
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise VerificationError(f"{label} must be non-symlink directory")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    after = os.fstat(descriptor)
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        os.close(descriptor)
        raise VerificationError(f"{label} changed while opening")
    _validate_dirfd(descriptor, label)
    return descriptor


def verify(
    request: Mapping[str, Any],
    *,
    trusted_input_root: Path,
    trusted_output_root: Path,
    reference_replay: Callable[[Mapping[str, bytes]], Any] | None = None,
    reference_code_bytes: bytes | None = None,
    reference_contract_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Non-release path/FD adapter retained only for adversarial unit tests."""
    if _SEALED_RUNTIME or _MODULE_PATH is None:
        raise VerificationError("path-loaded verifier adapter unavailable")
    request_bytes = canonical_bytes(dict(request)) + b"\n"
    opened_fds: list[int] = []
    try:
        input_fd = _open_trusted_directory(trusted_input_root, "input root")
        opened_fds.append(input_fd)
        output_fd = _open_trusted_directory(trusted_output_root, "output root")
        opened_fds.append(output_fd)
        code_fd = os.open(_MODULE_PATH, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        opened_fds.append(code_fd)
        schema_fd = os.open(
            _MODULE_PATH.parent / VERIFIER_SCHEMA_NAME,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        opened_fds.append(schema_fd)
        oracle_code_fd = os.open(
            _MODULE_PATH.parent / "paper_research_jpy_oracle_v2.py",
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        opened_fds.append(oracle_code_fd)
        oracle_contract_fd = os.open(
            _MODULE_PATH.parent / "PAPER_RESEARCH_JPY_ORACLE_CONTRACT_V2.json",
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        opened_fds.append(oracle_contract_fd)
        oracle_schema_fd = os.open(
            _MODULE_PATH.parent / "paper_research_jpy_oracle_schema_v2.json",
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        opened_fds.append(oracle_schema_fd)
        reference_code_parts = _relative_parts(
            request.get("reference_code_snapshot", {}).get("relative_path"),
            "test reference code snapshot",
        )
        reference_contract_parts = _relative_parts(
            request.get("reference_contract_snapshot", {}).get("relative_path"),
            "test reference contract snapshot",
        )
        reference_code_fd = os.open(
            trusted_input_root.joinpath(*reference_code_parts),
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        opened_fds.append(reference_code_fd)
        reference_contract_fd = os.open(
            trusted_input_root.joinpath(*reference_contract_parts),
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        opened_fds.append(reference_contract_fd)
        return verify_from_fds(
            request_bytes,
            input_root_fd=input_fd,
            output_root_fd=output_fd,
            code_fd=code_fd,
            schema_fd=schema_fd,
            oracle_code_fd=oracle_code_fd,
            oracle_contract_fd=oracle_contract_fd,
            oracle_schema_fd=oracle_schema_fd,
            reference_code_fd=reference_code_fd,
            reference_contract_fd=reference_contract_fd,
            _test_reference_replay=reference_replay,
            _test_reference_code_bytes=reference_code_bytes,
            _test_reference_contract_bytes=reference_contract_bytes,
        )
    finally:
        for descriptor in reversed(opened_fds):
            os.close(descriptor)


def _audit_hook(event: str, _: tuple[Any, ...]) -> None:
    if event.startswith(("socket.", "subprocess.")) or event in {
        "import",
        "os.system", "os.posix_spawn", "os.exec", "os.spawn",
    }:
        raise VerificationError(f"runtime capability denied: {event}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-fd", type=int, required=True)
    parser.add_argument("--input-root-fd", type=int, required=True)
    parser.add_argument("--output-root-fd", type=int, required=True)
    parser.add_argument("--code-fd", type=int, required=True)
    parser.add_argument("--schema-fd", type=int, required=True)
    parser.add_argument("--oracle-code-fd", type=int, required=True)
    parser.add_argument("--oracle-contract-fd", type=int, required=True)
    parser.add_argument("--oracle-schema-fd", type=int, required=True)
    parser.add_argument("--reference-code-fd", type=int, required=True)
    parser.add_argument("--reference-contract-fd", type=int, required=True)
    args = parser.parse_args()
    request_bytes = _read_fd_snapshot(args.request_fd, "verifier request")
    sys.addaudithook(_audit_hook)
    result = verify_from_fds(
        request_bytes,
        input_root_fd=args.input_root_fd,
        output_root_fd=args.output_root_fd,
        code_fd=args.code_fd,
        schema_fd=args.schema_fd,
        oracle_code_fd=args.oracle_code_fd,
        oracle_contract_fd=args.oracle_contract_fd,
        oracle_schema_fd=args.oracle_schema_fd,
        reference_code_fd=args.reference_code_fd,
        reference_contract_fd=args.reference_contract_fd,
    )
    print(json.dumps({
        "ok": True,
        "status": result["receipt"]["status"],
        "classification": result["receipt"]["classification"],
        "verifier_receipt_sha256": result["receipt"]["verifier_receipt_sha256"],
        "output_directory": result["output_directory"],
    }, sort_keys=True, separators=(",", ":")))
    return 0


if not _SEALED_RUNTIME and __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as error:
        print(json.dumps({
            "ok": False,
            "error_code": "VERIFIER_FAIL_CLOSED",
            "error_sha256": sha256_bytes(str(error).encode("utf-8")),
        }, sort_keys=True, separators=(",", ":")))
        raise SystemExit(2)
