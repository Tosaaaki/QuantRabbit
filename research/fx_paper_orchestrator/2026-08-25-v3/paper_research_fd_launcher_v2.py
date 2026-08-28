#!/usr/bin/env python3
"""Fixed-operation sealed-FD launcher for paper-research Oracle V2.

This process has exactly two operations: execute the pinned accounting Oracle,
or execute the pinned independent verifier.  It never accepts a program path,
module name, callable name, or child argv.  Runtime, contract, schema, request,
reference release, and root capabilities arrive only as inherited descriptors.
For verifier operation the launcher executes the pinned reference first, turns
its result into one canonical immutable byte snapshot, destroys the temporary
reference namespace, and gives the verifier only immutable values.  No live
reference callable, module, descriptor, path, or publication capability crosses
the verifier boundary.

On Darwin, immutable directory publication is delegated to
``renameatx_np(..., RENAME_EXCL)``.  Sealed execution has no non-native rename
fallback: an unavailable symbol or unsupported platform fails closed before
the pinned runtime is compiled.
"""

from __future__ import annotations

import argparse
import ast
import _strptime
import base64
import binascii
import builtins
import collections
import ctypes
import dataclasses
import datetime
import decimal
import dis
import errno
import fcntl
import fractions
import hashlib
import json
import os
import pathlib
import re
import stat
import struct
import sys
import types
import typing
from types import CodeType
from typing import Any, Callable, Mapping


LAUNCHER_NAME = "PAPER_RESEARCH_FD_LAUNCHER_V2"
OPERATIONS = ("ORACLE", "VERIFIER")
RENAME_EXCL = 0x00000004
# 64 MiB is a capability ceiling for code/contracts/schemas.  It is fixed to
# prevent an inherited descriptor from turning launcher snapshotting into an
# unbounded allocation; changing it requires a launcher contract revision.
MAX_RUNTIME_ARTIFACT_BYTES = 64 * 1024 * 1024
# 128 MiB bounds the interpreter image snapshot used only for attestation.  It
# is intentionally larger than the current executable while remaining finite;
# a larger interpreter requires a reviewed launcher revision.
MAX_INTERPRETER_ARTIFACT_BYTES = 128 * 1024 * 1024
# 32 MiB matches the pinned Oracle/Verifier JSON request ceiling.  Keeping the
# launcher at the same bound prevents it from accepting an envelope the target
# must reject later.
MAX_REQUEST_BYTES = 32 * 1024 * 1024
MAX_JSON_ARTIFACT_BYTES = 32 * 1024 * 1024
MAX_SOURCE_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024
ZERO_SHA256 = "0" * 64
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
VERIFIER_ARTIFACT_ROLES = (
    "accounting_policy",
    "authority_policy",
    "evaluation_policy",
    "execution_policy",
    "instrument_registry",
    "inventory_policy",
    "oracle_code_snapshot",
    "oracle_commit",
    "oracle_contract_snapshot",
    "oracle_intent",
    "oracle_ledger",
    "oracle_manifest",
    "oracle_request",
    "oracle_schema_snapshot",
    "proposal",
    "reference_code_snapshot",
    "reference_contract_snapshot",
    "source_blob",
    "source_manifest",
)
REFERENCE_RAW_ARTIFACT_ROLES = (
    "accounting_policy",
    "authority_policy",
    "evaluation_policy",
    "execution_policy",
    "instrument_registry",
    "inventory_policy",
    "proposal",
    "source_blob",
    "source_manifest",
)
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
PURE_VERIFIER_BUILTINS = frozenset({
    "abs",
    "all",
    "any",
    "BaseException",
    "bool",
    "bytes",
    "dict",
    "enumerate",
    "Exception",
    "frozenset",
    "IndexError",
    "int",
    "isinstance",
    "KeyError",
    "len",
    "list",
    "max",
    "min",
    "OverflowError",
    "range",
    "RuntimeError",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "TypeError",
    "UnicodeDecodeError",
    "UnicodeEncodeError",
    "ValueError",
    "zip",
    "type",
})
VERIFIER_INITIALIZATION_BUILTINS = frozenset({"__build_class__", "__import__"})
PURE_REFERENCE_BUILTINS = frozenset({
    "KeyError",
    "TypeError",
    "UnicodeDecodeError",
    "ValueError",
    "abs",
    "all",
    "any",
    "bool",
    "bytes",
    "dict",
    "enumerate",
    "frozenset",
    "getattr",
    "int",
    "isinstance",
    "len",
    "list",
    "max",
    "min",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "type",
    "zip",
})
REFERENCE_INITIALIZATION_BUILTINS = frozenset({
    "RuntimeError",
    "__build_class__",
    "__import__",
})

# These are release allowlists, not values supplied by a request.  Update a pin
# only after the corresponding byte set is frozen and its adversarial tests
# pass.  Keeping all pins in this single literal makes final freeze review a
# bounded mechanical diff.
PINNED_RELEASES: dict[str, dict[str, str]] = {
    "ORACLE": {
        "code_sha256": "3c7a059576714e67cbf92e5689c2e88e1f7c600c62dde686a6fa63fb2b7a82c5",
        "contract_sha256": "abbdd484354f86a48c8c001fc8521f0cd96ab8aec6b46023f15bcb4502cf4467",
        "schema_sha256": "641c1b8ee69827e078fbcb49cb8f1bc54d59d03fa33f56b42550b8587f99e841",
    },
    "VERIFIER": {
        "code_sha256": "7a79e9ba4c0d93aab1f09221c86335d43101c4df33a2119ece4f03b43a7c5439",
        "schema_sha256": "ae1aa91189df6b03fd114931a80e50351c56f284360766ee7831672fff180868",
    },
    "REFERENCE": {
        "code_sha256": "cbac8e308bc11cd334f1cd23d23e4e75019074c1bdcfb66873b9254e3d6d520f",
        "contract_sha256": "276c34f4174a15d188406ef870d86a8d0bcbbc1b64b1f45381a033e20eb5d8f5",
    },
}

# The outer bootstrap is deliberately small, fixed, and stdlib-only.  A caller
# supplies the expected launcher SHA on argv; this bootstrap checks that hash
# against an inherited read-only FD *before* compiling any launcher byte.  A
# Python ``-c`` process cannot prove the bytes that already started it, so this
# is a host-local consistency receipt, not an external trust anchor.  Release
# eligibility remains false until an independent reviewer pins the builder,
# bootstrap, launcher, argv, and descriptor map outside this process.
FIXED_BOOTSTRAP_SOURCE = r'''import ctypes,errno,fcntl,hashlib,json,os,stat,sys

_MAX_LAUNCHER_BYTES=64*1024*1024
_MAX_INTERPRETER_BYTES=128*1024*1024
_ACL_TYPE_EXTENDED=0x00000100

if sys.platform!="darwin":
    raise RuntimeError("bootstrap extended ACL inspection unavailable")
_acl_lib=ctypes.CDLL("/usr/lib/libSystem.B.dylib",use_errno=True)
_acl_get=_acl_lib.acl_get_fd_np
_acl_get.argtypes=(ctypes.c_int,ctypes.c_int)
_acl_get.restype=ctypes.c_void_p
_acl_free=_acl_lib.acl_free
_acl_free.argtypes=(ctypes.c_void_p,)
_acl_free.restype=ctypes.c_int

def _no_acl(fd,label):
    ctypes.set_errno(0)
    pointer=_acl_get(fd,_ACL_TYPE_EXTENDED)
    saved=ctypes.get_errno()
    if pointer:
        ctypes.set_errno(0)
        result=_acl_free(pointer)
        free_errno=ctypes.get_errno()
        if result!=0:
            raise RuntimeError(label+" ACL release failed errno "+str(free_errno))
        raise RuntimeError(label+" has extended ACL")
    if saved!=errno.ENOENT:
        raise RuntimeError(label+" ACL inspection failed errno "+str(saved))

def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")

def _write_all(fd,value):
    offset=0
    while offset<len(value):
        written=os.write(fd,value[offset:])
        if written<=0:
            raise RuntimeError("bootstrap stdout short write")
        offset+=written

def _fail(error):
    payload={"ok":False,"error_code":"SEALED_FD_BOOTSTRAP_FAIL_CLOSED","error_sha256":hashlib.sha256(str(error).encode("utf-8")).hexdigest()}
    _write_all(1,_canonical(payload)+b"\n")
    raise SystemExit(120)

def _snapshot(fd,label,limit):
    if type(fd) is not int or fd<0:
        raise RuntimeError(label+" descriptor invalid")
    if fcntl.fcntl(fd,fcntl.F_GETFL)&os.O_ACCMODE!=os.O_RDONLY:
        raise RuntimeError(label+" descriptor is not read-only")
    before=os.fstat(fd)
    if not stat.S_ISREG(before.st_mode) or before.st_size<0 or before.st_size>limit:
        raise RuntimeError(label+" descriptor is not bounded regular data")
    _no_acl(fd,label)
    parts=[]
    offset=0
    while offset<before.st_size:
        chunk=os.pread(fd,min(1024*1024,before.st_size-offset),offset)
        if not chunk:
            raise RuntimeError(label+" truncated during snapshot")
        parts.append(chunk)
        offset+=len(chunk)
    if os.pread(fd,1,before.st_size):
        raise RuntimeError(label+" grew during snapshot")
    after=os.fstat(fd)
    _no_acl(fd,label+" final fence")
    identity=lambda item:(item.st_dev,item.st_ino,item.st_size,item.st_mtime_ns,item.st_ctime_ns)
    if identity(before)!=identity(after):
        raise RuntimeError(label+" changed during snapshot")
    return b"".join(parts)

def _flags():
    return {"isolated":int(sys.flags.isolated),"no_site":int(sys.flags.no_site),"ignore_environment":int(sys.flags.ignore_environment),"safe_path":int(sys.flags.safe_path),"no_user_site":int(sys.flags.no_user_site),"dont_write_bytecode":int(sys.flags.dont_write_bytecode)}

def _minimal_sys_path():
    paths=list(sys.path)
    stdlib_parent=os.path.realpath(os.path.join(sys.base_prefix,"lib"))
    cwd=os.path.realpath(os.getcwd())
    if not 1<=len(paths)<=4:
        raise RuntimeError("bootstrap sys.path entry count is not minimal")
    for entry in paths:
        if type(entry) is not str or not entry:
            raise RuntimeError("bootstrap sys.path has an empty or non-text entry")
        lowered=entry.lower()
        real=os.path.realpath(entry)
        if "site-packages" in lowered or "dist-packages" in lowered or real==cwd:
            raise RuntimeError("bootstrap sys.path contains a caller/site entry")
        if real!=stdlib_parent and not real.startswith(stdlib_parent+os.sep):
            raise RuntimeError("bootstrap sys.path escapes the interpreter stdlib")
    return paths

try:
    if len(sys.argv)<7 or sys.argv[1]!="--launcher-fd" or sys.argv[3]!="--expected-launcher-sha256" or sys.argv[5]!="--bootstrap-source-sha256":
        raise RuntimeError("fixed bootstrap arguments malformed")
    launcher_fd=int(sys.argv[2])
    expected_launcher_sha=sys.argv[4]
    bootstrap_source_sha=sys.argv[6]
    if len(expected_launcher_sha)!=64 or any(ch not in "0123456789abcdef" for ch in expected_launcher_sha):
        raise RuntimeError("expected launcher SHA malformed")
    if len(bootstrap_source_sha)!=64 or any(ch not in "0123456789abcdef" for ch in bootstrap_source_sha):
        raise RuntimeError("bootstrap source SHA malformed")
    required_flags={"isolated":1,"no_site":1,"ignore_environment":1,"safe_path":1,"no_user_site":1,"dont_write_bytecode":1}
    actual_flags=_flags()
    if actual_flags!=required_flags:
        raise RuntimeError("required -I -S -B interpreter flags are absent")
    actual_path=_minimal_sys_path()
    launcher_bytes=_snapshot(launcher_fd,"launcher",_MAX_LAUNCHER_BYTES)
    launcher_sha=hashlib.sha256(launcher_bytes).hexdigest()
    if launcher_sha!=expected_launcher_sha:
        raise RuntimeError("launcher FD differs from caller-fixed expected SHA")
    executable_path=os.path.realpath(sys.executable)
    executable_fd=os.open(executable_path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0))
    try:
        executable_bytes=_snapshot(executable_fd,"interpreter executable",_MAX_INTERPRETER_BYTES)
    finally:
        os.close(executable_fd)
    interpreter_identity={"implementation":sys.implementation.name,"cache_tag":sys.implementation.cache_tag,"version":sys.version,"hexversion":sys.hexversion,"executable_realpath":executable_path}
    attestation={
        "schema_version":1,
        "bootstrap_id":"PAPER_RESEARCH_FIXED_FD_BOOTSTRAP_V1",
        "caller_asserted_bootstrap_source_sha256":bootstrap_source_sha,
        "bootstrap_provenance":"PYTHON_C_NOT_SELF_AUTHENTICATING",
        "pre_audit_capability_absence_proven":False,
        "outer_launch_provenance_status":"CALLER_ASSERTED_PENDING_EXTERNAL_ANCHOR",
        "launcher_code_sha256":launcher_sha,
        "interpreter_executable_sha256":hashlib.sha256(executable_bytes).hexdigest(),
        "interpreter_identity":interpreter_identity,
        "interpreter_identity_sha256":hashlib.sha256(_canonical(interpreter_identity)).hexdigest(),
        "interpreter_flags":actual_flags,
        "interpreter_flags_sha256":hashlib.sha256(_canonical(actual_flags)).hexdigest(),
        "sys_path":actual_path,
        "sys_path_sha256":hashlib.sha256(_canonical(actual_path)).hexdigest(),
    }
    attestation["bootstrap_attestation_sha256"]=hashlib.sha256(_canonical(attestation)).hexdigest()
    attestation_bytes=_canonical(attestation)+b"\n"
    remaining=sys.argv[7:]
    sys.argv=["<sealed-fd-launcher-v2>","--launcher-fd",str(launcher_fd),*remaining]
    namespace={"__name__":"__main__","__file__":"<sealed-fd-launcher-v2>","__package__":None,"__spec__":None,"_SEALED_LAUNCHER_CODE_BYTES":launcher_bytes,"_SEALED_BOOTSTRAP_ATTESTATION":attestation_bytes}
    code=compile(launcher_bytes,"<sealed-fd-launcher-v2>","exec",dont_inherit=True,optimize=0)
    exec(code,namespace,namespace)
except SystemExit:
    raise
except BaseException as error:
    _fail(error)
'''
PINNED_BOOTSTRAP_SOURCE_SHA256 = hashlib.sha256(
    FIXED_BOOTSTRAP_SOURCE.encode("utf-8")
).hexdigest()

_INJECTED_LAUNCHER_CODE_BYTES = globals().get("_SEALED_LAUNCHER_CODE_BYTES")
_INJECTED_BOOTSTRAP_ATTESTATION = globals().get("_SEALED_BOOTSTRAP_ATTESTATION")


class LauncherError(RuntimeError):
    """Fail-closed sealed launcher violation."""


# Darwin <sys/acl.h> defines ACL_TYPE_EXTENDED as 0x00000100.
LAUNCHER_ACL_TYPE_EXTENDED = 0x00000100


def _bind_launcher_extended_acl_api(
    platform_name: str | None = None,
    library: Any | None = None,
) -> tuple[Any, Any, Any]:
    platform_name = sys.platform if platform_name is None else platform_name
    if platform_name != "darwin":
        raise LauncherError("extended ACL inspection is unavailable on this host")
    try:
        if library is None:
            library = ctypes.CDLL(
                "/usr/lib/libSystem.B.dylib",
                use_errno=True,
            )
        get_acl = library.acl_get_fd_np
        free_acl = library.acl_free
    except (OSError, AttributeError) as error:
        raise LauncherError("extended ACL inspection API is unavailable") from error
    get_acl.argtypes = (ctypes.c_int, ctypes.c_int)
    get_acl.restype = ctypes.c_void_p
    free_acl.argtypes = (ctypes.c_void_p,)
    free_acl.restype = ctypes.c_int
    return library, get_acl, free_acl


(
    _LAUNCHER_ACL_LIBRARY,
    _LAUNCHER_ACL_GET_FD_NP,
    _LAUNCHER_ACL_FREE,
) = _bind_launcher_extended_acl_api()


def _require_launcher_no_extended_acl_fd(
    descriptor: int,
    label: str,
) -> None:
    ctypes.set_errno(0)
    acl_pointer = _LAUNCHER_ACL_GET_FD_NP(
        descriptor, LAUNCHER_ACL_TYPE_EXTENDED
    )
    saved_errno = ctypes.get_errno()
    if acl_pointer:
        ctypes.set_errno(0)
        free_result = _LAUNCHER_ACL_FREE(acl_pointer)
        free_errno = ctypes.get_errno()
        if free_result != 0:
            raise LauncherError(
                f"{label} ACL release failed with errno {free_errno}"
            )
        raise LauncherError(f"{label} has an extended ACL")
    if saved_errno != errno.ENOENT:
        raise LauncherError(
            f"{label} ACL inspection failed with errno {saved_errno}"
        )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _assert_canonical_value(value: Any, label: str) -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _assert_canonical_value(item, f"{label}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise LauncherError(f"{label} has a non-text key")
            _assert_canonical_value(item, f"{label}.{key}")
        return
    raise LauncherError(f"{label} has a non-canonical value")


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LauncherError("duplicate JSON key is forbidden")
        result[key] = value
    return result


def _json_int(token: str) -> int:
    if token == "-0":
        raise LauncherError("negative-zero JSON integer is forbidden")
    return int(token)


def _json_float(_: str) -> Any:
    raise LauncherError("floating-point JSON number is forbidden")


def _strict_json_object(data: bytes, label: str) -> dict[str, Any]:
    if type(data) is not bytes or len(data) > MAX_JSON_ARTIFACT_BYTES \
            or data.startswith(b"\xef\xbb\xbf") \
            or not data.endswith(b"\n") or data.endswith(b"\n\n"):
        raise LauncherError(f"{label} canonical JSON bytes are invalid")
    body = data[:-1]
    try:
        value = json.loads(
            body.decode("utf-8", errors="strict"),
            object_pairs_hook=_json_pairs,
            parse_int=_json_int,
            parse_float=_json_float,
            parse_constant=_json_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LauncherError(f"{label} JSON is invalid") from error
    if type(value) is not dict:
        raise LauncherError(f"{label} must be one JSON object")
    _assert_canonical_value(value, label)
    if _canonical_bytes(value) != body:
        raise LauncherError(f"{label} is not canonical JSON")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str] | set[str],
    label: str,
) -> None:
    if set(value) != set(expected):
        raise LauncherError(f"{label} schema is not exact")


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise LauncherError(f"{label} must be SHA-256")
    return value


def _write_all(descriptor: int, value: bytes) -> None:
    offset = 0
    while offset < len(value):
        try:
            written = os.write(descriptor, value[offset:])
        except OSError as error:
            raise LauncherError("launcher output write failed") from error
        if written <= 0:
            raise LauncherError("launcher output short write")
        offset += written


def _fd_access_mode(descriptor: int) -> int:
    try:
        return fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE
    except OSError as error:
        raise LauncherError("inherited descriptor is not open") from error


def _snapshot_readonly_regular_fd(
    descriptor: int,
    label: str,
    *,
    maximum_bytes: int,
) -> bytes:
    if type(descriptor) is not int or descriptor < 0:
        raise LauncherError(f"{label} descriptor is invalid")
    if _fd_access_mode(descriptor) != os.O_RDONLY:
        raise LauncherError(f"{label} descriptor must be read-only")
    try:
        before = os.fstat(descriptor)
    except OSError as error:
        raise LauncherError(f"{label} descriptor cannot be inspected") from error
    if not stat.S_ISREG(before.st_mode):
        raise LauncherError(f"{label} descriptor must reference a regular file")
    _require_launcher_no_extended_acl_fd(descriptor, label)
    if before.st_size < 0 or before.st_size > maximum_bytes:
        raise LauncherError(f"{label} exceeds the sealed byte ceiling")
    chunks: list[bytes] = []
    offset = 0
    while offset < before.st_size:
        try:
            chunk = os.pread(
                descriptor,
                min(1024 * 1024, before.st_size - offset),
                offset,
            )
        except OSError as error:
            raise LauncherError(f"{label} snapshot read failed") from error
        if not chunk:
            raise LauncherError(f"{label} changed or truncated during snapshot")
        chunks.append(chunk)
        offset += len(chunk)
    # A read past the snapshotted size detects append races without depending
    # only on timestamp granularity.
    if os.pread(descriptor, 1, before.st_size):
        raise LauncherError(f"{label} grew during snapshot")
    after = os.fstat(descriptor)
    _require_launcher_no_extended_acl_fd(descriptor, f"{label} final fence")
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after):
        raise LauncherError(f"{label} changed during snapshot")
    return b"".join(chunks)


def _validate_root_fd(descriptor: int, label: str) -> os.stat_result:
    if type(descriptor) is not int or descriptor < 0:
        raise LauncherError(f"{label} descriptor is invalid")
    if _fd_access_mode(descriptor) != os.O_RDONLY:
        raise LauncherError(f"{label} descriptor must be read-only")
    try:
        info = os.fstat(descriptor)
    except OSError as error:
        raise LauncherError(f"{label} descriptor cannot be inspected") from error
    if not stat.S_ISDIR(info.st_mode):
        raise LauncherError(f"{label} descriptor must reference a directory")
    if info.st_uid != os.geteuid() or info.st_mode & 0o022:
        raise LauncherError(f"{label} ownership or mode is unsafe")
    _require_launcher_no_extended_acl_fd(descriptor, label)
    return info


def _relative_parts(value: Any, label: str) -> tuple[str, ...]:
    if type(value) is not str or not value or len(value) > 512 \
            or value.startswith("/") or value.endswith("/") or "//" in value:
        raise LauncherError(f"{label} relative path is invalid")
    parts = tuple(value.split("/"))
    if any(
        part in {"", ".", ".."} or SAFE_COMPONENT_RE.fullmatch(part) is None
        for part in parts
    ):
        raise LauncherError(f"{label} relative path is unsafe")
    return parts


def _snapshot_relative_artifact(
    root_fd: int,
    relative_path: Any,
    label: str,
    *,
    expected_size: int,
    maximum_bytes: int,
) -> tuple[bytes, tuple[int, int]]:
    _validate_root_fd(root_fd, "input root")
    parts = _relative_parts(relative_path, label)
    current = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            _validate_root_fd(current, f"{label} parent")
            try:
                next_fd = os.open(
                    part,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=current,
                )
            except OSError as error:
                raise LauncherError(f"{label} parent cannot be opened safely") from error
            info = os.fstat(next_fd)
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() \
                    or info.st_mode & 0o022:
                os.close(next_fd)
                raise LauncherError(f"{label} parent directory is unsafe")
            _require_launcher_no_extended_acl_fd(
                next_fd, f"{label} parent directory"
            )
            os.close(current)
            current = next_fd
        try:
            descriptor = os.open(
                parts[-1],
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=current,
            )
        except OSError as error:
            raise LauncherError(f"{label} cannot be opened safely") from error
        try:
            before = os.fstat(descriptor)
            if _fd_access_mode(descriptor) != os.O_RDONLY \
                    or not stat.S_ISREG(before.st_mode) \
                    or before.st_uid != os.geteuid() \
                    or before.st_mode & 0o022 \
                    or before.st_nlink != 1:
                raise LauncherError(f"{label} is not an owned immutable artifact")
            _require_launcher_no_extended_acl_fd(descriptor, label)
            if type(expected_size) is not int or expected_size < 0 \
                    or expected_size > maximum_bytes \
                    or before.st_size != expected_size:
                raise LauncherError(f"{label} declared size differs")
            chunks: list[bytes] = []
            offset = 0
            while offset < before.st_size:
                try:
                    chunk = os.pread(
                        descriptor,
                        min(1024 * 1024, before.st_size - offset),
                        offset,
                    )
                except OSError as error:
                    raise LauncherError(f"{label} snapshot read failed") from error
                if not chunk:
                    raise LauncherError(f"{label} truncated during snapshot")
                chunks.append(chunk)
                offset += len(chunk)
            if os.pread(descriptor, 1, before.st_size):
                raise LauncherError(f"{label} grew during snapshot")
            after = os.fstat(descriptor)
            _require_launcher_no_extended_acl_fd(
                descriptor, f"{label} final fence"
            )
            identity = lambda item: (
                item.st_dev,
                item.st_ino,
                item.st_size,
                item.st_mtime_ns,
                item.st_ctime_ns,
                item.st_nlink,
            )
            if identity(before) != identity(after):
                raise LauncherError(f"{label} changed during snapshot")
            return b"".join(chunks), (before.st_dev, before.st_ino)
        finally:
            os.close(descriptor)
    finally:
        os.close(current)


def _snapshot_verifier_request_artifacts(
    request_bytes: bytes,
    input_root_fd: int,
    *,
    oracle_release_blobs: Mapping[str, bytes],
    reference_code_bytes: bytes,
    reference_contract_bytes: bytes,
) -> tuple[dict[str, Any], tuple[tuple[str, bytes], ...], Mapping[str, bytes]]:
    request = _strict_json_object(request_bytes, "verifier request")
    expected_keys = frozenset({
        "schema_version",
        "output_directory",
        *VERIFIER_ARTIFACT_ROLES,
    })
    _require_exact_keys(request, expected_keys, "verifier request")
    if type(request["schema_version"]) is not int or request["schema_version"] != 2:
        raise LauncherError("verifier request schema version mismatch")
    output_name = request["output_directory"]
    if type(output_name) is not str or SAFE_COMPONENT_RE.fullmatch(output_name) is None:
        raise LauncherError("verifier output directory is unsafe")
    blobs: dict[str, bytes] = {}
    relative_paths: set[str] = set()
    identities: set[tuple[int, int]] = set()
    for role in VERIFIER_ARTIFACT_ROLES:
        descriptor = request[role]
        if type(descriptor) is not dict:
            raise LauncherError(f"{role} descriptor must be an object")
        _require_exact_keys(
            descriptor,
            frozenset({"artifact_id", "relative_path", "sha256", "size_bytes"}),
            f"{role} descriptor",
        )
        if descriptor["artifact_id"] != role:
            raise LauncherError(f"{role} descriptor identity mismatch")
        expected_hash = _require_sha256(descriptor["sha256"], f"{role}.sha256")
        size = descriptor["size_bytes"]
        maximum = (
            MAX_SOURCE_ARTIFACT_BYTES
            if role in {"source_blob", "oracle_ledger"}
            else MAX_JSON_ARTIFACT_BYTES
        )
        data, identity = _snapshot_relative_artifact(
            input_root_fd,
            descriptor["relative_path"],
            role,
            expected_size=size,
            maximum_bytes=maximum,
        )
        if len(data) != size or _sha256(data) != expected_hash:
            raise LauncherError(f"{role} exact-byte binding mismatch")
        if descriptor["relative_path"] in relative_paths or identity in identities:
            raise LauncherError("verifier artifact descriptors alias each other")
        relative_paths.add(descriptor["relative_path"])
        identities.add(identity)
        blobs[role] = data
    frozen_release_bindings = {
        "oracle_code_snapshot": oracle_release_blobs["code_bytes"],
        "oracle_contract_snapshot": oracle_release_blobs["contract_bytes"],
        "oracle_schema_snapshot": oracle_release_blobs["schema_bytes"],
        "reference_code_snapshot": reference_code_bytes,
        "reference_contract_snapshot": reference_contract_bytes,
    }
    for role, expected in frozen_release_bindings.items():
        if blobs[role] != expected:
            raise LauncherError(f"{role} differs from its inherited release FD")
    artifact_tuple = tuple(sorted(blobs.items()))
    raw_artifacts = types.MappingProxyType({
        role: blobs[role]
        for role in REFERENCE_RAW_ARTIFACT_ROLES
    })
    return request, artifact_tuple, raw_artifacts


def _require_pinned_bytes(operation: str, role: str, value: bytes) -> None:
    expected = PINNED_RELEASES[operation].get(f"{role}_sha256")
    if expected is None or _sha256(value) != expected:
        raise LauncherError(f"{operation.lower()} {role} FD is not the pinned release")


def _safe_leaf(value: str, label: str) -> bytes:
    if type(value) is not str or not value or value in {".", ".."}:
        raise LauncherError(f"{label} is not a safe leaf")
    if "/" in value or "\x00" in value or len(value.encode("utf-8")) > 255:
        raise LauncherError(f"{label} is not a safe leaf")
    return os.fsencode(value)


def _build_native_rename_exclusive(
    *,
    platform_name: str | None = None,
    library: Any | None = None,
) -> Callable[[int, str, str], None]:
    """Return the narrowly bounded Darwin ``RENAME_EXCL`` primitive.

    The optional arguments exist only to make unsupported-platform and missing
    symbol behavior directly testable.  Production calls this function with no
    arguments and never selects a different symbol or flag.
    """

    effective_platform = sys.platform if platform_name is None else platform_name
    if effective_platform != "darwin":
        raise LauncherError("sealed exclusive rename requires Darwin renameatx_np")
    try:
        libc = ctypes.CDLL(None, use_errno=True) if library is None else library
        renameatx_np = libc.renameatx_np
    except (AttributeError, OSError) as error:
        raise LauncherError("Darwin renameatx_np is unavailable") from error
    renameatx_np.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameatx_np.restype = ctypes.c_int

    def rename_exclusive(root_fd: int, source_leaf: str, destination_leaf: str) -> None:
        _validate_root_fd(root_fd, "rename root")
        source = _safe_leaf(source_leaf, "rename source")
        destination = _safe_leaf(destination_leaf, "rename destination")
        ctypes.set_errno(0)
        result = renameatx_np(
            root_fd,
            source,
            root_fd,
            destination,
            RENAME_EXCL,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(error_number, "exclusive destination exists", destination_leaf)
        raise OSError(error_number, os.strerror(error_number), destination_leaf)

    return rename_exclusive


class _AuditBoundary:
    """Defense in depth for the already hash-pinned runtime bytes.

    This hook is not a hermetic sandbox: the pinned Oracle and verifier still
    own narrowly reviewed file operations.  It prevents them from compiling a
    second program or opening process/network/native-code capabilities.
    """

    def __init__(self) -> None:
        self._allowed_compile: tuple[bytes, str] | None = None
        self._allowed_code: CodeType | None = None
        self._allowed_imports: frozenset[str] | None = None
        self._dataclass_codegen_armed = False
        self._reference_phase_active = False
        self._runtime_phase_active = False

    def begin_reference_phase(self) -> None:
        if self._reference_phase_active \
                or self._runtime_phase_active \
                or self._allowed_compile is not None \
                or self._allowed_code is not None \
                or self._allowed_imports is not None \
                or self._dataclass_codegen_armed:
            raise LauncherError("reference audit phase cannot be nested or pre-armed")
        self._reference_phase_active = True

    def begin_runtime_phase(self) -> None:
        if self._reference_phase_active \
                or self._runtime_phase_active \
                or self._allowed_compile is not None \
                or self._allowed_code is not None \
                or self._allowed_imports is not None \
                or self._dataclass_codegen_armed:
            raise LauncherError("runtime audit phase cannot be nested or pre-armed")
        self._runtime_phase_active = True

    def end_reference_phase(self) -> None:
        self._allowed_compile = None
        self._allowed_code = None
        self._allowed_imports = None
        self._dataclass_codegen_armed = False
        self._reference_phase_active = False

    def end_runtime_phase(self) -> None:
        self._allowed_compile = None
        self._allowed_code = None
        self._allowed_imports = None
        self._dataclass_codegen_armed = False
        self._runtime_phase_active = False

    def allow_exact_compile(self, source: bytes, filename: str) -> None:
        if self._allowed_compile is not None:
            raise LauncherError("sealed compile token already armed")
        self._allowed_compile = (source, filename)

    def allow_exact_exec(self, code: CodeType) -> None:
        if self._allowed_code is not None:
            raise LauncherError("sealed exec token already armed")
        self._allowed_code = code

    def allow_exact_imports(self, names: frozenset[str]) -> None:
        if self._allowed_imports is not None:
            raise LauncherError("sealed import phase already armed")
        self._allowed_imports = names

    def allow_dataclass_codegen(self) -> None:
        if self._dataclass_codegen_armed:
            raise LauncherError("dataclass code generation phase already armed")
        self._dataclass_codegen_armed = True

    def seal_dataclass_codegen(self) -> None:
        self._dataclass_codegen_armed = False

    def _called_from_dataclass_codegen(self) -> bool:
        if not self._dataclass_codegen_armed:
            return False
        frame = sys._getframe(1)
        expected_code = dataclasses._create_fn.__code__
        while frame is not None:
            if frame.f_code is expected_code:
                return True
            frame = frame.f_back
        return False

    def sealed_import(
        self,
        name: str,
        globals_: Mapping[str, Any] | None = None,
        locals_: Mapping[str, Any] | None = None,
        fromlist: tuple[str, ...] | list[str] = (),
        level: int = 0,
    ) -> Any:
        allowed = self._allowed_imports
        if allowed is None:
            raise LauncherError("dynamic import capability denied")
        if type(name) is not str or name not in allowed or type(level) is not int or level != 0:
            raise LauncherError("runtime import is outside the frozen dependency set")
        if name not in sys.modules:
            raise LauncherError("runtime dependency was not initialized before the audit boundary")
        return builtins.__import__(name, globals_, locals_, fromlist, level)

    def seal_imports(self) -> None:
        self._allowed_imports = None

    def hook(self, event: str, args: tuple[Any, ...]) -> None:
        if not self._reference_phase_active and not self._runtime_phase_active:
            return
        if self._runtime_phase_active \
                and event in {"ctypes.set_errno", "ctypes.get_errno"}:
            # The already-created Darwin renameatx_np closure needs only errno
            # bookkeeping.  Loading or resolving any new native symbol remains
            # denied below.
            return
        if event == "compile":
            candidate = (
                args[0] if len(args) > 0 else None,
                args[1] if len(args) > 1 else None,
            )
            if candidate == self._allowed_compile:
                self._allowed_compile = None
                return
            if candidate[1] == "<string>" \
                    and self._called_from_dataclass_codegen():
                return
            raise LauncherError("dynamic compile capability denied")
        if event == "exec":
            code = args[0] if args else None
            if code is self._allowed_code:
                self._allowed_code = None
                return
            if isinstance(code, CodeType) and code.co_filename == "<string>" \
                    and self._called_from_dataclass_codegen():
                return
            raise LauncherError("dynamic exec capability denied")
        if event == "import":
            name = args[0] if args else None
            allowed = self._allowed_imports
            if type(name) is str and allowed is not None \
                    and name in allowed and name in sys.modules:
                return
            raise LauncherError("dynamic import capability denied")
        if self._reference_phase_active and (
            event == "open"
            or event.startswith((
                "os.",
                "pathlib.",
                "time.",
                "datetime.",
                "random.",
                "secrets.",
                "uuid.",
            ))
            or event in {"sys.settrace", "sys.setprofile"}
        ):
            raise LauncherError(f"reference replay capability denied: {event}")
        if event.startswith((
            "socket.",
            "subprocess.",
            "ssl.",
            "http.client.",
            "urllib.",
            "ftplib.",
            "smtplib.",
            "ctypes.",
        )) or event in {
            "code.__new__",
            "function.__new__",
            "marshal.loads",
            "os.system",
            "os.posix_spawn",
            "os.posix_spawnp",
            "os.exec",
            "os.spawn",
            "os.fork",
            "os.forkpty",
        }:
            raise LauncherError(f"runtime capability denied: {event}")


_REFERENCE_IMPORTS = (
    ("from", "__future__", (("annotations", None),), 0),
    ("from", "collections", (("Counter", None), ("defaultdict", None)), 0),
    ("from", "dataclasses", (("dataclass", None),), 0),
    ("from", "fractions", (("Fraction", None),), 0),
    ("import", (("hashlib", None),)),
    ("import", (("json", None),)),
    ("import", (("re", None),)),
    ("from", "types", (("MappingProxyType", None),), 0),
    (
        "from",
        "typing",
        (("Any", None), ("Iterable", None), ("Mapping", None), ("Sequence", None)),
        0,
    ),
)
_REFERENCE_DATACLASS_FIELDS = {
    "MarketTick": (
        "provider_id", "instrument", "bid_ticks", "ask_ticks", "tick_scale",
        "source_ts_ns", "arrival_ts_ns", "sequence", "source_event_sha256",
        "source_prefix_root_sha256",
    ),
    "Proposal": (
        "ordinal", "decision_source_ts_ns", "decision_arrival_ts_ns",
        "decision_source_event_sha256", "completed_data_watermark_source_ts_ns",
        "completed_data_prefix_root_sha256", "instrument", "direction",
        "target_notional_jpy_micros", "max_age_ns", "worker_key",
    ),
    "ArmTerms": (
        "latency_ns", "slippage_micropips_per_side",
        "commission_ppm_per_side", "financing_ppm_per_day", "raw_mid",
    ),
    "Posting": ("account", "amount"),
    "JournalTransaction": (
        "sequence", "arrival_ts_ns", "arm", "proposal_ordinal", "event_kind",
        "event_id", "source_event_sha256", "postings",
    ),
    "ReferenceInput": (
        "ticks", "books", "proposals", "candidate_key", "provenance", "arms",
        "max_trade_quote_staleness_ns", "inventory", "accounting", "evaluation",
        "authority", "registry", "execution_policy_sha256", "raw_hashes",
    ),
    "PositionLot": (
        "arm", "proposal", "signal_id", "economic_lot_id", "common", "entry",
        "entry_price", "entry_price_numerator", "entry_price_denominator",
        "units_micros", "entry_notional_exact", "entry_notional_rounded",
        "due_arrival_ns", "entry_commission_exact", "last_mark_pnl_exact",
        "last_mark_arrival_ns", "signed_exposure_after_entry", "gross_after_entry",
        "marked_equity_after_entry", "required_margin_after_entry",
        "free_margin_after_entry", "closed_disposition",
    ),
    "ClosedDisposition": (
        "position", "exit_tick", "exit_reason", "settlement_arrival_ns", "values",
        "common_gross_jpy_micros", "arm_common_gross_jpy_micros",
        "fill_sizing_drag_jpy_micros", "execution_drag_jpy_micros",
    ),
    "RejectedDisposition": (
        "arm", "proposal", "signal_id", "economic_lot_id", "reason", "common",
        "known_arrival_ns", "settlement_arrival_ns",
    ),
    "PendingRejection": (
        "arm", "proposal", "signal_id", "economic_lot_id", "reason", "common",
        "known_arrival_ns", "settlement_arrival_ns",
    ),
    "RiskSnapshot": (
        "arrival_ts_ns", "source_watermark_ts_ns", "marked_equity_jpy_micros",
        "gross_notional_jpy_micros", "required_margin_jpy_micros",
        "free_margin_jpy_micros", "signed_currency_exposure_jpy_micros",
        "margin_ratio_pass",
    ),
    "ArmReplay": ("positions", "risk_snapshots", "boundary_equities"),
}
_REFERENCE_COST_FIELDS = (
    "latency_ns",
    "slippage_micropips_per_side",
    "commission_ppm_per_side",
    "financing_ppm_per_day",
)
_REFERENCE_FORBIDDEN_NAMES = frozenset({
    "breakpoint", "builtins", "compile", "ctypes", "delattr", "eval", "exec",
    "fcntl", "frame", "globals", "hasattr", "importlib", "input", "locals",
    "marshal", "open", "os", "pathlib", "random", "secrets", "setattr",
    "socket", "subprocess", "sys", "time", "traceback", "uuid", "vars",
    "_getframe", "currentframe", "f_back", "f_builtins", "f_code", "f_globals",
    "f_locals", "gi_frame", "cr_frame", "tb_frame", "__import__",
    "__loader__", "__subclasses__",
})
_REFERENCE_ALLOWED_MODULE_ATTRIBUTES = frozenset({
    ("hashlib", "sha256"),
    ("json", "JSONDecodeError"),
    ("json", "dumps"),
    ("json", "loads"),
    ("re", "compile"),
    ("re", "fullmatch"),
    ("re", "split"),
    ("re", "sub"),
})


def _reference_import_signature(node: ast.stmt) -> tuple[Any, ...] | None:
    if isinstance(node, ast.Import):
        return ("import", tuple((item.name, item.asname) for item in node.names))
    if isinstance(node, ast.ImportFrom):
        return (
            "from",
            node.module,
            tuple((item.name, item.asname) for item in node.names),
            node.level,
        )
    return None


def _scan_reference_source(code_bytes: bytes, filename: str) -> ast.Module:
    """Reject capabilities before any pinned reference byte is executed."""

    try:
        tree = ast.parse(code_bytes, filename=filename, mode="exec")
    except (SyntaxError, UnicodeDecodeError) as error:
        raise LauncherError("pinned reference source cannot be parsed") from error
    imports = tuple(
        signature
        for node in tree.body
        if (signature := _reference_import_signature(node)) is not None
    )
    if imports != _REFERENCE_IMPORTS \
            or any(
                isinstance(node, (ast.Import, ast.ImportFrom))
                for statement in tree.body
                for node in ast.walk(statement)
                if node is not statement
            ):
        raise LauncherError("reference import surface is not exact")

    class_fields: dict[str, tuple[str, ...]] = {}
    ordinary_classes: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        fields = tuple(
            statement.target.id
            for statement in node.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
        )
        if node.name in _REFERENCE_DATACLASS_FIELDS:
            class_fields[node.name] = fields
            decorator_ok = False
            if node.name == "PositionLot":
                decorator_ok = len(node.decorator_list) == 1 \
                    and isinstance(node.decorator_list[0], ast.Name) \
                    and node.decorator_list[0].id == "dataclass"
            else:
                decorator = node.decorator_list[0] if len(node.decorator_list) == 1 else None
                decorator_ok = isinstance(decorator, ast.Call) \
                    and isinstance(decorator.func, ast.Name) \
                    and decorator.func.id == "dataclass" \
                    and not decorator.args \
                    and len(decorator.keywords) == 1 \
                    and decorator.keywords[0].arg == "frozen" \
                    and isinstance(decorator.keywords[0].value, ast.Constant) \
                    and decorator.keywords[0].value.value is True
            if not decorator_ok:
                raise LauncherError("reference dataclass declaration changed")
        else:
            ordinary_classes.add(node.name)
    if class_fields != _REFERENCE_DATACLASS_FIELDS \
            or ordinary_classes != {"ReferenceError", "_Journal"}:
        raise LauncherError("reference class or field surface changed")

    policy_function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_decode_policies"
        ),
        None,
    )
    if policy_function is None:
        raise LauncherError("reference policy decoder is missing")
    cost_assignments = [
        node
        for node in policy_function.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "cost_fields"
    ]
    if len(cost_assignments) != 1 \
            or not isinstance(cost_assignments[0].value, ast.Tuple) \
            or tuple(
                item.value
                for item in cost_assignments[0].value.elts
                if isinstance(item, ast.Constant) and type(item.value) is str
            ) != _REFERENCE_COST_FIELDS \
            or len(cost_assignments[0].value.elts) != len(_REFERENCE_COST_FIELDS):
        raise LauncherError("reference getattr field source changed")

    class ReferenceVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.functions: list[str] = []
            self.getattr_calls = 0
            self.getattr_objects: dict[str, int] = {}
            self.cost_field_generator_depth = 0
            self.cost_predicates: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.functions.append(node.name)
            self.generic_visit(node)
            self.functions.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.functions.append(node.name)
            self.generic_visit(node)
            self.functions.pop()

        def visit_Name(self, node: ast.Name) -> None:
            if (
                node.id == "getattr" and isinstance(node.ctx, ast.Load)
            ) or node.id in _REFERENCE_FORBIDDEN_NAMES \
                    or (
                        node.id.startswith("__") and node.id.endswith("__")
                        and not (
                            node.id == "__all__"
                            and isinstance(node.ctx, ast.Store)
                            and not self.functions
                        )
                    ):
                raise LauncherError("reference source references a forbidden capability")

        def visit_Constant(self, node: ast.Constant) -> None:
            if type(node.value) is str \
                    and node.value.startswith("__") \
                    and node.value.endswith("__"):
                raise LauncherError("reference source contains a dunder token")

        def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
            raise LauncherError("reference source contains assignment-expression rebinding")

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if node.attr.startswith("__") and node.attr.endswith("__"):
                raise LauncherError("reference source uses dunder reflection")
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in {"hashlib", "json", "re"}:
                if not isinstance(node.value, ast.Name) \
                        or (node.value.id, node.attr) \
                        not in _REFERENCE_ALLOWED_MODULE_ATTRIBUTES:
                    raise LauncherError("reference module attribute surface changed")
            self.generic_visit(node)

        def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
            contains_getattr = any(
                isinstance(candidate, ast.Call)
                and isinstance(candidate.func, ast.Name)
                and candidate.func.id == "getattr"
                for candidate in ast.walk(node)
            )
            exact_cost_generator = len(node.generators) == 1 \
                and isinstance(node.generators[0].target, ast.Name) \
                and node.generators[0].target.id == "field" \
                and isinstance(node.generators[0].iter, ast.Name) \
                and node.generators[0].iter.id == "cost_fields" \
                and not node.generators[0].ifs \
                and node.generators[0].is_async == 0
            if not contains_getattr:
                self.generic_visit(node)
                return
            if not exact_cost_generator:
                raise LauncherError("reference getattr generator binding changed")

            def exact_getattr(candidate: ast.AST, object_name: str) -> bool:
                return isinstance(candidate, ast.Call) \
                    and isinstance(candidate.func, ast.Name) \
                    and candidate.func.id == "getattr" \
                    and not candidate.keywords \
                    and len(candidate.args) == 2 \
                    and isinstance(candidate.args[0], ast.Name) \
                    and candidate.args[0].id == object_name \
                    and isinstance(candidate.args[1], ast.Name) \
                    and candidate.args[1].id == "field"

            predicate = None
            if isinstance(node.elt, ast.Compare) \
                    and len(node.elt.ops) == 1 \
                    and len(node.elt.comparators) == 1:
                comparator = node.elt.comparators[0]
                if exact_getattr(node.elt.left, "raw_arm") \
                        and isinstance(node.elt.ops[0], ast.NotEq) \
                        and isinstance(comparator, ast.Constant) \
                        and type(comparator.value) is int \
                        and comparator.value == 0:
                    predicate = "RAW_NE_ZERO"
                elif exact_getattr(node.elt.left, "adverse_arm") \
                        and exact_getattr(comparator, "base_arm") \
                        and isinstance(node.elt.ops[0], (ast.Lt, ast.Gt)):
                    predicate = (
                        "ADVERSE_LT_BASE"
                        if isinstance(node.elt.ops[0], ast.Lt)
                        else "ADVERSE_GT_BASE"
                    )
            if predicate is None:
                raise LauncherError("reference getattr predicate changed")
            self.cost_predicates.append(predicate)
            self.cost_field_generator_depth += 1
            try:
                self.visit(node.elt)
                self.visit(node.generators[0].target)
                self.visit(node.generators[0].iter)
            finally:
                self.cost_field_generator_depth -= 1

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name) and node.func.id == "getattr":
                self.getattr_calls += 1
                if self.functions[-1:] != ["_decode_policies"] \
                        or self.cost_field_generator_depth != 1 \
                        or len(node.args) != 2 or node.keywords \
                        or not isinstance(node.args[0], ast.Name) \
                        or node.args[0].id not in {
                            "raw_arm", "base_arm", "adverse_arm"
                        } \
                        or not isinstance(node.args[1], ast.Name) \
                        or node.args[1].id != "field":
                    raise LauncherError("reference getattr capability is outside its exact use")
                object_name = node.args[0].id
                self.getattr_objects[object_name] = (
                    self.getattr_objects.get(object_name, 0) + 1
                )
                for argument in node.args:
                    self.visit(argument)
                return
            self.generic_visit(node)

    visitor = ReferenceVisitor()
    visitor.visit(tree)
    if visitor.getattr_calls != 5 or visitor.getattr_objects != {
        "raw_arm": 1,
        "base_arm": 2,
        "adverse_arm": 2,
    } or sorted(visitor.cost_predicates) != [
        "ADVERSE_GT_BASE", "ADVERSE_LT_BASE", "RAW_NE_ZERO"
    ]:
        raise LauncherError("reference getattr use count changed")
    return tree


def _scan_reference_code(code: CodeType) -> None:
    pending: list[tuple[CodeType, bool]] = [(code, True)]
    while pending:
        current, module_level = pending.pop()
        for instruction in dis.get_instructions(current):
            if not module_level and instruction.opname in {
                "IMPORT_FROM", "IMPORT_NAME", "IMPORT_STAR", "LOAD_BUILD_CLASS"
            }:
                raise LauncherError("reference nested code contains import capability")
        for constant in current.co_consts:
            if type(constant) is CodeType:
                pending.append((constant, False))
            elif type(constant) is str and constant in _REFERENCE_FORBIDDEN_NAMES:
                raise LauncherError("reference code contains a forbidden capability token")


def _compile_and_load_reference(
    code_bytes: bytes,
    audit_boundary: _AuditBoundary,
) -> tuple[types.ModuleType, Callable[[Mapping[str, bytes]], dict[str, Any]]]:
    """Load the fixed pure reference engine without a project import path."""

    filename = "<sealed-reference-v2>"
    module_name = "_sealed_double_entry_reference_v2"
    if not audit_boundary._reference_phase_active:
        raise LauncherError("reference code requires its active audit phase")
    if module_name in sys.modules:
        raise LauncherError("sealed reference module name is already occupied")
    audit_boundary.allow_exact_compile(code_bytes, filename)
    try:
        code = compile(code_bytes, filename, "exec", dont_inherit=True, optimize=0)
    except (SyntaxError, ValueError) as error:
        raise LauncherError("pinned reference code cannot be compiled") from error
    _scan_reference_code(code)
    runtime_builtins = {
        name: vars(builtins)[name]
        for name in PURE_REFERENCE_BUILTINS | REFERENCE_INITIALIZATION_BUILTINS
    }
    runtime_builtins["__import__"] = audit_boundary.sealed_import
    module = types.ModuleType(module_name)
    module.__dict__.update({
        "__file__": filename,
        "__package__": None,
        "__spec__": None,
        "__builtins__": runtime_builtins,
    })
    audit_boundary.allow_exact_imports(frozenset({
        "__future__",
        "collections",
        "dataclasses",
        "fractions",
        "hashlib",
        "json",
        "re",
        "types",
        "typing",
    }))
    audit_boundary.allow_dataclass_codegen()
    audit_boundary.allow_exact_exec(code)
    sys.modules[module_name] = module
    try:
        try:
            exec(code, module.__dict__, module.__dict__)
        except BaseException:
            module.__dict__.clear()
            raise
    finally:
        sys.modules.pop(module_name, None)
        audit_boundary.seal_dataclass_codegen()
        audit_boundary.seal_imports()
    for name in REFERENCE_INITIALIZATION_BUILTINS:
        runtime_builtins.pop(name, None)
    for name in (
        "Any",
        "Iterable",
        "Sequence",
        "dataclass",
        "__file__",
        "__loader__",
        "__package__",
        "__spec__",
    ):
        module.__dict__.pop(name, None)
    replay = module.__dict__.get("replay_reference")
    if type(replay) is not types.FunctionType:
        raise LauncherError("pinned reference lacks replay_reference")
    return module, replay


def _pure_reference_target(
    namespace: Mapping[str, Any],
) -> types.FunctionType:
    target = namespace.get("replay_reference")
    decoder = namespace.get("decode_reference_input")
    if type(target) is not types.FunctionType or target.__closure__ is not None \
            or type(decoder) is not types.FunctionType or decoder.__closure__ is not None:
        raise LauncherError("pinned reference pure entrypoints changed")
    if target.__globals__ is not namespace or decoder.__globals__ is not namespace:
        raise LauncherError("pinned reference entrypoint globals changed")
    runtime_builtins = namespace.get("__builtins__")
    if type(runtime_builtins) is not dict \
            or set(runtime_builtins) != PURE_REFERENCE_BUILTINS:
        raise LauncherError("sealed reference builtin capability set is not exact")
    if any(name in runtime_builtins for name in REFERENCE_INITIALIZATION_BUILTINS):
        raise LauncherError("sealed reference retained an initialization capability")
    expected_modules = {
        "hashlib": hashlib,
        "json": json,
        "re": re,
    }
    actual_modules = {
        name: value
        for name, value in namespace.items()
        if type(value) is types.ModuleType
    }
    if actual_modules != expected_modules:
        raise LauncherError("sealed reference retained a forbidden module capability")
    expected_imports = {
        "Counter": collections.Counter,
        "defaultdict": collections.defaultdict,
        "Fraction": fractions.Fraction,
        "MappingProxyType": types.MappingProxyType,
        "Mapping": typing.Mapping,
    }
    if any(namespace.get(name) is not value for name, value in expected_imports.items()):
        raise LauncherError("sealed reference retained import identities changed")
    if namespace.get("ARMS") != (
        "RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS"
    ) or type(namespace.get("ARMS")) is not tuple:
        raise LauncherError("sealed reference arm identity changed")
    if namespace.get("ENGINE_ID") != "EVENT_SOURCED_DOUBLE_ENTRY_REFERENCE_V1":
        raise LauncherError("sealed reference engine identity changed")
    for class_name, expected_fields in _REFERENCE_DATACLASS_FIELDS.items():
        candidate = namespace.get(class_name)
        fields = getattr(candidate, "__dataclass_fields__", None)
        if type(candidate) is not type or type(fields) is not dict \
                or tuple(fields) != expected_fields:
            raise LauncherError("sealed reference dataclass runtime surface changed")
    reference_error = namespace.get("ReferenceError")
    journal = namespace.get("_Journal")
    if type(reference_error) is not type \
            or reference_error.__bases__ != (RuntimeError,) \
            or type(journal) is not type:
        raise LauncherError("sealed reference class identities changed")
    return target


def _embedded_sha256(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    _assert_canonical_value(unsigned, field)
    return _sha256(_canonical_bytes(unsigned))


def _serialize_reference_result(
    result: Any,
    raw_artifacts: Mapping[str, bytes],
) -> bytes:
    if type(result) is not dict:
        raise LauncherError("reference result container must be an exact dict")
    expected_keys = frozenset({
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
    _require_exact_keys(result, expected_keys, "reference result")
    if result["engine_id"] != "EVENT_SOURCED_DOUBLE_ENTRY_REFERENCE_V1" \
            or type(result["all_transactions_balanced"]) is not bool \
            or result["all_transactions_balanced"] is not True:
        raise LauncherError("reference result identity or balance proof is invalid")
    for field in (
        "input_root_sha256",
        "ledger_terminal_hash",
        "proposal_provenance_root_sha256",
        "journal_root_sha256",
        "economic_projection_sha256",
    ):
        _require_sha256(result[field], f"reference result {field}")
    for field in ("ledger_row_count", "journal_transaction_count"):
        if type(result[field]) is not int or result[field] < 0:
            raise LauncherError(f"reference result {field} must be a nonnegative integer")
    ledger_bytes = result["ledger_bytes"]
    if type(ledger_bytes) is not bytes:
        raise LauncherError("reference ledger must be immutable bytes")
    ledger_rows: list[dict[str, Any]] = []
    if ledger_bytes:
        if not ledger_bytes.endswith(b"\n") or ledger_bytes.endswith(b"\n\n"):
            raise LauncherError("reference ledger text encoding is not canonical JSONL")
        for index, line in enumerate(ledger_bytes.splitlines(keepends=True), 1):
            row = _strict_json_object(line, f"reference ledger row {index}")
            if row.get("ledger_sequence") != index \
                    or type(row.get("ledger_sequence")) is not int:
                raise LauncherError("reference ledger sequence is not canonical")
            previous = ZERO_SHA256 if index == 1 else ledger_rows[-1]["record_hash"]
            if row.get("previous_hash") != previous \
                    or _require_sha256(
                        row.get("record_hash"), "reference ledger record hash"
                    ) != _embedded_sha256(row, "record_hash"):
                raise LauncherError("reference ledger hash chain is invalid")
            ledger_rows.append(row)
    if len(ledger_rows) != result["ledger_row_count"]:
        raise LauncherError("reference ledger row count mismatch")
    expected_terminal = ledger_rows[-1]["record_hash"] if ledger_rows else ZERO_SHA256
    if result["ledger_terminal_hash"] != expected_terminal:
        raise LauncherError("reference ledger terminal hash mismatch")
    metrics = result["oracle_metrics"]
    if type(metrics) is not dict:
        raise LauncherError("reference metrics must be an exact object")
    _assert_canonical_value(metrics, "reference metrics")
    if _require_sha256(metrics.get("metrics_sha256"), "reference metrics hash") \
            != _embedded_sha256(metrics, "metrics_sha256") \
            or type(metrics.get("external_orders")) is not int \
            or metrics["external_orders"] != 0 \
            or type(metrics.get("terminal_inventory_mtm_jpy_micros")) is not int \
            or metrics["terminal_inventory_mtm_jpy_micros"] != 0:
        raise LauncherError("reference metrics hash or terminal state is invalid")
    expected_input_root = _sha256(_canonical_bytes({
        "artifact_sha256": {
            role: _sha256(raw_artifacts[role])
            for role in sorted(raw_artifacts)
        },
    }))
    if result["input_root_sha256"] != expected_input_root:
        raise LauncherError("reference input root differs from snapshotted artifacts")
    projection = {
        "all_transactions_balanced": True,
        "engine_id": result["engine_id"],
        "input_root_sha256": result["input_root_sha256"],
        "journal_root_sha256": result["journal_root_sha256"],
        "journal_transaction_count": result["journal_transaction_count"],
        "ledger_row_count": result["ledger_row_count"],
        "ledger_sha256": _sha256(ledger_bytes),
        "ledger_terminal_hash": result["ledger_terminal_hash"],
        "oracle_metrics_sha256": metrics["metrics_sha256"],
        "proposal_provenance_root_sha256": result[
            "proposal_provenance_root_sha256"
        ],
    }
    if result["economic_projection_sha256"] != _sha256(
        _canonical_bytes(projection)
    ):
        raise LauncherError("reference economic projection hash mismatch")
    encoded_ledger = base64.b64encode(ledger_bytes).decode("ascii")
    try:
        decoded_ledger = base64.b64decode(encoded_ledger, validate=True)
    except (ValueError, TypeError) as error:
        raise LauncherError("reference ledger base64 encoding failed") from error
    if decoded_ledger != ledger_bytes \
            or base64.b64encode(decoded_ledger).decode("ascii") != encoded_ledger:
        raise LauncherError("reference ledger base64 encoding is not standard padded form")
    snapshot = {
        key: value
        for key, value in result.items()
        if key != "ledger_bytes"
    }
    snapshot["ledger_bytes_base64"] = encoded_ledger
    _assert_canonical_value(snapshot, "reference result snapshot")
    snapshot_bytes = _canonical_bytes(snapshot) + b"\n"
    if _strict_json_object(snapshot_bytes, "reference result snapshot") != snapshot:
        raise LauncherError("reference result snapshot round-trip mismatch")
    return snapshot_bytes


def _run_reference_snapshot(
    code_bytes: bytes,
    raw_artifacts: Mapping[str, bytes],
    audit_boundary: _AuditBoundary,
) -> tuple[bytes, str]:
    namespace: dict[str, Any] | None = None
    _scan_reference_source(code_bytes, "<sealed-reference-v2>")
    audit_boundary.begin_reference_phase()
    try:
        module, _ = _compile_and_load_reference(code_bytes, audit_boundary)
        namespace = module.__dict__
        replay = _pure_reference_target(namespace)
        result = replay(types.MappingProxyType(dict(raw_artifacts)))
        snapshot_bytes = _serialize_reference_result(result, raw_artifacts)
        snapshot_sha256 = _sha256(snapshot_bytes)
    finally:
        if namespace is not None:
            namespace.clear()
        audit_boundary.end_reference_phase()
    return snapshot_bytes, snapshot_sha256


def _validate_verifier_return(
    value: Any,
    *,
    request_bytes: bytes,
    artifact_blobs: tuple[tuple[str, bytes], ...],
    reference_result_bytes: bytes,
    reference_result_sha256: str,
    launcher_sha256: str,
) -> tuple[bytes, bytes, dict[str, Any]]:
    if type(value) is not tuple or len(value) != 2 \
            or type(value[0]) is not bytes or type(value[1]) is not bytes:
        raise LauncherError("verifier return container must be an exact byte pair")
    receipt_bytes, commit_bytes = value
    receipt = _strict_json_object(receipt_bytes, "verifier receipt")
    commit = _strict_json_object(commit_bytes, "verifier COMMIT")
    reference_snapshot = _strict_json_object(
        reference_result_bytes,
        "reference result snapshot",
    )
    blobs = dict(artifact_blobs)
    if tuple(sorted(blobs)) != VERIFIER_ARTIFACT_ROLES \
            or len(blobs) != len(artifact_blobs):
        raise LauncherError("verifier artifact tuple changed before return validation")
    _require_exact_keys(receipt, VERIFIER_RECEIPT_KEYS, "verifier receipt")
    _require_exact_keys(
        commit,
        frozenset({
            "schema_version",
            "request_sha256",
            "receipt_sha256",
            "receipt_size_bytes",
            "verifier_receipt_sha256",
        }),
        "verifier COMMIT",
    )
    if type(receipt["schema_version"]) is not int \
            or receipt["schema_version"] != 2 \
            or receipt["verifier_implementation"] != "INDEPENDENT_JPY_ORACLE_VERIFIER_V2" \
            or receipt["status"] != "VERIFIED_ACCOUNTING_ONLY" \
            or receipt["classification"] \
                != "FUTURE_ONLY_ACCOUNTING_ONLY_LOCAL_UNANCHORED_NOT_ADMISSIBLE" \
            or receipt["anchor_status"] != "LOCAL_UNANCHORED":
        raise LauncherError("verifier receipt identity or classification mismatch")
    exact_booleans = {
        "causal_signal_admission": False,
        "release_evidence_eligible": False,
        "admission_eligible": False,
        "detector_replay_receipt_required": True,
        "independently_rebuilt_ledger": True,
        "independently_rebuilt_metrics": True,
        "producer_result_or_metrics_used": False,
        "reference_all_transactions_balanced": True,
        "reference_accounting_diagnostics_only": True,
        "reference_n_eff_statistical_admission_allowed": False,
        "reference_direction_accuracy_profit_gate_allowed": False,
    }
    if any(
        type(receipt[field]) is not bool or receipt[field] is not expected
        for field, expected in exact_booleans.items()
    ):
        raise LauncherError("verifier receipt boolean authority gate mismatch")
    authority = receipt["authority"]
    expected_authority = {
        "paper_only": True,
        "live_authority": False,
        "broker_account_access": False,
        "credential_access": False,
        "order_endpoint": False,
        "external_orders": 0,
        "deploy": False,
        "external_config_mutation": False,
    }
    if type(authority) is not dict or authority != expected_authority \
            or any(type(authority[key]) is not type(expected) for key, expected in expected_authority.items()) \
            or type(receipt["external_orders"]) is not int \
            or receipt["external_orders"] != 0 \
            or type(receipt["terminal_inventory_mtm_jpy_micros"]) is not int \
            or receipt["terminal_inventory_mtm_jpy_micros"] != 0:
        raise LauncherError("verifier receipt authority or terminal state mismatch")
    expected_input_hashes = {
        role: _sha256(data)
        for role, data in artifact_blobs
    }
    if receipt["input_artifact_sha256"] != expected_input_hashes:
        raise LauncherError("verifier receipt input artifact binding mismatch")
    oracle_manifest = _strict_json_object(blobs["oracle_manifest"], "oracle manifest")
    if receipt["oracle_manifest_sha256"] != _sha256(blobs["oracle_manifest"]) \
            or type(receipt["oracle_manifest_size_bytes"]) is not int \
            or receipt["oracle_manifest_size_bytes"] != len(blobs["oracle_manifest"]) \
            or receipt["oracle_ledger_sha256"] != _sha256(blobs["oracle_ledger"]) \
            or type(receipt["oracle_ledger_size_bytes"]) is not int \
            or receipt["oracle_ledger_size_bytes"] != len(blobs["oracle_ledger"]) \
            or receipt["oracle_root_sha256"] != oracle_manifest.get("oracle_root_sha256") \
            or receipt["oracle_ledger_terminal_hash"] \
                != oracle_manifest.get("oracle_ledger_terminal_hash") \
            or receipt["raw_source_manifest_sha256"] \
                != _sha256(blobs["source_manifest"]) \
            or receipt["oracle_request_sha256"] != _sha256(blobs["oracle_request"]) \
            or receipt["oracle_release_content_binding"] \
                != oracle_manifest.get("oracle_release_content_binding") \
            or receipt["oracle_execution_provenance_scope"] \
                != oracle_manifest.get("oracle_execution_provenance_scope"):
        raise LauncherError("verifier receipt Oracle evidence binding mismatch")
    try:
        reference_ledger = base64.b64decode(
            reference_snapshot["ledger_bytes_base64"],
            validate=True,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise LauncherError("reference result ledger snapshot is invalid") from error
    if base64.b64encode(reference_ledger).decode("ascii") \
            != reference_snapshot["ledger_bytes_base64"]:
        raise LauncherError("reference result ledger snapshot is not standard base64")
    if receipt["expected_canonical_ledger_sha256"] != _sha256(reference_ledger) \
            or receipt["verified_oracle_metrics"] \
                != reference_snapshot.get("oracle_metrics") \
            or receipt["reference_engine_id"] != reference_snapshot.get("engine_id") \
            or receipt["reference_input_root_sha256"] \
                != reference_snapshot.get("input_root_sha256") \
            or receipt["reference_journal_root_sha256"] \
                != reference_snapshot.get("journal_root_sha256") \
            or receipt["reference_journal_transaction_count"] \
                != reference_snapshot.get("journal_transaction_count") \
            or type(receipt["reference_journal_transaction_count"]) is not int \
            or receipt["reference_economic_projection_sha256"] \
                != reference_snapshot.get("economic_projection_sha256"):
        raise LauncherError("verifier receipt reference result binding mismatch")
    reference_release = PINNED_RELEASES["REFERENCE"]
    expected_binding = {
        "code_sha256": PINNED_RELEASES["VERIFIER"]["code_sha256"],
        "schema_sha256": PINNED_RELEASES["VERIFIER"]["schema_sha256"],
        "launcher_sha256": launcher_sha256,
        "snapshot_mode": "SEALED_FD_COMPILE_EXEC_V2",
        "reference_code_sha256": reference_release["code_sha256"],
        "reference_contract_sha256": reference_release["contract_sha256"],
        "reference_result_sha256": reference_result_sha256,
    }
    if receipt["verifier_release_content_binding"] != expected_binding \
            or receipt["reference_code_sha256"] != reference_release["code_sha256"] \
            or receipt["reference_contract_sha256"] \
                != reference_release["contract_sha256"] \
            or receipt["reference_result_sha256"] != reference_result_sha256 \
            or receipt["verifier_execution_provenance_scope"] != (
                "LOCAL_CALLER_ASSERTED_CONTENT_BINDING_NOT_EXECUTION_ATTESTATION_"
                "NOT_EXTERNALLY_ANCHORED"
            ) \
            or _require_sha256(
                receipt["verifier_receipt_sha256"],
                "verifier receipt embedded hash",
            ) != _embedded_sha256(receipt, "verifier_receipt_sha256"):
        raise LauncherError("verifier receipt release or self binding mismatch")
    if type(commit["schema_version"]) is not int or commit["schema_version"] != 2 \
            or commit["request_sha256"] != _sha256(request_bytes) \
            or commit["receipt_sha256"] != _sha256(receipt_bytes) \
            or type(commit["receipt_size_bytes"]) is not int \
            or commit["receipt_size_bytes"] != len(receipt_bytes) \
            or commit["verifier_receipt_sha256"] \
                != receipt["verifier_receipt_sha256"]:
        raise LauncherError("verifier COMMIT binding mismatch")
    return receipt_bytes, commit_bytes, receipt


def _lstat_at(directory_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _assert_publish_lock(
    output_root_fd: int,
    lock_name: str,
    lock_fd: int,
) -> None:
    _validate_root_fd(output_root_fd, "verifier publication root")
    try:
        held = os.fstat(lock_fd)
        named = os.stat(lock_name, dir_fd=output_root_fd, follow_symlinks=False)
    except OSError as error:
        raise LauncherError("verifier publication lock identity changed") from error
    if _fd_access_mode(lock_fd) != os.O_RDWR:
        raise LauncherError("verifier publication lock access mode changed")
    for info in (held, named):
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() \
                or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600:
            raise LauncherError("verifier publication lock is unsafe")
    _require_launcher_no_extended_acl_fd(
        lock_fd, "verifier publication lock"
    )
    if (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino):
        raise LauncherError("verifier publication lock pathname changed")


def _assert_directory_path_identity(
    root_fd: int,
    name: str,
    directory_fd: int,
    label: str,
) -> None:
    _validate_root_fd(root_fd, f"{label} root")
    try:
        held = os.fstat(directory_fd)
        named = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    except OSError as error:
        raise LauncherError(f"{label} pathname changed") from error
    if not stat.S_ISDIR(held.st_mode) or not stat.S_ISDIR(named.st_mode) \
            or (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino):
        raise LauncherError(f"{label} pathname changed")
    if held.st_uid != os.geteuid() or held.st_mode & 0o022:
        raise LauncherError(f"{label} directory is unsafe")
    _require_launcher_no_extended_acl_fd(directory_fd, label)


def _write_immutable_file_at(
    directory_fd: int,
    name: str,
    data: bytes,
) -> None:
    if SAFE_COMPONENT_RE.fullmatch(name) is None:
        raise LauncherError("verifier output filename is unsafe")
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
    except OSError as error:
        raise LauncherError("verifier output file cannot be created exclusively") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid() \
                or before.st_nlink != 1 or stat.S_IMODE(before.st_mode) != 0o600:
            raise LauncherError("verifier output file is unsafe")
        _require_launcher_no_extended_acl_fd(
            descriptor, f"new verifier output {name}"
        )
        _write_all(descriptor, data)
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        _require_launcher_no_extended_acl_fd(
            descriptor, f"written verifier output {name}"
        )
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino) \
                or after.st_size != len(data) or after.st_nlink != 1:
            raise LauncherError("verifier output file changed while writing")
    finally:
        os.close(descriptor)


def _read_immutable_file_at(
    directory_fd: int,
    name: str,
    label: str,
) -> bytes:
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_fd,
        )
    except OSError as error:
        raise LauncherError(f"{label} cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid() \
                or before.st_nlink != 1 or stat.S_IMODE(before.st_mode) != 0o600 \
                or before.st_size > MAX_JSON_ARTIFACT_BYTES:
            raise LauncherError(f"{label} is not an immutable output file")
        _require_launcher_no_extended_acl_fd(descriptor, label)
        chunks: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(
                descriptor,
                min(1024 * 1024, before.st_size - offset),
                offset,
            )
            if not chunk:
                raise LauncherError(f"{label} truncated during validation")
            chunks.append(chunk)
            offset += len(chunk)
        if os.pread(descriptor, 1, before.st_size):
            raise LauncherError(f"{label} grew during validation")
        after = os.fstat(descriptor)
        _require_launcher_no_extended_acl_fd(
            descriptor, f"{label} final fence"
        )
        identity = lambda info: (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
            info.st_nlink,
        )
        if identity(before) != identity(after):
            raise LauncherError(f"{label} changed during validation")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _validate_verifier_output_directory(
    directory_fd: int,
    receipt_bytes: bytes,
    commit_bytes: bytes,
) -> None:
    info = os.fstat(directory_fd)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() \
            or info.st_mode & 0o022:
        raise LauncherError("verifier output directory is unsafe")
    _require_launcher_no_extended_acl_fd(
        directory_fd, "verifier output directory"
    )
    expected = {
        "verifier_receipt.json": receipt_bytes,
        "COMMIT.json": commit_bytes,
    }
    if set(os.listdir(directory_fd)) != set(expected):
        raise LauncherError("verifier output file set is not exact")
    for name, data in expected.items():
        if _read_immutable_file_at(
            directory_fd,
            name,
            f"verifier output {name}",
        ) != data:
            raise LauncherError(f"verifier output {name} differs from sealed bytes")


def _validate_runtime_output_acl_tree(
    output_root_fd: int,
    output_name: str,
    expected_files: frozenset[str],
    label: str,
) -> None:
    """Validate final runtime children through opened FDs, including ACLs."""
    _validate_root_fd(output_root_fd, f"{label} root")
    lock_name = f".{output_name}.lock"
    lock_fd = os.open(
        lock_name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=output_root_fd,
    )
    try:
        lock_info = os.fstat(lock_fd)
        if not stat.S_ISREG(lock_info.st_mode) \
                or lock_info.st_uid != os.geteuid() \
                or lock_info.st_nlink != 1 or lock_info.st_mode & 0o022:
            raise LauncherError(f"{label} lock is unsafe")
        _require_launcher_no_extended_acl_fd(lock_fd, f"{label} lock")
    finally:
        os.close(lock_fd)
    directory_fd = os.open(
        output_name,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=output_root_fd,
    )
    try:
        _assert_directory_path_identity(
            output_root_fd, output_name, directory_fd, label
        )
        if set(os.listdir(directory_fd)) != set(expected_files):
            raise LauncherError(f"{label} file set is not exact")
        for name in sorted(expected_files):
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=directory_fd,
            )
            try:
                info = os.fstat(descriptor)
                if not stat.S_ISREG(info.st_mode) \
                        or info.st_uid != os.geteuid() \
                        or info.st_nlink != 1 or info.st_mode & 0o022:
                    raise LauncherError(f"{label} file {name} is unsafe")
                _require_launcher_no_extended_acl_fd(
                    descriptor, f"{label} file {name}"
                )
            finally:
                os.close(descriptor)
        _assert_directory_path_identity(
            output_root_fd, output_name, directory_fd, f"{label} final fence"
        )
    finally:
        os.close(directory_fd)
    _validate_root_fd(output_root_fd, f"{label} root final fence")


def _publish_verifier_bytes(
    output_root_fd: int,
    output_name: str,
    receipt_bytes: bytes,
    commit_bytes: bytes,
    rename_exclusive: Callable[[int, str, str], None],
) -> None:
    _validate_root_fd(output_root_fd, "output root")
    if type(output_name) is not str or SAFE_COMPONENT_RE.fullmatch(output_name) is None:
        raise LauncherError("verifier output directory is unsafe")
    lock_name = f".{output_name}.lock"
    try:
        lock_fd = os.open(
            lock_name,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=output_root_fd,
        )
    except OSError as error:
        raise LauncherError("verifier publication lock cannot be opened") from error
    locked = False
    try:
        _assert_publish_lock(output_root_fd, lock_name, lock_fd)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise LauncherError("concurrent verifier publication is forbidden") from error
        locked = True
        _assert_publish_lock(output_root_fd, lock_name, lock_fd)
        os.ftruncate(lock_fd, 0)
        os.lseek(lock_fd, 0, os.SEEK_SET)
        _write_all(lock_fd, _canonical_bytes({
            "receipt_sha256": _sha256(receipt_bytes),
            "commit_sha256": _sha256(commit_bytes),
        }) + b"\n")
        os.fsync(lock_fd)
        os.fsync(output_root_fd)
        _assert_publish_lock(output_root_fd, lock_name, lock_fd)
        existing = _lstat_at(output_root_fd, output_name)
        if existing is not None:
            if not stat.S_ISDIR(existing.st_mode):
                raise LauncherError("verifier output leaf is not a directory")
            final_fd = os.open(
                output_name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=output_root_fd,
            )
            try:
                _assert_directory_path_identity(
                    output_root_fd,
                    output_name,
                    final_fd,
                    "verifier output",
                )
                _validate_verifier_output_directory(
                    final_fd,
                    receipt_bytes,
                    commit_bytes,
                )
                _assert_directory_path_identity(
                    output_root_fd,
                    output_name,
                    final_fd,
                    "verifier output",
                )
            finally:
                os.close(final_fd)
            _assert_publish_lock(output_root_fd, lock_name, lock_fd)
            return
        transaction = _sha256(_canonical_bytes({
            "output_name": output_name,
            "receipt_sha256": _sha256(receipt_bytes),
            "commit_sha256": _sha256(commit_bytes),
        }))
        stage_name = f".{output_name}.{transaction[:16]}.stage"
        stage_info = _lstat_at(output_root_fd, stage_name)
        if stage_info is None:
            os.mkdir(stage_name, 0o700, dir_fd=output_root_fd)
            os.fsync(output_root_fd)
        elif not stat.S_ISDIR(stage_info.st_mode):
            raise LauncherError("verifier publication stage is not a directory")
        stage_fd = os.open(
            stage_name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=output_root_fd,
        )
        try:
            held_stage = os.fstat(stage_fd)
            _require_launcher_no_extended_acl_fd(
                stage_fd, "verifier publication stage"
            )
            _assert_directory_path_identity(
                output_root_fd,
                stage_name,
                stage_fd,
                "verifier publication stage",
            )
            if not stat.S_ISDIR(held_stage.st_mode) \
                    or held_stage.st_uid != os.geteuid() \
                    or held_stage.st_mode & 0o022:
                raise LauncherError("verifier publication stage is unsafe")
            present = set(os.listdir(stage_fd))
            expected = {
                "verifier_receipt.json": receipt_bytes,
                "COMMIT.json": commit_bytes,
            }
            if not present <= set(expected) \
                    or "COMMIT.json" in present and present != set(expected):
                raise LauncherError("verifier publication stage is partial or unexpected")
            for name in ("verifier_receipt.json", "COMMIT.json"):
                _assert_publish_lock(output_root_fd, lock_name, lock_fd)
                if name in present:
                    if _read_immutable_file_at(
                        stage_fd,
                        name,
                        f"verifier stage {name}",
                    ) != expected[name]:
                        raise LauncherError(f"verifier stage {name} differs")
                else:
                    _write_immutable_file_at(stage_fd, name, expected[name])
            os.fsync(stage_fd)
            _validate_verifier_output_directory(
                stage_fd,
                receipt_bytes,
                commit_bytes,
            )
            stage_path = _lstat_at(output_root_fd, stage_name)
            if stage_path is None or (stage_path.st_dev, stage_path.st_ino) != (
                held_stage.st_dev,
                held_stage.st_ino,
            ):
                raise LauncherError("verifier publication stage pathname changed")
            _assert_publish_lock(output_root_fd, lock_name, lock_fd)
            rename_exclusive(output_root_fd, stage_name, output_name)
            os.fsync(output_root_fd)
            final_fd = os.open(
                output_name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=output_root_fd,
            )
            try:
                final_info = os.fstat(final_fd)
                _require_launcher_no_extended_acl_fd(
                    final_fd, "published verifier directory"
                )
                if (final_info.st_dev, final_info.st_ino) != (
                    held_stage.st_dev,
                    held_stage.st_ino,
                ):
                    raise LauncherError("published verifier directory inode mismatch")
                _assert_directory_path_identity(
                    output_root_fd,
                    output_name,
                    final_fd,
                    "published verifier directory",
                )
                _validate_verifier_output_directory(
                    final_fd,
                    receipt_bytes,
                    commit_bytes,
                )
                _assert_directory_path_identity(
                    output_root_fd,
                    output_name,
                    final_fd,
                    "published verifier directory",
                )
            finally:
                os.close(final_fd)
            _assert_publish_lock(output_root_fd, lock_name, lock_fd)
        finally:
            os.close(stage_fd)
    finally:
        lock_error: BaseException | None = None
        if locked:
            try:
                _assert_publish_lock(output_root_fd, lock_name, lock_fd)
            except BaseException as error:
                lock_error = error
        try:
            if locked:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
        if lock_error is not None:
            raise lock_error


def _freeze_verifier_global(value: Any) -> Any:
    if type(value) is dict:
        return types.MappingProxyType({
            key: _freeze_verifier_global(item)
            for key, item in value.items()
        })
    if type(value) is list:
        return tuple(_freeze_verifier_global(item) for item in value)
    if type(value) is set:
        return frozenset(_freeze_verifier_global(item) for item in value)
    if type(value) is tuple:
        return tuple(_freeze_verifier_global(item) for item in value)
    return value


def _compile_and_load(
    operation: str,
    code_bytes: bytes,
    schema_bytes: bytes,
    launcher_sha256: str,
    rename_exclusive: Callable[[int, str, str], None] | None,
    audit_boundary: _AuditBoundary,
    *,
    contract_bytes: bytes | None = None,
) -> dict[str, Any]:
    filename = f"<sealed-{operation.lower()}-v2>"
    audit_boundary.allow_exact_compile(code_bytes, filename)
    try:
        code = compile(code_bytes, filename, "exec", dont_inherit=True, optimize=0)
    except (SyntaxError, ValueError) as error:
        raise LauncherError(f"pinned {operation.lower()} code cannot be compiled") from error
    if operation == "VERIFIER":
        runtime_builtins = {
            name: vars(builtins)[name]
            for name in PURE_VERIFIER_BUILTINS | VERIFIER_INITIALIZATION_BUILTINS
        }
    else:
        runtime_builtins = dict(vars(builtins))
    runtime_builtins["__import__"] = audit_boundary.sealed_import
    namespace: dict[str, Any] = {
        "__name__": f"_sealed_{operation.lower()}_v2",
        "__file__": filename,
        "__package__": None,
        "__spec__": None,
        "__builtins__": runtime_builtins,
        "_SEALED_RUNTIME_CODE_BYTES": code_bytes,
        "_SEALED_SCHEMA_BYTES": schema_bytes,
        "_SEALED_LAUNCHER_SHA256": launcher_sha256,
    }
    if operation == "ORACLE":
        if contract_bytes is None:
            raise LauncherError("pinned Oracle contract bytes are required")
        if rename_exclusive is None:
            raise LauncherError("pinned Oracle rename capability is required")
        namespace["_SEALED_CONTRACT_BYTES"] = contract_bytes
        namespace["_SEALED_RENAME_EXCLUSIVE"] = rename_exclusive
        allowed_imports = frozenset({
            "__future__",
            "argparse",
            "collections",
            "datetime",
            "fcntl",
            "fractions",
            "hashlib",
            "json",
            "os",
            "pathlib",
            "re",
            "stat",
            "sys",
            "typing",
        })
    else:
        namespace["_SEALED_RUNTIME_CODE_SHA256"] = _sha256(code_bytes)
        namespace["_SEALED_SCHEMA_SHA256"] = _sha256(schema_bytes)
        allowed_imports = frozenset({
            "__future__",
            "base64",
            "collections",
            "datetime",
            "fractions",
            "hashlib",
            "json",
            "re",
            "typing",
        })
    audit_boundary.allow_exact_imports(allowed_imports)
    audit_boundary.allow_exact_exec(code)
    try:
        exec(code, namespace, namespace)
    finally:
        audit_boundary.seal_imports()
    if operation == "VERIFIER":
        for name in VERIFIER_INITIALIZATION_BUILTINS:
            runtime_builtins.pop(name, None)
        for name, value in tuple(namespace.items()):
            if name != "__builtins__":
                namespace[name] = _freeze_verifier_global(value)
    if operation == "VERIFIER" and (
        any(name.startswith("_SEALED_REFERENCE_") for name in namespace)
        or "_SEALED_RENAME_EXCLUSIVE" in namespace
    ):
        raise LauncherError("sealed verifier acquired a forbidden live capability")
    return namespace


def _pure_verifier_target(namespace: Mapping[str, Any]) -> types.FunctionType:
    target = namespace.get("verify_sealed_bytes")
    if type(target) is not types.FunctionType or target.__closure__ is not None:
        raise LauncherError("pinned verifier lacks a closure-free verify_sealed_bytes")
    target_code = target.__code__
    if target_code.co_argcount != 5 or target_code.co_posonlyargcount != 0 \
            or target_code.co_kwonlyargcount != 0 \
            or target_code.co_flags & 0x0C \
            or target_code.co_varnames[:5] != (
                "request_bytes",
                "artifact_blobs",
                "oracle_release_blobs",
                "reference_result_bytes",
                "reference_attestation",
            ):
        raise LauncherError("pure verifier entrypoint signature is not exact")
    sealed_names = {
        name
        for name in namespace
        if name.startswith("_SEALED_")
    }
    expected_sealed_names = {
        "_SEALED_RUNTIME_CODE_BYTES",
        "_SEALED_RUNTIME_CODE_SHA256",
        "_SEALED_SCHEMA_BYTES",
        "_SEALED_SCHEMA_SHA256",
        "_SEALED_LAUNCHER_SHA256",
        "_SEALED_RUNTIME",
    }
    if sealed_names != expected_sealed_names \
            or namespace.get("_SEALED_RUNTIME") is not True:
        raise LauncherError("sealed verifier global injection set is not exact")
    runtime_builtins = namespace.get("__builtins__")
    if type(runtime_builtins) is not dict \
            or set(runtime_builtins) != PURE_VERIFIER_BUILTINS:
        raise LauncherError("sealed verifier builtin capability set is not exact")
    forbidden_names = frozenset({
        "__loader__",
        "__subclasses__",
        "_getframe",
        "currentframe",
        "delattr",
        "f_back",
        "f_builtins",
        "f_code",
        "f_globals",
        "f_locals",
        "frame",
        "getattr",
        "gi_frame",
        "cr_frame",
        "builtins",
        "compile",
        "eval",
        "exec",
        "fcntl",
        "globals",
        "importlib",
        "locals",
        "open",
        "os",
        "pathlib",
        "setattr",
        "socket",
        "stat",
        "subprocess",
        "sys",
        "tb_frame",
        "traceback",
        "vars",
    })
    allowed_module_names = frozenset({
        "base64",
        "collections",
        "fractions",
        "hashlib",
        "json",
        "re",
        "typing",
    })
    if any(
        type(value) is types.ModuleType and value.__name__ not in allowed_module_names
        for value in namespace.values()
    ):
        raise LauncherError("sealed verifier retained a forbidden module capability")
    if any(type(value) is types.MethodType for value in namespace.values()) \
            or any(
                type(value) is types.FunctionType and value.__closure__ is not None
                for value in namespace.values()
            ):
        raise LauncherError("sealed verifier retained a closure capability")
    pending = [target]
    visited: set[int] = set()
    while pending:
        function = pending.pop()
        if id(function) in visited:
            continue
        visited.add(id(function))
        code_pending = [function.__code__]
        names: set[str] = set()
        string_constants: set[str] = set()
        while code_pending:
            code_object = code_pending.pop()
            names.update(code_object.co_names)
            for instruction in dis.get_instructions(code_object):
                if instruction.opname in {
                    "DELETE_ATTR",
                    "DELETE_GLOBAL",
                    "IMPORT_FROM",
                    "IMPORT_NAME",
                    "IMPORT_STAR",
                    "LOAD_BUILD_CLASS",
                    "STORE_ATTR",
                    "STORE_GLOBAL",
                }:
                    raise LauncherError(
                        "pure verifier call graph contains dynamic code capability"
                    )
            for constant in code_object.co_consts:
                if type(constant) is CodeType:
                    code_pending.append(constant)
                elif type(constant) is str:
                    string_constants.add(constant)
        reflective_names = {
            name
            for name in names | string_constants
            if name.startswith("__") and name.endswith("__")
        }
        if names & forbidden_names or string_constants & forbidden_names \
                or reflective_names:
            raise LauncherError(
                "pure verifier call graph references a forbidden capability"
            )
        for name in names:
            dependency = namespace.get(name)
            if type(dependency) is types.FunctionType:
                if dependency.__closure__ is not None:
                    raise LauncherError("pure verifier call graph contains a closure")
                pending.append(dependency)
    return target


def _bootstrap_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LauncherError("bootstrap attestation has a duplicate key")
        result[key] = value
    return result


def _bootstrap_int(token: str) -> int:
    if token == "-0":
        raise LauncherError("bootstrap attestation negative zero is forbidden")
    return int(token)


def _bootstrap_float(_: str) -> Any:
    raise LauncherError("bootstrap attestation floating point is forbidden")


def _assert_bootstrap_value(value: Any) -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is list:
        for item in value:
            _assert_bootstrap_value(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise LauncherError("bootstrap attestation key must be text")
            _assert_bootstrap_value(item)
        return
    raise LauncherError("bootstrap attestation contains a non-canonical type")


def _strict_bootstrap_attestation(data: bytes) -> dict[str, Any]:
    if type(data) is not bytes or not data.endswith(b"\n") or data.endswith(b"\n\n"):
        raise LauncherError("canonical bootstrap attestation bytes are absent")
    try:
        value = json.loads(
            data[:-1].decode("utf-8"),
            object_pairs_hook=_bootstrap_pairs,
            parse_int=_bootstrap_int,
            parse_float=_bootstrap_float,
            parse_constant=_bootstrap_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LauncherError("bootstrap attestation JSON is invalid") from error
    if type(value) is not dict:
        raise LauncherError("bootstrap attestation must be an object")
    _assert_bootstrap_value(value)
    if _canonical_bytes(value) + b"\n" != data:
        raise LauncherError("bootstrap attestation is not canonical JSON")
    return value


def _interpreter_flags() -> dict[str, int]:
    return {
        "isolated": int(sys.flags.isolated),
        "no_site": int(sys.flags.no_site),
        "ignore_environment": int(sys.flags.ignore_environment),
        "safe_path": int(sys.flags.safe_path),
        "no_user_site": int(sys.flags.no_user_site),
        "dont_write_bytecode": int(sys.flags.dont_write_bytecode),
    }


def _minimal_stdlib_sys_path() -> list[str]:
    paths = list(sys.path)
    stdlib_parent = os.path.realpath(os.path.join(sys.base_prefix, "lib"))
    cwd = os.path.realpath(os.getcwd())
    if not 1 <= len(paths) <= 4:
        raise LauncherError("launcher sys.path entry count is not minimal")
    for entry in paths:
        if type(entry) is not str or not entry:
            raise LauncherError("launcher sys.path has an empty or non-text entry")
        lowered = entry.lower()
        real = os.path.realpath(entry)
        if "site-packages" in lowered or "dist-packages" in lowered or real == cwd:
            raise LauncherError("launcher sys.path contains a caller/site entry")
        if real != stdlib_parent and not real.startswith(stdlib_parent + os.sep):
            raise LauncherError("launcher sys.path escapes the interpreter stdlib")
    return paths


def _interpreter_evidence() -> tuple[dict[str, Any], str]:
    executable_path = os.path.realpath(sys.executable)
    descriptor = os.open(
        executable_path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        executable_bytes = _snapshot_readonly_regular_fd(
            descriptor,
            "interpreter executable",
            maximum_bytes=MAX_INTERPRETER_ARTIFACT_BYTES,
        )
    finally:
        os.close(descriptor)
    identity = {
        "implementation": sys.implementation.name,
        "cache_tag": sys.implementation.cache_tag,
        "version": sys.version,
        "hexversion": sys.hexversion,
        "executable_realpath": executable_path,
    }
    return identity, _sha256(executable_bytes)


def _validate_bootstrap(descriptor: int) -> tuple[bytes, dict[str, Any]]:
    """Re-derive the outer bootstrap receipt without reopening ``__file__``."""

    if __file__ != "<sealed-fd-launcher-v2>" or sys.argv[0] != "<sealed-fd-launcher-v2>":
        raise LauncherError("launcher was not compiled by the fixed outer bootstrap")
    if type(_INJECTED_LAUNCHER_CODE_BYTES) is not bytes:
        raise LauncherError("outer bootstrap launcher bytes are absent")
    attestation = _strict_bootstrap_attestation(_INJECTED_BOOTSTRAP_ATTESTATION)
    expected_keys = {
        "schema_version",
        "bootstrap_id",
        "caller_asserted_bootstrap_source_sha256",
        "bootstrap_provenance",
        "pre_audit_capability_absence_proven",
        "outer_launch_provenance_status",
        "launcher_code_sha256",
        "interpreter_executable_sha256",
        "interpreter_identity",
        "interpreter_identity_sha256",
        "interpreter_flags",
        "interpreter_flags_sha256",
        "sys_path",
        "sys_path_sha256",
        "bootstrap_attestation_sha256",
    }
    if set(attestation) != expected_keys:
        raise LauncherError("bootstrap attestation schema is not exact")
    if attestation["schema_version"] != 1 \
            or attestation["bootstrap_id"] != "PAPER_RESEARCH_FIXED_FD_BOOTSTRAP_V1":
        raise LauncherError("bootstrap identity mismatch")
    unsigned = dict(attestation)
    actual_attestation_sha = unsigned.pop("bootstrap_attestation_sha256")
    if type(actual_attestation_sha) is not str \
            or actual_attestation_sha != _sha256(_canonical_bytes(unsigned)):
        raise LauncherError("bootstrap attestation self-hash mismatch")
    if attestation["caller_asserted_bootstrap_source_sha256"] != PINNED_BOOTSTRAP_SOURCE_SHA256 \
            or attestation["bootstrap_provenance"] != "PYTHON_C_NOT_SELF_AUTHENTICATING" \
            or type(attestation["pre_audit_capability_absence_proven"]) is not bool \
            or attestation["pre_audit_capability_absence_proven"] is not False \
            or attestation["outer_launch_provenance_status"] \
                != "CALLER_ASSERTED_PENDING_EXTERNAL_ANCHOR":
        raise LauncherError("outer bootstrap source is not the reviewed fixed source")
    launcher_bytes = _snapshot_readonly_regular_fd(
        descriptor,
        "launcher code",
        maximum_bytes=MAX_RUNTIME_ARTIFACT_BYTES,
    )
    if launcher_bytes != _INJECTED_LAUNCHER_CODE_BYTES \
            or attestation["launcher_code_sha256"] != _sha256(launcher_bytes):
        raise LauncherError("compiled launcher bytes differ from the inherited FD snapshot")
    required_flags = {
        "isolated": 1,
        "no_site": 1,
        "ignore_environment": 1,
        "safe_path": 1,
        "no_user_site": 1,
        "dont_write_bytecode": 1,
    }
    flags = _interpreter_flags()
    if flags != required_flags or attestation["interpreter_flags"] != flags \
            or attestation["interpreter_flags_sha256"] != _sha256(_canonical_bytes(flags)):
        raise LauncherError("bootstrap interpreter flags are not exactly -I -S -B")
    sys_path = _minimal_stdlib_sys_path()
    if attestation["sys_path"] != sys_path \
            or attestation["sys_path_sha256"] != _sha256(_canonical_bytes(sys_path)):
        raise LauncherError("bootstrap sys.path attestation mismatch")
    interpreter_identity, interpreter_sha = _interpreter_evidence()
    if attestation["interpreter_identity"] != interpreter_identity \
            or attestation["interpreter_identity_sha256"] != _sha256(
                _canonical_bytes(interpreter_identity)
            ) \
            or attestation["interpreter_executable_sha256"] != interpreter_sha:
        raise LauncherError("bootstrap interpreter attestation mismatch")
    return launcher_bytes, attestation


class _FailClosedArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise LauncherError("launcher arguments are invalid")

    def exit(self, status: int = 0, message: str | None = None) -> None:
        raise LauncherError("launcher argument processing attempted to exit")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _FailClosedArgumentParser(prog=LAUNCHER_NAME, add_help=False)
    parser.add_argument("--operation", required=True, choices=OPERATIONS)
    parser.add_argument("--launcher-fd", required=True, type=int)
    parser.add_argument("--request-fd", required=True, type=int)
    parser.add_argument("--input-root-fd", required=True, type=int)
    parser.add_argument("--output-root-fd", required=True, type=int)
    parser.add_argument("--code-fd", required=True, type=int)
    parser.add_argument("--schema-fd", required=True, type=int)
    parser.add_argument("--contract-fd", type=int)
    parser.add_argument("--oracle-code-fd", type=int)
    parser.add_argument("--oracle-contract-fd", type=int)
    parser.add_argument("--oracle-schema-fd", type=int)
    parser.add_argument("--reference-code-fd", type=int)
    parser.add_argument("--reference-contract-fd", type=int)
    return parser.parse_args(argv)


def _reject_irrelevant_descriptors(args: argparse.Namespace) -> None:
    if args.operation == "ORACLE":
        if args.contract_fd is None:
            raise LauncherError("Oracle operation requires --contract-fd")
        if any(value is not None for value in (
            args.oracle_code_fd,
            args.oracle_contract_fd,
            args.oracle_schema_fd,
            args.reference_code_fd,
            args.reference_contract_fd,
        )):
            raise LauncherError("Oracle operation rejects verifier-only release FDs")
        return
    if args.contract_fd is not None:
        raise LauncherError("Verifier operation has no verifier contract FD")
    if any(value is None for value in (
        args.oracle_code_fd,
        args.oracle_contract_fd,
        args.oracle_schema_fd,
        args.reference_code_fd,
        args.reference_contract_fd,
    )):
        raise LauncherError(
            "Verifier operation requires trusted Oracle and reference release FDs"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    _reject_irrelevant_descriptors(args)
    launcher_bytes, bootstrap_attestation = _validate_bootstrap(args.launcher_fd)
    launcher_sha256 = _sha256(launcher_bytes)
    input_root = _validate_root_fd(args.input_root_fd, "input root")
    output_root = _validate_root_fd(args.output_root_fd, "output root")
    if (input_root.st_dev, input_root.st_ino) == (output_root.st_dev, output_root.st_ino):
        raise LauncherError("input and output roots must be distinct directory inodes")
    request_bytes = _snapshot_readonly_regular_fd(
        args.request_fd,
        "request",
        maximum_bytes=MAX_REQUEST_BYTES,
    )
    code_bytes = _snapshot_readonly_regular_fd(
        args.code_fd,
        f"{args.operation.lower()} code",
        maximum_bytes=MAX_RUNTIME_ARTIFACT_BYTES,
    )
    schema_bytes = _snapshot_readonly_regular_fd(
        args.schema_fd,
        f"{args.operation.lower()} schema",
        maximum_bytes=MAX_RUNTIME_ARTIFACT_BYTES,
    )
    _require_pinned_bytes(args.operation, "code", code_bytes)
    _require_pinned_bytes(args.operation, "schema", schema_bytes)
    contract_bytes: bytes | None = None
    oracle_release_blobs: Mapping[str, bytes] | None = None
    reference_code_bytes: bytes | None = None
    reference_contract_bytes: bytes | None = None
    verifier_request: dict[str, Any] | None = None
    artifact_blobs: tuple[tuple[str, bytes], ...] | None = None
    raw_artifacts: Mapping[str, bytes] | None = None
    if args.operation == "ORACLE":
        contract_bytes = _snapshot_readonly_regular_fd(
            args.contract_fd,
            "oracle contract",
            maximum_bytes=MAX_RUNTIME_ARTIFACT_BYTES,
        )
        _require_pinned_bytes("ORACLE", "contract", contract_bytes)
    else:
        mutable_oracle_release: dict[str, bytes] = {}
        for role, descriptor in (
            ("code", args.oracle_code_fd),
            ("contract", args.oracle_contract_fd),
            ("schema", args.oracle_schema_fd),
        ):
            value = _snapshot_readonly_regular_fd(
                descriptor,
                f"trusted Oracle {role}",
                maximum_bytes=MAX_RUNTIME_ARTIFACT_BYTES,
            )
            _require_pinned_bytes("ORACLE", role, value)
            mutable_oracle_release[f"{role}_bytes"] = value
        oracle_release_blobs = types.MappingProxyType(mutable_oracle_release)
        reference_code_bytes = _snapshot_readonly_regular_fd(
            args.reference_code_fd,
            "reference code",
            maximum_bytes=MAX_RUNTIME_ARTIFACT_BYTES,
        )
        reference_contract_bytes = _snapshot_readonly_regular_fd(
            args.reference_contract_fd,
            "reference contract",
            maximum_bytes=MAX_RUNTIME_ARTIFACT_BYTES,
        )
        _require_pinned_bytes("REFERENCE", "code", reference_code_bytes)
        _require_pinned_bytes("REFERENCE", "contract", reference_contract_bytes)
        verifier_request, artifact_blobs, raw_artifacts = (
            _snapshot_verifier_request_artifacts(
                request_bytes,
                args.input_root_fd,
                oracle_release_blobs=oracle_release_blobs,
                reference_code_bytes=reference_code_bytes,
                reference_contract_bytes=reference_contract_bytes,
            )
        )

    rename_exclusive = _build_native_rename_exclusive()
    audit_boundary = _AuditBoundary()
    sys.addaudithook(audit_boundary.hook)
    if args.operation == "ORACLE":
        audit_boundary.begin_runtime_phase()
        try:
            namespace = _compile_and_load(
                "ORACLE",
                code_bytes,
                schema_bytes,
                launcher_sha256,
                rename_exclusive,
                audit_boundary,
                contract_bytes=contract_bytes,
            )
            target = namespace.get("execute_from_fds")
            if not callable(target):
                raise LauncherError("pinned Oracle lacks execute_from_fds")
            result = target(
                request_bytes,
                input_root_fd=args.input_root_fd,
                output_root_fd=args.output_root_fd,
                code_fd=args.code_fd,
                contract_fd=args.contract_fd,
                schema_fd=args.schema_fd,
            )
        finally:
            audit_boundary.end_runtime_phase()
        _validate_runtime_output_acl_tree(
            args.output_root_fd,
            result["output_directory"],
            frozenset({
                "intent.json",
                "oracle_ledger.jsonl",
                "oracle_manifest.json",
                "COMMIT.json",
            }),
            "Oracle output",
        )
        content_binding = result["manifest"]["oracle_release_content_binding"]
        provenance_scope = result["manifest"]["oracle_execution_provenance_scope"]
        evidence_eligible = result["manifest"]["release_evidence_eligible"]
        output_directory = result["output_directory"]
        reference_result_sha256: str | None = None
    else:
        if reference_code_bytes is None or reference_contract_bytes is None \
                or oracle_release_blobs is None or verifier_request is None \
                or artifact_blobs is None or raw_artifacts is None:
            raise LauncherError("verifier immutable snapshot state is incomplete")
        reference_result_bytes, reference_result_sha256 = _run_reference_snapshot(
            reference_code_bytes,
            raw_artifacts,
            audit_boundary,
        )
        reference_attestation = tuple(sorted({
            "reference_code_sha256": PINNED_RELEASES["REFERENCE"][
                "code_sha256"
            ],
            "reference_contract_sha256": PINNED_RELEASES["REFERENCE"][
                "contract_sha256"
            ],
            "reference_result_sha256": reference_result_sha256,
        }.items()))
        oracle_release_tuple = tuple(sorted(oracle_release_blobs.items()))
        del raw_artifacts
        del reference_code_bytes
        del reference_contract_bytes
        audit_boundary.begin_runtime_phase()
        try:
            namespace = _compile_and_load(
                "VERIFIER",
                code_bytes,
                schema_bytes,
                launcher_sha256,
                None,
                audit_boundary,
            )
            target = _pure_verifier_target(namespace)
            try:
                returned = target(
                    request_bytes,
                    artifact_blobs,
                    oracle_release_tuple,
                    reference_result_bytes,
                    reference_attestation,
                )
            finally:
                namespace.clear()
                del target
        finally:
            audit_boundary.end_runtime_phase()
        receipt_bytes, commit_bytes, receipt = _validate_verifier_return(
            returned,
            request_bytes=request_bytes,
            artifact_blobs=artifact_blobs,
            reference_result_bytes=reference_result_bytes,
            reference_result_sha256=reference_result_sha256,
            launcher_sha256=launcher_sha256,
        )
        _publish_verifier_bytes(
            args.output_root_fd,
            verifier_request["output_directory"],
            receipt_bytes,
            commit_bytes,
            rename_exclusive,
        )
        _validate_runtime_output_acl_tree(
            args.output_root_fd,
            verifier_request["output_directory"],
            frozenset({"verifier_receipt.json", "COMMIT.json"}),
            "verifier output",
        )
        content_binding = receipt["verifier_release_content_binding"]
        provenance_scope = receipt["verifier_execution_provenance_scope"]
        evidence_eligible = receipt["release_evidence_eligible"]
        output_directory = verifier_request["output_directory"]
    if content_binding != {
        **({"code_sha256": PINNED_RELEASES[args.operation]["code_sha256"]}),
        **({"contract_sha256": PINNED_RELEASES["ORACLE"]["contract_sha256"]}
           if args.operation == "ORACLE" else {}),
        **({
            "reference_code_sha256": PINNED_RELEASES["REFERENCE"]["code_sha256"],
            "reference_contract_sha256": PINNED_RELEASES["REFERENCE"][
                "contract_sha256"
            ],
            "reference_result_sha256": reference_result_sha256,
        } if args.operation == "VERIFIER" else {}),
        "schema_sha256": PINNED_RELEASES[args.operation]["schema_sha256"],
        "launcher_sha256": launcher_sha256,
        "snapshot_mode": "SEALED_FD_COMPILE_EXEC_V2",
    }:
        raise LauncherError("sealed runtime emitted an unexpected release content binding")
    if provenance_scope != (
        "LOCAL_CALLER_ASSERTED_CONTENT_BINDING_NOT_EXECUTION_ATTESTATION_"
        "NOT_EXTERNALLY_ANCHORED"
    ):
        raise LauncherError("sealed runtime overclaimed execution provenance")
    if evidence_eligible is not False:
        raise LauncherError("sealed runtime overclaimed release evidence eligibility")
    _validate_root_fd(args.input_root_fd, "input root final fence")
    _validate_root_fd(args.output_root_fd, "output root final fence")
    return {
        "ok": True,
        "operation": args.operation,
        "output_directory": output_directory,
        "launcher_sha256": launcher_sha256,
        "bootstrap_attestation_sha256": bootstrap_attestation[
            "bootstrap_attestation_sha256"
        ],
        "caller_asserted_bootstrap_source_sha256": bootstrap_attestation[
            "caller_asserted_bootstrap_source_sha256"
        ],
        "bootstrap_provenance": bootstrap_attestation["bootstrap_provenance"],
        "pre_audit_capability_absence_proven": bootstrap_attestation[
            "pre_audit_capability_absence_proven"
        ],
        "interpreter_executable_sha256": bootstrap_attestation[
            "interpreter_executable_sha256"
        ],
        "interpreter_identity_sha256": bootstrap_attestation[
            "interpreter_identity_sha256"
        ],
        "interpreter_flags_sha256": bootstrap_attestation[
            "interpreter_flags_sha256"
        ],
        "sys_path_sha256": bootstrap_attestation["sys_path_sha256"],
        "release_evidence_eligible": False,
        "local_reproducible_only": True,
        "outer_launch_provenance_status": "CALLER_ASSERTED_PENDING_EXTERNAL_ANCHOR",
        "runtime_environment_scope": (
            "HOST_LOCAL_NON_HERMETIC_STDLIB_NOT_BYTE_PINNED"
        ),
        "snapshot_mode": "SEALED_FD_COMPILE_EXEC_V2",
    }


def main(argv: list[str] | None = None) -> int:
    try:
        result = run(_parse_args(argv))
    except BaseException as error:
        failure = {
            "ok": False,
            "error_code": "SEALED_FD_LAUNCHER_FAIL_CLOSED",
            "error_sha256": _sha256(str(error).encode("utf-8")),
        }
        try:
            _write_all(1, _canonical_bytes(failure) + b"\n")
        except BaseException:
            return 3
        return 2
    try:
        _write_all(1, _canonical_bytes(result) + b"\n")
    except BaseException:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
