"""Pure evidence contracts for one-shot DOJO AI model execution.

The actual model subprocess deliberately lives under ``tools/`` so production
QuantRabbit code never acquires a model/API execution path.  This module only
builds and verifies immutable, bounded request/receipt artifacts.  Provider
identity remains explicitly unverified: the receipt proves the local procedure
and captured CLI event stream, not a provider-signed model identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from quant_rabbit.dojo_ai_discretion import (
    FIXED_RESPONSE_SCHEMA,
    canonical_sha256,
)


EXECUTION_REQUEST_CONTRACT = "QR_DOJO_AI_MODEL_EXECUTION_REQUEST_V1"
EXECUTION_RECEIPT_CONTRACT = "QR_DOJO_AI_MODEL_EXECUTION_RECEIPT_V1"
MAX_STDOUT_BYTES = 524_288
MAX_STDERR_BYTES = 65_536
MAX_EVENT_COUNT = 512
_HEX64 = re.compile(r"[0-9a-f]{64}")
_ALLOWED_EVENT_TYPES = frozenset(
    {
        "thread.started",
        "turn.started",
        "item.started",
        "item.updated",
        "item.completed",
        "turn.completed",
        "error",
    }
)
_ALLOWED_ITEM_TYPES = frozenset({"reasoning", "agent_message"})
_DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "apps",
    "plugins",
    "tool_call_mcp_elicitation",
    "skill_mcp_dependency_install",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "in_app_browser",
    "computer_use",
    "image_generation",
    "multi_agent",
    "skill_search",
    "auth_elicitation",
)
_ENVIRONMENT_KEYS = (
    "CODEX_HOME",
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "TMPDIR",
)
_API_KEY_NAMES = (
    "OPENAI_API_KEY",
    "QR_OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)


class DojoAIExecutionError(ValueError):
    """The one-shot model execution evidence is malformed or inconsistent."""


def render_model_prompt(
    prompt_text: str, packet: Mapping[str, Any]
) -> str:
    """Render the exact bounded prompt sent to a fresh no-tool context."""

    if not isinstance(prompt_text, str) or not prompt_text or len(prompt_text) > 8_000:
        raise DojoAIExecutionError("execution prompt text is invalid")
    packet_value = _mapping(packet, "execution packet")
    packet_bytes = _canonical_bytes(packet_value)
    if len(packet_bytes) > 1_500_000:
        raise DojoAIExecutionError("execution packet is too large")
    instruction = (
        "\n\nEvaluate only the sealed, date-anonymized packet below. "
        "Return exactly one JSON object matching the supplied output schema. "
        "Do not request or use tools, files, network access, prior conversation, "
        "memory, repositories, dates, or an answer key.\nPACKET_JSON:\n"
    )
    return prompt_text + instruction + packet_bytes.decode("utf-8")


def build_exact_output_schema(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Narrow the preregistered response schema to this packet identity."""

    value = _snapshot(FIXED_RESPONSE_SCHEMA)
    packet_value = _mapping(packet, "execution packet")
    observations = packet_value.get("observations")
    if not isinstance(observations, list) or not observations:
        raise DojoAIExecutionError("execution packet observations are invalid")
    refs = [row.get("id") for row in observations if isinstance(row, Mapping)]
    if len(refs) != len(observations) or any(not isinstance(item, str) for item in refs):
        raise DojoAIExecutionError("execution packet evidence references are invalid")
    value["properties"]["trial_id"] = {"const": packet_value.get("trial_id")}
    value["properties"]["pair"] = {"const": packet_value.get("pair")}
    value["properties"]["evidence_refs"]["items"] = {"enum": refs}
    return value


def build_execution_request(
    *,
    precommit_sha256: str,
    day_seal_sha256: str,
    cell: Mapping[str, Any],
    cli_binary_path: str,
    cli_binary_sha256: str,
    cli_version: str,
    auth_mode_probe_sha256: str,
    runtime_root_identity_sha256: str,
    created_at_utc: datetime,
) -> dict[str, Any]:
    """Freeze the only permitted launch before the subprocess starts."""

    cell_value = _mapping(cell, "execution cell")
    packet = _mapping(cell_value.get("packet"), "execution packet")
    prompt_lock = _mapping(cell_value.get("prompt_lock"), "execution prompt lock")
    model = _mapping(cell_value.get("model_manifest"), "execution model manifest")
    capability = _mapping(
        cell_value.get("capability_manifest"), "execution capability manifest"
    )
    request = _mapping(cell_value.get("request_receipt"), "execution request receipt")
    rendered_prompt = render_model_prompt(str(prompt_lock.get("prompt_text")), packet)
    output_schema = build_exact_output_schema(packet)
    created = _utc(created_at_utc, "created_at_utc")
    deadline = _parse_utc(request.get("response_deadline_utc"), "response deadline")
    if created > deadline:
        raise DojoAIExecutionError("execution request is after its response deadline")
    if request.get("answer_key_present") is not False:
        raise DojoAIExecutionError("execution request does not prove answer-key absence")
    if capability.get("declared_answer_key_physically_absent") is not True:
        raise DojoAIExecutionError("execution capability does not exclude answer key")
    tools = capability.get("tool_access", capability.get("declared_tools"))
    if tools != []:
        raise DojoAIExecutionError("execution capability allows tools")
    body = {
        "contract": EXECUTION_REQUEST_CONTRACT,
        "schema_version": 1,
        "state": "LAUNCH_INTENT_PERSISTED",
        "created_at_utc": _iso(created),
        "response_deadline_utc": request["response_deadline_utc"],
        "precommit_sha256": _sha(precommit_sha256, "precommit_sha256"),
        "day_seal_sha256": _sha(day_seal_sha256, "day_seal_sha256"),
        "cell_id": _text(request.get("cell_id"), "cell_id", 200),
        "variant_id": _text(request.get("variant_id"), "variant_id", 200),
        "context_id": _text(request.get("context_id"), "context_id", 200),
        "request_receipt_sha256": _sha(
            request.get("request_receipt_sha256"), "request_receipt_sha256"
        ),
        "packet_sha256": _sha(packet.get("packet_sha256"), "packet_sha256"),
        "prompt_lock_sha256": _sha(
            prompt_lock.get("prompt_lock_sha256"), "prompt_lock_sha256"
        ),
        "prompt_sha256": _sha(prompt_lock.get("prompt_sha256"), "prompt_sha256"),
        "model_sha256": _sha(model.get("model_sha256"), "model_sha256"),
        "capability_manifest_sha256": _sha(
            capability.get("capability_manifest_sha256"),
            "capability_manifest_sha256",
        ),
        "requested_model": _text(model.get("model_name"), "model_name", 200),
        "requested_model_version": _text(
            model.get("model_version"), "model_version", 200
        ),
        "requested_model_lineage": _text(
            model.get("declared_model_lineage"), "model lineage", 200
        ),
        "reasoning_effort": _text(
            model.get("reasoning_effort"), "reasoning effort", 80
        ),
        "stdin_envelope": rendered_prompt,
        "stdin_envelope_sha256": hashlib.sha256(
            rendered_prompt.encode("utf-8")
        ).hexdigest(),
        "output_schema": output_schema,
        "output_schema_sha256": canonical_sha256(output_schema),
        "runtime": {
            "cli_binary_path": _text(cli_binary_path, "CLI binary path", 1_000),
            "cli_binary_sha256": _sha(cli_binary_sha256, "CLI binary sha256"),
            "cli_version": _text(cli_version, "CLI version", 200),
            "auth_mode": "CHATGPT_LOGIN_STATUS_VERIFIED_LOCALLY",
            "auth_mode_probe_sha256": _sha(
                auth_mode_probe_sha256, "auth mode probe sha256"
            ),
            "runtime_root_identity_sha256": _sha(
                runtime_root_identity_sha256, "runtime root identity sha256"
            ),
            "fresh_empty_working_directory": True,
            "fresh_task_scoped_home": True,
            "fresh_task_scoped_codex_home": True,
            "auth_material_role": "READ_ONLY_AUTH_JSON_SYMLINK_ONLY",
            "auth_material_captured": False,
            "environment_policy": "EMPTY_THEN_EXACT_ALLOWLIST",
            "environment_keys": list(_ENVIRONMENT_KEYS),
            "forbidden_api_key_names": list(_API_KEY_NAMES),
            "forbidden_api_keys_absent": True,
            "ephemeral": True,
            "ignore_user_config": True,
            "ignore_rules": True,
            "skip_git_repo_check": True,
            "sandbox": "read-only",
            "disabled_features": list(_DISABLED_FEATURES),
            "web_search": "disabled",
            "output_schema_enforced": True,
            "jsonl_capture_required": True,
            "one_turn_only": True,
            "retry_allowed": False,
            "response_selection_allowed": False,
            "full_provider_request_observable": False,
            "provider_base_prompt_attested": False,
            "required_launch_boundary": "EXTERNAL_NON_CODEX_PARENT",
        },
        "answer_key_present": False,
        "repository_mounted": False,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "provider_execution_attestation_present": False,
        "provider_identity_status": "REQUESTED_MODEL_CLI_REPORTED_UNVERIFIED",
    }
    return _seal(body, "execution_request_sha256")


def validate_execution_request(value: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _mapping(value, "execution request")
    _validate_seal(artifact, "execution_request_sha256")
    if artifact.get("contract") != EXECUTION_REQUEST_CONTRACT:
        raise DojoAIExecutionError("execution request contract is invalid")
    if artifact.get("schema_version") != 1 or artifact.get("state") != "LAUNCH_INTENT_PERSISTED":
        raise DojoAIExecutionError("execution request identity is invalid")
    _parse_utc(artifact.get("created_at_utc"), "created_at_utc")
    _parse_utc(artifact.get("response_deadline_utc"), "response deadline")
    for key in (
        "precommit_sha256",
        "day_seal_sha256",
        "request_receipt_sha256",
        "packet_sha256",
        "prompt_lock_sha256",
        "prompt_sha256",
        "model_sha256",
        "capability_manifest_sha256",
        "stdin_envelope_sha256",
        "output_schema_sha256",
    ):
        _sha(artifact.get(key), key)
    prompt = artifact.get("stdin_envelope")
    if not isinstance(prompt, str) or hashlib.sha256(prompt.encode("utf-8")).hexdigest() != artifact.get("stdin_envelope_sha256"):
        raise DojoAIExecutionError("execution stdin envelope drifted")
    schema = _mapping(artifact.get("output_schema"), "execution output schema")
    if canonical_sha256(schema) != artifact.get("output_schema_sha256"):
        raise DojoAIExecutionError("execution output schema drifted")
    runtime = _mapping(artifact.get("runtime"), "execution runtime")
    expected_runtime_flags = {
        "auth_mode": "CHATGPT_LOGIN_STATUS_VERIFIED_LOCALLY",
        "fresh_empty_working_directory": True,
        "fresh_task_scoped_home": True,
        "fresh_task_scoped_codex_home": True,
        "auth_material_role": "READ_ONLY_AUTH_JSON_SYMLINK_ONLY",
        "auth_material_captured": False,
        "environment_policy": "EMPTY_THEN_EXACT_ALLOWLIST",
        "environment_keys": list(_ENVIRONMENT_KEYS),
        "forbidden_api_key_names": list(_API_KEY_NAMES),
        "forbidden_api_keys_absent": True,
        "ephemeral": True,
        "ignore_user_config": True,
        "ignore_rules": True,
        "skip_git_repo_check": True,
        "sandbox": "read-only",
        "disabled_features": list(_DISABLED_FEATURES),
        "web_search": "disabled",
        "output_schema_enforced": True,
        "jsonl_capture_required": True,
        "one_turn_only": True,
        "retry_allowed": False,
        "response_selection_allowed": False,
        "full_provider_request_observable": False,
        "provider_base_prompt_attested": False,
        "required_launch_boundary": "EXTERNAL_NON_CODEX_PARENT",
    }
    for key, expected in expected_runtime_flags.items():
        if runtime.get(key) != expected:
            raise DojoAIExecutionError(f"execution runtime policy drifted: {key}")
    for key in ("cli_binary_sha256", "auth_mode_probe_sha256", "runtime_root_identity_sha256"):
        _sha(runtime.get(key), key)
    if (
        artifact.get("answer_key_present") is not False
        or artifact.get("repository_mounted") is not False
        or artifact.get("live_permission") is not False
        or artifact.get("broker_mutation_allowed") is not False
        or artifact.get("provider_execution_attestation_present") is not False
        or artifact.get("provider_identity_status")
        != "REQUESTED_MODEL_CLI_REPORTED_UNVERIFIED"
    ):
        raise DojoAIExecutionError("execution request authority boundary drifted")
    return _snapshot(artifact)


def build_execution_receipt(
    request: Mapping[str, Any],
    *,
    raw_stdout_jsonl: bytes,
    raw_stderr: bytes,
    output_last_message: bytes,
    returncode: int | None,
    timed_out: bool,
    started_at_utc: datetime,
    completed_at_utc: datetime,
    launch_started: bool = True,
) -> dict[str, Any]:
    """Capture the first and only launch outcome, including every raw event."""

    launch = validate_execution_request(request)
    if len(raw_stdout_jsonl) > MAX_STDOUT_BYTES or len(raw_stderr) > MAX_STDERR_BYTES:
        raise DojoAIExecutionError("execution output exceeds the evidence bound")
    if len(output_last_message) > MAX_STDOUT_BYTES:
        raise DojoAIExecutionError("execution last message exceeds the evidence bound")
    started = _utc(started_at_utc, "started_at_utc")
    completed = _utc(completed_at_utc, "completed_at_utc")
    if completed < started or started < _parse_utc(launch["created_at_utc"], "request created"):
        raise DojoAIExecutionError("execution receipt chronology is invalid")
    stdout_text = _decode_utf8(raw_stdout_jsonl, "stdout JSONL")
    stderr_text = _decode_utf8(raw_stderr, "stderr")
    output_text = _decode_utf8(output_last_message, "last message")
    events, parse_error = _parse_jsonl(stdout_text)
    summary = _summarize_events(events)
    response: dict[str, Any] | None = None
    response_parse_error: str | None = None
    if output_text.strip():
        try:
            parsed = _strict_json_text(output_text)
            response = _mapping(parsed, "execution response")
        except DojoAIExecutionError as exc:
            response_parse_error = str(exc)
    agent_messages = summary["agent_messages"]
    message_matches = (
        len(agent_messages) == 1
        and output_text == agent_messages[0]
    )
    if not isinstance(launch_started, bool):
        raise DojoAIExecutionError("execution launch_started must be boolean")
    success = (
        launch_started
        and returncode == 0
        and timed_out is False
        and parse_error is None
        and response_parse_error is None
        and response is not None
        and summary["thread_count"] == 1
        and summary["turn_started_count"] == 1
        and summary["turn_completed_count"] == 1
        and summary["forbidden_event_types"] == []
        and summary["unknown_event_types"] == []
        and message_matches
    )
    failure_code = None
    if not success:
        failure_code = _failure_code(
            launch_started=launch_started,
            timed_out=timed_out,
            returncode=returncode,
            parse_error=parse_error,
            response_parse_error=response_parse_error,
            summary=summary,
            message_matches=message_matches,
        )
        response = None
    body = {
        "contract": EXECUTION_RECEIPT_CONTRACT,
        "schema_version": 1,
        "state": "EXECUTION_SUCCEEDED" if success else "EXECUTION_FAILED",
        "started_at_utc": _iso(started),
        "completed_at_utc": _iso(completed),
        "execution_request": launch,
        "execution_request_sha256": launch["execution_request_sha256"],
        "returncode": returncode,
        "launch_started": launch_started,
        "timed_out": timed_out,
        "raw_stdout_jsonl": stdout_text,
        "raw_stdout_sha256": hashlib.sha256(raw_stdout_jsonl).hexdigest(),
        "raw_stderr": stderr_text,
        "raw_stderr_sha256": hashlib.sha256(raw_stderr).hexdigest(),
        "output_last_message": output_text,
        "output_last_message_sha256": hashlib.sha256(output_last_message).hexdigest(),
        "event_count": len(events),
        "thread_id": summary["thread_ids"][0] if summary["thread_count"] == 1 else None,
        "thread_count": summary["thread_count"],
        "turn_started_count": summary["turn_started_count"],
        "turn_completed_count": summary["turn_completed_count"],
        "agent_message_count": len(agent_messages),
        "forbidden_event_types": summary["forbidden_event_types"],
        "unknown_event_types": summary["unknown_event_types"],
        "jsonl_parse_error": parse_error,
        "response_parse_error": response_parse_error,
        "output_matches_only_agent_message": message_matches,
        "response": response,
        "response_sha256": canonical_sha256(response) if response is not None else None,
        "failure_code": failure_code,
        "first_launch_only": True,
        "retry_allowed": False,
        "response_selection_allowed": False,
        "answer_key_opened": False,
        "local_execution_procedure_verified": success,
        "provider_execution_attestation_present": False,
        "provider_identity_status": "REQUESTED_MODEL_CLI_REPORTED_UNVERIFIED",
        "live_permission": False,
        "broker_mutation_allowed": False,
    }
    return _seal(body, "execution_receipt_sha256")


def validate_execution_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _mapping(value, "execution receipt")
    _validate_seal(artifact, "execution_receipt_sha256")
    request = validate_execution_request(
        _mapping(artifact.get("execution_request"), "embedded execution request")
    )
    if artifact.get("execution_request_sha256") != request["execution_request_sha256"]:
        raise DojoAIExecutionError("execution receipt request binding drifted")
    returncode = artifact.get("returncode")
    if returncode is not None and (isinstance(returncode, bool) or not isinstance(returncode, int)):
        raise DojoAIExecutionError("execution receipt returncode is invalid")
    rebuilt = build_execution_receipt(
        request,
        raw_stdout_jsonl=_text_bytes(artifact.get("raw_stdout_jsonl"), "stdout JSONL"),
        raw_stderr=_text_bytes(artifact.get("raw_stderr"), "stderr"),
        output_last_message=_text_bytes(
            artifact.get("output_last_message"), "last message"
        ),
        returncode=returncode,
        timed_out=artifact.get("timed_out"),
        started_at_utc=_parse_utc(artifact.get("started_at_utc"), "started_at_utc"),
        completed_at_utc=_parse_utc(
            artifact.get("completed_at_utc"), "completed_at_utc"
        ),
        launch_started=artifact.get("launch_started"),
    )
    if artifact != rebuilt:
        raise DojoAIExecutionError("execution receipt bytes or semantics drifted")
    return _snapshot(artifact)


def _summarize_events(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    thread_ids: list[str] = []
    agent_messages: list[str] = []
    forbidden: set[str] = set()
    unknown: set[str] = set()
    turn_started = 0
    turn_completed = 0
    for event in events:
        event_type = event.get("type")
        if not isinstance(event_type, str) or event_type not in _ALLOWED_EVENT_TYPES:
            unknown.add(str(event_type))
            continue
        if event_type == "thread.started":
            thread_id = event.get("thread_id")
            if isinstance(thread_id, str) and thread_id:
                thread_ids.append(thread_id)
            else:
                forbidden.add("thread.started:missing_thread_id")
        elif event_type == "turn.started":
            turn_started += 1
        elif event_type == "turn.completed":
            turn_completed += 1
        elif event_type == "error":
            forbidden.add("error")
        elif event_type.startswith("item."):
            item = event.get("item")
            if not isinstance(item, Mapping):
                forbidden.add(f"{event_type}:missing_item")
                continue
            item_type = item.get("type")
            if item_type not in _ALLOWED_ITEM_TYPES:
                forbidden.add(f"item:{item_type}")
                continue
            if event_type == "item.completed" and item_type == "agent_message":
                text = item.get("text")
                if isinstance(text, str):
                    agent_messages.append(text)
                else:
                    forbidden.add("agent_message:missing_text")
    return {
        "thread_ids": thread_ids,
        "thread_count": len(thread_ids),
        "turn_started_count": turn_started,
        "turn_completed_count": turn_completed,
        "agent_messages": agent_messages,
        "forbidden_event_types": sorted(forbidden),
        "unknown_event_types": sorted(unknown),
    }


def _parse_jsonl(value: str) -> tuple[list[dict[str, Any]], str | None]:
    events: list[dict[str, Any]] = []
    if not value:
        return events, "EMPTY_STDOUT_JSONL"
    for ordinal, line in enumerate(value.splitlines(), start=1):
        if not line.strip():
            continue
        if len(events) >= MAX_EVENT_COUNT:
            return events, "EVENT_COUNT_EXCEEDED"
        try:
            parsed = _strict_json_text(line)
            events.append(_mapping(parsed, f"JSONL event {ordinal}"))
        except DojoAIExecutionError:
            return events, f"INVALID_JSONL_EVENT_{ordinal}"
    if not events:
        return events, "EMPTY_STDOUT_JSONL"
    return events, None


def _failure_code(
    *,
    launch_started: bool,
    timed_out: bool,
    returncode: int | None,
    parse_error: str | None,
    response_parse_error: str | None,
    summary: Mapping[str, Any],
    message_matches: bool,
) -> str:
    if not launch_started:
        return "LAUNCH_INTENT_WITHOUT_FIRST_TRANSCRIPT"
    if timed_out:
        return "EXECUTOR_TIMEOUT"
    if summary["forbidden_event_types"]:
        return "CAPABILITY_VIOLATION"
    if parse_error is not None or summary["unknown_event_types"]:
        return "EVENT_STREAM_INVALID"
    if returncode != 0:
        return "EXECUTOR_NONZERO_EXIT"
    if response_parse_error is not None:
        return "SCHEMA_INVALID_RESPONSE"
    if summary["thread_count"] != 1 or summary["turn_started_count"] != 1 or summary["turn_completed_count"] != 1:
        return "FRESH_CONTEXT_OR_TURN_COUNT_INVALID"
    if not message_matches:
        return "FINAL_MESSAGE_DIVERGENCE"
    return "EXECUTION_EVIDENCE_INVALID"


def _strict_json_text(value: str) -> Any:
    def reject_constant(token: str) -> None:
        raise DojoAIExecutionError(f"non-finite JSON number: {token}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise DojoAIExecutionError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    try:
        return json.loads(
            value,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise DojoAIExecutionError("strict JSON parse failed") from exc


def _seal(body: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = _snapshot(body)
    value[key] = canonical_sha256(value)
    return value


def _validate_seal(value: Mapping[str, Any], key: str) -> None:
    digest = value.get(key)
    _sha(digest, key)
    body = {name: item for name, item in value.items() if name != key}
    if canonical_sha256(body) != digest:
        raise DojoAIExecutionError(f"{key} mismatch")


def _snapshot(value: Any) -> Any:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise DojoAIExecutionError("execution artifact is not strict JSON") from exc


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DojoAIExecutionError(f"{label} must be an object")
    return _snapshot(value)


def _text(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise DojoAIExecutionError(f"{label} is invalid")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise DojoAIExecutionError(f"{label} is not sha256")
    return value


def _utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise DojoAIExecutionError(f"{label} must be timezone-aware")
    result = value.astimezone(timezone.utc)
    if not math.isfinite(result.timestamp()):
        raise DojoAIExecutionError(f"{label} is invalid")
    return result


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise DojoAIExecutionError(f"{label} is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DojoAIExecutionError(f"{label} is invalid") from exc
    return _utc(parsed, label)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _decode_utf8(value: bytes, label: str) -> str:
    if not isinstance(value, bytes):
        raise DojoAIExecutionError(f"{label} must be bytes")
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DojoAIExecutionError(f"{label} is not UTF-8") from exc


def _text_bytes(value: Any, label: str) -> bytes:
    if not isinstance(value, str):
        raise DojoAIExecutionError(f"{label} is invalid")
    return value.encode("utf-8")


__all__ = [
    "DojoAIExecutionError",
    "EXECUTION_RECEIPT_CONTRACT",
    "EXECUTION_REQUEST_CONTRACT",
    "MAX_STDERR_BYTES",
    "MAX_STDOUT_BYTES",
    "build_exact_output_schema",
    "build_execution_receipt",
    "build_execution_request",
    "render_model_prompt",
    "validate_execution_receipt",
    "validate_execution_request",
]
