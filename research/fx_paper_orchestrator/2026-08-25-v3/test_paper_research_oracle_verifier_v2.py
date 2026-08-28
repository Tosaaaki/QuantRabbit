from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

import paper_research_oracle_verifier_v2 as verifier
import paper_research_double_entry_reference_v2 as reference
from paper_research_jpy_oracle_golden_v2 import build_golden_payload


ROOT = Path(__file__).resolve().parent
PYTHON = Path("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3")
ORACLE_PATH = ROOT / "paper_research_jpy_oracle_v2.py"
ORACLE_CONTRACT = ROOT / "PAPER_RESEARCH_JPY_ORACLE_CONTRACT_V2.json"
ORACLE_SCHEMA = ROOT / "paper_research_jpy_oracle_schema_v2.json"
VERIFIER_PATH = ROOT / "paper_research_oracle_verifier_v2.py"
VERIFIER_SCHEMA = ROOT / "paper_research_oracle_verifier_schema_v2.json"
REFERENCE_PATH = ROOT / "paper_research_double_entry_reference_v2.py"
REFERENCE_CONTRACT = ROOT / "PAPER_RESEARCH_DOUBLE_ENTRY_REFERENCE_CONTRACT_V2.json"
START_NS = 1_767_225_600_000_000_000
PROVIDER = "ORACLE_FIXTURE"
SEALED_TEST_LAUNCHER_SHA256 = "a" * 64


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def embedded(value: dict, field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return digest(canonical(unsigned))


def seal(value: dict, field: str) -> dict:
    value[field] = embedded(value, field)
    return value


def write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")
    return path


def artifact(root: Path, path: Path, label: str) -> dict:
    data = path.read_bytes()
    return {
        "artifact_id": label,
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": digest(data),
        "size_bytes": len(data),
    }


def registry_payload() -> dict:
    return seal({
        "schema_version": 1,
        "registry_id": "FROZEN_FX_INSTRUMENT_REGISTRY_V1",
        "instruments": {
            "EUR_USD": {"pip_ticks": 10, "price_scale": 100_000},
            "USD_JPY": {"pip_ticks": 1, "price_scale": 100},
        },
    }, "registry_sha256")


def source_rows(
    eur_bid_ticks: tuple[int, ...] | None = None,
    usd_jpy_bid_ticks: tuple[int, ...] | None = None,
) -> list[dict]:
    offsets = (0, 1, 2, 301, 302, 360, 361, 662, 900)
    if eur_bid_ticks is not None and len(eur_bid_ticks) != len(offsets):
        raise ValueError("EUR fixture price path length mismatch")
    if usd_jpy_bid_ticks is not None and len(usd_jpy_bid_ticks) != len(offsets):
        raise ValueError("USD/JPY fixture price path length mismatch")
    rows: list[dict] = []
    for sequence, seconds in enumerate(offsets, 1):
        for instrument, arrival_offset in (("EUR_USD", 100_000_000), ("USD_JPY", 200_000_000)):
            if instrument == "EUR_USD":
                bid = (
                    eur_bid_ticks[sequence - 1]
                    if eur_bid_ticks is not None
                    else 110_000 + sequence * 8
                )
                spread, scale = 12, 100_000
            else:
                bid = (
                    usd_jpy_bid_ticks[sequence - 1]
                    if usd_jpy_bid_ticks is not None
                    else 15_000 + sequence * 3
                )
                spread, scale = 2, 100
            source = START_NS + seconds * 1_000_000_000
            rows.append({
                "schema_version": 1,
                "provider_id": PROVIDER,
                "instrument": instrument,
                "bid_ticks": bid,
                "ask_ticks": bid + spread,
                "tick_scale": scale,
                "source_ts_ns": source,
                "arrival_ts_ns": source + arrival_offset,
                "provider_event_id": f"{instrument}-{sequence}",
                "sequence": sequence,
                "heartbeat": False,
                "quality_flags": [],
            })
    return sorted(rows, key=lambda row: (
        row["arrival_ts_ns"], row["source_ts_ns"], row["provider_id"],
        row["instrument"], row["sequence"],
    ))


def write_source(
    root: Path,
    registry: dict,
    eur_bid_ticks: tuple[int, ...] | None = None,
    usd_jpy_bid_ticks: tuple[int, ...] | None = None,
) -> tuple[Path, Path, list[dict]]:
    rows = source_rows(eur_bid_ticks, usd_jpy_bid_ticks)
    lines = [canonical(row) + b"\n" for row in rows]
    blob_bytes = b"".join(lines)
    blob = root / "inputs" / "source.jsonl"
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(blob_bytes)
    prefix = "0" * 64
    enriched: list[dict] = []
    for row, line in zip(rows, lines):
        event_hash = digest(line)
        prefix = digest(canonical({"previous_hash": prefix, "source_event_sha256": event_hash}))
        enriched.append({**row, "source_event_sha256": event_hash, "source_prefix_root_sha256": prefix})
    policies = []
    for instrument in ("EUR_USD", "USD_JPY"):
        stream = [row for row in rows if row["instrument"] == instrument]
        policies.append({
            "provider_id": PROVIDER,
            "instrument": instrument,
            "sequence_required": True,
            "first_sequence": stream[0]["sequence"],
            "last_sequence": stream[-1]["sequence"],
            "event_count": len(stream),
            "max_source_gap_ns": 400_000_000_000,
            "max_arrival_gap_ns": 400_000_000_000,
        })
    manifest = seal({
        "schema_version": 2,
        "source_bytes_sha256": digest(blob_bytes),
        "source_size_bytes": len(blob_bytes),
        "event_count": len(rows),
        "first_source_ts_ns": min(row["source_ts_ns"] for row in rows),
        "last_source_ts_ns": max(row["source_ts_ns"] for row in rows),
        "provider_allowlist": [PROVIDER],
        "instrument_registry_sha256": registry["registry_sha256"],
        "stream_policies": policies,
        "lossless": True,
    }, "manifest_sha256")
    return blob, write_json(root / "inputs" / "source_manifest.json", manifest), enriched


def proposal_row(
    enriched: list[dict],
    ordinal: int,
    event_index: int,
    direction: int,
    max_age_ns: int = 300_000_000_000,
    notional_jpy_micros: int = 27_000_000_000,
) -> dict:
    event = enriched[event_index]
    available = [row for row in enriched if row["arrival_ts_ns"] <= event["arrival_ts_ns"]]
    return {
        "proposal_ordinal": ordinal,
        "decision_source_ts_ns": event["source_ts_ns"],
        "decision_arrival_ts_ns": event["arrival_ts_ns"],
        "available_at_ns": event["arrival_ts_ns"],
        "decision_source_event_sha256": event["source_event_sha256"],
        "completed_data_watermark_source_ts_ns": max(row["source_ts_ns"] for row in available),
        "completed_data_prefix_root_sha256": available[-1]["source_prefix_root_sha256"],
        "instrument": event["instrument"],
        "direction": direction,
        "notional_jpy_micros": notional_jpy_micros,
        "max_age_ns": max_age_ns,
        "worker_key": "FIXED_DETECTOR",
        "action": "ENTER",
    }


def build_inputs(
    root: Path,
    *,
    proposal_specs: tuple[tuple[int, int], ...] = ((0, 1), (11, -1)),
    proposal_max_age_ns: int = 300_000_000_000,
    proposal_notional_jpy_micros: int = 27_000_000_000,
    initial_equity_jpy_micros: int = 200_000_000_000,
    eur_bid_ticks: tuple[int, ...] | None = None,
    usd_jpy_bid_ticks: tuple[int, ...] | None = None,
    max_currency_notional_jpy_micros: int = 200_000_000_000,
) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    registry = registry_payload()
    registry_path = write_json(root / "inputs" / "instrument_registry.json", registry)
    blob, source_manifest, enriched = write_source(
        root, registry, eur_bid_ticks, usd_jpy_bid_ticks
    )
    proposal = seal({
        "schema_version": 2,
        "candidate_key": "ORACLE-V2-INDEPENDENT-VERIFIER-FIXTURE",
        "provenance": {
            "detector_code_sha256": "1" * 64,
            "detector_policy_sha256": "2" * 64,
            "generator_policy_sha256": "3" * 64,
            "source_acquisition_contract_sha256": "4" * 64,
        },
        "rows": [
            proposal_row(
                enriched,
                ordinal,
                event_index,
                direction,
                proposal_max_age_ns,
                proposal_notional_jpy_micros,
            )
            for ordinal, (event_index, direction) in enumerate(proposal_specs, 1)
        ],
    }, "proposal_sha256")
    execution = seal({
        "schema_version": 2,
        "policy_id": "FROZEN_EXECUTION_POLICY_V2",
        "arms": {
            "RAW_SIGNAL": {"latency_ns": 0, "slippage_micropips_per_side": 0, "commission_ppm_per_side": 0, "financing_ppm_per_day": 0, "raw_mid": True},
            "EXECUTABLE_BASE": {"latency_ns": 500_000_000, "slippage_micropips_per_side": 100_000, "commission_ppm_per_side": 2, "financing_ppm_per_day": 1, "raw_mid": False},
            "ADVERSE_STRESS": {"latency_ns": 1_500_000_000, "slippage_micropips_per_side": 300_000, "commission_ppm_per_side": 6, "financing_ppm_per_day": 3, "raw_mid": False},
        },
        "max_trade_quote_staleness_ns": 400_000_000_000,
    }, "execution_policy_sha256")
    inventory = seal({
        "schema_version": 2,
        "policy_id": "FROZEN_INVENTORY_POLICY_V2",
        "max_gross_notional_jpy_micros": 200_000_000_000,
        "max_currency_notional_jpy_micros": max_currency_notional_jpy_micros,
        "max_open_positions": 4,
        "same_pair_collision": "REJECT_NEW",
        "terminal_liquidation": True,
    }, "inventory_policy_sha256")
    accounting = seal({
        "schema_version": 2,
        "policy_id": "FROZEN_ACCOUNTING_POLICY_V2",
        "jpy_micros_per_yen": 1_000_000,
        "base_microunits_per_unit": 1_000_000,
        "max_conversion_staleness_ns": 400_000_000_000,
        "supported_quote_currencies": ["CAD", "CHF", "JPY", "USD"],
        "asset_conversion_side": "BID",
        "liability_conversion_side": "ASK",
        "positive_cost_rounding": "CEILING",
    }, "accounting_policy_sha256")
    evaluation = seal({
        "schema_version": 2,
        "policy_id": "FROZEN_EVALUATION_POLICY_V2",
        "period_start_ts_ns": START_NS,
        "period_end_ts_ns": START_NS + 901_000_000_000,
        "initial_equity_jpy_micros": initial_equity_jpy_micros,
        "margin_notional_cap_jpy_micros": 200_000_000_000,
        "margin_rate_bps": 500,
        "max_gross_to_equity_bps": 20_000,
        "cvar_tail_bps": 500,
        "cluster_window_ns": 3_600_000_000_000,
        "full_month_ids": [],
        "holdout_state": "UNOPENED",
    }, "evaluation_policy_sha256")
    authority = seal({
        "schema_version": 2,
        "policy_id": "FROZEN_PAPER_AUTHORITY_V1",
        "paper_only": True,
        "live_authority": False,
        "broker_account_access": False,
        "credential_access": False,
        "order_endpoint": False,
        "external_orders": 0,
        "deploy": False,
        "external_config_mutation": False,
    }, "authority_policy_sha256")
    values = {
        "proposal": proposal,
        "execution_policy": execution,
        "inventory_policy": inventory,
        "accounting_policy": accounting,
        "evaluation_policy": evaluation,
        "authority_policy": authority,
    }
    paths = {label: write_json(root / "inputs" / f"{label}.json", value) for label, value in values.items()}
    request = {
        "schema_version": 2,
        "source_blob": artifact(root, blob, "source_blob"),
        "source_manifest": artifact(root, source_manifest, "source_manifest"),
        "proposal": artifact(root, paths["proposal"], "proposal"),
        "execution_policy": artifact(root, paths["execution_policy"], "execution_policy"),
        "inventory_policy": artifact(root, paths["inventory_policy"], "inventory_policy"),
        "accounting_policy": artifact(root, paths["accounting_policy"], "accounting_policy"),
        "evaluation_policy": artifact(root, paths["evaluation_policy"], "evaluation_policy"),
        "instrument_registry": artifact(root, registry_path, "instrument_registry"),
        "authority_policy": artifact(root, paths["authority_policy"], "authority_policy"),
        "output_directory": "oracle_output",
    }
    write_json(root / "inputs" / "oracle_request.json", request)
    return request


def run_fd_cli(script: Path, args: list[str], fds: tuple[int, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(PYTHON), "-I", str(script), *args],
        cwd=ROOT,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"},
        pass_fds=fds,
        check=False,
        capture_output=True,
        text=True,
    )


def oracle_output_root(root: Path) -> Path:
    result = root / "oracle-publish"
    result.mkdir(mode=0o700, exist_ok=True)
    result.chmod(0o700)
    return result


def verifier_output_root(root: Path) -> Path:
    result = root / "verifier-publish"
    result.mkdir(mode=0o700, exist_ok=True)
    result.chmod(0o700)
    return result


def run_verifier(
    root: Path,
    request: dict,
    *,
    replay: object = reference.replay_reference,
) -> dict:
    return verifier.verify(
        request,
        trusted_input_root=root,
        trusted_output_root=verifier_output_root(root),
        reference_replay=replay,
        reference_code_bytes=REFERENCE_PATH.read_bytes(),
        reference_contract_bytes=REFERENCE_CONTRACT.read_bytes(),
    )


def bind_fixture_to_sealed_launcher(root: Path, request: dict) -> None:
    manifest_path = root / request["oracle_manifest"]["relative_path"]
    manifest = json.loads(manifest_path.read_text())
    manifest["oracle_release_content_binding"][
        "launcher_sha256"
    ] = SEALED_TEST_LAUNCHER_SHA256
    manifest["oracle_release_content_binding"][
        "snapshot_mode"
    ] = "SEALED_FD_COMPILE_EXEC_V2"
    manifest["oracle_root_sha256"] = embedded(manifest, "oracle_root_sha256")
    write_json(manifest_path, manifest)
    request["oracle_manifest"] = artifact(
        root, manifest_path, "oracle_manifest"
    )
    reseal_oracle_commit(root, request)


def sealed_verifier_namespace() -> dict[str, object]:
    code_bytes = VERIFIER_PATH.read_bytes()
    namespace: dict[str, object] = {
        "__name__": "sealed_paper_research_oracle_verifier_v2",
        "__package__": None,
        "__spec__": None,
        "_SEALED_RUNTIME_CODE_BYTES": code_bytes,
        "_SEALED_SCHEMA_BYTES": VERIFIER_SCHEMA.read_bytes(),
        "_SEALED_LAUNCHER_SHA256": SEALED_TEST_LAUNCHER_SHA256,
    }
    exec(compile(code_bytes, "<sealed-pure-verifier-v2>", "exec"), namespace)
    return namespace


def pure_verifier_arguments(
    root: Path,
    request: dict,
    *,
    result: dict | None = None,
) -> tuple[bytes, tuple, tuple, bytes, tuple]:
    bind_fixture_to_sealed_launcher(root, request)
    request_bytes = canonical(request) + b"\n"
    artifact_blobs = tuple(
        (
            role,
            (root / request[role]["relative_path"]).read_bytes(),
        )
        for role in verifier.SEALED_ARTIFACT_ROLES
    )
    artifact_map = dict(artifact_blobs)
    if result is None:
        result = reference.replay_reference({
            role: artifact_map[role]
            for role in verifier.REFERENCE_INPUT_LABELS
        })
    reference_result_bytes = verifier._reference_result_snapshot_bytes(result)
    oracle_release_blobs = (
        ("code_bytes", ORACLE_PATH.read_bytes()),
        ("contract_bytes", ORACLE_CONTRACT.read_bytes()),
        ("schema_bytes", ORACLE_SCHEMA.read_bytes()),
    )
    reference_attestation = (
        ("reference_code_sha256", digest(REFERENCE_PATH.read_bytes())),
        ("reference_contract_sha256", digest(REFERENCE_CONTRACT.read_bytes())),
        ("reference_result_sha256", digest(reference_result_bytes)),
    )
    return (
        request_bytes,
        artifact_blobs,
        oracle_release_blobs,
        reference_result_bytes,
        reference_attestation,
    )


def run_pure_verifier(
    root: Path,
    request: dict,
    *,
    result: dict | None = None,
) -> tuple[bytes, bytes]:
    namespace = sealed_verifier_namespace()
    entrypoint = namespace["verify_sealed_bytes"]
    return entrypoint(*pure_verifier_arguments(root, request, result=result))


def reseal_reference_projection(result: dict) -> None:
    result["economic_projection_sha256"] = digest(canonical({
        "all_transactions_balanced": result["all_transactions_balanced"],
        "engine_id": result["engine_id"],
        "input_root_sha256": result["input_root_sha256"],
        "journal_root_sha256": result["journal_root_sha256"],
        "journal_transaction_count": result["journal_transaction_count"],
        "ledger_row_count": result["ledger_row_count"],
        "ledger_sha256": digest(result["ledger_bytes"]),
        "ledger_terminal_hash": result["ledger_terminal_hash"],
        "oracle_metrics_sha256": result["oracle_metrics"]["metrics_sha256"],
        "proposal_provenance_root_sha256": result[
            "proposal_provenance_root_sha256"
        ],
    }))


def coherently_reseal_reference_and_oracle(
    root: Path,
    request: dict,
    mutate: Callable[[dict], None],
) -> tuple[bytes, tuple, tuple, bytes, tuple]:
    """Reseal every caller-controlled link after one semantic mutation."""
    bind_fixture_to_sealed_launcher(root, request)
    artifact_map = {
        role: (root / request[role]["relative_path"]).read_bytes()
        for role in verifier.SEALED_ARTIFACT_ROLES
    }
    result = reference.replay_reference({
        role: artifact_map[role]
        for role in verifier.REFERENCE_INPUT_LABELS
    })
    mutate(result)
    ledger_path = root / request["oracle_ledger"]["relative_path"]
    ledger_path.write_bytes(result["ledger_bytes"])
    request["oracle_ledger"] = artifact(root, ledger_path, "oracle_ledger")
    manifest_path = root / request["oracle_manifest"]["relative_path"]
    manifest = json.loads(manifest_path.read_text())
    manifest.update({
        "oracle_ledger_sha256": request["oracle_ledger"]["sha256"],
        "oracle_ledger_size_bytes": request["oracle_ledger"]["size_bytes"],
        "oracle_ledger_row_count": result["ledger_row_count"],
        "oracle_ledger_terminal_hash": result["ledger_terminal_hash"],
        "oracle_metrics": result["oracle_metrics"],
        "proposal_provenance_root_sha256": result[
            "proposal_provenance_root_sha256"
        ],
    })
    manifest["oracle_root_sha256"] = embedded(manifest, "oracle_root_sha256")
    write_json(manifest_path, manifest)
    request["oracle_manifest"] = artifact(
        root, manifest_path, "oracle_manifest"
    )
    reseal_oracle_commit(root, request)
    return pure_verifier_arguments(root, request, result=result)


def reseal_reference_metrics(result: dict) -> None:
    result["oracle_metrics"]["metrics_sha256"] = embedded(
        result["oracle_metrics"],
        "metrics_sha256",
    )
    reseal_reference_projection(result)


def rechained_reference_ledger(result: dict, mutate: Callable[[dict], None]) -> None:
    rows = [json.loads(line) for line in result["ledger_bytes"].splitlines()]
    mutate(rows[0])
    previous = "0" * 64
    for sequence, row in enumerate(rows, 1):
        row["ledger_sequence"] = sequence
        row["previous_hash"] = previous
        row["record_hash"] = embedded(row, "record_hash")
        previous = row["record_hash"]
    result["ledger_bytes"] = b"".join(canonical(row) + b"\n" for row in rows)
    result["ledger_row_count"] = len(rows)
    result["ledger_terminal_hash"] = previous
    reseal_reference_projection(result)


def execute_oracle(root: Path, request: dict) -> None:
    request_path = root / "inputs" / "oracle_request.json"
    publish_root = oracle_output_root(root)
    descriptors = [
        os.open(request_path, os.O_RDONLY),
        os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)),
        os.open(publish_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)),
        os.open(ORACLE_PATH, os.O_RDONLY),
        os.open(ORACLE_CONTRACT, os.O_RDONLY),
        os.open(ORACLE_SCHEMA, os.O_RDONLY),
    ]
    try:
        completed = run_fd_cli(ORACLE_PATH, [
            "--request-fd", str(descriptors[0]),
            "--input-root-fd", str(descriptors[1]),
            "--output-root-fd", str(descriptors[2]),
            "--code-fd", str(descriptors[3]),
            "--contract-fd", str(descriptors[4]),
            "--schema-fd", str(descriptors[5]),
        ], tuple(descriptors))
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)["ok"] is True


def verifier_fixture(root: Path, **input_options: object) -> tuple[dict, dict]:
    oracle_request = build_inputs(root, **input_options)
    execute_oracle(root, oracle_request)
    snapshots = {
        "oracle_code_snapshot": ORACLE_PATH,
        "oracle_contract_snapshot": ORACLE_CONTRACT,
        "oracle_schema_snapshot": ORACLE_SCHEMA,
        "reference_code_snapshot": REFERENCE_PATH,
        "reference_contract_snapshot": REFERENCE_CONTRACT,
    }
    snapshot_paths = {}
    for label, source in snapshots.items():
        target = root / "inputs" / f"{label}{source.suffix}"
        target.write_bytes(source.read_bytes())
        snapshot_paths[label] = target
    request = {
        "schema_version": 2,
        **{label: oracle_request[label] for label in (
            "source_blob", "source_manifest", "proposal", "execution_policy", "inventory_policy",
            "accounting_policy", "evaluation_policy", "instrument_registry", "authority_policy",
        )},
        "oracle_request": artifact(root, root / "inputs" / "oracle_request.json", "oracle_request"),
        **{label: artifact(root, path, label) for label, path in snapshot_paths.items()},
        "oracle_intent": artifact(root, oracle_output_root(root) / "oracle_output" / "intent.json", "oracle_intent"),
        "oracle_commit": artifact(root, oracle_output_root(root) / "oracle_output" / "COMMIT.json", "oracle_commit"),
        "oracle_ledger": artifact(root, oracle_output_root(root) / "oracle_output" / "oracle_ledger.jsonl", "oracle_ledger"),
        "oracle_manifest": artifact(root, oracle_output_root(root) / "oracle_output" / "oracle_manifest.json", "oracle_manifest"),
        "output_directory": "verifier_output",
    }
    manifest = json.loads(
        (oracle_output_root(root) / "oracle_output" / "oracle_manifest.json").read_text()
    )
    return request, manifest


def exclusive_verifier_rename_at(
    root_fd: int, source: str, destination: str
) -> None:
    if verifier._lstat_at(root_fd, destination) is not None:
        raise FileExistsError(destination)
    os.rename(source, destination, src_dir_fd=root_fd, dst_dir_fd=root_fd)


def leave_recoverable_verifier_stage(
    root: Path,
    request: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    original = verifier._write_file_at

    def fail_before_commit(directory_fd: int, name: str, data: bytes) -> None:
        if name == "COMMIT.json":
            raise OSError("injected recoverable verifier commit fault")
        original(directory_fd, name, data)

    monkeypatch.setattr(verifier, "_write_file_at", fail_before_commit)
    with pytest.raises(OSError, match="recoverable verifier commit fault"):
        run_verifier(root, request)
    monkeypatch.setattr(verifier, "_write_file_at", original)
    stages = list(verifier_output_root(root).glob(".verifier_output.*.stage"))
    assert len(stages) == 1
    return stages[0]


def verifier_expected_evidence(root: Path, request: dict) -> tuple[bytes, dict]:
    root_fd = os.open(
        root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    release = {
        "code_bytes": ORACLE_PATH.read_bytes(),
        "contract_bytes": ORACLE_CONTRACT.read_bytes(),
        "schema_bytes": ORACLE_SCHEMA.read_bytes(),
    }
    release["hashes"] = {
        "code_sha256": digest(release["code_bytes"]),
        "contract_sha256": digest(release["contract_bytes"]),
        "schema_sha256": digest(release["schema_bytes"]),
    }
    reference_release = {
        "code_bytes": REFERENCE_PATH.read_bytes(),
        "contract_bytes": REFERENCE_CONTRACT.read_bytes(),
    }
    reference_release["hashes"] = {
        "code_sha256": digest(reference_release["code_bytes"]),
        "contract_sha256": digest(reference_release["contract_bytes"]),
    }
    try:
        state = verifier._load_request(
            request, root_fd, release, reference_release
        )
        return verifier._expected_evidence(state)
    finally:
        os.close(root_fd)


def reseal_oracle_commit(root: Path, request: dict) -> None:
    commit_path = root / request["oracle_commit"]["relative_path"]
    commit = json.loads(commit_path.read_text())
    intent_path = root / request["oracle_intent"]["relative_path"]
    ledger_path = root / request["oracle_ledger"]["relative_path"]
    manifest_path = root / request["oracle_manifest"]["relative_path"]
    manifest = json.loads(manifest_path.read_text())
    intent_bytes = intent_path.read_bytes()
    ledger_bytes = ledger_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    commit.update({
        "intent_sha256": digest(intent_bytes),
        "ledger_sha256": digest(ledger_bytes),
        "ledger_size_bytes": len(ledger_bytes),
        "manifest_sha256": digest(manifest_bytes),
        "manifest_size_bytes": len(manifest_bytes),
        "terminal_hash": manifest["oracle_ledger_terminal_hash"],
    })
    write_json(commit_path, commit)
    request["oracle_commit"] = artifact(root, commit_path, "oracle_commit")


def golden_verifier_fixture(root: Path) -> tuple[dict, dict]:
    """Build a verifier envelope without executing/importing the oracle."""
    payload = build_golden_payload()
    inputs = payload["inputs"]
    expected = payload["expected"]
    input_dir = root / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    source_path = input_dir / "source_blob.jsonl"
    source_path.write_bytes(inputs["source_blob_utf8"].encode("utf-8"))
    input_paths = {
        label: write_json(input_dir / f"{label}.json", inputs[label])
        for label in (
            "source_manifest", "proposal", "execution_policy", "inventory_policy",
            "accounting_policy", "evaluation_policy", "instrument_registry", "authority_policy",
        )
    }
    oracle_request = {
        "schema_version": 2,
        "source_blob": artifact(root, source_path, "source_blob"),
        **{label: artifact(root, path, label) for label, path in input_paths.items()},
        "output_directory": "oracle_output",
    }
    oracle_request_path = write_json(input_dir / "oracle_request.json", oracle_request)
    snapshot_paths: dict[str, Path] = {}
    for label, source in {
        "oracle_code_snapshot": ORACLE_PATH,
        "oracle_contract_snapshot": ORACLE_CONTRACT,
        "oracle_schema_snapshot": ORACLE_SCHEMA,
        "reference_code_snapshot": REFERENCE_PATH,
        "reference_contract_snapshot": REFERENCE_CONTRACT,
    }.items():
        target = input_dir / f"{label}{source.suffix}"
        target.write_bytes(source.read_bytes())
        snapshot_paths[label] = target
    ledger_path = root / "golden_oracle" / "oracle_ledger.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_bytes(expected["ledger_utf8"].encode("utf-8"))
    input_hashes = {
        label: oracle_request[label]["sha256"]
        for label in (
            "source_blob", "source_manifest", "proposal", "execution_policy",
            "inventory_policy", "accounting_policy", "evaluation_policy",
            "instrument_registry", "authority_policy",
        )
    }
    proposal = inputs["proposal"]
    provenance_root = digest(canonical({
        "provenance": proposal["provenance"],
        "rows": [{
            "proposal_ordinal": row["proposal_ordinal"],
            "decision_source_event_sha256": row["decision_source_event_sha256"],
            "completed_data_watermark_source_ts_ns": row["completed_data_watermark_source_ts_ns"],
            "completed_data_prefix_root_sha256": row["completed_data_prefix_root_sha256"],
        } for row in proposal["rows"]],
    }))
    ledger_rows = [json.loads(line) for line in expected["ledger_utf8"].splitlines()]
    manifest = {
        "schema_version": 2,
        "oracle_implementation": "INDEPENDENT_JPY_ORACLE_V2",
        "status": "COMPLETE",
        "classification": (
            "FUTURE_ONLY_ACCOUNTING_ONLY_LOCAL_UNANCHORED_NOT_ADMISSIBLE"
        ),
        "causal_signal_admission": False,
        "release_evidence_eligible": False,
        "detector_replay_receipt_required": True,
        "authority": {
            "paper_only": True, "live_authority": False, "broker_account_access": False,
            "credential_access": False, "order_endpoint": False, "external_orders": 0,
            "deploy": False, "external_config_mutation": False,
        },
        "oracle_release_content_binding": {
            "code_sha256": digest(snapshot_paths["oracle_code_snapshot"].read_bytes()),
            "contract_sha256": digest(snapshot_paths["oracle_contract_snapshot"].read_bytes()),
            "schema_sha256": digest(snapshot_paths["oracle_schema_snapshot"].read_bytes()),
            "launcher_sha256": None,
            "snapshot_mode": "PATH_LOADED_TEST_ADAPTER_NOT_RELEASE_EVIDENCE",
        },
        "oracle_execution_provenance_scope": (
            "LOCAL_CALLER_ASSERTED_CONTENT_BINDING_NOT_EXECUTION_ATTESTATION_"
            "NOT_EXTERNALLY_ANCHORED"
        ),
        "request_sha256": digest(oracle_request_path.read_bytes()),
        "input_artifact_sha256": dict(sorted(input_hashes.items())),
        "raw_source_manifest_sha256": input_hashes["source_manifest"],
        "proposal_provenance_root_sha256": provenance_root,
        "producer_result_or_metrics_used": False,
        "proposal_identity_generated_by_oracle": True,
        "oracle_ledger_file": "oracle_ledger.jsonl",
        "oracle_ledger_sha256": expected["ledger_sha256"],
        "oracle_ledger_size_bytes": expected["ledger_size_bytes"],
        "oracle_ledger_row_count": len(ledger_rows),
        "oracle_ledger_terminal_hash": ledger_rows[-1]["record_hash"],
        "oracle_metrics": expected["oracle_metrics"],
        "terminal_inventory_mtm_jpy_micros": 0,
        "external_orders": 0,
        "anchor_status": "LOCAL_UNANCHORED",
    }
    manifest["oracle_root_sha256"] = embedded(manifest, "oracle_root_sha256")
    manifest_path = write_json(root / "golden_oracle" / "oracle_manifest.json", manifest)
    request_sha = digest(oracle_request_path.read_bytes())
    release = manifest["oracle_release_content_binding"]
    transaction_id = digest(canonical({
        "request_sha256": request_sha,
        "code_sha256": release["code_sha256"],
        "contract_sha256": release["contract_sha256"],
        "schema_sha256": release["schema_sha256"],
    }))
    intent = {
        "schema_version": 1,
        "transaction_id": transaction_id,
        "request_sha256": request_sha,
        "code_sha256": release["code_sha256"],
        "contract_sha256": release["contract_sha256"],
        "schema_sha256": release["schema_sha256"],
    }
    intent_path = write_json(root / "golden_oracle" / "intent.json", intent)
    commit = {
        "schema_version": 1,
        "transaction_id": transaction_id,
        "request_sha256": request_sha,
        "intent_sha256": digest(intent_path.read_bytes()),
        "ledger_sha256": digest(ledger_path.read_bytes()),
        "ledger_size_bytes": len(ledger_path.read_bytes()),
        "manifest_sha256": digest(manifest_path.read_bytes()),
        "manifest_size_bytes": len(manifest_path.read_bytes()),
        "terminal_hash": ledger_rows[-1]["record_hash"],
    }
    commit_path = write_json(root / "golden_oracle" / "COMMIT.json", commit)
    request = {
        "schema_version": 2,
        **{label: oracle_request[label] for label in (
            "source_blob", "source_manifest", "proposal", "execution_policy",
            "inventory_policy", "accounting_policy", "evaluation_policy",
            "instrument_registry", "authority_policy",
        )},
        "oracle_request": artifact(root, oracle_request_path, "oracle_request"),
        **{label: artifact(root, path, label) for label, path in snapshot_paths.items()},
        "oracle_intent": artifact(root, intent_path, "oracle_intent"),
        "oracle_commit": artifact(root, commit_path, "oracle_commit"),
        "oracle_ledger": artifact(root, ledger_path, "oracle_ledger"),
        "oracle_manifest": artifact(root, manifest_path, "oracle_manifest"),
        "output_directory": "golden_verifier_output",
    }
    return request, expected


def test_independent_verifier_rebuilds_exact_ledger_and_metrics(tmp_path: Path) -> None:
    request, manifest = verifier_fixture(tmp_path)
    result = run_verifier(tmp_path, request)
    receipt = result["receipt"]
    assert receipt["status"] == "VERIFIED_ACCOUNTING_ONLY"
    assert receipt["classification"] == verifier.CLASSIFICATION
    assert receipt["causal_signal_admission"] is False
    assert receipt["release_evidence_eligible"] is False
    assert receipt["admission_eligible"] is False
    assert receipt["oracle_root_sha256"] == manifest["oracle_root_sha256"]
    assert receipt["verified_oracle_metrics"] == manifest["oracle_metrics"]
    assert receipt["independently_rebuilt_ledger"] is True
    assert receipt["reference_engine_id"] == reference.ENGINE_ID
    assert receipt["reference_code_sha256"] == digest(REFERENCE_PATH.read_bytes())
    assert receipt["reference_contract_sha256"] == digest(
        REFERENCE_CONTRACT.read_bytes()
    )
    assert receipt["reference_all_transactions_balanced"] is True
    assert receipt["reference_accounting_diagnostics_only"] is True
    assert receipt["reference_n_eff_statistical_admission_allowed"] is False
    assert receipt["reference_direction_accuracy_profit_gate_allowed"] is False
    assert receipt["verifier_release_content_binding"][
        "reference_code_sha256"
    ] == receipt["reference_code_sha256"]
    assert receipt["verifier_release_content_binding"][
        "reference_contract_sha256"
    ] == receipt["reference_contract_sha256"]
    assert receipt["verifier_release_content_binding"][
        "reference_result_sha256"
    ] == receipt["reference_result_sha256"]
    assert receipt["external_orders"] == 0
    assert receipt["terminal_inventory_mtm_jpy_micros"] == 0


def test_supported_oracle_release_triplet_matches_exact_checked_in_bytes() -> None:
    assert verifier.SUPPORTED_ORACLE_RELEASE == {
        "code_sha256": digest(ORACLE_PATH.read_bytes()),
        "contract_sha256": digest(ORACLE_CONTRACT.read_bytes()),
        "schema_sha256": digest(ORACLE_SCHEMA.read_bytes()),
    }


def test_supported_reference_release_pair_matches_exact_checked_in_bytes() -> None:
    assert verifier.SUPPORTED_REFERENCE_RELEASE == {
        "code_sha256": digest(REFERENCE_PATH.read_bytes()),
        "contract_sha256": digest(REFERENCE_CONTRACT.read_bytes()),
    }


def test_pure_sealed_verifier_returns_exact_canonical_output_at_28bn(
    tmp_path: Path,
) -> None:
    request, manifest = verifier_fixture(
        tmp_path,
        proposal_notional_jpy_micros=28_000_000_000,
    )
    namespace = sealed_verifier_namespace()
    arguments = pure_verifier_arguments(tmp_path, request)
    output = namespace["verify_sealed_bytes"](*arguments)
    assert type(output) is tuple and len(output) == 2
    receipt_bytes, commit_bytes = output
    assert type(receipt_bytes) is bytes
    assert type(commit_bytes) is bytes
    receipt = json.loads(receipt_bytes)
    commit = json.loads(commit_bytes)
    assert receipt_bytes == canonical(receipt) + b"\n"
    assert commit_bytes == canonical(commit) + b"\n"
    assert set(receipt) == set(namespace["VERIFIER_RECEIPT_KEYS"])
    assert set(commit) == {
        "schema_version",
        "request_sha256",
        "receipt_sha256",
        "receipt_size_bytes",
        "verifier_receipt_sha256",
    }
    assert receipt["verified_oracle_metrics"] == manifest["oracle_metrics"]
    assert receipt["reference_all_transactions_balanced"] is True
    assert receipt["reference_result_sha256"] == arguments[4][2][1]
    assert receipt["verifier_release_content_binding"][
        "reference_result_sha256"
    ] == receipt["reference_result_sha256"]
    assert commit["request_sha256"] == digest(arguments[0])
    assert commit["receipt_sha256"] == digest(receipt_bytes)
    assert commit["receipt_size_bytes"] == len(receipt_bytes)
    assert commit["verifier_receipt_sha256"] == receipt[
        "verifier_receipt_sha256"
    ]


def test_pure_sealed_verifier_rejects_capability_and_tuple_shape_injection(
    tmp_path: Path,
) -> None:
    request, _ = verifier_fixture(tmp_path)
    namespace = sealed_verifier_namespace()
    arguments = list(pure_verifier_arguments(tmp_path, request))
    entrypoint = namespace["verify_sealed_bytes"]
    error = namespace["VerificationError"]

    arguments[1] = list(arguments[1])
    with pytest.raises(error, match="exact fixed tuple"):
        entrypoint(*arguments)

    arguments = list(pure_verifier_arguments(tmp_path, request))
    arguments[1] = tuple(reversed(arguments[1]))
    with pytest.raises(error, match="keys or ordering"):
        entrypoint(*arguments)

    called = False

    def forbidden_callable() -> bytes:
        nonlocal called
        called = True
        return b"forbidden"

    arguments = list(pure_verifier_arguments(tmp_path, request))
    first_role = arguments[1][0][0]
    arguments[1] = ((first_role, forbidden_callable), *arguments[1][1:])
    with pytest.raises(error, match="type mismatch"):
        entrypoint(*arguments)
    assert called is False

    arguments = list(pure_verifier_arguments(tmp_path, request))
    arguments[4] = dict(arguments[4])
    with pytest.raises(error, match="exact fixed tuple"):
        entrypoint(*arguments)


def test_pure_sealed_verifier_rejects_reference_snapshot_tamper_and_reseal(
    tmp_path: Path,
) -> None:
    request, _ = verifier_fixture(tmp_path)
    namespace = sealed_verifier_namespace()
    error = namespace["VerificationError"]
    arguments = list(pure_verifier_arguments(tmp_path, request))

    snapshot = json.loads(arguments[3])
    snapshot["journal_root_sha256"] = "e" * 64
    tampered = canonical(snapshot) + b"\n"
    arguments[3] = tampered
    with pytest.raises(error, match="snapshot hash mismatch"):
        namespace["verify_sealed_bytes"](*arguments)

    artifact_map = dict(pure_verifier_arguments(tmp_path, request)[1])
    result = reference.replay_reference({
        role: artifact_map[role]
        for role in verifier.REFERENCE_INPUT_LABELS
    })
    metrics = json.loads(json.dumps(result["oracle_metrics"]))
    metrics["arms"]["ADVERSE_STRESS"][
        "minimum_free_margin_jpy_micros"
    ] += 1
    metrics["metrics_sha256"] = embedded(metrics, "metrics_sha256")
    result["oracle_metrics"] = metrics
    reseal_reference_projection(result)
    resealed_arguments = pure_verifier_arguments(
        tmp_path,
        request,
        result=result,
    )
    with pytest.raises(error, match="oracle metrics differ"):
        namespace["verify_sealed_bytes"](*resealed_arguments)


@pytest.mark.parametrize(
    "case",
    (
        "arm_extra_key",
        "arm_missing_key",
        "unknown_arm",
        "arm_terminal_bool",
        "arm_terminal_nonzero",
        "arm_integer_bool",
        "arm_boolean_int",
        "arm_hash_malformed",
        "arm_ratio_malformed",
        "cluster_extra_key",
        "cluster_integer_bool",
        "month_missing_key",
        "month_boolean_int",
        "initial_equity_mismatch",
        "initial_equity_bool",
    ),
)
def test_pure_verifier_rejects_coherently_resealed_recursive_metrics(
    tmp_path: Path,
    case: str,
) -> None:
    root = tmp_path / case
    request, _ = verifier_fixture(root)

    def mutate(result: dict) -> None:
        metrics = json.loads(json.dumps(result["oracle_metrics"]))
        arm = metrics["arms"]["RAW_SIGNAL"]
        if case == "arm_extra_key":
            arm["unexpected_authority_alias"] = 1
        elif case == "arm_missing_key":
            arm.pop("terminal_open_positions")
        elif case == "unknown_arm":
            metrics["arms"]["UNKNOWN_ARM"] = dict(arm)
        elif case == "arm_terminal_bool":
            arm["terminal_open_positions"] = False
        elif case == "arm_terminal_nonzero":
            arm["terminal_inventory_mtm_jpy_micros"] = 1
        elif case == "arm_integer_bool":
            arm["proposal_count"] = True
        elif case == "arm_boolean_int":
            arm["margin_guard_pass"] = 1
        elif case == "arm_hash_malformed":
            arm["signal_id_set_sha256"] = "not-a-digest"
        elif case == "arm_ratio_malformed":
            arm["direction_accuracy"] = "1.0"
        elif case == "cluster_extra_key":
            arm["currency_time_cluster_observations"][0]["unexpected"] = 0
        elif case == "cluster_integer_bool":
            arm["currency_time_cluster_observations"][0]["time_bucket"] = False
        elif case == "month_missing_key":
            arm["monthly"][0].pop("ruin_observed")
        elif case == "month_boolean_int":
            arm["monthly"][0]["comparable_full_month"] = 0
        elif case == "initial_equity_mismatch":
            metrics["initial_equity_jpy_micros"] += 1
        else:
            metrics["initial_equity_jpy_micros"] = True
        result["oracle_metrics"] = metrics
        reseal_reference_metrics(result)

    arguments = coherently_reseal_reference_and_oracle(
        root,
        request,
        mutate,
    )
    namespace = sealed_verifier_namespace()
    with pytest.raises(namespace["VerificationError"]):
        namespace["verify_sealed_bytes"](*arguments)


@pytest.mark.parametrize("arm_name", verifier.ARMS)
@pytest.mark.parametrize("mutation", ("extra", "missing"))
def test_pure_verifier_freezes_every_arm_metric_key_set_under_full_reseal(
    tmp_path: Path,
    arm_name: str,
    mutation: str,
) -> None:
    root = tmp_path / f"{arm_name}-{mutation}"
    request, _ = verifier_fixture(root)

    def mutate(result: dict) -> None:
        metrics = json.loads(json.dumps(result["oracle_metrics"]))
        arm = metrics["arms"][arm_name]
        if mutation == "extra":
            arm["unexpected"] = 0
        else:
            arm.pop("terminal_open_positions")
        result["oracle_metrics"] = metrics
        reseal_reference_metrics(result)

    arguments = coherently_reseal_reference_and_oracle(
        root,
        request,
        mutate,
    )
    namespace = sealed_verifier_namespace()
    with pytest.raises(namespace["VerificationError"], match="schema mismatch"):
        namespace["verify_sealed_bytes"](*arguments)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("external_order_count", False),
        ("external_order_count", 1),
        ("terminal_inventory_mtm_jpy_micros", False),
        ("terminal_inventory_mtm_jpy_micros", 1),
    ),
)
def test_pure_verifier_rejects_coherently_resealed_ledger_authority(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    request, _ = verifier_fixture(tmp_path)

    def mutate(result: dict) -> None:
        rechained_reference_ledger(
            result,
            lambda row: row.__setitem__(field, value),
        )

    arguments = coherently_reseal_reference_and_oracle(
        tmp_path,
        request,
        mutate,
    )
    namespace = sealed_verifier_namespace()
    with pytest.raises(
        namespace["VerificationError"],
        match="authority invariant",
    ):
        namespace["verify_sealed_bytes"](*arguments)


def test_pure_verifier_rejects_coherently_resealed_provenance_root(
    tmp_path: Path,
) -> None:
    request, _ = verifier_fixture(tmp_path)

    def mutate(result: dict) -> None:
        result["proposal_provenance_root_sha256"] = "e" * 64
        reseal_reference_projection(result)

    arguments = coherently_reseal_reference_and_oracle(
        tmp_path,
        request,
        mutate,
    )
    namespace = sealed_verifier_namespace()
    with pytest.raises(
        namespace["VerificationError"],
        match="provenance root mismatch",
    ):
        namespace["verify_sealed_bytes"](*arguments)


@pytest.mark.parametrize(
    "case",
    ("input_root", "ledger_coverage", "economic_projection"),
)
def test_pure_verifier_rejects_coherently_resealed_semantic_roots(
    tmp_path: Path,
    case: str,
) -> None:
    root = tmp_path / case
    request, _ = verifier_fixture(root)

    def mutate(result: dict) -> None:
        if case == "input_root":
            result["input_root_sha256"] = "e" * 64
            reseal_reference_projection(result)
        elif case == "economic_projection":
            result["economic_projection_sha256"] = "e" * 64
        else:
            rows = [
                json.loads(line)
                for line in result["ledger_bytes"].splitlines()
            ][:-1]
            previous = "0" * 64
            for sequence, row in enumerate(rows, 1):
                row["ledger_sequence"] = sequence
                row["previous_hash"] = previous
                row["record_hash"] = embedded(row, "record_hash")
                previous = row["record_hash"]
            result["ledger_bytes"] = b"".join(
                canonical(row) + b"\n" for row in rows
            )
            result["ledger_row_count"] = len(rows)
            result["ledger_terminal_hash"] = previous
            reseal_reference_projection(result)

    arguments = coherently_reseal_reference_and_oracle(
        root,
        request,
        mutate,
    )
    namespace = sealed_verifier_namespace()
    with pytest.raises(namespace["VerificationError"]):
        namespace["verify_sealed_bytes"](*arguments)


def test_pure_verifier_rejects_coherently_resealed_manifest_authority(
    tmp_path: Path,
) -> None:
    request, _ = verifier_fixture(tmp_path)
    bind_fixture_to_sealed_launcher(tmp_path, request)
    manifest_path = tmp_path / request["oracle_manifest"]["relative_path"]
    manifest = json.loads(manifest_path.read_text())
    manifest["authority"]["live_authority"] = True
    manifest["oracle_root_sha256"] = embedded(manifest, "oracle_root_sha256")
    write_json(manifest_path, manifest)
    request["oracle_manifest"] = artifact(
        tmp_path,
        manifest_path,
        "oracle_manifest",
    )
    reseal_oracle_commit(tmp_path, request)
    arguments = pure_verifier_arguments(tmp_path, request)
    namespace = sealed_verifier_namespace()
    with pytest.raises(namespace["VerificationError"]):
        namespace["verify_sealed_bytes"](*arguments)


def test_pure_sealed_verifier_rejects_noncanonical_reference_ledger_base64(
    tmp_path: Path,
) -> None:
    request, _ = verifier_fixture(tmp_path)
    namespace = sealed_verifier_namespace()
    error = namespace["VerificationError"]
    arguments = list(pure_verifier_arguments(tmp_path, request))
    snapshot = json.loads(arguments[3])
    snapshot["ledger_bytes_base64"] += "="
    arguments[3] = canonical(snapshot) + b"\n"
    arguments[4] = (
        arguments[4][0],
        arguments[4][1],
        ("reference_result_sha256", digest(arguments[3])),
    )
    with pytest.raises(error, match="base64"):
        namespace["verify_sealed_bytes"](*arguments)


def test_pure_sealed_verifier_bypasses_callable_and_cloned_economic_paths(
    tmp_path: Path,
) -> None:
    request, manifest = verifier_fixture(tmp_path)
    namespace = sealed_verifier_namespace()

    def forbidden(*_: object, **__: object) -> object:
        raise AssertionError("callable or cloned economics path was reached")

    namespace["_expected_evidence"] = forbidden
    namespace["_verify_actual_evidence"] = forbidden
    receipt_bytes, _ = namespace["verify_sealed_bytes"](
        *pure_verifier_arguments(tmp_path, request)
    )
    assert json.loads(receipt_bytes)["verified_oracle_metrics"] == manifest[
        "oracle_metrics"
    ]


def test_pure_output_validator_rejects_resealed_receipt_and_commit_links(
    tmp_path: Path,
) -> None:
    request, _ = verifier_fixture(tmp_path)
    namespace = sealed_verifier_namespace()
    arguments = pure_verifier_arguments(tmp_path, request)
    receipt_bytes, commit_bytes = namespace["verify_sealed_bytes"](*arguments)
    receipt = json.loads(receipt_bytes)
    commit = json.loads(commit_bytes)
    receipt["reference_result_sha256"] = "e" * 64
    receipt["verifier_receipt_sha256"] = embedded(
        receipt, "verifier_receipt_sha256"
    )
    tampered_receipt_bytes = canonical(receipt) + b"\n"
    commit["receipt_sha256"] = digest(tampered_receipt_bytes)
    commit["receipt_size_bytes"] = len(tampered_receipt_bytes)
    commit["verifier_receipt_sha256"] = receipt["verifier_receipt_sha256"]
    tampered_commit_bytes = canonical(commit) + b"\n"
    with pytest.raises(
        namespace["VerificationError"],
        match="receipt semantic output mismatch",
    ):
        namespace["_validate_pure_output_bytes"](
            arguments[0],
            json.loads(receipt_bytes),
            tampered_receipt_bytes,
            tampered_commit_bytes,
        )


def test_verifier_requires_explicit_reference_callable_in_test_adapter(
    tmp_path: Path,
) -> None:
    request, _ = verifier_fixture(tmp_path)
    with pytest.raises(verifier.VerificationError, match="incomplete"):
        verifier.verify(
            request,
            trusted_input_root=tmp_path,
            trusted_output_root=verifier_output_root(tmp_path),
            reference_code_bytes=REFERENCE_PATH.read_bytes(),
            reference_contract_bytes=REFERENCE_CONTRACT.read_bytes(),
        )


def test_verifier_rejects_non_mapping_reference_result(tmp_path: Path) -> None:
    request, _ = verifier_fixture(tmp_path)
    with pytest.raises(verifier.VerificationError, match="exact dict"):
        run_verifier(tmp_path, request, replay=lambda _: None)


def test_reference_callable_receives_only_nine_raw_byte_artifacts(
    tmp_path: Path,
) -> None:
    request, _ = verifier_fixture(tmp_path)
    observed: dict[str, object] = {}

    def replay(artifacts: object) -> dict:
        assert type(artifacts) is dict
        observed.update(artifacts)
        return reference.replay_reference(artifacts)

    run_verifier(tmp_path, request, replay=replay)
    assert tuple(observed) == verifier.REFERENCE_INPUT_LABELS
    assert all(type(value) is bytes for value in observed.values())
    assert set(observed).isdisjoint({
        "oracle_request", "oracle_intent", "oracle_ledger", "oracle_manifest",
        "oracle_commit", "producer_result", "producer_metrics",
    })


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("unknown_key", "schema mismatch"),
        ("wrong_engine", "engine identity"),
        ("bool_ledger_count", "ledger_row_count must be integer"),
        ("unbalanced", "not exactly balanced"),
    ],
)
def test_verifier_rejects_malformed_reference_result_shape_and_types(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    root = tmp_path / case
    request, _ = verifier_fixture(root)

    def replay(artifacts: object) -> dict:
        result = reference.replay_reference(artifacts)
        if case == "unknown_key":
            result["unexpected"] = 1
        elif case == "wrong_engine":
            result["engine_id"] = "WRONG_REFERENCE_ENGINE"
        elif case == "bool_ledger_count":
            result["ledger_row_count"] = True
        else:
            result["all_transactions_balanced"] = False
        return result

    with pytest.raises(verifier.VerificationError, match=message):
        run_verifier(root, request, replay=replay)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("ledger", "reference ledger"),
        ("metrics", "ending equity does not reconcile"),
        ("provenance", "proposal provenance root mismatch"),
        ("journal", "economic projection"),
        ("projection", "economic projection"),
        ("input_root", "input root"),
        ("terminal", "terminal hash"),
    ],
)
def test_verifier_rejects_tampered_reference_economic_result(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    root = tmp_path / case
    request, _ = verifier_fixture(root)

    def replay(artifacts: object) -> dict:
        result = reference.replay_reference(artifacts)
        if case == "ledger":
            result["ledger_bytes"] += b"x"
        elif case == "metrics":
            metrics = json.loads(json.dumps(result["oracle_metrics"]))
            metrics["arms"]["EXECUTABLE_BASE"][
                "ending_equity_jpy_micros"
            ] += 1
            metrics["metrics_sha256"] = embedded(metrics, "metrics_sha256")
            result["oracle_metrics"] = metrics
            reseal_reference_projection(result)
        elif case == "provenance":
            result["proposal_provenance_root_sha256"] = "e" * 64
            reseal_reference_projection(result)
        elif case == "journal":
            result["journal_root_sha256"] = "e" * 64
        elif case == "projection":
            result["economic_projection_sha256"] = "e" * 64
        elif case == "input_root":
            result["input_root_sha256"] = "e" * 64
            reseal_reference_projection(result)
        else:
            result["ledger_terminal_hash"] = "e" * 64
            reseal_reference_projection(result)
        return result

    with pytest.raises(verifier.VerificationError, match=message):
        run_verifier(root, request, replay=replay)


def test_legacy_expected_evidence_is_not_authoritative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, manifest = verifier_fixture(tmp_path)

    def forbidden(_: object) -> object:
        raise AssertionError("legacy cloned economics was reached")

    monkeypatch.setattr(verifier, "_expected_evidence", forbidden)
    receipt = run_verifier(tmp_path, request)["receipt"]
    assert receipt["verified_oracle_metrics"] == manifest["oracle_metrics"]


def test_reference_and_oracle_match_at_28bn_free_margin_boundary(
    tmp_path: Path,
) -> None:
    request, _ = verifier_fixture(
        tmp_path,
        proposal_notional_jpy_micros=28_000_000_000,
    )
    receipt = run_verifier(tmp_path, request)["receipt"]
    assert receipt["reference_all_transactions_balanced"] is True


def test_reference_rejects_one_micro_free_margin_divergence_at_28bn(
    tmp_path: Path,
) -> None:
    request, _ = verifier_fixture(
        tmp_path,
        proposal_notional_jpy_micros=28_000_000_000,
    )

    def replay(artifacts: object) -> dict:
        result = reference.replay_reference(artifacts)
        metrics = json.loads(json.dumps(result["oracle_metrics"]))
        metrics["arms"]["ADVERSE_STRESS"][
            "minimum_free_margin_jpy_micros"
        ] += 1
        metrics["metrics_sha256"] = embedded(metrics, "metrics_sha256")
        result["oracle_metrics"] = metrics
        reseal_reference_projection(result)
        return result

    with pytest.raises(verifier.VerificationError, match="oracle metrics differ"):
        run_verifier(tmp_path, request, replay=replay)


def test_verifier_accepts_independently_hand_derived_golden_ledger(tmp_path: Path) -> None:
    request, expected = golden_verifier_fixture(tmp_path)
    result = run_verifier(tmp_path, request)
    receipt = result["receipt"]
    assert receipt["verified_oracle_metrics"] == expected["oracle_metrics"]
    assert receipt["expected_canonical_ledger_sha256"] == (
        "fe8520a5b77a37c6cc2b5f22db109fb9094253a714a8ee578831fa46a03e8145"
    )
    assert receipt["oracle_ledger_size_bytes"] == 9_100


def test_verifier_exact_rational_golden_vector_is_hand_derived() -> None:
    numerator = 8_439_773_459_423_196_600_373_401_704_310_476_485_109
    denominator = 279_443_915_986
    expected = numerator * 1_000_000 // denominator
    assert verifier._asset_micros(verifier.Fraction(numerator, denominator)) == expected
    assert expected == 30_202_029_733_386_734_448_141_701_487_551_657


def test_verifier_equal_arrival_conversion_sees_complete_batch() -> None:
    books = {
        "EUR_USD": [{"source_ts_ns": 100, "arrival_ts_ns": 200, "bid_ticks": 100, "ask_ticks": 101, "tick_scale": 100}],
        "USD_JPY": [
            {"source_ts_ns": 90, "arrival_ts_ns": 190, "bid_ticks": 100, "ask_ticks": 101, "tick_scale": 100},
            {"source_ts_ns": 150, "arrival_ts_ns": 200, "bid_ticks": 200, "ask_ticks": 201, "tick_scale": 100},
        ],
    }
    watermark = verifier._arrival_watermark_from_books(books, 200)
    assert watermark == 150
    assert verifier._jpy_value(
        verifier.Fraction(1, 1), "USD", watermark, 200, books, 200
    ) == verifier.Fraction(2, 1)


def test_verifier_signed_currency_exposure_uses_independent_marked_nodes() -> None:
    books = {
        "EUR_USD": [{
            "source_ts_ns": 100,
            "arrival_ts_ns": 100,
            "bid_ticks": 1,
            "ask_ticks": 1,
            "tick_scale": 1,
        }],
        "USD_JPY": [{
            "source_ts_ns": 100,
            "arrival_ts_ns": 100,
            "bid_ticks": 100,
            "ask_ticks": 110,
            "tick_scale": 1,
        }],
    }
    registry = {
        "EUR_USD": {"price_scale": 1, "pip_ticks": 1},
        "USD_JPY": {"price_scale": 1, "pip_ticks": 1},
    }
    accounting = {"max_conversion_staleness_ns": 1}

    def position(instrument: str, direction: int, units: int = 1_000_000) -> dict:
        return {
            "proposal": {"instrument": instrument, "direction": direction},
            "units_micros": units,
        }

    def exposure(
        positions: list[dict], marks: list[verifier.Fraction]
    ) -> dict[str, int]:
        return verifier._signed_exposure(
            positions,
            [{"mark_price": price} for price in marks],
            100,
            100,
            books,
            accounting,
            registry,
        )

    assert exposure(
        [position("EUR_USD", 1)], [verifier.Fraction(1)]
    ) == {"EUR": 100_000_000, "USD": -110_000_000}
    assert exposure(
        [position("EUR_USD", -1)], [verifier.Fraction(1)]
    ) == {"EUR": -110_000_000, "USD": 100_000_000}
    assert exposure(
        [position("EUR_USD", 1), position("USD_JPY", 1)],
        [verifier.Fraction(1), verifier.Fraction(100)],
    ) == {"EUR": 100_000_000, "JPY": -100_000_000, "USD": -10_000_000}
    assert exposure(
        [position("EUR_USD", 1), position("EUR_USD", -1)],
        [verifier.Fraction(1), verifier.Fraction(1)],
    ) == {"EUR": -10_000_000, "USD": -10_000_000}


def test_verifier_replay_enforces_same_pair_collision_in_every_arm(
    tmp_path: Path,
) -> None:
    request, _ = verifier_fixture(
        tmp_path,
        proposal_specs=((0, 1), (4, -1)),
        proposal_max_age_ns=600_000_000_000,
    )
    ledger_bytes, _ = verifier_expected_evidence(tmp_path, request)
    rows = [json.loads(line) for line in ledger_bytes.splitlines()]
    for arm in verifier.ARMS:
        dispositions = {
            row["proposal_ordinal"]: row["status"]
            for row in rows
            if row["arm"] == arm
        }
        assert dispositions[1] == "FILLED_CLOSED"
        assert dispositions[2] == "SAME_PAIR_COLLISION_REJECTED"


def test_verifier_zero_activity_full_month_grid_is_retained() -> None:
    start = verifier._month_bounds_ns("2026-01")[0]
    end = verifier._month_bounds_ns("2026-04")[0]
    evaluation = {
        "period_start_ts_ns": start,
        "period_end_ts_ns": end,
        "initial_equity_jpy_micros": 100_000_000,
        "margin_notional_cap_jpy_micros": 100_000_000,
        "cvar_tail_bps": 500,
        "cluster_window_ns": 3_600_000_000_000,
    }
    metrics = verifier._arm_metrics(
        [], [], [], [], {}, {}, {}, evaluation, 1
    )
    assert [row["month_id"] for row in metrics["monthly"]] == [
        "2026-01", "2026-02", "2026-03"
    ]
    assert all(row["comparable_full_month"] is True for row in metrics["monthly"])
    assert all(
        row["equity_multiple"] == "1.000000000000000000"
        for row in metrics["monthly"]
    )


def test_verifier_cluster_cvar_is_ticket_partition_invariant() -> None:
    evaluation = {
        "initial_equity_jpy_micros": 1_000_000,
        "cluster_window_ns": 1_000,
        "cvar_tail_bps": 5_000,
    }
    whole = [{
        "status": "FILLED_CLOSED",
        "entry_arrival_ts_ns": 1_500,
        "instrument": "EUR_USD",
        "signal_id": "a" * 64,
        "economic_lot_id": "a" * 64,
        "net_pnl_jpy_micros": -100,
        "economic_net_pnl_jpy_micros_numerator": -100,
        "economic_net_pnl_jpy_micros_denominator": 1,
    }]
    split = [
        {**whole[0], "signal_id": "b" * 64, "net_pnl_jpy_micros": -41,
         "economic_net_pnl_jpy_micros_numerator": -40},
        {**whole[0], "signal_id": "c" * 64, "net_pnl_jpy_micros": -61,
         "economic_net_pnl_jpy_micros_numerator": -60},
    ]
    whole_metrics = verifier._cluster_metrics(whole, evaluation)
    split_metrics = verifier._cluster_metrics(split, evaluation)
    assert whole_metrics[:3] == split_metrics[:3] == (
        1, -100, "-0.000100000000000000"
    )
    assert whole_metrics[3][0]["time_bucket"] == split_metrics[3][0]["time_bucket"]
    assert whole_metrics[3][0]["currency_nodes"] == ["EUR", "USD"]
    assert split_metrics[3][0]["cluster_risk_net_pnl_jpy_micros"] == -100
    assert split_metrics[3][0]["ledger_net_pnl_jpy_micros"] == -102


@pytest.mark.parametrize("adverse_change", ("EASIER_LATENCY", "NO_STRICT_WORSENING"))
def test_verifier_rejects_adverse_policy_that_is_not_strictly_harder(
    tmp_path: Path, adverse_change: str
) -> None:
    build_inputs(tmp_path)
    load = lambda name: json.loads((tmp_path / "inputs" / f"{name}.json").read_text())
    execution = load("execution_policy")
    base = execution["arms"]["EXECUTABLE_BASE"]
    adverse = execution["arms"]["ADVERSE_STRESS"]
    if adverse_change == "EASIER_LATENCY":
        adverse["latency_ns"] = base["latency_ns"] - 1
    else:
        execution["arms"]["ADVERSE_STRESS"] = dict(base)
    execution["execution_policy_sha256"] = embedded(
        execution, "execution_policy_sha256"
    )
    with pytest.raises(verifier.VerificationError, match="ordering invalid"):
        verifier._validate_policies(
            execution,
            load("inventory_policy"),
            load("accounting_policy"),
            load("evaluation_policy"),
            load("authority_policy"),
        )


def test_verifier_margin_closeout_halts_later_proposal_in_every_arm(
    tmp_path: Path,
) -> None:
    request, _ = verifier_fixture(
        tmp_path,
        proposal_specs=((0, 1), (8, 1)),
        proposal_max_age_ns=600_000_000_000,
        initial_equity_jpy_micros=20_000_000_000,
        eur_bid_ticks=(110_000, 110_000, 110_000, 50_000, 50_000, 50_000, 50_000, 50_000, 50_000),
    )
    ledger_bytes, manifest = verifier_expected_evidence(tmp_path, request)
    rows = [json.loads(line) for line in ledger_bytes.splitlines()]
    for arm in verifier.ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        first = next(row for row in arm_rows if row["proposal_ordinal"] == 1)
        second = next(row for row in arm_rows if row["proposal_ordinal"] == 2)
        assert first["exit_disposition"] == "MARGIN_CLOSEOUT"
        assert second["status"] == "ACCOUNT_HALTED"
        metrics = manifest["oracle_metrics"]["arms"][arm]
        assert metrics["margin_guard_pass"] is False
        assert metrics["terminal_open_positions"] == 0
        assert metrics["terminal_inventory_mtm_jpy_micros"] == 0


def test_verifier_inventory_cap_closeout_halts_later_proposal_in_every_arm(
    tmp_path: Path,
) -> None:
    request, _ = verifier_fixture(
        tmp_path,
        proposal_specs=((0, 1), (10, 1)),
        proposal_max_age_ns=800_000_000_000,
        proposal_notional_jpy_micros=100_000_000,
        eur_bid_ticks=(99_999,) * 9,
        usd_jpy_bid_ticks=(10_000, 10_000, 10_000, 12_000, 12_000, 12_000, 12_000, 12_000, 12_000),
        max_currency_notional_jpy_micros=105_000_000,
    )
    ledger_bytes, manifest = verifier_expected_evidence(tmp_path, request)
    rows = [json.loads(line) for line in ledger_bytes.splitlines()]
    for arm in verifier.ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        first = next(row for row in arm_rows if row["proposal_ordinal"] == 1)
        second = next(row for row in arm_rows if row["proposal_ordinal"] == 2)
        assert first["status"] == "FILLED_CLOSED"
        assert first["exit_disposition"] == "INVENTORY_CAP_CLOSEOUT"
        assert second["status"] == "ACCOUNT_HALTED"
        metrics = manifest["oracle_metrics"]["arms"][arm]
        assert metrics["margin_guard_pass"] is False
        assert metrics["terminal_open_positions"] == 0
        assert metrics["terminal_inventory_mtm_jpy_micros"] == 0


@pytest.mark.parametrize(
    "reason", ("MARGIN_CLOSEOUT", "INVENTORY_CAP_CLOSEOUT")
)
def test_legacy_replay_no_tick_terminal_boundary_closes_without_batch_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
) -> None:
    request, _ = verifier_fixture(
        tmp_path,
        proposal_specs=((0, 1),),
        proposal_max_age_ns=2_000_000_000_000,
        initial_equity_jpy_micros=20_000_000_000,
    )
    evaluation = json.loads(
        (tmp_path / "inputs/evaluation_policy.json").read_text()
    )
    terminal_arrival_ns = evaluation["period_end_ts_ns"] - 1
    original = verifier._risk_closeout_reason

    def forced_boundary_reason(mark: dict, evaluation: dict, inventory: dict):
        actual = original(mark, evaluation, inventory)
        if actual is not None:
            return actual
        if mark["arrival_ts_ns"] == terminal_arrival_ns \
                and mark["gross_notional_jpy_micros"] > 0:
            return reason
        return None

    monkeypatch.setattr(verifier, "_risk_closeout_reason", forced_boundary_reason)
    ledger_bytes, _ = verifier_expected_evidence(tmp_path, request)
    rows = [json.loads(line) for line in ledger_bytes.splitlines()]
    for arm in verifier.ARMS:
        row = next(item for item in rows if item["arm"] == arm)
        assert row["status"] == "FILLED_CLOSED"
        assert row["exit_disposition"] == reason
        assert row["exit_arrival_ts_ns"] == terminal_arrival_ns


def test_verifier_risk_reason_covers_margin_and_inventory_hard_caps() -> None:
    evaluation = {"margin_notional_cap_jpy_micros": 100}
    inventory = {
        "max_gross_notional_jpy_micros": 100,
        "max_currency_notional_jpy_micros": 100,
    }
    healthy = {
        "marked_equity_jpy_micros": 100,
        "free_margin_jpy_micros": 10,
        "margin_ratio_pass": True,
        "gross_notional_jpy_micros": 90,
        "signed_currency_exposure_jpy_micros": {"EUR": 90, "USD": -90},
    }
    assert verifier._risk_closeout_reason(
        healthy, evaluation, inventory
    ) is None
    for changed in (
        {"marked_equity_jpy_micros": 0},
        {"free_margin_jpy_micros": -1},
        {"margin_ratio_pass": False},
        {"gross_notional_jpy_micros": 101},
    ):
        assert verifier._risk_closeout_reason(
            {**healthy, **changed}, evaluation, inventory
        ) == "MARGIN_CLOSEOUT"
    assert verifier._risk_closeout_reason(
        {
            **healthy,
            "signed_currency_exposure_jpy_micros": {"USD": -101},
        },
        evaluation,
        inventory,
    ) == "INVENTORY_CAP_CLOSEOUT"


def test_verifier_short_entry_ignores_filled_notional_admission_substitution() -> None:
    event = {
        "instrument": "USD_JPY",
        "source_ts_ns": 1,
        "arrival_ts_ns": 1,
        "bid_ticks": 10_000,
        "ask_ticks": 10_100,
        "tick_scale": 100,
    }
    position = {
        "proposal": {"instrument": "USD_JPY", "direction": -1},
        "policy": {
            "raw_mid": False,
            "slippage_micropips_per_side": 0,
            "commission_ppm_per_side": 0,
            "financing_ppm_per_day": 0,
        },
        "entry": event,
        "entry_price": verifier.Fraction(100, 1),
        "units_micros": verifier.BASE_MICROUNITS_PER_UNIT,
        "entry_notional_exact_jpy_micros": verifier.Fraction(100_000_000, 1),
        "entry_notional_jpy_micros": 100_000_000,
        # A legacy/internal marker must never switch risk back to filled
        # opening notional; same-clock executable liquidation is ASK 101.
        "pending_entry_admission": True,
    }
    evaluation = {
        "initial_equity_jpy_micros": 1_000_000_000,
        "margin_rate_bps": 500,
        "max_gross_to_equity_bps": 20_000,
        "margin_notional_cap_jpy_micros": 100_500_000,
    }
    mark = verifier._mark(
        [position],
        [],
        1,
        [event],
        {"USD_JPY": [event]},
        {"max_conversion_staleness_ns": 10},
        {"USD_JPY": {"price_scale": 100, "pip_ticks": 1}},
        evaluation,
        10,
    )

    assert mark["gross_notional_jpy_micros"] == 101_000_000
    assert mark["signed_currency_exposure_jpy_micros"] == {
        "JPY": 101_000_000,
        "USD": -101_000_000,
    }
    assert mark["required_margin_jpy_micros"] == 5_050_000
    assert verifier._risk_closeout_reason(
        mark,
        evaluation,
        {
            "max_gross_notional_jpy_micros": 1_000_000_000,
            "max_currency_notional_jpy_micros": 1_000_000_000,
        },
    ) == "MARGIN_CLOSEOUT"


def test_rechained_and_resealed_economic_tamper_is_rejected(tmp_path: Path) -> None:
    request, _ = verifier_fixture(tmp_path)
    ledger_path = tmp_path / request["oracle_ledger"]["relative_path"]
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    target = next(row for row in rows if row["status"] == "FILLED_CLOSED")
    target["net_pnl_jpy_micros"] += 7
    target["realized_cost_jpy_micros"] -= 7
    previous = "0" * 64
    for sequence, row in enumerate(rows, 1):
        row["ledger_sequence"] = sequence
        row["previous_hash"] = previous
        row["record_hash"] = embedded(row, "record_hash")
        previous = row["record_hash"]
    ledger_path.write_bytes(b"".join(canonical(row) + b"\n" for row in rows))
    request["oracle_ledger"] = artifact(tmp_path, ledger_path, "oracle_ledger")
    manifest_path = tmp_path / request["oracle_manifest"]["relative_path"]
    manifest = json.loads(manifest_path.read_text())
    manifest["oracle_ledger_sha256"] = request["oracle_ledger"]["sha256"]
    manifest["oracle_ledger_size_bytes"] = request["oracle_ledger"]["size_bytes"]
    manifest["oracle_ledger_terminal_hash"] = rows[-1]["record_hash"]
    manifest["oracle_root_sha256"] = embedded(manifest, "oracle_root_sha256")
    write_json(manifest_path, manifest)
    request["oracle_manifest"] = artifact(tmp_path, manifest_path, "oracle_manifest")
    reseal_oracle_commit(tmp_path, request)
    with pytest.raises(verifier.VerificationError, match="ledger differs"):
        run_verifier(tmp_path, request)


def test_metrics_or_authority_reseal_is_rejected(tmp_path: Path) -> None:
    for case in ("metrics", "authority"):
        root = tmp_path / case
        request, _ = verifier_fixture(root)
        manifest_path = root / request["oracle_manifest"]["relative_path"]
        manifest = json.loads(manifest_path.read_text())
        if case == "metrics":
            manifest["oracle_metrics"]["arms"]["EXECUTABLE_BASE"]["ending_equity_jpy_micros"] += 1
            manifest["oracle_metrics"]["metrics_sha256"] = embedded(manifest["oracle_metrics"], "metrics_sha256")
        else:
            manifest["authority"]["live_authority"] = True
        manifest["oracle_root_sha256"] = embedded(manifest, "oracle_root_sha256")
        write_json(manifest_path, manifest)
        request["oracle_manifest"] = artifact(root, manifest_path, "oracle_manifest")
        reseal_oracle_commit(root, request)
        with pytest.raises(
            verifier.VerificationError,
            match="oracle metrics differ|oracle manifest differs",
        ):
            run_verifier(root, request)


def test_oracle_request_shared_artifact_mismatch_is_rejected(tmp_path: Path) -> None:
    request, _ = verifier_fixture(tmp_path)
    request["proposal"] = dict(request["proposal"])
    request["proposal"]["artifact_id"] = "source_blob"
    with pytest.raises(verifier.VerificationError):
        run_verifier(tmp_path, request)


def test_verifier_schema_freezes_every_artifact_role_and_path_grammar() -> None:
    schema = json.loads(VERIFIER_SCHEMA.read_text())
    roles = (
        "source_blob", "source_manifest", "proposal", "execution_policy",
        "inventory_policy", "accounting_policy", "evaluation_policy",
        "instrument_registry", "authority_policy", "oracle_request",
        "oracle_code_snapshot", "oracle_contract_snapshot",
        "oracle_schema_snapshot", "reference_code_snapshot",
        "reference_contract_snapshot", "oracle_intent", "oracle_commit",
        "oracle_ledger", "oracle_manifest",
    )
    for role in roles:
        role_schema = schema["$defs"][f"{role}_artifact"]
        assert role_schema["allOf"][1]["properties"]["artifact_id"]["const"] == role
        assert role in schema["required"]
        assert role in schema["properties"]
    pattern = schema["$defs"]["artifact"]["properties"]["relative_path"]["pattern"]
    assert re.fullmatch(pattern, "inputs/oracle_request.json")
    assert re.fullmatch(pattern, "inputs/./oracle_request.json") is None
    assert re.fullmatch(pattern, "inputs/.hidden") is None
    interface = schema["x-sealed-production-interface"]
    assert interface["entrypoint"] == "verify_sealed_bytes"
    assert tuple(interface["artifact_blob_roles_sorted"]) == tuple(sorted(roles))
    assert tuple(interface["oracle_release_roles_sorted"]) == (
        "code_bytes", "contract_bytes", "schema_bytes",
    )
    assert tuple(interface["reference_attestation_keys_sorted"]) == (
        "reference_code_sha256",
        "reference_contract_sha256",
        "reference_result_sha256",
    )
    result_schema = schema["$defs"]["reference_result_snapshot"]
    assert result_schema["additionalProperties"] is False
    assert "ledger_bytes_base64" in result_schema["required"]
    assert "ledger_bytes" not in result_schema["properties"]
    assert schema["$defs"]["verifier_commit"]["additionalProperties"] is False


def test_verifier_schema_freezes_exact_receipt_and_commit_outputs() -> None:
    schema = json.loads(VERIFIER_SCHEMA.read_text())
    receipt = schema["$defs"]["verifier_receipt"]
    assert receipt["additionalProperties"] is False
    assert set(receipt["required"]) == set(verifier.VERIFIER_RECEIPT_KEYS)
    assert set(receipt["properties"]) == set(verifier.VERIFIER_RECEIPT_KEYS)
    authority = receipt["properties"]["authority"]
    assert authority["additionalProperties"] is False
    assert set(authority["required"]) == {
        key for key, _ in verifier._authority_items()
    }
    input_hashes = receipt["properties"]["input_artifact_sha256"]
    assert input_hashes["additionalProperties"] is False
    assert set(input_hashes["required"]) == set(
        verifier.SEALED_ARTIFACT_ROLES
    )
    binding = receipt["properties"]["verifier_release_content_binding"]
    assert binding["additionalProperties"] is False
    assert set(binding["required"]) == set(
        verifier.VERIFIER_RELEASE_BINDING_KEYS
    )
    commit = schema["$defs"]["verifier_commit"]
    assert commit["additionalProperties"] is False
    assert set(commit["required"]) == set(commit["properties"])


def test_oracle_snapshot_swap_is_rejected_by_exact_manifest(tmp_path: Path) -> None:
    request, _ = verifier_fixture(tmp_path)
    code_path = tmp_path / request["oracle_code_snapshot"]["relative_path"]
    code_path.write_bytes(code_path.read_bytes() + b"\n# mutation\n")
    request["oracle_code_snapshot"] = artifact(tmp_path, code_path, "oracle_code_snapshot")
    with pytest.raises(verifier.VerificationError, match="trusted Oracle release"):
        run_verifier(tmp_path, request)


@pytest.mark.parametrize(
    "label",
    ["reference_code_snapshot", "reference_contract_snapshot"],
)
def test_reference_snapshot_swap_is_rejected_by_trusted_fd_pair(
    tmp_path: Path,
    label: str,
) -> None:
    request, _ = verifier_fixture(tmp_path)
    snapshot_path = tmp_path / request[label]["relative_path"]
    snapshot_path.write_bytes(snapshot_path.read_bytes() + b"\n")
    request[label] = artifact(tmp_path, snapshot_path, label)
    with pytest.raises(
        verifier.VerificationError,
        match="trusted reference release FDs differ from injection|differs from trusted reference release FD",
    ):
        run_verifier(tmp_path, request)


def test_fully_resealed_unpinned_oracle_release_is_rejected(tmp_path: Path) -> None:
    request, _ = verifier_fixture(tmp_path)
    code_path = tmp_path / request["oracle_code_snapshot"]["relative_path"]
    code_path.write_bytes(code_path.read_bytes() + b"\n# coordinated malicious release\n")
    request["oracle_code_snapshot"] = artifact(
        tmp_path, code_path, "oracle_code_snapshot"
    )
    request_sha = request["oracle_request"]["sha256"]
    contract_sha = request["oracle_contract_snapshot"]["sha256"]
    schema_sha = request["oracle_schema_snapshot"]["sha256"]
    transaction_id = digest(canonical({
        "request_sha256": request_sha,
        "code_sha256": request["oracle_code_snapshot"]["sha256"],
        "contract_sha256": contract_sha,
        "schema_sha256": schema_sha,
    }))
    intent_path = tmp_path / request["oracle_intent"]["relative_path"]
    intent = {
        "schema_version": 1,
        "transaction_id": transaction_id,
        "request_sha256": request_sha,
        "code_sha256": request["oracle_code_snapshot"]["sha256"],
        "contract_sha256": contract_sha,
        "schema_sha256": schema_sha,
    }
    write_json(intent_path, intent)
    request["oracle_intent"] = artifact(tmp_path, intent_path, "oracle_intent")
    manifest_path = tmp_path / request["oracle_manifest"]["relative_path"]
    manifest = json.loads(manifest_path.read_text())
    manifest["oracle_release_content_binding"]["code_sha256"] = request["oracle_code_snapshot"]["sha256"]
    manifest["oracle_root_sha256"] = embedded(manifest, "oracle_root_sha256")
    write_json(manifest_path, manifest)
    request["oracle_manifest"] = artifact(tmp_path, manifest_path, "oracle_manifest")
    commit_path = tmp_path / request["oracle_commit"]["relative_path"]
    commit = json.loads(commit_path.read_text())
    commit["transaction_id"] = transaction_id
    write_json(commit_path, commit)
    request["oracle_commit"] = artifact(tmp_path, commit_path, "oracle_commit")
    reseal_oracle_commit(tmp_path, request)
    with pytest.raises(verifier.VerificationError, match="trusted Oracle release"):
        run_verifier(tmp_path, request)


@pytest.mark.parametrize(
    "writable_label",
    [
        "verifier_code", "verifier_schema", "oracle_code",
        "oracle_contract", "oracle_schema", "reference_code",
        "reference_contract",
    ],
)
def test_all_runtime_release_fds_must_be_read_only(
    tmp_path: Path, writable_label: str
) -> None:
    request, _ = verifier_fixture(tmp_path)
    paths = {
        "verifier_code": VERIFIER_PATH,
        "verifier_schema": VERIFIER_SCHEMA,
        "oracle_code": ORACLE_PATH,
        "oracle_contract": ORACLE_CONTRACT,
        "oracle_schema": ORACLE_SCHEMA,
        "reference_code": REFERENCE_PATH,
        "reference_contract": REFERENCE_CONTRACT,
    }
    descriptors = {
        label: os.open(
            path,
            (os.O_RDWR if label == writable_label else os.O_RDONLY)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        for label, path in paths.items()
    }
    input_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    output_fd = os.open(
        verifier_output_root(tmp_path),
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        with pytest.raises(verifier.VerificationError, match="read-only"):
            verifier.verify_from_fds(
                canonical(request) + b"\n",
                input_root_fd=input_fd,
                output_root_fd=output_fd,
                code_fd=descriptors["verifier_code"],
                schema_fd=descriptors["verifier_schema"],
                oracle_code_fd=descriptors["oracle_code"],
                oracle_contract_fd=descriptors["oracle_contract"],
                oracle_schema_fd=descriptors["oracle_schema"],
                reference_code_fd=descriptors["reference_code"],
                reference_contract_fd=descriptors["reference_contract"],
                _test_reference_replay=reference.replay_reference,
                _test_reference_code_bytes=REFERENCE_PATH.read_bytes(),
                _test_reference_contract_bytes=REFERENCE_CONTRACT.read_bytes(),
            )
    finally:
        os.close(output_fd)
        os.close(input_fd)
        for descriptor in descriptors.values():
            os.close(descriptor)


def test_direct_verifier_rejects_same_input_output_inode(tmp_path: Path) -> None:
    request, _ = verifier_fixture(tmp_path)
    with pytest.raises(
        verifier.VerificationError, match="distinct directory inodes"
    ):
        verifier.verify(
            request,
            trusted_input_root=tmp_path,
            trusted_output_root=tmp_path,
            reference_replay=reference.replay_reference,
            reference_code_bytes=REFERENCE_PATH.read_bytes(),
            reference_contract_bytes=REFERENCE_CONTRACT.read_bytes(),
        )


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b'{"schema_version":2,"schema_version":2}\n', "duplicate"),
        (b'{"schema_version":-0}\n', "negative zero"),
        (b'{"schema_version":2, "x":1}\n', "canonical"),
    ],
)
def test_verifier_strict_decoder(raw: bytes, message: str) -> None:
    with pytest.raises(verifier.VerificationError, match=message):
        verifier.strict_json(raw, "attack")


def test_verifier_recursively_rejects_producer_self_reported_fields() -> None:
    with pytest.raises(verifier.VerificationError, match="producer outcome field forbidden"):
        verifier._reject_producer_fields({
            "safe": [{"nested": {"Pr_O-FiT": 1}}],
        })


def test_mutable_public_authority_cannot_enable_live_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, _ = verifier_fixture(tmp_path)
    monkeypatch.setitem(verifier.AUTHORITY, "live_authority", True)
    monkeypatch.setitem(verifier.AUTHORITY, "external_orders", 99)
    receipt = run_verifier(tmp_path, request)["receipt"]
    assert receipt["authority"]["live_authority"] is False
    assert type(receipt["authority"]["live_authority"]) is bool
    assert receipt["authority"]["external_orders"] == 0
    assert type(receipt["authority"]["external_orders"]) is int


@pytest.mark.parametrize(
    ("field", "wrong_value", "message"),
    [
        ("live_authority", 0, "exact boolean mismatch"),
        ("external_orders", False, "exact integer mismatch"),
    ],
)
def test_authority_validation_rejects_bool_integer_aliases(
    field: str,
    wrong_value: object,
    message: str,
) -> None:
    authority = dict(verifier._authority_items())
    authority[field] = wrong_value
    with pytest.raises(verifier.VerificationError, match=message):
        verifier._validate_authority_exact(authority, "attack authority")


def test_source_manifest_timestamp_rejects_bool_equal_to_integer_one() -> None:
    registry_payload_value = registry_payload()
    registry = verifier._validate_registry(registry_payload_value)
    event = {
        "schema_version": 1,
        "provider_id": PROVIDER,
        "instrument": "USD_JPY",
        "bid_ticks": 15_000,
        "ask_ticks": 15_002,
        "tick_scale": 100,
        "source_ts_ns": 1,
        "arrival_ts_ns": 1,
        "provider_event_id": "USD_JPY-1",
        "sequence": 1,
        "heartbeat": False,
        "quality_flags": [],
    }
    blob = canonical(event) + b"\n"
    manifest = seal({
        "schema_version": 2,
        "source_bytes_sha256": digest(blob),
        "source_size_bytes": len(blob),
        "event_count": 1,
        # bool is numerically equal to 1 in Python; strict schemas must reject it.
        "first_source_ts_ns": True,
        "last_source_ts_ns": True,
        "provider_allowlist": [PROVIDER],
        "instrument_registry_sha256": registry_payload_value["registry_sha256"],
        "stream_policies": [{
            "provider_id": PROVIDER,
            "instrument": "USD_JPY",
            "sequence_required": True,
            "first_sequence": 1,
            "last_sequence": 1,
            "event_count": 1,
            "max_source_gap_ns": 1,
            "max_arrival_gap_ns": 1,
        }],
        "lossless": True,
    }, "manifest_sha256")
    with pytest.raises(verifier.VerificationError, match="first source timestamp"):
        verifier._parse_source(blob, manifest, registry_payload_value, registry)


def test_dangling_receipt_symlink_is_rejected(tmp_path: Path) -> None:
    request, _ = verifier_fixture(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-missing-outside"
    (verifier_output_root(tmp_path) / "verifier_output").symlink_to(
        outside, target_is_directory=True
    )
    with pytest.raises(verifier.VerificationError, match="output leaf"):
        run_verifier(tmp_path, request)
    assert not outside.exists()


def test_verifier_idempotent_exact_readback(tmp_path: Path) -> None:
    request, _ = verifier_fixture(tmp_path)
    first = run_verifier(tmp_path, request)
    second = run_verifier(tmp_path, request)
    assert first["receipt"]["verifier_receipt_sha256"] == second["receipt"]["verifier_receipt_sha256"]


def test_reference_receipt_link_tamper_survives_reseal_but_is_rejected(
    tmp_path: Path,
) -> None:
    request, _ = verifier_fixture(tmp_path)
    run_verifier(tmp_path, request)
    output = verifier_output_root(tmp_path) / "verifier_output"
    receipt_path = output / "verifier_receipt.json"
    commit_path = output / "COMMIT.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["reference_code_sha256"] = "e" * 64
    receipt["verifier_receipt_sha256"] = embedded(
        receipt, "verifier_receipt_sha256"
    )
    write_json(receipt_path, receipt)
    receipt_bytes = receipt_path.read_bytes()
    commit = json.loads(commit_path.read_text())
    commit.update({
        "receipt_sha256": digest(receipt_bytes),
        "receipt_size_bytes": len(receipt_bytes),
        "verifier_receipt_sha256": receipt["verifier_receipt_sha256"],
    })
    write_json(commit_path, commit)
    with pytest.raises(verifier.VerificationError, match="verifier output binding mismatch"):
        run_verifier(tmp_path, request)


def test_verifier_receipt_only_stage_recovers_without_stale_lock_poison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = verifier_fixture(tmp_path)
    original = verifier._write_file_at

    def fail_before_commit(directory_fd: int, name: str, data: bytes) -> None:
        if name == "COMMIT.json":
            raise OSError("injected verifier commit fault")
        original(directory_fd, name, data)

    monkeypatch.setattr(verifier, "_write_file_at", fail_before_commit)
    with pytest.raises(OSError, match="injected"):
        run_verifier(tmp_path, request)
    monkeypatch.setattr(verifier, "_write_file_at", original)
    result = run_verifier(tmp_path, request)
    assert result["receipt"]["status"] == "VERIFIED_ACCOUNTING_ONLY"
    assert (
        verifier_output_root(tmp_path) / "verifier_output" / "COMMIT.json"
    ).is_file()


def test_dangling_child_in_recoverable_verifier_stage_is_quarantined_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = verifier_fixture(tmp_path)
    stage = leave_recoverable_verifier_stage(tmp_path, request, monkeypatch)
    poisoned_child = stage / "verifier_receipt.json"
    poisoned_child.unlink()
    poisoned_child.symlink_to("missing-stage-child")
    publish_root = verifier_output_root(tmp_path)
    lock_path = publish_root / ".verifier_output.lock"
    lock_before = lock_path.stat()

    with pytest.raises(
        verifier.VerificationError,
        match="FAILED_VISIBLE_PARTIAL_VERIFIER_OUTPUT",
    ):
        run_verifier(tmp_path, request)

    failed = list(publish_root.glob(".verifier_output.*.failed"))
    assert len(failed) == 1
    failed_info = failed[0].stat()
    assert not stage.exists()
    assert (failed[0] / "verifier_receipt.json").is_symlink()
    assert os.readlink(failed[0] / "verifier_receipt.json") == "missing-stage-child"
    assert not (publish_root / "verifier_output").exists()
    lock_after_quarantine = lock_path.stat()
    assert (lock_after_quarantine.st_dev, lock_after_quarantine.st_ino) == (
        lock_before.st_dev,
        lock_before.st_ino,
    )
    assert lock_after_quarantine.st_nlink == 1
    assert lock_after_quarantine.st_mode & 0o777 == 0o600

    with pytest.raises(
        verifier.VerificationError,
        match="prior partial verifier output failure is preserved",
    ):
        run_verifier(tmp_path, request)

    failed_after_retry = list(publish_root.glob(".verifier_output.*.failed"))
    assert failed_after_retry == failed
    assert (failed_after_retry[0].stat().st_dev, failed_after_retry[0].stat().st_ino) == (
        failed_info.st_dev,
        failed_info.st_ino,
    )
    assert not stage.exists()
    assert not (publish_root / "verifier_output").exists()


def test_native_recovered_verifier_destination_collision_preserves_both_sides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = verifier_fixture(tmp_path)
    stage = leave_recoverable_verifier_stage(tmp_path, request, monkeypatch)
    marker = b"preserve-verifier-collision\n"

    def collide(root_fd: int, source: str, destination: str) -> None:
        del root_fd, source
        target = verifier_output_root(tmp_path) / destination
        target.mkdir(mode=0o700)
        (target / "marker.txt").write_bytes(marker)
        raise FileExistsError(destination)

    monkeypatch.setattr(verifier, "_RENAME_EXCLUSIVE", collide)
    with pytest.raises(FileExistsError):
        run_verifier(tmp_path, request)
    assert stage.is_dir()
    assert (
        verifier_output_root(tmp_path) / "verifier_output" / "marker.txt"
    ).read_bytes() == marker
    (verifier_output_root(tmp_path) / "verifier_output" / "marker.txt").unlink()
    (verifier_output_root(tmp_path) / "verifier_output").rmdir()

    monkeypatch.setattr(verifier, "_RENAME_EXCLUSIVE", exclusive_verifier_rename_at)
    recovered = run_verifier(tmp_path, request)
    assert recovered["receipt"]["status"] == "VERIFIED_ACCOUNTING_ONLY"


def test_native_recovered_verifier_substitution_fails_then_can_reverify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = verifier_fixture(tmp_path)
    stage = leave_recoverable_verifier_stage(tmp_path, request, monkeypatch)
    held_name = stage.name + ".held"

    def substitute_then_quarantine(
        root_fd: int, source: str, destination: str
    ) -> None:
        if destination == "verifier_output":
            os.rename(source, held_name, src_dir_fd=root_fd, dst_dir_fd=root_fd)
            os.mkdir(source, 0o700, dir_fd=root_fd)
            os.rename(source, destination, src_dir_fd=root_fd, dst_dir_fd=root_fd)
            return
        if destination.endswith(".failed"):
            os.rename(held_name, destination, src_dir_fd=root_fd, dst_dir_fd=root_fd)
            return
        raise AssertionError("unexpected verifier native rename target")

    monkeypatch.setattr(verifier, "_RENAME_EXCLUSIVE", substitute_then_quarantine)
    with pytest.raises(
        verifier.VerificationError,
        match="VERIFIER_STAGE_PATH_SUBSTITUTED",
    ):
        run_verifier(tmp_path, request)
    assert not (
        verifier_output_root(tmp_path) / "verifier_output" / "COMMIT.json"
    ).exists()
    held = verifier_output_root(tmp_path) / held_name
    assert held.is_dir()
    assert list(verifier_output_root(tmp_path).glob(".verifier_output.*.failed")) == []
    (verifier_output_root(tmp_path) / "verifier_output").rmdir()
    held.rename(stage)

    monkeypatch.setattr(verifier, "_RENAME_EXCLUSIVE", exclusive_verifier_rename_at)
    recovered = run_verifier(tmp_path, request)
    assert recovered["receipt"]["status"] == "VERIFIED_ACCOUNTING_ONLY"


def test_native_recovered_verifier_post_rename_root_fsync_fault_reverifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = verifier_fixture(tmp_path)
    leave_recoverable_verifier_stage(tmp_path, request, monkeypatch)
    original_fsync = os.fsync
    renamed = False
    failed = False

    def rename_then_flag(root_fd: int, source: str, destination: str) -> None:
        nonlocal renamed
        exclusive_verifier_rename_at(root_fd, source, destination)
        renamed = True

    def fail_first_post_rename_fsync(descriptor: int) -> None:
        nonlocal failed
        if renamed and not failed:
            failed = True
            raise OSError("injected verifier post-rename root fsync fault")
        original_fsync(descriptor)

    monkeypatch.setattr(verifier, "_RENAME_EXCLUSIVE", rename_then_flag)
    monkeypatch.setattr(verifier.os, "fsync", fail_first_post_rename_fsync)
    with pytest.raises(OSError, match="verifier post-rename root fsync fault"):
        run_verifier(tmp_path, request)
    assert failed is True
    assert (
        verifier_output_root(tmp_path) / "verifier_output" / "COMMIT.json"
    ).is_file()

    monkeypatch.setattr(verifier.os, "fsync", original_fsync)
    recovered = run_verifier(tmp_path, request)
    assert recovered["receipt"]["status"] == "VERIFIED_ACCOUNTING_ONLY"


def test_native_verifier_stage_path_substitution_never_returns_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = verifier_fixture(tmp_path)

    def substitute_then_rename(root_fd: int, source: str, destination: str) -> None:
        held = source + ".held"
        os.rename(source, held, src_dir_fd=root_fd, dst_dir_fd=root_fd)
        os.mkdir(source, 0o700, dir_fd=root_fd)
        os.rename(source, destination, src_dir_fd=root_fd, dst_dir_fd=root_fd)

    monkeypatch.setattr(verifier, "_RENAME_EXCLUSIVE", substitute_then_rename)
    with pytest.raises(
        verifier.VerificationError,
        match="published verifier inode mismatch",
    ):
        run_verifier(tmp_path, request)
    assert not (
        verifier_output_root(tmp_path) / "verifier_output" / "COMMIT.json"
    ).exists()


def test_native_verifier_final_path_substitution_never_returns_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = verifier_fixture(tmp_path)

    def exclusive_rename(root_fd: int, source: str, destination: str) -> None:
        if verifier._lstat_at(root_fd, destination) is not None:
            raise FileExistsError(destination)
        os.rename(source, destination, src_dir_fd=root_fd, dst_dir_fd=root_fd)

    original_validate = verifier._validate_receipt_output_fd
    validations = 0

    def validate_then_substitute(*args: object, **kwargs: object) -> dict:
        nonlocal validations
        result = original_validate(*args, **kwargs)
        validations += 1
        if validations == 2:
            publish_root = verifier_output_root(tmp_path)
            os.rename(
                publish_root / "verifier_output",
                publish_root / ".verifier_output.detached",
            )
            (publish_root / "verifier_output").mkdir(mode=0o700)
        return result

    monkeypatch.setattr(verifier, "_RENAME_EXCLUSIVE", exclusive_rename)
    monkeypatch.setattr(
        verifier,
        "_validate_receipt_output_fd",
        validate_then_substitute,
    )
    with pytest.raises(
        verifier.VerificationError,
        match="published verifier pathname changed during validation",
    ):
        run_verifier(tmp_path, request)
    assert validations == 2
    assert not (
        verifier_output_root(tmp_path) / "verifier_output" / "COMMIT.json"
    ).exists()


def test_extra_verifier_output_file_invalidates_exact_commit(tmp_path: Path) -> None:
    request, _ = verifier_fixture(tmp_path)
    run_verifier(tmp_path, request)
    (
        verifier_output_root(tmp_path) / "verifier_output" / "unexpected.txt"
    ).write_text("x")
    with pytest.raises(verifier.VerificationError, match="file set"):
        run_verifier(tmp_path, request)


def test_hardlinked_verifier_lock_is_rejected_without_victim_write(tmp_path: Path) -> None:
    request, _ = verifier_fixture(tmp_path)
    victim = tmp_path / "verifier-victim.txt"
    victim.write_bytes(b"preserve-verifier-victim\n")
    victim.chmod(0o600)
    os.link(victim, verifier_output_root(tmp_path) / ".verifier_output.lock")
    with pytest.raises(verifier.VerificationError, match="lock file"):
        run_verifier(tmp_path, request)
    assert victim.read_bytes() == b"preserve-verifier-victim\n"


def test_named_verifier_lock_replacement_never_quarantines_active_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = verifier_fixture(tmp_path)
    publish_root = verifier_output_root(tmp_path)
    original = verifier._write_file_at
    replaced = False

    def replace_lock_after_first_stage_write(
        directory_fd: int, name: str, data: bytes
    ) -> None:
        nonlocal replaced
        original(directory_fd, name, data)
        if not replaced:
            lock_path = publish_root / ".verifier_output.lock"
            lock_path.unlink()
            lock_path.write_bytes(b"")
            lock_path.chmod(0o600)
            replaced = True

    monkeypatch.setattr(
        verifier, "_write_file_at", replace_lock_after_first_stage_write
    )
    with pytest.raises(verifier.LockIdentityError):
        run_verifier(tmp_path, request)
    assert replaced is True
    assert len(list(publish_root.glob(".verifier_output.*.stage"))) == 1
    assert list(publish_root.glob(".verifier_output.*.failed")) == []
    assert not (publish_root / "verifier_output").exists()


def test_named_verifier_lock_replacement_at_commit_boundary_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = verifier_fixture(tmp_path)
    publish_root = verifier_output_root(tmp_path)
    renamed = False

    def rename_then_replace_lock(
        root_fd: int, source: str, destination: str
    ) -> None:
        nonlocal renamed
        exclusive_verifier_rename_at(root_fd, source, destination)
        lock_path = publish_root / ".verifier_output.lock"
        lock_path.unlink()
        lock_path.write_bytes(b"")
        lock_path.chmod(0o600)
        renamed = True

    monkeypatch.setattr(verifier, "_RENAME_EXCLUSIVE", rename_then_replace_lock)
    with pytest.raises(verifier.LockIdentityError):
        run_verifier(tmp_path, request)
    assert renamed is True
    assert (publish_root / "verifier_output" / "COMMIT.json").is_file()
    assert list(publish_root.glob(".verifier_output.*.failed")) == []


def test_verifier_month_end_mark_enters_running_peak_drawdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import datetime, timezone

    start = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()) * 1_000_000_000
    feb = int(datetime(2026, 2, 1, tzinfo=timezone.utc).timestamp()) * 1_000_000_000
    end = int(datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp()) * 1_000_000_000
    evaluation = {
        "period_start_ts_ns": start,
        "period_end_ts_ns": end,
        "initial_equity_jpy_micros": 100,
        "margin_notional_cap_jpy_micros": 1_000,
        "margin_rate_bps": 500,
        "max_gross_to_equity_bps": 10_000,
        "cvar_tail_bps": 500,
        "cluster_window_ns": verifier.DAY_NS,
    }
    risk = [
        {"arrival_ts_ns": start + 1, "marked_equity_jpy_micros": 120,
         "gross_notional_jpy_micros": 0, "required_margin_jpy_micros": 0,
         "free_margin_jpy_micros": 120, "margin_ratio_pass": True},
        {"arrival_ts_ns": feb + 1, "marked_equity_jpy_micros": 115,
         "gross_notional_jpy_micros": 0, "required_margin_jpy_micros": 0,
         "free_margin_jpy_micros": 115, "margin_ratio_pass": True},
    ]

    def boundary_equity(*args: object, **kwargs: object) -> int:
        del kwargs
        cutoff = args[1]
        if cutoff == feb - 1:
            return 80
        if cutoff == end - 1:
            return 115
        return 100

    monkeypatch.setattr(verifier, "_equity_at", boundary_equity)
    metrics = verifier._arm_metrics(
        [], [], risk, [], {}, {}, {}, evaluation, verifier.DAY_NS,
    )
    assert metrics["max_drawdown_jpy_micros"] == 40
    assert metrics["max_drawdown_ratio"] == "0.333333333333333334"
    assert metrics["minimum_marked_equity_jpy_micros"] == 80


def test_unsealed_python312_verifier_cli_rejects_missing_reference_callable(
    tmp_path: Path,
) -> None:
    request, _ = verifier_fixture(tmp_path)
    request["output_directory"] = "verifier_cli_output"
    request_path = write_json(tmp_path / "inputs" / "verifier_request.json", request)
    descriptors = [
        os.open(request_path, os.O_RDONLY),
        os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)),
        os.open(
            verifier_output_root(tmp_path),
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        ),
        os.open(VERIFIER_PATH, os.O_RDONLY),
        os.open(VERIFIER_SCHEMA, os.O_RDONLY),
        os.open(ORACLE_PATH, os.O_RDONLY),
        os.open(ORACLE_CONTRACT, os.O_RDONLY),
        os.open(ORACLE_SCHEMA, os.O_RDONLY),
        os.open(REFERENCE_PATH, os.O_RDONLY),
        os.open(REFERENCE_CONTRACT, os.O_RDONLY),
    ]
    try:
        completed = run_fd_cli(VERIFIER_PATH, [
            "--request-fd", str(descriptors[0]),
            "--input-root-fd", str(descriptors[1]),
            "--output-root-fd", str(descriptors[2]),
            "--code-fd", str(descriptors[3]),
            "--schema-fd", str(descriptors[4]),
            "--oracle-code-fd", str(descriptors[5]),
            "--oracle-contract-fd", str(descriptors[6]),
            "--oracle-schema-fd", str(descriptors[7]),
            "--reference-code-fd", str(descriptors[8]),
            "--reference-contract-fd", str(descriptors[9]),
        ], tuple(descriptors))
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    assert completed.returncode == 2, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)
    assert result["ok"] is False
    assert result["error_code"] == "VERIFIER_FAIL_CLOSED"


def test_verifier_import_graph_has_no_oracle_runner_process_or_network_dependency() -> None:
    source = VERIFIER_PATH.read_text()
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    forbidden = {
        "paper_research_jpy_oracle_v1", "paper_research_jpy_oracle_v2",
        "paper_research_double_entry_reference_v2",
        "paper_research_template_runner_v3", "paper_research_system_v3",
        "jpy_accounting_v2", "shadow_jpy_accounting_v1", "result_validator",
        "socket", "requests", "subprocess", "ctypes", "importlib",
        "builtins", "_posixsubprocess",
    }
    assert imports.isdisjoint(forbidden)
    assert not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id in {"float", "eval", "exec", "__import__"}
        for node in ast.walk(tree)
    )
    for removed_sealed_capability in (
        "_SEALED_REFERENCE_REPLAY",
        "_SEALED_REFERENCE_CODE_BYTES",
        "_SEALED_REFERENCE_CONTRACT_BYTES",
        "_SEALED_REFERENCE_CODE_SHA256",
        "_SEALED_REFERENCE_CONTRACT_SHA256",
        "_SEALED_RENAME_EXCLUSIVE",
    ):
        assert removed_sealed_capability not in source
    assert "def verify_sealed_bytes(" in source


def test_sealed_production_call_graph_has_no_capability_or_reflection_escape() -> None:
    tree = ast.parse(VERIFIER_PATH.read_text())
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    entrypoint = functions["verify_sealed_bytes"]
    assert [argument.arg for argument in entrypoint.args.args] == [
        "request_bytes",
        "artifact_blobs",
        "oracle_release_blobs",
        "reference_result_bytes",
        "reference_attestation",
    ]
    assert entrypoint.args.vararg is None
    assert entrypoint.args.kwarg is None
    assert entrypoint.args.defaults == []

    reachable = {"verify_sealed_bytes"}
    pending = ["verify_sealed_bytes"]
    while pending:
        name = pending.pop()
        for node in ast.walk(functions[name]):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                target = node.func.id
                if target in functions and target not in reachable:
                    reachable.add(target)
                    pending.append(target)

    forbidden_names = {
        "argparse", "fcntl", "os", "Path", "pathlib", "stat", "sys",
        "globals", "locals", "vars", "getattr", "setattr", "delattr",
        "eval", "exec", "compile", "open", "input", "breakpoint",
        "callable", "__import__", "builtins", "traceback",
    }
    forbidden_attributes = {
        "__builtins__", "__class__", "__code__", "__dict__",
        "__globals__", "__mro__", "__subclasses__", "f_back",
        "f_builtins", "f_globals", "f_locals", "tb_frame",
    }
    for name in sorted(reachable):
        for node in ast.walk(functions[name]):
            assert not isinstance(node, (ast.Import, ast.ImportFrom)), name
            if isinstance(node, ast.Name):
                assert node.id not in forbidden_names, (name, node.id)
            elif isinstance(node, ast.Attribute):
                assert node.attr not in forbidden_attributes, (name, node.attr)
                assert not (
                    isinstance(node.value, ast.Name)
                    and node.value.id == "sys"
                    and node.attr == "modules"
                ), name

    top_level_imports = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert top_level_imports.isdisjoint({
        "argparse", "fcntl", "os", "pathlib", "stat", "sys",
    })
