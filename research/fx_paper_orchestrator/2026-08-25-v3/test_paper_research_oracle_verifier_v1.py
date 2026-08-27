from __future__ import annotations

import ast
import json
import os
import subprocess
from pathlib import Path

import pytest

import paper_research_jpy_oracle_v1 as oracle
import paper_research_oracle_verifier_v1 as verifier
from test_paper_research_jpy_oracle_v1 import artifact, fixture, write_json


ROOT = Path(__file__).resolve().parent
PYTHON = Path("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3")


def verifier_request(tmp_path: Path) -> tuple[dict, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_path.resolve()
    oracle_request, _ = fixture(tmp_path / "producer")
    oracle_result = oracle.execute(oracle_request)
    manifest_path = Path(oracle_result["manifest_path"])
    ledger_path = Path(oracle_result["ledger_path"])
    request = {
        "schema_version": 1,
        "input_root": str(tmp_path),
        "output_root": str(tmp_path / "verified"),
        "oracle_manifest": artifact(manifest_path),
        "oracle_ledger": artifact(ledger_path),
        "source_blob": oracle_request["source_blob"],
        "source_manifest": oracle_request["source_manifest"],
        "proposal": oracle_request["proposal"],
        "execution_policy": oracle_request["execution_policy"],
        "inventory_policy": oracle_request["inventory_policy"],
        "accounting_policy": oracle_request["accounting_policy"],
        "evaluation_policy": oracle_request["evaluation_policy"],
        "receipt_name": "verifier_receipt.json",
    }
    Path(request["output_root"]).mkdir(parents=True)
    return request, oracle_result["manifest"]


def test_separate_verifier_reaggregates_oracle_ledger(tmp_path: Path) -> None:
    request, manifest = verifier_request(tmp_path)
    receipt = verifier.verify(request)
    assert receipt["status"] == "VERIFIED"
    assert receipt["oracle_root_sha256"] == manifest["oracle_root_sha256"]
    assert receipt["verified_oracle_metrics"] == manifest["oracle_metrics"]
    assert receipt["producer_metrics_used"] is False
    assert receipt["external_orders"] == 0
    assert receipt["terminal_inventory_mtm_jpy_micros"] == 0
    assert receipt["anchor_status"] == "LOCAL_REPRODUCIBLE"
    assert receipt["verifier_receipt_sha256"] == verifier.embedded_hash(
        receipt, "verifier_receipt_sha256"
    )


def test_verifier_rejects_tampered_self_reported_metrics_even_when_resealed(
    tmp_path: Path,
) -> None:
    request, _ = verifier_request(tmp_path)
    manifest_path = Path(request["oracle_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text())
    manifest["oracle_metrics"]["arms"]["EXECUTABLE_BASE"][
        "ending_equity_jpy_micros"
    ] += 999_999_999
    manifest["oracle_metrics"]["metrics_sha256"] = oracle.embedded_hash(
        manifest["oracle_metrics"], "metrics_sha256"
    )
    manifest["oracle_root_sha256"] = oracle.embedded_hash(
        manifest, "oracle_root_sha256"
    )
    write_json(manifest_path, manifest)
    request["oracle_manifest"] = artifact(manifest_path)
    with pytest.raises(verifier.VerificationError, match="metrics differ"):
        verifier.verify(request)


def test_verifier_rejects_ledger_source_reference_and_chain_tampering(
    tmp_path: Path,
) -> None:
    request, _ = verifier_request(tmp_path)
    ledger_path = Path(request["oracle_ledger"]["path"])
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    rows[0]["net_pnl_jpy_micros"] += 1
    write_json(ledger_path, rows[0])
    with pytest.raises(verifier.VerificationError, match="artifact binding mismatch"):
        verifier.verify(request)

    request_two, _ = verifier_request(tmp_path / "source-ref")
    ledger_two = Path(request_two["oracle_ledger"]["path"])
    rows = [json.loads(line) for line in ledger_two.read_text().splitlines()]
    rows[0]["entry_source_reference"]["provider_id"] = "FORGED"
    previous = "0" * 64
    for sequence, row in enumerate(rows, 1):
        row["ledger_sequence"] = sequence
        row["previous_hash"] = previous
        row["record_hash"] = verifier.embedded_hash(row, "record_hash")
        previous = row["record_hash"]
    ledger_two.write_bytes(b"".join(verifier.canonical_bytes(row) + b"\n" for row in rows))
    request_two["oracle_ledger"] = artifact(ledger_two)
    manifest_path = Path(request_two["oracle_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text())
    manifest["oracle_ledger_sha256"] = request_two["oracle_ledger"]["sha256"]
    manifest["oracle_ledger_terminal_hash"] = rows[-1]["record_hash"]
    manifest["oracle_root_sha256"] = verifier.embedded_hash(manifest, "oracle_root_sha256")
    write_json(manifest_path, manifest)
    request_two["oracle_manifest"] = artifact(manifest_path)
    with pytest.raises(verifier.VerificationError, match="source reference changed"):
        verifier.verify(request_two)


def test_verifier_recomputes_economics_instead_of_trusting_resealed_ledger(
    tmp_path: Path,
) -> None:
    request, _ = verifier_request(tmp_path)
    ledger_path = Path(request["oracle_ledger"]["path"])
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    target = next(row for row in rows if row["status"] == "FILLED_CLOSED")
    target["net_pnl_jpy_micros"] += 7_000_000
    target["realized_cost_jpy_micros"] -= 7_000_000
    previous = "0" * 64
    for sequence, row in enumerate(rows, 1):
        row["ledger_sequence"] = sequence
        row["previous_hash"] = previous
        row["record_hash"] = verifier.embedded_hash(row, "record_hash")
        previous = row["record_hash"]
    ledger_path.write_bytes(b"".join(verifier.canonical_bytes(row) + b"\n" for row in rows))
    request["oracle_ledger"] = artifact(ledger_path)
    manifest_path = Path(request["oracle_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text())
    evaluation = json.loads(Path(request["evaluation_policy"]["path"]).read_text())
    manifest["oracle_ledger_sha256"] = request["oracle_ledger"]["sha256"]
    manifest["oracle_ledger_terminal_hash"] = rows[-1]["record_hash"]
    manifest["oracle_metrics"] = verifier.reaggregate(rows, evaluation)
    manifest["oracle_root_sha256"] = verifier.embedded_hash(
        manifest, "oracle_root_sha256"
    )
    write_json(manifest_path, manifest)
    request["oracle_manifest"] = artifact(manifest_path)
    with pytest.raises(verifier.VerificationError, match="independent economic replay"):
        verifier.verify(request)


def test_verifier_paths_are_capability_root_bounded(tmp_path: Path) -> None:
    request, _ = verifier_request(tmp_path)
    request["input_root"] = str((tmp_path / "producer" / "oracle_output").resolve())
    with pytest.raises(verifier.VerificationError, match="escapes capability root"):
        verifier.verify(request)

    request_two, _ = verifier_request(tmp_path / "output")
    request_two["receipt_name"] = "../receipt.json"
    with pytest.raises(verifier.VerificationError, match="receipt name invalid"):
        verifier.verify(request_two)


def test_verifier_cli_runs_in_clean_python312_process(tmp_path: Path) -> None:
    request, _ = verifier_request(tmp_path)
    request_path = write_json(tmp_path / "verifier_request.json", request)
    completed = subprocess.run(
        [
            str(PYTHON), "-I", str(ROOT / "paper_research_oracle_verifier_v1.py"),
            str(request_path),
        ],
        cwd=ROOT,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["ok"] is True


def test_verifier_import_graph_is_independent() -> None:
    source = (ROOT / "paper_research_oracle_verifier_v1.py").read_text()
    tree = ast.parse(source)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    forbidden = {
        "paper_research_jpy_oracle_v1",
        "paper_research_template_runner_v3",
        "paper_research_system_v3",
        "jpy_accounting_v2",
        "shadow_jpy_accounting_v1",
        "result_validator",
        "socket",
        "requests",
        "subprocess",
    }
    assert imports.isdisjoint(forbidden)
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "float"
        for node in ast.walk(tree)
    )
