from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import paper_research_jpy_oracle_v1 as oracle


ROOT = Path(__file__).resolve().parent
PYTHON = Path("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3")
START_NS = 1_767_225_600_000_000_000  # 2026-01-01T00:00:00Z


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def seal(value: dict, field: str) -> dict:
    value[field] = oracle.embedded_hash(value, field)
    return value


def write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")
    return path


def artifact(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def source_rows() -> list[dict]:
    offsets = (0, 1, 2, 302, 360, 361, 362, 662, 900)
    result = []
    for instrument in ("EUR_USD", "USD_JPY"):
        for sequence, seconds in enumerate(offsets, 1):
            if instrument == "EUR_USD":
                bid = 110_000 + sequence * 8
                ask = bid + 12
                scale = 100_000
            else:
                bid = 15_000 + sequence * 3
                ask = bid + 2
                scale = 100
            source = START_NS + seconds * 1_000_000_000
            result.append({
                "schema_version": 1,
                "provider_id": "ORACLE_FIXTURE",
                "instrument": instrument,
                "bid_ticks": bid,
                "ask_ticks": ask,
                "tick_scale": scale,
                "source_ts_ns": source,
                "arrival_ts_ns": source + 100_000_000,
                "provider_event_id": f"{instrument}-{sequence}",
                "sequence": sequence,
                "heartbeat": False,
                "quality_flags": [],
            })
    return sorted(result, key=lambda row: (
        row["source_ts_ns"], row["instrument"], row["sequence"]
    ))


def write_source(root: Path, rows: list[dict] | None = None) -> tuple[Path, Path]:
    rows = source_rows() if rows is None else rows
    blob = root / "source.jsonl"
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob_bytes = b"".join(canonical(row) + b"\n" for row in rows)
    blob.write_bytes(blob_bytes)
    manifest = seal({
        "schema_version": 1,
        "source_bytes_sha256": hashlib.sha256(blob_bytes).hexdigest(),
        "source_size_bytes": len(blob_bytes),
        "event_count": len(rows),
        "first_source_ts_ns": min(row["source_ts_ns"] for row in rows),
        "last_source_ts_ns": max(row["source_ts_ns"] for row in rows),
    }, "manifest_sha256")
    return blob, write_json(root / "source_manifest.json", manifest)


def fixture(root: Path, *, proposal_mutation: dict | None = None) -> tuple[dict, dict]:
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    blob, source_manifest = write_source(root)
    proposal = {
        "schema_version": 1,
        "candidate_key": "ORACLE-FIXTURE-CANDIDATE",
        "rows": [
            {
                "proposal_ordinal": 1,
                "decision_source_ts_ns": START_NS,
                "decision_arrival_ts_ns": START_NS + 100_000_000,
                "available_at_ns": START_NS + 100_000_000,
                "instrument": "EUR_USD",
                "direction": 1,
                "notional_jpy_micros": 28_000 * 1_000_000,
                "max_age_ns": 300_000_000_000,
                "worker_key": "EUR_HIERARCHICAL",
                "action": "ENTER",
            },
            {
                "proposal_ordinal": 2,
                "decision_source_ts_ns": START_NS + 360_000_000_000,
                "decision_arrival_ts_ns": START_NS + 360_100_000_000,
                "available_at_ns": START_NS + 360_100_000_000,
                "instrument": "USD_JPY",
                "direction": -1,
                "notional_jpy_micros": 28_000 * 1_000_000,
                "max_age_ns": 300_000_000_000,
                "worker_key": "JPY_COST_TO_MFE",
                "action": "ENTER",
            },
        ],
    }
    if proposal_mutation:
        proposal["rows"][0].update(proposal_mutation)
    seal(proposal, "proposal_sha256")
    execution = seal({
        "schema_version": 1,
        "policy_id": "FROZEN_EXECUTION_POLICY_V1",
        "arms": {
            "RAW_SIGNAL": {
                "latency_ns": 0,
                "slippage_ticks_per_side": 0,
                "commission_ppm_per_side": 0,
                "financing_ppm_per_day": 0,
                "raw_mid": True,
            },
            "EXECUTABLE_BASE": {
                "latency_ns": 500_000_000,
                "slippage_ticks_per_side": 1,
                "commission_ppm_per_side": 2,
                "financing_ppm_per_day": 1,
                "raw_mid": False,
            },
            "ADVERSE_STRESS": {
                "latency_ns": 1_500_000_000,
                "slippage_ticks_per_side": 3,
                "commission_ppm_per_side": 6,
                "financing_ppm_per_day": 3,
                "raw_mid": False,
            },
        },
    }, "execution_policy_sha256")
    inventory = seal({
        "schema_version": 1,
        "policy_id": "FROZEN_INVENTORY_POLICY_V1",
        "max_gross_notional_jpy_micros": 200_000 * 1_000_000,
        "max_currency_notional_jpy_micros": 200_000 * 1_000_000,
        "max_open_positions": 4,
        "same_pair_collision": "REJECT_NEW",
        "terminal_liquidation": True,
    }, "inventory_policy_sha256")
    accounting = seal({
        "schema_version": 1,
        "policy_id": "FROZEN_ACCOUNTING_POLICY_V1",
        "jpy_micros_per_yen": 1_000_000,
        "base_microunits_per_unit": 1_000_000,
        "max_conversion_staleness_ns": 400_000_000_000,
        "supported_quote_currencies": ["CAD", "CHF", "JPY", "USD"],
        "asset_conversion_side": "BID",
        "liability_conversion_side": "ASK",
    }, "accounting_policy_sha256")
    evaluation = seal({
        "schema_version": 1,
        "policy_id": "FROZEN_EVALUATION_POLICY_V1",
        "period_start_ts_ns": START_NS,
        "period_end_ts_ns": START_NS + 900_100_000_000,
        "initial_equity_jpy_micros": 200_000 * 1_000_000,
        "margin_notional_cap_jpy_micros": 200_000 * 1_000_000,
        "cvar_tail_bps": 500,
        "holdout_state": "UNOPENED",
    }, "evaluation_policy_sha256")
    paths = {
        "proposal": write_json(root / "proposal.json", proposal),
        "execution_policy": write_json(root / "execution.json", execution),
        "inventory_policy": write_json(root / "inventory.json", inventory),
        "accounting_policy": write_json(root / "accounting.json", accounting),
        "evaluation_policy": write_json(root / "evaluation.json", evaluation),
    }
    request = {
        "schema_version": 1,
        "input_root": str(root),
        "output_root": str(root),
        "source_blob": artifact(blob),
        "source_manifest": artifact(source_manifest),
        **{name: artifact(path) for name, path in paths.items()},
        "output_directory": "oracle_output",
    }
    return request, {
        "proposal": proposal,
        "execution": execution,
        "inventory": inventory,
        "accounting": accounting,
        "evaluation": evaluation,
    }


def assert_no_float(value: object) -> None:
    if isinstance(value, float):
        raise AssertionError("float leaked into canonical oracle evidence")
    if isinstance(value, dict):
        for item in value.values():
            assert_no_float(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_float(item)


def test_independent_oracle_recomputes_all_arm_dispositions_and_risk(tmp_path: Path) -> None:
    request, _ = fixture(tmp_path)
    result = oracle.execute(request)
    manifest = result["manifest"]
    assert manifest["oracle_implementation"] == "INDEPENDENT_JPY_ORACLE_V1"
    assert manifest["producer_result_or_metrics_used"] is False
    assert manifest["oracle_ledger_row_count"] == 6
    assert manifest["anchor_status"] == "LOCAL_REPRODUCIBLE"
    assert manifest["external_orders"] == 0
    assert manifest["terminal_inventory_mtm_jpy_micros"] == 0
    metrics = manifest["oracle_metrics"]
    assert metrics["same_signal_ids_all_arms"] is True
    assert metrics["all_proposals_have_all_arm_dispositions"] is True
    assert metrics["action_label_contract_all_arms"] is True
    assert all(metrics["arms"][arm]["executed_count"] == 2 for arm in oracle.ARMS)
    assert all(metrics["arms"][arm]["terminal_open_positions"] == 0 for arm in oracle.ARMS)
    assert metrics["arms"]["EXECUTABLE_BASE"]["ending_equity_jpy_micros"] >= (
        metrics["arms"]["ADVERSE_STRESS"]["ending_equity_jpy_micros"]
    )
    assert_no_float(manifest)
    rows = [json.loads(line) for line in Path(result["ledger_path"]).read_text().splitlines()]
    assert len({row["record_hash"] for row in rows}) == len(rows)
    assert rows[0]["previous_hash"] == "0" * 64
    assert rows[-1]["record_hash"] == manifest["oracle_ledger_terminal_hash"]
    for row in rows:
        assert row["external_order_count"] == 0
        if row["status"] == "FILLED_CLOSED":
            assert row["entry_source_reference"]["source_event_sha256"]
            assert row["entry_source_reference"]["provider_id"] == "ORACLE_FIXTURE"
            assert row["units_micros"] > 0


@pytest.mark.parametrize(
    "field",
    ["signal_id", "fill_price", "path", "mfe", "mae", "pnl", "equity", "cvar"],
)
def test_ex_ante_proposal_recursively_rejects_identifiers_and_outcomes(
    tmp_path: Path, field: str
) -> None:
    request, _ = fixture(tmp_path, proposal_mutation={field: "forbidden"})
    with pytest.raises(oracle.OracleError, match="proposal outcome/identifier forbidden"):
        oracle.execute(request)


def test_exact_source_bytes_manifest_and_policy_hashes_are_mandatory(tmp_path: Path) -> None:
    request, _ = fixture(tmp_path)
    Path(request["source_blob"]["path"]).write_bytes(
        Path(request["source_blob"]["path"]).read_bytes() + b"\n"
    )
    with pytest.raises(oracle.OracleError, match="artifact (size|hash) mismatch"):
        oracle.execute(request)

    request_two, _ = fixture(tmp_path / "policy")
    policy_path = Path(request_two["execution_policy"]["path"])
    policy = json.loads(policy_path.read_text())
    policy["arms"]["EXECUTABLE_BASE"]["latency_ns"] += 1
    write_json(policy_path, policy)
    request_two["execution_policy"] = artifact(policy_path)
    with pytest.raises(oracle.OracleError, match="embedded hash mismatch"):
        oracle.execute(request_two)


def test_input_stream_clock_reversal_is_rejected_without_sorting_it_away(
    tmp_path: Path,
) -> None:
    request, _ = fixture(tmp_path)
    source_path = Path(request["source_blob"]["path"])
    rows = [json.loads(line) for line in source_path.read_text().splitlines()]
    eur_indices = [index for index, row in enumerate(rows) if row["instrument"] == "EUR_USD"]
    first, second = eur_indices[:2]
    rows[first], rows[second] = rows[second], rows[first]
    blob, manifest = write_source(tmp_path, rows)
    request["source_blob"] = artifact(blob)
    request["source_manifest"] = artifact(manifest)
    with pytest.raises(oracle.OracleError, match="input order is not strictly increasing"):
        oracle.execute(request)


def test_input_and_output_paths_are_capability_root_bounded(tmp_path: Path) -> None:
    root = tmp_path / "bounded"
    request, _ = fixture(root)
    outside = tmp_path / "outside.json"
    outside.write_bytes(Path(request["proposal"]["path"]).read_bytes())
    request["proposal"] = artifact(outside)
    with pytest.raises(oracle.OracleError, match="escapes capability root"):
        oracle.execute(request)

    request_two, _ = fixture(tmp_path / "symlink")
    original = Path(request_two["proposal"]["path"])
    link = original.with_name("proposal-link.json")
    link.symlink_to(original)
    request_two["proposal"] = artifact(original)
    request_two["proposal"]["path"] = str(link)
    with pytest.raises(oracle.OracleError, match="symlink"):
        oracle.execute(request_two)

    request_three, _ = fixture(tmp_path / "output")
    request_three["output_directory"] = "../escape"
    with pytest.raises(oracle.OracleError, match="output directory name invalid"):
        oracle.execute(request_three)


def test_oracle_generates_ids_and_long_short_pnl_is_linear(tmp_path: Path) -> None:
    request, _ = fixture(tmp_path)
    result = oracle.execute(request)
    manifest = result["manifest"]
    rows = [json.loads(line) for line in Path(result["ledger_path"]).read_text().splitlines()]
    by_arm = {
        (row["arm"], row["proposal_ordinal"]): row for row in rows
    }
    long_row = by_arm[("RAW_SIGNAL", 1)]
    short_row = by_arm[("RAW_SIGNAL", 2)]
    assert long_row["signal_id"] == by_arm[("EXECUTABLE_BASE", 1)]["signal_id"]
    assert short_row["signal_id"] == by_arm[("ADVERSE_STRESS", 2)]["signal_id"]
    assert long_row["signal_id"] != short_row["signal_id"]
    assert short_row["direction"] == -1
    assert short_row["gross_pnl_jpy_micros"] < 0  # fixture USDJPY rises
    assert "signal_id" not in json.loads(Path(request["proposal"]["path"]).read_text())["rows"][0]


def test_caps_produce_visible_dispositions_for_every_arm(tmp_path: Path) -> None:
    request, _ = fixture(tmp_path)
    inventory_path = Path(request["inventory_policy"]["path"])
    inventory = json.loads(inventory_path.read_text())
    inventory["max_gross_notional_jpy_micros"] = 1
    seal(inventory, "inventory_policy_sha256")
    write_json(inventory_path, inventory)
    request["inventory_policy"] = artifact(inventory_path)
    manifest = oracle.execute(request)["manifest"]
    for arm in oracle.ARMS:
        metric = manifest["oracle_metrics"]["arms"][arm]
        assert metric["executed_count"] == 0
        assert metric["disposition_counts"] == {"GROSS_CAP_REJECTED": 2}
        assert metric["proposal_count"] == 2


def test_clean_python312_cli_has_no_hidden_project_import_path(tmp_path: Path) -> None:
    request, _ = fixture(tmp_path)
    request_path = write_json(tmp_path / "request.json", request)
    completed = subprocess.run(
        [str(PYTHON), "-I", str(ROOT / "paper_research_jpy_oracle_v1.py"), str(request_path)],
        cwd=ROOT,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert Path(payload["manifest_path"]).is_file()


def test_oracle_import_graph_excludes_producer_and_accounting_modules() -> None:
    tree = ast.parse((ROOT / "paper_research_jpy_oracle_v1.py").read_text())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    forbidden = {
        "jpy_accounting_v2",
        "shadow_jpy_accounting_v1",
        "paper_research_template_runner_v3",
        "paper_research_orchestrator_v2",
        "paper_research_system_v3",
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


def test_legacy_oracle_coverage_cannot_be_promoted_by_this_oracle(tmp_path: Path) -> None:
    request, _ = fixture(tmp_path)
    manifest = oracle.execute(request)["manifest"]
    assert "legacy" not in manifest
    assert manifest["anchor_status"] != "EXTERNALLY_ANCHORED"
