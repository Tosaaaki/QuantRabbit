from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import build_forward_shadow_core_checkpoint_v2 as evidence
import forward_shadow_core_v2 as shadow


ROOT = Path(__file__).resolve().parent
PYTHON = Path("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3")


def write_jsonl(
    path: Path,
    rows: list[dict],
    *,
    duplicate_index: int | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    evidence._write_jsonl(path, rows, duplicate_index=duplicate_index)
    return path


def valid_store(tmp_path: Path) -> tuple[shadow.ShadowStore, Path, list[dict]]:
    rows = evidence._valid_records()
    source = write_jsonl(tmp_path / "valid.jsonl", rows, duplicate_index=0)
    store = shadow.ShadowStore(tmp_path / "state")
    store.ingest(shadow.OfflineBBOFile(source))
    return store, source, rows


def decisions(action: str = "ENABLE", cap: float = 1.0) -> dict[str, dict]:
    return {
        arm: {
            "action": action,
            "pair_cap": cap,
            "currency_cap": cap,
            "provenance": "FROZEN_TEST_POLICY_NO_MODEL_CALL",
        }
        for arm in shadow.WORKER_ARMS
    }


def proposal(
    proposal_id: str = "SHADOW-P1",
    signal_id: str = "SHADOW-S1",
    minute: int = 240,
    max_age_seconds: int = 300,
) -> shadow.Proposal:
    return shadow.Proposal(
        proposal_id,
        signal_id,
        evidence._start_ns() + minute * 60 * 1_000_000_000,
        "EUR_USD",
        1,
        25_000.0,
        max_age_seconds,
        "1" * 64,
    )


def test_jsonl_and_csv_schema_are_local_exact_and_source_immutable(tmp_path: Path) -> None:
    rows = evidence._valid_records()[:4]
    jsonl = write_jsonl(tmp_path / "events.jsonl", rows)
    csv_path = tmp_path / "events.csv"
    evidence._write_csv(csv_path, rows)
    before = {
        path: (path.read_bytes(), path.stat().st_size, path.stat().st_mtime_ns)
        for path in (jsonl, csv_path)
    }
    json_result = shadow.validate_schema(jsonl)
    csv_result = shadow.validate_schema(csv_path)
    assert json_result["event_count"] == csv_result["event_count"] == 4
    assert json_result["lossless"] is True
    assert csv_result["lossless"] is True
    for path in (jsonl, csv_path):
        assert (path.read_bytes(), path.stat().st_size, path.stat().st_mtime_ns) == before[path]
    event = next(shadow.OfflineBBOFile(jsonl).events(shadow.OfflineBBOFile(jsonl).snapshot()))
    assert set(event.as_dict()) == {
        "schema_version", "provider_id", "instrument", "bid", "ask",
        "liquidity_optional", "source_ts_ns", "arrival_ts_ns",
        "provider_event_id", "sequence", "heartbeat",
        "raw_payload_sha256", "quality_flags",
    }
    assert len(event.raw_payload_sha256) == 64
    with pytest.raises(shadow.ShadowCoreError, match="LOCAL_FILE_REQUIRED"):
        shadow.OfflineBBOFile("https://example.invalid/bbo.jsonl")


def test_manifest_exact_duplicate_dedupe_restart_and_idempotent_reingest(tmp_path: Path) -> None:
    store, source, rows = valid_store(tmp_path)
    manifest = next(iter(store.manifests.values()))
    assert manifest["event_count"] == len(rows) + 1
    assert manifest["accepted_event_count"] == len(rows)
    assert manifest["exact_duplicate_count"] == 1
    assert manifest["lossless"] is True
    assert manifest["invalid_interval_count"] == 0
    assert manifest["manifest_sha256"] == shadow.embedded_hash(
        manifest, "manifest_sha256"
    )
    status_before = store.status()
    restarted = shadow.ShadowStore(store.state_dir)
    assert restarted.status()["state_sha256"] == status_before["state_sha256"]
    receipt = restarted.ingest(shadow.OfflineBBOFile(source))
    assert receipt["idempotent_reingest"] is True
    assert restarted.status()["raw_ledger_terminal_hash"] == status_before[
        "raw_ledger_terminal_hash"
    ]


def test_append_only_source_extension_and_prefix_mutation_fail_closed(tmp_path: Path) -> None:
    rows = evidence._valid_records()[:6]
    source = write_jsonl(tmp_path / "append.jsonl", rows[:2])
    store = shadow.ShadowStore(tmp_path / "state")
    store.ingest(shadow.OfflineBBOFile(source))
    write_jsonl(source, rows[:4])
    extension = store.ingest(shadow.OfflineBBOFile(source))["manifest"]
    assert extension["accepted_event_count"] == 2
    assert extension["exact_duplicate_count"] == 2
    mutated = [dict(item) for item in rows[:6]]
    mutated[0]["bid"] = "1.00000"
    mutated[0]["ask"] = "1.00012"
    write_jsonl(source, mutated)
    with pytest.raises(shadow.ShadowCoreError, match="SOURCE_PREFIX_CHANGED"):
        store.ingest(shadow.OfflineBBOFile(source))
    assert store.status()["halt_new_actions"] is True


def test_gap_heartbeat_reconnect_out_of_order_and_clock_reversal_halt(tmp_path: Path) -> None:
    gap_source = write_jsonl(tmp_path / "gap.jsonl", evidence._gap_records())
    gap_store = shadow.ShadowStore(tmp_path / "gap_state")
    manifest = gap_store.ingest(shadow.OfflineBBOFile(gap_source))["manifest"]
    reasons = {
        reason
        for row in gap_store.raw_ledger.rows
        for reason in row.get("quality_reasons", [])
    }
    assert manifest["lossless"] is False
    assert gap_store.status()["halt_new_actions"] is True
    assert {
        "SOURCE_OR_ARRIVAL_GAP", "HEARTBEAT_FAILURE", "RECONNECT_BOUNDARY"
    } <= reasons

    ordering_source = write_jsonl(
        tmp_path / "ordering.jsonl", evidence._ordering_records()
    )
    ordering_store = shadow.ShadowStore(tmp_path / "ordering_state")
    ordering_store.ingest(shadow.OfflineBBOFile(ordering_source))
    ordering_reasons = {
        reason
        for row in ordering_store.raw_ledger.rows
        for reason in row.get("quality_reasons", [])
    }
    assert {"OUT_OF_ORDER_EVENT", "CLOCK_REVERSAL"} <= ordering_reasons
    assert ordering_store.status()["halt_new_actions"] is True


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    [
        ({"schema_version": 99}, "UNKNOWN_SCHEMA"),
        ({"bid": "0", "ask": "1"}, "NONPOSITIVE_PRICE"),
        ({"bid": "1.2", "ask": "1.1"}, "SPREAD_INVERSION"),
    ],
)
def test_schema_price_and_spread_failures_are_visible(
    tmp_path: Path, mutation: dict, error_code: str
) -> None:
    row = dict(evidence._valid_records()[0])
    row.update(mutation)
    source = write_jsonl(tmp_path / f"{error_code}.jsonl", [row])
    store = shadow.ShadowStore(tmp_path / f"{error_code}_state")
    with pytest.raises(shadow.ShadowCoreError) as caught:
        store.ingest(shadow.OfflineBBOFile(source))
    assert caught.value.code == error_code
    manifest = store.manifests[shadow.sha256_bytes(source.read_bytes())]
    assert manifest["status"] == "FAILED"
    assert manifest["failure_code"] == error_code
    assert store.status()["halt_new_actions"] is True


def test_conflicting_duplicate_truncated_source_and_partial_ledger_fail_closed(
    tmp_path: Path,
) -> None:
    first = dict(evidence._valid_records()[0])
    second = dict(first)
    second["bid"] = "1.00000"
    second["ask"] = "1.00012"
    conflict = write_jsonl(tmp_path / "conflict.jsonl", [first, second])
    conflict_store = shadow.ShadowStore(tmp_path / "conflict_state")
    with pytest.raises(shadow.ShadowCoreError) as caught:
        conflict_store.ingest(shadow.OfflineBBOFile(conflict))
    assert caught.value.code == "CONFLICTING_DUPLICATE"
    assert conflict_store.status()["halt_new_actions"] is True

    truncated = tmp_path / "truncated.jsonl"
    truncated.write_text(json.dumps(first), encoding="utf-8")
    with pytest.raises(shadow.ShadowCoreError, match="TRUNCATED_SOURCE_RECORD"):
        shadow.validate_schema(truncated)

    partial_state = tmp_path / "partial_state"
    partial_state.mkdir()
    (partial_state / "raw_bbo_ledger.jsonl").write_bytes(b'{"partial":true}')
    with pytest.raises(shadow.ShadowCoreError, match="PARTIAL_LEDGER_RECORD"):
        shadow.ShadowStore(partial_state)


def test_checkpoint_and_manifest_tamper_fail_closed(tmp_path: Path) -> None:
    row = evidence._valid_records()[0]
    source = write_jsonl(tmp_path / "one.jsonl", [row])
    checkpoint_store = shadow.ShadowStore(tmp_path / "checkpoint_state")
    checkpoint_store.ingest(shadow.OfflineBBOFile(source))
    checkpoint = json.loads(checkpoint_store.checkpoint_path.read_text())
    checkpoint["raw_ledger_terminal_hash"] = "f" * 64
    checkpoint_store.checkpoint_path.write_text(
        json.dumps(checkpoint) + "\n", encoding="utf-8"
    )
    with pytest.raises(
        shadow.ShadowCoreError, match="CHECKPOINT_(MISMATCH|AHEAD_OF_LEDGER)"
    ):
        shadow.ShadowStore(checkpoint_store.state_dir)

    manifest_store = shadow.ShadowStore(tmp_path / "manifest_state")
    manifest_store.ingest(shadow.OfflineBBOFile(source))
    manifest_path = next(manifest_store.manifest_dir.glob("*.json"))
    manifest = json.loads(manifest_path.read_text())
    manifest["event_count"] += 1
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(shadow.ShadowCoreError, match="MANIFEST_MISMATCH"):
        shadow.ShadowStore(manifest_store.state_dir)


def test_completed_bars_are_causal_complete_and_future_invariant(tmp_path: Path) -> None:
    rows = evidence._valid_records()
    decision = evidence._start_ns() + 240 * 60 * 1_000_000_000
    causal_rows = [row for row in rows if row["source_ts_ns"] <= decision]
    causal_source = write_jsonl(tmp_path / "causal.jsonl", causal_rows)
    full_source = write_jsonl(tmp_path / "full.jsonl", rows)
    causal_store = shadow.ShadowStore(tmp_path / "causal_state")
    full_store = shadow.ShadowStore(tmp_path / "full_state")
    causal_store.ingest(shadow.OfflineBBOFile(causal_source))
    full_store.ingest(shadow.OfflineBBOFile(full_source))
    causal_bar = next(
        bar for bar in shadow.completed_bars(causal_store)
        if bar["instrument"] == "EUR_USD"
        and bar["timeframe"] == "M5"
        and bar["end_ts_ns"] == decision
    )
    full_bars = shadow.completed_bars(full_store)
    full_bar = next(
        bar for bar in full_bars
        if bar["instrument"] == "EUR_USD"
        and bar["timeframe"] == "M5"
        and bar["end_ts_ns"] == decision
    )
    assert causal_bar == full_bar
    assert causal_bar["burn_in_complete"] is True
    assert causal_bar["new_decision_or_fill_allowed"] is True
    assert {
        timeframe: sum(bar["timeframe"] == timeframe for bar in full_bars)
        for timeframe in shadow.TIMEFRAMES_SECONDS
    } == {"M5": 100, "M15": 32, "H1": 8, "H4": 2}
    for bar in full_bars:
        if bar["timeframe"] != "M5" and bar["valid"]:
            assert bar["m5_bundle_count"] == bar["required_m5_bundle_count"]


def test_shared_proposal_stream_cost_arms_latency_idempotence_and_scope(tmp_path: Path) -> None:
    store, _, _ = valid_store(tmp_path)
    frozen = decisions()
    item = proposal()
    receipt = shadow.route_shared_proposal(store, item, frozen)
    assert receipt["virtual_fill_count"] == 4
    assert receipt["same_content_addressed_proposal_all_arms"] is True
    fills = [
        row for row in store.virtual_ledger.rows
        if row.get("record_type") == "VIRTUAL_FILL"
    ]
    assert {row["proposal_sha256"] for row in fills} == {item.proposal_sha256}
    assert {row["worker_arm"] for row in fills} == set(shadow.WORKER_ARMS)
    assert {row["cost_arm"] for row in fills} == set(shadow.EXECUTION_SCENARIOS)
    for row in fills:
        minimum = row["decision_arrival_ts_ns"] + shadow.EXECUTION_SCENARIOS[
            row["cost_arm"]
        ]["latency_ns"]
        assert row["entry_arrival_ts_ns"] >= minimum
        assert row["entry_ts_ns"] > item.decision_ts_ns
        assert row["source_event_identity_sha256"]
        assert row["source_raw_payload_sha256"]
        assert row["external_order_count"] == 0
        assert row["actual_llm_called"] is False
        assert row["position_currency_inventory"]
        assert row["portfolio_currency_inventory_after"]
    before = store.status()
    idempotent = shadow.route_shared_proposal(store, item, frozen)
    assert idempotent["idempotent"] is True
    assert store.status()["virtual_ledger_terminal_hash"] == before[
        "virtual_ledger_terminal_hash"
    ]
    changed = decisions()
    changed["ACTUAL_LLM_INVENTORY_POLICY"]["provenance"] = "DIFFERENT"
    with pytest.raises(shadow.ShadowCoreError, match="CONFLICTING_PROPOSAL_ID"):
        shadow.route_shared_proposal(store, item, changed)


def test_llm_scope_actual_call_and_risk_caps_fail_closed(tmp_path: Path) -> None:
    store, _, _ = valid_store(tmp_path)
    with pytest.raises(shadow.ShadowCoreError, match="ACTUAL_LLM_CALL_NOT_AUTHORIZED"):
        shadow.route_shared_proposal(
            store, proposal("LLM", "LLM-S"), decisions(), actual_llm_called=True
        )
    forbidden = decisions()
    forbidden["ACTUAL_LLM_INVENTORY_POLICY"]["direction"] = -1
    with pytest.raises(shadow.ShadowCoreError, match="LLM_POLICY_SCOPE_VIOLATION"):
        shadow.route_shared_proposal(store, proposal("SCOPE", "SCOPE-S"), forbidden)

    cap_store, _, _ = valid_store(tmp_path / "cap")
    with pytest.raises(shadow.ShadowCoreError, match="PAIR_CAP_GUARD"):
        shadow.route_shared_proposal(
            cap_store, proposal("CAP", "CAP-S"), decisions(cap=0.1)
        )
    assert cap_store.status()["external_order_count"] == 0


def test_kill_after_expected_order_resumes_without_duplicate_or_missing_fill(
    tmp_path: Path,
) -> None:
    store, _, _ = valid_store(tmp_path)
    item = proposal("CRASH", "CRASH-S")
    original_append = store.virtual_ledger.append
    crashed = False

    def append_then_crash(value: dict) -> dict:
        nonlocal crashed
        row = original_append(value)
        if value.get("record_type") == "EXPECTED_ORDER" and not crashed:
            crashed = True
            raise RuntimeError("simulated kill after durable expected order")
        return row

    store.virtual_ledger.append = append_then_crash  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="simulated kill"):
        shadow.route_shared_proposal(store, item, decisions())
    restarted = shadow.ShadowStore(store.state_dir)
    receipt = shadow.route_shared_proposal(restarted, item, decisions())
    assert receipt["resumed_partial_execution"] is True
    orders = [
        row for row in restarted.virtual_ledger.rows
        if row.get("record_type") == "EXPECTED_ORDER"
        and row.get("proposal_sha256") == item.proposal_sha256
    ]
    fills = [
        row for row in restarted.virtual_ledger.rows
        if row.get("record_type") == "VIRTUAL_FILL"
        and row.get("proposal_sha256") == item.proposal_sha256
    ]
    assert len(orders) == len(fills) == 4
    assert len({_key(row) for row in orders}) == 4
    assert len({_key(row) for row in fills}) == 4
    second = shadow.route_shared_proposal(restarted, item, decisions())
    assert second["idempotent"] is True


def _key(row: dict) -> tuple[str, str]:
    return row["worker_arm"], row["cost_arm"]


def test_finite_max_age_terminal_liquidation_jpy_reconciliation_and_idempotence(
    tmp_path: Path,
) -> None:
    store, _, _ = valid_store(tmp_path)
    shadow.route_shared_proposal(store, proposal(), decisions())
    shadow.route_shared_proposal(
        store, proposal("SHADOW-P2", "SHADOW-S2", 245, 3600), decisions()
    )
    period_end = evidence._start_ns() + 250 * 60 * 1_000_000_000
    summary = shadow.finalize_period(store, period_end)
    assert summary["max_age_close_count"] == 4
    assert summary["terminal_liquidation_count"] == 4
    assert summary["terminal_inventory_mtm_jpy"] == 0.0
    assert summary["terminal_currency_inventory"] == {}
    assert summary["external_order_count"] == 0
    assert shadow._open_virtual_positions(store) == {}
    assert len(summary["worker_cost_arms"]) == 4
    for worker in shadow.WORKER_ARMS:
        base = summary["worker_cost_arms"][f"{worker}|EXECUTABLE_BASE"]
        adverse = summary["worker_cost_arms"][f"{worker}|ADVERSE_STRESS"]
        assert base["terminal_open_positions"] == adverse["terminal_open_positions"] == 0
        assert base["terminal_inventory_mtm_jpy"] == 0.0
        assert adverse["terminal_inventory_mtm_jpy"] == 0.0
        assert base["ending_equity_jpy"] >= adverse["ending_equity_jpy"]
    before = store.status()["virtual_ledger_terminal_hash"]
    assert shadow.finalize_period(store, period_end) == summary
    assert store.status()["virtual_ledger_terminal_hash"] == before


def test_finalization_is_an_immutable_cutoff_and_never_uses_future_activity(
    tmp_path: Path,
) -> None:
    store, _, _ = valid_store(tmp_path)
    shadow.route_shared_proposal(store, proposal(), decisions())
    period_end = evidence._start_ns() + 250 * 60 * 1_000_000_000
    summary = shadow.finalize_period(store, period_end)
    close_count = sum(
        row.get("record_type") == "VIRTUAL_CLOSE" for row in store.virtual_ledger.rows
    )
    with pytest.raises(shadow.ShadowCoreError, match="PERIOD_ALREADY_FINALIZED"):
        shadow.route_shared_proposal(
            store, proposal("AFTER", "AFTER-S", 245), decisions()
        )
    assert shadow.finalize_period(store, period_end) == summary
    assert sum(
        row.get("record_type") == "VIRTUAL_CLOSE" for row in store.virtual_ledger.rows
    ) == close_count

    cutoff_store, _, _ = valid_store(tmp_path / "past_cutoff")
    shadow.route_shared_proposal(
        cutoff_store, proposal("FUTURE", "FUTURE-S", 245), decisions()
    )
    past_cutoff = evidence._start_ns() + 242 * 60 * 1_000_000_000
    before = len(cutoff_store.virtual_ledger.rows)
    with pytest.raises(
        shadow.ShadowCoreError, match="FUTURE_ACTIVITY_BEYOND_FINALIZATION_CUTOFF"
    ):
        shadow.finalize_period(cutoff_store, past_cutoff)
    assert len(cutoff_store.virtual_ledger.rows) == before
    assert shadow._open_virtual_positions(cutoff_store)


def test_terminal_quote_older_than_90_seconds_fails_without_hiding_inventory(
    tmp_path: Path,
) -> None:
    store, _, _ = valid_store(tmp_path)
    shadow.route_shared_proposal(
        store, proposal("STALE", "STALE-S", 245, 100_000), decisions()
    )
    stale_cutoff = evidence._start_ns() + 250 * 60 * 1_000_000_000 + 91_000_000_000
    with pytest.raises(shadow.ShadowCoreError, match="TERMINAL_DATA_STALE"):
        shadow.finalize_period(store, stale_cutoff)
    assert shadow._open_virtual_positions(store)
    assert not any(
        row.get("record_type") == "PERIOD_FINALIZED" for row in store.virtual_ledger.rows
    )


def test_fill_latency_uses_arrival_time_and_completed_bar_arrival_causality(
    tmp_path: Path,
) -> None:
    rows = evidence._valid_records()
    decision = evidence._start_ns() + 240 * 60 * 1_000_000_000
    eur_rows = [dict(row) for row in rows if row["instrument"] == "EUR_USD"]
    template = dict(eur_rows[-1])
    additions = []
    for suffix, source_delta, arrival_delta in (
        ("PRE", -1, 0),
        ("TOO_FAST", 50_000_000, 100_000_000),
    ):
        row = dict(template)
        row["source_ts_ns"] = decision + source_delta
        row["arrival_ts_ns"] = decision + arrival_delta
        row["provider_event_id"] = suffix
        additions.append(row)
    eur_rows.extend(additions)
    eur_rows.sort(key=lambda row: row["source_ts_ns"])
    for index, row in enumerate(eur_rows, 1):
        row["sequence"] = index
        row["provider_event_id"] = f"LAT-{index:04d}"
    combined = sorted(
        [row for row in rows if row["instrument"] != "EUR_USD"] + eur_rows,
        key=lambda row: (row["source_ts_ns"], row["instrument"]),
    )
    source = write_jsonl(tmp_path / "arrival.jsonl", combined)
    store = shadow.ShadowStore(tmp_path / "arrival_state")
    store.ingest(shadow.OfflineBBOFile(source))
    shadow.route_shared_proposal(store, proposal(), decisions())
    fills = [
        row for row in store.virtual_ledger.rows
        if row.get("record_type") == "VIRTUAL_FILL"
    ]
    by_arm = {row["cost_arm"]: row for row in fills if row["worker_arm"] == "BOT_ONLY"}
    assert by_arm["EXECUTABLE_BASE"]["entry_arrival_ts_ns"] >= decision + 500_000_000
    assert by_arm["ADVERSE_STRESS"]["entry_arrival_ts_ns"] >= decision + 1_500_000_000
    assert all(row["entry_ts_ns"] > decision for row in fills)
    assert not any(row["entry_arrival_ts_ns"] == decision + 100_000_000 for row in fills)


@pytest.mark.parametrize("payload", [b"", b'{"partial":true}', b"\xff\xfe\n"])
def test_ingest_snapshot_failures_are_durable_halts(
    tmp_path: Path, payload: bytes,
) -> None:
    store, _, _ = valid_store(tmp_path)
    source = tmp_path / f"bad-{sha256_for_test(payload)}.jsonl"
    source.write_bytes(payload)
    with pytest.raises(shadow.ShadowCoreError) as caught:
        store.ingest(shadow.OfflineBBOFile(source))
    assert caught.value.code in {"TRUNCATED_SOURCE_RECORD", "INVALID_UTF8_RECORD"}
    assert store.status()["halt_new_actions"] is True
    assert any(item["status"] == "FAILED" for item in store.manifests.values())
    assert any(
        row.get("record_type") == "BATCH_FAILURE" for row in store.raw_ledger.rows
    )
    with pytest.raises(shadow.ShadowCoreError, match="DATA_QUALITY_HALT"):
        shadow.route_shared_proposal(store, proposal(), decisions())


def sha256_for_test(value: bytes) -> str:
    return __import__("hashlib").sha256(value).hexdigest()[:12]


def test_source_symlink_is_rejected_and_records_a_durable_halt(tmp_path: Path) -> None:
    target = write_jsonl(tmp_path / "target.jsonl", evidence._valid_records()[:2])
    link = tmp_path / "link.jsonl"
    link.symlink_to(target)
    store = shadow.ShadowStore(tmp_path / "state")
    with pytest.raises(shadow.ShadowCoreError, match="SYMLINK_FORBIDDEN"):
        store.ingest(shadow.OfflineBBOFile(link))
    assert store.status()["halt_new_actions"] is True
    assert any(item["failure_code"] == "SYMLINK_FORBIDDEN" for item in store.manifests.values())


def test_same_bytes_with_only_mtime_change_is_idempotent(tmp_path: Path) -> None:
    store, source, _ = valid_store(tmp_path)
    before = source.stat().st_mtime_ns
    os.utime(source, ns=(before + 10_000_000_000, before + 10_000_000_000))
    receipt = store.ingest(shadow.OfflineBBOFile(source))
    assert receipt["idempotent_reingest"] is True
    assert receipt["manifest"]["source_mtime_ns"] == before


def test_manifest_ledger_checkpoint_and_semantic_reseal_are_cross_bound(
    tmp_path: Path,
) -> None:
    store, _, _ = valid_store(tmp_path / "base")
    base = store.state_dir

    manifest_only = tmp_path / "manifest_only"
    shutil.copytree(base / "batch_manifests", manifest_only / "batch_manifests")
    shutil.copytree(base / "source_blobs", manifest_only / "source_blobs")
    with pytest.raises(shadow.ShadowCoreError, match="MANIFEST_LEDGER_BINDING_MISMATCH"):
        shadow.ShadowStore(manifest_only)

    ledger_only = tmp_path / "ledger_only"
    shutil.copytree(base, ledger_only)
    shutil.rmtree(ledger_only / "batch_manifests")
    with pytest.raises(shadow.ShadowCoreError, match="MANIFEST_LEDGER_BINDING_MISMATCH"):
        shadow.ShadowStore(ledger_only)

    checkpoint_only = tmp_path / "checkpoint_only"
    checkpoint_only.mkdir()
    shutil.copy2(base / "restart_checkpoint.json", checkpoint_only / "restart_checkpoint.json")
    with pytest.raises(
        shadow.ShadowCoreError, match="CHECKPOINT_(MISMATCH|AHEAD_OF_LEDGER)"
    ):
        shadow.ShadowStore(checkpoint_only)

    semantic = tmp_path / "semantic"
    shutil.copytree(base, semantic)
    manifest_path = next((semantic / "batch_manifests").glob("*.json"))
    payload = json.loads(manifest_path.read_text())
    payload["event_count"] = 999
    payload["manifest_sha256"] = shadow.embedded_hash(payload, "manifest_sha256")
    manifest_path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    with pytest.raises(shadow.ShadowCoreError, match="SOURCE_MANIFEST_COUNT_MISMATCH"):
        shadow.ShadowStore(semantic)

    orphan = tmp_path / "orphan_blob"
    shutil.copytree(base, orphan)
    (orphan / "source_blobs" / f"{'f' * 64}.blob").write_bytes(b"orphan")
    with pytest.raises(shadow.ShadowCoreError, match="SOURCE_BLOB_BINDING_MISMATCH"):
        shadow.ShadowStore(orphan)


def test_kill_after_first_raw_row_never_allows_same_name_prefix_replacement(
    tmp_path: Path,
) -> None:
    rows = evidence._valid_records()[:4]
    source = write_jsonl(tmp_path / "kill.jsonl", rows)
    state = tmp_path / "kill_state"
    store = shadow.ShadowStore(state)
    original = store.raw_ledger.append

    def durable_first_row_then_kill(value: dict) -> dict:
        row = original(value)
        if value.get("record_type") == "BBO_EVENT":
            raise RuntimeError("KILLED_AFTER_FIRST_RAW_ROW")
        return row

    store.raw_ledger.append = durable_first_row_then_kill  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="KILLED_AFTER_FIRST_RAW_ROW"):
        store.ingest(shadow.OfflineBBOFile(source))
    replacement = [dict(item) for item in rows]
    replacement[0]["bid"] = "1.00000"
    replacement[0]["ask"] = "1.00012"
    write_jsonl(source, replacement)
    with pytest.raises(shadow.ShadowCoreError, match="MANIFEST_LEDGER_BINDING_MISMATCH"):
        shadow.ShadowStore(state)


@pytest.mark.parametrize(
    "relative",
    [
        "raw_bbo_ledger.jsonl",
        "proposal_stream_ledger.jsonl",
        "virtual_execution_ledger.jsonl",
        "restart_checkpoint.json",
        "batch_manifests",
        "source_blobs",
    ],
)
def test_every_state_path_rejects_symlinks(tmp_path: Path, relative: str) -> None:
    store, _, _ = valid_store(tmp_path / "base")
    shadow.route_shared_proposal(store, proposal(), decisions())
    target = store.state_dir / relative
    external = tmp_path / f"external-{relative.replace('/', '-') }"
    if target.is_dir():
        shutil.move(target, external)
        target.symlink_to(external, target_is_directory=True)
    else:
        shutil.copy2(target, external)
        target.unlink()
        target.symlink_to(external)
    with pytest.raises(shadow.ShadowCoreError):
        shadow.ShadowStore(store.state_dir)


def test_state_root_symlink_is_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "linked_state"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(shadow.ShadowCoreError, match="SECURE_DIRECTORY_REQUIRED"):
        shadow.ShadowStore(link)


@pytest.mark.parametrize("kind", ["manifest", "source_blob"])
def test_individual_manifest_and_source_blob_symlinks_are_rejected(
    tmp_path: Path, kind: str,
) -> None:
    store, _, _ = valid_store(tmp_path / "base")
    if kind == "manifest":
        target = next(store.manifest_dir.glob("*.json"))
    else:
        target = next(store.source_blob_dir.glob("*.blob"))
    external = tmp_path / f"external-{kind}"
    shutil.copy2(target, external)
    target.unlink()
    target.symlink_to(external)
    with pytest.raises(shadow.ShadowCoreError):
        shadow.ShadowStore(store.state_dir)


def test_runtime_has_file_only_dependency_and_zero_external_capability() -> None:
    imports = set()
    trees = []
    for runtime in ("forward_shadow_core_v2.py", "shadow_jpy_accounting_v1.py"):
        tree = ast.parse((ROOT / runtime).read_text(encoding="utf-8"))
        trees.append(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
    assert imports <= {
        "__future__", "argparse", "bisect", "calendar", "csv", "dataclasses", "datetime",
        "decimal", "hashlib", "io", "json", "shadow_jpy_accounting_v1", "math", "os",
        "pathlib", "re", "stat", "tempfile", "typing",
    }
    assert not imports & {
        "aiohttp", "boto3", "http", "httpx", "oandapyV20", "requests",
        "socket", "urllib", "websocket",
    }
    assert "subprocess" not in imports
    assert not any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
        and node.attr in {"environ", "getenv", "system", "popen", "spawnl", "spawnv"}
        for tree in trees for node in ast.walk(tree)
    )
    assert shadow.AUTHORITY == {
        "paper_only": True,
        "live_authority": False,
        "broker_account_access": False,
        "credential_access": False,
        "order_endpoint": False,
        "external_orders": 0,
        "deploy": False,
        "external_config_mutation": False,
    }


def test_cli_validate_ingest_resume_status_and_finalize(tmp_path: Path) -> None:
    source = write_jsonl(tmp_path / "cli.jsonl", evidence._valid_records()[:4])
    state = tmp_path / "cli_state"
    commands = [
        ["validate-schema", str(source)],
        ["ingest-batch", str(source), "--state-dir", str(state)],
        ["resume", "--state-dir", str(state)],
        ["status", "--state-dir", str(state)],
        [
            "finalize-period", "--state-dir", str(state),
            "--period-end-ts-ns", str(evidence._start_ns() + 60_000_000_000),
        ],
    ]
    environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8"}
    for command in commands:
        completed = subprocess.run(
            [str(PYTHON), str(ROOT / "forward_shadow_core_v2.py"), *command],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        assert payload["ok"] is True
        result = payload["result"]
        if isinstance(result, dict) and "external_order_count" in result:
            assert result["external_order_count"] == 0
