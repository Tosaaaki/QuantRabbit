from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/run-dojo-ai-capital-recycle-train.py"
SPEC = importlib.util.spec_from_file_location(
    "dojo_ai_capital_recycle_train", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _packet(
    case_id: str, existing_side: str = "LONG", next_side: str = "SHORT"
) -> dict:
    body = {
        "id": case_id,
        "capacity_contract": "FIXED_HIERARCHICAL_GATE; NO_MODEL_ALLOCATION",
        "decision_price_basis": "DECISION_BAR_OPEN",
        "context_cutoff": "STRICTLY_BEFORE_DECISION_BAR",
        "horizon": "TP_OR_HARD_EXIT_PLUS_4H",
        "asset_a_existing": {"side": existing_side},
        "asset_b_new_candidate": {"side": next_side},
    }
    return {**body, "packet_canonical_sha256": MODULE.canonical_sha(body)}


def _cases() -> list[dict]:
    cases = []
    for index in range(MODULE.CELL_COUNT):
        packet = _packet(f"P{index + 1:02d}")
        cases.append(
            {
                "packet": packet,
                "private": {
                    "existing_pair": "EUR_USD",
                    "existing_side": "LONG",
                    "entry": float(100 + index),
                    "entry_epoch": 1_800_000_000 + index * 86_400 - 3_600,
                    "tp": float(103 + index),
                    "pip": 1.0,
                    "decision_bar": index,
                    "decision_epoch": 1_800_000_000 + index * 86_400,
                    "candidate_pair": "USD_JPY",
                    "candidate_side": "SHORT",
                    "candidate_entry": float(200 + index),
                    "candidate_tp": float(197 + index),
                    "candidate_pip": 1.0,
                    "candidate_bar": index,
                },
            }
        )
    return cases


def _args(tmp_path: Path) -> argparse.Namespace:
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text('{"source":"fixture"}\n', encoding="utf-8")
    return argparse.Namespace(
        root=tmp_path / "history",
        source_manifest=source_manifest,
        out_dir=tmp_path / "experiment",
        model_id="gpt-test-fixed",
        model_lineage="gpt-test-lineage",
    )


def _raw_cell(case: dict, index: int, *, context_id: str | None = None) -> dict:
    cell_id = case["packet"]["id"]
    return {
        "contract": "QR_DOJO_CAPITAL_RECYCLE_RAW_CELL_RESPONSE_V2",
        "experiment_id": MODULE.EXPERIMENT_ID,
        "id": cell_id,
        "execution": {
            "context_id": context_id or f"fresh-context-{cell_id}",
            "agent_receipt_id": f"agent-receipt-{cell_id}",
            "model_id": "gpt-test-fixed",
            "model_lineage": "gpt-test-lineage",
            "provider_attestation": "SELF_ATTESTED_NO_PROVIDER_SIGNATURE",
            "fork_turns_none": True,
            "tools_used": False,
        },
        "decision": {
            "id": cell_id,
            "existing_direction": "KEEP_LONG" if index in {0, 3} else "FLAT",
            "next_direction": "SHORT" if index in {1, 4} else "FLAT",
            "conviction": 2,
            "reason": "cutoff-safe fixture judgment",
        },
    }


def _prepare(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[argparse.Namespace, list[dict]]:
    cases = _cases()
    monkeypatch.setattr(MODULE, "build_cases", lambda root: (cases, {"attempts": 17}))
    monkeypatch.setattr(
        MODULE,
        "validate_source_corpus",
        lambda root, manifest: {
            "canonical_sha256": "c" * 64,
            "shard_count": 56,
        },
    )
    args = _args(tmp_path)
    assert MODULE.prepare(args) == 0
    return args, cases


def _seal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[argparse.Namespace, list[dict]]:
    args, cases = _prepare(monkeypatch, tmp_path)
    for index, case in enumerate(cases):
        cell_id = case["packet"]["id"]
        response_path = tmp_path / f"raw-{cell_id}.json"
        response_path.write_text(
            json.dumps(_raw_cell(case, index), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        seal_args = argparse.Namespace(
            **vars(args), cell_id=cell_id, response=response_path
        )
        assert MODULE.seal_cell(seal_args) == 0
    assert MODULE.finalize_responses(args) == 0
    return args, cases


def test_window_is_exactly_distinct_2026_h1_worn_train() -> None:
    assert MODULE.FROM == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert MODULE.TO == datetime(2026, 7, 1, tzinfo=timezone.utc)
    assert MODULE.CELL_COUNT == 6
    assert "percentages" in MODULE.PROMPT_TEMPLATE
    assert "KEEP means 100% ASSET_A/full HOLD" in MODULE.PROMPT_TEMPLATE


def test_prepare_writes_no_truth_and_never_calls_outcome(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        MODULE,
        "outcome",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("prepare must not calculate truth")
        ),
    )
    args, _ = _prepare(monkeypatch, tmp_path)

    assert not (args.out_dir / "answer_key.json").exists()
    assert not (args.out_dir / "response_manifest.json").exists()
    packets = json.loads((args.out_dir / "packets.json").read_text(encoding="utf-8"))
    prereg = json.loads(
        (args.out_dir / "preregistration.json").read_text(encoding="utf-8")
    )
    assert packets["answer_key_present"] is False
    assert len(packets["packets"]) == 6
    assert prereg["answer_key_present_at_prepare"] is False
    assert prereg["model_allocation_discretion"] is False
    assert prereg["fixed_policy"]["partial_allocations_allowed"] is False


def test_response_schema_rejects_model_allocation_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args, cases = _prepare(monkeypatch, tmp_path)
    payload = _raw_cell(cases[0], 0)
    payload["decision"]["existing_pct"] = 50
    decision_path = tmp_path / "raw-P01.json"
    decision_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    assert (
        MODULE.seal_cell(
            argparse.Namespace(**vars(args), cell_id="P01", response=decision_path)
        )
        == 0
    )
    terminal = json.loads(
        (args.out_dir / "responses/P01.json").read_text(encoding="utf-8")
    )
    assert terminal["status"] == "SYNTHETIC_FAILURE"
    assert terminal["effective_action"] == "CUT_TO_RESERVE"
    assert terminal["decision"] is None
    assert (
        args.out_dir / "raw-responses/P01.json"
    ).read_bytes() == decision_path.read_bytes()


def test_score_requires_sealed_responses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args, _ = _prepare(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="response_manifest.json must finalize"):
        MODULE.score(args)
    assert not (args.out_dir / "answer_key.json").exists()


def test_score_reconstructs_packets_then_creates_truth_and_fixed_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args, cases = _seal(monkeypatch, tmp_path)
    assert not (args.out_dir / "answer_key.json").exists()
    monkeypatch.setattr(MODULE, "build_cases", lambda root: (cases, {"attempts": 17}))
    monkeypatch.setattr(
        MODULE,
        "load_data",
        lambda root: {"EUR_USD": object(), "USD_JPY": object()},
    )
    cut_hold = [(-1.0, 2.0), (1.0, -3.0), (2.0, 1.0)] * 2
    next_values = [5.0, 4.0, -4.0] * 2

    def fake_outcome(rows, *, entry, **kwargs):
        if entry < 200:
            return cut_hold[int(entry - 100)]
        return 0.0, next_values[int(entry - 200)]

    monkeypatch.setattr(MODULE, "outcome", fake_outcome)

    assert MODULE.score(args) == 0

    answer = json.loads((args.out_dir / "answer_key.json").read_text(encoding="utf-8"))
    evidence = json.loads((args.out_dir / "evidence.json").read_text(encoding="utf-8"))
    assert len(answer["answers"]) == MODULE.CELL_COUNT
    assert answer["generated_after_response_manifest_raw_sha256"] == MODULE.raw_sha(
        args.out_dir / "response_manifest.json"
    )
    assert [cell["fixed_policy_action"] for cell in evidence["cells"]] == [
        "FULL_HOLD",
        "ROTATE_FULL_TO_NEXT",
        "CUT_TO_RESERVE",
        "FULL_HOLD",
        "ROTATE_FULL_TO_NEXT",
        "CUT_TO_RESERVE",
    ]
    totals = evidence["score"]["totals_capacity_pips"]
    assert totals["hierarchical"] == 18.0
    assert totals["full_hold"] == 0.0
    assert totals["cut_to_reserve"] == 4.0
    assert totals["rotate_full_to_next"] == 14.0
    assert totals["full_allocation_oracle"] == 22.0
    assert evidence["proof_eligible"] is False
    assert evidence["live_permission"] is False


def test_packet_mutation_is_rejected_before_response_seal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args, cases = _prepare(monkeypatch, tmp_path)
    packet_path = args.out_dir / "packets.json"
    packets = json.loads(packet_path.read_text(encoding="utf-8"))
    packets["packets"][0]["asset_a_existing"]["side"] = "SHORT"
    packet_path.write_text(json.dumps(packets) + "\n", encoding="utf-8")
    decision_path = tmp_path / "raw-P01.json"
    decision_path.write_text(
        json.dumps(_raw_cell(cases[0], 0)) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="packets canonical SHA-256 mismatch"):
        MODULE.seal_cell(
            argparse.Namespace(**vars(args), cell_id="P01", response=decision_path)
        )


def test_cell_first_attempt_cannot_be_overwritten(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args, cases = _prepare(monkeypatch, tmp_path)
    response_path = tmp_path / "raw-P01.json"
    response_path.write_text(
        json.dumps(_raw_cell(cases[0], 0)) + "\n", encoding="utf-8"
    )
    seal_args = argparse.Namespace(**vars(args), cell_id="P01", response=response_path)
    assert MODULE.seal_cell(seal_args) == 0
    preserved = (args.out_dir / "raw-responses/P01.json").read_bytes()

    with pytest.raises(ValueError, match="refusing to overwrite first attempt"):
        MODULE.seal_cell(seal_args)

    assert (args.out_dir / "raw-responses/P01.json").read_bytes() == preserved


def test_missing_cell_never_shrinks_denominator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args, cases = _prepare(monkeypatch, tmp_path)
    for index, case in enumerate(cases[:5]):
        cell_id = case["packet"]["id"]
        response_path = tmp_path / f"raw-{cell_id}.json"
        response_path.write_text(
            json.dumps(_raw_cell(case, index)) + "\n", encoding="utf-8"
        )
        MODULE.seal_cell(
            argparse.Namespace(**vars(args), cell_id=cell_id, response=response_path)
        )

    with pytest.raises(ValueError, match="exact six cell terminals required"):
        MODULE.finalize_responses(args)
    assert not (args.out_dir / "response_manifest.json").exists()
    assert not (args.out_dir / "answer_key.json").exists()

    MODULE.seal_missing_cell(argparse.Namespace(**vars(args), cell_id="P06"))
    assert MODULE.finalize_responses(args) == 0
    manifest = json.loads(
        (args.out_dir / "response_manifest.json").read_text(encoding="utf-8")
    )
    missing = next(row for row in manifest["cells"] if row["id"] == "P06")
    assert manifest["fixed_denominator"] == MODULE.CELL_COUNT
    assert len(manifest["cells"]) == MODULE.CELL_COUNT
    assert missing["effective_status"] == "SYNTHETIC_MISSING"
    assert missing["effective_action"] == "CUT_TO_RESERVE"


def test_duplicate_contexts_become_synthetic_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args, cases = _prepare(monkeypatch, tmp_path)
    for index, case in enumerate(cases):
        cell_id = case["packet"]["id"]
        context_id = "duplicate-context" if index in {0, 1} else None
        response_path = tmp_path / f"raw-{cell_id}.json"
        response_path.write_text(
            json.dumps(_raw_cell(case, index, context_id=context_id)) + "\n",
            encoding="utf-8",
        )
        MODULE.seal_cell(
            argparse.Namespace(**vars(args), cell_id=cell_id, response=response_path)
        )

    assert MODULE.finalize_responses(args) == 0
    manifest = json.loads(
        (args.out_dir / "response_manifest.json").read_text(encoding="utf-8")
    )
    by_id = {row["id"]: row for row in manifest["cells"]}
    assert by_id["P01"]["effective_status"] == "SYNTHETIC_FAILURE_DUPLICATE_CONTEXT"
    assert by_id["P02"]["effective_status"] == "SYNTHETIC_FAILURE_DUPLICATE_CONTEXT"
    assert by_id["P01"]["effective_action"] == "CUT_TO_RESERVE"
    assert by_id["P03"]["effective_status"] == "VALID"


def test_raw_response_tamper_after_finalize_blocks_scoring(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args, _ = _seal(monkeypatch, tmp_path)
    raw_path = args.out_dir / "raw-responses/P01.json"
    raw_path.write_bytes(raw_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="raw response SHA-256 mismatch"):
        MODULE.score(args)
    assert not (args.out_dir / "answer_key.json").exists()


def test_source_corpus_inventory_hashes_exact_consumed_shards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(MODULE, "PAIRS", ("EUR_USD",))
    root = tmp_path / "history"
    shard = (
        root / "run-1/EUR_USD/EUR_USD_M5_BA_20260101T000000Z_20260701T000000Z.jsonl.gz"
    )
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"sealed-shard-bytes")
    relative = str(shard.relative_to(root))
    manifest = {
        "source_root": str(root.resolve()),
        "sources": [
            {
                "pair": "EUR_USD",
                "relative_path": relative,
                "file_size_bytes": shard.stat().st_size,
                "file_sha256": MODULE.raw_sha(shard),
                "from_utc": "2026-01-01T00:00:00+00:00",
                "to_utc": "2026-07-01T00:00:00+00:00",
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    receipt = MODULE.validate_source_corpus(root, manifest_path)
    assert receipt["shard_count"] == 1
    assert receipt["shards"][0]["file_sha256"] == MODULE.raw_sha(shard)

    shard.write_bytes(b"mutated-shard-bytes")
    with pytest.raises(ValueError, match="source shard size mismatch"):
        MODULE.validate_source_corpus(root, manifest_path)


def test_outcome_allows_tp_after_decision_open_within_decision_bar() -> None:
    rows = [
        (0, 100.0, 103.0, 99.0, 101.0, 100.1, 103.1, 99.1, 101.1),
        (
            MODULE.HOLD_HORIZON_S,
            101.0,
            101.0,
            101.0,
            101.0,
            101.1,
            101.1,
            101.1,
            101.1,
        ),
    ]

    cut, held = MODULE.outcome(
        rows,
        decision_bar=0,
        side="LONG",
        entry=100.0,
        tp=103.0,
        pip=1.0,
    )

    assert cut == 0.0
    assert held == MODULE.TP_PIPS


def test_entry_bar_tp_touch_excludes_position_before_later_decision() -> None:
    rows = [
        (0, 100.0, 103.0, 99.0, 101.0, 100.1, 103.1, 99.1, 101.1),
        (300, 101.0, 102.0, 100.0, 101.0, 101.1, 102.1, 100.1, 101.1),
    ]

    assert MODULE._tp_filled_before_decision(
        rows,
        0,
        2,
        side="LONG",
        tp=103.0,
    )


def test_continuity_gate_rejects_missing_m5_bar() -> None:
    contiguous = [(index * 300,) for index in range(5)]
    missing = [(0,), (300,), (900,), (1200,)]

    assert MODULE._is_contiguous(contiguous, 0, 1200)
    assert not MODULE._is_contiguous(missing, 0, 1200)
