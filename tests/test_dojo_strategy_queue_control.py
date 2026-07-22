from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from quant_rabbit.dojo_strategy_queue_control import (
    DojoStrategyQueueControlError,
    commit_queue_transition,
    initialize_queue_store,
    verify_queue_store,
)
from quant_rabbit.dojo_strategy_research_queue import (
    EVIDENCE_CLASS,
    SELECTION_BASIS,
    TRIGGER_CONTRACT,
    build_research_queue,
)


def _artifact(tmp_path: Path, seed: str) -> tuple[Path, str]:
    path = tmp_path / f"result-{seed}.json"
    raw = json.dumps(
        {"seed": seed, "terminal": True},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def _trigger(
    artifact_sha256: str,
    *,
    semantic_seed: str,
    candidate_id: str | None = None,
) -> dict[str, object]:
    return {
        "contract": TRIGGER_CONTRACT,
        "schema_version": 1,
        "trigger_kind": "TERMINAL_RESULT",
        "result_artifact_sha256": artifact_sha256,
        "semantic_result_sha256": hashlib.sha256(
            semantic_seed.encode("utf-8")
        ).hexdigest(),
        "result_candidate_id": candidate_id,
        "source_partition": "TRAIN",
        "evidence_class": EVIDENCE_CLASS,
        "terminal_result": True,
        "material_change": True,
        "holdout_opened": False,
        "prospective_window_opened": False,
        "global_untouched_holdout_claimed": False,
        "target_multiple_backsolve_used": False,
        "selection_basis": SELECTION_BASIS,
        "economics": {
            "recorded_bid_ask_costed": True,
            "slippage_costed": True,
            "financing_costed": True,
            "continuous_mtm_complete": True,
            "margin_replayed": True,
            "lopo_complete": True,
            "fixed_denominator_complete": True,
        },
    }


def _commit(
    events: Path,
    *,
    trigger: dict[str, object],
    artifact: Path,
    snapshot: dict[str, object],
) -> dict[str, object]:
    return commit_queue_transition(
        events,
        queue=build_research_queue(),
        trigger=trigger,
        result_artifact_path=artifact,
        expected_tip_event_sha256=str(snapshot["latest_event_sha256"]),
        expected_parent_state_sha256=str(
            snapshot["latest_state"]["state_sha256"]
        ),
    )


def test_store_reserves_completes_and_exhausts_in_canonical_order(
    tmp_path: Path,
) -> None:
    events = tmp_path / "events"
    snapshot = initialize_queue_store(events, build_research_queue())
    assert snapshot["event_count"] == 1
    assert snapshot["cas_ready"] is True

    candidates = [
        None,
        "asia_sweep_reclaim_be",
        "h1_donchian_break_atr_trailing",
        "g8_relative_strength_risk_budget",
    ]
    expected_active = [
        "asia_sweep_reclaim_be",
        "h1_donchian_break_atr_trailing",
        "g8_relative_strength_risk_budget",
        None,
    ]
    for index, (closed, expected) in enumerate(
        zip(candidates, expected_active, strict=True), start=1
    ):
        artifact, digest = _artifact(tmp_path, str(index))
        snapshot = _commit(
            events,
            trigger=_trigger(
                digest,
                semantic_seed=f"semantic-{index}",
                candidate_id=closed,
            ),
            artifact=artifact,
            snapshot=snapshot,
        )
        active = snapshot["latest_state"]["active_reservation"]
        assert (None if active is None else active["candidate_id"]) == expected

    assert snapshot["event_count"] == 5
    assert snapshot["latest_event"]["decision"]["action"] == "QUEUE_EXHAUSTED"
    assert snapshot["latest_state"]["authority"]["live_permission"] is False
    assert verify_queue_store(events, build_research_queue()) == {
        key: value for key, value in snapshot.items() if key != "idempotent_replay"
    }


def test_identical_retry_is_idempotent_but_conflicting_child_is_rejected(
    tmp_path: Path,
) -> None:
    events = tmp_path / "events"
    genesis = initialize_queue_store(events, build_research_queue())
    artifact, digest = _artifact(tmp_path, "retry")
    trigger = _trigger(digest, semantic_seed="retry")

    committed = _commit(
        events, trigger=trigger, artifact=artifact, snapshot=genesis
    )
    replay = _commit(events, trigger=trigger, artifact=artifact, snapshot=genesis)

    assert replay["idempotent_replay"] is True
    assert replay["event_count"] == committed["event_count"] == 2

    other_artifact, other_digest = _artifact(tmp_path, "other")
    with pytest.raises(
        DojoStrategyQueueControlError,
        match="different committed child",
    ):
        _commit(
            events,
            trigger=_trigger(other_digest, semantic_seed="other"),
            artifact=other_artifact,
            snapshot=genesis,
        )


def test_stale_cas_and_result_byte_substitution_fail_closed(tmp_path: Path) -> None:
    events = tmp_path / "events"
    genesis = initialize_queue_store(events, build_research_queue())
    artifact, digest = _artifact(tmp_path, "one")
    committed = _commit(
        events,
        trigger=_trigger(digest, semantic_seed="one"),
        artifact=artifact,
        snapshot=genesis,
    )

    next_artifact, next_digest = _artifact(tmp_path, "two")
    with pytest.raises(
        DojoStrategyQueueControlError, match="different committed child"
    ):
        _commit(
            events,
            trigger=_trigger(
                next_digest,
                semantic_seed="two",
                candidate_id="asia_sweep_reclaim_be",
            ),
            artifact=next_artifact,
            snapshot=genesis,
        )

    next_artifact.write_bytes(b"substituted")
    with pytest.raises(DojoStrategyQueueControlError, match="SHA-256 mismatch"):
        _commit(
            events,
            trigger=_trigger(
                next_digest,
                semantic_seed="two",
                candidate_id="asia_sweep_reclaim_be",
            ),
            artifact=next_artifact,
            snapshot=committed,
        )


def test_event_tamper_gap_and_unexpected_entry_are_rejected(tmp_path: Path) -> None:
    events = tmp_path / "events"
    snapshot = initialize_queue_store(events, build_research_queue())
    artifact, digest = _artifact(tmp_path, "tamper")
    _commit(
        events,
        trigger=_trigger(digest, semantic_seed="tamper"),
        artifact=artifact,
        snapshot=snapshot,
    )

    event_path = events / "000001.json"
    event = json.loads(event_path.read_text(encoding="utf-8"))
    event["decision"]["action"] = "QUEUE_EXHAUSTED"
    event_path.write_text(
        json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DojoStrategyQueueControlError, match="event SHA-256"):
        verify_queue_store(events, build_research_queue())

    clean_events = tmp_path / "clean-events"
    initialize_queue_store(clean_events, build_research_queue())
    (clean_events / "notes.txt").write_text("not allowed", encoding="utf-8")
    with pytest.raises(DojoStrategyQueueControlError, match="unexpected entries"):
        verify_queue_store(clean_events, build_research_queue())


def test_queue_or_trigger_mutation_cannot_be_resealed_into_store(tmp_path: Path) -> None:
    events = tmp_path / "events"
    queue = build_research_queue()
    initialize_queue_store(events, queue)
    artifact, digest = _artifact(tmp_path, "mutated")
    trigger = _trigger(digest, semantic_seed="mutated")
    trigger["holdout_opened"] = True

    with pytest.raises(Exception, match="holdout/prospective"):
        commit_queue_transition(
            events,
            queue=copy.deepcopy(queue),
            trigger=trigger,
            result_artifact_path=artifact,
            expected_tip_event_sha256=verify_queue_store(events, queue)[
                "latest_event_sha256"
            ],
            expected_parent_state_sha256=verify_queue_store(events, queue)[
                "latest_state"
            ]["state_sha256"],
        )
