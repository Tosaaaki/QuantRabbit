from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quant_rabbit.dojo_holdout_burn import (
    EVENT_CONTRACT,
    HoldoutBurnError,
    append_burn_intent,
    append_legacy_burn,
    bind_result,
    initialize_registry,
    reserve_holdout,
    status_artifact,
    verify_registry,
)


UTC = timezone.utc
NOW = datetime(2026, 7, 19, 1, 0, tzinfo=UTC)
SHA = {
    name: character * 64
    for name, character in {
        "prompt": "1",
        "model": "2",
        "scorer": "3",
        "code": "4",
        "corpus": "5",
        "custody": "6",
        "legacy": "7",
        "reveal": "8",
        "result": "9",
    }.items()
}


def _events(tmp_path: Path, name: str = "events") -> Path:
    events = tmp_path / name
    initialize_registry(
        events,
        registry_id=f"registry-{name}",
        created_by="test-custodian",
        event_at_utc=NOW,
    )
    return events


def _reservation(
    reservation_id: str = "reservation-1",
    *,
    candidate_id: str = "candidate-1",
    instrument: str = "USD_JPY",
    outcome_domain: str = "ENTRY_240M_DIRECTION",
    window_from: datetime | None = None,
    window_to: datetime | None = None,
    granularity: str = "S5",
    corpus_sha: str | None = None,
) -> dict[str, object]:
    return {
        "reservation_id": reservation_id,
        "candidate_id": candidate_id,
        "task_kind": "AI_DISCRETION",
        "family_id": "entry-direction-v1",
        "selection_lineage_id": "selection-lineage-v1",
        "unused_relative_to": "prompt/model/scorer/code lineage v1",
        "instruments": [instrument],
        "target_outcome_domain": outcome_domain,
        "window_from_utc": (window_from or datetime(2026, 1, 1, tzinfo=UTC))
        .isoformat()
        .replace("+00:00", "Z"),
        "window_to_utc": (window_to or datetime(2026, 2, 1, tzinfo=UTC))
        .isoformat()
        .replace("+00:00", "Z"),
        "granularity": granularity,
        "input_modalities": ["CANDLE", "CALENDAR"],
        "prompt_set_sha256": SHA["prompt"],
        "model_policy_sha256": SHA["model"],
        "scorer_sha256": SHA["scorer"],
        "code_sha256": SHA["code"],
        "corpus_manifest_sha256": corpus_sha or SHA["corpus"],
        "custody_policy_sha256": SHA["custody"],
    }


def _legacy(
    burn_id: str = "legacy-1",
    *,
    instrument: str = "USD_JPY",
    outcome_domain: str = "ENTRY_240M_DIRECTION",
    window_from: datetime | None = None,
    window_to: datetime | None = None,
    granularity: str = "M5",
) -> dict[str, object]:
    return {
        "burn_id": burn_id,
        "task_kind": "AI_DISCRETION",
        "selection_lineage_id": "legacy-global",
        "instruments": [instrument],
        "target_outcome_domain": outcome_domain,
        "window_from_utc": (window_from or datetime(2025, 1, 1, tzinfo=UTC))
        .isoformat()
        .replace("+00:00", "Z"),
        "window_to_utc": (window_to or datetime(2026, 1, 1, tzinfo=UTC))
        .isoformat()
        .replace("+00:00", "Z"),
        "granularity": granularity,
        "input_modalities": ["CANDLE"],
        "legacy_source": "Fable W52/W53 durable audit",
        "legacy_evidence_sha256": SHA["legacy"],
        "reason": "LEGACY_OUTCOME_EXPOSURE",
    }


def _burn_intent(reservation_id: str = "reservation-1") -> dict[str, object]:
    return {
        "reservation_id": reservation_id,
        "reveal_material_sha256": SHA["reveal"],
        "revealed_by": "test-custodian",
        "permanence_acknowledged": True,
    }


def _result(reservation_id: str = "reservation-1") -> dict[str, object]:
    return {
        "reservation_id": reservation_id,
        "result_sha256": SHA["result"],
        "result_contract": "QR_DOJO_TEST_SCORE_V1",
        "bound_by": "test-scorer",
    }


def test_reservation_burn_and_result_are_permanent_and_never_authoritative(
    tmp_path: Path,
) -> None:
    events = _events(tmp_path)
    reserve_holdout(events, reservation=_reservation(), event_at_utc=NOW)

    reserved = status_artifact(events)
    assert reserved["reservations"][0]["state"] == (
        "LOCALLY_UNOPENED_RELATIVE_TO_DECLARED_SELECTION_LINEAGE"
    )
    assert reserved["reservations"][0]["permanently_burned"] is False

    append_burn_intent(events, intent=_burn_intent(), event_at_utc=NOW)
    burned = status_artifact(events)
    assert burned["reservations"][0]["state"] == "BURNED_PENDING_RESULT"
    assert burned["reservations"][0]["permanently_burned"] is True

    with pytest.raises(HoldoutBurnError, match="already permanently burned"):
        append_burn_intent(events, intent=_burn_intent(), event_at_utc=NOW)

    bind_result(events, result=_result(), event_at_utc=NOW)
    complete = status_artifact(events)
    assert complete["reservations"][0]["state"] == "RESULT_BOUND"
    assert complete["reservations"][0]["permanently_burned"] is True
    assert complete["reservations"][0]["result_bound"] is True
    for field in (
        "proof_eligible",
        "promotion_eligible",
        "live_permission",
        "broker_mutation_allowed",
    ):
        assert complete[field] is False
    for path in sorted(events.glob("*.json")):
        event = json.loads(path.read_text(encoding="utf-8"))
        assert event["contract"] == EVENT_CONTRACT
        assert event["external_witness_status"] == "ABSENT"
        assert event["proof_eligible"] is False
        assert event["promotion_eligible"] is False
        assert event["live_permission"] is False
        assert event["broker_mutation_allowed"] is False


def test_result_before_burn_and_duplicate_result_are_rejected(tmp_path: Path) -> None:
    events = _events(tmp_path)
    reserve_holdout(events, reservation=_reservation(), event_at_utc=NOW)
    with pytest.raises(HoldoutBurnError, match="before permanent burn"):
        bind_result(events, result=_result(), event_at_utc=NOW)
    append_burn_intent(events, intent=_burn_intent(), event_at_utc=NOW)
    bind_result(events, result=_result(), event_at_utc=NOW)
    with pytest.raises(HoldoutBurnError, match="already bound"):
        bind_result(events, result=_result(), event_at_utc=NOW)


@pytest.mark.parametrize(
    "missing_binding",
    [
        "prompt_set_sha256",
        "model_policy_sha256",
        "scorer_sha256",
        "code_sha256",
        "corpus_manifest_sha256",
        "custody_policy_sha256",
    ],
)
def test_reservation_requires_all_prompt_model_scorer_code_corpus_and_custody_bindings(
    tmp_path: Path, missing_binding: str
) -> None:
    events = _events(tmp_path, missing_binding)
    reservation = _reservation()
    reservation.pop(missing_binding)
    with pytest.raises(HoldoutBurnError, match="bindings are incomplete"):
        reserve_holdout(events, reservation=reservation, event_at_utc=NOW)


def test_overlap_identity_ignores_candidate_corpus_prompt_and_granularity_relabels(
    tmp_path: Path,
) -> None:
    events = _events(tmp_path)
    reserve_holdout(events, reservation=_reservation(), event_at_utc=NOW)
    relabeled = _reservation(
        "reservation-renamed",
        candidate_id="entirely-new-candidate-name",
        granularity="M5",
        corpus_sha="a" * 64,
    )
    relabeled["prompt_set_sha256"] = "b" * 64
    with pytest.raises(HoldoutBurnError, match="regardless of candidate"):
        reserve_holdout(events, reservation=relabeled, event_at_utc=NOW)


def test_overlap_is_scoped_by_instrument_and_time_not_outcome_domain(
    tmp_path: Path,
) -> None:
    events = _events(tmp_path)
    reserve_holdout(events, reservation=_reservation(), event_at_utc=NOW)
    reserve_holdout(
        events,
        reservation=_reservation(
            "adjacent",
            window_from=datetime(2026, 2, 1, tzinfo=UTC),
            window_to=datetime(2026, 3, 1, tzinfo=UTC),
        ),
        event_at_utc=NOW,
    )
    reserve_holdout(
        events,
        reservation=_reservation("other-pair", instrument="EUR_USD"),
        event_at_utc=NOW,
    )
    with pytest.raises(HoldoutBurnError, match="regardless of candidate"):
        reserve_holdout(
            events,
            reservation=_reservation("other-domain", outcome_domain="EXIT_CUT_OR_HOLD"),
            event_at_utc=NOW,
        )
    assert status_artifact(events)["reservation_count"] == 3


@pytest.mark.parametrize(
    ("first_domain", "second_domain"),
    [
        ("ENTRY_240M_DIRECTION", "EXIT_CUT_OR_HOLD"),
        ("EXIT_CUT_OR_HOLD", "ENTRY_240M_DIRECTION"),
    ],
)
def test_entry_and_exit_cannot_double_reserve_the_same_market_path(
    tmp_path: Path, first_domain: str, second_domain: str
) -> None:
    events = _events(tmp_path, f"entry-exit-{first_domain}")
    reserve_holdout(
        events,
        reservation=_reservation("first", outcome_domain=first_domain),
        event_at_utc=NOW,
    )
    with pytest.raises(HoldoutBurnError, match="regardless of candidate"):
        reserve_holdout(
            events,
            reservation=_reservation("second", outcome_domain=second_domain),
            event_at_utc=NOW,
        )


@pytest.mark.parametrize("derived_domain", ["ENTRY_240M_DIRECTION", "EXIT_CUT_OR_HOLD"])
def test_legacy_market_price_path_burns_every_derived_outcome_domain(
    tmp_path: Path, derived_domain: str
) -> None:
    events = _events(tmp_path, derived_domain)
    append_legacy_burn(
        events,
        burn=_legacy(
            outcome_domain="MARKET_PRICE_PATH",
            window_from=datetime(2025, 12, 1, tzinfo=UTC),
            window_to=datetime(2026, 1, 15, tzinfo=UTC),
        ),
        event_at_utc=NOW,
    )
    with pytest.raises(HoldoutBurnError, match="overlaps a legacy burn"):
        reserve_holdout(
            events,
            reservation=_reservation(outcome_domain=derived_domain),
            event_at_utc=NOW,
        )


@pytest.mark.parametrize("market_path_first", [True, False])
def test_market_price_path_reservation_conflict_is_symmetric(
    tmp_path: Path, market_path_first: bool
) -> None:
    events = _events(tmp_path, f"path-symmetry-{market_path_first}")
    first_domain = "MARKET_PRICE_PATH" if market_path_first else "EXIT_CUT_OR_HOLD"
    second_domain = "EXIT_CUT_OR_HOLD" if market_path_first else "MARKET_PRICE_PATH"
    reserve_holdout(
        events,
        reservation=_reservation("first", outcome_domain=first_domain),
        event_at_utc=NOW,
    )
    with pytest.raises(HoldoutBurnError, match="regardless of candidate"):
        reserve_holdout(
            events,
            reservation=_reservation("second", outcome_domain=second_domain),
            event_at_utc=NOW,
        )


def test_legacy_burn_blocks_future_overlap_but_allows_adjacent_window(
    tmp_path: Path,
) -> None:
    events = _events(tmp_path)
    append_legacy_burn(events, burn=_legacy(), event_at_utc=NOW)
    overlapping = _reservation(
        window_from=datetime(2025, 6, 1, tzinfo=UTC),
        window_to=datetime(2025, 7, 1, tzinfo=UTC),
        granularity="S5",
    )
    with pytest.raises(HoldoutBurnError, match="overlaps a legacy burn"):
        reserve_holdout(events, reservation=overlapping, event_at_utc=NOW)
    adjacent = _reservation(
        window_from=datetime(2026, 1, 1, tzinfo=UTC),
        window_to=datetime(2026, 2, 1, tzinfo=UTC),
    )
    reserve_holdout(events, reservation=adjacent, event_at_utc=NOW)

    append_legacy_burn(
        events,
        burn=_legacy(
            "legacy-nested-evidence",
            window_from=datetime(2025, 12, 1, tzinfo=UTC),
            window_to=datetime(2026, 1, 15, tzinfo=UTC),
            granularity="S5",
        ),
        event_at_utc=NOW,
    )
    with pytest.raises(HoldoutBurnError, match="already recorded"):
        append_legacy_burn(
            events,
            burn=_legacy(
                "legacy-nested-evidence",
                window_from=datetime(2025, 12, 1, tzinfo=UTC),
                window_to=datetime(2026, 1, 15, tzinfo=UTC),
                granularity="S5",
            ),
            event_at_utc=NOW,
        )


def test_later_discovered_legacy_overlap_invalidates_derived_reservation_status(
    tmp_path: Path,
) -> None:
    events = _events(tmp_path)
    reserve_holdout(events, reservation=_reservation(), event_at_utc=NOW)
    append_legacy_burn(
        events,
        burn=_legacy(
            window_from=datetime(2026, 1, 15, tzinfo=UTC),
            window_to=datetime(2026, 1, 20, tzinfo=UTC),
        ),
        event_at_utc=NOW + timedelta(seconds=1),
    )
    row = status_artifact(events)["reservations"][0]
    assert row["state"] == "INVALIDATED_BY_LEGACY_BURN"
    assert row["invalidating_legacy_burn_ids"] == ["legacy-1"]
    with pytest.raises(HoldoutBurnError, match="invalidated reservation"):
        append_burn_intent(
            events,
            intent=_burn_intent(),
            event_at_utc=NOW + timedelta(seconds=2),
        )


def test_later_legacy_discovery_does_not_erase_prior_burn_or_result_binding(
    tmp_path: Path,
) -> None:
    events = _events(tmp_path)
    reserve_holdout(events, reservation=_reservation(), event_at_utc=NOW)
    append_burn_intent(events, intent=_burn_intent(), event_at_utc=NOW)
    append_legacy_burn(
        events,
        burn=_legacy(
            window_from=datetime(2026, 1, 15, tzinfo=UTC),
            window_to=datetime(2026, 1, 20, tzinfo=UTC),
        ),
        event_at_utc=NOW + timedelta(seconds=1),
    )
    bind_result(
        events,
        result=_result(),
        event_at_utc=NOW + timedelta(seconds=2),
    )
    row = status_artifact(events)["reservations"][0]
    assert row["state"] == "INVALIDATED_BY_LEGACY_BURN"
    assert row["permanently_burned"] is True
    assert row["result_bound"] is True


def test_hash_tamper_gap_and_unexpected_file_fail_closed(tmp_path: Path) -> None:
    tampered = _events(tmp_path, "tampered")
    reserve_holdout(tampered, reservation=_reservation(), event_at_utc=NOW)
    event_path = tampered / "000001.json"
    value = json.loads(event_path.read_text(encoding="utf-8"))
    value["body"]["candidate_id"] = "attacker-rewrite"
    event_path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(HoldoutBurnError, match="event_sha256 mismatch"):
        verify_registry(tampered)

    gap = _events(tmp_path, "gap")
    reserve_holdout(gap, reservation=_reservation(), event_at_utc=NOW)
    (gap / "000001.json").rename(gap / "000002.json")
    with pytest.raises(HoldoutBurnError, match="sequence has a gap"):
        verify_registry(gap)

    unexpected = _events(tmp_path, "unexpected")
    (unexpected / "README").write_text("not an event", encoding="utf-8")
    with pytest.raises(HoldoutBurnError, match="unexpected file"):
        verify_registry(unexpected)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b'{"x":1,"x":2}\n', "duplicate JSON key"),
        (b'{"x":NaN}\n', "non-finite JSON constant"),
        (b'{\n  "x": 1\n}\n', "not canonical JSON"),
    ],
)
def test_duplicate_key_nan_and_noncanonical_json_fail_closed(
    tmp_path: Path, raw: bytes, message: str
) -> None:
    events = _events(tmp_path, message.replace(" ", "-"))
    (events / "000001.json").write_bytes(raw)
    with pytest.raises(HoldoutBurnError, match=message):
        verify_registry(events)


def test_event_file_and_event_directory_symlinks_fail_closed(tmp_path: Path) -> None:
    file_link = _events(tmp_path, "file-link")
    os.symlink(file_link / "000000.json", file_link / "000001.json")
    with pytest.raises(HoldoutBurnError, match="regular file"):
        verify_registry(file_link)

    real = tmp_path / "real-events"
    real.mkdir()
    alias = tmp_path / "events-alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(HoldoutBurnError, match="real directory"):
        initialize_registry(
            alias,
            registry_id="symlink-registry",
            created_by="test",
            event_at_utc=NOW,
        )


def test_time_cannot_move_backward_and_windows_must_be_positive(tmp_path: Path) -> None:
    events = _events(tmp_path)
    with pytest.raises(HoldoutBurnError, match="positive and half-open"):
        reserve_holdout(
            events,
            reservation=_reservation(
                window_from=datetime(2026, 2, 1, tzinfo=UTC),
                window_to=datetime(2026, 2, 1, tzinfo=UTC),
            ),
            event_at_utc=NOW,
        )
    with pytest.raises(HoldoutBurnError, match="cannot move backward"):
        reserve_holdout(
            events,
            reservation=_reservation(),
            event_at_utc=NOW - timedelta(seconds=1),
        )


def test_cli_initializes_and_reports_conservative_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts/run-dojo-holdout-registry.py"
    )
    spec = importlib.util.spec_from_file_location("dojo_holdout_cli_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    events = tmp_path / "cli-events"

    assert (
        module.main(
            [
                "init",
                "--events-dir",
                str(events),
                "--registry-id",
                "cli-registry",
                "--created-by",
                "cli-test",
                "--event-at-utc",
                "2026-07-19T01:00:00Z",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["event_count"] == 1
    assert result["proof_eligible"] is False
    assert result["live_permission"] is False
