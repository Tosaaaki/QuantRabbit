from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quant_rabbit.prospective_registry import (
    ProspectiveRegistryError,
    empty_registry,
    evaluation_admissible,
    family_denominators,
    register_candidate,
)

UTC = timezone.utc
NOW = datetime(2026, 7, 18, tzinfo=UTC)


def _registered() -> dict:
    registry = register_candidate(
        empty_registry(),
        candidate_id="XS-alpha",
        family_id="S5_CROSS_SECTIONAL",
        lock_sha256="a" * 64,
        window_from_utc=NOW + timedelta(days=2),
        window_to_utc=NOW + timedelta(days=16),
        registered_at_utc=NOW,
    )
    return register_candidate(
        registry,
        candidate_id="M5-beta",
        family_id="M5_CURRENCY_STRENGTH",
        lock_sha256="b" * 64,
        window_from_utc=NOW + timedelta(days=5),
        window_to_utc=NOW + timedelta(days=19),
        registered_at_utc=NOW,
    )


def test_parallel_registration_chains_and_counts_denominators() -> None:
    registry = _registered()

    assert len(registry["entries"]) == 2
    assert registry["entries"][1]["previous_entry_sha256"] == (
        registry["entries"][0]["entry_sha256"]
    )
    assert family_denominators(registry) == {
        "S5_CROSS_SECTIONAL": 1,
        "M5_CURRENCY_STRENGTH": 1,
    }


def test_evaluation_stays_closed_until_window_maturity() -> None:
    registry = _registered()

    early = evaluation_admissible(
        registry, candidate_id="XS-alpha", now_utc=NOW + timedelta(days=10)
    )
    assert early["admissible"] is False
    assert early["reason"] == "WINDOW_NOT_YET_MATURED"

    matured = evaluation_admissible(
        registry, candidate_id="XS-alpha", now_utc=NOW + timedelta(days=16)
    )
    assert matured["admissible"] is True
    assert matured["grants_positive_result"] is False

    unknown = evaluation_admissible(
        registry, candidate_id="ghost", now_utc=NOW + timedelta(days=30)
    )
    assert unknown["admissible"] is False
    assert unknown["reason"] == "CANDIDATE_NOT_REGISTERED"


def test_backdated_duplicate_and_tampered_registrations_are_refused() -> None:
    registry = _registered()

    with pytest.raises(ProspectiveRegistryError, match="strictly after"):
        register_candidate(
            registry,
            candidate_id="late",
            family_id="S5_CROSS_SECTIONAL",
            lock_sha256="c" * 64,
            window_from_utc=NOW - timedelta(days=1),
            window_to_utc=NOW + timedelta(days=5),
            registered_at_utc=NOW,
        )
    with pytest.raises(ProspectiveRegistryError, match="already registered"):
        register_candidate(
            registry,
            candidate_id="XS-alpha",
            family_id="S5_CROSS_SECTIONAL",
            lock_sha256="c" * 64,
            window_from_utc=NOW + timedelta(days=2),
            window_to_utc=NOW + timedelta(days=9),
            registered_at_utc=NOW,
        )

    tampered = dict(registry)
    entries = [dict(row) for row in registry["entries"]]
    entries[0]["window_to_utc"] = (NOW + timedelta(days=1)).isoformat()
    tampered["entries"] = entries
    with pytest.raises(ProspectiveRegistryError, match="digest"):
        evaluation_admissible(tampered, candidate_id="XS-alpha", now_utc=NOW)
