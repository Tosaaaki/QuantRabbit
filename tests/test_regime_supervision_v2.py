from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quant_rabbit.regime_supervision_v2 import (
    RegimeSupervisionError,
    build_regime_supervision_v2,
    effective_family_action,
)

UTC = timezone.utc
NOW = datetime(2026, 7, 18, 6, 0, tzinfo=UTC)


def _supervision(**overrides: object) -> dict:
    values: dict = {
        "pair": "EUR_USD",
        "declared_regime": "RANGE",
        "pair_action": "GO",
        "family_rows": [
            {
                "family_id": "RANGE_RAIL",
                "action": "GO",
                "regime_affinity": ["RANGE", "SQUEEZE"],
            },
            {
                "family_id": "MOMENTUM_BREAK",
                "action": "GO",
                "regime_affinity": ["TREND"],
            },
        ],
        "observed_at_utc": NOW,
        "expires_at_utc": NOW + timedelta(hours=6),
        "observation_sha256": "a" * 64,
    }
    values.update(overrides)
    return build_regime_supervision_v2(**values)


def test_regime_mismatch_demotes_go_to_caution() -> None:
    supervision = _supervision()

    rows = {row["family_id"]: row for row in supervision["family_rows"]}
    assert rows["RANGE_RAIL"]["effective_action"] == "GO"
    assert rows["MOMENTUM_BREAK"]["effective_action"] == "CAUTION"
    assert rows["MOMENTUM_BREAK"]["demoted_for_regime_mismatch"] is True

    assert (
        effective_family_action(
            supervision, family_id="RANGE_RAIL", now_utc=NOW + timedelta(hours=1)
        )
        == "GO"
    )
    assert (
        effective_family_action(
            supervision, family_id="MOMENTUM_BREAK", now_utc=NOW + timedelta(hours=1)
        )
        == "CAUTION"
    )


def test_expiry_and_unknown_family_fail_closed_to_unsupervised() -> None:
    supervision = _supervision()

    assert (
        effective_family_action(
            supervision, family_id="RANGE_RAIL", now_utc=NOW + timedelta(hours=7)
        )
        == "UNSUPERVISED"
    )
    assert (
        effective_family_action(
            supervision, family_id="UNKNOWN", now_utc=NOW + timedelta(hours=1)
        )
        == "UNSUPERVISED"
    )


def test_pair_action_caps_family_rows_and_tamper_is_refused() -> None:
    stopped = _supervision(pair_action="STOP")
    assert (
        effective_family_action(
            stopped, family_id="RANGE_RAIL", now_utc=NOW + timedelta(hours=1)
        )
        == "STOP"
    )
    cautious = _supervision(pair_action="CAUTION")
    assert (
        effective_family_action(
            cautious, family_id="RANGE_RAIL", now_utc=NOW + timedelta(hours=1)
        )
        == "CAUTION"
    )

    tampered = dict(_supervision())
    tampered["pair_action"] = "GO"
    tampered["live_permission"] = True
    with pytest.raises(RegimeSupervisionError, match="digest"):
        effective_family_action(
            tampered, family_id="RANGE_RAIL", now_utc=NOW + timedelta(hours=1)
        )

    with pytest.raises(RegimeSupervisionError, match="TTL"):
        _supervision(expires_at_utc=NOW + timedelta(hours=7))
    with pytest.raises(RegimeSupervisionError, match="duplicate"):
        _supervision(
            family_rows=[
                {"family_id": "X", "action": "GO", "regime_affinity": ["RANGE"]},
                {"family_id": "X", "action": "GO", "regime_affinity": ["RANGE"]},
            ]
        )
