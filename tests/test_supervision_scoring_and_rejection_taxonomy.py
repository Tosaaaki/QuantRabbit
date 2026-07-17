from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quant_rabbit.rejection_taxonomy import (
    RejectionTaxonomyError,
    aggregate_rejections,
    build_rejection_record,
)
from quant_rabbit.supervision_outcome_scorer import (
    SupervisionScoringError,
    build_supervision_scorecard,
    row_hit,
)

NOW = datetime(2026, 7, 18, tzinfo=timezone.utc)


def test_row_hit_semantics() -> None:
    assert row_hit("GO", 12.0) is True
    assert row_hit("GO", -1.0) is False
    assert row_hit("STOP", -20.0) is True
    assert row_hit("STOP", 0.0) is True
    assert row_hit("STOP", 3.0) is False
    assert row_hit("CAUTION", 4.9) is True
    assert row_hit("CAUTION", -30.0) is False
    with pytest.raises(SupervisionScoringError, match="action"):
        row_hit("HOLD", 1.0)


def test_low_accuracy_family_is_flagged_for_auto_caution() -> None:
    unreliable = [
        {"family_id": "MOMO", "action": "GO", "realized_stressed_pips": -5.0}
    ] * 8 + [
        {"family_id": "MOMO", "action": "GO", "realized_stressed_pips": 5.0}
    ] * 4
    reliable = [
        {"family_id": "RANGE", "action": "GO", "realized_stressed_pips": 5.0}
    ] * 9 + [
        {"family_id": "RANGE", "action": "GO", "realized_stressed_pips": -5.0}
    ] * 3
    sparse = [{"family_id": "NEW", "action": "STOP", "realized_stressed_pips": -1.0}]

    scorecard = build_supervision_scorecard(
        [*unreliable, *reliable, *sparse], window_label="2026-07"
    )

    rows = {row["family_id"]: row for row in scorecard["family_rows"]}
    assert rows["MOMO"]["below_accuracy_floor"] is True
    assert rows["RANGE"]["below_accuracy_floor"] is False
    assert rows["NEW"]["measurable"] is False
    assert scorecard["supervision_auto_caution_required"] == ["MOMO"]
    assert scorecard["measurement_only"] is True


def test_rejection_records_require_fixed_codes_and_aggregate() -> None:
    records = [
        build_rejection_record(
            candidate_id=f"XS-{index}",
            family_id="S5_CROSS_SECTIONAL",
            death_code=code,
            evidence_sha256="a" * 64,
            rejected_at_utc=NOW,
        )
        for index, code in enumerate(
            ["CLOSE_CROSSING_LEAK", "CLOSE_CROSSING_LEAK", "ROBUSTNESS_FLOOR_FAILED"]
        )
    ]

    aggregate = aggregate_rejections(records, window_label="2026Q3")

    row = aggregate["family_rows"][0]
    assert row["dominant_death_code"] == "CLOSE_CROSSING_LEAK"
    assert row["total_rejections"] == 3

    with pytest.raises(RejectionTaxonomyError, match="fixed taxonomy"):
        build_rejection_record(
            candidate_id="XS-x",
            family_id="F",
            death_code="BAD_LUCK",
            evidence_sha256="a" * 64,
            rejected_at_utc=NOW,
        )
    tampered = dict(records[0])
    tampered["death_code"] = "DIRECTION_WRONG"
    with pytest.raises(RejectionTaxonomyError, match="digest"):
        aggregate_rejections([tampered], window_label="2026Q3")
