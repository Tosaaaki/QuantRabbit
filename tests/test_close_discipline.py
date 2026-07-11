from quant_rabbit.close_discipline import (
    thesis_evolution_reason_has_hard_close_evidence,
)


def test_thesis_evolution_hard_close_requires_invalidation_and_confirmation() -> None:
    assert thesis_evolution_reason_has_hard_close_evidence(
        "invalidation hit: current ask 1.16310 >= buffered invalidation "
        "1.16290 (raw 1.16270, buffer 2.0p); technical invalidation "
        "confirmed against SHORT: H1 BOS_UP",
        expected_side="SHORT",
    )
    assert not thesis_evolution_reason_has_hard_close_evidence(
        "invalidation hit: current ask 1.16310 >= buffered invalidation "
        "1.16290 (raw 1.16270, buffer 2.0p); without technical confirmation",
        expected_side="SHORT",
    )


def test_thesis_evolution_forecast_and_expiry_labels_are_soft() -> None:
    assert not thesis_evolution_reason_has_hard_close_evidence(
        "FORECAST FLIPPED: entry UP to current DOWN",
        expected_side="LONG",
    )
    assert not thesis_evolution_reason_has_hard_close_evidence(
        "THESIS_EXPIRED: age exceeded horizon across consecutive checks",
        expected_side="LONG",
    )


def test_thesis_evolution_structural_prose_alone_is_soft() -> None:
    assert not thesis_evolution_reason_has_hard_close_evidence(
        "loss-cut: close-confirmed structural break against LONG",
        expected_side="LONG",
    )


def test_thesis_evolution_hard_close_rejects_wrong_side_and_unbuffered_text() -> None:
    canonical = (
        "invalidation hit: current bid 1.16900 <= buffered invalidation "
        "1.16930 (raw 1.16950, buffer 2.0p); technical invalidation "
        "confirmed against LONG: H1 MACD-; M15 ST-"
    )
    assert not thesis_evolution_reason_has_hard_close_evidence(
        canonical,
        expected_side="SHORT",
    )
    assert not thesis_evolution_reason_has_hard_close_evidence(
        "invalidation hit: current bid 1.16900 <= raw invalidation 1.16930; "
        "technical invalidation confirmed against LONG: H1 MACD-",
        expected_side="LONG",
    )


def test_thesis_evolution_hard_close_rejects_wrong_price_geometry_or_zero_buffer() -> None:
    assert not thesis_evolution_reason_has_hard_close_evidence(
        "invalidation hit: current ask 1.16900 >= buffered invalidation "
        "1.16930 (raw 1.16950, buffer 2.0p); technical invalidation "
        "confirmed against LONG: H1 MACD-",
        expected_side="LONG",
    )
    assert not thesis_evolution_reason_has_hard_close_evidence(
        "invalidation hit: current bid 1.16310 <= buffered invalidation "
        "1.16290 (raw 1.16270, buffer 2.0p); technical invalidation "
        "confirmed against SHORT: H1 MACD+",
        expected_side="SHORT",
    )
    assert not thesis_evolution_reason_has_hard_close_evidence(
        "invalidation hit: current bid 1.16930 <= buffered invalidation "
        "1.16930 (raw 1.16930, buffer 0.0p); technical invalidation "
        "confirmed against LONG: H1 MACD-",
        expected_side="LONG",
    )


def test_thesis_evolution_hard_close_rejects_negated_or_missing_side_text() -> None:
    for reason in (
        "no invalidation hit: current bid 1.16900 <= buffered invalidation "
        "1.16930 (raw 1.16950, buffer 2.0p); technical invalidation "
        "confirmed against LONG: H1 MACD-",
        "invalidation not hit; technical invalidation confirmed against LONG: H1 MACD-",
        "invalidation hit: current bid 1.16900 <= buffered invalidation "
        "1.16930 (raw 1.16950, buffer 2.0p); technical invalidation not "
        "confirmed against LONG: H1 MACD-",
        "invalidation hit: current bid 1.16900 <= buffered invalidation "
        "1.16930 (raw 1.16950, buffer 2.0p); without technical confirmation",
    ):
        assert not thesis_evolution_reason_has_hard_close_evidence(
            reason,
            expected_side="LONG",
        )
    assert not thesis_evolution_reason_has_hard_close_evidence(
        "invalidation hit: current bid 1.16900 <= buffered invalidation "
        "1.16930 (raw 1.16950, buffer 2.0p); technical invalidation "
        "confirmed against LONG: H1 MACD-",
        expected_side="",
    )
