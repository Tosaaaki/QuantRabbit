from __future__ import annotations

import math

import pytest

from quant_rabbit.dojo_monthly_mtm_scorer import (
    EXPECTED_FAMILIES,
    EXPECTED_MONTHS,
    EXPECTED_PAIRS,
    DojoMonthlyMtmScoringError,
    score_monthly_mtm,
)


def _cell(*, multiple: float = 3.1, dd: float = 0.09, margin: float = 0.44) -> dict:
    return {
        "ending_multiple": multiple,
        "max_drawdown_fraction": dd,
        "peak_margin_fraction": margin,
        "margin_closeouts": 0,
        "evidence_complete": True,
        "verified_result_sha256": "1" * 64,
        "source_digest_sha256": "2" * 64,
        "evidence_tier": "WORN_RETROSPECTIVE_DIAGNOSTIC",
    }


def _month(month: str, *, multiple: float = 3.1) -> dict:
    cells = {
        scenario: {path: _cell(multiple=multiple) for path in ("OHLC", "OLHC")}
        for scenario in ("BASE", "STRESS")
    }
    lopo = {path: multiple for path in ("OHLC", "OLHC")}
    return {
        "month": month,
        "cells": cells,
        "pair_lopo": {label: dict(lopo) for label in EXPECTED_PAIRS},
        "family_lopo": {label: dict(lopo) for label in EXPECTED_FAMILIES},
    }


def _denominator() -> list[dict]:
    return [_month(month) for month in EXPECTED_MONTHS]


def test_exact_30_month_all_three_x_is_diagnostic_only() -> None:
    score = score_monthly_mtm(_denominator())
    assert score["month_count"] == 30
    assert score["every_month_3x"] is True
    assert score["research_gate_pass"] is True
    assert score["promotion_eligible"] is False
    assert score["live_permission"] is False
    assert score["order_authority"] == "NONE"
    assert score["promotion_blockers"] == [
        "HISTORICAL_WORN_DIAGNOSTIC_HAS_NO_PROMOTION_AUTHORITY"
    ]


def test_one_losing_month_cannot_be_hidden_by_large_average() -> None:
    rows = _denominator()
    rows[7] = _month(EXPECTED_MONTHS[7], multiple=0.8)
    rows[8] = _month(EXPECTED_MONTHS[8], multiple=100.0)
    score = score_monthly_mtm(rows)
    assert score["average_pessimistic_stress_multiple"] > 3.0
    assert score["every_month_3x"] is False
    assert score["research_gate_pass"] is False
    assert score["losing_months"] == [EXPECTED_MONTHS[7]]


def test_missing_or_extra_month_and_incomplete_lopo_denominator_are_rejected() -> None:
    with pytest.raises(DojoMonthlyMtmScoringError, match="exact 2024"):
        score_monthly_mtm(_denominator()[:-1])
    rows = _denominator()
    del rows[0]["pair_lopo"][EXPECTED_PAIRS[0]]
    with pytest.raises(DojoMonthlyMtmScoringError, match="exactly"):
        score_monthly_mtm(rows)


def test_risk_caps_are_fixed_at_normal10_stress15_and_margin45() -> None:
    rows = _denominator()
    rows[0]["cells"]["BASE"]["OHLC"] = _cell(dd=0.100001)
    rows[1]["cells"]["STRESS"]["OHLC"] = _cell(dd=0.150001)
    rows[2]["cells"]["BASE"]["OHLC"] = _cell(margin=0.450001)
    score = score_monthly_mtm(rows)
    assert score["months"][0]["gates"]["normal_drawdown"] is False
    assert score["months"][1]["gates"]["stress_drawdown"] is False
    assert score["months"][2]["gates"]["peak_margin"] is False
    assert score["research_gate_pass"] is False


def test_lopo_drop_is_path_aligned_and_all_labels_are_required() -> None:
    rows = _denominator()
    rows[0]["cells"]["STRESS"]["OHLC"] = _cell(multiple=10.0)
    rows[0]["cells"]["STRESS"]["OLHC"] = _cell(multiple=2.0)
    rows[0]["pair_lopo"][EXPECTED_PAIRS[0]] = {"OHLC": 5.0, "OLHC": 2.0}
    score = score_monthly_mtm(rows)
    concentration = score["months"][0]["pair_lopo_concentration"]
    first = concentration["rows"][0]
    assert first["path_aligned_profit_drop_fractions"]["OHLC"] == pytest.approx(5 / 9)
    assert concentration["gate_pass"] is False


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_result_is_rejected(bad: float) -> None:
    rows = _denominator()
    rows[0]["cells"]["STRESS"]["OHLC"]["ending_multiple"] = bad
    with pytest.raises(DojoMonthlyMtmScoringError, match="finite"):
        score_monthly_mtm(rows)


def test_hash_and_output_are_input_order_invariant() -> None:
    forward = score_monthly_mtm(_denominator())
    reverse = score_monthly_mtm(list(reversed(_denominator())))
    assert forward == reverse
