from __future__ import annotations

from datetime import datetime, timezone

from quant_rabbit.fast_bot_shadow_orchestrator import (
    SHADOW_ADMISSION_BINDING_CONTRACT_V1,
    _canonical_sha,
    build_fast_bot_shadow_admission_binding_v1,
)

UTC = timezone.utc
MIDWEEK = datetime(2026, 7, 15, 9, 0, tzinfo=UTC)
FRIDAY_LATE = datetime(2026, 7, 17, 20, 0, tzinfo=UTC)
CYCLE = "20260715T090000Z-SHADOW"

OPEN_POSITIONS = [
    {"pair": "EUR_USD", "side": "LONG", "nav_exposure_fraction": 0.2},
    {"pair": "USD_JPY", "side": "SHORT", "nav_exposure_fraction": 0.2},
]


def test_midweek_candidate_admission_binds_all_three_gates() -> None:
    binding = build_fast_bot_shadow_admission_binding_v1(
        cycle_id=CYCLE,
        decision_utc=MIDWEEK,
        hold_minutes=180,
        open_positions=OPEN_POSITIONS,
        candidates=[
            {"pair": "GBP_USD", "side": "LONG", "nav_exposure_fraction": 0.2},
            {"pair": "USD_CHF", "side": "LONG", "nav_exposure_fraction": 0.2},
        ],
    )

    assert binding["contract"] == SHADOW_ADMISSION_BINDING_CONTRACT_V1
    rows = {row["pair"]: row for row in binding["candidate_rows"]}
    # GBP_USD LONG stacks USD shorts to -0.6 -> refused; USD_CHF hedges back.
    assert rows["GBP_USD"]["admitted"] is False
    assert rows["GBP_USD"]["refusal_reasons"] == ["CURRENCY_EXPOSURE_CAP_EXCEEDED"]
    assert rows["USD_CHF"]["admitted"] is True
    assert binding["admitted_candidate_count"] == 1
    assert binding["go_risk_jpy"] == 0.0
    assert binding["order_intents"] == []
    body = {key: value for key, value in binding.items() if key != "contract_sha256"}
    assert binding["contract_sha256"] == _canonical_sha(body)


def test_close_crossing_cycle_refuses_every_candidate() -> None:
    binding = build_fast_bot_shadow_admission_binding_v1(
        cycle_id=CYCLE,
        decision_utc=FRIDAY_LATE,
        hold_minutes=720,
        open_positions=[],
        candidates=[
            {"pair": "EUR_USD", "side": "LONG", "nav_exposure_fraction": 0.1}
        ],
    )

    assert binding["admitted_candidate_count"] == 0
    reasons = binding["candidate_rows"][0]["refusal_reasons"]
    assert "HOLD_WOULD_CROSS_NEXT_FX_CLOSE" in reasons
    assert binding["future_go_contract_must_bind_this_artifact"] is True
    assert binding["close_distance_gate"]["uses_post_entry_information"] is False
