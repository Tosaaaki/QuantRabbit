from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quant_rabbit.micro_live_promotion_contract import (
    APPROVAL_CONTRACT,
    APPROVAL_STATEMENT,
    MicroLivePromotionError,
    _canonical_sha,
    build_micro_live_promotion_contract,
    validate_micro_live_activation,
)


NOW = datetime(2026, 7, 18, tzinfo=timezone.utc)


def _scorecard(**overrides: object) -> dict:
    body = {
        "valid_fill_count": 120,
        "filled_day_count": 12,
        "stressed_profit_factor": 1.08,
    }
    body.update(overrides)
    return body


def _approval(contract: dict, **overrides: object) -> dict:
    body = {
        "contract": APPROVAL_CONTRACT,
        "approved_contract_sha256": contract["contract_sha256"],
        "statement": APPROVAL_STATEMENT,
        "operator": "tosaki-yuki",
    }
    body.update(overrides)
    return {**body, "approval_sha256": _canonical_sha(body)}


def test_contract_seals_without_granting_any_authority() -> None:
    contract = build_micro_live_promotion_contract(
        lane_id="FAST_BOT_PRIMARY",
        shadow_scorecard=_scorecard(),
        scorecard_sha256="a" * 64,
        declared_at_utc=NOW,
    )

    assert contract["live_permission"] is False
    assert contract["order_authority"] == "NONE"
    assert contract["operator_approval_required"] is True
    assert contract["risk_budget"]["nav_risk_pool_fraction"] == 0.02
    body = {k: v for k, v in contract.items() if k != "contract_sha256"}
    assert contract["contract_sha256"] == _canonical_sha(body)


def test_admission_floor_is_fail_closed() -> None:
    for weak in (
        _scorecard(valid_fill_count=99),
        _scorecard(filled_day_count=9),
        _scorecard(stressed_profit_factor=1.04),
    ):
        with pytest.raises(MicroLivePromotionError, match="admission floor"):
            build_micro_live_promotion_contract(
                lane_id="FAST_BOT_PRIMARY",
                shadow_scorecard=weak,
                scorecard_sha256="a" * 64,
                declared_at_utc=NOW,
            )


def test_activation_requires_exact_operator_approval_binding() -> None:
    contract = build_micro_live_promotion_contract(
        lane_id="FAST_BOT_PRIMARY",
        shadow_scorecard=_scorecard(),
        scorecard_sha256="a" * 64,
        declared_at_utc=NOW,
    )

    review = validate_micro_live_activation(contract, _approval(contract))
    assert review["pair_coherent"] is True
    assert review["grants_live_permission"] is False
    assert review["runtime_revalidation_required"] is True

    with pytest.raises(MicroLivePromotionError, match="not bound"):
        validate_micro_live_activation(
            contract, _approval(contract, approved_contract_sha256="b" * 64)
        )
    with pytest.raises(MicroLivePromotionError, match="statement"):
        validate_micro_live_activation(
            contract, _approval(contract, statement="approved")
        )

    tampered = dict(contract)
    tampered["live_permission"] = True
    with pytest.raises(MicroLivePromotionError, match="digest"):
        validate_micro_live_activation(tampered, _approval(contract))
