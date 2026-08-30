from __future__ import annotations

from datetime import datetime, timedelta, timezone

from quant_rabbit.fast_bot_shock_causal_audit import audit_shock_episode


AT = datetime(2026, 8, 28, 14, 3, 15, tzinfo=timezone.utc)


def _signal(signal_id: str, pair: str, side: str, method: str, seconds: int = 0):
    return {
        "signal_id": signal_id,
        "generated_at_utc": (AT + timedelta(seconds=seconds)).isoformat(),
        "pair": pair,
        "side": side,
        "method": method,
        "strategy_id": method.lower(),
        "regime_score": -2.0,
        "entry": 1.1 if pair == "EUR_USD" else 159.66,
        "spread_pips": 0.8,
        "stop_loss_pips": 3.2,
        "take_profit_pips": 2.4,
        "shadow_only": True,
        "live_permission": False,
    }


def _outcome(signal_id: str, *, filled: bool, pips: float, reason: str):
    return {
        "signal_id": signal_id,
        "filled": filled,
        "realized_pips": pips,
        "exit_reason": reason,
        "truth_request_coverage_proved": True,
        "truth_grid_slot_count": 197,
        "truth_no_tick_slot_count": 0,
    }


def test_actual_labels_drive_symmetric_mismatch_and_usdjpy_unfilled_reason():
    signals = [
        _signal("eu", "EUR_USD", "LONG", "RANGE_ROTATION"),
        _signal("uj", "USD_JPY", "LONG", "BREAKOUT_FAILURE"),
    ]
    outcomes = [
        _outcome("eu", filled=True, pips=-3.2, reason="STOP_LOSS_AMBIGUOUS_FILL_S5"),
        _outcome("uj", filled=False, pips=0.0, reason="UNFILLED"),
    ]
    result = audit_shock_episode(
        signals=signals,
        outcomes=outcomes,
        window_start_utc=AT - timedelta(minutes=1),
        window_end_utc=AT + timedelta(minutes=10),
        shock_at_utc=AT,
        shock_pair="EUR_USD",
        shock_direction="DOWN",
        historical_episode_class="CONTINUATION",
    )
    assert result["regime_transition_mismatch"]["count"] == 1
    assert result["timeline"][0]["side_relative_alignment"] == "COUNTERTREND"
    assert result["timeline"][0]["strategy_label_source"] == "ACTUAL_SIGNAL_LEDGER"
    assert result["usdjpy_participation_reason"] == "PASSIVE_LIMIT_NOT_TOUCHED_WITHIN_TTL"
    assert result["arms_same_proposal_stream"]["shock_freeze_5m"]["net_pips"] == 0.0
    assert result["execution_authority"] == "NONE"
    assert result["gateway_invocations"] == 0


def test_mirror_direction_uses_same_rule_and_missing_label_is_not_proxied():
    signal = _signal("up", "EUR_USD", "SHORT", "RANGE_ROTATION")
    signal.pop("strategy_id")
    result = audit_shock_episode(
        signals=[signal],
        outcomes=[_outcome("up", filled=True, pips=-3.2, reason="STOP_LOSS")],
        window_start_utc=AT - timedelta(minutes=1),
        window_end_utc=AT + timedelta(minutes=1),
        shock_at_utc=AT,
        shock_pair="EUR_USD",
        shock_direction="UP",
        historical_episode_class="CONTINUATION",
    )
    assert result["regime_transition_mismatch"]["count"] == 1
    assert result["strategy_label_coverage"]["missing"] == 1
    assert result["strategy_label_coverage"]["price_proxy_used"] is False
