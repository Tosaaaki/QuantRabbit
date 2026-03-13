from __future__ import annotations

import asyncio
import pathlib
import sys
from types import SimpleNamespace

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import execution.order_manager as order_manager
from execution.order_manager import _neg_exit_decision


def test_neg_exit_policy_explicit_reason_allows_without_global_gate() -> None:
    allowed, near_be = _neg_exit_decision(
        exit_reason="trend_failure",
        est_pips=-5.0,
        emergency_allow=False,
        reason_allow=False,
        worker_allow=False,
        neg_policy={
            "enabled": True,
            "allow_reasons": ["trend_failure"],
            "deny_reasons": [],
        },
    )
    assert allowed is True
    assert near_be is False


def test_neg_exit_policy_blocks_reason_outside_strategy_allow_list() -> None:
    allowed, near_be = _neg_exit_decision(
        exit_reason="max_adverse",
        est_pips=-8.0,
        emergency_allow=False,
        reason_allow=True,  # global allow alone should not bypass strategy deny-by-omission
        worker_allow=False,
        neg_policy={
            "enabled": True,
            "allow_reasons": ["trend_failure"],
            "deny_reasons": [],
        },
    )
    assert allowed is False
    assert near_be is False


def test_neg_exit_policy_near_be_path_still_works() -> None:
    allowed, near_be = _neg_exit_decision(
        exit_reason="near_be",
        est_pips=-0.2,
        emergency_allow=False,
        reason_allow=False,
        worker_allow=False,
        neg_policy={
            "enabled": True,
            "allow_reasons": ["near_be"],
            "deny_reasons": [],
        },
    )
    assert near_be is True
    assert allowed is True


def test_neg_exit_policy_strict_no_negative_blocks_all() -> None:
    allowed, near_be = _neg_exit_decision(
        exit_reason="hard_stop",
        est_pips=-0.1,
        emergency_allow=True,
        reason_allow=True,
        worker_allow=True,
        neg_policy={
            "enabled": True,
            "strict_no_negative": True,
            "allow_reasons": ["hard_stop"],
            "deny_reasons": [],
        },
    )
    assert near_be is False
    assert allowed is False


def test_neg_exit_policy_strict_allows_explicit_override_reason() -> None:
    allowed, near_be = _neg_exit_decision(
        exit_reason="profit_bank_release",
        est_pips=-3.4,
        emergency_allow=False,
        reason_allow=False,
        worker_allow=False,
        neg_policy={
            "enabled": True,
            "strict_no_negative": True,
            "strict_allow_reasons": ["profit_bank_release"],
            "allow_reasons": ["profit_bank_release"],
            "deny_reasons": [],
        },
    )
    assert near_be is False
    assert allowed is True


def test_neg_exit_policy_wildcard_allows_when_reason_missing() -> None:
    allowed, near_be = _neg_exit_decision(
        exit_reason=None,
        est_pips=-1.2,
        emergency_allow=False,
        reason_allow=False,
        worker_allow=False,
        neg_policy={
            "enabled": True,
            "allow_reasons": ["*"],
            "deny_reasons": [],
        },
    )
    assert near_be is False
    assert allowed is True


def test_flow_neg_exit_policy_allows_de_risk_with_strategy_context() -> None:
    allowed, near_be = _neg_exit_decision(
        exit_reason="__de_risk__",
        est_pips=-1.8,
        emergency_allow=False,
        reason_allow=False,
        worker_allow=True,
        neg_policy=order_manager._strategy_neg_exit_policy("scalp_ping_5s_flow_live"),
    )
    assert near_be is False
    assert allowed is True


def test_default_neg_exit_policy_blocks_de_risk_without_strategy_context() -> None:
    allowed, near_be = _neg_exit_decision(
        exit_reason="__de_risk__",
        est_pips=-1.8,
        emergency_allow=False,
        reason_allow=False,
        worker_allow=True,
        neg_policy=order_manager._strategy_neg_exit_policy(None),
    )
    assert near_be is False
    assert allowed is False


@pytest.mark.parametrize(
    ("strategy_tag", "exit_reason", "worker_allow"),
    [
        ("RangeFader-buy-fade", "max_adverse", True),
        ("RangeFader-neutral-fade", "max_hold_loss", True),
        ("RangeFader-buy-fade", "reversion_pullback", False),
    ],
)
def test_rangefader_derived_tag_neg_exit_policy_keeps_default_and_reversion_reasons(
    strategy_tag: str,
    exit_reason: str,
    worker_allow: bool,
) -> None:
    allowed, near_be = _neg_exit_decision(
        exit_reason=exit_reason,
        est_pips=-2.4,
        emergency_allow=False,
        reason_allow=False,
        worker_allow=worker_allow,
        neg_policy=order_manager._strategy_neg_exit_policy(strategy_tag),
    )
    assert near_be is False
    assert allowed is True


def test_close_trade_uses_explicit_flow_context_for_negative_close(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def _fake_service_request(path, payload):
        seen["service_path"] = path
        seen["service_payload"] = dict(payload)
        return order_manager._ORDER_MANAGER_SERVICE_UNHANDLED

    def _fake_min_profit_pips(pocket, strategy_tag):
        seen["min_profit_pips"] = (pocket, strategy_tag)
        return None

    def _fake_fetch_quote(instrument):
        seen["fetch_quote"] = instrument
        return {"bid": 158.010, "ask": 158.030, "mid": 158.020}

    def _fake_strategy_neg_exit_policy(strategy_tag):
        seen["neg_exit_strategy"] = strategy_tag
        if strategy_tag != "scalp_ping_5s_flow_live":
            return {
                "enabled": True,
                "allow_reasons": [],
                "deny_reasons": [],
            }
        return {
            "enabled": True,
            "allow_reasons": ["__de_risk__"],
            "deny_reasons": [],
        }

    class _DummyTradeClose:
        def __init__(self, accountID, tradeID, data):
            self.accountID = accountID
            self.tradeID = tradeID
            self.data = data
            self.response = {}

    def _fake_api_request(req):
        seen["request_trade_id"] = req.tradeID
        seen["request_data"] = dict(req.data)
        req.response = {"orderFillTransaction": {"price": "158.015"}}

    monkeypatch.setattr(order_manager, "_order_manager_service_request_async", _fake_service_request)
    monkeypatch.setattr(order_manager, "_is_valid_live_trade_id", lambda _trade_id: True)
    monkeypatch.setattr(order_manager, "_exit_context_snapshot", lambda _reason: {})
    monkeypatch.setattr(order_manager, "_log_order", lambda **_kwargs: None)
    monkeypatch.setattr(order_manager, "_console_order_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(order_manager, "log_metric", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(order_manager, "_strategy_tag_from_client_id", lambda _client_id: None)
    monkeypatch.setattr(
        order_manager,
        "_load_exit_trade_context",
        lambda _trade_id, _client_id: {
            "entry_price": 158.050,
            "units": 1000,
        },
    )
    monkeypatch.setattr(order_manager, "_reject_exit_by_control", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(order_manager, "_min_profit_pips", _fake_min_profit_pips)
    monkeypatch.setattr(order_manager, "_latest_bid_ask", lambda: (None, None))
    monkeypatch.setattr(order_manager, "_fetch_quote", _fake_fetch_quote)
    monkeypatch.setattr(order_manager, "_estimate_trade_pnl_pips", lambda **_kwargs: -1.8)
    monkeypatch.setattr(order_manager, "_exit_end_reversal_eval", lambda **_kwargs: {})
    monkeypatch.setattr(order_manager, "_min_profit_ratio", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(order_manager, "_hold_until_profit_match", lambda *_args, **_kwargs: (False, 0.0, False))
    monkeypatch.setattr(order_manager, "_current_trade_unrealized_pl", lambda _trade_id: -18.0)
    monkeypatch.setattr(order_manager, "_should_allow_negative_close", lambda _client_id: False)
    monkeypatch.setattr(order_manager, "_reason_allows_negative", lambda _reason: False)
    monkeypatch.setattr(order_manager, "_reason_force_allow", lambda _reason: False)
    monkeypatch.setattr(order_manager, "_strategy_neg_exit_policy", _fake_strategy_neg_exit_policy)
    monkeypatch.setattr(order_manager, "_current_trade_units", lambda _trade_id: 1000)
    monkeypatch.setattr(order_manager, "is_market_open", lambda: True)
    monkeypatch.setattr(order_manager, "TradeClose", _DummyTradeClose)
    monkeypatch.setattr(order_manager, "api", SimpleNamespace(request=_fake_api_request))
    monkeypatch.setattr(order_manager, "_clear_strategy_control_exit_block", lambda **_kwargs: None)
    monkeypatch.setattr(order_manager, "_EXIT_NO_NEGATIVE_CLOSE", True)
    monkeypatch.setattr(order_manager, "_EXIT_ALLOW_NEGATIVE_BY_WORKER", True)

    ok = asyncio.run(
        order_manager.close_trade(
            "424165",
            -1000,
            client_order_id="cid-explicit-only",
            allow_negative=True,
            exit_reason="__de_risk__",
            strategy_tag="scalp_ping_5s_flow_live",
            pocket="scalp_fast",
            instrument="USD_JPY",
        )
    )

    assert ok is True
    assert seen["service_path"] == "/order/close_trade"
    assert seen["service_payload"] == {
        "trade_id": "424165",
        "units": -1000,
        "client_order_id": "cid-explicit-only",
        "allow_negative": True,
        "exit_reason": "__de_risk__",
        "strategy_tag": "scalp_ping_5s_flow_live",
        "pocket": "scalp_fast",
        "instrument": "USD_JPY",
    }
    assert seen["min_profit_pips"] == ("scalp_fast", "scalp_ping_5s_flow_live")
    assert seen["fetch_quote"] == "USD_JPY"
    assert seen["neg_exit_strategy"] == "scalp_ping_5s_flow_live"
    assert seen["request_trade_id"] == "424165"
    assert seen["request_data"] == {"units": "1000"}


def test_close_trade_blocks_extrema_candle_exit_until_tp_ratio(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    async def _fake_service_request(path, payload):
        return order_manager._ORDER_MANAGER_SERVICE_UNHANDLED

    monkeypatch.setattr(order_manager, "_order_manager_service_request_async", _fake_service_request)
    monkeypatch.setattr(order_manager, "_is_valid_live_trade_id", lambda _trade_id: True)
    monkeypatch.setattr(order_manager, "_exit_context_snapshot", lambda _reason: {})
    monkeypatch.setattr(order_manager, "_log_order", lambda **kwargs: captured.append(dict(kwargs)))
    monkeypatch.setattr(order_manager, "_console_order_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(order_manager, "log_metric", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(order_manager, "_strategy_tag_from_client_id", lambda _client_id: None)
    monkeypatch.setattr(
        order_manager,
        "_load_exit_trade_context",
        lambda _trade_id, _client_id: {
            "entry_price": 159.406,
            "units": -152,
            "tp_price": 159.383,
        },
    )
    monkeypatch.setattr(order_manager, "_reject_exit_by_control", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(order_manager, "_min_profit_pips", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(order_manager, "_latest_bid_ask", lambda: (159.393, 159.395))
    monkeypatch.setattr(order_manager, "_estimate_trade_pnl_pips", lambda **_kwargs: 1.2)
    monkeypatch.setattr(order_manager, "_exit_end_reversal_eval", lambda **_kwargs: {"triggered": False})
    monkeypatch.setattr(order_manager, "_min_profit_ratio", lambda *_args, **_kwargs: 0.60)
    monkeypatch.setattr(order_manager, "_min_profit_ratio_reasons", lambda *_args, **_kwargs: {"candle_*"})
    monkeypatch.setattr(order_manager, "_min_profit_ratio_min_tp_pips", lambda *_args, **_kwargs: 2.0)
    monkeypatch.setattr(order_manager, "_hold_until_profit_match", lambda *_args, **_kwargs: (False, 0.0, False))
    monkeypatch.setattr(order_manager, "_current_trade_unrealized_pl", lambda _trade_id: 18.24)
    monkeypatch.setattr(order_manager, "_should_allow_negative_close", lambda _client_id: False)
    monkeypatch.setattr(order_manager, "_reason_force_allow", lambda _reason: False)
    monkeypatch.setattr(order_manager, "is_market_open", lambda: True)

    ok = asyncio.run(
        order_manager.close_trade(
            "460291",
            152,
            client_order_id="cid-extrema",
            allow_negative=False,
            exit_reason="candle_bearish_engulfing",
            strategy_tag="scalp_extrema_reversal_live",
            pocket="scalp_fast",
            instrument="USD_JPY",
        )
    )

    assert ok is False
    assert any(str(item.get("status")) == "close_reject_profit_ratio" for item in captured)


def test_close_trade_allows_extrema_lock_floor_near_be(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def _fake_service_request(path, payload):
        seen["service_path"] = path
        seen["service_payload"] = dict(payload)
        return order_manager._ORDER_MANAGER_SERVICE_UNHANDLED

    class _DummyTradeClose:
        def __init__(self, accountID, tradeID, data):
            self.accountID = accountID
            self.tradeID = tradeID
            self.data = data
            self.response = {}

    def _fake_api_request(req):
        seen["request_trade_id"] = req.tradeID
        seen["request_data"] = dict(req.data)
        req.response = {"orderFillTransaction": {"price": "159.404"}}

    monkeypatch.setattr(order_manager, "_order_manager_service_request_async", _fake_service_request)
    monkeypatch.setattr(order_manager, "_is_valid_live_trade_id", lambda _trade_id: True)
    monkeypatch.setattr(order_manager, "_exit_context_snapshot", lambda _reason: {})
    monkeypatch.setattr(order_manager, "_log_order", lambda **_kwargs: None)
    monkeypatch.setattr(order_manager, "_console_order_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(order_manager, "log_metric", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(order_manager, "_strategy_tag_from_client_id", lambda _client_id: None)
    monkeypatch.setattr(
        order_manager,
        "_load_exit_trade_context",
        lambda _trade_id, _client_id: {
            "entry_price": 159.128,
            "units": -180,
            "tp_price": 159.094,
        },
    )
    monkeypatch.setattr(order_manager, "_reject_exit_by_control", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(order_manager, "_min_profit_pips", lambda *_args, **_kwargs: 0.1)
    monkeypatch.setattr(order_manager, "_latest_bid_ask", lambda: (159.137, 159.138))
    monkeypatch.setattr(order_manager, "_estimate_trade_pnl_pips", lambda **_kwargs: 0.2)
    monkeypatch.setattr(order_manager, "_exit_end_reversal_eval", lambda **_kwargs: {"triggered": False})
    monkeypatch.setattr(order_manager, "_min_profit_ratio", lambda *_args, **_kwargs: 0.60)
    monkeypatch.setattr(order_manager, "_min_profit_ratio_reasons", lambda *_args, **_kwargs: {"candle_*", "take_profit", "range_timeout"})
    monkeypatch.setattr(order_manager, "_min_profit_ratio_min_tp_pips", lambda *_args, **_kwargs: 2.0)
    monkeypatch.setattr(order_manager, "_hold_until_profit_match", lambda *_args, **_kwargs: (False, 0.0, False))
    monkeypatch.setattr(order_manager, "_current_trade_unrealized_pl", lambda _trade_id: 3.6)
    monkeypatch.setattr(order_manager, "_should_allow_negative_close", lambda _client_id: False)
    monkeypatch.setattr(order_manager, "_reason_force_allow", lambda _reason: False)
    monkeypatch.setattr(order_manager, "is_market_open", lambda: True)
    monkeypatch.setattr(order_manager, "TradeClose", _DummyTradeClose)
    monkeypatch.setattr(order_manager, "api", SimpleNamespace(request=_fake_api_request))
    monkeypatch.setattr(order_manager, "_clear_strategy_control_exit_block", lambda **_kwargs: None)

    ok = asyncio.run(
        order_manager.close_trade(
            "460187",
            180,
            client_order_id="cid-extrema-lock",
            allow_negative=False,
            exit_reason="lock_floor",
            strategy_tag="scalp_extrema_reversal_live",
            pocket="scalp_fast",
            instrument="USD_JPY",
        )
    )

    assert ok is True
    assert seen["service_path"] == "/order/close_trade"
    assert seen["service_payload"] == {
        "trade_id": "460187",
        "units": 180,
        "client_order_id": "cid-extrema-lock",
        "allow_negative": False,
        "exit_reason": "lock_floor",
        "strategy_tag": "scalp_extrema_reversal_live",
        "pocket": "scalp_fast",
        "instrument": "USD_JPY",
    }


def test_strategy_min_profit_buffers_allow_near_be_for_wick_profiles() -> None:
    from execution import order_manager

    assert order_manager._min_profit_pips("scalp", "PrecisionLowVol") == pytest.approx(0.1)
    assert order_manager._min_profit_pips("scalp", "DroughtRevert") == pytest.approx(0.1)
    assert order_manager._min_profit_pips("scalp", "WickReversalBlend") == pytest.approx(0.2)


def test_strategy_control_exit_failopen_threshold_path(monkeypatch) -> None:
    monkeypatch.setattr(order_manager, "_ORDER_STRATEGY_CONTROL_EXIT_FAILOPEN_ENABLED", True)
    monkeypatch.setattr(order_manager, "_ORDER_STRATEGY_CONTROL_EXIT_FAILOPEN_BLOCK_THRESHOLD", 3)
    monkeypatch.setattr(order_manager, "_ORDER_STRATEGY_CONTROL_EXIT_FAILOPEN_WINDOW_SEC", 60.0)
    monkeypatch.setattr(order_manager, "_ORDER_STRATEGY_CONTROL_EXIT_FAILOPEN_EMERGENCY_ONLY", False)

    assert (
        order_manager._strategy_control_exit_failopen_reason(
            block_count=2,
            block_age_sec=90.0,
            emergency_allow=False,
        )
        is None
    )
    assert (
        order_manager._strategy_control_exit_failopen_reason(
            block_count=3,
            block_age_sec=10.0,
            emergency_allow=False,
        )
        is None
    )
    assert (
        order_manager._strategy_control_exit_failopen_reason(
            block_count=3,
            block_age_sec=90.0,
            emergency_allow=False,
        )
        == "strategy_control_exit_failopen_threshold"
    )


def test_strategy_control_exit_failopen_emergency_only(monkeypatch) -> None:
    monkeypatch.setattr(order_manager, "_ORDER_STRATEGY_CONTROL_EXIT_FAILOPEN_ENABLED", True)
    monkeypatch.setattr(order_manager, "_ORDER_STRATEGY_CONTROL_EXIT_FAILOPEN_BLOCK_THRESHOLD", 2)
    monkeypatch.setattr(order_manager, "_ORDER_STRATEGY_CONTROL_EXIT_FAILOPEN_WINDOW_SEC", 30.0)
    monkeypatch.setattr(order_manager, "_ORDER_STRATEGY_CONTROL_EXIT_FAILOPEN_EMERGENCY_ONLY", True)

    assert (
        order_manager._strategy_control_exit_failopen_reason(
            block_count=2,
            block_age_sec=45.0,
            emergency_allow=False,
        )
        is None
    )
    assert (
        order_manager._strategy_control_exit_failopen_reason(
            block_count=2,
            block_age_sec=45.0,
            emergency_allow=True,
        )
        == "strategy_control_exit_failopen_emergency"
    )


def test_strategy_control_exit_immediate_bypass_reason_match(monkeypatch) -> None:
    monkeypatch.setattr(
        order_manager,
        "_ORDER_STRATEGY_CONTROL_EXIT_IMMEDIATE_BYPASS_REASONS",
        {"max_adverse", "time_stop"},
    )

    assert (
        order_manager._strategy_control_exit_immediate_bypass_reason(
            exit_reason="max_adverse",
        )
        == "strategy_control_exit_immediate_reason"
    )


def test_strategy_control_exit_immediate_bypass_reason_no_match(monkeypatch) -> None:
    monkeypatch.setattr(
        order_manager,
        "_ORDER_STRATEGY_CONTROL_EXIT_IMMEDIATE_BYPASS_REASONS",
        {"max_adverse", "time_stop"},
    )

    assert (
        order_manager._strategy_control_exit_immediate_bypass_reason(
            exit_reason="trail_lock",
        )
        is None
    )
