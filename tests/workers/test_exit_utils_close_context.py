from __future__ import annotations

import asyncio

from workers.scalp_ping_5s_flow import exit_utils as flow_exit_utils
from workers.scalp_rangefader import exit_utils as range_exit_utils


def test_flow_exit_utils_forwards_explicit_close_context(monkeypatch):
    captured: dict[str, object] = {}

    async def _fake_close_trade(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return True

    monkeypatch.setattr(flow_exit_utils, "_close_trade", _fake_close_trade)
    monkeypatch.setattr(flow_exit_utils, "_composite_exit_allowed", lambda **_kwargs: True)

    ok = asyncio.run(
        flow_exit_utils.close_trade(
            "424165",
            -1000,
            client_order_id="cid-flow",
            allow_negative=True,
            exit_reason="__de_risk__",
            env_prefix="SCALP_PING_5S",
            strategy_tag="scalp_ping_5s_flow_live",
            pocket="scalp_fast",
            instrument="USD_JPY",
        )
    )

    assert ok is True
    assert captured["args"] == ("424165", -1000)
    assert captured["kwargs"] == {
        "client_order_id": "cid-flow",
        "allow_negative": True,
        "exit_reason": "__de_risk__",
        "strategy_tag": "scalp_ping_5s_flow_live",
        "pocket": "scalp_fast",
        "instrument": "USD_JPY",
    }


def test_range_exit_utils_forwards_explicit_close_context(monkeypatch):
    captured: dict[str, object] = {}

    async def _fake_close_trade(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return True

    monkeypatch.setattr(range_exit_utils, "_close_trade", _fake_close_trade)
    monkeypatch.setattr(range_exit_utils, "_composite_exit_allowed", lambda **_kwargs: True)

    ok = asyncio.run(
        range_exit_utils.close_trade(
            "453760",
            -123,
            client_order_id="cid-range",
            allow_negative=True,
            exit_reason="max_hold_loss",
            env_prefix="RANGEFADER",
            strategy_tag="RangeFader-buy-fade",
            pocket="scalp",
            instrument="USD_JPY",
        )
    )

    assert ok is True
    assert captured["args"] == ("453760", -123)
    assert captured["kwargs"] == {
        "client_order_id": "cid-range",
        "allow_negative": True,
        "exit_reason": "max_hold_loss",
        "strategy_tag": "RangeFader-buy-fade",
        "pocket": "scalp",
        "instrument": "USD_JPY",
    }
