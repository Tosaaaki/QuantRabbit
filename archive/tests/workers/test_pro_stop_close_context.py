from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from workers.scalp_ping_5s_flow import pro_stop as flow_pro_stop
from workers.scalp_rangefader import pro_stop as range_pro_stop


def test_flow_pro_stop_forwards_explicit_close_context(monkeypatch):
    captured: dict[str, object] = {}

    async def _fake_close_trade(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return True

    monkeypatch.setattr(flow_pro_stop, "_close_trade", _fake_close_trade)
    monkeypatch.setattr(
        flow_pro_stop,
        "plan_pro_stop_closes",
        lambda *_args, **_kwargs: [{"reason": "pro_stop_loss"}],
    )
    monkeypatch.setattr(flow_pro_stop, "all_factors", lambda: {"M1": {}, "H4": {}})
    monkeypatch.setattr(flow_pro_stop, "log_metric", lambda *_args, **_kwargs: None)

    ok = asyncio.run(
        flow_pro_stop.maybe_close_pro_stop(
            {
                "trade_id": "T-flow-pro-stop",
                "units": 1000,
                "client_order_id": "cid-flow-pro-stop",
            },
            now=datetime.now(timezone.utc),
            pocket="scalp_fast",
            strategy_tag="scalp_ping_5s_flow_live",
            instrument="USD_JPY",
        )
    )

    assert ok is True
    assert captured["args"] == ("T-flow-pro-stop", -1000)
    assert captured["kwargs"] == {
        "client_order_id": "cid-flow-pro-stop",
        "allow_negative": True,
        "exit_reason": "pro_stop_loss",
        "strategy_tag": "scalp_ping_5s_flow_live",
        "pocket": "scalp_fast",
        "instrument": "USD_JPY",
    }


def test_range_pro_stop_forwards_explicit_close_context(monkeypatch):
    captured: dict[str, object] = {}

    async def _fake_close_trade(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return True

    monkeypatch.setattr(range_pro_stop, "_close_trade", _fake_close_trade)
    monkeypatch.setattr(
        range_pro_stop,
        "plan_pro_stop_closes",
        lambda *_args, **_kwargs: [{"reason": "range_pro_stop"}],
    )
    monkeypatch.setattr(range_pro_stop, "all_factors", lambda: {"M1": {}, "H4": {}})
    monkeypatch.setattr(range_pro_stop, "log_metric", lambda *_args, **_kwargs: None)

    ok = asyncio.run(
        range_pro_stop.maybe_close_pro_stop(
            {
                "trade_id": "T-range-pro-stop",
                "units": -250,
                "client_order_id": "cid-range-pro-stop",
            },
            now=datetime.now(timezone.utc),
            pocket="scalp",
            strategy_tag="RangeFader-buy-fade",
            instrument="USD_JPY",
        )
    )

    assert ok is True
    assert captured["args"] == ("T-range-pro-stop", 250)
    assert captured["kwargs"] == {
        "client_order_id": "cid-range-pro-stop",
        "allow_negative": True,
        "exit_reason": "range_pro_stop",
        "strategy_tag": "RangeFader-buy-fade",
        "pocket": "scalp",
        "instrument": "USD_JPY",
    }
