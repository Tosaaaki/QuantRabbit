from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workers.common import addon_live


class _DummyFeed:
    default_timeframe = "M5"

    def last(self, symbol: str) -> float:
        return 150.0

    def best_bid_ask(self, symbol: str):
        return (149.995, 150.005)

    def get_bars(self, symbol: str, tf: str, n: int):
        return [
            {"open": 150.0, "high": 150.1, "low": 149.9, "close": 150.0},
            {"open": 150.0, "high": 150.1, "low": 149.9, "close": 150.0},
        ]


def test_send_falls_back_to_worker_id_strategy_tag(monkeypatch):
    captured: dict[str, object] = {}

    async def _fake_market_order(**kwargs):
        captured.update(kwargs)
        return "TICKET-1"

    monkeypatch.setattr(addon_live, "market_order", _fake_market_order)
    monkeypatch.setattr(addon_live, "all_factors", lambda: {"M1": {}, "H4": {}})
    monkeypatch.setattr(
        addon_live,
        "get_account_snapshot",
        lambda: SimpleNamespace(nav=1_000_000.0, margin_available=800_000.0, margin_rate=0.04),
    )
    monkeypatch.setattr(addon_live, "allowed_lot", lambda *args, **kwargs: 0.1)
    monkeypatch.setattr(addon_live, "clamp_sl_tp", lambda **kwargs: (kwargs["sl"], kwargs["tp"]))

    broker = addon_live.AddonLiveBroker(
        worker_id="session_open",
        pocket="micro",
        datafeed=_DummyFeed(),
        exit_cfg={"stop_pips": 1.0, "tp_pips": 2.0},
    )

    ticket = broker.send(
        {
            "symbol": "USD_JPY",
            "side": "buy",
            "type": "market",
        }
    )

    assert ticket == "TICKET-1"
    assert captured.get("strategy_tag") == "session_open"
    thesis = captured.get("entry_thesis")
    assert isinstance(thesis, dict)
    assert thesis.get("strategy_tag") == "session_open"
    client_order_id = str(captured.get("client_order_id") or "")
    assert "-sessionopen-" in client_order_id
