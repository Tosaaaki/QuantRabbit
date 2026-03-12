from workers.session_open import SessionOpenWorker, DEFAULT_CONFIG as CFG

class DummyFeed:
    def __init__(self, bars):
        self._bars = bars
    def get_bars(self, sym, tf, n):
        return self._bars[-n:]
    def last(self, sym):
        return self._bars[-1]["close"]
    def best_bid_ask(self, sym):
        p = self.last(sym)
        return (p*0.999, p*1.001)

def _bars(n, start_ts):
    out = []
    px = 100.0
    ts = start_ts
    for i in range(n):
        px += (1 if i%10==0 else 0) * 0.2
        out.append({"open":px, "high":px+0.2, "low":px-0.2, "close":px, "timestamp": ts})
        ts += 60
    return out

def test_construct_ok():
    import time
    now = 1730697600  # fixed ts
    bars = _bars(200, now-3600)
    w = SessionOpenWorker({**CFG, "universe":["X"], "sessions":[{"tz":"UTC","start":"07:00","build_minutes":10,"hold_minutes":120}], "place_orders": False},
                          broker=None, datafeed=DummyFeed(bars))
    intents = w.run_once(now=now)
    assert isinstance(intents, list)


def test_mk_order_requests_technical_context():
    worker = SessionOpenWorker(
        {**CFG, "universe": ["USD_JPY"], "place_orders": False},
        broker=None,
        datafeed=DummyFeed(_bars(200, 1730694000)),
    )

    order = worker._mk_order(
        "USD_JPY",
        {"side": "long", "px": 100.2, "meta": {}},
        size_mult=1.0,
        entry_probability=0.62,
    )

    assert order["technical_context_tfs"] == ["M1", "M5", "H1"]
    assert order["technical_context_fields"] == [
        "ma10",
        "ma20",
        "ema12",
        "ema24",
        "atr",
        "atr_pips",
        "adx",
        "plus_di",
        "minus_di",
        "bbw",
        "rsi",
        "macd",
    ]
    assert order["technical_context_ticks"] == [
        "latest_bid",
        "latest_ask",
        "latest_mid",
        "spread_pips",
        "tick_rate",
    ]
    assert order["technical_context_candle_counts"] == {"M1": 120, "M5": 90, "H1": 60}
