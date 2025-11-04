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
