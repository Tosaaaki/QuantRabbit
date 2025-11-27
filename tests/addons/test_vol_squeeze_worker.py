from workers.vol_squeeze import VolSqueezeWorker, DEFAULT_CONFIG as CFG

class DummyFeed:
    def __init__(self, bars):
        self._bars = bars
    def get_bars(self, sym, tf, n):
        return self._bars[-n:]
    def last(self, sym):
        return self._bars[-1]["close"]

def _bars(n):
    out=[]; px=100.0
    for i in range(n):
        # make a squeeze then breakout
        if i < 80: px += (0.01*(i%5-2))
        else: px += (0.3 if i==100 else 0.02)
        out.append({"open":px, "high":px+0.1, "low":px-0.1, "close":px})
    return out

def test_construct_ok():
    bars = _bars(200)
    w = VolSqueezeWorker({**CFG, "universe":["X"], "place_orders": False}, broker=None, datafeed=DummyFeed(bars))
    intents = w.run_once()
    assert isinstance(intents, list)
