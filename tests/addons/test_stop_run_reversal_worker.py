from workers.stop_run_reversal import StopRunReversalWorker, DEFAULT_CONFIG as CFG

class DummyFeed:
    def __init__(self, bars):
        self._bars = bars
    def get_bars(self, sym, tf, n):
        return self._bars[-n:]
    def last(self, sym):
        return self._bars[-1]["close"]

def _bars():
    out=[]; px=100.0
    # make a stop-run candle with long upper wick
    for i in range(50):
        if i == 48:
            out.append({"open":px, "high":px+2.0, "low":px-0.2, "close":px+0.1})
            px += 0.1
        elif i == 49:
            out.append({"open":px, "high":px+0.1, "low":px-0.1, "close":px-1.0})
            px -= 1.0
        else:
            out.append({"open":px, "high":px+0.2, "low":px-0.2, "close":px})
    return out

def test_construct_ok():
    bars = _bars()
    w = StopRunReversalWorker({**CFG, "universe":["X"], "place_orders": False}, broker=None, datafeed=DummyFeed(bars))
    intents = w.run_once()
    assert isinstance(intents, list)
