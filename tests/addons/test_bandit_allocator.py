from allocator import BanditAllocator

def test_allocate_shape():
    alloc = BanditAllocator(total_budget_bps=100, cap_bps=40, floor_bps=5, seed=1)
    metrics = {
        "w1": {"wins":10, "trades":20, "ev":0.1, "hit":0.55, "sharpe":1.0, "max_dd":-0.05},
        "w2": {"wins":15, "trades":30, "ev":0.05, "hit":0.52, "sharpe":0.7, "max_dd":-0.06},
        "w3": {"wins":5,  "trades":25, "ev":0.02, "hit":0.48, "sharpe":0.4, "max_dd":-0.08},
    }
    out = alloc.allocate(metrics)
    assert abs(sum(out.values())-100.0) < 1e-6
    assert set(out.keys()) == {"w1","w2","w3"}
