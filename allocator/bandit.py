from __future__ import annotations
from typing import Dict, Any, List, Callable
import random, math

def default_score(metrics: Dict[str, Any]) -> float:
    """
    Convert worker metrics to a [0..1] 'quality' score.
    metrics should include:
      - ev: expected value per trade or per unit risk (float)
      - hit: hit rate in [0..1]
      - sharpe: realized sharpe (float)
      - max_dd: max drawdown (negative or positive number; we penalize |dd|)
      - trades: number of observations
    """
    ev = float(metrics.get("ev", 0.0))
    hit = float(metrics.get("hit", 0.5))
    shp = float(metrics.get("sharpe", 0.0))
    dd  = float(metrics.get("max_dd", 0.0))
    n   = float(metrics.get("trades", 1.0))
    # softplus to keep positive scale, cap influence by sample size
    ev_term = 1.0/(1.0 + math.exp(-ev))
    hit_term = hit
    shp_term = 1.0/(1.0 + math.exp(-0.5*shp))
    dd_term = 1.0/(1.0 + math.exp( 0.5*abs(dd)))  # larger dd -> smaller
    obs_term = min(1.0, n / 50.0)  # saturate at 50 obs
    return 0.15*ev_term + 0.35*hit_term + 0.35*shp_term + 0.10*dd_term + 0.05*obs_term

class BanditAllocator:
    """
    Thompson Sampling over a Beta prior for 'success rate', blended with quality score.
    Each worker i has (wins_i, trades_i), plus optional quality score q_i in [0..1].
    We sample theta_i ~ Beta(1+wins, 1+(trades-wins)) and compute s_i = theta_i * (0.5 + 0.5*q_i).
    Budgets are proportional to s_i with caps and floor.
    """
    def __init__(self, total_budget_bps: float = 100.0, cap_bps: float = 40.0, floor_bps: float = 5.0,
                 quality_fn: Callable[[Dict[str, Any]], float] = default_score, seed: int = 0):
        self.total = float(total_budget_bps)
        self.cap = float(cap_bps)
        self.floor = float(floor_bps)
        self.qf = quality_fn
        self.rng = random.Random(seed)

    def allocate(self, metrics_by_worker: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        scores = {}
        for wid, m in metrics_by_worker.items():
            wins = max(0.0, float(m.get("wins", 0.0)))
            trades = max(wins, float(m.get("trades", wins)))
            alpha = 1.0 + wins
            beta  = 1.0 + max(0.0, trades - wins)
            theta = self._sample_beta(alpha, beta)
            q = max(0.0, min(1.0, self.qf(m)))
            s = theta * (0.5 + 0.5*q)
            scores[wid] = s

        # normalize
        tot = sum(scores.values()) or 1.0
        raw = {wid: s / tot for wid, s in scores.items()}
        # convert to bps with cap/floor
        out = {}
        remain = self.total
        # first assign floors
        keys = list(raw.keys())
        for wid in keys:
            out[wid] = self.floor
            remain -= self.floor
        if remain < 0:
            # not enough to meet floors; scale down
            scale = self.total / (self.floor * len(keys))
            return {wid: self.floor*scale for wid in keys}

        # distribute remainder by raw weights with cap
        for wid in keys:
            add = min(self.cap - out[wid], raw[wid] * remain)
            out[wid] += add

        # any leftover due to caps -> distribute round-robin under caps
        leftover = self.total - sum(out.values())
        if leftover > 1e-6:
            open_ids = [k for k in keys if out[k] < self.cap - 1e-9]
            while leftover > 1e-6 and open_ids:
                for wid in list(open_ids):
                    inc = min(leftover, min(1.0, self.cap - out[wid]))
                    out[wid] += inc
                    leftover -= inc
                    if out[wid] >= self.cap - 1e-9:
                        open_ids.remove(wid)
                    if leftover <= 1e-6:
                        break
        return out

    def _sample_beta(self, a: float, b: float) -> float:
        # simple Beta sampling via two Gamma draws
        x = self._sample_gamma(a, 1.0)
        y = self._sample_gamma(b, 1.0)
        return x/(x+y)

    def _sample_gamma(self, k: float, theta: float) -> float:
        # Marsaglia and Tsang method for k>1, and boost for k<=1
        import random, math
        if k < 1.0:
            # boost: Gamma(k,θ) = Gamma(k+1,θ)*U^(1/k)
            u = self.rng.random()
            return self._sample_gamma(k + 1.0, theta) * (u ** (1.0 / k))
        d = k - 1.0/3.0
        c = 1.0 / math.sqrt(9.0 * d)
        while True:
            x = self.rng.gauss(0, 1)
            v = (1 + c*x)
            if v <= 0:
                continue
            v = v*v*v
            u = self.rng.random()
            if u < 1 - 0.0331 * (x**4):
                return d * v * theta
            if math.log(u) < 0.5 * x*x + d * (1 - v + math.log(v)):
                return d * v * theta
