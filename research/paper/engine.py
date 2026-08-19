"""Paper-trading engine — runs every registered strategy and scores it against
the gate that `QR_AI_STRATEGY_ALLOCATION_V1` actually requires.

The point is not to produce a backtest number. It is to produce, for each named
strategy, the exact evidence the contract amendment demands before capital may be
raised: an exact strategy identity, filled active days, cumulative P/L, and a
one-sided 95% lower bound over DAY means rather than per-trade iid.

Execution model, identical for every strategy so comparisons are honest:
  * entry at the M1 close of the signal bar, paying half the median spread
  * exit at the close `HOLD` bars later, paying half the spread again
  * a fixed disaster stop; because the stop price is fixed, a rolling forward
    min/max decides breach exactly, in O(n)
  * one position per strategy per pair at a time (cooldown = HOLD), so a day is
    not inflated by stacking the same signal
  * no TP, per the SL-free / TP-free house doctrine

Causal throughout: features at bar i use bars <= i; the forward window is only
ever used to price an entry that was already taken at i.

Results stay in pips. Converting to JPY needs a per-bar quote-currency rate the
corpus does not carry for USD-quoted pairs, and inventing one is exactly the
silent approximation that produced defect #4 in the previous session.

    python3 engine.py                  # full report, TRAIN 2020-2024 / TEST 2025-2026
    python3 engine.py USD_JPY EUR_USD  # subset of pairs
"""
import glob, gzip, json, os, statistics, sys
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategies import REGISTRY  # noqa: E402

M1 = "/Users/tossaki/App/QuantRabbit-live/logs/replay/oanda_history_m1_2020_2026"
PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD"]
HOLD = 480          # 8h in M1 bars
DIS = 50.0          # disaster stop, pips
WARM = 1500
Z = 1.645
MIN_DAYS = 5
TRAIN_END = "2025-01-01"


def load(pair):
    ts = []; h = []; l = []; c = []; sp = []
    for sh in sorted(glob.glob(f"{M1}/*/{pair}/{pair}_M1_BA_*.jsonl.gz")):
        with gzip.open(sh, "rt") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line); b, a = r["bid"], r["ask"]
                ts.append(r["time"])
                h.append((float(b["h"]) + float(a["h"])) / 2)
                l.append((float(b["l"]) + float(a["l"])) / 2)
                c.append((float(b["c"]) + float(a["c"])) / 2)
                sp.append(float(a["c"]) - float(b["c"]))
    return ts, h, l, c, sp


def roll_back(v, w, mx):
    """Trailing extreme over the previous w bars, inclusive of i."""
    out = [0.0] * len(v); dq = deque()
    for i, x in enumerate(v):
        while dq and ((v[dq[-1]] <= x) if mx else (v[dq[-1]] >= x)):
            dq.pop()
        dq.append(i)
        while dq[0] <= i - w:
            dq.popleft()
        out[i] = v[dq[0]]
    return out


def roll_fwd(v, w, mx):
    """Extreme over bars i+1 .. i+w. Read the front BEFORE pushing i, otherwise
    bar i contaminates its own forward window."""
    n = len(v); out = [0.0] * n; dq = deque()
    for i in range(n - 1, -1, -1):
        while dq and dq[0] > i + w:
            dq.popleft()
        out[i] = v[dq[0]] if dq else v[i]
        while dq and ((v[dq[-1]] <= v[i]) if mx else (v[dq[-1]] >= v[i])):
            dq.pop()
        dq.append(i)
    return out


def features(ts, h, l, c):
    n = len(c)
    atr = [0.0] * n; atr_slow = [0.0] * n
    e20 = [0.0] * n; e120 = [0.0] * n; e1440 = [0.0] * n
    eff = [0.0] * n; z20 = [0.0] * n; mom60 = [0.0] * n
    a = as_ = None; x20 = x120 = x1440 = c[0]; path = 0.0
    W = 360
    for i in range(1, n):
        tr = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        a = tr if a is None else a + (tr - a) / 14.0
        as_ = tr if as_ is None else as_ + (tr - as_) / 240.0
        atr[i] = a; atr_slow[i] = as_
        x20 += (c[i] - x20) * (2 / 21); e20[i] = x20
        x120 += (c[i] - x120) * (2 / 121); e120[i] = x120
        x1440 += (c[i] - x1440) * (2 / 1441); e1440[i] = x1440
        path += abs(c[i] - c[i - 1])
        drop = i - W + 1
        if drop >= 1:
            path -= abs(c[drop] - c[drop - 1])
        if i >= W - 1:
            eff[i] = abs(c[i] - c[i - W + 1]) / path if path else 0.0
        if a and a > 0:
            z20[i] = (c[i] - x20) / a
            mom60[i] = (c[i] - c[i - 60]) / a if i >= 60 else 0.0
    hi6 = roll_back(c, 360, True); lo6 = roll_back(c, 360, False)
    hi24 = roll_back(c, 1440, True); lo24 = roll_back(c, 1440, False)
    loc6 = [((c[i] - lo6[i]) / (hi6[i] - lo6[i])) if hi6[i] > lo6[i] else 0.5
            for i in range(n)]
    loc24 = [((c[i] - lo24[i]) / (hi24[i] - lo24[i])) if hi24[i] > lo24[i] else 0.5
             for i in range(n)]
    return {"c": c, "atr": atr, "atr_slow": atr_slow, "e20": e20, "e120": e120,
            "e1440": e1440, "eff": eff, "z20": z20, "mom60": mom60,
            "hi6h": hi6, "lo6h": lo6, "hi24h": hi24, "lo24h": lo24,
            "loc6h": loc6, "loc24h": loc24,
            "hour": [int(t[11:13]) for t in ts]}


def run_pair(pair, since=None):
    """Paper-trade every registered strategy on one pair.

    Returns {(strategy, day): pips}. `since` (YYYY-MM-DD) restricts which signal
    days are reported; warm-up and feature history always use the full series, so
    a restricted run produces exactly the same numbers as a full one.
    """
    PIP = 0.01 if pair.endswith("JPY") else 0.0001
    ts, h, l, c, sp = load(pair)
    n = len(c)
    half = statistics.median(sp) / 2
    f = features(ts, h, l, c)
    fmin = roll_fwd(l, HOLD, False)
    fmax = roll_fwd(h, HOLD, True)
    out = defaultdict(float)
    counts = defaultdict(int)
    for name, fn in REGISTRY.items():
        last = -10 ** 9
        for i in range(WARM, n - HOLD):
            if i - last < HOLD or f["atr"][i] <= 0:
                continue
            side = fn(f, i)
            if not side:
                continue
            last = i
            day = ts[i][:10]
            if since and day < since:
                continue
            px = c[i] + side * half
            if side > 0:
                worst = (fmin[i] + half - px) / PIP
                r = -DIS if worst <= -DIS else (c[i + HOLD] - half - px) / PIP
            else:
                worst = (px - (fmax[i] - half)) / PIP
                r = -DIS if worst <= -DIS else (px - (c[i + HOLD] + half)) / PIP
            out[(name, day)] += r
            counts[name] += 1
    return out, counts, half / PIP, n


def lower_bound(v):
    if len(v) < 2:
        return float("-inf")
    se = statistics.pstdev(v) / (len(v) ** 0.5)
    return statistics.mean(v) - Z * se if se > 0 else statistics.mean(v)


def main(pairs):
    split = defaultdict(lambda: {"TRAIN": defaultdict(float), "TEST": defaultdict(float)})
    total = defaultdict(int)
    for pair in pairs:
        res, counts, halfp, n = run_pair(pair)
        print(f"{pair}: {n:,} bars  half={halfp:.2f}p", file=sys.stderr)
        for (name, day), pips in res.items():
            split[name]["TRAIN" if day < TRAIN_END else "TEST"][(pair, day)] += pips
        for k, v in counts.items():
            total[k] += v

    print(f"\n試した戦略={len(REGISTRY)}  ペア={len(pairs)}  "
          f"執行=8時間保有 / 災害SL{DIS:.0f}pips / スプレッド半値を出入り両方")
    print(f"TRAIN < {TRAIN_END} <= TEST")
    print("合格条件（QR_AI_STRATEGY_ALLOCATION_V1）: "
          f"{MIN_DAYS}営業日以上 / 累積プラス / 日次平均の片側95%下限>0\n")
    print(f"{'strategy':22s} {'取引':>6s} | {'TR日':>5s} {'TR日次':>8s} {'TR下限':>8s} | "
          f"{'TE日':>5s} {'TE日次':>8s} {'TE下限':>8s} {'TE勝日':>7s} {'合格':>5s}")
    print("-" * 108)
    rows = []
    for name in REGISTRY:
        tr = list(split[name]["TRAIN"].values())
        te = list(split[name]["TEST"].values())
        if not tr or not te:
            continue
        elb = lower_bound(te)
        ok = len(te) >= MIN_DAYS and sum(te) > 0 and elb > 0
        rows.append((elb, name, total[name], tr, te, lower_bound(tr), ok))
    for elb, name, nn, tr, te, tlb, ok in sorted(rows, reverse=True):
        print(f"{name:22s} {nn:6d} | {len(tr):5d} {statistics.mean(tr):8.2f} {tlb:8.2f} | "
              f"{len(te):5d} {statistics.mean(te):8.2f} {elb:8.2f} "
              f"{100*sum(1 for x in te if x>0)/len(te):6.0f}% {'○' if ok else '×':>5s}")
    q = [r for r in rows if r[6]]
    print(f"\n合格 {len(q)}/{len(rows)}")
    for r in q:
        print(f"  ○ {r[1]}")
    print("\n注意: これは TEST 上の判定。TEST を見て戦略を選べば同じ罠に落ちる。")
    print("      採用前に forward paper（ledger.py）で未見の日を貯めて再確認すること。")


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("-")] or PAIRS)
