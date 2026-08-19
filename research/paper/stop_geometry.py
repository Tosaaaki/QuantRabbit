"""出口の幾何 — 固定50pips が48時間保有を壊していた。

`mom_break_audit` の全11ペア集計で、**日次の中央値が -100.00**（＝典型的な1日が災害SLに
刺さっている）だった。平均を作っていたのは少数の大勝ち。これは戦略の問題ではなく
**幾何の不整合**である疑いが強い:

  * 固定 50 pips のSLは、保有8時間なら日次ボラに対して緩い
  * 同じ50 pips が、保有48時間では **ほぼ確実に触れる**。2日あればどの通貨も50 pips 動く
  * つまり保有を伸ばすほど「時間で出る」前に「ストップで出る」確率が1に近づき、
    伸ばした保有の意味が消える

しかもこの固定値は、このプロジェクト自身の承認済み幾何に反している。
memory `feedback_disaster_stop_approved`(2026-06-11) は災害ストップを
**H4 ATR × 2.5 × session** と定めており、固定 pips ではない。

なので3つの幾何を同一エントリー・同一保有で比較する:

    FIX50    固定 50 pips（これまでの実装）
    ATR2.5   2.5 x H4 ATR（承認済み幾何）
    NONE     ストップなし・時間出口のみ（SL-free の家の方針）

コストはペア別の**実約定スプレッド**（台帳 ORDER_FILL 495件の実測）を使う。
コーパス提示値は実際の2.12倍なので、それで採点すると幾何の差が見えない。
"""
import os, statistics, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import load, features, roll_fwd, lower_bound, WARM, TRAIN_END, MIN_DAYS  # noqa: E402
from strategies import REGISTRY                                                       # noqa: E402

PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD"]
HOLDS = [480, 1440, 2880]
# 台帳 execution_ledger.db の ORDER_FILL 495件、約定時点の fullPrice 板の中央値
REAL_SPREAD = {"USD_JPY": 0.80, "EUR_USD": 0.80, "GBP_USD": 1.30, "AUD_USD": 1.40}
GEOMS = ["FIX50", "ATR2.5", "NONE"]


def h4_atr(h, l, c, pip):
    """M1 を H4(240本) に集約して ATR14。M1 の各 index にマップして返す（pips）。"""
    n = len(c)
    out = [0.0] * n
    a = None
    prev_close = None
    i = 0
    cur = 0.0
    while i < n:
        j = min(i + 240, n)
        hh = max(h[i:j]); ll = min(l[i:j]); cc = c[j - 1]
        tr = (hh - ll) if prev_close is None else max(hh - ll, abs(hh - prev_close),
                                                      abs(ll - prev_close))
        a = tr if a is None else a + (tr - a) / 14.0
        prev_close = cc
        # このH4足の値は次のH4足から使う（因果性）
        for k in range(i, j):
            out[k] = cur / pip
        cur = a
        i = j
    return out


def run(pair):
    PIP = 0.01 if pair.endswith("JPY") else 0.0001
    ts, h, l, c, sp = load(pair)
    n = len(c)
    half = REAL_SPREAD.get(pair, 1.0) / 2 * PIP
    f = features(ts, h, l, c)
    atr4 = h4_atr(h, l, c, PIP)
    res = defaultdict(lambda: defaultdict(float))
    stopped = defaultdict(int); total = defaultdict(int)
    for H in HOLDS:
        fmin = roll_fwd(l, H, False)
        fmax = roll_fwd(h, H, True)
        for name, fn in REGISTRY.items():
            last = -10 ** 9
            for i in range(WARM, n - H):
                if i - last < H or f["atr"][i] <= 0 or atr4[i] <= 0:
                    continue
                side = fn(f, i)
                if not side:
                    continue
                last = i
                px = c[i] + side * half
                if side > 0:
                    worst = (fmin[i] + half - px) / PIP
                    timeexit = (c[i + H] - half - px) / PIP
                else:
                    worst = (px - (fmax[i] - half)) / PIP
                    timeexit = (px - (c[i + H] + half)) / PIP
                for g in GEOMS:
                    if g == "FIX50":
                        dist = 50.0
                    elif g == "ATR2.5":
                        dist = 2.5 * atr4[i]
                    else:
                        dist = None
                    if dist is not None and worst <= -dist:
                        r = -dist
                        stopped[(name, H, g)] += 1
                    else:
                        r = timeexit
                    total[(name, H, g)] += 1
                    res[(name, H, g)][ts[i][:10]] += r
    return res, stopped, total


agg = defaultdict(lambda: defaultdict(float))
ST = defaultdict(int); TO = defaultdict(int)
for pair in (sys.argv[1:] or PAIRS):
    print(f"{pair} ...", file=sys.stderr)
    r, st, to = run(pair)
    for k, dd in r.items():
        for d, v in dd.items():
            agg[k][d] += v
    for k in to:
        ST[k] += st[k]; TO[k] += to[k]

print(f"\n=== 出口の幾何の比較（4ペア・実約定スプレッド・保有{HOLDS}分）===")
print("FIX50=固定50pips / ATR2.5=2.5×H4 ATR（承認済み幾何）/ NONE=ストップなし\n")
print(f"{'strategy':22s} {'hold':>5s} {'幾何':>7s} {'SL率':>5s} | {'TR日次':>8s} | "
      f"{'TE日次':>8s} {'TE中央':>8s} {'TE下限':>8s} {'合格':>4s}")
print("-" * 92)
rows = []
for k, dd in agg.items():
    tr = [v for d, v in dd.items() if d < TRAIN_END]
    te = [v for d, v in dd.items() if d >= TRAIN_END]
    if len(tr) < 50 or len(te) < MIN_DAYS:
        continue
    elb = lower_bound(te)
    rows.append((statistics.mean(te), k, tr, te, elb, sum(te) > 0 and elb > 0))
for m, k, tr, te, elb, ok in sorted(rows, reverse=True)[:30]:
    rate = 100 * ST[k] / TO[k] if TO[k] else 0
    print(f"{k[0]:22s} {k[1]:5d} {k[2]:>7s} {rate:4.0f}% | {statistics.mean(tr):8.2f} | "
          f"{m:8.2f} {statistics.median(te):8.2f} {elb:8.2f} {'○' if ok else '×':>4s}")

print(f"\n=== 幾何ごとの要約（全戦略・全保有）===")
print(f"{'幾何':>8s} {'候補':>5s} {'SL率':>6s} {'TE日次の中央値':>14s} {'TE日次>0の候補':>14s}")
print("-" * 56)
for g in GEOMS:
    sub = [r for r in rows if r[1][2] == g]
    if not sub:
        continue
    rates = [100 * ST[r[1]] / TO[r[1]] for r in sub if TO[r[1]]]
    print(f"{g:>8s} {len(sub):5d} {statistics.mean(rates):5.0f}% "
          f"{statistics.median([r[0] for r in sub]):14.2f} "
          f"{sum(1 for r in sub if r[0] > 0):9d}/{len(sub):<4d}")
print(f"\n合格 {sum(1 for r in rows if r[5])}/{len(rows)}")
