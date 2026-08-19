"""Many strategies, complementing each other, at the measured cost.

Two corrections and one construction.

CORRECTION 1 — the cost was roughly double.
engine.py charged the corpus's own median spread, and HANDOFF §2.1 established
that the corpus spread is 1.70 pips against a measured effective 0.80 on real
fills. Every paper result so far was therefore taxed about twice what the broker
actually charges. cost_decomposition.py showed why that matters: 12 of 24
candidates have POSITIVE gross and are pushed negative by that inflated cost.

CORRECTION 2 — what may NOT be claimed.
Passive LIMIT entry would remove most of the remaining cost, and HANDOFF §3 found
post-fill drift on passive limits is favourable. It is deliberately NOT modelled
here: §2.2 established that M1 replay cannot measure passive-limit strategies at
all (33/33 M1 wins became 22/22 S5 losses). Correcting the spread LEVEL is
legitimate; simulating limit FILLS on M1 is defect #1. Only the level is changed.

CONSTRUCTION — do the strategies actually complement?
"Combining" cannot manufacture expectancy: E[sum] = sum of E, so a basket of
negative-mean strategies stays negative no matter how uncorrelated. What a basket
CAN do is cut variance, which lifts the one-sided lower bound the allocation gate
tests. So the basket is built from TRAIN only, evaluated on TEST, and the daily
correlation matrix is printed so the diversification is visible rather than
assumed.

    python3 portfolio.py                 # effective spread 0.80 round trip
    python3 portfolio.py --cost 1.70     # the corpus figure, for comparison
"""
import os, statistics, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import load, features, roll_fwd, lower_bound, WARM, DIS, TRAIN_END, MIN_DAYS  # noqa: E402
from strategies import REGISTRY                                                            # noqa: E402

PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD"]
HOLDS = [960, 1440, 2880]
EFFECTIVE_ROUND_TRIP = 0.80      # HANDOFF §2.1, measured on 509 real fills


def run(pair, holds, half_pips):
    PIP = 0.01 if pair.endswith("JPY") else 0.0001
    ts, h, l, c, sp = load(pair)
    n = len(c)
    half = half_pips * PIP
    f = features(ts, h, l, c)
    day = defaultdict(lambda: defaultdict(float))
    for H in holds:
        fmin = roll_fwd(l, H, False)
        fmax = roll_fwd(h, H, True)
        for name, fn in REGISTRY.items():
            last = -10 ** 9
            for i in range(WARM, n - H):
                if i - last < H or f["atr"][i] <= 0:
                    continue
                side = fn(f, i)
                if not side:
                    continue
                last = i
                px = c[i] + side * half
                if side > 0:
                    worst = (fmin[i] + half - px) / PIP
                    r = -DIS if worst <= -DIS else (c[i + H] - half - px) / PIP
                else:
                    worst = (px - (fmax[i] - half)) / PIP
                    r = -DIS if worst <= -DIS else (px - (c[i + H] + half)) / PIP
                day[(name, H)][ts[i][:10]] += r
    return day


def corr(a, b):
    ks = sorted(set(a) & set(b))
    if len(ks) < 30:
        return float("nan")
    x = [a[k] for k in ks]; y = [b[k] for k in ks]
    mx = statistics.mean(x); my = statistics.mean(y)
    sx = statistics.pstdev(x); sy = statistics.pstdev(y)
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (len(ks) * sx * sy)


cost = EFFECTIVE_ROUND_TRIP
if "--cost" in sys.argv:
    cost = float(sys.argv[sys.argv.index("--cost") + 1])
half_pips = cost / 2

merged = defaultdict(lambda: defaultdict(float))
for pair in PAIRS:
    print(f"{pair} ...", file=sys.stderr)
    for k, dd in run(pair, HOLDS, half_pips).items():
        for d, v in dd.items():
            merged[k][d] += v

print(f"\n=== 実効スプレッド {cost:.2f} pips 往復で採点（4ペア合算・保有 {HOLDS} 分）===")
print("HANDOFF §2.1: 実約定509件で実効0.80。コーパスの1.70はその2倍以上。")
print("受動指値は **モデル化しない**（§2.2: M1リプレイは受動指値を測れない＝欠陥#1）\n")
print(f"{'strategy':22s} {'hold':>5s} | {'TR日':>5s} {'TR日次':>8s} | "
      f"{'TE日':>5s} {'TE日次':>8s} {'TE下限':>8s} {'TE勝日':>7s} {'合格':>5s}")
print("-" * 92)
rows = []
for (name, H), dd in merged.items():
    tr = {d: v for d, v in dd.items() if d < TRAIN_END}
    te = {d: v for d, v in dd.items() if d >= TRAIN_END}
    if len(tr) < 50 or len(te) < MIN_DAYS:
        continue
    trv = list(tr.values()); tev = list(te.values())
    elb = lower_bound(tev)
    ok = sum(tev) > 0 and elb > 0
    rows.append((statistics.mean(trv), name, H, tr, te, elb, ok))
for m, name, H, tr, te, elb, ok in sorted(rows, reverse=True):
    tev = list(te.values())
    print(f"{name:22s} {H:5d} | {len(tr):5d} {m:8.2f} | {len(te):5d} "
          f"{statistics.mean(tev):8.2f} {elb:8.2f} "
          f"{100*sum(1 for x in tev if x>0)/len(tev):6.0f}% {'○' if ok else '×':>5s}")

# --- basket, selected on TRAIN only ---------------------------------------
picked = [r for r in rows if r[0] > 0]
print(f"\n=== バスケット（TRAIN の日次平均がプラスのものだけを採用。TEST は見ていない）===")
print(f"採用 {len(picked)} 本: {', '.join(f'{r[1]}@{r[2]}' for r in picked) or 'なし'}")
if len(picked) >= 2:
    print(f"\n日次リターンの相関（TRAIN）:")
    hdr = "".join(f"{r[1][:9]:>10s}" for r in picked)
    print(f"{'':22s}{hdr}")
    for a in picked:
        line = "".join(f"{corr(a[3], b[3]):10.2f}" for b in picked)
        print(f"{a[1][:20]:22s}{line}")
    days = defaultdict(float)
    for r in picked:
        for d, v in {**r[3], **r[4]}.items():
            days[d] += v / len(picked)          # equal weight
    tr = [v for d, v in days.items() if d < TRAIN_END]
    te = [v for d, v in days.items() if d >= TRAIN_END]
    elb = lower_bound(te)
    print(f"\nバスケット TRAIN {len(tr)}日 日次 {statistics.mean(tr):+.2f} pips")
    print(f"バスケット TEST  {len(te)}日 日次 {statistics.mean(te):+.2f} pips  "
          f"片側95%下限 {elb:+.2f}  勝ち日 {100*sum(1 for x in te if x>0)/len(te):.0f}%")
    print(f"合格: {'○' if (len(te)>=MIN_DAYS and sum(te)>0 and elb>0) else '×'}")
    best = max((statistics.mean(list(r[4].values())), r[1]) for r in picked)
    print(f"\n単体最良の TEST 日次 {best[0]:+.2f}（{best[1]}）に対しバスケットは "
          f"{statistics.mean(te):+.2f}")
    print("**組み合わせは平均を作らない（E[和]=和[E]）。効くのは分散＝下限のほう。**")
