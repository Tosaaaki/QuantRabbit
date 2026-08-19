"""Where does the loss actually come from — the signal, or the spread?

Every one of 52 candidates lost, and they lost by suspiciously similar amounts
(-2 to -6 pips/day). That pattern is what a fixed per-trade tax looks like, not
what thirteen different broken ideas look like. So before tuning any threshold,
split each strategy's result into:

    GROSS   the same entries and exits with NO spread charged
    COST    what the round-trip spread took
    NET     what the engine reported

If GROSS is positive and NET is negative, the signals are not the problem and no
amount of threshold tuning fixes it — the fix is execution: fewer trades, longer
holds, or passive LIMIT entry instead of paying the spread to get in. HANDOFF
§2.1 measured the real effective spread at 0.80 pips and §3 found post-fill drift
on passive limits is *favourable* (+0.055 to +0.177), so that lever is real.

If GROSS is also negative, the signals genuinely have no edge and execution
changes cannot rescue them. Either way this says which of the two to work on,
which "it lost" does not.
"""
import os, statistics, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import load, features, roll_fwd, WARM, DIS, TRAIN_END  # noqa: E402
from strategies import REGISTRY                                    # noqa: E402

PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD"]
HOLDS = [480, 960, 1440, 2880]


def run(pair, holds):
    PIP = 0.01 if pair.endswith("JPY") else 0.0001
    ts, h, l, c, sp = load(pair)
    n = len(c)
    half = statistics.median(sp) / 2
    f = features(ts, h, l, c)
    out = defaultdict(lambda: {"gross": 0.0, "net": 0.0, "n": 0})
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
                for tag, hh in (("net", half), ("gross", 0.0)):
                    px = c[i] + side * hh
                    if side > 0:
                        worst = (fmin[i] + hh - px) / PIP
                        r = -DIS if worst <= -DIS else (c[i + H] - hh - px) / PIP
                    else:
                        worst = (px - (fmax[i] - hh)) / PIP
                        r = -DIS if worst <= -DIS else (px - (c[i + H] + hh)) / PIP
                    out[(name, H)][tag] += r
                out[(name, H)]["n"] += 1
    return out, half / PIP


agg = defaultdict(lambda: {"gross": 0.0, "net": 0.0, "n": 0})
halves = {}
for pair in (sys.argv[1:] or PAIRS):
    o, hp = run(pair, HOLDS)
    halves[pair] = hp
    print(f"{pair}: half={hp:.2f}p", file=sys.stderr)
    for k, v in o.items():
        for kk in ("gross", "net", "n"):
            agg[k][kk] += v[kk]

print(f"\n=== コスト分解（4ペア合算・{HOLDS} 分保有）===")
print("往復コスト = スプレッド半値 × 2。GROSS はスプレッドを一切課さない同一エントリー。\n")
print(f"{'strategy':22s} {'hold':>5s} {'取引':>7s} {'GROSS/取引':>11s} {'COST/取引':>10s} "
      f"{'NET/取引':>10s}  判定")
print("-" * 92)
rows = []
for (name, H), v in agg.items():
    if v["n"] < 100:
        continue
    g = v["gross"] / v["n"]; nt = v["net"] / v["n"]
    rows.append((g, name, H, v["n"], g, nt))
for g, name, H, nn, gross, net in sorted(rows, reverse=True):
    verdict = ("シグナルは黒→執行の問題" if gross > 0 > net else
               ("シグナルも黒" if gross > 0 and net > 0 else "シグナル自体が赤"))
    print(f"{name:22s} {H:5d} {nn:7d} {gross:11.3f} {gross-net:10.3f} {net:10.3f}  {verdict}")

pos = [r for r in rows if r[4] > 0]
print(f"\nGROSS がプラスの候補: {len(pos)}/{len(rows)}")
if pos:
    print("→ **シグナルには符号がある。負けているのは執行コスト。**")
    print("   打ち手は閾値いじりではなく: 取引数を減らす / 保有を伸ばす / 受動指値で入る")
    print("   （HANDOFF §2.1 実効スプレッド0.80、§3 受動指値の約定後ドリフトは +0.055〜+0.177 で有利）")
else:
    print("→ **シグナル自体が赤。執行を直しても救えない。** 閾値調整は無意味。")
