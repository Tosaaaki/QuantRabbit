"""Broaden the search: 28 pairs x a hold grid, with the multiplicity stated.

engine.py looked at 4 pairs and one 8-hour hold. The corpus carries 28 pairs, and
holding period is the axis the ORACLE work said the money lives on, so both are
widened here.

That makes 13 strategies x 4 holds = 52 portfolio candidates (pairs are pooled
per candidate, which is the portfolio a bot would actually run). At a one-sided
95% bar, roughly 2.6 of 52 are expected to pass on noise alone. So passing TEST
is NOT adoption -- it is a shortlist, and the count of candidates is printed
next to the count of passes so the two are always read together.

Everything else matches engine.py exactly: same causal features, same execution
model, same gate. Only the pair set and hold length change.

    python3 sweep.py                 # all 28 pairs, holds 60/120/240/480
    python3 sweep.py --holds 240,480
"""
import os, statistics, sys, glob
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import (load, features, roll_fwd, lower_bound, M1, WARM,
                    TRAIN_END, MIN_DAYS, DIS)          # noqa: E402
from strategies import REGISTRY                        # noqa: E402

HOLDS = [60, 120, 240, 480]


def all_pairs():
    seen = set()
    for d in glob.glob(f"{M1}/*/"):
        for name in os.listdir(d):
            if "_" in name and not name.endswith(".gz") and name != "summary.json":
                seen.add(name)
    return sorted(seen)


def run(pairs, holds):
    # (strategy, hold) -> split -> (pair, day) -> pips
    acc = defaultdict(lambda: {"TRAIN": defaultdict(float), "TEST": defaultdict(float)})
    trades = defaultdict(int)
    for pair in pairs:
        PIP = 0.01 if pair.endswith("JPY") else 0.0001
        try:
            ts, h, l, c, sp = load(pair)
        except Exception as exc:
            print(f"{pair}: 読み込み失敗 {exc}", file=sys.stderr)
            continue
        n = len(c)
        if n < 200000:
            print(f"{pair}: バー数不足 {n:,} — 除外", file=sys.stderr)
            continue
        half = statistics.median(sp) / 2
        f = features(ts, h, l, c)
        print(f"{pair}: {n:,} bars  half={half/PIP:.2f}p", file=sys.stderr)
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
                    day = ts[i][:10]
                    key = (name, H)
                    acc[key]["TRAIN" if day < TRAIN_END else "TEST"][(pair, day)] += r
                    trades[key] += 1
    return acc, trades


if __name__ == "__main__":
    holds = HOLDS
    if "--holds" in sys.argv:
        holds = [int(x) for x in sys.argv[sys.argv.index("--holds") + 1].split(",")]
    pairs = [a for a in sys.argv[1:] if "_" in a and not a.startswith("-")] or all_pairs()
    print(f"ペア {len(pairs)} / 戦略 {len(REGISTRY)} / 保有 {holds} "
          f"= 候補 {len(REGISTRY)*len(holds)} 通り", file=sys.stderr)
    acc, trades = run(pairs, holds)

    cands = len(REGISTRY) * len(holds)
    print(f"\n=== 28ペア横断スイープ ===")
    print(f"候補 {cands} 通り（戦略 {len(REGISTRY)} × 保有 {len(holds)}）、ペアはプールして"
          f"1候補＝1ポートフォリオ")
    print(f"片側95%の水準で、**偶然だけで約 {cands*0.05:.1f} 通りが合格しうる**\n")
    print(f"{'strategy':22s} {'hold':>5s} {'取引':>7s} | {'TR日':>5s} {'TR日次':>8s} | "
          f"{'TE日':>5s} {'TE日次':>8s} {'TE下限':>8s} {'TE勝日':>7s} {'合格':>5s}")
    print("-" * 104)
    rows = []
    for (name, H), d in acc.items():
        tr = list(d["TRAIN"].values()); te = list(d["TEST"].values())
        if not tr or not te:
            continue
        elb = lower_bound(te)
        ok = len(te) >= MIN_DAYS and sum(te) > 0 and elb > 0
        rows.append((elb, name, H, trades[(name, H)], tr, te, ok))
    for elb, name, H, nn, tr, te, ok in sorted(rows, reverse=True)[:25]:
        print(f"{name:22s} {H:5d} {nn:7d} | {len(tr):5d} {statistics.mean(tr):8.2f} | "
              f"{len(te):5d} {statistics.mean(te):8.2f} {elb:8.2f} "
              f"{100*sum(1 for x in te if x>0)/len(te):6.0f}% {'○' if ok else '×':>5s}")
    q = [r for r in rows if r[6]]
    print(f"\n合格 {len(q)}/{len(rows)}（偶然の期待数 ≈ {cands*0.05:.1f}）")
    for r in q:
        print(f"  ○ {r[1]}  hold={r[2]}  TE日次={statistics.mean(r[5]):+.2f} pips  下限={r[0]:+.2f}")
    if len(q) <= cands * 0.05:
        print("\n合格数が偶然の期待数を超えていない。**この探索面からは何も出ていない。**")
