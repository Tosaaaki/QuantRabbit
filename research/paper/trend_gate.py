"""「今トレンドか」を事前に判定できるか — L024 の実装。

L023 で `mom_break@2880` の成否は「そのペアがその期間トレンドしたか」で説明される
ところまで来た。ならば **トレンドかどうかを事前に判定できれば使える**。
L001 の判定器は「逆張りと順張りのどちらが勝つか」で落ちたが、それは2択の当てっこであって、
**「今トレンドか」の一点だけを問う検定はまだしていない。**

事前宣言（`docs/RESEARCH_LOG.md` L024）どおり:
  指標は EFF20 / ADX14 / RET20-over-ATR の3つだけ。チューニングしない
  閾値は TRAIN(2024) の分位で決め、TEST(2025-2026) にそのまま当てる
  棄却条件は3つ。特に「既にプラスのペアだけで改善」なら、時間ではなくペアを
  選び直しているだけなので棄却

全11ペア。部分集合は作らない（L020 の罠）。
コストはペア別の実約定スプレッド（L017）。
"""
import os, random, statistics, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import load, features, roll_fwd, WARM, TRAIN_END  # noqa: E402
from strategies import REGISTRY                               # noqa: E402

PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD",
         "AUD_JPY", "EUR_JPY", "GBP_JPY", "CAD_JPY", "CHF_JPY", "NZD_JPY"]
H = 2880
DIS = 50.0
NAME = "mom_break"
REAL_SPREAD = {"USD_JPY": 0.80, "EUR_USD": 0.80, "GBP_USD": 1.30, "AUD_USD": 1.40,
               "NZD_USD": 1.50, "AUD_JPY": 1.60, "EUR_JPY": 1.80, "CAD_JPY": 2.30,
               "NZD_JPY": 2.70, "GBP_JPY": 3.20, "CHF_JPY": 3.50}
QUANTILES = [0.50, 0.67, 0.80]      # 事前に3つだけ
DRAWS = 400
random.seed(20260806)


def daily_bars(ts, h, l, c):
    """M1 から日足 OHLC。日境界は UTC。"""
    days = []; O = []; Hh = []; Ll = []; C = []
    cur = None
    for i in range(len(c)):
        d = ts[i][:10]
        if d != cur:
            days.append(d); O.append(c[i]); Hh.append(h[i]); Ll.append(l[i]); C.append(c[i])
            cur = d
        else:
            if h[i] > Hh[-1]:
                Hh[-1] = h[i]
            if l[i] < Ll[-1]:
                Ll[-1] = l[i]
            C[-1] = c[i]
    return days, O, Hh, Ll, C


def trend_measures(days, Hh, Ll, C):
    """バー t までの日足だけで作る因果的な指標。値は「その日の朝に既知」。"""
    n = len(C)
    eff = [None] * n; adx = [None] * n; rat = [None] * n
    atr = [0.0] * n
    a = None
    tr_s = pdm = ndm = 0.0
    for i in range(1, n):
        tr = max(Hh[i] - Ll[i], abs(Hh[i] - C[i - 1]), abs(Ll[i] - C[i - 1]))
        a = tr if a is None else a + (tr - a) / 14.0
        atr[i] = a
        up = Hh[i] - Hh[i - 1]; dn = Ll[i - 1] - Ll[i]
        p = up if (up > dn and up > 0) else 0.0
        q = dn if (dn > up and dn > 0) else 0.0
        tr_s = tr if tr_s == 0 else tr_s + (tr - tr_s) / 14.0
        pdm = p if pdm == 0 else pdm + (p - pdm) / 14.0
        ndm = q if ndm == 0 else ndm + (q - ndm) / 14.0
        if tr_s > 0:
            pdi = 100 * pdm / tr_s; ndi = 100 * ndm / tr_s
            t = pdi + ndi
            adx[i] = 100 * abs(pdi - ndi) / t if t else 0.0
        if i >= 20:
            path = sum(abs(C[k] - C[k - 1]) for k in range(i - 19, i + 1))
            eff[i] = abs(C[i] - C[i - 20]) / path if path else 0.0
            if a and a > 0:
                rat[i] = abs(C[i] - C[i - 20]) / a
    # 「その日の朝に既知」= 前日終値までの値をその日に割り当てる
    out = {}
    for i in range(1, n):
        out[days[i]] = {"EFF20": eff[i - 1], "ADX14": adx[i - 1],
                        "RET20_ATR": rat[i - 1]}
    return out


def strategy_days(pair):
    PIP = 0.01 if pair.endswith("JPY") else 0.0001
    ts, h, l, c, sp = load(pair)
    n = len(c)
    if n < 200000:
        return None, None
    half = REAL_SPREAD.get(pair, 2.0) / 2 * PIP
    f = features(ts, h, l, c)
    fmin = roll_fwd(l, H, False); fmax = roll_fwd(h, H, True)
    fn = REGISTRY[NAME]
    byday = defaultdict(float)
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
        byday[ts[i][:10]] += r
    days, O, Hh, Ll, C = daily_bars(ts, h, l, c)
    return byday, trend_measures(days, Hh, Ll, C)


if __name__ == "__main__":
    perf = {}; meas = {}
    for pair in (sys.argv[1:] or PAIRS):
        print(f"{pair} ...", file=sys.stderr)
        bd, tm = strategy_days(pair)
        if bd:
            perf[pair] = bd; meas[pair] = tm

    print(f"\n=== L024 トレンド判定ゲート / {NAME}@{H} / {len(perf)}ペア / 実約定スプレッド ===")
    print("閾値は TRAIN(2024) の分位で決め、TEST(2025-)にそのまま適用。TESTを見て動かさない\n")
    print(f"{'指標':11s} {'分位':>5s} {'閾値':>8s} | {'TE無条件':>9s} {'TE通過':>9s} {'採用日':>7s} "
          f"{'改善':>8s} {'乱数超え':>7s} {'条件3':>18s}")
    print("-" * 104)

    rows = []
    for key in ("EFF20", "ADX14", "RET20_ATR"):
        trvals = [meas[p][d][key] for p in perf for d in perf[p]
                  if d < TRAIN_END and d in meas[p] and meas[p][d][key] is not None]
        if not trvals:
            continue
        trvals.sort()
        for q in QUANTILES:
            thr = trvals[int(len(trvals) * q)]
            uncond = []; gated = []
            per_pair = defaultdict(lambda: [[], []])
            for p in perf:
                for d, v in perf[p].items():
                    if d < TRAIN_END or d not in meas[p]:
                        continue
                    m = meas[p][d][key]
                    if m is None:
                        continue
                    uncond.append(v)
                    per_pair[p][0].append(v)
                    if m >= thr:
                        gated.append(v)
                        per_pair[p][1].append(v)
            if len(gated) < 30:
                continue
            mu = statistics.mean(uncond); mg = statistics.mean(gated)
            # 帰無: 同じ日数をランダムに選ぶ
            null = []
            for _ in range(DRAWS):
                pick = random.sample(uncond, len(gated))
                null.append(statistics.mean(pick) - mu)
            pct = sum(1 for x in null if x < mg - mu) / DRAWS
            # 条件3: 改善が既にプラスのペアだけで起きていないか
            improved_pos = improved_neg = 0
            for p, (allv, gv) in per_pair.items():
                if len(gv) < 10 or not allv:
                    continue
                base = statistics.mean(allv)
                if statistics.mean(gv) > base:
                    if base > 0:
                        improved_pos += 1
                    else:
                        improved_neg += 1
            c3 = f"改善: 元+{improved_pos} / 元-{improved_neg}"
            rows.append((mg - mu, key, q, thr, mu, mg, len(gated), pct, c3,
                         improved_neg))
            print(f"{key:11s} {q:5.2f} {thr:8.3f} | {mu:9.2f} {mg:9.2f} {len(gated):7d} "
                  f"{mg-mu:+8.2f} {pct:7.0%} {c3:>18s}")

    print("\n=== 棄却条件の判定 ===")
    best = max(rows, default=None)
    if not best:
        print("評価可能な行なし")
    else:
        d, key, q, thr, mu, mg, n, pct, c3, imp_neg = best
        print(f"最良: {key} 上位{100*(1-q):.0f}%  改善 {d:+.2f} pips/日  乱数超え {pct:.0%}  {c3}")
        print(f"  条件1（無条件を上回らない）: {'該当→棄却' if d <= 0 else '通過'}")
        print(f"  条件2（乱数分布に埋もれる）: {'該当→棄却' if pct < 0.95 else '通過'}")
        print(f"  条件3（元プラスのペアだけで改善）: "
              f"{'該当→棄却' if imp_neg == 0 else f'通過（元マイナス{imp_neg}ペアでも改善）'}")
