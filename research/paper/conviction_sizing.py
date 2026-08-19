"""L029 — サイジング。27実験で一度も測っていなかった軸。

L001〜L028 は全て固定サイズで測っている。ところが L005（初日）で自分が出した数字:

    人           加重 +141.5 pips → 348.0 JPY/pip → +49,247円
    M1Scalper    加重 +307.5 pips →  11.0 JPY/pip →  +3,375円

**人の2倍以上のpipsを取ったボットが、金額では7分の1。差は全部サイジング。**
それを見ておきながら、以後ずっと pips だけを探していた。

算数の訂正: `収益 = Σ(pips_i × size_i)`。固定サイズなら `mean(pips) × size` にしかならず、
**size と結果の相関ぶんを丸ごと捨てている**。3倍の要求「33.7 pips/日」は
正しくは「33.7 pips/日 × 一定サイズ」であって、サイズが効けば必要pipsは割り算で減る。

問い: エントリー時点で既知の情報から、その取引の結果を予測できるか。
      できるならサイズを比例させて、固定サイズを上回るか。

重要な制約: サイズを上げる方向は証拠金上限で頭打ち。なので現実には
**「悪い取引を小さくする」ことでしか効かない**。サイズは [0.2, 1.0] に制限する。

TRAIN で係数を推定し、TEST では**固定して**使う。TESTを見て係数を動かさない。
"""
import os, random, statistics, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import load, features, roll_fwd, WARM  # noqa: E402
from strategies import REGISTRY                    # noqa: E402

PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD",
         "AUD_JPY", "EUR_JPY", "GBP_JPY", "CAD_JPY", "CHF_JPY", "NZD_JPY"]
H = 2880
DIS = 50.0
TRAIN_END = "2025-01-01"
SIZE_MIN, SIZE_MAX = 0.2, 1.0
DRAWS = 400
random.seed(20260806)
REAL_SPREAD = {"USD_JPY": 0.80, "EUR_USD": 0.80, "GBP_USD": 1.30, "AUD_USD": 1.40,
               "NZD_USD": 1.50, "AUD_JPY": 1.60, "EUR_JPY": 1.80, "CAD_JPY": 2.30,
               "NZD_JPY": 2.70, "GBP_JPY": 3.20, "CHF_JPY": 3.50}


def trades_for(pair, name):
    """(features_at_entry, realised_pips, day) を返す。"""
    PIP = 0.01 if pair.endswith("JPY") else 0.0001
    ts, h, l, c, sp = load(pair)
    n = len(c)
    if n < 200000:
        return []
    half = REAL_SPREAD.get(pair, 2.0) / 2 * PIP
    f = features(ts, h, l, c)
    fmin = roll_fwd(l, H, False); fmax = roll_fwd(h, H, True)
    fn = REGISTRY[name]
    out = []
    last = -10 ** 9
    atr_hist = sorted(v for v in f["atr"][WARM:] if v > 0)

    def pct(v):
        if not atr_hist or v <= 0:
            return 0.5
        lo, hi = 0, len(atr_hist)
        while lo < hi:
            m = (lo + hi) // 2
            if atr_hist[m] < v:
                lo = m + 1
            else:
                hi = m
        return lo / len(atr_hist)

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
        feat = {
            "eff": f["eff"][i],
            "atr_pct": pct(f["atr"][i]),
            "spread_rel": (sp[i] / PIP) / max(f["atr"][i] / PIP, 1e-9),
            "z20_abs": abs(f["z20"][i]),
            "mom60_abs": abs(f["mom60"][i]),
            "loc_extreme": abs(f["loc24h"][i] - 0.5) * 2,
        }
        out.append((feat, r, ts[i][:10]))
    return out


NAME = sys.argv[1] if len(sys.argv) > 1 else "mom_break"
rows = []
for pair in PAIRS:
    got = trades_for(pair, NAME)
    rows += got
    print(f"{pair} {len(got)}件", file=sys.stderr)

tr = [x for x in rows if x[2] < TRAIN_END]
te = [x for x in rows if x[2] >= TRAIN_END]
print(f"\n=== L029 サイジング / {NAME}@{H} / 11ペア / TRAIN {len(tr)}件 TEST {len(te)}件 ===")
print(f"サイズは [{SIZE_MIN}, {SIZE_MAX}]。上げる方向は証拠金で頭打ちなので"
      f"**悪い取引を絞る**ことでしか効かない\n")

KEYS = ["eff", "atr_pct", "spread_rel", "z20_abs", "mom60_abs", "loc_extreme"]
print(f"{'特徴':13s} {'TRAIN相関':>10s} {'TEST相関':>10s} {'符号一致':>8s}  "
      f"{'TRAIN 下位1/3の平均':>18s} {'上位1/3の平均':>15s}")
print("-" * 84)


def corr(xs, ys):
    if len(xs) < 30:
        return 0.0
    mx = statistics.mean(xs); my = statistics.mean(ys)
    sx = statistics.pstdev(xs); sy = statistics.pstdev(ys)
    if sx == 0 or sy == 0:
        return 0.0
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (len(xs) * sx * sy)


usable = []
for k in KEYS:
    xs = [x[0][k] for x in tr]; ys = [x[1] for x in tr]
    ct = corr(xs, ys)
    ce = corr([x[0][k] for x in te], [x[1] for x in te])
    s = sorted(tr, key=lambda x: x[0][k])
    lo = statistics.mean([x[1] for x in s[:len(s) // 3]])
    hi = statistics.mean([x[1] for x in s[-len(s) // 3:]])
    agree = "○" if (ct > 0) == (ce > 0) and abs(ct) > 0.02 else "×"
    if agree == "○":
        usable.append((k, ct))
    print(f"{k:13s} {ct:10.3f} {ce:10.3f} {agree:>8s}  {lo:18.2f} {hi:15.2f}")

print(f"\nTRAIN/TESTで符号が一致した特徴: {len(usable)}/{len(KEYS)}"
      f" — {', '.join(k for k, _ in usable) if usable else 'なし'}")

if not usable:
    print("\n=== 棄却条件の判定 ===")
    print("  使える特徴がゼロ → **条件3で棄却**。サイズを決める材料が無い")
    sys.exit(0)

# TRAIN の分位で連続サイズを決める。係数は TEST で動かさない
def size_of(feat):
    sc = 0.0
    for k, ct in usable:
        vals = sorted(x[0][k] for x in tr)
        v = feat[k]
        lo, hi = 0, len(vals)
        while lo < hi:
            m = (lo + hi) // 2
            if vals[m] < v:
                lo = m + 1
            else:
                hi = m
        q = lo / len(vals)
        sc += (q if ct > 0 else 1 - q)
    q = sc / len(usable)
    return SIZE_MIN + (SIZE_MAX - SIZE_MIN) * q


fixed = statistics.mean([x[1] for x in te])
sized_num = sum(x[1] * size_of(x[0]) for x in te)
sized_den = sum(size_of(x[0]) for x in te)
sized = sized_num / sized_den if sized_den else 0.0
per_unit = sized_num / len(te)

null = []
sizes = [size_of(x[0]) for x in te]
for _ in range(DRAWS):
    sh = sizes[:]; random.shuffle(sh)
    num = sum(x[1] * s for x, s in zip(te, sh))
    den = sum(sh)
    null.append(num / den if den else 0.0)
pct = sum(1 for x in null if x < sized) / DRAWS

print(f"\n=== TEST での比較 ===")
print(f"  固定サイズ（1.0）      1取引あたり {fixed:+.3f} pips")
print(f"  サイズ連動             1取引あたり {sized:+.3f} pips（サイズ加重平均）")
print(f"  同・投入資本あたり     {per_unit:+.3f} pips（サイズ和で割らない＝実効収益）")
print(f"  平均サイズ             {statistics.mean(sizes):.3f}")
print(f"  乱数サイズ超え         {pct:.0%}")

print("\n=== 棄却条件の判定 ===")
print(f"  条件1（固定を上回らない）: {'該当→棄却' if sized <= fixed else '通過'}")
print(f"  条件2（乱数95%を超えない）: {'該当→棄却' if pct < 0.95 else '通過'}")
print(f"  条件3（符号反転）: 通過（{len(usable)}特徴が一致）")
