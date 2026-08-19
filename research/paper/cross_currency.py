"""L028 — 通貨横断の相対情報。1ペアの過去以外の情報源を初めて使う。

L001〜L027 のルールは全部「そのペア自身の過去」から作られている。
1取引あたり 0.2〜3 pips という帯は、その情報源の限界かもしれない。

ペアは2通貨の比なので、28ペアを同時に見れば **通貨そのものの強弱** という
別の情報が作れる。引き継ぎ書 §6 が「28ペア全部を単独審判しただけ」として
未検証のまま残していた軸。

設計（事前宣言 L028）:
  各通貨の強さ = その通貨を含む全ペアでの符号調整済みリターンの平均
  最強通貨を買い、最弱通貨を売る。**それを表現する1ペアで建てる（2レッグにしない）**
  2レッグにしない理由: L009 で、レッグの数だけブローカーの取り分を払うと確定済み

方向は順張り・逆張りの両方を試す。保有は 1/3/5日（高回転は L027 で閉じた）。
コストはペア別の実約定スプレッド（L017）。全28ペアを使い、後から部分集合を選ばない。
"""
import glob, gzip, json, os, random, statistics, sys
from collections import defaultdict

M1 = "/Users/tossaki/App/QuantRabbit-live/logs/replay/oanda_history_m1_2020_2026"
CCY = ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"]
HOLDS = [1, 3, 5]                 # 日
WIDTHS = [1, 2]                   # 上位/下位 何通貨まで見るか
LOOKBACK = 5                      # 強弱を測る日数（事前に1つだけ固定）
TRAIN_END = "2024-01-01"
DRAWS = 400
random.seed(20260806)
# 台帳 ORDER_FILL 実測 + 同等ペアからの外挿（外挿は保守側に丸める）
SPREAD = {"USD_JPY": 0.80, "EUR_USD": 0.80, "GBP_USD": 1.30, "AUD_USD": 1.40,
          "NZD_USD": 1.50, "AUD_JPY": 1.60, "EUR_JPY": 1.80, "USD_CAD": 1.80,
          "USD_CHF": 1.60, "EUR_GBP": 1.40, "EUR_CHF": 1.60, "AUD_CAD": 2.50,
          "AUD_NZD": 2.70, "NZD_CAD": 2.70, "CAD_JPY": 2.30, "NZD_JPY": 2.70,
          "GBP_JPY": 3.20, "CHF_JPY": 3.50, "AUD_CHF": 1.50, "GBP_CHF": 2.40,
          "CAD_CHF": 2.50, "EUR_AUD": 2.80, "EUR_CAD": 2.40, "EUR_NZD": 3.50,
          "GBP_AUD": 3.20, "GBP_CAD": 3.00, "GBP_NZD": 4.00, "NZD_CHF": 3.00}


def pairs_available():
    out = set()
    for d in glob.glob(f"{M1}/*/"):
        for name in os.listdir(d):
            if "_" in name and not name.endswith(".gz") and name != "summary.json":
                out.add(name)
    return sorted(p for p in out if p[:3] in CCY and p[4:] in CCY)


def daily_close(pair):
    """日足終値のみ。O(n) で軽い。"""
    out = {}
    for sh in sorted(glob.glob(f"{M1}/*/{pair}/{pair}_M1_BA_*.jsonl.gz")):
        with gzip.open(sh, "rt") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                out[r["time"][:10]] = ((float(r["bid"]["c"]) + float(r["ask"]["c"])) / 2)
    return out


PAIRS = pairs_available()
print(f"利用可能ペア {len(PAIRS)}: {' '.join(PAIRS)}", file=sys.stderr)
close = {}
for p in PAIRS:
    close[p] = daily_close(p)
    print(f"{p} {len(close[p])}日", file=sys.stderr)

days = sorted(set().union(*[set(v) for v in close.values()]))
print(f"共通日数 {len(days)}  {days[0]}..{days[-1]}", file=sys.stderr)

# --- 通貨強弱（因果的: 当日の朝に既知＝前日終値までのリターン） -----------------
strength = {}
for i in range(LOOKBACK, len(days)):
    d = days[i]; d0 = days[i - LOOKBACK]
    acc = defaultdict(list)
    for p in PAIRS:
        a, b = p[:3], p[4:]
        if d not in close[p] or d0 not in close[p]:
            continue
        prev = days[i - 1]
        if prev not in close[p]:
            continue
        r = (close[p][prev] - close[p][d0]) / close[p][d0]
        acc[a].append(r); acc[b].append(-r)
    if len(acc) >= 6:
        strength[d] = {c: statistics.mean(v) for c, v in acc.items() if v}

print(f"強弱を作れた日 {len(strength)}", file=sys.stderr)


def express(long_c, short_c):
    """その通貨対を表現するペアと向き。無ければ None。"""
    p = f"{long_c}_{short_c}"
    if p in close:
        return p, +1
    p = f"{short_c}_{long_c}"
    if p in close:
        return p, -1
    return None, 0


def run(direction, hold, width):
    """direction: +1 順張り（強い通貨を買う）/ -1 逆張り"""
    byday = defaultdict(float)
    trades = 0; total = 0.0
    for i in range(LOOKBACK, len(days) - hold):
        d = days[i]
        if d not in strength:
            continue
        rank = sorted(strength[d].items(), key=lambda kv: -kv[1])
        if len(rank) < 2 * width:
            continue
        tops = [c for c, _ in rank[:width]]
        bots = [c for c, _ in rank[-width:]]
        if direction < 0:
            tops, bots = bots, tops
        for lc in tops:
            for sc in bots:
                pair, sign = express(lc, sc)
                if not pair:
                    continue
                d1 = days[i + hold]
                if d not in close[pair] or d1 not in close[pair]:
                    continue
                pip = 0.01 if pair.endswith("JPY") else 0.0001
                cost = SPREAD.get(pair, 3.0)
                move = (close[pair][d1] - close[pair][d]) / pip
                r = sign * move - cost
                byday[d] += r
                trades += 1; total += r
    return byday, trades, total


print(f"\n=== L028 通貨横断の相対情報 / lookback {LOOKBACK}日 / 全{len(PAIRS)}ペア ===")
print(f"探索面: 方向2 × 保有{len(HOLDS)} × 幅{len(WIDTHS)} = "
      f"{2*len(HOLDS)*len(WIDTHS)}設定（偶然の期待通過 約{2*len(HOLDS)*len(WIDTHS)*0.05:.1f}）\n")
print(f"{'方向':>5s} {'保有':>4s} {'幅':>3s} {'取引':>7s} {'net/取引':>9s} | "
      f"{'TR日次':>8s} {'TE日次':>8s} {'TE下限':>8s} {'乱数超え':>7s} {'合格':>5s}")
print("-" * 88)
rows = []
for direction, dname in ((+1, "順張り"), (-1, "逆張り")):
    for hold in HOLDS:
        for width in WIDTHS:
            byday, trades, total = run(direction, hold, width)
            if trades < 200:
                continue
            tr = [v for d, v in byday.items() if d < TRAIN_END]
            te = [v for d, v in byday.items() if d >= TRAIN_END]
            if len(tr) < 100 or len(te) < 50:
                continue
            mte = statistics.mean(te)
            se = statistics.pstdev(te) / (len(te) ** 0.5)
            lb = mte - 1.645 * se if se > 0 else mte
            pool = list(byday.values())
            null = [statistics.mean(random.sample(pool, len(te))) for _ in range(DRAWS)]
            pct = sum(1 for x in null if x < mte) / DRAWS
            ok = mte > 0 and lb > 0 and (statistics.mean(tr) > 0) == (mte > 0)
            rows.append((mte, dname, hold, width, trades, total / trades,
                         statistics.mean(tr), mte, lb, pct, ok))
            print(f"{dname:>5s} {hold:4d} {width:3d} {trades:7d} {total/trades:9.3f} | "
                  f"{statistics.mean(tr):8.2f} {mte:8.2f} {lb:8.2f} {pct:7.0%} "
                  f"{'○' if ok else '×':>5s}")

print("\n=== 棄却条件の判定 ===")
if not rows:
    print("評価可能な行なし")
else:
    pos = [r for r in rows if r[5] > 0]
    print(f"  条件1（1取引あたり純益がプラス）: {len(pos)}/{len(rows)}"
          f"{' → 全滅なら棄却' if not pos else ''}")
    best = max(rows)
    print(f"  最良: {best[1]} 保有{best[2]}日 幅{best[3]}  "
          f"net {best[5]:+.3f}/取引  TR {best[6]:+.2f} / TE {best[7]:+.2f}  "
          f"下限 {best[8]:+.2f}  乱数 {best[9]:.0%}")
    print(f"  条件2（TRAIN/TESTで符号反転）: "
          f"{'該当→棄却' if (best[6] > 0) != (best[7] > 0) else '通過'}")
    print(f"  条件3（乱数95%を超えない）: {'該当→棄却' if best[9] < 0.95 else '通過'}")
    print(f"\n3条件すべて通過: {sum(1 for r in rows if r[10])}/{len(rows)}")
