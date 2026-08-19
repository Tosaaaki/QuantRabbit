"""L033 — 構造に紐づくナンピンとピラミッティング。

ゆうきさんの説明を逐語で機械化する:

  「チャートの形が良いと思った。**長いタイムフレームでみて、ここまでは下がらないだろう、
    ここまではいくだろう**と判断した。平均取得単価を有利な方向へもっていった。」

L032 との決定的な差は **下限の線を見るか / 割ったら切るか**。
L032 は `-X ATR` で機械的に足すだけで、下限を割っても足し続けた。
それはナンピンではなく無条件の倍賭けで、台帳の `pullback_s5` -36,099円がまさにそれ。

機械化:
  下限 = H4 の直近スイング安値（ロング時）。「ここまでは下がらない」の線
  目標 = H4 の直近スイング高値。「ここまではいく」の線
  ナンピン  = 下限を割らない範囲で、押したら1回追加（平均取得単価を有利に）
  無効化    = **H4終値が下限を割ったら全部切る**（L032 に無かった要素）
  ピラミッド = 目標に向かって順行したら1回追加

比較: 単独 / ナンピン / ピラミッド / 両方。
コストはペア別の実約定スプレッド。追加分も建値で正しくスプレッドを払う。
"""
import os, statistics, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import load, features, WARM   # noqa: E402
from strategies import REGISTRY           # noqa: E402

PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD",
         "AUD_JPY", "EUR_JPY", "GBP_JPY", "CAD_JPY", "CHF_JPY", "NZD_JPY"]
H4 = 240                 # H4 = M1 240本
SWING = 5                # スイング判定: 前後5本の極値
MAX_HOLD = 2880          # 48時間で時間切れ
NAME = "mom_break"
TRAIN_END = "2025-01-01"
REAL_SPREAD = {"USD_JPY": 0.80, "EUR_USD": 0.80, "GBP_USD": 1.30, "AUD_USD": 1.40,
               "NZD_USD": 1.50, "AUD_JPY": 1.60, "EUR_JPY": 1.80, "CAD_JPY": 2.30,
               "NZD_JPY": 2.70, "GBP_JPY": 3.20, "CHF_JPY": 3.50}
MODES = ["単独", "ナンピン", "ピラミッド", "両方"]


def h4_swings(h, l, c):
    """H4足を作り、確定したスイング高値/安値をM1indexへマップ（因果的）。
    スイングは前後SWING本が確定して初めて既知になるので、SWING本ぶん遅延させる。"""
    n = len(c)
    hi = []; lo = []; idx = []
    i = 0
    while i < n:
        j = min(i + H4, n)
        hi.append(max(h[i:j])); lo.append(min(l[i:j])); idx.append(j - 1)
        i = j
    m = len(hi)
    sw_hi = [None] * n; sw_lo = [None] * n
    last_hi = last_lo = None
    for k in range(m):
        # k がスイングとして確定するのは k+SWING 本目以降
        if SWING <= k < m - SWING:
            if hi[k] == max(hi[k - SWING:k + SWING + 1]):
                last_hi = hi[k]
            if lo[k] == min(lo[k - SWING:k + SWING + 1]):
                last_lo = lo[k]
        known_from = idx[min(k + SWING, m - 1)]
        stop = idx[min(k + SWING + 1, m - 1)]
        for x in range(known_from, stop):
            sw_hi[x] = last_hi; sw_lo[x] = last_lo
    # 末尾を埋める
    for x in range(n):
        if sw_hi[x] is None and last_hi is not None and x > idx[-1]:
            sw_hi[x] = last_hi
        if sw_lo[x] is None and last_lo is not None and x > idx[-1]:
            sw_lo[x] = last_lo
    return sw_hi, sw_lo


def run(pair):
    PIP = 0.01 if pair.endswith("JPY") else 0.0001
    ts, h, l, c, sp = load(pair)
    n = len(c)
    if n < 200000:
        return None
    half = REAL_SPREAD.get(pair, 2.0) / 2 * PIP
    f = features(ts, h, l, c)
    sw_hi, sw_lo = h4_swings(h, l, c)
    fn = REGISTRY[NAME]
    out = defaultdict(lambda: defaultdict(float))
    stat = defaultdict(lambda: [0, 0, 0.0, 0])   # trades, wins, sum, adds
    last = -10 ** 9
    for i in range(WARM, n - MAX_HOLD):
        if i - last < MAX_HOLD or f["atr"][i] <= 0:
            continue
        side = fn(f, i)
        if not side:
            continue
        floor = sw_lo[i] if side > 0 else sw_hi[i]
        target = sw_hi[i] if side > 0 else sw_lo[i]
        if floor is None or target is None:
            continue
        entry = c[i] + side * half
        # 下限が既に反対側／目標が既に到達済みなら、そもそも形が成立していない
        if side * (entry - floor) <= 0 or side * (target - entry) <= 0:
            continue
        last = i
        span = side * (target - entry)
        room = side * (entry - floor)
        for mode in MODES:
            units = 1.0
            cost_basis = entry
            added_dn = added_up = False
            r = None
            for j in range(i + 1, i + MAX_HOLD + 1):
                px_lo = l[j]; px_hi = h[j]
                # --- 無効化: 下限を割ったら全部切る（L032 に無かった要素） ---
                broke = (px_lo < floor) if side > 0 else (px_hi > floor)
                if broke:
                    exitpx = floor - side * half
                    r = units * side * (exitpx - cost_basis) / PIP
                    break
                # --- 目標到達で利確 ---
                reached = (px_hi >= target) if side > 0 else (px_lo <= target)
                if reached:
                    exitpx = target - side * half
                    r = units * side * (exitpx - cost_basis) / PIP
                    break
                # --- ナンピン: 下限までの半分まで押したら1回だけ追加 ---
                if mode in ("ナンピン", "両方") and not added_dn:
                    trg = entry - side * room * 0.5
                    hit = (px_lo <= trg) if side > 0 else (px_hi >= trg)
                    if hit:
                        addpx = trg + side * half
                        cost_basis = (cost_basis * units + addpx) / (units + 1)
                        units += 1
                        added_dn = True
                        stat[mode][3] += 1
                # --- ピラミッド: 目標までの半分まで順行したら1回だけ追加 ---
                if mode in ("ピラミッド", "両方") and not added_up:
                    trg = entry + side * span * 0.5
                    hit = (px_hi >= trg) if side > 0 else (px_lo <= trg)
                    if hit:
                        addpx = trg + side * half
                        cost_basis = (cost_basis * units + addpx) / (units + 1)
                        units += 1
                        added_up = True
                        stat[mode][3] += 1
            if r is None:
                exitpx = c[i + MAX_HOLD] - side * half
                r = units * side * (exitpx - cost_basis) / PIP
            d = ts[i][:10]
            out[mode][d] += r
            stat[mode][0] += 1
            stat[mode][1] += (r > 0)
            stat[mode][2] += r
    return out, stat


agg = defaultdict(lambda: defaultdict(float))
ST = defaultdict(lambda: [0, 0, 0.0, 0])
for pair in (sys.argv[1:] or PAIRS):
    got = run(pair)
    if not got:
        continue
    o, st = got
    for k, dd in o.items():
        for d, v in dd.items():
            agg[k][d] += v
    for k, v in st.items():
        for z in range(4):
            ST[k][z] += v[z]
    print(f"{pair} 完了", file=sys.stderr)

print(f"\n=== L033 構造に紐づくナンピン/ピラミッド / {NAME} / 11ペア ===")
print("下限=H4直近スイング安値（割ったら全切り）/ 目標=H4直近スイング高値（到達で利確）")
print("ナンピン=下限までの半分で1回追加 / ピラミッド=目標までの半分で1回追加\n")
print(f"{'条件':>9s} {'取引':>7s} {'追加':>6s} {'net/取引':>9s} {'勝率':>5s} | "
      f"{'TR日次':>8s} {'TE日次':>8s} {'TE中央':>8s} {'TE最悪日':>9s} {'合格':>5s}")
print("-" * 92)
rows = []
for mode in MODES:
    if mode not in agg:
        continue
    dd = agg[mode]
    tr = [v for d, v in dd.items() if d < TRAIN_END]
    te = [v for d, v in dd.items() if d >= TRAIN_END]
    if not tr or not te:
        continue
    n_, w_, s_, a_ = ST[mode]
    rows.append((mode, n_, a_, s_ / n_, 100 * w_ / n_, statistics.mean(tr),
                 statistics.mean(te), statistics.median(te), min(te)))
base = rows[0] if rows else None
for mode, n_, a_, pt, wr, mtr, mte, med, worst in rows:
    ok = (mode != "単独" and base and pt > base[3] and worst > 2 * base[8]
          and (mtr > 0) == (mte > 0))
    print(f"{mode:>9s} {n_:7d} {a_:6d} {pt:9.3f} {wr:4.0f}% | {mtr:8.2f} {mte:8.2f} "
          f"{med:8.2f} {worst:9.1f} {'○' if ok else '×':>5s}")

print("\n=== 棄却条件の判定 ===")
if base:
    for mode, n_, a_, pt, wr, mtr, mte, med, worst in rows[1:]:
        print(f"  {mode:>6s}: 条件1(1取引あたり {pt:+.3f} vs 単独 {base[3]:+.3f}) "
              f"{'通過' if pt > base[3] else '該当→棄却'} / "
              f"条件2(最悪日 {worst:.0f} vs 単独の2倍 {2*base[8]:.0f}) "
              f"{'通過' if worst > 2*base[8] else '該当→棄却'} / "
              f"条件3(符号) {'通過' if (mtr>0)==(mte>0) else '該当→棄却'}")
    print("\nL032 との差: L032 は下限を見ず、割っても追加し続けた。L033 は見る・割ったら切る。")
    print("この差だけで符号が変わったかどうかが、ここで分かる。")
