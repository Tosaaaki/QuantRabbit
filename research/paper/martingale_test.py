"""L032 — 機械的ナンピンの検定。

L031 の台帳実測: 人の7件のナンピンが +27,025円（利益の過半）を出す一方、
`pullback_s5` の -37,661円の穴は98%がナンピン玉で、強制決済14件もそこに集中していた。
全体では中央値 -0.20 対 +1.40、勝率47% 対 59% でナンピンのほうが悪い。

問い: 発動条件を機械化したナンピンは、単独エントリーを上回るか。

設計:
  `mom_break@2880`（唯一の生存候補）に、逆行 X pips で同サイズを **1回だけ** 追加する層。
  X は ATR連動（0.5 / 1.0 / 1.5 × H4 ATR）。
  **無限ナンピンはやらない** —— 強制決済の温床であることを台帳が示している。

測るもの: 1取引あたり純益 / 勝率 / **中央値** / 最悪日 / 必要証拠金の増加。

事前の予想: **棄却されると予想する。** 台帳が示す方向がそうだから。
予想を先に書くのは、外れたときに気づくため。

棄却条件:
  1. 追加ありが追加なしを1取引あたりで上回らない
  2. 上回っても**最悪日が2倍以上悪化**する（NAV20万では耐えられない）
  3. TRAIN/TEST で符号が反転する
"""
import os, statistics, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import load, features, roll_fwd, WARM  # noqa: E402
from strategies import REGISTRY                    # noqa: E402
from stop_geometry import h4_atr                   # noqa: E402

PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD",
         "AUD_JPY", "EUR_JPY", "GBP_JPY", "CAD_JPY", "CHF_JPY", "NZD_JPY"]
H = 2880
DIS = 50.0
NAME = "mom_break"
TRAIN_END = "2025-01-01"
ADD_MULTS = [None, 0.5, 1.0, 1.5]      # None = 追加なし（基準）
REAL_SPREAD = {"USD_JPY": 0.80, "EUR_USD": 0.80, "GBP_USD": 1.30, "AUD_USD": 1.40,
               "NZD_USD": 1.50, "AUD_JPY": 1.60, "EUR_JPY": 1.80, "CAD_JPY": 2.30,
               "NZD_JPY": 2.70, "GBP_JPY": 3.20, "CHF_JPY": 3.50}


def run(pair):
    PIP = 0.01 if pair.endswith("JPY") else 0.0001
    ts, h, l, c, sp = load(pair)
    n = len(c)
    if n < 200000:
        return None
    half = REAL_SPREAD.get(pair, 2.0) / 2 * PIP
    hp = REAL_SPREAD.get(pair, 2.0) / 2
    f = features(ts, h, l, c)
    atr4 = h4_atr(h, l, c, PIP)
    fmin = roll_fwd(l, H, False); fmax = roll_fwd(h, H, True)
    fn = REGISTRY[NAME]
    out = defaultdict(lambda: defaultdict(float))
    stat = defaultdict(lambda: [0, 0, 0.0])       # trades, wins, sum
    last = -10 ** 9
    for i in range(WARM, n - H):
        if i - last < H or f["atr"][i] <= 0 or atr4[i] <= 0:
            continue
        side = fn(f, i)
        if not side:
            continue
        last = i
        px = c[i] + side * half
        # 単独玉の結果
        if side > 0:
            worst_p = (fmin[i] + half - px) / PIP
            base = -DIS if worst_p <= -DIS else (c[i + H] - half - px) / PIP
        else:
            worst_p = (px - (fmax[i] - half)) / PIP
            base = -DIS if worst_p <= -DIS else (px - (c[i + H] + half)) / PIP
        for m in ADD_MULTS:
            key = ("なし" if m is None else f"{m}xATR", )
            if m is None:
                r = base
            else:
                trigger = m * atr4[i]              # pips
                if worst_p > -trigger:
                    r = base                        # 追加は発動しなかった
                else:
                    # 追加は「逆行 trigger pips」の価格で約定したとみなす。
                    # そこから出口までの値幅を、同サイズで加算する。
                    # 追加分もスプレッドを払う。2玉なので平均は (base + add)/2 相当だが、
                    # 実際は建玉が2倍なので合計損益をそのまま使う。
                    add_entry = px - side * trigger * PIP
                    if side > 0:
                        add_worst = (fmin[i] + half - add_entry) / PIP
                        add = (-DIS if add_worst <= -DIS
                               else (c[i + H] - half - add_entry) / PIP)
                    else:
                        add_worst = (add_entry - (fmax[i] - half)) / PIP
                        add = (-DIS if add_worst <= -DIS
                               else (add_entry - (c[i + H] + half)) / PIP)
                    r = base + add - 2 * hp * 0     # スプレッドは既に建値に織り込み済み
            d = ts[i][:10]
            out[key][d] += r
            stat[key][0] += 1
            stat[key][1] += (r > 0)
            stat[key][2] += r
    return out, stat


agg = defaultdict(lambda: defaultdict(float))
ST = defaultdict(lambda: [0, 0, 0.0])
for pair in (sys.argv[1:] or PAIRS):
    got = run(pair)
    if not got:
        continue
    o, st = got
    for k, dd in o.items():
        for d, v in dd.items():
            agg[k][d] += v
    for k, v in st.items():
        ST[k][0] += v[0]; ST[k][1] += v[1]; ST[k][2] += v[2]
    print(f"{pair} 完了", file=sys.stderr)

print(f"\n=== L032 機械的ナンピン / {NAME}@{H} / 11ペア / 追加は1回だけ ===")
print("追加分もスプレッドを払う。建玉は2倍になるので必要証拠金も2倍\n")
print(f"{'発動閾値':>10s} {'取引':>7s} {'net/取引':>9s} {'勝率':>5s} | "
      f"{'TR日次':>8s} {'TE日次':>8s} {'TE中央':>8s} {'TE最悪日':>9s} {'合格':>5s}")
print("-" * 88)
base_worst = None
rows = []
for k in [("なし",)] + [(f"{m}xATR",) for m in ADD_MULTS if m]:
    if k not in agg:
        continue
    dd = agg[k]
    tr = [v for d, v in dd.items() if d < TRAIN_END]
    te = [v for d, v in dd.items() if d >= TRAIN_END]
    if not tr or not te:
        continue
    n_, w_, s_ = ST[k]
    worst = min(te)
    if k[0] == "なし":
        base_worst = worst
        base_pt = s_ / n_
    rows.append((k[0], n_, s_ / n_, 100 * w_ / n_, statistics.mean(tr),
                 statistics.mean(te), statistics.median(te), worst))
for name, n_, pt, wr, mtr, mte, med, worst in rows:
    ok = (name != "なし" and pt > base_pt and worst > 2 * base_worst
          and (mtr > 0) == (mte > 0))
    print(f"{name:>10s} {n_:7d} {pt:9.3f} {wr:4.0f}% | {mtr:8.2f} {mte:8.2f} "
          f"{med:8.2f} {worst:9.1f} {'○' if ok else '×':>5s}")

print("\n=== 棄却条件の判定 ===")
if len(rows) < 2:
    print("比較対象が足りない")
else:
    base = rows[0]
    for name, n_, pt, wr, mtr, mte, med, worst in rows[1:]:
        c1 = pt > base[2]
        c2 = worst > 2 * base[7]
        c3 = (mtr > 0) == (mte > 0)
        print(f"  {name:>8s}: 条件1(1取引あたり) {'通過' if c1 else '該当→棄却'} / "
              f"条件2(最悪日が2倍以上悪化) {'通過' if c2 else '該当→棄却'}"
              f"（{worst:.0f} vs 基準{base[7]:.0f}）/ "
              f"条件3(符号反転) {'通過' if c3 else '該当→棄却'}")
    print(f"\n事前の予想は『棄却される』だった。"
          f"実際: {'予想どおり棄却' if not any(r[2] > base[2] and r[7] > 2*base[7] for r in rows[1:]) else '**予想が外れた**'}")
