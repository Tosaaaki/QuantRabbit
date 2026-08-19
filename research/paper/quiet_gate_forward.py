"""L030 — 前向き未経過期間での確認。`docs/RESEARCH_LOG.md` L026「次に必要なこと 2」。

既存 M1 コーパスは 2026-07-09 で終わっている。その先を OANDA から取り
（`fetch_forward_m1.py`）、コーパスに **連結して** 読む。連結する理由はウォームアップ:
EFF20 は日足20本、WARM は M1 1500本を要求するので、前向き分だけでは指標が立たない。
**評価するのは連結後の 2026-07-10 以降の日付だけ。** それ以前は指標を作るためだけに使う。

窓を2つに分けて別々に報告する:
  W1 2026-07-10 → 2026-08-05  コーパス外だが、仮説確定(2026-08-06)時点で **経過済み**
  W2 2026-08-06 → 現在        仮説確定時点で **未経過**。これだけが本当の前向き検定

設定は固定: EFF20 下位67%、閾値は 2024年の分位。**前向き期間を見て何も決め直さない。**
"""
import glob
import gzip
import json
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import M1, features, roll_fwd, WARM  # noqa: E402
from strategies import REGISTRY  # noqa: E402
from trend_gate import (PAIRS, NAME, H, DIS, REAL_SPREAD,  # noqa: E402
                        daily_bars, trend_measures)

FORWARD = os.path.join(HERE, "forward_m1")
KEY, Q = "EFF20", 0.67
LOCK = "2026-08-06"          # 仮説確定日（RESEARCH_LOG L026 の記載日）
W1_START = "2026-07-10"      # コーパスの終端の翌日
Z = 1.645


def load_joined(pair):
    """コーパス + 前向き分を時系列連結。重複時刻は先勝ちで捨てる。"""
    ts = []; h = []; l = []; c = []
    seen = set()

    def push(r):
        t = r["time"]
        if t in seen:
            return
        seen.add(t)
        b, a = r["bid"], r["ask"]
        ts.append(t)
        h.append((float(b["h"]) + float(a["h"])) / 2)
        l.append((float(b["l"]) + float(a["l"])) / 2)
        c.append((float(b["c"]) + float(a["c"])) / 2)

    for sh in sorted(glob.glob(f"{M1}/*/{pair}/{pair}_M1_BA_*.jsonl.gz")):
        with gzip.open(sh, "rt") as fh:
            for line in fh:
                if line.strip():
                    push(json.loads(line))
    fwd = os.path.join(FORWARD, f"{pair}_M1_BA_forward.jsonl.gz")
    if os.path.exists(fwd):
        with gzip.open(fwd, "rt") as fh:
            for line in fh:
                if line.strip():
                    push(json.loads(line))
    order = sorted(range(len(ts)), key=lambda i: ts[i])
    return ([ts[i] for i in order], [h[i] for i in order],
            [l[i] for i in order], [c[i] for i in order])


def strategy_days_joined(pair):
    PIP = 0.01 if pair.endswith("JPY") else 0.0001
    ts, h, l, c = load_joined(pair)
    n = len(c)
    if n < 200000:
        return None, None
    half = REAL_SPREAD.get(pair, 2.0) / 2 * PIP
    f = features(ts, h, l, c)
    fmin = roll_fwd(l, H, False)
    fmax = roll_fwd(h, H, True)
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


def lb(series):
    if len(series) < 2:
        return float("nan")
    return statistics.mean(series) - Z * statistics.pstdev(series) / (len(series) ** 0.5)


def main():
    perf = {}
    meas = {}
    for pair in PAIRS:
        print(f"{pair} ...", file=sys.stderr)
        bd, tm = strategy_days_joined(pair)
        if bd:
            perf[pair] = bd
            meas[pair] = tm

    cal = sorted(meas[p][d][KEY] for p in perf for d in perf[p]
                 if d[:4] == "2024" and d in meas[p] and meas[p][d][KEY] is not None)
    thr = cal[int(len(cal) * Q)]
    print(f"\n=== L030 前向き検定 / {NAME}@{H} / {KEY} 下位{100*Q:.0f}% / {len(perf)}ペア ===")
    print(f"閾値 {thr:.4f}（2024年の分位。前向き期間を見て決め直していない）\n")

    windows = {
        "W1 07-10→08-05 (コーパス外・経過済)": lambda d: W1_START <= d < LOCK,
        "W2 08-06→現在 (未経過・本当の前向き)": lambda d: d >= LOCK,
        "W1+W2 合算": lambda d: d >= W1_START,
    }
    print(f"{'窓':40s} {'日数':>5s} {'平均/日':>9s} {'中央値':>9s} {'下限':>9s} {'勝ち日':>7s}")
    print("-" * 86)
    for label, pred in windows.items():
        per_day = defaultdict(float)
        gated = 0
        for p in perf:
            for d, v in perf[p].items():
                if not pred(d) or d not in meas[p]:
                    continue
                m = meas[p][d][KEY]
                if m is None or m >= thr:
                    continue
                per_day[d] += v
                gated += 1
        days = sorted(per_day)
        s = [per_day[d] for d in days]
        if not s:
            print(f"{label:40s}   (ゲート通過なし)")
            continue
        print(f"{label:40s} {len(s):5d} {statistics.mean(s):9.2f} "
              f"{statistics.median(s):9.2f} {lb(s):9.2f} "
              f"{100*sum(1 for x in s if x>0)/len(s):6.0f}%")
        if label.startswith("W2"):
            print(f"    採用ペア×日 {gated}, 日別: " +
                  " ".join(f"{d[5:]}:{per_day[d]:+.0f}" for d in days))


if __name__ == "__main__":
    main()
