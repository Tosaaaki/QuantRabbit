"""L026 — 真に未見の 2020-2023 で「静かな側」ゲートを検定する。

L024 も L025 も **2024年以降しか使っていない**。M1コーパスは2020年からあるので、
**2020-2023 の4年間はこの仮説に一度も触れられていない**。
仮説（EFF20 が低い＝静かな局面から入ると `mom_break@2880` が払う）は
2025年のデータを見て立てたものなので、2020-2023 は値付け・期間・相場のすべてで独立。

閾値は **2024年（L024/L025 と同じTRAIN）の分位をそのまま使う**。
2020-2023 を見て決め直さない。

棄却条件（事前宣言 L026）:
  1. 2020-2023 プールで改善が正でない
  2. 年別（2020/2021/2022/2023）で符号が過半数揃わない
     ——L025 で「符号だけ見て magnitude を見なかった」反省から年別も出す
  3. 乱数帰無分布の95パーセンタイルを超えない

**ここを通れば、値付け・期間・相場のすべてで独立な確認になる。**
"""
import os, random, statistics, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from trend_gate import (daily_bars, trend_measures, strategy_days,
                        PAIRS, NAME, H)          # noqa: E402

QUANTILES = [0.33, 0.50, 0.67]
DRAWS = 400
random.seed(20260806)
UNSEEN_YEARS = ("2020", "2021", "2022", "2023")

perf = {}; meas = {}
for pair in (sys.argv[1:] or PAIRS):
    print(f"{pair} ...", file=sys.stderr)
    bd, tm = strategy_days(pair)
    if bd:
        perf[pair] = bd; meas[pair] = tm

# 閾値は 2024年だけで決める（L024/L025 と同じ TRAIN）
print(f"\n=== L026 未見期間 {UNSEEN_YEARS[0]}-{UNSEEN_YEARS[-1]} / {NAME}@{H} / "
      f"{len(perf)}ペア ===")
print("閾値は 2024年の分位（L024/L025 と同一）。未見期間を見て決め直さない\n")
print(f"{'指標':11s} {'分位':>5s} {'閾値':>8s} | {'無条件':>8s} {'静か':>8s} {'改善':>8s} "
      f"{'採用日':>7s} {'乱数超え':>7s} | {'年別の符号':>22s}")
print("-" * 108)

results = []
GATE = []
for key in ("EFF20", "ADX14", "RET20_ATR"):
    cal = [meas[p][d][key] for p in perf for d in perf[p]
           if d[:4] == "2024" and d in meas[p] and meas[p][d][key] is not None]
    if len(cal) < 100:
        continue
    cal.sort()
    for q in QUANTILES:
        thr = cal[int(len(cal) * q)]
        allv = []; qv = []
        by_year = defaultdict(lambda: [[], []])
        for p in perf:
            for d, v in perf[p].items():
                if d[:4] not in UNSEEN_YEARS or d not in meas[p]:
                    continue
                m = meas[p][d][key]
                if m is None:
                    continue
                allv.append(v); by_year[d[:4]][0].append(v)
                if m < thr:
                    qv.append(v); by_year[d[:4]][1].append(v)
        if len(qv) < 50:
            continue
        mu = statistics.mean(allv); mq = statistics.mean(qv)
        null = [statistics.mean(random.sample(allv, len(qv))) - mu for _ in range(DRAWS)]
        pct = sum(1 for x in null if x < mq - mu) / DRAWS
        signs = []
        for y in UNSEEN_YEARS:
            a, g = by_year[y]
            if len(g) < 10 or not a:
                signs.append("·"); continue
            signs.append("+" if statistics.mean(g) > statistics.mean(a) else "-")
        agree = sum(1 for s in signs if s == "+")
        valid = sum(1 for s in signs if s in "+-")
        # 採用可否は「改善したか」ではなく「それ自体がプラスと言えるか」で決まる。
        # 契約 QR_AI_STRATEGY_ALLOCATION_V1 と同じ片側95%下限を、ゲート後の系列に当てる。
        se = statistics.pstdev(qv) / (len(qv) ** 0.5) if len(qv) > 1 else 0.0
        lb = mq - 1.645 * se if se > 0 else mq
        med = statistics.median(qv)
        win = 100 * sum(1 for x in qv if x > 0) / len(qv)
        GATE.append((key, q, mq, lb, med, win, len(qv)))
        results.append((mq - mu, key, q, pct, agree, valid, signs))
        detail = " ".join(f"{y[2:]}:{s}" for y, s in zip(UNSEEN_YEARS, signs))
        print(f"{key:11s} {q:5.2f} {thr:8.3f} | {mu:8.2f} {mq:8.2f} {mq-mu:+8.2f} "
              f"{len(qv):7d} {pct:7.0%} | {detail:>22s}")

print("\n=== 棄却条件の判定 ===")
if not results:
    print("評価可能な行なし")
else:
    passed = []
    for d, key, q, pct, agree, valid, signs in sorted(results, reverse=True):
        c1 = d > 0
        c2 = valid > 0 and agree > valid / 2
        c3 = pct >= 0.95
        if c1 and c2 and c3:
            passed.append((d, key, q))
    d, key, q, pct, agree, valid, signs = max(results)
    print(f"最良: {key} 下位{100*q:.0f}%  改善 {d:+.2f} pips/日  乱数超え {pct:.0%}  "
          f"年別 +{agree}/{valid}")
    print(f"  条件1（改善が正でない）: {'該当→棄却' if d <= 0 else '通過'}")
    print(f"  条件2（年別で符号が過半数揃わない）: "
          f"{'該当→棄却' if not (valid and agree > valid/2) else '通過'}")
    print(f"  条件3（乱数95%を超えない）: {'該当→棄却' if pct < 0.95 else '通過'}")
    print(f"\n3条件すべて通過: {len(passed)}/{len(results)}")
    if passed:
        print("→ 値付け・期間・相場のすべてで独立な確認を通過した。")
        print("  それでも『発見』と呼ぶには、**前向きの未経過期間**での再確認が要る。")
    else:
        print("→ **未見期間で棄却。** 2025年で見えていたものは、その期間の性質だった。")

print("\n=== ゲート後の系列そのものは合格するか（契約と同じ片側95%下限）===")
print(f"{'指標':11s} {'分位':>5s} {'日数':>6s} {'日次平均':>9s} {'中央値':>9s} "
      f"{'片側95%下限':>12s} {'勝ち日':>6s} {'合格':>5s}")
print("-" * 70)
for key, q, mq, lb, med, win, n in GATE:
    print(f"{key:11s} {q:5.2f} {n:6d} {mq:9.2f} {med:9.2f} {lb:12.2f} {win:5.0f}% "
          f"{'○' if lb > 0 else '×':>5s}")
print("\n注: 日数は『ペア×日』の延べ。同日に複数ペアが建つので独立ではない。")
print("    契約の基準は1日1行なので、これは**楽観側の近似**である。")
