"""L029 — L026 の「楽観側の近似」を、事前宣言どおり 1日1行 に直す。

L026 は EFF20 下位67%ゲート後の系列に片側95%下限 +0.20 を出して合格としたが、
その日数は **「ペア×日」の延べ** だった。同じ日に複数ペアが建つので独立でない。
`docs/RESEARCH_LOG.md` L026 の「次に必要なこと 1」がこれ:

    1. 1日1行に集約した正しい下限（ペア×日の延べを使わない）

**この実行は探索ではない。** 設定は EFF20 下位67% に固定する。分位も指標も増やさない
（L020 の罠）。閾値も L024/L025/L026 と同じ 2024年の分位をそのまま使う。
変えるのは集計の単位だけ。

出す数字:
  A. ポートフォリオ日次 = その日にゲートを通った全ペアの pips 合計（1日1行）
  B. 参考: その日の採用ペア数の平均（Aの規模を解釈するため）
棄却条件（L026 と同じ契約基準）: 片側95%下限が正でなければ不合格。
"""
import json
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from trend_gate import PAIRS, NAME, H, strategy_days  # noqa: E402

KEY = "EFF20"          # 固定。動かさない
Q = 0.67               # 固定。動かさない
UNSEEN = ("2020", "2021", "2022", "2023")
Z = 1.645              # 片側95%
CACHE = os.path.join(HERE, "quiet_gate_cache.json")


def build():
    """全ペアの日次損益と因果的トレンド指標。重いので JSON にキャッシュする。"""
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as fh:
            blob = json.load(fh)
        return blob["perf"], blob["meas"]
    perf = {}
    meas = {}
    for pair in PAIRS:
        print(f"{pair} ...", file=sys.stderr)
        byday, tm = strategy_days(pair)
        if not byday:
            continue
        perf[pair] = byday
        meas[pair] = tm
    with open(CACHE, "w", encoding="utf-8") as fh:
        json.dump({"perf": perf, "meas": meas}, fh)
    return perf, meas


def bound(series):
    """片側95%下限。契約 QR_AI_STRATEGY_ALLOCATION_V1 と同じ形。"""
    if len(series) < 2:
        return float("nan")
    se = statistics.pstdev(series) / (len(series) ** 0.5)
    return statistics.mean(series) - Z * se


def report(label, series):
    if not series:
        print(f"{label:34s}  (行なし)")
        return
    mu = statistics.mean(series)
    med = statistics.median(series)
    lb = bound(series)
    win = 100 * sum(1 for x in series if x > 0) / len(series)
    print(f"{label:34s} {len(series):6d} {mu:10.2f} {med:10.2f} {lb:12.2f} {win:6.0f}% "
          f"{'○' if lb > 0 else '×':>5s}")


def main():
    perf, meas = build()
    # 閾値は 2024年だけで決める（L024/L025/L026 と同一。未見期間を見て決め直さない）
    cal = [meas[p][d][KEY] for p in perf for d in perf[p]
           if d[:4] == "2024" and d in meas[p] and meas[p][d][KEY] is not None]
    cal.sort()
    thr = cal[int(len(cal) * Q)]
    print(f"=== L029 1日1行に直した下限 / {NAME}@{H} / {KEY} 下位{100*Q:.0f}% / "
          f"{len(perf)}ペア ===")
    print(f"閾値 {thr:.4f}（2024年の分位。L024-L026 と同一。再探索なし）")
    print(f"未見期間 {UNSEEN[0]}-{UNSEEN[-1]}\n")

    pair_day = []                       # L026 と同じ「ペア×日」の延べ
    per_day = defaultdict(float)        # 1日1行（ポートフォリオ合計）
    per_day_n = defaultdict(int)
    for p in perf:
        for d, v in perf[p].items():
            if d[:4] not in UNSEEN or d not in meas[p]:
                continue
            m = meas[p][d][KEY]
            if m is None or m >= thr:
                continue
            pair_day.append(v)
            per_day[d] += v
            per_day_n[d] += 1

    days = sorted(per_day)
    daily_total = [per_day[d] for d in days]

    print(f"{'集計単位':34s} {'n':>6s} {'平均':>10s} {'中央値':>10s} "
          f"{'片側95%下限':>12s} {'勝ち率':>7s} {'合格':>5s}")
    print("-" * 92)
    report("A. ペア×日の延べ（L026 の集計）", pair_day)
    report("B. 1日1行・ポートフォリオ合計", daily_total)

    if per_day_n:
        avg_pairs = statistics.mean(per_day_n.values())
        print(f"\n1日あたり採用ペア数の平均: {avg_pairs:.2f}")
        print(f"B の平均 {statistics.mean(daily_total):.2f} = "
              f"A の平均 {statistics.mean(pair_day):.2f} × {avg_pairs:.2f} と整合するはず")

    lb_a = bound(pair_day)
    lb_b = bound(daily_total)
    print(f"\n=== 判定（契約と同じ片側95%下限）===")
    print(f"  L026 が報告した下限（A）: {lb_a:+.2f}")
    print(f"  正しい下限（B・1日1行） : {lb_b:+.2f}")
    if lb_b > 0:
        print("  → 集計を直しても正。L026 の生存例は残る。次は前向き未経過期間。")
    else:
        print("  → **集計を直すと下限が正でない。L026 の合格は延べ日数による楽観だった。**")
        print("     契約基準では不合格。設定を探し直すのは L020 の罠なのでしない。")


if __name__ == "__main__":
    main()
