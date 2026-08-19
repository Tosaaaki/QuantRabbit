"""L031 — 検出力は設計変数か。共通ファクターを抜いた残差にアルファは残るか。

L029 が突きつけた壁は「エッジの大きさ」ではなく **検定に要る標本数** だった:

    n必要 = (1.645 * sd / mu)^2 = (1.645 * 100.7 / 5.56)^2 ~ 888日 ~ 3.5年

`mu` は探索し尽くした。しかし **`sd` は自然定数ではなく設計変数** である。
sd=100.7 が大きい理由は、同日に建つ 1.9 ペアが共通ファクター（USD強弱・JPY強弱・
リスクオンオフ）をまるごと被っているから。共通分散を落とせば、**新しいエッジを
発見しなくても** 必要nが縮む。

過去の「ヘッジは利益も消した」は **ペア対ペア** の検定であって、これとは別物。
別ペアを引けばそのペアのアルファも引くが、**ファクターを引いてもアルファは引かれない**
——それが成り立つかどうかが、まさにここで測ること。

判定は両方向に決定的:
  * 残差の平均がゼロに落ちる → エッジの正体はファクター露出だった。**閉鎖確定**
  * 平均が残って sd が落ちる → 必要nが縮む。**閉じたセルの多くは「測れていなかった」に戻る**

方法:
  1. 全11ペアの日次対数リターンから、通貨強弱ファクターを最小二乗で復元
     （r_XY ~ s_X - s_Y、sum(s)=0 で一意化）。戦略の情報は一切使わない
  2. ゲート通過日のポートフォリオ日次損益 P_d を、その日のファクターに時系列回帰
     P_d = a + Σ b_c f_c,d + e_d
  3. ヘッジ後の系列は (a + e_d)。その mu=a, sd=sd(e) で必要nを引き直す

注意（結果の読み方に必須）:
  * 同じ889日での回帰なので **アルファ推定は楽観側**。年別でも出す
  * ヘッジ自体にコストがかかる。ここでは測っていない（上限としての数字）
"""
import json
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

GATE_CACHE = os.path.join(HERE, "quiet_gate_cache.json")
CLOSE_CACHE = os.path.join(HERE, "daily_close_cache.json")
KEY, Q = "EFF20", 0.67
UNSEEN = tuple((os.environ.get("QR_WIN") or "2020,2021,2022,2023").split(","))
Z = 1.645


def n_required(mu, sd):
    if mu <= 0 or sd <= 0:
        return float("inf")
    return (Z * sd / mu) ** 2


def solve(A, b):
    """素の最小二乗（正規方程式＋ガウス消去）。外部依存を持ち込まない。"""
    k = len(A[0])
    M = [[sum(A[r][i] * A[r][j] for r in range(len(A))) for j in range(k)]
         + [sum(A[r][i] * b[r] for r in range(len(A)))] for i in range(k)]
    for i in range(k):
        p = max(range(i, k), key=lambda r: abs(M[r][i]))
        if abs(M[p][i]) < 1e-12:
            return None
        M[i], M[p] = M[p], M[i]
        pv = M[i][i]
        M[i] = [x / pv for x in M[i]]
        for r in range(k):
            if r != i and M[r][i]:
                f = M[r][i]
                M[r] = [x - f * y for x, y in zip(M[r], M[i])]
    return [M[i][k] for i in range(k)]


def currency_factors(closes, pairs):
    """日次の通貨強弱。戦略とは無関係に、価格だけから作る。"""
    import math
    ccys = sorted({c for p in pairs for c in p.split("_")})
    idx = {c: i for i, c in enumerate(ccys)}
    rets = {}
    for p in pairs:
        s = closes.get(p, {})
        ds = sorted(s)
        for prev, cur in zip(ds[:-1], ds[1:]):
            if s[prev] > 0 and s[cur] > 0:
                rets.setdefault(cur, {})[p] = math.log(s[cur] / s[prev])
    factors = {}
    for d, rr in rets.items():
        # その日に実在する通貨だけで組む。コーパスの被覆がペアごとに違うので
        # 固定のペア数閾値は使えない（2020-2023 は5ペアしか存在しない）。
        present = sorted({c for p in rr for c in p.split("_")})
        if len(rr) < len(present) - 1:
            continue
        pidx = {c: i for i, c in enumerate(present)}
        A = []; b = []
        for p, r in rr.items():
            x, y = p.split("_")
            row = [0.0] * len(present); row[pidx[x]] = 1.0; row[pidx[y]] = -1.0
            A.append(row); b.append(r)
        A.append([1.0] * len(present)); b.append(0.0)
        sol = solve(A, b)
        if sol:
            factors[d] = {c: 0.0 for c in ccys}
            factors[d].update(dict(zip(present, sol)))
    return ccys, factors


def main():
    for path in (GATE_CACHE, CLOSE_CACHE):
        if not os.path.exists(path):
            print(f"missing cache: {path}", file=sys.stderr)
            return 1
    blob = json.load(open(GATE_CACHE, encoding="utf-8"))
    perf, meas = blob["perf"], blob["meas"]
    closes = json.load(open(CLOSE_CACHE, encoding="utf-8"))
    pairs = sorted(perf)

    cal = sorted(meas[p][d][KEY] for p in perf for d in perf[p]
                 if d[:4] == "2024" and d in meas[p] and meas[p][d][KEY] is not None)
    thr = cal[int(len(cal) * Q)]

    per_day = defaultdict(float)
    per_pair_day = defaultdict(dict)
    for p in perf:
        for d, v in perf[p].items():
            if d[:4] not in UNSEEN or d not in meas[p]:
                continue
            m = meas[p][d][KEY]
            if m is None or m >= thr:
                continue
            per_day[d] += v
            per_pair_day[d][p] = v

    ccys, factors = currency_factors(closes, pairs)
    days = sorted(d for d in per_day if d in factors)
    P = [per_day[d] for d in days]
    print(f"=== L031 検出力は設計変数か / {KEY} 下位{100*Q:.0f}% / {len(pairs)}ペア ===")
    print(f"通貨 {','.join(ccys)} / ゲート通過日 {len(days)}（ファクター有る日のみ）\n")

    mu0, sd0 = statistics.mean(P), statistics.pstdev(P)
    print(f"{'系列':30s} {'n':>5s} {'平均':>9s} {'sd':>9s} {'片側95%下限':>12s} {'必要n':>9s} {'年数':>7s}")
    print("-" * 88)

    def row(label, mu, sd, n):
        lb = mu - Z * sd / (n ** 0.5)
        need = n_required(mu, sd)
        yrs = need / 252 if need != float("inf") else float("inf")
        print(f"{label:30s} {n:5d} {mu:9.3f} {sd:9.2f} {lb:12.3f} "
              f"{(f'{need:,.0f}' if need != float('inf') else 'inf'):>9s} "
              f"{(f'{yrs:.1f}' if yrs != float('inf') else 'inf'):>7s}")

    row("素のポートフォリオ日次", mu0, sd0, len(P))

    # ファクター回帰。窓の中で実際に変動する通貨だけを使う
    # （2020-2023 には CAD/CHF が存在せず、その列は全ゼロで設計行列が特異になる）
    live = [c for c in ccys
            if len({round(factors[d][c], 12) for d in days}) > 1]
    use = live[:-1]                                   # 1つを基準に落とす
    print(f"回帰に使う通貨ファクター: {','.join(use)}  (基準 {live[-1]})")
    A = [[1.0] + [factors[d][c] for c in use] for d in days]
    beta = solve(A, P)
    if not beta:
        print("回帰が解けない")
        return 1
    fitted = [sum(a * b for a, b in zip(A[i], beta)) for i in range(len(days))]
    resid = [P[i] - fitted[i] for i in range(len(days))]
    alpha = beta[0]
    sd_e = statistics.pstdev(resid)
    row("ファクター中立後 (a + e)", alpha, sd_e, len(resid))

    r2 = 1 - (statistics.pvariance(resid) / statistics.pvariance(P)) if statistics.pvariance(P) else 0
    print(f"\n共通ファクターが説明した分散: {100*r2:.1f}%   sd {sd0:.1f} -> {sd_e:.1f}")
    print(f"必要n {n_required(mu0, sd0):,.0f}日 -> {n_required(alpha, sd_e):,.0f}日")

    print("\n=== 年別（アルファが期間で安定か。回帰は全期間の beta を流用）===")
    print(f"{'年':6s} {'n':>5s} {'素の平均':>10s} {'残差+a の平均':>14s}")
    for y in UNSEEN:
        ix = [i for i, d in enumerate(days) if d[:4] == y]
        if len(ix) < 10:
            continue
        print(f"{y:6s} {len(ix):5d} {statistics.mean([P[i] for i in ix]):10.2f} "
              f"{statistics.mean([resid[i] + alpha for i in ix]):14.2f}")

    print("\n=== 実効的な独立ベット数（分散から見た多様化の余地）===")
    allp = sorted({p for d in per_pair_day for p in per_pair_day[d]})
    series = {p: [per_pair_day[d].get(p, 0.0) for d in days] for p in allp}
    var_sum = sum(statistics.pvariance(series[p]) for p in allp if len(set(series[p])) > 1)
    var_port = statistics.pvariance(P)
    if var_port > 0:
        print(f"  Σ個別分散 / ポートフォリオ分散 = {var_sum/var_port:.2f}")
        print("  （1.0 なら完全相関＝多様化ゼロ。ペア数に近いほど独立）")
    print(f"  1日あたり採用ペア数 {statistics.mean(len(per_pair_day[d]) for d in days):.2f}")

    print("\n=== 判定 ===")
    if alpha <= 0:
        print("  残差アルファが正でない → **エッジの正体は共通ファクターへの露出だった。閉鎖確定。**")
    elif n_required(alpha, sd_e) < n_required(mu0, sd0) / 2:
        print(f"  アルファが残り、必要nが {n_required(mu0,sd0)/n_required(alpha,sd_e):.1f}倍 縮んだ。")
        print("  → 検出力は設計変数だった。ただし (a) 同一標本の回帰なので楽観 "
              "(b) ヘッジ実行コスト未計上。次はこの2つを潰す。")
    else:
        print("  アルファは残るが必要nが実質縮まない → この方向では壁は動かない。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
