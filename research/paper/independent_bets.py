"""L032 — 無相関な機会は何本作れるか（軸①）。FX と crypto を同一手法で比較する。

L031 の結論: 分散は共通ファクター由来ではなく固有。したがって **独立なベット k を
増やせば sd は 1/√k で落ちる**。必要n は (1.645·sd/μ)² なので k を増やすと 1/k で縮む。
現状 FX は 1日1.9本。k=20 なら必要n は 888日 → 約85日（1四半期）になる。

問い: **k を供給できる商品はあるか。**
FX メジャーは通貨レッグを共有するので k は作れない（L031 実測 Σvar/var_port = 0.87、
2020-2023 の全1,686レッグに USD）。crypto はどうか。

測るもの（両市場で同一）:
  1. 日次対数リターンの相関行列 → **実効独立本数 N_eff = (Σλ)² / Σλ²**
     （全独立なら N、完全相関なら 1。相関行列の固有値の参加率）
  2. **値動き / 往復コスト比**。L031 の feasibility gate と同じ考え方で、
     コストは絶対値でなく値幅との比でしか意味を持たない

crypto のコストは手数料率（maker/taker）で入れる。既定は bitbank 公表値の
maker -0.02% / taker 0.12% を仮定値として置く（実測ではない。要検証）。
FX は実約定スプレッド（L017 実測、trend_gate.REAL_SPREAD）。
"""
import json
import math
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from trend_gate import REAL_SPREAD  # noqa: E402

CLOSE_CACHE = os.path.join(HERE, "daily_close_cache.json")
CRYPTO_CACHE = os.path.join(HERE, "bitbank_daily_cache.json")
YEARS = ("2025", "2026")

# bitbank 公表手数料の仮定値。実測ではないので結論に効く場合は必ず確認すること。
TAKER_RATE = 0.0012
MAKER_RATE = -0.0002


def fetch_bitbank(pairs, years=YEARS):
    if os.path.exists(CRYPTO_CACHE):
        return json.load(open(CRYPTO_CACHE, encoding="utf-8"))
    out = {}
    for pair in pairs:
        rows = {}
        for y in years:
            url = f"https://public.bitbank.cc/{pair}/candlestick/1day/{y}"
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    d = json.loads(r.read())
            except Exception as exc:
                print(f"  {pair} {y}: {type(exc).__name__}", file=sys.stderr)
                continue
            if d.get("success") != 1:
                continue
            for c in d["data"]["candlestick"]:
                for o, h, l, cl, v, ms in c["ohlcv"]:
                    day = datetime.fromtimestamp(int(ms) / 1000, timezone.utc).strftime("%Y-%m-%d")
                    rows[day] = {"o": float(o), "h": float(h), "l": float(l),
                                 "c": float(cl), "v": float(v)}
            time.sleep(0.12)
        if len(rows) > 200:
            out[pair] = rows
            print(f"  {pair}: {len(rows)} days", file=sys.stderr)
    json.dump(out, open(CRYPTO_CACHE, "w", encoding="utf-8"), separators=(",", ":"))
    return out


def returns_matrix(series, days):
    """days 順に揃えた日次対数リターン行列（行=日、列=銘柄）。欠損日は除外。"""
    names = sorted(series)
    rows = []
    kept = []
    for prev, cur in zip(days[:-1], days[1:]):
        r = []
        ok = True
        for n in names:
            a, b = series[n].get(prev), series[n].get(cur)
            if not a or not b or a <= 0 or b <= 0:
                ok = False
                break
            r.append(math.log(b / a))
        if ok:
            rows.append(r)
            kept.append(cur)
    return names, np.array(rows), kept


def effective_bets(R):
    """相関行列の固有値の参加率。等ウェイトで実際に何本のベットになっているか。"""
    if R.shape[0] < 10 or R.shape[1] < 2:
        return float("nan"), None
    C = np.corrcoef(R, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    lam = np.linalg.eigvalsh(C)
    lam = np.clip(lam, 0, None)
    return (lam.sum() ** 2) / (lam ** 2).sum(), C


def report(label, names, R, cost_bps):
    n_eff, C = effective_bets(R)
    k = R.shape[1]
    iu = np.triu_indices(k, 1)
    mean_rho = float(C[iu].mean()) if C is not None else float("nan")
    # 等ウェイト・ポートフォリオの分散削減率
    w = np.ones(k) / k
    port_sd = float(np.sqrt(w @ np.cov(R, rowvar=False) @ w))
    avg_sd = float(np.mean(R.std(axis=0)))
    print(f"\n### {label}  ({k} 銘柄 / {R.shape[0]} 日)")
    print(f"  平均ペアワイズ相関      : {mean_rho:+.3f}")
    print(f"  実効独立本数 N_eff      : {n_eff:.2f}  （銘柄数 {k} のうち）")
    print(f"  分散削減 sd_avg -> sd_pf: {avg_sd*100:.2f}% -> {port_sd*100:.2f}%  "
          f"(比 {port_sd/avg_sd:.2f}、完全独立なら {1/math.sqrt(k):.2f})")
    print(f"  {'銘柄':12s} {'日次|移動|中央値':>16s} {'往復コスト':>12s} {'移動/コスト':>12s}")
    med = np.median(np.abs(R), axis=0)
    for i, nm in enumerate(names):
        c = cost_bps[nm]
        print(f"  {nm:12s} {100*med[i]:15.2f}% {100*c:11.3f}% "
              f"{(med[i]/c if c > 0 else float('inf')):12.1f}")
    return n_eff, mean_rho


def main():
    print("=== L032 無相関な機会は何本作れるか / FX vs crypto（同一手法）===")
    print(f"実効独立本数 N_eff = (Σλ)²/Σλ²  ・  必要n は k に対して 1/k で縮む\n")

    # --- FX ---
    fx = json.load(open(CLOSE_CACHE, encoding="utf-8"))
    fx = {p: {d: v for d, v in s.items() if d[:4] in YEARS} for p, s in fx.items()}
    fx = {p: s for p, s in fx.items() if len(s) > 200}
    days = sorted(set.intersection(*[set(s) for s in fx.values()]))
    names, R, _ = returns_matrix(fx, days)
    fx_cost = {}
    for p in names:
        pip = 0.01 if p.endswith("JPY") else 0.0001
        px = fx[p][days[-1]]
        fx_cost[p] = REAL_SPREAD.get(p, 2.0) * pip / px      # 片道スプレッド（比率）
    n_fx, rho_fx = report("FX メジャー（実約定スプレッド・往復1回分）", names, R, fx_cost)

    # --- crypto ---
    tickers = json.loads(urllib.request.urlopen(
        "https://public.bitbank.cc/tickers", timeout=30).read())
    liquid = [t["pair"] for t in sorted(tickers["data"],
              key=lambda t: -float(t["vol"]) * float(t["last"]))
              if t["pair"].endswith("_jpy")][:16]
    print(f"\nbitbank 出来高上位16: {' '.join(liquid)}", file=sys.stderr)
    cr = fetch_bitbank(liquid)
    cr = {p: {d: v["c"] for d, v in s.items() if d[:4] in YEARS} for p, s in cr.items()}
    cr = {p: s for p, s in cr.items() if len(s) > 200}
    if not cr:
        print("crypto データ取得できず")
        return 1
    cdays = sorted(set.intersection(*[set(s) for s in cr.values()]))
    cnames, CR, _ = returns_matrix(cr, cdays)
    taker = {p: TAKER_RATE * 2 for p in cnames}
    n_cr, rho_cr = report(f"bitbank JPY（taker 往復 {TAKER_RATE*2*100:.2f}%・仮定値）",
                          cnames, CR, taker)

    print("\n=== 判定：k は増えるか ===")
    print(f"  FX     : {len(names):2d} 銘柄 → 実効 {n_fx:.2f} 本（平均相関 {rho_fx:+.3f}）")
    print(f"  crypto : {len(cnames):2d} 銘柄 → 実効 {n_cr:.2f} 本（平均相関 {rho_cr:+.3f}）")
    base = 888
    print(f"\n  現状 FX の実運用は 1日1.9本 → 必要n {base}日")
    for label, ne in (("FX 全銘柄同時", n_fx), ("crypto 全銘柄同時", n_cr),
                      ("FX+crypto 合算(相関0と仮定した上限)", n_fx + n_cr)):
        scaled = base * 1.9 / ne if ne > 0 else float("inf")
        print(f"    {label:34s} k={ne:5.2f} → 必要n {scaled:7.0f}日 ({scaled/252:.1f}年)")
    print("\n  注: 必要n の換算は sd∝1/√k のみを仮定し、μ が k を増やしても保たれる前提。")
    print("      μ の保存は別途検定が要る（銘柄を増やせば平均エッジは薄まりうる）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
