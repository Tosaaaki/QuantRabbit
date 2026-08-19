"""Forward paper ledger — accumulate the evidence the allocation gate needs.

engine.py answers "did this ever work". It cannot answer "does it work now", and
`QR_AI_STRATEGY_ALLOCATION_V1` deliberately asks the second question: five FILLED
ACTIVE DAYS, cumulative positive, positive one-sided 95% lower bound over day
means, for a NAMED strategy.

So this appends per-strategy per-pair-day paper results to an append-only ledger,
idempotent on (strategy, pair, day). Run it as the corpus advances and a new
strategy accumulates real forward evidence at zero risk; the same rows then feed
the scorecard that decides whether it may ever be allocated to.

Paper only. Reads the replay corpus, writes one JSONL file, and touches no
broker, no order path, and no live permission.

    python3 ledger.py --since 2026-01-01     # append those days
    python3 ledger.py --since 2026-01-01 USD_JPY EUR_USD
    python3 ledger.py --scorecard            # apply the gate to what is stored
"""
import json, os, statistics, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LEDGER = f"{HERE}/paper_ledger.jsonl"
Z = 1.645
MIN_DAYS = 5


def read_ledger():
    rows = {}
    if os.path.exists(LEDGER):
        with open(LEDGER) as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    rows[(d["strategy"], d["pair"], d["day"])] = d
    return rows


def append(pairs, since):
    from engine import run_pair, HOLD, DIS
    existing = read_ledger()
    added = skipped = 0
    with open(LEDGER, "a") as fh:
        for pair in pairs:
            res, counts, halfp, n = run_pair(pair, since=since)
            print(f"{pair}: {n:,} bars  half={halfp:.2f}p  "
                  f"signal-days={len({d for _, d in res})}", file=sys.stderr)
            for (name, day), pips in sorted(res.items(), key=lambda kv: kv[0][1]):
                key = (name, pair, day)
                if key in existing:
                    skipped += 1
                    continue
                fh.write(json.dumps({
                    "strategy": name, "pair": pair, "day": day,
                    "pips": round(pips, 4),
                    "exec": {"hold_bars": HOLD, "disaster_stop_pips": DIS,
                             "half_spread_pips": round(halfp, 3)},
                    "paper": True, "live_permission": False}) + "\n")
                added += 1
    print(f"追記 {added} 行 / 既存につき飛ばした {skipped} 行 -> {LEDGER}")


def scorecard():
    rows = read_ledger()
    if not rows:
        print(f"台帳が空: {LEDGER}\n先に `python3 ledger.py --since YYYY-MM-DD` を実行すること")
        return
    byname = defaultdict(lambda: defaultdict(float))
    for (s, p, d), r in rows.items():
        byname[s][d] += r["pips"]
    print(f"台帳 {LEDGER}  行数={len(rows)}  戦略={len(byname)}")
    print(f"合格条件: {MIN_DAYS}営業日以上 / 累積プラス / 日次平均の片側95%下限>0\n")
    print(f"{'strategy':24s} {'日数':>5s} {'累積pips':>9s} {'日次平均':>9s} "
          f"{'片側95%下限':>11s} {'勝ち日':>7s} {'提案倍率':>8s}")
    print("-" * 82)
    for s, dd in sorted(byname.items(), key=lambda kv: -sum(kv[1].values())):
        days = list(dd.values())
        m = statistics.mean(days)
        se = statistics.pstdev(days) / (len(days) ** 0.5) if len(days) > 1 else 0.0
        lb = m - Z * se if se > 0 else m
        ok = len(days) >= MIN_DAYS and sum(days) > 0 and lb > 0
        mult = "2.0" if ok else ("1.0" if sum(days) > 0 else "0.0")
        print(f"{s:24s} {len(days):5d} {sum(days):9.1f} {m:9.2f} {lb:11.2f} "
              f"{100*sum(1 for x in days if x>0)/len(days):6.0f}% {mult:>8s}")
    print("\n提案倍率は提案であって権限ではない。契約どおり RiskEngine / gateway / NAV / 証拠金が")
    print("独立に再検証し、減らす方向にしか効かない。0.0 は累積マイナス＝STOP 候補。")
    print("倍率>1.0 には戦略名の確定が要る（名指しできない戦略は増量しない）。")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--scorecard" in args:
        scorecard()
    else:
        since = None
        if "--since" in args:
            since = args[args.index("--since") + 1]
        pairs = [a for a in args if "_" in a and not a.startswith("-")]
        from engine import PAIRS
        append(pairs or PAIRS, since)
