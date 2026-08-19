#!/usr/bin/env python3
"""Verify the scalp method against a frozen pre-registration. Read-only.

    tools/scalp_verify.py            # running state
    tools/scalp_verify.py --detail   # every trade and why it counted or did not

Why this exists
---------------
The 15-month record contains one recurring failure: a number is measured, a
parameter is chosen by looking at that same number, and the result is reported
as evidence. It happened to the contextual candidate engine (the CHF cluster),
to the quiet gate (the 10-pip... the 67% quantile), and it nearly happened to
the scalp stop — 10 pips was picked by scanning five values on 17 trades.

So the rule is frozen in `research/paper/scalp_preregistration_v1.json` before
the data arrives, this tool reads it, and it refuses to run if the parameters
changed after clean trades began accruing. The prior 17 trades are fitting data
and are never counted.

Discipline is measured, not assumed. A trade opened without a broker stop, or
above the registered margin fraction, is recorded as a VIOLATION and excluded —
visibly, with a reason, never silently dropped.

Read-only by construction: this tool never constructs an order payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "research" / "paper" / "scalp_preregistration_v1.json"
STATE = ROOT / "research" / "paper" / "scalp_verify_state.json"
ENV = Path("/Users/tossaki/App/QuantRabbit/.env.local")


def creds():
    env = {}
    for line in ENV.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k] = v.strip()
    return env["QR_OANDA_TOKEN"], env["QR_OANDA_ACCOUNT_ID"], env.get(
        "QR_OANDA_BASE_URL", "https://api-fxtrade.oanda.com")


def get(base, token, path, query=None):
    url = f"{base}{path}" + (f"?{urllib.parse.urlencode(query)}" if query else "")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(re.sub(r"\.(\d{6})\d*", r".\1", s).replace("Z", "+00:00"))


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("_JPY") else 0.0001


def prereg_fingerprint(reg: dict) -> str:
    """Hash only the fields that must not move once evidence starts accruing."""
    frozen = {
        "stop": reg["rule"]["stop_distance_pips"],
        "tp_max": reg["population"]["take_profit_max_pips"],
        "margin": reg["rule"]["max_margin_used_fraction"],
        "required_n": reg["required_n"],
        "start": reg["evidence_start_transaction_id"],
    }
    return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()[:16]


def collect(reg: dict):
    """Pull trades after the registered start id and classify them."""
    token, acct, base = creds()
    start = reg["evidence_start_transaction_id"]
    tx = get(base, token, f"/v3/accounts/{acct}/transactions/sinceid", {"id": start})["transactions"]
    summary = get(base, token, f"/v3/accounts/{acct}/summary")["account"]
    nav = float(summary["NAV"])

    tp_price, sl_price = {}, {}
    for d in tx:
        tid = d.get("tradeID")
        if not tid:
            continue
        if d["type"] == "TAKE_PROFIT_ORDER":
            tp_price.setdefault(tid, float(d["price"]))
        elif d["type"] == "STOP_LOSS_ORDER":
            sl_price.setdefault(tid, float(d["price"]))

    opened = {}
    for d in tx:
        if d["type"] == "ORDER_FILL" and d.get("tradeOpened"):
            to = d["tradeOpened"]
            opened[to["tradeID"]] = {
                "time": parse_time(d["time"]), "units": int(to["units"]),
                "price": float(d["price"]), "pair": d["instrument"],
            }
    closes = {}
    for d in tx:
        if d["type"] != "ORDER_FILL":
            continue
        for tc in d.get("tradesClosed", []) + d.get("tradesReduced", []):
            closes.setdefault(tc["tradeID"], []).append(
                {"pl": float(tc["realizedPL"]), "time": parse_time(d["time"]),
                 "price": float(d["price"]), "reason": d.get("reason")})

    rows = []
    for tid, o in opened.items():
        tp = tp_price.get(tid)
        if tp is None:
            continue
        pip = pip_size(o["pair"])
        tp_dist = abs(tp - o["price"]) / pip
        if tp_dist > reg["population"]["take_profit_max_pips"]:
            continue                      # not the registered population
        legs = closes.get(tid)
        if not legs:
            continue                      # still open, not yet evidence
        last = max(legs, key=lambda c: c["time"])
        pips = (last["price"] - o["price"]) / pip * (1 if o["units"] > 0 else -1)

        violations = []
        sl = sl_price.get(tid)
        if reg["rule"]["stop_required"] and sl is None:
            violations.append("NO_STOP_ATTACHED")
        elif sl is not None:
            d_sl = abs(sl - o["price"]) / pip
            if d_sl > reg["rule"]["stop_distance_pips"] + 0.5:
                violations.append(f"STOP_TOO_FAR_{d_sl:.1f}p")
        # margin at entry is not in the transaction stream; use current NAV as the
        # reference and flag notional that could not have satisfied the fraction.
        margin = abs(o["units"]) * o["price"] / 25.0
        if margin / nav > reg["rule"]["max_margin_used_fraction"]:
            violations.append(f"MARGIN_{margin/nav:.0%}")
        if pips <= -25.0:
            violations.append(f"LOSS_BEYOND_ABORT_{pips:.1f}p")

        rows.append({
            "trade_id": tid, "pair": o["pair"], "opened": o["time"].strftime("%m-%d %H:%M"),
            "units": abs(o["units"]), "tp_dist": round(tp_dist, 1),
            "stop_dist": round(abs(sl - o["price"]) / pip, 1) if sl else None,
            "pips": round(pips, 1), "pl": sum(c["pl"] for c in legs),
            "reason": last["reason"], "violations": violations,
        })
    rows.sort(key=lambda r: r["opened"])
    return rows, nav


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detail", action="store_true", help="list every trade")
    args = ap.parse_args(argv)

    reg = json.loads(PREREG.read_text(encoding="utf-8"))
    fp = prereg_fingerprint(reg)

    rows, nav = collect(reg)
    clean = [r for r in rows if not r["violations"]]
    dirty = [r for r in rows if r["violations"]]

    # Tamper check: the frozen parameters may not move once evidence exists.
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    if state.get("fingerprint") and state["fingerprint"] != fp and state.get("clean_count", 0) > 0:
        print(f"REFUSING TO REPORT: pre-registration changed after {state['clean_count']} clean "
              f"trades were already recorded ({state['fingerprint']} -> {fp}).", file=sys.stderr)
        print("The registered rule is frozen. Start a new registration instead of editing this one.",
              file=sys.stderr)
        return 2

    stop = reg["rule"]["stop_distance_pips"]
    need = reg["required_n"]
    print(f"=== scalp verification / prereg {fp} / stop -{stop:.0f}p / TP<={reg['population']['take_profit_max_pips']:.0f}p ===")
    print(f"evidence starts after transaction {reg['evidence_start_transaction_id']}  |  NAV {nav:,.0f}")
    print(f"the 17 prior trades are FITTING DATA and are not counted\n")

    if args.detail and rows:
        print(f"{'opened':13}{'pair':9}{'units':>7}{'TP':>6}{'SL':>6}{'pips':>7}{'PL':>9}  status")
        for r in rows:
            st = "counted" if not r["violations"] else " / ".join(r["violations"])
            print(f"{r['opened']:13}{r['pair']:9}{r['units']:>7}{r['tp_dist']:>6.1f}"
                  f"{(r['stop_dist'] if r['stop_dist'] is not None else float('nan')):>6.1f}"
                  f"{r['pips']:>7.1f}{r['pl']:>9,.0f}  {st}")
        print()

    print(f"clean trades {len(clean)} / {need} required     discipline violations {len(dirty)}")
    if dirty:
        from collections import Counter
        for k, v in Counter(v for r in dirty for v in r["violations"]).most_common():
            print(f"    {k}: {v}")

    if len(clean) >= 2:
        p = [r["pips"] for r in clean]
        mu, sd = statistics.mean(p), statistics.pstdev(p)
        lb = mu - reg["measurement"]["z"] * sd / (len(p) ** 0.5)
        win = 100 * sum(1 for x in p if x > 0) / len(p)
        print(f"\n  mean {mu:+.2f} pips   sd {sd:.2f}   win {win:.0f}%   "
              f"one-sided 95% LB {lb:+.2f} pips")
        if len(clean) >= need:
            verdict = "ACCEPT — the edge is demonstrated" if lb > 0 else (
                "REJECT" if len(clean) >= 90 else "NOT YET — LB not positive, keep collecting")
            print(f"  VERDICT: {verdict}")
        else:
            print(f"  VERDICT: NOT YET — {need - len(clean)} more clean trades before any verdict is admissible")
    elif clean:
        print("\n  1 clean trade — no statistic is admissible yet")
    else:
        print("\n  no clean trades yet")

    STATE.write_text(json.dumps({
        "fingerprint": fp, "clean_count": len(clean), "violation_count": len(dirty),
        "updated_at_utc": datetime.now().astimezone().isoformat(),
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
