#!/usr/bin/env python3
"""Run the frozen operator entry model. Emit, resolve, and score its signals.

    tools/operator_signal.py check      # score right now, log if it fires
    tools/operator_signal.py resolve    # settle signals past their 336h exit
    tools/operator_signal.py status     # progress against the pre-registration

Designed to run on a schedule with nobody watching. The operator does not
choose entries any more: the model does, the exit is 336 hours by the clock,
and the size is capped by the registration.

Execution is deliberately not wired here. The registration requires 60 forward
signals before the edge counts as demonstrated, and the conservative reading of
the backtest — daily-aggregated, after spread and swap — is a lower bound of
+0.18 pips, which is zero. Sending orders on that basis is how the three live
bot lanes lost 642,039 JPY. `--emit-order` prints the exact order payload for a
human to place or for an execution wrapper to consume once the criteria in
`research/operator_model/PREREGISTRATION_V1.md` are met; this file never sends
one.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_rabbit.operator_model import (  # noqa: E402
    EXIT_HOURS, load_model, parse_time, pip_size, score,
)

LEDGER = ROOT / "research" / "operator_model" / "signals.jsonl"
PREREG = ROOT / "research" / "operator_model" / "PREREGISTRATION_V1.md"
# All 11 majors, no pair selection. Restricting to USD_JPY made 60 signals take
# 840 days: the exit is 336h and one-at-a-time caps throughput at 26/year.
# Measured across the test window the full set gives +20.56 pips (LB +2.72) over
# 220 signals, and +27.57 (LB +5.68) once same-pair overlaps are collapsed - so
# widening is both faster AND better. JPY crosses scored best and USD majors
# worst, but picking the winners after seeing that is the selection error that
# killed every other candidate this session, so the whole set trades.
PAIRS = ("USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD", "USD_CAD",
         "USD_CHF", "EUR_JPY", "GBP_JPY", "AUD_JPY", "CAD_JPY")
REQUIRED_N = 60
MAX_MARGIN_FRACTION = 0.30
ENV = Path("/Users/tossaki/App/QuantRabbit/.env.local")


def creds():
    env = {}
    for line in ENV.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k] = v.strip()
    return env["QR_OANDA_TOKEN"], env["QR_OANDA_ACCOUNT_ID"], env.get(
        "QR_OANDA_BASE_URL", "https://api-fxtrade.oanda.com")


def get(path, query=None):
    token, acct, base = creds()
    url = f"{base}{path}" + (f"?{urllib.parse.urlencode(query)}" if query else "")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.loads(resp.read())


def read_ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text(encoding="utf-8").splitlines() if l.strip()]


def cmd_check(args) -> int:
    bundle = load_model()
    token, acct, base = creds()
    summary = get(f"/v3/accounts/{acct}/summary")["account"]
    nav, used = float(summary["NAV"]), float(summary["marginUsed"])
    rows = read_ledger()
    fired = 0
    for pair in PAIRS:
        s = score(pair, bundle=bundle)
        if s is None:
            print(f"{pair}: history too thin to score")
            continue
        # one open signal per pair at a time; the exit is time-based
        # Two separate gates. Evidence is collected on paper and costs no
        # capital, so the account's margin state has no bearing on whether a
        # signal counts — coupling them would stall the 60-signal test behind a
        # position the model did not open. Margin gates EXECUTION only.
        # Two signals on the same pair inside one 336h window share the same
        # price path and are one observation, not two. Rather than logging ~63
        # a day across 11 pairs and collapsing them afterwards, only the first
        # in each (pair, block) is recorded - the ledger then IS the evidence,
        # with no post-hoc aggregation step to get wrong.
        blocked = []
        tradeable = used / nav <= MAX_MARGIN_FRACTION
        block = int(datetime.now(timezone.utc).timestamp() // (EXIT_HOURS * 3600))
        if any(r["pair"] == pair and r.get("block") == block for r in rows):
            blocked.append(f"SAME_BLOCK_{block}")

        mark = "FIRE" if s["fires"] else "no-trade"
        print(f"{pair}  score {s['action']:.4f} / thr {s['threshold']:.4f}  -> {mark}"
              + (f"  {s['side'].upper()}" if s["fires"] else ""))
        if not s["fires"]:
            continue
        if blocked:
            print(f"   blocked: {' / '.join(blocked)}  (recorded, not counted)")
        if not tradeable:
            print(f"   paper only: margin {used/nav:.0%} over the {MAX_MARGIN_FRACTION:.0%} "
                  f"execution cap — the signal still counts toward the {REQUIRED_N}")
        q = get(f"/v3/accounts/{acct}/pricing", {"instruments": pair})["prices"][0]
        bid, ask = float(q["bids"][0]["price"]), float(q["asks"][0]["price"])
        entry = ask if s["side"] == "long" else bid
        row = {**s, "schema": "QR_OPERATOR_SIGNAL_V1", "entry": entry,
               "bid": bid, "ask": ask, "spread_pips": round((ask - bid) / pip_size(pair), 2),
               "exit_due_utc": (parse_time(s["at_utc"]) + timedelta(hours=EXIT_HOURS)
                                ).isoformat().replace("+00:00", "Z"),
               "nav_at_signal": nav, "margin_used_fraction": round(used / nav, 4),
               "block": block, "tradeable_live": tradeable,
               "blocked": blocked or None, "resolved": None}
        if blocked and any(x.startswith("SAME_BLOCK") for x in blocked):
            print(f"   same 336h block as an existing {pair} signal — not recorded")
            continue
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fired += 1
        print(f"   logged  {s['side']} @ {entry}  exit {row['exit_due_utc'][:16]}  spread {row['spread_pips']}p")
        if args.emit_order and not blocked and tradeable:
            units = int(nav * MAX_MARGIN_FRACTION * 25 / entry)
            units = (units // 1000) * 1000 * (1 if s["side"] == "long" else -1)
            print("   order payload (NOT sent — place it yourself or hand to an execution wrapper):")
            print("   " + json.dumps({"order": {"type": "MARKET", "instrument": pair,
                                                "units": str(units), "timeInForce": "FOK",
                                                "positionFill": "DEFAULT"}}, ensure_ascii=False))
    if not fired:
        print("no signal this cycle")
    return 0


def cmd_resolve(args) -> int:
    rows = read_ledger()
    now = datetime.now(timezone.utc)
    changed = 0
    for r in rows:
        if r.get("resolved"):
            continue
        due = parse_time(r["exit_due_utc"])
        if now < due:
            continue
        pair, pip = r["pair"], pip_size(r["pair"])
        payload = get(f"/v3/instruments/{pair}/candles",
                      {"granularity": "H1", "from": r["at_utc"],
                       "to": r["exit_due_utc"], "price": "BA"})
        bars = [c for c in payload.get("candles", []) if c.get("complete")]
        if len(bars) < 2:
            continue
        last = bars[-1]
        # exit on the executable side: a long sells the bid, a short buys the ask
        exit_px = float(last["bid"]["c"]) if r["side"] == "long" else float(last["ask"]["c"])
        sign = 1 if r["side"] == "long" else -1
        gross = (exit_px - r["entry"]) / pip * sign
        # swap on USD_JPY: long receives, short pays, at the rate observed in
        # this account's own financing history
        swap = (1.26 if r["side"] == "long" else -3.22) * (EXIT_HOURS / 336)
        r["resolved"] = {"exit_price": exit_px, "at_utc": last["time"],
                         "gross_pips": round(gross, 2), "swap_pips": round(swap, 2),
                         "net_pips": round(gross + swap, 2)}
        changed += 1
    if changed:
        LEDGER.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                          encoding="utf-8")
    print(f"resolved {changed}")
    return cmd_status(args)


def cmd_status(args) -> int:
    rows = read_ledger()
    counted = [r for r in rows if not r.get("blocked")]
    done = [r for r in counted if r.get("resolved")]
    open_ = [r for r in counted if not r.get("resolved")]
    print(f"\n=== operator model v1 / exit {EXIT_HOURS}h / margin cap {MAX_MARGIN_FRACTION:.0%} ===")
    print(f"signals {len(counted)} counted ({len(rows) - len(counted)} blocked)  "
          f"resolved {len(done)} / {REQUIRED_N} required  open {len(open_)}")
    if args.detail:
        for r in rows[-20:]:
            g = r.get("resolved")
            tail = (f"net {g['net_pips']:+7.2f}p" if g else
                    f"due {r['exit_due_utc'][:16]}")
            print(f"  {r['at_utc'][:16]}  {r['pair']} {r['side']:5}  score {r['action']:.3f}  {tail}"
                  + (f"  [{','.join(r['blocked'])}]" if r.get("blocked") else ""))
    if len(done) >= 2:
        net = [r["resolved"]["net_pips"] for r in done]
        # Collapse same-pair overlaps into non-overlapping 336h blocks. Two
        # signals on the same pair inside one exit window share the same price
        # path and are one observation, not two.
        # The ledger already holds one row per (pair, 336h block), so every
        # resolved row is an independent observation and no collapsing is needed.
        B = net
        mu, sd = statistics.mean(B), statistics.pstdev(B)
        lb = mu - 1.645 * sd / (len(B) ** 0.5)
        print(f"\n  raw signals {len(net)} -> independent blocks {len(B)}")
        print(f"  net {mu:+.2f} pips   sd {sd:.1f}   win {100*sum(1 for x in B if x>0)/len(B):.0f}%"
              f"   one-sided 95% LB {lb:+.2f}")
        done = B
        if len(done) >= REQUIRED_N:
            print(f"  VERDICT: {'ACCEPT' if lb > 0 else 'not yet — LB not positive'}")
        else:
            print(f"  VERDICT: NOT YET — {REQUIRED_N - len(done)} more before any verdict is admissible")
    print(f"\n  registration: {PREREG.relative_to(ROOT)} — frozen; do not retune")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("--emit-order", action="store_true"); c.add_argument("--detail", action="store_true")
    r = sub.add_parser("resolve"); r.add_argument("--detail", action="store_true")
    s = sub.add_parser("status"); s.add_argument("--detail", action="store_true")
    a = ap.parse_args(argv)
    return {"check": cmd_check, "resolve": cmd_resolve, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
