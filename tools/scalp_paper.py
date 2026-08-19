#!/usr/bin/env python3
"""Paper loop for the frozen scalp rule: record entries live, resolve them from price.

    tools/scalp_paper.py log "USDJPY long 3.0"    # 3.0 = take-profit distance in pips
    tools/scalp_paper.py log "USDJPY short 5"
    tools/scalp_paper.py resolve                  # settle everything that has run its course
    tools/scalp_paper.py status

Why paper is enough
-------------------
Both exits are now numbers: the take-profit is chosen at entry and the stop is
frozen at -10 pips by `scalp_preregistration_v1.json`. Once the entry decision
exists with a timestamp, the outcome is arithmetic on price — no execution is
needed to learn whether the entry has an edge. So the scarce input is 45
point-in-time entry decisions, not 45 fills, and the 93%-margin book does not
block collecting them.

What paper does not capture: stop slippage in fast markets (the one difference
that genuinely matters), and the fact that a paper entry is psychologically
cheaper than a real one — which biases paper *optimistic*. Neither can make a
non-existent edge appear.

No hindsight is possible by construction:
  * the entry price is the executable quote at the moment of logging (ask for a
    long, bid for a short), fetched live and written immediately
  * take-profit and stop are fixed at entry and never edited
  * resolution walks S5 bars forward from the entry and stops at the first
    touch; a bar that touches both is settled as the STOP, the worst case
  * an entry cannot be logged with a timestamp in the past

Read-only against the broker. Nothing here places an order.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "research" / "paper" / "scalp_preregistration_v1.json"
LEDGER = ROOT / "research" / "paper" / "scalp_paper_ledger.jsonl"
ENV = Path("/Users/tossaki/App/QuantRabbit/.env.local")

# Scalps in the observed sample ran 0.03-3.3 hours. Beyond this the trade is no
# longer the thing being measured, so it is settled at market and marked.
MAX_HOLD_HOURS = 4.0


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


def iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("_JPY") else 0.0001


def normalize_pair(raw: str) -> str:
    m = re.match(r"^([A-Za-z]{3})[_/]?([A-Za-z]{3})$", raw.strip())
    if not m:
        raise SystemExit(f"unrecognised pair: {raw!r}")
    return f"{m.group(1).upper()}_{m.group(2).upper()}"


def read_ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text(encoding="utf-8").splitlines() if l.strip()]


def cmd_log(args, reg) -> int:
    parts = args.intake.split()
    if len(parts) < 2:
        raise SystemExit("need '<pair> <long|short> <tp_pips>' or '<pair> skip [note]'")
    pair = normalize_pair(parts[0])
    side = parts[1].lower()

    # SKIP is a first-class record, not the absence of one. An entry rule cannot
    # be recovered from entries alone: a classifier trained only on positives has
    # no decision boundary. Phase 1 (does the entry have an edge?) needs only
    # ENTERs, but phase 2 (what IS the rule?) needs the declines, and collecting
    # them later means collecting everything again.
    if side in ("skip", "pass"):
        token, acct, base = creds()
        q = get(base, token, f"/v3/accounts/{acct}/pricing", {"instruments": pair})["prices"][0]
        bid, ask = float(q["bids"][0]["price"]), float(q["asks"][0]["price"])
        row = {
            "schema": "QR_SCALP_PAPER_SKIP_V1",
            "logged_at_utc": iso(datetime.now(timezone.utc)),
            "quote_time_utc": q["time"], "pair": pair, "side": "skip",
            "bid_at_entry": bid, "ask_at_entry": ask,
            "spread_pips": round((ask - bid) / pip_size(pair), 2),
            "note": " ".join(parts[2:]) or None,
            # a decline is attributed only to this pair at this clock; it says
            # nothing about any other pair or any other moment
            "attribution": "THIS_PAIR_THIS_CLOCK_ONLY",
            "resolved": {"outcome": "SKIP", "pips": None, "at_utc": None},
        }
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows = read_ledger()
        skips = sum(1 for r in rows if r.get("side") == "skip")
        enters = sum(1 for r in rows if r.get("side") in ("long", "short"))
        print(f"logged SKIP  {pair} @ {bid}/{ask}" + (f"  note: {row['note']}" if row["note"] else ""))
        print(f"  entries {enters} / skips {skips}  (skips do not count toward the {reg['required_n']} "
              f"edge test; they are the phase-2 training material)")
        return 0

    if side not in ("long", "short"):
        raise SystemExit("side must be long, short or skip")
    if len(parts) < 3:
        raise SystemExit("an entry needs a take-profit distance, e.g. 'USDJPY long 3.0'")
    tp_pips = float(parts[2])
    tp_max = reg["population"]["take_profit_max_pips"]
    if not 0 < tp_pips <= tp_max:
        raise SystemExit(f"take-profit must be within (0, {tp_max}] to be in the registered population")

    token, acct, base = creds()
    q = get(base, token, f"/v3/accounts/{acct}/pricing", {"instruments": pair})["prices"][0]
    bid, ask = float(q["bids"][0]["price"]), float(q["asks"][0]["price"])
    pip = pip_size(pair)
    stop_pips = reg["rule"]["stop_distance_pips"]

    entry = ask if side == "long" else bid
    sign = 1 if side == "long" else -1
    row = {
        "schema": "QR_SCALP_PAPER_ENTRY_V1",
        "logged_at_utc": iso(datetime.now(timezone.utc)),
        "quote_time_utc": q["time"],
        "pair": pair, "side": side,
        "entry": entry, "bid_at_entry": bid, "ask_at_entry": ask,
        "spread_pips": round((ask - bid) / pip, 2),
        "tp_pips": tp_pips, "stop_pips": stop_pips,
        "tp_price": round(entry + sign * tp_pips * pip, 5),
        "stop_price": round(entry - sign * stop_pips * pip, 5),
        "prereg_stop": stop_pips, "prereg_tp_max": tp_max,
        "resolved": None,
    }
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    n = len([r for r in read_ledger() if r.get("resolved")])
    print(f"logged #{len(read_ledger())}  {pair} {side} entry {entry} "
          f"TP {row['tp_price']} ({tp_pips}p)  STOP {row['stop_price']} (-{stop_pips}p)  "
          f"spread {row['spread_pips']}p")
    print(f"  resolved so far: {n} / {reg['required_n']}")
    return 0


def resolve_one(base, token, row) -> dict | None:
    """Walk S5 bars from the entry; first touch wins; a bar touching both is a stop."""
    t0 = parse_time(row["quote_time_utc"])
    if datetime.now(timezone.utc) - t0 < timedelta(minutes=1):
        return None                                   # let at least one bar close
    pair, pip = row["pair"], pip_size(row["pair"])
    frm = iso(t0)
    try:
        candles = get(base, token, f"/v3/instruments/{pair}/candles",
                      {"granularity": "S5", "from": frm,
                       "count": str(int(MAX_HOLD_HOURS * 720) + 10), "price": "BA"})["candles"]
    except Exception as exc:
        print(f"  fetch failed for {row['logged_at_utc']}: {type(exc).__name__}", file=sys.stderr)
        return None
    deadline = t0 + timedelta(hours=MAX_HOLD_HOURS)
    long_ = row["side"] == "long"
    last_px = None
    for c in candles:
        if not c.get("complete"):
            continue
        t = parse_time(c["time"])
        if t < t0:
            continue
        if t > deadline:
            break
        # a long exits on the bid, a short on the ask
        lo = float(c["bid"]["l"]) if long_ else float(c["ask"]["l"])
        hi = float(c["bid"]["h"]) if long_ else float(c["ask"]["h"])
        last_px = float(c["bid"]["c"]) if long_ else float(c["ask"]["c"])
        hit_stop = lo <= row["stop_price"] if long_ else hi >= row["stop_price"]
        hit_tp = hi >= row["tp_price"] if long_ else lo <= row["tp_price"]
        if hit_stop:                                   # conservative when both touch
            return {"outcome": "STOP", "pips": -row["stop_pips"], "at_utc": iso(t)}
        if hit_tp:
            return {"outcome": "TP", "pips": row["tp_pips"], "at_utc": iso(t)}
    if last_px is not None and datetime.now(timezone.utc) > deadline:
        sign = 1 if long_ else -1
        return {"outcome": "TIMEOUT", "at_utc": iso(deadline),
                "pips": round(sign * (last_px - row["entry"]) / pip, 2)}
    return None                                        # still running


def cmd_resolve(args, reg) -> int:
    rows = read_ledger()
    token, acct, base = creds()
    changed = 0
    for r in rows:
        if r.get("resolved"):
            continue
        out = resolve_one(base, token, r)
        if out:
            r["resolved"] = out
            changed += 1
    if changed:
        LEDGER.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                          encoding="utf-8")
    print(f"resolved {changed} entr{'y' if changed == 1 else 'ies'}")
    return cmd_status(args, reg)


def cmd_status(args, reg) -> int:
    rows = read_ledger()
    skips = [r for r in rows if r.get("side") == "skip"]
    entries = [r for r in rows if r.get("side") in ("long", "short")]
    done = [r for r in entries if r.get("resolved")]
    open_ = [r for r in entries if not r.get("resolved")]
    need = reg["required_n"]
    print(f"\n=== paper scalp ledger / stop -{reg['rule']['stop_distance_pips']:.0f}p "
          f"/ TP<={reg['population']['take_profit_max_pips']:.0f}p ===")
    print(f"resolved {len(done)} / {need} required   still running {len(open_)}")
    print(f"phase 1 (does the entry pay?): {len(done)}/{need} resolved entries")
    print(f"phase 2 (what IS the rule?)  : {len(entries)} entries + {len(skips)} skips "
          f"= {len(rows)} labelled decisions")
    if args.detail and done:
        print(f"\n{'logged':17}{'pair':9}{'side':6}{'TP':>5}{'spread':>7}{'out':>9}{'pips':>7}")
        for r in done:
            g = r["resolved"]
            print(f"{r['logged_at_utc'][5:16]:17}{r['pair']:9}{r['side']:6}{r['tp_pips']:>5.1f}"
                  f"{r['spread_pips']:>7.1f}{g['outcome']:>9}{g['pips']:>7.1f}")
    if len(done) >= 2:
        p = [r["resolved"]["pips"] for r in done]
        mu, sd = statistics.mean(p), statistics.pstdev(p)
        lb = mu - reg["measurement"]["z"] * sd / (len(p) ** 0.5)
        win = 100 * sum(1 for x in p if x > 0) / len(p)
        from collections import Counter
        mix = Counter(r["resolved"]["outcome"] for r in done)
        print(f"\n  mean {mu:+.2f}p   sd {sd:.2f}   win {win:.0f}%   "
              f"one-sided 95% LB {lb:+.2f}p   {dict(mix)}")
        if len(done) < need:
            print(f"  VERDICT: NOT YET — {need - len(done)} more before any verdict is admissible")
        else:
            print(f"  VERDICT: {'ACCEPT' if lb > 0 else 'REJECT / keep collecting'}")
        print("\n  paper omits stop slippage and is psychologically cheaper than live;")
        print("  both bias this optimistic. It can refute an edge, and it can only suggest one.")
    return 0


def main(argv=None) -> int:
    reg = json.loads(PREREG.read_text(encoding="utf-8"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lg = sub.add_parser("log"); lg.add_argument("intake"); lg.add_argument("--detail", action="store_true")
    rs = sub.add_parser("resolve"); rs.add_argument("--detail", action="store_true")
    st = sub.add_parser("status"); st.add_argument("--detail", action="store_true")
    args = ap.parse_args(argv)
    return {"log": cmd_log, "resolve": cmd_resolve, "status": cmd_status}[args.cmd](args, reg)


if __name__ == "__main__":
    raise SystemExit(main())
