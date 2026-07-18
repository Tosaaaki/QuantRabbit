#!/usr/bin/env python3
"""Virtual market session: real-mechanics paper trading for the duty agent.

Binds the VirtualBroker (OANDA mechanics, fills only at real quotes,
hash-chained ledger) to one of two honest feeds:

  --feed live    the account's own live pricing, polled read-only every
                 5s.  Wall-clock time.  Market closed / stale quotes =>
                 no fills, no order processing, refusal logged.
  --feed replay  the sealed M1 corpus between --from/--to, delivered
                 strictly in order as 4 quotes per bar (O,H,L,C bid/ask).
                 The sim clock is the bar's historical timestamp; no
                 lookahead is possible because the state file never
                 contains anything past the cursor.

Agent protocol (file-based, works from any agent harness):
  session-dir/
    state.json        account, positions, resting orders, latest quotes,
                      sim_time — rewritten atomically every tick
    inbox/            the agent drops one JSON per action:
                      {"action":"MARKET","pair":"USD_JPY","side":"LONG",
                       "units":10000,"tp_pips":5,"sl_pips":null}
                      {"action":"LIMIT","pair":...,"price":...}
                      {"action":"CLOSE","trade_id":"T000001","units":null}
                      {"action":"CANCEL","order_id":"O000001"}
                      {"action":"SET_EXIT","trade_id":...,"tp_price":...,
                       "sl_price":...}
    inbox/processed/  handled action files are renamed here (never deleted)
    ledger.jsonl      the broker's hash-chained ledger (every quote-caused
                      fill records the exact quote)

In replay mode --step makes the session turn-based: it advances one bar
whenever the agent writes inbox/STEP (renamed after consumption), so a
discretionary agent can think per bar.  ORDER AUTHORITY: NONE (paper
account only; this process cannot reach the real broker's order API).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time as time_mod
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quant_rabbit.analysis.market_status import compute_market_status
from quant_rabbit.virtual_broker import VirtualBroker, VirtualBrokerError

UTC = timezone.utc
POLL_SECONDS = 5.0
STALE_QUOTE_MAX_S = 90.0


def _write_state(session_dir: Path, broker: VirtualBroker, sim_time: str,
                 mode: str, note: str = "") -> None:
    state = {
        "mode": mode,
        "sim_time_utc": sim_time,
        "wall_time_utc": datetime.now(UTC).isoformat(),
        "account": broker.account() if broker.last_quotes else None,
        "positions": [vars(p) for p in broker.positions.values()],
        "resting_orders": [vars(o) for o in broker.orders.values()],
        "quotes": {
            pair: {"bid": q[0], "ask": q[1], "ts": q[2]}
            for pair, q in broker.last_quotes.items()
        },
        "note": note,
    }
    tmp = session_dir / ".state.json.tmp"
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))
    os.replace(tmp, session_dir / "state.json")


def _process_inbox(session_dir: Path, broker: VirtualBroker) -> int:
    inbox = session_dir / "inbox"
    done = inbox / "processed"
    done.mkdir(parents=True, exist_ok=True)
    handled = 0
    for path in sorted(inbox.glob("*.json")):
        try:
            action = json.loads(path.read_text())
            kind = action.get("action")
            if kind == "MARKET":
                broker.market_order(
                    action["pair"], action["side"], float(action["units"]),
                    tp_pips=action.get("tp_pips"), sl_pips=action.get("sl_pips"),
                )
            elif kind == "LIMIT":
                broker.limit_order(
                    action["pair"], action["side"], float(action["units"]),
                    price=float(action["price"]),
                    tp_pips=action.get("tp_pips"), sl_pips=action.get("sl_pips"),
                )
            elif kind == "CLOSE":
                units = action.get("units")
                broker.close_trade(action["trade_id"],
                                   units=float(units) if units else None)
            elif kind == "CANCEL":
                broker.cancel_order(action["order_id"])
            elif kind == "SET_EXIT":
                broker.set_exit(action["trade_id"],
                                tp_price=action.get("tp_price"),
                                sl_price=action.get("sl_price"))
            else:
                broker._log("AGENT_ACTION_REJECTED",
                            {"file": path.name, "error": f"unknown action {kind}"})
        except (VirtualBrokerError, KeyError, ValueError, json.JSONDecodeError) as exc:
            broker._log("AGENT_ACTION_REJECTED",
                        {"file": path.name, "error": str(exc)[:200]})
        os.replace(path, done / f"{int(time_mod.time()*1000)}_{path.name}")
        handled += 1
    return handled


def run_live(args, broker: VirtualBroker, session_dir: Path) -> None:
    from quant_rabbit.broker.oanda import OandaReadOnlyClient

    client = OandaReadOnlyClient()
    pairs = args.pairs.split(",")
    deadline = time_mod.time() + args.minutes * 60.0
    while time_mod.time() < deadline:
        now = datetime.now(UTC)
        if not compute_market_status(now).is_fx_open:
            _write_state(session_dir, broker, now.isoformat(), "live",
                         "MARKET_CLOSED: no fills, orders not processed")
            time_mod.sleep(30.0)
            continue
        try:
            quotes = client.quotes(pairs)
        except Exception as exc:
            broker._log("QUOTE_ERROR", {"error": str(exc)[:200]})
            time_mod.sleep(POLL_SECONDS)
            continue
        stale = any(
            (now - q.timestamp_utc).total_seconds() > STALE_QUOTE_MAX_S
            for q in quotes.values()
        )
        if stale:
            _write_state(session_dir, broker, now.isoformat(), "live",
                         "STALE_QUOTES: refusing fills and order processing")
            time_mod.sleep(POLL_SECONDS)
            continue
        for pair, q in quotes.items():
            broker.on_quote(pair, q.bid, q.ask, q.timestamp_utc.isoformat())
        _process_inbox(session_dir, broker)
        _write_state(session_dir, broker, now.isoformat(), "live")
        time_mod.sleep(POLL_SECONDS)


def _iter_replay_quotes(root: Path, pairs: list[str], time_from: str, time_to: str):
    """Merge pairs' M1 bars in time order; yield (epoch, pair, bid, ask, phase)."""

    rows = []
    for pair in pairs:
        for shard in sorted(root.glob(f"*/{pair}/{pair}_M1_BA_*.jsonl.gz")):
            with gzip.open(shard, "rt", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    stamp = row["time"][:19]
                    if not (time_from <= stamp < time_to):
                        continue
                    epoch = int(datetime.fromisoformat(stamp + "+00:00").timestamp())
                    rows.append((epoch, pair, row))
    rows.sort(key=lambda r: (r[0], r[1]))
    seen = set()
    for epoch, pair, row in rows:
        if (epoch, pair) in seen:
            continue
        seen.add((epoch, pair))
        b, a = row["bid"], row["ask"]
        for phase, key in (("O", "o"), ("H", "h"), ("L", "l"), ("C", "c")):
            yield epoch, pair, float(b[key]), float(a[key]), phase


def run_replay(args, broker: VirtualBroker, session_dir: Path) -> None:
    pairs = args.pairs.split(",")
    root = Path(args.corpus_root)
    bar_sleep = 1.0 / max(args.bars_per_second, 0.01)
    step_file = session_dir / "inbox" / "STEP"
    current_epoch = None
    for epoch, pair, bid, ask, phase in _iter_replay_quotes(
            root, pairs, args.time_from, args.time_to):
        if epoch != current_epoch:
            # bar boundary: let the agent act, pace the clock
            if current_epoch is not None:
                _process_inbox(session_dir, broker)
                _write_state(
                    session_dir, broker,
                    datetime.fromtimestamp(current_epoch + 60, tz=UTC).isoformat(),
                    "replay")
                if args.step:
                    while not step_file.exists():
                        time_mod.sleep(0.2)
                        _process_inbox(session_dir, broker)
                    os.replace(step_file, session_dir / "inbox" / "processed" /
                               f"{int(time_mod.time()*1000)}_STEP")
                else:
                    time_mod.sleep(bar_sleep)
            current_epoch = epoch
        broker.on_quote(pair, bid, ask, datetime.fromtimestamp(
            epoch, tz=UTC).isoformat() + f"#{phase}")
    _process_inbox(session_dir, broker)
    _write_state(session_dir, broker, "REPLAY_END", "replay", "replay finished")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feed", choices=["live", "replay"], required=True)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--pairs", default="USD_JPY,EUR_USD")
    parser.add_argument("--balance", type=float, default=200_000.0)
    parser.add_argument("--minutes", type=float, default=480.0, help="live mode duration")
    parser.add_argument("--corpus-root", default=(
        "/Users/tossaki/App/QuantRabbit-live/logs/replay/oanda_history_m1_2020_2026"))
    parser.add_argument("--from", dest="time_from", default="2026-01-05T00:00:00")
    parser.add_argument("--to", dest="time_to", default="2026-01-10T00:00:00")
    parser.add_argument("--bars-per-second", type=float, default=20.0)
    parser.add_argument("--step", action="store_true",
                        help="replay: advance one bar per inbox/STEP file")
    args = parser.parse_args()

    session_dir = args.session_dir
    (session_dir / "inbox" / "processed").mkdir(parents=True, exist_ok=True)
    broker = VirtualBroker(
        ledger_path=session_dir / "ledger.jsonl", balance_jpy=args.balance)
    snap_path = session_dir / "broker_snapshot.json"
    if snap_path.exists():
        broker.restore(json.loads(snap_path.read_text()))
    broker._log("SESSION_START", {
        "contract": "QR_VIRTUAL_MARKET_SESSION_V1",
        "feed": args.feed, "pairs": args.pairs, "balance": broker.balance_jpy,
        "order_authority": "NONE",
    })
    try:
        if args.feed == "live":
            run_live(args, broker, session_dir)
        else:
            run_replay(args, broker, session_dir)
    finally:
        tmp = session_dir / ".broker_snapshot.json.tmp"
        tmp.write_text(json.dumps(broker.snapshot(), ensure_ascii=False, sort_keys=True))
        os.replace(tmp, snap_path)
        broker._log("SESSION_STOP", {"account": broker.account() if broker.last_quotes else None})
    print(json.dumps({"status": "SESSION_DONE",
                      "account": broker.account() if broker.last_quotes else None},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
