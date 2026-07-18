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


class GoldenBurstBot:
    """Worker bot living INSIDE the virtual session: same broker, same
    fill engine, same ledger as the duty agent.  Runs the vendored
    golden-day MomentumBurst with arsenal protections (max 3 concurrent,
    4h hard ceiling).  Sizing: NAV-proportional 4.3x per position."""

    WARMUP = 40
    MAX_CONCURRENT = 3
    CEILING_S = 4 * 3600
    PAIR = "USD_JPY"

    def __init__(self, broker: VirtualBroker, blind_spread: bool = False):
        import importlib.util
        from types import ModuleType
        vendor = REPO_ROOT / "vendored" / "golden_20251209"

        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod

        pkg = ModuleType("analysis"); pkg.__path__ = []
        sys.modules["analysis"] = pkg
        pkg.ma_projection = load("analysis.ma_projection", vendor / "analysis_ma_projection.py")
        self.calc_core = load("golden_calc_core", vendor / "indicators_calc_core.py")
        self.strategy = load("golden_momentum_burst", vendor / "strategies_micro_momentum_burst.py").MomentumBurstMicro
        self.broker = broker
        self.blind_spread = blind_spread
        self.bars: list[dict] = []
        self.my_trades: dict[str, float] = {}  # trade_id -> entry_epoch

    def on_bar_closed(self, pair: str, bar: dict, bar_epoch: int) -> None:
        if pair != self.PAIR:
            return
        self.bars.append(bar)
        if len(self.bars) > 2000:
            self.bars.pop(0)
        # ceiling exits for my open trades
        for trade_id in list(self.my_trades):
            if trade_id not in self.broker.positions:
                del self.my_trades[trade_id]
                continue
            if bar_epoch - self.my_trades[trade_id] >= self.CEILING_S:
                try:
                    self.broker.close_trade(trade_id)
                except VirtualBrokerError:
                    pass
                del self.my_trades[trade_id]
        if len(self.bars) < self.WARMUP:
            return
        live_mine = [t for t in self.my_trades if t in self.broker.positions]
        if len(live_mine) >= self.MAX_CONCURRENT:
            return
        import pandas as pd
        mid_c = pd.Series([(b["bid_c"] + b["ask_c"]) / 2 for b in self.bars])
        mid_h = pd.Series([(b["bid_h"] + b["ask_h"]) / 2 for b in self.bars])
        mid_l = pd.Series([(b["bid_l"] + b["ask_l"]) / 2 for b in self.bars])
        ma10 = mid_c.rolling(10, min_periods=10).mean().iloc[-1]
        ma20 = mid_c.rolling(20, min_periods=20).mean().iloc[-1]
        ema20 = mid_c.ewm(span=20, adjust=False, min_periods=20).mean().iloc[-1]
        vol_5m = (mid_c.diff().abs().rolling(5, min_periods=5).mean() / 0.01).iloc[-1]
        rsi = self.calc_core._rsi(mid_c, 14).iloc[-1]
        atr = self.calc_core._atr(mid_h, mid_l, mid_c, 14).iloc[-1]
        adx = self.calc_core._adx(mid_h, mid_l, mid_c, 14).iloc[-1]
        if any(pd.isna(v) for v in (ma10, ma20, ema20, vol_5m, rsi, atr, adx)):
            return
        last = self.bars[-1]
        fac = {
            "close": float(mid_c.iloc[-1]),
            "ma10": float(ma10), "ma20": float(ma20), "ema20": float(ema20),
            "rsi": float(rsi), "atr": float(atr), "adx": float(adx),
            "vol_5m": float(vol_5m),
            "spread_pips": 0.0 if self.blind_spread else (last["ask_c"] - last["bid_c"]) / 0.01,
            "candles": [
                {"high": (b["bid_h"] + b["ask_h"]) / 2, "low": (b["bid_l"] + b["ask_l"]) / 2,
                 "open": (b["bid_o"] + b["ask_o"]) / 2, "close": (b["bid_c"] + b["ask_c"]) / 2}
                for b in self.bars[-4:]
            ],
        }
        signal = self.strategy.check(fac)
        if not signal or signal.get("action") not in {"OPEN_LONG", "OPEN_SHORT"}:
            return
        side = "LONG" if signal["action"] == "OPEN_LONG" else "SHORT"
        try:
            acct = self.broker.account()
            units = max(acct["equity_jpy"], 0.0) * 4.3 / fac["close"]
            if units <= 0:
                return
            trade_id = self.broker.market_order(
                self.PAIR, side, units,
                tp_pips=float(signal["tp_pips"]), sl_pips=float(signal["sl_pips"]))
            self.my_trades[trade_id] = bar_epoch
        except VirtualBrokerError:
            return

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
    now = time_mod.time()
    for path in sorted(inbox.glob("*.json")):
        try:
            if now - path.stat().st_mtime < 0.5:
                continue  # writer may still be mid-write; pick up next tick
        except OSError:
            continue
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


def run_live(args, broker: VirtualBroker, session_dir: Path, bot=None) -> None:
    from quant_rabbit.broker.oanda import OandaReadOnlyClient

    client = OandaReadOnlyClient()
    pairs = args.pairs.split(",")
    deadline = time_mod.time() + args.minutes * 60.0
    live_bars: dict[str, dict] = {}
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
            minute = int(now.timestamp() // 60) * 60
            bar = live_bars.get(pair)
            if bar is not None and bar["epoch"] != minute:
                if bot is not None:
                    bot.on_bar_closed(pair, bar, bar["epoch"])
                bar = None
            if bar is None:
                live_bars[pair] = {"epoch": minute,
                                   "bid_o": q.bid, "bid_h": q.bid, "bid_l": q.bid, "bid_c": q.bid,
                                   "ask_o": q.ask, "ask_h": q.ask, "ask_l": q.ask, "ask_c": q.ask}
            else:
                bar["bid_h"] = max(bar["bid_h"], q.bid); bar["bid_l"] = min(bar["bid_l"], q.bid)
                bar["ask_h"] = max(bar["ask_h"], q.ask); bar["ask_l"] = min(bar["ask_l"], q.ask)
                bar["bid_c"] = q.bid; bar["ask_c"] = q.ask
        _process_inbox(session_dir, broker)
        _write_state(session_dir, broker, now.isoformat(), "live")
        time_mod.sleep(POLL_SECONDS)


def _iter_replay_quotes(root: Path, pairs: list[str], time_from: str,
                        time_to: str, intrabar: str = "OHLC",
                        granularity: str = "M1"):
    """Merge pairs' M1 bars in time order; yield (epoch, pair, bid, ask, phase).

    Streams year by year so full-history sessions stay in bounded memory.
    ``intrabar`` declares the synthetic intrabar path (the true tick path
    is unknown): OHLC favors longs' TP on both-touch bars, OLHC favors
    shorts'/SL — run both to bracket ambiguous outcomes.
    """

    phase_keys = {"OHLC": (("O", "o"), ("H", "h"), ("L", "l"), ("C", "c")),
                  "OLHC": (("O", "o"), ("L", "l"), ("H", "h"), ("C", "c"))}[intrabar]
    for year in range(int(time_from[:4]), int(time_to[:4]) + 1):
        rows = []
        for pair in pairs:
            for shard in sorted(root.glob(f"*/{pair}/{pair}_{granularity}_BA_{year}*.jsonl.gz")):
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
            for phase, key in phase_keys:
                yield epoch, pair, float(b[key]), float(a[key]), phase


def run_replay(args, broker: VirtualBroker, session_dir: Path, bot=None) -> None:
    pairs = args.pairs.split(",")
    root = Path(args.corpus_root)
    bar_sleep = 1.0 / max(args.bars_per_second, 0.01)
    no_sleep = args.bars_per_second >= 10000
    bar_count = 0
    step_file = session_dir / "inbox" / "STEP"
    current_epoch = None
    pending_bars: dict[str, dict] = {}
    bot_minute: dict[str, dict] = {}
    aggregate_bot_bars = bot is not None and args.bot_bar == "M1" and args.granularity != "M1"
    for epoch, pair, bid, ask, phase in _iter_replay_quotes(
            root, pairs, args.time_from, args.time_to, args.intrabar,
            args.granularity):
        if aggregate_bot_bars:
            minute = epoch // 60 * 60
            mb = bot_minute.get(pair)
            if mb is not None and mb["epoch"] != minute:
                bot.on_bar_closed(pair, mb, mb["epoch"])
                mb = None
            if mb is None:
                bot_minute[pair] = {"epoch": minute,
                                    "bid_o": bid, "bid_h": bid, "bid_l": bid, "bid_c": bid,
                                    "ask_o": ask, "ask_h": ask, "ask_l": ask, "ask_c": ask}
            else:
                mb["bid_h"] = max(mb["bid_h"], bid); mb["bid_l"] = min(mb["bid_l"], bid)
                mb["ask_h"] = max(mb["ask_h"], ask); mb["ask_l"] = min(mb["ask_l"], ask)
                mb["bid_c"] = bid; mb["ask_c"] = ask
        boundary = epoch != current_epoch
        if boundary and current_epoch is not None and bot is not None and not aggregate_bot_bars:
            # previous bar(s) are complete NOW, before the new bar's O
            # quote overwrites pending_bars
            for bpair, bbar in list(pending_bars.items()):
                if bbar["epoch"] == current_epoch:
                    bot.on_bar_closed(bpair, bbar, current_epoch)
        if phase == "O":
            pending_bars[pair] = {"bid_o": bid, "ask_o": ask, "bid_h": bid, "bid_l": bid,
                                  "ask_h": ask, "ask_l": ask, "bid_c": bid, "ask_c": ask,
                                  "epoch": epoch}
        else:
            pb = pending_bars.get(pair)
            if pb is not None:
                pb["bid_h"] = max(pb["bid_h"], bid); pb["bid_l"] = min(pb["bid_l"], bid)
                pb["ask_h"] = max(pb["ask_h"], ask); pb["ask_l"] = min(pb["ask_l"], ask)
                pb["bid_c"] = bid; pb["ask_c"] = ask
        if epoch != current_epoch:
            # bar boundary: let the agent act, pace the clock
            if current_epoch is not None:
                bar_count += 1
                if bar_count % args.state_every == 0:
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
                elif not no_sleep:
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
    parser.add_argument("--bot", choices=["golden_burst", "golden_burst_blindspread"], default=None,
                        help="built-in worker bot (same broker/ledger)")
    parser.add_argument("--bot-module", default=None,
                        help="path to ANY bot file: <file.py>[:ClassName]; the class "
                             "takes (broker) and implements on_bar_closed(pair, bar, epoch)")
    parser.add_argument("--granularity", choices=["M1", "S5"], default="M1",
                        help="replay feed granularity (S5 = 12x finer fill realism)")
    parser.add_argument("--bot-bar", choices=["feed", "M1"], default="feed",
                        help="bot decision cadence: per feed bar, or aggregated M1")
    parser.add_argument("--state-every", type=int, default=1,
                        help="replay: write state/process inbox every N bars (lab speed)")
    parser.add_argument("--fast-ledger", action="store_true",
                        help="ledger flush without fsync (lab runs)")
    parser.add_argument("--intrabar", choices=["OHLC", "OLHC"], default="OHLC",
                        help="declared synthetic intrabar path; run both to bracket "
                             "both-touch ambiguity")
    args = parser.parse_args()

    session_dir = args.session_dir
    (session_dir / "inbox" / "processed").mkdir(parents=True, exist_ok=True)
    broker = VirtualBroker(
        ledger_path=session_dir / "ledger.jsonl", balance_jpy=args.balance,
        fast_ledger=args.fast_ledger)
    snap_path = session_dir / "broker_snapshot.json"
    if snap_path.exists():
        broker.restore(json.loads(snap_path.read_text()))
    broker._log("SESSION_START", {
        "contract": "QR_VIRTUAL_MARKET_SESSION_V1",
        "feed": args.feed, "pairs": args.pairs, "balance": broker.balance_jpy,
        "order_authority": "NONE",
    })
    bot = None
    if args.bot == "golden_burst":
        bot = GoldenBurstBot(broker)
    elif args.bot == "golden_burst_blindspread":
        # live-faithful configuration: the 2025-12 live worker's spread
        # monitor supplied nothing, so its gate never saw the spread
        bot = GoldenBurstBot(broker, blind_spread=True)
    elif args.bot_module:
        import importlib.util as _ilu
        spec_str = args.bot_module
        module_path, _, class_name = spec_str.partition(":")
        _spec = _ilu.spec_from_file_location("dojo_custom_bot", module_path)
        _mod = _ilu.module_from_spec(_spec)
        sys.modules["dojo_custom_bot"] = _mod
        _spec.loader.exec_module(_mod)
        bot_cls = getattr(_mod, class_name or "Bot")
        bot = bot_cls(broker)
        broker._log("BOT_LOADED", {"module": module_path, "class": class_name or "Bot"})
    try:
        if args.feed == "live":
            run_live(args, broker, session_dir, bot=bot)
        else:
            run_replay(args, broker, session_dir, bot=bot)
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
