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
import fcntl
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
from quant_rabbit.dojo_paper_contract import (
    DojoPaperContractError,
    bot_contract,
    build_session_contract,
    file_sha256,
    prepare_drain_contract,
    publish_immutable_json,
    runtime_manifest,
)
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
_RUNTIME_LOCK_HANDLE = None


def _acquire_runtime_lock(session_dir: Path) -> None:
    """Hold one non-blocking process lock for the whole session lifetime."""

    global _RUNTIME_LOCK_HANDLE
    handle = (session_dir / ".virtual-market-runtime.lock").open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise DojoPaperContractError(
            f"another virtual-market process owns {session_dir}"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    _RUNTIME_LOCK_HANDLE = handle


class DrainOnlyController:
    """Resolve existing paper exposure without admitting another entry.

    TP/SL and margin events remain owned by ``VirtualBroker.on_quote``.  The
    controller adds only the original time ceiling, measured from the
    persisted ``opened_ts``.  It has no entry/order method by construction.
    """

    def __init__(self, broker: VirtualBroker, ceiling_minutes: int):
        self.broker = broker
        self.ceiling_seconds = ceiling_minutes * 60

    def on_quote(self, pair: str, quote_ts: str) -> None:
        quote_epoch = self.broker._ts_epoch(quote_ts)
        if quote_epoch is None:
            return
        for trade_id, pos in list(self.broker.positions.items()):
            if pos.pair != pair:
                continue
            opened_epoch = self.broker._ts_epoch(pos.opened_ts)
            if opened_epoch is None:
                raise VirtualBrokerError(
                    f"drain cannot parse opened_ts for {trade_id}"
                )
            if quote_epoch - opened_epoch < self.ceiling_seconds:
                continue
            self.broker._log(
                "DRAIN_CEILING_DUE",
                {
                    "trade_id": trade_id,
                    "pair": pair,
                    "strategy_tag": pos.strategy_tag,
                    "opened_ts": pos.opened_ts,
                    "ceiling_seconds": self.ceiling_seconds,
                    "quote_ts": quote_ts,
                },
            )
            self.broker.close_trade(trade_id)

    def on_bar_closed(self, pair: str, bar: dict, epoch: int) -> None:
        del pair, bar, epoch


def _write_broker_snapshot(session_dir: Path, broker: VirtualBroker) -> None:
    tmp = session_dir / ".broker_snapshot.json.tmp"
    tmp.write_text(json.dumps(broker.snapshot(), ensure_ascii=False, sort_keys=True))
    os.replace(tmp, session_dir / "broker_snapshot.json")


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
    # Keep the resumable account checkpoint close to the atomically-written
    # runtime state.  A crash after a ledger mutation still fails closed via
    # ledger_sha instead of silently restoring an older account.
    _write_broker_snapshot(session_dir, broker)


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


def _seed_bot_from_history(args, bot, pairs) -> int:
    """Warm bot indicators from recent M1 history (no trading side effects)."""

    import gzip
    root = Path(args.seed_m1_root)
    cutoff = time_mod.time() - args.seed_hours * 3600
    seeded = 0
    for pair in pairs:
        rows = []
        for shard in sorted(root.glob(f"*/{pair}/{pair}_M1_BA_*.jsonl.gz")):
            with shard.open("rb") as raw:
                with gzip.open(raw, "rt", encoding="utf-8") as handle:
                    for line in handle:
                        row = json.loads(line)
                        epoch = int(datetime.fromisoformat(
                            row["time"][:19] + "+00:00").timestamp())
                        if epoch < cutoff:
                            continue
                        rows.append((epoch, row))
        rows.sort(key=lambda r: r[0])
        last = None
        for epoch, row in rows:
            if epoch == last:
                continue
            last = epoch
            b, a = row["bid"], row["ask"]
            bar = {"epoch": epoch,
                   "bid_o": float(b["o"]), "bid_h": float(b["h"]),
                   "bid_l": float(b["l"]), "bid_c": float(b["c"]),
                   "ask_o": float(a["o"]), "ask_h": float(a["h"]),
                   "ask_l": float(a["l"]), "ask_c": float(a["c"])}
            if hasattr(bot, "seed_bar"):
                bot.seed_bar(pair, bar)
                seeded += 1
    return seeded


def run_live(args, broker: VirtualBroker, session_dir: Path, bot=None) -> None:
    from quant_rabbit.broker.oanda import OandaReadOnlyClient

    client = OandaReadOnlyClient()
    pairs = args.pairs.split(",")
    if bot is not None and args.seed_m1_root and args.seed_hours > 0:
        n = _seed_bot_from_history(args, bot, pairs)
        broker._log("BOT_SEEDED", {"bars": n, "hours": args.seed_hours,
                                   "root": str(args.seed_m1_root)})
    window_start = (
        datetime.fromisoformat(args.window_start_utc).timestamp()
        if args.window_start_utc
        else None
    )
    deadline = (
        datetime.fromisoformat(args.window_end_utc).timestamp()
        if args.window_end_utc
        else time_mod.time() + args.minutes * 60.0
    )
    live_bars: dict[str, dict] = {}
    while time_mod.time() < deadline:
        if window_start is not None and time_mod.time() < window_start:
            now = datetime.now(UTC)
            _write_state(
                session_dir,
                broker,
                now.isoformat(),
                "live",
                "WAITING_FOR_PRECOMMITTED_WINDOW: no quote consumed",
            )
            time_mod.sleep(min(POLL_SECONDS, max(window_start - time_mod.time(), 0.1)))
            continue
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
            if args.drain_only:
                bot.on_quote(pair, q.timestamp_utc.isoformat())
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
        if not args.drain_only:
            _process_inbox(session_dir, broker)
        _write_state(session_dir, broker, now.isoformat(), "live")
        if args.drain_only and not broker.positions and not broker.orders:
            return
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
    parser.add_argument(
        "--window-start-utc",
        default=None,
        help="formal live: precommitted timezone-aware ISO window start",
    )
    parser.add_argument(
        "--window-end-utc",
        default=None,
        help="formal live: precommitted timezone-aware ISO window end",
    )
    parser.add_argument("--corpus-root", default=(
        "/Users/tossaki/App/QuantRabbit-live/logs/replay/oanda_history_m1_2020_2026"))
    parser.add_argument("--from", dest="time_from", default="2026-01-05T00:00:00")
    parser.add_argument("--to", dest="time_to", default="2026-01-10T00:00:00")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
        help="replay: sealed corpus manifest bound by SHA-256",
    )
    parser.add_argument("--bars-per-second", type=float, default=20.0)
    parser.add_argument("--step", action="store_true",
                        help="replay: advance one bar per inbox/STEP file")
    parser.add_argument("--bot", choices=["golden_burst", "golden_burst_blindspread"], default=None,
                        help="built-in worker bot (same broker/ledger)")
    parser.add_argument("--seed-m1-root", default=None,
                        help="live: warm bot indicators from this M1 corpus root")
    parser.add_argument("--seed-hours", type=float, default=0.0,
                        help="live: how many hours of history to seed")
    parser.add_argument("--bot-module", default=None,
                        help="path to ANY bot file: <file.py>[:ClassName]; the class "
                             "takes (broker) and implements on_bar_closed(pair, bar, epoch)")
    parser.add_argument("--bot-config-env", default=None,
                        help="JSON environment variable bound into the room contract")
    parser.add_argument(
        "--runtime-dependency",
        type=Path,
        action="append",
        default=[],
        help="additional launcher/registry file bound into the runtime manifest",
    )
    parser.add_argument("--granularity", choices=["M1", "S5"], default="M1",
                        help="replay feed granularity (S5 = 12x finer fill realism)")
    parser.add_argument("--bot-bar", choices=["feed", "M1"], default="feed",
                        help="bot decision cadence: per feed bar, or aggregated M1")
    parser.add_argument("--state-every", type=int, default=1,
                        help="replay: write state/process inbox every N bars (lab speed)")
    parser.add_argument("--fast-ledger", action="store_true",
                        help="ledger flush without fsync (lab runs)")
    parser.add_argument("--slippage-pips", type=float, default=None,
                        help="stress: extra pips against the trader on every fill")
    parser.add_argument("--financing-pips-day", type=float, default=None,
                        help="holding cost in pips per 24h held (pro-rata)")
    parser.add_argument("--leverage", type=float, default=None,
                        help="virtual account leverage; formal paper must set it explicitly")
    parser.add_argument("--paper-proof-mode", choices=["diagnostic", "formal"],
                        default="diagnostic")
    parser.add_argument("--room-kind",
                        choices=["diagnostic", "single_strategy", "integrated", "ai"],
                        default="diagnostic")
    parser.add_argument("--experiment-id", default="diagnostic-unregistered")
    parser.add_argument("--room-id", default="diagnostic-unregistered")
    parser.add_argument("--candidate-id", default="diagnostic-unregistered")
    parser.add_argument("--drain-only", action="store_true",
                        help="resume terminal paper exposure with new entries disabled")
    parser.add_argument("--drain-ceiling-min", type=int, default=None,
                        help="original per-position ceiling for drain-only mode")
    parser.add_argument("--allow-legacy-untagged", action="store_true",
                        help="diagnostic drain only; never proof-eligible")
    parser.add_argument("--intrabar", choices=["OHLC", "OLHC"], default="OHLC",
                        help="declared synthetic intrabar path; run both to bracket "
                             "both-touch ambiguity")
    args = parser.parse_args()

    session_dir = args.session_dir
    (session_dir / "inbox" / "processed").mkdir(parents=True, exist_ok=True)
    _acquire_runtime_lock(session_dir)
    if args.drain_only and args.feed != "live":
        parser.error("--drain-only requires --feed live")
    if args.drain_only and (args.bot or args.bot_module):
        parser.error("--drain-only forbids every bot")
    if args.drain_only and args.drain_ceiling_min is None:
        parser.error("--drain-only requires --drain-ceiling-min")
    if args.paper_proof_mode == "formal" and args.drain_only:
        parser.error("drain inherits proof status; do not set formal mode")
    if bool(args.window_start_utc) != bool(args.window_end_utc):
        parser.error("live paper window requires both --window-start-utc and --window-end-utc")
    if args.feed == "replay" and (args.window_start_utc or args.window_end_utc):
        parser.error("fixed live window arguments cannot be used with replay")
    if args.feed == "live" and args.source_manifest is not None:
        parser.error("--source-manifest is replay-only")
    if args.source_manifest is not None and not args.source_manifest.is_file():
        parser.error(f"source manifest is missing: {args.source_manifest}")

    ledger_path = session_dir / "ledger.jsonl"
    ledger_had_records = ledger_path.is_file() and ledger_path.stat().st_size > 0
    if args.window_start_utc:
        try:
            window_start = datetime.fromisoformat(args.window_start_utc)
            window_end = datetime.fromisoformat(args.window_end_utc)
        except ValueError as exc:
            parser.error(f"invalid live paper ISO window: {exc}")
        if window_start.tzinfo is None or window_end.tzinfo is None:
            parser.error("live paper window must be timezone-aware")
        if window_end <= window_start:
            parser.error("live paper window end must follow its start")
        now = datetime.now(UTC)
        if now >= window_end:
            parser.error("live paper window has already ended")
        if (
            args.paper_proof_mode == "formal"
            and not ledger_had_records
            and now.timestamp() > window_start.timestamp() + 60.0
        ):
            parser.error("fresh formal live paper started over 60s late")

    costs_explicit = (
        args.slippage_pips is not None
        and args.financing_pips_day is not None
        and args.leverage is not None
    )
    slippage_pips = float(args.slippage_pips or 0.0)
    financing_pips_day = float(args.financing_pips_day or 0.0)
    leverage = float(args.leverage or 25.0)
    pairs = [pair for pair in args.pairs.split(",") if pair]
    if not pairs or len(pairs) != len(set(pairs)):
        parser.error("--pairs must be a non-empty unique comma-separated list")

    raw_module_path = None
    if args.bot_module:
        raw_module_path = Path(args.bot_module.partition(":")[0])
        if not raw_module_path.is_absolute():
            raw_module_path = REPO_ROOT / raw_module_path
    runtime = runtime_manifest(
        repo_root=REPO_ROOT,
        runner_path=Path(__file__),
        bot_module_path=raw_module_path,
        extra_paths=[
            path if path.is_absolute() else REPO_ROOT / path
            for path in args.runtime_dependency
        ],
    )
    bot_spec = bot_contract(
        repo_root=REPO_ROOT,
        built_in_bot=args.bot,
        bot_module=args.bot_module,
        bot_config_env=args.bot_config_env,
    )

    broker = VirtualBroker(
        ledger_path=ledger_path,
        balance_jpy=args.balance,
        fast_ledger=args.fast_ledger,
        slippage_pips=slippage_pips,
        financing_pips_per_day=financing_pips_day,
        leverage=leverage,
    )
    snap_path = session_dir / "broker_snapshot.json"
    snapshot_recovery = None
    if snap_path.exists():
        snap = json.loads(snap_path.read_text())
        broker.restore(snap, require_ledger_match="ledger_sha" in snap)
        if snap.get("recovery"):
            snapshot_recovery = snap["recovery"]

    if args.drain_only:
        drain = prepare_drain_contract(
            session_dir=session_dir,
            pairs=pairs,
            ceiling_minutes=args.drain_ceiling_min,
            allow_legacy_untagged=args.allow_legacy_untagged,
            slippage_pips=slippage_pips,
            financing_pips_per_day=financing_pips_day,
            leverage=leverage,
            runtime=runtime,
        )
        if snapshot_recovery:
            broker._log("DRAIN_RECOVERY", snapshot_recovery)
        for order_id, order in list(broker.orders.items()):
            broker._log(
                "DRAIN_PENDING_ORDER_CANCEL",
                {
                    "order_id": order_id,
                    "pair": order.pair,
                    "strategy_tag": order.strategy_tag,
                    "drain_contract_sha256": drain["drain_contract_sha256"],
                },
            )
            broker.cancel_order(order_id)
        broker._log(
            "DRAIN_START",
            {
                "contract": drain["contract"],
                "drain_contract_sha256": drain["drain_contract_sha256"],
                "source_terminal_sha256": drain["source_terminal_sha256"],
                "proof_eligible": drain["proof_eligible"],
                "positions": len(broker.positions),
                "orders": len(broker.orders),
                "order_authority": "NONE",
            },
        )
        controller = DrainOnlyController(broker, args.drain_ceiling_min)
        try:
            if broker.positions:
                run_live(args, broker, session_dir, bot=controller)
        finally:
            sealed = not broker.positions and not broker.orders
            broker._log(
                "DRAIN_STOP",
                {
                    "drain_contract_sha256": drain["drain_contract_sha256"],
                    "status": "SEALED" if sealed else "DRAIN_INCOMPLETE",
                    "positions": len(broker.positions),
                    "orders": len(broker.orders),
                    "account": broker.account() if broker.last_quotes else None,
                },
            )
            _write_broker_snapshot(session_dir, broker)
        print(
            json.dumps(
                {
                    "status": "DRAIN_SEALED" if sealed else "DRAIN_INCOMPLETE",
                    "positions": len(broker.positions),
                    "orders": len(broker.orders),
                    "drain_contract_sha256": drain["drain_contract_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0 if sealed else 75

    session_contract_path = session_dir / "session_contract.json"
    if ledger_had_records and not session_contract_path.exists():
        raise DojoPaperContractError(
            "normal session cannot bootstrap a contract over a legacy ledger; "
            "use the drain-only boundary"
        )
    source = {
        "kind": "live_read_only_pricing" if args.feed == "live" else "sealed_replay",
        "poll_seconds": POLL_SECONDS if args.feed == "live" else None,
        "stale_quote_max_seconds": STALE_QUOTE_MAX_S if args.feed == "live" else None,
        "window_start_utc": args.window_start_utc if args.feed == "live" else None,
        "window_end_utc": args.window_end_utc if args.feed == "live" else None,
        "corpus_root": str(Path(args.corpus_root).resolve()) if args.feed == "replay" else None,
        "source_manifest_path": (
            str(args.source_manifest.resolve()) if args.source_manifest else None
        ),
        "source_manifest_sha256": (
            file_sha256(args.source_manifest) if args.source_manifest else None
        ),
        "time_from": args.time_from if args.feed == "replay" else None,
        "time_to": args.time_to if args.feed == "replay" else None,
        "granularity": args.granularity if args.feed == "replay" else "live_poll",
        "intrabar": args.intrabar if args.feed == "replay" else None,
    }
    session_contract = build_session_contract(
        experiment_id=args.experiment_id,
        room_id=args.room_id,
        candidate_id=args.candidate_id,
        room_kind=args.room_kind,
        proof_mode=args.paper_proof_mode,
        feed=args.feed,
        pairs=pairs,
        initial_balance_jpy=args.balance,
        slippage_pips=slippage_pips,
        financing_pips_per_day=financing_pips_day,
        leverage=leverage,
        costs_explicit=costs_explicit,
        runtime=runtime,
        bot=bot_spec,
        source=source,
    )
    publish_immutable_json(session_contract_path, session_contract)
    if snapshot_recovery:
        broker._log("SESSION_RECOVERY", snapshot_recovery)
    broker._log(
        "SESSION_START",
        {
            "contract": session_contract["contract"],
            "session_contract_sha256": session_contract["session_contract_sha256"],
            "runtime_manifest_sha256": runtime["manifest_sha256"],
            "bot_contract_sha256": bot_spec["bot_contract_sha256"],
            "experiment_id": args.experiment_id,
            "room_id": args.room_id,
            "candidate_id": args.candidate_id,
            "proof_mode": args.paper_proof_mode,
            "proof_eligible": session_contract["proof_eligible"],
            "feed": args.feed,
            "pairs": pairs,
            "balance": broker.balance_jpy,
            "costs": session_contract["costs"],
            "order_authority": "NONE",
        },
    )
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
        broker._log(
            "BOT_LOADED",
            {
                "module": module_path,
                "class": class_name or "Bot",
                "bot_contract_sha256": bot_spec["bot_contract_sha256"],
                "module_sha256": bot_spec["module_sha256"],
                "config_sha256": bot_spec["config_sha256"],
                "strategy_tags": bot_spec["strategy_tags"],
            },
        )
    try:
        if args.feed == "live":
            run_live(args, broker, session_dir, bot=bot)
        else:
            run_replay(args, broker, session_dir, bot=bot)
    finally:
        broker._log(
            "SESSION_STOP",
            {
                "account": broker.account() if broker.last_quotes else None,
                "session_contract_sha256": session_contract[
                    "session_contract_sha256"
                ],
                "proof_eligible": session_contract["proof_eligible"],
            },
        )
        _write_broker_snapshot(session_dir, broker)
    print(json.dumps({"status": "SESSION_DONE",
                      "account": broker.account() if broker.last_quotes else None},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
