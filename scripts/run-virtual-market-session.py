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
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time as time_mod
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quant_rabbit.analysis.market_status import compute_market_status
from quant_rabbit.virtual_broker import VirtualBroker, VirtualBrokerError


PHASE_ORDERS = {
    "OHLC": {"O": 0, "H": 1, "L": 2, "C": 3},
    "OLHC": {"O": 0, "L": 1, "H": 2, "C": 3},
}


def _reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_json_loads(value: str):
    return json.loads(value, parse_constant=_reject_json_constant)


def _finite_number(name: str, value, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _normalized_pairs(raw: str) -> list[str]:
    pairs = [pair.strip() for pair in raw.split(",") if pair.strip()]
    if not pairs:
        raise ValueError("at least one pair is required")
    if len(pairs) != len(set(pairs)):
        raise ValueError("duplicate pairs are forbidden")
    for pair in pairs:
        parts = pair.split("_")
        if (
            len(parts) != 2
            or any(
                len(part) != 3 or not part.isalpha() or not part.isupper()
                for part in parts
            )
        ):
            raise ValueError(f"invalid pair: {pair}")
    return sorted(pairs)


def _parse_utc_bound(value: str) -> datetime:
    text = str(value).strip()
    match = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:\d{2})?",
        text,
    )
    if match:
        head, fraction, zone = match.groups()
        text = head
        if fraction:
            text += "." + fraction[:6]
        text += "+00:00" if zone in {None, "Z"} else zone
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"invalid UTC timestamp: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    else:
        parsed = parsed.astimezone(UTC)
    return parsed


def _validate_time_window(time_from: str, time_to: str) -> tuple[datetime, datetime]:
    start = _parse_utc_bound(time_from)
    end = _parse_utc_bound(time_to)
    if start >= end:
        raise ValueError("--from must be earlier than --to")
    return start, end


def _parse_corpus_time(value: object) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("corpus row time is missing")
    return _parse_utc_bound(text)


def _canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    head = result.stdout.strip()
    if len(head) != 40 or any(ch not in "0123456789abcdef" for ch in head):
        raise RuntimeError("unable to bind a valid git HEAD")
    return head


def _selected_corpus_shards(
    root: Path,
    pairs: list[str],
    time_from: str,
    time_to: str,
    granularity: str,
) -> list[Path]:
    start, end = _validate_time_window(time_from, time_to)
    shards: set[Path] = set()
    for year in range(start.year, end.year + 1):
        for pair in pairs:
            shards.update(
                path.resolve()
                for path in root.glob(
                    f"*/{pair}/{pair}_{granularity}_BA_{year}*.jsonl.gz"
                )
            )
    return sorted(shards)


def _validate_replay_row(row: object, pair: str, *, expected_year: int) -> datetime:
    if not isinstance(row, dict):
        raise ValueError(f"corpus row for {pair} must be an object")
    stamp = _parse_corpus_time(row.get("time"))
    if stamp.year != expected_year:
        raise ValueError(f"corpus row year/file mismatch for {pair}")
    for side in ("bid", "ask"):
        values = row.get(side)
        if not isinstance(values, dict) or set(values) < {"o", "h", "l", "c"}:
            raise ValueError(f"corpus {side} OHLC is incomplete for {pair}")
        parsed = {
            key: _finite_number(f"{pair} {side}.{key}", values[key], positive=True)
            for key in ("o", "h", "l", "c")
        }
        if parsed["h"] < max(parsed["o"], parsed["l"], parsed["c"]):
            raise ValueError(f"corpus {side} high geometry is invalid for {pair}")
        if parsed["l"] > min(parsed["o"], parsed["h"], parsed["c"]):
            raise ValueError(f"corpus {side} low geometry is invalid for {pair}")
    for key in ("o", "h", "l", "c"):
        if float(row["ask"][key]) < float(row["bid"][key]):
            raise ValueError(f"corpus ask is below bid for {pair} at {key}")
    return stamp


def _validate_corpus_pair_coverage(
    root: Path,
    pairs: list[str],
    time_from: str,
    time_to: str,
    granularity: str,
    shards: list[Path],
) -> None:
    start, end = _validate_time_window(time_from, time_to)
    by_pair: dict[str, list[Path]] = {pair: [] for pair in pairs}
    for path in shards:
        for pair in pairs:
            if path.parent.name == pair:
                by_pair[pair].append(path)
                break
    missing_shards = [pair for pair, paths in by_pair.items() if not paths]
    if missing_shards:
        raise ValueError(
            "replay corpus missing requested pair shards: " + ",".join(missing_shards)
        )
    covered: set[str] = set()
    for pair, paths in by_pair.items():
        for shard in paths:
            try:
                expected_year = int(shard.name.split(f"_{granularity}_BA_", 1)[1][:4])
            except (IndexError, ValueError) as exc:
                raise ValueError(f"invalid corpus shard name: {shard.name}") from exc
            with gzip.open(shard, "rt", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    try:
                        row = _strict_json_loads(line)
                        stamp = _validate_replay_row(
                            row, pair, expected_year=expected_year
                        )
                    except (json.JSONDecodeError, ValueError) as exc:
                        raise ValueError(
                            f"invalid corpus row {shard.name}:{line_number}: {exc}"
                        ) from exc
                    if start <= stamp < end:
                        covered.add(pair)
                        break
            if pair in covered:
                break
    missing_period = [pair for pair in pairs if pair not in covered]
    if missing_period:
        raise ValueError(
            "replay corpus has no rows in requested period for: "
            + ",".join(missing_period)
        )


def _build_reproducibility_manifest(args) -> dict:
    """Bind every material replay, cost, source, and worker input.

    The corpus and code digests make a SESSION_START self-contained proof of
    which bytes were eligible to influence the run.  Known bot-configuration
    environment values are digest-bound without leaking their contents.
    """

    pairs = _normalized_pairs(args.pairs)
    _validate_time_window(args.time_from, args.time_to)
    _finite_number("balance", args.balance, positive=True)
    _finite_number("slippage_pips", args.slippage_pips)
    _finite_number("financing_pips_day", args.financing_pips_day)
    if args.slippage_pips < 0 or args.financing_pips_day < 0:
        raise ValueError("cost parameters must be non-negative")
    root = Path(args.corpus_root).expanduser().resolve()
    shards = (
        _selected_corpus_shards(
            root, pairs, args.time_from, args.time_to, args.granularity
        )
        if args.feed == "replay"
        else []
    )
    if args.feed == "replay" and not shards:
        raise ValueError("replay corpus contains no selected shards")
    if args.feed == "replay":
        _validate_corpus_pair_coverage(
            root,
            pairs,
            args.time_from,
            args.time_to,
            args.granularity,
            shards,
        )
    shard_rows = [
        {
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in shards
    ]

    if args.bot and args.bot_module:
        raise ValueError("--bot and --bot-module are mutually exclusive")
    if args.bot_module:
        module_path_raw, _, class_name = args.bot_module.partition(":")
        module_path = Path(module_path_raw).expanduser().resolve()
        if not module_path.is_file():
            raise ValueError(f"bot module is not a file: {module_path}")
        bot_manifest = {
            "kind": "custom_module",
            "name": None,
            "module_path": str(module_path),
            "module_sha256": _file_sha256(module_path),
            "class": class_name or "Bot",
        }
    elif args.bot:
        vendor = REPO_ROOT / "vendored" / "golden_20251209"
        dependency_paths = [
            vendor / "analysis_ma_projection.py",
            vendor / "indicators_calc_core.py",
            vendor / "strategies_micro_momentum_burst.py",
        ]
        bot_manifest = {
            "kind": "built_in",
            "name": args.bot,
            "module_path": str(Path(__file__).resolve()),
            "module_sha256": _file_sha256(Path(__file__).resolve()),
            "class": "GoldenBurstBot",
            "dependency_sha256": {
                str(path.relative_to(REPO_ROOT)): _file_sha256(path)
                for path in dependency_paths
            },
        }
    else:
        bot_manifest = {
            "kind": "none",
            "name": None,
            "module_path": None,
            "module_sha256": None,
            "class": None,
            "dependency_sha256": {},
        }
    if "dependency_sha256" not in bot_manifest:
        bot_manifest["dependency_sha256"] = {}
    bot_manifest["configuration_bindings"] = {
        key: {
            "sha256": hashlib.sha256(os.environ[key].encode("utf-8")).hexdigest(),
            "length": len(os.environ[key]),
        }
        for key in ("DOJO_BOT_CONFIG", "DOJO_BOT_COMBO")
        if key in os.environ
    }
    session_dir = getattr(args, "session_dir", None)
    snapshot_path = Path(session_dir) / "broker_snapshot.json" if session_dir else None
    resume_snapshot = (
        {
            "path": str(snapshot_path.resolve()),
            "size_bytes": snapshot_path.stat().st_size,
            "sha256": _file_sha256(snapshot_path),
        }
        if snapshot_path is not None and snapshot_path.is_file()
        else None
    )

    corpus_manifest = {
        "root": str(root) if args.feed == "replay" else None,
        "shards": shard_rows,
    }
    corpus_manifest["corpus_sha256"] = _canonical_sha256(corpus_manifest)
    manifest = {
        "schema": "QR_VIRTUAL_SESSION_REPRODUCIBILITY_V1",
        "source": {
            "git_head": _git_head(),
            "session_script_sha256": _file_sha256(Path(__file__).resolve()),
            "virtual_broker_sha256": _file_sha256(
                REPO_ROOT / "src" / "quant_rabbit" / "virtual_broker.py"
            ),
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": sys.version,
        },
        "replay": {
            "feed": args.feed,
            "pairs": pairs,
            "time_from": args.time_from,
            "time_to": args.time_to,
            "granularity": args.granularity,
            "intrabar": args.intrabar,
            "bot_bar": args.bot_bar,
        },
        "corpus": corpus_manifest,
        "costs": {
            "slippage_pips_per_fill": args.slippage_pips,
            "financing_pips_per_day": args.financing_pips_day,
            "leverage": 25.0,
        },
        "initial_balance_jpy": args.balance,
        "resume_snapshot": resume_snapshot,
        "bot": bot_manifest,
        "pacing": {
            "bars_per_second": getattr(args, "bars_per_second", None),
            "step": getattr(args, "step", False),
            "state_every": getattr(args, "state_every", 1),
        },
        "order_authority": "NONE",
    }
    return {**manifest, "manifest_sha256": _canonical_sha256(manifest)}


def _replay_identity_sha256(manifest: dict) -> str:
    """Material identity that a replay cursor is allowed to resume."""

    material = {
        "source": manifest["source"],
        "replay": manifest["replay"],
        "corpus": manifest["corpus"],
        "costs": manifest["costs"],
        "initial_balance_jpy": manifest["initial_balance_jpy"],
        "bot": manifest["bot"],
        "order_authority": manifest["order_authority"],
    }
    return _canonical_sha256(material)


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


def _process_inbox(
    session_dir: Path,
    broker: VirtualBroker,
    *,
    allowed_pairs: set[str] | None = None,
) -> int:
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
            action = _strict_json_loads(path.read_text())
            if not isinstance(action, dict):
                raise ValueError("action must be a JSON object")
            kind = action.get("action")
            action_pair = action.get("pair")
            if (
                action_pair is not None
                and allowed_pairs is not None
                and action_pair not in allowed_pairs
            ):
                raise ValueError(f"pair is not active in this session: {action_pair}")
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
                                   units=float(units) if units is not None else None)
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
    pairs = _normalized_pairs(args.pairs)
    exposure_pairs = {
        position.pair for position in broker.positions.values()
    } | {order.pair for order in broker.orders.values()}
    if not exposure_pairs.issubset(set(pairs)):
        raise VirtualBrokerError(
            "live session pairs do not cover restored exposure: "
            + ",".join(sorted(exposure_pairs - set(pairs)))
        )
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
        missing = sorted(set(pairs) - set(quotes))
        unexpected = sorted(set(quotes) - set(pairs))
        invalid_freshness: list[str] = []
        for pair, q in quotes.items():
            age_s = (now - q.timestamp_utc).total_seconds()
            prior = broker.last_quotes.get(pair)
            prior_epoch = broker._ts_epoch(prior[2]) if prior is not None else None
            quote_epoch = q.timestamp_utc.timestamp()
            if age_s < -1.0 or age_s > STALE_QUOTE_MAX_S:
                invalid_freshness.append(f"{pair}:age={age_s:.3f}")
            elif prior_epoch is not None and quote_epoch < prior_epoch:
                invalid_freshness.append(f"{pair}:timestamp_rewind")
        if missing or unexpected or invalid_freshness:
            detail = {
                "missing_pairs": missing,
                "unexpected_pairs": unexpected,
                "invalid_freshness": invalid_freshness,
            }
            broker._log("QUOTE_BATCH_REJECTED", detail)
            _write_state(session_dir, broker, now.isoformat(), "live",
                         "INCOMPLETE_OR_STALE_QUOTES: refusing actions")
            time_mod.sleep(POLL_SECONDS)
            continue
        broker.on_quote_batch([
            (pair, quotes[pair].bid, quotes[pair].ask, quotes[pair].timestamp_utc.isoformat())
            for pair in pairs
        ])
        for pair in pairs:
            q = quotes[pair]
            minute = int(q.timestamp_utc.timestamp() // 60) * 60
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
        _process_inbox(session_dir, broker, allowed_pairs=set(pairs))
        _write_state(session_dir, broker, now.isoformat(), "live")
        time_mod.sleep(POLL_SECONDS)


def _iter_replay_quotes(root: Path, pairs: list[str], time_from: str,
                        time_to: str, intrabar: str = "OHLC",
                        granularity: str = "M1", expected_shards=None):
    """Merge pairs' M1 bars in time order; yield (epoch, pair, bid, ask, phase).

    Streams year by year so full-history sessions stay in bounded memory.
    ``intrabar`` declares the synthetic intrabar path (the true tick path
    is unknown): OHLC favors longs' TP on both-touch bars, OLHC favors
    shorts'/SL — run both to bracket ambiguous outcomes.
    """

    root = root.expanduser().resolve()
    pairs = sorted(pairs)
    if len(pairs) != len(set(pairs)) or not pairs:
        raise ValueError("replay pairs must be unique and non-empty")
    start, end = _validate_time_window(time_from, time_to)
    phase_keys = {"OHLC": (("O", "o"), ("H", "h"), ("L", "l"), ("C", "c")),
                  "OLHC": (("O", "o"), ("L", "l"), ("H", "h"), ("C", "c"))}[intrabar]
    discovered = _selected_corpus_shards(
        root, pairs, time_from, time_to, granularity
    )
    expected_by_path: dict[Path, dict] = {}
    if expected_shards is not None:
        for row in expected_shards:
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                raise ValueError("invalid expected corpus shard seal")
            path = (root / row["path"]).resolve()
            if root not in path.parents:
                raise ValueError("corpus shard escapes corpus root")
            expected_by_path[path] = row
        if set(discovered) != set(expected_by_path):
            raise ValueError("corpus shard set changed after manifest sealing")
        for path, row in expected_by_path.items():
            if path.stat().st_size != int(row["size_bytes"]):
                raise ValueError(f"corpus shard size changed: {path.name}")
    eligible_pairs: set[str] = set()
    for year in range(start.year, end.year + 1):
        rows = []
        for pair in pairs:
            pair_shards = [
                shard
                for shard in discovered
                if shard.parent.name == pair
                and f"_{granularity}_BA_{year}" in shard.name
            ]
            for shard in pair_shards:
                with gzip.open(shard, "rt", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        try:
                            row = _strict_json_loads(line)
                            stamp = _validate_replay_row(
                                row, pair, expected_year=year
                            )
                        except (json.JSONDecodeError, ValueError) as exc:
                            raise ValueError(
                                f"invalid corpus row {shard.name}:{line_number}: {exc}"
                            ) from exc
                        if not (start <= stamp < end):
                            continue
                        eligible_pairs.add(pair)
                        epoch = int(stamp.timestamp())
                        rows.append((epoch, pair, row))
                expected = expected_by_path.get(shard)
                if expected is not None and _file_sha256(shard) != expected["sha256"]:
                    raise ValueError(f"corpus shard changed during replay: {shard.name}")
        rows.sort(key=lambda r: (r[0], r[1]))
        cursor = 0
        while cursor < len(rows):
            epoch = rows[cursor][0]
            epoch_rows: dict[str, dict] = {}
            while cursor < len(rows) and rows[cursor][0] == epoch:
                _, pair, row = rows[cursor]
                if pair in epoch_rows:
                    raise ValueError(
                        f"duplicate corpus row for {pair} at epoch {epoch}"
                    )
                epoch_rows[pair] = row
                cursor += 1
            # Synchronize every pair at O before any H/L/C, then repeat for
            # each phase.  This prevents one pair's close from leaking into a
            # different pair's open at the same epoch.
            for phase, key in phase_keys:
                for pair in sorted(epoch_rows):
                    row = epoch_rows[pair]
                    bid_values, ask_values = row["bid"], row["ask"]
                    yield (
                        epoch,
                        pair,
                        float(bid_values[key]),
                        float(ask_values[key]),
                        phase,
                    )
    missing = [pair for pair in pairs if pair not in eligible_pairs]
    if missing:
        raise ValueError(
            "replay corpus has no eligible rows for: " + ",".join(missing)
        )


def _iter_replay_quote_batches(*args, **kwargs):
    batch_key = None
    batch: list[tuple[str, float, float, str]] = []
    for epoch, pair, bid, ask, phase in _iter_replay_quotes(*args, **kwargs):
        key = (epoch, phase)
        if batch_key is not None and key != batch_key:
            yield batch_key[0], batch_key[1], batch
            batch = []
        batch_key = key
        ts = datetime.fromtimestamp(epoch, tz=UTC).isoformat() + f"#{phase}"
        batch.append((pair, bid, ask, ts))
    if batch_key is not None:
        yield batch_key[0], batch_key[1], batch


def _cursor_key(epoch: int, phase: str, intrabar: str) -> tuple[int, int]:
    try:
        return epoch, PHASE_ORDERS[intrabar][phase]
    except KeyError as exc:
        raise ValueError("invalid replay cursor phase") from exc


def run_replay(
    args,
    broker: VirtualBroker,
    session_dir: Path,
    bot=None,
    *,
    replay_identity_sha256: str | None = None,
    expected_shards=None,
) -> None:
    pairs = _normalized_pairs(args.pairs)
    root = Path(args.corpus_root).expanduser().resolve()
    bars_per_second = _finite_number(
        "bars_per_second", args.bars_per_second, positive=True
    )
    state_every_value = _finite_number("state_every", args.state_every, positive=True)
    if not state_every_value.is_integer():
        raise ValueError("state_every must be an integer")
    state_every = int(state_every_value)
    bar_sleep = 1.0 / bars_per_second
    no_sleep = bars_per_second >= 10000
    step_file = session_dir / "inbox" / "STEP"
    current_epoch = None
    pending_bars: dict[str, dict] = {}
    bot_minute: dict[str, dict] = {}
    aggregate_bot_bars = (
        bot is not None and args.bot_bar == "M1" and args.granularity != "M1"
    )
    cursor = broker.feed_cursor
    resume_key: tuple[int, int] | None = None
    bar_count = 0
    if cursor is not None:
        if cursor.get("mode") != "replay":
            raise VirtualBrokerError("snapshot feed mode is not replay")
        if replay_identity_sha256 is None or cursor.get(
            "replay_identity_sha256"
        ) != replay_identity_sha256:
            raise VirtualBrokerError("replay resume identity mismatch")
        if bot is not None:
            raise VirtualBrokerError("bot replay resume requires persisted bot state")
        epoch_value = _finite_number("resume cursor epoch", cursor.get("epoch"))
        if not epoch_value.is_integer():
            raise VirtualBrokerError("resume cursor epoch must be an integer")
        resume_key = _cursor_key(int(epoch_value), str(cursor.get("phase")), args.intrabar)
        current_epoch = resume_key[0]
        bar_count_value = _finite_number(
            "resume cursor bar_count", cursor.get("bar_count", 0)
        )
        if not bar_count_value.is_integer() or bar_count_value < 0:
            raise VirtualBrokerError("resume cursor bar_count must be non-negative integer")
        bar_count = int(bar_count_value)
    elif broker.positions or broker.orders:
        raise VirtualBrokerError(
            "replay snapshot with exposure has no causal feed cursor"
        )

    bar_seconds = 5 if args.granularity == "S5" else 60
    processed_any = False
    for epoch, phase, quote_batch in _iter_replay_quote_batches(
        root,
        pairs,
        args.time_from,
        args.time_to,
        args.intrabar,
        args.granularity,
        expected_shards=expected_shards,
    ):
        key = _cursor_key(epoch, phase, args.intrabar)
        if resume_key is not None and key <= resume_key:
            continue
        boundary = phase == "O" and epoch != current_epoch
        publish_state = False
        closed_bars: list[tuple[str, dict]] = []
        closed_minutes: list[tuple[str, dict]] = []
        if boundary and current_epoch is not None:
            bar_count += 1
            publish_state = bar_count % state_every == 0
            if publish_state or args.step:
                _write_state(
                    session_dir,
                    broker,
                    datetime.fromtimestamp(
                        current_epoch + bar_seconds, tz=UTC
                    ).isoformat(),
                    "replay",
                )
            if args.step:
                while not step_file.exists():
                    time_mod.sleep(0.2)
                os.replace(
                    step_file,
                    session_dir / "inbox" / "processed" /
                    f"{int(time_mod.time()*1000)}_STEP",
                )
            elif not no_sleep:
                time_mod.sleep(bar_sleep)
            if bot is not None and not aggregate_bot_bars:
                closed_bars = [
                    (bpair, bbar)
                    for bpair, bbar in pending_bars.items()
                    if bbar["epoch"] == current_epoch
                ]
            if aggregate_bot_bars:
                minute = epoch // 60 * 60
                closed_minutes = [
                    (bpair, bbar)
                    for bpair, bbar in bot_minute.items()
                    if bbar["epoch"] != minute
                ]

        # Existing orders/exits see the next executable phase first.  Only
        # after the new O is staged may a closed-bar decision submit MARKET.
        broker.on_quote_batch(quote_batch)
        processed_any = True
        broker.feed_cursor = {
            "mode": "replay",
            "epoch": epoch,
            "phase": phase,
            "bar_count": bar_count,
            "completed": False,
            "replay_identity_sha256": replay_identity_sha256,
        }
        if boundary:
            for bpair, bbar in closed_bars:
                bot.on_bar_closed(bpair, bbar, bbar["epoch"])
            for bpair, bbar in closed_minutes:
                bot.on_bar_closed(bpair, bbar, bbar["epoch"])
                del bot_minute[bpair]
            if publish_state or args.step:
                _process_inbox(session_dir, broker, allowed_pairs=set(pairs))
            current_epoch = epoch

        for pair, bid, ask, _ in quote_batch:
            if phase == "O":
                pending_bars[pair] = {
                    "bid_o": bid, "ask_o": ask,
                    "bid_h": bid, "bid_l": bid,
                    "ask_h": ask, "ask_l": ask,
                    "bid_c": bid, "ask_c": ask,
                    "epoch": epoch,
                }
            else:
                pb = pending_bars.get(pair)
                if pb is not None:
                    pb["bid_h"] = max(pb["bid_h"], bid)
                    pb["bid_l"] = min(pb["bid_l"], bid)
                    pb["ask_h"] = max(pb["ask_h"], ask)
                    pb["ask_l"] = min(pb["ask_l"], ask)
                    pb["bid_c"] = bid
                    pb["ask_c"] = ask
            if aggregate_bot_bars:
                minute = epoch // 60 * 60
                mb = bot_minute.get(pair)
                if mb is None:
                    bot_minute[pair] = {
                        "epoch": minute,
                        "bid_o": bid, "bid_h": bid, "bid_l": bid, "bid_c": bid,
                        "ask_o": ask, "ask_h": ask, "ask_l": ask, "ask_c": ask,
                    }
                else:
                    mb["bid_h"] = max(mb["bid_h"], bid)
                    mb["bid_l"] = min(mb["bid_l"], bid)
                    mb["ask_h"] = max(mb["ask_h"], ask)
                    mb["ask_l"] = min(mb["ask_l"], ask)
                    mb["bid_c"] = bid
                    mb["ask_c"] = ask

    if processed_any and broker.feed_cursor is not None:
        broker.feed_cursor["completed"] = True
    elif cursor is not None:
        broker.feed_cursor = dict(cursor)
        broker.feed_cursor["completed"] = True
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
    bot_group = parser.add_mutually_exclusive_group()
    bot_group.add_argument(
        "--bot",
        choices=["golden_burst", "golden_burst_blindspread"],
        default=None,
        help="built-in worker bot (same broker/ledger)",
    )
    bot_group.add_argument(
        "--bot-module",
        default=None,
        help="path to ANY bot file: <file.py>[:ClassName]; the class "
        "takes (broker) and implements on_bar_closed(pair, bar, epoch)",
    )
    parser.add_argument("--granularity", choices=["M1", "S5"], default="M1",
                        help="replay feed granularity (S5 = 12x finer fill realism)")
    parser.add_argument("--bot-bar", choices=["feed", "M1"], default="feed",
                        help="bot decision cadence: per feed bar, or aggregated M1")
    parser.add_argument("--state-every", type=int, default=1,
                        help="replay: write state/process inbox every N bars (lab speed)")
    parser.add_argument("--fast-ledger", action="store_true",
                        help="ledger flush without fsync (lab runs)")
    parser.add_argument("--slippage-pips", type=float, default=0.0,
                        help="stress: extra pips against the trader on every fill")
    parser.add_argument("--financing-pips-day", type=float, default=0.0,
                        help="holding cost in pips per 24h held (pro-rata)")
    parser.add_argument("--intrabar", choices=["OHLC", "OLHC"], default="OHLC",
                        help="declared synthetic intrabar path; run both to bracket "
                             "both-touch ambiguity")
    args = parser.parse_args()

    session_dir = args.session_dir
    (session_dir / "inbox" / "processed").mkdir(parents=True, exist_ok=True)
    broker = VirtualBroker(
        ledger_path=session_dir / "ledger.jsonl", balance_jpy=args.balance,
        fast_ledger=args.fast_ledger, slippage_pips=args.slippage_pips,
        financing_pips_per_day=args.financing_pips_day)
    snap_path = session_dir / "broker_snapshot.json"
    if snap_path.exists():
        broker.restore(_strict_json_loads(snap_path.read_text()))
    reproducibility_manifest = _build_reproducibility_manifest(args)
    replay_identity = (
        _replay_identity_sha256(reproducibility_manifest)
        if args.feed == "replay"
        else None
    )
    broker._log("SESSION_START", {
        "contract": "QR_VIRTUAL_MARKET_SESSION_V1",
        "feed": args.feed, "pairs": args.pairs, "balance": broker.balance_jpy,
        "order_authority": "NONE",
        "reproducibility_manifest": reproducibility_manifest,
        "reproducibility_manifest_sha256": reproducibility_manifest[
            "manifest_sha256"
        ],
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
        module_path = str(Path(module_path).expanduser().resolve())
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
            run_replay(
                args,
                broker,
                session_dir,
                bot=bot,
                replay_identity_sha256=replay_identity,
                expected_shards=reproducibility_manifest["corpus"]["shards"],
            )
    finally:
        try:
            stop_account = broker.account() if broker.last_quotes else None
            stop_error = None
        except VirtualBrokerError as exc:
            stop_account = None
            stop_error = str(exc)[:200]
        broker._log(
            "SESSION_STOP", {"account": stop_account, "account_error": stop_error}
        )
        tmp = session_dir / ".broker_snapshot.json.tmp"
        tmp.write_text(
            json.dumps(
                broker.snapshot(), ensure_ascii=False, sort_keys=True, allow_nan=False
            )
        )
        os.replace(tmp, snap_path)
    print(json.dumps({"status": "SESSION_DONE",
                      "account": broker.account() if broker.last_quotes else None},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
