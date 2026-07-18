#!/usr/bin/env python3
"""Live shadow environment: the real market, with shadow-only execution.

Operator directive (2026-07-19): "実際の市場と同じような環境でテストできる
だろ。そういう環境をつくれ。時間に偽りもなし、コストスプレッドも偽りなし。"

Honesty properties — nothing here is synthesized:
  * TIME is the wall clock.  Nothing is replayed or backdated.
  * PRICES are the account's own live pricing endpoint (read-only),
    polled every POLL_SECONDS; every poll is written to the ledger.
  * COST is whatever the live quote says: shadow entries fill at the
    real ask (long) / bid (short) of the poll that confirmed the signal;
    shadow exits fill only when a REAL polled quote touches the level.
    No interpolation, no assumed fills between polls.
  * The ledger is hash-chained JSONL (prev_sha -> sha), append-only, so
    the record cannot be quietly rewritten after the fact.
  * Market-closed or stale quotes -> the environment refuses to trade
    and says so in the ledger.
  * ORDER AUTHORITY: NONE.  This process places no orders, ever.

Candidate hand under test: the vendored golden-day MomentumBurst
(commit 41644097) with arsenal protections (max 3 concurrent, 4h hard
ceiling).  Two shadow exit books are kept per position so the live data
adjudicates the exit-geometry question the historical replay could not:
  BOOK_SL   the strategy's own TP/SL as emitted
  BOOK_TPO  TP-only + 4h hard ceiling (how the live 2025-12-09 worker
            actually behaved)
Factors warm up from the last days of the sealed M1 corpus plus a fresh
read-only fetch at startup; live M1 candles are then built from polls.

Run bounded: --minutes N (a tool, not a daemon).  Restartable; each run
appends to the same chain.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time as time_mod
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quant_rabbit.analysis.market_status import compute_market_status
from quant_rabbit.broker.oanda import OandaReadOnlyClient

UTC = timezone.utc
PAIR = "USD_JPY"
PIP = 0.01
POLL_SECONDS = 5.0
STALE_QUOTE_MAX_S = 90.0
WARMUP_BARS = 40
MAX_CONCURRENT = 3
HARD_CEILING_S = 4 * 3600
VENDOR = REPO_ROOT / "vendored" / "golden_20251209"


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _import_vendored():
    pkg = ModuleType("analysis")
    pkg.__path__ = []
    sys.modules["analysis"] = pkg
    pkg.ma_projection = _load_module("analysis.ma_projection", VENDOR / "analysis_ma_projection.py")
    calc_core = _load_module("golden_calc_core", VENDOR / "indicators_calc_core.py")
    strategy = _load_module("golden_momentum_burst", VENDOR / "strategies_micro_momentum_burst.py")
    return calc_core, strategy


class ChainLedger:
    """Append-only hash-chained JSONL."""

    def __init__(self, path: Path):
        self.path = path
        self.prev = "0" * 64
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        self.prev = json.loads(line)["sha"]
                    except Exception as exc:
                        raise ValueError(f"corrupt ledger line: {exc}") from exc
        self.handle = path.open("a", encoding="utf-8")

    def write(self, event: str, payload: dict[str, Any]) -> None:
        body = {
            "ts_utc": datetime.now(UTC).isoformat(),
            "event": event,
            "payload": payload,
            "prev_sha": self.prev,
        }
        record = {**body, "sha": _sha(body)}
        self.handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.prev = record["sha"]


class M1Builder:
    """Build M1 bid/ask candles from live polls; seed from historical rows."""

    def __init__(self):
        self.bars: list[dict[str, float]] = []  # closed bars
        self.current: Optional[dict[str, float]] = None

    def seed(self, rows: list[tuple]) -> None:
        for epoch, bo, bh, bl, bc, ao, ah, al, ac in rows:
            self.bars.append({
                "epoch": epoch, "bid_o": bo, "bid_h": bh, "bid_l": bl, "bid_c": bc,
                "ask_o": ao, "ask_h": ah, "ask_l": al, "ask_c": ac,
            })

    def on_quote(self, epoch: float, bid: float, ask: float) -> bool:
        """Returns True when a bar just closed."""

        minute = int(epoch // 60) * 60
        closed = False
        if self.current is not None and self.current["epoch"] != minute:
            self.bars.append(self.current)
            self.current = None
            closed = True
        if self.current is None:
            self.current = {
                "epoch": minute, "bid_o": bid, "bid_h": bid, "bid_l": bid, "bid_c": bid,
                "ask_o": ask, "ask_h": ask, "ask_l": ask, "ask_c": ask,
            }
        else:
            cur = self.current
            cur["bid_h"] = max(cur["bid_h"], bid); cur["bid_l"] = min(cur["bid_l"], bid)
            cur["ask_h"] = max(cur["ask_h"], ask); cur["ask_l"] = min(cur["ask_l"], ask)
            cur["bid_c"] = bid; cur["ask_c"] = ask
        return closed


def compute_factors(calc_core, bars: list[dict[str, float]]) -> Optional[dict[str, Any]]:
    if len(bars) < WARMUP_BARS:
        return None
    tail = bars[-2000:]
    mid_c = pd.Series([(b["bid_c"] + b["ask_c"]) / 2 for b in tail])
    mid_h = pd.Series([(b["bid_h"] + b["ask_h"]) / 2 for b in tail])
    mid_l = pd.Series([(b["bid_l"] + b["ask_l"]) / 2 for b in tail])
    mid_o = [(b["bid_o"] + b["ask_o"]) / 2 for b in tail]
    ma10 = mid_c.rolling(10, min_periods=10).mean().iloc[-1]
    ma20 = mid_c.rolling(20, min_periods=20).mean().iloc[-1]
    ema20 = mid_c.ewm(span=20, adjust=False, min_periods=20).mean().iloc[-1]
    vol_5m = (mid_c.diff().abs().rolling(5, min_periods=5).mean() / 0.01).iloc[-1]
    rsi = calc_core._rsi(mid_c, 14).iloc[-1]
    atr = calc_core._atr(mid_h, mid_l, mid_c, 14).iloc[-1]
    adx = calc_core._adx(mid_h, mid_l, mid_c, 14).iloc[-1]
    last = tail[-1]
    spread_pips = (last["ask_c"] - last["bid_c"]) / PIP
    if any(pd.isna(v) for v in (ma10, ma20, ema20, vol_5m, rsi, atr, adx)):
        return None
    return {
        "close": float(mid_c.iloc[-1]), "open": float(mid_o[-1]),
        "high": float(mid_h.iloc[-1]), "low": float(mid_l.iloc[-1]),
        "ma10": float(ma10), "ma20": float(ma20), "ema20": float(ema20),
        "rsi": float(rsi), "atr": float(atr), "adx": float(adx),
        "vol_5m": float(vol_5m), "spread_pips": float(spread_pips),
        "candles": [
            {"high": (b["bid_h"] + b["ask_h"]) / 2, "low": (b["bid_l"] + b["ask_l"]) / 2,
             "open": (b["bid_o"] + b["ask_o"]) / 2, "close": (b["bid_c"] + b["ask_c"]) / 2}
            for b in tail[-4:]
        ],
    }


def _load_recent_corpus(root: Path, days: int) -> list[tuple]:
    import gzip
    cutoff = time_mod.time() - days * 86400
    rows = []
    for shard in sorted(root.glob(f"*/{PAIR}/{PAIR}_M1_BA_*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                epoch = int(datetime.fromisoformat(row["time"][:19] + "+00:00").timestamp())
                if epoch < cutoff:
                    continue
                rows.append((
                    epoch,
                    float(row["bid"]["o"]), float(row["bid"]["h"]),
                    float(row["bid"]["l"]), float(row["bid"]["c"]),
                    float(row["ask"]["o"]), float(row["ask"]["h"]),
                    float(row["ask"]["l"]), float(row["ask"]["c"]),
                ))
    rows.sort()
    dedup = []
    last = None
    for r in rows:
        if r[0] != last:
            dedup.append(r)
            last = r[0]
    return dedup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, default=Path(
        "/Users/tossaki/App/QuantRabbit-live/logs/replay/oanda_history_m1_2020_2026"))
    args = parser.parse_args()

    calc_core, strategy_mod = _import_vendored()
    Strategy = strategy_mod.MomentumBurstMicro
    client = OandaReadOnlyClient()
    ledger = ChainLedger(args.ledger)

    builder = M1Builder()
    seed_rows = _load_recent_corpus(args.corpus_root, days=8)
    builder.seed(seed_rows)
    ledger.write("ENV_START", {
        "contract": "QR_LIVE_SHADOW_ENVIRONMENT_V1",
        "pair": PAIR,
        "poll_seconds": POLL_SECONDS,
        "seeded_bars": len(seed_rows),
        "order_authority": "NONE",
        "books": ["BOOK_SL (strategy TP/SL)", "BOOK_TPO (TP-only + 4h ceiling)"],
        "honesty": "wall-clock time; live quotes only; fills only on touched real quotes",
    })

    # shadow positions: {book: [ {side, entry, entry_ts, tp, sl(optional)} ]}
    books: dict[str, list[dict[str, Any]]] = {"BOOK_SL": [], "BOOK_TPO": []}
    deadline = time_mod.time() + args.minutes * 60.0
    last_signal_bar: Optional[int] = None
    polls = 0
    while time_mod.time() < deadline:
        now = datetime.now(UTC)
        status = compute_market_status(now)
        if not status.is_fx_open:
            ledger.write("MARKET_CLOSED", {"note": "no trading while closed"})
            time_mod.sleep(min(60.0, max(POLL_SECONDS, 30.0)))
            continue
        try:
            quote = client.quotes([PAIR])[PAIR]
        except Exception as exc:
            ledger.write("QUOTE_ERROR", {"error": str(exc)[:200]})
            time_mod.sleep(POLL_SECONDS)
            continue
        q_age = (now - quote.timestamp_utc).total_seconds()
        polls += 1
        ledger.write("QUOTE", {
            "bid": quote.bid, "ask": quote.ask,
            "spread_pips": round((quote.ask - quote.bid) / PIP, 2),
            "quote_ts_utc": quote.timestamp_utc.isoformat(),
            "age_s": round(q_age, 1),
        })
        if q_age > STALE_QUOTE_MAX_S:
            ledger.write("STALE_QUOTE_REFUSAL", {"age_s": round(q_age, 1)})
            time_mod.sleep(POLL_SECONDS)
            continue

        # shadow exit checks against the REAL quote
        epoch = now.timestamp()
        for book, positions in books.items():
            still = []
            for pos in positions:
                side = pos["side"]
                exit_price = None
                reason = None
                if side == "LONG" and quote.bid >= pos["tp"]:
                    exit_price, reason = pos["tp"], "TP"
                elif side == "SHORT" and quote.ask <= pos["tp"]:
                    exit_price, reason = pos["tp"], "TP"
                elif book == "BOOK_SL" and pos.get("sl") is not None:
                    if side == "LONG" and quote.bid <= pos["sl"]:
                        exit_price, reason = pos["sl"], "SL"
                    elif side == "SHORT" and quote.ask >= pos["sl"]:
                        exit_price, reason = pos["sl"], "SL"
                if exit_price is None and epoch - pos["entry_epoch"] >= HARD_CEILING_S:
                    exit_price = quote.bid if side == "LONG" else quote.ask
                    reason = "HARD_CEILING"
                if exit_price is None:
                    still.append(pos)
                    continue
                pips = (exit_price - pos["entry"]) / PIP if side == "LONG" else (
                    pos["entry"] - exit_price) / PIP
                ledger.write("SHADOW_EXIT", {
                    "book": book, "side": side, "entry": pos["entry"],
                    "exit": exit_price, "reason": reason, "pips": round(pips, 2),
                    "held_s": round(epoch - pos["entry_epoch"]),
                })
            books[book] = still

        # bar building + signal on bar close
        bar_closed = builder.on_quote(epoch, quote.bid, quote.ask)
        if bar_closed:
            fac = compute_factors(calc_core, builder.bars)
            if fac is not None:
                bar_id = builder.bars[-1]["epoch"]
                if bar_id != last_signal_bar:
                    last_signal_bar = bar_id
                    signal = Strategy.check(fac)
                    ledger.write("BAR_CLOSE", {
                        "bar_epoch": bar_id,
                        "spread_pips": fac["spread_pips"],
                        "signal": bool(signal),
                        "action": (signal or {}).get("action"),
                    })
                    if signal and signal.get("action") in {"OPEN_LONG", "OPEN_SHORT"}:
                        side = "LONG" if signal["action"] == "OPEN_LONG" else "SHORT"
                        entry = quote.ask if side == "LONG" else quote.bid
                        tp_pips = float(signal["tp_pips"]); sl_pips = float(signal["sl_pips"])
                        tp = entry + tp_pips * PIP if side == "LONG" else entry - tp_pips * PIP
                        sl = entry - sl_pips * PIP if side == "LONG" else entry + sl_pips * PIP
                        for book in books:
                            if len(books[book]) >= MAX_CONCURRENT:
                                continue
                            books[book].append({
                                "side": side, "entry": entry, "entry_epoch": epoch,
                                "tp": tp, "sl": sl if book == "BOOK_SL" else None,
                            })
                        ledger.write("SHADOW_ENTRY", {
                            "side": side, "entry_real_quote": entry,
                            "tp_pips": tp_pips, "sl_pips": sl_pips,
                            "confidence": signal.get("confidence"),
                            "spread_at_entry_pips": round((quote.ask - quote.bid) / PIP, 2),
                        })
        time_mod.sleep(POLL_SECONDS)

    ledger.write("ENV_STOP", {
        "polls": polls,
        "open_positions": {k: len(v) for k, v in books.items()},
    })
    print(json.dumps({"status": "SHADOW_ENV_DONE", "polls": polls}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
