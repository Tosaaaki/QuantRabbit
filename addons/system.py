"""Replay-oriented harness for QuantRabbit addon workers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from allocator import BanditAllocator
from workers.mm_lite import MMLiteWorker, DEFAULT_CONFIG as MM_LITE_DEFAULT
from workers.session_open import (
    SessionOpenWorker,
    DEFAULT_CONFIG as SESSION_OPEN_DEFAULT,
)
from workers.stop_run_reversal import (
    StopRunReversalWorker,
    DEFAULT_CONFIG as STOP_RUN_DEFAULT,
)
from workers.vol_squeeze import VolSqueezeWorker, DEFAULT_CONFIG as VOL_SQUEEZE_DEFAULT

Timeline = Iterable[float]
PIP_SIZE = 0.01  # USD/JPY pip value


def _copy_cfg(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(base)
    if overrides:
        merged.update(overrides)
    return merged


class ReplayDataFeed:
    """Minimal data feed for replaying workers on cached candle data."""

    def __init__(self, *, spread_bp: float = 1.2):
        self._bars: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        self._cursor: Dict[Tuple[str, str], int] = {}
        self._last_price: Dict[str, float] = {}
        self._spread_bp = float(spread_bp)

    def add_bars(self, symbol: str, timeframe: str, bars: List[Dict[str, Any]]) -> None:
        normalized: List[Dict[str, Any]] = []
        for row in bars:
            ts_val = row.get("timestamp") or row.get("ts") or row.get("time")
            try:
                ts = float(ts_val)
            except (TypeError, ValueError):
                ts = 0.0
            normalized.append(
                {
                    "open": float(row.get("open", row.get("o", 0.0))),
                    "high": float(row.get("high", row.get("h", 0.0))),
                    "low": float(row.get("low", row.get("l", 0.0))),
                    "close": float(row.get("close", row.get("c", 0.0))),
                    "timestamp": ts,
                    "volume": int(row.get("volume", row.get("vol", 0)) or 0),
                }
            )
        normalized.sort(key=lambda x: x["timestamp"])
        key = (symbol, timeframe)
        self._bars[key] = normalized
        self._cursor[key] = 0

    def advance(self, now_epoch: float) -> None:
        for key, rows in self._bars.items():
            idx = self._cursor.get(key, 0)
            total = len(rows)
            while idx < total and rows[idx]["timestamp"] <= now_epoch:
                idx += 1
            if idx != self._cursor.get(key, 0):
                self._cursor[key] = idx
                if idx > 0:
                    symbol = key[0]
                    self._last_price[symbol] = rows[idx - 1]["close"]

    def get_bars(self, symbol: str, timeframe: str, n: int) -> List[Dict[str, Any]]:
        key = (symbol, timeframe)
        rows = self._bars.get(key)
        if not rows:
            return []
        idx = self._cursor.get(key, 0)
        if idx <= 0:
            return []
        window = rows[:idx]
        if n <= 0 or n >= len(window):
            return list(window)
        return window[-n:]

    def last(self, symbol: str) -> float:
        return float(self._last_price.get(symbol, 0.0))

    def best_bid_ask(self, symbol: str) -> Optional[Tuple[float, float]]:
        mid = self._last_price.get(symbol)
        if mid is None:
            return None
        half_spread = mid * (self._spread_bp * 1e-4) * 0.5
        return (mid - half_spread, mid + half_spread)


class ReplayBroker:
    """Order sink that records intents without touching live infrastructure."""

    def __init__(self):
        self.sent_orders: List[Dict[str, Any]] = []
        self.cancelled_orders: List[str] = []

    def send(self, order: Dict[str, Any]) -> Dict[str, Any]:
        self.sent_orders.append(order)
        return {"status": "accepted", "order": order}

    def cancel(self, order_id: str) -> Dict[str, str]:
        self.cancelled_orders.append(order_id)
        return {"status": "cancelled", "order_id": order_id}


@dataclass
class StepResult:
    ts: float
    iso: str
    signals: Dict[str, List[Dict[str, Any]]]


@dataclass
class SimTrade:
    worker: str
    symbol: str
    side: str
    entry_ts: float
    entry_iso: str
    entry_price: float
    stop: float
    take: float
    meta: Dict[str, Any] = field(default_factory=dict)
    exit_ts: Optional[float] = None
    exit_iso: Optional[str] = None
    exit_price: Optional[float] = None
    outcome: str = "open"
    duration_min: float = 0.0
    pnl_pips: float = 0.0


@dataclass
class RunOutcome:
    steps: List[StepResult]
    trades: List[SimTrade]


class AddonSystem:
    """Orchestrates addon workers over a replay timeline and estimates PnL."""

    def __init__(
        self,
        datafeed: ReplayDataFeed,
        broker: ReplayBroker,
        *,
        session_open_cfg: Optional[Dict[str, Any]] = None,
        vol_squeeze_cfg: Optional[Dict[str, Any]] = None,
        stop_run_cfg: Optional[Dict[str, Any]] = None,
        mm_lite_cfg: Optional[Dict[str, Any]] = None,
        logger: Optional[Any] = None,
    ):
        self.datafeed = datafeed
        self.broker = broker
        self.logger = logger
        self._configs: Dict[str, Dict[str, Any]] = {}

        self.session_open = None
        if session_open_cfg is not False:
            cfg = _copy_cfg(SESSION_OPEN_DEFAULT, session_open_cfg or {})
            self.session_open = SessionOpenWorker(cfg, broker, datafeed, logger=logger)
            self._configs["session_open"] = cfg

        self.vol_squeeze = None
        if vol_squeeze_cfg is not False:
            cfg = _copy_cfg(VOL_SQUEEZE_DEFAULT, vol_squeeze_cfg or {})
            self.vol_squeeze = VolSqueezeWorker(cfg, broker, datafeed, logger=logger)
            self._configs["vol_squeeze"] = cfg

        self.stop_run = None
        if stop_run_cfg is not False:
            cfg = _copy_cfg(STOP_RUN_DEFAULT, stop_run_cfg or {})
            self.stop_run = StopRunReversalWorker(cfg, broker, datafeed, logger=logger)
            self._configs["stop_run_reversal"] = cfg

        self.mm_lite = None
        if mm_lite_cfg is not False:
            cfg = _copy_cfg(MM_LITE_DEFAULT, mm_lite_cfg or {})
            self.mm_lite = MMLiteWorker(cfg, broker, datafeed, event_flags=None, logger=logger)
            self._configs["mm_lite"] = cfg

        self._registry: List[Tuple[str, Any, bool]] = []
        if self.session_open:
            self._registry.append(("session_open", self.session_open, True))
        if self.vol_squeeze:
            self._registry.append(("vol_squeeze", self.vol_squeeze, False))
        if self.stop_run:
            self._registry.append(("stop_run_reversal", self.stop_run, False))
        if self.mm_lite:
            self._registry.append(("mm_lite", self.mm_lite, False))

        self._open_trades: List[SimTrade] = []
        self._closed_trades: List[SimTrade] = []

    def step(self, now_epoch: float) -> StepResult:
        self.datafeed.advance(now_epoch)
        step_signals: Dict[str, List[Dict[str, Any]]] = {}
        for name, worker, needs_now in self._registry:
            try:
                intents = worker.run_once(now=now_epoch) if needs_now else worker.run_once()
            except TypeError:
                intents = worker.run_once()
            if not intents:
                continue
            step_signals[name] = intents
            for intent in intents:
                self._maybe_open_trade(name, intent, now_epoch)
        self._update_open_trades(now_epoch)
        iso = datetime.fromtimestamp(now_epoch, tz=timezone.utc).isoformat()
        return StepResult(ts=now_epoch, iso=iso, signals=step_signals)

    def run(self, timeline: Timeline) -> RunOutcome:
        self._open_trades = []
        self._closed_trades = []
        results: List[StepResult] = []
        last_ts: Optional[float] = None
        for ts in timeline:
            last_ts = ts
            step = self.step(ts)
            if step.signals:
                results.append(step)
        if last_ts is not None:
            self._close_remaining(last_ts)
        return RunOutcome(steps=results, trades=list(self._closed_trades))

    def _maybe_open_trade(self, worker_name: str, intent: Dict[str, Any], now_epoch: float) -> None:
        if worker_name == "mm_lite":
            return  # quoting only
        cfg = self._configs.get(worker_name, {})
        exit_cfg = cfg.get("exit") if isinstance(cfg, dict) else None
        if not exit_cfg:
            return
        stop_mult = float(exit_cfg.get("stop_atr", 0.0) or 0.0)
        tp_mult = float(exit_cfg.get("tp_atr", 0.0) or 0.0)
        if stop_mult <= 0.0 or tp_mult <= 0.0:
            return
        meta = intent.get("meta") or {}
        atr = float(meta.get("atr", 0.0) or 0.0)
        if atr <= 0.0:
            return
        symbol = intent.get("symbol")
        side = intent.get("side")
        if not symbol or side not in {"long", "short"}:
            return
        entry_px = float(intent.get("px") or self.datafeed.last(symbol) or 0.0)
        if entry_px <= 0.0:
            return
        if side == "long":
            stop = entry_px - stop_mult * atr
            take = entry_px + tp_mult * atr
        else:
            stop = entry_px + stop_mult * atr
            take = entry_px - tp_mult * atr
        meta_copy = dict(meta)
        meta_copy.setdefault("atr", atr)
        breakeven_mult = float(exit_cfg.get("breakeven_mult", 0.0) or 0.0)
        if breakeven_mult > 0.0:
            meta_copy["breakeven_mult"] = breakeven_mult
        trade = SimTrade(
            worker=worker_name,
            symbol=symbol,
            side=side,
            entry_ts=now_epoch,
            entry_iso=datetime.fromtimestamp(now_epoch, tz=timezone.utc).isoformat(),
            entry_price=entry_px,
            stop=stop,
            take=take,
            meta=meta_copy,
        )
        self._open_trades.append(trade)

    def _update_open_trades(self, now_epoch: float) -> None:
        if not self._open_trades:
            return
        for trade in list(self._open_trades):
            bars = self.datafeed.get_bars(trade.symbol, "1m", 1)
            if not bars:
                continue
            bar = bars[-1]
            bar_ts = float(bar.get("timestamp", now_epoch))
            if bar_ts <= trade.entry_ts:
                continue
            low = float(bar.get("low", trade.entry_price))
            high = float(bar.get("high", trade.entry_price))
            atr = float(trade.meta.get("atr", 0.0) or 0.0)
            breakeven_mult = float(trade.meta.get("breakeven_mult", 0.0) or 0.0)
            if atr > 0.0 and breakeven_mult > 0.0:
                if trade.side == "long":
                    trigger = trade.entry_price + breakeven_mult * atr
                    if high >= trigger and trade.stop < trade.entry_price:
                        trade.stop = trade.entry_price
                else:
                    trigger = trade.entry_price - breakeven_mult * atr
                    if low <= trigger and trade.stop > trade.entry_price:
                        trade.stop = trade.entry_price
            if trade.side == "long":
                if high >= trade.take:
                    self._close_trade(trade, trade.take, bar_ts, "take")
                    continue
                if low <= trade.stop:
                    self._close_trade(trade, trade.stop, bar_ts, "stop")
                    continue
            else:
                if low <= trade.take:
                    self._close_trade(trade, trade.take, bar_ts, "take")
                    continue
                if high >= trade.stop:
                    self._close_trade(trade, trade.stop, bar_ts, "stop")
                    continue

    def _close_trade(self, trade: SimTrade, price: float, ts: float, outcome: str) -> None:
        if trade not in self._open_trades:
            return
        self._open_trades.remove(trade)
        trade.exit_ts = ts
        trade.exit_iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        trade.exit_price = float(price)
        trade.outcome = outcome
        trade.duration_min = max(0.0, (ts - trade.entry_ts) / 60.0)
        if trade.side == "long":
            trade.pnl_pips = (trade.exit_price - trade.entry_price) / PIP_SIZE
        else:
            trade.pnl_pips = (trade.entry_price - trade.exit_price) / PIP_SIZE
        self._closed_trades.append(trade)

    def _close_remaining(self, last_ts: float) -> None:
        if not self._open_trades:
            return
        for trade in list(self._open_trades):
            bars = self.datafeed.get_bars(trade.symbol, "1m", 1)
            if bars:
                bar = bars[-1]
                close_px = float(bar.get("close", trade.entry_price))
                bar_ts = float(bar.get("timestamp", last_ts))
            else:
                close_px = self.datafeed.last(trade.symbol) or trade.entry_price
                bar_ts = last_ts
            self._close_trade(trade, close_px, bar_ts, "close")

    @staticmethod
    def summarize(step_results: Iterable[StepResult]) -> Dict[str, Dict[str, Any]]:
        summary: Dict[str, Dict[str, Any]] = {}
        for step in step_results:
            for name, intents in step.signals.items():
                entry = summary.setdefault(
                    name,
                    {"signals": 0, "long": 0, "short": 0, "avg_px": 0.0},
                )
                prior = entry["signals"]
                batch = len(intents)
                entry["signals"] = prior + batch
                entry["long"] += sum(1 for it in intents if it.get("side") == "long")
                entry["short"] += sum(1 for it in intents if it.get("side") == "short")
                if batch:
                    avg_price = sum(float(it.get("px", 0.0)) for it in intents) / batch
                    entry["avg_px"] = (
                        (entry["avg_px"] * prior + avg_price * batch)
                        / max(entry["signals"], 1)
                    )
        return summary

    @staticmethod
    def summarize_trades(trades: Iterable[SimTrade]) -> Dict[str, Dict[str, Any]]:
        summary: Dict[str, Dict[str, Any]] = {}
        for trade in trades:
            entry = summary.setdefault(
                trade.worker,
                {"trades": 0, "wins": 0, "losses": 0, "pnl_pips": 0.0},
            )
            entry["trades"] += 1
            entry["pnl_pips"] += trade.pnl_pips
            if trade.pnl_pips > 0:
                entry["wins"] += 1
            elif trade.pnl_pips < 0:
                entry["losses"] += 1
        return summary

    @staticmethod
    def allocate_budgets(summary: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        metrics = {}
        for name, stats in summary.items():
            signals = float(stats.get("signals", 0))
            wins = float(stats.get("long", 0))
            losses = float(stats.get("short", 0))
            net = wins - losses
            metrics[name] = {
                "wins": max(wins, 0.0),
                "trades": max(signals, wins),
                "ev": net / max(signals, 1.0),
                "hit": wins / max(signals, 1.0) if signals else 0.0,
                "sharpe": net / max((wins + losses) ** 0.5, 1.0),
                "max_dd": -abs(losses) / max(signals, 1.0) if signals else 0.0,
            }
        allocator = BanditAllocator(total_budget_bps=120.0, cap_bps=45.0, floor_bps=6.0)
        return allocator.allocate(metrics)
