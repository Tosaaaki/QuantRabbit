#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
system_backtest.py
~~~~~~~~~~~~~~~~~~
QuantRabbit のメインループ構成要素を簡易に模したオフライン・バックテストハーネス。

主な特徴:
  • `indicators.factor_cache` を利用して M1/H4 因子を更新
  • レジーム/フォーカス判定 → ヒューリスティック GPT フォールバックで戦略選択
  • `_dynamic_risk_pct` + `allowed_lot` によるリスク計算とポケット配分
  • シンプルなポジション管理（SL/TP 到達、最大保有時間）

注意:
  本スクリプトは「約定順序」や `ExitManager` の複雑な分割利確ロジックを再現しません。
  StageTracker・ニュースモジュール・RiskGuard のドローダウン判定も簡易化しています。
  実運用ロジックとの差分を理解した上で参考値として利用してください。
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.focus_decider import decide_focus
from analysis.local_decider import heuristic_decision
from analysis.param_context import ParamContext, ParamSnapshot
from analysis.range_guard import detect_range_mode
from analysis.regime_classifier import classify
from indicators import factor_cache as factor_cache_module
from execution.risk_guard import allowed_lot
from signals.pocket_allocator import alloc, DEFAULT_SCALP_SHARE

import main as main_mod

try:
    from scripts import replay_offline as replay_mod
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"[FATAL] replay_offline import failed: {exc}") from exc


PIP_VALUE = 0.01  # USD/JPY
MIN_UNITS = 100  # 最小発注単位（バックテストでは 100 通貨を許容）


@dataclass
class TradeRecord:
    trade_id: int
    pocket: str
    strategy: str
    direction: str  # long / short
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    sl_price: float
    tp_price: float
    units: int
    pnl_jpy: float
    pnl_pips: float
    outcome: str  # TP / SL / TIME / MANUAL / OPEN


@dataclass
class Position:
    trade_id: int
    pocket: str
    strategy: str
    direction: str
    units: int
    entry_price: float
    sl_price: float
    tp_price: float
    entry_time: datetime
    ttl_minutes: int

    def hit_sl(self, candle: dict) -> bool:
        low = candle["low"]
        high = candle["high"]
        if self.direction == "long":
            if low <= self.sl_price <= high:
                return True
            return low <= self.sl_price
        else:
            if low <= self.sl_price <= high:
                return True
            return high >= self.sl_price

    def hit_tp(self, candle: dict) -> bool:
        low = candle["low"]
        high = candle["high"]
        if self.direction == "long":
            return high >= self.tp_price
        return low <= self.tp_price


@dataclass
class BacktestMetrics:
    total_pnl_jpy: float
    total_pnl_pips: float
    trades: List[TradeRecord]

    def summary(self) -> Dict[str, float]:
        wins = [t for t in self.trades if t.pnl_pips > 0]
        losses = [t for t in self.trades if t.pnl_pips < 0]
        win_rate = len(wins) / len(self.trades) if self.trades else 0.0
        gross_win = sum(t.pnl_pips for t in wins)
        gross_loss = abs(sum(t.pnl_pips for t in losses))
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else math.inf
        max_dd = self._max_drawdown()
        return {
            "total_pnl_pips": round(self.total_pnl_pips, 2),
            "total_pnl_jpy": round(self.total_pnl_jpy, 0),
            "trades": len(self.trades),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4) if math.isfinite(profit_factor) else float("inf"),
            "max_dd_pips": round(max_dd, 2),
        }

    def _max_drawdown(self) -> float:
        equity_curve = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in self.trades:
            equity_curve += t.pnl_pips
            peak = max(peak, equity_curve)
            drawdown = peak - equity_curve
            max_dd = max(max_dd, drawdown)
        return max_dd


def _load_candles(path: Path) -> List[dict]:
    return replay_mod._load_oanda_candles(path)


async def _feed_historical(tf: str, candles: Iterable[dict]) -> None:
    on_candle = factor_cache_module.on_candle
    for candle in candles:
        await on_candle(tf, candle)


def _price_from_pips(side: str, entry: float, pips: float) -> float:
    sign = -1 if side == "BUY" else 1
    value = entry + sign * pips * PIP_VALUE
    return round(value, 3)


def _action_to_side(action: str) -> str:
    return "BUY" if action == "OPEN_LONG" else "SELL"


def _ensure_tz(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


class SystemBacktester:
    def __init__(
        self,
        *,
        m1_path: Path,
        h4_path: Optional[Path],
        initial_equity: float,
        margin_available: float,
        margin_rate: float,
        lot_cap: Optional[float],
        max_hold_minutes: int,
        use_range_override: bool,
    ) -> None:
        self.m1_path = m1_path
        self.h4_path = h4_path
        self.initial_equity = initial_equity
        self.margin_available = margin_available
        self.margin_rate = margin_rate
        self.lot_cap = lot_cap
        self.max_hold_minutes = max(1, max_hold_minutes)
        self.use_range_override = use_range_override

        self.positions: Dict[int, Position] = {}
        self.trades: List[TradeRecord] = []
        self._next_trade_id = 1
        self.equity = initial_equity
        self.param_ctx = ParamContext()

    async def run(self) -> BacktestMetrics:
        importlib.reload(factor_cache_module)
        m1_candles = _load_candles(self.m1_path)
        if not m1_candles:
            raise RuntimeError(f"No candles in {self.m1_path}")
        h4_candles = replay_mod._collect_h4_history(self.m1_path, self.h4_path)
        await _feed_historical("H4", h4_candles)

        last_fac_h4: Optional[dict] = None
        on_candle = factor_cache_module.on_candle
        all_factors = factor_cache_module.all_factors

        for candle in m1_candles:
            await on_candle("M1", candle)
            factors = all_factors()
            fac_m1 = factors.get("M1")
            fac_h4 = factors.get("H4")
            if fac_m1 is None:
                continue
            if fac_h4:
                last_fac_h4 = fac_h4
            elif last_fac_h4 is not None:
                fac_h4 = last_fac_h4
            else:
                continue

            ts = _ensure_tz(candle["time"])
            self._update_positions(candle, ts)

            macro_regime = classify(fac_h4, "H4")
            micro_regime = classify(fac_m1, "M1")
            focus_tag, weight_macro = decide_focus(
                macro_regime,
                micro_regime,
                event_soon=False,
            )
            payload = {
                "ts": ts.isoformat(timespec="seconds"),
                "reg_macro": macro_regime,
                "reg_micro": micro_regime,
                "factors_m1": {k: v for k, v in fac_m1.items() if k != "candles"},
                "factors_h4": {k: v for k, v in fac_h4.items() if k != "candles"},
                "perf": {},
                "news_short": [],
                "event_soon": False,
            }
            gpt_like = heuristic_decision(payload)
            ranked = list(gpt_like.get("ranked_strategies", []))
            if not ranked:
                continue
            weight_macro = gpt_like.get("weight_macro", weight_macro)
            weight_scalp_raw = gpt_like.get("weight_scalp")
            try:
                weight_scalp = max(0.0, min(1.0, float(weight_scalp_raw)))
            except (TypeError, ValueError):
                weight_scalp = None
            focus_tag = gpt_like.get("focus_tag", focus_tag)

            range_ctx = detect_range_mode(fac_m1, fac_h4)
            if self.use_range_override and range_ctx.active:
                focus_tag = "micro"
                weight_macro = min(weight_macro, 0.15)
                if weight_scalp is not None:
                    weight_scalp = min(weight_scalp, 0.2)

            param_snapshot: ParamSnapshot = self.param_ctx.update(
                now=ts,
                fac_m1=fac_m1,
                fac_h4=fac_h4,
                spread_snapshot=None,
            )

            stage_overrides, _, _ = self.param_ctx.stage_overrides(
                main_mod._BASE_STAGE_RATIOS, range_active=range_ctx.active
            )
            main_mod._set_stage_plan_overrides(stage_overrides)

            signals = self._evaluate_strategies(ranked, fac_m1, range_ctx.active)
            if not signals:
                continue

            risk_pct = main_mod._dynamic_risk_pct(
                signals,
                range_ctx.active,
                weight_macro,
                context=param_snapshot,
            )
            avg_sl = self._average_sl(signals)
            price = float(fac_m1.get("close") or 0.0)
            lot_total = allowed_lot(
                self.equity,
                sl_pips=max(1.0, avg_sl),
                margin_available=self.margin_available,
                price=price if price > 0 else None,
                margin_rate=self.margin_rate,
                risk_pct_override=risk_pct,
            )
            if self.lot_cap is not None:
                lot_total = min(lot_total, self.lot_cap)
            if lot_total <= 0:
                continue

            scalp_share = DEFAULT_SCALP_SHARE if weight_scalp is None else 0.0
            pocket_lots = alloc(
                lot_total,
                weight_macro,
                weight_scalp=weight_scalp,
                scalp_share=scalp_share,
            )
            self._process_signals(signals, pocket_lots, price, ts)

        self._close_remaining(_ensure_tz(m1_candles[-1]["time"]))

        total_jpy = sum(t.pnl_jpy for t in self.trades)
        total_pips = sum(t.pnl_pips for t in self.trades)
        return BacktestMetrics(total_pnl_jpy=total_jpy, total_pnl_pips=total_pips, trades=self.trades)

    def _evaluate_strategies(self, ranked: List[str], fac_m1: dict, range_active: bool) -> List[dict]:
        signals: List[dict] = []
        for name in ranked:
            cls = main_mod.STRATEGIES.get(name)
            if not cls:
                continue
            if range_active and cls.name not in main_mod.ALLOWED_RANGE_STRATEGIES:
                continue
            if name == "NewsSpikeReversal":
                raw = cls.check(fac_m1, [])  # type: ignore[arg-type]
            else:
                raw = cls.check(fac_m1)
            if not raw:
                continue
            if raw.get("action") not in {"OPEN_LONG", "OPEN_SHORT"}:
                continue
            payload = dict(raw)
            payload["strategy"] = name
            payload["pocket"] = getattr(cls, "pocket", "macro")
            signals.append(payload)
        return signals

    def _process_signals(
        self,
        signals: List[dict],
        pocket_lots: Dict[str, float],
        price: float,
        ts: datetime,
    ) -> None:
        for sig in signals:
            pocket = sig["pocket"]
            lot_capacity = pocket_lots.get(pocket, 0.0)
            if lot_capacity <= 0:
                continue
            units = int(round(lot_capacity * 100000))
            if abs(units) < MIN_UNITS:
                continue
            action = sig["action"]
            side = _action_to_side(action)
            direction = "long" if side == "BUY" else "short"
            sl_pips = float(sig["sl_pips"])
            tp_pips = float(sig["tp_pips"])
            sl_price = _price_from_pips(side, price, sl_pips)
            tp_price = _price_from_pips("SELL" if side == "BUY" else "BUY", price, tp_pips)
            ttl = int(sig.get("timeout_sec", 0) or 0)
            if ttl <= 0:
                ttl = self.max_hold_minutes * 60

            trade_id = self._next_trade_id
            self._next_trade_id += 1
            pos = Position(
                trade_id=trade_id,
                pocket=pocket,
                strategy=sig["strategy"],
                direction=direction,
                units=units if direction == "long" else -units,
                entry_price=price,
                sl_price=sl_price,
                tp_price=tp_price,
                entry_time=ts,
                ttl_minutes=max(1, ttl // 60),
            )
            self.positions[trade_id] = pos

    def _update_positions(self, candle: dict, ts: datetime) -> None:
        closing_ids: List[int] = []
        for trade_id, pos in list(self.positions.items()):
            age_minutes = int((ts - pos.entry_time).total_seconds() // 60)
            exit_reason: Optional[str] = None
            exit_price: Optional[float] = None

            if pos.hit_sl(candle):
                exit_reason = "SL"
                exit_price = pos.sl_price
            elif pos.hit_tp(candle):
                exit_reason = "TP"
                exit_price = pos.tp_price
            elif age_minutes >= pos.ttl_minutes:
                exit_reason = "TIME"
                exit_price = candle["close"]

            if exit_reason:
                closing_ids.append(trade_id)
                self._close_position(trade_id, exit_price, ts, exit_reason)

        for trade_id in closing_ids:
            self.positions.pop(trade_id, None)

    def _close_position(self, trade_id: int, exit_price: float, ts: datetime, reason: str) -> None:
        pos = self.positions.get(trade_id)
        if not pos:
            return
        pnl_jpy = (exit_price - pos.entry_price) * pos.units
        pnl_pips = (exit_price - pos.entry_price) / PIP_VALUE
        if pos.direction == "short":
            pnl_pips *= -1
        self.equity += pnl_jpy
        record = TradeRecord(
            trade_id=trade_id,
            pocket=pos.pocket,
            strategy=pos.strategy,
            direction=pos.direction,
            entry_time=pos.entry_time,
            exit_time=ts,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            sl_price=pos.sl_price,
            tp_price=pos.tp_price,
            units=pos.units,
            pnl_jpy=pnl_jpy,
            pnl_pips=pnl_pips,
            outcome=reason,
        )
        self.trades.append(record)

    def _close_remaining(self, ts: datetime) -> None:
        for trade_id in list(self.positions.keys()):
            pos = self.positions[trade_id]
            self._close_position(trade_id, pos.entry_price, ts, "OPEN")
            self.positions.pop(trade_id, None)

    @staticmethod
    def _average_sl(signals: Iterable[dict]) -> float:
        values = [
            float(sig.get("sl_pips") or 0.0)
            for sig in signals
            if sig.get("sl_pips")
        ]
        return sum(values) / len(values) if values else 12.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline backtest for QuantRabbit main loop (approximation).")
    parser.add_argument("--m1", required=True, help="M1 candle JSON path (logs/candles_M1_*.json)")
    parser.add_argument("--h4", default="", help="Optional H4 candle JSON path")
    parser.add_argument("--equity", type=float, default=12000.0, help="Initial equity (JPY)")
    parser.add_argument("--margin-available", type=float, default=9000.0, help="Available margin (JPY)")
    parser.add_argument("--margin-rate", type=float, default=0.04, help="Margin rate (OANDA default ≈ 4%)")
    parser.add_argument("--lot-cap", type=float, default=0.0, help="Optional hard lot cap (0=none)")
    parser.add_argument("--max-hold-minutes", type=int, default=90, help="Time based exit fallback (minutes)")
    parser.add_argument("--range-focus", action="store_true", help="Force range mode override behaviour.")
    parser.add_argument("--json-out", default="", help="Optional JSON output path for detailed trades.")
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    m1_path = Path(args.m1)
    h4_path = Path(args.h4) if args.h4 else None
    lot_cap = args.lot_cap if args.lot_cap > 0 else None

    engine = SystemBacktester(
        m1_path=m1_path,
        h4_path=h4_path,
        initial_equity=args.equity,
        margin_available=args.margin_available,
        margin_rate=args.margin_rate,
        lot_cap=lot_cap,
        max_hold_minutes=args.max_hold_minutes,
        use_range_override=args.range_focus,
    )
    metrics = await engine.run()

    summary = metrics.summary()
    print(json.dumps(summary, ensure_ascii=False))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": summary,
            "trades": [
                {
                    "trade_id": t.trade_id,
                    "pocket": t.pocket,
                    "strategy": t.strategy,
                    "direction": t.direction,
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat(),
                    "entry_price": round(t.entry_price, 5),
                    "exit_price": round(t.exit_price, 5),
                    "sl_price": round(t.sl_price, 5),
                    "tp_price": round(t.tp_price, 5),
                    "units": t.units,
                    "pnl_jpy": round(t.pnl_jpy, 2),
                    "pnl_pips": round(t.pnl_pips, 2),
                    "outcome": t.outcome,
                }
                for t in metrics.trades
            ],
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        raise SystemExit(0)


if __name__ == "__main__":
    main()
