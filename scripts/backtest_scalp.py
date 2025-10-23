#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scalp strategy backtester with JSON output and parameter overrides.

Examples:
  python scripts/backtest_scalp.py --candles logs/candles_M1_20251022.json
  python scripts/backtest_scalp.py --candles logs/candles_M1_20251022.json \
        --strategies M1Scalper,PulseBreak \
        --params-json configs/scalp_active_params.json \
        --json-out logs/tuning/sample.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
import math

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.scalping.m1_scalper import M1Scalper
from strategies.scalping.range_fader import RangeFader
from strategies.scalping.pulse_break import PulseBreak
from strategies.mean_reversion.bb_rsi import BBRsi
from analysis.ma_projection import compute_ma_projection
from strategies.trend.ma_cross import MovingAverageCross
from strategies.breakout.donchian55 import Donchian55


PIP_VALUE = 0.01
DEFAULT_TIMEOUT_SEC = 30 * 60  # 30 minutes
DEFAULT_PARAM_FILE = REPO_ROOT / "configs" / "scalp_active_params.json"

TIMEFRAME_RULES = {
    "M1": None,
    "M5": "5min",
    "M15": "15min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1d",
}

STRATEGY_MAP = {
    M1Scalper.name: M1Scalper,
    RangeFader.name: RangeFader,
    PulseBreak.name: PulseBreak,
    BBRsi.name: BBRsi,
    MovingAverageCross.name: MovingAverageCross,
    Donchian55.name: Donchian55,
}


def parse_time(ts: str) -> datetime:
    ts = ts.rstrip("Z")
    if "." in ts:
        head, frac = ts.split(".", 1)
        frac = (frac + "000000")[:6]
        ts = f"{head}.{frac}+00:00"
    else:
        ts = ts + "+00:00"
    return datetime.fromisoformat(ts)


def _resample_dataframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    rule = TIMEFRAME_RULES.get(timeframe)
    if not rule:
        return df

    if "time" not in df.columns:
        return df

    resampled = (
        df.set_index("time")
        .resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    )
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])
    resampled.reset_index(inplace=True)
    return resampled


def load_candles(path: Path, timeframe: str = "M1") -> pd.DataFrame:
    with path.open() as f:
        payload = json.load(f)

    candles = payload.get("candles", payload)
    rows = []
    for c in candles:
        mid = c.get("mid") or {}
        rows.append(
            {
                "time": parse_time(c["time"]),
                "open": float(mid.get("o", mid.get("open", 0.0))),
                "high": float(mid.get("h", mid.get("high", 0.0))),
                "low": float(mid.get("l", mid.get("low", 0.0))),
                "close": float(mid.get("c", mid.get("close", 0.0))),
            }
        )

    df = pd.DataFrame(rows)
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return _resample_dataframe(df, timeframe)


def compute_indicators(df: pd.DataFrame, *, timeframe: str = "M1") -> pd.DataFrame:
    prices = df["close"]
    df["ema20"] = prices.ewm(span=20, adjust=False, min_periods=20).mean()
    df["ema50"] = prices.ewm(span=50, adjust=False, min_periods=50).mean()
    df["ema100"] = prices.ewm(span=100, adjust=False, min_periods=100).mean()
    df["ma10"] = prices.rolling(window=10, min_periods=10).mean()
    df["ma20"] = prices.rolling(window=20, min_periods=20).mean()
    df["ma50"] = prices.rolling(window=50, min_periods=50).mean()

    std20 = prices.rolling(window=20, min_periods=20).std()
    # Bollinger band width relative to moving average
    df["bbw"] = (
        (std20 * 4).div(df["ma20"].abs().replace(0.0, np.nan))
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    df["rsi"] = (100 - (100 / (1 + rs))).fillna(50.0)

    prev_close = prices.shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().fillna(0.0)
    df["atr"] = atr
    df["atr_pips"] = df["atr"] / PIP_VALUE

    df["adx"] = _compute_adx(df["high"], df["low"], prices)

    pip_move = prices.diff().abs() / PIP_VALUE
    df["vol_5m"] = pip_move.rolling(window=5, min_periods=5).mean().fillna(0.0)

    return df


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_di = (
        pd.Series(plus_dm, index=high.index)
        .ewm(alpha=1 / period, adjust=False, min_periods=period)
        .mean()
    )
    minus_di = (
        pd.Series(minus_dm, index=low.index)
        .ewm(alpha=1 / period, adjust=False, min_periods=period)
        .mean()
    )
    atr_safe = atr.replace(0.0, np.nan)
    plus_di = 100 * (plus_di / atr_safe)
    minus_di = 100 * (minus_di / atr_safe)

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    dx = dx.fillna(0.0) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx.fillna(0.0)


def ensure_ema(df: pd.DataFrame, periods: Iterable[int]) -> None:
    prices = df["close"]
    for period in set(periods):
        if period <= 0:
            continue
        col = f"ema{period}"
        if col in df.columns:
            continue
        df[col] = prices.ewm(span=period, adjust=False, min_periods=period).mean()


@dataclass
class Trade:
    strategy: str
    side: str
    entry_index: int
    entry_time: datetime
    entry_price: float
    tp_price: float
    sl_price: float
    timeout_sec: int
    timeframe: str
    outcome: Optional[str] = None
    exit_index: Optional[int] = None
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl_pips: float = 0.0


def load_params(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Parameter file must contain a JSON object")
    return {str(k): v for k, v in data.items() if isinstance(v, dict)}


def merge_params(
    base: Dict[str, Dict[str, Any]], overrides: Optional[Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {k: dict(v) for k, v in base.items()}
    if overrides:
        for name, params in overrides.items():
            merged.setdefault(name, {})
            merged[name].update(params)
    return merged


def select_strategies(names: Optional[List[str]]) -> List[Any]:
    if not names:
        return list(STRATEGY_MAP.values())
    selected = []
    for name in names:
        if name not in STRATEGY_MAP:
            raise ValueError(f"Unknown strategy: {name}")
        selected.append(STRATEGY_MAP[name])
    return selected


def should_skip_by_params(
    strat_name: str, params: Dict[str, Any], fac: Dict[str, float], row: pd.Series
) -> bool:
    # Generic volatility / ATR caps
    vol = fac.get("vol_5m")
    if "vol_5m_max" in params and vol is not None and vol > float(params["vol_5m_max"]):
        return True
    if "vol_5m_min" in params and vol is not None and vol < float(params["vol_5m_min"]):
        return True
    atr_pips = fac.get("atr_pips")
    if "atr_pips_max" in params and atr_pips is not None and atr_pips > float(params["atr_pips_max"]):
        return True
    if "atr_pips_min" in params and atr_pips is not None and atr_pips < float(params["atr_pips_min"]):
        return True

    # trend_ema parameter (mainly for PulseBreak)
    if "trend_ema" in params:
        period = int(params["trend_ema"])
        col = f"ema{period}"
        if col in row and not pd.isna(row[col]):
            ema_short = float(row["ema20"]) if "ema20" in row else float(row[col])
            ema_trend = float(row[col])
            if abs(ema_short - ema_trend) < 0.0008:
                return True

    bbw = fac.get("bbw")
    if "bbw_min" in params and bbw is not None and bbw < float(params["bbw_min"]):
        return True
    if "bbw_max" in params and bbw is not None and bbw > float(params["bbw_max"]):
        return True

    adx = fac.get("adx")
    if "adx_min" in params and adx is not None and adx < float(params["adx_min"]):
        return True
    if "adx_max" in params and adx is not None and adx > float(params["adx_max"]):
        return True

    ma10 = fac.get("ma10")
    ma20 = fac.get("ma20")
    ma_gap_pips = None
    if ma10 is not None and ma20 is not None:
        ma_gap_pips = abs(ma10 - ma20) / PIP_VALUE

    if "ma_gap_min_pips" in params:
        if ma_gap_pips is None or ma_gap_pips < float(params["ma_gap_min_pips"]):
            return True
    if "ma_gap_max_pips" in params:
        if ma_gap_pips is None or ma_gap_pips > float(params["ma_gap_max_pips"]):
            return True

    if "ma_spread_min" in params or "ma_spread_max" in params:
        if ma10 is None or ma20 is None or ma20 == 0:
            return True
        spread = abs(ma10 - ma20) / abs(ma20)
        if "ma_spread_min" in params and spread < float(params["ma_spread_min"]):
            return True
        if "ma_spread_max" in params and spread > float(params["ma_spread_max"]):
            return True

    price_to_fast = fac.get("price_to_fast_pips")
    if "price_to_fast_max_pips" in params and price_to_fast is not None:
        if abs(price_to_fast) > float(params["price_to_fast_max_pips"]):
            return True

    projected_cross = fac.get("projected_cross_minutes")
    if "projected_cross_minutes_max" in params and projected_cross is not None:
        if projected_cross <= 0 or projected_cross < float(params["projected_cross_minutes_max"]):
            return True

    gap_slope = fac.get("gap_slope_pips")
    min_slope = params.get("gap_slope_min_pips")
    if min_slope is not None and gap_slope is not None:
        if gap_slope < float(min_slope):
            return True

    return False


def apply_signal_overrides(
    strat_name: str,
    signal: Dict[str, Any],
    params: Dict[str, Any],
    fac: Dict[str, float],
) -> Optional[Tuple[float, float, int]]:
    sl = float(signal.get("sl_pips") or 0.0)
    tp = float(signal.get("tp_pips") or 0.0)
    if params:
        if "sl_pips" in params:
            sl = float(params["sl_pips"])
        if "tp_pips" in params:
            tp = float(params["tp_pips"])
    if sl <= 0 or tp <= 0:
        return None
    timeout_sec = int(params.get("timeout_sec", DEFAULT_TIMEOUT_SEC))

    # Strategy specific filters
    rsi = fac.get("rsi")
    action = signal.get("action")
    if strat_name == M1Scalper.name and rsi is not None:
        low = params.get("rsi_entry_low")
        high = params.get("rsi_entry_high")
        if action == "OPEN_LONG" and low is not None and rsi > float(low):
            return None
        if action == "OPEN_SHORT" and high is not None and rsi < float(high):
            return None

    if strat_name == RangeFader.name and rsi is not None:
        lower = params.get("rsi_lower")
        upper = params.get("rsi_upper")
        if action == "OPEN_LONG" and lower is not None and rsi > float(lower):
            return None
        if action == "OPEN_SHORT" and upper is not None and rsi < float(upper):
            return None

    if strat_name == BBRsi.name:
        tp *= float(params.get("tp_scale", 1.0))
        sl *= float(params.get("sl_scale", 1.0))
        if rsi is not None:
            lower = params.get("rsi_lower")
            upper = params.get("rsi_upper")
            if action == "OPEN_LONG" and lower is not None and rsi > float(lower):
                return None
            if action == "OPEN_SHORT" and upper is not None and rsi < float(upper):
                return None

    if strat_name == Donchian55.name:
        buffer = params.get("breakout_buffer_pips")
        price = fac.get("close")
        if buffer is not None and price is not None:
            high_val = fac.get("donchian_high")
            low_val = fac.get("donchian_low")
            buffer_px = float(buffer) * PIP_VALUE
            if action == "OPEN_LONG" and high_val is not None and price < high_val + buffer_px:
                return None
            if action == "OPEN_SHORT" and low_val is not None and price > low_val - buffer_px:
                return None
        tp_factor = params.get("tp_factor")
        if tp_factor is not None and sl > 0:
            tp = float(tp_factor) * sl

    if strat_name == MovingAverageCross.name:
        spread_boost = params.get("ma_spread_scale")
        if spread_boost is not None:
            ma10 = fac.get("ma10")
            ma20 = fac.get("ma20")
            if ma10 is not None and ma20 is not None and ma20 != 0:
                spread = abs(ma10 - ma20) / abs(ma20)
                tp *= 1.0 + spread * float(spread_boost)

    return sl, tp, timeout_sec


def simulate(
    df: pd.DataFrame,
    strategies: List[Any],
    params_per_strategy: Dict[str, Dict[str, Any]],
    *,
    timeframe: str,
) -> Dict[str, List[Trade]]:
    open_trades: Dict[str, List[Trade]] = defaultdict(list)
    closed_trades: Dict[str, List[Trade]] = defaultdict(list)
    fac_keys = [
        "open",
        "close",
        "ema20",
        "ema50",
        "ema100",
        "ma10",
        "ma20",
        "ma50",
        "bbw",
        "adx",
        "rsi",
        "atr",
        "atr_pips",
        "vol_5m",
    ]

    for idx, row in df.iterrows():
        # Update open trades
        for strat_name, trades in list(open_trades.items()):
            remaining = []
            for trd in trades:
                if idx <= trd.entry_index:
                    remaining.append(trd)
                    continue
                candle_high = float(row["high"])
                candle_low = float(row["low"])
                exit_price = None
                exit_reason = None

                if trd.side == "LONG":
                    if candle_high >= trd.tp_price:
                        exit_price = trd.tp_price
                        exit_reason = "TP"
                    elif candle_low <= trd.sl_price:
                        exit_price = trd.sl_price
                        exit_reason = "SL"
                else:
                    if candle_low <= trd.tp_price:
                        exit_price = trd.tp_price
                        exit_reason = "TP"
                    elif candle_high >= trd.sl_price:
                        exit_price = trd.sl_price
                        exit_reason = "SL"

                if exit_price is None and trd.timeout_sec > 0:
                    if (row["time"] - trd.entry_time) >= timedelta(seconds=trd.timeout_sec):
                        exit_price = float(row["close"])
                        exit_reason = "TIME"

                if exit_price is None:
                    remaining.append(trd)
                    continue

                trd.exit_index = idx
                trd.exit_time = row["time"]
                trd.exit_price = exit_price
                trd.outcome = exit_reason
                direction = 1 if trd.side == "LONG" else -1
                trd.pnl_pips = (exit_price - trd.entry_price) * direction / PIP_VALUE
                closed_trades[strat_name].append(trd)
            open_trades[strat_name] = remaining

        # Prepare factors for new signals
        fac = {k: float(row[k]) for k in fac_keys if k in row and not pd.isna(row[k])}
        fac.setdefault("atr_pips", float(row.get("atr_pips", 0.0)))
        fac.setdefault("vol_5m", float(row.get("vol_5m", 0.0)))
        fac.setdefault("timeframe", timeframe)

        context_window = 120 if timeframe in {"H1", "H4", "D1"} else 80
        start_idx = max(0, idx - context_window + 1)
        ctx_df = df.iloc[start_idx : idx + 1]
        candles_list: List[Dict[str, float]] = []
        for item in ctx_df.itertuples(index=False):
            candles_list.append(
                {
                    "open": float(item.open),
                    "high": float(item.high),
                    "low": float(item.low),
                    "close": float(item.close),
                }
            )
        fac["candles"] = candles_list[-60:]
        if len(candles_list) >= 56:
            history = candles_list[:-1]
            fac["donchian_high"] = max(c["high"] for c in history)
            fac["donchian_low"] = min(c["low"] for c in history)
        else:
            fac["donchian_high"] = None
            fac["donchian_low"] = None

        projection = compute_ma_projection({"candles": fac["candles"]}, timeframe_minutes=1.0)
        if projection:
            fac["ma_gap_pips"] = projection.gap_pips
            fac["gap_slope_pips"] = projection.gap_slope_pips
            fac["price_to_fast_pips"] = projection.price_to_fast_pips
            fac["price_to_slow_pips"] = projection.price_to_slow_pips
            fac["projected_cross_minutes"] = projection.projected_cross_minutes
            fac["macd_cross_minutes"] = projection.macd_cross_minutes

        for cls in strategies:
            strat_name = cls.name
            params = params_per_strategy.get(strat_name, {})

            if should_skip_by_params(strat_name, params, fac, row):
                continue

            signal = cls.check(fac)
            if not signal:
                continue
            if open_trades[strat_name]:
                continue

            action = signal.get("action")
            if action not in {"OPEN_LONG", "OPEN_SHORT"}:
                continue

            overrides = apply_signal_overrides(strat_name, signal, params, fac)
            if overrides is None:
                continue
            sl, tp, timeout_sec = overrides
            if sl <= 0 or tp <= 0:
                continue

            entry_price = float(row["close"])
            if action == "OPEN_LONG":
                sl_price = entry_price - sl * PIP_VALUE
                tp_price = entry_price + tp * PIP_VALUE
                side = "LONG"
            else:
                sl_price = entry_price + sl * PIP_VALUE
                tp_price = entry_price - tp * PIP_VALUE
                side = "SHORT"

            trade = Trade(
                strategy=strat_name,
                side=side,
                entry_index=idx,
                entry_time=row["time"],
                entry_price=entry_price,
                tp_price=tp_price,
                sl_price=sl_price,
                timeout_sec=timeout_sec,
                timeframe=timeframe,
            )
            open_trades[strat_name].append(trade)

    # Close remaining trades at final close
    final_time = df.iloc[-1]["time"]
    final_close = float(df.iloc[-1]["close"])
    for strat, trades in open_trades.items():
        for trd in trades:
            trd.exit_index = len(df) - 1
            trd.exit_time = final_time
            trd.exit_price = final_close
            trd.outcome = "EOD"
            direction = 1 if trd.side == "LONG" else -1
            trd.pnl_pips = (final_close - trd.entry_price) * direction / PIP_VALUE
            closed_trades[strat].append(trd)

    return closed_trades


def calc_profit_factor(pnl: List[float]) -> float:
    gains = sum(p for p in pnl if p > 0)
    losses = -sum(p for p in pnl if p < 0)
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def calc_max_drawdown(trades: List[Trade]) -> float:
    if not trades:
        return 0.0
    trades_sorted = sorted(trades, key=lambda t: (t.exit_time or t.entry_time))
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for trd in trades_sorted:
        cum += trd.pnl_pips
        if cum > peak:
            peak = cum
        drawdown = peak - cum
        if drawdown > max_dd:
            max_dd = drawdown
    return round(max_dd, 2)


def summarise_trades(trades: Dict[str, List[Trade]]) -> Dict[str, Any]:
    all_trades = [trade for items in trades.values() for trade in items]
    pnl = [tr.pnl_pips for tr in all_trades]
    pf = calc_profit_factor(pnl)
    def _pf_value(pf_val: float) -> float:
        if pf_val == float("inf") or math.isinf(pf_val):
            return float("inf")
        return round(pf_val, 4)

    summary = {
        "profit_pips": round(sum(pnl), 2),
        "trades": len(all_trades),
        "win_rate": round(sum(1 for p in pnl if p > 0) / len(pnl), 4) if pnl else 0.0,
        "profit_factor": _pf_value(pf) if pnl else 0.0,
        "max_dd_pips": calc_max_drawdown(all_trades),
    }

    by_strategy = {}
    for strat, items in trades.items():
        strat_pnl = [tr.pnl_pips for tr in items]
        pf_strat = calc_profit_factor(strat_pnl)
        by_strategy[strat] = {
            "profit_pips": round(sum(strat_pnl), 2),
            "trades": len(items),
            "win_rate": round(
                sum(1 for p in strat_pnl if p > 0) / len(strat_pnl), 4
            )
            if strat_pnl
            else 0.0,
            "profit_factor": _pf_value(pf_strat) if strat_pnl else 0.0,
            "max_dd_pips": calc_max_drawdown(items),
        }
    return {"summary": summary, "by_strategy": by_strategy}


def trades_to_dict(trades: Dict[str, List[Trade]]) -> List[Dict[str, Any]]:
    out = []
    for strat, items in trades.items():
        for t in items:
            data = asdict(t)
            data["strategy"] = strat
            data["entry_time"] = t.entry_time.isoformat()
            data["exit_time"] = t.exit_time.isoformat() if t.exit_time else None
            out.append(data)
    return out


def run_backtest(
    candles_path: str,
    params_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    strategies: Optional[List[str]] = None,
    timeframe: str = "M1",
) -> Dict[str, Any]:
    candles_file = Path(candles_path)
    df = load_candles(candles_file, timeframe=timeframe)
    df = compute_indicators(df, timeframe=timeframe)

    active_params = load_params(DEFAULT_PARAM_FILE if DEFAULT_PARAM_FILE.exists() else None)
    overrides = params_overrides or {}
    merged_params = merge_params(active_params, overrides)

    selected_strategies = select_strategies(strategies)

    # Precompute any EMA lengths requested via trend_ema
    ema_periods = []
    for strat in selected_strategies:
        params = merged_params.get(strat.name, {})
        if "trend_ema" in params:
            ema_periods.append(int(params["trend_ema"]))
    ensure_ema(df, ema_periods)

    trades = simulate(df, selected_strategies, merged_params, timeframe=timeframe)
    metrics = summarise_trades(trades)
    stem = candles_file.stem
    if stem.startswith("candles_"):
        candle_date = stem.split("_", 2)[-1]
    else:
        candle_date = stem
    result = {
        "date": candle_date,
        "timeframe": timeframe,
        "summary": metrics["summary"],
        "by_strategy": metrics["by_strategy"],
        "params_used": {k: merged_params.get(k, {}) for k in metrics["by_strategy"].keys()},
        "trades": trades_to_dict(trades),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay scalp strategies on candle data.")
    parser.add_argument("--candles", required=True, help="logs/candles_M1_YYYYMMDD.json")
    parser.add_argument("--params-json", default="", help="Parameter override JSON file")
    parser.add_argument(
        "--strategies",
        default="",
        help="Comma separated strategy names (e.g. M1Scalper,PulseBreak)",
    )
    parser.add_argument("--json-out", default="", help="Output JSON path")
    parser.add_argument(
        "--timeframe",
        default="M1",
        choices=sorted(TIMEFRAME_RULES.keys()),
        help="Timeframe to evaluate (default: M1)",
    )
    args = parser.parse_args()

    overrides = load_params(Path(args.params_json)) if args.params_json else {}
    strat_list = [s for s in args.strategies.split(",") if s] if args.strategies else None
    result = run_backtest(args.candles, overrides, strat_list, timeframe=args.timeframe)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    summary = result["summary"]
    print(
        json.dumps(
            {
                "date": result["date"],
                "profit_pips": summary["profit_pips"],
                "trades": summary["trades"],
                "win_rate": summary["win_rate"],
                "profit_factor": summary["profit_factor"],
                "max_dd_pips": summary["max_dd_pips"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
