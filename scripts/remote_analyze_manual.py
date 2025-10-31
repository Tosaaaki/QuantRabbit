#!/usr/bin/env python3
"""
Deep analysis for today's manual trades using factor_cache.json and local indicators.

Run on the VM:
  cd ~/QuantRabbit && source .venv/bin/activate && python scripts/remote_analyze_manual.py
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

import sys
BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
from indicators.calc_core import IndicatorEngine


DB_PATH = Path("/home/tossaki/QuantRabbit/logs/trades.db")
FACTOR_JSON = Path("/home/tossaki/QuantRabbit/logs/factor_cache.json")


def _to_dt(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@dataclass
class Trade:
    ticket_id: str
    pocket: str
    units: int
    entry_price: float
    close_price: float
    pl_pips: float
    entry_time: datetime
    close_time: datetime


def load_manual_trades_today(limit: int = 10) -> List[Trade]:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT ticket_id, pocket, units, entry_price, close_price, pl_pips, entry_time, close_time
        FROM trades
        WHERE pocket='manual'
          AND state='CLOSED'
          AND DATE(close_time)=DATE('now')
        ORDER BY datetime(close_time) ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    con.close()
    out: List[Trade] = []
    for r in rows:
        out.append(
            Trade(
                ticket_id=str(r["ticket_id"]),
                pocket=str(r["pocket"]),
                units=int(r["units"] or 0),
                entry_price=float(r["entry_price"] or 0.0),
                close_price=float(r["close_price"] or 0.0),
                pl_pips=float(r["pl_pips"] or 0.0),
                entry_time=_to_dt(r["entry_time"]),
                close_time=_to_dt(r["close_time"]),
            )
        )
    return out


def _candles_to_df(candles: List[Dict]) -> pd.DataFrame:
    # candles item keys may be 'time' or 'timestamp'
    recs = []
    for c in candles:
        t = c.get("time") or c.get("timestamp")
        if not t:
            continue
        dt = _to_dt(str(t))
        recs.append(
            {
                "time": dt,
                "open": float(c.get("open", 0.0)),
                "high": float(c.get("high", 0.0)),
                "low": float(c.get("low", 0.0)),
                "close": float(c.get("close", 0.0)),
            }
        )
    if not recs:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close"]).set_index("time")
    df = pd.DataFrame(recs).sort_values("time").set_index("time")
    return df


def _nearest_index_at_or_before(df: pd.DataFrame, t: datetime) -> Optional[pd.Timestamp]:
    if df.empty:
        return None
    idx = df.index[df.index <= pd.Timestamp(t)].max()
    return idx if pd.notna(idx) else None


def compute_factors_at(df: pd.DataFrame, when: datetime) -> Optional[Dict[str, float]]:
    idx = _nearest_index_at_or_before(df, when)
    if idx is None:
        return None
    sub = df.loc[:idx]
    return IndicatorEngine.compute(sub)


def zscore_vs_bb(df: pd.DataFrame, when: datetime) -> Optional[float]:
    idx = _nearest_index_at_or_before(df, when)
    if idx is None:
        return None
    # reuse IndicatorEngine internals by recomputing bollinger window
    sub = df.loc[:idx]
    if len(sub) < 20:
        return None
    close = sub["close"].astype(float)
    mid = close.rolling(window=20, min_periods=20).mean().iloc[-1]
    std = close.rolling(window=20, min_periods=20).std(ddof=0).iloc[-1]
    if std == 0:
        return 0.0
    return float((close.iloc[-1] - mid) / std)


def _slice(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]


def mae_mfe(df: pd.DataFrame, trade: Trade) -> Dict[str, float]:
    seg = _slice(df, trade.entry_time, trade.close_time)
    if seg is None or seg.empty:
        return {"mae_pips": 0.0, "mfe_pips": 0.0}
    if trade.units < 0:  # short
        worst = float(seg["high"].max())
        best = float(seg["low"].min())
        mae = max(0.0, (worst - trade.entry_price) * 100.0)
        mfe = max(0.0, (trade.entry_price - best) * 100.0)
    else:  # long
        worst = float(seg["low"].min())
        best = float(seg["high"].max())
        mae = max(0.0, (trade.entry_price - worst) * 100.0)
        mfe = max(0.0, (best - trade.entry_price) * 100.0)
    return {"mae_pips": round(mae, 2), "mfe_pips": round(mfe, 2)}


def post_exit_extension(df: pd.DataFrame, trade: Trade, minutes: List[int]) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    if df is None or df.empty:
        return out
    close_ts = trade.close_time
    for m in minutes:
        end = close_ts + pd.Timedelta(minutes=m)
        seg = _slice(df, close_ts, end)
        if seg is None or seg.empty:
            out[m] = {"further_profit_pips": 0.0, "rebound_risk_pips": 0.0}
            continue
        if trade.units < 0:  # short
            future_min = float(seg["low"].min())
            future_max = float(seg["high"].max())
            further = max(0.0, (trade.close_price - future_min) * 100.0)
            rebound = max(0.0, (future_max - trade.close_price) * 100.0)
        else:  # long
            future_max = float(seg["high"].max())
            future_min = float(seg["low"].min())
            further = max(0.0, (future_max - trade.close_price) * 100.0)
            rebound = max(0.0, (trade.close_price - future_min) * 100.0)
        out[m] = {
            "further_profit_pips": round(further, 2),
            "rebound_risk_pips": round(rebound, 2),
        }
    return out


def main():
    trades = load_manual_trades_today(limit=50)
    if not trades:
        print("no trades")
        return
    js = json.loads(FACTOR_JSON.read_text())

    frames: Dict[str, pd.DataFrame] = {}
    for tf in ("M1", "M5", "H1", "H4"):
        v = js.get(tf) or {}
        candles = v.get("candles") or []
        frames[tf] = _candles_to_df(candles)

    def _fmt(tf: str, fac: Optional[Dict[str, float]], zbb: Optional[float]) -> str:
        if not fac:
            return f"{tf}: n/a"
        adx = fac.get("adx", 0.0)
        bbw = fac.get("bbw", 0.0)
        rsi = fac.get("rsi", 0.0)
        atr_pips = fac.get("atr", 0.0) * 100.0  # USDJPY 1 pip = 0.01
        ztxt = f", zBB={zbb:.2f}" if zbb is not None else ""
        return (
            f"{tf}: ADX={adx:.1f}, BBW={bbw:.3f}, RSI={rsi:.1f}, ATR={atr_pips:.1f}p{ztxt}"
        )

    for tr in trades:
        print("--- manual", tr.ticket_id, "units=", tr.units, "pl_pips=", tr.pl_pips)
        for label, t in (("entry", tr.entry_time), ("exit", tr.close_time)):
            print(label, t.isoformat())
            lines = []
            for tf in ("M1", "M5", "H1", "H4"):
                df = frames.get(tf)
                if df is None:
                    df = pd.DataFrame(columns=["time","open","high","low","close"]).set_index("time")
                fac = compute_factors_at(df, t)
                zbb = zscore_vs_bb(df, t) if tf in ("M1", "M5") else None
                lines.append(_fmt(tf, fac, zbb))
            print("  ".join(lines))

        # Detect range mode on M1 at entry
        m1 = frames.get("M1")
        f_entry = compute_factors_at(m1, tr.entry_time) if m1 is not None else None
        if f_entry:
            adx = f_entry.get("adx", 0.0)
            bbw = f_entry.get("bbw", 0.0)
            atr_pips = f_entry.get("atr", 0.0) * 100.0
            range_mode = (adx <= 22.0 and bbw <= 0.20 and atr_pips <= 6.0)
            print(
                f"range_mode@entry={range_mode} (ADX<={adx:.1f} BBW<={bbw:.2f} ATRp<={atr_pips:.1f})"
            )

        # In-trade excursions and post-exit continuation on M1
        m1 = frames.get("M1")
        if m1 is not None and not m1.empty:
            ex = mae_mfe(m1, tr)
            print(f"hold_excursion: MAE={ex['mae_pips']:.2f}p, MFE={ex['mfe_pips']:.2f}p")
            cont = post_exit_extension(m1, tr, [15, 30, 60])
            parts = []
            for k in (15, 30, 60):
                v = cont.get(k) or {"further_profit_pips": 0.0, "rebound_risk_pips": 0.0}
                parts.append(
                    f"+{k}m Δprofit={v['further_profit_pips']:.2f}p / rebound={v['rebound_risk_pips']:.2f}p"
                )
            print("post_exit:", " | ".join(parts))

    print("done")


if __name__ == "__main__":
    main()
