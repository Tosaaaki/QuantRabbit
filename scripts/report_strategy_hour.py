#!/usr/bin/env python3
"""
Report strategy performance by time-of-day buckets with optional dimensions.

Notes:
- Time bucket can be based on entry(open_time) or exit(close_time).
- Supports N-hour buckets (e.g. 3h buckets: 0,3,6,...,21).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def _parse_thesis(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _parse_time(value: str) -> dt.datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return dt.datetime.fromisoformat(text)


def _load_df(
    db_path: Path,
    *,
    lookback_days: int,
    time_col: str,
    time_from: str,
    time_to: str,
    include_open: bool,
) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    uri = f"file:{db_path}?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=8.0, isolation_level=None)
    try:
        if time_col not in ("open_time", "close_time"):
            raise ValueError(f"Invalid time_col: {time_col}")

        where = [f"{time_col} IS NOT NULL"]
        params: list[Any] = []
        if not include_open:
            where.append("close_time IS NOT NULL")
        if time_from:
            where.append(f"datetime({time_col}) >= datetime(?)")
            params.append(time_from)
        if time_to:
            where.append(f"datetime({time_col}) < datetime(?)")
            params.append(time_to)
        if not time_from and not time_to:
            where.append(f"datetime({time_col}) >= datetime('now', ?)")
            params.append(f"-{int(lookback_days)} day")

        where_sql = " AND ".join(where)
        query = """
        SELECT open_time, close_time, close_reason, state, pocket,
               strategy_tag, strategy, units, pl_pips, realized_pl,
               commission, financing, entry_thesis
        FROM trades
        WHERE {where_sql}
        """
        return pd.read_sql_query(
            query.format(where_sql=where_sql),
            con,
            params=tuple(params),
            parse_dates=["open_time", "close_time"],
        )
    finally:
        con.close()


def _coerce_jpy(df: pd.DataFrame) -> pd.Series:
    realized = df.get("realized_pl")
    commission = df.get("commission")
    financing = df.get("financing")
    if realized is not None and realized.notna().any():
        base = realized.fillna(0.0)
        if commission is not None:
            base = base + commission.fillna(0.0)
        if financing is not None:
            base = base + financing.fillna(0.0)
        return base
    units = df.get("units")
    if units is None:
        return pd.Series(0.0, index=df.index)
    return df["pl_pips"].astype(float) * units.abs().astype(float) * 0.01


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="logs/trades.db", help="Path to trades.db")
    ap.add_argument("--lookback-days", type=int, default=14, help="Lookback window in days")
    ap.add_argument(
        "--time-col",
        choices=("open_time", "close_time"),
        default="close_time",
        help="Bucket by entry(open_time) or exit(close_time) time (default: close_time)",
    )
    ap.add_argument(
        "--time-from",
        default="",
        help="Optional lower bound (inclusive) for time-col (ISO8601 with timezone)",
    )
    ap.add_argument(
        "--time-to",
        default="",
        help="Optional upper bound (exclusive) for time-col (ISO8601 with timezone)",
    )
    ap.add_argument(
        "--tz",
        default="UTC",
        help="Timezone for time-of-day bucket (default: UTC). Use 'JST' for Asia/Tokyo.",
    )
    ap.add_argument(
        "--bucket-hours",
        type=int,
        default=1,
        help="Bucket size in hours (default: 1). Example: 3 => 0,3,6,...,21",
    )
    ap.add_argument(
        "--include-open",
        action="store_true",
        help="Include OPEN trades (NOTE: unrealized metrics are not computed here)",
    )
    ap.add_argument("--min-trades", type=int, default=8, help="Min trades per group")
    ap.add_argument("--by-pocket", action="store_true", help="Group by pocket")
    ap.add_argument("--by-source", action="store_true", help="Group by decision source")
    ap.add_argument("--by-range", action="store_true", help="Group by range_active flag")
    ap.add_argument("--top", type=int, default=60, help="Max rows to display")
    args = ap.parse_args()

    tz = str(args.tz).strip() or "UTC"
    if tz.upper() == "JST":
        tz = "Asia/Tokyo"
    bucket_hours = int(args.bucket_hours)
    if bucket_hours <= 0 or bucket_hours > 24:
        raise SystemExit("--bucket-hours must be 1..24")
    if 24 % bucket_hours != 0:
        raise SystemExit("--bucket-hours must divide 24 (e.g. 1,2,3,4,6,8,12,24)")

    time_from = ""
    time_to = ""
    if args.time_from:
        time_from = _parse_time(args.time_from).astimezone(dt.timezone.utc).isoformat()
    if args.time_to:
        time_to = _parse_time(args.time_to).astimezone(dt.timezone.utc).isoformat()

    df = _load_df(
        Path(args.db),
        lookback_days=args.lookback_days,
        time_col=args.time_col,
        time_from=time_from,
        time_to=time_to,
        include_open=bool(args.include_open),
    )
    if df.empty:
        print("[report] no trades")
        return

    df["strategy"] = (
        df.get("strategy_tag")
        .fillna("")
        .replace("", pd.NA)
        .fillna(df.get("strategy"))
        .fillna("unknown")
    )
    ts = df[args.time_col]
    if ts.isna().any():
        df = df[ts.notna()].copy()
        ts = df[args.time_col]
    df["bucket_tz"] = ts.dt.tz_convert(tz)
    df["bucket_hour"] = (df["bucket_tz"].dt.hour // bucket_hours) * bucket_hours
    df["pl_pips"] = df["pl_pips"].fillna(0.0).astype(float)
    df["jpy"] = _coerce_jpy(df)
    df["hold_sec"] = (df["close_time"] - df["open_time"]).dt.total_seconds()

    decision_meta = df["entry_thesis"].map(_parse_thesis).map(
        lambda t: t.get("decision_meta") if isinstance(t, dict) else {}
    )
    df["decision_source"] = decision_meta.map(
        lambda d: (d or {}).get("source") if isinstance(d, dict) else None
    ).fillna("unknown")
    range_active_raw = decision_meta.map(
        lambda d: (d or {}).get("range_active") if isinstance(d, dict) else None
    )
    df["range_active"] = range_active_raw.astype("boolean").fillna(False).astype(bool)

    df["win_pips"] = df["pl_pips"].clip(lower=0.0)
    df["loss_pips"] = (-df["pl_pips"]).clip(lower=0.0)
    df["win_flag"] = (df["pl_pips"] > 0).astype(int)
    df["sl_flag"] = (df.get("close_reason") == "STOP_LOSS_ORDER").astype(int)
    df["tp_flag"] = (df.get("close_reason") == "TAKE_PROFIT_ORDER").astype(int)
    df["market_close_flag"] = (df.get("close_reason") == "MARKET_ORDER_TRADE_CLOSE").astype(int)

    group_cols = ["strategy", "bucket_hour"]
    if args.by_pocket:
        group_cols.append("pocket")
    if args.by_source:
        group_cols.append("decision_source")
    if args.by_range:
        group_cols.append("range_active")

    grouped = df.groupby(group_cols, dropna=False)
    agg = grouped.agg(
        trades=("pl_pips", "count"),
        wins=("win_flag", "sum"),
        sl=("sl_flag", "sum"),
        tp=("tp_flag", "sum"),
        market_close=("market_close_flag", "sum"),
        sum_pips=("pl_pips", "sum"),
        sum_jpy=("jpy", "sum"),
        avg_pips=("pl_pips", "mean"),
        avg_hold_sec=("hold_sec", "mean"),
        win_pips=("win_pips", "sum"),
        loss_pips=("loss_pips", "sum"),
    )
    agg["win_rate"] = (agg["wins"] / agg["trades"]).round(3)
    loss = agg["loss_pips"].astype(float).replace(0.0, float("nan"))
    agg["pf"] = (agg["win_pips"].astype(float) / loss).round(3)
    agg["sl_rate"] = (agg["sl"] / agg["trades"]).round(3)
    agg["tp_rate"] = (agg["tp"] / agg["trades"]).round(3)
    agg["market_close_rate"] = (agg["market_close"] / agg["trades"]).round(3)
    agg = agg.reset_index()
    agg = agg[agg["trades"] >= int(args.min_trades)]
    agg = agg.sort_values(["sum_jpy", "sum_pips"], ascending=False).head(int(args.top))

    cols = group_cols + [
        "trades",
        "win_rate",
        "pf",
        "sl_rate",
        "tp_rate",
        "market_close_rate",
        "sum_pips",
        "avg_pips",
        "avg_hold_sec",
        "sum_jpy",
    ]
    print(agg[cols].to_string(index=False))


if __name__ == "__main__":
    main()
