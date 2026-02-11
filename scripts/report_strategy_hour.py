#!/usr/bin/env python3
"""
Report strategy performance by time bucket with optional dimensions.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict

import numpy as np
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


def _load_df(
    db_path: Path,
    lookback_days: int,
    lookback_hours: int | None,
    time_col: str,
    instrument: str | None,
) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    time_col = "close_time" if time_col != "open_time" else "open_time"
    uri = f"file:{db_path}?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=8.0, isolation_level=None)
    try:
        lookback_expr = (
            f"-{int(lookback_hours)} hour"
            if lookback_hours is not None
            else f"-{int(lookback_days)} day"
        )
        instrument_clause = " AND instrument = :instrument " if instrument else ""
        query = """
        SELECT close_time, open_time, instrument, pocket, strategy_tag, strategy, units,
               pl_pips, realized_pl, commission, financing, entry_thesis
        FROM trades
        WHERE {time_col} IS NOT NULL
          AND {time_col} >= datetime('now', :lookback)
          {instrument_clause}
        """
        query = query.format(time_col=time_col, instrument_clause=instrument_clause)
        params: Dict[str, Any] = {"lookback": lookback_expr}
        if instrument:
            params["instrument"] = instrument
        return pd.read_sql_query(
            query,
            con,
            params=params,
            parse_dates=["close_time", "open_time"],
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
    ap.add_argument("--lookback-hours", type=int, default=None, help="Lookback window in hours")
    ap.add_argument(
        "--time-col",
        choices=["close_time", "open_time"],
        default="close_time",
        help="Timestamp column used for lookback and hour bucket",
    )
    ap.add_argument("--instrument", default=None, help="Filter instrument (e.g. USD_JPY)")
    ap.add_argument(
        "--bucket-timezone",
        default="UTC",
        help="Timezone name for hour bucket (e.g. UTC, Asia/Tokyo)",
    )
    ap.add_argument(
        "--bucket-shift-hours",
        type=int,
        default=0,
        help="Additional hour shift applied after timezone conversion",
    )
    ap.add_argument(
        "--sort",
        choices=["best", "worst"],
        default="best",
        help="best: descending performance / worst: ascending performance",
    )
    ap.add_argument("--min-trades", type=int, default=8, help="Min trades per group")
    ap.add_argument("--by-pocket", action="store_true", help="Group by pocket")
    ap.add_argument("--by-source", action="store_true", help="Group by decision source")
    ap.add_argument("--by-range", action="store_true", help="Group by range_active flag")
    ap.add_argument("--top", type=int, default=60, help="Max rows to display")
    args = ap.parse_args()

    df = _load_df(
        Path(args.db),
        args.lookback_days,
        args.lookback_hours,
        args.time_col,
        args.instrument,
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
    time_series = pd.to_datetime(df[args.time_col], utc=True, errors="coerce")
    df = df.loc[time_series.notna()].copy()
    if df.empty:
        print("[report] no valid timestamp rows")
        return
    try:
        bucket_time = time_series.loc[time_series.notna()].dt.tz_convert(args.bucket_timezone)
    except Exception as exc:
        raise SystemExit(f"[report] invalid --bucket-timezone: {args.bucket_timezone}: {exc}")
    if args.bucket_shift_hours:
        bucket_time = bucket_time + pd.to_timedelta(int(args.bucket_shift_hours), unit="h")
    df["bucket_hour"] = bucket_time.dt.hour
    df["pl_pips"] = df["pl_pips"].fillna(0.0).astype(float)
    df["jpy"] = _coerce_jpy(df)

    decision_meta = df["entry_thesis"].map(_parse_thesis).map(
        lambda t: t.get("decision_meta") if isinstance(t, dict) else {}
    )
    df["decision_source"] = decision_meta.map(
        lambda d: (d or {}).get("source") if isinstance(d, dict) else None
    ).fillna("unknown")
    df["range_active"] = decision_meta.map(
        lambda d: (d or {}).get("range_active") if isinstance(d, dict) else None
    ).fillna(False)

    df["win_pips"] = df["pl_pips"].clip(lower=0.0)
    df["loss_pips"] = (-df["pl_pips"]).clip(lower=0.0)
    df["win_flag"] = (df["pl_pips"] > 0).astype(int)

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
        sum_pips=("pl_pips", "sum"),
        sum_jpy=("jpy", "sum"),
        avg_pips=("pl_pips", "mean"),
        win_pips=("win_pips", "sum"),
        loss_pips=("loss_pips", "sum"),
    )
    agg["win_rate"] = (agg["wins"] / agg["trades"]).round(3)
    agg["pf"] = (agg["win_pips"] / agg["loss_pips"].replace(0.0, np.nan)).round(3)
    agg = agg.reset_index()
    agg = agg[agg["trades"] >= int(args.min_trades)]
    if args.sort == "worst":
        agg = agg.sort_values(["sum_jpy", "sum_pips"], ascending=True).head(int(args.top))
    else:
        agg = agg.sort_values(["sum_jpy", "sum_pips"], ascending=False).head(int(args.top))

    cols = group_cols + [
        "trades",
        "win_rate",
        "pf",
        "loss_pips",
        "sum_pips",
        "avg_pips",
        "sum_jpy",
    ]
    print(agg[cols].to_string(index=False))


if __name__ == "__main__":
    main()
