"""
analytics.worker_return_wait_report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compute per-strategy return-wait stats (hold time, win rate, avg pips, MFE/MAE, BE hits).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS = REPO_ROOT / "logs"
PIP = 0.01

_ALIAS_BASE = {
    "mlr": "MicroLevelReactor",
    "trendma": "TrendMA",
    "donchian": "Donchian55",
    "h1momentum": "H1Momentum",
    "m1scalper": "M1Scalper",
    "bbrsi": "BB_RSI",
    "bb_rsi": "BB_RSI",
}
_DEFAULT_REENTRY_CONFIG = {
    "cooldown_win_sec": 60,
    "cooldown_loss_sec": 180,
    "same_dir_reentry_pips": 1.8,
    "allow_jst_hours": [],
    "block_jst_hours": [],
    "return_wait_bias": "neutral",
}
_REENTRY_COOLDOWN_WIN_RANGE = (30, 300)
_REENTRY_COOLDOWN_LOSS_RANGE = (60, 900)
_REENTRY_PIPS_RANGE = (0.8, 8.0)


def _parse_iso(ts: Optional[str]) -> Optional[dt.datetime]:
    if not ts:
        return None
    raw = ts.strip()
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        if "." in raw and "+" in raw and raw.rfind("+") > raw.find("."):
            head, frac_plus = raw.split(".", 1)
            frac, plus = frac_plus.split("+", 1)
            frac_digits = "".join(ch for ch in frac if ch.isdigit())[:6]
            raw = f"{head}.{frac_digits}+{plus}"
        elif "." in raw and "+" not in raw:
            head, frac = raw.split(".", 1)
            frac_digits = "".join(ch for ch in frac if ch.isdigit())[:6]
            raw = f"{head}.{frac_digits}+00:00"
        elif "+" not in raw:
            raw = f"{raw}+00:00"
        return dt.datetime.fromisoformat(raw).astimezone(dt.timezone.utc)
    except Exception:
        try:
            trimmed = raw.split(".", 1)[0].rstrip("Z") + "+00:00"
            return dt.datetime.fromisoformat(trimmed).astimezone(dt.timezone.utc)
        except Exception:
            return None


def _resolve_strategy_tag(raw_tag: Optional[str], thesis_raw: Optional[str]) -> str:
    tag = (raw_tag or "").strip()
    if tag:
        return tag
    if thesis_raw:
        try:
            payload = json.loads(thesis_raw)
            if isinstance(payload, dict):
                tag = payload.get("strategy_tag") or payload.get("strategy")
        except Exception:
            tag = None
    return str(tag).strip() if tag else "unknown"


def _base_strategy_tag(tag: str) -> str:
    if not tag:
        return "unknown"
    base = tag.split("-", 1)[0].strip()
    if not base:
        base = tag
    alias = _ALIAS_BASE.get(base.lower())
    return alias or base


@dataclass
class Candle:
    ts: dt.datetime
    o: float
    h: float
    l: float
    c: float


def _load_m1_for_dates(dates: Iterable[dt.date]) -> Dict[dt.date, List[Candle]]:
    out: Dict[dt.date, List[Candle]] = {}
    for d in sorted(set(dates)):
        path = LOGS / f"candles_M1_{d.strftime('%Y%m%d')}.json"
        if not path.exists():
            agg = LOGS / "candles_M1_20251001_20251022.json"
            if agg.exists():
                path = agg
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        arr = data.get("candles") or []
        entries: List[Candle] = []
        for row in arr:
            t = _parse_iso(row.get("time"))
            if not t:
                continue
            mid = row.get("mid") or {}
            try:
                entries.append(
                    Candle(
                        ts=t,
                        o=float(mid.get("o")),
                        h=float(mid.get("h")),
                        l=float(mid.get("l")),
                        c=float(mid.get("c")),
                    )
                )
            except Exception:
                continue
        if entries:
            out[d] = entries
    return out


def _iter_m1_between(
    candles_by_day: Dict[dt.date, List[Candle]], start: dt.datetime, end: dt.datetime
) -> Iterable[Candle]:
    cur = start
    while cur <= end:
        day = cur.date()
        arr = candles_by_day.get(day)
        if arr:
            for c in arr:
                if start <= c.ts <= end:
                    yield c
        cur += dt.timedelta(days=1)


@dataclass
class TradeRow:
    id: int
    pocket: str
    units: int
    entry_price: float
    close_price: float
    pl_pips: float
    open_time: dt.datetime
    close_time: dt.datetime
    strategy_tag: str


def _load_trades(days: int) -> List[TradeRow]:
    path = LOGS / "trades.db"
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    utc_now = dt.datetime.now(dt.timezone.utc)
    since = utc_now - dt.timedelta(days=max(1, days))
    rows = con.execute(
        """
        SELECT id, pocket, units, entry_price, close_price, pl_pips, open_time, close_time, strategy_tag, entry_thesis
        FROM trades
        WHERE close_time IS NOT NULL
          AND close_time >= ?
        ORDER BY id ASC
        """,
        (since.isoformat(),),
    ).fetchall()
    con.close()
    out: List[TradeRow] = []
    for r in rows:
        ot = _parse_iso(r["open_time"])
        ct = _parse_iso(r["close_time"])
        if not ot or not ct:
            continue
        tag = _resolve_strategy_tag(r["strategy_tag"], r["entry_thesis"])
        try:
            out.append(
                TradeRow(
                    id=int(r["id"]),
                    pocket=str(r["pocket"] or ""),
                    units=int(r["units"] or 0),
                    entry_price=float(r["entry_price"]),
                    close_price=float(r["close_price"]),
                    pl_pips=float(r["pl_pips"] or 0.0),
                    open_time=ot,
                    close_time=ct,
                    strategy_tag=tag,
                )
            )
        except Exception:
            continue
    return out


def _scan_hold_excursion(
    trade: TradeRow,
    candles_by_day: Dict[dt.date, List[Candle]],
) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    start = trade.open_time
    end = trade.close_time
    entry = trade.entry_price
    long = trade.units > 0
    mfe = 0.0
    mae = 0.0
    t_be: Optional[int] = None
    idx = 0
    seen = False
    for c in _iter_m1_between(candles_by_day, start, end):
        seen = True
        if long:
            fav = (c.h - entry) / PIP
            adv = (entry - c.l) / PIP
            if t_be is None and c.h >= entry:
                t_be = idx
        else:
            fav = (entry - c.l) / PIP
            adv = (c.h - entry) / PIP
            if t_be is None and c.l <= entry:
                t_be = idx
        if fav > mfe:
            mfe = fav
        if adv > mae:
            mae = adv
        idx += 1
    if not seen:
        return None, None, None
    return round(mfe, 2), round(mae, 2), t_be


def _be_post_hit(
    trade: TradeRow,
    candles_by_day: Dict[dt.date, List[Candle]],
    post_minutes: int,
) -> Optional[bool]:
    start = trade.close_time
    end = start + dt.timedelta(minutes=max(1, post_minutes))
    entry = trade.entry_price
    long = trade.units > 0
    seen = False
    for c in _iter_m1_between(candles_by_day, start, end):
        seen = True
        if long and c.h >= entry:
            return True
        if not long and c.l <= entry:
            return True
    return False if seen else None


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    seq = sorted(values)
    if len(seq) == 1:
        return float(seq[0])
    k = (len(seq) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(seq[int(k)])
    d0 = seq[int(f)] * (c - k)
    d1 = seq[int(c)] * (k - f)
    return float(d0 + d1)


def _classify_return_wait(stats: dict, min_trades: int) -> Tuple[str, str]:
    count = stats["count"]
    if count < min_trades:
        return "neutral", "low_sample"
    avg_mfe = stats["mfe_sum"] / stats["mfe_count"] if stats["mfe_count"] else None
    avg_mae = stats["mae_sum"] / stats["mae_count"] if stats["mae_count"] else None
    be_post_30 = stats["be_post_hit"].get(30, 0) / max(1, count)
    be_post_60 = stats["be_post_hit"].get(60, 0) / max(1, count)
    avg_be_time = stats["be_hold_sum"] / stats["be_hold_hits"] if stats["be_hold_hits"] else None
    win_p50 = _percentile(stats["hold_win"], 50) if stats["hold_win"] else None
    loss_p50 = _percentile(stats["hold_loss"], 50) if stats["hold_loss"] else None

    if (
        be_post_30 >= 0.35
        and avg_be_time is not None
        and avg_be_time <= 60.0
        and avg_mae is not None
        and avg_mfe is not None
        and avg_mae <= max(8.0, avg_mfe * 1.4)
    ):
        return "favor", "be_post_30+mae"
    if (
        be_post_60 <= 0.20
        or (
            avg_mae is not None
            and avg_mfe is not None
            and avg_mfe > 0.0
            and avg_mae >= avg_mfe * 2.0
        )
        or (
            win_p50 is not None
            and loss_p50 is not None
            and win_p50 < 20.0
            and loss_p50 > 60.0
        )
    ):
        return "avoid", "be_post_60_or_hold_skew"
    return "neutral", "default"


def _yaml_value(value: object) -> str:
    return json.dumps(value, ensure_ascii=True)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _recommend_reentry_params(stats: dict) -> dict:
    avg_hold = stats["hold_sum"] / max(1, stats["count"])
    win_p50 = _percentile(stats["hold_win"], 50) or avg_hold
    loss_p50 = _percentile(stats["hold_loss"], 50) or avg_hold
    cooldown_win = int(round(win_p50 * 60.0 * 0.2))
    cooldown_loss = int(round(loss_p50 * 60.0 * 0.45))
    cooldown_win = int(_clamp(cooldown_win, *_REENTRY_COOLDOWN_WIN_RANGE))
    cooldown_loss = int(_clamp(cooldown_loss, *_REENTRY_COOLDOWN_LOSS_RANGE))
    avg_mfe = stats["mfe_sum"] / stats["mfe_count"] if stats["mfe_count"] else None
    avg_mae = stats["mae_sum"] / stats["mae_count"] if stats["mae_count"] else None
    if avg_mae is not None and avg_mfe is not None and avg_mfe > 0:
        reentry_pips = min(avg_mae * 0.35, max(_REENTRY_PIPS_RANGE[0], avg_mfe * 0.6))
    elif avg_mae is not None:
        reentry_pips = avg_mae * 0.35
    elif avg_mfe is not None and avg_mfe > 0:
        reentry_pips = avg_mfe * 0.6
    else:
        reentry_pips = _DEFAULT_REENTRY_CONFIG["same_dir_reentry_pips"]
    reentry_pips = round(_clamp(reentry_pips, *_REENTRY_PIPS_RANGE), 3)
    return {
        "cooldown_win_sec": cooldown_win,
        "cooldown_loss_sec": cooldown_loss,
        "same_dir_reentry_pips": reentry_pips,
    }


def _write_reentry_yaml(
    path: Path,
    strategies: Dict[str, dict],
    defaults_override: Optional[Dict[str, object]] = None,
) -> None:
    defaults = dict(_DEFAULT_REENTRY_CONFIG)
    if defaults_override:
        defaults.update(defaults_override)
    lines: list[str] = []
    lines.append("defaults:")
    for key, val in defaults.items():
        lines.append(f"  {key}: {_yaml_value(val)}")
    lines.append("")
    lines.append("strategies:")
    for name in sorted(strategies):
        lines.append(f"  {name}:")
        for key, val in strategies[name].items():
            lines.append(f"    {key}: {_yaml_value(val)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _suggest_block_hours(
    hour_stats: Dict[int, dict],
    *,
    min_trades: int,
    max_avg_mfe: float,
    max_avg_mae: float,
    top_n: int,
    hour_filter: Optional[Iterable[int]] = None,
) -> list[int]:
    allowed_hours = set(int(h) % 24 for h in hour_filter or [])
    candidates: list[tuple[int, float, int]] = []
    for hour, stats in hour_stats.items():
        count = int(stats.get("count", 0) or 0)
        if count < min_trades:
            continue
        if allowed_hours and hour not in allowed_hours:
            continue
        mfe_count = int(stats.get("mfe_count", 0) or 0)
        mae_count = int(stats.get("mae_count", 0) or 0)
        if mfe_count <= 0 or mae_count <= 0:
            continue
        avg_mfe = float(stats.get("mfe_sum", 0.0) or 0.0) / max(1, mfe_count)
        avg_mae = float(stats.get("mae_sum", 0.0) or 0.0) / max(1, mae_count)
        if avg_mfe <= max_avg_mfe and avg_mae <= max_avg_mae:
            candidates.append((hour, avg_mfe + avg_mae, count))
    if not candidates:
        return []
    candidates.sort(key=lambda item: (item[1], -item[2]))
    selected = candidates[: max(0, top_n)] if top_n > 0 else candidates
    return sorted([hour for hour, _, _ in selected])


def _init_hour_stats() -> dict:
    return {
        "count": 0,
        "wins": 0,
        "pips": 0.0,
        "mfe_sum": 0.0,
        "mae_sum": 0.0,
        "mfe_count": 0,
        "mae_count": 0,
    }


def _parse_hour_window(value: Optional[str]) -> list[int]:
    if not value:
        return []
    text = str(value).strip()
    if not text:
        return []
    hours: list[int] = []
    parts = [p.strip() for p in text.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            try:
                start_str, end_str = part.split("-", 1)
                start = int(start_str)
                end = int(end_str)
            except Exception:
                continue
            if start < 0 or end < 0:
                continue
            if start <= end:
                hours.extend(list(range(start, end + 1)))
            else:
                hours.extend(list(range(start, 24)))
                hours.extend(list(range(0, end + 1)))
        else:
            try:
                hours.append(int(part))
            except Exception:
                continue
    return sorted(set(h % 24 for h in hours))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--post-min", type=int, nargs="*", default=[30, 60])
    ap.add_argument("--min-trades", type=int, default=30)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--raw-tag", action="store_true", help="use raw strategy_tag (no base grouping)")
    ap.add_argument("--out-json", default="", help="optional output JSON path")
    ap.add_argument("--out-yaml", default="", help="optional worker_reentry.yaml output path")
    ap.add_argument("--block-hour-trades", type=int, default=20)
    ap.add_argument("--block-hour-mfe-max", type=float, default=1.2)
    ap.add_argument("--block-hour-mae-max", type=float, default=1.2)
    ap.add_argument("--block-hour-top", type=int, default=4)
    ap.add_argument(
        "--block-hour-window",
        default="",
        help="optional JST hour window, e.g. 3-6 or 3,4,5",
    )
    ap.add_argument("--block-hours-scope", default="global", choices=["global", "per_strategy"])
    args = ap.parse_args()
    block_hour_window = _parse_hour_window(args.block_hour_window)

    trades = _load_trades(args.days)
    if not trades:
        print("No closed trades in lookback window.")
        return
    dates = set()
    for t in trades:
        dates.add(t.open_time.date())
        dates.add(t.close_time.date())
    candles_by_day = _load_m1_for_dates(dates)

    stats_by_strategy: Dict[str, dict] = {}
    global_hours: Dict[int, dict] = {}
    post_windows = sorted(set(int(x) for x in args.post_min))
    for tr in trades:
        tag = tr.strategy_tag or "unknown"
        if not args.raw_tag:
            tag = _base_strategy_tag(tag)
        stats = stats_by_strategy.setdefault(
            tag,
            {
                "count": 0,
                "wins": 0,
                "losses": 0,
                "pips_sum": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "hold_sum": 0.0,
                "hold_win": [],
                "hold_loss": [],
                "mfe_sum": 0.0,
                "mae_sum": 0.0,
                "mfe_count": 0,
                "mae_count": 0,
                "be_hold_hits": 0,
                "be_hold_sum": 0.0,
                "be_post_hit": {w: 0 for w in post_windows},
                "hours": {},
            },
        )
        hold_min = max(0.0, (tr.close_time - tr.open_time).total_seconds() / 60.0)
        stats["count"] += 1
        stats["pips_sum"] += tr.pl_pips
        stats["hold_sum"] += hold_min
        if tr.pl_pips > 0:
            stats["wins"] += 1
            stats["hold_win"].append(hold_min)
            stats["gross_profit"] += tr.pl_pips
        elif tr.pl_pips < 0:
            stats["losses"] += 1
            stats["hold_loss"].append(hold_min)
            stats["gross_loss"] += abs(tr.pl_pips)
        entry_hour = int((tr.open_time + dt.timedelta(hours=9)).hour)
        hour_stats = stats["hours"].setdefault(
            entry_hour,
            _init_hour_stats(),
        )
        global_stats = global_hours.setdefault(entry_hour, _init_hour_stats())
        hour_stats["count"] += 1
        global_stats["count"] += 1
        if tr.pl_pips > 0:
            hour_stats["wins"] += 1
            global_stats["wins"] += 1
        hour_stats["pips"] += tr.pl_pips
        global_stats["pips"] += tr.pl_pips

        mfe, mae, be_min = _scan_hold_excursion(tr, candles_by_day)
        if mfe is not None:
            stats["mfe_sum"] += mfe
            stats["mfe_count"] += 1
            hour_stats["mfe_sum"] += mfe
            hour_stats["mfe_count"] += 1
            global_stats["mfe_sum"] += mfe
            global_stats["mfe_count"] += 1
        if mae is not None:
            stats["mae_sum"] += mae
            stats["mae_count"] += 1
            hour_stats["mae_sum"] += mae
            hour_stats["mae_count"] += 1
            global_stats["mae_sum"] += mae
            global_stats["mae_count"] += 1
        if be_min is not None:
            stats["be_hold_hits"] += 1
            stats["be_hold_sum"] += be_min
        for window in post_windows:
            hit = _be_post_hit(tr, candles_by_day, window)
            if hit:
                stats["be_post_hit"][window] += 1

    ranked = sorted(stats_by_strategy.items(), key=lambda item: item[1]["count"], reverse=True)
    global_block_hours = _suggest_block_hours(
        global_hours,
        min_trades=args.block_hour_trades,
        max_avg_mfe=args.block_hour_mfe_max,
        max_avg_mae=args.block_hour_mae_max,
        top_n=args.block_hour_top,
        hour_filter=block_hour_window,
    )
    print("=== Worker return-wait summary ===")
    if block_hour_window:
        print(f"Block hour window (JST): {block_hour_window}")
    if args.block_hours_scope == "global" and global_block_hours:
        print(f"Global low-vol block hours (JST): {global_block_hours}")
    for tag, stats in ranked[: max(1, args.top)]:
        count = stats["count"]
        win_rate = stats["wins"] / max(1, count)
        avg_pips = stats["pips_sum"] / max(1, count)
        avg_hold = stats["hold_sum"] / max(1, count)
        avg_mfe = stats["mfe_sum"] / stats["mfe_count"] if stats["mfe_count"] else 0.0
        avg_mae = stats["mae_sum"] / stats["mae_count"] if stats["mae_count"] else 0.0
        win_p50 = _percentile(stats["hold_win"], 50) or 0.0
        loss_p50 = _percentile(stats["hold_loss"], 50) or 0.0
        win_p90 = _percentile(stats["hold_win"], 90) or 0.0
        loss_p90 = _percentile(stats["hold_loss"], 90) or 0.0
        be_avg = stats["be_hold_sum"] / stats["be_hold_hits"] if stats["be_hold_hits"] else None
        bias, reason = _classify_return_wait(stats, args.min_trades)
        pf = (
            stats["gross_profit"] / max(1e-6, stats["gross_loss"])
            if stats["gross_profit"] > 0 or stats["gross_loss"] > 0
            else 0.0
        )
        parts = []
        for window in post_windows:
            ratio = stats["be_post_hit"][window] / max(1, count)
            parts.append(f"BE_post{window}={ratio:.2f}")
        block_hours: list[int] = []
        if args.block_hours_scope == "per_strategy":
            block_hours = _suggest_block_hours(
                stats["hours"],
                min_trades=args.block_hour_trades,
                max_avg_mfe=args.block_hour_mfe_max,
                max_avg_mae=args.block_hour_mae_max,
                top_n=args.block_hour_top,
                hour_filter=block_hour_window,
            )
        reentry_params = _recommend_reentry_params(stats)
        line = (
            f"{tag}: n={count} win_rate={win_rate:.2f} avg_pips={avg_pips:.2f} "
            f"avg_hold={avg_hold:.1f}m win_p50={win_p50:.1f}m loss_p50={loss_p50:.1f}m "
            f"win_p90={win_p90:.1f}m loss_p90={loss_p90:.1f}m "
            f"avg_MFE={avg_mfe:.2f}p avg_MAE={avg_mae:.2f}p pf={pf:.2f} "
        )
        if be_avg is not None:
            line += f"avg_BE={be_avg:.1f}m "
        line += " ".join(parts)
        if block_hours:
            line += f" block_hours={block_hours}"
        line += (
            f" cd_win={reentry_params['cooldown_win_sec']}s"
            f" cd_loss={reentry_params['cooldown_loss_sec']}s"
            f" re_pips={reentry_params['same_dir_reentry_pips']}"
        )
        line += f" bias={bias} reason={reason}"
        print(line)

    if args.out_json:
        payload = {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "days": args.days,
            "min_trades": args.min_trades,
            "post_windows": post_windows,
            "defaults": {
                "block_jst_hours": global_block_hours if args.block_hours_scope == "global" else [],
                "block_hours_scope": args.block_hours_scope,
                "block_hour_window": block_hour_window,
                "block_hour_thresholds": {
                    "min_trades": args.block_hour_trades,
                    "max_avg_mfe": args.block_hour_mfe_max,
                    "max_avg_mae": args.block_hour_mae_max,
                    "top_n": args.block_hour_top,
                },
            },
            "strategies": {},
        }
        for tag, stats in ranked:
            bias, reason = _classify_return_wait(stats, args.min_trades)
            count = stats["count"]
            block_hours: list[int] = []
            if args.block_hours_scope == "per_strategy":
                block_hours = _suggest_block_hours(
                    stats["hours"],
                    min_trades=args.block_hour_trades,
                    max_avg_mfe=args.block_hour_mfe_max,
                    max_avg_mae=args.block_hour_mae_max,
                    top_n=args.block_hour_top,
                    hour_filter=block_hour_window,
                )
            payload["strategies"][tag] = {
                "count": count,
                "wins": stats["wins"],
                "losses": stats["losses"],
                "win_rate": stats["wins"] / max(1, count),
                "avg_pips": stats["pips_sum"] / max(1, count),
                "profit_factor": (
                    stats["gross_profit"] / max(1e-6, stats["gross_loss"])
                    if stats["gross_profit"] > 0 or stats["gross_loss"] > 0
                    else 0.0
                ),
                "avg_hold_min": stats["hold_sum"] / max(1, count),
                "hold_win_p50": _percentile(stats["hold_win"], 50),
                "hold_win_p90": _percentile(stats["hold_win"], 90),
                "hold_loss_p50": _percentile(stats["hold_loss"], 50),
                "hold_loss_p90": _percentile(stats["hold_loss"], 90),
                "avg_mfe": stats["mfe_sum"] / stats["mfe_count"] if stats["mfe_count"] else None,
                "avg_mae": stats["mae_sum"] / stats["mae_count"] if stats["mae_count"] else None,
                "avg_be_min": stats["be_hold_sum"] / stats["be_hold_hits"] if stats["be_hold_hits"] else None,
                "be_post_hit": {
                    str(w): stats["be_post_hit"][w] / max(1, count) for w in post_windows
                },
                "block_jst_hours": block_hours,
                "return_wait_bias": bias,
                "return_wait_reason": reason,
                "reentry_recommendation": _recommend_reentry_params(stats),
            }
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2))
        print(f"Wrote {out_path}")

    if args.out_yaml:
        yaml_strategies: Dict[str, dict] = {}
        for tag, stats in ranked:
            if stats["count"] < args.min_trades:
                continue
            bias, reason = _classify_return_wait(stats, args.min_trades)
            block_hours: list[int] = []
            if args.block_hours_scope == "per_strategy":
                block_hours = _suggest_block_hours(
                    stats["hours"],
                    min_trades=args.block_hour_trades,
                    max_avg_mfe=args.block_hour_mfe_max,
                    max_avg_mae=args.block_hour_mae_max,
                    top_n=args.block_hour_top,
                    hour_filter=block_hour_window,
                )
            reentry_params = _recommend_reentry_params(stats)
            yaml_strategies[tag] = {
                "return_wait_bias": bias,
                "return_wait_reason": reason,
                "block_jst_hours": block_hours,
                **reentry_params,
            }
        out_path = Path(args.out_yaml)
        defaults_override = {}
        if args.block_hours_scope == "global":
            defaults_override["block_jst_hours"] = global_block_hours
        _write_reentry_yaml(out_path, yaml_strategies, defaults_override=defaults_override)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
