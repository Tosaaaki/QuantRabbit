"""
analytics.entry_thesis_flag_report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Summarize win-rate/PF/avg pips deltas by entry_thesis flags per strategy.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS = REPO_ROOT / "logs"

_ALIAS_BASE = {
    "mlr": "MicroLevelReactor",
    "trendma": "TrendMA",
    "donchian": "Donchian55",
    "h1momentum": "H1Momentum",
    "m1scalper": "M1Scalper",
    "bbrsi": "BB_RSI",
    "bb_rsi": "BB_RSI",
}

_FLAG_KEYS = (
    "trend_bias",
    "trend_score",
    "size_factor_hint",
    "range_snapshot",
    "entry_mean",
    "reversion_failure",
    "mr_guard",
    "mr_overlay",
    "tp_mode",
    "tp_target",
    "section_axis",
    "tech_entry",
    "pattern_tag",
    "pattern_meta",
    "profile",
    "entry_guard_override",
)


@dataclass
class Stat:
    count: int = 0
    wins: int = 0
    pips_sum: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0

    def win_rate(self) -> float:
        return self.wins / self.count if self.count else 0.0

    def avg_pips(self) -> float:
        return self.pips_sum / self.count if self.count else 0.0

    def pf(self) -> float:
        if self.gross_profit <= 0 and self.gross_loss <= 0:
            return 0.0
        return self.gross_profit / max(1e-6, self.gross_loss)


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


def _extract_flags(thesis_raw: Optional[str]) -> List[str]:
    if not thesis_raw:
        return []
    try:
        payload = json.loads(thesis_raw)
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    flags: set[str] = set()
    existing = payload.get("flags")
    if isinstance(existing, (list, tuple, set)):
        for item in existing:
            if item:
                flags.add(str(item))
    elif isinstance(existing, str):
        for token in existing.split(","):
            token = token.strip()
            if token:
                flags.add(token)
    for key in _FLAG_KEYS:
        val = payload.get(key)
        if val in (None, False, "", 0):
            continue
        flags.add(key)
    for key, val in payload.items():
        if key.startswith("entry_guard_") and val:
            flags.add(key)
    return sorted(flags)


def _load_rows(days: int) -> Iterable[sqlite3.Row]:
    path = LOGS / "trades.db"
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    utc_now = dt.datetime.now(dt.timezone.utc)
    since = utc_now - dt.timedelta(days=max(1, days))
    rows = con.execute(
        """
        SELECT strategy_tag, entry_thesis, pl_pips, close_time
        FROM trades
        WHERE close_time IS NOT NULL
          AND close_time >= ?
        ORDER BY id ASC
        """,
        (since.isoformat(),),
    ).fetchall()
    con.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--min-trades", type=int, default=25)
    ap.add_argument("--top-strategies", type=int, default=20)
    ap.add_argument("--top-flags", type=int, default=6)
    ap.add_argument("--raw-tag", action="store_true")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    base_stats: Dict[str, Stat] = {}
    flag_stats: Dict[Tuple[str, str], Stat] = {}
    for row in _load_rows(args.days):
        tag = _resolve_strategy_tag(row["strategy_tag"], row["entry_thesis"])
        if not args.raw_tag:
            tag = _base_strategy_tag(tag)
        if not tag:
            continue
        pl_pips = float(row["pl_pips"] or 0.0)
        stat = base_stats.setdefault(tag, Stat())
        stat.count += 1
        stat.pips_sum += pl_pips
        if pl_pips > 0:
            stat.wins += 1
            stat.gross_profit += pl_pips
        elif pl_pips < 0:
            stat.gross_loss += abs(pl_pips)

        flags = _extract_flags(row["entry_thesis"])
        if not flags:
            continue
        for flag in flags:
            key = (tag, flag)
            fstat = flag_stats.setdefault(key, Stat())
            fstat.count += 1
            fstat.pips_sum += pl_pips
            if pl_pips > 0:
                fstat.wins += 1
                fstat.gross_profit += pl_pips
            elif pl_pips < 0:
                fstat.gross_loss += abs(pl_pips)

    ranked = sorted(base_stats.items(), key=lambda item: item[1].count, reverse=True)
    print("=== entry_thesis flag delta summary ===")
    for tag, stat in ranked[: max(1, args.top_strategies)]:
        if stat.count == 0:
            continue
        base_wr = stat.win_rate()
        base_avg = stat.avg_pips()
        base_pf = stat.pf()
        print(
            f"{tag}: n={stat.count} win_rate={base_wr:.2f} avg_pips={base_avg:.2f} pf={base_pf:.2f}"
        )
        candidates: list[tuple[str, Stat]] = []
        for (s_tag, flag), fstat in flag_stats.items():
            if s_tag != tag:
                continue
            if fstat.count < args.min_trades:
                continue
            candidates.append((flag, fstat))
        candidates.sort(
            key=lambda item: (
                abs(item[1].win_rate() - base_wr),
                item[1].count,
            ),
            reverse=True,
        )
        for flag, fstat in candidates[: max(1, args.top_flags)]:
            delta_wr = fstat.win_rate() - base_wr
            delta_avg = fstat.avg_pips() - base_avg
            print(
                f"  flag={flag} n={fstat.count} win_rate={fstat.win_rate():.2f} "
                f"avg_pips={fstat.avg_pips():.2f} pf={fstat.pf():.2f} "
                f"delta_wr={delta_wr:+.2f} delta_avg={delta_avg:+.2f}"
            )

    if args.out_json:
        payload: Dict[str, object] = {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "days": args.days,
            "min_trades": args.min_trades,
            "strategies": {},
        }
        for tag, stat in base_stats.items():
            entry = {
                "count": stat.count,
                "win_rate": stat.win_rate(),
                "avg_pips": stat.avg_pips(),
                "profit_factor": stat.pf(),
                "flags": {},
            }
            for (s_tag, flag), fstat in flag_stats.items():
                if s_tag != tag or fstat.count < args.min_trades:
                    continue
                entry["flags"][flag] = {
                    "count": fstat.count,
                    "win_rate": fstat.win_rate(),
                    "avg_pips": fstat.avg_pips(),
                    "profit_factor": fstat.pf(),
                    "delta_win_rate": fstat.win_rate() - stat.win_rate(),
                    "delta_avg_pips": fstat.avg_pips() - stat.avg_pips(),
                }
            payload["strategies"][tag] = entry
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2))
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
