"""
analytics.wfo_overfit_report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Walk-forward robustness report with a lightweight overfit proxy.

Output:
- WFO summary (train/test rolling windows)
- PBO-lite (selected-on-train strategy underperforms on test)
- Per-strategy Sharpe diagnostics with an approximate Deflated Sharpe Ratio
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sqlite3
from statistics import NormalDist
from typing import Any


_ND = NormalDist()
_EULER_GAMMA = 0.5772156649015329


@dataclass(frozen=True)
class TradeRow:
    close_time: datetime
    strategy: str
    pl_pips: float

    @property
    def close_date(self) -> date:
        return self.close_time.date()


def _parse_time(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if math.isnan(out) or math.isinf(out):
        return float(default)
    return out


def _sample_moments(values: list[float]) -> tuple[float, float, float, float]:
    n = len(values)
    if n <= 1:
        return 0.0, 0.0, 0.0, 3.0
    mean = sum(values) / n
    centered = [x - mean for x in values]
    m2 = sum(v * v for v in centered) / n
    if m2 <= 1e-12:
        return mean, 0.0, 0.0, 3.0
    m3 = sum(v * v * v for v in centered) / n
    m4 = sum(v * v * v * v for v in centered) / n
    skew = m3 / (m2 ** 1.5)
    kurt = m4 / (m2 * m2)
    # Use sample std for Sharpe stability.
    sample_var = sum(v * v for v in centered) / (n - 1)
    std = math.sqrt(max(sample_var, 0.0))
    return mean, std, skew, kurt


def _trade_stats(values: list[float]) -> dict[str, float]:
    n = len(values)
    if n == 0:
        return {
            "n": 0.0,
            "sum_pips": 0.0,
            "avg_pips": 0.0,
            "win_rate": 0.0,
            "pf": 0.0,
            "sharpe": 0.0,
            "skew": 0.0,
            "kurtosis": 3.0,
        }
    mean, std, skew, kurt = _sample_moments(values)
    wins = [v for v in values if v > 0.0]
    losses = [v for v in values if v < 0.0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    sharpe = mean / std if std > 0 else 0.0
    return {
        "n": float(n),
        "sum_pips": float(sum(values)),
        "avg_pips": float(mean),
        "win_rate": float(len(wins) / n),
        "pf": float(pf),
        "sharpe": float(sharpe),
        "skew": float(skew),
        "kurtosis": float(kurt),
    }


def _probabilistic_sharpe_ratio(
    *,
    sharpe: float,
    sr_ref: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
) -> float:
    if n_obs <= 1:
        return 0.0
    denom = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * (sharpe**2)
    denom = math.sqrt(max(1e-12, denom))
    z = (sharpe - sr_ref) * math.sqrt(max(1.0, n_obs - 1)) / denom
    return float(_ND.cdf(z))


def _deflated_sharpe_ratio(
    *,
    sharpe: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    n_trials: int,
) -> tuple[float, float]:
    if n_obs <= 1:
        return 0.0, 0.0
    n_trials = max(1, int(n_trials))
    variance_term = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * (sharpe**2)
    std_sr = math.sqrt(max(1e-12, variance_term / max(1.0, n_obs - 1)))
    if n_trials <= 1:
        sr_star = 0.0
    else:
        p1 = max(1e-9, min(1 - 1e-9, 1.0 - 1.0 / n_trials))
        p2 = max(1e-9, min(1 - 1e-9, 1.0 - 1.0 / (n_trials * math.e)))
        sr_star = std_sr * ((1.0 - _EULER_GAMMA) * _ND.inv_cdf(p1) + _EULER_GAMMA * _ND.inv_cdf(p2))
    dsr = _probabilistic_sharpe_ratio(
        sharpe=sharpe,
        sr_ref=sr_star,
        n_obs=n_obs,
        skew=skew,
        kurtosis=kurtosis,
    )
    return float(dsr), float(sr_star)


def _metric_score(stats: dict[str, float], metric: str) -> float:
    key = (metric or "").strip().lower()
    if key == "avg_pips":
        return float(stats.get("avg_pips") or 0.0)
    if key == "sharpe":
        return float(stats.get("sharpe") or 0.0)
    return float(stats.get("pf") or 0.0)


def load_trades(
    *,
    db_path: Path,
    instrument: str = "USD_JPY",
    days: int = 180,
    strategy: str | None = None,
) -> list[TradeRow]:
    if not db_path.exists():
        return []
    sql = """
    SELECT close_time,
           COALESCE(NULLIF(strategy_tag, ''), NULLIF(strategy, ''), 'unknown') AS strategy_name,
           COALESCE(pl_pips, 0.0) AS pl_pips
    FROM trades
    WHERE close_time IS NOT NULL
      AND instrument = ?
      AND close_time >= datetime('now', ?)
    """
    params: list[Any] = [instrument, f"-{int(max(1, days))} days"]
    if strategy:
        sql += " AND (strategy_tag = ? OR strategy = ?)"
        params.extend([strategy, strategy])
    sql += " ORDER BY close_time ASC"
    rows: list[TradeRow] = []
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.execute(sql, params)
        for close_time, strategy_name, pl_pips in cur.fetchall():
            ts = _parse_time(close_time)
            if ts is None:
                continue
            rows.append(
                TradeRow(
                    close_time=ts,
                    strategy=str(strategy_name or "unknown"),
                    pl_pips=_safe_float(pl_pips),
                )
            )
    finally:
        con.close()
    return rows


def _build_day_range(rows: list[TradeRow]) -> tuple[date, date] | None:
    if not rows:
        return None
    start = rows[0].close_date
    end = rows[-1].close_date
    return start, end


def _rows_between(
    rows: list[TradeRow],
    *,
    start_day: date,
    end_day: date,
) -> list[TradeRow]:
    return [row for row in rows if start_day <= row.close_date < end_day]


def build_report(
    rows: list[TradeRow],
    *,
    train_days: int,
    test_days: int,
    step_days: int,
    min_train_trades: int,
    min_test_trades: int,
    metric: str,
) -> dict[str, Any]:
    if not rows:
        return {
            "summary": {
                "windows_total": 0,
                "windows_evaluated": 0,
                "pbo_lite": 0.0,
                "selected_positive_rate": 0.0,
                "median_test_percentile": 0.0,
            },
            "windows": [],
            "strategy_stats": [],
        }

    period = _build_day_range(rows)
    if period is None:
        return {"summary": {}, "windows": [], "strategy_stats": []}
    first_day, last_day = period
    train_days = max(5, int(train_days))
    test_days = max(3, int(test_days))
    step_days = max(1, int(step_days))
    cursor = first_day + timedelta(days=train_days)
    tail = last_day + timedelta(days=1)

    windows: list[dict[str, Any]] = []
    overfit_flags: list[int] = []
    percentiles: list[float] = []
    selected_positive: list[int] = []

    while cursor + timedelta(days=test_days) <= tail:
        train_start = cursor - timedelta(days=train_days)
        train_end = cursor
        test_start = cursor
        test_end = cursor + timedelta(days=test_days)

        train_rows = _rows_between(rows, start_day=train_start, end_day=train_end)
        test_rows = _rows_between(rows, start_day=test_start, end_day=test_end)

        train_by_strategy: dict[str, list[float]] = {}
        test_by_strategy: dict[str, list[float]] = {}
        for row in train_rows:
            train_by_strategy.setdefault(row.strategy, []).append(row.pl_pips)
        for row in test_rows:
            test_by_strategy.setdefault(row.strategy, []).append(row.pl_pips)

        candidates = [
            name
            for name, vals in train_by_strategy.items()
            if len(vals) >= min_train_trades and len(test_by_strategy.get(name, [])) >= min_test_trades
        ]
        if len(candidates) >= 2:
            train_stats = {name: _trade_stats(train_by_strategy[name]) for name in candidates}
            test_stats = {name: _trade_stats(test_by_strategy[name]) for name in candidates}
            ranked_train = sorted(
                candidates,
                key=lambda name: _metric_score(train_stats[name], metric),
                reverse=True,
            )
            selected = ranked_train[0]
            ranked_test = sorted(
                candidates,
                key=lambda name: _metric_score(test_stats[name], metric),
                reverse=True,
            )
            idx = ranked_test.index(selected)
            if len(ranked_test) <= 1:
                percentile = 1.0
            else:
                percentile = 1.0 - (idx / (len(ranked_test) - 1))
            overfit = 1 if percentile < 0.5 else 0
            positive = 1 if _metric_score(test_stats[selected], metric) > 0.0 else 0
            overfit_flags.append(overfit)
            percentiles.append(percentile)
            selected_positive.append(positive)
            windows.append(
                {
                    "train_start": train_start.isoformat(),
                    "train_end": train_end.isoformat(),
                    "test_start": test_start.isoformat(),
                    "test_end": test_end.isoformat(),
                    "candidate_count": len(candidates),
                    "selected_strategy": selected,
                    "selected_train": train_stats[selected],
                    "selected_test": test_stats[selected],
                    "test_percentile": round(float(percentile), 4),
                    "overfit_flag": bool(overfit),
                }
            )
        cursor += timedelta(days=step_days)

    overall_by_strategy: dict[str, list[float]] = {}
    for row in rows:
        overall_by_strategy.setdefault(row.strategy, []).append(row.pl_pips)
    strategy_rows: list[dict[str, Any]] = []
    trials = max(1, len(overall_by_strategy))
    for name, vals in overall_by_strategy.items():
        stats = _trade_stats(vals)
        n_obs = int(stats["n"])
        dsr, sr_star = _deflated_sharpe_ratio(
            sharpe=float(stats["sharpe"]),
            n_obs=n_obs,
            skew=float(stats["skew"]),
            kurtosis=float(stats["kurtosis"]),
            n_trials=trials,
        )
        stats["dsr"] = dsr
        stats["sr_star"] = sr_star
        strategy_rows.append({"strategy": name, **stats})
    strategy_rows.sort(key=lambda row: float(row.get("dsr") or 0.0), reverse=True)

    evaluated = len(windows)
    pbo_lite = (sum(overfit_flags) / evaluated) if evaluated else 0.0
    median_pct = 0.0
    if percentiles:
        sorted_pct = sorted(percentiles)
        mid = len(sorted_pct) // 2
        if len(sorted_pct) % 2:
            median_pct = sorted_pct[mid]
        else:
            median_pct = 0.5 * (sorted_pct[mid - 1] + sorted_pct[mid])
    positive_rate = (sum(selected_positive) / evaluated) if evaluated else 0.0

    return {
        "summary": {
            "period_start": first_day.isoformat(),
            "period_end": last_day.isoformat(),
            "trade_count": len(rows),
            "strategy_count": len(overall_by_strategy),
            "metric": metric,
            "train_days": train_days,
            "test_days": test_days,
            "step_days": step_days,
            "min_train_trades": min_train_trades,
            "min_test_trades": min_test_trades,
            "windows_total": len(windows),
            "windows_evaluated": evaluated,
            "pbo_lite": round(float(pbo_lite), 6),
            "selected_positive_rate": round(float(positive_rate), 6),
            "median_test_percentile": round(float(median_pct), 6),
        },
        "windows": windows,
        "strategy_stats": strategy_rows,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# WFO / Overfit Report",
        "",
        "## Summary",
        f"- period: {summary.get('period_start')} -> {summary.get('period_end')}",
        f"- trades: {summary.get('trade_count')}",
        f"- strategies: {summary.get('strategy_count')}",
        f"- metric: {summary.get('metric')}",
        f"- windows evaluated: {summary.get('windows_evaluated')}",
        f"- PBO-lite: {summary.get('pbo_lite')}",
        f"- selected positive rate: {summary.get('selected_positive_rate')}",
        f"- median selected test percentile: {summary.get('median_test_percentile')}",
        "",
        "## Top Strategies (DSR)",
        "| strategy | n | avg_pips | pf | sharpe | dsr |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in (report.get("strategy_stats") or [])[:10]:
        lines.append(
            "| {strategy} | {n:.0f} | {avg_pips:.3f} | {pf:.3f} | {sharpe:.3f} | {dsr:.3f} |".format(
                strategy=row.get("strategy", "unknown"),
                n=float(row.get("n") or 0.0),
                avg_pips=float(row.get("avg_pips") or 0.0),
                pf=float(row.get("pf") or 0.0),
                sharpe=float(row.get("sharpe") or 0.0),
                dsr=float(row.get("dsr") or 0.0),
            )
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WFO + overfit diagnostics report")
    parser.add_argument("--db", default="logs/trades.db", help="Path to trades.db")
    parser.add_argument("--instrument", default="USD_JPY")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--train-days", type=int, default=28)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--min-train-trades", type=int, default=15)
    parser.add_argument("--min-test-trades", type=int, default=8)
    parser.add_argument("--metric", choices=("pf", "avg_pips", "sharpe"), default="pf")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_trades(
        db_path=Path(args.db),
        instrument=args.instrument,
        days=args.days,
        strategy=args.strategy,
    )
    report = build_report(
        rows,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        min_train_trades=max(1, int(args.min_train_trades)),
        min_test_trades=max(1, int(args.min_test_trades)),
        metric=args.metric,
    )
    print(json.dumps(report.get("summary", {}), ensure_ascii=False, indent=2))

    out_json = Path(args.out_json) if args.out_json else None
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    out_md = Path(args.out_md) if args.out_md else None
    if out_md is not None:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(_render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
