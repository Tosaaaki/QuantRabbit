#!/usr/bin/env python3
"""Walk-forward tuner for regime-router route mapping.

Input:
- replay JSON for ping5s C and D variants (output from replay_exit_workers.py)

Output:
- per-fold route mapping (trend/range/breakout/mixed/event/unknown -> C|D)
- selected test metrics
- suggested env overrides for quant-regime-router
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

from analytics.replay_quality_gate import compute_trade_metrics
from scripts.replay_exit_workers import _candidate_regime_route


ROUTES = ("trend", "breakout", "range", "mixed", "event", "unknown")


@dataclass(frozen=True)
class RouteTrade:
    worker: str
    route: str
    day: str
    entry_time: str
    pnl_pips: float
    pnl_jpy: float
    reason: str


def _parse_iso(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _route_from_trade(trade: dict[str, Any]) -> str:
    route_raw = str(trade.get("regime_route") or "").strip().lower()
    if route_raw in ROUTES:
        return route_raw
    macro = trade.get("macro_regime")
    micro = trade.get("micro_regime")
    route = _candidate_regime_route(macro, micro)
    return route if route in ROUTES else "unknown"


def _load_route_trades(path: Path, *, worker: str, exclude_end_of_replay: bool) -> list[RouteTrade]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    trades = payload.get("trades")
    if not isinstance(trades, list):
        return []
    out: list[RouteTrade] = []
    for row in trades:
        if not isinstance(row, dict):
            continue
        reason = str(row.get("reason") or "")
        if exclude_end_of_replay and reason == "end_of_replay":
            continue
        entry_time = _parse_iso(row.get("entry_time"))
        if entry_time is None:
            continue
        try:
            pnl_pips = float(row.get("pnl_pips"))
            pnl_jpy = float(row.get("pnl_jpy") or 0.0)
        except Exception:
            continue
        out.append(
            RouteTrade(
                worker=worker,
                route=_route_from_trade(row),
                day=entry_time.date().isoformat(),
                entry_time=entry_time.isoformat(),
                pnl_pips=pnl_pips,
                pnl_jpy=pnl_jpy,
                reason=reason,
            )
        )
    return out


def _build_day_folds(
    days: list[str],
    *,
    train_days: int,
    test_days: int,
    step_days: int,
) -> list[dict[str, list[str]]]:
    train_n = max(1, int(train_days))
    test_n = max(1, int(test_days))
    step_n = max(1, int(step_days))
    total = len(days)
    if total < train_n + test_n:
        return []
    folds: list[dict[str, list[str]]] = []
    cursor = train_n
    while cursor + test_n <= total:
        folds.append(
            {
                "train": list(days[cursor - train_n : cursor]),
                "test": list(days[cursor : cursor + test_n]),
            }
        )
        cursor += step_n
    return folds


def _rows_to_metric_trades(rows: Iterable[RouteTrade]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "entry_time": row.entry_time,
                "exit_time": row.entry_time,
                "pnl_pips": row.pnl_pips,
                "pnl_jpy": row.pnl_jpy,
                "reason": row.reason,
            }
        )
    return out


def _route_worker_stats(rows: Iterable[RouteTrade]) -> dict[tuple[str, str], dict[str, float]]:
    bucket: dict[tuple[str, str], list[RouteTrade]] = {}
    for row in rows:
        key = (row.route, row.worker)
        bucket.setdefault(key, []).append(row)
    out: dict[tuple[str, str], dict[str, float]] = {}
    for key, grouped in bucket.items():
        metrics = compute_trade_metrics(_rows_to_metric_trades(grouped))
        out[key] = metrics
    return out


def _choose_route_mapping(
    train_rows: list[RouteTrade],
    *,
    workers: tuple[str, ...],
    default_worker: str,
    min_train_route_trades: int,
) -> dict[str, str]:
    stats = _route_worker_stats(train_rows)
    mapping: dict[str, str] = {}
    for route in ROUTES:
        best_worker = default_worker
        best_score = float("-inf")
        for worker in workers:
            metric = stats.get((route, worker))
            if not metric:
                continue
            trades = int(metric.get("trade_count") or 0)
            if trades < min_train_route_trades:
                continue
            # Primary: total_jpy; secondary: jpy_per_hour then profit_factor.
            total_jpy = float(metric.get("total_jpy") or 0.0)
            jpy_per_hour = float(metric.get("jpy_per_hour") or 0.0)
            pf = float(metric.get("profit_factor") or 0.0)
            score = total_jpy + (jpy_per_hour * 0.1) + (pf * 0.01)
            if score > best_score:
                best_score = score
                best_worker = worker
        mapping[route] = best_worker
    return mapping


def _apply_mapping(rows: list[RouteTrade], mapping: dict[str, str]) -> list[RouteTrade]:
    selected: list[RouteTrade] = []
    for row in rows:
        if mapping.get(row.route) == row.worker:
            selected.append(row)
    return selected


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _build_env_suggestion(
    fold_mappings: list[dict[str, str]],
    *,
    worker_to_strategy: dict[str, str],
) -> dict[str, str]:
    votes: dict[str, dict[str, int]] = {route: {} for route in ROUTES}
    for mapping in fold_mappings:
        for route, worker in mapping.items():
            if route not in votes:
                continue
            votes[route][worker] = votes[route].get(worker, 0) + 1

    final_mapping: dict[str, str] = {}
    for route in ROUTES:
        worker_votes = votes.get(route) or {}
        if not worker_votes:
            continue
        winner = sorted(worker_votes.items(), key=lambda item: (-item[1], item[0]))[0][0]
        final_mapping[route] = winner

    out: dict[str, str] = {}
    for route in ROUTES:
        worker = final_mapping.get(route)
        if not worker:
            continue
        strategy_slug = worker_to_strategy.get(worker, "")
        if not strategy_slug:
            continue
        out[f"REGIME_ROUTER_{route.upper()}_ENTRY_STRATEGIES"] = strategy_slug
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="WFO tuner for regime-router mapping (ping5s C/D)")
    ap.add_argument("--replay-c", type=Path, required=True)
    ap.add_argument("--replay-d", type=Path, required=True)
    ap.add_argument("--worker-c", default="C")
    ap.add_argument("--worker-d", default="D")
    ap.add_argument("--strategy-c", default="scalp_ping_5s_c")
    ap.add_argument("--strategy-d", default="scalp_ping_5s_d")
    ap.add_argument("--train-days", type=int, default=3)
    ap.add_argument("--test-days", type=int, default=1)
    ap.add_argument("--step-days", type=int, default=1)
    ap.add_argument("--min-train-route-trades", type=int, default=2)
    ap.add_argument("--default-worker", default="C")
    ap.add_argument("--exclude-end-of-replay", action="store_true")
    ap.add_argument("--target-jpy-per-hour", type=float, default=2000.0)
    ap.add_argument("--out-json", type=Path, default=Path("tmp/replay_regime_router_wfo.json"))
    ap.add_argument("--out-md", type=Path, default=Path("tmp/replay_regime_router_wfo.md"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    worker_c = str(args.worker_c).strip() or "C"
    worker_d = str(args.worker_d).strip() or "D"
    workers = (worker_c, worker_d)
    default_worker = str(args.default_worker).strip() or worker_c
    if default_worker not in workers:
        default_worker = worker_c

    c_rows = _load_route_trades(
        args.replay_c.resolve(),
        worker=worker_c,
        exclude_end_of_replay=bool(args.exclude_end_of_replay),
    )
    d_rows = _load_route_trades(
        args.replay_d.resolve(),
        worker=worker_d,
        exclude_end_of_replay=bool(args.exclude_end_of_replay),
    )
    all_rows = c_rows + d_rows
    if not all_rows:
        raise SystemExit("no trades found in replay inputs")

    days = sorted({row.day for row in all_rows})
    folds = _build_day_folds(
        days,
        train_days=int(args.train_days),
        test_days=int(args.test_days),
        step_days=int(args.step_days),
    )
    if not folds:
        raise SystemExit(
            f"not enough days for folds: days={len(days)} train={args.train_days} test={args.test_days}"
        )

    fold_results: list[dict[str, Any]] = []
    selected_all: list[RouteTrade] = []
    for idx, fold in enumerate(folds, start=1):
        train_set = set(fold["train"])
        test_set = set(fold["test"])
        train_rows = [row for row in all_rows if row.day in train_set]
        test_rows = [row for row in all_rows if row.day in test_set]
        mapping = _choose_route_mapping(
            train_rows,
            workers=workers,
            default_worker=default_worker,
            min_train_route_trades=int(args.min_train_route_trades),
        )
        selected_test = _apply_mapping(test_rows, mapping)
        selected_all.extend(selected_test)
        train_metrics = compute_trade_metrics(_rows_to_metric_trades(_apply_mapping(train_rows, mapping)))
        test_metrics = compute_trade_metrics(_rows_to_metric_trades(selected_test))

        fold_results.append(
            {
                "fold": idx,
                "train_days": fold["train"],
                "test_days": fold["test"],
                "route_mapping": mapping,
                "train_metrics_selected": train_metrics,
                "test_metrics_selected": test_metrics,
                "test_trade_count_selected": int(test_metrics.get("trade_count") or 0.0),
            }
        )

    test_jpy_per_hour_values = [
        float(item["test_metrics_selected"].get("jpy_per_hour") or 0.0) for item in fold_results
    ]
    test_total_jpy_values = [
        float(item["test_metrics_selected"].get("total_jpy") or 0.0) for item in fold_results
    ]
    summary_selected = compute_trade_metrics(_rows_to_metric_trades(selected_all))
    achieved_jpy_per_hour = float(summary_selected.get("jpy_per_hour") or 0.0)
    achieved_trades_per_hour = 0.0
    duration_hours = float(summary_selected.get("duration_hours") or 0.0)
    trade_count = float(summary_selected.get("trade_count") or 0.0)
    if duration_hours > 0.0:
        achieved_trades_per_hour = trade_count / duration_hours
    achieved_jpy_per_trade = 0.0 if trade_count <= 0 else float(summary_selected.get("total_jpy") or 0.0) / trade_count

    target = float(args.target_jpy_per_hour)
    required_jpy_per_trade = 0.0
    if achieved_trades_per_hour > 0.0:
        required_jpy_per_trade = target / achieved_trades_per_hour
    required_trades_per_hour = 0.0
    if achieved_jpy_per_trade > 0.0:
        required_trades_per_hour = target / achieved_jpy_per_trade

    worker_to_strategy = {
        worker_c: str(args.strategy_c).strip(),
        worker_d: str(args.strategy_d).strip(),
    }
    env_suggestion = _build_env_suggestion(
        [item["route_mapping"] for item in fold_results],
        worker_to_strategy=worker_to_strategy,
    )

    payload = {
        "meta": {
            "replay_c": str(args.replay_c.resolve()),
            "replay_d": str(args.replay_d.resolve()),
            "workers": list(workers),
            "strategies": worker_to_strategy,
            "days": days,
            "train_days": int(args.train_days),
            "test_days": int(args.test_days),
            "step_days": int(args.step_days),
            "min_train_route_trades": int(args.min_train_route_trades),
            "default_worker": default_worker,
            "exclude_end_of_replay": bool(args.exclude_end_of_replay),
            "target_jpy_per_hour": target,
        },
        "fold_results": fold_results,
        "summary": {
            "fold_count": len(fold_results),
            "median_test_jpy_per_hour_selected": _median(test_jpy_per_hour_values),
            "median_test_total_jpy_selected": _median(test_total_jpy_values),
            "selected_metrics_all": summary_selected,
            "achieved_jpy_per_hour": achieved_jpy_per_hour,
            "achieved_trades_per_hour": achieved_trades_per_hour,
            "achieved_jpy_per_trade": achieved_jpy_per_trade,
            "required_jpy_per_trade_at_current_freq": required_jpy_per_trade,
            "required_trades_per_hour_at_current_ev": required_trades_per_hour,
        },
        "env_suggestion": env_suggestion,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Replay Regime Router WFO",
        "",
        f"- replay_c: `{args.replay_c.resolve()}`",
        f"- replay_d: `{args.replay_d.resolve()}`",
        f"- folds: `{len(fold_results)}`",
        f"- median test jpy/h (selected): `{_median(test_jpy_per_hour_values):.3f}`",
        f"- achieved jpy/h (all selected test): `{achieved_jpy_per_hour:.3f}`",
        f"- target jpy/h: `{target:.3f}`",
        f"- required jpy/trade at current freq: `{required_jpy_per_trade:.3f}`",
        f"- required trades/h at current ev: `{required_trades_per_hour:.3f}`",
        "",
        "## Suggested Env",
        "",
    ]
    for key in sorted(env_suggestion):
        md_lines.append(f"- `{key}={env_suggestion[key]}`")
    if not env_suggestion:
        md_lines.append("- `(none)`")
    md_lines.append("")
    md_lines.append("## Fold Mappings")
    md_lines.append("")
    for item in fold_results:
        mapping = item["route_mapping"]
        route_text = ", ".join(f"{route}:{mapping.get(route, '-')}" for route in ROUTES)
        test_metrics = item["test_metrics_selected"]
        md_lines.append(
            f"- fold {item['fold']} test={item['test_days']} routes=`{route_text}` "
            f"test_jpy/h={float(test_metrics.get('jpy_per_hour') or 0.0):.3f} "
            f"test_total_jpy={float(test_metrics.get('total_jpy') or 0.0):.2f}"
        )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(str(args.out_json))


if __name__ == "__main__":
    main()

