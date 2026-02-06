#!/usr/bin/env python3
"""Local-only ops report (no LLM).

This script keeps the historical CLI surface for systemd/cron callers.
It never calls any LLM providers and only reads local artifacts (sqlite DBs,
health snapshot JSON, and systemd status when available).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from analytics.policy_diff import normalize_policy_diff


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2))

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    return _utcnow().isoformat(timespec="seconds")


def _parse_iso(text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    if q <= 0:
        return float(min(values))
    if q >= 1:
        return float(max(values))
    data = sorted(values)
    n = len(data)
    if n == 1:
        return float(data[0])
    k = (n - 1) * q
    f = int(k)
    c = min(f + 1, n - 1)
    if f == c:
        return float(data[int(k)])
    d0 = data[f] * (c - k)
    d1 = data[c] * (k - f)
    return float(d0 + d1)


def _run_cmd(args: List[str], timeout: float = 2.0) -> Optional[str]:
    try:
        proc = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return None
    out = (proc.stdout or "").strip()
    if proc.returncode != 0 and not out:
        return None
    return out or None


def _git_rev(root: Path) -> Optional[str]:
    if not (root / ".git").exists():
        return None
    out = _run_cmd(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], timeout=2.0)
    return out.strip() if out else None


def _systemd_unit_info(unit: str) -> Optional[Dict[str, Any]]:
    if not shutil.which("systemctl"):
        return None
    out = _run_cmd(
        [
            "systemctl",
            "show",
            unit,
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "Result",
            "-p",
            "NRestarts",
            "-p",
            "ActiveEnterTimestamp",
            "-p",
            "ExecMainStartTimestamp",
            "-p",
            "ExecMainExitTimestamp",
            "-p",
            "ExecMainStatus",
        ],
        timeout=2.0,
    )
    if not out:
        return None
    info: Dict[str, Any] = {"unit": unit}
    for line in out.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        info[key] = value
    return info


def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _sqlite_rows(db_path: Path, query: str, params: Tuple[Any, ...]) -> List[sqlite3.Row]:
    if not db_path.exists():
        return []
    con = sqlite3.connect(str(db_path))
    try:
        con.row_factory = sqlite3.Row
        cur = con.execute(query, params)
        return cur.fetchall()
    except Exception:
        return []
    finally:
        try:
            con.close()
        except Exception:
            pass


def _metric_values(metrics_db: Path, metric: str, since_ts: str) -> List[float]:
    rows = _sqlite_rows(
        metrics_db,
        "SELECT value FROM metrics WHERE metric = ? AND ts >= ?",
        (metric, since_ts),
    )
    out: List[float] = []
    for row in rows:
        try:
            out.append(float(row[0]))
        except Exception:
            continue
    return out


def _metric_last(metrics_db: Path, metric: str) -> Optional[Dict[str, Any]]:
    rows = _sqlite_rows(
        metrics_db,
        "SELECT ts,value,tags FROM metrics WHERE metric = ? ORDER BY ts DESC LIMIT 1",
        (metric,),
    )
    if not rows:
        return None
    row = rows[0]
    tags = None
    try:
        tags_raw = row["tags"]
        if tags_raw:
            tags = json.loads(tags_raw)
    except Exception:
        tags = None
    return {"ts": row["ts"], "value": row["value"], "tags": tags}


def _metrics_summary(metrics_db: Path, metric: str, since_ts: str) -> Dict[str, Any]:
    values = _metric_values(metrics_db, metric, since_ts)
    last = _metric_last(metrics_db, metric)
    summary: Dict[str, Any] = {
        "count": len(values),
        "last": last,
    }
    if values:
        summary.update(
            {
                "min": min(values),
                "max": max(values),
                "p50": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
            }
        )
    return summary


def _orders_status_counts(orders_db: Path, since_ts: str) -> Dict[str, Any]:
    rows = _sqlite_rows(
        orders_db,
        "SELECT status, error_code FROM orders WHERE ts >= ?",
        (since_ts,),
    )
    status = Counter()
    error_codes = Counter()
    failed = 0
    total = 0
    for row in rows:
        total += 1
        st = str(row["status"] or "unknown")
        status[st] += 1
        code = row["error_code"]
        if code:
            error_codes[str(code)] += 1
            failed += 1
        else:
            if st.upper() in {"REJECTED", "FAILED", "ERROR", "CANCELLED"}:
                failed += 1
    out: Dict[str, Any] = {
        "total": total,
        "failed": failed,
        "fail_rate": round((failed / total), 4) if total else 0.0,
        "by_status": dict(status.most_common(40)),
        "top_error_codes": dict(error_codes.most_common(20)),
    }
    # Reject streak: consecutive latest failures (cap at 20)
    streak_rows = _sqlite_rows(
        orders_db,
        "SELECT status, error_code FROM orders ORDER BY ts DESC LIMIT 20",
        (),
    )
    streak = 0
    for row in streak_rows:
        st = str(row["status"] or "").upper()
        code = row["error_code"]
        failed_row = bool(code) or st in {"REJECTED", "FAILED", "ERROR", "CANCELLED"}
        if not failed_row:
            break
        streak += 1
    out["reject_streak"] = streak
    return out


def _parse_entry_thesis(text: Optional[str]) -> Dict[str, Any]:
    if not text:
        return {}
    raw = str(text).strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _trade_rows(trades_db: Path, since_ts: str) -> List[sqlite3.Row]:
    return _sqlite_rows(
        trades_db,
        "SELECT pocket,strategy_tag,strategy,pl_pips,realized_pl,entry_time,close_time,state,entry_thesis "
        "FROM trades WHERE close_time >= ?",
        (since_ts,),
    )


def _trade_summary(trades_db: Path, since_ts: str) -> Dict[str, Any]:
    rows = _trade_rows(trades_db, since_ts)
    total = 0
    wins = 0
    total_pips = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    by_pocket: Dict[str, Dict[str, float]] = defaultdict(lambda: {"n": 0, "wins": 0, "pips": 0.0})
    by_strategy: Dict[str, Dict[str, float]] = defaultdict(lambda: {"n": 0, "wins": 0, "pips": 0.0})
    for row in rows:
        if str(row["state"] or "").upper() != "CLOSED":
            continue
        total += 1
        pips = 0.0
        try:
            pips = float(row["pl_pips"] or 0.0)
        except Exception:
            pips = 0.0
        total_pips += pips
        if pips > 0:
            wins += 1
            gross_profit += pips
        elif pips < 0:
            gross_loss += abs(pips)
        pocket = str(row["pocket"] or "unknown")
        tag = str(row["strategy_tag"] or row["strategy"] or "unknown")
        by_pocket[pocket]["n"] += 1
        by_pocket[pocket]["wins"] += 1 if pips > 0 else 0
        by_pocket[pocket]["pips"] += pips
        by_strategy[tag]["n"] += 1
        by_strategy[tag]["wins"] += 1 if pips > 0 else 0
        by_strategy[tag]["pips"] += pips

    pf = (gross_profit / gross_loss) if gross_loss > 0 else None
    win_rate = (wins / total) if total else None
    pocket_rank = sorted(
        (
            {
                "pocket": p,
                "trades": int(v["n"]),
                "win_rate": round((v["wins"] / v["n"]), 4) if v["n"] else None,
                "total_pips": round(v["pips"], 2),
            }
            for p, v in by_pocket.items()
        ),
        key=lambda x: x.get("total_pips") or 0.0,
    )
    strat_rank = sorted(
        (
            {
                "strategy_tag": s,
                "trades": int(v["n"]),
                "win_rate": round((v["wins"] / v["n"]), 4) if v["n"] else None,
                "total_pips": round(v["pips"], 2),
            }
            for s, v in by_strategy.items()
        ),
        key=lambda x: x.get("total_pips") or 0.0,
    )
    return {
        "closed_trades": total,
        "win_rate": round(win_rate, 4) if win_rate is not None else None,
        "profit_factor": round(pf, 4) if pf is not None else None,
        "total_pips": round(total_pips, 2),
        "pockets_top": pocket_rank[-8:][::-1],
        "pockets_bottom": pocket_rank[:8],
        "strategies_top": strat_rank[-10:][::-1],
        "strategies_bottom": strat_rank[:10],
    }


def _journal_summary(trades_db: Path, since_ts: str, *, min_trades: int = 6) -> Dict[str, Any]:
    rows = _trade_rows(trades_db, since_ts)
    # Aggregate by (pocket, strategy_tag, reason/pattern_tag)
    buckets: Dict[Tuple[str, str, str], Dict[str, float]] = defaultdict(
        lambda: {"n": 0, "wins": 0, "gross_profit": 0.0, "gross_loss": 0.0, "pips": 0.0}
    )
    for row in rows:
        if str(row["state"] or "").upper() != "CLOSED":
            continue
        pocket = str(row["pocket"] or "unknown")
        strategy_tag = str(row["strategy_tag"] or row["strategy"] or "unknown")
        thesis = _parse_entry_thesis(row["entry_thesis"])
        reason = str(thesis.get("reason") or thesis.get("pattern_tag") or thesis.get("entry_type") or "unknown")
        pips = 0.0
        try:
            pips = float(row["pl_pips"] or 0.0)
        except Exception:
            pips = 0.0
        key = (pocket, strategy_tag, reason)
        b = buckets[key]
        b["n"] += 1
        if pips > 0:
            b["wins"] += 1
            b["gross_profit"] += pips
        elif pips < 0:
            b["gross_loss"] += abs(pips)
        b["pips"] += pips

    rows_out: List[Dict[str, Any]] = []
    for (pocket, strategy_tag, reason), b in buckets.items():
        n = int(b["n"])
        if n < min_trades:
            continue
        gp = float(b["gross_profit"])
        gl = float(b["gross_loss"])
        pf = (gp / gl) if gl > 0 else None
        win = float(b["wins"]) / n if n else None
        rows_out.append(
            {
                "pocket": pocket,
                "strategy_tag": strategy_tag,
                "reason": reason,
                "trades": n,
                "win_rate": round(win, 4) if win is not None else None,
                "profit_factor": round(pf, 4) if pf is not None else None,
                "total_pips": round(float(b["pips"]), 2),
            }
        )
    # Worst first (low total_pips)
    rows_out.sort(key=lambda x: x.get("total_pips") or 0.0)
    return {
        "since_ts": since_ts,
        "min_trades": min_trades,
        "worst": rows_out[:12],
        "best": rows_out[-12:][::-1],
    }


def _build_findings(
    *,
    metrics_db: Path,
    orders_db: Path,
    trades_db: Path,
    since_ts: str,
) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    if metrics_db.exists():
        # SLO-ish checks (same knobs as policy_guard defaults).
        decision = _metrics_summary(metrics_db, "decision_latency_ms", since_ts)
        lag = _metrics_summary(metrics_db, "data_lag_ms", since_ts)
        drawdown = _metrics_summary(metrics_db, "drawdown_pct", since_ts)
        reject_rate = _metrics_summary(metrics_db, "reject_rate", since_ts)

        if decision.get("p95") is not None and float(decision["p95"]) > float(os.getenv("POLICY_GUARD_MAX_DECISION_MS", "2000")):
            findings.append(
                {
                    "severity": "P1",
                    "code": "decision_latency_high",
                    "message": "decision_latency_ms p95 exceeds threshold",
                    "data": {"p95": decision.get("p95")},
                }
            )
        if lag.get("p95") is not None and float(lag["p95"]) > float(os.getenv("POLICY_GUARD_MAX_DATA_LAG_MS", "1500")):
            findings.append(
                {
                    "severity": "P1",
                    "code": "data_lag_high",
                    "message": "data_lag_ms p95 exceeds threshold",
                    "data": {"p95": lag.get("p95")},
                }
            )
        if drawdown.get("max") is not None and float(drawdown["max"]) > float(os.getenv("POLICY_GUARD_MAX_DRAWDOWN_PCT", "0.18")):
            findings.append(
                {
                    "severity": "P1",
                    "code": "drawdown_high",
                    "message": "drawdown_pct max exceeds threshold",
                    "data": {"max": drawdown.get("max")},
                }
            )
        if reject_rate.get("max") is not None and float(reject_rate["max"]) > float(os.getenv("POLICY_GUARD_MAX_REJECT_RATE", "0.01")):
            findings.append(
                {
                    "severity": "P1",
                    "code": "reject_rate_high",
                    "message": "reject_rate exceeds threshold",
                    "data": {"max": reject_rate.get("max")},
                }
            )

        brain_last = _metric_last(metrics_db, "brain_latency_ms")
        if brain_last and isinstance(brain_last.get("tags"), dict):
            ok = bool((brain_last["tags"] or {}).get("ok"))
            if not ok:
                findings.append(
                    {
                        "severity": "P2",
                        "code": "brain_call_failed",
                        "message": "brain_latency_ms recorded with ok=false (LLM call failed or disabled mid-flight)",
                        "data": {"last": brain_last},
                    }
                )

    if orders_db.exists():
        o = _orders_status_counts(orders_db, since_ts)
        if int(o.get("reject_streak") or 0) >= int(os.getenv("POLICY_GUARD_REJECT_STREAK", "3")):
            findings.append(
                {
                    "severity": "P1",
                    "code": "reject_streak",
                    "message": "recent consecutive order failures exceed threshold",
                    "data": {"reject_streak": o.get("reject_streak")},
                }
            )
        if float(o.get("fail_rate") or 0.0) >= 0.02 and int(o.get("total") or 0) >= 50:
            findings.append(
                {
                    "severity": "P2",
                    "code": "order_fail_rate_high",
                    "message": "order fail rate elevated",
                    "data": {"fail_rate": o.get("fail_rate"), "total": o.get("total")},
                }
            )

    if trades_db.exists():
        t = _trade_summary(trades_db, since_ts)
        if t.get("profit_factor") is not None and float(t["profit_factor"]) < 0.9 and int(t.get("closed_trades") or 0) >= 30:
            findings.append(
                {
                    "severity": "P2",
                    "code": "pf_low",
                    "message": "recent profit factor is low",
                    "data": {"profit_factor": t.get("profit_factor"), "closed_trades": t.get("closed_trades")},
                }
            )

    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description="Ops report (local-only; no LLM)")
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--output", default="logs/gpt_ops_report.json")
    ap.add_argument("--policy", action="store_true")
    ap.add_argument("--policy-output", default="logs/policy_diff_ops.json")
    ap.add_argument("--apply-policy", action="store_true")
    ap.add_argument("--gpt", action="store_true")
    ap.add_argument("--journal-days", type=float, default=7.0)
    ap.add_argument("--log-level", default="INFO")
    args, _ = ap.parse_known_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    project_root = Path(ROOT_DIR)
    logs_dir = project_root / "logs"
    metrics_db = Path(os.getenv("OPS_METRICS_DB", str(logs_dir / "metrics.db")))
    orders_db = Path(os.getenv("OPS_ORDERS_DB", str(logs_dir / "orders.db")))
    trades_db = Path(os.getenv("OPS_TRADES_DB", str(logs_dir / "trades.db")))
    health_path = Path(os.getenv("OPS_HEALTH_SNAPSHOT_PATH", str(logs_dir / "health_snapshot.json")))

    now = _utcnow()
    since = now - timedelta(hours=float(args.hours))
    since_ts = since.replace(tzinfo=None).isoformat()
    journal_since = now - timedelta(days=float(args.journal_days))
    journal_since_ts = journal_since.replace(tzinfo=None).isoformat()

    report: Dict[str, Any] = {
        "report_version": 2,
        "generated_at": _utcnow_iso(),
        "llm_disabled": True,
        "window_hours": float(args.hours),
        "journal_window_days": float(args.journal_days),
        "hostname": socket.gethostname(),
        "git_rev": _git_rev(project_root),
        "paths": {
            "metrics_db": str(metrics_db),
            "orders_db": str(orders_db),
            "trades_db": str(trades_db),
            "health_snapshot": str(health_path),
        },
        "env": {
            # Keep it explicit: Brain is the only supported LLM gate; do not leak secrets.
            "BRAIN_ENABLED": os.getenv("BRAIN_ENABLED", ""),
            "VERTEX_PROJECT_ID": "***" if os.getenv("VERTEX_PROJECT_ID") else "",
        },
    }

    health = _safe_load_json(health_path)
    if health:
        # keep the payload small: only retain core indicators
        report["health_snapshot"] = {
            "generated_at": health.get("generated_at"),
            "deploy_id": health.get("deploy_id"),
            "git_rev": health.get("git_rev"),
            "uptime_sec": health.get("uptime_sec"),
            "trades_last_entry": health.get("trades_last_entry"),
            "trades_last_close": health.get("trades_last_close"),
            "orders_last_ts": health.get("orders_last_ts"),
            "signals_last_ts": health.get("signals_last_ts"),
            "services": health.get("services"),
        }

    # systemd (best-effort)
    units = [
        "quantrabbit.service",
        "quant-bq-sync.service",
        "quant-ui-snapshot.service",
        "quant-health-snapshot.service",
        "quant-ops-policy.timer",
        "quant-policy-cycle.timer",
        "quant-policy-guard.timer",
    ]
    unit_info = []
    for unit in units:
        info = _systemd_unit_info(unit)
        if info:
            unit_info.append(info)
    if unit_info:
        report["systemd"] = {"units": unit_info}

    if metrics_db.exists():
        report["metrics"] = {
            "since_ts": since_ts,
            "decision_latency_ms": _metrics_summary(metrics_db, "decision_latency_ms", since_ts),
            "data_lag_ms": _metrics_summary(metrics_db, "data_lag_ms", since_ts),
            "drawdown_pct": _metrics_summary(metrics_db, "drawdown_pct", since_ts),
            "reject_rate": _metrics_summary(metrics_db, "reject_rate", since_ts),
            "order_success_rate": _metrics_summary(metrics_db, "order_success_rate", since_ts),
            "brain_latency_ms": _metrics_summary(metrics_db, "brain_latency_ms", since_ts),
        }

    if orders_db.exists():
        report["orders"] = {
            "since_ts": since_ts,
            "summary": _orders_status_counts(orders_db, since_ts),
        }

    if trades_db.exists():
        report["trades"] = {
            "since_ts": since_ts,
            "summary": _trade_summary(trades_db, since_ts),
        }
        report["journal"] = _journal_summary(trades_db, journal_since_ts, min_trades=6)

    report["findings"] = _build_findings(
        metrics_db=metrics_db,
        orders_db=orders_db,
        trades_db=trades_db,
        since_ts=since_ts,
    )

    _write_json(Path(args.output), report)
    logging.info("[OPS_REPORT] wrote %s", args.output)

    if args.policy or args.apply_policy:
        diff = normalize_policy_diff(
            {"no_change": True, "reason": "llm_disabled", "source": "ops_stub"},
            source="ops_stub",
        )
        _write_json(Path(args.policy_output), diff)
        logging.info("[OPS_POLICY] wrote %s", args.policy_output)
        if args.apply_policy:
            logging.info("[OPS_POLICY] apply requested but LLM is disabled; skipping.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
