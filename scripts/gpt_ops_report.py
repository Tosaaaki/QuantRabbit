#!/usr/bin/env python3
"""Generate an ops report from local logs and optionally summarise with GPT.

Usage:
  python scripts/gpt_ops_report.py --hours 24 --output logs/gpt_ops_report.json
  python scripts/gpt_ops_report.py --hours 24 --gpt
  python scripts/gpt_ops_report.py --hours 6 --policy --policy-output logs/policy_diff_ops.json
  python scripts/gpt_ops_report.py --hours 6 --policy --apply-policy
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

from utils.secrets import get_secret
from utils.vertex_client import call_vertex_text
from analytics.policy_apply import apply_policy_diff_to_paths
from analytics.policy_diff import POLICY_DIFF_SCHEMA, normalize_policy_diff, validate_policy_diff

_SUMMARY_PROVIDER = (
    os.getenv("GPT_SUMMARIZER_PROVIDER")
    or os.getenv("LLM_SUMMARY_PROVIDER")
    or os.getenv("LLM_PROVIDER")
    or "openai"
).strip().lower()
_VERTEX_SUMMARIZER_MODEL = (
    os.getenv("VERTEX_SUMMARIZER_MODEL")
    or os.getenv("VERTEX_MODEL")
    or os.getenv("VERTEX_POLICY_MODEL")
    or "gemini-2.0-flash"
)
_VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", os.getenv("GCP_LOCATION", "us-central1"))
_VERTEX_PROJECT = os.getenv("VERTEX_PROJECT_ID") or os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
_VERTEX_TIMEOUT_SEC = float(os.getenv("VERTEX_SUMMARIZER_TIMEOUT_SEC", "20"))
_VERTEX_PROVIDERS = {"vertex", "vertex_ai", "vertexai", "gemini", "gcp"}
_OPENAI_PROVIDERS = {"openai", "oai"}
_POLICY_PROVIDER = (
    os.getenv("LLM_OPS_POLICY_PROVIDER")
    or os.getenv("LLM_PROVIDER")
    or _SUMMARY_PROVIDER
    or "vertex"
).strip().lower()
_VERTEX_POLICY_MODEL = (
    os.getenv("LLM_OPS_POLICY_MODEL")
    or os.getenv("VERTEX_POLICY_MODEL")
    or _VERTEX_SUMMARIZER_MODEL
    or "gemini-2.0-flash"
)
_VERTEX_POLICY_TIMEOUT_SEC = float(os.getenv("LLM_OPS_POLICY_TIMEOUT_SEC", _VERTEX_TIMEOUT_SEC))
_POLICY_MAX_TOKENS = int(os.getenv("LLM_OPS_POLICY_MAX_TOKENS", "900"))


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _iso(dt_obj: dt.datetime) -> str:
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=dt.timezone.utc)
    return dt_obj.astimezone(dt.timezone.utc).isoformat()


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


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
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(data[int(k)])
    d0 = data[int(f)] * (c - k)
    d1 = data[int(c)] * (k - f)
    return float(d0 + d1)


def _connect(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def _trade_aggregate(con: sqlite3.Connection, since_ts: str) -> Dict[str, Any]:
    row = con.execute(
        """
        SELECT
          COUNT(*) AS trades,
          SUM(CASE WHEN pl_pips > 0 THEN 1 ELSE 0 END) AS wins,
          SUM(CASE WHEN pl_pips < 0 THEN 1 ELSE 0 END) AS losses,
          SUM(CASE WHEN pl_pips > 0 THEN pl_pips ELSE 0 END) AS win_pips,
          SUM(CASE WHEN pl_pips < 0 THEN pl_pips ELSE 0 END) AS loss_pips,
          SUM(pl_pips) AS sum_pips,
          SUM(realized_pl) AS sum_jpy
        FROM trades
        WHERE close_time IS NOT NULL AND close_time >= ?
        """,
        (since_ts,),
    ).fetchone()
    trades = _safe_int(row["trades"]) if row else 0
    wins = _safe_int(row["wins"]) if row else 0
    losses = _safe_int(row["losses"]) if row else 0
    win_pips = _safe_float(row["win_pips"]) if row else 0.0
    loss_pips = _safe_float(row["loss_pips"]) if row else 0.0
    sum_pips = _safe_float(row["sum_pips"]) if row else 0.0
    sum_jpy = _safe_float(row["sum_jpy"]) if row else 0.0
    win_rate = (wins / trades) if trades else 0.0
    pf = (win_pips / abs(loss_pips)) if loss_pips < 0 else None
    avg_pips = (sum_pips / trades) if trades else 0.0
    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "pf": round(pf, 3) if pf is not None else None,
        "sum_pips": round(sum_pips, 2),
        "avg_pips": round(avg_pips, 2),
        "sum_jpy": round(sum_jpy, 2),
    }


def _trade_by_pocket(con: sqlite3.Connection, since_ts: str) -> Dict[str, Any]:
    rows = con.execute(
        """
        SELECT
          COALESCE(pocket, 'unknown') AS pocket,
          COUNT(*) AS trades,
          SUM(CASE WHEN pl_pips > 0 THEN 1 ELSE 0 END) AS wins,
          SUM(CASE WHEN pl_pips < 0 THEN 1 ELSE 0 END) AS losses,
          SUM(CASE WHEN pl_pips > 0 THEN pl_pips ELSE 0 END) AS win_pips,
          SUM(CASE WHEN pl_pips < 0 THEN pl_pips ELSE 0 END) AS loss_pips,
          SUM(pl_pips) AS sum_pips,
          SUM(realized_pl) AS sum_jpy
        FROM trades
        WHERE close_time IS NOT NULL AND close_time >= ?
        GROUP BY pocket
        """,
        (since_ts,),
    ).fetchall()
    out: Dict[str, Any] = {}
    for row in rows:
        pocket = str(row["pocket"] or "unknown")
        trades = _safe_int(row["trades"])
        wins = _safe_int(row["wins"])
        losses = _safe_int(row["losses"])
        win_pips = _safe_float(row["win_pips"])
        loss_pips = _safe_float(row["loss_pips"])
        sum_pips = _safe_float(row["sum_pips"])
        sum_jpy = _safe_float(row["sum_jpy"])
        win_rate = (wins / trades) if trades else 0.0
        pf = (win_pips / abs(loss_pips)) if loss_pips < 0 else None
        avg_pips = (sum_pips / trades) if trades else 0.0
        out[pocket] = {
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 4),
            "pf": round(pf, 3) if pf is not None else None,
            "sum_pips": round(sum_pips, 2),
            "avg_pips": round(avg_pips, 2),
            "sum_jpy": round(sum_jpy, 2),
        }
    return out


def _trade_by_strategy(
    con: sqlite3.Connection,
    since_ts: str,
    *,
    limit: int = 8,
    min_trades: int = 3,
) -> Dict[str, List[Dict[str, Any]]]:
    rows = con.execute(
        """
        SELECT
          COALESCE(strategy_tag, strategy, 'unknown') AS strategy,
          COALESCE(pocket, 'unknown') AS pocket,
          COUNT(*) AS trades,
          SUM(CASE WHEN pl_pips > 0 THEN 1 ELSE 0 END) AS wins,
          SUM(CASE WHEN pl_pips < 0 THEN 1 ELSE 0 END) AS losses,
          SUM(CASE WHEN pl_pips > 0 THEN pl_pips ELSE 0 END) AS win_pips,
          SUM(CASE WHEN pl_pips < 0 THEN pl_pips ELSE 0 END) AS loss_pips,
          SUM(pl_pips) AS sum_pips,
          SUM(realized_pl) AS sum_jpy
        FROM trades
        WHERE close_time IS NOT NULL AND close_time >= ?
        GROUP BY strategy, pocket
        """,
        (since_ts,),
    ).fetchall()
    results: List[Dict[str, Any]] = []
    for row in rows:
        trades = _safe_int(row["trades"])
        if trades < min_trades:
            continue
        wins = _safe_int(row["wins"])
        losses = _safe_int(row["losses"])
        win_pips = _safe_float(row["win_pips"])
        loss_pips = _safe_float(row["loss_pips"])
        sum_pips = _safe_float(row["sum_pips"])
        sum_jpy = _safe_float(row["sum_jpy"])
        win_rate = (wins / trades) if trades else 0.0
        pf = (win_pips / abs(loss_pips)) if loss_pips < 0 else None
        avg_pips = (sum_pips / trades) if trades else 0.0
        results.append(
            {
                "strategy": str(row["strategy"] or "unknown"),
                "pocket": str(row["pocket"] or "unknown"),
                "trades": trades,
                "wins": wins,
                "losses": losses,
                "win_rate": round(win_rate, 4),
                "pf": round(pf, 3) if pf is not None else None,
                "sum_pips": round(sum_pips, 2),
                "avg_pips": round(avg_pips, 2),
                "sum_jpy": round(sum_jpy, 2),
            }
        )
    top = sorted(results, key=lambda r: r["sum_pips"], reverse=True)[:limit]
    bottom = sorted(results, key=lambda r: r["sum_pips"])[:limit]
    return {"top": top, "bottom": bottom}


def _order_stats(con: sqlite3.Connection, since_ts: str) -> Dict[str, Any]:
    rows = con.execute(
        "SELECT status, error_code FROM orders WHERE ts >= ?",
        (since_ts,),
    ).fetchall()
    total = len(rows)
    status_counts: Dict[str, int] = {}
    error_counts: Dict[str, int] = {}
    failed = 0
    for row in rows:
        status = str(row["status"] or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        error_code = row["error_code"]
        if error_code:
            code = str(error_code)
            error_counts[code] = error_counts.get(code, 0) + 1
        status_upper = status.upper()
        if status_upper in {"REJECTED", "FAILED", "CANCELLED", "ERROR"} or error_code:
            failed += 1
    reject_rate = (failed / total) if total else 0.0
    top_errors = sorted(error_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return {
        "total": total,
        "failed": failed,
        "reject_rate": round(reject_rate, 4),
        "status_counts": status_counts,
        "top_error_codes": [{"code": k, "count": v} for k, v in top_errors],
    }


def _metric_stats(con: sqlite3.Connection, since_ts: str, metric: str) -> Dict[str, Any]:
    rows = con.execute(
        "SELECT value FROM metrics WHERE metric = ? AND ts >= ?",
        (metric, since_ts),
    ).fetchall()
    values = [_safe_float(row["value"]) for row in rows]
    p50 = _percentile(values, 0.50)
    p95 = _percentile(values, 0.95)
    out = {
        "count": len(values),
        "p50": round(p50, 3) if p50 is not None else None,
        "p95": round(p95, 3) if p95 is not None else None,
        "max": round(max(values), 3) if values else None,
    }
    return out


def _build_flags(report: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    latency = report.get("metrics", {}).get("decision_latency_ms", {})
    data_lag = report.get("metrics", {}).get("data_lag_ms", {})
    orders = report.get("orders", {})
    overall = report.get("overall", {})
    if latency.get("p95") and latency["p95"] > 2000:
        flags.append("decision_latency_p95_high")
    if data_lag.get("p95") and data_lag["p95"] > 1500:
        flags.append("data_lag_p95_high")
    if orders.get("reject_rate") and orders["reject_rate"] > 0.01:
        flags.append("order_reject_rate_high")
    if overall.get("trades", 0) >= 10 and overall.get("win_rate", 1.0) < 0.48:
        flags.append("overall_win_rate_low")
    if overall.get("trades", 0) >= 10 and overall.get("pf") and overall["pf"] < 0.9:
        flags.append("overall_pf_low")
    return flags


def _resolve_model() -> str:
    for key in ("OPENAI_SUMMARIZER_MODEL", "OPENAI_MODEL"):
        val = os.getenv(key)
        if val:
            return val
    try:
        return get_secret("openai_model_summarizer")
    except Exception:
        pass
    try:
        return get_secret("openai_model")
    except Exception:
        return "gpt-4o-mini"


def _resolve_provider(value: str) -> str:
    raw = (value or "").strip().lower()
    if not raw or raw in _OPENAI_PROVIDERS or raw in {"auto", "default"}:
        return "openai"
    if raw in _VERTEX_PROVIDERS:
        return "vertex"
    return raw


def _parse_json_text(text: str) -> Optional[Dict[str, Any]]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json", "", 1).strip()
    try:
        return json.loads(raw)
    except Exception:
        return None


def _extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
    raw = text.strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json", "", 1).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except Exception:
        return None


def _parse_policy_diff(text: str, *, source: str) -> Optional[Dict[str, Any]]:
    payload = _extract_json_payload(text)
    if not isinstance(payload, dict):
        return None
    payload = normalize_policy_diff(payload, source=source)
    errors = validate_policy_diff(payload)
    if errors:
        logging.warning("[OPS_POLICY] invalid policy_diff: %s", ", ".join(errors))
        return None
    return payload


def _build_policy_prompt(report: Dict[str, Any]) -> str:
    schema_text = json.dumps(POLICY_DIFF_SCHEMA, ensure_ascii=True, separators=(",", ":"))
    payload = {
        "window": report.get("window"),
        "overall": report.get("overall"),
        "pockets": report.get("pockets"),
        "strategy_top": report.get("strategies", {}).get("top", []),
        "strategy_bottom": report.get("strategies", {}).get("bottom", []),
        "orders": report.get("orders"),
        "metrics": report.get("metrics"),
        "flags": report.get("flags"),
    }
    payload_text = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    return (
        "You are QuantRabbit's ops policy agent.\n"
        "Return ONLY JSON that conforms to this JSON schema:\n"
        f"{schema_text}\n\n"
        "Rules:\n"
        "- If data is insufficient or uncertain, set no_change=true and omit patch.\n"
        "- Prefer targeted changes: use pockets.*.strategies allowlist, bias, and entry_gates.\n"
        "- Avoid blocking all pockets unless severe risk is present.\n"
        "- You may use tuning_overrides for exit/partial/trail tweaks (small deltas).\n\n"
        "Input JSON:\n"
        f"{payload_text}\n"
    )


def _openai_summary(report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if OpenAI is None:
        return None
    try:
        api_key = get_secret("openai_api_key")
    except Exception:
        api_key = None
    if not api_key:
        return None

    client = OpenAI(api_key=api_key)
    model = _resolve_model()
    payload = {
        "window": report.get("window"),
        "overall": report.get("overall"),
        "pockets": report.get("pockets"),
        "strategy_top": report.get("strategies", {}).get("top", []),
        "strategy_bottom": report.get("strategies", {}).get("bottom", []),
        "orders": report.get("orders"),
        "metrics": report.get("metrics"),
        "flags": report.get("flags"),
    }
    system_prompt = (
        "You are an FX ops analyst. Summarize performance, highlight risks, and "
        "propose 3-5 actions. Do not suggest direct trade entries. "
        "Return JSON with keys: summary, risks(list), actions(list of {type,target,reason,confidence})."
    )
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
        ],
        max_output_tokens=400,
        temperature=0.2,
    )
    try:
        text = response.output_text.strip()
    except Exception:
        text = ""
    parsed = _parse_json_text(text) if text else None
    if parsed is not None:
        return parsed
    return {"summary": text}


def _vertex_summary(report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    payload = {
        "window": report.get("window"),
        "overall": report.get("overall"),
        "pockets": report.get("pockets"),
        "strategy_top": report.get("strategies", {}).get("top", []),
        "strategy_bottom": report.get("strategies", {}).get("bottom", []),
        "orders": report.get("orders"),
        "metrics": report.get("metrics"),
        "flags": report.get("flags"),
    }
    system_prompt = (
        "You are an FX ops analyst. Summarize performance, highlight risks, and "
        "propose 3-5 actions. Do not suggest direct trade entries. "
        "Return JSON with keys: summary, risks(list), actions(list of {type,target,reason,confidence})."
    )
    prompt = (
        system_prompt
        + "\n\nInput JSON:\n"
        + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    )
    response = call_vertex_text(
        prompt,
        project_id=_VERTEX_PROJECT,
        location=_VERTEX_LOCATION,
        model=_VERTEX_SUMMARIZER_MODEL,
        temperature=0.2,
        max_tokens=400,
        timeout_sec=_VERTEX_TIMEOUT_SEC,
    )
    if not response or not response.text:
        return None
    text = response.text.strip()
    parsed = _parse_json_text(text)
    if parsed is not None:
        return parsed
    return {"summary": text}


def _gpt_summary(report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    provider = _resolve_provider(_SUMMARY_PROVIDER)
    if provider == "vertex":
        return _vertex_summary(report)
    if provider == "openai":
        return _openai_summary(report)
    return None


def _openai_policy_diff(report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if OpenAI is None:
        return None
    try:
        api_key = get_secret("openai_api_key")
    except Exception:
        api_key = None
    if not api_key:
        return None
    client = OpenAI(api_key=api_key)
    model = _resolve_model()
    system_prompt = _build_policy_prompt(report)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Return ONLY the JSON object."},
        ],
        max_output_tokens=_POLICY_MAX_TOKENS,
        temperature=0.2,
    )
    try:
        text = response.output_text.strip()
    except Exception:
        text = ""
    return _parse_policy_diff(text, source="ops_openai") if text else None


def _vertex_policy_diff(report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    prompt = _build_policy_prompt(report)
    response = call_vertex_text(
        prompt,
        project_id=_VERTEX_PROJECT,
        location=_VERTEX_LOCATION,
        model=_VERTEX_POLICY_MODEL,
        temperature=0.2,
        max_tokens=_POLICY_MAX_TOKENS,
        timeout_sec=_VERTEX_POLICY_TIMEOUT_SEC,
        response_mime_type="application/json",
    )
    if not response or not response.text:
        return None
    return _parse_policy_diff(response.text, source="ops_vertex")


def _ops_policy_diff(report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    provider = _resolve_provider(_POLICY_PROVIDER)
    if provider == "vertex":
        return _vertex_policy_diff(report)
    if provider == "openai":
        return _openai_policy_diff(report)
    return None


def run(args: argparse.Namespace) -> Dict[str, Any]:
    now = _utc_now()
    since = now - dt.timedelta(hours=args.hours)
    since_ts = _iso(since)
    report: Dict[str, Any] = {
        "generated_at": _iso(now),
        "window": {"hours": args.hours, "start": since_ts, "end": _iso(now)},
    }

    with _connect(args.trades_db) as con:
        report["overall"] = _trade_aggregate(con, since_ts)
        report["pockets"] = _trade_by_pocket(con, since_ts)
        report["strategies"] = _trade_by_strategy(
            con,
            since_ts,
            limit=args.strategy_limit,
            min_trades=args.strategy_min_trades,
        )

    with _connect(args.orders_db) as con:
        report["orders"] = _order_stats(con, since_ts)

    with _connect(args.metrics_db) as con:
        report["metrics"] = {
            "decision_latency_ms": _metric_stats(con, since_ts, "decision_latency_ms"),
            "data_lag_ms": _metric_stats(con, since_ts, "data_lag_ms"),
        }

    report["flags"] = _build_flags(report)

    gpt_used = False
    if args.gpt:
        summary = _gpt_summary(report)
        if summary:
            report["gpt_summary"] = summary
            gpt_used = True
    report["gpt_used"] = gpt_used
    if args.policy:
        diff = _ops_policy_diff(report)
        if diff:
            report["policy_diff"] = diff
            report["policy_used"] = True
            if args.apply_policy:
                updated, changed, flags = apply_policy_diff_to_paths(
                    diff,
                    overlay_path=Path(args.overlay_path),
                    history_dir=Path(args.history_dir),
                    latest_path=Path(args.latest_path),
                )
                report["policy_applied"] = changed
                report["policy_apply_flags"] = flags
                report["policy_overlay"] = updated
        else:
            report["policy_used"] = False
    return report


def _write_output(report: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=True, indent=2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ops report for GPT summarisation")
    parser.add_argument("--trades-db", default="logs/trades.db")
    parser.add_argument("--orders-db", default="logs/orders.db")
    parser.add_argument("--metrics-db", default="logs/metrics.db")
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--strategy-limit", type=int, default=8)
    parser.add_argument("--strategy-min-trades", type=int, default=3)
    parser.add_argument("--output", default="logs/gpt_ops_report.json")
    parser.add_argument("--gpt", action="store_true", help="Enable GPT summary")
    parser.add_argument("--policy", action="store_true", help="Generate policy diff via LLM")
    parser.add_argument("--policy-output", default="logs/policy_diff_ops.json")
    parser.add_argument("--apply-policy", action="store_true", help="Apply policy diff to overlay")
    parser.add_argument("--overlay-path", default=os.getenv("POLICY_OVERLAY_PATH", "logs/policy_overlay.json"))
    parser.add_argument("--history-dir", default=os.getenv("POLICY_HISTORY_DIR", "logs/policy_history"))
    parser.add_argument("--latest-path", default=os.getenv("POLICY_LATEST_PATH", "logs/policy_latest.json"))
    parser.add_argument("--stdout", action="store_true", help="Print report to stdout only")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    report = run(args)
    if args.stdout:
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return
    if report.get("policy_used") and report.get("policy_diff") and args.policy_output:
        _write_output(report.get("policy_diff"), args.policy_output)
    _write_output(report, args.output)
    print(f"[OK] report saved: {args.output}")


if __name__ == "__main__":
    main()
