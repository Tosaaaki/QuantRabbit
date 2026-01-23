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
from collections import deque
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
_DEFAULT_POLICY_HOURS = float(os.getenv("LLM_OPS_POLICY_HOURS") or os.getenv("LLM_OPS_REPORT_HOURS") or 24.0)

def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except Exception:
        return default


_REPLAY_DIR = Path(os.getenv("OPS_REPLAY_DIR", "logs/replay/USD_JPY"))
_TREND_STRENGTH_MIN = _env_float("POLICY_TREND_STRENGTH_MIN", 1.0)
_TREND_STRENGTH_MIN_H1 = _env_float("POLICY_TREND_STRENGTH_MIN_H1", _TREND_STRENGTH_MIN)
_SPREAD_P95_THROTTLE = _env_float(
    "POLICY_SPREAD_P95_PIPS",
    _env_float("M1SCALP_MAX_SPREAD_PIPS", 1.4),
)
_RANGE_MICRO_STRATEGIES = ["BB_RSI", "BB_RSI_Fast"]
_TREND_MACRO_STRATEGIES = ["TrendMA", "H1Momentum", "Donchian55"]
_TREND_MICRO_STRATEGIES = ["MomentumBurst", "MicroPullbackEMA", "TrendMomentumMicro", "MicroMomentumStack"]


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


def _latest_metric(con: sqlite3.Connection, metric: str) -> Optional[Dict[str, Any]]:
    row = con.execute(
        "SELECT ts, value, tags FROM metrics WHERE metric = ? ORDER BY ts DESC LIMIT 1",
        (metric,),
    ).fetchone()
    if not row:
        return None
    tags_raw = row["tags"] if isinstance(row, sqlite3.Row) else row[2]
    tags: Optional[Dict[str, Any]] = None
    if tags_raw:
        try:
            tags = json.loads(tags_raw)
        except Exception:
            tags = None
    try:
        value = float(row["value"])
    except Exception:
        value = None
    return {
        "ts": row["ts"],
        "value": value,
        "tags": tags,
    }


def _build_flags(report: Dict[str, Any], *, include_perf: bool = True) -> List[str]:
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
    if include_perf:
        if overall.get("trades", 0) >= 10 and overall.get("win_rate", 1.0) < 0.48:
            flags.append("overall_win_rate_low")
        if overall.get("trades", 0) >= 10 and overall.get("pf") and overall["pf"] < 0.9:
            flags.append("overall_pf_low")
    return flags


def _strip_perf(report: Dict[str, Any]) -> Dict[str, Any]:
    # Keep non-performance telemetry for policy input.
    clone = json.loads(json.dumps(report))
    overall = clone.get("overall")
    if isinstance(overall, dict):
        trades = overall.get("trades")
        clone["overall"] = {"trades": trades} if trades is not None else {}
    pockets = clone.get("pockets")
    if isinstance(pockets, dict):
        trimmed: Dict[str, Any] = {}
        for key, value in pockets.items():
            if isinstance(value, dict):
                trades = value.get("trades")
                trimmed[key] = {"trades": trades} if trades is not None else {}
        clone["pockets"] = trimmed
    elif isinstance(pockets, list):
        trimmed_list = []
        for value in pockets:
            if isinstance(value, dict):
                trades = value.get("trades")
                trimmed_list.append({"trades": trades} if trades is not None else {})
        clone["pockets"] = trimmed_list
    clone["strategies"] = {"top": [], "bottom": []}
    return clone


def _load_recent_candles(pattern: str, maxlen: int) -> List[Dict[str, Any]]:
    if not _REPLAY_DIR.exists():
        return []
    files = sorted(_REPLAY_DIR.glob(pattern))
    if not files:
        return []
    data: deque = deque(maxlen=maxlen)
    for path in files[-2:]:
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            continue
    return list(data)


def _trend_from_candles(candles: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not candles or len(candles) < 5:
        return None
    candles = sorted(candles, key=lambda r: r.get("ts") or "")
    closes = [c.get("close") for c in candles if isinstance(c.get("close"), (int, float))]
    highs = [c.get("high") for c in candles if isinstance(c.get("high"), (int, float))]
    lows = [c.get("low") for c in candles if isinstance(c.get("low"), (int, float))]
    if not closes or not highs or not lows:
        return None
    start = closes[0]
    end = closes[-1]
    net = end - start
    ranges = [h - l for h, l in zip(highs, lows)]
    avg_range = sum(ranges) / len(ranges) if ranges else 0.0
    strength = abs(net) / avg_range if avg_range else 0.0
    if strength < 1.0:
        direction = "flat"
    else:
        direction = "up" if net > 0 else "down"
    return {
        "direction": direction,
        "strength": round(strength, 3),
        "net": round(net, 4),
        "avg_range": round(avg_range, 4),
        "last_ts": candles[-1].get("ts"),
        "count": len(candles),
    }


def _trend_snapshot() -> Dict[str, Any]:
    return {
        "m1": _trend_from_candles(_load_recent_candles("USD_JPY_M1_*.jsonl", 120)),
        "h1": _trend_from_candles(_load_recent_candles("USD_JPY_H1_*.jsonl", 24)),
        "h4": _trend_from_candles(_load_recent_candles("USD_JPY_H4_*.jsonl", 30)),
    }


def _load_policy_snapshot(path: Path) -> Dict[str, Any]:
    try:
        raw = path.read_text()
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _auto_policy_state(notes: Dict[str, Any], key: str) -> Optional[str]:
    if not isinstance(notes, dict):
        return None
    auto = notes.get("auto_policy")
    if not isinstance(auto, dict):
        return None
    entry = auto.get(key)
    if isinstance(entry, dict):
        return str(entry.get("state") or "")
    if isinstance(entry, str):
        return entry
    return None


def _policy_rules_diff(report: Dict[str, Any], latest_path: Path) -> Optional[Dict[str, Any]]:
    if not _env_bool("POLICY_RULES_ENABLED", True):
        return None
    current = _load_policy_snapshot(latest_path)
    current_notes = current.get("notes") if isinstance(current.get("notes"), dict) else {}
    patch: Dict[str, Any] = {}
    notes: Dict[str, Any] = {}
    auto_notes: Dict[str, Any] = {}
    actions: List[str] = []

    range_state = report.get("range_state") or {}
    range_active = bool(range_state.get("active")) if isinstance(range_state, dict) else False
    range_reason = None
    range_score = None
    if isinstance(range_state, dict):
        tags = range_state.get("tags") or {}
        if isinstance(tags, dict):
            range_reason = tags.get("reason")
            range_score = tags.get("score")

    def _set_entry_gate(pocket: str, allow_new: bool) -> None:
        patch.setdefault("pockets", {}).setdefault(pocket, {}).setdefault("entry_gates", {})[
            "allow_new"
        ] = allow_new

    def _set_strategies(pocket: str, strategies: List[str]) -> None:
        patch.setdefault("pockets", {}).setdefault(pocket, {})["strategies"] = strategies

    def _set_bias(pocket: str, bias: str) -> None:
        patch.setdefault("pockets", {}).setdefault(pocket, {})["bias"] = bias

    # Range regime guard (strategy restriction)
    if range_active:
        patch["range_mode"] = True
        _set_entry_gate("macro", False)
        _set_strategies("micro", list(_RANGE_MICRO_STRATEGIES))
        _set_bias("micro", "neutral")
        actions.append("range_guard_on")
        auto_notes["range_guard"] = {
            "state": "active",
            "reason": range_reason,
            "score": range_score,
        }
    else:
        if _auto_policy_state(current_notes, "range_guard") == "active":
            patch["range_mode"] = False
            _set_entry_gate("macro", True)
            _set_strategies("micro", [])
            actions.append("range_guard_off")
            auto_notes["range_guard"] = {"state": "cleared"}

    # Trend bias (directional throttle) when not in range mode
    if not range_active:
        trend = report.get("trend") or {}
        macro_trend = trend.get("h4") if isinstance(trend, dict) else None
        micro_trend = trend.get("h1") if isinstance(trend, dict) else None
        if (
            isinstance(macro_trend, dict)
            and macro_trend.get("direction") in {"up", "down"}
            and (macro_trend.get("strength") or 0) >= _TREND_STRENGTH_MIN
        ):
            bias = "long" if macro_trend["direction"] == "up" else "short"
            _set_bias("macro", bias)
            _set_strategies("macro", list(_TREND_MACRO_STRATEGIES))
            actions.append(f"macro_trend_{bias}")
            auto_notes["trend_guard"] = {
                "state": "active",
                "macro_dir": macro_trend["direction"],
                "macro_strength": macro_trend.get("strength"),
            }
        elif _auto_policy_state(current_notes, "trend_guard") == "active":
            _set_bias("macro", "neutral")
            _set_strategies("macro", [])
            actions.append("macro_trend_clear")
            auto_notes["trend_guard"] = {"state": "cleared"}
        if (
            isinstance(micro_trend, dict)
            and micro_trend.get("direction") in {"up", "down"}
            and (micro_trend.get("strength") or 0) >= _TREND_STRENGTH_MIN_H1
        ):
            bias = "long" if micro_trend["direction"] == "up" else "short"
            _set_bias("micro", bias)
            _set_strategies("micro", list(_TREND_MICRO_STRATEGIES))
            actions.append(f"micro_trend_{bias}")
            auto_notes["trend_guard_micro"] = {
                "state": "active",
                "micro_dir": micro_trend["direction"],
                "micro_strength": micro_trend.get("strength"),
            }
        elif _auto_policy_state(current_notes, "trend_guard_micro") == "active":
            _set_bias("micro", "neutral")
            _set_strategies("micro", [])
            actions.append("micro_trend_clear")
            auto_notes["trend_guard_micro"] = {"state": "cleared"}

    # Spread soft throttle for scalp
    spread_stats = report.get("metrics", {}).get("decision_spread_pips", {})
    spread_p95 = spread_stats.get("p95") if isinstance(spread_stats, dict) else None
    if isinstance(spread_p95, (int, float)) and spread_p95 >= _SPREAD_P95_THROTTLE:
        _set_entry_gate("scalp", False)
        actions.append("spread_throttle_on")
        auto_notes["spread_guard"] = {
            "state": "active",
            "p95": spread_p95,
            "limit": _SPREAD_P95_THROTTLE,
        }
    else:
        if _auto_policy_state(current_notes, "spread_guard") == "active":
            _set_entry_gate("scalp", True)
            actions.append("spread_throttle_off")
            auto_notes["spread_guard"] = {"state": "cleared"}

    if not patch:
        return None
    if auto_notes:
        notes["auto_policy"] = auto_notes
    return {
        "policy_id": f"ops_rules_{int(_utc_now().timestamp())}",
        "generated_at": _iso(_utc_now()),
        "source": "ops_rules",
        "no_change": False,
        "reason": ",".join(actions) if actions else "rules_apply",
        "patch": patch,
        "notes": notes,
    }


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


def _normalize_policy_patch(payload: Dict[str, Any]) -> None:
    patch = payload.get("patch")
    if not isinstance(patch, dict):
        return
    for key in ("tuning_overrides", "reentry_overrides", "metrics_window", "slo_guard"):
        if key in patch and key not in payload:
            payload[key] = patch.pop(key)
    pockets = patch.get("pockets")
    if not isinstance(pockets, dict):
        return
    for _, pocket_cfg in pockets.items():
        if not isinstance(pocket_cfg, dict):
            continue
        strategies = pocket_cfg.get("strategies")
        if isinstance(strategies, dict):
            allowlist = strategies.get("allowlist") or strategies.get("allow")
            if isinstance(allowlist, str):
                pocket_cfg["strategies"] = [allowlist]
            elif isinstance(allowlist, (list, tuple, set)):
                pocket_cfg["strategies"] = [str(item) for item in allowlist if item]
        if isinstance(pocket_cfg.get("strategies"), list):
            entry_gates = pocket_cfg.get("entry_gates")
            if not isinstance(entry_gates, dict):
                entry_gates = {}
                pocket_cfg["entry_gates"] = entry_gates
            if "allow_new" not in entry_gates:
                entry_gates["allow_new"] = True


def _sanitize_tuning_overrides(payload: Dict[str, Any]) -> None:
    overrides = payload.get("tuning_overrides")
    if overrides is None:
        payload.pop("tuning_overrides", None)
        return
    if not isinstance(overrides, dict):
        payload.pop("tuning_overrides", None)
        return
    sanitized: Dict[str, Any] = {}

    exit_cfg = overrides.get("exit")
    if isinstance(exit_cfg, dict):
        lowvol = exit_cfg.get("lowvol")
        if isinstance(lowvol, dict):
            allowed_lowvol = {
                "upper_bound_max_sec",
                "hazard_cost_spread_base",
                "hazard_cost_latency_base_ms",
                "hazard_debounce_ticks",
                "min_grace_before_scratch_ms",
                "scratch_requires_events",
            }
            kept = {k: lowvol[k] for k in allowed_lowvol if k in lowvol}
            if kept:
                sanitized.setdefault("exit", {})["lowvol"] = kept

    strategies_cfg = overrides.get("strategies")
    if isinstance(strategies_cfg, dict):
        allowed_strategies = {
            "MomentumPulse": {"min_confidence"},
            "MicroVWAPRevert": {"vwap_z_min"},
            "VolCompressionBreak": {"accel_pctile"},
            "BB_RSI_Fast": {"reentry_block_s"},
        }
        kept_strategies: Dict[str, Any] = {}
        for strat, allowed_keys in allowed_strategies.items():
            cfg = strategies_cfg.get(strat)
            if not isinstance(cfg, dict):
                continue
            kept = {k: cfg[k] for k in allowed_keys if k in cfg}
            if kept:
                kept_strategies[strat] = kept
        if kept_strategies:
            sanitized["strategies"] = kept_strategies

    if sanitized:
        payload["tuning_overrides"] = sanitized
    else:
        payload.pop("tuning_overrides", None)


def _sanitize_optional_object(payload: Dict[str, Any], key: str) -> None:
    value = payload.get(key)
    if value is None:
        payload.pop(key, None)
        return
    if not isinstance(value, dict):
        payload.pop(key, None)


def _parse_policy_diff(text: str, *, source: str) -> Optional[Dict[str, Any]]:
    payload = _extract_json_payload(text)
    if not isinstance(payload, dict):
        preview = " ".join((text or "").strip().split())[:200]
        if preview:
            logging.warning("[OPS_POLICY] no JSON payload returned from LLM. preview=%s", preview)
        else:
            logging.warning("[OPS_POLICY] no JSON payload returned from LLM.")
        return None
    payload["source"] = source
    _normalize_policy_patch(payload)
    _sanitize_tuning_overrides(payload)
    _sanitize_optional_object(payload, "reentry_overrides")
    _sanitize_optional_object(payload, "metrics_window")
    _sanitize_optional_object(payload, "slo_guard")
    payload = normalize_policy_diff(payload, source=source)
    errors = validate_policy_diff(payload)
    if errors:
        logging.warning("[OPS_POLICY] invalid policy_diff: %s", ", ".join(errors))
        return None
    return payload


def _build_policy_prompt(report: Dict[str, Any]) -> str:
    schema_text = json.dumps(POLICY_DIFF_SCHEMA, ensure_ascii=True, separators=(",", ":"))
    patch_keys = ["air_score", "uncertainty", "event_lock", "range_mode", "notes", "pockets"]
    pocket_keys = [
        "enabled",
        "bias",
        "confidence",
        "units_cap",
        "entry_gates",
        "exit_profile",
        "be_profile",
        "partial_profile",
        "strategies",
        "pending_orders",
    ]
    payload = {
        "window": report.get("window"),
        "overall": report.get("overall"),
        "pockets": report.get("pockets"),
        "strategy_top": report.get("strategies", {}).get("top", []),
        "strategy_bottom": report.get("strategies", {}).get("bottom", []),
        "orders": report.get("orders"),
        "metrics": report.get("metrics"),
        "range_state": report.get("range_state"),
        "trend": report.get("trend"),
        "flags": report.get("flags"),
        "perf_excluded": report.get("perf_excluded"),
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
        "- tuning_overrides and reentry_overrides must be top-level (not inside patch).\n"
        f"- Allowed patch keys: {patch_keys}.\n"
        f"- Allowed pocket keys: {pocket_keys}.\n"
        "- strategies must be a list of strings (no allowlist object).\n"
        "- tuning_overrides keys:\n"
        "  - exit.lowvol.{upper_bound_max_sec,hazard_cost_spread_base,hazard_cost_latency_base_ms,"
        "hazard_debounce_ticks,min_grace_before_scratch_ms,scratch_requires_events}\n"
        "  - strategies.MomentumPulse.min_confidence\n"
        "  - strategies.MicroVWAPRevert.vwap_z_min\n"
        "  - strategies.VolCompressionBreak.accel_pctile\n"
        "  - strategies.BB_RSI_Fast.reentry_block_s\n\n"
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
    )
    if not response or not response.text:
        return None
    return _parse_policy_diff(response.text, source="ops_vertex")


def _ops_policy_diff(report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    provider = _resolve_provider(_POLICY_PROVIDER)
    if provider == "vertex":
        diff = _vertex_policy_diff(report)
        if diff is None:
            logging.warning("[OPS_POLICY] vertex policy generation returned empty response.")
        return diff
    if provider == "openai":
        diff = _openai_policy_diff(report)
        if diff is None:
            logging.warning("[OPS_POLICY] openai policy generation returned empty response.")
        return diff
    logging.warning("[OPS_POLICY] unknown provider=%s", provider)
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
            "decision_spread_pips": _metric_stats(con, since_ts, "decision_spread_pips"),
        }
        range_latest = _latest_metric(con, "range_mode_active")
        if range_latest:
            report["range_state"] = {
                "active": bool(range_latest.get("value", 0.0)),
                "ts": range_latest.get("ts"),
                "tags": range_latest.get("tags"),
            }

    report["trend"] = _trend_snapshot()

    report["flags"] = _build_flags(report)

    gpt_used = False
    if args.gpt:
        summary = _gpt_summary(report)
        if summary:
            report["gpt_summary"] = summary
            gpt_used = True
    report["gpt_used"] = gpt_used
    if args.policy:
        diff = _policy_rules_diff(report, Path(args.latest_path))
        if diff:
            report["policy_rules_used"] = True
        else:
            policy_report = report
            if _env_bool("LLM_OPS_POLICY_EXCLUDE_PERF"):
                policy_report = _strip_perf(report)
                policy_report["perf_excluded"] = True
                policy_report["flags"] = _build_flags(policy_report, include_perf=False)
                report["policy_perf_excluded"] = True
            diff = _ops_policy_diff(policy_report)
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
    parser.add_argument("--hours", type=float, default=_DEFAULT_POLICY_HOURS)
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
    args = parser.parse_args()
    if _env_bool("LLM_OPS_POLICY_APPLY") and not args.apply_policy:
        args.apply_policy = True
    return args


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
