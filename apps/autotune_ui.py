from __future__ import annotations

import json
import os
import socket
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Form, HTTPException, Header, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from autotune.database import (
    dump_dict,
    get_run,
    set_settings,
    update_status,
)
from utils.secrets import get_secret

try:  # pragma: no cover - optional dependency
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover
    storage = None

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = REPO_ROOT / "templates" / "autotune"
CONFIG_PATH = REPO_ROOT / "configs" / "scalp_active_params.json"
METRICS_DB = Path("logs/metrics.db")
ORDERS_DB = Path("logs/orders.db")
SIGNALS_DB = Path("logs/signals.db")

_LIVE_SNAPSHOT_TTL_SEC = int(os.getenv("LIVE_SNAPSHOT_TTL_SEC", "8"))
_REMOTE_SNAPSHOT_TIMEOUT_SEC = float(os.getenv("UI_SNAPSHOT_TIMEOUT_SEC", "4.0"))
_live_snapshot_lock = threading.Lock()
_live_snapshot_cache: dict[str, Any] | None = None
_live_snapshot_ts: float = 0.0

app = FastAPI(title="QuantRabbit Console")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


def _build_summary(run: dict) -> str:
    lines: list[str] = []
    profile = None
    timeframe = None
    train_meta = run.get("train") or {}
    if isinstance(train_meta, dict):
        profile = train_meta.get("profile")
        timeframe = train_meta.get("timeframe") or train_meta.get("tf")
    if not profile:
        valid_meta = run.get("valid") or {}
        if isinstance(valid_meta, dict):
            profile = valid_meta.get("profile")
            timeframe = timeframe or valid_meta.get("timeframe")
    if not profile:
        profile = run.get("profile")
    if profile:
        tf_hint = f" / TF {timeframe}" if timeframe else ""
        lines.append(f"プロファイル: {profile}{tf_hint}")

    score = run.get("score")
    if score is not None:
        lines.append(f"スコア: {score:.3f}")

    def outline(metrics: dict, label: str) -> None:
        if not metrics:
            return
        profit_pips = metrics.get("profit_pips")
        profit_jpy = metrics.get("profit_jpy")
        pf = metrics.get("profit_factor")
        trades = metrics.get("trades")
        dd = metrics.get("max_dd_pips")
        wr = metrics.get("win_rate")
        parts: list[str] = []
        if profit_pips is not None:
            parts.append(f"PL {profit_pips:+.1f} pips")
        if profit_jpy is not None:
            parts.append(f"({profit_jpy:+.0f} 円)" )
        if pf is not None:
            parts.append(f"PF {pf:.2f}")
        if wr is not None:
            parts.append(f"Win {wr*100:.1f}%")
        if trades is not None:
            parts.append(f"Trades {trades}")
        if dd is not None:
            parts.append(f"MaxDD {dd:.1f} pips")
        if parts:
            lines.append(f"{label}: " + ", ".join(parts))

    outline(run.get("train") or {}, "学習期間")
    outline(run.get("valid") or {}, "検証期間")

    params = run.get("params") or {}
    if params:
        sl = params.get("sl_pips")
        tp = params.get("tp_pips")
        timeout = params.get("timeout_sec")
        extras = []
        if sl is not None and tp is not None:
            extras.append(f"SL {sl} pips / TP {tp} pips")
        if timeout is not None:
            extras.append(f"タイムアウト {timeout//60} 分")
        if extras:
            lines.append("提案パラメータ: " + "、".join(extras))

    if not lines:
        lines.append("メトリクス情報が不足しています。")
    return "\n".join(lines)


def _dashboard_defaults(error: Optional[str] = None) -> Dict[str, Any]:
    return {
        "available": False,
        "error": error,
        "generated_at": None,
        "generated_label": None,
        "recent_trades": [],
        "performance": {
            "daily_pl_pips": 0.0,
            "daily_pl_jpy": 0.0,
            "daily_pl_eq1l": 0.0,
            "weekly_pl_pips": 0.0,
            "weekly_pl_jpy": 0.0,
            "weekly_pl_eq1l": 0.0,
            "total_pips": 0.0,
            "total_jpy": 0.0,
            "total_eq1l": 0.0,
            "total_trades": 0,
            "recent_closed": 0,
            "win_rate": 0.0,
            "win_rate_percent": 0.0,
            "wins": 0,
            "losses": 0,
            "open_positions": 0,
            "net_units": 0.0,
            "new_trades": 0,
            "last_trade_at": None,
            "unrealized_pl_pips": 0.0,
            "unrealized_pl_jpy": 0.0,
            "pockets": [],
        },
        "open_summary": {
            "total_positions": 0,
            "net_units": 0.0,
            "unrealized_pl_pips": 0.0,
            "unrealized_pl_jpy": 0.0,
            "pockets": [],
        },
        "system": {
            "data_lag_ms": None,
            "data_lag_level": "level-neutral",
            "decision_latency_ms": None,
            "decision_latency_level": "level-neutral",
            "healthbeat_label": None,
            "healthbeat_age_min": None,
            "healthbeat_level": "level-neutral",
            "signals_last_label": None,
            "signals_last_age_min": None,
            "signals_last_level": "level-neutral",
            "orders_total_1h": None,
            "orders_status_1h": [],
            "orders_last": [],
            "signals_recent": [],
        },
        "highlights": [],  # backward compatibility (top winners / losers)
        "highlights_top": [],
        "highlights_recent": [],
    }


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return 0.0


def _opt_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    formats = (
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    )
    for fmt in formats:
        try:
            dt = datetime.strptime(text, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(text)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:  # pragma: no cover - defensive
        return None


def _format_dt(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _parse_ts_ms(ts_ms: Any) -> Optional[datetime]:
    try:
        if ts_ms is None:
            return None
        return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc)
    except Exception:  # pragma: no cover - defensive
        return None


def _age_minutes(dt: Optional[datetime], now: datetime) -> Optional[int]:
    if not dt:
        return None
    delta = now - dt
    return max(0, int(delta.total_seconds() // 60))


def _level_for_value(value: Optional[float], warn: float, bad: float) -> str:
    if value is None:
        return "level-neutral"
    if value <= warn:
        return "level-ok"
    if value <= bad:
        return "level-warn"
    return "level-bad"


def _level_for_age(age_min: Optional[int], warn: int, bad: int) -> str:
    if age_min is None:
        return "level-neutral"
    if age_min <= warn:
        return "level-ok"
    if age_min <= bad:
        return "level-warn"
    return "level-bad"


def _status_level(status: str) -> str:
    name = (status or "").lower()
    if any(word in name for word in ("reject", "cancel", "error", "fail")):
        return "level-bad"
    if any(word in name for word in ("filled", "accepted", "ok", "success")):
        return "level-ok"
    return "level-warn"


def _shorten(text: Any, limit: int = 20) -> str:
    if text is None:
        return "-"
    value = str(text)
    if len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return value[: limit - 3] + "..."


def _get_secret_optional(key: str) -> Optional[str]:
    try:
        value = get_secret(key)
    except KeyError:
        return None
    value = str(value).strip()
    return value or None


def _load_latest_metric(metric: str) -> Optional[float]:
    if not METRICS_DB.exists():
        return None
    try:
        con = sqlite3.connect(METRICS_DB)
        cur = con.execute(
            "SELECT value FROM metrics WHERE metric = ? ORDER BY ts DESC LIMIT 1",
            (metric,),
        )
        row = cur.fetchone()
        con.close()
        if not row:
            return None
        return float(row[0])
    except Exception:
        return None


def _load_last_metric_ts(metric: str) -> Optional[str]:
    if not METRICS_DB.exists():
        return None
    try:
        con = sqlite3.connect(METRICS_DB)
        cur = con.execute(
            "SELECT ts FROM metrics WHERE metric = ? ORDER BY ts DESC LIMIT 1",
            (metric,),
        )
        row = cur.fetchone()
        con.close()
        if not row:
            return None
        return str(row[0]) if row[0] is not None else None
    except Exception:
        return None


def _load_last_orders(limit: int = 5) -> list[dict]:
    if not ORDERS_DB.exists():
        return []
    try:
        con = sqlite3.connect(ORDERS_DB)
        con.row_factory = sqlite3.Row
        cur = con.execute(
            "SELECT ts, pocket, side, units, status, client_order_id "
            "FROM orders ORDER BY ts DESC LIMIT ?",
            (int(limit),),
        )
        rows = [dict(r) for r in cur.fetchall()]
        con.close()
        return rows
    except Exception:
        return []


def _load_order_status_counts(limit: int = 8, hours: int = 1) -> list[dict]:
    if not ORDERS_DB.exists():
        return []
    try:
        con = sqlite3.connect(ORDERS_DB)
        con.row_factory = sqlite3.Row
        cur = con.execute(
            "SELECT status, count(*) AS count FROM orders "
            "WHERE ts >= datetime('now', ?) "
            "GROUP BY status ORDER BY count DESC LIMIT ?",
            (f"-{int(hours)} hour", int(limit)),
        )
        rows = [dict(r) for r in cur.fetchall()]
        con.close()
        return rows
    except Exception:
        return []


def _load_last_signal_ts_ms() -> Optional[int]:
    if not SIGNALS_DB.exists():
        return None
    try:
        con = sqlite3.connect(SIGNALS_DB)
        cur = con.execute("SELECT max(ts_ms) FROM signals")
        row = cur.fetchone()
        con.close()
        if not row:
            return None
        val = row[0]
        return int(val) if val is not None else None
    except Exception:
        return None


def _load_recent_signals(limit: int = 5) -> list[dict]:
    if not SIGNALS_DB.exists():
        return []
    try:
        con = sqlite3.connect(SIGNALS_DB)
        cur = con.execute(
            "SELECT ts_ms, payload FROM signals ORDER BY ts_ms DESC LIMIT ?",
            (int(limit),),
        )
        rows: list[dict] = []
        for ts_ms, payload in cur.fetchall():
            item: dict = {"ts_ms": ts_ms}
            try:
                data = json.loads(payload)
            except Exception:
                data = None
            if isinstance(data, dict):
                for key in (
                    "pocket",
                    "strategy",
                    "confidence",
                    "action",
                    "client_order_id",
                    "proposed_units",
                ):
                    if key in data:
                        item[key] = data[key]
            rows.append(item)
        con.close()
        return rows
    except Exception:
        return []


def _build_live_snapshot() -> dict:
    try:
        from execution.position_manager import PositionManager
    except Exception:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "new_trades": [],
            "recent_trades": [],
            "open_positions": {},
            "metrics": {},
        }

    pm = PositionManager()
    new_trades: list[dict] = []
    if os.getenv("UI_SNAPSHOT_SYNC_TRADES", "").lower() in {"1", "true", "yes"}:
        try:
            new_trades = list(pm.sync_trades())
        except Exception:
            new_trades = []

    try:
        open_positions = pm.get_open_positions()
    except Exception:
        open_positions = {}

    try:
        recent_trades = pm.fetch_recent_trades(limit=50)
    except Exception:
        recent_trades = []

    try:
        metrics = pm.get_performance_summary()
    except Exception:
        metrics = {}

    if isinstance(metrics, dict):
        data_lag_ms = _load_latest_metric("data_lag_ms")
        decision_latency_ms = _load_latest_metric("decision_latency_ms")
        if data_lag_ms is not None:
            metrics["data_lag_ms"] = data_lag_ms
        if decision_latency_ms is not None:
            metrics["decision_latency_ms"] = decision_latency_ms
        metrics["orders_last"] = _load_last_orders()
        metrics["orders_status_1h"] = _load_order_status_counts()
        last_signal_ts = _load_last_signal_ts_ms()
        if last_signal_ts is not None:
            metrics["signals_last_ts_ms"] = last_signal_ts
        metrics["signals_recent"] = _load_recent_signals()
        metrics["healthbeat_ts"] = _load_last_metric_ts("healthbeat")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "new_trades": list(new_trades),
        "recent_trades": list(recent_trades),
        "open_positions": dict(open_positions),
        "metrics": dict(metrics) if isinstance(metrics, dict) else {},
    }


def _cached_live_snapshot() -> dict:
    global _live_snapshot_cache, _live_snapshot_ts
    now_mono = time.monotonic()
    if _live_snapshot_cache and (now_mono - _live_snapshot_ts) < _LIVE_SNAPSHOT_TTL_SEC:
        return _live_snapshot_cache
    with _live_snapshot_lock:
        now_mono = time.monotonic()
        if _live_snapshot_cache and (now_mono - _live_snapshot_ts) < _LIVE_SNAPSHOT_TTL_SEC:
            return _live_snapshot_cache
        snapshot = _build_live_snapshot()
        _live_snapshot_cache = snapshot
        _live_snapshot_ts = now_mono
        return snapshot


def _fetch_remote_snapshot() -> Optional[dict]:
    url = _get_secret_optional("ui_snapshot_url")
    if not url:
        return None
    debug = os.getenv("UI_SNAPSHOT_DEBUG", "").lower() in {"1", "true", "yes"}
    token = _get_secret_optional("ui_snapshot_token")
    req = urllib.request.Request(url)
    if token:
        req.add_header("X-QR-Token", token)
        req.add_header("Authorization", f"Bearer {token}")
    try:
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=_REMOTE_SNAPSHOT_TIMEOUT_SEC) as resp:
            raw = resp.read()
        data = json.loads(raw)
        if debug:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            print(f"[ui_snapshot] remote_ok elapsed_ms={elapsed_ms}")
        return data
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, socket.timeout) as exc:
        if debug:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            print(
                "[ui_snapshot] remote_error elapsed_ms=%s error=%s detail=%s"
                % (elapsed_ms, type(exc).__name__, exc)
            )
        return None


def _summarise_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    base = _dashboard_defaults()
    now = datetime.now(timezone.utc)

    metrics_snapshot = snapshot.get("metrics") or {}
    trades_raw = snapshot.get("recent_trades") or []
    parsed_trades: list[Dict[str, Any]] = []
    for item in trades_raw:
        close_dt = _parse_dt(item.get("close_time") or item.get("updated_at"))
        parsed_trades.append(
            {
                "ticket_id": str(item.get("ticket_id") or ""),
                "pocket": (item.get("pocket") or "-").strip() or "-",
                "pl_pips": _safe_float(item.get("pl_pips")),
                "pl_jpy": _safe_float(item.get("realized_pl")),
                "close_time": close_dt,
                "close_label": _format_dt(close_dt),
            }
        )

    recent_trades_display: list[Dict[str, Any]] = []
    for item in trades_raw[:12]:
        entry_dt = _parse_dt(item.get("entry_time"))
        close_dt = _parse_dt(item.get("close_time"))
        updated_dt = _parse_dt(item.get("updated_at") or item.get("close_time") or item.get("entry_time"))
        units_val = _safe_float(item.get("units"))
        direction = "Long" if units_val > 0 else "Short" if units_val < 0 else "Flat"
        units_abs = int(round(abs(units_val)))
        pl_pips = _safe_float(item.get("pl_pips"))
        pl_jpy = _safe_float(item.get("realized_pl"))
        kind = "gain" if pl_pips > 0 else "loss" if pl_pips < 0 else "neutral"
        recent_trades_display.append(
            {
                "ticket_id": str(item.get("ticket_id") or ""),
                "pocket": (item.get("pocket") or "-").strip() or "-",
                "direction": direction,
                "units_abs": units_abs,
                "entry_price": _opt_float(item.get("entry_price")),
                "close_price": _opt_float(item.get("close_price")),
                "pl_pips": round(pl_pips, 2),
                "pl_jpy": round(pl_jpy, 2),
                "state": str(item.get("state") or "-").upper(),
                "close_reason": item.get("close_reason") or "-",
                "entry_label": _format_dt(entry_dt) or "-",
                "close_label": _format_dt(close_dt) or "-",
                "updated_label": _format_dt(updated_dt) or "-",
                "kind": kind,
            }
        )
    base["recent_trades"] = recent_trades_display

    closed_trades = [t for t in parsed_trades if t["close_time"]]
    closed_trades.sort(key=lambda t: t["close_time"], reverse=True)
    week_cutoff = now - timedelta(days=7)
    today_date = now.date()

    def _sum_if(predicate) -> float:
        return sum(t["pl_pips"] for t in closed_trades if predicate(t))

    def _sum_jpy(predicate) -> float:
        return sum(t["pl_jpy"] for t in closed_trades if predicate(t))

    perf = base["performance"]
    perf["new_trades"] = len(snapshot.get("new_trades") or [])

    def _apply_metrics(data: dict | None, *, target: str) -> None:
        if not data:
            return
        if target == "daily":
            perf["daily_pl_pips"] = data.get("pips", 0.0)
            perf["daily_pl_jpy"] = data.get("jpy", 0.0)
            perf["recent_closed"] = data.get("trades", 0)
        elif target == "weekly":
            perf["weekly_pl_pips"] = data.get("pips", 0.0)
            perf["weekly_pl_jpy"] = data.get("jpy", 0.0)
        elif target == "total":
            perf["total_pips"] = data.get("pips", 0.0)
            perf["total_jpy"] = data.get("jpy", 0.0)
            perf["wins"] = data.get("wins", 0)
            perf["losses"] = data.get("losses", 0)
            wr = data.get("win_rate", 0.0)
            perf["win_rate"] = wr
            perf["win_rate_percent"] = round(wr * 100.0, 1)
            perf["total_trades"] = data.get("trades", 0)

    _apply_metrics(metrics_snapshot.get("daily"), target="daily")
    _apply_metrics(metrics_snapshot.get("weekly"), target="weekly")
    _apply_metrics(metrics_snapshot.get("total"), target="total")

    if perf.get("recent_closed", 0) == 0:
        perf["recent_closed"] = len(closed_trades)
    if perf.get("total_trades") in (None, 0):
        perf["total_trades"] = len(parsed_trades)

    if perf.get("daily_pl_pips") is None:
        perf["daily_pl_pips"] = round(
            _sum_if(lambda t: t["close_time"].date() == today_date), 2
        ) if closed_trades else 0.0
    if perf.get("daily_pl_jpy") is None:
        perf["daily_pl_jpy"] = round(
            _sum_jpy(lambda t: t["close_time"].date() == today_date), 2
        ) if closed_trades else 0.0
    if perf.get("weekly_pl_pips") is None:
        perf["weekly_pl_pips"] = round(
            _sum_if(lambda t: t["close_time"] >= week_cutoff), 2
        ) if closed_trades else 0.0
    if perf.get("weekly_pl_jpy") is None:
        perf["weekly_pl_jpy"] = round(
            _sum_jpy(lambda t: t["close_time"] >= week_cutoff), 2
        ) if closed_trades else 0.0
    if perf.get("total_pips") is None:
        perf["total_pips"] = round(sum(t["pl_pips"] for t in closed_trades), 2)
    if perf.get("total_jpy") is None:
        perf["total_jpy"] = round(sum(t["pl_jpy"] for t in closed_trades), 2)

    perf["daily_pl_eq1l"] = round((perf.get("daily_pl_jpy", 0.0) or 0.0) / 1000.0, 2)
    perf["weekly_pl_eq1l"] = round((perf.get("weekly_pl_jpy", 0.0) or 0.0) / 1000.0, 2)
    perf["total_eq1l"] = round((perf.get("total_jpy", 0.0) or 0.0) / 1000.0, 2)

    if perf.get("wins") is None or perf.get("losses") is None:
        wins = sum(1 for t in closed_trades if t["pl_pips"] > 0)
        losses = sum(1 for t in closed_trades if t["pl_pips"] < 0)
        perf["wins"] = wins
        perf["losses"] = losses
        perf["win_rate"] = (wins / perf["recent_closed"]) if perf["recent_closed"] else 0.0
        perf["win_rate_percent"] = round(perf["win_rate"] * 100.0, 1)

    if perf.get("last_trade_at") is None:
        last_trade = metrics_snapshot.get("last_trade_at")
        if not last_trade and closed_trades:
            last_trade = max(closed_trades, key=lambda t: t["close_time"])["close_time"]
        perf["last_trade_at"] = _format_dt(_parse_dt(last_trade)) if last_trade else None

    open_positions = snapshot.get("open_positions") or {}
    open_entries: list[Dict[str, Any]] = []
    total_positions = 0
    total_unrealized_pips = 0.0
    total_unrealized_jpy = 0.0
    for name, info in open_positions.items():
        if name == "__net__":
            continue
        trades = info.get("open_trades") or []
        total_positions += len(trades)
        units_val = _safe_float(info.get("units"))
        direction = "Long" if units_val > 0 else "Short" if units_val < 0 else "Flat"
        unrealized_pips = round(_safe_float(info.get("unrealized_pl_pips")), 2)
        unrealized_jpy = round(_safe_float(info.get("unrealized_pl")), 2)
        total_unrealized_pips += unrealized_pips
        total_unrealized_jpy += unrealized_jpy
        units_abs = abs(units_val)
        open_entries.append(
            {
                "pocket": name,
                "direction": direction,
                "units": units_val,
                "units_abs": int(round(units_abs)),
                "avg_price": _safe_float(info.get("avg_price")),
                "unrealized_pips": unrealized_pips,
                "unrealized_jpy": unrealized_jpy,
                "trades": len(trades),
            }
        )
    open_entries.sort(key=lambda row: row["pocket"])

    net_units = _safe_float((open_positions.get("__net__") or {}).get("units"))
    base["open_summary"] = {
        "pockets": open_entries,
        "total_positions": total_positions,
        "net_units": net_units,
        "unrealized_pl_pips": round(total_unrealized_pips, 2),
        "unrealized_pl_jpy": round(total_unrealized_jpy, 2),
    }
    perf["open_positions"] = total_positions
    perf["net_units"] = net_units
    perf["unrealized_pl_pips"] = round(total_unrealized_pips, 2)
    perf["unrealized_pl_jpy"] = round(total_unrealized_jpy, 2)

    pocket_perf = []
    pocket_raw = metrics_snapshot.get("pockets") or {}
    order_hint = {"micro": 0, "macro": 1, "scalp": 2, "scalp_fast": 3}
    for name, raw in pocket_raw.items():
        pips = raw.get("pips")
        jpy = raw.get("jpy")
        trades = raw.get("trades")
        win_rate = raw.get("win_rate")
        pf = raw.get("pf")
        pocket_perf.append(
            {
                "pocket": name,
                "pips": round(_safe_float(pips), 2),
                "jpy": round(_safe_float(jpy), 2),
                "trades": int(trades) if trades is not None else 0,
                "win_rate": round(_safe_float(win_rate) * 100.0, 1) if win_rate is not None else None,
                "pf": round(_safe_float(pf), 2) if pf is not None else None,
            }
        )
    pocket_perf.sort(key=lambda row: order_hint.get(row["pocket"], 99))
    perf["pockets"] = pocket_perf

    system = base["system"]
    data_lag_ms = metrics_snapshot.get("data_lag_ms")
    if data_lag_ms is not None:
        data_lag_ms = float(data_lag_ms)
        system["data_lag_ms"] = round(data_lag_ms, 0)
        system["data_lag_level"] = _level_for_value(data_lag_ms, 1500.0, 3000.0)

    decision_latency_ms = metrics_snapshot.get("decision_latency_ms")
    if decision_latency_ms is not None:
        decision_latency_ms = float(decision_latency_ms)
        system["decision_latency_ms"] = round(decision_latency_ms, 0)
        system["decision_latency_level"] = _level_for_value(decision_latency_ms, 2000.0, 4000.0)

    healthbeat_dt = _parse_dt(metrics_snapshot.get("healthbeat_ts"))
    healthbeat_age = _age_minutes(healthbeat_dt, now)
    system["healthbeat_label"] = _format_dt(healthbeat_dt)
    system["healthbeat_age_min"] = healthbeat_age
    system["healthbeat_level"] = _level_for_age(healthbeat_age, 5, 10)

    signal_dt = _parse_ts_ms(metrics_snapshot.get("signals_last_ts_ms"))
    signal_age = _age_minutes(signal_dt, now)
    system["signals_last_label"] = _format_dt(signal_dt)
    system["signals_last_age_min"] = signal_age
    system["signals_last_level"] = _level_for_age(signal_age, 5, 15)

    orders_status = []
    for row in metrics_snapshot.get("orders_status_1h") or []:
        status = str(row.get("status") or "-")
        orders_status.append(
            {
                "status": status,
                "count": int(row.get("count") or 0),
                "level": _status_level(status),
            }
        )
    system["orders_status_1h"] = orders_status
    system["orders_total_1h"] = sum(row["count"] for row in orders_status)

    orders_last = []
    for row in metrics_snapshot.get("orders_last") or []:
        ts_label = _format_dt(_parse_dt(row.get("ts"))) or str(row.get("ts") or "-")
        order_id = row.get("client_order_id")
        status = str(row.get("status") or "-")
        orders_last.append(
            {
                "ts_label": ts_label,
                "pocket": row.get("pocket") or "-",
                "side": row.get("side") or "-",
                "units": row.get("units"),
                "status": status,
                "status_level": _status_level(status),
                "id_short": _shorten(order_id, 22),
                "id_full": str(order_id or ""),
            }
        )
    system["orders_last"] = orders_last

    signals_recent = []
    for row in metrics_snapshot.get("signals_recent") or []:
        ts_label = _format_dt(_parse_ts_ms(row.get("ts_ms"))) or str(row.get("ts_ms") or "-")
        signal_id = row.get("client_order_id")
        signals_recent.append(
            {
                "ts_label": ts_label,
                "pocket": row.get("pocket") or "-",
                "strategy": row.get("strategy") or "-",
                "action": row.get("action") or "-",
                "confidence": row.get("confidence"),
                "units": row.get("proposed_units"),
                "id_short": _shorten(signal_id, 22),
                "id_full": str(signal_id or ""),
            }
        )
    system["signals_recent"] = signals_recent

    winners = sorted(closed_trades, key=lambda t: t["pl_pips"], reverse=True)[:3]
    losers = sorted(closed_trades, key=lambda t: t["pl_pips"])[:3]
    highlights: list[Dict[str, Any]] = []
    seen = set()
    for tr in winners:
        if tr["pl_pips"] <= 0:
            continue
        if tr["ticket_id"] in seen:
            continue
        highlights.append(
            {
                "ticket_id": tr["ticket_id"],
                "pocket": tr["pocket"],
                "pl_pips": round(tr["pl_pips"], 2),
                "pl_jpy": round(tr["pl_jpy"], 2),
                "closed_at": tr["close_label"],
                "kind": "gain",
            }
        )
        seen.add(tr["ticket_id"])
    for tr in losers:
        if tr["pl_pips"] >= 0:
            continue
        if tr["ticket_id"] in seen:
            continue
        highlights.append(
            {
                "ticket_id": tr["ticket_id"],
                "pocket": tr["pocket"],
                "pl_pips": round(tr["pl_pips"], 2),
                "pl_jpy": round(tr["pl_jpy"], 2),
                "closed_at": tr["close_label"],
                "kind": "loss",
            }
        )
        seen.add(tr["ticket_id"])

    latest_highlights: list[Dict[str, Any]] = []
    for tr in closed_trades[:6]:
        kind = "gain" if tr["pl_pips"] > 0 else "loss" if tr["pl_pips"] < 0 else "flat"
        latest_highlights.append(
            {
                "ticket_id": tr["ticket_id"],
                "pocket": tr["pocket"],
                "pl_pips": round(tr["pl_pips"], 2),
                "pl_jpy": round(tr["pl_jpy"], 2),
                "closed_at": tr["close_label"],
                "kind": kind,
            }
        )

    base["highlights"] = highlights
    base["highlights_top"] = highlights
    base["highlights_recent"] = latest_highlights

    gen_dt = _parse_dt(snapshot.get("generated_at"))
    base["generated_at"] = snapshot.get("generated_at")
    base["generated_label"] = _format_dt(gen_dt) or snapshot.get("generated_at")
    base["available"] = True
    base["error"] = None
    return base


def _load_dashboard_data() -> Dict[str, Any]:
    base = _dashboard_defaults()
    remote_snapshot = _fetch_remote_snapshot()
    if remote_snapshot:
        return _summarise_snapshot(remote_snapshot)

    try:
        bucket_name = get_secret("ui_bucket_name")
        object_path = get_secret("ui_state_object_path")
    except KeyError as exc:  # pragma: no cover - missing config
        base["error"] = f"{exc.args[0]} が未設定です"
        return base

    if storage is None:  # pragma: no cover - optional dependency
        base["error"] = "google-cloud-storage クライアントが利用できません"
        return base

    try:
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(object_path)
        raw = blob.download_as_text(timeout=5)
        snapshot = json.loads(raw)
    except Exception as exc:  # pragma: no cover - network/credential issues
        base["error"] = str(exc)
        return base

    return _summarise_snapshot(snapshot)


def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None


@app.get("/")
def root_redirect():
    return RedirectResponse(url="/dashboard", status_code=307)


@app.get("/api/snapshot")
def api_snapshot(
    token: Optional[str] = None,
    x_qr_token: Optional[str] = Header(default=None, alias="X-QR-Token"),
    authorization: Optional[str] = Header(default=None),
):
    required = _get_secret_optional("ui_snapshot_token")
    provided = token or x_qr_token or _extract_bearer(authorization)
    if required and provided != required:
        raise HTTPException(status_code=401, detail="Unauthorized")
    snapshot = _cached_live_snapshot()
    return JSONResponse(snapshot, headers={"Cache-Control": "no-store"})


@app.get("/dashboard")
def dashboard(request: Request):
    dashboard_data = _load_dashboard_data()
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "dashboard": dashboard_data,
            "active_tab": "dashboard",
        },
    )


@app.get("/excursion")
def excursion_report(request: Request, file: Optional[str] = None):
    return RedirectResponse(url="/dashboard", status_code=307)


@app.get("/autotune")
def autotune_home(
    request: Request,
    status: str = "all",
    sort: str = "updated",
    order: str = "desc",
    q: str = "",
    page: int = 1,
    page_size: int = 50,
):
    return RedirectResponse(url="/dashboard", status_code=307)


@app.get("/runs/{run_id}/{strategy}")
def run_detail(request: Request, run_id: str, strategy: str):
    row = get_run(None, run_id, strategy)
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")
    run = dump_dict(row)
    run["params_pretty"] = json.dumps(run.get("params", {}), ensure_ascii=False, indent=2)
    run["train_pretty"] = json.dumps(run.get("train", {}), ensure_ascii=False, indent=2)
    run["valid_pretty"] = json.dumps(run.get("valid", {}), ensure_ascii=False, indent=2)
    run["summary_jp"] = _build_summary(run)
    return templates.TemplateResponse(
        "detail.html",
        {
            "request": request,
            "run": run,
        },
    )


@app.post("/runs/{run_id}/{strategy}/decision")
def set_decision(
    run_id: str,
    strategy: str,
    action: str = Form(...),
    reviewer: str = Form(""),
    comment: str = Form(""),
):
    if action not in {"approve", "reject"}:
        raise HTTPException(status_code=400, detail="Invalid action")
    status = "approved" if action == "approve" else "rejected"
    run_record = get_run(None, run_id, strategy)
    if not run_record:
        raise HTTPException(status_code=404, detail="Run not found")
    run = dump_dict(run_record)
    update_status(None, run_id, strategy, status, reviewer or None, comment or None)
    if status == "approved":
        _apply_params_to_config(run)
    return RedirectResponse(url=f"/runs/{run_id}/{strategy}", status_code=303)


@app.post("/settings/autotune")
def update_settings(action: str = Form(...), reviewer: str = Form("")):
    if action not in {"enable", "disable"}:
        raise HTTPException(status_code=400, detail="Invalid action")
    enabled = action == "enable"
    set_settings(None, enabled=enabled, updated_by=reviewer or None)
    return RedirectResponse(url="/", status_code=303)


def _apply_params_to_config(run: dict) -> None:
    params = run.get("params") or {}
    strategy = run.get("strategy")
    if not params or not strategy:
        return

    try:
        if CONFIG_PATH.exists():
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                current = json.load(f)
        else:
            current = {}
    except Exception:
        current = {}

    current[strategy] = params

    tmp_path = CONFIG_PATH.with_suffix(".json.tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(current, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, CONFIG_PATH)
