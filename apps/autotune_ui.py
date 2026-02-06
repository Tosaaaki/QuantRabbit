from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlencode
from typing import Any, Dict, Optional

from fastapi import FastAPI, Form, HTTPException, Header, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
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
TRADES_DB = Path("logs/trades.db")
HEALTH_SNAPSHOT = Path("logs/health_snapshot.json")

_LIVE_SNAPSHOT_TTL_SEC = int(os.getenv("LIVE_SNAPSHOT_TTL_SEC", "8"))
_REMOTE_SNAPSHOT_TIMEOUT_SEC = float(os.getenv("UI_SNAPSHOT_TIMEOUT_SEC", "4.0"))
_LIVE_SNAPSHOT_LITE_TTL_SEC = int(os.getenv("LIVE_SNAPSHOT_LITE_TTL_SEC", "5"))
_LITE_SNAPSHOT_FAST = (
    os.getenv("UI_SNAPSHOT_LITE_MODE", "full").strip().lower()
    in {"fast", "minimal"}
)
_INCLUDE_POSITIONS = os.getenv("UI_SNAPSHOT_INCLUDE_POSITIONS", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
_AUTO_REFRESH_SEC = max(5, int(os.getenv("UI_AUTO_REFRESH_SEC", "15")))
_RECENT_TRADES_LIMIT = max(20, int(os.getenv("UI_RECENT_TRADES_LIMIT", "80")))
_RECENT_TRADES_DISPLAY = max(
    10, min(int(os.getenv("UI_RECENT_TRADES_DISPLAY", "30")), _RECENT_TRADES_LIMIT)
)
_HOURLY_TRADES_LOOKBACK = max(6, int(os.getenv("UI_HOURLY_LOOKBACK_HOURS", "24")))
_DB_READ_TIMEOUT_SEC = float(os.getenv("UI_DB_READ_TIMEOUT_SEC", "0.2"))
_OPS_REMOTE_TIMEOUT_SEC = float(os.getenv("UI_OPS_TIMEOUT_SEC", "4.0"))
_OPS_COMMAND_TIMEOUT_SEC = float(os.getenv("UI_OPS_CMD_TIMEOUT_SEC", "6.0"))
_DEFAULT_UI_STATE_OBJECT = "realtime/ui_state.json"
_LOCAL_FALLBACK_ENABLED = os.getenv("UI_DASHBOARD_LOCAL_FALLBACK", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}
_LOCAL_FALLBACK_MODE = os.getenv("UI_DASHBOARD_LOCAL_MODE", "lite").strip().lower()
_LITE_SYNC_TRADES_ENABLED = os.getenv("UI_SNAPSHOT_SYNC_TRADES", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
_LITE_SYNC_TTL_SEC = float(os.getenv("UI_SNAPSHOT_SYNC_TTL_SEC", "60"))
_LITE_SYNC_MARKER = Path(
    os.getenv("UI_SNAPSHOT_SYNC_MARKER", "logs/ui_snapshot_sync.json")
)
_live_snapshot_lock = threading.Lock()
_live_snapshot_cache: dict[str, Any] | None = None
_live_snapshot_ts: float = 0.0
_lite_snapshot_lock = threading.Lock()
_lite_snapshot_cache: dict[str, Any] | None = None
_lite_snapshot_ts: float = 0.0

app = FastAPI(title="QuantRabbit Console")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

_WORKER_TAG_MAP = {
    "TrendMA": "macro_trendma",
    "H1Momentum": "macro_h1momentum",
    "LondonMomentum": "london_momentum",
    "trend_h1": "trend_h1",
    "manual_swing": "manual_swing",
    "OnePipMakerS1": "onepip_maker_s1",
    "M1Scalper": "scalp_m1scalper",
    "ImpulseRetrace": "scalp_impulseretrace",
    "ImpulseRetraceScalp": "scalp_impulseretrace",
    "RangeFader": "scalp_rangefader",
    "PulseBreak": "scalp_pulsebreak",
    "BB_RSI": "micro_bbrsi",
    "BB_RSI_Fast": "micro_bbrsi",
    "MicroLevelReactor": "micro_levelreactor",
    "MicroMomentumBurst": "micro_momentumburst",
    "MicroMomentumStack": "micro_momentumstack",
    "MicroPullbackEMA": "micro_pullbackema",
    "MicroRangeBreak": "micro_rangebreak",
    "MicroVWAPBound": "micro_vwapbound",
    "MicroVWAPRevert": "micro_vwapbound",
    "TrendMomentumMicro": "micro_trendmomentum",
    "MomentumBurst": "micro_momentumburst",
    "MomentumPulse": "micro_momentumburst",
    "VolCompressionBreak": "micro_multistrat",
    "VolSpikeRider": "vol_spike_rider",
}


def _coerce_thesis(value: Any) -> Optional[dict]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _infer_worker_name(item: dict) -> Optional[str]:
    thesis = _coerce_thesis(item.get("entry_thesis"))
    if isinstance(thesis, dict):
        worker = thesis.get("worker_id") or thesis.get("worker")
        if worker:
            return str(worker)
        tag = thesis.get("strategy_tag") or thesis.get("strategy")
    else:
        tag = None

    if not tag:
        tag = item.get("strategy_tag") or item.get("strategy")
    if not tag:
        return None

    tag_str = str(tag).strip()
    if not tag_str:
        return None

    return _WORKER_TAG_MAP.get(tag_str) or _WORKER_TAG_MAP.get(tag_str.lower()) or tag_str


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
        "snapshot": {
            "mode": None,
            "source": None,
            "age_sec": None,
            "stale": False,
        },
        "account": {
            "nav": None,
            "balance": None,
            "margin_usage_ratio": None,
            "margin_usage_pct": None,
            "free_margin_ratio": None,
            "free_margin_pct": None,
            "health_buffer": None,
            "health_buffer_pct": None,
        },
        "auto_refresh_sec": _AUTO_REFRESH_SEC,
        "recent_trades_limit": _RECENT_TRADES_DISPLAY,
        "recent_trades": [],
        "performance": {
            "daily_pl_pips": 0.0,
            "daily_pl_jpy": 0.0,
            "daily_pl_eq1l": 0.0,
            "yesterday_pl_pips": 0.0,
            "yesterday_pl_jpy": 0.0,
            "daily_change_pips": 0.0,
            "daily_change_jpy": 0.0,
            "daily_change_pct": None,
            "daily_change_equity": None,
            "daily_change_equity_source": None,
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
        "cashflow": {
            "daily": {"net": 0.0, "in": 0.0, "out": 0.0, "count": 0},
            "yesterday": {"net": 0.0, "in": 0.0, "out": 0.0, "count": 0},
            "weekly": {"net": 0.0, "in": 0.0, "out": 0.0, "count": 0},
            "total": {"net": 0.0, "in": 0.0, "out": 0.0, "count": 0},
            "ytd": {"net": 0.0, "in": 0.0, "out": 0.0, "count": 0},
        },
        "open_summary": {
            "total_positions": 0,
            "net_units": 0.0,
            "long_units": 0.0,
            "short_units": 0.0,
            "unrealized_pl_pips": 0.0,
            "unrealized_pl_jpy": 0.0,
            "pockets": [],
            "open_trades": [],
            "meta": {},
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
            "orders_errors": [],
            "log_errors": [],
            "signals_recent": [],
        },
        "health": {
            "hostname": None,
            "deploy_id": None,
            "git_rev": None,
            "uptime_sec": None,
            "uptime_label": None,
            "service_active": {},
            "service_info": [],
        },
        "profit_tables": {
            "timezone": "JST",
            "year_start": None,
            "exclude_manual": True,
            "daily": [],
            "weekly": [],
            "monthly": [],
        },
        "hourly_trades": {
            "timezone": "JST",
            "lookback_hours": _HOURLY_TRADES_LOOKBACK,
            "exclude_manual": True,
            "hours": [],
        },
        "ytd_summary": {
            "year_start": None,
            "timezone": "JST",
            "exclude_manual": True,
            "start_balance": None,
            "start_balance_source": None,
            "current_balance": None,
            "balance_growth_pct": None,
            "balance_growth_ex_cashflow_jpy": None,
            "balance_growth_ex_cashflow_pct": None,
            "bot_profit_jpy": 0.0,
            "bot_profit_pips": 0.0,
            "bot_return_pct": None,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "cashflow_net": 0.0,
            "cashflow_in": 0.0,
            "cashflow_out": 0.0,
        },
        "charts": {
            "available": False,
            "performance": {"default_range": None, "ranges": {}},
            "price": {"default_range": None, "ranges": {}},
        },
        "charts_json": "{}",
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


_JST = timezone(timedelta(hours=9))


def _format_dt(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    return dt.astimezone(_JST).strftime("%Y-%m-%d %H:%M JST")


def _ts_ms_to_dt(ts_ms: Any) -> Optional[datetime]:
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


def _format_uptime(uptime_sec: Optional[float]) -> Optional[str]:
    if uptime_sec is None:
        return None
    try:
        total = int(uptime_sec)
    except Exception:
        return None
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _pocket_label(pocket: Optional[str]) -> str:
    if not pocket:
        return "-"
    key = str(pocket).strip().lower()
    labels = {
        "micro": "micro/短期",
        "macro": "macro/中期",
        "scalp": "scalp/超短期",
        "scalp_fast": "scalp_fast/超短期+",
        "manual": "manual/手動",
    }
    return labels.get(key, str(pocket))


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


def _load_health_snapshot_local() -> Optional[dict]:
    if not HEALTH_SNAPSHOT.exists():
        return None
    try:
        data = json.loads(HEALTH_SNAPSHOT.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _resolve_ui_state_object_path() -> str:
    value = _get_secret_optional("ui_state_object_path") or os.getenv("UI_STATE_OBJECT_PATH")
    value = (value or _DEFAULT_UI_STATE_OBJECT).strip()
    return value or _DEFAULT_UI_STATE_OBJECT


def _snapshot_sort_key(source: str, snapshot: dict) -> tuple[float, int]:
    dt = _parse_dt(snapshot.get("generated_at"))
    ts = dt.timestamp() if dt else 0.0
    priority = {"local": 0, "gcs": 1, "remote": 2}.get(source, 0)
    return (ts, priority)


def _pick_latest_snapshot(candidates: list[tuple[str, dict]]) -> Optional[tuple[str, dict]]:
    if not candidates:
        return None
    return max(candidates, key=lambda item: _snapshot_sort_key(item[0], item[1]))


def _fetch_gcs_snapshot() -> tuple[Optional[dict], Optional[str]]:
    bucket_name = _get_secret_optional("ui_bucket_name")
    if not bucket_name:
        return None, "ui_bucket_name が未設定です"
    if storage is None:
        return None, "google-cloud-storage クライアントが利用できません"
    object_path = _resolve_ui_state_object_path()
    try:
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(object_path)
        raw = blob.download_as_text(timeout=5)
        return json.loads(raw), None
    except Exception as exc:  # pragma: no cover - network/credential issues
        return None, str(exc)


def _should_sync_trades() -> bool:
    if not _LITE_SYNC_TRADES_ENABLED:
        return False
    if _LITE_SYNC_TTL_SEC <= 0:
        return True
    if not _LITE_SYNC_MARKER.exists():
        return True
    try:
        data = json.loads(_LITE_SYNC_MARKER.read_text(encoding="utf-8"))
    except Exception:
        return True
    last_ts = float(data.get("ts") or 0.0)
    return (time.time() - last_ts) >= _LITE_SYNC_TTL_SEC


def _mark_sync_trades(count: int) -> None:
    try:
        _LITE_SYNC_MARKER.parent.mkdir(parents=True, exist_ok=True)
        _LITE_SYNC_MARKER.write_text(
            json.dumps({"ts": time.time(), "count": int(count)}),
            encoding="utf-8",
        )
    except Exception:
        pass


def _build_local_snapshot() -> Optional[dict]:
    if not _LOCAL_FALLBACK_ENABLED:
        return None
    try:
        if _LOCAL_FALLBACK_MODE in {"live", "full"}:
            return _cached_live_snapshot()
        return _cached_lite_snapshot()
    except Exception:
        return None


def _fallback_dashboard_local() -> Optional[Dict[str, Any]]:
    snapshot = _build_local_snapshot()
    if not snapshot:
        return None
    base = _summarise_snapshot(snapshot)
    if not base.get("snapshot", {}).get("source"):
        base.setdefault("snapshot", {})["source"] = "local"
    return base


def _ops_required_token() -> Optional[str]:
    return _get_secret_optional("ui_ops_token") or _get_secret_optional("ui_snapshot_token")


def _ops_remote_url() -> Optional[str]:
    value = _get_secret_optional("ui_ops_url")
    return value.rstrip("/") if value else None


def _discover_trade_units() -> list[str]:
    raw = _get_secret_optional("ui_trade_units") or os.getenv("UI_TRADE_UNITS")
    if raw:
        return sorted({u.strip() for u in raw.replace(",", " ").split() if u.strip()})

    units: set[str] = set()
    base = Path("/etc/systemd/system")
    if base.exists():
        for path in base.glob("quant-*.service"):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if "workers." in text or "/workers/" in text:
                units.add(path.name)
        if (base / "quantrabbit.service").exists():
            units.add("quantrabbit.service")
    return sorted(units)


def _ensure_sudo() -> None:
    try:
        result = subprocess.run(
            ["sudo", "-n", "true"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"sudo check failed: {exc}") from exc
    if result.returncode != 0:
        raise HTTPException(status_code=403, detail="sudo permission is required")


def _local_ops_action(action: str) -> dict:
    units = _discover_trade_units()
    if not units:
        return {"ok": False, "error": "trade units not found"}
    _ensure_sudo()
    results = []
    for unit in units:
        try:
            proc = subprocess.run(
                ["sudo", "-n", "systemctl", action, unit],
                capture_output=True,
                text=True,
                timeout=_OPS_COMMAND_TIMEOUT_SEC,
                check=False,
            )
            ok = proc.returncode == 0
            results.append(
                {
                    "unit": unit,
                    "ok": ok,
                    "error": (proc.stderr or "").strip() if not ok else "",
                }
            )
        except Exception as exc:
            results.append({"unit": unit, "ok": False, "error": str(exc)})
    ok_count = sum(1 for row in results if row["ok"])
    failed = [row for row in results if not row["ok"]]
    return {
        "ok": ok_count == len(results),
        "action": action,
        "total": len(results),
        "ok_count": ok_count,
        "failed": failed,
    }


def _proxy_ops_action(action: str, confirm: str, token: str) -> dict:
    ops_url = _ops_remote_url()
    if not ops_url:
        return {"ok": False, "error": "ops remote url is not configured"}
    payload = urlencode({"action": action, "confirm": confirm, "ops_token": token}).encode("utf-8")
    req = urllib.request.Request(f"{ops_url}/api/ops/control", data=payload, method="POST")
    if token:
        req.add_header("X-QR-Token", token)
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=_OPS_REMOTE_TIMEOUT_SEC) as resp:
            raw = resp.read()
        return json.loads(raw)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _handle_ops_action(
    action: str,
    confirm: str,
    token: Optional[str],
    x_qr_token: Optional[str],
    authorization: Optional[str],
    *,
    allow_proxy: bool,
) -> dict:
    required = _ops_required_token()
    if not required:
        raise HTTPException(status_code=503, detail="ui_ops_token is not configured")
    provided = token or x_qr_token or _extract_bearer(authorization)
    if provided != required:
        raise HTTPException(status_code=401, detail="Unauthorized")
    action = (action or "").strip().lower()
    if action not in {"stop", "start"}:
        raise HTTPException(status_code=400, detail="Invalid action")
    expected = "STOP" if action == "stop" else "START"
    if (confirm or "").strip().upper() != expected:
        raise HTTPException(status_code=400, detail=f"Confirm with {expected}")
    if allow_proxy and _ops_remote_url():
        return _proxy_ops_action(action, confirm, provided)
    return _local_ops_action(action)


def _load_latest_metric(metric: str) -> Optional[float]:
    if not METRICS_DB.exists():
        return None
    try:
        con = sqlite3.connect(METRICS_DB, timeout=_DB_READ_TIMEOUT_SEC)
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
        con = sqlite3.connect(METRICS_DB, timeout=_DB_READ_TIMEOUT_SEC)
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
        con = sqlite3.connect(ORDERS_DB, timeout=_DB_READ_TIMEOUT_SEC)
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
        con = sqlite3.connect(ORDERS_DB, timeout=_DB_READ_TIMEOUT_SEC)
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
        return _load_last_order_ts_ms()
    try:
        con = sqlite3.connect(SIGNALS_DB, timeout=_DB_READ_TIMEOUT_SEC)
        cur = con.execute("SELECT max(ts_ms) FROM signals")
        row = cur.fetchone()
        con.close()
        if not row:
            return _load_last_order_ts_ms()
        val = row[0]
        sig_ts = int(val) if val is not None else None
    except Exception:
        sig_ts = None
    order_ts = _load_last_order_ts_ms()
    if order_ts is None:
        return sig_ts
    if sig_ts is None:
        return order_ts
    if order_ts > sig_ts + 30 * 60 * 1000:
        return order_ts
    return sig_ts


def _dt_to_ts_ms(ts: object) -> Optional[int]:
    dt = _parse_dt(ts)
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _load_last_order_ts_ms() -> Optional[int]:
    if not ORDERS_DB.exists():
        return None
    try:
        con = sqlite3.connect(ORDERS_DB, timeout=_DB_READ_TIMEOUT_SEC)
        cur = con.execute("SELECT max(ts) FROM orders WHERE request_json IS NOT NULL")
        row = cur.fetchone()
        con.close()
        if not row:
            return None
        return _dt_to_ts_ms(row[0])
    except Exception:
        return None


def _load_recent_order_signals(limit: int = 5) -> list[dict]:
    if not ORDERS_DB.exists():
        return []
    try:
        con = sqlite3.connect(ORDERS_DB, timeout=_DB_READ_TIMEOUT_SEC)
        con.row_factory = sqlite3.Row
        cur = con.execute(
            "SELECT ts, pocket, side, units, status, client_order_id, request_json "
            "FROM orders WHERE request_json IS NOT NULL "
            "AND status NOT LIKE 'close_%' "
            "ORDER BY ts DESC LIMIT ?",
            (int(limit),),
        )
        rows = [dict(r) for r in cur.fetchall()]
        con.close()
    except Exception:
        return []

    results: list[dict] = []
    for row in rows:
        ts_ms = _dt_to_ts_ms(row.get("ts"))
        if ts_ms is None:
            continue
        try:
            payload = json.loads(row.get("request_json") or "{}")
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        entry_thesis = payload.get("entry_thesis") or {}
        if not isinstance(entry_thesis, dict):
            entry_thesis = {}
        strategy = (
            entry_thesis.get("strategy_tag")
            or payload.get("strategy_tag")
            or payload.get("strategy")
            or (payload.get("meta") or {}).get("strategy_tag")
        )
        if not strategy:
            continue
        confidence = entry_thesis.get("confidence") or payload.get("confidence")
        side = (row.get("side") or "").lower()
        action = "OPEN_LONG" if side == "buy" else "OPEN_SHORT" if side == "sell" else None
        results.append(
            {
                "ts_ms": ts_ms,
                "pocket": row.get("pocket"),
                "strategy": strategy,
                "confidence": confidence,
                "action": action,
                "client_order_id": row.get("client_order_id"),
                "proposed_units": row.get("units"),
            }
        )
    return results


def _load_recent_signals(limit: int = 5) -> list[dict]:
    if not SIGNALS_DB.exists():
        return _load_recent_order_signals(limit=limit)
    try:
        con = sqlite3.connect(SIGNALS_DB, timeout=_DB_READ_TIMEOUT_SEC)
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
        if rows:
            order_signals = _load_recent_order_signals(limit=limit)
            if order_signals:
                sig_ts = rows[0].get("ts_ms")
                ord_ts = order_signals[0].get("ts_ms")
                if isinstance(sig_ts, int) and isinstance(ord_ts, int):
                    if ord_ts > sig_ts + 30 * 60 * 1000:
                        return order_signals
            return rows
        return _load_recent_order_signals(limit=limit)
    except Exception:
        return _load_recent_order_signals(limit=limit)


def _load_recent_trades(limit: int = 50) -> list[dict]:
    if not TRADES_DB.exists():
        return []
    try:
        con = sqlite3.connect(TRADES_DB, timeout=_DB_READ_TIMEOUT_SEC)
        con.row_factory = sqlite3.Row
        cur = con.execute(
            """
            SELECT ticket_id, pocket, instrument, units, closed_units, entry_price, close_price,
                   fill_price, pl_pips, realized_pl, commission, financing,
                   entry_time, close_time, close_reason,
                   state, updated_at,
                   strategy_tag, strategy, client_order_id, entry_thesis
            FROM trades
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        rows = [dict(r) for r in cur.fetchall()]
        con.close()
        return rows
    except Exception:
        return []


def _build_hourly_fallback(trades: list[dict]) -> dict:
    now = datetime.now(timezone.utc)
    jst = timezone(timedelta(hours=9))
    now_jst = now.astimezone(jst)
    now_hour = now_jst.replace(minute=0, second=0, microsecond=0)
    start_hour = now_hour - timedelta(hours=_HOURLY_TRADES_LOOKBACK - 1)
    buckets: dict[datetime, dict[str, float | int]] = {}
    for i in range(_HOURLY_TRADES_LOOKBACK):
        hour = now_hour - timedelta(hours=i)
        buckets[hour] = {"pips": 0.0, "jpy": 0.0, "trades": 0, "wins": 0, "losses": 0}

    for item in trades or []:
        pocket = (item.get("pocket") or "").lower()
        if pocket == "manual":
            continue
        close_dt = _parse_dt(item.get("close_time") or item.get("updated_at"))
        if not close_dt:
            continue
        close_jst = close_dt.astimezone(jst)
        if close_jst < start_hour:
            continue
        hour_key = close_jst.replace(minute=0, second=0, microsecond=0)
        bucket = buckets.get(hour_key)
        if not bucket:
            continue
        pl_pips = _safe_float(item.get("pl_pips"))
        pl_jpy = _safe_float(item.get("realized_pl"))
        bucket["pips"] += pl_pips
        bucket["jpy"] += pl_jpy
        bucket["trades"] += 1
        if pl_pips > 0:
            bucket["wins"] += 1
        elif pl_pips < 0:
            bucket["losses"] += 1

    rows: list[dict] = []
    for hour in sorted(buckets.keys(), reverse=True):
        data = buckets[hour]
        trades = data["trades"]
        win_rate = (data["wins"] / trades) if trades else 0.0
        rows.append(
            {
                "key": hour.isoformat(),
                "label": hour.strftime("%m/%d %H:00"),
                "pips": round(float(data["pips"]), 2),
                "jpy": round(float(data["jpy"]), 2),
                "trades": int(trades),
                "wins": int(data["wins"]),
                "losses": int(data["losses"]),
                "win_rate": win_rate,
            }
        )
    return {
        "timezone": "JST",
        "lookback_hours": _HOURLY_TRADES_LOOKBACK,
        "exclude_manual": True,
        "hours": rows,
    }


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
        open_positions = pm.get_open_positions(include_unknown=True)
    except Exception:
        open_positions = {}

    try:
        recent_trades = pm.fetch_recent_trades(limit=_RECENT_TRADES_LIMIT)
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
        health_snapshot = _load_health_snapshot_local()
        if health_snapshot:
            metrics["health_snapshot"] = health_snapshot
        for key in (
            "account.nav",
            "account.balance",
            "account.free_margin_ratio",
            "account.margin_usage_ratio",
            "account.health_buffer",
        ):
            value = _load_latest_metric(key)
            if value is not None:
                metrics[key] = value
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
        "snapshot_mode": "live",
        "snapshot_source": "local",
    }


def _build_lite_snapshot() -> dict:
    recent_trades = _load_recent_trades(limit=_RECENT_TRADES_LIMIT)
    metrics: dict[str, Any] = {}
    open_positions: dict[str, Any] = {}
    pm = None
    sync_needed = _should_sync_trades()
    try:
        from execution.position_manager import PositionManager
    except Exception:
        PositionManager = None  # type: ignore[assignment]
    if PositionManager is not None and (
        sync_needed or (not _LITE_SNAPSHOT_FAST) or _INCLUDE_POSITIONS
    ):
        pm = PositionManager()
        try:
            if sync_needed:
                try:
                    synced = pm.sync_trades()
                    _mark_sync_trades(len(synced or []))
                except Exception:
                    pass
            try:
                metrics = pm.get_performance_summary()
            except Exception:
                metrics = {}
            try:
                recent_trades = pm.fetch_recent_trades(limit=_RECENT_TRADES_LIMIT)
            except Exception:
                recent_trades = _load_recent_trades(limit=_RECENT_TRADES_LIMIT)
            if (not _LITE_SNAPSHOT_FAST) or _INCLUDE_POSITIONS:
                try:
                    open_positions = pm.get_open_positions(include_unknown=True) or {}
                except Exception:
                    open_positions = {}
        finally:
            try:
                pm.close()
            except Exception:
                pass
    data_lag_ms = _load_latest_metric("data_lag_ms")
    decision_latency_ms = _load_latest_metric("decision_latency_ms")
    if data_lag_ms is not None:
        metrics["data_lag_ms"] = data_lag_ms
    if decision_latency_ms is not None:
        metrics["decision_latency_ms"] = decision_latency_ms
    for key in (
        "account.nav",
        "account.balance",
        "account.free_margin_ratio",
        "account.margin_usage_ratio",
        "account.health_buffer",
    ):
        value = _load_latest_metric(key)
        if value is not None:
            metrics[key] = value
    health_snapshot = _load_health_snapshot_local()
    if health_snapshot:
        metrics["health_snapshot"] = health_snapshot
    metrics["healthbeat_ts"] = _load_last_metric_ts("healthbeat")
    if not _LITE_SNAPSHOT_FAST:
        metrics["orders_last"] = _load_last_orders()
        metrics["orders_status_1h"] = _load_order_status_counts()
        last_signal_ts = _load_last_signal_ts_ms()
        if last_signal_ts is not None:
            metrics["signals_last_ts_ms"] = last_signal_ts
        metrics["signals_recent"] = _load_recent_signals()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "new_trades": [],
        "recent_trades": list(recent_trades),
        "open_positions": dict(open_positions) if isinstance(open_positions, dict) else {},
        "metrics": metrics,
        "snapshot_mode": "lite-fast" if _LITE_SNAPSHOT_FAST else "lite",
        "snapshot_source": "local",
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


def _cached_lite_snapshot() -> dict:
    global _lite_snapshot_cache, _lite_snapshot_ts
    now_mono = time.monotonic()
    if _lite_snapshot_cache and (now_mono - _lite_snapshot_ts) < _LIVE_SNAPSHOT_LITE_TTL_SEC:
        return _lite_snapshot_cache
    with _lite_snapshot_lock:
        now_mono = time.monotonic()
        if _lite_snapshot_cache and (now_mono - _lite_snapshot_ts) < _LIVE_SNAPSHOT_LITE_TTL_SEC:
            return _lite_snapshot_cache
        snapshot = _build_lite_snapshot()
        _lite_snapshot_cache = snapshot
        _lite_snapshot_ts = now_mono
        return snapshot


def _fetch_remote_snapshot(key: str = "ui_snapshot_url") -> Optional[dict]:
    url = _get_secret_optional(key)
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
    jst = timezone(timedelta(hours=9))
    now_jst = now.astimezone(jst)

    metrics_snapshot = snapshot.get("metrics") or {}
    account = base.get("account") or {}
    account_nav = _opt_float(metrics_snapshot.get("account.nav"))
    account_balance = _opt_float(metrics_snapshot.get("account.balance"))
    margin_usage_ratio = _opt_float(metrics_snapshot.get("account.margin_usage_ratio"))
    free_margin_ratio = _opt_float(metrics_snapshot.get("account.free_margin_ratio"))
    health_buffer = _opt_float(metrics_snapshot.get("account.health_buffer"))
    account.update(
        {
            "nav": account_nav,
            "balance": account_balance,
            "margin_usage_ratio": margin_usage_ratio,
            "margin_usage_pct": round(margin_usage_ratio * 100.0, 1)
            if margin_usage_ratio is not None
            else None,
            "free_margin_ratio": free_margin_ratio,
            "free_margin_pct": round(free_margin_ratio * 100.0, 1)
            if free_margin_ratio is not None
            else None,
            "health_buffer": health_buffer,
            "health_buffer_pct": round(health_buffer * 100.0, 1)
            if health_buffer is not None
            else None,
        }
    )
    base["account"] = account
    trades_raw = snapshot.get("recent_trades") or []
    parsed_trades: list[Dict[str, Any]] = []
    for item in trades_raw:
        close_dt = _parse_dt(item.get("close_time") or item.get("updated_at"))
        close_dt_jst = close_dt.astimezone(jst) if close_dt else None
        worker = _infer_worker_name(item)
        parsed_trades.append(
            {
                "ticket_id": str(item.get("ticket_id") or ""),
                "pocket": (item.get("pocket") or "-").strip() or "-",
                "worker": worker or "-",
                "pl_pips": _safe_float(item.get("pl_pips")),
                "pl_jpy": _safe_float(item.get("realized_pl")),
                "close_time": close_dt,
                "close_time_jst": close_dt_jst,
                "close_label": _format_dt(close_dt),
            }
        )

    recent_trades_display: list[Dict[str, Any]] = []
    for item in trades_raw[:_RECENT_TRADES_DISPLAY]:
        entry_dt = _parse_dt(item.get("entry_time"))
        close_dt = _parse_dt(item.get("close_time"))
        updated_dt = _parse_dt(item.get("updated_at") or item.get("close_time") or item.get("entry_time"))
        units_val = _safe_float(item.get("units"))
        direction = "Long" if units_val > 0 else "Short" if units_val < 0 else "Flat"
        units_abs = int(round(abs(units_val)))
        pl_pips = _safe_float(item.get("pl_pips"))
        pl_jpy = _safe_float(item.get("realized_pl"))
        kind = "gain" if pl_pips > 0 else "loss" if pl_pips < 0 else "neutral"
        worker = _infer_worker_name(item)
        strategy = item.get("strategy_tag") or item.get("strategy") or "-"
        recent_trades_display.append(
            {
                "ticket_id": str(item.get("ticket_id") or ""),
                "pocket": (item.get("pocket") or "-").strip() or "-",
                "worker": worker or "-",
                "strategy": strategy,
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

    hourly_trades = metrics_snapshot.get("hourly_trades")
    if isinstance(hourly_trades, dict):
        base["hourly_trades"] = hourly_trades
    else:
        base["hourly_trades"] = _build_hourly_fallback(trades_raw)

    closed_trades = [t for t in parsed_trades if t["close_time"]]
    closed_trades.sort(key=lambda t: t["close_time"], reverse=True)
    today_start_jst = now_jst.replace(hour=0, minute=0, second=0, microsecond=0)
    # Rolling 7-day window in JST (includes today).
    week_cutoff = today_start_jst - timedelta(days=6)
    today_date = now_jst.date()

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
        elif target == "yesterday":
            perf["yesterday_pl_pips"] = data.get("pips", 0.0)
            perf["yesterday_pl_jpy"] = data.get("jpy", 0.0)
        elif target == "daily_change":
            perf["daily_change_pips"] = data.get("pips", 0.0)
            perf["daily_change_jpy"] = data.get("jpy", 0.0)
            perf["daily_change_pct"] = data.get("jpy_pct")
            perf["daily_change_equity"] = data.get("equity_nav")
            perf["daily_change_equity_source"] = data.get("equity_source")
        elif target == "weekly":
            perf["weekly_pl_pips"] = data.get("pips", 0.0)
            perf["weekly_pl_jpy"] = data.get("jpy", 0.0)
            # "recent_*" metrics in the UI correspond to the same rolling 7-day window.
            trades = int(data.get("trades", 0) or 0)
            perf["recent_closed"] = trades
            perf["wins"] = int(data.get("wins", 0) or 0)
            perf["losses"] = int(data.get("losses", 0) or 0)
            wr = data.get("win_rate")
            if wr is None:
                wr = (perf["wins"] / trades) if trades else 0.0
            perf["win_rate"] = float(wr)
            perf["win_rate_percent"] = round(float(wr) * 100.0, 1)
        elif target == "total":
            perf["total_pips"] = data.get("pips", 0.0)
            perf["total_jpy"] = data.get("jpy", 0.0)
            perf["total_trades"] = data.get("trades", 0)

    _apply_metrics(metrics_snapshot.get("daily"), target="daily")
    _apply_metrics(metrics_snapshot.get("yesterday"), target="yesterday")
    _apply_metrics(metrics_snapshot.get("daily_change"), target="daily_change")
    _apply_metrics(metrics_snapshot.get("weekly"), target="weekly")
    _apply_metrics(metrics_snapshot.get("total"), target="total")

    if perf.get("total_trades") in (None, 0):
        perf["total_trades"] = len(parsed_trades)

    weekly_metrics = metrics_snapshot.get("weekly")
    if not isinstance(weekly_metrics, dict) or not weekly_metrics:
        recent_trades = [
            t
            for t in closed_trades
            if t.get("close_time_jst") and t["close_time_jst"] >= week_cutoff
        ]
        perf["recent_closed"] = len(recent_trades)
        wins = sum(1 for t in recent_trades if t["pl_pips"] > 0)
        losses = sum(1 for t in recent_trades if t["pl_pips"] < 0)
        perf["wins"] = wins
        perf["losses"] = losses
        perf["win_rate"] = (wins / perf["recent_closed"]) if perf["recent_closed"] else 0.0
        perf["win_rate_percent"] = round(perf["win_rate"] * 100.0, 1)

    if perf.get("daily_pl_pips") is None:
        perf["daily_pl_pips"] = round(
            _sum_if(
                lambda t: t.get("close_time_jst")
                and t["close_time_jst"].date() == today_date
            ),
            2,
        ) if closed_trades else 0.0
    if perf.get("daily_pl_jpy") is None:
        perf["daily_pl_jpy"] = round(
            _sum_jpy(
                lambda t: t.get("close_time_jst")
                and t["close_time_jst"].date() == today_date
            ),
            2,
        ) if closed_trades else 0.0
    if perf.get("weekly_pl_pips") is None:
        perf["weekly_pl_pips"] = round(
            _sum_if(
                lambda t: t.get("close_time_jst")
                and t["close_time_jst"] >= week_cutoff
            ),
            2,
        ) if closed_trades else 0.0
    if perf.get("weekly_pl_jpy") is None:
        perf["weekly_pl_jpy"] = round(
            _sum_jpy(
                lambda t: t.get("close_time_jst")
                and t["close_time_jst"] >= week_cutoff
            ),
            2,
        ) if closed_trades else 0.0

    if metrics_snapshot.get("yesterday") is None:
        yesterday_date = today_date - timedelta(days=1)
        perf["yesterday_pl_pips"] = round(
            _sum_if(
                lambda t: t.get("close_time_jst")
                and t["close_time_jst"].date() == yesterday_date
            ),
            2,
        ) if closed_trades else 0.0
        perf["yesterday_pl_jpy"] = round(
            _sum_jpy(
                lambda t: t.get("close_time_jst")
                and t["close_time_jst"].date() == yesterday_date
            ),
            2,
        ) if closed_trades else 0.0
    if perf.get("total_pips") is None:
        perf["total_pips"] = round(sum(t["pl_pips"] for t in closed_trades), 2)
    if perf.get("total_jpy") is None:
        perf["total_jpy"] = round(sum(t["pl_jpy"] for t in closed_trades), 2)

    perf["daily_pl_eq1l"] = round((perf.get("daily_pl_jpy", 0.0) or 0.0) / 1000.0, 2)
    perf["weekly_pl_eq1l"] = round((perf.get("weekly_pl_jpy", 0.0) or 0.0) / 1000.0, 2)
    perf["total_eq1l"] = round((perf.get("total_jpy", 0.0) or 0.0) / 1000.0, 2)

    if metrics_snapshot.get("daily_change") is None or perf.get("daily_change_pct") is None:
        today_pips = perf.get("daily_pl_pips", 0.0) or 0.0
        yest_pips = perf.get("yesterday_pl_pips", 0.0) or 0.0
        today_jpy = perf.get("daily_pl_jpy", 0.0) or 0.0
        yest_jpy = perf.get("yesterday_pl_jpy", 0.0) or 0.0
        perf["daily_change_pips"] = round(today_pips - yest_pips, 2)
        perf["daily_change_jpy"] = round(today_jpy - yest_jpy, 2)
        equity_val = perf.get("daily_change_equity")
        if equity_val in (None, 0.0):
            equity_val = _load_latest_metric("account.nav")
            equity_source = "nav"
            if equity_val in (None, 0.0):
                equity_val = _load_latest_metric("account.balance")
                equity_source = "balance"
            if equity_val:
                perf["daily_change_equity"] = equity_val
                perf["daily_change_equity_source"] = equity_source
        if equity_val:
            perf["daily_change_pct"] = round(
                (today_jpy - yest_jpy) * 100.0 / float(equity_val), 2
            )
        else:
            perf["daily_change_pct"] = None

    if perf.get("last_trade_at") is None:
        last_trade = metrics_snapshot.get("last_trade_at")
        if not last_trade and closed_trades:
            last_trade = max(closed_trades, key=lambda t: t["close_time"])["close_time"]
        perf["last_trade_at"] = _format_dt(_parse_dt(last_trade)) if last_trade else None

    open_positions = snapshot.get("open_positions") or {}
    open_entries: list[Dict[str, Any]] = []
    open_trades_detail: list[Dict[str, Any]] = []
    total_positions = 0
    total_unrealized_pips = 0.0
    total_unrealized_jpy = 0.0
    total_long_units = 0.0
    total_short_units = 0.0
    for name, info in open_positions.items():
        if name.startswith("__"):
            continue
        trades = info.get("open_trades") or []
        total_positions += len(trades)
        units_val = _safe_float(info.get("units"))
        direction = "Long" if units_val > 0 else "Short" if units_val < 0 else "Flat"
        unrealized_pips = round(_safe_float(info.get("unrealized_pl_pips")), 2)
        unrealized_jpy = round(_safe_float(info.get("unrealized_pl")), 2)
        total_unrealized_pips += unrealized_pips
        total_unrealized_jpy += unrealized_jpy
        long_units = _safe_float(info.get("long_units"))
        short_units = _safe_float(info.get("short_units"))
        total_long_units += max(long_units, 0.0)
        total_short_units += max(short_units, 0.0)
        units_abs = abs(units_val)
        open_entries.append(
            {
                "pocket": name,
                "direction": direction,
                "units": units_val,
                "units_abs": int(round(units_abs)),
                "long_units": int(round(max(long_units, 0.0))),
                "short_units": int(round(max(short_units, 0.0))),
                "long_avg_price": _opt_float(info.get("long_avg_price")),
                "short_avg_price": _opt_float(info.get("short_avg_price")),
                "avg_price": _safe_float(info.get("avg_price")),
                "unrealized_pips": unrealized_pips,
                "unrealized_jpy": unrealized_jpy,
                "trades": len(trades),
            }
        )
        for trade in trades:
            trade_units = _safe_float(trade.get("units"))
            trade_side = trade.get("side") or ("Long" if trade_units > 0 else "Short" if trade_units < 0 else "Flat")
            if isinstance(trade_side, str):
                trade_side = trade_side.capitalize()
            open_dt = _parse_dt(trade.get("open_time"))
            worker = _infer_worker_name(trade)
            strategy = trade.get("strategy_tag") or trade.get("strategy") or "-"
            open_trades_detail.append(
                {
                    "trade_id": trade.get("trade_id") or "-",
                    "pocket": name,
                    "worker": worker or "-",
                    "strategy": strategy,
                    "side": trade_side,
                    "units_abs": int(round(abs(trade_units))),
                    "entry_price": _opt_float(trade.get("price")),
                    "open_label": _format_dt(open_dt) or "-",
                    "open_time": open_dt,
                    "unrealized_pips": round(_safe_float(trade.get("unrealized_pl_pips")), 2),
                    "unrealized_jpy": round(_safe_float(trade.get("unrealized_pl")), 2),
                }
            )
    open_entries.sort(key=lambda row: row["pocket"])
    open_trades_detail.sort(
        key=lambda row: row.get("open_time") or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    if len(open_trades_detail) > int(os.getenv("UI_OPEN_TRADES_LIMIT", "50")):
        open_trades_detail = open_trades_detail[: int(os.getenv("UI_OPEN_TRADES_LIMIT", "50"))]

    net_units = _safe_float((open_positions.get("__net__") or {}).get("units"))
    base["open_summary"] = {
        "pockets": open_entries,
        "total_positions": total_positions,
        "net_units": net_units,
        "long_units": total_long_units,
        "short_units": total_short_units,
        "unrealized_pl_pips": round(total_unrealized_pips, 2),
        "unrealized_pl_jpy": round(total_unrealized_jpy, 2),
        "open_trades": open_trades_detail,
        "meta": open_positions.get("__meta__", {}) if isinstance(open_positions, dict) else {},
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

    signal_dt = _ts_ms_to_dt(metrics_snapshot.get("signals_last_ts_ms"))
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
        strategy = row.get("strategy") or row.get("strategy_tag") or "-"
        worker = row.get("worker")
        if not worker:
            worker = _infer_worker_name(
                {"entry_thesis": row.get("entry_thesis"), "strategy": strategy}
            )
        orders_last.append(
            {
                "ts_label": ts_label,
                "pocket": row.get("pocket") or "-",
                "pocket_label": _pocket_label(row.get("pocket")),
                "strategy": strategy,
                "worker": worker or "-",
                "side": row.get("side") or "-",
                "units": row.get("units"),
                "status": status,
                "status_level": _status_level(status),
                "error_code": row.get("error_code"),
                "error_message": row.get("error_message"),
                "id_short": _shorten(order_id, 22),
                "id_full": str(order_id or ""),
            }
        )
    system["orders_last"] = orders_last

    signals_recent = []
    for row in metrics_snapshot.get("signals_recent") or []:
        ts_label = _format_dt(_ts_ms_to_dt(row.get("ts_ms"))) or str(row.get("ts_ms") or "-")
        signal_id = row.get("client_order_id")
        worker = _infer_worker_name(row)
        signals_recent.append(
            {
                "ts_label": ts_label,
                "pocket": row.get("pocket") or "-",
                "pocket_label": _pocket_label(row.get("pocket")),
                "strategy": row.get("strategy") or "-",
                "worker": worker or "-",
                "action": row.get("action") or "-",
                "confidence": row.get("confidence"),
                "units": row.get("proposed_units"),
                "id_short": _shorten(signal_id, 22),
                "id_full": str(signal_id or ""),
            }
        )
    system["signals_recent"] = signals_recent

    orders_errors = []
    for row in metrics_snapshot.get("orders_errors_recent") or []:
        ts_label = _format_dt(_parse_dt(row.get("ts"))) or str(row.get("ts") or "-")
        order_id = row.get("client_order_id")
        status = str(row.get("status") or "-")
        strategy = row.get("strategy") or row.get("strategy_tag") or "-"
        worker = row.get("worker")
        if not worker:
            worker = _infer_worker_name(
                {"entry_thesis": row.get("entry_thesis"), "strategy": strategy}
            )
        orders_errors.append(
            {
                "ts_label": ts_label,
                "pocket": row.get("pocket") or "-",
                "pocket_label": _pocket_label(row.get("pocket")),
                "strategy": strategy,
                "worker": worker or "-",
                "side": row.get("side") or "-",
                "units": row.get("units"),
                "status": status,
                "status_level": _status_level(status),
                "error_code": row.get("error_code") or "-",
                "error_message": row.get("error_message") or "-",
                "id_short": _shorten(order_id, 22),
                "id_full": str(order_id or ""),
            }
        )
    system["orders_errors"] = orders_errors

    log_errors = []
    for row in metrics_snapshot.get("log_errors_recent") or []:
        log_errors.append(
            {
                "ts": row.get("ts"),
                "level": row.get("level"),
                "message": row.get("message"),
                "source": row.get("source"),
            }
        )
    system["log_errors"] = log_errors

    health_snapshot = metrics_snapshot.get("health_snapshot")
    if isinstance(health_snapshot, dict):
        health = base["health"]
        health["hostname"] = health_snapshot.get("hostname")
        health["deploy_id"] = health_snapshot.get("deploy_id")
        health["git_rev"] = health_snapshot.get("git_rev")
        health["uptime_sec"] = health_snapshot.get("uptime_sec")
        health["uptime_label"] = _format_uptime(_safe_float(health_snapshot.get("uptime_sec")))
        health["service_active"] = health_snapshot.get("service_active") or {}
        service_info_rows = []
        for unit, info in (health_snapshot.get("service_info") or {}).items():
            if not isinstance(info, dict):
                continue
            service_info_rows.append(
                {
                    "unit": unit,
                    "active_state": info.get("ActiveState"),
                    "sub_state": info.get("SubState"),
                    "result": info.get("Result"),
                    "n_restarts": info.get("NRestarts"),
                    "active_enter_ts": info.get("ActiveEnterTimestamp"),
                    "exec_start_ts": info.get("ExecMainStartTimestamp"),
                    "exec_exit_ts": info.get("ExecMainExitTimestamp"),
                    "exec_status": info.get("ExecMainStatus"),
                }
            )
        service_info_rows.sort(key=lambda row: row["unit"])
        health["service_info"] = service_info_rows

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
                "worker": tr.get("worker") or "-",
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
                "worker": tr.get("worker") or "-",
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
                "worker": tr.get("worker") or "-",
                "pl_pips": round(tr["pl_pips"], 2),
                "pl_jpy": round(tr["pl_jpy"], 2),
                "closed_at": tr["close_label"],
                "kind": kind,
            }
        )

    base["highlights"] = highlights
    base["highlights_top"] = highlights
    base["highlights_recent"] = latest_highlights

    profit_tables = metrics_snapshot.get("profit_tables")
    if isinstance(profit_tables, dict):
        base["profit_tables"] = profit_tables

    cashflow = metrics_snapshot.get("cashflow")
    if isinstance(cashflow, dict):
        base["cashflow"] = cashflow

    ytd_summary = metrics_snapshot.get("ytd_summary")
    if isinstance(ytd_summary, dict):
        base["ytd_summary"] = ytd_summary

    chart_data = metrics_snapshot.get("chart_data")
    if isinstance(chart_data, dict) and chart_data:
        base["charts"] = chart_data
        try:
            payload = json.dumps(chart_data, ensure_ascii=True, separators=(",", ":"))
            base["charts_json"] = payload.replace("<", "\\u003c")
        except Exception:
            base["charts_json"] = "{}"

    gen_dt = _parse_dt(snapshot.get("generated_at"))
    snapshot_mode = snapshot.get("snapshot_mode")
    snapshot_source = snapshot.get("snapshot_source")
    age_sec = None
    stale = False
    if gen_dt:
        age_sec = int(max(0, (now - gen_dt).total_seconds()))
        stale = age_sec > max(60, _AUTO_REFRESH_SEC * 2)
    base["snapshot"] = {
        "mode": snapshot_mode,
        "source": snapshot_source,
        "age_sec": age_sec,
        "stale": stale,
    }
    base["generated_at"] = snapshot.get("generated_at")
    base["generated_label"] = _format_dt(gen_dt) or snapshot.get("generated_at")
    base["available"] = True
    base["error"] = None
    return base


def _load_dashboard_data() -> Dict[str, Any]:
    base = _dashboard_defaults()
    candidates: list[tuple[str, dict]] = []

    remote_snapshot = _fetch_remote_snapshot("ui_snapshot_lite_url")
    if not remote_snapshot:
        remote_snapshot = _fetch_remote_snapshot()
    if remote_snapshot:
        candidates.append(("remote", remote_snapshot))

    gcs_snapshot, gcs_error = _fetch_gcs_snapshot()
    if gcs_snapshot:
        candidates.append(("gcs", gcs_snapshot))

    local_snapshot = _build_local_snapshot()
    if local_snapshot:
        candidates.append(("local", local_snapshot))

    picked = _pick_latest_snapshot(candidates)
    if picked:
        source, snapshot = picked
        base = _summarise_snapshot(snapshot)
        base.setdefault("snapshot", {})["source"] = source
        return base

    if gcs_error:
        base["error"] = gcs_error
    else:
        base["error"] = "スナップショットを取得できませんでした"
    return base


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


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


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


@app.get("/api/snapshot-lite")
def api_snapshot_lite(
    token: Optional[str] = None,
    x_qr_token: Optional[str] = Header(default=None, alias="X-QR-Token"),
    authorization: Optional[str] = Header(default=None),
):
    required = _get_secret_optional("ui_snapshot_token")
    provided = token or x_qr_token or _extract_bearer(authorization)
    if required and provided != required:
        raise HTTPException(status_code=401, detail="Unauthorized")
    snapshot = _cached_lite_snapshot()
    return JSONResponse(snapshot, headers={"Cache-Control": "no-store"})


@app.get("/dashboard")
def dashboard(request: Request):
    dashboard_data = _load_dashboard_data()
    ops_notice = request.query_params.get("ops_notice")
    ops_error = request.query_params.get("ops_error")
    ops_enabled = _ops_required_token() is not None
    response = templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "dashboard": dashboard_data,
            "active_tab": "dashboard",
            "ops_notice": ops_notice,
            "ops_error": ops_error,
            "ops_enabled": ops_enabled,
        },
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.post("/api/ops/control")
def api_ops_control(
    action: str = Form(...),
    confirm: str = Form(""),
    ops_token: Optional[str] = Form(None),
    x_qr_token: Optional[str] = Header(default=None, alias="X-QR-Token"),
    authorization: Optional[str] = Header(default=None),
):
    result = _handle_ops_action(
        action,
        confirm,
        ops_token,
        x_qr_token,
        authorization,
        allow_proxy=False,
    )
    return JSONResponse(result)


@app.post("/ops/control")
def ops_control(
    request: Request,
    action: str = Form(...),
    confirm: str = Form(""),
    ops_token: Optional[str] = Form(None),
    x_qr_token: Optional[str] = Header(default=None, alias="X-QR-Token"),
    authorization: Optional[str] = Header(default=None),
):
    accept = request.headers.get("accept", "")
    try:
        result = _handle_ops_action(
            action,
            confirm,
            ops_token,
            x_qr_token,
            authorization,
            allow_proxy=True,
        )
    except HTTPException as exc:
        if "application/json" in accept:
            raise
        query = urlencode({"tab": "ops", "ops_error": str(exc.detail)})
        return RedirectResponse(url=f"/dashboard?{query}", status_code=303)
    if "application/json" in accept:
        return JSONResponse(result)
    if result.get("ok"):
        notice = "停止" if action == "stop" else "再開"
        summary = f"{notice}しました（成功 {result.get('ok_count', 0)} / {result.get('total', 0)}）"
        query = urlencode({"tab": "ops", "ops_notice": summary})
        return RedirectResponse(url=f"/dashboard?{query}", status_code=303)
    error = result.get("error") or "操作に失敗しました"
    query = urlencode({"tab": "ops", "ops_error": error})
    return RedirectResponse(url=f"/dashboard?{query}", status_code=303)


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
