from __future__ import annotations

import json
import os
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from autotune.database import (
    USE_BIGQUERY,
    dump_dict,
    get_run,
    get_stats,
    get_settings,
    list_runs,
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

app = FastAPI(title="QuantRabbit Console")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


def _excursion_base_dir() -> Path:
    """エクスカージョン・レポートのベースディレクトリを解決する。
    優先順: secret(exursion_reports_dir) -> REPO_ROOT/logs/reports/excursion -> /home/tossaki/QuantRabbit/logs/reports/excursion
    """
    # Secret 優先
    try:
        secret_path = get_secret("excursion_reports_dir")
        if secret_path:
            p = Path(secret_path)
            if p.exists():
                return p
    except Exception:
        pass
    # リポ内ログ
    local = REPO_ROOT / "logs" / "reports" / "excursion"
    if local.exists():
        return local
    # VM 既定パス
    vm_default = Path("/home/tossaki/QuantRabbit/logs/reports/excursion")
    return vm_default


def _read_text(path: Path, limit_bytes: int = 512_000) -> str:
    try:
        data = path.read_bytes()
        if len(data) > limit_bytes:
            data = data[-limit_bytes:]
        return data.decode("utf-8", errors="replace")
    except Exception as exc:
        return f"[read_error] {exc}"


def _normalize_numbers(payload: Optional[dict]) -> dict:
    if not payload:
        return {}
    converted = {}
    for key, value in payload.items():
        if isinstance(value, Decimal):
            converted[key] = int(value) if value == int(value) else float(value)
        else:
            converted[key] = value
    return converted


def _normalize_settings(settings: Optional[dict]) -> dict:
    base = settings or {"enabled": True}
    result = dict(base)
    result["enabled"] = bool(base.get("enabled", True))
    return result


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
        },
        "open_summary": {
            "total_positions": 0,
            "net_units": 0.0,
            "unrealized_pl_pips": 0.0,
            "unrealized_pl_jpy": 0.0,
            "pockets": [],
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


@app.get("/")
def root_redirect():
    return RedirectResponse(url="/dashboard", status_code=307)


@app.get("/dashboard")
def dashboard(request: Request):
    dashboard_data = _load_dashboard_data()
    # Excursion latest and recent list for inline dashboard panel
    base = _excursion_base_dir()
    hourly_dir = base / "hourly"
    latest_path = base / "latest.txt"
    excursion_hours: list[dict] = []
    if hourly_dir.exists():
        files = sorted(hourly_dir.glob("*.txt"), key=lambda p: p.name, reverse=True)[:12]
        for p in files:
            try:
                size = p.stat().st_size
            except Exception:
                size = 0
            excursion_hours.append({"name": p.name, "size": size})
    excursion_content = ""
    if latest_path.exists():
        excursion_content = _read_text(latest_path, limit_bytes=256_000)
    elif excursion_hours:
        excursion_content = _read_text(hourly_dir / excursion_hours[0]["name"], limit_bytes=256_000)
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "dashboard": dashboard_data,
            "active_tab": "dashboard",
            "excursion_hours": excursion_hours,
            "excursion_content": excursion_content,
        },
    )


@app.get("/excursion")
def excursion_report(request: Request, file: Optional[str] = None):
    base = _excursion_base_dir()
    hourly_dir = base / "hourly"
    latest_path = base / "latest.txt"
    # 一覧（最新順）
    hours: list[dict] = []
    if hourly_dir.exists():
        for p in sorted(hourly_dir.glob("*.txt"), key=lambda x: x.name, reverse=True):
            hours.append({
                "name": p.name,
                "size": p.stat().st_size if p.exists() else 0,
            })
    selected_name = (file or "").strip()
    content = ""
    if selected_name:
        target = hourly_dir / Path(selected_name).name
        if target.exists():
            content = _read_text(target)
        else:
            content = f"[not_found] {target}"
    else:
        if latest_path.exists():
            content = _read_text(latest_path)
        else:
            # フォールバック: 最新ファイルを選択
            if hours:
                target = hourly_dir / hours[0]["name"]
                content = _read_text(target)
                selected_name = hours[0]["name"]
            else:
                content = "レポートが見つかりません。ジョブが稼働しているか確認してください。"
    return templates.TemplateResponse(
        "excursion.html",
        {
            "request": request,
            "active_tab": "excursion",
            "hours": hours,
            "selected": selected_name,
            "content": content,
        },
    )


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
    stats = _normalize_numbers(get_stats())
    settings = _normalize_settings(get_settings())

    # fetch runs (larger limit then filter/sort/paginate in memory)
    raw_runs = [dump_dict(row) for row in list_runs(status=None if status == "all" else status, limit=500)]

    # normalize fields for sorting/filtering
    def derive_metrics(run: dict) -> dict:
        train = run.get("train") or {}
        valid = run.get("valid") or {}
        base = valid if valid else train
        pf = base.get("profit_factor")
        trades = base.get("trades")
        dd = base.get("max_dd_pips")
        wr = base.get("win_rate")
        run["_pf"] = float(pf) if pf is not None else 0.0
        run["_trades"] = int(trades) if trades is not None else 0
        run["_dd"] = float(dd) if dd is not None else 0.0
        run["_wr"] = float(wr) if wr is not None else 0.0
        run["_score"] = float(run.get("score") or 0.0)
        run["_updated"] = run.get("updated_at") or run.get("created_at") or ""
        run["_created"] = run.get("created_at") or ""
        run["_summary"] = _build_summary(run)
        return run

    items = [derive_metrics(r) for r in raw_runs]

    # free-text filter
    q_norm = (q or "").strip().lower()
    if q_norm:
        items = [r for r in items if q_norm in str(r.get("strategy", "")).lower() or q_norm in str(r.get("run_id", "")).lower()]

    # sort
    keymap = {
        "score": "_score",
        "pf": "_pf",
        "trades": "_trades",
        "dd": "_dd",
        "wr": "_wr",
        "updated": "_updated",
        "created": "_created",
    }
    sort_key = keymap.get((sort or "").lower(), "_updated")
    reverse = (order or "desc").lower() != "asc"
    items.sort(key=lambda r: r.get(sort_key) or 0, reverse=reverse)

    # pagination
    try:
        page = max(1, int(page))
    except Exception:
        page = 1
    try:
        page_size = max(10, min(200, int(page_size)))
    except Exception:
        page_size = 50
    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = items[start:end]
    pages = (total + page_size - 1) // page_size
    return templates.TemplateResponse(
        "autotune.html",
        {
            "request": request,
            "stats": stats,
            "using_bigquery": USE_BIGQUERY,
            "settings": settings,
            "active_tab": "autotune",
            "items": page_items,
            "filters": {
                "status": status,
                "sort": sort,
                "order": order,
                "q": q,
                "page": page,
                "page_size": page_size,
                "pages": pages,
                "total": total,
            },
        },
    )


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
