from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Optional

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

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = REPO_ROOT / "templates" / "autotune"
CONFIG_PATH = REPO_ROOT / "configs" / "scalp_active_params.json"

app = FastAPI(title="QuantRabbit Autotune Review")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


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
    score = run.get("score")
    if score is not None:
        lines.append(f"スコアは {score:.3f} でした。")

    train = run.get("train") or {}
    valid = run.get("valid") or {}
    def describe(metrics: dict, label: str) -> None:
        if not metrics:
            return
        pf = metrics.get("profit_factor")
        trades = metrics.get("trades")
        dd = metrics.get("max_dd_pips")
        profit = metrics.get("profit_pips")
        parts = []
        if pf is not None:
            if pf >= 1.5:
                parts.append(f"PF {pf:.2f} と十分な利益率です")
            elif pf >= 1.1:
                parts.append(f"PF {pf:.2f} と安定しています")
            else:
                parts.append(f"PF {pf:.2f} と低いため再調整が必要です")
        if trades is not None and trades < 10:
            parts.append(f"検証件数が {trades} 件と少ない点に注意してください")
        if dd is not None and dd > 12:
            parts.append(f"最大ドローダウンは {dd:.1f} pips と大きめです")
        if profit is not None:
            if profit >= 0:
                parts.append(f"総利益は {profit:.1f} pips でした")
            else:
                parts.append(f"総損失は {abs(profit):.1f} pips でした")
        if parts:
            lines.append(f"{label}: " + "、".join(parts) + "。")

    describe(train, "学習期間")
    describe(valid, "検証期間")

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
            lines.append("提案パラメータ: " + "、".join(extras) + "。")

    if not lines:
        lines.append("メトリクス情報が不足しているため、詳細評価はありません。")
    return "\n".join(lines)


@app.get("/")
def dashboard(request: Request):
    pending = [dump_dict(row) for row in list_runs(status="pending", limit=50)]
    recent = [dump_dict(row) for row in list_runs(status=None, limit=10)]
    stats = _normalize_numbers(get_stats())
    settings = _normalize_settings(get_settings())
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "pending": pending,
            "recent": recent,
            "stats": stats,
            "using_bigquery": USE_BIGQUERY,
            "settings": settings,
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
