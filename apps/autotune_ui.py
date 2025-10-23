from __future__ import annotations

import json
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
    list_runs,
    update_status,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = REPO_ROOT / "templates" / "autotune"

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


@app.get("/")
def dashboard(request: Request):
    pending = [dump_dict(row) for row in list_runs(status="pending", limit=50)]
    recent = [dump_dict(row) for row in list_runs(status=None, limit=10)]
    stats = _normalize_numbers(get_stats())
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "pending": pending,
            "recent": recent,
            "stats": stats,
            "using_bigquery": USE_BIGQUERY,
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
    if not get_run(None, run_id, strategy):
        raise HTTPException(status_code=404, detail="Run not found")
    update_status(None, run_id, strategy, status, reviewer or None, comment or None)
    return RedirectResponse(url=f"/runs/{run_id}/{strategy}", status_code=303)
