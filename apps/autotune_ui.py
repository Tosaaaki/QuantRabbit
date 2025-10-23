from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from autotune.database import (
    DEFAULT_DB_PATH,
    dump_dict,
    get_connection,
    list_runs,
    get_run,
    update_status,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = REPO_ROOT / "templates" / "autotune"

app = FastAPI(title="QuantRabbit Autotune Review")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


@app.get("/")
def dashboard(request: Request):
    conn = get_connection(DEFAULT_DB_PATH)
    pending = [dump_dict(row) for row in list_runs(conn, status="pending", limit=50)]
    recent = [dump_dict(row) for row in list_runs(conn, status=None, limit=10)]
    conn.close()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "pending": pending,
            "recent": recent,
        },
    )


@app.get("/runs/{run_id}/{strategy}")
def run_detail(request: Request, run_id: str, strategy: str):
    conn = get_connection(DEFAULT_DB_PATH)
    row = get_run(conn, run_id, strategy)
    conn.close()
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
    conn = get_connection(DEFAULT_DB_PATH)
    if not get_run(conn, run_id, strategy):
        conn.close()
        raise HTTPException(status_code=404, detail="Run not found")
    update_status(conn, run_id, strategy, status, reviewer or None, comment or None)
    conn.close()
    return RedirectResponse(url=f"/runs/{run_id}/{strategy}", status_code=303)
