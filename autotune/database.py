from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "logs" / "autotune.db"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tuning_runs (
            run_id TEXT NOT NULL,
            strategy TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            score REAL,
            params_json TEXT NOT NULL,
            train_json TEXT NOT NULL,
            valid_json TEXT NOT NULL,
            source_file TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            reviewer TEXT,
            comment TEXT,
            PRIMARY KEY (run_id, strategy)
        )
        """
    )
    conn.commit()


def record_run(
    conn: sqlite3.Connection,
    run_id: str,
    strategy: str,
    params: Dict[str, Any],
    train: Dict[str, Any],
    valid: Dict[str, Any],
    score: float,
    source_file: Optional[str] = None,
) -> None:
    now = _utc_now()
    payload = {
        "run_id": run_id,
        "strategy": strategy,
        "status": "pending",
        "score": score,
        "params_json": json.dumps(params, ensure_ascii=False),
        "train_json": json.dumps(train, ensure_ascii=False),
        "valid_json": json.dumps(valid, ensure_ascii=False),
        "source_file": source_file,
        "created_at": now,
        "updated_at": now,
    }
    conn.execute(
        """
        INSERT INTO tuning_runs(
            run_id, strategy, status, score,
            params_json, train_json, valid_json,
            source_file, created_at, updated_at
        ) VALUES (
            :run_id, :strategy, :status, :score,
            :params_json, :train_json, :valid_json,
            :source_file, :created_at, :updated_at
        )
        ON CONFLICT(run_id, strategy) DO UPDATE SET
            status='pending',
            score=excluded.score,
            params_json=excluded.params_json,
            train_json=excluded.train_json,
            valid_json=excluded.valid_json,
            source_file=excluded.source_file,
            updated_at=excluded.updated_at,
            reviewer=NULL,
            comment=NULL
        """,
        payload,
    )
    conn.commit()


def list_runs(
    conn: sqlite3.Connection, status: Optional[str] = None, limit: int = 100
) -> List[Dict[str, Any]]:
    sql = "SELECT * FROM tuning_runs"
    params: Tuple[Any, ...] = ()
    if status:
        sql += " WHERE status = ?"
        params = (status,)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params = params + (limit,)
    rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def get_run(
    conn: sqlite3.Connection, run_id: str, strategy: str
) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM tuning_runs WHERE run_id=? AND strategy=?",
        (run_id, strategy),
    ).fetchone()
    return dict(row) if row else None


def update_status(
    conn: sqlite3.Connection,
    run_id: str,
    strategy: str,
    status: str,
    reviewer: Optional[str] = None,
    comment: Optional[str] = None,
) -> None:
    if status not in {"pending", "approved", "rejected"}:
        raise ValueError("Invalid status")
    conn.execute(
        """
        UPDATE tuning_runs
        SET status=?, reviewer=?, comment=?, updated_at=?
        WHERE run_id=? AND strategy=?
        """,
        (status, reviewer, comment, _utc_now(), run_id, strategy),
    )
    conn.commit()


def dump_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    for key in ("params_json", "train_json", "valid_json"):
        if key in out and out[key]:
            out[key[:-5]] = json.loads(out[key])
            del out[key]
    return out
