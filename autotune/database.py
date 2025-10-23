from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from google.cloud import bigquery  # type: ignore
except Exception:  # pragma: no cover - bigquery optional
    bigquery = None

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "logs" / "autotune.db"
AUTOTUNE_BQ_TABLE = os.getenv("AUTOTUNE_BQ_TABLE", "")
USE_BIGQUERY = bool(AUTOTUNE_BQ_TABLE) and bigquery is not None


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


def _get_sqlite_rows(
    conn: sqlite3.Connection, status: Optional[str], limit: int
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


def _parse_table(table: str) -> Tuple[str, str, str]:
    parts = table.split(".")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        project = (
            os.getenv("BQ_PROJECT")
            or os.getenv("GOOGLE_CLOUD_PROJECT")
            or _get_bq_client().project
        )
        return project, parts[0], parts[1]
    raise ValueError(
        f"AUTOTUNE_BQ_TABLE must be <project>.<dataset>.<table> or <dataset>.<table>, got: {table}"
    )


@lru_cache()
def _get_bq_client() -> "bigquery.Client":
    if bigquery is None:  # pragma: no cover - safety guard
        raise RuntimeError("google-cloud-bigquery is not installed")
    return bigquery.Client()


def record_run_bigquery(
    run_id: str,
    strategy: str,
    params: Dict[str, Any],
    train: Dict[str, Any],
    valid: Dict[str, Any],
    score: float,
    source_file: Optional[str] = None,
    table_override: Optional[str] = None,
) -> None:
    table_id = table_override or AUTOTUNE_BQ_TABLE
    if not table_id:
        return
    client = _get_bq_client()
    project, dataset, table = _parse_table(table_id)
    table_fqn = f"{project}.{dataset}.{table}"
    now = _utc_now()
    now_dt = datetime.fromisoformat(now)
    params_json = json.dumps(params, ensure_ascii=False)
    train_json = json.dumps(train, ensure_ascii=False)
    valid_json = json.dumps(valid, ensure_ascii=False)
    query = f"""
    MERGE `{table_fqn}` T
    USING (
        SELECT
          @run_id AS run_id,
          @strategy AS strategy,
          @status AS status,
          @score AS score,
          @params_json AS params_json,
          @train_json AS train_json,
          @valid_json AS valid_json,
          @source_file AS source_file,
          @created_at AS created_at,
          @updated_at AS updated_at
    ) S
    ON T.run_id = S.run_id AND T.strategy = S.strategy
    WHEN MATCHED THEN
      UPDATE SET
        status = 'pending',
        score = S.score,
        params_json = S.params_json,
        train_json = S.train_json,
        valid_json = S.valid_json,
        source_file = S.source_file,
        updated_at = S.updated_at,
        reviewer = NULL,
        comment = NULL
    WHEN NOT MATCHED THEN
      INSERT(run_id, strategy, status, score, params_json, train_json, valid_json, source_file, created_at, updated_at)
      VALUES(S.run_id, S.strategy, S.status, S.score, S.params_json, S.train_json, S.valid_json, S.source_file, S.created_at, S.updated_at)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
            bigquery.ScalarQueryParameter("strategy", "STRING", strategy),
            bigquery.ScalarQueryParameter("status", "STRING", "pending"),
            bigquery.ScalarQueryParameter("score", "FLOAT64", score),
            bigquery.ScalarQueryParameter("params_json", "STRING", params_json),
            bigquery.ScalarQueryParameter("train_json", "STRING", train_json),
            bigquery.ScalarQueryParameter("valid_json", "STRING", valid_json),
            bigquery.ScalarQueryParameter("source_file", "STRING", source_file),
            bigquery.ScalarQueryParameter("created_at", "TIMESTAMP", now_dt),
            bigquery.ScalarQueryParameter("updated_at", "TIMESTAMP", now_dt),
        ]
    )
    client.query(query, job_config=job_config).result()


def list_runs_bigquery(
    status: Optional[str] = None,
    limit: int = 100,
    table_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    table_id = table_override or AUTOTUNE_BQ_TABLE
    if not table_id:
        return []
    client = _get_bq_client()
    project, dataset, table = _parse_table(table_id)
    table_fqn = f"{project}.{dataset}.{table}"
    query = f"""
    SELECT run_id, strategy, status, score, params_json, train_json, valid_json,
           source_file, created_at, updated_at, reviewer, comment
    FROM `{table_fqn}`
    """
    params_cfg: List["bigquery.ScalarQueryParameter"] = []
    if status:
        query += " WHERE status = @status"
        params_cfg.append(bigquery.ScalarQueryParameter("status", "STRING", status))
    query += " ORDER BY created_at DESC LIMIT @limit"
    params_cfg.append(bigquery.ScalarQueryParameter("limit", "INT64", limit))
    job_config = bigquery.QueryJobConfig(query_parameters=params_cfg)
    result = client.query(query, job_config=job_config).result()
    return [dict(row) for row in result]


def get_run_bigquery(
    run_id: str, strategy: str, table_override: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    table_id = table_override or AUTOTUNE_BQ_TABLE
    if not table_id:
        return None
    client = _get_bq_client()
    project, dataset, table = _parse_table(table_id)
    table_fqn = f"{project}.{dataset}.{table}"
    query = f"""
    SELECT run_id, strategy, status, score, params_json, train_json, valid_json,
           source_file, created_at, updated_at, reviewer, comment
    FROM `{table_fqn}`
    WHERE run_id = @run_id AND strategy = @strategy
    LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
            bigquery.ScalarQueryParameter("strategy", "STRING", strategy),
        ]
    )
    rows = list(client.query(query, job_config=job_config).result())
    return dict(rows[0]) if rows else None


def update_status_bigquery(
    run_id: str,
    strategy: str,
    status: str,
    reviewer: Optional[str],
    comment: Optional[str],
    table_override: Optional[str] = None,
) -> None:
    table_id = table_override or AUTOTUNE_BQ_TABLE
    if not table_id:
        return
    if status not in {"pending", "approved", "rejected"}:
        raise ValueError("Invalid status")
    client = _get_bq_client()
    project, dataset, table = _parse_table(table_id)
    table_fqn = f"{project}.{dataset}.{table}"
    query = f"""
    UPDATE `{table_fqn}`
    SET status = @status,
        reviewer = @reviewer,
        comment = @comment,
        updated_at = @updated_at
    WHERE run_id = @run_id AND strategy = @strategy
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("status", "STRING", status),
            bigquery.ScalarQueryParameter("reviewer", "STRING", reviewer),
            bigquery.ScalarQueryParameter("comment", "STRING", comment),
            bigquery.ScalarQueryParameter("updated_at", "STRING", _utc_now()),
            bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
            bigquery.ScalarQueryParameter("strategy", "STRING", strategy),
        ]
    )
    client.query(query, job_config=job_config).result()


def get_stats_bigquery(table_override: Optional[str] = None) -> Dict[str, Any]:
    table_id = table_override or AUTOTUNE_BQ_TABLE
    if not table_id:
        return {}
    client = _get_bq_client()
    project, dataset, table = _parse_table(table_id)
    table_fqn = f"{project}.{dataset}.{table}"
    query = f"""
    SELECT
      COUNT(*) AS total,
      COUNTIF(status='pending') AS pending,
      COUNTIF(status='approved') AS approved,
      COUNTIF(status='rejected') AS rejected,
      MAX(IF(status='approved', updated_at, NULL)) AS last_approved_at
    FROM `{table_fqn}`
    """
    row = list(client.query(query).result())[0]
    return dict(row)


def list_runs(
    conn: Optional[sqlite3.Connection] = None,
    status: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    if USE_BIGQUERY and conn is None:
        return list_runs_bigquery(status=status, limit=limit)
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True
    rows = _get_sqlite_rows(conn, status, limit)
    if close_after:
        conn.close()
    return rows


def get_run(
    conn: Optional[sqlite3.Connection],
    run_id: str,
    strategy: str,
) -> Optional[Dict[str, Any]]:
    if USE_BIGQUERY and conn is None:
        return get_run_bigquery(run_id, strategy)
    if conn is None:
        conn = get_connection()
        close_after = True
    else:
        close_after = False
    row = conn.execute(
        "SELECT * FROM tuning_runs WHERE run_id=? AND strategy=?",
        (run_id, strategy),
    ).fetchone()
    if close_after:
        conn.close()
    return dict(row) if row else None


def update_status(
    conn: Optional[sqlite3.Connection],
    run_id: str,
    strategy: str,
    status: str,
    reviewer: Optional[str] = None,
    comment: Optional[str] = None,
) -> None:
    if USE_BIGQUERY and conn is None:
        update_status_bigquery(run_id, strategy, status, reviewer, comment)
        return
    if status not in {"pending", "approved", "rejected"}:
        raise ValueError("Invalid status")
    if conn is None:
        conn = get_connection()
        close_after = True
    else:
        close_after = False
    conn.execute(
        """
        UPDATE tuning_runs
        SET status=?, reviewer=?, comment=?, updated_at=?
        WHERE run_id=? AND strategy=?
        """,
        (status, reviewer, comment, _utc_now(), run_id, strategy),
    )
    conn.commit()
    if close_after:
        conn.close()


def get_stats(conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
    if USE_BIGQUERY and conn is None:
        return get_stats_bigquery()
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True
    row = conn.execute(
        """
        SELECT
          COUNT(*) AS total,
          SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS pending,
          SUM(CASE WHEN status='approved' THEN 1 ELSE 0 END) AS approved,
          SUM(CASE WHEN status='rejected' THEN 1 ELSE 0 END) AS rejected
        FROM tuning_runs
        """
    ).fetchone()
    stats = dict(row)
    last_row = conn.execute(
        "SELECT updated_at FROM tuning_runs WHERE status='approved' ORDER BY updated_at DESC LIMIT 1"
    ).fetchone()
    stats["last_approved_at"] = last_row["updated_at"] if last_row else None
    if close_after:
        conn.close()
    return stats


def dump_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    for key, value in list(out.items()):
        if isinstance(value, Decimal):
            out[key] = int(value) if value == int(value) else float(value)
        elif isinstance(value, datetime):
            out[key] = value.isoformat()
    for key in ("params_json", "train_json", "valid_json"):
        if key in out and out[key]:
            out[key[:-5]] = json.loads(out[key])
            del out[key]
    return out


__all__ = [
    "AUTOTUNE_BQ_TABLE",
    "USE_BIGQUERY",
    "DEFAULT_DB_PATH",
    "get_connection",
    "ensure_schema",
    "record_run",
    "record_run_bigquery",
    "list_runs",
    "get_run",
    "update_status",
    "get_stats",
    "dump_dict",
]
