from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from google.api_core import exceptions as gexc
from google.cloud import bigquery

try:
    from utils.secrets import get_secret
except Exception:  # pragma: no cover - optional secret manager integration
    get_secret = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "logs" / "autotune.db"

def _load_autotune_env(name: str) -> str:
    """Fetch AUTOTUNE_* settings with fallback to env.toml / Secret Manager."""
    value = os.getenv(name, "").strip()
    if value:
        return value
    if get_secret is None:
        return ""
    for key in (name, name.lower()):
        try:
            candidate = get_secret(key)
        except Exception:
            continue
        if candidate:
            return str(candidate).strip()
    return ""


AUTOTUNE_BQ_TABLE = _load_autotune_env("AUTOTUNE_BQ_TABLE")
AUTOTUNE_BQ_SETTINGS_TABLE = _load_autotune_env("AUTOTUNE_BQ_SETTINGS_TABLE")
USE_BIGQUERY = True


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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS autotune_settings (
            id TEXT PRIMARY KEY,
            enabled INTEGER NOT NULL,
            updated_at TEXT NOT NULL,
            updated_by TEXT
        )
        """
    )
    # ensure default row
    cur = conn.execute("SELECT COUNT(*) FROM autotune_settings WHERE id='default'")
    if cur.fetchone()[0] == 0:
        conn.execute(
            "INSERT INTO autotune_settings(id, enabled, updated_at) VALUES('default', 1, ?)",
            (_utc_now(),),
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
        f"BigQuery table must be <project>.<dataset>.<table> or <dataset>.<table>, got: {table}"
    )


@lru_cache()
def _get_bq_client() -> "bigquery.Client":
    return bigquery.Client()


def _require_autotune_table() -> str:
    table_id = AUTOTUNE_BQ_TABLE.strip()
    if not table_id:
        raise RuntimeError("AUTOTUNE_BQ_TABLE is required (SQLite fallback disabled).")
    return table_id


def _ensure_dataset(client: "bigquery.Client", project: str, dataset: str) -> None:
    dataset_ref = bigquery.DatasetReference(project, dataset)
    try:
        client.get_dataset(dataset_ref)
    except gexc.NotFound:
        ds = bigquery.Dataset(dataset_ref)
        ds.location = os.getenv("BQ_LOCATION", "US")
        client.create_dataset(ds, exists_ok=True)


def _ensure_run_table(client: "bigquery.Client", table_id: str) -> None:
    project, dataset, table = _parse_table(table_id)
    _ensure_dataset(client, project, dataset)
    table_ref = f"{project}.{dataset}.{table}"
    schema = [
        bigquery.SchemaField("run_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("strategy", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("status", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("score", "FLOAT64"),
        bigquery.SchemaField("params_json", "STRING"),
        bigquery.SchemaField("train_json", "STRING"),
        bigquery.SchemaField("valid_json", "STRING"),
        bigquery.SchemaField("source_file", "STRING"),
        bigquery.SchemaField("created_at", "TIMESTAMP"),
        bigquery.SchemaField("updated_at", "TIMESTAMP"),
        bigquery.SchemaField("reviewer", "STRING"),
        bigquery.SchemaField("comment", "STRING"),
    ]
    try:
        client.get_table(table_ref)
    except gexc.NotFound:
        client.create_table(bigquery.Table(table_ref, schema=schema))


def _ensure_settings_table(client: "bigquery.Client", table_id: str) -> None:
    project, dataset, table = _parse_table(table_id)
    _ensure_dataset(client, project, dataset)
    table_ref = f"{project}.{dataset}.{table}"
    schema = [
        bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("enabled", "BOOL", mode="REQUIRED"),
        bigquery.SchemaField("updated_at", "TIMESTAMP"),
        bigquery.SchemaField("updated_by", "STRING"),
    ]
    try:
        client.get_table(table_ref)
    except gexc.NotFound:
        client.create_table(bigquery.Table(table_ref, schema=schema))


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
    table_id = table_override or _require_autotune_table()
    client = _get_bq_client()
    _ensure_run_table(client, table_id)
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
    table_id = table_override or _require_autotune_table()
    client = _get_bq_client()
    _ensure_run_table(client, table_id)
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
    table_id = table_override or _require_autotune_table()
    client = _get_bq_client()
    _ensure_run_table(client, table_id)
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
    table_id = table_override or _require_autotune_table()
    if status not in {"pending", "approved", "rejected"}:
        raise ValueError("Invalid status")
    client = _get_bq_client()
    _ensure_run_table(client, table_id)
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
    table_id = table_override or _require_autotune_table()
    client = _get_bq_client()
    _ensure_run_table(client, table_id)
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
    data = dict(row)
    if data.get("last_approved_at") is not None:
        data["last_approved_at"] = data["last_approved_at"].isoformat()
    return data


def _settings_table_id(table_override: Optional[str] = None) -> str:
    if table_override:
        return table_override
    if AUTOTUNE_BQ_SETTINGS_TABLE:
        return AUTOTUNE_BQ_SETTINGS_TABLE
    project, dataset, _ = _parse_table(_require_autotune_table())
    return f"{project}.{dataset}.autotune_settings"


def get_settings_bigquery(table_override: Optional[str] = None) -> Dict[str, Any]:
    table_id = _settings_table_id(table_override)
    client = _get_bq_client()
    _ensure_settings_table(client, table_id)
    project, dataset, table = _parse_table(table_id)
    table_fqn = f"{project}.{dataset}.{table}"
    query = f"""
    SELECT id, enabled, updated_at, updated_by
    FROM `{table_fqn}`
    WHERE id = 'default'
    LIMIT 1
    """
    rows = list(client.query(query).result())
    if not rows:
        return {"id": "default", "enabled": True, "updated_at": None, "updated_by": None}
    row = dict(rows[0])
    return {
        "id": row.get("id", "default"),
        "enabled": bool(row.get("enabled", True)),
        "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
        "updated_by": row.get("updated_by"),
    }


def set_settings_bigquery(
    enabled: bool,
    updated_by: Optional[str] = None,
    table_override: Optional[str] = None,
) -> None:
    table_id = _settings_table_id(table_override)
    client = _get_bq_client()
    _ensure_settings_table(client, table_id)
    project, dataset, table = _parse_table(table_id)
    table_fqn = f"{project}.{dataset}.{table}"
    now = datetime.utcnow()
    query = f"""
    MERGE `{table_fqn}` T
    USING (
      SELECT 'default' AS id,
             @enabled AS enabled,
             @updated_at AS updated_at,
             @updated_by AS updated_by
    ) S
    ON T.id = S.id
    WHEN MATCHED THEN UPDATE SET
      enabled = S.enabled,
      updated_at = S.updated_at,
      updated_by = S.updated_by
    WHEN NOT MATCHED THEN INSERT(id, enabled, updated_at, updated_by)
      VALUES(S.id, S.enabled, S.updated_at, S.updated_by)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("enabled", "BOOL", bool(enabled)),
            bigquery.ScalarQueryParameter("updated_at", "TIMESTAMP", now),
            bigquery.ScalarQueryParameter("updated_by", "STRING", updated_by),
        ]
    )
    client.query(query, job_config=job_config).result()


def list_runs(
    conn: Optional[sqlite3.Connection] = None,
    status: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    if conn is None:
        return list_runs_bigquery(status=status, limit=limit)
    rows = _get_sqlite_rows(conn, status, limit)
    return rows


def get_run(
    conn: Optional[sqlite3.Connection],
    run_id: str,
    strategy: str,
) -> Optional[Dict[str, Any]]:
    if conn is None:
        return get_run_bigquery(run_id, strategy)
    row = conn.execute(
        "SELECT * FROM tuning_runs WHERE run_id=? AND strategy=?",
        (run_id, strategy),
    ).fetchone()
    return dict(row) if row else None


def update_status(
    conn: Optional[sqlite3.Connection],
    run_id: str,
    strategy: str,
    status: str,
    reviewer: Optional[str] = None,
    comment: Optional[str] = None,
) -> None:
    if conn is None:
        update_status_bigquery(run_id, strategy, status, reviewer, comment)
        return
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


def get_stats(conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
    if conn is None:
        return get_stats_bigquery()
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
    return stats


def get_settings(conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
    if conn is None:
        return get_settings_bigquery()
    row = conn.execute(
        "SELECT id, enabled, updated_at, updated_by FROM autotune_settings WHERE id='default'"
    ).fetchone()
    if not row:
        return {"id": "default", "enabled": True, "updated_at": None, "updated_by": None}
    return {
        "id": row["id"],
        "enabled": bool(row["enabled"]),
        "updated_at": row["updated_at"],
        "updated_by": row["updated_by"],
    }


def set_settings(
    conn: Optional[sqlite3.Connection],
    enabled: bool,
    updated_by: Optional[str] = None,
) -> None:
    if conn is None:
        set_settings_bigquery(enabled, updated_by)
        return
    conn.execute(
        """
        INSERT INTO autotune_settings(id, enabled, updated_at, updated_by)
        VALUES('default', ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            enabled = excluded.enabled,
            updated_at = excluded.updated_at,
            updated_by = excluded.updated_by
        """,
        (1 if enabled else 0, _utc_now(), updated_by),
    )
    conn.commit()


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
    "AUTOTUNE_BQ_SETTINGS_TABLE",
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
    "get_settings",
    "set_settings",
    "dump_dict",
]
