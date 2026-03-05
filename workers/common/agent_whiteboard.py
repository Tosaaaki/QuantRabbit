from __future__ import annotations

import os
import sqlite3
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Optional

DEFAULT_DB_PATH = Path(os.getenv("AGENT_WHITEBOARD_DB_PATH", "logs/agent_whiteboard.db"))
ALLOWED_STATUSES = ("open", "resolved", "archived")
DB_TIMEOUT_SEC = 3.0


@dataclass(frozen=True)
class WhiteboardTask:
    id: int
    task: str
    body: str
    author: str
    status: str
    created_at: str
    updated_at: str
    resolved_at: Optional[str]
    archived_at: Optional[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_status(status: str) -> str:
    normalized = str(status or "").strip().lower()
    if normalized == "all":
        return "all"
    if normalized not in ALLOWED_STATUSES:
        raise ValueError(f"unsupported status: {status}")
    return normalized


def _resolve_db_path(db_path: Path | str | None) -> Path:
    if db_path is None:
        return DEFAULT_DB_PATH
    return Path(db_path).expanduser()


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path), timeout=DB_TIMEOUT_SEC)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


def init_db(db_path: Path | str | None = None) -> None:
    target = _resolve_db_path(db_path)
    with _connect(target) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_whiteboard_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT NOT NULL,
                body TEXT NOT NULL,
                author TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open'
                    CHECK(status IN ('open', 'resolved', 'archived')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                resolved_at TEXT,
                archived_at TEXT
            )
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_agent_whiteboard_status_updated
            ON agent_whiteboard_tasks(status, updated_at DESC, id DESC)
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_agent_whiteboard_updated
            ON agent_whiteboard_tasks(updated_at DESC, id DESC)
            """
        )
        con.commit()


def _row_to_task(row: sqlite3.Row) -> WhiteboardTask:
    return WhiteboardTask(
        id=int(row["id"]),
        task=str(row["task"]),
        body=str(row["body"]),
        author=str(row["author"]),
        status=str(row["status"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        resolved_at=row["resolved_at"],
        archived_at=row["archived_at"],
    )


def get_task(task_id: int, *, db_path: Path | str | None = None) -> Optional[WhiteboardTask]:
    target = _resolve_db_path(db_path)
    init_db(target)
    with _connect(target) as con:
        row = con.execute(
            """
            SELECT id, task, body, author, status, created_at, updated_at, resolved_at, archived_at
            FROM agent_whiteboard_tasks
            WHERE id = ?
            """,
            (int(task_id),),
        ).fetchone()
        if row is None:
            return None
        return _row_to_task(row)


def post_task(
    *,
    task: str,
    body: str = "",
    author: str = "unknown",
    db_path: Path | str | None = None,
) -> WhiteboardTask:
    task_text = str(task or "").strip()
    if not task_text:
        raise ValueError("task is required")

    target = _resolve_db_path(db_path)
    init_db(target)
    now = _now_iso()
    with _connect(target) as con:
        cur = con.execute(
            """
            INSERT INTO agent_whiteboard_tasks
            (task, body, author, status, created_at, updated_at, resolved_at, archived_at)
            VALUES (?, ?, ?, 'open', ?, ?, NULL, NULL)
            """,
            (task_text, str(body or ""), str(author or "unknown"), now, now),
        )
        con.commit()
        task_id = int(cur.lastrowid)
    created = get_task(task_id, db_path=target)
    if created is None:
        raise RuntimeError("failed to read inserted whiteboard task")
    return created


def list_tasks(
    *,
    status: str = "open",
    limit: int = 100,
    db_path: Path | str | None = None,
) -> list[WhiteboardTask]:
    target = _resolve_db_path(db_path)
    init_db(target)
    normalized_status = _normalize_status(status)
    safe_limit = max(1, min(int(limit), 1000))

    where_clause = ""
    params: tuple[object, ...] = (safe_limit,)
    if normalized_status != "all":
        where_clause = "WHERE status = ?"
        params = (normalized_status, safe_limit)

    with _connect(target) as con:
        rows = con.execute(
            f"""
            SELECT id, task, body, author, status, created_at, updated_at, resolved_at, archived_at
            FROM agent_whiteboard_tasks
            {where_clause}
            ORDER BY id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

    return [_row_to_task(row) for row in rows]


def watch_tasks(
    *,
    since_id: int = 0,
    status: str = "all",
    limit: int = 100,
    db_path: Path | str | None = None,
) -> list[WhiteboardTask]:
    target = _resolve_db_path(db_path)
    init_db(target)
    normalized_status = _normalize_status(status)
    safe_limit = max(1, min(int(limit), 1000))
    safe_since = max(0, int(since_id))

    params: list[object] = [safe_since]
    where = "WHERE id > ?"
    if normalized_status != "all":
        where = "WHERE id > ? AND status = ?"
        params.append(normalized_status)
    params.append(safe_limit)

    with _connect(target) as con:
        rows = con.execute(
            f"""
            SELECT id, task, body, author, status, created_at, updated_at, resolved_at, archived_at
            FROM agent_whiteboard_tasks
            {where}
            ORDER BY id ASC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
    return [_row_to_task(row) for row in rows]


def resolve_task(task_id: int, *, db_path: Path | str | None = None) -> Optional[WhiteboardTask]:
    target = _resolve_db_path(db_path)
    init_db(target)
    now = _now_iso()
    with _connect(target) as con:
        con.execute(
            """
            UPDATE agent_whiteboard_tasks
            SET status = 'resolved',
                resolved_at = COALESCE(resolved_at, ?),
                updated_at = ?
            WHERE id = ?
              AND status != 'archived'
            """,
            (now, now, int(task_id)),
        )
        con.commit()
    return get_task(task_id, db_path=target)


def archive_task(task_id: int, *, db_path: Path | str | None = None) -> Optional[WhiteboardTask]:
    target = _resolve_db_path(db_path)
    init_db(target)
    now = _now_iso()
    with _connect(target) as con:
        con.execute(
            """
            UPDATE agent_whiteboard_tasks
            SET status = 'archived',
                archived_at = COALESCE(archived_at, ?),
                updated_at = ?
            WHERE id = ?
            """,
            (now, now, int(task_id)),
        )
        con.commit()
    return get_task(task_id, db_path=target)


def purge_task(task_id: int, *, db_path: Path | str | None = None) -> bool:
    target = _resolve_db_path(db_path)
    init_db(target)
    with _connect(target) as con:
        cur = con.execute(
            "DELETE FROM agent_whiteboard_tasks WHERE id = ?",
            (int(task_id),),
        )
        con.commit()
        return int(cur.rowcount or 0) > 0
