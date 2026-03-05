from __future__ import annotations

import json
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
ALLOWED_EVENT_TYPES = ("event", "note", "error", "system")
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


@dataclass(frozen=True)
class WhiteboardEvent:
    id: int
    task_id: int
    event_type: str
    body: str
    author: str
    created_at: str
    metadata: Optional[object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class WhiteboardActivity:
    id: int
    kind: str
    task_id: int
    created_at: str
    task: WhiteboardTask
    event: Optional[WhiteboardEvent]

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.id,
            "kind": self.kind,
            "task_id": self.task_id,
            "created_at": self.created_at,
            "task": self.task.to_dict(),
        }
        if self.event is not None:
            payload["event"] = self.event.to_dict()
        return payload


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_status(status: str) -> str:
    normalized = str(status or "").strip().lower()
    if normalized == "all":
        return "all"
    if normalized not in ALLOWED_STATUSES:
        raise ValueError(f"unsupported status: {status}")
    return normalized


def _normalize_event_type(event_type: str) -> str:
    normalized = str(event_type or "").strip().lower()
    if normalized not in ALLOWED_EVENT_TYPES:
        raise ValueError(f"unsupported event_type: {event_type}")
    return normalized


def _resolve_db_path(db_path: Path | str | None) -> Path:
    if db_path is None:
        return DEFAULT_DB_PATH
    return Path(db_path).expanduser()


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path), timeout=DB_TIMEOUT_SEC)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON;")
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
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_whiteboard_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER NOT NULL,
                event_type TEXT NOT NULL
                    CHECK(event_type IN ('event', 'note', 'error', 'system')),
                body TEXT NOT NULL,
                author TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY(task_id) REFERENCES agent_whiteboard_tasks(id) ON DELETE CASCADE
            )
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_agent_whiteboard_events_task
            ON agent_whiteboard_events(task_id, id DESC)
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_agent_whiteboard_events_created
            ON agent_whiteboard_events(created_at DESC, id DESC)
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_whiteboard_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL CHECK(kind IN ('task', 'event')),
                task_id INTEGER NOT NULL,
                event_id INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY(task_id) REFERENCES agent_whiteboard_tasks(id) ON DELETE CASCADE,
                FOREIGN KEY(event_id) REFERENCES agent_whiteboard_events(id) ON DELETE CASCADE
            )
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_agent_whiteboard_activity_task
            ON agent_whiteboard_activity(task_id, id DESC)
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_agent_whiteboard_activity_created
            ON agent_whiteboard_activity(created_at DESC, id DESC)
            """
        )
        con.execute(
            """
            INSERT INTO agent_whiteboard_activity (kind, task_id, event_id, created_at)
            SELECT 'task', t.id, NULL, t.created_at
            FROM agent_whiteboard_tasks t
            WHERE NOT EXISTS (
                SELECT 1
                FROM agent_whiteboard_activity a
                WHERE a.kind = 'task'
                  AND a.task_id = t.id
            )
            ORDER BY t.id ASC
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


def _decode_metadata(raw: object) -> Optional[object]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return {"raw": text}


def _row_to_event(row: sqlite3.Row) -> WhiteboardEvent:
    return WhiteboardEvent(
        id=int(row["id"]),
        task_id=int(row["task_id"]),
        event_type=str(row["event_type"]),
        body=str(row["body"]),
        author=str(row["author"]),
        created_at=str(row["created_at"]),
        metadata=_decode_metadata(row["metadata"]),
    )


def _task_from_activity_row(row: sqlite3.Row) -> WhiteboardTask:
    return WhiteboardTask(
        id=int(row["task_id"]),
        task=str(row["task_task"]),
        body=str(row["task_body"]),
        author=str(row["task_author"]),
        status=str(row["task_status"]),
        created_at=str(row["task_created_at"]),
        updated_at=str(row["task_updated_at"]),
        resolved_at=row["task_resolved_at"],
        archived_at=row["task_archived_at"],
    )


def _event_from_activity_row(row: sqlite3.Row) -> Optional[WhiteboardEvent]:
    if row["event_id"] is None:
        return None
    return WhiteboardEvent(
        id=int(row["event_id"]),
        task_id=int(row["event_task_id"]),
        event_type=str(row["event_type"]),
        body=str(row["event_body"]),
        author=str(row["event_author"]),
        created_at=str(row["event_created_at"]),
        metadata=_decode_metadata(row["event_metadata"]),
    )


def _append_task_activity(con: sqlite3.Connection, *, task_id: int, created_at: str) -> None:
    con.execute(
        """
        INSERT INTO agent_whiteboard_activity (kind, task_id, event_id, created_at)
        VALUES ('task', ?, NULL, ?)
        """,
        (int(task_id), str(created_at)),
    )


def _append_event_activity(
    con: sqlite3.Connection,
    *,
    task_id: int,
    event_id: int,
    created_at: str,
) -> None:
    con.execute(
        """
        INSERT INTO agent_whiteboard_activity (kind, task_id, event_id, created_at)
        VALUES ('event', ?, ?, ?)
        """,
        (int(task_id), int(event_id), str(created_at)),
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
        task_id = int(cur.lastrowid)
        _append_task_activity(con, task_id=task_id, created_at=now)
        con.commit()
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
        if int(con.total_changes or 0) > 0:
            _append_task_activity(con, task_id=int(task_id), created_at=now)
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
        if int(con.total_changes or 0) > 0:
            _append_task_activity(con, task_id=int(task_id), created_at=now)
        con.commit()
    return get_task(task_id, db_path=target)


def get_event(event_id: int, *, db_path: Path | str | None = None) -> Optional[WhiteboardEvent]:
    target = _resolve_db_path(db_path)
    init_db(target)
    with _connect(target) as con:
        row = con.execute(
            """
            SELECT id, task_id, event_type, body, author, created_at, metadata
            FROM agent_whiteboard_events
            WHERE id = ?
            """,
            (int(event_id),),
        ).fetchone()
        if row is None:
            return None
        return _row_to_event(row)


def post_event(
    *,
    task_id: int,
    body: str,
    author: str = "unknown",
    event_type: str = "event",
    metadata: Optional[object] = None,
    db_path: Path | str | None = None,
) -> WhiteboardEvent:
    normalized_type = _normalize_event_type(event_type)
    body_text = str(body or "").strip()
    if not body_text:
        raise ValueError("body is required")

    target = _resolve_db_path(db_path)
    init_db(target)
    now = _now_iso()
    metadata_text: Optional[str] = None
    if metadata is not None:
        metadata_text = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))

    with _connect(target) as con:
        exists = con.execute(
            "SELECT 1 FROM agent_whiteboard_tasks WHERE id = ?",
            (int(task_id),),
        ).fetchone()
        if exists is None:
            raise ValueError(f"task not found: {int(task_id)}")
        cur = con.execute(
            """
            INSERT INTO agent_whiteboard_events
            (task_id, event_type, body, author, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                int(task_id),
                normalized_type,
                body_text,
                str(author or "unknown"),
                now,
                metadata_text,
            ),
        )
        event_id = int(cur.lastrowid)
        con.execute(
            """
            UPDATE agent_whiteboard_tasks
            SET updated_at = ?
            WHERE id = ?
            """,
            (now, int(task_id)),
        )
        _append_event_activity(
            con,
            task_id=int(task_id),
            event_id=event_id,
            created_at=now,
        )
        con.commit()
    created = get_event(event_id, db_path=target)
    if created is None:
        raise RuntimeError("failed to read inserted whiteboard event")
    return created


def post_note(
    *,
    task_id: int,
    body: str,
    author: str = "unknown",
    metadata: Optional[object] = None,
    db_path: Path | str | None = None,
) -> WhiteboardEvent:
    return post_event(
        task_id=task_id,
        body=body,
        author=author,
        event_type="note",
        metadata=metadata,
        db_path=db_path,
    )


def list_events(
    *,
    task_id: Optional[int] = None,
    since_id: int = 0,
    limit: int = 100,
    db_path: Path | str | None = None,
) -> list[WhiteboardEvent]:
    target = _resolve_db_path(db_path)
    init_db(target)
    safe_limit = max(1, min(int(limit), 1000))
    safe_since = max(0, int(since_id))

    params: list[object] = [safe_since]
    where = "WHERE id > ?"
    if task_id is not None:
        where = "WHERE id > ? AND task_id = ?"
        params.append(int(task_id))
    params.append(safe_limit)

    with _connect(target) as con:
        rows = con.execute(
            f"""
            SELECT id, task_id, event_type, body, author, created_at, metadata
            FROM agent_whiteboard_events
            {where}
            ORDER BY id ASC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
    return [_row_to_event(row) for row in rows]


def watch_activity(
    *,
    since_id: int = 0,
    status: str = "all",
    limit: int = 100,
    task_id: Optional[int] = None,
    db_path: Path | str | None = None,
) -> list[WhiteboardActivity]:
    target = _resolve_db_path(db_path)
    init_db(target)
    normalized_status = _normalize_status(status)
    safe_limit = max(1, min(int(limit), 1000))
    safe_since = max(0, int(since_id))

    params: list[object] = [safe_since]
    filters = ["a.id > ?"]
    if normalized_status != "all":
        filters.append("t.status = ?")
        params.append(normalized_status)
    if task_id is not None:
        filters.append("a.task_id = ?")
        params.append(int(task_id))
    params.append(safe_limit)
    where = " AND ".join(filters)

    with _connect(target) as con:
        rows = con.execute(
            f"""
            SELECT
                a.id AS activity_id,
                a.kind AS activity_kind,
                a.task_id AS activity_task_id,
                a.created_at AS activity_created_at,
                t.id AS task_id,
                t.task AS task_task,
                t.body AS task_body,
                t.author AS task_author,
                t.status AS task_status,
                t.created_at AS task_created_at,
                t.updated_at AS task_updated_at,
                t.resolved_at AS task_resolved_at,
                t.archived_at AS task_archived_at,
                e.id AS event_id,
                e.task_id AS event_task_id,
                e.event_type AS event_type,
                e.body AS event_body,
                e.author AS event_author,
                e.created_at AS event_created_at,
                e.metadata AS event_metadata
            FROM agent_whiteboard_activity a
            INNER JOIN agent_whiteboard_tasks t
                ON t.id = a.task_id
            LEFT JOIN agent_whiteboard_events e
                ON e.id = a.event_id
            WHERE {where}
            ORDER BY a.id ASC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()

    activities: list[WhiteboardActivity] = []
    for row in rows:
        activities.append(
            WhiteboardActivity(
                id=int(row["activity_id"]),
                kind=str(row["activity_kind"]),
                task_id=int(row["activity_task_id"]),
                created_at=str(row["activity_created_at"]),
                task=_task_from_activity_row(row),
                event=_event_from_activity_row(row),
            )
        )
    return activities


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
