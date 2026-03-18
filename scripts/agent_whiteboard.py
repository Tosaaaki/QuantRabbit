#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from workers.common import agent_whiteboard as whiteboard


def _emit_task(task: whiteboard.WhiteboardTask, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(task.to_dict(), ensure_ascii=False))
        return
    print(
        f"#{task.id} [{task.status}] {task.task} "
        f"(author={task.author}, updated_at={task.updated_at})"
    )
    if task.body:
        print(f"  {task.body}")


def _emit_event(event: whiteboard.WhiteboardEvent, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(event.to_dict(), ensure_ascii=False))
        return
    print(
        f"@{event.id} task#{event.task_id} [{event.event_type}] {event.body} "
        f"(author={event.author}, created_at={event.created_at})"
    )
    if event.metadata is not None:
        print(f"  metadata={json.dumps(event.metadata, ensure_ascii=False)}")


def _emit_activity(activity: whiteboard.WhiteboardActivity, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(activity.to_dict(), ensure_ascii=False))
        return
    prefix = f"[activity#{activity.id}]"
    if activity.kind == "event" and activity.event is not None:
        event = activity.event
        print(
            f"{prefix} task#{event.task_id} [{event.event_type}] {event.body} "
            f"(author={event.author}, created_at={event.created_at})"
        )
        return
    task = activity.task
    print(
        f"{prefix} task#{task.id} [{task.status}] {task.task} "
        f"(author={task.author}, updated_at={task.updated_at})"
    )
    if task.body:
        print(f"  {task.body}")


def _emit_tasks(tasks: list[whiteboard.WhiteboardTask], *, as_json: bool) -> None:
    if as_json:
        payload = {"count": len(tasks), "tasks": [task.to_dict() for task in tasks]}
        print(json.dumps(payload, ensure_ascii=False))
        return
    for task in tasks:
        _emit_task(task, as_json=False)
    if not tasks:
        print("(no tasks)")


def _emit_events(events: list[whiteboard.WhiteboardEvent], *, as_json: bool) -> None:
    if as_json:
        payload = {
            "count": len(events),
            "events": [event.to_dict() for event in events],
        }
        print(json.dumps(payload, ensure_ascii=False))
        return
    for event in events:
        _emit_event(event, as_json=False)
    if not events:
        print("(no events)")


def _parse_metadata_arg(raw: str) -> object | None:
    text = str(raw or "").strip()
    if not text:
        return None
    return json.loads(text)


def _parse_args() -> argparse.Namespace:
    raw_argv = list(sys.argv[1:])
    json_anywhere = "--json" in raw_argv
    filtered_argv = [token for token in raw_argv if token != "--json"]

    parser = argparse.ArgumentParser(description="Local shared whiteboard CLI (SQLite)")
    parser.add_argument(
        "--db",
        default=str(whiteboard.DEFAULT_DB_PATH),
        help="SQLite DB path (default: logs/agent_whiteboard.db)",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON")

    subparsers = parser.add_subparsers(dest="command", required=True)

    post_parser = subparsers.add_parser("post", help="Create a whiteboard task")
    post_parser.add_argument("--task", required=True, help="Task title")
    post_parser.add_argument("--body", default="", help="Task body")
    post_parser.add_argument(
        "--author", default=os.getenv("USER", "agent"), help="Author"
    )

    list_parser = subparsers.add_parser("list", help="List tasks")
    list_parser.add_argument(
        "--status",
        default="open",
        choices=("open", "resolved", "archived", "all"),
        help="Task status filter",
    )
    list_parser.add_argument("--limit", type=int, default=50, help="Max rows")

    watch_parser = subparsers.add_parser("watch", help="Watch new tasks")
    watch_parser.add_argument(
        "--since-id", type=int, default=0, help="Only show id > since-id"
    )
    watch_parser.add_argument(
        "--status",
        default="all",
        choices=("open", "resolved", "archived", "all"),
        help="Task status filter",
    )
    watch_parser.add_argument(
        "--task-id", type=int, default=None, help="Only show one task"
    )
    watch_parser.add_argument("--limit", type=int, default=100, help="Rows per poll")
    watch_parser.add_argument(
        "--interval-sec", type=float, default=2.0, help="Polling interval"
    )
    watch_parser.add_argument(
        "--timeout-sec", type=float, default=0.0, help="Stop after timeout"
    )
    watch_parser.add_argument("--once", action="store_true", help="Poll once and exit")

    resolve_parser = subparsers.add_parser("resolve", help="Resolve task")
    resolve_parser.add_argument("task_id", type=int, help="Task ID")

    archive_parser = subparsers.add_parser("archive-task", help="Archive task")
    archive_parser.add_argument("task_id", type=int, help="Task ID")

    purge_parser = subparsers.add_parser("purge-task", help="Delete task permanently")
    purge_parser.add_argument("task_id", type=int, help="Task ID")
    purge_parser.add_argument(
        "--yes", action="store_true", help="Required safety switch"
    )

    note_parser = subparsers.add_parser("note", help="Add note to task")
    note_parser.add_argument("task_id", type=int, help="Task ID")
    note_parser.add_argument("--body", required=True, help="Note body")
    note_parser.add_argument(
        "--author", default=os.getenv("USER", "agent"), help="Author"
    )
    note_parser.add_argument("--metadata", default="", help="Optional JSON metadata")

    event_parser = subparsers.add_parser("event", help="Add event to task")
    event_parser.add_argument("task_id", type=int, help="Task ID")
    event_parser.add_argument(
        "--type",
        default="event",
        choices=whiteboard.ALLOWED_EVENT_TYPES,
        help="Event type",
    )
    event_parser.add_argument("--body", required=True, help="Event body")
    event_parser.add_argument(
        "--author", default=os.getenv("USER", "agent"), help="Author"
    )
    event_parser.add_argument("--metadata", default="", help="Optional JSON metadata")

    events_parser = subparsers.add_parser("events", help="List events")
    events_parser.add_argument(
        "--task-id", type=int, default=None, help="Task ID filter"
    )
    events_parser.add_argument(
        "--since-id", type=int, default=0, help="Only show id > since-id"
    )
    events_parser.add_argument("--limit", type=int, default=100, help="Max rows")

    auto_parser = subparsers.add_parser(
        "auto-session",
        help="Run a command with automatic task lifecycle management",
    )
    auto_parser.add_argument("--task", required=True, help="Task title")
    auto_parser.add_argument("--body", default="", help="Task body")
    auto_parser.add_argument(
        "--author", default=os.getenv("USER", "agent"), help="Author"
    )
    auto_parser.add_argument(
        "--start-note", default="auto-session started", help="Start event message"
    )
    auto_parser.add_argument(
        "--success-note",
        default="auto-session completed",
        help="Success event message",
    )
    auto_parser.add_argument(
        "--failure-note-prefix",
        default="auto-session failed",
        help="Failure note prefix",
    )
    auto_parser.add_argument(
        "run_command",
        nargs=argparse.REMAINDER,
        help="Command to run. Use '--' before command arguments.",
    )

    args = parser.parse_args(filtered_argv)
    args.json = bool(getattr(args, "json", False) or json_anywhere)
    return args


def _run_watch(args: argparse.Namespace, *, db_path: Path) -> int:
    cursor = max(0, int(args.since_id))
    started = time.monotonic()
    poll_interval = max(0.1, float(args.interval_sec))
    timeout_sec = max(0.0, float(args.timeout_sec))

    while True:
        rows = whiteboard.watch_activity(
            since_id=cursor,
            status=args.status,
            limit=args.limit,
            task_id=args.task_id,
            db_path=db_path,
        )
        if rows:
            for row in rows:
                _emit_activity(row, as_json=args.json)
            cursor = max(cursor, rows[-1].id)

        if args.once:
            return 0
        if timeout_sec > 0 and (time.monotonic() - started) >= timeout_sec:
            return 0
        time.sleep(poll_interval)


def _normalize_auto_session_command(command: list[str]) -> list[str]:
    cmd = list(command)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    return cmd


def _run_auto_session(args: argparse.Namespace, *, db_path: Path) -> int:
    cmd = _normalize_auto_session_command(list(args.run_command))
    if not cmd:
        print("auto-session requires a command. use '-- <cmd> ...'", file=sys.stderr)
        return 2

    task = whiteboard.post_task(
        task=args.task,
        body=args.body,
        author=args.author,
        db_path=db_path,
    )
    whiteboard.post_event(
        task_id=task.id,
        body=args.start_note,
        author=args.author,
        event_type="system",
        metadata={"mode": "auto-session", "phase": "start", "command": cmd},
        db_path=db_path,
    )

    try:
        completed = subprocess.run(cmd, check=False)
        exit_code = int(completed.returncode)
    except Exception as exc:
        whiteboard.post_event(
            task_id=task.id,
            body=f"{args.failure_note_prefix} (exception={exc!r})",
            author=args.author,
            event_type="error",
            metadata={"mode": "auto-session", "phase": "exception", "command": cmd},
            db_path=db_path,
        )
        failed = whiteboard.get_task(task.id, db_path=db_path)
        payload = {
            "task": failed.to_dict() if failed else {"id": task.id},
            "exit_code": 1,
            "status": "open",
            "command": cmd,
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(
                f"auto-session exception (task kept open): task#{task.id}",
                file=sys.stderr,
            )
        return 1

    if exit_code == 0:
        whiteboard.post_event(
            task_id=task.id,
            body=args.success_note,
            author=args.author,
            event_type="system",
            metadata={"mode": "auto-session", "phase": "success", "command": cmd},
            db_path=db_path,
        )
        whiteboard.resolve_task(task.id, db_path=db_path)
        final = whiteboard.archive_task(task.id, db_path=db_path)
        payload = {
            "task": final.to_dict() if final else {"id": task.id},
            "exit_code": 0,
            "status": "archived",
            "command": cmd,
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False))
        elif final is not None:
            _emit_task(final, as_json=False)
        return 0

    whiteboard.post_event(
        task_id=task.id,
        body=f"{args.failure_note_prefix} (exit_code={exit_code})",
        author=args.author,
        event_type="error",
        metadata={
            "mode": "auto-session",
            "phase": "failure",
            "command": cmd,
            "exit_code": exit_code,
        },
        db_path=db_path,
    )
    failed = whiteboard.get_task(task.id, db_path=db_path)
    payload = {
        "task": failed.to_dict() if failed else {"id": task.id},
        "exit_code": exit_code,
        "status": "open",
        "command": cmd,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(f"auto-session failed (task kept open): task#{task.id}", file=sys.stderr)
    return exit_code if exit_code > 0 else 1


def main() -> int:
    args = _parse_args()
    db_path = Path(args.db).expanduser()
    whiteboard.init_db(db_path)

    if args.command == "post":
        row = whiteboard.post_task(
            task=args.task,
            body=args.body,
            author=args.author,
            db_path=db_path,
        )
        _emit_task(row, as_json=args.json)
        return 0

    if args.command == "list":
        rows = whiteboard.list_tasks(
            status=args.status,
            limit=args.limit,
            db_path=db_path,
        )
        _emit_tasks(rows, as_json=args.json)
        return 0

    if args.command == "watch":
        return _run_watch(args, db_path=db_path)

    if args.command == "resolve":
        row = whiteboard.resolve_task(args.task_id, db_path=db_path)
        if row is None:
            print(f"task not found: {args.task_id}", file=sys.stderr)
            return 2
        _emit_task(row, as_json=args.json)
        return 0

    if args.command == "archive-task":
        row = whiteboard.archive_task(args.task_id, db_path=db_path)
        if row is None:
            print(f"task not found: {args.task_id}", file=sys.stderr)
            return 2
        _emit_task(row, as_json=args.json)
        return 0

    if args.command == "purge-task":
        if not args.yes:
            print("refusing to purge without --yes", file=sys.stderr)
            return 2
        removed = whiteboard.purge_task(args.task_id, db_path=db_path)
        if not removed:
            print(f"task not found: {args.task_id}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps({"purged": True, "task_id": int(args.task_id)}))
        else:
            print(f"purged task #{int(args.task_id)}")
        return 0

    if args.command == "note":
        try:
            metadata = _parse_metadata_arg(args.metadata)
        except json.JSONDecodeError as exc:
            print(f"invalid --metadata JSON: {exc}", file=sys.stderr)
            return 2
        try:
            row = whiteboard.post_note(
                task_id=args.task_id,
                body=args.body,
                author=args.author,
                metadata=metadata,
                db_path=db_path,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        _emit_event(row, as_json=args.json)
        return 0

    if args.command == "event":
        try:
            metadata = _parse_metadata_arg(args.metadata)
        except json.JSONDecodeError as exc:
            print(f"invalid --metadata JSON: {exc}", file=sys.stderr)
            return 2
        try:
            row = whiteboard.post_event(
                task_id=args.task_id,
                body=args.body,
                author=args.author,
                event_type=args.type,
                metadata=metadata,
                db_path=db_path,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        _emit_event(row, as_json=args.json)
        return 0

    if args.command == "events":
        rows = whiteboard.list_events(
            task_id=args.task_id,
            since_id=args.since_id,
            limit=args.limit,
            db_path=db_path,
        )
        _emit_events(rows, as_json=args.json)
        return 0

    if args.command == "auto-session":
        return _run_auto_session(args, db_path=db_path)

    print(f"unsupported command: {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
