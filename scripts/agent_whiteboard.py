#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
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


def _emit_tasks(tasks: list[whiteboard.WhiteboardTask], *, as_json: bool) -> None:
    if as_json:
        payload = {"count": len(tasks), "tasks": [task.to_dict() for task in tasks]}
        print(json.dumps(payload, ensure_ascii=False))
        return
    for task in tasks:
        _emit_task(task, as_json=False)
    if not tasks:
        print("(no tasks)")


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
    post_parser.add_argument("--author", default=os.getenv("USER", "agent"), help="Author")

    list_parser = subparsers.add_parser("list", help="List tasks")
    list_parser.add_argument(
        "--status",
        default="open",
        choices=("open", "resolved", "archived", "all"),
        help="Task status filter",
    )
    list_parser.add_argument("--limit", type=int, default=50, help="Max rows")

    watch_parser = subparsers.add_parser("watch", help="Watch new tasks")
    watch_parser.add_argument("--since-id", type=int, default=0, help="Only show id > since-id")
    watch_parser.add_argument(
        "--status",
        default="all",
        choices=("open", "resolved", "archived", "all"),
        help="Task status filter",
    )
    watch_parser.add_argument("--limit", type=int, default=100, help="Rows per poll")
    watch_parser.add_argument("--interval-sec", type=float, default=2.0, help="Polling interval")
    watch_parser.add_argument("--timeout-sec", type=float, default=0.0, help="Stop after timeout")
    watch_parser.add_argument("--once", action="store_true", help="Poll once and exit")

    resolve_parser = subparsers.add_parser("resolve", help="Resolve task")
    resolve_parser.add_argument("task_id", type=int, help="Task ID")

    archive_parser = subparsers.add_parser("archive-task", help="Archive task")
    archive_parser.add_argument("task_id", type=int, help="Task ID")

    purge_parser = subparsers.add_parser("purge-task", help="Delete task permanently")
    purge_parser.add_argument("task_id", type=int, help="Task ID")
    purge_parser.add_argument("--yes", action="store_true", help="Required safety switch")

    args = parser.parse_args(filtered_argv)
    args.json = bool(getattr(args, "json", False) or json_anywhere)
    return args


def _run_watch(args: argparse.Namespace, *, db_path: Path) -> int:
    cursor = max(0, int(args.since_id))
    started = time.monotonic()
    poll_interval = max(0.1, float(args.interval_sec))
    timeout_sec = max(0.0, float(args.timeout_sec))

    while True:
        rows = whiteboard.watch_tasks(
            since_id=cursor,
            status=args.status,
            limit=args.limit,
            db_path=db_path,
        )
        if rows:
            if args.json:
                for row in rows:
                    print(json.dumps(row.to_dict(), ensure_ascii=False))
            else:
                for row in rows:
                    _emit_task(row, as_json=False)
            cursor = max(cursor, rows[-1].id)

        if args.once:
            return 0
        if timeout_sec > 0 and (time.monotonic() - started) >= timeout_sec:
            return 0
        time.sleep(poll_interval)


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

    print(f"unsupported command: {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
