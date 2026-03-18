#!/usr/bin/env python3
"""Read-only MCP server for local SQLite databases.

Intended use: local read-only market logs for QuantRabbit.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

JSONRPC_VERSION = "2.0"
MAX_RESULT_ROWS = 1000
DEFAULT_RESULT_ROWS = 200


def _read_mcp_message() -> dict | None:
    """Read one MCP JSON-RPC message from stdin."""
    raw = sys.stdin.buffer
    headers = {}
    while True:
        line = raw.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        key, _, value = line.decode("utf-8").partition(":")
        if key and value:
            headers[key.strip().lower()] = value.strip()

    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    payload = raw.read(length)
    if not payload:
        return None
    return json.loads(payload.decode("utf-8"))


def _write_mcp_message(message: dict) -> None:
    data = json.dumps(message, ensure_ascii=False).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode("utf-8"))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def _json_response(
    message_id: Any, result: dict | None = None, error: dict | None = None
) -> dict:
    base = {
        "jsonrpc": JSONRPC_VERSION,
        "id": message_id,
    }
    if error is not None:
        base["error"] = error
    else:
        base["result"] = result or {}
    return base


def _text_content(payload: dict) -> dict:
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload, ensure_ascii=False, indent=2),
            }
        ]
    }


def _is_readonly_sql(sql: str) -> bool:
    trimmed = re.sub(
        r"(--.*?$|/\*.*?\*/)", "", sql, flags=re.MULTILINE | re.DOTALL
    ).strip()
    lowered = trimmed.lower()
    return bool(lowered.startswith(("select", "with", "pragma", "explain")))


class ReadonlySqliteClient:
    def __init__(self, db_path: str) -> None:
        path = Path(db_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Database not found: {path}")
        self.conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row

    def query(self, sql: str, max_rows: int | None = None) -> dict:
        if not _is_readonly_sql(sql):
            raise PermissionError(
                "Only read-only SQL is allowed (SELECT/WITH/PRAGMA/EXPLAIN)."
            )
        if max_rows is None:
            max_rows = DEFAULT_RESULT_ROWS
        if not isinstance(max_rows, int):
            raise TypeError("max_rows must be integer")
        if max_rows < 1 or max_rows > MAX_RESULT_ROWS:
            raise ValueError(f"max_rows must be between 1 and {MAX_RESULT_ROWS}")

        cur = self.conn.execute(sql)
        rows = []
        truncated = False
        target_rows = max_rows + 1
        while len(rows) < target_rows:
            batch = cur.fetchmany(min(100, target_rows - len(rows)))
            if not batch:
                break
            rows.extend(batch)
        if len(rows) > max_rows:
            rows = rows[:max_rows]
            truncated = True

        return {
            "columns": [d[0] for d in (cur.description or [])],
            "rows": [dict(r) for r in rows],
            "row_count": len(rows),
            "truncated": truncated,
        }


def _tool_list() -> dict:
    return {
        "tools": [
            {
                "name": "query",
                "description": "Execute a read-only SQL query on the configured SQLite database.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "SQL statement (read-only only).",
                        },
                        "max_rows": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_RESULT_ROWS,
                            "default": DEFAULT_RESULT_ROWS,
                            "description": "Optional max rows to return.",
                        },
                    },
                    "required": ["sql"],
                },
            }
        ]
    }


def _tool_call(
    name: str, args: dict[str, Any] | None, client: ReadonlySqliteClient
) -> dict:
    if name != "query":
        raise RuntimeError(f"Unknown tool: {name}")
    if not args:
        raise RuntimeError("Missing tool arguments")

    sql = args.get("sql", "")
    if not isinstance(sql, str) or not sql.strip():
        raise RuntimeError("Missing required SQL string")

    max_rows = args.get("max_rows", DEFAULT_RESULT_ROWS)
    if isinstance(max_rows, bool) or not isinstance(max_rows, int):
        raise RuntimeError("max_rows must be integer")

    return client.query(sql=sql, max_rows=max_rows)


def _handle_message(message: dict, client: ReadonlySqliteClient) -> None:
    if not isinstance(message, dict):
        return

    method = message.get("method")
    message_id = message.get("id")

    if method == "initialize":
        _write_mcp_message(
            _json_response(
                message_id,
                {
                    "protocolVersion": "2025-06-18",
                    "serverInfo": {"name": "qr-sqlite-readonly", "version": "1.1.0"},
                    "capabilities": {"tools": {"listChanged": False}},
                },
            )
        )
        return

    if method == "tools/list":
        _write_mcp_message(_json_response(message_id, _tool_list()))
        return

    if method == "tools/call":
        params = message.get("params", {})
        try:
            tool = (params or {}).get("name")
            args = (params or {}).get("arguments", {})
            payload = _tool_call(tool, args, client)
            _write_mcp_message(
                _json_response(
                    message_id,
                    {
                        **_text_content(payload),
                        "isError": False,
                    },
                )
            )
        except Exception as exc:  # pragma: no cover
            _write_mcp_message(
                _json_response(
                    message_id,
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": f"SQLite MCP failed: {exc}",
                            }
                        ],
                        "isError": True,
                    },
                )
            )
        return

    if method in {"notifications/initialized", "notifications/cancelled"}:
        return

    if method == "resources/list":
        _write_mcp_message(_json_response(message_id, {"resources": []}))
        return

    if method is None:
        return

    _write_mcp_message(
        _json_response(
            message_id,
            error={
                "code": -32601,
                "message": f"Method not found: {method}",
            },
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only SQLite MCP server")
    parser.add_argument(
        "--db", required=True, help="Absolute or relative path to SQLite DB"
    )
    args = parser.parse_args()
    try:
        client = ReadonlySqliteClient(args.db)
    except Exception as exc:  # pragma: no cover
        client = None
        startup_error = str(exc)
    else:
        startup_error = None

    while True:
        message = _read_mcp_message()
        if message is None:
            return 0
        if message.get("method") in {
            "notifications/initialized",
            "notifications/cancelled",
        }:
            continue
        if message.get("method") == "initialize" and startup_error:
            _write_mcp_message(
                _json_response(
                    message.get("id"),
                    {
                        "protocolVersion": "2025-06-18",
                        "serverInfo": {
                            "name": "qr-sqlite-readonly",
                            "version": "1.1.0",
                        },
                        "capabilities": {},
                    },
                )
            )
            continue
        if startup_error:
            _write_mcp_message(
                _json_response(
                    message.get("id"),
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": f"startup config error: {startup_error}",
                            }
                        ],
                        "isError": True,
                    },
                )
            )
            continue
        _handle_message(message, client)


if __name__ == "__main__":
    raise SystemExit(main())
