#!/usr/bin/env python3
"""Read-only MCP server for OANDA market observations.

Exports only observation tools (pricing/summary/open_trades/candles) and no
mutation operations.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.secrets import get_secret


JSONRPC_VERSION = "2.0"
SERVER_VERSION = "1.2.0"
ALLOWED_GRANULARITIES = {
    "S5",
    "S10",
    "S15",
    "S30",
    "M1",
    "M2",
    "M4",
    "M5",
    "M10",
    "M15",
    "M30",
    "H1",
    "H2",
    "H3",
    "H4",
    "H6",
    "H8",
    "H12",
    "D",
    "W",
    "M",
}
INSTRUMENT_RE = re.compile(r"^[A-Z0-9]+_[A-Z0-9]+$")


def _read_mcp_message() -> dict | None:
    """Read one MCP message from stdin using Content-Length framing."""
    raw = sys.stdin.buffer
    header = {}
    while True:
        line = raw.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        key, _, value = line.decode("utf-8").rstrip("\r\n").partition(":")
        if value:
            header[key.strip().lower()] = value.strip()

    length = int(header.get("content-length", "0"))
    if length <= 0:
        return None
    payload = raw.read(length)
    if not payload:
        return None
    return json.loads(payload.decode("utf-8"))


def _write_mcp_message(message: dict) -> None:
    """Write one MCP message to stdout with Content-Length framing."""
    data = json.dumps(message, ensure_ascii=False).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode("utf-8"))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def _json_response(message_id, result=None, error=None):
    if error is None:
        return {
            "jsonrpc": JSONRPC_VERSION,
            "id": message_id,
            "result": result or {},
        }
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": message_id,
        "error": error,
    }


def _load_secret(name: str, *, default: str | None = None) -> str:
    try:
        value = get_secret(name)
    except KeyError:
        if default is not None:
            return default
        raise RuntimeError(f"{name} is not set in env/config/env.toml/Secret Manager") from None
    except Exception as exc:
        raise RuntimeError(
            f"failed to resolve {name} via env/config/env.toml/Secret Manager: {exc}"
        ) from exc
    return str(value)


def _require_instrument(value: object) -> str:
    if not isinstance(value, str):
        raise RuntimeError("instrument must be string")
    instrument = value.strip().upper()
    if not instrument or not INSTRUMENT_RE.fullmatch(instrument):
        raise RuntimeError("instrument must look like USD_JPY")
    return instrument


def _require_bool(value: object, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise RuntimeError(f"{name} must be boolean")


def _require_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{name} must be integer")
    if value < minimum or value > maximum:
        raise RuntimeError(f"{name} must be between {minimum} and {maximum}")
    return value


def _require_granularity(value: object) -> str:
    if not isinstance(value, str):
        raise RuntimeError("granularity must be string")
    granularity = value.strip().upper()
    if granularity not in ALLOWED_GRANULARITIES:
        raise RuntimeError(f"granularity must be one of {sorted(ALLOWED_GRANULARITIES)}")
    return granularity


class OandaReadOnlyClient:
    def __init__(self) -> None:
        self.account_id = _load_secret("oanda_account_id")
        self.token = _load_secret("oanda_token")
        practice = _load_secret("oanda_practice", default="false").strip().lower()
        self.host = "https://api-fxpractice.oanda.com" if practice in ("1", "true", "yes") else "https://api-fxtrade.oanda.com"

    def _request(self, path: str, params: dict | None = None) -> dict:
        query = ""
        if params:
            query = "?" + urllib.parse.urlencode(params, doseq=True)
        url = f"{self.host}/v3{path}{query}"

        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept-Datetime-Format": "RFC3339",
                "Content-Type": "application/json",
            },
            method="GET",
        )
        context = ssl.create_default_context()
        started_at = time.time()
        try:
            with urllib.request.urlopen(request, context=context, timeout=8.0) as response:
                body = response.read().decode("utf-8")
                return {
                    "status": response.getcode(),
                    "elapsed_ms": int((time.time() - started_at) * 1000),
                    "payload": json.loads(body),
                }
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc

    def pricing(self, instrument: str = "USD_JPY", include_units_available: bool = True) -> dict:
        return self._request(
            f"/accounts/{self.account_id}/pricing",
            {
                "instruments": instrument,
                "includeUnitsAvailable": "true" if include_units_available else "false",
            },
        )

    def summary(self) -> dict:
        return self._request(f"/accounts/{self.account_id}/summary")

    def open_trades(self) -> dict:
        return self._request(f"/accounts/{self.account_id}/openTrades")

    def candles(
        self,
        instrument: str = "USD_JPY",
        count: int = 30,
        granularity: str = "M5",
        smooth: bool = False,
    ) -> dict:
        return self._request(
            f"/instruments/{instrument}/candles",
            {
                "count": str(count),
                "granularity": granularity,
                "price": "M",
                "smooth": str(smooth).lower(),
            },
        )


def _tool_list():
    return {
        "tools": [
            {
                "name": "pricing",
                "description": "Get OANDA pricing snapshot for a symbol.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "instrument": {"type": "string", "default": "USD_JPY"},
                        "include_units_available": {"type": "boolean", "default": True},
                    },
                },
            },
            {
                "name": "summary",
                "description": "Get OANDA account summary.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "open_trades",
                "description": "Get current open trades from OANDA account.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "candles",
                "description": "Get latest candles for an instrument.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "instrument": {"type": "string", "default": "USD_JPY"},
                        "count": {"type": "integer", "default": 30},
                        "granularity": {"type": "string", "default": "M5"},
                        "smooth": {"type": "boolean", "default": False},
                    },
                },
            },
        ]
    }


def _text_content(payload: dict) -> dict:
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload, ensure_ascii=False, indent=2),
            }
        ]
    }


def _tool_call(name: str, args: dict | None, client: OandaReadOnlyClient) -> dict:
    args = args or {}
    if name == "pricing":
        return client.pricing(
            instrument=_require_instrument(args.get("instrument", "USD_JPY")),
            include_units_available=_require_bool(
                args.get("include_units_available", True),
                name="include_units_available",
            ),
        )
    if name == "summary":
        return client.summary()
    if name == "open_trades":
        return client.open_trades()
    if name == "candles":
        return client.candles(
            instrument=_require_instrument(args.get("instrument", "USD_JPY")),
            count=_require_int(args.get("count", 30), name="count", minimum=1, maximum=5000),
            granularity=_require_granularity(args.get("granularity", "M5")),
            smooth=_require_bool(args.get("smooth", False), name="smooth"),
        )
    raise RuntimeError(f"Unknown tool: {name}")


def _handle_message(message: dict, client: OandaReadOnlyClient) -> None:
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
                        "serverInfo": {"name": "qr-oanda-observer", "version": SERVER_VERSION},
                        "capabilities": {
                            "tools": {"listChanged": False},
                        },
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
        except Exception as exc:
            _write_mcp_message(
                _json_response(
                    message_id,
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": f"OANDA MCP failed: {exc}",
                            }
                        ],
                        "isError": True,
                    },
                )
            )
        return

    if method == "notifications/initialized":
        return

    if method == "resources/list":
        _write_mcp_message(_json_response(message_id, {"resources": []}))
        return

    if method == "notifications/cancelled":
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--readonly", action="store_true")
    _ = parser.parse_args()
    try:
        client = OandaReadOnlyClient()
    except Exception as exc:
        # Keep server bootable but fail loud on first initialize/tool call.
        client = None
        startup_error = str(exc)
    else:
        startup_error = None

    while True:
        message = _read_mcp_message()
        if message is None:
            return 0
        if message.get("method") in {"notifications/initialized", "notifications/cancelled"}:
            continue

        if message.get("method") == "initialize" and startup_error:
            _write_mcp_message(
                _json_response(
                    message.get("id"),
                    {
                        "protocolVersion": "2025-06-18",
                        "serverInfo": {"name": "qr-oanda-observer", "version": SERVER_VERSION},
                        "capabilities": {},
                    },
                )
            )
            continue

        if startup_error:
            _write_mcp_message(
                _json_response(
                    message.get("id"),
                    {"content": [{"type": "text", "text": f"startup config error: {startup_error}"}], "isError": True}
                )
            )
            if message.get("method") != "initialize":
                continue

        _handle_message(message, client)


if __name__ == "__main__":
    raise SystemExit(main())
