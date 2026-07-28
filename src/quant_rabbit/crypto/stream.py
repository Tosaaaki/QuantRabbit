from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable
from typing import Any

from .config import ScannerConfig


class BitbankStreamError(RuntimeError):
    pass


def parse_socketio_message(frame: str) -> dict[str, Any] | None:
    if frame == "2":
        return {"engine_event": "ping"}
    if not frame.startswith("42"):
        return None
    try:
        event = json.loads(frame[2:])
    except json.JSONDecodeError as exc:
        raise BitbankStreamError("malformed Socket.IO event") from exc
    if not isinstance(event, list) or len(event) != 2 or event[0] != "message":
        return None
    payload = event[1]
    if not isinstance(payload, dict) or "room_name" not in payload:
        raise BitbankStreamError("Socket.IO message missing room_name")
    return payload


class BitbankPublicStream:
    """Small Socket.IO v4 reader for candidate-only public rooms."""

    def __init__(self, config: ScannerConfig | None = None) -> None:
        self.config = config or ScannerConfig.from_env()

    async def messages(
        self, rooms: Iterable[str], *, max_messages: int | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        try:
            import websockets
        except ImportError as exc:
            raise BitbankStreamError(
                "Install the crypto-stream extra to use Public Stream"
            ) from exc
        delivered = 0
        async with websockets.connect(
            self.config.stream_url,
            open_timeout=self.config.request_timeout_sec,
            close_timeout=2,
            ping_interval=None,
        ) as socket:
            opening = await socket.recv()
            if not isinstance(opening, str) or not opening.startswith("0"):
                raise BitbankStreamError("missing Socket.IO opening packet")
            await socket.send("40")
            while True:
                connected = await socket.recv()
                if connected == "2":
                    await socket.send("3")
                    continue
                if isinstance(connected, str) and connected.startswith("40"):
                    break
                if isinstance(connected, str) and connected.startswith("44"):
                    raise BitbankStreamError("Socket.IO namespace connection failed")
            for room in rooms:
                await socket.send(f'42["join-room",{json.dumps(room)}]')
            async for raw in socket:
                if raw == "2":
                    await socket.send("3")
                    continue
                message = parse_socketio_message(raw)
                if message is None:
                    continue
                yield message
                delivered += 1
                if max_messages is not None and delivered >= max_messages:
                    return

    async def collect(
        self, rooms: Iterable[str], *, max_messages: int, timeout_sec: float
    ) -> list[dict[str, Any]]:
        async def _collect() -> list[dict[str, Any]]:
            return [
                item
                async for item in self.messages(rooms, max_messages=max_messages)
            ]

        return await asyncio.wait_for(_collect(), timeout=timeout_sec)
