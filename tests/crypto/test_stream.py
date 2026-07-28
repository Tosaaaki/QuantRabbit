from __future__ import annotations

import pytest

from quant_rabbit.crypto.stream import (
    BitbankStreamError,
    parse_socketio_message,
)


def test_parse_public_socketio_ticker_message() -> None:
    payload = parse_socketio_message(
        '42["message",{"room_name":"ticker_btc_jpy","message":{"data":{"last":"1"}}}]'
    )
    assert payload is not None
    assert payload["room_name"] == "ticker_btc_jpy"
    assert parse_socketio_message("2") == {"engine_event": "ping"}
    assert parse_socketio_message("40") is None


def test_malformed_public_stream_message_fails_closed() -> None:
    with pytest.raises(BitbankStreamError):
        parse_socketio_message('42["message",{"message":{}}]')
