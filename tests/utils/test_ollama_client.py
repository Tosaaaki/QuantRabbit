from __future__ import annotations

from typing import Any

import utils.ollama_client as oc


class _Resp:
    def __init__(self, body: dict[str, Any], status_code: int = 200) -> None:
        self._body = body
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError("http error")

    def json(self) -> dict[str, Any]:
        return self._body


def test_call_ollama_chat_json_uses_content_json(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_post(url: str, json: dict[str, Any], timeout: float):
        captured["json"] = json
        return _Resp({"message": {"content": '{"action":"ALLOW","scale":1.0,"reason":"ok"}'}})

    monkeypatch.setattr(oc.requests, "post", _fake_post)
    payload = oc.call_ollama_chat_json(
        "prompt",
        model="gpt-oss:20b",
        url="http://127.0.0.1:11434/api/chat",
        timeout_sec=6,
    )

    assert payload is not None
    assert payload["action"] == "ALLOW"
    assert captured["json"]["think"] is False


def test_call_ollama_chat_json_falls_back_to_thinking_json(monkeypatch) -> None:
    def _fake_post(_url: str, json: dict[str, Any], timeout: float):
        assert json["think"] is False
        assert timeout >= 1.0
        return _Resp(
            {
                "message": {
                    "content": "",
                    "thinking": 'prelude {"action":"REDUCE","scale":0.4,"reason":"risk"}',
                }
            }
        )

    monkeypatch.setattr(oc.requests, "post", _fake_post)
    payload = oc.call_ollama_chat_json(
        "prompt",
        model="gpt-oss:20b",
        url="http://127.0.0.1:11434/api/chat",
        timeout_sec=6,
    )

    assert payload is not None
    assert payload["action"] == "REDUCE"
    assert payload["scale"] == 0.4


def test_call_ollama_chat_json_think_none_omits_field(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_post(_url: str, json: dict[str, Any], timeout: float):
        captured["json"] = json
        assert timeout >= 1.0
        return _Resp({"message": {"content": '{"action":"BLOCK","scale":0.0,"reason":"avoid"}'}})

    monkeypatch.setattr(oc.requests, "post", _fake_post)
    payload = oc.call_ollama_chat_json(
        "prompt",
        model="gpt-oss:20b",
        url="http://127.0.0.1:11434/api/chat",
        timeout_sec=6,
        think=None,
    )

    assert payload is not None
    assert payload["action"] == "BLOCK"
    assert "think" not in captured["json"]


def test_call_ollama_chat_json_retries_when_truncated_thinking(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def _fake_post(_url: str, json: dict[str, Any], timeout: float):
        calls.append(json)
        assert timeout >= 1.0
        if len(calls) == 1:
            return _Resp(
                {
                    "done_reason": "length",
                    "message": {"content": "", "thinking": "long thought without json"},
                }
            )
        return _Resp({"done_reason": "stop", "message": {"content": '{"action":"ALLOW","scale":1.0,"reason":"ok"}'}})

    monkeypatch.setattr(oc.requests, "post", _fake_post)
    payload = oc.call_ollama_chat_json(
        "prompt",
        model="gpt-oss:20b",
        url="http://127.0.0.1:11434/api/chat",
        timeout_sec=20,
        max_tokens=256,
    )

    assert payload is not None
    assert payload["action"] == "ALLOW"
    assert len(calls) == 2
    assert calls[0]["options"]["num_predict"] == 256
    assert calls[1]["options"]["num_predict"] >= 2048


def test_call_ollama_chat_json_does_not_retry_on_short_timeout(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def _fake_post(_url: str, json: dict[str, Any], timeout: float):
        calls.append(json)
        assert timeout >= 1.0
        return _Resp(
            {
                "done_reason": "length",
                "message": {"content": "", "thinking": "long thought without json"},
            }
        )

    monkeypatch.setattr(oc.requests, "post", _fake_post)
    payload = oc.call_ollama_chat_json(
        "prompt",
        model="gpt-oss:20b",
        url="http://127.0.0.1:11434/api/chat",
        timeout_sec=6,
        max_tokens=256,
    )

    assert payload is None
    assert len(calls) == 1
