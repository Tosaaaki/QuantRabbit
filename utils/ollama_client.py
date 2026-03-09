from __future__ import annotations

import json
from typing import Any, Optional

import requests


def _extract_json_payload(text: str) -> Optional[dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(raw[start : end + 1])
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _normalize_keep_alive(value: Optional[str]) -> Any:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        numeric = float(text)
    except Exception:
        return text
    if numeric.is_integer():
        return int(numeric)
    return numeric


def call_ollama_chat_json(
    prompt: str,
    *,
    model: str,
    url: str,
    timeout_sec: float,
    temperature: float = 0.2,
    max_tokens: int = 256,
    think: Optional[bool] = False,
    keep_alive: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    base_payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "temperature": max(0.0, min(float(temperature), 1.0)),
        },
    }
    if think is not None:
        base_payload["think"] = bool(think)
    normalized_keep_alive = _normalize_keep_alive(keep_alive)
    if normalized_keep_alive is not None:
        base_payload["keep_alive"] = normalized_keep_alive

    def _post_with_tokens(num_predict: int) -> Optional[dict[str, Any]]:
        payload = dict(base_payload)
        payload["options"] = dict(base_payload["options"])
        payload["options"]["num_predict"] = max(64, int(num_predict))
        try:
            resp = requests.post(url, json=payload, timeout=max(1.0, float(timeout_sec)))
            resp.raise_for_status()
            body = resp.json()
        except Exception:
            return None
        if not isinstance(body, dict):
            return None
        return body

    def _parse_body(body: dict[str, Any]) -> tuple[Optional[dict[str, Any]], bool]:
        message = body.get("message")
        if not isinstance(message, dict):
            return None, False
        content = str(message.get("content") or "")
        payload_from_content = _extract_json_payload(content)
        if payload_from_content is not None:
            return payload_from_content, False
        thinking = str(message.get("thinking") or "")
        payload_from_thinking = _extract_json_payload(thinking)
        if payload_from_thinking is not None:
            return payload_from_thinking, False
        done_reason = str(body.get("done_reason") or "").strip().lower()
        should_retry = bool(not content and done_reason == "length" and thinking)
        return None, should_retry

    requested_tokens = max(64, int(max_tokens))
    body = _post_with_tokens(requested_tokens)
    if body is None:
        return None
    parsed, should_retry = _parse_body(body)
    if parsed is not None:
        return parsed
    if not should_retry or float(timeout_sec) < 15.0:
        return None

    boosted_tokens = max(requested_tokens * 2, 2048)
    boosted_tokens = min(boosted_tokens, 4096)
    if boosted_tokens <= requested_tokens:
        return None
    retry_body = _post_with_tokens(boosted_tokens)
    if retry_body is None:
        return None
    retry_parsed, _retry_flag = _parse_body(retry_body)
    return retry_parsed


__all__ = ["call_ollama_chat_json"]
