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


def call_ollama_chat_json(
    prompt: str,
    *,
    model: str,
    url: str,
    timeout_sec: float,
    temperature: float = 0.2,
    max_tokens: int = 256,
) -> Optional[dict[str, Any]]:
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "temperature": max(0.0, min(float(temperature), 1.0)),
            "num_predict": max(64, int(max_tokens)),
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=max(1.0, float(timeout_sec)))
        resp.raise_for_status()
        body = resp.json()
    except Exception:
        return None

    message = body.get("message")
    if not isinstance(message, dict):
        return None
    content = str(message.get("content") or "")
    return _extract_json_payload(content)


__all__ = ["call_ollama_chat_json"]
