from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests

try:
    import google.auth  # type: ignore
    from google.auth.transport.requests import Request as GoogleAuthRequest  # type: ignore
except Exception:  # pragma: no cover
    google = None  # type: ignore
    GoogleAuthRequest = None  # type: ignore


DEFAULT_VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", os.getenv("GCP_LOCATION", "us-central1"))
DEFAULT_VERTEX_MODEL = os.getenv("VERTEX_MODEL", "")
DEFAULT_TIMEOUT_SEC = float(os.getenv("VERTEX_TIMEOUT_SEC", "20"))

_TOKEN_LOCK = threading.Lock()
_TOKEN: Optional[str] = None
_TOKEN_EXP: float = 0.0
_TOKEN_PROJECT: Optional[str] = None


@dataclass
class VertexResponse:
    text: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


def _refresh_token(project_id: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if google is None or GoogleAuthRequest is None:
        return None, project_id
    creds, project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(GoogleAuthRequest())
    token = creds.token
    exp = time.time() + 300
    try:
        if creds.expiry is not None:
            exp = creds.expiry.timestamp()
    except Exception:
        pass
    with _TOKEN_LOCK:
        global _TOKEN, _TOKEN_EXP, _TOKEN_PROJECT
        _TOKEN = token
        _TOKEN_EXP = exp
        _TOKEN_PROJECT = project
    return token, project_id or project


def _get_token(project_id: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    with _TOKEN_LOCK:
        if _TOKEN and time.time() < _TOKEN_EXP - 60:
            return _TOKEN, project_id or _TOKEN_PROJECT
    return _refresh_token(project_id)


def _extract_text(payload: Dict[str, Any]) -> Optional[str]:
    candidates = payload.get("candidates") or []
    if not candidates:
        return None
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    if not parts:
        return None
    part = parts[0]
    if isinstance(part, dict):
        return part.get("text")
    if isinstance(part, str):
        return part
    return None


def call_vertex_text(
    prompt: str,
    *,
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
    timeout_sec: Optional[float] = None,
    response_mime_type: Optional[str] = None,
) -> Optional[VertexResponse]:
    resolved_model = model or DEFAULT_VERTEX_MODEL or os.getenv("VERTEX_POLICY_MODEL") or "gemini-2.0-flash"
    resolved_location = location or DEFAULT_VERTEX_LOCATION
    resolved_project = (
        project_id
        or os.getenv("VERTEX_PROJECT_ID")
        or os.getenv("GCP_PROJECT")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
    )
    token, project = _get_token(resolved_project)
    if not token or not project:
        logging.warning("[VERTEX] auth unavailable.")
        return None

    url = (
        f"https://{resolved_location}-aiplatform.googleapis.com/v1/projects/{project}"
        f"/locations/{resolved_location}/publishers/google/models/{resolved_model}:generateContent"
    )
    generation: Dict[str, Any] = {
        "temperature": temperature,
        "maxOutputTokens": int(max_tokens),
    }
    if response_mime_type:
        generation["responseMimeType"] = response_mime_type
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation,
    }
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout_sec or DEFAULT_TIMEOUT_SEC)
        resp.raise_for_status()
        data = resp.json()
        text = _extract_text(data)
        if not text:
            return None
        usage = data.get("usageMetadata") or {}
        return VertexResponse(
            text=text,
            prompt_tokens=int(usage.get("promptTokenCount") or 0),
            output_tokens=int(usage.get("candidatesTokenCount") or 0),
            total_tokens=int(usage.get("totalTokenCount") or 0),
            model=data.get("modelVersion") or resolved_model,
            raw=data,
        )
    except Exception as exc:
        logging.warning("[VERTEX] call failed: %s", exc)
        return None
