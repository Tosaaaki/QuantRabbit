from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from analytics.policy_diff import (
    POLICY_DIFF_SCHEMA,
    aggregate_rows,
    normalize_policy_diff,
    utc_now_iso,
    validate_policy_diff,
)

try:
    import google.auth  # type: ignore
    from google.auth.transport.requests import Request as GoogleAuthRequest  # type: ignore
except Exception:  # pragma: no cover
    google = None  # type: ignore
    GoogleAuthRequest = None  # type: ignore

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

DEFAULT_VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
DEFAULT_VERTEX_MODEL = os.getenv("VERTEX_POLICY_MODEL", "gemini-2.0-flash")
DEFAULT_OPENAI_MODEL = os.getenv("POLICY_SHADOW_MODEL", "gpt-5-mini")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _normalize_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                "pocket": row.get("pocket") or "unknown",
                "strategy": row.get("strategy") or "unknown",
                "regime": row.get("regime") or "unknown",
                "time_band": row.get("time_band") or "unknown",
                "vol_bucket": row.get("vol_bucket") or "unknown",
                "trade_count": _safe_int(row.get("trade_count")),
                "wins": _safe_int(row.get("wins")),
                "gross_profit": _safe_float(row.get("gross_profit")),
                "gross_loss": _safe_float(row.get("gross_loss")),
                "total_pips": _safe_float(row.get("total_pips")),
                "avg_pips": _safe_float(row.get("avg_pips")),
                "avg_hold_minutes": _safe_float(row.get("avg_hold_minutes")),
            }
        )
    return normalized


def _format_agg(rows: List, keys: Tuple[str, ...]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in rows:
        entry = {keys[i]: item.key[i] for i in range(len(keys))}
        entry.update(item.as_summary())
        out.append(entry)
    return out


def _top_bottom(items: List[Dict[str, Any]], *, key: str, n: int) -> Dict[str, List[Dict[str, Any]]]:
    if not items:
        return {"top": [], "bottom": []}
    sorted_items = sorted(items, key=lambda x: x.get(key) or 0.0)
    return {
        "top": sorted_items[-n:][::-1],
        "bottom": sorted_items[:n],
    }


def summarize_policy_rows(rows: Iterable[Dict[str, Any]], *, top_n: int = 6) -> Dict[str, Any]:
    normalized = _normalize_rows(rows)
    pockets = aggregate_rows(normalized, ("pocket",))
    strategies = aggregate_rows(normalized, ("pocket", "strategy"))
    regimes = aggregate_rows(normalized, ("regime",))
    time_bands = aggregate_rows(normalized, ("time_band",))
    vol_buckets = aggregate_rows(normalized, ("vol_bucket",))

    pockets_formatted = _format_agg(pockets, ("pocket",))
    strategies_formatted = _format_agg(strategies, ("pocket", "strategy"))
    regimes_formatted = _format_agg(regimes, ("regime",))
    time_formatted = _format_agg(time_bands, ("time_band",))
    vol_formatted = _format_agg(vol_buckets, ("vol_bucket",))

    return {
        "generated_at": utc_now_iso(),
        "pockets": pockets_formatted,
        "strategies": _top_bottom(strategies_formatted, key="total_pips", n=top_n),
        "regimes": _top_bottom(regimes_formatted, key="total_pips", n=top_n),
        "time_bands": _top_bottom(time_formatted, key="total_pips", n=top_n),
        "vol_buckets": _top_bottom(vol_formatted, key="total_pips", n=top_n),
    }


def _build_prompt(summary: Dict[str, Any]) -> str:
    schema_text = json.dumps(POLICY_DIFF_SCHEMA, ensure_ascii=True, separators=(",", ":"))
    summary_text = json.dumps(summary, ensure_ascii=True, separators=(",", ":"))
    return (
        "You are QuantRabbit's automated policy diff generator.\n"
        "Return ONLY JSON that conforms to this JSON schema:\n"
        f"{schema_text}\n\n"
        "Rules:\n"
        "- If data is insufficient or uncertain, set no_change=true and omit patch.\n"
        "- Keep changes conservative; avoid aggressive risk increases.\n"
        "- Only use keys allowed by the schema.\n\n"
        "Input summary (aggregated metrics):\n"
        f"{summary_text}\n"
    )


def build_policy_prompt(summary: Dict[str, Any]) -> str:
    return _build_prompt(summary)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    raw = text.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except Exception:
        return None


def parse_policy_diff(text: str, *, source: str) -> Optional[Dict[str, Any]]:
    payload = _extract_json(text)
    if not isinstance(payload, dict):
        return None
    payload = normalize_policy_diff(payload, source=source)
    errors = validate_policy_diff(payload)
    if errors:
        logging.warning("[POLICY_GEN] invalid policy_diff: %s", ", ".join(errors))
        return None
    return payload


def _vertex_token(project_id: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if google is None or GoogleAuthRequest is None:
        return None, project_id
    try:
        creds, project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(GoogleAuthRequest())
        return creds.token, project_id or project
    except Exception:
        return None, project_id


def call_vertex_gemini(
    prompt: str,
    *,
    project_id: Optional[str],
    location: str,
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    timeout_sec: float = 20.0,
) -> Optional[str]:
    token, resolved_project = _vertex_token(project_id)
    if not token or not resolved_project:
        logging.warning("[POLICY_GEN] Vertex auth unavailable.")
        return None
    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{resolved_project}"
        f"/locations/{location}/publishers/google/models/{model}:generateContent"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return None
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        if parts and isinstance(parts[0], dict):
            return parts[0].get("text")
        return None
    except Exception as exc:
        logging.warning("[POLICY_GEN] Vertex call failed: %s", exc)
        return None


def call_openai(
    prompt: str,
    *,
    api_key: Optional[str] = None,
    model: str = DEFAULT_OPENAI_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> Optional[str]:
    if OpenAI is None:
        logging.warning("[POLICY_GEN] openai package not available.")
        return None
    try:
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        text = getattr(resp, "output_text", None)
        if text:
            return text
        # fallback parse
        output = getattr(resp, "output", None) or []
        if output:
            content = output[0].content or []
            if content:
                return content[0].text
        return None
    except Exception as exc:
        logging.warning("[POLICY_GEN] OpenAI call failed: %s", exc)
        return None


def heuristic_policy_diff(
    summary: Dict[str, Any],
    *,
    pf_min: float = 0.9,
    win_min: float = 0.48,
    min_trades: int = 12,
) -> Dict[str, Any]:
    if not _env_bool("POLICY_HEURISTIC_PERF_BLOCK_ENABLED", True):
        return {
            "policy_id": f"heuristic-{int(time.time())}",
            "generated_at": utc_now_iso(),
            "source": "heuristic",
            "no_change": True,
            "reason": "perf_block_disabled",
            "notes": {"perf_block_enabled": False},
        }
    pockets = summary.get("pockets") or []
    patch: Dict[str, Any] = {}
    notes: Dict[str, Any] = {}
    blocked: List[str] = []
    for item in pockets:
        pocket = str(item.get("pocket") or "unknown").lower()
        trades = _safe_int(item.get("trade_count"))
        pf = _safe_float(item.get("profit_factor"))
        win_rate = _safe_float(item.get("win_rate"))
        if trades < min_trades:
            continue
        if pf < pf_min and win_rate < win_min:
            blocked.append(pocket)
            patch.setdefault("pockets", {}).setdefault(pocket, {}).setdefault("entry_gates", {})[
                "allow_new"
            ] = False

    if blocked:
        notes["heuristic_blocked_pockets"] = blocked

    if patch:
        return {
            "policy_id": f"heuristic-{int(time.time())}",
            "generated_at": utc_now_iso(),
            "source": "heuristic",
            "no_change": False,
            "reason": "auto_block_low_pf_pockets",
            "patch": patch,
            "notes": notes,
        }
    return {
        "policy_id": f"heuristic-{int(time.time())}",
        "generated_at": utc_now_iso(),
        "source": "heuristic",
        "no_change": True,
        "reason": "no_change",
        "notes": notes,
    }


def generate_policy_diff(
    rows: Iterable[Dict[str, Any]],
    *,
    use_vertex: bool = True,
    project_id: Optional[str] = None,
    location: str = DEFAULT_VERTEX_LOCATION,
    model: str = DEFAULT_VERTEX_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    fallback_min_trades: int = 12,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    summary = summarize_policy_rows(rows)
    prompt = _build_prompt(summary)
    if use_vertex:
        text = call_vertex_gemini(
            prompt,
            project_id=project_id,
            location=location,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if text:
            payload = parse_policy_diff(text, source="vertex_ai")
            if payload:
                return payload, summary
    diff = heuristic_policy_diff(summary, min_trades=fallback_min_trades)
    diff = normalize_policy_diff(diff, source="heuristic")
    return diff, summary
