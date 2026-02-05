"""
LLM-backed decision brain for per-strategy entry gating.
Each strategy_tag+pocket pair keeps its own memory and decision cache.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import random
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from utils.vertex_client import call_vertex_text

LOG = logging.getLogger(__name__)

_DB_PATH = pathlib.Path("logs/brain_state.db")
_DB_TIMEOUT = float(os.getenv("BRAIN_DB_TIMEOUT_SEC", "3.0"))

_ENABLED = os.getenv("BRAIN_ENABLED", "0").strip().lower() not in {"", "0", "false", "no", "off"}
_ALLOWLIST = {item.strip().lower() for item in os.getenv("BRAIN_STRATEGY_ALLOWLIST", "").split(",") if item.strip()}
_POCKET_ALLOWLIST = {item.strip().lower() for item in os.getenv("BRAIN_POCKET_ALLOWLIST", "").split(",") if item.strip()}
_SAMPLE_RATE = max(0.0, min(1.0, float(os.getenv("BRAIN_SAMPLE_RATE", "1.0") or 1.0)))

_TTL_SEC = max(5.0, float(os.getenv("BRAIN_TTL_SEC", "90") or 90.0))
_MEMORY_TTL_H = max(1.0, float(os.getenv("BRAIN_MEMORY_TTL_H", "72") or 72.0))
_MAX_CONTEXT_CHARS = max(200, int(float(os.getenv("BRAIN_MAX_CONTEXT_CHARS", "1200") or 1200)))
_MIN_SCALE = max(0.05, float(os.getenv("BRAIN_MIN_SCALE", "0.2") or 0.2))

_MODEL = os.getenv("BRAIN_VERTEX_MODEL", "") or os.getenv("VERTEX_DECIDER_MODEL") or os.getenv("VERTEX_MODEL") or "gemini-2.0-flash"
_TEMP = max(0.0, min(1.0, float(os.getenv("BRAIN_TEMPERATURE", "0.2") or 0.2)))
_MAX_TOKENS = max(64, int(float(os.getenv("BRAIN_MAX_TOKENS", "256") or 256)))
_TIMEOUT_SEC = max(2.0, float(os.getenv("BRAIN_TIMEOUT_SEC", "6") or 6))

_PERSONA_ENABLED = os.getenv("BRAIN_PERSONA_ENABLED", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
    "off",
}
_PERSONA_MODE = (os.getenv("BRAIN_PERSONA_MODE", "auto") or "auto").strip().lower()
_PERSONA_DEFAULT = os.getenv(
    "BRAIN_PERSONA_DEFAULT",
    "Seasoned discretionary FX trader focused on risk-first entries.",
)
_RAW_PERSONA_OVERRIDES = os.getenv("BRAIN_PERSONA_OVERRIDES", "")

_POCKET_TRAITS = {
    "macro": "Patient macro trader; values regime alignment and avoids noise.",
    "micro": "Tactical micro trader; values precise timing and quick invalidation.",
    "scalp": "High-speed scalper; avoids hesitation and rejects low edge setups.",
    "scalp_fast": "Ultra-fast scalp; only highest edge entries, minimal tolerance.",
}

_PERSONA_PRESETS = {
    "trend": {
        "name": "Trend Hunter",
        "traits": "Prefers strong directional alignment and momentum follow-through.",
        "bias": "Allow only when trend evidence is clear; otherwise reduce or block.",
        "avoid": "Avoid mean-reversion entries or weak trend conditions.",
    },
    "momentum": {
        "name": "Momentum Specialist",
        "traits": "Enters on acceleration and continuation, avoids fading moves.",
        "bias": "Allow only when acceleration/impulse is present.",
        "avoid": "Avoid chop, fading, or low-volatility setups.",
    },
    "range": {
        "name": "Range Fader",
        "traits": "Trades mean reversion; seeks stretched moves back to mean.",
        "bias": "Allow only when range/mean-revert signals are present.",
        "avoid": "Avoid breakout or strong trend continuation entries.",
    },
    "pullback": {
        "name": "Pullback Sniper",
        "traits": "Waits for retrace to high-probability zones in trend.",
        "bias": "Allow only when pullback quality is clear; otherwise reduce.",
        "avoid": "Avoid chasing late entries or shallow/noisy pullbacks.",
    },
    "scalp": {
        "name": "Precision Scalper",
        "traits": "Seeks tight execution; rejects anything ambiguous.",
        "bias": "Prefer reduce/block if spread or context is uncertain.",
        "avoid": "Avoid slow or wide-spread entries.",
    },
    "neutral": {
        "name": "Balanced Trader",
        "traits": "Risk-first, selective, avoids low edge entries.",
        "bias": "If uncertain, reduce or block.",
        "avoid": "Avoid conflicting signals.",
    },
}

_CACHE: dict[tuple[str, str], tuple[float, "BrainDecision"]] = {}


@dataclass(frozen=True)
class BrainDecision:
    allowed: bool
    scale: float
    reason: str
    action: str
    memory: Optional[str] = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_schema() -> None:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(_DB_PATH, timeout=_DB_TIMEOUT)
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS brain_memory (
                strategy_tag TEXT NOT NULL,
                pocket TEXT NOT NULL,
                memory TEXT,
                last_action TEXT,
                last_scale REAL,
                last_reason TEXT,
                updated_at TEXT,
                last_ts REAL,
                PRIMARY KEY (strategy_tag, pocket)
            )
            """
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_brain_memory_updated ON brain_memory(updated_at)"
        )
        con.commit()
    finally:
        try:
            con.close()
        except Exception:
            pass


def _should_use(strategy_tag: Optional[str], pocket: Optional[str]) -> bool:
    if not _ENABLED:
        return False
    if not strategy_tag or not pocket:
        return False
    if _ALLOWLIST:
        tag_key = strategy_tag.strip().lower()
        base = tag_key.split("-", 1)[0]
        if tag_key not in _ALLOWLIST and base not in _ALLOWLIST:
            return False
    if _POCKET_ALLOWLIST:
        if pocket.strip().lower() not in _POCKET_ALLOWLIST:
            return False
    if _SAMPLE_RATE < 1.0 and random.random() > _SAMPLE_RATE:
        return False
    return True


def _load_memory(strategy_tag: str, pocket: str) -> Optional[str]:
    if not _DB_PATH.exists():
        return None
    cutoff = time.time() - (_MEMORY_TTL_H * 3600.0)
    con = sqlite3.connect(_DB_PATH, timeout=_DB_TIMEOUT)
    try:
        row = con.execute(
            """
            SELECT memory, last_ts
            FROM brain_memory
            WHERE strategy_tag = ? AND pocket = ?
            """,
            (strategy_tag, pocket),
        ).fetchone()
    except Exception:
        row = None
    finally:
        try:
            con.close()
        except Exception:
            pass
    if not row:
        return None
    memory, last_ts = row
    try:
        ts = float(last_ts or 0.0)
    except Exception:
        ts = 0.0
    if ts <= 0 or ts < cutoff:
        return None
    if not memory:
        return None
    return str(memory)


def _save_memory(
    strategy_tag: str,
    pocket: str,
    *,
    memory: Optional[str],
    decision: BrainDecision,
) -> None:
    _ensure_schema()
    con = sqlite3.connect(_DB_PATH, timeout=_DB_TIMEOUT)
    try:
        con.execute(
            """
            INSERT INTO brain_memory(strategy_tag, pocket, memory, last_action, last_scale, last_reason, updated_at, last_ts)
            VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(strategy_tag, pocket) DO UPDATE SET
                memory=excluded.memory,
                last_action=excluded.last_action,
                last_scale=excluded.last_scale,
                last_reason=excluded.last_reason,
                updated_at=excluded.updated_at,
                last_ts=excluded.last_ts
            """,
            (
                strategy_tag,
                pocket,
                memory,
                decision.action,
                float(decision.scale),
                decision.reason,
                _now_iso(),
                time.time(),
            ),
        )
        con.commit()
    except Exception:
        pass
    finally:
        try:
            con.close()
        except Exception:
            pass


def _stringify(obj: Any, limit: int) -> str:
    if obj is None:
        return ""
    try:
        text = json.dumps(obj, ensure_ascii=True, default=str)
    except Exception:
        text = str(obj)
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _load_persona_overrides() -> dict[str, Any]:
    if not _RAW_PERSONA_OVERRIDES:
        return {}
    try:
        data = json.loads(_RAW_PERSONA_OVERRIDES)
    except Exception:
        return {}
    if isinstance(data, dict):
        return {str(k).strip().lower(): v for k, v in data.items()}
    return {}


_PERSONA_OVERRIDES = _load_persona_overrides()


def _normalize_tag(tag: str) -> str:
    text = str(tag or "").strip().lower()
    if not text:
        return ""
    base = text.split("-", 1)[0].split("_", 1)[0]
    return base or text


def _select_persona_key(tag: str) -> str:
    text = str(tag or "").strip().lower()
    if not text:
        return "neutral"
    tokens = [t for t in text.replace("-", "_").split("_") if t]
    token_set = set(tokens)
    if "range" in token_set or "revert" in token_set or "reversion" in token_set or "fade" in token_set:
        return "range"
    if "bbrsi" in token_set or "bb" in token_set or "rsi" in token_set:
        return "range"
    if "vwap" in token_set or "levelreactor" in token_set or "magnet" in token_set:
        return "range"
    if "pullback" in token_set or "retest" in token_set:
        return "pullback"
    if "trend" in token_set or "donchian" in token_set or "breakout" in token_set:
        return "trend"
    if "momentum" in token_set or "impulse" in token_set or "burst" in token_set or "squeeze" in token_set:
        return "momentum"
    if "scalp" in token_set or "m1scalper" in token_set:
        return "scalp"
    return "neutral"


def _persona_text(strategy_tag: str, pocket: str) -> str:
    if not _PERSONA_ENABLED:
        return _PERSONA_DEFAULT
    if _PERSONA_MODE == "uniform":
        return _PERSONA_DEFAULT
    tag_key = _normalize_tag(strategy_tag)
    # Overrides can be string or dict with fields.
    override = _PERSONA_OVERRIDES.get(tag_key) or _PERSONA_OVERRIDES.get(strategy_tag.strip().lower())
    if override:
        if isinstance(override, str):
            return override
        if isinstance(override, dict):
            name = str(override.get("name") or "").strip()
            traits = str(override.get("traits") or "").strip()
            bias = str(override.get("bias") or "").strip()
            avoid = str(override.get("avoid") or "").strip()
            parts = [p for p in (name, traits, bias, avoid) if p]
            if parts:
                return " ".join(parts)
    key = _select_persona_key(strategy_tag)
    profile = _PERSONA_PRESETS.get(key, _PERSONA_PRESETS["neutral"])
    pocket_trait = _POCKET_TRAITS.get(pocket.strip().lower(), "")
    parts = [
        profile.get("name", ""),
        profile.get("traits", ""),
        profile.get("bias", ""),
        profile.get("avoid", ""),
        pocket_trait,
    ]
    return " ".join([p for p in parts if p])


def _build_prompt(context: dict[str, Any]) -> str:
    strategy_tag = str(context.get("strategy_tag") or "")
    pocket = str(context.get("pocket") or "")
    persona = _persona_text(strategy_tag, pocket)
    ctx_text = _stringify(context, _MAX_CONTEXT_CHARS)
    return (
        "You are the decision brain for an automated USD/JPY trading worker. "
        "Decide whether to allow, reduce, or block a single entry candidate. "
        "Respond with JSON only.\n\n"
        f"Persona: {persona}\n\n"
        "Rules:\n"
        "- action must be one of: ALLOW, REDUCE, BLOCK.\n"
        "- scale must be between 0.2 and 1.0 (never > 1.0). Use 1.0 for ALLOW.\n"
        "- Prefer blocking on uncertainty or missing context.\n"
        "- memory_update must be a short summary (<=200 chars) or empty string to keep memory.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "action": "ALLOW|REDUCE|BLOCK",\n'
        '  "scale": 0.2,\n'
        '  "reason": "short reason",\n'
        '  "memory_update": "short memory or empty"\n'
        "}\n\n"
        "Context:\n"
        f"{ctx_text}\n"
    )


def _parse_response(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    # try to extract JSON object from text
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return None
    return None


def decide(
    *,
    strategy_tag: Optional[str],
    pocket: Optional[str],
    side: str,
    units: int,
    sl_price: Optional[float] = None,
    tp_price: Optional[float] = None,
    entry_thesis: Optional[dict] = None,
    meta: Optional[dict] = None,
    confidence: Optional[int] = None,
) -> BrainDecision:
    if not _should_use(strategy_tag, pocket):
        return BrainDecision(True, 1.0, "disabled", "ALLOW")
    tag = str(strategy_tag).strip()
    pocket_key = str(pocket).strip().lower()
    cache_key = (tag, pocket_key)
    now = time.monotonic()
    cached = _CACHE.get(cache_key)
    if cached and now - cached[0] <= _TTL_SEC:
        return cached[1]

    memory = _load_memory(tag, pocket_key)
    context = {
        "ts": _now_iso(),
        "strategy_tag": tag,
        "pocket": pocket_key,
        "side": side,
        "units": int(units),
        "sl_price": sl_price,
        "tp_price": tp_price,
        "confidence": confidence,
        "memory": memory or "",
        "entry_thesis": entry_thesis or {},
        "meta": meta or {},
    }
    prompt = _build_prompt(context)
    resp = call_vertex_text(
        prompt,
        model=_MODEL,
        temperature=_TEMP,
        max_tokens=_MAX_TOKENS,
        timeout_sec=_TIMEOUT_SEC,
        response_mime_type="application/json",
    )
    if resp is None or not resp.text:
        decision = BrainDecision(True, 1.0, "no_llm", "ALLOW", memory=memory)
        _CACHE[cache_key] = (now, decision)
        return decision

    payload = _parse_response(resp.text)
    if not isinstance(payload, dict):
        decision = BrainDecision(True, 1.0, "bad_response", "ALLOW", memory=memory)
        _CACHE[cache_key] = (now, decision)
        return decision

    action = str(payload.get("action") or "ALLOW").strip().upper()
    reason = str(payload.get("reason") or "").strip()
    if not reason:
        reason = "llm_decision"
    try:
        scale = float(payload.get("scale", 1.0))
    except Exception:
        scale = 1.0
    scale = max(_MIN_SCALE, min(scale, 1.0))

    if action not in {"ALLOW", "REDUCE", "BLOCK"}:
        action = "ALLOW"
    if action == "BLOCK":
        allowed = False
        scale = 0.0
    elif action == "REDUCE":
        allowed = True
    else:
        allowed = True
        if scale < 1.0:
            action = "REDUCE"

    memory_update = str(payload.get("memory_update") or "").strip()
    if memory_update:
        if len(memory_update) > 200:
            memory_update = memory_update[:200] + "..."
        memory = memory_update

    decision = BrainDecision(allowed, scale, reason, action, memory=memory)
    if memory_update:
        _save_memory(tag, pocket_key, memory=memory, decision=decision)
    else:
        # still persist decision metadata for observability
        _save_memory(tag, pocket_key, memory=memory, decision=decision)

    _CACHE[cache_key] = (now, decision)
    return decision


__all__ = ["BrainDecision", "decide"]
