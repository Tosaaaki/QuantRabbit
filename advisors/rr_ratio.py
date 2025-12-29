from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from openai import AsyncOpenAI

from utils.cost_guard import add_tokens
from utils.gpt_monitor import track_gpt_call
from utils.secrets import get_secret


def _read_bool(name: str, default: bool) -> bool:
    env_key = name.upper()
    raw = os.environ.get(env_key)
    if raw is not None:
        return raw.strip().lower() not in {"0", "false", "no"}
    try:
        val = get_secret(name.lower())
        return str(val).strip().lower() not in {"0", "false", "no"}
    except KeyError:
        return default


def _safe_int_secret(key: str, default: int) -> int:
    try:
        return int(get_secret(key))
    except Exception:
        return default


@dataclass(frozen=True)
class RRHint:
    ratio: float
    confidence: float
    reason: Optional[str] = None
    model_used: Optional[str] = None


class RRRatioAdvisor:
    """Suggest a target RR (tp/sl ratio) per signal."""

    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        min_ratio: float = 0.9,
        max_ratio: float = 1.8,
        cache_ttl_seconds: int = 420,
    ) -> None:
        self.enabled = _read_bool("RR_ADVISOR_ENABLED", True) if enabled is None else enabled
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self._ttl = cache_ttl_seconds
        self._cache: Dict[str, tuple[RRHint, dt.datetime]] = {}
        self._model = (
            os.environ.get("OPENAI_MODEL_RR_ADVISOR")
            or os.environ.get("OPENAI_MODEL")
        )
        if not self._model:
            try:
                self._model = get_secret("openai_model_rr_advisor")
            except KeyError:
                try:
                    self._model = get_secret("openai_model")
                except KeyError:
                    self._model = "gpt-4o-mini"
        try:
            api_key = get_secret("openai_api_key")
        except KeyError:
            api_key = None
        if not api_key:
            logging.warning("[RR_ADVISOR] OPENAI_API_KEY missing; advisor disabled.")
            self.enabled = False
            self._client = None  # type: ignore[assignment]
        else:
            self._client = AsyncOpenAI(api_key=api_key)
        self._max_month_tokens = _safe_int_secret("openai_max_month_tokens", 300000)

    def clear_cache(self) -> None:
        self._cache.clear()

    async def advise(self, context: Dict[str, Any]) -> Optional[RRHint]:
        if not self.enabled or not self._client:
            return None
        sl_pips = float(context.get("sl_pips") or 0.0)
        if sl_pips <= 0:
            return None
        signature = self._hash_context(context)
        cached = self._cache.get(signature)
        now = dt.datetime.utcnow()
        if cached:
            hint, ts = cached
            if (now - ts).total_seconds() <= self._ttl:
                return hint

        if not add_tokens(0, self._max_month_tokens):
            logging.warning("[RR_ADVISOR] token budget exceeded; skip GPT call.")
            return None

        payload = self._build_payload(context)
        try:
            model_output = await self._call_model(payload)
        except Exception as exc:
            logging.warning("[RR_ADVISOR] GPT call failed: %s", exc)
            return None

        ratio = self._clamp_ratio(model_output.get("rr"))
        confidence = float(model_output.get("confidence") or 0.5)
        reason = model_output.get("reason") or model_output.get("rationale")
        hint = RRHint(
            ratio=ratio,
            confidence=max(0.0, min(confidence, 1.0)),
            reason=reason,
            model_used=model_output.get("model_used"),
        )
        self._cache[signature] = (hint, now)
        return hint

    def _clamp_ratio(self, value: Any) -> float:
        try:
            ratio = float(value)
        except (TypeError, ValueError):
            ratio = 1.2
        return round(max(self.min_ratio, min(self.max_ratio, ratio)), 2)

    def _hash_context(self, context: Dict[str, Any]) -> str:
        serialized = json.dumps(context, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(serialized.encode("utf-8")).hexdigest()

    def _build_payload(self, context: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(context)
        return payload

    async def _call_model(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an FX risk advisor. "
                    "Given stop-loss width, technical context, and pocket type, "
                    "recommend a take-profit ratio (tp/sl). "
                    "Respond ONLY with JSON: "
                    "{"
                    "\"rr\": float between 0.8 and 2.0, "
                    "\"confidence\": 0-1, "
                    "\"reason\": short explanation"
                    "}."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload),
            },
        ]
        with track_gpt_call("rr_ratio_advisor") as tracker:
            resp = await self._client.responses.create(  # type: ignore[union-attr]
                model=self._model,
                input=messages,
                max_output_tokens=200,
                temperature=0.2,
            )
            usage = getattr(resp, "usage", None)
            used_tokens = 0
            if usage is not None:
                used_tokens = getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0)
            if used_tokens:
                tracker.add_tag("tokens", used_tokens)
                add_tokens(used_tokens, self._max_month_tokens)
            content_parts: list[str] = []
            for item in resp.output or []:
                if getattr(item, "type", "") != "message":
                    continue
                for block in getattr(item, "content", []) or []:
                    text = getattr(block, "text", None)
                    if text:
                        content_parts.append(text)
            content = "".join(content_parts).strip()
            if content.startswith("```"):
                content = content.strip("`")
                content = content.replace("json", "", 1).strip()
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                logging.warning("[RR_ADVISOR] Invalid JSON response: %s", content)
                tracker.mark_status("invalid_json")
                data = {}
            model_used = getattr(resp, "model", self._model)
            tracker.set_model(model_used)
        data["model_used"] = model_used
        return data
