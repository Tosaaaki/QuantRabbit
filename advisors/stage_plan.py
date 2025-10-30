from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from openai import AsyncOpenAI

from utils.cost_guard import add_tokens
from utils.gpt_monitor import track_gpt_call
from utils.secrets import get_secret


def _read_bool(name: str, default: bool) -> bool:
    env = os.environ.get(name.upper())
    if env is not None:
        return env.strip().lower() not in {"0", "false", "no"}
    try:
        val = get_secret(name.lower())
        return str(val).strip().lower() not in {"0", "false", "no"}
    except KeyError:
        return default


def _int_secret(key: str, default: int) -> int:
    try:
        return int(get_secret(key))
    except Exception:
        return default


@dataclass(frozen=True)
class StagePlanHint:
    plans: Dict[str, Tuple[float, ...]]
    confidence: float
    reason: Optional[str] = None
    model_used: Optional[str] = None


class StagePlanAdvisor:
    """Return pocket-specific stage ratio overrides."""

    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        cache_ttl_seconds: int = 600,
    ) -> None:
        self.enabled = _read_bool("STAGE_PLAN_ADVISOR_ENABLED", True) if enabled is None else enabled
        self._ttl = cache_ttl_seconds
        self._cache: Dict[str, tuple[StagePlanHint, dt.datetime]] = {}
        self._model = os.environ.get("OPENAI_MODEL_STAGE_PLAN")
        if not self._model:
            try:
                self._model = get_secret("openai_model_stage_plan")
            except KeyError:
                self._model = "gpt-4o-mini"
        try:
            api_key = get_secret("openai_api_key")
        except KeyError:
            api_key = None
        if not api_key:
            logging.warning("[STAGE_PLAN] OPENAI_API_KEY missing; advisor disabled.")
            self.enabled = False
            self._client = None  # type: ignore[assignment]
        else:
            self._client = AsyncOpenAI(api_key=api_key)
        self._max_month_tokens = _int_secret("openai_max_month_tokens", 300000)

    def clear_cache(self) -> None:
        self._cache.clear()

    async def advise(self, context: Dict[str, Any]) -> Optional[StagePlanHint]:
        if not self.enabled or not self._client:
            return None
        signature = self._hash_context(context)
        cached = self._cache.get(signature)
        now = dt.datetime.utcnow()
        if cached:
            hint, ts = cached
            if (now - ts).total_seconds() <= self._ttl:
                return hint

        if not add_tokens(0, self._max_month_tokens):
            logging.warning("[STAGE_PLAN] token budget exceeded; skip.")
            return None

        try:
            data = await self._call_model(context)
        except Exception as exc:
            logging.debug("[STAGE_PLAN] GPT call failed: %s", exc)
            return None

        hint = self._parse_hint(data)
        if hint:
            self._cache[signature] = (hint, now)
        return hint

    def _hash_context(self, context: Dict[str, Any]) -> str:
        serialized = json.dumps(context, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(serialized.encode("utf-8")).hexdigest()

    def _parse_hint(self, data: Dict[str, Any]) -> Optional[StagePlanHint]:
        if not data:
            return None
        plans_raw = data.get("plans")
        if not isinstance(plans_raw, dict):
            return None
        plans: Dict[str, Tuple[float, ...]] = {}
        for pocket, seq in plans_raw.items():
            if pocket not in {"macro", "micro", "scalp"}:
                continue
            if not isinstance(seq, (list, tuple)):
                continue
            cleaned = []
            total = 0.0
            for value in seq:
                try:
                    v = float(value)
                except (TypeError, ValueError):
                    continue
                if v <= 0:
                    continue
                cleaned.append(v)
                total += v
            if not cleaned:
                continue
            if total > 0:
                cleaned = [max(0.05, min(0.8, v / total)) for v in cleaned]
            plans[pocket] = tuple(cleaned[:6])
        if not plans:
            return None
        try:
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        return StagePlanHint(
            plans=plans,
            confidence=max(0.0, min(1.0, confidence)),
            reason=data.get("reason"),
            model_used=data.get("model_used"),
        )

    async def _call_model(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You design staged entry ratios for FX pockets. "
                    "Return JSON {\"plans\": {\"macro\": [...], ...}, \"confidence\": float, \"reason\": str}. "
                    "Each plan should have 1-6 positive numbers representing cumulative fractions."
                ),
            },
            {"role": "user", "content": json.dumps(payload)},
        ]
        with track_gpt_call("stage_plan_advisor") as tracker:
            resp = await self._client.responses.create(  # type: ignore[union-attr]
                model=self._model,
                input=messages,
                max_output_tokens=240,
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
                logging.warning("[STAGE_PLAN] invalid JSON response: %s", content)
                tracker.mark_status("invalid_json")
                data = {}
            model_used = getattr(resp, "model", self._model)
            tracker.set_model(model_used)
        data["model_used"] = model_used
        return data
