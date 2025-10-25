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
class PartialHint:
    thresholds: Dict[str, Tuple[float, float]]
    confidence: float
    reason: Optional[str] = None
    model_used: Optional[str] = None


class PartialReductionAdvisor:
    """Suggest pocket-specific partial profit thresholds."""

    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        cache_ttl_seconds: int = 420,
    ) -> None:
        self.enabled = _read_bool("PARTIAL_ADVISOR_ENABLED", True) if enabled is None else enabled
        self._ttl = cache_ttl_seconds
        self._cache: Dict[str, tuple[PartialHint, dt.datetime]] = {}
        self._model = (
            os.environ.get("OPENAI_MODEL_PARTIAL_ADVISOR")
            or os.environ.get("OPENAI_MODEL")
        )
        if not self._model:
            try:
                self._model = get_secret("openai_model_partial_advisor")
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
            logging.warning("[PARTIAL_ADVISOR] OPENAI_API_KEY missing; advisor disabled.")
            self.enabled = False
            self._client = None  # type: ignore[assignment]
        else:
            self._client = AsyncOpenAI(api_key=api_key)
        self._max_month_tokens = _int_secret("openai_max_month_tokens", 300000)

    def clear_cache(self) -> None:
        self._cache.clear()

    async def advise(self, context: Dict[str, Any]) -> Optional[PartialHint]:
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
            logging.warning("[PARTIAL_ADVISOR] token budget exceeded; skip.")
            return None

        try:
            data = await self._call_model(context)
        except Exception as exc:
            logging.debug("[PARTIAL_ADVISOR] GPT call failed: %s", exc)
            return None

        hint = self._parse_hint(data)
        if hint:
            self._cache[signature] = (hint, now)
        return hint

    def _hash_context(self, context: Dict[str, Any]) -> str:
        serialized = json.dumps(context, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(serialized.encode("utf-8")).hexdigest()

    def _parse_hint(self, data: Dict[str, Any]) -> Optional[PartialHint]:
        if not data:
            return None
        thresholds_raw = data.get("thresholds")
        if not isinstance(thresholds_raw, dict):
            return None
        result: Dict[str, Tuple[float, float]] = {}
        for pocket, values in thresholds_raw.items():
            if pocket not in {"macro", "micro", "scalp"}:
                continue
            if not isinstance(values, (list, tuple)) or len(values) < 2:
                continue
            try:
                low = max(1.0, min(40.0, float(values[0])))
                high = max(low + 0.5, min(60.0, float(values[1])))
            except (TypeError, ValueError):
                continue
            result[pocket] = (round(low, 2), round(high, 2))
        if not result:
            return None
        try:
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        return PartialHint(
            thresholds=result,
            confidence=max(0.0, min(1.0, confidence)),
            reason=data.get("reason"),
            model_used=data.get("model_used"),
        )

    async def _call_model(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        messages = [
            {
                "role": "system",
                    "content": (
                    "You determine partial take-profit thresholds for FX strategies. "
                    "Return JSON {\"thresholds\": {\"macro\": [low, high], ...}, \"confidence\": float, \"reason\": str}. "
                    "Values are pip gains before scaling."
                ),
            },
            {"role": "user", "content": json.dumps(payload)},
        ]
        resp = await self._client.responses.create(  # type: ignore[union-attr]
            model=self._model,
            input=messages,
            max_output_tokens=220,
            temperature=0.2,
        )
        usage = getattr(resp, "usage", None)
        used_tokens = 0
        if usage is not None:
            used_tokens = getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0)
        if used_tokens:
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
            logging.warning("[PARTIAL_ADVISOR] invalid JSON response: %s", content)
            data = {}
        data["model_used"] = getattr(resp, "model", self._model)
        return data

