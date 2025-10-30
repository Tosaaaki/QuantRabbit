from __future__ import annotations

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
class ConfidenceHint:
    scale: float
    confidence: float
    reason: Optional[str] = None
    model_used: Optional[str] = None


class StrategyConfidenceAdvisor:
    """Adjust strategy confidence scaling using GPT guidance."""

    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        cache_ttl_seconds: int = 420,
        min_scale: float = 0.6,
        max_scale: float = 1.3,
    ) -> None:
        self.enabled = _read_bool("STRATEGY_CONF_ADVISOR_ENABLED", True) if enabled is None else enabled
        self._ttl = cache_ttl_seconds
        self.min_scale = min_scale
        self.max_scale = max_scale
        self._cache: Dict[str, tuple[ConfidenceHint, dt.datetime]] = {}
        self._model = os.environ.get("OPENAI_MODEL_STRATEGY_CONF")
        if not self._model:
            try:
                self._model = get_secret("openai_model_strategy_conf")
            except KeyError:
                self._model = "gpt-4o-mini"
        try:
            api_key = get_secret("openai_api_key")
        except KeyError:
            api_key = None
        if not api_key:
            logging.warning("[STRAT_CONF] OPENAI_API_KEY missing; advisor disabled.")
            self.enabled = False
            self._client = None  # type: ignore[assignment]
        else:
            self._client = AsyncOpenAI(api_key=api_key)
        self._max_month_tokens = _int_secret("openai_max_month_tokens", 300000)

    def clear_cache(self) -> None:
        self._cache.clear()

    async def advise(self, strategy: str, context: Dict[str, Any]) -> Optional[ConfidenceHint]:
        if not self.enabled or not self._client:
            return None
        signature = self._hash_context(strategy, context)
        cached = self._cache.get(signature)
        now = dt.datetime.utcnow()
        if cached:
            hint, ts = cached
            if (now - ts).total_seconds() <= self._ttl:
                return hint

        if not add_tokens(0, self._max_month_tokens):
            logging.warning("[STRAT_CONF] token budget exceeded; skip GPT call.")
            return None

        payload = dict(context)
        payload["strategy"] = strategy
        payload.setdefault("pocket", context.get("pocket"))
        try:
            data = await self._call_model(payload)
        except Exception as exc:
            logging.debug("[STRAT_CONF] GPT call failed: %s", exc)
            return None

        hint = self._parse_hint(data)
        if hint:
            self._cache[signature] = (hint, now)
        return hint

    def _hash_context(self, strategy: str, context: Dict[str, Any]) -> str:
        sanitized = {
            k: context[k]
            for k in sorted(context)
            if k not in {"news_short", "news_long"}
        }
        sanitized["strategy"] = strategy
        serialized = json.dumps(sanitized, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(serialized.encode("utf-8")).hexdigest()

    def _parse_hint(self, data: Dict[str, Any]) -> Optional[ConfidenceHint]:
        if not data:
            return None
        try:
            scale = float(data.get("scale", 1.0))
        except (TypeError, ValueError):
            scale = 1.0
        scale = max(self.min_scale, min(self.max_scale, scale))
        try:
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        return ConfidenceHint(
            scale=round(scale, 3),
            confidence=max(0.0, min(1.0, confidence)),
            reason=data.get("reason"),
            model_used=data.get("model_used"),
        )

    async def _call_model(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You evaluate FX trading strategies. "
                    "Given pocket type, technical regime, performance metrics, and range state, "
                    "recommend a multiplicative confidence scale between 0.6 and 1.3. "
                    "Return JSON: {\"scale\": float, \"confidence\": 0-1, \"reason\": str}."
                ),
            },
            {"role": "user", "content": json.dumps(payload)},
        ]
        with track_gpt_call("strategy_confidence_advisor") as tracker:
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
                logging.warning("[STRAT_CONF] invalid JSON response: %s", content)
                tracker.mark_status("invalid_json")
                data = {}
            model_used = getattr(resp, "model", self._model)
            tracker.set_model(model_used)
        data["model_used"] = model_used
        return data
