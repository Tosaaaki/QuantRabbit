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
class FocusHint:
    focus_tag: Optional[str]
    weight_macro: Optional[float]
    weight_scalp: Optional[float]
    confidence: float
    reason: Optional[str] = None
    model_used: Optional[str] = None


class FocusOverrideAdvisor:
    """Suggest focus_tag / weight overrides on critical patterns."""

    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        cache_ttl_seconds: int = 180,
    ) -> None:
        self.enabled = _read_bool("FOCUS_ADVISOR_ENABLED", True) if enabled is None else enabled
        self._ttl = cache_ttl_seconds
        self._cache: Dict[str, tuple[FocusHint, dt.datetime]] = {}
        self._model = (
            os.environ.get("OPENAI_MODEL_FOCUS_ADVISOR")
            or os.environ.get("OPENAI_MODEL")
        )
        if not self._model:
            try:
                self._model = get_secret("openai_model_focus_advisor")
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
            logging.warning("[FOCUS_ADVISOR] OPENAI_API_KEY missing; advisor disabled.")
            self.enabled = False
            self._client = None  # type: ignore[assignment]
        else:
            self._client = AsyncOpenAI(api_key=api_key)
        self._max_month_tokens = _int_secret("openai_max_month_tokens", 300000)

    def clear_cache(self) -> None:
        self._cache.clear()

    async def advise(self, context: Dict[str, Any]) -> Optional[FocusHint]:
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
            logging.warning("[FOCUS_ADVISOR] token budget exceeded; skip.")
            return None

        try:
            data = await self._call_model(context)
        except Exception as exc:
            logging.debug("[FOCUS_ADVISOR] GPT call failed: %s", exc)
            return None

        hint = self._parse_hint(data)
        if hint:
            self._cache[signature] = (hint, now)
        return hint

    def _hash_context(self, context: Dict[str, Any]) -> str:
        sanitized = {
            k: context[k]
            for k in sorted(context)
            if k not in {"news_short", "news_long"}
        }
        serialized = json.dumps(sanitized, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(serialized.encode("utf-8")).hexdigest()

    def _parse_hint(self, data: Dict[str, Any]) -> Optional[FocusHint]:
        if not data:
            return None
        focus = data.get("focus_tag")
        if focus not in {"micro", "macro", "hybrid", "event", None}:
            focus = None
        weight_macro = data.get("weight_macro")
        weight_scalp = data.get("weight_scalp")
        try:
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        macro_val = None
        if weight_macro is not None:
            try:
                macro_val = max(0.0, min(1.0, float(weight_macro)))
            except (TypeError, ValueError):
                macro_val = None
        scalp_val = None
        if weight_scalp is not None:
            try:
                scalp_val = max(0.0, min(1.0, float(weight_scalp)))
            except (TypeError, ValueError):
                scalp_val = None
        return FocusHint(
            focus_tag=focus,
            weight_macro=macro_val,
            weight_scalp=scalp_val,
            confidence=max(0.0, min(1.0, confidence)),
            reason=data.get("reason"),
            model_used=data.get("model_used"),
        )

    async def _call_model(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading focus supervisor. "
                    "Given multi-timeframe indicators, news sentiment, and event flags, "
                    "decide if focus_tag or macro/scalp weights should be overridden. "
                    "Return JSON with keys "
                    "\"focus_tag\" (macro/micro/hybrid/event or null), "
                    "\"weight_macro\", \"weight_scalp\", \"confidence\", \"reason\"."
                ),
            },
            {"role": "user", "content": json.dumps(payload)},
        ]
        with track_gpt_call("focus_override_advisor") as tracker:
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
                logging.warning("[FOCUS_ADVISOR] invalid JSON response: %s", content)
                tracker.mark_status("invalid_json")
                data = {}
            model_used = getattr(resp, "model", self._model)
            tracker.set_model(model_used)
        data["model_used"] = model_used
        return data
