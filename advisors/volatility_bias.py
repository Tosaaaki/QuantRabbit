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
class VolatilityHint:
    bias: float
    confidence: float
    reason: Optional[str] = None
    model_used: Optional[str] = None


class VolatilityBiasAdvisor:
    """Suggest small adjustments to risk appetite based on GPT analysis."""

    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        cache_ttl_seconds: int = 240,
        min_bias: float = -0.2,
        max_bias: float = 0.2,
    ) -> None:
        self.enabled = _read_bool("VOLATILITY_ADVISOR_ENABLED", True) if enabled is None else enabled
        self._ttl = cache_ttl_seconds
        self.min_bias = min_bias
        self.max_bias = max_bias
        self._cache: Dict[str, tuple[VolatilityHint, dt.datetime]] = {}
        self._model = os.environ.get("OPENAI_MODEL_VOLATILITY")
        if not self._model:
            try:
                self._model = get_secret("openai_model_volatility")
            except KeyError:
                self._model = "gpt-4o-mini"
        try:
            api_key = get_secret("openai_api_key")
        except KeyError:
            api_key = None
        if not api_key:
            logging.warning("[VOL_ADVISOR] OPENAI_API_KEY missing; advisor disabled.")
            self.enabled = False
            self._client = None  # type: ignore[assignment]
        else:
            self._client = AsyncOpenAI(api_key=api_key)
        self._max_month_tokens = _int_secret("openai_max_month_tokens", 300000)

    def clear_cache(self) -> None:
        self._cache.clear()

    async def advise(self, context: Dict[str, Any]) -> Optional[VolatilityHint]:
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
            logging.warning("[VOL_ADVISOR] token budget exceeded; skip.")
            return None

        try:
            data = await self._call_model(context)
        except Exception as exc:
            logging.debug("[VOL_ADVISOR] GPT call failed: %s", exc)
            return None

        hint = self._parse_hint(data)
        if hint:
            self._cache[signature] = (hint, now)
        return hint

    def _hash_context(self, context: Dict[str, Any]) -> str:
        serialized = json.dumps(context, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(serialized.encode("utf-8")).hexdigest()

    def _parse_hint(self, data: Dict[str, Any]) -> Optional[VolatilityHint]:
        if not data:
            return None
        try:
            bias = float(data.get("bias", 0.0))
        except (TypeError, ValueError):
            bias = 0.0
        bias = max(self.min_bias, min(self.max_bias, bias))
        try:
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        return VolatilityHint(
            bias=round(bias, 3),
            confidence=max(0.0, min(1.0, confidence)),
            reason=data.get("reason"),
            model_used=data.get("model_used"),
        )

    async def _call_model(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        def _build_messages(ctx: Dict[str, Any]) -> list[Dict[str, str]]:
            system = (
                "You adjust risk appetite based on volatility and liquidity context. "
                "Respond ONLY with a single JSON object with exactly these keys: "
                "bias (number in [-0.2, 0.2]), confidence (number in [0, 1]), reason (string <= 140 chars). "
                "No markdown, no code fences, no extra text, no extra keys."
            )
            return [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(ctx, separators=(",", ":"))},
            ]

        async def _invoke(messages: list[Dict[str, str]]) -> tuple[Dict[str, Any], int, str]:
            base_kwargs: Dict[str, Any] = {
                "model": self._model,
                "input": messages,
                "max_output_tokens": 140,
                "temperature": 0.0,
            }
            resp = None
            last_error: Optional[Exception] = None
            # Try with response_format forcing JSON, then fallback if unsupported
            for extra in ({"response_format": {"type": "json_object"}}, {}):
                try:
                    resp = await self._client.responses.create(  # type: ignore[union-attr]
                        **base_kwargs, **extra
                    )
                    break
                except Exception as exc:
                    msg = str(exc)
                    last_error = exc
                    if "Unsupported parameter" in msg or "response_format" in msg:
                        continue
                    raise
            if resp is None:
                raise last_error or RuntimeError("failed to call OpenAI responses API")
            usage = getattr(resp, "usage", None)
            used = 0
            if usage is not None:
                used = getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0)
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
                data = {}
            return data, used, getattr(resp, "model", self._model)

        messages = _build_messages(payload)
        with track_gpt_call("volatility_bias_advisor") as tracker:
            data, used_tokens, model_used = await _invoke(messages)
            if used_tokens:
                tracker.add_tag("tokens", used_tokens)
                add_tokens(used_tokens, self._max_month_tokens)
            if not data or not isinstance(data, dict) or "bias" not in data:
                # One lightweight retry with explicit reminder
                retry_msgs = messages + [
                    {
                        "role": "user",
                        "content": (
                            "The previous response was invalid JSON. "
                            "Return ONLY a JSON object matching the constraints; no markdown, no code fences."
                        ),
                    }
                ]
                data2, used2, model2 = await _invoke(retry_msgs)
                if used2:
                    tracker.add_tag("tokens", (used_tokens or 0) + used2)
                    add_tokens(used2, self._max_month_tokens)
                if data2 and isinstance(data2, dict) and "bias" in data2:
                    data, model_used = data2, model2
                else:
                    logging.warning("[VOL_ADVISOR] invalid JSON response after retry")
                    tracker.mark_status("invalid_json")
                    data = {"bias": 0.0, "confidence": 0.5, "reason": "fallback"}
            tracker.set_model(model_used)
        data["model_used"] = model_used
        return data
