"""
analysis.gpt_decider
~~~~~~~~~~~~~~~~~~~~
OpenAI API を呼び出し、JSON スキーマ検証・
フォールバック・トークン課金記録までを一括で行うハイレベル関数。
"""

from __future__ import annotations

import os
import asyncio
import datetime as dt
import json
import logging
import os
import random
import re
import time
from typing import Dict, List, Optional

from openai import AsyncOpenAI

from analysis.gpt_prompter import (
    build_messages,
    OPENAI_MODEL as MODEL,
    MAX_TOKENS_MONTH,
)
from analysis.local_decider import heuristic_decision
from utils.cost_guard import add_tokens
from utils.gpt_monitor import log_gpt_fallback, track_gpt_call
from utils.secrets import get_secret

logger = logging.getLogger(__name__)

_LLM_MODE = os.getenv("LLM_MODE", "").strip().lower()
_DUMMY_MODES = {"dummy", "mock", "offline", "test"}

_CLIENT: AsyncOpenAI | None = None
_FENCE_PREFIX_RE = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)


def _get_openai_client() -> AsyncOpenAI:
    global _CLIENT
    if _CLIENT is None:
        api_key = get_secret("openai_api_key")
        _CLIENT = AsyncOpenAI(api_key=api_key)
    return _CLIENT


# GPT の出力は「モード/バイアス+フォーカス/ウェイト」が最小限。
# ranked_strategies はローカルで決めるため必須ではない。
_REQUIRED_KEYS = (
    "focus_tag",
    "weight_macro",
)

_FOCUS_TAGS = {"micro", "macro", "hybrid", "event"}
_ALLOWED_MODES = {"DEFENSIVE", "TREND_FOLLOW", "RANGE_SCALP", "TRANSITION"}
_ALLOWED_RISK = {"high", "neutral", "low"}
_ALLOWED_LIQ = {"tight", "normal", "loose"}

_MAX_COMPLETION_TOKENS = 200
_GPT5_MAX_OUTPUT_TOKENS = 256
_MODEL_OUTPUT_LIMITS = {
    "gpt-5-mini": 256,
    "gpt-5-mini-2025-08-07": 256,
    "gpt-5.1-mini": 256,
    "gpt-4o-mini": 220,
    "gpt-4o-mini-2024-07-18": 220,
}

_MODEL_TIMEOUT_SECONDS = {
    "gpt-5-mini": 20,
    "gpt-5-mini-2025-08-07": 20,
    "gpt-5.1-mini": 20,
    "gpt-4o-mini": 15,
    "gpt-4o-mini-2024-07-18": 15,
}

# フォールバックモデルは明示的に無効化（primary のみを使用）
_FALLBACK_MODELS: list[str] = []

_MAX_MODEL_ATTEMPTS = max(1, int(os.getenv("GPT_MAX_MODEL_ATTEMPTS", "4")))
_RETRY_BASE_SEC = max(0.1, float(os.getenv("GPT_RETRY_BASE_SEC", "0.6")))
_RETRY_JITTER_SEC = max(0.0, float(os.getenv("GPT_RETRY_JITTER_SEC", "0.25")))
_MIN_CALL_INTERVAL_SEC = max(0.1, float(os.getenv("GPT_MIN_CALL_INTERVAL_SEC", "0.6")))
_MODEL_TEMPERATURE = float(os.getenv("GPT_DECIDER_TEMPERATURE", "0.0") or 0.0)

# 単純なレートリミット（call間隔を確保して 429 を防ぐ）
_LAST_CALL_TS: float | None = None
_CALL_LOCK = asyncio.Lock()

def _normalize_json_content(raw: str) -> str:
    """Remove Markdown fences / whitespace around JSON payloads."""
    text = raw.strip()
    if text.startswith("```"):
        text = _FENCE_PREFIX_RE.sub("", text, count=1)
        if text.endswith("```"):
            text = text[: -3]
    return text.strip()

_LAST_DECISION_TS: dt.datetime | None = None
_LAST_DECISION_DATA: Dict[str, object] | None = None
_LAST_FAILURE_TS: dt.datetime | None = None


class GPTTimeout(Exception): ...

def _extract_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _load_json_payload(content: str) -> Dict:
    if not content:
        raise ValueError("Invalid JSON: (empty string) – check model/temperature/response_format")
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        candidate = _extract_json_object(content)
        if candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Invalid JSON: {content}") from exc


async def _call_model(payload: Dict, messages: List[Dict], model: str) -> Dict:
    tier = "primary" if model == MODEL else "fallback"
    with track_gpt_call(
        "gpt_decider",
        extra_tags={"tier": tier, "model": model},
    ) as tracker:
        content: str = ""
        # simple rate limit to avoid 429
        async with _CALL_LOCK:
            global _LAST_CALL_TS
            now = time.monotonic()
            if _LAST_CALL_TS is not None:
                delta = now - _LAST_CALL_TS
                if delta < _MIN_CALL_INTERVAL_SEC:
                    await asyncio.sleep(_MIN_CALL_INTERVAL_SEC - delta)
            _LAST_CALL_TS = time.monotonic()
        timeout = _MODEL_TIMEOUT_SECONDS.get(model, 15)
        base_kwargs = {
            "model": model,
            "messages": messages,
            "timeout": timeout,
            "response_format": {"type": "json_object"},
            "temperature": _MODEL_TEMPERATURE,
        }
        token_kwargs_order = [
            {"max_completion_tokens": _MAX_COMPLETION_TOKENS},
            {"max_tokens": _MAX_COMPLETION_TOKENS},
        ]

        async def _sleep_with_jitter(attempt: int) -> None:
            delay = _RETRY_BASE_SEC * (2 ** attempt)
            if _RETRY_JITTER_SEC > 0:
                delay += random.uniform(0.0, _RETRY_JITTER_SEC)
            await asyncio.sleep(delay)

        def _is_retryable(exc: Exception) -> bool:
            msg = str(exc).lower()
            fatal_tokens = ("authentication", "invalid api key", "invalid_request_error", "invalid request")
            if any(tok in msg for tok in fatal_tokens):
                return False
            if "unsupported parameter" in msg:
                return False
            if "insufficient_quota" in msg or "rate limit" in msg or "429" in msg:
                return True
            return True

        usage_in = usage_out = 0
        last_error: Exception | None = None
        for attempt in range(_MAX_MODEL_ATTEMPTS):
            for token_kwargs in token_kwargs_order:
                try:
                    client = _get_openai_client()
                    resp = await client.chat.completions.create(
                        **base_kwargs, **token_kwargs
                    )
                    usage_in = resp.usage.prompt_tokens
                    usage_out = resp.usage.completion_tokens
                    content = _normalize_json_content(resp.choices[0].message.content or "")
                    tracker.add_tag("tokens", usage_in + usage_out)
                    add_tokens(usage_in + usage_out, MAX_TOKENS_MONTH)
                    data = _load_json_payload(content)

                    for key in _REQUIRED_KEYS:
                        if key not in data:
                            raise ValueError(f"key {key} missing")

                    mode = str(data.get("mode") or "").strip().upper()
                    if mode not in _ALLOWED_MODES:
                        mode = "DEFENSIVE"
                    data["mode"] = mode

                    risk_bias = str(data.get("risk_bias") or "").strip().lower()
                    if risk_bias not in _ALLOWED_RISK:
                        risk_bias = "neutral"
                    data["risk_bias"] = risk_bias

                    liquidity_bias = str(data.get("liquidity_bias") or "").strip().lower()
                    if liquidity_bias not in _ALLOWED_LIQ:
                        liquidity_bias = "normal"
                    data["liquidity_bias"] = liquidity_bias

                    try:
                        rc = float(data.get("range_confidence") or 0.0)
                    except (TypeError, ValueError):
                        rc = 0.0
                    data["range_confidence"] = max(0.0, min(1.0, rc))

                    hints = data.get("pattern_hint") or []
                    if isinstance(hints, str):
                        hints = [hints]
                    clean_hints: list[str] = []
                    for h in hints:
                        if isinstance(h, str) and h.strip():
                            clean_hints.append(h.strip()[:24])
                    data["pattern_hint"] = clean_hints[:5]

                    try:
                        weight_macro = float(data["weight_macro"])
                    except (TypeError, ValueError) as exc:
                        raise ValueError("weight_macro type error") from exc
                    weight_macro = max(0.0, min(1.0, weight_macro))
                    data["weight_macro"] = round(weight_macro, 2)

                    focus_tag = data.get("focus_tag")
                    if not isinstance(focus_tag, str) or focus_tag not in _FOCUS_TAGS:
                        focus_tag = "hybrid"
                    data["focus_tag"] = focus_tag

                    weight_scalp_raw = data.get("weight_scalp")
                    weight_scalp: float | None = None
                    default_scalp = min(0.25, max(0.1, 1.0 - weight_macro))
                    if weight_scalp_raw is not None:
                        try:
                            weight_scalp = float(weight_scalp_raw)
                        except (TypeError, ValueError):
                            weight_scalp = None
                    if weight_scalp is None:
                        weight_scalp = default_scalp
                    weight_scalp = max(0.0, min(1.0, weight_scalp))
                    if weight_scalp + weight_macro > 0.95:
                        weight_scalp = max(0.0, 0.95 - weight_macro)
                    data["weight_scalp"] = round(weight_scalp, 2)

                    data["ranked_strategies"] = data.get("ranked_strategies") or []
                    data["model_used"] = model
                    return data
                except Exception as exc:
                    last_error = exc
                    if isinstance(exc, ValueError):
                        logger.warning(
                            "GPT validation failed (model=%s): %s content=%s",
                            model,
                            exc,
                            (content or "")[:500],
                        )
                    if not _is_retryable(exc):
                        raise GPTTimeout(str(exc)) from exc
                    # unsupported token kwargs: try next token kwarg without counting attempt
                    msg = str(exc)
                    param_name = next(iter(token_kwargs))
                    if "Unsupported parameter" in msg and param_name in msg:
                        continue
                    break  # will go to retry loop
            if attempt < _MAX_MODEL_ATTEMPTS - 1:
                await _sleep_with_jitter(attempt)

        raise GPTTimeout(str(last_error) if last_error else "unable to call OpenAI")


async def call_openai(payload: Dict) -> Dict:
    """非同期で GPT を呼ぶ → dict を返す（フォールバック不要値は None）"""
    if _LLM_MODE in _DUMMY_MODES:
        raise RuntimeError("LLM_MODE dummy is not allowed when fallback is disabled")

    if not add_tokens(0, MAX_TOKENS_MONTH):
        raise RuntimeError("GPT token limit exceeded")

    messages = build_messages(payload)
    models_to_try = [MODEL] + [m for m in _FALLBACK_MODELS if m != MODEL]
    last_exc: Exception | None = None
    for idx, model in enumerate(models_to_try, start=1):
        try:
            result = await _call_model(payload, messages, model)
            if model != MODEL:
                logger.warning("GPT decision succeeded via fallback model %s", model)
            # ranked_strategies はローカルで決定する方針のため空配列を保証
            if "ranked_strategies" not in result or result.get("ranked_strategies") is None:
                result["ranked_strategies"] = []
            return result
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "GPT model %s failed attempt %d (%s: %s)",
                model,
                idx,
                type(exc).__name__,
                str(exc) or "no message",
            )
            continue

    raise GPTTimeout(str(last_exc) if last_exc else "all GPT models failed")


async def get_decision(payload: Dict) -> Dict:
    """
    上位ラッパ：GPT 呼び出し（リトライあり、失敗時は例外とする）
    """
    global _LAST_DECISION_TS, _LAST_DECISION_DATA, _LAST_FAILURE_TS
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            fresh = await call_openai(payload)
            if not isinstance(fresh, dict):
                raise ValueError("GPT response must be dict")
            _LAST_DECISION_TS = dt.datetime.utcnow()
            _LAST_DECISION_DATA = {k: v for k, v in fresh.items() if k != "reason"}
            _LAST_FAILURE_TS = None
            return fresh
        except Exception as e:
            last_exc = e
            _LAST_FAILURE_TS = dt.datetime.utcnow()
            logger.warning(
                "GPT decision attempt %d failed (%s: %s)",
                attempt + 1,
                type(e).__name__,
                str(e) or "no message",
            )
            await asyncio.sleep(0.6 * (attempt + 1))
            continue

    if _LAST_DECISION_DATA:
        logger.warning(
            "GPT decision failed after retries (%s); reusing previous decision", last_exc
        )
        return dict(_LAST_DECISION_DATA, reason="reuse_previous")

    raise GPTTimeout(str(last_exc) if last_exc else "all GPT models failed")


def fallback_decision(payload: Dict, last_decision: Dict | None = None):
    """
    GPT が利用できない場合の明示的フォールバック。
    ここでは最後に成功した GPT 決定を再利用し、無ければ例外を投げる。
    """
    if last_decision:
        logger.warning("fallback_decision using last_decision")
        return dict(last_decision, reason="reuse_previous")
    raise RuntimeError("fallback_decision unavailable: no previous decision")


# ---------- CLI self‑test ----------
if __name__ == "__main__":
    import datetime
    import asyncio
    import pprint

    dummy = {
        "ts": datetime.datetime.utcnow().isoformat(timespec="seconds"),
        "reg_macro": "Trend",
        "reg_micro": "Range",
        "factors_m1": {"ma10": 157.2, "ma20": 157.1, "adx": 30},
        "factors_h4": {"ma10": 157.0, "ma20": 156.8, "adx": 25},
        "perf": {"macro_pf": 1.3, "micro_pf": 1.1},
    }

    res = asyncio.run(get_decision(dummy))
    pprint.pp(res)
