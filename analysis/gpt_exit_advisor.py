"""
analysis.gpt_exit_advisor
~~~~~~~~~~~~~~~~~~~~~~~~~
Exit判断を補助する GPT 連携モジュール。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Any, Iterable, List

from openai import AsyncOpenAI, BadRequestError

from utils.cost_guard import add_cost, add_tokens, within_budget_usd
from utils.secrets import get_secret

_MODEL = None
_MAX_MONTH_USD = None
_PRICE_IN_PER_M = None
_PRICE_OUT_PER_M = None
_MAX_TOKENS_MONTH = None
_CLIENT: AsyncOpenAI | None = None
ALLOW_EXIT_FALLBACK = os.getenv("ALLOW_EXIT_FALLBACK", "false").lower() == "true"

logger = logging.getLogger(__name__)

_STR_TEMPLATE = (
    "You are an FX exit advisor for USD/JPY trades.\n"
    "Decide whether to lock gains, adjust targets, or hold, balancing drawdown control and trend continuation.\n"
    "Return strict JSON with keys: close_now(bool), target_tp_pips(number), target_sl_pips(number), confidence(number 0-1), note(str)."
)


class GPTExitAdvisorError(RuntimeError):
    """Raised when GPT exit advice cannot be retrieved."""


def _get_float_secret(keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        try:
            return float(get_secret(key))
        except Exception:
            continue
    return default


async def _ensure_client() -> AsyncOpenAI:
    global _CLIENT, _MODEL, _MAX_MONTH_USD, _PRICE_IN_PER_M, _PRICE_OUT_PER_M, _MAX_TOKENS_MONTH
    if _MODEL is None:
        try:
            _MODEL = get_secret("openai_exit_model")
        except Exception:
            try:
                _MODEL = get_secret("openai_model")
            except Exception:
                _MODEL = "gpt-5"
    if _MAX_TOKENS_MONTH is None:
        try:
            _MAX_TOKENS_MONTH = int(get_secret("openai_exit_max_month_tokens"))
        except Exception:
            _MAX_TOKENS_MONTH = int(
                _get_float_secret(("openai_max_month_tokens",), 500_000.0)
            )
    if _MAX_MONTH_USD is None:
        _MAX_MONTH_USD = _get_float_secret(
            ("openai_exit_max_month_usd", "openai_max_month_usd"),
            120.0,
        )
    if _PRICE_IN_PER_M is None:
        _PRICE_IN_PER_M = _get_float_secret(
            ("openai_exit_cost_per_million_input", "openai_cost_per_million_input"),
            1.25,
        )
    if _PRICE_OUT_PER_M is None:
        _PRICE_OUT_PER_M = _get_float_secret(
            ("openai_exit_cost_per_million_output", "openai_cost_per_million_output"),
            10.0,
        )
    if _CLIENT is None:
        try:
            api_key = get_secret("openai_api_key")
        except Exception:
            api_key = None
        _CLIENT = AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()
    return _CLIENT


def _fallback(payload: Dict[str, Any]) -> Dict[str, Any]:
    move = float(payload.get("unrealized_pips", 0.0) or 0.0)
    direction = 1 if payload.get("direction") == "long" else -1
    atr_pips = max(
        float(payload.get("atr_h4_pips", 0.0) or 0.0),
        float(payload.get("atr_m1_pips", 0.0) or 0.0),
    )
    atr_pips = max(atr_pips, 5.0)
    target_tp = move + 0.5 * atr_pips
    target_sl = max(3.0, min(move * 0.6, atr_pips))
    close_now = move <= 0.5
    return {
        "close_now": bool(close_now and move < 1.0),
        "target_tp_pips": round(max(target_tp, 6.0), 2),
        "target_sl_pips": round(max(target_sl, 4.0), 2),
        "confidence": 0.35,
        "note": "fallback",
    }


def _normalize(resp: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(resp, dict):
        return _fallback({})
    close_now = bool(resp.get("close_now", False))
    try:
        tp = float(resp.get("target_tp_pips", 0.0))
    except (TypeError, ValueError):
        tp = 0.0
    try:
        sl = float(resp.get("target_sl_pips", 0.0))
    except (TypeError, ValueError):
        sl = 0.0
    try:
        conf = float(resp.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    note = resp.get("note") or ""
    tp = round(max(0.0, min(tp, 200.0)), 2)
    sl = round(max(0.0, min(sl, 150.0)), 2)
    conf = max(0.0, min(conf, 1.0))
    if tp == 0.0:
        tp = 12.0
    if sl == 0.0:
        sl = min(tp * 0.6, 20.0)
    return {
        "close_now": close_now,
        "target_tp_pips": tp,
        "target_sl_pips": sl,
        "confidence": conf,
        "note": note,
    }


def _model_candidates() -> List[str]:
    """Return primary model followed by safe fallbacks (deduplicated)."""

    seen: set[str] = set()
    models: List[str] = []

    def _push(candidate: str | None) -> None:
        if not candidate:
            return
        cand = candidate.strip()
        if not cand or cand in seen:
            return
        seen.add(cand)
        models.append(cand)

    _push(_MODEL)

    for key in ("openai_exit_fallback_model", "openai_model"):
        try:
            _push(get_secret(key))
        except Exception:
            continue

    env_chain = os.environ.get("EXIT_GPT_FALLBACK_MODELS")
    if env_chain:
        for item in env_chain.split(","):
            _push(item)

    for default in ("gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"):
        _push(default)

    return models


async def _call_model(model: str, messages: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Invoke OpenAI chat.completions and return decoded JSON."""

    client = await _ensure_client()
    messages_payload = list(messages)
    base_kwargs = {
        "model": model,
        "messages": messages_payload,
        "timeout": 7,
    }
    try:
        resp = await client.chat.completions.create(
            **base_kwargs,
            max_tokens=180,
        )
    except BadRequestError as exc:
        detail = str(exc).lower()
        if "max_tokens" not in detail:
            raise GPTExitAdvisorError(f"{model} call failed: {exc}") from exc
        try:
            resp = await client.chat.completions.create(
                **base_kwargs,
                max_completion_tokens=180,
            )
        except Exception as inner:
            raise GPTExitAdvisorError(f"{model} call failed after retry: {inner}") from inner
    except Exception as exc:
        raise GPTExitAdvisorError(f"{model} call failed: {exc}") from exc

    usage_in = int(resp.usage.prompt_tokens or 0)
    usage_out = int(resp.usage.completion_tokens or 0)
    add_tokens(usage_in + usage_out, int(_MAX_TOKENS_MONTH or 500_000))
    add_cost(
        model=model,
        prompt_tokens=usage_in,
        completion_tokens=usage_out,
        price_in_per_m=float(_PRICE_IN_PER_M or 0.15),
        price_out_per_m=float(_PRICE_OUT_PER_M or 0.60),
        max_month_usd=float(_MAX_MONTH_USD or 30.0),
    )

    content = resp.choices[0].message.content or ""
    content = content.strip()
    if not content:
        raise GPTExitAdvisorError(f"{model} returned empty content")

    content = content.lstrip("```json").rstrip("```").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise GPTExitAdvisorError(f"Invalid JSON from {model}: {content}") from exc


async def get_exit_advice(payload: Dict[str, Any]) -> Dict[str, Any]:
    await _ensure_client()
    if not within_budget_usd(_MAX_MONTH_USD):
        raise GPTExitAdvisorError("USD budget exceeded")

    user_blob = json.dumps(payload, ensure_ascii=False)
    messages = [
        {"role": "system", "content": _STR_TEMPLATE},
        {"role": "user", "content": f"Context JSON: {user_blob}"},
    ]
    errors: list[str] = []
    for idx, model in enumerate(_model_candidates()):
        try:
            data = await _call_model(model, messages)
            if idx > 0:
                logger.warning(
                    "[GPT_EXIT] primary model %s failed; used fallback %s",
                    _MODEL,
                    model,
                )
            return _normalize(data)
        except GPTExitAdvisorError as exc:
            errors.append(f"{model}: {exc}")
            logger.warning("[GPT_EXIT] model attempt failed: %s", errors[-1])
            continue
    raise GPTExitAdvisorError("; ".join(errors))


async def advise_or_fallback(
    payload: Dict[str, Any], *, force_fallback: bool = False
) -> Dict[str, Any]:
    try:
        if force_fallback:
            if not ALLOW_EXIT_FALLBACK:
                raise GPTExitAdvisorError("Fallback disabled for exit advisor")
            logger.info("[GPT_EXIT] forced fallback requested; skipping model call")
            return _fallback(payload)
        if os.environ.get("GPT_DISABLE_ALL") or os.environ.get("DISABLE_GPT_EXIT"):
            if not ALLOW_EXIT_FALLBACK:
                raise GPTExitAdvisorError("Exit advisor disabled and fallback prohibited")
            logging.info("[GPT_EXIT] disabled via env; returning fallback")
            return _fallback(payload)
        return await get_exit_advice(payload)
    except GPTExitAdvisorError as exc:
        if not ALLOW_EXIT_FALLBACK:
            logger.error("[GPT_EXIT] error with fallback disabled: %s", exc)
            raise
        logger.warning("[GPT_EXIT] fallback due to GPT error: %s", exc)
        return _fallback(payload)
    except Exception as exc:
        if not ALLOW_EXIT_FALLBACK:
            logger.exception("[GPT_EXIT] unexpected error with fallback disabled")
            raise
        logger.exception("[GPT_EXIT] unexpected error -> fallback: %s", exc)
        return _fallback(payload)
