"""OpenAI client wrapper for the manual trading assistant."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from openai import AsyncOpenAI

from utils.cost_guard import add_cost, add_tokens, within_budget_usd
from utils.secrets import get_secret

from .context import ManualContext
from .prompt_builder import build_manual_messages

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None
_MODEL_CANDIDATES: List[str] | None = None
_PRICE_IN_PER_M = None
_PRICE_OUT_PER_M = None
_MAX_MONTH_TOKENS = None
_MAX_MONTH_USD = None


def _get_str_secret(keys: tuple[str, ...]) -> str | None:
    for key in keys:
        try:
            value = get_secret(key)
            if value:
                return str(value)
        except Exception:
            continue
    return None


def _get_float_secret(keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        try:
            return float(get_secret(key))
        except Exception:
            continue
    return default


def _get_int_secret(keys: tuple[str, ...], default: int) -> int:
    for key in keys:
        try:
            return int(get_secret(key))
        except Exception:
            continue
    return default


def _model_candidates() -> List[str]:
    global _MODEL_CANDIDATES
    if _MODEL_CANDIDATES is not None:
        return _MODEL_CANDIDATES

    seen: set[str] = set()
    models: List[str] = []

    def _push(name: str | None) -> None:
        if not name:
            return
        normalized = name.strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        models.append(normalized)

    for key in (
        "openai_manual_model",
        "openai_planner_model",
        "openai_decider_model",
        "openai_model",
    ):
        try:
            _push(get_secret(key))
        except Exception:
            continue

    env_chain = _get_str_secret(("OPENAI_MANUAL_FALLBACK_MODELS", "GPT_DECIDER_FALLBACK_MODELS"))
    if env_chain:
        for item in env_chain.split(","):
            _push(item)

    for default in ("gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"):
        _push(default)

    if not models:
        models.append("gpt-4o-mini")

    _MODEL_CANDIDATES = models
    return models


async def _ensure_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = _get_str_secret(("openai_manual_api_key", "openai_api_key"))
        _client = AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()
    return _client


def _ensure_budget_settings() -> None:
    global _PRICE_IN_PER_M, _PRICE_OUT_PER_M, _MAX_MONTH_TOKENS, _MAX_MONTH_USD
    if _PRICE_IN_PER_M is None:
        _PRICE_IN_PER_M = _get_float_secret(
            (
                "openai_manual_cost_per_million_input",
                "openai_cost_per_million_input",
            ),
            0.25,
        )
    if _PRICE_OUT_PER_M is None:
        _PRICE_OUT_PER_M = _get_float_secret(
            (
                "openai_manual_cost_per_million_output",
                "openai_cost_per_million_output",
            ),
            2.0,
        )
    if _MAX_MONTH_TOKENS is None:
        _MAX_MONTH_TOKENS = _get_int_secret(
            (
                "openai_manual_max_month_tokens",
                "openai_max_month_tokens",
            ),
            500_000,
        )
    if _MAX_MONTH_USD is None:
        _MAX_MONTH_USD = _get_float_secret(
            (
                "openai_manual_max_month_usd",
                "openai_max_month_usd",
            ),
            60.0,
        )


def _fallback_guidance(ctx: ManualContext, *, error: str) -> Dict[str, Any]:
    macro = ctx.macro
    micro = ctx.micro
    close = float(micro.factors.get("close", 0.0) or 0.0)
    ma20 = float(micro.factors.get("ma20", close) or close)
    bias = "neutral"
    if close and ma20:
        if close - ma20 > 0.05:
            bias = "long"
        elif ma20 - close > 0.05:
            bias = "short"
    trade_ideas: List[Dict[str, Any]] = []
    rationale = (
        f"Fallback due to GPT error: {error}. Using moving-average bias for guidance."
    )
    if bias != "neutral":
        trade_ideas.append(
            {
                "label": "MA20 bias placeholder",
                "direction": bias,
                "style": "market",
                "entry_note": "Observe price action around M1 MA20 before executing.",
                "stop_loss": "Place beyond recent swing (M1 last 10 candles).",
                "take_profit": "Aim for 1.5x risk or H4 range edge.",
                "risk_reward": "~1:1.5",
                "rationale": rationale,
                "conditions": [
                    "Confirm no high-impact news within the next hour.",
                    "Spread below 1.5 pips.",
                ],
            }
        )
    return {
        "bias": bias,
        "confidence": 30,
        "market_view": {
            "macro": f"Macro regime: {macro.regime} (close {float(macro.factors.get('close', 0.0)):.3f}).",
            "micro": f"Micro regime: {micro.regime} with close vs MA20 delta {float(close - ma20):.3f}.",
            "news": "Limited context available; review news manually.",
        },
        "trade_ideas": trade_ideas,
        "risk_notes": [
            "Fallback guidance active; rely on personal discretion.",
            "Maintain conservative sizing until GPT service is restored.",
        ],
        "next_steps": [
            "Refresh context once GPT connectivity stabilises.",
        ],
    }


async def get_manual_guidance(ctx: ManualContext) -> Dict[str, Any]:
    """Call OpenAI to obtain a manual trading plan; fallback on error."""

    _ensure_budget_settings()
    if not within_budget_usd(float(_MAX_MONTH_USD)):
        return _fallback_guidance(ctx, error="monthly USD budget exhausted")

    messages = build_manual_messages(ctx)
    client = await _ensure_client()

    errors: List[str] = []
    for model in _model_candidates():
        try:
            response = await client.chat.completions.create(  # type: ignore[call-arg]
                model=model,
                messages=messages,
                max_completion_tokens=400,
                timeout=10,
            )
        except Exception as exc:  # pragma: no cover - network call
            logger.warning("Manual GPT call failed for %s: %s", model, exc)
            errors.append(f"{model}: {exc}")
            continue

        usage_in = int(response.usage.prompt_tokens or 0)
        usage_out = int(response.usage.completion_tokens or 0)
        add_tokens(usage_in + usage_out, int(_MAX_MONTH_TOKENS))
        add_cost(
            model=model,
            prompt_tokens=usage_in,
            completion_tokens=usage_out,
            price_in_per_m=float(_PRICE_IN_PER_M),
            price_out_per_m=float(_PRICE_OUT_PER_M),
            max_month_usd=float(_MAX_MONTH_USD),
        )

        content = (response.choices[0].message.content or "").strip()
        content = content.lstrip("```json").rstrip("```").strip()
        if not content:
            errors.append(f"{model}: empty response")
            continue

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            errors.append(f"{model}: invalid JSON {exc}")
            continue

        if not isinstance(data, dict):
            errors.append(f"{model}: non-object response")
            continue

        return data

    fallback_error = "; ".join(errors) if errors else "unknown error"
    return _fallback_guidance(ctx, error=fallback_error)
