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
import re
from typing import Dict, List

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


def _get_openai_client() -> AsyncOpenAI:
    global _CLIENT
    if _CLIENT is None:
        api_key = get_secret("openai_api_key")
        _CLIENT = AsyncOpenAI(api_key=api_key)
    return _CLIENT


_REQUIRED_KEYS = ("focus_tag", "weight_macro", "ranked_strategies")

_FOCUS_TAGS = {"micro", "macro", "hybrid", "event"}
_ALLOWED_STRATEGIES = [
    "TrendMA",
    "Donchian55",
    "H1Momentum",
    "BB_RSI",
    "BB_RSI_Fast",
    "M1Scalper",
    "RangeFader",
    "PulseBreak",
    "ImpulseRetrace",
    "MomentumBurst",
    "TrendMomentumMicro",
    "MicroMomentumStack",
    "MicroPullbackEMA",
    "MicroRangeBreak",
    "MicroLevelReactor",
]

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

_FALLBACK_MODELS = [
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
]

_REUSE_WINDOW_SECONDS = 300
_FAIL_OPEN_SECONDS = int(os.getenv("GPT_FAIL_OPEN_SECONDS", "120") or 120)
_FALLBACK_DECISION = {
    "focus_tag": "hybrid",
    "weight_macro": 0.7,
    "weight_scalp": 0.1,
    "ranked_strategies": [
        "TrendMA",
        "H1Momentum",
        "Donchian55",
        "MomentumBurst",
        "BB_RSI",
        "BB_RSI_Fast",
        "MomentumBurst",
        "TrendMomentumMicro",
        "MicroMomentumStack",
        "MicroPullbackEMA",
        "MicroRangeBreak",
        "MicroLevelReactor",
        "MomentumPulse",
        "VolCompressionBreak",
        "MicroVWAPRevert",
        "RangeFader",
        "PulseBreak",
        "M1Scalper",
        "M1Scalper",
        "RangeFader",
    ],
    "reason": "fallback",
}

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
        raise ValueError("Invalid JSON: (empty string)")
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
        timeout = _MODEL_TIMEOUT_SECONDS.get(model, 15)
        base_kwargs = {
            "model": model,
            "messages": messages,
            "timeout": timeout,
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }
        token_kwargs_order = [
            {"max_completion_tokens": _MAX_COMPLETION_TOKENS},
            {"max_tokens": _MAX_COMPLETION_TOKENS},
        ]
        usage_in = usage_out = 0
        resp = None
        last_error: Exception | None = None
        for token_kwargs in token_kwargs_order:
            try:
                client = _get_openai_client()
                resp = await client.chat.completions.create(
                    **base_kwargs, **token_kwargs
                )
                break
            except Exception as exc:
                msg = str(exc)
                param_name = next(iter(token_kwargs))
                if "Unsupported parameter" in msg and param_name in msg:
                    last_error = exc
                    continue
                raise GPTTimeout(msg) from exc
        else:
            raise GPTTimeout(str(last_error) if last_error else "unable to call OpenAI")

        usage_in = resp.usage.prompt_tokens
        usage_out = resp.usage.completion_tokens
        content = _normalize_json_content(resp.choices[0].message.content or "")

        tracker.add_tag("tokens", usage_in + usage_out)
        add_tokens(usage_in + usage_out, MAX_TOKENS_MONTH)
        data = _load_json_payload(content)

        for key in _REQUIRED_KEYS:
            if key not in data:
                raise ValueError(f"key {key} missing")
        if not isinstance(data["ranked_strategies"], list):
            raise ValueError("ranked_strategies type error")

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
        if weight_scalp_raw is not None:
            try:
                weight_scalp = float(weight_scalp_raw)
            except (TypeError, ValueError):
                weight_scalp = None
        if weight_scalp is not None:
            weight_scalp = max(0.0, min(1.0, weight_scalp))
            if weight_scalp + weight_macro > 1.0:
                weight_scalp = max(0.0, 1.0 - weight_macro)
            data["weight_scalp"] = round(weight_scalp, 2)
        else:
            data["weight_scalp"] = None

        ranked = [
            s for s in data.get("ranked_strategies", []) if s in _ALLOWED_STRATEGIES
        ]
        data["ranked_strategies"] = ranked
        data["model_used"] = model
        return data


async def call_openai(payload: Dict) -> Dict:
    """非同期で GPT を呼ぶ → dict を返す（フォールバック不要値は None）"""
    if _LLM_MODE in _DUMMY_MODES:
        logger.info("[GPT] dummy mode active (LLM_MODE=%s)", _LLM_MODE or "dummy")
        decision = heuristic_decision(
            payload, _LAST_DECISION_DATA or _FALLBACK_DECISION
        )
        decision["reason"] = "dummy_mode"
        return decision

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
    上位ラッパ：GPT 呼び出し（リトライあり、失敗時は再利用/フォールバック決定）
    """
    global _LAST_DECISION_TS, _LAST_DECISION_DATA, _LAST_FAILURE_TS
    now = dt.datetime.utcnow()
    if _LAST_FAILURE_TS:
        elapsed = (now - _LAST_FAILURE_TS).total_seconds()
        if elapsed <= _FAIL_OPEN_SECONDS:
            remaining = max(0, _FAIL_OPEN_SECONDS - int(elapsed))
            logger.warning(
                "GPT fail-open active (remaining %ss); using fallback decision.",
                remaining,
            )
            return fallback_decision(
                payload,
                last_decision=_LAST_DECISION_DATA,
                reason="fail_open",
                now=now,
                log_reason=True,
                last_exception=None,
            )
    # 最大2回リトライ（Responses→Chat のフォールバック込み）
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            fresh = await call_openai(payload)
            if not isinstance(fresh, dict):
                raise ValueError("GPT response must be dict")
            # 決定情報を保持（reasonは揮発値なので除外）
            _LAST_DECISION_TS = dt.datetime.utcnow()
            _LAST_DECISION_DATA = {k: v for k, v in fresh.items() if k != "reason"}
            _LAST_FAILURE_TS = None
            return fresh
        except Exception as e:
            last_exc = e
            logger.warning(
                "GPT decision attempt %d failed (%s: %s)",
                attempt + 1,
                type(e).__name__,
                str(e) or "no message",
            )
            await asyncio.sleep(1.5)
    # フォールバック：直近 5 分以内の決定を再利用、なければデフォルト構成を返す
    now = dt.datetime.utcnow()
    _LAST_FAILURE_TS = now
    if _LAST_DECISION_TS and _LAST_DECISION_DATA:
        age = (now - _LAST_DECISION_TS).total_seconds()
        if age <= _REUSE_WINDOW_SECONDS:
            reused = fallback_decision(
                payload,
                last_decision=_LAST_DECISION_DATA,
                reason="reuse_previous",
                now=now,
                log_reason=True,
                last_exception=last_exc,
            )
            return reused

    return fallback_decision(
        payload,
        last_decision=_LAST_DECISION_DATA,
        reason="heuristic",
        now=now,
        log_reason=True,
        last_exception=last_exc,
    )


def fallback_decision(
    payload: Dict,
    *,
    last_decision: Dict[str, object] | None = None,
    reason: str = "fallback",
    now: dt.datetime | None = None,
    log_reason: bool = False,
    last_exception: Exception | None = None,
) -> Dict:
    global _LAST_DECISION_TS, _LAST_DECISION_DATA
    use_now = now or dt.datetime.utcnow()
    decision: Dict[str, object]
    try:
        decision = heuristic_decision(payload, last_decision)
        decision["reason"] = reason
        if log_reason:
            log_gpt_fallback("gpt_decider", "heuristic")
            if last_exception is not None:
                logger.warning(
                    "GPT decision failed (%s); using heuristic fallback decision.",
                    str(last_exception),
                )
    except Exception as heur_exc:  # pragma: no cover - defensive path
        decision = dict(_FALLBACK_DECISION)
        decision["reason"] = reason
        if log_reason:
            log_gpt_fallback("gpt_decider", "static_config")
            logger.error(
                "GPT decision fallback to static config (%s -> %s)",
                str(last_exception),
                heur_exc,
            )
    _LAST_DECISION_TS = use_now
    _LAST_DECISION_DATA = {k: v for k, v in decision.items() if k != "reason"}
    return decision


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
        "news_features": {"news_count_total": 1.0, "news_latest_age_minutes": 5.0},
        "perf": {"macro_pf": 1.3, "micro_pf": 1.1},
    }

    res = asyncio.run(get_decision(dummy))
    pprint.pp(res)
