"""
analysis.gpt_decider
~~~~~~~~~~~~~~~~~~~~
OpenAI API を呼び出し、JSON スキーマ検証・
フォールバック・トークン課金記録までを一括で行うハイレベル関数。
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
from typing import Dict

from openai import AsyncOpenAI

from analysis.gpt_prompter import (
    build_messages,
    OPENAI_MODEL as MODEL,
    MAX_TOKENS_MONTH,
)
from analysis.local_decider import heuristic_decision
from utils.cost_guard import add_tokens
from utils.secrets import get_secret

client = AsyncOpenAI(api_key=get_secret("openai_api_key"))


_SCHEMA = {
    "focus_tag": str,
    "weight_macro": float,
    "ranked_strategies": list,
}

_FOCUS_TAGS = {"micro", "macro", "hybrid", "event"}
_ALLOWED_STRATEGIES = [
    "TrendMA",
    "Donchian55",
    "BB_RSI",
    "NewsSpikeReversal",
    "M1Scalper",
]

_MAX_COMPLETION_TOKENS = 320
_GPT5_MAX_OUTPUT_TOKENS = 800

_REUSE_WINDOW_SECONDS = 300
_FALLBACK_DECISION = {
    "focus_tag": "hybrid",
    "weight_macro": 0.5,
    "ranked_strategies": [
        "TrendMA",
        "Donchian55",
        "BB_RSI",
        "NewsSpikeReversal",
    ],
    "reason": "fallback",
}

_LAST_DECISION_TS: dt.datetime | None = None
_LAST_DECISION_DATA: Dict[str, object] | None = None

logger = logging.getLogger(__name__)


class GPTTimeout(Exception): ...


async def call_openai(payload: Dict) -> Dict:
    """非同期で GPT を呼ぶ → dict を返す（フォールバック不要値は None）"""
    # コストガード
    if not add_tokens(0, MAX_TOKENS_MONTH):
        raise RuntimeError("GPT token limit exceeded")

    msgs = build_messages(payload)

    is_gpt5 = "gpt-5" in MODEL

    if is_gpt5:
        inputs = [{"role": msg["role"], "content": msg["content"]} for msg in msgs]
        try:
            resp = await client.responses.create(
                model=MODEL,
                input=inputs,
                reasoning={"effort": "low"},
                max_output_tokens=_GPT5_MAX_OUTPUT_TOKENS,
                timeout=15,
            )
        except Exception as exc:
            raise GPTTimeout(str(exc)) from exc

        usage = getattr(resp, "usage", None)
        usage_in = getattr(usage, "input_tokens", 0) if usage else 0
        usage_out = getattr(usage, "output_tokens", 0) if usage else 0
        add_tokens(usage_in + usage_out, MAX_TOKENS_MONTH)

        content_parts: list[str] = []
        for item in resp.output or []:
            if getattr(item, "type", "") != "message":
                continue
            for block in getattr(item, "content", []) or []:
                text = getattr(block, "text", None)
                if text:
                    content_parts.append(text)
        content = "".join(content_parts).strip()
    else:
        base_kwargs = {
            "model": MODEL,
            "messages": msgs,
            "timeout": 7,
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }
        token_kwargs_order = [
            {"max_completion_tokens": _MAX_COMPLETION_TOKENS},
            {"max_tokens": _MAX_COMPLETION_TOKENS},
        ]

        resp = None
        last_error: Exception | None = None
        for token_kwargs in token_kwargs_order:
            try:
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
            raise GPTTimeout(
                str(last_error) if last_error else "unable to call OpenAI"
            )

        usage_in = resp.usage.prompt_tokens
        usage_out = resp.usage.completion_tokens
        add_tokens(usage_in + usage_out, MAX_TOKENS_MONTH)

        content = resp.choices[0].message.content.strip()
        content = content.lstrip("```json").rstrip("```").strip()

    if not content:
        raise ValueError("Invalid JSON: (empty string)")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {content}") from e

    # スキーマ簡易検証
    for k, typ in _SCHEMA.items():
        if k not in data:
            raise ValueError(f"key {k} missing")
        if not isinstance(data[k], typ):
            raise ValueError(f"{k} type error")
    data["weight_macro"] = round(float(data["weight_macro"]), 2)
    focus_tag = data.get("focus_tag")
    if focus_tag not in _FOCUS_TAGS:
        data["focus_tag"] = "hybrid"

    weight = data.get("weight_macro", 0.5)
    data["weight_macro"] = max(0.0, min(1.0, weight))

    ranked = [
        s for s in data.get("ranked_strategies", []) if s in _ALLOWED_STRATEGIES
    ]
    data["ranked_strategies"] = ranked
    return data


async def get_decision(payload: Dict) -> Dict:
    """
    上位ラッパ：GPT 呼び出し（リトライあり、失敗時は再利用/フォールバック決定）
    """
    # 最大2回リトライ（合計最大 ~9秒）
    global _LAST_DECISION_TS, _LAST_DECISION_DATA
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            fresh = await asyncio.wait_for(call_openai(payload), timeout=9)
            if not isinstance(fresh, dict):
                raise ValueError("GPT response must be dict")
            # 決定情報を保持（reasonは揮発値なので除外）
            _LAST_DECISION_TS = dt.datetime.utcnow()
            _LAST_DECISION_DATA = {k: v for k, v in fresh.items() if k != "reason"}
            return fresh
        except Exception as e:
            last_exc = e
            await asyncio.sleep(1.5)
    # フォールバック：直近 5 分以内の決定を再利用、なければデフォルト構成を返す
    now = dt.datetime.utcnow()
    if _LAST_DECISION_TS and _LAST_DECISION_DATA:
        age = (now - _LAST_DECISION_TS).total_seconds()
        if age <= _REUSE_WINDOW_SECONDS:
            reused = dict(_LAST_DECISION_DATA)
            reused["reason"] = "reuse_previous"
            logger.warning(
                "GPT decision failed (%s); reusing previous decision from %.0fs ago.",
                str(last_exc),
                age,
            )
            return reused

    try:
        heuristic = heuristic_decision(payload, _LAST_DECISION_DATA)
    except Exception as heur_exc:  # pragma: no cover - defensive path
        logger.error(
            "GPT decision failed (%s) and heuristic fallback errored (%s); "
            "falling back to static configuration.",
            str(last_exc),
            heur_exc,
        )
        result = dict(_FALLBACK_DECISION)
        _LAST_DECISION_TS = now
        _LAST_DECISION_DATA = {k: v for k, v in result.items() if k != "reason"}
        return result

    logger.warning(
        "GPT decision failed (%s); using heuristic fallback decision.",
        str(last_exc),
    )
    _LAST_DECISION_TS = now
    _LAST_DECISION_DATA = {k: v for k, v in heuristic.items() if k != "reason"}
    return heuristic


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
        "news_short": [],
        "news_long": [],
        "perf": {"macro_pf": 1.3, "micro_pf": 1.1},
    }

    res = asyncio.run(get_decision(dummy))
    pprint.pp(res)
