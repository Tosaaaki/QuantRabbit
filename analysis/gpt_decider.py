"""
analysis.gpt_decider
~~~~~~~~~~~~~~~~~~~~
OpenAI API を呼び出し、JSON スキーマ検証・
フォールバック・トークン課金記録までを一括で行うハイレベル関数。
"""

from __future__ import annotations

import asyncio
import json
from openai import AsyncOpenAI

from typing import Dict

from utils.cost_guard import add_tokens
from utils.secrets import get_secret
from analysis.gpt_prompter import (
    build_messages,
    OPENAI_MODEL as MODEL,
    MAX_TOKENS_MONTH,
)

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
    上位ラッパ：GPT 呼び出し（フォールバックなし、リトライあり）
    """
    # 最大2回リトライ（合計最大 ~9秒）
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            return await asyncio.wait_for(call_openai(payload), timeout=9)
        except Exception as e:
            last_exc = e
            await asyncio.sleep(1.5)
    # 最後まで失敗したら例外を上位へ
    raise GPTTimeout(str(last_exc) if last_exc else "unknown error")


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
