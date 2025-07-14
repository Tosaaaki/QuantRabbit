"""
analysis.gpt_decider
~~~~~~~~~~~~~~~~~~~~
OpenAI API を呼び出し、JSON スキーマ検証・
フォールバック・トークン課金記録までを一括で行うハイレベル関数。
"""

from __future__ import annotations

import asyncio
import json
import openai

from typing import Dict

from utils.cost_guard import add_tokens
from analysis.gpt_prompter import (
    build_messages,
    OPENAI_MODEL as MODEL,
    MAX_TOKENS_MONTH,
)


_SCHEMA = {
    "focus_tag": str,
    "weight_macro": float,
    "ranked_strategies": list,
}


class GPTTimeout(Exception): ...


async def call_openai(payload: Dict) -> Dict:
    """非同期で GPT を呼ぶ → dict を返す（フォールバック不要値は None）"""
    # コストガード
    if not add_tokens(0, MAX_TOKENS_MONTH):
        raise RuntimeError("GPT token limit exceeded")

    msgs = build_messages(payload)

    try:
        resp = await openai.ChatCompletion.acreate(
            model=MODEL,
            messages=msgs,
            temperature=0.2,
            max_tokens=120,
            timeout=7,
        )
    except Exception as e:  # API 障害や timeout
        raise GPTTimeout(str(e)) from e

    usage_in = resp.usage.prompt_tokens
    usage_out = resp.usage.completion_tokens
    add_tokens(usage_in + usage_out, MAX_TOKENS_MONTH)

    content = resp.choices[0].message.content.strip()
    # シングル行に余計な ```json``` ブロックが付く場合がある
    content = content.lstrip("```json").rstrip("```").strip()

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
    return data


# ------------ 自前フォールバック ------------
_FALLBACK = {
    "focus_tag": "hybrid",
    "weight_macro": 0.5,
    "ranked_strategies": ["TrendMA", "Donchian55", "BB_RSI"],
}


async def get_decision(payload: Dict) -> Dict:
    """
    上位ラッパ：GPT 呼び出し + 失敗時にフォールバックを返す
    """
    try:
        return await asyncio.wait_for(call_openai(payload), timeout=9)
    except Exception as e:
        print("GPT error -> fallback:", e)
        return _FALLBACK


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
