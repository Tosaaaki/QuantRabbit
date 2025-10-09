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

from utils.cost_guard import add_tokens, add_cost, within_budget_usd
from analysis.gpt_prompter import (
    build_messages,
    OPENAI_MODEL as MODEL,
    MAX_TOKENS_MONTH,
)
from utils.secrets import get_secret

# Lazy client; avoid import-time failure when OPENAI_API_KEY is missing
_client: AsyncOpenAI | None = None
_PRICE_IN_PER_M = None
_PRICE_OUT_PER_M = None
_MAX_MONTH_USD = None


_SCHEMA = {
    "focus_tag": str,
    "weight_macro": float,
    "ranked_strategies": list,
}

_STRATEGY_KEYS = ("TrendMA", "Donchian55", "BB_RSI", "NewsSpikeReversal")
_DEFAULT_DIRECTIVE = {"enabled": True, "risk_bias": 1.0}


def _normalize_strategy_directives(raw: Dict | None) -> Dict[str, Dict[str, float | bool]]:
    """Ensure directives cover all known strategies with sane values."""

    directives: Dict[str, Dict[str, float | bool]] = {}
    if isinstance(raw, dict):
        for name, cfg in raw.items():
            if name not in _STRATEGY_KEYS:
                continue
            enabled = True
            risk = 1.0
            if isinstance(cfg, dict):
                enabled = bool(cfg.get("enabled", True))
                try:
                    risk = float(cfg.get("risk_bias", 1.0))
                except (TypeError, ValueError):
                    risk = 1.0
            elif isinstance(cfg, bool):
                enabled = cfg
            elif isinstance(cfg, (int, float)):
                risk = float(cfg)
            # clamp risk multiplier to avoid extreme leverage suggestions
            risk = round(max(0.1, min(risk, 2.0)), 2)
            directives[name] = {"enabled": enabled, "risk_bias": risk}

    for name in _STRATEGY_KEYS:
        directives.setdefault(name, dict(_DEFAULT_DIRECTIVE))
    return directives


class GPTTimeout(Exception): ...


async def call_openai(payload: Dict) -> Dict:
    """非同期で GPT を呼ぶ → dict を返す（フォールバック不要値は None）"""
    # 価格設定の読込（初回だけ）
    global _PRICE_IN_PER_M, _PRICE_OUT_PER_M, _MAX_MONTH_USD
    if _PRICE_IN_PER_M is None:
        try:
            _PRICE_IN_PER_M = float(get_secret("openai_cost_per_million_input"))
        except Exception:
            _PRICE_IN_PER_M = 0.15  # default for gpt-4o-mini input
    if _PRICE_OUT_PER_M is None:
        try:
            _PRICE_OUT_PER_M = float(get_secret("openai_cost_per_million_output"))
        except Exception:
            _PRICE_OUT_PER_M = 0.60  # default for gpt-4o-mini output
    if _MAX_MONTH_USD is None:
        try:
            _MAX_MONTH_USD = float(get_secret("openai_max_month_usd"))
        except Exception:
            _MAX_MONTH_USD = 30.0

    # コストガード（USD上限の事前チェック）
    if not within_budget_usd(_MAX_MONTH_USD):
        raise RuntimeError("GPT USD budget exceeded")

    msgs = build_messages(payload)

    # Lazy init OpenAI client; if key is missing, raise to trigger fallback
    global _client
    if _client is None:
        try:
            api_key = None
            try:
                api_key = get_secret("openai_api_key")
            except Exception:
                api_key = None
            _client = AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()
        except Exception as e:
            raise GPTTimeout(f"OPENAI init failed: {e}")

    try:
        resp = await _client.chat.completions.create(
            model=MODEL,
            messages=msgs,
            temperature=0.2,
            max_tokens=96,
            timeout=7,
        )
    except Exception as e:  # API 障害や timeout
        raise GPTTimeout(str(e)) from e

    usage_in = int(resp.usage.prompt_tokens or 0)
    usage_out = int(resp.usage.completion_tokens or 0)
    # 互換：トークン上限の旧ロジックも維持
    add_tokens(usage_in + usage_out, MAX_TOKENS_MONTH)
    # USD課金を加算
    add_cost(
        model=MODEL,
        prompt_tokens=usage_in,
        completion_tokens=usage_out,
        price_in_per_m=_PRICE_IN_PER_M,
        price_out_per_m=_PRICE_OUT_PER_M,
        max_month_usd=_MAX_MONTH_USD,
    )

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
    directives_raw = data.get("strategy_directives") if isinstance(data, dict) else None
    data["strategy_directives"] = _normalize_strategy_directives(directives_raw)
    ranked = []
    for name in data.get("ranked_strategies", []):
        if isinstance(name, str) and name in _STRATEGY_KEYS:
            ranked.append(name)
    if not ranked:
        ranked = list(_STRATEGY_KEYS)
    data["ranked_strategies"] = ranked
    return data


# ------------ 自前フォールバック ------------
_FALLBACK = {
    "focus_tag": "hybrid",
    "weight_macro": 0.5,
    "ranked_strategies": ["TrendMA", "Donchian55", "BB_RSI"],
    "strategy_directives": _normalize_strategy_directives({}),
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
