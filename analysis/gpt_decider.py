"""
analysis.gpt_decider
~~~~~~~~~~~~~~~~~~~~
OpenAI API を呼び出し、JSON スキーマ検証・
フォールバック・トークン課金記録までを一括で行うハイレベル関数。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Dict, List

from openai import AsyncOpenAI

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

logger = logging.getLogger(__name__)


_SCHEMA = {
    "focus_tag": str,
    "weight_macro": (int, float),
    "weight_scalp": (int, float),
    "ranked_strategies": list,
}

_STRATEGY_KEYS = ("TrendMA", "Donchian55", "BB_RSI", "NewsSpikeReversal", "MicroTrendPullback")
_DEFAULT_DIRECTIVE = {"enabled": True, "risk_bias": 1.0}
_LOG_TRUNCATE = 1200


def _compact(obj: Dict | List | None, limit: int = _LOG_TRUNCATE) -> str:
    """Serialize obj for logging while keeping payload manageable."""

    try:
        text = json.dumps(obj, ensure_ascii=False, default=str)
    except TypeError:
        text = str(obj)
    if limit and len(text) > limit:
        return f"{text[: limit - 3]}..."
    return text


def _get_float_secret(candidates: tuple[str, ...], default: float) -> float:
    for key in candidates:
        try:
            return float(get_secret(key))
        except Exception:
            continue
    return default


def _model_candidates() -> List[str]:
    seen: set[str] = set()
    out: List[str] = []

    def _push(candidate: str | None) -> None:
        if not candidate:
            return
        cand = candidate.strip()
        if not cand or cand in seen:
            return
        seen.add(cand)
        out.append(cand)

    _push(MODEL)

    for key in ("openai_decider_fallback_model", "openai_model"):
        try:
            _push(get_secret(key))
        except Exception:
            continue

    env_chain = os.environ.get("GPT_DECIDER_FALLBACK_MODELS")
    if env_chain:
        for item in env_chain.split(","):
            _push(item)

    for default in ("gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"):
        _push(default)

    return out


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
        _PRICE_IN_PER_M = _get_float_secret(
            ("openai_decider_cost_per_million_input", "openai_cost_per_million_input"),
            0.25,
        )  # default aligns with GPT-5 mini pricing
    if _PRICE_OUT_PER_M is None:
        _PRICE_OUT_PER_M = _get_float_secret(
            ("openai_decider_cost_per_million_output", "openai_cost_per_million_output"),
            2.0,
        )
    if _MAX_MONTH_USD is None:
        _MAX_MONTH_USD = _get_float_secret(
            ("openai_decider_max_month_usd", "openai_max_month_usd"),
            60.0,
        )

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

    errors: list[str] = []
    for idx, model in enumerate(_model_candidates()):
        try:
            resp = await _client.chat.completions.create(
                model=model,
                messages=msgs,
                max_completion_tokens=120,
                timeout=7,
            )
        except Exception as e:  # API 障害や timeout
            errors.append(f"{model}: {e}")
            logger.warning("[GPT_DECIDER] model call failed: %s", errors[-1])
            continue

        usage_in = int(resp.usage.prompt_tokens or 0)
        usage_out = int(resp.usage.completion_tokens or 0)
        add_tokens(usage_in + usage_out, MAX_TOKENS_MONTH)
        add_cost(
            model=model,
            prompt_tokens=usage_in,
            completion_tokens=usage_out,
            price_in_per_m=_PRICE_IN_PER_M,
            price_out_per_m=_PRICE_OUT_PER_M,
            max_month_usd=_MAX_MONTH_USD,
        )

        content = (resp.choices[0].message.content or "").strip()
        content = content.lstrip("```json").rstrip("```").strip()
        if not content:
            errors.append(f"{model}: empty content")
            logger.warning("[GPT_DECIDER] model returned empty content: %s", model)
            continue

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            errors.append(f"{model}: invalid JSON {content}")
            logger.warning("[GPT_DECIDER] invalid JSON from %s: %s", model, content)
            continue

        # スキーマ簡易検証
        try:
            for k, typ in _SCHEMA.items():
                if k not in data:
                    raise ValueError(f"key {k} missing")
                if not isinstance(data[k], typ):
                    raise ValueError(f"{k} type error")
        except ValueError as e:
            errors.append(f"{model}: {e}")
            logger.warning("[GPT_DECIDER] schema validation failed for %s: %s", model, e)
            continue

        data["weight_macro"] = round(float(data["weight_macro"]), 2)
        weight_scalp = data.get("weight_scalp")
        try:
            weight_scalp = float(weight_scalp)
        except (TypeError, ValueError):
            weight_scalp = 0.12
        data["weight_scalp"] = round(max(0.0, min(weight_scalp, 0.4)), 2)
        directives_raw = data.get("strategy_directives") if isinstance(data, dict) else None
        data["strategy_directives"] = _normalize_strategy_directives(directives_raw)
        ranked = []
        for name in data.get("ranked_strategies", []):
            if isinstance(name, str) and name in _STRATEGY_KEYS:
                ranked.append(name)
        if not ranked:
            ranked = list(_STRATEGY_KEYS)
        data["ranked_strategies"] = ranked

        if idx > 0:
            logger.warning(
                "[GPT_DECIDER] primary model %s failed; used fallback %s",
                MODEL,
                model,
            )
        return data

    raise GPTTimeout("; ".join(errors))


# ------------ 自前フォールバック ------------
_FALLBACK = {
    "focus_tag": "hybrid",
    "weight_macro": 0.5,
    "weight_scalp": 0.12,
    "ranked_strategies": ["TrendMA", "Donchian55", "BB_RSI", "MicroTrendPullback"],
    "strategy_directives": _normalize_strategy_directives({}),
}


async def get_decision(payload: Dict, *, force_fallback: bool = False) -> Dict:
    """
    上位ラッパ：GPT 呼び出し + 失敗時にフォールバックを返す
    """
    logger.info("[GPT_DECIDER] request=%s", _compact(payload))
    if force_fallback:
        logger.info("[GPT_DECIDER] forced fallback requested; skipping model call")
        return _FALLBACK
    if os.environ.get("GPT_DISABLE_ALL") or os.environ.get("DISABLE_GPT_DECIDER"):
        logging.info("[GPT_DECIDER] disabled via env; returning fallback")
        return _FALLBACK
    try:
        result = await asyncio.wait_for(call_openai(payload), timeout=9)
        logger.info("[GPT_DECIDER] response=%s", _compact(result))
        return result
    except Exception as e:
        logger.warning("[GPT_DECIDER] fallback due to error: %s payload=%s", e, _compact(payload))
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
