"""Prompt construction for the manual trading assistant."""

from __future__ import annotations

import json
from typing import Dict, List

from .context import FrameContext, ManualContext


def _compact_factors(frame: FrameContext) -> Dict[str, float]:
    keys = (
        "ma10",
        "ma20",
        "ema20",
        "rsi",
        "atr",
        "adx",
        "bbw",
        "macd",
        "macd_signal",
        "macd_hist",
        "open",
        "high",
        "low",
        "close",
    )
    out: Dict[str, float] = {}
    for key in keys:
        val = frame.factors.get(key)
        if val is None:
            continue
        out[key] = round(float(val), 6)
    return out


def _format_news(ctx: ManualContext) -> List[Dict[str, str | int | None]]:
    items: List[Dict[str, str | int | None]] = []
    for horizon in ("short", "long"):
        for src in ctx.latest_news.get(horizon, []) or []:
            items.append(
                {
                    "horizon": horizon,
                    "summary": src.get("summary"),
                    "impact": src.get("impact"),
                    "sentiment": src.get("sentiment"),
                    "pair_bias": src.get("pair_bias"),
                    "event_time": src.get("event_time"),
                    "ts": src.get("ts"),
                }
            )
    return items


def build_manual_messages(ctx: ManualContext) -> List[Dict[str, str]]:
    """Construct OpenAI chat messages describing the manual trading context."""

    macro_payload = {
        "timeframe": ctx.macro.timeframe,
        "regime": ctx.macro.regime,
        "factors": _compact_factors(ctx.macro),
        "price_snapshot": ctx.macro.price_snapshot,
        "recent_closes": [round(item["close"], 5) for item in ctx.macro.recent_ohlc],
    }
    micro_payload = {
        "timeframe": ctx.micro.timeframe,
        "regime": ctx.micro.regime,
        "factors": _compact_factors(ctx.micro),
        "price_snapshot": ctx.micro.price_snapshot,
        "recent_closes": [round(item["close"], 5) for item in ctx.micro.recent_ohlc],
    }

    user_payload = {
        "instrument": ctx.instrument,
        "timestamp": ctx.timestamp,
        "event_window": bool(ctx.event_window),
        "macro": macro_payload,
        "micro": micro_payload,
        "news_items": _format_news(ctx),
    }

    system_prompt = (
        "You are QuantRabbit's USD/JPY manual trading co-pilot. "
        "Your job is to convert technical context, regimes, and recent news into a clear plan "
        "that a human discretionary trader can review before submitting orders. "
        "Highlight alignment or conflicts between the H4 (macro) and M1 (micro) views, "
        "call out event risks, and keep trade ideas high-quality. Edge cases: if data suggests "
        "standing aside, say so explicitly."
    )

    schema_prompt = (
        "Return **only** a JSON object with the following schema (no prose):\n"
        "{\n"
        "  'bias': 'long'|'short'|'neutral',\n"
        "  'confidence': integer 0-100,\n"
        "  'market_view': { 'macro': string, 'micro': string, 'news': string },\n"
        "  'trade_ideas': [\n"
        "     {\n"
        "       'label': string,\n"
        "       'direction': 'long'|'short',\n"
        "       'style': 'market'|'limit'|'stop',\n"
        "       'entry_note': string,\n"
        "       'stop_loss': string,\n"
        "       'take_profit': string,\n"
        "       'risk_reward': string,\n"
        "       'rationale': string,\n"
        "       'conditions': [string]\n"
        "     }\n"
        "  ] (max 2 ideas; use empty list if no trade),\n"
        "  'risk_notes': [string],\n"
        "  'next_steps': [string]\n"
        "}.\n"
        "Use price levels with 3 decimal places when possible. "
        "If in an economic-event-safe mode (event_window true), emphasise caution and note forbidden setups."
    )

    user_content = (
        "Context JSON:\n" + json.dumps(user_payload, ensure_ascii=False, indent=2)
    )

    return [
        {"role": "system", "content": system_prompt + "\n\n" + schema_prompt},
        {"role": "user", "content": user_content},
    ]
