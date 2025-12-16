import json
from typing import Any, Dict, List
from utils.secrets import get_secret


def _resolve_model() -> str:
    """
    Allow dedicated model override for GPT decider while remaining
    backward compatible with the legacy openai_model key.
    """
    for key in ("openai_model_decider", "openai_model"):
        try:
            value = get_secret(key)
            if value:
                return value
        except KeyError:
            continue
    # Final fallback to a lightweight model to keep latency/token costs low.
    return "gpt-4o-mini"


OPENAI_MODEL = _resolve_model()
MAX_TOKENS_MONTH = int(get_secret("openai_max_month_tokens"))


def build_messages(payload: Dict) -> List[Dict]:
    """payload を英語説明文に整形し、OpenAI 用メッセージを返す"""

    def _prune(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        if isinstance(value, dict):
            pruned: Dict[str, Any] = {}
            for key, inner in value.items():
                cleaned = _prune(inner)
                if cleaned in (None, {}, []):
                    continue
                pruned[key] = cleaned
            return pruned
        if isinstance(value, list):
            items = []
            for inner in value:
                cleaned = _prune(inner)
                if cleaned in (None, {}, []):
                    continue
                items.append(cleaned)
            return items
        return value

    def _compact_json(value: Dict[str, Any]) -> str:
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)

    factors: Dict[str, Any] = {}
    for key, label in (("factors_m1", "M1"), ("factors_h1", "H1"), ("factors_h4", "H4")):
        data = payload.get(key)
        if data:
            factors[label] = data

    compact_payload = _prune(
        {
            "ts": payload.get("ts"),
            "regime": {
                "macro": payload.get("reg_macro"),
                "micro": payload.get("reg_micro"),
            },
            "factors": factors,
            "perf": payload.get("perf") or {},
        }
    )

    user_content = (
        "Latest USD/JPY snapshot JSON:\n"
        + _compact_json(compact_payload or {})
    )

    system_content = (
        "You are an FX trading assistant for USD/JPY. Respond ONLY with a single JSON "
        "object (no markdown, no text) containing keys:\n"
        "- mode: one of ['DEFENSIVE','TREND_FOLLOW','RANGE_SCALP','TRANSITION']\n"
        "- risk_bias: one of ['high','neutral','low']\n"
        "- liquidity_bias: one of ['tight','normal','loose']\n"
        "- range_confidence: float 0.0–1.0 (higher = more range-like)\n"
        "- pattern_hint: array of zero or more short tags for recent candle patterns "
        "(examples: 'long_wick_down','bull_flag','double_top','inside_bar','engulfing_bear')\n"
        "- focus_tag: one of ['micro','macro','hybrid']\n"
        "- weight_macro: float 0.0–1.0\n"
        "- weight_scalp: float 0.0–1.0 (weight_macro + weight_scalp <= 1.0)\n"
        "Do NOT include any other keys. Output format: compact JSON, e.g. "
        "{\"mode\":\"DEFENSIVE\",\"risk_bias\":\"low\",\"liquidity_bias\":\"tight\","
        "\"range_confidence\":0.7,\"pattern_hint\":[\"inside_bar\"],"
        "\"focus_tag\":\"hybrid\",\"weight_macro\":0.4,\"weight_scalp\":0.25}"
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    return messages
