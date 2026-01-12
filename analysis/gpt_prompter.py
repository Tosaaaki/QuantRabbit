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

    user_content = _compact_json(compact_payload or {})

    system_content = (
        "Return ONLY compact JSON with fields: "
        "{\"mode\":\"DEFENSIVE|TREND_FOLLOW|RANGE_SCALP|TRANSITION\","
        "\"risk_bias\":\"high|neutral|low\","
        "\"liquidity_bias\":\"tight|normal|loose\","
        "\"range_confidence\":0-1,"
        "\"pattern_hint\":[\"short\"],"
        "\"focus_tag\":\"micro|macro|hybrid|event\","
        "\"weight_macro\":0-1,"
        "\"weight_scalp\":0-1,"
        "\"forecast_bias\":\"up|down|flat\","
        "\"forecast_confidence\":0-1,"
        "\"forecast_horizon_min\":int}. "
        "Constraints: weight_macro+weight_scalp<=1.0; pattern_hint<=5 items. No markdown."
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    return messages
