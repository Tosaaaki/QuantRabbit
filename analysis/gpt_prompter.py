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

    news_features = payload.get("news_features") or {}
    if isinstance(news_features, dict):
        keys = (
            "news_count_total",
            "news_latest_age_minutes",
            "news_sentiment_mean",
            "news_impact_max",
        )
        news_features = {
            key: news_features.get(key)
            for key in keys
            if news_features.get(key) not in (None, "", [])
        }

    compact_payload = _prune(
        {
            "ts": payload.get("ts"),
            "regime": {
                "macro": payload.get("reg_macro"),
                "micro": payload.get("reg_micro"),
            },
            "factors": factors,
            "perf": payload.get("perf") or {},
            "news_features": news_features,
            "event_soon": payload.get("event_soon", False),
        }
    )

    user_content = (
        "Latest USD/JPY snapshot JSON:\n"
        + _compact_json(compact_payload or {})
    )

    system_content = (
        "You are an FX trading assistant for USD/JPY. Respond only with a JSON "
        "object that contains the keys 'focus_tag', 'weight_macro', and "
        "'ranked_strategies'. "
        "Constraints:\n"
        "- focus_tag must be one of ['micro', 'macro', 'hybrid', 'event'].\n"
        "- weight_macro must be a float between 0 and 1 inclusive.\n"
        "- ranked_strategies must be an array containing zero or more of "
        "['TrendMA', 'Donchian55', 'BB_RSI', 'NewsSpikeReversal', 'M1Scalper', "
        "'RangeFader', 'PulseBreak', 'ImpulseRetrace', 'MomentumBurst', "
        "'TrendMomentumMicro', 'MicroMomentumStack', 'MicroPullbackEMA', "
        "'MicroRangeBreak', 'MicroLevelReactor'] ordered from highest to lowest priority. "
        "Never invent other strategy names."
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    return messages
