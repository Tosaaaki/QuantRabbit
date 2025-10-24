import json
from typing import Dict, List
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
    # Final fallback to GPT-5 mini so fresh installs default to the new tier.
    return "gpt-5-mini"


# ---------- 読み込み：env.toml ----------
OPENAI_MODEL = _resolve_model()
MAX_TOKENS_MONTH = int(get_secret("openai_max_month_tokens"))


def build_messages(payload: Dict) -> List[Dict]:
    """payload を英語説明文に整形し、OpenAI 用メッセージを返す"""

    # payload 内の情報を英語で説明文化する
    lines = [
        f"UTC timestamp: {payload.get('ts')}",
        f"Macro regime: {payload.get('reg_macro')}",
        f"Micro regime: {payload.get('reg_micro')}",
        f"Technical factors M1: {json.dumps(payload.get('factors_m1', {}))}",
        f"Technical factors H4: {json.dumps(payload.get('factors_h4', {}))}",
        f"Short-term news: {json.dumps(payload.get('news_short', []))}",
        f"Long-term news: {json.dumps(payload.get('news_long', []))}",
        f"Performance metrics: {json.dumps(payload.get('perf', {}))}",
    ]

    user_content = "Here is the latest market data:\n" + "\n".join(lines)

    # GPT への指示は英語で、JSON 形式の回答のみを求める
    system_content = (
        "You are an FX trading assistant for USD/JPY. Respond only with a JSON "
        "object that contains the keys 'focus_tag', 'weight_macro', "
        "'weight_scalp', and 'ranked_strategies'. "
        "Constraints:\n"
        "- focus_tag must be one of ['micro', 'macro', 'hybrid', 'event'].\n"
        "- weight_macro must be a float between 0 and 1 inclusive.\n"
        "- weight_scalp must be a float between 0 and 1 inclusive (set to 0 when scalping should be disabled).\n"
        "- The sum of weight_macro and weight_scalp must not exceed 1.0. Remaining weight implicitly belongs to the micro pocket.\n"
        "- ranked_strategies must be an array containing zero or more of "
        "['TrendMA', 'Donchian55', 'BB_RSI', 'NewsSpikeReversal', 'M1Scalper', 'RangeFader', 'PulseBreak'] ordered from "
        "highest to lowest priority. Never invent other strategy names."
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    return messages
