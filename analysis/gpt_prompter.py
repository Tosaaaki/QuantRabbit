import openai
import json
from typing import Dict, List
from utils.secrets import get_secret

# ---------- 読み込み：env.toml ----------
openai.api_key = get_secret("openai_api_key")
OPENAI_MODEL = get_secret("openai_model")
MAX_TOKENS_MONTH = int(get_secret("openai_max_month_tokens"))


def build_messages(payload: Dict) -> List[Dict]:
    """payload を英語説明文に整形し、OpenAI 用メッセージを返す"""

    # payload 内の情報を英語で説明文化する
    lines = [
        f"UTC timestamp: {payload.get('ts')}",
        f"Macro regime: {payload.get('reg_macro')}",
        f"Micro regime: {payload.get('reg_micro')}",
        f"Technical factors: {json.dumps(payload.get('factors', {}))}",
        f"Short-term news: {json.dumps(payload.get('news_short', []))}",
        f"Long-term news: {json.dumps(payload.get('news_long', []))}",
        f"Performance metrics: {json.dumps(payload.get('perf', {}))}",
    ]

    user_content = "Here is the latest market data:\n" + "\n".join(lines)

    # GPT への指示は英語で、JSON 形式の回答のみを求める
    system_content = (
        "You are an FX trading assistant. Based on the provided market "
        "information, reply only with a JSON object containing the keys "
        "'focus_tag', 'weight_macro', and 'ranked_strategies'."
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    return messages
