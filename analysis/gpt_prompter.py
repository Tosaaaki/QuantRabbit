import openai
import json
from typing import List, Dict
from utils.secrets import get_secret

# ---------- 読み込み：env.toml ----------
openai.api_key = get_secret("openai_api_key")
OPENAI_MODEL = get_secret("openai_model")
MAX_TOKENS_MONTH = int(get_secret("openai_max_month_tokens"))


def build_messages(payload: Dict) -> List[Dict]:
    # ここにプロンプト構築ロジックを実装
    # 例:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": json.dumps(payload)},
    ]
    return messages
