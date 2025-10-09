import json
from typing import Dict, List
from utils.secrets import get_secret

# ---------- 読み込み：env.toml ----------
try:
    OPENAI_MODEL = get_secret("openai_model")
except Exception:
    OPENAI_MODEL = "gpt-4o-mini"
try:
    MAX_TOKENS_MONTH = int(get_secret("openai_max_month_tokens"))
except Exception:
    MAX_TOKENS_MONTH = 500_000


def build_messages(payload: Dict) -> List[Dict]:
    """payload を英語説明文に整形し、OpenAI 用メッセージを返す

    改善点:
    - 目的/制約/利用可能戦略を明示し、判断の一貫性を高める
    - JSON スキーマを明示し、出力バリデーションを容易にする
    """

    # context lines
    lines = [
        f"UTC timestamp: {payload.get('ts')}",
        f"Macro regime: {payload.get('reg_macro')}",
        f"Micro regime: {payload.get('reg_micro')}",
        f"Technical factors M1: {json.dumps(payload.get('factors_m1', {}))}",
        f"Technical factors H4: {json.dumps(payload.get('factors_h4', {}))}",
        f"Short-term news: {json.dumps(payload.get('news_short', []))}",
        f"Long-term news: {json.dumps(payload.get('news_long', []))}",
        f"Performance metrics: {json.dumps(payload.get('perf', {}))}",
        f"Event window active: {bool(payload.get('event_soon'))}",
    ]

    # strategy catalog (簡潔に用途と前提を共有)
    strategy_notes = (
        "Available strategies (choose and rank):\n"
        "- TrendMA: Favors strong H4 trend; follows pullbacks.\n"
        "- Donchian55: Breakout bias; avoid tight ranges.\n"
        "- BB_RSI: Mean-reversion in calm ranges (low ADX, narrow BBW).\n"
        "- NewsSpikeReversal: Only around impactful events; fade overreactions.\n"
        "For each strategy you may set directives with fields enabled(bool) and risk_bias(0.3-1.7)."
    )

    user_content = (
        "Here is the latest market context (USD/JPY focus):\n" + "\n".join(lines) + "\n\n" + strategy_notes
    )

    # instruction: goal, constraints, schema, and deterministic JSON-only output
    system_content = (
        "You are an FX trading assistant optimizing daily P/L (+100 pips target) "
        "while minimizing drawdowns. Consider H4 (macro) vs M1 (micro) regimes, "
        "news/event proximity and recent performance.\n"
        "Constraints:\n"
        "- If an impactful event is within ±30 minutes, disallow micro except NewsSpikeReversal.\n"
        "- Prefer macro weight when H4 shows strong trend; reduce after poor macro PF.\n"
        "- In tight ranges (low ADX, low BBW), prioritize BB_RSI; avoid breakouts.\n"
        "- In emerging momentum (rising ADX + expanding BBW), prefer Donchian55.\n"
        "Output strictly the following JSON schema with no extra text: {\n"
        "  'focus_tag': 'micro'|'macro'|'hybrid'|'event',\n"
        "  'weight_macro': number in [0,1] rounded to 0.05,\n"
        "  'ranked_strategies': string[] subset of ['TrendMA','Donchian55','BB_RSI','NewsSpikeReversal'],\n"
        "  'strategy_directives': {strategy: {'enabled': bool, 'risk_bias': number}}\n"
        "}."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
