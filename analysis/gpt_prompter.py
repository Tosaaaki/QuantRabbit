import json
from typing import Dict, List
from utils.secrets import get_secret

# ---------- 読み込み：env.toml ----------
try:
    OPENAI_MODEL = get_secret("openai_decider_model")
except Exception:
    try:
        OPENAI_MODEL = get_secret("openai_model")
    except Exception:
        OPENAI_MODEL = "gpt-5-mini"
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

    baseline_focus = payload.get("focus_baseline")
    if baseline_focus:
        lines.append(f"Baseline focus suggestion: {json.dumps(baseline_focus)}")

    noop_streak = payload.get("noop_streak")
    if noop_streak is not None:
        lines.append(f"No-trade streak (minutes): {noop_streak}")

    decision_hints = payload.get("decision_hints")
    if decision_hints:
        lines.append(f"Decision hints: {json.dumps(decision_hints)}")

    # strategy catalog (簡潔に用途と前提を共有)
    strategy_notes = (
        "Available strategies (choose and rank):\n"
        "- TrendMA: Favors strong H4 trend; follows pullbacks.\n"
        "- Donchian55: Breakout bias; avoid tight ranges.\n"
        "- BB_RSI: Mean-reversion in calm ranges (low ADX, narrow BBW).\n"
        "- MicroTrendPullback: Quick continuation plays after shallow pullbacks.\n"
        "- NewsSpikeReversal: Only around impactful events; fade overreactions.\n"
        "Scalp pocket trades via rules in parallel; adjust weight_scalp to boost or cap its activity.\n"
        "For each strategy you may set directives with fields enabled(bool) and risk_bias(0.3-1.7)."
    )

    user_content = (
        "Here is the latest market context (USD/JPY focus):\n" + "\n".join(lines) + "\n\n" + strategy_notes
    )

    # instruction: goal, constraints, schema, and deterministic JSON-only output
    system_content = (
        "You are an FX trading assistant targeting +100 pips per day while respecting drawdown guardrails. "
        "Blend macro (H4) and micro (M1) context, scheduled events and recent performance to deliver proactive trade directives.\n"
        "Decision policy:\n"
        "- Lean into decision_hints.regime_bias (e.g. 'macro_trend', 'micro_breakout', 'range_reversion') when ranking strategies and adjusting weight_macro/weight_scalp.\n"
        "- decision_hints.loss_recovery lists pockets with negative performance; prioritise those pockets and set their risk_bias between 1.1 and 1.3 until recovered.\n"
        "- noop_streak >= 2 means the system has been idle; never leave ranked_strategies empty and keep at least one macro and one micro strategy enabled unless event_soon blocks micro.\n"
        "- If an impactful event is within ±30 minutes, only allow NewsSpikeReversal on micro, otherwise focus on macro trend followers.\n"
        "- Tight ranges (low ADX and narrow BBW) favour BB_RSI or MicroTrendPullback; rising ADX with expanding BBW favours Donchian55 and TrendMA.\n"
        "- Use focus_baseline as a starting point but shift weight_macro by up to ±0.15 and weight_scalp by ±0.05 to accelerate recovery or express conviction. Round each to the nearest 0.05 and keep weight_macro + weight_scalp ≤ 0.9 so micro retains flow.\n"
        "- Disable strategies only when clearly unsuitable; otherwise leave them enabled with calibrated risk_bias values (0.7-1.3 typical).\n"
        "Output strictly the following JSON schema with no extra text: {\n"
        "  'focus_tag': 'micro'|'macro'|'hybrid'|'event',\n"
        "  'weight_macro': number in [0,1] rounded to 0.05,\n"
        "  'weight_scalp': number in [0,1] rounded to 0.05,\n"
        "  'ranked_strategies': string[] subset of ['TrendMA','Donchian55','BB_RSI','NewsSpikeReversal','MicroTrendPullback'],\n"
        "  'strategy_directives': {strategy: {'enabled': bool, 'risk_bias': number}}\n"
        "}."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
