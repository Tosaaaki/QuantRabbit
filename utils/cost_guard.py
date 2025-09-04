"""
utils.cost_guard
~~~~~~~~~~~~~~~~

GPT トークン使用量をローカル JSON で累計し、
月の上限 (env.toml → openai.max_month_tokens) を超えたら
True / False を返して呼び出し側で動的にニュース要約等を停止する用途。
"""

from __future__ import annotations
import json
import datetime
import pathlib
from typing import TypedDict

# 保存パス (プロジェクト直下 .cache)
_CACHE = pathlib.Path(".cache")
_CACHE.mkdir(exist_ok=True)
_JSON = _CACHE / "token_usage.json"


class _State(TypedDict):
    year_month: str
    total_tokens: int
    total_usd: float


def _load_state() -> _State:
    if _JSON.exists():
        d = json.loads(_JSON.read_text())
        # 後方互換（旧ファイルに total_usd が無い場合）
        if "total_usd" not in d:
            d["total_usd"] = 0.0
        return d
    ym = datetime.datetime.utcnow().strftime("%Y-%m")
    return {"year_month": ym, "total_tokens": 0, "total_usd": 0.0}


def _save_state(state: _State) -> None:
    _JSON.write_text(json.dumps(state))


def add_tokens(n: int, max_month_tokens: int) -> bool:
    """
    n: 今回消費したトークン数
    max_month_tokens: 月上限 (env.toml)
    return: True = 上限内, False = 超過
    """
    state = _load_state()
    ym_now = datetime.datetime.utcnow().strftime("%Y-%m")

    # 月が変わったらリセット
    if state["year_month"] != ym_now:
        state = {"year_month": ym_now, "total_tokens": 0, "total_usd": 0.0}

    state["total_tokens"] += n
    _save_state(state)

    return state["total_tokens"] <= max_month_tokens


def within_budget_usd(max_month_usd: float) -> bool:
    """月次USD上限の事前チェック（呼び出し前プリフライト）"""
    state = _load_state()
    ym_now = datetime.datetime.utcnow().strftime("%Y-%m")
    if state["year_month"] != ym_now:
        return True
    return float(state.get("total_usd", 0.0)) <= float(max_month_usd)


def add_cost(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    price_in_per_m: float,
    price_out_per_m: float,
    max_month_usd: float,
) -> bool:
    """
    OpenAIのusageからUSDコストを加算して上限判定。
    price_*_per_m は 1M tokens あたりのUSD（例: 0.15 / 0.60）
    戻り値: True=上限内, False=超過
    """
    state = _load_state()
    ym_now = datetime.datetime.utcnow().strftime("%Y-%m")
    if state["year_month"] != ym_now:
        state = {"year_month": ym_now, "total_tokens": 0, "total_usd": 0.0}

    # トークン合計も引き続き記録（デバッグ用途）
    used_tok = int(prompt_tokens) + int(completion_tokens)
    state["total_tokens"] += used_tok

    # USD加算
    cost = (prompt_tokens / 1_000_000.0) * float(price_in_per_m) + (
        completion_tokens / 1_000_000.0
    ) * float(price_out_per_m)
    state["total_usd"] = float(state.get("total_usd", 0.0)) + float(cost)

    _save_state(state)
    return float(state["total_usd"]) <= float(max_month_usd)


if __name__ == "__main__":
    # 簡易 self‑test
    ok = add_tokens(1234, 10000)
    print("within limit:", ok)
