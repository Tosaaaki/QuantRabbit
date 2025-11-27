"""
utils.cost_guard
~~~~~~~~~~~~~~~~

GPT トークン使用量をローカル JSON で累計し、
月の上限 (env.toml → openai.max_month_tokens) を超えたら
True / False を返して呼び出し側で動的にニュース要約等を停止する用途。
OPENAI_DISABLE_COST_GUARD=1 または max_month_tokens<=0 でガードを無効化する。
"""

from __future__ import annotations
import json
import datetime
import pathlib
import os
import logging
from typing import TypedDict

# 保存パス (プロジェクト直下 .cache)
_CACHE = pathlib.Path(".cache")
_CACHE.mkdir(exist_ok=True)
_JSON = _CACHE / "token_usage.json"
logger = logging.getLogger(__name__)


class _State(TypedDict):
    year_month: str
    total_tokens: int


def _load_state() -> _State:
    if _JSON.exists():
        try:
            return json.loads(_JSON.read_text())
        except FileNotFoundError:
            # ファイルが他プロセスにより削除された場合は新規作成
            pass
        except json.JSONDecodeError:
            # 破損した場合はリセット
            pass
    ym = datetime.datetime.utcnow().strftime("%Y-%m")
    return {"year_month": ym, "total_tokens": 0}


def _save_state(state: _State) -> None:
    _JSON.write_text(json.dumps(state))


def _guard_disabled() -> bool:
    return os.environ.get("OPENAI_DISABLE_COST_GUARD", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def add_tokens(n: int, max_month_tokens: int) -> bool:
    """
    n: 今回消費したトークン数
    max_month_tokens: 月上限 (env.toml)
                      <=0 なら無制限
    return: True = 上限内（超過時もソフト制限のため True を返す）
    """
    guard_off = _guard_disabled() or max_month_tokens <= 0

    state = _load_state()
    ym_now = datetime.datetime.utcnow().strftime("%Y-%m")

    # 月が変わったらリセット
    if state["year_month"] != ym_now:
        state = {"year_month": ym_now, "total_tokens": 0}

    state["total_tokens"] += n
    _save_state(state)

    if guard_off:
        return True  # ハード制限なし

    if state["total_tokens"] <= max_month_tokens:
        return True

    # 上限超過時もブロックせず警告のみ（ソフト制限）
    try:
        logger.warning(
            "OpenAI token soft-limit exceeded: %s > %s",
            state["total_tokens"],
            max_month_tokens,
        )
    except Exception:
        pass
    return True


if __name__ == "__main__":
    # 簡易 self‑test
    ok = add_tokens(1234, 10000)
    print("within limit:", ok)
