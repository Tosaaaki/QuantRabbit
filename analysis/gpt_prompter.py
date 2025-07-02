"""
analysis.gpt_prompter
~~~~~~~~~~~~~~~~~~~~~
最新指標・ニュース・パフォーマンスを受け取り
OpenAI ChatCompletion 向け messages(list[dict]) を生成する。

‣ 入力を極力短縮した圧縮 JSON にまとめ、トークン消費を抑える。
"""

from __future__ import annotations

import datetime
import json
import tomllib
from typing import Dict, List

CONF = tomllib.loads(open("config/env.toml", "rb").read())
OPENAI_MODEL = CONF["openai"]["model"]

_SYS = (
    "あなたは短期為替トレーダーのアシスタントです。"
    "次のJSONに基づき、レジーム補足・戦略順位 (TrendMA, Donchian55, BB_RSI, NewsSpikeReversal) と "
    '"weight_macro" (0-1) を返してください。'
    "出力は JSON のみ。思考過程は不要。"
)

def build_messages(payload: Dict) -> List[Dict]:
    """
    payload 格納例
    -------------
    {
      "ts": "...",
      "reg_macro": "Trend",
      "reg_micro": "Range",
      "factors": {...},          # calc_core の dict
      "news_short": [...],       # 直近 3 件
      "news_long":  [...],       # 長期 5 件
      "perf": {                  # perf_monitor の dict
          "micro": {"pf": 1.2, ...},
          "macro": {"pf": 0.9, ...}
      }
    }
    """
    # 不要なキーや大きすぎる値は除外してトークンを節約
    if "factors" in payload and "candles" in payload["factors"]:
        del payload["factors"]["candles"]

    user = json.dumps(payload, separators=(",", ":"))
    return [
        {"role": "system", "content": _SYS},
        {"role": "user", "content": user},
    ]


if __name__ == "__main__":
    # quick self‑test
    pay = {
        "ts": datetime.datetime.utcnow().isoformat(timespec="seconds"),
        "reg_macro": "Trend",
        "reg_micro": "Range",
        "factors": {"ma10": 157.2, "ma20": 157.1, "adx": 30, "candles": [{"o":1,"h":2,"l":0,"c":1}]},
        "perf": {
            "micro": {"pf": 1.3, "sharpe": 0.9, "win_rate": 0.6, "avg_pips": 8.2},
            "macro": {"pf": 0.8, "sharpe": -0.2, "win_rate": 0.4, "avg_pips": -5.1},
        }
    }
    import pprint
    pprint.pp(build_messages(pay))
    # check if candles are removed
    assert "candles" not in json.loads(build_messages(pay)[1]["content"])["factors"]