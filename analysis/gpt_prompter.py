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
    "次のJSONに基づき、レジーム補足・戦略順位と "
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
      "perf": { "macro_pf":1.4, "micro_pf":1.1 }
    }
    """
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
        "factors": {"ma10": 157.2, "ma20": 157.1, "adx": 30},
    }
    import pprint
    pprint.pp(build_messages(pay))