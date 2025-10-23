#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpt_guard.py
- LLM（任意）にバックテスト結果の採否レビューを頼むためのプレースホルダ
- デフォルトはヒューリスティックのみで判定（LLM は無効）
"""
import os, json
from typing import Dict, Any

def heuristic_review(agg: Dict[str, Any]) -> (bool, str):
    ok = True
    reasons = []
    if agg.get("profit_factor", 0) < 1.05:
        ok = False; reasons.append("PF<1.05")
    if agg.get("trades", 0) < 8:
        ok = False; reasons.append("trades<8")
    if agg.get("max_dd_pips", 0) > 12:
        ok = False; reasons.append("max_dd>12p")
    msg = " & ".join(reasons) if reasons else "looks ok"
    return ok, msg

def review_with_llm(agg: Dict[str, Any], use_llm: bool=False) -> (bool, str):
    # NOTE: 実装はダミー。必要なら OpenAI/Vertex などのクライアントをここで呼び出してください。
    # ここでは常にヒューリスティックにフォールバックします。
    return heuristic_review(agg)

if __name__ == "__main__":
    import sys
    data = json.load(sys.stdin)
    ok, msg = review_with_llm(data, use_llm=False)
    print(json.dumps({"approve": ok, "reason": msg}))
