#!/usr/bin/env python3
"""
Secret Manager / 環境変数 から必要なキーを収集し、config/env.toml を生成する補助スクリプト。

優先度: OS環境変数 > Secret Manager > 既存 env.toml の値

必要なキー（必要に応じて増やす）:
  - oanda_account_id, oanda_token, oanda_practice, oanda_hedging_enabled
  - openai_api_key, openai_model, openai_model_decider, openai_model_summarizer, openai_max_month_tokens
  - gcp_project_id, gcp_pubsub_topic
"""
from __future__ import annotations
import os
import sys
import toml
import pathlib
from typing import Dict

from utils.secrets import get_secret, _load_toml  # type: ignore

TARGET = pathlib.Path("config/env.toml")

KEYS = [
    "oanda_account_id",
    "oanda_token",
    "oanda_practice",
    "oanda_hedging_enabled",
    "openai_api_key",
    "openai_model",
    "openai_model_decider",
    "openai_model_summarizer",
    "openai_max_month_tokens",
    "gcp_project_id",
    "gcp_pubsub_topic",
]


def main() -> int:
    out: Dict[str, str] = {}

    # 既存値をベースにしつつ上書き
    existing = {}
    if TARGET.exists():
        try:
            existing = toml.loads(TARGET.read_text())
        except Exception:
            existing = {}

    for k in KEYS:
        try:
            v = get_secret(k)
            out[k] = str(v)
            continue
        except Exception:
            # 取得失敗→既存値を温存
            if k in existing:
                out[k] = str(existing[k])

    if not out:
        print("No secrets resolved. Ensure env vars or Secret Manager entries exist.")
        return 1

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(toml.dumps(out), encoding="utf-8")
    print(f"Wrote {TARGET} with {len(out)} keys.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
