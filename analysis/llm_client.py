"""analysis.llm_client
~~~~~~~~~~~~~~~~~~~~~~~
ニュース要約などで必ず OpenAI API を利用するヘルパ。
"""

from __future__ import annotations

import os
import pathlib
from typing import Tuple

import toml
from openai import OpenAI

DEFAULT_OPENAI_MODEL = "gpt-5-nano"

_CONFIG_PATH = pathlib.Path(__file__).parent.parent / "config/env.toml"
if not _CONFIG_PATH.exists():
    raise FileNotFoundError(f"Configuration file not found at {_CONFIG_PATH}")

_config = toml.load(_CONFIG_PATH)
_llm_cfg = {k: v for k, v in (_config.get("llm") or {}).items()}


def _resolve_model() -> str:
    return (
        os.environ.get("OPENAI_SUMMARIZER_MODEL")
        or _llm_cfg.get("model")
        or DEFAULT_OPENAI_MODEL
    )


def _resolve_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY") or _llm_cfg.get("api_key")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for llm_client. Local LLM fallback is disabled."
        )
    return api_key


def _build_client() -> Tuple[OpenAI, str]:
    api_key = _resolve_api_key()
    base_url = os.environ.get("OPENAI_BASE_URL") or _llm_cfg.get("base_url") or None
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    model = _resolve_model()
    return client, model


CLIENT, MODEL = _build_client()


def summarize(text: str) -> str:
    """OpenAI API でニュース要約を返す。"""

    system_prompt = "You are a concise news summarizer. Respond in Japanese in 3 bullet points."

    response = CLIENT.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        max_completion_tokens=400,
    )
    choice = (response.choices or [None])[0]
    if not choice or not choice.message or not choice.message.content:
        raise RuntimeError("OpenAI summarization returned empty response")
    return choice.message.content.strip()

# ---------- CLI self-test ----------
if __name__ == "__main__":
    sample_text = """
    In a move that shocked markets, the central bank of a major economic powerhouse announced a surprise interest rate hike of 50 basis points.
    The decision was made to combat rising inflation, which has been a persistent issue for the past several quarters.
    Analysts are now scrambling to predict the short-term and long-term effects on the global economy.
    """
    print("--- Testing OpenAI Summarization ---")
    summary = summarize(sample_text)
    print(f"Original Text:\n{sample_text}")
    print(f"\nSummary:\n{summary}")
