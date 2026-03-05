from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


def _reload_brain_module():
    module_name = "workers.common.brain"
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def _prepare_brain(monkeypatch, tmp_path: Path):
    db_path = tmp_path / "brain_state.db"
    profile_path = tmp_path / "brain_prompt_profile.json"
    runtime_profile_path = tmp_path / "brain_runtime_param_profile.json"
    trades_path = tmp_path / "trades.db"

    monkeypatch.setenv("BRAIN_ENABLED", "1")
    monkeypatch.setenv("BRAIN_BACKEND", "ollama")
    monkeypatch.setenv("BRAIN_SAMPLE_RATE", "1.0")
    monkeypatch.setenv("BRAIN_PROMPT_AUTO_TUNE_ENABLED", "0")
    monkeypatch.setenv("BRAIN_PROMPT_PROFILE_PATH", str(profile_path))
    monkeypatch.setenv("BRAIN_RUNTIME_PARAM_PROFILE_PATH", str(runtime_profile_path))
    monkeypatch.setenv("BRAIN_RUNTIME_PARAM_AUTO_TUNE_ENABLED", "0")
    brain = _reload_brain_module()
    monkeypatch.setattr(brain, "_DB_PATH", db_path)
    monkeypatch.setattr(brain, "_TRADES_DB_PATH", trades_path)
    monkeypatch.setattr(brain, "_load_memory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(brain, "_save_memory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(brain, "log_metric", lambda *_args, **_kwargs: True)
    brain._CACHE.clear()
    brain._PROMPT_PROFILE_CACHE = (0.0, {})
    brain._RUNTIME_PARAM_PROFILE_CACHE = (0.0, {})
    return brain


def test_brain_decide_uses_ollama_backend(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_OLLAMA_MODEL", "gpt-oss:test")
    monkeypatch.setenv("BRAIN_OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
    monkeypatch.setenv("BRAIN_FAIL_POLICY", "allow")
    brain = _prepare_brain(monkeypatch, tmp_path)

    captured: dict[str, object] = {}

    def _fake_ollama(
        prompt: str,
        *,
        model: str,
        url: str,
        timeout_sec: float,
        temperature: float = 0.2,
        max_tokens: int = 256,
    ):
        captured["prompt"] = prompt
        captured["model"] = model
        captured["url"] = url
        captured["timeout_sec"] = timeout_sec
        captured["temperature"] = temperature
        captured["max_tokens"] = max_tokens
        return {
            "action": "BLOCK",
            "scale": 0.3,
            "reason": "risk",
            "memory_update": "short memo",
        }

    monkeypatch.setattr(brain, "call_ollama_chat_json", _fake_ollama)

    decision = brain.decide(
        strategy_tag="ScalpTest",
        pocket="scalp",
        side="buy",
        units=1000,
    )

    assert decision.allowed is False
    assert decision.scale == 0.0
    assert decision.reason == "risk"
    assert decision.action == "BLOCK"
    assert decision.memory == "short memo"
    assert captured["model"] == "gpt-oss:test"
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"


def test_brain_fail_policy_reduce_for_ollama_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_FAIL_POLICY", "reduce")
    brain = _prepare_brain(monkeypatch, tmp_path)

    monkeypatch.setattr(
        brain,
        "call_ollama_chat_json",
        lambda *_args, **_kwargs: None,
    )

    decision = brain.decide(
        strategy_tag="ScalpTest",
        pocket="scalp",
        side="sell",
        units=-1200,
    )

    assert decision.allowed is True
    assert decision.action == "REDUCE"
    assert decision.reason == "no_llm_reduce"
    assert decision.scale == pytest.approx(0.5)
