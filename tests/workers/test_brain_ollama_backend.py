from __future__ import annotations

import importlib
import json
import sqlite3
import sys
import time
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
    if hasattr(brain, "_FAILFAST_STATE"):
        brain._FAILFAST_STATE.clear()
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
        keep_alive=None,
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


def test_brain_failfast_skips_repeated_timeout_without_reducing_frequency(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAIN_FAILFAST_CONSECUTIVE_FAILURES", "2")
    monkeypatch.setenv("BRAIN_FAILFAST_COOLDOWN_SEC", "30")
    monkeypatch.setenv("BRAIN_FAILFAST_WINDOW_SEC", "60")
    monkeypatch.setenv("BRAIN_FAIL_POLICY", "allow")
    brain = _prepare_brain(monkeypatch, tmp_path)

    call_count = {"n": 0}

    def _fake_ollama(*_args, **_kwargs):
        call_count["n"] += 1
        return None

    monkeypatch.setattr(brain, "call_ollama_chat_json", _fake_ollama)

    first = brain.decide(
        strategy_tag="MicroLevelReactor-bounce-lower",
        pocket="micro",
        side="buy",
        units=600,
        entry_thesis={"entry_probability": 0.51, "entry_units_intent": 600},
    )
    second = brain.decide(
        strategy_tag="MicroLevelReactor-bounce-lower",
        pocket="micro",
        side="buy",
        units=620,
        entry_thesis={"entry_probability": 0.64, "entry_units_intent": 620},
    )
    third = brain.decide(
        strategy_tag="MicroLevelReactor-bounce-lower",
        pocket="micro",
        side="buy",
        units=640,
        entry_thesis={"entry_probability": 0.77, "entry_units_intent": 640},
    )

    assert first.action == "ALLOW"
    assert second.action == "ALLOW"
    assert third.action == "ALLOW"
    assert third.reason == "recent_llm_timeout_cooldown"
    assert call_count["n"] == 2

    con = sqlite3.connect(brain._DB_PATH)
    try:
        row = con.execute(
            """
            SELECT source, llm_ok, error
            FROM brain_decisions
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    finally:
        con.close()

    assert row == ("llm_fail_fast", 0, "recent_llm_timeout_cooldown")


def test_brain_failfast_resets_after_success(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_FAILFAST_CONSECUTIVE_FAILURES", "2")
    monkeypatch.setenv("BRAIN_FAILFAST_COOLDOWN_SEC", "30")
    monkeypatch.setenv("BRAIN_FAILFAST_WINDOW_SEC", "60")
    monkeypatch.setenv("BRAIN_FAIL_POLICY", "allow")
    brain = _prepare_brain(monkeypatch, tmp_path)

    responses = iter(
        [
            None,
            {
                "action": "ALLOW",
                "scale": 1.0,
                "reason": "recovered",
                "memory_update": "",
            },
            None,
        ]
    )
    call_count = {"n": 0}

    def _fake_ollama(*_args, **_kwargs):
        call_count["n"] += 1
        return next(responses)

    monkeypatch.setattr(brain, "call_ollama_chat_json", _fake_ollama)

    fail = brain.decide(
        strategy_tag="MicroLevelReactor-bounce-lower",
        pocket="micro",
        side="buy",
        units=600,
        entry_thesis={"entry_probability": 0.51, "entry_units_intent": 600},
    )
    recovered = brain.decide(
        strategy_tag="MicroLevelReactor-bounce-lower",
        pocket="micro",
        side="buy",
        units=700,
        entry_thesis={"entry_probability": 0.88, "entry_units_intent": 700},
    )
    fail_again = brain.decide(
        strategy_tag="MicroLevelReactor-bounce-lower",
        pocket="micro",
        side="sell",
        units=-620,
        entry_thesis={"entry_probability": 0.67, "entry_units_intent": 620},
    )

    assert fail.action == "ALLOW"
    assert recovered.reason == "recovered"
    assert fail_again.reason == "no_llm"
    assert call_count["n"] == 3


def test_brain_context_keeps_entry_and_meta_with_sl_tp(monkeypatch, tmp_path: Path) -> None:
    brain = _prepare_brain(monkeypatch, tmp_path)

    captured: dict[str, str] = {}

    def _fake_ollama(
        prompt: str,
        *,
        model: str,
        url: str,
        timeout_sec: float,
        temperature: float = 0.2,
        max_tokens: int = 256,
        keep_alive=None,
    ):
        captured["prompt"] = prompt
        return {
            "action": "ALLOW",
            "scale": 1.0,
            "reason": "clean_setup",
            "memory_update": "",
        }

    monkeypatch.setattr(brain, "call_ollama_chat_json", _fake_ollama)

    client_order_id = "test-brain-context-keep"
    decision = brain.decide(
        strategy_tag="MomentumBurst-open_long",
        pocket="micro",
        side="buy",
        units=5042,
        sl_price=158.505,
        tp_price=158.596,
        confidence=0.8424,
        client_order_id=client_order_id,
        entry_thesis={
            "confidence": 80,
            "entry_probability": 0.8424,
            "entry_units_intent": 5042,
            "tp_pips": 5.09,
            "sl_pips": 3.97,
            "dynamic_alloc": {"lot_multiplier": 0.68, "score": 0.55},
        },
        meta={
            "spread_pips": 0.8,
            "market_regime": "Trend",
            "session_label": "tokyo",
        },
    )

    assert decision.action == "ALLOW"
    prompt = captured["prompt"]
    assert '"entry_thesis": {' in prompt
    assert '"entry_probability": 0.8424' in prompt
    assert '"entry_units_intent": 5042' in prompt
    assert '"confidence": 80.0' in prompt
    assert "confidence is a score from 0 to 100." in prompt
    assert "Do not compare confidence and entry_probability as if they share the same numeric scale." in prompt
    assert '"meta": {' in prompt
    assert '"spread_pips": 0.8' in prompt

    con = sqlite3.connect(brain._DB_PATH)
    try:
        row = con.execute(
            """
            SELECT context_json
            FROM brain_decisions
            WHERE client_order_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (client_order_id,),
        ).fetchone()
    finally:
        con.close()

    assert row is not None
    context = json.loads(row[0])
    assert context["confidence"] == pytest.approx(80.0)
    assert context["entry_thesis"]["entry_probability"] == pytest.approx(0.8424)
    assert context["entry_thesis"]["entry_units_intent"] == 5042
    assert context["meta"]["spread_pips"] == pytest.approx(0.8)
    assert context["meta"]["market_regime"] == "Trend"


def test_brain_cache_fingerprint_separates_side_and_reuses_matching_context(
    monkeypatch,
    tmp_path: Path,
) -> None:
    brain = _prepare_brain(monkeypatch, tmp_path)

    calls: list[str] = []
    responses = iter(
        [
            {
                "action": "ALLOW",
                "scale": 1.0,
                "reason": "buy_ok",
                "memory_update": "",
            },
            {
                "action": "REDUCE",
                "scale": 0.6,
                "reason": "sell_reduce",
                "memory_update": "",
            },
        ]
    )

    def _fake_ollama(
        prompt: str,
        *,
        model: str,
        url: str,
        timeout_sec: float,
        temperature: float = 0.2,
        max_tokens: int = 256,
        keep_alive=None,
    ):
        calls.append(prompt)
        return next(responses)

    monkeypatch.setattr(brain, "call_ollama_chat_json", _fake_ollama)

    allow = brain.decide(
        strategy_tag="MomentumBurst-open_long",
        pocket="micro",
        side="buy",
        units=1200,
        entry_thesis={"entry_probability": 0.88, "entry_units_intent": 1200},
    )
    reduce = brain.decide(
        strategy_tag="MomentumBurst-open_long",
        pocket="micro",
        side="sell",
        units=-1200,
        entry_thesis={"entry_probability": 0.52, "entry_units_intent": 1200},
    )
    reduce_cached = brain.decide(
        strategy_tag="MomentumBurst-open_long",
        pocket="micro",
        side="sell",
        units=-1200,
        entry_thesis={"entry_probability": 0.52, "entry_units_intent": 1200},
    )

    assert allow.action == "ALLOW"
    assert reduce.action == "REDUCE"
    assert reduce_cached.action == "REDUCE"
    assert len(calls) == 2


def test_brain_context_backfills_live_tick_and_m1_factors(monkeypatch, tmp_path: Path) -> None:
    brain = _prepare_brain(monkeypatch, tmp_path)

    captured: dict[str, str] = {}

    def _fake_ollama(
        prompt: str,
        *,
        model: str,
        url: str,
        timeout_sec: float,
        temperature: float = 0.2,
        max_tokens: int = 256,
        keep_alive=None,
    ):
        captured["prompt"] = prompt
        return {
            "action": "ALLOW",
            "scale": 1.0,
            "reason": "live_context_ok",
            "memory_update": "",
        }

    class _TickWindow:
        @staticmethod
        def recent_ticks(*, seconds: int, limit: int):
            return [
                {
                    "epoch": time.time() - 0.2,
                    "bid": 158.410,
                    "ask": 158.418,
                    "mid": 158.414,
                }
            ]

    monkeypatch.setattr(brain, "call_ollama_chat_json", _fake_ollama)
    monkeypatch.setattr(brain, "tick_window", _TickWindow)
    monkeypatch.setattr(
        brain,
        "all_factors",
        lambda: {
            "M1": {
                "atr_pips": 2.6,
                "range_score": 0.33,
                "adx": 9.3,
                "bbw": 0.0006,
                "rsi": 42.7,
                "regime": "Range",
            }
        },
    )

    client_order_id = "test-brain-live-context"
    decision = brain.decide(
        strategy_tag="MicroLevelReactor-bounce-lower",
        pocket="micro",
        side="buy",
        units=500,
        client_order_id=client_order_id,
        entry_thesis={
            "entry_probability": 0.54,
            "entry_units_intent": 500,
            "sl_pips": 7.0,
            "tp_pips": 10.3,
        },
    )

    assert decision.action == "ALLOW"
    prompt = captured["prompt"]
    assert '"ticks": {' in prompt
    assert '"spread_pips": 0.8' in prompt
    assert '"atr_pips": 2.6' in prompt
    assert '"range_score": 0.33' in prompt

    con = sqlite3.connect(brain._DB_PATH)
    try:
        row = con.execute(
            """
            SELECT context_json
            FROM brain_decisions
            WHERE client_order_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (client_order_id,),
        ).fetchone()
    finally:
        con.close()

    assert row is not None
    context = json.loads(row[0])
    assert context["meta"]["spread_pips"] == pytest.approx(0.8)
    assert context["meta"]["atr_pips"] == pytest.approx(2.6)
    assert context["meta"]["market_regime"] == "Range"
    assert context["entry_thesis"]["atr_pips"] == pytest.approx(2.6)
    assert context["entry_thesis"]["range_score"] == pytest.approx(0.33)
