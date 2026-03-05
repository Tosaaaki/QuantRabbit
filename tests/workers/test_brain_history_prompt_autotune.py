from __future__ import annotations

import importlib
import json
import sqlite3
import sys
import time
from pathlib import Path


def _reload_brain_module():
    module_name = "workers.common.brain"
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def _prepare_brain(monkeypatch, tmp_path: Path):
    db_path = tmp_path / "brain_state.db"
    profile_path = tmp_path / "brain_prompt_profile.json"
    trades_path = tmp_path / "trades.db"

    monkeypatch.setenv("BRAIN_ENABLED", "1")
    monkeypatch.setenv("BRAIN_BACKEND", "ollama")
    monkeypatch.setenv("BRAIN_SAMPLE_RATE", "1.0")
    monkeypatch.setenv("BRAIN_PROMPT_AUTO_TUNE_ENABLED", "0")
    monkeypatch.setenv("BRAIN_PROMPT_PROFILE_PATH", str(profile_path))

    brain = _reload_brain_module()
    monkeypatch.setattr(brain, "_DB_PATH", db_path)
    monkeypatch.setattr(brain, "_TRADES_DB_PATH", trades_path)
    monkeypatch.setattr(brain, "log_metric", lambda *_args, **_kwargs: True)
    brain._CACHE.clear()
    brain._PROMPT_PROFILE_CACHE = (0.0, {})
    return brain, db_path, profile_path


def test_brain_logs_decision_history(monkeypatch, tmp_path: Path) -> None:
    brain, db_path, _profile_path = _prepare_brain(monkeypatch, tmp_path)

    monkeypatch.setattr(
        brain,
        "call_ollama_chat_json",
        lambda *_args, **_kwargs: {
            "action": "REDUCE",
            "scale": 0.5,
            "reason": "volatility_uncertain",
            "memory_update": "prefer tight entries",
        },
    )

    decision = brain.decide(
        strategy_tag="scalp_ping_5s_b_live",
        pocket="scalp_fast",
        side="buy",
        units=100,
        confidence=70,
        client_order_id="test-coid-1",
    )

    assert decision.action == "REDUCE"
    assert db_path.exists()

    con = sqlite3.connect(db_path)
    try:
        row = con.execute(
            """
            SELECT action, reason, client_order_id, source, llm_ok, scale
            FROM brain_decisions
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    finally:
        con.close()

    assert row is not None
    assert row[0] == "REDUCE"
    assert row[1] == "volatility_uncertain"
    assert row[2] == "test-coid-1"
    assert row[3] == "llm"
    assert int(row[4]) == 1
    assert float(row[5]) == 0.5


def test_brain_prompt_profile_is_injected(monkeypatch, tmp_path: Path) -> None:
    brain, _db_path, profile_path = _prepare_brain(monkeypatch, tmp_path)
    profile_path.write_text(
        json.dumps(
            {
                "version": "v2",
                "updated_at": "2026-03-05T00:00:00+00:00",
                "focus": "Block low-edge range chop entries.",
                "risk_bias": "Prefer REDUCE when spread expands.",
                "extra_rules": [
                    "If spread spikes, prefer REDUCE or BLOCK.",
                    "Avoid entries when confidence and momentum conflict.",
                ],
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, str] = {}

    def _fake_ollama(prompt: str, **_kwargs):
        captured["prompt"] = prompt
        return {
            "action": "ALLOW",
            "scale": 1.0,
            "reason": "clean_setup",
            "memory_update": "",
        }

    monkeypatch.setattr(brain, "call_ollama_chat_json", _fake_ollama)

    brain.decide(
        strategy_tag="scalp_ping_5s_b_live",
        pocket="scalp_fast",
        side="buy",
        units=100,
        confidence=72,
        client_order_id="test-coid-2",
    )

    prompt_text = captured.get("prompt", "")
    assert "Adaptive rules (recent live outcomes):" in prompt_text
    assert "If spread spikes, prefer REDUCE or BLOCK." in prompt_text
    assert "Focus: Block low-edge range chop entries." in prompt_text


def test_brain_auto_tune_updates_profile(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_PROMPT_AUTO_TUNE_ENABLED", "1")
    monkeypatch.setenv("BRAIN_PROMPT_AUTO_TUNE_MODEL", "gpt-oss:test")
    monkeypatch.setenv("BRAIN_PROMPT_AUTO_TUNE_URL", "http://127.0.0.1:11434/api/chat")
    brain, db_path, profile_path = _prepare_brain(monkeypatch, tmp_path)
    monkeypatch.setattr(brain, "_PROMPT_AUTOTUNE_ENABLED", True)
    monkeypatch.setattr(brain, "_PROMPT_AUTOTUNE_MIN_DECISIONS", 1)
    monkeypatch.setattr(brain, "_PROMPT_AUTOTUNE_INTERVAL_SEC", 60.0)
    monkeypatch.setattr(brain, "_PROMPT_AUTOTUNE_MODEL", "gpt-oss:test")
    monkeypatch.setattr(brain, "_PROMPT_AUTOTUNE_URL", "http://127.0.0.1:11434/api/chat")
    brain._LAST_PROMPT_AUTOTUNE_TS = 0.0

    def _fake_ollama(prompt: str, **_kwargs):
        if "You optimize an FX decision gate prompt" in prompt:
            return {
                "version": "v3",
                "focus": "Avoid low-quality continuation after spread shock.",
                "risk_bias": "Prefer BLOCK on conflicting momentum and confidence.",
                "extra_rules": [
                    "Block when spread is elevated and confidence < 0.6.",
                    "Reduce size when momentum is weak after volatility spike.",
                    "Block entries when regime and direction disagree.",
                ],
            }
        return {
            "action": "ALLOW",
            "scale": 1.0,
            "reason": "baseline_allow",
            "memory_update": "",
        }

    monkeypatch.setattr(brain, "call_ollama_chat_json", _fake_ollama)

    brain.decide(
        strategy_tag="scalp_ping_5s_b_live",
        pocket="scalp_fast",
        side="buy",
        units=100,
        confidence=68,
        client_order_id="test-coid-3",
    )

    for _ in range(30):
        if profile_path.exists():
            break
        time.sleep(0.05)
    assert profile_path.exists()
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile.get("version") == "v3"
    assert len(profile.get("extra_rules") or []) >= 3

    con = sqlite3.connect(db_path)
    try:
        row = con.execute(
            """
            SELECT applied, profile_version
            FROM brain_prompt_runs
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    finally:
        con.close()
    assert row is not None
    assert int(row[0]) == 1
    assert row[1] == "v3"


def test_collect_autotune_summary_includes_trade_outcomes(monkeypatch, tmp_path: Path) -> None:
    brain, _db_path, _profile_path = _prepare_brain(monkeypatch, tmp_path)

    monkeypatch.setattr(
        brain,
        "call_ollama_chat_json",
        lambda *_args, **_kwargs: {
            "action": "ALLOW",
            "scale": 1.0,
            "reason": "clean_setup",
            "memory_update": "",
        },
    )

    client_order_id = "summary-join-coid-1"
    brain.decide(
        strategy_tag="scalp_ping_5s_b_live",
        pocket="scalp_fast",
        side="buy",
        units=100,
        confidence=75,
        client_order_id=client_order_id,
    )

    con = sqlite3.connect(brain._TRADES_DB_PATH)
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_order_id TEXT,
                realized_pl REAL,
                pl_pips REAL,
                close_time TEXT
            )
            """
        )
        con.execute(
            """
            INSERT INTO trades(client_order_id, realized_pl, pl_pips, close_time)
            VALUES(?,?,?,?)
            """,
            (client_order_id, 12.5, 3.2, "2026-03-05T00:00:00+00:00"),
        )
        con.commit()
    finally:
        con.close()

    summary = brain._collect_autotune_summary(24.0)
    allow = summary.get("filled_trade_outcome", {}).get("ALLOW", {})
    assert allow.get("trades") == 1
    assert allow.get("wins") == 1
    assert allow.get("win_rate") == 1.0
