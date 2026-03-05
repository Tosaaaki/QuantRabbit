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
    runtime_profile_path = tmp_path / "brain_runtime_param_profile.json"
    trades_path = tmp_path / "trades.db"
    report_latest_path = tmp_path / "brain_prompt_autotune_latest.json"
    report_history_path = tmp_path / "brain_prompt_autotune_history.jsonl"
    runtime_report_latest_path = tmp_path / "brain_runtime_param_autotune_latest.json"
    runtime_report_history_path = tmp_path / "brain_runtime_param_autotune_history.jsonl"

    monkeypatch.setenv("BRAIN_ENABLED", "1")
    monkeypatch.setenv("BRAIN_BACKEND", "ollama")
    monkeypatch.setenv("BRAIN_SAMPLE_RATE", "1.0")
    monkeypatch.setenv("BRAIN_PROMPT_AUTO_TUNE_ENABLED", "0")
    monkeypatch.setenv("BRAIN_PROMPT_PROFILE_PATH", str(profile_path))
    monkeypatch.setenv("BRAIN_PROMPT_REPORT_LATEST_PATH", str(report_latest_path))
    monkeypatch.setenv("BRAIN_PROMPT_REPORT_HISTORY_PATH", str(report_history_path))
    monkeypatch.setenv("BRAIN_RUNTIME_PARAM_PROFILE_PATH", str(runtime_profile_path))
    monkeypatch.setenv("BRAIN_RUNTIME_PARAM_AUTO_TUNE_ENABLED", "0")
    monkeypatch.setenv("BRAIN_RUNTIME_PARAM_REPORT_LATEST_PATH", str(runtime_report_latest_path))
    monkeypatch.setenv("BRAIN_RUNTIME_PARAM_REPORT_HISTORY_PATH", str(runtime_report_history_path))

    brain = _reload_brain_module()
    monkeypatch.setattr(brain, "_DB_PATH", db_path)
    monkeypatch.setattr(brain, "_TRADES_DB_PATH", trades_path)
    monkeypatch.setattr(brain, "log_metric", lambda *_args, **_kwargs: True)
    brain._CACHE.clear()
    brain._PROMPT_PROFILE_CACHE = (0.0, {})
    brain._RUNTIME_PARAM_PROFILE_CACHE = (0.0, {})
    brain._LAST_RUNTIME_PARAM_AUTOTUNE_TS = 0.0
    return brain, db_path, profile_path, report_latest_path, report_history_path


def test_brain_logs_decision_history(monkeypatch, tmp_path: Path) -> None:
    brain, db_path, _profile_path, _report_latest, _report_history = _prepare_brain(monkeypatch, tmp_path)

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
    brain, _db_path, profile_path, _report_latest, _report_history = _prepare_brain(monkeypatch, tmp_path)
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
    brain, db_path, profile_path, _report_latest, _report_history = _prepare_brain(monkeypatch, tmp_path)
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
    brain, _db_path, _profile_path, _report_latest, _report_history = _prepare_brain(monkeypatch, tmp_path)

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


def _insert_trade_row(trades_db_path: Path, client_order_id: str, realized_pl: float, pl_pips: float) -> None:
    con = sqlite3.connect(trades_db_path)
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
            (client_order_id, realized_pl, pl_pips, "2026-03-05T00:00:00+00:00"),
        )
        con.commit()
    finally:
        con.close()


def _insert_brain_decision_row(
    brain,
    *,
    strategy_tag: str,
    pocket: str,
    action: str,
    reason: str = "seed",
) -> None:
    scale = 0.0 if action == "BLOCK" else (0.5 if action == "REDUCE" else 1.0)
    decision = brain.BrainDecision(
        allowed=action != "BLOCK",
        scale=scale,
        reason=reason,
        action=action,
        memory=None,
    )
    brain._log_decision_row(
        strategy_tag=strategy_tag,
        pocket=pocket,
        side="buy",
        units=100,
        sl_price=None,
        tp_price=None,
        confidence=70,
        client_order_id=f"seed-{action}-{time.time_ns()}",
        backend="ollama",
        source="seed",
        llm_ok=True,
        latency_ms=0.0,
        decision=decision,
        memory_before=None,
        memory_after=None,
        profile_version="v1",
        context={"entry_thesis": {"spread_pips": 0.8, "atr_pips": 2.1}, "meta": {}},
        payload={"seed": True},
        error=None,
    )


def test_runtime_profile_min_scale_is_applied(monkeypatch, tmp_path: Path) -> None:
    brain, _db_path, _profile_path, _report_latest, _report_history = _prepare_brain(monkeypatch, tmp_path)
    runtime_profile_path = tmp_path / "brain_runtime_param_profile.json"
    runtime_profile_path.write_text(
        json.dumps(
            {
                "version": "rp-v2",
                "min_scale": 0.44,
                "block_rate_soft_limit": 0.7,
                "activity_rate_floor": 0.2,
                "block_to_reduce_scale": 0.5,
                "guard_window_decisions": 60,
                "min_guard_samples": 20,
                "max_block_streak": 5,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        brain,
        "call_ollama_chat_json",
        lambda *_args, **_kwargs: {
            "action": "REDUCE",
            "scale": 0.12,
            "reason": "weak_edge",
            "memory_update": "",
        },
    )

    decision = brain.decide(
        strategy_tag="scalp_ping_5s_b_live",
        pocket="scalp_fast",
        side="buy",
        units=100,
        confidence=65,
        client_order_id="runtime-min-scale-1",
    )
    assert decision.action == "REDUCE"
    assert decision.allowed is True
    assert decision.scale == 0.44


def test_runtime_guard_biases_block_to_reduce_on_overblocking(monkeypatch, tmp_path: Path) -> None:
    brain, _db_path, _profile_path, _report_latest, _report_history = _prepare_brain(monkeypatch, tmp_path)
    runtime_profile_path = tmp_path / "brain_runtime_param_profile.json"
    runtime_profile_path.write_text(
        json.dumps(
            {
                "version": "rp-v3",
                "min_scale": 0.25,
                "block_rate_soft_limit": 0.45,
                "activity_rate_floor": 0.4,
                "block_to_reduce_scale": 0.37,
                "guard_window_decisions": 20,
                "min_guard_samples": 5,
                "max_block_streak": 3,
            }
        ),
        encoding="utf-8",
    )

    for _ in range(10):
        _insert_brain_decision_row(
            brain,
            strategy_tag="scalp_ping_5s_b_live",
            pocket="scalp_fast",
            action="BLOCK",
            reason="seed_block",
        )

    monkeypatch.setattr(
        brain,
        "call_ollama_chat_json",
        lambda *_args, **_kwargs: {
            "action": "BLOCK",
            "scale": 0.0,
            "reason": "uncertain_setup",
            "memory_update": "",
        },
    )

    decision = brain.decide(
        strategy_tag="scalp_ping_5s_b_live",
        pocket="scalp_fast",
        side="buy",
        units=100,
        confidence=63,
        client_order_id="runtime-bias-1",
    )
    assert decision.action == "REDUCE"
    assert decision.allowed is True
    assert decision.scale >= 0.37
    assert "no_trade_bias_reduce" in decision.reason


def test_runtime_autotune_updates_runtime_profile(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_RUNTIME_PARAM_AUTO_TUNE_ENABLED", "1")
    monkeypatch.setenv("BRAIN_RUNTIME_PARAM_AUTO_TUNE_MODEL", "gpt-oss:test")
    monkeypatch.setenv("BRAIN_RUNTIME_PARAM_AUTO_TUNE_URL", "http://127.0.0.1:11434/api/chat")
    brain, db_path, _profile_path, _report_latest, _report_history = _prepare_brain(monkeypatch, tmp_path)
    runtime_profile_path = tmp_path / "brain_runtime_param_profile.json"
    runtime_report_latest_path = tmp_path / "brain_runtime_param_autotune_latest.json"
    monkeypatch.setattr(brain, "_RUNTIME_PARAM_AUTOTUNE_ENABLED", True)
    monkeypatch.setattr(brain, "_RUNTIME_PARAM_AUTOTUNE_MIN_DECISIONS", 1)
    monkeypatch.setattr(brain, "_RUNTIME_PARAM_AUTOTUNE_INTERVAL_SEC", 60.0)
    monkeypatch.setattr(brain, "_RUNTIME_PARAM_AUTOTUNE_MODEL", "gpt-oss:test")
    monkeypatch.setattr(brain, "_RUNTIME_PARAM_AUTOTUNE_URL", "http://127.0.0.1:11434/api/chat")
    monkeypatch.setattr(brain, "_maybe_autotune_runtime_param_profile_async", lambda: None)
    brain._LAST_RUNTIME_PARAM_AUTOTUNE_TS = 0.0

    def _fake_ollama(prompt: str, **_kwargs):
        if "You optimize runtime risk parameters for an FX decision gate" in prompt:
            return {
                "version": "rp-v4",
                "min_scale": 0.31,
                "block_rate_soft_limit": 0.56,
                "activity_rate_floor": 0.38,
                "block_to_reduce_scale": 0.41,
                "guard_window_decisions": 140,
                "min_guard_samples": 28,
                "max_block_streak": 4,
                "notes": ["push block decisions to reduce when activity drops"],
            }
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
        confidence=71,
        client_order_id="runtime-auto-1",
        entry_thesis={"spread_pips": 0.8, "atr_pips": 2.2},
    )

    brain._LAST_RUNTIME_PARAM_AUTOTUNE_TS = 0.0
    brain._maybe_autotune_runtime_param_profile()

    payload = json.loads(runtime_profile_path.read_text(encoding="utf-8"))
    assert payload["version"] == "rp-v4"
    assert payload["min_scale"] == 0.31
    assert payload["block_to_reduce_scale"] == 0.41

    con = sqlite3.connect(db_path)
    try:
        row = con.execute(
            """
            SELECT applied, profile_version
            FROM brain_runtime_param_runs
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    finally:
        con.close()
    assert row is not None
    assert int(row[0]) == 1
    assert row[1] == "rp-v4"
    assert runtime_report_latest_path.exists()


def test_collect_runtime_autotune_summary_includes_market_metrics(monkeypatch, tmp_path: Path) -> None:
    brain, _db_path, _profile_path, _report_latest, _report_history = _prepare_brain(monkeypatch, tmp_path)

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

    brain.decide(
        strategy_tag="scalp_ping_5s_b_live",
        pocket="scalp_fast",
        side="buy",
        units=100,
        confidence=72,
        client_order_id="runtime-summary-1",
        entry_thesis={
            "spread_pips": 0.8,
            "factors": {"M1": {"atr_pips": 2.25}},
        },
    )

    summary = brain._collect_runtime_autotune_summary(24.0)
    market = summary.get("market_summary", {})
    assert market["spread_pips"]["count"] >= 1
    assert market["atr_pips"]["count"] >= 1
    assert market["confidence"]["count"] >= 1


def test_prompt_report_writes_before_after_comparison(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_PROMPT_AUTO_TUNE_ENABLED", "1")
    monkeypatch.setenv("BRAIN_PROMPT_AUTO_TUNE_MODEL", "gpt-oss:test")
    monkeypatch.setenv("BRAIN_PROMPT_AUTO_TUNE_URL", "http://127.0.0.1:11434/api/chat")
    brain, _db_path, _profile_path, report_latest_path, report_history_path = _prepare_brain(monkeypatch, tmp_path)
    monkeypatch.setattr(brain, "_PROMPT_AUTOTUNE_ENABLED", True)
    monkeypatch.setattr(brain, "_PROMPT_AUTOTUNE_MIN_DECISIONS", 1)
    monkeypatch.setattr(brain, "_PROMPT_AUTOTUNE_INTERVAL_SEC", 60.0)
    monkeypatch.setattr(brain, "_PROMPT_AUTOTUNE_MODEL", "gpt-oss:test")
    monkeypatch.setattr(brain, "_PROMPT_AUTOTUNE_URL", "http://127.0.0.1:11434/api/chat")
    monkeypatch.setattr(brain, "_maybe_autotune_prompt_profile_async", lambda: None)
    brain._LAST_PROMPT_AUTOTUNE_TS = 0.0

    tune_count = {"n": 0}

    def _fake_ollama(prompt: str, **_kwargs):
        if "You optimize an FX decision gate prompt" in prompt:
            tune_count["n"] += 1
            return {
                "version": f"v{2 + tune_count['n']}",
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
            "reason": "clean_setup",
            "memory_update": "",
        }

    monkeypatch.setattr(brain, "call_ollama_chat_json", _fake_ollama)

    # first run
    brain.decide(
        strategy_tag="scalp_ping_5s_b_live",
        pocket="scalp_fast",
        side="buy",
        units=100,
        confidence=70,
        client_order_id="report-coid-1",
    )
    _insert_trade_row(brain._TRADES_DB_PATH, "report-coid-1", 12.5, 3.2)
    brain._LAST_PROMPT_AUTOTUNE_TS = 0.0
    brain._maybe_autotune_prompt_profile()

    assert report_latest_path.exists()
    first_report = json.loads(report_latest_path.read_text(encoding="utf-8"))
    assert first_report["profile_version_before"] is None
    assert first_report["profile_version_after"] == "v3"

    # second run
    brain.decide(
        strategy_tag="scalp_ping_5s_b_live",
        pocket="scalp_fast",
        side="buy",
        units=100,
        confidence=68,
        client_order_id="report-coid-2",
    )
    _insert_trade_row(brain._TRADES_DB_PATH, "report-coid-2", -6.0, -1.8)
    brain._LAST_PROMPT_AUTOTUNE_TS = 0.0
    brain._maybe_autotune_prompt_profile()

    second_report = json.loads(report_latest_path.read_text(encoding="utf-8"))
    assert second_report["profile_version_before"] == "v3"
    assert second_report["profile_version_after"] == "v4"
    combined_delta = second_report["comparison"]["COMBINED"]["delta"]
    assert combined_delta["win_rate"] is not None
    assert combined_delta["profit_factor"] is not None

    history_lines = [line for line in report_history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(history_lines) >= 2
    last_history = json.loads(history_lines[-1])
    assert last_history["profile_version_after"] == "v4"
