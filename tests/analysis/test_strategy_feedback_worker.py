from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis import strategy_feedback_worker as worker


def _seed_trades(db_path: Path, *, strategy_tag: str, count: int = 12) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE trades (
            strategy_tag TEXT,
            strategy TEXT,
            pl_pips REAL,
            open_time TEXT,
            close_time TEXT
        )
        """
    )
    now = dt.datetime.utcnow()
    for idx in range(count):
        close_time = (now - dt.timedelta(minutes=idx)).strftime("%Y-%m-%d %H:%M:%S")
        open_time = (now - dt.timedelta(minutes=idx, seconds=45)).strftime("%Y-%m-%d %H:%M:%S")
        pl_pips = 1.2 if idx < 9 else -0.4
        conn.execute(
            "INSERT INTO trades(strategy_tag, strategy, pl_pips, open_time, close_time) VALUES (?, ?, ?, ?, ?)",
            (strategy_tag, strategy_tag, pl_pips, open_time, close_time),
        )
    conn.commit()
    conn.close()


def _seed_trades_with_setup_context(
    db_path: Path,
    *,
    strategy_tag: str,
    count: int,
    setup_fingerprint: str,
    flow_regime: str,
    microstructure_bucket: str,
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE trades (
            strategy_tag TEXT,
            strategy TEXT,
            pl_pips REAL,
            open_time TEXT,
            close_time TEXT,
            entry_thesis TEXT
        )
        """
    )
    now = dt.datetime.utcnow()
    thesis = json.dumps(
        {
            "setup_fingerprint": setup_fingerprint,
            "flow_regime": flow_regime,
            "microstructure_bucket": microstructure_bucket,
        },
        ensure_ascii=True,
    )
    for idx in range(count):
        close_time = (now - dt.timedelta(minutes=idx)).strftime("%Y-%m-%d %H:%M:%S")
        open_time = (now - dt.timedelta(minutes=idx, seconds=40)).strftime("%Y-%m-%d %H:%M:%S")
        pl_pips = -0.9 if idx < max(3, count - 2) else 0.4
        conn.execute(
            """
            INSERT INTO trades(strategy_tag, strategy, pl_pips, open_time, close_time, entry_thesis)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (strategy_tag, strategy_tag, pl_pips, open_time, close_time, thesis),
        )
    conn.commit()
    conn.close()


def _seed_trades_with_derived_setup_context(
    db_path: Path,
    *,
    strategy_tag: str,
    count: int,
    units: int,
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE trades (
            strategy_tag TEXT,
            strategy TEXT,
            units INTEGER,
            pl_pips REAL,
            open_time TEXT,
            close_time TEXT,
            entry_thesis TEXT
        )
        """
    )
    now = dt.datetime.utcnow()
    thesis = json.dumps(
        {
            "strategy_tag": strategy_tag,
            "range_mode": "trend",
            "range_score": 0.18,
            "range_reason": "volatility_compression",
            "spread_pips": 0.8,
            "technical_context": {
                "ticks": {"spread_pips": 0.8, "tick_rate": 9.1},
                "indicators": {
                    "M1": {
                        "atr_pips": 2.4,
                        "rsi": 67.0,
                        "adx": 29.0,
                        "plus_di": 31.0,
                        "minus_di": 14.0,
                        "ma10": 158.110,
                        "ma20": 158.080,
                    }
                },
            },
        },
        ensure_ascii=True,
    )
    for idx in range(count):
        close_time = (now - dt.timedelta(minutes=idx)).strftime("%Y-%m-%d %H:%M:%S")
        open_time = (now - dt.timedelta(minutes=idx, seconds=40)).strftime("%Y-%m-%d %H:%M:%S")
        pl_pips = -0.9 if idx < max(3, count - 2) else 0.4
        conn.execute(
            """
            INSERT INTO trades(strategy_tag, strategy, units, pl_pips, open_time, close_time, entry_thesis)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (strategy_tag, strategy_tag, units, pl_pips, open_time, close_time, thesis),
        )
    conn.commit()
    conn.close()


def test_build_payload_discovers_local_v2_services(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path
    systemd_dir = repo / "systemd"
    env_dir = repo / "ops" / "env"
    log_dir = repo / "logs"
    pid_dir = log_dir / "local_v2_stack" / "pids"
    trades_db = log_dir / "trades.db"

    systemd_dir.mkdir(parents=True)
    env_dir.mkdir(parents=True)
    pid_dir.mkdir(parents=True)

    (env_dir / "quant-v2-runtime.env").write_text("", encoding="utf-8")
    (env_dir / "quant-scalp-ping-5s-b.env").write_text(
        "SCALP_PING_5S_B_MODE=scalp_ping_5s_b_live\n",
        encoding="utf-8",
    )
    (systemd_dir / "quant-scalp-ping-5s-b.service").write_text(
        "\n".join(
            [
                "[Service]",
                "EnvironmentFile=-/home/tossaki/QuantRabbit/ops/env/quant-v2-runtime.env",
                "EnvironmentFile=-/home/tossaki/QuantRabbit/ops/env/quant-scalp-ping-5s-b.env",
                "ExecStart=/home/tossaki/QuantRabbit/.venv/bin/python -m workers.scalp_ping_5s_b.worker",
            ]
        ),
        encoding="utf-8",
    )
    (pid_dir / "quant-scalp-ping-5s-b.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    _seed_trades(trades_db, strategy_tag="scalp_ping_5s_b_live")

    monkeypatch.setattr(worker, "BASE_DIR", repo)
    monkeypatch.setattr(worker, "_systemctl_available", lambda: False)
    monkeypatch.setattr(worker, "_systemctl_running_services", lambda: set())
    monkeypatch.setattr(worker, "_discover_from_control", lambda: {})
    monkeypatch.setenv("STRATEGY_FEEDBACK_TRADES_DB", str(trades_db))
    monkeypatch.setenv("STRATEGY_FEEDBACK_SYSTEMD_DIR", str(systemd_dir))
    monkeypatch.setenv("STRATEGY_FEEDBACK_LOCAL_PID_DIR", str(pid_dir))
    monkeypatch.setenv("STRATEGY_FEEDBACK_PATH", str(log_dir / "strategy_feedback.json"))

    payload = worker._build_payload(worker.WorkerConfig())

    strategies = payload["strategies"]
    assert "scalp_ping_5s_b_live" in strategies
    advice = strategies["scalp_ping_5s_b_live"]
    assert advice["strategy_params"]["configured_params"]["SCALP_PING_5S_B_MODE"] == "scalp_ping_5s_b_live"
    assert advice["entry_probability_multiplier"] > 1.0


def test_build_payload_remaps_directional_trade_tags_to_discovered_base_strategy(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path
    systemd_dir = repo / "systemd"
    env_dir = repo / "ops" / "env"
    log_dir = repo / "logs"
    pid_dir = log_dir / "local_v2_stack" / "pids"
    trades_db = log_dir / "trades.db"

    systemd_dir.mkdir(parents=True)
    env_dir.mkdir(parents=True)
    pid_dir.mkdir(parents=True)

    (env_dir / "quant-v2-runtime.env").write_text("", encoding="utf-8")
    (env_dir / "quant-micro-trendretest.env").write_text(
        "\n".join(
            [
                "MICRO_STRATEGY_ALLOWLIST=MicroTrendRetest",
                "MICRO_MULTI_LOG_PREFIX=[MicroTrendRetest]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (systemd_dir / "quant-micro-trendretest.service").write_text(
        "\n".join(
            [
                "[Service]",
                "EnvironmentFile=-/home/tossaki/QuantRabbit/ops/env/quant-v2-runtime.env",
                "EnvironmentFile=-/home/tossaki/QuantRabbit/ops/env/quant-micro-trendretest.env",
                "ExecStart=/home/tossaki/QuantRabbit/.venv/bin/python -m workers.micro_trendretest.worker",
            ]
        ),
        encoding="utf-8",
    )
    (pid_dir / "quant-micro-trendretest.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    _seed_trades(trades_db, strategy_tag="MicroTrendRetest-long", count=14)

    monkeypatch.setattr(worker, "BASE_DIR", repo)
    monkeypatch.setattr(worker, "_systemctl_available", lambda: False)
    monkeypatch.setattr(worker, "_systemctl_running_services", lambda: set())
    monkeypatch.setattr(worker, "_discover_from_control", lambda: {})
    monkeypatch.setenv("STRATEGY_FEEDBACK_TRADES_DB", str(trades_db))
    monkeypatch.setenv("STRATEGY_FEEDBACK_SYSTEMD_DIR", str(systemd_dir))
    monkeypatch.setenv("STRATEGY_FEEDBACK_LOCAL_PID_DIR", str(pid_dir))
    monkeypatch.setenv("STRATEGY_FEEDBACK_PATH", str(log_dir / "strategy_feedback.json"))

    payload = worker._build_payload(worker.WorkerConfig())

    strategies = payload["strategies"]
    assert "MicroTrendRetest" in strategies
    advice = strategies["MicroTrendRetest"]
    assert advice["strategy_params"]["trades"] == 14
    assert advice["entry_probability_multiplier"] > 1.0


def test_norm_tag_resolves_strategy_aliases() -> None:
    assert worker._norm_tag("m1scalper_m1") == "M1Scalper-M1"
    assert worker._norm_tag("microlevelreactor") == "MicroLevelReactor"


def test_squad_recommendation_keeps_metadata_for_neutral_strategy() -> None:
    stats = worker.StrategyStats(
        tag="MicroLevelReactor",
        trades=12,
        wins=6,
        losses=6,
        sum_pips=0.6,
        avg_pips=0.05,
        avg_abs_pips=0.8,
        gross_win=4.8,
        gross_loss=4.2,
        avg_hold_sec=120.0,
        last_closed="2026-03-10 00:00:00",
    )

    advice = worker._squad_recommendation("MicroLevelReactor", stats, 12, strategy_params={"FOO": "bar"})

    assert advice["strategy_params"]["analysis_squad"] == "micro"
    assert advice["strategy_params"]["configured_params"]["FOO"] == "bar"
    assert "entry_probability_multiplier" not in advice


def test_squad_recommendation_handles_zero_win_counts() -> None:
    stats = worker.StrategyStats(
        tag="scalp_extrema_reversal_live",
        trades=12,
        wins=0,
        losses=12,
        sum_pips=-4.8,
        avg_pips=-0.4,
        avg_abs_pips=0.8,
        gross_win=0.0,
        gross_loss=4.8,
        avg_hold_sec=45.0,
        last_closed="2026-03-10 00:00:00",
    )

    advice = worker._squad_recommendation("scalp_extrema_reversal_live", stats, 12)

    assert advice["strategy_params"]["analysis_squad"] == "scalp"
    assert advice["entry_probability_multiplier"] < 1.0


def test_squad_recommendation_blocks_positive_adjustment_without_payoff_edge() -> None:
    stats = worker.StrategyStats(
        tag="M1Scalper-M1",
        trades=12,
        wins=8,
        losses=4,
        sum_pips=0.6,
        avg_pips=0.05,
        avg_abs_pips=0.62,
        gross_win=4.0,
        gross_loss=3.4,
        avg_hold_sec=75.0,
        last_closed="2026-03-10 00:00:00",
    )

    advice = worker._squad_recommendation("M1Scalper-M1", stats, 12)

    assert "entry_probability_multiplier" not in advice
    assert "entry_units_multiplier" not in advice
    assert advice["strategy_params"]["feedback_growth_gate"]["payoff_ok"] is False
    assert advice["strategy_params"]["feedback_growth_gate"]["allow_positive_adjustment"] is False


def test_squad_recommendation_requires_improvement_for_positive_adjustment() -> None:
    stats = worker.StrategyStats(
        tag="MomentumBurst",
        trades=12,
        wins=8,
        losses=4,
        sum_pips=2.2,
        avg_pips=0.1833,
        avg_abs_pips=0.6167,
        gross_win=4.8,
        gross_loss=2.6,
        avg_hold_sec=120.0,
        last_closed="2026-03-10 00:00:00",
    )

    advice = worker._squad_recommendation(
        "MomentumBurst",
        stats,
        12,
        previous_feedback={
            "strategy_params": {
                "profit_factor": 1.95,
                "avg_pips": 0.25,
                "loss_asymmetry": 0.9,
            }
        },
    )

    assert "entry_probability_multiplier" not in advice
    assert "entry_units_multiplier" not in advice
    assert advice["strategy_params"]["feedback_growth_gate"]["payoff_ok"] is True
    assert advice["strategy_params"]["feedback_growth_gate"]["improved_vs_prev"] is False
    assert advice["strategy_params"]["feedback_growth_gate"]["allow_positive_adjustment"] is False


def test_squad_recommendation_allows_positive_adjustment_for_improving_profitable_setup() -> None:
    stats = worker.StrategyStats(
        tag="MomentumBurst",
        trades=12,
        wins=8,
        losses=4,
        sum_pips=2.2,
        avg_pips=0.1833,
        avg_abs_pips=0.6167,
        gross_win=4.8,
        gross_loss=2.6,
        avg_hold_sec=120.0,
        last_closed="2026-03-10 00:00:00",
    )

    advice = worker._squad_recommendation(
        "MomentumBurst",
        stats,
        12,
        previous_feedback={
            "strategy_params": {
                "profit_factor": 1.25,
                "avg_pips": 0.07,
                "loss_asymmetry": 1.25,
            }
        },
    )

    assert advice["entry_probability_multiplier"] > 1.0
    assert advice["entry_units_multiplier"] > 1.0
    assert advice["strategy_params"]["feedback_growth_gate"]["improved_vs_prev"] is True
    assert advice["strategy_params"]["feedback_growth_gate"]["allow_positive_adjustment"] is True


def test_build_payload_keeps_boosted_low_sample_lane_in_feedback(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path
    log_dir = repo / "logs"
    trades_db = log_dir / "trades.db"
    participation_path = repo / "config" / "participation_alloc.json"

    _seed_trades(trades_db, strategy_tag="PrecisionLowVol", count=4)
    participation_path.parent.mkdir(parents=True, exist_ok=True)
    participation_path.write_text(
        json.dumps(
            {
                "as_of": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                "strategies": {
                    "PrecisionLowVol": {
                        "action": "boost_participation",
                        "lot_multiplier": 1.03,
                        "probability_boost": 0.01,
                        "cadence_floor": 1.07,
                        "attempts": 4,
                        "fills": 4,
                    }
                },
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(worker, "BASE_DIR", repo)
    monkeypatch.setattr(worker, "_systemctl_available", lambda: False)
    monkeypatch.setattr(worker, "_systemctl_running_services", lambda: set())
    monkeypatch.setattr(worker, "_local_stack_running_services", lambda _pid_dir: set())
    monkeypatch.setattr(
        worker,
        "_discover_from_control",
        lambda: {
            "PrecisionLowVol": worker.StrategyRecord(
                canonical_tag="PrecisionLowVol",
                active=True,
                entry_active=True,
                exit_active=False,
            )
        },
    )
    monkeypatch.setattr(worker, "_discover_from_systemd", lambda *_args, **_kwargs: {})
    monkeypatch.setenv("STRATEGY_FEEDBACK_TRADES_DB", str(trades_db))
    monkeypatch.setenv("STRATEGY_FEEDBACK_PATH", str(log_dir / "strategy_feedback.json"))
    monkeypatch.setenv("STRATEGY_FEEDBACK_SYSTEMD_DIR", str(repo / "systemd"))
    monkeypatch.setenv("STRATEGY_FEEDBACK_LOCAL_PID_DIR", str(repo / "logs" / "local_v2_stack" / "pids"))
    monkeypatch.setenv("STRATEGY_FEEDBACK_PARTICIPATION_PATH", str(participation_path))

    payload = worker._build_payload(worker.WorkerConfig())

    advice = payload["strategies"]["PrecisionLowVol"]
    assert "entry_probability_multiplier" not in advice
    assert advice["strategy_params"]["feedback_probe"]["source"] == "participation_alloc"
    assert advice["strategy_params"]["feedback_probe"]["mode"] == "low_sample_safe"
    assert advice["strategy_params"]["feedback_probe"]["lot_multiplier"] == 1.03


def test_build_payload_remaps_directional_boost_probe_to_canonical_strategy(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path
    log_dir = repo / "logs"
    trades_db = log_dir / "trades.db"
    participation_path = repo / "config" / "participation_alloc.json"

    _seed_trades(trades_db, strategy_tag="MomentumBurst", count=4)
    participation_path.parent.mkdir(parents=True, exist_ok=True)
    participation_path.write_text(
        json.dumps(
            {
                "as_of": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                "strategies": {
                    "MomentumBurst-open_long": {
                        "action": "boost_participation",
                        "lot_multiplier": 1.2064,
                        "probability_boost": 0.07,
                        "cadence_floor": 1.2,
                        "attempts": 2,
                        "fills": 2,
                    }
                },
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(worker, "BASE_DIR", repo)
    monkeypatch.setattr(worker, "_systemctl_available", lambda: False)
    monkeypatch.setattr(worker, "_systemctl_running_services", lambda: set())
    monkeypatch.setattr(worker, "_local_stack_running_services", lambda _pid_dir: set())
    monkeypatch.setattr(
        worker,
        "_discover_from_control",
        lambda: {
            "MomentumBurst": worker.StrategyRecord(
                canonical_tag="MomentumBurst",
                active=True,
                entry_active=True,
                exit_active=False,
            )
        },
    )
    monkeypatch.setattr(worker, "_discover_from_systemd", lambda *_args, **_kwargs: {})
    monkeypatch.setenv("STRATEGY_FEEDBACK_TRADES_DB", str(trades_db))
    monkeypatch.setenv("STRATEGY_FEEDBACK_PATH", str(log_dir / "strategy_feedback.json"))
    monkeypatch.setenv("STRATEGY_FEEDBACK_SYSTEMD_DIR", str(repo / "systemd"))
    monkeypatch.setenv("STRATEGY_FEEDBACK_LOCAL_PID_DIR", str(repo / "logs" / "local_v2_stack" / "pids"))
    monkeypatch.setenv("STRATEGY_FEEDBACK_PARTICIPATION_PATH", str(participation_path))

    payload = worker._build_payload(worker.WorkerConfig())

    assert "MomentumBurst" in payload["strategies"]
    assert "MomentumBurst-open_long" not in payload["strategies"]
    advice = payload["strategies"]["MomentumBurst"]
    assert advice["strategy_params"]["feedback_probe"]["source"] == "participation_alloc"
    assert advice["strategy_params"]["feedback_probe"]["lot_multiplier"] == 1.2064


def test_build_payload_emits_setup_overrides_from_recent_entry_thesis(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path
    log_dir = repo / "logs"
    trades_db = log_dir / "trades.db"

    _seed_trades_with_setup_context(
        trades_db,
        strategy_tag="RangeFader-sell-fade",
        count=8,
        setup_fingerprint="RangeFader|short|sell-fade|trend_long|p2",
        flow_regime="trend_long",
        microstructure_bucket="tight_fast",
    )

    monkeypatch.setattr(worker, "BASE_DIR", repo)
    monkeypatch.setattr(worker, "_systemctl_available", lambda: False)
    monkeypatch.setattr(worker, "_systemctl_running_services", lambda: set())
    monkeypatch.setattr(worker, "_local_stack_running_services", lambda _pid_dir: set())
    monkeypatch.setattr(
        worker,
        "_discover_from_control",
        lambda: {
            "RangeFader": worker.StrategyRecord(
                canonical_tag="RangeFader",
                active=True,
                entry_active=True,
                exit_active=False,
            )
        },
    )
    monkeypatch.setattr(worker, "_discover_from_systemd", lambda *_args, **_kwargs: {})
    monkeypatch.setenv("STRATEGY_FEEDBACK_TRADES_DB", str(trades_db))
    monkeypatch.setenv("STRATEGY_FEEDBACK_PATH", str(log_dir / "strategy_feedback.json"))
    monkeypatch.setenv("STRATEGY_FEEDBACK_SYSTEMD_DIR", str(repo / "systemd"))
    monkeypatch.setenv("STRATEGY_FEEDBACK_LOCAL_PID_DIR", str(repo / "logs" / "local_v2_stack" / "pids"))
    monkeypatch.setenv("STRATEGY_FEEDBACK_MIN_TRADES", "6")

    payload = worker._build_payload(worker.WorkerConfig())

    advice = payload["strategies"]["RangeFader"]
    assert advice["strategy_params"]["trades"] == 8
    assert isinstance(advice.get("setup_overrides"), list)
    exact = next(item for item in advice["setup_overrides"] if item["match_dimension"] == "setup_fingerprint")
    assert exact["setup_fingerprint"] == "RangeFader|short|sell-fade|trend_long|p2"
    assert exact["flow_regime"] == "trend_long"
    assert exact["microstructure_bucket"] == "tight_fast"
    assert exact["trades"] == 8
    assert exact["entry_probability_multiplier"] < 1.0


def test_build_payload_derives_setup_overrides_from_technical_context(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path
    log_dir = repo / "logs"
    trades_db = log_dir / "trades.db"

    _seed_trades_with_derived_setup_context(
        trades_db,
        strategy_tag="RangeFader-sell-fade",
        count=8,
        units=-120,
    )

    monkeypatch.setattr(worker, "BASE_DIR", repo)
    monkeypatch.setattr(worker, "_systemctl_available", lambda: False)
    monkeypatch.setattr(worker, "_systemctl_running_services", lambda: set())
    monkeypatch.setattr(worker, "_local_stack_running_services", lambda _pid_dir: set())
    monkeypatch.setattr(
        worker,
        "_discover_from_control",
        lambda: {
            "RangeFader": worker.StrategyRecord(
                canonical_tag="RangeFader",
                active=True,
                entry_active=True,
                exit_active=False,
            )
        },
    )
    monkeypatch.setattr(worker, "_discover_from_systemd", lambda *_args, **_kwargs: {})
    monkeypatch.setenv("STRATEGY_FEEDBACK_TRADES_DB", str(trades_db))
    monkeypatch.setenv("STRATEGY_FEEDBACK_PATH", str(log_dir / "strategy_feedback.json"))
    monkeypatch.setenv("STRATEGY_FEEDBACK_SYSTEMD_DIR", str(repo / "systemd"))
    monkeypatch.setenv("STRATEGY_FEEDBACK_LOCAL_PID_DIR", str(repo / "logs" / "local_v2_stack" / "pids"))
    monkeypatch.setenv("STRATEGY_FEEDBACK_MIN_TRADES", "6")

    payload = worker._build_payload(worker.WorkerConfig())

    advice = payload["strategies"]["RangeFader"]
    assert isinstance(advice.get("setup_overrides"), list)
    flow_micro = next(item for item in advice["setup_overrides"] if item["match_dimension"] == "flow_micro")
    assert flow_micro["flow_regime"] == "trend_long"
    assert flow_micro["microstructure_bucket"] == "tight_fast"
    exact = next(item for item in advice["setup_overrides"] if item["match_dimension"] == "setup_fingerprint")
    assert exact["setup_fingerprint"].startswith("RangeFader-sell-fade|short|trend_long|tight_fast|")


def test_remap_stats_prefers_display_case_base_key_over_lowercase_control_slug() -> None:
    stats_by_tag = {
        "MicroTrendRetest-long": worker.StrategyStats(
            tag="MicroTrendRetest-long",
            trades=14,
            wins=9,
            losses=5,
            sum_pips=8.8,
            avg_pips=0.6286,
            avg_abs_pips=0.9143,
            gross_win=10.8,
            gross_loss=2.0,
            avg_hold_sec=45.0,
            last_closed="2026-03-10 00:00:00",
        )
    }
    remapped, _ = worker._remap_stats_to_known_keys(
        stats_by_tag,
        {"MicroTrendRetest-long": "2026-03-10 00:00:00"},
        ["microtrendretest", "MicroTrendRetest"],
    )

    assert "MicroTrendRetest" in remapped
    assert "microtrendretest" not in remapped


def test_main_loop_runs_once_and_sleeps(monkeypatch, tmp_path: Path) -> None:
    feedback_path = tmp_path / "strategy_feedback.json"
    config = SimpleNamespace(loop_sec=0.0, feedback_path=feedback_path)
    run_calls: list[tuple[object, bool]] = []
    sleep_calls: list[float] = []

    monkeypatch.setattr(
        worker,
        "_parse_args",
        lambda: SimpleNamespace(nowrite=False, loop_sec=0.5),
    )
    monkeypatch.setattr(worker, "WorkerConfig", lambda: config)
    monkeypatch.setattr(worker.logging, "basicConfig", lambda **_: None)

    def fake_run_once(actual_config: object, *, nowrite: bool) -> None:
        run_calls.append((actual_config, nowrite))

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        raise KeyboardInterrupt

    monkeypatch.setattr(worker, "_run_once", fake_run_once)
    monkeypatch.setattr(worker.time, "sleep", fake_sleep)

    with pytest.raises(KeyboardInterrupt):
        worker.main()

    assert run_calls == [(config, False)]
    assert sleep_calls == [0.5]
