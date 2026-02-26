from __future__ import annotations

import importlib
import sqlite3
from pathlib import Path

from analysis.pattern_book import build_pattern_id


def _create_pattern_db(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE pattern_scores (
          pattern_id TEXT PRIMARY KEY,
          strategy_tag TEXT,
          pocket TEXT,
          direction TEXT,
          trades INTEGER,
          wins INTEGER,
          losses INTEGER,
          win_rate REAL,
          bayes_win_rate REAL,
          avg_pips REAL,
          shrink_avg_pips REAL,
          total_pips REAL,
          profit_factor REAL,
          avg_hold_sec REAL,
          spread_pips_mean REAL,
          tp_pips_mean REAL,
          sl_pips_mean REAL,
          confidence_mean REAL,
          chase_risk_rate REAL,
          p_value REAL,
          z_edge REAL,
          robust_score REAL,
          suggested_multiplier REAL,
          quality TEXT,
          is_significant INTEGER,
          score_rank INTEGER,
          boot_ci_low REAL,
          boot_ci_high REAL,
          last_close_time TEXT,
          updated_at TEXT
        );
        CREATE TABLE pattern_drift (
          pattern_id TEXT PRIMARY KEY,
          recent_trades INTEGER,
          prev_trades INTEGER,
          recent_avg_pips REAL,
          prev_avg_pips REAL,
          delta_avg_pips REAL,
          recent_win_rate REAL,
          prev_win_rate REAL,
          delta_win_rate REAL,
          p_value REAL,
          drift_state TEXT,
          updated_at TEXT
        );
        CREATE TABLE pattern_actions (
          pattern_id TEXT PRIMARY KEY,
          action TEXT NOT NULL,
          lot_multiplier REAL NOT NULL,
          reason TEXT NOT NULL,
          trades INTEGER NOT NULL,
          win_rate REAL NOT NULL,
          profit_factor REAL NOT NULL,
          avg_pips REAL NOT NULL,
          updated_at TEXT NOT NULL
        );
        """
    )
    con.commit()
    con.close()


def _insert_score(
    db_path: Path,
    *,
    pattern_id: str,
    trades: int,
    quality: str,
    suggested_multiplier: float,
    robust_score: float,
    p_value: float,
) -> None:
    con = sqlite3.connect(db_path)
    con.execute(
        """
        INSERT INTO pattern_scores (
          pattern_id, strategy_tag, pocket, direction, trades, wins, losses,
          win_rate, bayes_win_rate, avg_pips, shrink_avg_pips, total_pips,
          profit_factor, avg_hold_sec, spread_pips_mean, tp_pips_mean,
          sl_pips_mean, confidence_mean, chase_risk_rate, p_value, z_edge,
          robust_score, suggested_multiplier, quality, is_significant,
          score_rank, boot_ci_low, boot_ci_high, last_close_time, updated_at
        ) VALUES (
          ?, 'scalp_ping_5s_live', 'scalp_fast', 'short',
          ?, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          ?, 0.0, ?, ?, ?, 0, 1, NULL, NULL, '', '2026-02-12T09:33:10+00:00'
        )
        """,
        (
            pattern_id,
            int(trades),
            float(p_value),
            float(robust_score),
            float(suggested_multiplier),
            str(quality),
        ),
    )
    con.commit()
    con.close()


def _insert_drift(
    db_path: Path,
    *,
    pattern_id: str,
    drift_state: str,
) -> None:
    con = sqlite3.connect(db_path)
    con.execute(
        """
        INSERT INTO pattern_drift (
          pattern_id, recent_trades, prev_trades, recent_avg_pips, prev_avg_pips,
          delta_avg_pips, recent_win_rate, prev_win_rate, delta_win_rate, p_value,
          drift_state, updated_at
        ) VALUES (?, 14, 80, -0.18, 0.03, -0.21, 0.35, 0.54, -0.19, 0.03, ?, '2026-02-12T09:33:10+00:00')
        """,
        (pattern_id, drift_state),
    )
    con.commit()
    con.close()


def _reload_gate(monkeypatch, db_path: Path):
    monkeypatch.setenv("ORDER_PATTERN_GATE_ENABLED", "1")
    monkeypatch.setenv("ORDER_PATTERN_GATE_DB_PATH", str(db_path))
    monkeypatch.setenv("ORDER_PATTERN_GATE_TTL_SEC", "0")
    monkeypatch.setenv("ORDER_PATTERN_GATE_SCALE_MIN_TRADES", "30")
    monkeypatch.setenv("ORDER_PATTERN_GATE_BLOCK_MIN_TRADES", "80")
    monkeypatch.setenv("ORDER_PATTERN_GATE_SCALE_MAX", "1.18")
    monkeypatch.setenv("ORDER_PATTERN_GATE_SCALE_MIN", "0.70")
    monkeypatch.setenv("ORDER_PATTERN_GATE_MIN_SCALE_DELTA", "0.01")
    monkeypatch.setenv("ORDER_PATTERN_GATE_ALLOW_BOOST", "1")
    monkeypatch.setenv("ORDER_PATTERN_GATE_POCKET_ALLOWLIST", "scalp_fast")
    monkeypatch.setenv("ORDER_PATTERN_GATE_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("ORDER_PATTERN_GATE_FALLBACK_DISABLE_BLOCK", "1")
    monkeypatch.setenv("ORDER_PATTERN_GATE_FALLBACK_ALLOW_BOOST", "0")
    monkeypatch.setenv("ORDER_PATTERN_GATE_FALLBACK_SCALE_MIN", "0.85")
    monkeypatch.setenv("ORDER_PATTERN_GATE_FALLBACK_SCALE_MAX", "1.05")
    import workers.common.pattern_gate as pattern_gate

    return importlib.reload(pattern_gate)


def _entry_thesis_for_pattern() -> dict:
    return {
        "strategy_tag": "scalp_ping_5s_live",
        "pattern_gate_opt_in": True,
        "signal_mode": "market_ping_5s",
        "mtf_regime_gate": "mtf_reversion_aligned",
        "horizon_gate": "horizon_align",
        "extrema_gate_reason": "short_bottom_soft",
        "section_axis": {"high": 153.22, "low": 152.91},
        "entry_ref": 152.95,
        "pattern_tag": "m5_ping_short",
    }


def test_pattern_gate_blocks_avoid(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "patterns.db"
    _create_pattern_db(db_path)
    entry = _entry_thesis_for_pattern()
    pattern_id = build_pattern_id(
        entry_thesis=entry,
        units=-1400,
        pocket="scalp_fast",
        strategy_tag_fallback="scalp_ping_5s_live",
    )
    _insert_score(
        db_path,
        pattern_id=pattern_id,
        trades=120,
        quality="avoid",
        suggested_multiplier=0.72,
        robust_score=-1.6,
        p_value=0.02,
    )
    gate = _reload_gate(monkeypatch, db_path)

    decision = gate.decide(
        strategy_tag="scalp_ping_5s_live",
        pocket="scalp_fast",
        side="sell",
        units=-1400,
        entry_thesis=entry,
    )
    assert decision is not None
    assert decision.allowed is False
    assert decision.action == "block"
    assert decision.reason == "pattern_avoid"
    assert decision.pattern_id == pattern_id


def test_pattern_gate_scales_with_drift_penalty(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "patterns.db"
    _create_pattern_db(db_path)
    entry = _entry_thesis_for_pattern()
    pattern_id = build_pattern_id(
        entry_thesis=entry,
        units=-1800,
        pocket="scalp_fast",
        strategy_tag_fallback="scalp_ping_5s_live",
    )
    _insert_score(
        db_path,
        pattern_id=pattern_id,
        trades=90,
        quality="weak",
        suggested_multiplier=0.95,
        robust_score=-0.8,
        p_value=0.28,
    )
    _insert_drift(db_path, pattern_id=pattern_id, drift_state="deterioration")
    gate = _reload_gate(monkeypatch, db_path)

    decision = gate.decide(
        strategy_tag="scalp_ping_5s_live",
        pocket="scalp_fast",
        side="sell",
        units=-1800,
        entry_thesis=entry,
    )
    assert decision is not None
    assert decision.allowed is True
    assert decision.action == "scale"
    assert decision.scale < 1.0
    assert decision.drift_state == "deterioration"


def test_pattern_gate_boosts_and_clamps_scale(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "patterns.db"
    _create_pattern_db(db_path)
    entry = _entry_thesis_for_pattern()
    pattern_id = build_pattern_id(
        entry_thesis=entry,
        units=-2200,
        pocket="scalp_fast",
        strategy_tag_fallback="scalp_ping_5s_live",
    )
    _insert_score(
        db_path,
        pattern_id=pattern_id,
        trades=100,
        quality="robust",
        suggested_multiplier=1.33,
        robust_score=2.3,
        p_value=0.03,
    )
    gate = _reload_gate(monkeypatch, db_path)

    decision = gate.decide(
        strategy_tag="scalp_ping_5s_live",
        pocket="scalp_fast",
        side="sell",
        units=-2200,
        entry_thesis=entry,
    )
    assert decision is not None
    assert decision.allowed is True
    assert decision.action == "scale"
    assert decision.scale == 1.18
    assert decision.reason == "pattern_boost"


def test_pattern_gate_no_opt_in_returns_none(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "patterns.db"
    _create_pattern_db(db_path)
    entry = _entry_thesis_for_pattern()
    entry.pop("pattern_gate_opt_in", None)
    pattern_id = build_pattern_id(
        entry_thesis=entry,
        units=-1800,
        pocket="scalp_fast",
        strategy_tag_fallback="scalp_ping_5s_live",
    )
    _insert_score(
        db_path,
        pattern_id=pattern_id,
        trades=120,
        quality="avoid",
        suggested_multiplier=0.70,
        robust_score=-1.3,
        p_value=0.03,
    )
    gate = _reload_gate(monkeypatch, db_path)
    decision = gate.decide(
        strategy_tag="scalp_ping_5s_live",
        pocket="scalp_fast",
        side="sell",
        units=-1800,
        entry_thesis=entry,
    )
    assert decision is None


def test_pattern_gate_fallback_matches_when_range_bucket_differs(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "patterns.db"
    _create_pattern_db(db_path)
    base_entry = _entry_thesis_for_pattern()
    base_pid = build_pattern_id(
        entry_thesis=base_entry,
        units=-1800,
        pocket="scalp_fast",
        strategy_tag_fallback="scalp_ping_5s_live",
    )
    _insert_score(
        db_path,
        pattern_id=base_pid,
        trades=120,
        quality="weak",
        suggested_multiplier=0.72,
        robust_score=-0.7,
        p_value=0.12,
    )
    gate = _reload_gate(monkeypatch, db_path)

    entry = _entry_thesis_for_pattern()
    entry["entry_ref"] = 153.20
    entry["pattern_tag"] = "new_pattern_tag"
    fallback_pid = build_pattern_id(
        entry_thesis=entry,
        units=-1800,
        pocket="scalp_fast",
        strategy_tag_fallback="scalp_ping_5s_live",
    )
    assert fallback_pid != base_pid

    decision = gate.decide(
        strategy_tag="scalp_ping_5s_live",
        pocket="scalp_fast",
        side="sell",
        units=-1800,
        entry_thesis=entry,
    )
    assert decision is not None
    assert decision.match_mode != "exact"
    assert decision.action == "scale"
    assert decision.reason == "pattern_fallback_reduce"
    assert decision.scale == 0.85
    assert decision.pattern_id == base_pid
    assert decision.requested_pattern_id == fallback_pid


def test_pattern_gate_fallback_does_not_block_avoid_by_default(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "patterns.db"
    _create_pattern_db(db_path)
    base_entry = _entry_thesis_for_pattern()
    base_pid = build_pattern_id(
        entry_thesis=base_entry,
        units=-1600,
        pocket="scalp_fast",
        strategy_tag_fallback="scalp_ping_5s_live",
    )
    _insert_score(
        db_path,
        pattern_id=base_pid,
        trades=180,
        quality="avoid",
        suggested_multiplier=0.70,
        robust_score=-1.4,
        p_value=0.01,
    )
    gate = _reload_gate(monkeypatch, db_path)

    entry = _entry_thesis_for_pattern()
    entry["entry_ref"] = 153.19
    entry["pattern_tag"] = "fallback_case"
    requested_pid = build_pattern_id(
        entry_thesis=entry,
        units=-1600,
        pocket="scalp_fast",
        strategy_tag_fallback="scalp_ping_5s_live",
    )
    assert requested_pid != base_pid

    decision = gate.decide(
        strategy_tag="scalp_ping_5s_live",
        pocket="scalp_fast",
        side="sell",
        units=-1600,
        entry_thesis=entry,
    )
    assert decision is not None
    assert decision.allowed is True
    assert decision.action == "scale"
    assert decision.reason == "pattern_fallback_reduce"
    assert decision.match_mode != "exact"
    assert decision.pattern_id == base_pid
