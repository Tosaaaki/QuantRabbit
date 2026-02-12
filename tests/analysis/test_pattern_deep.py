from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from analysis.pattern_deep import DeepPatternConfig, run_pattern_deep_analysis


def _create_features_table(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE pattern_trade_features (
          pattern_id TEXT NOT NULL,
          pocket TEXT NOT NULL,
          strategy_tag TEXT NOT NULL,
          direction TEXT NOT NULL,
          close_time TEXT NOT NULL,
          hold_sec REAL NOT NULL,
          pl_pips REAL NOT NULL,
          signal_mode TEXT,
          mtf_gate TEXT,
          horizon_gate TEXT,
          extrema_reason TEXT,
          confidence REAL,
          spread_pips REAL,
          tp_pips REAL,
          sl_pips REAL
        );
        """
    )


def _insert_trade(
    con: sqlite3.Connection,
    *,
    pattern_id: str,
    side: str,
    close_time: str,
    pl_pips: float,
    hold_sec: float = 55.0,
) -> None:
    con.execute(
        """
        INSERT INTO pattern_trade_features (
          pattern_id, pocket, strategy_tag, direction, close_time,
          hold_sec, pl_pips, signal_mode, mtf_gate, horizon_gate,
          extrema_reason, confidence, spread_pips, tp_pips, sl_pips
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            pattern_id,
            "scalp_fast",
            "scalp_ping_5s_live",
            side,
            close_time,
            hold_sec,
            pl_pips,
            "momentum_hz",
            "mtf_reversion_aligned",
            "horizon_align",
            "na",
            65.0,
            0.11,
            0.85,
            0.65,
        ),
    )


def test_run_pattern_deep_analysis_detects_quality_and_drift(tmp_path: Path) -> None:
    db_path = tmp_path / "patterns.db"
    con = sqlite3.connect(db_path)
    _create_features_table(con)

    as_of = datetime(2026, 2, 12, 9, 20, tzinfo=timezone.utc)
    pattern_good = "st:scalp_ping_5s_live|pk:scalp_fast|sd:short|sg:mom|mtf:ok|hz:ok|ex:na|rg:high|pt:good"
    pattern_bad = "st:scalp_ping_5s_live|pk:scalp_fast|sd:long|sg:mom|mtf:ok|hz:ok|ex:na|rg:top|pt:bad"
    pattern_drift = "st:scalp_ping_5s_live|pk:scalp_fast|sd:short|sg:mom|mtf:ok|hz:ok|ex:na|rg:mid|pt:drift"

    for i in range(60):
        close_ts = (as_of - timedelta(days=14) + timedelta(hours=i)).isoformat()
        _insert_trade(
            con,
            pattern_id=pattern_good,
            side="short",
            close_time=close_ts,
            pl_pips=0.26 if i % 2 == 0 else 0.14,
        )

    for i in range(60):
        close_ts = (as_of - timedelta(days=13) + timedelta(hours=i)).isoformat()
        _insert_trade(
            con,
            pattern_id=pattern_bad,
            side="long",
            close_time=close_ts,
            pl_pips=-0.30 if i % 2 == 0 else -0.16,
        )

    for i in range(40):
        close_ts = (as_of - timedelta(days=10) + timedelta(hours=i)).isoformat()
        _insert_trade(
            con,
            pattern_id=pattern_drift,
            side="short",
            close_time=close_ts,
            pl_pips=0.22 if i % 2 == 0 else 0.10,
        )
    for i in range(12):
        close_ts = (as_of - timedelta(days=2) + timedelta(hours=i)).isoformat()
        _insert_trade(
            con,
            pattern_id=pattern_drift,
            side="short",
            close_time=close_ts,
            pl_pips=-0.34 if i % 2 == 0 else -0.24,
        )
    con.commit()

    output_path = tmp_path / "pattern_book_deep.json"
    summary = run_pattern_deep_analysis(
        con,
        cutoff_iso=(as_of - timedelta(days=30)).isoformat(),
        as_of=as_of.isoformat(timespec="seconds"),
        output_path=output_path,
        config=DeepPatternConfig(
            min_samples=20,
            prior_strength=18,
            recent_days=3,
            baseline_days=20,
            min_recent_samples=8,
            min_prev_samples=20,
            bootstrap_samples=120,
            cluster_min=2,
            cluster_max=4,
            cluster_min_samples=20,
            random_state=7,
        ),
    )
    con.commit()

    assert summary["patterns_scored"] >= 3
    assert summary["cluster_count"] >= 1
    assert output_path.exists()

    quality_rows = dict(
        con.execute("SELECT pattern_id, quality FROM pattern_scores").fetchall()
    )
    assert quality_rows[pattern_good] in {"robust", "candidate"}
    assert quality_rows[pattern_bad] in {"avoid", "weak"}

    chase_rate = con.execute(
        "SELECT chase_risk_rate FROM pattern_scores WHERE pattern_id=?",
        (pattern_bad,),
    ).fetchone()
    assert chase_rate is not None
    assert float(chase_rate[0]) >= 0.99

    drift = con.execute(
        "SELECT drift_state FROM pattern_drift WHERE pattern_id=?",
        (pattern_drift,),
    ).fetchone()
    assert drift is not None
    assert drift[0] in {"deterioration", "soft_deterioration"}

    payload = json.loads(output_path.read_text())
    assert payload["patterns_scored"] >= 3
    assert len(payload["top_robust"]) > 0
    assert len(payload["top_weak"]) > 0
    assert isinstance(payload["cluster_summary"], list)


def test_run_pattern_deep_analysis_empty_input(tmp_path: Path) -> None:
    db_path = tmp_path / "patterns.db"
    con = sqlite3.connect(db_path)
    _create_features_table(con)
    con.commit()

    output_path = tmp_path / "pattern_book_deep_empty.json"
    summary = run_pattern_deep_analysis(
        con,
        cutoff_iso="2026-01-01T00:00:00+00:00",
        as_of="2026-02-12T09:20:00+00:00",
        output_path=output_path,
        config=DeepPatternConfig(min_samples=20),
    )
    con.commit()

    assert summary["rows_total"] == 0
    assert summary["patterns_scored"] == 0
    assert output_path.exists()
    assert con.execute("SELECT COUNT(*) FROM pattern_scores").fetchone()[0] == 0
