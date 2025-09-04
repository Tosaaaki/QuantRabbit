from __future__ import annotations

"""
analysis.learning
~~~~~~~~~~~~~~~~~
シンプルな自己改善ロジック:
- 各戦略×(macro_regime, micro_regime) の実績を `logs/trades.db` で集計
- GPT の提案ランクに対して、直近実績で再ランク付け
- リスク配分を成績に応じて微調整
"""

import sqlite3
import pathlib
import datetime
from typing import Dict, List, Tuple

_DB = pathlib.Path("logs/trades.db")
_DB.parent.mkdir(exist_ok=True)
con = sqlite3.connect(_DB)


def _ensure_tables() -> None:
    cur = con.cursor()
    # trades テーブルは他モジュールで作成される想定。存在しない可能性に配慮
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
          id INTEGER PRIMARY KEY,
          ticket_id TEXT UNIQUE,
          pocket TEXT,
          instrument TEXT,
          units INTEGER,
          entry_price REAL,
          close_price REAL,
          pl_pips REAL,
          entry_time TEXT,
          close_time TEXT,
          strategy TEXT,
          macro_regime TEXT,
          micro_regime TEXT
        )
        """
    )
    # 戦略実績テーブル
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS strategy_perf (
          strategy TEXT,
          macro_regime TEXT,
          micro_regime TEXT,
          trades INTEGER,
          wins INTEGER,
          losses INTEGER,
          sum_pips REAL,
          avg_pips REAL,
          updated_at TEXT,
          PRIMARY KEY(strategy, macro_regime, micro_regime)
        )
        """
    )
    con.commit()


_ensure_tables()


def record_trade_performance(strategy: str | None, macro_regime: str | None, micro_regime: str | None, pl_pips: float | None) -> None:
    """決済毎に呼び出し、戦略の実績を更新する。"""
    if not strategy or pl_pips is None:
        return
    macro = macro_regime or "?"
    micro = micro_regime or "?"
    cur = con.cursor()
    # 既存データ取得
    row = cur.execute(
        "SELECT trades, wins, losses, sum_pips FROM strategy_perf WHERE strategy=? AND macro_regime=? AND micro_regime=?",
        (strategy, macro, micro),
    ).fetchone()
    if row:
        trades, wins, losses, sum_pips = row
    else:
        trades, wins, losses, sum_pips = 0, 0, 0, 0.0
    trades += 1
    if pl_pips > 0:
        wins += 1
    elif pl_pips < 0:
        losses += 1
    sum_pips = float(sum_pips) + float(pl_pips)
    avg_pips = sum_pips / trades if trades else 0.0
    cur.execute(
        """
        INSERT INTO strategy_perf(strategy, macro_regime, micro_regime, trades, wins, losses, sum_pips, avg_pips, updated_at)
        VALUES(?,?,?,?,?,?,?,?,?)
        ON CONFLICT(strategy, macro_regime, micro_regime) DO UPDATE SET
          trades=excluded.trades,
          wins=excluded.wins,
          losses=excluded.losses,
          sum_pips=excluded.sum_pips,
          avg_pips=excluded.avg_pips,
          updated_at=excluded.updated_at
        """,
        (
            strategy,
            macro,
            micro,
            trades,
            wins,
            losses,
            sum_pips,
            avg_pips,
            datetime.datetime.utcnow().isoformat(timespec="seconds"),
        ),
    )
    con.commit()


def _score_row(trades: int, wins: int, losses: int, avg_pips: float) -> float:
    # シンプルなスコア: 勝率*補正 + 平均pips。少数サンプルにはペナルティ
    total = max(trades, 1)
    win_rate = wins / total
    conf = min(total / 20.0, 1.0)  # 20件で係数1に到達
    return (win_rate * 2.0 - 1.0) * 0.5 * conf + (avg_pips / 20.0) * conf


def get_strategy_scores(macro_regime: str, micro_regime: str) -> Dict[str, float]:
    """現在のレジームに近い行でスコアを返す。Exact → Macro/Micro 片方一致 → 全体平均。"""
    cur = con.cursor()
    scores: Dict[str, float] = {}

    def acc(rows):
        local: Dict[str, Tuple[int,int,int,float]] = {}
        for r in rows:
            s, tr, w, l, avg = r
            if s not in local:
                local[s] = (0,0,0,0.0)
            tr0, w0, l0, avg0 = local[s]
            # 重み付き和（簡易）
            n = tr0 + tr
            avg_comb = (avg0*tr0 + avg*tr) / n if n else 0.0
            local[s] = (tr0+tr, w0+w, l0+l, avg_comb)
        for s,(tr,w,l,avg) in local.items():
            scores[s] = max(scores.get(s, -1e9), _score_row(tr, w, l, avg))

    # Exact match
    rows = cur.execute(
        """
        SELECT strategy, trades, wins, losses, avg_pips
        FROM strategy_perf WHERE macro_regime=? AND micro_regime=?
        """,
        (macro_regime, micro_regime),
    ).fetchall()
    acc(rows)
    # One-side match
    rows = cur.execute(
        "SELECT strategy, trades, wins, losses, avg_pips FROM strategy_perf WHERE macro_regime=? AND micro_regime='?'",
        (macro_regime,),
    ).fetchall()
    acc(rows)
    rows = cur.execute(
        "SELECT strategy, trades, wins, losses, avg_pips FROM strategy_perf WHERE macro_regime='?' AND micro_regime=?",
        (micro_regime,),
    ).fetchall()
    acc(rows)
    # Global fallback
    rows = cur.execute(
        "SELECT strategy, trades, wins, losses, avg_pips FROM strategy_perf",
    ).fetchall()
    acc(rows)
    return scores


def re_rank_strategies(candidates: List[str], macro_regime: str, micro_regime: str) -> List[str]:
    scores = get_strategy_scores(macro_regime, micro_regime)
    # デフォルト順に小さな事前重みを与え、学習スコアで並び替え
    priors = {s: (len(candidates) - i) * 0.01 for i, s in enumerate(candidates)}
    return sorted(candidates, key=lambda s: -(priors.get(s,0.0) + scores.get(s, 0.0)))


def risk_multiplier(pocket: str, strategy: str) -> float:
    """過去実績からリスク係数を返す。安全側のクリップあり。"""
    # 最近の全体平均を使用（簡易）。より精緻にはレジーム別に可能。
    cur = con.cursor()
    row = cur.execute(
        "SELECT trades, wins, losses, avg_pips FROM strategy_perf WHERE strategy=?",
        (strategy,),
    ).fetchone()
    if not row:
        return 1.0
    tr, w, l, avg = row
    score = _score_row(tr, w, l, avg)
    # スコアにより 0.5x〜1.5x の範囲で調整
    if score > 0.1:
        return 1.3
    if score < -0.1:
        return 0.7
    return 1.0

