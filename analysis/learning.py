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
import time
from datetime import datetime as dt, timezone
from typing import Dict, List, Tuple, Optional

try:  # Firestore is optional in some environments
    from google.cloud import firestore  # type: ignore
    _FS_AVAILABLE = True
except Exception:  # noqa: BLE001
    firestore = None  # type: ignore
    _FS_AVAILABLE = False

try:
    from analysis import risk_feed  # type: ignore
except Exception:  # noqa: BLE001
    risk_feed = None  # type: ignore

try:
    from analysis import bq_stats  # type: ignore
except Exception:  # noqa: BLE001
    bq_stats = None  # type: ignore

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
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS strategy_meta_perf (
          strategy TEXT,
          session TEXT,
          direction TEXT,
          pocket TEXT,
          hold_bucket TEXT,
          trades INTEGER,
          wins INTEGER,
          losses INTEGER,
          sum_pips REAL,
          avg_pips REAL,
          updated_at TEXT,
          PRIMARY KEY(strategy, session, direction, pocket, hold_bucket)
        )
        """
    )
    con.commit()


_ensure_tables()


def _parse_iso(ts: Optional[str]) -> Optional[dt]:
    if not ts:
        return None
    ts = ts.replace("Z", "+00:00")
    try:
        parsed = dt.fromisoformat(ts)
        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    except ValueError:
        return None


def _session_of(entry: Optional[dt]) -> str:
    if not entry:
        return "unknown"
    hour = entry.hour
    if 0 <= hour < 7:
        return "asia"
    if 7 <= hour < 12:
        return "europe"
    if 12 <= hour < 20:
        return "us"
    return "late_us"


def _hold_bucket(minutes: Optional[float]) -> str:
    if minutes is None:
        return "unknown"
    if minutes < 30:
        return "<30m"
    if minutes < 120:
        return "30-120m"
    if minutes < 360:
        return "120-360m"
    return ">=360m"


def record_trade_performance(
    strategy: str | None,
    macro_regime: str | None,
    micro_regime: str | None,
    pl_pips: float | None,
    context: Optional[dict] = None,
) -> None:
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

    if context:
        entry_dt = _parse_iso(context.get("entry_time")) if isinstance(context, dict) else None
        close_dt = _parse_iso(context.get("close_time")) if isinstance(context, dict) else None
        session = context.get("session") or _session_of(entry_dt)
        units = context.get("units") if isinstance(context, dict) else None
        try:
            direction = "long" if float(units or 0) > 0 else "short"
        except (TypeError, ValueError):
            direction = "unknown"
        pocket_ctx = context.get("pocket") if isinstance(context, dict) else None
        pocket_key = pocket_ctx or (macro_regime or "?")
        hold_minutes = None
        if entry_dt and close_dt:
            hold_minutes = (close_dt - entry_dt).total_seconds() / 60.0
        hold_bucket = _hold_bucket(hold_minutes)
        meta_row = cur.execute(
            "SELECT trades, wins, losses, sum_pips FROM strategy_meta_perf WHERE strategy=? AND session=? AND direction=? AND pocket=? AND hold_bucket=?",
            (strategy, session, direction, pocket_key, hold_bucket),
        ).fetchone()
        if meta_row:
            m_trades, m_wins, m_losses, m_sum = meta_row
        else:
            m_trades = m_wins = m_losses = 0
            m_sum = 0.0
        m_trades += 1
        if pl_pips > 0:
            m_wins += 1
        elif pl_pips < 0:
            m_losses += 1
        m_sum = float(m_sum) + float(pl_pips)
        m_avg = m_sum / m_trades if m_trades else 0.0
        cur.execute(
            """
            INSERT INTO strategy_meta_perf(strategy, session, direction, pocket, hold_bucket, trades, wins, losses, sum_pips, avg_pips, updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(strategy, session, direction, pocket, hold_bucket) DO UPDATE SET
              trades=excluded.trades,
              wins=excluded.wins,
              losses=excluded.losses,
              sum_pips=excluded.sum_pips,
              avg_pips=excluded.avg_pips,
              updated_at=excluded.updated_at
            """,
            (
                strategy,
                session,
                direction,
                pocket_key,
                hold_bucket,
                m_trades,
                m_wins,
                m_losses,
                m_sum,
                m_avg,
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
    """過去実績からリスク係数を返す。安全側のクリップあり。

    優先順位:
      0) Vertex/BigQuery ML からの最新推論（Pub/Sub 経由 risk_feed）
      0.5) BigQuery 生統計（`analysis.bq_stats`）
      1) Firestore "status/trader.learning_scores"（Cloud Run 本番）
      2) ローカル SQLite `strategy_perf`（ローカル/回帰用）
    """

    # 0) Pub/Sub (Vertex AI / BQ ML) feed
    if risk_feed is not None:
        try:
            feed_mult = risk_feed.get_multiplier(pocket, strategy)
        except Exception:
            feed_mult = None
        if feed_mult is not None:
            return max(0.4, min(1.6, float(feed_mult)))

    # 0.5) BigQuery aggregated stats
    if bq_stats is not None:
        try:
            stat_mult = bq_stats.multiplier_hint(pocket, strategy)
        except Exception:
            stat_mult = None
        if stat_mult is not None:
            return max(0.4, min(1.5, float(stat_mult)))

    # 1) Firestore の snapshot にある最新スコア（trader_service が更新）
    score = _fs_score(strategy)
    if score is None:
        # 2) ローカルSQLiteの総合スコア（簡易）
        cur = con.cursor()
        row = cur.execute(
            "SELECT trades, wins, losses, avg_pips FROM strategy_perf WHERE strategy=?",
            (strategy,),
        ).fetchone()
        if row:
            tr, w, l, avg = row
            score = _score_row(tr, w, l, avg)

    if score is None:
        return 1.0

    # スコアにより 0.7x〜1.3x の範囲で段階調整
    try:
        val = float(score)
    except (TypeError, ValueError):
        val = 0.0
    if val > 0.15:
        return 1.3
    if val > 0.05:
        return 1.15
    if val < -0.15:
        return 0.7
    if val < -0.05:
        return 0.85
    return 1.0


# --- Firestore learning score cache ---
_FS_CACHE: Dict[str, object] = {"ts": 0.0, "scores": {}}
_FS_TTL = float(pathlib.os.getenv("LEARNING_FS_CACHE_SEC", "60"))


def _fs_score(strategy: str) -> Optional[float]:
    if not _FS_AVAILABLE:
        return None
    now = time.time()
    try:
        if now - float(_FS_CACHE.get("ts", 0.0)) > _FS_TTL:
            fs = firestore.Client()
            doc = fs.collection("status").document("trader").get()
            data = doc.to_dict() if doc and doc.exists else None
            scores = data.get("learning_scores") if isinstance(data, dict) else None
            if isinstance(scores, dict):
                _FS_CACHE["scores"] = scores
                _FS_CACHE["ts"] = now
        scores_dict = _FS_CACHE.get("scores") or {}
        if isinstance(scores_dict, dict):
            raw = scores_dict.get(strategy)
            try:
                return float(raw)
            except (TypeError, ValueError):
                return None
    except Exception:
        return None
    return None
