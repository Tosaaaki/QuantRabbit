"""
QuantRabbit Trading Memory — DB Schema & Init
SQLite + sqlite-vec (via APSW) for vector-searchable trade memory DB
"""
from __future__ import annotations

import apsw
import sqlite_vec
import struct
from pathlib import Path

DB_PATH = Path(__file__).parent / "memory.db"
VEC_DIM = 256  # Ruri v3-30m


def get_conn() -> apsw.Connection:
    conn = apsw.Connection(str(DB_PATH))
    conn.setbusytimeout(5000)  # 5秒リトライ（cron並列アクセス時のBusyError防止）
    conn.enableloadextension(True)
    conn.loadextension(sqlite_vec.loadable_path())
    conn.enableloadextension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _table_columns(conn: apsw.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _ensure_column(conn: apsw.Connection, table: str, column_def: str):
    column = column_def.split()[0]
    if column not in _table_columns(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")


def init_db():
    conn = get_conn()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_date TEXT NOT NULL,
            chunk_type TEXT NOT NULL,
            question TEXT,
            content TEXT NOT NULL,
            pair TEXT,
            direction TEXT,
            tags TEXT,
            source_file TEXT,
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    """)
    _ensure_column(conn, "chunks", "direction TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_date ON chunks(session_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_pair ON chunks(pair)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_pair_direction ON chunks(pair, direction)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type)")

    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding float[{VEC_DIM}]
        )
    """)

    # --- ② 構造化トレード記録 ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_date TEXT NOT NULL,
            trade_id TEXT,              -- OANDA Trade ID
            pair TEXT NOT NULL,
            direction TEXT NOT NULL,     -- LONG / SHORT
            units INTEGER,
            entry_price REAL,
            exit_price REAL,
            pl REAL,                    -- 確定損益（円）
            -- テクニカルコンテキスト（エントリー時点）
            h1_adx REAL,
            h1_trend TEXT,              -- BULL / BEAR / FLAT
            m5_adx REAL,
            m5_trend TEXT,
            rsi REAL,
            stoch_rsi REAL,
            -- マーケットコンテキスト
            regime TEXT,                -- quiet / trending / headline / thin_liquidity
            vix REAL,
            dxy REAL,
            active_headlines TEXT,       -- 自由記述 "Iran/Hormuz escalation"
            event_risk TEXT,            -- geopolitical / central_bank / data_release / none
            session_hour INTEGER,       -- UTC時間 (0-23)
            -- 行動分析
            entry_type TEXT,            -- planned / fomo / revenge / user_directed / re_entry
            had_sl INTEGER DEFAULT 0,   -- 0/1
            reason TEXT,
            lesson TEXT,
            user_call_id INTEGER,       -- user_calls との紐付け
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_pair ON trades(pair)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(session_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_regime ON trades(regime)")

    # --- ③ マーケットイベント記録 ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS market_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_date TEXT NOT NULL,
            timestamp TEXT,             -- ISO8601
            event_type TEXT NOT NULL,    -- geopolitical / central_bank / data_release / technical_break
            headline TEXT,              -- イベント内容
            pairs_affected TEXT,        -- カンマ区切り
            spike_pips REAL,            -- 最大変動幅
            spike_direction TEXT,       -- UP / DOWN
            duration_min INTEGER,       -- スパイク持続時間
            pre_vix REAL,
            post_vix REAL,
            impact TEXT,                -- high / medium / low
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    """)

    # --- ④ ユーザー直感記録 ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_date TEXT NOT NULL,
            timestamp TEXT,
            pair TEXT,
            direction TEXT,             -- UP / DOWN
            call_text TEXT,             -- 原文 "ドルストあがるよ"
            -- エントリー時のコンディション
            conditions TEXT,            -- JSON: {"stoch_rsi": 0.0, "rsi": 33, "h1_trend": "BULL", ...}
            price_at_call REAL,
            -- アウトカム
            outcome TEXT,               -- correct / incorrect / partial / pending
            pl_after_30m REAL,
            pl_after_1h REAL,
            price_after_30m REAL,
            price_after_1h REAL,
            -- メタ
            confidence TEXT,            -- strong / normal / tentative
            acted_on INTEGER DEFAULT 0, -- 実際にトレードに反映したか
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_user_calls_pair ON user_calls(pair)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_user_calls_outcome ON user_calls(outcome)")

    # --- ⑤ pretrade_outcomes: pretrade_checkの予測 vs 実際の結果 ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pretrade_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_date TEXT NOT NULL,
            trade_id TEXT,
            pair TEXT NOT NULL,
            direction TEXT NOT NULL,
            pretrade_level TEXT,
            pretrade_score INTEGER,
            pretrade_warnings TEXT,
            pl REAL,
            thesis TEXT,
            lesson_from_review TEXT,
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pretrade_pair ON pretrade_outcomes(pair)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pretrade_date ON pretrade_outcomes(session_date)")

    # --- ⑥ seat_outcomes: S-hunt discovery / deployment / capture / miss chain ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS seat_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_date TEXT NOT NULL,
            state_last_updated TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 's_hunt',
            horizon TEXT NOT NULL,
            pair TEXT NOT NULL,
            direction TEXT NOT NULL,
            setup_type TEXT,
            why TEXT,
            mtf_chain TEXT,
            payout_path TEXT,
            orderability TEXT,
            deployment_result TEXT,
            trigger TEXT,
            invalidation TEXT,
            reference_price REAL,
            discovered INTEGER DEFAULT 1,
            orderable INTEGER DEFAULT 0,
            deployed INTEGER DEFAULT 0,
            captured INTEGER DEFAULT 0,
            missed INTEGER DEFAULT 0,
            directionally_correct INTEGER,
            deployment_status TEXT,
            outcome_status TEXT,
            matched_trade_count INTEGER DEFAULT 0,
            matched_trade_ids TEXT,
            realized_pl REAL,
            eval_price REAL,
            eval_price_source TEXT,
            pip_move REAL,
            notes TEXT,
            created_at TEXT DEFAULT (datetime('now', 'localtime')),
            updated_at TEXT DEFAULT (datetime('now', 'localtime')),
            UNIQUE(state_last_updated, horizon, pair, direction)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_seat_outcomes_date ON seat_outcomes(session_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_seat_outcomes_pair ON seat_outcomes(pair, direction)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_seat_outcomes_horizon ON seat_outcomes(horizon)")

    print(f"memory.db initialized at {DB_PATH}")


def serialize_f32(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


# --- APSW row helper (dict-like access) ---

def fetchall_dict(conn: apsw.Connection, sql: str, params=None) -> list[dict]:
    cursor = conn.execute(sql, params or ())
    try:
        desc = cursor.getdescription()
    except apsw.ExecutionCompleteError:
        return []
    cols = [d[0] for d in desc]
    return [dict(zip(cols, row)) for row in cursor]


def fetchone_dict(conn: apsw.Connection, sql: str, params=None) -> dict | None:
    cursor = conn.execute(sql, params or ())
    try:
        desc = cursor.getdescription()
    except apsw.ExecutionCompleteError:
        return None
    cols = [d[0] for d in desc]
    row = next(cursor, None)
    return dict(zip(cols, row)) if row else None


def fetchone_val(conn: apsw.Connection, sql: str, params=None):
    row = next(conn.execute(sql, params or ()), None)
    return row[0] if row else None


if __name__ == "__main__":
    init_db()
