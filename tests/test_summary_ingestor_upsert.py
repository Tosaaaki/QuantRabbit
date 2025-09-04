import importlib
import sqlite3
import tempfile
from pathlib import Path


def test_upsert_idempotent(tmp_path: Path):
    # Prepare isolated sqlite DB
    db_file = tmp_path / "news.db"
    con = sqlite3.connect(db_file)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS news(
            uid TEXT PRIMARY KEY,
            ts_utc TEXT,
            event_time TEXT,
            horizon TEXT,
            summary TEXT,
            sentiment INTEGER,
            impact INTEGER,
            pair_bias TEXT
        );
        """
    )
    con.commit()

    # Import module and monkeypatch its DB handles
    s = importlib.import_module('analysis.summary_ingestor')
    s.conn = con
    s.cur = cur

    payload = {
        "uid": "u1",
        "event_time": None,
        "horizon": "short",
        "summary": "A",
        "sentiment": 1,
        "impact": 3,
        "pair_bias": "USD_JPY_UP",
    }

    s._upsert(payload)
    # update payload
    payload2 = dict(payload)
    payload2["summary"] = "B"
    s._upsert(payload2)

    rows = list(cur.execute("select uid, summary from news"))
    assert rows == [("u1", "B")]

