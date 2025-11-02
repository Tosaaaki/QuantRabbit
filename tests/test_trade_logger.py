import os
import sqlite3
import tempfile
import importlib
from pathlib import Path
import sys


def setup_trade_logger(tmp_path: Path):
    os.environ["TRADE_LOGGER_DB"] = str(tmp_path / "trades.db")
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Reload module so it picks up new env var and resets connection
    module = importlib.import_module("utils.trade_logger")
    importlib.reload(module)
    module._reset_connection_for_tests()  # type: ignore[attr-defined]
    return module


def fetch_row(db_path: Path, ticket_id: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM trades WHERE ticket_id=?", (ticket_id,)).fetchone()
    conn.close()
    return row


def test_log_trade_snapshot_inserts_and_updates(tmp_path):
    module = setup_trade_logger(tmp_path)
    db_path = tmp_path / "trades.db"

    module.log_trade_snapshot(
        {
            "trade_id": "T123",
            "pocket": "micro",
            "instrument": "USD_JPY",
            "units": 10,
            "price": 150.5,
            "entry_time": "2025-10-21T03:30:00Z",
            "strategy": "RangeBounce",
        }
    )

    row = fetch_row(db_path, "T123")
    assert row is not None
    assert row["ticket_id"] == "T123"
    assert row["state"] in ("OPEN", "PLACED")
    assert row["entry_time"] == "2025-10-21T03:30:00Z"
    assert row["close_time"] is None

    module.log_trade_snapshot(
        {
            "trade_id": "T123",
            "state": "CLOSED",
            "close_time": "2025-10-21T03:45:00Z",
            "pl_pips": 15.2,
            "realized_pl": 12.34,
        }
    )

    row = fetch_row(db_path, "T123")
    assert row["state"] == "CLOSED"
    assert row["close_time"] == "2025-10-21T03:45:00Z"
    assert abs(row["pl_pips"] - 15.2) < 1e-9
    assert abs(row["realized_pl"] - 12.34) < 1e-9


def test_invalid_trade_id_is_ignored(tmp_path):
    module = setup_trade_logger(tmp_path)
    db_path = tmp_path / "trades.db"

    module.log_trade_snapshot({"trade_id": "", "units": 10})
    module.log_trade_snapshot({"trade_id": "FAIL", "units": 10})

    if not db_path.exists():
        return
    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    except sqlite3.OperationalError:
        count = 0
    finally:
        conn.close()
    assert count == 0
