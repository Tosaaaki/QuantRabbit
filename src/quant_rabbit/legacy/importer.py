from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from quant_rabbit.paths import DEFAULT_HISTORY_DB, DEFAULT_IMPORT_REPORT, DEFAULT_LEGACY_ARCHIVE


TEXT_SUFFIXES = {".md", ".txt", ".json", ".jsonl"}
STRUCTURED_DB = "collab_trade/memory/memory.db"
SENSITIVE_PARTS = {".git", ".venv", ".gcloud", ".gcloud_config", "__pycache__", ".pytest_cache"}
NOISY_PARTS = {"archive/tmp", "logs/archive_legacy"}
LIVE_LOG_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\]\s+"
    r"(?P<action>[A-Z_]+)\s+"
    r"(?P<pair>[A-Z]{3}_[A-Z]{3})?\s*"
    r"(?P<side>LONG|SHORT)?\s*"
    r"(?P<units>\d+)?u?\s*"
    r"(?:@(?P<price>MARKET|[-+]?\d+(?:\.\d+)?))?"
    r".*$"
)


@dataclass(frozen=True)
class ImportSummary:
    archive: Path
    db_path: Path
    source_files: int
    legacy_rows: dict[str, int]
    live_trade_events: int
    journal_events: int
    report_path: Path | None = None


class LegacyImporter:
    def __init__(
        self,
        archive: Path = DEFAULT_LEGACY_ARCHIVE,
        db_path: Path = DEFAULT_HISTORY_DB,
        report_path: Path = DEFAULT_IMPORT_REPORT,
    ) -> None:
        self.archive = archive
        self.db_path = db_path
        self.report_path = report_path

    def run(self) -> ImportSummary:
        if not self.archive.exists():
            raise FileNotFoundError(f"legacy archive not found: {self.archive}")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            self._create_schema(conn)
            self._clear(conn)
            source_count = self._import_source_files(conn)
            legacy_rows = self._import_memory_db(conn)
            live_events = self._import_live_trade_log(conn)
            journal_events = self._import_jsonl_events(conn, "logs/trader_journal.jsonl", "trader_journal")
            self._import_jsonl_events(conn, "logs/s_hunt_ledger.jsonl", "s_hunt_ledger")
            self._import_jsonl_events(conn, "logs/audit_history.jsonl", "audit_history")
            conn.commit()
            self._write_report(conn, source_count, legacy_rows, live_events, journal_events)
        return ImportSummary(
            archive=self.archive,
            db_path=self.db_path,
            source_files=source_count,
            legacy_rows=legacy_rows,
            live_trade_events=live_events,
            journal_events=journal_events,
            report_path=self.report_path,
        )

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS source_files (
                rel_path TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                mtime_utc TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS legacy_records (
                source_table TEXT NOT NULL,
                source_id TEXT,
                session_date TEXT,
                pair TEXT,
                direction TEXT,
                pl REAL,
                execution_style TEXT,
                allocation_band TEXT,
                thesis TEXT,
                raw_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_legacy_records_table ON legacy_records(source_table);
            CREATE INDEX IF NOT EXISTS idx_legacy_records_pair ON legacy_records(pair, direction);
            CREATE INDEX IF NOT EXISTS idx_legacy_records_pl ON legacy_records(pl);
            CREATE TABLE IF NOT EXISTS live_trade_events (
                line_no INTEGER PRIMARY KEY,
                timestamp_text TEXT,
                action TEXT,
                pair TEXT,
                direction TEXT,
                units INTEGER,
                price TEXT,
                pl_jpy REAL,
                spread_pips REAL,
                trade_id TEXT,
                reason TEXT,
                raw_line TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_live_trade_pair ON live_trade_events(pair, direction);
            CREATE INDEX IF NOT EXISTS idx_live_trade_action ON live_trade_events(action);
            CREATE TABLE IF NOT EXISTS jsonl_events (
                source_name TEXT NOT NULL,
                line_no INTEGER NOT NULL,
                event_type TEXT,
                timestamp_utc TEXT,
                pair TEXT,
                direction TEXT,
                raw_json TEXT NOT NULL,
                PRIMARY KEY (source_name, line_no)
            );
            CREATE INDEX IF NOT EXISTS idx_jsonl_events_source ON jsonl_events(source_name);
            CREATE TABLE IF NOT EXISTS import_notes (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )

    def _clear(self, conn: sqlite3.Connection) -> None:
        for table in ("source_files", "legacy_records", "live_trade_events", "jsonl_events", "import_notes"):
            conn.execute(f"DELETE FROM {table}")

    def _import_source_files(self, conn: sqlite3.Connection) -> int:
        count = 0
        for path in self._iter_source_files():
            rel = path.relative_to(self.archive).as_posix()
            try:
                stat = path.stat()
                digest = self._sha256(path)
            except OSError:
                continue
            conn.execute(
                """
                INSERT INTO source_files(rel_path, kind, size_bytes, sha256, mtime_utc)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    rel,
                    path.suffix.lower().lstrip(".") or "file",
                    stat.st_size,
                    digest,
                    datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                ),
            )
            count += 1
        conn.execute("INSERT INTO import_notes(key, value) VALUES (?, ?)", ("archive", str(self.archive)))
        conn.execute(
            "INSERT INTO import_notes(key, value) VALUES (?, ?)",
            ("imported_at_utc", datetime.now(timezone.utc).isoformat()),
        )
        return count

    def _iter_source_files(self) -> Iterable[Path]:
        for path in self.archive.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(self.archive).as_posix()
            parts = set(path.relative_to(self.archive).parts)
            if parts & SENSITIVE_PARTS:
                continue
            if any(rel.startswith(prefix + "/") for prefix in NOISY_PARTS):
                continue
            if path.name == "env.toml" or path.suffix.lower() not in TEXT_SUFFIXES | {".db"}:
                continue
            # Keep source manifest lightweight. Large raw tick/replay artifacts stay in archive.
            try:
                if path.stat().st_size > 25_000_000:
                    continue
            except OSError:
                continue
            yield path

    def _sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _import_memory_db(self, conn: sqlite3.Connection) -> dict[str, int]:
        db_path = self.archive / STRUCTURED_DB
        counts: dict[str, int] = {}
        if not db_path.exists():
            return counts
        src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        src.row_factory = sqlite3.Row
        try:
            for table in ("trades", "pretrade_outcomes", "seat_outcomes", "chunks", "user_calls", "market_events"):
                try:
                    rows = src.execute(f"SELECT * FROM {table}").fetchall()
                except sqlite3.Error:
                    continue
                counts[table] = len(rows)
                for row in rows:
                    payload = dict(row)
                    conn.execute(
                        """
                        INSERT INTO legacy_records(
                            source_table, source_id, session_date, pair, direction, pl,
                            execution_style, allocation_band, thesis, raw_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            table,
                            str(payload.get("id") or payload.get("trade_id") or ""),
                            payload.get("session_date"),
                            payload.get("pair"),
                            payload.get("direction"),
                            _float_or_none(payload.get("pl") if "pl" in payload else payload.get("realized_pl")),
                            payload.get("execution_style"),
                            payload.get("allocation_band"),
                            payload.get("thesis") or payload.get("why"),
                            json.dumps(payload, ensure_ascii=False, sort_keys=True),
                        ),
                    )
        finally:
            src.close()
        return counts

    def _import_live_trade_log(self, conn: sqlite3.Connection) -> int:
        path = self.archive / "logs/live_trade_log.txt"
        if not path.exists():
            return 0
        count = 0
        for line_no, raw in enumerate(path.read_text(errors="replace").splitlines(), 1):
            line = raw.strip()
            if not line:
                continue
            parsed = self._parse_live_line(line)
            conn.execute(
                """
                INSERT INTO live_trade_events(
                    line_no, timestamp_text, action, pair, direction, units, price,
                    pl_jpy, spread_pips, trade_id, reason, raw_line
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    line_no,
                    parsed.get("timestamp_text"),
                    parsed.get("action"),
                    parsed.get("pair"),
                    parsed.get("direction"),
                    parsed.get("units"),
                    parsed.get("price"),
                    parsed.get("pl_jpy"),
                    parsed.get("spread_pips"),
                    parsed.get("trade_id"),
                    parsed.get("reason"),
                    line,
                ),
            )
            count += 1
        return count

    def _parse_live_line(self, line: str) -> dict[str, object]:
        out: dict[str, object] = {}
        timestamp_text, body = _split_timestamp_body(line)
        if timestamp_text:
            out["timestamp_text"] = timestamp_text

        match = LIVE_LOG_RE.match(line)
        if match:
            out.update(
                {
                    "timestamp_text": out.get("timestamp_text") or match.group("ts"),
                    "action": match.group("action"),
                    "pair": match.group("pair"),
                    "direction": match.group("side"),
                    "units": _int_or_none(match.group("units")),
                    "price": match.group("price"),
                }
            )
        if not body:
            body = line

        pair = _first_pair(body) or _first_pair(line)
        if pair and not out.get("pair"):
            out["pair"] = pair
        if not out.get("action"):
            out["action"] = _classify_live_body(body, pair)
        if not out.get("direction"):
            side = _first_text(line, r"\b(LONG|SHORT)\b")
            if side:
                out["direction"] = side
        if not out.get("direction") and out.get("pair"):
            short_side = _first_text(body, r"\b([LS])\b")
            if short_side == "L":
                out["direction"] = "LONG"
            elif short_side == "S":
                out["direction"] = "SHORT"
        if not out.get("units"):
            units = _first_text(line, r"([+-]?\d+)\s*u\b")
            if units:
                out["units"] = abs(int(units))
                if not out.get("direction"):
                    out["direction"] = "SHORT" if units.startswith("-") else "LONG" if units.startswith("+") else None
        if not out.get("price"):
            price = (
                _first_text(line, r"@\s*([-+]?\d+(?:\.\d+)?)")
                or _first_text(line, r"\bfill=([-+]?\d+(?:\.\d+)?)")
            )
            if price:
                out["price"] = price
        pl_jpy = (
            _first_float(line, r"\bP/L\s*[:=]\s*([+-]?\d+(?:\.\d+)?)\s*JPY\b")
            or _first_float(line, r"\bPL=([+-]?\d+(?:\.\d+)?)\s*JPY\b")
            or _first_float(line, r"\bPL=([+-]?\d+(?:\.\d+)?)JPY\b")
        )
        out["pl_jpy"] = pl_jpy if _is_individual_trade_result(body, out, pl_jpy) else None
        out["spread_pips"] = _first_float(line, r"\bSp=([+-]?\d+(?:\.\d+)?)pip\b")
        out["trade_id"] = _first_text(line, r"\bid=(\d+)\b") or _first_text(line, r"#(\d+)")
        out["reason"] = _first_text(line, r"\breason=([^|]+?)(?:\s+id=|\s+txn=|$)")
        return out

    def _import_jsonl_events(self, conn: sqlite3.Connection, rel_path: str, source_name: str) -> int:
        path = self.archive / rel_path
        if not path.exists():
            return 0
        count = 0
        for line_no, raw in enumerate(path.read_text(errors="replace").splitlines(), 1):
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                payload = {"raw": line}
            if not isinstance(payload, dict):
                payload = {"value": payload}
            conn.execute(
                """
                INSERT OR REPLACE INTO jsonl_events(
                    source_name, line_no, event_type, timestamp_utc, pair, direction, raw_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_name,
                    line_no,
                    payload.get("event") or payload.get("type") or payload.get("action"),
                    payload.get("timestamp_utc") or payload.get("ts"),
                    payload.get("pair"),
                    payload.get("direction"),
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                ),
            )
            count += 1
        return count

    def _write_report(
        self,
        conn: sqlite3.Connection,
        source_count: int,
        legacy_rows: dict[str, int],
        live_events: int,
        journal_events: int,
    ) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        live_summary = conn.execute(
            """
            SELECT pair, direction, COUNT(*) n, ROUND(SUM(pl_jpy), 1) net_jpy, ROUND(AVG(pl_jpy), 1) avg_jpy
            FROM live_trade_events
            WHERE pl_jpy IS NOT NULL AND pair IS NOT NULL
            GROUP BY pair, direction
            ORDER BY net_jpy ASC
            LIMIT 20
            """
        ).fetchall()
        worst = conn.execute(
            """
            SELECT timestamp_text, action, pair, direction, units, pl_jpy, reason, raw_line
            FROM live_trade_events
            WHERE pl_jpy IS NOT NULL AND pair IS NOT NULL
            ORDER BY pl_jpy ASC
            LIMIT 12
            """
        ).fetchall()
        action_summary = conn.execute(
            """
            SELECT COALESCE(NULLIF(action, ''), '(empty)') action, COUNT(*) n
            FROM live_trade_events
            GROUP BY action
            ORDER BY n DESC
            LIMIT 20
            """
        ).fetchall()
        reject_reasons = conn.execute(
            """
            SELECT reason, COUNT(*) n
            FROM live_trade_events
            WHERE action='ORDER_REJECT' AND reason IS NOT NULL
            GROUP BY reason
            ORDER BY n DESC
            LIMIT 15
            """
        ).fetchall()
        source_kinds = conn.execute(
            "SELECT kind, COUNT(*) n FROM source_files GROUP BY kind ORDER BY n DESC"
        ).fetchall()
        top_pretrade = conn.execute(
            """
            SELECT pair, direction, COUNT(*) n, ROUND(SUM(pl), 1) net_jpy, ROUND(AVG(pl), 1) avg_jpy
            FROM legacy_records
            WHERE source_table='pretrade_outcomes' AND pl IS NOT NULL
            GROUP BY pair, direction
            ORDER BY net_jpy DESC
            LIMIT 12
            """
        ).fetchall()
        lines = [
            "# Legacy Import Report",
            "",
            f"- Archive: `{self.archive}`",
            f"- History DB: `{self.db_path}`",
            f"- Imported at UTC: `{datetime.now(timezone.utc).isoformat()}`",
            "",
            "## Coverage",
            "",
            f"- Source files indexed: `{source_count}`",
            f"- Live trade log events: `{live_events}`",
            f"- Trader journal events: `{journal_events}`",
        ]
        for table, count in legacy_rows.items():
            lines.append(f"- `{table}` rows: `{count}`")
        lines.extend(["", "## Source Kinds", ""])
        for row in source_kinds:
            lines.append(f"- `{row['kind']}`: `{row['n']}`")
        lines.extend(["", "## Live Log Action Coverage", ""])
        for row in action_summary:
            lines.append(f"- `{row['action']}`: `{row['n']}`")
        lines.extend(["", "## Strongest Historical Pair/Direction Edges", ""])
        for row in top_pretrade:
            lines.append(
                f"- `{row['pair']} {row['direction']}`: n={row['n']} net={row['net_jpy']} JPY avg={row['avg_jpy']} JPY"
            )
        lines.extend(["", "## Worst Live Losses To Design Against", ""])
        for row in worst:
            lines.append(
                f"- `{row['timestamp_text']}` `{row['pair']} {row['direction']}` "
                f"{row['units']}u P/L={row['pl_jpy']} reason=`{row['reason']}`"
            )
        lines.extend(["", "## Live Log Net By Pair/Direction", ""])
        for row in live_summary:
            lines.append(
                f"- `{row['pair']} {row['direction']}`: n={row['n']} net={row['net_jpy']} JPY avg={row['avg_jpy']} JPY"
            )
        lines.extend(["", "## Rejection Reasons Feeding vNext Guards", ""])
        for row in reject_reasons:
            lines.append(f"- n={row['n']} `{row['reason']}`")
        lines.extend(
            [
                "",
                "## Mandatory vNext Implications",
                "",
                "- Broker-synced or manual/tagless exposure must block fresh entries until adopted or closed.",
                "- Any live order path must compute JPY loss before send; risk above 500 JPY is not an execution detail.",
                "- Reward/risk below 1.2x and targets/stops inside live spread friction are hard rejects.",
                "- The importer intentionally excludes secrets, Python environments, Git internals, and large replay/tick caches from the source index; the legacy archive still contains them.",
            ]
        )
        self.report_path.write_text("\n".join(lines) + "\n")


def _float_or_none(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: object) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _split_timestamp_body(line: str) -> tuple[str | None, str]:
    match = re.match(r"^\[(?P<inside>[^\]]+)\]\s*(?P<body>.*)$", line)
    if not match:
        return None, line
    inside = match.group("inside").strip()
    body = match.group("body").strip()
    if body:
        return inside, body
    compact = re.match(r"^(?P<ts>\d{2}:\d{2}Z|\d{4}-\d{2}-\d{2}T\S+|\d{4}-\d{2}-\d{2}\s+\S+(?:\s+UTC)?)\s+(?P<body>.+)$", inside)
    if compact:
        return compact.group("ts"), compact.group("body")
    return inside, ""


def _first_pair(text: str) -> str | None:
    match = re.search(r"\b([A-Z]{3})[_/]([A-Z]{3})\b", text)
    if not match:
        return None
    return f"{match.group(1)}_{match.group(2)}"


def _classify_live_body(body: str, pair: str | None) -> str:
    stripped = body.strip()
    upper = stripped.upper()
    if not stripped:
        return "NOTE"
    token = re.match(r"^([A-Z_]+)\b", upper)
    if token and token.group(1) in {
        "ENTRY",
        "ENTRY_ORDER",
        "CLOSE",
        "MODIFY",
        "CANCEL",
        "ORDER_REJECT",
        "PARTIAL_CLOSE",
        "RANGE_BOT_LIMIT",
        "TREND_BOT_MARKET",
        "LIMIT",
    }:
        return token.group(1)
    if stripped.startswith("==="):
        return "SESSION_SUMMARY" if "P/L" in upper or "LOOP END" in upper else "SESSION_MARKER"
    if "ORDER_REJECT" in upper:
        return "ORDER_REJECT"
    if "CANCELLED" in upper or "CANCEL" in upper:
        return "CANCEL_ORDER"
    if "NAV=" in upper or "BALANCE=" in upper or stripped.startswith("口座"):
        return "ACCOUNT_SNAPSHOT"
    if upper.startswith("PRICE:") or "通貨強弱" in stripped or "5分変動" in stripped:
        return "MARKET_SNAPSHOT"
    if pair and ("UPL=" in upper or "現値=" in stripped):
        return "POSITION_SNAPSHOT"
    if _looks_like_close_or_result(upper, stripped):
        return "CLOSE"
    if pair and re.search(r"\b(LONG|SHORT)\b", upper) and "@" in stripped:
        return "LEGACY_ENTRY"
    if pair and ("見送り" in stripped or "検討" in stripped or "スコア" in stripped):
        return "SIGNAL_NOTE"
    return "NOTE"


def _looks_like_close_or_result(upper: str, body: str) -> bool:
    return bool(
        " CLOSE " in f" {upper} "
        or "CLOSED" in upper
        or "SL HIT" in upper
        or "TP HIT" in upper
        or "STOP_LOSS" in upper
        or "TAKE_PROFIT" in upper
        or "クローズ" in body
        or "手動クローズ" in body
    )


def _is_individual_trade_result(body: str, parsed: dict[str, object], pl_jpy: float | None) -> bool:
    if pl_jpy is None:
        return False
    upper = body.upper()
    if "SESSION P/L" in upper or "TODAY:" in upper or "FILLS" in upper:
        return False
    if "本日" in body or "当日" in body or "勝率" in body or "統計" in body:
        return False
    if not parsed.get("pair"):
        return False
    if parsed.get("action") in {"CLOSE", "LEGACY_RESULT", "PARTIAL_CLOSE"}:
        return True
    return bool(parsed.get("trade_id") and parsed.get("direction"))


def _first_float(text: str, pattern: str) -> float | None:
    value = _first_text(text, pattern)
    return _float_or_none(value)


def _first_text(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text)
    if not match:
        return None
    return match.group(1).strip()
