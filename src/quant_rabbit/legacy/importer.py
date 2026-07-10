from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat as stat_module
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from quant_rabbit.paths import DEFAULT_HISTORY_DB, DEFAULT_IMPORT_REPORT, DEFAULT_LEGACY_ARCHIVE


TEXT_SUFFIXES = {".md", ".txt", ".json", ".jsonl"}
SOURCE_SUFFIXES = TEXT_SUFFIXES | {".db"}
STRUCTURED_DB = "collab_trade/memory/memory.db"
SENSITIVE_PARTS = {".git", ".venv", ".gcloud", ".gcloud_config", "__pycache__", ".pytest_cache"}
NOISY_PARTS = {"archive/tmp", "logs/archive_legacy"}
MAX_SOURCE_FILE_BYTES = 25_000_000
IMPORTER_FORMAT_VERSION = "2"
IMPORT_STATUS_COMPLETE = "COMPLETE"
SOURCE_ENTRY_DIRECTORY = "DIRECTORY"
SOURCE_ENTRY_CANDIDATE = "CANDIDATE"
SOURCE_ENTRY_DEPENDENCY = "DEPENDENCY"
SOURCE_DEPENDENCY_PATHS = {f"{STRUCTURED_DB}-wal"}
LEGACY_TABLES = (
    "trades",
    "pretrade_outcomes",
    "seat_outcomes",
    "chunks",
    "user_calls",
    "market_events",
)
JSONL_SOURCES = (
    ("logs/trader_journal.jsonl", "trader_journal"),
    ("logs/s_hunt_ledger.jsonl", "s_hunt_ledger"),
    ("logs/audit_history.jsonl", "audit_history"),
)
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


@dataclass(frozen=True)
class _SourceScanEntry:
    path: Path
    rel_path: str
    entry_kind: str
    stat_result: os.stat_result


@dataclass(frozen=True)
class _ImportedCounts:
    source_files: int
    legacy_rows: dict[str, int]
    live_trade_events: int
    jsonl_events: dict[str, int]
    source_scan_state: int

    def to_dict(self) -> dict[str, object]:
        return {
            "source_files": self.source_files,
            "legacy_rows": self.legacy_rows,
            "live_trade_events": self.live_trade_events,
            "jsonl_events": self.jsonl_events,
            "source_scan_state": self.source_scan_state,
        }


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
            conn.execute("BEGIN IMMEDIATE")
            try:
                counts = self._reusable_import_counts(conn)
                if counts is None:
                    self._clear(conn)
                    source_count = self._import_source_files(conn)
                    legacy_rows = self._import_memory_db(conn)
                    live_events = self._import_live_trade_log(conn)
                    jsonl_events: dict[str, int] = {}
                    for rel_path, source_name in JSONL_SOURCES:
                        jsonl_events[source_name] = self._import_jsonl_events(
                            conn,
                            rel_path,
                            source_name,
                        )
                    if not self._source_scan_state_matches(conn):
                        raise RuntimeError("legacy archive changed during import; retry required")
                    counts = _ImportedCounts(
                        source_files=source_count,
                        legacy_rows=legacy_rows,
                        live_trade_events=live_events,
                        jsonl_events=jsonl_events,
                        source_scan_state=int(
                            conn.execute("SELECT COUNT(*) FROM source_scan_state").fetchone()[0]
                        ),
                    )
                    if not _stored_counts_match_database(
                        counts,
                        self._database_counts(conn),
                    ):
                        raise RuntimeError("legacy import row counts are inconsistent; retry required")
                    self._write_import_completion(conn, counts)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            journal_events = counts.jsonl_events.get("trader_journal", 0)
            self._write_report(
                conn,
                counts.source_files,
                counts.legacy_rows,
                counts.live_trade_events,
                journal_events,
            )
        return ImportSummary(
            archive=self.archive,
            db_path=self.db_path,
            source_files=counts.source_files,
            legacy_rows=counts.legacy_rows,
            live_trade_events=counts.live_trade_events,
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
            CREATE TABLE IF NOT EXISTS source_scan_state (
                rel_path TEXT PRIMARY KEY,
                entry_kind TEXT NOT NULL,
                st_dev INTEGER NOT NULL,
                st_ino INTEGER NOT NULL,
                st_mode INTEGER NOT NULL,
                size_bytes INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                ctime_ns INTEGER NOT NULL
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
        for table in (
            "source_files",
            "source_scan_state",
            "legacy_records",
            "live_trade_events",
            "jsonl_events",
            "import_notes",
        ):
            conn.execute(f"DELETE FROM {table}")

    def _import_source_files(self, conn: sqlite3.Connection) -> int:
        count = 0
        for entry in self._scan_source_tree():
            stat_result = entry.stat_result
            conn.execute(
                """
                INSERT INTO source_scan_state(
                    rel_path, entry_kind, st_dev, st_ino, st_mode,
                    size_bytes, mtime_ns, ctime_ns
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.rel_path,
                    entry.entry_kind,
                    int(stat_result.st_dev),
                    int(stat_result.st_ino),
                    int(stat_result.st_mode),
                    int(stat_result.st_size),
                    int(stat_result.st_mtime_ns),
                    int(stat_result.st_ctime_ns),
                ),
            )
            if entry.entry_kind != SOURCE_ENTRY_CANDIDATE:
                continue
            if stat_result.st_size > MAX_SOURCE_FILE_BYTES:
                continue
            digest = self._sha256(entry.path)
            conn.execute(
                """
                INSERT INTO source_files(rel_path, kind, size_bytes, sha256, mtime_utc)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    entry.rel_path,
                    entry.path.suffix.lower().lstrip(".") or "file",
                    stat_result.st_size,
                    digest,
                    datetime.fromtimestamp(stat_result.st_mtime, timezone.utc).isoformat(),
                ),
            )
            count += 1
        return count

    def _scan_source_tree(self) -> Iterator[_SourceScanEntry]:
        def raise_walk_error(error: OSError) -> None:
            raise error

        for raw_dir_path, dir_names, file_names in os.walk(
            self.archive,
            topdown=True,
            onerror=raise_walk_error,
            followlinks=False,
        ):
            dir_path = Path(raw_dir_path)
            rel_dir = dir_path.relative_to(self.archive).as_posix() or "."
            # ``os.walk`` has already enumerated this directory before it
            # yields.  A source created between that enumeration and the stat
            # below would otherwise be absent from ``file_names`` while the
            # post-create directory token was persisted, producing a permanent
            # false cache hit.  Re-list after the stat so either the listing is
            # stable or this rebuild rolls back and retries.  A mutation after
            # this check is still caught by the end-of-import stat validation.
            listed_entries = frozenset(
                [(name, SOURCE_ENTRY_DIRECTORY) for name in dir_names]
                + [(name, SOURCE_ENTRY_CANDIDATE) for name in file_names]
            )
            dir_stat = dir_path.stat()
            with os.scandir(dir_path) as current_dir:
                current_entries = frozenset(
                    (
                        entry.name,
                        (
                            SOURCE_ENTRY_DIRECTORY
                            if entry.is_dir()
                            else SOURCE_ENTRY_CANDIDATE
                        ),
                    )
                    for entry in current_dir
                )
            if current_entries != listed_entries:
                raise RuntimeError(
                    f"legacy archive directory changed during source scan: {rel_dir}"
                )
            dir_names[:] = sorted(
                name
                for name in dir_names
                if not self._prune_source_directory(rel_dir, name)
            )
            if stat_module.S_ISDIR(dir_stat.st_mode):
                yield _SourceScanEntry(
                    path=dir_path,
                    rel_path=rel_dir,
                    entry_kind=SOURCE_ENTRY_DIRECTORY,
                    stat_result=dir_stat,
                )
            for name in sorted(file_names):
                path = dir_path / name
                rel_path = path.relative_to(self.archive).as_posix()
                is_dependency = rel_path in SOURCE_DEPENDENCY_PATHS
                if (
                    not is_dependency
                    and (name == "env.toml" or path.suffix.lower() not in SOURCE_SUFFIXES)
                ):
                    continue
                stat_result = path.stat()
                if not stat_module.S_ISREG(stat_result.st_mode):
                    continue
                yield _SourceScanEntry(
                    path=path,
                    rel_path=rel_path,
                    entry_kind=(
                        SOURCE_ENTRY_DEPENDENCY
                        if is_dependency
                        else SOURCE_ENTRY_CANDIDATE
                    ),
                    stat_result=stat_result,
                )

    @staticmethod
    def _prune_source_directory(parent_rel: str, name: str) -> bool:
        if name in SENSITIVE_PARTS:
            return True
        child_rel = name if parent_rel == "." else f"{parent_rel}/{name}"
        return any(
            child_rel == prefix or child_rel.startswith(prefix + "/")
            for prefix in NOISY_PARTS
        )

    def _sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _reusable_import_counts(self, conn: sqlite3.Connection) -> _ImportedCounts | None:
        notes = {
            str(row["key"]): str(row["value"])
            for row in conn.execute("SELECT key, value FROM import_notes").fetchall()
        }
        if notes.get("status") != IMPORT_STATUS_COMPLETE:
            return None
        if notes.get("importer_format_version") != IMPORTER_FORMAT_VERSION:
            return None
        if notes.get("archive_realpath") != self._archive_realpath():
            return None
        try:
            stored_payload = json.loads(notes.get("summary_counts_json") or "")
        except json.JSONDecodeError:
            return None
        stored_counts = _imported_counts_from_dict(stored_payload)
        if stored_counts is None:
            return None
        if not _stored_counts_match_database(
            stored_counts,
            self._database_counts(conn),
        ):
            return None
        if not self._source_scan_state_matches(conn):
            return None
        return stored_counts

    def _database_counts(self, conn: sqlite3.Connection) -> _ImportedCounts:
        legacy_raw = {
            str(row["source_table"]): int(row["n"])
            for row in conn.execute(
                "SELECT source_table, COUNT(*) n FROM legacy_records GROUP BY source_table"
            ).fetchall()
        }
        jsonl_raw = {
            str(row["source_name"]): int(row["n"])
            for row in conn.execute(
                "SELECT source_name, COUNT(*) n FROM jsonl_events GROUP BY source_name"
            ).fetchall()
        }
        return _ImportedCounts(
            source_files=int(conn.execute("SELECT COUNT(*) FROM source_files").fetchone()[0]),
            legacy_rows=_ordered_counts(legacy_raw, LEGACY_TABLES),
            live_trade_events=int(
                conn.execute("SELECT COUNT(*) FROM live_trade_events").fetchone()[0]
            ),
            jsonl_events=_ordered_counts(
                jsonl_raw,
                tuple(source_name for _, source_name in JSONL_SOURCES),
            ),
            source_scan_state=int(
                conn.execute("SELECT COUNT(*) FROM source_scan_state").fetchone()[0]
            ),
        )

    def _source_scan_state_matches(self, conn: sqlite3.Connection) -> bool:
        rows = conn.execute(
            """
            SELECT rel_path, entry_kind, st_dev, st_ino, st_mode,
                   size_bytes, mtime_ns, ctime_ns
            FROM source_scan_state
            """
        ).fetchall()
        if not rows:
            return False
        for row in rows:
            rel_path = str(row["rel_path"] or "")
            relative = Path(rel_path)
            if rel_path == ".":
                path = self.archive
            elif not rel_path or relative.is_absolute() or ".." in relative.parts:
                return False
            else:
                path = self.archive / relative
            try:
                current = path.stat()
            except OSError:
                return False
            entry_kind = str(row["entry_kind"] or "")
            if entry_kind == SOURCE_ENTRY_DIRECTORY:
                if not stat_module.S_ISDIR(current.st_mode):
                    return False
            elif entry_kind == SOURCE_ENTRY_CANDIDATE:
                if not stat_module.S_ISREG(current.st_mode):
                    return False
            elif entry_kind == SOURCE_ENTRY_DEPENDENCY:
                if not stat_module.S_ISREG(current.st_mode):
                    return False
            else:
                return False
            stored_token = (
                int(row["st_dev"]),
                int(row["st_ino"]),
                int(row["st_mode"]),
                int(row["size_bytes"]),
                int(row["mtime_ns"]),
                int(row["ctime_ns"]),
            )
            current_token = (
                int(current.st_dev),
                int(current.st_ino),
                int(current.st_mode),
                int(current.st_size),
                int(current.st_mtime_ns),
                int(current.st_ctime_ns),
            )
            if current_token != stored_token:
                return False
        return True

    def _write_import_completion(
        self,
        conn: sqlite3.Connection,
        counts: _ImportedCounts,
    ) -> None:
        conn.executemany(
            "INSERT INTO import_notes(key, value) VALUES (?, ?)",
            (
                ("archive", str(self.archive)),
                ("archive_realpath", self._archive_realpath()),
                ("importer_format_version", IMPORTER_FORMAT_VERSION),
                ("imported_at_utc", datetime.now(timezone.utc).isoformat()),
                (
                    "summary_counts_json",
                    json.dumps(counts.to_dict(), ensure_ascii=False, sort_keys=True),
                ),
            ),
        )
        # COMPLETE is deliberately the final write in the rebuild transaction.
        conn.execute(
            "INSERT INTO import_notes(key, value) VALUES (?, ?)",
            ("status", IMPORT_STATUS_COMPLETE),
        )

    def _archive_realpath(self) -> str:
        return str(self.archive.resolve(strict=True))

    def _import_memory_db(self, conn: sqlite3.Connection) -> dict[str, int]:
        db_path = self.archive / STRUCTURED_DB
        counts: dict[str, int] = {}
        if not db_path.exists():
            return counts
        src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        src.row_factory = sqlite3.Row
        try:
            for table in LEGACY_TABLES:
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
                "- Any live order path must compute JPY loss before send; risk above the active equity-derived cap is not an execution detail.",
                "- Reward/risk below 1.2x and targets/stops inside live spread friction are hard rejects.",
                "- The importer intentionally excludes secrets, Python environments, Git internals, and large replay/tick caches from the source index; the legacy archive still contains them.",
            ]
        )
        self.report_path.write_text("\n".join(lines) + "\n")


def _ordered_counts(
    counts: dict[str, int],
    preferred_order: tuple[str, ...],
) -> dict[str, int]:
    ordered = {key: counts[key] for key in preferred_order if key in counts}
    ordered.update({key: counts[key] for key in sorted(counts) if key not in ordered})
    return ordered


def _imported_counts_from_dict(payload: object) -> _ImportedCounts | None:
    if not isinstance(payload, dict):
        return None
    legacy_rows = _count_map(payload.get("legacy_rows"))
    jsonl_events = _count_map(payload.get("jsonl_events"))
    if legacy_rows is None or jsonl_events is None:
        return None
    if not set(legacy_rows).issubset(LEGACY_TABLES):
        return None
    expected_jsonl_sources = tuple(source_name for _, source_name in JSONL_SOURCES)
    if not set(jsonl_events).issubset(expected_jsonl_sources):
        return None
    legacy_rows = _ordered_counts(legacy_rows, LEGACY_TABLES)
    jsonl_events = _ordered_counts(
        jsonl_events,
        expected_jsonl_sources,
    )
    try:
        source_files = int(payload["source_files"])
        live_trade_events = int(payload["live_trade_events"])
        source_scan_state = int(payload["source_scan_state"])
    except (KeyError, TypeError, ValueError):
        return None
    if min(source_files, live_trade_events, source_scan_state, *legacy_rows.values(), *jsonl_events.values()) < 0:
        return None
    return _ImportedCounts(
        source_files=source_files,
        legacy_rows=legacy_rows,
        live_trade_events=live_trade_events,
        jsonl_events=jsonl_events,
        source_scan_state=source_scan_state,
    )


def _count_map(value: object) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    try:
        return {str(key): int(count) for key, count in value.items()}
    except (TypeError, ValueError):
        return None


def _stored_counts_match_database(
    stored: _ImportedCounts,
    database: _ImportedCounts,
) -> bool:
    return bool(
        stored.source_files == database.source_files
        and stored.live_trade_events == database.live_trade_events
        and stored.source_scan_state == database.source_scan_state
        and _nonzero_counts(stored.legacy_rows) == database.legacy_rows
        and _nonzero_counts(stored.jsonl_events) == database.jsonl_events
    )


def _nonzero_counts(counts: dict[str, int]) -> dict[str, int]:
    return {key: count for key, count in counts.items() if count != 0}


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
