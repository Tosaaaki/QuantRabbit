from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import quant_rabbit.legacy.importer as importer_module
from quant_rabbit.legacy.importer import LegacyImporter


class LegacyImporterTest(unittest.TestCase):
    def test_imports_structured_rows_and_live_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "legacy"
            memory = archive / "collab_trade/memory"
            logs = archive / "logs"
            memory.mkdir(parents=True)
            logs.mkdir(parents=True)
            (archive / "docs").mkdir()
            (archive / "docs/SKILL_trader.md").write_text("# Trader\nold prompt\n")

            db = sqlite3.connect(memory / "memory.db")
            db.executescript(
                """
                CREATE TABLE trades (id INTEGER, session_date TEXT, pair TEXT, direction TEXT, pl REAL, thesis TEXT);
                CREATE TABLE pretrade_outcomes (
                    id INTEGER, session_date TEXT, pair TEXT, direction TEXT, pl REAL,
                    execution_style TEXT, allocation_band TEXT, thesis TEXT
                );
                CREATE TABLE seat_outcomes (id INTEGER, session_date TEXT, pair TEXT, direction TEXT, realized_pl REAL, why TEXT);
                CREATE TABLE chunks (id INTEGER, text TEXT);
                CREATE TABLE user_calls (id INTEGER, text TEXT);
                CREATE TABLE market_events (id INTEGER, text TEXT);
                INSERT INTO trades VALUES (1, '2026-04-30', 'EUR_USD', 'LONG', 120.0, 'test');
                INSERT INTO pretrade_outcomes VALUES (2, '2026-04-30', 'EUR_USD', 'LONG', 120.0, 'MARKET', 'B+', 'test');
                INSERT INTO seat_outcomes VALUES (3, '2026-04-30', 'EUR_USD', 'LONG', 120.0, 'seat');
                INSERT INTO chunks VALUES (4, 'lesson');
                INSERT INTO user_calls VALUES (5, 'call');
                INSERT INTO market_events VALUES (6, 'event');
                """
            )
            db.commit()
            db.close()

            (logs / "live_trade_log.txt").write_text(
                "[2026-04-30 15:16:39 UTC] ENTRY EUR_USD LONG 3000u @1.17326 id=470016 TP=1.17554 SL=1.17234 Sp=0.8pip | thesis=x\n"
                "[2026-04-30 15:20:00 UTC] CLOSE EUR_USD LONG 3000u @1.17554 P/L=+1000.0000JPY Sp=0.8pip reason=TAKE_PROFIT_ORDER id=470016 txn=470020\n"
            )
            (logs / "trader_journal.jsonl").write_text(
                json.dumps({"event": "order_sent", "pair": "EUR_USD", "direction": "LONG"}) + "\n"
            )

            out_db = root / "history.db"
            report = root / "report.md"
            summary = LegacyImporter(archive, out_db, report).run()

            self.assertEqual(summary.legacy_rows["trades"], 1)
            self.assertEqual(summary.legacy_rows["pretrade_outcomes"], 1)
            self.assertEqual(summary.live_trade_events, 2)
            self.assertEqual(summary.journal_events, 1)
            self.assertTrue(report.exists())

            conn = sqlite3.connect(out_db)
            try:
                live_count = conn.execute("SELECT COUNT(*) FROM live_trade_events").fetchone()[0]
                source_count = conn.execute("SELECT COUNT(*) FROM source_files").fetchone()[0]
            finally:
                conn.close()
            self.assertEqual(live_count, 2)
            self.assertGreaterEqual(source_count, 3)

    def test_classifies_unbracketed_legacy_lines_without_fake_trade_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "legacy"
            logs = archive / "logs"
            logs.mkdir(parents=True)
            (logs / "live_trade_log.txt").write_text(
                "=== LOOP END | Session P/L: -437.2 JPY | Trades: 7 ===\n"
                "GBP_USD LONG 1500u @1.33565 | UPL=+243 JPY | TP=1.33979 SL=1.33279 | M5 H20\n"
                "[06:20Z CLOSE GBP_USD LONG 2000u @1.34154 PL=-626JPY reason=H1_flip Sp=1.3pip]\n"
                "Today: 49 fills W=19/L=30 WR=38.8% PL=-4294.82JPY (bot legacy trades included).\n"
            )

            out_db = root / "history.db"
            LegacyImporter(archive, out_db, root / "report.md").run()

            conn = sqlite3.connect(out_db)
            try:
                rows = conn.execute(
                    "SELECT action, pair, direction, units, pl_jpy FROM live_trade_events ORDER BY line_no"
                ).fetchall()
            finally:
                conn.close()

            self.assertEqual(rows[0], ("SESSION_SUMMARY", None, None, None, None))
            self.assertEqual(rows[1], ("POSITION_SNAPSHOT", "GBP_USD", "LONG", 1500, None))
            self.assertEqual(rows[2], ("CLOSE", "GBP_USD", "LONG", 2000, -626.0))
            self.assertEqual(rows[3], ("NOTE", None, None, None, None))

    def test_second_run_reuses_complete_import_without_scan_hash_or_parse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _seed_incremental_archive(root / "legacy")
            importer = LegacyImporter(paths["archive"], root / "history.db", root / "report.md")
            first = importer.run()

            with (
                mock.patch.object(
                    importer,
                    "_scan_source_tree",
                    side_effect=AssertionError("source tree was rescanned"),
                ),
                mock.patch.object(
                    importer,
                    "_sha256",
                    side_effect=AssertionError("source file was rehashed"),
                ),
                mock.patch.object(
                    importer,
                    "_import_memory_db",
                    side_effect=AssertionError("memory DB was reparsed"),
                ),
                mock.patch.object(
                    importer,
                    "_import_live_trade_log",
                    side_effect=AssertionError("live log was reparsed"),
                ),
                mock.patch.object(
                    importer,
                    "_import_jsonl_events",
                    side_effect=AssertionError("JSONL was reparsed"),
                ),
            ):
                second = importer.run()

            self.assertEqual(second, first)
            self.assertEqual(list(second.legacy_rows), list(first.legacy_rows))
            with sqlite3.connect(root / "history.db") as conn:
                status = conn.execute(
                    "SELECT value FROM import_notes WHERE key='status'"
                ).fetchone()[0]
                scan_rows = conn.execute("SELECT COUNT(*) FROM source_scan_state").fetchone()[0]
            self.assertEqual(status, "COMPLETE")
            self.assertGreater(scan_rows, first.source_files)

    def test_empty_existing_legacy_table_remains_in_reused_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _seed_incremental_archive(root / "legacy")
            with sqlite3.connect(paths["memory_db"]) as conn:
                conn.execute(
                    """
                    CREATE TABLE pretrade_outcomes (
                        id INTEGER,
                        session_date TEXT,
                        pair TEXT,
                        direction TEXT,
                        pl REAL
                    )
                    """
                )
            importer = LegacyImporter(paths["archive"], root / "history.db", root / "report.md")

            first = importer.run()
            second = importer.run()

            self.assertEqual(first.legacy_rows["pretrade_outcomes"], 0)
            self.assertEqual(second.legacy_rows["pretrade_outcomes"], 0)
            self.assertEqual(list(second.legacy_rows), list(first.legacy_rows))
            self.assertEqual(second, first)

    def test_new_nested_source_file_invalidates_complete_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _seed_incremental_archive(root / "legacy")
            importer = LegacyImporter(paths["archive"], root / "history.db", root / "report.md")
            first = importer.run()
            nested = paths["docs"] / "fresh"
            nested.mkdir()
            new_source = nested / "new_nested.md"
            new_source.write_text("new evidence\n")

            second = importer.run()

            self.assertEqual(second.source_files, first.source_files + 1)
            with sqlite3.connect(root / "history.db") as conn:
                imported = conn.execute(
                    "SELECT 1 FROM source_files WHERE rel_path='docs/fresh/new_nested.md'"
                ).fetchone()
            self.assertIsNotNone(imported)

    def test_file_created_after_walk_enumeration_rolls_back_and_retries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _seed_incremental_archive(root / "legacy")
            importer = LegacyImporter(paths["archive"], root / "history.db", root / "report.md")
            original_walk = os.walk
            late_source = paths["docs"] / "late.md"
            created = False

            def racing_walk(*args: object, **kwargs: object):
                nonlocal created
                for raw_dir, dir_names, file_names in original_walk(*args, **kwargs):
                    if Path(raw_dir) == paths["docs"] and not created:
                        # original_walk has enumerated the directory, but the
                        # importer has not yet received this yielded tuple.
                        late_source.write_text("late evidence\n")
                        created = True
                    yield raw_dir, dir_names, file_names

            with mock.patch.object(importer_module.os, "walk", side_effect=racing_walk):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "directory changed during source scan",
                ):
                    importer.run()

            summary = importer.run()

            with sqlite3.connect(root / "history.db") as conn:
                imported = conn.execute(
                    "SELECT 1 FROM source_files WHERE rel_path='docs/late.md'"
                ).fetchone()
                status = conn.execute(
                    "SELECT value FROM import_notes WHERE key='status'"
                ).fetchone()[0]
            self.assertTrue(created)
            self.assertIsNotNone(imported)
            self.assertEqual(status, "COMPLETE")
            self.assertGreaterEqual(summary.source_files, 5)

    def test_file_replaced_by_same_name_directory_after_walk_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _seed_incremental_archive(root / "legacy")
            importer = LegacyImporter(paths["archive"], root / "history.db", root / "report.md")
            original_walk = os.walk
            swapped = False

            def racing_walk(*args: object, **kwargs: object):
                nonlocal swapped
                for raw_dir, dir_names, file_names in original_walk(*args, **kwargs):
                    if Path(raw_dir) == paths["docs"] and not swapped:
                        paths["doc"].unlink()
                        paths["doc"].mkdir()
                        (paths["doc"] / "inside.md").write_text("nested evidence\n")
                        swapped = True
                    yield raw_dir, dir_names, file_names

            with mock.patch.object(importer_module.os, "walk", side_effect=racing_walk):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "directory changed during source scan",
                ):
                    importer.run()

            importer.run()

            with sqlite3.connect(root / "history.db") as conn:
                imported = conn.execute(
                    "SELECT 1 FROM source_files WHERE rel_path='docs/seed.md/inside.md'"
                ).fetchone()
                status = conn.execute(
                    "SELECT value FROM import_notes WHERE key='status'"
                ).fetchone()[0]
            self.assertTrue(swapped)
            self.assertIsNotNone(imported)
            self.assertEqual(status, "COMPLETE")

    def test_full_scan_prunes_sensitive_and_noisy_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _seed_incremental_archive(root / "legacy")
            excluded = (
                paths["archive"] / ".venv/ignored.md",
                paths["archive"] / ".git/ignored.json",
                paths["archive"] / "archive/tmp/ignored.txt",
                paths["archive"] / "logs/archive_legacy/ignored.jsonl",
            )
            for path in excluded:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("must stay excluded\n")

            summary = LegacyImporter(
                paths["archive"],
                root / "history.db",
                root / "report.md",
            ).run()

            with sqlite3.connect(root / "history.db") as conn:
                excluded_rows = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM source_files
                    WHERE rel_path LIKE '.venv/%'
                       OR rel_path LIKE '.git/%'
                       OR rel_path LIKE 'archive/tmp/%'
                       OR rel_path LIKE 'logs/archive_legacy/%'
                    """
                ).fetchone()[0]
            self.assertEqual(excluded_rows, 0)
            self.assertEqual(summary.source_files, 4)

    def test_same_inode_ctime_change_with_restored_size_and_mtime_invalidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _seed_incremental_archive(root / "legacy")
            importer = LegacyImporter(paths["archive"], root / "history.db", root / "report.md")
            importer.run()
            before_stat = paths["doc"].stat()
            with sqlite3.connect(root / "history.db") as conn:
                before_sha = conn.execute(
                    "SELECT sha256 FROM source_files WHERE rel_path='docs/seed.md'"
                ).fetchone()[0]

            paths["doc"].write_text("bravo\n")
            os.utime(
                paths["doc"],
                ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns),
            )
            changed_stat = paths["doc"].stat()
            self.assertEqual(changed_stat.st_ino, before_stat.st_ino)
            self.assertEqual(changed_stat.st_size, before_stat.st_size)
            self.assertEqual(changed_stat.st_mtime_ns, before_stat.st_mtime_ns)
            self.assertNotEqual(changed_stat.st_ctime_ns, before_stat.st_ctime_ns)

            importer.run()

            with sqlite3.connect(root / "history.db") as conn:
                after_sha = conn.execute(
                    "SELECT sha256 FROM source_files WHERE rel_path='docs/seed.md'"
                ).fetchone()[0]
            self.assertNotEqual(after_sha, before_sha)

    def test_inode_replacement_with_restored_size_and_mtime_invalidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _seed_incremental_archive(root / "legacy")
            importer = LegacyImporter(paths["archive"], root / "history.db", root / "report.md")
            importer.run()
            before_stat = paths["doc"].stat()
            replacement = root / "replacement.md"
            replacement.write_text("gamma\n")
            os.utime(
                replacement,
                ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns),
            )
            os.replace(replacement, paths["doc"])
            replaced_stat = paths["doc"].stat()
            self.assertNotEqual(replaced_stat.st_ino, before_stat.st_ino)
            self.assertEqual(replaced_stat.st_size, before_stat.st_size)
            self.assertEqual(replaced_stat.st_mtime_ns, before_stat.st_mtime_ns)

            with mock.patch.object(importer, "_sha256", wraps=importer._sha256) as sha256:
                importer.run()

            self.assertGreater(sha256.call_count, 0)

    def test_memory_db_update_invalidates_and_rebuilds_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _seed_incremental_archive(root / "legacy")
            importer = LegacyImporter(paths["archive"], root / "history.db", root / "report.md")
            first = importer.run()
            with sqlite3.connect(paths["memory_db"]) as conn:
                conn.execute(
                    "INSERT INTO trades VALUES (2, '2026-05-01', 'USD_JPY', 'SHORT', 80.0, 'new')"
                )

            second = importer.run()

            self.assertEqual(first.legacy_rows["trades"], 1)
            self.assertEqual(second.legacy_rows["trades"], 2)

    def test_memory_db_wal_append_invalidates_without_main_file_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _seed_incremental_archive(root / "legacy")
            importer = LegacyImporter(paths["archive"], root / "history.db", root / "report.md")
            writer = sqlite3.connect(paths["memory_db"])
            try:
                self.assertEqual(writer.execute("PRAGMA journal_mode=WAL").fetchone()[0], "wal")
                writer.execute("PRAGMA wal_autocheckpoint=0")
                writer.execute("UPDATE trades SET thesis='wal-seeded' WHERE id=1")
                writer.commit()
                first = importer.run()
                main_before = paths["memory_db"].stat()
                wal_path = Path(str(paths["memory_db"]) + "-wal")
                wal_before = wal_path.stat()

                writer.execute(
                    "INSERT INTO trades VALUES (2, '2026-05-01', 'USD_JPY', 'SHORT', 80.0, 'wal-new')"
                )
                writer.commit()
                main_after = paths["memory_db"].stat()
                wal_after = wal_path.stat()

                second = importer.run()
            finally:
                writer.close()

            self.assertEqual(main_after.st_size, main_before.st_size)
            self.assertEqual(main_after.st_mtime_ns, main_before.st_mtime_ns)
            self.assertNotEqual(
                (wal_after.st_size, wal_after.st_mtime_ns, wal_after.st_ctime_ns),
                (wal_before.st_size, wal_before.st_mtime_ns, wal_before.st_ctime_ns),
            )
            self.assertEqual(first.legacy_rows["trades"], 1)
            self.assertEqual(second.legacy_rows["trades"], 2)
            with sqlite3.connect(root / "history.db") as conn:
                dependency = conn.execute(
                    """
                    SELECT entry_kind
                    FROM source_scan_state
                    WHERE rel_path='collab_trade/memory/memory.db-wal'
                    """
                ).fetchone()
            self.assertEqual(dependency, ("DEPENDENCY",))

    def test_live_log_append_invalidates_and_rebuilds_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _seed_incremental_archive(root / "legacy")
            importer = LegacyImporter(paths["archive"], root / "history.db", root / "report.md")
            first = importer.run()
            paths["live_log"].write_text(
                paths["live_log"].read_text()
                + "[2026-05-01 00:01:00 UTC] CLOSE USD_JPY SHORT 1000u @158.10 P/L=+80JPY id=2\n"
            )

            second = importer.run()

            self.assertEqual(first.live_trade_events, 1)
            self.assertEqual(second.live_trade_events, 2)

    def test_format_version_change_forces_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _seed_incremental_archive(root / "legacy")
            importer = LegacyImporter(paths["archive"], root / "history.db", root / "report.md")
            importer.run()

            with (
                mock.patch.object(importer_module, "IMPORTER_FORMAT_VERSION", "test-next"),
                mock.patch.object(importer, "_sha256", wraps=importer._sha256) as sha256,
            ):
                importer.run()
                with sqlite3.connect(root / "history.db") as conn:
                    version = conn.execute(
                        "SELECT value FROM import_notes WHERE key='importer_format_version'"
                    ).fetchone()[0]

            self.assertGreater(sha256.call_count, 0)
            self.assertEqual(version, "test-next")

    def test_archive_path_change_forces_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_paths = _seed_incremental_archive(root / "legacy-a", doc_text="alpha\n")
            second_paths = _seed_incremental_archive(root / "legacy-b", doc_text="bravo\n")
            db_path = root / "history.db"
            report_path = root / "report.md"
            LegacyImporter(first_paths["archive"], db_path, report_path).run()
            with sqlite3.connect(db_path) as conn:
                first_sha = conn.execute(
                    "SELECT sha256 FROM source_files WHERE rel_path='docs/seed.md'"
                ).fetchone()[0]
            second_importer = LegacyImporter(second_paths["archive"], db_path, report_path)

            with mock.patch.object(
                second_importer,
                "_sha256",
                wraps=second_importer._sha256,
            ) as sha256:
                second_importer.run()

            with sqlite3.connect(db_path) as conn:
                second_sha = conn.execute(
                    "SELECT sha256 FROM source_files WHERE rel_path='docs/seed.md'"
                ).fetchone()[0]
                archive_realpath = conn.execute(
                    "SELECT value FROM import_notes WHERE key='archive_realpath'"
                ).fetchone()[0]
            self.assertGreater(sha256.call_count, 0)
            self.assertNotEqual(second_sha, first_sha)
            self.assertEqual(archive_realpath, str(second_paths["archive"].resolve()))

    def test_oversize_candidate_is_tracked_and_imported_after_shrink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _seed_incremental_archive(root / "legacy")
            oversized = paths["docs"] / "oversized.json"
            with oversized.open("wb") as handle:
                handle.truncate(importer_module.MAX_SOURCE_FILE_BYTES + 1)
            importer = LegacyImporter(paths["archive"], root / "history.db", root / "report.md")
            first = importer.run()
            with sqlite3.connect(root / "history.db") as conn:
                tracked = conn.execute(
                    "SELECT entry_kind FROM source_scan_state WHERE rel_path='docs/oversized.json'"
                ).fetchone()
                imported_before = conn.execute(
                    "SELECT 1 FROM source_files WHERE rel_path='docs/oversized.json'"
                ).fetchone()
            oversized.write_text("{}\n")

            second = importer.run()

            self.assertEqual(tracked, ("CANDIDATE",))
            self.assertIsNone(imported_before)
            self.assertEqual(second.source_files, first.source_files + 1)

    def test_output_row_count_mismatch_forces_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _seed_incremental_archive(root / "legacy")
            importer = LegacyImporter(paths["archive"], root / "history.db", root / "report.md")
            importer.run()
            with sqlite3.connect(root / "history.db") as conn:
                conn.execute("DELETE FROM live_trade_events")

            with mock.patch.object(
                importer,
                "_import_live_trade_log",
                wraps=importer._import_live_trade_log,
            ) as live_import:
                summary = importer.run()

            self.assertEqual(live_import.call_count, 1)
            self.assertEqual(summary.live_trade_events, 1)

    def test_failed_rebuild_preserves_old_complete_snapshot_and_retries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _seed_incremental_archive(root / "legacy")
            importer = LegacyImporter(paths["archive"], root / "history.db", root / "report.md")
            first = importer.run()
            with sqlite3.connect(root / "history.db") as conn:
                old_notes = dict(conn.execute("SELECT key, value FROM import_notes").fetchall())
                old_scan_rows = conn.execute("SELECT COUNT(*) FROM source_scan_state").fetchone()[0]
            paths["live_log"].write_text(
                paths["live_log"].read_text()
                + "[2026-05-01 00:01:00 UTC] CLOSE USD_JPY SHORT 1000u @158.10 P/L=+80JPY id=2\n"
            )

            with mock.patch.object(
                importer,
                "_import_live_trade_log",
                side_effect=RuntimeError("forced import failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "forced import failure"):
                    importer.run()

            with sqlite3.connect(root / "history.db") as conn:
                notes_after_failure = dict(conn.execute("SELECT key, value FROM import_notes").fetchall())
                scan_rows_after_failure = conn.execute(
                    "SELECT COUNT(*) FROM source_scan_state"
                ).fetchone()[0]
                live_rows_after_failure = conn.execute(
                    "SELECT COUNT(*) FROM live_trade_events"
                ).fetchone()[0]
            self.assertEqual(notes_after_failure, old_notes)
            self.assertEqual(scan_rows_after_failure, old_scan_rows)
            self.assertEqual(live_rows_after_failure, first.live_trade_events)

            retried = importer.run()

            self.assertEqual(retried.live_trade_events, first.live_trade_events + 1)

    def test_source_change_during_rebuild_rolls_back_complete_state_and_retries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _seed_incremental_archive(root / "legacy")
            importer = LegacyImporter(paths["archive"], root / "history.db", root / "report.md")
            importer.run()
            with sqlite3.connect(root / "history.db") as conn:
                old_notes = dict(conn.execute("SELECT key, value FROM import_notes").fetchall())
                old_sha = conn.execute(
                    "SELECT sha256 FROM source_files WHERE rel_path='docs/seed.md'"
                ).fetchone()[0]
            paths["doc"].write_text("bravo\n")
            original_live_import = importer._import_live_trade_log

            def mutate_source_after_scan(conn: sqlite3.Connection) -> int:
                result = original_live_import(conn)
                paths["doc"].write_text("delta\n")
                return result

            with mock.patch.object(
                importer,
                "_import_live_trade_log",
                side_effect=mutate_source_after_scan,
            ):
                with self.assertRaisesRegex(RuntimeError, "archive changed during import"):
                    importer.run()

            with sqlite3.connect(root / "history.db") as conn:
                failed_notes = dict(conn.execute("SELECT key, value FROM import_notes").fetchall())
                failed_sha = conn.execute(
                    "SELECT sha256 FROM source_files WHERE rel_path='docs/seed.md'"
                ).fetchone()[0]
            self.assertEqual(failed_notes, old_notes)
            self.assertEqual(failed_sha, old_sha)

            importer.run()

            with sqlite3.connect(root / "history.db") as conn:
                retried_sha = conn.execute(
                    "SELECT sha256 FROM source_files WHERE rel_path='docs/seed.md'"
                ).fetchone()[0]
            self.assertNotEqual(retried_sha, old_sha)


def _seed_incremental_archive(
    archive: Path,
    *,
    doc_text: str = "alpha\n",
) -> dict[str, Path]:
    memory = archive / "collab_trade/memory"
    logs = archive / "logs"
    docs = archive / "docs"
    memory.mkdir(parents=True)
    logs.mkdir(parents=True)
    docs.mkdir(parents=True)
    doc = docs / "seed.md"
    doc.write_text(doc_text)
    memory_db = memory / "memory.db"
    with sqlite3.connect(memory_db) as conn:
        conn.execute(
            """
            CREATE TABLE trades (
                id INTEGER,
                session_date TEXT,
                pair TEXT,
                direction TEXT,
                pl REAL,
                thesis TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO trades VALUES (1, '2026-04-30', 'EUR_USD', 'LONG', 120.0, 'seed')"
        )
    live_log = logs / "live_trade_log.txt"
    live_log.write_text(
        "[2026-04-30 15:20:00 UTC] CLOSE EUR_USD LONG 1000u @1.17554 P/L=+120JPY id=1\n"
    )
    journal = logs / "trader_journal.jsonl"
    journal.write_text(
        json.dumps({"event": "order_sent", "pair": "EUR_USD", "direction": "LONG"})
        + "\n"
    )
    return {
        "archive": archive,
        "memory_db": memory_db,
        "logs": logs,
        "docs": docs,
        "doc": doc,
        "live_log": live_log,
        "journal": journal,
    }


if __name__ == "__main__":
    unittest.main()
