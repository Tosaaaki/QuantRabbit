from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "dojo_evidence_inventory.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("dojo_evidence_inventory", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


inventory = _load_module()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", os.fspath(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _repository(root: Path) -> str:
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "DOJO Test")
    _git(root, "config", "user.email", "dojo-test@example.invalid")
    tracked = root / "evidence" / "tracked.txt"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("tracked\n", encoding="utf-8")
    _git(root, "add", "evidence/tracked.txt")
    _git(root, "commit", "-m", "test fixture")
    return _git(root, "rev-parse", "HEAD")


def _mirror_fixture(base: Path) -> tuple[Path, Path]:
    root = base / "repo"
    mirror = base / "mirror"
    root.mkdir()
    mirror.mkdir()
    _repository(root)
    source = root / "evidence" / "source"
    nested = source / "nested"
    nested.mkdir(parents=True)
    (source / "alpha.bin").write_bytes(b"alpha")
    (nested / "beta.txt").write_bytes(b"beta-data")
    (mirror / "nested").mkdir()
    (mirror / "alpha.bin").write_bytes(b"alpha")
    (mirror / "nested" / "beta.txt").write_bytes(b"beta-data")
    return root, mirror


class DojoEvidenceInventoryTest(unittest.TestCase):
    def test_manifest_is_deterministic_and_excludes_git_tracked_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            head = _repository(root)
            alpha = root / "evidence" / "alpha.bin"
            beta = root / "evidence" / "nested" / "beta.txt"
            beta.parent.mkdir()
            alpha.write_bytes(b"alpha")
            beta.write_bytes(b"beta-data")

            first = inventory.build_manifest(
                project_root=root,
                source_paths=["evidence", "evidence/nested"],
            )
            second = inventory.build_manifest(
                project_root=root,
                source_paths=["evidence/nested", "evidence"],
            )

            self.assertEqual(first, second)
            self.assertEqual(first["schema_version"], "DOJO_EVIDENCE_INVENTORY_V1")
            self.assertEqual(first["source_worktree"], {"branch": "main", "head": head})
            self.assertEqual(first["selected_paths"], ["evidence", "evidence/nested"])
            self.assertEqual(first["file_count"], 2)
            self.assertEqual(first["total_bytes"], 14)
            self.assertEqual(
                first["files"],
                [
                    {
                        "path": "evidence/alpha.bin",
                        "size": 5,
                        "sha256": hashlib.sha256(b"alpha").hexdigest(),
                    },
                    {
                        "path": "evidence/nested/beta.txt",
                        "size": 9,
                        "sha256": hashlib.sha256(b"beta-data").hexdigest(),
                    },
                ],
            )
            body = {
                key: value for key, value in first.items() if key != "manifest_sha256"
            }
            self.assertEqual(
                first["manifest_sha256"],
                hashlib.sha256(inventory._canonical_json_bytes(body)).hexdigest(),
            )

            whole_worktree = inventory.build_manifest(
                project_root=root,
                source_paths=["."],
            )
            self.assertEqual(whole_worktree["file_count"], 2)
            self.assertEqual(whole_worktree["selected_paths"], ["."])

    def test_symlink_and_out_of_scope_paths_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            _repository(root)
            external = Path(directory) / "outside.bin"
            external.write_bytes(b"outside")
            link = root / "evidence" / "outside-link"
            link.symlink_to(external)

            with self.assertRaisesRegex(
                inventory.InventoryError, "symlink is forbidden"
            ):
                inventory.build_manifest(project_root=root, source_paths=["evidence"])
            with self.assertRaisesRegex(inventory.InventoryError, "project-relative"):
                inventory.build_manifest(
                    project_root=root, source_paths=["../outside.bin"]
                )
            with self.assertRaisesRegex(inventory.InventoryError, "project-relative"):
                inventory.build_manifest(project_root=root, source_paths=[external])

    def test_atomic_replacement_during_hash_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _repository(root)
            changing = root / "evidence" / "changing.bin"
            changing.write_bytes(b"original")

            def replace_after_read(handle, expected_size):
                raw = handle.read()
                replacement = changing.with_suffix(".replacement")
                replacement.write_bytes(b"replaced")
                replacement.replace(changing)
                return hashlib.sha256(raw).hexdigest(), expected_size

            with self.assertRaisesRegex(
                inventory.InventoryError, "changed while hashing"
            ):
                inventory.build_manifest(
                    project_root=root,
                    source_paths=["evidence/changing.bin"],
                    _hash_reader=replace_after_read,
                )

    def test_synthetic_sparse_4_4gb_fixture_uses_integer_byte_totals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _repository(root)
            logical_size = 4_400_000_000
            large = root / "evidence" / "synthetic-4.4gb.bin"
            with large.open("wb") as handle:
                handle.truncate(logical_size)
            synthetic_digest = hashlib.sha256(b"synthetic sparse fixture").hexdigest()

            def synthetic_reader(_handle, expected_size):
                self.assertEqual(expected_size, logical_size)
                return synthetic_digest, expected_size

            manifest = inventory.build_manifest(
                project_root=root,
                source_paths=["evidence/synthetic-4.4gb.bin"],
                _hash_reader=synthetic_reader,
            )

            self.assertEqual(manifest["file_count"], 1)
            self.assertEqual(manifest["total_bytes"], logical_size)
            self.assertEqual(
                manifest["files"],
                [
                    {
                        "path": "evidence/synthetic-4.4gb.bin",
                        "size": logical_size,
                        "sha256": synthetic_digest,
                    }
                ],
            )

    def test_exact_mirror_receipt_is_canonical_and_bound_to_source_identity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, mirror = _mirror_fixture(Path(directory))
            head = _git(root, "rev-parse", "HEAD")

            manifest = inventory.build_manifest(
                project_root=root,
                source_paths=["evidence/source"],
                mirror_root=mirror,
            )

            self.assertEqual(
                manifest["source_worktree"], {"branch": "main", "head": head}
            )
            self.assertEqual(manifest["file_count"], 2)
            self.assertEqual(manifest["total_bytes"], 14)
            self.assertEqual(
                [row["path"] for row in manifest["files"]],
                ["evidence/source/alpha.bin", "evidence/source/nested/beta.txt"],
            )
            receipt = manifest["mirror_comparison"]
            self.assertEqual(receipt["status"], "VERIFIED_EXACT")
            self.assertEqual(receipt["source_selected_path"], "evidence/source")
            self.assertEqual(receipt["mirror_root"], os.fspath(mirror.resolve()))
            self.assertEqual(receipt["file_count"], 2)
            self.assertEqual(receipt["total_bytes"], 14)
            self.assertEqual(
                receipt["source_inventory_sha256"],
                receipt["mirror_inventory_sha256"],
            )
            receipt_body = {
                key: value for key, value in receipt.items() if key != "receipt_sha256"
            }
            self.assertEqual(
                receipt["receipt_sha256"],
                hashlib.sha256(
                    inventory._canonical_json_bytes(receipt_body)
                ).hexdigest(),
            )
            manifest_body = {
                key: value
                for key, value in manifest.items()
                if key != "manifest_sha256"
            }
            self.assertEqual(
                manifest["manifest_sha256"],
                hashlib.sha256(
                    inventory._canonical_json_bytes(manifest_body)
                ).hexdigest(),
            )

    def test_mirror_requires_one_source_directory_and_real_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, mirror = _mirror_fixture(base)
            with self.assertRaisesRegex(inventory.InventoryError, "exactly one"):
                inventory.build_manifest(
                    project_root=root,
                    source_paths=["evidence/source", "evidence/source/nested"],
                    mirror_root=mirror,
                )
            with self.assertRaisesRegex(inventory.InventoryError, "source directory"):
                inventory.build_manifest(
                    project_root=root,
                    source_paths=["evidence/source/alpha.bin"],
                    mirror_root=mirror,
                )
            mirror_link = base / "mirror-link"
            mirror_link.symlink_to(mirror, target_is_directory=True)
            with self.assertRaisesRegex(inventory.InventoryError, "root symlink"):
                inventory.build_manifest(
                    project_root=root,
                    source_paths=["evidence/source"],
                    mirror_root=mirror_link,
                )

            tracked_inside = root / "evidence" / "source" / "tracked-inside.txt"
            tracked_inside.write_text("tracked\n", encoding="utf-8")
            _git(root, "add", "evidence/source/tracked-inside.txt")
            _git(root, "commit", "-m", "track a source-scope file")
            with self.assertRaisesRegex(inventory.InventoryError, "only Git-external"):
                inventory.build_manifest(
                    project_root=root,
                    source_paths=["evidence/source"],
                    mirror_root=mirror,
                )

    def test_mirror_rejects_missing_extra_symlink_and_special_files(self) -> None:
        scenarios = ("missing", "extra", "symlink", "special")
        for scenario in scenarios:
            with self.subTest(
                scenario=scenario
            ), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                root, mirror = _mirror_fixture(base)
                expected = ""
                if scenario == "missing":
                    (mirror / "nested" / "beta.txt").unlink()
                    expected = "file set mismatch"
                elif scenario == "extra":
                    (mirror / "extra.txt").write_bytes(b"extra")
                    expected = "file set mismatch"
                elif scenario == "symlink":
                    (mirror / "link").symlink_to(mirror / "alpha.bin")
                    expected = "symlink is forbidden"
                else:
                    if not hasattr(os, "mkfifo"):
                        self.skipTest("FIFO fixtures are unavailable")
                    os.mkfifo(mirror / "fifo")
                    expected = "non-regular source is forbidden"
                with self.assertRaisesRegex(inventory.InventoryError, expected):
                    inventory.build_manifest(
                        project_root=root,
                        source_paths=["evidence/source"],
                        mirror_root=mirror,
                    )

    def test_mirror_rejects_size_hash_replacement_and_late_extra_file(self) -> None:
        for scenario in ("size", "hash", "replacement", "late-extra"):
            with self.subTest(
                scenario=scenario
            ), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                root, mirror = _mirror_fixture(base)
                kwargs = {}
                if scenario == "size":
                    (mirror / "alpha.bin").write_bytes(b"wrong-size")
                    expected = "size mismatch"
                elif scenario == "hash":
                    (mirror / "alpha.bin").write_bytes(b"ALPHA")
                    expected = "SHA-256 mismatch"
                elif scenario == "replacement":
                    target = mirror / "alpha.bin"

                    def replace_during_hash(handle, expected_size):
                        raw = handle.read()
                        replacement = target.with_suffix(".replacement")
                        replacement.write_bytes(raw)
                        replacement.replace(target)
                        return hashlib.sha256(raw).hexdigest(), expected_size

                    kwargs["_mirror_hash_reader"] = replace_during_hash
                    expected = "changed while hashing"
                else:
                    added = False

                    def add_extra_during_hash(handle, expected_size):
                        nonlocal added
                        raw = handle.read()
                        if not added:
                            (mirror / "late-extra.txt").write_bytes(b"late")
                            added = True
                        return hashlib.sha256(raw).hexdigest(), expected_size

                    kwargs["_mirror_hash_reader"] = add_extra_during_hash
                    expected = "file set changed"
                with self.assertRaisesRegex(inventory.InventoryError, expected):
                    inventory.build_manifest(
                        project_root=root,
                        source_paths=["evidence/source"],
                        mirror_root=mirror,
                        **kwargs,
                    )

    def test_cli_preserves_stdout_and_output_is_create_new_outside_scopes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, mirror = _mirror_fixture(base)
            stdout_result = subprocess.run(
                [
                    sys.executable,
                    os.fspath(SCRIPT),
                    "--project-root",
                    os.fspath(root),
                    "evidence/source",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(stdout_result.returncode, 0, stdout_result.stderr)
            stdout_manifest = json.loads(stdout_result.stdout)
            self.assertEqual(stdout_manifest["file_count"], 2)
            self.assertNotIn("mirror_comparison", stdout_manifest)

            output = base / "manifest.json"
            command = [
                sys.executable,
                os.fspath(SCRIPT),
                "--project-root",
                os.fspath(root),
                "--mirror-root",
                os.fspath(mirror),
                "--output",
                os.fspath(output),
                "evidence/source",
            ]
            create_result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(create_result.returncode, 0, create_result.stderr)
            summary = json.loads(create_result.stdout)
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "CREATED")
            self.assertEqual(summary["manifest_sha256"], saved["manifest_sha256"])
            self.assertEqual(summary["mirror_status"], "VERIFIED_EXACT")

            original = output.read_bytes()
            existing_result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(existing_result.returncode, 2)
            self.assertIn("output already exists", existing_result.stderr)
            self.assertEqual(output.read_bytes(), original)

    def test_output_rejects_source_mirror_and_symlink_resolved_scopes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, mirror = _mirror_fixture(base)
            source = root / "evidence" / "source"
            source_alias = base / "source-alias"
            source_alias.symlink_to(source, target_is_directory=True)
            targets = (
                source / "manifest.json",
                mirror / "manifest.json",
                source_alias / "manifest.json",
            )
            for target in targets:
                with self.subTest(target=target):
                    command = [
                        sys.executable,
                        os.fspath(SCRIPT),
                        "--project-root",
                        os.fspath(root),
                        "--mirror-root",
                        os.fspath(mirror),
                        "--output",
                        os.fspath(target),
                        "evidence/source",
                    ]
                    result = subprocess.run(
                        command,
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("outside", result.stderr)
                    self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
